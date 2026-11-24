"""Online processing of a running protocol: channel composites, stitched
tile scans and timelapse videos, made while the acquisition goes on.

Each product is made as soon as everything it needs has been acquired:

- composite: one file per position and timepoint, once all colours and
  z planes of that position are saved;
- stitch: one image per tile group (the tiles of one well) and timepoint,
  once all tiles of the group are saved;
- video: one per stitched group (or composited position), once the last
  timepoint is done.

Shading correction (``shading``):

- 'reference': the frames were corrected with the empty-field reference
  while they were taken; nothing more is done here.
- 'auto': Quark's algorithmic shading correction (``quark_stitch``, the
  compiled Quark code through PyO3). Each colour's illumination pattern is
  measured from the run's own images once the first timepoint is complete
  (the per-pixel median over the plain fields; no reference image), stored
  in ``shading/`` and divided out of every composite, stitch and video,
  and a corrected copy of the run is written to ``shading_corrected/`` in
  the run's save format. The raw data is never changed. Products wait for
  the measurement, so the first ones appear after the first timepoint.

The work runs on its own thread and reads the saved raw files back from
disk, so it never delays the acquisition and never changes the raw data.
Everything is written below the run folder: ``composites/``,
``stitched/``, ``previews/``, ``videos/``, ``shading/`` and
``shading_corrected/``, with ``processing.log``.

No hardware and no UI in here.
"""

from __future__ import annotations

import math
import os
import queue
import re
import threading
import time
import traceback

import numpy as np

try:
    import cv2
except Exception:                                   # pragma: no cover
    cv2 = None
try:
    import tifffile
except Exception:                                   # pragma: no cover
    tifffile = None

TRANSMITTED = ('BF', 'PC', 'DF')
# Display colours, the same as the live view.
CHANNEL_RGB = {
    'BF': (1.0, 1.0, 1.0), 'PC': (1.0, 1.0, 1.0), 'DF': (1.0, 1.0, 1.0),
    'BLUE': (0.0, 1.0, 1.0), 'GREEN': (0.0, 1.0, 0.0),
    'RED': (1.0, 0.0, 0.0),
}

# A stitched plane larger than this is made at reduced resolution (the
# raw tiles keep full resolution).
MAX_STITCH_PIXELS = 120e6
# Above this size the file is written as OME BigTIFF instead of an
# ImageJ hyperstack (classic TIFF ends at 4 GB).
MAX_IMAGEJ_BYTES = 3.8e9
PREVIEW_MAX_SIDE = 1600
VIDEO_FPS = 5
SHADING_MODES = ('off', 'auto', 'reference')
# At most this many fields per colour are read to measure the shading
# (spread evenly over the first timepoint), to bound the memory used.
MAX_SHADING_FIELDS = 64
# Quark needs at least this many plain fields to measure a pattern.
MIN_SHADING_FIELDS = 4

_TILE_SITE = re.compile(r'^(?:(R\d+)-)?T[A-Z]+\d+$')


def available() -> bool:
    return cv2 is not None and tifffile is not None


# ------------------------------------------------------------ grouping

def stitch_groups(positions: list[dict]) -> dict[tuple, list[int]]:
    """Positions that form tile scans, as {(well, region): [indices]}.

    A tile is a position whose site is a tile label (``TA1``, or
    ``R2-TA1`` for the second free region without a plate); random points
    and single images are never stitched. Groups of one are left out."""
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(positions):
        m = _TILE_SITE.match(str(p.get('site') or ''))
        if m:
            groups.setdefault((p.get('well'), m.group(1)), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def group_name(key: tuple) -> str:
    return '_'.join(str(t) for t in key if t)


# ------------------------------------------------------------ stitching

def tile_offsets(stage_xy, px_per_um, scale: float = 1.0
                 ) -> list[tuple[int, int]]:
    """Top-left pixel of every tile on the stitched canvas.

    ``px_per_um`` is the 2x2 matrix from a stage displacement (um) to the
    displacement of the tile on the canvas (px); it carries the image
    scale, the axis directions and any rotation of the camera against
    the stage. ``scale`` > 1 shrinks the canvas by that factor."""
    m = np.asarray(px_per_um, dtype=np.float64).reshape(2, 2)
    pts = (np.asarray(stage_xy, dtype=np.float64) @ m.T) / float(scale)
    pts -= pts.min(axis=0)
    return [(int(round(x)), int(round(y))) for x, y in pts]


def tile_margins(offsets: list[tuple[int, int]], tw: int, th: int
                 ) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) pixels to trim from each tile: half of
    its overlap with the neighbour on that side, so that the seam runs
    down the middle of the overlap and every output pixel is one camera
    pixel (no blending, so no ghosting where tiles are a little off)."""
    out = []
    for i, (x, y) in enumerate(offsets):
        left = top = right = bottom = 0
        for j, (u, v) in enumerate(offsets):
            if i == j:
                continue
            dx, dy = u - x, v - y
            if abs(dy) < th / 2 and 0 < abs(dx) < tw:
                half = (tw - abs(dx)) // 2
                if dx < 0:
                    left = max(left, half)
                else:
                    right = max(right, half)
            if abs(dx) < tw / 2 and 0 < abs(dy) < th:
                half = (th - abs(dy)) // 2
                if dy < 0:
                    top = max(top, half)
                else:
                    bottom = max(bottom, half)
        out.append((left, top, right, bottom))
    return out


def canvas_size(offsets, tw: int, th: int) -> tuple[int, int]:
    """(width, height) of the stitched canvas."""
    return (max(x for x, _ in offsets) + tw,
            max(y for _, y in offsets) + th)


def stitch_scale(stage_xy, px_per_um, tw: int, th: int,
                 max_pixels: float = MAX_STITCH_PIXELS) -> int:
    """Whole-number reduction needed to keep one stitched plane under
    ``max_pixels`` (1 = full resolution)."""
    w, h = canvas_size(tile_offsets(stage_xy, px_per_um), tw, th)
    return max(int(math.ceil(math.sqrt(w * h / float(max_pixels)))), 1)


def stitch_plane(tiles, offsets, margins, tw: int, th: int, dtype
                 ) -> np.ndarray:
    """Paste the tiles (any iterable of 2-D arrays of th x tw, in the
    order of ``offsets``) onto one canvas."""
    w, h = canvas_size(offsets, tw, th)
    canvas = np.zeros((h, w), dtype=dtype)
    for img, (x, y), (ml, mt, mr, mb) in zip(tiles, offsets, margins):
        canvas[y + mt:y + th - mb, x + ml:x + tw - mr] = \
            img[mt:th - mb, ml:tw - mr]
    return canvas


def shrink(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(w // factor, 1), max(h // factor, 1)),
                      interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------- display

def display_range(img: np.ndarray) -> tuple[float, float]:
    """Black and white points for display (0.5 and 99.9 percentiles)."""
    sub = img[::8, ::8] if min(img.shape[:2]) >= 64 else img
    lo, hi = np.percentile(sub, (0.5, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def render_rgb(planes: dict[str, np.ndarray],
               ranges: dict[str, tuple[float, float]],
               max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    """8-bit RGB picture of one plane per channel (height and width made
    even for the video encoder). Fluorescence is added in its colour; a
    transmitted-light channel is drawn in grey underneath, dimmed when
    fluorescence is shown on top of it."""
    first = next(iter(planes.values()))
    h, w = first.shape[:2]
    f = max(h, w) / float(max_side)
    if f > 1.0:
        w, h = int(round(w / f)), int(round(h / f))
    w, h = max(w - w % 2, 2), max(h - h % 2, 2)
    grey, colour = [], np.zeros((h, w, 3), dtype=np.float32)
    has_colour = False
    for ch, img in planes.items():
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        lo, hi = ranges[ch]
        norm = np.clip((small.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
        if ch in TRANSMITTED:
            grey.append(norm)
        else:
            has_colour = True
            for k, t in enumerate(CHANNEL_RGB.get(ch, (1.0, 1.0, 1.0))):
                if t:
                    colour[..., k] += norm * t
    if grey:
        g = np.mean(grey, axis=0) * (0.6 if has_colour else 1.0)
        colour += g[..., None]
    return (np.clip(colour, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def label_frame(rgb: np.ndarray, text: str) -> np.ndarray:
    out = np.ascontiguousarray(rgb)
    size = max(out.shape[0] / 900.0, 0.5)
    org = (int(12 * size) + 4, int(34 * size) + 4)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0),
                max(int(4 * size), 2), cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, size,
                (255, 255, 255), max(int(1.6 * size), 1), cv2.LINE_AA)
    return out


def elapsed_text(seconds: float) -> str:
    s = int(round(max(seconds, 0.0)))
    return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'


def save_png(path: str, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode('.png', rgb[..., ::-1])
    if not ok:
        raise IOError(f'could not encode {path}')
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def load_png(path: str) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)        # BGR


def video_fps(n_frames: int) -> float:
    """Frame rate that makes a short timelapse last about six seconds
    (six timepoints at the usual rate would be over in one second) and a
    long one play at no more than 10 frames a second."""
    return float(min(max(round(n_frames / 6.0), 1), 10))


def write_video(path: str, frame_paths: list[str],
                fps: float | None = None) -> int:
    """Movie from the picture files, in the order given: ``.avi`` is
    written as Motion JPEG (opens in ImageJ/Fiji and any player), anything
    else as MPEG-4. ``fps`` None = :func:`video_fps`. The movie is written
    beside ``path`` first and swapped in, so an older version that is open
    in a player is never left half written. Returns the number of frames
    written (0 = no video)."""
    ext = os.path.splitext(path)[1].lower() or '.mp4'
    codec = 'MJPG' if ext == '.avi' else 'mp4v'
    if fps is None:
        fps = video_fps(len(frame_paths))
    tmp = os.path.splitext(path)[0] + '.writing' + ext
    writer, size, n = None, None, 0
    try:
        for fp in frame_paths:
            bgr = load_png(fp)
            if bgr is None:
                continue
            if writer is None:
                size = (bgr.shape[1], bgr.shape[0])
                writer = cv2.VideoWriter(
                    tmp, cv2.VideoWriter_fourcc(*codec), float(fps), size)
                if not writer.isOpened():
                    return 0
                if codec == 'MJPG':
                    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 92)
            if (bgr.shape[1], bgr.shape[0]) != size:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            writer.write(bgr)
            n += 1
    finally:
        if writer is not None:
            writer.release()
    if n:
        os.replace(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return n


# --------------------------------------------------------------- files

def describe_shading(info: dict) -> str:
    """A few words on a measured gain map (as quark_stitch says it)."""
    if 'error' in info:
        return f"not measured ({info['error']})"
    parts = ['measured from the images (Quark)']
    if info.get('fields_used') is not None:
        of = (f" of {info['fields_read']}"
              if info.get('fields_read') is not None else '')
        parts.append(f"{info['fields_used']}{of} fields")
    if info.get('split_half_rms') is not None:
        parts.append(f"known to {100.0 * float(info['split_half_rms']):.1f} %")
    return ', '.join(parts)


def _lut(rgb) -> np.ndarray:
    ramp = np.arange(256, dtype=np.float32)
    return np.stack([(ramp * t).astype(np.uint8) for t in rgb])


def write_hyperstack(path: str, planes, n_z: int, channels: list[str],
                     height: int, width: int, dtype, *,
                     ranges: dict[str, tuple[float, float]] | None = None,
                     um_per_px: float = 0.0, z_step_um: float = 0.0,
                     info: str = '') -> None:
    """One TIFF holding ``n_z`` planes of every channel.

    ``planes`` yields the 2-D images plane by plane, channels inside z
    (z1c1, z1c2, ..., z2c1, ...), so only one plane is in memory at a
    time. Written as an ImageJ hyperstack with a colour per channel
    (opens as a composite in Fiji); files too large for classic TIFF are
    written as OME BigTIFF instead."""
    n_c = len(channels)
    dtype = np.dtype(dtype)
    shape = (n_z, n_c, height, width)
    big = n_z * n_c * height * width * dtype.itemsize > MAX_IMAGEJ_BYTES
    kwargs: dict = {}
    if um_per_px > 0:
        kwargs['resolution'] = (1.0 / um_per_px, 1.0 / um_per_px)
    if big:
        meta = {'axes': 'ZCYX', 'Channel': {'Name': list(channels)}}
        if um_per_px > 0:
            meta.update(PhysicalSizeX=um_per_px, PhysicalSizeXUnit='µm',
                        PhysicalSizeY=um_per_px, PhysicalSizeYUnit='µm')
        if z_step_um > 0:
            meta.update(PhysicalSizeZ=z_step_um, PhysicalSizeZUnit='µm')
        if info:
            meta['Description'] = info
    else:
        meta = {'axes': 'ZCYX', 'unit': 'um',
                'mode': 'composite' if n_c > 1 else 'grayscale',
                'LUTs': [_lut(CHANNEL_RGB.get(ch, (1.0, 1.0, 1.0)))
                         for ch in channels],
                'Labels': [f'{ch} Z{z + 1:03d}' for z in range(n_z)
                           for ch in channels]}
        if z_step_um > 0:
            meta['spacing'] = float(z_step_um)
        if ranges:
            meta['Ranges'] = tuple(float(v) for ch in channels
                                   for v in ranges.get(ch, (0.0, 255.0)))
        if info:
            meta['Info'] = info
    with tifffile.TiffWriter(path, bigtiff=big, imagej=not big,
                             ome=big) as tw:
        tw.write((np.ascontiguousarray(p, dtype=dtype) for p in planes),
                 shape=shape, dtype=dtype, metadata=meta,
                 photometric='minisblack', **kwargs)


def plane_path(paths: list[str], z: int) -> str:
    """File of z plane ``z`` (0-based) of a channel. A channel that was
    not stacked has one plane, which stands in on every z."""
    return paths[min(z, len(paths) - 1)]


# ----------------------------------------------------------- processor

class OnlineProcessor:
    """Collects what the acquisition reports as finished and makes the
    composites, stitches and videos on a background thread.

    ``positions`` are the run's positions (dicts with ``well``, ``site``,
    ``x_um``, ``y_um``); they are referred to by index afterwards.
    ``px_per_um`` is the stage-to-canvas matrix of :func:`tile_offsets`.

    ``shading='auto'`` measures and divides out the shading (see the module
    notes). ``flat_field`` (tiles -> (gain, info)) and ``apply_gain``
    ((tile, gain) -> tile) default to Quark's; ``corrected_saver`` (a
    ``well_file.RunSaver``) receives the corrected copy of every frame.
    """

    def __init__(self, base_dir: str, positions: list[dict],
                 channels: list[str], *, compose: bool, stitch: bool,
                 px_per_um, um_per_px: float, z_step_um: float = 0.0,
                 cycles: int = 1, status_cb=None, done_cb=None,
                 max_stitch_pixels: float = MAX_STITCH_PIXELS,
                 shading: str = 'off', flat_field=None, apply_gain=None,
                 corrected_saver=None):
        if shading not in SHADING_MODES:
            raise ValueError(f'shading must be one of {SHADING_MODES}')
        self.base_dir = base_dir
        self.positions = [dict(p) for p in positions]
        self.channels = list(channels)
        self.compose = bool(compose)
        self.stitch = bool(stitch)
        self.px_per_um = np.asarray(px_per_um, dtype=np.float64).reshape(2, 2)
        self.um_per_px = float(um_per_px)
        self.z_step_um = float(z_step_um)
        self.cycles = int(cycles)
        self.status_cb = status_cb
        # done_cb(summary, video_paths): called once everything is made.
        self.done_cb = done_cb
        self.videos: list[str] = []
        self.max_stitch_pixels = float(max_stitch_pixels)
        self.groups = stitch_groups(self.positions) if self.stitch else {}
        self._group_of = {i: k for k, idx in self.groups.items() for i in idx}
        self._pending: dict[tuple, dict[int, dict]] = {}
        self._ranges: dict[tuple, tuple[float, float]] = {}
        self._frames: dict[str, list[str]] = {}
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._finishing = False
        self.counts = {'composites': 0, 'stitches': 0, 'videos': 0,
                       'corrected': 0, 'errors': 0}
        self.notes: list[str] = []
        # Automatic shading: products are held until the pattern of every
        # colour is measured from the first timepoint.
        self.shading = shading
        self.auto = shading == 'auto'
        if self.auto and (flat_field is None or apply_gain is None):
            import quark_stitch
            flat_field = flat_field or quark_stitch.flat_field_from_tiles
            apply_gain = apply_gain or quark_stitch.apply_flat_field
        self._flat_field = flat_field
        self._apply_gain = apply_gain
        self.corrected_saver = corrected_saver if self.auto else None
        self.gains: dict[str, np.ndarray | None] = {}
        self.shading_text: dict[str, str] = {}
        self._samples: dict[str, list[str]] = {}
        self._first_done: set[int] = set()
        self._held: list[tuple] = []
        self._measured = not self.auto
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='online-processing')
        self._thread.start()

    # --- called by the acquisition

    def position_done(self, index: int, timepoint: int,
                      files: dict[str, list[str]],
                      elapsed_s: float = 0.0) -> None:
        """Every colour and z plane of position ``index`` is saved for
        ``timepoint`` (counted from 1). ``files`` maps each channel to
        its plane files in z order. Only call this for a position that
        was acquired completely."""
        files = {ch: list(v) for ch, v in files.items() if v}
        if not files:
            return
        if self.auto:
            self._put(('correct', index, timepoint, files))
        key = self._group_of.get(index)
        if key is not None:
            with self._lock:
                got = self._pending.setdefault((key, timepoint), {})
                got[index] = files
                ready = len(got) == len(self.groups[key])
                if ready:
                    del self._pending[(key, timepoint)]
            if ready:
                self._put(('stitch', key, timepoint, got, elapsed_s))
        elif self.compose:
            self._put(('composite', index, timepoint, files, elapsed_s))
        if self.auto:
            # Sampled after the products are queued, so the last position
            # of the first timepoint is held and released with the rest.
            self._shading_sample(index, timepoint, files)

    def _put(self, task: tuple) -> None:
        with self._lock:
            if not self._measured:
                self._held.append(task)
                return
        self._q.put(task)

    def _shading_sample(self, index: int, timepoint: int,
                        files: dict) -> None:
        """Keep the middle plane of every colour of the first timepoint
        for the shading measurement; measure once all positions are in
        (or the run is past its first timepoint)."""
        with self._lock:
            if timepoint == 1:
                for ch, paths in files.items():
                    self._samples.setdefault(ch, []).append(
                        paths[len(paths) // 2])
                self._first_done.add(index)
            complete = (timepoint > 1 or
                        len(self._first_done) >= len(self.positions))
        if complete:
            self._release()

    def _release(self) -> None:
        """Measure the shading now, then let the held products through."""
        with self._lock:
            if self._measured:
                return
            self._measured = True
            held, self._held = self._held, []
        self._q.put(('measure',))
        for task in held:
            self._q.put(task)

    def finish(self) -> None:
        """The acquisition is over (finished or stopped): make the videos
        from what is complete, then end. Does not wait."""
        with self._lock:
            left = sorted(self._pending)
            self._pending.clear()
        for key, tp in left:
            self.notes.append(f'{group_name(key) or "tiles"} t{tp:04d} not '
                              f'stitched: not all of its tiles were acquired')
        if self.auto:
            # Stopped during the first timepoint: measure from what is in.
            self._release()
        self._finishing = True
        self._q.put(('videos',))
        self._q.put(None)

    def wait(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def busy(self) -> bool:
        return self._thread.is_alive()

    # --- worker

    def _log(self, msg: str) -> None:
        try:
            with open(os.path.join(self.base_dir, 'processing.log'), 'a',
                      encoding='utf-8') as f:
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}\n')
        except Exception:
            pass

    def _status(self, msg: str) -> None:
        if self.status_cb:
            try:
                self.status_cb(msg)
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            task = self._q.get()
            if task is None:
                break
            try:
                getattr(self, '_do_' + task[0])(*task[1:])
            except Exception:
                self.counts['errors'] += 1
                self._log(f'ERROR in {task[0]} {task[1:3]}\n'
                          f'{traceback.format_exc()}')
            if self._finishing:
                self._status(f'Acquisition done; processing, '
                             f'{self._q.qsize()} tasks left...')
        if self.corrected_saver is not None:
            try:
                self.corrected_saver.close()
            except Exception:
                self.counts['errors'] += 1
                self._log(f'ERROR closing the corrected copy\n'
                          f'{traceback.format_exc()}')
        for n in self.notes:
            self._log(n)
        self._log(f'finished: {self.summary()}')
        self._status(f'Processing finished: {self.summary()} -> '
                     f'{self.base_dir}')
        if self.done_cb:
            try:
                self.done_cb(self.summary(), list(self.videos))
            except Exception:
                pass

    def summary(self) -> str:
        c = self.counts
        parts = [f'{c["composites"]} composites', f'{c["stitches"]} stitches',
                 f'{c["videos"]} videos']
        if self.auto:
            done = [ch for ch, g in self.gains.items() if g is not None]
            parts.append(f'shading measured for '
                         f'{", ".join(done) or "NO colour"}, '
                         f'{c["corrected"]} images corrected')
        if c['errors']:
            parts.append(f'{c["errors"]} ERRORS (see processing.log)')
        if self.notes:
            parts.append(f'{len(self.notes)} incomplete (see processing.log)')
        return ', '.join(parts)

    def _dir(self, *parts: str) -> str:
        path = os.path.join(self.base_dir, *parts)
        os.makedirs(path, exist_ok=True)
        return path

    def _fixed_range(self, name: str, ch: str, img: np.ndarray
                     ) -> tuple[float, float]:
        """Display range of a channel, fixed at the first timepoint so a
        timelapse does not flicker."""
        return self._ranges.setdefault((name, ch), display_range(img))

    def _preview(self, name: str, timepoint: int, mids: dict, ranges: dict,
                 elapsed_s: float) -> None:
        rgb = render_rgb(mids, ranges)
        rgb = label_frame(rgb, f'{name}  t{timepoint:04d}  '
                               f'{elapsed_text(elapsed_s)}')
        path = os.path.join(self._dir('previews', name),
                            f'{name}_t{timepoint:04d}.png')
        save_png(path, rgb)
        frames = self._frames.setdefault(name, [])
        frames.append(path)
        # Keep a movie of the timepoints so far, so there is always one to
        # look at during the run (and if the run is stopped or the PC
        # goes down). Every timepoint at first, every tenth in long runs.
        n = len(frames)
        if n >= 2 and (n <= 60 or n % 10 == 0):
            try:
                write_video(os.path.join(self._dir('videos'), f'{name}.mp4'),
                            sorted(frames))
            except Exception:
                self._log(f'video of {name} not updated at timepoint '
                          f'{timepoint} (is it open in a player?)')

    # --- shading

    def _read(self, path: str, ch: str) -> np.ndarray:
        """A saved plane, with the measured shading divided out."""
        img = tifffile.imread(path)
        gain = self.gains.get(ch)
        if gain is None or img.ndim != 2 or img.shape != gain.shape or \
                img.dtype not in (np.uint8, np.uint16):
            return img
        fixed = self._apply_gain(img.astype(np.uint16), gain)
        if img.dtype == np.uint8:
            fixed = np.clip(fixed, 0, 255).astype(np.uint8)
        return fixed

    def _shading_info(self, chans: list[str]) -> str:
        if not self.auto:
            return ''
        done = [ch for ch in chans if self.gains.get(ch) is not None]
        if not done:
            return ' No shading correction (not measured).'
        return (f' Shading measured from the run (Quark) and divided out '
                f'for {", ".join(done)}.')

    def _do_measure(self) -> None:
        """Each colour's shading from the first timepoint's fields."""
        import json
        folder = self._dir('shading')
        record = {'format': 'lvp-shading-1', 'method': 'quark-flat-field',
                  'channels': {}}
        for ch in self.channels:
            paths = self._samples.get(ch, [])
            if len(paths) > MAX_SHADING_FIELDS:
                step = len(paths) / MAX_SHADING_FIELDS
                paths = [paths[int(k * step)]
                         for k in range(MAX_SHADING_FIELDS)]
            self._status(f'Measuring the shading of {ch} from '
                         f'{len(paths)} images (Quark)...')
            gain, info = None, {}
            try:
                tiles = [tifffile.imread(p) for p in paths]
                tiles = [t for t in tiles if t.ndim == 2 and
                         t.dtype in (np.uint8, np.uint16) and
                         t.shape == tiles[0].shape]
                if len(tiles) < MIN_SHADING_FIELDS:
                    raise ValueError(
                        f'{len(tiles)} fields; at least {MIN_SHADING_FIELDS}'
                        f' are needed (more positions per colour)')
                gain, info = self._flat_field(
                    [t.astype(np.uint16) for t in tiles])
                gain = np.asarray(gain, dtype=np.float32)
                del tiles
            except Exception as e:
                gain, info = None, {'error': str(e)}
            self.gains[ch] = gain
            text = describe_shading(info)
            self.shading_text[ch] = text
            record['channels'][ch] = {'measured': text, 'info': info,
                                      'fields': len(paths)}
            if gain is not None:
                np.save(os.path.join(folder, f'gain_{ch}.npy'), gain)
            self._log(f'shading {ch}: {text}')
        with open(os.path.join(folder, 'shading.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(record, f, indent=1, default=str)
        self._status('Shading: ' + '; '.join(
            f'{ch} {t}' for ch, t in self.shading_text.items()))

    def _do_correct(self, index: int, timepoint: int, files: dict) -> None:
        """The corrected copy of one position's frames."""
        if self.corrected_saver is None:
            return
        p = self.positions[index]
        for ch in self.channels:
            for zi, path in enumerate(files.get(ch, ()), 1):
                self.corrected_saver.save(
                    self._read(path, ch), well=p.get('well'),
                    site=p.get('site') or 'P1', z_index=zi, channel=ch,
                    timepoint=timepoint, x_um=p.get('x_um'),
                    y_um=p.get('y_um'))
                self.counts['corrected'] += 1

    def _do_composite(self, index: int, timepoint: int, files: dict,
                      elapsed_s: float) -> None:
        p = self.positions[index]
        chans = [ch for ch in self.channels if ch in files]
        name = '_'.join(t for t in (p.get('well'), p.get('site') or 'P1')
                        if t)
        n_z = max(len(files[ch]) for ch in chans)
        mid = n_z // 2
        mids = {ch: self._read(plane_path(files[ch], mid), ch)
                for ch in chans}
        first = mids[chans[0]]
        ranges = {ch: self._fixed_range(name, ch, mids[ch]) for ch in chans}

        def planes():
            for z in range(n_z):
                for ch in chans:
                    yield (mids[ch] if z == mid else
                           self._read(plane_path(files[ch], z), ch))

        single = [ch for ch in chans if len(files[ch]) == 1 and n_z > 1]
        info = (f'Composite of {", ".join(chans)}; Z001 is the top plane.'
                + self._shading_info(chans)
                + (f' {", ".join(single)} had no z-stack: its one plane is '
                   f'repeated on every z.' if single else ''))
        out = os.path.join(self._dir('composites', p.get('well') or
                                     'positions'),
                           f'{name}_composite_t{timepoint:04d}.tiff')
        write_hyperstack(out, planes(), n_z, chans, first.shape[0],
                         first.shape[1], first.dtype, ranges=ranges,
                         um_per_px=self.um_per_px, z_step_um=self.z_step_um,
                         info=info)
        self.counts['composites'] += 1
        if self.cycles > 1:
            self._preview(name, timepoint, mids, ranges, elapsed_s)

    def _do_stitch(self, key: tuple, timepoint: int, got: dict,
                   elapsed_s: float) -> None:
        members = self.groups[key]
        name = (group_name(key) + '_stitched') if group_name(key) \
            else 'stitched'
        chans = [ch for ch in self.channels
                 if all(ch in got[i] for i in members)]
        if not chans:
            return
        xy = [(self.positions[i]['x_um'], self.positions[i]['y_um'])
              for i in members]
        n_z = max(len(got[i][ch]) for i in members for ch in chans)
        mid = n_z // 2
        probe = {ch: self._read(plane_path(got[members[0]][ch], mid), ch)
                 for ch in chans}
        first = probe[chans[0]]
        th, tw = first.shape[:2]
        factor = stitch_scale(xy, self.px_per_um, tw, th,
                              self.max_stitch_pixels)
        tw_s, th_s = max(tw // factor, 1), max(th // factor, 1)
        offsets = tile_offsets(xy, self.px_per_um, factor)
        margins = tile_margins(offsets, tw_s, th_s)
        width, height = canvas_size(offsets, tw_s, th_s)
        ranges = {ch: self._fixed_range(name, ch, probe[ch]) for ch in chans}
        mids: dict[str, np.ndarray] = {}
        pv = max(max(width, height) // PREVIEW_MAX_SIDE, 1)

        def one(ch, z):
            tiles = (shrink(self._read(plane_path(got[i][ch], z), ch),
                            factor) for i in members)
            plane = stitch_plane(tiles, offsets, margins, tw_s, th_s,
                                 first.dtype)
            if z == mid:
                mids[ch] = shrink(plane, pv)
            return plane

        info = (f'Stitched from {len(members)} tiles by stage position; '
                f'Z001 is the top plane.' + self._shading_info(chans)
                + (f' Reduced {factor}x to stay under '
                   f'{self.max_stitch_pixels / 1e6:.0f} megapixels; the '
                   f'tiles keep full resolution.' if factor > 1 else ''))
        folder = self._dir('stitched')
        kw = dict(ranges=ranges, um_per_px=self.um_per_px * factor,
                  z_step_um=self.z_step_um, info=info)
        if self.compose or len(chans) == 1:
            # With compositing on, all colours go into the one file.
            tag = '' if self.compose else f'_{chans[0]}'
            write_hyperstack(
                os.path.join(folder, f'{name}{tag}_t{timepoint:04d}.tiff'),
                (one(ch, z) for z in range(n_z) for ch in chans),
                n_z, chans, height, width, first.dtype, **kw)
        else:
            for ch in chans:
                write_hyperstack(
                    os.path.join(folder,
                                 f'{name}_{ch}_t{timepoint:04d}.tiff'),
                    (one(ch, z) for z in range(n_z)),
                    n_z, [ch], height, width, first.dtype, **kw)
        if factor > 1 and timepoint == 1:
            self._log(f'{name}: stitched at 1/{factor} resolution '
                      f'({width} x {height} px)')
        self.counts['stitches'] += 1
        if self.cycles > 1:
            self._preview(name, timepoint, mids, ranges, elapsed_s)

    def _do_videos(self) -> None:
        for name, frames in sorted(self._frames.items()):
            if len(frames) < 2:
                continue
            made = []
            # MP4 for viewing, AVI (Motion JPEG) for ImageJ/Fiji.
            for ext in ('.mp4', '.avi'):
                path = os.path.join(self._dir('videos'), name + ext)
                try:
                    if write_video(path, sorted(frames)):
                        made.append(path)
                    else:
                        self._log(f'ERROR: {name}{ext} could not be written')
                except Exception:
                    self._log(f'ERROR in video {name}{ext}\n'
                              f'{traceback.format_exc()}')
            if made:
                self.counts['videos'] += 1
                self.videos.extend(made)
                self._log(f'video of {name}: {len(frames)} timepoints at '
                          f'{video_fps(len(frames)):g} frames/s -> '
                          + ', '.join(os.path.basename(p) for p in made))
            if len(made) < 2:
                self.counts['errors'] += 1
