"""Quark's tile stitching and shading (flat-field) correction, for LumaViewPro.

Quark is a separate program (Rust). Its stitching and shading code is used here as it is, not
re-written in Python: the compiled module ``quark_tiles`` (built from Quark's ``quark-py``
crate with PyO3) is loaded and its functions wrapped so that they take and return NumPy
arrays. One module file serves every Python from 3.9 on.

What it does that LumaViewPro's own code does not:

* **Shading without a reference image.** The illumination pattern every tile shares is
  measured from the tiles themselves (the per-pixel median over the plain ones) and divided
  out. No empty-field reference has to be captured first.
* **Stitching by the overlaps.** Tiles start at their stage positions and are moved only
  where their overlaps agree better elsewhere; overlaps are blended instead of one tile
  overwriting the other.

Where the module is looked for: ``lib/quark/quark_tiles.pyd`` in this folder (not in the
repository: it is a compiled file, copied in or built with ``tools/build_quark_tiles.ps1``),
then anywhere on ``sys.path``. Without it every function here raises ``QuarkUnavailable``
with instructions, and LumaViewPro carries on with its own stitching.

Folders it reads:

* a **tile scan** written by ``acquisition.run_tile_scan``: ``tile_NNNN_<channel>.tiff`` with
  ``tile_scan.json`` (the stage position of every tile and the pixel size);
* a **plate protocol** folder: ``experiment.json`` and one sub-folder per well holding
  ``<well>_<site>_Z<n>_<channel>_t<n>.tiff``.

Shading correction on its own (no stitching):

* ``measure_shading(folder)``: the gain map of every channel, measured from the folder's
  tiles. LumaViewPro stores it as that channel's shading reference
  (``ShadingCorrector.set_measured``), so the **Shading** switch applies it to the live
  view and to everything captured afterwards, with no empty-field reference.
* ``correct_folder(folder)``: a shading-corrected copy of every tile, written to
  ``<folder>/shading_corrected`` with the gain maps and a record of what was measured. The
  tiles themselves are never changed.

Command line (writes ``<name>_stitched.ome.tif`` per well into ``<folder>/quark``)::

    .venv\\Scripts\\python quark_stitch.py <folder> [--well D5] [--out DIR] [--reduce 2]
    .venv\\Scripts\\python quark_stitch.py <folder> --correct     (corrected tiles instead)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys

import numpy as np

TILE_SCAN_SIDECAR = 'tile_scan.json'
SHADING_OUT = 'shading_corrected'
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(_HERE, 'lib', 'quark')
_module = None
_why = ''


class QuarkUnavailable(RuntimeError):
    """The compiled Quark module is not installed on this computer."""


def _load():
    """The ``quark_tiles`` module, or None (the reason is kept for ``why_unavailable``)."""
    global _module, _why
    if _module is not None:
        return _module
    if os.path.isdir(_MODULE_DIR) and _MODULE_DIR not in sys.path:
        sys.path.insert(0, _MODULE_DIR)
    try:
        import quark_tiles
        _module = quark_tiles
        _why = ''
    except Exception as e:          # ImportError, or a DLL that will not load
        _why = (f'Quark stitching is not installed ({e}). Copy quark_tiles.pyd into '
                f'{_MODULE_DIR}, or build it with tools\\build_quark_tiles.ps1.')
    return _module


def available() -> bool:
    """Whether Quark's stitching can be used on this computer."""
    return _load() is not None


def why_unavailable() -> str:
    """Why not (empty when it is available)."""
    _load()
    return _why


def version() -> str:
    return _need().version()


def _need():
    m = _load()
    if m is None:
        raise QuarkUnavailable(_why)
    return m


# ------------------------------------------------------------------ folders

def scan(folder: str) -> dict:
    """What a tile-scan or plate folder holds: name, wells (a tile scan is one well named
    ``scan``), channels, time points, tile size, stage pitch, notes."""
    return json.loads(_need().scan(os.fspath(folder)))


def flat_field(folder: str, channel: int = 0, time: int | None = None,
               z: int | None = None, max_fields: int = 64):
    """The illumination pattern the tiles of ``channel`` share, measured from the folder's
    own tiles. Returns ``(gain, info)``: ``gain`` is float32 (rows, columns) with mean 1,
    ``stored = gain x true``; ``info`` says how many tiles were plain enough to use and how
    well the pattern is known (``split_half_rms``, 0.004 = 0.4 %)."""
    raw, w, h, info = _need().flat_field(os.fspath(folder), channel, time, z, max_fields)
    return np.frombuffer(raw, dtype='<f4').reshape(h, w).copy(), json.loads(info)


def stitch_well(folder: str, well: str = 'scan', channels: list[int] | None = None,
                time: int | None = None, z: int | None = None, reduce: int = 1,
                pixel_um: float | None = None, shading: bool = True,
                register: bool = True):
    """One well (or the tile scan) stitched as Quark opens it.

    Returns ``(mosaic, info)``: ``mosaic`` is uint16 (channels, rows, columns); ``info``
    states the pixel size used and where it came from, what the shading correction found
    for each channel, what the registration found, and any tile that was missing.

    ``channels``: indices into ``scan(folder)['channels']`` (default all). ``reduce``: each
    output pixel is the mean of ``reduce x reduce`` stored ones. ``pixel_um``: micrometres
    per stored pixel (default: the scan's own, else measured from the plate, else the size
    at which the tiles abut). ``shading`` / ``register``: switch either step off.
    """
    raw, w, h, c, info = _need().stitch_well(os.fspath(folder), well, channels, time, z,
                                             max(int(reduce), 1), pixel_um, shading, register)
    return np.frombuffer(raw, dtype='<u2').reshape(c, h, w).copy(), json.loads(info)


def export_well(folder: str, well: str = 'scan', out_dir: str | None = None,
                stem: str | None = None, **kwargs):
    """``stitch_well`` written as an OME-TIFF (channel names and pixel size in its metadata).
    Returns ``(path, info)``. ``out_dir`` defaults to ``<folder>/quark``."""
    out_dir = os.fspath(out_dir) if out_dir else os.path.join(os.fspath(folder), 'quark')
    path, info = _need().export_well(os.fspath(folder), well, out_dir, stem,
                                     kwargs.get('channels'), kwargs.get('time'),
                                     kwargs.get('z'), max(int(kwargs.get('reduce', 1)), 1),
                                     kwargs.get('pixel_um'), kwargs.get('shading', True),
                                     kwargs.get('register', True))
    return path, json.loads(info)


def export_folder(folder: str, out_dir: str | None = None, wells: list[str] | None = None,
                  status_cb=None, cancel=None, **kwargs) -> list[str]:
    """Every well of a folder (or the wells named) stitched and written, one OME-TIFF each.
    ``status_cb(text)`` is told what is being done; ``cancel`` is a ``threading.Event``."""
    listed = scan(folder)
    names = wells or [w['name'] for w in listed['wells']]
    written = []
    for i, name in enumerate(names):
        if cancel is not None and cancel.is_set():
            break
        if status_cb:
            status_cb(f'Quark stitching {name} ({i + 1}/{len(names)})...')
        path, _ = export_well(folder, name, out_dir, **kwargs)
        written.append(path)
    return written


# ------------------------------------------------------------- in memory

def flat_field_from_tiles(tiles: list[np.ndarray]):
    """The shared illumination pattern of tiles in memory (same shape, 2-D, at least four
    of them plain). Returns ``(gain, info)`` as ``flat_field`` does."""
    if not tiles:
        raise ValueError('no tiles')
    h, w = tiles[0].shape[:2]
    raw, info = _need().flat_field_from_tiles(
        [np.ascontiguousarray(t, dtype='<u2').tobytes() for t in tiles], w, h)
    return np.frombuffer(raw, dtype='<f4').reshape(h, w).copy(), json.loads(info)


def apply_flat_field(tile: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """A tile with the illumination divided out (``true = stored / gain``), uint16."""
    if tile.shape[:2] != gain.shape[:2]:
        raise ValueError(f'tile {tile.shape} and gain {gain.shape} differ in size')
    raw = _need().apply_flat_field(np.ascontiguousarray(tile, dtype='<u2').tobytes(),
                                   np.ascontiguousarray(gain, dtype='<f4').tobytes())
    return np.frombuffer(raw, dtype='<u2').reshape(tile.shape).copy()


# -------------------------------------------------------------- tile scans

# ------------------------------------------------------ shading on its own

def measure_shading(folder: str, channels: list[str] | None = None, status_cb=None) -> dict:
    """The shading of every channel of a tile-scan or plate folder, measured from its own
    tiles: ``{channel name: (gain, info)}``. ``gain`` is ``None`` where it could not be
    measured (too few plain tiles), with the reason in ``info['error']``."""
    listed = scan(folder)
    out = {}
    for index, ch in enumerate(listed['channels']):
        if channels is not None and ch not in channels:
            continue
        if status_cb:
            status_cb(f'Measuring the shading of {ch} from the images (Quark)...')
        try:
            out[ch] = flat_field(folder, index)
        except Exception as e:
            out[ch] = (None, {'error': str(e)})
    return out


def describe_shading(info: dict) -> str:
    """A few words on a measured gain map, for the status line and the stored note."""
    if 'error' in info:
        return f"not measured ({info['error']})"
    parts = ['measured from images (Quark)']
    if info.get('fields_used') is not None:
        of = f" of {info['fields_read']}" if info.get('fields_read') is not None else ''
        parts.append(f"{info['fields_used']}{of} tiles")
    if info.get('split_half_rms') is not None:
        parts.append(f"known to {100.0 * float(info['split_half_rms']):.1f} %")
    return ', '.join(parts)


def shading_was_applied(folder: str) -> bool:
    """Whether a tile scan's tiles were saved with LumaViewPro's shading switched on (what
    is measured from them is then what that reference left, not the whole pattern)."""
    try:
        with open(os.path.join(folder, TILE_SCAN_SIDECAR), encoding='utf-8') as f:
            return bool(json.load(f).get('shading_applied', False))
    except Exception:
        return False


def _tile_files(folder: str, channel: str) -> list[str]:
    """The tiles of ``channel`` under ``folder`` (a tile scan's ``tile_NNNN_<ch>.tiff``, a
    plate's ``<well>_<site>_Z<n>_<ch>_t<n>.tiff``), not what this module wrote."""
    named = re.compile(rf'(^|_){re.escape(channel)}(_t\d+)?\.tiff?$', re.IGNORECASE)
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ('quark', SHADING_OUT)]
        for name in sorted(files):
            if named.search(name) and not name.lower().startswith('stitched'):
                found.append(os.path.join(root, name))
    return found


def correct_folder(folder: str, out_dir: str | None = None, channels: list[str] | None = None,
                   status_cb=None) -> dict:
    """Write a shading-corrected copy of every tile of a tile-scan or plate folder to
    ``<folder>/shading_corrected`` (same names and sub-folders), with each channel's gain map
    (``gain_<channel>.npy``) and ``shading.json`` saying what was measured and done. The
    shading is measured from the folder's own tiles. Returns that record."""
    import tifffile
    folder = os.fspath(folder)
    out_dir = out_dir or os.path.join(folder, SHADING_OUT)
    os.makedirs(out_dir, exist_ok=True)
    record = {'format': 'lvp-shading-1', 'source': folder, 'channels': {}}
    for ch, (gain, info) in measure_shading(folder, channels, status_cb).items():
        entry = {'measured': describe_shading(info), 'info': info, 'corrected': 0, 'skipped': []}
        record['channels'][ch] = entry
        if gain is None:
            continue
        np.save(os.path.join(out_dir, f'gain_{ch}.npy'), gain)
        files = _tile_files(folder, ch)
        for k, path in enumerate(files):
            if status_cb and k % 10 == 0:
                status_cb(f'Shading correction {ch}: tile {k + 1} of {len(files)}')
            tile = tifffile.imread(path)
            if tile.ndim != 2 or tile.shape != gain.shape or tile.dtype not in (np.uint8, np.uint16):
                entry['skipped'].append(os.path.relpath(path, folder))
                continue
            fixed = apply_flat_field(tile.astype(np.uint16), gain)
            if tile.dtype == np.uint8:
                fixed = np.clip(fixed, 0, 255).astype(np.uint8)
            target = os.path.join(out_dir, os.path.relpath(path, folder))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            tifffile.imwrite(target, fixed)
            entry['corrected'] += 1
    # The records that make the corrected copy a folder of its own (stitchable as it is).
    try:
        with open(os.path.join(folder, TILE_SCAN_SIDECAR), encoding='utf-8') as f:
            scan_record = json.load(f)
        scan_record['shading_applied'] = True
        with open(os.path.join(out_dir, TILE_SCAN_SIDECAR), 'w', encoding='utf-8') as f:
            json.dump(scan_record, f, indent=1)
    except FileNotFoundError:
        pass
    if os.path.exists(os.path.join(folder, 'experiment.json')):
        shutil.copyfile(os.path.join(folder, 'experiment.json'), os.path.join(out_dir, 'experiment.json'))
    with open(os.path.join(out_dir, 'shading.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    record['out_dir'] = out_dir
    return record


def write_tile_scan_record(out_dir: str, positions, um_per_px: float,
                           channels: list[str], pattern: str = '',
                           shading_applied: bool = False) -> str:
    """The record that makes a tile-scan folder readable later (by Quark and by this module):
    each tile's stage position, the pixel size and the channels. Plain JSON; needs no Quark.
    ``positions``: ``[(tile index, x_um, y_um), ...]`` for the tiles that were captured."""
    record = {
        'format': 'lvp-tile-scan-1',
        'um_per_px': float(um_per_px),
        'channels': list(channels),
        'pattern': pattern,
        'shading_applied': bool(shading_applied),
        'positions': [{'index': int(i), 'x_um': float(x), 'y_um': float(y)}
                      for (i, x, y) in positions],
    }
    path = os.path.join(out_dir, TILE_SCAN_SIDECAR)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)
    return path


def stitch_tile_scan(out_dir: str, channel_names: list[str], status_cb=None):
    """Stitch a finished tile scan with Quark: ``{channel: mosaic}`` (uint16, 2-D) and what
    was found (``{channel: info}``). Channels Quark could not do are left out, with the
    reason in ``info[channel]['error']``."""
    listed = scan(out_dir)
    mosaics, infos = {}, {}
    for ch in channel_names:
        if ch not in listed['channels']:
            continue
        if status_cb:
            status_cb(f'Stitching {ch} with shading correction (Quark)...')
        try:
            image, info = stitch_well(out_dir, 'scan', channels=[listed['channels'].index(ch)])
            mosaics[ch] = image[0]
            infos[ch] = info
        except Exception as e:      # this channel falls back to the plain stitch
            infos[ch] = {'error': str(e)}
    return mosaics, infos


def describe(info: dict) -> str:
    """One line for the status bar from a stitch's ``info``."""
    if 'error' in info:
        return f"Quark: {info['error']}"
    parts = []
    for s in info.get('shading', []):
        if s.get('corrected'):
            parts.append(f"{s['channel']}: shading measured from {s['fields_used']} of "
                         f"{s['fields_read']} tiles (gain {s['gain_min']:.2f} to {s['gain_max']:.2f})")
        else:
            parts.append(f"{s['channel']}: shading not corrected ({s.get('why', '')})")
    reg = (info.get('plate') or {}).get('registration') or {}
    if reg:
        parts.append('positions refined from the overlaps' if reg.get('applied')
                     else 'stage positions kept')
    if info.get('pixel_um'):
        parts.append(f"{info['pixel_um']:.4f} um/px ({info.get('pixel_source', '')})")
    return '; '.join(parts)


def _main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description='Stitch a LumaViewPro tile scan or plate folder with Quark.')
    p.add_argument('folder')
    p.add_argument('--well', action='append', help='well name (repeat for several; default all)')
    p.add_argument('--out', help='output folder (default <folder>/quark)')
    p.add_argument('--reduce', type=int, default=1, help='average NxN pixels (default 1)')
    p.add_argument('--no-shading', action='store_true')
    p.add_argument('--no-register', action='store_true')
    p.add_argument('--correct', action='store_true',
                   help='write shading-corrected tiles to <folder>/shading_corrected instead of stitching')
    a = p.parse_args(argv)
    if not available():
        print(why_unavailable())
        return 2
    if a.correct:
        record = correct_folder(a.folder, a.out, status_cb=print)
        for ch, entry in record['channels'].items():
            print(f"{ch}: {entry['measured']}; {entry['corrected']} tiles corrected, {len(entry['skipped'])} skipped")
        print(record['out_dir'])
        return 0
    for path in export_folder(a.folder, a.out, a.well, status_cb=print, reduce=a.reduce,
                              shading=not a.no_shading, register=not a.no_register):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
