"""Acquisition engine for the custom LumaViewPro app.

Hardware orchestration and processing that sits between the stock Etaluma
drivers (pyloncamera / ledboard / motorboard) and the Kivy UI:

- per-channel settings (exposure, gain, LED power, chromatic Z offset,
  z-stack configuration)
- autofocus (contrast sweep over a user-defined top/bottom range)
- focus surface map anchored to an autofocus reference point, updated
  per cycle
- shading (flat-field) correction: capture per channel, apply live
- tile scans (grid snake or center-out spiral) with stitching
- z-stacks per channel

All motion and acquisition helpers are blocking and must be called from a
worker thread, never the UI thread. ``AcquisitionWorker`` provides that
thread with cancel support and status callbacks.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - opencv is in requirements
    cv2 = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


# Channels with an LED (official Etaluma designations / board indices).
CHANNELS = ['BF', 'PC', 'DF', 'BLUE', 'GREEN', 'RED']
LED_CHANNEL_MAP = {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'BF': 3, 'PC': 4, 'DF': 5}

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'lvp_custom_settings.json')
SHADING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'shading')


def _pixel_format_bits(pixel_format: str) -> int:
    """Significant bits for a GenICam mono pixel format name."""
    digits = ''.join(ch for ch in pixel_format if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return 8


@dataclass
class ZStackConfig:
    enabled: bool = False
    range_um: float = 50.0     # total range, centered on the focus position
    step_um: float = 5.0


@dataclass
class ChannelSettings:
    exposure_ms: float = 20.0
    gain_db: float = 0.0
    led_power: int = 50        # value sent on the wire: LED{ch}_{value} (mA)
    enabled: bool = True       # include in acquisitions (tiles / z-stacks)
    z_offset_um: float = 0.0   # chromatic aberration compensation
    zstack: ZStackConfig = field(default_factory=ZStackConfig)


def default_channel_settings() -> dict[str, ChannelSettings]:
    defaults = {
        'BF': ChannelSettings(exposure_ms=20.0, led_power=50),
        'PC': ChannelSettings(exposure_ms=30.0, led_power=50),
        'DF': ChannelSettings(exposure_ms=50.0, led_power=100),
        'BLUE': ChannelSettings(exposure_ms=100.0, led_power=100),
        'GREEN': ChannelSettings(exposure_ms=100.0, led_power=100),
        'RED': ChannelSettings(exposure_ms=100.0, led_power=100),
    }
    return defaults


def save_settings(payload: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


class ScopeHW:
    """Thread-safe facade over the stock camera / LED / stage drivers."""

    def __init__(self, camera=None, led=None, stage=None):
        self.camera = camera
        self.led = led
        self.stage = stage
        self.lock = threading.RLock()
        self.active_channel: str | None = None

    # ------------------------------------------------------------- LEDs

    def leds_off(self) -> None:
        with self.lock:
            if self.led is not None and getattr(self.led, 'driver', False):
                try:
                    self.led.exchange_command('LEDS_OFF')
                except Exception:
                    pass
            self.active_channel = None

    def led_on(self, channel: str, power: int) -> None:
        with self.lock:
            if self.led is None or not getattr(self.led, 'driver', False):
                return
            idx = LED_CHANNEL_MAP.get(channel, 3)
            try:
                self.led.exchange_command('LEDS_OFF')
                self.led.exchange_command('LEDS_ENT')
                self.led.exchange_command(f'LED{idx}_{int(power)}')
            except Exception:
                pass

    # ----------------------------------------------------------- camera

    def apply_camera_settings(self, settings: ChannelSettings) -> None:
        with self.lock:
            cam = self.camera
            if cam is None or not getattr(cam, 'active', False):
                return
            try:
                cam.exposure_t(float(settings.exposure_ms))
            except Exception:
                pass
            try:
                cam.gain(float(settings.gain_db))
            except Exception:
                pass

    def set_channel(self, channel: str | None,
                    settings: ChannelSettings | None = None) -> None:
        """Make ``channel`` live (LED + camera settings), or all off."""
        with self.lock:
            if channel is None:
                self.leds_off()
                return
            if settings is not None:
                self.apply_camera_settings(settings)
            self.led_on(channel, settings.led_power if settings else 50)
            self.active_channel = channel

    def grab_fresh(self, timeout_s: float = 3.0) -> np.ndarray | None:
        """Block for a genuinely new frame; returns a copy or None."""
        with self.lock:
            cam = self.camera
            if cam is None or not getattr(cam, 'active', False):
                return None
            try:
                ok, _ = cam.grab_new_capture(timeout_s)
            except Exception:
                return None
            if not ok:
                return None
            arr = getattr(cam, 'array', None)
            if isinstance(arr, np.ndarray) and arr.size > 0:
                return arr.copy()
            return None

    def grab_settled(self, settle_s: float = 0.0,
                     timeout_s: float = 3.0) -> np.ndarray | None:
        if settle_s > 0:
            time.sleep(settle_s)
        # First fresh frame may predate settings taking effect; use second.
        self.grab_fresh(timeout_s)
        return self.grab_fresh(timeout_s)

    def significant_bits(self) -> int:
        with self.lock:
            cam = self.camera
            if cam is None or not getattr(cam, 'active', False):
                return 8
            try:
                return _pixel_format_bits(cam.get_pixel_format())
            except Exception:
                return 8

    # ------------------------------------------------------------ stage

    def stage_ok(self) -> bool:
        return self.stage is not None and getattr(self.stage, 'driver', False)

    def get_pos(self) -> tuple[float, float, float]:
        """Current (x, y, z) in um."""
        with self.lock:
            if not self.stage_ok():
                return (0.0, 0.0, 0.0)
            try:
                return (float(self.stage.current_pos('X')),
                        float(self.stage.current_pos('Y')),
                        float(self.stage.current_pos('Z')))
            except Exception:
                return (0.0, 0.0, 0.0)

    def _wait_axis(self, axis: str, timeout_s: float = 30.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self.stage.target_status(axis):
                    return True
            except Exception:
                return False
            time.sleep(0.02)
        return False

    def move_xy(self, x_um: float, y_um: float, wait: bool = True) -> None:
        with self.lock:
            if not self.stage_ok():
                return
            try:
                self.stage.move_abs_pos('X', float(x_um))
                self.stage.move_abs_pos('Y', float(y_um))
            except Exception:
                return
        if wait:
            self._wait_axis('X')
            self._wait_axis('Y')

    def move_z(self, z_um: float, wait: bool = True) -> None:
        with self.lock:
            if not self.stage_ok():
                return
            try:
                self.stage.move_abs_pos('Z', float(z_um))
            except Exception:
                return
        if wait:
            self._wait_axis('Z')

    def move_rel(self, axis: str, um: float, wait: bool = True) -> None:
        with self.lock:
            if not self.stage_ok():
                return
            try:
                self.stage.move_rel_pos(axis, float(um))
            except Exception:
                return
        if wait:
            self._wait_axis(axis)

    def home_and_center(self, status_cb=None) -> bool:
        """Startup calibration: objective fully lowered, XY homed against
        the limit switches, then stage moved to the center position."""
        def _status(msg):
            if status_cb:
                status_cb(msg)
        if not self.stage_ok():
            _status('Stage not connected - skipping calibration')
            return False
        with self.lock:
            try:
                _status('Calibrating: lowering objective (Z home)...')
                self.stage.zhome()
                _status('Calibrating: homing XY to limit switches...')
                self.stage.xyhome()  # firmware homes Z first, then X and Y
            except Exception as e:
                _status(f'Homing failed: {e}')
                return False
        # xyhome blocks until the firmware replies 'XYZ home complete'
        with self.lock:
            try:
                _status('Calibrating: moving stage to center...')
                self.stage.xycenter()
            except Exception as e:
                _status(f'Centering failed: {e}')
                return False
        self._wait_axis('X', 60.0)
        self._wait_axis('Y', 60.0)
        _status('Stage calibrated (homed and centered)')
        return True


# ---------------------------------------------------------------- focus

def focus_metric(img: np.ndarray) -> float:
    """Contrast metric: variance of the Laplacian (higher = sharper)."""
    if img is None or img.size == 0:
        return 0.0
    small = img
    # Downsample very large frames for speed; sharpness ranking survives.
    if small.shape[0] > 1024:
        f = small.shape[0] // 1024 + 1
        small = small[::f, ::f]
    small = small.astype(np.float32)
    if cv2 is not None:
        return float(cv2.Laplacian(small, cv2.CV_32F).var())
    gy, gx = np.gradient(small)
    return float((gx ** 2 + gy ** 2).mean())


@dataclass
class AutofocusConfig:
    z_bottom_um: float = 0.0
    z_top_um: float = 200.0
    step_um: float = 10.0
    channel: str = 'BF'
    settle_s: float = 0.1


def autofocus_sweep(hw: ScopeHW, cfg: AutofocusConfig,
                    settings: ChannelSettings,
                    cancel: threading.Event | None = None,
                    status_cb=None) -> float | None:
    """Sweep Z from bottom to top at ``step_um``; return best-focus Z (um).

    Approaches from below throughout (single direction + driver backlash
    handling) and refines the peak with a parabolic fit over the metric.
    """
    lo = min(cfg.z_bottom_um, cfg.z_top_um)
    hi = max(cfg.z_bottom_um, cfg.z_top_um)
    step = max(abs(cfg.step_um), 0.1)
    hw.set_channel(cfg.channel, settings)
    zs, scores = [], []
    z = lo
    while z <= hi + 1e-6:
        if cancel is not None and cancel.is_set():
            return None
        hw.move_z(z)
        frame = hw.grab_settled(settle_s=cfg.settle_s)
        score = focus_metric(frame) if frame is not None else 0.0
        zs.append(z)
        scores.append(score)
        if status_cb:
            status_cb(f'AF: z={z:.1f}um metric={score:.1f}')
        z += step
    if not scores or max(scores) <= 0:
        return None
    i = int(np.argmax(scores))
    best = zs[i]
    # Parabolic refinement using the neighbors of the peak.
    if 0 < i < len(zs) - 1:
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            best = zs[i] + 0.5 * step * (y0 - y2) / denom
            best = min(max(best, lo), hi)
    hw.move_z(best)
    return float(best)


class FocusSurface:
    """Focus map anchored to an autofocus reference point.

    The anchor is an (x, y, channel) position where autofocus is run each
    cycle. Surface points are stored as offsets from the anchor's reference
    Z, so re-running the anchor AF shifts the whole surface by the measured
    drift - every map point updates each cycle.
    """

    def __init__(self):
        self.anchor_xy: tuple[float, float] | None = None
        self.anchor_channel: str = 'BF'
        self.ref_z: float | None = None      # anchor AF result at definition
        self.current_z: float | None = None  # anchor AF result this cycle
        # Points: list of (x_um, y_um, z_offset_um_from_ref)
        self.points: list[tuple[float, float, float]] = []

    def set_anchor(self, x_um: float, y_um: float, channel: str) -> None:
        self.anchor_xy = (float(x_um), float(y_um))
        self.anchor_channel = channel

    def set_reference_z(self, z_um: float) -> None:
        self.ref_z = float(z_um)
        self.current_z = float(z_um)

    def update_cycle(self, new_anchor_z: float) -> None:
        """Per-cycle update: record this cycle's anchor AF result."""
        if self.ref_z is None:
            self.set_reference_z(new_anchor_z)
        else:
            self.current_z = float(new_anchor_z)

    @property
    def drift_um(self) -> float:
        if self.ref_z is None or self.current_z is None:
            return 0.0
        return self.current_z - self.ref_z

    def add_point(self, x_um: float, y_um: float, z_um: float) -> None:
        """Record a surface point at its *absolute* focus z (current cycle)."""
        base = self.current_z if self.current_z is not None else 0.0
        self.points.append((float(x_um), float(y_um), float(z_um) - base))

    def clear_points(self) -> None:
        self.points = []

    def z_at(self, x_um: float, y_um: float) -> float | None:
        """Focus Z for (x, y) this cycle: plane fit over the surface points
        (>= 3 points), nearest point (1-2 points), or anchor Z alone."""
        base = self.current_z
        if base is None:
            return None
        pts = self.points
        if len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.float64)
            A = np.column_stack([arr[:, 0], arr[:, 1], np.ones(len(arr))])
            coef, *_ = np.linalg.lstsq(A, arr[:, 2], rcond=None)
            off = coef[0] * x_um + coef[1] * y_um + coef[2]
            return base + float(off)
        if pts:
            d = [math.hypot(x_um - p[0], y_um - p[1]) for p in pts]
            return base + pts[int(np.argmin(d))][2]
        return base

    def to_dict(self) -> dict:
        return {
            'anchor_xy': self.anchor_xy,
            'anchor_channel': self.anchor_channel,
            'ref_z': self.ref_z,
            'current_z': self.current_z,
            'points': self.points,
        }

    def from_dict(self, d: dict) -> None:
        self.anchor_xy = tuple(d['anchor_xy']) if d.get('anchor_xy') else None
        self.anchor_channel = d.get('anchor_channel', 'BF')
        self.ref_z = d.get('ref_z')
        self.current_z = d.get('current_z')
        self.points = [tuple(p) for p in d.get('points', [])]


# -------------------------------------------------------------- shading

class ShadingCorrector:
    """Per-channel flat-field (shading) correction.

    ``capture_reference`` averages frames of an empty/uniform field and
    smooths them; ``apply`` divides a raw frame by the normalized reference
    (gain map), preserving the frame's mean brightness. References persist
    to data/shading/<channel>.npy.
    """

    def __init__(self):
        self.references: dict[str, np.ndarray] = {}
        self.enabled = False

    def capture_reference(self, hw: ScopeHW, channel: str,
                          n_frames: int = 8) -> bool:
        frames = []
        for _ in range(n_frames):
            f = hw.grab_fresh()
            if f is not None:
                frames.append(f.astype(np.float32))
        if not frames:
            return False
        ref = np.mean(frames, axis=0)
        if cv2 is not None:
            k = max(int(min(ref.shape[:2]) * 0.05) | 1, 31)
            ref = cv2.GaussianBlur(ref, (k, k), 0)
        ref[ref <= 0] = 1.0
        gain = ref / float(ref.mean())  # normalized gain map, mean 1.0
        self.references[channel] = gain.astype(np.float32)
        self._save(channel)
        return True

    def has_reference(self, channel: str) -> bool:
        return channel in self.references

    def apply(self, img: np.ndarray, channel: str) -> np.ndarray:
        gain = self.references.get(channel)
        if gain is None or img is None:
            return img
        if gain.shape != img.shape[:2]:
            if cv2 is not None:
                gain = cv2.resize(gain, (img.shape[1], img.shape[0]))
            else:
                return img
        out = img.astype(np.float32) / gain
        if img.dtype == np.uint8:
            return np.clip(out, 0, 255).astype(np.uint8)
        return np.clip(out, 0, 65535).astype(np.uint16)

    def _save(self, channel: str) -> None:
        try:
            os.makedirs(SHADING_DIR, exist_ok=True)
            np.save(os.path.join(SHADING_DIR, f'{channel}.npy'),
                    self.references[channel])
        except Exception:
            pass

    def load_all(self) -> None:
        try:
            for ch in CHANNELS:
                p = os.path.join(SHADING_DIR, f'{ch}.npy')
                if os.path.exists(p):
                    self.references[ch] = np.load(p)
        except Exception:
            pass


# --------------------------------------------------------------- tiling

def tile_positions_grid(center_x: float, center_y: float,
                        fov_x_um: float, fov_y_um: float,
                        rows: int, cols: int,
                        overlap_pct: float = 10.0) -> list[tuple[float, float]]:
    """Snake-ordered grid of tile centers around (center_x, center_y)."""
    sx = fov_x_um * (1.0 - overlap_pct / 100.0)
    sy = fov_y_um * (1.0 - overlap_pct / 100.0)
    x0 = center_x - sx * (cols - 1) / 2.0
    y0 = center_y - sy * (rows - 1) / 2.0
    out = []
    for r in range(rows):
        rng = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in rng:
            out.append((x0 + c * sx, y0 + r * sy))
    return out


def tile_positions_spiral(center_x: float, center_y: float,
                          fov_x_um: float, fov_y_um: float,
                          n_rings: int,
                          overlap_pct: float = 10.0) -> list[tuple[float, float]]:
    """Center-out square spiral of tile centers (ring 0 = center tile)."""
    sx = fov_x_um * (1.0 - overlap_pct / 100.0)
    sy = fov_y_um * (1.0 - overlap_pct / 100.0)
    out = [(center_x, center_y)]
    for ring in range(1, n_rings + 1):
        # Top edge, left->right
        for c in range(-ring, ring + 1):
            out.append((center_x + c * sx, center_y - ring * sy))
        # Right edge, top->bottom (excl corners)
        for r in range(-ring + 1, ring):
            out.append((center_x + ring * sx, center_y + r * sy))
        # Bottom edge, right->left
        for c in range(ring, -ring - 1, -1):
            out.append((center_x + c * sx, center_y + ring * sy))
        # Left edge, bottom->top (excl corners)
        for r in range(ring - 1, -ring, -1):
            out.append((center_x - ring * sx, center_y + r * sy))
    return out


def stitch_tiles(tiles: list[np.ndarray],
                 positions: list[tuple[float, float]],
                 um_per_px: float) -> np.ndarray | None:
    """Stitch tiles by stage position (nearest-pixel placement).

    Later tiles overwrite earlier ones in overlap regions, which is robust
    for shading-corrected input.
    """
    if not tiles or um_per_px <= 0:
        return None
    th, tw = tiles[0].shape[:2]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, min_y = min(xs), min(ys)
    px = [int(round((x - min_x) / um_per_px)) for x in xs]
    py = [int(round((y - min_y) / um_per_px)) for y in ys]
    W = max(px) + tw
    H = max(py) + th
    if W * H > 60000 * 60000:
        return None
    canvas = np.zeros((H, W), dtype=tiles[0].dtype)
    for img, x, y in zip(tiles, px, py):
        canvas[y:y + th, x:x + tw] = img[:th, :tw]
    return canvas


# ---------------------------------------------------------------- worker

class AcquisitionWorker:
    """Single background worker for long-running acquisition jobs."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self.cancel = threading.Event()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, target, *args, **kwargs) -> bool:
        if self.busy:
            return False
        self.cancel.clear()
        self._thread = threading.Thread(
            target=target, args=args, kwargs=kwargs, daemon=True)
        self._thread.start()
        return True

    def request_cancel(self) -> None:
        self.cancel.set()


def timestamp_dir(base: str, prefix: str) -> str:
    path = os.path.join(base, f'{prefix}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(path, exist_ok=True)
    return path


def save_image(path: str, img: np.ndarray) -> None:
    if tifffile is not None:
        tifffile.imwrite(path, img)
    elif cv2 is not None:
        cv2.imwrite(path, img)


def run_tile_scan(hw: ScopeHW,
                  positions: list[tuple[float, float]],
                  channel_settings: dict[str, ChannelSettings],
                  surface: FocusSurface | None,
                  shading: ShadingCorrector | None,
                  um_per_px: float,
                  out_dir: str,
                  cancel: threading.Event,
                  status_cb=None,
                  settle_s: float = 0.2,
                  stitch: bool = True) -> dict[str, str]:
    """Acquire every enabled channel at each tile position, then stitch.

    Per tile: move XY, look up the focus surface Z (if defined), then for
    each enabled channel apply its chromatic z offset, LED, exposure and
    gain, grab, shading-correct, and save. Returns {channel: stitched_path}.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    enabled = [ch for ch in CHANNELS
               if ch in channel_settings and channel_settings[ch].enabled]
    if not enabled:
        _status('Tile scan: no channels enabled')
        return {}
    collected: dict[str, list[np.ndarray]] = {ch: [] for ch in enabled}
    used_positions: list[tuple[float, float]] = []

    for i, (x, y) in enumerate(positions):
        if cancel.is_set():
            _status('Tile scan canceled')
            break
        _status(f'Tile {i + 1}/{len(positions)}: moving to '
                f'({x / 1000:.2f}, {y / 1000:.2f}) mm')
        hw.move_xy(x, y)
        base_z = surface.z_at(x, y) if surface is not None else None
        used_positions.append((x, y))
        for ch in enabled:
            if cancel.is_set():
                break
            cs = channel_settings[ch]
            if base_z is not None:
                hw.move_z(base_z + cs.z_offset_um)
            hw.set_channel(ch, cs)
            frame = hw.grab_settled(settle_s=settle_s)
            if frame is None:
                _status(f'Tile {i + 1}: no frame for {ch}')
                continue
            if shading is not None and shading.enabled:
                frame = shading.apply(frame, ch)
            save_image(os.path.join(out_dir, f'tile_{i:04d}_{ch}.tiff'), frame)
            collected[ch].append(frame)
    hw.leds_off()

    stitched_paths: dict[str, str] = {}
    if stitch and not cancel.is_set():
        for ch in enabled:
            tiles = collected[ch]
            if len(tiles) == len(used_positions) and tiles:
                _status(f'Stitching {ch} ({len(tiles)} tiles)...')
                mosaic = stitch_tiles(tiles, used_positions, um_per_px)
                if mosaic is not None:
                    p = os.path.join(out_dir, f'stitched_{ch}.tiff')
                    save_image(p, mosaic)
                    stitched_paths[ch] = p
    _status(f'Tile scan finished -> {out_dir}')
    return stitched_paths


def run_z_stack(hw: ScopeHW,
                channel_settings: dict[str, ChannelSettings],
                channels: list[str],
                center_z_um: float,
                shading: ShadingCorrector | None,
                out_dir: str,
                cancel: threading.Event,
                status_cb=None,
                settle_s: float = 0.1) -> None:
    """Acquire a z-stack for each channel using its own z-stack config.

    The stack is centered on ``center_z_um`` plus the channel's chromatic
    offset; range/step come from the channel's ZStackConfig.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    for ch in channels:
        if cancel.is_set():
            break
        cs = channel_settings.get(ch)
        if cs is None:
            continue
        zc = cs.zstack
        half = abs(zc.range_um) / 2.0
        step = max(abs(zc.step_um), 0.1)
        z0 = center_z_um + cs.z_offset_um
        zs = np.arange(z0 - half, z0 + half + 1e-6, step)
        hw.set_channel(ch, cs)
        _status(f'Z-stack {ch}: {len(zs)} slices '
                f'({z0 - half:.1f} to {z0 + half:.1f} um)')
        for j, z in enumerate(zs):
            if cancel.is_set():
                break
            hw.move_z(float(z))
            frame = hw.grab_settled(settle_s=settle_s)
            if frame is None:
                continue
            if shading is not None and shading.enabled:
                frame = shading.apply(frame, ch)
            save_image(os.path.join(out_dir, f'zstack_{ch}_{j:03d}_'
                                             f'{z:.1f}um.tiff'), frame)
    hw.move_z(center_z_um)
    hw.leds_off()
    _status(f'Z-stack finished -> {out_dir}')


def channel_settings_to_dict(cs: dict[str, ChannelSettings]) -> dict:
    return {k: asdict(v) for k, v in cs.items()}


def channel_settings_from_dict(d: dict) -> dict[str, ChannelSettings]:
    out = default_channel_settings()
    for ch, v in (d or {}).items():
        if ch not in out:
            continue
        z = v.get('zstack', {})
        out[ch] = ChannelSettings(
            exposure_ms=float(v.get('exposure_ms', 20.0)),
            gain_db=float(v.get('gain_db', 0.0)),
            led_power=int(v.get('led_power', 50)),
            enabled=bool(v.get('enabled', True)),
            z_offset_um=float(v.get('z_offset_um', 0.0)),
            zstack=ZStackConfig(
                enabled=bool(z.get('enabled', False)),
                range_um=float(z.get('range_um', 50.0)),
                step_um=float(z.get('step_um', 5.0)),
            ),
        )
    return out
