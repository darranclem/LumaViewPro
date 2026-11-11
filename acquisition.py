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
    enabled: bool = False      # include in protocols - must be actively set
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


SLIDER_STEPS = 1000.0   # slider position range 0..SLIDER_STEPS


def log_slider_value(pos: float, lo: float, hi: float,
                     zero_at_start: bool = False) -> float:
    """Slider position (0..SLIDER_STEPS) -> value on a log scale lo..hi.

    Exposure and LED current act multiplicatively on brightness, so equal
    slider travel should give equal brightness *ratios*: on a linear
    1-1000 ms slider the whole useful brightfield range (< 5 ms) would sit
    in the first few pixels. With ``zero_at_start`` position 0 is exactly
    0 (e.g. LED off) and the log range starts one step later.
    """
    pos = min(max(float(pos), 0.0), SLIDER_STEPS)
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.000001)
    if zero_at_start:
        if pos <= 0.0:
            return 0.0
        frac = (pos - 1.0) / (SLIDER_STEPS - 1.0)
    else:
        frac = pos / SLIDER_STEPS
    return float(lo * (hi / lo) ** max(frac, 0.0))


def log_slider_pos(value: float, lo: float, hi: float,
                   zero_at_start: bool = False) -> float:
    """Inverse of ``log_slider_value``."""
    lo, hi = max(lo, 1e-9), max(hi, lo * 1.000001)
    if zero_at_start and value <= 0:
        return 0.0
    v = min(max(float(value), lo), hi)
    frac = math.log(v / lo) / math.log(hi / lo)
    if zero_at_start:
        return 1.0 + frac * (SLIDER_STEPS - 1.0)
    return frac * SLIDER_STEPS


def saturation_level(significant_bits: int) -> int:
    return (1 << max(int(significant_bits), 1)) - 1


def saturated_fraction(img: np.ndarray, significant_bits: int) -> float:
    """Fraction of pixels at the sensor's full-scale value (clipped)."""
    if img is None or img.size == 0:
        return 0.0
    return float(np.count_nonzero(img >= saturation_level(significant_bits))
                 / img.size)


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
                if channel != self.active_channel:
                    # Switching channel: everything else off first.
                    self.led.exchange_command('LEDS_OFF')
                    self.led.exchange_command('LEDS_ENT')
                # Same channel (e.g. a slider drag): just change the
                # current - no off/on flash between steps.
                self.led.exchange_command(f'LED{idx}_{int(round(power))}')
                self.active_channel = channel
            except Exception:
                self.active_channel = None   # unknown state: resync next time

    # ----------------------------------------------------------- camera

    def camera_limits(self) -> tuple[float, float, float, float]:
        """(exposure_min_ms, exposure_max_ms, gain_min_db, gain_max_db) the
        camera actually accepts; conservative defaults without a camera."""
        exp_lo, exp_hi, g_lo, g_hi = 0.02, 1000.0, 0.0, 24.0
        cam = self.camera
        prof = getattr(cam, 'profile', None)
        try:
            v = getattr(prof, 'exposure_min_us', None)
            if v:
                exp_lo = float(v) / 1000.0
            v = getattr(prof, 'exposure_max_us', None)
            if v:
                exp_hi = float(v) / 1000.0
        except Exception:
            pass
        try:   # the driver refuses anything above its own cap
            exp_hi = min(exp_hi, float(getattr(cam, 'max_exposure', exp_hi)))
        except Exception:
            pass
        try:
            g = getattr(prof, 'gain', None)
            if g is not None and g.total_max_db is not None:
                g_lo = float(g.total_min_db or 0.0)
                g_hi = float(g.total_max_db)
        except Exception:
            pass
        exp_lo = max(exp_lo, 0.001)
        return exp_lo, max(exp_hi, exp_lo * 2), g_lo, max(g_hi, g_lo + 1.0)

    def camera_auto_state(self) -> dict[str, str]:
        """{'GainAuto': ..., 'ExposureAuto': ...} as the camera reports
        them right now ('Off' / 'Once' / 'Continuous'); empty when they
        cannot be read."""
        out = {}
        cam = self.camera
        node_owner = getattr(cam, 'active', None)
        if not node_owner:
            return out
        for name in ('GainAuto', 'ExposureAuto'):
            try:
                out[name] = str(getattr(node_owner, name).GetValue())
            except Exception:
                pass
        return out

    def ensure_manual_exposure(self) -> bool:
        """Force the camera's auto gain / auto exposure OFF and verify it.

        While either runs, the camera overwrites the exposure and gain the
        user sets, so sliders appear not to work. The driver turns them
        off at connect but only logs if that fails; this re-asserts and
        reads back. Returns True when both read back 'Off' (or the camera
        has no such nodes)."""
        with self.lock:
            state = self.camera_auto_state()
            if all(v == 'Off' for v in state.values()):
                return True
            cam = self.camera
            try:
                cam.auto_gain(state=False)
            except Exception:
                pass
            for name in ('GainAuto', 'ExposureAuto'):
                if self.camera_auto_state().get(name, 'Off') != 'Off':
                    try:
                        getattr(cam.active, name).SetValue('Off')
                    except Exception:
                        pass
            return all(v == 'Off' for v in self.camera_auto_state().values())

    def apply_camera_settings(self, settings: ChannelSettings) -> None:
        with self.lock:
            cam = self.camera
            if cam is None or not getattr(cam, 'active', False):
                return
            self.ensure_manual_exposure()
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
            if self.led is None or not getattr(self.led, 'driver', False):
                self.active_channel = channel   # no LED board: track anyway

    def grab_fresh(self, timeout_s: float = 3.0) -> np.ndarray | None:
        """Block for a genuinely new frame; returns a copy or None.
        The camera grab happens OUTSIDE the hardware lock so position
        polling and UI reads stay responsive during sweeps."""
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
        """Vector move: both axis targets are written back-to-back so X and
        Y travel simultaneously (straight diagonal), then wait for both."""
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

    def move_xy_rel(self, dx_um: float, dy_um: float, wait: bool = True) -> None:
        """Relative vector move along (dx, dy): targets computed first, then
        written together so the path is a straight line, not X-then-Y."""
        with self.lock:
            if not self.stage_ok():
                return
            try:
                tx = float(self.stage.target_pos('X')) + float(dx_um)
                ty = float(self.stage.target_pos('Y')) + float(dy_um)
                self.stage.move_abs_pos('X', tx)
                self.stage.move_abs_pos('Y', ty)
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

FLUOR_CHANNELS = {'BLUE', 'GREEN', 'RED'}


def _prep_focus_img(img: np.ndarray) -> np.ndarray:
    """Center crop + downsample + denoise for focus scoring."""
    h, w = img.shape[:2]
    s = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if s.shape[0] > 512:
        f = s.shape[0] // 512 + 1
        s = s[::f, ::f]
    s = s.astype(np.float32)
    if cv2 is not None:
        s = cv2.GaussianBlur(s, (5, 5), 0)   # kill shot noise before gradients
    return s


def focus_metric(img: np.ndarray, mode: str = 'transmitted') -> float:
    """Sharpness score (higher = sharper).

    transmitted: Tenengrad (mean squared Sobel magnitude) - strong for
    BF/PC/DF detail. fluor: normalized variance (var/mean) - robust to the
    shot noise and intensity changes of fluorescence imaging. Both operate
    on a denoised center crop.
    """
    if img is None or img.size == 0:
        return 0.0
    s = _prep_focus_img(img)
    if cv2 is None:
        gy, gx = np.gradient(s)
        return float((gx * gx + gy * gy).mean())
    if mode == 'fluor':
        m = float(s.mean())
        return float(s.var() / (m + 1e-3))
    sx = cv2.Sobel(s, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(s, cv2.CV_32F, 0, 1)
    return float((sx * sx + sy * sy).mean())


def make_focus_template(img: np.ndarray) -> np.ndarray | None:
    """Reference descriptor of a user-approved in-focus view."""
    if img is None or img.size == 0:
        return None
    return _prep_focus_img(img)


def template_score(img: np.ndarray, tpl: np.ndarray) -> float:
    """Normalized cross-correlation (-1..1) against the stored in-focus
    reference - peaks exactly where the image looks like the marked focus."""
    s = _prep_focus_img(img)
    if cv2 is not None and s.shape != tpl.shape:
        s = cv2.resize(s, (tpl.shape[1], tpl.shape[0]))
    elif s.shape != tpl.shape:
        return 0.0
    a = s - s.mean()
    b = tpl - tpl.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a * b).sum() / denom)


@dataclass
class AutofocusConfig:
    z_bottom_um: float = 0.0
    z_top_um: float = 200.0
    step_um: float = 10.0
    channel: str = 'BF'
    settle_s: float = 0.05


def _af_measure(hw, z, settle_s, pos_cb, scorer):
    hw.move_z(z)
    if pos_cb:
        pos_cb(z)
    if settle_s > 0:
        time.sleep(settle_s)
    frame = hw.grab_fresh(2.0)
    return scorer(frame) if frame is not None else 0.0


def _parabolic_peak(zs, scores, i, lo, hi, step):
    best = zs[i]
    if 0 < i < len(zs) - 1:
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            best = zs[i] + 0.5 * step * (y0 - y2) / denom
    return min(max(best, lo), hi)


# ---------------------------------------------------- global curve fitting

AF_METHODS = ('two_pass', 'vcurve')
AF_METHOD_LABELS = {'two_pass': 'Two-pass', 'vcurve': 'Global V-curve fit'}


@dataclass
class FocusFit:
    """Result of a global focus-curve fit."""
    z: float            # estimated best focus
    model: str          # 'parabola' (log-parabola / Gaussian) or 'vcurve'
    r2: float           # goodness of fit on the points used (0..1)
    n_points: int       # points used in the fit
    z_alt: float | None = None   # the other model's estimate, if available


def _peak_window(zs: np.ndarray, s: np.ndarray, floor_frac: float):
    """Indices of the contiguous region around the global maximum whose
    normalized score stays above ``floor_frac``. Restricting the fit to
    the main peak keeps secondary maxima (coverslip / dust reflections)
    out of the model."""
    i = int(np.argmax(s))
    a = i
    while a > 0 and s[a - 1] >= floor_frac:
        a -= 1
    b = i
    while b < len(s) - 1 and s[b + 1] >= floor_frac:
        b += 1
    return a, b, i


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return max(0.0, 1.0 - ss_res / ss_tot)


def _fit_log_parabola(zs: np.ndarray, sn: np.ndarray):
    """Weighted least-squares parabola through log(score) over the peak.

    A defocus curve is close to Gaussian, b + A*exp(-(z-z0)^2 / 2s^2), so
    after baseline removal its log is an exact parabola whose vertex is the
    focus. Weights ~ score^2 (Caruana) favour the well-measured top of the
    peak over the noisy flanks. One pass of outlier rejection drops a
    single bad frame (vibration, LED flicker). Returns (z0, r2, n) or None.
    """
    a, b, _ = _peak_window(zs, sn, 0.15)
    z, y = zs[a:b + 1], sn[a:b + 1]
    if len(z) < 4:
        return None
    zc = z.mean()   # center Z for numerical conditioning
    ly = np.log(np.maximum(y, 1e-6))

    def _fit(zk, lyk, wk):
        return np.polyfit(zk - zc, lyk, 2, w=wk)

    c = _fit(z, ly, y)
    zk, yk = z, y
    if len(z) > 4:
        # Weighted residuals; drop the single worst frame if it is a
        # clear outlier, then refit once.
        resid = np.abs(np.polyval(c, z - zc) - ly) * y
        worst = int(np.argmax(resid))
        if resid[worst] > 6.0 * (float(np.median(resid)) + 1e-9):
            keep = np.arange(len(z)) != worst
            zk, yk = z[keep], y[keep]
            c = _fit(zk, ly[keep], yk)
    if c[0] >= -1e-12:          # opens upward / flat: no peak
        return None
    z0 = zc - c[1] / (2.0 * c[0])
    # The vertex must lie within the fitted span (no wild extrapolation).
    span = zk.max() - zk.min()
    if not (zk.min() - 0.1 * span <= z0 <= zk.max() + 0.1 * span):
        return None
    # Judge the fit in linear score space - that is what was measured.
    yhat = np.exp(np.polyval(c, zk - zc))
    return float(z0), _r2(yk, yhat), int(len(zk))


def _fit_vcurve(zs: np.ndarray, sn: np.ndarray):
    """Symmetric V-curve: s = c - m*|z - z0| fitted to the flanks of the
    peak (the rounded top, > 90 %, is excluded). Insensitive to the exact
    peak shape - flat-topped, saturated or plateau peaks that a parabola
    describes poorly - because it relies on the steep, information-rich
    flanks. The shared slope makes it robust with few points per side.
    Solved exactly: for each candidate z0 on a fine grid the model is
    linear in (c, m); the z0 with least squared error wins. Returns
    (z0, r2, n) or None."""
    a, b, i = _peak_window(zs, sn, 0.10)
    idx = [k for k in range(a, b + 1) if 0.15 <= sn[k] <= 0.90]
    if not any(k < i for k in idx) or not any(k > i for k in idx):
        return None
    z, y = zs[idx], sn[idx]
    if len(z) < 5:               # 3 parameters: keep >= 2 spare dof
        return None
    left_z, right_z = zs[max(a, 0)], zs[min(b, len(zs) - 1)]
    grid = np.linspace(left_z, right_z, 2001)
    d = np.abs(z[None, :] - grid[:, None])            # G x n
    dm = d - d.mean(axis=1, keepdims=True)
    ym = y - y.mean()
    var = (dm * dm).sum(axis=1) + 1e-12
    slope = (dm * ym).sum(axis=1) / var               # = -m
    sse = ((ym[None, :] - slope[:, None] * dm) ** 2).sum(axis=1)
    g = int(np.argmin(sse))
    if slope[g] >= 0:            # not a peak
        return None
    yhat = y.mean() + slope[g] * dm[g]
    return float(grid[g]), _r2(y, yhat), int(len(z))


def fit_focus_curve(zs, scores, min_r2: float = 0.85) -> FocusFit | None:
    """Global fit of a whole focus curve (all sampled Z positions).

    Scores are normalized to 0..1 (baseline = low percentile, so a
    background offset does not bias the shape), then fitted with a
    log-parabola (Gaussian model). If that fit is poor or impossible the
    symmetric V-curve is used instead. Returns None when neither
    model describes the data well enough - the caller should then sample
    more finely.
    """
    zs = np.asarray(zs, dtype=float)
    s = np.asarray(scores, dtype=float)
    if len(zs) < 4 or not np.all(np.isfinite(s)):
        return None
    order = np.argsort(zs)
    zs, s = zs[order], s[order]
    base = float(np.percentile(s, 10))
    amp = float(s.max()) - base
    if amp <= 1e-12:
        return None
    sn = np.clip((s - base) / amp, 0.0, None)
    par = _fit_log_parabola(zs, sn)
    vc = _fit_vcurve(zs, sn)
    # The Gaussian model matches real defocus curves best (sharp metrics
    # are cusp-like with convex flanks, which bias straight-line flank
    # fits), so it is preferred; the V-curve covers peaks it cannot
    # describe, e.g. flat-topped / saturated ones.
    if par is not None and par[1] >= min_r2:
        return FocusFit(par[0], 'parabola', par[1], par[2],
                        vc[0] if vc else None)
    if vc is not None and vc[1] >= min_r2:
        return FocusFit(vc[0], 'vcurve', vc[1], vc[2],
                        par[0] if par else None)
    return None


def _describe_fit(tag: str, fit: FocusFit) -> str:
    alt = '' if fit.z_alt is None else f', other model z={fit.z_alt:.2f}'
    return (f'{tag}: {fit.model} z={fit.z:.2f} um  R2={fit.r2:.3f}  '
            f'n={fit.n_points}{alt}')


def autofocus_sweep(hw: ScopeHW, cfg: AutofocusConfig,
                    settings: ChannelSettings,
                    cancel: threading.Event | None = None,
                    status_cb=None,
                    pos_cb=None,
                    template: np.ndarray | None = None,
                    high_stability: bool = False,
                    method: str = 'two_pass') -> tuple[float | None, bool]:
    """Autofocus over [bottom, top] (z-stack style range).

    ``method``:

    - ``'two_pass'``: coarse pass, then a fine pass around the coarse
      maximum with a 3-point parabolic refinement (described below).
    - ``'vcurve'``: global curve fit. The WHOLE coarse curve is fitted
      (log-parabola / Gaussian model, V-curve flank intersection as the
      fallback model - see ``fit_focus_curve``). When the fit is good the
      stage goes straight to the fitted focus: a single pass, using every
      sample rather than the three around the maximum, so it is faster
      and less sensitive to one noisy frame. When the coarse curve is too
      sparse to fit (interval wide relative to depth of field) a fine pass
      is run around the best estimate and fitted the same way. High
      stability always runs the template fine pass, centered on the fit.

    Coarse pass at ``step_um`` using a channel-appropriate sharpness
    metric (Tenengrad for transmitted, normalized variance for
    fluorescence), then a fine pass (step/5) around the peak with a
    parabolic fit - fast and sub-step accurate.

    High stability: when ``template`` (a stored in-focus reference) is
    provided and ``high_stability`` is True, the fine pass maximizes the
    normalized cross-correlation against that reference instead - it
    returns to the focus THE USER marked, not just a generically sharp
    plane. Slightly slower, highly repeatable.

    Returns (z, ok). On an unusable contrast profile the stage parks at
    the RANGE MIDDLE and (middle, False) is returned.
    """
    lo = min(cfg.z_bottom_um, cfg.z_top_um)
    hi = max(cfg.z_bottom_um, cfg.z_top_um)
    mid = (lo + hi) / 2.0
    step = max(abs(cfg.step_um), 0.5)
    if hi - lo < 1e-6:
        hw.move_z(mid)
        return mid, False
    mode = 'fluor' if cfg.channel in FLUOR_CHANNELS else 'transmitted'
    coarse_scorer = lambda f: focus_metric(f, mode)
    if high_stability and template is not None:
        fine_scorer = lambda f: template_score(f, template)
    else:
        fine_scorer = coarse_scorer
    hw.set_channel(cfg.channel, settings)
    time.sleep(0.15)   # LED + exposure settle before the first metric

    def canceled():
        return cancel is not None and cancel.is_set()

    # ---- coarse pass (bottom -> top)
    zs, scores = [], []
    z = lo
    while z <= hi + 1e-6:
        if canceled():
            return None, False
        s = _af_measure(hw, z, cfg.settle_s, pos_cb, coarse_scorer)
        zs.append(z)
        scores.append(s)
        if status_cb:
            status_cb(f'AF coarse: z={z:.1f} um  metric={s:.2f}')
        z += step
    smax, smin = max(scores), min(scores)
    if smax <= 1e-9 or (smax - smin) / abs(smax) < 0.05:
        if status_cb:
            status_cb('AF: no usable contrast - parking at range middle')
        hw.move_z(mid)
        if pos_cb:
            pos_cb(mid)
        return mid, False
    i = int(np.argmax(scores))
    z_coarse = zs[i]
    use_template = fine_scorer is not coarse_scorer

    if method == 'vcurve':
        fit = fit_focus_curve(zs, scores)
        if fit is not None:
            if status_cb:
                status_cb(_describe_fit('AF global fit', fit))
            z_coarse = min(max(fit.z, lo), hi)
            if not use_template:
                best = z_coarse
                # Approach from below for repeatable mechanics.
                hw.move_z(max(best - 5.0, 0.0))
                hw.move_z(best)
                if pos_cb:
                    pos_cb(best)
                if status_cb:
                    status_cb(f'AF best focus (global {fit.model} fit, '
                              f'1 pass): z={best:.2f} um')
                return float(best), True
        elif status_cb:
            status_cb('AF global fit: coarse curve too sparse - refining')

    # ---- fine pass around the coarse peak
    fstep = max(step / 5.0, 0.4)
    flo = max(z_coarse - step, lo)
    fhi = min(z_coarse + step, hi)
    # Approach from below again for repeatable mechanics.
    hw.move_z(max(flo - 5.0, 0.0))
    fzs, fscores = [], []
    z = flo
    while z <= fhi + 1e-6:
        if canceled():
            return None, False
        s = _af_measure(hw, z, cfg.settle_s, pos_cb, fine_scorer)
        fzs.append(z)
        fscores.append(s)
        if status_cb:
            tag = 'AF fine (hi-stab)' if use_template else 'AF fine'
            status_cb(f'{tag}: z={z:.1f} um  score={s:.3f}')
        z += fstep
    if not fscores or max(fscores) <= -1.0:
        hw.move_z(mid)
        return mid, False
    j = int(np.argmax(fscores))
    best = None
    if method == 'vcurve':
        ffit = fit_focus_curve(fzs, fscores)
        if ffit is not None:
            best = min(max(ffit.z, lo), hi)
            if status_cb:
                status_cb(_describe_fit('AF fine fit', ffit))
    if best is None:
        best = _parabolic_peak(fzs, fscores, j, lo, hi, fstep)
    hw.move_z(best)
    if pos_cb:
        pos_cb(best)
    if status_cb:
        status_cb(f'AF best focus: z={best:.2f} um')
    return float(best), True


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

def zstack_planes(center_z_um: float, cs: ChannelSettings,
                  use_channel_z_offsets: bool = True,
                  force_stack: bool = False) -> np.ndarray:
    """The Z planes a channel is acquired at around ``center_z_um``.

    Single source for both the z-stack acquisition and the shading
    reference stack, so the shading planes are exactly the acquisition
    planes. A channel without an enabled z-stack is acquired at one plane
    (center + chromatic offset) unless ``force_stack``.
    """
    z0 = center_z_um + (cs.z_offset_um if use_channel_z_offsets else 0.0)
    zc = cs.zstack
    if not (zc.enabled or force_stack) or abs(zc.range_um) < 1e-6:
        return np.array([z0], dtype=np.float64)
    half = abs(zc.range_um) / 2.0
    step = max(abs(zc.step_um), 0.1)
    return np.arange(z0 - half, z0 + half + 1e-6, step, dtype=np.float64)


def field_structure(img: np.ndarray) -> float:
    """Fine detail relative to brightness (high-pass std / mean, on a
    downsampled copy). Near the noise floor for an empty field; clearly
    higher when cells / a plate / dust are in view."""
    if img is None or img.size == 0 or cv2 is None:
        return 0.0
    s = img.astype(np.float32)
    if s.ndim == 3:
        s = s.mean(axis=2)
    f = max(max(s.shape) // 512, 1)
    s = s[::f, ::f]
    s = cv2.GaussianBlur(s, (3, 3), 0)          # suppress pixel noise
    hp = s - cv2.GaussianBlur(s, (0, 0), 6.0)
    return float(hp.std() / (s.mean() + 1e-6))


class ShadingCorrector:
    """Per-channel flat-field (shading) correction.

    ``capture_reference`` averages frames of an empty/uniform field and
    smooths them; ``apply`` divides a raw frame by the normalized reference
    (gain map), preserving the frame's mean brightness. References persist
    to data/shading/<channel>.npy.

    Depth-resolved (Z-stack) references: phase contrast in particular has
    a shading pattern that changes with focal depth (annulus / meniscus
    refraction), so one flat field is wrong away from the Z it was taken
    at. ``capture_reference_stack`` records a gain map per Z plane over a
    range; ``apply(img, ch, z_um)`` then linearly blends the two planes
    bracketing ``z_um`` (clamped at the ends). Stacks persist to
    data/shading/<channel>_zstack.npz. A channel holds either a single
    map or a stack - capturing one replaces the other.
    """

    STACK_MAX_SIDE = 512     # stack planes stored downsampled (smooth maps)
    _CACHE_Z_RES_UM = 0.5    # blended map reused within this Z tolerance

    def __init__(self):
        self.references: dict[str, np.ndarray] = {}
        self.stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.enabled = False
        self.last_structure = 0.0
        self._cache: dict[str, tuple[tuple, np.ndarray]] = {}

    # ------------------------------------------------------------ capture

    @staticmethod
    def _gain_from_frames(frames: list[np.ndarray]) -> np.ndarray:
        ref = np.mean(frames, axis=0)
        if ref.ndim == 3:            # colour camera: one map for all planes
            ref = ref.mean(axis=2)
        if cv2 is not None:
            k = max(int(min(ref.shape[:2]) * 0.05) | 1, 31)
            ref = cv2.GaussianBlur(ref, (k, k), 0)
        ref[ref <= 0] = 1.0
        return (ref / float(ref.mean())).astype(np.float32)  # mean 1.0

    @staticmethod
    def _grab_frames(hw, n_frames: int) -> list[np.ndarray]:
        frames = []
        for _ in range(n_frames):
            f = hw.grab_fresh()
            if f is not None:
                frames.append(f.astype(np.float32))
        return frames

    def capture_reference(self, hw: ScopeHW, channel: str,
                          n_frames: int = 8) -> bool:
        frames = self._grab_frames(hw, n_frames)
        if not frames:
            return False
        self.references[channel] = self._gain_from_frames(frames)
        self._drop_stack(channel)
        self._cache.pop(channel, None)
        self._save(channel)
        return True

    # High-pass detail above this (fraction of mean) means the field is
    # probably not empty - labware / sample still in the light path.
    STRUCTURE_WARN = 0.03

    def capture_reference_stack(self, hw: ScopeHW, channel: str,
                                zs, n_frames: int = 6,
                                cancel: threading.Event | None = None,
                                status_cb=None,
                                settle_s: float = 0.2,
                                return_z: float | None = None) -> bool:
        """Capture a gain map at each Z in ``zs`` over an empty field (the
        labware removed). Planes are visited bottom to top (always moving
        up, for repeatable mechanics); the stage then goes to ``return_z``
        (default: the lowest plane). Returns False on cancel or if any
        plane fails - the old reference is then kept.

        ``self.last_structure`` holds the largest field-structure value
        seen; above ``STRUCTURE_WARN`` the field was probably not empty.
        """
        zs = np.unique(np.asarray(zs, dtype=np.float64))
        if zs.size == 0:
            return False
        lo = float(zs[0])
        self.last_structure = 0.0
        hw.move_z(max(lo - 5.0, 0.0))
        maps = []
        for k, z in enumerate(zs):
            if cancel is not None and cancel.is_set():
                return False
            if status_cb:
                status_cb(f'Shading stack {channel}: plane {k + 1}/{len(zs)} '
                          f'z={z:.1f} um')
            hw.move_z(float(z))
            if settle_s > 0:
                time.sleep(settle_s)
            hw.grab_fresh()          # discard a frame exposed during the move
            frames = self._grab_frames(hw, n_frames)
            if not frames:
                return False
            self.last_structure = max(self.last_structure,
                                      field_structure(frames[0]))
            maps.append(self._downsample(self._gain_from_frames(frames)))
        hw.move_z(float(lo if return_z is None else return_z))
        self.stacks[channel] = (zs, np.stack(maps).astype(np.float32))
        self.references.pop(channel, None)
        self._cache.pop(channel, None)
        self._save(channel)
        return True

    @classmethod
    def _downsample(cls, gain: np.ndarray) -> np.ndarray:
        h, w = gain.shape[:2]
        f = max(h, w) / cls.STACK_MAX_SIDE
        if f <= 1.0 or cv2 is None:
            return gain
        return cv2.resize(gain, (max(int(round(w / f)), 1),
                                 max(int(round(h / f)), 1)),
                          interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------- query

    def has_reference(self, channel: str) -> bool:
        return channel in self.references or channel in self.stacks

    def has_stack(self, channel: str) -> bool:
        return channel in self.stacks

    def describe(self, channel: str) -> str:
        if channel in self.stacks:
            zs = self.stacks[channel][0]
            return (f'{channel}: Z-stack, {len(zs)} planes '
                    f'{zs[0]:.0f}-{zs[-1]:.0f} um')
        if channel in self.references:
            return f'{channel}: single plane'
        return f'{channel}: none'

    def gain_at(self, channel: str, z_um: float | None = None
                ) -> np.ndarray | None:
        """Gain map for ``channel`` at ``z_um`` (stack: linear blend of the
        bracketing planes, clamped at the ends; no Z given -> the middle of
        the stack). Single-plane references ignore Z."""
        st = self.stacks.get(channel)
        if st is None:
            return self.references.get(channel)
        zs, maps = st
        if len(zs) == 1:
            return maps[0]
        z = float(zs[len(zs) // 2]) if z_um is None else \
            min(max(float(z_um), float(zs[0])), float(zs[-1]))
        j = int(np.clip(np.searchsorted(zs, z) - 1, 0, len(zs) - 2))
        t = (z - zs[j]) / (zs[j + 1] - zs[j])
        return ((1.0 - t) * maps[j] + t * maps[j + 1]).astype(np.float32)

    # -------------------------------------------------------------- apply

    def _full_gain(self, channel: str, z_um: float | None,
                   shape: tuple) -> np.ndarray | None:
        zq = None if z_um is None or channel not in self.stacks else \
            round(float(z_um) / self._CACHE_Z_RES_UM)
        key = (zq, shape)
        hit = self._cache.get(channel)
        if hit is not None and hit[0] == key:
            return hit[1]
        gain = self.gain_at(channel, z_um)
        if gain is None:
            return None
        if gain.shape != shape:
            if cv2 is None:
                return None
            gain = cv2.resize(gain, (shape[1], shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        self._cache[channel] = (key, gain)
        return gain

    def apply(self, img: np.ndarray, channel: str,
              z_um: float | None = None) -> np.ndarray:
        if img is None:
            return img
        gain = self._full_gain(channel, z_um, img.shape[:2])
        if gain is None:
            return img
        if img.ndim == 3:
            gain = gain[:, :, None]
        out = img.astype(np.float32) / gain
        if img.dtype == np.uint8:
            return np.clip(out, 0, 255).astype(np.uint8)
        return np.clip(out, 0, 65535).astype(np.uint16)

    # -------------------------------------------------------- persistence

    def _paths(self, channel: str) -> tuple[str, str]:
        return (os.path.join(SHADING_DIR, f'{channel}.npy'),
                os.path.join(SHADING_DIR, f'{channel}_zstack.npz'))

    def _drop_stack(self, channel: str) -> None:
        self.stacks.pop(channel, None)

    def _save(self, channel: str) -> None:
        single, stack = self._paths(channel)
        try:
            os.makedirs(SHADING_DIR, exist_ok=True)
            if channel in self.stacks:
                zs, maps = self.stacks[channel]
                np.savez_compressed(stack, zs=zs, maps=maps)
                if os.path.exists(single):
                    os.remove(single)
            elif channel in self.references:
                np.save(single, self.references[channel])
                if os.path.exists(stack):
                    os.remove(stack)
        except Exception:
            pass

    def load_all(self) -> None:
        for ch in CHANNELS:
            single, stack = self._paths(ch)
            try:
                if os.path.exists(stack):
                    with np.load(stack) as d:
                        self.stacks[ch] = (d['zs'].astype(np.float64),
                                           d['maps'].astype(np.float32))
                elif os.path.exists(single):
                    self.references[ch] = np.load(single)
            except Exception:
                pass
        self._cache.clear()


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


STAGE_X_MAX_UM = 120000.0
STAGE_Y_MAX_UM = 80000.0


def in_stage_bounds(x_um: float, y_um: float) -> bool:
    return 0.0 <= x_um <= STAGE_X_MAX_UM and 0.0 <= y_um <= STAGE_Y_MAX_UM


def tile_positions_spiral(center_x: float, center_y: float,
                          fov_x_um: float, fov_y_um: float,
                          n_rings: int,
                          overlap_pct: float = 10.0) -> list[tuple[float, float]]:
    """Center-out square spiral of tile centers (ring 0 = center tile)."""
    out = [(center_x, center_y)]
    gen = spiral_positions_unbounded(center_x, center_y, fov_x_um, fov_y_um,
                                     overlap_pct)
    next(gen)  # skip center (already added)
    sx = fov_x_um * (1.0 - overlap_pct / 100.0)
    per_ring = lambda r: 8 * r
    want = sum(per_ring(r) for r in range(1, n_rings + 1))
    for _ in range(want):
        try:
            out.append(next(gen))
        except StopIteration:
            break
    return out


def spiral_positions_unbounded(center_x: float, center_y: float,
                               fov_x_um: float, fov_y_um: float,
                               overlap_pct: float = 10.0):
    """Endless center-out square spiral generator.

    Yields only positions inside the stage travel; raises StopIteration
    once an entire ring falls outside the physical limits (the scan has
    reached the edges in every direction).
    """
    sx = fov_x_um * (1.0 - overlap_pct / 100.0)
    sy = fov_y_um * (1.0 - overlap_pct / 100.0)
    if in_stage_bounds(center_x, center_y):
        yield (center_x, center_y)
    ring = 1
    while True:
        pts = []
        for c in range(-ring, ring + 1):                       # top edge
            pts.append((center_x + c * sx, center_y - ring * sy))
        for r in range(-ring + 1, ring):                        # right edge
            pts.append((center_x + ring * sx, center_y + r * sy))
        for c in range(ring, -ring - 1, -1):                    # bottom edge
            pts.append((center_x + c * sx, center_y + ring * sy))
        for r in range(ring - 1, -ring, -1):                    # left edge
            pts.append((center_x - ring * sx, center_y + r * sy))
        emitted = 0
        for p in pts:
            if in_stage_bounds(*p):
                emitted += 1
                yield p
        if emitted == 0:
            return   # whole ring out of bounds: physical edge reached
        ring += 1


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
                  positions,
                  channel_settings: dict[str, ChannelSettings],
                  surface: FocusSurface | None,
                  shading: ShadingCorrector | None,
                  um_per_px: float,
                  out_dir: str,
                  cancel: threading.Event,
                  status_cb=None,
                  settle_s: float = 0.2,
                  stitch: bool = True,
                  use_channel_z_offsets: bool = True,
                  total: int | None = None,
                  on_tile=None) -> dict[str, str]:
    """Acquire every enabled channel at each tile position, then stitch.

    ``positions`` may be a list or an endless generator (spiral-until-edge);
    out-of-bounds positions are skipped. ``on_tile(channel, frame, x, y)``
    is called per captured tile for live mosaic preview. Returns
    {channel: stitched_path}.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    enabled = [ch for ch in CHANNELS
               if ch in channel_settings and channel_settings[ch].enabled]
    if not enabled:
        _status('Tile scan: no channels enabled ("Use in protocol")')
        return {}
    collected: dict[str, list[np.ndarray]] = {ch: [] for ch in enabled}
    used_positions: list[tuple[float, float]] = []
    n_txt = str(total) if total else '...'

    i = -1
    for (x, y) in positions:
        i += 1
        if cancel.is_set():
            _status('Tile scan stopped')
            break
        if not in_stage_bounds(x, y):
            continue
        _status(f'Tile {i + 1}/{n_txt}: moving to '
                f'({x / 1000:.2f}, {y / 1000:.2f}) mm')
        hw.move_xy(x, y)
        base_z = surface.z_at(x, y) if surface is not None else None
        used_positions.append((x, y))
        for ch in enabled:
            if cancel.is_set():
                break
            cs = channel_settings[ch]
            z_now = None
            if base_z is not None:
                z_off = cs.z_offset_um if use_channel_z_offsets else 0.0
                z_now = base_z + z_off
                hw.move_z(z_now)
            hw.set_channel(ch, cs)
            frame = hw.grab_settled(settle_s=settle_s)
            if frame is None:
                _status(f'Tile {i + 1}: no frame for {ch}')
                continue
            if shading is not None and shading.enabled:
                if z_now is None and shading.has_stack(ch):
                    z_now = hw.get_pos()[2]
                frame = shading.apply(frame, ch, z_now)
            save_image(os.path.join(out_dir, f'tile_{i:04d}_{ch}.tiff'), frame)
            collected[ch].append(frame)
            if on_tile is not None:
                try:
                    on_tile(ch, frame, x, y)
                except Exception:
                    pass
    hw.leds_off()

    stitched_paths: dict[str, str] = {}
    if stitch:
        for ch in enabled:
            tiles = collected[ch]
            if len(tiles) == len(used_positions) and tiles:
                _status(f'Stitching {ch} ({len(tiles)} tiles)...')
                mosaic = stitch_tiles(tiles, used_positions, um_per_px)
                if mosaic is not None:
                    p = os.path.join(out_dir, f'stitched_{ch}.tiff')
                    save_image(p, mosaic)
                    stitched_paths[ch] = p
    _status(f'Tile scan finished ({len(used_positions)} tiles) -> {out_dir}')
    return stitched_paths


def run_z_stack(hw: ScopeHW,
                channel_settings: dict[str, ChannelSettings],
                channels: list[str],
                center_z_um: float,
                shading: ShadingCorrector | None,
                out_dir: str,
                cancel: threading.Event,
                status_cb=None,
                settle_s: float = 0.1,
                use_channel_z_offsets: bool = True) -> None:
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
        zs = zstack_planes(center_z_um, cs, use_channel_z_offsets,
                           force_stack=True)
        hw.set_channel(ch, cs)
        _status(f'Z-stack {ch}: {len(zs)} slices '
                f'({zs[0]:.1f} to {zs[-1]:.1f} um)')
        for j, z in enumerate(zs):
            if cancel.is_set():
                break
            hw.move_z(float(z))
            frame = hw.grab_settled(settle_s=settle_s)
            if frame is None:
                continue
            if shading is not None and shading.enabled:
                frame = shading.apply(frame, ch, float(z))
            save_image(os.path.join(out_dir, f'zstack_{ch}_{j:03d}_'
                                             f'{z:.1f}um.tiff'), frame)
    hw.move_z(center_z_um)
    hw.leds_off()
    _status(f'Z-stack finished -> {out_dir}')


def measure_image_shift(img_a: np.ndarray, img_b: np.ndarray,
                        downsample_to: int = 512):
    """Translation of img_b relative to img_a in full-res pixels via phase
    correlation. Returns (dx_px, dy_px, response) or None. Image coords:
    +x right, +y down."""
    if cv2 is None or img_a is None or img_b is None:
        return None
    if img_a.shape != img_b.shape or img_a.size == 0:
        return None
    f = max(img_a.shape[0] // downsample_to, 1)
    a = img_a[::f, ::f].astype(np.float32)
    b = img_b[::f, ::f].astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    try:
        (sx, sy), resp = cv2.phaseCorrelate(a, b)
    except Exception:
        return None
    return sx * f, sy * f, float(resp)


def channel_settings_to_dict(cs: dict[str, ChannelSettings]) -> dict:
    return {k: asdict(v) for k, v in cs.items()}


# ----------------------------------------------------- plate calibration

@dataclass
class CalTarget:
    """One guided calibration position (LAS X style)."""
    label: str                 # e.g. "A1 12 o'clock"
    plate_mm: tuple[float, float]   # nominal position, plate top-left origin
    marked_stage_um: tuple[float, float] | None = None


class PlateCalibration:
    """LAS X-style plate calibration.

    Guided targets: the 12, 3, 6 and 9 o'clock well-edge positions of A1,
    the 9 o'clock position of the last well of the first row, and the
    12 o'clock position of the last well of the first column. The user
    centers the FOV on each physical position and marks it.

    From the six marks we solve:
    - the A1 well center (stage um) and the measured well diameter,
    - a full 2x2 affine mapping plate-mm -> stage-um, which captures the
      well spacing/distances, any plate rotation, and axis directions.
    """

    def __init__(self, labware):
        self.labware = labware
        self.targets = self._build_targets(labware)
        self.index = 0
        # Results
        self.a1_center_stage_um: tuple[float, float] | None = None
        self.matrix: list[list[float]] | None = None   # 2x2, mm -> um
        self.measured_well_diameter_mm: float | None = None
        self.rotation_deg: float | None = None
        self.measured_spacing_x_mm: float | None = None
        self.measured_spacing_y_mm: float | None = None

    @staticmethod
    def _build_targets(lw) -> list[CalTarget]:
        rx = lw.well_size_x_mm / 2.0
        ry = lw.well_size_y_mm / 2.0
        a1x, a1y = lw.a1_offset_x_mm, lw.a1_offset_y_mm
        last_col = lw.cols - 1
        last_row = lw.rows - 1
        row_end_x = a1x + last_col * lw.spacing_x_mm
        col_end_y = a1y + last_row * lw.spacing_y_mm
        first_row_name = lw.get_well_name(0, last_col)
        first_col_name = lw.get_well_name(last_row, 0)
        return [
            CalTarget("A1 12 o'clock", (a1x, a1y - ry)),
            CalTarget("A1 3 o'clock", (a1x + rx, a1y)),
            CalTarget("A1 6 o'clock", (a1x, a1y + ry)),
            CalTarget("A1 9 o'clock", (a1x - rx, a1y)),
            CalTarget(f"{first_row_name} 9 o'clock",
                      (row_end_x - rx, a1y)),
            CalTarget(f"{first_col_name} 12 o'clock",
                      (a1x, col_end_y - ry)),
        ]

    @property
    def current(self) -> CalTarget:
        return self.targets[self.index]

    @property
    def complete(self) -> bool:
        return all(t.marked_stage_um is not None for t in self.targets)

    def mark_current(self, stage_x_um: float, stage_y_um: float) -> None:
        self.targets[self.index].marked_stage_um = (float(stage_x_um),
                                                    float(stage_y_um))

    def next(self) -> bool:
        if self.index < len(self.targets) - 1:
            self.index += 1
            return True
        return False

    def prev(self) -> bool:
        if self.index > 0:
            self.index -= 1
            return True
        return False

    def provisional_solve(self) -> bool:
        """Axis-aligned estimate from the four A1 edge marks, used to
        auto-drive the stage near the remaining targets."""
        if any(t.marked_stage_um is None for t in self.targets[:4]):
            return False
        lw = self.labware
        m12, m3, m6, m9 = [np.asarray(t.marked_stage_um, dtype=np.float64)
                           for t in self.targets[:4]]
        c_a1 = (m12 + m3 + m6 + m9) / 4.0
        self.a1_center_stage_um = (float(c_a1[0]), float(c_a1[1]))
        rx = lw.well_size_x_mm / 2.0
        ry = lw.well_size_y_mm / 2.0
        ex = (m3 - m9) / (2 * rx)
        ey = (m6 - m12) / (2 * ry)
        self.matrix = [[float(ex[0]), float(ey[0])],
                       [float(ex[1]), float(ey[1])]]
        return True

    def estimate_stage_for(self, target: CalTarget,
                           fallback_transform) -> tuple[float, float] | None:
        """Best-known stage position for a target: its own mark, else the
        current (possibly partial) solve, else the caller's transform."""
        if target.marked_stage_um is not None:
            return target.marked_stage_um
        if self.matrix is not None and self.a1_center_stage_um is not None:
            return self.plate_mm_to_stage_um(*target.plate_mm)
        if fallback_transform is not None:
            lw = self.labware
            rel = (target.plate_mm[0] - lw.a1_offset_x_mm,
                   target.plate_mm[1] - lw.a1_offset_y_mm)
            return fallback_transform(rel[0], rel[1])
        return None

    def solve(self) -> bool:
        """Compute A1 center, well size, spacing, rotation and the affine
        plate->stage transform from the six marks."""
        if not self.complete:
            return False
        lw = self.labware
        m12, m3, m6, m9, m_row, m_col = [np.asarray(t.marked_stage_um,
                                                    dtype=np.float64)
                                         for t in self.targets]
        # A1 center: midpoints of the opposing edge marks.
        c_a1 = (m12 + m3 + m6 + m9) / 4.0
        self.a1_center_stage_um = (float(c_a1[0]), float(c_a1[1]))

        # Plate-frame nominal vectors from the A1 center (mm):
        rx = lw.well_size_x_mm / 2.0
        ry = lw.well_size_y_mm / 2.0
        last_col = lw.cols - 1
        last_row = lw.rows - 1
        n_row = np.array([last_col * lw.spacing_x_mm - rx, 0.0])
        n_col = np.array([0.0, last_row * lw.spacing_y_mm - ry])
        v_row = m_row - c_a1
        v_col = m_col - c_a1
        if last_col == 0 or last_row == 0 or abs(n_row[0]) < 1e-9 or abs(n_col[1]) < 1e-9:
            # Degenerate (single-row/column labware): axis-aligned fallback
            # from the A1 edge marks only.
            ex = (m3 - m9)
            ey = (m6 - m12)
            M = np.column_stack([ex / (2 * rx), ey / (2 * ry)])
        else:
            N = np.column_stack([n_row, n_col])    # plate-frame
            V = np.column_stack([v_row, v_col])    # stage-frame
            try:
                M = V @ np.linalg.inv(N)
            except np.linalg.LinAlgError:
                return False
        self.matrix = [[float(M[0, 0]), float(M[0, 1])],
                       [float(M[1, 0]), float(M[1, 1])]]

        # Measured quantities for display / labware refinement.
        scale_x = float(np.linalg.norm(M[:, 0]))   # um per plate-mm along x
        scale_y = float(np.linalg.norm(M[:, 1]))
        d_x = float(np.linalg.norm(m3 - m9)) / scale_x if scale_x > 0 else 0.0
        d_y = float(np.linalg.norm(m6 - m12)) / scale_y if scale_y > 0 else 0.0
        self.measured_well_diameter_mm = (d_x + d_y) / 2.0
        self.rotation_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if last_col > 0 and scale_x > 0:
            row_len_mm = float(np.linalg.norm(v_row)) / scale_x
            self.measured_spacing_x_mm = (row_len_mm + d_x / 2.0) / last_col
        if last_row > 0 and scale_y > 0:
            col_len_mm = float(np.linalg.norm(v_col)) / scale_y
            self.measured_spacing_y_mm = (col_len_mm + d_y / 2.0) / last_row
        return True

    # ------------------------------------------------------- transforms

    def plate_mm_to_stage_um(self, plate_x_mm: float,
                             plate_y_mm: float) -> tuple[float, float]:
        """Plate coords (top-left origin, mm) -> stage um via the solve."""
        lw = self.labware
        rel = np.array([plate_x_mm - lw.a1_offset_x_mm,
                        plate_y_mm - lw.a1_offset_y_mm])
        M = np.asarray(self.matrix)
        c = np.asarray(self.a1_center_stage_um)
        out = c + M @ rel
        return float(out[0]), float(out[1])

    def stage_um_to_plate_mm(self, x_um: float,
                             y_um: float) -> tuple[float, float] | None:
        try:
            M = np.asarray(self.matrix)
            c = np.asarray(self.a1_center_stage_um)
            rel = np.linalg.solve(M, np.array([x_um, y_um]) - c)
        except Exception:
            return None
        lw = self.labware
        return (float(rel[0]) + lw.a1_offset_x_mm,
                float(rel[1]) + lw.a1_offset_y_mm)

    def to_dict(self) -> dict:
        return {
            'labware_name': getattr(self.labware, 'name', ''),
            'a1_center_stage_um': self.a1_center_stage_um,
            'matrix': self.matrix,
            'measured_well_diameter_mm': self.measured_well_diameter_mm,
            'rotation_deg': self.rotation_deg,
            'measured_spacing_x_mm': self.measured_spacing_x_mm,
            'measured_spacing_y_mm': self.measured_spacing_y_mm,
            'marks': [t.marked_stage_um for t in self.targets],
        }

    def from_dict(self, d: dict) -> None:
        a1 = d.get('a1_center_stage_um')
        self.a1_center_stage_um = tuple(a1) if a1 else None
        self.matrix = d.get('matrix')
        self.measured_well_diameter_mm = d.get('measured_well_diameter_mm')
        self.rotation_deg = d.get('rotation_deg')
        self.measured_spacing_x_mm = d.get('measured_spacing_x_mm')
        self.measured_spacing_y_mm = d.get('measured_spacing_y_mm')
        marks = d.get('marks') or []
        for t, m in zip(self.targets, marks):
            t.marked_stage_um = tuple(m) if m else None


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
            enabled=bool(v.get('enabled', False)),
            z_offset_um=float(v.get('z_offset_um', 0.0)),
            zstack=ZStackConfig(
                enabled=bool(z.get('enabled', False)),
                range_um=float(z.get('range_um', 50.0)),
                step_um=float(z.get('step_um', 5.0)),
            ),
        )
    return out
