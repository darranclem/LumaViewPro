"""Brightfield image controls (acquisition.py): log sliders, saturation,
flicker-free LED changes, and enforcing the camera's manual exposure."""

from __future__ import annotations

import numpy as np
import pytest

import acquisition as acq


# ------------------------------------------------------------ sliders

@pytest.mark.parametrize('lo,hi', [(0.02, 1000.0), (0.005, 10000.0)])
def test_log_slider_round_trips_and_spans_the_range(lo, hi):
    assert acq.log_slider_value(0, lo, hi) == pytest.approx(lo)
    assert acq.log_slider_value(acq.SLIDER_STEPS, lo, hi) == pytest.approx(hi)
    for v in (lo, 0.1, 1.0, 7.5, hi):
        pos = acq.log_slider_pos(v, lo, hi)
        assert acq.log_slider_value(pos, lo, hi) == pytest.approx(v)


def test_brightfield_exposures_get_real_slider_travel():
    # 0.02 - 1000 ms: the sub-5 ms brightfield range now gets about half
    # the slider (it had < 0.5 % on the old linear 1-1000 ms slider).
    span = acq.log_slider_pos(5.0, 0.02, 1000) - acq.log_slider_pos(0.02, 0.02, 1000)
    assert span / acq.SLIDER_STEPS > 0.45


def test_equal_travel_is_an_equal_brightness_ratio():
    a = acq.log_slider_value(200, 0.02, 1000)
    b = acq.log_slider_value(300, 0.02, 1000)
    c = acq.log_slider_value(400, 0.02, 1000)
    assert b / a == pytest.approx(c / b)


def test_led_slider_has_an_exact_off_and_low_current_resolution():
    assert acq.log_slider_value(0, 1, 600, zero_at_start=True) == 0.0
    assert acq.log_slider_value(1, 1, 600, zero_at_start=True) == pytest.approx(1)
    assert acq.log_slider_pos(0, 1, 600, zero_at_start=True) == 0.0
    # 1 - 20 mA (typical brightfield) gets ~ half the travel, not 3 %.
    assert acq.log_slider_pos(20, 1, 600, zero_at_start=True) > 450


# --------------------------------------------------------- saturation

def test_saturated_fraction_uses_the_sensor_bit_depth():
    img = np.zeros((10, 10), np.uint16)
    img[0, :5] = 4095                     # 12-bit full scale
    img[1, :5] = 4094
    assert acq.saturation_level(12) == 4095
    assert acq.saturated_fraction(img, 12) == pytest.approx(0.05)
    assert acq.saturated_fraction(img.astype(np.uint8) * 0 + 255, 8) == 1.0


# ---------------------------------------------------------------- LED

class _Board:
    driver = True

    def __init__(self):
        self.sent = []

    def exchange_command(self, cmd):
        self.sent.append(cmd)


def test_changing_power_on_the_live_channel_does_not_blink_the_led():
    led = _Board()
    hw = acq.ScopeHW(led=led)
    hw.set_channel('BF', acq.ChannelSettings(led_power=10))
    assert led.sent == ['LEDS_OFF', 'LEDS_ENT', 'LED3_10']
    led.sent.clear()
    hw.set_channel('BF', acq.ChannelSettings(led_power=12))
    assert led.sent == ['LED3_12']            # no LEDS_OFF flash
    led.sent.clear()
    hw.set_channel('PC', acq.ChannelSettings(led_power=12))
    assert led.sent[:2] == ['LEDS_OFF', 'LEDS_ENT']   # switching: others off
    hw.leds_off()
    led.sent.clear()
    hw.set_channel('PC', acq.ChannelSettings(led_power=12))
    assert led.sent[0] == 'LEDS_OFF'          # after off, a full re-enable


# ------------------------------------------------- camera auto modes

class _Node:
    def __init__(self, value, sticky=False):
        self.value = value
        self.sticky = sticky              # refuses to change

    def GetValue(self):
        return self.value

    def SetValue(self, v):
        if not self.sticky:
            self.value = v


class _Pylon:
    def __init__(self, gain_auto='Continuous', exp_auto='Continuous',
                 sticky=False):
        self.GainAuto = _Node(gain_auto, sticky)
        self.ExposureAuto = _Node(exp_auto, sticky)


class _Cam:
    def __init__(self, active, auto_gain_raises=False):
        self.active = active
        self.exposures, self.gains = [], []
        self._raise = auto_gain_raises

    def auto_gain(self, state=True):
        if self._raise:
            raise RuntimeError('node locked')
        self.active.GainAuto.SetValue('Off')
        self.active.ExposureAuto.SetValue('Off')

    def exposure_t(self, ms):
        self.exposures.append(ms)

    def gain(self, db):
        self.gains.append(db)


def test_camera_left_in_auto_is_forced_manual_before_settings_apply():
    cam = _Cam(_Pylon())
    hw = acq.ScopeHW(camera=cam)
    hw.apply_camera_settings(acq.ChannelSettings(exposure_ms=0.3))
    assert hw.camera_auto_state() == {'GainAuto': 'Off', 'ExposureAuto': 'Off'}
    assert cam.exposures == [0.3]


def test_auto_is_forced_off_node_by_node_if_the_driver_call_fails():
    cam = _Cam(_Pylon(), auto_gain_raises=True)
    hw = acq.ScopeHW(camera=cam)
    assert hw.ensure_manual_exposure()
    assert set(hw.camera_auto_state().values()) == {'Off'}


def test_auto_that_will_not_turn_off_is_reported():
    hw = acq.ScopeHW(camera=_Cam(_Pylon(sticky=True), auto_gain_raises=True))
    assert hw.ensure_manual_exposure() is False


def test_manual_camera_is_left_alone():
    cam = _Cam(_Pylon('Off', 'Off'))
    hw = acq.ScopeHW(camera=cam)
    assert hw.ensure_manual_exposure()


def test_camera_limits_come_from_the_camera():
    class Prof:
        exposure_min_us = 19.0
        exposure_max_us = 10_000_000.0

        class gain:
            total_min_db = 0.0
            total_max_db = 36.0
    cam = _Cam(_Pylon('Off', 'Off'))
    cam.profile = Prof
    cam.max_exposure = 1000.0            # driver cap is tighter
    lo, hi, glo, ghi = acq.ScopeHW(camera=cam).camera_limits()
    assert lo == pytest.approx(0.019) and hi == 1000.0
    assert (glo, ghi) == (0.0, 36.0)


# ------------------------------------------- double-click centering check

def _scene(seed=0, size=(600, 800)):
    rng = np.random.default_rng(seed)
    img = rng.normal(100, 5, size).astype(np.float32)
    for _ in range(150):
        y, x = rng.integers(10, size[0] - 10), rng.integers(10, size[1] - 10)
        img[y - 4:y + 4, x - 4:x + 4] += rng.uniform(40, 120)
    return img


def _shifted(img, dx, dy):
    """Content moved by (dx, dy) px; uncovered area filled with noise."""
    out = np.random.default_rng(9).normal(100, 5, img.shape).astype(np.float32)
    h, w = img.shape
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    out[yd:h - ys, xd:w - xs] = img[ys:h - yd, xs:w - xd]
    return out


@pytest.mark.parametrize('click, error', [((650, 120), (0, 0)),
                                          ((650, 120), (9, -6)),
                                          ((90, 520), (-14, 11)),
                                          ((780, 590), (5, 5))])
def test_the_clicked_spot_is_found_after_a_move_that_leaves_little_overlap(
        click, error):
    """A click far from the center means the two frames barely overlap;
    the clicked neighbourhood must still be located, including the error
    the move left."""
    a = _scene()
    cx, cy = 400, 300
    dx = cx - click[0] + error[0]         # where the content actually went
    dy = cy - click[1] + error[1]
    b = _shifted(a, dx, dy)
    x, y, score = acq.locate_patch(a, click, b, (cx, cy))
    assert score > 0.8
    assert (x - cx, y - cy) == pytest.approx(error, abs=1.0)


def test_a_featureless_field_gives_no_measurement():
    flat = np.full((600, 800), 50, dtype=np.float32)
    assert acq.locate_patch(flat, (500, 200), flat, (400, 300)) is None


def test_whole_frame_shift_measurement_keeps_its_sign_convention():
    a = _scene(3)
    s = acq.measure_image_shift(a, _shifted(a, 20, -12))
    assert (s[0], s[1]) == pytest.approx((20, -12), abs=0.5)


# ------------------------------------------------------------ shutdown

def test_parking_lowers_the_objective_before_centering_the_stage():
    calls = []

    class Stage:
        driver = True

        def move_abs_pos(self, axis, pos):
            calls.append(('move', axis, pos))

        def xycenter(self):
            calls.append(('center',))

        def target_status(self, axis):
            calls.append(('wait', axis))
            return True

        def current_pos(self, axis):
            return self.z

        def zhome(self):
            calls.append(('zhome',))

    stage = Stage()
    stage.z = 0.0
    hw = acq.ScopeHW(stage=stage)
    assert hw.park() is True
    assert calls == [('move', 'Z', 0.0), ('wait', 'Z'), ('center',),
                     ('wait', 'X'), ('wait', 'Y')]
    assert acq.ScopeHW().park() is False        # no stage: nothing to do

    # Objective not confirmed down (even after homing Z): XY must not move.
    calls.clear()
    stage.z = 3000.0
    assert hw.park() is False
    assert ('zhome',) in calls and ('center',) not in calls


def test_parking_gives_up_when_a_stuck_thread_holds_the_hardware_lock():
    import threading

    class Stage:
        driver = True

    hw = acq.ScopeHW(stage=Stage())
    held, release = threading.Event(), threading.Event()

    def _stuck():
        with hw.lock:
            held.set()
            release.wait(5.0)

    t = threading.Thread(target=_stuck, daemon=True)
    t.start()
    held.wait(2.0)
    try:
        assert hw.park(lock_timeout_s=0.1) is False
        assert hw.get_pos_now(lock_timeout_s=0.1) is None
    finally:
        release.set()
        t.join(2.0)


# ------------------------------- spiral overview: one picture per field

_TINTS = {'BF': (1.0, 1.0, 1.0), 'GREEN': (0.0, 1.0, 0.0),
          'RED': (1.0, 0.0, 0.0)}


def _field(value, spot=None, shape=(40, 60)):
    f = np.full(shape, value, np.uint16)
    if spot is not None:
        f[10:20, 10:20] = spot
    return f


def test_overview_puts_every_colour_of_a_field_in_one_picture():
    limits = {}
    green = _field(100, 4000)
    red = _field(100)
    red[25:35, 40:50] = 3000
    rgb = acq.overview_rgb({'GREEN': green, 'RED': red}, _TINTS, limits)
    assert rgb.dtype == np.uint8 and rgb.shape == (40, 60, 3)
    assert rgb[15, 15, 1] > 200 and rgb[15, 15, 0] < 30    # green spot
    assert rgb[30, 45, 0] > 200 and rgb[30, 45, 1] < 30    # red spot
    assert rgb[2, 2].max() < 30                            # background
    assert set(limits) == {'GREEN', 'RED'}


def test_overview_is_made_small():
    rgb = acq.overview_rgb({'BF': _field(500, shape=(1200, 1920))},
                           _TINTS, {}, max_px=192)
    assert max(rgb.shape[:2]) <= 192


def test_overview_scales_every_field_of_a_scan_alike():
    limits = {}
    acq.overview_rgb({'GREEN': _field(100, 4000)}, _TINTS, limits)
    # a dimmer field later in the scan is drawn dimmer, not stretched
    dim = acq.overview_rgb({'GREEN': _field(100, 1000)}, _TINTS, limits)
    alone = acq.overview_rgb({'GREEN': _field(100, 1000)}, _TINTS, {})
    assert dim[15, 15, 1] < 100 < alone[15, 15, 1]


def test_overview_does_not_stretch_an_empty_field_into_noise():
    rng = np.random.default_rng(1)
    empty = (100 + rng.integers(0, 4, (40, 60))).astype(np.uint16)
    rgb = acq.overview_rgb({'GREEN': empty}, _TINTS, {})
    assert rgb.max() < 80


def test_overview_transmitted_light_does_not_drown_fluorescence():
    bf = _field(3000)
    with_fluo = acq.overview_rgb({'BF': bf, 'GREEN': _field(100, 4000)},
                                 _TINTS, {})
    alone = acq.overview_rgb({'BF': bf}, _TINTS, {})
    assert alone[2, 2, 0] > 200            # full strength on its own
    assert 100 < with_fluo[2, 2, 0] < 150  # half strength beside a colour
    assert with_fluo[15, 15, 1] == 255     # the green spot shows on top


def test_overview_of_nothing_is_none():
    assert acq.overview_rgb({}, _TINTS, {}) is None
