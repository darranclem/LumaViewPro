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
