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


# --------------------------- double-click centering: never a second move

def test_a_check_frame_from_before_the_move_is_not_corrected():
    # A 300 px move; the "new" frame still shows the old view, so the spot
    # is found 300 px off. Correcting that would make the move twice.
    assert not acq.trim_allowed((-300, 0), (300, 0), 2160, True)
    assert not acq.trim_allowed((-200, -150), (200, 150), 2160, True)


def test_nothing_is_corrected_before_the_stage_has_arrived():
    assert not acq.trim_allowed((-300, 0), (20, 5), 2160, False)


def test_a_small_real_offset_is_trimmed_once():
    assert acq.trim_allowed((-300, 0), (20, 5), 2160, True)
    assert acq.trim_allowed((-800, 400), (60, -30), 2160, True)
    # a short move lost in the drive's backlash is made up
    assert acq.trim_allowed((-30, 0), (28, 0), 2160, True)
    # nothing to do within 3 px; too far off to be a trim
    assert not acq.trim_allowed((-300, 0), (2, 1), 2160, True)
    assert not acq.trim_allowed((-900, 0), (400, 0), 2160, True)


def test_the_image_moving_twice_as_far_as_planned_is_noticed():
    assert acq.centering_scale((-300, 100), (-600, 200)) == pytest.approx(2.0)
    assert acq.centering_scale((-300, 100), (-300, 100)) == pytest.approx(1.0)
    assert acq.centering_scale((-300, 0), (0, 0)) == pytest.approx(0.0)
    assert acq.centering_scale((0, 0), (5, 5)) is None


def test_stage_is_waited_for_until_it_reports_being_at_its_target():
    import threading

    class Stage:
        def __init__(self):
            self.cur = {'X': 0.0, 'Y': 0.0}
            self.tgt = {'X': 50.0, 'Y': -20.0}
            self.reads = 0

        def current_pos(self, a):
            self.reads += 1
            if self.reads > 6:                 # arrives after a few polls
                self.cur = dict(self.tgt)
            return self.cur[a]

        def target_pos(self, a):
            return self.tgt[a]

    hw = acq.ScopeHW.__new__(acq.ScopeHW)
    hw.lock = threading.RLock()
    hw.stage = Stage()
    hw.stage_ok = lambda: True
    assert hw.xy_arrived(timeout_s=2.0) is True
    stuck = Stage()
    stuck.current_pos = lambda a: 0.0
    hw.stage = stuck
    assert hw.xy_arrived(timeout_s=0.1) is False


def test_only_a_move_short_enough_to_be_lost_in_backlash_is_made_up_whole():
    assert acq.trim_allowed((-30, 0), (30, 0), 2160, True, floor_px=33)
    assert not acq.trim_allowed((-100, -50), (100, 50), 2160, True,
                                floor_px=33)


# ------------------------------------------------ the Autofocus channel

def test_binning_sums_blocks_and_stops_at_full_scale():
    img = np.arange(16, dtype=np.uint16).reshape(4, 4)
    out = acq.bin_sum(img, 2)
    assert out.dtype == np.uint16 and out.shape == (2, 2)
    assert out.tolist() == [[10, 18], [42, 50]]
    assert acq.bin_sum(img, 2, top=40).tolist() == [[10, 18], [40, 40]]
    assert acq.bin_sum(img, 1) is img                  # unbinned: untouched
    bright = np.full((8, 8), 200, dtype=np.uint8)
    assert acq.bin_sum(bright, 8).tolist() == [[255]]  # 8 bit: cut at 255
    odd = np.ones((5, 7), dtype=np.uint16)             # the rest is dropped
    assert acq.bin_sum(odd, 2).shape == (2, 3)


def test_autofocus_channel_is_there_by_default_and_is_not_a_colour():
    cs = acq.default_channel_settings()
    assert acq.AF_CHANNEL in cs and acq.AF_CHANNEL not in acq.CHANNELS
    af = cs[acq.AF_CHANNEL]
    assert af.enabled and af.source == 'BF' and af.binning in acq.AF_BINNINGS
    assert all(cs[ch].binning == 1 for ch in acq.CHANNELS)


def test_autofocus_channel_lights_the_light_chosen_for_it():
    led = _Board()
    hw = acq.ScopeHW(led=led)
    af = acq.ChannelSettings(led_power=80, source='GREEN', binning=4)
    hw.set_channel(acq.AF_CHANNEL, af)
    assert led.sent == ['LEDS_OFF', 'LEDS_ENT', 'LED1_80']
    assert hw.active_channel == acq.AF_CHANNEL and hw.soft_bin == 4
    led.sent.clear()
    af.led_power = 90
    hw.set_channel(acq.AF_CHANNEL, af)
    assert led.sent == ['LED1_90']                 # same light: no blink
    led.sent.clear()
    af.source = 'BF'                               # another light: the
    hw.set_channel(acq.AF_CHANNEL, af)             # old one goes off first
    assert led.sent == ['LEDS_OFF', 'LEDS_ENT', 'LED3_90']
    hw.set_channel('BF', acq.ChannelSettings(led_power=10))
    assert hw.soft_bin == 1                        # colours are never binned
    hw.set_channel(acq.AF_CHANNEL, af)
    hw.set_channel(None)
    assert hw.soft_bin == 1


def test_frames_are_binned_only_while_the_autofocus_channel_is_lit():
    class Cam:
        active = True
        array = np.full((8, 8), 10, dtype=np.uint16)

        def grab_new_capture(self, timeout):
            return True, 0.0

        def get_pixel_format(self):
            return 'Mono12'

        def exposure_t(self, ms):
            pass

        def gain(self, db):
            pass

    hw = acq.ScopeHW(camera=Cam())
    hw.ensure_manual_exposure = lambda: True
    hw.set_channel(acq.AF_CHANNEL,
                   acq.ChannelSettings(source='BF', binning=2))
    f = hw.grab_fresh()
    assert f.shape == (4, 4) and int(f[0, 0]) == 40
    Cam.array = np.full((8, 8), 2000, dtype=np.uint16)
    assert int(hw.grab_fresh()[0, 0]) == 4095      # 12 bit full scale
    hw.set_channel('BF', acq.ChannelSettings())
    assert hw.grab_fresh().shape == (8, 8)


def test_settings_without_an_autofocus_channel_adopt_the_old_one():
    old = {'GREEN': {'exposure_ms': 80.0, 'gain_db': 6.0, 'led_power': 300,
                     'enabled': True}}
    cs = acq.channel_settings_from_dict(old)
    acq.adopt_af_channel(cs, 'GREEN', enabled=False)
    af = cs[acq.AF_CHANNEL]
    assert (af.source, af.exposure_ms, af.gain_db, af.led_power) == \
        ('GREEN', 80.0, 6.0, 300)
    assert af.binning == 1 and af.enabled is False
    acq.adopt_af_channel(cs, 'nonsense')
    assert cs[acq.AF_CHANNEL].source == 'BF'


def test_autofocus_channel_settings_survive_saving_and_are_checked():
    cs = acq.default_channel_settings()
    cs[acq.AF_CHANNEL] = acq.ChannelSettings(
        exposure_ms=2.0, led_power=400, enabled=False, source='RED',
        binning=8)
    back = acq.channel_settings_from_dict(acq.channel_settings_to_dict(cs))
    af = back[acq.AF_CHANNEL]
    assert (af.source, af.binning, af.enabled, af.exposure_ms) == \
        ('RED', 8, False, 2.0)
    bad = acq.channel_settings_to_dict(cs)
    bad[acq.AF_CHANNEL].update(source='UV', binning=3, z_offset_um=7.0)
    bad[acq.AF_CHANNEL]['zstack']['enabled'] = True
    af = acq.channel_settings_from_dict(bad)[acq.AF_CHANNEL]
    assert (af.source, af.binning, af.z_offset_um, af.zstack.enabled) == \
        ('BF', 1, 0.0, False)


def test_autofocus_scores_by_the_light_not_by_the_channel_name():
    assert acq.led_channel('GREEN', None) == 'GREEN'
    assert acq.led_channel(acq.AF_CHANNEL,
                           acq.ChannelSettings(source='RED')) == 'RED'
    assert acq.led_channel(acq.AF_CHANNEL, acq.ChannelSettings()) == 'BF'


def test_imaged_colours_share_one_binning_and_autofocus_keeps_its_own():
    cs = acq.default_channel_settings()
    af_bin = cs[acq.AF_CHANNEL].binning
    assert acq.set_imaging_binning(cs, 4) == 4
    assert {cs[ch].binning for ch in acq.CHANNELS} == {4}
    assert cs[acq.AF_CHANNEL].binning == af_bin        # independent
    assert acq.set_imaging_binning(cs, 3) == 1         # not a binning
    cs[acq.AF_CHANNEL].binning = 8
    acq.set_imaging_binning(cs, 2)
    back = acq.channel_settings_from_dict(acq.channel_settings_to_dict(cs))
    assert acq.imaging_binning(back) == 2
    assert back[acq.AF_CHANNEL].binning == 8
    # A file in which the colours disagree: they all take the first one's.
    d = acq.channel_settings_to_dict(cs)
    d['RED']['binning'] = 8
    mixed = acq.channel_settings_from_dict(d)
    assert {mixed[ch].binning for ch in acq.CHANNELS} == {2}
