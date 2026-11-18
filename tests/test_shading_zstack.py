"""Depth-resolved (Z-stack) shading correction (acquisition.ShadingCorrector).

Phase contrast shading changes with focal depth, so the reference is a
stack of gain maps over Z, blended at the imaging Z.
"""

from __future__ import annotations

import numpy as np
import pytest

import acquisition as acq

cv2 = pytest.importorskip('cv2')

H, W = 600, 800


def _pattern(z):
    """Smooth vignetting whose centre slides and depth deepens with Z."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W / 2 + (z - 50.0) * 3.0
    r2 = ((xx - cx) / W) ** 2 + ((yy - H / 2) / H) ** 2
    return 1.0 - (0.3 + 0.004 * z) * r2


class _FlatFieldScope:
    """Empty field: frame = level * pattern(z) + noise."""

    def __init__(self, seed=0):
        self.z = 0.0
        self.z_history = []
        self.rng = np.random.default_rng(seed)

    def move_z(self, z, wait=True):
        self.z = float(z)
        self.z_history.append(self.z)

    def grab_fresh(self, timeout_s=3.0):
        img = 2000.0 * _pattern(self.z) + self.rng.normal(0, 10, (H, W))
        return np.clip(img, 0, 65535).astype(np.uint16)

    def get_pos(self):
        return (0.0, 0.0, self.z)


@pytest.fixture
def shading(tmp_path, monkeypatch):
    monkeypatch.setattr(acq, 'SHADING_DIR', str(tmp_path))
    monkeypatch.setattr(acq.time, 'sleep', lambda s: None)
    return acq.ShadingCorrector()


def _flatness(img):
    c = img.astype(np.float64)[50:-50, 50:-50]
    return c.std() / c.mean()


def _capture(shading, scope, lo=0.0, hi=100.0, step=20.0, **kw):
    return shading.capture_reference_stack(
        scope, 'PC', np.arange(lo, hi + 1e-6, step), n_frames=2,
        settle_s=0, **kw)


def test_the_stack_is_captured_bottom_to_top(shading):
    scope = _FlatFieldScope()
    assert _capture(shading, scope)
    zs, maps = shading.stacks['PC']
    assert list(zs) == [0, 20, 40, 60, 80, 100]
    assert maps.shape[0] == 6 and max(maps.shape[1:]) <= 512
    planes = [z for z in scope.z_history if z >= 0][:-1]   # drop return
    assert planes == sorted(planes)                          # always upward


@pytest.mark.parametrize('z', [0.0, 30.0, 47.3, 100.0])
def test_a_stack_flattens_at_every_depth_a_single_plane_cannot(shading, z):
    scope = _FlatFieldScope()
    single = acq.ShadingCorrector()
    scope.move_z(50.0)
    single.references['PC'] = single._gain_from_frames(
        [scope.grab_fresh().astype(np.float32)])
    assert _capture(shading, scope)

    scope.move_z(z)
    raw = scope.grab_fresh()
    stacked = _flatness(shading.apply(raw, 'PC', z))
    flat_single = _flatness(single.apply(raw, 'PC'))
    assert stacked < 0.02
    if abs(z - 50.0) > 20:
        assert stacked < flat_single / 2


def test_between_planes_the_maps_are_blended(shading):
    assert _capture(shading, _FlatFieldScope())
    zs, maps = shading.stacks['PC']
    mid = shading.gain_at('PC', 30.0)
    np.testing.assert_allclose(mid, 0.5 * (maps[1] + maps[2]), rtol=1e-6)
    np.testing.assert_array_equal(shading.gain_at('PC', -50.0), maps[0])
    np.testing.assert_array_equal(shading.gain_at('PC', 500.0), maps[-1])


def test_mean_brightness_is_preserved(shading):
    scope = _FlatFieldScope()
    assert _capture(shading, scope)
    scope.move_z(70.0)
    raw = scope.grab_fresh()
    out = shading.apply(raw, 'PC', 70.0)
    assert out.dtype == raw.dtype
    assert abs(out.mean() / raw.mean() - 1.0) < 0.02


def test_stack_persists_and_replaces_a_single_reference(shading, tmp_path):
    scope = _FlatFieldScope()
    assert shading.capture_reference(scope, 'PC', n_frames=1)
    assert (tmp_path / 'PC.npy').exists()
    assert _capture(shading, scope)
    assert not (tmp_path / 'PC.npy').exists()
    assert (tmp_path / 'PC_zstack.npz').exists()

    fresh = acq.ShadingCorrector()
    fresh.load_all()
    assert fresh.has_stack('PC') and 'PC' not in fresh.references
    np.testing.assert_array_equal(fresh.gain_at('PC', 40.0),
                                  shading.gain_at('PC', 40.0))

    # ...and a single capture replaces the stack again.
    assert fresh.capture_reference(scope, 'PC', n_frames=1)
    assert not fresh.has_stack('PC')
    assert not (tmp_path / 'PC_zstack.npz').exists()


def test_a_canceled_capture_keeps_the_old_reference(shading):
    scope = _FlatFieldScope()
    assert shading.capture_reference(scope, 'PC', n_frames=1)
    cancel = acq.threading.Event()
    cancel.set()
    assert not shading.capture_reference_stack(
        scope, 'PC', [0, 20, 40], cancel=cancel)
    assert 'PC' in shading.references and not shading.has_stack('PC')


def test_the_blended_map_is_cached_per_z(shading):
    scope = _FlatFieldScope()
    assert _capture(shading, scope)
    g1 = shading._full_gain('PC', 30.0, (H, W))
    assert shading._full_gain('PC', 30.1, (H, W)) is g1   # within 0.5 um
    assert shading._full_gain('PC', 45.0, (H, W)) is not g1


def test_z_stack_acquisition_applies_the_plane_at_each_slice(shading, tmp_path,
                                                             monkeypatch):
    scope = _FlatFieldScope()
    assert _capture(shading, scope)
    shading.enabled = True
    seen = []
    real = shading.apply
    monkeypatch.setattr(shading, 'apply',
                        lambda img, ch, z=None: seen.append(z) or real(img, ch, z))
    scope.set_channel = lambda ch, cs: None
    scope.grab_settled = lambda settle_s=0: scope.grab_fresh()
    scope.leds_off = lambda: None
    cs = acq.ChannelSettings()
    cs.zstack.range_um, cs.zstack.step_um = 20.0, 10.0
    monkeypatch.setattr(acq, 'save_image', lambda p, img: None)
    acq.run_z_stack(scope, {'PC': cs}, ['PC'], 50.0, shading, str(tmp_path),
                    acq.threading.Event(), use_channel_z_offsets=False)
    assert seen == [60.0, 50.0, 40.0]        # top of the range first


def _pc(range_um=20.0, step_um=10.0, offset=0.0, enabled=True):
    cs = acq.ChannelSettings(z_offset_um=offset)
    cs.zstack.enabled = enabled
    cs.zstack.range_um, cs.zstack.step_um = range_um, step_um
    return cs


def test_shading_planes_are_the_acquisition_planes(shading, tmp_path,
                                                   monkeypatch):
    """Capture at zstack_planes(focus) -> every acquired slice falls exactly
    on a captured plane (no interpolation)."""
    cs = _pc(range_um=40.0, step_um=5.0, offset=3.0)
    focus = 517.4
    planes = acq.zstack_planes(focus, cs, True, force_stack=True)
    assert planes[0] == pytest.approx(500.4) and planes[-1] == pytest.approx(540.4)
    scope = _FlatFieldScope()
    assert shading.capture_reference_stack(scope, 'PC', planes, n_frames=1,
                                           settle_s=0, return_z=focus)
    assert scope.z == pytest.approx(focus)          # back at focus
    np.testing.assert_allclose(shading.stacks['PC'][0], planes)

    shading.enabled = True
    seen = []
    real = shading.apply
    monkeypatch.setattr(shading, 'apply',
                        lambda img, ch, z=None: seen.append(z) or real(img, ch, z))
    scope.set_channel = lambda ch, c: None
    scope.grab_settled = lambda settle_s=0: scope.grab_fresh()
    scope.leds_off = lambda: None
    monkeypatch.setattr(acq, 'save_image', lambda p, img: None)
    acq.run_z_stack(scope, {'PC': cs}, ['PC'], focus, shading, str(tmp_path),
                    acq.threading.Event())
    np.testing.assert_allclose(seen, planes[::-1])   # acquired top first


def test_a_channel_without_a_stack_is_one_plane_at_its_offset():
    planes = acq.zstack_planes(100.0, _pc(offset=-2.5, enabled=False))
    assert list(planes) == [97.5]
    assert len(acq.zstack_planes(100.0, _pc(enabled=False),
                                 force_stack=True)) == 3


def test_an_empty_field_passes_and_labware_left_in_is_flagged(shading):
    scope = _FlatFieldScope()
    assert _capture(shading, scope)
    assert shading.last_structure < shading.STRUCTURE_WARN

    class _WithSample(_FlatFieldScope):
        def grab_fresh(self, timeout_s=3.0):
            img = super().grab_fresh().astype(np.float32)
            rng = np.random.default_rng(3)
            cells = cv2.GaussianBlur(
                (rng.random((H, W)) > 0.995).astype(np.float32), (0, 0), 3)
            img *= 1.0 - 8.0 * cells             # dark cells / debris
            return np.clip(img, 0, 65535).astype(np.uint16)

    assert _capture(shading, _WithSample())
    assert shading.last_structure > shading.STRUCTURE_WARN


# ------------------------------------------- live z-stack: order of work

class _StackScope:
    """Records what a z-stack asks of the microscope."""

    def __init__(self):
        self.log = []
        self.z = None
        self.ch = None

    def move_z(self, z, **_):
        self.z = z
        self.log.append(('z', z))

    def set_channel(self, ch, cs):
        self.ch = ch
        self.log.append(('ch', ch))

    def grab_settled(self, settle_s=0.1):
        self.log.append(('grab', self.ch, self.z))
        return np.zeros((4, 4), np.uint16)

    def leds_off(self):
        self.log.append(('off',))


def _run_planes(tmp_path, order, channels=('BF', 'Green'), cancel_after=None):
    import threading
    scope = _StackScope()
    cancel = threading.Event()
    settings = {ch: acq.ChannelSettings() for ch in ('BF', 'Green', 'Red')}
    seen = []

    def status(msg):
        seen.append(msg)
        if cancel_after is not None and len(seen) >= cancel_after:
            cancel.set()
    saved = acq.run_z_planes(scope, settings, list(channels),
                             [100.0, 110.0, 120.0], None, str(tmp_path),
                             cancel, status_cb=status, return_z_um=105.0,
                             order=order)
    grabs = [(e[1], e[2]) for e in scope.log if e[0] == 'grab']
    return scope, grabs, saved


def test_stack_slice_by_slice_takes_every_colour_at_a_plane(tmp_path):
    scope, grabs, saved = _run_planes(tmp_path, 'slice')
    assert grabs == [('BF', 100.0), ('Green', 100.0), ('BF', 110.0),
                     ('Green', 110.0), ('BF', 120.0), ('Green', 120.0)]
    assert saved == 6
    # one Z move per plane, then back to where it started
    assert [e[1] for e in scope.log if e[0] == 'z'] == \
        [100.0, 110.0, 120.0, 105.0]
    assert scope.log[-1] == ('off',)


def test_stack_colour_by_colour_finishes_one_colour_first(tmp_path):
    scope, grabs, saved = _run_planes(tmp_path, 'channel')
    assert grabs == [('BF', 100.0), ('BF', 110.0), ('BF', 120.0),
                     ('Green', 100.0), ('Green', 110.0), ('Green', 120.0)]
    assert [e[1] for e in scope.log if e[0] == 'ch'] == ['BF', 'Green']
    assert saved == 6


def test_stack_default_order_is_slice_by_slice(tmp_path):
    import inspect
    assert inspect.signature(acq.run_z_planes).parameters['order'].default \
        == 'slice'


def test_stack_images_only_the_colours_asked_for(tmp_path):
    import os
    _, grabs, saved = _run_planes(tmp_path, 'slice', channels=('Green',))
    assert {ch for ch, _ in grabs} == {'Green'}
    names = sorted(os.listdir(tmp_path))
    assert saved == 3 and len(names) == 3
    assert all(n.startswith('zstack_Green_') for n in names)


def test_stack_file_names_are_the_same_in_either_order(tmp_path):
    import os
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir(), b.mkdir()
    _run_planes(a, 'slice')
    _run_planes(b, 'channel')
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    assert 'zstack_BF_000_100.0um.tiff' in os.listdir(a)


def test_stack_stopped_part_way_still_returns_and_switches_off(tmp_path):
    scope, grabs, saved = _run_planes(tmp_path, 'slice', cancel_after=2)
    assert saved == 2 and len(grabs) == 2
    assert scope.log[-2:] == [('z', 105.0), ('off',)]
