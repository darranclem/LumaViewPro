"""Global V-curve / parabolic curve-fit autofocus (acquisition.py).

Checks the curve fitter on synthetic focus curves and the full sweep on a
simulated scope whose image blurs with distance from a known focal plane.
"""

from __future__ import annotations

import numpy as np
import pytest

import acquisition as acq

cv2 = pytest.importorskip('cv2')


# ------------------------------------------------------------ fit only

def _gauss_curve(zs, z0, sigma, amp=100.0, base=5.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = base + amp * np.exp(-(zs - z0) ** 2 / (2 * sigma ** 2))
    return s + rng.normal(0, noise * amp, len(zs))


@pytest.mark.parametrize('z0', [93.0, 101.7, 108.4])
def test_parabola_fit_finds_an_off_grid_peak(z0):
    zs = np.arange(0, 201, 5.0)
    fit = acq.fit_focus_curve(zs, _gauss_curve(zs, z0, sigma=12.0, noise=0.01))
    assert fit is not None and fit.model == 'parabola'
    assert abs(fit.z - z0) < 0.5      # 10x better than the 5 um step


def test_a_single_bad_frame_does_not_drag_the_fit():
    zs = np.arange(0, 201, 5.0)
    s = _gauss_curve(zs, 101.7, sigma=12.0, noise=0.01)
    s[np.searchsorted(zs, 110.0)] *= 0.5   # one vibration-blurred frame
    fit = acq.fit_focus_curve(zs, s)
    assert fit is not None
    assert abs(fit.z - 101.7) < 1.0


def test_a_secondary_peak_is_not_fitted():
    zs = np.arange(0, 201, 5.0)
    s = _gauss_curve(zs, 120.0, sigma=10.0) + \
        _gauss_curve(zs, 40.0, sigma=6.0, amp=40.0, base=0.0)
    fit = acq.fit_focus_curve(zs, s)
    assert fit is not None and abs(fit.z - 120.0) < 0.5


@pytest.mark.parametrize('z0', [97.0, 98.3])
def test_a_flat_topped_peak_is_located(z0):
    # Saturated / plateau peak: not Gaussian, but the flanks are symmetric.
    zs = np.arange(0, 201, 4.0)
    s = np.clip(1.0 - np.abs(zs - z0) / 40.0, 0, None)
    s = np.minimum(s, 0.7) * 100 + 3
    fit = acq.fit_focus_curve(zs, s)
    assert fit is not None and abs(fit.z - z0) < 0.5
    sn = (s - s.min()) / (s.max() - s.min())
    vz, vr2, _ = acq._fit_vcurve(zs, sn)
    assert abs(vz - z0) < 0.2 and vr2 > 0.99


def test_the_vcurve_needs_spare_points():
    zs = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    sn = np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    assert acq._fit_vcurve(zs, sn) is None   # only 4 flank points


def test_too_few_points_on_the_peak_is_not_a_fit():
    zs = np.arange(0, 201, 20.0)
    assert acq.fit_focus_curve(zs, _gauss_curve(zs, 100.0, sigma=3.0)) is None


def test_a_flat_curve_is_not_a_fit():
    zs = np.arange(0, 201, 10.0)
    assert acq.fit_focus_curve(zs, np.full(len(zs), 7.0)) is None


# --------------------------------------------------- simulated sweep

class _SimScope:
    """Stage + camera stand-in: a fixed texture blurred by |z - focus|."""

    def __init__(self, focus_z, blur_per_um=0.25, fluor=False, seed=1):
        rng = np.random.default_rng(seed)
        if fluor:
            tex = np.zeros((480, 640), np.float32)
            ys, xs = rng.integers(0, 480, 300), rng.integers(0, 640, 300)
            tex[ys, xs] = 4000.0
            self.tex = cv2.GaussianBlur(tex, (0, 0), 1.2) + 100.0
        else:
            tex = rng.random((480, 640)).astype(np.float32)
            self.tex = cv2.GaussianBlur(tex, (0, 0), 2.0) * 3000 + 500
        self.focus_z = focus_z
        self.blur_per_um = blur_per_um
        self.z = 0.0
        self.moves = 0
        self.grabs = 0
        self.rng = np.random.default_rng(seed + 1)

    def set_channel(self, channel, settings):
        pass

    def move_z(self, z, wait=True):
        self.z = float(z)
        self.moves += 1

    def grab_fresh(self, timeout_s=3.0):
        self.grabs += 1
        sigma = 0.3 + self.blur_per_um * abs(self.z - self.focus_z)
        img = cv2.GaussianBlur(self.tex, (0, 0), sigma)
        img = self.rng.poisson(np.maximum(img, 0)).astype(np.float32)
        return img


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(acq.time, 'sleep', lambda s: None)


def _run(scope, method, channel='BF', step=10.0, **kw):
    cfg = acq.AutofocusConfig(z_bottom_um=0, z_top_um=200, step_um=step,
                              channel=channel, settle_s=0)
    return acq.autofocus_sweep(scope, cfg, acq.ChannelSettings(),
                               method=method, **kw)


@pytest.mark.parametrize('channel,fluor', [('BF', False), ('GREEN', True)])
def test_global_fit_focuses_in_one_pass(channel, fluor):
    scope = _SimScope(focus_z=103.3, blur_per_um=0.12, fluor=fluor)
    best, ok = _run(scope, 'vcurve', channel=channel)
    assert ok
    assert abs(best - 103.3) < 1.5
    assert abs(scope.z - best) < 1e-9
    assert scope.grabs == 21          # coarse pass only, no fine pass


def test_global_fit_is_not_slower_than_two_pass_and_as_accurate():
    a = _SimScope(focus_z=103.3, blur_per_um=0.12)
    b = _SimScope(focus_z=103.3, blur_per_um=0.12)
    best_v, ok_v = _run(a, 'vcurve')
    best_t, ok_t = _run(b, 'two_pass')
    assert ok_v and ok_t
    assert a.grabs < b.grabs
    assert abs(best_v - 103.3) <= abs(best_t - 103.3) + 0.5


def test_sparse_coarse_curve_falls_back_to_a_fitted_fine_pass():
    # Shallow depth of field relative to the 20 um interval.
    scope = _SimScope(focus_z=96.6, blur_per_um=1.5)
    best, ok = _run(scope, 'vcurve', step=20.0)
    assert ok
    assert scope.grabs > 11           # a fine pass was needed
    assert abs(best - 96.6) < 1.5


def test_two_pass_is_unchanged_by_default():
    scope = _SimScope(focus_z=103.3, blur_per_um=0.12)
    best, ok = acq.autofocus_sweep(
        scope, acq.AutofocusConfig(0, 200, 10.0, 'BF', 0),
        acq.ChannelSettings())
    assert ok and abs(best - 103.3) < 2.0
    assert scope.grabs > 21           # coarse + fine
