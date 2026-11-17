"""Tile scans stitched by Quark (quark_stitch + acquisition.run_tile_scan).

A simulated scope looks at a fixed scene through a field that is darkened from
top to bottom (the shading the owner's scans show). The scan must leave a
record of its tile positions, and, where Quark's module is installed, a mosaic
with the shading removed and no seams; without the module it must still stitch.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pytest

import acquisition as acq
import quark_stitch

tifffile = pytest.importorskip('tifffile')

TW, TH = 160, 120          # tile size, px
UM = 2.0                   # um per px
SHADE = 0.3                # darkening from the top row to the bottom row


def _scene(x_px, y_px):
    """Plain background with sample on it that differs from tile to tile."""
    return (20000.0 + 400.0 * (((x_px // 9) + (y_px // 7)) % 2)
            + 1200.0 * np.sin(x_px / 53.0) * np.cos(y_px / 31.0))


class _Scope:
    """Stage + camera: a frame is the scene under the field, shaded."""

    def __init__(self):
        self.x = self.y = 0.0
        self.moves = []

    def move_xy(self, x, y, wait=True):
        self.x, self.y = float(x), float(y)
        self.moves.append((self.x, self.y))

    def move_z(self, z, wait=True):
        pass

    def set_channel(self, ch, cs=None):
        pass

    def leds_off(self):
        pass

    def get_pos(self):
        return self.x, self.y, 0.0

    def grab_settled(self, settle_s=0.0, **_):
        x0 = int(round(self.x / UM))
        y0 = int(round(self.y / UM))
        yy, xx = np.mgrid[0:TH, 0:TW]
        shade = 1.0 - SHADE * yy / (TH - 1)
        return np.clip(_scene(xx + x0, yy + y0) * shade, 0, 65535).astype(np.uint16)


def _scan(tmp_path, rows=3, cols=3):
    hw = _Scope()
    settings = {'BF': acq.ChannelSettings(enabled=True)}
    # Tile centres 20 % overlapped, well inside the stage.
    positions = acq.tile_positions_grid(20000.0, 20000.0, TW * UM, TH * UM,
                                        rows, cols, overlap_pct=20.0)
    out = str(tmp_path / 'tiles')
    os.makedirs(out)
    said = []
    paths = acq.run_tile_scan(hw, positions, settings, None, None, UM, out,
                              threading.Event(), status_cb=said.append,
                              settle_s=0.0, total=len(positions))
    return out, positions, paths, said


def _shading_left(mosaic, positions):
    """How far the mosaic is from the scene, up to one overall factor."""
    x0 = int(round(min(p[0] for p in positions) / UM))
    y0 = int(round(min(p[1] for p in positions) / UM))
    h, w = mosaic.shape
    yy, xx = np.mgrid[0:h:3, 0:w:3]
    ratio = mosaic[::3, ::3].astype(np.float64) / _scene(xx + x0, yy + y0)
    ratio = ratio[mosaic[::3, ::3] > 0]
    return float(np.std(ratio / ratio.mean()))


def test_a_tile_scan_records_where_every_tile_was(tmp_path):
    out, positions, paths, _ = _scan(tmp_path)
    with open(os.path.join(out, quark_stitch.TILE_SCAN_SIDECAR), encoding='utf-8') as f:
        record = json.load(f)
    assert record['format'] == 'lvp-tile-scan-1'
    assert record['um_per_px'] == UM and record['channels'] == ['BF']
    assert len(record['positions']) == len(positions) == 9
    for i, (x, y) in enumerate(positions):
        p = record['positions'][i]
        assert (p['index'], p['x_um'], p['y_um']) == (i, x, y)
        assert os.path.exists(os.path.join(out, f'tile_{i:04d}_BF.tiff'))
    # A mosaic is written whether or not Quark is installed.
    assert os.path.exists(paths['BF'])


def test_without_quark_the_scan_still_stitches_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(quark_stitch, '_load', lambda: None)
    monkeypatch.setattr(quark_stitch, '_why', 'not installed (test)')
    assert not quark_stitch.available()
    with pytest.raises(quark_stitch.QuarkUnavailable):
        quark_stitch.scan(str(tmp_path))
    out, positions, paths, said = _scan(tmp_path)
    mosaic = tifffile.imread(paths['BF'])
    assert mosaic.shape[0] > TH and mosaic.shape[1] > TW
    assert 'Quark stitching not installed' in said[-1]
    # The plain stitch keeps the shading: that is what Quark is for.
    assert _shading_left(mosaic, positions) > 0.05


@pytest.mark.skipif(not quark_stitch.available(), reason=quark_stitch.why_unavailable())
def test_with_quark_the_mosaic_has_its_shading_removed(tmp_path):
    out, positions, paths, said = _scan(tmp_path)
    mosaic = tifffile.imread(paths['BF'])
    assert mosaic.dtype == np.uint16 and mosaic.ndim == 2
    # The size the stage positions give (tiles 20 % overlapped).
    step_x, step_y = round(TW * 0.8), round(TH * 0.8)
    assert abs(mosaic.shape[1] - (2 * step_x + TW)) <= 2
    assert abs(mosaic.shape[0] - (2 * step_y + TH)) <= 2
    left = _shading_left(mosaic, positions)
    assert left < 0.015, f'shading left in the mosaic: {left:.4f}'
    assert 'shading measured from' in said[-1]

    # The same folder, read again later.
    listed = quark_stitch.scan(out)
    assert listed['wells'][0]['name'] == 'scan' and listed['wells'][0]['tiles'] == 9
    assert listed['pixel_um_stated'] == UM
    gain, info = quark_stitch.flat_field(out, 0)
    assert gain.shape == (TH, TW) and abs(float(gain.mean()) - 1.0) < 1e-3
    # Top of the field over the bottom: 1 / (1 - SHADE).
    assert abs(gain[0].mean() / gain[-1].mean() - 1.0 / (1.0 - SHADE)) < 0.04
    again, info = quark_stitch.stitch_well(out, 'scan')
    assert again.shape[0] == 1 and np.array_equal(again[0], mosaic)
    assert info['pixel_um'] == UM and info['shading'][0]['corrected']
    # Written as an OME-TIFF beside the tiles, not over them.
    path, _ = quark_stitch.export_well(out)
    assert os.path.dirname(path) == os.path.join(out, 'quark') and os.path.exists(path)


@pytest.mark.skipif(not quark_stitch.available(), reason=quark_stitch.why_unavailable())
def test_shading_from_tiles_in_memory():
    rng = np.random.default_rng(3)
    yy = np.linspace(1.0, 1.0 - SHADE, TH, dtype=np.float32)[:, None] * np.ones((1, TW), np.float32)
    tiles = [np.clip((3000 + rng.normal(0, 8, (TH, TW))) * yy * (0.9 + 0.03 * i), 0, 65535).astype(np.uint16)
             for i in range(8)]
    gain, info = quark_stitch.flat_field_from_tiles(tiles)
    assert info['fields_used'] == 8
    assert np.sqrt(np.mean((gain / (yy / yy.mean()) - 1.0) ** 2)) < 0.005
    fixed = quark_stitch.apply_flat_field(tiles[0], gain)
    assert fixed.dtype == np.uint16 and fixed.shape == tiles[0].shape
    assert abs(fixed[:6].mean() / fixed[-6:].mean() - 1.0) < 0.01
    with pytest.raises(RuntimeError):
        quark_stitch.flat_field_from_tiles(tiles[:2])      # too few to measure from

@pytest.mark.skipif(not quark_stitch.available(), reason=quark_stitch.why_unavailable())
def test_shading_measured_from_a_scan_becomes_the_reference_for_new_frames(tmp_path, monkeypatch):
    """The owner's case: tiles darker from top to bottom. The pattern measured from a
    finished scan is stored as the channel's shading reference, and a frame taken afterwards
    (another place, same optics) comes out even."""
    monkeypatch.setattr(acq, 'SHADING_DIR', str(tmp_path / 'shading'))
    out, positions, paths, said = _scan(tmp_path)
    assert quark_stitch.shading_was_applied(out) is False
    measured = quark_stitch.measure_shading(out)
    gain, info = measured['BF']
    assert gain is not None and 'measured from images' in quark_stitch.describe_shading(info)
    corrector = acq.ShadingCorrector()
    corrector.set_measured('BF', gain, quark_stitch.describe_shading(info))
    assert corrector.has_reference('BF') and 'Quark' in corrector.describe('BF')
    # A new frame somewhere else on the sample.
    hw = _Scope()
    hw.move_xy(23456.0, 21234.0)
    raw = hw.grab_settled()
    yy, xx = np.mgrid[0:TH, 0:TW]
    truth = _scene(xx + int(round(hw.x / UM)), yy + int(round(hw.y / UM)))
    uneven = lambda img: float(np.std((img / truth) / (img / truth).mean()))
    fixed = corrector.apply(raw, 'BF')
    assert fixed.dtype == np.uint16
    assert uneven(raw.astype(np.float64)) > 0.07
    assert uneven(fixed.astype(np.float64)) < 0.015, uneven(fixed.astype(np.float64))
    # It is kept, with where it came from, for the next start.
    again = acq.ShadingCorrector()
    again.load_all()
    assert again.has_reference('BF') and 'measured from images' in again.describe('BF')
    # A captured reference replaces it, note and all.
    class _Flat:
        def grab_fresh(self):
            return np.full((TH, TW), 1000, np.uint16)
    assert again.capture_reference(_Flat(), 'BF', n_frames=2)
    assert again.describe('BF') == 'BF: single plane'


@pytest.mark.skipif(not quark_stitch.available(), reason=quark_stitch.why_unavailable())
def test_a_folders_tiles_are_written_shading_corrected_beside_the_originals(tmp_path):
    out, positions, paths, said = _scan(tmp_path)
    before = {name: os.path.getmtime(os.path.join(out, name)) for name in os.listdir(out)}
    record = quark_stitch.correct_folder(out)
    target = os.path.join(out, quark_stitch.SHADING_OUT)
    assert record['out_dir'] == target and record['channels']['BF']['corrected'] == 9
    assert not record['channels']['BF']['skipped']
    # The originals are untouched.
    assert all(os.path.getmtime(os.path.join(out, name)) == t for name, t in before.items())
    # Every corrected tile is even from top to bottom (the raw ones are 30 % darker below).
    for k, (x, y) in enumerate(positions):
        name = f'tile_{k:04d}_BF.tiff'
        raw = tifffile.imread(os.path.join(out, name)).astype(np.float64)
        fixed = tifffile.imread(os.path.join(target, name)).astype(np.float64)
        yy, xx = np.mgrid[0:TH, 0:TW]
        truth = _scene(xx + int(round(x / UM)), yy + int(round(y / UM)))
        assert fixed.shape == raw.shape
        assert raw[:8].mean() / raw[-8:].mean() > 1.25
        ratio = fixed / truth
        assert abs(ratio[:8].mean() / ratio[-8:].mean() - 1.0) < 0.03, name
    # What was done is written down, and the corrected copy is a tile scan of its own.
    with open(os.path.join(target, 'shading.json'), encoding='utf-8') as f:
        assert json.load(f)['channels']['BF']['corrected'] == 9
    assert os.path.exists(os.path.join(target, 'gain_BF.npy'))
    assert quark_stitch.shading_was_applied(target) is True
    assert quark_stitch.scan(target)['wells'][0]['tiles'] == 9
    # A second run does not take its own output for tiles.
    assert quark_stitch.correct_folder(out)['channels']['BF']['corrected'] == 9
