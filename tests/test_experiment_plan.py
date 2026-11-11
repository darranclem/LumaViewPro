"""Experiment planning (experiment.py): imaging sites per well, focus map
points, the interpolated focus surface and the file naming."""

from __future__ import annotations

import math
import random

import pytest

import experiment as ex
from acquisition import FocusSurface

FOV = (1.0, 0.75)      # mm


# ------------------------------------------------------------- naming

def test_file_name_order_is_well_site_z_colour_timepoint():
    assert ex.file_name('B3', 'TA1', 1, 'GREEN', 1) == \
        'B3_TA1_Z001_GREEN_T0001.tiff'
    assert ex.file_name('H12', 'P2', 12, 'BF', 345) == \
        'H12_P2_Z012_BF_T0345.tiff'


def test_file_name_without_a_well_starts_at_the_site():
    assert ex.file_name(None, 'P1', 1, 'RED', 2) == 'P1_Z001_RED_T0002.tiff'


def test_tiles_are_lettered_by_line_and_numbered_along_it():
    labels = sorted(l for l, _, _ in ex.tile_grid(2, 3, *FOV, 10))
    assert labels == ['TA1', 'TA2', 'TA3', 'TB1', 'TB2', 'TB3']
    assert ex.row_letters(25) == 'Z' and ex.row_letters(26) == 'AA'


def test_positions_without_a_designation_get_a_well_and_the_next_p_number():
    pos = [{'name': 'a', 'k': 'A1'}, {'name': 'b', 'k': 'A1'},
           {'name': 'c', 'k': 'A2', 'well': 'A2', 'site': 'P2'},
           {'name': 'd', 'k': 'A2'}]
    out = ex.assign_designations(pos, lambda p: p['k'])
    assert [(p['well'], p['site']) for p in out] == [
        ('A1', 'P1'), ('A1', 'P2'), ('A2', 'P2'), ('A2', 'P1')]
    assert 'site' not in pos[0]                 # input untouched


def test_no_plate_means_no_well_in_any_name():
    pos = [{'name': 'a', 'well': 'A1', 'site': 'TA1'}, {'name': 'b'}]
    out = ex.assign_designations(pos, None)
    assert [(p['well'], p['site']) for p in out] == [(None, 'TA1'),
                                                     (None, 'P1')]


# ---------------------------------------------------------- tile grids

def test_tile_spacing_follows_the_overlap():
    tiles = {l: (x, y) for l, x, y in ex.tile_grid(2, 2, *FOV, 20)}
    assert tiles['TA2'][0] - tiles['TA1'][0] == pytest.approx(0.8)
    assert tiles['TB1'][1] - tiles['TA1'][1] == pytest.approx(0.6)
    # centered on the well
    assert sum(x for x, _ in tiles.values()) == pytest.approx(0.0)


def test_tiles_are_visited_in_serpentine_order():
    order = [l for l, _, _ in ex.tile_grid(2, 3, *FOV, 10)]
    assert order == ['TA1', 'TA2', 'TA3', 'TB3', 'TB2', 'TB1']


def _covered(tiles, x, y):
    return any(abs(x - tx) <= FOV[0] / 2 + 1e-9 and
               abs(y - ty) <= FOV[1] / 2 + 1e-9 for _, tx, ty in tiles)


@pytest.mark.parametrize('overlap', [0, 10, 25])
def test_full_coverage_images_every_part_of_a_round_well(overlap):
    d = 6.4
    tiles = ex.coverage_tiles(d, d, 'round', *FOV, overlap, 100)
    rng = random.Random(1)
    for _ in range(4000):
        r, a = d / 2 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        assert _covered(tiles, r * math.cos(a), r * math.sin(a))
    # rim points too: the grid overfills the well
    for k in range(360):
        a = math.radians(k)
        assert _covered(tiles, d / 2 * math.cos(a), d / 2 * math.sin(a))


def test_full_coverage_of_a_round_well_leaves_out_the_empty_grid_corners():
    tiles = ex.coverage_tiles(6.4, 6.4, 'round', *FOV, 10, 100)
    labels = {l for l, _, _ in tiles}
    assert 'TA1' not in labels                  # corner of the grid
    square = ex.coverage_tiles(6.4, 6.4, 'square', *FOV, 10, 100)
    assert len(square) > len(tiles) and 'TA1' in {l for l, _, _ in square}


def test_lower_coverage_needs_fewer_tiles_and_stays_centered():
    full = ex.coverage_tiles(6.4, 6.4, 'round', *FOV, 10, 100)
    half = ex.coverage_tiles(6.4, 6.4, 'round', *FOV, 10, 50)
    tiny = ex.coverage_tiles(6.4, 6.4, 'round', *FOV, 10, 1)
    assert len(full) > len(half) > len(tiny) == 1
    assert tiny[0][1:] == (0.0, 0.0)


# -------------------------------------------------------- random points

def test_random_points_stay_inside_the_well_clear_of_the_rim():
    pts = ex.random_points(20, 6.4, 6.4, 'round', *FOV, random.Random(3))
    assert [l for l, _, _ in pts] == [f'P{i + 1}' for i in range(20)]
    for _, x, y in pts:
        assert (x / (3.2 - 0.5)) ** 2 + (y / (3.2 - 0.375)) ** 2 <= 1 + 1e-9


def test_random_points_do_not_overlap_when_there_is_room():
    pts = ex.random_points(5, 15.0, 15.0, 'square', *FOV, random.Random(7))
    for i, (_, x0, y0) in enumerate(pts):
        for _, x1, y1 in pts[i + 1:]:
            assert abs(x0 - x1) >= FOV[0] or abs(y0 - y1) >= FOV[1]


def test_random_modes_differ_per_well_or_repeat_the_same_pattern():
    kw = dict(well_x_mm=6.4, well_y_mm=6.4, shape='round',
              fov_x_mm=FOV[0], fov_y_mm=FOV[1], n_random=4, seed=5)
    wells = [(0, 0), (0, 1)]
    same = ex.plan_sites('random_same', wells, **kw)
    diff = ex.plan_sites('random', wells, **kw)
    assert same[(0, 0)] == same[(0, 1)]
    assert diff[(0, 0)] != diff[(0, 1)]
    assert all(len(v) == 4 for v in (*same.values(), *diff.values()))


def test_single_mode_is_one_position_at_the_center():
    out = ex.plan_sites('single', [(2, 2)], well_x_mm=6.4, well_y_mm=6.4,
                        shape='round', fov_x_mm=1, fov_y_mm=1)
    assert out == {(2, 2): [('P1', 0.0, 0.0)]}


# --------------------------------------------------------- focus points

def test_corner_mode_is_four_corners_and_the_middle_of_the_selection():
    wells = [(r, c) for r in range(1, 6) for c in range(2, 9)]
    pts = ex.focus_points('corners', wells, well_x_mm=6.4, well_y_mm=6.4)
    assert [w for w, _, _ in pts] == [(1, 2), (1, 8), (5, 8), (5, 2), (3, 5)]


def test_corner_mode_on_one_row_or_one_well_has_no_duplicates():
    row = ex.focus_points('corners', [(0, c) for c in range(5)],
                          well_x_mm=6.4, well_y_mm=6.4)
    assert [w for w, _, _ in row] == [(0, 0), (0, 4), (0, 2)]
    one = ex.focus_points('corners', [(3, 3)], well_x_mm=6.4, well_y_mm=6.4)
    assert [w for w, _, _ in one] == [(3, 3)]


def test_per_well_and_multi_point_modes():
    wells = [(0, 0), (0, 1)]
    assert len(ex.focus_points('per_well', wells, well_x_mm=6.4,
                               well_y_mm=6.4)) == 2
    multi = ex.focus_points('multi', wells, well_x_mm=6.4, well_y_mm=6.4,
                            n_per_well=4)
    assert len(multi) == 8
    for _, dx, dy in multi:
        assert math.hypot(dx, dy) == pytest.approx(0.3 * 6.4)


# -------------------------------------------------------- focus surface

CORNERS = [(0, 0, 10.0), (100, 0, 30.0), (100, 60, 55.0), (0, 60, 20.0),
           (50, 30, 12.0)]


def test_the_surface_passes_through_every_focus_point():
    for x, y, z in CORNERS:
        assert ex.interpolate_offsets(CORNERS, x, y) == pytest.approx(z)


def test_a_tilted_plane_is_interpolated_and_extrapolated_exactly():
    def plane(x, y):
        return 0.2 * x - 0.1 * y + 5.0
    pts = [(x, y, plane(x, y)) for x, y, _ in CORNERS]
    for x, y in [(25, 15), (75, 50), (-40, -20), (300, 200)]:
        assert ex.interpolate_offsets(pts, x, y) == pytest.approx(plane(x, y))


def test_between_points_the_surface_follows_the_local_dip():
    # the center point sits well below the plane through the corners
    mid = ex.interpolate_offsets(CORNERS, 45, 28)
    flat = ex.interpolate_offsets(CORNERS[:4], 45, 28)
    assert mid < flat - 5.0


def test_few_points_fall_back_to_nearest_or_none():
    assert ex.interpolate_offsets([], 1, 1) is None
    assert ex.interpolate_offsets([(0, 0, 4.0), (10, 0, 9.0)], 8, 0) == 9.0


def test_autofocus_drift_moves_the_whole_surface_by_the_same_amount():
    s = FocusSurface()
    s.set_anchor(50, 30, 'BF')
    s.set_reference_z(1000.0)
    for x, y, off in CORNERS:
        s.add_point(x, y, 1000.0 + off)
    probes = [(0, 0), (50, 30), (20, 45), (250, 10)]
    before = [s.z_at(x, y) for x, y in probes]
    s.update_cycle(1007.5)                    # AF point found 7.5 um higher
    after = [s.z_at(x, y) for x, y in probes]
    assert [a - b for a, b in zip(after, before)] == \
        pytest.approx([7.5] * len(probes))
