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
        'B3_TA1_Z001_GREEN_t0001.tiff'
    assert ex.file_name('H12', 'P2', 12, 'BF', 345) == \
        'H12_P2_Z012_BF_t0345.tiff'


def test_file_name_without_a_well_starts_at_the_site():
    assert ex.file_name(None, 'P1', 1, 'RED', 2) == 'P1_Z001_RED_t0002.tiff'


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
    pts = ex.random_points(20, 6.4, 6.4, 'round', *FOV, random.Random(3),
                           allow_overlap=True)
    assert [l for l, _, _ in pts] == [f'P{i + 1}' for i in range(20)]
    for _, x, y in pts:
        assert (x / (3.2 - 0.5)) ** 2 + (y / (3.2 - 0.375)) ** 2 <= 1 + 1e-9


def test_random_points_do_not_overlap_when_there_is_room():
    pts = ex.random_points(5, 15.0, 15.0, 'square', *FOV, random.Random(7))
    for i, (_, x0, y0) in enumerate(pts):
        for _, x1, y1 in pts[i + 1:]:
            assert abs(x0 - x1) >= FOV[0] or abs(y0 - y1) >= FOV[1]


def _overlaps(pts):
    return any(abs(x0 - x1) < FOV[0] and abs(y0 - y1) < FOV[1]
               for i, (_, x0, y0) in enumerate(pts)
               for _, x1, y1 in pts[i + 1:])


def test_a_full_well_gets_fewer_points_rather_than_overlapping_ones():
    pts = ex.random_points(200, 6.4, 6.4, 'round', *FOV, random.Random(2))
    assert 5 < len(pts) < 200
    assert not _overlaps(pts)
    assert [l for l, _, _ in pts] == [f'P{i + 1}' for i in range(len(pts))]


def test_the_overlap_check_can_be_switched_off():
    pts = ex.random_points(200, 6.4, 6.4, 'round', *FOV, random.Random(2),
                           allow_overlap=True)
    assert len(pts) == 200 and _overlaps(pts)


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


# -------------------------------------------------------------- z-stacks

def test_a_stack_runs_from_the_top_of_the_range_down_through_the_center():
    assert ex.stack_offsets(100, 25) == [50, 25, 0, -25, -50]


def test_a_stack_always_includes_the_center_and_never_falls_short():
    offs = ex.stack_offsets(100, 30)
    assert 0 in offs
    assert offs == [-o for o in reversed(offs)]          # symmetric
    assert offs[0] >= 50 and offs[-1] <= -50             # covers the range
    assert ex.stack_offsets(0, 5) == [0]


def test_optical_section_shrinks_with_na_and_grows_with_wavelength():
    lo = ex.optical_section_um(0.13, 550, 4, 2.0)
    hi = ex.optical_section_um(0.65, 550, 40, 2.0)
    assert lo > 10 * hi
    assert ex.optical_section_um(0.3, 610, 10, 2.0) > \
        ex.optical_section_um(0.3, 450, 10, 2.0)
    # 10x / 0.30 at 520 nm, 2 um pixels: 0.52/0.09 + 2/(10*0.3)
    assert ex.optical_section_um(0.3, 520, 10, 2.0) == \
        pytest.approx(0.52 / 0.09 + 2.0 / 3.0)


def test_step_is_the_section_over_the_sampling_rate_unless_typed_in():
    assert ex.stack_step_um(6.0) == 6.0                          # 1x
    assert ex.stack_step_um(6.9, ex.NYQUIST_SAMPLING) == pytest.approx(3.0)
    assert ex.stack_step_um(6.0, 2.3, override_um=1.5) == 1.5
    assert ex.stack_step_um(6.0, 2.0, override_um=0) == 3.0


def test_acquisition_planes_are_centered_on_focus_plus_the_channel_offset():
    import acquisition as acq
    cs = acq.ChannelSettings(z_offset_um=4.0)
    cs.zstack.enabled = True
    cs.zstack.range_um, cs.zstack.step_um = 100.0, 25.0
    planes = acq.zstack_planes(1000.0, cs, True)
    assert list(planes) == [954.0, 979.0, 1004.0, 1029.0, 1054.0]
    cs.zstack.enabled = False
    assert list(acq.zstack_planes(1000.0, cs, True)) == [1004.0]


# ----------------------------------------------- plate calibration moves

def test_after_the_first_mark_the_next_rim_points_are_estimated_from_it():
    import acquisition as acq
    import labware_defs
    lw = labware_defs.LabwareLibrary().get('96-well plate')
    cal = acq.PlateCalibration(lw)
    assert cal.estimate_stage_for(cal.targets[1], None) is None   # no clue yet
    cal.mark_current(30000.0, 20000.0)          # A1 12 o'clock
    r = lw.well_size_x_mm / 2 * 1000
    # 3 o'clock: half a well right and down (stage X reversed here)
    assert cal.estimate_stage_for(cal.targets[1], None, (-1.0, 1.0)) == \
        pytest.approx((30000.0 - r, 20000.0 + r))
    # 6 o'clock: a full well diameter down
    assert cal.estimate_stage_for(cal.targets[2], None, (-1.0, 1.0)) == \
        pytest.approx((30000.0, 20000.0 + 2 * r))
    # a marked target keeps its own mark
    assert cal.estimate_stage_for(cal.targets[0], None) == (30000.0, 20000.0)


# ------------------------------------------- plate calibration accuracy

def _marks(lw, a1=(9032.0, 7480.0), well_mm=4.5, pitch=(9.0, 9.0),
           rot_deg=0.0, last_row=None):
    """Stage marks for a real plate whose wells are ``well_mm`` across,
    whatever the labware definition says."""
    import numpy as np
    t = math.radians(rot_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    r = well_mm / 2.0
    rows = (lw.rows - 1) if last_row is None else last_row
    pts = [(0, -r), (r, 0), (0, r), (-r, 0),
           ((lw.cols - 1) * pitch[0] - r, 0), (0, rows * pitch[1] - r)]
    return [tuple(np.asarray(a1) + R @ (np.asarray(p) * 1000.0))
            for p in pts]


def _calibrated(lw, **kw):
    import acquisition as acq
    cal = acq.PlateCalibration(lw)
    for i, m in enumerate(_marks(lw, **kw)):
        cal.index = i
        cal.mark_current(*m)
    assert cal.solve()
    return cal


def _lw96():
    import labware_defs
    return labware_defs.LabwareLibrary().get('96-well plate')


def test_well_centers_are_exact_when_wells_are_smaller_than_the_definition():
    lw = _lw96()                      # definition says 6.4 mm wells
    cal = _calibrated(lw, well_mm=4.5, rot_deg=0.4)
    t = math.radians(0.4)
    for row, col in ((0, 0), (0, 11), (7, 0), (7, 11), (3, 6)):
        x, y = cal.plate_mm_to_stage_um(lw.a1_offset_x_mm + col * 9.0,
                                        lw.a1_offset_y_mm + row * 9.0)
        ex = 9032.0 + (col * math.cos(t) - row * math.sin(t)) * 9000.0
        ey = 7480.0 + (col * math.sin(t) + row * math.cos(t)) * 9000.0
        assert (x, y) == pytest.approx((ex, ey), abs=1.0)
    assert cal.measured_well_diameter_mm == pytest.approx(4.5, abs=1e-6)
    assert cal.measured_well_x_mm == pytest.approx(4.5, abs=1e-6)
    assert cal.measured_spacing_x_mm == pytest.approx(9.0, abs=1e-6)
    assert cal.measured_spacing_y_mm == pytest.approx(9.0, abs=1e-6)
    assert cal.rotation_deg == pytest.approx(0.4, abs=1e-6)
    assert cal.scale_ok() and cal.scale_problems() == []


def test_a_slightly_different_pitch_is_absorbed_not_rejected():
    lw = _lw96()
    cal = _calibrated(lw, pitch=(9.02, 8.97))
    x, y = cal.plate_mm_to_stage_um(lw.a1_offset_x_mm + 11 * 9.0,
                                    lw.a1_offset_y_mm + 7 * 9.0)
    assert (x, y) == pytest.approx((9032.0 + 11 * 9020.0,
                                    7480.0 + 7 * 8970.0), abs=1.0)
    assert cal.measured_spacing_x_mm == pytest.approx(9.02, abs=1e-6)
    assert cal.scale_ok()


def test_a_bespoke_plate_is_mapped_from_the_measurements_alone():
    # 96-well class, but 10.5 mm pitch and 3 mm wells: nothing like the
    # 9.0 / 6.4 mm definition. Size and spacing come from the marks.
    lw = _lw96()
    cal = _calibrated(lw, well_mm=3.0, pitch=(10.5, 10.5), rot_deg=-0.7)
    assert cal.scale_ok() and cal.scale_problems() == []
    t = math.radians(-0.7)
    for row, col in ((0, 0), (0, 11), (7, 0), (7, 11), (4, 5)):
        x, y = cal.plate_mm_to_stage_um(lw.a1_offset_x_mm + col * 9.0,
                                        lw.a1_offset_y_mm + row * 9.0)
        ex = 9032.0 + (col * math.cos(t) - row * math.sin(t)) * 10500.0
        ey = 7480.0 + (col * math.sin(t) + row * math.cos(t)) * 10500.0
        assert (x, y) == pytest.approx((ex, ey), abs=1.0)
    assert cal.measured_well_diameter_mm == pytest.approx(3.0, abs=1e-6)
    assert cal.measured_spacing_x_mm == pytest.approx(10.5, abs=1e-6)
    assert cal.measured_spacing_y_mm == pytest.approx(10.5, abs=1e-6)
    # the map draws the well in the definition's units: 3 mm of a 10.5 mm
    # pitch is the same fraction of the 9 mm pitch it is drawn with
    assert cal.measured_well_x_mm / 9.0 == pytest.approx(3.0 / 10.5)


def test_a_far_mark_next_to_a1_is_refused_as_overlapping_wells():
    lw = _lw96()
    marks = _marks(lw)
    marks[5] = (marks[0][0], marks[0][1] + 500.0)
    import acquisition as acq
    cal = acq.PlateCalibration(lw)
    for i, m in enumerate(marks):
        cal.index = i
        cal.mark_current(*m)
    assert cal.solve()
    (idx, text), = cal.scale_problems()
    assert idx == 5 and 'overlap' in text and not cal.scale_ok()


def test_after_a1_the_stage_is_sent_to_the_true_far_wells():
    import acquisition as acq
    lw = _lw96()
    marks = _marks(lw, well_mm=4.5)
    cal = acq.PlateCalibration(lw)
    for i in range(4):
        cal.index = i
        cal.mark_current(*marks[i])
    assert cal.provisional_solve()
    # not scaled by 4.5 / 6.4 (which sent the stage to row F for H1)
    assert cal.estimate_stage_for(cal.targets[4], None) == \
        pytest.approx(marks[4], abs=1.0)
    assert cal.estimate_stage_for(cal.targets[5], None) == \
        pytest.approx(marks[5], abs=1.0)


def test_a_far_mark_on_the_wrong_row_is_named():
    lw = _lw96()
    cal = _calibrated(lw, last_row=5)           # F1 confirmed as H1
    assert not cal.scale_ok()
    (idx, text), = cal.scale_problems()
    assert idx == 5
    assert '5.0 wells' in text and 'F1' in text and 'H1' in text


def test_the_marks_logged_on_13_november_are_recognised_as_row_f():
    import acquisition as acq
    cal = acq.PlateCalibration(_lw96())
    for i, m in enumerate([(9022, 5223), (11292, 7471), (9042, 9736),
                           (6774, 7535), (105931, 7366), (9170, 50256)]):
        cal.index = i
        cal.mark_current(*m)
    assert cal.solve()
    (idx, text), = cal.scale_problems()
    assert idx == 5 and 'F1' in text


# ------------------------------------------ do the moves fit the lens

def _tiles(step_um, n=3, well='B2'):
    return [{'well': well, 'site': ex.tile_label(r, c),
             'x_um': 1000.0 + c * step_um, 'y_um': 2000.0 + r * step_um}
            for r in range(n) for c in range(n)]


def test_tile_step_is_the_distance_between_neighbouring_tiles():
    pos = _tiles(900.0) + _tiles(900.0, well='C5') + \
        [{'well': 'D1', 'site': 'P1', 'x_um': 0.0, 'y_um': 0.0}]
    assert ex.tile_step_um(pos) == pytest.approx(900.0)
    assert ex.tile_step_um([p for p in pos if p['site'] == 'P1']) is None


def test_tiles_laid_out_for_the_same_lens_fit():
    verdict, text = ex.lens_fit(900.0, (1000.0, 1000.0))
    assert verdict == 'ok' and '10% overlap' in text


def test_tiles_for_10x_leave_gaps_under_a_20x_lens():
    # 10x field 1.24 mm, 10% overlap; the 20x field is half that
    verdict, text = ex.lens_fit(1116.0, (620.0, 620.0))
    assert verdict == 'gaps' and 'GAPS of 80%' in text


def test_tiles_for_20x_are_far_too_dense_under_a_10x_lens():
    verdict, _ = ex.lens_fit(558.0, (1240.0, 1240.0))
    assert verdict == 'dense'


def test_single_fields_have_no_tile_step_to_check():
    assert ex.lens_fit(None, (1000.0, 1000.0))[0] == 'none'


def test_measured_image_scale_confirms_or_refutes_the_selected_lens():
    assert ex.lens_scale_check(None, 0.65)[0] == 'unknown'
    assert ex.lens_scale_check(0.63, 0.65)[0] == 'ok'
    verdict, text = ex.lens_scale_check(0.325, 0.65)   # a 20x under "10x"
    assert verdict == 'wrong' and '2.00 times' in text


# ------------------------------------- editing positions inside a well

def _well_positions():
    """B2 with a 2 x 2 tile scan, C3 with one center position."""
    pos = [{'name': f'B2_{ex.tile_label(r, c)}', 'well': 'B2',
            'site': ex.tile_label(r, c), 'x_um': 1000.0 + c * 900.0,
            'y_um': 2000.0 + r * 900.0, 'z_um': 50.0}
           for r in range(2) for c in range(2)]
    pos.append({'name': 'C3', 'well': 'C3', 'site': 'P1',
                'x_um': 9000.0, 'y_um': 9000.0, 'z_um': 60.0})
    return pos


def test_a_single_position_moves_and_keeps_its_designation():
    pos = _well_positions()
    new, what = ex.move_in_well(pos, 'C3', 9400.0, 8700.0)
    assert what == 'C3'
    p = new[-1]
    assert (p['x_um'], p['y_um']) == (9400.0, 8700.0)
    assert (p['name'], p['well'], p['site'], p['z_um']) == \
        ('C3', 'C3', 'P1', 60.0)
    assert new[:-1] == pos[:-1]                 # nothing else touched
    assert pos[-1]['x_um'] == 9000.0            # the input is not modified


def test_a_tile_scan_moves_as_a_whole_centered_on_the_point():
    pos = _well_positions()
    new, what = ex.move_in_well(pos, 'B2', 2000.0, 2100.0)
    assert 'B2' in what and '4 tiles' in what
    tiles = new[:4]
    xs = [p['x_um'] for p in tiles]
    ys = [p['y_um'] for p in tiles]
    assert ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2) == \
        pytest.approx((2000.0, 2100.0))
    assert max(xs) - min(xs) == pytest.approx(900.0)   # spacing unchanged
    assert [(p['name'], p['site']) for p in tiles] == \
        [(p['name'], p['site']) for p in pos[:4]]
    assert new[4] == pos[4]


def test_moving_asks_for_the_focus_at_the_new_place():
    new, _ = ex.move_in_well(_well_positions(), 'C3', 9100.0, 9100.0,
                             z_of=lambda x, y, old: old + 5.0)
    assert new[-1]['z_um'] == 65.0


def test_a_well_without_positions_has_nothing_to_move():
    pos = _well_positions()
    new, what = ex.move_in_well(pos, 'D4', 1.0, 2.0)
    assert what == '' and new == pos


def test_added_points_take_the_well_name_and_count_up():
    pos = _well_positions()
    pos, a = ex.add_point(pos, 'C3', 9300.0, 9300.0, 61.0)
    pos, b = ex.add_point(pos, 'C3', 8700.0, 9300.0, 62.0)
    pos, c = ex.add_point(pos, 'B2', 1500.0, 2500.0, 63.0)   # beside tiles
    pos, d = ex.add_point(pos, 'E5', 100.0, 100.0, 64.0)     # empty well
    assert [p['name'] for p in (a, b, c, d)] == \
        ['C3_P2', 'C3_P3', 'B2_P1', 'E5_P1']
    assert (a['well'], a['site']) == ('C3', 'P2')
    assert ex.file_name(b['well'], b['site'], 1, 'GREEN', 1) == \
        'C3_P3_Z001_GREEN_t0001.tiff'


def test_a_point_tag_is_not_reused_after_one_is_removed():
    pos = _well_positions()
    pos, _ = ex.add_point(pos, 'C3', 1.0, 1.0, 0.0)          # C3_P2
    pos, _ = ex.add_point(pos, 'C3', 2.0, 2.0, 0.0)          # C3_P3
    pos = [p for p in pos if p['name'] != 'C3_P2']
    pos, p = ex.add_point(pos, 'C3', 3.0, 3.0, 0.0)
    assert p['name'] == 'C3_P4'


def test_an_added_point_moves_by_itself_beside_a_tile_scan():
    pos, pt = ex.add_point(_well_positions(), 'B2', 3000.0, 3500.0, 0.0)
    new, what = ex.move_in_well(pos, 'B2', 3100.0, 3600.0)   # nearest: point
    assert what == 'B2_P1'
    assert (new[-1]['x_um'], new[-1]['y_um']) == (3100.0, 3600.0)
    assert new[:4] == pos[:4]                                # tiles stay


def test_points_without_a_plate_are_numbered_in_one_series():
    pos, a = ex.add_point([], None, 1.0, 1.0, 0.0)
    pos, b = ex.add_point(pos, None, 2.0, 2.0, 0.0)
    assert [a['name'], b['name']] == ['P1', 'P2']


# ----------------------------------------- z-stack from begin to end

def test_stack_between_includes_both_ends_in_the_order_given():
    assert ex.stack_between(3000.0, 3050.0, 10.0) == \
        pytest.approx([3000, 3010, 3020, 3030, 3040, 3050])
    assert ex.stack_between(3050.0, 3000.0, 25.0) == \
        pytest.approx([3050, 3025, 3000])          # top down if set so


def test_stack_between_evens_out_an_interval_that_does_not_divide():
    planes = ex.stack_between(100.0, 147.0, 10.0)   # 4.7 intervals -> 5
    assert planes[0] == 100.0 and planes[-1] == pytest.approx(147.0)
    assert len(planes) == 6
    steps = {round(b - a, 6) for a, b in zip(planes, planes[1:])}
    assert steps == {9.4}


def test_stack_between_degenerate_cases():
    assert ex.stack_between(50.0, 50.0, 5.0) == [50.0]
    assert ex.stack_between(0.0, 3.0, 10.0) == pytest.approx([0.0, 3.0])


def test_optimised_interval_follows_the_lens():
    # 10x NA 0.3 green, 0.65 um/px: section = 0.52/0.09 + 0.65/0.3
    section = ex.optical_section_um(0.3, 520.0, 1.0, 0.65)
    assert section == pytest.approx(5.778 + 2.167, abs=0.01)
    assert ex.stack_step_um(section, 1.0) == pytest.approx(section)
    assert ex.stack_step_um(section, 2.3) == pytest.approx(section / 2.3)
    # a higher NA lens gives a thinner section, so a finer interval
    assert ex.optical_section_um(0.45, 520.0, 1.0, 0.325) < section
