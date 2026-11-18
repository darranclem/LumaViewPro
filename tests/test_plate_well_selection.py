"""Plate navigator well on/off selection: click toggles, drag a rectangle
to switch a group (plate_navigator.WellPlateMapWidget)."""

from __future__ import annotations

import pytest

import plate_navigator as pn


class FakeTouch:
    def __init__(self, x, y, double=False, button='left'):
        self.x, self.y = x, y
        self.is_double_tap = double
        self.button = button
        self.grab_current = None

    @property
    def pos(self):
        return (self.x, self.y)

    def grab(self, w):
        self.grab_current = w

    def ungrab(self, w):
        self.grab_current = None


class Plate:
    """A 96-well map whose owner applies every toggle request."""

    def __init__(self, owner=True):
        self.w = pn.WellPlateMapWidget()
        self.w.pos = (0, 0)
        self.w.size = (640, 440)
        self.mods: set[str] = set()
        self.w._modifiers = lambda: self.mods
        self.w.click_delay_s = 0            # act on a click at once
        self.calls: list[tuple[list, bool]] = []
        self.moves: list[tuple[int, int]] = []
        self.w.on_well_double_click = lambda r, c, x, y: self.moves.append((r, c))
        if owner:
            self.w.on_wells_toggle = self._toggle

    def _toggle(self, wells, turn_on):
        self.calls.append((list(wells), turn_on))
        active = set(self.w.active_wells)
        active = active | set(wells) if turn_on else active - set(wells)
        self.w.set_active_wells(active)

    def xy(self, row, col):
        return self.w._mm_to_screen(*self.w._well_center_mm(row, col))

    def click(self, row, col, double=False):
        t = FakeTouch(*self.xy(row, col), double=double)
        self.w.on_touch_down(t)
        if t.grab_current is self.w:
            self.w.on_touch_up(t)

    def drag(self, start, end, pad=3.0):
        """Drag from just outside ``start``'s center to just past ``end``'s."""
        (x0, y0), (x1, y1) = self.xy(*start), self.xy(*end)
        sx = pad if x1 >= x0 else -pad
        sy = pad if y1 >= y0 else -pad
        t = FakeTouch(x0 - sx, y0 - sy)
        self.w.on_touch_down(t)
        t.x, t.y = x1 + sx, y1 + sy
        self.w.on_touch_move(t)
        self.w.on_touch_up(t)

    @property
    def active(self):
        return self.w.active_wells


def test_a_click_turns_a_well_on_and_a_second_click_turns_it_off():
    p = Plate()
    p.click(2, 3)
    assert p.active == {(2, 3)}
    p.click(2, 3)
    assert p.active == set()
    assert p.calls == [([(2, 3)], True), ([(2, 3)], False)]


def test_a_rectangle_turns_on_every_well_whose_center_is_inside():
    p = Plate()
    p.drag((0, 0), (1, 2))
    assert p.active == {(r, c) for r in range(2) for c in range(3)}
    assert len(p.calls) == 1 and p.calls[0][1] is True


def test_a_rectangle_can_be_dragged_in_any_direction():
    p = Plate()
    p.drag((3, 5), (1, 2))
    assert p.active == {(r, c) for r in (1, 2, 3) for c in (2, 3, 4, 5)}


def test_a_rectangle_started_on_an_on_well_turns_the_group_off():
    p = Plate()
    p.drag((0, 0), (3, 3))
    p.drag((1, 1), (2, 2))
    assert p.calls[-1][1] is False
    assert len(p.active) == 16 - 4
    assert (1, 1) not in p.active and (0, 0) in p.active


def test_shift_forces_on_and_ctrl_forces_off():
    p = Plate()
    p.click(0, 0)
    p.mods = {'shift'}
    p.drag((0, 0), (0, 2))          # starts on an on well, still adds
    assert p.active == {(0, 0), (0, 1), (0, 2)}
    p.mods = {'ctrl'}
    p.drag((1, 0), (0, 1))          # starts on an off well, still removes
    assert p.active == {(0, 2)}


def test_a_small_wobble_is_still_a_click():
    p = Plate()
    x, y = p.xy(4, 4)
    t = FakeTouch(x, y)
    p.w.on_touch_down(t)
    t.x += 3
    p.w.on_touch_move(t)
    p.w.on_touch_up(t)
    assert p.calls == [([(4, 4)], True)]


def test_a_double_click_moves_the_stage_and_leaves_the_well_as_it_was():
    p = Plate()
    p.click(1, 1)                    # on
    p.click(5, 5)                    # first click of the double-click
    p.click(5, 5, double=True)
    assert p.moves == [(5, 5)]
    assert p.active == {(1, 1)}
    p.click(1, 1)                    # first click of the double-click
    p.click(1, 1, double=True)
    assert p.moves == [(5, 5), (1, 1)]
    assert p.active == {(1, 1)}


def test_the_widget_never_edits_the_selection_itself():
    p = Plate()
    p.w.on_wells_toggle = lambda wells, on: None    # owner refuses
    p.click(0, 0)
    p.drag((0, 0), (2, 2))
    assert p.active == set()


def test_without_an_owner_a_click_only_selects():
    p = Plate(owner=False)
    assert not p.w.captures_drag
    t = FakeTouch(*p.xy(0, 1))
    p.w.on_touch_down(t)
    assert t.grab_current is None
    assert p.w.selected_well == (0, 1) and p.active == set()


def test_ctrl_wheel_zooms_and_never_toggles_a_well():
    p = Plate()
    p.mods = {'ctrl'}
    t = FakeTouch(*p.xy(0, 0), button='scrolldown')
    assert p.w.on_touch_down(t) is True
    assert p.w.zoom > 1.0 and p.calls == []


def test_the_wheel_alone_is_left_to_the_panel_on_the_small_map():
    p = Plate()
    focus = []
    p.w.on_focus_scroll = focus.append
    t = FakeTouch(*p.xy(0, 0), button='scrolldown')
    assert p.w.on_touch_down(t) is False        # the panel scrolls
    assert p.w.zoom == 1.0 and focus == [] and p.calls == []


def test_the_wheel_alone_focuses_when_the_map_is_the_main_view():
    p = Plate()
    focus = []
    p.w.on_focus_scroll = focus.append
    p.w.wheel_focus = True
    assert p.w.on_touch_down(
        FakeTouch(*p.xy(2, 2), button='scrolldown')) is True
    assert p.w.on_touch_down(
        FakeTouch(*p.xy(2, 2), button='scrollup')) is True
    assert focus == [1, -1]                     # as on the camera view
    assert p.w.zoom == 1.0 and p.calls == []
    p.mods = {'ctrl'}                           # Ctrl + wheel still zooms
    p.w.on_touch_down(FakeTouch(*p.xy(2, 2), button='scrolldown'))
    assert p.w.zoom > 1.0 and focus == [1, -1]


def test_changing_labware_clears_the_shown_selection():
    p = Plate()
    p.click(0, 0)
    p.w.set_labware(None)
    assert p.active == set()


# ------------------------------------------------------ position list

def test_serpentine_snakes_row_by_row():
    wells = [(1, 0), (0, 2), (0, 0), (1, 2), (0, 1), (1, 1), (0, 0)]
    assert pn.serpentine(wells) == [(0, 0), (0, 1), (0, 2),
                                    (1, 2), (1, 1), (1, 0)]


def _well_of(p):
    return p.get('well')


def _make(well):
    return {'name': f'{well}', 'well': well}


def test_turning_wells_on_adds_only_the_missing_ones():
    manual = {'name': 'P01', 'well': (0, 1), 'z_um': 123.0}
    out, n = pn.toggle_well_positions([manual], [(0, 0), (0, 1), (0, 2)],
                                      True, _well_of, _make)
    assert n == 2
    assert out[0] is manual                     # tuned position kept as is
    assert [p['well'] for p in out] == [(0, 1), (0, 0), (0, 2)]


def test_turning_wells_off_removes_every_position_inside_them():
    pos = [_make((0, 0)), _make((0, 0)), _make((1, 1)),
           {'name': 'off-plate', 'well': None}]
    out, n = pn.toggle_well_positions(pos, [(0, 0), (5, 5)], False,
                                      _well_of, _make)
    assert n == 2
    assert [p['name'] for p in out] == ['(1, 1)', 'off-plate']


@pytest.mark.parametrize('turn_on', [True, False])
def test_the_original_list_is_not_modified(turn_on):
    pos = [_make((0, 0))]
    pn.toggle_well_positions(pos, [(0, 0), (0, 1)], turn_on, _well_of, _make)
    assert pos == [_make((0, 0))]


# ------------------------------------------------------- collapsible

def test_a_collapsed_section_can_still_be_opened_after_garbage_collection():
    import gc
    from kivy.lang import Builder
    from kivy.uix.label import Label

    class _Section(pn.Collapsible):
        pass

    Builder.load_string('''
<_Section>:
    BoxLayout:
        id: body
''')
    sec = _Section()                 # starts collapsed: body out of the tree
    sec.add_widget(Label(text='inside'))
    gc.collect()
    sec.open = True
    assert sec._body.parent is sec
    assert [w.text for w in sec._body.children] == ['inside']
    sec.open = False
    gc.collect()
    sec.open = True
    assert sec._body.parent is sec


def test_a_click_waits_out_the_double_click_window_before_toggling():
    p = Plate()
    p.w.click_delay_s = 0.28
    p.click(2, 2)
    assert p.calls == []                 # nothing yet: might be a double
    p.w._fire_pending()                  # the window passed
    assert p.calls == [([(2, 2)], True)]


def test_a_double_click_cancels_the_waiting_toggle_and_only_moves():
    p = Plate()
    p.w.click_delay_s = 0.28
    p.click(2, 2)
    p.click(2, 2, double=True)
    p.w._fire_pending()
    assert p.calls == [] and p.moves == [(2, 2)] and p.active == set()


def test_a_click_on_another_well_applies_the_waiting_one_first():
    p = Plate()
    p.w.click_delay_s = 0.28
    p.click(0, 0)
    p.click(0, 1)
    p.w._fire_pending()
    assert p.calls == [([(0, 0)], True), ([(0, 1)], True)]


# --------------------------------------------- second marking (opt-outs)

def test_mark_mode_edits_the_flagged_wells_and_leaves_the_selection_alone():
    p = Plate()
    p.drag((0, 0), (1, 1))
    flags = []

    def _flag(wells, on):
        flags.append((list(wells), on))
        cur = set(p.w.flagged_wells)
        p.w.flagged_wells = cur | set(wells) if on else cur - set(wells)

    p.w.on_wells_flag = _flag
    p.w.mark_mode = True
    n_calls = len(p.calls)
    p.click(0, 0)                       # an on well, not flagged yet: flag
    p.click(0, 0)                       # now flagged: unflag
    p.drag((0, 0), (0, 1))
    assert flags == [([(0, 0)], True), ([(0, 0)], False),
                     ([(0, 0), (0, 1)], True)]
    assert len(p.calls) == n_calls and len(p.active) == 4
    p.w.mark_mode = False
    p.click(1, 1)
    assert (1, 1) not in p.active       # back to normal on/off


# ------------------------------------------------------- zoom and pan

def test_wheel_zoom_keeps_the_point_under_the_cursor_in_place():
    p = Plate()
    p.mods = {'ctrl'}
    sx, sy = p.xy(3, 7)
    before = p.w._screen_to_mm(sx, sy)
    for _ in range(6):
        assert p.w.on_touch_down(FakeTouch(sx, sy, button='scrolldown')) is True
    assert p.w.zoom == pytest.approx(1.25 ** 6)
    assert p.w._screen_to_mm(sx, sy) == pytest.approx(before, abs=1e-6)
    assert p.calls == []                # the wheel never toggles a well


def test_zooming_out_stops_at_the_whole_plate():
    p = Plate()
    fit = p.w._plate_draw_rect()
    p.w.zoom_by(8.0, anchor=p.xy(1, 1))
    p.mods = {'ctrl'}
    for _ in range(40):
        p.w.on_touch_down(FakeTouch(*p.w.center, button='scrollup'))
    assert p.w.zoom == 1.0 and p.w._plate_draw_rect() == pytest.approx(fit)


def test_clicks_still_land_on_the_right_well_when_zoomed_in():
    p = Plate()
    p.w.zoom_by(5.0, anchor=p.xy(4, 6))
    p.click(4, 6)
    p.click(4, 7)
    assert p.active == {(4, 6), (4, 7)}


def test_follow_keeps_the_stage_position_in_the_middle():
    p = Plate()
    p.w.follow_stage = True
    p.w.set_marker_plate_mm(18.0, 27.0)         # C3 center
    p.w.zoom_to_stage()
    assert p.w.zoom > 1.0
    assert p.xy(3, 2) == pytest.approx(p.w.center, abs=1e-6)
    p.w.set_marker_plate_mm(63.0, 9.0)          # stage moves to B8
    assert p.xy(1, 7) == pytest.approx(p.w.center, abs=1e-6)


def test_right_drag_pans_and_stops_following():
    p = Plate()
    p.w.follow_stage = True
    p.w.set_marker_plate_mm(18.0, 27.0)
    p.w.zoom_to_stage()
    x0, y0 = p.xy(3, 2)
    t = FakeTouch(x0, y0, button='right')
    p.w.on_touch_down(t)
    t.x, t.y = x0 + 50, y0 - 30
    p.w.on_touch_move(t)
    p.w.on_touch_up(t)
    assert p.xy(3, 2) == pytest.approx((x0 + 50, y0 - 30), abs=1e-6)
    assert p.w.follow_stage is False and p.calls == []


def test_zoomed_in_double_click_goes_to_the_exact_point():
    p = Plate()
    points = []
    p.w.on_point_double_click = lambda x, y, well: points.append((x, y, well))
    p.click(2, 2, double=True)                  # whole plate: well center
    assert p.moves == [(2, 2)] and points == []
    p.w.zoom_to_span(p.w._well_center_mm(2, 2), 10.0)
    assert p.w.precise
    sx, sy = p.xy(2, 2)
    px_per_mm = p.w._plate_draw_rect()[2] / p.w._plate_dims()[0]
    p.w.on_touch_down(FakeTouch(sx + 2.0 * px_per_mm, sy + 1.0 * px_per_mm,
                                double=True))
    # 2 mm right of and 1 mm above (toward row A) the C3 center
    assert points[0][:2] == pytest.approx((18.0 + 2.0, 18.0 - 1.0), abs=1e-6)
    assert points[0][2] == (2, 2) and p.moves == [(2, 2)] and p.calls == []


def test_changing_labware_shows_the_whole_plate_again():
    p = Plate()
    p.w.zoom_by(6.0, anchor=p.xy(1, 1))
    p.w.set_labware(None)
    assert p.w.zoom == 1.0 and p.w.view_center_mm is None


def test_live_image_corners_are_stored_from_the_plate_corner():
    p = Plate()
    lw = p.w._lw()
    p.w.set_live_image(object(), [(-1, -1), (1, -1), (1, 1), (-1, 1)])
    assert p.w.live_corners_mm[0] == pytest.approx(
        (lw.a1_offset_x - 1, lw.a1_offset_y - 1))
    p.w.set_live_image(None, None)
    assert p.w.live_corners_mm is None and p.w.live_texture is None


# --------------------------------------------- point clicks (edit mode)

def test_in_point_mode_a_click_reports_the_point_and_leaves_the_wells():
    p = Plate()
    got = []
    p.w.on_point_click = lambda x, y, well: got.append((x, y, well))
    p.w.point_click_mode = True
    sx, sy = p.xy(2, 3)
    px_per_mm = p.w._plate_draw_rect()[2] / p.w._plate_dims()[0]
    t = FakeTouch(sx + 1.5 * px_per_mm, sy - 1.0 * px_per_mm)
    assert p.w.on_touch_down(t) is True
    assert t.grab_current is None and p.calls == []      # no well toggled
    (x, y, well), = got
    sp = p.w._spacing()
    assert well == (2, 3)
    assert (x, y) == pytest.approx((3 * sp[0] + 1.5, 2 * sp[1] + 1.0),
                                   abs=1e-6)


def test_in_point_mode_a_double_click_still_drives_the_stage_only():
    p = Plate()
    got = []
    p.w.on_point_click = lambda x, y, well: got.append(well)
    p.w.point_click_mode = True
    p.click(1, 1, double=True)
    assert got == [] and p.moves == [(1, 1)] and p.calls == []


def test_out_of_point_mode_clicks_switch_wells_as_before():
    p = Plate()
    got = []
    p.w.on_point_click = lambda x, y, well: got.append(well)
    p.click(1, 1)
    assert got == [] and (1, 1) in p.active


# ------------------------------------------------- mirrored plate map

def test_mirrored_map_draws_column_1_on_the_right():
    p = Plate()
    a1, a12 = p.xy(0, 0), p.xy(0, 11)
    assert a1[0] < a12[0]
    p.w.flip_x = True
    f1, f12 = p.xy(0, 0), p.xy(0, 11)
    assert f1[0] > f12[0]                       # A1 now on the right
    assert f1[1] == pytest.approx(a1[1])        # rows are not touched
    # mirror image of where it was, about the middle of the plate picture
    x, _, w, _ = p.w._plate_draw_rect()
    assert f1[0] == pytest.approx(2 * x + w - a1[0])


def test_clicks_drags_and_zoom_still_hit_what_is_drawn_when_mirrored():
    p = Plate()
    p.w.flip_x = True
    p.click(2, 9)
    assert p.active == {(2, 9)}
    p.drag((0, 0), (1, 2))
    assert {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)} <= p.active
    p.mods = {'ctrl'}
    sx, sy = p.xy(5, 3)
    before = p.w._screen_to_mm(sx, sy)
    for _ in range(5):
        p.w.on_touch_down(FakeTouch(sx, sy, button='scrolldown'))
    assert p.w.zoom > 1.0                       # the cursor point stays put
    assert p.w._screen_to_mm(sx, sy) == pytest.approx(before, abs=1e-6)
    p.mods = set()
    p.click(5, 3)
    assert (5, 3) in p.active


def test_screen_and_plate_coordinates_round_trip_when_mirrored():
    p = Plate()
    p.w.flip_x = True
    p.w.zoom_by(4.0, anchor=p.xy(3, 3))
    for mm in ((10.0, 12.0), (100.5, 70.25)):
        assert p.w._screen_to_mm(*p.w._mm_to_screen(*mm)) == \
            pytest.approx(mm, abs=1e-6)
