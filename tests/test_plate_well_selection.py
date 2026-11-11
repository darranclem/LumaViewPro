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


def test_the_mouse_wheel_is_left_to_the_scroll_view():
    p = Plate()
    t = FakeTouch(*p.xy(0, 0), button='scrolldown')
    assert p.w.on_touch_down(t) is False
    assert p.calls == []


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
