from __future__ import annotations
import math
import numpy as np
from kivy.graphics import Color, Line, Rectangle, Ellipse
from kivy.graphics.texture import Texture
from kivy.properties import BooleanProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget


class DragAwareScrollView(ScrollView):
    """ScrollView that hands a press straight to a child marked
    ``captures_drag`` instead of treating the drag as a scroll (a quick
    drag on the plate map would otherwise scroll the panel)."""

    def on_touch_down(self, touch):
        button = getattr(touch, 'button', None) or ''
        if self.collide_point(*touch.pos) and not button.startswith('scroll'):
            touch.push()
            touch.apply_transform_2d(self.to_local)
            try:
                for w in self.walk(restrict=True):
                    if (getattr(w, 'captures_drag', False)
                            and w.collide_point(*touch.pos)):
                        return w.on_touch_down(touch)
            finally:
                touch.pop()
        return super().on_touch_down(touch)


def serpentine(wells) -> list[tuple[int, int]]:
    """Order wells row by row, alternating direction, so the stage
    snakes through a group instead of flying back at each row end."""
    rows: dict[int, list[int]] = {}
    for r, c in set(wells):
        rows.setdefault(r, []).append(c)
    out = []
    for i, r in enumerate(sorted(rows)):
        cols = sorted(rows[r], reverse=bool(i % 2))
        out.extend((r, c) for c in cols)
    return out


def toggle_well_positions(positions: list, wells, turn_on: bool,
                          well_of, make) -> tuple[list, int]:
    """Turn wells on or off in a list of positions.

    ``well_of(position)`` gives the (row, col) a position lies in (or
    None); ``make((row, col))`` builds a position at a well center.
    Turning on appends one position per well that has none, in serpentine
    order; turning off drops every position inside the given wells.
    Positions outside any well are never touched. Returns the new list and
    how many positions were added or removed."""
    if turn_on:
        have = {well_of(p) for p in positions}
        new = [make(w) for w in serpentine(wells) if w not in have]
        return positions + new, len(new)
    gone = set(wells)
    kept = [p for p in positions if well_of(p) not in gone]
    return kept, len(positions) - len(kept)


class Collapsible(BoxLayout):
    """Accordion-style section: header button toggles the body open/shut.
    Child widgets declared inside it in kv are redirected into the body.
    The body is physically removed from the tree when collapsed (hidden
    widgets would otherwise still overlap and swallow touches)."""

    title = StringProperty('')
    open = BooleanProperty(False)
    show_header = BooleanProperty(True)

    def __init__(self, **kwargs):
        self._body = None
        super().__init__(**kwargs)

    def _find_body(self):
        # ids hold weak references: keep a strong one, or the body is
        # garbage-collected while it is out of the tree (collapsed) and
        # the section can never be opened again.
        body = self.ids.get('body') if self.ids else None
        return getattr(body, '__self__', body)

    def add_widget(self, widget, *args, **kwargs):
        body = self._body or self._find_body()
        if body is not None and widget is not body:
            return body.add_widget(widget, *args, **kwargs)
        return super().add_widget(widget, *args, **kwargs)

    def on_kv_post(self, base_widget):
        self._body = self._find_body()
        if self._body is not None and not self.open and self._body.parent is self:
            super().remove_widget(self._body)
        return super().on_kv_post(base_widget)

    def on_open(self, instance, value):
        body = self._body
        if body is None:
            return
        if value and body.parent is None:
            super().add_widget(body)
        elif not value and body.parent is self:
            super().remove_widget(body)


class WellPlateFormat:
    """ANSI/SLAS microplate geometry (default: 96-well)."""

    PLATE_LENGTH_MM = 127.76
    PLATE_WIDTH_MM = 85.48

    def __init__(self, num_wells: int = 96):
        self.num_wells = num_wells
        if num_wells == 384:
            self.rows, self.cols = 16, 24
            self.well_spacing_mm = 4.5
            self.well_diameter = 3.3
            self.a1_offset_x = 12.13
            self.a1_offset_y = 8.99
        elif num_wells == 24:
            self.rows, self.cols = 4, 6
            self.well_spacing_mm = 19.3
            self.well_diameter = 15.6
            self.a1_offset_x = 14.38
            self.a1_offset_y = 13.49
        else:  # 96-well
            self.rows, self.cols = 8, 12
            self.well_spacing_mm = 9.0
            self.well_diameter = 6.4
            self.a1_offset_x = 14.38
            self.a1_offset_y = 11.24

    def get_well_name(self, row: int, col: int) -> str:
        return f"{chr(65 + row)}{col + 1}"

    def get_nominal_position(self, row: int, col: int) -> tuple[float, float]:
        """Well center relative to the A1 center, in mm (+x right, +y down)."""
        return col * self.well_spacing_mm, row * self.well_spacing_mm


class NavigatorPlateCalibration:
    def __init__(self, plate_format: WellPlateFormat):
        self.plate_format = plate_format


class FocusMap:
    def __init__(self):
        self.points = {}

    def add_point(self, well_name: str, x_mm: float, y_mm: float, z_mm: float):
        self.points[well_name] = {'x': x_mm, 'y': y_mm, 'z': z_mm}


class PlateFOVManager:
    def __init__(self, plate_format: WellPlateFormat, calib: NavigatorPlateCalibration, focus_map: FocusMap):
        self.plate_format = plate_format
        self.calib = calib
        self.focus_map = focus_map


class ZoomPanCameraWidget(Image):
    """Live camera viewport.

    - Bit-depth-aware display scaling: frames are scaled by the camera's
      significant bits (not min/max), so changing between 8/10/12/16-bit
      does NOT change on-screen brightness. An optional autoscale mode
      stretches to the frame's own min/max instead.
    - Per-channel LUT tint (grayscale / cyan / green / red / magenta).
    - Toggleable crosshair overlay for calibration.
    - Double-click reports the clicked point relative to the image center
      (in image pixels) via ``on_image_double_click(dx_px, dy_px)`` so the
      app can center the stage on it.
    """

    show_crosshair = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fit_mode = 'contain'
        self.autoscale = False
        self.on_image_double_click = None  # callable(dx_px, dy_px)
        self.on_focus_scroll = None        # callable(direction: +1 up / -1 down)
        self._frame_px_w = 0
        self._frame_px_h = 0
        self.set_standby_texture()
        self.bind(pos=self._redraw_overlay, size=self._redraw_overlay,
                  show_crosshair=self._redraw_overlay)

    def set_standby_texture(self):
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = [15, 18, 24]
        tex = Texture.create(size=(w, h), colorfmt='rgb')
        tex.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        tex.flip_vertical()
        self.texture = tex

    # ------------------------------------------------------------ display

    def _to_display_8bit(self, frame_np: np.ndarray,
                         significant_bits: int) -> np.ndarray:
        """Scale a mono frame to 8-bit for display without losing apparent
        brightness when the camera bit depth changes."""
        if self.autoscale:
            f_min = float(frame_np.min())
            f_max = float(frame_np.max())
            if f_max > f_min:
                norm = (frame_np.astype(np.float32) - f_min) * (255.0 / (f_max - f_min))
                return norm.astype(np.uint8)
            return np.clip(frame_np, 0, 255).astype(np.uint8)
        bits = max(int(significant_bits), 8)
        if frame_np.dtype == np.uint8 and bits <= 8:
            return frame_np
        if bits > 8:
            return (frame_np >> (bits - 8)).astype(np.uint8)
        return np.clip(frame_np, 0, 255).astype(np.uint8)

    def update_frame_texture(self, frame_np: np.ndarray, tint=None,
                             significant_bits: int = 8, clip_level=None):
        """``clip_level``: raw value at/above which a pixel is saturated;
        those pixels are painted red so clipping is visible regardless of
        display autoscale or LUT tint."""
        if frame_np is None or not isinstance(frame_np, np.ndarray) or frame_np.size == 0:
            return

        try:
            frame_8u = self._to_display_8bit(frame_np, significant_bits)

            if len(frame_8u.shape) == 2:
                # Channel LUT tint (RGB weights 0..1). Weights of 0/1 are
                # the common case (grayscale, cyan, green, red, magenta)
                # and need no float math.
                if tint is None:
                    tint = (1.0, 1.0, 1.0)
                zeros = None
                chans = []
                for weight in tint[:3]:
                    if weight >= 0.999:
                        chans.append(frame_8u)
                    elif weight <= 0.001:
                        if zeros is None:
                            zeros = np.zeros_like(frame_8u)
                        chans.append(zeros)
                    else:
                        chans.append((frame_8u.astype(np.float32) * weight).astype(np.uint8))
                frame_rgb = np.stack(chans, axis=-1)
            else:
                frame_rgb = frame_8u

            if clip_level is not None:
                raw = frame_np if frame_np.ndim == 2 else frame_np.max(axis=2)
                clipped = raw >= clip_level
                if clipped.any():
                    if frame_rgb is frame_np or not frame_rgb.flags.writeable:
                        frame_rgb = frame_rgb.copy()
                    frame_rgb[clipped] = (255, 0, 0)

            h, w = frame_rgb.shape[:2]
            self._frame_px_w, self._frame_px_h = w, h

            if self.texture is None or self.texture.width != w or self.texture.height != h:
                tex = Texture.create(size=(w, h), colorfmt='rgb')
                self.texture = tex

            buf = np.ascontiguousarray(frame_rgb[::-1, :]).tobytes()
            self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.canvas.ask_update()
            self._redraw_overlay()
        except Exception:
            pass

    # ------------------------------------------------- geometry / overlay

    def _display_rect(self) -> tuple[float, float, float, float]:
        """(x, y, w, h) of the letterboxed image area inside the widget."""
        nw, nh = self.norm_image_size
        x = self.center_x - nw / 2.0
        y = self.center_y - nh / 2.0
        return x, y, nw, nh

    def _redraw_overlay(self, *args):
        self.canvas.after.clear()
        if not self.show_crosshair:
            return
        x, y, w, h = self._display_rect()
        cx, cy = x + w / 2.0, y + h / 2.0
        with self.canvas.after:
            Color(0.1, 1.0, 0.3, 0.9)
            Line(points=[x, cy, x + w, cy], width=1.0)
            Line(points=[cx, y, cx, y + h], width=1.0)
            Line(circle=(cx, cy, min(w, h) * 0.05), width=1.0)
            Line(circle=(cx, cy, min(w, h) * 0.15), width=1.0)

    # --------------------------------------------------------------- input

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            # Mouse wheel over the image = fine focus (Z).
            if getattr(touch, 'button', '') in ('scrollup', 'scrolldown'):
                if self.on_focus_scroll:
                    self.on_focus_scroll(1 if touch.button == 'scrolldown' else -1)
                return True
            if touch.is_double_tap:
                x, y, w, h = self._display_rect()
                if w > 1 and h > 1 and self._frame_px_w > 0:
                    fx = (touch.x - (x + w / 2.0)) / w       # -0.5 .. +0.5
                    fy = (touch.y - (y + h / 2.0)) / h
                    if -0.5 <= fx <= 0.5 and -0.5 <= fy <= 0.5:
                        dx_px = fx * self._frame_px_w
                        # Screen +y is up; image rows grow downward.
                        dy_px = -fy * self._frame_px_h
                        if self.on_image_double_click:
                            self.on_image_double_click(dx_px, dy_px)
                        return True
        return super().on_touch_down(touch)


class WellPlateMapWidget(Widget):
    """Interactive well plate navigator map (LAS X Navigator style).

    Draws the current labware (round or square wells); shows the stage
    position as a marker and, during plate calibration, the guided target
    position. Double-click fires
    ``on_well_double_click(row, col, plate_x_mm, plate_y_mm)`` where the
    plate coordinates are the well center relative to the A1 center
    (+x toward the last column, +y toward the last row).

    Well on/off selection (only when ``on_wells_toggle`` is set): the
    widget shows ``active_wells`` filled and asks the owner to change them
    through ``on_wells_toggle(wells, turn_on)``; it never edits the set
    itself. A click toggles one well. A drag draws a rectangle and applies
    one action to every well whose center is inside it: the action a click
    on the starting well would have had (start on an on well = turn off,
    otherwise turn on); Shift forces on, Ctrl forces off.
    """

    DRAG_THRESHOLD_PX = 6.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = WellPlateFormat(num_wells=96)
        self.labware = None            # LabwareDef overrides plate_format
        self.selected_well: tuple[int, int] | None = None
        self.marker_plate_mm: tuple[float, float] | None = None
        self.target_plate_mm: tuple[float, float] | None = None  # calibration
        self.position_markers: list[tuple[float, float]] = []    # protocol
        self.highlight_marker: int | None = None
        self.on_well_double_click = None  # callable(row, col, x_mm, y_mm)
        self.on_well_selected = None      # callable(row, col, x_mm, y_mm)
        self.active_wells: set[tuple[int, int]] = set()
        self.on_wells_toggle = None       # callable(wells, turn_on)
        self._drag: dict | None = None
        self._last_toggle: tuple[tuple[int, int], bool] | None = None
        self.bind(pos=self.redraw, size=self.redraw)

    @property
    def captures_drag(self) -> bool:
        return self.on_wells_toggle is not None

    def set_active_wells(self, wells):
        self.active_wells = set(wells)
        self.redraw()

    # ----------------------------------------------------- labware access

    def _lw(self):
        return self.labware if self.labware is not None else self.plate_format

    def _spacing(self) -> tuple[float, float]:
        lw = self._lw()
        sx = getattr(lw, 'spacing_x_mm', None)
        sy = getattr(lw, 'spacing_y_mm', None)
        if sx is None:
            sx = sy = lw.well_spacing_mm
        return sx, sy

    def _well_size(self) -> tuple[float, float]:
        lw = self._lw()
        wx = getattr(lw, 'well_size_x_mm', None)
        wy = getattr(lw, 'well_size_y_mm', None)
        if wx is None:
            wx = wy = lw.well_diameter
        return wx, wy

    def _plate_dims(self) -> tuple[float, float]:
        lw = self._lw()
        return (getattr(lw, 'plate_length_mm', WellPlateFormat.PLATE_LENGTH_MM),
                getattr(lw, 'plate_width_mm', WellPlateFormat.PLATE_WIDTH_MM))

    def set_labware(self, labware):
        self.labware = labware
        self.selected_well = None
        self.active_wells = set()
        self._last_toggle = None
        self.redraw()

    # ------------------------------------------------------------ mapping

    def _plate_draw_rect(self) -> tuple[float, float, float, float]:
        """(x, y, w, h) of the plate outline inside the widget, keeping the
        physical aspect ratio."""
        pad = 8.0
        pl, pw = self._plate_dims()
        avail_w = max(self.width - 2 * pad, 10.0)
        avail_h = max(self.height - 2 * pad, 10.0)
        aspect = pl / pw
        if avail_w / avail_h > aspect:
            h = avail_h
            w = h * aspect
        else:
            w = avail_w
            h = w / aspect
        x = self.x + (self.width - w) / 2.0
        y = self.y + (self.height - h) / 2.0
        return x, y, w, h

    def _mm_to_screen(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Plate mm (origin = plate top-left corner, +y down) -> screen."""
        x, y, w, h = self._plate_draw_rect()
        pl, pw = self._plate_dims()
        sx = x + (mm_x / pl) * w
        sy = y + h - (mm_y / pw) * h
        return sx, sy

    def _screen_to_mm(self, sx: float, sy: float) -> tuple[float, float]:
        x, y, w, h = self._plate_draw_rect()
        pl, pw = self._plate_dims()
        mm_x = (sx - x) / w * pl
        mm_y = (y + h - sy) / h * pw
        return mm_x, mm_y

    def _well_center_mm(self, row: int, col: int) -> tuple[float, float]:
        lw = self._lw()
        sx, sy = self._spacing()
        return (lw.a1_offset_x + col * sx, lw.a1_offset_y + row * sy)

    def _nearest_well(self, mm_x: float, mm_y: float) -> tuple[int, int] | None:
        lw = self._lw()
        sx, sy = self._spacing()
        col = round((mm_x - lw.a1_offset_x) / sx) if sx > 0 else 0
        row = round((mm_y - lw.a1_offset_y) / sy) if sy > 0 else 0
        if 0 <= row < lw.rows and 0 <= col < lw.cols:
            wx, wy = self._well_center_mm(row, col)
            lim = max(sx, sy) * 0.75
            if math.hypot(mm_x - wx, mm_y - wy) <= lim:
                return row, col
        return None

    def set_marker_plate_mm(self, x_mm_from_a1: float, y_mm_from_a1: float):
        """Place the stage-position marker (coords relative to A1 center)."""
        lw = self._lw()
        self.marker_plate_mm = (lw.a1_offset_x + x_mm_from_a1,
                                lw.a1_offset_y + y_mm_from_a1)
        self.redraw()

    def set_target_plate_mm(self, plate_mm: tuple[float, float] | None):
        """Show/clear the guided calibration target (plate top-left origin)."""
        self.target_plate_mm = plate_mm
        self.redraw()

    def set_position_markers(self, plate_mm_list: list[tuple[float, float]],
                             highlight: int | None = None):
        """Show the protocol interaction positions (plate top-left origin)."""
        self.position_markers = list(plate_mm_list)
        self.highlight_marker = highlight
        self.redraw()

    # ------------------------------------------------------------- drawing

    def redraw(self, *args):
        self.canvas.clear()
        lw = self._lw()
        x, y, w, h = self._plate_draw_rect()
        pl, pw = self._plate_dims()
        wx_mm, wy_mm = self._well_size()
        well_rx = (wx_mm / 2.0) / pl * w
        well_ry = (wy_mm / 2.0) / pw * h
        shape = getattr(lw, 'shape', 'round')
        with self.canvas:
            # Plate body
            Color(0.13, 0.15, 0.19, 1)
            Rectangle(pos=(x, y), size=(w, h))
            Color(0.45, 0.5, 0.58, 1)
            Line(rectangle=(x, y, w, h), width=1.2)
            # A1 corner chamfer marker (top-left)
            Line(points=[x, y + h - 14, x + 14, y + h], width=1.2)
            # Wells
            for r in range(lw.rows):
                for c in range(lw.cols):
                    mmx, mmy = self._well_center_mm(r, c)
                    sx, sy = self._mm_to_screen(mmx, mmy)
                    if (r, c) in self.active_wells:
                        Color(0.1, 0.75, 0.45, 0.55)
                        fill = Rectangle if shape == 'square' else Ellipse
                        fill(pos=(sx - well_rx, sy - well_ry),
                             size=(2 * well_rx, 2 * well_ry))
                    if self.selected_well == (r, c):
                        Color(0.1, 0.8, 1.0, 1)
                    else:
                        Color(0.35, 0.4, 0.48, 1)
                    if shape == 'square':
                        Line(rectangle=(sx - well_rx, sy - well_ry,
                                        2 * well_rx, 2 * well_ry), width=1.0)
                    else:
                        Line(ellipse=(sx - well_rx, sy - well_ry,
                                      2 * well_rx, 2 * well_ry), width=1.0)
            # Protocol position markers
            # Many positions (tiles): small dots, so the pattern inside
            # each well stays readable instead of merging into a blob.
            dense = len(self.position_markers) > 150
            for i, pm in enumerate(self.position_markers):
                sx, sy = self._mm_to_screen(*pm)
                if i == self.highlight_marker:
                    Color(1.0, 0.85, 0.1, 1)
                    Line(points=[sx - 6, sy, sx, sy + 6, sx + 6, sy, sx, sy - 6,
                                 sx - 6, sy], width=1.6)
                elif dense:
                    Color(0.95, 0.75, 0.1, 0.9)
                    Rectangle(pos=(sx - 1, sy - 1), size=(2, 2))
                else:
                    Color(0.95, 0.75, 0.1, 0.9)
                    Line(points=[sx - 4, sy, sx, sy + 4, sx + 4, sy, sx, sy - 4,
                                 sx - 4, sy], width=1.1)
            # Stage position marker
            if self.marker_plate_mm is not None:
                sx, sy = self._mm_to_screen(*self.marker_plate_mm)
                Color(0.15, 1.0, 0.35, 1)
                Line(points=[sx - 7, sy, sx + 7, sy], width=1.4)
                Line(points=[sx, sy - 7, sx, sy + 7], width=1.4)
                Line(circle=(sx, sy, 5), width=1.2)
            # Calibration target marker
            if self.target_plate_mm is not None:
                sx, sy = self._mm_to_screen(*self.target_plate_mm)
                Color(1.0, 0.35, 0.25, 1)
                Line(circle=(sx, sy, 8), width=1.6)
                Line(points=[sx - 12, sy, sx - 4, sy], width=1.6)
                Line(points=[sx + 4, sy, sx + 12, sy], width=1.6)
                Line(points=[sx, sy - 12, sx, sy - 4], width=1.6)
                Line(points=[sx, sy + 4, sx, sy + 12], width=1.6)

    # --------------------------------------------------------------- input

    def _modifiers(self) -> set[str]:
        try:
            from kivy.core.window import Window
            return set(Window.modifiers)
        except Exception:
            return set()

    def _rel_mm(self, well: tuple[int, int]) -> tuple[float, float]:
        """Well center relative to the A1 center."""
        sx, sy = self._spacing()
        return well[1] * sx, well[0] * sy

    def wells_in_screen_rect(self, x0: float, y0: float, x1: float,
                             y1: float) -> list[tuple[int, int]]:
        """Wells whose center lies inside the screen rectangle."""
        ax, ay = self._screen_to_mm(x0, y0)
        bx, by = self._screen_to_mm(x1, y1)
        lo_x, hi_x = min(ax, bx), max(ax, bx)
        lo_y, hi_y = min(ay, by), max(ay, by)
        lw = self._lw()
        out = []
        for r in range(lw.rows):
            for c in range(lw.cols):
                mx, my = self._well_center_mm(r, c)
                if lo_x <= mx <= hi_x and lo_y <= my <= hi_y:
                    out.append((r, c))
        return out

    def _draw_lasso(self):
        self.canvas.after.clear()
        d = self._drag
        if not d or not d['moved']:
            return
        (x0, y0), (x1, y1) = d['start'], d['cur']
        rgb = (1.0, 0.3, 0.25) if d['remove'] else (0.1, 0.8, 1.0)
        with self.canvas.after:
            Color(*rgb, 0.18)
            Rectangle(pos=(min(x0, x1), min(y0, y1)),
                      size=(abs(x1 - x0), abs(y1 - y0)))
            Color(*rgb, 1)
            Line(rectangle=(min(x0, x1), min(y0, y1),
                            abs(x1 - x0), abs(y1 - y0)), width=1.2)

    def _select(self, well: tuple[int, int]):
        self.selected_well = well
        self.redraw()
        if self.on_well_selected:
            self.on_well_selected(well[0], well[1], *self._rel_mm(well))

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if (getattr(touch, 'button', None) or '').startswith('scroll'):
            return False
        well = self._nearest_well(*self._screen_to_mm(touch.x, touch.y))

        if touch.is_double_tap:
            if well is None:
                return True
            # The first click of the double-click toggled this well: put
            # it back, a double-click only moves the stage.
            if self._last_toggle and self._last_toggle[0] == well:
                self.on_wells_toggle([well], not self._last_toggle[1])
            self._last_toggle = None
            self.selected_well = well
            self.redraw()
            if self.on_well_double_click:
                self.on_well_double_click(well[0], well[1],
                                          *self._rel_mm(well))
            return True

        if self.on_wells_toggle is None:
            if well is not None:
                self._select(well)
            return True

        mods = self._modifiers()
        if 'shift' in mods:
            remove = False
        elif 'ctrl' in mods:
            remove = True
        else:
            remove = well is not None and well in self.active_wells
        self._drag = {'start': touch.pos, 'cur': touch.pos, 'well': well,
                      'remove': remove, 'moved': False}
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self or self._drag is None:
            return super().on_touch_move(touch)
        d = self._drag
        d['cur'] = touch.pos
        if not d['moved'] and math.hypot(
                touch.x - d['start'][0],
                touch.y - d['start'][1]) > self.DRAG_THRESHOLD_PX:
            d['moved'] = True
        self._draw_lasso()
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return super().on_touch_up(touch)
        touch.ungrab(self)
        d, self._drag = self._drag, None
        self.canvas.after.clear()
        if d is None:
            return True
        if d['moved']:
            wells = self.wells_in_screen_rect(*d['start'], *d['cur'])
            self._last_toggle = None
            if wells:
                self.on_wells_toggle(wells, not d['remove'])
        elif d['well'] is not None:
            well = d['well']
            self._select(well)
            turn_on = not d['remove']
            self._last_toggle = (well, turn_on)
            self.on_wells_toggle([well], turn_on)
        return True


class InteractiveScaledPlateCanvas(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = WellPlateFormat(num_wells=96)


def attach_navigator_to_lumaviewpro(app_instance):
    plate_fmt = WellPlateFormat(num_wells=96)
    calib = NavigatorPlateCalibration(plate_fmt)
    focus_m = FocusMap()
    fov_m = PlateFOVManager(plate_fmt, calib, focus_m)

    app_instance.plate_format = plate_fmt
    app_instance.plate_calibration = calib
    app_instance.focus_map = focus_m
    app_instance.fov_manager = fov_m
