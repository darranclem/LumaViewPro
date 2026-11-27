from __future__ import annotations
import math
import numpy as np
from kivy.graphics import (Color, Line, Rectangle, Mesh,
                           StencilPush, StencilUse, StencilUnUse, StencilPop)
from kivy.graphics.texture import Texture
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
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
        # The mouse wheel goes to a child marked ``captures_scroll`` (the
        # plate map zooms with it) instead of scrolling the panel.
        want = 'captures_scroll' if button.startswith('scroll') \
            else 'captures_drag'
        if self.collide_point(*touch.pos):
            touch.push()
            touch.apply_transform_2d(self.to_local)
            try:
                for w in self.walk(restrict=True):
                    if (getattr(w, want, False)
                            and w.collide_point(*touch.pos)):
                        if w.on_touch_down(touch) or want == 'captures_drag':
                            return True
                        break       # wheel not used by the child: scroll
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

    View: ``zoom`` 1 shows the whole plate; Ctrl + mouse wheel zooms about
    the cursor (the wheel alone focuses when the map is the main view),
    a right- or middle-button drag pans, and ``follow_stage``
    keeps the stage position in the middle. The live camera image is
    drawn on the plate at its true size and place (``set_live_image``).
    Once a well is drawn large (``precise``), a double-click goes to the
    exact point clicked instead of the well center.
    """

    zoom = NumericProperty(1.0)
    follow_stage = BooleanProperty(False)
    # Draw the plate mirrored left to right (column 1 on the right): the
    # plate as it lies on this microscope. Only the picture is mirrored;
    # wells, positions and the live image all go through the same
    # mapping, so every click still lands on what is drawn there.
    flip_x = BooleanProperty(False)
    captures_scroll = True
    MAX_ZOOM = 400.0
    ZOOM_STEP = 1.25
    PRECISE_WELL_PX = 60.0     # well drawn at least this wide = precise

    DRAG_THRESHOLD_PX = 6.0
    # A single click acts after this delay, so that a double-click (move
    # the stage) is recognised first and never toggles the well. Without
    # it the work done on the first click delayed the second one past the
    # double-click window. 0 = act at once.
    CLICK_DELAY_S = 0.28

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = WellPlateFormat(num_wells=96)
        self.labware = None            # LabwareDef overrides plate_format
        self.selected_well: tuple[int, int] | None = None
        self.marker_plate_mm: tuple[float, float] | None = None
        self.target_plate_mm: tuple[float, float] | None = None  # calibration
        self.position_markers: list[tuple[float, float]] = []    # protocol
        self.highlight_marker: int | None = None
        # Where focus will be set for the focus map: (x_mm, y_mm, moved).
        self.focus_markers: list[tuple[float, float, bool]] = []
        self.highlight_focus: int | None = None
        self.on_well_double_click = None  # callable(row, col, x_mm, y_mm)
        self.on_well_selected = None      # callable(row, col, x_mm, y_mm)
        self.active_wells: set[tuple[int, int]] = set()
        self.on_wells_toggle = None       # callable(wells, turn_on)
        # A second, independent well marking (drawn struck through), e.g.
        # wells opted out of z-stacks. While mark_mode is on, clicks and
        # rectangles edit this marking through on_wells_flag instead.
        self.flagged_wells: set[tuple[int, int]] = set()
        self.on_wells_flag = None         # callable(wells, flag_on)
        self.mark_mode = False
        self._drag: dict | None = None
        self._last_toggle: tuple | None = None   # (well, turned_on, callback)
        self._pending: tuple | None = None       # (well, turn_on, cb, event)
        self.click_delay_s = self.CLICK_DELAY_S
        self.view_center_mm: tuple[float, float] | None = None
        self.live_texture = None
        self.live_corners_mm: list[tuple[float, float]] | None = None
        self.snapshots: list[dict] = []
        self.on_point_double_click = None  # callable(x_mm, y_mm, well|None)
        # A double-click on the live image drawn on the map, at any zoom:
        # callable(fx, fy), the point as a fraction of the image from its
        # center (-0.5 .. 0.5, +x right, +y down in the image).
        self.on_live_double_click = None
        # Ctrl + wheel zooms. The wheel alone focuses (as on the camera
        # view) while wheel_focus is on, i.e. when the map is the main
        # view; otherwise it is left to the panel to scroll.
        self.on_focus_scroll = None        # callable(direction)
        self.wheel_focus = False
        # While point_click_mode is on, a single click reports the exact
        # point (on_point_click(x_mm, y_mm, well|None), mm from the A1
        # center) instead of switching a well; used to move or add
        # positions. A double-click still goes to the stage callbacks.
        self.on_point_click = None
        self.point_click_mode = False
        self._pending_point = None
        self._pan: dict | None = None
        self.bind(pos=self.redraw, size=self.redraw, zoom=self.redraw,
                  flip_x=self.redraw)

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
        # The size measured by the plate calibration, when there is one.
        if getattr(self, 'well_size_override', None):
            return self.well_size_override
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
        self.flagged_wells = set()
        self._last_toggle = None
        self.view_center_mm = None
        self.live_corners_mm = None
        self.snapshots = []
        if self.zoom != 1.0:
            self.zoom = 1.0        # redraws
        else:
            self.redraw()

    # ------------------------------------------------------------ mapping

    def _fit_size(self) -> tuple[float, float]:
        """Plate size on screen at zoom 1 (whole plate visible)."""
        pad = 8.0
        pl, pw = self._plate_dims()
        avail_w = max(self.width - 2 * pad, 10.0)
        avail_h = max(self.height - 2 * pad, 10.0)
        aspect = pl / pw
        if avail_w / avail_h > aspect:
            return avail_h * aspect, avail_h
        return avail_w, avail_w / aspect

    def _plate_draw_rect(self) -> tuple[float, float, float, float]:
        """(x, y, w, h) of the plate outline, keeping the physical aspect
        ratio. At zoom 1 it is centered in the widget; zoomed in, it is
        placed so that ``view_center_mm`` is in the middle (most of the
        plate is then outside the widget and clipped)."""
        w, h = self._fit_size()
        w, h = w * self.zoom, h * self.zoom
        if self.zoom <= 1.0 or self.view_center_mm is None:
            return (self.x + (self.width - w) / 2.0,
                    self.y + (self.height - h) / 2.0, w, h)
        pl, pw = self._plate_dims()
        cx, cy = self.view_center_mm
        return (self.center_x - self._vx(cx) / pl * w,
                self.center_y - h + cy / pw * h, w, h)

    def _vx(self, mm_x: float) -> float:
        """Plate x as drawn (from the left edge of the picture); its own
        inverse."""
        return self._plate_dims()[0] - mm_x if self.flip_x else mm_x

    # --------------------------------------------------------- zoom / pan

    @property
    def precise(self) -> bool:
        """True when a well is drawn large enough to point inside it."""
        pl, _ = self._plate_dims()
        return self._well_size()[0] / pl * self._plate_draw_rect()[2] \
            >= self.PRECISE_WELL_PX

    def _set_view_center(self, mm_x: float, mm_y: float):
        pl, pw = self._plate_dims()
        self.view_center_mm = (min(max(mm_x, 0.0), pl),
                               min(max(mm_y, 0.0), pw))

    def set_view(self, zoom: float, center_mm=None):
        """Set zoom and (optionally) the plate point shown in the middle."""
        zoom = min(max(float(zoom), 1.0), self.MAX_ZOOM)
        if center_mm is not None:
            self._set_view_center(*center_mm)
        if zoom <= 1.0:
            self.view_center_mm = None
        elif self.view_center_mm is None:
            pl, pw = self._plate_dims()
            self.view_center_mm = (pl / 2.0, pw / 2.0)
        if zoom != self.zoom:
            self.zoom = zoom       # redraws
        else:
            self.redraw()

    def zoom_by(self, factor: float, anchor=None):
        """Zoom in (>1) or out (<1). ``anchor`` is the screen point that
        stays put (the cursor); while following the stage, or with no
        anchor, the middle of the view stays put."""
        new = min(max(self.zoom * factor, 1.0), self.MAX_ZOOM)
        if self.follow_stage and self.marker_plate_mm is not None:
            self.set_view(new, self.marker_plate_mm)
            return
        if anchor is None:
            anchor = self.center
        mm = self._screen_to_mm(*anchor)
        fw, fh = self._fit_size()
        pl, pw = self._plate_dims()
        side = -1.0 if self.flip_x else 1.0
        cx = mm[0] - side * (anchor[0] - self.center_x) / (fw * new / pl)
        cy = mm[1] + (anchor[1] - self.center_y) / (fh * new / pw)
        self.set_view(new, (cx, cy))

    def zoom_fit(self):
        """Show the whole plate."""
        self.set_view(1.0)

    def zoom_to_span(self, center_mm, span_mm: float):
        """Center on a plate point with ``span_mm`` filling the smaller
        side of the widget."""
        fw, _ = self._fit_size()
        pl, _ = self._plate_dims()
        px_per_mm = fw / pl
        side = max(min(self.width, self.height), 10.0)
        self.set_view(side / (max(span_mm, 1e-3) * px_per_mm), center_mm)

    def zoom_to_stage(self, fields: float = 4.0):
        """Center on the stage position, with the camera field of view
        about 1/``fields`` of the view (one well if the field of view is
        not known)."""
        if self.marker_plate_mm is None:
            return
        if self.live_corners_mm:
            xs = [p[0] for p in self.live_corners_mm]
            ys = [p[1] for p in self.live_corners_mm]
            span = max(max(xs) - min(xs), max(ys) - min(ys)) * fields
        else:
            span = max(self._well_size()) * 1.6
        self.zoom_to_span(self.marker_plate_mm, span)

    def set_live_image(self, texture, corners_rel_mm):
        """Show the camera image on the plate. ``corners_rel_mm``: where
        the image's top-left, top-right, bottom-right and bottom-left
        corners are on the plate, relative to the A1 center (None hides
        the image). Does not redraw: follow with set_marker_plate_mm."""
        if texture is None or not corners_rel_mm:
            self.live_texture = self.live_corners_mm = None
            return
        lw = self._lw()
        self.live_texture = texture
        self.live_corners_mm = [(lw.a1_offset_x + x, lw.a1_offset_y + y)
                                for x, y in corners_rel_mm]

    @staticmethod
    def selected_line_width(well_radius_px: float) -> float:
        """Half-thickness of the green line around a selected well: thick
        (3 px each side), thinner on wells drawn very small so the line
        does not fill them."""
        return float(min(3.0, max(1.5, 0.3 * well_radius_px)))

    def live_image_fraction(self, mm_x: float, mm_y: float):
        """Where a plate point is in the live image: (fx, fy) from the
        image center as a fraction of its width and height (+y down the
        image), or None when the point is outside it or no image is
        shown."""
        c = self.live_corners_mm
        if not c or len(c) != 4:
            return None
        (tlx, tly), (trx, try_), _br, (blx, bly) = c
        ux, uy = trx - tlx, try_ - tly          # along the top edge
        vx, vy = blx - tlx, bly - tly           # down the left edge
        det = ux * vy - uy * vx
        if abs(det) < 1e-12:
            return None
        px, py = mm_x - tlx, mm_y - tly
        a = (px * vy - py * vx) / det
        b = (ux * py - uy * px) / det
        if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
            return None
        return a - 0.5, b - 0.5

    # Preview images left on the plate wherever an image was taken or the
    # live view rested. A new one replaces those it lands on.
    MAX_SNAPSHOTS = 600

    def add_snapshot(self, rgb: np.ndarray, corners_rel_mm) -> bool:
        """Leave a preview on the plate. ``rgb``: uint8 (h, w, 3), first
        row = top of the image. ``corners_rel_mm``: its top-left,
        top-right, bottom-right and bottom-left corners on the plate,
        relative to the A1 center. Must be called on the UI thread."""
        if rgb is None or not corners_rel_mm or rgb.ndim != 3:
            return False
        lw = self._lw()
        corners = [(lw.a1_offset_x + x, lw.a1_offset_y + y)
                   for x, y in corners_rel_mm]
        cx = sum(p[0] for p in corners) / 4.0
        cy = sum(p[1] for p in corners) / 4.0
        # Size of the image on the plate: an older preview whose center
        # is within a quarter of it is the same place and is replaced.
        span = min(math.hypot(corners[1][0] - corners[0][0],
                              corners[1][1] - corners[0][1]),
                   math.hypot(corners[3][0] - corners[0][0],
                              corners[3][1] - corners[0][1]))
        h, w = rgb.shape[:2]
        try:
            tex = Texture.create(size=(w, h), colorfmt='rgb')
            tex.blit_buffer(np.ascontiguousarray(rgb[::-1]).tobytes(),
                            colorfmt='rgb', bufferfmt='ubyte')
        except Exception:
            return False
        self.snapshots = [s for s in self.snapshots if math.hypot(
            s['center'][0] - cx, s['center'][1] - cy) > span * 0.25]
        self.snapshots.append({'center': (cx, cy), 'corners': corners,
                               'texture': tex})
        del self.snapshots[:-self.MAX_SNAPSHOTS]
        self.redraw()
        return True

    def clear_snapshots(self):
        self.snapshots = []
        self.redraw()

    def _mm_to_screen(self, mm_x: float, mm_y: float) -> tuple[float, float]:
        """Plate mm (origin = plate top-left corner, +y down) -> screen."""
        x, y, w, h = self._plate_draw_rect()
        pl, pw = self._plate_dims()
        sx = x + (self._vx(mm_x) / pl) * w
        sy = y + h - (mm_y / pw) * h
        return sx, sy

    def _screen_to_mm(self, sx: float, sy: float) -> tuple[float, float]:
        x, y, w, h = self._plate_draw_rect()
        pl, pw = self._plate_dims()
        mm_x = self._vx((sx - x) / w * pl)
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
        if self.follow_stage and self.zoom > 1.0 and self._pan is None:
            self._set_view_center(*self.marker_plate_mm)
        self.redraw()

    def on_follow_stage(self, instance, value):
        if value and self.marker_plate_mm is not None and self.zoom > 1.0:
            self._set_view_center(*self.marker_plate_mm)
            self.redraw()

    def set_target_plate_mm(self, plate_mm: tuple[float, float] | None):
        """Show/clear the guided calibration target (plate top-left origin).
        A view zoomed in on the stage (the large map) is zoomed out as far
        as it takes to have the target in it too."""
        self.target_plate_mm = plate_mm
        if plate_mm is not None:
            self.keep_in_view(plate_mm)
        self.redraw()

    TARGET_RIM_PX = 26.0       # an out-of-view target is drawn this far in

    def keep_in_view(self, mm: tuple[float, float]):
        """Zoom out (never in) until the plate point ``mm`` is inside the
        view, keeping the middle of the view where it is."""
        if self.zoom <= 1.0 or self.view_center_mm is None \
                or self.width < 4 * self.TARGET_RIM_PX \
                or self.height < 4 * self.TARGET_RIM_PX:
            return
        sx, sy = self._mm_to_screen(*mm)
        dx, dy = abs(sx - self.center_x), abs(sy - self.center_y)
        half_w = self.width / 2.0 - self.TARGET_RIM_PX
        half_h = self.height / 2.0 - self.TARGET_RIM_PX
        need = max(dx / half_w, dy / half_h)
        if need > 1.0:
            self.set_view(self.zoom / need, self.view_center_mm)

    def target_on_screen(self) -> tuple[float, float, bool] | None:
        """Where the calibration target is drawn: at its own place, or,
        when that is outside the view, at the nearest point just inside
        the rim (third value True: it is then drawn with an arrow)."""
        if self.target_plate_mm is None:
            return None
        sx, sy = self._mm_to_screen(*self.target_plate_mm)
        m = min(self.TARGET_RIM_PX, self.width / 4.0, self.height / 4.0)
        cx = min(max(sx, self.x + m), self.right - m)
        cy = min(max(sy, self.y + m), self.top - m)
        return cx, cy, (abs(cx - sx) > 0.5 or abs(cy - sy) > 0.5)

    def set_position_markers(self, plate_mm_list: list[tuple[float, float]],
                             highlight: int | None = None):
        """Show the protocol interaction positions (plate top-left origin)."""
        self.position_markers = list(plate_mm_list)
        self.highlight_marker = highlight
        self.redraw()

    def set_focus_markers(self, marks, highlight: int | None = None):
        """Show the focus map points ((x_mm, y_mm, moved), plate top-left
        origin); an empty list hides them."""
        marks = [(float(m[0]), float(m[1]), bool(m[2])) for m in marks]
        if marks == self.focus_markers and highlight == self.highlight_focus:
            return
        self.focus_markers = marks
        self.highlight_focus = highlight
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
        left, right = self.x, self.right
        bottom, top = self.y, self.top
        with self.canvas:
            # Clip to the widget: a zoomed-in plate is larger than it.
            StencilPush()
            Rectangle(pos=self.pos, size=self.size)
            StencilUse()
            # Plate body
            Color(0.13, 0.15, 0.19, 1)
            Rectangle(pos=(x, y), size=(w, h))
            Color(0.45, 0.5, 0.58, 1)
            Line(rectangle=(x, y, w, h), width=1.2)
            # A1 corner chamfer marker (top-left; top-right when mirrored)
            if self.flip_x:
                Line(points=[x + w, y + h - 14, x + w - 14, y + h], width=1.2)
            else:
                Line(points=[x, y + h - 14, x + 14, y + h], width=1.2)
            # Previews left behind, oldest first (newer ones on top)
            for s in self.snapshots:
                pts = [self._mm_to_screen(*p) for p in s['corners']]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if (max(xs) < left or min(xs) > right or
                        max(ys) < bottom or min(ys) > top):
                    continue
                tl, tr, br, bl = pts
                tc = s['texture'].tex_coords       # BL, BR, TR, TL
                Color(1, 1, 1, 1)
                Mesh(vertices=[bl[0], bl[1], tc[0], tc[1],
                               br[0], br[1], tc[2], tc[3],
                               tr[0], tr[1], tc[4], tc[5],
                               tl[0], tl[1], tc[6], tc[7]],
                     indices=[0, 1, 2, 3], mode='triangle_fan',
                     texture=s['texture'])
            # Live camera image, at its true size and place on the plate
            box = None
            if self.live_texture is not None and self.live_corners_mm:
                tl, tr, br, bl = [self._mm_to_screen(*p)
                                  for p in self.live_corners_mm]
                tc = self.live_texture.tex_coords   # BL, BR, TR, TL
                Color(1, 1, 1, 1)
                Mesh(vertices=[bl[0], bl[1], tc[0], tc[1],
                               br[0], br[1], tc[2], tc[3],
                               tr[0], tr[1], tc[4], tc[5],
                               tl[0], tl[1], tc[6], tc[7]],
                     indices=[0, 1, 2, 3], mode='triangle_fan',
                     texture=self.live_texture)
                box = [*tl, *tr, *br, *bl, *tl]
            # Wells
            for r in range(lw.rows):
                for c in range(lw.cols):
                    mmx, mmy = self._well_center_mm(r, c)
                    sx, sy = self._mm_to_screen(mmx, mmy)
                    if (sx + well_rx < left or sx - well_rx > right or
                            sy + well_ry < bottom or sy - well_ry > top):
                        continue        # outside the zoomed view
                    if (r, c) in self.flagged_wells:
                        Color(1.0, 0.3, 0.25, 1)
                        Line(points=[sx - well_rx, sy - well_ry,
                                     sx + well_rx, sy + well_ry], width=1.4)
                        Line(points=[sx - well_rx, sy + well_ry,
                                     sx + well_rx, sy - well_ry], width=1.4)
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
                    if (r, c) in self.active_wells:
                        # Selected for the experiment: a thick green line
                        # along the well edge (no fill: the previews and
                        # the live image inside stay visible).
                        w = self.selected_line_width(min(well_rx, well_ry))
                        Color(0.15, 0.9, 0.4, 1)
                        if shape == 'square':
                            Line(rectangle=(sx - well_rx, sy - well_ry,
                                            2 * well_rx, 2 * well_ry),
                                 width=w, joint='miter')
                        else:
                            Line(ellipse=(sx - well_rx, sy - well_ry,
                                          2 * well_rx, 2 * well_ry), width=w)
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
            # Focus map points: a square with a cross (white once moved
            # off the place it was generated at; larger for the one the
            # focus walk is at).
            for i, (fx, fy, moved) in enumerate(self.focus_markers):
                sx, sy = self._mm_to_screen(fx, fy)
                if sx < left - 12 or sx > right + 12 or \
                        sy < bottom - 12 or sy > top + 12:
                    continue
                r = 8 if i == self.highlight_focus else 5
                Color(0, 0, 0, 0.75)
                Line(rectangle=(sx - r, sy - r, 2 * r, 2 * r), width=2.6)
                if moved:
                    Color(1.0, 1.0, 1.0, 1)
                else:
                    Color(0.25, 0.85, 1.0, 1)
                Line(rectangle=(sx - r, sy - r, 2 * r, 2 * r), width=1.3)
                Line(points=[sx - r, sy, sx + r, sy], width=1.1)
                Line(points=[sx, sy - r, sx, sy + r], width=1.1)
            # Stage position marker: the field-of-view outline, plus the
            # cross and circle while the outline is too small to see.
            if box is not None:
                Color(0.15, 1.0, 0.35, 1)
                Line(points=box, width=1.1)
            if self.marker_plate_mm is not None:
                sx, sy = self._mm_to_screen(*self.marker_plate_mm)
                Color(0.15, 1.0, 0.35, 1)
                small = box is None or max(
                    abs(box[4] - box[0]), abs(box[5] - box[1])) < 40
                if small:
                    Line(points=[sx - 7, sy, sx + 7, sy], width=1.4)
                    Line(points=[sx, sy - 7, sx, sy + 7], width=1.4)
                    Line(circle=(sx, sy, 5), width=1.2)
                else:
                    Line(points=[sx - 5, sy, sx + 5, sy], width=1.0)
                    Line(points=[sx, sy - 5, sx, sy + 5], width=1.0)
            # Calibration target marker
            # (never out of sight: outside the view it is drawn at the rim
            # with an arrow toward where it is; a dark edge keeps it
            # readable over the live image)
            target = self.target_on_screen()
            if target is not None:
                sx, sy, outside = target
                marks = [[sx - 12, sy, sx - 4, sy], [sx + 4, sy, sx + 12, sy],
                         [sx, sy - 12, sx, sy - 4], [sx, sy + 4, sx, sy + 12]]
                if outside:
                    tx, ty = self._mm_to_screen(*self.target_plate_mm)
                    d = max(((tx - sx) ** 2 + (ty - sy) ** 2) ** 0.5, 1e-6)
                    ux, uy = (tx - sx) / d, (ty - sy) / d
                    tip = (sx + 20 * ux, sy + 20 * uy)
                    marks.append([sx + 9 * ux, sy + 9 * uy, *tip])
                    for side in (1.0, -1.0):
                        marks.append([*tip,
                                      tip[0] - 7 * ux + side * 5 * uy,
                                      tip[1] - 7 * uy - side * 5 * ux])
                for colour, extra in (((0, 0, 0, 0.8), 1.6),
                                      ((1.0, 0.35, 0.25, 1), 0.0)):
                    Color(*colour)
                    Line(circle=(sx, sy, 8), width=1.6 + extra)
                    for pts in marks:
                        Line(points=pts, width=1.6 + extra)
            StencilUnUse()
            Rectangle(pos=self.pos, size=self.size)
            StencilPop()

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
        button = getattr(touch, 'button', None) or ''
        if button.startswith('scroll'):
            if 'ctrl' in self._modifiers():
                # Kivy reports wheel-forward as 'scrolldown': zoom in.
                self.zoom_by(self.ZOOM_STEP if button == 'scrolldown'
                             else 1.0 / self.ZOOM_STEP, anchor=touch.pos)
                return True
            if self.wheel_focus and self.on_focus_scroll is not None:
                # Same direction as on the camera view.
                self.on_focus_scroll(1 if button == 'scrolldown' else -1)
                return True
            return False
        if button in ('right', 'middle'):
            if self.zoom > 1.0 and self.view_center_mm is not None:
                self._pan = {'start': touch.pos,
                             'center': self.view_center_mm}
                touch.grab(self)
            return True
        mm = self._screen_to_mm(touch.x, touch.y)
        well = self._nearest_well(*mm)

        if touch.is_double_tap and self._pending_point is not None:
            # The first click of this double-click was not a point edit.
            self._pending_point.cancel()
            self._pending_point = None
        if self.point_click_mode and self.on_point_click is not None \
                and not touch.is_double_tap:
            pl, pw = self._plate_dims()
            if 0.0 <= mm[0] <= pl and 0.0 <= mm[1] <= pw:
                lw = self._lw()
                args = (mm[0] - lw.a1_offset_x, mm[1] - lw.a1_offset_y, well)
                if self._pending_point is not None:
                    self._pending_point.cancel()
                    self._pending_point = None
                if self.click_delay_s > 0:
                    from kivy.clock import Clock

                    def _fire(dt, args=args):
                        self._pending_point = None
                        self.on_point_click(*args)
                    self._pending_point = Clock.schedule_once(
                        _fire, self.click_delay_s)
                else:
                    self.on_point_click(*args)
            return True

        if touch.is_double_tap and self.on_live_double_click is not None:
            # On the live image: bring that pixel to the center, whatever
            # the zoom (never the well center), by the image itself.
            frac = self.live_image_fraction(*mm)
            if frac is not None:
                if self._pending is not None:
                    self._pending[3].cancel()
                    self._pending = None
                elif self._last_toggle and self._last_toggle[0] == well:
                    self._last_toggle[2]([well], not self._last_toggle[1])
                self._last_toggle = None
                self.on_live_double_click(*frac)
                return True

        if touch.is_double_tap:
            if self.precise and self.on_point_double_click is not None:
                # Zoomed in: go to the exact point, not the well center.
                pl, pw = self._plate_dims()
                if 0.0 <= mm[0] <= pl and 0.0 <= mm[1] <= pw:
                    if self._pending is not None:
                        self._pending[3].cancel()
                        self._pending = None
                    elif self._last_toggle and self._last_toggle[0] == well:
                        self._last_toggle[2]([well], not self._last_toggle[1])
                    self._last_toggle = None
                    lw = self._lw()
                    self.on_point_double_click(mm[0] - lw.a1_offset_x,
                                               mm[1] - lw.a1_offset_y, well)
                return True
            if well is None:
                return True
            # A double-click only moves the stage: drop the first click's
            # pending toggle, or undo it if it has already been applied.
            if self._pending is not None:
                self._pending[3].cancel()
                self._pending = None
            elif self._last_toggle and self._last_toggle[0] == well:
                self._last_toggle[2]([well], not self._last_toggle[1])
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

        self._fire_pending()        # an earlier click on another well
        marking = self.mark_mode and self.on_wells_flag is not None
        current = self.flagged_wells if marking else self.active_wells
        mods = self._modifiers()
        if 'shift' in mods:
            remove = False
        elif 'ctrl' in mods:
            remove = True
        else:
            remove = well is not None and well in current
        self._drag = {'start': touch.pos, 'cur': touch.pos, 'well': well,
                      'remove': remove, 'moved': False,
                      'cb': self.on_wells_flag if marking
                      else self.on_wells_toggle}
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is self and self._pan is not None:
            fw, fh = self._fit_size()
            pl, pw = self._plate_dims()
            p = self._pan
            self.follow_stage = False
            self._set_view_center(
                p['center'][0] - (-1.0 if self.flip_x else 1.0)
                * (touch.x - p['start'][0]) / (fw * self.zoom / pl),
                p['center'][1] + (touch.y - p['start'][1])
                / (fh * self.zoom / pw))
            self.redraw()
            return True
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
        if self._pan is not None:
            self._pan = None
            return True
        d, self._drag = self._drag, None
        self.canvas.after.clear()
        if d is None:
            return True
        if d['moved']:
            wells = self.wells_in_screen_rect(*d['start'], *d['cur'])
            self._last_toggle = None
            if wells:
                d['cb'](wells, not d['remove'])
        elif d['well'] is not None:
            well = d['well']
            self._select(well)
            turn_on = not d['remove']
            if self.click_delay_s > 0:
                from kivy.clock import Clock
                self._pending = (well, turn_on, d['cb'], Clock.schedule_once(
                    lambda dt: self._fire_pending(), self.click_delay_s))
            else:
                self._last_toggle = (well, turn_on, d['cb'])
                d['cb']([well], turn_on)
        return True

    def _fire_pending(self):
        """Apply the single click that was waiting out the double-click
        window."""
        p, self._pending = self._pending, None
        if p is not None:
            p[3].cancel()
            self._last_toggle = (p[0], p[1], p[2])
            p[2]([p[0]], p[1])


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
