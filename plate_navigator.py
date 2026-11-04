from __future__ import annotations
import math
import numpy as np
from kivy.app import App
from kivy.graphics import Color, Line, Rectangle, Ellipse
from kivy.graphics.texture import Texture
from kivy.properties import BooleanProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup


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
                             significant_bits: int = 8):
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
        if self.collide_point(*touch.pos) and touch.is_double_tap:
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
    position. Single click selects a well; double-click fires
    ``on_well_double_click(row, col, plate_x_mm, plate_y_mm)`` where the
    plate coordinates are the well center relative to the A1 center
    (+x toward the last column, +y toward the last row).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = WellPlateFormat(num_wells=96)
        self.labware = None            # LabwareDef overrides plate_format
        self.selected_well: tuple[int, int] | None = None
        self.marker_plate_mm: tuple[float, float] | None = None
        self.target_plate_mm: tuple[float, float] | None = None  # calibration
        self.on_well_double_click = None  # callable(row, col, x_mm, y_mm)
        self.on_well_selected = None      # callable(row, col, x_mm, y_mm)
        self.bind(pos=self.redraw, size=self.redraw)

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

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        mm_x, mm_y = self._screen_to_mm(touch.x, touch.y)
        well = self._nearest_well(mm_x, mm_y)
        if well is None:
            return True
        row, col = well
        sx, sy = self._spacing()
        rel_x_mm = col * sx   # relative to A1 center
        rel_y_mm = row * sy
        self.selected_well = well
        self.redraw()
        if touch.is_double_tap:
            if self.on_well_double_click:
                self.on_well_double_click(row, col, rel_x_mm, rel_y_mm)
        else:
            if self.on_well_selected:
                self.on_well_selected(row, col, rel_x_mm, rel_y_mm)
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
