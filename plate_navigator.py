from __future__ import annotations
import math
import sys
import numpy as np
from kivy.app import App
from kivy.graphics import Color, Line, Rectangle, Ellipse
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup

class WellPlateFormat:
    PLATE_LENGTH_MM = 127.76
    PLATE_WIDTH_MM = 85.48

    def __init__(self, num_wells: int = 96):
        self.num_wells = num_wells
        if num_wells == 96:
            self.rows = 8
            self.cols = 12
            self.well_spacing_mm = 9.0
            self.well_diameter = 6.4
            self.a1_offset_x = 14.38
            self.a1_offset_y = 11.24
        else:
            self.rows = 8
            self.cols = 12
            self.well_spacing_mm = 9.0
            self.well_diameter = 6.4
            self.a1_offset_x = 14.38
            self.a1_offset_y = 11.24

    def get_well_name(self, row: int, col: int) -> str:
        return f"{chr(65 + row)}{col + 1}"

    def get_nominal_position(self, row: int, col: int) -> tuple[float, float]:
        x_mm = col * self.well_spacing_mm
        y_mm = row * self.well_spacing_mm
        return x_mm, y_mm


class NavigatorPlateCalibration:
    def __init__(self, plate_format: WellPlateFormat):
        self.plate_format = plate_format


class FocusMap:
    def __init__(self):
        self.points = {}

    def add_point(self, well_name: str, x_mm: float, y_mm: float, z_mm: float):
        self.points[well_name] = {'x': x_mm, 'y': y_mm, 'z': z_mm}

    def clear(self):
        self.points.clear()


class PlateFOVManager:
    def __init__(self, plate_format: WellPlateFormat, calib: NavigatorPlateCalibration, focus_map: FocusMap):
        self.plate_format = plate_format
        self.calib = calib
        self.focus_map = focus_map
        self.autofocus_points = []

    def add_autofocus_point(self, well_name: str):
        if well_name not in self.autofocus_points:
            self.autofocus_points.append(well_name)


try:
    import labware
    if not hasattr(labware, 'WellPlateFormat'):
        setattr(labware, 'WellPlateFormat', WellPlateFormat)
    if not hasattr(labware, 'NavigatorPlateCalibration'):
        setattr(labware, 'NavigatorPlateCalibration', NavigatorPlateCalibration)
    if not hasattr(labware, 'FocusMap'):
        setattr(labware, 'FocusMap', FocusMap)
    if not hasattr(labware, 'PlateFOVManager'):
        setattr(labware, 'PlateFOVManager', PlateFOVManager)
except ImportError:
    pass


class ZoomPanCameraWidget(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        self.bind(pos=self.redraw_overlay, size=self.redraw_overlay)
        self.set_standby_texture()

    def set_standby_texture(self):
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = [15, 18, 24]
        img[h // 2 - 1:h // 2 + 2, :] = [50, 70, 90]
        img[:, w // 2 - 1:w // 2 + 2] = [50, 70, 90]

        tex = Texture.create(size=(w, h), colorfmt='rgb')
        tex.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        tex.flip_vertical()
        self.texture = tex

    def update_frame_texture(self, frame_np: np.ndarray):
        if frame_np is None or not isinstance(frame_np, np.ndarray) or frame_np.size == 0:
            return

        try:
            if frame_np.dtype == np.uint16:
                f_min = float(frame_np.min())
                f_max = float(frame_np.max())
                if f_max - f_min > 20.0:
                    frame_8u = np.clip((frame_np.astype(np.float32) - f_min) / (f_max - f_min) * 255.0, 0, 255).astype(np.uint8)
                else:
                    frame_8u = np.clip(frame_np >> 4, 0, 255).astype(np.uint8)
            else:
                frame_8u = np.clip(frame_np, 0, 255).astype(np.uint8)

            if len(frame_8u.shape) == 2:
                frame_rgb = np.stack([frame_8u] * 3, axis=-1)
            elif len(frame_8u.shape) == 3 and frame_8u.shape[2] == 1:
                frame_rgb = np.repeat(frame_8u, 3, axis=-1)
            elif len(frame_8u.shape) == 3 and frame_8u.shape[2] == 4:
                frame_rgb = frame_8u[:, :, :3]
            else:
                frame_rgb = frame_8u

            h, w, c = frame_rgb.shape
            buf = frame_rgb.tobytes()

            tex = Texture.create(size=(w, h), colorfmt='rgb')
            tex.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            tex.flip_vertical()
            self.texture = tex
            self.canvas.ask_update()
        except Exception as e:
            print(f'[ZoomPanCameraWidget] Error blitting frame: {e}', flush=True)

    def redraw_overlay(self, *args):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(1.0, 0.2, 0.2, 0.7)
            cx, cy = self.center
            Line(points=[cx - 20, cy, cx + 20, cy], width=1.5)
            Line(points=[cx, cy - 20, cx, cy + 20], width=1.5)


class InteractiveScaledPlateCanvas(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = WellPlateFormat(num_wells=96)
        self.calibration = NavigatorPlateCalibration(self.plate_format)
        self.focus_map = FocusMap()
        self.fov_manager = PlateFOVManager(self.plate_format, self.calibration, self.focus_map)
        self.bind(pos=self.redraw_canvas, size=self.redraw_canvas)

    def redraw_canvas(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        margin = 20.0
        avail_w = self.width - 2 * margin
        avail_h = self.height - 2 * margin

        plate_w_mm = self.plate_format.PLATE_LENGTH_MM
        plate_h_mm = self.plate_format.PLATE_WIDTH_MM

        scale_x = avail_w / plate_w_mm
        scale_y = avail_h / plate_h_mm
        scale = min(scale_x, scale_y)

        draw_w = plate_w_mm * scale
        draw_h = plate_h_mm * scale

        offset_x = self.x + (self.width - draw_w) / 2.0
        offset_y = self.y + (self.height + draw_h) / 2.0

        with self.canvas:
            Color(0.20, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.50, 0.55, 0.65, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = (self.plate_format.well_diameter / 2.0) * scale

            for r in range(self.plate_format.rows):
                for c in range(self.plate_format.cols):
                    w_name = self.plate_format.get_well_name(r, c)
                    x_nom, y_nom = self.plate_format.get_nominal_position(r, c)

                    wx_px = offset_x + (self.plate_format.a1_offset_x + x_nom) * scale
                    wy_px = offset_y - (self.plate_format.a1_offset_y + y_nom) * scale

                    if w_name in self.focus_map.points:
                        Color(0.2, 0.8, 0.2, 0.9)
                    elif w_name in self.fov_manager.autofocus_points:
                        Color(0.9, 0.8, 0.2, 0.9)
                    else:
                        Color(0.12, 0.14, 0.18, 0.8)

                    Ellipse(pos=(wx_px - well_r_px, wy_px - well_r_px), size=(2 * well_r_px, 2 * well_r_px))
                    Color(0.35, 0.38, 0.45, 1.0)
                    Line(ellipse=(wx_px - well_r_px, wy_px - well_r_px, 2 * well_r_px, 2 * well_r_px), width=1.0)

            app = App.get_running_app()
            if app and hasattr(app, 'stage_x') and hasattr(app, 'stage_y'):
                stg_x = app.stage_x
                stg_y = app.stage_y

                norm_x = (stg_x - (-plate_w_mm / 2.0)) / plate_w_mm
                norm_y = ((plate_h_mm / 2.0) - stg_y) / plate_h_mm

                cur_px_x = offset_x + norm_x * draw_w
                cur_px_y = offset_y - norm_y * draw_h

                box_w_px = max(6.0, (0.60 / plate_w_mm) * draw_w)
                box_h_px = max(4.0, (0.45 / plate_h_mm) * draw_h)

                Color(0.0, 0.9, 1.0, 0.9)
                Line(rectangle=(cur_px_x - box_w_px / 2.0, cur_px_y - box_h_px / 2.0, box_w_px, box_h_px), width=1.5)

                Color(1.0, 0.2, 0.2, 1.0)
                Ellipse(pos=(cur_px_x - 3, cur_px_y - 3), size=(6, 6))


class PlateNavigatorPopup(Popup):
    def __init__(self, app_controller=None, **kwargs):
        super().__init__(**kwargs)
        self.app_controller = app_controller
        self.title = "Interactive Well Plate Navigator & Protocol Studio"
        self.size_hint = (0.95, 0.95)


def attach_navigator_to_lumaviewpro(app_instance):
    plate_fmt = WellPlateFormat(num_wells=96)
    calib = NavigatorPlateCalibration(plate_fmt)
    focus_m = FocusMap()
    fov_m = PlateFOVManager(plate_fmt, calib, focus_m)

    app_instance.plate_format = plate_fmt
    app_instance.plate_calibration = calib
    app_instance.focus_map = focus_m
    app_instance.fov_manager = fov_m
