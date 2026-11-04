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
            f_min = float(frame_np.min())
            f_max = float(frame_np.max())

            if f_max > f_min:
                frame_norm = ((frame_np.astype(np.float32) - f_min) / (f_max - f_min)) * 255.0
                frame_8u = np.clip(frame_norm, 0, 255).astype(np.uint8)
            else:
                frame_8u = np.clip(frame_np, 0, 255).astype(np.uint8)

            if len(frame_8u.shape) == 2:
                frame_rgb = np.stack([frame_8u] * 3, axis=-1)
            else:
                frame_rgb = frame_8u

            h, w = frame_rgb.shape[0], frame_rgb.shape[1]

            if self.texture is None or self.texture.size != (w, h):
                self.texture = Texture.create(size=(w, h), colorfmt='rgb')

            buf = np.ascontiguousarray(frame_rgb[::-1, :]).tobytes()
            self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.canvas.ask_update()
        except Exception as e:
            print(f'[ZoomPanCameraWidget] Error rendering frame texture: {e}', flush=True)

    def redraw_overlay(self, *args):
        pass


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


def attach_navigator_to_lumaviewpro(app_instance):
    plate_fmt = WellPlateFormat(num_wells=96)
    calib = NavigatorPlateCalibration(plate_fmt)
    focus_m = FocusMap()
    fov_m = PlateFOVManager(plate_fmt, calib, focus_m)

    app_instance.plate_format = plate_fmt
    app_instance.plate_calibration = calib
    app_instance.focus_map = focus_m
    app_instance.fov_manager = fov_m
