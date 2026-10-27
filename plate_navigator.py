"""
plate_navigator.py - Microplate Canvas & Live Viewport Extension for LumaViewPro.

Provides:
- ZoomPanCameraWidget: Viewport with zoom/pan, central red reticle, double-click pixel centering.
- InteractiveScaledPlateCanvas: Physically accurate microplate canvas with 6-point calibration targets,
  cyan active FOV box, red stage center dot, and double-click well/coordinate stage navigation.
- PlateNavigatorPopup: Studio modal popup window.
- attach_navigator_to_lumaviewpro(): Bridge helper function.
"""

from __future__ import annotations
import os
import sys
import math
import re
import json
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

import image_utils
import labware

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.properties import StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty


class ZoomPanCameraWidget(Widget):
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.texture = None
        self.bind(pos=self.redraw_canvas, size=self.redraw_canvas, zoom_level=self.redraw_canvas, pan_x=self.redraw_canvas, pan_y=self.redraw_canvas)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_canvas()

    def set_frame(self, frame_array: np.ndarray):
        if frame_array is None or frame_array.size == 0:
            return

        try:
            h, w = frame_array.shape[:2]
            flipped = np.ascontiguousarray(frame_array[::-1, :, :])
            
            texture = Texture.create(size=(w, h), colorfmt='bgr')
            texture.blit_buffer(flipped.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture
            self.redraw_canvas()
        except Exception as e:
            print(f"[ZoomPanCameraWidget set_frame error] {e}")

    def redraw_canvas(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
            Color(0.05, 0.05, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self.texture is not None:
                tw, th = self.texture.size
                scale_w = self.width / float(tw)
                scale_h = self.height / float(th)
                base_scale = min(scale_w, scale_h)

                draw_w = tw * base_scale * self.zoom_level
                draw_h = th * base_scale * self.zoom_level

                center_x = self.x + self.width / 2.0 + self.pan_x
                center_y = self.y + self.height / 2.0 + self.pan_y

                draw_x = center_x - draw_w / 2.0
                draw_y = center_y - draw_h / 2.0

                Color(1.0, 1.0, 1.0, 1.0)
                Rectangle(texture=self.texture, pos=(draw_x, draw_y), size=(draw_w, draw_h))

            # Draw Central Red Reticle Crosshairs
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.85)
            Line(points=[cx - 20, cy, cx + 20, cy], width=1.5)
            Line(points=[cx, cy - 20, cx, cy + 20], width=1.5)
            Line(circle=(cx, cy, 6), width=1.5)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        # Double-click pixel centering in live image viewport
        if touch.is_double_tap and touch.button == 'left':
            center_x = self.x + self.width / 2.0 + self.pan_x
            center_y = self.y + self.height / 2.0 + self.pan_y

            dx_px = touch.x - center_x
            dy_px = touch.y - center_y

            app = App.get_running_app()
            if app and hasattr(app, 'focus_map'):
                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.60)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)

                draw_w = max(1.0, self.width * self.zoom_level)
                draw_h = max(1.0, self.height * self.zoom_level)

                dx_mm = (dx_px / draw_w) * fov_w
                dy_mm = -(dy_px / draw_h) * fov_h

                if hasattr(app, 'jog_stage'):
                    app.jog_stage(dx_mm, dy_mm, 0.0)
                return True

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                self.zoom_level = max(0.5, self.zoom_level - 0.1)
            elif touch.button == 'scrollup':
                self.zoom_level = min(5.0, self.zoom_level + 0.1)
            return True

        if touch.button == 'right':
            self.is_panning = True
            self.last_touch_pos = touch.pos
            return True

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.is_panning:
            dx = touch.x - self.last_touch_pos[0]
            dy = touch.y - self.last_touch_pos[1]
            self.pan_x += dx
            self.pan_y += dy
            self.last_touch_pos = touch.pos
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if touch.button == 'right':
            self.is_panning = False
            return True
        return super().on_touch_up(touch)


class InteractiveScaledPlateCanvas(Widget):
    """
    Custom Kivy canvas widget rendering a physically accurate representation
    of the labware plate based on ANSI/SLAS standards and 2D Affine calibration matrix.
    Orientation: Well A1 is positioned at the TOP-LEFT corner.
    Features DOUBLE-CLICK navigation: Double-clicking any well or point on the map moves the FOV directly to that coordinate!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = labware.WellPlateFormat(num_wells=96)
        self.calibration = labware.NavigatorPlateCalibration(self.plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.plate_format, self.calibration, self.focus_map)
        self.bind(pos=self.redraw_canvas, size=self.redraw_canvas)

    def set_references(
        self,
        plate_format: labware.WellPlateFormat,
        calibration: labware.NavigatorPlateCalibration,
        focus_map: labware.FocusMap,
        fov_manager: labware.PlateFOVManager
    ):
        self.plate_format = plate_format
        self.calibration = calibration
        self.focus_map = focus_map
        self.fov_manager = fov_manager
        self.redraw_canvas()

    def redraw_canvas(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        margin = 25.0
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

        rot_angle = self.calibration.rotation_angle_deg if self.calibration.is_calibrated else 0.0

        with self.canvas:
            PushMatrix()
            if rot_angle != 0.0:
                Rotate(angle=rot_angle, origin=(offset_x + draw_w / 2.0, offset_y - draw_h / 2.0))

            # Outer plate boundary
            Color(0.2, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.5, 0.5, 0.6, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = (self.plate_format.well_diameter / 2.0) * scale

            # Draw Wells
            for r in range(self.plate_format.rows):
                for c in range(self.plate_format.cols):
                    w_name = self.plate_format.get_well_name(r, c)
                    x_nom, y_nom = self.plate_format.get_nominal_position(r, c)

                    wx_px = offset_x + (self.plate_format.a1_offset_x + x_nom) * scale
                    wy_px = offset_y - (self.plate_format.a1_offset_y + y_nom) * scale

                    is_selected = w_name in self.fov_manager.selected_wells

                    if is_selected:
                        Color(0.2, 0.6, 0.9, 0.6)
                        Ellipse(pos=(wx_px - well_r_px, wy_px - well_r_px), size=(2 * well_r_px, 2 * well_r_px))
                        Color(0.3, 0.8, 1.0, 1.0)
                        Line(ellipse=(wx_px - well_r_px, wy_px - well_r_px, 2 * well_r_px, 2 * well_r_px), width=1.5)
                    else:
                        Color(0.1, 0.12, 0.15, 0.8)
                        Ellipse(pos=(wx_px - well_r_px, wy_px - well_r_px), size=(2 * well_r_px, 2 * well_r_px))
                        Color(0.35, 0.38, 0.45, 1.0)
                        Line(ellipse=(wx_px - well_r_px, wy_px - well_r_px, 2 * well_r_px, 2 * well_r_px), width=1.0)

            # Draw 6 Calibration Targets ONLY when is_calibrating or is_calibrated is True
            if getattr(self.calibration, 'is_calibrating', False) or getattr(self.calibration, 'is_calibrated', False):
                cal_steps = self.calibration.CALIBRATION_STEPS
                for idx, step_info in enumerate(cal_steps):
                    step_id = step_info["id"]
                    is_recorded = step_id in self.calibration.recorded_points
                    is_active = (idx == self.calibration.current_step_index) and getattr(self.calibration, 'is_calibrating', False)

                    tx_px, ty_px = self._get_step_target_px(step_id, offset_x, offset_y, scale)

                    if is_recorded:
                        Color(0.2, 0.9, 0.3, 1.0)  # Green recorded point
                        Ellipse(pos=(tx_px - 5, ty_px - 5), size=(10, 10))
                        Line(circle=(tx_px, ty_px, 8), width=1.5)
                    elif is_active:
                        Color(1.0, 0.9, 0.1, 1.0)  # Active target ring
                        Ellipse(pos=(tx_px - 6, ty_px - 6), size=(12, 12))
                        Line(circle=(tx_px, ty_px, 10), width=1.8)
                    else:
                        Color(0.8, 0.8, 0.2, 0.5)
                        Ellipse(pos=(tx_px - 4, ty_px - 4), size=(8, 8))

            # Draw Active Stage FOV Position Reticle Indicator (Cyan Box & Red Dot)
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x') and hasattr(app, 'stage_y'):
                sx_px = offset_x + (app.stage_x + plate_w_mm / 2.0) * scale
                sy_px = offset_y - (plate_h_mm / 2.0 - app.stage_y) * scale

                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.60)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)
                fov_w_px = max(8.0, fov_w * scale)
                fov_h_px = max(6.0, fov_h * scale)

                Color(0.0, 1.0, 1.0, 0.9)  # Bright Cyan FOV Box
                Line(rectangle=(sx_px - fov_w_px / 2.0, sy_px - fov_h_px / 2.0, fov_w_px, fov_h_px), width=1.8)
                Color(1.0, 0.2, 0.2, 0.9)  # Bright Red Center Dot
                Ellipse(pos=(sx_px - 3, sy_px - 3), size=(6, 6))

            PopMatrix()

    def _get_step_target_px(self, step_id: str, offset_x: float, offset_y: float, scale: float) -> Tuple[float, float]:
        well_r_mm = self.plate_format.well_diameter / 2.0
        
        if step_id.startswith("A1_"):
            base_x = offset_x + self.plate_format.a1_offset_x * scale
            base_y = offset_y - self.plate_format.a1_offset_y * scale
            if step_id == "A1_12":
                return base_x, base_y + well_r_mm * scale
            elif step_id == "A1_3":
                return base_x + well_r_mm * scale, base_y
            elif step_id == "A1_6":
                return base_x, base_y - well_r_mm * scale
            elif step_id == "A1_9":
                return base_x - well_r_mm * scale, base_y

        elif step_id == "ROW_END_3":
            x_nom = (self.plate_format.cols - 1) * self.plate_format.pitch_x
            base_x = offset_x + (self.plate_format.a1_offset_x + x_nom + well_r_mm) * scale
            base_y = offset_y - self.plate_format.a1_offset_y * scale
            return base_x, base_y

        elif step_id == "COL_END_12":
            y_nom = (self.plate_format.rows - 1) * self.plate_format.pitch_y
            base_x = offset_x + self.plate_format.a1_offset_x * scale
            base_y = offset_y - (self.plate_format.a1_offset_y + y_nom + well_r_mm) * scale
            return base_x, base_y

        return offset_x, offset_y

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        margin = 25.0
        draw_w = self.plate_format.PLATE_LENGTH_MM
        draw_h = self.plate_format.PLATE_WIDTH_MM
        scale_x = (self.width - 2 * margin) / draw_w
        scale_y = (self.height - 2 * margin) / draw_h
        scale = min(scale_x, scale_y)

        offset_x = self.x + (self.width - draw_w * scale) / 2.0
        offset_y = self.y + (self.height + draw_h) / 2.0

        app = App.get_running_app()

        # DOUBLE CLICK ANYWHERE ON PLATE TEMPLATE -> Move FOV Center to clicked coordinate
        if touch.is_double_tap:
            rel_x_px = touch.x - (offset_x + self.plate_format.a1_offset_x * scale)
            rel_y_px = (offset_y - self.plate_format.a1_offset_y * scale) - touch.y

            a1_stage_x = -draw_w / 2.0 + self.plate_format.a1_offset_x
            a1_stage_y = draw_h / 2.0 - self.plate_format.a1_offset_y
            target_stage_x = a1_stage_x + rel_x_px / scale
            target_stage_y = a1_stage_y - rel_y_px / scale

            if app and hasattr(app, 'jog_stage_to'):
                app.jog_stage_to(target_stage_x, target_stage_y)
                self.redraw_canvas()
                return True

        # SINGLE CLICK ON WELL -> Toggle selection
        for r in range(self.plate_format.rows):
            for c in range(self.plate_format.cols):
                x_nom, y_nom = self.plate_format.get_nominal_position(r, c)
                wx_px = offset_x + (self.plate_format.a1_offset_x + x_nom) * scale
                wy_px = offset_y - (self.plate_format.a1_offset_y + y_nom) * scale
                well_r_px = (self.plate_format.well_diameter / 2.0) * scale

                dist = math.hypot(touch.x - wx_px, touch.y - wy_px)
                if dist <= well_r_px:
                    w_name = self.plate_format.get_well_name(r, c)
                    self.fov_manager.toggle_well_selected(w_name)
                    self.redraw_canvas()
                    return True

        return super().on_touch_down(touch)


class PlateNavigatorPopup(Popup):
    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app_ctrl = app_controller
        self.title = "LumaViewPro - Plate Navigator & Registration Studio"
        self.size_hint = (0.96, 0.96)


def attach_navigator_to_lumaviewpro(app_instance):
    if not hasattr(app_instance, 'well_plate_format'):
        app_instance.well_plate_format = labware.WellPlateFormat(num_wells=96)
    if not hasattr(app_instance, 'plate_calibration'):
        app_instance.plate_calibration = labware.NavigatorPlateCalibration(app_instance.well_plate_format)
    if not hasattr(app_instance, 'focus_map'):
        app_instance.focus_map = labware.FocusMap()
    if not hasattr(app_instance, 'fov_manager'):
        app_instance.fov_manager = labware.PlateFOVManager(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map
        )
    return None
