from __future__ import annotations
"""
plate_navigator.py - Modular Plate Navigator & Registration Studio Extension for LumaViewPro.

Provides:
- InteractiveScaledPlateCanvas: Physically accurate microplate canvas with Top-Left A1 orientation,
  active FOV cyan reticle box, 6-point target calibration overlays, and double-click stage navigation.
- ZoomPanCameraWidget: Interactive camera viewport with smooth mouse wheel zooming and right-click panning.
- PlateNavigatorPopup: Modal dialog window with 3 core tabs.
"""

import os
import sys
import math
import re
import json
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import image_utils
import labware

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.graphics.texture import Texture
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty, DictProperty
)

# Load plate_navigator.kv layout safely exactly once
_kv_file = os.path.join(script_dir, 'plate_navigator.kv')
if os.path.exists(_kv_file) and _kv_file not in Builder.files:
    Builder.load_file(_kv_file)


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - Mouse Wheel Zooming (Zoom In / Zoom Out)
    - Right-Click Drag Panning in X and Y
    - Reset View Button
    - Double-Click Pixel Centering: Double-clicking any pixel moves the stage
      so that point becomes the new center under reticle crosshairs.
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.current_texture = None
        self.bind(pos=self.redraw_viewport, size=self.redraw_viewport)

    def set_frame(self, frame_array: np.ndarray):
        """Converts in-memory BGR NumPy frame to Kivy Texture and renders on Canvas."""
        if frame_array is None or frame_array.size == 0:
            return

        try:
            h, w = frame_array.shape[:2]
            if frame_array.ndim == 3 and frame_array.shape[2] == 3:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            elif frame_array.ndim == 2:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
            else:
                return

            texture = Texture.create(size=(w, h), colorfmt='rgb')
            texture.blit_buffer(rgb_buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()

            self.current_texture = texture
            self.redraw_viewport()
        except Exception as e:
            print(f"[ZoomPanCameraWidget set_frame error] {e}")

    def redraw_viewport(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
            # Background
            Color(0.05, 0.05, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self.current_texture is not None:
                tw, th = self.current_texture.size
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
                Rectangle(texture=self.current_texture, pos=(draw_x, draw_y), size=(draw_w, draw_h))

            # Viewport Reticle Crosshairs at Center
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.85)
            Line(points=[cx - 15, cy, cx + 15, cy], width=1.5)
            Line(points=[cx, cy - 15, cx, cy + 15], width=1.5)
            Line(circle=(cx, cy, 8), width=1.2)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                self.zoom_level = max(0.5, self.zoom_level - 0.1)
            elif touch.button == 'scrollup':
                self.zoom_level = min(5.0, self.zoom_level + 0.1)
            return True

        # Double click image -> center clicked feature under crosshairs
        if touch.is_double_tap and touch.button == 'left':
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x'):
                center_x = self.x + self.width / 2.0
                center_y = self.y + self.height / 2.0

                dx_px = touch.x - center_x
                dy_px = touch.y - center_y

                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.60)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)

                dx_mm = (dx_px / self.width) * fov_w
                dy_mm = (dy_px / self.height) * fov_h

                app.stage_x += dx_mm
                app.stage_y += dy_mm
                if hasattr(app, 'clamp_stage_position'):
                    app.clamp_stage_position()
                if hasattr(app, 'update_live_feed'):
                    app.update_live_feed(0)
            return True

        if touch.button == 'right':
            self.is_panning = True
            self.last_touch_pos = touch.pos
            return True

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if hasattr(self, 'is_panning') and self.is_panning:
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
    Features DOUBLE-CLICK navigation: Double-clicking anywhere on plate canvas jumps stage focus directly!
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

        plate_w_mm = self.plate_format.PLATE_LENGTH_MM  # 127.76 mm
        plate_h_mm = self.plate_format.PLATE_WIDTH_MM   # 85.48 mm

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

            # Plate Outer Boundary
            Color(0.2, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.5, 0.5, 0.6, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = (self.plate_format.well_diameter / 2.0) * scale

            # Draw Wells (A1 Top-Left -> Rows down, Cols right)
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

            # Draw Calibration Targets
            cal_steps = self.calibration.CALIBRATION_STEPS
            for idx, step_info in enumerate(cal_steps):
                step_id = step_info["id"]
                is_recorded = step_id in self.calibration.recorded_points
                is_active = (idx == self.calibration.current_step_index) and not self.calibration.is_calibrated

                if is_recorded:
                    Color(0.2, 0.9, 0.3, 1.0)
                elif is_active:
                    Color(1.0, 0.9, 0.1, 1.0)
                else:
                    Color(0.6, 0.6, 0.2, 0.5)

                tx_px, ty_px = self._get_step_target_px(step_id, offset_x, offset_y, scale)
                Ellipse(pos=(tx_px - 4, ty_px - 4), size=(8, 8))
                if is_active:
                    Line(circle=(tx_px, ty_px, 8), width=1.5)

            # Draw Master Auto Point (Red)
            if self.focus_map.master_auto_point is not None:
                pt = self.focus_map.master_auto_point
                ax_px = offset_x + (self.plate_format.a1_offset_x + (pt.x_stage - (-49.5))) * scale
                ay_px = offset_y - (self.plate_format.a1_offset_y + (31.5 - pt.y_stage)) * scale
                Color(1.0, 0.1, 0.1, 1.0)
                Ellipse(pos=(ax_px - 6, ay_px - 6), size=(12, 12))
                Color(1.0, 0.5, 0.5, 1.0)
                Line(circle=(ax_px, ay_px, 10), width=1.5)

            # Active Stage Motor & FOV Marker (Cyan Box + Red Center Dot + Green Crosshairs)
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x'):
                stg_x = app.stage_x
                stg_y = app.stage_y

                # Map stage coords (-63.88 to +63.88 mm) to 0..PLATE_LENGTH_MM
                norm_x = (stg_x - (-plate_w_mm / 2.0)) / plate_w_mm
                norm_y = ((plate_h_mm / 2.0) - stg_y) / plate_h_mm

                cur_px_x = offset_x + norm_x * draw_w
                cur_px_y = offset_y - norm_y * draw_h

                fov_w_mm = getattr(self.focus_map, 'fov_width_mm', 0.60)
                fov_h_mm = getattr(self.focus_map, 'fov_height_mm', 0.45)

                box_w_px = max(4.0, (fov_w_mm / plate_w_mm) * draw_w)
                box_h_px = max(3.0, (fov_h_mm / plate_h_mm) * draw_h)

                # Cyan Active FOV Box
                Color(0.0, 0.9, 1.0, 0.9)
                Line(rectangle=(cur_px_x - box_w_px / 2.0, cur_px_y - box_h_px / 2.0, box_w_px, box_h_px), width=1.5)

                # Red Stage Center Dot & Green Crosshairs
                Color(1.0, 0.2, 0.2, 1.0)
                Ellipse(pos=(cur_px_x - 3, cur_px_y - 3), size=(6, 6))
                Color(0.2, 0.9, 0.3, 0.8)
                Line(points=[cur_px_x - 8, cur_px_y, cur_px_x + 8, cur_px_y], width=1.0)
                Line(points=[cur_px_x, cur_px_y - 8, cur_px_x, cur_px_y + 8], width=1.0)

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
        plate_w_mm = self.plate_format.PLATE_LENGTH_MM
        plate_h_mm = self.plate_format.PLATE_WIDTH_MM
        scale_x = (self.width - 2 * margin) / plate_w_mm
        scale_y = (self.height - 2 * margin) / plate_h_mm
        scale = min(scale_x, scale_y)

        draw_w = plate_w_mm * scale
        draw_h = plate_h_mm * scale

        offset_x = self.x + (self.width - draw_w) / 2.0
        offset_y = self.y + (self.height + draw_h) / 2.0

        rel_px_x = touch.x - offset_x
        rel_px_y = offset_y - touch.y

        app = App.get_running_app()

        # Check if touch is within plate bounding rectangle
        if 0 <= rel_px_x <= draw_w and 0 <= rel_px_y <= draw_h:
            norm_x = rel_px_x / draw_w
            norm_y = rel_px_y / draw_h

            target_stage_x = -plate_w_mm / 2.0 + norm_x * plate_w_mm
            target_stage_y = plate_h_mm / 2.0 - norm_y * plate_h_mm

            if touch.is_double_tap:
                if app and hasattr(app, 'jog_stage_to'):
                    app.jog_stage_to(target_stage_x, target_stage_y)
                elif app and hasattr(app, 'stage_x'):
                    app.stage_x = target_stage_x
                    app.stage_y = target_stage_y
                    if hasattr(app, 'clamp_stage_position'):
                        app.clamp_stage_position()
                self.redraw_canvas()
                return True

            # Single tap: toggle well selection
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
    """Modal Plate Navigator Window overlaying stock LumaViewPro."""
    
    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app_ctrl = app_controller
        self.title = "LumaViewPro - Plate Navigator & Registration Studio"
        self.size_hint = (0.96, 0.96)


def attach_navigator_to_lumaviewpro(app_instance):
    """Helper function to launch Plate Navigator studio modal popup."""
    if hasattr(app_instance, 'navigator_popup') and app_instance.navigator_popup is not None:
        try:
            app_instance.navigator_popup.open()
            return app_instance.navigator_popup
        except Exception:
            pass

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

    popup = PlateNavigatorPopup(app_instance)
    app_instance.navigator_popup = popup

    def on_dismiss_cb(instance):
        app_instance.navigator_popup = None

    popup.bind(on_dismiss=on_dismiss_cb)

    if hasattr(popup, 'ids') and 'scaled_canvas_tab1' in popup.ids:
        popup.ids.scaled_canvas_tab1.set_references(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map,
            app_instance.fov_manager
        )

    popup.open()
    return popup
