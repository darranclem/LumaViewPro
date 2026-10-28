"""
plate_navigator.py - Modular Plate Navigator & Registration Studio Extension for LumaViewPro.

Provides:
- InteractiveScaledPlateCanvas: Physically accurate microplate canvas with Top-Left A1 orientation,
  bright 6-point target calibration overlays, active FOV reticle box, and double-click well stage navigation.
- ZoomPanCameraWidget: Interactive live camera viewport with central reticle crosshairs, mouse wheel zooming,
  right-click panning, and double-click pixel stage centering.
- PlateNavigatorPopup: Modal studio window with single-child content architecture.
- attach_navigator_to_lumaviewpro(): Bridge attaching data models and handlers to any LumaViewPro instance.
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
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty, DictProperty
)

# Load plate_navigator.kv layout safely exactly once
_nav_kv_file = os.path.join(script_dir, 'plate_navigator.kv')
if os.path.exists(_nav_kv_file) and _nav_kv_file not in Builder.files:
    Builder.load_file(_nav_kv_file)


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - NumPy image frame conversion via set_frame(frame_array)
    - Center Reticle Crosshairs
    - Mouse Wheel Zooming (Zoom In / Zoom Out)
    - Right-Click Drag Panning in X and Y
    - Double-Click Pixel Direct Stage Centering
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self._texture = None
        self._last_frame_shape = (512, 512)
        self.bind(pos=self._redraw_viewport, size=self._redraw_viewport, zoom_level=self._redraw_viewport, pan_x=self._redraw_viewport, pan_y=self._redraw_viewport)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._redraw_viewport()

    def set_frame(self, frame_array: np.ndarray):
        """Converts a NumPy BGR/RGB image array to Kivy Texture and renders on viewport canvas."""
        if frame_array is None or frame_array.size == 0:
            return

        h, w = frame_array.shape[:2]
        self._last_frame_shape = (w, h)

        if len(frame_array.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
        elif frame_array.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_array[:, :, :3]

        buf = np.ascontiguousarray(frame_rgb[::-1, :, :], dtype=np.uint8).tobytes()

        if self._texture is None or self._texture.width != w or self._texture.height != h:
            self._texture = Texture.create(size=(w, h), colorfmt='rgb')

        self._texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self._redraw_viewport()

    def _redraw_viewport(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
            # Dark background
            Color(0.05, 0.05, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self._texture is not None:
                tw, th = self._texture.size
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
                Rectangle(texture=self._texture, pos=(draw_x, draw_y), size=(draw_w, draw_h))

            # Central Reticle Crosshairs Overlay
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.85)  # Bright Red
            Line(points=[cx - 18, cy, cx + 18, cy], width=1.5)
            Line(points=[cx, cy - 18, cx, cy + 18], width=1.5)
            Line(circle=(cx, cy, 6), width=1.2)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

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

        # Double-click live image pixel centering
        if touch.is_double_tap:
            app = App.get_running_app()
            if app and hasattr(app, 'focus_map'):
                fov_w = app.focus_map.fov_width_mm
                fov_h = app.focus_map.fov_height_mm
                cx = self.x + self.width / 2.0
                cy = self.y + self.height / 2.0

                dx_px = touch.x - cx
                dy_px = touch.y - cy

                # Scale pixel offsets to mm
                dx_mm = (dx_px / (self.width * self.zoom_level)) * fov_w
                dy_mm = (dy_px / (self.height * self.zoom_level)) * fov_h

                if hasattr(app, 'jog_stage_to'):
                    app.jog_stage_to(app.stage_x + dx_mm, app.stage_y - dy_mm)
                    if hasattr(app, 'root_layout') and app.root_layout:
                        app.root_layout.status_msg = f"Pixel Centered: Stage moved by dx={dx_mm:+.3f} mm, dy={dy_mm:+.3f} mm"
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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = labware.WellPlateFormat(num_wells=96)
        self.calibration = labware.NavigatorPlateCalibration(self.plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.plate_format, self.calibration, self.focus_map)
        self.bind(pos=self.redraw_canvas, size=self.redraw_canvas)

    def set_references(self, plate_format, calibration, focus_map, fov_manager):
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

            # Plate outer boundary
            Color(0.20, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.50, 0.55, 0.65, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = max(2.0, (self.plate_format.well_diameter / 2.0) * scale)

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
                        Color(0.12, 0.14, 0.18, 0.8)
                        Ellipse(pos=(wx_px - well_r_px, wy_px - well_r_px), size=(2 * well_r_px, 2 * well_r_px))
                        Color(0.35, 0.38, 0.45, 1.0)
                        Line(ellipse=(wx_px - well_r_px, wy_px - well_r_px, 2 * well_r_px, 2 * well_r_px), width=1.0)

                    # Tiled FOV Overlay
                    if is_selected and w_name in self.fov_manager.tiled_groups:
                        group = self.fov_manager.tiled_groups[w_name]
                        Color(1.0, 0.8, 0.2, 0.8)
                        for tile in group.tiles:
                            tx_px = wx_px + tile.offset_x_mm * scale
                            ty_px = wy_px - tile.offset_y_mm * scale
                            tw_px = group.fov_width_mm * scale * 0.35
                            th_px = group.fov_height_mm * scale * 0.35
                            Line(rectangle=(tx_px - tw_px/2, ty_px - th_px/2, tw_px, th_px), width=1.0)

            # Draw Active Stage Position Reticle Box & Red Center Dot
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x'):
                curr_x = app.stage_x
                curr_y = app.stage_y
                
                # Nominally relative to A1
                a1_st_x = -draw_w / 2.0 + self.plate_format.a1_offset_x * scale
                a1_st_y = draw_h / 2.0 - self.plate_format.a1_offset_y * scale

                # Stage position to canvas pixel coordinates
                st_px_x = offset_x + (curr_x + draw_w / 2.0)
                st_px_y = offset_y - (curr_y - draw_h / 2.0)

                fov_w_mm = self.focus_map.fov_width_mm
                fov_h_mm = self.focus_map.fov_height_mm
                fov_w_px = max(6.0, fov_w_mm * scale)
                fov_h_px = max(4.0, fov_h_mm * scale)

                # Active FOV Reticle Box (Cyan)
                Color(0.0, 0.9, 1.0, 0.9)
                Line(rectangle=(st_px_x - fov_w_px/2, st_px_y - fov_h_px/2, fov_w_px, fov_h_px), width=1.5)
                Color(1.0, 0.1, 0.1, 1.0)
                Ellipse(pos=(st_px_x - 3, st_px_y - 3), size=(6, 6))

            # Prominent Leica Calibration Target Overlays (Yellow/Green)
            cal_steps = self.calibration.CALIBRATION_STEPS
            for idx, step_info in enumerate(cal_steps):
                step_id = step_info["id"]
                is_recorded = step_id in self.calibration.recorded_points
                is_active = (idx == self.calibration.current_step_index) and not self.calibration.is_calibrated

                tx_px, ty_px = self._get_step_target_px(step_id, offset_x, offset_y, scale)

                if is_recorded:
                    # Recorded Point: Solid Green Target
                    Color(0.2, 0.9, 0.3, 1.0)
                    Ellipse(pos=(tx_px - 6, ty_px - 6), size=(12, 12))
                    Line(circle=(tx_px, ty_px, 10), width=1.5)
                elif is_active:
                    # Active Targeting Step: Bright Yellow Target
                    Color(1.0, 0.9, 0.1, 1.0)
                    Ellipse(pos=(tx_px - 8, ty_px - 8), size=(16, 16))
                    Line(circle=(tx_px, ty_px, 12), width=1.8)
                    Line(circle=(tx_px, ty_px, 6), width=1.0)
                    Line(points=[tx_px - 12, ty_px, tx_px + 12, ty_px], width=1.2)
                    Line(points=[tx_px, ty_px - 12, tx_px, ty_px + 12], width=1.2)
                else:
                    # Upcoming Target: Translucent Yellow Ring
                    Color(0.8, 0.8, 0.2, 0.5)
                    Ellipse(pos=(tx_px - 5, ty_px - 5), size=(10, 10))
                    Line(circle=(tx_px, ty_px, 8), width=1.0)

            # Master Auto Focus Point (Red "Auto")
            if self.focus_map.master_auto_point is not None:
                pt = self.focus_map.master_auto_point
                ax_px = offset_x + pt.x_stage * scale
                ay_px = offset_y - pt.y_stage * scale
                Color(1.0, 0.1, 0.1, 1.0)
                Ellipse(pos=(ax_px - 6, ay_px - 6), size=(12, 12))
                Color(1.0, 0.5, 0.5, 1.0)
                Line(circle=(ax_px, ay_px, 10), width=1.5)

            # Set Focus Points (Blue)
            for set_pt in self.focus_map.set_points:
                sx_px = offset_x + set_pt.x_stage * scale
                sy_px = offset_y - set_pt.y_stage * scale
                Color(0.1, 0.5, 1.0, 1.0)
                Ellipse(pos=(sx_px - 5, sy_px - 5), size=(10, 10))
                Line(circle=(sx_px, sy_px, 8), width=1.2)

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
        offset_y = self.y + (self.height + draw_h * scale) / 2.0

        for r in range(self.plate_format.rows):
            for c in range(self.plate_format.cols):
                x_nom, y_nom = self.plate_format.get_nominal_position(r, c)
                wx_px = offset_x + (self.plate_format.a1_offset_x + x_nom) * scale
                wy_px = offset_y - (self.plate_format.a1_offset_y + y_nom) * scale
                well_r_px = (self.plate_format.well_diameter / 2.0) * scale

                dist = math.hypot(touch.x - wx_px, touch.y - wy_px)
                if dist <= well_r_px:
                    w_name = self.plate_format.get_well_name(r, c)
                    app = App.get_running_app()

                    if touch.is_double_tap:
                        # Double-click well: move stage focus directly to well center
                        x_stage, y_stage = self.calibration.get_well_stage_coordinates(r, c)
                        if hasattr(app, 'jog_stage_to'):
                            app.jog_stage_to(x_stage, y_stage)
                        elif hasattr(app, 'stage_x'):
                            app.stage_x = x_stage
                            app.stage_y = y_stage
                            if hasattr(app, 'root_layout') and app.root_layout:
                                app.root_layout.stage_coord_str = f"Stage: X={x_stage:.2f}, Y={y_stage:.2f}, Z={app.stage_z:.3f} mm"
                                app.root_layout.current_well_display = w_name
                                app.root_layout.status_msg = f"Navigated to Well {w_name} center"
                        self.redraw_canvas()
                        return True

                    # Single click: toggle well selection
                    self.fov_manager.toggle_well_selected(w_name)
                    self.redraw_canvas()
                    return True

        return super().on_touch_down(touch)


class PlateNavigatorPopup(Popup):
    """
    Modal Plate Navigator Window with single-child content architecture.
    """
    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app_ctrl = app_controller
        self.title = "LumaViewPro - Plate Navigator & Registration Studio"
        self.size_hint = (0.96, 0.96)


def attach_navigator_to_lumaviewpro(app_instance):
    """Helper function to attach data models and open Plate Navigator Popup cleanly."""
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

    if hasattr(app_instance, '_active_navigator_popup') and app_instance._active_navigator_popup:
        try:
            app_instance._active_navigator_popup.dismiss()
        except Exception:
            pass

    popup = PlateNavigatorPopup(app_instance)
    app_instance._active_navigator_popup = popup

    if hasattr(popup, 'ids') and 'scaled_canvas_tab1' in popup.ids:
        popup.ids.scaled_canvas_tab1.set_references(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map,
            app_instance.fov_manager
        )

    popup.open()
    return popup
