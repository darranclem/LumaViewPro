"""
plate_navigator.py - Modular Plate Navigator & Registration Studio Extension for LumaViewPro.
"""

from __future__ import annotations

import os
import sys
import math
import re
import json
from typing import Dict, Optional, List, Tuple

import image_utils
import labware

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
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

_plate_navigator_kv_loaded = False
_kv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plate_navigator.kv')
if os.path.exists(_kv_file) and not getattr(Builder, '_plate_navigator_kv_loaded', False):
    try:
        Builder.load_file(_kv_file)
        setattr(Builder, '_plate_navigator_kv_loaded', True)
    except Exception as e:
        print(f"[plate_navigator.py] KV load note: {e}")


class InteractiveScaledPlateCanvas(Widget):
    """
    Custom Kivy canvas widget rendering a physically accurate representation
    of the labware plate based on ANSI/SLAS standards and 2D Affine calibration matrix.
    Orientation: Well A1 is positioned at the TOP-LEFT corner.
    Features:
    - Double-Click Well Stage Navigation
    - Active Stage Position Reticle Box
    - 6-Point Target Registration Overlays
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

        rot_angle = self.calibration.rotation_angle_deg if self.calibration.is_calibrated else 0.0

        with self.canvas:
            PushMatrix()
            if rot_angle != 0.0:
                Rotate(angle=rot_angle, origin=(offset_x + draw_w / 2.0, offset_y - draw_h / 2.0))

            # Plate outer boundary
            Color(0.18, 0.20, 0.25, 1.0)
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

            # Draw 6-point target registration overlays
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
                    Line(circle=(tx_px, ty_px, 10), width=1.5)

            # Draw Active Microscope Stage Position Reticle (Cyan Box + Red Center Dot)
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x'):
                st_x, st_y = app.stage_x, app.stage_y
                a1_stage_x = -49.5
                a1_stage_y = 31.5

                rel_x_mm = st_x - a1_stage_x
                rel_y_mm = a1_stage_y - st_y

                spx = offset_x + (self.plate_format.a1_offset_x + rel_x_mm) * scale
                spy = offset_y - (self.plate_format.a1_offset_y + rel_y_mm) * scale

                fov_w_mm = getattr(self.focus_map, 'fov_width_mm', 0.60)
                fov_h_mm = getattr(self.focus_map, 'fov_height_mm', 0.45)
                rect_w = fov_w_mm * scale
                rect_h = fov_h_mm * scale

                Color(0.0, 0.9, 1.0, 0.9) # Bright Cyan FOV Box
                Line(rectangle=(spx - rect_w / 2.0, spy - rect_h / 2.0, rect_w, rect_h), width=1.8)
                Color(1.0, 0.2, 0.2, 1.0) # Red Center Dot
                Ellipse(pos=(spx - 3, spy - 3), size=(6, 6))

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

        margin = 20.0
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
                        # DOUBLE CLICK WELL: Move stage focus directly to well center!
                        a1_x, a1_y = -49.5, 31.5
                        target_x = a1_x + c * self.plate_format.pitch_x
                        target_y = a1_y - r * self.plate_format.pitch_y
                        
                        if app and hasattr(app, 'jog_stage_to'):
                            app.jog_stage_to(target_x, target_y)
                        elif app and hasattr(app, 'stage_x'):
                            app.stage_x = target_x
                            app.stage_y = target_y
                            if hasattr(app, 'root_layout') and app.root_layout:
                                app.root_layout.stage_coord_str = f"Stage: X={target_x:.2f}, Y={target_y:.2f}, Z={app.stage_z:.3f} mm"
                                app.root_layout.current_well_display = w_name
                        self.redraw_canvas()
                        return True

                    # SINGLE CLICK WELL: Toggle selection
                    self.fov_manager.toggle_well_selected(w_name)
                    self.redraw_canvas()
                    return True

        return super().on_touch_down(touch)


class PlateNavigatorPopup(Popup):
    """Modal Plate Navigator Window overlaying stock LumaViewPro."""
    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app_ctrl = app_controller
        self.title = "LumaViewPro - Plate Navigator Studio"
        self.size_hint = (0.96, 0.96)


def attach_navigator_to_lumaviewpro(app_instance):
    """Helper function to attach Navigator data models to running LumaViewPro App instance."""
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
