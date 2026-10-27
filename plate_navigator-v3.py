"""
plate_navigator.py - Modular Plate Navigator & Registration Studio Extension for LumaViewPro.

Provides:
- PlateNavigatorPopup: Modal dialog window with 3 core tabs:
  1. Calibration (ANSI/SLAS template with Top-Left A1 orientation, 6-point target registration, Master Auto Point, test AF run)
  2. AF Dialogue (AF search range/step, camera binning, exposure time, optical Nyquist calculator, skip cycle frequency, time-lapse schedule)
  3. Hardware Protocol (1x1 to 12x12 tiling grid, channel accept/reject list, secondary chromatic shift Z-offsets, protocol export)
- ZoomPanCameraWidget: Interactive camera viewport with smooth mouse wheel zooming and right-click panning.
- InteractiveScaledPlateCanvas: Physically accurate microplate canvas with Top-Left A1 orientation.
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
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty, DictProperty
)

# Load plate_navigator.kv automatically into Kivy Builder if not loaded
kv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plate_navigator.kv')
if os.path.exists(kv_file) and kv_file not in Builder.files:
    Builder.load_file(kv_file)


class ZoomPanCameraWidget(Widget):
    """Interactive Live Camera Viewport Widget with Mouse Wheel Zoom and Right-Click Drag Pan."""
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

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
    """Custom Kivy canvas widget rendering microplate representation with Top-Left A1 Orientation."""

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

            Color(0.2, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.5, 0.5, 0.6, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = (self.plate_format.well_diameter / 2.0) * scale

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

                    if is_selected and w_name in self.fov_manager.tiled_groups:
                        group = self.fov_manager.tiled_groups[w_name]
                        Color(1.0, 0.8, 0.2, 0.8)
                        for tile in group.tiles:
                            tx_px = wx_px + tile.offset_x_mm * scale
                            ty_px = wy_px - tile.offset_y_mm * scale
                            tw_px = group.fov_width_mm * scale * 0.35
                            th_px = group.fov_height_mm * scale * 0.35
                            Line(rectangle=(tx_px - tw_px/2, ty_px - th_px/2, tw_px, th_px), width=1.0)

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

            if self.focus_map.master_auto_point is not None:
                pt = self.focus_map.master_auto_point
                ax_px = offset_x + pt.x_stage * scale
                ay_px = offset_y - pt.y_stage * scale
                Color(1.0, 0.1, 0.1, 1.0)
                Ellipse(pos=(ax_px - 6, ay_px - 6), size=(12, 12))
                Color(1.0, 0.5, 0.5, 1.0)
                Line(circle=(ax_px, ay_px, 10), width=1.5)

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
        offset_y = self.y + (self.height + draw_h) / 2.0

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
    """Modal Plate Navigator Window overlay."""
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

    popup = PlateNavigatorPopup(app_instance)
    if hasattr(popup.ids, 'scaled_canvas_tab1'):
        popup.ids.scaled_canvas_tab1.set_references(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map,
            app_instance.fov_manager
        )
    popup.open()
    return popup
