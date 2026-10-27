"""
plate_navigator.py - Modular Plate Navigator & Microplate Registration Studio for LumaViewPro.

Provides:
- InteractiveScaledPlateCanvas: Physically accurate microplate canvas with Top-Left A1 orientation,
  bright target calibration overlays, cyan active FOV reticle box, red center dot, and double-click stage navigation.
- ZoomPanCameraWidget: Interactive live camera viewport with mouse wheel zooming, right-click panning,
  reticle crosshairs overlay, and double-click pixel stage centering.
- PlateNavigatorPopup: Modal studio window with single-child content architecture.
- attach_navigator_to_lumaviewpro(): Bridge attaching data models and handlers to any LumaViewPro instance.
"""

from __future__ import annotations

import os
import sys
import math
import re
import json
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
import image_utils
import labware

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.properties import StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty

# Load plate_navigator.kv layout safely exactly once
_kv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plate_navigator.kv')
if os.path.exists(_kv_file) and _kv_file not in Builder.files:
    try:
        Builder.load_file(_kv_file)
    except Exception:
        pass


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - Live Texture Rendering & Frame Blitting
    - Reticle Crosshairs Overlay
    - Double-Click Pixel Centering (jogs stage so clicked point moves under reticle)
    - Mouse Wheel Zooming & Right-Click Drag Panning
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.current_texture = None
        self.bind(
            pos=self.redraw_viewport,
            size=self.redraw_viewport,
            zoom_level=self.redraw_viewport,
            pan_x=self.redraw_viewport,
            pan_y=self.redraw_viewport
        )

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

    def set_texture(self, texture):
        self.current_texture = texture
        self.redraw_viewport()

    def set_frame(self, frame_np: np.ndarray):
        if frame_np is None or frame_np.size == 0:
            return
        try:
            h, w = frame_np.shape[:2]
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame_np

            flipped = np.ascontiguousarray(frame_rgb[::-1, :, :], dtype=np.uint8)

            if self.current_texture is None or self.current_texture.size != (w, h):
                self.current_texture = Texture.create(size=(w, h), colorfmt='rgb')

            self.current_texture.blit_buffer(flipped.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.redraw_viewport()
        except Exception as e:
            print(f"[ZoomPanCameraWidget set_frame error] {e}")

    def redraw_viewport(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
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

                # Render Central Reticle Crosshairs
                Color(1.0, 0.2, 0.2, 0.85)  # Bright Red
                Line(points=[self.x, center_y, self.x + self.width, center_y], width=1.2)
                Line(points=[center_x, self.y, center_x, self.y + self.height], width=1.2)
                Line(circle=(center_x, center_y, 12), width=1.5)
            else:
                Color(0.12, 0.14, 0.18, 1.0)
                Rectangle(pos=self.pos, size=self.size)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                self.zoom_level = max(0.5, self.zoom_level - 0.1)
            elif touch.button == 'scrollup':
                self.zoom_level = min(5.0, self.zoom_level + 0.1)
            return True

        if touch.is_double_tap:
            # Double-click pixel centering: Moves clicked point to center of FOV
            center_x = self.x + self.width / 2.0 + self.pan_x
            center_y = self.y + self.height / 2.0 + self.pan_y
            dx_px = touch.x - center_x
            dy_px = touch.y - center_y

            app = App.get_running_app()
            if app and hasattr(app, 'focus_map'):
                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.60)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)

                dx_mm = (dx_px / max(1.0, self.width * self.zoom_level)) * fov_w
                dy_mm = (dy_px / max(1.0, self.height * self.zoom_level)) * fov_h

                target_x = app.stage_x + dx_mm
                target_y = app.stage_y + dy_mm

                if hasattr(app, 'jog_stage_to'):
                    app.jog_stage_to(target_x, target_y)
                elif hasattr(app, 'jog_stage'):
                    app.jog_stage(dx_mm, dy_mm, 0.0)

                if hasattr(app, 'root_layout') and hasattr(app.root_layout, 'status_msg'):
                    app.root_layout.status_msg = f"Centered FOV on clicked pixel (Stage: X={app.stage_x:.2f}, Y={app.stage_y:.2f} mm)"
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
    Supports Double-Click Navigation: Double-clicking any well or coordinate moves stage focus directly there.
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

            # Outer boundary
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
                            tw_px = max(1.0, group.fov_width_mm * scale * 0.35)
                            th_px = max(1.0, group.fov_height_mm * scale * 0.35)
                            Line(rectangle=(tx_px - tw_px/2, ty_px - th_px/2, tw_px, th_px), width=1.0)

            # Draw Active Stage Position Indicator (Cyan FOV Box + Red Center Dot)
            app = App.get_running_app()
            if app and hasattr(app, 'stage_x') and hasattr(app, 'stage_y'):
                # Exact 1:1 Stage Position Mapping relative to plate top-left
                plate_x_mm = app.stage_x + plate_w_mm / 2.0
                plate_y_mm = plate_h_mm / 2.0 - app.stage_y

                cur_px_x = offset_x + plate_x_mm * scale
                cur_px_y = offset_y - plate_y_mm * scale

                fov_w_mm = getattr(self.focus_map, 'fov_width_mm', 0.60)
                fov_h_mm = getattr(self.focus_map, 'fov_height_mm', 0.45)
                fov_w_px = max(4.0, fov_w_mm * scale)
                fov_h_px = max(3.0, fov_h_mm * scale)

                # Active FOV Cyan Rectangle
                Color(0.0, 0.85, 1.0, 0.9)
                Line(rectangle=(cur_px_x - fov_w_px / 2.0, cur_px_y - fov_h_px / 2.0, fov_w_px, fov_h_px), width=1.5)

                # Center Reticle Dot & Crosshair
                Color(1.0, 0.2, 0.2, 1.0)
                Ellipse(pos=(cur_px_x - 3, cur_px_y - 3), size=(6, 6))
                Color(0.0, 1.0, 0.3, 0.9)
                Line(points=[cur_px_x - 10, cur_px_y, cur_px_x + 10, cur_px_y], width=1.2)
                Line(points=[cur_px_x, cur_px_y - 10, cur_px_x, cur_px_y + 10], width=1.2)

            PopMatrix()

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        margin = 25.0
        draw_w = self.plate_format.PLATE_LENGTH_MM
        draw_h = self.plate_format.PLATE_WIDTH_MM
        scale_x = (self.width - 2 * margin) / draw_w
        scale_y = (self.height - 2 * margin) / draw_h
        scale = min(scale_x, scale_y)

        draw_w_px = draw_w * scale
        draw_h_px = draw_h * scale

        offset_x = self.x + (self.width - draw_w_px) / 2.0
        offset_y = self.y + (self.height + draw_h_px) / 2.0

        # Calculate exact clicked plate coordinates in mm from top-left plate edge
        click_plate_x_mm = (touch.x - offset_x) / scale
        click_plate_y_mm = (offset_y - touch.y) / scale

        app = App.get_running_app()

        if touch.is_double_tap:
            # DOUBLE-CLICK ANYWHERE ON PLATE CANVAS: Rapid Stage Navigation!
            target_stage_x = click_plate_x_mm - draw_w / 2.0
            target_stage_y = draw_h / 2.0 - click_plate_y_mm

            if app and hasattr(app, 'jog_stage_to'):
                app.jog_stage_to(target_stage_x, target_stage_y)
            elif app and hasattr(app, 'stage_x'):
                cx, cy, _ = app.clamp_stage_position(target_stage_x, target_stage_y)
                app.stage_x = cx
                app.stage_y = cy
                app.root_layout.stage_coord_str = f"Stage: X={cx:.2f}, Y={cy:.2f}, Z={app.stage_z:.3f} mm"
                app.update_active_well_display()
                app.update_live_feed(0)

            self.redraw_canvas()
            return True

        # Single tap: toggle well selection if clicked inside a well
        for r in range(self.plate_format.rows):
            for c in range(self.plate_format.cols):
                x_nom, y_nom = self.plate_format.get_nominal_position(r, c)
                wx_px = offset_x + (self.plate_format.a1_offset_x + x_nom) * scale
                wy_px = offset_y - (self.plate_format.a1_offset_y + y_nom) * scale
                well_r_px = max(2.0, (self.plate_format.well_diameter / 2.0) * scale)

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
    if hasattr(popup, 'ids') and 'scaled_canvas_tab1' in popup.ids:
        popup.ids.scaled_canvas_tab1.set_references(
            app_instance.well_plate_format,
            app_instance.plate_calibration,
            app_instance.focus_map,
            app_instance.fov_manager
        )
    popup.open()
    return popup
