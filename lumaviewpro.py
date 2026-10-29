"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation.

Unified Single-Window UI Architecture:
- Embedded Live Camera Viewport (Top-Left) with interactive Zoom and Pan.
- Hardware Motion Stage & Illumination Controls (Bottom-Left).
- Interactive Microplate Map Display (Top-Right) with double-click well navigation and format selection (6, 12, 24, 48, 96, 384 wells).
- Registration & Optical Channel Controls (Bottom-Right).
"""

from __future__ import annotations
import os
import sys

# Force local repository directory to be #1 in Python's module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import json
import math
import re
import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple

import image_utils
import labware
import plate_navigator

# Kivy framework imports
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
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics import Color, Ellipse, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ObjectProperty, ListProperty, DictProperty
)


class ChannelConfig:
    """Stores configuration and optical illumination/emission specs for an imaging channel."""
    
    def __init__(
        self,
        name: str,
        excitation_range_nm: str = "",
        emission_range_nm: str = "",
        color_hex: str = "#FFFFFF",
        is_transmitted: bool = True,
        btn_color: List[float] = None
    ):
        self.name: str = name
        self.excitation_range_nm: str = excitation_range_nm
        self.emission_range_nm: str = emission_range_nm
        self.color_hex: str = color_hex
        self.is_transmitted: bool = is_transmitted
        self.btn_color: List[float] = btn_color if btn_color else [0.5, 0.5, 0.5, 1.0]
        self.active: bool = False
        self.exposure_ms: float = 20.0
        self.gain: float = 1.0
        self.ref_image_path: Optional[str] = None
        self.ref_image_array: Optional[object] = None
        self.shading_correction_enabled: bool = True

    def load_ref_image(self, filepath: str) -> bool:
        try:
            img = image_utils.load_image(filepath)
            if img is not None:
                self.ref_image_path = filepath
                self.ref_image_array = img
                return True
        except Exception as e:
            print(f"[Error] Failed to load reference image for {self.name}: {e}")
        return False


class ChannelManager:
    """Manages optical imaging channels with strict single-channel mutual exclusion."""
    
    ALIASES = {
        "BF": "Brightfield",
        "Pseudo Darkfield": "Darkfield",
        "Pseudo DF": "Darkfield",
        "DAPI": "Blue",
        "FITC": "Green",
        "TxRed": "Red",
        "Cy5": "Red"
    }

    def __init__(self):
        self.channels: Dict[str, ChannelConfig] = {
            "Brightfield": ChannelConfig("Brightfield", excitation_range_nm="White LED", emission_range_nm="Pass-through", color_hex="#FFFFFF", is_transmitted=True, btn_color=[0.9, 0.9, 0.9, 1.0]),
            "Phase": ChannelConfig("Phase", excitation_range_nm="White Ring", emission_range_nm="Phase Shift", color_hex="#CCCCCC", is_transmitted=True, btn_color=[0.65, 0.7, 0.75, 1.0]),
            "Darkfield": ChannelConfig("Darkfield", excitation_range_nm="Oblique LED", emission_range_nm="Pass-through", color_hex="#888888", is_transmitted=True, btn_color=[0.4, 0.4, 0.45, 1.0]),
            "Blue": ChannelConfig("Blue", excitation_range_nm="370-410 nm", emission_range_nm="429-462 nm", color_hex="#0080FF", is_transmitted=False, btn_color=[0.0, 0.45, 1.0, 1.0]),
            "Green": ChannelConfig("Green", excitation_range_nm="473-491 nm", emission_range_nm="502-561 nm", color_hex="#00FF40", is_transmitted=False, btn_color=[0.0, 0.85, 0.25, 1.0]),
            "Red": ChannelConfig("Red", excitation_range_nm="580-598 nm", emission_range_nm="612-680 nm", color_hex="#FF3030", is_transmitted=False, btn_color=[0.95, 0.15, 0.15, 1.0])
        }

    def resolve_channel_name(self, name: str) -> str:
        return self.ALIASES.get(name, name)

    def set_channel_active(self, channel_name: str, active: bool):
        real_name = self.resolve_channel_name(channel_name)
        if active:
            for name, ch_cfg in self.channels.items():
                ch_cfg.active = (name == real_name)
        else:
            if real_name in self.channels:
                self.channels[real_name].active = False

    def global_sync_active_channels(self, active_channel_states: Optional[Dict[str, bool]] = None):
        if active_channel_states:
            for name, active_val in active_channel_states.items():
                real_name = self.resolve_channel_name(name)
                if real_name in self.channels:
                    self.channels[real_name].active = active_val

    def get_active_channel_names(self) -> List[str]:
        return [ch_name for ch_name, ch_cfg in self.channels.items() if ch_cfg.active]

    def process_channel_image(self, channel_name: str, raw_frame: object) -> object:
        real_name = self.resolve_channel_name(channel_name)
        ch = self.channels.get(real_name)
        if ch and ch.shading_correction_enabled and ch.ref_image_array is not None:
            return image_utils.apply_shading_correction(raw_frame, ch.ref_image_array)
        return raw_frame


class ChannelControlWidget(BoxLayout):
    channel_name = StringProperty("")
    channel_spec_info = StringProperty("")
    ref_image_path = StringProperty("No Ref Image")
    is_active = BooleanProperty(False)
    btn_bg_color = ListProperty([0.2, 0.6, 0.8, 1.0])

    def open_ref_chooser(self):
        App.get_running_app().open_file_chooser_for_channel(self.channel_name)


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget with:
    - Reticle Crosshairs at Center
    - Double-click Pixel Centering Motion
    - Mouse Wheel Zooming & Right-Click Panning
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.texture = None
        self.bind(pos=self.redraw_viewport, size=self.redraw_viewport, zoom_level=self.redraw_viewport, pan_x=self.redraw_viewport, pan_y=self.redraw_viewport)

    def set_frame(self, frame_array: np.ndarray):
        if frame_array is None or frame_array.size == 0:
            return
        try:
            h, w = frame_array.shape[:2]
            if len(frame_array.shape) == 2:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
            elif frame_array.shape[2] == 3:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            else:
                return

            texture = Texture.create(size=(w, h), colorfmt='rgb')
            texture.blit_buffer(rgb_buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            texture.flip_vertical()

            self.current_texture = texture
            self.redraw_viewport()
        except Exception as e:
            print(f"[ZoomPanCameraWidget Error] set_frame: {e}")

    def redraw_viewport(self, *args):
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
            Color(0.05, 0.05, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self.current_texture is not None:
                vw = self.width * self.zoom_level
                vh = self.height * self.zoom_level
                vx = self.x + (self.width - vw) / 2.0 + self.pan_x
                vy = self.y + (self.height - vh) / 2.0 + self.pan_y

                Color(1.0, 1.0, 1.0, 1.0)
                Rectangle(texture=self.current_texture, pos=(vx, vy), size=(vw, vh))
            else:
                Color(0.2, 0.2, 0.25, 1.0)
                Rectangle(pos=self.pos, size=self.size)

            # Draw Center Reticle Crosshairs Overlay
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.8) # Bright Red Crosshairs
            Line(points=[cx - 20, cy, cx + 20, cy], width=1.5)
            Line(points=[cx, cy - 20, cx, cy + 20], width=1.5)
            Line(circle=(cx, cy, 8), width=1.2)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        # Double-click pixel centering in live viewport
        if touch.is_double_tap and touch.button == 'left':
            center_x = self.x + self.width / 2.0
            center_y = self.y + self.height / 2.0
            dx_px = touch.x - center_x
            dy_px = touch.y - center_y

            app = App.get_running_app()
            if app and hasattr(app, 'focus_map'):
                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.6)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)
                scale_x = fov_w / float(self.width)
                scale_y = fov_h / float(self.height)

                dx_mm = dx_px * scale_x
                dy_mm = dy_px * scale_y

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


class LumaViewProMainLayout(BoxLayout):
    """Main Application Root Window Layout."""
    status_msg = StringProperty("LumaViewPro Microscope Automation Engine Ready")
    stage_coord_str = StringProperty("Stage: X=-49.50, Y=31.50, Z=0.000 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")
    current_well_display = StringProperty("A1")
    is_live_imaging = BooleanProperty(True)
    is_light_on = BooleanProperty(False)
    live_fps_str = StringProperty("20 FPS")
    exposure_str = StringProperty("20.0")
    gain_str = StringProperty("1.0")
    led_power_str = StringProperty("100.0")
    selected_plate_format_str = StringProperty("96-Well Plate")
    camera_gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)


class LumaViewProApp(App):
    # Class-level Kivy properties so KV bindings evaluate safely
    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)
    """Main Application Controller."""

    # Explicit class properties bound to KV layout and callbacks
    is_live_imaging = BooleanProperty(True)
    is_light_on = BooleanProperty(False)  # LAUNCHES WITH LIGHT OFF
    active_channel = StringProperty("Brightfield")
    exposure_ms = NumericProperty(20.0)
    camera_exposure_ms = NumericProperty(20.0)
    gain = NumericProperty(1.0)
    camera_gain = NumericProperty(1.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)
    stage_x = NumericProperty(-49.5)
    stage_y = NumericProperty(31.5)
    stage_z = NumericProperty(0.0)

    def load_kv(self, filename=None):
        return None

    def build(self):
        self.title = "LumaViewPro Microscope Automation"
        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(
            self.well_plate_format,
            self.plate_calibration,
            self.focus_map
        )

        self._updating_channel_ui = False
        self.camera = None
        self.init_hardware()

        self.root_layout = LumaViewProMainLayout()

        # Connect microplate canvas references in top-right
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.set_references(
                self.well_plate_format,
                self.plate_calibration,
                self.focus_map,
                self.fov_manager
            )

        self.build_channel_controls()
        self.update_active_well_display()

        if not getattr(self, 'stage_connected', False):
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

        # Start live camera feed update loop at 20 FPS (0.05s interval)
        Clock.schedule_interval(self.update_live_feed, 0.05)
        return self.root_layout

    def init_hardware(self):
        self.camera_connected = False
        self.stage_connected = False
        self.focus_connected = False
        self.led_connected = False

        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            if cam.connect():
                self.camera = cam
                self.camera_connected = True
            else:
                self.camera = None
        except Exception:
            self.camera = None

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            if stg.connect():
                self.motor_stage = stg
                self.stage_connected = True
                self.focus_connected = True
            else:
                self.motor_stage = None
        except Exception:
            self.motor_stage = None

        try:
            import ledboard
            led = ledboard.LEDBoard()
            self.leds = led
            self.led_connected = True
        except Exception:
            self.leds = None

    def clamp_stage_position(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Hard software guardrail: FOV edges never exceed physical well plate boundary."""
        fov_w = getattr(self.focus_map, 'fov_width_mm', 0.60)
        fov_h = getattr(self.focus_map, 'fov_height_mm', 0.45)

        plate_w = getattr(self.well_plate_format, 'PLATE_LENGTH_MM', 127.76)
        plate_h = getattr(self.well_plate_format, 'PLATE_WIDTH_MM', 85.48)

        min_x = -plate_w / 2.0 + fov_w / 2.0
        max_x =  plate_w / 2.0 - fov_w / 2.0
        min_y = -plate_h / 2.0 + fov_h / 2.0
        max_y =  plate_h / 2.0 - fov_h / 2.0

        clamped_x = max(min_x, min(max_x, float(x)))
        clamped_y = max(min_y, min(max_y, float(y)))
        clamped_z = max(0.0, min(10.0, float(z)))

        return clamped_x, clamped_y, clamped_z

    def update_active_well_display(self):
        try:
            a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
            a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
            c = int(round((self.stage_x - a1_x) / self.well_plate_format.pitch_x))
            r = int(round((a1_y - self.stage_y) / self.well_plate_format.pitch_y))
            c = max(0, min(self.well_plate_format.cols - 1, c))
            r = max(0, min(self.well_plate_format.rows - 1, r))
            w_name = self.well_plate_format.get_well_name(r, c)
            self.root_layout.current_well_display = w_name
        except Exception:
            pass

    def redraw_plate_canvas(self):
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.redraw_canvas()

    def build_channel_controls(self):
        if getattr(self, '_updating_channel_ui', False):
            return
        self._updating_channel_ui = True

        try:
            if hasattr(self.root_layout, 'ids') and hasattr(self.root_layout.ids, 'main_channel_container'):
                container = self.root_layout.ids.main_channel_container
                container.clear_widgets()

                for ch_name, ch_cfg in self.channel_manager.channels.items():
                    widget = ChannelControlWidget()
                    widget.channel_name = ch_name
                    widget.is_active = (ch_name == self.active_channel) and self.is_light_on
                    widget.btn_bg_color = ch_cfg.btn_color
                    if ch_cfg.is_transmitted:
                        widget.channel_spec_info = f"{ch_cfg.excitation_range_nm}"
                    else:
                        widget.channel_spec_info = f"Ex: {ch_cfg.excitation_range_nm} | Em: {ch_cfg.emission_range_nm}"
                    widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref"
                    container.add_widget(widget)
        finally:
            self._updating_channel_ui = False

    def select_optical_channel(self, channel_name: str):
        if getattr(self, '_updating_channel_ui', False):
            return

        real_ch = self.channel_manager.resolve_channel_name(channel_name)
        if self.active_channel == real_ch and self.is_light_on:
            self.is_light_on = False
            self.root_layout.is_light_on = False
            self.channel_manager.set_channel_active(real_ch, False)
            self.root_layout.status_msg = f"Turned OFF Channel {real_ch}"
        else:
            self.active_channel = real_ch
            self.is_light_on = True
            self.root_layout.is_light_on = True
            self.channel_manager.set_channel_active(real_ch, True)
            self.root_layout.status_msg = f"Selected Channel: {real_ch} | Light: ON"

        self.build_channel_controls()
        self.update_live_feed(0)

    def set_active_channel(self, channel_name: str, active_bool: bool = True):
        self.select_optical_channel(channel_name)

    def toggle_light_state(self, is_on: Optional[bool] = None):
        if is_on is not None:
            self.is_light_on = is_on
        else:
            self.is_light_on = not self.is_light_on
        self.root_layout.is_light_on = self.is_light_on
        status = "ON" if self.is_light_on else "OFF (Dark)"
        self.root_layout.status_msg = f"Illumination Light toggled {status}"
        self.build_channel_controls()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    def set_well_plate_format(self, format_str: str):
        try:
            digits = re.findall(r'\d+', format_str)
            num_wells = int(digits[0]) if digits else 96
            self.well_plate_format = labware.WellPlateFormat(num_wells=num_wells)
            self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
            self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

            # Center stage at Well A1 of selected format
            a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
            a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
            cx, cy, cz = self.clamp_stage_position(a1_x, a1_y, self.stage_z)
            self.stage_x, self.stage_y, self.stage_z = cx, cy, cz

            if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                self.root_layout.ids.main_plate_canvas.set_references(
                    self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
                )

            self.root_layout.selected_plate_format_str = f"{num_wells}-Well Plate"
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
            self.root_layout.status_msg = f"Well Plate Map updated to {num_wells}-Well Format"
            self.update_active_well_display()
            self.redraw_plate_canvas()
            self.update_live_feed(0)
        except Exception as e:
            self.root_layout.status_msg = f"Failed to set plate format: {e}"

    def change_well_plate_format(self, format_str: str):
        self.set_well_plate_format(format_str)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        if hasattr(self, 'fov_manager'):
            self.fov_manager._rebuild_all_groups()

        # Re-clamp stage position with new objective FOV guardrail!
        cx, cy, cz = self.clamp_stage_position(self.stage_x, self.stage_y, self.stage_z)
        self.stage_x, self.stage_y, self.stage_z = cx, cy, cz

        self.root_layout.status_msg = f"Objective Lens set to: {preset_name} (Mag: {self.focus_map.lens_mag}x, FOV: {self.focus_map.fov_width_mm:.2f}x{self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def jog_stage(self, dx: float, dy: float, dz: float = 0.0):
        if not getattr(self, 'stage_connected', False) and (dx != 0.0 or dy != 0.0):
            if hasattr(self, 'root_layout') and self.root_layout:
                self.root_layout.status_msg = "Stage Motor DISCONNECTED - Motion Disabled"
            return
        if not getattr(self, 'focus_connected', False) and dz != 0.0:
            if hasattr(self, 'root_layout') and self.root_layout:
                self.root_layout.status_msg = "Z Focus Motor DISCONNECTED - Focus Motion Disabled"
            return

        step = self.jog_step_mm
        tx = self.stage_x + dx * step
        ty = self.stage_y + dy * step
        tz = self.stage_z + dz

        cx, cy, cz = self.clamp_stage_position(tx, ty, tz)
        self.stage_x, self.stage_y, self.stage_z = cx, cy, cz

        if hasattr(self, 'motor_stage') and self.motor_stage and hasattr(self.motor_stage, 'move_rel'):
            try:
                self.motor_stage.move_rel(dx * step, dy * step, dz)
            except Exception as e:
                print(f"[Hardware Error] move_rel failed: {e}")

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        if not getattr(self, 'stage_connected', False):
            if hasattr(self, 'root_layout') and self.root_layout:
                self.root_layout.status_msg = "Stage Motor DISCONNECTED - Motion Disabled"
            return

        tz = z_mm if z_mm is not None else self.stage_z
        cx, cy, cz = self.clamp_stage_position(x_mm, y_mm, tz)
        self.stage_x, self.stage_y, self.stage_z = cx, cy, cz

        if hasattr(self, 'motor_stage') and self.motor_stage and hasattr(self.motor_stage, 'move_abs'):
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[Hardware Error] move_abs failed: {e}")

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def jog_z_focus(self, delta_um: float):
        self.jog_stage(0.0, 0.0, delta_um / 1000.0)

    def jog_z_microns(self, delta_um: float):
        self.jog_z_focus(delta_um)

    def jog_z(self, delta_um: float):
        self.jog_z_focus(delta_um)

    def set_jog_step_size(self, step_str: str):
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
        except (IndexError, ValueError):
            self.jog_step_mm = 1.0

    def update_live_feed(self, dt: float):
        if not hasattr(self, 'root_layout') or self.root_layout is None:
            return

        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        if hasattr(self, 'camera') and self.camera is not None and getattr(self.camera, 'is_connected', False):
            if hasattr(self.camera, 'stage_x'):
                self.camera.stage_x = self.stage_x
                self.camera.stage_y = self.stage_y
                self.camera.stage_z = self.stage_z
                self.camera.active_channel = self.active_channel
                self.camera.is_light_on = self.is_light_on
                self.camera.led_power = self.led_power
                self.camera.exposure_ms = self.exposure_ms
                self.camera.gain = self.gain
                self.camera.fov_size_mm = (fov_w, fov_h)
                self.camera.num_wells = self.well_plate_format.num_wells

            raw_frame = self.camera.get_frame()
            if raw_frame is None or not self.is_light_on:
                raw_frame = image_utils.generate_well_plate_cell_frame(
                    stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                    best_z=0.0, channel=self.active_channel, exposure_ms=self.exposure_ms,
                    gain=self.gain, led_power=self.led_power, is_light_on=self.is_light_on,
                    fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
                )
        else:
            raw_frame = image_utils.generate_well_plate_cell_frame(
                stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                best_z=0.0, channel=self.active_channel, exposure_ms=self.exposure_ms,
                gain=self.gain, led_power=self.led_power, is_light_on=self.is_light_on,
                fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
            )

        proc_frame = self.channel_manager.process_channel_image(self.active_channel, raw_frame)
        self.render_frame_to_viewports(proc_frame)

    def render_frame_to_viewports(self, frame: np.ndarray):
        if frame is None or self.root_layout is None:
            return

        if hasattr(self.root_layout, 'ids') and 'main_camera_viewport' in self.root_layout.ids:
            vp = self.root_layout.ids.main_camera_viewport
            if hasattr(vp, 'set_frame'):
                vp.set_frame(frame)

    def set_exposure_ms(self, text_val: str):
        try:
            val = float(text_val)
            self.exposure_ms = val
            self.camera_exposure_ms = val
            self.root_layout.exposure_str = f"{val:.1f}"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_gain_val(self, text_val: str):
        try:
            val = float(text_val)
            self.gain = val
            self.camera_gain = val
            self.root_layout.gain_str = f"{val:.1f}"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            self.root_layout.led_power_str = f"{self.led_power:.0f}"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_primary_af_channel(self, channel_name: str):
        self.focus_map.set_primary_af_channel(channel_name)
        self.root_layout.status_msg = f"Primary AF Channel set to: {channel_name}"

    def set_af_binning(self, binning_str: str):
        self.focus_map.set_af_camera_config(binning_str, self.focus_map.af_exposure_ms)
        self.root_layout.status_msg = f"AF Camera Binning: {binning_str}"

    def set_af_exposure(self, exposure_str: str):
        self.set_exposure_ms(exposure_str)

    def set_af_cycle_interval(self, text_or_num):
        try:
            digits = re.findall(r'\d+', str(text_or_num))
            val = int(digits[0]) if digits else 1
            self.focus_map.set_af_cycle_interval(val)
            self.root_layout.status_msg = f"Autofocus configured every {val} cycle(s)"
        except Exception:
            pass

    def set_time_lapse_config(self, enabled: bool, interval_mins_str: str, cycles_str: str):
        try:
            inv_m = float(interval_mins_str) if interval_mins_str else 10.0
            cyc = int(cycles_str) if cycles_str else 24
            self.focus_map.set_time_lapse_config(enabled, inv_m, cyc)
        except ValueError:
            pass

    def set_chromatic_z_offset(self, channel_name: str):
        self.focus_map.set_chromatic_z_offset(channel_name, self.stage_z)
        delta_z = self.focus_map.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        self.root_layout.status_msg = f"Saved Chromatic Z-Offset for {channel_name}: {delta_z*1000.0:.2f} um"

    def apply_nyquist_calculation(self):
        dof_um, nyquist_step_um = self.focus_map.calculate_optical_nyquist()
        self.focus_map.apply_nyquist_to_af_config()
        self.root_layout.status_msg = f"Applied Nyquist Z-Step: {nyquist_step_um:.3f} um (DoF: {dof_um:.3f} um)"

    def apply_grid_settings(self, rows_str: str, cols_str: str, single_af: bool = True):
        try:
            r = int(rows_str)
            c = int(cols_str)
            self.fov_manager.set_grid_dimensions(r, c, single_af=single_af)
            self.root_layout.status_msg = f"Tile Grid set to {r}x{c}"
            if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                self.root_layout.ids.main_plate_canvas.redraw_canvas()
        except ValueError:
            pass

    def record_calibration_step(self):
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)
        self.root_layout.status_msg = msg
        self.redraw_plate_canvas()

    def finish_plate_calibration(self):
        success = self.plate_calibration.compute_calibration()
        if success:
            self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
            self.root_layout.status_msg = "Plate Registration Matrix successfully computed!"
        else:
            self.root_layout.status_msg = "Calibration failed. Verify edge target points."
        self.redraw_plate_canvas()

    def set_master_auto_point(self):
        sim_frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=0.0, channel=self.focus_map.primary_af_channel, is_light_on=True
        )
        af_score = image_utils.compute_focus_score(sim_frame, method="laplacian")
        del sim_frame

        self.focus_map.set_master_auto_point(self.stage_x, self.stage_y, self.stage_z)
        self.root_layout.status_msg = f"Master Auto Point set (Red). Primary: {self.focus_map.primary_af_channel}, Score: {af_score:.1f}"
        self.redraw_plate_canvas()

    def test_autofocus_run(self):
        af_frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=self.stage_z, channel=self.focus_map.primary_af_channel, is_light_on=True
        )
        score = image_utils.compute_focus_score(af_frame, method="laplacian")
        del af_frame
        self.root_layout.status_msg = f"[Test AF Run] Score: {score:.2f} (Frame Discarded)"

    def generate_and_apply_protocol(self) -> dict:
        protocol = labware.ExperimentProtocol.generate_protocol(
            self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
        )
        print("[Protocol Generated]", json.dumps(protocol, indent=2))
        self.root_layout.status_msg = f"Hardware Protocol Saved! Selected Wells: {protocol['total_selected_wells']}"
        return protocol

    def capture_snapshot(self):
        frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=0.0, channel=self.active_channel, is_light_on=self.is_light_on,
            led_power=self.led_power, num_wells=self.well_plate_format.num_wells
        )
        os.makedirs("snapshots", exist_ok=True)
        fn = f"snapshots/snapshot_{self.active_channel}_{self.root_layout.current_well_display}.png"
        image_utils.save_image(fn, frame)
        self.root_layout.status_msg = f"Saved Snapshot: {fn}"

    def open_navigator_popup(self):
        plate_navigator.attach_navigator_to_lumaviewpro(self)

    def open_plate_navigator(self):
        self.open_navigator_popup()



    def global_sync_active_channels(self, active_channel_states: Optional[Dict[str, bool]] = None):
        self.channel_manager.global_sync_active_channels(active_channel_states)
        active_list = self.channel_manager.get_active_channel_names()
        self.fov_manager.apply_channels_to_all_selected(active_list)
        if hasattr(self.root_layout, 'status_msg'):
            self.root_layout.status_msg = f"Synced active channels ({', '.join(active_list)}) across selected wells"

    def start_calibration_wizard(self):
        self.plate_calibration.reset_calibration()
        self.plate_calibration.is_calibrating = True
        self.is_light_on = True
        self.root_layout.is_light_on = True

        tx, ty = self.plate_calibration.get_step_nominal_stage_coords(0)
        self.jog_stage_to(tx, ty)

        step_info = self.plate_calibration.get_current_step_info()
        self.root_layout.calibration_wizard_text = f"Step 1/6: Align reticle to {step_info['desc']}, then click Mark"
        self.root_layout.status_msg = "Calibration Wizard Started. Light ON. Stage moved to A1 12 o'clock."
        self.redraw_plate_canvas()

    def mark_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            self.start_calibration_wizard()
            return

        step_idx = self.plate_calibration.current_step_index
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS):
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Point {step_idx + 1} recorded. Moving to Step {next_idx + 1}..."
        else:
            success = self.plate_calibration.compute_calibration()
            if success:
                self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
                self.root_layout.calibration_wizard_text = f"Calibration Complete! Rotation: {self.plate_calibration.rotation_angle_deg:.2f}°"
                self.root_layout.status_msg = "Plate Registration Matrix successfully computed and applied!"
            else:
                self.root_layout.calibration_wizard_text = "Calibration failed. Verify edge target points."
                self.root_layout.status_msg = "Calibration computation failed. Verify edge target alignments."

        self.redraw_plate_canvas()

    def next_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            self.start_calibration_wizard()
            return

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS) - 1:
            self.plate_calibration.current_step_index += 1
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved to Step {next_idx + 1}/6."
            self.redraw_plate_canvas()

    def previous_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            return

        if self.plate_calibration.current_step_index > 0:
            self.plate_calibration.current_step_index -= 1
            prev_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(prev_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {prev_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved back to Step {prev_idx + 1}/6."
            self.redraw_plate_canvas()

    def reset_calibration(self):
        self.plate_calibration.reset_calibration()
        self.root_layout.calibrated_status = "Plate Uncalibrated"
        self.root_layout.calibration_wizard_text = "Click 'Start Calib' to begin 6-point plate alignment."
        self.root_layout.status_msg = "Calibration Reset."
        self.redraw_plate_canvas()

# Register custom widget classes with Factory BEFORE loading KV files
from kivy.factory import Factory

if 'ZoomPanCameraWidget' not in Factory.classes:
    Factory.register('ZoomPanCameraWidget', ZoomPanCameraWidget)
if 'ChannelControlWidget' not in Factory.classes:
    Factory.register('ChannelControlWidget', ChannelControlWidget)
if 'LumaViewProMainLayout' not in Factory.classes:
    Factory.register('LumaViewProMainLayout', LumaViewProMainLayout)
if 'InteractiveScaledPlateCanvas' not in Factory.classes:
    Factory.register('InteractiveScaledPlateCanvas', plate_navigator.InteractiveScaledPlateCanvas)
if 'PlateNavigatorPopup' not in Factory.classes:
    Factory.register('PlateNavigatorPopup', plate_navigator.PlateNavigatorPopup)

_main_kv = os.path.join(script_dir, 'lumaviewpro.kv')
if os.path.exists(_main_kv) and _main_kv not in Builder.files:
    Builder.load_file(_main_kv)



    def start_calibration_wizard(self):
        self.plate_calibration.reset_calibration()
        self.plate_calibration.is_calibrating = True
        self.is_light_on = True
        self.root_layout.is_light_on = True

        tx, ty = self.plate_calibration.get_step_nominal_stage_coords(0)
        self.jog_stage_to(tx, ty)

        step_info = self.plate_calibration.get_current_step_info()
        self.root_layout.calibration_wizard_text = f"Step 1/6: Align reticle to {step_info['desc']}, then click Mark"
        self.root_layout.status_msg = "Calibration Wizard Started. Light ON. Stage moved to A1 12 o'clock."
        self.redraw_plate_canvas()

    def mark_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            self.start_calibration_wizard()
            return

        step_idx = self.plate_calibration.current_step_index
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS):
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Point {step_idx + 1} recorded. Moving to Step {next_idx + 1}..."
        else:
            success = self.plate_calibration.compute_calibration()
            if success:
                self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
                self.root_layout.calibration_wizard_text = f"Calibration Complete! Rotation: {self.plate_calibration.rotation_angle_deg:.2f}°"
                self.root_layout.status_msg = "Plate Registration Matrix successfully computed and applied!"
            else:
                self.root_layout.calibration_wizard_text = "Calibration failed. Verify edge target points."
                self.root_layout.status_msg = "Calibration computation failed. Verify edge target alignments."

        self.redraw_plate_canvas()

    def next_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            self.start_calibration_wizard()
            return

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS) - 1:
            self.plate_calibration.current_step_index += 1
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved to Step {next_idx + 1}/6."
            self.redraw_plate_canvas()

    def previous_calibration_step(self):
        if not getattr(self.plate_calibration, 'is_calibrating', False):
            return

        if self.plate_calibration.current_step_index > 0:
            self.plate_calibration.current_step_index -= 1
            prev_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(prev_idx)
            self.jog_stage_to(tx, ty)

            step_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {prev_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved back to Step {prev_idx + 1}/6."
            self.redraw_plate_canvas()

    def reset_calibration(self):
        self.plate_calibration.reset_calibration()
        self.root_layout.calibrated_status = "Plate Uncalibrated"
        self.root_layout.calibration_wizard_text = "Click 'Start Calib' to begin 6-point plate alignment."
        self.root_layout.status_msg = "Calibration Reset."
        self.redraw_plate_canvas()

    def global_sync_active_channels(self):
        active_list = self.channel_manager.get_active_channel_names()
        self.fov_manager.apply_channels_to_all_selected(active_list)
        self.root_layout.status_msg = f"Synced active channels ({', '.join(active_list)}) across selected wells"


if __name__ == '__main__':
    LumaViewProApp().run()
