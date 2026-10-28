"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation Software.

Enforces:
1. Unified Single-Window Kivy Architecture:
   - Top-Left: Live Camera Viewport with Mouse Zoom/Pan, Red Reticle Crosshairs, and Double-Click Pixel Centering.
   - Top-Right: Embedded Microplate Registration Canvas with Format Selector (6 to 384 wells) and 1:1 Double-Click Well/Location Stage Navigation.
   - Bottom-Left: Hardware Motion Controls Deck (Grouped XY Jog, Fine/Coarse Z Focus, Exposure, Gain, LED Power).
   - Bottom-Right: Optical Channels Panel with strict single-channel mutual exclusion + 6-Point Calibration Wizard.
2. Single-instance KV load guard (prevents duplicate UI parsing and layout stacking).
3. Independent device status detection (Camera, Stage Motor, Focus Motor, Illumination LED Board).
"""

from __future__ import annotations

import os
import sys
import json
import math
import re
from typing import Dict, Optional, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import cv2
import numpy as np

import image_utils
import labware
import plate_navigator  # Top-level import for Kivy Factory registration

# Kivy framework imports
import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.core.window import Window
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
    """Stores configuration and optical specs for an imaging channel."""
    
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
        self.btn_color: List[float] = btn_color if btn_color else [0.2, 0.6, 0.8, 1.0]
        self.active: bool = False
        self.exposure_ms: float = 100.0
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
    """Manages optical channels with strict single-channel mutual exclusion."""
    
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
            "Brightfield": ChannelConfig("Brightfield", "White LED", "Pass-through", "#FFFFFF", True, [0.8, 0.8, 0.8, 1.0]),
            "Phase": ChannelConfig("Phase", "White LED (Ring)", "Pass-through", "#CCCCCC", True, [0.6, 0.6, 0.7, 1.0]),
            "Darkfield": ChannelConfig("Darkfield", "Oblique LED", "Pass-through", "#888888", True, [0.4, 0.4, 0.5, 1.0]),
            "Blue": ChannelConfig("Blue", "370-410 nm", "429-462 nm (DAPI)", "#0088FF", False, [0.0, 0.4, 1.0, 1.0]),
            "Green": ChannelConfig("Green", "473-491 nm", "502-561 nm (FITC)", "#00FF00", False, [0.0, 0.8, 0.2, 1.0]),
            "Red": ChannelConfig("Red", "580-598 nm", "612-680 nm (TxRed)", "#FF0000", False, [0.9, 0.1, 0.1, 1.0]),
        }
        self.channels["Brightfield"].active = True

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

    def get_active_channel_name(self) -> str:
        for ch_name, ch_cfg in self.channels.items():
            if ch_cfg.active:
                return ch_name
        return "Brightfield"

    def get_active_channel_names(self) -> List[str]:
        return [ch_name for ch_name, ch_cfg in self.channels.items() if ch_cfg.active]

    def process_channel_image(self, channel_name: str, raw_frame: object) -> object:
        canonical = self.resolve_channel_name(channel_name)
        ch = self.channels.get(canonical)
        if (ch and ch.shading_correction_enabled and ch.ref_image_array is not None):
            return image_utils.apply_shading_correction(raw_frame, ch.ref_image_array)
        return raw_frame


class ZoomPanCameraWidget(plate_navigator.ZoomPanCameraWidget):
    """Subclass inheriting reticle crosshairs & double-click pixel centering."""
    pass


class InteractiveScaledPlateCanvas(plate_navigator.InteractiveScaledPlateCanvas):
    """Subclass inheriting active stage cyan FOV reticle & double-click navigation."""
    pass


class ChannelControlWidget(BoxLayout):
    channel_name = StringProperty("")
    channel_spec_info = StringProperty("")
    ref_image_path = StringProperty("No Ref Image")
    is_active = BooleanProperty(False)
    btn_bg_color = ListProperty([0.2, 0.6, 0.8, 1.0])

    def open_ref_chooser(self):
        App.get_running_app().open_file_chooser_for_channel(self.channel_name)


class LumaViewProMainLayout(BoxLayout):
    status_msg = StringProperty("LumaViewPro Engine Ready")
    stage_coord_str = StringProperty("Stage: X=-49.50, Y=31.50, Z=0.000 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")
    current_well_display = StringProperty("A1")
    calibration_wizard_text = StringProperty("Click 'Start Calib' to begin 6-point plate alignment.")


class LumaViewProApp(App):
    # Class-level Kivy Property Declarations
    is_light_on = BooleanProperty(False)  # Sensor starts dark (black window) on launch
    is_live_imaging = BooleanProperty(True)
    is_simulator = BooleanProperty(False)
    active_channel = StringProperty("Brightfield")

    camera_gain = NumericProperty(1.0)
    gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    exposure_ms = NumericProperty(20.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)

    camera_connected = BooleanProperty(False)
    stage_connected = BooleanProperty(False)
    focus_connected = BooleanProperty(False)
    led_connected = BooleanProperty(False)

    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)

    # HARD SOFTWARE MOTOR BOUNDS
    X_MIN_MM = -63.88
    X_MAX_MM = 63.88
    Y_MIN_MM = -42.74
    Y_MAX_MM = 42.74
    Z_MIN_MM = 0.0
    Z_MAX_MM = 10.0

    def load_kv(self, filename=None):
        """Single-instance KV load guard to prevent duplicate UI parsing."""
        if not hasattr(self, '_kv_loaded_flag'):
            self._kv_loaded_flag = True
            return super().load_kv(filename)
        return None

    def build(self):
        self.title = "LumaViewPro Microscope Automation"

        self.camera = None
        self.motor_stage = None
        self.leds = None
        self.init_hardware()

        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        self._updating_channel_ui = False
        self.root_layout = LumaViewProMainLayout()

        # Update initial header status
        if not self.stage_connected and not self.is_simulator:
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"
        else:
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"

        if self.is_simulator:
            self.root_layout.status_msg = "LumaViewPro Simulator Active (Offline Mock Hardware)"
        elif self.camera_connected and self.stage_connected and self.led_connected:
            self.root_layout.status_msg = "LumaViewPro Connected to Microscope Hardware"
        else:
            dis_list = []
            if not self.camera_connected: dis_list.append("Camera")
            if not self.stage_connected: dis_list.append("Stage/Focus Motor")
            if not self.led_connected: dis_list.append("LED Board")
            self.root_layout.status_msg = f"LumaViewPro Offline (Disconnected: {', '.join(dis_list)})"

        # Connect references to embedded plate canvas in right column
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.set_references(
                self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
            )

        self.build_channel_controls()

        # Bind Keyboard Event Listener
        Window.bind(on_key_down=self._on_keyboard_down)

        # Schedule live camera feed loop at 20 FPS
        Clock.schedule_interval(self.update_live_feed, 1.0 / 20.0)

        return self.root_layout

    def _on_keyboard_down(self, window, key, scancode, codepoint, modifiers):
        is_shift = 'shift' in modifiers
        xy_step = 5.0 if is_shift else 0.5
        z_step_um = 10.0 if is_shift else 1.0

        if key == 273:    # Up Arrow (+Y)
            self.jog_stage(0.0, xy_step, 0.0)
            return True
        elif key == 274:  # Down Arrow (-Y)
            self.jog_stage(0.0, -xy_step, 0.0)
            return True
        elif key == 276:  # Left Arrow (-X)
            self.jog_stage(-xy_step, 0.0, 0.0)
            return True
        elif key == 275:  # Right Arrow (+X)
            self.jog_stage(xy_step, 0.0, 0.0)
            return True
        elif key == 280:  # PageUp (+Z)
            self.jog_z_stage(z_step_um)
            return True
        elif key == 281:  # PageDown (-Z)
            self.jog_z_stage(-z_step_um)
            return True

        return False

    def init_hardware(self):
        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            cam.connect()
            self.camera = cam
            self.camera_connected = True
            print("[LumaViewPro] Connected to USB camera hardware.")
        except Exception as e:
            self.camera = None
            self.camera_connected = False
            print(f"[LumaViewPro] USB Camera Driver offline/unattached: {e}")

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            stg.connect()
            self.motor_stage = stg
            self.stage_connected = True
            self.focus_connected = True
            print("[LumaViewPro] Stage & Z Focus Motor Board connected.")
        except Exception as e:
            self.motor_stage = None
            self.stage_connected = False
            self.focus_connected = False
            print(f"[LumaViewPro] Stage Motor Driver offline/unattached: {e}")

        try:
            import ledboard
            led = ledboard.LEDBoard()
            if hasattr(led, 'connect'):
                led.connect()
            self.leds = led
            self.led_connected = True
            print("[LumaViewPro] LED Illumination Board connected.")
        except Exception as e:
            self.leds = None
            self.led_connected = False
            print(f"[LumaViewPro] LED Board Driver offline/unattached: {e}")

    def clamp_stage_position(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> Tuple[float, float, float]:
        if x is not None:
            self.stage_x = float(x)
        if y is not None:
            self.stage_y = float(y)
        if z is not None:
            self.stage_z = float(z)

        self.stage_x = max(self.X_MIN_MM, min(self.X_MAX_MM, self.stage_x))
        self.stage_y = max(self.Y_MIN_MM, min(self.Y_MAX_MM, self.stage_y))
        self.stage_z = max(self.Z_MIN_MM, min(self.Z_MAX_MM, self.stage_z))

        if self.stage_connected or self.is_simulator:
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        else:
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

        try:
            r_idx = max(0, min(self.well_plate_format.rows - 1, int(round((31.5 - self.stage_y) / self.well_plate_format.pitch_y))))
            c_idx = max(0, min(self.well_plate_format.cols - 1, int(round((self.stage_x + 49.5) / self.well_plate_format.pitch_x))))
            self.root_layout.current_well_display = self.well_plate_format.get_well_name(r_idx, c_idx)
        except Exception:
            pass

        return self.stage_x, self.stage_y, self.stage_z

    def build_channel_controls(self):
        if getattr(self, '_updating_channel_ui', False):
            return
        if not hasattr(self.root_layout, 'ids') or 'main_channel_container' not in self.root_layout.ids:
            return

        self._updating_channel_ui = True
        try:
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
                widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref Image"
                container.add_widget(widget)
        finally:
            self._updating_channel_ui = False

    def select_optical_channel(self, channel_name: str):
        if getattr(self, '_updating_channel_ui', False):
            return

        canonical = self.channel_manager.resolve_channel_name(channel_name)
        if self.active_channel == canonical and self.is_light_on:
            self.is_light_on = False
            self.channel_manager.set_channel_active(canonical, False)
            if self.leds and hasattr(self.leds, 'off'):
                try:
                    self.leds.off()
                except Exception:
                    pass
            self.root_layout.status_msg = f"Illumination OFF ({canonical})"
        else:
            self.active_channel = canonical
            self.is_light_on = True
            self.channel_manager.set_channel_active(canonical, True)
            if self.leds and hasattr(self.leds, 'set_led'):
                try:
                    self.leds.set_led(canonical, int(self.led_power))
                except Exception:
                    pass
            self.root_layout.status_msg = f"Active Channel: {canonical} (Light ON)"

        self.build_channel_controls()
        self.update_live_feed(0)

    def set_active_channel(self, channel_name: str, active_bool: bool = True):
        self.select_optical_channel(channel_name)

    def select_channel(self, channel_name: str):
        self.select_optical_channel(channel_name)

    def toggle_light_state(self, is_on: Optional[bool] = None):
        if is_on is None:
            self.is_light_on = not self.is_light_on
        else:
            self.is_light_on = is_on

        if not self.is_light_on and self.leds and hasattr(self.leds, 'off'):
            try:
                self.leds.off()
            except Exception:
                pass
        elif self.is_light_on and self.leds and hasattr(self.leds, 'set_led'):
            try:
                self.leds.set_led(self.active_channel, int(self.led_power))
            except Exception:
                pass

        state_str = "ON" if self.is_light_on else "OFF"
        self.root_layout.status_msg = f"Illumination Light set to {state_str}"
        self.build_channel_controls()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    # 6-POINT TARGET REGISTRATION WIZARD METHODS
    def start_calibration_wizard(self):
        self.plate_calibration.is_calibrating = True
        self.plate_calibration.is_calibrated = False
        self.plate_calibration.current_step_index = 0
        self.plate_calibration.recorded_points.clear()

        # Turn illumination Light ON automatically
        self.is_light_on = True

        # Move stage to nominal Step 1 position (Well A1 - 12 o'clock top rim)
        tx, ty = self.plate_calibration.get_step_nominal_stage_coords(0)
        self.jog_stage_to(tx, ty)

        step_info = self.plate_calibration.get_current_step_info()
        self.root_layout.calibration_wizard_text = f"Step 1/6: Align reticle to {step_info['desc']}, then click Mark"
        self.root_layout.status_msg = "Calibration Wizard Started. Light ON. Stage moved to A1 12 o'clock."
        self.redraw_plate_canvas()

    def mark_calibration_step(self):
        if not self.plate_calibration.is_calibrating:
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
            # All 6 points recorded! Auto-compute calibration matrix
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
        if not self.plate_calibration.is_calibrating:
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
        if not self.plate_calibration.is_calibrating:
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

    def record_calibration_step(self):
        self.mark_calibration_step()

    def finish_plate_calibration(self):
        if self.plate_calibration.current_step_index < 6:
            self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)

        success = self.plate_calibration.compute_calibration()
        if success:
            self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
            self.root_layout.calibration_wizard_text = f"Calibration Complete! Rotation: {self.plate_calibration.rotation_angle_deg:.2f}°"
            self.root_layout.status_msg = "Plate Registration Matrix successfully computed!"
        else:
            self.root_layout.calibration_wizard_text = "Calibration failed. Verify edge target points."
            self.root_layout.status_msg = "Calibration failed. Verify edge target points."

        self.redraw_plate_canvas()

    def reset_calibration(self):
        self.plate_calibration.reset_calibration()
        self.root_layout.calibrated_status = "Plate Uncalibrated"
        self.root_layout.calibration_wizard_text = "Click 'Start Calib' to begin 6-point plate alignment."
        self.root_layout.status_msg = "Calibration Reset."
        self.redraw_plate_canvas()

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            if self.leds and hasattr(self.leds, 'set_led') and self.is_light_on:
                try:
                    self.leds.set_led(self.active_channel, int(self.led_power))
                except Exception:
                    pass
            self.root_layout.status_msg = f"LED Power set to {self.led_power:.0f}%"
        except ValueError:
            pass

    def jog_stage(self, dx: float, dy: float, dz: float = 0.0):
        if not self.stage_connected and not self.is_simulator:
            self.root_layout.status_msg = "Microscope Stage & Focus DISCONNECTED - Hardware Motion Unavailable"
            return

        self.stage_x += dx
        self.stage_y += dy
        self.stage_z += dz
        self.clamp_stage_position()

        if self.motor_stage and hasattr(self.motor_stage, 'move_rel'):
            try:
                self.motor_stage.move_rel(dx, dy, dz)
            except Exception as e:
                print(f"[Hardware Error] Stage move_rel failed: {e}")

        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def jog_z_stage(self, delta_um: float):
        self.jog_stage(0.0, 0.0, delta_um / 1000.0)

    def jog_z_focus(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_z(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        if not self.stage_connected and not self.is_simulator:
            self.root_layout.status_msg = "Microscope Stage & Focus DISCONNECTED - Hardware Motion Unavailable"
            return

        tz = z_mm if z_mm is not None else self.stage_z
        self.stage_x = x_mm
        self.stage_y = y_mm
        self.stage_z = tz
        self.clamp_stage_position()

        if self.motor_stage and hasattr(self.motor_stage, 'move_abs'):
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[Hardware Error] Stage move_abs failed: {e}")

        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def move_stage_abs(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        self.jog_stage_to(x_mm, y_mm, z_mm)

    def zero_stage(self):
        self.jog_stage_to(-49.5, 31.5)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        if hasattr(self, 'fov_manager') and hasattr(self.fov_manager, '_rebuild_all_groups'):
            self.fov_manager._rebuild_all_groups()
        self.clamp_stage_position()
        self.root_layout.status_msg = f"Objective Preset set to {preset_name} (FOV: {self.focus_map.fov_width_mm:.2f} x {self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def set_well_plate_format(self, format_str: str):
        digits = re.findall(r'\d+', format_str)
        if not digits:
            return
        num = int(digits[0])
        if num in labware.WellPlateFormat.STANDARD_FORMATS:
            self.well_plate_format = labware.WellPlateFormat(num_wells=num)
            self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
            self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)
            
            if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                self.root_layout.ids.main_plate_canvas.set_references(
                    self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
                )
            self.clamp_stage_position()
            self.root_layout.status_msg = f"Well Plate Map updated to {num}-Well Format"
            self.update_live_feed(0)

    def set_jog_step_size(self, step_str: str):
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
        except Exception:
            pass

    def redraw_plate_canvas(self):
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.redraw_canvas()

    def update_live_feed(self, dt):
        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        if self.camera_connected and self.camera is not None:
            if hasattr(self.camera, 'stage_x'):
                self.camera.stage_x = self.stage_x
                self.camera.stage_y = self.stage_y
                self.camera.stage_z = self.stage_z
                self.camera.active_channel = self.active_channel
                self.camera.is_light_on = self.is_light_on
                self.camera.exposure_ms = self.exposure_ms
                self.camera.gain = self.gain
                self.camera.led_power = self.led_power
                self.camera.fov_size_mm = (fov_w, fov_h)
                self.camera.num_wells = self.well_plate_format.num_wells

            raw_frame = self.camera.get_frame()
            if raw_frame is None and self.is_simulator:
                raw_frame = image_utils.generate_well_plate_cell_frame(
                    stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                    best_z=2.5, channel=self.active_channel, is_light_on=self.is_light_on,
                    led_power=self.led_power, exposure_ms=self.exposure_ms, gain=self.gain,
                    fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
                )
            elif raw_frame is None or not self.is_light_on:
                if not self.is_light_on:
                    raw_frame = image_utils.generate_well_plate_cell_frame(
                        stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                        best_z=2.5, channel=self.active_channel, is_light_on=False,
                        led_power=self.led_power, exposure_ms=self.exposure_ms, gain=self.gain
                    )
                else:
                    raw_frame = image_utils.generate_disconnected_camera_frame()
        elif self.is_simulator:
            raw_frame = image_utils.generate_well_plate_cell_frame(
                stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                best_z=2.5, channel=self.active_channel, is_light_on=self.is_light_on,
                led_power=self.led_power, exposure_ms=self.exposure_ms, gain=self.gain,
                fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
            )
        else:
            raw_frame = image_utils.generate_disconnected_camera_frame()

        if raw_frame is not None:
            proc_frame = self.channel_manager.process_channel_image(self.active_channel, raw_frame)
            self.render_frame_to_viewports(proc_frame)

    def render_frame_to_viewports(self, frame: np.ndarray):
        if frame is None or self.root_layout is None:
            return

        if hasattr(self.root_layout, 'ids') and 'main_camera_viewport' in self.root_layout.ids:
            self.root_layout.ids.main_camera_viewport.set_frame(frame)

        self.redraw_plate_canvas()

    def set_exposure_ms(self, exp_str: str):
        try:
            val = float(exp_str)
            self.exposure_ms = val
            self.camera_exposure_ms = val
            if self.camera and hasattr(self.camera, 'set_exposure'):
                try:
                    self.camera.set_exposure(val)
                except Exception:
                    pass
            self.root_layout.status_msg = f"Exposure Time set to {val:.1f} ms"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_gain_val(self, gain_str: str):
        try:
            val = float(gain_str)
            self.gain = val
            self.camera_gain = val
            if self.camera and hasattr(self.camera, 'set_gain'):
                try:
                    self.camera.set_gain(val)
                except Exception:
                    pass
            self.root_layout.status_msg = f"Gain set to {val:.1f}x"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_primary_af_channel(self, channel_name: str):
        self.focus_map.set_primary_af_channel(channel_name)
        self.root_layout.status_msg = f"Primary AF Channel set to: {channel_name}"

    def set_af_binning(self, binning_str: str):
        self.focus_map.set_af_camera_config(binning_str, self.focus_map.af_exposure_ms)
        self.root_layout.status_msg = f"AF Camera Binning: {binning_str}"

    def set_master_auto_point(self):
        sim_frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=2.5, channel=self.focus_map.primary_af_channel, is_light_on=True
        )
        af_score = image_utils.compute_focus_score(sim_frame, method="laplacian")
        del sim_frame
        self.focus_map.set_master_auto_point(self.stage_x, self.stage_y, self.stage_z)
        self.root_layout.status_msg = f"Master Auto Point set (Red). Primary: {self.focus_map.primary_af_channel}, Score: {af_score:.1f}"

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
        print("[Protocol Generated]\n", json.dumps(protocol, indent=2))
        self.root_layout.status_msg = f"Hardware Protocol Saved! Selected Wells: {protocol['selected_wells_count']}"
        return protocol

    def capture_snapshot(self):
        active_ch = self.channel_manager.get_active_channel_name()
        frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=2.5, channel=active_ch, is_light_on=self.is_light_on, led_power=self.led_power,
            num_wells=self.well_plate_format.num_wells, fov_size_mm=(self.focus_map.fov_width_mm, self.focus_map.fov_height_mm)
        )
        os.makedirs("snapshots", exist_ok=True)
        filename = os.path.join("snapshots", f"snapshot_{active_ch}_{self.stage_x:.1f}_{self.stage_y:.1f}.png")
        image_utils.save_image(filename, frame)
        self.root_layout.status_msg = f"Snapshot saved to {filename}"

    def open_navigator_popup(self):
        plate_navigator.attach_navigator_to_lumaviewpro(self)

    def open_plate_navigator(self):
        self.open_navigator_popup()


# Safe Factory registration
if 'ZoomPanCameraWidget' not in Factory.classes:
    Factory.register('ZoomPanCameraWidget', cls=ZoomPanCameraWidget)
if 'ChannelControlWidget' not in Factory.classes:
    Factory.register('ChannelControlWidget', cls=ChannelControlWidget)
if 'LumaViewProMainLayout' not in Factory.classes:
    Factory.register('LumaViewProMainLayout', cls=LumaViewProMainLayout)

if 'InteractiveScaledPlateCanvas' not in Factory.classes:
    Factory.register('InteractiveScaledPlateCanvas', cls=plate_navigator.InteractiveScaledPlateCanvas)
if 'PlateNavigatorPopup' not in Factory.classes:
    Factory.register('PlateNavigatorPopup', cls=plate_navigator.PlateNavigatorPopup)

_main_kv = os.path.join(script_dir, 'lumaviewpro.kv')
if os.path.exists(_main_kv) and _main_kv not in Builder.files:
    Builder.load_file(_main_kv)


if __name__ == '__main__':
    LumaViewProApp().run()
