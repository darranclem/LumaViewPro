from __future__ import annotations
# VERSION: 4.5.0-FULL-MICROSCOPE-HANDSHAKE
"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation.

Unified Single-Window UI Architecture (Zero Popups):
- Window Maximized to Full Desktop Size on launch.
- Top Header: Application Title, Active Well Display, Stage Coordinates, Calibration Status.
- Top-Left (50% Width): Live Camera Viewport (ZoomPanCameraWidget) with center reticle crosshairs & double-click pixel stage centering.
- Bottom-Left (50% Width): Hardware Motion Deck with Ergonomic 3x3 XY Cross (Y+ ↑, X− ←, Home A1, → X+, ↓ Y−), Step Size Spinner, X/Y Position Readouts, and Z Focus Vertical Bar.
- Top-Right (50% Width): Microplate Canvas Map (InteractiveScaledPlateCanvas) with Format Selector (6 to 384 wells), Objective Selector (1.25x to 100x Oil), active FOV cyan reticle, and 1:1 double-click stage navigation.
- Bottom-Right (50% Width): Optical Channels Panel (Single-Channel Mutual Exclusion) + 6-Point Target Calibration Wizard.
"""

import os
import sys

# 1. Force local repository directory to be #1 in Python's module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

# 2. MANDATORY TOP-LEVEL EXECUTION OF STOCK ETALUMA JAVA/MAVEN/DRIVER BOOTSTRAPPER
import etaluma_init
etaluma_init.run_etaluma_stock_initialization()

import json
import math
import re
import cv2
import numpy as np
import threading
from typing import Dict, Optional, List, Tuple

import image_utils
import labware
import plate_navigator

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


def install_crash_handler():
    """Installs global exception interceptors to record tracebacks in lumaviewpro_crash.log."""
    def _excepthook(exctype, value, tb):
        import traceback
        err_msg = "".join(traceback.format_exception(exctype, value, tb))
        print(f"\n[LUMAVIEWPRO CRASH INTERCEPTED]\n{err_msg}")
        try:
            log_path = os.path.join(script_dir, "lumaviewpro_crash.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== CRASH LOG ({Clock.get_time():.3f}s) ===\n{err_msg}\n")
        except Exception:
            pass

    sys.excepthook = _excepthook
    if hasattr(threading, 'excepthook'):
        threading.excepthook = lambda args: _excepthook(args.exc_type, args.exc_value, args.exc_traceback)


install_crash_handler()


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
        self.ref_image_array: Optional[np.ndarray] = None
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

    def process_channel_image(self, channel_name: str, raw_frame: np.ndarray) -> np.ndarray:
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


class ZoomPanCameraWidget(Widget):
    """Interactive Live Camera Viewport Widget with Mouse Zoom, Pan, and Reticle Crosshairs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.texture = None
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.bind(size=self.redraw_viewport, pos=self.redraw_viewport)

    def set_frame(self, frame_np: np.ndarray):
        if frame_np is None:
            return
        try:
            sanitized = image_utils.sanitize_camera_frame(frame_np)
            h, w = sanitized.shape[:2]

            # Flip vertically for OpenGL coordinate system
            frame_flipped = np.ascontiguousarray(sanitized[::-1, :, :], dtype=np.uint8)

            if self.texture is None or self.texture.size != (w, h):
                self.texture = Texture.create(size=(w, h), colorfmt='rgb')

            self.texture.blit_buffer(frame_flipped.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.redraw_viewport()
        except Exception as e:
            print(f"[ZoomPanCameraWidget Error] set_frame: {e}")

    def redraw_viewport(self, *args):
        self.canvas.clear()
        if self.texture is None:
            with self.canvas:
                Color(0.1, 0.1, 0.14, 1.0)
                Rectangle(pos=self.pos, size=self.size)
            return

        with self.canvas:
            Color(1, 1, 1, 1)
            # Render video texture
            Rectangle(texture=self.texture, pos=self.pos, size=self.size)

            # Draw Red Reticle Center Crosshairs
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.85)
            Line(points=[cx - 20, cy, cx + 20, cy], width=1.5)
            Line(points=[cx, cy - 20, cx, cy + 20], width=1.5)
            Line(circle=(cx, cy, 10), width=1.2)


class LumaViewProMainLayout(BoxLayout):
    """Main Application Root Window Layout."""
    status_msg = StringProperty("LumaViewPro Engine Ready")
    stage_coord_str = StringProperty("Stage: X=-49.50, Y=31.50, Z=0.000 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")
    current_well_display = StringProperty("A1")
    calibration_wizard_text = StringProperty("Click 'Start Calib' to begin 6-point plate alignment.")


class LumaViewProApp(App):
    """Main Application Controller."""

    # Class-level Kivy Property Declarations
    is_light_on = BooleanProperty(False)  # Sensor starts dark on launch
    active_channel = StringProperty("Brightfield")
    camera_gain = NumericProperty(1.0)
    gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    exposure_ms = NumericProperty(20.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)

    stage_x = NumericProperty(-49.5)
    stage_y = NumericProperty(31.5)
    stage_z = NumericProperty(0.0)

    camera_connected = BooleanProperty(False)
    stage_connected = BooleanProperty(False)
    focus_connected = BooleanProperty(False)
    led_connected = BooleanProperty(False)
    is_simulator = BooleanProperty(False)

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
            kv_path = os.path.join(script_dir, 'lumaviewpro.kv')
            if os.path.exists(kv_path) and kv_path not in Builder.files:
                Builder.load_file(kv_path)
        return None

    def build(self):
        self.title = "LumaViewPro Microscope Automation"

        # MAXIMIZE WINDOW TO FULL DESKTOP SIZE
        try:
            Window.maximize()
        except Exception:
            pass

        self.camera = None
        self.motor_stage = None
        self.leds = None

        # Execute Hardware Handshake with Basler Camera, Motor Board, and LED Board
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

        # Report startup hardware connection status on footer bar
        if self.camera_connected:
            self.root_layout.status_msg = "✓ Basler USB Camera CONNECTED | LumaViewPro System Ready"
        else:
            self.root_layout.status_msg = "X COULD NOT CONNECT TO MICROSCOPE HARDWARE - Check USB & Power Connections"

        return self.root_layout

    def log_hw(self, msg: str):
        """Logs diagnostic messages to stdout and hardware_debug.log."""
        try:
            formatted = f"[HW_LOG {Clock.get_time():.3f}s] {msg}"
            print(formatted)
            log_path = os.path.join(script_dir, "hardware_debug.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")
        except Exception:
            pass

    def init_hardware(self):
        """Connects to real Etaluma hardware drivers (USB Camera, Motor Board, LED Board)."""
        self.camera_connected = False
        self.stage_connected = False
        self.focus_connected = False
        self.led_connected = False

        self.log_hw("=== Starting LumaViewPro Hardware Handshake ===")

        # 1. BASLER USB CAMERA (pyloncamera)
        try:
            import pyloncamera
            self.log_hw("Imported pyloncamera. Instantiating PylonCamera()...")
            cam = pyloncamera.PylonCamera()
            if hasattr(cam, 'connect'):
                res = cam.connect()
                self.log_hw(f"cam.connect() returned: {repr(res)}")
                if res is not False and getattr(cam, 'is_connected', True) is not False:
                    self.camera = cam
                    self.camera_connected = True
                    self.log_hw("✓ Basler USB Camera CONNECTED successfully.")
                else:
                    self.camera = None
                    self.log_hw("X pyloncamera.connect() returned False.")
        except Exception as e:
            self.camera = None
            self.camera_connected = False
            self.log_hw(f"X Basler USB Camera Driver error/unattached: {e}")

        # 2. MOTOR BOARD (motorboard)
        try:
            import motorboard
            self.log_hw("Imported motorboard. Instantiating MotorBoard()...")
            stg = motorboard.MotorBoard()
            if hasattr(stg, 'connect'):
                res = stg.connect()
                self.log_hw(f"stg.connect() returned: {repr(res)}")
                if res is not False and getattr(stg, 'is_connected', True) is not False:
                    self.motor_stage = stg
                    self.stage_connected = True
                    self.focus_connected = True
                    self.log_hw("✓ Stage & Focus Motor Board CONNECTED successfully.")
                else:
                    self.motor_stage = None
                    self.stage_connected = False
                    self.focus_connected = False
                    self.log_hw("X motorboard.connect() returned False.")
        except Exception as e:
            self.motor_stage = None
            self.stage_connected = False
            self.focus_connected = False
            self.log_hw(f"X Motor Board Driver error/unattached: {e}")

        # 3. LED BOARD (ledboard)
        try:
            import ledboard
            self.log_hw("Imported ledboard. Instantiating LEDBoard()...")
            leds = ledboard.LEDBoard()
            if hasattr(leds, 'connect'):
                res = leds.connect()
                self.log_hw(f"leds.connect() returned: {repr(res)}")
                if res is not False and getattr(leds, 'is_connected', True) is not False:
                    self.leds = leds
                    self.led_connected = True
                    self.log_hw("✓ LED Illumination Board CONNECTED successfully.")
                else:
                    self.leds = None
                    self.led_connected = False
                    self.log_hw("X ledboard.connect() returned False.")
        except Exception as e:
            self.leds = None
            self.led_connected = False
            self.log_hw(f"X LED Board Driver error/unattached: {e}")

        hw_summary = f"Hardware Status -> Camera: {self.camera_connected}, Stage: {self.stage_connected}, LED: {self.led_connected}"
        self.log_hw(hw_summary)

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

    def clamp_stage_position(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> Tuple[float, float, float]:
        fov_w = getattr(self.focus_map, 'fov_width_mm', 0.60)
        fov_h = getattr(self.focus_map, 'fov_height_mm', 0.45)

        plate_w_mm = getattr(self.well_plate_format, 'PLATE_LENGTH_MM', 127.76)
        plate_h_mm = getattr(self.well_plate_format, 'PLATE_WIDTH_MM', 85.48)

        max_x = (plate_w_mm / 2.0) - (fov_w / 2.0)
        min_x = -max_x
        max_y = (plate_h_mm / 2.0) - (fov_h / 2.0)
        min_y = -max_y

        target_x = x if x is not None else self.stage_x
        target_y = y if y is not None else self.stage_y
        target_z = z if z is not None else self.stage_z

        clamped_x = max(min_x, min(max_x, float(target_x)))
        clamped_y = max(min_y, min(max_y, float(target_y)))
        clamped_z = max(self.Z_MIN_MM, min(self.Z_MAX_MM, float(target_z)))

        self.stage_x = clamped_x
        self.stage_y = clamped_y
        self.stage_z = clamped_z

        if hasattr(self, 'root_layout') and self.root_layout:
            if self.stage_connected or self.is_simulator:
                self.root_layout.stage_coord_str = f"Stage: X={clamped_x:.2f}, Y={clamped_y:.2f}, Z={clamped_z:.3f} mm"
            else:
                self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

        self.update_active_well_display()
        return clamped_x, clamped_y, clamped_z

    def update_active_well_display(self):
        try:
            r = max(0, min(self.well_plate_format.rows - 1, int(round((31.5 - self.stage_y) / self.well_plate_format.pitch_y))))
            c = max(0, min(self.well_plate_format.cols - 1, int(round((self.stage_x + 49.5) / self.well_plate_format.pitch_x))))
            w_name = self.well_plate_format.get_well_name(r, c)
            if hasattr(self, 'root_layout') and self.root_layout:
                self.root_layout.current_well_display = w_name
        except Exception:
            pass

    def redraw_plate_canvas(self):
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.redraw_canvas()

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

    def set_active_channel(self, channel_name: str, active_bool: bool = True):
        self.select_optical_channel(channel_name)

    def select_channel(self, channel_name: str):
        self.select_optical_channel(channel_name)

    def on_channel_toggle(self, channel_name: str, is_down: bool):
        self.set_active_channel(channel_name, is_down)

    def toggle_light_state(self, is_on: Optional[bool] = None):
        if is_on is None:
            self.is_light_on = not self.is_light_on
        else:
            self.is_light_on = is_on

        if self.is_light_on and self.leds and hasattr(self.leds, 'set_led'):
            try:
                self.leds.set_led(self.active_channel, int(self.led_power))
            except Exception:
                pass
        elif not self.is_light_on and self.leds and hasattr(self.leds, 'off'):
            try:
                self.leds.off()
            except Exception:
                pass

        state_str = "ON" if self.is_light_on else "OFF"
        self.root_layout.status_msg = f"Illumination Light set to {state_str}"
        self.build_channel_controls()

    def toggle_light(self):
        self.toggle_light_state()

    def toggle_light_master(self, is_down: bool):
        self.toggle_light_state(is_down)

    def set_camera_exposure(self, exp_str: str):
        self.set_exposure_ms(exp_str)

    def set_camera_gain(self, gain_str: str):
        self.set_gain_val(gain_str)

    def set_jog_step(self, step_str: str):
        self.set_jog_step_size(step_str)

    def start_calibration_wizard(self):
        self.plate_calibration.is_calibrating = True
        self.plate_calibration.is_calibrated = False
        self.plate_calibration.current_step_index = 0
        self.plate_calibration.recorded_points.clear()

        self.is_light_on = True

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
            self.root_layout.status_msg = f"LED Power set to {self.led_power:.0f}%"
            if self.leds and hasattr(self.leds, 'set_led'):
                try:
                    self.leds.set_led(self.active_channel, int(self.led_power))
                except Exception:
                    pass
        except ValueError:
            pass

    def jog_stage(self, dx: float, dy: float, dz: float = 0.0):
        if not self.stage_connected and not self.is_simulator:
            self.root_layout.status_msg = "Microscope Stage & Focus DISCONNECTED - Hardware Motion Unavailable"
            return

        step = self.jog_step_mm
        raw_x = self.stage_x + dx * step
        raw_y = self.stage_y + dy * step
        raw_z = self.stage_z + dz

        cx, cy, cz = self.clamp_stage_position(raw_x, raw_y, raw_z)

        if self.motor_stage and hasattr(self.motor_stage, 'move_rel'):
            try:
                self.motor_stage.move_rel(dx * step, dy * step, dz)
            except Exception:
                pass

        self.redraw_plate_canvas()

    def jog_z_stage(self, delta_um: float):
        self.jog_stage(0.0, 0.0, delta_um / 1000.0)

    def jog_z_focus(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_z_microns(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_z(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        if not self.stage_connected and not self.is_simulator:
            self.root_layout.status_msg = "Microscope Stage & Focus DISCONNECTED - Hardware Motion Unavailable"
            return

        tz = z_mm if z_mm is not None else self.stage_z
        cx, cy, cz = self.clamp_stage_position(x_mm, y_mm, tz)

        if self.motor_stage and hasattr(self.motor_stage, 'move_abs'):
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception:
                pass

        self.redraw_plate_canvas()

    def zero_stage(self):
        a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
        a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
        self.jog_stage_to(a1_x, a1_y, 0.0)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        if hasattr(self, 'fov_manager'):
            self.fov_manager._rebuild_all_groups()

        self.clamp_stage_position()
        self.root_layout.status_msg = f"Objective Preset set to {preset_name} (FOV: {self.focus_map.fov_width_mm:.2f} x {self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()

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

            if self.stage_connected or self.is_simulator:
                a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
                a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
                self.clamp_stage_position(a1_x, a1_y, self.stage_z)
            else:
                self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

            self.root_layout.status_msg = f"Well Plate Map updated to {num}-Well Format"

    def set_jog_step_size(self, step_str: str):
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
            self.root_layout.status_msg = f"XY Jog Step set to {val:.1f} mm"
        except Exception:
            pass

    def update_live_feed(self, dt):
        if not hasattr(self, 'root_layout') or self.root_layout is None:
            return

        fov_w = getattr(self.focus_map, 'fov_width_mm', 0.60)
        fov_h = getattr(self.focus_map, 'fov_height_mm', 0.45)

        raw_frame = None

        if self.camera_connected and self.camera is not None:
            try:
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
            except Exception as e:
                self.log_hw(f"Camera get_frame exception: {e}")
                raw_frame = None

            if raw_frame is None:
                if not self.is_light_on:
                    # Dark sensor state when light is OFF
                    raw_frame = image_utils.generate_well_plate_cell_frame(
                        stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                        best_z=2.5, channel=self.active_channel, is_light_on=False,
                        led_power=self.led_power, exposure_ms=self.exposure_ms, gain=self.gain
                    )
                else:
                    # Could not capture real frame from connected camera
                    raw_frame = image_utils.generate_disconnected_camera_frame(
                        msg="COULD NOT CONNECT TO CAMERA SENSOR"
                    )
        elif self.is_simulator:
            # Simulator mode (luma_sim.py) renders cell colonies
            raw_frame = image_utils.generate_well_plate_cell_frame(
                stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                best_z=2.5, channel=self.active_channel, is_light_on=self.is_light_on,
                led_power=self.led_power, exposure_ms=self.exposure_ms, gain=self.gain,
                fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
            )
        else:
            # Application on unattached PC shows COULD NOT CONNECT banner
            raw_frame = image_utils.generate_disconnected_camera_frame(
                msg="COULD NOT CONNECT TO MICROSCOPE HARDWARE"
            )

        try:
            proc_frame = self.channel_manager.process_channel_image(self.active_channel, raw_frame)
            self.render_frame_to_viewports(proc_frame)
        except Exception as e:
            self.log_hw(f"Process channel image exception: {e}")

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
            self.root_layout.status_msg = f"Exposure set to {val:.1f} ms"
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
        except ValueError:
            pass


# Register custom widget classes with Factory BEFORE loading KV files
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


if __name__ == '__main__':
    LumaViewProApp().run()
