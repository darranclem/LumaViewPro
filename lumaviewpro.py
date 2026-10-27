"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation Software.

Single-Window UI Architecture with:
- Top Header: Title, Active Well Display, Stage Coordinates, Calibration Status
- Top-Left: Embedded Live Camera Viewport (ZoomPanCameraWidget) with reticle crosshairs & double-click pixel centering
- Top-Right: Embedded Interactive Microplate Map Canvas (InteractiveScaledPlateCanvas) with cyan active FOV reticle & double-click navigation
- Bottom-Left: Hardware Motion Deck with Grouped XY Cross (2-button direction pairs ++Y/+Y, --Y/-Y, --X/-X, ++X/+X) & Z Focus Column
- Bottom-Right: Optical Channels Panel (Single-Channel Mutual Exclusion) & 3-Tab Calibration/Protocol Studio
- Full Keyboard Navigation: Arrow keys for fine (0.5mm) & Shift+Arrow for coarse (5.0mm) XY jogging; PageUp/PageDown for Z focus
- Hard Software Motion Limits & FOV Physical Guardrails
"""

from __future__ import annotations

import os
import sys
import json
import math
import re
from typing import Dict, Optional, List, Tuple

# Force local repository directory to be #1 in Python module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import cv2
import numpy as np

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


class ChannelConfig:
    """Stores configuration and optical specifications for an imaging channel."""
    
    def __init__(
        self,
        name: str,
        excitation_info: str = "White LED",
        emission_info: str = "Pass-through",
        color_hex: str = "#FFFFFF",
        is_transmitted: bool = True,
        btn_color: List[float] = None
    ):
        self.name: str = name
        self.excitation_info: str = excitation_info
        self.emission_info: str = emission_info
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
            "Brightfield": ChannelConfig("Brightfield", excitation_info="White LED", emission_info="Pass-through", color_hex="#FFFFFF", is_transmitted=True, btn_color=[0.9, 0.9, 0.9, 1.0]),
            "Phase": ChannelConfig("Phase", excitation_info="White Ring", emission_info="Phase Shift", color_hex="#C0C0C0", is_transmitted=True, btn_color=[0.65, 0.7, 0.75, 1.0]),
            "Darkfield": ChannelConfig("Darkfield", excitation_info="Oblique LED", emission_info="Darkfield", color_hex="#808080", is_transmitted=True, btn_color=[0.4, 0.4, 0.45, 1.0]),
            "Blue": ChannelConfig("Blue", excitation_info="405nm (370-410)", emission_info="429-462nm (DAPI)", color_hex="#0080FF", is_transmitted=False, btn_color=[0.0, 0.45, 1.0, 1.0]),
            "Green": ChannelConfig("Green", excitation_info="488nm (473-491)", emission_info="502-561nm (FITC)", color_hex="#00FF40", is_transmitted=False, btn_color=[0.0, 0.85, 0.25, 1.0]),
            "Red": ChannelConfig("Red", excitation_info="594nm (580-598)", emission_info="612-680nm (TxRed)", color_hex="#FF3030", is_transmitted=False, btn_color=[0.95, 0.15, 0.15, 1.0])
        }

    def resolve_channel_name(self, name: str) -> str:
        return self.ALIASES.get(name, name)

    def set_channel_active(self, channel_name: str, active: bool):
        real_name = self.resolve_channel_name(channel_name)
        if real_name in self.channels:
            if active:
                for ch_key, ch_cfg in self.channels.items():
                    ch_cfg.active = (ch_key == real_name)
            else:
                self.channels[real_name].active = False

    def global_sync_active_channels(self, active_channel_states: Optional[Dict[str, bool]] = None):
        if active_channel_states:
            for name, active_val in active_channel_states.items():
                real_name = self.resolve_channel_name(name)
                if real_name in self.channels:
                    self.channels[real_name].active = active_val

    def get_active_channel_names(self) -> List[str]:
        return [ch_name for ch_name, ch_cfg in self.channels.items() if ch_cfg.active]

    def get_active_channel_name(self) -> str:
        names = self.get_active_channel_names()
        return names[0] if names else "Brightfield"

    def process_channel_image(self, channel_name: str, raw_frame: object) -> object:
        real_name = self.resolve_channel_name(channel_name)
        ch = self.channels.get(real_name)
        if (ch and ch.shading_correction_enabled and ch.ref_image_array is not None):
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


class ZoomPanCameraWidget(plate_navigator.ZoomPanCameraWidget):
    """Subclass inheriting reticle crosshairs & double-click pixel centering."""
    pass


class InteractiveScaledPlateCanvas(plate_navigator.InteractiveScaledPlateCanvas):
    """Subclass inheriting active stage cyan FOV reticle & double-click navigation."""
    pass


class LumaViewProMainLayout(BoxLayout):
    """Root main layout for LumaViewPro single window interface."""
    status_msg = StringProperty("LumaViewPro Microscope Automation Engine Ready")
    stage_coord_str = StringProperty("Stage: X=-49.50, Y=31.50, Z=0.000 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")
    current_well_display = StringProperty("A1")
    is_light_on = BooleanProperty(False)


class LumaViewProApp(App):
    """Main Application Controller for LumaViewPro."""

    # Class-level Kivy property declarations
    is_light_on = BooleanProperty(False)  # Launch with Light OFF by default
    is_live_imaging = BooleanProperty(True)
    active_channel = StringProperty("Brightfield")

    camera_exposure_ms = NumericProperty(20.0)
    exposure_ms = NumericProperty(20.0)

    camera_gain = NumericProperty(1.0)
    gain = NumericProperty(1.0)

    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)

    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)

    def load_kv(self, filename=None):
        """Overrides load_kv to prevent duplicate loading of lumaviewpro.kv."""
        pass

    def build(self):
        self.title = "LumaViewPro Microscope Automation"
        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

        # Default Stage Coordinates (Well A1 center)
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        self._updating_channel_ui = False
        self._active_navigator_popup = None

        self.root_layout = LumaViewProMainLayout()
        self.build_channel_controls()

        # Connect canvas references
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.set_references(
                self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
            )

        # Initialize hardware or simulator driver
        self.init_hardware()

        # Bind Keyboard Event Listener
        Window.bind(on_key_down=self._on_keyboard_down)

        # Start live camera feed clock loop @ 20 FPS (0.05s)
        Clock.schedule_interval(self.update_live_feed, 0.05)

        return self.root_layout

    def init_hardware(self):
        """Attempts connection to physical Basler USB camera or mock driver."""
        try:
            import pyloncamera
            self.camera = pyloncamera.PylonCamera()
            if not self.camera.connect():
                print("[LumaViewPro] USB camera hardware not detected; running in simulator mode.")
        except Exception:
            self.camera = None

    def clamp_stage_position(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Hard software guardrail clamping so FOV rectangle never extends past physical plate edges."""
        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        half_fov_x = fov_w / 2.0
        half_fov_y = fov_h / 2.0

        plate_w = self.well_plate_format.PLATE_LENGTH_MM  # 127.76 mm
        plate_h = self.well_plate_format.PLATE_WIDTH_MM   # 85.48 mm

        min_x = -plate_w / 2.0 + half_fov_x
        max_x = plate_w / 2.0 - half_fov_x
        min_y = -plate_h / 2.0 + half_fov_y
        max_y = plate_h / 2.0 - half_fov_y

        clamped_x = max(min_x, min(max_x, x_mm))
        clamped_y = max(min_y, min(max_y, y_mm))
        return clamped_x, clamped_y

    def _on_keyboard_down(self, window, key, scancode, codepoint, modifier):
        """
        Keyboard Motion Control Listener:
        - Arrow Keys: Fine (0.5mm) & Shift+Arrow Coarse (5.0mm) XY Jogging
        - PageUp / PageDown: Fine (1um) & Shift+PageUp/PageDown Coarse (10um) Z Focus
        """
        is_shift = 'shift' in modifier
        xy_step = 5.0 if is_shift else 0.5
        z_step_um = 10.0 if is_shift else 1.0

        if key in (273, 82) or codepoint == 'up':
            self.jog_stage_delta(0.0, xy_step, 0.0)
            self.root_layout.status_msg = f"Keyboard Motion: Jogged +Y {xy_step:.1f} mm"
            return True
        elif key in (274, 81) or codepoint == 'down':
            self.jog_stage_delta(0.0, -xy_step, 0.0)
            self.root_layout.status_msg = f"Keyboard Motion: Jogged -Y {xy_step:.1f} mm"
            return True
        elif key in (276, 80) or codepoint == 'left':
            self.jog_stage_delta(-xy_step, 0.0, 0.0)
            self.root_layout.status_msg = f"Keyboard Motion: Jogged -X {xy_step:.1f} mm"
            return True
        elif key in (275, 79) or codepoint == 'right':
            self.jog_stage_delta(xy_step, 0.0, 0.0)
            self.root_layout.status_msg = f"Keyboard Motion: Jogged +X {xy_step:.1f} mm"
            return True
        elif key == 280 or codepoint == 'pageup':
            self.jog_z_stage(z_step_um)
            self.root_layout.status_msg = f"Keyboard Z Focus: Jogged +Z {z_step_um:.1f} um"
            return True
        elif key == 281 or codepoint == 'pagedown':
            self.jog_z_stage(-z_step_um)
            self.root_layout.status_msg = f"Keyboard Z Focus: Jogged -Z {z_step_um:.1f} um"
            return True

        return False

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
                    widget.channel_spec_info = f"{ch_cfg.excitation_info}"
                else:
                    widget.channel_spec_info = f"Ex: {ch_cfg.excitation_info} | Em: {ch_cfg.emission_info}"
                widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref Image"
                container.add_widget(widget)
        finally:
            self._updating_channel_ui = False

    def set_active_channel(self, channel_name: str, active_state: bool):
        if getattr(self, '_updating_channel_ui', False):
            return

        self._updating_channel_ui = True
        try:
            real_ch = self.channel_manager.resolve_channel_name(channel_name)
            if active_state:
                self.channel_manager.set_channel_active(real_ch, True)
                self.active_channel = real_ch
                self.is_light_on = True
                self.root_layout.status_msg = f"Optical Channel {real_ch}: Active (Light ON)"
            else:
                if real_ch == self.active_channel:
                    self.is_light_on = False
                    self.channel_manager.channels[real_ch].active = False
                    self.root_layout.status_msg = f"Optical Channel {real_ch}: OFF (Sensor Dark)"

            self.root_layout.is_light_on = self.is_light_on
            self.sync_channel_ui_toggles()
        finally:
            self._updating_channel_ui = False
            self.update_live_feed(0)

    def select_optical_channel(self, channel_name: str):
        self.set_active_channel(channel_name, True)

    def select_channel(self, channel_name: str):
        self.set_active_channel(channel_name, True)

    def on_channel_toggle(self, channel_name: str, is_down: bool):
        self.set_active_channel(channel_name, is_down)

    def sync_channel_ui_toggles(self):
        if not hasattr(self.root_layout, 'ids') or 'main_channel_container' not in self.root_layout.ids:
            return

        container = self.root_layout.ids.main_channel_container
        for child in container.children:
            if isinstance(child, ChannelControlWidget):
                real_name = self.channel_manager.resolve_channel_name(child.channel_name)
                child.is_active = (real_name == self.active_channel) and self.is_light_on

    def toggle_light_state(self, is_on: Optional[bool] = None):
        if is_on is not None:
            self.is_light_on = is_on
        else:
            self.is_light_on = not self.is_light_on

        self.root_layout.is_light_on = self.is_light_on
        state_str = "ON" if self.is_light_on else "OFF (Dark Sensor)"
        self.root_layout.status_msg = f"Illumination Light: {state_str}"
        self.sync_channel_ui_toggles()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    def toggle_illumination_light(self):
        self.toggle_light_state()

    def toggle_light_master(self, is_down: bool):
        self.toggle_light_state(is_down)

    def toggle_live_feed(self):
        self.is_live_imaging = not self.is_live_imaging

    def set_live_fps(self, fps_str: str):
        try:
            digits = re.findall(r'\d+', fps_str)
            fps_val = int(digits[0]) if digits else 20
            self.live_fps = fps_val
        except Exception:
            pass

    def update_live_feed(self, dt):
        if not hasattr(self.root_layout, 'ids') or not self.is_live_imaging:
            return

        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        if hasattr(self, 'camera') and self.camera and hasattr(self.camera, 'get_frame'):
            if hasattr(self.camera, 'stage_x'):
                self.camera.stage_x = self.stage_x
                self.camera.stage_y = self.stage_y
                self.camera.stage_z = self.stage_z
                self.camera.active_channel = self.active_channel
                self.camera.exposure_ms = self.camera_exposure_ms
                self.camera.gain = self.camera_gain

            raw_frame = self.camera.get_frame()
            if raw_frame is None or not self.is_light_on:
                raw_frame = image_utils.generate_well_plate_cell_frame(
                    stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                    best_z=0.0, channel=self.active_channel, exposure_ms=self.camera_exposure_ms,
                    gain=self.camera_gain, led_power=self.led_power, is_light_on=self.is_light_on,
                    fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
                )
        else:
            raw_frame = image_utils.generate_well_plate_cell_frame(
                stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                best_z=0.0, channel=self.active_channel, exposure_ms=self.camera_exposure_ms,
                gain=self.camera_gain, led_power=self.led_power, is_light_on=self.is_light_on,
                fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
            )

        proc_frame = self.channel_manager.process_channel_image(self.active_channel, raw_frame)
        self.render_frame_to_viewports(proc_frame)

    def render_frame_to_viewports(self, frame: np.ndarray):
        if hasattr(self.root_layout, 'ids') and 'main_camera_viewport' in self.root_layout.ids:
            viewport = self.root_layout.ids.main_camera_viewport
            if hasattr(viewport, 'set_frame'):
                viewport.set_frame(frame)

        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.redraw_canvas()

    def set_well_plate_format(self, num_wells_str: str):
        try:
            digits = re.findall(r'\d+', num_wells_str)
            num_wells = int(digits[0]) if digits else 96
            self.well_plate_format = labware.WellPlateFormat(num_wells=num_wells)
            self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
            self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

            # Center on A1 well
            self.stage_x = -49.5
            self.stage_y = 31.5

            if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                canvas_widget = self.root_layout.ids.main_plate_canvas
                canvas_widget.set_references(
                    self.well_plate_format,
                    self.plate_calibration,
                    self.focus_map,
                    self.fov_manager
                )

            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
            self.root_layout.status_msg = f"Well Plate Format set to {num_wells}-Well Plate"
            self.update_live_feed(0)
        except Exception as e:
            self.root_layout.status_msg = f"Failed to set plate format: {e}"

    def change_well_plate_format(self, num_wells_str: str):
        self.set_well_plate_format(num_wells_str)

    def select_well_plate_format(self, num_wells_str: str):
        self.set_well_plate_format(num_wells_str)

    def set_jog_step(self, step_str: str):
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
        except (IndexError, ValueError):
            self.jog_step_mm = 1.0

    def set_jog_step_size(self, step_str: str):
        self.set_jog_step(step_str)

    def jog_stage(self, dx_dir: float, dy_dir: float, dz_dir: float = 0.0):
        step = self.jog_step_mm
        self.jog_stage_delta(dx_dir * step, dy_dir * step, dz_dir)

    def jog_stage_delta(self, dx_mm: float, dy_mm: float, dz_mm: float = 0.0):
        new_x = self.stage_x + dx_mm
        new_y = self.stage_y + dy_mm
        clamped_x, clamped_y = self.clamp_stage_position(new_x, new_y)

        self.stage_x = clamped_x
        self.stage_y = clamped_y
        self.stage_z = max(0.0, min(10.0, self.stage_z + dz_mm))

        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.update_live_feed(0)

    def jog_z_stage(self, delta_um: float):
        self.jog_stage_delta(0.0, 0.0, delta_um / 1000.0)

    def jog_z_focus(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_z(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_z_microns(self, delta_um: float):
        self.jog_z_stage(delta_um)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        clamped_x, clamped_y = self.clamp_stage_position(x_mm, y_mm)
        self.stage_x = clamped_x
        self.stage_y = clamped_y
        if z_mm is not None:
            self.stage_z = max(0.0, min(10.0, z_mm))

        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.update_live_feed(0)

    def move_stage_abs(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        self.jog_stage_to(x_mm, y_mm, z_mm)

    def zero_stage(self):
        """Homes stage to Well A1 center."""
        a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
        a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
        self.jog_stage_to(a1_x, a1_y)
        self.root_layout.status_msg = "Stage reset to Well A1 Center"

    def set_exposure_ms(self, text_val: str):
        try:
            val = float(text_val)
            self.exposure_ms = max(0.1, min(5000.0, val))
            self.camera_exposure_ms = self.exposure_ms
            self.root_layout.exposure_str = f"{self.exposure_ms:.1f}"
            self.root_layout.status_msg = f"Exposure Time set to {self.exposure_ms:.1f} ms"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_exposure_time(self, text_val: str):
        self.set_exposure_ms(text_val)

    def set_camera_exposure(self, text_val: str):
        self.set_exposure_ms(text_val)

    def set_gain_val(self, text_val: str):
        try:
            val = float(text_val)
            self.gain = max(1.0, min(20.0, val))
            self.camera_gain = self.gain
            self.root_layout.gain_str = f"{self.gain:.1f}"
            self.root_layout.status_msg = f"Camera Gain set to {self.gain:.1f}x"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_camera_gain(self, text_val: str):
        self.set_gain_val(text_val)

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            self.root_layout.led_power_str = f"{self.led_power:.0f}"
            self.root_layout.status_msg = f"LED Power set to {self.led_power:.0f}%"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        self.fov_manager._rebuild_all_groups()
        self.root_layout.status_msg = f"Objective Lens set to: {preset_name} (Mag: {self.focus_map.lens_mag}x, FOV: {self.focus_map.fov_width_mm:.2f}x{self.focus_map.fov_height_mm:.2f} mm)"
        self.update_live_feed(0)

    def record_calibration_step(self):
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)
        self.root_layout.status_msg = msg

    def finish_plate_calibration(self):
        success = self.plate_calibration.compute_calibration()
        if success:
            self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
            self.root_layout.status_msg = "Plate Registration Matrix successfully computed!"
        else:
            self.root_layout.status_msg = "Calibration failed. Verify edge target points."

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
        except Exception:
            pass

    def set_time_lapse_config(self, enabled: bool, interval_mins_str: str, cycles_str: str):
        try:
            inv_m = float(interval_mins_str) if interval_mins_str else 10.0
            cyc = int(cycles_str) if cycles_str else 24
            self.focus_map.set_time_lapse_config(enabled, inv_m, cyc)
        except ValueError:
            pass

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

    def set_chromatic_z_offset(self, channel_name: str):
        self.focus_map.set_chromatic_z_offset(channel_name, self.stage_z)
        delta_z = self.focus_map.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        self.root_layout.status_msg = f"Saved Chromatic Z-Offset for {channel_name}: {delta_z*1000.0:.2f} um"

    def apply_nyquist_calculation(self):
        dof_um, nyquist_step_um = self.focus_map.calculate_optical_nyquist()
        self.focus_map.apply_nyquist_to_af_config()
        self.root_layout.status_msg = f"Applied Nyquist Z-Step: {nyquist_step_um:.3f} um (DoF: {dof_um:.3f} um)"

    def apply_grid_settings(self, rows_str: str, cols_str: str, single_af: bool):
        try:
            r = int(rows_str)
            c = int(cols_str)
            self.fov_manager.set_grid_dimensions(r, c, single_af=single_af)
            self.root_layout.status_msg = f"Tile Grid set to {r}x{c}"
        except ValueError:
            pass

    def generate_and_apply_protocol(self) -> dict:
        protocol = labware.ExperimentProtocol.generate_protocol(
            self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
        )
        print("[Protocol Generated]", json.dumps(protocol, indent=2))
        self.root_layout.status_msg = f"Hardware Protocol Saved! Selected Wells: {protocol['total_selected_wells']}"
        return protocol

    def capture_snapshot(self):
        active_ch = self.channel_manager.get_active_channel_name()
        frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=2.5, channel=active_ch, is_light_on=self.is_light_on,
            led_power=self.led_power, num_wells=self.well_plate_format.num_wells
        )
        os.makedirs("snapshots", exist_ok=True)
        filename = f"snapshots/snapshot_{active_ch}_{self.stage_x:.1f}_{self.stage_y:.1f}.png"
        image_utils.save_image(filename, frame)
        self.root_layout.status_msg = f"Saved Snapshot: {filename}"

    def open_file_chooser_for_channel(self, channel_name: str):
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        filechooser = FileChooserListView(path=os.path.expanduser('~'))
        content.add_widget(filechooser)

        btn_layout = BoxLayout(size_hint_y=0.15, spacing=10)
        select_btn = Button(text="Select Reference Image")
        cancel_btn = Button(text="Cancel")
        btn_layout.add_widget(select_btn)
        btn_layout.add_widget(cancel_btn)
        content.add_widget(btn_layout)

        popup = Popup(title=f"Select Reference Image for {channel_name}", content=content, size_hint=(0.85, 0.85))

        def on_select(instance):
            if filechooser.selection:
                selected = filechooser.selection[0]
                real_ch = self.channel_manager.resolve_channel_name(channel_name)
                if self.channel_manager.channels[real_ch].load_ref_image(selected):
                    self.root_layout.status_msg = f"Loaded shading ref for {real_ch}"
                    self.build_channel_controls()
            popup.dismiss()

        select_btn.bind(on_release=on_select)
        cancel_btn.bind(on_release=popup.dismiss)
        popup.open()


# Register custom widget classes with Kivy's Factory BEFORE loading KV layout rules
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
