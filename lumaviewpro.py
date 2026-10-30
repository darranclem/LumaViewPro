"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation Software.

Single-Window UI Architecture with:
- Top Header: Title, Active Well Display, Stage Coordinates, Calibration Status
- Top-Left: Embedded Live Camera Viewport (ZoomPanCameraWidget) with reticle crosshairs & double-click pixel centering
- Top-Right: Embedded Interactive Microplate Map Canvas (InteractiveScaledPlateCanvas) with cyan active FOV reticle & double-click navigation
- Bottom-Left: Hardware Motion Deck with Grouped XY Cross (2-button direction pairs ++Y/+Y, --Y/-Y, --X/-X, ++X/+X) & Z Focus Column
- Bottom-Right: Optical Channels Panel (Single-Channel Mutual Exclusion) & 3-Tab Calibration/Protocol Studio
- Full Keyboard Navigation: Arrow keys for fine (0.5mm) & Shift+Arrow for coarse (5.0mm) XY jogging; PageUp/PageDown for Z focus
- Physical Hardware Driver Connection (pyloncamera, motorboard, ledboard) with clear DISCONNECTED feedback when run unattached
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
import plate_navigator

import etaluma_init
etaluma_init.run_etaluma_stock_initialization()

# Kivy framework imports
import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
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
    """Manages 6 optical imaging channels with strict single-channel mutual exclusion."""
    
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
            "Phase": ChannelConfig("Phase", excitation_info="White Ring", emission_info="Phase Shift", color_hex="#CCCCCC", is_transmitted=True, btn_color=[0.65, 0.7, 0.75, 1.0]),
            "Darkfield": ChannelConfig("Darkfield", excitation_info="Oblique LED", emission_info="Pass-through", color_hex="#888888", is_transmitted=True, btn_color=[0.4, 0.4, 0.45, 1.0]),
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
                for name, ch_cfg in self.channels.items():
                    ch_cfg.active = (name == real_name)
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


class ZoomPanCameraWidget(plate_navigator.ZoomPanCameraWidget):
    """Subclass inheriting reticle crosshairs & double-click pixel centering."""
    pass


class InteractiveScaledPlateCanvas(plate_navigator.InteractiveScaledPlateCanvas):
    """Subclass inheriting active stage cyan FOV reticle & double-click navigation."""
    pass


class LumaViewProMainLayout(BoxLayout):
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
    calibration_wizard_text = StringProperty("Click 'Start Calib' to begin 6-point plate alignment.")
    camera_gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)


class LumaViewProApp(App):
    """Main Application Controller."""

    # Explicit class properties bound to KV layout and callbacks
    is_live_imaging = BooleanProperty(True)
    is_light_on = BooleanProperty(False)  # Launch with light OFF
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

    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)

    def load_kv(self, filename=None):
        """Overrides load_kv to prevent duplicate loading of lumaviewpro.kv."""
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

        # Stage Coordinates (Default Well A1)
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        self.camera = None
        self.motor_stage = None
        self.leds = None

        self._updating_channel_ui = False
        self.root_layout = LumaViewProMainLayout()

        # Connect hardware camera, stage, and LED drivers
        self.init_hardware()

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

        # Start live camera feed update loop at 20 FPS (0.05s interval)
        Clock.schedule_interval(self.update_live_feed, 0.05)
        return self.root_layout

    def init_hardware(self):
        """Connects to real Etaluma hardware drivers if available on physical microscope PC."""
        self.camera_connected = False
        self.stage_connected = False
        self.led_connected = False

        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            if hasattr(cam, 'connect') and cam.connect():
                self.camera = cam
                self.camera_connected = True
                print("[LumaViewPro] Connected to physical Basler USB Camera.")
        except Exception as e:
            self.camera = None
            print(f"[LumaViewPro] USB Camera Driver Info: {e}")

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            if hasattr(stg, 'connect') and stg.connect():
                self.motor_stage = stg
                self.stage_connected = True
                print("[LumaViewPro] Connected to physical Stage Motor Board.")
        except Exception as e:
            self.motor_stage = None
            print(f"[LumaViewPro] Motor Board Driver Info: {e}")

        try:
            import ledboard
            leds = ledboard.LEDBoard()
            self.leds = leds
            self.led_connected = True
            print("[LumaViewPro] Connected to physical LED Illumination Board.")
        except Exception as e:
            self.leds = None
            print(f"[LumaViewPro] LED Board Driver Info: {e}")

        if not self.stage_connected and hasattr(self, 'root_layout') and self.root_layout:
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

    def clamp_stage_position(self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None) -> Tuple[float, float, float]:
        """Physical Guardrails: Clamps stage position so the active FOV never extends beyond the physical plate."""
        if x is None:
            x = self.stage_x
        if y is None:
            y = self.stage_y
        if z is None:
            z = self.stage_z

        fov_w = getattr(self.focus_map, 'fov_width_mm', 0.60)
        fov_h = getattr(self.focus_map, 'fov_height_mm', 0.45)

        plate_w_mm = getattr(self.well_plate_format, 'PLATE_LENGTH_MM', 127.76)
        plate_h_mm = getattr(self.well_plate_format, 'PLATE_WIDTH_MM', 85.48)

        max_x = (plate_w_mm / 2.0) - (fov_w / 2.0)
        min_x = -max_x
        max_y = (plate_h_mm / 2.0) - (fov_h / 2.0)
        min_y = -max_y

        clamped_x = max(min_x, min(max_x, float(x)))
        clamped_y = max(min_y, min(max_y, float(y)))
        clamped_z = max(0.0, min(10.0, float(z)))

        return clamped_x, clamped_y, clamped_z

    def update_active_well_display(self):
        try:
            r = max(0, min(self.well_plate_format.rows - 1, int(round((31.5 - self.stage_y) / self.well_plate_format.pitch_y))))
            c = max(0, min(self.well_plate_format.cols - 1, int(round((self.stage_x + 49.5) / self.well_plate_format.pitch_x))))
            w_name = self.well_plate_format.get_well_name(r, c)
            if hasattr(self.root_layout, 'current_well_display'):
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
                    widget.channel_spec_info = f"Ex: {ch_cfg.excitation_info} | Em: {ch_cfg.emission_info}" if not ch_cfg.is_transmitted else f"{ch_cfg.excitation_info}"
                    widget.is_active = (ch_name == self.active_channel) and self.is_light_on
                    widget.btn_bg_color = ch_cfg.btn_color
                    widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref"
                    container.add_widget(widget)
        finally:
            self._updating_channel_ui = False

    def set_active_channel(self, channel_name: str, active_bool: bool = True):
        if getattr(self, '_updating_channel_ui', False):
            return

        self._updating_channel_ui = True
        try:
            real_ch = self.channel_manager.resolve_channel_name(channel_name)
            if active_bool:
                self.channel_manager.set_channel_active(real_ch, True)
                self.active_channel = real_ch
                self.is_light_on = True
                self.root_layout.is_light_on = True
                self.root_layout.status_msg = f"Illumination ON | Channel: {real_ch}"
                if self.leds and hasattr(self.leds, 'set_led'):
                    try:
                        image_utils.safe_set_led(self.leds, real_ch, int(self.led_power))
                    except Exception as e:
                        print(f"[LEDBoard set_led error] {e}")
            else:
                if real_ch == self.active_channel:
                    self.is_light_on = False
                    self.root_layout.is_light_on = False
                    self.channel_manager.set_channel_active(real_ch, False)
                    self.root_layout.status_msg = f"Illumination OFF ({real_ch})"
                    if self.leds and hasattr(self.leds, 'off'):
                        try:
                            self.leds.off()
                        except Exception as e:
                            print(f"[LEDBoard off error] {e}")

            self.sync_channel_ui_toggles()
        finally:
            self._updating_channel_ui = False
            self.update_live_feed(0)

    def select_channel(self, channel_name: str):
        self.set_active_channel(channel_name, True)

    def select_optical_channel(self, channel_name: str):
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
        status = f"ON ({self.active_channel})" if self.is_light_on else "OFF (Dark Sensor)"
        self.root_layout.status_msg = f"Illumination Light: {status}"

        if self.is_light_on:
            self.channel_manager.set_channel_active(self.active_channel, True)
            if self.leds and hasattr(self.leds, 'set_led'):
                try:
                    image_utils.safe_set_led(self.leds, self.active_channel, int(self.led_power))
                except Exception as e:
                    print(f"[LEDBoard set_led error] {e}")
        else:
            for ch in self.channel_manager.channels.values():
                ch.active = False
            if self.leds and hasattr(self.leds, 'off'):
                try:
                    self.leds.off()
                except Exception as e:
                    print(f"[LEDBoard off error] {e}")

        self.sync_channel_ui_toggles()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    def toggle_illumination_light(self):
        self.toggle_light_state()

    def toggle_light_master(self, is_down: bool):
        self.toggle_light_state(is_down)

    def set_exposure_ms(self, text_val: str):
        try:
            val = float(text_val)
            self.exposure_ms = val
            self.camera_exposure_ms = val
            self.root_layout.exposure_str = f"{val:.1f}"
            if self.camera and hasattr(self.camera, 'set_exposure'):
                try:
                    self.camera.set_exposure(val)
                except Exception as e:
                    print(f"[Camera set_exposure error] {e}")
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_camera_exposure(self, val_text: str):
        self.set_exposure_ms(val_text)

    def set_gain_val(self, text_val: str):
        try:
            val = float(text_val)
            self.gain = val
            self.camera_gain = val
            self.root_layout.gain_str = f"{val:.1f}"
            if self.camera and hasattr(self.camera, 'set_gain'):
                try:
                    self.camera.set_gain(val)
                except Exception as e:
                    print(f"[Camera set_gain error] {e}")
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_camera_gain(self, val_text: str):
        self.set_gain_val(val_text)

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            self.root_layout.led_power_str = f"{self.led_power:.0f}"
            if self.is_light_on and self.leds and hasattr(self.leds, 'set_led'):
                try:
                    image_utils.safe_set_led(self.leds, self.active_channel, int(self.led_power))
                except Exception as e:
                    print(f"[LEDBoard set_led error] {e}")
            self.update_live_feed(0)
        except ValueError:
            pass

    def update_live_feed(self, dt: float):
        if not hasattr(self, 'root_layout') or self.root_layout is None:
            return

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
                self.camera.fov_size_mm = (self.focus_map.fov_width_mm, self.focus_map.fov_height_mm)
                self.camera.num_wells = self.well_plate_format.num_wells

            try:
                cam_frame = self.camera.get_frame()
                if cam_frame is not None:
                    raw_frame = np.array(cam_frame, copy=True)
                else:
                    raw_frame = None
            except Exception as e:
                print(f"[Camera get_frame error] {e}")
                raw_frame = None

            if raw_frame is None or not self.is_light_on:
                raw_frame = image_utils.generate_well_plate_cell_frame(
                    stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                    best_z=0.0, channel=self.active_channel, exposure_ms=self.exposure_ms,
                    gain=self.gain, led_power=self.led_power, is_light_on=self.is_light_on,
                    fov_size_mm=(self.focus_map.fov_width_mm, self.focus_map.fov_height_mm),
                    num_wells=self.well_plate_format.num_wells
                )
        else:
            # Stock lumaviewpro.py on unconnected PC renders DISCONNECTED camera overlay frame
            raw_frame = image_utils.generate_disconnected_camera_frame()

        proc_frame = self.channel_manager.process_channel_image(self.active_channel, raw_frame)
        self.render_frame_to_viewports(proc_frame)

    def render_frame_to_viewports(self, frame: np.ndarray):
        if frame is None or self.root_layout is None:
            return

        if hasattr(self.root_layout, 'ids') and 'main_camera_viewport' in self.root_layout.ids:
            vp = self.root_layout.ids.main_camera_viewport
            if hasattr(vp, 'set_frame'):
                vp.set_frame(frame)

        self.redraw_plate_canvas()

    def set_well_plate_format(self, num_wells_str: str):
        try:
            digits = re.findall(r'\d+', num_wells_str)
            num_wells = int(digits[0]) if digits else 96
            self.well_plate_format = labware.WellPlateFormat(num_wells=num_wells)
            self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
            self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

            # Move stage to Well A1 center of new format
            a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
            a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
            cx, cy, cz = self.clamp_stage_position(a1_x, a1_y, 0.0)
            self.stage_x, self.stage_y, self.stage_z = cx, cy, cz

            self.root_layout.selected_plate_format_str = f"{num_wells}-Well Plate"
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm" if getattr(self, 'stage_connected', False) else "Stage: DISCONNECTED"
            self.root_layout.status_msg = f"Well Plate Map set to: {num_wells}-Well Format"

            if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                self.root_layout.ids.main_plate_canvas.set_references(
                    self.well_plate_format,
                    self.plate_calibration,
                    self.focus_map,
                    self.fov_manager
                )

            self.update_active_well_display()
            self.update_live_feed(0)
        except Exception as e:
            self.root_layout.status_msg = f"Failed to set plate format: {e}"

    def change_well_plate_format(self, format_str: str):
        self.set_well_plate_format(format_str)

    def select_well_plate_format(self, format_str: str):
        self.set_well_plate_format(format_str)

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
        dx = dx_dir * step
        dy = dy_dir * step
        dz = dz_dir if abs(dz_dir) < 1.0 else dz_dir / 1000.0

        raw_x = self.stage_x + dx
        raw_y = self.stage_y + dy
        raw_z = self.stage_z + dz

        cx, cy, cz = self.clamp_stage_position(raw_x, raw_y, raw_z)
        self.stage_x = cx
        self.stage_y = cy
        self.stage_z = cz

        if self.motor_stage and hasattr(self.motor_stage, 'move_abs'):
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[MotorBoard move_abs error] {e}")

        self.update_active_well_display()
        if getattr(self, 'stage_connected', False):
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        else:
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

        self.update_live_feed(0)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        cx, cy, cz = self.clamp_stage_position(x_mm, y_mm, z_mm)
        self.stage_x = cx
        self.stage_y = cy
        self.stage_z = cz

        if self.motor_stage and hasattr(self.motor_stage, 'move_abs'):
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[MotorBoard move_abs error] {e}")

        self.update_active_well_display()
        if getattr(self, 'stage_connected', False):
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        else:
            self.root_layout.stage_coord_str = "Stage: DISCONNECTED"

        self.update_live_feed(0)

    def move_stage_abs(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        self.jog_stage_to(x_mm, y_mm, z_mm)

    def jog_z_stage(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z_focus(self, delta_um: float):
        self.jog_stage(0.0, 0.0, delta_um / 1000.0)

    def jog_z(self, delta_um: float):
        self.jog_z_focus(delta_um)

    def jog_z_microns(self, delta_um: float):
        self.jog_z_focus(delta_um)

    def zero_stage(self):
        a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
        a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
        self.jog_stage_to(a1_x, a1_y, 0.0)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        if hasattr(self, 'fov_manager') and hasattr(self.fov_manager, '_rebuild_all_groups'):
            self.fov_manager._rebuild_all_groups()

        # Re-clamp stage position so current FOV box stays inside physical plate
        cx, cy, cz = self.clamp_stage_position(self.stage_x, self.stage_y, self.stage_z)
        self.stage_x = cx
        self.stage_y = cy
        self.stage_z = cz

        self.root_layout.status_msg = f"Objective Lens set to: {preset_name} (Mag: {self.focus_map.lens_mag}x, FOV: {self.focus_map.fov_width_mm:.2f}x{self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    # CALIBRATION WIZARD WORKFLOW CONTROLLER
    def start_calibration_wizard(self):
        """
        1. Turns illumination Light ON (is_light_on = True).
        2. Shows 6 calibration target points on microplate canvas map.
        3. Drives stage to Step 1 nominal position (A1_12: Well A1 12 o'clock top rim).
        4. Updates instruction banner.
        """
        self.plate_calibration.is_calibrating = True
        self.plate_calibration.is_calibrated = False
        self.plate_calibration.current_step_index = 0
        self.plate_calibration.recorded_points.clear()

        # Automatically turn illumination Light ON
        self.is_light_on = True
        self.root_layout.is_light_on = True
        if self.leds and hasattr(self.leds, 'set_led'):
            try:
                image_utils.safe_set_led(self.leds, self.active_channel, int(self.led_power))
            except Exception as e:
                print(f"[LEDBoard set_led error] {e}")

        # Drive stage to Step 1 nominal target position
        tx, ty = self.plate_calibration.get_step_nominal_stage_coords(0)
        self.jog_stage_to(tx, ty)

        step_info = self.plate_calibration.get_current_step_info()
        self.root_layout.calibration_wizard_text = f"Step 1/6: Align reticle to {step_info['desc']}, then click Mark"
        self.root_layout.status_msg = "Calibration Wizard Started. Light ON. Stage moved to A1 12 o'clock."
        self.redraw_plate_canvas()

    def mark_calibration_step(self):
        """
        Records current stage position (self.stage_x, self.stage_y) for active step,
        then automatically advances to the next step and drives stage there.
        """
        if not self.plate_calibration.is_calibrating:
            self.start_calibration_wizard()
            return

        cur_idx = self.plate_calibration.current_step_index
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS):
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            next_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {next_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Marked Step {cur_idx + 1}. Stage moved to Step {next_idx + 1}."
        else:
            # All 6 points recorded! Compute matrix
            success = self.plate_calibration.compute_calibration()
            if success:
                self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
                self.root_layout.calibration_wizard_text = f"Calibration Complete! Rotation: {self.plate_calibration.rotation_angle_deg:.2f}°"
                self.root_layout.status_msg = "All 6 points marked! Plate Calibration Matrix computed & applied."
            else:
                self.root_layout.calibration_wizard_text = "Calibration failed. Verify target alignments."
                self.root_layout.status_msg = "Calibration computation failed."

        self.redraw_plate_canvas()

    def next_calibration_step(self):
        """Advances to the next step position without marking."""
        if not self.plate_calibration.is_calibrating:
            self.start_calibration_wizard()
            return

        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS) - 1:
            self.plate_calibration.current_step_index += 1
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)

            next_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {next_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved to Step {next_idx + 1}."
            self.redraw_plate_canvas()

    def previous_calibration_step(self):
        """Moves back to the previous step position without marking."""
        if not self.plate_calibration.is_calibrating:
            return

        if self.plate_calibration.current_step_index > 0:
            self.plate_calibration.current_step_index -= 1
            prev_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(prev_idx)
            self.jog_stage_to(tx, ty)

            prev_info = self.plate_calibration.get_current_step_info()
            self.root_layout.calibration_wizard_text = f"Step {prev_idx + 1}/6: Align reticle to {prev_info['desc']}, then click Mark"
            self.root_layout.status_msg = f"Moved back to Step {prev_idx + 1}."
            self.redraw_plate_canvas()

    def record_calibration_step(self):
        self.mark_calibration_step()

    def finish_plate_calibration(self):
        """Computes calibration matrix and applies it (Done button)."""
        success = self.plate_calibration.compute_calibration()
        if success:
            self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
            self.root_layout.calibration_wizard_text = f"Calibration Complete! Rotation: {self.plate_calibration.rotation_angle_deg:.2f}°"
            self.root_layout.status_msg = "Plate Calibration Matrix successfully computed & applied!"
        else:
            self.root_layout.calibration_wizard_text = "Calibration complete using nominal plate parameters."
            self.root_layout.calibrated_status = "Calibrated (Nominal)"
            self.root_layout.status_msg = "Calibration completed."
        self.redraw_plate_canvas()

    def reset_calibration(self):
        """Resets calibration state and hides target marks."""
        self.plate_calibration.reset_calibration()
        self.root_layout.calibrated_status = "Plate Uncalibrated"
        self.root_layout.calibration_wizard_text = "Click 'Start Calib' to begin 6-point target alignment."
        self.root_layout.status_msg = "Calibration Reset."
        self.redraw_plate_canvas()

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

_nav_kv = os.path.join(script_dir, 'plate_navigator.kv')
if os.path.exists(_nav_kv) and _nav_kv not in Builder.files:
    Builder.load_file(_nav_kv)


if __name__ == '__main__':
    LumaViewProApp().run()
