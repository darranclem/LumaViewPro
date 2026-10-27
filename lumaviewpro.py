"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation.

Unified Single-Window UI Architecture (Zero Popups, Zero Redundant Launch Buttons):
- Top Navigation Bar: Application Title, Active Well Display, Stage Coordinates, Calibration Status.
- Top-Left (50% Width): Live Camera Viewport (ZoomPanCameraWidget) with reticle crosshairs & double-click pixel centering.
- Bottom-Left (50% Width): Hardware Motion Stage & Illumination Deck (Exp, Gain, LED %, Light ON/OFF, Grouped XY Cross, Z Column, Snapshot).
- Top-Right (50% Width): Microplate Canvas Map (InteractiveScaledPlateCanvas) with Format Selector (6 to 384 wells), Objective Selector (1.25x to 100x Oil), active FOV cyan reticle, and double-click well stage navigation.
- Bottom-Right (50% Width): Optical Channels List (Single-channel mutual exclusion) + 3-Tab Registration, Autofocus & Protocol Studio.
"""

from __future__ import annotations
import os
import sys
import json
import math
import re
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
import plate_navigator

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
        excitation_range_nm: str = "",
        emission_range_nm: str = "",
        color_hex: str = "#FFFFFF",
        is_transmitted: bool = False,
        btn_color: List[float] = None
    ):
        self.name: str = name
        self.excitation_range_nm: str = excitation_range_nm
        self.emission_range_nm: str = emission_range_nm
        self.color_hex: str = color_hex
        self.is_transmitted: bool = is_transmitted
        self.btn_color: List[float] = btn_color if btn_color else [0.2, 0.6, 0.8, 1.0]
        self.active: bool = False
        self.exposure_ms: float = 20.0
        self.gain: float = 1.0
        self.led_power: float = 100.0
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
            "Brightfield": ChannelConfig(
                "Brightfield", excitation_range_nm="White LED", emission_range_nm="Pass-through",
                color_hex="#FFFFFF", is_transmitted=True, btn_color=[0.9, 0.9, 0.9, 1.0]
            ),
            "Phase": ChannelConfig(
                "Phase", excitation_range_nm="White LED (Phase Ring)", emission_range_nm="Pass-through",
                color_hex="#CCCCCC", is_transmitted=True, btn_color=[0.65, 0.7, 0.75, 1.0]
            ),
            "Darkfield": ChannelConfig(
                "Darkfield", excitation_range_nm="Oblique LED", emission_range_nm="Pass-through",
                color_hex="#888888", is_transmitted=True, btn_color=[0.4, 0.4, 0.45, 1.0]
            ),
            "Blue": ChannelConfig(
                "Blue", excitation_range_nm="370-410 nm", emission_range_nm="429-462 nm",
                color_hex="#0080FF", is_transmitted=False, btn_color=[0.0, 0.45, 1.0, 1.0]
            ),
            "Green": ChannelConfig(
                "Green", excitation_range_nm="473-491 nm", emission_range_nm="502-561 nm",
                color_hex="#00FF40", is_transmitted=False, btn_color=[0.0, 0.85, 0.25, 1.0]
            ),
            "Red": ChannelConfig(
                "Red", excitation_range_nm="580-598 nm", emission_range_nm="612-680 nm",
                color_hex="#FF3030", is_transmitted=False, btn_color=[0.95, 0.15, 0.15, 1.0]
            )
        }
        self.channels["Brightfield"].active = True

    def resolve_channel_name(self, name: str) -> str:
        return self.ALIASES.get(name, name)

    def set_channel_active(self, channel_name: str, active: bool):
        canonical = self.resolve_channel_name(channel_name)
        if canonical in self.channels:
            if active:
                for ch in self.channels.values():
                    ch.active = False
                self.channels[canonical].active = True
            else:
                self.channels[canonical].active = False

    def global_sync_active_channels(self, active_channel_states: Optional[Dict[str, bool]] = None):
        if active_channel_states:
            for name, active_val in active_channel_states.items():
                canonical = self.resolve_channel_name(name)
                if canonical in self.channels:
                    self.channels[canonical].active = active_val

    def get_active_channel_names(self) -> List[str]:
        return [ch_name for ch_name, ch_cfg in self.channels.items() if ch_cfg.active]

    def get_active_channel_name(self) -> str:
        names = self.get_active_channel_names()
        return names[0] if names else "Brightfield"

    def process_channel_image(self, channel_name: str, raw_frame: object) -> object:
        canonical = self.resolve_channel_name(channel_name)
        ch = self.channels.get(canonical)
        if (ch and ch.shading_correction_enabled and ch.ref_image_array is not None):
            return image_utils.apply_shading_correction(raw_frame, ch.ref_image_array)
        return raw_frame


class ChannelControlWidget(BoxLayout):
    """UI Widget representing an optical channel row in the main window."""
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
        self.bind(pos=self.redraw_viewport, size=self.redraw_viewport,
                  zoom_level=self.redraw_viewport, pan_x=self.redraw_viewport, pan_y=self.redraw_viewport)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

    def set_frame(self, frame_array: np.ndarray):
        if frame_array is None or frame_array.size == 0:
            return

        h, w = frame_array.shape[:2]
        if len(frame_array.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
        elif frame_array.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_array[:, :, :3]

        frame_flipped = np.ascontiguousarray(frame_rgb[::-1, :, :], dtype=np.uint8)

        try:
            if self.texture is None or self.texture.size != (w, h):
                self.texture = Texture.create(size=(w, h), colorfmt='rgb')

            self.texture.blit_buffer(frame_flipped.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
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

            if self.texture is not None:
                vw = self.width * self.zoom_level
                vh = self.height * self.zoom_level
                vx = self.x + (self.width - vw) / 2.0 + self.pan_x
                vy = self.y + (self.height - vh) / 2.0 + self.pan_y

                Color(1.0, 1.0, 1.0, 1.0)
                Rectangle(texture=self.texture, pos=(vx, vy), size=(vw, vh))

            # Reticle Crosshairs
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0

            Color(1.0, 0.2, 0.2, 0.85)  # Red Crosshair
            Line(points=[cx - 20, cy, cx + 20, cy], width=1.5)
            Line(points=[cx, cy - 20, cx, cy + 20], width=1.5)
            Line(circle=(cx, cy, 8), width=1.2)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                self.zoom_level = max(0.5, self.zoom_level - 0.1)
            elif touch.button == 'scrollup':
                self.zoom_level = min(5.0, self.zoom_level + 0.1)
            return True

        if touch.is_double_tap and touch.button == 'left':
            app = App.get_running_app()
            if app and hasattr(app, 'focus_map'):
                fov_w = app.focus_map.fov_width_mm
                fov_h = app.focus_map.fov_height_mm

                cx = self.x + self.width / 2.0 + self.pan_x
                cy = self.y + self.height / 2.0 + self.pan_y

                vw = self.width * self.zoom_level
                vh = self.height * self.zoom_level

                off_px_x = touch.x - cx
                off_px_y = touch.y - cy

                dx_mm = (off_px_x / vw) * fov_w
                dy_mm = -(off_px_y / vh) * fov_h

                if hasattr(app, 'jog_stage_to'):
                    app.jog_stage_to(app.stage_x + dx_mm, app.stage_y + dy_mm)
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
    camera_gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    led_power_str = StringProperty("100.0")
    selected_plate_format_str = StringProperty("96-Well Plate")
    calibration_wizard_text = StringProperty("Click 'Start Calib' to begin 6-point plate alignment.")


class LumaViewProApp(App):
    """Main Application Controller."""

    is_light_on = BooleanProperty(False)
    active_channel = StringProperty("Brightfield")
    exposure_ms = NumericProperty(20.0)
    camera_exposure_ms = NumericProperty(20.0)
    gain = NumericProperty(1.0)
    camera_gain = NumericProperty(1.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)
    coarse_xy_step_mm = NumericProperty(5.0)
    fine_xy_step_mm = NumericProperty(0.5)
    coarse_z_step_um = NumericProperty(10.0)
    fine_z_step_um = NumericProperty(1.0)
    live_fps = NumericProperty(20)
    is_live_imaging = BooleanProperty(True)
    protocol_output_dir = StringProperty(os.path.expanduser('~/LumaViewPro_Experiments'))

    # Selective hardware device connection status
    camera_connected = BooleanProperty(False)
    stage_connected = BooleanProperty(False)
    focus_connected = BooleanProperty(False)
    led_connected = BooleanProperty(False)

    def load_kv(self, filename=None):
        """Single-instance KV load override to prevent duplicate layout parsing."""
        if not hasattr(self, '_kv_loaded_flag'):
            self._kv_loaded_flag = True
            kv_path = os.path.join(script_dir, 'lumaviewpro.kv')
            if os.path.exists(kv_path) and kv_path not in Builder.files:
                Builder.load_file(kv_path)
        return None

    def build(self):
        self.title = "LumaViewPro Microscope Automation"
        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

        # Stage coordinates (default Well A1)
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        self._updating_channel_ui = False
        self.root_layout = LumaViewProMainLayout()

        # Connect hardware devices selectively
        self.init_hardware()

        self.build_channel_controls()
        self.update_active_well_display()

        # Schedule live 20 FPS camera feed update loop
        Clock.schedule_interval(self.update_live_feed, 1.0 / 20.0)

        return self.root_layout

    def init_hardware(self):
        # Camera Driver
        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            if cam.connect():
                self.camera = cam
                self.camera_connected = True
            else:
                self.camera = None
                self.camera_connected = False
        except Exception:
            self.camera = None
            self.camera_connected = False

        # Stage Motion Driver
        try:
            import motorboard
            stg = motorboard.MotorBoard()
            if stg.connect():
                self.motor_stage = stg
                self.stage_connected = True
                self.focus_connected = True
            else:
                self.motor_stage = None
                self.stage_connected = False
                self.focus_connected = False
        except Exception:
            self.motor_stage = None
            self.stage_connected = False
            self.focus_connected = False

        # LED Illumination Driver
        try:
            import ledboard
            led = ledboard.LEDBoard()
            self.leds = led
            self.led_connected = True
        except Exception:
            self.leds = None
            self.led_connected = False

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
        if not hasattr(self.root_layout, 'ids') or 'main_channel_container' not in self.root_layout.ids:
            return

        self._updating_channel_ui = True
        try:
            container = self.root_layout.ids.main_channel_container
            container.clear_widgets()

            for ch_name, ch_cfg in self.channel_manager.channels.items():
                widget = ChannelControlWidget()
                widget.channel_name = ch_name
                if ch_cfg.is_transmitted:
                    widget.channel_spec_info = f"{ch_cfg.excitation_range_nm}"
                else:
                    widget.channel_spec_info = f"Ex: {ch_cfg.excitation_range_nm} | Em: {ch_cfg.emission_range_nm}"
                widget.is_active = (ch_name == self.active_channel) and self.is_light_on
                widget.btn_bg_color = ch_cfg.btn_color
                widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref Image"
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
                for name, cfg in self.channel_manager.channels.items():
                    cfg.active = (name == real_ch)
                self.active_channel = real_ch
                self.is_light_on = True
                self.root_layout.is_light_on = True
                self.root_layout.status_msg = f"Optical Channel {real_ch}: Active | Light: ON"
            else:
                if self.active_channel == real_ch:
                    self.is_light_on = False
                    self.root_layout.is_light_on = False
                    self.channel_manager.channels[real_ch].active = False
                    self.root_layout.status_msg = f"Optical Channel {real_ch}: OFF"

            self.sync_channel_ui_toggles()
        finally:
            self._updating_channel_ui = False
            self.update_live_feed(0)

    def select_channel(self, channel_name: str):
        self.set_active_channel(channel_name, True)

    def select_optical_channel(self, channel_name: str):
        self.set_active_channel(channel_name, True)

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
        if not self.is_light_on:
            for ch in self.channel_manager.channels.values():
                ch.active = False
            self.root_layout.status_msg = "Illumination Light: OFF (Dark Sensor State)"
        else:
            self.channel_manager.set_channel_active(self.active_channel, True)
            self.root_layout.status_msg = f"Illumination Light: ON | Active Channel: {self.active_channel}"

        self.sync_channel_ui_toggles()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    def set_exposure_ms(self, val_text: str):
        try:
            val = float(val_text)
            self.exposure_ms = max(0.1, min(5000.0, val))
            self.camera_exposure_ms = self.exposure_ms
            self.root_layout.exposure_str = f"{self.exposure_ms:.1f}"
            self.root_layout.status_msg = f"Exposure Time set to {self.exposure_ms:.1f} ms"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_gain_val(self, val_text: str):
        try:
            val = float(val_text)
            self.gain = max(1.0, min(20.0, val))
            self.camera_gain = self.gain
            self.root_layout.gain_str = f"{self.gain:.1f}"
            self.root_layout.status_msg = f"Camera Gain set to {self.gain:.1f}x"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            self.root_layout.led_power_str = f"{self.led_power:.0f}"
            self.root_layout.status_msg = f"LED Power set to {self.led_power:.0f}%"
            self.update_live_feed(0)
        except ValueError:
            pass

    def update_live_feed(self, dt):
        if not hasattr(self, 'root_layout') or self.root_layout is None:
            return

        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        if self.camera is not None and getattr(self.camera, 'is_connected', False):
            if hasattr(self.camera, 'stage_x'):
                self.camera.stage_x = self.stage_x
                self.camera.stage_y = self.stage_y
                self.camera.stage_z = self.stage_z
                self.camera.active_channel = self.active_channel
                self.camera.is_light_on = self.is_light_on
                self.camera.exposure_ms = self.exposure_ms
                self.camera.gain = self.gain
                self.camera.led_power = self.led_power

            if self.is_light_on:
                raw_frame = self.camera.get_frame()
            else:
                raw_frame = image_utils.generate_well_plate_cell_frame(
                    stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
                    best_z=0.0, channel=self.active_channel, exposure_ms=self.exposure_ms,
                    gain=self.gain, led_power=self.led_power, is_light_on=False,
                    fov_size_mm=(fov_w, fov_h), num_wells=self.well_plate_format.num_wells
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
            viewport = self.root_layout.ids.main_camera_viewport
            if hasattr(viewport, 'set_frame'):
                viewport.set_frame(frame)

        self.redraw_plate_canvas()

    def set_well_plate_format(self, num_wells_str: str):
        try:
            digits = re.findall(r'\d+', num_wells_str)
            num_wells = int(digits[0]) if digits else 96
            if num_wells in labware.WellPlateFormat.STANDARD_FORMATS:
                self.well_plate_format = labware.WellPlateFormat(num_wells=num_wells)
                self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
                self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

                self.stage_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
                self.stage_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y

                if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
                    canvas_widget = self.root_layout.ids.main_plate_canvas
                    canvas_widget.set_references(
                        self.well_plate_format,
                        self.plate_calibration,
                        self.focus_map,
                        self.fov_manager
                    )

                self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
                self.root_layout.selected_plate_format_str = f"{num_wells}-Well Plate"
                self.root_layout.current_well_display = "A1"
                self.root_layout.status_msg = f"Well Plate Map updated to {num_wells}-Well Plate"
                self.update_live_feed(0)
        except Exception as e:
            if hasattr(self.root_layout, 'status_msg'):
                self.root_layout.status_msg = f"Failed to set plate format: {e}"

    def set_jog_step(self, step_str: str):
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
            self.root_layout.status_msg = f"XY Step Size set to {val:.1f} mm"
        except (IndexError, ValueError):
            self.jog_step_mm = 1.0

    def set_jog_step_size(self, step_str: str):
        self.set_jog_step(step_str)

    def jog_stage(self, dx_dir: float, dy_dir: float, dz_dir: float = 0.0):
        dx = dx_dir * self.jog_step_mm if abs(dx_dir) <= 10.0 else dx_dir
        dy = dy_dir * self.jog_step_mm if abs(dy_dir) <= 10.0 else dy_dir
        dz = dz_dir

        max_x = self.well_plate_format.PLATE_LENGTH_MM / 2.0
        max_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0

        self.stage_x = max(-max_x, min(max_x, self.stage_x + dx))
        self.stage_y = max(-max_y, min(max_y, self.stage_y + dy))
        self.stage_z = max(0.0, min(10.0, self.stage_z + dz))

        if hasattr(self, 'motor_stage') and self.motor_stage:
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception:
                pass

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def jog_z_stage(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z_focus(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        max_x = self.well_plate_format.PLATE_LENGTH_MM / 2.0
        max_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0

        self.stage_x = max(-max_x, min(max_x, float(x_mm)))
        self.stage_y = max(-max_y, min(max_y, float(y_mm)))
        if z_mm is not None:
            self.stage_z = max(0.0, min(10.0, float(z_mm)))

        if hasattr(self, 'motor_stage') and self.motor_stage:
            try:
                self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception:
                pass

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def zero_stage(self):
        a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
        a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
        self.jog_stage_to(a1_x, a1_y, 0.0)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        self.fov_manager._rebuild_all_groups()
        self.root_layout.status_msg = f"Objective Lens set to: {preset_name} (Mag: {self.focus_map.lens_mag}x, FOV: {self.focus_map.fov_width_mm:.2f}x{self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

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
        self.root_layout.calibration_wizard_text = f"Step 1/6: Align reticle to {step_info['desc']}, then click Mark Target"
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
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark Target"
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
            self.root_layout.calibration_wizard_text = f"Step {next_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark Target"
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
            self.root_layout.calibration_wizard_text = f"Step {prev_idx + 1}/6: Align reticle to {step_info['desc']}, then click Mark Target"
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

    def set_primary_af_channel(self, channel_name: str):
        self.focus_map.set_primary_af_channel(channel_name)
        self.root_layout.status_msg = f"Primary AF Channel set to: {channel_name}"

    def set_af_binning(self, binning_str: str):
        self.focus_map.set_af_camera_config(binning_str, self.focus_map.af_exposure_ms)
        self.root_layout.status_msg = f"AF Camera Binning: {binning_str}"

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
        active_ch = self.channel_manager.get_active_channel_names()
        ch_name = active_ch[0] if active_ch else "Brightfield"
        frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x, stage_y=self.stage_y, stage_z=self.stage_z,
            best_z=0.0, channel=ch_name, is_light_on=self.is_light_on,
            led_power=self.led_power, num_wells=self.well_plate_format.num_wells
        )
        os.makedirs("snapshots", exist_ok=True)
        fn = f"snapshots/snapshot_{ch_name}_{self.root_layout.current_well_display}.png"
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
if 'ZoomPanCameraWidget' not in Factory.classes:
    Factory.register('ZoomPanCameraWidget', cls=ZoomPanCameraWidget)
if 'ChannelControlWidget' not in Factory.classes:
    Factory.register('ChannelControlWidget', cls=ChannelControlWidget)
if 'LumaViewProMainLayout' not in Factory.classes:
    Factory.register('LumaViewProMainLayout', cls=LumaViewProMainLayout)
if 'InteractiveScaledPlateCanvas' not in Factory.classes:
    Factory.register('InteractiveScaledPlateCanvas', cls=plate_navigator.InteractiveScaledPlateCanvas)


if __name__ == '__main__':
    LumaViewProApp().run()
