"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation Software.

Unified Single-Window UI Architecture (Zero Popups):
- Embedded Live Camera Viewport (Top-Left) with center reticle crosshairs & double-click pixel centering.
- Embedded Microplate Map Canvas (Top-Right) with double-click well/coordinate stage navigation.
- Hardware Motion Stage & Illumination Controls (Bottom-Left) with coarse/fine XY jogging & Z focus column.
- Keyboard navigation (Arrow keys for XY jog, Shift+Arrows for coarse, PageUp/PageDown for Z focus).
- Optical Channels Panel & Studio (Bottom-Right) with strict single-channel mutual exclusion.
- Physical Motor Guardrails: Clamps stage coordinates so FOV never extends beyond the physical plate.
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
    """Stores configuration and optical illumination/emission specs for an imaging channel."""
    
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
                "Brightfield", excitation_info="White LED", emission_info="Pass-through",
                color_hex="#FFFFFF", is_transmitted=True, btn_color=[0.9, 0.9, 0.9, 1.0]
            ),
            "Phase": ChannelConfig(
                "Phase", excitation_info="White Ring", emission_info="Phase Shift",
                color_hex="#CCCCCC", is_transmitted=True, btn_color=[0.65, 0.7, 0.75, 1.0]
            ),
            "Darkfield": ChannelConfig(
                "Darkfield", excitation_info="Oblique LED", emission_info="Darkfield",
                color_hex="#888888", is_transmitted=True, btn_color=[0.4, 0.4, 0.45, 1.0]
            ),
            "Blue": ChannelConfig(
                "Blue", excitation_info="370-410 nm", emission_info="429-462 nm",
                color_hex="#0080FF", is_transmitted=False, btn_color=[0.0, 0.45, 1.0, 1.0]
            ),
            "Green": ChannelConfig(
                "Green", excitation_info="473-491 nm", emission_info="502-561 nm",
                color_hex="#00FF40", is_transmitted=False, btn_color=[0.0, 0.85, 0.25, 1.0]
            ),
            "Red": ChannelConfig(
                "Red", excitation_info="594nm (580-598)", emission_info="612-680nm (TxRed)",
                color_hex="#FF3030", is_transmitted=False, btn_color=[0.95, 0.15, 0.15, 1.0]
            )
        }

    def resolve_channel_name(self, name: str) -> str:
        return self.ALIASES.get(name, name)

    def set_channel_active(self, channel_name: str, active: bool):
        real_name = self.resolve_channel_name(channel_name)
        if real_name in self.channels:
            if active:
                for ch in self.channels.values():
                    ch.active = False
                self.channels[real_name].active = True
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


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - Live Frame Texture Rendering from numpy array
    - Mouse Wheel Zooming & Right-Click Drag Panning
    - Center Reticle Crosshairs Overlay
    - Double-Click Pixel Centering (moves stage to clicked feature)
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.current_texture = None
        self.bind(pos=self.redraw_viewport, size=self.redraw_viewport, zoom_level=self.redraw_viewport, pan_x=self.redraw_viewport, pan_y=self.redraw_viewport)

    def set_frame(self, frame_array: np.ndarray):
        if frame_array is None or frame_array.size == 0:
            return
        try:
            h, w = frame_array.shape[:2]
            if frame_array.ndim == 3 and frame_array.shape[2] == 3:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            elif frame_array.ndim == 2:
                rgb_buf = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
            else:
                return

            buf = np.ascontiguousarray(rgb_buf[::-1, :, :], dtype=np.uint8).tobytes()

            if self.current_texture is None or self.current_texture.width != w or self.current_texture.height != h:
                self.current_texture = Texture.create(size=(w, h), colorfmt='rgb')

            self.current_texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
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
            else:
                Color(0.2, 0.2, 0.25, 1.0)
                Rectangle(pos=self.pos, size=self.size)

            # Red Reticle Crosshairs Overlay at center of viewport
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            Color(1.0, 0.2, 0.2, 0.8)
            Line(points=[self.x, cy, self.x + self.width, cy], width=1.0)
            Line(points=[cx, self.y, cx, self.y + self.height], width=1.0)
            Line(circle=(cx, cy, 12), width=1.2)

    def reset_zoom_pan(self):
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

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
            if app and hasattr(app, 'center_fov_on_pixel'):
                app.center_fov_on_pixel(self, touch.x, touch.y)
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


class ChannelControlWidget(BoxLayout):
    channel_name = StringProperty("")
    channel_spec_info = StringProperty("")
    ref_image_path = StringProperty("No Ref Image")
    is_active = BooleanProperty(False)
    btn_bg_color = ListProperty([0.2, 0.6, 0.8, 1.0])

    def open_ref_chooser(self):
        App.get_running_app().open_file_chooser_for_channel(self.channel_name)


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
    camera_gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    led_power_str = StringProperty("100.0")
    selected_plate_format_str = StringProperty("96-Well Plate")


class LumaViewProApp(App):
    """Main Application Controller."""

    # Class-level Kivy properties bound to layout inputs and callbacks
    is_live_imaging = BooleanProperty(True)
    is_light_on = BooleanProperty(False)  # Launch with light OFF
    active_channel = StringProperty("Brightfield")
    camera_gain = NumericProperty(1.0)
    gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    exposure_ms = NumericProperty(20.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)
    live_fps = NumericProperty(20)

    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)

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

        # Stage Coordinates (Default Well A1)
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        self.camera = None
        self._updating_channel_ui = False
        self.navigator_popup = None

        # Bind Keyboard Navigation (Arrow Keys for XY jog, PageUp/Down for Z focus)
        Window.bind(on_key_down=self._on_keyboard_down)

        self.root_layout = LumaViewProMainLayout()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        
        self.build_channel_controls()

        # Connect microplate canvas references in top-right
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.set_references(
                self.well_plate_format,
                self.plate_calibration,
                self.focus_map,
                self.fov_manager
            )

        # Connect hardware or simulator driver
        self.init_hardware()

        # Start live camera feed update clock loop @ 20 FPS (0.05s interval)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 20.0)

        return self.root_layout

    def init_hardware(self):
        try:
            import pyloncamera
            self.camera = pyloncamera.PylonCamera()
            if not self.camera.connect():
                self.camera = None
        except Exception:
            self.camera = None

    def clamp_stage_position(self, x: float, y: float, z: Optional[float] = None) -> Tuple[float, float, float]:
        """Physical Guardrails: Clamps stage position so the active FOV never extends beyond the physical plate."""
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

        if z is not None:
            clamped_z = max(0.0, min(10.0, float(z)))
        else:
            clamped_z = self.stage_z

        return clamped_x, clamped_y, clamped_z

    def _on_keyboard_down(self, window, key, scancode, codepoint, modifiers):
        """Keyboard Navigation: Arrow keys for XY motion, PageUp/Down for Z focus."""
        is_shift = 'shift' in modifiers
        step_xy = 5.0 if is_shift else 0.5
        step_z = 0.010 if is_shift else 0.001

        if key == 273:    # Up Arrow (+Y)
            self.jog_stage(0, step_xy, 0)
            return True
        elif key == 274:  # Down Arrow (-Y)
            self.jog_stage(0, -step_xy, 0)
            return True
        elif key == 276:  # Left Arrow (-X)
            self.jog_stage(-step_xy, 0, 0)
            return True
        elif key == 275:  # Right Arrow (+X)
            self.jog_stage(step_xy, 0, 0)
            return True
        elif key == 280:  # PageUp (+Z)
            self.jog_z_focus(step_z * 1000.0)
            return True
        elif key == 281:  # PageDown (-Z)
            self.jog_z_focus(-step_z * 1000.0)
            return True
        return False

    def center_fov_on_pixel(self, widget, touch_x: float, touch_y: float):
        """Calculates stage distance from clicked viewport pixel to center crosshairs and jogs stage."""
        if not widget or widget.width <= 0 or widget.height <= 0:
            return

        cx = widget.x + widget.width / 2.0 + widget.pan_x
        cy = widget.y + widget.height / 2.0 + widget.pan_y

        dx_px = touch_x - cx
        dy_px = touch_y - cy

        if widget.current_texture is not None:
            tw, th = widget.current_texture.size
            scale_w = widget.width / float(tw)
            scale_h = widget.height / float(th)
            base_scale = min(scale_w, scale_h)
            draw_w = tw * base_scale * widget.zoom_level
            draw_h = th * base_scale * widget.zoom_level
        else:
            draw_w = widget.width * widget.zoom_level
            draw_h = widget.height * widget.zoom_level

        fov_w = self.focus_map.fov_width_mm
        fov_h = self.focus_map.fov_height_mm

        dx_mm = (dx_px / max(1.0, draw_w)) * fov_w
        dy_mm = (dy_px / max(1.0, draw_h)) * fov_h

        target_x = self.stage_x + dx_mm
        target_y = self.stage_y + dy_mm

        self.jog_stage_to(target_x, target_y)
        self.root_layout.status_msg = f"Centered FOV on clicked pixel (X={self.stage_x:.2f}, Y={self.stage_y:.2f} mm)"

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

    def update_active_well_display(self):
        try:
            r = max(0, min(self.well_plate_format.rows - 1, int(round((31.5 - self.stage_y) / self.well_plate_format.pitch_y))))
            c = max(0, min(self.well_plate_format.cols - 1, int(round((self.stage_x + 49.5) / self.well_plate_format.pitch_x))))
            w_name = self.well_plate_format.get_well_name(r, c)
            self.root_layout.current_well_display = w_name
        except Exception:
            pass

    def redraw_plate_canvas(self):
        if hasattr(self.root_layout, 'ids') and 'main_plate_canvas' in self.root_layout.ids:
            self.root_layout.ids.main_plate_canvas.redraw_canvas()

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
            else:
                if real_ch == self.active_channel:
                    self.is_light_on = False
                    self.root_layout.is_light_on = False
                    self.channel_manager.set_channel_active(real_ch, False)
                    self.root_layout.status_msg = f"Illumination OFF ({real_ch})"

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
        self.sync_channel_ui_toggles()
        self.update_live_feed(0)

    def toggle_light(self):
        self.toggle_light_state()

    def toggle_illumination_light(self):
        self.toggle_light_state()

    def toggle_light_master(self, is_down: bool):
        self.toggle_light_state(is_down)

    def toggle_live_feed(self):
        self.root_layout.is_live_imaging = not self.root_layout.is_live_imaging

    def set_live_fps(self, fps_str: str):
        try:
            digits = re.findall(r'\d+', fps_str)
            fps_val = int(digits[0]) if digits else 20
            self.root_layout.live_fps_str = f"{fps_val} FPS"
        except Exception:
            pass

    def set_exposure_ms(self, val_text: str):
        try:
            val = float(val_text)
            self.exposure_ms = val
            self.camera_exposure_ms = val
            self.root_layout.exposure_str = f"{val:.1f}"
            self.update_live_feed(0)
        except ValueError:
            pass

    def set_camera_exposure(self, val_text: str):
        self.set_exposure_ms(val_text)

    def set_gain_val(self, val_text: str):
        try:
            val = float(val_text)
            self.gain = val
            self.camera_gain = val
            self.root_layout.gain_str = f"{val:.1f}"
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
            self.update_live_feed(0)
        except ValueError:
            pass

    def update_live_feed(self, dt):
        if not hasattr(self.root_layout, 'is_live_imaging') or not self.root_layout.is_live_imaging:
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
                self.camera.num_wells = self.well_plate_format.num_wells
                self.camera.fov_size_mm = (fov_w, fov_h)

            raw_frame = self.camera.get_frame()
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
            viewport = self.root_layout.ids.main_camera_viewport
            if hasattr(viewport, 'set_frame'):
                viewport.set_frame(frame)

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
            self.stage_x, self.stage_y, self.stage_z = self.clamp_stage_position(a1_x, a1_y, 0.0)

            self.root_layout.selected_plate_format_str = f"{num_wells}-Well Plate"
            self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
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

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.update_live_feed(0)

    def jog_stage_to(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        cx, cy, cz = self.clamp_stage_position(x_mm, y_mm, z_mm)
        self.stage_x = cx
        self.stage_y = cy
        self.stage_z = cz

        self.update_active_well_display()
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"
        self.update_live_feed(0)

    def move_stage_abs(self, x_mm: float, y_mm: float, z_mm: Optional[float] = None):
        self.jog_stage_to(x_mm, y_mm, z_mm)

    def jog_z_stage(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z_focus(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def jog_z_microns(self, dz_um: float):
        self.jog_stage(0.0, 0.0, dz_um / 1000.0)

    def zero_stage(self):
        a1_x = -self.well_plate_format.PLATE_LENGTH_MM / 2.0 + self.well_plate_format.a1_offset_x
        a1_y = self.well_plate_format.PLATE_WIDTH_MM / 2.0 - self.well_plate_format.a1_offset_y
        self.jog_stage_to(a1_x, a1_y, 0.0)

    def set_objective_preset(self, preset_name: str):
        self.focus_map.set_objective_preset(preset_name)
        self.fov_manager._rebuild_all_groups()

        # Re-clamp stage position so current FOV box stays inside physical plate
        cx, cy, cz = self.clamp_stage_position(self.stage_x, self.stage_y, self.stage_z)
        self.stage_x = cx
        self.stage_y = cy
        self.stage_z = cz

        self.root_layout.status_msg = f"Objective Lens set to: {preset_name} (Mag: {self.focus_map.lens_mag}x, FOV: {self.focus_map.fov_width_mm:.2f}x{self.focus_map.fov_height_mm:.2f} mm)"
        self.redraw_plate_canvas()
        self.update_live_feed(0)

    def set_turret_hardware_type(self, type_str: str):
        self.focus_map.set_hardware_nosepiece(type_str)
        self.root_layout.status_msg = f"Nosepiece Hardware set to: {type_str}"

    def rotate_turret_slot(self, slot_num: int):
        slot_map = {1: "4x", 2: "10x", 3: "20x Air", 4: "40x", 5: "60x Oil", 6: "100x Oil"}
        preset_name = slot_map.get(slot_num, "20x Air")
        self.set_objective_preset(preset_name)

    def assign_channel_turret_slot(self, assign_str: str):
        if "->" in assign_str:
            ch_part, slot_part = assign_str.split("->")
            ch_name = ch_part.strip()
            digits = re.findall(r'\d+', slot_part)
            slot_num = int(digits[0]) if digits else 1
            self.focus_map.per_channel_turret_slots[ch_name] = slot_num
            self.root_layout.status_msg = f"Assigned {ch_name} -> Turret Slot {slot_num}"

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

    def set_primary_af_channel(self, channel_name: str):
        self.focus_map.set_primary_af_channel(channel_name)
        self.root_layout.status_msg = f"Primary AF Channel set to: {channel_name}"

    def set_af_binning(self, binning_str: str):
        self.focus_map.set_af_camera_config(binning_str, self.focus_map.af_exposure_ms)
        self.root_layout.status_msg = f"AF Camera Binning: {binning_str}"

    def set_af_exposure(self, exposure_str: str):
        try:
            val = float(exposure_str)
            self.exposure_ms = val
            self.camera_exposure_ms = val
            self.focus_map.set_af_camera_config(self.focus_map.af_binning, val)
            self.root_layout.status_msg = f"Exposure Time set to {val:.1f} ms"
        except ValueError:
            pass

    def set_af_cycle_interval(self, text_or_num):
        try:
            digits = re.findall(r'\d+', str(text_or_num))
            val = int(digits[0]) if digits else 1
            self.focus_map.set_af_cycle_interval(val)
            self.root_layout.status_msg = f"Autofocus configured to run every {val} cycle(s)"
        except Exception:
            pass

    def set_time_lapse_config(self, enabled: bool, interval_mins_str: str, cycles_str: str):
        try:
            inv_m = float(interval_mins_str) if interval_mins_str else 10.0
            cyc = int(cycles_str) if cycles_str else 24
            self.focus_map.set_time_lapse_config(enabled, inv_m, cyc)
            status = "Enabled" if enabled else "Disabled"
            self.root_layout.status_msg = f"Time-Lapse {status}: {cyc} cycles @ {inv_m} min interval"
        except Exception:
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

    def set_chromatic_z_offset(self, channel_name: str):
        self.focus_map.set_chromatic_z_offset(channel_name, self.stage_z)
        delta_z = self.focus_map.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        self.root_layout.status_msg = f"Saved Chromatic Z-Offset for {channel_name}: {delta_z*1000.0:.2f} um"

    def apply_nyquist_calculation(self):
        dof_um, nyquist_step_um = self.focus_map.calculate_optical_nyquist()
        self.focus_map.apply_nyquist_to_af_config()
        self.root_layout.status_msg = f"Nyquist Step Applied: {nyquist_step_um:.3f} um (Depth of Field: {dof_um:.3f} um)"

    def apply_grid_settings(self, rows_str: str, cols_str: str, single_af: bool):
        try:
            r = int(rows_str)
            c = int(cols_str)
            self.fov_manager.set_grid_dimensions(r, c, single_af=single_af)
            self.root_layout.status_msg = f"Tile Grid set to {r}x{c}"
            self.redraw_plate_canvas()
        except ValueError:
            pass

    def global_sync_active_channels(self, active_channel_states: Optional[Dict[str, bool]] = None):
        self.channel_manager.global_sync_active_channels(active_channel_states)
        active_list = self.channel_manager.get_active_channel_names()
        self.fov_manager.apply_channels_to_all_selected(active_list)
        self.root_layout.status_msg = f"Synced Channels ({', '.join(active_list)}) across selected wells"

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


# Register custom widget classes with Factory BEFORE loading KV layout rules
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
