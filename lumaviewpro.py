"""
lumaviewpro.py - Main Application Controller for LumaViewPro Microscope Automation Software.

Enforces:
1. Unified Single-Window Kivy Architecture:
   - Top-Left: Live Camera Viewport with Mouse Zoom/Pan, Red Reticle Crosshairs, and Double-Click Pixel Centering.
   - Top-Right: Embedded Microplate Registration Studio & Canvas with Format Selector (6 to 384 wells) and Double-Click Well Navigation.
   - Bottom-Left: Hardware Motion & Camera Controls Deck (XY Stage Jog, Fine/Coarse Z Focus, Exp, Gain, LED Power, AF Search Range).
   - Bottom-Right: Optical Channels Panel with strict single-channel mutual exclusion.
2. Single-instance KV loading (prevents layout duplication and PopupException).
3. Dark sensor launch state (is_light_on = False -> black camera feed).
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
import plate_navigator  # TOP-LEVEL IMPORT REQUIRED FOR FACTORY REGISTRATION

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
        """Loads and assigns a shading correction reference image for this channel."""
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
        # Brightfield selected by default but illumination starts OFF
        self.channels["Brightfield"].active = True

    def resolve_channel_name(self, name: str) -> str:
        return self.ALIASES.get(name, name)

    def set_channel_active(self, channel_name: str, active: bool):
        """MUTUAL EXCLUSION: Turns selected channel ON and deactivates all others."""
        real_name = self.resolve_channel_name(channel_name)
        if active:
            for ch_key, ch_cfg in self.channels.items():
                ch_cfg.active = (ch_key == real_name)
        else:
            if real_name in self.channels:
                self.channels[real_name].active = False

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


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - Live Frame Texture Rendering
    - Red Reticle Crosshairs Overlay at Center of Viewport
    - Mouse Wheel Zooming (0.5x to 5.0x) and Right-Click Drag Panning
    - Double-Click Pixel Centering (moves stage so clicked pixel becomes FOV center)
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)
        self.texture = None

    def reset_zoom_pan(self):
        """Resets zoom and pan state back to default center."""
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.redraw_viewport()

    def set_frame(self, frame_array: np.ndarray):
        """Converts in-memory BGR NumPy frame to Kivy Texture and renders on Canvas with Reticle."""
        if frame_array is None or frame_array.size == 0:
            return

        try:
            h, w = frame_array.shape[:2]
            # Flip image vertically for Kivy coordinate space using pure NumPy
            flipped = np.ascontiguousarray(frame_array[::-1, :, :])
            
            texture = Texture.create(size=(w, h), colorfmt='bgr')
            texture.blit_buffer(flipped.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture
            self.redraw_viewport()
        except Exception as e:
            print(f"[ZoomPanCameraWidget Error] set_frame failed: {e}")

    def redraw_viewport(self, *args):
        """Draws camera texture centered on canvas with reticle crosshairs overlay."""
        self.canvas.clear()
        if self.width <= 0 or self.height <= 0:
            return

        with self.canvas:
            # Background viewport frame
            Color(0.05, 0.05, 0.08, 1.0)
            Rectangle(pos=self.pos, size=self.size)

            if self.texture is not None:
                tw, th = self.texture.size
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
                Rectangle(texture=self.texture, pos=(draw_x, draw_y), size=(draw_w, draw_h))

                # DRAW CENTRAL RED RETICLE CROSSHAIRS
                Color(1.0, 0.15, 0.15, 0.85) # Red reticle
                # Horizontal line
                Line(points=[self.x, center_y, self.x + self.width, center_y], width=1.2)
                # Vertical line
                Line(points=[center_x, self.y, center_x, self.y + self.height], width=1.2)
                # Target Center Circle
                Line(circle=(center_x, center_y, 12), width=1.5)
                Ellipse(pos=(center_x - 2, center_y - 2), size=(4, 4))
            else:
                Color(0.2, 0.2, 0.25, 1.0)
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

        # DOUBLE-CLICK PIXEL CENTERING: Moves clicked pixel to center of FOV
        if touch.is_double_tap and touch.button == 'left':
            app = App.get_running_app()
            if app and hasattr(app, 'jog_stage'):
                center_x = self.x + self.width / 2.0 + self.pan_x
                center_y = self.y + self.height / 2.0 + self.pan_y

                dx_px = touch.x - center_x
                dy_px = touch.y - center_y

                fov_w = getattr(app.focus_map, 'fov_width_mm', 0.60)
                fov_h = getattr(app.focus_map, 'fov_height_mm', 0.45)

                draw_w = self.width * self.zoom_level
                draw_h = self.height * self.zoom_level

                if draw_w > 0 and draw_h > 0:
                    dx_mm = (dx_px / draw_w) * fov_w
                    dy_mm = -(dy_px / draw_h) * fov_h
                    app.jog_stage(dx_mm, dy_mm, 0.0)
                    if hasattr(app, 'root_layout') and app.root_layout:
                        app.root_layout.status_msg = f"Viewport Centered on Clicked Pixel (dx: {dx_mm:+.3f} mm, dy: {dy_mm:+.3f} mm)"
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
    """UI Widget representing an optical channel row in the main window."""
    channel_name = StringProperty("")
    channel_spec_info = StringProperty("")
    ref_image_path = StringProperty("No Ref Image")
    is_active = BooleanProperty(False)
    btn_bg_color = ListProperty([0.2, 0.6, 0.8, 1.0])

    def open_ref_chooser(self):
        App.get_running_app().open_file_chooser_for_channel(self.channel_name)


class LumaViewProMainLayout(BoxLayout):
    """Main Application Root Window Layout."""
    status_msg = StringProperty("LumaViewPro Engine Ready")
    stage_coord_str = StringProperty("Stage: X=-49.50, Y=31.50, Z=0.000 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")
    current_well_display = StringProperty("A1")


class LumaViewProApp(App):

    root_layout = ObjectProperty(None)

    def select_optical_channel(self, channel_name: str):
        self.select_channel(channel_name)

    def set_exposure_ms(self, text_val: str):
        self.set_af_exposure(text_val)

    def set_gain_val(self, text_val: str):
        try:
            val = float(text_val)
            self.gain = val
            self.camera_gain = val
            self.root_layout.status_msg = f"Camera Gain set to {val:.1f}x"
        except ValueError:
            pass

    def set_jog_step_size(self, step_str: str):
        self.set_jog_step(step_str)

    def start_calibration_wizard(self):
        self.plate_calibration.start_wizard()
        self.is_light_on = True
        self.root_layout.is_light_on = True
        tx, ty = self.plate_calibration.get_step_nominal_stage_coords(0)
        self.jog_stage_to(tx, ty)
        step_info = self.plate_calibration.get_current_step_info()
        self.root_layout.status_msg = f"Calibration Step 1/6: Align reticle to {step_info['desc']}, then click Mark Target"

    def mark_calibration_step(self):
        if not self.plate_calibration.is_calibrating:
            self.start_calibration_wizard()
            return

        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)
        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS):
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)
            next_info = self.plate_calibration.get_current_step_info()
            self.root_layout.status_msg = f"Marked step. Moving to Step {next_idx + 1}/6: {next_info['desc']}"
        else:
            success = self.plate_calibration.compute_calibration()
            if success:
                self.root_layout.calibrated_status = f"Calibrated ({self.plate_calibration.rotation_angle_deg:.2f}°)"
                self.root_layout.status_msg = "All 6 calibration targets recorded! Matrix computed & applied."
            else:
                self.root_layout.status_msg = "Calibration computation failed. Check target alignments."

    def next_calibration_step(self):
        if not self.plate_calibration.is_calibrating:
            self.start_calibration_wizard()
            return
        if self.plate_calibration.current_step_index < len(self.plate_calibration.CALIBRATION_STEPS) - 1:
            self.plate_calibration.current_step_index += 1
            next_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(next_idx)
            self.jog_stage_to(tx, ty)
            next_info = self.plate_calibration.get_current_step_info()
            self.root_layout.status_msg = f"Moved to Step {next_idx + 1}/6: {next_info['desc']}"

    def previous_calibration_step(self):
        if not self.plate_calibration.is_calibrating:
            return
        if self.plate_calibration.current_step_index > 0:
            self.plate_calibration.current_step_index -= 1
            prev_idx = self.plate_calibration.current_step_index
            tx, ty = self.plate_calibration.get_step_nominal_stage_coords(prev_idx)
            self.jog_stage_to(tx, ty)
            prev_info = self.plate_calibration.get_current_step_info()
            self.root_layout.status_msg = f"Moved back to Step {prev_idx + 1}/6: {prev_info['desc']}"

    def reset_calibration(self):
        self.plate_calibration.reset_calibration()
        self.root_layout.calibrated_status = "Plate Uncalibrated"
        self.root_layout.status_msg = "Calibration reset."

    """Main Application Controller."""

    # Class-level Kivy Property Declarations (matching KV bindings)
    is_light_on = BooleanProperty(False)  # Sensor starts dark (black window) on launch
    active_channel = StringProperty("Brightfield")
    camera_gain = NumericProperty(1.0)
    gain = NumericProperty(1.0)
    camera_exposure_ms = NumericProperty(20.0)
    exposure_ms = NumericProperty(20.0)
    led_power = NumericProperty(100.0)
    jog_step_mm = NumericProperty(1.0)

    fov_manager = ObjectProperty(None)
    focus_map = ObjectProperty(None)
    plate_calibration = ObjectProperty(None)
    well_plate_format = ObjectProperty(None)
    channel_manager = ObjectProperty(None)

    def load_kv(self, filename=None):
        """Single-instance load guard: prevents Kivy App from loading lumaviewpro.kv a second time."""
        return None

    def build(self):
        self.title = "LumaViewPro Microscope Automation"
        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

        # Stage Coordinates (Well A1 center default)
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0

        # Hardware Driver Connection (Physical Hardware or Simulator Mock)
        self.camera = None
        self.init_hardware()

        self._updating_channel_ui = False
        self.navigator_popup = None

        self.root_layout = LumaViewProMainLayout()
        self.build_channel_controls()

        # Connect references to embedded plate canvas in right column
        if hasattr(self.root_layout, 'ids') and 'scaled_canvas_tab1' in self.root_layout.ids:
            self.root_layout.ids.scaled_canvas_tab1.set_references(
                self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
            )

        # Schedule live camera feed loop at 20 FPS
        Clock.schedule_interval(self.update_live_feed, 1.0 / 20.0)

        return self.root_layout

    def init_hardware(self):
        """Connects to real Etaluma hardware drivers if available on physical microscope PC."""
        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            if hasattr(cam, 'connect'):
                try:
                    cam.connect()
                except Exception as ce:
                    print(f"[LumaViewPro] Camera connect note: {ce}")
            if getattr(cam, 'is_connected', True):
                self.camera = cam
                print("[LumaViewPro] Successfully attached USB Camera Driver.")
        except Exception as e:
            print(f"[LumaViewPro] USB Camera Driver offline/unattached: {e}")
            self.camera = None

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            if hasattr(stg, 'connect'):
                try:
                    stg.connect()
                except Exception as me:
                    print(f"[LumaViewPro] MotorBoard connect note: {me}")
            if getattr(stg, 'is_connected', True):
                self.motor_stage = stg
                print("[LumaViewPro] Successfully attached Stage Motor Driver.")
        except Exception as e:
            print(f"[LumaViewPro] Stage Motor Driver offline/unattached: {e}")
            self.motor_stage = None

        try:
            import ledboard
            leds = ledboard.LEDBoard()
            if hasattr(leds, 'connect'):
                try:
                    leds.connect()
                except Exception as le:
                    print(f"[LumaViewPro] LEDBoard connect note: {le}")
            self.leds = leds
            print("[LumaViewPro] Successfully attached LED Board Driver.")
        except Exception as e:
            print(f"[LumaViewPro] LED Board Driver offline/unattached: {e}")
            self.leds = None
    def update_live_feed(self, dt):
        """Clock callback to render live camera frames and update active well indicator."""
        # Calculate active well name from stage coordinates
        try:
            r_idx = max(0, min(self.well_plate_format.rows - 1, int(round((31.5 - self.stage_y) / self.well_plate_format.pitch_y))))
            c_idx = max(0, min(self.well_plate_format.cols - 1, int(round((self.stage_x + 49.5) / self.well_plate_format.pitch_x))))
            well_str = self.well_plate_format.get_well_name(r_idx, c_idx)
            if hasattr(self.root_layout, 'current_well_display'):
                self.root_layout.current_well_display = well_str
        except Exception:
            pass

        try:
            active_ch = self.channel_manager.get_active_channel_name()
            if self.camera is not None and hasattr(self.camera, 'get_frame'):
                if hasattr(self.camera, 'stage_x'):
                    self.camera.stage_x = self.stage_x
                    self.camera.stage_y = self.stage_y
                    self.camera.stage_z = self.stage_z
                    self.camera.active_channel = active_ch
                    self.camera.is_light_on = self.is_light_on
                    self.camera.led_power = self.led_power
                    self.camera.exposure_ms = self.exposure_ms
                    self.camera.gain = self.gain
                    self.camera.fov_size_mm = (self.focus_map.fov_width_mm, self.focus_map.fov_height_mm)
                    self.camera.num_wells = self.well_plate_format.num_wells

                raw_frame = self.camera.get_frame()
                if raw_frame is None:
                    raw_frame = image_utils.generate_disconnected_camera_frame()
            else:
                raw_frame = image_utils.generate_disconnected_camera_frame()
        except Exception as e:
            pass

    def render_frame_to_viewports(self, frame: np.ndarray):
        """Pushes current camera frame to active viewports and updates plate map reticle."""
        if frame is None or self.root_layout is None:
            return

        if hasattr(self.root_layout, 'ids') and 'main_camera_viewport' in self.root_layout.ids:
            self.root_layout.ids.main_camera_viewport.set_frame(frame)

        # Update plate canvas active FOV reticle
        if hasattr(self.root_layout, 'ids') and 'scaled_canvas_tab1' in self.root_layout.ids:
            self.root_layout.ids.scaled_canvas_tab1.redraw_canvas()

    def select_channel(self, channel_name: str):
        """
        MUTUAL EXCLUSION: Activates specified channel and turns illumination ON.
        Toggling the active channel a second time switches illumination OFF (dark frame).
        """
        if getattr(self, '_updating_channel_ui', False):
            return

        canonical = self.channel_manager.resolve_channel_name(channel_name)
        if self.active_channel == canonical and self.is_light_on:
            self.is_light_on = False
            self.channel_manager.set_channel_active(canonical, False)
            self.root_layout.status_msg = f"Illumination OFF ({canonical})"
        else:
            self.active_channel = canonical
            self.is_light_on = True
            self.channel_manager.set_channel_active(canonical, True)
            self.root_layout.status_msg = f"Active Channel: {canonical} (Light ON)"

        self.build_channel_controls()

    def set_active_channel(self, channel_name: str, active_bool: bool = True):
        """Alias for channel selection."""
        self.select_channel(channel_name)

    def toggle_light_state(self, is_on: bool = None):
        """Toggles camera illumination light ON or OFF."""
        if is_on is None:
            self.is_light_on = not self.is_light_on
        else:
            self.is_light_on = is_on
        state_str = "ON" if self.is_light_on else "OFF"
        self.root_layout.status_msg = f"Illumination Light set to {state_str}"
        self.build_channel_controls()

    def toggle_light(self):
        self.toggle_light_state()

    def set_led_power(self, power_str: str):
        try:
            val = float(power_str)
            self.led_power = max(0.0, min(100.0, val))
            self.root_layout.status_msg = f"LED Power set to {self.led_power:.0f}%"
        except ValueError:
            pass

    def jog_stage(self, dx: float, dy: float, dz: float):
        """Jogs stage coordinates along X, Y, or Z axis and commands physical motor stage."""
        if self.motor_stage is None and not hasattr(self.camera, 'stage_x'):
            self.root_layout.status_msg = "Microscope Disconnected - Hardware Motion Unavailable"
            return

        self.stage_x += dx
        self.stage_y += dy
        self.stage_z += dz
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"

        if self.motor_stage is not None:
            try:
                if hasattr(self.motor_stage, 'move_rel'):
                    self.motor_stage.move_rel(dx, dy, dz)
                elif hasattr(self.motor_stage, 'move_abs'):
                    self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[LumaViewPro Motor Exception] move_rel failed: {e}")

    def jog_z_stage(self, delta_um: float):
        """Jogs Z focus in micrometers."""
        self.jog_stage(0.0, 0.0, delta_um / 1000.0)

    def jog_z_focus(self, delta_um: float):
        """Alias for Z focus jog."""
        self.jog_z_stage(delta_um)

    def jog_z(self, delta_um: float):
        """Alias for Z focus jog."""
        self.jog_z_stage(delta_um)

    def jog_stage_to(self, x_mm: float, y_mm: float):
        """Moves stage directly to target X, Y coordinates and commands physical motor stage."""
        if self.motor_stage is None and not hasattr(self.camera, 'stage_x'):
            self.root_layout.status_msg = "Microscope Disconnected - Hardware Motion Unavailable"
            return

        self.stage_x = x_mm
        self.stage_y = y_mm
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"

        if self.motor_stage is not None:
            try:
                if hasattr(self.motor_stage, 'move_abs'):
                    self.motor_stage.move_abs(self.stage_x, self.stage_y, self.stage_z)
            except Exception as e:
                print(f"[LumaViewPro Motor Exception] move_abs failed: {e}")

    def move_stage_abs(self, x_mm: float, y_mm: float):
        """Alias for absolute stage motion."""
        self.jog_stage_to(x_mm, y_mm)

    def set_objective_preset(self, preset_name: str):
        """Updates active objective lens preset and recalculates FOV scaling and Nyquist steps."""
        self.focus_map.set_objective_preset(preset_name)
        self.fov_manager._rebuild_all_groups()
        self.root_layout.status_msg = f"Objective Preset set to {preset_name} (FOV: {self.focus_map.fov_width_mm:.2f} x {self.focus_map.fov_height_mm:.2f} mm)"
        if hasattr(self.root_layout, 'ids') and 'scaled_canvas_tab1' in self.root_layout.ids:
            self.root_layout.ids.scaled_canvas_tab1.redraw_canvas()

    def set_well_plate_format(self, format_str: str):
        """Updates well plate map layout (6, 12, 24, 48, 96, 384 wells)."""
        digits = re.findall(r'\d+', format_str)
        if not digits:
            return
        num = int(digits[0])
        if num in labware.WellPlateFormat.STANDARD_FORMATS:
            self.well_plate_format = labware.WellPlateFormat(num_wells=num)
            self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
            self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)
            
            if hasattr(self.root_layout, 'ids') and 'scaled_canvas_tab1' in self.root_layout.ids:
                self.root_layout.ids.scaled_canvas_tab1.set_references(
                    self.well_plate_format, self.plate_calibration, self.focus_map, self.fov_manager
                )
            self.root_layout.status_msg = f"Well Plate Map updated to {num}-Well Format"

    def change_well_plate_format(self, format_str: str):
        """Alias for well plate format change."""
        self.set_well_plate_format(format_str)

    def set_jog_step(self, step_str: str):
        """Updates XY stage jogging step size."""
        try:
            val = float(re.findall(r'[\d\.]+', step_str)[0])
            self.jog_step_mm = val
            self.root_layout.status_msg = f"XY Jog Step set to {val:.1f} mm"
        except Exception:
            pass

    def open_navigator_popup(self):
        """Single-instance popup launcher guard."""
        if self.navigator_popup is not None:
            try:
                self.navigator_popup.open()
                return
            except Exception:
                pass

        import plate_navigator
        self.navigator_popup = plate_navigator.attach_navigator_to_lumaviewpro(self)

    def open_plate_navigator(self):
        self.open_navigator_popup()

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
        try:
            val = float(exposure_str)
            self.camera_exposure_ms = val
            self.exposure_ms = val
            self.focus_map.set_af_camera_config(self.focus_map.af_binning, val)
            self.root_layout.status_msg = f"AF Exposure Time set to {val:.1f} ms"
        except ValueError:
            pass

    def set_af_search_range(self, range_str: str):
        try:
            val = float(range_str)
            self.focus_map.af_search_range_um = abs(val)
            self.root_layout.status_msg = f"AF Search Range set to +/- {val:.1f} um"
        except ValueError:
            pass

    def set_af_search_range_um(self, range_str: str):
        self.set_af_search_range(range_str)

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
            status = "Enabled" if enabled else "Disabled"
            self.root_layout.status_msg = f"Time-Lapse {status}: {cyc} cycles @ {inv_m} min interval"
        except Exception:
            pass

    def set_master_auto_point(self):
        sim_frame = image_utils.generate_simulated_frame(
            stage_z=self.stage_z,
            best_z=2.5,
            is_light_on=True
        )
        af_score = image_utils.compute_focus_score(sim_frame, method="laplacian")
        del sim_frame
        self.focus_map.set_master_auto_point(self.stage_x, self.stage_y, self.stage_z)
        self.root_layout.status_msg = f"Master Auto Point set (Red). Primary Channel: {self.focus_map.primary_af_channel}, Score: {af_score:.1f}"

    def test_autofocus_run(self):
        af_frame = image_utils.generate_simulated_frame(
            stage_z=self.stage_z,
            best_z=self.stage_z,
            is_light_on=True
        )
        score = image_utils.compute_focus_score(af_frame, method="laplacian")
        del af_frame
        self.root_layout.status_msg = f"[Test AF Run] Channel: {self.focus_map.primary_af_channel} | Score: {score:.2f} (Frame Discarded)"

    def set_chromatic_z_offset(self, channel_name: str):
        self.focus_map.set_chromatic_z_offset(channel_name, self.stage_z)
        delta_z = self.focus_map.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        self.root_layout.status_msg = f"Saved Chromatic Z-Offset for {channel_name}: {delta_z*1000.0:.2f} um relative to Primary AF"

    def apply_nyquist_calculation(self):
        dof_um, nyquist_step_um = self.focus_map.calculate_optical_nyquist()
        self.focus_map.apply_nyquist_to_af_config()
        self.root_layout.status_msg = f"Nyquist Step Applied: {nyquist_step_um:.3f} um (Depth of Field: {dof_um:.3f} um, Total Steps: {self.focus_map.af_num_steps})"

    def apply_grid_settings(self, rows_str: str, cols_str: str, single_af: bool = True):
        try:
            r = int(rows_str)
            c = int(cols_str)
            self.fov_manager.set_grid_dimensions(r, c, single_af=single_af)
            self.root_layout.status_msg = f"Tile Grid set to {r}x{c}"
        except ValueError:
            pass

    def global_sync_active_channels(self):
        active_list = self.channel_manager.get_active_channel_names()
        self.fov_manager.apply_channels_to_all_selected(active_list)
        self.root_layout.status_msg = f"Synced Channels ({', '.join(active_list)}) across selected wells"

    def generate_and_apply_protocol(self) -> dict:
        protocol = labware.ExperimentProtocol.generate_protocol(
            self.well_plate_format,
            self.plate_calibration,
            self.focus_map,
            self.fov_manager
        )
        print("[Protocol Generated]\n", json.dumps(protocol, indent=2))
        self.root_layout.status_msg = f"Hardware Protocol Saved! Selected Wells: {protocol['selected_wells_count']}"
        return protocol

    def capture_snapshot(self):
        active_ch = self.channel_manager.get_active_channel_name()
        frame = image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x,
            stage_y=self.stage_y,
            stage_z=self.stage_z,
            best_z=2.5,
            channel=active_ch,
            is_light_on=self.is_light_on,
            led_power=self.led_power,
            sensor_size=(512, 512),
            fov_size_mm=(self.focus_map.fov_width_mm, self.focus_map.fov_height_mm),
            num_wells=self.well_plate_format.num_wells
        )
        filename = f"snapshot_{active_ch}_{self.stage_x:.1f}_{self.stage_y:.1f}.png"
        image_utils.save_image(filename, frame)
        self.root_layout.status_msg = f"Snapshot saved to {filename}"

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
                if self.channel_manager.channels[channel_name].load_ref_image(selected):
                    self.root_layout.status_msg = f"Loaded shading ref for {channel_name}"
                    self.build_channel_controls()
            popup.dismiss()

        select_btn.bind(on_select=on_select)
        cancel_btn.bind(on_release=popup.dismiss)
        popup.open()


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

_nav_kv = os.path.join(script_dir, 'plate_navigator.kv')
if os.path.exists(_nav_kv) and _nav_kv not in Builder.files:
    Builder.load_file(_nav_kv)


if __name__ == '__main__':
    LumaViewProApp().run()
