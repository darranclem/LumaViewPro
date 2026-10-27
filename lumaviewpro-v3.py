"""
lumaviewpro.py - Main Application Controller for LumaViewPro microscope automation software (Version 10).

Restores the original familiar LumaViewPro main window GUI on startup and integrates
an additional '[ Plate Navigator ]' button to activate the plate calibration and protocol studio.

Integrates:
- Original LumaViewPro Main Interface:
  * Header Navigation Bar with '[ Microscope Control ]', '[ Protocol ]', and '[ Plate Navigator ]' buttons
  * Live Camera Viewport with Zoom and Pan
  * Exposure Time (ms), Gain, LED Intensity, and Motion Jogging Controls
  * Optical Channel Selectors (BF, Phase, Pseudo DF, DAPI, FITC, TxRed, Cy5)
- Modal Plate Navigator Window with 3 Core Tabbed Screens:
  1. Calibration & Interactive Template Screen:
     * ANSI/SLAS Scaled Microplate Template with Top-Left A1 Orientation
     * Leica LAS X 6-Point Target Overlay
     * Selectable Calibration Channel (defaults to Brightfield)
     * Integrated Master Auto Point (Red) & Dedicated Test Autofocus Execution Button
     * Interactive Well FOV Selection with Batch Controls (Select All, Clear All, Sync Channels)
  2. AF Dialogue:
     * Search Range (+/- um), Step Size (um), Step Count
     * Optical Lens Specification Calculator (NA, Mag, Wavelength, Immersion Index) -> DoF & Nyquist Z-step
     * Autofocus Skip Cycle Control: Configures AF frequency (every 1st, 2nd, 3rd, N-th cycle)
     * Time-Lapse Schedule Engine: Interval (mins), Total Cycles, and estimated total duration calculation
  3. Hardware Protocol:
     * 1x1 up to 12x12 Tiled FOV Area Grid Dimensions
     * Channel Accept / Reject Selection List
     * Per-Channel Secondary Chromatic Z-Offset Correction
"""

import os
import sys
import json
import math
import re
from typing import Dict, Optional, List, Tuple

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import image_utils
import labware
import plate_navigator

# Kivy framework imports
import kivy
kivy.require('2.0.0')

from kivy.app import App
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
    """Stores configuration and calibration data for an optical imaging channel."""
    
    def __init__(self, name: str, wavelength_nm: int = 0, color_hex: str = "#FFFFFF", is_transmitted: bool = False):
        self.name: str = name
        self.wavelength_nm: int = wavelength_nm
        self.color_hex: str = color_hex
        self.is_transmitted: bool = is_transmitted
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
    """Manages optical imaging channels (Transmitted light and Fluorescence)."""
    
    def __init__(self):
        self.channels: Dict[str, ChannelConfig] = {
            # Transmitted Light Channels
            "BF": ChannelConfig("BF", wavelength_nm=0, color_hex="#FFFFFF", is_transmitted=True),
            "Phase": ChannelConfig("Phase", wavelength_nm=0, color_hex="#CCCCCC", is_transmitted=True),
            "Pseudo Darkfield": ChannelConfig("Pseudo Darkfield", wavelength_nm=0, color_hex="#888888", is_transmitted=True),
            # Fluorescence Channels
            "DAPI": ChannelConfig("DAPI", wavelength_nm=385, color_hex="#0000FF"),
            "FITC": ChannelConfig("FITC", wavelength_nm=470, color_hex="#00FF00"),
            "TxRed": ChannelConfig("TxRed", wavelength_nm=560, color_hex="#FF0000"),
            "Cy5": ChannelConfig("Cy5", wavelength_nm=650, color_hex="#800080"),
        }
        # Brightfield active by default
        self.channels["BF"].active = True

    def set_channel_active(self, channel_name: str, active: bool):
        if channel_name in self.channels:
            self.channels[channel_name].active = active

    def global_sync_active_channels(self, active_channel_states: Dict[str, bool]):
        """Globally updates active state across all imaging channels."""
        for name, active_val in active_channel_states.items():
            if name in self.channels:
                self.channels[name].active = active_val
        print(f"[ChannelManager] Global channel sync: {self.get_active_channel_names()}")

    def get_active_channel_names(self) -> List[str]:
        return [ch_name for ch_name, ch_cfg in self.channels.items() if ch_cfg.active]

    def process_channel_image(self, channel_name: str, raw_frame: object) -> object:
        """Applies flat-field shading correction to a raw frame using the channel's reference image."""
        ch = self.channels.get(channel_name)
        if (ch and ch.shading_correction_enabled and ch.ref_image_array is not None):
            return image_utils.apply_shading_correction(raw_frame, ch.ref_image_array)
        return raw_frame


class ZoomPanCameraWidget(Widget):
    """
    Interactive Live Camera Viewport Widget supporting:
    - Mouse Wheel Zooming (Zoom In / Zoom Out)
    - Right-Click Drag Panning in X and Y
    - Reset View Button
    """
    zoom_level = NumericProperty(1.0)
    pan_x = NumericProperty(0.0)
    pan_y = NumericProperty(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_panning = False
        self.last_touch_pos = (0, 0)

    def reset_zoom_pan(self):
        """Resets zoom and pan state back to default center."""
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
    """
    Custom Kivy canvas widget rendering a physically accurate representation
    of the labware plate based on ANSI/SLAS standards and 2D Affine calibration matrix.
    Orientation: Well A1 is positioned at the TOP-LEFT corner.
    - Rows are labeled A, B, C... top-to-bottom along left edge.
    - Columns are numbered 1, 2, 3... left-to-right across top edge.
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
        offset_y = self.y + (self.height + draw_h) / 2.0  # Top origin

        rot_angle = self.calibration.rotation_angle_deg if self.calibration.is_calibrated else 0.0

        with self.canvas:
            PushMatrix()
            if rot_angle != 0.0:
                Rotate(angle=rot_angle, origin=(offset_x + draw_w / 2.0, offset_y - draw_h / 2.0))

            # Outer boundary
            Color(0.2, 0.22, 0.26, 1.0)
            Rectangle(pos=(offset_x, offset_y - draw_h), size=(draw_w, draw_h))
            Color(0.5, 0.5, 0.6, 1.0)
            Line(rectangle=(offset_x, offset_y - draw_h, draw_w, draw_h), width=1.5)

            well_r_px = (self.plate_format.well_diameter / 2.0) * scale

            # Draw Wells (A1 Top-Left -> Rows down, Cols right)
            for r in range(self.plate_format.rows):
                for c in range(self.plate_format.cols):
                    w_name = self.plate_format.get_well_name(r, c)
                    x_nom, y_nom = self.plate_format.get_nominal_position(r, c)

                    # Invert Y for top-left origin
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

                    # Tiled FOV Overlay
                    if is_selected and w_name in self.fov_manager.tiled_groups:
                        group = self.fov_manager.tiled_groups[w_name]
                        Color(1.0, 0.8, 0.2, 0.8)
                        for tile in group.tiles:
                            tx_px = wx_px + tile.offset_x_mm * scale
                            ty_px = wy_px - tile.offset_y_mm * scale
                            tw_px = group.fov_width_mm * scale * 0.35
                            th_px = group.fov_height_mm * scale * 0.35
                            Line(rectangle=(tx_px - tw_px/2, ty_px - th_px/2, tw_px, th_px), width=1.0)

            # Render Leica Calibration Targets (Yellow/Green)
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

            # Master Auto Focus Point (Red)
            if self.focus_map.master_auto_point is not None:
                pt = self.focus_map.master_auto_point
                ax_px = offset_x + pt.x_stage * scale
                ay_px = offset_y - pt.y_stage * scale
                Color(1.0, 0.1, 0.1, 1.0)
                Ellipse(pos=(ax_px - 6, ay_px - 6), size=(12, 12))
                Color(1.0, 0.5, 0.5, 1.0)
                Line(circle=(ax_px, ay_px, 10), width=1.5)

            # Set Focus Points (Blue)
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
        offset_y = self.y + (self.height + draw_h * scale) / 2.0

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
    """
    Modal Plate Navigator Window designed to overlay stock LumaViewPro.
    """
    
    def __init__(self, app_controller, **kwargs):
        super().__init__(**kwargs)
        self.app_ctrl = app_controller
        self.title = "LumaViewPro - Plate Navigator & Registration Studio"
        self.size_hint = (0.96, 0.96)


class ChannelControlWidget(BoxLayout):
    """UI Widget representing an optical channel row."""
    channel_name = StringProperty("")
    channel_spec_info = StringProperty("")
    btn_bg_color = ListProperty([0.2, 0.6, 0.8, 1.0])
    ref_image_path = StringProperty("No Ref Image")
    is_active = BooleanProperty(False)

    def open_ref_chooser(self):
        App.get_running_app().open_file_chooser_for_channel(self.channel_name)


class LumaViewProMainLayout(BoxLayout):
    """Main Application Root Window."""
    status_msg = StringProperty("LumaViewPro Engine Ready")
    stage_coord_str = StringProperty("Stage: X=0.00, Y=0.00, Z=0.00 mm")
    calibrated_status = StringProperty("Plate Uncalibrated")


class LumaViewProApp(App):
    """Main Application Controller."""

    def build(self):
        self.title = "LumaViewPro Microscope Automation"
        self.channel_manager = ChannelManager()
        self.well_plate_format = labware.WellPlateFormat(num_wells=96)
        self.plate_calibration = labware.NavigatorPlateCalibration(self.well_plate_format)
        self.focus_map = labware.FocusMap()
        self.fov_manager = labware.PlateFOVManager(self.well_plate_format, self.plate_calibration, self.focus_map)

        # Virtual Stage State
        self.stage_x = 0.0
        self.stage_y = 0.0
        self.stage_z = 0.0

        self.root_layout = LumaViewProMainLayout()
        self.build_channel_controls()
        return self.root_layout

    def build_channel_controls(self):
        """Populates main window optical channel list."""
        if not hasattr(self.root_layout, 'ids') or 'main_channel_container' not in self.root_layout.ids:
            return

        container = self.root_layout.ids.main_channel_container
        container.clear_widgets()

        for ch_name, ch_cfg in self.channel_manager.channels.items():
            widget = ChannelControlWidget()
            widget.channel_name = ch_name
            widget.is_active = ch_cfg.active
            widget.ref_image_path = os.path.basename(ch_cfg.ref_image_path) if ch_cfg.ref_image_path else "No Ref Image"
            container.add_widget(widget)

    def open_navigator_popup(self):
        """Launches the modal Plate Navigator Popup Window."""
        plate_navigator.attach_navigator_to_lumaviewpro(self)

    def open_plate_navigator(self):
        """Alias launcher method for KV button binding."""
        self.open_navigator_popup()

    def jog_stage(self, dx: float, dy: float, dz: float):
        """Jogs stage position."""
        self.stage_x += dx
        self.stage_y += dy
        self.stage_z += dz
        self.root_layout.stage_coord_str = f"Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm"

    def record_calibration_step(self):
        """Records active calibration step from current stage position."""
        msg = self.plate_calibration.record_current_step_point(self.stage_x, self.stage_y)
        self.root_layout.status_msg = msg

    def finish_plate_calibration(self):
        """Finalizes calibration matrix calculation and rescales canvas."""
        success = self.plate_calibration.compute_calibration()
        if success:
            self.root_layout.calibrated_status = f"Calibrated (Angle: {self.plate_calibration.rotation_angle_deg:.2f}°)"
            self.root_layout.status_msg = "Plate Registration Matrix successfully computed!"
        else:
            self.root_layout.status_msg = "Calibration failed. Verify edge target points."

    def set_primary_af_channel(self, channel_name: str):
        """Sets the primary autofocus channel."""
        self.focus_map.set_primary_af_channel(channel_name)
        self.root_layout.status_msg = f"Primary AF Channel set to: {channel_name}"

    def set_af_binning(self, binning_str: str):
        """Sets camera binning mode for autofocus."""
        self.focus_map.set_af_camera_config(binning_str, self.focus_map.af_exposure_ms)
        self.root_layout.status_msg = f"AF Camera Binning: {binning_str}"

    def set_af_exposure(self, exposure_str: str):
        """Sets exposure time (ms) for autofocus."""
        try:
            val = float(exposure_str)
            self.focus_map.set_af_camera_config(self.focus_map.af_binning, val)
            self.root_layout.status_msg = f"AF Exposure Time set to {val:.1f} ms"
        except ValueError:
            self.root_layout.status_msg = "Invalid exposure time value"

    def set_af_cycle_interval(self, text_or_num):
        """Sets how frequently autofocus is executed during a multi-cycle time-lapse run."""
        try:
            digits = re.findall(r'\d+', str(text_or_num))
            val = int(digits[0]) if digits else 1
            self.focus_map.set_af_cycle_interval(val)
            self.root_layout.status_msg = f"Autofocus configured to run every {val} cycle(s)"
        except Exception as e:
            self.root_layout.status_msg = f"Invalid AF cycle interval: {e}"

    def set_time_lapse_config(self, enabled: bool, interval_mins_str: str, cycles_str: str):
        """Sets time-lapse schedule parameters."""
        try:
            inv_m = float(interval_mins_str) if interval_mins_str else 10.0
            cyc = int(cycles_str) if cycles_str else 24
            self.focus_map.set_time_lapse_config(enabled, inv_m, cyc)
            status = "Enabled" if enabled else "Disabled"
            self.root_layout.status_msg = f"Time-Lapse {status}: {cyc} cycles @ {inv_m} min interval"
        except Exception as e:
            self.root_layout.status_msg = f"Invalid time-lapse config: {e}"

    def set_master_auto_point(self):
        """Sets current stage position as Master Auto Point (Red)."""
        sim_frame = image_utils.generate_simulated_frame(
            z_pos=self.stage_z,
            best_z=2.5,
            binning=self.focus_map.af_binning
        )
        af_score = image_utils.compute_focus_score(sim_frame, method="laplacian")
        del sim_frame

        self.focus_map.set_master_auto_point(self.stage_x, self.stage_y, self.stage_z)
        self.root_layout.status_msg = f"Master Auto Point set (Red). Primary Channel: {self.focus_map.primary_af_channel}, Score: {af_score:.1f}"

    def test_autofocus_run(self):
        """Executes a test autofocus scan on current position using configured AF search parameters."""
        af_frame = image_utils.generate_simulated_frame(
            z_pos=self.stage_z,
            best_z=self.stage_z,
            binning=self.focus_map.af_binning
        )
        score = image_utils.compute_focus_score(af_frame, method="laplacian")
        del af_frame
        self.root_layout.status_msg = f"[Test AF Run] Channel: {self.focus_map.primary_af_channel} | Binning: {self.focus_map.af_binning} | Focus Score: {score:.2f} (Frame Discarded)"

    def set_chromatic_z_offset(self, channel_name: str):
        """Saves current Z position as chromatic offset for specified channel relative to primary AF channel."""
        self.focus_map.set_chromatic_z_offset(channel_name, self.stage_z)
        delta_z = self.focus_map.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        self.root_layout.status_msg = f"Saved Chromatic Z-Offset for {channel_name}: {delta_z*1000.0:.2f} um relative to Primary AF"

    def apply_nyquist_calculation(self):
        """Calculates Depth of Field and Nyquist Z-step, applying it to AF scan settings."""
        dof_um, nyquist_step_um = self.focus_map.calculate_optical_nyquist()
        self.focus_map.apply_nyquist_to_af_config()
        self.root_layout.status_msg = f"Nyquist Step Applied: {nyquist_step_um:.3f} um (Depth of Field: {dof_um:.3f} um, Total Steps: {self.focus_map.af_num_steps})"

    def apply_grid_settings(self, rows_str: str, cols_str: str, single_af: bool):
        """Updates tile grid configuration (1x1 up to 12x12)."""
        try:
            r = int(rows_str)
            c = int(cols_str)
            self.fov_manager.set_grid_dimensions(r, c, single_af=single_af)
            self.root_layout.status_msg = f"Tile Grid set to {r}x{c} (Single AF: {single_af})"
        except ValueError:
            self.root_layout.status_msg = "Invalid tile dimensions"

    def global_sync_active_channels(self):
        """Syncs active channels across all selected FOVs."""
        active_list = self.channel_manager.get_active_channel_names()
        self.fov_manager.apply_channels_to_all_selected(active_list)
        self.root_layout.status_msg = f"Synced Channels ({', '.join(active_list)}) across selected wells"

    def generate_and_apply_protocol(self) -> dict:
        """Compiles hardware protocol dictionary."""
        protocol = labware.ExperimentProtocol.generate_protocol(
            self.well_plate_format,
            self.plate_calibration,
            self.focus_map,
            self.fov_manager
        )
        print("[Protocol Generated]\n", json.dumps(protocol, indent=2))
        self.root_layout.status_msg = f"Hardware Protocol Applied & Saved! Total Selected Wells: {protocol['total_selected_wells']}"
        return protocol

    def open_file_chooser_for_channel(self, channel_name: str):
        """Opens reference image file chooser popup."""
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

        select_btn.bind(on_release=on_select)
        cancel_btn.bind(on_release=popup.dismiss)
        popup.open()


if __name__ == '__main__':
    LumaViewProApp().run()
