"""
labware.py - Labware representations, well plate registration, focus map gradient engine,
tiled FOV management, autofocus camera binning, objective presets, and hardware protocol compilation for LumaViewPro.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, List, Optional, Set, Union


class WellPlateFormat:
    """Standard multi-well plate physical specifications (ANSI/SLAS standards)."""
    
    STANDARD_FORMATS = {
        6: {"rows": 2, "cols": 3, "pitch_x": 39.12, "pitch_y": 39.12, "well_diameter": 34.8, "a1_offset_x": 24.76, "a1_offset_y": 23.16},
        12: {"rows": 3, "cols": 4, "pitch_x": 26.0, "pitch_y": 26.0, "well_diameter": 22.1, "a1_offset_x": 24.9, "a1_offset_y": 16.7},
        24: {"rows": 4, "cols": 6, "pitch_x": 19.3, "pitch_y": 19.3, "well_diameter": 15.6, "a1_offset_x": 17.5, "a1_offset_y": 13.8},
        48: {"rows": 6, "cols": 8, "pitch_x": 13.0, "pitch_y": 13.0, "well_diameter": 11.0, "a1_offset_x": 18.4, "a1_offset_y": 10.1},
        96: {"rows": 8, "cols": 12, "pitch_x": 9.0, "pitch_y": 9.0, "well_diameter": 6.4, "a1_offset_x": 14.38, "a1_offset_y": 11.24},
        384: {"rows": 16, "cols": 24, "pitch_x": 4.5, "pitch_y": 4.5, "well_diameter": 3.8, "a1_offset_x": 12.13, "a1_offset_y": 8.99},
    }

    PLATE_LENGTH_MM = 127.76
    PLATE_WIDTH_MM = 85.48

    def __init__(self, num_wells: int = 96, custom_config: Optional[dict] = None):
        if custom_config:
            cfg = custom_config
            self.num_wells = cfg.get("num_wells", num_wells)
        elif num_wells in self.STANDARD_FORMATS:
            cfg = self.STANDARD_FORMATS[num_wells]
            self.num_wells = num_wells
        else:
            cfg = self.STANDARD_FORMATS[96]
            self.num_wells = 96

        self.rows: int = cfg["rows"]
        self.cols: int = cfg["cols"]
        self.pitch_x: float = cfg["pitch_x"]
        self.pitch_y: float = cfg["pitch_y"]
        self.well_diameter: float = cfg["well_diameter"]
        self.a1_offset_x: float = cfg["a1_offset_x"]
        self.a1_offset_y: float = cfg["a1_offset_y"]

    def get_well_name(self, row_idx: int, col_idx: int) -> str:
        """Converts 0-indexed (row, col) to well string like 'A1' or 'H12'."""
        if not (0 <= row_idx < self.rows and 0 <= col_idx < self.cols):
            raise IndexError(f"Well index ({row_idx}, {col_idx}) out of bounds")
        return f"{chr(ord('A') + row_idx)}{col_idx + 1}"

    def parse_well_name(self, well_name: str) -> Tuple[int, int]:
        """Parses well name string ('A1', 'H12') into 0-indexed (row_idx, col_idx)."""
        well_name = well_name.strip().upper()
        return ord(well_name[0]) - ord('A'), int(well_name[1:]) - 1

    def get_nominal_position(self, row_idx: int, col_idx: int) -> Tuple[float, float]:
        """Returns nominal (X, Y) relative position in mm where A1 center is (0.0, 0.0)."""
        return (col_idx * self.pitch_x, row_idx * self.pitch_y)


class NavigatorPlateCalibration:
    """Computes 2D Affine Transformation matrix mapping nominal well coords to stage motor coords."""

    CALIBRATION_STEPS = [
        {"id": "A1_12", "well": "A1 (Top-Left)", "position": "12 o'clock", "desc": "Well A1 - 12 o'clock position (top rim)"},
        {"id": "A1_3", "well": "A1 (Top-Left)", "position": "3 o'clock", "desc": "Well A1 - 3 o'clock position (right rim)"},
        {"id": "A1_6", "well": "A1 (Top-Left)", "position": "6 o'clock", "desc": "Well A1 - 6 o'clock position (bottom rim)"},
        {"id": "A1_9", "well": "A1 (Top-Left)", "position": "9 o'clock", "desc": "Well A1 - 9 o'clock position (left rim)"},
        {"id": "ROW_END_3", "well": "Row 1 End", "position": "3 o'clock", "desc": "Last Well in Row 1 - 3 o'clock position (right rim)"},
        {"id": "COL_END_12", "well": "Col 1 End", "position": "12 o'clock", "desc": "Last Well in Column 1 - 12 o'clock position (top rim)"}
    ]

    def __init__(self, plate_format: WellPlateFormat):
        self.plate_format = plate_format
        self.transform_matrix: np.ndarray = np.eye(3)
        self.is_calibrated: bool = False
        self.is_calibrating: bool = False
        self.recorded_points: Dict[str, Tuple[float, float]] = {}
        self.current_step_index: int = 0
        self.rotation_angle_deg: float = 0.0
        self.a1_center_stage: Tuple[float, float] = (-49.5, 31.5)
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

    def reset_calibration(self):
        self.transform_matrix = np.eye(3)
        self.is_calibrated = False
        self.is_calibrating = False
        self.recorded_points.clear()
        self.current_step_index = 0
        self.rotation_angle_deg = 0.0
        self.a1_center_stage = (-49.5, 31.5)
        self.scale_x = 1.0
        self.scale_y = 1.0

    def get_current_step_info(self) -> dict:
        if 0 <= self.current_step_index < len(self.CALIBRATION_STEPS):
            info = self.CALIBRATION_STEPS[self.current_step_index].copy()
            info["step_num"] = self.current_step_index + 1
            info["total_steps"] = len(self.CALIBRATION_STEPS)
            return info
        return {"id": "DONE", "well": "All", "position": "Done", "desc": "Calibration Complete", "step_num": 6, "total_steps": 6}

    def get_step_nominal_stage_coords(self, step_idx: int) -> Tuple[float, float]:
        a1_x = -self.plate_format.PLATE_LENGTH_MM / 2.0 + self.plate_format.a1_offset_x
        a1_y = self.plate_format.PLATE_WIDTH_MM / 2.0 - self.plate_format.a1_offset_y
        well_r = self.plate_format.well_diameter / 2.0

        if step_idx == 0:    # A1_12 (top rim)
            return (a1_x, a1_y + well_r)
        elif step_idx == 1:  # A1_3 (right rim)
            return (a1_x + well_r, a1_y)
        elif step_idx == 2:  # A1_6 (bottom rim)
            return (a1_x, a1_y - well_r)
        elif step_idx == 3:  # A1_9 (left rim)
            return (a1_x - well_r, a1_y)
        elif step_idx == 4:  # ROW_END_3 (last well in Row 1, right rim)
            x_row = a1_x + (self.plate_format.cols - 1) * self.plate_format.pitch_x + well_r
            return (x_row, a1_y)
        elif step_idx == 5:  # COL_END_12 (last well in Col 1, top rim)
            y_col = a1_y - (self.plate_format.rows - 1) * self.plate_format.pitch_y + well_r
            return (a1_x, y_col)

        return (a1_x, a1_y)

    def record_current_step_point(self, x_stage: float, y_stage: float) -> str:
        if self.current_step_index >= len(self.CALIBRATION_STEPS):
            return "Calibration steps already complete. Click Finish Calibration."

        step_info = self.CALIBRATION_STEPS[self.current_step_index]
        step_id = step_info["id"]
        self.recorded_points[step_id] = (float(x_stage), float(y_stage))
        self.current_step_index += 1

        if self.current_step_index < len(self.CALIBRATION_STEPS):
            next_step = self.CALIBRATION_STEPS[self.current_step_index]
            return f"Recorded {step_id}. Next: {next_step['desc']}"
        return "All 6 calibration points recorded! Computing transformation matrix..."

    def compute_calibration(self) -> bool:
        required = ["A1_12", "A1_3", "A1_6", "A1_9", "ROW_END_3", "COL_END_12"]
        if not all(k in self.recorded_points for k in required):
            self.is_calibrated = True
            self.is_calibrating = False
            return True

        p12 = self.recorded_points["A1_12"]
        p3 = self.recorded_points["A1_3"]
        p6 = self.recorded_points["A1_6"]
        p9 = self.recorded_points["A1_9"]
        p_row = self.recorded_points["ROW_END_3"]
        p_col = self.recorded_points["COL_END_12"]

        a1_center_x = (p3[0] + p9[0]) / 2.0
        a1_center_y = (p12[1] + p6[1]) / 2.0
        self.a1_center_stage = (a1_center_x, a1_center_y)

        meas_r_x = abs(p3[0] - p9[0]) / 2.0
        meas_r_y = abs(p12[1] - p6[1]) / 2.0

        row_end_center_x = p_row[0] - meas_r_x
        row_end_center_y = p_row[1]

        col_end_center_x = p_col[0]
        col_end_center_y = p_col[1] - meas_r_y

        dx_row = row_end_center_x - a1_center_x
        dy_row = row_end_center_y - a1_center_y
        nom_dist_x = (self.plate_format.cols - 1) * self.plate_format.pitch_x

        if nom_dist_x > 0:
            meas_dist_x = math.hypot(dx_row, dy_row)
            self.scale_x = meas_dist_x / nom_dist_x
            theta = math.atan2(dy_row, dx_row)
            self.rotation_angle_deg = math.degrees(theta)
        else:
            theta = 0.0
            self.rotation_angle_deg = 0.0
            self.scale_x = 1.0

        nom_dist_y = (self.plate_format.rows - 1) * self.plate_format.pitch_y
        dx_col = col_end_center_x - a1_center_x
        dy_col = col_end_center_y - a1_center_y
        if nom_dist_y > 0:
            meas_dist_y = math.hypot(dx_col, dy_col)
            self.scale_y = meas_dist_y / nom_dist_y
        else:
            self.scale_y = 1.0

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        self.transform_matrix = np.array([
            [self.scale_x * cos_t, -self.scale_y * sin_t, a1_center_x],
            [self.scale_x * sin_t,  self.scale_y * cos_t, a1_center_y],
            [0.0,                   0.0,                  1.0]
        ], dtype=np.float64)

        self.is_calibrated = True
        self.is_calibrating = False
        return True

    def nominal_to_stage(self, x_nom: float, y_nom: float) -> Tuple[float, float]:
        if not self.is_calibrated:
            a1_x, a1_y = self.a1_center_stage
            return (a1_x + x_nom, a1_y - y_nom)

        vec = np.array([x_nom, y_nom, 1.0], dtype=np.float64)
        st_vec = self.transform_matrix @ vec
        return (float(st_vec[0]), float(st_vec[1]))

    def get_well_stage_coordinates(self, row_idx: int, col_idx: int) -> Tuple[float, float]:
        x_nom, y_nom = self.plate_format.get_nominal_position(row_idx, col_idx)
        return self.nominal_to_stage(x_nom, y_nom)


class FocusPoint:
    def __init__(self, x_stage: float, y_stage: float, z_stage: float, point_type: str = "SET"):
        self.x_stage = x_stage
        self.y_stage = y_stage
        self.z_stage = z_stage
        self.point_type = point_type


class FocusMap:
    OBJECTIVE_PRESETS = {
        "1.25x": {"mag": 1.25, "na": 0.04, "fov_w_mm": 9.60, "fov_h_mm": 7.20, "refractive_index": 1.0},
        "4x": {"mag": 4.0, "na": 0.13, "fov_w_mm": 3.00, "fov_h_mm": 2.25, "refractive_index": 1.0},
        "5x": {"mag": 5.0, "na": 0.15, "fov_w_mm": 2.40, "fov_h_mm": 1.80, "refractive_index": 1.0},
        "10x": {"mag": 10.0, "na": 0.30, "fov_w_mm": 1.20, "fov_h_mm": 0.90, "refractive_index": 1.0},
        "20x Air": {"mag": 20.0, "na": 0.45, "fov_w_mm": 0.60, "fov_h_mm": 0.45, "refractive_index": 1.0},
        "20x": {"mag": 20.0, "na": 0.45, "fov_w_mm": 0.60, "fov_h_mm": 0.45, "refractive_index": 1.0},
        "40x": {"mag": 40.0, "na": 0.60, "fov_w_mm": 0.30, "fov_h_mm": 0.225, "refractive_index": 1.0},
        "60x Oil": {"mag": 60.0, "na": 1.35, "fov_w_mm": 0.20, "fov_h_mm": 0.15, "refractive_index": 1.518},
        "63x": {"mag": 63.0, "na": 1.40, "fov_w_mm": 0.19, "fov_h_mm": 0.1425, "refractive_index": 1.518},
        "100x Oil": {"mag": 100.0, "na": 1.40, "fov_w_mm": 0.12, "fov_h_mm": 0.09, "refractive_index": 1.518},
    }

    def __init__(self):
        self.primary_af_channel: str = "Brightfield"
        self.af_binning: str = "2x2"
        self.af_exposure_ms: float = 20.0
        self.af_search_range_um: float = 50.0
        self.af_step_size_um: float = 2.0
        self.af_num_steps: int = 50
        self.af_cycle_interval: int = 1

        self.time_lapse_enabled: bool = False
        self.time_lapse_interval_mins: float = 10.0
        self.time_lapse_cycles: int = 24

        self.selected_objective: str = "20x Air"
        self.lens_mag: float = 20.0
        self.lens_na: float = 0.45
        self.wavelength_nm: float = 525.0
        self.refractive_index: float = 1.0

        self.turret_hardware_type: str = "Manual Nosepiece"
        self.per_channel_turret_slots: Dict[str, int] = {
            "Brightfield": 1, "Phase": 2, "Darkfield": 3, "Blue": 4, "Green": 5, "Red": 6
        }

        self.channel_chromatic_offsets_mm: Dict[str, float] = {
            "Brightfield": 0.0, "Phase": 0.0, "Darkfield": 0.0, "Blue": 0.0, "Green": 0.0, "Red": 0.0
        }
        self.accepted_channels: Dict[str, bool] = {
            "Brightfield": True, "Phase": True, "Darkfield": True, "Blue": True, "Green": True, "Red": True
        }

        self.master_auto_point: Optional[FocusPoint] = None
        self.set_points: List[FocusPoint] = []

    @property
    def fov_width_mm(self) -> float:
        spec = self.OBJECTIVE_PRESETS.get(self.selected_objective, self.OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_w_mm"])

    @property
    def fov_height_mm(self) -> float:
        spec = self.OBJECTIVE_PRESETS.get(self.selected_objective, self.OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_h_mm"])

    def set_primary_af_channel(self, ch_name: str):
        self.primary_af_channel = ch_name

    def set_objective_preset(self, preset_name: str):
        if preset_name in self.OBJECTIVE_PRESETS:
            self.selected_objective = preset_name
            cfg = self.OBJECTIVE_PRESETS[preset_name]
            self.lens_mag = cfg["mag"]
            self.lens_na = cfg["na"]
            self.refractive_index = cfg["refractive_index"]


class PlateFOVManager:
    def __init__(self, plate_format: WellPlateFormat, calibration: NavigatorPlateCalibration, focus_map: FocusMap):
        self.plate_format = plate_format
        self.calibration = calibration
        self.focus_map = focus_map
        self.selected_wells: Set[str] = set()

    def select_all_wells(self):
        for r in range(self.plate_format.rows):
            for c in range(self.plate_format.cols):
                self.selected_wells.add(self.plate_format.get_well_name(r, c))

    def clear_all_wells(self):
        self.selected_wells.clear()

    def toggle_well_selected(self, well_name: str):
        if well_name in self.selected_wells:
            self.selected_wells.remove(well_name)
        else:
            self.selected_wells.add(well_name)
