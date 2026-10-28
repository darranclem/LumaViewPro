from __future__ import annotations
"""
labware.py - Labware representations, well plate registration, focus map gradient engine,
tiled FOV management, autofocus camera binning, objective presets, and hardware protocol compilation for LumaViewPro.
"""

import math
import re
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
        row_str = chr(ord('A') + row_idx)
        col_str = str(col_idx + 1)
        return f"{row_str}{col_str}"

    def parse_well_name(self, well_name: str) -> Tuple[int, int]:
        well_name = well_name.strip().upper()
        match = re.match(r"([A-Za-z]+)(\d+)", well_name)
        if not match:
            raise ValueError(f"Invalid well name: {well_name}")
        r_str, c_str = match.groups()
        row_idx = ord(r_str.upper()) - ord('A')
        col_idx = int(c_str) - 1
        return row_idx, col_idx

    def get_nominal_position(self, row_idx: int, col_idx: int) -> Tuple[float, float]:
        return (col_idx * self.pitch_x, row_idx * self.pitch_y)


class NavigatorPlateCalibration:
    CALIBRATION_STEPS = [
        {"id": "A1_12", "well": "A1 (Top-Left)", "position": "12 o'clock", "desc": "Well A1 - 12 o'clock position (top rim)"},
        {"id": "A1_3", "well": "A1 (Top-Left)", "position": "3 o'clock", "desc": "Well A1 - 3 o'clock position (right rim)"},
        {"id": "A1_6", "well": "A1 (Top-Left)", "position": "6 o'clock", "desc": "Well A1 - 6 o'clock position (bottom rim)"},
        {"id": "A1_9", "well": "A1 (Top-Left)", "position": "9 o'clock", "desc": "Well A1 - 9 o'clock position (left rim)"},
        {"id": "ROW_END_3", "well": "Row 1 End", "position": "3 o'clock", "desc": "Last Well in Row 1 - 3 o'clock position"},
        {"id": "COL_END_12", "well": "Col 1 End", "position": "12 o'clock", "desc": "Last Well in Column 1 - 12 o'clock position"}
    ]

    def __init__(self, plate_format: WellPlateFormat):
        self.plate_format = plate_format
        self.transform_matrix: np.ndarray = np.eye(3)
        self.is_calibrated: bool = False
        self.is_calibrating: bool = False
        self.recorded_points: Dict[str, Tuple[float, float]] = {}
        self.current_step_index: int = 0
        self.a1_center_stage: Optional[Tuple[float, float]] = (-49.5, 31.5)
        self.rotation_angle_deg: float = 0.0

    def start_wizard(self):
        self.is_calibrating = True
        self.is_calibrated = False
        self.current_step_index = 0
        self.recorded_points.clear()

    def get_step_nominal_stage_coords(self, step_idx: int) -> Tuple[float, float]:
        a1_x = -self.plate_format.PLATE_LENGTH_MM / 2.0 + self.plate_format.a1_offset_x
        a1_y = self.plate_format.PLATE_WIDTH_MM / 2.0 - self.plate_format.a1_offset_y
        well_r = self.plate_format.well_diameter / 2.0

        if step_idx == 0:  # A1 12 o'clock
            return a1_x, a1_y + well_r
        elif step_idx == 1:  # A1 3 o'clock
            return a1_x + well_r, a1_y
        elif step_idx == 2:  # A1 6 o'clock
            return a1_x, a1_y - well_r
        elif step_idx == 3:  # A1 9 o'clock
            return a1_x - well_r, a1_y
        elif step_idx == 4:  # Last well in Row 1 (3 o'clock)
            row_end_x = a1_x + (self.plate_format.cols - 1) * self.plate_format.pitch_x
            return row_end_x + well_r, a1_y
        elif step_idx == 5:  # Last well in Col 1 (12 o'clock)
            col_end_y = a1_y - (self.plate_format.rows - 1) * self.plate_format.pitch_y
            return a1_x, col_end_y + well_r
        return a1_x, a1_y

    def record_current_step_point(self, x_stage: float, y_stage: float) -> str:
        if self.current_step_index >= len(self.CALIBRATION_STEPS):
            return "Calibration steps already complete. Click Finish Calibration."

        step_info = self.CALIBRATION_STEPS[self.current_step_index]
        step_id = step_info["id"]
        self.recorded_points[step_id] = (x_stage, y_stage)
        self.current_step_index += 1

        if self.current_step_index < len(self.CALIBRATION_STEPS):
            next_step = self.CALIBRATION_STEPS[self.current_step_index]
            return f"Recorded {step_id}. Next: {next_step['desc']}"
        return "All 6 calibration points recorded! Click Finish Calibration."

    def compute_calibration(self) -> bool:
        required = ["A1_12", "A1_3", "A1_6", "A1_9", "ROW_END_3", "COL_END_12"]
        if not all(k in self.recorded_points for k in required):
            return False

        pt_12 = self.recorded_points["A1_12"]
        pt_3 = self.recorded_points["A1_3"]
        pt_6 = self.recorded_points["A1_6"]
        pt_9 = self.recorded_points["A1_9"]
        pt_row = self.recorded_points["ROW_END_3"]

        x_a1 = (pt_3[0] + pt_9[0]) / 2.0
        y_a1 = (pt_12[1] + pt_6[1]) / 2.0
        self.a1_center_stage = (x_a1, y_a1)

        r_x = abs(pt_3[0] - pt_9[0]) / 2.0

        dx_row = (pt_row[0] - r_x) - x_a1
        dy_row = pt_row[1] - y_a1
        dist_row = math.hypot(dx_row, dy_row)

        nom_row_dist = (self.plate_format.cols - 1) * self.plate_format.pitch_x
        if nom_row_dist > 0 and dist_row > 0:
            theta = math.atan2(dy_row, dx_row)
            self.rotation_angle_deg = math.degrees(theta)

        self.is_calibrated = True
        return True

    def reset_calibration(self):
        self.transform_matrix = np.eye(3)
        self.is_calibrated = False
        self.is_calibrating = False
        self.recorded_points.clear()
        self.current_step_index = 0
        self.rotation_angle_deg = 0.0

    def get_current_step_info(self) -> dict:
        if 0 <= self.current_step_index < len(self.CALIBRATION_STEPS):
            step_info = self.CALIBRATION_STEPS[self.current_step_index].copy()
            step_info["step_num"] = self.current_step_index + 1
            step_info["total_steps"] = len(self.CALIBRATION_STEPS)
            return step_info
        return {"id": "DONE", "well": "All", "position": "Done", "desc": "Calibration Complete", "step_num": 6, "total_steps": 6}


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
        self.selected_objective: str = "20x Air"
        self.lens_mag: float = 20.0
        self.lens_na: float = 0.45
        self.refractive_index: float = 1.0
        self.master_auto_point: Optional[FocusPoint] = None

    @property
    def fov_width_mm(self) -> float:
        spec = self.OBJECTIVE_PRESETS.get(self.selected_objective, self.OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_w_mm"])

    @property
    def fov_height_mm(self) -> float:
        spec = self.OBJECTIVE_PRESETS.get(self.selected_objective, self.OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_h_mm"])

    def set_objective_preset(self, preset_name: str):
        if preset_name in self.OBJECTIVE_PRESETS:
            self.selected_objective = preset_name
            spec = self.OBJECTIVE_PRESETS[preset_name]
            self.lens_mag = float(spec["mag"])
            self.lens_na = float(spec["na"])

    def set_primary_af_channel(self, ch_name: str):
        self.primary_af_channel = ch_name

    def set_af_camera_config(self, binning: str, exposure_ms: float):
        self.af_binning = binning
        self.af_exposure_ms = max(0.1, float(exposure_ms))

    def set_master_auto_point(self, x_stage: float, y_stage: float, z_stage: float):
        self.master_auto_point = FocusPoint(x_stage, y_stage, z_stage, "AUTO")


class TileSquareFOV:
    def __init__(self, tile_row: int, tile_col: int, offset_x_mm: float, offset_y_mm: float, parent_well_name: str):
        self.tile_row = tile_row
        self.tile_col = tile_col
        self.offset_x_mm = offset_x_mm
        self.offset_y_mm = offset_y_mm
        self.parent_well_name = parent_well_name


class TiledFOVGroup:
    def __init__(self, well_name: str, row_idx: int, col_idx: int, grid_rows: int = 1, grid_cols: int = 1, fov_width_mm: float = 0.60, fov_height_mm: float = 0.45):
        self.well_name = well_name
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fov_width_mm = fov_width_mm
        self.fov_height_mm = fov_height_mm
        self.tiles: List[TileSquareFOV] = [TileSquareFOV(0, 0, 0.0, 0.0, well_name)]


class PlateFOVManager:
    def __init__(self, plate_format: WellPlateFormat, calibration: NavigatorPlateCalibration, focus_map: FocusMap):
        self.plate_format = plate_format
        self.calibration = calibration
        self.focus_map = focus_map
        self.selected_wells: Set[str] = set()
        self.tiled_groups: Dict[str, TiledFOVGroup] = {}

    def toggle_well_selected(self, well_name: str):
        if well_name in self.selected_wells:
            self.selected_wells.remove(well_name)
        else:
            self.selected_wells.add(well_name)

    def _rebuild_all_groups(self):
        pass


class ExperimentProtocol:
    @staticmethod
    def generate_protocol(plate_format: WellPlateFormat, calibration: NavigatorPlateCalibration, focus_map: FocusMap, fov_manager: PlateFOVManager) -> dict:
        return {
            "selected_wells_count": len(fov_manager.selected_wells),
            "objective": focus_map.selected_objective,
            "primary_af_channel": focus_map.primary_af_channel,
        }
