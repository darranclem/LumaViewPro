"""
labware.py - Labware representations, well plate registration, focus map gradient engine,
tiled FOV management, autofocus camera binning, objective presets, and hardware protocol compilation for LumaViewPro.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, List, Optional, Set, Union


# Objective Lens Physical Specifications & Presets
OBJECTIVE_PRESETS = {
    "1.25x": {
        "mag": 1.25,
        "magnification": 1.25,
        "na": 0.04,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 9.60,
        "fov_height_mm": 7.20,
        "fov_w_mm": 9.60,
        "fov_h_mm": 7.20,
        "turret_slot": 1,
        "desc": "1.25x Ultra-Wide Objective (Air, NA 0.04)"
    },
    "4x": {
        "mag": 4.0,
        "magnification": 4.0,
        "na": 0.13,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 3.00,
        "fov_height_mm": 2.25,
        "fov_w_mm": 3.00,
        "fov_h_mm": 2.25,
        "turret_slot": 1,
        "desc": "4x Scanner Objective (Air, NA 0.13)"
    },
    "5x": {
        "mag": 5.0,
        "magnification": 5.0,
        "na": 0.15,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 2.40,
        "fov_height_mm": 1.80,
        "fov_w_mm": 2.40,
        "fov_h_mm": 1.80,
        "turret_slot": 2,
        "desc": "5x Objective (Air, NA 0.15)"
    },
    "10x": {
        "mag": 10.0,
        "magnification": 10.0,
        "na": 0.30,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 1.20,
        "fov_height_mm": 0.90,
        "fov_w_mm": 1.20,
        "fov_h_mm": 0.90,
        "turret_slot": 2,
        "desc": "10x Plan Fluor Objective (Air, NA 0.30)"
    },
    "20x Air": {
        "mag": 20.0,
        "magnification": 20.0,
        "na": 0.45,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 0.60,
        "fov_height_mm": 0.45,
        "fov_w_mm": 0.60,
        "fov_h_mm": 0.45,
        "turret_slot": 3,
        "desc": "20x LWD Plan Fluor Objective (Air, NA 0.45)"
    },
    "20x": {
        "mag": 20.0,
        "magnification": 20.0,
        "na": 0.45,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 0.60,
        "fov_height_mm": 0.45,
        "fov_w_mm": 0.60,
        "fov_h_mm": 0.45,
        "turret_slot": 3,
        "desc": "20x Objective (Air, NA 0.45)"
    },
    "40x": {
        "mag": 40.0,
        "magnification": 40.0,
        "na": 0.60,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 0.30,
        "fov_height_mm": 0.225,
        "fov_w_mm": 0.30,
        "fov_h_mm": 0.225,
        "turret_slot": 4,
        "desc": "40x High-Mag Objective (Air, NA 0.60)"
    },
    "60x Oil": {
        "mag": 60.0,
        "magnification": 60.0,
        "na": 1.40,
        "refractive_index": 1.518,  # Oil Immersion
        "fov_width_mm": 0.20,
        "fov_height_mm": 0.15,
        "fov_w_mm": 0.20,
        "fov_h_mm": 0.15,
        "turret_slot": 5,
        "desc": "60x Oil Immersion Objective (Oil, NA 1.40)"
    },
    "63x": {
        "mag": 63.0,
        "magnification": 63.0,
        "na": 1.40,
        "refractive_index": 1.518,  # Oil Immersion
        "fov_width_mm": 0.19,
        "fov_height_mm": 0.143,
        "fov_w_mm": 0.19,
        "fov_h_mm": 0.143,
        "turret_slot": 5,
        "desc": "63x Oil Immersion Objective (Oil, NA 1.40)"
    },
    "100x Oil": {
        "mag": 100.0,
        "magnification": 100.0,
        "na": 1.40,
        "refractive_index": 1.518,  # Oil Immersion
        "fov_width_mm": 0.12,
        "fov_height_mm": 0.09,
        "fov_w_mm": 0.12,
        "fov_h_mm": 0.09,
        "turret_slot": 6,
        "desc": "100x Objective (Oil, NA 1.40)"
    }
}


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

    def reset_calibration(self):
        self.transform_matrix = np.eye(3)
        self.is_calibrated = False
        self.is_calibrating = False
        self.recorded_points.clear()
        self.current_step_index = 0
        self.rotation_angle_deg = 0.0

    def get_step_nominal_stage_coords(self, step_index: int) -> Tuple[float, float]:
        """Returns nominal stage motor coordinates (X, Y) for a calibration step index (0..5)."""
        well_r_mm = self.plate_format.well_diameter / 2.0
        a1_x = -49.50
        a1_y = 31.50

        if step_index == 0:    # A1 12 o'clock (top rim)
            return (a1_x, a1_y + well_r_mm)
        elif step_index == 1:  # A1 3 o'clock (right rim)
            return (a1_x + well_r_mm, a1_y)
        elif step_index == 2:  # A1 6 o'clock (bottom rim)
            return (a1_x, a1_y - well_r_mm)
        elif step_index == 3:  # A1 9 o'clock (left rim)
            return (a1_x - well_r_mm, a1_y)
        elif step_index == 4:  # ROW_END_3 (Row 1 last well 3 o'clock right rim)
            row_end_x = a1_x + (self.plate_format.cols - 1) * self.plate_format.pitch_x + well_r_mm
            return (row_end_x, a1_y)
        elif step_index == 5:  # COL_END_12 (Col 1 last well 12 o'clock top rim)
            col_end_y = a1_y - (self.plate_format.rows - 1) * self.plate_format.pitch_y + well_r_mm
            return (a1_x, col_end_y)

        return (a1_x, a1_y)

    def get_current_step_info(self) -> dict:
        if 0 <= self.current_step_index < len(self.CALIBRATION_STEPS):
            step_info = self.CALIBRATION_STEPS[self.current_step_index].copy()
            step_info["step_num"] = self.current_step_index + 1
            step_info["total_steps"] = len(self.CALIBRATION_STEPS)
            return step_info
        return {"id": "DONE", "well": "All", "position": "Done", "desc": "Calibration Complete", "step_num": 6, "total_steps": 6}

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
        pt_col = self.recorded_points["COL_END_12"]

        x_a1 = (pt_3[0] + pt_9[0]) / 2.0
        y_a1 = (pt_12[1] + pt_6[1]) / 2.0

        measured_r_x = abs(pt_3[0] - pt_9[0]) / 2.0
        measured_r_y = abs(pt_12[1] - pt_6[1]) / 2.0

        x_row_end = pt_row[0] - measured_r_x
        y_row_end = pt_row[1]

        nom_row_dist = (self.plate_format.cols - 1) * self.plate_format.pitch_x
        dx_row = x_row_end - x_a1
        dy_row = y_row_end - y_a1
        dist_row_measured = math.hypot(dx_row, dy_row)

        if nom_row_dist <= 0 or dist_row_measured <= 0:
            return False

        theta = math.atan2(dy_row, dx_row)
        self.rotation_angle_deg = math.degrees(theta)

        scale_x = dist_row_measured / nom_row_dist
        scale_y = scale_x

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        self.transform_matrix = np.array([
            [scale_x * cos_t, -scale_y * sin_t, x_a1],
            [scale_x * sin_t,  scale_y * cos_t, y_a1],
            [0.0,              0.0,             1.0]
        ], dtype=np.float64)

        self.is_calibrated = True
        self.is_calibrating = False
        return True

    def nominal_to_stage(self, x_nom: float, y_nom: float) -> Tuple[float, float]:
        if not self.is_calibrated:
            a1_x = -49.5
            a1_y = 31.5
            return (a1_x + x_nom, a1_y - y_nom)

        vec = np.array([x_nom, y_nom, 1.0], dtype=np.float64)
        stage_vec = self.transform_matrix @ vec
        return (float(stage_vec[0]), float(stage_vec[1]))

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
        spec = OBJECTIVE_PRESETS.get(self.selected_objective, OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_w_mm"])

    @property
    def fov_height_mm(self) -> float:
        spec = OBJECTIVE_PRESETS.get(self.selected_objective, OBJECTIVE_PRESETS["20x Air"])
        return float(spec["fov_h_mm"])

    def set_objective_preset(self, preset_name: str):
        if preset_name in OBJECTIVE_PRESETS:
            self.selected_objective = preset_name
            spec = OBJECTIVE_PRESETS[preset_name]
            self.lens_mag = float(spec["mag"])
            self.lens_na = float(spec["na"])
            self.refractive_index = float(spec["refractive_index"])

    def set_primary_af_channel(self, channel_name: str):
        self.primary_af_channel = channel_name

    def set_af_camera_config(self, binning: str, exposure_ms: float):
        self.af_binning = binning
        self.af_exposure_ms = max(0.1, float(exposure_ms))

    def set_af_cycle_interval(self, interval_cycles: int):
        self.af_cycle_interval = max(1, int(interval_cycles))

    def set_time_lapse_config(self, enabled: bool, interval_mins: float, cycles: int):
        self.time_lapse_enabled = enabled
        self.time_interval_minutes = max(0.1, float(interval_mins))
        self.total_cycles = max(1, int(cycles))

    def set_master_auto_point(self, x_stage: float, y_stage: float, z_stage: float):
        self.master_auto_point = FocusPoint(x_stage, y_stage, z_stage, "AUTO")

    def set_chromatic_z_offset(self, channel_name: str, z_val: float):
        self.channel_chromatic_offsets_mm[channel_name] = z_val

    def calculate_optical_nyquist(self) -> Tuple[float, float]:
        na = max(0.01, float(self.lens_na))
        n = max(1.0, float(self.refractive_index))
        lambda_um = float(self.wavelength_nm) / 1000.0

        dof_um = (lambda_um * n) / (na ** 2)
        nyquist_step_um = dof_um / 2.0
        return float(dof_um), float(nyquist_step_um)

    def apply_nyquist_to_af_config(self):
        _, nyquist_step_um = self.calculate_optical_nyquist()
        self.af_step_size_um = round(nyquist_step_um, 3)


class TileSquareFOV:
    def __init__(self, tile_row: int, tile_col: int, offset_x_mm: float, offset_y_mm: float, parent_well_name: str):
        self.tile_row: int = tile_row
        self.tile_col: int = tile_col
        self.offset_x_mm: float = offset_x_mm
        self.offset_y_mm: float = offset_y_mm
        self.parent_well_name: str = parent_well_name


class TiledFOVGroup:
    def __init__(self, well_name: str, row_idx: int, col_idx: int, grid_rows: int = 1, grid_cols: int = 1, fov_width_mm: float = 0.60, fov_height_mm: float = 0.45, overlap_percent: float = 10.0):
        self.well_name: str = well_name
        self.row_idx: int = row_idx
        self.col_idx: int = col_idx
        self.grid_rows: int = min(12, max(1, grid_rows))
        self.grid_cols: int = min(12, max(1, grid_cols))
        self.fov_width_mm: float = fov_width_mm
        self.fov_height_mm: float = fov_height_mm
        self.tiles: List[TileSquareFOV] = []
        self._build_tiles()

    def _build_tiles(self):
        self.tiles.clear()
        step_x = self.fov_width_mm * 0.90
        step_y = self.fov_height_mm * 0.90
        start_x = -((self.grid_cols - 1) * step_x) / 2.0
        start_y = ((self.grid_rows - 1) * step_y) / 2.0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                ox = start_x + c * step_x
                oy = start_y - r * step_y
                self.tiles.append(TileSquareFOV(r, c, ox, oy, self.well_name))


class PlateFOVManager:
    def __init__(self, plate_format: WellPlateFormat, calibration: NavigatorPlateCalibration, focus_map: FocusMap):
        self.plate_format = plate_format
        self.calibration = calibration
        self.focus_map = focus_map
        
        self.selected_wells: Set[str] = set()
        self.tiled_groups: Dict[str, TiledFOVGroup] = {}
        
        self.default_grid_rows: int = 1
        self.default_grid_cols: int = 1
        self.default_single_af: bool = True

    def set_grid_dimensions(self, rows: int, cols: int, single_af: bool = True):
        self.default_grid_rows = min(12, max(1, rows))
        self.default_grid_cols = min(12, max(1, cols))
        self.default_single_af = single_af
        self._rebuild_all_groups()

    def toggle_well_selected(self, well_name: str) -> bool:
        if well_name in self.selected_wells:
            self.selected_wells.remove(well_name)
            if well_name in self.tiled_groups:
                del self.tiled_groups[well_name]
            return False
        else:
            self.selected_wells.add(well_name)
            r_idx, c_idx = self.plate_format.parse_well_name(well_name)
            group = TiledFOVGroup(
                well_name, r_idx, c_idx,
                grid_rows=self.default_grid_rows,
                grid_cols=self.default_grid_cols,
                fov_width_mm=self.focus_map.fov_width_mm,
                fov_height_mm=self.focus_map.fov_height_mm
            )
            self.tiled_groups[well_name] = group
            return True

    def select_all_wells(self):
        for r in range(self.plate_format.rows):
            for c in range(self.plate_format.cols):
                w_name = self.plate_format.get_well_name(r, c)
                if w_name not in self.selected_wells:
                    self.toggle_well_selected(w_name)

    def clear_all_wells(self):
        self.selected_wells.clear()
        self.tiled_groups.clear()

    def _rebuild_all_groups(self):
        for w_name in list(self.selected_wells):
            r_idx, c_idx = self.plate_format.parse_well_name(w_name)
            group = TiledFOVGroup(
                w_name, r_idx, c_idx,
                grid_rows=self.default_grid_rows,
                grid_cols=self.default_grid_cols,
                fov_width_mm=self.focus_map.fov_width_mm,
                fov_height_mm=self.focus_map.fov_height_mm
            )
            self.tiled_groups[w_name] = group


class ExperimentProtocol:
    @staticmethod
    def generate_protocol(
        plate_format: WellPlateFormat,
        calibration: NavigatorPlateCalibration,
        focus_map: FocusMap,
        fov_manager: PlateFOVManager
    ) -> dict:
        return {
            "num_wells": plate_format.num_wells,
            "is_calibrated": calibration.is_calibrated,
            "selected_objective": focus_map.selected_objective,
            "primary_af_channel": focus_map.primary_af_channel,
            "selected_wells_count": len(fov_manager.selected_wells),
            "selected_wells": list(fov_manager.selected_wells)
        }
