"""
labware.py - Labware representations, well plate registration, focus map gradient engine,
tiled FOV management, autofocus camera binning, objective presets, motorised turret configuration,
and time-lapse cycle skip configuration for LumaViewPro.

Provides:
- Standard ANSI/SLAS multi-well plate specifications (6, 12, 24, 48, 96, 384 well plates)
- Top-Left A1 Orientation (Row letters A,B,C... down left; Column numbers 1,2,3... across top)
- Precision 6-Point Target Plate Registration Engine (2D Affine Transformation)
- Primary Autofocus Channel, Camera Binning & Per-Channel Z-Offset Architecture
- Objective Lens Presets (1.25x, 4x, 5x, 10x, 20x Air, 20x, 40x, 60x Oil, 63x, 100x Oil) with NA, Mag, and true FOV dimensions
- Autofocus Search Range (+/- um), Nyquist Step Calculator (DoF, NA, Wavelength, Immersion Index)
- Autofocus Skip Cycle Control & Time-Lapse Schedule Engine
- Tiled Field-Of-View (FOV) Matrix Generator (1x1 up to 12x12 tiles per well)
- Hardware Protocol Generator bridging software settings to underlying motion/camera layers.
"""

from __future__ import annotations
import math
import re
import numpy as np
from typing import Dict, Tuple, List, Optional, Set, Union


# Objective Lens Physical Specifications & Presets Dictionary
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
        "na": 0.16,
        "refractive_index": 1.0,  # Air
        "fov_width_mm": 2.40,
        "fov_height_mm": 1.80,
        "fov_w_mm": 2.40,
        "fov_h_mm": 1.80,
        "turret_slot": 2,
        "desc": "5x Wide-field Objective (Air, NA 0.16)"
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
        "fov_height_mm": 0.1425,
        "fov_w_mm": 0.19,
        "fov_h_mm": 0.1425,
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
        "desc": "100x High-NA Oil Objective (Oil, NA 1.40)"
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
        match = re.match(r"([A-Za-z]+)(\d+)", well_name)
        if not match:
            raise ValueError(f"Invalid well name: {well_name}")
        r_str, c_str = match.groups()
        row_idx = ord(r_str[0]) - ord('A')
        col_idx = int(c_str) - 1
        return row_idx, col_idx

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
        {"id": "ROW_END_3", "well": "Row 1 End", "position": "3 o'clock", "desc": "Last Well in Row 1 - 3 o'clock position"},
        {"id": "COL_END_12", "well": "Col 1 End", "position": "12 o'clock", "desc": "Last Well in Column 1 - 12 o'clock position"}
    ]

    def __init__(self, plate_format: WellPlateFormat):
        self.plate_format = plate_format
        self.transform_matrix: np.ndarray = np.eye(3)
        self.is_calibrated: bool = False
        self.recorded_points: Dict[str, Tuple[float, float]] = {}
        self.current_step_index: int = 0

        self.a1_center_stage: Optional[Tuple[float, float]] = (-49.5, 31.5)
        self.row_end_center_stage: Optional[Tuple[float, float]] = None
        self.col_end_center_stage: Optional[Tuple[float, float]] = None
        
        self.measured_well_radius_x: float = 0.0
        self.measured_well_radius_y: float = 0.0
        self.rotation_angle_deg: float = 0.0
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0

    def reset_calibration(self):
        self.transform_matrix = np.eye(3)
        self.is_calibrated = False
        self.recorded_points.clear()
        self.current_step_index = 0
        self.a1_center_stage = (-49.5, 31.5)
        self.rotation_angle_deg = 0.0
        self.scale_x = 1.0
        self.scale_y = 1.0

    def get_current_step_info(self) -> dict:
        if 0 <= self.current_step_index < len(self.CALIBRATION_STEPS):
            step_info = self.CALIBRATION_STEPS[self.current_step_index].copy()
            step_info["step_num"] = self.current_step_index + 1
            step_info["total_steps"] = len(self.CALIBRATION_STEPS)
            return step_info
        return {"id": "DONE", "well": "All", "position": "Done", "desc": "Calibration Complete", "step_num": 6, "total_steps": 6}

    def record_current_step_point(self, x_stage: float, y_stage: float) -> str:
        if self.current_step_index >= len(self.CALIBRATION_STEPS):
            return "Calibration sequence already completed."

        step_info = self.CALIBRATION_STEPS[self.current_step_index]
        step_id = step_info["id"]
        self.recorded_points[step_id] = (x_stage, y_stage)
        self.current_step_index += 1

        if self.current_step_index == len(self.CALIBRATION_STEPS):
            return "All 6 calibration points recorded! Click Finish Calibration to compute matrix."
        
        next_step = self.CALIBRATION_STEPS[self.current_step_index]
        return f"Point recorded. Next: {next_step['desc']}"

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
        self.a1_center_stage = (x_a1, y_a1)

        self.measured_well_radius_x = abs(pt_3[0] - pt_9[0]) / 2.0
        self.measured_well_radius_y = abs(pt_12[1] - pt_6[1]) / 2.0

        x_row_end = pt_row[0] - self.measured_well_radius_x
        y_row_end = pt_row[1]
        x_col_end = pt_col[0]
        y_col_end = pt_col[1] - self.measured_well_radius_y

        nom_row_dist = (self.plate_format.cols - 1) * self.plate_format.pitch_x
        nom_col_dist = (self.plate_format.rows - 1) * self.plate_format.pitch_y

        dx_row = x_row_end - x_a1
        dy_row = y_row_end - y_a1
        dist_row_measured = math.hypot(dx_row, dy_row)

        if nom_row_dist <= 0 or dist_row_measured <= 0:
            return False

        theta = math.atan2(dy_row, dx_row)
        self.rotation_angle_deg = math.degrees(theta)

        self.scale_x = dist_row_measured / nom_row_dist
        dx_col = x_col_end - x_a1
        dy_col = y_col_end - y_a1
        dist_col_measured = math.hypot(dx_col, dy_col)
        self.scale_y = dist_col_measured / nom_col_dist if nom_col_dist > 0 else self.scale_x

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        self.transform_matrix = np.array([
            [self.scale_x * cos_t, -self.scale_y * sin_t, x_a1],
            [self.scale_x * sin_t,  self.scale_y * cos_t, y_a1],
            [0.0,                   0.0,                  1.0]
        ], dtype=np.float64)

        self.is_calibrated = True
        return True

    def nominal_to_stage(self, x_nom: float, y_nom: float) -> Tuple[float, float]:
        if not self.is_calibrated:
            a1_x = self.a1_center_stage[0] if self.a1_center_stage else -49.5
            a1_y = self.a1_center_stage[1] if self.a1_center_stage else 31.5
            return (a1_x + x_nom, a1_y - y_nom)

        vec = np.array([x_nom, y_nom, 1.0], dtype=np.float64)
        stage_vec = self.transform_matrix @ vec
        return (float(stage_vec[0]), float(stage_vec[1]))

    def get_well_stage_coordinates(self, row_idx: int, col_idx: int) -> Tuple[float, float]:
        x_nom, y_nom = self.plate_format.get_nominal_position(row_idx, col_idx)
        return self.nominal_to_stage(x_nom, y_nom)


class FocusPoint:
    def __init__(self, point_type: str, label: str, x_stage: float, y_stage: float, z_primary: float = 0.0, is_autofocus: bool = True):
        self.point_type: str = point_type.upper()
        self.label: str = label
        self.x_stage: float = x_stage
        self.y_stage: float = y_stage
        self.z_primary: float = z_primary
        self.is_autofocus: bool = is_autofocus


class FocusMap:
    SUPPORTED_PRIMARY_CHANNELS = ["Brightfield", "Phase", "Darkfield", "Blue", "Green", "Red"]
    SUPPORTED_BINNING_MODES = ["1x1", "2x2", "4x4", "8x8"]

    def __init__(self):
        self.primary_af_channel: str = "Brightfield"
        self.af_binning: str = "2x2"
        self.af_exposure_ms: float = 20.0
        self.save_af_images: bool = False

        self.active_objective_preset: str = "20x Air"

        self.af_search_range_um: float = 50.0
        self.af_step_size_um: float = 2.0
        self.af_num_steps: int = 51
        self.af_cycle_interval: int = 1

        self.timelapse_enabled: bool = False
        self.timelapse_interval_mins: float = 10.0
        self.timelapse_total_cycles: int = 24

        self.lens_na: float = 0.45
        self.lens_mag: float = 20.0
        self.wavelength_nm: float = 525.0
        self.refractive_index: float = 1.0

        self.master_auto_point: Optional[FocusPoint] = None
        self.set_points: List[FocusPoint] = []
        self.channel_chromatic_offsets_mm: Dict[str, float] = {ch: 0.0 for ch in self.SUPPORTED_PRIMARY_CHANNELS}
        self.accepted_channels: Dict[str, bool] = {ch: True for ch in self.SUPPORTED_PRIMARY_CHANNELS}
        self.plane_params: Optional[Tuple[float, float, float]] = None

    @property
    def fov_width_mm(self) -> float:
        preset = OBJECTIVE_PRESETS.get(self.active_objective_preset, OBJECTIVE_PRESETS["20x Air"])
        return float(preset.get("fov_width_mm", preset.get("fov_w_mm", 0.60)))

    @property
    def fov_height_mm(self) -> float:
        preset = OBJECTIVE_PRESETS.get(self.active_objective_preset, OBJECTIVE_PRESETS["20x Air"])
        return float(preset.get("fov_height_mm", preset.get("fov_h_mm", 0.45)))

    @property
    def selected_objective(self) -> str:
        return self.active_objective_preset

    @selected_objective.setter
    def selected_objective(self, val: str):
        self.set_objective_preset(val)

    def set_objective_preset(self, preset_name: str):
        if preset_name in OBJECTIVE_PRESETS:
            self.active_objective_preset = preset_name
            cfg = OBJECTIVE_PRESETS[preset_name]
            self.lens_mag = float(cfg.get("mag", cfg.get("magnification", 20.0)))
            self.lens_na = float(cfg.get("na", 0.45))
            self.refractive_index = float(cfg.get("refractive_index", 1.0))
            self.apply_nyquist_to_af_config()

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
        if self.af_step_size_um > 0:
            self.af_num_steps = int((2.0 * self.af_search_range_um) / self.af_step_size_um) + 1

    def set_primary_af_channel(self, channel_name: str):
        if channel_name in self.SUPPORTED_PRIMARY_CHANNELS:
            self.primary_af_channel = channel_name

    def set_af_camera_config(self, binning: str, exposure_ms: float):
        if binning in self.SUPPORTED_BINNING_MODES:
            self.af_binning = binning
        self.af_exposure_ms = max(0.1, float(exposure_ms))

    def set_af_cycle_interval(self, val: int):
        self.af_cycle_interval = max(1, int(val))

    def set_time_lapse_config(self, enabled: bool, interval_mins: float, cycles: int):
        self.timelapse_enabled = enabled
        self.time_interval_minutes = max(0.1, float(interval_mins))
        self.total_cycles = max(1, int(cycles))

    def set_master_auto_point(self, x_stage: float, y_stage: float, z_primary: float):
        self.master_auto_point = FocusPoint('AUTO', 'Auto', x_stage, y_stage, z_primary, is_autofocus=True)
        self.fit_focus_plane()

    def add_set_point(self, x_stage: float, y_stage: float, z_primary: float, is_autofocus: bool = True) -> FocusPoint:
        label = f"Set {len(self.set_points) + 1}"
        pt = FocusPoint('SET', label, x_stage, y_stage, z_primary, is_autofocus=is_autofocus)
        self.set_points.append(pt)
        self.fit_focus_plane()
        return pt

    def set_chromatic_z_offset(self, channel_name: str, z_channel_val: float):
        if self.master_auto_point is not None:
            self.channel_chromatic_offsets_mm[channel_name] = z_channel_val - self.master_auto_point.z_primary
        else:
            self.channel_chromatic_offsets_mm[channel_name] = 0.0

    def get_all_active_points(self) -> List[FocusPoint]:
        pts = []
        if self.master_auto_point is not None:
            pts.append(self.master_auto_point)
        pts.extend(self.set_points)
        return pts

    def fit_focus_plane(self) -> bool:
        active_pts = self.get_all_active_points()
        if len(active_pts) < 3:
            self.plane_params = None
            return False

        coords = np.array([[pt.x_stage, pt.y_stage] for pt in active_pts], dtype=np.float64)
        z_vals = np.array([pt.z_primary for pt in active_pts], dtype=np.float64)

        X_mat = np.column_stack([coords[:, 0], coords[:, 1], np.ones(len(coords))])
        try:
            params, _, _, _ = np.linalg.lstsq(X_mat, z_vals, rcond=None)
            self.plane_params = (float(params[0]), float(params[1]), float(params[2]))
            return True
        except np.linalg.LinAlgError:
            self.plane_params = None
            return False

    def interpolate_z(self, x_stage: float, y_stage: float, channel_name: str = "Brightfield") -> float:
        chromatic_offset = self.channel_chromatic_offsets_mm.get(channel_name, 0.0)

        if self.plane_params is not None:
            a, b, c = self.plane_params
            z_primary_pred = a * x_stage + b * y_stage + c
            return float(z_primary_pred + chromatic_offset)

        active_pts = self.get_all_active_points()
        if len(active_pts) == 1:
            return float(active_pts[0].z_primary + chromatic_offset)
        elif len(active_pts) == 2:
            avg_z = (active_pts[0].z_primary + active_pts[1].z_primary) / 2.0
            return float(avg_z + chromatic_offset)

        return chromatic_offset


class TileSquareFOV:
    def __init__(self, tile_row: int, tile_col: int, offset_x_mm: float, offset_y_mm: float, parent_well_name: str):
        self.tile_row: int = tile_row
        self.tile_col: int = tile_col
        self.offset_x_mm: float = offset_x_mm
        self.offset_y_mm: float = offset_y_mm
        self.parent_well_name: str = parent_well_name
        self.tile_id: str = f"{parent_well_name}_T{tile_row+1}x{tile_col+1}"


class TiledFOVGroup:
    def __init__(self, well_name: str, row_idx: int, col_idx: int, grid_rows: int = 1, grid_cols: int = 1, fov_width_mm: float = 0.60, fov_height_mm: float = 0.45, overlap_percent: float = 10.0):
        self.well_name: str = well_name
        self.row_idx: int = row_idx
        self.col_idx: int = col_idx
        self.grid_rows: int = min(12, max(1, grid_rows))
        self.grid_cols: int = min(12, max(1, grid_cols))
        self.fov_width_mm: float = fov_width_mm
        self.fov_height_mm: float = fov_height_mm
        self.overlap_percent: float = overlap_percent
        self.single_af_per_tiled_area: bool = True
        self.active_channels: List[str] = ["Brightfield"]
        self.tiles: List[TileSquareFOV] = []
        self.generate_tiles()

    def generate_tiles(self):
        self.tiles.clear()
        step_x = self.fov_width_mm * (1.0 - self.overlap_percent / 100.0)
        step_y = self.fov_height_mm * (1.0 - self.overlap_percent / 100.0)

        start_x = -((self.grid_cols - 1) * step_x) / 2.0
        start_y = -((self.grid_rows - 1) * step_y) / 2.0

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                off_x = start_x + c * step_x
                off_y = start_y + r * step_y
                self.tiles.append(TileSquareFOV(r, c, off_x, off_y, self.well_name))


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
            group.single_af_per_tiled_area = self.default_single_af
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

    def apply_channels_to_all_selected(self, channels: List[str]):
        for group in self.tiled_groups.values():
            group.active_channels = list(channels)

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
            group.single_af_per_tiled_area = self.default_single_af
            self.tiled_groups[w_name] = group


class ExperimentProtocol:
    @staticmethod
    def generate_protocol(
        plate_format: WellPlateFormat,
        calibration: NavigatorPlateCalibration,
        focus_map: FocusMap,
        fov_manager: PlateFOVManager
    ) -> dict:
        protocol_wells = []
        accepted_channel_list = [ch for ch, is_acc in focus_map.accepted_channels.items() if is_acc]
        dof_um, nyquist_step_um = focus_map.calculate_optical_nyquist()

        for well_name, group in fov_manager.tiled_groups.items():
            base_x, base_y = calibration.get_well_stage_coordinates(group.row_idx, group.col_idx)
            group_center_z = focus_map.interpolate_z(base_x, base_y, focus_map.primary_af_channel)

            tiles_data = []
            for tile in group.tiles:
                tile_stage_x = base_x + tile.offset_x_mm
                tile_stage_y = base_y + tile.offset_y_mm
                
                channel_z_map = {}
                for ch in accepted_channel_list:
                    chromatic_z_off = focus_map.channel_chromatic_offsets_mm.get(ch, 0.0)
                    channel_z_map[ch] = float(group_center_z + chromatic_z_off)

                tiles_data.append({
                    "tile_id": tile.tile_id,
                    "tile_row": tile.tile_row,
                    "tile_col": tile.tile_col,
                    "stage_x_mm": float(tile_stage_x),
                    "stage_y_mm": float(tile_stage_y),
                    "channel_z_heights_mm": channel_z_map
                })

            protocol_wells.append({
                "well_name": well_name,
                "row_index": group.row_idx,
                "column_index": group.col_idx,
                "calibrated_well_center_x_mm": float(base_x),
                "calibrated_well_center_y_mm": float(base_y),
                "grid_dimensions": f"{group.grid_rows}x{group.grid_cols}",
                "single_af_per_tiled_area": group.single_af_per_tiled_area,
                "active_channels": accepted_channel_list,
                "tiles": tiles_data
            })

        return {
            "labware_format": f"{plate_format.num_wells}-Well Plate",
            "is_calibrated": calibration.is_calibrated,
            "calibration_rotation_deg": float(calibration.rotation_angle_deg),
            "selected_objective": focus_map.selected_objective,
            "fov_width_mm": float(focus_map.fov_width_mm),
            "fov_height_mm": float(focus_map.fov_height_mm),
            "autofocus_configuration": {
                "primary_autofocus_channel": focus_map.primary_af_channel,
                "camera_binning": focus_map.af_binning,
                "af_exposure_ms": float(focus_map.af_exposure_ms),
                "af_search_range_um": float(focus_map.af_search_range_um),
                "af_step_size_um": float(focus_map.af_step_size_um),
                "af_num_steps": int(focus_map.af_num_steps),
                "af_cycle_interval": int(focus_map.af_cycle_interval),
            },
            "total_selected_wells": len(protocol_wells),
            "wells": protocol_wells
        }
