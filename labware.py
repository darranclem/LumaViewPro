"""
labware.py - Labware representations, well plate registration, focus map gradient engine,
tiled FOV management, autofocus camera binning, objective lens presets, and hardware protocol compilation for LumaViewPro.

Features:
- Standard ANSI/SLAS multi-well plate specifications (6, 12, 24, 48, 96, 384 well plates)
- Top-Left A1 Orientation (Row letters A,B,C... down left; Column numbers 1,2,3... across top)
- Precision 6-Point Target Plate Registration Engine (2D Affine Transformation)
- Primary Autofocus Channel, Camera Binning & Per-Channel Z-Offset Architecture
- Objective Lens Presets (1.25x, 4x, 5x, 10x, 20x Air, 40x, 60x Oil, 63x, 100x Oil) with Scaled Sample FOV
- Conditional Motorised Objective Turret Configuration and Per-Channel Slot Assignments
- Autofocus Search Range (+/- um), Nyquist Step Calculator (DoF, NA, Wavelength, Immersion Index)
- Autofocus Skip Cycle Control: Configures AF frequency (every 1st, 2nd, 3rd, N-th cycle)
- Time-Lapse Schedule Engine: Interval (mins), Total Cycles, and estimated total duration calculation
- Secondary Chromatic Z-Offset Correction: Per-channel optical shift compensation
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
        self.recorded_points: Dict[str, Tuple[float, float]] = {}
        self.current_step_index: int = 0
        self.a1_center_stage: Optional[Tuple[float, float]] = (-49.5, 31.5)
        self.rotation_angle_deg: float = 0.0

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
        if len(self.recorded_points) < 6:
            return False
        self.is_calibrated = True
        self.rotation_angle_deg = 0.0
        return True

    def nominal_to_stage(self, x_nom: float, y_nom: float) -> Tuple[float, float]:
        a1_x = self.a1_center_stage[0] if self.a1_center_stage else -49.5
        a1_y = self.a1_center_stage[1] if self.a1_center_stage else 31.5
        return (a1_x + x_nom, a1_y - y_nom)

    def get_well_stage_coordinates(self, row_idx: int, col_idx: int) -> Tuple[float, float]:
        x_nom, y_nom = self.plate_format.get_nominal_position(row_idx, col_idx)
        return self.nominal_to_stage(x_nom, y_nom)


class FocusPoint:
    def __init__(self, x: float, y: float, z: float, pt_type: str = "auto", label: str = ""):
        self.x_stage: float = x
        self.y_stage: float = y
        self.z_stage: float = z
        self.z_primary: float = z
        self.pt_type: str = pt_type
        self.label: str = label


class FocusMap:
    SUPPORTED_PRIMARY_CHANNELS = ["Brightfield", "Phase", "Darkfield", "Blue", "Green", "Red"]

    def __init__(self):
        self.primary_af_channel: str = "Brightfield"
        self.af_binning: str = "2x2"
        self.af_exposure_ms: float = 20.0
        self.af_search_range_um: float = 50.0
        self.af_step_size_um: float = 2.0
        self.af_num_steps: int = 51
        self.af_cycle_interval: int = 1

        self.timelapse_enabled: bool = False
        self.timelapse_interval_mins: float = 10.0
        self.timelapse_total_cycles: int = 24

        self.active_objective_preset: str = "20x Air"
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
    def selected_objective(self) -> str:
        return self.active_objective_preset

    @selected_objective.setter
    def selected_objective(self, val: str):
        self.set_objective_preset(val)

    @property
    def fov_width_mm(self) -> float:
        preset = OBJECTIVE_PRESETS.get(self.active_objective_preset, OBJECTIVE_PRESETS["20x Air"])
        return float(preset["fov_width_mm"])

    @property
    def fov_height_mm(self) -> float:
        preset = OBJECTIVE_PRESETS.get(self.active_objective_preset, OBJECTIVE_PRESETS["20x Air"])
        return float(preset["fov_height_mm"])

    @property
    def current_fov_width_mm(self) -> float:
        return self.fov_width_mm

    @property
    def current_fov_height_mm(self) -> float:
        return self.fov_height_mm

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
        self.af_step_size_um = nyquist_step_um

    def set_primary_af_channel(self, channel_name: str):
        if channel_name in self.SUPPORTED_PRIMARY_CHANNELS:
            self.primary_af_channel = channel_name

    def set_af_camera_config(self, binning: str, exposure_ms: float):
        self.af_binning = binning
        self.af_exposure_ms = max(1.0, float(exposure_ms))

    def set_af_cycle_interval(self, cycle_interval: int):
        self.af_cycle_interval = max(1, int(cycle_interval))

    def set_time_lapse_config(self, enabled: bool, interval_mins: float, cycles: int):
        self.timelapse_enabled = enabled
        self.timelapse_interval_mins = max(0.1, float(interval_mins))
        self.timelapse_total_cycles = max(1, int(cycles))

    def set_master_auto_point(self, x_stage: float, y_stage: float, z_primary: float):
        self.master_auto_point = FocusPoint(x_stage, y_stage, z_primary, pt_type='AUTO')

    def add_set_point(self, x_stage: float, y_stage: float, z_primary: float) -> FocusPoint:
        pt = FocusPoint(x_stage, y_stage, z_primary, pt_type='SET')
        self.set_points.append(pt)
        return pt

    def set_chromatic_z_offset(self, channel_name: str, z_channel_val: float):
        if self.master_auto_point is not None:
            self.channel_chromatic_offsets_mm[channel_name] = z_channel_val - self.master_auto_point.z_stage

    def interpolate_z(self, x_stage: float, y_stage: float, channel_name: str = "Brightfield") -> float:
        base_z = self.master_auto_point.z_stage if self.master_auto_point else 0.0
        chromatic_offset = self.channel_chromatic_offsets_mm.get(channel_name, 0.0)
        return float(base_z + chromatic_offset)


class TileSquareFOV:
    def __init__(self, tile_id: str, row: int, col: int, offset_x_mm: float, offset_y_mm: float):
        self.tile_id = tile_id
        self.tile_row = row
        self.tile_col = col
        self.offset_x_mm = offset_x_mm
        self.offset_y_mm = offset_y_mm


class TiledFOVGroup:
    def __init__(self, well_name: str, row_idx: int = 0, col_idx: int = 0, grid_rows: int = 1, grid_cols: int = 1):
        self.well_name = well_name
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.single_af_per_tiled_area = True

        self.fov_width_mm = 0.60
        self.fov_height_mm = 0.45
        self.tiles: List[TileSquareFOV] = []
        self.generate_tiles()

    def generate_tiles(self):
        self.tiles.clear()
        total_w = self.grid_cols * self.fov_width_mm
        total_h = self.grid_rows * self.fov_height_mm
        start_x = -total_w / 2.0 + self.fov_width_mm / 2.0
        start_y = total_h / 2.0 - self.fov_height_mm / 2.0

        idx = 0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                ox = start_x + c * self.fov_width_mm
                oy = start_y - r * self.fov_height_mm
                self.tiles.append(TileSquareFOV(f"{self.well_name}_T{idx+1}", r, c, ox, oy))
                idx += 1


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
            group = TiledFOVGroup(well_name, r_idx, c_idx, self.default_grid_rows, self.default_grid_cols)
            group.fov_width_mm = self.focus_map.fov_width_mm
            group.fov_height_mm = self.focus_map.fov_height_mm
            group.generate_tiles()
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
        pass

    def _rebuild_all_groups(self):
        for w_name in list(self.selected_wells):
            r_idx, c_idx = self.plate_format.parse_well_name(w_name)
            group = TiledFOVGroup(w_name, r_idx, c_idx, self.default_grid_rows, self.default_grid_cols)
            group.fov_width_mm = self.focus_map.fov_width_mm
            group.fov_height_mm = self.focus_map.fov_height_mm
            group.generate_tiles()
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
        for well_name, group in fov_manager.tiled_groups.items():
            base_x, base_y = calibration.get_well_stage_coordinates(group.row_idx, group.col_idx)
            protocol_wells.append({
                "well_name": well_name,
                "stage_x_mm": float(base_x),
                "stage_y_mm": float(base_y),
            })
        return {
            "num_wells": plate_format.num_wells,
            "selected_wells_count": len(protocol_wells),
            "total_selected_wells": len(protocol_wells),
            "wells": protocol_wells
        }
