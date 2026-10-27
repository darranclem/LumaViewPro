"""
image_utils.py - Fast, cached image processing, shading correction, focus metrics,
and optically accurate synthetic camera frame generation for LumaViewPro.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, List

# Pre-computed cell colony coordinate cache for fast live simulation
_WELL_CELLS_CACHE: Dict[int, Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
_WELL_GRID_CACHE: Dict[int, Tuple[List[Tuple[float, float, int, int, str]], Tuple[int, int, float, float, float, float, float]]] = {}
_DARK_NOISE_FRAME: Optional[np.ndarray] = None


def load_image(filepath: str) -> Optional[np.ndarray]:
    if not os.path.exists(filepath):
        return None
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"[Error] Failed to load image {filepath}: {e}")
        return None


def save_image(filepath: str, img: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        return cv2.imwrite(filepath, img)
    except Exception as e:
        print(f"[Error] Failed to save image {filepath}: {e}")
        return False


def apply_shading_correction(
    raw_img: np.ndarray,
    ref_img: np.ndarray,
    dark_img: Optional[np.ndarray] = None,
    clip_output: bool = True
) -> np.ndarray:
    if raw_img is None or ref_img is None:
        return raw_img

    if raw_img.shape != ref_img.shape:
        return raw_img

    orig_dtype = raw_img.dtype
    raw_f = raw_img.astype(np.float32)
    ref_f = ref_img.astype(np.float32)

    if dark_img is not None and dark_img.shape == raw_img.shape:
        dark_f = dark_img.astype(np.float32)
        raw_f = np.maximum(0.0, raw_f - dark_f)
        ref_f = np.maximum(1.0e-5, ref_f - dark_f)
    else:
        ref_f = np.maximum(1.0e-5, ref_f)

    ref_mean = float(np.mean(ref_f))
    if ref_mean < 1.0e-5:
        ref_mean = 1.0

    corrected = (raw_f / ref_f) * ref_mean

    if not clip_output:
        return corrected

    if orig_dtype == np.uint8:
        return np.clip(corrected, 0, 255).astype(np.uint8)
    elif orig_dtype == np.uint16:
        return np.clip(corrected, 0, 65535).astype(np.uint16)
    return corrected.astype(orig_dtype)


def compute_focus_score(img: np.ndarray, method: str = "laplacian") -> float:
    if img is None or img.size == 0:
        return 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray_f = gray.astype(np.float32)

    if method.lower() == "laplacian":
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        return float(np.var(lap))
    elif method.lower() == "tenengrad":
        sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(sobel_x**2 + sobel_y**2))
    elif method.lower() == "brenner":
        diff_x = gray_f[:, 2:] - gray_f[:, :-2]
        return float(np.mean(diff_x**2))
    else:
        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        return float(np.var(lap))


def _get_cached_plate_grid(num_wells: int = 96):
    if num_wells in _WELL_GRID_CACHE:
        return _WELL_GRID_CACHE[num_wells]

    well_formats = {
        6: (2, 3, 39.12, 39.12, 34.8, -39.12, 19.56),
        12: (3, 4, 26.0, 26.0, 22.1, -39.0, 26.0),
        24: (4, 6, 19.3, 19.3, 15.6, -48.25, 28.95),
        48: (6, 8, 13.0, 13.0, 11.0, -45.5, 32.5),
        96: (8, 12, 9.0, 9.0, 6.4, -49.5, 31.5),
        384: (16, 24, 4.5, 4.5, 3.8, -51.75, 33.75)
    }
    params = well_formats.get(num_wells, well_formats[96])
    rows, cols, pitch_x, pitch_y, well_diameter, a1_x, a1_y = params
    well_radius_mm = well_diameter / 2.0

    wells = []
    cell_dict = {}

    np.random.seed(42 + num_wells)
    for r in range(rows):
        wy = a1_y - r * pitch_y
        for c in range(cols):
            wx = a1_x + c * pitch_x
            w_name = f"{chr(65+r)}{c+1}"
            wells.append((wx, wy, r, c, w_name))

            # Cellular colony & organelle distribution inside each well
            num_cells = 300
            angles = np.random.uniform(0, 2 * np.pi, num_cells)
            radii = np.random.uniform(0, well_radius_mm * 0.92, num_cells)
            cx = wx + radii * np.cos(angles)
            cy = wy + radii * np.sin(angles)
            cr = np.random.uniform(0.018, 0.040, num_cells)  # Cell radius in mm

            cell_dict[(r, c)] = (
                cx.astype(np.float32),
                cy.astype(np.float32),
                cr.astype(np.float32)
            )

    _WELL_GRID_CACHE[num_wells] = (wells, params)
    _WELL_CELLS_CACHE[num_wells] = cell_dict
    return wells, params


def generate_disconnected_camera_frame(sensor_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generates a high-contrast CAMERA DISCONNECTED warning frame when USB camera is not attached."""
    w, h = sensor_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (15, 15), (w - 15, h - 15), (35, 30, 80), -1)
    cv2.rectangle(frame, (15, 15), (w - 15, h - 15), (80, 80, 220), 2)
    cv2.putText(frame, "CAMERA DISCONNECTED", (w // 2 - 160, h // 2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
    cv2.putText(frame, "Microscope Stage & Focus Active", (w // 2 - 140, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 220), 1)
    return frame


def generate_well_plate_cell_frame(
    stage_x: float = -49.5,
    stage_y: float = 31.5,
    stage_z: float = 0.0,
    best_z: float = 0.0,
    channel: str = "Brightfield",
    exposure_ms: float = 20.0,
    gain: float = 1.0,
    led_power: float = 100.0,
    is_light_on: bool = False,
    fov_size_mm: Tuple[float, float] = (0.60, 0.45),
    num_wells: int = 96,
    sensor_size: Tuple[int, int] = (512, 512),
    **kwargs
) -> np.ndarray:
    """
    Optically realistic synthetic camera frame generator:
    - Light OFF: Dark sensor Poisson noise
    - Light ON:
      1. Inter-well plastic deck surface with structural molding grid lines
      2. Outer plate skirt boundaries & metal stage frame edges when approaching physical bounds
      3. High-contrast double-ring well wall rims
      4. Rich internal cellular colony structures (cytoplasm, nuclei, membrane borders)
      5. Z-focus Gaussian defocus blur
    """
    global _DARK_NOISE_FRAME
    w, h = sensor_size

    # 1. Dark Launch / Light OFF State (returns dark sensor noise)
    if not is_light_on or led_power <= 0:
        if _DARK_NOISE_FRAME is None or _DARK_NOISE_FRAME.shape != (h, w, 3):
            np.random.seed(0)
            _DARK_NOISE_FRAME = np.random.poisson(lam=2.5, size=(h, w, 3)).clip(0, 255).astype(np.uint8)
        return _DARK_NOISE_FRAME

    # 2. Light ON State: Calculate illumination brightness
    brightness = max(0.1, min(3.0, (exposure_ms / 20.0) * (led_power / 100.0) * gain))

    fov_w, fov_h = fov_size_mm
    fov_w = max(0.05, float(fov_w))
    fov_h = max(0.05, float(fov_h))

    px_per_mm_x = w / fov_w
    px_per_mm_y = h / fov_h

    fov_left = stage_x - fov_w / 2.0
    fov_right = stage_x + fov_w / 2.0
    fov_bottom = stage_y - fov_h / 2.0
    fov_top = stage_y + fov_h / 2.0

    wells, (rows, cols, pitch_x, pitch_y, well_diameter, a1_x, a1_y) = _get_cached_plate_grid(num_wells)
    cell_dict = _WELL_CELLS_CACHE[num_wells]
    well_r_mm = well_diameter / 2.0

    ch_l = channel.lower()

    # Determine Base Background Color, Cell Color, Nucleus Color & Rim Color (BGR)
    if "blue" in ch_l or "dapi" in ch_l:
        bg_color = np.array([25, 10, 5], dtype=np.float32)
        cell_color = np.array([255, 180, 40], dtype=np.float32)
        nuc_color = np.array([255, 240, 160], dtype=np.float32)
        rim_color = np.array([140, 70, 30], dtype=np.float32)
        deck_color = np.array([12, 5, 2], dtype=np.float32)
    elif "green" in ch_l or "fitc" in ch_l or "gfp" in ch_l:
        bg_color = np.array([10, 22, 5], dtype=np.float32)
        cell_color = np.array([40, 245, 60], dtype=np.float32)
        nuc_color = np.array([180, 255, 180], dtype=np.float32)
        rim_color = np.array([30, 130, 40], dtype=np.float32)
        deck_color = np.array([5, 12, 2], dtype=np.float32)
    elif "red" in ch_l or "txred" in ch_l or "mcherry" in ch_l or "cy5" in ch_l:
        bg_color = np.array([5, 5, 22], dtype=np.float32)
        cell_color = np.array([20, 30, 255], dtype=np.float32)
        nuc_color = np.array([160, 180, 255], dtype=np.float32)
        rim_color = np.array([30, 40, 130], dtype=np.float32)
        deck_color = np.array([2, 2, 12], dtype=np.float32)
    elif "darkfield" in ch_l or "df" in ch_l:
        bg_color = np.array([18, 18, 22], dtype=np.float32)
        cell_color = np.array([220, 220, 240], dtype=np.float32)
        nuc_color = np.array([255, 255, 255], dtype=np.float32)
        rim_color = np.array([180, 180, 200], dtype=np.float32)
        deck_color = np.array([10, 10, 12], dtype=np.float32)
    elif "phase" in ch_l:
        bg_color = np.array([175, 175, 180], dtype=np.float32)
        cell_color = np.array([45, 45, 50], dtype=np.float32)
        nuc_color = np.array([220, 220, 230], dtype=np.float32)
        rim_color = np.array([90, 90, 100], dtype=np.float32)
        deck_color = np.array([60, 60, 65], dtype=np.float32)
    else:  # Brightfield
        bg_color = np.array([225, 225, 225], dtype=np.float32)
        cell_color = np.array([60, 60, 70], dtype=np.float32)
        nuc_color = np.array([25, 25, 30], dtype=np.float32)
        rim_color = np.array([100, 100, 110], dtype=np.float32)
        deck_color = np.array([80, 80, 85], dtype=np.float32)

    # Base deck frame
    deck_c = tuple(float(x) for x in np.clip(deck_color * brightness, 0, 255))
    frame = np.full((h, w, 3), deck_c, dtype=np.float32)

    bg_c = tuple(float(x) for x in np.clip(bg_color * brightness, 0, 255))
    cell_c = tuple(float(x) for x in np.clip(cell_color * brightness, 0, 255))
    nuc_c = tuple(float(x) for x in np.clip(nuc_color * brightness, 0, 255))
    rim_c = tuple(float(x) for x in np.clip(rim_color * brightness, 0, 255))

    # Render Inter-Well Plastic Grid Structural Ribs
    grid_c = tuple(float(x) for x in np.clip(rim_color * brightness * 0.4, 0, 255))
    grid_w_px = max(1, int(0.10 * px_per_mm_x))

    for r in range(rows):
        wy = a1_y - r * pitch_y
        if fov_bottom - 0.5 <= wy <= fov_top + 0.5:
            ry_px = int(((fov_top - wy) / fov_h) * h)
            if 0 <= ry_px < h:
                cv2.line(frame, (0, ry_px), (w, ry_px), grid_c, grid_w_px)

    for c in range(cols):
        wx = a1_x + c * pitch_x
        if fov_left - 0.5 <= wx <= fov_right + 0.5:
            cx_px = int(((wx - fov_left) / fov_w) * w)
            if 0 <= cx_px < w:
                cv2.line(frame, (cx_px, 0), (cx_px, h), grid_c, grid_w_px)

    # Spatial filtering for wells overlapping FOV
    max_fov_radius = math.hypot(fov_w / 2.0, fov_h / 2.0)
    
    for wx, wy, r_idx, c_idx, w_name in wells:
        dist_well = math.hypot(wx - stage_x, wy - stage_y)
        if dist_well <= (max_fov_radius + well_r_mm + 0.5):
            well_px_x = int(((wx - fov_left) / fov_w) * w)
            well_px_y = int(((fov_top - wy) / fov_h) * h)
            well_r_px = int(well_r_mm * px_per_mm_x)

            # Draw filled well interior disk
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, bg_c, -1)

            # Draw cell bodies and internal nuclei structures inside well
            sub_x, sub_y, sub_r = cell_dict[(r_idx, c_idx)]
            for cx, cy, cr in zip(sub_x, sub_y, sub_r):
                if fov_left - 0.1 <= cx <= fov_right + 0.1 and fov_bottom - 0.1 <= cy <= fov_top + 0.1:
                    c_px_x = int(((cx - fov_left) / fov_w) * w)
                    c_px_y = int(((fov_top - cy) / fov_h) * h)
                    c_r_px = max(2, int(cr * px_per_mm_x))
                    if -c_r_px <= c_px_x <= w + c_r_px and -c_r_px <= c_px_y <= h + c_r_px:
                        # Cell cytoplasm body
                        cv2.circle(frame, (c_px_x, c_px_y), c_r_px, cell_c, -1)
                        # Internal nucleus structure
                        if c_r_px >= 3:
                            cv2.circle(frame, (c_px_x, c_px_y), max(1, c_r_px // 2), nuc_c, -1)

            # Draw Double-Ring Well Rim Wall
            rim_outer_px = max(2, int(0.12 * px_per_mm_x))
            rim_inner_px = max(1, int(0.06 * px_per_mm_x))
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, rim_c, rim_outer_px)
            if well_r_px > 10:
                cv2.circle(frame, (well_px_x, well_px_y), well_r_px - rim_outer_px, (30, 30, 35), rim_inner_px)

    # Render Physical Plate Outer Skirt Borders & Metal Stage Holder (when approaching plate boundaries)
    plate_max_x = 127.76 / 2.0  # 63.88 mm
    plate_max_y = 85.48 / 2.0   # 42.74 mm
    border_c = (40, 40, 45)
    holder_c = (15, 15, 18)

    # Left plate edge
    if fov_left <= -plate_max_x <= fov_right:
        edge_px = int(((-plate_max_x - fov_left) / fov_w) * w)
        cv2.rectangle(frame, (0, 0), (edge_px, h), holder_c, -1)
        cv2.line(frame, (edge_px, 0), (edge_px, h), border_c, max(2, int(0.2 * px_per_mm_x)))

    # Right plate edge
    if fov_left <= plate_max_x <= fov_right:
        edge_px = int(((plate_max_x - fov_left) / fov_w) * w)
        cv2.rectangle(frame, (edge_px, 0), (w, h), holder_c, -1)
        cv2.line(frame, (edge_px, 0), (edge_px, h), border_c, max(2, int(0.2 * px_per_mm_x)))

    # Top plate edge
    if fov_bottom <= plate_max_y <= fov_top:
        edge_py = int(((fov_top - plate_max_y) / fov_h) * h)
        cv2.rectangle(frame, (0, 0), (w, edge_py), holder_c, -1)
        cv2.line(frame, (0, edge_py), (w, edge_py), border_c, max(2, int(0.2 * px_per_mm_y)))

    # Bottom plate edge
    if fov_bottom <= -plate_max_y <= fov_top:
        edge_py = int(((fov_top - (-plate_max_y)) / fov_h) * h)
        cv2.rectangle(frame, (0, edge_py), (w, h), holder_c, -1)
        cv2.line(frame, (0, edge_py), (w, edge_py), border_c, max(2, int(0.2 * px_per_mm_y)))

    out_img = np.clip(frame, 0, 255).astype(np.uint8)

    # Z defocus blur
    dist_z = abs(stage_z - best_z)
    if dist_z > 0.001:
        blur_k = int(dist_z * 25.0)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_k = max(1, min(blur_k, 31))
        if blur_k > 1:
            out_img = cv2.GaussianBlur(out_img, (blur_k, blur_k), 0)

    return out_img


def generate_simulated_frame(*args, **kwargs) -> np.ndarray:
    return generate_well_plate_cell_frame(*args, **kwargs)
