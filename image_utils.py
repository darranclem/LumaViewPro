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

# Pre-computed cell colony coordinate cache for fast live simulation (20 FPS)
_CELL_COORDINATES_CACHE: Dict[int, Dict[str, object]] = {}
_DARK_FRAME_CACHE: Optional[np.ndarray] = None


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


def _get_cached_cell_positions(num_wells: int = 96):
    if num_wells in _CELL_COORDINATES_CACHE:
        d = _CELL_COORDINATES_CACHE[num_wells]
        return d["by_well"], d["wells"], d["params"]

    well_formats = {
        6: (2, 3, 39.12, 39.12, 34.8, -39.12, 19.56),
        12: (3, 4, 26.0, 26.0, 22.1, -39.0, 26.0),
        24: (4, 6, 19.3, 19.3, 15.6, -48.25, 28.95),
        48: (6, 8, 13.0, 13.0, 11.0, -45.5, 32.5),
        96: (8, 12, 9.0, 9.0, 6.4, -49.5, 31.5),
        384: (16, 24, 4.5, 4.5, 3.8, -51.75, 33.75)
    }
    rows, cols, pitch_x, pitch_y, well_diameter, a1_x, a1_y = well_formats.get(num_wells, well_formats[96])

    wells = []
    by_well = {}

    np.random.seed(42 + num_wells)
    for r in range(rows):
        wy = a1_y - r * pitch_y
        for c in range(cols):
            wx = a1_x + c * pitch_x
            w_name = f"{chr(65+r)}{c+1}"
            wells.append((wx, wy, r, c, w_name))

            num_cells = np.random.randint(25, 55)
            angles = np.random.uniform(0, 2 * np.pi, num_cells)
            radii = np.random.uniform(0, pitch_x * 0.32, num_cells)
            cx = wx + radii * np.cos(angles)
            cy = wy + radii * np.sin(angles)
            cr = np.random.uniform(0.015, 0.035, num_cells)

            by_well[(r, c)] = (
                cx.astype(np.float32),
                cy.astype(np.float32),
                cr.astype(np.float32)
            )

    params = (rows, cols, pitch_x, pitch_y, well_diameter, a1_x, a1_y)
    _CELL_COORDINATES_CACHE[num_wells] = {
        "by_well": by_well, "wells": wells, "params": params
    }
    return by_well, wells, params


def generate_well_plate_cell_frame(
    stage_x: float = -49.5,
    stage_y: float = 31.5,
    stage_z: float = 0.0,
    best_z: float = 0.0,
    channel: str = "Brightfield",
    exposure_ms: float = 20.0,
    gain: float = 1.0,
    led_power: float = 100.0,
    is_light_on: bool = True,
    fov_size_mm: Tuple[float, float] = (0.60, 0.45),
    num_wells: int = 96,
    sensor_size: Tuple[int, int] = (512, 512),
    **kwargs
) -> np.ndarray:
    """
    Generates synthetic high-resolution camera frames with ultra-fast OpenCV rendering.
    - Outside well rims: 0 intensity (pure black plastic background)
    - Inside wells: filled background + cell colonies + well rim borders
    - Dark launch state: low Poisson sensor noise
    """
    global _DARK_FRAME_CACHE
    w, h = sensor_size

    # 1. Dark Launch / Light OFF State (returns dark sensor noise instantaneously - 0.001 ms CPU time)
    if not is_light_on or led_power <= 0:
        if _DARK_FRAME_CACHE is None or _DARK_FRAME_CACHE.shape != (h, w, 3):
            np.random.seed(0)
            _DARK_FRAME_CACHE = np.random.poisson(lam=2.5, size=(h, w, 3)).clip(0, 255).astype(np.uint8)
        return _DARK_FRAME_CACHE

    # 2. Light ON State: Calculate brightness factor
    brightness = max(0.1, min(3.0, (exposure_ms / 20.0) * (led_power / 100.0) * gain))

    fov_w, fov_h = fov_size_mm
    fov_w = max(0.05, float(fov_w))
    fov_h = max(0.05, float(fov_h))

    px_per_mm_x = w / fov_w

    fov_left = stage_x - fov_w / 2.0
    fov_right = stage_x + fov_w / 2.0
    fov_bottom = stage_y - fov_h / 2.0
    fov_top = stage_y + fov_h / 2.0

    by_well, wells, (rows, cols, pitch_x, pitch_y, well_diameter, a1_x, a1_y) = _get_cached_cell_positions(num_wells)
    well_r_mm = well_diameter / 2.0

    ch_l = channel.lower()

    # Determine Base Background Color & LUT (BGR format)
    if "blue" in ch_l or "dapi" in ch_l:
        bg_color = np.array([25, 8, 4], dtype=np.float32)
        cell_color = np.array([255, 180, 40], dtype=np.float32)  # Pure Blue
        rim_color = np.array([120, 60, 20], dtype=np.float32)
    elif "green" in ch_l or "fitc" in ch_l or "gfp" in ch_l:
        bg_color = np.array([8, 20, 4], dtype=np.float32)
        cell_color = np.array([40, 240, 60], dtype=np.float32)   # Pure Green
        rim_color = np.array([40, 120, 40], dtype=np.float32)
    elif "red" in ch_l or "txred" in ch_l or "mcherry" in ch_l or "cy5" in ch_l:
        bg_color = np.array([4, 4, 20], dtype=np.float32)
        cell_color = np.array([20, 30, 255], dtype=np.float32)   # Pure Red
        rim_color = np.array([30, 40, 120], dtype=np.float32)
    elif "darkfield" in ch_l or "df" in ch_l:
        bg_color = np.array([15, 15, 20], dtype=np.float32)      # Darkfield
        cell_color = np.array([220, 220, 240], dtype=np.float32)
        rim_color = np.array([180, 180, 200], dtype=np.float32)
    elif "phase" in ch_l:
        bg_color = np.array([175, 175, 180], dtype=np.float32)   # Phase Contrast
        cell_color = np.array([45, 45, 50], dtype=np.float32)
        rim_color = np.array([100, 100, 110], dtype=np.float32)
    else:  # Brightfield
        bg_color = np.array([220, 220, 220], dtype=np.float32)   # Brightfield Grayscale
        cell_color = np.array([55, 55, 65], dtype=np.float32)
        rim_color = np.array([110, 110, 120], dtype=np.float32)

    # Frame starts as 0 intensity (pure black plastic between wells)
    frame = np.zeros((h, w, 3), dtype=np.float32)

    bg_c = tuple(float(x) for x in np.clip(bg_color * brightness, 0, 255))
    cell_c = tuple(float(x) for x in np.clip(cell_color * brightness, 0, 255))
    rim_c = tuple(float(x) for x in np.clip(rim_color * brightness, 0, 255))

    # Render wells overlapping active FOV
    max_fov_radius = math.hypot(fov_w / 2.0, fov_h / 2.0)
    
    for wx, wy, r_idx, c_idx, w_name in wells:
        dist_well = math.hypot(wx - stage_x, wy - stage_y)
        if dist_well <= (max_fov_radius + well_r_mm + 1.0):
            well_px_x = int(((wx - fov_left) / fov_w) * w)
            well_px_y = int(((fov_top - wy) / fov_h) * h)
            well_r_px = int(well_r_mm * px_per_mm_x)

            # Draw filled well interior
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, bg_c, -1)

            # Draw cells in this well directly from dict
            sub_x, sub_y, sub_r = by_well[(r_idx, c_idx)]

            for cx, cy, cr in zip(sub_x, sub_y, sub_r):
                if fov_left - 0.1 <= cx <= fov_right + 0.1 and fov_bottom - 0.1 <= cy <= fov_top + 0.1:
                    c_px_x = int(((cx - fov_left) / fov_w) * w)
                    c_px_y = int(((fov_top - cy) / fov_h) * h)
                    c_r_px = max(2, int(cr * px_per_mm_x * 12.0))
                    if -c_r_px <= c_px_x <= w + c_r_px and -c_r_px <= c_px_y <= h + c_r_px:
                        cv2.circle(frame, (c_px_x, c_px_y), c_r_px, cell_c, -1)

            # Draw well rim border line
            rim_w_px = max(2, int(0.12 * px_per_mm_x))
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, rim_c, rim_w_px)

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
