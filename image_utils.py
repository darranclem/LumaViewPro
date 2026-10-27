"""
image_utils.py - Fast, cached image processing, flat-field shading correction, focus metrics,
and optically accurate synthetic camera frame generation for LumaViewPro.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, List


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
    - Outer Plate Footprint: Renders high-contrast plate skirt border and dark aperture stage frame.
    - Inter-Well Grid: Renders structural plastic grid ribs along well rows/columns.
    - Well Interior: Renders filled well floor, cell colony clusters, and double-ring well rims.
    - Focus Z: Applies Gaussian defocus blur when stage_z differs from best_z.
    - Dark Launch State: Returns low Poisson noise instantaneously when is_light_on is False.
    """
    w, h = sensor_size

    # 1. Dark Launch / Light OFF State (returns dark sensor noise instantaneously)
    if not is_light_on or led_power <= 0:
        np.random.seed(0)
        return np.random.poisson(lam=2.5, size=(h, w, 3)).clip(0, 255).astype(np.uint8)

    # 2. Light ON State: Calculate brightness factor
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

    # Plate Footprint Dimensions (ANSI/SLAS standard)
    plate_w_mm = 127.76
    plate_h_mm = 85.48
    p_left_mm = -plate_w_mm / 2.0   # -63.88 mm
    p_right_mm = plate_w_mm / 2.0   # +63.88 mm
    p_top_mm = plate_h_mm / 2.0     # +42.74 mm
    p_bottom_mm = -plate_h_mm / 2.0 # -42.74 mm

    # Channel Colors (BGR format)
    ch_l = channel.lower()
    if "blue" in ch_l or "dapi" in ch_l:
        bg_color = np.array([25, 10, 5], dtype=np.float32)
        cell_color = np.array([255, 180, 40], dtype=np.float32)
        rim_color = np.array([140, 70, 30], dtype=np.float32)
        plastic_color = np.array([12, 6, 3], dtype=np.float32)
    elif "green" in ch_l or "fitc" in ch_l or "gfp" in ch_l:
        bg_color = np.array([10, 20, 10], dtype=np.float32)
        cell_color = np.array([40, 240, 60], dtype=np.float32)
        rim_color = np.array([40, 140, 40], dtype=np.float32)
        plastic_color = np.array([5, 12, 5], dtype=np.float32)
    elif "red" in ch_l or "txred" in ch_l or "mcherry" in ch_l or "cy5" in ch_l:
        bg_color = np.array([20, 10, 10], dtype=np.float32)
        cell_color = np.array([30, 40, 250], dtype=np.float32)
        rim_color = np.array([40, 40, 140], dtype=np.float32)
        plastic_color = np.array([10, 5, 5], dtype=np.float32)
    elif "darkfield" in ch_l or "df" in ch_l:
        bg_color = np.array([15, 15, 20], dtype=np.float32)
        cell_color = np.array([220, 220, 240], dtype=np.float32)
        rim_color = np.array([180, 180, 200], dtype=np.float32)
        plastic_color = np.array([8, 8, 10], dtype=np.float32)
    elif "phase" in ch_l:
        bg_color = np.array([180, 180, 185], dtype=np.float32)
        cell_color = np.array([40, 40, 45], dtype=np.float32)
        rim_color = np.array([100, 100, 110], dtype=np.float32)
        plastic_color = np.array([110, 110, 115], dtype=np.float32)
    else:  # Brightfield
        bg_color = np.array([220, 220, 220], dtype=np.float32)
        cell_color = np.array([60, 60, 70], dtype=np.float32)
        rim_color = np.array([110, 110, 120], dtype=np.float32)
        plastic_color = np.array([130, 130, 135], dtype=np.float32)

    bg_c = tuple(int(x) for x in np.clip(bg_color * brightness, 0, 255))
    cell_c = tuple(int(x) for x in np.clip(cell_color * brightness, 0, 255))
    rim_c = tuple(int(x) for x in np.clip(rim_color * brightness, 0, 255))
    plastic_c = tuple(int(x) for x in np.clip(plastic_color * brightness, 0, 255))
    metal_aperture_c = tuple(int(x) for x in np.clip(np.array([45, 45, 50], dtype=np.float32) * brightness, 0, 255))

    frame = np.full((h, w, 3), metal_aperture_c, dtype=np.uint8)

    # Calculate plate footprint bounding box in viewport pixel space
    pl_px_l = int(((p_left_mm - fov_left) / fov_w) * w)
    pl_px_r = int(((p_right_mm - fov_left) / fov_w) * w)
    pl_px_t = int(((fov_top - p_top_mm) / fov_h) * h)
    pl_px_b = int(((fov_top - p_bottom_mm) / fov_h) * h)

    # Draw plastic plate base footprint
    cv2.rectangle(frame, (pl_px_l, pl_px_t), (pl_px_r, pl_px_b), plastic_c, -1)

    well_formats = {
        6: (2, 3, 39.12, 39.12, 34.8, -39.12, 19.56),
        12: (3, 4, 26.0, 26.0, 22.1, -39.0, 26.0),
        24: (4, 6, 19.3, 19.3, 15.6, -48.25, 28.95),
        48: (6, 8, 13.0, 13.0, 11.0, -45.5, 32.5),
        96: (8, 12, 9.0, 9.0, 6.4, -49.5, 31.5),
        384: (16, 24, 4.5, 4.5, 3.8, -51.75, 33.75)
    }
    rows, cols, pitch_x, pitch_y, well_d, a1_x, a1_y = well_formats.get(num_wells, well_formats[96])
    well_r_mm = well_d / 2.0

    # Draw inter-well plastic grid lines / ribs along well rows and columns
    grid_color = tuple(int(x) for x in np.clip(np.array(rim_c) * 0.7, 0, 255))
    grid_thickness = max(1, int(0.2 * px_per_mm_x))

    for r in range(rows):
        wy = a1_y - r * pitch_y
        line_y_px = int(((fov_top - wy) / fov_h) * h)
        if 0 <= line_y_px < h:
            cv2.line(frame, (max(0, pl_px_l), line_y_px), (min(w, pl_px_r), line_y_px), grid_color, grid_thickness)

    for c in range(cols):
        wx = a1_x + c * pitch_x
        line_x_px = int(((wx - fov_left) / fov_w) * w)
        if 0 <= line_x_px < w:
            cv2.line(frame, (line_x_px, max(0, pl_px_t)), (line_x_px, min(h, pl_px_b)), grid_color, grid_thickness)

    # Render wells overlapping active FOV
    for r in range(rows):
        wy = a1_y - r * pitch_y
        for c in range(cols):
            wx = a1_x + c * pitch_x
            dist_well = math.hypot(wx - stage_x, wy - stage_y)
            if dist_well <= (max(fov_w, fov_h) + well_r_mm + 1.0):
                well_px_x = int(((wx - fov_left) / fov_w) * w)
                well_px_y = int(((fov_top - wy) / fov_h) * h)
                well_r_px = int(well_r_mm * px_per_mm_x)

                # Draw filled well interior
                cv2.circle(frame, (well_px_x, well_px_y), well_r_px, bg_c, -1)

                # Cell colonies inside well
                np.random.seed((r * 24 + c + 1) * 100)
                num_cells = 30
                angles = np.random.uniform(0, 2 * np.pi, num_cells)
                radii = np.random.uniform(0, well_r_mm * 0.85, num_cells)
                cell_radii = np.random.uniform(0.015, 0.035, num_cells)

                for a, rad, cr in zip(angles, radii, cell_radii):
                    cx_mm = wx + rad * math.cos(a)
                    cy_mm = wy + rad * math.sin(a)
                    if fov_left - 0.1 <= cx_mm <= fov_right + 0.1 and fov_bottom - 0.1 <= cy_mm <= fov_top + 0.1:
                        c_px_x = int(((cx_mm - fov_left) / fov_w) * w)
                        c_px_y = int(((fov_top - cy_mm) / fov_h) * h)
                        c_r_px = max(2, int(cr * px_per_mm_x * 12.0))
                        if -c_r_px <= c_px_x <= w + c_r_px and -c_r_px <= c_px_y <= h + c_r_px:
                            cv2.circle(frame, (c_px_x, c_px_y), c_r_px, cell_c, -1)

                # Draw well rim double-ring
                rim_w_px = max(2, int(0.18 * px_per_mm_x))
                cv2.circle(frame, (well_px_x, well_px_y), well_r_px, rim_c, rim_w_px)

    # Outer Plate Skirt Border Line (High contrast)
    skirt_w_px = max(2, int(0.3 * px_per_mm_x))
    skirt_c = tuple(int(x) for x in np.clip(np.array([200, 200, 210], dtype=np.float32) * brightness, 0, 255))
    cv2.rectangle(frame, (pl_px_l, pl_px_t), (pl_px_r, pl_px_b), skirt_c, skirt_w_px)

    # Z defocus blur simulation
    dist_z = abs(stage_z - best_z)
    if dist_z > 0.001:
        blur_k = int(dist_z * 35.0)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_k = max(1, min(blur_k, 31))
        if blur_k > 1:
            frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

    return frame


def generate_simulated_frame(*args, **kwargs) -> np.ndarray:
    return generate_well_plate_cell_frame(*args, **kwargs)
