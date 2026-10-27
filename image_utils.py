"""
image_utils.py - Fast, cached image processing, shading correction, focus metrics,
and optically accurate synthetic camera frame generation for LumaViewPro.
"""

from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional, Tuple, Union, List, Dict

# Pre-computed cell colony coordinate cache for 20 FPS live simulation
_CELL_CACHE: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}


def _init_cell_cache():
    global _CELL_CACHE
    if _CELL_CACHE:
        return
    for r in range(16):
        for c in range(24):
            seed = (r * 24 + c + 1) * 1000
            np.random.seed(seed)
            num_cells = 45
            angles = np.random.uniform(0, 2 * np.pi, num_cells)
            radii = np.random.uniform(0.1, 3.2, num_cells)
            cell_rs = np.random.uniform(10, 22, num_cells)
            
            cell_list = []
            for i in range(num_cells):
                dx = radii[i] * np.cos(angles[i])
                dy = radii[i] * np.sin(angles[i])
                cell_list.append((dx, dy, cell_rs[i]))
            _CELL_CACHE[(r, c)] = cell_list


_init_cell_cache()


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
    rows, cols = 8, 12
    pitch_x, pitch_y = 9.0, 9.0
    a1_x, a1_y = -49.5, 31.5

    if num_wells == 6:
        rows, cols, pitch_x, pitch_y, a1_x, a1_y = 2, 3, 39.12, 39.12, -39.12, 19.56
    elif num_wells == 12:
        rows, cols, pitch_x, pitch_y, a1_x, a1_y = 3, 4, 26.0, 26.0, -39.0, 26.0
    elif num_wells == 24:
        rows, cols, pitch_x, pitch_y, a1_x, a1_y = 4, 6, 19.3, 19.3, -48.25, 28.95
    elif num_wells == 48:
        rows, cols, pitch_x, pitch_y, a1_x, a1_y = 6, 8, 13.0, 13.0, -45.5, 32.5
    elif num_wells == 384:
        rows, cols, pitch_x, pitch_y, a1_x, a1_y = 16, 24, 4.5, 4.5, -51.75, 33.75

    wells = []
    all_x, all_y, all_r = [], [], []

    for r in range(rows):
        wy = a1_y - r * pitch_y
        for c in range(cols):
            wx = a1_x + c * pitch_x
            w_name = f"{chr(65+r)}{c+1}"
            wells.append((wx, wy, w_name))

            cell_list = _CELL_CACHE.get((r % 16, c % 24), [])
            for dx, dy, cr in cell_list:
                all_x.append(wx + dx)
                all_y.append(wy + dy)
                all_r.append(cr * 0.002)

    return np.array(all_x, dtype=np.float32), np.array(all_y, dtype=np.float32), np.array(all_r, dtype=np.float32), wells


def generate_well_plate_cell_frame(
    stage_x: float = -49.5,
    stage_y: float = 31.5,
    stage_z: float = 0.0,
    best_z: float = 0.0,
    channel: str = "Brightfield",
    exposure_ms: float = 20.0,
    gain: float = 1.0,
    led_power: float = 100.0,
    power: float = 100.0,
    sensor_size: Tuple[int, int] = (512, 512),
    fov_size_mm: Tuple[float, float] = (0.60, 0.45),
    is_light_on: bool = False,
    num_wells: int = 96,
    **kwargs
) -> np.ndarray:
    w, h = sensor_size

    # DARK SENSOR LAUNCH STATE
    if not is_light_on or led_power <= 0:
        dark_noise = np.random.poisson(lam=2.0, size=(h, w, 3)).astype(np.uint8)
        return dark_noise

    p_val = led_power if led_power is not None else power
    brightness = (exposure_ms / 20.0) * (p_val / 100.0) * gain
    brightness = max(0.1, min(3.0, brightness))

    fov_w, fov_h = fov_size_mm
    fov_w = max(0.05, float(fov_w))
    fov_h = max(0.05, float(fov_h))

    px_per_mm = w / fov_w

    fov_left = stage_x - fov_w / 2.0
    fov_right = stage_x + fov_w / 2.0
    fov_bottom = stage_y - fov_h / 2.0
    fov_top = stage_y + fov_h / 2.0

    cells_x, cells_y, cells_r, wells = _get_cached_cell_positions(num_wells)

    # Filter visible cells inside active FOV
    mask = (cells_x >= fov_left - 0.05) & (cells_x <= fov_right + 0.05) & \
           (cells_y >= fov_bottom - 0.05) & (cells_y <= fov_top + 0.05)

    vis_x = cells_x[mask]
    vis_y = cells_y[mask]
    vis_r = cells_r[mask]

    vis_px_x = ((vis_x - fov_left) / fov_w) * w
    vis_px_y = ((fov_top - vis_y) / fov_h) * h
    vis_px_r = vis_r * px_per_mm

    frame = np.zeros((h, w, 3), dtype=np.float32)

    ch_l = channel.lower()
    if "blue" in ch_l or "dapi" in ch_l:
        bg_color = np.array([15, 10, 5], dtype=np.float32)
        cell_color = np.array([255, 180, 40], dtype=np.float32)
    elif "green" in ch_l or "fitc" in ch_l or "gfp" in ch_l:
        bg_color = np.array([10, 20, 10], dtype=np.float32)
        cell_color = np.array([40, 240, 60], dtype=np.float32)
    elif "red" in ch_l or "txred" in ch_l or "mcherry" in ch_l or "cy5" in ch_l:
        bg_color = np.array([20, 10, 10], dtype=np.float32)
        cell_color = np.array([30, 40, 250], dtype=np.float32)
    elif "darkfield" in ch_l or "df" in ch_l:
        bg_color = np.array([15, 15, 20], dtype=np.float32)
        cell_color = np.array([220, 220, 240], dtype=np.float32)
    elif "phase" in ch_l:
        bg_color = np.array([170, 170, 175], dtype=np.float32)
        cell_color = np.array([45, 45, 50], dtype=np.float32)
    else:  # Brightfield
        bg_color = np.array([215, 215, 215], dtype=np.float32)
        cell_color = np.array([55, 55, 65], dtype=np.float32)

    frame[:] = bg_color * brightness
    c_draw_color = tuple(float(x) for x in np.clip(cell_color * brightness, 0, 255))
    halo_color = tuple(float(x) for x in np.clip(np.array([240, 240, 245]) * brightness, 0, 255))

    # Draw visible cell bodies
    for px, py, pr in zip(vis_px_x, vis_px_y, vis_px_r):
        ix, iy, ir = int(px), int(py), max(2, int(pr))
        if -ir <= ix <= w + ir and -ir <= iy <= h + ir:
            cv2.circle(frame, (ix, iy), ir, c_draw_color, -1)
            if "phase" in ch_l:
                cv2.circle(frame, (ix, iy), ir + 1, halo_color, 1)

    # Well rim outline if well center is close to active FOV
    well_diameter_mm = 6.4
    if num_wells in [6, 12, 24, 48, 96, 384]:
        formats = {6: 34.8, 12: 22.1, 24: 15.6, 48: 11.0, 96: 6.4, 384: 3.8}
        well_diameter_mm = formats[num_wells]

    well_r_mm = well_diameter_mm / 2.0
    rim_color = tuple(float(x) for x in np.clip(np.array([100, 110, 120]) * brightness, 0, 255))
    for wx, wy, _ in wells:
        if abs(wx - stage_x) <= (fov_w / 2.0 + well_r_mm) and abs(wy - stage_y) <= (fov_h / 2.0 + well_r_mm):
            well_px_x = int(((wx - fov_left) / fov_w) * w)
            well_px_y = int(((fov_top - wy) / fov_h) * h)
            well_r_px = int(well_r_mm * px_per_mm)
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, rim_color, 2)

    # In-between wells (outside all well boundaries): set background intensity to 0
    min_dist_to_well = min([np.hypot(stage_x - wx, stage_y - wy) for wx, wy, _ in wells])
    if min_dist_to_well > well_r_mm + max(fov_w, fov_h):
        frame[:] = 0.0

    # Z defocus blur simulation
    dist_z = abs(stage_z - best_z)
    if dist_z > 0.001:
        blur_k = int(dist_z * 25.0)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_k = max(1, min(blur_k, 31))
        if blur_k > 1:
            frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)

    out_img = np.clip(frame, 0, 255).astype(np.uint8)
    return out_img


def generate_simulated_frame(*args, **kwargs) -> np.ndarray:
    return generate_well_plate_cell_frame(*args, **kwargs)
