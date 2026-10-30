from __future__ import annotations
"""
image_utils.py - Fast, cached image processing, flat-field shading correction, focus metrics,
C-memory frame sanitization, and hardware LED/Camera control for LumaViewPro.
"""

import os
import cv2
import numpy as np
import math
from typing import Optional, Tuple, Union, Dict, List

# Pre-computed cell colony coordinate cache
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
            cell_rs = np.random.uniform(0.015, 0.035, num_cells)
            
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


def sanitize_camera_frame(frame: Optional[np.ndarray]) -> np.ndarray:
    """
    Ensures any raw camera frame (Mono8, Mono16, BGR, RGB, Float) from Basler Pylon
    is safely converted to a 3-channel contiguous uint8 RGB numpy array of shape (H, W, 3).
    Prevents Kivy C-extension OpenGL texture memory segfaults (0xC0000005).
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return generate_disconnected_camera_frame()

    try:
        # Make a deep copy to detach from PyPylon C++ circular ring buffer
        frame_copy = np.array(frame, copy=True)

        if len(frame_copy.shape) == 1:
            side = int(math.sqrt(frame_copy.size))
            if side * side == frame_copy.size:
                frame_copy = frame_copy.reshape((side, side))
            else:
                return generate_disconnected_camera_frame()

        if frame_copy.dtype != np.uint8:
            if frame_copy.dtype == np.uint16:
                max_val = frame_copy.max()
                frame_8u = (frame_copy >> 8).astype(np.uint8) if max_val > 255 else frame_copy.astype(np.uint8)
            else:
                frame_8u = cv2.normalize(frame_copy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            frame_8u = frame_copy

        if len(frame_8u.shape) == 2:
            rgb = cv2.cvtColor(frame_8u, cv2.COLOR_GRAY2RGB)
        elif len(frame_8u.shape) == 3:
            channels = frame_8u.shape[2]
            if channels == 1:
                rgb = cv2.cvtColor(frame_8u, cv2.COLOR_GRAY2RGB)
            elif channels == 3:
                rgb = cv2.cvtColor(frame_8u, cv2.COLOR_BGR2RGB)
            elif channels == 4:
                rgb = cv2.cvtColor(frame_8u, cv2.COLOR_BGRA2RGB)
            else:
                rgb = frame_8u[:, :, :3]
        else:
            return generate_disconnected_camera_frame()

        return np.ascontiguousarray(rgb, dtype=np.uint8)
    except Exception as e:
        print(f"[image_utils] sanitize_camera_frame error: {e}")
        return generate_disconnected_camera_frame()


def safe_set_led(leds_obj: object, channel_name: str, power_percent: float) -> bool:
    """
    Safely controls physical Etaluma LED Mainboard.
    Maps optical channel names ('Brightfield', 'Blue', 'Green', 'Red') to integer IDs (0, 1, 2, 3)
    and handles multiple driver method signatures without crashing.
    """
    if leds_obj is None:
        return False

    CHANNEL_MAP = {
        "Brightfield": 0, "BF": 0, "Phase": 0, "Darkfield": 0,
        "Blue": 1, "DAPI": 1,
        "Green": 2, "FITC": 2, "GFP": 2,
        "Red": 3, "TxRed": 3, "Cy5": 3, "mCherry": 3
    }
    ch_idx = CHANNEL_MAP.get(channel_name, 0)
    p_val = max(0, min(100, int(power_percent)))

    for method_name in ["set_led", "set_channel", "set_power"]:
        if hasattr(leds_obj, method_name):
            method = getattr(leds_obj, method_name)
            try:
                method(ch_idx, p_val)
                return True
            except Exception:
                try:
                    method(channel_name, p_val)
                    return True
                except Exception as e:
                    print(f"[LEDBoard {method_name} error]: {e}")
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


def generate_disconnected_camera_frame(sensor_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Generates a clear 'MICROSCOPE NOT CONNECTED' warning overlay when hardware is absent or unattached."""
    w, h = sensor_size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Dark slate blue background with border
    frame[:] = (25, 20, 15)
    cv2.rectangle(frame, (15, 15), (w - 15, h - 15), (50, 50, 160), 2)

    # Disconnected Camera Icon / Warning
    cx, cy = w // 2, h // 2
    cv2.circle(frame, (cx, cy - 35), 35, (60, 60, 200), 3)
    cv2.line(frame, (cx - 25, cy - 60), (cx + 25, cy - 10), (60, 60, 200), 4)

    cv2.putText(frame, "MICROSCOPE NOT CONNECTED", (cx - 175, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100, 100, 255), 2)
    cv2.putText(frame, "Hardware Drivers (pyloncamera, motorboard, ledboard)", (cx - 190, cy + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 210), 1)
    cv2.putText(frame, "or USB Microscope Hardware Not Detected", (cx - 150, cy + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 210), 1)
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
    power: float = 100.0,
    sensor_size: Tuple[int, int] = (512, 512),
    fov_size_mm: Tuple[float, float] = (0.60, 0.45),
    is_light_on: bool = False,
    num_wells: int = 96,
    **kwargs
) -> np.ndarray:
    """Generates synthetic camera frames for offline simulator."""
    w, h = sensor_size

    if not is_light_on or channel.upper() in ["OFF", "NONE", ""]:
        dark_noise = np.random.normal(loc=2.5, scale=1.0, size=(h, w, 3)).astype(np.int16)
        return np.clip(dark_noise, 0, 255).astype(np.uint8)

    p_val = led_power if led_power is not None else power
    p_factor = max(0.05, min(1.5, float(p_val) / 100.0))
    exp_factor = max(0.1, min(3.0, (exposure_ms / 20.0) * gain))
    brightness = p_factor * exp_factor

    fov_w, fov_h = fov_size_mm
    fov_w = max(0.05, float(fov_w))
    fov_h = max(0.05, float(fov_h))

    fov_left = stage_x - fov_w / 2.0
    fov_right = stage_x + fov_w / 2.0
    fov_bottom = stage_y - fov_h / 2.0
    fov_top = stage_y + fov_h / 2.0

    well_specs = {
        6: (2, 3, 39.12, 39.12, 34.8, -39.12, 19.56),
        12: (3, 4, 26.0, 26.0, 22.1, -39.0, 26.0),
        24: (4, 6, 19.3, 19.3, 15.6, -48.25, 28.95),
        48: (6, 8, 13.0, 13.0, 11.0, -45.5, 32.5),
        96: (8, 12, 9.0, 9.0, 6.4, -49.5, 31.5),
        384: (16, 24, 4.5, 4.5, 3.8, -51.75, 33.75)
    }
    rows, cols, pitch_x, pitch_y, well_d, a1_x, a1_y = well_specs.get(num_wells, well_specs[96])
    well_r_mm = well_d / 2.0

    wells = []
    for r in range(rows):
        wy = a1_y - r * pitch_y
        for c in range(cols):
            wx = a1_x + c * pitch_x
            wells.append((wx, wy, r, c))

    ch_l = channel.lower()

    if "blue" in ch_l or "dapi" in ch_l:
        bg_color = np.array([25, 10, 5], dtype=np.float32)
        cell_color = np.array([255, 180, 40], dtype=np.float32)
        rim_color = np.array([120, 60, 20], dtype=np.float32)
    elif "green" in ch_l or "fitc" in ch_l or "gfp" in ch_l:
        bg_color = np.array([10, 20, 10], dtype=np.float32)
        cell_color = np.array([40, 240, 60], dtype=np.float32)
        rim_color = np.array([30, 120, 30], dtype=np.float32)
    elif "red" in ch_l or "txred" in ch_l or "mcherry" in ch_l or "cy5" in ch_l:
        bg_color = np.array([20, 10, 10], dtype=np.float32)
        cell_color = np.array([30, 40, 250], dtype=np.float32)
        rim_color = np.array([30, 30, 120], dtype=np.float32)
    elif "darkfield" in ch_l or "df" in ch_l:
        bg_color = np.array([15, 15, 20], dtype=np.float32)
        cell_color = np.array([220, 220, 240], dtype=np.float32)
        rim_color = np.array([180, 180, 200], dtype=np.float32)
    elif "phase" in ch_l:
        bg_color = np.array([175, 175, 180], dtype=np.float32)
        cell_color = np.array([40, 40, 45], dtype=np.float32)
        rim_color = np.array([100, 100, 110], dtype=np.float32)
    else:  # Brightfield
        bg_color = np.array([220, 220, 220], dtype=np.float32)
        cell_color = np.array([70, 70, 80], dtype=np.float32)
        rim_color = np.array([110, 110, 120], dtype=np.float32)

    bg_c = tuple(int(x) for x in np.clip(bg_color * brightness, 0, 255))
    cell_c = tuple(int(x) for x in np.clip(cell_color * brightness, 0, 255))
    rim_c = tuple(int(x) for x in np.clip(rim_color * brightness, 0, 255))

    frame = np.zeros((h, w, 3), dtype=np.float32)

    for wx, wy, r_idx, c_idx in wells:
        dist_well = math.hypot(wx - stage_x, wy - stage_y)
        if dist_well <= (max(fov_w, fov_h) + well_r_mm + 1.0):
            well_px_x = int(((wx - fov_left) / fov_w) * w)
            well_px_y = int(((fov_top - wy) / fov_h) * h)
            well_r_px = int((well_r_mm / fov_w) * w)

            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, bg_c, -1)

            cell_list = _CELL_CACHE.get((r_idx % 16, c_idx % 24), [])
            for dx, dy, cr_px in cell_list:
                cx_mm = wx + dx
                cy_mm = wy + dy
                if math.hypot(dx, dy) < (well_r_mm - 0.05):
                    if fov_left - 0.1 <= cx_mm <= fov_right + 0.1 and fov_bottom - 0.1 <= cy_mm <= fov_top + 0.1:
                        c_px_x = int(((cx_mm - fov_left) / fov_w) * w)
                        c_px_y = int(((fov_top - cy_mm) / fov_h) * h)
                        c_r_px = max(2, int((cr_px / 512.0) * (1.2 / fov_w) * w))
                        if 0 <= c_px_x < w and 0 <= c_px_y < h:
                            cv2.circle(frame, (c_px_x, c_px_y), c_r_px, cell_c, -1)

            rim_w_px = max(2, int((0.15 / fov_w) * w))
            cv2.circle(frame, (well_px_x, well_px_y), well_r_px, rim_c, rim_w_px)

    out_img = np.clip(frame, 0, 255).astype(np.uint8)

    dist_z = abs(stage_z - best_z)
    if dist_z > 0.001:
        blur_k = int(dist_z * 35.0)
        if blur_k % 2 == 0:
            blur_k += 1
        blur_k = max(1, min(blur_k, 31))
        if blur_k > 1:
            out_img = cv2.GaussianBlur(out_img, (blur_k, blur_k), 0)

    return out_img


def generate_simulated_frame(*args, **kwargs) -> np.ndarray:
    return generate_well_plate_cell_frame(*args, **kwargs)
