"""
image_utils.py - Image processing, flat-field shading correction, autofocus metrics,
and synthetic cellular image generation for offline simulation.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFilter


def load_image(filepath: str) -> Optional[np.ndarray]:
    if not os.path.exists(filepath):
        return None
    try:
        img = Image.open(filepath)
        return np.array(img)
    except Exception as e:
        print(f"[Error] Failed to load image {filepath}: {e}")
        return None


def apply_shading_correction(
    raw_img: np.ndarray,
    ref_img: np.ndarray,
    dark_img: Optional[np.ndarray] = None
) -> np.ndarray:
    if raw_img is None or ref_img is None:
        return raw_img

    raw_f = raw_img.astype(np.float32)
    ref_f = ref_img.astype(np.float32)

    if dark_img is not None:
        dark_f = dark_img.astype(np.float32)
        raw_f = np.maximum(0.0, raw_f - dark_f)
        ref_f = np.maximum(1e-5, ref_f - dark_f)
    else:
        ref_f = np.maximum(1e-5, ref_f)

    ref_mean = float(np.mean(ref_f))
    corrected = (raw_f / ref_f) * ref_mean

    if raw_img.dtype == np.uint8:
        return np.clip(corrected, 0, 255).astype(np.uint8)
    elif raw_img.dtype == np.uint16:
        return np.clip(corrected, 0, 65535).astype(np.uint16)
    return corrected


def compute_focus_score(img: np.ndarray, method: str = "laplacian") -> float:
    if img is None or img.size == 0:
        return 0.0

    img_f = img.astype(np.float32)
    if img_f.ndim == 3:
        img_f = np.mean(img_f, axis=2)

    if method == "brenner":
        diff_x = np.diff(img_f, n=2, axis=1)
        return float(np.mean(diff_x ** 2))

    elif method == "tenengrad":
        gx = np.zeros_like(img_f)
        gy = np.zeros_like(img_f)
        gx[:, 1:-1] = img_f[:, 2:] - img_f[:, :-2]
        gy[1:-1, :] = img_f[2:, :] - img_f[:-2, :]
        mag = np.hypot(gx, gy)
        return float(np.mean(mag ** 2))

    else:
        H, W = img_f.shape
        if H < 3 or W < 3:
            return 0.0

        lap = (
            img_f[1:-1, 2:] + img_f[1:-1, :-2] +
            img_f[2:, 1:-1] + img_f[:-2, 1:-1] -
            4.0 * img_f[1:-1, 1:-1]
        )
        return float(np.var(lap))


def generate_well_plate_cell_frame(
    stage_x: float = 0.0,
    stage_y: float = 0.0,
    stage_z: float = 0.0,
    best_z: float = 0.0,
    channel: str = "Brightfield",
    sensor_size: Tuple[int, int] = (512, 512),
    fov_size_mm: Tuple[float, float] = (1.2, 1.2)
) -> np.ndarray:
    """
    Generates synthetic high-resolution mock cellular images across a SBS 96-well microplate.
    Displays realistic cell colonies, cytoplasm, fluo-labeled nuclei, well borders,
    and realistic Z-defocus blur for offline microscope simulation.
    """
    w, h = sensor_size
    fov_w, fov_h = fov_size_mm

    # SBS 96-Well Plate Layout: 8 rows (A-H) x 12 cols (1-12)
    wells = []
    for row in range(8):
        wy = 31.5 - row * 9.0
        for col in range(12):
            wx = -49.5 + col * 9.0
            wells.append((wx, wy, f"{chr(65+row)}{col+1}"))

    np.random.seed(42)
    all_cells_x, all_cells_y, all_cells_r = [], [], []

    for wx, wy, _ in wells:
        num_cells = np.random.randint(30, 70)
        angles = np.random.uniform(0, 2 * np.pi, num_cells)
        radii = np.random.uniform(0, 2.8, num_cells)
        cx = wx + radii * np.cos(angles)
        cy = wy + radii * np.sin(angles)
        cr = np.random.uniform(10, 22, num_cells)
        all_cells_x.extend(cx)
        all_cells_y.extend(cy)
        all_cells_r.extend(cr)

    all_cells_x = np.array(all_cells_x)
    all_cells_y = np.array(all_cells_y)
    all_cells_r = np.array(all_cells_r)

    x_min, x_max = stage_x - fov_w / 2.0, stage_x + fov_w / 2.0
    y_min, y_max = stage_y - fov_h / 2.0, stage_y + fov_h / 2.0

    in_fov = (
        (all_cells_x >= x_min - 0.2) & (all_cells_x <= x_max + 0.2) &
        (all_cells_y >= y_min - 0.2) & (all_cells_y <= y_max + 0.2)
    )
    vis_x = all_cells_x[in_fov]
    vis_y = all_cells_y[in_fov]
    vis_r = all_cells_r[in_fov]

    px = ((vis_x - x_min) / fov_w) * w
    py = ((vis_y - y_min) / fov_h) * h

    defocus = abs(stage_z - best_z)
    blur_radius = max(0.1, defocus * 8.0)

    chan_key = channel.lower()

    if any(k in chan_key for k in ["brightfield", "bf", "phase"]):
        base = Image.new("RGB", (w, h), (215, 220, 215))
        draw = ImageDraw.Draw(base)

        for wx, wy, _ in wells:
            w_px = ((wx - x_min) / fov_w) * w
            w_py = ((wy - y_min) / fov_h) * h
            w_rad_px = (3.2 / fov_w) * w
            draw.ellipse(
                [w_px - w_rad_px, w_py - w_rad_px, w_px + w_rad_px, w_py + w_rad_px],
                outline=(140, 150, 140),
                width=4
            )

        for x, y, r in zip(px, py, vis_r):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(50, 60, 50), width=3, fill=(175, 190, 180))
            nr = r * 0.4
            draw.ellipse([x - nr, y - nr, x + nr, y + nr], fill=(120, 135, 125))

    elif any(k in chan_key for k in ["blue", "dapi"]):
        base = Image.new("RGB", (w, h), (8, 10, 22))
        draw = ImageDraw.Draw(base)
        for x, y, r in zip(px, py, vis_r):
            nr = r * 0.45
            draw.ellipse([x - nr, y - nr, x + nr, y + nr], fill=(40, 180, 255))

    elif any(k in chan_key for k in ["green", "fitc", "gfp"]):
        base = Image.new("RGB", (w, h), (8, 18, 10))
        draw = ImageDraw.Draw(base)
        for x, y, r in zip(px, py, vis_r):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(30, 240, 80))

    elif any(k in chan_key for k in ["red", "txred", "mcherry", "cy5"]):
        base = Image.new("RGB", (w, h), (18, 8, 10))
        draw = ImageDraw.Draw(base)
        for x, y, r in zip(px, py, vis_r):
            draw.ellipse([x - r * 0.7, y - r * 0.7, x + r * 0.7, y + r * 0.7], fill=(255, 50, 60))

    else:
        base = Image.new("RGB", (w, h), (12, 12, 15))
        draw = ImageDraw.Draw(base)
        for x, y, r in zip(px, py, vis_r):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(220, 220, 230), width=2)

    if blur_radius > 0.5:
        base = base.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return np.array(base)


def generate_simulated_frame(
    z_pos: float = 0.0,
    best_z: float = 2.5,
    sensor_size: Tuple[int, int] = (512, 512),
    binning: str = "2x2"
) -> np.ndarray:
    return generate_well_plate_cell_frame(
        stage_x=0.0,
        stage_y=0.0,
        stage_z=z_pos,
        best_z=best_z,
        channel="Brightfield",
        sensor_size=sensor_size
    )
