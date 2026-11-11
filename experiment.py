"""Experiment planning: where to image inside each well, where to map
focus, and how the acquired files are named.

Pure geometry and naming, no hardware and no UI. Offsets are millimetres
from the well center in the plate frame (+x toward the last column, +y
toward the last row); the caller converts them to stage coordinates
through the plate calibration.
"""

from __future__ import annotations

import math
import random
import re

# Sampling modes offered by the setup wizard.
SAMPLING_MODES = ('single', 'tiles', 'random', 'random_same', 'coverage')
SAMPLING_LABELS = {
    'single': 'Single image at the well center',
    'tiles': 'Tile grid (rows x columns)',
    'random': 'Random points, different in every well',
    'random_same': 'Random points, same pattern in every well',
    'coverage': 'Well coverage (% of well area)',
}

FOCUS_MODES = ('corners', 'per_well', 'multi', 'none')
FOCUS_LABELS = {
    'corners': 'Four corners + center of the selected wells',
    'per_well': 'One point in every selected well',
    'multi': 'Several points in every selected well',
    'none': 'No focus map (AF point focus everywhere)',
}

Offset = tuple[str, float, float]       # (label, dx_mm, dy_mm)


# ------------------------------------------------------------- labels

def row_letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet style)."""
    out = ''
    n = int(index) + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def tile_label(row: int, col: int) -> str:
    """Tile designation: TA1, TA2, ... along the first line of tiles, then
    TB1, TB2, ... along the second."""
    return f'T{row_letters(row)}{col + 1}'


def point_label(index: int) -> str:
    return f'P{index + 1}'


def _clean(token: str) -> str:
    return re.sub(r'[^A-Za-z0-9-]+', '', str(token))


def file_name(well: str | None, site: str, z_index: int, channel: str,
              timepoint: int, ext: str = 'tiff') -> str:
    """``B3_TA1_Z001_GREEN_T0001.tiff``: well, tile/position, Z plane,
    colour, timepoint - in that order. The well is left out when there
    is none (no plate definition, single dish or slide). Z planes and
    timepoints count from 1."""
    parts = [_clean(well)] if well else []
    parts += [_clean(site) or 'P1', f'Z{int(z_index):03d}',
              _clean(channel), f'T{int(timepoint):04d}']
    return '_'.join(parts) + '.' + ext


def assign_designations(positions: list[dict], well_of=None) -> list[dict]:
    """Give every position a ``well`` and a ``site`` for file naming.

    Positions that already carry them are kept. Otherwise the well comes
    from ``well_of(position)`` (None/absent callback = no wells: dish,
    slide or no plate definition) and the site is the next free P-number
    inside that well. Returns new dicts; the input is not modified."""
    out = []
    used: dict[str | None, set[str]] = {}
    for p in positions:
        q = dict(p)
        if 'well' not in q:
            q['well'] = well_of(p) if well_of else None
        if well_of is None:
            q['well'] = None
        used.setdefault(q['well'], set())
        if q.get('site'):
            used[q['well']].add(q['site'])
        out.append(q)
    for q in out:
        taken = used[q['well']]
        if not q.get('site'):
            i = 0
            while point_label(i) in taken:
                i += 1
            q['site'] = point_label(i)
            taken.add(q['site'])
    return out


# ------------------------------------------------------- sample points

def tile_grid(rows: int, cols: int, fov_x_mm: float, fov_y_mm: float,
              overlap_pct: float) -> list[Offset]:
    """Centered rows x cols grid, visited in serpentine order. Labels
    follow the grid (TA1 is the top-left tile), not the visiting order."""
    rows, cols = max(int(rows), 1), max(int(cols), 1)
    f = 1.0 - min(max(overlap_pct, 0.0), 90.0) / 100.0
    sx, sy = fov_x_mm * f, fov_y_mm * f
    out = []
    for r in range(rows):
        order = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in order:
            out.append((tile_label(r, c),
                        (c - (cols - 1) / 2.0) * sx,
                        (r - (rows - 1) / 2.0) * sy))
    return out


def _tiles_to_span(length_mm: float, fov_mm: float, step_mm: float) -> int:
    """Fewest tiles whose union is at least ``length_mm`` long."""
    if length_mm <= fov_mm or step_mm <= 0:
        return 1
    return int(math.ceil((length_mm - fov_mm) / step_mm - 1e-9)) + 1


def _rect_touches_region(cx: float, cy: float, hw: float, hh: float,
                         a: float, b: float, shape: str) -> bool:
    """Does the tile (center cx, cy; half-size hw, hh) touch the region
    (half-size a, b; ellipse for round wells, rectangle otherwise)?"""
    if shape == 'square':
        return abs(cx) < a + hw and abs(cy) < b + hh
    if a <= 0 or b <= 0:
        return abs(cx) <= hw and abs(cy) <= hh
    # Nearest point of the tile to the well center, in unit-circle space.
    nx = min(max(0.0, cx - hw), cx + hw) / a
    ny = min(max(0.0, cy - hh), cy + hh) / b
    return nx * nx + ny * ny < 1.0


def coverage_tiles(well_x_mm: float, well_y_mm: float, shape: str,
                   fov_x_mm: float, fov_y_mm: float, overlap_pct: float,
                   coverage_pct: float) -> list[Offset]:
    """Tiles covering ``coverage_pct`` of the well area, centered.

    The covered region is the well scaled down to that fraction of its
    area. The grid always spans at least the whole region, so 100 %
    overfills the well: every part of it is imaged, with the outer tiles
    reaching past the rim. Tiles that do not touch the region (the
    corners of the grid on a round well) are left out."""
    k = math.sqrt(min(max(coverage_pct, 0.0), 100.0) / 100.0)
    reg_x, reg_y = well_x_mm * k, well_y_mm * k
    f = 1.0 - min(max(overlap_pct, 0.0), 90.0) / 100.0
    sx, sy = fov_x_mm * f, fov_y_mm * f
    cols = _tiles_to_span(reg_x, fov_x_mm, sx)
    rows = _tiles_to_span(reg_y, fov_y_mm, sy)
    out = []
    for r in range(rows):
        order = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in order:
            dx = (c - (cols - 1) / 2.0) * sx
            dy = (r - (rows - 1) / 2.0) * sy
            if rows * cols == 1 or _rect_touches_region(
                    dx, dy, fov_x_mm / 2.0, fov_y_mm / 2.0,
                    reg_x / 2.0, reg_y / 2.0, shape):
                out.append((tile_label(r, c), dx, dy))
    return out


def random_points(n: int, well_x_mm: float, well_y_mm: float, shape: str,
                  fov_x_mm: float, fov_y_mm: float,
                  rng: random.Random) -> list[Offset]:
    """``n`` random image centers inside the well, kept half a field of
    view away from the rim and (where there is room) not overlapping
    each other. Visited nearest-neighbour first; labelled P1.. in that
    order."""
    a = max(well_x_mm / 2.0 - fov_x_mm / 2.0, 0.0)
    b = max(well_y_mm / 2.0 - fov_y_mm / 2.0, 0.0)
    pts: list[tuple[float, float]] = []
    for _ in range(max(int(n), 1)):
        cand = (0.0, 0.0)
        for _attempt in range(200):
            if shape == 'square':
                cand = (rng.uniform(-a, a), rng.uniform(-b, b))
            else:
                rad = math.sqrt(rng.random())
                ang = rng.uniform(0.0, 2.0 * math.pi)
                cand = (a * rad * math.cos(ang), b * rad * math.sin(ang))
            if all(abs(cand[0] - p[0]) >= fov_x_mm or
                   abs(cand[1] - p[1]) >= fov_y_mm for p in pts):
                break
        pts.append(cand)
    ordered = [pts.pop(0)]
    while pts:
        last = ordered[-1]
        nxt = min(pts, key=lambda p: math.hypot(p[0] - last[0],
                                                p[1] - last[1]))
        pts.remove(nxt)
        ordered.append(nxt)
    return [(point_label(i), x, y) for i, (x, y) in enumerate(ordered)]


def plan_sites(mode: str, wells: list[tuple[int, int]], *,
               well_x_mm: float, well_y_mm: float, shape: str,
               fov_x_mm: float, fov_y_mm: float,
               rows: int = 1, cols: int = 1, overlap_pct: float = 10.0,
               n_random: int = 3, coverage_pct: float = 100.0,
               seed: int = 0) -> dict[tuple[int, int], list[Offset]]:
    """Imaging sites for each well as {(row, col): [(label, dx, dy)]}."""
    if mode == 'tiles':
        pattern = tile_grid(rows, cols, fov_x_mm, fov_y_mm, overlap_pct)
    elif mode == 'coverage':
        pattern = coverage_tiles(well_x_mm, well_y_mm, shape, fov_x_mm,
                                 fov_y_mm, overlap_pct, coverage_pct)
    elif mode == 'random_same':
        pattern = random_points(n_random, well_x_mm, well_y_mm, shape,
                                fov_x_mm, fov_y_mm, random.Random(seed))
    elif mode == 'random':
        return {w: random_points(n_random, well_x_mm, well_y_mm, shape,
                                 fov_x_mm, fov_y_mm,
                                 random.Random(f'{seed}-{w[0]}-{w[1]}'))
                for w in wells}
    else:
        pattern = [(point_label(0), 0.0, 0.0)]
    return {w: list(pattern) for w in wells}


# -------------------------------------------------------- focus points

def focus_points(mode: str, wells: list[tuple[int, int]], *,
                 well_x_mm: float, well_y_mm: float,
                 n_per_well: int = 3
                 ) -> list[tuple[tuple[int, int], float, float]]:
    """Where to set focus for the focus map: [((row, col), dx, dy)].

    - ``corners``: the four corner wells of the block spanned by the
      selected wells, plus its middle well (fewer when the block is a
      single row, column or well).
    - ``per_well``: the center of every selected well.
    - ``multi``: ``n_per_well`` points in every selected well, evenly
      spaced on a ring 60 % of the way from the center to the rim.
    """
    wells = sorted(set(wells))
    if not wells or mode == 'none':
        return []
    if mode == 'corners':
        r0, r1 = min(w[0] for w in wells), max(w[0] for w in wells)
        c0, c1 = min(w[1] for w in wells), max(w[1] for w in wells)
        picks = [(r0, c0), (r0, c1), (r1, c1), (r1, c0),
                 ((r0 + r1) // 2, (c0 + c1) // 2)]
        seen, out = set(), []
        for w in picks:
            if w not in seen:
                seen.add(w)
                out.append((w, 0.0, 0.0))
        return out
    if mode == 'multi' and n_per_well > 1:
        out = []
        for w in wells:
            for i in range(int(n_per_well)):
                ang = math.pi / 4.0 + 2.0 * math.pi * i / int(n_per_well)
                out.append((w, 0.3 * well_x_mm * math.cos(ang),
                            0.3 * well_y_mm * math.sin(ang)))
        return out
    return [(w, 0.0, 0.0) for w in wells]


# ------------------------------------------------------- focus surface

def interpolate_offsets(points: list[tuple[float, float, float]],
                        x: float, y: float) -> float | None:
    """Focus offset at (x, y) from scattered (x, y, offset) points.

    One or two points: the nearest point's offset. Three: the plane
    through them. Four or more: the best-fit plane (the plate's overall
    tilt, which is what carries the extrapolation beyond the outermost
    points) plus an inverse-distance blend of each point's deviation from
    that plane - so the surface passes exactly through every focus point
    and follows the local gradients between them."""
    import numpy as np
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) < 3:
        d = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
        return float(arr[int(np.argmin(d)), 2])
    cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
    A = np.column_stack([arr[:, 0] - cx, arr[:, 1] - cy, np.ones(len(arr))])
    coef, *_ = np.linalg.lstsq(A, arr[:, 2], rcond=None)
    plane = float(coef[0] * (x - cx) + coef[1] * (y - cy) + coef[2])
    if len(arr) == 3:
        return plane
    resid = arr[:, 2] - A @ coef
    d2 = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2
    near = int(np.argmin(d2))
    if d2[near] < 1e-6:
        return plane + float(resid[near])
    w = 1.0 / d2
    return plane + float((w * resid).sum() / w.sum())
