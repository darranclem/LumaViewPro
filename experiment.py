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
    """``B3_TA1_Z001_GREEN_t0001.tiff``: well, tile/position, Z plane,
    colour, timepoint - in that order. Tiles use an upper-case T, the
    timepoint a lower-case t. The well is left out when there is none
    (no plate definition, single dish or slide). Z planes and timepoints
    count from 1."""
    parts = [_clean(well)] if well else []
    parts += [_clean(site) or 'P1', f'Z{int(z_index):03d}',
              _clean(channel), f't{int(timepoint):04d}']
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


# ------------------------------------- editing positions inside a well

_TILE_SITE = re.compile(r'^(?:(R\d+)-)?T[A-Z]+\d+$')


def position_name(well: str | None, site: str) -> str:
    return f'{well}_{site}' if well else site


def next_point_site(positions: list[dict], well: str | None) -> str:
    """The next free point tag (P1, P2, ...) in a well: one more than the
    highest in use there, so tags are never reused or reshuffled."""
    top = 0
    for p in positions:
        if p.get('well') == well:
            m = re.match(r'^P(\d+)$', str(p.get('site') or ''))
            if m:
                top = max(top, int(m.group(1)))
    return point_label(top)


def add_point(positions: list[dict], well: str | None, x_um: float,
              y_um: float, z_um: float) -> tuple[list[dict], dict]:
    """Append a point that takes the name of its well and the next point
    tag there. ``positions`` must carry their designations (see
    :func:`assign_designations`). Returns the new list and the point."""
    site = next_point_site(positions, well)
    point = {'name': position_name(well, site), 'well': well, 'site': site,
             'x_um': float(x_um), 'y_um': float(y_um), 'z_um': float(z_um)}
    return positions + [point], point


def _tile_groups(positions: list[dict], well: str | None
                 ) -> dict[str | None, list[int]]:
    """Tile scans in a well as {region: [indices]}."""
    groups: dict[str | None, list[int]] = {}
    for i, p in enumerate(positions):
        if p.get('well') == well:
            m = _TILE_SITE.match(str(p.get('site') or ''))
            if m:
                groups.setdefault(m.group(1), []).append(i)
    return groups


def move_in_well(positions: list[dict], well: str | None, x_um: float,
                 y_um: float, z_of=None) -> tuple[list[dict], str]:
    """Move what is nearest to (x_um, y_um) in a well there, keeping every
    designation: a single position moves by itself; a tile scan moves as
    a whole, so that its center (the middle of its outermost tiles) lands
    on the point. ``z_of(x, y, old_z)`` gives the focus at a new place
    (default: unchanged). Returns the new list and what was moved ('' =
    the well holds no position)."""
    out = [dict(p) for p in positions]
    cands = []                 # (distance, indices, center, description)
    tiles: set[int] = set()
    for region, idx in _tile_groups(out, well).items():
        tiles.update(idx)
        xs = [out[i]['x_um'] for i in idx]
        ys = [out[i]['y_um'] for i in idx]
        c = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        what = ' '.join(t for t in (well, region, 'tile scan',
                                    f'({len(idx)} tiles)') if t)
        cands.append((math.hypot(c[0] - x_um, c[1] - y_um), idx, c, what))
    for i, p in enumerate(out):
        if p.get('well') == well and i not in tiles:
            c = (p['x_um'], p['y_um'])
            cands.append((math.hypot(c[0] - x_um, c[1] - y_um), [i], c,
                          p.get('name') or position_name(well,
                                                         p.get('site') or '')))
    if not cands:
        return out, ''
    _, idx, c, what = min(cands, key=lambda t: t[0])
    dx, dy = x_um - c[0], y_um - c[1]
    for i in idx:
        p = out[i]
        p['x_um'] += dx
        p['y_um'] += dy
        if z_of is not None:
            p['z_um'] = float(z_of(p['x_um'], p['y_um'], p['z_um']))
    return out, what


# ------------------------------------------- do the moves fit the lens


def tile_step_um(positions: list[dict]) -> float | None:
    """Distance the stage moves between neighbouring tiles: the median
    nearest-neighbour distance inside each tile scan (positions whose
    site is a tile label, grouped by well and region). None when the
    positions hold no tile scan."""
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for p in positions:
        m = _TILE_SITE.match(str(p.get('site') or ''))
        if m:
            groups.setdefault((p.get('well'), m.group(1)), []).append(
                (float(p['x_um']), float(p['y_um'])))
    near = []
    for pts in groups.values():
        if len(pts) < 2:
            continue
        for i, (x, y) in enumerate(pts):
            near.append(min(math.hypot(x - u, y - v)
                            for j, (u, v) in enumerate(pts) if j != i))
    if not near:
        return None
    near.sort()
    return near[len(near) // 2]


def lens_fit(step_um: float | None, fov_um: tuple[float, float]
             ) -> tuple[str, str]:
    """Does the tile step suit the field of view of the lens in use?
    Returns (verdict, text); verdict is 'none' (no tile scan), 'ok',
    'gaps' (the stage moves further than one image: parts of the sample
    are missed) or 'dense' (the images overlap by more than half: the
    positions were laid out for a higher magnification)."""
    if step_um is None:
        return 'none', 'No tile scans: every position is a single field.'
    side = min(fov_um)
    if side <= 0:
        return 'none', 'Field of view not known.'
    overlap = 1.0 - step_um / side
    head = (f'Tiles are {step_um / 1000:.3f} mm apart; one image is '
            f'{side / 1000:.3f} mm across')
    if overlap < -1.0:
        return 'gaps', (f'{head}: the stage moves {step_um / side:.1f} '
                        f'fields per step, so most of the sample is '
                        f'missed with this lens.')
    if overlap < -0.005:
        return 'gaps', (f'{head}: GAPS of {-overlap * 100:.0f}% between '
                        f'the tiles with this lens.')
    if overlap > 0.5:
        return 'dense', (f'{head}: the tiles overlap by '
                         f'{overlap * 100:.0f}%, far more than needed '
                         f'with this lens.')
    return 'ok', f'{head}: {overlap * 100:.0f}% overlap.'


def lens_scale_check(measured_um_per_px: float | None,
                     expected_um_per_px: float, tolerance: float = 0.15
                     ) -> tuple[str, str]:
    """Compare the image scale measured by moving the stage with the
    scale of the objective selected. Returns (verdict, text); verdict is
    'unknown', 'ok' or 'wrong'."""
    if not measured_um_per_px or expected_um_per_px <= 0:
        return 'unknown', ('Image scale not measured with this lens '
                           'selected (the lens cannot be confirmed).')
    ratio = measured_um_per_px / expected_um_per_px
    text = (f'Image scale measured {measured_um_per_px:.3f} um/px; the '
            f'selected objective should give {expected_um_per_px:.3f}')
    if abs(ratio - 1.0) <= tolerance:
        return 'ok', text + ': the lens matches.'
    return 'wrong', (text + f'. The lens on the microscope magnifies '
                     f'{1.0 / ratio:.2f} times what is selected: it is '
                     f'NOT the selected objective.')


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
                  fov_x_mm: float, fov_y_mm: float, rng: random.Random,
                  allow_overlap: bool = False) -> list[Offset]:
    """``n`` random image centers inside the well, kept half a field of
    view away from the rim.

    By default no two fields of view overlap; when the well has no room
    for ``n`` such fields, FEWER points are returned (the caller reports
    it) rather than overlapping ones. ``allow_overlap`` drops the check.
    Visited nearest-neighbour first; labelled P1.. in that order."""
    a = max(well_x_mm / 2.0 - fov_x_mm / 2.0, 0.0)
    b = max(well_y_mm / 2.0 - fov_y_mm / 2.0, 0.0)
    pts: list[tuple[float, float]] = []
    for _ in range(max(int(n), 1)):
        placed = False
        for _attempt in range(1 if allow_overlap else 2000):
            if shape == 'square':
                cand = (rng.uniform(-a, a), rng.uniform(-b, b))
            else:
                rad = math.sqrt(rng.random())
                ang = rng.uniform(0.0, 2.0 * math.pi)
                cand = (a * rad * math.cos(ang), b * rad * math.sin(ang))
            if allow_overlap or all(abs(cand[0] - p[0]) >= fov_x_mm or
                                    abs(cand[1] - p[1]) >= fov_y_mm
                                    for p in pts):
                pts.append(cand)
                placed = True
                break
        if not placed:
            break                       # the well is full
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
               seed: int = 0, allow_overlap: bool = False
               ) -> dict[tuple[int, int], list[Offset]]:
    """Imaging sites for each well as {(row, col): [(label, dx, dy)]}."""
    if mode == 'tiles':
        pattern = tile_grid(rows, cols, fov_x_mm, fov_y_mm, overlap_pct)
    elif mode == 'coverage':
        pattern = coverage_tiles(well_x_mm, well_y_mm, shape, fov_x_mm,
                                 fov_y_mm, overlap_pct, coverage_pct)
    elif mode == 'random_same':
        pattern = random_points(n_random, well_x_mm, well_y_mm, shape,
                                fov_x_mm, fov_y_mm, random.Random(seed),
                                allow_overlap)
    elif mode == 'random':
        return {w: random_points(n_random, well_x_mm, well_y_mm, shape,
                                 fov_x_mm, fov_y_mm,
                                 random.Random(f'{seed}-{w[0]}-{w[1]}'),
                                 allow_overlap)
                for w in wells}
    else:
        pattern = [(point_label(0), 0.0, 0.0)]
    return {w: list(pattern) for w in wells}


# ------------------------------------------------------------ z-stacks

# Wavelength used for the optical-section estimate: emission for the
# fluorescence channels, mid-spectrum for transmitted white light.
CHANNEL_WAVELENGTH_NM = {'BF': 550.0, 'PC': 550.0, 'DF': 550.0,
                         'BLUE': 450.0, 'GREEN': 520.0, 'RED': 610.0}

NYQUIST_SAMPLING = 2.3

# Typical dry-objective NA by magnification, only to pre-fill the field.
TYPICAL_NA = {2: 0.06, 4: 0.13, 10: 0.30, 20: 0.45, 40: 0.65, 60: 0.80,
              100: 0.90}


def typical_na(magnification: float) -> float:
    return TYPICAL_NA[min(TYPICAL_NA, key=lambda m: abs(m - magnification))]


def optical_section_um(na: float, wavelength_nm: float, magnification: float,
                       pixel_um: float, medium_index: float = 1.0) -> float:
    """Depth of field of a widefield objective (the "optical section"):

        n * wavelength / NA^2  +  n * pixel / (M * NA)

    the diffraction term plus the geometric term from the camera pixel
    size projected into the sample."""
    na = max(float(na), 1e-3)
    mag = max(float(magnification), 1e-3)
    lam_um = wavelength_nm / 1000.0
    return (medium_index * lam_um / na ** 2
            + medium_index * pixel_um / (mag * na))


def stack_step_um(section_um: float, sampling: float = 1.0,
                  override_um: float | None = None) -> float:
    """Z interval: the optical section divided by the sampling rate (1 =
    one section per step, 2.3 = Nyquist), unless a step is typed in."""
    if override_um is not None and override_um > 0:
        return float(override_um)
    return section_um / max(float(sampling), 0.1)


def stack_offsets(range_um: float, step_um: float) -> list[float]:
    """Plane offsets from the stack center, top first: +range/2 down to
    -range/2. The center plane is always included and the planes are
    symmetric about it; when the range is not a whole number of steps the
    stack is extended by part of a step at each end rather than cut
    short."""
    half = abs(range_um) / 2.0
    step = max(abs(step_um), 0.01)
    n = int(-(-(half - 1e-6) // step)) if half > 1e-6 else 0
    return [i * step for i in range(n, -n - 1, -1)]


def stack_between(begin_um: float, end_um: float, step_um: float
                  ) -> list[float]:
    """Planes of a z-stack defined by its two ends, in the order begin to
    end. Both ends are planes and the planes are evenly spaced: when the
    span is not a whole number of intervals the interval is made slightly
    smaller, never larger, so nothing is sampled more coarsely than asked."""
    span = float(end_um) - float(begin_um)
    step = max(abs(float(step_um)), 0.01)
    n = max(int(math.ceil(abs(span) / step - 1e-6)), 1) \
        if abs(span) > 1e-6 else 0
    if n == 0:
        return [float(begin_um)]
    return [float(begin_um) + span * i / n for i in range(n + 1)]


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
