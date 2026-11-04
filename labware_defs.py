"""Labware definitions for the plate navigator.

Built-in ANSI/SLAS plate designs plus user-defined labware persisted to
data/custom_labware.json. Wells can be round or square. Users can
duplicate any design and edit the copy; calibration results can be
written back into a custom design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, replace

CUSTOM_LABWARE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'data', 'custom_labware.json')


@dataclass
class LabwareDef:
    name: str
    rows: int
    cols: int
    spacing_x_mm: float
    spacing_y_mm: float
    well_size_x_mm: float          # diameter (round) or width (square)
    well_size_y_mm: float          # equals size_x for round wells
    shape: str = 'round'           # 'round' | 'square'
    a1_offset_x_mm: float = 14.38  # A1 center from plate left edge
    a1_offset_y_mm: float = 11.24  # A1 center from plate top edge
    plate_length_mm: float = 127.76
    plate_width_mm: float = 85.48
    builtin: bool = False

    # Compatibility helpers (plate map widget / app code)

    @property
    def well_spacing_mm(self) -> float:
        return self.spacing_x_mm

    @property
    def well_diameter(self) -> float:
        return self.well_size_x_mm

    @property
    def a1_offset_x(self) -> float:
        return self.a1_offset_x_mm

    @property
    def a1_offset_y(self) -> float:
        return self.a1_offset_y_mm

    def get_well_name(self, row: int, col: int) -> str:
        # Support >26 rows gracefully (AA, AB, ...)
        if row < 26:
            prefix = chr(65 + row)
        else:
            prefix = chr(64 + row // 26) + chr(65 + row % 26)
        return f"{prefix}{col + 1}"

    def well_center_rel_a1_mm(self, row: int, col: int) -> tuple[float, float]:
        """Well center relative to the A1 center (+x right, +y down), mm."""
        return col * self.spacing_x_mm, row * self.spacing_y_mm

    def well_center_plate_mm(self, row: int, col: int) -> tuple[float, float]:
        """Well center relative to the plate top-left corner, mm."""
        return (self.a1_offset_x_mm + col * self.spacing_x_mm,
                self.a1_offset_y_mm + row * self.spacing_y_mm)


def builtin_labware() -> dict[str, LabwareDef]:
    defs = [
        LabwareDef('96-well plate', 8, 12, 9.0, 9.0, 6.4, 6.4, 'round',
                   14.38, 11.24, builtin=True),
        LabwareDef('96-well plate (square wells)', 8, 12, 9.0, 9.0, 7.8, 7.8,
                   'square', 14.38, 11.24, builtin=True),
        LabwareDef('384-well plate', 16, 24, 4.5, 4.5, 3.3, 3.3, 'round',
                   12.13, 8.99, builtin=True),
        LabwareDef('384-well plate (square wells)', 16, 24, 4.5, 4.5, 3.7, 3.7,
                   'square', 12.13, 8.99, builtin=True),
        LabwareDef('48-well plate', 6, 8, 13.0, 13.0, 11.37, 11.37, 'round',
                   18.16, 10.08, builtin=True),
        LabwareDef('24-well plate', 4, 6, 19.3, 19.3, 15.6, 15.6, 'round',
                   14.38, 13.49, builtin=True),
        LabwareDef('12-well plate', 3, 4, 26.01, 26.01, 22.11, 22.11, 'round',
                   24.94, 16.79, builtin=True),
        LabwareDef('6-well plate', 2, 3, 39.12, 39.12, 34.8, 34.8, 'round',
                   24.55, 23.16, builtin=True),
        LabwareDef('Slide 75x25 mm', 1, 1, 1.0, 1.0, 22.0, 60.0, 'square',
                   63.88, 42.74, builtin=True),
    ]
    return {d.name: d for d in defs}


def load_custom_labware() -> dict[str, LabwareDef]:
    try:
        with open(CUSTOM_LABWARE_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        out = {}
        for item in raw:
            item.pop('builtin', None)
            d = LabwareDef(builtin=False, **item)
            out[d.name] = d
        return out
    except Exception:
        return {}


def save_custom_labware(customs: dict[str, LabwareDef]) -> None:
    os.makedirs(os.path.dirname(CUSTOM_LABWARE_PATH), exist_ok=True)
    payload = []
    for d in customs.values():
        item = asdict(d)
        item.pop('builtin', None)
        payload.append(item)
    with open(CUSTOM_LABWARE_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


class LabwareLibrary:
    """All labware: built-ins plus user customs (persisted)."""

    def __init__(self):
        self.builtins = builtin_labware()
        self.customs = load_custom_labware()

    def names(self) -> list[str]:
        return list(self.builtins.keys()) + list(self.customs.keys())

    def get(self, name: str) -> LabwareDef | None:
        return self.builtins.get(name) or self.customs.get(name)

    def duplicate(self, name: str, new_name: str | None = None) -> LabwareDef | None:
        src = self.get(name)
        if src is None:
            return None
        if not new_name:
            new_name = f'{src.name} (copy)'
            i = 2
            while self.get(new_name) is not None:
                new_name = f'{src.name} (copy {i})'
                i += 1
        dup = replace(src, name=new_name, builtin=False)
        self.customs[new_name] = dup
        save_custom_labware(self.customs)
        return dup

    def save(self, d: LabwareDef) -> None:
        """Create or update a custom design (built-ins stay immutable)."""
        d.builtin = False
        self.customs[d.name] = d
        save_custom_labware(self.customs)

    def delete(self, name: str) -> bool:
        if name in self.customs:
            del self.customs[name]
            save_custom_labware(self.customs)
            return True
        return False
