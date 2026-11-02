import os
import sys
import enum

driver_paths = [
    r"C:\Program Files (x86)\LumaViewPro-stock",
    r"C:\Program Files (x86)\Etaluma\LumaViewPro",
    r"C:\Program Files (x86)\LumaViewPro-main",
    os.path.dirname(os.path.abspath(__file__))
]
for p in driver_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

if not hasattr(enum, 'StrEnum'):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

print("=== LumaViewPro Stock Driver Environment Initialized ===")
