"""
luma_sim.py - Offline Hardware Simulator for LumaViewPro 4.0.0-beta

Mocks camera acquisition (pyloncamera), stage motion (motorboard), and illumination (ledboard)
with real-time synthetic cell colony generation across SBS 96-well microplates.
Launches the full, stock LumaViewPro UI offline so development can proceed seamlessly on desktop PCs.
"""

from __future__ import annotations
import sys
import os
import types

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import image_utils

print("[LumaViewPro Simulator] Initializing offline hardware & camera mocks...")


class MockCamera:
    def __init__(self):
        self.is_connected = True
        self.exposure_ms = 100.0
        self.gain = 1.0

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def get_frame(self):
        return None

    def set_exposure(self, exp: float):
        self.exposure_ms = float(exp)

    def set_gain(self, g: float):
        self.gain = float(g)


mock_pylon = types.ModuleType("pyloncamera")
mock_pylon.PylonCamera = MockCamera
sys.modules["pyloncamera"] = mock_pylon


class MockStage:
    def __init__(self):
        self.x = -49.5
        self.y = 31.5
        self.z = 0.0

    def connect(self) -> bool:
        return True

    def move_abs(self, x: float, y: float, z: float):
        self.x, self.y, self.z = x, y, z

    def move_rel(self, dx: float, dy: float, dz: float):
        self.x += dx
        self.y += dy
        self.z += dz

    def get_pos(self):
        return (self.x, self.y, self.z)


mock_motor = types.ModuleType("motorboard")
mock_motor.MotorBoard = MockStage
sys.modules["motorboard"] = mock_motor


class MockLEDBoard:
    def __init__(self):
        self.active_channel = "Brightfield"
        self.power = 100

    def set_led(self, channel: str, power: int):
        self.active_channel = channel
        self.power = power

    def off(self):
        self.power = 0


mock_led = types.ModuleType("ledboard")
mock_led.LEDBoard = MockLEDBoard
sys.modules["ledboard"] = mock_led

print("[LumaViewPro Simulator] Hardware Mocks Ready. Starting LumaViewPro...")

import lumaviewpro

if __name__ == "__main__":
    lumaviewpro.LumaViewProApp().run()
