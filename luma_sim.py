"""
luma_sim.py - Offline Hardware Simulator for LumaViewPro 4.0.0-beta

Mocks camera acquisition (pyloncamera) and stage motion (motorboard / ledboard)
so LumaViewPro and Plate Navigator can be developed and tested offline without
triggering "No Microscope Detected" hardware failure popups.
"""

import sys
import os
import types

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

print("[LumaViewPro Simulator] Initializing offline hardware & camera mocks...")

# Create mock module for pyloncamera / Basler camera driver
mock_pylon = types.ModuleType("pyloncamera")
class MockCamera:
    def __init__(self):
        self.is_connected = False
        self.exposure_ms = 100.0
        self.gain = 1.0
    def connect(self): return False
    def disconnect(self): pass
    def get_frame(self): return None
    def set_exposure(self, exp): self.exposure_ms = exp
    def set_gain(self, g): self.gain = g

mock_pylon.PylonCamera = MockCamera
sys.modules["pyloncamera"] = mock_pylon

# Create mock module for motorboard / stage motion driver
mock_motor = types.ModuleType("motorboard")
class MockStage:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
    def connect(self): return True
    def move_abs(self, x, y, z): self.x, self.y, self.z = x, y, z
    def move_rel(self, dx, dy, dz): self.x += dx; self.y += dy; self.z += dz
    def get_pos(self): return (self.x, self.y, self.z)

mock_motor.MotorBoard = MockStage
sys.modules["motorboard"] = mock_motor

# Create mock module for ledboard / illumination controller
mock_led = types.ModuleType("ledboard")
class MockLEDBoard:
    def __init__(self):
        self.active_channel = "BF"
        self.power = 0
    def set_led(self, channel, power): self.active_channel = channel; self.power = power
    def off(self): self.power = 0

mock_led.LEDBoard = MockLEDBoard
sys.modules["ledboard"] = mock_led

print("[LumaViewPro Simulator] Mocks loaded successfully. Launching LumaViewPro...")

import lumaviewpro

if __name__ == "__main__":
    lumaviewpro.LumaViewProApp().run()
