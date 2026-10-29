"""
luma_sim.py - Offline Hardware Simulator for LumaViewPro 4.0.0-beta.

Mocks camera acquisition (pyloncamera), stage motion (motorboard), and illumination (ledboard)
with real-time synthetic cell colony generation across multi-well microplates.
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
        self.exposure_ms = 20.0
        self.gain = 1.0
        self.stage_x = -49.5  # Well A1 default
        self.stage_y = 31.5
        self.stage_z = 0.0
        self.best_z = 0.0
        self.active_channel = "Brightfield"
        self.is_light_on = False  # Starts OFF (dark sensor noise)
        self.led_power = 100.0
        self.fov_size_mm = (0.60, 0.45)
        self.num_wells = 96

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def get_frame(self):
        return image_utils.generate_well_plate_cell_frame(
            stage_x=self.stage_x,
            stage_y=self.stage_y,
            stage_z=self.stage_z,
            best_z=self.best_z,
            channel=self.active_channel,
            is_light_on=self.is_light_on,
            exposure_ms=self.exposure_ms,
            gain=self.gain,
            led_power=self.led_power,
            fov_size_mm=self.fov_size_mm,
            num_wells=self.num_wells,
            sensor_size=(512, 512)
        )

    def set_exposure(self, exp: float):
        self.exposure_ms = float(exp)

    def set_gain(self, g: float):
        self.gain = float(g)

    def set_light_on(self, on: bool):
        self.is_light_on = bool(on)

    def set_led_power(self, power: float):
        self.led_power = float(power)

    def set_channel(self, channel: str):
        self.active_channel = str(channel)


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

    def move_abs(self, x: float, y: float, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def move_rel(self, dx: float, dy: float, dz: float = 0.0):
        self.x += float(dx)
        self.y += float(dy)
        self.z += float(dz)

    def get_pos(self):
        return (self.x, self.y, self.z)


mock_motor = types.ModuleType("motorboard")
mock_motor.MotorBoard = MockStage
sys.modules["motorboard"] = mock_motor


class MockLEDBoard:
    def __init__(self):
        self.active_channel = "Brightfield"
        self.power = 100
        self.is_on = False

    def set_led(self, channel: str, power: int):
        self.active_channel = channel
        self.power = power
        self.is_on = True

    def off(self):
        self.is_on = False


mock_led = types.ModuleType("ledboard")
mock_led.LEDBoard = MockLEDBoard
sys.modules["ledboard"] = mock_led

print("[LumaViewPro Simulator] Hardware Mocks Ready. Starting LumaViewPro...")

import lumaviewpro

if __name__ == "__main__":
    lumaviewpro.LumaViewProApp().run()
