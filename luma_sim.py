from __future__ import annotations
"""
luma_sim.py - Offline Hardware Simulator for LumaViewPro 4.0.0-beta

Mocks camera acquisition (pyloncamera) and stage motion (motorboard / ledboard)
with real-time synthetic cell colony generation across SBS 96-well microplates.
Enables offline desktop development and testing of LumaViewPro and Plate Navigator without physical hardware.
"""

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
        self.is_light_on = False  # Starts OFF (black window) by default
        self.led_power = 100.0
        self.active_channel = "Brightfield"
        self.stage_x = -49.5  # Well A1 default
        self.stage_y = 31.5
        self.stage_z = 0.0
        self.best_z = 0.0

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
            led_power=self.led_power,
            exposure_ms=self.exposure_ms,
            gain=self.gain,
            sensor_size=(512, 512)
        )

    def set_exposure(self, exp: float):
        self.exposure_ms = exp

    def set_gain(self, g: float):
        self.gain = g

    def set_light_on(self, on: bool):
        self.is_light_on = on

    def set_led_power(self, power: float):
        self.led_power = power

    def set_channel(self, channel: str):
        self.active_channel = channel


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
