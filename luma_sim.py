"""
luma_sim.py - Offline Hardware Simulator for LumaViewPro 4.0.0-beta

Mocks camera acquisition (pyloncamera) and stage motion (motorboard / ledboard)
with real-time synthetic cell colony generation across multi-well microplates.
Enables offline desktop development and testing without physical microscope hardware.
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
        self.exposure_ms = 20.0
        self.gain = 1.0
        self.stage_x = -49.5
        self.stage_y = 31.5
        self.stage_z = 0.0
        self.best_z = 0.0
        self.active_channel = "Brightfield"
        self.is_light_on = False  # DARK LAUNCH STATE
        self.led_power = 100.0
        self.num_wells = 96
        self.fov_size_mm = (0.60, 0.45)

    def connect(self) -> bool:
        return True

    def disconnect(self):
        pass

    def get_frame(self):
        from kivy.app import App
        app = App.get_running_app()
        if app:
            stg_x = getattr(app, 'stage_x', self.stage_x)
            stg_y = getattr(app, 'stage_y', self.stage_y)
            stg_z = getattr(app, 'stage_z', self.stage_z)
            ch = getattr(app, 'active_channel', self.active_channel)
            light_on = getattr(app, 'is_light_on', self.is_light_on)
            exp = getattr(app, 'exposure_ms', self.exposure_ms)
            gn = getattr(app, 'gain', self.gain)
            led_pwr = getattr(app, 'led_power', self.led_power)

            fov_w = app.focus_map.fov_width_mm if hasattr(app, 'focus_map') else 0.60
            fov_h = app.focus_map.fov_height_mm if hasattr(app, 'focus_map') else 0.45
            wells = app.well_plate_format.num_wells if hasattr(app, 'well_plate_format') else 96
        else:
            stg_x, stg_y, stg_z = self.stage_x, self.stage_y, self.stage_z
            ch = self.active_channel
            light_on = self.is_light_on
            exp, gn, led_pwr = self.exposure_ms, self.gain, self.led_power
            fov_w, fov_h = self.fov_size_mm
            wells = self.num_wells

        return image_utils.generate_well_plate_cell_frame(
            stage_x=stg_x,
            stage_y=stg_y,
            stage_z=stg_z,
            best_z=0.0,
            channel=ch,
            exposure_ms=exp,
            gain=gn,
            led_power=led_pwr,
            is_light_on=light_on,
            fov_size_mm=(fov_w, fov_h),
            num_wells=wells,
            sensor_size=(512, 512)
        )

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
