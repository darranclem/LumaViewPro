import os
import sys
import cv2
import numpy as np

import etaluma_init
etaluma_init.initialize_etaluma_environment()

from kivy.app import App
from kivy.lang import Builder
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout

import plate_navigator

script_dir = os.path.dirname(os.path.abspath(__file__))
kv_file = os.path.join(script_dir, 'lumaviewpro.kv')
if os.path.exists(kv_file):
    Builder.load_file(kv_file)


class LumaViewProMainLayout(BoxLayout):
    pass


class LumaViewProApp(App):
    title = 'LumaViewPro Microscope Control System'
    stage_coord_str = StringProperty('Stage: X=0.00, Y=0.00, Z=0.000 mm')

    def build(self):
        self.root_widget = LumaViewProMainLayout()

        plate_navigator.attach_navigator_to_lumaviewpro(self)

        self.camera_connected = False
        self.camera = None
        self.stage_connected = False
        self.motor_stage = None
        self.led_connected = False
        self.led_board = None

        self.stage_x = 0.0
        self.stage_y = 0.0
        self.stage_z = 0.0
        self.jog_step_mm = 1.0

        self.init_hardware()

        Clock.schedule_interval(self.update_live_feed, 1.0 / 15.0)

        return self.root_widget

    def init_hardware(self):
        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            cam.connect()
            self.camera = cam
            self.camera_connected = True
            print('[LumaViewPro] Connected to physical Basler USB Camera.')
            self.set_camera_exposure(20.0)
        except Exception as e:
            print(f'[LumaViewPro] Camera connection error: {e}')

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            stg.connect()
            self.motor_stage = stg
            self.stage_connected = True
            print('[LumaViewPro] Connected to physical Stage Motor Board.')
        except Exception as e:
            print(f'[LumaViewPro] Stage connection error: {e}')

        try:
            import ledboard
            led = ledboard.LEDBoard()
            led.connect()
            self.led_board = led
            self.led_connected = True
            print('[LumaViewPro] Connected to physical LED Illumination Board.')
            self.turn_off_all_leds()
        except Exception as e:
            print(f'[LumaViewPro] LED connection error: {e}')

    def set_camera_exposure(self, val_ms: float):
        if not self.camera_connected or not self.camera:
            return
        try:
            exp_us = int(float(val_ms) * 1000.0)
            if hasattr(self.camera, 'exposure_t'):
                self.camera.exposure_t(exp_us)
                print(f'[LumaViewPro] Camera exposure set to {val_ms}ms ({exp_us}us)')
        except Exception as e:
            print(f'[LumaViewPro] Error setting exposure: {e}')

    def turn_off_all_leds(self):
        if not self.led_connected or not self.led_board:
            return
        try:
            for ch in range(6):
                self.led_board.exchange_command(f'LED{ch}_OFF')
            self.led_board.exchange_command('LEDS_OFF')
            print('[LumaViewPro] All LED channels turned OFF.')
        except Exception as e:
            print(f'[LumaViewPro] Error turning off LEDs: {e}')

    def set_active_channel(self, channel_name: str, power_pct: int = 100):
        if not self.led_connected or not self.led_board:
            return
        self.turn_off_all_leds()
        ch_map = {'BF': 4, 'BLUE': 0, 'GREEN': 1, 'RED': 2, 'VIOLET': 3}
        ch_num = ch_map.get(channel_name.upper(), 4)

        try:
            self.led_board.exchange_command('LEDS_ENT')
            self.led_board.exchange_command(f'LED{ch_num}_{power_pct}')
            print(f'[LumaViewPro] Switched active illumination channel to {channel_name} ({power_pct}%).')
        except Exception as e:
            print(f'[LumaViewPro] Error setting channel {channel_name}: {e}')

    def jog_stage(self, dx_dir: float, dy_dir: float):
        step_mm = self.jog_step_mm
        dx_mm = dx_dir * step_mm
        dy_mm = dy_dir * step_mm

        dx_um = dx_mm * 1000.0
        dy_um = dy_mm * 1000.0

        self.stage_x += dx_mm
        self.stage_y += dy_mm
        self.update_stage_coordinates_display()

        if self.stage_connected and self.motor_stage:
            stg = self.motor_stage
            if dx_um != 0:
                try:
                    stg.move_rel_pos('x', int(dx_um))
                except Exception:
                    try:
                        stg.move_rel_pos('X', int(dx_um))
                    except Exception as e:
                        print(f'[LumaViewPro] Jog X error: {e}')
            if dy_um != 0:
                try:
                    stg.move_rel_pos('y', int(dy_um))
                except Exception:
                    try:
                        stg.move_rel_pos('Y', int(dy_um))
                    except Exception as e:
                        print(f'[LumaViewPro] Jog Y error: {e}')

    def home_stage_and_synchronize(self):
        if self.stage_connected and self.motor_stage:
            try:
                stg = self.motor_stage
                print('[LumaViewPro] Homing X and Y stage axes...')
                if hasattr(stg, 'xyhome'):
                    stg.xyhome()
                if hasattr(stg, 'xycenter'):
                    stg.xycenter()
                self.stage_x = 0.0
                self.stage_y = 0.0
                self.update_stage_coordinates_display()
                print('[LumaViewPro] Stage homed and centered at (0.0, 0.0).')
            except Exception as e:
                print(f'[LumaViewPro] Stage homing error: {e}')

    def update_stage_coordinates_display(self):
        self.stage_coord_str = f'Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm'
        if hasattr(self, 'root_widget') and self.root_widget:
            if hasattr(self.root_widget, 'ids') and 'main_plate_canvas' in self.root_widget.ids:
                canvas_widget = self.root_widget.ids.main_plate_canvas
                if hasattr(canvas_widget, 'redraw_canvas'):
                    canvas_widget.redraw_canvas()

    def update_live_feed(self, dt: float):
        if not self.camera_connected or not self.camera:
            return

        frame = None
        cam = self.camera

        if hasattr(cam, 'array') and isinstance(cam.array, np.ndarray) and cam.array.size > 0:
            frame = cam.array.copy()

        if frame is None and hasattr(cam, 'grab'):
            try:
                res = cam.grab()
                if isinstance(res, np.ndarray) and res.size > 0:
                    frame = res
            except Exception:
                pass

        if frame is None and hasattr(cam, 'grab_new_capture'):
            try:
                res = cam.grab_new_capture(100)
                if isinstance(res, np.ndarray) and res.size > 0:
                    frame = res
            except Exception:
                pass

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            self.render_frame_to_viewports(frame)

    def render_frame_to_viewports(self, frame: np.ndarray):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        if hasattr(self, 'root_widget') and self.root_widget:
            if hasattr(self.root_widget, 'ids') and 'main_camera_viewport' in self.root_widget.ids:
                viewport = self.root_widget.ids.main_camera_viewport
                if hasattr(viewport, 'set_frame'):
                    viewport.set_frame(frame)


if __name__ == '__main__':
    if hasattr(plate_navigator, 'InteractiveScaledPlateCanvas'):
        Factory.register('InteractiveScaledPlateCanvas', plate_navigator.InteractiveScaledPlateCanvas)
    if hasattr(plate_navigator, 'PlateNavigatorPopup'):
        Factory.register('PlateNavigatorPopup', plate_navigator.PlateNavigatorPopup)
    LumaViewProApp().run()
