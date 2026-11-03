from __future__ import annotations
import os
import sys
import enum
import time
import threading

if not hasattr(enum, 'StrEnum'):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

driver_paths = [
    r"C:\Program Files (x86)\LumaViewPro-stock",
    r"C:\Program Files (x86)\Etaluma\LumaViewPro",
    r"C:\Program Files (x86)\LumaViewPro-main",
    os.path.dirname(os.path.abspath(__file__))
]
for p in driver_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
import plate_navigator


class LumaViewProMainLayout(BoxLayout):
    pass


class LumaViewProApp(App):
    title = 'LumaViewPro Microscope Control System'
    stage_coord_str = StringProperty('Stage: X=0.00, Y=0.00, Z=0.000 mm')
    status_msg = StringProperty('Ready')

    camera_exposure_ms = NumericProperty(20.0)
    led_power_pct = NumericProperty(100.0)
    auto_gain_active = BooleanProperty(False)
    auto_exp_active = BooleanProperty(False)

    active_channel_name = 'OFF'

    stage_x = 0.0
    stage_y = 0.0
    stage_z = 0.0
    stage_z_um = NumericProperty(0.0)
    jog_step_mm = 1.0
    jog_step_z_um = NumericProperty(10.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_led_power = {
            'BF': 100.0, 'PC': 100.0, 'DF': 100.0,
            'VIOLET': 100.0, 'BLUE': 100.0, 'GREEN': 100.0, 'RED': 100.0
        }
        self.channel_exposure_ms = {
            'BF': 20.0, 'PC': 30.0, 'DF': 50.0,
            'VIOLET': 100.0, 'BLUE': 100.0, 'GREEN': 100.0, 'RED': 100.0
        }
        self.latest_frame = None
        self.last_cam_err = 'None'
        self.frame_count = 0
        self.last_log_time = time.time()

    def load_kv(self, filename=None):
        return super().load_kv(filename)

    def build(self):
        self.root_widget = LumaViewProMainLayout()
        plate_navigator.attach_navigator_to_lumaviewpro(self)

        self.camera_connected = False
        self.camera = None
        self.stage_connected = False
        self.motor_stage = None
        self.led_connected = False
        self.led_board = None

        Clock.schedule_once(self._start_async_hardware_init, 0.05)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 30.0)

        return self.root_widget

    def _start_async_hardware_init(self, dt=0):
        threading.Thread(target=self.init_hardware_synchronized, daemon=True).start()

    def init_hardware_synchronized(self):
        Clock.schedule_once(lambda dt: self._set_status('Initializing Camera...'), 0)
        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            time.sleep(0.1)

            if hasattr(cam, 'auto_gain'):
                try:
                    cam.auto_gain(False)
                except Exception:
                    pass
            if hasattr(cam, 'set_frame_size'):
                try:
                    cam.set_frame_size(1900, 1900)
                except Exception:
                    pass
            if hasattr(cam, 'exposure_t'):
                try:
                    cam.exposure_t(20.0)
                except Exception:
                    pass
            if hasattr(cam, 'set_max_acquisition_frame_rate'):
                try:
                    cam.set_max_acquisition_frame_rate(30.0)
                except Exception:
                    pass

            cam.connect()
            if hasattr(cam, 'start_grabbing'):
                try:
                    cam.start_grabbing()
                except Exception:
                    pass

            self.camera = cam
            self.camera_connected = True
            print('[LumaViewPro] Connected to Basler USB Camera.', flush=True)
            self._introspect_camera(cam)
        except Exception as e:
            print(f'[LumaViewPro] Camera connection error: {e}', flush=True)
            self.last_cam_err = str(e)

        time.sleep(0.2)
        Clock.schedule_once(lambda dt: self._set_status('Initializing Stage...'), 0)

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            stg.connect()
            self.motor_stage = stg
            self.stage_connected = True
            print('[LumaViewPro] Connected to Stage & Focus Motor Board.', flush=True)
            self.update_stage_coordinates_display()
        except Exception as e:
            print(f'[LumaViewPro] Stage connection error: {e}', flush=True)

        time.sleep(0.2)
        Clock.schedule_once(lambda dt: self._set_status('Initializing LEDs...'), 0)

        try:
            import ledboard
            led = ledboard.LEDBoard()
            led.connect()
            self.led_board = led
            self.led_connected = True
            print('[LumaViewPro] Connected to LED Illumination Board.', flush=True)
            # Default state: ALL LEDs OFF on startup
            self.turn_off_all_leds()
        except Exception as e:
            print(f'[LumaViewPro] LED connection error: {e}', flush=True)

        Clock.schedule_once(lambda dt: self._set_status('System Ready (LEDs Off)'), 0)

    def _set_status(self, msg: str):
        self.status_msg = msg

    def _introspect_camera(self, cam):
        print('=== PYLONCAMERA HARDWARE INTROSPECTION PROBE ===', flush=True)
        print(f'Camera Class: {type(cam).__module__}.{type(cam).__name__}', flush=True)
        if hasattr(cam, 'cam_image_handler'):
            handler = getattr(cam, 'cam_image_handler')
            print(f'Handler Class: {type(handler).__module__}.{type(handler).__name__}', flush=True)
            if hasattr(handler, 'last_img'):
                img = getattr(handler, 'last_img')
                shape = getattr(img, 'shape', None)
                print(f"  [HANDLER ATTR] 'last_img': type={type(img).__name__}, shape={shape}", flush=True)
        print('================================================', flush=True)

    def _extract_numpy_array(self, obj):
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj if obj.size > 0 else None

        for attr_name in ['last_img', 'Array', 'GetArray', 'array', 'buffer', 'image', 'img', 'frame']:
            if hasattr(obj, attr_name):
                try:
                    val = getattr(obj, attr_name)
                    if callable(val):
                        val = val()
                    if isinstance(val, np.ndarray) and val.size > 0:
                        return val
                except Exception:
                    pass
        return None

    def get_channel_index(self, channel_name: str) -> int:
        ch = str(channel_name).upper()
        ch_map = {
            'BLUE': 0, 'GREEN_FLUOR': 0,
            'GREEN': 1, 'RED_FLUOR': 1,
            'RED': 2, 'FAR_RED_FLUOR': 2,
            'VIOLET': 3, 'BLUE_FLUOR': 3,
            'BF': 4, 'BRIGHTFIELD': 4,
            'PC': 5, 'PHASE': 5,
            'DF': 6, 'DARKFIELD': 6,
        }
        return ch_map.get(ch, 4)

    def set_active_channel(self, channel_name: str, power_pct: int = None):
        self.active_channel_name = channel_name.upper()
        if power_pct is not None:
            self.channel_led_power[self.active_channel_name] = float(power_pct)

        saved_power = self.channel_led_power.get(self.active_channel_name, self.led_power_pct)
        saved_exp = self.channel_exposure_ms.get(self.active_channel_name, 20.0)
        self.led_power_pct = saved_power
        self.camera_exposure_ms = saved_exp

        if not self.led_connected or not self.led_board:
            print(f'[LumaViewPro] Offline channel switch: {self.active_channel_name}', flush=True)
            return

        def _exec():
            try:
                # Rule: Switching channels MUST turn off all existing LEDs first
                self.led_board.exchange_command('LEDS_OFF')
                for ch in range(7):
                    try:
                        self.led_board.exchange_command(f'LED{ch}_OFF')
                    except Exception:
                        pass

                ch_num = self.get_channel_index(self.active_channel_name)
                self.led_board.exchange_command('LEDS_ENT')
                self.led_board.exchange_command(f'LED{ch_num}_{int(saved_power)}')
                print(f'[LumaViewPro] Switched illumination to {self.active_channel_name} (Index {ch_num}) at {int(saved_power)}%.', flush=True)
                self.set_camera_exposure(saved_exp)
            except Exception as e:
                print(f'[LumaViewPro] Error switching channel {channel_name}: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def set_led_power(self, val_pct: float):
        self.led_power_pct = float(val_pct)
        if self.active_channel_name != 'OFF':
            self.channel_led_power[self.active_channel_name] = float(val_pct)
        if not self.led_connected or not self.led_board or self.active_channel_name == 'OFF':
            return
        def _exec():
            try:
                ch_num = self.get_channel_index(self.active_channel_name)
                self.led_board.exchange_command('LEDS_ENT')
                self.led_board.exchange_command(f'LED{ch_num}_{int(val_pct)}')
                print(f'[LumaViewPro] Updated {self.active_channel_name} LED power to {int(val_pct)}%.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error setting LED power: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def turn_off_all_leds(self):
        self.active_channel_name = 'OFF'
        if not self.led_connected or not self.led_board:
            print('[LumaViewPro] Offline: All LEDs set to OFF.', flush=True)
            return
        def _exec():
            try:
                for ch in range(7):
                    try:
                        self.led_board.exchange_command(f'LED{ch}_OFF')
                    except Exception:
                        pass
                self.led_board.exchange_command('LEDS_OFF')
                print('[LumaViewPro] All LED channels turned OFF.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error turning off LEDs: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def set_camera_exposure(self, val_ms: float):
        self.camera_exposure_ms = float(val_ms)
        if self.active_channel_name != 'OFF':
            self.channel_exposure_ms[self.active_channel_name] = float(val_ms)
        if not self.camera_connected or not self.camera:
            return
        def _exec():
            try:
                val = float(val_ms)
                if hasattr(self.camera, 'exposure_t'):
                    self.camera.exposure_t(val)
                    print(f'[LumaViewPro] Camera exposure set to {val} ms.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error setting exposure: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def jog_z_focus(self, dz_um: float):
        self.stage_z_um += float(dz_um)
        self.stage_z = self.stage_z_um / 1000.0
        self.update_stage_coordinates_display()
        self._safe_move_z_rel(float(dz_um))

    def set_stage_z_um(self, new_z_um: float):
        dz = new_z_um - self.stage_z_um
        if abs(dz) < 0.1:
            return
        self.stage_z_um = float(new_z_um)
        self.stage_z = self.stage_z_um / 1000.0
        self.update_stage_coordinates_display()
        self._safe_move_z_rel(dz)

    def zero_z_position(self):
        self.stage_z_um = 0.0
        self.stage_z = 0.0
        self.update_stage_coordinates_display()
        self.status_msg = 'Z Baseline Zeroed'

    def set_jog_step_z_um(self, val_um: float):
        self.jog_step_z_um = float(val_um)

    def _safe_move_z_rel(self, um_dist: float):
        if not self.stage_connected or not self.motor_stage:
            return
        def _exec():
            try:
                stg = self.motor_stage
                dist_int = int(round(um_dist))
                if dist_int == 0:
                    return
                # MotorBoard requires UPPERCASE 'Z'
                try:
                    stg.move_rel_pos('Z', dist_int)
                    print(f'[LumaViewPro] MotorBoard.move_rel_pos("Z", {dist_int}) executed.', flush=True)
                except Exception:
                    try:
                        stg.move_z_rel(dist_int)
                        print(f'[LumaViewPro] MotorBoard.move_z_rel({dist_int}) executed.', flush=True)
                    except Exception as e2:
                        print(f'[LumaViewPro] Z move error: {e2}', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Z move thread error: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def jog_stage(self, dx_dir: float, dy_dir: float):
        step_mm = self.jog_step_mm
        dx_mm = dx_dir * step_mm
        dy_mm = dy_dir * step_mm

        self.stage_x += dx_mm
        self.stage_y += dy_mm
        self.update_stage_coordinates_display()

        if not self.stage_connected or not self.motor_stage:
            return

        def _exec():
            try:
                if dx_dir != 0:
                    dx_um = int(dx_mm * 1000.0)
                    self._safe_move_rel('X', dx_um)
                if dy_dir != 0:
                    dy_um = int(dy_mm * 1000.0)
                    self._safe_move_rel('Y', dy_um)
            except Exception as e:
                print(f'[LumaViewPro] Stage jog error: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def _safe_move_rel(self, axis: str, um_dist: int):
        if not self.motor_stage:
            return
        stg = self.motor_stage
        for ax_str in [axis.upper(), axis.lower()]:
            try:
                stg.move_rel_pos(ax_str, um_dist)
                return
            except Exception:
                pass

    def home_stage_and_synchronize(self):
        self.stage_x = 0.0
        self.stage_y = 0.0
        self.update_stage_coordinates_display()

        if not self.stage_connected or not self.motor_stage:
            return

        def _exec():
            try:
                if hasattr(self.motor_stage, 'xyhome'):
                    self.motor_stage.xyhome()
                if hasattr(self.motor_stage, 'xycenter'):
                    self.motor_stage.xycenter()
                print('[LumaViewPro] Stage XY homed & centered.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error homing stage: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def update_stage_coordinates_display(self):
        self.stage_coord_str = f'Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm'
        if hasattr(self, 'root_widget') and self.root_widget:
            try:
                canvas_widget = self.root_widget.ids.main_plate_canvas
                if hasattr(canvas_widget, 'redraw_canvas'):
                    canvas_widget.redraw_canvas()
            except Exception:
                pass

    def toggle_auto_gain(self):
        self.auto_gain_active = not self.auto_gain_active
        if self.camera_connected and self.camera and hasattr(self.camera, 'auto_gain'):
            try:
                self.camera.auto_gain(self.auto_gain_active)
                print(f'[LumaViewPro] Auto Gain set to {self.auto_gain_active}', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Auto gain error: {e}', flush=True)

    def toggle_auto_exposure(self):
        self.auto_exp_active = not self.auto_exp_active
        if self.camera_connected and self.camera and hasattr(self.camera, 'auto_exposure'):
            try:
                self.camera.auto_exposure(self.auto_exp_active)
                print(f'[LumaViewPro] Auto Exposure set to {self.auto_exp_active}', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Auto exposure error: {e}', flush=True)

    def run_autofocus(self):
        self.status_msg = 'Running Autofocus...'
        def _exec():
            try:
                if self.motor_stage and hasattr(self.motor_stage, 'autofocus'):
                    self.motor_stage.autofocus()
                elif self.camera and hasattr(self.camera, 'autofocus'):
                    self.camera.autofocus()
                print('[LumaViewPro] Autofocus completed.', flush=True)
                self.status_msg = 'Ready'
            except Exception as e:
                print(f'[LumaViewPro] Autofocus error: {e}', flush=True)
                self.status_msg = 'Autofocus Error'
        threading.Thread(target=_exec, daemon=True).start()

    def add_current_focus_map_point(self):
        well_name = 'A1'
        self.focus_map.add_point(well_name, self.stage_x, self.stage_y, self.stage_z)
        self.status_msg = f'Added Focus Map Pt ({well_name})'
        self.update_stage_coordinates_display()

    def add_current_autofocus_point(self):
        well_name = 'A1'
        self.fov_manager.add_autofocus_point(well_name)
        self.status_msg = f'Added AF Step Pt ({well_name})'
        self.update_stage_coordinates_display()

    def run_hardware_stream_diagnostic(self):
        diag_lines = []
        if self.camera_connected and self.camera:
            diag_lines.append('Camera Driver Status: CONNECTED')
            diag_lines.append(f'  Total Frames Streamed: {self.frame_count}')
            frame = self.latest_frame
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                diag_lines.append('Live Frame Buffer Stream: ACTIVE')
                diag_lines.append(f'  Resolution: {frame.shape[1]} x {frame.shape[0]}')
                diag_lines.append(f'  Data Type: {frame.dtype}')
                diag_lines.append(f'  Pixel Range: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}')
            else:
                diag_lines.append('Live Frame Buffer Stream: INACTIVE')
        else:
            diag_lines.append('Camera Status: Offline')

        content = BoxLayout(orientation='vertical', padding=10, spacing=8)
        content.add_widget(Label(text='Hardware Diagnostic Report', font_size='16sp', bold=True))
        report_text = chr(10).join(diag_lines)
        content.add_widget(Label(text=report_text, font_size='13sp', halign='left', valign='top'))
        close_btn = Button(text='Close', size_hint_y=None, height='36dp')
        content.add_widget(close_btn)

        popup = Popup(title='Hardware Diagnostic', content=content, size_hint=(0.8, 0.6))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def update_live_feed(self, dt: float):
        if not self.camera_connected or not self.camera:
            return

        cam = self.camera
        frame_arr = None

        # 1. Primary fast source: cam.cam_image_handler.last_img
        if hasattr(cam, 'cam_image_handler'):
            handler = getattr(cam, 'cam_image_handler')
            if hasattr(handler, 'last_img'):
                raw = getattr(handler, 'last_img')
                if isinstance(raw, np.ndarray) and raw.size > 0:
                    frame_arr = raw

        # 2. Secondary source: cam.array
        if frame_arr is None:
            raw_arr = getattr(cam, 'array', None)
            frame_arr = self._extract_numpy_array(raw_arr)

        if frame_arr is not None and isinstance(frame_arr, np.ndarray) and frame_arr.size > 0:
            self.latest_frame = frame_arr
            self.frame_count += 1
            now = time.time()
            if now - self.last_log_time >= 3.0:
                self.last_log_time = now
                print(f'[CAM STREAM SUCCESS] Frame #{self.frame_count}: shape={frame_arr.shape}, dtype={frame_arr.dtype}, min={frame_arr.min()}, max={frame_arr.max()}', flush=True)
            if hasattr(self, 'root_widget') and self.root_widget:
                try:
                    viewport = self.root_widget.ids.get('main_camera_viewport')
                    if viewport:
                        viewport.update_frame_texture(frame_arr)
                except Exception as e:
                    print(f'[LumaViewPro] Viewport update error: {e}', flush=True)


if __name__ == '__main__':
    LumaViewProApp().run()
