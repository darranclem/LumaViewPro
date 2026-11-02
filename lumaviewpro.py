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
from kivy.graphics.texture import Texture
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

    active_channel_name = 'BF'

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
        self.cam_debug_logs = []

    def _add_debug_log(self, msg: str):
        self.cam_debug_logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.cam_debug_logs) > 30:
            self.cam_debug_logs.pop(0)

    def extract_numpy_array(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x if x.size > 0 else None

        for attr in ['Array', 'GetArray', 'array', 'get_array', 'Buffer', 'buffer', 'Image', 'image']:
            if hasattr(x, attr):
                val = getattr(x, attr)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        continue
                if isinstance(val, np.ndarray) and val.size > 0:
                    return val
                res = self.extract_numpy_array(val)
                if res is not None:
                    return res

        try:
            arr = np.asarray(x)
            if isinstance(arr, np.ndarray) and arr.size > 0 and arr.dtype != object:
                return arr
        except Exception:
            pass

        return None

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

        self.stage_x = 0.0
        self.stage_y = 0.0
        self.stage_z = 0.0
        self.jog_step_mm = 1.0

        Clock.schedule_once(self._start_async_hardware_init, 0.05)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 20.0)

        return self.root_widget

    def _start_async_hardware_init(self, dt=0):
        threading.Thread(target=self.init_hardware_synchronized, daemon=True).start()

    def init_hardware_synchronized(self):
        self.status_msg = 'Initializing Camera...'
        self._add_debug_log('Starting Basler Camera Handshake...')

        try:
            import pyloncamera
            cam = pyloncamera.PylonCamera()
            time.sleep(0.2)

            if hasattr(cam, 'auto_gain'):
                try:
                    cam.auto_gain(False)
                except Exception as e:
                    self._add_debug_log(f'auto_gain err: {e}')

            if hasattr(cam, 'exposure_t'):
                try:
                    cam.exposure_t(20)
                except Exception as e:
                    self._add_debug_log(f'exposure_t err: {e}')

            if hasattr(cam, 'set_frame_size'):
                try:
                    cam.set_frame_size(1900, 1900)
                except Exception as e:
                    self._add_debug_log(f'set_frame_size err: {e}')

            cam.connect()
            self.camera = cam
            self.camera_connected = True
            msg = 'Connected to Basler USB Camera.'
            print(f'[LumaViewPro] {msg}', flush=True)
            self._add_debug_log(msg)

            threading.Thread(target=self._camera_grab_loop, daemon=True).start()
            self.set_camera_exposure(self.camera_exposure_ms)
        except Exception as e:
            err_msg = f'Camera connection error: {e}'
            print(f'[LumaViewPro] {err_msg}', flush=True)
            self._add_debug_log(err_msg)

        time.sleep(0.3)
        self.status_msg = 'Initializing Stage...'

        try:
            import motorboard
            stg = motorboard.MotorBoard()
            stg.connect()
            self.motor_stage = stg
            self.stage_connected = True
            print('[LumaViewPro] Connected to Stage Motor Board.', flush=True)
            self.update_stage_coordinates_display()
        except Exception as e:
            print(f'[LumaViewPro] Stage connection error: {e}', flush=True)

        time.sleep(0.3)
        self.status_msg = 'Initializing LEDs...'

        try:
            import ledboard
            led = ledboard.LEDBoard()
            led.connect()
            self.led_board = led
            self.led_connected = True
            print('[LumaViewPro] Connected to LED Illumination Board.', flush=True)
            self.set_active_channel('BF', int(self.channel_led_power['BF']))
        except Exception as e:
            print(f'[LumaViewPro] LED connection error: {e}', flush=True)

        self.status_msg = 'System Ready'

    def _camera_grab_loop(self):
        print('[CAM LOOP] Starting live frame grab loop...', flush=True)
        self._add_debug_log('Starting live frame grab loop...')
        frame_count = 0
        last_log_time = time.time()

        while self.camera_connected and self.camera:
            cam = self.camera
            frame_arr = None
            last_err = None

            for timeout_val in [1000, 500, 100]:
                if hasattr(cam, 'grab_new_capture'):
                    try:
                        res = cam.grab_new_capture(timeout_val)
                        frame_arr = self.extract_numpy_array(res)
                        if frame_arr is not None:
                            break
                    except Exception as e:
                        last_err = f'grab_new_capture({timeout_val}) err: {e}'

            if frame_arr is None and hasattr(cam, 'grab'):
                try:
                    res = cam.grab()
                    frame_arr = self.extract_numpy_array(res)
                except Exception as e:
                    if not last_err:
                        last_err = f'grab() err: {e}'

            if frame_arr is None:
                for method_name in ['grab_frame', 'capture', 'get_image', 'get_array', 'get_frame']:
                    if hasattr(cam, method_name):
                        try:
                            fn = getattr(cam, method_name)
                            res = fn()
                            frame_arr = self.extract_numpy_array(res)
                            if frame_arr is not None:
                                break
                        except Exception as e:
                            if not last_err:
                                last_err = f'{method_name}() err: {e}'

            if frame_arr is None:
                for attr_name in ['array', 'img', 'frame', 'image', 'raw_array', 'latest_frame', '_array', '_img', '_frame']:
                    if hasattr(cam, attr_name):
                        try:
                            val = getattr(cam, attr_name)
                            frame_arr = self.extract_numpy_array(val)
                            if frame_arr is not None:
                                break
                        except Exception:
                            pass

            if frame_arr is not None and frame_arr.size > 0:
                self.latest_frame = frame_arr
                frame_count += 1
                now = time.time()
                if now - last_log_time >= 3.0:
                    last_log_time = now
                    msg = f'Frame #{frame_count}: shape={frame_arr.shape}, dtype={frame_arr.dtype}, range=[{frame_arr.min()}, {frame_arr.max()}], mean={frame_arr.mean():.1f}'
                    print(f'[CAM STREAM SUCCESS] {msg}', flush=True)
                    self._add_debug_log(f'SUCCESS {msg}')
                    Clock.schedule_once(lambda dt, m=frame_arr.mean(): self._update_status_msg(f'Live Stream OK | Mean: {m:.0f}'), 0)
            else:
                now = time.time()
                if now - last_log_time >= 3.0:
                    last_log_time = now
                    msg = f'STREAM WAITING. Last err: {last_err}'
                    print(f'[CAM STREAM WAITING] {msg}', flush=True)
                    self._add_debug_log(msg)

            time.sleep(0.033)

    def _update_status_msg(self, msg: str):
        self.status_msg = msg

    def run_hardware_stream_diagnostic(self):
        diag_lines = []
        if self.camera_connected and self.camera:
            diag_lines.append("Camera Driver Status: CONNECTED")
            diag_lines.append(f"  Camera Object Class: {type(self.camera).__name__}")
            methods = [m for m in dir(self.camera) if not m.startswith('_')]
            diag_lines.append(f"  Available Methods: {', '.join(methods[:12])}")

            frame = self.latest_frame
            if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                diag_lines.append("Live Frame Buffer Stream: ACTIVE")
                diag_lines.append(f"  Resolution: {frame.shape[1]} x {frame.shape[0]}")
                diag_lines.append(f"  Data Type: {frame.dtype}")
                diag_lines.append(f"  Pixel Range: min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
                non_zero = float((frame > 0).sum()) / frame.size * 100.0
                diag_lines.append(f"  Active Light Coverage: {non_zero:.1f}%")
            else:
                diag_lines.append("Live Frame Buffer Stream: INACTIVE (No frame array received yet)")
        else:
            diag_lines.append("Hardware Diagnosis: Camera hardware object not initialized or failed connection.")

        diag_lines.append("")
        diag_lines.append("Recent Camera Log History:")
        for log_entry in self.cam_debug_logs[-10:]:
            diag_lines.append(f"  {log_entry}")

        content = BoxLayout(orientation='vertical', padding=10, spacing=8)
        content.add_widget(Label(text="Hardware Stream Diagnostic Report", font_size='16sp', bold=True))
        report_text = chr(10).join(diag_lines)
        content.add_widget(Label(text=report_text, font_size='12sp', halign='left', valign='top'))
        close_btn = Button(text="Close Report", size_hint_y=None, height='36dp')
        content.add_widget(close_btn)

        popup = Popup(title="Hardware Stream Diagnostic Report", content=content, size_hint=(0.85, 0.75))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def set_camera_exposure(self, val_ms: float):
        self.camera_exposure_ms = float(val_ms)
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

    def set_led_power(self, val_pct: float):
        self.led_power_pct = float(val_pct)
        self.channel_led_power[self.active_channel_name] = float(val_pct)
        if not self.led_connected or not self.led_board:
            return
        def _exec():
            try:
                ch_map = {
                    'BF': 4, 'BRIGHTFIELD': 4,
                    'PC': 5, 'PHASE': 5,
                    'DF': 6, 'DARKFIELD': 6,
                    'VIOLET': 3, 'BLUE_FLUOR': 3,
                    'BLUE': 0, 'GREEN_FLUOR': 0,
                    'GREEN': 1, 'RED_FLUOR': 1,
                    'RED': 2, 'FAR_RED': 2
                }
                ch_num = ch_map.get(self.active_channel_name.upper(), 4)
                self.led_board.exchange_command('LEDS_ENT')
                self.led_board.exchange_command(f'LED{ch_num}_{int(val_pct)}')
                print(f'[LumaViewPro] Updated active LED {self.active_channel_name} power to {int(val_pct)}%.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error setting LED power: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def turn_off_all_leds(self):
        if not self.led_connected or not self.led_board:
            return
        def _exec():
            try:
                for ch in range(7):
                    self.led_board.exchange_command(f'LED{ch}_OFF')
                self.led_board.exchange_command('LEDS_OFF')
                print('[LumaViewPro] All LED channels turned OFF.', flush=True)
            except Exception as e:
                print(f'[LumaViewPro] Error turning off LEDs: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

    def set_active_channel(self, channel_name: str, power_pct: int = None):
        self.active_channel_name = channel_name.upper()
        if power_pct is not None:
            self.channel_led_power[self.active_channel_name] = float(power_pct)

        saved_power = self.channel_led_power.get(self.active_channel_name, 100.0)
        saved_exp = self.channel_exposure_ms.get(self.active_channel_name, 20.0)
        self.led_power_pct = saved_power
        self.camera_exposure_ms = saved_exp

        if not self.led_connected or not self.led_board:
            return

        def _exec():
            try:
                for ch in range(7):
                    self.led_board.exchange_command(f'LED{ch}_OFF')

                ch_map = {
                    'BF': 4, 'BRIGHTFIELD': 4,
                    'PC': 5, 'PHASE': 5,
                    'DF': 6, 'DARKFIELD': 6,
                    'VIOLET': 3, 'BLUE_FLUOR': 3,
                    'BLUE': 0, 'GREEN_FLUOR': 0,
                    'GREEN': 1, 'RED_FLUOR': 1,
                    'RED': 2, 'FAR_RED': 2
                }
                ch_num = ch_map.get(self.active_channel_name, 4)

                self.led_board.exchange_command('LEDS_ENT')
                self.led_board.exchange_command(f'LED{ch_num}_{int(saved_power)}')
                print(f'[LumaViewPro] Switched illumination channel to {self.active_channel_name} (Index {ch_num}, {int(saved_power)}%).', flush=True)
                self.set_camera_exposure(saved_exp)
            except Exception as e:
                print(f'[LumaViewPro] Error setting channel {channel_name}: {e}', flush=True)
        threading.Thread(target=_exec, daemon=True).start()

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
        if axis.upper() == 'X' and hasattr(stg, 'move_x_rel'):
            stg.move_x_rel(um_dist)
        elif axis.upper() == 'Y' and hasattr(stg, 'move_y_rel'):
            stg.move_y_rel(um_dist)

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
                print('[LumaViewPro] Stage homed & centered.', flush=True)
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

    def update_live_feed(self, dt: float):
        if not self.camera_connected:
            return

        frame = self.latest_frame
        if frame is None and self.camera:
            for attr in ['array', 'img', 'frame', 'image', 'raw_array']:
                if hasattr(self.camera, attr):
                    val = getattr(self.camera, attr)
                    frame = self.extract_numpy_array(val)
                    if frame is not None:
                        self.latest_frame = frame
                        break

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            if hasattr(self, 'root_widget') and self.root_widget:
                try:
                    viewport = self.root_widget.ids.get('main_camera_viewport')
                    if viewport is not None:
                        viewport.update_frame_texture(frame)
                except Exception as e:
                    print(f'[LumaViewPro] Viewport update error: {e}', flush=True)


if __name__ == '__main__':
    LumaViewProApp().run()
