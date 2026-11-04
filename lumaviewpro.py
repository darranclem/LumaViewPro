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


def _ensure_lvp_logger():
    """Make `from lvp_logger import logger` work for the stock drivers.

    This repo's own lvp_logger.py pulls in modules that use Python 3.12-only
    syntax, so importing it on Python 3.10 raises SyntaxError -- which made
    `import pyloncamera` (stock driver) fail and the camera never connect.
    Fall back to a minimal stand-in module that just provides `logger`.
    """
    import types
    import logging
    try:
        import lvp_logger  # noqa: F401
        return
    except Exception:
        pass
    logger = logging.getLogger('lumaviewpro')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    mod = types.ModuleType('lvp_logger')
    mod.logger = logger
    sys.modules['lvp_logger'] = mod


_ensure_lvp_logger()

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.factory import Factory
import plate_navigator


# Official Etaluma LED board channel mapping (stock ledboard.py color2ch):
#   0 = Blue (470), 1 = Green (530), 2 = Red (625),
#   3 = BF (Brightfield), 4 = PC (Phase Contrast), 5 = DF (Low Mag Darkfield)
# Wire command format: 'LED{channel}_{value}' / 'LED{channel}_OFF' / 'LEDS_OFF'
LED_CHANNEL_MAP = {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'BF': 3, 'PC': 4, 'DF': 5}


class LumaViewProMainLayout(BoxLayout):
    pass


class LumaViewProApp(App):
    title = 'LumaViewPro Microscope Control System'
    stage_coord_str = StringProperty('Stage: X=0.00, Y=0.00, Z=0.000 mm')
    status_msg = StringProperty('Ready')

    stage_x = NumericProperty(0.0)
    stage_y = NumericProperty(0.0)
    stage_z = NumericProperty(0.0)

    camera_exposure_ms = NumericProperty(20.0)
    led_power_pct = NumericProperty(100.0)
    auto_gain_active = BooleanProperty(False)
    auto_exp_active = BooleanProperty(False)

    active_channel_name = 'BF'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channel_led_power = {
            'BF': 100.0, 'PC': 100.0, 'DF': 100.0,
            'BLUE': 100.0, 'GREEN': 100.0, 'RED': 100.0
        }
        self.channel_exposure_ms = {
            'BF': 20.0, 'PC': 30.0, 'DF': 50.0,
            'BLUE': 100.0, 'GREEN': 100.0, 'RED': 100.0
        }
        self.latest_frame = None
        self.last_cam_err = 'None'
        self.frame_count = 0
        self.last_log_time = 0.0

    def build(self):
        Factory.register('ZoomPanCameraWidget', cls=plate_navigator.ZoomPanCameraWidget)
        Factory.register('InteractiveScaledPlateCanvas', cls=plate_navigator.InteractiveScaledPlateCanvas)

        self.root_widget = LumaViewProMainLayout()
        plate_navigator.attach_navigator_to_lumaviewpro(self)

        self.camera_connected = False
        self.camera = None
        self.stage_connected = False
        self.motor_stage = None
        self.led_connected = False
        self.led_board = None

        self.jog_step_mm = 1.0

        Clock.schedule_once(self._start_async_hardware_init, 0.05)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 30.0)

        return self.root_widget

    def _start_async_hardware_init(self, dt=0):
        threading.Thread(target=self.init_hardware_synchronized, daemon=True).start()

    def init_hardware_synchronized(self):
        self.status_msg = 'Initializing Camera...'
        try:
            import pyloncamera
            # Stock PylonCamera.__init__() auto-connects and starts grabbing.
            # Calling connect() again while already open makes the second
            # Open() fail, which the stock driver swallows by setting
            # active=False and replacing the image handler with a dead one --
            # no frames ever arrive. Only retry connect() if the initial
            # auto-connect did not succeed.
            cam = pyloncamera.PylonCamera()
            if not getattr(cam, 'active', False):
                cam.connect()

            if getattr(cam, 'active', False):
                # Configure while connected. set_frame_size() stops/restarts
                # grabbing internally via update_camera_config().
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
                        cam.exposure_t(self.camera_exposure_ms)
                    except Exception:
                        pass

                self.camera = cam
                self.camera_connected = True
                print('[LumaViewPro] Connected to Basler USB Camera.', flush=True)
            else:
                self.last_cam_err = 'PylonCamera.connect() failed (no device or device busy)'
                print(f'[LumaViewPro] Camera connection failed: {self.last_cam_err}', flush=True)

        except Exception as e:
            print(f'[LumaViewPro] Camera connection error: {e}', flush=True)
            self.last_cam_err = str(e)

        time.sleep(0.2)
        self.status_msg = 'Initializing Stage...'

        try:
            import motorboard
            # MotorBoard.__init__() also auto-connects; reconnecting while the
            # serial port is already open fails on Windows. Only retry if the
            # driver is not up.
            stg = motorboard.MotorBoard()
            if not getattr(stg, 'driver', False):
                stg.connect()
            self.motor_stage = stg
            self.stage_connected = bool(getattr(stg, 'driver', False))
            if self.stage_connected:
                print('[LumaViewPro] Connected to Stage Motor Board.', flush=True)
            else:
                print('[LumaViewPro] Stage motor board not connected.', flush=True)
            self.update_stage_coordinates_display()
        except Exception as e:
            print(f'[LumaViewPro] Stage connection error: {e}', flush=True)

        time.sleep(0.2)
        self.status_msg = 'Initializing LEDs...'

        try:
            import ledboard
            # LEDBoard.__init__() auto-connects too -- same rule as above.
            led = ledboard.LEDBoard()
            if not getattr(led, 'driver', False):
                led.connect()
            self.led_board = led
            self.led_connected = bool(getattr(led, 'driver', False))
            if self.led_connected:
                print('[LumaViewPro] Connected to LED Illumination Board.', flush=True)
                self.turn_off_all_leds()
            else:
                print('[LumaViewPro] LED board not connected.', flush=True)
        except Exception as e:
            print(f'[LumaViewPro] LED connection error: {e}', flush=True)

        self.status_msg = 'System Ready'

    def update_live_feed(self, dt: float):
        if not self.camera_connected or not self.camera:
            return

        cam = self.camera
        raw_frame = None

        # Non-blocking: grab() pulls the latest frame already delivered by
        # the pylon grab loop into cam.array (stock live-view pattern).
        # Never call blocking helpers like grab_new_capture() here -- this
        # runs on the Kivy UI thread.
        if hasattr(cam, 'grab'):
            try:
                cam.grab()
            except Exception as e:
                self.last_cam_err = f'grab err: {e}'

        if hasattr(cam, 'array'):
            arr = getattr(cam, 'array')
            if isinstance(arr, np.ndarray) and arr.size > 0:
                raw_frame = arr

        if raw_frame is None and hasattr(cam, 'cam_image_handler'):
            handler = getattr(cam, 'cam_image_handler')
            if hasattr(handler, 'last_img'):
                arr = getattr(handler, 'last_img')
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    raw_frame = arr

        frame_arr = self._extract_numpy_array(raw_frame)

        if frame_arr is not None and isinstance(frame_arr, np.ndarray) and frame_arr.size > 0:
            self.latest_frame = frame_arr
            self.frame_count += 1
            now = time.time()
            if now - self.last_log_time >= 2.0:
                self.last_log_time = now
                f_min, f_max = frame_arr.min(), frame_arr.max()
                f_mean = float(frame_arr.mean())
                f_std = float(frame_arr.std())
                print(f'[CAM STREAM LIVE 30FPS] Frame #{self.frame_count}: shape={frame_arr.shape}, dtype={frame_arr.dtype}, min={f_min}, max={f_max}, mean={f_mean:.1f}, std={f_std:.1f}', flush=True)

            if hasattr(self, 'root_widget') and self.root_widget:
                try:
                    viewport = self.root_widget.ids.get('main_camera_viewport')
                    if viewport:
                        viewport.update_frame_texture(frame_arr)
                except Exception as e:
                    print(f'[LumaViewPro] Viewport update error: {e}', flush=True)

    def _extract_numpy_array(self, obj):
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj if obj.size > 0 else None
        for attr in ['Array', 'GetArray', 'array', 'buffer', 'image', 'img', 'frame', 'last_img']:
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    if callable(val):
                        val = val()
                    if isinstance(val, np.ndarray) and val.size > 0:
                        return val
                except Exception:
                    pass
        return None

    def jog_z(self, dz_um: float):
        self.stage_z += (dz_um / 1000.0)
        self.update_stage_coordinates_display()
        if not self.motor_stage:
            print(f'[STAGE Z LOG] (Simulated) Z adjusted by {dz_um:.1f} um. New Z = {self.stage_z:.3f} mm', flush=True)
            return

        stg = self.motor_stage
        moved = False
        for m_name in ['move_z_rel', 'move_rel_z', 'MOVE_REL_Z']:
            if hasattr(stg, m_name):
                try:
                    getattr(stg, m_name)(int(dz_um))
                    moved = True
                    break
                except Exception:
                    pass

        if not moved and hasattr(stg, 'move_rel_pos'):
            try:
                stg.move_rel_pos('Z', int(dz_um))
                moved = True
            except Exception:
                pass

        if not moved and hasattr(stg, 'exchange_command'):
            try:
                res = stg.exchange_command(f'TARGET_WZ{int(dz_um)}')
                print(f'[STAGE Z LOG] Serial TARGET_WZ response: {res}', flush=True)
            except Exception as e:
                print(f'[STAGE Z LOG] Serial Z error: {e}', flush=True)

    def set_z_focus_absolute(self, z_mm: float):
        dz_um = (z_mm - self.stage_z) * 1000.0
        self.jog_z(dz_um)

    def zero_z_baseline(self):
        self.stage_z = 0.0
        self.update_stage_coordinates_display()

    def set_active_channel(self, ch_name: str, power_pct: float = None):
        if power_pct is None:
            power_pct = self.channel_led_power.get(ch_name, 100.0)

        self.turn_off_all_leds()
        time.sleep(0.05)

        self.active_channel_name = ch_name
        self.led_power_pct = power_pct

        if ch_name in self.channel_exposure_ms:
            self.set_camera_exposure(self.channel_exposure_ms[ch_name])

        if not self.led_board:
            return

        led = self.led_board
        idx = LED_CHANNEL_MAP.get(ch_name, 3)  # default BF, matching stock color2ch

        try:
            if hasattr(led, 'exchange_command'):
                led.exchange_command('LEDS_ENT')
                res = led.exchange_command(f'LED{idx}_{int(power_pct)}')
                print(f'[LED CTRL] Switched to {ch_name} (Index {idx}, {int(power_pct)}%) -> {res}', flush=True)
        except Exception as e:
            print(f'[LED CTRL] Channel switch error: {e}', flush=True)

    def turn_off_all_leds(self):
        if not self.led_board:
            return
        led = self.led_board
        try:
            if hasattr(led, 'exchange_command'):
                led.exchange_command('LEDS_OFF')
                print('[LED CTRL] All LED channels turned OFF.', flush=True)
        except Exception as e:
            print(f'[LED CTRL] Turn OFF error: {e}', flush=True)

    def set_camera_exposure(self, exp_ms: float):
        self.camera_exposure_ms = exp_ms
        if self.camera_connected and self.camera and hasattr(self.camera, 'exposure_t'):
            try:
                self.camera.exposure_t(exp_ms)
            except Exception:
                pass

    def set_led_power(self, power_pct: float):
        self.led_power_pct = power_pct
        if self.active_channel_name:
            self.channel_led_power[self.active_channel_name] = power_pct
            if self.led_connected and self.led_board:
                idx = LED_CHANNEL_MAP.get(self.active_channel_name, 3)
                if hasattr(self.led_board, 'exchange_command'):
                    try:
                        self.led_board.exchange_command(f'LED{idx}_{int(power_pct)}')
                    except Exception:
                        pass

    def toggle_auto_gain(self):
        self.auto_gain_active = not self.auto_gain_active
        if self.camera_connected and self.camera and hasattr(self.camera, 'auto_gain'):
            try:
                self.camera.auto_gain(self.auto_gain_active)
            except Exception:
                pass

    def toggle_auto_exposure(self):
        self.auto_exp_active = not self.auto_exp_active

    def jog_stage(self, dx_dir: float, dy_dir: float):
        dist_um = int(self.jog_step_mm * 1000.0)
        self.stage_x += dx_dir * self.jog_step_mm
        self.stage_y += dy_dir * self.jog_step_mm
        self.update_stage_coordinates_display()

        if self.motor_stage:
            stg = self.motor_stage
            if dx_dir != 0:
                for m in ['move_x_rel', 'move_rel_x']:
                    if hasattr(stg, m):
                        try:
                            getattr(stg, m)(int(dx_dir * dist_um))
                            break
                        except Exception:
                            pass
            if dy_dir != 0:
                for m in ['move_y_rel', 'move_rel_y']:
                    if hasattr(stg, m):
                        try:
                            getattr(stg, m)(int(dy_dir * dist_um))
                            break
                        except Exception:
                            pass

    def home_stage_and_synchronize(self):
        self.stage_x, self.stage_y, self.stage_z = 0.0, 0.0, 0.0
        self.update_stage_coordinates_display()
        if self.motor_stage:
            for m in ['xyhome', 'zhome', 'home']:
                if hasattr(self.motor_stage, m):
                    try:
                        getattr(self.motor_stage, m)()
                    except Exception:
                        pass

    def update_stage_coordinates_display(self):
        self.stage_coord_str = f'Stage: X={self.stage_x:.2f}, Y={self.stage_y:.2f}, Z={self.stage_z:.3f} mm'

    def run_hardware_stream_diagnostic(self):
        diag = [
            f"Camera Connected: {self.camera_connected}",
            f"Total Frames Captured: {self.frame_count}",
            f"Last Camera Error: {self.last_cam_err}",
            f"Active LED Channel: {self.active_channel_name} ({int(self.led_power_pct)}%)"
        ]
        if self.latest_frame is not None:
            f = self.latest_frame
            diag.append(f"Latest Frame Shape: {f.shape}, min={f.min()}, max={f.max()}, mean={f.mean():.1f}")
        else:
            diag.append("Latest Frame: None (Waiting for USB stream)")

        popup = Popup(title='Stream Diagnostic', content=Label(text='\n'.join(diag)), size_hint=(0.7, 0.5))
        popup.open()


if __name__ == '__main__':
    LumaViewProApp().run()