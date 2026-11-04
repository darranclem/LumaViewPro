"""LumaViewPro custom control application.

LAS X Navigator-inspired interface over the stock Etaluma drivers:
live view with per-channel LUTs, interactive well plate navigator with
double-click stage navigation, double-click-to-center on the image,
startup stage calibration against the limit switches, per-channel
exposure/gain/LED control, bit depth and binning switching, autofocus
with focus surface mapping, shading correction, tile/spiral scans with
stitching, and z-stacks. Fails safe on exit (all LEDs off, camera
stopped).
"""

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
    syntax, so importing it on older interpreters raises SyntaxError -- which
    made `import pyloncamera` (stock driver) fail and the camera never
    connect. Fall back to a minimal stand-in module providing `logger`.
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
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.factory import Factory

import plate_navigator
import acquisition as acq
from acquisition import (CHANNELS, LED_CHANNEL_MAP, ChannelSettings,
                         AutofocusConfig, ScopeHW, FocusSurface,
                         ShadingCorrector, AcquisitionWorker)

Factory.register('ZoomPanCameraWidget', cls=plate_navigator.ZoomPanCameraWidget)
Factory.register('WellPlateMapWidget', cls=plate_navigator.WellPlateMapWidget)
Factory.register('InteractiveScaledPlateCanvas', cls=plate_navigator.InteractiveScaledPlateCanvas)


# Display LUTs: (R, G, B) tint applied to the mono camera image per channel.
# Fluorescence channels get false color matched to their emission band;
# transmitted channels (BF/PC/DF) stay grayscale. 'LONG' is reserved for a
# longer excitation/detection channel (shown magenta) if such hardware is
# fitted -- standard Etaluma LED boards do not have one.
CHANNEL_LUTS = {
    'BF': (1.0, 1.0, 1.0),
    'PC': (1.0, 1.0, 1.0),
    'DF': (1.0, 1.0, 1.0),
    'BLUE': (0.0, 1.0, 1.0),   # ~420 nm emission -> cyan
    'GREEN': (0.0, 1.0, 0.0),  # 500 nm+ -> green
    'RED': (1.0, 0.0, 0.0),    # 560-650 nm -> red
    'LONG': (1.0, 0.0, 1.0),   # longer ex/em (if fitted) -> magenta
}

CAPTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captures')

# Basler daA3840-45um sensor pixel pitch is 2.0 um; divide by objective
# magnification for the default sample-plane scale (editable in the UI).
DEFAULT_UM_PER_PX = 0.5   # 2.0 um / 4x objective


class LumaViewProMainLayout(BoxLayout):
    pass


class LumaViewProApp(App):
    title = 'LumaViewPro Navigator'

    status_msg = StringProperty('Starting...')
    stage_coord_str = StringProperty('X 0.000  Y 0.000  Z 0.000 mm')
    selected_well_str = StringProperty('--')
    live_channel = StringProperty('OFF')
    selected_channel = StringProperty('BF')
    bit_depth_options = ListProperty(['8-bit'])
    bit_depth = StringProperty('8-bit')
    binning = StringProperty('1')
    crosshair_on = BooleanProperty(False)
    autoscale_on = BooleanProperty(False)
    shading_on = BooleanProperty(False)
    surface_info = StringProperty('Surface: no anchor')
    worker_busy = BooleanProperty(False)

    # -------------------------------------------------------------- setup

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hw = ScopeHW()
        self.worker = AcquisitionWorker()
        self.surface = FocusSurface()
        self.shading = ShadingCorrector()
        self.channel_settings: dict[str, ChannelSettings] = acq.default_channel_settings()
        self.af_config = AutofocusConfig()
        self.camera_connected = False
        self.led_connected = False
        self.stage_connected = False
        self.latest_frame = None
        self.significant_bits = 8
        self.frame_count = 0
        self.last_log_time = 0.0
        self.last_cam_err = 'None'

        # Stage/plate mapping calibration
        self.a1_stage_um: tuple[float, float] | None = None
        self.plate_sign_x = 1.0    # +1: stage +X moves toward column 12
        self.plate_sign_y = 1.0    # +1: stage +Y moves toward row H
        self.base_um_per_px = DEFAULT_UM_PER_PX
        self.img_sign_x = 1.0      # double-click-to-center direction trims
        self.img_sign_y = 1.0

        self.stage_x_um = 0.0
        self.stage_y_um = 0.0
        self.stage_z_um = 0.0

        self._load_persisted()

    def build(self):
        pass  # root comes from lumaviewpro.kv

    def on_start(self):
        self.root_widget = self.root
        viewport = self.root.ids.get('main_camera_viewport')
        if viewport:
            viewport.on_image_double_click = self.on_image_double_click
        plate = self.root.ids.get('plate_map')
        if plate:
            plate.on_well_double_click = self.on_well_double_click
            plate.on_well_selected = self.on_well_selected
        self._sync_channel_fields()
        self._sync_af_fields()
        Clock.schedule_once(lambda dt: self._start_init_thread(), 0.1)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 30.0)
        threading.Thread(target=self._position_poller, daemon=True).start()

    # --------------------------------------------------- hardware startup

    def _start_init_thread(self):
        threading.Thread(target=self._init_hardware, daemon=True).start()

    def _set_status(self, msg: str):
        Clock.schedule_once(lambda dt: setattr(self, 'status_msg', msg), 0)

    def _init_hardware(self):
        self._set_status('Initializing camera...')
        try:
            import pyloncamera
            # Stock PylonCamera.__init__() auto-connects and starts grabbing.
            # A second connect() while open fails and leaves a dead handler,
            # so only retry when the auto-connect did not succeed.
            cam = pyloncamera.PylonCamera()
            if not getattr(cam, 'active', False):
                cam.connect()
            if getattr(cam, 'active', False):
                self.hw.camera = cam
                self.camera_connected = True
                try:
                    formats = [f for f in cam.get_supported_pixel_formats()
                               if f.startswith('Mono') and f[4:].isdigit()]
                    opts = [f'{int(f[4:])}-bit' for f in formats]
                    cur_bits = int(''.join(c for c in cam.get_pixel_format()
                                           if c.isdigit()) or 8)
                    def _apply(dt, o=opts, b=cur_bits):
                        self.bit_depth_options = o or ['8-bit']
                        self.bit_depth = f'{b}-bit'
                    Clock.schedule_once(_apply, 0)
                    self.significant_bits = cur_bits
                except Exception:
                    pass
                self._set_status('Camera connected')
            else:
                self.last_cam_err = 'PylonCamera.connect() failed'
                self._set_status('Camera connection FAILED')
        except Exception as e:
            self.last_cam_err = str(e)
            self._set_status(f'Camera error: {e}')

        self._set_status('Initializing LED board...')
        try:
            import ledboard
            led = ledboard.LEDBoard()
            if not getattr(led, 'driver', False):
                led.connect()
            self.hw.led = led
            self.led_connected = bool(getattr(led, 'driver', False))
            self.hw.leds_off()
        except Exception as e:
            self._set_status(f'LED board error: {e}')

        self._set_status('Initializing stage...')
        try:
            import motorboard
            stg = motorboard.MotorBoard()
            if not getattr(stg, 'driver', False):
                stg.connect()
            self.hw.stage = stg
            self.stage_connected = bool(getattr(stg, 'driver', False))
        except Exception as e:
            self._set_status(f'Stage error: {e}')

        # Startup calibration: objective fully lowered, XY homed against the
        # limit switches (firmware measures travel), then stage centered.
        if self.stage_connected:
            self.hw.home_and_center(status_cb=self._set_status)
        else:
            self._set_status('Ready (no stage)')

    # ------------------------------------------------- live view pipeline

    def update_live_feed(self, dt: float):
        cam = self.hw.camera
        if cam is None or not getattr(cam, 'active', False):
            return
        try:
            cam.grab()   # non-blocking: refresh cam.array from the grab loop
        except Exception as e:
            self.last_cam_err = f'grab err: {e}'
        arr = getattr(cam, 'array', None)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return
        frame = arr
        ch = self.live_channel if self.live_channel in CHANNELS else \
            (self.hw.active_channel or 'BF')
        if self.shading_on and self.shading.has_reference(ch):
            frame = self.shading.apply(frame, ch)
        self.latest_frame = frame
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            self.last_log_time = now
            print(f'[CAM STREAM] Frame #{self.frame_count}: shape={frame.shape}, '
                  f'dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, '
                  f'mean={frame.mean():.1f}', flush=True)
        viewport = self.root.ids.get('main_camera_viewport') if self.root else None
        if viewport:
            tint = CHANNEL_LUTS.get(ch, (1.0, 1.0, 1.0))
            viewport.autoscale = self.autoscale_on
            viewport.update_frame_texture(frame, tint=tint,
                                          significant_bits=self.significant_bits)

    def _position_poller(self):
        while True:
            time.sleep(0.5)
            if not self.stage_connected:
                continue
            try:
                x, y, z = self.hw.get_pos()
            except Exception:
                continue
            def _apply(dt, x=x, y=y, z=z):
                self.stage_x_um, self.stage_y_um, self.stage_z_um = x, y, z
                self.stage_coord_str = (f'X {x / 1000:.3f}  Y {y / 1000:.3f}  '
                                        f'Z {z / 1000:.3f} mm')
                self._update_plate_marker()
            Clock.schedule_once(_apply, 0)

    def _update_plate_marker(self):
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is None or self.a1_stage_um is None:
            return
        rel_x_mm = (self.stage_x_um - self.a1_stage_um[0]) * self.plate_sign_x / 1000.0
        rel_y_mm = (self.stage_y_um - self.a1_stage_um[1]) * self.plate_sign_y / 1000.0
        plate.set_marker_plate_mm(rel_x_mm, rel_y_mm)

    # --------------------------------------------------------- bg helper

    def _bg(self, fn, *args, busy_job=False, **kwargs):
        """Run hardware IO off the UI thread. busy_job routes through the
        acquisition worker (one long job at a time, cancelable)."""
        if busy_job:
            def wrapped():
                Clock.schedule_once(lambda dt: setattr(self, 'worker_busy', True), 0)
                try:
                    fn(*args, **kwargs)
                finally:
                    Clock.schedule_once(lambda dt: setattr(self, 'worker_busy', False), 0)
            if not self.worker.start(wrapped):
                self._set_status('Busy: another acquisition is running')
            return
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()

    def cancel_job(self):
        self.worker.request_cancel()
        self._set_status('Cancel requested...')

    # ------------------------------------------------------ channel control

    def _cs(self, channel: str | None = None) -> ChannelSettings:
        return self.channel_settings[channel or self.selected_channel]

    def select_channel(self, channel: str):
        self.selected_channel = channel
        self._sync_channel_fields()

    def _sync_channel_fields(self):
        if not self.root:
            return
        cs = self._cs()
        ids = self.root.ids
        def _set(k, v):
            w = ids.get(k)
            if w is not None:
                w.text = str(v)
        _set('exposure_input', f'{cs.exposure_ms:g}')
        _set('gain_input', f'{cs.gain_db:g}')
        _set('led_input', f'{cs.led_power}')
        _set('zoffset_input', f'{cs.z_offset_um:g}')
        _set('zstack_range_input', f'{cs.zstack.range_um:g}')
        _set('zstack_step_input', f'{cs.zstack.step_um:g}')
        enabled = ids.get('channel_enabled_chk')
        if enabled is not None:
            enabled.active = cs.enabled
        zstack_chk = ids.get('zstack_enabled_chk')
        if zstack_chk is not None:
            zstack_chk.active = cs.zstack.enabled

    @staticmethod
    def _f(widget_text, fallback):
        try:
            return float(widget_text)
        except Exception:
            return fallback

    def apply_channel_fields(self):
        """Read the per-channel fields into the selected channel's settings
        and apply them to hardware if that channel is live."""
        if not self.root:
            return
        ids = self.root.ids
        cs = self._cs()
        cs.exposure_ms = self._f(ids.exposure_input.text, cs.exposure_ms)
        cs.gain_db = self._f(ids.gain_input.text, cs.gain_db)
        cs.led_power = int(self._f(ids.led_input.text, cs.led_power))
        cs.z_offset_um = self._f(ids.zoffset_input.text, cs.z_offset_um)
        cs.zstack.range_um = self._f(ids.zstack_range_input.text, cs.zstack.range_um)
        cs.zstack.step_um = self._f(ids.zstack_step_input.text, cs.zstack.step_um)
        cs.enabled = bool(ids.channel_enabled_chk.active)
        cs.zstack.enabled = bool(ids.zstack_enabled_chk.active)
        self._persist()
        if self.live_channel == self.selected_channel:
            self._bg(self.hw.set_channel, self.selected_channel, cs)
        self._set_status(f'{self.selected_channel} settings applied')

    def set_live(self, channel: str):
        """Make a channel live (LED on + camera settings), or 'OFF' for all
        channels off."""
        if self.worker_busy:
            self._set_status('Busy: acquisition running')
            return
        if channel == 'OFF':
            self.live_channel = 'OFF'
            self._bg(self.hw.set_channel, None)
            self._set_status('All channels off')
            return
        self.live_channel = channel
        self.selected_channel = channel
        self._sync_channel_fields()
        self._bg(self.hw.set_channel, channel, self._cs(channel))
        self._set_status(f'Live: {channel}')

    # ---------------------------------------------------- camera controls

    def set_bit_depth(self, label: str):
        cam = self.hw.camera
        if cam is None or not self.camera_connected:
            return
        bits = int(''.join(c for c in label if c.isdigit()) or 8)
        def _apply():
            try:
                ok = cam.set_pixel_format(f'Mono{bits}')
                if ok:
                    self.significant_bits = bits
                    self._set_status(f'Pixel format: Mono{bits}')
                else:
                    self._set_status(f'Mono{bits} not supported by camera')
            except Exception as e:
                self._set_status(f'Bit depth error: {e}')
        self._bg(_apply)

    def set_binning(self, label: str):
        cam = self.hw.camera
        if cam is None or not self.camera_connected:
            return
        try:
            size = int(label)
        except ValueError:
            return
        def _apply():
            try:
                if cam.set_binning_size(size):
                    self._set_status(f'Binning {size}x{size}')
                else:
                    self._set_status(f'Binning {size} rejected')
            except Exception as e:
                self._set_status(f'Binning error: {e}')
        self._bg(_apply)

    def effective_um_per_px(self) -> float:
        bin_f = 1
        try:
            bin_f = int(self.binning)
        except Exception:
            pass
        return self.base_um_per_px * max(bin_f, 1)

    def set_um_per_px(self, text: str):
        self.base_um_per_px = self._f(text, self.base_um_per_px)
        self._persist()

    def snap_image(self):
        frame = self.latest_frame
        if frame is None:
            self._set_status('No frame to save')
            return
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        ch = self.live_channel if self.live_channel in CHANNELS else 'IMG'
        path = os.path.join(CAPTURES_DIR,
                            f'snap_{ch}_{time.strftime("%Y%m%d_%H%M%S")}.tiff')
        acq.save_image(path, frame)
        self._set_status(f'Saved {os.path.basename(path)}')

    # --------------------------------------------------- stage navigation

    def jog(self, axis: str, um: float):
        if not self.stage_connected:
            return
        self._bg(self.hw.move_rel, axis, um)

    def goto_center(self):
        if not self.stage_connected:
            return
        self._bg(lambda: (self.hw.stage.xycenter()))

    def rehome(self):
        if not self.stage_connected:
            return
        self._bg(self.hw.home_and_center, status_cb=self._set_status,
                 busy_job=True)

    def set_a1_here(self):
        """Calibrate the plate map: current stage position = A1 center."""
        self.a1_stage_um = (self.stage_x_um, self.stage_y_um)
        self._persist()
        self._update_plate_marker()
        self._set_status(f'A1 set to X={self.stage_x_um / 1000:.2f} mm, '
                         f'Y={self.stage_y_um / 1000:.2f} mm')

    def on_well_selected(self, row: int, col: int, rel_x_mm: float, rel_y_mm: float):
        name = self.root.ids.plate_map.plate_format.get_well_name(row, col)
        self.selected_well_str = name

    def on_well_double_click(self, row: int, col: int, rel_x_mm: float, rel_y_mm: float):
        """Double-click on the plate navigator: move stage to that well."""
        name = self.root.ids.plate_map.plate_format.get_well_name(row, col)
        self.selected_well_str = name
        if not self.stage_connected:
            self._set_status(f'{name}: stage not connected')
            return
        if self.a1_stage_um is None:
            self._set_status('Set A1 position first (move to A1, press "Set A1 = here")')
            return
        tx = self.a1_stage_um[0] + rel_x_mm * 1000.0 * self.plate_sign_x
        ty = self.a1_stage_um[1] + rel_y_mm * 1000.0 * self.plate_sign_y
        self._set_status(f'Moving to {name}...')
        def _move():
            self.hw.move_xy(tx, ty)
            z = self.surface.z_at(tx, ty)
            if z is not None:
                self.hw.move_z(z)
            self._set_status(f'At {name}')
        self._bg(_move)

    def on_image_double_click(self, dx_px: float, dy_px: float):
        """Double-click on the live image: center the clicked point."""
        if not self.stage_connected:
            return
        scale = self.effective_um_per_px()
        dx_um = dx_px * scale * self.img_sign_x
        dy_um = dy_px * scale * self.img_sign_y
        self._set_status(f'Centering: dx={dx_um:.0f} um, dy={dy_um:.0f} um')
        def _move():
            self.hw.move_rel('X', dx_um)
            self.hw.move_rel('Y', dy_um)
        self._bg(_move)

    def toggle_invert(self, which: str):
        if which == 'plate_x':
            self.plate_sign_x *= -1.0
        elif which == 'plate_y':
            self.plate_sign_y *= -1.0
        elif which == 'img_x':
            self.img_sign_x *= -1.0
        elif which == 'img_y':
            self.img_sign_y *= -1.0
        self._persist()
        self._set_status(f'Inverted {which}')

    # ---------------------------------------------------------- autofocus

    def _sync_af_fields(self):
        if not self.root:
            return
        ids = self.root.ids
        ids.af_bottom_input.text = f'{self.af_config.z_bottom_um:g}'
        ids.af_top_input.text = f'{self.af_config.z_top_um:g}'
        ids.af_step_input.text = f'{self.af_config.step_um:g}'

    def af_set_bottom_here(self):
        self.af_config.z_bottom_um = self.stage_z_um
        self._sync_af_fields()
        self._persist()
        self._set_status(f'AF bottom = {self.stage_z_um:.1f} um')

    def af_set_top_here(self):
        self.af_config.z_top_um = self.stage_z_um
        self._sync_af_fields()
        self._persist()
        self._set_status(f'AF top = {self.stage_z_um:.1f} um')

    def af_apply_fields(self):
        ids = self.root.ids
        self.af_config.z_bottom_um = self._f(ids.af_bottom_input.text,
                                             self.af_config.z_bottom_um)
        self.af_config.z_top_um = self._f(ids.af_top_input.text,
                                          self.af_config.z_top_um)
        self.af_config.step_um = self._f(ids.af_step_input.text,
                                         self.af_config.step_um)
        self._persist()

    def af_set_anchor_here(self):
        """The autofocus point + channel used to re-reference every cycle."""
        self.af_config.channel = self.selected_channel
        self.surface.set_anchor(self.stage_x_um, self.stage_y_um,
                                self.selected_channel)
        self._persist()
        self._refresh_surface_info()
        self._set_status(f'AF anchor set at X={self.stage_x_um / 1000:.2f}, '
                         f'Y={self.stage_y_um / 1000:.2f} mm ({self.selected_channel})')

    def af_goto_anchor(self):
        if self.surface.anchor_xy is None:
            self._set_status('No AF anchor set')
            return
        x, y = self.surface.anchor_xy
        self._bg(self.hw.move_xy, x, y)

    def run_af_cycle(self):
        """Move to the AF anchor, autofocus over the configured range, and
        update the focus surface so every map point follows the new focus."""
        if self.surface.anchor_xy is None:
            self._set_status('Set the AF anchor first')
            return
        self.af_apply_fields()
        def _job():
            x, y = self.surface.anchor_xy
            ch = self.surface.anchor_channel
            self._set_status('AF cycle: moving to anchor...')
            self.hw.move_xy(x, y)
            cfg = AutofocusConfig(z_bottom_um=self.af_config.z_bottom_um,
                                  z_top_um=self.af_config.z_top_um,
                                  step_um=self.af_config.step_um,
                                  channel=ch)
            best = acq.autofocus_sweep(self.hw, cfg, self._cs(ch),
                                       cancel=self.worker.cancel,
                                       status_cb=self._set_status)
            self.hw.leds_off()
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
            if best is None:
                self._set_status('Autofocus failed / canceled')
                return
            if self.surface.ref_z is None:
                self.surface.set_reference_z(best)
            else:
                self.surface.update_cycle(best)
            self._persist()
            Clock.schedule_once(lambda dt: self._refresh_surface_info(), 0)
            self._set_status(f'AF done: z={best:.1f} um '
                             f'(drift {self.surface.drift_um:+.1f} um)')
        self._bg(_job, busy_job=True)

    # ------------------------------------------------------ focus surface

    def _refresh_surface_info(self):
        s = self.surface
        if s.anchor_xy is None:
            self.surface_info = 'Surface: no anchor'
            return
        z = f'{s.current_z:.1f}' if s.current_z is not None else '?'
        self.surface_info = (f'Surface: anchor ({s.anchor_xy[0] / 1000:.2f}, '
                             f'{s.anchor_xy[1] / 1000:.2f}) mm [{s.anchor_channel}]  '
                             f'z={z} um  drift {s.drift_um:+.1f} um  '
                             f'{len(s.points)} points')

    def surface_add_point_here(self):
        """Record the current position/focus as a surface point. Offsets are
        stored relative to the anchor focus, so each AF cycle updates them."""
        if self.surface.current_z is None:
            self._set_status('Run an AF cycle first to reference the surface')
            return
        self.surface.add_point(self.stage_x_um, self.stage_y_um, self.stage_z_um)
        self._persist()
        self._refresh_surface_info()
        self._set_status(f'Surface point added ({len(self.surface.points)} total)')

    def surface_clear_points(self):
        self.surface.clear_points()
        self._persist()
        self._refresh_surface_info()

    def surface_goto_here_z(self):
        """Move Z to the surface focus for the current XY."""
        z = self.surface.z_at(self.stage_x_um, self.stage_y_um)
        if z is None:
            self._set_status('Surface not referenced yet')
            return
        self._bg(self.hw.move_z, z)

    # ------------------------------------------------------------ shading

    def capture_shading_reference(self):
        """Capture a flat-field reference for the selected channel. Point the
        optics at an empty/uniform field first."""
        ch = self.selected_channel
        def _job():
            self._set_status(f'Shading: capturing reference for {ch}...')
            self.hw.set_channel(ch, self._cs(ch))
            time.sleep(0.3)
            ok = self.shading.capture_reference(self.hw, ch)
            self.hw.leds_off()
            self._set_status(f'Shading reference for {ch} '
                             f'{"captured" if ok else "FAILED"}')
        self._bg(_job, busy_job=True)

    def toggle_shading(self, active: bool):
        self.shading_on = active
        self.shading.enabled = active

    # ------------------------------------------------------- acquisitions

    def _tile_params(self):
        ids = self.root.ids
        rows = int(self._f(ids.tile_rows_input.text, 3))
        cols = int(self._f(ids.tile_cols_input.text, 3))
        overlap = self._f(ids.tile_overlap_input.text, 10.0)
        return max(rows, 1), max(cols, 1), min(max(overlap, 0.0), 60.0)

    def _fov_um(self) -> tuple[float, float]:
        scale = self.effective_um_per_px()
        if self.latest_frame is not None:
            h, w = self.latest_frame.shape[:2]
        else:
            w = h = 1900
        return w * scale, h * scale

    def run_tile_scan(self, pattern: str = 'grid'):
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Tile scan needs stage + camera')
            return
        rows, cols, overlap = self._tile_params()
        fov_x, fov_y = self._fov_um()
        cx, cy = self.stage_x_um, self.stage_y_um
        if pattern == 'spiral':
            rings = max(rows, cols) // 2
            positions = acq.tile_positions_spiral(cx, cy, fov_x, fov_y,
                                                  n_rings=max(rings, 1),
                                                  overlap_pct=overlap)
        else:
            positions = acq.tile_positions_grid(cx, cy, fov_x, fov_y,
                                                rows, cols, overlap_pct=overlap)
        out_dir = acq.timestamp_dir(CAPTURES_DIR, f'tiles_{pattern}')
        def _job():
            acq.run_tile_scan(self.hw, positions, self.channel_settings,
                              self.surface if self.surface.current_z is not None else None,
                              self.shading if self.shading_on else None,
                              self.effective_um_per_px(), out_dir,
                              self.worker.cancel, status_cb=self._set_status)
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    def run_zstack(self, all_channels: bool = False):
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Z-stack needs stage + camera')
            return
        if all_channels:
            channels = [ch for ch in CHANNELS if self.channel_settings[ch].enabled]
        else:
            channels = [ch for ch in CHANNELS
                        if self.channel_settings[ch].enabled
                        and self.channel_settings[ch].zstack.enabled]
        if not channels:
            self._set_status('No channels enabled for z-stack')
            return
        center_z = self.surface.z_at(self.stage_x_um, self.stage_y_um)
        if center_z is None:
            center_z = self.stage_z_um
        out_dir = acq.timestamp_dir(CAPTURES_DIR, 'zstack')
        def _job():
            acq.run_z_stack(self.hw, self.channel_settings, channels,
                            center_z, self.shading if self.shading_on else None,
                            out_dir, self.worker.cancel,
                            status_cb=self._set_status)
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    # -------------------------------------------------------- persistence

    def _persist(self):
        payload = {
            'channel_settings': acq.channel_settings_to_dict(self.channel_settings),
            'a1_stage_um': self.a1_stage_um,
            'plate_sign_x': self.plate_sign_x,
            'plate_sign_y': self.plate_sign_y,
            'base_um_per_px': self.base_um_per_px,
            'img_sign_x': self.img_sign_x,
            'img_sign_y': self.img_sign_y,
            'af_config': {
                'z_bottom_um': self.af_config.z_bottom_um,
                'z_top_um': self.af_config.z_top_um,
                'step_um': self.af_config.step_um,
                'channel': self.af_config.channel,
            },
            'surface': self.surface.to_dict(),
        }
        try:
            acq.save_settings(payload)
        except Exception:
            pass

    def _load_persisted(self):
        d = acq.load_settings()
        if not d:
            self.shading.load_all()
            return
        try:
            self.channel_settings = acq.channel_settings_from_dict(
                d.get('channel_settings', {}))
            a1 = d.get('a1_stage_um')
            self.a1_stage_um = tuple(a1) if a1 else None
            self.plate_sign_x = float(d.get('plate_sign_x', 1.0))
            self.plate_sign_y = float(d.get('plate_sign_y', 1.0))
            self.base_um_per_px = float(d.get('base_um_per_px', DEFAULT_UM_PER_PX))
            self.img_sign_x = float(d.get('img_sign_x', 1.0))
            self.img_sign_y = float(d.get('img_sign_y', 1.0))
            af = d.get('af_config', {})
            self.af_config = AutofocusConfig(
                z_bottom_um=float(af.get('z_bottom_um', 0.0)),
                z_top_um=float(af.get('z_top_um', 200.0)),
                step_um=float(af.get('step_um', 10.0)),
                channel=af.get('channel', 'BF'))
            if d.get('surface'):
                self.surface.from_dict(d['surface'])
        except Exception:
            pass
        self.shading.load_all()

    # ---------------------------------------------------------- fail safe

    def on_stop(self):
        """Fail safe on application exit: cancel any acquisition, turn every
        LED off, and stop/disconnect the camera."""
        try:
            self.worker.request_cancel()
        except Exception:
            pass
        try:
            self.hw.leds_off()
        except Exception:
            pass
        cam = self.hw.camera
        if cam is not None:
            try:
                cam.stop_grabbing()
            except Exception:
                pass
            try:
                cam.disconnect()
            except Exception:
                pass
        try:
            self._persist()
        except Exception:
            pass
        print('[LumaViewPro] Fail-safe shutdown complete (LEDs off, camera stopped).',
              flush=True)


if __name__ == '__main__':
    LumaViewProApp().run()
