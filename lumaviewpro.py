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
import math

if not hasattr(enum, 'StrEnum'):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

driver_paths = [
    r"C:\Program Files (x86)\LumaViewPro-stock",
    r"C:\Program Files (x86)\Etaluma\LumaViewPro",
]
# APPEND the stock install paths so this repo's own modules (lvp_logger,
# modules/, image_utils, ...) always win; the stock drivers (pyloncamera,
# ledboard, motorboard) have no same-named files here, so they still
# resolve from the stock folder.
for p in driver_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.append(p)


def _ensure_lvp_logger():
    """Make `from lvp_logger import logger` work for the stock drivers.

    With this repo first on sys.path, a bare import resolves to the repo's
    own lvp_logger (requires Python 3.12+ for its module chain). If that
    fails (older interpreter, missing deps), fall back to a minimal
    stand-in module that just provides `logger`.
    """
    import types
    import logging
    try:
        import lvp_logger
        if hasattr(lvp_logger, 'logger'):
            return
    except Exception:
        sys.modules.pop('lvp_logger', None)
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

# LED current ceiling for the sliders/fields (mA).
LED_MAX_MA = 600
import labware_defs
from acquisition import (CHANNELS, LED_CHANNEL_MAP, ChannelSettings,
                         AutofocusConfig, ScopeHW, FocusSurface,
                         ShadingCorrector, AcquisitionWorker, PlateCalibration)

Factory.register('ZoomPanCameraWidget', cls=plate_navigator.ZoomPanCameraWidget)
Factory.register('WellPlateMapWidget', cls=plate_navigator.WellPlateMapWidget)
Factory.register('InteractiveScaledPlateCanvas', cls=plate_navigator.InteractiveScaledPlateCanvas)
Factory.register('Collapsible', cls=plate_navigator.Collapsible)

LAYOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'layouts')


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
PROTOCOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'protocols')

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
    af_channel = StringProperty('BF')   # the ONE channel used for autofocus
    bit_depth_options = ListProperty(['8-bit'])
    bit_depth = StringProperty('8-bit')
    binning = StringProperty('1')
    crosshair_on = BooleanProperty(False)
    autoscale_on = BooleanProperty(True)
    clip_overlay_on = BooleanProperty(True)
    exp_min_ms = NumericProperty(0.02)
    exp_max_ms = NumericProperty(1000.0)
    gain_min_db = NumericProperty(0.0)
    gain_max_db = NumericProperty(24.0)
    sat_pct = NumericProperty(0.0)
    camera_auto_warning = StringProperty('')
    high_stability_on = BooleanProperty(False)
    af_method = StringProperty('two_pass')
    shading_info = StringProperty('')
    initializing = BooleanProperty(False)
    frame_stats = StringProperty('No frames yet')
    shading_on = BooleanProperty(False)
    surface_info = StringProperty('Surface: no anchor')
    worker_busy = BooleanProperty(False)
    labware_options = ListProperty(['96-well plate'])
    labware_name = StringProperty('96-well plate')
    builtin_labware_names = ListProperty([])
    cal_active = BooleanProperty(False)
    cal_label = StringProperty('')
    cal_info = StringProperty('Plate not calibrated')
    use_surface_on = BooleanProperty(True)
    use_z_offsets_on = BooleanProperty(True)
    xy_step_um = NumericProperty(100.0)
    stage_x_um = NumericProperty(0.0)
    stage_y_um = NumericProperty(0.0)
    stage_z_um = NumericProperty(0.0)
    af_each_cycle_on = BooleanProperty(True)
    output_dir = StringProperty(CAPTURES_DIR)
    positions_info = StringProperty('No positions defined')
    pos_review_active = BooleanProperty(False)
    pos_review_label = StringProperty('')
    protocol_name = StringProperty('protocol1')
    protocol_files = ListProperty([])
    protocol_dir = StringProperty(PROTOCOLS_DIR)
    layout_files = ListProperty(['(none)'])

    # -------------------------------------------------------------- setup

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hw = ScopeHW()
        self.worker = AcquisitionWorker()
        self.surface = FocusSurface()
        self.shading = ShadingCorrector()
        self.shading_focus_z: float | None = None
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
        self._mosaic = None
        self._mosaic_meta = None

        # Stage/plate mapping calibration
        self.a1_stage_um: tuple[float, float] | None = None
        self.plate_sign_x = 1.0    # +1: stage +X moves toward column 12
        self.plate_sign_y = 1.0    # +1: stage +Y moves toward row H
        self.base_um_per_px = DEFAULT_UM_PER_PX
        self.img_sign_x = 1.0      # double-click-to-center direction trims
        self.img_sign_y = 1.0
        # Measured pixel->stage mapping: {'A': 2x2 (px shift per stage um),
        # 'binning': int}. Set by auto-calibration; None = use um/px+signs.
        self.img_cal: dict | None = None
        # In-focus reference template for high-stability AF.
        self.af_template = None
        try:
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'af_template.npy')
            if os.path.exists(p):
                self.af_template = np.load(p)
        except Exception:
            pass

        self.stage_x_um = 0.0
        self.stage_y_um = 0.0
        self.stage_z_um = 0.0  # NumericProperties; values reset here

        # Labware + plate calibration
        self.labware_lib = labware_defs.LabwareLibrary()
        self.builtin_labware_names = list(self.labware_lib.builtins)
        self.current_labware = self.labware_lib.get('96-well plate')
        self.plate_cal: PlateCalibration | None = None   # solved calibration
        self.cal_wizard: PlateCalibration | None = None  # in-progress wizard

        # Protocol interaction positions: dicts with name, x_um, y_um, z_um.
        # z_um is stored drift-normalized (minus the surface drift at the
        # time it was set) so per-cycle AF re-referencing shifts every
        # position consistently.
        self.protocol_positions: list[dict] = []
        self._pos_review_index = 0
        self.travel_clearance_um = 500.0

        self._load_persisted()

    def build(self):
        pass  # root comes from lumaviewpro.kv

    def on_start(self):
        from kivy.core.window import Window
        Window.maximize()
        self.root_widget = self.root
        viewport = self.root.ids.get('main_camera_viewport')
        if viewport:
            viewport.on_image_double_click = self.on_image_double_click
            viewport.on_focus_scroll = self.on_focus_scroll
        plate = self.root.ids.get('plate_map')
        if plate:
            plate.on_well_double_click = self.on_well_double_click
            plate.on_well_selected = self.on_well_selected
            plate.set_labware(self.current_labware)
        self.labware_options = self.labware_lib.names()
        self.labware_name = self.current_labware.name
        self._refresh_cal_info()
        self._refresh_positions()
        self._refresh_protocol_files()
        self._refresh_layout_files()
        self._sync_channel_fields()
        self._sync_af_fields()
        self._sync_shading_fields()
        Clock.schedule_once(lambda dt: self._start_init_thread(), 0.1)
        Clock.schedule_interval(self.update_live_feed, 1.0 / 30.0)
        threading.Thread(target=self._position_poller, daemon=True).start()
        from kivy.core.window import Window
        Window.bind(on_key_down=self._on_key_down, on_key_up=self._on_key_up)

    def _on_key_up(self, window, key, scancode):
        if key in (304, 303):
            self._shift_down = False
        elif key in (305, 306):
            self._ctrl_down = False
        return False

    def _text_input_focused(self) -> bool:
        if not self.root:
            return False
        return any(getattr(w, 'focus', False) for w in self.root.walk())

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        """Keyboard nudge: arrows = XY jog by the step size (Shift = 10x);
        PgUp/PgDn = Z +/-10 um (Shift = 100 um). Also tracks Shift/Ctrl
        for mouse-wheel focus speed."""
        if key in (304, 303):
            self._shift_down = True
            return False
        if key in (305, 306):
            self._ctrl_down = True
            return False
        if self._text_input_focused():
            return False
        mult = 10.0 if 'shift' in (modifiers or []) else 1.0
        s = self.xy_step_um * mult
        if key == 276:    self.jog('X', -s)
        elif key == 275:  self.jog('X', s)
        elif key == 273:  self.jog('Y', s)
        elif key == 274:  self.jog('Y', -s)
        elif key == 280:  self.jog('Z', 10.0 * mult)
        elif key == 281:  self.jog('Z', -10.0 * mult)
        else:
            return False
        return True

    # --------------------------------------------------- hardware startup

    def _start_init_thread(self):
        threading.Thread(target=self._init_hardware, daemon=True).start()

    def _set_status(self, msg: str):
        Clock.schedule_once(lambda dt: setattr(self, 'status_msg', msg), 0)

    def _init_hardware(self):
        Clock.schedule_once(lambda dt: setattr(self, 'initializing', True), 0)
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
                Clock.schedule_once(lambda dt: self._refresh_camera_limits(), 0)
                self._check_camera_manual()
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
            if self.led_connected:
                # The firmware reset issued during connect() leaves a
                # partial line in the serial buffer, so the first command
                # gets a garbled response ('...mand not recognized').
                # Absorb it with a STATUS exchange before real commands.
                try:
                    led.exchange_command('STATUS')
                except Exception:
                    pass
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
        Clock.schedule_once(lambda dt: setattr(self, 'initializing', False), 0)

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
            frame = self.shading.apply(frame, ch, self.stage_z_um)
        self.latest_frame = frame
        self.frame_count += 1
        now = time.time()
        if now - self.last_log_time >= 5.0:
            self.last_log_time = now
            print(f'[CAM STREAM] Frame #{self.frame_count}: shape={frame.shape}, '
                  f'dtype={frame.dtype}, min={frame.min()}, max={frame.max()}, '
                  f'mean={frame.mean():.1f}', flush=True)
        if self.frame_count % 15 == 0:
            sat = acq.saturated_fraction(frame, self.significant_bits) * 100.0
            self.sat_pct = sat
            full = acq.saturation_level(self.significant_bits)
            self.frame_stats = (f'Live #{self.frame_count}  {frame.shape[1]}x'
                                f'{frame.shape[0]}  {self.significant_bits}-bit  '
                                f'min {frame.min()}  max {frame.max()}/{full}  '
                                f'mean {frame.mean():.1f}  '
                                f'SAT {sat:.2f}%{self.camera_auto_warning}')
        viewport = self.root.ids.get('main_camera_viewport') if self.root else None
        if viewport:
            # During a tile scan show the growing mosaic instead of live frames.
            if self.worker_busy and self._mosaic is not None:
                viewport.autoscale = False
                viewport.update_frame_texture(self._mosaic, significant_bits=8)
                return
            tint = CHANNEL_LUTS.get(ch, (1.0, 1.0, 1.0))
            viewport.autoscale = self.autoscale_on
            viewport.update_frame_texture(
                frame, tint=tint, significant_bits=self.significant_bits,
                clip_level=(acq.saturation_level(self.significant_bits)
                            if self.clip_overlay_on else None))

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
                self._sync_z_slider(z)
            Clock.schedule_once(_apply, 0)

    def _update_plate_marker(self):
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is None:
            return
        rel = self.stage_to_plate_rel(self.stage_x_um, self.stage_y_um)
        if rel is None:
            return
        plate.set_marker_plate_mm(rel[0], rel[1])

    # -------------------------------------------- plate <-> stage transform

    def _default_a1_stage(self) -> tuple[float, float]:
        """Approximate A1 position assuming the plate is centered on the
        stage (X 120 mm, Y 80 mm travel) - used for the FOV marker before
        any calibration has been done."""
        lw = self.current_labware
        return (60000.0 - (lw.plate_length_mm / 2.0 - lw.a1_offset_x_mm) * 1000.0,
                40000.0 - (lw.plate_width_mm / 2.0 - lw.a1_offset_y_mm) * 1000.0)

    def plate_rel_to_stage(self, rel_x_mm: float,
                           rel_y_mm: float) -> tuple[float, float] | None:
        """Plate position relative to the A1 center (mm) -> stage um.

        Uses the full plate calibration (rotation-aware affine) when solved,
        else the quick 'Set A1 = here' anchor with the flip signs, else an
        approximate plate-centered-on-stage assumption."""
        if self.plate_cal is not None and self.plate_cal.matrix is not None:
            lw = self.current_labware
            return self.plate_cal.plate_mm_to_stage_um(
                lw.a1_offset_x_mm + rel_x_mm, lw.a1_offset_y_mm + rel_y_mm)
        a1 = self.a1_stage_um or self._default_a1_stage()
        return (a1[0] + rel_x_mm * 1000.0 * self.plate_sign_x,
                a1[1] + rel_y_mm * 1000.0 * self.plate_sign_y)

    def stage_to_plate_rel(self, x_um: float,
                           y_um: float) -> tuple[float, float] | None:
        if self.plate_cal is not None and self.plate_cal.matrix is not None:
            p = self.plate_cal.stage_um_to_plate_mm(x_um, y_um)
            if p is None:
                return None
            lw = self.current_labware
            return (p[0] - lw.a1_offset_x_mm, p[1] - lw.a1_offset_y_mm)
        a1 = self.a1_stage_um or self._default_a1_stage()
        return ((x_um - a1[0]) * self.plate_sign_x / 1000.0,
                (y_um - a1[1]) * self.plate_sign_y / 1000.0)

    # ----------------------------------------------------- labware library

    def select_labware(self, name: str, force: bool = False):
        lw = self.labware_lib.get(name)
        if lw is None:
            return
        if not force and lw is self.current_labware and \
                self.labware_name == lw.name:
            return   # spinner echo of the current selection
        self.current_labware = lw
        self.labware_name = lw.name
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_labware(lw)
        # A calibration belongs to a specific labware/mounting.
        if self.plate_cal is not None and \
                getattr(self.plate_cal.labware, 'name', '') != lw.name:
            self.plate_cal = None
            self._refresh_cal_info()
        self._persist()

    def duplicate_labware(self):
        dup = self.labware_lib.duplicate(self.labware_name)
        if dup is None:
            return
        self.labware_options = self.labware_lib.names()
        self.select_labware(dup.name)
        self._set_status(f'Created "{dup.name}" - use Edit to modify it')

    def delete_labware(self):
        name = self.labware_name
        lw = self.labware_lib.get(name)
        if lw is None or lw.builtin:
            self._set_status('Built-in labware cannot be deleted')
            return
        from kivy.uix.button import Button
        box = BoxLayout(orientation='vertical', spacing=6, padding=6)
        box.add_widget(Label(text=f'Delete custom labware\n"{name}"?',
                             halign='center'))
        btns = BoxLayout(size_hint_y=None, height='36dp', spacing=6)
        yes, no = Button(text='Delete'), Button(text='Cancel')
        btns.add_widget(yes)
        btns.add_widget(no)
        box.add_widget(btns)
        popup = Popup(title='Delete labware', content=box,
                      size_hint=(None, None), size=('320dp', '160dp'))

        def _do(*_):
            popup.dismiss()
            self.labware_lib.delete(name)
            self.labware_options = self.labware_lib.names()
            self.select_labware('96-well plate')
            self._set_status(f'Deleted "{name}"')

        yes.bind(on_press=_do)
        no.bind(on_press=popup.dismiss)
        popup.open()

    def new_labware(self):
        """Design a new labware from scratch (SBS footprint defaults)."""
        blank = labware_defs.LabwareDef(
            name=self.labware_lib.unique_name('My labware'),
            rows=8, cols=12, spacing_x_mm=9.0, spacing_y_mm=9.0,
            well_size_x_mm=6.4, well_size_y_mm=6.4, shape='round')
        self.open_labware_editor(blank, is_new=True)

    def open_labware_editor(self, src=None, is_new: bool = False):
        """Labware designer with a live preview. Editing a built-in saves a
        new custom copy; editing a custom updates (or renames) it."""
        from kivy.uix.gridlayout import GridLayout
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button
        from kivy.uix.spinner import Spinner

        if src is None:
            src = self.current_labware
        # The custom being edited (may be renamed); None = save as new.
        replacing = None if (is_new or src.builtin) else src.name
        if src.builtin and not is_new:
            suggested = self.labware_lib.unique_name(f'{src.name} (custom)')
        else:
            suggested = src.name

        grid = GridLayout(cols=2, spacing=4, size_hint_y=None,
                          row_default_height='28dp', row_force_default=True)
        grid.bind(minimum_height=grid.setter('height'))
        fields = {}
        spec = [
            ('name', 'Name', suggested),
            ('rows', 'Rows', src.rows),
            ('cols', 'Columns', src.cols),
            ('spacing_x_mm', 'Spacing X (mm)', src.spacing_x_mm),
            ('spacing_y_mm', 'Spacing Y (mm)', src.spacing_y_mm),
            ('well_size_x_mm', 'Well size X (mm)', src.well_size_x_mm),
            ('well_size_y_mm', 'Well size Y (mm)', src.well_size_y_mm),
            ('a1_offset_x_mm', 'A1 centre from left (mm)', src.a1_offset_x_mm),
            ('a1_offset_y_mm', 'A1 centre from top (mm)', src.a1_offset_y_mm),
            ('plate_length_mm', 'Plate length X (mm)', src.plate_length_mm),
            ('plate_width_mm', 'Plate width Y (mm)', src.plate_width_mm),
        ]
        for key, label, value in spec:
            grid.add_widget(Label(text=label, font_size='12sp'))
            txt = f'{value:g}' if isinstance(value, float) else str(value)
            ti = TextInput(text=txt, multiline=False, write_tab=False,
                           font_size='12sp')
            grid.add_widget(ti)
            fields[key] = ti
        grid.add_widget(Label(text='Well shape', font_size='12sp'))
        shape_spin = Spinner(text=src.shape, values=['round', 'square'],
                             font_size='12sp')
        grid.add_widget(shape_spin)

        preview = plate_navigator.WellPlateMapWidget(size_hint_y=None,
                                                     height='200dp')
        info = Label(text='', font_size='11sp', size_hint_y=None,
                     height='44dp', halign='left', valign='top')
        info.bind(size=lambda w, s: setattr(w, 'text_size', s))

        box = BoxLayout(orientation='vertical', spacing=6, padding=6)
        box.add_widget(preview)
        box.add_widget(grid)
        box.add_widget(info)
        btns = BoxLayout(size_hint_y=None, height='36dp', spacing=6)
        center_btn = Button(text='Centre grid on plate')
        save_btn = Button(text='Save')
        cancel_btn = Button(text='Cancel')
        for b in (center_btn, save_btn, cancel_btn):
            btns.add_widget(b)
        box.add_widget(btns)
        if is_new:
            title = 'New labware'
        elif src.builtin:
            title = f'Labware designer - new copy of {src.name}'
        else:
            title = f'Labware designer - {src.name}'
        popup = Popup(title=title, content=box, size_hint=(None, None),
                      size=('460dp', '720dp'))
        red, amber, green = (1.0, 0.45, 0.35, 1), (1.0, 0.8, 0.3, 1), \
            (0.6, 0.9, 0.6, 1)

        def _read():
            """Current design from the fields, or (None, error text)."""
            try:
                vals = {k: fields[k].text.strip() for k, _, _ in spec}
                d = labware_defs.LabwareDef(
                    name=vals['name'],
                    rows=int(float(vals['rows'])),
                    cols=int(float(vals['cols'])),
                    shape=shape_spin.text,
                    **{k: float(vals[k]) for k, _, _ in spec
                       if k.endswith('_mm')})
            except ValueError:
                return None, 'A field does not contain a valid number'
            return d, None

        def _problems(d):
            errors, warnings = labware_defs.validate_labware(d)
            name_err = self.labware_lib.check_name(d.name, replacing)
            if name_err:
                errors = [name_err] + [e for e in errors
                                       if e != 'Name is empty']
            return errors, warnings

        def _refresh(*_):
            d, err = _read()
            if d is None:
                info.text, info.color = err, red
                save_btn.disabled = True
                return
            errors, warnings = _problems(d)
            geometry_ok = not [e for e in errors if 'name' not in e.lower()
                               and 'design' not in e]
            if geometry_ok:
                preview.set_labware(d)
            save_btn.disabled = bool(errors)
            if errors:
                info.text, info.color = 'Cannot save: ' + '; '.join(errors), red
            elif warnings:
                info.text, info.color = 'Note: ' + '; '.join(warnings), amber
            else:
                info.text = (f'{d.rows} x {d.cols} = {d.rows * d.cols} '
                             f'{d.shape} wells. Preview: plate seen from '
                             f'above, A1 top-left.')
                info.color = green

        def _center(*_):
            d, err = _read()
            if d is None:
                info.text, info.color = err, red
                return
            ax, ay = labware_defs.centered_a1_offsets(
                d.rows, d.cols, d.spacing_x_mm, d.spacing_y_mm,
                d.plate_length_mm, d.plate_width_mm)
            fields['a1_offset_x_mm'].text = f'{ax:.3f}'.rstrip('0').rstrip('.')
            fields['a1_offset_y_mm'].text = f'{ay:.3f}'.rstrip('0').rstrip('.')

        def _save(*_):
            d, _err = _read()
            if d is None or _problems(d)[0]:
                _refresh()
                return
            self.labware_lib.save(d, replacing=replacing)
            self.labware_options = self.labware_lib.names()
            self.select_labware(d.name, force=True)
            popup.dismiss()
            self._set_status(f'Saved labware "{d.name}"')

        for ti in fields.values():
            ti.bind(text=_refresh)
        shape_spin.bind(text=_refresh)
        center_btn.bind(on_press=_center)
        save_btn.bind(on_press=_save)
        cancel_btn.bind(on_press=popup.dismiss)
        _refresh()
        popup.open()

    # ----------------------------------------------- plate calibration wizard

    def cal_start(self):
        """Begin the guided LAS X-style plate calibration."""
        if not self.stage_connected:
            self._set_status('Calibration needs the stage')
            return
        self.cal_wizard = PlateCalibration(self.current_labware)
        self.cal_active = True
        self._cal_show_current(move=False)
        self._set_status('Calibration: drive the FOV center to the marked '
                         'position, then press Mark')

    def _cal_show_current(self, move: bool = True):
        wiz = self.cal_wizard
        if wiz is None:
            return
        t = wiz.current
        n = len(wiz.targets)
        done = '*' if t.marked_stage_um is not None else ''
        self.cal_label = f'[{wiz.index + 1}/{n}] {t.label} {done}'
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(t.plate_mm)
        if move:
            est = wiz.estimate_stage_for(
                t, self.plate_rel_to_stage
                if (self.plate_cal or self.a1_stage_um) else None)
            if est is not None:
                self._bg(self.hw.move_xy, est[0], est[1])

    def cal_mark(self):
        """Record the current stage position for the shown target and
        advance to the next one."""
        wiz = self.cal_wizard
        if wiz is None:
            return
        wiz.mark_current(self.stage_x_um, self.stage_y_um)
        self._set_status(f'Marked {wiz.current.label}')
        if wiz.index == 3:
            # All four A1 edges marked: provisional solve enables
            # auto-driving to the two long-range targets.
            wiz.provisional_solve()
        if wiz.complete:
            self.cal_finish()
            return
        if wiz.next():
            self._cal_show_current(move=True)

    def cal_next(self):
        if self.cal_wizard and self.cal_wizard.next():
            self._cal_show_current(move=True)

    def cal_prev(self):
        if self.cal_wizard and self.cal_wizard.prev():
            self._cal_show_current(move=True)

    def cal_cancel(self):
        self.cal_wizard = None
        self.cal_active = False
        self.cal_label = ''
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(None)
        self._set_status('Calibration canceled')

    def cal_finish(self):
        wiz = self.cal_wizard
        if wiz is None:
            return
        if not wiz.complete:
            self._set_status('Calibration: not all positions are marked yet')
            return
        if not wiz.solve():
            self._set_status('Calibration solve FAILED - review the marks')
            return
        self.plate_cal = wiz
        self.cal_wizard = None
        self.cal_active = False
        self.cal_label = ''
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(None)
        self._refresh_cal_info()
        self._persist()
        self._update_plate_marker()
        self._set_status('Plate calibrated: ' + self.cal_info)

    def _refresh_cal_info(self):
        c = self.plate_cal
        if c is None or c.matrix is None:
            self.cal_info = 'Plate not calibrated'
            return
        parts = []
        if c.measured_well_diameter_mm is not None:
            parts.append(f'well {c.measured_well_diameter_mm:.2f} mm')
        if c.measured_spacing_x_mm is not None:
            parts.append(f'pitch X {c.measured_spacing_x_mm:.2f} mm')
        if c.measured_spacing_y_mm is not None:
            parts.append(f'pitch Y {c.measured_spacing_y_mm:.2f} mm')
        if c.rotation_deg is not None:
            parts.append(f'rot {c.rotation_deg:+.2f} deg')
        self.cal_info = 'Calibrated: ' + ', '.join(parts)

    def cal_save_measured_labware(self):
        """Store the measured geometry as a custom copy of the labware."""
        c = self.plate_cal
        if c is None or c.matrix is None:
            self._set_status('Calibrate the plate first')
            return
        import dataclasses
        src = self.current_labware
        d = dataclasses.replace(
            src,
            name=f'{src.name} (measured)',
            builtin=False,
            spacing_x_mm=c.measured_spacing_x_mm or src.spacing_x_mm,
            spacing_y_mm=c.measured_spacing_y_mm or src.spacing_y_mm,
            well_size_x_mm=c.measured_well_diameter_mm or src.well_size_x_mm,
            well_size_y_mm=c.measured_well_diameter_mm or src.well_size_y_mm,
        )
        self.labware_lib.save(d)
        self.labware_options = self.labware_lib.names()
        self._set_status(f'Saved measured geometry as "{d.name}"')

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
        self._refresh_shading_info()

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
        af_chk = ids.get('af_channel_chk')
        if af_chk is not None:
            self._af_chk_sync = True
            af_chk.active = (self.selected_channel == self.af_channel)
            self._af_chk_sync = False
        self._sync_channel_sliders()

    @staticmethod
    def _f(widget_text, fallback):
        try:
            return float(widget_text)
        except Exception:
            return fallback

    def set_af_channel(self, active: bool):
        """Designate the selected channel as THE autofocus channel (single
        choice: all other channels' protocol Z derives from its AF result
        via focus points and the chromatic corrections)."""
        if getattr(self, '_af_chk_sync', False):
            return
        if active:
            self.af_channel = self.selected_channel
            self.af_config.channel = self.af_channel
            self.surface.anchor_channel = self.af_channel
            self._persist()
            self._set_status(f'Autofocus channel: {self.af_channel}')
        else:
            # Re-check: exactly one AF channel must exist.
            Clock.schedule_once(lambda dt: self._sync_channel_fields(), 0)

    def apply_channel_fields(self):
        """Read the per-channel fields into the selected channel's settings
        and apply them to hardware if that channel is live."""
        if not self.root:
            return
        ids = self.root.ids
        cs = self._cs()
        cs.exposure_ms = min(max(self._f(ids.exposure_input.text,
                                         cs.exposure_ms), self.exp_min_ms),
                             self.exp_max_ms)
        cs.gain_db = min(max(self._f(ids.gain_input.text, cs.gain_db),
                             self.gain_min_db), self.gain_max_db)
        cs.led_power = int(min(max(round(self._f(ids.led_input.text,
                                                 cs.led_power)), 0),
                               LED_MAX_MA))
        cs.z_offset_um = self._f(ids.zoffset_input.text, cs.z_offset_um)
        cs.zstack.range_um = self._f(ids.zstack_range_input.text, cs.zstack.range_um)
        cs.zstack.step_um = self._f(ids.zstack_step_input.text, cs.zstack.step_um)
        cs.enabled = bool(ids.channel_enabled_chk.active)
        cs.zstack.enabled = bool(ids.zstack_enabled_chk.active)
        self._sync_channel_fields()   # show the clamped values
        self._refresh_shading_info()  # z-stack / offset change the planes
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

    def cycle_channel(self):
        """Single-toggle channel skip: advance live view to the next channel
        (wraps around; OFF -> first channel)."""
        if self.live_channel in CHANNELS:
            idx = (CHANNELS.index(self.live_channel) + 1) % len(CHANNELS)
        else:
            idx = 0
        self.set_live(CHANNELS[idx])

    # -------- sliders: sync with the text fields, debounce live apply

    def slider_changed(self, kind: str, pos: float):
        """Exposure and LED sliders are logarithmic (equal travel = equal
        brightness ratio) over what the hardware accepts; gain is in dB,
        already logarithmic, so its slider is linear over the camera's
        range."""
        if not self.root or getattr(self, '_slider_sync', False):
            return
        ids = self.root.ids
        cs = self._cs()
        if kind == 'exposure':
            cs.exposure_ms = self._round_sig(acq.log_slider_value(
                pos, self.exp_min_ms, self.exp_max_ms))
            ids.exposure_input.text = f'{cs.exposure_ms:g}'
        elif kind == 'gain':
            cs.gain_db = round(float(pos), 1)
            ids.gain_input.text = f'{cs.gain_db:g}'
        elif kind == 'led':
            cs.led_power = int(round(acq.log_slider_value(
                pos, 1.0, LED_MAX_MA, zero_at_start=True)))
            ids.led_input.text = f'{cs.led_power}'
        if self.live_channel == self.selected_channel:
            self._live_apply_debounced()

    @staticmethod
    def _round_sig(v: float, sig: int = 3) -> float:
        if v <= 0:
            return 0.0
        return round(v, max(sig - 1 - int(math.floor(math.log10(v))), 0))

    def _live_apply_debounced(self):
        if not hasattr(self, '_apply_trigger'):
            def _do(dt):
                ch = self.selected_channel
                if self.live_channel == ch and not self.worker_busy:
                    self._bg(self.hw.set_channel, ch, self._cs(ch))
            self._apply_trigger = Clock.create_trigger(_do, 0.08)
        self._apply_trigger()

    def _sync_channel_sliders(self):
        if not self.root:
            return
        ids = self.root.ids
        cs = self._cs()
        self._slider_sync = True
        try:
            for key, pos in (
                    ('exposure_slider', acq.log_slider_pos(
                        cs.exposure_ms, self.exp_min_ms, self.exp_max_ms)),
                    ('gain_slider', min(max(cs.gain_db, self.gain_min_db),
                                        self.gain_max_db)),
                    ('led_slider', acq.log_slider_pos(
                        cs.led_power, 1.0, LED_MAX_MA, zero_at_start=True))):
                w = ids.get(key)
                if w is not None:
                    w.value = float(pos)
        finally:
            self._slider_sync = False

    def toggle_autoscale(self):
        self.autoscale_on = not self.autoscale_on
        self._persist()
        self._set_status('Display autoscale ' + ('ON - brightness is '
                         're-stretched per frame, so exposure/LED changes '
                         'will not look brighter/darker' if self.autoscale_on
                         else 'off - display follows the real signal'))

    def toggle_clip_overlay(self):
        self.clip_overlay_on = not self.clip_overlay_on
        self._persist()

    def _check_camera_manual(self):
        """Make sure the camera's own auto gain/exposure is really off
        (it would override the sliders) and flag it if it will not stay
        off. Runs on the hardware-init thread."""
        ok = self.hw.ensure_manual_exposure()
        state = self.hw.camera_auto_state()
        print(f'[CAM] auto state after connect: {state} (manual ok={ok})',
              flush=True)
        warn = '' if ok else '   !! CAMERA AUTO EXPOSURE/GAIN STILL ON'
        Clock.schedule_once(
            lambda dt: setattr(self, 'camera_auto_warning', warn), 0)
        if not ok:
            self._set_status(f'WARNING: camera auto gain/exposure could not '
                             f'be turned off ({state}) - sliders will be '
                             f'overridden')

    def _refresh_camera_limits(self):
        """Fit the sliders to the connected camera's real ranges."""
        lo, hi, glo, ghi = self.hw.camera_limits()
        self.exp_min_ms, self.exp_max_ms = lo, hi
        self.gain_min_db, self.gain_max_db = glo, ghi
        self._sync_channel_sliders()

    def nudge(self, kind: str, direction: int):
        """Fine step buttons: exposure / LED by ~10 %, gain by 1 dB."""
        cs = self._cs()
        if kind == 'exposure':
            v = cs.exposure_ms * (1.1 if direction > 0 else 1 / 1.1)
            cs.exposure_ms = self._round_sig(
                min(max(v, self.exp_min_ms), self.exp_max_ms))
        elif kind == 'led':
            cur = cs.led_power
            v = max(cur * 1.1, cur + 1) if direction > 0 else \
                min(cur / 1.1, cur - 1)
            cs.led_power = int(min(max(round(v), 0), LED_MAX_MA))
        elif kind == 'gain':
            cs.gain_db = round(min(max(cs.gain_db + direction,
                                       self.gain_min_db), self.gain_max_db), 1)
        self._sync_channel_fields()
        if self.live_channel == self.selected_channel:
            self._live_apply_debounced()

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
        os.makedirs(self._out_base(), exist_ok=True)
        ch = self.live_channel if self.live_channel in CHANNELS else 'IMG'
        path = os.path.join(self._out_base(),
                            f'snap_{ch}_{time.strftime("%Y%m%d_%H%M%S")}.tiff')
        acq.save_image(path, frame)
        self._set_status(f'Saved {os.path.basename(path)}')

    # --------------------------------------------------- stage navigation

    def jog(self, axis: str, um: float):
        if not self.stage_connected:
            return
        self._bg(self.hw.move_rel, axis, um)

    def jog_xy(self, dx_um: float, dy_um: float):
        """Diagonal jog: both axes move together along the vector."""
        if not self.stage_connected:
            return
        self._bg(self.hw.move_xy_rel, dx_um, dy_um)

    # ------------------------------------------------------- Z focus control

    def on_focus_scroll(self, direction: int):
        """Mouse wheel over the live image: focus jog. Default 10 um per
        notch; Shift = 2 um (fine), Ctrl = 100 um (coarse)."""
        if getattr(self, '_shift_down', False):
            step = 2.0
        elif getattr(self, '_ctrl_down', False):
            step = 100.0
        else:
            step = 10.0
        self.jog('Z', direction * step)

    def _sync_z_slider(self, z_um: float):
        ids = self.root.ids if self.root else None
        if not ids:
            return
        # Don't fight the user: skip sync shortly after a manual slider move.
        if time.time() - getattr(self, '_z_user_ts', 0.0) < 1.5:
            return
        slider = ids.get('z_slider')
        if slider is not None:
            self._z_slider_sync = True
            slider.value = min(max(z_um, slider.min), slider.max)
            self._z_slider_sync = False

    def z_slider_changed(self, value: float):
        """Absolute Z from the focus slider (debounced while dragging)."""
        if getattr(self, '_z_slider_sync', False):
            return
        if not self.stage_connected:
            return
        self._z_user_ts = time.time()
        self._z_target = float(value)
        if not hasattr(self, '_z_trigger'):
            def _go(dt):
                self._bg(self.hw.move_z, self._z_target)
            self._z_trigger = Clock.create_trigger(_go, 0.2)
        self._z_trigger()

    def z_goto(self, text: str):
        """Absolute Z position typed in um."""
        z = self._f(text, None)
        if z is None or not self.stage_connected:
            return
        self._bg(self.hw.move_z, float(z))

    # ------------------------------------------------------ output directory

    def _out_base(self) -> str:
        base = (self.output_dir or '').strip() or CAPTURES_DIR
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            base = CAPTURES_DIR
            os.makedirs(base, exist_ok=True)
        return base

    def set_output_dir(self, text: str):
        self.output_dir = text.strip() or CAPTURES_DIR
        self._persist()
        self._set_status(f'Protocol data directory: {self.output_dir}')

    def open_output_dir(self):
        try:
            os.startfile(self._out_base())
        except Exception as e:
            self._set_status(f'Open folder failed: {e}')

    # ---------------------------------------------- protocol positions

    def _safe_goto(self, x_um: float, y_um: float, z_um: float | None):
        """Move to a position safely: drop Z by the travel clearance first,
        make the XY move, then go to the target focus Z."""
        clear = max(self.travel_clearance_um, 0.0)
        cur_z = self.hw.get_pos()[2]
        travel_z = max(cur_z - clear, 0.0)
        if clear > 0:
            self.hw.move_z(travel_z)
        self.hw.move_xy(x_um, y_um)
        if z_um is not None:
            self.hw.move_z(max(z_um, 0.0))

    def _pos_well_name(self, x_um: float, y_um: float) -> str | None:
        rel = self.stage_to_plate_rel(x_um, y_um)
        if rel is None:
            return None
        lw = self.current_labware
        sx = lw.spacing_x_mm or 1.0
        sy = lw.spacing_y_mm or 1.0
        col = round(rel[0] / sx)
        row = round(rel[1] / sy)
        if 0 <= row < lw.rows and 0 <= col < lw.cols:
            wx, wy = col * sx, row * sy
            if abs(rel[0] - wx) < sx * 0.5 and abs(rel[1] - wy) < sy * 0.5:
                return lw.get_well_name(row, col)
        return None

    def _refresh_positions(self, highlight: int | None = None):
        n = len(self.protocol_positions)
        if n == 0:
            self.positions_info = 'No positions defined'
        else:
            names = ', '.join(p['name'] for p in self.protocol_positions[:8])
            if n > 8:
                names += ', ...'
            self.positions_info = f'{n} positions: {names}'
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is not None:
            lw = self.current_labware
            marks = []
            for p in self.protocol_positions:
                rel = self.stage_to_plate_rel(p['x_um'], p['y_um'])
                if rel is not None:
                    marks.append((lw.a1_offset_x_mm + rel[0],
                                  lw.a1_offset_y_mm + rel[1]))
            plate.set_position_markers(marks, highlight)

    def position_add_here(self):
        """Add the current stage position (and focus) as a protocol
        interaction point."""
        drift = self.surface.drift_um
        well = self._pos_well_name(self.stage_x_um, self.stage_y_um)
        name = well or f'P{len(self.protocol_positions) + 1:02d}'
        self.protocol_positions.append({
            'name': name,
            'x_um': self.stage_x_um,
            'y_um': self.stage_y_um,
            'z_um': self.stage_z_um - drift,
        })
        self._persist()
        self._refresh_positions()
        self._set_status(f'Position {name} added '
                         f'({len(self.protocol_positions)} total)')

    def position_remove_last(self):
        if self.protocol_positions:
            gone = self.protocol_positions.pop()
            self._persist()
            self._refresh_positions()
            self._set_status(f'Removed position {gone["name"]}')

    def position_clear(self):
        self.protocol_positions = []
        self._persist()
        self._refresh_positions()
        self._set_status('Positions cleared')

    # --------------------------------------- position review walkthrough

    def _pos_current_z(self, p: dict) -> float:
        return p['z_um'] + self.surface.drift_um

    def pos_review_start(self):
        """Walk through every stored position: the stage moves there safely,
        you fine-tune focus, press 'Set focus = here', then Next."""
        if not self.protocol_positions:
            self._set_status('No positions to review - add some first')
            return
        if not self.stage_connected:
            self._set_status('Position review needs the stage')
            return
        self._pos_review_index = 0
        self.pos_review_active = True
        self._pos_review_goto()

    def _pos_review_goto(self):
        i = self._pos_review_index
        p = self.protocol_positions[i]
        n = len(self.protocol_positions)
        self.pos_review_label = f'[{i + 1}/{n}] {p["name"]}'
        self._refresh_positions(highlight=i)
        z = self._pos_current_z(p)
        self._set_status(f'Moving to {p["name"]}...')
        self._bg(self._safe_goto, p['x_um'], p['y_um'], z)

    def pos_review_next(self):
        if not self.pos_review_active:
            return
        if self._pos_review_index < len(self.protocol_positions) - 1:
            self._pos_review_index += 1
            self._pos_review_goto()
        else:
            self._set_status('Last position - press Done when finished')

    def pos_review_prev(self):
        if not self.pos_review_active:
            return
        if self._pos_review_index > 0:
            self._pos_review_index -= 1
            self._pos_review_goto()

    def pos_review_set_focus(self):
        """Store the current focus (and XY trim) into the shown position."""
        if not self.pos_review_active:
            return
        p = self.protocol_positions[self._pos_review_index]
        p['x_um'] = self.stage_x_um
        p['y_um'] = self.stage_y_um
        p['z_um'] = self.stage_z_um - self.surface.drift_um
        self._persist()
        self._set_status(f'{p["name"]}: focus updated '
                         f'(z={self.stage_z_um:.1f} um)')

    def pos_review_done(self):
        self.pos_review_active = False
        self.pos_review_label = ''
        self._refresh_positions(highlight=None)
        self._set_status('Position review finished')

    # ------------------------------------------------ protocol save/load

    def _refresh_protocol_files(self):
        try:
            os.makedirs(self.protocol_dir, exist_ok=True)
            names = [f[:-5] for f in os.listdir(self.protocol_dir)
                     if f.endswith('.json')]
        except Exception:
            names = []
        self.protocol_files = names or ['(none saved)']

    def set_protocol_dir(self, text: str):
        self.protocol_dir = text.strip() or PROTOCOLS_DIR
        self._refresh_protocol_files()
        self._persist()
        self._set_status(f'Protocol autosave directory: {self.protocol_dir}')

    def _protocol_payload(self, name: str) -> dict:
        ids = self.root.ids if self.root else {}
        return {
            'format': 'lvp-protocol-1',
            'name': name,
            'labware_name': self.labware_name,
            'labware_def': (None if self.current_labware.builtin else
                            labware_defs.labware_to_dict(
                                self.current_labware)),
            'channel_settings': acq.channel_settings_to_dict(self.channel_settings),
            'af_config': {
                'z_bottom_um': self.af_config.z_bottom_um,
                'z_top_um': self.af_config.z_top_um,
                'step_um': self.af_config.step_um,
                'channel': self.af_config.channel,
            },
            'surface': self.surface.to_dict(),
            'positions': self.protocol_positions,
            'cycles': ids.protocol_cycles_input.text if ids else '1',
            'interval_min': ids.protocol_interval_input.text if ids else '5',
            'af_each_cycle': self.af_each_cycle_on,
            'af_channel': self.af_channel,
            'af_method': self.af_method,
            'use_surface': self.use_surface_on,
            'use_z_offsets': self.use_z_offsets_on,
            'travel_clearance_um': self.travel_clearance_um,
            'output_dir': self.output_dir,
        }

    def _write_protocol(self, name: str) -> bool:
        import json
        try:
            os.makedirs(self.protocol_dir, exist_ok=True)
            with open(os.path.join(self.protocol_dir, f'{name}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(self._protocol_payload(name), f, indent=2)
            return True
        except Exception:
            return False

    def _autosave_protocol(self):
        """Continuous snapshot of the current protocol state."""
        try:
            self._write_protocol('_autosave')
        except Exception:
            pass

    def save_protocol(self, name: str):
        """Save the full protocol definition: labware, channel settings,
        AF configuration, focus surface, positions and timelapse settings.
        Plate calibration is intentionally NOT included - redo that (plus
        the position focus review) when the plate is remounted."""
        name = (name or '').strip() or 'protocol1'
        self.apply_channel_fields_quiet()
        self.af_apply_fields()
        if self._write_protocol(name):
            self.protocol_name = name
            self._refresh_protocol_files()
            self._set_status(f'Protocol "{name}" saved to {self.protocol_dir}')
        else:
            self._set_status('Protocol save FAILED (directory writable?)')

    def load_protocol(self, name: str):
        """Load a saved protocol. Afterwards only the plate calibration and
        the position focus review are needed before running."""
        name = (name or '').strip()
        if not name or name == '(none saved)':
            return
        try:
            import json
            with open(os.path.join(self.protocol_dir, f'{name}.json'), 'r',
                      encoding='utf-8') as f:
                d = json.load(f)
        except Exception as e:
            self._set_status(f'Protocol load failed: {e}')
            return
        try:
            self.channel_settings = acq.channel_settings_from_dict(
                d.get('channel_settings', {}))
            af = d.get('af_config', {})
            self.af_config = AutofocusConfig(
                z_bottom_um=float(af.get('z_bottom_um', 0.0)),
                z_top_um=float(af.get('z_top_um', 200.0)),
                step_um=float(af.get('step_um', 10.0)),
                channel=af.get('channel', 'BF'))
            if d.get('surface'):
                self.surface.from_dict(d['surface'])
            self.protocol_positions = [dict(p) for p in d.get('positions', [])]
            self.af_each_cycle_on = bool(d.get('af_each_cycle', True))
            self.af_channel = d.get('af_channel', 'BF')
            if d.get('af_method') in acq.AF_METHODS:
                self.af_method = d['af_method']
            self.surface.anchor_channel = self.af_channel
            self.use_surface_on = bool(d.get('use_surface', True))
            self.use_z_offsets_on = bool(d.get('use_z_offsets', True))
            self.travel_clearance_um = float(d.get('travel_clearance_um', 500.0))
            if d.get('output_dir'):
                self.output_dir = d['output_dir']
            lw_name = d.get('labware_name')
            if lw_name and self.labware_lib.get(lw_name) is None and \
                    d.get('labware_def'):
                try:
                    lw_def = labware_defs.labware_from_dict(d['labware_def'])
                    if not labware_defs.validate_labware(lw_def)[0]:
                        self.labware_lib.save(lw_def)
                        self.labware_options = self.labware_lib.names()
                except Exception:
                    pass
            if lw_name and self.labware_lib.get(lw_name) is not None:
                self.select_labware(lw_name)
            ids = self.root.ids
            ids.protocol_cycles_input.text = str(d.get('cycles', '1'))
            ids.protocol_interval_input.text = str(d.get('interval_min', '5'))
            self.protocol_name = name
            self._sync_channel_fields()
            self._sync_af_fields()
            self._refresh_surface_info()
            self._refresh_positions()
            self._persist()
            self._set_status(f'Protocol "{name}" loaded - calibrate the '
                             f'plate, then run "Review positions"')
        except Exception as e:
            self._set_status(f'Protocol load failed: {e}')

    def apply_channel_fields_quiet(self):
        try:
            self.apply_channel_fields()
        except Exception:
            pass

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
        name = self.current_labware.get_well_name(row, col)
        self.selected_well_str = name

    def on_well_double_click(self, row: int, col: int, rel_x_mm: float, rel_y_mm: float):
        """Double-click on the plate navigator: move stage to that well."""
        lw = self.current_labware
        name = lw.get_well_name(row, col)
        self.selected_well_str = name
        if not self.stage_connected:
            self._set_status(f'{name}: stage not connected')
            return
        target = self.plate_rel_to_stage(rel_x_mm, rel_y_mm)
        if target is None:
            self._set_status('Calibrate the plate first (or "Set A1 = here")')
            return
        tx, ty = target
        self._set_status(f'Moving to {name}...')
        def _move():
            self.hw.move_xy(tx, ty)
            if self.use_surface_on:
                z = self.surface.z_at(tx, ty)
                if z is not None:
                    self.hw.move_z(z)
            self._set_status(f'At {name}')
        self._bg(_move)

    def _img_cal_matrix(self):
        """Current px-shift-per-stage-um 2x2 (binning compensated), or None."""
        if not self.img_cal or not self.img_cal.get('A'):
            return None
        A = np.asarray(self.img_cal['A'], dtype=np.float64)
        cal_bin = max(int(self.img_cal.get('binning', 1)), 1)
        try:
            cur_bin = max(int(self.binning), 1)
        except Exception:
            cur_bin = 1
        return A * (cal_bin / cur_bin)

    def _image_cal_sync(self, ch: str) -> bool:
        """Blocking pixel->stage calibration (two 100 um probe moves)."""
        D = 100.0
        self.hw.set_channel(ch, self._cs(ch))
        time.sleep(0.3)
        f0 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(D, 0)
        time.sleep(0.1)
        f1 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(0, D)
        time.sleep(0.1)
        f2 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(-D, -D)   # return to start
        sx = acq.measure_image_shift(f0, f1)
        sy = acq.measure_image_shift(f1, f2)
        if (sx is None or sy is None or sx[2] < 0.1 or sy[2] < 0.1 or
                (abs(sx[0]) + abs(sx[1])) < 2 or (abs(sy[0]) + abs(sy[1])) < 2):
            return False
        A = [[sx[0] / D, sy[0] / D], [sx[1] / D, sy[1] / D]]
        try:
            cur_bin = max(int(self.binning), 1)
        except Exception:
            cur_bin = 1
        self.img_cal = {'A': A, 'binning': cur_bin}
        self._persist()
        return True

    def calibrate_image_motion(self):
        """Solve the exact pixel->stage mapping (scale, direction, rotation)
        by moving the stage ~100 um in X then Y and measuring the image
        shift with phase correlation. Needs a lit, feature-rich field."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Image calibration needs stage + camera')
            return
        ch = self.live_channel if self.live_channel in CHANNELS else self.selected_channel
        def _job():
            if self._image_cal_sync(ch):
                A = np.asarray(self.img_cal['A'])
                scale = 1.0 / float(np.linalg.norm(A[:, 0]))
                self._set_status(f'Image motion calibrated: {scale:.3f} um/px')
            else:
                self._set_status('Image calibration FAILED - need a lit field '
                                 'with visible features (try focusing first)')
        self._bg(_job, busy_job=True)

    def on_image_double_click(self, dx_px: float, dy_px: float):
        """Double-click on the live image: bring the clicked point to the
        center. Auto-calibrates the pixel->stage mapping on first use, then
        closes the loop: measures the actual image shift and applies up to
        three corrective trims until the click lands centered."""
        if not self.stage_connected or self.worker_busy:
            return
        ch = self.live_channel if self.live_channel in CHANNELS else self.selected_channel
        target_px = np.array([-dx_px, -dy_px])   # required content shift

        def _job():
            if self._img_cal_matrix() is None:
                self._set_status('First double-click: calibrating image '
                                 'motion (0.1 mm probe moves)...')
                if not self._image_cal_sync(ch):
                    self._set_status('Centering calibration FAILED - light a '
                                     'featured field and try again')
                    return
            A = self._img_cal_matrix()
            try:
                stage = np.linalg.solve(A, target_px)
            except np.linalg.LinAlgError:
                self._set_status('Centering: calibration degenerate - recalibrate')
                return
            self._set_status(f'Centering: dx={stage[0]:.0f}, dy={stage[1]:.0f} um')
            remaining = target_px.copy()
            move = stage
            for i in range(3):
                f0 = self.hw.grab_fresh(1.0)
                self.hw.move_xy_rel(float(move[0]), float(move[1]))
                time.sleep(0.12)
                f1 = self.hw.grab_fresh(1.0)
                s = acq.measure_image_shift(f0, f1)
                if s is None or s[2] < 0.1:
                    break   # can't verify (dark/featureless) - open loop done
                remaining = remaining - np.array([s[0], s[1]])
                if np.linalg.norm(remaining) <= 4:
                    break
                try:
                    move = np.linalg.solve(A, remaining)
                except np.linalg.LinAlgError:
                    break
                if np.linalg.norm(move) > np.linalg.norm(stage) * 1.5:
                    break   # diverging - stop rather than wander
            self._set_status('Centered')
        self._bg(_job)

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

    def cycle_af_method(self):
        """Step to the next autofocus algorithm (two-pass / global fit)."""
        methods = acq.AF_METHODS
        i = methods.index(self.af_method) if self.af_method in methods else -1
        self.af_method = methods[(i + 1) % len(methods)]
        self._persist()

    @staticmethod
    def af_method_text(method: str) -> str:
        return 'AF method: ' + acq.AF_METHOD_LABELS.get(method, method)

    @staticmethod
    def af_range_text(bottom_txt: str, top_txt: str) -> str:
        try:
            rng = abs(float(top_txt) - float(bottom_txt))
            return f'Range thickness: {rng:.1f} um'
        except (TypeError, ValueError):
            return 'Range thickness: ?'

    def _save_af_template(self):
        try:
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'af_template.npy')
            if self.af_template is not None:
                np.save(p, self.af_template)
        except Exception:
            pass

    def af_set_anchor_here(self):
        """The autofocus point used to re-reference every cycle (uses the
        designated AF channel). Also captures an in-focus reference image
        of this view for high-stability AF - you marked it in focus for a
        reason, so AF can return to exactly this appearance."""
        self.af_config.channel = self.af_channel
        self.surface.set_anchor(self.stage_x_um, self.stage_y_um,
                                self.af_channel)
        self._persist()
        self._refresh_surface_info()
        ch = self.af_channel
        def _capture():
            self.hw.set_channel(ch, self._cs(ch))
            time.sleep(0.3)
            frame = self.hw.grab_fresh(2.0)
            tpl = acq.make_focus_template(frame)
            if tpl is not None:
                self.af_template = tpl
                self._save_af_template()
                Clock.schedule_once(
                    lambda dt: setattr(self, 'live_channel', ch), 0)
                self._set_status(f'AF anchor + focus reference captured '
                                 f'({ch})')
            else:
                self._set_status('AF anchor set (no reference frame)')
        self._bg(_capture)

    def af_goto_anchor(self):
        if self.surface.anchor_xy is None:
            self._set_status('No AF anchor set')
            return
        x, y = self.surface.anchor_xy
        self._bg(self.hw.move_xy, x, y)

    def run_af_test(self):
        """Test autofocus on the CURRENT field of view with the selected
        channel and the configured bottom/top/interval. Does not touch the
        focus surface."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('AF test needs stage + camera')
            return
        self.af_apply_fields()
        ch = self.af_channel
        cfg = AutofocusConfig(z_bottom_um=self.af_config.z_bottom_um,
                              z_top_um=self.af_config.z_top_um,
                              step_um=self.af_config.step_um,
                              channel=ch)
        def _job():
            t0 = time.time()
            def _pos(z):
                Clock.schedule_once(lambda dt: setattr(self, 'stage_z_um', z), 0)
            best, ok = acq.autofocus_sweep(self.hw, cfg, self._cs(ch),
                                           cancel=self.worker.cancel,
                                           status_cb=self._set_status,
                                           pos_cb=_pos,
                                           template=self.af_template,
                                           high_stability=self.high_stability_on,
                                           method=self.af_method)
            # Keep the AF channel lit so the image stream stays visible.
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', ch), 0)
            dt_s = time.time() - t0
            if best is None:
                self._set_status('AF test canceled')
            elif ok:
                self._set_status(f'AF TEST: best focus z={best:.2f} um '
                                 f'({ch}, {dt_s:.1f} s)')
            else:
                self._set_status(f'AF TEST: LOW CONTRAST - parked mid-range '
                                 f'z={best:.1f} um ({dt_s:.1f} s)')
        self._bg(_job, busy_job=True)

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
            def _pos(z):
                Clock.schedule_once(lambda dt: setattr(self, 'stage_z_um', z), 0)
            best, ok = acq.autofocus_sweep(self.hw, cfg, self._cs(ch),
                                           cancel=self.worker.cancel,
                                           status_cb=self._set_status,
                                           pos_cb=_pos,
                                           template=self.af_template,
                                           high_stability=self.high_stability_on,
                                           method=self.af_method)
            # Keep the AF channel lit so the image stream stays visible.
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', ch), 0)
            if best is None:
                self._set_status('Autofocus canceled')
                return
            if not ok:
                self._set_status(f'AF: LOW CONTRAST - parked mid-range at '
                                 f'z={best:.1f} um (surface NOT updated)')
                return
            if self.surface.ref_z is None:
                self.surface.set_reference_z(best)
            else:
                self.surface.update_cycle(best)
            self._persist()
            Clock.schedule_once(lambda dt: self._refresh_surface_info(), 0)
            self._set_status(f'AF done: best focus z={best:.2f} um '
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
            Clock.schedule_once(lambda dt: self._refresh_shading_info(), 0)
            self._set_status(f'Shading reference for {ch} '
                             f'{"captured" if ok else "FAILED"}')
        self._bg(_job, busy_job=True)

    def _shading_planes(self, channel: str | None = None):
        """The acquisition z-stack planes for ``channel`` around the marked
        focus Z (exactly what run_z_stack will visit), or None."""
        if self.shading_focus_z is None:
            return None
        ch = channel or self.selected_channel
        cs = self._cs(ch)
        if self.root and ch == self.selected_channel:
            ids = self.root.ids   # use what is typed, even if not applied
            cs.zstack.range_um = self._f(ids.zstack_range_input.text,
                                         cs.zstack.range_um)
            cs.zstack.step_um = self._f(ids.zstack_step_input.text,
                                        cs.zstack.step_um)
            cs.z_offset_um = self._f(ids.zoffset_input.text, cs.z_offset_um)
        return acq.zstack_planes(self.shading_focus_z, cs,
                                 self.use_z_offsets_on, force_stack=True)

    def shading_focus_here(self):
        """Step 1 (labware IN, sample in focus): mark the focus Z the
        acquisition z-stack will be centered on."""
        self.shading_focus_z = float(self.stage_z_um)
        self._persist()
        self._refresh_shading_info()
        self._set_status(f'Shading: focus Z = {self.shading_focus_z:.1f} um. '
                         f'Now remove the labware and capture.')

    def capture_shading_stack(self):
        """Step 2 (labware REMOVED): capture a flat-field reference at every
        plane of the selected channel's acquisition z-stack (z-stack range /
        step and colour Z offset around the marked focus Z) - the same
        planes the run will acquire. Confirms first."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Shading stack needs stage + camera')
            return
        zs = self._shading_planes()
        if zs is None:
            self._set_status('Shading: focus on the sample (labware in) and '
                             'press "Focus Z = here" first')
            return
        ch = self.selected_channel
        from kivy.uix.button import Button
        box = BoxLayout(orientation='vertical', spacing=6, padding=6)
        planes = ', '.join(f'{z:.1f}' for z in zs[:12]) + \
            (' ...' if len(zs) > 12 else '')
        msg = Label(text=(f'Remove the labware so the field is empty.\n\n'
                          f'{ch}: {len(zs)} planes, {zs[0]:.1f} to '
                          f'{zs[-1]:.1f} um\n({planes})\n\n'
                          f'These are the planes the {ch} z-stack acquires '
                          f'around focus Z {self.shading_focus_z:.1f} um.'),
                    halign='center', valign='middle', font_size='12sp')
        msg.bind(size=lambda w, s: setattr(w, 'text_size', s))
        box.add_widget(msg)
        btns = BoxLayout(size_hint_y=None, height='36dp', spacing=6)
        go, no = Button(text='Labware removed - capture'), Button(text='Cancel')
        btns.add_widget(go)
        btns.add_widget(no)
        box.add_widget(btns)
        popup = Popup(title='Capture shading Z-stack', content=box,
                      size_hint=(None, None), size=('440dp', '300dp'))
        focus_z = self.shading_focus_z

        def _job():
            self.hw.set_channel(ch, self._cs(ch))
            time.sleep(0.3)
            ok = self.shading.capture_reference_stack(
                self.hw, ch, zs, cancel=self.worker.cancel,
                status_cb=self._set_status, return_z=focus_z)
            Clock.schedule_once(lambda dt: self._refresh_shading_info(), 0)
            if not ok:
                self._set_status(f'Shading Z-stack for {ch} FAILED/canceled '
                                 f'(previous reference kept)')
            elif self.shading.last_structure > self.shading.STRUCTURE_WARN:
                self._set_status(
                    f'Shading Z-stack for {ch} captured, BUT the field does '
                    f'not look empty (detail {self.shading.last_structure:.1%})'
                    f' - is the labware still in? Recapture if so.')
            else:
                self._set_status(f'Shading Z-stack for {ch} captured: '
                                 f'{len(zs)} planes {zs[0]:.1f}-{zs[-1]:.1f} um')

        def _go(*_):
            popup.dismiss()
            self._bg(_job, busy_job=True)

        go.bind(on_press=_go)
        no.bind(on_press=popup.dismiss)
        popup.open()

    def _sync_shading_fields(self):
        self._refresh_shading_info()

    def _refresh_shading_info(self):
        ch = self.selected_channel
        info = 'Stored: ' + self.shading.describe(ch)
        zs = None
        try:
            zs = self._shading_planes(ch)
        except Exception:
            pass
        if zs is None:
            plan = 'Focus Z not marked'
        else:
            cs = self._cs(ch)
            plan = (f'Will capture {len(zs)} planes {zs[0]:.1f}-{zs[-1]:.1f} um'
                    f' (focus {self.shading_focus_z:.1f}'
                    f'{", z-stack OFF for " + ch if not cs.zstack.enabled else ""})')
        self.shading_info = f'{plan}\n{info}'

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

    def toggle_tile_scan(self, pattern: str = 'grid'):
        """Click-on/click-off: starts the tile scan, or cancels the running
        acquisition when pressed again."""
        if self.worker_busy:
            self.cancel_job()
            return
        self.run_tile_scan(pattern)

    def run_tile_scan(self, pattern: str = 'grid'):
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Tile scan needs stage + camera')
            return
        rows, cols, overlap = self._tile_params()
        fov_x, fov_y = self._fov_um()
        cx, cy = self.stage_x_um, self.stage_y_um
        stitch = True
        total = None
        if pattern == 'spiral':
            # Endless spiral: runs until STOP or the stage's physical edges.
            positions = acq.spiral_positions_unbounded(cx, cy, fov_x, fov_y,
                                                       overlap_pct=overlap)
            stitch = False   # unbounded: tiles saved per-position instead
        else:
            positions = acq.tile_positions_grid(cx, cy, fov_x, fov_y,
                                                rows, cols, overlap_pct=overlap)
            total = len(positions)
        out_dir = acq.timestamp_dir(self._out_base(), f'tiles_{pattern}')
        use_surface = self.use_surface_on and self.surface.current_z is not None
        self._mosaic_reset()
        def _job():
            acq.run_tile_scan(self.hw, positions, self.channel_settings,
                              self.surface if use_surface else None,
                              self.shading if self.shading_on else None,
                              self.effective_um_per_px(), out_dir,
                              self.worker.cancel, status_cb=self._set_status,
                              use_channel_z_offsets=self.use_z_offsets_on,
                              total=total, stitch=stitch,
                              on_tile=self._mosaic_add)
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    # ---------------------------------------------- live mosaic preview

    def _mosaic_reset(self):
        self._mosaic = None
        self._mosaic_meta = None

    def _mosaic_add(self, ch: str, frame, x_um: float, y_um: float):
        """Overlay a captured tile into the on-the-fly mosaic preview
        (shown in the viewport while the scan runs). Worker-thread safe:
        the canvas reference is swapped atomically."""
        try:
            h, w = frame.shape[:2]
            f = max(w // 160, 1)
            small = frame[::f, ::f]
            if small.dtype != np.uint8:
                shift = max(self.significant_bits - 8, 0)
                small = (small >> shift).astype(np.uint8)
            tint = CHANNEL_LUTS.get(ch, (1.0, 1.0, 1.0))
            chans = []
            for t in tint[:3]:
                if t >= 0.999:
                    chans.append(small)
                elif t <= 0.001:
                    chans.append(np.zeros_like(small))
                else:
                    chans.append((small.astype(np.float32) * t).astype(np.uint8))
            rgb = np.stack(chans, axis=-1)
            th, tw = rgb.shape[:2]
            pv_um_per_px = self.effective_um_per_px() * f
            meta = self._mosaic_meta
            if meta is None:
                size = 1024
                canvas = np.zeros((size, size, 3), np.uint8)
                meta = {'ox_um': x_um, 'oy_um': y_um, 'scale': pv_um_per_px,
                        'cx': size // 2, 'cy': size // 2}
                self._mosaic_meta = meta
                self._mosaic = canvas
            canvas = self._mosaic
            px = int((x_um - meta['ox_um']) / meta['scale']) + meta['cx'] - tw // 2
            py = int((y_um - meta['oy_um']) / meta['scale']) + meta['cy'] - th // 2
            # Expand the canvas (up to 4096 px) when a tile falls outside.
            while (px < 0 or py < 0 or px + tw > canvas.shape[1] or
                   py + th > canvas.shape[0]) and canvas.shape[0] < 4096:
                old = canvas
                oh, ow = old.shape[:2]
                canvas = np.zeros((oh * 2, ow * 2, 3), np.uint8)
                canvas[oh // 2: oh // 2 + oh, ow // 2: ow // 2 + ow] = old
                meta['cx'] += ow // 2
                meta['cy'] += oh // 2
                px = int((x_um - meta['ox_um']) / meta['scale']) + meta['cx'] - tw // 2
                py = int((y_um - meta['oy_um']) / meta['scale']) + meta['cy'] - th // 2
            if (0 <= px and 0 <= py and px + tw <= canvas.shape[1] and
                    py + th <= canvas.shape[0]):
                canvas[py:py + th, px:px + tw] = rgb   # simple overlay
            self._mosaic = canvas
        except Exception:
            pass

    # ----------------------------------------------------- saved layouts

    def _refresh_layout_files(self):
        try:
            os.makedirs(LAYOUTS_DIR, exist_ok=True)
            names = [f[:-5] for f in os.listdir(LAYOUTS_DIR)
                     if f.endswith('.json')]
        except Exception:
            names = []
        self.layout_files = names or ['(none)']

    def _collapsibles(self):
        import plate_navigator as pn
        return [w for w in self.root.walk()
                if isinstance(w, pn.Collapsible)] if self.root else []

    def save_layout(self, name: str):
        """Save panel widths and which sections are open as a named layout."""
        name = (name or '').strip() or 'default'
        ids = self.root.ids
        data = {
            'left_w': ids.left_split.width if ids.get('left_split') else 250,
            'right_w': ids.right_split.width if ids.get('right_split') else 300,
            'sections': {c.title: c.open for c in self._collapsibles()
                         if c.title},
        }
        try:
            import json
            os.makedirs(LAYOUTS_DIR, exist_ok=True)
            with open(os.path.join(LAYOUTS_DIR, f'{name}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            self._refresh_layout_files()
            self._set_status(f'Layout "{name}" saved')
        except Exception as e:
            self._set_status(f'Layout save failed: {e}')

    def load_layout(self, name: str):
        name = (name or '').strip()
        if not name or name == '(none)':
            return
        try:
            import json
            with open(os.path.join(LAYOUTS_DIR, f'{name}.json'), 'r',
                      encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self._set_status(f'Layout load failed: {e}')
            return
        ids = self.root.ids
        if ids.get('left_split'):
            ids.left_split.width = float(data.get('left_w', 250))
        if ids.get('right_split'):
            ids.right_split.width = float(data.get('right_w', 300))
        sections = data.get('sections', {})
        for c in self._collapsibles():
            if c.title and c.title in sections:
                c.open = bool(sections[c.title])
        self._set_status(f'Layout "{name}" loaded')

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
        center_z = None
        if self.use_surface_on:
            center_z = self.surface.z_at(self.stage_x_um, self.stage_y_um)
        if center_z is None:
            center_z = self.stage_z_um
        out_dir = acq.timestamp_dir(self._out_base(), 'zstack')
        def _job():
            acq.run_z_stack(self.hw, self.channel_settings, channels,
                            center_z, self.shading if self.shading_on else None,
                            out_dir, self.worker.cancel,
                            status_cb=self._set_status,
                            use_channel_z_offsets=self.use_z_offsets_on)
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    # ------------------------------------------------- timelapse protocol

    def run_protocol(self):
        """Timelapse protocol over all stored positions.

        Every cycle: optionally re-reference focus at the AF anchor (the
        measured drift shifts every position's stored focus), then visit
        each interaction position with safe Z-travel moves and acquire all
        protocol-marked channels (plus their z-stacks). With no positions
        defined, the current location is used. Focus per position: the
        focus surface (if enabled and referenced with points), else the
        position's reviewed focus plus drift."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Protocol needs stage + camera')
            return
        ids = self.root.ids
        cycles = max(int(self._f(ids.protocol_cycles_input.text, 1)), 1)
        interval_s = max(self._f(ids.protocol_interval_input.text, 5.0), 0.0) * 60.0
        channels = [ch for ch in CHANNELS if self.channel_settings[ch].enabled]
        if not channels:
            self._set_status('No channels marked "Use in protocol"')
            return
        do_af = (self.af_each_cycle_on and
                 self.surface.anchor_xy is not None)
        self.af_apply_fields()
        positions = [dict(p) for p in self.protocol_positions]
        if not positions:
            positions = [{'name': 'POS', 'x_um': self.stage_x_um,
                          'y_um': self.stage_y_um,
                          'z_um': self.stage_z_um - self.surface.drift_um}]
        base_dir = acq.timestamp_dir(self._out_base(), 'protocol')
        cancel = self.worker.cancel

        def _job():
            for cyc in range(cycles):
                t0 = time.time()
                if cancel.is_set():
                    break
                cyc_dir = os.path.join(base_dir, f'cycle_{cyc:04d}')
                os.makedirs(cyc_dir, exist_ok=True)
                self._set_status(f'Protocol cycle {cyc + 1}/{cycles}...')

                if do_af:
                    ax, ay = self.surface.anchor_xy
                    self._safe_goto(ax, ay, None)
                    cfg = AutofocusConfig(
                        z_bottom_um=self.af_config.z_bottom_um,
                        z_top_um=self.af_config.z_top_um,
                        step_um=self.af_config.step_um,
                        channel=self.surface.anchor_channel)
                    best, af_ok = acq.autofocus_sweep(
                        self.hw, cfg, self._cs(self.surface.anchor_channel),
                        cancel=cancel, status_cb=self._set_status,
                        template=self.af_template,
                        high_stability=self.high_stability_on,
                        method=self.af_method)
                    if best is not None and af_ok:
                        if self.surface.ref_z is None:
                            self.surface.set_reference_z(best)
                        else:
                            self.surface.update_cycle(best)
                        Clock.schedule_once(
                            lambda dt: self._refresh_surface_info(), 0)
                    if cancel.is_set():
                        break

                drift = self.surface.drift_um
                for p in positions:
                    if cancel.is_set():
                        break
                    base_z = None
                    if self.use_surface_on and self.surface.points and \
                            self.surface.current_z is not None:
                        base_z = self.surface.z_at(p['x_um'], p['y_um'])
                    if base_z is None:
                        base_z = p['z_um'] + drift
                    self._set_status(f'Cycle {cyc + 1}: {p["name"]}...')
                    self._safe_goto(p['x_um'], p['y_um'], base_z)
                    for ch in channels:
                        if cancel.is_set():
                            break
                        cs = self.channel_settings[ch]
                        z_off = cs.z_offset_um if self.use_z_offsets_on else 0.0
                        if abs(z_off) > 0.01:
                            self.hw.move_z(base_z + z_off)
                        self.hw.set_channel(ch, cs)
                        frame = self.hw.grab_settled(settle_s=0.2)
                        if frame is not None:
                            if self.shading_on:
                                frame = self.shading.apply(
                                    frame, ch, base_z + z_off)
                            acq.save_image(
                                os.path.join(cyc_dir,
                                             f'{p["name"]}_{ch}.tiff'), frame)
                        if abs(z_off) > 0.01:
                            self.hw.move_z(base_z)
                        if cs.zstack.enabled and not cancel.is_set():
                            acq.run_z_stack(
                                self.hw, self.channel_settings, [ch], base_z,
                                self.shading if self.shading_on else None,
                                cyc_dir, cancel, status_cb=self._set_status,
                                use_channel_z_offsets=self.use_z_offsets_on)
                    self.hw.leds_off()
                self._persist()

                if cyc < cycles - 1:
                    while time.time() - t0 < interval_s:
                        if cancel.is_set():
                            break
                        remaining = interval_s - (time.time() - t0)
                        self._set_status(
                            f'Protocol: cycle {cyc + 1}/{cycles} done, '
                            f'next in {remaining / 60:.1f} min')
                        time.sleep(min(2.0, max(remaining, 0.1)))
                    if cancel.is_set():
                        break
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
            self._set_status(f'Protocol finished -> {base_dir}')
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
            'labware_name': self.labware_name,
            'plate_cal': self.plate_cal.to_dict() if self.plate_cal else None,
            'use_surface_on': self.use_surface_on,
            'use_z_offsets_on': self.use_z_offsets_on,
            'af_each_cycle_on': self.af_each_cycle_on,
            'output_dir': self.output_dir,
            'protocol_positions': self.protocol_positions,
            'travel_clearance_um': self.travel_clearance_um,
            'img_cal': self.img_cal,
            'af_channel': self.af_channel,
            'high_stability_on': self.high_stability_on,
            'af_method': self.af_method,
            'autoscale_on': self.autoscale_on,
            'clip_overlay_on': self.clip_overlay_on,
            'shading_focus_z': self.shading_focus_z,
            'protocol_dir': self.protocol_dir,
            'protocol_optin_v2': True,
        }
        try:
            acq.save_settings(payload)
        except Exception:
            pass
        self._autosave_protocol()

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
            lw_name = d.get('labware_name')
            if lw_name and self.labware_lib.get(lw_name) is not None:
                self.current_labware = self.labware_lib.get(lw_name)
            cal = d.get('plate_cal')
            if cal and cal.get('matrix') and \
                    cal.get('labware_name') == self.current_labware.name:
                pc = PlateCalibration(self.current_labware)
                pc.from_dict(cal)
                self.plate_cal = pc
            self.use_surface_on = bool(d.get('use_surface_on', True))
            self.use_z_offsets_on = bool(d.get('use_z_offsets_on', True))
            self.af_each_cycle_on = bool(d.get('af_each_cycle_on', True))
            self.output_dir = d.get('output_dir') or CAPTURES_DIR
            self.protocol_positions = [dict(p) for p
                                       in d.get('protocol_positions', [])]
            self.travel_clearance_um = float(d.get('travel_clearance_um', 500.0))
            self.img_cal = d.get('img_cal')
            self.af_channel = d.get('af_channel', 'BF')
            self.high_stability_on = bool(d.get('high_stability_on', False))
            if d.get('af_method') in acq.AF_METHODS:
                self.af_method = d['af_method']
            self.autoscale_on = bool(d.get('autoscale_on', self.autoscale_on))
            self.clip_overlay_on = bool(d.get('clip_overlay_on', True))
            fz = d.get('shading_focus_z')
            self.shading_focus_z = float(fz) if fz is not None else None
            self.protocol_dir = d.get('protocol_dir') or PROTOCOLS_DIR
            # One-time migration: protocol membership became opt-in; clear
            # any enables carried over from older settings files.
            if not d.get('protocol_optin_v2'):
                for cs in self.channel_settings.values():
                    cs.enabled = False
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
