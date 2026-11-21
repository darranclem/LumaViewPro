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


def _relaunch_in_project_venv():
    """Re-run this script with the repo's .venv Python if it exists and we
    were started with some other interpreter (e.g. a bare `python` that
    has none of the dependencies installed)."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv = os.path.join(here, '.venv')
    exe = os.path.join(venv, 'Scripts' if os.name == 'nt' else 'bin',
                       'python.exe' if os.name == 'nt' else 'python')
    if (not os.path.isfile(exe)
            or os.environ.get('LVP_VENV_RELAUNCHED')
            or os.path.normcase(os.path.abspath(sys.prefix))
            == os.path.normcase(os.path.abspath(venv))):
        return
    import subprocess
    print(f'[LumaViewPro] Relaunching with project venv: {exe}')
    env = dict(os.environ, LVP_VENV_RELAUNCHED='1')
    sys.exit(subprocess.call([exe, os.path.abspath(__file__), *sys.argv[1:]],
                             env=env))


if __name__ == '__main__':
    _relaunch_in_project_venv()

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
from kivy.config import Config
# Right/middle mouse buttons pan the plate map; without this Kivy turns a
# right-click into a simulated second finger and leaves a red dot behind.
Config.set('input', 'mouse', 'mouse,disable_multitouch')
from kivy.app import App
from kivy.base import ExceptionHandler, ExceptionManager
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, ListProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.splitter import Splitter, SplitterStrip
from kivy.factory import Factory

import plate_navigator
import acquisition as acq
import experiment
import online_processing
import well_file
from experiment_wizard import ExperimentWizardMixin

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
Factory.register('DragAwareScrollView', cls=plate_navigator.DragAwareScrollView)

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

# Folders used until the user chooses their own: with the program for the
# account that can write there, else in the user's Documents.
if acq.dir_writable(os.path.join(acq.APP_DIR, 'data')):
    CAPTURES_DIR = os.path.join(acq.APP_DIR, 'captures')
    PROTOCOLS_DIR = os.path.join(acq.APP_DIR, 'data', 'protocols')
else:
    CAPTURES_DIR = os.path.join(acq.user_documents_dir(), 'captures')
    PROTOCOLS_DIR = os.path.join(acq.user_documents_dir(), 'protocols')

# Basler daA3840-45um sensor pixel pitch is 2.0 um; divide by objective
# magnification for the default sample-plane scale (editable in the UI).
DEFAULT_UM_PER_PX = 0.5   # 2.0 um / 4x objective

OBJECTIVES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'objectives.json')
CAMERA_PIXEL_UM = 2.0     # sensor pixel pitch, unbinned
# Travel used to take up XY drive backlash before a measured move.
XY_BACKLASH_UM = 100.0
# Lumascope tube lens focal length (data/scopes.json: 47.8 mm on most
# models). Image scale = pixel pitch * objective focal length / this.
TUBE_FOCAL_MM = 47.8


LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
# Written on a clean shutdown after the stage was parked; the park guard
# process parks the stage itself when the app ends without leaving it.
PARKED_FLAG = os.path.join(LOGS_DIR, 'parked.flag')


class _LogAndContinue(ExceptionHandler):
    """An error in a button handler or a scheduled callback used to close
    the whole application (and, with it, any experiment being set up).
    Record the traceback in logs/errors.log, show it in the status bar and
    keep running."""

    _last = None
    _repeats = 0

    def handle_exception(self, exc):
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            return ExceptionManager.RAISE
        import traceback
        text = ''.join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        # The same error every frame must not flood the log.
        if text == self._last:
            self._repeats += 1
        else:
            self._last, self._repeats = text, 0
        if self._repeats < 3:
            try:
                os.makedirs(LOGS_DIR, exist_ok=True)
                with open(os.path.join(LOGS_DIR, 'errors.log'), 'a',
                          encoding='utf-8') as f:
                    f.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                            f'{text}\n')
            except Exception:
                pass
            print(text, flush=True)
        app = App.get_running_app()
        if app is not None:
            app.status_msg = (f'ERROR: {type(exc).__name__}: {exc}  '
                              f'(details in logs/errors.log)')
        return ExceptionManager.PASS


ExceptionManager.add_handler(_LogAndContinue())


class LumaViewProMainLayout(BoxLayout):
    pass


class ColumnStrip(SplitterStrip):
    """The divider between two columns of the window. Drag it to resize
    the column beside it; double-click it for the usual width. It lights
    up, and the mouse pointer becomes a left-right arrow, when the mouse
    is on it. The width is remembered for the user when it is let go."""
    hot = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from kivy.core.window import Window
        Window.bind(mouse_pos=self._on_mouse)

    def _on_mouse(self, window, pos):
        if self.get_root_window() is None:
            return
        hot = self.state == 'down' or self.collide_point(*self.to_widget(*pos))
        if hot != self.hot:
            self.hot = hot
            try:
                window.set_system_cursor('size_we' if hot else 'arrow')
            except Exception:
                pass

    def on_touch_down(self, touch):
        if touch.is_double_tap and self.collide_point(*touch.pos):
            app = App.get_running_app()
            if app is not None and self.parent is not None:
                app.reset_panel_width(self.parent)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        # Not on_release: that is not sent when the mouse is let go off
        # the strip, as it is when the column has reached its limit.
        mine = touch.grab_current is self
        result = super().on_touch_up(touch)
        app = App.get_running_app()
        if mine and app is not None:
            app.remember_panel_widths()
        return result


class ColumnSplitter(Splitter):
    """A column of the window with a ColumnStrip as its divider."""
    strip_cls = ObjectProperty(ColumnStrip)


class LumaViewProApp(ExperimentWizardMixin, App):
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
    # Progress of a protocol run, shown in a strip under the image.
    run_active = BooleanProperty(False)
    run_fraction = NumericProperty(0.0)
    run_line = StringProperty('')
    map_big = BooleanProperty(False)      # plate map fills the image area
    pos_edit_mode = StringProperty('')    # map clicks: '', 'move' or 'add'
    zman_info = StringProperty('')        # manual z-stack: what will be run
    zman_mode = StringProperty('ends')    # set by 'ends' or 'centre' +/-
    zman_order = StringProperty('slice')  # 'slice' by slice, or by 'channel'
    zman_step_text = StringProperty('5')  # its interval: one for all colours
    zman_marked = StringProperty('')      # colours ticked "Use in protocol"
    marked_channels = ListProperty([])    # the same, for the channel buttons
    channel_summary = StringProperty('')  # protocol colours and autofocus
    zstack_dir = StringProperty('')       # live z-stacks; '' = data folder
    # How protocol runs and z-stacks are saved: one file per well / stack
    # ('ome-tiff' or 'zarr'), one file for the whole run ('run-ome-tiff',
    # 'run-zarr'), or single-page TIFFs ('single').
    save_format = StringProperty('ome-tiff')
    lens_pos_info = StringProperty('')    # work position of the lens in use
    work_at_start_on = BooleanProperty(True)   # raise to it after homing
    # The plate map is drawn as a plate is always drawn: A1 top left. (It
    # was mirrored by default for a short while, together with the stage
    # direction; the stage direction was right, the mirrored map was not.)
    map_flip_x = BooleanProperty(False)
    # Settings are kept per Windows account.
    user_name = StringProperty(os.environ.get('USERNAME') or 'this user')
    settings_path = StringProperty(acq.SETTINGS_PATH)
    labware_options = ListProperty(['96-well plate'])
    objective_name = StringProperty('')
    objective_options = ListProperty([])
    labware_name = StringProperty('96-well plate')
    builtin_labware_names = ListProperty([])
    cal_active = BooleanProperty(False)
    cal_label = StringProperty('')
    cal_progress = StringProperty('')
    protocol_loaded = StringProperty('')
    cal_info = StringProperty('Plate not calibrated')
    use_surface_on = BooleanProperty(True)
    use_z_offsets_on = BooleanProperty(True)
    xy_step_um = NumericProperty(100.0)
    stage_x_um = NumericProperty(0.0)
    stage_y_um = NumericProperty(0.0)
    stage_z_um = NumericProperty(0.0)
    af_each_cycle_on = BooleanProperty(True)
    output_dir = StringProperty(CAPTURES_DIR)
    # The folder "Stitch folder" works on: the last tile scan, or one typed in.
    quark_folder = StringProperty('')
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

        self._load_objectives()

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
        # Focus map points the user has put somewhere else in their well:
        # {label: [dx_mm, dy_mm] from the well center}.
        self.focus_moves: dict = {}

        # Global z-stack plan (setup wizard): lens data for the suggested
        # step, and the opt-outs. zstack_optout maps a well name to the
        # channels that are NOT stacked there ('*' = no channel is).
        self.zplan: dict = {'na': None, 'mag': None, 'rate': 1.0,
                            'override_um': None, 'channels_off': []}
        self.zstack_optout: dict[str, list[str]] = {}
        # Work position per objective: the Z at which that lens is roughly
        # in focus, {objective name: z um}. The load position is the same
        # for every lens: objective fully down, stage centered.
        self.lens_work_z: dict[str, float] = {}
        # Widths of the left and right columns in dp, as the user left
        # them ({} = the usual widths).
        self.panel_widths: dict[str, float] = {}

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
            plate.on_point_double_click = self.on_plate_point_double_click
            plate.on_well_selected = self.on_well_selected
            plate.on_wells_toggle = self.on_wells_toggled
            plate.on_focus_scroll = self.on_focus_scroll
            plate.on_point_click = self.on_plate_point_click
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
        self._refresh_lens_info()
        self._apply_panel_widths()
        for split in self._panel_splits().values():
            split.bind(width=self._limit_panel)
        Window.bind(size=lambda *_: [self._limit_panel(s, s.width) for s in
                                     self._panel_splits().values()])
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
        if key in (13, 271) and self.cal_active:
            self.cal_mark()         # Enter = confirm the calibration point
            return True
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
            # Only for the real motor board, never a simulated stage.
            if self.stage_connected and                     type(stg).__module__ == 'motorboard':
                self._start_park_guard()
        except Exception as e:
            self._set_status(f'Stage error: {e}')

        # Startup calibration: objective fully lowered, XY homed against the
        # limit switches (firmware measures travel), then stage centered.
        if self.stage_connected:
            homed = self.hw.home_and_center(status_cb=self._set_status)
            # Then up to where this lens was last roughly in focus.
            z = self.lens_work_z.get(self.objective_name)
            if homed and self.work_at_start_on and z is not None:
                self._set_status(f'Raising {self.objective_name} to its work '
                                 f'position (z = {z:.0f} um)...')
                self.hw.move_z(float(z))
                self._set_status(f'Ready: {self.objective_name} at its work '
                                 f'position (z = {z:.0f} um). Use Coarse AF '
                                 f'to find focus.')
        else:
            self._set_status('Ready (no stage)')
        Clock.schedule_once(lambda dt: setattr(self, 'initializing', False), 0)

    def _start_park_guard(self):
        """Start the separate process that parks the stage (objective
        down, then stage centered) if this application ever ends without
        doing so itself: a crash, or a frozen window that gets killed."""
        if getattr(self, '_park_guard', None) is not None:
            return
        try:
            import subprocess
            here = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(LOGS_DIR, exist_ok=True)
            try:
                os.remove(PARKED_FLAG)
            except OSError:
                pass
            flags = 0
            if os.name == 'nt':
                # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: it must
                # outlive this process and its console window.
                flags = 0x00000008 | 0x00000200
            self._park_guard = subprocess.Popen(
                [sys.executable, os.path.join(here, 'park_guard.py'),
                 str(os.getpid()), PARKED_FLAG,
                 os.path.join(LOGS_DIR, 'park_guard.log')],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, close_fds=True,
                creationflags=flags, cwd=here)
        except Exception as e:
            print(f'[LumaViewPro] park guard NOT started: {e}', flush=True)

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
            # The plate map draws the same texture: repaint it too.
            plate = self.root.ids.get('plate_map')
            if plate is not None and plate.live_texture is not None:
                plate.canvas.ask_update()
            self._preview_when_resting(frame, ch)

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
        viewport = self.root.ids.get('main_camera_viewport')
        corners = self._live_image_corners_rel()
        plate.set_live_image(viewport.texture if viewport and corners
                             else None, corners)
        plate.set_marker_plate_mm(rel[0], rel[1])

    def _live_image_corners_rel(self):
        """Where the corners of the live image (top-left, top-right,
        bottom-right, bottom-left) are on the plate, in mm from the A1
        center; None when there is no live image to place.

        With the image motion calibrated, each corner is placed where the
        stage would have to be to bring it to the center, so the image is
        drawn with its real orientation and scale. Without it, the image
        is drawn upright at the scale given by the objective."""
        f = self.latest_frame
        if f is None or (self.worker_busy and self._mosaic is not None):
            return None
        h, w = f.shape[:2]
        return self._image_corners_rel(self.stage_x_um, self.stage_y_um,
                                       w, h)

    def _image_corners_rel(self, x_um: float, y_um: float, w: int, h: int):
        """Plate position (mm from the A1 center) of the corners of a
        ``w`` x ``h`` pixel image taken with the stage at (x_um, y_um)."""
        px = ((-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0),
              (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0))
        inv = None
        A = self._img_cal_matrix()
        if A is not None:
            try:
                inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                inv = None
        out = []
        if inv is not None:
            for p in px:
                s = -(inv @ np.asarray(p))
                rel = self.stage_to_plate_rel(x_um + float(s[0]),
                                              y_um + float(s[1]))
                if rel is None:
                    return None
                out.append(rel)
            return out
        c = self.stage_to_plate_rel(x_um, y_um)
        if c is None:
            return None
        mm_per_px = self.effective_um_per_px() / 1000.0
        return [(c[0] + p[0] * mm_per_px, c[1] + p[1] * mm_per_px)
                for p in px]

    # ------------------------------------- previews left on the plate map

    PREVIEW_PX = 192          # longer side of a preview left on the map
    PREVIEW_REST_S = 2.0      # live view at rest this long leaves one

    def _preview_rgb(self, frame, ch: str):
        """Small display copy of a frame in its channel colour (uint8,
        h x w x 3), scaled like the live display. Thread safe."""
        if frame is None or frame.ndim != 2 or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        f = max(max(w, h) // self.PREVIEW_PX, 1)
        small = frame[::f, ::f]
        if self.autoscale_on:
            lo, hi = float(small.min()), float(small.max())
            g = ((small.astype(np.float32) - lo) * (255.0 / (hi - lo))
                 if hi > lo else np.zeros(small.shape, np.float32))
        else:
            g = small.astype(np.float32) / float(
                2 ** max(self.significant_bits - 8, 0))
        g = np.clip(g, 0, 255)
        tint = CHANNEL_LUTS.get(ch, (1.0, 1.0, 1.0))
        return np.stack([(g * t).astype(np.uint8) for t in tint[:3]],
                        axis=-1)

    def leave_preview(self, ch: str, frame, x_um: float, y_um: float):
        """Leave (or renew) the preview of an image at the stage position
        it was taken at. May be called from the acquisition thread."""
        try:
            rgb = self._preview_rgb(frame, ch)
        except Exception:
            return
        if rgb is None:
            return
        h, w = frame.shape[:2]

        def _add(dt):
            plate = self._plate_map()
            if plate is None or self.no_plate_on:
                return
            corners = self._image_corners_rel(x_um, y_um, w, h)
            if corners:
                plate.add_snapshot(rgb, corners)
        Clock.schedule_once(_add, 0)

    def clear_map_previews(self):
        plate = self._plate_map()
        if plate is not None:
            plate.clear_snapshots()

    def _preview_when_resting(self, frame, ch: str):
        """Called for every live frame: once the stage has been still
        with the light on for PREVIEW_REST_S, the frame stays on the map
        (again after each further move or refocus)."""
        if self.worker_busy or ch not in CHANNELS or \
                self.live_channel not in CHANNELS:
            self._rest = None
            return
        key = (round(self.stage_x_um), round(self.stage_y_um),
               round(self.stage_z_um), ch)
        now = time.time()
        rest = getattr(self, '_rest', None)
        if rest is None or rest['key'] != key:
            self._rest = {'key': key, 'since': now, 'done': False}
        elif not rest['done'] and now - rest['since'] >= self.PREVIEW_REST_S:
            rest['done'] = True
            self.leave_preview(ch, frame, self.stage_x_um, self.stage_y_um)

    def _plate_map(self):
        return self.root.ids.get('plate_map') if self.root else None

    def set_map_big(self, big: bool):
        """Show the plate map in the image area (large) or in its panel.
        Large, the camera view makes way for it: the live image is drawn
        on the map at its real place, size and channel colour."""
        ids = self.root.ids if self.root else {}
        plate, home = ids.get('plate_map'), ids.get('plate_map_home')
        holder, viewport = ids.get('big_map_holder'), \
            ids.get('main_camera_viewport')
        if None in (plate, home, holder, viewport) or \
                bool(big) == self.map_big:
            return
        plate = getattr(plate, '__self__', plate)   # keep it alive unparented
        if plate.parent is not None:
            plate.parent.remove_widget(plate)
        if big:
            viewport.size_hint_x, viewport.width = None, 0
            viewport.opacity = 0
            holder.size_hint_x = 1
            holder.add_widget(plate)
            # Zooming in then goes to the image, and the view stays with
            # it as the stage moves (a right-drag pan releases it).
            plate.follow_stage = True
        else:
            holder.size_hint_x, holder.width = None, 0
            viewport.size_hint_x = 1
            viewport.opacity = 1
            home.add_widget(plate)
        plate.wheel_focus = bool(big)
        self.map_big = bool(big)
        # The same plate point stays in the middle at the new size.
        Clock.schedule_once(lambda dt: (plate.redraw(),
                                        self._update_plate_marker()), 0)

    def toggle_map_flip(self):
        """Mirror the plate map left to right (the picture only), and
        remember the choice for this user."""
        self.map_flip_x = not self.map_flip_x
        self._persist()
        self._set_status('Plate map: column 1 on the '
                         + ('right (mirrored)' if self.map_flip_x else 'left'))

    def plate_zoom(self, what: str):
        """Zoom buttons under the plate map: in, out, fit, stage."""
        plate = self._plate_map()
        if plate is None:
            return
        if what == 'in':
            plate.zoom_by(1.5)
        elif what == 'out':
            plate.zoom_by(1.0 / 1.5)
        elif what == 'fit':
            plate.zoom_fit()
        elif what == 'stage':
            plate.follow_stage = True
            plate.zoom_to_stage()

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
        self._refresh_positions()
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
        self._cal_rejected = {}
        self.cal_active = True
        self._cal_log(f'--- start, labware {self.current_labware.name}')
        self._set_status('Calibration: drive the FOV center to the marked '
                         'position, then press CONFIRM THIS POINT (or Enter)')
        # Show the well being calibrated, with the live image on it, and
        # keep the stage position in the middle as it moves between rims.
        plate = self._plate_map()
        if plate is not None and plate.marker_plate_mm is not None:
            plate.follow_stage = True
            plate.zoom_to_span(plate.marker_plate_mm,
                               max(plate._well_size()) * 1.6)
        self._cal_show_current(move=True)

    def _cal_show_current(self, move: bool = True):
        wiz = self.cal_wizard
        if wiz is None:
            return
        t = wiz.current
        n = len(wiz.targets)
        done = '  - CONFIRMED' if t.marked_stage_um is not None else ''
        self.cal_label = f'Point {wiz.index + 1} of {n}: {t.label}{done}'
        self.cal_progress = self._cal_progress_text()
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(t.plate_mm)
        if move:
            # Drive to where the target is expected: from the marks made
            # so far, else the previous calibration / A1 anchor, else the
            # plate-centered-on-the-stage assumption.
            est = wiz.estimate_stage_for(
                t, self.plate_rel_to_stage,
                (self.plate_sign_x, self.plate_sign_y))
            if est is None:
                return
            if not acq.in_stage_bounds(*est):
                self._set_status(f'{t.label}: expected position is outside '
                                 f'the stage travel - jog there by hand')
                return
            self._bg(self.hw.move_xy, est[0], est[1])

    def _cal_missing(self) -> list[int]:
        wiz = self.cal_wizard
        return [] if wiz is None else [
            i for i, t in enumerate(wiz.targets) if t.marked_stage_um is None]

    def _cal_progress_text(self) -> str:
        wiz = self.cal_wizard
        if wiz is None:
            return ''
        missing = self._cal_missing()
        n = len(wiz.targets)
        text = f'{n - len(missing)} of {n} points confirmed.'
        if missing:
            text += ' Still to confirm: ' + ', '.join(
                wiz.targets[i].label for i in missing) + '.'
        return text

    def _cal_goto_missing(self, why: str) -> bool:
        """Go to the first point that has not been confirmed; False when
        every point is confirmed."""
        wiz = self.cal_wizard
        missing = self._cal_missing()
        if wiz is None or not missing:
            return False
        wiz.index = missing[0]
        self._cal_show_current(move=True)
        self._set_status(
            f'{why}{len(missing)} point{"s" if len(missing) != 1 else ""} '
            f'not confirmed yet - going to {wiz.current.label}. Center the '
            f'rim, then press CONFIRM THIS POINT (or Enter).')
        return True

    def cal_mark(self):
        """Confirm the shown target: record where the stage is now, then
        go on to the next one."""
        wiz = self.cal_wizard
        if wiz is None:
            return
        if not self.hw.xy_settled() and \
                getattr(self, '_cal_moving_warned', None) != wiz.index:
            # Ask once per point; a second press confirms regardless, so
            # a stage that never reports "arrived" cannot block the user.
            self._cal_moving_warned = wiz.index
            self._cal_log(f'confirm pressed at {wiz.current.label} while '
                          f'the stage reported moving - asked to repeat')
            self._set_status('The stage reports it is still moving. Wait '
                             'for it to stop, then press CONFIRM again.')
            return
        self._cal_moving_warned = None
        # Read the stage itself: the displayed position is refreshed only
        # twice a second and may lag a jog made just before the press.
        now = self.hw.get_pos_now()
        x, y = (now[0], now[1]) if now is not None else \
            (self.stage_x_um, self.stage_y_um)
        wiz.mark_current(x, y)
        self._cal_log(f'marked {wiz.current.label} at ({x:.0f}, {y:.0f})')
        self._set_status(f'Confirmed {wiz.current.label}')
        if all(t.marked_stage_um is not None for t in wiz.targets[:4]) \
                and wiz.matrix is None:
            # All four A1 edges marked: provisional solve enables
            # auto-driving to the two long-range targets.
            wiz.provisional_solve()
        if wiz.complete:
            self.cal_finish()
            return
        if wiz.next():
            self._cal_show_current(move=True)
            return
        # Last target marked but an earlier one was stepped over: go back
        # to it instead of leaving the calibration silently unfinished.
        self._cal_goto_missing('')

    def cal_next(self):
        """Skip to the next point WITHOUT confirming the current one."""
        wiz = self.cal_wizard
        if wiz is None:
            return
        t = wiz.current
        skipped = t.marked_stage_um is None
        self._cal_log(f'skip pressed at {t.label} '
                      f'({"not confirmed" if skipped else "confirmed"})')
        if wiz.next():
            self._cal_show_current(move=True)
            if skipped:
                self._set_status(
                    f'{t.label} was skipped, NOT confirmed. A point only '
                    f'counts when you press CONFIRM THIS POINT (or Enter).')
        elif not self._cal_goto_missing('That was the last point. '):
            self.cal_finish()

    def cal_prev(self):
        if self.cal_wizard and self.cal_wizard.prev():
            self._cal_log(f'back to {self.cal_wizard.current.label}')
            self._cal_show_current(move=True)

    def cal_cancel(self):
        self.cal_wizard = None
        self.cal_active = False
        self.cal_label = ''
        self.cal_progress = ''
        self._cal_log('canceled')
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(None)
            plate.zoom_fit()
        self._set_status('Calibration canceled')

    def cal_finish(self):
        wiz = self.cal_wizard
        if wiz is None:
            return
        if not wiz.complete:
            self._cal_log('finish pressed with points missing')
            self._cal_goto_missing('Cannot finish: ')
            return
        try:
            solved = wiz.solve()
        except Exception as e:
            self._cal_log(f'solve raised {type(e).__name__}: {e}')
            solved = False
        if not solved:
            self._cal_log('solve FAILED, marks: ' + str(
                [t.marked_stage_um for t in wiz.targets]))
            self._set_status('Calibration solve FAILED - two marks may be '
                             'at the same place. Use Previous to re-mark.')
            return
        problems = wiz.scale_problems()
        if problems:
            idx, text = problems[0]
            mark = wiz.targets[idx].marked_stage_um
            before = getattr(self, '_cal_rejected', {}).get(idx)
            if before is not None and math.hypot(
                    mark[0] - before[0], mark[1] - before[1]) < 300.0:
                self._cal_log(f'accepted on the second confirmation: {text}')
            else:
                # A plate cannot be stretched: this mark is on another
                # well (or the definition is for another plate). Take the
                # mark back and drive to where the well should be.
                self._cal_rejected = {idx: mark}
                label = wiz.targets[idx].label
                wiz.targets[idx].marked_stage_um = None
                wiz.matrix = None
                wiz.provisional_solve()
                wiz.index = idx
                self._cal_log(f'NOT ACCEPTED: {text}')
                self._cal_show_current(move=True)
                self._set_status(f'Calibration NOT accepted: {text}. '
                                 f'Confirm {label} again.')
                self._info_popup(
                    'Calibration NOT accepted',
                    f'{text}.\n\nThe stage is going to where {label} '
                    f'should be. Bring that rim to the image center and '
                    f'confirm it again.\n\n(If this plate really is spaced '
                    f'like that, confirm at the same place a second time '
                    f'to accept it.)')
                return
        # Positions, the AF point and the focus points belong to the
        # plate: carry them over to where the plate is now.
        anchored = self._plate_anchored_snapshot()
        self.plate_cal = wiz
        self._plate_anchored_restore(anchored)
        self.cal_wizard = None
        self.cal_active = False
        self.cal_label = 'CALIBRATION COMPLETE'
        self.cal_progress = 'All 6 points confirmed.'
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate:
            plate.set_target_plate_mm(None)
            plate.zoom_fit()
            # Previews were placed with the old plate position.
            plate.clear_snapshots()
        self._refresh_cal_info()
        self._cal_log('complete: ' + self.cal_info)
        self._set_status('Plate calibrated: ' + self.cal_info)
        self._persist()
        self._update_plate_marker()
        # Positions and selected wells are drawn through the calibration.
        self._refresh_positions()
        self._refresh_surface_info()
        self._cal_done_popup()

    def _info_popup(self, title: str, text: str):
        try:
            from kivy.metrics import dp
            from kivy.uix.button import Button
            box = BoxLayout(orientation='vertical', spacing=dp(8),
                            padding=dp(10))
            msg = Label(text=text, halign='center', valign='middle')
            msg.bind(size=lambda w, s: setattr(w, 'text_size', s))
            ok = Button(text='OK', size_hint_y=None, height=dp(36))
            box.add_widget(msg)
            box.add_widget(ok)
            popup = Popup(title=title, content=box, size_hint=(None, None),
                          size=(dp(460), dp(320)))
            ok.bind(on_press=popup.dismiss)
            popup.open()
        except Exception:
            pass

    def _cal_done_popup(self):
        """Unmissable confirmation that the calibration was accepted."""
        try:
            from kivy.metrics import dp
            from kivy.uix.button import Button
            box = BoxLayout(orientation='vertical', spacing=dp(8),
                            padding=dp(10))
            msg = Label(text='The plate is calibrated.\n\n'
                        + self.cal_info.replace('Calibrated: ', '')
                        .replace(', ', '\n'), halign='center')
            msg.bind(size=lambda w, s: setattr(w, 'text_size', s))
            ok = Button(text='OK', size_hint_y=None, height=dp(36))
            box.add_widget(msg)
            box.add_widget(ok)
            popup = Popup(title='Plate calibration complete', content=box,
                          size_hint=(None, None), size=(dp(420), dp(280)))
            ok.bind(on_press=popup.dismiss)
            popup.open()
        except Exception:
            pass

    # ------------------------------ things that are fixed to the plate

    def _plate_located(self) -> bool:
        """True when the plate position is measured, not just assumed."""
        return (self.plate_cal is not None
                and self.plate_cal.matrix is not None) or \
            bool(self.a1_stage_um)

    def _plate_anchored_snapshot(self):
        """Plate coordinates (mm from the A1 center) of every position,
        the AF point and the focus points; None when they cannot be tied
        to the plate (no plate, or the plate was never located).
        A configuration loaded before calibrating supplies its own."""
        pending, self._pending_plate_rel = \
            getattr(self, '_pending_plate_rel', None), None
        if self.no_plate_on:
            return None
        if pending:
            return pending
        if not self._plate_located():
            return None
        rel = self.stage_to_plate_rel
        a = self.surface.anchor_xy
        return {
            'positions': [rel(p['x_um'], p['y_um'])
                          for p in self.protocol_positions],
            'anchor': rel(a[0], a[1]) if a else None,
            'points': [rel(p[0], p[1]) for p in self.surface.points],
        }

    def _plate_anchored_restore(self, snap) -> None:
        """Put positions, AF point and focus points back at their plate
        coordinates using the current plate transform."""
        if not snap:
            return
        to_stage = self.plate_rel_to_stage
        rels = snap.get('positions') or []
        if len(rels) == len(self.protocol_positions):
            for p, r in zip(self.protocol_positions, rels):
                if r is not None:
                    p['x_um'], p['y_um'] = [float(v) for v in to_stage(*r)]
        if snap.get('anchor') and self.surface.anchor_xy:
            self.surface.anchor_xy = tuple(
                float(v) for v in to_stage(*snap['anchor']))
        rels = snap.get('points') or []
        if len(rels) == len(self.surface.points):
            self.surface.points = [
                (*[float(v) for v in to_stage(*r)], *p[2:])
                if r is not None else p
                for p, r in zip(self.surface.points, rels)]

    def _cal_log(self, msg: str):
        """Append to logs/calibration.log (what was marked, the result)."""
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open(os.path.join(LOGS_DIR, 'calibration.log'), 'a',
                      encoding='utf-8') as f:
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}\n')
        except Exception:
            pass

    def well_size_mm(self) -> tuple[float, float]:
        """Well size (x, y) in plate mm: as measured by the plate
        calibration, else from the labware definition."""
        c = self.plate_cal
        if c is not None and c.matrix is not None and \
                c.measured_well_x_mm and c.measured_well_y_mm:
            return float(c.measured_well_x_mm), float(c.measured_well_y_mm)
        lw = self.current_labware
        return lw.well_size_x_mm, lw.well_size_y_mm

    def _refresh_cal_info(self):
        c = self.plate_cal
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is not None:
            measured = c is not None and c.matrix is not None and \
                c.measured_well_x_mm and c.measured_well_y_mm
            plate.well_size_override = self.well_size_mm() if measured \
                else None
            plate.redraw()
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
        self._refresh_channel_marks()
        self.zman_refresh()   # marked colours, optical section of this one

    # -------- which colours are imaged and which one focuses: the P and
    # AF marks under each channel button

    def _refresh_channel_marks(self, *_):
        if not getattr(self, 'channel_settings', None):
            return      # a setting changed before the app was built
        marked =[ch for ch in CHANNELS if self.channel_settings[ch].enabled]
        self.marked_channels = marked
        af = self.af_channel
        others = [ch for ch in marked if ch != af]
        text = ('Imaged: ' + ', '.join(marked) if marked
                else 'No colour is imaged (press P under a colour)')
        text += f'.  Autofocus: on {af} only'
        if others:
            text += ('; ' + ', '.join(others) + ' follow'
                     + ('s' if len(others) == 1 else '') + ' by Z offset'
                     if self.use_z_offsets_on else
                     '; ' + ', '.join(others) + ' at the same Z')
        if marked and af not in marked:
            text += f'.  {af} focuses but is not imaged'
        self.channel_summary = text + '.'

    def on_af_channel(self, *_):
        self._refresh_channel_marks()

    def on_use_z_offsets_on(self, *_):
        self._refresh_channel_marks()

    def toggle_channel_protocol(self, channel: str):
        """P under a channel button: image this colour in protocols,
        marked stacks and scans, or leave it out."""
        cs = self.channel_settings[channel]
        cs.enabled = not cs.enabled
        self._sync_channel_fields()
        self._persist()
        self._set_status(f'{channel} ' + ('is imaged' if cs.enabled
                                          else 'is left out') +
                         ' in protocols, marked stacks and scans')

    def set_af_channel_to(self, channel: str):
        """AF under a channel button: this is the colour autofocus runs
        in (there is one; the others follow by their Z offsets)."""
        if channel == self.af_channel:
            self._set_status(f'Autofocus already runs in {channel}; press '
                             f'AF under another colour to change it')
            return
        self.af_channel = channel
        self.af_config.channel = channel
        self.surface.anchor_channel = channel
        self._sync_channel_fields()
        self._persist()
        self._set_status(f'Autofocus channel: {channel}')

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
            self._refresh_channel_marks()
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

    def _load_objectives(self):
        """Objective list (Etaluma data/objectives.json): magnification,
        focal length, NA and autofocus step suggestions per lens."""
        import json
        self.objectives = {}
        try:
            with open(OBJECTIVES_PATH, encoding='utf-8') as f:
                self.objectives = json.load(f)
        except Exception:
            pass
        self.objective_options = list(self.objectives)

    def objective_na(self) -> float | None:
        obj = self.objectives.get(self.objective_name)
        return float(obj['aperture']) if obj and obj.get('aperture') else None

    def set_objective(self, name: str):
        """Select the objective in use: sets the image scale from its
        focal length and pre-fills the lens-dependent wizard fields."""
        obj = self.objectives.get(name)
        if obj is None or name == self.objective_name:
            return
        previous = self.objective_name
        self.objective_name = name
        self.base_um_per_px = round(
            CAMERA_PIXEL_UM * float(obj['focal_length']) / TUBE_FOCAL_MM, 4)
        # A measured pixel-to-stage mapping belongs to the previous lens.
        self.img_cal = None
        self.zplan['na'] = None
        ids = self.root.ids if self.root else {}
        if ids:
            ids.um_per_px_input.text = str(self.base_um_per_px)
            if obj.get('AF_max'):
                ids.wiz_coarse_step.text = f"{float(obj['AF_max']):g}"
            if obj.get('AF_range'):
                ids.wiz_fine_range.text = f"{float(obj['AF_range']) / 2.0:g}"
            if obj.get('z_fine'):
                ids.wiz_fine_step.text = f"{float(obj['z_fine']):g}"
        self._persist()
        note = ''
        laid = getattr(self, 'positions_objective', '')
        if self.protocol_positions and laid and laid != name:
            note = (f' The positions were laid out for {laid}: generate '
                    f'them again (wizard step 6) so the fields fit this '
                    f'lens.')
        self._set_status(
            f'Objective {name}: NA {obj.get("aperture", "?")}, image scale '
            f'{self.base_um_per_px:g} um/px. Re-run "Auto-calibrate '
            f'centering" for this lens.' + note)
        if previous and self.wiz_active:
            self._wiz_refresh_info()
        self._refresh_lens_info()
        self.zman_refresh()

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
        if ch in CHANNELS:
            self.leave_preview(ch, frame, self.stage_x_um, self.stage_y_um)
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

    # ---------------------- load / work position per lens, coarse focus

    Z_TRAVEL_UM = 14000.0
    COARSE_AF_STEPS_EACH_WAY = 30     # "near here": this many lens steps

    def _lens_af_step(self) -> float:
        """Large autofocus step for the lens in use (objectives.json
        AF_max: the widest step that still cannot step over the focus)."""
        obj = self.objectives.get(self.objective_name) or {}
        try:
            step = float(obj.get('AF_max') or 0.0)
        except (TypeError, ValueError):
            step = 0.0
        return step if step > 0 else 100.0

    def _refresh_lens_info(self):
        name = self.objective_name
        if not name:
            self.lens_pos_info = ('No objective selected: select it in the '
                                  'top bar first.')
            return
        z = self.lens_work_z.get(name)
        step = self._lens_af_step()
        near = step * self.COARSE_AF_STEPS_EACH_WAY
        self.lens_pos_info = (
            f'{name}: work position '
            + (f'z = {z:.0f} um' if z is not None else
               'NOT SET (focus, then press "Work position = here")')
            + f'. Coarse autofocus steps {step:g} um; "near here" searches '
              f'{near:g} um each way. Load position: objective fully down, '
              f'stage centered.')

    def lens_set_work_here(self):
        """Remember the present Z as the work position of this lens."""
        name = self.objective_name
        if not name:
            self._set_status('Select the objective first')
            return
        self.lens_work_z[name] = round(float(self.stage_z_um), 1)
        self._persist()
        self._refresh_lens_info()
        self._set_status(f'{name}: work position = z '
                         f'{self.lens_work_z[name]:.0f} um (saved for '
                         f'{self.user_name})')

    def lens_goto_work(self):
        z = self.lens_work_z.get(self.objective_name)
        if z is None:
            self._set_status(f'No work position saved for '
                             f'{self.objective_name or "this lens"} yet')
            return
        if not self.stage_connected or self.worker_busy:
            return
        self._set_status(f'{self.objective_name}: going to the work '
                         f'position (z = {z:.0f} um)...')

        def _job():
            self.hw.move_z(float(z))
            self._set_status(f'{self.objective_name} at its work position '
                             f'(z = {z:.0f} um)')
        self._bg(_job)

    def lens_goto_load(self):
        """Objective fully down (confirmed), then the stage to the center:
        for changing the lens or the plate."""
        if not self.stage_connected or self.worker_busy:
            return
        self._set_status('Going to the load position: lowering the '
                         'objective...')

        def _job():
            ok = self.hw.park()
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel',
                                                   'OFF'), 0)
            self.hw.leds_off()
            self._set_status(
                'Load position: objective down, stage centered' if ok else
                'LOAD POSITION NOT REACHED: the objective could not be '
                'confirmed down, so the stage was not moved')
        self._bg(_job)

    def coarse_af(self, whole: bool = False):
        """Find focus here in large steps suited to the lens: around the
        present Z, or over the whole travel. The fine autofocus range is
        then centered on what was found."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Autofocus needs stage + camera')
            return
        if not self.objective_name:
            self._set_status('Select the objective first: the step is '
                             'taken from the lens')
            return
        step = self._lens_af_step()
        z0 = self.stage_z_um
        if whole:
            lo, hi = 0.0, self.Z_TRAVEL_UM
        else:
            half = step * self.COARSE_AF_STEPS_EACH_WAY
            lo, hi = max(z0 - half, 0.0), min(z0 + half, self.Z_TRAVEL_UM)
        cfg = AutofocusConfig(z_bottom_um=lo, z_top_um=hi, step_um=step,
                              channel=self.af_channel)
        self._set_status(f'Coarse autofocus ({self.objective_name}): '
                         f'{lo:.0f} to {hi:.0f} um in {step:g} um steps...')

        def _job():
            best, ok = self._wiz_sweep(cfg)
            if best is None:
                self._set_status('Coarse autofocus canceled')
                return
            if not ok:
                self.hw.move_z(z0)
                self._set_status(
                    f'No clear focus between {lo:.0f} and {hi:.0f} um. Is '
                    f'there structure in view and is the exposure right?'
                    + ('' if whole else ' Otherwise try "Whole travel".'))
                return
            self.hw.move_z(best)
            Clock.schedule_once(lambda dt: self._wiz_fine_window(best), 0)
            self._set_status(f'Coarse focus found at z = {best:.1f} um; '
                             f'fine autofocus range centered on it')
        self._bg(_job, busy_job=True)

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

    def _dir_note(self, path: str) -> str:
        """' (saved for <user>)', or a warning when nothing can be written
        there now. The choice is kept either way (a network drive may
        simply not be connected yet)."""
        if acq.dir_writable(path):
            return f' (saved for {self.user_name})'
        return (f' - CANNOT WRITE THERE NOW (drive not connected, or no '
                f'permission); the choice is still saved for '
                f'{self.user_name}')

    def set_output_dir(self, text: str):
        self.output_dir = text.strip() or CAPTURES_DIR
        self._persist()
        self._set_status(f'Protocol data directory: {self.output_dir}'
                         + self._dir_note(self.output_dir))

    def browse_dir(self, which: str):
        """Pick the data ('output') or autosave ('protocol') folder in the
        Windows folder window."""
        current = {'output': self.output_dir,
                   'zstack': self.zstack_dir or self.output_dir
                   }.get(which, self.protocol_dir)
        try:
            import tkinter
            from tkinter import filedialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askdirectory(
                parent=root, mustexist=False,
                initialdir=current if os.path.isdir(current) else None,
                title={'output': 'Folder for the image data',
                       'zstack': 'Folder for z-stacks taken by hand'
                       }.get(which, 'Folder for autosaved and saved protocols'))
            root.destroy()
        except Exception as e:
            self._set_status(f'Folder window not available ({e}): type the '
                             f'path and press Set')
            return
        if not path:
            return
        path = os.path.normpath(path)
        if which == 'output':
            self.set_output_dir(path)
        elif which == 'zstack':
            self.set_zstack_dir(path)
        else:
            self.set_protocol_dir(path)

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

    def _pos_well(self, x_um: float, y_um: float) -> tuple[int, int] | None:
        """(row, col) of the well a stage position falls in, or None."""
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
                return row, col
        return None

    def _pos_well_name(self, x_um: float, y_um: float) -> str | None:
        well = self._pos_well(x_um, y_um)
        return None if well is None else             self.current_labware.get_well_name(*well)

    def on_wells_toggled(self, wells: list[tuple[int, int]], turn_on: bool):
        """Plate navigator click / rectangle: wells that are "on" are the
        wells holding a protocol position. Turning a well on adds a position
        at its center; turning it off removes every position inside it."""
        if self.pos_review_active:
            self._set_status('Finish the position review before editing wells')
            return
        lw = self.current_labware
        drift = self.surface.drift_um

        def _make(well):
            row, col = well
            x_um, y_um = self.plate_rel_to_stage(
                col * (lw.spacing_x_mm or 0.0),
                row * (lw.spacing_y_mm or 0.0))
            z_um = self.surface.z_at(x_um, y_um)
            if z_um is None:
                z_um = self.stage_z_um
            return {'name': lw.get_well_name(row, col),
                    'x_um': x_um, 'y_um': y_um, 'z_um': z_um - drift}

        self.protocol_positions, n = plate_navigator.toggle_well_positions(
            self.protocol_positions, wells, turn_on,
            lambda p: self._pos_well(p['x_um'], p['y_um']), _make)
        if not n:
            return
        verb = 'added' if turn_on else 'removed'
        self._persist()
        self._refresh_positions()
        note = ''
        if turn_on and self.plate_cal is None and self.a1_stage_um is None:
            note = ' - plate not calibrated, positions are approximate'
        self._set_status(f'{n} position{"s" if n != 1 else ""} {verb} '
                         f'({len(self.protocol_positions)} total){note}')

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
            plate.active_wells = {
                w for w in (self._pos_well(p['x_um'], p['y_um'])
                            for p in self.protocol_positions)
                if w is not None}
            plate.set_position_markers(marks, highlight)
        self.fmap_marks_refresh()
        if self.wiz_active:
            self._wiz_refresh_info()

    # ------------------ moving and adding positions, designations kept

    def _designated_positions(self) -> list[dict]:
        """The positions with their well and tile/point tag written in,
        so that moving one does not change what it is called."""
        if self.no_plate_on:
            # Free positions: the wizard still tells the places that were
            # added by hand (no tag yet) from the sites it generated.
            return [dict(p) for p in self.protocol_positions]
        well_of = (lambda p: self._pos_well_name(p['x_um'], p['y_um'])) \
            if self._use_wells() else None
        return experiment.assign_designations(self.protocol_positions,
                                              well_of)

    def _edit_target_well(self, x_um: float, y_um: float):
        """(ok, well name or None) for a point that is to hold a position."""
        if self.pos_review_active:
            self._set_status('Finish the position review first')
            return False, None
        if not acq.in_stage_bounds(x_um, y_um):
            self._set_status('That point is outside the stage travel')
            return False, None
        if not self._use_wells():
            return True, None
        rc = self._pos_well(x_um, y_um)
        inside = False
        if rc is not None:
            # Inside the well itself (as measured by the calibration),
            # with a little margin - not merely nearest to it.
            lw = self.current_labware
            rel = self.stage_to_plate_rel(x_um, y_um)
            dx = rel[0] - rc[1] * (lw.spacing_x_mm or 0.0)
            dy = rel[1] - rc[0] * (lw.spacing_y_mm or 0.0)
            hx, hy = (v / 2.0 * 1.1 for v in self.well_size_mm())
            if getattr(lw, 'shape', 'round') == 'square':
                inside = abs(dx) <= hx and abs(dy) <= hy
            else:
                inside = (dx / hx) ** 2 + (dy / hy) ** 2 <= 1.0
        if not inside:
            self._set_status('That point is not inside a well')
            return False, None
        return True, self.current_labware.get_well_name(*rc)

    def _edit_z_of(self, z_here: float | None):
        """Focus for a moved or added position: the focus surface where
        there is one, else the focus the stage is at now (when the point
        is where the stage is), else what the position had."""
        drift = self.surface.drift_um

        def z_of(x, y, old):
            z = self.surface.z_at(x, y)
            if z is not None:
                return z - drift
            return old if z_here is None else z_here
        return z_of

    def pos_move_to(self, x_um: float, y_um: float,
                    z_here: float | None = None):
        """Move the nearest position of the well at (x_um, y_um) there; a
        tile scan moves as a whole, centered on the point. The well and
        the tile/point tags stay as they are."""
        ok, well = self._edit_target_well(x_um, y_um)
        if not ok:
            return
        new, what = experiment.move_in_well(
            self._designated_positions(), well, x_um, y_um,
            z_of=self._edit_z_of(z_here))
        if not what:
            self._set_status(f'{well or "There"} holds no position to move: '
                             f'switch the well on, or add a point')
            return
        self.protocol_positions = new
        self._persist()
        self._refresh_positions()
        self._set_status(f'{what} moved; it keeps its name')

    def pos_add_at(self, x_um: float, y_um: float,
                   z_here: float | None = None):
        """Add a point that takes the name of the well it is dropped in
        and the next point tag there (B3_P1, B3_P2, ...)."""
        ok, well = self._edit_target_well(x_um, y_um)
        if not ok:
            return
        z = self._edit_z_of(z_here)(
            x_um, y_um, self.stage_z_um - self.surface.drift_um)
        self.protocol_positions, point = experiment.add_point(
            self._designated_positions(), well, x_um, y_um, z)
        self._persist()
        self._refresh_positions()
        self._set_status(f'Point {point["name"]} added '
                         f'({len(self.protocol_positions)} positions)')

    def pos_move_here(self):
        """Move the nearest position (or tile scan) of this well to where
        the stage is now."""
        self.pos_move_to(self.stage_x_um, self.stage_y_um,
                         z_here=self.stage_z_um - self.surface.drift_um)

    def set_pos_edit(self, mode: str):
        """Clicks on the plate map move ('move') or add ('add') positions,
        or move a focus map point ('focus'), instead of switching wells on
        and off; the same mode again, or '', goes back to normal."""
        self.pos_edit_mode = '' if mode == self.pos_edit_mode else mode
        plate = self._plate_map()
        if plate is not None:
            plate.point_click_mode = bool(self.pos_edit_mode)
        self.fmap_marks_refresh()
        self._set_status({
            'focus': 'Click in a well: its focus point (the square with a '
                     'cross) moves to the click. Zoom in first (Ctrl + '
                     'wheel). A double-click still drives the stage.',
            'move': 'Click in a well: its position (or the center of its '
                    'tile scan) moves to the click. Zoom in first (Ctrl + '
                    'wheel). A double-click still drives the stage.',
            'add': 'Click in a well: a point is added there, named after '
                   'the well. Zoom in first (Ctrl + wheel). A double-click '
                   'still drives the stage.',
        }.get(self.pos_edit_mode, 'Map clicks switch wells on and off again'))

    def on_plate_point_click(self, rel_x_mm: float, rel_y_mm: float,
                             well=None):
        """A single click on the plate map while a position edit mode is
        on (mm from the A1 center)."""
        target = self.plate_rel_to_stage(rel_x_mm, rel_y_mm)
        if target is None:
            self._set_status('Calibrate the plate first (or "Set A1 = here")')
            return
        if self.pos_edit_mode == 'move':
            self.pos_move_to(float(target[0]), float(target[1]))
        elif self.pos_edit_mode == 'add':
            self.pos_add_at(float(target[0]), float(target[1]))
        elif self.pos_edit_mode == 'focus':
            self.focus_point_move_to(float(target[0]), float(target[1]))

    def position_add_here(self):
        """Add the current stage position (and focus) as a protocol
        interaction point. With a plate it takes the name of the well and
        the next point tag in it."""
        if not self.no_plate_on:
            self.pos_add_at(self.stage_x_um, self.stage_y_um,
                            z_here=self.stage_z_um - self.surface.drift_um)
            return
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
        self._set_status(f'Protocol autosave directory: {self.protocol_dir}'
                         + self._dir_note(self.protocol_dir))

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
            'no_plate': self.no_plate_on,
            'focus_mode': self.wiz_focus_mode,
            'focus_moves': self.focus_moves,
            'af_every_n': int(self.af_every_n),
            'zplan': self.zplan,
            'zstack_optout': self.zstack_optout,
            'compose_on': bool(self.compose_on),
            'stitch_on': bool(self.stitch_on),
            'objective_name': self.objective_name,
            'positions_objective': getattr(self, 'positions_objective', ''),
            'travel_clearance_um': self.travel_clearance_um,
            'output_dir': self.output_dir,
            # Where everything is on the PLATE (mm from the A1 center),
            # so the configuration can be reused after the plate has been
            # taken off and put back: calibrate, and it all follows.
            'plate_rel': self._plate_anchored_snapshot_keep(),
        }

    def _plate_anchored_snapshot_keep(self):
        pending = getattr(self, '_pending_plate_rel', None)
        snap = self._plate_anchored_snapshot()
        self._pending_plate_rel = pending
        return snap

    def update_protocol(self):
        """Save the current state over the configuration that was loaded
        (or last saved), without asking for a new name."""
        if not self.protocol_loaded:
            self._set_status('Nothing to update: load or save a '
                             'configuration first')
            return
        self.save_protocol(self.protocol_loaded)

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
            self.protocol_loaded = '' if name.startswith('_') else name
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
            self.no_plate_on = bool(d.get('no_plate', False))
            if d.get('focus_mode') in experiment.FOCUS_MODES:
                self.wiz_focus_mode = d['focus_mode']
            if 'focus_moves' in d:
                self.focus_moves = dict(d.get('focus_moves') or {})
            self.af_every_n = max(int(d.get('af_every_n', 1)), 1)
            self.zplan.update(d.get('zplan') or {})
            self.zstack_optout = dict(d.get('zstack_optout') or {})
            self.compose_on = bool(d.get('compose_on', False))
            self.stitch_on = bool(d.get('stitch_on', False))
            self.positions_objective = (d.get('positions_objective')
                                        or d.get('objective_name') or '')
            self.travel_clearance_um = float(d.get('travel_clearance_um', 500.0))
            # The data folder is the user's own setting: a loaded
            # configuration (perhaps someone else's) does not change it.
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
            # Place everything by its plate coordinates: now if the plate
            # is already calibrated, otherwise when it is.
            self._pending_plate_rel = None
            snap = d.get('plate_rel')
            located = self._plate_located()
            if snap and not self.no_plate_on:
                if located:
                    self._plate_anchored_restore(snap)
                else:
                    self._pending_plate_rel = snap
            ids = self.root.ids
            ids.protocol_cycles_input.text = str(d.get('cycles', '1'))
            ids.protocol_interval_input.text = str(d.get('interval_min', '5'))
            self.protocol_name = 'protocol1' if name.startswith('_') else name
            self.protocol_loaded = '' if name.startswith('_') else name
            self._sync_channel_fields()
            self._sync_af_fields()
            self._refresh_surface_info()
            self._refresh_positions()
            self._persist()
            self._set_status(
                f'"{name}" loaded. ' + (
                    'Positions placed on the calibrated plate. '
                    if (snap and located) else
                    'Calibrate the plate next: the positions will follow '
                    'it. ') + 'Then use "Update focus" (wizard step 7).'
                + (f' NOTE: it was set up with {self.positions_objective}; '
                   f'the objective selected now is '
                   f'{self.objective_name or "not set"}.'
                   if self.positions_objective and
                   self.positions_objective != self.objective_name else ''))
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

    def on_plate_point_double_click(self, rel_x_mm: float, rel_y_mm: float,
                                    well=None):
        """Double-click on the zoomed-in plate navigator: move the stage
        to the exact point clicked (mm from the A1 center)."""
        where = (f'point in {self.current_labware.get_well_name(*well)}'
                 if well is not None else 'point on the plate')
        if well is not None:
            self.selected_well_str = self.current_labware.get_well_name(*well)
        if not self.stage_connected:
            self._set_status(f'{where}: stage not connected')
            return
        target = self.plate_rel_to_stage(rel_x_mm, rel_y_mm)
        if target is None:
            self._set_status('Calibrate the plate first (or "Set A1 = here")')
            return
        tx, ty = target
        if not acq.in_stage_bounds(tx, ty):
            self._set_status(f'{where}: outside the stage travel')
            return
        self._set_status(f'Moving to {where}...')
        def _move():
            self.hw.move_xy(tx, ty)
            if self.use_surface_on:
                z = self.surface.z_at(tx, ty)
                if z is not None:
                    self.hw.move_z(z)
            self._set_status(f'At {where}')
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
        """Blocking pixel->stage calibration from two probe moves.

        The probes are about an eighth of the field of view (large enough
        to measure precisely, small enough that the frames still overlap
        well). Drive backlash is taken up first and every measured move
        runs in the same direction, so lost motion does not end up in the
        measured scale."""
        D = float(min(max(0.12 * min(self._fov_um()), 50.0), 1000.0))
        B = XY_BACKLASH_UM
        self.hw.set_channel(ch, self._cs(ch))
        self.hw.move_xy_rel(-B, -B)
        self.hw.move_xy_rel(B, B)     # both axes now loaded in +
        time.sleep(0.3)
        f0 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(D, 0)
        time.sleep(0.2)
        f1 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(0, D)
        time.sleep(0.2)
        f2 = self.hw.grab_fresh(2.0)
        self.hw.move_xy_rel(-D - B, -D - B)   # back to the start,
        self.hw.move_xy_rel(B, B)             # arriving in + again
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
        self._lens_measured_at = time.time()
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
        center.

        The whole move is planned up front from the calibrated
        pixel->stage mapping and made as ONE straight vector move. The
        result is then checked once by finding the clicked spot itself in
        the new frame; if it is measurably off center, a single small trim
        follows. There is no further iteration, so the stage never hunts.
        The mapping is calibrated automatically on first use."""
        if not self.stage_connected or self.worker_busy:
            return
        if getattr(self, '_centering', False):
            return          # one centering move at a time
        self._centering = True
        ch = self.live_channel if self.live_channel in CHANNELS else self.selected_channel
        target_px = np.array([-dx_px, -dy_px])   # required content shift

        def _job():
            try:
                _center()
            finally:
                self._centering = False

        def _center():
            if self._img_cal_matrix() is None:
                self._set_status('First double-click: calibrating image '
                                 'motion (two short probe moves)...')
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
            f0 = self.hw.grab_fresh(1.0)
            self.hw.move_xy_rel(float(stage[0]), float(stage[1]))
            time.sleep(0.2)                      # let the stage settle
            f1 = self.hw.grab_fresh(1.0)
            note = ''
            if f0 is not None and f1 is not None and f0.ndim == 2:
                h, w = f0.shape
                cx, cy = w / 2.0, h / 2.0
                found = acq.locate_patch(f0, (cx + dx_px, cy + dy_px),
                                         f1, (cx, cy))
                if found is not None and found[2] >= 0.6:
                    resid = np.array([found[0] - cx, found[1] - cy])
                    err = float(np.linalg.norm(resid))
                    if 3.0 < err < 0.12 * min(h, w):
                        try:
                            trim = np.linalg.solve(A, -resid)
                            self.hw.move_xy_rel(float(trim[0]), float(trim[1]))
                            note = f' (trimmed {err:.0f} px)'
                        except np.linalg.LinAlgError:
                            pass
            self._set_status(f'Centered: moved dx={stage[0]:.0f}, '
                             f'dy={stage[1]:.0f} um{note}')
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
                                self.af_channel, z_um=self.stage_z_um)
        self._af_window_follow()
        self._persist()
        self._refresh_surface_info()
        ch = self.af_channel
        where = (f'AF point set at X {self.stage_x_um / 1000:.3f}, '
                 f'Y {self.stage_y_um / 1000:.3f} mm, Z '
                 f'{self.stage_z_um:.1f} um: autofocus searches '
                 f'{self.af_config.z_bottom_um:.0f} to '
                 f'{self.af_config.z_top_um:.0f} um')
        self._set_status(where)
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
                self._set_status(f'{where} (focus reference image '
                                 f'taken, {ch})')
            else:
                self._set_status(f'{where} (no reference image)')
        self._bg(_capture)

    def _anchor_af_cfg(self) -> AutofocusConfig:
        """The autofocus search at the AF point: the range and interval
        around the Z kept with the point (the Z it was set at, then the
        last focus found there), so the search always starts in range."""
        cfg = AutofocusConfig(z_bottom_um=self.af_config.z_bottom_um,
                              z_top_um=self.af_config.z_top_um,
                              step_um=self.af_config.step_um,
                              channel=self.surface.anchor_channel)
        z = self.surface.anchor_z
        return cfg if z is None else acq.af_window(z, cfg)

    def _af_window_follow(self):
        """Show the range where the next autofocus at the AF point will
        search (Bottom / Top follow the point's Z)."""
        z = self.surface.anchor_z
        if z is None:
            return
        w = acq.af_window(z, self.af_config)
        self.af_config.z_bottom_um = w.z_bottom_um
        self.af_config.z_top_um = w.z_top_um
        Clock.schedule_once(lambda dt: self._sync_af_fields(), 0)

    def af_goto_anchor(self):
        """Back to the AF point: X, Y and the Z kept with it."""
        if self.surface.anchor_xy is None:
            self._set_status('No AF anchor set')
            return
        x, y = self.surface.anchor_xy
        self._bg(self._safe_goto, x, y, self.surface.anchor_z)

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
            cfg = self._anchor_af_cfg()
            def _pos(z):
                Clock.schedule_once(lambda dt: setattr(self, 'stage_z_um', z), 0)
            best, ok = acq.autofocus_checked(
                self.hw, cfg, self._cs(ch), cancel=self.worker.cancel,
                status_cb=self._set_status, pos_cb=_pos,
                template=self.af_template,
                high_stability=self.high_stability_on, method=self.af_method)
            # Keep the AF channel lit so the image stream stays visible.
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', ch), 0)
            if best is None:
                self._set_status('Autofocus canceled')
                return
            if not ok:
                self._set_status(
                    f'AF: NO FOCUS FOUND between '
                    f'{cfg.z_bottom_um:.0f} and {cfg.z_top_um:.0f} um or '
                    f'just beyond (low contrast, or focus is outside the '
                    f'range) - focus surface NOT changed')
                return
            if self.surface.ref_z is None:
                self.surface.set_reference_z(best)
            else:
                self.surface.update_cycle(best)
            self._af_window_follow()
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
        self.quark_folder = out_dir
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
                              on_tile=self._mosaic_add,
                              save_format=self.save_format)
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    # ------------------- spiral overview on the plate map (no files)

    OVERVIEW_OVERLAP_PCT = 2.0     # fields just meet: nothing is stitched

    def toggle_overview_spiral(self):
        """Click-on / click-off, like the tile scan buttons."""
        if self.worker_busy:
            self.cancel_job()
            return
        self.run_overview_spiral()

    def run_overview_spiral(self):
        """Spiral outwards from here and leave every field on the plate
        map, all the imaged colours of a field in one picture. Nothing is
        saved: it is an overview of the sample to navigate by. Runs until
        STOP, the stage edges, or the map holds as many as it keeps."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('The spiral needs stage + camera')
            return
        plate = self._plate_map()
        if plate is None or self.no_plate_on or self._image_corners_rel(
                self.stage_x_um, self.stage_y_um, 10, 10) is None:
            self._set_status('The spiral overview is drawn on the plate '
                             'map: choose and calibrate the plate first')
            return
        channels = self._marked_channels()
        if not channels:
            self._set_status('No colour is imaged: press P under the '
                             'colours to scan (top of the left column)')
            return
        fov_x, fov_y = self._fov_um()
        cx, cy, z0 = self.stage_x_um, self.stage_y_um, self.stage_z_um
        positions = acq.spiral_positions_unbounded(
            cx, cy, fov_x, fov_y, overlap_pct=self.OVERVIEW_OVERLAP_PCT)
        surface = (self.surface if self.use_surface_on
                   and self.surface.current_z is not None else None)
        shading = self.shading if self.shading_on else None
        offsets = self.use_z_offsets_on
        settings = self.channel_settings
        limit = int(plate.MAX_SNAPSHOTS)
        cancel = self.worker.cancel
        said = ' + '.join(channels)

        def _leave(rgb, x, y, w, h):
            def _add(dt):
                corners = self._image_corners_rel(x, y, w, h)
                if corners:
                    plate.add_snapshot(rgb, corners)
            Clock.schedule_once(_add, 0)

        def _job():
            limits: dict[str, list[float]] = {}
            n = 0
            try:
                for (x, y) in positions:
                    if cancel.is_set() or n >= limit:
                        break
                    self._set_status(f'Spiral overview, field {n + 1} '
                                     f'({said}; nothing is saved). STOP '
                                     f'to end')
                    self.hw.move_xy(x, y)
                    base_z = surface.z_at(x, y) if surface is not None \
                        else None
                    planes = {}
                    for ch in channels:
                        if cancel.is_set():
                            break
                        cs = settings[ch]
                        z_now = None
                        if base_z is not None:
                            z_now = base_z + (cs.z_offset_um if offsets
                                              else 0.0)
                            self.hw.move_z(z_now)
                        self.hw.set_channel(ch, cs)
                        frame = self.hw.grab_settled(settle_s=0.15)
                        if frame is None:
                            continue
                        if shading is not None and shading.enabled:
                            if z_now is None and shading.has_stack(ch):
                                z_now = self.hw.get_pos()[2]
                            frame = shading.apply(frame, ch, z_now)
                        planes[ch] = frame
                    if len(planes) < len(channels):
                        continue      # stopped part way, or no frame
                    rgb = acq.overview_rgb(planes, CHANNEL_LUTS, limits,
                                           self.PREVIEW_PX)
                    if rgb is not None:
                        h, w = next(iter(planes.values())).shape[:2]
                        _leave(rgb, x, y, w, h)
                        n += 1
            finally:
                self.hw.leds_off()
                # Back to where it started, at the focus it started with.
                self.hw.move_xy(cx, cy)
                if surface is not None:
                    self.hw.move_z(z0)
                self._set_status(
                    f'Spiral overview: {n} fields ({said}) left on the '
                    f'plate map; no files were saved'
                    + (f'. Stopped at {limit}, the most the map keeps'
                       if n >= limit else ''))
                Clock.schedule_once(
                    lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    # ------------------------------------------- stitching with Quark

    def quark_stitch_folder(self, text: str = ''):
        """Stitch a finished tile scan or plate folder with Quark's stitching
        and shading correction (see quark_stitch.py): one OME-TIFF per well
        into ``<folder>/quark``. Touches no hardware; runs off the UI thread.
        The tiles themselves are never changed."""
        import quark_stitch
        folder = (text or '').strip().strip('"') or self.quark_folder
        if not folder:
            self._set_status('Stitch folder: type a tile scan or plate folder, or run a tile scan first')
            return
        if not os.path.isdir(folder):
            self._set_status(f'Stitch folder: {folder} is not a folder')
            return
        if not quark_stitch.available():
            self._set_status(quark_stitch.why_unavailable())
            return
        self.quark_folder = folder

        def _job():
            try:
                written = quark_stitch.export_folder(folder, status_cb=self._set_status)
                self._set_status(f'Quark: {len(written)} stitched file(s) written to '
                                 f'{os.path.join(folder, "quark")}')
            except Exception as e:
                self._set_status(f'Quark stitching failed: {e}')
        self._bg(_job)

    def _quark_folder_for(self, text: str, what: str):
        """The folder typed (or the last tile scan's) for a Quark tool, or None with the
        reason on the status line."""
        import quark_stitch
        folder = (text or '').strip().strip('"') or self.quark_folder
        if not folder:
            self._set_status(f'{what}: type a tile scan or plate folder, or run a tile scan first')
            return None
        if not os.path.isdir(folder):
            self._set_status(f'{what}: {folder} is not a folder')
            return None
        if not quark_stitch.available():
            self._set_status(quark_stitch.why_unavailable())
            return None
        self.quark_folder = folder
        return folder

    def quark_measure_shading(self, text: str = ''):
        """Measure every channel's shading from the tiles of a folder (Quark's algorithmic
        shading correction: no empty-field reference) and store it as that channel's shading
        reference. The Shading switch then applies it to the live view, to snapshots, tile
        scans and protocol runs. Touches no hardware; runs off the UI thread."""
        import quark_stitch
        folder = self._quark_folder_for(text, 'Shading from folder')
        if folder is None:
            return

        def _job():
            try:
                measured = quark_stitch.measure_shading(folder, status_cb=self._set_status)
                on_top = quark_stitch.shading_was_applied(folder)
                said = []
                for ch, (gain, info) in measured.items():
                    note = quark_stitch.describe_shading(info)
                    if gain is None:
                        said.append(f'{ch}: {note}')
                        continue
                    if ch not in acq.CHANNELS:
                        said.append(f'{ch}: not a channel of this microscope')
                        continue
                    # Tiles saved with the shading switch on hold only what the reference of
                    # the time left: the new map is that reference times what was measured.
                    old = self.shading.gain_at(ch) if on_top else None
                    if old is not None:
                        if old.shape != gain.shape and acq.cv2 is not None:
                            old = acq.cv2.resize(old, (gain.shape[1], gain.shape[0]),
                                                 interpolation=acq.cv2.INTER_LINEAR)
                        if old.shape == gain.shape:
                            gain = gain * old
                            note += ', on top of the reference in use when the tiles were taken'
                    self.shading.set_measured(ch, gain, note)
                    said.append(f'{ch}: {note}')
                Clock.schedule_once(lambda dt: self._refresh_shading_info(), 0)
                self._set_status('Shading ' + '; '.join(said)
                                 + ('' if self.shading_on else ' | switch Shading ON to use it'))
            except Exception as e:
                self._set_status(f'Shading from folder failed: {e}')
        self._bg(_job)

    def quark_correct_folder(self, text: str = ''):
        """Write a shading-corrected copy of every tile of a folder to
        ``<folder>/shading_corrected`` (Quark; the shading is measured from the tiles
        themselves, the tiles are never changed). Touches no hardware."""
        import quark_stitch
        folder = self._quark_folder_for(text, 'Correct tiles')
        if folder is None:
            return

        def _job():
            try:
                record = quark_stitch.correct_folder(folder, status_cb=self._set_status)
                said = [f"{ch}: {e['corrected']} tiles ({e['measured']})"
                        + (f", {len(e['skipped'])} skipped" if e['skipped'] else '')
                        for ch, e in record['channels'].items()]
                self._set_status('Shading-corrected tiles written to '
                                 f"{record['out_dir']} | " + '; '.join(said))
            except Exception as e:
                self._set_status(f'Shading correction of the folder failed: {e}')
        self._bg(_job)

    def quark_flatten_folder(self, text: str = ''):
        """Write a copy of every transmitted-light image of a folder with its own illumination
        (meniscus bright spot and rings, lamp hot spot, well-wall shadow) divided out to
        ``<folder>/field_flattened`` (Quark; each field is measured from itself, the originals
        are never changed). Touches no hardware."""
        import quark_stitch
        folder = self._quark_folder_for(text, 'Flatten fields')
        if folder is None:
            return

        def _job():
            try:
                record = quark_stitch.flatten_folder(folder, status_cb=self._set_status)
                done = list(record['corrected'].values())
                if not done:
                    self._set_status('No transmitted-light images (BF, PC, ...) were found in '
                                     f"{folder}; {len(record['skipped'])} skipped")
                    return
                before = sorted(i['spread_before'] for i in done)[len(done) // 2]
                after = sorted(i['spread_after'] for i in done)[len(done) // 2]
                self._set_status(f"{len(done)} fields flattened to {record['out_dir']} | brightness "
                                 f'spread over the field {before * 100:.0f} % -> {after * 100:.1f} % '
                                 f"(typical){', ' + str(len(record['skipped'])) + ' skipped' if record['skipped'] else ''}")
            except Exception as e:
                self._set_status(f'Flattening the folder failed: {e}')
        self._bg(_job)

    # ---------------------------------------------- live mosaic preview

    def _mosaic_reset(self):
        self._mosaic = None
        self._mosaic_meta = None

    def _mosaic_add(self, ch: str, frame, x_um: float, y_um: float):
        """Overlay a captured tile into the on-the-fly mosaic preview
        (shown in the viewport while the scan runs). Worker-thread safe:
        the canvas reference is swapped atomically."""
        self.leave_preview(ch, frame, x_um, y_um)   # and on the plate map
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

    # ------------------------------------------------ column dividers

    PANEL_DEFAULT_DP = {'left': 260.0, 'right': 320.0}
    IMAGE_MIN_DP = 240.0      # the image area is never squeezed below this

    def _panel_splits(self) -> dict:
        ids = self.root.ids if self.root else {}
        return {k: ids[f'{k}_split'] for k in ('left', 'right')
                if ids.get(f'{k}_split') is not None}

    def _limit_panel(self, split, width):
        """A column cannot be dragged over the image area: the two
        columns always leave it IMAGE_MIN_DP."""
        from kivy.metrics import dp
        if not self.root or self.root.width <= 0:
            return
        others = sum(s.width for s in self._panel_splits().values()
                     if s is not split)
        room = self.root.width - others - dp(self.IMAGE_MIN_DP)
        if width > room and room >= split.min_size:
            split.width = room

    def _apply_panel_widths(self):
        from kivy.metrics import dp
        for key, split in self._panel_splits().items():
            w = self.panel_widths.get(key)
            if w:
                split.width = min(max(dp(w), split.min_size), split.max_size)

    def remember_panel_widths(self):
        """Keep the column widths for this user (called when a divider
        is let go)."""
        from kivy.metrics import dp
        self.panel_widths = {k: round(s.width / dp(1), 1)
                             for k, s in self._panel_splits().items()}
        self._persist()

    def reset_panel_width(self, which):
        """Put a column ('left', 'right' or its splitter) back to its
        usual width."""
        from kivy.metrics import dp
        for key, split in self._panel_splits().items():
            if which == key or which is split:
                split.width = dp(self.PANEL_DEFAULT_DP[key])
                self.remember_panel_widths()
                self._set_status(f'{key.capitalize()} column back to its '
                                 f'usual width')

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
        self.remember_panel_widths()
        sections = data.get('sections', {})
        for c in self._collapsibles():
            if c.title and c.title in sections:
                c.open = bool(sections[c.title])
        self._set_status(f'Layout "{name}" loaded')

    # --------------- live z-stack: begin / end, or a centre and a range

    def set_channel_enabled(self, active: bool):
        """'Use in protocol' takes effect when it is ticked (it used to
        wait for the channel fields to be applied, and was lost when
        another channel was selected first)."""
        cs = self._cs()
        if cs.enabled == bool(active):
            return
        cs.enabled = bool(active)
        self._persist()
        self._refresh_channel_marks()
        self.zman_refresh()
        self._set_status(f'{self.selected_channel} '
                         + ('marked' if cs.enabled else 'not marked')
                         + ' for protocols and "Stack marked colours"')

    def _marked_channels(self) -> list[str]:
        return [ch for ch in CHANNELS if self.channel_settings[ch].enabled]

    def _zman_num(self, widget_id: str) -> float | None:
        try:
            return float(self.root.ids[widget_id].text.strip())
        except (ValueError, KeyError):
            return None

    def _zman_fields(self) -> tuple[float | None, float | None, float]:
        """(begin, end, interval) of the live z-stack, from whichever
        pair of controls is in use. A centre and a range runs upwards,
        from centre - below to centre + above."""
        step = self._zman_num('zman_step_input')
        step = abs(step) if step else 0.0
        if self.zman_mode == 'centre':
            centre = self._zman_num('zman_centre_input')
            if centre is None:
                return None, None, step
            above = abs(self._zman_num('zman_plus_input') or 0.0)
            below = abs(self._zman_num('zman_minus_input') or 0.0)
            return (max(centre - below, 0.0),
                    min(centre + above, self.Z_TRAVEL_UM), step)
        return (self._zman_num('zman_begin_input'),
                self._zman_num('zman_end_input'), step)

    def set_zman_mode(self, mode: str):
        """Set the stack by its two ends or by a centre and a range. The
        other pair of fields shows the same stack, so switching keeps it."""
        self.zman_mode = 'centre' if mode == 'centre' else 'ends'
        self._persist()
        self.zman_refresh()

    def set_zman_order(self, order: str):
        self.zman_order = 'channel' if order == 'channel' else 'slice'
        self._persist()
        self.zman_refresh()

    def zman_set(self, which: str):
        """Take the present focus position as the begin, end or centre."""
        self.root.ids[f'zman_{which}_input'].text = f'{self.stage_z_um:.1f}'
        self.zman_refresh()

    def zman_goto(self, which: str):
        z = self._zman_num(f'zman_{which}_input')
        if z is None or not self.stage_connected:
            return
        self._bg(self.hw.move_z, min(max(z, 0.0), self.Z_TRAVEL_UM))

    def _lens_section(self, channels: list[str]) -> tuple[float, str, float]:
        """(optical section um, the channel it is for, NA) for this lens:
        the thinnest section among the channels, so none is undersampled."""
        na = self._wiz_na()
        scale = self.effective_um_per_px()
        section, ch = min(
            (experiment.optical_section_um(
                na, experiment.CHANNEL_WAVELENGTH_NM.get(c, 550.0), 1.0,
                scale), c) for c in channels)
        return section, ch, na

    def zman_optimise(self, sampling: float = 1.0):
        """Set the interval from the optical section (depth of field) of
        the lens in use: 1 = one section per step, 2.3 = Nyquist."""
        section, ch, na = self._lens_section([self.selected_channel])
        step = experiment.stack_step_um(section, sampling)
        self.zman_step_text = f'{step:.2f}'
        self._persist()
        self.zman_refresh()
        self._set_status(
            f'Step {step:.2f} um: optical section {section:.2f} um for '
            f'{self.objective_name or "this lens"} (NA {na:g}, {ch}) at '
            f'{sampling:g}x sampling')

    def _zman_mirror(self, begin: float | None, end: float | None):
        """Show the stack in the pair of fields that is not in use."""
        if begin is None or end is None:
            return
        ids = self.root.ids
        if self.zman_mode == 'centre':
            shown = {'zman_begin_input': f'{begin:.1f}',
                     'zman_end_input': f'{end:.1f}'}
        else:
            half = abs(end - begin) / 2.0
            shown = {'zman_centre_input': f'{(begin + end) / 2.0:.1f}',
                     'zman_plus_input': f'{half:g}',
                     'zman_minus_input': f'{half:g}'}
        self._zman_sync = True
        try:
            for widget_id, text in shown.items():
                if ids[widget_id].text != text:
                    ids[widget_id].text = text
        finally:
            self._zman_sync = False

    def zman_refresh(self, *_):
        """Show what the z-stack controls amount to."""
        if getattr(self, '_zman_sync', False):
            return
        if not self.root or 'zman_centre_input' not in self.root.ids:
            return
        marked = self._marked_channels()
        self.zman_marked = ('Marked colours: ' + ', '.join(marked) if marked
                            else 'No colour is ticked "Use in protocol"')
        begin, end, step = self._zman_fields()
        self._zman_mirror(begin, end)
        section, ch, na = self._lens_section([self.selected_channel])
        lens = (f'Optical section {section:.2f} um ({ch}, NA {na:g}, '
                f'{self.objective_name or "no objective selected"}).')
        if begin is None or end is None:
            self.zman_info = (
                ('Focus to the middle of the sample and press "= here" on '
                 'Centre, then type how far above (+) and below (-) to go. '
                 if self.zman_mode == 'centre' else
                 'Focus to one end of the sample and press "= here" on '
                 'Begin, then the other end and "= here" on End. ') + lens)
            return
        if step <= 0:
            self.zman_info = 'Type an interval, or press Optimise. ' + lens
            return
        planes = experiment.stack_between(begin, end, step)
        real = abs(planes[1] - planes[0]) if len(planes) > 1 else 0.0
        self.zman_info = (
            f'{len(planes)} planes from {begin:.1f} to {end:.1f} um '
            f'({abs(end - begin):.1f} um), every {real:.2f} um. ' + lens
            + (' The interval is wider than the optical section: planes '
               'are skipped.' if real > section * 1.05 else ''))

    # Where live z-stacks are saved: their own folder, kept per user.

    def _zstack_base(self) -> str:
        base = (self.zstack_dir or '').strip()
        if base:
            try:
                os.makedirs(base, exist_ok=True)
                if acq.dir_writable(base):
                    return base
            except Exception:
                pass
            self._set_status(f'CANNOT WRITE TO {base}: this stack goes to '
                             f'the data directory instead')
        return self._out_base()

    def cycle_save_format(self):
        """The next way of saving protocol runs and z-stacks."""
        order = well_file.SAVE_FORMATS
        at = order.index(self.save_format) if self.save_format in order else -1
        self.save_format = order[(at + 1) % len(order)]
        self._set_status('Saving as ' + well_file.SAVE_FORMAT_LABEL[self.save_format])
        self._persist()

    def save_format_label(self, fmt: str) -> str:
        return well_file.SAVE_FORMAT_LABEL.get(fmt, fmt)

    def set_zstack_dir(self, text: str):
        self.zstack_dir = text.strip()
        self._persist()
        if not self.zstack_dir:
            self._set_status('Z-stacks are saved in the data directory: '
                             f'{self.output_dir}')
            return
        self._set_status(f'Z-stack directory: {self.zstack_dir}'
                         + self._dir_note(self.zstack_dir))

    def open_zstack_dir(self):
        try:
            os.startfile(self._zstack_base())
        except Exception as e:
            self._set_status(f'Open folder failed: {e}')

    def toggle_zstack(self, marked: bool = False):
        """Click-on / click-off: start the stack, or stop the one running."""
        if self.worker_busy:
            self.cancel_job()
            return
        self.run_zstack_between(marked)

    def run_zstack_between(self, marked: bool = False):
        """Z-stack here over the planes shown, for the selected colour or
        for exactly the colours ticked "Use in protocol"."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Z-stack needs stage + camera')
            return
        begin, end, step = self._zman_fields()
        if begin is None or end is None:
            self._set_status('Set the centre of the stack first'
                             if self.zman_mode == 'centre' else
                             'Set the begin and the end of the stack first')
            return
        if step <= 0:
            self._set_status('Type an interval, or press Optimise')
            return
        self.apply_channel_fields()
        channels = (self._marked_channels() if marked
                    else [self.selected_channel])
        if not channels:
            self._set_status('No colour is ticked "Use in protocol": tick '
                             'the colours to stack, or use "Stack this '
                             'colour"')
            return
        planes = experiment.stack_between(begin, end, step)
        out_dir = acq.timestamp_dir(self._zstack_base(), 'zstack')
        back = self.stage_z_um
        order = self.zman_order
        said = (f'Z-stack: {len(planes)} planes x {", ".join(channels)}, '
                + ('colour by colour' if order == 'channel'
                   else 'slice by slice') + f' -> {out_dir}')

        def _job():
            self._set_status(said)
            acq.run_z_planes(self.hw, self.channel_settings, channels,
                             planes, self.shading if self.shading_on else None,
                             out_dir, self.worker.cancel,
                             status_cb=self._set_status, return_z_um=back,
                             order=order, save_format=self.save_format,
                             um_per_px=self.effective_um_per_px())
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
        self._bg(_job, busy_job=True)

    def zman_to_range(self):
        """Use the begin / end span and the interval as this channel's
        centered z-stack (the one protocols run around each position's
        focus)."""
        begin, end, step = self._zman_fields()
        if begin is None or end is None or step <= 0:
            self._set_status('Set begin, end and interval first')
            return
        ids = self.root.ids
        ids.zstack_range_input.text = f'{abs(end - begin):g}'
        ids.zstack_step_input.text = f'{step:g}'
        ids.zstack_enabled_chk.active = True
        self.apply_channel_fields()
        self._set_status(
            f'{self.selected_channel}: protocols stack {abs(end - begin):g} '
            f'um around each position\'s focus, every {step:g} um')

    # ------------------------------------------------- timelapse protocol

    def _processing_done_popup(self, base_dir: str, summary: str,
                               videos: list[str]):
        """Say what the online processing made, and where the movies
        are (the status bar alone is easy to miss after a long run)."""
        try:
            from kivy.metrics import dp
            from kivy.uix.button import Button
            folder = os.path.join(base_dir, 'videos') if videos else base_dir
            text = f'{summary}\n\n'
            if videos:
                text += 'Movies:\n' + '\n'.join(
                    os.path.basename(v) for v in videos) + \
                    '\n\n(.mp4 for viewing, .avi opens in ImageJ/Fiji)\n\n'
            elif self.stitch_on or self.compose_on:
                text += ('No movie: a movie needs at least two completed '
                         'timepoints.\n\n')
            text += folder
            box = BoxLayout(orientation='vertical', spacing=dp(8),
                            padding=dp(10))
            msg = Label(text=text, halign='center', valign='middle')
            msg.bind(size=lambda w, s: setattr(w, 'text_size', s))
            row = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
            open_b = Button(text='Open the folder')
            ok = Button(text='OK')
            row.add_widget(open_b)
            row.add_widget(ok)
            box.add_widget(msg)
            box.add_widget(row)
            popup = Popup(title='Online processing finished', content=box,
                          size_hint=(None, None), size=(dp(560), dp(360)))

            def _open(*_):
                try:
                    os.startfile(folder)
                except Exception as e:
                    self._set_status(f'Open folder failed: {e}')
            open_b.bind(on_press=_open)
            ok.bind(on_press=popup.dismiss)
            popup.open()
        except Exception:
            pass

    def _lens_report(self) -> tuple[list[str], list[str]]:
        """What is known about the objective against the moves the
        protocol will make: (lines that are in order, problems)."""
        name = self.objective_name
        fov = self._fov_um()
        lines, problems = [], []
        if name:
            lines.append(f'Objective selected: {name}. Field of view '
                         f'{fov[0] / 1000:.3f} x {fov[1] / 1000:.3f} mm.')
        else:
            problems.append('No objective is selected.')
        laid = getattr(self, 'positions_objective', '')
        if self.protocol_positions and laid:
            if laid != name:
                problems.append(f'The positions were laid out for {laid}. '
                                f'Generate them again (wizard step 6).')
            else:
                lines.append('The positions were laid out for this lens.')
        verdict, text = experiment.lens_fit(
            experiment.tile_step_um(self.protocol_positions), fov)
        (problems if verdict in ('gaps', 'dense') else lines).append(text)
        measured = None
        A = self._img_cal_matrix()
        if A is not None:
            n = float(np.linalg.norm(A[:, 0]))
            measured = 1.0 / n if n > 0 else None
        verdict, text = experiment.lens_scale_check(
            measured, self.effective_um_per_px())
        if measured:
            at = getattr(self, '_lens_measured_at', None)
            text += (f' (measured at {time.strftime("%H:%M", time.localtime(at))})'
                     if at else ' (measured in an earlier session: press '
                                '"Measure the lens" to check it now)')
        (problems if verdict == 'wrong' else lines).append(text)
        return lines, problems

    def _objective_gate(self):
        """Last check before a run: the objective, and whether the stage
        moves between tiles fit its field of view. The run starts only
        from this window."""
        from kivy.metrics import dp
        from kivy.uix.button import Button
        box = BoxLayout(orientation='vertical', spacing=dp(8), padding=dp(10))
        msg = Label(halign='left', valign='top')
        msg.bind(size=lambda w, s: setattr(w, 'text_size', s))
        row = BoxLayout(size_hint_y=None, height=dp(38), spacing=dp(6))
        measure = Button(text='Measure the lens')
        cancel = Button(text='Cancel')
        run = Button(text='RUN')
        for b in (measure, cancel, run):
            row.add_widget(b)
        box.add_widget(msg)
        box.add_widget(row)
        popup = Popup(title='Check the objective before the run',
                      content=box, size_hint=(None, None),
                      size=(dp(600), dp(400)), auto_dismiss=False)

        def fill(note: str = ''):
            lines, problems = self._lens_report()
            msg.text = '\n\n'.join(
                lines + ['PROBLEM: ' + p for p in problems]
                + ([note] if note else [])
                + ['"Measure the lens" moves the stage about an eighth of '
                   'a field and back, and needs a lit field with something '
                   'in it.'])
            run.disabled = not self.objective_name
            run.text = ('Run anyway' if problems else
                        f'{self.objective_name} is fitted: RUN')
            run.background_color = (0.75, 0.2, 0.17, 1) if problems \
                else (0.1, 0.6, 0.35, 1)

        def do_measure(*_):
            if self.worker_busy:
                return
            ch = self.live_channel if self.live_channel in CHANNELS \
                else self.selected_channel
            measure.disabled = run.disabled = True
            msg.text = 'Measuring the image scale...'

            def _job():
                try:
                    ok = self._image_cal_sync(ch)
                except Exception as e:
                    ok = False
                    self._set_status(f'Lens measurement error: {e}')

                def _done(dt):
                    measure.disabled = False
                    fill('' if ok else 'MEASUREMENT FAILED: light a field '
                         'with visible features, focus, and try again.')
                Clock.schedule_once(_done, 0)
            self._bg(_job, busy_job=True)

        def do_run(*_):
            popup.dismiss()
            self.run_protocol(confirmed=True)

        measure.bind(on_press=do_measure)
        cancel.bind(on_press=popup.dismiss)
        run.bind(on_press=do_run)
        fill()
        popup.open()

    def run_protocol(self, confirmed: bool = False):
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
        if self.worker_busy:
            self._set_status('Another job is running')
            return
        if not confirmed:
            # The objective is checked here, at the last moment before
            # anything moves; the run starts from that window.
            self._objective_gate()
            return
        do_af = (self.af_each_cycle_on and
                 self.surface.anchor_xy is not None)
        af_every = max(int(self.af_every_n), 1)
        self.af_apply_fields()
        positions = [dict(p) for p in self.protocol_positions]
        if not positions:
            positions = [{'name': 'POS', 'x_um': self.stage_x_um,
                          'y_um': self.stage_y_um,
                          'z_um': self.stage_z_um - self.surface.drift_um}]
        # Every position gets a well (unless there is no plate definition
        # or the labware is a single dish/slide) and a tile/position
        # designation; both go into the file names.
        positions = experiment.assign_designations(
            positions,
            (lambda p: self._pos_well_name(p['x_um'], p['y_um']))
            if self._use_wells() else None)
        base_dir = acq.timestamp_dir(self._out_base(), 'protocol')
        try:
            import json
            manifest = self._protocol_payload(self.protocol_name)
            manifest['positions'] = positions
            manifest['file_name_order'] = ['well', 'tile_or_position',
                                           'z_plane', 'colour', 'timepoint']
            with open(os.path.join(base_dir, 'experiment.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass
        cancel = self.worker.cancel

        # Online compositing / stitching: told about every completed
        # position, it works on its own thread from the saved files.
        proc = None
        if self.compose_on or self.stitch_on:
            if online_processing.available():
                A = self._img_cal_matrix()
                scale = max(self.effective_um_per_px(), 1e-6)
                stacked = [self.channel_settings[ch] for ch in channels
                           if self.channel_settings[ch].zstack.enabled]
                proc = online_processing.OnlineProcessor(
                    base_dir, positions, channels,
                    compose=self.compose_on, stitch=self.stitch_on,
                    # A is the image shift per um of stage travel; a tile
                    # sits on the canvas the opposite way.
                    px_per_um=(-A if A is not None else np.eye(2) / scale),
                    um_per_px=scale,
                    z_step_um=stacked[0].zstack.step_um if stacked else 0.0,
                    cycles=cycles, status_cb=self._set_status,
                    done_cb=lambda summary, videos: (
                        # The planes kept for the processing are no longer
                        # needed: the well files are the data.
                        saver.remove_planes(),
                        Clock.schedule_once(
                        lambda dt: self._processing_done_popup(
                            base_dir, summary, videos), 0)))
            else:
                self._set_status('Online processing needs the tifffile and '
                                 'opencv-python packages - running without')
        t_run = time.time()
        stacked_all = [self.channel_settings[ch] for ch in channels
                       if self.channel_settings[ch].zstack.enabled]
        # One file per well, one for the whole run, or single-page TIFFs:
        # see well_file.RunSaver.
        saver = well_file.RunSaver(
            base_dir, self.save_format,
            um_per_px=self.effective_um_per_px(),
            z_step_um=stacked_all[0].zstack.step_um if stacked_all else 0.0,
            keep_planes=proc is not None,
            info={'protocol': self.protocol_name or '',
                  'objective': self.objective_name or ''},
            single_writer=acq.save_image,
            name_of=experiment.file_name)

        # What the run will take, for the progress strip: the images of
        # one timepoint, the wells in the order they are visited, and each
        # position's place among those of its well.
        use_offs = self.use_z_offsets_on
        cycle_total = 0
        for p in positions:
            opt = self.zstack_optout.get(p['well'] or '', ())
            for ch in channels:
                cycle_total += 1 if ('*' in opt or ch in opt) else len(
                    acq.zstack_planes(0.0, self.channel_settings[ch],
                                      use_offs))
        total = cycle_total * cycles
        well_order = list(dict.fromkeys(p['well'] or '' for p in positions))
        in_well: dict = {}
        site_no = []
        for p in positions:
            in_well[p['well'] or ''] = in_well.get(p['well'] or '', 0) + 1
            site_no.append(in_well[p['well'] or ''])
        fmt = self.save_format
        prog = {'done': 0, 'cyc_done': 0, 't0': time.time()}

        def _progress(cyc, note='', **where):
            done = prog['done']
            left = experiment.time_left_s(
                done, total, cyc, cycles, prog['cyc_done'], cycle_total,
                time.time() - prog['t0'], interval_s)
            if note:
                line = f'{100.0 * done / max(total, 1):.0f}%   |   {note}'
                if left is not None:
                    line += ('   |   about '
                             f'{experiment.duration_text(left)} left')
            else:
                line = experiment.progress_line(
                    done=done, total=total, cycle=cyc + 1, cycles=cycles,
                    left_s=left, **where)
            frac = min(done / max(total, 1), 1.0)

            def _show(dt):
                self.run_fraction = frac
                self.run_line = line
            Clock.schedule_once(_show, 0)

        def _job():
            Clock.schedule_once(lambda dt: (
                setattr(self, 'run_fraction', 0.0),
                setattr(self, 'run_line', f'0%   |   {total} images to take'),
                setattr(self, 'run_active', True)), 0)
            try:
                _run()
            finally:
                Clock.schedule_once(
                    lambda dt: setattr(self, 'run_active', False), 0)
                # The well files get their metadata now, whatever happened.
                try:
                    saver.close()
                except Exception as e:
                    self._set_status(f'Closing the well files failed: {e}')

        # The autofocus range is centered on the Z kept with the AF point,
        # which follows the focus found there: a slow drift over a long
        # run does not walk out of it.
        af_state = {'missed': 0}

        def _run():
            for cyc in range(cycles):
                t0 = time.time()
                if cancel.is_set():
                    break
                self._set_status(f'Protocol timepoint {cyc + 1}/{cycles}...')
                prog['cyc_done'] = 0
                prog['t0'] = t0

                # Autofocus at the one AF point (every af_every-th
                # timepoint). Its result moves the whole focus surface up
                # or down by the measured drift.
                if do_af and cyc % af_every == 0:
                    _progress(cyc, note=f'timepoint {cyc + 1} of {cycles}: '
                                        f'autofocus at the AF point')
                    ax, ay = self.surface.anchor_xy
                    self._safe_goto(ax, ay, None)
                    cfg = self._anchor_af_cfg()
                    best, af_ok = acq.autofocus_checked(
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
                        self._af_window_follow()
                        Clock.schedule_once(
                            lambda dt: self._refresh_surface_info(), 0)
                    elif best is not None:
                        # No focus found: this timepoint is imaged at the
                        # focus of the last good autofocus.
                        af_state['missed'] += 1
                    if cancel.is_set():
                        break

                drift = self.surface.drift_um
                for pi, p in enumerate(positions):
                    if cancel.is_set():
                        break
                    files: dict = {}
                    complete = True
                    base_z = None
                    if self.use_surface_on and self.surface.points and \
                            self.surface.current_z is not None:
                        base_z = self.surface.z_at(p['x_um'], p['y_um'])
                    if base_z is None:
                        base_z = p['z_um'] + drift
                    self._set_status(f'Timepoint {cyc + 1}/{cycles}: '
                                     f'{p["name"]}...')
                    self._safe_goto(p['x_um'], p['y_um'], base_z)
                    at_z = base_z
                    for ch in channels:
                        if cancel.is_set():
                            break
                        cs = self.channel_settings[ch]
                        # The channel's z-stack, centered on the focus
                        # surface (plus the chromatic offset) and taken
                        # from the top of the range down - or that one
                        # center plane where the well opted out.
                        opt = self.zstack_optout.get(p['well'] or '', ())
                        if '*' in opt or ch in opt:
                            planes = [base_z + (cs.z_offset_um if
                                                self.use_z_offsets_on else 0.0)]
                        else:
                            planes = acq.zstack_planes(
                                base_z, cs, self.use_z_offsets_on)[::-1]
                        self.hw.set_channel(ch, cs)
                        for zi, z in enumerate(planes, 1):
                            if cancel.is_set():
                                break
                            z = max(float(z), 0.0)
                            if abs(z - at_z) > 0.01:
                                self.hw.move_z(z)
                                at_z = z
                            frame = self.hw.grab_settled(
                                settle_s=0.2 if zi == 1 else 0.1)
                            if frame is None:
                                complete = False
                                continue
                            if self.shading_on:
                                frame = self.shading.apply(frame, ch, z)
                            path = saver.save(
                                frame, well=p['well'], site=p['site'],
                                z_index=zi, channel=ch, timepoint=cyc + 1,
                                x_um=p['x_um'], y_um=p['y_um'], z_um=z,
                                t_s=time.time() - t_run)
                            if path is not None:
                                files.setdefault(ch, []).append(path)
                            prog['done'] += 1
                            prog['cyc_done'] += 1
                            _progress(
                                cyc, well=p['well'],
                                well_i=well_order.index(p['well'] or '') + 1,
                                wells=len(well_order), site=p['site'],
                                site_i=site_no[pi],
                                sites=in_well[p['well'] or ''], channel=ch,
                                plane=zi, planes=len(planes),
                                file=os.path.basename(
                                    path if (path and fmt == 'single')
                                    else well_file.well_path(
                                        base_dir, p['well'], fmt)))
                            # The image stays on the plate map where it
                            # was taken (middle plane of a z-stack).
                            if zi == (len(planes) + 1) // 2:
                                self.leave_preview(ch, frame, p['x_um'],
                                                   p['y_um'])
                    self.hw.leds_off()
                    # All colours and planes of this position are on
                    # disk: it can be composited, and it counts toward
                    # its well's stitch.
                    if proc is not None and complete and not cancel.is_set():
                        proc.position_done(pi, cyc + 1, files, t0 - t_run)
                self._persist()

                if cyc < cycles - 1:
                    while time.time() - t0 < interval_s:
                        if cancel.is_set():
                            break
                        remaining = interval_s - (time.time() - t0)
                        self._set_status(
                            f'Protocol: cycle {cyc + 1}/{cycles} done, '
                            f'next in {remaining / 60:.1f} min')
                        _progress(cyc, note=(
                            f'timepoint {cyc + 1} of {cycles} done, next '
                            f'in {experiment.duration_text(remaining)}'))
                        time.sleep(min(2.0, max(remaining, 0.1)))
                    if cancel.is_set():
                        break
            Clock.schedule_once(lambda dt: setattr(self, 'live_channel', 'OFF'), 0)
            self._set_status(
                f'Protocol finished -> {base_dir}' + (
                    f'  |  AUTOFOCUS FOUND NO FOCUS at '
                    f'{af_state["missed"]} of {cycles} timepoints: those '
                    f'were imaged at the last good focus'
                    if af_state['missed'] else ''))
            if proc is not None:
                # Videos need every timepoint, so they are made now; the
                # processing thread reports when it is through.
                proc.finish()
        self._bg(_job, busy_job=True)

    # -------------------------------------------------------- persistence

    def _persist(self):
        payload = {
            'channel_settings': acq.channel_settings_to_dict(self.channel_settings),
            'a1_stage_um': self.a1_stage_um,
            'plate_sign_x': self.plate_sign_x,
            'plate_sign_y': self.plate_sign_y,
            'base_um_per_px': self.base_um_per_px,
            'objective_name': self.objective_name,
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
            'zstack_dir': self.zstack_dir,
            'save_format': self.save_format,
            'zman_mode': self.zman_mode,
            'zman_order': self.zman_order,
            'panel_widths': self.panel_widths,
            'zman_step': self.zman_step_text,
            'map_mirror': bool(self.map_flip_x),
            'lens_work_z': self.lens_work_z,
            'work_at_start_on': bool(self.work_at_start_on),
            'protocol_optin_v2': True,
            'no_plate_on': self.no_plate_on,
            'focus_mode': self.wiz_focus_mode,
            'focus_moves': self.focus_moves,
            'af_every_n': int(self.af_every_n),
            'zplan': self.zplan,
            'zstack_optout': self.zstack_optout,
            'compose_on': bool(self.compose_on),
            'stitch_on': bool(self.stitch_on),
        }
        try:
            acq.save_settings(payload)
            self._settings_save_failed = False
        except Exception as e:
            # Say so (once): settings that cannot be saved are the ones
            # that "revert" at the next launch.
            if not getattr(self, '_settings_save_failed', False):
                self._settings_save_failed = True
                self._set_status(f'SETTINGS NOT SAVED to {acq.SETTINGS_PATH} '
                                 f'({type(e).__name__}: {e})')
        self._autosave_protocol()

    def _load_persisted(self):
        d = acq.load_settings()
        if not d:
            self.shading.load_all()
            return
        # The user's folders first and on their own: a problem with any
        # other setting below must not put them back to the defaults.
        if isinstance(d.get('output_dir'), str) and d['output_dir'].strip():
            self.output_dir = d['output_dir']
        if isinstance(d.get('protocol_dir'), str) and d['protocol_dir'].strip():
            self.protocol_dir = d['protocol_dir']
        if isinstance(d.get('zstack_dir'), str):
            self.zstack_dir = d['zstack_dir'].strip()
        if d.get('save_format') in well_file.SAVE_FORMATS:
            self.save_format = d['save_format']
        self.zman_mode = ('centre' if d.get('zman_mode') == 'centre'
                          else 'ends')
        self.zman_order = ('channel' if d.get('zman_order') == 'channel'
                           else 'slice')
        try:
            self.zman_step_text = f"{abs(float(d.get('zman_step', 5))):g}"
        except (TypeError, ValueError):
            pass
        try:
            self.panel_widths = {
                k: float(v) for k, v in (d.get('panel_widths') or {}).items()
                if k in self.PANEL_DEFAULT_DP and float(v) > 0}
        except (TypeError, ValueError, AttributeError):
            self.panel_widths = {}
        # (a new key: the 'map_flip_x' saved while mirrored was the
        # default is not read back, so A1 returns to the top left)
        self.map_flip_x = bool(d.get('map_mirror', False))
        self.work_at_start_on = bool(d.get('work_at_start_on', True))
        try:
            self.lens_work_z = {
                str(k): min(max(float(v), 0.0), self.Z_TRAVEL_UM)
                for k, v in (d.get('lens_work_z') or {}).items()}
        except (TypeError, ValueError):
            self.lens_work_z = {}
        try:
            self.channel_settings = acq.channel_settings_from_dict(
                d.get('channel_settings', {}))
            a1 = d.get('a1_stage_um')
            self.a1_stage_um = tuple(a1) if a1 else None
            self.plate_sign_x = float(d.get('plate_sign_x', 1.0))
            self.plate_sign_y = float(d.get('plate_sign_y', 1.0))
            self.base_um_per_px = float(d.get('base_um_per_px', DEFAULT_UM_PER_PX))
            if d.get('objective_name') in self.objectives:
                self.objective_name = d['objective_name']
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
                # Re-solve from the stored marks, so a calibration made
                # by an earlier version gets the current (measured-well)
                # solution; one that cannot be right is not used.
                if pc.complete:
                    pc.solve()
                if pc.scale_ok():
                    self.plate_cal = pc
                else:
                    self._cal_log('stored calibration dropped at launch: '
                                  + '; '.join(t for _, t in
                                              pc.scale_problems())
                                  or 'row and column pitch do not agree')
            self.use_surface_on = bool(d.get('use_surface_on', True))
            self.use_z_offsets_on = bool(d.get('use_z_offsets_on', True))
            self.no_plate_on = bool(d.get('no_plate_on', False))
            if d.get('focus_mode') in experiment.FOCUS_MODES:
                self.wiz_focus_mode = d['focus_mode']
            self.focus_moves = dict(d.get('focus_moves') or {})
            self.af_every_n = max(int(d.get('af_every_n', 1)), 1)
            self.zplan.update(d.get('zplan') or {})
            self.af_each_cycle_on = bool(d.get('af_each_cycle_on', True))
            self.compose_on = bool(d.get('compose_on', False))
            self.stitch_on = bool(d.get('stitch_on', False))
            # Every launch starts with an empty plate: no selected wells,
            # positions, z-stack opt-outs or focus map from the last run.
            # (Instrument settings and the plate calibration are kept; a
            # saved configuration brings its own back when loaded.)
            self.protocol_positions = []
            self.zstack_optout = {}
            self.surface.clear_points()
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
            # One-time migration: protocol membership became opt-in; clear
            # any enables carried over from older settings files.
            if not d.get('protocol_optin_v2'):
                for cs in self.channel_settings.values():
                    cs.enabled = False
        except Exception as e:
            # Not silent: the rest of the settings after the failing one
            # keep their defaults for this session.
            try:
                import traceback
                os.makedirs(LOGS_DIR, exist_ok=True)
                with open(os.path.join(LOGS_DIR, 'errors.log'), 'a',
                          encoding='utf-8') as f:
                    f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  settings '
                            f'not fully loaded ({type(e).__name__}: {e})\n'
                            f'{traceback.format_exc()}\n')
            except Exception:
                pass
        self._keep_last_session()
        self.shading.load_all()

    def _keep_last_session(self):
        """The previous run's working state stays loadable as the
        configuration "_last_session" (the live autosave is about to be
        overwritten by this run's empty plate)."""
        try:
            import json
            import shutil
            src = os.path.join(self.protocol_dir, '_autosave.json')
            with open(src, 'r', encoding='utf-8') as f:
                if json.load(f).get('positions'):
                    shutil.copyfile(src, os.path.join(
                        self.protocol_dir, '_last_session.json'))
        except Exception:
            pass

    # ---------------------------------------------------------- fail safe

    def on_stop(self):
        """Tidy, fail-safe shutdown: cancel any acquisition, turn every LED
        off, park the stage (objective fully lowered, stage centered for
        lens access), then stop/disconnect the camera."""
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        try:
            self.worker.request_cancel()
            job = self.worker._thread
            if job is not None and job.is_alive():
                job.join(10.0)      # let it stop moving before we park
        except Exception:
            pass
        try:
            self.hw.leds_off()
        except Exception:
            pass
        parked = False
        if self.stage_connected:
            print('[LumaViewPro] Parking: lowering the objective and '
                  'centering the stage...', flush=True)
            try:
                parked = self.hw.park()
            except Exception:
                pass
            if parked:
                # Tell the park guard there is nothing left for it to do.
                try:
                    os.makedirs(LOGS_DIR, exist_ok=True)
                    with open(PARKED_FLAG, 'w', encoding='utf-8') as f:
                        f.write(str(os.getpid()))
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
        print('[LumaViewPro] Shutdown complete (LEDs off, camera stopped, '
              + ('objective lowered and stage centered' if parked
                 else 'stage NOT parked') + ').', flush=True)


if __name__ == '__main__':
    _app = LumaViewProApp()
    try:
        _app.run()
    finally:
        # However the main loop ended, leave the microscope safe. If even
        # this cannot run (process killed), the park guard does it.
        try:
            _app.on_stop()
        except Exception:
            pass
