"""Experiment setup wizard: a guided path through plate calibration,
autofocus, well and channel selection, sampling, focus mapping, z-stacks
and timelapse, ending in a saved protocol.

Mixed into the application class. The wizard owns no acquisition logic:
every step drives the same functions as the panels (calibration, AF,
focus surface, positions, protocol), so anything set up here can still be
inspected and adjusted there.
"""

from __future__ import annotations

import math
import time

from kivy.clock import Clock
from kivy.properties import (BooleanProperty, NumericProperty,
                             StringProperty)

import acquisition as acq
import experiment
import online_processing
import plate_navigator
from acquisition import AutofocusConfig, CHANNELS

# Basler daA3840-45um sensor pixel pitch (unbinned).
CAMERA_PIXEL_UM = 2.0

# (key, title, help)
WIZARD_STEPS = [
    ('plate', 'Choose the plate',
     'Choose the labware in PLATE NAVIGATOR below. For a single dish or '
     'slide pick that labware. If there is no plate definition at all, '
     'switch "No plate" on: positions are then free points with no well '
     'names.'),
    ('calibrate', 'Calibrate the plate position',
     'Press Start. The stage goes to six rim positions in turn: A1 at 12, '
     '3, 6 and 9 o\'clock, the last well of the first row at 9 o\'clock '
     'and the last well of the first column at 12 o\'clock. Each time, '
     'jog until the well rim crosses the image center, then press CONFIRM '
     'THIS POINT (or Enter). Only confirmed points count; when the sixth '
     'is confirmed the calibration finishes by itself. The navigator '
     'below zooms in on the well and shows the live image on it. '
     '"Next step" without starting keeps the current calibration.'),
    ('autofocus', 'Set up autofocus',
     '1. Pick the autofocus channel.  2. Drive to a spot with clear '
     'structure that will stay put.  3. Find focus (coarse) sweeps the '
     'whole range in big steps to find roughly where focus is.  4. Type '
     'the fine range and step used for every later autofocus.  5. Set AF '
     'point here: it keeps this X, Y and Z, and the range is searched '
     'around that Z.  6. Run autofocus to record the sharp position. '
     'Every other focus is stored relative to this result.'),
    ('wells', 'Choose the wells',
     'On the navigator: click a well to switch it on or off, drag a box '
     'to switch a group (start on an on well to switch the group off). '
     'With "No plate" on, drive to each place and press Add position.'),
    ('channels', 'Choose the colours',
     'Switch on every channel to acquire. Set exposure, gain and LED '
     'power per channel in CHANNEL SETTINGS on the left. For lenses with '
     'chromatic focus shift: focus with the autofocus channel and press '
     'Reference = here; then select another channel on the left, refocus '
     'and press Set offset. The offset is applied whenever that channel '
     'is imaged.'),
    ('sampling', 'Choose what to image in each well',
     'Choose how each selected well is sampled, fill in the numbers for '
     'that choice and press Generate positions. Overlap applies to tiles '
     'and coverage. 100 % coverage overfills the well so all of it is '
     'imaged. Generating again replaces the previous positions.'),
    ('focusmap', 'Set focus across the plate',
     'Choose where focus is set, then press Start. Autofocus runs at the '
     'AF point; the stage then goes to each focus point in turn. Focus by '
     'eye (mouse wheel over the image) or press AF here, then Set focus. '
     'Each focus is stored as an offset from the autofocus result, and '
     'the surface between and beyond the points is interpolated.'),
    ('acquire', 'Z-stack (optional)',
     'Switch "Z-stack everywhere" on to image several planes at every '
     'position, for every well and colour unless opted out below. '
     'Each stack is centered on the position\'s focus and runs from the '
     'top of the range down. The step is suggested from the lens (NA, '
     'wavelength and the image scale, which carries the magnification - '
     'pick the objective in the top bar): 1x sampling steps by one optical '
     'section, 2.3x is Nyquist, or type your own step. Leave it off for '
     'one plane per position.'),
    ('timelapse', 'Time-lapse (optional)',
     'Switch "Time-lapse" on to image everything again and again: type '
     'how many timepoints and how many minutes from the start of one to '
     'the start of the next. Left off, everything is imaged once. '
     'Autofocus can re-run at the one AF point (every timepoint, or '
     'every N timepoints); if focus has drifted, the whole focus surface '
     'moves up or down by that amount, so every position follows.'),
    ('save', 'Check, save and run',
     'Online compositing and stitching run while the experiment goes on: '
     'a position is composited when all its colours and planes are in, a '
     'well is stitched when all its tiles are in, and the time-lapse video '
     'is made after the last timepoint. The raw files are always kept. '
     'Check the summary, name the protocol and save it. Files are named '
     'well_tile-or-position_Z_colour_timepoint, for example '
     'B3_TA1_Z001_GREEN_t0001.tiff (T = tile, t = timepoint; Z001 is the '
     'top plane of a stack).'),
]


class ExperimentWizardMixin:
    wiz_active = BooleanProperty(False)
    wiz_step = NumericProperty(0)
    wiz_key = StringProperty('plate')
    wiz_title = StringProperty('')
    wiz_help = StringProperty('')
    wiz_info = StringProperty('')
    wiz_sampling = StringProperty('single')
    wiz_focus_mode = StringProperty('per_well')
    wiz_zstack_on = BooleanProperty(False)
    wiz_timelapse_on = BooleanProperty(False)
    wiz_timelapse_text = StringProperty('')
    wiz_rev = NumericProperty(0)          # bumps to refresh channel buttons
    fmap_active = BooleanProperty(False)
    fmap_label = StringProperty('')
    no_plate_on = BooleanProperty(False)
    af_every_n = NumericProperty(1)       # autofocus every n-th timepoint
    wiz_step_hint = StringProperty('')
    wiz_no_overlap = BooleanProperty(True)   # random fields must not overlap
    zmark_on = BooleanProperty(False)     # navigator marks z-stack opt-outs
    compose_on = BooleanProperty(False)   # online: channels into one file
    stitch_on = BooleanProperty(False)    # online: tiles into one image
    # Shading correction of a protocol run: 'off', 'auto' (measured from
    # the run's own images by Quark) or 'reference' (the empty-field
    # reference of each colour, applied as the frames are taken).
    run_shading = StringProperty('off')
    shading_labels = ['Off', 'Automatic (Quark)', 'Reference image']
    _SHADING_BY_LABEL = {'Off': 'off', 'Automatic (Quark)': 'auto',
                         'Reference image': 'reference'}
    zmark_scope = StringProperty('All channels')

    sampling_labels = [experiment.SAMPLING_LABELS[m]
                       for m in experiment.SAMPLING_MODES]
    focus_labels = [experiment.FOCUS_LABELS[m]
                    for m in experiment.FOCUS_MODES]

    def on_status_msg(self, *_):
        # Most steps finish through a status message (AF done, positions
        # generated...): keep the wizard's result line in step with it.
        if self.wiz_active and self.root:
            self._wiz_refresh_info()

    # ---------------------------------------------------------- stepping

    def wiz_start(self):
        self.wiz_active = True
        self._wiz_goto(0)
        # The plate map becomes the main view for the setup (the live
        # image is drawn on it); "camera view" under it switches back.
        if not self.no_plate_on:
            self.set_map_big(True)

    def wiz_open_tool(self, key: str):
        """Open one wizard step by itself, for manual work: its tools
        appear in the wizard panel without going through the steps before
        it, and the view stays as it is. Close ends it as usual."""
        index = next((i for i, s in enumerate(WIZARD_STEPS) if s[0] == key),
                     None)
        if index is None:
            return
        if self.wiz_active:
            self._wiz_leave()
        self.wiz_active = True
        self._wiz_goto(index)
        self._set_status(f'{WIZARD_STEPS[index][1]}: tools opened in the '
                         f'EXPERIMENT SETUP WIZARD panel (right)')

    def wiz_close(self):
        self._wiz_leave()
        self.wiz_active = False
        self.fmap_marks_refresh()
        self.set_map_big(False)

    def _wiz_skipped(self, index: int) -> bool:
        return self.no_plate_on and WIZARD_STEPS[index][0] == 'calibrate'

    def wiz_next(self):
        if self.wiz_key == 'plate' and not self.objective_name:
            self._set_status('Select the objective that is on the '
                             'microscope first: image size, tile spacing '
                             'and z steps all depend on it')
            return
        self._wiz_leave()
        i = int(self.wiz_step) + 1
        while i < len(WIZARD_STEPS) and self._wiz_skipped(i):
            i += 1
        if i < len(WIZARD_STEPS):
            self._wiz_goto(i)

    def wiz_jump(self, index: int):
        """Go straight to a step from the numbered strip under the title."""
        index = int(index)
        if not 0 <= index < len(WIZARD_STEPS) or index == int(self.wiz_step):
            return
        if self.cal_active:
            self._set_status('Finish or cancel the plate calibration first')
            return
        if self._wiz_skipped(index):
            self._set_status(f'{WIZARD_STEPS[index][1]}: not used with '
                             f'"No plate" on')
            return
        if self.wiz_key == 'plate' and not self.objective_name:
            self._set_status('Select the objective that is on the '
                             'microscope first: image size, tile spacing '
                             'and z steps all depend on it')
            return
        self._wiz_leave()
        self._wiz_goto(index)

    def wiz_step_name(self, index: int) -> str:
        """Title of a step, for the numbered strip."""
        return WIZARD_STEPS[int(index)][1] if \
            0 <= int(index) < len(WIZARD_STEPS) else ''

    def wiz_prev(self):
        self._wiz_leave()
        i = int(self.wiz_step) - 1
        while i >= 0 and self._wiz_skipped(i):
            i -= 1
        if i >= 0:
            self._wiz_goto(i)

    def _wiz_leave(self):
        if self.wiz_key in ('acquire', 'timelapse'):
            self.wiz_apply_acquire()
        if self.fmap_active:
            self.wiz_fmap_cancel()
        if self.pos_edit_mode == 'focus':
            self.set_pos_edit('focus')        # map clicks back to normal

    def _wiz_goto(self, index: int):
        key, title, text = WIZARD_STEPS[index]
        self.wiz_step = index
        self.wiz_key = key
        self.wiz_title = f'Step {index + 1} of {len(WIZARD_STEPS)}: {title}'
        self.wiz_help = text
        self.wiz_rev += 1
        self._wiz_refresh_info()
        self.fmap_marks_refresh()
        if key in ('acquire', 'timelapse'):
            self._wiz_sync_acquire()

    def _wiz_refresh_info(self):
        key = self.wiz_key
        if key == 'plate':
            lw = self.current_labware
            fov = self._fov_um()
            self.wiz_info = (
                ('No plate: free positions, no well names'
                 if self.no_plate_on else
                 f'{lw.name}: {lw.rows} x {lw.cols} wells')
                + (f'\nObjective {self.objective_name}: field of view '
                   f'{fov[0] / 1000:.2f} x {fov[1] / 1000:.2f} mm'
                   if self.objective_name else
                   '\nNO OBJECTIVE SELECTED'))
        elif key == 'calibrate':
            self.wiz_info = self.cal_info
        elif key == 'autofocus':
            s = self.surface
            anchor = 'not set' if s.anchor_xy is None else 'set'
            ref = ('not run' if s.current_z is None
                   else f'z = {s.current_z:.1f} um')
            if s.anchor_xy is not None and s.anchor_z is not None:
                anchor = (f'at X {s.anchor_xy[0] / 1000:.2f}, Y '
                          f'{s.anchor_xy[1] / 1000:.2f} mm, Z '
                          f'{s.anchor_z:.1f} um')
            self.wiz_info = (f'Autofocus channel: {self.af_channel_text()}'
                             f'{"" if self._af_cs().enabled else " (NOT in the protocol)"}'
                             f'. AF point {anchor}. '
                             f'Autofocus {ref}. Fine range '
                             f'{self.af_config.z_bottom_um:.0f} to '
                             f'{self.af_config.z_top_um:.0f} um, step '
                             f'{self.af_config.step_um:g} um.')
        elif key == 'wells':
            n = len(self.protocol_positions)
            self.wiz_info = (f'{n} position{"s" if n != 1 else ""}'
                             if not self._use_wells() else
                             f'{len(self._wiz_wells())} wells selected')
        elif key == 'channels':
            on = [ch for ch in CHANNELS if self.channel_settings[ch].enabled]
            offs = ', '.join(f'{ch} {self.channel_settings[ch].z_offset_um:+.1f}'
                             for ch in on)
            self.wiz_info = (f'Acquiring: {", ".join(on)}. Focus offsets '
                             f'(um): {offs}' if on else
                             'No channel switched on yet')
        elif key == 'sampling':
            self.wiz_info = self.positions_info
        elif key == 'focusmap':
            plan = self._focus_plan()
            moved = sum(1 for p in plan if p['moved'])
            self.wiz_info = (
                f'{len(plan)} focus point{"" if len(plan) == 1 else "s"} '
                f'planned' + (f' ({moved} moved by you)' if moved else '')
                + f'; {len(self.surface.points)} stored in the focus map')
        elif key == 'acquire':
            self.wiz_info = self._optout_text()
        elif key == 'timelapse':
            self.wiz_info = self.wiz_timelapse_text
        elif key == 'save':
            self.wiz_info = self.wiz_summary()

    # ------------------------------------------------------------ plate

    def wiz_toggle_no_plate(self):
        self.no_plate_on = not self.no_plate_on
        self._persist()
        self._wiz_refresh_info()

    def _use_wells(self) -> bool:
        """Positions carry a well name unless there is no plate definition
        or the labware is a single dish or slide."""
        lw = self.current_labware
        return not self.no_plate_on and lw.rows * lw.cols > 1

    # -------------------------------------------------------- autofocus

    def wiz_af_changed(self):
        """The Autofocus channel's light or binning was chosen here."""
        if self.root:
            self._wiz_refresh_info()

    def _wiz_fine_window(self, center_z: float):
        ids = self.root.ids
        half = abs(self._f(ids.wiz_fine_range.text, 100.0))
        step = max(abs(self._f(ids.wiz_fine_step.text, 5.0)), 0.1)
        self.af_config.z_bottom_um = max(center_z - half, 0.0)
        self.af_config.z_top_um = center_z + half
        self.af_config.step_um = step
        self._sync_af_fields()
        self._persist()

    def _wiz_af_center(self) -> float:
        """The Z kept with the AF point, or the current Z before one is
        set."""
        z = self.surface.anchor_z
        return self.stage_z_um if z is None else z

    def wiz_set_af_point(self):
        """The AF point = this X, Y and Z, with the range and interval
        typed here around that Z."""
        self._wiz_fine_window(self.stage_z_um)
        self.af_set_anchor_here()
        self._wiz_refresh_info()

    def wiz_run_af(self):
        """Autofocus at the AF point with the range and interval typed
        here, around the Z kept with the point."""
        self._wiz_fine_window(self._wiz_af_center())
        self.run_af_cycle()

    def wiz_apply_fine(self):
        """Use the range and interval typed here, around the Z kept with
        the AF point (the current Z before a point is set)."""
        self._wiz_fine_window(self._wiz_af_center())
        self._wiz_refresh_info()
        self._set_status(f'Autofocus range {self.af_config.z_bottom_um:.0f} '
                         f'to {self.af_config.z_top_um:.0f} um, step '
                         f'{self.af_config.step_um:g} um')

    def _wiz_sweep(self, cfg: AutofocusConfig, check: bool = True):
        """Autofocus over ``cfg``. ``check``: a focus at the edge of the
        range is not accepted (acq.autofocus_checked)."""
        def _pos(z):
            Clock.schedule_once(lambda dt: setattr(self, 'stage_z_um', z), 0)
        ch = cfg.channel
        best, ok = (acq.autofocus_checked if check else acq.autofocus_sweep)(
            self.hw, cfg, self._cs(ch), cancel=self.worker.cancel,
            status_cb=self._set_status, pos_cb=_pos,
            template=self.af_template,
            high_stability=self.high_stability_on, method=self.af_method)
        Clock.schedule_once(lambda dt: setattr(self, 'live_channel', ch), 0)
        return best, ok

    def wiz_find_focus(self):
        """Coarse sweep over the whole range at the current position, then
        center the fine autofocus window on what it found."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Autofocus needs stage + camera')
            return
        ids = self.root.ids
        lo = self._f(ids.wiz_coarse_bottom.text, 0.0)
        hi = self._f(ids.wiz_coarse_top.text, 14000.0)
        step = max(abs(self._f(ids.wiz_coarse_step.text, 100.0)), 1.0)
        cfg = AutofocusConfig(z_bottom_um=min(lo, hi), z_top_um=max(lo, hi),
                              step_um=step, channel=self.af_channel)

        def _job():
            best, ok = self._wiz_sweep(cfg, check=False)
            if best is None:
                self._set_status('Coarse focus search canceled')
                return
            if not ok:
                self._set_status('Coarse search found no clear focus - is '
                                 'there structure in view and is the '
                                 'exposure right?')
                return
            self.hw.move_z(best)

            def _done(dt):
                self._wiz_fine_window(best)
                self._wiz_refresh_info()
            Clock.schedule_once(_done, 0)
            self._set_status(f'Coarse focus found at z = {best:.1f} um; '
                             f'fine range centered on it')
        self._bg(_job, busy_job=True)

    def wiz_af_here(self):
        """Autofocus at the current position over the fine range centered
        on the current Z. Does not touch the AF point or the focus map."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Autofocus needs stage + camera')
            return
        half = abs(self.af_config.z_top_um - self.af_config.z_bottom_um) / 2.0
        z = self.stage_z_um
        cfg = AutofocusConfig(z_bottom_um=max(z - half, 0.0),
                              z_top_um=z + half,
                              step_um=self.af_config.step_um,
                              channel=self.af_channel)

        def _job():
            best, ok = self._wiz_sweep(cfg)
            if best is None:
                self._set_status('Autofocus canceled')
            elif ok:
                self.hw.move_z(best)
                self._set_status(f'Focus found at z = {best:.1f} um - adjust '
                                 f'if needed, then Set focus')
            else:
                self.hw.move_z(z)
                self._set_status('No clear focus here - focus by eye')
        self._bg(_job, busy_job=True)

    # ------------------------------------------------------------ wells

    def _wiz_wells(self) -> list[tuple[int, int]]:
        """Selected wells, in the order they first appear in the positions."""
        seen: dict[tuple[int, int], None] = {}
        for p in self.protocol_positions:
            w = self._pos_well(p['x_um'], p['y_um'])
            if w is not None:
                seen.setdefault(w)
        return list(seen)

    def wiz_select_all_wells(self):
        lw = self.current_labware
        self.on_wells_toggled([(r, c) for r in range(lw.rows)
                               for c in range(lw.cols)], True)
        self._wiz_refresh_info()

    def wiz_clear_wells(self):
        self.position_clear()
        self._wiz_centers = None
        self._wiz_refresh_info()

    def wiz_add_position(self):
        self.position_add_here()
        self._wiz_refresh_info()

    # ---------------------------------------------------------- colours

    def wiz_channel_on(self, channel: str, _rev=0) -> bool:
        return self.channel_settings[channel].enabled

    def wiz_toggle_channel(self, channel: str):
        cs = self.channel_settings[channel]
        cs.enabled = not cs.enabled
        self._sync_channel_fields()
        self._persist()
        self.wiz_rev += 1
        self._wiz_refresh_info()

    def wiz_offset_reference_here(self):
        """The current Z is where the Autofocus channel is sharp."""
        self._wiz_ref_z = self.stage_z_um
        self._set_status(f'Reference focus (Autofocus channel) = '
                         f'{self.stage_z_um:.1f} um. Now select another '
                         f'channel, refocus and press Set offset.')

    def wiz_set_offset_here(self):
        ref = getattr(self, '_wiz_ref_z', None)
        if ref is None:
            self._set_status('Press "Reference = here" with the autofocus '
                             'channel in focus first')
            return
        ch = self.selected_channel
        if ch not in CHANNELS:
            self._set_status('Select the colour to set the offset for: the '
                             'Autofocus channel is the reference itself')
            return
        self.channel_settings[ch].z_offset_um = round(self.stage_z_um - ref, 2)
        self.use_z_offsets_on = True
        self._sync_channel_fields()
        self._persist()
        self._wiz_refresh_info()
        self._set_status(f'{ch} focus offset = '
                         f'{self.channel_settings[ch].z_offset_um:+.1f} um '
                         f'relative to the Autofocus channel')

    # --------------------------------------------------------- sampling

    @staticmethod
    def sampling_text(mode: str) -> str:
        return experiment.SAMPLING_LABELS.get(mode, '')

    @staticmethod
    def focus_text(mode: str) -> str:
        return experiment.FOCUS_LABELS.get(mode, '')

    def wiz_set_sampling(self, label: str):
        for mode, text in experiment.SAMPLING_LABELS.items():
            if text == label:
                self.wiz_sampling = mode

    def _wiz_site_z(self, x_um: float, y_um: float) -> float:
        """Drift-normalized starting focus for a new position."""
        z = self.surface.z_at(x_um, y_um)
        if z is None:
            z = self.stage_z_um
        return z - self.surface.drift_um

    def wiz_generate(self):
        """Replace the protocol positions with the imaging sites for the
        chosen sampling of every selected well (or free position)."""
        ids = self.root.ids
        mode = self.wiz_sampling
        lw = self.current_labware
        fov_x, fov_y = (v / 1000.0 for v in self._fov_um())
        well_x, well_y = self.well_size_mm()
        params = dict(
            well_x_mm=well_x, well_y_mm=well_y,
            shape=lw.shape, fov_x_mm=fov_x, fov_y_mm=fov_y,
            rows=int(self._f(ids.wiz_tile_rows.text, 3)),
            cols=int(self._f(ids.wiz_tile_cols.text, 3)),
            overlap_pct=self._f(ids.wiz_overlap.text, 10.0),
            n_random=int(self._f(ids.wiz_random_n.text, 3)),
            coverage_pct=self._f(ids.wiz_coverage.text, 100.0),
            seed=int(time.time()), allow_overlap=not self.wiz_no_overlap)
        new, skipped = [], 0

        if self.no_plate_on:
            if mode not in ('single', 'tiles'):
                self._set_status('Without a plate only single images or a '
                                 'tile grid are possible (no well outline '
                                 'to fill)')
                return
            centers = [p for p in self.protocol_positions if not p.get('site')]
            centers = centers or getattr(self, '_wiz_centers', None) or []
            if not centers:
                self._set_status('Add at least one position first (step 4)')
                return
            self._wiz_centers = [dict(p) for p in centers]
            pattern = experiment.plan_sites(mode, [(0, 0)], **params)[(0, 0)]
            for i, c in enumerate(centers):
                for label, dx, dy in pattern:
                    site = (f'P{i + 1}' if mode == 'single' else
                            label if len(centers) == 1 else
                            f'R{i + 1}-{label}')
                    x, y = c['x_um'] + dx * 1000.0, c['y_um'] + dy * 1000.0
                    if not acq.in_stage_bounds(x, y):
                        skipped += 1
                        continue
                    new.append({'name': site, 'well': None, 'site': site,
                                'x_um': x, 'y_um': y, 'z_um': c['z_um']})
            groups = f'{len(centers)} place{"s" if len(centers) != 1 else ""}'
        else:
            wells = plate_navigator.serpentine(self._wiz_wells())
            if not wells:
                self._set_status('Select at least one well first (step 4)')
                return
            named = self._use_wells()
            sites = experiment.plan_sites(mode, wells, **params)
            for row, col in wells:
                well = lw.get_well_name(row, col)
                for label, dx, dy in sites[(row, col)]:
                    x, y = self.plate_rel_to_stage(
                        col * (lw.spacing_x_mm or 0.0) + dx,
                        row * (lw.spacing_y_mm or 0.0) + dy)
                    if not acq.in_stage_bounds(x, y):
                        skipped += 1
                        continue
                    new.append({
                        'name': f'{well}_{label}' if named else label,
                        'well': well if named else None, 'site': label,
                        'x_um': x, 'y_um': y,
                        'z_um': self._wiz_site_z(x, y)})
            groups = f'{len(wells)} well{"s" if len(wells) != 1 else ""}'

        if not new:
            self._set_status('No position is inside the stage travel')
            return
        self.protocol_positions = new
        self.positions_objective = self.objective_name
        self._persist()
        self._refresh_positions()
        note = f' ({skipped} outside the stage travel left out)' if skipped else ''
        if mode in ('random', 'random_same') and not self.no_plate_on:
            fewest = min(len(v) for v in sites.values())
            if fewest < params['n_random']:
                note += (f'. Only {fewest} of {params["n_random"]} random '
                         f'fields fit in a well without overlapping - ask '
                         f'for fewer, or switch "No overlap" off')
        self._set_status(f'{len(new)} positions generated in {groups}{note}')
        self._wiz_refresh_info()

    # -------------------------------------------------------- focus map

    def wiz_set_focus_mode(self, label: str):
        for mode, text in experiment.FOCUS_LABELS.items():
            if text == label and mode != self.wiz_focus_mode:
                self.wiz_focus_mode = mode
                self._persist()
        if self.root:
            self.fmap_marks_refresh()
            self._wiz_refresh_info()

    def _focus_plan(self) -> list[dict]:
        """The focus points for the wells in use (experiment.focus_plan),
        each where the user has put it; [] when there are no wells."""
        if self.no_plate_on or not self.root:
            return []
        wells = self._wiz_wells()
        if not wells:
            return []
        n = max(int(self._f(self.root.ids.wiz_fmap_n.text, 3)), 1)
        well_x, well_y = self.well_size_mm()
        # A point moved on another plate may not be inside this one's well.
        moved = {k: v for k, v in self.focus_moves.items()
                 if abs(v[0]) <= well_x * 0.55 and abs(v[1]) <= well_y * 0.55}
        return experiment.focus_plan(
            self.wiz_focus_mode, wells, well_x_mm=well_x, well_y_mm=well_y,
            n_per_well=n, well_name=self.current_labware.get_well_name,
            moved=moved)

    def _focus_point_place(self, p: dict):
        """(stage x, stage y, plate mm) of a point of the focus plan."""
        lw = self.current_labware
        rel = (p['well'][1] * (lw.spacing_x_mm or 0.0) + p['dx'],
               p['well'][0] * (lw.spacing_y_mm or 0.0) + p['dy'])
        x, y = self.plate_rel_to_stage(*rel)
        return x, y, (lw.a1_offset_x_mm + rel[0], lw.a1_offset_y_mm + rel[1])

    def fmap_marks_refresh(self):
        """Draw the focus points on the plate map while the focus map
        step is open, they are being moved, or the focus walk runs."""
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is None:
            return
        marks, highlight = [], None
        if self.fmap_active:
            for i, p in enumerate(self._fmap_points):
                if p.get('plate_mm') is None:
                    continue
                if i == self._fmap_index:
                    highlight = len(marks)
                marks.append((*p['plate_mm'], p['label'] in self.focus_moves))
        elif self.pos_edit_mode == 'focus' or (
                self.wiz_active and self.wiz_key == 'focusmap'):
            for p in self._focus_plan():
                marks.append((*self._focus_point_place(p)[2], p['moved']))
        plate.set_focus_markers(marks, highlight)

    def focus_point_move_to(self, x_um: float, y_um: float,
                            drive: bool = True):
        """Put the focus point of the well at (x_um, y_um) there (the
        nearest one, where a well has several). It stays there for every
        later focus map until the points are reset."""
        plan = self._focus_plan()
        if not plan:
            self._set_status('No focus points to move: select wells first, '
                             'and a focus map other than "No focus map"')
            return
        ok, _ = self._edit_target_well(x_um, y_um)
        if not ok:
            return
        rc = self._pos_well(x_um, y_um)
        rel = self.stage_to_plate_rel(x_um, y_um)
        if rc is None or rel is None:
            self._set_status('That point is not inside a well')
            return
        lw = self.current_labware
        dx = rel[0] - rc[1] * (lw.spacing_x_mm or 0.0)
        dy = rel[1] - rc[0] * (lw.spacing_y_mm or 0.0)
        i = experiment.nearest_focus_point(plan, rc, dx, dy)
        if i is None:
            self._set_status(
                f'{lw.get_well_name(*rc)} has no focus point with "'
                f'{experiment.FOCUS_LABELS[self.wiz_focus_mode]}": choose '
                f'one point in every well, or click a well that has one')
            return
        label = plan[i]['label']
        self.focus_moves[label] = [round(dx, 4), round(dy, 4)]
        self._persist()
        if self.fmap_active:
            for k, p in enumerate(self._fmap_points):
                if p['label'] == label and p.get('well') is not None:
                    p['x'], p['y'] = float(x_um), float(y_um)
                    p['plate_mm'] = (lw.a1_offset_x_mm + rel[0],
                                     lw.a1_offset_y_mm + rel[1])
                    self._fmap_done.pop(k, None)   # focus was set elsewhere
                    if k == self._fmap_index and drive:
                        self._fmap_goto()
        self.fmap_marks_refresh()
        self._wiz_refresh_info()
        self._set_status(f'Focus point {label} moved ({dx:+.2f}, {dy:+.2f} '
                         f'mm from the well center); it stays there until '
                         f'the points are reset')

    def wiz_fmap_point_here(self):
        """The focus point of the well the stage is in = where the stage
        is now."""
        self.focus_point_move_to(self.stage_x_um, self.stage_y_um,
                                 drive=False)

    def wiz_fmap_reset(self):
        """Every focus point back to where it is generated."""
        n = len(self.focus_moves)
        self.focus_moves = {}
        self._persist()
        self.fmap_marks_refresh()
        self._wiz_refresh_info()
        self._set_status(f'{n} moved focus point{"" if n == 1 else "s"} put '
                         f'back' if n else 'No focus point had been moved')

    def _fmap_plan(self) -> list[dict]:
        if self.no_plate_on or not self._wiz_wells():
            # No wells to hang points on: one focus point per position
            # (corners mode: the first, last and middle positions).
            pos = self.protocol_positions
            if self.wiz_focus_mode == 'corners' and len(pos) > 5:
                idx = sorted({0, len(pos) // 4, len(pos) // 2,
                              3 * len(pos) // 4, len(pos) - 1})
                pos = [pos[i] for i in idx]
            return [{'label': p['name'], 'x': p['x_um'], 'y': p['y_um'],
                     'plate_mm': None} for p in pos]
        out = []
        for p in self._focus_plan():
            x, y, plate_mm = self._focus_point_place(p)
            out.append({'label': p['label'], 'x': x, 'y': y,
                        'plate_mm': plate_mm, 'well': p['well']})
        return [p for p in out if acq.in_stage_bounds(p['x'], p['y'])]

    def wiz_fmap_start(self):
        """Autofocus at the AF point, then walk the focus points."""
        if self.wiz_focus_mode == 'none':
            self.surface.clear_points()
            self._persist()
            self._refresh_surface_info()
            self._wiz_refresh_info()
            self._set_status('Focus map cleared: every position uses the '
                             'autofocus result')
            return
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Focus mapping needs stage + camera')
            return
        if self.surface.anchor_xy is None:
            self._set_status('Set the AF point first (step 3)')
            return
        plan = self._fmap_plan()
        if not plan:
            self._set_status('No focus points: select wells or generate '
                             'positions first')
            return
        self._fmap_run(plan, update=False)

    def wiz_load(self, name: str):
        """Load a saved configuration from inside the wizard."""
        self.load_protocol(name)
        self._wiz_refresh_info()

    def wiz_focus_update(self):
        """Refresh the focus of the current (loaded) configuration without
        redefining it: autofocus at its AF point, then revisit each of its
        stored focus points at the expected focus for a touch-up."""
        if not (self.stage_connected and self.camera_connected):
            self._set_status('Updating focus needs stage + camera')
            return
        if self.surface.anchor_xy is None:
            self._set_status('This configuration has no AF point: set one '
                             'in step 3')
            return
        lw = self.current_labware
        plan = []
        for i, p in enumerate(self.surface.points):
            rel = self.stage_to_plate_rel(p[0], p[1])
            plan.append({
                'label': self._pos_well_name(p[0], p[1])
                or f'Focus point {i + 1}',
                'x': p[0], 'y': p[1],
                'plate_mm': None if rel is None else (
                    lw.a1_offset_x_mm + rel[0], lw.a1_offset_y_mm + rel[1])})
        self._fmap_run(plan, update=True)

    def _fmap_run(self, plan: list[dict], update: bool):
        """Autofocus at the AF point, then walk ``plan``. ``update``: the
        points are the stored focus map being refreshed (start each at
        its expected focus; a skipped point keeps its old value)."""
        self._fmap_points = plan
        self._fmap_update = update
        self._fmap_done = {}
        self._fmap_index = 0
        cfg = self._anchor_af_cfg()

        def _job():
            ax, ay = self.surface.anchor_xy
            self._set_status('Focus map: autofocus at the AF point...')
            self._safe_goto(ax, ay, None)
            best, ok = self._wiz_sweep(cfg)
            if best is None:
                self._set_status('Focus mapping canceled')
                return
            if not ok:
                self._set_status('Autofocus at the AF point found no clear '
                                 'focus inside its range - focus map not '
                                 'started. Focus there by eye, press "Set AF '
                                 'point here" in step 3 and try again.')
                return
            if self.surface.ref_z is None:
                self.surface.set_reference_z(best)
            else:
                self.surface.update_cycle(best)
            self._af_window_follow()

            def _begin(dt):
                self._refresh_surface_info()
                if not self._fmap_points:
                    self._persist()
                    self._set_status(
                        'AF point refreshed (this configuration has no '
                        'focus map points)' + self._update_hint())
                    return
                self.fmap_active = True
                self._fmap_goto()
            Clock.schedule_once(_begin, 0)
        self._bg(_job, busy_job=True)

    def _fmap_goto(self):
        i = self._fmap_index
        p = self._fmap_points[i]
        n = len(self._fmap_points)
        mark = ' (set)' if i in self._fmap_done else ''
        self.fmap_label = f'[{i + 1}/{n}] {p["label"]}{mark}'
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is not None:
            plate.set_target_plate_mm(p['plate_mm'])
        self.fmap_marks_refresh()
        if i in self._fmap_done:
            x, y, z = self._fmap_done[i]
        else:
            x, y = p['x'], p['y']
            if getattr(self, '_fmap_update', False):
                # The stored map, already shifted by the new autofocus.
                z = self.surface.z_at(x, y)
            else:
                z = experiment.interpolate_offsets(
                    list(self._fmap_done.values()), x, y)
            if z is None:
                z = self.surface.current_z
        self._set_status(f'Focus point {p["label"]}: focus, then Set focus')
        self._bg(self._safe_goto, x, y, z)

    def wiz_fmap_set(self):
        """Store the current XY and focus for the shown point, move on."""
        if not self.fmap_active:
            return
        self._fmap_done[self._fmap_index] = (
            self.stage_x_um, self.stage_y_um, self.stage_z_um)
        # Focus set somewhere else in the well than planned: that is where
        # this well's focus point is from now on.
        p = self._fmap_points[self._fmap_index]
        if p.get('well') is not None and math.hypot(
                self.stage_x_um - p['x'], self.stage_y_um - p['y']) > 25.0 \
                and self._pos_well(self.stage_x_um,
                                   self.stage_y_um) == tuple(p['well']):
            rel = self.stage_to_plate_rel(self.stage_x_um, self.stage_y_um)
            lw = self.current_labware
            self.focus_moves[p['label']] = [
                round(rel[0] - p['well'][1] * (lw.spacing_x_mm or 0.0), 4),
                round(rel[1] - p['well'][0] * (lw.spacing_y_mm or 0.0), 4)]
            p['x'], p['y'] = self.stage_x_um, self.stage_y_um
            p['plate_mm'] = (lw.a1_offset_x_mm + rel[0],
                             lw.a1_offset_y_mm + rel[1])
        if self._fmap_index < len(self._fmap_points) - 1:
            self._fmap_index += 1
            self._fmap_goto()
        else:
            self.wiz_fmap_finish()

    def wiz_fmap_step(self, delta: int):
        if not self.fmap_active:
            return
        i = self._fmap_index + delta
        if 0 <= i < len(self._fmap_points):
            self._fmap_index = i
            self._fmap_goto()

    def _fmap_end(self):
        self.fmap_active = False
        self.fmap_label = ''
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is not None:
            plate.set_target_plate_mm(None)
        self.fmap_marks_refresh()

    def wiz_fmap_cancel(self):
        self._fmap_end()
        self._set_status('Focus mapping canceled - previous map kept')

    def wiz_fmap_finish(self):
        """Replace the focus surface with the points set in this pass."""
        if not self.fmap_active:
            return
        done = [self._fmap_done[i] for i in sorted(self._fmap_done)]
        update = getattr(self, '_fmap_update', False)
        if not done and not update:
            self._set_status('No focus point set yet')
            return
        missed = len(self._fmap_points) - len(done)
        if update:
            # A point that was not revisited keeps its stored focus.
            done = []
            for i, p in enumerate(self._fmap_points):
                if i in self._fmap_done:
                    done.append(self._fmap_done[i])
                else:
                    z = self.surface.z_at(p['x'], p['y'])
                    if z is not None:
                        done.append((p['x'], p['y'], z))
        self.surface.clear_points()
        for x, y, z in done:
            self.surface.add_point(x, y, z)
        self.use_surface_on = True
        self._fmap_end()
        self._persist()
        self._refresh_surface_info()
        self._wiz_refresh_info()
        if update:
            note = f' ({missed} kept as stored)' if missed else ''
            self._set_status(f'Focus updated: {len(done)} points{note}'
                             + self._update_hint())
        else:
            note = f' ({missed} skipped)' if missed else ''
            self._set_status(f'Focus map stored: {len(done)} points{note}')

    def _update_hint(self) -> str:
        return (f'. Press Update "{self.protocol_loaded}" (step 9) to keep '
                f'it in the saved configuration.'
                if self.protocol_loaded else '')

    # ------------------------------------------- z-stacks and timelapse

    def _wiz_na(self) -> float:
        """Lens NA: as typed in, else the selected objective's, else a
        typical value for the magnification implied by the image scale."""
        return float(self.zplan.get('na') or self.objective_na()
                     or experiment.typical_na(
                         CAMERA_PIXEL_UM / max(self.base_um_per_px, 1e-6)))

    def _wiz_stack_channels(self) -> list[str]:
        off = self.zplan.get('channels_off') or []
        return [ch for ch in CHANNELS
                if self.channel_settings[ch].enabled and ch not in off]

    def wiz_stack_step(self) -> tuple[float, str]:
        """Z interval for the global z-stack and how it was arrived at:
        the lens's optical section (for the shortest wavelength being
        stacked, so no colour is undersampled) divided by the sampling
        rate, unless a step was typed in."""
        na = self._wiz_na()
        chans = self._wiz_stack_channels() or [self.af_source]
        # Camera pixel projected into the sample = the image scale, which
        # already contains the magnification (and the binning).
        scale = self.effective_um_per_px()
        section, ch = min(
            (experiment.optical_section_um(
                na, experiment.CHANNEL_WAVELENGTH_NM.get(c, 550.0), 1.0,
                scale), c) for c in chans)
        rate = self.zplan.get('rate') or 1.0
        override = self.zplan.get('override_um')
        step = experiment.stack_step_um(section, rate, override)
        text = (f'Optical section for NA {na:g} at {scale:g} um/px: '
                f'{section:.2f} um ({ch}). ')
        text += (f'Step {step:.2f} um (typed in).' if override else
                 f'Step {step:.2f} um at {rate:g}x sampling.')
        return step, text

    def wiz_update_step_hint(self):
        """Read the z-stack fields and show the resulting step."""
        if not self.root or getattr(self, '_wiz_syncing', False):
            return
        ids = self.root.ids
        if 'wiz_z_override' not in ids:
            return
        na = self._f(ids.wiz_z_na.text, 0.0)
        rate = self._f(ids.wiz_z_rate.text, 1.0)
        override = self._f(ids.wiz_z_override.text, 0.0)
        self.zplan.update({
            'na': na if na > 0 else None,
            'rate': rate if rate > 0 else 1.0,
            'override_um': override if override > 0 else None,
            'range_um': abs(self._f(ids.wiz_z_range.text, 50.0)),
        })
        step, text = self.wiz_stack_step()
        offs = experiment.stack_offsets(self.zplan['range_um'], step)
        self.wiz_step_hint = (f'{text} {len(offs)} planes from '
                              f'{offs[0]:+.1f} to {offs[-1]:+.1f} um around '
                              f'the focus point, top first.')

    def _wiz_sync_acquire(self):
        ids = self.root.ids
        colours = [self.channel_settings[ch] for ch in CHANNELS]
        on = [cs for cs in colours if cs.enabled]
        stacked = [cs for cs in on if cs.zstack.enabled]
        self.wiz_zstack_on = bool(self.zplan.get('on', bool(stacked)))
        src = (stacked or on or colours)[0]
        na = self._wiz_na()
        self._wiz_syncing = True
        ids.wiz_z_range.text = f"{self.zplan.get('range_um') or src.zstack.range_um:g}"
        ids.wiz_z_na.text = f'{na:g}'
        ids.wiz_z_rate.text = f"{self.zplan.get('rate') or 1.0:g}"
        ov = self.zplan.get('override_um')
        ids.wiz_z_override.text = f'{ov:g}' if ov else ''
        self._wiz_syncing = False
        cycles = max(int(self._f(ids.protocol_cycles_input.text, 1)), 1)
        self.wiz_timelapse_on = cycles > 1
        ids.wiz_cycles.text = str(cycles)
        ids.wiz_interval.text = ids.protocol_interval_input.text
        self.wiz_update_timelapse_text()
        ids.wiz_af_every.text = str(int(self.af_every_n))
        self.wiz_update_step_hint()
        self._refresh_flagged()

    def wiz_apply_acquire(self):
        """Write the global z-stack plan into every channel: stacked unless
        the channel was opted out, all with the same range and step."""
        ids = self.root.ids
        self.wiz_update_step_hint()
        step, _ = self.wiz_stack_step()
        self.zplan['on'] = bool(self.wiz_zstack_on)
        off = self.zplan.get('channels_off') or []
        for ch in CHANNELS:      # (the Autofocus channel is never imaged)
            cs = self.channel_settings[ch]
            cs.zstack.enabled = bool(self.wiz_zstack_on) and ch not in off
            cs.zstack.range_um = self.zplan['range_um']
            cs.zstack.step_um = round(step, 3)
        cycles = max(int(self._f(ids.wiz_cycles.text, 1)), 1) \
            if self.wiz_timelapse_on else 1
        interval = max(self._f(ids.wiz_interval.text, 5.0), 0.0)
        ids.protocol_cycles_input.text = str(cycles)
        ids.protocol_interval_input.text = f'{interval:g}'
        self.af_every_n = max(int(self._f(ids.wiz_af_every.text, 1)), 1)
        self._set_zmark(False)
        self._sync_channel_fields()
        self._persist()

    # --- time-lapse

    @staticmethod
    def timelapse_text(on: bool, cycles: int, interval_min: float) -> str:
        """What the time-lapse settings amount to, in words."""
        if not on or cycles <= 1:
            return 'Time-lapse off: everything is imaged once.'
        total = (cycles - 1) * max(interval_min, 0.0)
        h, m = int(total // 60), total % 60
        span = (f'{h} h {m:g} min' if h else f'{m:g} min')
        if interval_min <= 0:
            return (f'{cycles} timepoints, one straight after the other '
                    f'(no waiting).')
        return (f'{cycles} timepoints, one every {interval_min:g} min: the '
                f'last one starts {span} after the first. If one timepoint '
                f'takes longer than {interval_min:g} min, the next starts '
                f'as soon as it has finished.')

    def wiz_update_timelapse_text(self, *_):
        if not self.root:
            return
        ids = self.root.ids
        self.wiz_timelapse_text = self.timelapse_text(
            bool(self.wiz_timelapse_on),
            max(int(self._f(ids.wiz_cycles.text, 1)), 1),
            max(self._f(ids.wiz_interval.text, 5.0), 0.0))
        if self.wiz_key == 'timelapse':
            self.wiz_info = self.wiz_timelapse_text

    def wiz_toggle_timelapse(self):
        """Time-lapse on or off; switched on with one timepoint typed, it
        starts from two."""
        self.wiz_timelapse_on = not self.wiz_timelapse_on
        ids = self.root.ids
        if self.wiz_timelapse_on and \
                int(self._f(ids.wiz_cycles.text, 1)) < 2:
            ids.wiz_cycles.text = '2'
        self.wiz_update_timelapse_text()

    # --- opting out of the z-stack

    def wiz_stack_channel_on(self, channel: str, _rev=0) -> bool:
        return channel not in (self.zplan.get('channels_off') or [])

    def wiz_toggle_stack_channel(self, channel: str):
        off = set(self.zplan.get('channels_off') or [])
        off.symmetric_difference_update({channel})
        self.zplan['channels_off'] = sorted(off)
        self.wiz_rev += 1
        self.wiz_update_step_hint()

    def wiz_toggle_zmark(self):
        self._set_zmark(not self.zmark_on)

    def _set_zmark(self, on: bool):
        """While on, clicks and rectangles on the navigator mark wells as
        'no z-stack' instead of switching them in or out of the protocol."""
        self.zmark_on = on
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is not None:
            plate.on_wells_flag = self.on_wells_flagged
            plate.mark_mode = on
        if on:
            self._set_status('Click or drag wells on the map to take them '
                             'out of the z-stack; click a marked well to '
                             'put it back')

    def on_wells_flagged(self, wells: list[tuple[int, int]], flag_on: bool):
        lw = self.current_labware
        in_protocol = set(self._wiz_wells())
        scope = '*' if self.zmark_scope == 'All channels' else self.zmark_scope
        opt = dict(self.zstack_optout)
        for w in wells:
            if w not in in_protocol:
                continue
            name = lw.get_well_name(*w)
            if not flag_on:
                opt.pop(name, None)
            elif scope == '*':
                opt[name] = ['*']
            elif '*' not in opt.get(name, []):
                opt[name] = sorted(set(opt.get(name, [])) | {scope})
        self.zstack_optout = opt
        self._persist()
        self._refresh_flagged()
        self._wiz_refresh_info()

    def wiz_clear_optout(self):
        self.zstack_optout = {}
        self._persist()
        self._refresh_flagged()
        self._wiz_refresh_info()

    def _refresh_flagged(self):
        plate = self.root.ids.get('plate_map') if self.root else None
        if plate is None:
            return
        lw = self.current_labware
        plate.flagged_wells = {
            (r, c) for r in range(lw.rows) for c in range(lw.cols)
            if lw.get_well_name(r, c) in self.zstack_optout}
        plate.redraw()

    def _optout_text(self) -> str:
        if not self.zstack_optout:
            return 'Every selected well is stacked'
        parts = [f"{w} ({'all' if '*' in chs else ', '.join(chs)})"
                 for w, chs in sorted(self.zstack_optout.items())]
        if len(parts) > 12:
            parts = parts[:12] + [f'... {len(self.zstack_optout)} wells']
        return 'No z-stack in: ' + ', '.join(parts)

    def toggle_online(self, which: str):
        if which == 'compose':
            self.compose_on = not self.compose_on
        else:
            self.stitch_on = not self.stitch_on
        if (self.compose_on or self.stitch_on) and \
                not online_processing.available():
            self._set_status('Online processing needs the tifffile and '
                             'opencv-python packages in .venv')
        self._persist()
        if self.wiz_active:
            self._wiz_refresh_info()

    def run_shading_label(self, mode: str) -> str:
        return {v: k for k, v in self._SHADING_BY_LABEL.items()}.get(
            mode, 'Off')

    def set_run_shading(self, label: str):
        mode = self._SHADING_BY_LABEL.get(label, label)
        if mode not in online_processing.SHADING_MODES or \
                mode == self.run_shading:
            return       # (also the spinner showing a loaded setting)
        self.run_shading = mode
        if mode == 'auto':
            import quark_stitch
            if not quark_stitch.available():
                self._set_status('Automatic shading needs the Quark module: '
                                 + quark_stitch.why_unavailable())
            elif not online_processing.available():
                self._set_status('Automatic shading needs the tifffile and '
                                 'opencv-python packages in .venv')
            else:
                self._set_status(
                    'Shading: measured by Quark from the images of the run '
                    'after the first timepoint; corrected copy in '
                    'shading_corrected/, raw data unchanged')
        elif mode == 'reference':
            missing = self._shading_missing()
            self._set_status(
                'Shading: the reference image of each colour is applied as '
                'the frames are taken' + (
                    f'. NO REFERENCE for {", ".join(missing)} (Shading '
                    f'section): those are saved uncorrected'
                    if missing else ''))
        self._persist()
        if self.wiz_active:
            self._wiz_refresh_info()

    def _shading_missing(self) -> list[str]:
        chans = [ch for ch, cs in self.channel_settings.items()
                 if getattr(cs, 'enabled', False)]
        return [ch for ch in chans if not self.shading.has_reference(ch)]

    def _shading_text(self) -> str:
        if self.run_shading == 'auto':
            return ('automatic (measured by Quark from the first timepoint; '
                    'corrected copy in shading_corrected/)')
        if self.run_shading == 'reference':
            missing = self._shading_missing()
            return 'reference image' + (
                f' (NONE STORED for {", ".join(missing)})' if missing else '')
        return 'off'

    def _online_text(self, cycles: int) -> str:
        parts = []
        if self.compose_on:
            parts.append('compositing')
        if self.stitch_on:
            n = len(online_processing.stitch_groups(self.protocol_positions))
            parts.append(f'stitching ({n} tile scan{"s" if n != 1 else ""})'
                         if n else 'stitching (NO TILE SCANS in the '
                                   'positions)')
        if parts and cycles > 1:
            parts.append('timelapse video')
        return ', '.join(parts) or 'off'

    def _wiz_image_count(self, chans: list[str], cycles: int) -> int:
        named = self._use_wells()
        per_ch = {ch: len(acq.zstack_planes(0.0, self.channel_settings[ch]))
                  for ch in chans}
        total = 0
        for p in self.protocol_positions:
            well = p.get('well') if 'well' in p else (
                self._pos_well_name(p['x_um'], p['y_um']) if named else None)
            opt = self.zstack_optout.get(well or '', ())
            total += sum(1 if ('*' in opt or ch in opt) else per_ch[ch]
                         for ch in chans)
        return total * cycles

    # ------------------------------------------------------------- save

    def wiz_summary(self) -> str:
        ids = self.root.ids
        lw = self.current_labware
        chans = [ch for ch in CHANNELS if self.channel_settings[ch].enabled]
        n_pos = len(self.protocol_positions)
        planes = 1
        if chans:
            planes = max(len(acq.zstack_planes(0.0, self.channel_settings[ch]))
                         for ch in chans)
        cycles = max(int(self._f(ids.protocol_cycles_input.text, 1)), 1)
        images = self._wiz_image_count(chans, cycles)
        single = [ch for ch in chans
                  if not self.channel_settings[ch].zstack.enabled]
        stack = 'no z-stack'
        if planes > 1:
            cs = next(self.channel_settings[ch] for ch in chans
                      if self.channel_settings[ch].zstack.enabled)
            stack = (f'{planes} planes, {cs.zstack.range_um:g} um range, '
                     f'{cs.zstack.step_um:g} um step')
            if single:
                stack += f'; not {", ".join(single)}'
            if self.zstack_optout:
                stack += f'; {len(self.zstack_optout)} wells opted out'
        plate = ('no plate' if self.no_plate_on else
                 f'{lw.name}, {len(self._wiz_wells())} wells'
                 if self._use_wells() else lw.name)
        cal = ('' if self.no_plate_on or self.plate_cal is not None
               else '  NOT CALIBRATED')
        surf = (f'{len(self.surface.points)} focus points'
                if self.surface.points else 'no focus map')
        af = 'set' if self.surface.anchor_xy is not None else 'NOT SET'
        n = int(self.af_every_n)
        every = ('never re-run' if not self.af_each_cycle_on else
                 'every timepoint' if n <= 1 else f'every {n} timepoints')
        laid = getattr(self, 'positions_objective', '')
        lens = self.objective_name or 'NOT SELECTED'
        if n_pos and laid and laid != self.objective_name:
            lens += (f'  - POSITIONS WERE LAID OUT FOR {laid}: generate '
                     f'them again (step 6)')
        return (f'Objective: {lens}\n'
                f'Plate: {plate}{cal}\n'
                f'Positions: {n_pos}\n'
                f'Colours: {", ".join(chans) or "NONE SWITCHED ON"}\n'
                f'Z-stack: {stack}\n'
                f'Time-lapse: ' + (
                    f'{cycles} timepoints, one every '
                    f'{ids.protocol_interval_input.text} min'
                    if cycles > 1 else 'off (imaged once)') + '\n'
                f'Autofocus: {self.af_channel_text()}, AF point {af}, '
                f'{every}; '
                f'{surf}\n'
                f'Online processing: {self._online_text(cycles)}\n'
                f'Shading correction: {self._shading_text()}\n'
                f'Images in total: {images}')

    def wiz_save(self):
        self.save_protocol(self.root.ids.wiz_name.text)
        self._wiz_refresh_info()

    def wiz_run(self):
        self.save_protocol(self.root.ids.wiz_name.text)
        self.run_protocol()
