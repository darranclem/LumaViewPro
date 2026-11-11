"""Experiment setup wizard: a guided path through plate calibration,
autofocus, well and channel selection, sampling, focus mapping, z-stacks
and timelapse, ending in a saved protocol.

Mixed into the application class. The wizard owns no acquisition logic:
every step drives the same functions as the panels (calibration, AF,
focus surface, positions, protocol), so anything set up here can still be
inspected and adjusted there.
"""

from __future__ import annotations

import time

from kivy.clock import Clock
from kivy.properties import (BooleanProperty, NumericProperty,
                             StringProperty)

import acquisition as acq
import experiment
import plate_navigator
from acquisition import AutofocusConfig, CHANNELS

# (key, title, help)
WIZARD_STEPS = [
    ('plate', 'Plate',
     'Choose the labware in PLATE NAVIGATOR below. For a single dish or '
     'slide pick that labware. If there is no plate definition at all, '
     'switch "No plate" on: positions are then free points with no well '
     'names.'),
    ('calibrate', 'Calibrate plate',
     'Press Start. The stage goes to six rim positions in turn: A1 at 12, '
     '3, 6 and 9 o\'clock, the last well of the first row at 9 o\'clock '
     'and the last well of the first column at 12 o\'clock. Each time, '
     'jog until the well rim crosses the image center, then press MARK. '
     'The red target on the navigator shows which rim point is wanted. '
     'The result gives the A1 position, well size, well spacing and plate '
     'rotation. Next skips this step and keeps the current calibration.'),
    ('autofocus', 'Autofocus channel and point',
     '1. Pick the autofocus channel.  2. Drive to a spot with clear '
     'structure that will stay put.  3. Find focus (coarse) sweeps the '
     'whole range in big steps to find roughly where focus is.  4. Set '
     'the fine range and step used for every later autofocus.  5. Set AF '
     'point here, then Run autofocus to record the sharp position. Every '
     'other focus is stored relative to this result.'),
    ('wells', 'Wells',
     'On the navigator: click a well to switch it on or off, drag a box '
     'to switch a group (start on an on well to switch the group off). '
     'With "No plate" on, drive to each place and press Add position.'),
    ('channels', 'Colours',
     'Switch on every channel to acquire. Set exposure, gain and LED '
     'power per channel in CHANNEL SETTINGS on the left. For lenses with '
     'chromatic focus shift: focus with the autofocus channel and press '
     'Reference = here; then select another channel on the left, refocus '
     'and press Set offset. The offset is applied whenever that channel '
     'is imaged.'),
    ('sampling', 'What to image in each well',
     'Choose how each selected well is sampled, fill in the numbers for '
     'that choice and press Generate positions. Overlap applies to tiles '
     'and coverage. 100 % coverage overfills the well so all of it is '
     'imaged. Generating again replaces the previous positions.'),
    ('focusmap', 'Focus map',
     'Choose where focus is set, then press Start. Autofocus runs at the '
     'AF point; the stage then goes to each focus point in turn. Focus by '
     'eye (mouse wheel over the image) or press AF here, then Set focus. '
     'Each focus is stored as an offset from the autofocus result, and '
     'the surface between and beyond the points is interpolated.'),
    ('acquire', 'Z-stacks and timelapse',
     'Z-stack: the same range and step for every channel, centered on '
     'the focus surface. Timelapse: how many timepoints and how far '
     'apart. Autofocus re-runs at the one AF point (every timepoint, or '
     'every N timepoints); if focus has drifted, the whole focus surface '
     'moves up or down by that amount, so every position follows.'),
    ('save', 'Save and run',
     'Check the summary, name the protocol and save it. Files are named '
     'well_tile-or-position_Z_colour_timepoint, for example '
     'B3_TA1_Z001_GREEN_T0001.tiff.'),
]


class ExperimentWizardMixin:
    wiz_active = BooleanProperty(False)
    wiz_step = NumericProperty(0)
    wiz_key = StringProperty('plate')
    wiz_title = StringProperty('')
    wiz_help = StringProperty('')
    wiz_info = StringProperty('')
    wiz_sampling = StringProperty('single')
    wiz_focus_mode = StringProperty('corners')
    wiz_zstack_on = BooleanProperty(False)
    wiz_rev = NumericProperty(0)          # bumps to refresh channel buttons
    fmap_active = BooleanProperty(False)
    fmap_label = StringProperty('')
    no_plate_on = BooleanProperty(False)
    af_every_n = NumericProperty(1)       # autofocus every n-th timepoint

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

    def wiz_close(self):
        if self.fmap_active:
            self.wiz_fmap_cancel()
        self.wiz_active = False

    def _wiz_skipped(self, index: int) -> bool:
        return self.no_plate_on and WIZARD_STEPS[index][0] == 'calibrate'

    def wiz_next(self):
        self._wiz_leave()
        i = int(self.wiz_step) + 1
        while i < len(WIZARD_STEPS) and self._wiz_skipped(i):
            i += 1
        if i < len(WIZARD_STEPS):
            self._wiz_goto(i)

    def wiz_prev(self):
        self._wiz_leave()
        i = int(self.wiz_step) - 1
        while i >= 0 and self._wiz_skipped(i):
            i -= 1
        if i >= 0:
            self._wiz_goto(i)

    def _wiz_leave(self):
        if self.wiz_key == 'acquire':
            self.wiz_apply_acquire()
        if self.fmap_active:
            self.wiz_fmap_cancel()

    def _wiz_goto(self, index: int):
        key, title, text = WIZARD_STEPS[index]
        self.wiz_step = index
        self.wiz_key = key
        self.wiz_title = f'Step {index + 1} of {len(WIZARD_STEPS)}: {title}'
        self.wiz_help = text
        self.wiz_rev += 1
        self._wiz_refresh_info()
        if key == 'acquire':
            self._wiz_sync_acquire()

    def _wiz_refresh_info(self):
        key = self.wiz_key
        if key == 'plate':
            lw = self.current_labware
            self.wiz_info = ('No plate: free positions, no well names'
                             if self.no_plate_on else
                             f'{lw.name}: {lw.rows} x {lw.cols} wells')
        elif key == 'calibrate':
            self.wiz_info = self.cal_info
        elif key == 'autofocus':
            s = self.surface
            anchor = 'not set' if s.anchor_xy is None else 'set'
            ref = ('not run' if s.current_z is None
                   else f'z = {s.current_z:.1f} um')
            self.wiz_info = (f'Channel {self.af_channel}. AF point {anchor}. '
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
            self.wiz_info = f'{len(self.surface.points)} focus points stored'
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

    def wiz_set_af_channel(self, channel: str):
        if channel not in CHANNELS:
            return
        self.af_channel = channel
        self.af_config.channel = channel
        self.surface.anchor_channel = channel
        self.select_channel(channel)
        self._persist()
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

    def wiz_apply_fine(self):
        """Center the fine autofocus window on the current Z."""
        self._wiz_fine_window(self.stage_z_um)
        self._wiz_refresh_info()
        self._set_status(f'Autofocus range {self.af_config.z_bottom_um:.0f} '
                         f'to {self.af_config.z_top_um:.0f} um, step '
                         f'{self.af_config.step_um:g} um')

    def _wiz_sweep(self, cfg: AutofocusConfig):
        def _pos(z):
            Clock.schedule_once(lambda dt: setattr(self, 'stage_z_um', z), 0)
        ch = cfg.channel
        best, ok = acq.autofocus_sweep(
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
            best, ok = self._wiz_sweep(cfg)
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
        """The current Z is where the autofocus channel is sharp."""
        self._wiz_ref_z = (self.stage_z_um
                           - self.channel_settings[self.af_channel].z_offset_um)
        self._set_status(f'Reference focus ({self.af_channel}) = '
                         f'{self.stage_z_um:.1f} um. Now select another '
                         f'channel, refocus and press Set offset.')

    def wiz_set_offset_here(self):
        ref = getattr(self, '_wiz_ref_z', None)
        if ref is None:
            self._set_status('Press "Reference = here" with the autofocus '
                             'channel in focus first')
            return
        ch = self.selected_channel
        self.channel_settings[ch].z_offset_um = round(self.stage_z_um - ref, 2)
        self.use_z_offsets_on = True
        self._sync_channel_fields()
        self._persist()
        self._wiz_refresh_info()
        self._set_status(f'{ch} focus offset = '
                         f'{self.channel_settings[ch].z_offset_um:+.1f} um '
                         f'relative to {self.af_channel}')

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
        params = dict(
            well_x_mm=lw.well_size_x_mm, well_y_mm=lw.well_size_y_mm,
            shape=lw.shape, fov_x_mm=fov_x, fov_y_mm=fov_y,
            rows=int(self._f(ids.wiz_tile_rows.text, 3)),
            cols=int(self._f(ids.wiz_tile_cols.text, 3)),
            overlap_pct=self._f(ids.wiz_overlap.text, 10.0),
            n_random=int(self._f(ids.wiz_random_n.text, 3)),
            coverage_pct=self._f(ids.wiz_coverage.text, 100.0),
            seed=int(time.time()))
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
        self._persist()
        self._refresh_positions()
        note = f' ({skipped} outside the stage travel left out)' if skipped else ''
        self._set_status(f'{len(new)} positions generated in {groups}{note}')
        self._wiz_refresh_info()

    # -------------------------------------------------------- focus map

    def wiz_set_focus_mode(self, label: str):
        for mode, text in experiment.FOCUS_LABELS.items():
            if text == label:
                self.wiz_focus_mode = mode

    def _fmap_plan(self) -> list[dict]:
        lw = self.current_labware
        n = max(int(self._f(self.root.ids.wiz_fmap_n.text, 3)), 1)
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
        pts = experiment.focus_points(
            self.wiz_focus_mode, self._wiz_wells(),
            well_x_mm=lw.well_size_x_mm, well_y_mm=lw.well_size_y_mm,
            n_per_well=n)
        out = []
        count: dict[tuple[int, int], int] = {}
        for (row, col), dx, dy in pts:
            count[(row, col)] = count.get((row, col), 0) + 1
            rel = (col * (lw.spacing_x_mm or 0.0) + dx,
                   row * (lw.spacing_y_mm or 0.0) + dy)
            x, y = self.plate_rel_to_stage(*rel)
            name = lw.get_well_name(row, col)
            if self.wiz_focus_mode == 'multi' and n > 1:
                name += f' point {count[(row, col)]}'
            out.append({'label': name, 'x': x, 'y': y,
                        'plate_mm': (lw.a1_offset_x_mm + rel[0],
                                     lw.a1_offset_y_mm + rel[1])})
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
        self._fmap_points = plan
        self._fmap_done = {}
        self._fmap_index = 0
        cfg = AutofocusConfig(z_bottom_um=self.af_config.z_bottom_um,
                              z_top_um=self.af_config.z_top_um,
                              step_um=self.af_config.step_um,
                              channel=self.surface.anchor_channel)

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
                                 'focus - focus map not started')
                return
            if self.surface.ref_z is None:
                self.surface.set_reference_z(best)
            else:
                self.surface.update_cycle(best)

            def _begin(dt):
                self.fmap_active = True
                self._refresh_surface_info()
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
        if i in self._fmap_done:
            x, y, z = self._fmap_done[i]
        else:
            x, y = p['x'], p['y']
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

    def wiz_fmap_cancel(self):
        self._fmap_end()
        self._set_status('Focus mapping canceled - previous map kept')

    def wiz_fmap_finish(self):
        """Replace the focus surface with the points set in this pass."""
        if not self.fmap_active:
            return
        done = [self._fmap_done[i] for i in sorted(self._fmap_done)]
        if not done:
            self._set_status('No focus point set yet')
            return
        self.surface.clear_points()
        for x, y, z in done:
            self.surface.add_point(x, y, z)
        self.use_surface_on = True
        self._fmap_end()
        self._persist()
        self._refresh_surface_info()
        self._wiz_refresh_info()
        missed = len(self._fmap_points) - len(done)
        note = f' ({missed} skipped)' if missed else ''
        self._set_status(f'Focus map stored: {len(done)} points{note}')

    # ------------------------------------------- z-stacks and timelapse

    def _wiz_sync_acquire(self):
        ids = self.root.ids
        on = [cs for cs in self.channel_settings.values() if cs.enabled]
        stacked = [cs for cs in on if cs.zstack.enabled]
        self.wiz_zstack_on = bool(stacked)
        src = (stacked or on or list(self.channel_settings.values()))[0]
        ids.wiz_z_range.text = f'{src.zstack.range_um:g}'
        ids.wiz_z_step.text = f'{src.zstack.step_um:g}'
        ids.wiz_cycles.text = ids.protocol_cycles_input.text
        ids.wiz_interval.text = ids.protocol_interval_input.text
        ids.wiz_af_every.text = str(int(self.af_every_n))

    def wiz_apply_acquire(self):
        ids = self.root.ids
        rng = abs(self._f(ids.wiz_z_range.text, 50.0))
        step = max(abs(self._f(ids.wiz_z_step.text, 5.0)), 0.1)
        for cs in self.channel_settings.values():
            cs.zstack.enabled = bool(self.wiz_zstack_on)
            cs.zstack.range_um = rng
            cs.zstack.step_um = step
        cycles = max(int(self._f(ids.wiz_cycles.text, 1)), 1)
        interval = max(self._f(ids.wiz_interval.text, 5.0), 0.0)
        ids.protocol_cycles_input.text = str(cycles)
        ids.protocol_interval_input.text = f'{interval:g}'
        self.af_every_n = max(int(self._f(ids.wiz_af_every.text, 1)), 1)
        self._sync_channel_fields()
        self._persist()

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
        images = n_pos * len(chans) * planes * cycles
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
        return (f'Plate: {plate}{cal}\n'
                f'Positions: {n_pos}\n'
                f'Colours: {", ".join(chans) or "NONE SWITCHED ON"}\n'
                f'Z planes per colour: {planes}\n'
                f'Timepoints: {cycles}, every '
                f'{ids.protocol_interval_input.text} min\n'
                f'Autofocus: {self.af_channel}, AF point {af}, {every}; '
                f'{surf}\n'
                f'Images in total: {images}')

    def wiz_save(self):
        self.save_protocol(self.root.ids.wiz_name.text)
        self._wiz_refresh_info()

    def wiz_run(self):
        self.save_protocol(self.root.ids.wiz_name.text)
        self.run_protocol()
