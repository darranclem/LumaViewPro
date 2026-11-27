"""Fast time-lapse of one channel at the place the stage is: frames taken
one after the other with a set time from the START of one frame to the
START of the next.

- ``cycle_s`` 0: as fast as the camera gives frames.
- A cycle time shorter than a frame takes: the same, no pauses (the next
  frame starts as soon as the last one is in); it never tries to catch up
  with a burst.
- It ends after ``frames`` frames, after ``duration_s`` seconds (no frame
  is started later than that), whichever comes first, or when it is
  stopped; with neither it runs until it is stopped.

:func:`record` is the timing loop (no hardware: it is given ``grab()``).
:class:`Recorder` takes the frames and writes them on a thread of its own,
so that the disk does not set the pace: one OME-TIFF with a time axis
(``well_file.WellWriter``, no reduced copies), each frame with the time it
was started at, and ``timelapse.json`` beside it with what was asked for,
what was achieved and every frame's start time.

No hardware and no UI in here.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import well_file

# From this cycle time on the light is off between frames (the caller's
# business; kept here so that the window and the tests say the same).
LIGHT_OFF_FROM_S = 2.0
# Frames waiting to be written may take this much memory; beyond it the
# camera waits for the disk.
QUEUE_BYTES = 1024 * 2 ** 20


def record(grab, save, *, frames: int | None = None,
           duration_s: float | None = None, cycle_s: float = 0.0,
           should_stop=None, progress=None, max_missed: int = 5,
           clock=time.monotonic, sleep=time.sleep) -> dict:
    """Take frames with ``grab() -> frame or None`` and hand each to
    ``save(frame, number, t_s)`` (``number`` from 0, ``t_s`` the start of
    the frame in seconds after the start of the first).

    ``progress(taken, t_s, last_cycle_s)`` after every frame. Returns
    ``{'frames', 'elapsed_s', 'mean_cycle_s', 'missed', 'how'}``; ``how``
    is 'frames', 'duration', 'stopped' or 'no frames from the camera'."""
    frames = int(frames) if frames and frames > 0 else None
    duration_s = float(duration_s) if duration_s and duration_s > 0 else None
    cycle_s = max(float(cycle_s or 0.0), 0.0)
    stopped = should_stop or (lambda: False)
    taken = missed = in_a_row = 0
    t0 = None
    due = clock()                   # when the next frame is to start
    first = last = due
    how = 'stopped'
    while True:
        if stopped():
            break
        if frames is not None and taken >= frames:
            how = 'frames'
            break
        # Wait for the start of this frame (in short steps: Stop answers).
        while True:
            left = due - clock()
            if left <= 0 or stopped():
                break
            sleep(min(left, 0.05))
        if stopped():
            break
        start = clock()
        if t0 is None:
            t0 = start
        if duration_s is not None and start - t0 >= duration_s and taken:
            how = 'duration'
            break
        frame = grab()
        # The next start: one cycle after this one was due (a start that
        # was a little late does not shift the ones after it), or now when
        # that has already passed (a frame took longer than the cycle).
        due = max(due + cycle_s, clock()) if cycle_s > 0 else clock()
        if frame is None:
            missed += 1
            in_a_row += 1
            if in_a_row >= max_missed:
                how = 'no frames from the camera'
                break
            continue
        in_a_row = 0
        save(frame, taken, start - t0)
        if taken == 0:
            first = start
        cycle = start - last if taken else 0.0
        last = start
        taken += 1
        if progress is not None:
            progress(taken, start - t0, cycle)
    elapsed = (last - first) if taken else 0.0
    return {'frames': taken, 'elapsed_s': elapsed, 'missed': missed,
            'mean_cycle_s': elapsed / (taken - 1) if taken > 1 else 0.0,
            'how': how}


class Recorder:
    """Writes the frames of a fast time-lapse to ``path`` (an OME-TIFF) on
    its own thread. ``put(frame, number, t_s)`` is what :func:`record`
    saves with; ``close(result)`` waits for the frames still in memory,
    finishes the file and writes ``timelapse.json``."""

    def __init__(self, path: str, channel: str, *, um_per_px: float = 0.0,
                 x_um=None, y_um=None, z_um=None, exposure_ms=None,
                 info: dict | None = None, describe: dict | None = None,
                 queue_bytes: int = QUEUE_BYTES, should_stop=None):
        self.path = path
        self.channel = channel
        self.where = (x_um, y_um, z_um)
        self.exposure_ms = exposure_ms
        self.describe = dict(describe or {})
        self.queue_bytes = int(queue_bytes)
        self.times: list[float] = []
        self.written = 0
        self.waited_for_disk = 0    # frames the camera had to wait with
        self.error: BaseException | None = None
        self._q: deque = deque()
        self._held = 0
        self._cv = threading.Condition()
        self._done = False
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._writer = well_file.WellWriter(
            path, 'ome-tiff', um_per_px=um_per_px, levels=False,
            info=dict(info or {}, kind='fast time-lapse'),
            patience_s=20.0, should_stop=should_stop)
        self._thread = threading.Thread(target=self._write, daemon=True,
                                        name='fast-timelapse-writer')
        self._thread.start()

    @property
    def waiting(self) -> int:
        """Frames taken and not yet on the disk."""
        return len(self._q)

    def put(self, frame, number: int, t_s: float) -> None:
        plane = well_file.storable(frame)
        if plane is None:
            raise ValueError('a fast time-lapse takes grey frames only')
        with self._cv:
            if self.error is not None:
                raise self.error
            if self._held + plane.nbytes > self.queue_bytes and self._q:
                self.waited_for_disk += 1
                while self._held + plane.nbytes > self.queue_bytes and self._q \
                        and self.error is None:
                    self._cv.wait(0.1)
                if self.error is not None:
                    raise self.error
            self._q.append((plane, int(number), float(t_s)))
            self._held += plane.nbytes
            self.times.append(float(t_s))
            self._cv.notify_all()

    def _write(self) -> None:
        x, y, z = self.where
        while True:
            with self._cv:
                while not self._q and not self._done:
                    self._cv.wait(0.1)
                if not self._q:
                    return
                plane, number, t_s = self._q[0]
            try:
                self._writer.add(plane, tile='P1', z=0, channel=self.channel,
                                 timepoint=number, x_um=x, y_um=y, z_um=z,
                                 t_s=t_s, exposure_ms=self.exposure_ms)
            except BaseException as e:
                with self._cv:
                    self.error = e
                    self._q.clear()
                    self._held = 0
                    self._cv.notify_all()
                return
            with self._cv:
                self._q.popleft()
                self._held -= plane.nbytes
                self.written += 1
                self._cv.notify_all()

    def close(self, result: dict | None = None) -> dict:
        """Returns the description that was written to ``timelapse.json``.
        Raises what stopped the writing, after the file was finished with
        the frames that are in it."""
        with self._cv:
            self._done = True
            self._cv.notify_all()
        self._thread.join()
        summary = self._writer.close()
        times = self.times[:self.written]
        doc = dict(self.describe)
        doc.update({
            'lumaviewpro_fast_timelapse': 1,
            'file': os.path.basename(self.path),
            'channel': self.channel,
            'frames': self.written,
            'stage_x_um': self.where[0], 'stage_y_um': self.where[1],
            'stage_z_um': self.where[2],
            'exposure_ms': self.exposure_ms,
            'achieved': dict(result or {}),
            'frames_the_camera_waited_for_the_disk': self.waited_for_disk,
            'frame_start_s': [round(t, 4) for t in times],
            'axes': 'one OME image: time, y, x; frame_start_s is the start of '
                    'each frame in seconds after the start of the first',
        })
        if self.error is not None:
            doc['error'] = f'{type(self.error).__name__}: {self.error}'
        try:
            side = os.path.join(os.path.dirname(self.path), 'timelapse.json')
            with open(side + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=1)
            os.replace(side + '.tmp', side)
        except OSError:
            pass
        doc['summary'] = summary
        if self.error is not None:
            raise self.error
        return doc


def describe_cycle(cycle_s: float) -> str:
    """How a cycle time is taken, in words for the window."""
    if cycle_s <= 0:
        return 'full speed (no pause between frames)'
    text = f'a frame every {cycle_s:g} s, start to start'
    if cycle_s >= LIGHT_OFF_FROM_S:
        text += ', light off between frames'
    return text + ' (full speed if a frame takes longer)'
