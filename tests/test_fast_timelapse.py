"""Fast time-lapse of one channel: the timing from the start of a frame to
the start of the next, and the file it is written to (fast_timelapse)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tifffile = pytest.importorskip('tifffile')

import fast_timelapse  # noqa: E402


class Clock:
    """A clock that only moves when something sleeps or a frame is taken."""

    def __init__(self, frame_s: float):
        self.t = 100.0
        self.frame_s = frame_s
        self.sleeps = 0

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps += 1
        self.t += s

    def grab(self):
        self.t += self.frame_s
        return np.zeros((4, 4), np.uint8)


def run(frame_s, **kw):
    clock = Clock(frame_s)
    starts = []
    out = fast_timelapse.record(clock.grab, lambda f, n, t: starts.append(t),
                                clock=clock.now, sleep=clock.sleep, **kw)
    return out, starts, clock


def test_zero_cycle_is_full_speed():
    out, starts, clock = run(0.02, frames=50, cycle_s=0.0)
    assert out['frames'] == 50 and out['how'] == 'frames'
    assert clock.sleeps == 0
    assert np.allclose(np.diff(starts), 0.02)
    assert out['mean_cycle_s'] == pytest.approx(0.02)


def test_cycle_is_from_start_to_start():
    out, starts, _ = run(0.02, frames=20, cycle_s=0.5)
    # Not 0.5 s of pause after a 0.02 s frame: 0.5 s between starts.
    assert np.allclose(np.diff(starts), 0.5, atol=0.003)
    assert starts[0] == 0.0
    assert out['elapsed_s'] == pytest.approx(9.5, abs=0.05)


def test_a_cycle_shorter_than_a_frame_takes_runs_without_pauses():
    out, starts, clock = run(0.1, frames=30, cycle_s=0.03)
    assert clock.sleeps == 0
    assert np.allclose(np.diff(starts), 0.1)


def test_one_slow_frame_is_not_made_up_for_with_a_burst():
    clock = Clock(0.02)
    n = {'i': 0}

    def grab():
        n['i'] += 1
        clock.t += 1.5 if n['i'] == 3 else 0.02
        return np.zeros((4, 4), np.uint8)
    starts = []
    fast_timelapse.record(grab, lambda f, i, t: starts.append(t), frames=8,
                          cycle_s=0.5, clock=clock.now, sleep=clock.sleep)
    gaps = np.diff(starts)
    assert gaps[2] == pytest.approx(1.5, abs=0.01)      # the slow one
    assert np.allclose(np.delete(gaps, 2), 0.5, atol=0.003)


def test_duration_ends_it_and_frames_or_duration_whichever_first():
    out, starts, _ = run(0.02, duration_s=2.0, cycle_s=0.5)
    assert out['how'] == 'duration' and starts == pytest.approx(
        [0.0, 0.5, 1.0, 1.5], abs=0.01)
    out, starts, _ = run(0.02, frames=3, duration_s=60.0, cycle_s=0.5)
    assert out['how'] == 'frames' and len(starts) == 3
    out, starts, _ = run(0.02, frames=1000, duration_s=1.0, cycle_s=0.0)
    assert out['how'] == 'duration' and len(starts) in (50, 51)


def test_stop_ends_it_also_in_a_long_wait():
    clock = Clock(0.02)
    taken = []
    out = fast_timelapse.record(
        clock.grab, lambda f, n, t: taken.append(t), cycle_s=3600.0,
        should_stop=lambda: clock.t > 130.0, clock=clock.now, sleep=clock.sleep)
    assert out['how'] == 'stopped' and len(taken) == 1
    assert clock.t < 131.0


def test_a_camera_that_gives_nothing_ends_it():
    clock = Clock(0.02)
    out = fast_timelapse.record(lambda: None, lambda *a: None, frames=10,
                                clock=clock.now, sleep=clock.sleep)
    assert out == {'frames': 0, 'elapsed_s': 0.0, 'missed': 5,
                   'mean_cycle_s': 0.0, 'how': 'no frames from the camera'}


def test_the_recording_is_one_file_with_a_time_axis(tmp_path):
    path = str(tmp_path / 'fast_1' / 'GREEN.ome.tiff')
    rec = fast_timelapse.Recorder(path, 'GREEN', um_per_px=0.65, x_um=1.0,
                                  y_um=2.0, z_um=3.0, exposure_ms=5.0,
                                  describe={'asked_for': {'cycle_s': 0.0}})
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 4000, (64, 80), dtype=np.uint16) for _ in range(12)]
    result = fast_timelapse.record(iter(frames).__next__, rec.put, frames=12)
    doc = rec.close(result)
    assert doc['frames'] == 12 and rec.written == 12
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome and tf.series[0].axes == 'TYX'
        assert np.array_equal(tf.series[0].asarray(), np.stack(frames))
        assert not tf.pages[0].subifds          # no reduced copies: speed
    with open(tmp_path / 'fast_1' / 'timelapse.json', encoding='utf-8') as f:
        side = json.load(f)
    assert side['channel'] == 'GREEN' and side['frames'] == 12
    assert len(side['frame_start_s']) == 12 and side['frame_start_s'][0] == 0.0
    assert side['achieved']['how'] == 'frames'
    assert side['asked_for'] == {'cycle_s': 0.0}
    assert not os.path.exists(path + '.index.json')


def test_the_camera_waits_when_the_disk_is_behind(tmp_path):
    rec = fast_timelapse.Recorder(str(tmp_path / 'BF.ome.tiff'), 'BF',
                                  queue_bytes=3 * 64 * 64)
    for i in range(40):
        rec.put(np.full((64, 64), i, np.uint8), i, i * 0.01)
    rec.close()
    assert rec.written == 40
    data = tifffile.imread(tmp_path / 'BF.ome.tiff')
    assert [int(p[0, 0]) for p in data] == list(range(40))


def test_what_will_be_done_in_words():
    assert 'full speed' in fast_timelapse.describe_cycle(0)
    assert 'light off' not in fast_timelapse.describe_cycle(0.5)
    assert 'light off between frames' in fast_timelapse.describe_cycle(30)
