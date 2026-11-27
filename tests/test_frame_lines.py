"""The number of lines the camera reads (ScopeHW.set_frame_lines): a strip
centred on the sensor, and back to the full frame; frames one after the
other without throwing any away (ScopeHW.grab_next)."""

import contextlib
import os
import queue
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import acquisition as acq  # noqa: E402

SENSOR = 2160


class Node:
    def __init__(self, cam, name):
        self.cam, self.name = cam, name

    def GetValue(self):
        return self.cam.v[self.name]

    def GetMin(self):
        return 0 if self.name == 'OffsetY' else 4

    def GetInc(self):
        return 2

    def GetMax(self):
        # As a Basler camera: the room left beside the other of the two.
        other = 'OffsetY' if self.name == 'Height' else 'Height'
        return SENSOR - self.cam.v[other]

    def SetValue(self, v):
        assert not self.cam.grabbing, 'changed while grabbing'
        assert self.GetMin() <= v <= self.GetMax(), (self.name, v)
        assert v % self.GetInc() == 0
        self.cam.v[self.name] = v


class Centre:
    def __init__(self, cam):
        self.cam = cam

    def Execute(self):
        self.cam.v['OffsetY'] = (SENSOR - self.cam.v['Height']) // 2 // 2 * 2


class Active:
    def __init__(self, centre=True):
        self.v = {'Height': 1900, 'OffsetY': 130}
        self.grabbing = True
        self.Height = Node(self, 'Height')
        self.OffsetY = Node(self, 'OffsetY')
        if centre:
            self.BslCenterY = Centre(self)


class Handler:
    def __init__(self):
        self._frame_queue = queue.Queue(maxsize=1)

    def reset(self):
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()


class Camera:
    def __init__(self, centre=True):
        self.active = Active(centre)
        self.cam_image_handler = Handler()
        self.array = np.zeros((4, 4), np.uint8)

    @contextlib.contextmanager
    def update_camera_config(self):
        self.active.grabbing = False
        yield
        self.active.grabbing = True

    def grab_new_capture(self, timeout):
        return False, None


def hw_with(cam):
    hw = acq.ScopeHW.__new__(acq.ScopeHW)
    hw.lock = threading.RLock()
    hw.camera = cam
    hw.soft_bin = 1
    return hw


def test_a_strip_in_the_middle_and_back_to_the_full_frame():
    cam = Camera()
    hw = hw_with(cam)
    assert hw.frame_lines() == 1900
    assert hw.set_frame_lines(200) == 200
    assert cam.active.v == {'Height': 200, 'OffsetY': 980}
    # The stock set_frame_size could not do this: with the strip's offset
    # the camera's upper limit is 1180 lines.
    assert hw.set_frame_lines(1900) == 1900
    assert cam.active.v == {'Height': 1900, 'OffsetY': 130}
    assert cam.active.grabbing


def test_what_the_camera_does_not_take_is_rounded():
    hw = hw_with(Camera())
    assert hw.set_frame_lines(201) == 200       # steps of 2
    assert hw.set_frame_lines(1) == 4           # its smallest
    assert hw.set_frame_lines(99999) == SENSOR  # its largest


def test_a_camera_without_the_centre_command_is_centred_by_its_offset():
    cam = Camera(centre=False)
    assert hw_with(cam).set_frame_lines(400) == 400
    assert cam.active.v['OffsetY'] == 880


def test_a_camera_that_has_no_height_says_so():
    class Mock:
        active = True
    hw = hw_with(Mock())
    assert hw.frame_lines() is None
    assert hw.set_frame_lines(200) is None
    assert hw.camera_frame_rate() is None


def test_grab_next_takes_the_frame_that_is_waiting():
    cam = Camera()
    hw = hw_with(cam)
    frame = np.full((8, 8), 7, np.uint8)
    cam.cam_image_handler._frame_queue.put((True, frame, None))
    got = hw.grab_next(0.1)
    assert np.array_equal(got, frame) and got is not frame
    assert hw.grab_next(0.05) is None           # nothing new: no frame
