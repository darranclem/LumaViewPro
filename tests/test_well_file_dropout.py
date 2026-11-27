"""Storage that drops out while a run is saved (well_file.py): a write
that fails is tried again, an OME-TIFF left part-way is taken back to its
last whole frame, and nothing that was taken is lost or stored twice."""

import json
import os

import numpy as np
import pytest

import well_file

tifffile = pytest.importorskip('tifffile')

TILES, CHANS = ['TA1', 'TA2'], ['BF', 'GREEN']
PLANES = [(ti, t, ci, z) for t in range(2) for ti in range(2)
          for ci in range(2) for z in range(3)]


def frame(ti, t, ci, z, shape=(160, 200)):
    base = 1000 * ti + 100 * t + 10 * ci + z
    return (np.arange(shape[0] * shape[1]).reshape(shape) % 7 + base).astype(np.uint16)


def add_all(w, planes=PLANES):
    for ti, t, ci, z in planes:
        w.add(frame(ti, t, ci, z), tile=TILES[ti], z=z, channel=CHANS[ci],
              timepoint=t, x_um=100.0 * ti, y_um=5.0)


def check(path, planes=PLANES):
    got = well_file.read_well(path)
    for ti, t, ci, z in planes:
        assert np.array_equal(got['tiles'][TILES[ti]][t, ci, z], frame(ti, t, ci, z)), (ti, t, ci, z)
    assert len(well_file.read_index(path)['planes']) == len(planes)
    if os.path.isfile(path):
        # A proper TIFF for any other reader too: one page per frame, each
        # with its reduced copy, and the OME series.
        with tifffile.TiffFile(path) as tf:
            assert len(tf.pages) == len(planes)
            for (ti, t, ci, z), page in zip(planes, tf.pages):
                assert np.array_equal(page.asarray(), frame(ti, t, ci, z))
                assert len(page.pages) == 1
                assert np.array_equal(page.pages[0].asarray(),
                                      well_file.reduced(frame(ti, t, ci, z), 2))
            assert tf.is_ome and len(tf.series) == len({p[0] for p in planes})


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(well_file, '_sleep', lambda s: None)


def fail_writes(monkeypatch, at: set):
    """The n-th calls (counted from 1) of TiffWriter.write fail; an odd
    call is a frame, the even one after it its reduced copy, so an even
    number leaves a frame half written in the file."""
    real = tifffile.TiffWriter.write
    count = {'n': 0}

    def write(self, *a, **k):
        count['n'] += 1
        if count['n'] in at:
            if count['n'] % 2 == 0:
                # Something of the page is already in the file.
                self._fh.write(b'\x55' * 301)
            raise OSError(22, 'Invalid argument')
        return real(self, *a, **k)
    monkeypatch.setattr(tifffile.TiffWriter, 'write', write)
    return count


@pytest.mark.parametrize('at', [{1}, {2}, {2, 3, 4}, {9}, {10}, {10, 12, 13, 30}, {47, 48}])
def test_ome_tiff_carries_on_after_failed_writes(tmp_path, monkeypatch, at):
    path = str(tmp_path / 'B3.ome.tiff')
    said = []
    fail_writes(monkeypatch, at)
    with well_file.WellWriter(path, well='B3', on_wait=said.append) as w:
        add_all(w)
    assert said and 'not answering' in said[0]
    assert not os.path.exists(path + '.index.json')
    check(path)


def test_index_write_failing_does_not_store_the_frame_twice(tmp_path, monkeypatch):
    path = str(tmp_path / 'B3.ome.tiff')
    w = well_file.WellWriter(path)
    add_all(w, PLANES[:4])

    class Dead:
        def write(self, s):
            raise OSError(22, 'Invalid argument')

        def flush(self):
            raise OSError(22, 'Invalid argument')

        def close(self):
            raise OSError(22, 'Invalid argument')
    w._index = Dead()                       # as on 26 November: the flush failed
    add_all(w, PLANES[4:])
    with open(path + '.index.json', encoding='utf-8') as f:
        assert len(f.readlines()) == 1 + len(PLANES)
    w.close()
    check(path)


def test_zarr_carries_on(tmp_path, monkeypatch):
    path = str(tmp_path / 'B3.zarr')
    real, count = well_file.gzip.compress, {'n': 0}

    def compress(*a, **k):
        count['n'] += 1
        if count['n'] in (1, 7, 8):
            raise OSError(2, 'No such file or directory')
        return real(*a, **k)
    monkeypatch.setattr(well_file.gzip, 'compress', compress)
    with well_file.WellWriter(path, 'zarr') as w:
        add_all(w)
    check(path)


def test_patience_runs_out_and_what_was_taken_is_kept(tmp_path, monkeypatch):
    path = str(tmp_path / 'B3.ome.tiff')
    w = well_file.WellWriter(path, patience_s=0.0)
    add_all(w, PLANES[:6])
    count = fail_writes(monkeypatch, {2})
    with pytest.raises(OSError):
        add_all(w, PLANES[6:7])
    assert count['n'] == 2, 'no second try with no patience'
    monkeypatch.undo()
    # Unclosed: readable from the index, the half-written frame not in it.
    assert len(well_file.read_index(path)['planes']) == 6
    # The frame that failed may be given again, and the file finished.
    add_all(w, PLANES[6:])
    w.close()
    check(path)


def test_stop_ends_the_waiting(tmp_path, monkeypatch):
    stop = {'now': False}
    said = []

    def on_wait(text):
        said.append(text)
        stop['now'] = len(said) >= 3
    w = well_file.WellWriter(str(tmp_path / 'B3.ome.tiff'), on_wait=on_wait,
                             should_stop=lambda: stop['now'])
    add_all(w, PLANES[:2])
    monkeypatch.setattr(tifffile.TiffWriter, 'write',
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError(22, 'gone')))
    with pytest.raises(OSError):
        add_all(w, PLANES[2:3])
    assert len(said) == 3
    monkeypatch.undo()
    stop['now'] = False
    w.close()
    check(str(tmp_path / 'B3.ome.tiff'), PLANES[:2])


def test_storage_that_lost_data_is_said_so(tmp_path, monkeypatch):
    path = str(tmp_path / 'B3.ome.tiff')
    w = well_file.WellWriter(path)
    add_all(w, PLANES[:4])
    w._drop_handles()
    with open(path, 'r+b') as f:
        f.truncate(w._good[0] - 1000)
    w._broken = True
    with pytest.raises(well_file.WellFileDamaged):
        add_all(w, PLANES[4:5])
    # The frames the file still holds whole are read from the index.
    assert len(well_file.read_index(path)['planes']) == 3


def test_run_saver_finishes_every_file_it_can(tmp_path, monkeypatch):
    saver = well_file.RunSaver(str(tmp_path), 'ome-tiff', patience_s=0.0)
    for well in ('B2', 'B3', 'B4'):
        saver.save(frame(0, 0, 0, 0), well=well, site='TA1', z_index=1,
                   channel='BF', timepoint=1)
    real = tifffile.tiffcomment

    def comment(path, *a, **k):
        if os.path.basename(path).startswith('B2'):
            raise FileNotFoundError(2, 'No such file or directory', path)
        return real(path, *a, **k)
    monkeypatch.setattr(tifffile, 'tiffcomment', comment)
    with pytest.raises(OSError, match='1 of 3 files'):
        saver.close()
    monkeypatch.undo()
    names = sorted(os.listdir(tmp_path))
    assert names == ['B2.ome.tiff', 'B2.ome.tiff.index.json', 'B3.ome.tiff', 'B4.ome.tiff']
    # Closing again finishes the one that was left.
    assert len(saver.close()) == 3
    assert sorted(os.listdir(tmp_path)) == ['B2.ome.tiff', 'B3.ome.tiff', 'B4.ome.tiff']


def test_single_files_are_tried_again(tmp_path, monkeypatch):
    calls = {'n': 0}

    def writer(path, img):
        calls['n'] += 1
        if calls['n'] < 3:
            raise FileNotFoundError(2, 'No such file or directory', path)
        tifffile.imwrite(path, img)
    saver = well_file.RunSaver(str(tmp_path), 'single', single_writer=writer)
    path = saver.save(frame(0, 0, 0, 0), well='B2', site='TA1', z_index=1,
                      channel='BF', timepoint=1)
    assert calls['n'] == 3 and np.array_equal(tifffile.imread(path), frame(0, 0, 0, 0))


def test_finish_completes_a_file_that_was_never_closed(tmp_path):
    path = str(tmp_path / 'D7.ome.tiff')
    w = well_file.WellWriter(path, well='D7', um_per_px=0.65, info={'objective': '10x'})
    add_all(w)
    w._drop_handles()                       # the application ended here
    with open(path + '.index.json', 'a', encoding='utf-8') as f:
        f.write('{"page": 24, "tile": "TA1", "t": 2, "c"')    # a line cut off
    before = os.path.getsize(path)
    assert well_file.finish_folder(str(tmp_path)) == [path]
    assert not os.path.exists(path + '.index.json')
    assert os.path.getsize(path) > before
    check(path)
    doc = well_file.read_index(path)
    assert doc['well'] == 'D7' and doc['um_per_px'] == 0.65
    assert doc['images'][1]['x_um'] == 100.0
    assert not well_file.finish(path), 'nothing left to finish'
    assert json.dumps(doc)
