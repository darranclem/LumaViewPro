"""One file per position and timepoint (well_file 'position' format) and the
copying of a local run folder to where it is kept (run_mirror)."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tifffile = pytest.importorskip('tifffile')

import run_mirror  # noqa: E402
import well_file  # noqa: E402


def frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, (128, 160), dtype=np.uint16)


def take(saver, well, site, timepoint, channels=('BF', 'GREEN'), planes=3):
    n = 0
    for ci, ch in enumerate(channels):
        for z in range(1, planes + 1):
            saver.save(frame(1000 * timepoint + 10 * ci + z), well=well, site=site,
                       z_index=z, channel=ch, timepoint=timepoint,
                       x_um=1000.0, y_um=2000.0, z_um=5000.0 + z,
                       t_s=60.0 * timepoint, exposure_ms=12.5)
            n += 1
    saver.end_position()
    return n


# ------------------------------------------------------------ the files

def test_one_file_per_position_and_timepoint(tmp_path):
    saver = well_file.RunSaver(
        str(tmp_path), 'position', um_per_px=0.65, z_step_um=1.0,
        info={'protocol': 'week'}, describe={'protocol': 'week', 'colours': {
            'BF': {'exposure_ms': 12.5}}})
    for t in (1, 2):
        for well, site in (('B2', 'TA1'), ('B2', 'TA2'), ('C3', 'TA1')):
            take(saver, well, site, t)
            # Closed as soon as the position is through: nothing is open.
            assert not [n for _r, _d, names in os.walk(tmp_path)
                        for n in names if n.endswith('.index.json')]
    saver.close()
    assert saver.files_written == 6
    assert sorted(os.listdir(tmp_path)) == ['B2', 'C3']
    assert sorted(os.listdir(tmp_path / 'B2')) == [
        'B2_TA1_t0001.json', 'B2_TA1_t0001.ome.tiff',
        'B2_TA1_t0002.json', 'B2_TA1_t0002.ome.tiff',
        'B2_TA2_t0001.json', 'B2_TA2_t0001.ome.tiff',
        'B2_TA2_t0002.json', 'B2_TA2_t0002.ome.tiff']
    path = str(tmp_path / 'B2' / 'B2_TA2_t0002.ome.tiff')
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome
        series = tf.series[0]
        assert series.axes == 'CZYX' and series.shape == (2, 3, 128, 160)
        data = series.asarray()
    # colour 1 (GREEN), plane 2 of timepoint 2
    assert np.array_equal(data[1, 1], frame(2000 + 10 + 2))
    doc = well_file.read_index(path)
    assert doc['well'] == 'B2' and doc['info']['timepoint'] == 2


def test_the_sidecar_describes_the_file(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'position', um_per_px=0.65,
                               describe={'protocol': 'week', 'objective': '10x'})
    take(saver, 'B2', 'TA1', 7)
    saver.close()
    with open(tmp_path / 'B2' / 'B2_TA1_t0007.json', encoding='utf-8') as f:
        d = json.load(f)
    assert d['file'] == 'B2_TA1_t0007.ome.tiff'
    assert (d['well'], d['site'], d['timepoint']) == ('B2', 'TA1', 7)
    assert (d['stage_x_um'], d['stage_y_um']) == (1000.0, 2000.0)
    assert d['channels'] == ['BF', 'GREEN'] and d['z_planes'] == 3
    assert d['um_per_px'] == 0.65 and d['bits'] == 16
    assert d['acquisition'] == {'protocol': 'week', 'objective': '10x'}
    assert len(d['planes']) == 6
    assert d['planes'][4] == {'page': 4, 'channel': 'GREEN', 'z_index': 2,
                              'z_um': 5002.0, 't_s': 420.0, 'exposure_ms': 12.5}


def test_a_new_position_closes_the_one_before_and_nothing_is_overwritten(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'position')
    kw = dict(z_index=1, channel='BF', timepoint=1)
    saver.save(frame(1), well='B2', site='TA1', **kw)
    saver.save(frame(2), well='B2', site='TA2', **kw)     # no end_position
    assert os.path.exists(tmp_path / 'B2' / 'B2_TA1_t0001.json')
    saver.save(frame(3), well='B2', site='TA1', **kw)     # the first one again
    saver.close()
    assert np.array_equal(tifffile.imread(tmp_path / 'B2' / 'B2_TA1_t0001.ome.tiff'), frame(1))
    assert np.array_equal(tifffile.imread(tmp_path / 'B2' / 'B2_TA1_t0001_2.ome.tiff'), frame(3))


def test_without_wells_and_with_planes_kept_for_processing(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'position', keep_planes=True)
    path = saver.save(frame(1), well=None, site='P3', z_index=1, channel='BF',
                      timepoint=4)
    saver.close()
    assert os.path.exists(tmp_path / 'positions' / 'P3_t0004.ome.tiff')
    assert np.array_equal(tifffile.imread(path), frame(1))
    assert well_file.RunSaver.PLANES_DIR in path


def test_a_position_file_left_open_by_a_crash_is_finished(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'position')
    saver.save(frame(1), well='B2', site='TA1', z_index=1, channel='BF', timepoint=1)
    saver._pos[1]._drop_handles()                # the program ended here
    path = str(tmp_path / 'B2' / 'B2_TA1_t0001.ome.tiff')
    assert well_file.finish_folder(str(tmp_path)) == [path]
    assert np.array_equal(tifffile.imread(path), frame(1))


def test_stacks_and_tile_scans_take_the_format_as_one_file():
    assert well_file.container('position') == 'ome-tiff'
    assert not well_file.whole_run('position')
    assert well_file.SAVE_FORMATS[0] == 'position'


# ---------------------------------------------------------- the copying

def tree(folder) -> dict:
    out = {}
    for root, _dirs, names in os.walk(folder):
        for n in names:
            full = os.path.join(root, n)
            with open(full, 'rb') as f:
                out[os.path.relpath(full, folder).replace(os.sep, '/')] = f.read()
    return out


def run_folder(tmp_path):
    src = tmp_path / 'local' / 'protocol_1'
    saver = well_file.RunSaver(str(src), 'position')
    for t in (1, 2):
        take(saver, 'B2', 'TA1', t, planes=1)
    saver.close()
    (src / 'experiment.json').write_text('{}')
    return src


def test_the_run_folder_is_copied_and_checked(tmp_path):
    src = run_folder(tmp_path)
    dst = tmp_path / 'kept' / 'protocol_1'
    (src / '_processing_planes').mkdir()
    (src / '_processing_planes' / 'x.tiff').write_bytes(b'working file')
    (src / 'half.tmp').write_bytes(b'x')
    m = run_mirror.Mirror(str(src), str(dst), settle_s=0.0, pause_s=0.01)
    m.finish()
    m.start()
    assert m.wait(30.0) and m.complete
    want = {k: v for k, v in tree(src).items()
            if not k.startswith('_processing_planes') and k != 'half.tmp'
            and not k.startswith('transfer.')}
    assert tree(dst) == want and len(want) == 5
    with open(src / 'transfer.json', encoding='utf-8') as f:
        state = json.load(f)
    assert state['state'] == 'complete' and state['destination'] == str(dst)
    assert run_mirror.pending_runs(str(src.parent)) == []
    # Nothing local was removed.
    assert os.path.exists(src / 'B2' / 'B2_TA1_t0001.ome.tiff')


def test_a_destination_that_drops_out_is_waited_for(tmp_path, monkeypatch):
    src = run_folder(tmp_path)
    dst = tmp_path / 'kept' / 'protocol_1'
    said = []
    m = run_mirror.Mirror(str(src), str(dst), settle_s=0.0, pause_s=0.01,
                          status_cb=said.append)
    real = os.replace
    calls = {'n': 0}

    def replace(a, b):
        if str(b).startswith(str(dst)):
            calls['n'] += 1
            if 2 <= calls['n'] <= 4:
                raise OSError(22, 'Invalid argument')
        return real(a, b)
    monkeypatch.setattr(os, 'replace', replace)
    ok = [m.sync_once() for _ in range(6)]
    assert False in ok and ok[-1]
    assert any('IS WAITING' in t for t in said) and any('is back' in t for t in said)
    want = {k: v for k, v in tree(src).items() if not k.startswith('transfer.')}
    assert tree(dst) == want
    assert not [k for k in tree(dst) if k.endswith('.part')]


def test_a_copy_that_arrives_damaged_is_not_given_its_name(tmp_path, monkeypatch):
    src = run_folder(tmp_path)
    dst = tmp_path / 'kept' / 'protocol_1'
    m = run_mirror.Mirror(str(src), str(dst), settle_s=0.0)
    real = run_mirror._hash_file
    bad = {'on': True}
    monkeypatch.setattr(run_mirror, '_hash_file',
                        lambda p: 'wrong' if bad['on'] and p.endswith('.part') else real(p))
    assert not m.sync_once()
    assert tree(dst) == {} and 'not what was sent' in m.last_error
    bad['on'] = False
    assert m.sync_once() and len(tree(dst)) == 5


def test_a_file_still_being_written_waits_and_a_changed_one_goes_again(tmp_path):
    src = run_folder(tmp_path)
    dst = tmp_path / 'kept' / 'protocol_1'
    open_file = src / 'B2' / 'B2_TA9_t0001.ome.tiff'
    open_file.write_bytes(b'half a file')
    (src / 'B2' / 'B2_TA9_t0001.ome.tiff.index.json').write_text('{}')
    m = run_mirror.Mirror(str(src), str(dst), settle_s=0.0)
    assert m.sync_once() and m.held == 1
    assert 'B2/B2_TA9_t0001.ome.tiff' not in tree(dst)
    os.remove(src / 'B2' / 'B2_TA9_t0001.ome.tiff.index.json')
    (src / 'run.log').write_bytes(b'one line\n')
    assert m.sync_once() and m.held == 0
    (src / 'run.log').write_bytes(b'one line\nand another\n')
    assert m.sync_once()
    got = tree(dst)
    assert got['B2/B2_TA9_t0001.ome.tiff'] == b'half a file'
    assert got['run.log'] == b'one line\nand another\n'
    # A file written to in the last seconds is left for the next pass.
    slow = run_mirror.Mirror(str(src), str(tmp_path / 'other'), settle_s=3600.0)
    assert slow.sync_once() and slow.pending > 0 and tree(tmp_path / 'other') == {}


def test_copying_cut_short_is_carried_on(tmp_path):
    src = run_folder(tmp_path)
    dst = tmp_path / 'kept' / 'protocol_1'
    first = run_mirror.Mirror(str(src), str(dst), settle_s=0.0)
    first._save_state('running')                 # the program ended here
    assert run_mirror.pending_runs(str(src.parent)) == [(str(src), str(dst))]
    mirrors = run_mirror.resume_pending(str(src.parent))
    assert len(mirrors) == 1
    mirrors[0].settle_s = 0.0
    assert mirrors[0].wait(30.0) and mirrors[0].complete
    assert len(tree(dst)) == 5
    assert run_mirror.resume_pending(str(src.parent)) == []


def test_only_a_network_destination_is_written_locally_first(tmp_path, monkeypatch):
    assert run_mirror.spool_for(str(tmp_path)) is None
    # A local folder the user chose is always used, except the destination itself.
    chosen = tmp_path / 'chosen'
    assert run_mirror.spool_for(str(tmp_path / 'data'), str(chosen)) == str(chosen)
    assert run_mirror.spool_for(str(chosen), str(chosen)) is None
    assert run_mirror.spool_for(str(tmp_path), str(chosen)) is None
    monkeypatch.setattr(run_mirror, 'is_network_path',
                        lambda p: str(p).startswith('Y:'))
    spool = tmp_path / 'spool'
    assert run_mirror.spool_for('Y:\\data', str(spool)) == str(spool)
    assert run_mirror.spool_for('Y:\\data', 'Y:\\also', str(spool)) == str(spool)
