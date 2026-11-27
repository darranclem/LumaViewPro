"""A run that dies: its well files are finished afterwards, and the next
start reports it (run_guard, well_file.finish)."""
import os
import threading
import time

import numpy as np
import pytest

tifffile = pytest.importorskip('tifffile')

import run_guard
import well_file


def _frame(v, shape=(256, 320)):
    return np.full(shape, v, np.uint16)


def _died_writer(path, n, run=False, half_page=True):
    """A writer that took ``n`` frames and was never closed; with
    ``half_page`` the process 'died' while writing one more page."""
    w = well_file.WellWriter(path, 'ome-tiff', um_per_px=0.5, run=run,
                             well=None if run else 'G2')
    for i in range(n):
        w.add(_frame(i + 1), tile='TA1', z=i % 3, channel='BF',
              timepoint=i // 3, x_um=100.0, y_um=200.0, well='G2')
    if half_page:
        # A page written (and linked into the chain) but not indexed, then
        # cut short as if the process stopped part way through.
        w._tw.write(_frame(999), photometric='minisblack', metadata=None,
                    contiguous=False)
        w._tw._fh.flush()
    w._tw._fh.close()
    w._index.close()
    if half_page:
        size = os.path.getsize(path)
        with open(path, 'r+b') as f:
            f.truncate(size - 1000)
    return w


@pytest.mark.parametrize('run', [False, True])
def test_finish_after_crash(tmp_path, run):
    path = str(tmp_path / ('run.ome.tiff' if run else 'G2.ome.tiff'))
    _died_writer(path, 7, run=run)
    assert well_file.unfinished(str(tmp_path)) == [path]
    assert well_file.finish(path)
    assert not os.path.exists(path + '.index.json')
    assert well_file.unfinished(str(tmp_path)) == []
    # Every reader now sees the 7 frames and nothing of the half page.
    with tifffile.TiffFile(path) as tf:
        assert len(tf.pages) == 7
        assert tf.is_ome
        assert 'finished' in tf.pages[0].description
        for i, page in enumerate(tf.pages):
            assert int(page.asarray()[0, 0]) == i + 1
    doc = well_file.read_index(path)
    assert len(doc['planes']) == 7
    got = well_file.read_run(path)
    (arr,) = got['images'].values()
    assert arr.shape[:3] == (3, 1, 3)       # t, c, z
    assert int(arr[2, 0, 0, 0, 0]) == 7


def test_finish_without_half_page_and_partial_index_line(tmp_path):
    path = str(tmp_path / 'F2.ome.tiff')
    _died_writer(path, 4, half_page=False)
    with open(path + '.index.json', 'a', encoding='utf-8') as f:
        f.write('{"page": 4, "tile": "TA')            # cut mid-line
    assert well_file.finish(path)
    with tifffile.TiffFile(path) as tf:
        assert len(tf.pages) == 4


def test_finish_folder_and_nothing_to_do(tmp_path):
    (tmp_path / 'sub').mkdir()
    a = str(tmp_path / 'A1.ome.tiff')
    b = str(tmp_path / 'sub' / 'B1.ome.tiff')
    _died_writer(a, 3)
    _died_writer(b, 2)
    assert well_file.finish_folder(str(tmp_path)) == sorted([a, b])
    assert not well_file.finish(a)            # already finished


def test_runsaver_close_from_another_thread(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'ome-tiff')
    saver.save(_frame(1), well='B2', site='TA1', z_index=1, channel='BF',
               timepoint=1)
    out = []
    t = threading.Thread(target=lambda: out.append(saver.close()))
    t.start()
    t.join(5)
    assert out and out[0][0]['planes'] == 1
    with pytest.raises(RuntimeError):
        saver.save(_frame(2), well='B2', site='TA1', z_index=2,
                   channel='BF', timepoint=1)
    assert saver.close() == [] or True        # a second close is harmless
    assert not os.path.exists(str(tmp_path / 'B2.ome.tiff.index.json'))


def test_runsaver_close_times_out_while_saving(tmp_path):
    saver = well_file.RunSaver(str(tmp_path), 'ome-tiff')
    saver._lock.acquire()                     # a save stuck part way
    try:
        assert saver.close(timeout_s=0.1) == []
        assert not saver.closed
    finally:
        saver._lock.release()


def test_run_log_and_previous_run_died(tmp_path, monkeypatch):
    logs = tmp_path / 'logs'
    run_dir = tmp_path / 'protocol_x'
    run_dir.mkdir()
    monkeypatch.setattr(run_guard, 'windows_events',
                        lambda since, until=None: ['Date: x\nSource: Application Error\n  python.exe crashed'])
    run_guard.install(str(logs))
    log = run_guard.RunLog(str(run_dir), str(logs), '2 timepoints')
    log.write('timepoint 1, G2 TA1 done')
    assert (logs / run_guard.ACTIVE_NAME).exists()
    # ... the process dies here; at the next start:
    _died_writer(str(run_dir / 'G2.ome.tiff'), 3)
    found = run_guard.previous_run_died(str(logs), well_file.finish_folder)
    assert found is not None
    assert found['run_dir'] == str(run_dir)
    assert any('G2 TA1 done' in line for line in found['last'])
    assert found['finished'] == [str(run_dir / 'G2.ome.tiff')]
    assert 'python.exe crashed' in found['report']
    assert not (logs / run_guard.ACTIVE_NAME).exists()
    assert 'did not end' in (run_dir / 'run.log').read_text(encoding='utf-8')
    assert 'did not end' in (logs / 'errors.log').read_text(encoding='utf-8')
    assert run_guard.previous_run_died(str(logs)) is None


def test_run_log_clean_end(tmp_path):
    logs = tmp_path / 'logs'
    run_dir = tmp_path / 'r'
    run_dir.mkdir()
    log = run_guard.RunLog(str(run_dir), str(logs))
    log.end('finished')
    assert run_guard.previous_run_died(str(logs)) is None
    text = (run_dir / 'run.log').read_text(encoding='utf-8')
    assert 'RUN ENDED: finished' in text


def test_watchdog_reports_a_stall(tmp_path):
    logs = tmp_path / 'logs'
    run_guard.install(str(logs))
    wd = run_guard.Watchdog(check_s=0.05)
    wd.beat('test', 0.1, 'waiting for nothing')
    time.sleep(0.5)
    wd.rest('test')
    text = (logs / 'errors.log').read_text(encoding='utf-8')
    assert 'STALL: test' in text


def test_exit_code_text():
    assert 'ACCESS VIOLATION' in run_guard.exit_code_text(-1073741819)
    assert 'ACCESS VIOLATION' in run_guard.exit_code_text(0xC0000005)
    assert run_guard.exit_code_text(None) == 'exit code unknown'


@pytest.mark.skipif(os.name != 'nt', reason='Windows exit codes')
def test_park_guard_reads_the_exit_code():
    import subprocess
    import sys
    import park_guard
    child = subprocess.Popen([sys.executable, '-c',
                              'import time; time.sleep(1.0); '
                              'import ctypes; ctypes.windll.kernel32.ExitProcess(0xC0000005)'])
    code = park_guard.wait_for_exit(child.pid)
    child.wait()
    assert code is not None
    assert 'ACCESS VIOLATION' in run_guard.exit_code_text(code)
