"""One file per well (well_file.py): every tile, z plane, colour and
timepoint in a single OME-TIFF or OME-Zarr, written as frames arrive and
packed from folders of single-page TIFFs."""

import json
import os

import numpy as np
import pytest

import well_file

tifffile = pytest.importorskip('tifffile')


def frame(tile, t, c, z, shape=(24, 32), dtype=np.uint16):
    """A frame whose every sample says which plane it is."""
    base = 1000 * tile + 100 * t + 10 * c + z
    return (np.arange(shape[0] * shape[1]).reshape(shape) % 7 + base).astype(dtype)


def write(path, fmt, order='acquired'):
    tiles, n_t, chans, n_z = ['TA1', 'TA2', 'TB1'], 3, ['BF', 'GREEN'], 4
    planes = [(ti, t, ci, z) for t in range(n_t) for ti in range(len(tiles))
              for ci in range(len(chans)) for z in range(n_z)
              # GREEN is not stacked: one plane only.
              if not (chans[ci] == 'GREEN' and z > 0)]
    if order == 'shuffled':
        np.random.default_rng(3).shuffle(planes)
    with well_file.WellWriter(path, fmt, um_per_px=0.65, z_step_um=2.5, well='B3',
                              info={'objective': '10x'}) as w:
        for ti, t, ci, z in planes:
            w.add(frame(ti, t, ci, z), tile=tiles[ti], z=z, channel=chans[ci], timepoint=t,
                  x_um=1000.0 * ti, y_um=500.0, z_um=6400.0 + 2.5 * z, t_s=60.0 * t)
    return tiles, n_t, chans, n_z, w.summary()


@pytest.mark.parametrize('fmt', well_file.FORMATS)
@pytest.mark.parametrize('order', ['acquired', 'shuffled'])
def test_a_well_with_tiles_z_colours_and_time_is_one_file_and_reads_back(tmp_path, fmt, order):
    path = well_file.well_path(str(tmp_path), 'B3', fmt)
    tiles, n_t, chans, n_z, summary = write(path, fmt, order)
    assert summary['planes'] == 3 * 3 * (4 + 1) and summary['timepoints'] == 3
    assert os.path.exists(path) and not os.path.exists(path + '.index.json')
    assert [n for n in os.listdir(tmp_path)] == [os.path.basename(path)], 'one file per well'
    got = well_file.read_well(path)
    # Tiles and colours are listed in the order they first arrived.
    assert sorted(got['tiles']) == tiles and sorted(got['channels']) == chans
    if order == 'acquired':
        assert list(got['tiles']) == tiles and got['channels'] == chans
    bf, green = got['channels'].index('BF'), got['channels'].index('GREEN')
    for ti, tile in enumerate(tiles):
        arr = got['tiles'][tile]
        assert arr.shape == (n_t, 2, n_z, 24, 32) and arr.dtype == np.uint16
        assert got['tile_xy'][tile] == (1000.0 * ti, 500.0)
        for t in range(n_t):
            for z in range(n_z):
                assert np.array_equal(arr[t, bf, z], frame(ti, t, 0, z))
            assert np.array_equal(arr[t, green, 0], frame(ti, t, 1, 0))
            assert not arr[t, green, 1:].any(), 'planes that were not acquired are empty'


def test_the_ome_tiff_is_read_by_tifffile_as_one_series_per_tile(tmp_path):
    path = well_file.well_path(str(tmp_path), 'B3', 'ome-tiff')
    tiles, n_t, chans, n_z, _ = write(path, 'ome-tiff')
    with tifffile.TiffFile(path) as tf:
        assert tf.is_ome and tf.is_bigtiff
        assert len(tf.pages) == 45
        assert len(tf.series) == len(tiles)
        for ti, series in enumerate(tf.series):
            assert series.axes == 'TCZYX' and series.shape == (n_t, 2, n_z, 24, 32)
            data = series.asarray()
            assert np.array_equal(data[2, 0, 3], frame(ti, 2, 0, 3))
            assert np.array_equal(data[1, 1, 0], frame(ti, 1, 1, 0))
        xml = tf.ome_metadata
    assert 'PhysicalSizeX="0.65"' in xml and 'PhysicalSizeZ="2.5"' in xml
    assert 'Name="GREEN"' in xml and 'StageLabel Name="TB1" X="2000.000"' in xml


def test_a_crash_before_close_leaves_an_index_of_every_page(tmp_path):
    path = well_file.well_path(str(tmp_path), None, 'ome-tiff')
    assert path.endswith('image.ome.tiff')
    w = well_file.WellWriter(path)
    w.add(frame(0, 0, 0, 0), tile='P1', z=0, channel='BF')
    w.add(frame(0, 0, 0, 1), tile='P1', z=1, channel='BF')
    with open(path + '.index.json', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    assert lines[0]['header']['shape'] == [24, 32], 'the first line says what the frames are'
    assert [(r['page'], r['z']) for r in lines[1:]] == [(0, 0), (1, 1)]
    # A reader takes the planes of a file that was never closed from it.
    got = well_file.read_well(path)
    assert np.array_equal(got['tiles']['P1'][0, 0, 1], frame(0, 0, 0, 1))
    w.close()


def big(well, tile, t, c, z):
    """A frame large enough to have reduced copies, different everywhere."""
    y, x = np.mgrid[0:200, 0:260]
    return ((x * 7 + y * 13 + 1000 * well + 100 * tile + 10 * t + 3 * c + z) % 4000).astype(np.uint16)


@pytest.mark.parametrize('fmt', well_file.FORMATS)
def test_every_frame_has_its_reduced_copies_each_the_mean_of_its_block(tmp_path, fmt):
    assert well_file.level_shapes((1900, 1900)) == [(1900, 1900), (950, 950), (475, 475), (237, 237), (118, 118)]
    assert well_file.level_shapes((200, 260)) == [(200, 260), (100, 130)]
    path = well_file.well_path(str(tmp_path), 'B3', fmt)
    with well_file.WellWriter(path, fmt, um_per_px=0.65, well='B3') as w:
        for z in range(3):
            w.add(big(0, 0, 0, 0, z), tile='TA1', z=z, channel='BF', x_um=5.0, y_um=6.0)
    half = well_file.read_run(path, level=1)
    assert half['levels'] == [(200, 260), (100, 130)]
    for z in range(3):
        want = big(0, 0, 0, 0, z).reshape(100, 2, 130, 2).mean((1, 3))
        assert np.abs(half['images'][('', 'TA1')][0, 0, z].astype(float) - want).max() <= 0.5
    if fmt == 'ome-tiff':
        with tifffile.TiffFile(path) as tf:
            assert len(tf.pages) == 3, 'the reduced copies are not pages of their own'
            assert len(tf.series) == 1 and len(tf.series[0].levels) == 2
            assert tf.series[0].levels[1].shape[-2:] == (100, 130)
    else:
        with open(os.path.join(path, 'TA1', '.zattrs'), encoding='utf-8') as f:
            sets = json.load(f)['multiscales'][0]['datasets']
        assert [d['path'] for d in sets] == ['0', '1']
        assert sets[1]['coordinateTransformations'][0]['scale'][3:] == [1.3, 1.3]


@pytest.mark.parametrize('fmt', ['run-ome-tiff', 'run-zarr'])
def test_a_whole_run_is_one_file_with_its_wells_as_a_plate(tmp_path, fmt):
    import experiment
    base = tmp_path / 'protocol_20261120_101500'
    saver = well_file.RunSaver(str(base), fmt, um_per_px=0.65, z_step_um=2.0,
                               name_of=experiment.file_name)
    wells = [('B3', 'TA1'), ('B3', 'TA2'), ('C11', 'TA1')]
    for t in (1, 2):
        for wi, (well, site) in enumerate(wells):
            for ci, ch in enumerate(('BF', 'GREEN')):
                for z in (1, 2):
                    assert saver.save(big(wi, int(site[-1]), t, ci, z), well=well, site=site, z_index=z,
                                      channel=ch, timepoint=t, x_um=100.0 * wi, y_um=20.0) is None
    done = saver.close()
    ext = '.ome.tiff' if fmt == 'run-ome-tiff' else '.zarr'
    assert os.listdir(base) == ['protocol_20261120_101500' + ext], 'one file for the run'
    path = str(base / ('protocol_20261120_101500' + ext))
    assert done[0]['planes'] == 24 and done[0]['wells'] == ['B3', 'C11'] and done[0]['levels'] == 2
    got = well_file.read_run(path)
    assert list(got['images']) == [('B3', 'TA1'), ('B3', 'TA2'), ('C11', 'TA1')]
    assert got['images'][('C11', 'TA1')].shape == (2, 2, 2, 200, 260)
    assert np.array_equal(got['images'][('B3', 'TA2')][1, 1, 1], big(1, 2, 2, 1, 2))
    assert got['xy'][('C11', 'TA1')] == (200.0, 20.0)
    doc = well_file.read_index(path)
    assert doc['scope'] == 'run' and doc['version'] == 2
    if fmt == 'run-ome-tiff':
        assert all(len(row) == 4 + 2 for row in doc['planes']), 'an offset for the plane and for its copy'
        with tifffile.TiffFile(path) as tf:
            assert len(tf.series) == 3
            xml = tf.ome_metadata
        assert '<Plate ID="Plate:0"' in xml and 'Rows="3" Columns="11"' in xml
        assert '<Well ID="Well:0:1" Row="2" Column="10">' in xml and 'Name="C11 TA1"' in xml
    else:
        with open(os.path.join(path, '.zattrs'), encoding='utf-8') as f:
            plate = json.load(f)['plate']
        assert [w['path'] for w in plate['wells']] == ['B/3', 'C/11'] and plate['field_count'] == 2
        with open(os.path.join(path, 'B', '3', '.zattrs'), encoding='utf-8') as f:
            assert [i['path'] for i in json.load(f)['well']['images']] == ['TA1', 'TA2']
        assert os.path.exists(os.path.join(path, 'C', '11', 'TA1', '1', '.zarray'))


def test_a_zarr_store_says_what_it_holds_from_the_first_frame_of_every_timepoint(tmp_path):
    path = str(tmp_path / 'run.zarr')
    w = well_file.WellWriter(path, 'zarr', run=True)
    w.add(big(0, 0, 0, 0, 0), tile='TA1', well='B3')
    assert len(well_file.read_index(path)['planes']) == 1, 'readable while the run goes on'
    w.add(big(0, 1, 0, 0, 0), tile='TA2', well='B3')
    w.add(big(0, 0, 1, 0, 0), tile='TA1', well='B3', timepoint=1)
    assert len(well_file.read_index(path)['planes']) == 3
    w.close()


def test_a_folder_of_single_files_is_packed_into_one_file_for_the_run(tmp_path):
    src = tmp_path / 'run'
    for well in ('B3', 'C4'):
        (src / well).mkdir(parents=True)
        for t in (1, 2):
            tifffile.imwrite(str(src / well / f'{well}_TA1_Z001_BF_t{t:04d}.tiff'), big(1, 1, t, 0, 0))
    out = well_file.pack_run(str(src), 'run-zarr', out_dir=str(tmp_path / 'packed'))
    assert out[0]['planes'] == 4 and out[0]['wells'] == ['B3', 'C4']
    got = well_file.read_run(str(tmp_path / 'packed' / 'packed.zarr'))
    assert np.array_equal(got['images'][('C4', 'TA1')][1, 0, 0], big(1, 1, 2, 0, 0))


def test_what_cannot_be_one_well_is_refused(tmp_path):
    w = well_file.WellWriter(str(tmp_path / 'a.ome.tiff'))
    w.add(frame(0, 0, 0, 0), tile='P1')
    with pytest.raises(ValueError, match='twice'):
        w.add(frame(0, 0, 0, 0), tile='P1')
    with pytest.raises(ValueError, match='the well is'):
        w.add(frame(0, 0, 0, 1, shape=(10, 10)), tile='P1', z=1)
    with pytest.raises(ValueError, match='2-D'):
        w.add(np.zeros((4, 4, 3), np.uint8), tile='P1', z=2)
    w.close()
    with pytest.raises(RuntimeError):
        w.add(frame(0, 0, 0, 5), tile='P1', z=5)
    with pytest.raises(ValueError):
        well_file.WellWriter(str(tmp_path / 'b'), 'png')


@pytest.mark.parametrize('fmt', well_file.FORMATS)
def test_a_live_z_stack_folder_becomes_one_file_lowest_plane_first(tmp_path, fmt):
    folder = tmp_path / 'zstack_20261118_165509'
    folder.mkdir()
    # Written top down, as the microscope does.
    for ci, ch in enumerate(['BF', 'DF', 'PC']):
        for j in range(5):
            z_um = 6500.0 - 3.7 * j
            tifffile.imwrite(str(folder / f'zstack_{ch}_{j:03d}_{z_um:.1f}um.tiff'),
                             frame(0, 0, ci, j, dtype=np.uint8))
    summary = well_file.pack_zstack(str(folder), fmt)
    assert summary['path'] == str(folder) + well_file.EXTENSION[fmt]
    assert summary['channels'] == ['BF', 'DF', 'PC'] and summary['z_planes'] == 5
    got = well_file.read_well(summary['path'])
    arr = got['tiles']['P1']
    assert arr.shape == (1, 3, 5, 24, 32) and arr.dtype == np.uint8
    for ci in range(3):
        assert np.array_equal(arr[0, ci, 0], frame(0, 0, ci, 4, dtype=np.uint8)), 'lowest plane first'
        assert np.array_equal(arr[0, ci, 4], frame(0, 0, ci, 0, dtype=np.uint8))


def test_a_protocol_runs_single_files_become_one_file_per_well(tmp_path):
    import experiment
    run = tmp_path / 'run'
    for well in ('B3', 'C4'):
        (run / well).mkdir(parents=True)
        for site in ('TA1', 'TA2'):
            for t in (1, 2):
                for ch in ('BF', 'GREEN'):
                    for z in (1, 2):
                        tifffile.imwrite(str(run / well / experiment.file_name(well, site, z, ch, t)),
                                         frame(int(site[-1]), t, ch == 'GREEN', z))
    done = well_file.pack_run(str(run), 'ome-tiff', positions={('B3', 'TA2'): (1234.0, 56.0)})
    assert [d['well'] for d in done] == ['B3', 'C4']
    assert sorted(os.listdir(run / 'wells')) == ['B3.ome.tiff', 'C4.ome.tiff']
    got = well_file.read_well(str(run / 'wells' / 'B3.ome.tiff'))
    assert list(got['tiles']) == ['TA1', 'TA2'] and got['channels'] == ['BF', 'GREEN']
    assert got['tiles']['TA2'].shape == (2, 2, 2, 24, 32)
    assert np.array_equal(got['tiles']['TA2'][1, 1, 0], frame(2, 2, True, 1))
    assert got['tile_xy']['TA2'] == (1234.0, 56.0)


def test_a_protocol_runs_frames_go_to_one_file_per_well(tmp_path):
    import experiment
    saver = well_file.RunSaver(str(tmp_path), 'ome-tiff', um_per_px=0.65, z_step_um=2.0,
                               name_of=experiment.file_name)
    for t in (1, 2):
        for well, site in (('B3', 'TA1'), ('B3', 'TA2'), ('C4', 'P1')):
            for ch in ('BF', 'GREEN'):
                for z in (1, 2):
                    got = saver.save(frame(int(site[-1]), t, ch == 'GREEN', z), well=well, site=site,
                                     z_index=z, channel=ch, timepoint=t, x_um=10.0, y_um=20.0, z_um=5.0 + z)
                    assert got is None, 'no single-page file is written'
    done = saver.close()
    assert sorted(os.listdir(tmp_path)) == ['B3.ome.tiff', 'C4.ome.tiff']
    assert [d['planes'] for d in done] == [16, 8]
    b3 = well_file.read_well(str(tmp_path / 'B3.ome.tiff'))
    assert b3['tiles']['TA2'].shape == (2, 2, 2, 24, 32)
    assert np.array_equal(b3['tiles']['TA2'][1, 1, 1], frame(2, 2, True, 2))


def test_planes_kept_for_online_processing_are_removed_afterwards_and_single_is_as_before(tmp_path):
    import experiment
    keep = well_file.RunSaver(str(tmp_path / 'a'), 'zarr', keep_planes=True, name_of=experiment.file_name)
    path = keep.save(frame(1, 1, 0, 1), well='B3', site='TA1', z_index=1, channel='BF', timepoint=1)
    assert path.endswith(os.path.join('_processing_planes', 'B3', 'B3_TA1_Z001_BF_t0001.tiff'))
    assert np.array_equal(tifffile.imread(path), frame(1, 1, 0, 1))
    # A colour frame cannot go into a well file: it is kept as its own file.
    odd = keep.save(np.zeros((4, 4, 3), np.uint8), well='B3', site='TA1', z_index=2, channel='BF', timepoint=1)
    assert odd.endswith(os.path.join('B3', 'B3_TA1_Z002_BF_t0001.tiff')) and keep.fallback == 1
    keep.close()
    keep.remove_planes()
    assert sorted(os.listdir(tmp_path / 'a')) == ['B3', 'B3.zarr']
    single = well_file.RunSaver(str(tmp_path / 'b'), 'single', name_of=experiment.file_name)
    path = single.save(frame(1, 1, 0, 1), well=None, site='P1', z_index=1, channel='BF', timepoint=3)
    assert path == os.path.join(str(tmp_path / 'b'), 'positions', 'P1_Z001_BF_t0003.tiff')
    assert single.close() == []
    # A float frame (after a correction) is stored rounded.
    assert well_file.storable(np.array([[1.4, 70000.0]])).tolist() == [[1, 65535]]


@pytest.mark.parametrize('fmt', well_file.FORMATS)
def test_a_live_z_stack_is_one_file_and_its_folder_is_gone(tmp_path, fmt):
    out_dir = tmp_path / 'zstack_20261118_170000'
    out_dir.mkdir()
    heights = [50.0, 46.0, 42.0]            # top down, as acquired
    saver = well_file.StackSaver(str(out_dir), fmt, {'BF': heights, 'PC': heights})
    for j, z in enumerate(heights):
        for ci, ch in enumerate(('BF', 'PC')):
            saver.save(frame(0, 0, ci, j, dtype=np.uint8), ch, j, z)
    path = saver.close()
    assert path == str(out_dir) + well_file.EXTENSION[fmt]
    assert os.listdir(tmp_path) == [os.path.basename(path)], 'the empty folder is removed'
    arr = well_file.read_well(path)['tiles']['P1']
    assert arr.shape == (1, 2, 3, 24, 32)
    assert np.array_equal(arr[0, 1, 0], frame(0, 0, 1, 2, dtype=np.uint8)), 'lowest plane first'
    # 'single' keeps the files the stack always wrote.
    old = tmp_path / 'zstack_old'
    old.mkdir()
    s = well_file.StackSaver(str(old), 'single', {'BF': heights})
    s.save(frame(0, 0, 0, 0, dtype=np.uint8), 'BF', 0, 50.0)
    assert s.close() == str(old) and os.listdir(old) == ['zstack_BF_000_50.0um.tiff']
