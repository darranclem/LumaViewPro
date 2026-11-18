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
    assert [(r['page'], r['z']) for r in lines] == [(0, 0), (1, 1)]
    w.close()


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
