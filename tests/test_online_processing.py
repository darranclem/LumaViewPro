"""Online compositing, stitching and timelapse video
(online_processing.py)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import online_processing as op

tifffile = pytest.importorskip('tifffile')

TW, TH = 80, 60
STEP_X, STEP_Y = 64, 48                 # 20 % overlap
PX = [[1.0, 0.0], [0.0, 1.0]]           # 1 um = 1 px, no rotation


def scene(w=400, h=300, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(100, 4000, size=(h, w)).astype(np.uint16)


def grid_positions(rows=2, cols=3, well='B3', x0=1000.0, y0=2000.0):
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append({'well': well, 'site': f'T{"AB"[r]}{c + 1}',
                        'x_um': x0 + c * STEP_X, 'y_um': y0 + r * STEP_Y})
    return out


def write_tiles(folder, positions, channels, tp, planes=1, base=None):
    """Cut every tile out of one scene per channel and plane; returns
    {index: {channel: [paths]}} and the scenes."""
    scenes = {(ch, z): (base if base is not None else scene(seed=7))
              + 10 * k + z for k, ch in enumerate(channels)
              for z in range(planes)}
    x0 = min(p['x_um'] for p in positions)
    y0 = min(p['y_um'] for p in positions)
    files = {}
    for i, p in enumerate(positions):
        x, y = int(p['x_um'] - x0), int(p['y_um'] - y0)
        files[i] = {}
        for ch in channels:
            files[i][ch] = []
            for z in range(planes):
                path = os.path.join(
                    folder, f'{p["site"]}_Z{z + 1:03d}_{ch}_t{tp:04d}.tiff')
                tifffile.imwrite(path, scenes[(ch, z)][y:y + TH, x:x + TW])
                files[i][ch].append(path)
    return files, scenes


# ------------------------------------------------------------ grouping

def test_only_tile_sites_are_grouped_and_by_well():
    pos = (grid_positions(well='A1') + grid_positions(well='A2')
           + [{'well': 'A3', 'site': 'P1', 'x_um': 0, 'y_um': 0},
              {'well': 'A3', 'site': 'P2', 'x_um': 9, 'y_um': 9},
              {'well': 'A4', 'site': 'TA1', 'x_um': 0, 'y_um': 0}])
    groups = op.stitch_groups(pos)
    assert set(groups) == {('A1', None), ('A2', None)}
    assert groups[('A1', None)] == list(range(6))


def test_free_regions_without_a_plate_are_separate_groups():
    pos = [{'well': None, 'site': f'R{r}-TA{c}', 'x_um': c, 'y_um': r}
           for r in (1, 2) for c in (1, 2)]
    groups = op.stitch_groups(pos)
    assert set(groups) == {(None, 'R1'), (None, 'R2')}
    assert op.group_name((None, 'R1')) == 'R1'
    assert op.group_name(('B3', None)) == 'B3'


# ----------------------------------------------------------- stitching

def test_offsets_follow_the_stage_through_the_matrix():
    xy = [(0, 0), (100, 0), (0, 50)]
    assert op.tile_offsets(xy, PX) == [(0, 0), (100, 0), (0, 50)]
    # stage +x moves the tile toward -x on the canvas, 2 px per um
    assert op.tile_offsets(xy, [[-2, 0], [0, 2]]) == \
        [(200, 0), (0, 0), (200, 100)]
    assert op.tile_offsets(xy, PX, scale=2) == [(0, 0), (50, 0), (0, 25)]


def test_margins_split_each_overlap_down_the_middle():
    offs = op.tile_offsets([(p['x_um'], p['y_um'])
                            for p in grid_positions()], PX)
    m = op.tile_margins(offs, TW, TH)
    assert m[0] == (0, 0, 8, 6)          # top-left: trims right and bottom
    assert m[1] == (8, 0, 8, 6)          # top-middle
    assert m[5] == (8, 6, 0, 0)          # bottom-right


def test_stitching_rebuilds_the_scene_exactly(tmp_path):
    pos = grid_positions()
    files, scenes = write_tiles(str(tmp_path), pos, ['GREEN'], 1)
    xy = [(p['x_um'], p['y_um']) for p in pos]
    offs = op.tile_offsets(xy, PX)
    tiles = (tifffile.imread(files[i]['GREEN'][0]) for i in range(len(pos)))
    out = op.stitch_plane(tiles, offs, op.tile_margins(offs, TW, TH),
                          TW, TH, np.uint16)
    w, h = op.canvas_size(offs, TW, TH)
    assert (w, h) == (2 * STEP_X + TW, STEP_Y + TH)
    assert np.array_equal(out, scenes[('GREEN', 0)][:h, :w])


def test_large_stitches_are_reduced_by_a_whole_factor():
    xy = [(0, 0), (64, 0), (0, 48), (64, 48)]
    assert op.stitch_scale(xy, PX, TW, TH, max_pixels=1e9) == 1
    full = 144 * 108
    assert op.stitch_scale(xy, PX, TW, TH, max_pixels=full / 3.5) == 2


# ------------------------------------------------------------- display

def test_rgb_puts_fluorescence_in_colour_over_dimmed_brightfield():
    bf = np.full((40, 60), 1000, np.uint16)
    green = np.zeros((40, 60), np.uint16)
    green[:, 30:] = 500
    rgb = op.render_rgb({'BF': bf, 'GREEN': green},
                        {'BF': (0, 1000), 'GREEN': (0, 500)})
    assert rgb.shape == (40, 60, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[5, 5]) == (153, 153, 153)        # 0.6 grey
    assert tuple(rgb[5, 50]) == (153, 255, 153)       # + green
    odd = op.render_rgb({'BF': np.zeros((41, 61), np.uint8)}, {'BF': (0, 1)})
    assert odd.shape[0] % 2 == 0 and odd.shape[1] % 2 == 0


def test_elapsed_text():
    assert op.elapsed_text(0) == '00:00:00'
    assert op.elapsed_text(3725) == '01:02:05'


# --------------------------------------------------------------- files

def test_hyperstack_has_z_and_channels_and_opens_as_imagej(tmp_path):
    path = str(tmp_path / 'c.tiff')
    data = np.arange(3 * 2 * 20 * 30, dtype=np.uint16).reshape(3, 2, 20, 30)
    op.write_hyperstack(path, (data[z, c] for z in range(3)
                               for c in range(2)),
                        3, ['BF', 'GREEN'], 20, 30, np.uint16,
                        ranges={'BF': (0, 100), 'GREEN': (5, 50)},
                        um_per_px=0.5, z_step_um=2.0, info='hello')
    with tifffile.TiffFile(path) as tf:
        assert tf.is_imagej
        meta = tf.imagej_metadata
        assert meta['slices'] == 3 and meta['channels'] == 2
        assert meta['mode'] == 'composite' and meta['spacing'] == 2.0
        assert tf.series[0].axes == 'ZCYX'
        assert np.array_equal(tf.asarray(), data)


def test_a_channel_without_a_stack_stands_in_on_every_plane():
    assert op.plane_path(['a'], 3) == 'a'
    assert op.plane_path(['a', 'b', 'c'], 1) == 'b'


# ----------------------------------------------------------- processor

def run(proc):
    proc.finish()
    assert proc.wait(60)
    assert proc.counts['errors'] == 0, open(
        os.path.join(proc.base_dir, 'processing.log')).read()


def test_composite_is_written_when_a_position_is_done(tmp_path):
    pos = [{'well': 'B3', 'site': 'P1', 'x_um': 0, 'y_um': 0}]
    files, scenes = write_tiles(str(tmp_path), [dict(pos[0], site='P1')],
                                ['BF', 'GREEN'], 1, planes=3)
    files[0]['BF'] = files[0]['BF'][:1]             # BF not stacked
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF', 'GREEN'],
                              compose=True, stitch=True, px_per_um=PX,
                              um_per_px=1.0, z_step_um=2.0)
    proc.position_done(0, 1, files[0])
    run(proc)
    out = tmp_path / 'composites' / 'B3' / 'B3_P1_composite_t0001.tiff'
    arr = tifffile.imread(str(out))
    assert arr.shape == (3, 2, TH, TW)
    for z in range(3):
        assert np.array_equal(arr[z, 0], scenes[('BF', 0)][:TH, :TW])
        assert np.array_equal(arr[z, 1], scenes[('GREEN', z)][:TH, :TW])
    assert proc.counts == {'composites': 1, 'stitches': 0, 'videos': 0,
                           'corrected': 0, 'errors': 0}


def test_stitch_waits_for_the_last_tile_and_holds_all_channels(tmp_path):
    pos = grid_positions()
    files, scenes = write_tiles(str(tmp_path), pos, ['BF', 'RED'], 1,
                                planes=2)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF', 'RED'],
                              compose=True, stitch=True, px_per_um=PX,
                              um_per_px=1.0)
    for i in range(len(pos) - 1):
        proc.position_done(i, 1, files[i])
    assert proc._q.empty()                          # one tile still to come
    proc.position_done(len(pos) - 1, 1, files[len(pos) - 1])
    run(proc)
    arr = tifffile.imread(str(tmp_path / 'stitched' /
                              'B3_stitched_t0001.tiff'))
    h, w = STEP_Y + TH, 2 * STEP_X + TW
    assert arr.shape == (2, 2, h, w)
    assert np.array_equal(arr[1, 1], scenes[('RED', 1)][:h, :w])
    # tiles that are stitched are not also composited one by one
    assert not (tmp_path / 'composites').exists()
    assert proc.counts['stitches'] == 1


def test_stitching_alone_writes_one_file_per_channel(tmp_path):
    pos = grid_positions()
    files, _ = write_tiles(str(tmp_path), pos, ['BF', 'RED'], 1)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF', 'RED'],
                              compose=False, stitch=True, px_per_um=PX,
                              um_per_px=1.0)
    for i in range(len(pos)):
        proc.position_done(i, 1, files[i])
    run(proc)
    names = sorted(os.listdir(tmp_path / 'stitched'))
    assert names == ['B3_stitched_BF_t0001.tiff',
                     'B3_stitched_RED_t0001.tiff']


def test_oversized_stitch_is_reduced(tmp_path):
    pos = grid_positions()
    files, _ = write_tiles(str(tmp_path), pos, ['BF'], 1)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=False,
                              stitch=True, px_per_um=PX, um_per_px=1.0,
                              max_stitch_pixels=208 * 108 / 3.5)
    for i in range(len(pos)):
        proc.position_done(i, 1, files[i])
    run(proc)
    arr = tifffile.imread(str(tmp_path / 'stitched' /
                              'B3_stitched_BF_t0001.tiff'))
    assert arr.shape == (54, 104)


def _frames_in(path):
    import cv2
    v = cv2.VideoCapture(str(path))
    try:
        return int(v.get(cv2.CAP_PROP_FRAME_COUNT)), v.get(cv2.CAP_PROP_FPS)
    finally:
        v.release()


def test_videos_of_the_stitches_are_made_as_mp4_and_avi(tmp_path):
    pos = grid_positions()
    done = []
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF', 'GREEN'],
                              compose=True, stitch=True, px_per_um=PX,
                              um_per_px=1.0, cycles=3,
                              done_cb=lambda s, v: done.append((s, v)))
    for tp in (1, 2, 3):
        folder = tmp_path / f't{tp}'
        folder.mkdir()
        files, _ = write_tiles(str(folder), pos, ['BF', 'GREEN'], tp,
                               base=scene(seed=tp))
        for i in range(len(pos)):
            proc.position_done(i, tp, files[i], elapsed_s=tp * 600.0)
    run(proc)
    assert len(os.listdir(tmp_path / 'previews' / 'B3_stitched')) == 3
    mp4 = tmp_path / 'videos' / 'B3_stitched.mp4'
    avi = tmp_path / 'videos' / 'B3_stitched.avi'
    assert sorted(os.listdir(tmp_path / 'videos')) == [avi.name, mp4.name]
    assert _frames_in(mp4)[0] == 3 and _frames_in(avi)[0] == 3
    assert proc.counts['videos'] == 1 and proc.counts['errors'] == 0
    (summary, videos), = done
    assert '1 videos' in summary
    assert sorted(videos) == [str(avi), str(mp4)]
    assert 'video of B3_stitched: 3 timepoints' in \
        (tmp_path / 'processing.log').read_text()


def test_a_movie_exists_during_the_run_and_survives_a_stop(tmp_path):
    pos = grid_positions()
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=False,
                              stitch=True, px_per_um=PX, um_per_px=1.0,
                              cycles=30)
    mp4 = tmp_path / 'videos' / 'B3_stitched.mp4'
    for tp in (1, 2):
        folder = tmp_path / f't{tp}'
        folder.mkdir()
        files, _ = write_tiles(str(folder), pos, ['BF'], tp,
                               base=scene(seed=tp))
        for i in range(len(pos)):
            proc.position_done(i, tp, files[i], elapsed_s=tp * 600.0)
    for _ in range(200):                 # before finish(): made on the way
        if mp4.exists():
            break
        time.sleep(0.05)
    assert mp4.exists()
    run(proc)                            # stopped after 2 of 30 timepoints
    assert _frames_in(mp4)[0] == 2


def test_a_short_timelapse_is_slowed_down_to_be_watchable():
    assert op.video_fps(6) == 1          # 6 timepoints last 6 s, not 1.2 s
    assert op.video_fps(30) == 5
    assert op.video_fps(600) == 10


def test_an_incomplete_tile_group_is_reported_not_stitched(tmp_path):
    pos = grid_positions()
    files, _ = write_tiles(str(tmp_path), pos, ['BF'], 1)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=False,
                              stitch=True, px_per_um=PX, um_per_px=1.0)
    for i in range(len(pos) - 1):                   # stopped before the end
        proc.position_done(i, 1, files[i])
    run(proc)
    assert proc.counts['stitches'] == 0
    assert not (tmp_path / 'stitched').exists()
    assert 'incomplete' in proc.summary()


# ------------------------------------------------- automatic shading

def vignette(h=TH, w=TW, depth=0.4):
    """A centred illumination fall-off, 1 in the middle, mean 1."""
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    g = 1.0 - depth * r2 / 2.0
    return (g / g.mean()).astype(np.float32)


def shaded_fields(folder, positions, ch, tp, gain, level=2000, seed=3):
    """Plain textured fields with ``gain`` multiplied in, one per
    position: {index: {ch: [path]}}."""
    rng = np.random.default_rng(seed + tp)
    files = {}
    for i, p in enumerate(positions):
        true = level + rng.normal(0, level * 0.03, size=gain.shape)
        img = np.clip(true * gain, 0, 65535).astype(np.uint16)
        path = os.path.join(folder, f'{p["site"]}_{ch}_t{tp:04d}.tiff')
        tifffile.imwrite(path, img)
        files[i] = {ch: [path]}
    return files


class FakeSaver:
    def __init__(self):
        self.frames, self.closed = [], False

    def save(self, frame, **kw):
        self.frames.append((frame, kw))

    def close(self):
        self.closed = True


def fake_quark(gain):
    def flat_field(tiles):
        assert len(tiles) >= op.MIN_SHADING_FIELDS
        return gain, {'fields_used': len(tiles), 'fields_read': len(tiles)}

    def apply_gain(tile, g):
        return np.clip(np.rint(tile / g), 0, 65535).astype(np.uint16)
    return flat_field, apply_gain


def separate_positions(n, well='C4'):
    return [{'well': well, 'site': f'P{i + 1}', 'x_um': i * 5000.0,
             'y_um': 0.0} for i in range(n)]


def test_auto_shading_waits_for_the_first_timepoint_then_corrects(tmp_path):
    gain = vignette()
    pos = separate_positions(6)
    files = shaded_fields(str(tmp_path), pos, 'BF', 1, gain)
    saver = FakeSaver()
    ff, ap = fake_quark(gain)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=True,
                              stitch=False, px_per_um=PX, um_per_px=1.0,
                              shading='auto', flat_field=ff, apply_gain=ap,
                              corrected_saver=saver)
    for i in range(len(pos) - 1):
        proc.position_done(i, 1, files[i])
    assert proc._q.empty()              # held until every field is in
    proc.position_done(len(pos) - 1, 1, files[len(pos) - 1])
    run(proc)
    assert np.allclose(np.load(tmp_path / 'shading' / 'gain_BF.npy'), gain)
    assert (tmp_path / 'shading' / 'shading.json').exists()
    out = tifffile.imread(str(tmp_path / 'composites' / 'C4' /
                              'C4_P1_composite_t0001.tiff'))
    raw = tifffile.imread(files[0]['BF'][0])
    corner, middle = (slice(0, 6), slice(0, 6)), (slice(27, 33), slice(37, 43))
    # the raw field is darker in the corner; the composite is not
    assert raw[corner].mean() < 0.8 * raw[middle].mean()
    assert abs(out[corner].mean() / out[middle].mean() - 1) < 0.05
    assert len(saver.frames) == 6 and saver.closed
    frame, kw = saver.frames[0]
    assert kw['well'] == 'C4' and kw['channel'] == 'BF' and \
        kw['timepoint'] == 1 and kw['z_index'] == 1
    assert np.array_equal(frame, ap(raw, gain))
    assert np.array_equal(tifffile.imread(files[0]['BF'][0]), raw)  # raw kept
    assert proc.counts['corrected'] == 6
    assert 'shading measured for BF' in proc.summary()


def test_auto_shading_with_too_few_fields_leaves_images_as_stored(tmp_path):
    pos = separate_positions(2)
    files = shaded_fields(str(tmp_path), pos, 'BF', 1, vignette())
    saver = FakeSaver()
    ff, ap = fake_quark(vignette())
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=True,
                              stitch=False, px_per_um=PX, um_per_px=1.0,
                              shading='auto', flat_field=ff, apply_gain=ap,
                              corrected_saver=saver)
    for i in range(len(pos)):
        proc.position_done(i, 1, files[i])
    run(proc)
    assert proc.gains['BF'] is None
    assert 'not measured' in proc.shading_text['BF']
    out = tifffile.imread(str(tmp_path / 'composites' / 'C4' /
                              'C4_P1_composite_t0001.tiff'))
    assert np.array_equal(out, tifffile.imread(files[0]['BF'][0]))
    assert 'NO colour' in proc.summary()


def test_a_run_stopped_in_its_first_timepoint_still_gets_its_products(
        tmp_path):
    gain = vignette()
    pos = separate_positions(8)
    files = shaded_fields(str(tmp_path), pos, 'BF', 1, gain)
    ff, ap = fake_quark(gain)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['BF'], compose=True,
                              stitch=False, px_per_um=PX, um_per_px=1.0,
                              shading='auto', flat_field=ff, apply_gain=ap)
    for i in range(5):                   # stopped after 5 of 8
        proc.position_done(i, 1, files[i])
    run(proc)
    assert proc.counts['composites'] == 5
    assert proc.gains['BF'] is not None


def test_reference_shading_does_nothing_in_the_processor(tmp_path):
    proc = op.OnlineProcessor(str(tmp_path), [], ['BF'], compose=True,
                              stitch=False, px_per_um=PX, um_per_px=1.0,
                              shading='reference')
    assert not proc.auto and proc._measured
    run(proc)
    with pytest.raises(ValueError):
        op.OnlineProcessor(str(tmp_path), [], ['BF'], compose=True,
                           stitch=False, px_per_um=PX, um_per_px=1.0,
                           shading='sometimes')


def test_auto_shading_with_quark_divides_out_the_vignette(tmp_path):
    """The real Quark module (PyO3), as the Quark app uses on Etaluma
    plates: the pattern measured from plain fields and divided out."""
    import quark_stitch
    if not quark_stitch.available():
        pytest.skip(quark_stitch.why_unavailable())
    gain = vignette()
    pos = separate_positions(16)
    files = shaded_fields(str(tmp_path), pos, 'PC', 1, gain)
    proc = op.OnlineProcessor(str(tmp_path), pos, ['PC'], compose=True,
                              stitch=False, px_per_um=PX, um_per_px=1.0,
                              shading='auto')
    for i in range(len(pos)):
        proc.position_done(i, 1, files[i])
    run(proc)
    measured = proc.gains['PC']
    assert measured is not None, proc.shading_text
    assert np.abs(measured / gain - 1).mean() < 0.03
    out = tifffile.imread(str(tmp_path / 'composites' / 'C4' /
                              'C4_P3_composite_t0001.tiff')).astype(float)
    raw = tifffile.imread(files[2]['PC'][0]).astype(float)
    assert out.std() < 0.5 * raw.std()     # the fall-off is gone
