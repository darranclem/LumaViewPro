"""Quark's tile stitching and shading (flat-field) correction, for LumaViewPro.

Quark is a separate program (Rust). Its stitching and shading code is used here as it is, not
re-written in Python: the compiled module ``quark_tiles`` (built from Quark's ``quark-py``
crate with PyO3) is loaded and its functions wrapped so that they take and return NumPy
arrays. One module file serves every Python from 3.9 on.

What it does that LumaViewPro's own code does not:

* **Shading without a reference image.** The illumination pattern every tile shares is
  measured from the tiles themselves (the per-pixel median over the plain ones) and divided
  out. No empty-field reference has to be captured first.
* **Stitching by the overlaps.** Tiles start at their stage positions and are moved only
  where their overlaps agree better elsewhere; overlaps are blended instead of one tile
  overwriting the other.

Where the module is looked for: ``lib/quark/quark_tiles.pyd`` in this folder (not in the
repository: it is a compiled file, copied in or built with ``tools/build_quark_tiles.ps1``),
then anywhere on ``sys.path``. Without it every function here raises ``QuarkUnavailable``
with instructions, and LumaViewPro carries on with its own stitching.

Folders it reads:

* a **tile scan** written by ``acquisition.run_tile_scan``: ``tile_NNNN_<channel>.tiff`` with
  ``tile_scan.json`` (the stage position of every tile and the pixel size);
* a **plate protocol** folder: ``experiment.json`` and one sub-folder per well holding
  ``<well>_<site>_Z<n>_<channel>_t<n>.tiff``.

Command line (writes ``<name>_stitched.ome.tif`` per well into ``<folder>/quark``)::

    .venv\\Scripts\\python quark_stitch.py <folder> [--well D5] [--out DIR] [--reduce 2]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

TILE_SCAN_SIDECAR = 'tile_scan.json'
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_DIR = os.path.join(_HERE, 'lib', 'quark')
_module = None
_why = ''


class QuarkUnavailable(RuntimeError):
    """The compiled Quark module is not installed on this computer."""


def _load():
    """The ``quark_tiles`` module, or None (the reason is kept for ``why_unavailable``)."""
    global _module, _why
    if _module is not None:
        return _module
    if os.path.isdir(_MODULE_DIR) and _MODULE_DIR not in sys.path:
        sys.path.insert(0, _MODULE_DIR)
    try:
        import quark_tiles
        _module = quark_tiles
        _why = ''
    except Exception as e:          # ImportError, or a DLL that will not load
        _why = (f'Quark stitching is not installed ({e}). Copy quark_tiles.pyd into '
                f'{_MODULE_DIR}, or build it with tools\\build_quark_tiles.ps1.')
    return _module


def available() -> bool:
    """Whether Quark's stitching can be used on this computer."""
    return _load() is not None


def why_unavailable() -> str:
    """Why not (empty when it is available)."""
    _load()
    return _why


def version() -> str:
    return _need().version()


def _need():
    m = _load()
    if m is None:
        raise QuarkUnavailable(_why)
    return m


# ------------------------------------------------------------------ folders

def scan(folder: str) -> dict:
    """What a tile-scan or plate folder holds: name, wells (a tile scan is one well named
    ``scan``), channels, time points, tile size, stage pitch, notes."""
    return json.loads(_need().scan(os.fspath(folder)))


def flat_field(folder: str, channel: int = 0, time: int | None = None,
               z: int | None = None, max_fields: int = 64):
    """The illumination pattern the tiles of ``channel`` share, measured from the folder's
    own tiles. Returns ``(gain, info)``: ``gain`` is float32 (rows, columns) with mean 1,
    ``stored = gain x true``; ``info`` says how many tiles were plain enough to use and how
    well the pattern is known (``split_half_rms``, 0.004 = 0.4 %)."""
    raw, w, h, info = _need().flat_field(os.fspath(folder), channel, time, z, max_fields)
    return np.frombuffer(raw, dtype='<f4').reshape(h, w).copy(), json.loads(info)


def stitch_well(folder: str, well: str = 'scan', channels: list[int] | None = None,
                time: int | None = None, z: int | None = None, reduce: int = 1,
                pixel_um: float | None = None, shading: bool = True,
                register: bool = True):
    """One well (or the tile scan) stitched as Quark opens it.

    Returns ``(mosaic, info)``: ``mosaic`` is uint16 (channels, rows, columns); ``info``
    states the pixel size used and where it came from, what the shading correction found
    for each channel, what the registration found, and any tile that was missing.

    ``channels``: indices into ``scan(folder)['channels']`` (default all). ``reduce``: each
    output pixel is the mean of ``reduce x reduce`` stored ones. ``pixel_um``: micrometres
    per stored pixel (default: the scan's own, else measured from the plate, else the size
    at which the tiles abut). ``shading`` / ``register``: switch either step off.
    """
    raw, w, h, c, info = _need().stitch_well(os.fspath(folder), well, channels, time, z,
                                             max(int(reduce), 1), pixel_um, shading, register)
    return np.frombuffer(raw, dtype='<u2').reshape(c, h, w).copy(), json.loads(info)


def export_well(folder: str, well: str = 'scan', out_dir: str | None = None,
                stem: str | None = None, **kwargs):
    """``stitch_well`` written as an OME-TIFF (channel names and pixel size in its metadata).
    Returns ``(path, info)``. ``out_dir`` defaults to ``<folder>/quark``."""
    out_dir = os.fspath(out_dir) if out_dir else os.path.join(os.fspath(folder), 'quark')
    path, info = _need().export_well(os.fspath(folder), well, out_dir, stem,
                                     kwargs.get('channels'), kwargs.get('time'),
                                     kwargs.get('z'), max(int(kwargs.get('reduce', 1)), 1),
                                     kwargs.get('pixel_um'), kwargs.get('shading', True),
                                     kwargs.get('register', True))
    return path, json.loads(info)


def export_folder(folder: str, out_dir: str | None = None, wells: list[str] | None = None,
                  status_cb=None, cancel=None, **kwargs) -> list[str]:
    """Every well of a folder (or the wells named) stitched and written, one OME-TIFF each.
    ``status_cb(text)`` is told what is being done; ``cancel`` is a ``threading.Event``."""
    listed = scan(folder)
    names = wells or [w['name'] for w in listed['wells']]
    written = []
    for i, name in enumerate(names):
        if cancel is not None and cancel.is_set():
            break
        if status_cb:
            status_cb(f'Quark stitching {name} ({i + 1}/{len(names)})...')
        path, _ = export_well(folder, name, out_dir, **kwargs)
        written.append(path)
    return written


# ------------------------------------------------------------- in memory

def flat_field_from_tiles(tiles: list[np.ndarray]):
    """The shared illumination pattern of tiles in memory (same shape, 2-D, at least four
    of them plain). Returns ``(gain, info)`` as ``flat_field`` does."""
    if not tiles:
        raise ValueError('no tiles')
    h, w = tiles[0].shape[:2]
    raw, info = _need().flat_field_from_tiles(
        [np.ascontiguousarray(t, dtype='<u2').tobytes() for t in tiles], w, h)
    return np.frombuffer(raw, dtype='<f4').reshape(h, w).copy(), json.loads(info)


def apply_flat_field(tile: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """A tile with the illumination divided out (``true = stored / gain``), uint16."""
    if tile.shape[:2] != gain.shape[:2]:
        raise ValueError(f'tile {tile.shape} and gain {gain.shape} differ in size')
    raw = _need().apply_flat_field(np.ascontiguousarray(tile, dtype='<u2').tobytes(),
                                   np.ascontiguousarray(gain, dtype='<f4').tobytes())
    return np.frombuffer(raw, dtype='<u2').reshape(tile.shape).copy()


# -------------------------------------------------------------- tile scans

def write_tile_scan_record(out_dir: str, positions, um_per_px: float,
                           channels: list[str], pattern: str = '') -> str:
    """The record that makes a tile-scan folder readable later (by Quark and by this module):
    each tile's stage position, the pixel size and the channels. Plain JSON; needs no Quark.
    ``positions``: ``[(tile index, x_um, y_um), ...]`` for the tiles that were captured."""
    record = {
        'format': 'lvp-tile-scan-1',
        'um_per_px': float(um_per_px),
        'channels': list(channels),
        'pattern': pattern,
        'positions': [{'index': int(i), 'x_um': float(x), 'y_um': float(y)}
                      for (i, x, y) in positions],
    }
    path = os.path.join(out_dir, TILE_SCAN_SIDECAR)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)
    return path


def stitch_tile_scan(out_dir: str, channel_names: list[str], status_cb=None):
    """Stitch a finished tile scan with Quark: ``{channel: mosaic}`` (uint16, 2-D) and what
    was found (``{channel: info}``). Channels Quark could not do are left out, with the
    reason in ``info[channel]['error']``."""
    listed = scan(out_dir)
    mosaics, infos = {}, {}
    for ch in channel_names:
        if ch not in listed['channels']:
            continue
        if status_cb:
            status_cb(f'Stitching {ch} with shading correction (Quark)...')
        try:
            image, info = stitch_well(out_dir, 'scan', channels=[listed['channels'].index(ch)])
            mosaics[ch] = image[0]
            infos[ch] = info
        except Exception as e:      # this channel falls back to the plain stitch
            infos[ch] = {'error': str(e)}
    return mosaics, infos


def describe(info: dict) -> str:
    """One line for the status bar from a stitch's ``info``."""
    if 'error' in info:
        return f"Quark: {info['error']}"
    parts = []
    for s in info.get('shading', []):
        if s.get('corrected'):
            parts.append(f"{s['channel']}: shading measured from {s['fields_used']} of "
                         f"{s['fields_read']} tiles (gain {s['gain_min']:.2f} to {s['gain_max']:.2f})")
        else:
            parts.append(f"{s['channel']}: shading not corrected ({s.get('why', '')})")
    reg = (info.get('plate') or {}).get('registration') or {}
    if reg:
        parts.append('positions refined from the overlaps' if reg.get('applied')
                     else 'stage positions kept')
    if info.get('pixel_um'):
        parts.append(f"{info['pixel_um']:.4f} um/px ({info.get('pixel_source', '')})")
    return '; '.join(parts)


def _main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description='Stitch a LumaViewPro tile scan or plate folder with Quark.')
    p.add_argument('folder')
    p.add_argument('--well', action='append', help='well name (repeat for several; default all)')
    p.add_argument('--out', help='output folder (default <folder>/quark)')
    p.add_argument('--reduce', type=int, default=1, help='average NxN pixels (default 1)')
    p.add_argument('--no-shading', action='store_true')
    p.add_argument('--no-register', action='store_true')
    a = p.parse_args(argv)
    if not available():
        print(why_unavailable())
        return 2
    for path in export_folder(a.folder, a.out, a.well, status_cb=print, reduce=a.reduce,
                              shading=not a.no_shading, register=not a.no_register):
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main(sys.argv[1:]))
