"""One file per well: every tile, z plane, colour and timepoint of a well
in a single OME-TIFF (multi-page BigTIFF) or OME-Zarr store, instead of one
single-page TIFF per image.

Two ways in:

- :class:`WellWriter` takes frames as they are acquired, in any order
  (``add(frame, tile=, z=, channel=, timepoint=, ...)``), and finishes the
  file's metadata on ``close()``.
- :func:`pack_run` / :func:`pack_zstack` convert a folder of single-page
  TIFFs that a protocol run or a live z-stack already wrote
  (``B3_TA1_Z001_GREEN_t0001.tiff``, ``zstack_BF_000_6448.2um.tiff``).

Layout, the same for both formats: each TILE (position inside the well) is
one image with the axes time, colour, z, y, x; the tile's stage position is
recorded with it, so a reader can show one tile or stitch them.

OME-TIFF: pages are written in the order the frames arrive (nothing is
buffered); the OME-XML, written when the file is closed, says which page is
which (``TiffData`` with ``IFD``, ``FirstT/Z/C``), one ``Image`` per tile
with ``StageLabel`` and per-plane positions and times. If the program dies
before ``close()``, ``<file>.index.json`` (appended after every frame)
still says what every page is.

OME-Zarr (v2 arrays, NGFF 0.4 metadata): ``<well>.zarr/<tile>/0`` is the
tile's ``t, c, z, y, x`` array, one chunk per plane, zlib-compressed; the
store's root lists the tiles as the fields of a well.

No hardware and no UI in here. Needs ``numpy``; OME-TIFF needs ``tifffile``.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from xml.sax.saxutils import quoteattr

import numpy as np

try:
    import tifffile
except Exception:                                   # pragma: no cover
    tifffile = None

FORMATS = ('ome-tiff', 'zarr')
EXTENSION = {'ome-tiff': '.ome.tiff', 'zarr': '.zarr'}
VERSION = 1

# Display colours (RGB 0..1), the same as the live view.
CHANNEL_RGB = {
    'BF': (1.0, 1.0, 1.0), 'PC': (1.0, 1.0, 1.0), 'DF': (1.0, 1.0, 1.0),
    'BLUE': (0.0, 0.0, 1.0), 'GREEN': (0.0, 1.0, 0.0), 'RED': (1.0, 0.0, 0.0),
}


def well_path(folder: str, well: str | None, fmt: str) -> str:
    """``<folder>/<well>.ome.tiff`` (or ``.zarr``); ``well`` None or empty
    (a dish, a slide) gives ``image``."""
    if fmt not in FORMATS:
        raise ValueError(f'format must be one of {FORMATS}, not {fmt!r}')
    name = re.sub(r'[^A-Za-z0-9_-]+', '', str(well or '')) or 'image'
    return os.path.join(folder, name + EXTENSION[fmt])


class WellWriter:
    """Writes one well. Frames are 2-D uint8 or uint16, all the same size.

    ``tile``, ``channel`` are names; ``z`` and ``timepoint`` count from 0.
    Tiles, colours, planes and timepoints are discovered as they arrive:
    nothing has to be declared first, and a colour may have fewer planes
    than another (a colour that was not stacked).
    """

    def __init__(self, path: str, fmt: str = 'ome-tiff', *,
                 um_per_px: float = 0.0, z_step_um: float = 0.0,
                 well: str | None = None, info: dict | None = None):
        if fmt not in FORMATS:
            raise ValueError(f'format must be one of {FORMATS}, not {fmt!r}')
        if fmt == 'ome-tiff' and tifffile is None:
            raise RuntimeError('OME-TIFF needs the tifffile package')
        self.path = path
        self.fmt = fmt
        self.um_per_px = float(um_per_px or 0.0)
        self.z_step_um = float(z_step_um or 0.0)
        self.well = well
        self.info = dict(info or {})
        self.tiles: list[str] = []
        self.channels: list[str] = []
        self.tile_xy: dict[str, tuple[float, float]] = {}
        # One record per frame, in page order.
        self.planes: list[dict] = []
        self.shape: tuple[int, int] | None = None
        self.dtype: np.dtype | None = None
        self.closed = False
        self._seen: set[tuple] = set()
        if fmt == 'ome-tiff':
            self._tw = tifffile.TiffWriter(path, bigtiff=True)
            self._index = open(path + '.index.json', 'w', encoding='utf-8')
        else:
            os.makedirs(path, exist_ok=True)
            self._tw = None
            self._index = None

    # ------------------------------------------------------------ frames

    def add(self, frame: np.ndarray, *, tile: str = 'P1', z: int = 0,
            channel: str = 'BF', timepoint: int = 0,
            x_um: float | None = None, y_um: float | None = None,
            z_um: float | None = None, t_s: float | None = None,
            exposure_ms: float | None = None) -> None:
        if self.closed:
            raise RuntimeError('the well file is closed')
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.dtype not in (np.uint8, np.uint16):
            raise ValueError('frames must be 2-D uint8 or uint16, not '
                             f'{frame.shape} {frame.dtype}')
        if self.shape is None:
            self.shape, self.dtype = frame.shape, frame.dtype
        elif frame.shape != self.shape or frame.dtype != self.dtype:
            raise ValueError(f'frame is {frame.shape} {frame.dtype}; the well '
                             f'is {self.shape} {self.dtype}')
        tile, channel = str(tile), str(channel)
        key = (tile, int(timepoint), channel, int(z))
        if key in self._seen:
            raise ValueError(f'plane written twice: tile {tile}, timepoint '
                             f'{timepoint}, {channel}, z {z}')
        self._seen.add(key)
        if tile not in self.tiles:
            self.tiles.append(tile)
        if channel not in self.channels:
            self.channels.append(channel)
        if x_um is not None and y_um is not None:
            self.tile_xy.setdefault(tile, (float(x_um), float(y_um)))
        rec = {'page': len(self.planes), 'tile': tile, 't': int(timepoint),
               'c': channel, 'z': int(z)}
        for name, value in (('x_um', x_um), ('y_um', y_um), ('z_um', z_um),
                            ('t_s', t_s), ('exposure_ms', exposure_ms)):
            if value is not None:
                rec[name] = float(value)
        if self.fmt == 'ome-tiff':
            kwargs: dict = {}
            if self.um_per_px > 0:
                kwargs['resolution'] = (1e4 / self.um_per_px,
                                        1e4 / self.um_per_px)
                kwargs['resolutionunit'] = 'CENTIMETER'
            self._tw.write(np.ascontiguousarray(frame),
                           photometric='minisblack', metadata=None,
                           contiguous=False,
                           description='OME-XML is written when the file is '
                                       'closed' if not self.planes else None,
                           **kwargs)
            self._index.write(json.dumps(rec) + '\n')
            self._index.flush()
        else:
            ti = self.tiles.index(tile)
            ci = self.channels.index(channel)
            chunk = os.path.join(self.path, tile, '0', str(int(timepoint)),
                                 str(ci), str(int(z)), '0')
            os.makedirs(chunk, exist_ok=True)
            data = np.ascontiguousarray(frame).astype(
                frame.dtype.newbyteorder('<'), copy=False).tobytes()
            with open(os.path.join(chunk, '0'), 'wb') as f:
                f.write(zlib.compress(data, 1))
            rec['tile_index'] = ti
        self.planes.append(rec)

    # ------------------------------------------------------------- sizes

    def _sizes(self, tile: str) -> tuple[int, int, int]:
        mine = [p for p in self.planes if p['tile'] == tile]
        n_t = 1 + max((p['t'] for p in mine), default=0)
        n_z = 1 + max((p['z'] for p in mine), default=0)
        return n_t, len(self.channels), n_z

    def summary(self) -> dict:
        return {'path': self.path, 'format': self.fmt, 'well': self.well,
                'tiles': list(self.tiles), 'channels': list(self.channels),
                'planes': len(self.planes),
                'timepoints': 1 + max((p['t'] for p in self.planes), default=0),
                'z_planes': 1 + max((p['z'] for p in self.planes), default=0)}

    # ------------------------------------------------------------- close

    def close(self) -> dict:
        if self.closed:
            return self.summary()
        self.closed = True
        if self.fmt == 'ome-tiff':
            self._tw.close()
            self._index.close()
            if self.planes:
                tifffile.tiffcomment(self.path, self._ome_xml().encode('utf-8'))
                # The index was the safeguard against a crash: the OME-XML
                # now says the same.
                try:
                    os.remove(self.path + '.index.json')
                except OSError:
                    pass
            else:
                for p in (self.path, self.path + '.index.json'):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        else:
            self._zarr_metadata()
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------- OME-XML

    def _ome_xml(self) -> str:
        h, w = self.shape
        ptype = 'uint8' if self.dtype == np.uint8 else 'uint16'
        bits = 8 if self.dtype == np.uint8 else 16
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
               'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
               'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
               'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd" '
               'Creator="LumaViewPro well_file">']
        for ti, tile in enumerate(self.tiles):
            n_t, n_c, n_z = self._sizes(tile)
            name = f'{self.well} {tile}' if self.well else tile
            out.append(f'<Image ID="Image:{ti}" Name={quoteattr(name)}>')
            xy = self.tile_xy.get(tile)
            # Always named (the tile's name is how a reader tells tiles
            # apart); the position only when the stage position is known.
            where = (f' X="{xy[0]:.3f}" XUnit="µm" Y="{xy[1]:.3f}" YUnit="µm"'
                     if xy is not None else '')
            out.append(f'<StageLabel Name={quoteattr(tile)}{where}/>')
            phys = ''
            if self.um_per_px > 0:
                phys += (f' PhysicalSizeX="{self.um_per_px:.6g}" PhysicalSizeXUnit="µm"'
                         f' PhysicalSizeY="{self.um_per_px:.6g}" PhysicalSizeYUnit="µm"')
            if self.z_step_um > 0 and n_z > 1:
                phys += f' PhysicalSizeZ="{self.z_step_um:.6g}" PhysicalSizeZUnit="µm"'
            out.append(f'<Pixels ID="Pixels:{ti}" DimensionOrder="XYZCT" Type="{ptype}" '
                       f'SignificantBits="{bits}" BigEndian="false" Interleaved="false" '
                       f'SizeX="{w}" SizeY="{h}" SizeZ="{n_z}" SizeC="{n_c}" SizeT="{n_t}"{phys}>')
            for ci, ch in enumerate(self.channels):
                r, g, b = CHANNEL_RGB.get(ch.upper(), (1.0, 1.0, 1.0))
                rgba = (int(r * 255) << 24) | (int(g * 255) << 16) | (int(b * 255) << 8) | 255
                if rgba >= 1 << 31:
                    rgba -= 1 << 32
                out.append(f'<Channel ID="Channel:{ti}:{ci}" Name={quoteattr(ch)} '
                           f'SamplesPerPixel="1" Color="{rgba}"/>')
            mine = [p for p in self.planes if p['tile'] == tile]
            for p in mine:
                ci = self.channels.index(p['c'])
                out.append(f'<TiffData IFD="{p["page"]}" FirstC="{ci}" FirstZ="{p["z"]}" '
                           f'FirstT="{p["t"]}" PlaneCount="1"/>')
            for p in mine:
                ci = self.channels.index(p['c'])
                extra = ''
                if 't_s' in p:
                    extra += f' DeltaT="{p["t_s"]:.3f}" DeltaTUnit="s"'
                if 'exposure_ms' in p:
                    extra += f' ExposureTime="{p["exposure_ms"]:.4g}" ExposureTimeUnit="ms"'
                if 'x_um' in p and 'y_um' in p:
                    extra += (f' PositionX="{p["x_um"]:.3f}" PositionXUnit="µm"'
                              f' PositionY="{p["y_um"]:.3f}" PositionYUnit="µm"')
                if 'z_um' in p:
                    extra += f' PositionZ="{p["z_um"]:.3f}" PositionZUnit="µm"'
                out.append(f'<Plane TheC="{ci}" TheZ="{p["z"]}" TheT="{p["t"]}"{extra}/>')
            out.append('</Pixels></Image>')
        note = dict(self.info)
        note.update({'lumaviewpro_well_file': VERSION, 'well': self.well or '',
                     'tiles': ','.join(self.tiles)})
        out.append('<StructuredAnnotations><MapAnnotation ID="Annotation:0" '
                   'Namespace="lumaviewpro/well_file"><Value>')
        for k, v in note.items():
            out.append(f'<M K={quoteattr(str(k))}>{_text(v)}</M>')
        out.append('</Value></MapAnnotation></StructuredAnnotations></OME>')
        return ''.join(out)

    # -------------------------------------------------------------- Zarr

    def _zarr_metadata(self) -> None:
        def put(path: str, value: dict) -> None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(value, f, indent=1)

        h, w = self.shape if self.shape else (0, 0)
        dtype = '|u1' if self.dtype == np.uint8 else '<u2'
        put(os.path.join(self.path, '.zgroup'), {'zarr_format': 2})
        put(os.path.join(self.path, '.zattrs'), {
            'well': {'images': [{'path': t, 'acquisition': 0} for t in self.tiles],
                     'version': '0.4'},
            'lumaviewpro_well_file': {
                'version': VERSION, 'well': self.well or '', 'info': self.info,
                'tiles': [{'name': t, 'x_um': self.tile_xy.get(t, (None, None))[0],
                           'y_um': self.tile_xy.get(t, (None, None))[1]} for t in self.tiles],
                'planes': self.planes},
        })
        for tile in self.tiles:
            n_t, n_c, n_z = self._sizes(tile)
            scale = [1.0, 1.0, self.z_step_um or 1.0, self.um_per_px or 1.0, self.um_per_px or 1.0]
            unit = 'micrometer' if self.um_per_px > 0 else None
            axes = [{'name': 't', 'type': 'time'}, {'name': 'c', 'type': 'channel'},
                    {'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'},
                    {'name': 'x', 'type': 'space'}]
            if unit:
                for a in axes[2:]:
                    a['unit'] = unit
            xy = self.tile_xy.get(tile)
            transforms = [{'type': 'scale', 'scale': scale}]
            if xy is not None and self.um_per_px > 0:
                transforms.append({'type': 'translation',
                                   'translation': [0.0, 0.0, 0.0, xy[1], xy[0]]})
            group = os.path.join(self.path, tile)
            os.makedirs(os.path.join(group, '0'), exist_ok=True)
            put(os.path.join(group, '.zgroup'), {'zarr_format': 2})
            put(os.path.join(group, '.zattrs'), {
                'multiscales': [{'version': '0.4', 'name': tile, 'axes': axes,
                                 'datasets': [{'path': '0',
                                               'coordinateTransformations': transforms}]}],
                'omero': {'channels': [
                    {'label': ch, 'active': True,
                     'color': '%02X%02X%02X' % tuple(int(v * 255) for v in CHANNEL_RGB.get(ch.upper(), (1.0, 1.0, 1.0))),
                     'window': {'start': 0, 'end': 255 if self.dtype == np.uint8 else 4095,
                                'min': 0, 'max': 255 if self.dtype == np.uint8 else 65535}}
                    for ch in self.channels]},
            })
            put(os.path.join(group, '0', '.zarray'), {
                'zarr_format': 2, 'shape': [n_t, n_c, n_z, h, w],
                'chunks': [1, 1, 1, h, w], 'dtype': dtype, 'order': 'C',
                'compressor': {'id': 'zlib', 'level': 1}, 'fill_value': 0,
                'filters': None, 'dimension_separator': '/'})


def _text(value) -> str:
    return (str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ------------------------------------------------------------- reading

def read_well(path: str) -> dict:
    """What a well file holds: ``{'tiles': {name: array[t, c, z, y, x]},
    'channels': [...], 'tile_xy': {name: (x_um, y_um)}}``. For checks and
    small wells: everything is read into memory."""
    if path.lower().endswith('.zarr'):
        with open(os.path.join(path, '.zattrs'), encoding='utf-8') as f:
            attrs = json.load(f)
        mine = attrs['lumaviewpro_well_file']
        out = {'tiles': {}, 'channels': None,
               'tile_xy': {t['name']: (t['x_um'], t['y_um']) for t in mine['tiles']}}
        for image in attrs['well']['images']:
            tile = image['path']
            with open(os.path.join(path, tile, '0', '.zarray'), encoding='utf-8') as f:
                meta = json.load(f)
            with open(os.path.join(path, tile, '.zattrs'), encoding='utf-8') as f:
                out['channels'] = [c['label'] for c in json.load(f)['omero']['channels']]
            n_t, n_c, n_z, h, w = meta['shape']
            arr = np.zeros(meta['shape'], np.dtype(meta['dtype']))
            for t in range(n_t):
                for c in range(n_c):
                    for z in range(n_z):
                        chunk = os.path.join(path, tile, '0', str(t), str(c), str(z), '0', '0')
                        if os.path.exists(chunk):
                            with open(chunk, 'rb') as f:
                                arr[t, c, z] = np.frombuffer(zlib.decompress(f.read()), arr.dtype).reshape(h, w)
            out['tiles'][tile] = arr
        return out
    if tifffile is None:
        raise RuntimeError('OME-TIFF needs the tifffile package')
    import xml.etree.ElementTree as ET
    with tifffile.TiffFile(path) as tf:
        root = ET.fromstring(tf.pages[0].description)
        ns = {'o': root.tag.split('}')[0].strip('{')}
        out = {'tiles': {}, 'channels': None, 'tile_xy': {}}
        for image in root.findall('o:Image', ns):
            px = image.find('o:Pixels', ns)
            label = image.find('o:StageLabel', ns)
            tile = label.get('Name') if label is not None else image.get('Name')
            if label is not None and label.get('X') is not None:
                out['tile_xy'][tile] = (float(label.get('X')), float(label.get('Y')))
            out['channels'] = [c.get('Name') for c in px.findall('o:Channel', ns)]
            shape = tuple(int(px.get(k)) for k in ('SizeT', 'SizeC', 'SizeZ', 'SizeY', 'SizeX'))
            arr = np.zeros(shape, np.uint8 if px.get('Type') == 'uint8' else np.uint16)
            for td in px.findall('o:TiffData', ns):
                arr[int(td.get('FirstT')), int(td.get('FirstC')), int(td.get('FirstZ'))] = \
                    tf.pages[int(td.get('IFD'))].asarray()
            out['tiles'][tile] = arr
        return out


# ------------------------------------------- packing single-page folders

_RUN_NAME = re.compile(r'^(?:(?P<well>[A-Za-z]{1,2}\d{1,2})_)?(?P<site>[A-Za-z0-9-]+)_Z(?P<z>\d+)_'
                       r'(?P<ch>[A-Za-z0-9-]+)_t(?P<t>\d+)\.tiff?$')
_ZSTACK_NAME = re.compile(r'^zstack_(?P<ch>.+)_(?P<i>\d+)_(?P<z>-?\d+(?:\.\d+)?)um\.tiff?$', re.I)


def _read(path: str) -> np.ndarray:
    if tifffile is None:
        raise RuntimeError('reading the single-page files needs the tifffile package')
    return tifffile.imread(path)


def pack_zstack(folder: str, fmt: str = 'ome-tiff', out_path: str | None = None,
                um_per_px: float = 0.0, status_cb=None) -> dict:
    """A live z-stack folder (``zstack_<colour>_<number>_<height>um.tiff``)
    as ONE file beside it: ``<folder>.ome.tiff`` or ``<folder>.zarr``.
    Planes are ordered by stage height, lowest first."""
    found: dict[str, list[tuple[float, str]]] = {}
    for name in sorted(os.listdir(folder)):
        m = _ZSTACK_NAME.match(name)
        if m:
            found.setdefault(m.group('ch'), []).append((float(m.group('z')), os.path.join(folder, name)))
    if not found:
        raise ValueError(f'{folder}: no zstack_<colour>_<number>_<height>um.tiff files')
    for planes in found.values():
        planes.sort()
    first = next(iter(found.values()))
    steps = sorted(b[0] - a[0] for a, b in zip(first, first[1:]) if b[0] > a[0])
    step = (first[-1][0] - first[0][0]) / (len(first) - 1) if len(first) > 1 else 0.0
    if steps and abs(step - steps[len(steps) // 2]) > 0.11:
        step = steps[len(steps) // 2]
    out_path = out_path or folder.rstrip('/\\') + EXTENSION[fmt]
    total = sum(len(p) for p in found.values())
    done = 0
    with WellWriter(out_path, fmt, um_per_px=um_per_px, z_step_um=step,
                    info={'source': os.path.basename(folder.rstrip('/\\')), 'kind': 'z-stack'}) as w:
        for ch, planes in found.items():
            for zi, (z_um, path) in enumerate(planes):
                w.add(_read(path), tile='P1', z=zi, channel=ch, z_um=z_um)
                done += 1
                if status_cb:
                    status_cb(f'Packing z-stack: {done}/{total}')
    return w.summary()


def pack_run(folder: str, fmt: str = 'ome-tiff', out_dir: str | None = None,
             um_per_px: float = 0.0, z_step_um: float = 0.0,
             positions: dict | None = None, status_cb=None) -> list[dict]:
    """A protocol run's single-page files (``B3_TA1_Z001_GREEN_t0001.tiff``,
    in ``folder`` or one level of sub-folders) as one file per well in
    ``out_dir`` (default ``<folder>/wells``). ``positions`` may map
    ``(well, site)`` to ``(x_um, y_um)``. Returns each well's summary."""
    files: dict[str, list[tuple[str, int, str, int, str]]] = {}
    for root, _dirs, names in os.walk(folder):
        if os.path.abspath(root) != os.path.abspath(folder) and \
                os.path.dirname(os.path.abspath(root)) != os.path.abspath(folder):
            continue
        for name in sorted(names):
            m = _RUN_NAME.match(name)
            if m:
                files.setdefault(m.group('well') or '', []).append(
                    (m.group('site'), int(m.group('z')) - 1, m.group('ch'),
                     int(m.group('t')) - 1, os.path.join(root, name)))
    if not files:
        raise ValueError(f'{folder}: no <well>_<site>_Z<n>_<colour>_t<n>.tiff files')
    out_dir = out_dir or os.path.join(folder, 'wells')
    os.makedirs(out_dir, exist_ok=True)
    total = sum(len(v) for v in files.values())
    done, out = 0, []
    for well in sorted(files):
        with WellWriter(well_path(out_dir, well or None, fmt), fmt, um_per_px=um_per_px,
                        z_step_um=z_step_um, well=well or None,
                        info={'source': os.path.basename(folder.rstrip('/\\')), 'kind': 'protocol run'}) as w:
            # Timepoint by timepoint, as it was acquired.
            for site, z, ch, t, path in sorted(files[well], key=lambda f: (f[3], f[0], f[2], f[1])):
                xy = (positions or {}).get((well or None, site), (None, None))
                w.add(_read(path), tile=site, z=z, channel=ch, timepoint=t, x_um=xy[0], y_um=xy[1])
                done += 1
                if status_cb and done % 20 == 0:
                    status_cb(f'Packing wells: {done}/{total}')
        out.append(w.summary())
    return out


if __name__ == '__main__':                          # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description='Pack single-page TIFFs into one file per well or stack.')
    ap.add_argument('folder')
    ap.add_argument('--format', choices=FORMATS, default='ome-tiff')
    ap.add_argument('--um-per-px', type=float, default=0.0)
    ap.add_argument('--out')
    a = ap.parse_args()
    has_stack = any(_ZSTACK_NAME.match(n) for n in os.listdir(a.folder))
    if has_stack:
        print(json.dumps(pack_zstack(a.folder, a.format, a.out, a.um_per_px, print), indent=1))
    else:
        print(json.dumps(pack_run(a.folder, a.format, a.out, a.um_per_px, status_cb=print), indent=1))
