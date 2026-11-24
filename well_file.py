"""One file per well, or one for a whole run: every tile, z plane, colour
and timepoint in a single OME-TIFF (multi-page BigTIFF) or OME-Zarr store,
instead of one single-page TIFF per image.

Two ways in:

- :class:`WellWriter` takes frames as they are acquired, in any order
  (``add(frame, tile=, z=, channel=, timepoint=, well=, ...)``), and
  finishes the file's metadata on ``close()``.
- :func:`pack_run` / :func:`pack_zstack` convert a folder of single-page
  TIFFs that a protocol run or a live z-stack already wrote
  (``B3_TA1_Z001_GREEN_t0001.tiff``, ``zstack_BF_000_6448.2um.tiff``).

Layout, the same for both formats: each TILE (position inside a well) is
one image with the axes time, colour, z, y, x; the tile's stage position is
recorded with it, so a reader can show one tile or stitch them. A file that
holds a whole run (``run=True``) lists its wells as a plate.

Reduced copies (a pyramid): with every frame its halvings are stored (each
sample the mean of 2 x 2, 4 x 4, ... of the frame's), down to the last one
of 64 pixels or more a side, so that a reader shows a well or a whole plate
from the small copies without reading the frames.

The index: where every plane is (``WellWriter._index_doc``; for the
OME-TIFF the byte offset of the plane's samples and of each reduced copy,
which are stored uncompressed, little-endian, in one piece). A reader that
knows it opens the file by reading the index alone.

OME-TIFF: pages are written in the order the frames arrive (nothing is
buffered), reduced copies as SubIFDs of their page; the OME-XML, written
when the file is closed, says which page is which (``TiffData`` with
``IFD``, ``FirstT/Z/C``), one ``Image`` per tile with ``StageLabel`` and
per-plane positions and times, a ``Plate`` for a run, and the index as an
annotation. Until ``close()``, ``<file>.index.json`` (appended after every
frame) says the same: a run that is still going, or one that died, is read
from it.

OME-Zarr (v2 arrays, NGFF 0.4 metadata): ``<tile>/0`` is the tile's
``t, c, z, y, x`` array, ``<tile>/1``, ``/2``... its reduced copies, one
gzip chunk per plane; a well's store lists the tiles as the fields of a
well, a run's store is a plate (``<row>/<column>/<tile>``). The metadata is
rewritten at every new timepoint, so the store is readable during a run.

No hardware and no UI in here. Needs ``numpy``; OME-TIFF needs ``tifffile``.
"""

from __future__ import annotations

import json
import os
import re
import time
import gzip
from xml.sax.saxutils import quoteattr

import numpy as np

try:
    import tifffile
except Exception:                                   # pragma: no cover
    tifffile = None

# The OME unit for micrometres (the source stays ASCII).
_UM = chr(0xB5) + 'm'
FORMATS = ('ome-tiff', 'zarr')
EXTENSION = {'ome-tiff': '.ome.tiff', 'zarr': '.zarr'}
VERSION = 2
# Reduced copies are stored down to the last one this many pixels a side.
SMALLEST_LEVEL = 64
_WELL_NAME = re.compile(r'^([A-Za-z]{1,2})(\d{1,2})$')

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



def level_shapes(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """The sizes stored for a frame of ``shape``: itself, then each halving
    (2, 4, 8... to a side) that is still ``SMALLEST_LEVEL`` pixels a side."""
    h, w = int(shape[0]), int(shape[1])
    out, f = [(h, w)], 2
    while min(h, w) // f >= SMALLEST_LEVEL:
        out.append((h // f, w // f))
        f *= 2
    return out


def reduced(frame: np.ndarray, f: int) -> np.ndarray:
    """``frame`` with each sample the mean (rounded) of ``f x f`` of its
    own; rows and columns left over at the far edges are dropped."""
    h, w = frame.shape[0] // f, frame.shape[1] // f
    total = frame[:h * f, :w * f].reshape(h, f, w, f).sum((1, 3), dtype=np.uint64)
    return ((total + (f * f) // 2) // (f * f)).astype(frame.dtype)


class WellWriter:
    """Writes one well, or (``run=True``) every well of a run. Frames are
    2-D uint8 or uint16, all the same size.

    ``tile``, ``channel`` are names; ``z`` and ``timepoint`` count from 0.
    Wells, tiles, colours, planes and timepoints are discovered as they
    arrive: nothing has to be declared first, and a colour may have fewer
    planes than another (a colour that was not stacked).
    """

    def __init__(self, path: str, fmt: str = 'ome-tiff', *,
                 um_per_px: float = 0.0, z_step_um: float = 0.0,
                 well: str | None = None, info: dict | None = None,
                 run: bool = False, levels: bool = True):
        if fmt not in FORMATS:
            raise ValueError(f'format must be one of {FORMATS}, not {fmt!r}')
        if fmt == 'ome-tiff' and tifffile is None:
            raise RuntimeError('OME-TIFF needs the tifffile package')
        self.path = path
        self.fmt = fmt
        self.um_per_px = float(um_per_px or 0.0)
        self.z_step_um = float(z_step_um or 0.0)
        self.well = well
        self.run = bool(run)
        self.with_levels = bool(levels)
        self.info = dict(info or {})
        # One image per (well, tile), in the order they first arrived.
        self.images: list[tuple[str, str]] = []
        self.channels: list[str] = []
        self.image_xy: dict[int, tuple[float, float]] = {}
        # One record per frame, in page order.
        self.planes: list[dict] = []
        self.shape: tuple[int, int] | None = None
        self.dtype: np.dtype | None = None
        self.levels: list[tuple[int, int]] = []
        self.closed = False
        self._seen: set[tuple] = set()
        self._last_t = 0
        if fmt == 'ome-tiff':
            self._tw = tifffile.TiffWriter(path, bigtiff=True)
            self._index = open(path + '.index.json', 'w', encoding='utf-8')
        else:
            os.makedirs(path, exist_ok=True)
            self._tw = None
            self._index = None

    @property
    def tiles(self) -> list[str]:
        """The tile names, each once, in the order they first arrived."""
        out: list[str] = []
        for _well, tile in self.images:
            if tile not in out:
                out.append(tile)
        return out

    @property
    def wells(self) -> list[str]:
        out: list[str] = []
        for well, _tile in self.images:
            if well not in out:
                out.append(well)
        return out

    @property
    def tile_xy(self) -> dict[str, tuple[float, float]]:
        """Stage position by tile name (of a one-well file)."""
        return {self.images[i][1]: xy for i, xy in self.image_xy.items()}

    # ------------------------------------------------------------ frames

    def _header(self) -> dict:
        h, w = self.shape if self.shape else (0, 0)
        return {'version': VERSION,
                'scope': 'run' if self.run else 'well',
                'well': self.well or '', 'info': self.info,
                'shape': [h, w],
                'dtype': 'uint8' if self.dtype == np.uint8 else 'uint16',
                'um_per_px': self.um_per_px, 'z_step_um': self.z_step_um,
                'levels': [list(s) for s in self.levels]}

    def add(self, frame: np.ndarray, *, tile: str = 'P1', z: int = 0,
            channel: str = 'BF', timepoint: int = 0,
            x_um: float | None = None, y_um: float | None = None,
            z_um: float | None = None, t_s: float | None = None,
            exposure_ms: float | None = None, well: str | None = None) -> None:
        if self.closed:
            raise RuntimeError('the well file is closed')
        frame = np.asarray(frame)
        if frame.ndim != 2 or frame.dtype not in (np.uint8, np.uint16):
            raise ValueError('frames must be 2-D uint8 or uint16, not '
                             f'{frame.shape} {frame.dtype}')
        first = self.shape is None
        if first:
            self.shape, self.dtype = frame.shape, frame.dtype
            self.levels = level_shapes(frame.shape) if self.with_levels else [tuple(frame.shape)]
        elif frame.shape != self.shape or frame.dtype != self.dtype:
            raise ValueError(f'frame is {frame.shape} {frame.dtype}; the well '
                             f'is {self.shape} {self.dtype}')
        tile, channel = str(tile), str(channel)
        well = str(well if well is not None else (self.well or '')) if self.run else ''
        key = (well, tile, int(timepoint), channel, int(z))
        if key in self._seen:
            raise ValueError(f'plane written twice: tile {tile}, timepoint '
                             f'{timepoint}, {channel}, z {z}')
        self._seen.add(key)
        if (well, tile) not in self.images:
            self.images.append((well, tile))
        image = self.images.index((well, tile))
        if channel not in self.channels:
            self.channels.append(channel)
        if x_um is not None and y_um is not None:
            self.image_xy.setdefault(image, (float(x_um), float(y_um)))
        rec = {'page': len(self.planes), 'tile': tile, 't': int(timepoint),
               'c': channel, 'z': int(z)}
        if well:
            rec['well'] = well
        for name, value in (('x_um', x_um), ('y_um', y_um), ('z_um', z_um),
                            ('t_s', t_s), ('exposure_ms', exposure_ms)):
            if value is not None:
                rec[name] = float(value)
        frame = np.ascontiguousarray(frame)
        small = [reduced(frame, 2 ** k) for k in range(1, len(self.levels))]
        if self.fmt == 'ome-tiff':
            kwargs: dict = {}
            if self.um_per_px > 0:
                kwargs['resolution'] = (1e4 / self.um_per_px,
                                        1e4 / self.um_per_px)
                kwargs['resolutionunit'] = 'CENTIMETER'
            if small:
                kwargs['subifds'] = len(small)
            at = self._tw.write(frame,
                                photometric='minisblack', metadata=None,
                                contiguous=False, returnoffset=True,
                                description='OME-XML is written when the file is '
                                            'closed' if not self.planes else None,
                                **kwargs)
            offsets = [int(at[0])]
            for level in small:
                at = self._tw.write(level, photometric='minisblack', metadata=None,
                                    contiguous=False, returnoffset=True, subfiletype=1)
                offsets.append(int(at[0]))
            rec['offsets'] = offsets
            # On disk before the index says it is there (a reader may open
            # the file while the run goes on).
            try:
                self._tw._fh.flush()
            except Exception:
                pass
            if first:
                self._index.write(json.dumps({'header': self._header()}) + '\n')
            self._index.write(json.dumps(rec) + '\n')
            self._index.flush()
        else:
            ci = self.channels.index(channel)
            for k, plane in enumerate([frame] + small):
                chunk = os.path.join(self.path, self._image_path(image), str(k),
                                     str(int(timepoint)), str(ci), str(int(z)), '0')
                os.makedirs(chunk, exist_ok=True)
                data = plane.astype(plane.dtype.newbyteorder('<'), copy=False).tobytes()
                with open(os.path.join(chunk, '0'), 'wb') as f:
                    f.write(gzip.compress(data, 1, mtime=0))
        self.planes.append(rec)
        if self.fmt == 'zarr' and (first or int(timepoint) > self._last_t):
            # Readable while the run goes on: what is there so far is stated
            # at the start of every timepoint.
            self._zarr_metadata()
        self._last_t = max(self._last_t, int(timepoint))

    # ------------------------------------------------------------- sizes

    def _sizes(self, image: int) -> tuple[int, int, int]:
        well, tile = self.images[image]
        mine = [p for p in self.planes if p['tile'] == tile and p.get('well', '') == well]
        n_t = 1 + max((p['t'] for p in mine), default=0)
        n_z = 1 + max((p['z'] for p in mine), default=0)
        return n_t, len(self.channels), n_z

    def _plate_place(self, well: str) -> tuple[str, str]:
        """A well's row and column names in a run's Zarr store."""
        m = _WELL_NAME.match(well)
        if m:
            return m.group(1).upper(), str(int(m.group(2)))
        return re.sub(r'[^A-Za-z0-9]+', '', well) or 'X', '1'

    def _image_path(self, image: int) -> str:
        """The Zarr group of an image, below the store's root."""
        well, tile = self.images[image]
        if well:
            row, col = self._plate_place(well)
            return f'{row}/{col}/{tile}'
        return tile

    def _index_doc(self) -> dict:
        """Where everything is (module docstring): the header, the images
        (well, tile, stage position, Zarr group) and one row per plane,
        ``[image, colour, z, timepoint, offset of the samples, offsets of
        the reduced copies...]`` (offsets in the OME-TIFF only)."""
        doc = self._header()
        doc['channels'] = list(self.channels)
        doc['images'] = []
        for i, (well, tile) in enumerate(self.images):
            xy = self.image_xy.get(i, (None, None))
            entry = {'well': well, 'tile': tile, 'x_um': xy[0], 'y_um': xy[1]}
            if self.fmt == 'zarr':
                entry['path'] = self._image_path(i)
            doc['images'].append(entry)
        place = {img: i for i, img in enumerate(self.images)}
        chan = {c: i for i, c in enumerate(self.channels)}
        doc['planes'] = [[place[(p.get('well', ''), p['tile'])], chan[p['c']], p['z'], p['t']]
                         + list(p.get('offsets', ())) for p in self.planes]
        for name in ('z_um', 't_s', 'exposure_ms'):
            if any(name in p for p in self.planes):
                doc[name] = [p.get(name) for p in self.planes]
        return doc

    def summary(self) -> dict:
        return {'path': self.path, 'format': self.fmt, 'well': self.well,
                'wells': self.wells,
                'tiles': list(self.tiles), 'channels': list(self.channels),
                'planes': len(self.planes), 'levels': len(self.levels),
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
        elif self.planes:
            self._zarr_metadata()
        return self.summary()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ----------------------------------------------------------- OME-XML

    def _plate_xml(self) -> list[str]:
        """The wells of a run as an OME ``Plate`` (wells named as on a
        plate only: ``B3``)."""
        placed = []
        for i, (well, _tile) in enumerate(self.images):
            m = _WELL_NAME.match(well)
            if m:
                row = 0
                for ch in m.group(1).upper():
                    row = row * 26 + (ord(ch) - 64)
                placed.append((row - 1, int(m.group(2)) - 1, i))
        if not self.run or not placed:
            return []
        rows = 1 + max(p[0] for p in placed)
        cols = 1 + max(p[1] for p in placed)
        out = [f'<Plate ID="Plate:0" Name={quoteattr(os.path.basename(self.path))} '
               f'Rows="{rows}" Columns="{cols}" RowNamingConvention="letter" '
               'ColumnNamingConvention="number">']
        wells: dict[tuple[int, int], list[int]] = {}
        for row, col, i in placed:
            wells.setdefault((row, col), []).append(i)
        n = 0
        for wi, ((row, col), images) in enumerate(sorted(wells.items())):
            out.append(f'<Well ID="Well:0:{wi}" Row="{row}" Column="{col}">')
            for i in images:
                out.append(f'<WellSample ID="WellSample:0:{wi}:{n}" Index="{n}">'
                           f'<ImageRef ID="Image:{i}"/></WellSample>')
                n += 1
            out.append('</Well>')
        out.append('</Plate>')
        return out

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
        out.extend(self._plate_xml())
        by_image: dict[int, list[dict]] = {}
        for p in self.planes:
            by_image.setdefault(self.images.index((p.get('well', ''), p['tile'])), []).append(p)
        for ti, (well, tile) in enumerate(self.images):
            n_t, n_c, n_z = self._sizes(ti)
            well = well or self.well or ''
            name = f'{well} {tile}' if well else tile
            out.append(f'<Image ID="Image:{ti}" Name={quoteattr(name)}>')
            xy = self.image_xy.get(ti)
            # Always named (the tile's name is how a reader tells tiles
            # apart); the position only when the stage position is known.
            where = (f' X="{xy[0]:.3f}" XUnit="{_UM}" Y="{xy[1]:.3f}" YUnit="{_UM}"'
                     if xy is not None else '')
            out.append(f'<StageLabel Name={quoteattr(tile)}{where}/>')
            phys = ''
            if self.um_per_px > 0:
                phys += (f' PhysicalSizeX="{self.um_per_px:.6g}" PhysicalSizeXUnit="{_UM}"'
                         f' PhysicalSizeY="{self.um_per_px:.6g}" PhysicalSizeYUnit="{_UM}"')
            if self.z_step_um > 0 and n_z > 1:
                phys += f' PhysicalSizeZ="{self.z_step_um:.6g}" PhysicalSizeZUnit="{_UM}"'
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
            mine = by_image.get(ti, [])
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
                    extra += (f' PositionX="{p["x_um"]:.3f}" PositionXUnit="{_UM}"'
                              f' PositionY="{p["y_um"]:.3f}" PositionYUnit="{_UM}"')
                if 'z_um' in p:
                    extra += f' PositionZ="{p["z_um"]:.3f}" PositionZUnit="{_UM}"'
                out.append(f'<Plane TheC="{ci}" TheZ="{p["z"]}" TheT="{p["t"]}"{extra}/>')
            out.append('</Pixels></Image>')
        note = dict(self.info)
        note.update({'lumaviewpro_well_file': VERSION, 'well': self.well or '',
                     'tiles': ','.join(self.tiles)})
        out.append('<StructuredAnnotations><MapAnnotation ID="Annotation:0" '
                   'Namespace="lumaviewpro/well_file"><Value>')
        for k, v in note.items():
            out.append(f'<M K={quoteattr(str(k))}>{_text(v)}</M>')
        out.append('</Value></MapAnnotation>')
        # Where every plane lies, for a reader that opens the file from this
        # alone (module docstring).
        out.append('<MapAnnotation ID="Annotation:1" '
                   'Namespace="lumaviewpro/well_file/index"><Value><M K="index">')
        out.append(_text(json.dumps(self._index_doc(), separators=(',', ':'))))
        out.append('</M></Value></MapAnnotation></StructuredAnnotations></OME>')
        return ''.join(out)

    # -------------------------------------------------------------- Zarr

    def _zarr_metadata(self) -> None:
        def put(path: str, value: dict) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(value, f, separators=(',', ':'))
            # Windows can refuse the swap for a moment while another
            # program (virus scanner, indexer) has the file open.
            for attempt in range(20):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.05)

        dtype = '|u1' if self.dtype == np.uint8 else '<u2'
        doc = self._index_doc()
        root: dict = {'lumaviewpro_well_file': doc}
        plate_wells = [w_ for w_ in self.wells if w_]
        if plate_wells:
            places = {w_: self._plate_place(w_) for w_ in plate_wells}
            rows = sorted({p[0] for p in places.values()}, key=lambda v: (len(v), v))
            cols = sorted({p[1] for p in places.values()}, key=lambda v: (len(v), v))
            root['plate'] = {
                'version': '0.4', 'name': os.path.basename(_bare(self.path)),
                'acquisitions': [{'id': 0}],
                'field_count': max(sum(1 for im in self.images if im[0] == w_) for w_ in plate_wells),
                'rows': [{'name': r} for r in rows], 'columns': [{'name': c} for c in cols],
                'wells': [{'path': f'{places[w_][0]}/{places[w_][1]}',
                           'rowIndex': rows.index(places[w_][0]),
                           'columnIndex': cols.index(places[w_][1])} for w_ in plate_wells]}
            for w_ in plate_wells:
                row, col = places[w_]
                put(os.path.join(self.path, row, '.zgroup'), {'zarr_format': 2})
                put(os.path.join(self.path, row, col, '.zgroup'), {'zarr_format': 2})
                put(os.path.join(self.path, row, col, '.zattrs'), {
                    'well': {'images': [{'path': im[1], 'acquisition': 0}
                                        for im in self.images if im[0] == w_],
                             'version': '0.4'}})
        else:
            root['well'] = {'images': [{'path': t, 'acquisition': 0} for t in self.tiles],
                            'version': '0.4'}
        put(os.path.join(self.path, '.zgroup'), {'zarr_format': 2})
        put(os.path.join(self.path, '.zattrs'), root)
        for i, (_well, tile) in enumerate(self.images):
            n_t, n_c, n_z = self._sizes(i)
            unit = 'micrometer' if self.um_per_px > 0 else None
            axes = [{'name': 't', 'type': 'time'}, {'name': 'c', 'type': 'channel'},
                    {'name': 'z', 'type': 'space'}, {'name': 'y', 'type': 'space'},
                    {'name': 'x', 'type': 'space'}]
            if unit:
                for a in axes[2:]:
                    a['unit'] = unit
            xy = self.image_xy.get(i)
            group = os.path.join(self.path, self._image_path(i))
            datasets = []
            for k, (lh, lw) in enumerate(self.levels):
                f = float(2 ** k)
                px = self.um_per_px or 1.0
                transforms = [{'type': 'scale',
                               'scale': [1.0, 1.0, self.z_step_um or 1.0, px * f, px * f]}]
                if xy is not None and self.um_per_px > 0:
                    transforms.append({'type': 'translation',
                                       'translation': [0.0, 0.0, 0.0, xy[1], xy[0]]})
                datasets.append({'path': str(k), 'coordinateTransformations': transforms})
                put(os.path.join(group, str(k), '.zarray'), {
                    'zarr_format': 2, 'shape': [n_t, n_c, n_z, lh, lw],
                    'chunks': [1, 1, 1, lh, lw], 'dtype': dtype, 'order': 'C',
                    'compressor': {'id': 'gzip', 'level': 1}, 'fill_value': 0,
                    'filters': None, 'dimension_separator': '/'})
            put(os.path.join(group, '.zgroup'), {'zarr_format': 2})
            put(os.path.join(group, '.zattrs'), {
                'multiscales': [{'version': '0.4', 'name': tile, 'axes': axes,
                                 'datasets': datasets}],
                'omero': {'channels': [
                    {'label': ch, 'active': True,
                     'color': '%02X%02X%02X' % tuple(int(v * 255) for v in CHANNEL_RGB.get(ch.upper(), (1.0, 1.0, 1.0))),
                     'window': {'start': 0, 'end': 255 if self.dtype == np.uint8 else 4095,
                                'min': 0, 'max': 255 if self.dtype == np.uint8 else 65535}}
                    for ch in self.channels]},
            })


SAVE_FORMATS = ('ome-tiff', 'zarr', 'run-ome-tiff', 'run-zarr', 'single')
SAVE_FORMAT_LABEL = {'ome-tiff': 'one OME-TIFF per well / stack',
                     'zarr': 'one OME-Zarr per well / stack',
                     'run-ome-tiff': 'one OME-TIFF for the whole run',
                     'run-zarr': 'one OME-Zarr for the whole run',
                     'single': 'single-page TIFFs (one per image)'}


def container(fmt: str) -> str:
    """The file format of a save format ('run-zarr' is 'zarr')."""
    return fmt[4:] if fmt.startswith('run-') else fmt


def whole_run(fmt: str) -> bool:
    return fmt.startswith('run-')


def run_path(base_dir: str, fmt: str) -> str:
    """The one file of a whole run: ``<run folder>/<run folder's name>``
    with the format's extension."""
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', os.path.basename(_bare(base_dir))) or 'run'
    return os.path.join(base_dir, name + EXTENSION[container(fmt)])


def storable(frame) -> np.ndarray | None:
    """``frame`` as a well file takes it (2-D uint8 or uint16), or None
    when it is something else (a colour frame): floats from a correction
    are rounded and clipped to 16 bits."""
    frame = np.asarray(frame)
    if frame.ndim != 2:
        return None
    if frame.dtype in (np.uint8, np.uint16):
        return frame
    if frame.dtype.kind in 'fiu':
        return np.clip(np.rint(frame), 0, 65535).astype(np.uint16)
    return None


def _write_single(path: str, img) -> None:
    tifffile.imwrite(path, img)


def _bare(folder: str) -> str:
    """``folder`` without a trailing separator."""
    return folder.rstrip('/').rstrip(os.sep)


class RunSaver:
    """Where a protocol run's frames go: one file per well (``fmt``
    'ome-tiff' or 'zarr'), one file for the whole run ('run-ome-tiff',
    'run-zarr'), or single-page TIFFs as before ('single').

    With a per-well format, ``keep_planes`` also writes each frame as a
    single-page TIFF below ``<base_dir>/_processing_planes`` for the online
    processing, which reads planes back by path; ``remove_planes()`` deletes
    that folder once the processing is through. The per-well files are the
    data of record."""

    PLANES_DIR = '_processing_planes'

    def __init__(self, base_dir: str, fmt: str, *, um_per_px: float = 0.0,
                 z_step_um: float = 0.0, keep_planes: bool = False,
                 info: dict | None = None, single_writer=None, name_of=None):
        if fmt not in SAVE_FORMATS:
            raise ValueError(f'format must be one of {SAVE_FORMATS}, not {fmt!r}')
        self.base_dir = base_dir
        self.fmt = fmt
        self.um_per_px = um_per_px
        self.z_step_um = z_step_um
        self.keep_planes = keep_planes
        self.info = dict(info or {})
        self._single = single_writer or _write_single
        self._name = name_of
        self.writers: dict[str, WellWriter] = {}
        self.fallback = 0

    def _single_path(self, folder: str, well, site, z_index, channel, timepoint) -> str:
        os.makedirs(folder, exist_ok=True)
        if self._name is not None:
            name = self._name(well, site, z_index, channel, timepoint)
        else:
            name = '_'.join(([str(well)] if well else []) + [
                str(site), f'Z{int(z_index):03d}', str(channel), f't{int(timepoint):04d}']) + '.tiff'
        return os.path.join(folder, name)

    def save(self, frame, *, well: str | None, site: str, z_index: int,
             channel: str, timepoint: int, x_um=None, y_um=None, z_um=None,
             t_s=None, exposure_ms=None) -> str | None:
        """Save one frame (``z_index`` and ``timepoint`` count from 1, as
        in the file names). Returns the path of a single-page file holding
        this frame, when one was written (always for 'single'; with
        ``keep_planes``; or for a frame a well file cannot hold)."""
        folder = os.path.join(self.base_dir, well or 'positions')
        if self.fmt == 'single':
            path = self._single_path(folder, well, site, z_index, channel, timepoint)
            self._single(path, frame)
            return path
        plane = storable(frame)
        if plane is None:
            # Not a grey image: kept as its own file, the old way.
            self.fallback += 1
            path = self._single_path(folder, well, site, z_index, channel, timepoint)
            self._single(path, frame)
            return path
        run = whole_run(self.fmt)
        key = '' if run else (well or '')
        writer = self.writers.get(key)
        if writer is None:
            os.makedirs(self.base_dir, exist_ok=True)
            if run:
                writer = WellWriter(run_path(self.base_dir, self.fmt), container(self.fmt),
                                    um_per_px=self.um_per_px, z_step_um=self.z_step_um,
                                    info=self.info, run=True)
            else:
                writer = WellWriter(well_path(self.base_dir, well, self.fmt), self.fmt,
                                    um_per_px=self.um_per_px, z_step_um=self.z_step_um,
                                    well=well, info=self.info)
            self.writers[key] = writer
        writer.add(plane, tile=site, z=int(z_index) - 1, channel=channel,
                   timepoint=int(timepoint) - 1, x_um=x_um, y_um=y_um, z_um=z_um,
                   t_s=t_s, exposure_ms=exposure_ms, well=well or '')
        if not self.keep_planes:
            return None
        path = self._single_path(os.path.join(self.base_dir, self.PLANES_DIR, well or 'positions'),
                                 well, site, z_index, channel, timepoint)
        self._single(path, frame)
        return path

    def close(self) -> list[dict]:
        """Finish every well file (their metadata is written now)."""
        return [w.close() for w in self.writers.values()]

    def remove_planes(self) -> None:
        import shutil
        shutil.rmtree(os.path.join(self.base_dir, self.PLANES_DIR), ignore_errors=True)


class StackSaver:
    """Where a live z-stack's frames go: ``<out_dir>.ome.tiff`` /
    ``<out_dir>.zarr`` (one file for the stack; the empty folder is removed
    at the end), or ``zstack_<colour>_<number>_<height>um.tiff`` files in
    ``out_dir`` ('single'). Planes are numbered by stage height, lowest
    first, within each colour."""

    def __init__(self, out_dir: str, fmt: str, heights_um: dict[str, list[float]],
                 *, um_per_px: float = 0.0, single_writer=None):
        if fmt not in SAVE_FORMATS:
            raise ValueError(f'format must be one of {SAVE_FORMATS}, not {fmt!r}')
        self.out_dir = out_dir
        self.fmt = fmt
        self._single = single_writer or _write_single
        self.rank = {ch: {z: i for i, z in enumerate(sorted({round(float(z), 3) for z in zs}))}
                     for ch, zs in heights_um.items()}
        first = sorted({round(float(z), 3) for z in next(iter(heights_um.values()), [])})
        step = (first[-1] - first[0]) / (len(first) - 1) if len(first) > 1 else 0.0
        self.writer = None
        self.path = out_dir
        if fmt != 'single':
            # A stack is one file whichever of the per-well or whole-run
            # formats is chosen.
            self.path = _bare(out_dir) + EXTENSION[container(fmt)]
            self.writer = WellWriter(self.path, container(fmt), um_per_px=um_per_px, z_step_um=step,
                                     info={'kind': 'z-stack',
                                           'source': os.path.basename(_bare(out_dir))})

    def save(self, frame, channel: str, number: int, z_um: float) -> None:
        plane = storable(frame) if self.writer is not None else None
        if plane is None:
            os.makedirs(self.out_dir, exist_ok=True)
            self._single(os.path.join(self.out_dir, f'zstack_{channel}_{number:03d}_{z_um:.1f}um.tiff'), frame)
            return
        self.writer.add(plane, tile='P1', z=self.rank[channel][round(float(z_um), 3)],
                        channel=channel, z_um=float(z_um))

    def close(self) -> str:
        if self.writer is not None:
            self.writer.close()
            try:
                os.rmdir(self.out_dir)        # only if nothing was put in it
            except OSError:
                pass
        return self.path


def _text(value) -> str:
    return (str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


# ------------------------------------------------------------- reading

def read_index(path: str) -> dict:
    """The index of a well or run file (``WellWriter._index_doc``): from
    the Zarr store's root, from the OME-TIFF's annotation, or, for an
    OME-TIFF that was never closed, from ``<file>.index.json``."""
    if os.path.isdir(path):
        with open(os.path.join(path, '.zattrs'), encoding='utf-8') as f:
            return json.load(f)['lumaviewpro_well_file']
    if os.path.exists(path + '.index.json'):
        with open(path + '.index.json', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f if line.strip()]
        doc = dict(lines[0]['header'])
        doc['channels'], doc['images'], doc['planes'] = [], [], []
        place: dict = {}
        for r in lines[1:]:
            key = (r.get('well', ''), r['tile'])
            if key not in place:
                place[key] = len(doc['images'])
                doc['images'].append({'well': key[0], 'tile': key[1],
                                      'x_um': r.get('x_um'), 'y_um': r.get('y_um')})
            if r['c'] not in doc['channels']:
                doc['channels'].append(r['c'])
            doc['planes'].append([place[key], doc['channels'].index(r['c']), r['z'], r['t']]
                                 + list(r.get('offsets', ())))
        return doc
    if tifffile is None:
        raise RuntimeError('OME-TIFF needs the tifffile package')
    with tifffile.TiffFile(path) as tf:
        xml = tf.pages[0].description
    start = xml.index('<M K="index">') + len('<M K="index">')
    text = xml[start:xml.index('</M>', start)]
    return json.loads(text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&'))


def read_plane(path: str, doc: dict, row: list, level: int = 0) -> np.ndarray:
    """One plane (a row of the index's ``planes``), or one of its reduced
    copies, read from where the index says it is."""
    h, w = doc['levels'][level]
    dtype = np.dtype('u1' if doc['dtype'] == 'uint8' else '<u2')
    if os.path.isdir(path):
        image, c, z, t = row[:4]
        chunk = os.path.join(path, doc['images'][image]['path'], str(level),
                             str(t), str(c), str(z), '0', '0')
        with open(chunk, 'rb') as f:
            return np.frombuffer(gzip.decompress(f.read()), dtype).reshape(h, w)
    return np.fromfile(path, dtype=dtype, count=h * w, offset=row[4 + level]).reshape(h, w)


def read_run(path: str, level: int = 0) -> dict:
    """What a well or run file holds: ``{'images': {(well, tile):
    array[t, c, z, y, x]}, 'channels': [...], 'xy': {(well, tile): (x_um,
    y_um)}, 'levels': [...]}``; ``level`` picks a reduced copy. For checks
    and small files: everything is read into memory."""
    doc = read_index(path)
    h, w = doc['levels'][level]
    dtype = np.dtype('u1' if doc['dtype'] == 'uint8' else '<u2')
    keys = [(e['well'], e['tile']) for e in doc['images']]
    sizes = [[1, len(doc['channels']), 1] for _ in keys]
    for image, _c, z, t, *_ in doc['planes']:
        sizes[image][0] = max(sizes[image][0], t + 1)
        sizes[image][2] = max(sizes[image][2], z + 1)
    arrays = [np.zeros((n_t, n_c, n_z, h, w), dtype) for n_t, n_c, n_z in sizes]
    for row in doc['planes']:
        arrays[row[0]][row[3], row[1], row[2]] = read_plane(path, doc, row, level)
    return {'images': dict(zip(keys, arrays)), 'channels': list(doc['channels']),
            'xy': {k: (e['x_um'], e['y_um']) for k, e in zip(keys, doc['images'])
                   if e['x_um'] is not None},
            'levels': [tuple(s) for s in doc['levels']]}


def read_well(path: str) -> dict:
    """What a one-well file holds: ``{'tiles': {name: array[t, c, z, y,
    x]}, 'channels': [...], 'tile_xy': {name: (x_um, y_um)}}``. For checks
    and small wells: everything is read into memory."""
    got = read_run(path)
    return {'tiles': {k[1]: v for k, v in got['images'].items()},
            'channels': got['channels'],
            'tile_xy': {k[1]: v for k, v in got['xy'].items()}}


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
    fmt = container(fmt)
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
    if whole_run(fmt):
        # Everything in ONE file, timepoint by timepoint, as it was acquired.
        every = sorted((t, well, site, ch, z, path) for well in files
                       for site, z, ch, t, path in files[well])
        with WellWriter(run_path(out_dir, fmt), container(fmt), um_per_px=um_per_px,
                        z_step_um=z_step_um, run=True,
                        info={'source': os.path.basename(folder.rstrip('/\\')), 'kind': 'protocol run'}) as w:
            for t, well, site, ch, z, path in every:
                xy = (positions or {}).get((well or None, site), (None, None))
                w.add(_read(path), tile=site, z=z, channel=ch, timepoint=t, well=well,
                      x_um=xy[0], y_um=xy[1])
                done += 1
                if status_cb and done % 20 == 0:
                    status_cb(f'Packing the run: {done}/{total}')
        return [w.summary()]
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
    ap.add_argument('--format', choices=[f for f in SAVE_FORMATS if f != 'single'], default='ome-tiff')
    ap.add_argument('--um-per-px', type=float, default=0.0)
    ap.add_argument('--out')
    a = ap.parse_args()
    has_stack = any(_ZSTACK_NAME.match(n) for n in os.listdir(a.folder))
    if has_stack:
        print(json.dumps(pack_zstack(a.folder, a.format, a.out, a.um_per_px, print), indent=1))
    else:
        print(json.dumps(pack_run(a.folder, a.format, a.out, a.um_per_px, status_cb=print), indent=1))
