# Quark stitching module

`quark_stitch.py` loads `quark_tiles.pyd` from this folder: Quark's tile stitching and
flat-field (shading) correction, compiled from Quark's `quark-py` crate with PyO3.

`quark_tiles.pyd` **is kept in this repository** (owner's decision, 2026-11-18), so a `git pull`
installs it on every PC; one file works with 64-bit Windows Python 3.9 and later. It is a compiled
binary of Quark, which is proprietary (c) its owner: it is here to be used with this application,
not as open source, and the licence of this repository's Python code does not extend to it.

The file here was built from Quark build 2026-11-18c. It provides: reading plate and tile-scan
folders (this fork's and stock LumaViewPro protocol runs), stitching, flat-field correction
measured from a folder's own tiles, and `flatten_field` (a meniscus bright spot, rings, a lamp hot
spot and well-wall shadow divided out of one transmitted-light field, measured from that field).

To rebuild it from the Quark source: `powershell -ExecutionPolicy Bypass -File toolsuild_quark_tiles.ps1`.

Without it LumaViewPro still runs and stitches tile scans by stage position, as before; the
status line says that Quark stitching is not installed.
