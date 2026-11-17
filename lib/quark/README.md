# Quark stitching module

`quark_stitch.py` loads `quark_tiles.pyd` from this folder: Quark's tile stitching and
flat-field (shading) correction, compiled from Quark's `quark-py` crate with PyO3.

The `.pyd` is a compiled file and is **not kept in this repository**. To install it on a PC:

- copy `quark_tiles.pyd` here from a PC that has it (one file works with Python 3.9 and later), or
- build it from the Quark source: `powershell -ExecutionPolicy Bypass -File tools\build_quark_tiles.ps1`.

Without it LumaViewPro still runs and stitches tile scans by stage position, as before; the
status line says that Quark stitching is not installed.
