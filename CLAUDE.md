# LumaViewPro (darranclem fork)

Custom control application for an Etaluma Lumascope microscope, built on top of the stock Etaluma drivers. Developed on two PCs: the microscope-attached PC and an offline one, synced through `origin` (github.com/darranclem/LumaViewPro, branch `main`).

## Safety

- **Do not launch `lumaviewpro.py` on the microscope PC without asking.** It connects to the real stage, LEDs and camera and homes the stage at startup.
- To exercise the app without hardware, use the simulator: `.venv\Scripts\python luma_sim.py` (mocks camera, stage and LED board).
- `hw_debug.py` also talks to real hardware.
- The user tests on the real microscope themselves; a simulator run is not required before handing over a change. The simulator's mocks are older than the app (stage and camera show as not connected), so it only checks that the UI builds. If you do run it, run it from a copy of the repo so it cannot overwrite `data/current.json`.

## Environment

- Python 3.12 or 3.13. The project venv is `.venv` (Python 3.13); always use `.venv\Scripts\python`, never bare `python` (the system install has no packages).
- `lumaviewpro.py` relaunches itself with `.venv` when started from another Python. `luma_sim.py` does not, so call it with the venv Python.
- Dependencies: `requirements.txt`. The stock drivers (`pyloncamera`, `ledboard`, `motorboard`) are imported from `C:\Program Files (x86)\LumaViewPro-stock`, which is appended to `sys.path`.
- Shell is Windows PowerShell 5.1; Git Bash is also available.

## Where the code is

The custom app is a handful of top-level files; most work happens here:

- `lumaviewpro.py` / `lumaviewpro.kv`: the Kivy app and its layout
- `acquisition.py`: hardware wrapper, autofocus, shading correction, scans, z-stacks
- `plate_navigator.py` / `plate_navigator.kv`: well plate navigator widget
- `labware_defs.py`, `labware.py`: plate definitions and custom labware
- `experiment.py`: pure planning logic (imaging sites per well, focus map points, focus surface interpolation, file naming)
- `experiment_wizard.py`: the experiment setup wizard, mixed into the app class; its panel is in `lumaviewpro.kv`
- `image_utils.py`, `luma_sim.py`, `hw_debug.py`

`modules/`, `drivers/`, `ui/`, `rest/`, `scripts/` and most of `tests/` come from upstream LumaViewPro. Avoid editing them unless the task is about them.

## Stitching and shading by Quark

- `quark_stitch.py` wraps Quark's tile stitching and flat-field correction (the compiled module `quark_tiles`, built from the private Quark repository's `quark-py` crate with PyO3). It is not a port: the same Rust code Quark runs.
- The compiled file lives in `lib/quark/quark_tiles.pyd` and **is committed** (owner's decision, 2026-11-18; a proprietary Quark binary, see `lib/quark/README.md`). `quark_stitch.flatten_field` / `flatten_folder` (button "Flatten fields (meniscus, hot spot)", CLI `--flatten`) divide a meniscus bright spot, rings and a lamp hot spot out of each transmitted-light field, measured from the field itself. Commit a rebuilt file whenever `quark_stitch.py` starts using a function the committed one lacks; rebuild it with `tools\build_quark_tiles.ps1`. Without it everything still works; tile scans are stitched by stage position as before and the status line says so.
- `acquisition.run_tile_scan` writes `tile_scan.json` (tile positions, pixel size) beside the tiles and, when the module is present, makes `stitched_<channel>.tiff` with Quark (shading measured from the tiles themselves and divided out, tiles placed by their overlaps and blended).
- UI: Tile Scan section, "Stitch folder" re-stitches a finished tile scan or plate folder into `<folder>/quark`. Command line: `.venv\Scripts\python quark_stitch.py <folder>`.
- Shading correction on its own (no empty-field reference): "Shading from folder" measures every channel's shading from a folder's tiles (`quark_stitch.measure_shading`) and stores it as that channel's reference (`ShadingCorrector.set_measured`, note in `data/shading/<channel>.txt`), so the existing Shading switch applies it to live view, snapshots, tile scans and protocol runs. "Correct tiles" writes corrected copies to `<folder>/shading_corrected` (`quark_stitch.correct_folder`; command line `quark_stitch.py <folder> --correct`). Originals are never changed. `tile_scan.json` records `shading_applied`; a map measured from tiles saved with shading on is multiplied onto the reference then in use.
- Tests: `tests/test_quark_stitch.py` (the Quark tests skip themselves when the module is absent).

## Tests

- Needs `pytest` and `pytest-xdist` in `.venv` (`pyproject.toml` passes `--dist loadgroup`); `pyflakes` is used for linting. Install with `.venv\Scripts\python -m pip install pytest pytest-xdist pyflakes`.
- `tests/` holds over 800 upstream test files. Do not run the whole suite; run the specific files for the code you changed.
- Pass `--noconftest`: the upstream `tests/conftest.py` imports `modules.scope_session`, which currently fails (`modules/protocol.py` imports `FilePath`, which `modules/api_surface.py` no longer defines). Example:
  `.venv\Scripts\python -m pytest --noconftest tests/test_af_global_vcurve_fit.py`
- Tests for this fork's own code: `test_af_global_vcurve_fit.py`, `test_image_controls.py`, `test_labware_designer.py`, `test_shading_zstack.py`, `test_plate_well_selection.py`, `test_experiment_plan.py`.

## Git

- `pull.rebase` and `rebase.autoStash` are on: `git pull` before work, commit, `git push` after.
- Machine-specific state is ignored and must stay untracked: `data/af_template.npy`, `data/protocols/_autosave.json`, `data/current.json`, `run.bat`.
- Pushing to `origin/main` after a completed change is the normal workflow here.
