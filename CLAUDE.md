# LumaViewPro (darranclem fork)

Custom control application for an Etaluma Lumascope microscope, built on top of the stock Etaluma drivers. Developed on two PCs: the microscope-attached PC and an offline one, synced through `origin` (github.com/darranclem/LumaViewPro, branch `main`).

## Safety

- **Do not launch `lumaviewpro.py` on the microscope PC without asking.** It connects to the real stage, LEDs and camera and homes the stage at startup.
- To exercise the app without hardware, use the simulator: `.venv\Scripts\python luma_sim.py` (mocks camera, stage and LED board).
- `hw_debug.py` also talks to real hardware.

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
- `image_utils.py`, `luma_sim.py`, `hw_debug.py`

`modules/`, `drivers/`, `ui/`, `rest/`, `scripts/` and most of `tests/` come from upstream LumaViewPro. Avoid editing them unless the task is about them.

## Tests

- Needs `pytest` and `pytest-xdist` in `.venv` (`pyproject.toml` passes `--dist loadgroup`); `pyflakes` is used for linting. Install with `.venv\Scripts\python -m pip install pytest pytest-xdist pyflakes`.
- `tests/` holds over 800 upstream test files. Do not run the whole suite; run the specific files for the code you changed.
- Pass `--noconftest`: the upstream `tests/conftest.py` imports `modules.scope_session`, which currently fails (`modules/protocol.py` imports `FilePath`, which `modules/api_surface.py` no longer defines). Example:
  `.venv\Scripts\python -m pytest --noconftest tests/test_af_global_vcurve_fit.py`
- Tests for this fork's own code: `test_af_global_vcurve_fit.py`, `test_image_controls.py`, `test_labware_designer.py`, `test_shading_zstack.py`.

## Git

- `pull.rebase` and `rebase.autoStash` are on: `git pull` before work, commit, `git push` after.
- Machine-specific state is ignored and must stay untracked: `data/af_template.npy`, `data/protocols/_autosave.json`, `data/current.json`, `run.bat`.
- Pushing to `origin/main` after a completed change is the normal workflow here.
