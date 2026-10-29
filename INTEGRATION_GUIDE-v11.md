# LumaViewPro Architecture & Hardware Setup Guide

## Architecture Overview

LumaViewPro uses a dual-environment design separating real microscope hardware control from offline desktop simulation:

1. **`lumaviewpro.py` (Real Microscope Control Application)**:
   - Connects to physical Etaluma hardware modules (`pyloncamera`, `motorboard`, `ledboard`).
   - When connected to the microscope, commands drive real stage motors, objective turrets, and LED channels.
   - When launched on an unattached PC (no microscope connected), displays a clean **"CAMERA DISCONNECTED"** warning frame without simulating fake hardware motion.

2. **`luma_sim.py` (Offline Desktop Hardware Simulator)**:
   - Injects mock drivers for `pyloncamera`, `motorboard`, and `ledboard`.
   - Generates high-resolution synthetic camera frames with cell colonies, well rims, inter-well grid lines, and Z-focus defocus blur.

---

## 6-Point Target Calibration Wizard Workflow

Located in Tab 1 (`1. Calibration`) of the Studio Panel:

1. Click **`Start Calib`**:
   - Turns illumination Light **ON** automatically.
   - Reveals all 6 calibration target points on the microplate canvas map.
   - Drives the stage directly to Step 1 nominal position (`Well A1 — 12 o'clock / top rim`).
2. Align reticle using stage jog controls or double-clicking the live camera viewport.
3. Click **`Mark`**:
   - Records current stage position for Step 1 and automatically moves to Step 2 (`Well A1 — 3 o'clock / right rim`).
   - Repeat through Step 6 (`Col 1 End — 12 o'clock`).
4. Click **`Next >`** or **`< Prev`** to cycle through target points without re-marking.
5. Click **`Done`**: Computes and applies the 2D Affine Transformation matrix.
