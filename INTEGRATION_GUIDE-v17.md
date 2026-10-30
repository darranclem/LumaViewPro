# LumaViewPro Integration Guide & Dual-Execution Instructions

LumaViewPro is a unified, single-window microscope automation and registration control software written in Python and Kivy.

---

## Directory Structure

```text
LumaViewPro-main/
├── lumaviewpro.py       # Main Application Controller & Kivy Event Engine
├── lumaviewpro.kv       # Single-Window UI Layout Definition
├── luma_sim.py          # Offline Desktop Hardware Simulator Launcher
├── plate_navigator.py   # Microplate Registration Map & Canvas Engine
├── plate_navigator.kv   # Canvas Layout Rules & Styling Definitions
├── labware.py           # SLAS Specs, Objective Presets & 6-Point Target Calibration Matrix
├── image_utils.py       # Camera Feed Generator, Defocus Blur, and Focus Metrics
└── INTEGRATION_GUIDE.md # Execution Instructions
```

---

## Execution Modes

### 1. Offline Desktop Simulator Mode (Upstairs PC)
To run and test the complete single-window UI, microplate map, live camera feed, and 6-point wizard offline without microscope hardware:

```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python luma_sim.py
```

* **Default Launch**: Launches in dark sensor state (`is_light_on = False`).
* **Illumination Control**: Click the **`Light: OFF`** toggle or select any optical channel button (`Brightfield`, `Phase`, `Darkfield`, `Blue`, `Green`, `Red`) to turn illumination **ON**.
* **Navigation**: Double-click any well on the top-right microplate map, or double-click any pixel in the camera feed to center that feature directly under the red reticle.

### 2. Real Microscope Hardware Mode (Downstairs PC)
To operate the physical microscope attached via USB:

```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python lumaviewpro.py
```

* Connects directly to physical USB hardware drivers (`pyloncamera`, `motorboard`, `ledboard`).
* If camera hardware is not detected, displays a **`CAMERA DISCONNECTED`** warning overlay while keeping stage and focus controls functional.

---

## 6-Point Target Calibration Wizard Workflow

Located in Tab 1 (**`1. Calib`**) of the bottom-right Studio Panel:

1. **Click `Start Calib`**:
   - Turns illumination Light **ON**.
   - Displays the 6 target calibration marks on the microplate canvas map.
   - Drives stage to Step 1: **Well A1 — 12 o'clock top rim**.
2. **Align & Click `Mark`**:
   - Align the central red reticle on the well rim edge using stage jog controls or live viewport double clicks.
   - Click **`Mark`**: Records current position and drives the stage to the next target position.
3. **`Next >` & `< Prev`**:
   - Cycle forward or backward through the 6 calibration target points.
4. **Click `Done`**:
   - Computes and applies the 2D Affine Transformation matrix.
