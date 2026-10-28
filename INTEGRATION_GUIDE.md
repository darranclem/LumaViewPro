# LumaViewPro Integration & User Guide

## Architectural Highlights

1. **Unified Single-Window Kivy Architecture**:
   - **Top Navigation Bar**: Active Well indicator, real-time stage motor coordinates `(X, Y, Z)`, and Calibration Status.
   - **Top-Left Quadrant**: Live camera feed viewport (`ZoomPanCameraWidget`) with red reticle crosshairs and **1:1 double-click pixel centering**. Double-clicking any pixel feature in the camera image moves that specific feature directly under the central red reticle.
   - **Bottom-Left Quadrant**: Microscope Motion, Illumination & Camera Control Deck (`Light: ON/OFF`, Exposure, Gain, LED Power, Grouped XY Motion Cross, Z Focus Column, Snapshot Capture).
   - **Top-Right Quadrant**: Interactive Microplate Canvas Map (`InteractiveScaledPlateCanvas`) with **1:1 double-click stage navigation**. Double-clicking any well or coordinate jumps the FOV directly to that location.
   - **Bottom-Right Quadrant**: Optical Channels Selection Panel (strict single-channel mutual exclusion) + 3-Tab Registration Studio.

2. **Default Dark Sensor Launch State**:
   - The application launches in a dark sensor state (`is_light_on = False`). The live camera feed renders dark sensor noise until illumination is explicitly toggled **ON** or an optical channel button is selected.

3. **Selective Hardware Device Connectivity**:
   - `lumaviewpro.py` independently checks USB connectivity for `camera_connected`, `stage_connected`, `focus_connected`, and `led_connected`.
   - If the USB camera is not attached, the camera viewport displays a `CAMERA DISCONNECTED` warning frame while keeping stage motor controls, keyboard arrow shortcuts, Z focus, and microplate canvas navigation 100% active and responsive.

4. **Interactive 6-Point Target Registration Wizard**:
   - Located in Tab 1 (**`1. Calib`**) of the bottom-right Studio Panel:
     - **`Start Calib`**: Turns illumination Light **ON** and drives stage to Step 1 target (Well A1 - 12 o'clock top rim).
     - **`Mark Target`**: Records stage position for current target point and advances to the next step.
     - **`< Prev`** & **`Next >`**: Cycles forward/backward through all 6 target calibration points.
     - **`Done`**: Computes and applies the 2D Affine Transformation matrix (A1 center, well diameter, row/col pitch, rotation angle).
     - **`Reset`**: Clears recorded calibration points.

5. **Optically Accurate Frame Simulation & Z Defocus Blur**:
   - Renders double-ring well wall rims, rich internal cellular colony structures (cytoplasm, nuclei, membrane borders), inter-well plastic molding grid lines, and outer plate skirt borders.
   - Symmetrical Gaussian defocus blur simulates physical focus displacement `|stage_z - best_z|`.

---

## Execution Instructions

### Running in Offline Desktop Simulator Mode (Desktop PC)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python luma_sim.py
```

### Running on Physical Microscope PC (Downstairs Hardware)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python lumaviewpro.py
```

### Git Commit & Push Checkpoint
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
git add .
git commit -m "Update LumaViewPro single-window UI, 6-point wizard controls, and synthetic feed feature rendering"
git push -u origin main
```
