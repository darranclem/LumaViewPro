# LumaViewPro 4.0.0-beta Integration & User Guide

LumaViewPro is a single-window Python/Kivy application designed to operate and automate fluorescence microscopes, multi-axis stage positioning, and multi-channel experiment protocols.

---

## 1. Operating Modes

### A. Real Microscope Operation (Downstairs PC)
To run LumaViewPro connected to real physical microscope hardware:
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python lumaviewpro.py
```
* **Hardware Drivers**: Connects directly to Basler USB camera (`pyloncamera.PylonCamera`), stage motor controller (`motorboard.MotorBoard`), and LED illumination board (`ledboard.LEDBoard`).
* **Disconnected Guard**: If run on an unattached PC, `lumaviewpro.py` displays a `MICROSCOPE DISCONNECTED` warning frame and disables stage motion rather than faking movements.

### B. Offline Desktop Simulator Mode (Upstairs PC)
To develop and test the GUI, plate registration canvas, and live feed offline without physical hardware attached:
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python luma_sim.py
```
* **Synthetic Camera Stream**: Renders dark sensor noise when illumination is OFF, and realistic cellular colonies with Z defocus blur, well rims, outer plastic skirt borders, and inter-well grid lines when illumination is ON.

---

## 2. Key Features & Workflows

1. **Single-Window Architecture**: Live camera viewport (Top-Left), microplate map canvas (Top-Right), hardware control deck (Bottom-Left), and optical channel studio (Bottom-Right).
2. **Double-Click Pixel Centering**: Double-clicking any pixel feature in the camera feed moves that feature directly under the central red reticle.
3. **Double-Click Microplate Navigation**: Double-clicking any well or location on the microplate canvas map drives the stage directly to those coordinates.
4. **Magnification-Proportional FOV Scaling**: Selecting objective presets (`1.25x` to `100x Oil`) rescales both the camera view and the cyan FOV reticle box on the map.
5. **6-Point Target Calibration Wizard**: Step-by-step target wizard (`Start Calib`, `Mark Target`, `< Prev`, `Next >`, `Done`, `Reset`) to compute the 2D Affine Transformation matrix across well plate rim targets.
6. **Single-Channel Mutual Exclusion**: Selecting an optical channel (`Brightfield`, `Phase`, `Darkfield`, `Blue`, `Green`, `Red`) illuminates that channel exclusively.

---

## 3. GitHub Commit & Push Instructions

```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
git add .
git commit -m "Refactor hardware driver integration in lumaviewpro.py and mock hardware layer in luma_sim.py"
git push -u origin main
```
