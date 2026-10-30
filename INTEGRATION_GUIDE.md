# LumaViewPro Plate Navigator & Hardware Integration Guide

## Overview

This guide details the integration architecture for running **LumaViewPro** in both offline simulation mode on a desktop PC and physical hardware mode on the fluorescence microscope system.

---

## Key Features & Enhancements

1. **Clean Single-Loading KV Architecture**:
   - Overrides `App.load_kv()` to ensure `lumaviewpro.kv` is loaded exactly once into Kivy's `Builder`, eliminating UI vertical duplication and Kivy `PopupException` layout errors.

2. **Default Launch in Dark State & Illumination Control**:
   - On application startup, illumination starts **OFF** (`is_light_on = False`), displaying a dark sensor frame.
   - Turning light ON for a channel (or clicking **`Light: ON / OFF`**) illuminates the live camera stream with exact channel color LUTs:
     - **Brightfield & Phase**: Monochromatic Grayscale.
     - **Darkfield**: Inverted Monochromatic Grayscale.
     - **Blue (DAPI / 405 nm)**: Pure Blue channel LUT.
     - **Green (FITC / GFP / 488 nm)**: Pure Green channel LUT.
     - **Red (TxRed / mCherry / 594 nm)**: Pure Red channel LUT.

3. **Strict Single-Channel Mutual Exclusion**:
   - Selecting a channel turns illumination ON for that channel and automatically turns OFF any previously active channel. Clicking an active channel button again toggles light OFF and returns the view to dark.

4. **Z Focus Controls & AF Search Range**:
   - Dedicated Z focus jog buttons (`+10 um`, `+1 um`, `-1 um`, `-10 um`) alongside XY stage motion.
   - Configurable AF Search Range (+/- um) input in both main UI and Plate Navigator.

5. **Plate Navigator Double-Click Well Navigation**:
   - Double-clicking any well on the microplate canvas automatically moves the stage motor position directly to that well center and updates the active target crosshair.

6. **Well Plate Format Map Selection**:
   - Dropdown format selector for **6, 12, 24, 48, 96, and 384 well plates**, dynamically reading SLAS pitch and offset dimensions from `labware.py`.

---

## Dual-Mode Execution Commands (`PowerShell`)

### Mode 1: Offline Desktop Simulator (Upstairs PC)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python luma_sim.py
```

### Mode 2: Physical Microscope Operation (Downstairs PC)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python lumaviewpro.py
```
