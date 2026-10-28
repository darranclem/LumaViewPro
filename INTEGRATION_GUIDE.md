# LumaViewPro Integration & Hardware Diagnostics Guide

## 1. Overview
LumaViewPro supports both physical microscope hardware operation and an offline desktop simulator.

- **`lumaviewpro.py`**: Main microscope control controller. Handshakes directly with physical Etaluma drivers (`pyloncamera`, `motorboard`, `ledboard`). On unattached PCs, it renders the `CAMERA DISCONNECTED` warning frame and disables motion, providing a clear offline sense check.
- **`luma_sim.py`**: Offline desktop hardware simulator injecting mock drivers (`MockCamera`, `MockStage`, `MockLEDBoard`). Enables testing of live feeds, stage jogging, objective magnification scaling, and the 6-point target calibration wizard without physical hardware.
- **`hw_debug.py`**: Standalone hardware diagnostic script. Inspects installed drivers, tests connection handshakes, captures test camera frames, and writes a detailed report to `hardware_debug.log`.

---

## 2. Running Hardware Diagnostics on the Microscope PC

When setting up or troubleshooting the physical microscope, run:

```powershell
python hw_debug.py
```

This generates `hardware_debug.log` detailing:
- Module import status for `pyloncamera`, `motorboard`, and `ledboard`.
- Return types and values from `connect()` and `is_connected`.
- Frame grab results from `get_frame()`.
- Full tracebacks if any driver calls raise exceptions.

---

## 3. Running LumaViewPro Applications

### Physical Microscope Hardware Mode (Downstairs PC)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python lumaviewpro.py
```

### Offline Desktop Simulator Mode (Upstairs PC)
```powershell
Set-Location H:\Trae\Projects\LumaViewPro-main
.\venv\Scripts\Activate.ps1
python luma_sim.py
```

---

## 4. Hardware Driver Handshaking Summary

| Hardware Module | Class Name | Connection Method | Return Value Handshake |
| :--- | :--- | :--- | :--- |
| **USB Camera** | `pyloncamera.PylonCamera` | `cam.connect()` | `None` / `True` (Checked via `res is not False`) |
| **Stage & Focus Motors** | `motorboard.MotorBoard` | `stg.connect()` | `None` / `True` (Checked via `res is not False`) |
| **LED Illumination** | `ledboard.LEDBoard` | `leds.connect()` | `None` / `True` (Checked via `res is not False`) |
