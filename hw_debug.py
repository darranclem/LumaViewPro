from __future__ import annotations
# VERSION: 4.2.0-FINAL-INITIALIZATION
"""
hw_debug.py - Standalone Hardware Diagnostic & Inspection Tool for Etaluma LumaViewPro

Run this script directly on the microscope PC:
    python hw_debug.py

It will inspect pyloncamera, motorboard, ledboard, test all connection methods,
capture test frames, and write a complete diagnostic report to hardware_debug.log.
"""

import sys
import os
import traceback
import datetime

LOG_FILE = "hardware_debug.log"

def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def run_diagnostics():
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except Exception:
            pass

    log("==================================================")
    log("   LUMAVIEWPRO HARDWARE DIAGNOSTIC SUITE          ")
    log("==================================================")
    log(f"Python Executable: {sys.executable}")
    log(f"Python Version: {sys.version}")
    log(f"Working Directory: {os.getcwd()}")
    log(f"System Path: {sys.path[:3]}")

    # --- 1. INSPECT PYLONCAMERA (BASLER USB CAMERA) ---
    log("")
    log("--- [1/3] INSPECTING PYLONCAMERA (BASLER USB CAMERA) ---")
    camera_obj = None
    try:
        import pyloncamera
        log("✓ Successfully imported module: pyloncamera")
        log(f"  Module file: {getattr(pyloncamera, '__file__', 'Built-in/Unknown')}")
        log(f"  Module contents: {dir(pyloncamera)}")

        if hasattr(pyloncamera, "PylonCamera"):
            log("  Found class: pyloncamera.PylonCamera")
            cls = pyloncamera.PylonCamera
            log(f"  Class methods/attributes: {[a for a in dir(cls) if not a.startswith('__')]}")
            
            log("  Attempting instantiation: cam = pyloncamera.PylonCamera()...")
            camera_obj = pyloncamera.PylonCamera()
            log(f"  Instance created: {camera_obj}")
            log(f"  Instance attributes: {camera_obj.__dict__}")

            if hasattr(camera_obj, "connect"):
                log("  Calling camera_obj.connect()...")
                try:
                    res = camera_obj.connect()
                    log(f"  camera_obj.connect() returned: {repr(res)} (type: {type(res).__name__})")
                except Exception as e:
                    log(f"  X camera_obj.connect() EXCEPTION: {e}")
                    log(traceback.format_exc())

            is_conn = getattr(camera_obj, "is_connected", None)
            log(f"  camera_obj.is_connected = {repr(is_conn)}")

            log("  Testing frame acquisition via get_frame()...")
            if hasattr(camera_obj, "get_frame"):
                try:
                    frame = camera_obj.get_frame()
                    if frame is not None:
                        log(f"  ✓ SUCCESS! get_frame() returned frame with shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}")
                    else:
                        log("  ! get_frame() returned None (camera might need light or exposure trigger)")
                except Exception as e:
                    log(f"  X get_frame() EXCEPTION: {e}")
                    log(traceback.format_exc())
            else:
                log("  ! camera_obj has no get_frame method. Checking alternative grab methods...")
                for alt_m in ["grab", "get_image", "read", "capture_frame"]:
                    if hasattr(camera_obj, alt_m):
                        log(f"    Found alternative method: {alt_m}")
        else:
            log("  X Class PylonCamera not found in module pyloncamera.")
    except ModuleNotFoundError:
        log("  X ModuleNotFoundError: pyloncamera is NOT installed in this Python environment.")
    except Exception as e:
        log(f"  X Error inspecting pyloncamera: {e}")
        log(traceback.format_exc())

    # --- 2. INSPECT MOTORBOARD (STAGE & Z FOCUS MOTOR CONTROLLER) ---
    log("")
    log("--- [2/3] INSPECTING MOTORBOARD (STAGE & Z FOCUS MOTORS) ---")
    stage_obj = None
    try:
        import motorboard
        log("✓ Successfully imported module: motorboard")
        log(f"  Module file: {getattr(motorboard, '__file__', 'Built-in/Unknown')}")
        log(f"  Module contents: {dir(motorboard)}")

        if hasattr(motorboard, "MotorBoard"):
            log("  Found class: motorboard.MotorBoard")
            cls = motorboard.MotorBoard
            log(f"  Class methods/attributes: {[a for a in dir(cls) if not a.startswith('__')]}")

            log("  Attempting instantiation: stg = motorboard.MotorBoard()...")
            stage_obj = motorboard.MotorBoard()
            log(f"  Instance created: {stage_obj}")
            log(f"  Instance attributes: {stage_obj.__dict__}")

            if hasattr(stage_obj, "connect"):
                log("  Calling stage_obj.connect()...")
                try:
                    res = stage_obj.connect()
                    log(f"  stage_obj.connect() returned: {repr(res)} (type: {type(res).__name__})")
                except Exception as e:
                    log(f"  X stage_obj.connect() EXCEPTION: {e}")
                    log(traceback.format_exc())

            if hasattr(stage_obj, "get_pos"):
                try:
                    pos = stage_obj.get_pos()
                    log(f"  stage_obj.get_pos() returned: {pos}")
                except Exception as e:
                    log(f"  X get_pos() EXCEPTION: {e}")
        else:
            log("  X Class MotorBoard not found in module motorboard.")
    except ModuleNotFoundError:
        log("  X ModuleNotFoundError: motorboard is NOT installed in this Python environment.")
    except Exception as e:
        log(f"  X Error inspecting motorboard: {e}")
        log(traceback.format_exc())

    # --- 3. INSPECT LEDBOARD (ILLUMINATION CONTROLLER) ---
    log("")
    log("--- [3/3] INSPECTING LEDBOARD (LED ILLUMINATION) ---")
    led_obj = None
    try:
        import ledboard
        log("✓ Successfully imported module: ledboard")
        log(f"  Module file: {getattr(ledboard, '__file__', 'Built-in/Unknown')}")
        log(f"  Module contents: {dir(ledboard)}")

        if hasattr(ledboard, "LEDBoard"):
            log("  Found class: ledboard.LEDBoard")
            cls = ledboard.LEDBoard
            log(f"  Class methods/attributes: {[a for a in dir(cls) if not a.startswith('__')]}")

            log("  Attempting instantiation: leds = ledboard.LEDBoard()...")
            led_obj = ledboard.LEDBoard()
            log(f"  Instance created: {led_obj}")
            log(f"  Instance attributes: {led_obj.__dict__}")

            if hasattr(led_obj, "connect"):
                log("  Calling led_obj.connect()...")
                try:
                    res = led_obj.connect()
                    log(f"  led_obj.connect() returned: {repr(res)} (type: {type(res).__name__})")
                except Exception as e:
                    log(f"  X led_obj.connect() EXCEPTION: {e}")
                    log(traceback.format_exc())
        else:
            log("  X Class LEDBoard not found in module ledboard.")
    except ModuleNotFoundError:
        log("  X ModuleNotFoundError: ledboard is NOT installed in this Python environment.")
    except Exception as e:
        log(f"  X Error inspecting ledboard: {e}")
        log(traceback.format_exc())

    log("")
    log("==================================================")
    log("   DIAGNOSTIC SUITE COMPLETE                       ")
    log(f"   Report written to: {os.path.abspath(LOG_FILE)}")
    log("==================================================")

if __name__ == "__main__":
    run_diagnostics()
