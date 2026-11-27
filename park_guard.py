"""Park guard: leaves the microscope safe when LumaViewPro dies.

LumaViewPro starts this as a separate, detached process once the stage is
connected. It does nothing while the application runs. When the
application's process ends for ANY reason (closed, crashed, frozen and
killed from Task Manager) it checks whether the application parked the
stage itself; if not, it opens the motor board and parks it:

  1. objective lowered all the way (Z = 0), and confirmed down,
  2. only then the stage moved to the center of its travel.

If the objective cannot be confirmed down, the stage is NOT moved.

It cannot help if the PC loses power or the motor board is unplugged.

Usage: park_guard.py <app pid> <parked-flag file> <log file>
       park_guard.py --park-now <log file>      (park immediately)
"""
from __future__ import annotations

import os
import sys
import time

STOCK_DIR = r'C:\Program Files (x86)\LumaViewPro-stock'
Z_DOWN_UM = 50.0          # at or below this the objective counts as down
SERIAL_TIMEOUT_S = 20.0   # no serial read may block longer than this


def _log(path: str, msg: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {msg}\n')
    except Exception:
        pass


def wait_for_exit(pid: int) -> None:
    """Block until the process with this id has ended. Returns its exit
    code when it can be read (Windows), which says how it ended (see
    run_guard.EXIT_CODES), else None."""
    if os.name == 'nt':
        import ctypes
        SYNCHRONIZE = 0x00100000
        QUERY_LIMITED = 0x00001000
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        handle = k32.OpenProcess(SYNCHRONIZE | QUERY_LIMITED, False, int(pid))
        if not handle:
            handle = k32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not handle:
            return None                  # already gone
        k32.WaitForSingleObject(ctypes.c_void_p(handle), 0xFFFFFFFF)
        code = ctypes.c_ulong(0)
        ok = k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        k32.CloseHandle(ctypes.c_void_p(handle))
        return int(code.value) if ok else None
    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)


def app_parked(flag_path: str, pid: int) -> bool:
    """True if the application left a note that it parked the stage."""
    try:
        with open(flag_path, 'r', encoding='utf-8') as f:
            return f.read().strip() == str(pid)
    except Exception:
        return False


def _wait_axis(stage, axis: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if stage.target_status(axis):
                return True
        except Exception:
            return False
        time.sleep(0.05)
    return False


def park(stage, log=lambda msg: None, timeout_s: float = 40.0) -> bool:
    """Objective down first, confirmed; then the stage to the center.
    ``stage`` is a stock MotorBoard (or anything with the same methods).
    Returns True when both steps were done."""
    def z_now():
        try:
            return float(stage.current_pos('Z'))
        except Exception:
            return None

    try:
        stage.move_abs_pos('Z', 0.0)
    except Exception as e:
        log(f'Z move failed: {e}')
    _wait_axis(stage, 'Z', timeout_s)
    z = z_now()
    if z is None or z > Z_DOWN_UM:
        log(f'objective not confirmed down (Z = {z}); homing Z')
        try:
            stage.zhome()
        except Exception as e:
            log(f'Z home failed: {e}')
        _wait_axis(stage, 'Z', timeout_s)
        z = z_now()
    if z is None or z > Z_DOWN_UM:
        log(f'objective still not confirmed down (Z = {z}): '
            f'stage NOT moved')
        return False
    log(f'objective down (Z = {z:.0f} um); centering the stage')
    try:
        stage.xycenter()
    except Exception as e:
        log(f'centering failed: {e}')
        return False
    ok = _wait_axis(stage, 'X', timeout_s) and _wait_axis(stage, 'Y', timeout_s)
    log('stage centered' if ok else 'centering not confirmed in time')
    return ok


def open_stage(log, attempts: int = 8):
    """Open the motor board with the stock driver. The port is only free
    once the application's process is completely gone, so retry."""
    if STOCK_DIR not in sys.path:
        sys.path.append(STOCK_DIR)
    import motorboard
    for i in range(attempts):
        try:
            stage = motorboard.MotorBoard()
            if not getattr(stage, 'driver', False):
                stage.connect()
            if getattr(stage, 'driver', False):
                try:
                    stage.driver.timeout = SERIAL_TIMEOUT_S
                    stage.driver.write_timeout = SERIAL_TIMEOUT_S
                except Exception:
                    pass
                return stage
        except Exception as e:
            log(f'motor board not available (try {i + 1}): {e}')
        time.sleep(1.5)
    return None


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == '--park-now':
        log_path = argv[1]
        pid = None
    elif len(argv) >= 3:
        pid, flag_path, log_path = int(argv[0]), argv[1], argv[2]
    else:
        print(__doc__)
        return 2

    def log(msg):
        _log(log_path, msg)

    if pid is not None:
        log(f'guarding LumaViewPro (pid {pid})')
        code = wait_for_exit(pid)
        try:
            import run_guard
            how = run_guard.exit_code_text(code)
        except Exception:
            how = f'exit code {code}'
        log(f'LumaViewPro (pid {pid}) ended: {how}')
        if app_parked(flag_path, pid):
            log('application closed and parked the stage itself')
            try:
                os.remove(flag_path)
            except Exception:
                pass
            return 0
        log('application ended WITHOUT parking the stage: parking now')
        time.sleep(1.0)
    stage = open_stage(log)
    if stage is None:
        log('could not open the motor board: stage NOT parked')
        return 1
    ok = park(stage, log)
    try:
        stage.driver.close()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
