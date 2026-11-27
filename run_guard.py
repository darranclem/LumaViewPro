"""Staying up through a long protocol run, and saying why when it did not.

- :func:`install` (once, at start-up): Python's fault handler writes the
  stack of every thread to ``logs/crash.log`` if the process dies in
  native code (a camera, LED or graphics driver); an exception on any
  background thread goes to ``logs/errors.log`` instead of a console
  nobody sees.
- :class:`Watchdog`: a thread that dumps every thread's stack to
  ``logs/crash.log`` when the window or a run stops making progress
  (a freeze, a deadlock, a driver call that never returns).
- :class:`RunLog`: ``<run folder>/run.log``, a line per position (with the
  memory in use and the free disk space) and how the run ended; and
  ``logs/run_active.json`` while a run goes on.
- :func:`previous_run_died`: at the next start, a run that never ended is
  reported, with what Windows recorded about that time (an application
  crash and the DLL it happened in, a hang, sleep, a restart or a power
  loss), and its well files are finished (:func:`well_file.finish_folder`).
- :func:`keep_awake`: Windows does not sleep while a run goes on.

No Kivy in here.
"""
from __future__ import annotations

import faulthandler
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

_crash_file = None
_logs_dir = None
_lock = threading.Lock()

# Windows process exit codes worth naming (park_guard logs them too).
EXIT_CODES = {
    0: 'closed normally',
    1: 'ended by another program (e.g. Task Manager "End task") or exit(1)',
    0xC0000005: 'ACCESS VIOLATION in native code (a driver or library crashed)',
    0xC0000374: 'HEAP CORRUPTION in native code',
    0xC0000409: 'native code stopped itself (stack buffer overrun / fail-fast)',
    0xC00000FD: 'STACK OVERFLOW',
    0xC0000017: 'OUT OF MEMORY',
    0xC000013A: 'console window closed / Ctrl+C',
    0x40010004: 'killed because Windows was shutting down or logging off',
    0xC0000142: 'could not start (DLL initialisation failed)',
    0xE06D7363: 'unhandled C++ exception in native code',
    3: 'C runtime abort()',
}


def exit_code_text(code) -> str:
    if code is None:
        return 'exit code unknown'
    code = int(code) & 0xFFFFFFFF
    what = EXIT_CODES.get(code)
    return f'exit code 0x{code:08X}' + (f' = {what}' if what else '')


def _stamp() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def log_error(text: str) -> None:
    """Append to logs/errors.log (never raises)."""
    if not _logs_dir:
        return
    try:
        with _lock, open(os.path.join(_logs_dir, 'errors.log'), 'a',
                         encoding='utf-8') as f:
            f.write(f'=== {_stamp()}\n{text.rstrip()}\n\n')
    except Exception:
        pass


def install(logs_dir: str) -> None:
    """Fault handler into logs/crash.log, background-thread exceptions into
    logs/errors.log. Safe to call more than once."""
    global _crash_file, _logs_dir
    _logs_dir = logs_dir
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        return
    if _crash_file is None:
        try:
            _crash_file = open(os.path.join(logs_dir, 'crash.log'), 'a',
                               encoding='utf-8', buffering=1)
            _crash_file.write(f'\n=== {_stamp()}  LumaViewPro started '
                              f'(pid {os.getpid()}, Python '
                              f'{sys.version.split()[0]})\n')
            _crash_file.flush()
            faulthandler.enable(file=_crash_file, all_threads=True)
        except Exception:
            _crash_file = None

    def _hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else '?'
        text = ''.join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback))
        log_error(f'Uncaught error on thread {name}:\n{text}')
        try:
            sys.__stderr__.write(text)
        except Exception:
            pass
    threading.excepthook = _hook


def dump_stacks(reason: str) -> None:
    """Every thread's stack into logs/crash.log, with the reason."""
    f = _crash_file
    if f is None:
        return
    try:
        with _lock:
            f.write(f'\n--- {_stamp()}  {reason}: stacks of all threads\n')
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.flush()
    except Exception:
        pass


class Watchdog:
    """``beat(name, limit_s)`` says "``name`` is alive, and should say so
    again within ``limit_s``"; ``rest(name)`` stops watching it. When a
    beat is late, every thread's stack goes to crash.log (once per
    stall), and to errors.log a line saying so."""

    def __init__(self, check_s: float = 5.0):
        self._beats: dict[str, tuple[float, float, str]] = {}
        self._reported: set[str] = set()
        self._check_s = check_s
        t = threading.Thread(target=self._loop, name='watchdog', daemon=True)
        t.start()

    def beat(self, name: str, limit_s: float, doing: str = '') -> None:
        self._beats[name] = (time.time(), float(limit_s), doing)
        if name in self._reported:
            self._reported.discard(name)
            log_error(f'{name}: going again')

    def rest(self, name: str) -> None:
        self._beats.pop(name, None)
        self._reported.discard(name)

    def _loop(self):
        while True:
            time.sleep(self._check_s)
            now = time.time()
            for name, (t, limit, doing) in list(self._beats.items()):
                late = now - t
                if late > limit and name not in self._reported:
                    self._reported.add(name)
                    what = f'{name} has not moved on for {late:.0f} s' + (
                        f' (last: {doing})' if doing else '')
                    log_error(f'STALL: {what}; stacks in crash.log')
                    dump_stacks(f'STALL: {what}')


def memory_mb() -> tuple[float | None, float | None]:
    """(this process's memory in use, memory still free on the PC) in MB."""
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2 ** 20
        free = psutil.virtual_memory().available / 2 ** 20
        return rss, free
    except Exception:
        return None, None


def free_disk_gb(path: str) -> float | None:
    try:
        return shutil.disk_usage(path).free / 2 ** 30
    except Exception:
        return None


def keep_awake(on: bool) -> None:
    """While ``on``, Windows does not go to sleep (the display may still
    turn off). Applies to the calling thread: call it from the thread that
    runs the protocol, and again with False when the run ends; it also
    ends when that thread ends."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


# ------------------------------------------------------------- run log

ACTIVE_NAME = 'run_active.json'


class RunLog:
    """``<run folder>/run.log``, and ``logs/run_active.json`` until
    :meth:`end` (if that file is still there at the next start, the run
    never ended: :func:`previous_run_died`)."""

    def __init__(self, run_dir: str, logs_dir: str, about: str = ''):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, 'run.log')
        self._active = os.path.join(logs_dir, ACTIVE_NAME)
        self.started = time.time()
        self.ended = False
        try:
            os.makedirs(logs_dir, exist_ok=True)
            with open(self._active, 'w', encoding='utf-8') as f:
                json.dump({'pid': os.getpid(), 'run_dir': run_dir,
                           'started': self.started}, f)
        except Exception:
            pass
        self.write(f'run started (pid {os.getpid()}) {about}'.rstrip())
        self.state()

    def write(self, msg: str) -> None:
        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(f'{_stamp()}  {msg}\n')
                f.flush()
        except Exception:
            pass

    def state(self, prefix: str = '') -> None:
        """A line with the memory in use and the free disk space."""
        rss, free = memory_mb()
        disk = free_disk_gb(self.run_dir)
        parts = []
        if rss is not None:
            parts.append(f'memory used {rss:.0f} MB, free {free:.0f} MB')
        if disk is not None:
            parts.append(f'disk free {disk:.1f} GB')
        if parts:
            self.write((prefix + '  ' if prefix else '') + ', '.join(parts))

    def error(self, msg: str, exc: BaseException | None = None) -> None:
        text = msg
        if exc is not None:
            text += '\n' + ''.join(traceback.format_exception(
                type(exc), exc, exc.__traceback__))
        self.write('ERROR ' + text)
        log_error(f'run {os.path.basename(self.run_dir)}: {text}')

    def end(self, how: str) -> None:
        if self.ended:
            return
        self.ended = True
        self.state('at the end:')
        self.write(f'RUN ENDED: {how}')
        try:
            os.remove(self._active)
        except OSError:
            pass


def _iso_utc(t: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime(t))


def _wevt(log: str, query: str, count: int = 12) -> str:
    try:
        flags = 0x08000000 if os.name == 'nt' else 0   # no console window
        out = subprocess.run(
            ['wevtutil', 'qe', log, f'/q:{query}', '/f:text', f'/c:{count}',
             '/rd:true'], capture_output=True, text=True, timeout=20,
            creationflags=flags, errors='replace')
        return out.stdout
    except Exception:
        return ''


def _events(text: str) -> list[str]:
    """wevtutil text output split into events, each trimmed to what
    matters (time, source, id and the start of the message)."""
    events, cur = [], []
    for line in text.replace('\x00', '').splitlines():
        if line.startswith('Event[') and cur:
            events.append(cur)
            cur = []
        cur.append(line.rstrip())
    if cur:
        events.append(cur)
    out = []
    for ev in events:
        keep, desc = [], False
        for line in ev:
            s = line.strip()
            if s.startswith(('Date:', 'Source:', 'Event ID:')):
                keep.append(s)
            elif s.startswith('Description:'):
                desc = True
            elif desc and s:
                keep.append('  ' + s)
                if len(keep) > 14:
                    break
        if keep:
            out.append('\n'.join(keep))
    return out


def windows_events(since: float, until: float | None = None) -> list[str]:
    """What Windows recorded between ``since`` and ``until`` that can end a
    program: Python crashing or hanging (Application log) and sleep,
    restart, shutdown and power loss (System log)."""
    if os.name != 'nt':
        return []
    until = until or time.time()
    span = (f"TimeCreated[@SystemTime>='{_iso_utc(since - 60)}' and "
            f"@SystemTime<='{_iso_utc(until + 60)}']")
    app = _events(_wevt('Application', (
        "*[System[(Provider[@Name='Application Error'] or "
        "Provider[@Name='Application Hang'] or "
        "Provider[@Name='Windows Error Reporting']) and " + span + "]]")))
    app = [e for e in app if 'python' in e.lower()]
    system = _events(_wevt('System', (
        "*[System[(EventID=41 or EventID=42 or EventID=107 or EventID=1 or "
        "EventID=6006 or EventID=6008 or EventID=1074 or EventID=1076) and "
        + span + "]]")))
    # EventID 1 is also used by other sources; keep the power ones.
    system = [e for e in system if 'Event ID: 1\n' not in e + '\n'
              or 'Power-Troubleshooter' in e]
    return app + system


def _last_lines(path: str, n: int = 6) -> list[str]:
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return [line.rstrip() for line in f.readlines()[-n:]]
    except Exception:
        return []


def previous_run_died(logs_dir: str, finish_files=None) -> dict | None:
    """If the last protocol run never ended (``logs/run_active.json`` is
    still there), find out what is known about it, record it in its
    run.log and in errors.log, finish its well files with
    ``finish_files(run_dir) -> list[str]`` and return
    ``{'run_dir', 'last', 'events', 'finished', 'report'}``. None when the
    last run ended."""
    active = os.path.join(logs_dir, ACTIVE_NAME)
    try:
        with open(active, 'r', encoding='utf-8') as f:
            info = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        info = {}
    run_dir = info.get('run_dir') or ''
    started = float(info.get('started') or 0.0)
    log_path = os.path.join(run_dir, 'run.log')
    try:
        last_t = os.path.getmtime(log_path)
    except OSError:
        last_t = started or time.time()
    last = _last_lines(log_path)
    events = windows_events(started or last_t - 3600, None)
    guard = [line for line in _last_lines(os.path.join(logs_dir,
                                                       'park_guard.log'), 12)
             if str(info.get('pid', '?')) in line or 'exit code' in line]
    finished: list[str] = []
    if finish_files is not None and run_dir and os.path.isdir(run_dir):
        try:
            finished = list(finish_files(run_dir))
        except Exception as e:
            finished = [f'finishing the well files failed: {e}']
    report = [f'The protocol run in {run_dir} did not end: LumaViewPro '
              f'stopped while it was going on.',
              f'Last entry in its run.log ({time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_t))}):']
    report += ['    ' + line for line in (last or ['(none)'])]
    if guard:
        report.append('Park guard:')
        report += ['    ' + line for line in guard[-3:]]
    if events:
        report.append('Windows recorded, from the start of the run until now:')
        for e in events:
            report += ['    ' + line for line in e.splitlines()]
    else:
        report.append('Windows recorded no crash, hang, sleep or restart in '
                      'that time.')
    report.append('See also logs/crash.log (stacks at a native crash or '
                  'a stall).')
    if finished:
        report.append('Well files finished now: ' + ', '.join(
            os.path.basename(p) for p in finished))
    text = '\n'.join(report)
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f'{_stamp()}  FOUND AT THE NEXT START: the run did not '
                    f'end\n{text}\n')
    except Exception:
        pass
    log_error(text)
    try:
        os.remove(active)
    except OSError:
        pass
    return {'run_dir': run_dir, 'last': last, 'events': events,
            'finished': finished, 'report': text}
