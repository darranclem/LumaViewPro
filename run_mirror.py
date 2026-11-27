"""Local first: a protocol run is written to a local disk, and this copies
it to where it is to be kept (usually a network drive) on a thread of its
own, so that the acquisition never waits for the network.

The network drive has refused writes in the middle of runs (``OSError 22``,
``Errno 2``), whichever way the images were saved. Nothing in this program
can stop a file server from dropping out; what it can do is not depend on
it while the microscope is working. So:

- the run folder is on the local disk (:func:`spool_for`); everything the
  run and the online processing write goes there;
- :class:`Mirror` walks that folder again and again and copies every file
  that is finished and not yet across: to ``<name>.part`` at the
  destination, read back and compared with what was sent, then renamed to
  its name. A file is at the destination whole and verified, or not there
  under its name at all;
- when the destination does not answer, the copying waits and tries again,
  for as long as it takes (hours, days). The run carries on meanwhile;
- nothing local is ever deleted by this;
- ``transfer.json`` in the local run folder says where the run goes and
  what is across. A run whose copying was cut short (the program was
  closed, the PC restarted) is carried on at the next start
  (:func:`resume_pending`), or from the command line:
  ``.venv\\Scripts\\python run_mirror.py <local run folder> [<destination>]``.

Not copied: ``_processing_planes`` (working files of the online
processing), ``*.tmp`` / ``*.part``, a file that is still being written
(an OME-TIFF with its ``.index.json`` beside it; a file changed in the last
``settle_s`` seconds) and the transfer's own two files.

No hardware and no UI in here.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

STATE_NAME = 'transfer.json'
LOG_NAME = 'transfer.log'
SKIP_DIRS = {'_processing_planes'}
SKIP_ENDS = ('.tmp', '.part', '.index.json')
CHUNK = 4 * 2 ** 20
_sleep = time.sleep


def is_network_path(path: str) -> bool:
    """True for a UNC path or a drive letter that is a network share."""
    path = os.path.abspath(path)
    if path.startswith('\\\\') or path.startswith('//'):
        return True
    if os.name != 'nt':
        return False
    try:
        import ctypes
        root = os.path.splitdrive(path)[0] + '\\'
        return ctypes.windll.kernel32.GetDriveTypeW(root) == 4   # DRIVE_REMOTE
    except Exception:
        return False


def _writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, f'.write_test_{os.getpid()}')
        with open(probe, 'w'):
            pass
        os.remove(probe)
        return True
    except Exception:
        return False


def default_spool_root() -> str:
    """``D:\\LumaViewPro_runs`` when there is a local D: disk (apart from
    Windows and the program), else ``C:\\LumaViewPro_runs``."""
    if os.name == 'nt':
        try:
            import ctypes
            if ctypes.windll.kernel32.GetDriveTypeW('D:\\') == 3:   # fixed
                return 'D:\\LumaViewPro_runs'
        except Exception:
            pass
        return 'C:\\LumaViewPro_runs'
    return os.path.join(os.path.expanduser('~'), 'LumaViewPro_runs')


def spool_for(destination: str, spool_root: str = '', fallback: str = '') -> str | None:
    """The local folder runs are written to when they are to be kept in
    ``destination``, or None when they are written there directly.

    ``spool_root`` chosen by the user: runs always go there first, unless
    it is the destination itself (or inside it). ``spool_root`` '': only a
    destination on a network drive is written locally first, in
    :func:`default_spool_root`. ``fallback`` is tried when the folder
    cannot be written (another account's); a network drive is never taken
    as the local folder."""
    chosen = spool_root.strip()
    if chosen:
        a = os.path.normcase(os.path.abspath(chosen))
        b = os.path.normcase(os.path.abspath(destination))
        if a == b or a.startswith(b.rstrip('\\/') + os.sep):
            return None
    elif not is_network_path(destination):
        return None
    for root in (chosen or default_spool_root(), fallback):
        if root and not is_network_path(root) and _writable(root):
            return root
    return None


def _hash_file(path: str) -> str:
    h = hashlib.blake2b(digest_size=20)
    with open(path, 'rb') as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class Changed(Exception):
    """The file was written to while it was being copied."""


class Mirror:
    """Copies ``src`` (a local run folder) to ``dst`` while the run goes
    on. ``start()``; ``finish()`` when nothing more will be written (the
    thread ends once everything is across); ``wait()``.

    ``status_cb(text)`` is told when the destination stops answering, when
    it is back, and when everything is across."""

    def __init__(self, src: str, dst: str, *, status_cb=None,
                 settle_s: float = 10.0, pause_s: float = 5.0):
        self.src = os.path.abspath(src)
        self.dst = dst
        self.status_cb = status_cb
        self.settle_s = float(settle_s)
        self.pause_s = float(pause_s)
        # relative path -> [size, time of last change (ns)] when it was copied
        self.done: dict[str, list[int]] = {}
        self.copied = 0             # copies made by this object
        self.bytes = 0
        self.pending = 0            # seen in the last pass, not yet across
        self.held = 0               # still open (never closed by the run)
        self.waiting = False        # the destination is not answering
        self.last_error = ''
        self.complete = False
        self._finishing = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._load_state()

    # ------------------------------------------------------------- state

    def _load_state(self) -> None:
        try:
            with open(os.path.join(self.src, STATE_NAME), encoding='utf-8') as f:
                d = json.load(f)
            if d.get('destination') == self.dst:
                self.done = {k: list(v) for k, v in (d.get('files') or {}).items()}
        except Exception:
            pass

    def _save_state(self, state: str) -> None:
        doc = {'destination': self.dst, 'state': state,
               'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
               'files_across': len(self.done), 'waiting_to_go': self.pending,
               'still_open': self.held, 'last_error': self.last_error,
               'files': self.done}
        path = os.path.join(self.src, STATE_NAME)
        try:
            with open(path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(doc, f)
            os.replace(path + '.tmp', path)
        except Exception:
            pass

    def _log(self, text: str) -> None:
        try:
            with open(os.path.join(self.src, LOG_NAME), 'a', encoding='utf-8') as f:
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {text}\n')
        except Exception:
            pass

    def _status(self, text: str) -> None:
        if self.status_cb is not None:
            try:
                self.status_cb(text)
            except Exception:
                pass

    def progress(self) -> str:
        """A few words for a status line."""
        if self.complete:
            return f'all {len(self.done)} files are on {self.dst}'
        text = f'{len(self.done)} files copied to {self.dst}, {self.pending} to go'
        if self.waiting:
            text += ' (the drive is NOT ANSWERING; the copying waits for it)'
        return text

    # ------------------------------------------------------------ copying

    def _candidates(self):
        """(relative path, full path, [size, mtime_ns]) of what is not
        across yet and may be copied now; counts the rest."""
        pending = held = 0
        out = []
        now = time.time()
        for root, dirs, names in os.walk(self.src):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            here = set(names)
            for name in sorted(names):
                if name.endswith(SKIP_ENDS) or (
                        root == self.src and name in (STATE_NAME, LOG_NAME)):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, self.src)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                sig = [int(st.st_size), int(st.st_mtime_ns)]
                if self.done.get(rel) == sig:
                    continue
                if name + '.index.json' in here:
                    held += 1                   # still open
                    continue
                pending += 1
                if now - st.st_mtime >= self.settle_s:
                    out.append((rel, full, sig))
        self.pending, self.held = pending, held
        return out

    def _copy(self, rel: str, full: str, sig: list[int]) -> None:
        target = os.path.join(self.dst, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            same = os.path.getsize(target) == sig[0]
        except OSError:
            same = False
        if same and _hash_file(target) == _hash_file(full):
            return                              # already there (an earlier session)
        part = target + '.part'
        sent = hashlib.blake2b(digest_size=20)
        with open(full, 'rb') as fin, open(part, 'wb') as fout:
            while True:
                block = fin.read(CHUNK)
                if not block:
                    break
                sent.update(block)
                fout.write(block)
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except OSError:
                pass
        st = os.stat(full)
        if [int(st.st_size), int(st.st_mtime_ns)] != sig:
            try:
                os.remove(part)
            except OSError:
                pass
            raise Changed(rel)
        if _hash_file(part) != sent.hexdigest():
            try:
                os.remove(part)
            except OSError:
                pass
            raise OSError(f'{rel}: what arrived is not what was sent')
        os.replace(part, target)
        self.copied += 1
        self.bytes += sig[0]

    def sync_once(self) -> bool:
        """One pass. True when the destination took everything it was
        given (False: it stopped answering part way)."""
        saved = time.monotonic()
        for rel, full, sig in self._candidates():
            if self._stop.is_set():
                return True
            try:
                self._copy(rel, full, sig)
            except Changed:
                continue                        # next pass
            except FileNotFoundError as e:
                if not os.path.exists(full):
                    continue                    # removed meanwhile
                self._failed(e)
                return False
            except OSError as e:
                self._failed(e)
                return False
            self.done[rel] = sig
            self.pending = max(self.pending - 1, 0)
            if self.waiting:
                self.waiting = False
                self._log('the destination answers again')
                self._status(f'{self.dst} is back: copying the run there again')
            if time.monotonic() - saved > 30.0:
                self._save_state('running')
                saved = time.monotonic()
        return True

    def _failed(self, e: BaseException) -> None:
        self.last_error = f'{type(e).__name__}: {e}'
        if not self.waiting:
            self.waiting = True
            self._log(f'the destination does not answer ({self.last_error}); '
                      f'waiting for it')
            self._status(f'COPYING TO {self.dst} IS WAITING: the drive is not '
                         f'answering ({e}). The run is not affected: the '
                         f'images are in {self.src} and are copied when the '
                         f'drive is back.')

    def _loop(self) -> None:
        self._log(f'copying {self.src} to {self.dst}')
        wait = self.pause_s
        while not self._stop.is_set():
            before = self.copied
            ok = False
            try:
                ok = self.sync_once()
            except Exception as e:              # never let the thread die
                self._failed(e)
            if ok and self._finishing.is_set() and self.pending == 0 \
                    and self.copied == before:
                break
            self._save_state('running')
            # Longer pauses while the destination is away (up to a minute).
            wait = self.pause_s if ok else min(max(wait, self.pause_s) * 2, 60.0)
            self._stop.wait(wait)
        if self._stop.is_set():
            self._save_state('running')
            return
        self.complete = self.held == 0
        self._save_state('complete' if self.complete else 'incomplete')
        if self.complete:
            self._log(f'complete: {len(self.done)} files on {self.dst}')
            self._status(f'The run is on {self.dst}: {len(self.done)} files '
                         f'copied and checked. The local copy stays in {self.src}')
        else:
            self._log(f'{self.held} files were never closed and are not copied')
            self._status(f'{self.held} files in {self.src} were never closed and '
                         f'were not copied: "well_file.py <folder> --finish", '
                         f'then "run_mirror.py <folder>"')

    def start(self) -> 'Mirror':
        with self._lock:
            if self._thread is None:
                self._save_state('running')
                self._thread = threading.Thread(target=self._loop, daemon=True,
                                                name='run-mirror')
                self._thread.start()
        return self

    def finish(self) -> None:
        """Nothing more will be written to the run folder: stop once
        everything is across."""
        self._finishing.set()

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> bool:
        t = self._thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True


def pending_runs(spool_root: str) -> list[tuple[str, str]]:
    """(local run folder, destination) of the runs below ``spool_root``
    whose copying did not get to the end."""
    out = []
    try:
        names = sorted(os.listdir(spool_root))
    except OSError:
        return out
    for name in names:
        src = os.path.join(spool_root, name)
        try:
            with open(os.path.join(src, STATE_NAME), encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get('state') != 'complete' and d.get('destination'):
            out.append((src, d['destination']))
    return out


def resume_pending(spool_root: str, status_cb=None, skip=()) -> list[Mirror]:
    """Carry on the copying of every run below ``spool_root`` that was cut
    short (started, not waited for)."""
    mirrors = []
    skip = {os.path.abspath(s) for s in skip}
    for src, dst in pending_runs(spool_root):
        if os.path.abspath(src) in skip:
            continue
        m = Mirror(src, dst, status_cb=status_cb)
        m.finish()
        mirrors.append(m.start())
    return mirrors


if __name__ == '__main__':                          # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(
        description='Copy a local run folder to where it is kept (verified; '
                    'waits for a drive that does not answer).')
    ap.add_argument('folder')
    ap.add_argument('destination', nargs='?',
                    help='default: the one in the folder\'s transfer.json')
    a = ap.parse_args()
    dest = a.destination
    if not dest:
        try:
            with open(os.path.join(a.folder, STATE_NAME), encoding='utf-8') as fh:
                dest = json.load(fh)['destination']
        except Exception:
            raise SystemExit('no destination given and none in transfer.json')
    mirror = Mirror(a.folder, dest, status_cb=print, settle_s=0.0)
    mirror.finish()
    mirror.start()
    while not mirror.wait(10.0):
        print(mirror.progress())
    raise SystemExit(0 if mirror.complete else 1)
