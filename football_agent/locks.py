"""Crash-releasing, reentrant process locks for one Agent run at a time.

The lock file is permanent: removing a locked file can split POSIX lockers
across different inodes. Ownership lives in the operating system, never in a
PID/mtime lease. A process-local RLock also excludes different threads because
OS file-lock behavior alone does not consistently provide that guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import math
import os
from pathlib import Path
from threading import Lock, RLock
import time
from weakref import WeakValueDictionary


class RunBusyError(RuntimeError):
    """Another thread/process owns the requested run; no work was performed."""

    code = "RUN_BUSY"

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"RUN_BUSY: run {run_id} is executing in another thread or process")


class _LockState:
    def __init__(self):
        self.mutex = RLock()
        self.depth = 0
        self.fd: int | None = None


_registry_guard = Lock()
_registry: WeakValueDictionary[str, _LockState] = WeakValueDictionary()


def _after_fork():
    # flock descriptors inherited through fork share ownership. Close them in
    # the child, without LOCK_UN, and discard any inherited thread mutexes.
    global _registry_guard, _registry
    for state in list(_registry.values()):
        if state.fd is not None:
            os.close(state.fd)
    _registry_guard = Lock()
    _registry = WeakValueDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _try_os_lock(fd: int):
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(fd: int):
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def run_lock(path: Path, run_id: str, timeout: float = 0.0):
    """Acquire one lock across threads, RunStore instances, and processes.

    ``timeout=0`` fails promptly with RunBusyError; a positive timeout waits at
    most that many seconds, including time spent waiting for another thread.
    Reentry by the owning thread is supported, even through another RunStore.
    All callers must use the same resolved store path for a given run.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Lock timeout must be a finite nonnegative number")
    deadline = time.monotonic() + timeout
    path = Path(path).resolve()
    key = os.path.normcase(str(path))
    with _registry_guard:
        state = _registry.get(key)
        if state is None:
            state = _LockState()
            _registry[key] = state
    remaining = max(0.0, deadline - time.monotonic())
    acquired = state.mutex.acquire(blocking=False) if remaining == 0 else state.mutex.acquire(timeout=remaining)
    if not acquired:
        raise RunBusyError(run_id)
    try:
        if state.depth:
            state.depth += 1
            try:
                yield
            finally:
                state.depth -= 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        locked = False
        try:
            while True:
                try:
                    _try_os_lock(fd)
                    locked = True
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RunBusyError(run_id) from None
                    time.sleep(min(0.025, remaining))
            # OS byte-range locks can cover EOF. Initialize only after owning
            # the byte so concurrent first openers never write through a lock.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            state.fd, state.depth = fd, 1
            try:
                yield
            finally:
                state.fd, state.depth = None, 0
        finally:
            try:
                if locked:
                    _release_os_lock(fd)
            finally:
                os.close(fd)
    finally:
        state.mutex.release()
