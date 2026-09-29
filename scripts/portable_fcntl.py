"""Small ``fcntl.flock`` compatibility surface for repository lock files."""

from __future__ import annotations

import os
import time

try:  # pragma: no cover - exercised on POSIX hosts
    import fcntl as fcntl
except ModuleNotFoundError:  # pragma: no cover - selected on Windows
    if os.name != "nt":
        raise

    import msvcrt

    class _WindowsFcntl:
        LOCK_EX = 1
        LOCK_NB = 2
        LOCK_UN = 4

        @staticmethod
        def _position_lock_byte(stream) -> None:
            stream.flush()
            descriptor = stream.fileno()
            if os.fstat(descriptor).st_size == 0:
                stream.seek(0)
                if hasattr(stream, 'buffer'):
                    stream.buffer.write(b"\0")
                else:
                    stream.write(b"\0")
                stream.flush()
            os.lseek(descriptor, 0, os.SEEK_SET)

        @classmethod
        def flock(cls, stream, operation: int) -> None:
            cls._position_lock_byte(stream)
            descriptor = stream.fileno()
            if operation & cls.LOCK_UN:
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                return

            nonblocking = bool(operation & cls.LOCK_NB)
            while True:
                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    return
                except OSError as exc:
                    if nonblocking:
                        raise BlockingIOError(
                            exc.errno, "lock is already held", stream.name
                        ) from None
                    time.sleep(0.05)

    fcntl = _WindowsFcntl()


def process_exists(pid: int) -> bool:
    """Return whether a PID is live without sending it a Windows signal."""

    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


__all__ = ["fcntl", "process_exists"]
