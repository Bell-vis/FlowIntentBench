"""Standalone journal helper for Claude's scoped Python tool.

Linux Landlock confines reads to this case and installed libraries; seccomp
denies networking and process-inspection escapes. No namespaces are created.
The trusted parent owns the journal; model code can write only scratch/.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import signal
import subprocess
import sys
import time


def restrict(read_paths, scratch):
    if platform.machine() != 'x86_64' or sys.platform != 'linux':
        raise RuntimeError('Blind Python requires Linux x86_64 Landlock and seccomp')
    libc = ctypes.CDLL(None, use_errno=True)
    def check(result):
        if result < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        return result
    abi = check(libc.syscall(444, 0, 0, 1))
    if abi < 3:
        raise RuntimeError('Landlock ABI >= 3 required; refusing unconfined execution')
    handled = (1 << 15) - 1  # all filesystem rights through TRUNCATE
    class Ruleset(ctypes.Structure):
        _fields_ = [('access', ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('access', ctypes.c_uint64), ('parent_fd', ctypes.c_int)]
    fd = check(libc.syscall(444, ctypes.byref(Ruleset(handled)), 8, 0))
    try:
        for path, write in [(Path(p), False) for p in read_paths] + [(Path(scratch), True)]:
            if not path.exists():
                continue
            rights = handled if write else (1 | 4 | (8 if path.is_dir() else 0))
            parent = os.open(path, getattr(os, 'O_PATH', 0o10000000) | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, parent)
                check(libc.syscall(445, fd, 1, ctypes.byref(rule), 0))
            finally:
                os.close(parent)
        check(libc.prctl(38, 1, 0, 0, 0))  # no_new_privs
        check(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)
    class Filter(ctypes.Structure):
        _fields_ = [('code', ctypes.c_ushort), ('jt', ctypes.c_ubyte),
                    ('jf', ctypes.c_ubyte), ('k', ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ushort), ('filter', ctypes.POINTER(Filter))]
    # Reject alternate ABI and x32, sockets, tracing, kernel/namespace changes,
    # process signalling, and alternate asynchronous file access mechanisms.
    denied = [41, 42, 43, 49, 50, 53, 62, 90, 91, 92, 93, 94, 101, 109, 112,
              129, 132, 155, 157, 161, 165, 166, 175, 176, 188, 189, 190, 197,
              198, 199, 200, 234, 235, 246, 248, 249, 250, 260, 268, 272, 280,
              288, 297, 298, 303, 304, 308, 310, 311, 313, 317, 321, 323,
              424, 425, 426, 427, 438, 452]
    rows = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, 0xc000003e),
            Filter(0x06, 0, 0, 0x80000000), Filter(0x20, 0, 0, 0),
            Filter(0x35, 0, 1, 0x40000000), Filter(0x06, 0, 0, 0x80000000)]
    for number in denied:
        rows += [Filter(0x15, 0, 1, number), Filter(0x06, 0, 0, 0x50000 | errno.EPERM)]
    rows += [Filter(0x06, 0, 0, 0x7fff0000)]
    array = (Filter * len(rows))(*rows)
    check(libc.prctl(22, 2, ctypes.byref(Program(len(rows), array)), 0, 0))


def child(source, work):
    # The helper owns the execution deadline. Its death must not leave an
    # analysis process running after a CLI timeout or interrupted controller.
    expected_parent = int(os.environ.pop('FIB_HELPER_PARENT_PID'))
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError('Cannot bind analysis lifetime to the journal helper')
    if os.getppid() != expected_parent:
        raise RuntimeError('Journal helper exited before analysis startup')
    scope = json.loads((work / 'blind_scope.json').read_text())
    scratch = work / 'scratch'
    libraries = [sys.prefix, '/usr', '/lib', '/lib64', '/etc/ld.so.cache',
                 '/etc/localtime', '/etc/fonts', '/dev/null', '/dev/urandom']
    restrict([*libraries, *scope['read_paths'], source], scratch)
    os.chdir(scratch)
    sys.path = [p for p in sys.path if p and Path(p).resolve().is_relative_to(Path(sys.prefix))]
    runpy.run_path(str(source), run_name='__main__')


def main():
    work = Path(__file__).resolve().parent
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        child(Path(sys.argv[2]).resolve(), work)
        return 0
    if len(sys.argv) != 2:
        raise ValueError('Usage: python run_python.py analysis.py')
    source = Path(sys.argv[1]).resolve()
    if source.parent != work or source.suffix != '.py' or source.name == 'run_python.py':
        raise ValueError('Source must be an analysis file in the assigned work directory')
    with (work / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads((work / 'execution_budget.json').read_text())
        journal = work / 'execution_journal.jsonl'
        rows = [json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []
        remaining = budget['deadline_epoch'] - time.time()
        if (work / 'finished.json').exists() or remaining <= 0 or len(rows) >= budget['max_python_executions']:
            raise ValueError('Execution budget exhausted or slot finished')
        code = source.read_text()
        index = len(rows) + 1
        snapshot = work / 'execution_sources' / f'{index:03d}.py'
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(code)
        (work / 'scratch').mkdir(exist_ok=True)
        row = dict(event='python_execution', execution_index=index, source=source.name,
                   code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                   started_epoch=time.time(), returncode=None, stdout='', stderr='',
                   duration_seconds=0., finished_epoch=None)
        rows.append(row)
        def save():
            tmp = journal.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            tmp.replace(journal)
        save()
        env = {'PATH': str(Path(sys.executable).parent) + ':/usr/bin:/bin',
               'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'FIB_HELPER_PARENT_PID': str(os.getpid()),
               'TMPDIR': str(work / 'scratch'), 'MPLCONFIGDIR': str(work / 'scratch' / 'mpl')}
        process = subprocess.Popen([sys.executable, '-I', str(work / 'run_python.py'), '--child', str(snapshot)],
                                   cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            row['timed_out'] = True
        finally:
            # Reap any descendants even when their leader has exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        row.update(returncode=process.returncode, stdout=stdout.decode(errors='replace'),
                   stderr=stderr.decode(errors='replace'), finished_epoch=time.time(),
                   duration_seconds=time.time() - row['started_epoch'])
        save()
        sys.stdout.write(row['stdout'])
        sys.stderr.write(row['stderr'])
        return int(process.returncode != 0)


if __name__ == '__main__':
    raise SystemExit(main())
