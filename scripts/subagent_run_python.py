"""Standalone journaled Python executor, copied into each blind slot's work dir.

Usage: python run_python.py analysis.py
Every invocation starts a fresh Python interpreter. This helper enforces the
execution count/deadline, but is not a filesystem or network security boundary.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> int:
    work = Path(__file__).resolve().parent
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_python.py analysis.py")
    source = Path(sys.argv[1]).resolve()
    if not source.is_relative_to(work) or source.suffix != ".py":
        raise SystemExit("Analysis source must be a .py file in this slot's work directory")
    with (work / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        budget = json.loads((work / "execution_budget.json").read_text())
        if (work / "finished.json").exists():
            raise SystemExit("Slot already finished")
        journal = work / "execution_journal.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        remaining = budget["deadline_epoch"] - time.time()
        if remaining <= 0 or len(rows) >= budget["max_python_executions"]:
            raise SystemExit("Slot execution budget exhausted")
        code = source.read_text()
        index = len(rows) + 1
        snapshot = work / "execution_sources" / f"{index:03d}.py"
        snapshot.parent.mkdir(exist_ok=True)
        snapshot.write_text(code)
        started = time.time()
        row = {"event": "python_execution", "execution_index": index,
               "source": str(source.relative_to(work)), "code": code,
               "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
               "started_epoch": started, "returncode": None,
               "stdout": "", "stderr": "", "duration_seconds": 0.0,
               "finished_epoch": None}
        # Reserve the invocation before execution so process loss cannot erase it.
        rows.append(row)
        journal.write_text("".join(json.dumps(item) + "\n" for item in rows))
        process = subprocess.Popen([sys.executable, str(snapshot)], cwd=work,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            row["timed_out"] = True
        row.update(returncode=process.returncode,
                   stdout=stdout.decode("utf-8", errors="replace"),
                   stderr=stderr.decode("utf-8", errors="replace"),
                   finished_epoch=time.time(), duration_seconds=time.time() - started)
        temporary = journal.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(item) + "\n" for item in rows))
        temporary.replace(journal)
        sys.stdout.write(row["stdout"])
        sys.stderr.write(row["stderr"])
        return 0 if process.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
