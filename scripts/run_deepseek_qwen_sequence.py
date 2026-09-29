#!/usr/bin/env python3
"""Run DeepSeek's 96 N=1 cases, then Qwen's; monitor every 100 seconds."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_deepseek_qwen_cases as collector
from scripts.benchmark_runtime import require_benchmark_runtime


def snapshot(output, model, *, phase, pid=None, returncode=None):
    path = output / "collection_state.json"
    state = collector.original._read(path) if path.exists() else {}
    counts = {name: dict(Counter(s["status"] for s in state.get("slots", []) if s["model_id"] == name))
              for name in collector.MODELS}
    payload = {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
               "active_model": model, "pid": pid, "returncode": returncode, "by_model": counts}
    with (output / "monitor.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    collector.original._write(output / "sequence_status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def model_complete(output, model):
    path = output / "collection_state.json"
    if not path.exists():
        return False
    state = collector.original._read(path)
    slots = [slot for slot in state["slots"] if slot["model_id"] == model]
    if len(slots) != 96 or len({slot["case_id"] for slot in slots}) != 96:
        return False
    for slot in slots:
        if slot["status"] != "COMPLETED" or not slot.get("run_record_path"):
            return False
        record_path = (output / slot["run_record_path"]).resolve()
        if not record_path.is_relative_to(output.resolve()) or not record_path.is_file():
            return False
        if collector.original._sha256(record_path) != slot.get("run_record_sha256"):
            return False
        record = collector.original._read(record_path)
        answer = record_path.parent / "final_answer.md"
        if (record.get("model_id") != model or record.get("case_id") != slot["case_id"]
                or record.get("trial_index") != 1 or record.get("run_status") != "COMPLETED"
                or not answer.is_file() or not answer.read_text(encoding="utf-8").strip()
                or collector.original._sha256(answer) != record.get("final_answer_sha256")):
            return False
    return True


def run(args):
    output = args.output_root.resolve()
    with collector.original._output_lock(output, ".sequence.lock"):
        collector.prepare(output, host_network=getattr(args, "host_network", False))
        child = None
        stopping = False

        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for model in collector.MODELS:
                if stopping:
                    snapshot(output, model, phase="STOPPED")
                    return 2
                if model_complete(output, model):
                    snapshot(output, model, phase="MODEL_ALREADY_COMPLETE")
                    continue
                command = [sys.executable, str(ROOT / "scripts/run_deepseek_qwen_cases.py"),
                           "collect", "--model", model, "--output-root", str(output),
                           "--auth", str(args.auth.resolve()), "--workers", str(args.workers)]
                if args.retry_infrastructure:
                    command.append("--retry-infrastructure")
                if getattr(args, "host_network", False):
                    command.append("--host-network")
                command.extend(["--max-infrastructure-retries", str(getattr(args, "max_infrastructure_retries", 1))])
                with (output / f"{model}.collector.log").open("a", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    snapshot(output, model, phase="STARTED", pid=child.pid)
                    next_check = time.monotonic() + args.interval
                    while child.poll() is None:
                        remaining = max(0, next_check - time.monotonic())
                        try:
                            child.wait(timeout=min(5, remaining))
                        except subprocess.TimeoutExpired:
                            pass
                        if time.monotonic() >= next_check:
                            snapshot(output, model, phase="RUNNING", pid=child.pid)
                            next_check += args.interval
                returncode = child.returncode
                complete = model_complete(output, model)
                snapshot(output, model, phase="MODEL_COMPLETE" if complete else "BLOCKED",
                         pid=child.pid, returncode=returncode)
                child = None
                if not complete:
                    print(f"Collection stopped. See {output / (model + '.collector.log')}", flush=True)
                    return 2
            snapshot(output, None, phase="COMPLETE")
            return 0
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--host-network", action="store_true")
    parser.add_argument("--auth", type=Path, default=ROOT / "auth.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--interval", type=float, default=100)
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--max-infrastructure-retries", type=int, default=1)
    args = parser.parse_args()
    args.output_root = args.output_root or (collector.HOST_NETWORK_OUTPUT if args.host_network else collector.DEFAULT_OUTPUT)
    if not 1 <= args.workers <= 8 or args.interval <= 0:
        parser.error("workers must be 1..8 and interval must be positive")
    if not 0 <= args.max_infrastructure_retries <= 3:
        parser.error("max-infrastructure-retries must be 0..3")
    require_benchmark_runtime()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
