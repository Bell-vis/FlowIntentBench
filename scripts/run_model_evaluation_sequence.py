#!/usr/bin/env python3
"""Resume saved-answer evaluations in an explicit model order, with live progress.

The plan owns model order and collection paths. This runner never starts a
solver. Scientific bounds are retained; review completion is not point-score
identification. Each model gets independent reports, receipts and cost records.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from scripts.evaluate_model_answers import VERSION, read, sha
from scripts.portable_fcntl import fcntl


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(output, model_id):
    """Use atomic per-answer files so a long batch does not hide completions."""
    rows = [read(p) for p in (output / "cases").glob("*.json")]
    rows = [r for r in rows if r.get("model_id") == model_id]
    answered = [r for r in rows if r["collection_status"] == "COMPLETED"]
    reviewed = [r for r in answered if r.get("protocol") == VERSION and (r["status"] == "MODEL_NONCOMPLETION" or
                (r["status"] == "SCORED" and r.get("provenance", {}).get("source") in
                 {"TRUSTED_REVIEW_CACHE", "TRUSTED_HTTP_REVIEW"}))]
    # Receipts survive a process stop before the final invocation cost report.
    receipts = [read(p) for p in (output / "api").glob("*/receipt.json")]
    sent_receipts = [r for r in receipts
                     if "Shared API admission deadline exceeded before sending" not in str(r.get("error", ""))]
    usage = {key: sum(r.get("usage", {}).get(key, 0) for r in receipts)
             for key in ("input_tokens", "output_tokens")}
    metric_states = Counter(m["status"] for r in reviewed for m in r["metrics"].values())
    return {"model_id": model_id, "updated_utc": now(), "requested_slots": len(rows),
        "completed_answers": len(answered), "reviewed_answers": len(reviewed),
        "remaining_reviews": len(answered) - len(reviewed),
        "row_status_counts": dict(Counter(r["status"] for r in rows)),
        "metric_status_counts": dict(metric_states),
        "extraction_incomplete_answers": sum(r.get("extraction_complete") is False for r in reviewed),
        "new_api_attempts": len(receipts), "finished_api_attempts": sum("finished_epoch" in r for r in receipts),
        "local_admission_failures": len(receipts) - len(sent_receipts),
        "api_failures": sum("finished_epoch" in r and not r.get("completed") for r in sent_receipts),
        "usage_missing_finished_attempts": sum("finished_epoch" in r and not all(k in r.get("usage", {}) for k in usage) for r in sent_receipts),
        **usage, "new_total_tokens": sum(usage.values()),
        "review_completion": bool(answered) and len(reviewed) == len(answered),
        "scientific_identification": bool(answered) and len(reviewed) == len(answered)
            and not metric_states["INTERVAL"] and not any(m.get("applicability_unknown") for r in reviewed for m in r["metrics"].values()),
        "report": str(output / "reports/experiment_report.json")}


def command(args, item):
    result = [sys.executable, str(ROOT / "scripts/evaluate_model_answers.py"),
        "--collection", str(Path(item["collection"]).resolve()), "--models", item["model_id"],
        "--output", str(args.output / item["model_id"]),
        "--auth-path", str(args.auth_path), "--api-config", str(args.api_config),
        "--max-api-calls", str(args.batch_calls), "--max-wall-seconds", str(args.batch_seconds),
        "--timeout", str(args.timeout), "--workers", str(args.workers),
        "--shared-concurrency", str(args.shared_concurrency)]
    for source in args.reuse_trusted_from:
        result.extend(["--reuse-trusted-from", str(source)])
    if args.evidence_bank:
        result.extend(["--evidence-bank", str(args.evidence_bank)])
    if args.reference_package:
        result.extend(["--reference-package", str(args.reference_package)])
    if item.get("donor"):
        result.extend(["--donor", str(item["donor"])])
    replay_root = getattr(args, "replay_root", None)
    if replay_root and (replay_root / item["model_id"]).is_dir():
        result.extend(["--replay-from", str((replay_root / item["model_id"]).resolve())])
    for source in getattr(args,'additional_replay_from',[]) or []:
        result.extend(['--additional-replay-from',str(source.resolve())])
    if getattr(args,'structured_output',False):
        result.append('--structured-output')
    repairs_root = getattr(args, "review_repairs_root", None)
    if repairs_root and (repairs_root / item["model_id"]).is_dir():
        result.extend(["--review-repairs", str((repairs_root / item["model_id"]).resolve())])
    return result


def run_child(args, item, state):
    output = args.output / item["model_id"]
    output.mkdir(parents=True, exist_ok=True)
    log = output / "sequence_console.jsonl"
    child = subprocess.Popen(command(args, item), cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    state.update(child_pid=child.pid, active_model=item["model_id"], updated_utc=now())
    write_json(args.output / "sequence_state.json", state)
    messages = queue.Queue()
    def read_stdout():
        for line in child.stdout:
            messages.put(line)
    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    # Preserve cadence across batches; short batches must not continually reset
    # the user's 180-second reporting deadline.
    next_tick = state.setdefault("next_progress_epoch", time.time() + args.progress_seconds)
    with log.open("a", encoding="utf-8") as handle:
        while child.poll() is None or reader.is_alive() or not messages.empty():
            try:
                line = messages.get(timeout=1)
                handle.write(line)
                handle.flush()
                # Live task output remains small; the full traceback is saved.
                if line.startswith("{"):
                    print(line.rstrip(), flush=True)
            except queue.Empty:
                pass
            if time.time() >= next_tick:
                current = snapshot(output, item["model_id"])
                state["models"][item["model_id"]] = current
                state["updated_utc"] = now()
                next_tick = time.time() + args.progress_seconds
                state["next_progress_epoch"] = next_tick
                write_json(args.output / "sequence_state.json", state)
                write_json(output / "live_progress.json", current)
                print(json.dumps({"event": "PROGRESS_180_SECONDS", **current}), flush=True)
    reader.join(timeout=2)
    state["child_pid"] = None
    return child.returncode


def run(args):
    if getattr(args, "reuse_trusted_from", []):
        raise ValueError("v4 requires fresh reviews; do not import legacy/cross-run judgments")
    plan = read(args.plan)
    items = plan["models"]
    if not items or len({i["model_id"] for i in items}) != len(items):
        raise ValueError("ordered plan needs distinct nonempty model entries")
    if any(Path(i["model_id"]).name != i["model_id"] or i["model_id"] in {".", ".."} for i in items):
        raise ValueError("model directory names must be simple path components")
    state_path = args.output / "sequence_state.json"
    state = read(state_path) if state_path.exists() else {"models": {}, "batches": []}
    if state_path.exists() and state.get("protocol") != VERSION:
        raise ValueError("sequence output belongs to an older protocol; start a new v4 directory")
    if state.get("plan_sha256", sha(args.plan)) != sha(args.plan):
        raise ValueError("sequence plan changed; preserve the prior run in its own output")
    state.update(protocol=VERSION, plan_sha256=sha(args.plan), status="RUNNING", ordered_models=[i["model_id"] for i in items],
                 runner_pid=__import__("os").getpid(), updated_utc=now())
    write_json(state_path, state)
    for item in items:
        model = item["model_id"]
        output = args.output / model
        stalled = 0
        before = snapshot(output, model)
        # Always replay once to revalidate ledger/answer/GT identities. A prior
        # complete report alone does not prove current inputs are unchanged.
        while True:
            started = now()
            returncode = run_child(args, item, state)
            after = snapshot(output, model)
            state["models"][model] = after
            state["batches"].append({"model_id": model, "started_utc": started, "finished_utc": now(),
                "returncode": returncode, "new_reviewed_answers": after["reviewed_answers"] - before["reviewed_answers"],
                "new_api_attempts": after["new_api_attempts"] - before["new_api_attempts"]})
            stalled = stalled + 1 if after["reviewed_answers"] <= before["reviewed_answers"] else 0
            report_path = output / "reports/experiment_report.json"
            costs = read(report_path).get("evaluation_cost", {}) if report_path.exists() else {}
            state["updated_utc"] = now()
            if returncode not in {0, 2} or stalled >= 2 or costs.get("provider_circuit_open") or costs.get("startup_error"):
                state.update(status="NEEDS_INTERVENTION", reason="Provider, source or scoring failure; inspect current receipts/logs before retrying.")
                write_json(state_path, state)
                print(json.dumps({"event": "NEEDS_INTERVENTION", **after}), flush=True)
                return 2
            write_json(state_path, state)
            print(json.dumps({"event": "BATCH_FINISHED", **after}), flush=True)
            before = after
            if after["review_completion"]:
                break
        state["models"][model] = before
        print(json.dumps({"event": "MODEL_REVIEWS_COMPLETE", **before}), flush=True)
    state.update(status="ALL_EXISTING_ANSWERS_REVIEWED", active_model=None, child_pid=None, updated_utc=now())
    state["scientific_identification"] = all(r["scientific_identification"] for r in state["models"].values())
    write_json(state_path, state)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/model_answer_evaluation_v4/by_model")
    p.add_argument("--auth-path", type=Path, default=ROOT / "auth_yapi.json")
    p.add_argument("--api-config", type=Path, default=ROOT / "config/yiapi_direct.toml")
    p.add_argument("--reuse-trusted-from", action="append", type=Path, default=[])
    p.add_argument("--evidence-bank", type=Path)
    p.add_argument("--reference-package", type=Path)
    p.add_argument("--replay-root", type=Path, help="Explicit same-protocol host repair; per-model source directories must contain identical reviewer requests")
    p.add_argument('--additional-replay-from',type=Path,action='append',default=[],help='Exact-request pilot replay sources shared by the sequence')
    p.add_argument("--review-repairs-root", type=Path, help="Frozen source-bound review repairs in per-model directories; do not modify during a run")
    p.add_argument("--batch-calls", type=int, default=12)
    p.add_argument("--batch-seconds", type=float, default=600)
    p.add_argument("--timeout", type=float, default=150)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shared-concurrency", type=int, default=4)
    p.add_argument("--progress-seconds", type=float, default=180)
    p.add_argument('--structured-output',action='store_true',help='Use strict JSON output for every new reviewer request')
    args = p.parse_args()
    if min(args.batch_calls, args.batch_seconds, args.timeout, args.workers, args.shared_concurrency, args.progress_seconds) <= 0:
        raise SystemExit("budgets and progress interval must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".sequence.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
