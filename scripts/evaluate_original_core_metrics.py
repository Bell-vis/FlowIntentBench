"""Evaluate existing answers into the frozen O/URS/F/C metric contract.

This is a text-to-judgment adapter. The host arithmetic remains in
``flowintentbench.core_case_scoring`` and ``flowintentbench.evaluation_metrics``;
this script only obtains the structured extraction needed by those scorers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import (CORE_SCHEMA_VERSION, CoreCaseScorer,
                                                SCORER_IMPLEMENTATION_VERSION,
                                                _dimensions, judgment_schema, scoring_prompt)
from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import RunRecord
from scripts.evaluate_answered_outcomes import sha_file
from scripts.outcome_responses_transport import ResponsesOutcomeReviewer
from scripts.run_core_case_scoring import _parse_json
from scripts.benchmark_runtime import require_benchmark_runtime


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def metric_mean(rows, key):
    values = [row["metrics"].get(key) for row in rows if row.get("status") == "CORE_SCORED"
              and row["metrics"].get(key) is not None]
    return {"mean": statistics.fmean(values) if values else None, "n": len(values),
            "applicable": sum(row.get("metrics", {}).get(key) is not None
                               for row in rows if row.get("status") == "CORE_SCORED")}


def run(args):
    source = read_json(args.selection)
    manifest = {row["case_id"]: row for row in read_json(args.manifest)["cases"]}
    rows = source["answers"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", {**source, "adapter": "frozen-core-metrics-text-v1",
                                            "scorer_version": SCORER_IMPLEMENTATION_VERSION,
                                            "reviewer_model": args.reviewer_model})
    write_json(output / "metric_contract.json", {"metrics": ["o_score", "urs", "resolved_o_compliance",
        "finding_precision", "core_finding_recall", "finding_requirement_recall", "adequate_core_complete",
        "c_score", "branch_alignment"], "source": "flowintentbench.core_case_scoring",
        "formulas_unchanged": True, "pending_is_not_zero": True})
    reviewer = ResponsesOutcomeReviewer(args.api_config, concurrency=args.workers) if not args.offline else None
    prefill_cache = {}
    if args.prefill_source:
        prefill_root = args.prefill_source.resolve()
        for selected in rows:
            prior_path = prefill_root / "cases" / (selected["run_id"] + ".json")
            if not prior_path.exists():
                continue
            prior = read_json(prior_path)
            prior_id = prior.get("request_id") or prior.get("baseline_request_id")
            if not prior_id:
                continue
            judgment_path = prefill_root / "judgments" / (prior_id + ".json")
            if not judgment_path.exists():
                continue
            prior_judgment = read_json(judgment_path).get("judgment")
            if isinstance(prior_judgment, dict):
                prefill_cache[selected["run_id"]] = {
                    "reviewer_source": prior.get("reviewer_source"),
                    "method_ratings": [item for item in prior_judgment.get("ratings", [])
                                       if str(item.get("item_id", "")).startswith("method:")],
                    "branch_relations": prior_judgment.get("branch_relations", []),
                    "claims": prior_judgment.get("claims", []),
                    "claim_verification": prior.get("verification"),
                }
    lock = threading.Lock()
    progress = {"status": "RUNNING", "expected": len(rows), "completed": 0, "scored": 0,
                "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "progress.json", progress)

    def process(row):
        run_record_path = Path(row["run_path"])
        record = RunRecord.from_dict(read_json(run_record_path))
        case_input, metadata, gt, _material = load_development_case(ROOT, manifest[row["case_id"]])
        case_payload = case_input.model_dump(mode="json")
        case_payload.update(case_id=record.case_id,
                            principal_operationalization_dimensions=list(
                                getattr(metadata, "principal_operationalization_dimensions", ())),
                            unresolved_operationalization_dimensions=list(
                                getattr(metadata, "unresolved_operationalization_dimensions", ())))
        answer = record.final_response
        gt_packet = {"schema": judgment_schema(_dimensions({
            "operationalizations": [], "branches": []}))} if False else None
        prompt = scoring_prompt(case_payload, answer, gt)
        prefill = prefill_cache.get(row["run_id"])
        if prefill:
            prompt += ("\n\nPRIOR REVIEW EVIDENCE (untrusted extraction aid, never a score):\n"
                       "Reuse exact quotations and explicit values only when they are present in ANSWER. "
                       "You must still return every original-core field, including per-branch dimension "
                       "matches, finding eligibility/role/consistency and the original schema. Do not "
                       "copy outcome-rubric MET/PARTIAL labels into boolean fields.\n"
                       + json.dumps(prefill, ensure_ascii=False, separators=(",", ":")))
        schema = judgment_schema(_dimensions({
            "operationalizations": [{"branch_id": b.operationalization_id,
                                     "decisions": [{"dimension": getattr(d.dimension, "value", d.dimension)}
                                                    for d in b.decisions]}
                                    for b in gt.acceptable_operationalizations],
            "branches": []}))
        identity = digest({"prompt": prompt, "schema": schema, "answer_sha256": row["answer_sha256"],
                           "scorer": SCORER_IMPLEMENTATION_VERSION, "model": args.reviewer_model})
        result_path = output / "cases" / (row["run_id"] + ".json")
        request_path = output / "requests" / (identity + ".json")
        judgment_path = output / "judgments" / (identity + ".json")
        write_json(request_path, {"request_id": identity, "prompt": prompt, "schema": schema,
                                  "scorer_version": SCORER_IMPLEMENTATION_VERSION, "run_id": row["run_id"]})
        judgment = None
        receipts = []
        error = None
        if judgment_path.exists():
            cached = read_json(judgment_path)
            judgment = cached["judgment"]
            receipts = cached.get("reviewer_receipts", [])
        elif not args.offline:
            for attempt in range(args.attempts + 1):
                token = f"{identity}-{time.time_ns()}"
                receipt_path = output / "api" / token / "receipt.json"
                receipts.append(str(receipt_path))
                current_prompt = prompt if not error else prompt + "\nReturn a corrected complete judgment; prior host error: " + error
                receipt = reviewer(current_prompt, args.reviewer_model, output / "work" / token,
                                   output / "api" / token, timeout_seconds=args.timeout,
                                   output_schema=schema, reasoning_effort=args.effort)
                if receipt.get("completed"):
                    try:
                        judgment = _parse_json(receipt["final_text"])
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                else:
                    error = receipt.get("error") or "reviewer did not complete"
        if judgment is not None:
            try:
                efficiency = {key: getattr(record, key, None) for key in (
                    "input_tokens", "output_tokens", "model_turn_count", "tool_call_count",
                    "python_execution_count", "wall_clock_time", "provider_reported_cost")}
                result = CoreCaseScorer().score(case_payload, answer, gt, judgment=judgment,
                                                efficiency=efficiency)
                result.update(row, request_id=identity, reviewer_model=args.reviewer_model,
                              reviewer_receipts=receipts, reasoning_effort=args.effort,
                              judgment_sha256=digest(judgment), run_record_sha256=sha_file(run_record_path))
                write_json(judgment_path, {"request_id": identity, "judgment": judgment,
                                          "reviewer_receipts": receipts})
            except Exception as exc:
                result = {**row, "status": "REVIEW_ERROR", "error": f"{type(exc).__name__}: {exc}",
                          "request_id": identity}
        else:
            result = {**row, "status": "PENDING", "pending_reason": error or "missing offline judgment",
                      "request_id": identity}
        write_json(result_path, result)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, row) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with lock:
                progress["completed"] += 1
                progress["scored"] += int(result.get("status") == "CORE_SCORED")
                write_json(output / "progress.json", progress)
    results.sort(key=lambda row: (row["model_id"], row["case_id"], row["trial"]))
    metrics = ["o_score", "urs", "resolved_o_compliance", "finding_precision", "core_finding_recall",
               "finding_requirement_recall", "adequate_core_complete", "c_score", "branch_alignment"]
    report = {"protocol": "frozen-core-metrics-text-v1", "status": "COMPLETE" if all(
                  row.get("status") == "CORE_SCORED" for row in results) else "INCOMPLETE",
              "expected": len(rows), "status_counts": dict(Counter(row.get("status") for row in results)),
              "metrics": {key: metric_mean(results, key) for key in metrics},
              "results": results, "reviewer_model": args.reviewer_model,
              "metric_definitions_source": "flowintentbench.evaluation_metrics and core_case_scoring",
              "formula_changes": False}
    write_json(output / "reports/experiment_report.json", report)
    return report


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("trusted", "legacy"), default="trusted")
    parser.add_argument("--collection", type=Path, default=ROOT / "outputs/claude_resume_eval")
    parser.add_argument("--max-api-calls", type=int, default=8)
    parser.add_argument("--max-wall-seconds", type=float, default=300)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_normalized_177/selection.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_original_core_v4")
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--auth-path", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--reviewer-model", default="gpt-6-astra")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prefill-source", type=Path,
                        default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    args = parser.parse_args()
    if args.engine == "trusted":
        from scripts.evaluate_model_answers import parser as trusted_parser, run as trusted_run
        from scripts.portable_fcntl import fcntl
        trusted = trusted_parser().parse_args([])
        for name in ("output", "manifest", "collection", "offline", "workers", "effort", "reviewer_model",
                     "max_api_calls", "max_wall_seconds", "max_output_tokens"):
            setattr(trusted, name, getattr(args, name))
        trusted.donor = args.prefill_source or args.selection.parent
        if args.api_config is not None:
            trusted.api_config = args.api_config
        if args.auth_path is not None:
            trusted.auth_path = args.auth_path
        trusted.timeout = min(args.timeout, 150)
        if args.limit:
            trusted.max_api_calls = min(trusted.max_api_calls, args.limit)
        trusted.output.mkdir(parents=True, exist_ok=True)
        with (trusted.output / ".trusted_evaluation.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = trusted_run(trusted)
        return 0 if report["status"] in {"COMPLETE", "COMPLETE_WITH_BOUNDS"} else 2
    args.api_config = args.api_config or codex_home / "config.toml"
    args.auth_path = args.auth_path or codex_home / "auth.json"
    if not args.offline:
        # Configure only the explicitly selected home credential; transport resolves the key
        # from this config, while this check prevents accidental ambient-key fallback.
        from scripts.evaluate_answered_outcomes import configure_review_credentials
        configure_review_credentials(args.auth_path, args.api_config)
    if args.limit:
        source = read_json(args.selection)
        limited = {**source, "answers": source["answers"][:args.limit]}
        temp_selection = args.output / "selection_limited.json"
        write_json(temp_selection, limited)
        args.selection = temp_selection
    report = run(args)
    print(json.dumps({key: report[key] for key in ("status", "expected", "status_counts", "metrics")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"COMPLETE", "COMPLETE_WITH_BOUNDS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
