#!/usr/bin/env python3
"""Freeze and score all completed Claude trials under outcome-rubric-v2."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
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

from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import RunRecord
from flowintentbench.outcome_scoring import (
    OutcomeJudgment, POLICY, PROTOCOL, build_rubric, digest, review_prompt, score_outcome,
)
from flowintentbench.subagent_reporting import _ledger, current_slot_run_paths
from scripts.run_core_case_scoring import _parse_json
from scripts.benchmark_runtime import require_benchmark_runtime
from scripts.portable_fcntl import fcntl


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_review_credentials(auth_path, api_config=None):
    """Pin this process to the selected auth file, never an ambient fallback."""
    from flowintentbench.runtime_config import resolve_provider_configuration
    path = Path(auth_path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    key = document.get("OPENAI_API_KEY") if isinstance(document, dict) else None
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"selected auth file has no nonempty OPENAI_API_KEY: {path}")
    key = key.strip()
    runtime = resolve_provider_configuration(config_path=api_config or ROOT / "config/yiapi.toml")
    os.environ[runtime.api_key_env] = key
    return {"auth_path": str(path), "key_sha256_prefix": hashlib.sha256(key.encode()).hexdigest()[:12],
            "python": sys.executable, "python_prefix": sys.prefix,
            "endpoint": runtime.base_url}


def review_effort_for_row(output, row, default_effort, retry_effort):
    if retry_effort is None:
        return default_effort
    path = output / "cases" / (row["run_id"] + ".json")
    if not path.exists():
        return default_effort
    previous = read_json(path)
    if previous.get("status") == "SCORED":
        effort = previous.get("reasoning_effort")
        if effort not in {default_effort, retry_effort}:
            raise ValueError("existing review effort is outside the selected retry profile")
        return effort
    return retry_effort


def freeze_selection(source, manifest_path, output, expected):
    state, _ = _ledger(source)
    slots = {slot["slot_id"]: slot for slot in state["slots"] if slot["status"] == "COMPLETED"}
    paths = current_slot_run_paths(source)
    runs = {RunRecord.from_dict(read_json(path)).run_id: path for path in paths}
    manifest = read_json(manifest_path)
    cases = {row["case_id"]: row for row in manifest["cases"]}
    rows = []
    for slot in slots.values():
        # The collector's slot identity is the immutable run identity here.
        path = runs.get(slot["slot_id"])
        if path is None:
            raise ValueError("completed ledger slot has no matching current run")
        record = RunRecord.from_dict(read_json(path))
        if (record.run_status.value != "COMPLETED" or not record.final_response
                or record.case_id != slot["case_id"] or record.model_id != slot["model_id"]
                or record.trial_index != slot["trial_index"]):
            raise ValueError("completed slot/run identity mismatch")
        rows.append({"slot_id": slot["slot_id"], "run_id": record.run_id,
                     "model_id": record.model_id, "case_id": record.case_id,
                     "trial": record.trial_index, "condition": cases[record.case_id]["condition"],
                     "run_path": str(path.resolve()), "run_sha256": sha_file(path),
                     "answer_sha256": hashlib.sha256(record.final_response.encode()).hexdigest()})
    rows.sort(key=lambda row: (row["model_id"], row["case_id"], row["trial"], row["run_id"]))
    if len(rows) != expected or len({row["run_id"] for row in rows}) != expected:
        raise ValueError(f"expected {expected} distinct completed answers, got {len(rows)}")
    selection = {"protocol": PROTOCOL, "policy_sha256": digest(POLICY),
                 "collection_sha256": sha_file(source / "collection_state.json"),
                 "manifest_sha256": sha_file(manifest_path), "answers": rows}
    path = output / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise ValueError("frozen selection changed; use a new output directory")
    write_json(path, selection)
    return selection, cases


def write_reports(output, selection, results, *, protocol=PROTOCOL, policy=POLICY,
                  reviewer_model="gpt-6-astra"):
    by_run = {row["run_id"]: row for row in results}
    statuses = Counter(by_run.get(row["run_id"], {}).get("status", "NOT_REVIEWED")
                       for row in selection["answers"])
    report = {"protocol": protocol, "policy": policy,
              "reviewer": {"model": reviewer_model, "reasoning_efforts": sorted({
                  row["reasoning_effort"] for row in results if row.get("reasoning_effort")}), "api_review_tools": []},
              "updated_utc": datetime.now(timezone.utc).isoformat(),
              "selection_sha256": digest(selection), "expected": len(selection["answers"]),
              "status": "COMPLETE" if statuses["SCORED"] == len(selection["answers"]) else "INCOMPLETE",
              "status_counts": dict(statuses), "models": {}, "results": results}
    report["reviewer"]["source_counts"] = dict(Counter(
        row.get("reviewer_source", "HTTP_API") for row in results if row.get("status") == "SCORED"))
    report["reviewer"]["source_metrics"] = {}
    for source in report["reviewer"]["source_counts"]:
        cohort = [row for row in results if row.get("status") == "SCORED"
                  and row.get("reviewer_source", "HTTP_API") == source]
        report["reviewer"]["source_metrics"][source] = {
            "scored": len(cohort),
            "trial_mean_interval": [statistics.mean(row["rubric_score_interval"][i] for row in cohort)
                                    for i in (0, 1)],
            "claim_counts": {key: sum(row["verification"]["ALL"][key] for row in cohort)
                             for key in ("VERIFIED", "REFUTED", "UNVERIFIED", "total")},
            "reasoning_efforts": dict(Counter(row.get("reasoning_effort", "UNKNOWN") for row in cohort)),
            "model_evidence": ("Explicit Codex agent model selection; no HTTP model receipt"
                               if source == "EXPLICIT_GPT6_AGENT" else "Persisted provider receipts"),
        }
    for model in sorted({row["model_id"] for row in selection["answers"]}):
        selected = [row for row in results if row["model_id"] == model and row["status"] == "SCORED"]
        expected = sum(row["model_id"] == model for row in selection["answers"])
        case_groups = defaultdict(list)
        for row in selected:
            case_groups[row["case_id"]].append(row)
        def mean(values):
            values = list(values)
            return statistics.mean(values) if values else None
        report["models"][model] = {
            "scored": len(selected), "expected": expected, "case_count": len(case_groups),
            "trial_mean_interval": [mean(row["rubric_score_interval"][i] for row in selected) for i in (0, 1)],
            "case_macro_interval": [mean(mean(row["rubric_score_interval"][i] for row in group)
                                         for group in case_groups.values()) for i in (0, 1)],
            "rubric_group_intervals": {
                name: [mean(row["metrics"][name][bound] for row in selected) for bound in ("lower", "upper")]
                for name in POLICY["groups"]},
            "audit_counts": dict(Counter(row["audit_status"] for row in selected)),
            "rubric_assessed_items": sum(item["assessed"] for row in selected for item in row["metrics"].values()),
            "rubric_total_items": sum(item["total"] for row in selected for item in row["metrics"].values()),
            "unbound_rubric_items": sum(len(row.get("unbound_rubric_items", [])) for row in selected),
            "answers_with_score_interval": sum(row["rubric_score"] is None for row in selected),
            "claim_counts": {key: sum(row["verification"]["ALL"][key] for row in selected)
                             for key in ("VERIFIED", "REFUTED", "UNVERIFIED", "total")},
        }
    report_dir = output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "experiment_report.json", report)
    columns = ["run_id", "slot_id", "model_id", "case_id", "trial", "condition", "status",
               "rubric_lower", "rubric_upper", "method", "requirements", "consistency",
               "audit_status", "verified", "refuted", "unverified", "verification_coverage"]
    csv_path = report_dir / "trial_metrics.csv"
    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in selection["answers"]:
            result = by_run.get(item["run_id"], {})
            row = {key: item.get(key) for key in columns[:6]}
            row["status"] = result.get("status", "NOT_REVIEWED")
            if result.get("status") == "SCORED":
                row.update(rubric_lower=result["rubric_score_interval"][0],
                           rubric_upper=result["rubric_score_interval"][1],
                           audit_status=result["audit_status"],
                           **{name: result["metrics"][name]["score"] for name in POLICY["groups"]})
                counts = result["verification"]["ALL"]
                row.update(verified=counts["VERIFIED"], refuted=counts["REFUTED"],
                           unverified=counts["UNVERIFIED"], verification_coverage=counts["coverage"])
            writer.writerow(row)
    temp.replace(csv_path)
    lines = ["# Answered outcome evaluation", "", f"Protocol: `{protocol}`. Status: **{report['status']}**.",
             f"SCORED: **{statuses['SCORED']}/{report['expected']}** completed answers.", "",
             "Review sources (scored answers): `" + json.dumps(report["reviewer"]["source_counts"], sort_keys=True) + "`.",
             "Explicit agent reviews retain orchestration and packet provenance; they have no HTTP model receipt and use a separate review profile.", "",
             "SCORED means the frozen outcome rubric has been reviewed. It does not mean all claims are correct or independently reproduced.",
             "Rubric scores measure method adequacy, requested information coverage and internal consistency. Numerical correctness is reported separately.",
             "Unknown rubric items retain lower/upper bounds. Unverified claims are neither credited as true nor counted as false.",
             "Human calibration: NOT_ESTABLISHED. This is a new development protocol, not the original full-audit metric or a complete N=3 benchmark.", "",
             "| Model | Scored | Cases | Trial rubric interval | Case macro interval | Assessed rubric items | Verified / refuted / unknown claims |",
             "|---|---:|---:|---|---|---|---|"]
    for model, block in report["models"].items():
        def interval(key):
            a, b = block[key]
            return "N/A" if a is None else f"[{a:.4f}, {b:.4f}]"
        n = block["claim_counts"]
        lines.append(f"| {model} | {block['scored']}/{block['expected']} | {block['case_count']} | {interval('trial_mean_interval')} | {interval('case_macro_interval')} | {block['rubric_assessed_items']}/{block['rubric_total_items']} | {n['VERIFIED']} / {n['REFUTED']} / {n['UNVERIFIED']} |")
    if len(report["reviewer"]["source_metrics"]) > 1:
        lines += ["", "| Review source | Scored | Trial rubric interval | Verified / refuted / unknown claims |",
                  "|---|---:|---|---|"]
        for source, block in report["reviewer"]["source_metrics"].items():
            low, high = block["trial_mean_interval"]
            counts = block["claim_counts"]
            lines.append(f"| {source} | {block['scored']} | [{low:.4f}, {high:.4f}] | {counts['VERIFIED']} / {counts['REFUTED']} / {counts['UNVERIFIED']} |")
        lines += ["", "Review sources have different execution contexts and question cohorts; these summaries are descriptive, not a controlled reviewer comparison."]
    lines += ["", "Model cohorts contain different cases and repetitions; these unpaired means are descriptive, not a fair model ranking.",
              "Original answers, old evaluations and solver efficiency records are preserved. Reviewer usage belongs to api/*/receipt.json, not solver efficiency.", ""]
    markdown = report_dir / "experiment_report.md"
    tmp = markdown.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(markdown)
    return report


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_resume_eval")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_outcomes_v2_177")
    parser.add_argument("--expected", type=int, default=177)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attempts", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--effort", choices=("medium", "high", "xhigh"), default="medium")
    parser.add_argument("--retry-effort", choices=("medium", "high", "xhigh"),
                        help="use a separately identified effort for failed rows; retain successful review efforts")
    parser.add_argument("--reviewer-model", default="gpt-6-astra",
                        help="review model id; part of the immutable request identity")
    parser.add_argument("--auth-path", type=Path, default=ROOT / "auth.json",
                        help="explicit OPENAI_API_KEY JSON file; no ambient credential fallback")
    parser.add_argument("--api-config", type=Path, default=ROOT / "config/yiapi.toml")
    parser.add_argument("--reuse-reviewed-from", type=Path,
                        help="explicitly reuse validated same-rubric reviews from an earlier API route")
    parser.add_argument("--monitor-seconds", type=int, default=60)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    output, source = args.output.resolve(), args.source.resolve()
    if output == source or source.is_relative_to(output) or output.drive.lower() == "c:":
        parser.error("use a separate output directory on a data drive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.monitor_seconds < 1:
        parser.error("monitor-seconds must be positive")
    output.mkdir(parents=True, exist_ok=True)
    output_lock = (output / ".outcome_runner.lock").open("a+b")
    try:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        output_lock.close()
        raise SystemExit("another outcome evaluator owns this output directory")
    selection, cases = freeze_selection(source, args.manifest, output, args.expected)
    write_json(output / "policy.json", POLICY)
    schema = OutcomeJudgment.model_json_schema()
    # Validate all frozen inputs before any paid call and bind cache reuse to
    # the exact prompt, schema, scorer implementation and source run.
    prepared = []
    from flowintentbench.runtime_config import resolve_provider_configuration
    runtime = resolve_provider_configuration(config_path=args.api_config)
    responses_route = runtime.model_configuration.get("wire_api") == "responses"
    scorer_sha = sha_file(ROOT / "flowintentbench/outcome_scoring.py")
    for row in selection["answers"]:
        actual_effort = review_effort_for_row(output, row, args.effort, args.retry_effort)
        case_input, metadata, gt, material = load_development_case(ROOT, cases[row["case_id"]])
        record = RunRecord.from_dict(read_json(Path(row["run_path"])))
        rubric = build_rubric(metadata, material)
        prompt = review_prompt(case_input, metadata, material, gt, record.final_response, rubric)
        identity_fields = {"prompt": prompt, "schema": schema, "policy": POLICY,
                           "scorer_sha256": scorer_sha, "source": row["run_sha256"],
                           "verification_policy": material["finding_verification_policy"],
                           "reviewer_model": args.reviewer_model, "effort": actual_effort}
        if responses_route:
            identity_fields["api_route"] = {"endpoint": runtime.base_url, "wire_api": "responses"}
        identity = digest(identity_fields)
        prepared.append((row, record.final_response, rubric, gt, material, prompt, identity, actual_effort))
    print(json.dumps({"event": "selection_validated", "answers": len(prepared), "output": str(output)}), flush=True)
    write_json(output / "progress.json", {"status": "RUNNING", "pid": os.getpid(),
               "expected": len(prepared), "completed": 0, "scored": 0, "started_epoch": time.time()})
    monitor_stop = threading.Event()
    def monitor():
        while not monitor_stop.wait(args.monitor_seconds):
            try:
                progress = read_json(output / "progress.json")
                snapshot = {"event": "evaluation_monitor", "utc": datetime.now(timezone.utc).isoformat(),
                            **progress}
                with (output / "monitor.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(snapshot) + "\n")
                print(json.dumps(snapshot), flush=True)
            except (OSError, ValueError) as exc:
                print(json.dumps({"event": "monitor_error", "error": str(exc)}), flush=True)
    threading.Thread(target=monitor, daemon=True).start()
    controller = None
    agent = None
    if not args.offline:
        credential_record = configure_review_credentials(args.auth_path, args.api_config)
        write_json(output / "runtime" / f"credentials-{time.time_ns()}.json", credential_record)
        print(json.dumps({"event": "review_credentials_selected", **credential_record}), flush=True)
        if responses_route:
            from scripts.outcome_responses_transport import ResponsesOutcomeReviewer
            agent = ResponsesOutcomeReviewer(args.api_config, concurrency=args.workers)
        else:
            from scripts.run_claude_benchmark import ClaudeController
            controller = ClaudeController(output, ROOT / "settings.json", None, args.api_config, 0,
                                          api_judge_workers=args.workers, api_shared_concurrency=4)
            controller.agent.structured_only = True
            agent = controller.agent
            if not controller.judge_governor.check_health(startup=True):
                raise SystemExit("reviewer preflight failed; no answers scored")

    def process(item):
        row, answer, rubric, gt, material, prompt, identity, actual_effort = item
        result_path = output / "cases" / (row["run_id"] + ".json")
        judgment_path = output / "judgments" / (identity + ".json")
        request_path = output / "requests" / (identity + ".json")
        write_json(request_path, {"request_id": identity, "prompt": prompt, "schema": schema,
                                  "rubric": rubric, "run_id": row["run_id"], "scorer_sha256": scorer_sha})
        error = None
        previous = None
        receipts = []
        reused_from = None
        if judgment_path.exists():
            cached = read_json(judgment_path)
            if cached.get("request_id") != identity:
                raise ValueError("judgment identity mismatch")
            previous = cached["judgment"]
            receipts = cached.get("reviewer_receipts", [])
            if not receipts:
                receipts = [str(path) for path in sorted((output / "api").glob(identity + "-*/receipt.json"))]
            reused_from = cached.get("reused_from")
        elif args.reuse_reviewed_from:
            donor = args.reuse_reviewed_from.resolve()
            donor_path = donor / "cases" / (row["run_id"] + ".json")
            if donor_path.exists():
                old = read_json(donor_path)
                if old.get("status") == "SCORED":
                    request = read_json(donor / "requests" / (old["request_id"] + ".json"))
                    cached = read_json(donor / "judgments" / (old["request_id"] + ".json"))
                    if (request["prompt"] != prompt or request["schema"] != schema
                            or old["scorer_sha256"] != scorer_sha or old["run_sha256"] != row["run_sha256"]
                            or old["reviewer_model"] != args.reviewer_model or old["reasoning_effort"] != actual_effort
                            or digest(cached["judgment"]) != old["judgment_sha256"]):
                        raise ValueError("donor review does not match frozen scoring inputs")
                    previous = cached["judgment"]
                    receipts = old["reviewer_receipts"]
                    reused_from = {"result_path": str(donor_path), "result_sha256": sha_file(donor_path),
                                   "request_id": old["request_id"], "route": "original receipt"}
        for attempt in range(args.attempts + 1):
            try:
                if previous is None:
                    if args.offline or attempt == args.attempts:
                        break
                    token = f"{identity}-{time.time_ns()}"
                    receipts.append(str(output / "api" / token / "receipt.json"))
                    if error and controller is not None and not controller.judge_governor.check_health(startup=True):
                        raise RuntimeError("reviewer recovery preflight failed")
                    current_prompt = prompt
                    if error:
                        current_prompt += "\nPrevious response failed host validation: " + error + ". Return a corrected complete judgment. Use exact contiguous quotations."
                    receipt = agent(current_prompt, args.reviewer_model, output / "work" / token,
                                               output / "api" / token, timeout_seconds=args.timeout,
                                               output_schema=schema, reasoning_effort=actual_effort)
                    if not receipt.get("completed"):
                        raise RuntimeError(receipt.get("error") or "reviewer did not complete")
                    previous = _parse_json(receipt["final_text"])
                result = score_outcome(answer, rubric, gt, material["finding_verification_policy"], previous)
                write_json(judgment_path, {"request_id": identity, "judgment": previous,
                                          "reviewer_receipts": receipts, "reused_from": reused_from})
                result.update(row, request_id=identity, scorer_sha256=scorer_sha,
                              reviewer_model=args.reviewer_model, reasoning_effort=actual_effort,
                              reviewer_receipts=receipts,
                              reused_from=reused_from,
                              verifier_sha256=sha_file(ROOT / "flowintentbench/evaluator.py"))
                write_json(result_path, result)
                return result
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                previous = None
                if args.offline:
                    break
        result = {**row, "protocol": PROTOCOL, "request_id": identity,
                  "status": "REVIEW_ERROR", "error": error or "missing offline judgment"}
        write_json(result_path, result)
        return result

    work = prepared[:args.limit] if args.limit else prepared
    results = []
    # Reports always retain the full 177 denominator, including pilot runs.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item): item[0] for item in work}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda row: (row["model_id"], row["case_id"], row["trial"]))
            report = write_reports(output, selection, results, reviewer_model=args.reviewer_model)
            write_json(output / "progress.json", {"status": "RUNNING", "pid": os.getpid(),
                       "expected": len(prepared), "completed": len(results),
                       "scored": report["status_counts"].get("SCORED", 0), "updated_epoch": time.time()})
            print(json.dumps({"event": "answer_evaluated", "case": result["case_id"],
                              "trial": result["trial"], "status": result["status"],
                              "error": result.get("error"), "done": len(results),
                              "scored": report["status_counts"].get("SCORED", 0), "expected": len(prepared)}), flush=True)
    complete = len(results) == len(prepared) and all(row["status"] == "SCORED" for row in results)
    write_json(output / "progress.json", {"status": "COMPLETE" if complete else "INCOMPLETE",
               "pid": os.getpid(), "expected": len(prepared), "completed": len(results),
               "scored": sum(row["status"] == "SCORED" for row in results), "finished_epoch": time.time()})
    output_lock.close()
    monitor_stop.set()
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
