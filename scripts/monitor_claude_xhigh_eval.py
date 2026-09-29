#!/usr/bin/env python3
"""Continuously monitor the historical Fable/Sonnet xhigh evaluation.

The monitor is read-only. It does not submit requests and does not modify the
evaluation exchange or the published metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_runtime import require_benchmark_runtime

if __name__ == "__main__":
    require_benchmark_runtime()

try:
    from scripts.portable_fcntl import process_exists
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from portable_fcntl import process_exists


MODELS = ("claude-fable-5-1", "claude-sonnet-5")
MODEL_LABELS = {
    "claude-fable-5-1": "Fable 5.1",
    "claude-sonnet-5": "Sonnet 5",
}
METRICS = (
    "o_score",
    "urs",
    "finding_precision",
    "finding_requirement_recall",
    "c_score",
    "branch_alignment",
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/claude_resume_eval"))
    parser.add_argument(
        "--interval", type=float, default=60.0, help="seconds between snapshots (default: 60)"
    )
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    return parser.parse_args()


def read_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def rows_from(path: Path) -> list[dict]:
    try:
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    except (FileNotFoundError, OSError):
        return []


def valid(row: dict) -> bool:
    # O1 URS and inapplicable C may legitimately be blank.
    return (
        row.get("collection_status") == "COMPLETED"
        and row.get("evaluation_status") == "SCORED"
        and row.get("status") == "COMPLETED"
    )


def complete_n3(rows: list[dict]) -> int:
    cases = defaultdict(list)
    for row in rows:
        cases[row.get("case_id")].append(row)
    return sum(
        len(group) == 3
        and len({row.get("trial") for row in group}) == 3
        and all(valid(row) for row in group)
        for group in cases.values()
    )


def shown(value: object, digits: int = 1) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def evaluator_pids(output: Path) -> list[str]:
    marker = read_json(output / "console_process.json", {})
    if marker.get("status") == "RUNNING":
        try:
            if process_exists(int(marker.get("pid", 0))):
                return [f"pid={marker['pid']} (console_process.json)"]
        except (TypeError, ValueError):
            pass
    try:
        result = subprocess.run(
            ["pgrep", "-af", "[e]valuate_existing_claude_xhigh.py"],
            text=True,
            capture_output=True,
            check=False,
        )
        return [
            line
            for line in result.stdout.splitlines()
            if "monitor_claude_xhigh_eval.py" not in line
        ]
    except OSError:
        return []


def snapshot(output: Path, seen: set[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    for path in (output / "rubric_metrics_v4/reports/experiment_report.json",
                 output / "reports/experiment_report.json", output / "trusted_metrics/reports/experiment_report.json"):
        trusted = read_json(path, {})
        if trusted.get("protocol", "").startswith(("rubric-evidence-", "trusted-original-metrics")):
            print(f"\n[{trusted['updated_utc']}] trusted report={path}")
            print(f"answers={trusted['answered_slots']}/{trusted['requested_slots']}; "
                  f"metric completion={trusted['status']}; review disposition={trusted['answered_review_status']}")
            print("Intervals preserve unresolved evidence; no N3 closure filter.")
            for model, groups in trusted["models"].items():
                print(model)
                for name, metric in groups["answered_only"].items():
                    if metric["case_macro_lower"] is not None:
                        print(f"  {name}: [{metric['case_macro_lower']:.4f}, {metric['case_macro_upper']:.4f}] "
                              f"resolved={metric['resolved_trials']}/{metric['applicable_trials']}")
            progress = read_json(path.parent.parent / "review_progress.json", {})
            print("latest published invocation cost:", trusted["evaluation_cost"])
            if progress.get("updated_utc", "") > trusted["updated_utc"]:
                print("current review progress:", progress)
            print("local cumulative new review cost:", trusted.get("cumulative_new_review_cost"))
            return seen
    rows = rows_from(output / "reports/trial_metrics.csv")
    report = read_json(output / "reports/experiment_report.json", {})
    state = read_json(output / "collection_state.json", {})
    progress = read_json(output / "judgment_progress.json", {})
    health = read_json(output / "api_health.json", {})
    invocation = read_json(output / "evaluation_invocation.json", {})
    budget = read_json(output / "review_budget.json", {})
    marker = read_json(output / "console_process.json", {})
    started = marker.get("started_epoch", 0)
    stale_progress = bool(progress and progress.get("updated_epoch", 0) < started)
    if stale_progress:
        progress = {}
    if invocation.get("updated_epoch", 0) < started:
        invocation = {}
    budget_epoch = max(budget.get(key, 0) or 0 for key in
                       ("updated_epoch", "started_epoch", "finished_epoch"))
    if budget_epoch < started:
        budget = {}
    preparation = read_json(output / "preparation_progress.json", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n[{now}] output={output}")
    print(
        "evaluator: "
        + ("running" if evaluator_pids(output) else "not running")
        + " | API catalog preflight: "
        + ("ready" if health.get("health_ready") else str(health.get("failure_reason") or "unknown"))
    )
    if stale_progress:
        print("Current invocation has not published reviewer progress; previous invocation counts are hidden.")
    if preparation.get('pid') == marker.get('pid') and preparation.get('updated_epoch', 0) >= started:
        if preparation.get('status') == 'RUNNING' and marker.get('status') == 'RUNNING':
            preparation['elapsed_seconds'] = preparation.get('elapsed_seconds', 0) + max(
                0, time.time() - preparation['updated_epoch'])
        if preparation.get('phase') == 'REPLAYING_ANSWERS':
            replay = read_json(output / 'evaluation/replay_case_progress.json', {})
            if (replay.get('parent_pid') == marker.get('pid')
                    and replay.get('updated_epoch', 0) >= preparation['updated_epoch']):
                for key in ('completed', 'total', 'phase_eta_seconds'):
                    preparation[key] = replay.get(key)
        print(
            f"preparation: {preparation.get('phase')} ({preparation.get('status')}); "
            f"completed={preparation.get('completed')}/{preparation.get('total')}; "
            f"elapsed={shown(preparation.get('elapsed_seconds'))}s; "
            f"phase ETA={shown(preparation.get('phase_eta_seconds'))}s"
        )
    report_path = output / "reports/experiment_report.json"
    if report_path.exists() and report_path.stat().st_mtime < started:
        print("Scores shown below are from the previous report; this invocation has not published a new report yet.")
    print(
        "requests: attempted={attempted} resolved={resolved} held_pending={held} errors={errors} "
        "running={running} dispatches={dispatches}".format(
            attempted=progress.get("attempted", 0),
            resolved=progress.get("resolved", 0),
            held=progress.get("pending", 0),
            errors=progress.get("errors", 0),
            running=progress.get("running", 0),
            dispatches=progress.get("dispatches", 0),
        )
    )
    if invocation or budget:
        print(
            "invocation: historical={historical}; newly_scored={new}; API_calls={calls}; "
            "budget={status}({reason})".format(
                historical=invocation.get("historical_scientifically_scored_at_start", "N/A"),
                new=invocation.get("newly_scored_slots", "N/A"),
                calls=budget.get("api_calls", "N/A"),
                status=budget.get("status", "N/A"),
                reason=budget.get("reason", ""),
            )
        )
    updated = progress.get("updated_epoch")
    guard = read_json(output / 'score_progress_guard.json', {})
    if guard.get('updated_epoch', 0) >= started and guard:
        print('score guard: {status}; no complete-answer progress={elapsed}s/{limit}s'.format(
            status=guard.get('status'), elapsed=shown(guard.get('seconds_without_score_progress')),
            limit=shown(guard.get('timeout_seconds'))))
    if updated:
        print(
            f"progress age: {max(0, time.time() - updated):.0f}s; "
            "request counts are per invocation and reset on restart."
        )
    print("model             answered/total   SCORED   complete trials   complete N3 cases   PENDING")
    print("-" * 88)
    for model in MODELS:
        group = [row for row in rows if row.get("model") == model]
        answered = sum(row.get("collection_status") == "COMPLETED" for row in group)
        scored = sum(row.get("collection_status") == "COMPLETED"
                     and row.get("evaluation_status") == "SCORED" for row in group)
        complete = sum(valid(row) for row in group)
        pending = sum(row.get("evaluation_status") == "PENDING" for row in group)
        print(
            f"{MODEL_LABELS[model]:<17} {answered:>3}/{len(group):<6} {scored:>7} "
            f"{complete:>17} {complete_n3(group):>19} {pending:>9}"
        )
        stages = Counter(
            row.get("pending_type")
            for row in group
            if row.get("evaluation_status") == "PENDING"
        )
        print("  blockers:", dict(stages))
        noncompletion = sum(row.get("collection_status") == "MODEL_NONCOMPLETION" for row in group)
        if noncompletion:
            print(f"  model noncompletion: {noncompletion} (excluded from answered SCORED)")

    # Distinguish a judge-complete evaluation record from the stricter
    # scientific terminal gate.  A judge can resolve its packet while the
    # host still needs another dependency before the trial is SCORED.
    judge_complete = sum(
        row.get("collection_status") == "COMPLETED"
        and row.get("evaluation_status") == "SCORED"
        for row in rows
    )
    scientific_terminal = report.get("scientifically_terminal_slot_count", 0)
    print(
        f"score gate: judge-complete={judge_complete}; "
        f"scientific-terminal={scientific_terminal}; "
        "SCORED requires both the complete host record and terminal applicable metrics."
    )

    inventory_path = output / "evaluation/pending_requests.json"
    inventory = read_json(inventory_path, {})
    if inventory_path.exists() and inventory_path.stat().st_mtime < started:
        print('Active request inventory is from an earlier replay; preparation has not published a new inventory yet.')
    held = 0
    unanswered = 0
    for group in inventory.get("by_operation", {}).values():
        for item in group:
            response = read_json(
                output / "exchange/responses" / (item["request_id"] + ".json"), {}
            )
            held += response.get("status") == "PENDING"
            unanswered += not response
    print(
        f"active requests: unanswered={unanswered}; held PENDING={held} "
        "(held requests need evidence/evaluator repair)."
    )
    print(
        "O1 URS may be inapplicable. Trial completion and complete N=3 cases are counted separately."
    )

    quality = report.get("quality_efficiency", {}).get("by_model", {})
    print("quality: model             PASS  FAIL  unresolved  coverage   pass rate / bounds")
    for model in MODELS:
        overall = quality.get(model, {}).get("overall", {})
        counts = overall.get("quality_status_counts", {})
        unresolved = counts.get("PENDING", 0) + counts.get("NOT_FINISHED", 0)
        rate = overall.get("quality_pass_rate")
        lower = overall.get("quality_pass_rate_lower_bound")
        upper = overall.get("quality_pass_rate_upper_bound")
        bounds = (
            "N/A"
            if lower is None or upper is None
            else f"[{float(lower):.3f}, {float(upper):.3f}]"
        )
        print(
            f"  {MODEL_LABELS[model]:<17} {counts.get('PASS', 0):>4} "
            f"{counts.get('FAIL', 0):>5} {unresolved:>11} "
            f"{shown(overall.get('quality_resolution_coverage'), 3):>9} "
            f"{shown(rate, 3):>9} / {bounds}"
        )

    efficiency = report.get("solver_efficiency")
    if not efficiency and state and report.get("trial_rows"):
        try:
            from flowintentbench.claude_efficiency import build_claude_solver_efficiency

            efficiency, _ = build_claude_solver_efficiency(report, state, output)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            efficiency = None
    print("solver efficiency (reviewer/API failures/local code excluded):")
    if efficiency:
        answered = report.get("answered_evaluation") or efficiency.get("answered_evaluation", {})
        print(
            "  answered evaluation: status={status}; quality={quality}/{total}; "
            "efficiency audited={audited}/{total}; primary metrics={primary}/{total}".format(
                status=answered.get("status", "N/A"),
                quality=answered.get("quality_complete_count", 0),
                audited=answered.get("efficiency_audited_count", 0),
                primary=answered.get("primary_efficiency_available_count", 0),
                total=answered.get("answered_slot_count", 0),
            )
        )
        for model in MODELS:
            group = efficiency.get("by_model", {}).get(model, {})
            all_metrics = group.get("all_eligible_sessions", {})
            qualified = group.get("quality_qualified_sessions", {})
            tokens = all_metrics.get("total_tokens", {}).get("observation_mean")
            api_time = all_metrics.get("model_api_time_seconds", {}).get("observation_mean")
            qualified_tokens = qualified.get("total_tokens", {}).get("observation_mean")
            print(
                f"  {MODEL_LABELS[model]:<17} "
                f"eligible={group.get('eligible_session_count', 0):>3} "
                f"mean_tokens={shown(tokens)} mean_API_s={shown(api_time)} "
                f"quality_PASS_n={group.get('quality_qualified_session_count', 0)} "
                f"PASS_mean_tokens={shown(qualified_tokens)}"
            )
        print("  exclusions:", efficiency.get("exclusion_counts", {}))
        print("  API duration includes unobservable upstream queueing; it is not pure inference time.")
    else:
        print("  unavailable until a Claude-specific report snapshot is published.")

    current = set()
    for row in rows:
        if valid(row):
            key = (row.get("model", ""), row.get("case_id", ""), row.get("trial", ""))
            current.add(key)
            if key not in seen:
                print(
                    "new complete metrics: {} case={} trial={} O={} URS={} F_precision={} "
                    "F_recall={} C={} branch={}".format(
                        MODEL_LABELS.get(key[0], key[0]),
                        key[1],
                        key[2],
                        *(row.get(metric) or "N/A" for metric in METRICS),
                    )
                )
    if not current:
        print("No successful answered trial has complete applicable metric evaluation.")
    return current


def main() -> int:
    parsed = args()
    output = parsed.output.resolve()
    seen: set[tuple[str, str, str]] = set()
    while True:
        snapshot_started = time.monotonic()
        seen = snapshot(output, seen)
        if parsed.once:
            return 0
        try:
            time.sleep(max(1.0, parsed.interval - (time.monotonic() - snapshot_started)))
        except KeyboardInterrupt:
            print("\nMonitor stopped; evaluator state is unchanged.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
