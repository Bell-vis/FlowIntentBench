"""Auditable efficiency summaries for collected Claude Code solver sessions.

The scientific report owns answer quality. This module narrows efficiency to
successful target-model solver sessions and deliberately keeps reviewer and
host execution accounting out of the primary measurements.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from numbers import Real
from pathlib import Path
import statistics

from flowintentbench.quality_efficiency import quality_status


METRICS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_turn_count",
    "model_api_time_seconds",
)


def _number(value) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_summary(rows: list[dict]) -> dict:
    result = {}
    for name in METRICS:
        observations = []
        by_case = defaultdict(list)
        for row in rows:
            value = row.get(name)
            if not _number(value):
                continue
            value = float(value)
            observations.append(value)
            by_case[row["case_id"]].append(value)
        case_means = [statistics.fmean(values) for values in by_case.values()]
        result[name] = {
            "observation_count": len(observations),
            "missing_observation_count": len(rows) - len(observations),
            "case_count": len(case_means),
            "observation_mean": statistics.fmean(observations) if observations else None,
            "observation_median": statistics.median(observations) if observations else None,
            "case_macro_mean": statistics.fmean(case_means) if case_means else None,
        }
    return result


def _exclude(audit: dict, exclusions: Counter, reason: str) -> None:
    audit.update(included=False, exclusion_reason=reason)
    exclusions[reason] += 1


def build_claude_solver_efficiency(
    report: dict, state: dict, output: Path
) -> tuple[dict, list[dict]]:
    """Return a strict solver-only summary and one audit row per requested slot."""

    output = Path(output)
    slots = {slot["slot_id"]: slot for slot in state.get("slots", [])}
    eligible = []
    audit_rows = []
    exclusions = Counter()

    for row in report.get("trial_rows", []):
        slot_id = row.get("slot_id")
        slot = slots.get(slot_id)
        audit = {
            "slot_id": slot_id,
            "model": row.get("model"),
            "case_id": row.get("case_id"),
            "trial": row.get("trial"),
            "run_status": slot.get("status") if slot else row.get("collection_status"),
            "quality_status": quality_status(row),
            "included": False,
            "exclusion_reason": None,
            **{name: None for name in METRICS},
        }
        audit_rows.append(audit)
        if slot is None:
            _exclude(audit, exclusions, "missing_collection_slot")
            continue
        if slot.get("status") != "COMPLETED":
            _exclude(audit, exclusions, "run_not_completed")
            continue
        run_record = slot.get("run_record_path")
        if not run_record:
            _exclude(audit, exclusions, "missing_run_record")
            continue
        if not (output / run_record).is_file():
            _exclude(audit, exclusions, "unreadable_run_record")
            continue

        receipt_path = output / "claude_runs" / str(slot_id) / "receipt.json"
        try:
            receipt = _read(receipt_path)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            _exclude(audit, exclusions, "missing_or_unreadable_receipt")
            continue
        if receipt.get("transport") != "third_party_claude_code":
            _exclude(audit, exclusions, "non_claude_solver_transport")
            continue
        expected_model = row.get("model")
        if receipt.get("model") != expected_model or receipt.get("observed_models") != [expected_model]:
            _exclude(audit, exclusions, "target_model_identity_mismatch")
            continue
        if (
            receipt.get("completed") is not True
            or receipt.get("error")
            or receipt.get("timed_out") is True
            or receipt.get("returncode") != 0
        ):
            _exclude(audit, exclusions, "solver_or_api_failure")
            continue
        timing = receipt.get("transport_timing") or {}
        if timing.get("transport_affected") is True:
            _exclude(audit, exclusions, "known_transport_affected")
            continue

        usage = receipt.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        duration_ms = receipt.get("duration_api_ms")
        audit.update(
            included=True,
            exclusion_reason=None,
            input_tokens=input_tokens if _number(input_tokens) else None,
            output_tokens=output_tokens if _number(output_tokens) else None,
            total_tokens=(input_tokens + output_tokens)
            if _number(input_tokens) and _number(output_tokens)
            else None,
            model_turn_count=receipt.get("model_turn_count")
            if _number(receipt.get("model_turn_count"))
            else None,
            model_api_time_seconds=duration_ms / 1000.0 if _number(duration_ms) else None,
        )
        eligible.append(audit)

    models = sorted(
        {slot.get("model_id") for slot in state.get("slots", []) if slot.get("model_id")}
    )
    by_model = {}
    for model in models:
        rows = [row for row in eligible if row["model"] == model]
        accepted = [row for row in rows if row["quality_status"] == "PASS"]
        quality_counts = Counter(row["quality_status"] for row in rows)
        by_model[model] = {
            "eligible_session_count": len(rows),
            "quality_status_counts": dict(sorted(quality_counts.items())),
            "quality_qualified_session_count": len(accepted),
            "all_eligible_sessions": _metric_summary(rows),
            "quality_qualified_sessions": _metric_summary(accepted),
        }

    trial_by_slot = {row.get("slot_id"): row for row in report.get("trial_rows", [])}
    audit_by_slot = {row.get("slot_id"): row for row in audit_rows}
    answered_slots = [
        slot for slot in state.get("slots", []) if slot.get("status") == "COMPLETED"
    ]

    def quality_complete(slot: dict) -> bool:
        row = trial_by_slot.get(slot.get("slot_id"), {})
        return (
            row.get("collection_status") == "COMPLETED"
            and row.get("evaluation_status") == "SCORED"
            and row.get("status") == "COMPLETED"
        )

    def primary_efficiency_available(slot: dict) -> bool:
        row = audit_by_slot.get(slot.get("slot_id"), {})
        return bool(
            row.get("included")
            and _number(row.get("total_tokens"))
            and _number(row.get("model_api_time_seconds"))
        )

    answered_by_model = {}
    for model in models:
        model_slots = [slot for slot in answered_slots if slot.get("model_id") == model]
        model_audited = [
            slot for slot in model_slots if slot.get("slot_id") in audit_by_slot
        ]
        answered_by_model[model] = {
            "answered_slot_count": len(model_slots),
            "quality_complete_count": sum(quality_complete(slot) for slot in model_slots),
            "efficiency_audited_count": len(model_audited),
            "primary_efficiency_available_count": sum(
                primary_efficiency_available(slot) for slot in model_slots
            ),
        }
    quality_complete_count = sum(quality_complete(slot) for slot in answered_slots)
    efficiency_audited_count = sum(
        slot.get("slot_id") in audit_by_slot for slot in answered_slots
    )
    primary_available_count = sum(
        primary_efficiency_available(slot) for slot in answered_slots
    )
    answered_count = len(answered_slots)
    answered_evaluation = {
        "status": "COMPLETE"
        if answered_count
        and quality_complete_count == answered_count
        and efficiency_audited_count == answered_count
        else "INCOMPLETE",
        "scope": "Collected slots with collection status COMPLETED; uncollected and model-noncompletion slots are outside this existing-answer completion gate",
        "answered_slot_count": answered_count,
        "answered_case_count": len({slot.get("case_id") for slot in answered_slots}),
        "quality_complete_count": quality_complete_count,
        "quality_coverage": quality_complete_count / answered_count if answered_count else None,
        "efficiency_audited_count": efficiency_audited_count,
        "efficiency_audit_coverage": efficiency_audited_count / answered_count if answered_count else None,
        "primary_efficiency_available_count": primary_available_count,
        "primary_efficiency_coverage": primary_available_count / answered_count if answered_count else None,
        "by_model": answered_by_model,
    }

    summary = {
        "artifact_type": "ClaudeSolverEfficiencyReport",
        "policy_version": "target-claude-success-no-observed-fault-v1",
        "requested_slot_count": len(state.get("slots", [])),
        "eligible_session_count": len(eligible),
        "excluded_session_count": len(audit_rows) - len(eligible),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "primary_metrics": list(METRICS),
        "policy": {
            "included": "Completed target Claude solver sessions with exact model identity and no observed CLI, API, timeout, or transport failure",
            "excluded": "GPT-6 reviewer calls, preflight calls, host replay/scoring, local Python/tool execution time, unfinished runs, infrastructure failures, and known transport-affected runs",
            "token_scope": "Provider-reported target Claude session tokens only; cached input is included once in input_tokens",
            "time_scope": "Claude Code duration_api_ms only; local code/tool execution and evaluator time are not included",
            "quality_gate": "quality_qualified_sessions contains only trials that pass the frozen quality policy",
            "unobservable_limit": "Provider-internal queueing cannot be separated from duration_api_ms. CLI retries are disabled, but hidden provider-internal activity cannot be audited retroactively",
            "cross_model_pooling": False,
        },
        "answered_evaluation": answered_evaluation,
        "by_model": by_model,
    }
    return summary, audit_rows


def write_solver_efficiency_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "slot_id",
        "model",
        "case_id",
        "trial",
        "run_status",
        "quality_status",
        "included",
        "exclusion_reason",
        *METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
