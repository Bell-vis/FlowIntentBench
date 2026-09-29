"""Versioned quality thresholds and efficiency views of existing metric rows.

No scientific judgment is made here. Unknown judgments and unavailable usage
remain unknown; low quality and model noncompletion remain in the denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from numbers import Real

QUALITY_POLICY = {
    "version": "complete-scientific-task-v1",
    "required_scores": {
        "o_score": 1.0,
        "finding_precision": 1.0,
        "finding_requirement_recall": 1.0,
        "c_score": 1.0,
    },
    "f2_requires_semantic_adequate_core_complete": True,
    "branch_alignment_is_diagnostic_only": True,
    "model_noncompletion": "FAIL",
    "pending": "UNRESOLVED_NOT_ZERO",
    "infrastructure_invalid": "EXCLUDED_AND_REPORTED",
    "selection": "FROZEN_BEFORE_COLLECTION",
}
QUALITY_POLICY_SHA256 = hashlib.sha256(
    json.dumps(QUALITY_POLICY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
EFFICIENCY_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "wall_clock_time",
    "python_execution_total_time",
    "provider_reported_cost",
)


def _number(value):
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def quality_status(row):
    status = row.get("status", "NOT_STARTED")
    if status == "INFRASTRUCTURE_INVALID":
        return "INFRASTRUCTURE_INVALID"
    if status in {"NOT_STARTED", "NOT_COLLECTED", "RUNNING"}:
        return "NOT_FINISHED"
    if status == "MODEL_NONCOMPLETION":
        return "FAIL"
    if status != "COMPLETED":
        return "PENDING"
    values = [row.get(name) for name in QUALITY_POLICY["required_scores"]]
    if any(_number(v) and not 0 <= v <= 1 for v in values):
        raise ValueError("quality input score outside [0, 1]")
    # An explicitly failed requirement is sufficient for a completed answer
    # to miss the threshold even when C is correctly inapplicable to its O.
    if any(_number(v) and v < 1 for v in values):
        return "FAIL"
    if row.get("condition", "").endswith("F2"):
        if row.get("finding_recall_mode") != "SEMANTIC_ADEQUATE_CORE":
            return "PENDING"
        if row.get("adequate_core_complete") is False:
            return "FAIL"
        if row.get("adequate_core_complete") is not True:
            return "PENDING"
    return "PASS" if all(_number(v) and v == 1 for v in values) else "PENDING"


def _efficiency(rows):
    output = {}
    for name in EFFICIENCY_FIELDS:
        by_case = defaultdict(list)
        observations = []
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            if not _number(value) or value < 0:
                raise ValueError(f"invalid efficiency telemetry: {name}")
            observations.append(float(value))
            by_case[row["case_id"]].append(float(value))
        case_means = [statistics.fmean(values) for values in by_case.values()]
        output[name] = {
            "observation_count": len(observations),
            "missing_observation_count": len(rows) - len(observations),
            "case_count": len(case_means),
            "observation_mean": statistics.fmean(observations)
            if observations
            else None,
            "observation_median": statistics.median(observations)
            if observations
            else None,
            "case_macro_mean": statistics.fmean(case_means) if case_means else None,
        }
    return output


def _group(rows):
    counts = Counter(quality_status(row) for row in rows)
    resolved = counts["PASS"] + counts["FAIL"]
    eligible = resolved + counts["PENDING"] + counts["NOT_FINISHED"]
    complete = counts["PENDING"] == counts["NOT_FINISHED"] == 0
    accepted = [row for row in rows if quality_status(row) == "PASS"]
    model_runs = [
        row
        for row in rows
        if row.get("status") in {"COMPLETED", "MODEL_NONCOMPLETION", "PENDING"}
    ]
    return {
        "observed_row_count": len(rows),
        "quality_status_counts": {
            name: counts[name]
            for name in (
                "PASS",
                "FAIL",
                "PENDING",
                "NOT_FINISHED",
                "INFRASTRUCTURE_INVALID",
            )
        },
        "quality_eligible_n": eligible,
        "quality_resolved_n": resolved,
        "quality_resolution_coverage": resolved / eligible if eligible else None,
        "quality_pass_rate": counts["PASS"] / eligible
        if complete and eligible
        else None,
        "resolved_only_quality_pass_rate_diagnostic": counts["PASS"] / resolved
        if resolved
        else None,
        "quality_pass_rate_lower_bound": counts["PASS"] / eligible
        if eligible
        else None,
        "quality_pass_rate_upper_bound": (
            counts["PASS"] + counts["PENDING"] + counts["NOT_FINISHED"]
        )
        / eligible
        if eligible
        else None,
        "quality_qualified_efficiency": _efficiency(accepted),
        "all_observed_model_run_efficiency": _efficiency(model_runs),
        "infrastructure_efficiency": _efficiency(
            [row for row in rows if row.get("status") == "INFRASTRUCTURE_INVALID"]
        ),
    }


def quality_efficiency_summary(rows):
    rows = list(rows)
    seen = set()
    for row in rows:
        key = (row.get("model"), row.get("case_id"), row.get("trial"))
        if not all(value is not None for value in key):
            raise ValueError("quality rows require model/case_id/trial identity")
        if row.get("status") == "INFRASTRUCTURE_INVALID":
            continue
        if key in seen:
            raise ValueError("duplicate model/case/trial quality observation")
        seen.add(key)
    models = sorted({row["model"] for row in rows})
    return {
        "artifact_type": "QualityQualifiedEfficiencyReport",
        "quality_policy": QUALITY_POLICY,
        "quality_policy_sha256": QUALITY_POLICY_SHA256,
        "scope": "Supplied rows only; the experiment manifest owns requested-slot completeness",
        "by_model": {
            model: {
                "overall": _group([r for r in rows if r["model"] == model]),
                "by_condition": {
                    condition: _group(
                        [
                            r
                            for r in rows
                            if r["model"] == model and r["condition"] == condition
                        ]
                    )
                    for condition in sorted(
                        {r["condition"] for r in rows if r["model"] == model}
                    )
                },
            }
            for model in models
        },
    }
