"""Planning helpers for selective N=1 completion acquisition.

The completion layer is deliberately downstream of the canonical N=1
experiment.  It only reads rendered metric cells and collection observations;
it never changes Case/GT/scientific metric semantics and never turns a
scientific ``PENDING`` into a score.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


COMPLETE_CELL_STATES = frozenset({"NUMERIC", "NOT_APPLICABLE"})
RECOLLECT_STATES = frozenset({"MODEL_NONCOMPLETION"})
RETRY_STATES = frozenset({"PENDING", "INFRASTRUCTURE_INVALID", "UNEXPLAINED_NULL"})
SCIENTIFIC_METRICS = (
    "o_score", "urs", "resolved_o_compliance", "finding_precision",
    "core_finding_recall", "finding_requirement_recall", "adequate_core_complete",
    "c_score", "branch_alignment",
)


def build_completion_plan(
    metric_rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str] | None = None,
    round_number: int = 1,
    previous_progress: Mapping[str, Mapping[str, Any]] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Classify every expected case for a bounded completion round.

    A case is ``LOCK`` only when every rendered applicable cell is numeric or
    explicitly not applicable.  Missing/model-noncompletion observations are
    recollected; all other incomplete cells are evaluator-retried.  The
    distinction is operational only and leaves scientific pending decisions
    visible in the plan.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            grouped[case_id].append(row)
    # The rendered matrix is expected to contain one row per metric/case.
    # Deriving the metric universe once lets the planner detect a silently
    # truncated case (rather than treating the surviving cells as complete).
    metric_names = {
        str(row.get("metric") or "")
        for row in metric_rows
        if str(row.get("metric") or "")
    }
    if strict:
        metric_names = set(SCIENTIFIC_METRICS)
    expected = [str(value) for value in (expected_case_ids or sorted(grouped))]
    cases: list[dict[str, Any]] = []
    for case_id in expected:
        rows = grouped.get(case_id, [])
        statuses = [str(row.get("status") or "MISSING") for row in rows]
        missing_metrics = sorted(
            str(row.get("metric") or "")
            for row in rows
            if str(row.get("status") or "MISSING") not in COMPLETE_CELL_STATES
        )
        present_metrics = {
            str(row.get("metric") or "") for row in rows if row.get("metric")
        }
        missing_metrics = sorted(set(missing_metrics) | (metric_names - present_metrics))
        invalid_cells = []
        if strict:
            for row in rows:
                name, state = str(row.get("metric") or ""), row.get("status")
                if name not in metric_names or sum(r.get("metric") == name for r in rows) != 1:
                    invalid_cells.append(name)
                if state == "NUMERIC":
                    try:
                        value = float(row.get("value"))
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            invalid_cells.append(name)
                    except (TypeError, ValueError):
                        invalid_cells.append(name)
                elif state == "NOT_APPLICABLE" and (
                    not str(row.get("reason") or "").strip() or row.get("value") not in (None, "")
                ):
                    invalid_cells.append(name)
            missing_metrics = sorted(set(missing_metrics) | set(invalid_cells))
        progress = (previous_progress or {}).get(case_id, {})
        fingerprint = progress.get("fingerprint")
        same_pending = bool(
            fingerprint
            and progress.get("previous_fingerprint")
            and fingerprint == progress.get("previous_fingerprint")
            # Transport failures are retried independently.  Repeated-pending
            # detection applies only to persisted unresolved metric states;
            # it must never turn a provider failure into an implementation
            # review decision.
            and any(status in {"PENDING", "UNEXPLAINED_NULL"} for status in statuses)
        )
        if not rows:
            action = "RECOLLECT"
            reason = "no rendered metric cells for case"
        elif invalid_cells:
            action = "REVIEW"
            reason = "invalid, duplicate, or non-finite metric cells"
        elif not missing_metrics:
            action = "LOCK"
            reason = "all applicable metric cells are finalized or explicitly N/A"
        elif any(status in RECOLLECT_STATES for status in statuses):
            action = "RECOLLECT"
            reason = "model response is non-complete or unavailable"
        elif same_pending:
            action = "REVIEW"
            reason = "IMPLEMENTATION_REVIEW_REQUIRED: unchanged pending fingerprint"
        elif any(status in RETRY_STATES for status in statuses):
            action = "REEVALUATE"
            reason = "case has retryable evaluator/scientific cells"
        else:
            action = "REVIEW"
            reason = "unrecognized metric-cell state; fail closed"
        cases.append(
            {
                "case_id": case_id,
                "action": action,
                "status": "COMPLETE" if action == "LOCK" else "INCOMPLETE",
                "metric_cell_count": len(rows),
                "missing_metrics": missing_metrics,
                "observed_states": dict(sorted({state: statuses.count(state) for state in set(statuses)}.items())),
                "reason": reason,
                "pending_fingerprint": fingerprint,
            }
        )
    complete_count = sum(item["action"] == "LOCK" for item in cases)
    review_count = sum(item["action"] == "REVIEW" for item in cases)
    implementation_review_count = sum(
        item["action"] == "REVIEW"
        and str(item.get("reason", "")).startswith("IMPLEMENTATION_REVIEW_REQUIRED")
        for item in cases
    )
    if implementation_review_count:
        plan_status = "IMPLEMENTATION_REVIEW_REQUIRED"
    elif cases and complete_count == len(cases):
        plan_status = "FULL_METRIC_COMPLETE"
    else:
        plan_status = "INCOMPLETE"
    return {
        "record_type": "N1CompletionPlan",
        "schema_version": "n1-completion-plan-v1",
        "round": int(round_number),
        "expected_case_count": len(expected),
        "complete_case_count": complete_count,
        "incomplete_case_count": len(cases) - complete_count,
        "review_case_count": review_count,
        "implementation_review_case_count": implementation_review_count,
        "status": plan_status,
        "cases": cases,
    }


def completion_case_ids(plan: Mapping[str, Any], action: str) -> list[str]:
    """Return deterministic case IDs for one completion action."""

    return sorted(
        str(item.get("case_id"))
        for item in plan.get("cases", ())
        if isinstance(item, Mapping) and item.get("action") == action
    )
