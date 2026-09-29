"""Render CSV/Markdown views from existing evaluation artifacts.

This script is deliberately read-only with respect to scientific evaluation:
it never calls an evaluator component and never recomputes a metric.  PILOT
rows come directly from finalized CaseEvaluationRecord artifacts.  FORMAL
case, condition, and efficiency rows come from the existing summary and
CaseAggregate artifacts.  The denominator view only counts those stored
values and their frozen condition-level applicability; it does not introduce
or recompute a scientific judgment.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluation_records import iter_current_evaluation_records
from flowintentbench.quality_efficiency import quality_efficiency_summary


TRIAL_COLUMNS = (
    "model",
    "case_id",
    "condition",
    "trial",
    "status",
    "o_score",
    "urs",
    "finding_precision",
    "core_finding_recall",
    "resolved_o_compliance",
    "finding_requirement_recall",
    "finding_recall_mode",
    "adequate_core_complete",
    "c_score",
    "branch_alignment",
    "operationalization_determinate",
    "consistency_unavailable_reason",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "python_execution_total_time",
    "wall_clock_time",
    "provider_reported_cost",
)

CASE_COLUMNS = (
    "model",
    "case_id",
    "condition",
    "o_score",
    "urs",
    "finding_precision",
    "core_finding_recall",
    "resolved_o_compliance",
    "finding_requirement_recall",
    "finding_recall_mode",
    "adequate_core_complete",
    "c_score",
    "branch_alignment",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "python_execution_total_time",
    "wall_clock_time",
    "provider_reported_cost",
)

CONDITION_COLUMNS = (
    "model",
    "condition",
    "eligible_case_count",
    "o_score",
    "urs",
    "finding_precision",
    "core_finding_recall",
    "resolved_o_compliance",
    "finding_requirement_recall",
    "finding_recall_mode",
    "adequate_core_complete_rate",
    "c_score",
    "branch_alignment_rate",
)

EFFICIENCY_COLUMNS = (
    "model",
    "condition/overall",
    "metric",
    "mean",
    "median",
    "case_count",
)

PILOT_EFFICIENCY_METRICS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "python_execution_total_time",
    "wall_clock_time",
    "provider_reported_cost",
)

RELIABILITY_METRICS = (
    "infrastructure_retry_count",
    "infrastructure_failed_request_time",
    "retry_backoff_time",
    "raw_wall_clock_time",
    "clean_wall_clock_time",
)

# This is a coverage view, not another scientific metric.  It makes the
# distinction between an in-scope observation, a finalized value, and an
# unresolved/null value explicit for every frozen quality field.
DENOMINATOR_COLUMNS = (
    "model",
    "condition/overall",
    "metric",
    "mean",
    "applicable_n",
    "finalized_n",
    "null_n",
)

# Per-trial coverage view.  This is deliberately a reporting artifact rather
# than a new scientific metric: every cell is classified from the value and
# status already persisted by the evaluator.
METRIC_MATRIX_COLUMNS = (
    "model",
    "case_id",
    "condition",
    "trial",
    "metric",
    "value",
    "status",
    "reason",
)

METRIC_COVERAGE_COLUMNS = (
    "model",
    "condition/overall",
    "metric",
    "requested_n",
    "applicable_n",
    "numeric_n",
    "not_applicable_n",
    "pending_n",
    "infrastructure_n",
    "model_noncompletion_n",
    "unexplained_null_n",
)

DENOMINATOR_METRICS = (
    "o_score",
    "urs",
    "resolved_o_compliance",
    "finding_precision",
    "core_finding_recall",
    "finding_requirement_recall",
    "adequate_core_complete",
    "c_score",
    "branch_alignment",
)

_METRIC_MEAN_KEYS = {
    "o_score": "mean_o_score",
    "urs": "mean_urs",
    "resolved_o_compliance": "mean_resolved_o_compliance",
    "finding_precision": "mean_finding_precision",
    "core_finding_recall": "mean_core_finding_recall",
    "finding_requirement_recall": "mean_finding_requirement_recall",
    "adequate_core_complete": "adequate_core_complete_rate",
    "c_score": "mean_c_score",
    "branch_alignment": "branch_alignment_rate",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return value


def _metric_row(
    model: str,
    *,
    case_id: str,
    condition: str,
    trial: Any,
) -> dict[str, Any]:
    result = trial.get("result")
    pending = bool(trial.get("pending", False))
    explicit_na_reasons: dict[str, str] = {}
    if pending:
        status = "PENDING"
        scientifically_eligible = False
        metrics: Mapping[str, Any] = {}
        efficiency: Mapping[str, Any] = trial.get("efficiency") or {}
    elif isinstance(result, Mapping):
        status = result.get("run_status")
        scientifically_eligible = bool(
            result.get("eligible_for_scientific_aggregation", False)
        )
        metrics = result.get("metrics") or {}
        # Status records (for example INFRASTRUCTURE_INVALID) intentionally
        # keep metrics=null, but still retain the run telemetry at the record
        # level.  Prefer the metric payload for normal scored records and
        # fall back to the persisted trial-level observation.
        efficiency = {**(trial.get("efficiency") or {}), **(metrics.get("efficiency") or {})}
        # The frozen C/branch definitions return null when there is no
        # admissible O branch (or no applicable Finding set).  Preserve that
        # explicit reason in the report projection instead of misclassifying
        # it as an unexplained missing value.
        consistency = metrics.get("o_f_consistency") or {}
        compatible_branches = result.get("compatible_o_branches")
        f_app_ids = result.get("f_app_prediction_ids")
        if consistency.get("operationalization_determinate") is False:
            reason = consistency.get("unavailable_reason") or "INDETERMINATE_OPERATIONALIZATION"
            explicit_na_reasons["c_score"] = reason
            explicit_na_reasons["branch_alignment"] = reason
        elif consistency.get("unavailable_reason") == "NO_EXECUTABLE_O_EVIDENCE":
            # Explicit O with an execution coverage gap remains applicable;
            # it must not be conflated with an absent method.
            pass
        elif isinstance(compatible_branches, Sequence) and not isinstance(
            compatible_branches, (str, bytes)
        ) and not compatible_branches:
            reason = "No admissible operationalization branch exists for this response."
            explicit_na_reasons["c_score"] = reason
            explicit_na_reasons["branch_alignment"] = reason
        elif isinstance(f_app_ids, Sequence) and not isinstance(f_app_ids, (str, bytes)) and not f_app_ids:
            reason = "The response reported no admissible Finding in F_app."
            explicit_na_reasons["c_score"] = reason
            explicit_na_reasons["branch_alignment"] = reason
    else:
        raise ValueError("trial evaluation must contain result or pending=true")
    scientific = {
        "o_score": None,
        "urs": None,
        "resolved_o_compliance": None,
        "finding_precision": None,
        "core_finding_recall": None,
        "finding_requirement_recall": None,
        "finding_recall_mode": None,
        "adequate_core_complete": None,
        "c_score": None,
        "branch_alignment": None,
    }
    if isinstance(metrics, Mapping):
        operationalization = metrics.get("scientific_operationalization") or {}
        findings = metrics.get("scientific_findings") or {}
        consistency = metrics.get("o_f_consistency") or {}
        # Older records predate the F1/F2-aware public fields.  The condition
        # is already frozen in the record, so using it only to supply a
        # compatibility default for the mode does not infer a score.
        mode = findings.get("finding_recall_mode")
        if mode is None:
            mode = (
                "SEMANTIC_ADEQUATE_CORE"
                if condition.endswith("-F2")
                else "FIXED_REFERENCE_CORE"
            )
        requirement_recall = findings.get("finding_requirement_recall")
        if requirement_recall is None and mode == "FIXED_REFERENCE_CORE":
            requirement_recall = findings.get("core_finding_recall")
        scientific.update(
            {
                "o_score": operationalization.get("o_score"),
                "urs": operationalization.get("urs"),
                "resolved_o_compliance": operationalization.get(
                    "resolved_o_compliance"
                ),
                "finding_precision": findings.get("finding_precision"),
                "core_finding_recall": (
                    findings.get("core_finding_recall")
                    if mode == "FIXED_REFERENCE_CORE"
                    else None
                ),
                "finding_requirement_recall": requirement_recall,
                "finding_recall_mode": mode,
                "adequate_core_complete": findings.get("adequate_core_complete"),
                "c_score": consistency.get("c_score"),
                "branch_alignment": consistency.get("branch_alignment"),
                "operationalization_determinate": consistency.get("operationalization_determinate"),
                "consistency_unavailable_reason": consistency.get("unavailable_reason"),
            }
        )
    total_tokens = efficiency.get("total_tokens")
    if total_tokens is None:
        input_tokens = efficiency.get("input_tokens")
        output_tokens = efficiency.get("output_tokens")
        if isinstance(input_tokens, (int, float)) and isinstance(
            output_tokens, (int, float)
        ):
            total_tokens = input_tokens + output_tokens
    tool_call_count = efficiency.get("tool_call_count")
    return {
        "model": model,
        "case_id": case_id,
        "condition": condition,
        "trial": trial.get("trial_index"),
        "status": status,
        "_scientifically_eligible": scientifically_eligible,
        **scientific,
        "input_tokens": efficiency.get("input_tokens"),
        "output_tokens": efficiency.get("output_tokens"),
        "total_tokens": total_tokens,
        "model_turn_count": efficiency.get("model_turn_count"),
        "tool_call_count": tool_call_count,
        "python_execution_count": efficiency.get("python_execution_count"),
        "python_execution_total_time": efficiency.get("python_execution_total_time"),
        "wall_clock_time": efficiency.get("wall_clock_time"),
        "provider_reported_cost": efficiency.get("provider_reported_cost"),
        # Private report-only provenance used by the metric matrix.  These
        # fields are intentionally excluded from CSV trial columns and do not
        # alter evaluator records or scientific scoring.
        "_pending_type": trial.get("pending_type"),
        "_pending_reason": trial.get("pending_reason"),
        "_pending_category": trial.get("pending_category"),
        "_explicit_na_reasons": explicit_na_reasons,
    }


def _case_metric_row(model: str, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = aggregate["metrics"]
    efficiency = metrics.get("efficiency") or {}
    condition = str(aggregate["condition"])
    mode = metrics.get("finding_recall_mode")
    if mode is None:
        mode = (
            "SEMANTIC_ADEQUATE_CORE"
            if condition.endswith("-F2")
            else "FIXED_REFERENCE_CORE"
        )
    requirement_recall = metrics.get("finding_requirement_recall")
    if requirement_recall is None and mode == "FIXED_REFERENCE_CORE":
        requirement_recall = metrics.get("core_finding_recall")
    total_tokens = efficiency.get("total_tokens")
    if total_tokens is None:
        input_tokens = efficiency.get("input_tokens")
        output_tokens = efficiency.get("output_tokens")
        if isinstance(input_tokens, (int, float)) and isinstance(
            output_tokens, (int, float)
        ):
            total_tokens = input_tokens + output_tokens
    tool_call_count = efficiency.get("tool_call_count")
    return {
        "model": model,
        "case_id": aggregate["case_id"],
        "condition": condition,
        "o_score": metrics.get("o_score"),
        "urs": metrics.get("urs"),
        "resolved_o_compliance": metrics.get("resolved_o_compliance"),
        "finding_precision": metrics.get("finding_precision"),
        "core_finding_recall": (
            metrics.get("core_finding_recall")
            if mode == "FIXED_REFERENCE_CORE"
            else None
        ),
        "finding_requirement_recall": requirement_recall,
        "finding_recall_mode": mode,
        "adequate_core_complete": metrics.get("adequate_core_complete"),
        "c_score": metrics.get("c_score"),
        "branch_alignment": metrics.get("branch_alignment"),
        "input_tokens": efficiency.get("input_tokens"),
        "output_tokens": efficiency.get("output_tokens"),
        "total_tokens": total_tokens,
        "model_turn_count": efficiency.get("model_turn_count"),
        "tool_call_count": tool_call_count,
        "python_execution_count": efficiency.get("python_execution_count"),
        "python_execution_total_time": efficiency.get("python_execution_total_time"),
        "wall_clock_time": efficiency.get("wall_clock_time"),
        "provider_reported_cost": efficiency.get("provider_reported_cost"),
    }


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _artifact_trial_rows(evaluation_root: Path, model: str) -> list[dict[str, Any]]:
    finals: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    pending_candidates: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in iter_current_evaluation_records(evaluation_root):
        payload = _read_json(path)
        run_id = payload.get("run_id") or str(path.parent)
        if path.name == "case_evaluation_record.json":
            finals[str(run_id)] = (path, payload)
        elif path.name == "pending_case_evaluation.json" and str(run_id) not in finals:
            pending_candidates[str(run_id)] = (path, payload)

    rows: list[dict[str, Any]] = []
    for run_id, (_path, payload) in sorted(finals.items()):
        result = payload.get("result")
        if not isinstance(result, Mapping):
            raise ValueError("final evaluation record is missing result")
        rows.append(
            _metric_row(
                model,
                case_id=str(result["case_id"]),
                condition=str(result["condition"]),
                trial={
                    "trial_index": payload.get("trial_index"),
                    "pending": False,
                    "result": result,
                    "efficiency": payload.get("efficiency"),
                },
            )
        )
    # A finalized record is authoritative for its run even if a stale pending
    # file later receives a newer filesystem timestamp.
    for _run_id, (_path, payload) in sorted(pending_candidates.items()):
        rows.append(
            _metric_row(
                model,
                case_id=str(payload["case_id"]),
                condition=str(payload["condition"]),
                trial={
                    "trial_index": payload.get("trial_index"),
                    "pending": True,
                    "efficiency": payload.get("efficiency"),
                    "pending_type": payload.get("pending_type"),
                    "pending_reason": payload.get("pending_reason"),
                    "pending_category": payload.get("pending_category"),
                },
            )
        )
    rows.sort(key=lambda row: (row["case_id"], row["trial"] or 0, row["status"]))
    return rows


def _formal_trial_rows(summary: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for aggregate in summary.get("case_aggregates", ()):
        for trial in aggregate.get("trial_evaluations", ()):
            rows.append(
                _metric_row(
                    model,
                    case_id=str(aggregate["case_id"]),
                    condition=str(aggregate["condition"]),
                    trial=trial,
                )
            )
    for pending in summary.get("pending_records", ()):
        rows.append(
            _metric_row(
                model,
                case_id=str(pending["case_id"]),
                condition=str(pending["condition"]),
                trial={
                    "trial_index": pending.get("trial_index"),
                    "pending": True,
                    "efficiency": pending.get("efficiency"),
                    "pending_type": pending.get("pending_type"),
                    "pending_reason": pending.get("pending_reason"),
                    "pending_category": pending.get("pending_category"),
                },
            )
        )
    rows.sort(key=lambda row: (row["case_id"], row["trial"] or 0, row["status"]))
    return rows


def _condition_rows(summary: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    rows = []
    for condition, values in sorted((summary.get("summary") or {}).get("by_condition", {}).items()):
        mode = (
            "SEMANTIC_ADEQUATE_CORE"
            if str(condition).endswith("-F2")
            else "FIXED_REFERENCE_CORE"
        )
        requirement_recall = values.get("mean_finding_requirement_recall")
        if requirement_recall is None and mode == "FIXED_REFERENCE_CORE":
            requirement_recall = values.get("mean_core_finding_recall")
        rows.append(
            {
                "model": model,
                "condition": condition,
                "eligible_case_count": values.get("eligible_case_count"),
                "o_score": values.get("mean_o_score"),
                "urs": values.get("mean_urs"),
                "resolved_o_compliance": values.get("mean_resolved_o_compliance"),
                "finding_precision": values.get("mean_finding_precision"),
                "core_finding_recall": (
                    values.get("mean_core_finding_recall")
                    if mode == "FIXED_REFERENCE_CORE"
                    else None
                ),
                "finding_requirement_recall": requirement_recall,
                "finding_recall_mode": mode,
                "adequate_core_complete_rate": values.get(
                    "adequate_core_complete_rate"
                ),
                "c_score": values.get("mean_c_score"),
                "branch_alignment_rate": values.get("branch_alignment_rate"),
            }
        )
    return rows


def _pilot_condition_rows(
    trial_rows: Sequence[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Describe stored N=1 trial values without creating CaseAggregate values."""

    rows: list[dict[str, Any]] = []
    for condition in sorted({str(row["condition"]) for row in trial_rows}):
        eligible = [
            row
            for row in trial_rows
            if row.get("condition") == condition
            and row.get("_scientifically_eligible") is True
        ]

        def mean(field: str) -> float | None:
            values = [
                float(row[field])
                for row in eligible
                if isinstance(row.get(field), (int, float, bool))
            ]
            return statistics.fmean(values) if values else None

        modes = {
            str(row["finding_recall_mode"])
            for row in eligible
            if row.get("finding_recall_mode")
        }
        condition_mode = (
            next(iter(modes))
            if len(modes) == 1
            else ("SEMANTIC_ADEQUATE_CORE" if condition.endswith("-F2") else "FIXED_REFERENCE_CORE")
        )
        rows.append(
            {
                "model": model,
                "condition": condition,
                "eligible_case_count": len(eligible),
                "o_score": mean("o_score"),
                "urs": mean("urs"),
                "resolved_o_compliance": mean("resolved_o_compliance"),
                "finding_precision": mean("finding_precision"),
                "core_finding_recall": (
                    mean("core_finding_recall")
                        if condition_mode == "FIXED_REFERENCE_CORE"
                    else None
                ),
                "finding_requirement_recall": mean(
                    "finding_requirement_recall"
                ),
                "finding_recall_mode": (
                    condition_mode
                ),
                "adequate_core_complete_rate": mean("adequate_core_complete"),
                "c_score": mean("c_score"),
                "branch_alignment_rate": mean("branch_alignment"),
            }
        )
    return rows


def _pilot_efficiency_rows(
    trial_rows: Sequence[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Summarize only persisted tested-model telemetry for the N=1 pilot."""

    groups: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("overall", trial_rows)
    ]
    groups.extend(
        (
            condition,
            tuple(row for row in trial_rows if row.get("condition") == condition),
        )
        for condition in sorted({str(row["condition"]) for row in trial_rows})
    )
    output: list[dict[str, Any]] = []
    for condition, group in groups:
        for metric in PILOT_EFFICIENCY_METRICS:
            values = [
                float(row[metric])
                for row in group
                # Provider/runtime failures are not model efficiency
                # observations.  Keep them visible in trial_metrics, but do
                # not let their zero/short failure telemetry bias efficiency
                # means or case counts.
                if str(row.get("status", "")).upper() != "INFRASTRUCTURE_INVALID"
                if isinstance(row.get(metric), (int, float))
                and not isinstance(row.get(metric), bool)
            ]
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "case_count": len(values),
                }
            )
    return output


def _pilot_efficiency_rows_from_collection(
    collection_state: Mapping[str, Any], model: str
) -> list[dict[str, Any]]:
    """Summarize tested-model telemetry independently of evaluator status.

    Evaluator transport failures are not model efficiency observations.  The
    collection state is the authoritative source for N=1 runtime telemetry;
    it may contain a completed model run whose later evaluator replay failed.
    ``MODEL_NONCOMPLETION`` remains a valid accounted model observation and is
    included, while collection ``INFRASTRUCTURE_INVALID`` rows are excluded.
    """

    observations = collection_state.get("observations") or ()
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("collection state observations must be a sequence")
    valid_statuses = {"COMPLETED", "MODEL_NONCOMPLETION"}
    rows: list[dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        if str(observation.get("status", "")).upper() not in valid_statuses:
            continue
        efficiency = observation.get("efficiency") or {}
        if not isinstance(efficiency, Mapping):
            continue
        row = {
            "condition": observation.get("condition"),
            **{metric: efficiency.get(metric) for metric in PILOT_EFFICIENCY_METRICS},
        }
        rows.append(row)

    groups: list[tuple[str, Sequence[Mapping[str, Any]]]] = [("overall", rows)]
    groups.extend(
        (
            condition,
            tuple(row for row in rows if row.get("condition") == condition),
        )
        for condition in sorted(
            {str(row.get("condition")) for row in rows if row.get("condition")}
        )
    )
    output: list[dict[str, Any]] = []
    for condition, group in groups:
        for metric in PILOT_EFFICIENCY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if isinstance(row.get(metric), (int, float))
                and not isinstance(row.get(metric), bool)
            ]
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "case_count": len(values),
                }
            )
    return output


def _pilot_reliability_rows_from_collection(
    collection_state: Mapping[str, Any], model: str
) -> list[dict[str, Any]]:
    """Render provider/retry telemetry without mixing it into model scores."""

    observations = collection_state.get("observations") or ()
    valid_statuses = {"COMPLETED", "MODEL_NONCOMPLETION"}
    rows: list[Mapping[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        if str(observation.get("status", "")).upper() not in valid_statuses:
            continue
        reliability = observation.get("reliability") or {}
        efficiency = observation.get("efficiency") or {}
        if not isinstance(reliability, Mapping) or not isinstance(efficiency, Mapping):
            continue
        retry_count = reliability.get("infrastructure_retry_count")
        rows.append(
            {
                "condition": observation.get("condition"),
                "infrastructure_retry_count": retry_count,
                "infrastructure_failed_request_time": reliability.get(
                    "infrastructure_failed_request_time"
                ),
                "retry_backoff_time": reliability.get("retry_backoff_time"),
                "raw_wall_clock_time": efficiency.get("wall_clock_time"),
                "clean_wall_clock_time": (
                    efficiency.get("wall_clock_time")
                    if isinstance(retry_count, (int, float))
                    and not isinstance(retry_count, bool)
                    and retry_count == 0
                    else None
                ),
            }
        )

    groups: list[tuple[str, Sequence[Mapping[str, Any]]]] = [("overall", rows)]
    conditions = sorted({str(row.get("condition")) for row in rows if row.get("condition")})
    for condition in conditions:
        groups.append(
            (
                condition,
                tuple(row for row in rows if str(row.get("condition")) == condition),
            )
        )

    output: list[dict[str, Any]] = []
    for condition, group in groups:
        for metric in RELIABILITY_METRICS:
            values = [
                float(row[metric])
                for row in group
                if isinstance(row.get(metric), (int, float))
                and not isinstance(row.get(metric), bool)
            ]
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": statistics.fmean(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "case_count": len(values),
                }
            )
    return output


def _metric_applicable_for_condition(
    metric: str,
    condition: str,
    row: Mapping[str, Any] | None = None,
) -> bool:
    """Apply frozen condition and explicit response-O applicability rules.

    URS is defined for open-O conditions; resolved-O compliance is not
    applicable to O3, which has no resolved principal dimensions.  The
    C and alignment require all principal O dimensions to be explicit and
    nonconflicting. Execution gaps on explicit O remain applicable nulls.
    """

    if metric == "urs":
        return condition.startswith(("O2", "O3"))
    if metric == "resolved_o_compliance":
        return not condition.startswith("O3")
    mode = None if row is None else row.get("finding_recall_mode")
    if metric == "core_finding_recall":
        # Core recall is the fixed-reference metric.  F2's open finding
        # responsibility is evaluated through requirement recall/adequate
        # core, never by silently reusing a fixed core denominator.
        if mode is not None:
            return str(mode) == "FIXED_REFERENCE_CORE"
        return condition.endswith("-F1")
    if metric == "finding_requirement_recall":
        return condition.endswith(("-F1", "-F2"))
    if metric == "adequate_core_complete":
        return condition.endswith("-F2")
    if metric in {"c_score", "branch_alignment"} and row is not None:
        return (
            row.get("operationalization_determinate") is not False
            and row.get("consistency_unavailable_reason") != "NO_APPLICABLE_FINDINGS"
        )
    return True


def _row_metric_value(row: Mapping[str, Any], metric: str) -> Any:
    value = row.get(metric)
    if metric == "adequate_core_complete" and isinstance(value, bool):
        return float(value)
    return value


_NA_REASONS = {
    "urs": "URS applies only to open-operationalization conditions (O2/O3).",
    "resolved_o_compliance": "Resolved-O compliance is not applicable to O3.",
    "core_finding_recall": "F2 uses requirement/adequate-core recall, not fixed-reference core recall.",
    "adequate_core_complete": "Adequate-core completion applies only to F2.",
    "c_score": "Consistency requires every principal O dimension to be explicit and nonconflicting.",
    "branch_alignment": "Alignment requires every principal O dimension to be explicit and nonconflicting.",
}


def _metric_cell(row: Mapping[str, Any], metric: str) -> dict[str, Any]:
    """Classify one stored metric without making a scientific judgment."""

    condition = str(row.get("condition") or "")
    value = _row_metric_value(row, metric)
    raw_status = str(row.get("status") or "UNKNOWN").upper()
    if not _metric_applicable_for_condition(metric, condition, row):
        return {
            "value": None,
            "status": "NOT_APPLICABLE",
            "reason": (row.get("_explicit_na_reasons") or {}).get(metric)
            or _NA_REASONS.get(metric, "Metric is outside the frozen applicability scope."),
        }
    if raw_status == "INFRASTRUCTURE_INVALID":
        return {
            "value": None,
            "status": "INFRASTRUCTURE_INVALID",
            "reason": "Evaluator/provider infrastructure failure; no scientific value was produced.",
        }
    if raw_status == "MODEL_NONCOMPLETION":
        return {
            "value": None,
            "status": "MODEL_NONCOMPLETION",
            "reason": "The tested model did not complete this case.",
        }
    if metric in {"c_score", "branch_alignment"} and row.get("consistency_unavailable_reason") == "NO_EXECUTABLE_O_EVIDENCE":
        return {"value": None, "status": "PENDING", "reason": "NO_EXECUTABLE_O_EVIDENCE: explicit O has no verifiable execution branch."}
    explicit_na_reason = (row.get("_explicit_na_reasons") or {}).get(metric)
    if explicit_na_reason:
        return {"value": None, "status": "NOT_APPLICABLE", "reason": explicit_na_reason}
    if value is not None and value != "":
        return {"value": value, "status": "NUMERIC", "reason": "Finalized value copied from the evaluation record."}
    pending_reason = row.get("_pending_reason")
    pending_type = row.get("_pending_type")
    if pending_reason or pending_type or raw_status == "PENDING":
        detail = str(pending_reason or pending_type or "scientific evaluation remains unresolved")
        return {"value": None, "status": "PENDING", "reason": detail}
    # A completed record with an applicable missing value is not silently
    # treated as N/A or zero.  Keep it visible as an unexplained null so the
    # report consumer can fail closed.
    return {
        "value": None,
        "status": "UNEXPLAINED_NULL",
        "reason": "Applicable metric has no value in the finalized record.",
    }


def _metric_matrix_rows(rows: Sequence[Mapping[str, Any]], model: str) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for row in rows:
        for metric in DENOMINATOR_METRICS:
            cell = _metric_cell(row, metric)
            matrix.append(
                {
                    "model": model,
                    "case_id": row.get("case_id"),
                    "condition": row.get("condition"),
                    "trial": row.get("trial"),
                    "metric": metric,
                    **cell,
                }
            )
    return matrix


def _metric_coverage_rows(
    rows: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Count explicit cell states and assert that no cell disappears."""

    groups: list[tuple[str, Sequence[Mapping[str, Any]]]] = [("overall", rows)]
    groups.extend(
        (
            condition,
            tuple(row for row in rows if str(row.get("condition")) == condition),
        )
        for condition in sorted({str(row.get("condition")) for row in rows})
    )
    output: list[dict[str, Any]] = []
    for condition, group in groups:
        case_keys = {
            (str(row.get("case_id")), row.get("trial"))
            for row in group
        }
        for metric in DENOMINATOR_METRICS:
            cells = [
                cell
                for cell in matrix
                if cell.get("metric") == metric
                and (condition == "overall" or str(cell.get("condition")) == condition)
                and (condition == "overall" or (str(cell.get("case_id")), cell.get("trial")) in case_keys)
            ]
            counts = {
                state: sum(str(cell.get("status")) == state for cell in cells)
                for state in (
                    "NUMERIC",
                    "NOT_APPLICABLE",
                    "PENDING",
                    "INFRASTRUCTURE_INVALID",
                    "MODEL_NONCOMPLETION",
                    "UNEXPLAINED_NULL",
                )
            }
            requested_n = len(group)
            if len(cells) != requested_n:
                raise ValueError(
                    f"metric matrix coverage mismatch for {condition}/{metric}: "
                    f"expected {requested_n}, found {len(cells)}"
                )
            if sum(counts.values()) != requested_n:
                raise ValueError(
                    f"metric matrix state conservation failed for {condition}/{metric}"
                )
            applicable_n = counts["NUMERIC"] + counts["PENDING"] + counts["MODEL_NONCOMPLETION"]
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "requested_n": requested_n,
                    "applicable_n": applicable_n,
                    "numeric_n": counts["NUMERIC"],
                    "not_applicable_n": counts["NOT_APPLICABLE"],
                    "pending_n": counts["PENDING"],
                    "infrastructure_n": counts["INFRASTRUCTURE_INVALID"],
                    "model_noncompletion_n": counts["MODEL_NONCOMPLETION"],
                    "unexplained_null_n": counts["UNEXPLAINED_NULL"],
                }
            )
    return output


def _denominator_rows_from_trial_rows(
    rows: Sequence[Mapping[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Build a descriptive denominator view from existing N=1 trial rows.

    This function only counts and summarizes values already present in the
    artifacts.  It does not invoke extraction, matching, adjudication, or any
    other scientific evaluator component.
    """

    groups: list[tuple[str, str, Sequence[Mapping[str, Any]]]] = [("overall", "overall", rows)]
    conditions = sorted({str(row.get("condition", "")) for row in rows})
    groups.extend(
        (condition, condition, tuple(row for row in rows if row.get("condition") == condition))
        for condition in conditions
    )
    output: list[dict[str, Any]] = []
    for _scope, condition, group in groups:
        for metric in DENOMINATOR_METRICS:
            candidates = [
                row
                for row in group
                if str(row.get("status", "")).upper() != "INFRASTRUCTURE_INVALID"
                and _metric_applicable_for_condition(
                    metric,
                    str(row.get("condition", condition)),
                    row,
                )
            ]
            values = [
                _row_metric_value(row, metric)
                for row in candidates
                if _row_metric_value(row, metric) is not None
                and _row_metric_value(row, metric) != ""
            ]
            numeric_values: list[float] = []
            for value in values:
                if isinstance(value, bool):
                    numeric_values.append(float(value))
                elif isinstance(value, (int, float)):
                    numeric_values.append(float(value))
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": (
                        statistics.fmean(numeric_values) if numeric_values else None
                    ),
                    "applicable_n": len(candidates),
                    "finalized_n": len(values),
                    "null_n": len(candidates) - len(values),
                }
            )
    return output


def _formal_denominator_rows(
    summary: Mapping[str, Any],
    model: str,
    case_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project denominator metadata already present in FORMAL summary.

    Means are read from the existing ``aggregate_cases`` summary.  If an old
    summary predates nested denominator metadata, only counts are derived from
    its serialized CaseAggregate rows for compatibility.
    """

    summary_value = summary.get("summary") or {}
    blocks: list[tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]] = [
        ("overall", summary_value.get("overall") or {}, case_rows)
    ]
    blocks.extend(
        (
            str(condition),
            values,
            tuple(row for row in case_rows if row.get("condition") == condition),
        )
        for condition, values in sorted((summary_value.get("by_condition") or {}).items())
    )
    output: list[dict[str, Any]] = []
    for condition, block, rows in blocks:
        nested = block.get("metric_denominators") or {}
        fallback = {
            item["metric"]: item
            for item in _denominator_rows_from_trial_rows(rows, model)
            if item["condition/overall"] == condition
        }
        # For an empty condition group, fallback rows have no useful scope;
        # nested metadata (when present) remains authoritative.
        for metric in DENOMINATOR_METRICS:
            counts = nested.get(metric)
            if not isinstance(counts, Mapping):
                counts = fallback.get(metric) or {
                    "applicable_n": 0,
                    "finalized_n": 0,
                    "null_n": 0,
                }
            mean_key = _METRIC_MEAN_KEYS[metric]
            output.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": block.get(mean_key),
                    "applicable_n": counts.get("applicable_n"),
                    "finalized_n": counts.get("finalized_n"),
                    "null_n": counts.get("null_n"),
                }
            )
    return output


def _efficiency_rows(summary: Mapping[str, Any], model: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    summary_value = summary.get("summary") or {}
    blocks = [("overall", summary_value.get("overall") or {})]
    blocks.extend(
        (condition, values)
        for condition, values in sorted((summary_value.get("by_condition") or {}).items())
    )
    for condition, values in blocks:
        for metric, aggregate in sorted((values.get("efficiency") or {}).items()):
            result.append(
                {
                    "model": model,
                    "condition/overall": condition,
                    "metric": metric,
                    "mean": aggregate.get("mean"),
                    "median": aggregate.get("median"),
                    "case_count": aggregate.get("case_count"),
                }
            )
    return result


def _markdown_table(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return "" if value is None else str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def render_evaluation_report(
    *,
    evaluation_root: str | Path,
    summary_path: str | Path,
    output_dir: str | Path,
    model: str,
    collection_state_path: str | Path | None = None,
) -> dict[str, Path]:
    """Render metric CSV artifacts and one Markdown inspection report."""

    root = Path(evaluation_root)
    summary = _read_json(Path(summary_path))
    mode = str(summary.get("mode", "PILOT"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if mode == "FORMAL":
        trial_rows = _formal_trial_rows(summary, model)
        case_rows = [
            _case_metric_row(model, aggregate)
            for aggregate in summary.get("case_aggregates", ())
        ]
        condition_rows = _condition_rows(summary, model)
        efficiency_rows = _efficiency_rows(summary, model)
        denominator_rows = _formal_denominator_rows(summary, model, case_rows)
    else:
        trial_rows = _artifact_trial_rows(root, model)
        # N=1 remains a trial-level pilot: no CaseAggregate is fabricated.
        # Condition and efficiency tables are descriptive arithmetic views of
        # the exact stored trial values.
        case_rows = []
        condition_rows = _pilot_condition_rows(trial_rows, model)
        if collection_state_path is not None:
            collection_state = _read_json(Path(collection_state_path))
            efficiency_rows = _pilot_efficiency_rows_from_collection(
                collection_state, model
            )
        else:
            efficiency_rows = _pilot_efficiency_rows(trial_rows, model)
        denominator_rows = _denominator_rows_from_trial_rows(trial_rows, model)

    # A separate matrix makes every case×metric cell auditable.  It is a
    # projection of stored records only: pending scientific decisions remain
    # pending, model noncompletion remains distinct, and N/A is assigned only
    # by the frozen applicability rules above.
    metric_matrix_rows = _metric_matrix_rows(trial_rows, model)
    metric_coverage_rows = _metric_coverage_rows(
        trial_rows, metric_matrix_rows, model
    )

    paths = {
        "trial_metrics": output / "trial_metrics.csv",
        "case_metrics": output / "case_metrics.csv",
        "condition_metrics": output / "condition_metrics.csv",
        "efficiency_metrics": output / "efficiency_metrics.csv",
        "metric_denominators": output / "metric_denominators.csv",
        "metric_matrix": output / "metric_matrix.csv",
        "metric_coverage": output / "metric_coverage.csv",
        "infrastructure_reliability": output / "infrastructure_reliability.csv",
        "quality_efficiency": output / "quality_efficiency.json",
        "report": output / "report.md",
    }
    paths["quality_efficiency"].write_text(
        json.dumps(quality_efficiency_summary(trial_rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["trial_metrics"], TRIAL_COLUMNS, trial_rows)
    _write_csv(paths["case_metrics"], CASE_COLUMNS, case_rows)
    _write_csv(paths["condition_metrics"], CONDITION_COLUMNS, condition_rows)
    _write_csv(paths["efficiency_metrics"], EFFICIENCY_COLUMNS, efficiency_rows)
    _write_csv(paths["metric_denominators"], DENOMINATOR_COLUMNS, denominator_rows)
    _write_csv(paths["metric_matrix"], METRIC_MATRIX_COLUMNS, metric_matrix_rows)
    _write_csv(paths["metric_coverage"], METRIC_COVERAGE_COLUMNS, metric_coverage_rows)
    if mode == "FORMAL" or collection_state_path is None:
        reliability_rows = []
    else:
        reliability_rows = _pilot_reliability_rows_from_collection(
            _read_json(Path(collection_state_path)), model
        )
    _write_csv(
        paths["infrastructure_reliability"],
        EFFICIENCY_COLUMNS,
        reliability_rows,
    )

    status_counts: dict[str, int] = {}
    for row in trial_rows:
        status = str(row.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    scientifically_finalized_trial_count = sum(
        row.get("_scientifically_eligible") is True
        and str(row.get("status") or "UNKNOWN") == "COMPLETED"
        for row in trial_rows
    )
    terminal_record_count = sum(
        str(row.get("status") or "UNKNOWN") != "PENDING" for row in trial_rows
    )
    coverage_lines = [
        "## Evaluation Coverage",
        "",
        f"- Requested trials: `{len(trial_rows)}`",
        f"- Terminal records: `{terminal_record_count}`",
        f"- Scientifically finalized trials: `{scientifically_finalized_trial_count}`",
        f"- Evaluation pending: `{status_counts.get('PENDING', 0)}`",
        f"- Infrastructure invalid: `{status_counts.get('INFRASTRUCTURE_INVALID', 0)}`",
        f"- Status counts: `{dict(sorted(status_counts.items()))}`",
        "",
        "Per-metric counts use the frozen applicability scope; unresolved values "
        "remain null and are never converted to zero.",
        "",
        _markdown_table(DENOMINATOR_COLUMNS, denominator_rows),
        "",
    ]
    manifest_variants = summary.get("evaluation_manifest_variants")
    if isinstance(manifest_variants, Mapping) and len(manifest_variants) > 1:
        coverage_lines[0:0] = [
            "## Evaluator Configuration Notice",
            "",
            f"- Distinct persisted evaluator manifests: `{len(manifest_variants)}`",
            "- This pilot contains heterogeneous evaluator generation settings; scientific values are descriptive and must not be compared as a single calibrated benchmark estimate.",
            "",
        ]

    sections = [
        "# FlowIntentBench Evaluation Report",
        "",
        f"- Model: `{model}`",
        f"- Mode: `{mode}`",
        f"- Status: `{summary.get('status', '')}`",
        "",
        "## Trial Metrics",
        "",
        _markdown_table(TRIAL_COLUMNS, trial_rows),
        "",
        "## Case Metrics",
        "",
        _markdown_table(CASE_COLUMNS, case_rows),
        "",
        "## Condition Metrics",
        "",
        _markdown_table(CONDITION_COLUMNS, condition_rows),
        "",
        "## Efficiency Metrics",
        "",
        _markdown_table(EFFICIENCY_COLUMNS, efficiency_rows),
        "",
        "## Infrastructure Reliability",
        "",
        "Retry/failure telemetry is reported separately from scientific and model-efficiency metrics.",
        "",
        _markdown_table(EFFICIENCY_COLUMNS, reliability_rows),
        "",
        "## Metric Cell Matrix",
        "",
        "Each cell is explicitly `NUMERIC`, `NOT_APPLICABLE`, `PENDING`, `MODEL_NONCOMPLETION`, or `INFRASTRUCTURE_INVALID`. An applicable missing value is `UNEXPLAINED_NULL` and is never silently averaged.",
        "",
        _markdown_table(METRIC_COVERAGE_COLUMNS, metric_coverage_rows),
        "",
        _markdown_table(METRIC_MATRIX_COLUMNS, metric_matrix_rows),
        "",
        *coverage_lines,
    ]
    paths["report"].write_text("\n".join(sections), encoding="utf-8")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--collection-state",
        type=Path,
        default=None,
        help="optional tested-model collection_state.json used for N=1 efficiency telemetry",
    )
    args = parser.parse_args()
    render_evaluation_report(
        evaluation_root=args.evaluation_root,
        summary_path=args.summary,
        output_dir=args.output_dir,
        model=args.model,
        collection_state_path=args.collection_state,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
