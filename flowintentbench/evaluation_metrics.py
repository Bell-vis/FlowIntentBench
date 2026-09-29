"""Deterministic calculation layer for benchmark evaluation metrics.

This module intentionally consumes *adjudicated* evaluation inputs.  It does
not parse model natural-language answers, perform semantic matching, apply
tolerances, or decide which runs belong in an aggregation.  The separate
``evaluator`` module supplies those adjudicated inputs and status decisions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence


class UndefinedMetricError(ValueError):
    """Raised when a supplied evaluation structure is not mathematically usable."""


EFFICIENCY_METRIC_NAMES = (
    "input_tokens",
    "output_tokens",
    "model_turn_count",
    "tool_call_count",
    "python_execution_count",
    "wall_clock_time",
    "provider_reported_cost",
)

# Scientific fields that are already part of the frozen evaluator contract.
# ``core_finding_recall`` remains here as a compatibility diagnostic; the
# cross-condition headline for Findings is ``finding_requirement_recall``.
SCIENTIFIC_METRIC_NAMES = (
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


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _binary(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f"{field_name} must be 0 or 1")
    return int(value)


def _count(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _optional_non_negative_number(value: Any, field_name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a non-negative finite number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number or None")
    return value


@dataclass(frozen=True)
class OperationalizationBranchEvaluation:
    """External 0/1 judgments for one accepted operationalization branch."""

    branch_id: str
    principal_dimension_matches: Mapping[str, int]
    unresolved_dimension_matches: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty_text(self.branch_id, "branch_id")
        if not self.principal_dimension_matches:
            raise ValueError("principal_dimension_matches must be non-empty")
        principal = set(self.principal_dimension_matches)
        unresolved = set(self.unresolved_dimension_matches)
        if not unresolved <= principal:
            raise ValueError(
                "unresolved_dimension_matches must be a subset of principal dimensions"
            )
        for dimension, value in self.principal_dimension_matches.items():
            _non_empty_text(dimension, "principal dimension")
            _binary(value, f"principal_dimension_matches[{dimension!r}]")
        for dimension, value in self.unresolved_dimension_matches.items():
            _non_empty_text(dimension, "unresolved dimension")
            _binary(value, f"unresolved_dimension_matches[{dimension!r}]")


@dataclass(frozen=True)
class OperationalizationMetricResult:
    o_score: float
    best_o_branches: tuple[str, ...]
    urs: float | None
    resolved_o_compliance: float | None = None

    def to_dict(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        value = {
            "o_score": self.o_score,
            "best_o_branches": list(self.best_o_branches),
            "urs": self.urs,
        }
        if include_diagnostics:
            value["resolved_o_compliance"] = self.resolved_o_compliance
        return value


@dataclass(frozen=True)
class FindingBranchEvaluation:
    """External finding counts for one accepted O -> G(O) branch."""

    branch_id: str
    predicted_finding_count: int
    valid_predicted_finding_count: int
    core_finding_count: int
    matched_core_finding_count: int
    scoring_mode: str = "LEGACY_FIXED_CORE"
    mandatory_role_count: int = 0
    matched_mandatory_role_count: int = 0
    adequate_set_evaluations: tuple[Mapping[str, Any], ...] = ()
    best_adequate_set_ids: tuple[str, ...] = ()
    best_adequate_set_id: str | None = None
    best_adequate_set_recall: float | None = None
    novel_role_matches: tuple[Mapping[str, Any], ...] = ()
    role_match_provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty_text(self.branch_id, "branch_id")
        predicted = _count(self.predicted_finding_count, "predicted_finding_count")
        valid = _count(self.valid_predicted_finding_count, "valid_predicted_finding_count")
        core = _count(self.core_finding_count, "core_finding_count")
        matched = _count(self.matched_core_finding_count, "matched_core_finding_count")
        if valid > predicted:
            raise ValueError("valid_predicted_finding_count cannot exceed predicted_finding_count")
        if matched > core:
            raise ValueError("matched_core_finding_count cannot exceed core_finding_count")
        if self.scoring_mode not in {"LEGACY_FIXED_CORE", "SEMANTIC_ADEQUATE_CORE"}:
            raise ValueError("unsupported finding scoring mode")
        mandatory = _count(self.mandatory_role_count, "mandatory_role_count")
        matched_mandatory = _count(self.matched_mandatory_role_count, "matched_mandatory_role_count")
        if matched_mandatory > mandatory:
            raise ValueError("matched_mandatory_role_count cannot exceed mandatory_role_count")
        if self.scoring_mode == "SEMANTIC_ADEQUATE_CORE" and self.best_adequate_set_recall is None:
            raise ValueError("semantic adequate-core mode requires best_adequate_set_recall")
        if self.best_adequate_set_id is not None:
            _non_empty_text(self.best_adequate_set_id, "best_adequate_set_id")
            if not self.best_adequate_set_ids:
                object.__setattr__(self, "best_adequate_set_ids", (self.best_adequate_set_id,))
        elif len(self.best_adequate_set_ids) == 1:
            object.__setattr__(self, "best_adequate_set_id", self.best_adequate_set_ids[0])


@dataclass(frozen=True)
class FindingMetricResult:
    finding_precision: float | None
    core_finding_recall: float | None
    best_finding_branches: tuple[str, ...] | None
    finding_recall_mode: str = "FIXED_REFERENCE_CORE"
    finding_requirement_recall: float | None = None
    adequate_core_complete: bool | None = None

    def __post_init__(self) -> None:
        if self.finding_recall_mode not in {"FIXED_REFERENCE_CORE", "SEMANTIC_ADEQUATE_CORE"}:
            raise ValueError("unsupported finding_recall_mode")
        if self.finding_recall_mode == "FIXED_REFERENCE_CORE":
            if self.finding_requirement_recall is None:
                object.__setattr__(self, "finding_requirement_recall", self.core_finding_recall)
        elif self.finding_requirement_recall is None:
            raise ValueError(
                "semantic adequate-core mode requires finding_requirement_recall"
            )
        if self.finding_recall_mode == "SEMANTIC_ADEQUATE_CORE" and self.adequate_core_complete is None:
            object.__setattr__(self, "adequate_core_complete", False)

    def to_dict(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        value = {
            "finding_precision": self.finding_precision,
            "core_finding_recall": self.core_finding_recall,
            "best_finding_branches": (
                None if self.best_finding_branches is None else list(self.best_finding_branches)
            ),
        }
        if include_diagnostics:
            value.update({
                "finding_requirement_recall": self.finding_requirement_recall,
                "finding_recall_mode": self.finding_recall_mode,
                "adequate_core_complete": self.adequate_core_complete,
            })
        return value


@dataclass(frozen=True)
class ConsistencyEvaluation:
    """O-conditioned 0/1 judgments for response-level applicable Findings."""

    finding_results: Mapping[str, int]
    # None preserves the interpretation of records written before explicit
    # response-O applicability was recorded. False means missing/ambiguous/
    # conflicting O; a scientifically wrong but explicit O is still True.
    operationalization_determinate: bool | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for prediction_id, value in self.finding_results.items():
            _non_empty_text(prediction_id, "prediction_id")
            _binary(value, f"finding_results[{prediction_id!r}]")


@dataclass(frozen=True)
class EfficiencyObservation:
    """Raw efficiency telemetry; nullable values remain nullable."""

    input_tokens: int | None
    output_tokens: int | None
    model_turn_count: int | None
    python_execution_count: int | None
    wall_clock_time: float | None
    tool_call_count: int | None = None
    provider_reported_cost: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "model_turn_count",
            "tool_call_count",
            "python_execution_count",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _count(value, field_name)
        _optional_non_negative_number(self.wall_clock_time, "wall_clock_time")
        _optional_non_negative_number(
            self.provider_reported_cost, "provider_reported_cost"
        )

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_turn_count": self.model_turn_count,
            "tool_call_count": self.tool_call_count,
            "python_execution_count": self.python_execution_count,
            "wall_clock_time": self.wall_clock_time,
            "provider_reported_cost": self.provider_reported_cost,
        }


@dataclass(frozen=True)
class EfficiencyAggregate:
    """Case/benchmark efficiency values after explicit arithmetic averaging.

    Raw run telemetry uses integer counters and is represented by
    :class:`EfficiencyObservation`.  Trial and case aggregation may produce
    fractional means, so it has a separate type instead of weakening the raw
    telemetry contract.
    """

    input_tokens: float | None
    output_tokens: float | None
    model_turn_count: float | None
    python_execution_count: float | None
    wall_clock_time: float | None
    tool_call_count: float | None = None
    provider_reported_cost: float | None = None

    def __post_init__(self) -> None:
        for field_name in EFFICIENCY_METRIC_NAMES:
            _optional_non_negative_number(getattr(self, field_name), field_name)

    @property
    def total_tokens(self) -> float | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_turn_count": self.model_turn_count,
            "tool_call_count": self.tool_call_count,
            "python_execution_count": self.python_execution_count,
            "wall_clock_time": self.wall_clock_time,
            "provider_reported_cost": self.provider_reported_cost,
        }


def efficiency_observation_from_run_record(record: Any) -> EfficiencyObservation:
    """Read existing runner telemetry without applying status/missing policy."""

    return EfficiencyObservation(
        input_tokens=getattr(record, "input_tokens"),
        output_tokens=getattr(record, "output_tokens"),
        model_turn_count=getattr(record, "model_turn_count"),
        tool_call_count=getattr(
            record,
            "tool_call_count",
            getattr(record, "python_execution_count", None),
        ),
        python_execution_count=getattr(record, "python_execution_count"),
        wall_clock_time=getattr(record, "wall_clock_time"),
        provider_reported_cost=getattr(record, "provider_reported_cost", None),
    )


@dataclass(frozen=True)
class CaseMetricInput:
    """All externally adjudicated inputs needed for one case-level result."""

    case_id: str
    condition: str
    operationalization_branches: tuple[OperationalizationBranchEvaluation, ...]
    finding_branches: tuple[FindingBranchEvaluation, ...]
    consistency: ConsistencyEvaluation
    efficiency: EfficiencyObservation

    def __post_init__(self) -> None:
        _non_empty_text(self.case_id, "case_id")
        _non_empty_text(self.condition, "condition")
        if not self.operationalization_branches:
            raise ValueError("operationalization_branches must be non-empty")
        if not self.finding_branches:
            raise ValueError("finding_branches must be non-empty")
        _unique_branch_ids(self.operationalization_branches)
        _unique_branch_ids(self.finding_branches)


@dataclass(frozen=True)
class ConsistencyMetricResult:
    c_score: float | None
    branch_alignment: float | None
    operationalization_determinate: bool | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {"c_score": self.c_score, "branch_alignment": self.branch_alignment}
        if self.operationalization_determinate is not None:
            value["operationalization_determinate"] = self.operationalization_determinate
        if self.unavailable_reason is not None:
            value["unavailable_reason"] = self.unavailable_reason
        return value


@dataclass(frozen=True)
class CaseMetricResult:
    case_id: str
    condition: str
    scientific_operationalization: OperationalizationMetricResult
    scientific_findings: FindingMetricResult
    o_f_consistency: ConsistencyMetricResult
    efficiency: EfficiencyObservation | EfficiencyAggregate

    def to_dict(self) -> dict[str, Any]:
        # These fields are part of the frozen public evaluation contract for
        # both F1 and F2. ``core_finding_recall`` remains for compatibility,
        # but must not cause ROC or requirement-recall fields to disappear.
        include_diagnostics = True
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "scientific_operationalization": self.scientific_operationalization.to_dict(include_diagnostics=include_diagnostics),
            "scientific_findings": self.scientific_findings.to_dict(include_diagnostics=include_diagnostics),
            "o_f_consistency": self.o_f_consistency.to_dict(),
            "efficiency": self.efficiency.to_dict(),
        }


@dataclass(frozen=True)
class CaseAggregateMetricResult:
    """Scientific/efficiency estimates after trial-to-case aggregation.

    Trial branch identities are intentionally absent: a case-level aggregate
    has no scientifically privileged best branch.  Branch diagnostics remain
    in the serialized trial evaluation records.
    """

    case_id: str
    condition: str
    o_score: float
    urs: float | None
    finding_precision: float
    core_finding_recall: float | None
    c_score: float | None
    branch_alignment: float | None
    efficiency: EfficiencyAggregate
    resolved_o_compliance: float | None = None
    finding_requirement_recall: float | None = None
    finding_recall_mode: str = "FIXED_REFERENCE_CORE"
    adequate_core_complete: bool | None = None
    adequate_core_complete_rate: float | None = None
    metric_trial_denominators: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    determinate_o_trial_count: int | None = None
    determinate_o_trial_rate: float | None = None

    def __post_init__(self) -> None:
        if self.finding_recall_mode not in {"FIXED_REFERENCE_CORE", "SEMANTIC_ADEQUATE_CORE"}:
            raise ValueError("unsupported finding_recall_mode")
        if self.finding_recall_mode == "SEMANTIC_ADEQUATE_CORE":
            if self.core_finding_recall is not None:
                raise ValueError("F2 case aggregate must not expose core_finding_recall")
            if self.finding_requirement_recall is None:
                raise ValueError("F2 case aggregate requires finding_requirement_recall")
        elif self.core_finding_recall is None:
            raise ValueError("F1 case aggregate requires core_finding_recall")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "o_score": self.o_score,
            "urs": self.urs,
            "resolved_o_compliance": self.resolved_o_compliance,
            "finding_precision": self.finding_precision,
            "core_finding_recall": self.core_finding_recall,
            "finding_requirement_recall": self.finding_requirement_recall,
            "finding_recall_mode": self.finding_recall_mode,
            "adequate_core_complete": self.adequate_core_complete,
            "adequate_core_complete_rate": self.adequate_core_complete_rate,
            "c_score": self.c_score,
            "branch_alignment": self.branch_alignment,
            "metric_trial_denominators": dict(self.metric_trial_denominators),
            "determinate_o_trial_count": self.determinate_o_trial_count,
            "determinate_o_trial_rate": self.determinate_o_trial_rate,
            "efficiency": self.efficiency.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkMetricValues:
    """Explicitly selected valid case-level values for macro reporting.

    The caller, not this layer, decides which observations are valid and which
    URS-applicable cases belong in ``urs_scores``.
    """

    o_scores: tuple[float, ...] = ()
    resolved_o_compliance_scores: tuple[float, ...] = ()
    urs_scores: tuple[float, ...] = ()
    finding_precision_scores: tuple[float, ...] = ()
    core_finding_recall_scores: tuple[float, ...] = ()
    finding_requirement_recall_scores: tuple[float, ...] = ()
    c_scores: tuple[float, ...] = ()
    branch_alignment_scores: tuple[float, ...] = ()
    efficiency_values: Mapping[str, tuple[Real, ...]] = field(default_factory=dict)
    # Appended after the pre-existing fields to preserve positional
    # construction compatibility for callers of the original dataclass.
    adequate_core_complete_scores: tuple[float, ...] = ()
    # Optional applicability counts are supplied by the caller that knows
    # the case contract (for example, URS is inapplicable for O1).  When
    # omitted, a metric's non-null values are its complete denominator.  This
    # keeps the public type backward compatible while making null/pending
    # observations explicit whenever the caller has that information.
    metric_applicable_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        requirement_scores = self.finding_requirement_recall_scores
        series = {
            "o_score": self.o_scores,
            "urs": self.urs_scores,
            "resolved_o_compliance": self.resolved_o_compliance_scores,
            "finding_precision": self.finding_precision_scores,
            "core_finding_recall": self.core_finding_recall_scores,
            "finding_requirement_recall": requirement_scores,
            "adequate_core_complete": self.adequate_core_complete_scores,
            "c_score": self.c_scores,
            "branch_alignment": self.branch_alignment_scores,
        }
        unknown = set(self.metric_applicable_counts) - set(SCIENTIFIC_METRIC_NAMES)
        if unknown:
            raise ValueError(f"unknown scientific metric names: {sorted(unknown)}")
        for name, raw_count in self.metric_applicable_counts.items():
            count = _count(raw_count, f"metric_applicable_counts[{name!r}]")
            if count < len(series[name]):
                raise ValueError(
                    f"metric_applicable_counts[{name!r}] cannot be smaller than finalized values"
                )


def _unique_branch_ids(branches: Iterable[Any]) -> None:
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("branch_id values must be unique")


def _coerce_operation_branch(value: OperationalizationBranchEvaluation | Mapping[str, Any]) -> OperationalizationBranchEvaluation:
    if isinstance(value, OperationalizationBranchEvaluation):
        return value
    return OperationalizationBranchEvaluation(**value)


def _coerce_finding_branch(value: FindingBranchEvaluation | Mapping[str, Any]) -> FindingBranchEvaluation:
    if isinstance(value, FindingBranchEvaluation):
        return value
    return FindingBranchEvaluation(**value)


def _coerce_consistency(value: ConsistencyEvaluation | Mapping[str, int]) -> ConsistencyEvaluation:
    if isinstance(value, ConsistencyEvaluation):
        return value
    return ConsistencyEvaluation(value)


def _validated_operation_branches(
    branches: Iterable[OperationalizationBranchEvaluation | Mapping[str, Any]],
) -> tuple[OperationalizationBranchEvaluation, ...]:
    values = tuple(_coerce_operation_branch(branch) for branch in branches)
    if not values:
        raise ValueError("at least one operationalization branch is required")
    _unique_branch_ids(values)
    principal_dimensions = set(values[0].principal_dimension_matches)
    unresolved_dimensions = set(values[0].unresolved_dimension_matches)
    for branch in values[1:]:
        if set(branch.principal_dimension_matches) != principal_dimensions:
            raise ValueError("all branches must use the same principal dimensions")
        if set(branch.unresolved_dimension_matches) != unresolved_dimensions:
            raise ValueError("all branches must use the same unresolved dimensions")
    return values


def compute_o_score(
    branches: Iterable[OperationalizationBranchEvaluation | Mapping[str, Any]],
) -> OperationalizationMetricResult:
    """Compute branch-aware O-Score, URS, and best O branches."""

    values = _validated_operation_branches(branches)
    principal_scores = {
        branch.branch_id: sum(branch.principal_dimension_matches.values()) / len(branch.principal_dimension_matches)
        for branch in values
    }
    best_score = max(principal_scores.values())
    best_branches = tuple(
        branch_id for branch_id, score in principal_scores.items() if score == best_score
    )
    unresolved = next(iter(values)).unresolved_dimension_matches
    if not unresolved:
        urs = None
    else:
        urs_scores = [
            sum(branch.unresolved_dimension_matches.values()) / len(unresolved)
            for branch in values
        ]
        urs = max(urs_scores)
    resolved_dimensions = set(values[0].principal_dimension_matches) - set(unresolved)
    if not resolved_dimensions:
        resolved_compliance = None
    else:
        resolved_scores = [
            sum(branch.principal_dimension_matches[item] for item in resolved_dimensions)
            / len(resolved_dimensions)
            for branch in values
        ]
        resolved_compliance = max(resolved_scores)
    return OperationalizationMetricResult(
        o_score=best_score,
        best_o_branches=best_branches,
        urs=urs,
        resolved_o_compliance=resolved_compliance,
    )


def compute_urs(
    branches: Iterable[OperationalizationBranchEvaluation | Mapping[str, Any]],
) -> float | None:
    """Return URS; O1-style cases with no unresolved dimensions return None."""

    return compute_o_score(branches).urs


def compute_finding_precision(branch: FindingBranchEvaluation | Mapping[str, Any]) -> float | None:
    value = _coerce_finding_branch(branch)
    if value.predicted_finding_count == 0:
        return 0.0
    return value.valid_predicted_finding_count / value.predicted_finding_count


def compute_core_finding_recall(branch: FindingBranchEvaluation | Mapping[str, Any]) -> float | None:
    value = _coerce_finding_branch(branch)
    if value.scoring_mode == "SEMANTIC_ADEQUATE_CORE":
        if value.best_adequate_set_recall is None:
            raise UndefinedMetricError("semantic adequate-core score is missing")
        return float(value.best_adequate_set_recall)
    if value.core_finding_count == 0:
        raise UndefinedMetricError(
            "core_finding_count must be positive for a formally evaluated case"
        )
    return value.matched_core_finding_count / value.core_finding_count


def select_best_finding_branches(
    branches: Iterable[FindingBranchEvaluation | Mapping[str, Any]],
) -> tuple[str, ...] | None:
    """Select branches by (core recall, then precision), retaining all ties.

    A formally evaluated branch always has at least one core Finding.  An
    empty prediction set has Precision and Recall equal to zero.
    """

    values = tuple(_coerce_finding_branch(branch) for branch in branches)
    if not values:
        raise ValueError("at least one finding branch is required")
    _unique_branch_ids(values)
    scored: list[tuple[str, float, float]] = []
    for branch in values:
        precision = compute_finding_precision(branch)
        recall = compute_core_finding_recall(branch)
        if precision is None or recall is None:  # pragma: no cover - guarded above
            raise UndefinedMetricError("formal Finding metrics must be defined")
        scored.append((branch.branch_id, recall, precision))
    best_pair = max((recall, precision) for _, recall, precision in scored)
    return tuple(
        branch_id
        for branch_id, recall, precision in scored
        if (recall, precision) == best_pair
    )


def compute_finding_metrics(
    branches: Iterable[FindingBranchEvaluation | Mapping[str, Any]],
) -> FindingMetricResult:
    values = tuple(_coerce_finding_branch(branch) for branch in branches)
    # Every branch shares the response-level F_app denominator. Branches may
    # differ only in which members are valid/consistent with that branch.
    denominators = {branch.predicted_finding_count for branch in values}
    if len(denominators) != 1:
        raise ValueError(
            "all finding branches must use the same response-level F_app denominator"
        )
    denominator = next(iter(denominators))
    best = select_best_finding_branches(values)
    if best is None:
        return FindingMetricResult(None, None, None)
    by_id = {branch.branch_id: branch for branch in values}
    selected = by_id[best[0]]
    recall = compute_core_finding_recall(selected)
    mode = (
        "SEMANTIC_ADEQUATE_CORE"
        if selected.scoring_mode == "SEMANTIC_ADEQUATE_CORE"
        else "FIXED_REFERENCE_CORE"
    )
    adequate_complete = None
    if mode == "SEMANTIC_ADEQUATE_CORE":
        adequate_complete = (
            selected.matched_mandatory_role_count == selected.mandatory_role_count
            and recall is not None
            and recall >= 1.0
        )
    if denominator == 0:
        return FindingMetricResult(
            finding_precision=0.0,
            core_finding_recall=(recall if mode == "FIXED_REFERENCE_CORE" else None),
            best_finding_branches=None,
            finding_recall_mode=mode,
            finding_requirement_recall=recall,
            adequate_core_complete=adequate_complete,
        )
    return FindingMetricResult(
        finding_precision=compute_finding_precision(selected),
        core_finding_recall=(recall if mode == "FIXED_REFERENCE_CORE" else None),
        best_finding_branches=best,
        finding_recall_mode=mode,
        finding_requirement_recall=recall,
        adequate_core_complete=adequate_complete,
    )


def compute_c_score(
    consistency: ConsistencyEvaluation | Mapping[str, int],
) -> float | None:
    """Compute O-conditioned Finding consistency for one selected branch."""

    value = _coerce_consistency(consistency)
    if value.operationalization_determinate is False or not value.finding_results:
        return None
    return sum(value.finding_results.values()) / len(value.finding_results)


def compute_branch_alignment(
    best_o_branches: Iterable[str] | None,
    best_finding_branches: Iterable[str] | None,
) -> int | None:
    """Return 1 iff the two explicit best-branch sets intersect."""

    if best_o_branches is None or best_finding_branches is None:
        return None
    o_branches = {_non_empty_text(branch_id, "best O branch ID") for branch_id in best_o_branches}
    f_branches = {_non_empty_text(branch_id, "best finding branch ID") for branch_id in best_finding_branches}
    if not o_branches or not f_branches:
        return None
    return int(bool(o_branches & f_branches))


def compute_case_metrics(case: CaseMetricInput) -> CaseMetricResult:
    """Compute the complete case-level metric record without status decisions."""

    operation = compute_o_score(case.operationalization_branches)
    findings = compute_finding_metrics(case.finding_branches)
    consistency = ConsistencyMetricResult(
        c_score=compute_c_score(case.consistency),
        branch_alignment=(
            None
            if case.consistency.operationalization_determinate is False
            or not case.consistency.finding_results
            else compute_branch_alignment(
                operation.best_o_branches,
                findings.best_finding_branches,
            )
        ),
        operationalization_determinate=case.consistency.operationalization_determinate,
        unavailable_reason=case.consistency.unavailable_reason,
    )
    return CaseMetricResult(
        case_id=case.case_id,
        condition=case.condition,
        scientific_operationalization=operation,
        scientific_findings=findings,
        o_f_consistency=consistency,
        efficiency=case.efficiency,
    )


def macro_mean(values: Iterable[Real]) -> float | None:
    """Arithmetic mean of explicitly supplied scientific case-level scores."""

    numbers = tuple(_score(value, "case score") for value in values)
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _score(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field_name} must be a finite number in [0, 1]")
    return numeric


def efficiency_summary(
    values: Mapping[str, Iterable[Real]],
) -> dict[str, dict[str, float | None]]:
    """Compute mean/median for explicit valid efficiency value collections.

    Missing observations are not inferred or filled.  An empty collection is
    represented by ``None`` for that supplied metric.
    """

    unknown = sorted(set(values) - set(EFFICIENCY_METRIC_NAMES))
    if unknown:
        raise ValueError(f"unknown efficiency metric names: {unknown}")
    result: dict[str, dict[str, float | None]] = {}
    for name, raw_values in values.items():
        numbers: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} values must be finite non-negative numbers")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"{name} values must be finite non-negative numbers")
            numbers.append(numeric)
        result[name] = {
            "mean": statistics.fmean(numbers) if numbers else None,
            "median": statistics.median(numbers) if numbers else None,
            "case_count": len(numbers),
        }
    return result


def _metric_denominator_block(
    values: BenchmarkMetricValues,
    series: Mapping[str, Sequence[Any]],
) -> dict[str, dict[str, Any]]:
    """Return explicit applicability/finalization counts for each field.

    The metric layer receives already adjudicated values.  It never guesses a
    missing score or turns ``None`` into zero.  A caller may provide a larger
    ``applicable_n`` for an explicitly known scope (for example, pending O2
    trials are applicable to URS); the difference is reported as ``null_n``.
    """

    result: dict[str, dict[str, Any]] = {}
    for name, raw_values in series.items():
        finalized_n = len(raw_values)
        applicable_n = values.metric_applicable_counts.get(name, finalized_n)
        # __post_init__ validates this relation for explicit overrides.
        result[name] = {
            "applicable_n": int(applicable_n),
            "finalized_n": finalized_n,
            "null_n": int(applicable_n) - finalized_n,
        }
    return result


def _metric_block(values: BenchmarkMetricValues) -> dict[str, Any]:
    requirement_scores = values.finding_requirement_recall_scores
    series: dict[str, Sequence[Any]] = {
        "o_score": values.o_scores,
        "urs": values.urs_scores,
        "resolved_o_compliance": values.resolved_o_compliance_scores,
        "finding_precision": values.finding_precision_scores,
        "core_finding_recall": values.core_finding_recall_scores,
        "finding_requirement_recall": requirement_scores,
        "adequate_core_complete": values.adequate_core_complete_scores,
        "c_score": values.c_scores,
        "branch_alignment": values.branch_alignment_scores,
    }
    block = {
        "mean_o_score": macro_mean(values.o_scores),
        "mean_resolved_o_compliance": macro_mean(values.resolved_o_compliance_scores),
        "mean_urs": macro_mean(values.urs_scores),
        "urs_case_count": len(values.urs_scores),
        "mean_finding_precision": macro_mean(values.finding_precision_scores),
        "mean_core_finding_recall": macro_mean(values.core_finding_recall_scores),
        "mean_finding_requirement_recall": macro_mean(requirement_scores),
        "finding_requirement_recall_case_count": len(requirement_scores),
        "adequate_core_complete_rate": macro_mean(values.adequate_core_complete_scores),
        "adequate_core_complete_case_count": len(values.adequate_core_complete_scores),
        "mean_c_score": macro_mean(values.c_scores),
        "branch_alignment_rate": macro_mean(
            tuple(_score(value, "branch_alignment") for value in values.branch_alignment_scores)
        ),
        "efficiency": efficiency_summary(values.efficiency_values),
    }
    # Counts are metadata about the supplied observations, not an additional
    # scientific score.  Keep them nested so existing summary consumers remain
    # compatible with the established mean_* fields.
    denominators = _metric_denominator_block(values, series)
    means = {
        "o_score": block["mean_o_score"],
        "urs": block["mean_urs"],
        "resolved_o_compliance": block["mean_resolved_o_compliance"],
        "finding_precision": block["mean_finding_precision"],
        "core_finding_recall": block["mean_core_finding_recall"],
        "finding_requirement_recall": block["mean_finding_requirement_recall"],
        "adequate_core_complete": block["adequate_core_complete_rate"],
        "c_score": block["mean_c_score"],
        "branch_alignment": block["branch_alignment_rate"],
    }
    for name, counts in denominators.items():
        counts["mean"] = means[name]
    block["metric_denominators"] = denominators
    return block


def benchmark_summary(
    overall: BenchmarkMetricValues,
    *,
    by_condition: Mapping[str, BenchmarkMetricValues] | None = None,
    by_finding_recall_mode: Mapping[str, BenchmarkMetricValues] | None = None,
) -> dict[str, Any]:
    """Build the benchmark-level schema from explicitly selected values."""

    return {
        "overall": _metric_block(overall),
        "by_condition": {
            condition: _metric_block(values)
            for condition, values in (by_condition or {}).items()
        },
        "by_finding_recall_mode": {
            mode: _metric_block(values)
            for mode, values in (by_finding_recall_mode or {}).items()
        },
    }


__all__ = [
    "BenchmarkMetricValues",
    "CaseAggregateMetricResult",
    "CaseMetricInput",
    "CaseMetricResult",
    "ConsistencyEvaluation",
    "ConsistencyMetricResult",
    "EFFICIENCY_METRIC_NAMES",
    "EfficiencyAggregate",
    "EfficiencyObservation",
    "FindingBranchEvaluation",
    "FindingMetricResult",
    "OperationalizationBranchEvaluation",
    "OperationalizationMetricResult",
    "SCIENTIFIC_METRIC_NAMES",
    "UndefinedMetricError",
    "benchmark_summary",
    "compute_branch_alignment",
    "compute_c_score",
    "compute_case_metrics",
    "compute_core_finding_recall",
    "compute_finding_metrics",
    "compute_finding_precision",
    "compute_o_score",
    "compute_urs",
    "efficiency_observation_from_run_record",
    "efficiency_summary",
    "macro_mean",
    "select_best_finding_branches",
]
