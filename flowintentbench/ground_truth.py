"""Construction-time ground truth resources for controlled benchmark cases.

Ground truth is an authored scientific resource.  This module validates its
closed JSON structure and its binding to :class:`CaseConstructionMetadata`,
but it does not execute analyses or infer scientific correctness.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeAlias

from pydantic import Field, field_validator, model_validator

from .case_design import (
    CaseConstructionMetadata,
    CaseDesignModel,
    FindingRequirementCategory,
    OperationalizationDimension,
    normalize_case_text,
    primary_case_type,
)
from .context import EvidenceRecord


FindingValue: TypeAlias = str | int | float | list[float] | list[str] | None


class FindingImportance(str, Enum):
    CORE = "core"
    SUPPORTING = "supporting"


class VerificationSpec(CaseDesignModel):
    """Optional per-finding verification metadata.

    Tolerances are metadata only.  They do not implement scoring, unit
    conversion, or a mesh-dependent verification policy.
    """

    absolute_tolerance: float | None = Field(default=None, ge=0)
    relative_tolerance: float | None = Field(default=None, ge=0)
    spatial_tolerance: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_finite_tolerances(self) -> "VerificationSpec":
        for field_name in (
            "absolute_tolerance",
            "relative_tolerance",
            "spatial_tolerance",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        return self


class OperationalizationDecision(CaseDesignModel):
    dimension: OperationalizationDimension
    statement: str = Field(min_length=1)

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("statement must be a non-empty string")
        return value


class OperationalizationBundle(CaseDesignModel):
    operationalization_id: str = Field(min_length=1)
    decisions: list[OperationalizationDecision] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("operationalization_id")
    @classmethod
    def validate_operationalization_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("operationalization_id must be a non-empty string")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("evidence_ids must contain non-empty strings")
            normalized.append(value.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_ids must be unique within an operationalization")
        return normalized

    @model_validator(mode="after")
    def validate_decisions(self) -> "OperationalizationBundle":
        dimensions = [decision.dimension for decision in self.decisions]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(
                "OperationalizationDimension values must be unique within a bundle"
            )
        return self


def _validate_finding_value(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, bool):
        raise ValueError("ReferenceFinding.value does not accept bool")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("ReferenceFinding.value numeric values must be finite")
        return value
    if not isinstance(value, list):
        raise ValueError(
            "ReferenceFinding.value must be str, int, float, list[float], list[str], or null"
        )
    if all(isinstance(item, str) for item in value):
        return value
    if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        if any(isinstance(item, float) and not math.isfinite(item) for item in value):
            raise ValueError("ReferenceFinding.value numeric values must be finite")
        return value
    raise ValueError(
        "ReferenceFinding.value lists must contain only strings or finite numbers"
    )


class ReferenceFinding(CaseDesignModel):
    finding_id: str = Field(min_length=1)
    category: FindingRequirementCategory
    statement: str = Field(min_length=1)
    importance: FindingImportance
    value: FindingValue = None
    unit: str | None = None
    verification: VerificationSpec | None = None
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("finding_id", "statement", "unit")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("text fields must be non-empty strings when provided")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: Any) -> Any:
        return _validate_finding_value(value)

    @field_validator("evidence_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("evidence_ids must contain non-empty strings")
            normalized.append(value.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidence_ids must be unique within a finding")
        return normalized

    @model_validator(mode="after")
    def validate_verification_contract(self) -> "ReferenceFinding":
        """Reject explicit verification rules that are ambiguous or inapplicable.

        Whether a Finding needs a deterministic numeric rule is decided during
        construction/release review.  This validator only checks rules that the
        author actually supplied; it does not infer a mode from ``value``.
        """

        if self.verification is None:
            return self
        verification = self.verification
        has_scalar_tolerance = (
            verification.absolute_tolerance is not None
            or verification.relative_tolerance is not None
        )
        has_spatial_tolerance = verification.spatial_tolerance is not None
        if has_scalar_tolerance and has_spatial_tolerance:
            raise ValueError(
                "VerificationSpec cannot combine scalar and spatial tolerances"
            )
        if has_scalar_tolerance and (
            not isinstance(self.value, Real) or isinstance(self.value, bool)
        ):
            raise ValueError(
                "absolute/relative tolerance requires a finite continuous numeric scalar value"
            )
        if has_spatial_tolerance and not (
            isinstance(self.value, list)
            and bool(self.value)
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in self.value
            )
        ):
            raise ValueError(
                "spatial_tolerance requires a non-empty numeric-list value"
            )
        return self


class OperationalizationFindingBranch(CaseDesignModel):
    operationalization_id: str = Field(min_length=1)
    findings: list[ReferenceFinding] = Field(min_length=1)

    @field_validator("operationalization_id")
    @classmethod
    def validate_operationalization_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("operationalization_id must be a non-empty string")
        return value

    @model_validator(mode="after")
    def validate_findings(self) -> "OperationalizationFindingBranch":
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding_id values must be unique within one G(O) branch")
        statement_keys = [normalize_case_text(finding.statement) for finding in self.findings]
        if len(statement_keys) != len(set(statement_keys)):
            raise ValueError("duplicate finding statements are not allowed within one G(O) branch")
        return self


class GroundTruth(CaseDesignModel):
    dataset_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    case_family_id: str = Field(min_length=1)
    acceptable_operationalizations: list[OperationalizationBundle] = Field(min_length=1)
    findings_by_operationalization: list[OperationalizationFindingBranch] = Field(min_length=1)

    @field_validator(
        "dataset_id",
        "case_id",
        "case_family_id",
    )
    @classmethod
    def validate_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("GroundTruth identifiers must be non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_bindings(self) -> "GroundTruth":
        operationalization_ids = [
            bundle.operationalization_id for bundle in self.acceptable_operationalizations
        ]
        if len(operationalization_ids) != len(set(operationalization_ids)):
            raise ValueError("operationalization_id values must be unique within GroundTruth")
        branch_ids = [
            branch.operationalization_id
            for branch in self.findings_by_operationalization
        ]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("each accepted O may have only one findings branch")
        if set(branch_ids) != set(operationalization_ids):
            raise ValueError(
                "findings_by_operationalization must contain exactly one branch per accepted O"
            )
        return self


def _coerce_ground_truth(
    ground_truth: GroundTruth | Mapping[str, Any],
) -> GroundTruth:
    return (
        ground_truth
        if isinstance(ground_truth, GroundTruth)
        else GroundTruth.model_validate(ground_truth)
    )


def _coerce_metadata(
    metadata: CaseConstructionMetadata | Mapping[str, Any],
) -> CaseConstructionMetadata:
    return (
        metadata
        if isinstance(metadata, CaseConstructionMetadata)
        else CaseConstructionMetadata.model_validate(metadata)
    )


def _coerce_evidence_records(
    evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]],
) -> tuple[EvidenceRecord, ...]:
    records = tuple(
        record if isinstance(record, EvidenceRecord) else EvidenceRecord.model_validate(record)
        for record in evidence_records
    )
    evidence_ids = [record.evidence_id for record in records]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence_id values must be unique within evidence_records")
    return records


class GroundTruthValidator:
    """Validate GT structure and its binding to controlled-case metadata."""

    @staticmethod
    def validate(
        ground_truth: GroundTruth | Mapping[str, Any],
        case_metadata: CaseConstructionMetadata | Mapping[str, Any],
        *,
        evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
    ) -> GroundTruth:
        validated = _coerce_ground_truth(ground_truth)
        metadata = _coerce_metadata(case_metadata)

        for field_name in ("dataset_id", "case_id", "case_family_id"):
            gt_value = getattr(validated, field_name)
            metadata_value = getattr(metadata, field_name)
            if gt_value != metadata_value:
                raise ValueError(
                    f"{field_name} mismatch: ground_truth={gt_value!r}, "
                    f"metadata={metadata_value!r}"
                )

        case_type = primary_case_type(metadata)
        if case_type is None:
            raise ValueError(
                "Ground Truth v1 primary validation supports only "
                "O1-F1, O2-F1, O3-F1, and O1-F2"
            )

        principal_dimensions = set(metadata.principal_operationalization_dimensions)
        unresolved_dimensions = set(metadata.unresolved_operationalization_dimensions)
        bundles_by_id = {
            bundle.operationalization_id: bundle
            for bundle in validated.acceptable_operationalizations
        }
        branches_by_id = {
            branch.operationalization_id: branch
            for branch in validated.findings_by_operationalization
        }

        for bundle in validated.acceptable_operationalizations:
            dimensions = {decision.dimension for decision in bundle.decisions}
            if dimensions != principal_dimensions:
                missing = sorted(item.value for item in principal_dimensions - dimensions)
                extra = sorted(item.value for item in dimensions - principal_dimensions)
                raise ValueError(
                    f"operationalization {bundle.operationalization_id!r} dimensions do not "
                    f"match metadata (missing={missing!r}, extra={extra!r})"
                )
            if not branches_by_id.get(bundle.operationalization_id):
                raise ValueError(
                    f"operationalization {bundle.operationalization_id!r} has no G(O) branch"
                )

        if case_type in {"O1-F1", "O1-F2"} and len(bundles_by_id) != 1:
            raise ValueError(f"{case_type} requires exactly one accepted operationalization")

        if case_type == "O2-F1":
            resolved_dimensions = principal_dimensions - unresolved_dimensions
            resolved_statements: dict[OperationalizationDimension, str | None] = {
                dimension: None for dimension in resolved_dimensions
            }
            for bundle in validated.acceptable_operationalizations:
                statements = {
                    decision.dimension: normalize_case_text(decision.statement)
                    for decision in bundle.decisions
                }
                for dimension in resolved_dimensions:
                    statement = statements[dimension]
                    if resolved_statements[dimension] is None:
                        resolved_statements[dimension] = statement
                    elif statement != resolved_statements[dimension]:
                        raise ValueError(
                            "O2 resolved operationalization dimensions must match across "
                            f"all accepted statements: {dimension.value!r}"
                        )

        if case_type.endswith("F1") and not metadata.explicit_finding_requirements:
            raise ValueError("F1 metadata must contain explicit finding requirements")

        for branch in validated.findings_by_operationalization:
            if not any(finding.importance == FindingImportance.CORE for finding in branch.findings):
                raise ValueError(
                    f"G(O) branch {branch.operationalization_id!r} must contain a core finding"
                )

        if evidence_records is not None:
            records = _coerce_evidence_records(evidence_records)
            known_records = {record.evidence_id: record for record in records}
            for record in records:
                _validate_evidence_record_identity(record, validated.dataset_id)
            for bundle in validated.acceptable_operationalizations:
                for evidence_id in bundle.evidence_ids:
                    record = known_records.get(evidence_id)
                    if record is None:
                        raise ValueError(
                            f"operationalization evidence {evidence_id!r} does not exist"
                        )
                    if record.evidence_type != "operationalization":
                        raise ValueError(
                            f"evidence {evidence_id!r} must be operationalization evidence"
                        )
            for branch in validated.findings_by_operationalization:
                for finding in branch.findings:
                    for evidence_id in finding.evidence_ids:
                        record = known_records.get(evidence_id)
                        if record is None:
                            raise ValueError(f"finding evidence {evidence_id!r} does not exist")
                        if record.evidence_type != "finding":
                            raise ValueError(
                                f"evidence {evidence_id!r} must be finding evidence"
                            )

        return validated


def _validate_evidence_record_identity(record: EvidenceRecord, dataset_id: str) -> None:
    if record.dataset_id != dataset_id:
        raise ValueError(
            f"evidence {record.evidence_id!r} belongs to dataset {record.dataset_id!r}, "
            f"expected {dataset_id!r}"
        )


def load_ground_truth(
    path: str | Path,
    *,
    case_metadata: CaseConstructionMetadata | Mapping[str, Any] | None = None,
    evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
) -> GroundTruth:
    """Load and structurally validate a ``ground_truth.json`` file."""

    ground_truth_path = Path(path)
    try:
        payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load ground_truth.json: {ground_truth_path}") from exc
    if case_metadata is None:
        return _coerce_ground_truth(payload)
    return GroundTruthValidator.validate(
        payload,
        case_metadata,
        evidence_records=evidence_records,
    )


def save_ground_truth(
    ground_truth: GroundTruth | Mapping[str, Any],
    path: str | Path,
    *,
    case_metadata: CaseConstructionMetadata | Mapping[str, Any] | None = None,
    evidence_records: Iterable[EvidenceRecord | Mapping[str, Any]] | None = None,
    indent: int = 2,
) -> GroundTruth:
    """Validate and write a deterministic ``ground_truth.json`` file."""

    if case_metadata is None:
        validated = _coerce_ground_truth(ground_truth)
    else:
        validated = GroundTruthValidator.validate(
            ground_truth,
            case_metadata,
            evidence_records=evidence_records,
        )
    Path(path).write_text(
        json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return validated


__all__ = [
    "FindingImportance",
    "VerificationSpec",
    "OperationalizationDecision",
    "OperationalizationBundle",
    "ReferenceFinding",
    "OperationalizationFindingBranch",
    "GroundTruth",
    "GroundTruthValidator",
    "load_ground_truth",
    "save_ground_truth",
]
