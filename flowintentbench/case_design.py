"""Internal metadata for controlled case construction.

The models in this module are construction-time bookkeeping only.  They are
deliberately separate from :class:`BenchmarkCaseInput`, which is the sole
model-facing input contract.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .context import ContextSelection


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CaseDesignModel(BaseModel):
    """Closed, assignment-validated base for construction metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class OperationalizationResponsibility(str, Enum):
    USER_SPECIFIED = "user_specified"
    PARTIALLY_SPECIFIED = "partially_specified"
    MODEL_SELECTED = "model_selected"


class FindingOpenness(str, Enum):
    BOUNDED = "bounded"
    OPEN = "open"


class OperationalizationDimension(str, Enum):
    FEATURE_DEFINITION = "feature_definition"
    CRITERION = "criterion"
    PARAMETER = "parameter"
    PROPERTY_MEASURE = "property_measure"
    AGGREGATION_OR_REPRESENTATION = "aggregation_or_representation"
    COMPARISON_OR_NORMALIZATION = "comparison_or_normalization"
    INTERPRETATION_RULE = "interpretation_rule"


class MethodConstraintCategory(str, Enum):
    FEATURE_DEFINITION = "feature_definition"
    CRITERION = "criterion"
    PROPERTY_MEASURE = "property_measure"
    PARAMETER = "parameter"
    ANALYSIS_PROCEDURE = "analysis_procedure"
    INTERPRETATION_RULE = "interpretation_rule"
    OTHER = "other"


class FindingRequirementCategory(str, Enum):
    EXISTENCE_OR_IDENTITY = "existence_or_identity"
    QUANTITY = "quantity"
    LOCATION = "location"
    PROPERTY = "property"
    RANKING = "ranking"
    COMPARISON = "comparison"
    RELATION = "relation"
    CHARACTERIZATION = "characterization"
    OTHER = "other"


class ExplicitQuestionConstraint(CaseDesignModel):
    statement: NonEmptyString
    question_fragment: NonEmptyString


class ExplicitMethodConstraint(CaseDesignModel):
    category: MethodConstraintCategory
    statement: NonEmptyString
    question_fragment: NonEmptyString


class ExplicitFindingRequirement(CaseDesignModel):
    category: FindingRequirementCategory
    statement: NonEmptyString
    question_fragment: NonEmptyString


def normalize_case_text(value: str) -> str:
    """Return the frozen text form used for grounding and canonical identity."""

    value = unicodedata.normalize("NFC", value)
    return " ".join(value.strip().split()).casefold()


def _generic_keys(items: list[ExplicitQuestionConstraint]) -> tuple[str, ...]:
    return tuple(sorted(normalize_case_text(item.statement) for item in items))


def _method_keys(items: list[ExplicitMethodConstraint]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((item.category.value, normalize_case_text(item.statement)) for item in items)
    )


def _finding_keys(items: list[ExplicitFindingRequirement]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted((item.category.value, normalize_case_text(item.statement)) for item in items)
    )


def _dimension_keys(items: list[OperationalizationDimension]) -> tuple[str, ...]:
    return tuple(sorted(item.value for item in items))


def _reject_duplicate_keys(
    items: list[Any],
    key_builder: Any,
    label: str,
) -> None:
    seen: set[Any] = set()
    for item in items:
        key = key_builder(item)
        if key in seen:
            raise ValueError(f"duplicate {label} entry: {key!r}")
        seen.add(key)


class CaseConstructionMetadata(CaseDesignModel):
    """Authored semantic bookkeeping for one controlled benchmark case."""

    case_id: NonEmptyString
    case_family_id: NonEmptyString
    dataset_id: NonEmptyString

    scientific_target: NonEmptyString
    finding_goal: NonEmptyString

    operationalization_responsibility: OperationalizationResponsibility
    finding_openness: FindingOpenness

    principal_operationalization_dimensions: list[OperationalizationDimension]
    unresolved_operationalization_dimensions: list[OperationalizationDimension]

    scope_constraints: list[ExplicitQuestionConstraint] = Field(default_factory=list)
    condition_constraints: list[ExplicitQuestionConstraint] = Field(default_factory=list)
    selection_constraints: list[ExplicitQuestionConstraint] = Field(default_factory=list)

    explicit_method_constraints: list[ExplicitMethodConstraint] = Field(default_factory=list)
    explicit_finding_requirements: list[ExplicitFindingRequirement] = Field(default_factory=list)

    # These two construction-time contracts are authored from the same case
    # specification as the question and GT.  They are optional for legacy
    # synthetic/unit fixtures, but production candidate artifacts must carry
    # both contracts explicitly (the SCQ production path fails closed when
    # either is absent).
    responsibility_contract: dict[str, Any] | None = None
    evaluation_representability_contract: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_structure(self) -> "CaseConstructionMetadata":
        _reject_duplicate_keys(
            self.principal_operationalization_dimensions,
            lambda item: item.value,
            "principal operationalization dimension",
        )
        _reject_duplicate_keys(
            self.unresolved_operationalization_dimensions,
            lambda item: item.value,
            "unresolved operationalization dimension",
        )

        principal_dimensions = set(self.principal_operationalization_dimensions)
        unresolved_dimensions = set(self.unresolved_operationalization_dimensions)
        if not principal_dimensions:
            raise ValueError("principal_operationalization_dimensions must be non-empty")
        if not unresolved_dimensions <= principal_dimensions:
            raise ValueError(
                "unresolved_operationalization_dimensions must be a subset of "
                "principal_operationalization_dimensions"
            )
        if self.operationalization_responsibility == OperationalizationResponsibility.USER_SPECIFIED:
            if unresolved_dimensions:
                raise ValueError("user_specified operationalization requires no unresolved dimensions")
        elif self.operationalization_responsibility == OperationalizationResponsibility.PARTIALLY_SPECIFIED:
            if len(principal_dimensions) < 2:
                raise ValueError(
                    "partially_specified operationalization requires at least two principal dimensions"
                )
            if not unresolved_dimensions:
                raise ValueError(
                    "partially_specified operationalization requires at least one unresolved dimension"
                )
            if len(unresolved_dimensions) >= len(principal_dimensions):
                raise ValueError(
                    "partially_specified unresolved dimensions must be a proper subset"
                )
        elif unresolved_dimensions != principal_dimensions:
            raise ValueError(
                "model_selected operationalization requires all principal dimensions to be unresolved"
            )

        _reject_duplicate_keys(
            self.scope_constraints,
            lambda item: normalize_case_text(item.statement),
            "scope constraint",
        )
        _reject_duplicate_keys(
            self.condition_constraints,
            lambda item: normalize_case_text(item.statement),
            "condition constraint",
        )
        _reject_duplicate_keys(
            self.selection_constraints,
            lambda item: normalize_case_text(item.statement),
            "selection constraint",
        )
        _reject_duplicate_keys(
            self.explicit_method_constraints,
            lambda item: (item.category.value, normalize_case_text(item.statement)),
            "method constraint",
        )
        _reject_duplicate_keys(
            self.explicit_finding_requirements,
            lambda item: (item.category.value, normalize_case_text(item.statement)),
            "finding requirement",
        )

        generic_roles = {
            "scope_constraints": {
                normalize_case_text(item.statement) for item in self.scope_constraints
            },
            "condition_constraints": {
                normalize_case_text(item.statement) for item in self.condition_constraints
            },
            "selection_constraints": {
                normalize_case_text(item.statement) for item in self.selection_constraints
            },
        }
        role_names = tuple(generic_roles)
        for index, left_role in enumerate(role_names):
            for right_role in role_names[index + 1 :]:
                overlap = generic_roles[left_role] & generic_roles[right_role]
                if overlap:
                    raise ValueError(
                        "a constraint statement must have one primary metadata role; "
                        f"it is duplicated in {left_role} and {right_role}: {sorted(overlap)!r}"
                    )

        condition_statements = generic_roles["condition_constraints"]
        method_statements = {
            normalize_case_text(item.statement) for item in self.explicit_method_constraints
        }
        overlap = condition_statements & method_statements
        if overlap:
            raise ValueError(
                "a constraint statement cannot be both a condition constraint and "
                f"an explicit method constraint: {sorted(overlap)!r}"
            )

        if (
            self.operationalization_responsibility == OperationalizationResponsibility.MODEL_SELECTED
            and self.explicit_method_constraints
        ):
            raise ValueError(
                "model_selected operationalization cannot contain explicit_method_constraints"
            )
        if (
            self.operationalization_responsibility
            in {
                OperationalizationResponsibility.USER_SPECIFIED,
                OperationalizationResponsibility.PARTIALLY_SPECIFIED,
            }
            and not self.explicit_method_constraints
        ):
            raise ValueError(
                f"{self.operationalization_responsibility.value} operationalization requires "
                "at least one explicit_method_constraint"
            )
        if self.finding_openness == FindingOpenness.BOUNDED and not self.explicit_finding_requirements:
            raise ValueError("bounded finding requires at least one explicit_finding_requirement")
        return self


def _coerce_metadata(metadata: CaseConstructionMetadata | Mapping[str, Any]) -> CaseConstructionMetadata:
    return metadata if isinstance(metadata, CaseConstructionMetadata) else CaseConstructionMetadata.model_validate(metadata)


def _coerce_selection(selection: ContextSelection | Mapping[str, Any]) -> ContextSelection:
    return selection if isinstance(selection, ContextSelection) else ContextSelection.model_validate(selection)


class CaseConstructionMetadataValidator:
    """Validate authored fragments against the actual scientific question."""

    @staticmethod
    def validate(
        scientific_question: str,
        metadata: CaseConstructionMetadata | Mapping[str, Any],
        *,
        context_selection: ContextSelection | Mapping[str, Any] | None = None,
    ) -> CaseConstructionMetadata:
        if not isinstance(scientific_question, str) or not scientific_question.strip():
            raise ValueError("scientific_question must be a non-empty string")
        record = _coerce_metadata(metadata)
        normalized_question = normalize_case_text(scientific_question)
        for field_name in (
            "scope_constraints",
            "condition_constraints",
            "selection_constraints",
            "explicit_method_constraints",
            "explicit_finding_requirements",
        ):
            for item in getattr(record, field_name):
                fragment = normalize_case_text(item.question_fragment)
                if not fragment:
                    raise ValueError(
                        f"{field_name}.question_fragment must be non-empty after normalization"
                    )
                if fragment not in normalized_question:
                    raise ValueError(
                        f"{field_name}.question_fragment {item.question_fragment!r} "
                        "does not occur in scientific_question"
                    )

        if context_selection is not None:
            selection = _coerce_selection(context_selection)
            _validate_metadata_selection_identity(record, selection)
        return record


def _validate_metadata_selection_identity(
    metadata: CaseConstructionMetadata,
    selection: ContextSelection,
) -> None:
    if metadata.case_id != selection.case_id:
        raise ValueError(
            f"case_id mismatch: metadata={metadata.case_id!r}, selection={selection.case_id!r}"
        )
    if metadata.dataset_id != selection.dataset_id:
        raise ValueError(
            "dataset_id mismatch: "
            f"metadata={metadata.dataset_id!r}, selection={selection.dataset_id!r}"
        )


def primary_case_type(metadata: CaseConstructionMetadata) -> str | None:
    """Return the derived label for a primary controlled combination."""

    mapping = {
        (
            OperationalizationResponsibility.USER_SPECIFIED,
            FindingOpenness.BOUNDED,
        ): "O1-F1",
        (
            OperationalizationResponsibility.PARTIALLY_SPECIFIED,
            FindingOpenness.BOUNDED,
        ): "O2-F1",
        (
            OperationalizationResponsibility.MODEL_SELECTED,
            FindingOpenness.BOUNDED,
        ): "O3-F1",
        (
            OperationalizationResponsibility.USER_SPECIFIED,
            FindingOpenness.OPEN,
        ): "O1-F2",
    }
    return mapping.get(
        (metadata.operationalization_responsibility, metadata.finding_openness)
    )


class ControlledCaseFamilyValidator:
    """Pairwise validator for controlled operationalization/finding variants."""

    @staticmethod
    def validate_pair(
        left_metadata: CaseConstructionMetadata | Mapping[str, Any],
        right_metadata: CaseConstructionMetadata | Mapping[str, Any],
        *,
        axis: Literal["operationalization", "finding"],
        left_context_selection: ContextSelection | Mapping[str, Any] | None = None,
        right_context_selection: ContextSelection | Mapping[str, Any] | None = None,
    ) -> None:
        if axis not in {"operationalization", "finding"}:
            raise ValueError(f"unsupported family comparison axis: {axis!r}")
        left = _coerce_metadata(left_metadata)
        right = _coerce_metadata(right_metadata)

        if (left_context_selection is None) != (right_context_selection is None):
            raise ValueError("both context selections must be supplied or both omitted")
        if left_context_selection is not None and right_context_selection is not None:
            left_selection = _coerce_selection(left_context_selection)
            right_selection = _coerce_selection(right_context_selection)
            _validate_metadata_selection_identity(left, left_selection)
            _validate_metadata_selection_identity(right, right_selection)
            if set(left_selection.include_fact_ids) != set(right_selection.include_fact_ids):
                raise ValueError("context fact set drift between family cases")

        if axis == "operationalization":
            if (
                left.finding_openness != FindingOpenness.BOUNDED
                or right.finding_openness != FindingOpenness.BOUNDED
            ):
                raise ValueError(
                    "operationalization pair requires bounded finding_openness (F1)"
                )
            required = {
                "dataset_id": (left.dataset_id, right.dataset_id),
                "case_family_id": (left.case_family_id, right.case_family_id),
                "finding_openness": (left.finding_openness, right.finding_openness),
                "scientific_target": (
                    normalize_case_text(left.scientific_target),
                    normalize_case_text(right.scientific_target),
                ),
                "finding_goal": (
                    normalize_case_text(left.finding_goal),
                    normalize_case_text(right.finding_goal),
                ),
                "principal_operationalization_dimensions": (
                    _dimension_keys(left.principal_operationalization_dimensions),
                    _dimension_keys(right.principal_operationalization_dimensions),
                ),
                "scope_constraints": (
                    _generic_keys(left.scope_constraints),
                    _generic_keys(right.scope_constraints),
                ),
                "condition_constraints": (
                    _generic_keys(left.condition_constraints),
                    _generic_keys(right.condition_constraints),
                ),
                "selection_constraints": (
                    _generic_keys(left.selection_constraints),
                    _generic_keys(right.selection_constraints),
                ),
                "explicit_finding_requirements": (
                    _finding_keys(left.explicit_finding_requirements),
                    _finding_keys(right.explicit_finding_requirements),
                ),
            }
            if left.operationalization_responsibility == right.operationalization_responsibility:
                raise ValueError(
                    "operationalization pair requires different operationalization_responsibility"
                )
        else:
            for side, metadata in (("left", left), ("right", right)):
                if metadata.operationalization_responsibility != OperationalizationResponsibility.USER_SPECIFIED:
                    raise ValueError(
                        "finding pair drift in operationalization_responsibility: "
                        f"{side} metadata must use user_specified"
                    )
                if metadata.unresolved_operationalization_dimensions:
                    raise ValueError(
                        f"finding pair requires {side} unresolved_operationalization_dimensions to be empty"
                    )
            required = {
                "dataset_id": (left.dataset_id, right.dataset_id),
                "case_family_id": (left.case_family_id, right.case_family_id),
                "operationalization_responsibility": (
                    left.operationalization_responsibility,
                    right.operationalization_responsibility,
                ),
                "scientific_target": (
                    normalize_case_text(left.scientific_target),
                    normalize_case_text(right.scientific_target),
                ),
                "finding_goal": (
                    normalize_case_text(left.finding_goal),
                    normalize_case_text(right.finding_goal),
                ),
                "principal_operationalization_dimensions": (
                    _dimension_keys(left.principal_operationalization_dimensions),
                    _dimension_keys(right.principal_operationalization_dimensions),
                ),
                "scope_constraints": (
                    _generic_keys(left.scope_constraints),
                    _generic_keys(right.scope_constraints),
                ),
                "condition_constraints": (
                    _generic_keys(left.condition_constraints),
                    _generic_keys(right.condition_constraints),
                ),
                "selection_constraints": (
                    _generic_keys(left.selection_constraints),
                    _generic_keys(right.selection_constraints),
                ),
                "explicit_method_constraints": (
                    _method_keys(left.explicit_method_constraints),
                    _method_keys(right.explicit_method_constraints),
                ),
            }
            if left.finding_openness == right.finding_openness:
                raise ValueError("finding pair requires different finding_openness")

        drift = [field_name for field_name, (left_value, right_value) in required.items() if left_value != right_value]
        if drift:
            raise ValueError(f"{axis} family pair drift in: {', '.join(drift)}")


__all__ = [
    "CaseConstructionMetadata",
    "CaseConstructionMetadataValidator",
    "ControlledCaseFamilyValidator",
    "ExplicitFindingRequirement",
    "ExplicitMethodConstraint",
    "ExplicitQuestionConstraint",
    "FindingOpenness",
    "FindingRequirementCategory",
    "MethodConstraintCategory",
    "OperationalizationResponsibility",
    "OperationalizationDimension",
    "normalize_case_text",
    "primary_case_type",
]
