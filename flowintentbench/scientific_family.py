"""Small, explicit schema for a scientific concept family in the portfolio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .lifecycle import LifecycleState


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PROVISIONAL = "PROVISIONAL"
    INSUFFICIENT = "INSUFFICIENT"


class PortfolioLifecycle(str, Enum):
    """Compatibility projection for older portfolio artifacts.

    ``LifecycleState`` is the normative, full lifecycle used by release and
    evaluation gates. This shorter enum remains loadable for legacy family
    projections. Release code must project it through
    :func:`normative_lifecycle_state` instead of treating it as authority.
    """

    DISCOVERED = "DISCOVERED"
    DATA_SUPPORTED = "DATA_SUPPORTED"
    SCIENTIFICALLY_GROUNDED = "SCIENTIFICALLY_GROUNDED"
    CURATOR_CONFIRMED = "CURATOR_CONFIRMED"
    CONTROLLED_FAMILY_VALIDATED = "CONTROLLED_FAMILY_VALIDATED"
    RELEASE_ELIGIBLE = "RELEASE_ELIGIBLE"


# Portfolio cards intentionally have fewer states than the normative case
# lifecycle. This explicit mapping never invents a normative intermediate
# state such as SCQ_PASS, SRAC, or EVALUATOR_CALIBRATED.
PORTFOLIO_TO_NORMATIVE_LIFECYCLE: dict[PortfolioLifecycle, LifecycleState] = {
    PortfolioLifecycle.DISCOVERED: LifecycleState.DISCOVERED,
    PortfolioLifecycle.DATA_SUPPORTED: LifecycleState.DATA_SUPPORTED,
    PortfolioLifecycle.SCIENTIFICALLY_GROUNDED: LifecycleState.SCIENTIFICALLY_GROUNDED,
    PortfolioLifecycle.CURATOR_CONFIRMED: LifecycleState.GROUNDING_CURATOR_CONFIRMED,
    PortfolioLifecycle.CONTROLLED_FAMILY_VALIDATED: LifecycleState.CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED,
    PortfolioLifecycle.RELEASE_ELIGIBLE: LifecycleState.RELEASE_ELIGIBLE,
}

NORMATIVE_TO_PORTFOLIO_LIFECYCLE: dict[LifecycleState, PortfolioLifecycle] = {
    state: portfolio for portfolio, state in PORTFOLIO_TO_NORMATIVE_LIFECYCLE.items()
}


def normative_lifecycle_state(
    value: LifecycleState | PortfolioLifecycle | str,
) -> LifecycleState:
    """Return the normative :class:`LifecycleState` for a lifecycle value.

    A normative state is returned unchanged.  A legacy ``PortfolioLifecycle``
    value is projected through the explicit mapping above.  Unknown strings
    fail closed rather than being treated as an early or release state.
    """

    if isinstance(value, LifecycleState):
        return value
    if isinstance(value, PortfolioLifecycle):
        return PORTFOLIO_TO_NORMATIVE_LIFECYCLE[value]
    text = str(value).strip().upper()
    try:
        return LifecycleState(text)
    except ValueError:
        try:
            return PORTFOLIO_TO_NORMATIVE_LIFECYCLE[PortfolioLifecycle(text)]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"unknown lifecycle state: {value!r}") from exc


def portfolio_lifecycle_projection(
    value: LifecycleState | PortfolioLifecycle | str,
) -> PortfolioLifecycle:
    """Project a normative state back to the compact portfolio enum.

    Only states represented losslessly by ``PortfolioLifecycle`` are accepted.
    Richer normative states (for example ``SCQ_PASS`` or ``SRAC``) must not be
    silently collapsed to ``RELEASE_ELIGIBLE`` or another misleading state.
    """

    if isinstance(value, PortfolioLifecycle):
        return value
    state = normative_lifecycle_state(value)
    try:
        return NORMATIVE_TO_PORTFOLIO_LIFECYCLE[state]
    except KeyError as exc:
        raise ValueError(
            f"normative lifecycle state {state.value!r} has no lossless "
            "PortfolioLifecycle projection"
        ) from exc


def is_formal_release_eligible(family: Any) -> bool:
    """Return the fail-closed formal ``RELEASE_ELIGIBLE`` predicate.

    This helper accepts either a family dataclass or a mapping.  Formal
    release requires all four independent declarations: normative lifecycle,
    grounded scientific status, accountable curator confirmation, and the
    explicit release status.  Portfolio lifecycle is only interpreted through
    :func:`normative_lifecycle_state`.
    """

    def field(name: str, default: Any = None) -> Any:
        if isinstance(family, Mapping):
            return family.get(name, default)
        return getattr(family, name, default)

    try:
        lifecycle = normative_lifecycle_state(field("lifecycle_status"))
    except (TypeError, ValueError):
        return False

    def enum_text(value: Any) -> str:
        return str(getattr(value, "value", value) or "").strip().upper()

    return (
        lifecycle is LifecycleState.RELEASE_ELIGIBLE
        and enum_text(field("grounding_status")) == GroundingStatus.GROUNDED.value
        and enum_text(field("curator_status")) == "CONFIRMED"
        and enum_text(field("release_status")) == "ELIGIBLE"
    )


def to_normative_lifecycle_state(
    value: LifecycleState | PortfolioLifecycle | str,
) -> LifecycleState:
    """Backward-friendly alias for :func:`normative_lifecycle_state`."""

    return normative_lifecycle_state(value)


def is_formal_release_lifecycle(
    value: LifecycleState | PortfolioLifecycle | str | None,
) -> bool:
    """Return whether an explicit value is the normative release state."""

    if value is None:
        return False
    try:
        return normative_lifecycle_state(value) is LifecycleState.RELEASE_ELIGIBLE
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ScientificConcept:
    """Concept-level identity shared by dataset-specific instantiations."""

    concept_id: str
    scientific_target: str
    dataset_family_ids: tuple[str, ...] = ()
    required_capabilities: Mapping[str, Any] = field(default_factory=dict)
    optional_capabilities: Mapping[str, Any] = field(default_factory=dict)
    irrelevant_capabilities: tuple[str, ...] = ()
    scientific_semantic_capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "scientific_target": self.scientific_target,
            "dataset_family_ids": list(self.dataset_family_ids),
            "required_capabilities": dict(self.required_capabilities),
            "optional_capabilities": dict(self.optional_capabilities),
            "irrelevant_capabilities": list(self.irrelevant_capabilities),
            "scientific_semantic_capabilities": {str(k): dict(v) for k, v in self.scientific_semantic_capabilities.items()},
        }


@dataclass(frozen=True)
class DatasetConceptFamily:
    """A scientific concept instantiated for exactly one dataset."""

    family_id: str
    concept_id: str
    dataset_id: str
    scientific_target: str
    analysis_archetype: str = ""
    principal_dimensions: tuple[str, ...] = ()
    controlled_case_ids: tuple[str, ...] = ()
    grounding_status: GroundingStatus = GroundingStatus.INSUFFICIENT
    curator_status: str = "PENDING"
    release_status: str = "COMPUTED"
    lifecycle_status: PortfolioLifecycle = PortfolioLifecycle.DISCOVERED
    resolved_operationalization_clauses: Mapping[str, Any] = field(default_factory=dict)
    finding_requirement_contract: Mapping[str, Any] = field(default_factory=dict)
    model_visible_scientific_target: Mapping[str, Any] | None = None
    required_capabilities: Mapping[str, Any] = field(default_factory=dict)
    optional_capabilities: Mapping[str, Any] = field(default_factory=dict)
    irrelevant_capabilities: tuple[str, ...] = ()
    scientific_semantic_capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def normative_lifecycle_status(self) -> LifecycleState:
        """Return the normative lifecycle view of this portfolio family."""

        return normative_lifecycle_state(self.lifecycle_status)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["grounding_status"] = self.grounding_status.value
        value["lifecycle_status"] = self.lifecycle_status.value
        value["controlled_case_ids"] = list(self.controlled_case_ids)
        value["irrelevant_capabilities"] = list(self.irrelevant_capabilities)
        value["required_capabilities"] = dict(self.required_capabilities)
        value["optional_capabilities"] = dict(self.optional_capabilities)
        value["scientific_semantic_capabilities"] = {str(k): dict(v) for k, v in self.scientific_semantic_capabilities.items()}
        return value

    def contains_only_dataset_cases(self, case_records: Sequence[Mapping[str, Any]]) -> bool:
        allowed = set(self.controlled_case_ids)
        return all(
            str(record.get("case_id")) not in allowed or str(record.get("dataset_id")) == self.dataset_id
            for record in case_records
        )


@dataclass(frozen=True)
class ScientificConceptFamily:
    family_id: str
    dataset_id: str
    scientific_target: str
    question_template: str
    rationale: str
    evidence_bundle: Mapping[str, Any] = field(default_factory=dict)
    analysis_archetype: str = ""
    principal_dimensions: tuple[str, ...] = ()
    baseline_resolved_clauses: tuple[str, ...] = ()
    candidate_reference_operationalizations: tuple[str, ...] = ()
    finding_schema: Mapping[str, Any] = field(default_factory=dict)
    grounding_status: GroundingStatus = GroundingStatus.INSUFFICIENT
    curator_status: str = "PENDING"
    release_status: str = "NOT_ELIGIBLE"
    lifecycle_status: PortfolioLifecycle = PortfolioLifecycle.DISCOVERED
    concept_id: str = ""
    controlled_case_ids: tuple[str, ...] = ()
    resolved_operationalization_clauses: Mapping[str, Any] = field(default_factory=dict)
    finding_requirement_contract: Mapping[str, Any] = field(default_factory=dict)
    model_visible_scientific_target: Mapping[str, Any] | None = None
    required_capabilities: Mapping[str, Any] = field(default_factory=dict)
    optional_capabilities: Mapping[str, Any] = field(default_factory=dict)
    irrelevant_capabilities: tuple[str, ...] = ()
    scientific_semantic_capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def normative_lifecycle_status(self) -> LifecycleState:
        """Return the normative lifecycle view of this portfolio family."""

        return normative_lifecycle_state(self.lifecycle_status)

    @classmethod
    def from_design_card(cls, card: Mapping[str, Any]) -> "ScientificConceptFamily":
        status = str(card.get("screening_status", "PROVISIONAL")).upper()
        grounding = GroundingStatus.GROUNDED if status == "GROUNDED" else GroundingStatus.PROVISIONAL if status in {"STRONG_CANDIDATE", "PROVISIONAL"} else GroundingStatus.INSUFFICIENT
        lifecycle = PortfolioLifecycle.SCIENTIFICALLY_GROUNDED if grounding is GroundingStatus.GROUNDED else PortfolioLifecycle.DATA_SUPPORTED
        return cls(
            family_id=str(card.get("family_id") or card.get("concept_id")),
            dataset_id=str(card.get("dataset_id")),
            scientific_target=str(card.get("scientific_question_family") or card.get("scientific_concept") or ""),
            question_template=str(card.get("scientific_question_family") or ""),
            rationale=str(card.get("why_scientifically_meaningful") or card.get("scientific_coherence") or ""),
            evidence_bundle={
                "evidence_ids": list(card.get("evidence_ids", [])),
                "source_literature_grounding": card.get("source_literature_grounding"),
                "observable_data_requirements": card.get("observable_data_requirements"),
                "available_dataset_support": card.get("available_dataset_support"),
                "grounding_assessment": card.get("grounding_assessment"),
            },
            analysis_archetype=str(card.get("analysis_archetype") or {
                "kitchen_turbulence_activity": "EXTREMUM_FEATURE",
                "kitchen_concentration_heterogeneity": "FIELD_HETEROGENEITY",
                "combustor_density_features": "GRADIENT_FEATURE",
            }.get(str(card.get("concept_id")), "EXTREMUM_FEATURE")),
            principal_dimensions=tuple(str(x) for x in card.get("principal_dimensions", [])),
            baseline_resolved_clauses=tuple(str(x) for x in card.get("baseline_resolved_clauses", [])),
            candidate_reference_operationalizations=tuple(str(x) for x in card.get("candidate_accepted_o_branches", card.get("accepted_complete_operationalizations", []))),
            finding_schema={"f1_goal": card.get("f1_finding_goal"), "f2_goal": card.get("f2_finding_goal")},
            grounding_status=grounding,
            curator_status="PENDING_EXPERT_REVIEW",
            release_status="NOT_ELIGIBLE",
            lifecycle_status=lifecycle,
            concept_id=str(card.get("concept_id") or card.get("family_id") or ""),
            resolved_operationalization_clauses=dict(card.get("resolved_operationalization_clauses") or {}),
            finding_requirement_contract=dict(card.get("finding_requirement_contract") or {}),
            model_visible_scientific_target=card.get("model_visible_scientific_target"),
            required_capabilities=dict(card.get("required_capabilities") or card.get("capability_requirements") or {}),
            optional_capabilities=dict(card.get("optional_capabilities") or {}),
            irrelevant_capabilities=tuple(str(x) for x in card.get("irrelevant_capabilities", ()) or ()),
            scientific_semantic_capabilities={str(k): dict(v) for k, v in (card.get("scientific_semantic_capabilities", {}) or {}).items() if isinstance(v, Mapping)},
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["grounding_status"] = self.grounding_status.value
        value["lifecycle_status"] = self.lifecycle_status.value
        value["irrelevant_capabilities"] = list(self.irrelevant_capabilities)
        value["required_capabilities"] = dict(self.required_capabilities)
        value["optional_capabilities"] = dict(self.optional_capabilities)
        value["scientific_semantic_capabilities"] = {str(k): dict(v) for k, v in self.scientific_semantic_capabilities.items()}
        return value

    def release_eligible(self) -> bool:
        return is_formal_release_eligible(self)


__all__ = [
    "GroundingStatus",
    "NORMATIVE_TO_PORTFOLIO_LIFECYCLE",
    "PORTFOLIO_TO_NORMATIVE_LIFECYCLE",
    "PortfolioLifecycle",
    "is_formal_release_eligible",
    "is_formal_release_lifecycle",
    "normative_lifecycle_state",
    "portfolio_lifecycle_projection",
    "ScientificConcept",
    "DatasetConceptFamily",
    "ScientificConceptFamily",
    "to_normative_lifecycle_state",
]
