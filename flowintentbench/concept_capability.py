"""Dataset x concept capability cells for scientific portfolio construction.

The map is intentionally an evidence ledger, not a numerical-computability
catalogue.  A finite array can be data-observable while the corresponding
scientific concept remains unsupported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOWED_DECISIONS = frozenset({
    "SUPPORTED_FOR_GROUNDING_REVIEW",
    "EVIDENCE_GAP",
    "DATA_GAP",
    "SCIENTIFICALLY_UNJUSTIFIED",
    "REJECTED",
    "ALREADY_MATERIALIZED",
})


@dataclass(frozen=True)
class DatasetConceptCapability:
    dataset_id: str
    concept_id: str
    scientific_target: str
    candidate_analysis_archetype: str
    scientific_entity_type: str
    data_observability_status: str
    context_support_status: str
    geometry_support_status: str
    boundary_support_status: str
    reference_direction_support_status: str
    temporal_support_status: str
    scientific_grounding_status: str
    operationalization_plurality_status: str
    finding_verifiability_status: str
    evidence_ids: tuple[str, ...] = ()
    major_blockers: tuple[str, ...] = ()
    decision: str = "EVIDENCE_GAP"
    required_capabilities: Mapping[str, Any] = field(default_factory=dict)
    optional_capabilities: Mapping[str, Any] = field(default_factory=dict)
    irrelevant_capabilities: tuple[str, ...] = ()
    capability_evidence: Mapping[str, str] = field(default_factory=dict)
    scientific_semantic_capabilities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    temporal_data_status: str = "UNKNOWN"
    missing_required_capabilities: tuple[str, ...] = ()
    unknown_required_capabilities: tuple[str, ...] = ()
    missing_optional_capabilities: tuple[str, ...] = ()
    capability_status: str = "REQUIRED_CAPABILITY_UNKNOWN"

    def __post_init__(self) -> None:
        for name in ("dataset_id", "concept_id", "scientific_target", "candidate_analysis_archetype", "scientific_entity_type"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.decision not in ALLOWED_DECISIONS:
            raise ValueError(f"unsupported capability decision: {self.decision}")
        object.__setattr__(self, "evidence_ids", tuple(str(x) for x in self.evidence_ids))
        object.__setattr__(self, "major_blockers", tuple(str(x) for x in self.major_blockers))
        object.__setattr__(self, "irrelevant_capabilities", tuple(str(x) for x in self.irrelevant_capabilities))
        object.__setattr__(self, "missing_required_capabilities", tuple(str(x) for x in self.missing_required_capabilities))
        object.__setattr__(self, "unknown_required_capabilities", tuple(str(x) for x in self.unknown_required_capabilities))
        object.__setattr__(self, "missing_optional_capabilities", tuple(str(x) for x in self.missing_optional_capabilities))
        if self.temporal_data_status not in {"SINGLE_SNAPSHOT_ONLY", "TIME_RESOLVED_SUPPORTED", "UNKNOWN"}:
            raise ValueError(f"invalid temporal_data_status: {self.temporal_data_status}")
        if self.capability_status not in {"REQUIREMENTS_SATISFIED", "REQUIRED_CAPABILITY_MISSING", "REQUIRED_CAPABILITY_UNKNOWN"}:
            raise ValueError(f"invalid capability_status: {self.capability_status}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        value["major_blockers"] = list(self.major_blockers)
        value["irrelevant_capabilities"] = list(self.irrelevant_capabilities)
        value["missing_required_capabilities"] = list(self.missing_required_capabilities)
        value["unknown_required_capabilities"] = list(self.unknown_required_capabilities)
        value["missing_optional_capabilities"] = list(self.missing_optional_capabilities)
        value["required_capabilities"] = dict(self.required_capabilities)
        value["optional_capabilities"] = dict(self.optional_capabilities)
        value["capability_evidence"] = dict(self.capability_evidence)
        value["scientific_semantic_capabilities"] = {str(k): dict(v) for k, v in self.scientific_semantic_capabilities.items()}
        return value


@dataclass(frozen=True)
class DatasetConceptCapabilityMap:
    cells: tuple[DatasetConceptCapability, ...] = ()
    status: str = "PROVISIONAL"

    def __post_init__(self) -> None:
        keys = [(item.dataset_id, item.concept_id) for item in self.cells]
        if len(keys) != len(set(keys)):
            raise ValueError("Dataset x Concept cells must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_vocabulary": sorted(ALLOWED_DECISIONS),
            "cells": [item.to_dict() for item in self.cells],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetConceptCapabilityMap":
        cells = tuple(DatasetConceptCapability(**dict(item)) for item in value.get("cells", ()) if isinstance(item, Mapping))
        return cls(cells=cells, status=str(value.get("status", "PROVISIONAL")))


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _profile_status(profile: Mapping[str, Any] | None, field: str) -> str:
    if not profile:
        return "UNKNOWN"
    value = profile.get(field)
    if isinstance(value, str) and value:
        return "SUPPORTED" if value not in {"No approved boundary fact.", "No approved reference-direction fact.", "No separately reviewed geometry asset."} else "UNSUPPORTED"
    if isinstance(value, list):
        if value and all(isinstance(item, str) and item.strip() for item in value):
            return "SUPPORTED"
        approved = [item for item in value if isinstance(item, Mapping) and str(item.get("review_status", item.get("status", ""))).casefold() in {"approved", "reviewed", "resolved", "supported"}]
        return "SUPPORTED" if approved else "UNSUPPORTED"
    return "UNSUPPORTED" if field in {"geometry_information", "boundary_information", "reference_direction_information"} else "UNKNOWN"


def _legacy_capability_requirements() -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Historical controlled-pilot adapter; never used for formal candidates.

    The adapter has no scientific policy keyed by a concept identifier.  Its
    scope is selected by the historical family record, while formal families
    provide their own requirement declarations.
    """

    return ({"SPATIAL_COORDINATES": True}, {}, ("TIME_SEQUENCE",))


def _declared_capability_requirements(declaration: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Read requirements from a ScientificConcept/DatasetConceptFamily record."""

    value = declaration if isinstance(declaration, Mapping) else {}
    # An absent declaration is not evidence that coordinates alone suffice.
    required = value.get("required_capabilities", value.get("capability_requirements", {"EXPLICIT_REQUIREMENTS_DECLARATION": True}))
    optional = value.get("optional_capabilities", {})
    irrelevant = value.get("irrelevant_capabilities", ())
    return (
        dict(required) if isinstance(required, Mapping) else {str(item): True for item in required or ()},
        dict(optional) if isinstance(optional, Mapping) else {str(item): True for item in optional or ()},
        tuple(str(item) for item in irrelevant or ()),
    )


def evaluate_capability_requirements(
    required_capabilities: Mapping[str, Any],
    capability_evidence: Mapping[str, str],
    *,
    optional_capabilities: Mapping[str, Any] | None = None,
    irrelevant_capabilities: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate only declared requirements using semantic statuses."""

    evidence = {str(key): str(value).strip().upper() for key, value in capability_evidence.items()}
    missing_required = [key for key in required_capabilities if evidence.get(key) in {"MISSING", "UNSUPPORTED"}]
    unknown_required = [key for key in required_capabilities if evidence.get(key) not in {"SUPPORTED", "MISSING", "UNSUPPORTED"}]
    missing_optional = [key for key in (optional_capabilities or {}) if evidence.get(key) in {"MISSING", "UNSUPPORTED"}]
    if missing_required:
        status = "REQUIRED_CAPABILITY_MISSING"
    elif unknown_required:
        status = "REQUIRED_CAPABILITY_UNKNOWN"
    else:
        status = "REQUIREMENTS_SATISFIED"
    return {
        "required_capabilities": dict(required_capabilities),
        "capability_evidence": evidence,
        "missing_required_capabilities": missing_required,
        "unknown_required_capabilities": unknown_required,
        "missing_optional_capabilities": missing_optional,
        "irrelevant_capabilities": list(irrelevant_capabilities),
        "capability_status": status,
    }


_SCIENTIFIC_SEMANTIC_CAPABILITIES = frozenset({
    "FIELD_COMPARABILITY", "SPECIES_SEMANTICS", "VOLUME_WEIGHTING_JUSTIFICATION",
    "OBSERVABLE_INTERPRETATION", "REFERENCE_DIRECTION_MEANING",
})


def _declared_semantic_evidence(profile: Mapping[str, Any]) -> dict[str, str]:
    """Read scientific semantic support only from an explicit evidence record.

    A declaration is support-bearing only when it names both the claims and
    canonical sources that justify it.  A bare ``SUPPORTED`` label is kept
    as UNKNOWN so an array name or a computable expression cannot promote a
    scientific interpretation.
    """

    declared = profile.get("scientific_semantic_capabilities", {})
    if not isinstance(declared, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw in declared.items():
        name = str(key)
        if isinstance(raw, Mapping):
            status = str(raw.get("support_status", raw.get("status", "UNKNOWN"))).upper()
            claims = raw.get("supporting_claim_ids", raw.get("supported_claim_ids", ()))
            sources = raw.get("supporting_source_ids", raw.get("source_ids", ()))
            claims_ok = isinstance(claims, Sequence) and not isinstance(claims, (str, bytes)) and bool(tuple(claims))
            sources_ok = isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)) and bool(tuple(sources))
            if status == "SUPPORTED" and not (claims_ok and sources_ok):
                status = "UNKNOWN"
            result[name] = status
        elif isinstance(raw, str):
            result[name] = "UNKNOWN" if raw.upper() == "SUPPORTED" else raw.upper()
    return result


def _dataset_capability_evidence(profile: Mapping[str, Any]) -> dict[str, str]:
    variables = profile.get("available_variables", []) or []
    names = {str(item.get("name", "")) for item in variables if isinstance(item, Mapping)}
    semantics = {str(item.get("physical_quantity", "")).casefold() for item in variables if isinstance(item, Mapping)}
    evidence = {
        "SPATIAL_COORDINATES": "SUPPORTED" if profile.get("grid_type") else "UNKNOWN",
        "VECTOR_FIELD": "SUPPORTED" if any("vector" in str(item).casefold() or "velocity" in str(item).casefold() for item in names | semantics) else "MISSING",
        "SCALAR_FIELD_SEMANTICS": "SUPPORTED" if any(item and item not in {"none", "unknown"} for item in semantics) else "UNKNOWN",
        # Scientific semantic support must be explicitly authored in the
        # profile; array names and computability are insufficient evidence.
        "FIELD_COMPARABILITY": "UNKNOWN",
        "VOLUME_WEIGHTING_JUSTIFICATION": "UNKNOWN",
        "GEOMETRY": _profile_status(profile, "geometry_information"),
        "BOUNDARY_MAPPING": _profile_status(profile, "boundary_information"),
        "REFERENCE_DIRECTION": _profile_status(profile, "reference_direction_information"),
        "WALL_GEOMETRY": "SUPPORTED" if _profile_status(profile, "boundary_information") == "SUPPORTED" and "wall" in str(profile.get("geometry_information", "")).casefold() else "UNSUPPORTED",
        "CROSS_SECTION_DEFINITION": "UNKNOWN",
        # Scientific semantics are never inferred from a field prefix or
        # from the fact that a derivative can be computed numerically.
        "SPECIES_SEMANTICS": "UNKNOWN",
        "DERIVABLE_SPATIAL_GRADIENT": "UNKNOWN",
        "TIME_SEQUENCE": "SUPPORTED" if str(profile.get("temporal_support", "")).casefold() not in {"single_snapshot", ""} else "UNSUPPORTED",
        "TIME_METADATA": "SUPPORTED" if str(profile.get("temporal_support", "")).casefold() not in {"single_snapshot", ""} else "MISSING",
    }
    # Array presence is a data observation only.  Semantic field roles need a
    # profile declaration with supporting claims and sources; absent arrays
    # remain MISSING while present-but-unsupported semantics remain UNKNOWN.
    for token in ("ke", "ep", "Density"):
        evidence[f"SCALAR_FIELD_SEMANTICS:{token}"] = "MISSING" if token not in names else "UNKNOWN"
    derived = profile.get("derived_quantities_that_are_physically_supported", ()) or ()
    if any("gradient" in str(item).casefold() or "derivative" in str(item).casefold() for item in derived):
        evidence["DERIVABLE_SPATIAL_GRADIENT"] = "SUPPORTED"
    evidence.update(_declared_semantic_evidence(profile))
    return evidence


def build_dataset_concept_capability_map(repository_root: str | Path) -> DatasetConceptCapabilityMap:
    """Build the sparse, evidence-first map from existing authored inputs."""

    root = Path(repository_root)
    screening = _read(root / "artifacts/reference/concept_expansion_phase1/concept_screening.json", {})
    entries = screening.get("entries", []) if isinstance(screening, Mapping) else []
    entry_by_key = {(str(item.get("dataset_id")), str(item.get("concept_id"))): item for item in entries if isinstance(item, Mapping)}
    profiles_payload = _read(root / "artifacts/reference/concept_expansion_phase1/dataset_capability_profiles.json", {})
    profiles = {str(item.get("dataset_id")): item for item in profiles_payload.get("profiles", []) if isinstance(item, Mapping)}
    specs = {
        "kitchen_turbulence_activity": ("turbulence-activity hotspot", "EXTREMUM_FEATURE", "SPATIAL_REGION"),
        "kitchen_concentration_heterogeneity": ("spatial concentration heterogeneity", "FIELD_HETEROGENEITY", "FIELD"),
        "combustor_density_features": ("prominent stored-density feature", "GRADIENT_FEATURE", "POINT_FEATURE"),
        "profile_or_wall_shear_concept": ("profile or wall-flow comparison", "PROFILE_COMPARISON", "PROFILE"),
        "separation_or_wake_concept": ("wake or separation structure", "CONNECTED_OR_COHERENT_STRUCTURE", "COHERENT_STRUCTURE"),
        "post_wake_concept": ("post wake structure", "TOPOLOGICAL_OR_FLOW_STRUCTURE", "COHERENT_STRUCTURE"),
        "thermal_or_mixture_concept": ("thermal or mixture transport", "TRANSPORT_OR_DISTRIBUTION", "FIELD"),
        "new_scalar_concept": ("new scalar scientific concept", "EXTREMUM_FEATURE", "POINT_FEATURE"),
    }
    cells: list[DatasetConceptCapability] = []
    # Materialized historical high-speed cases are tracked explicitly, but are
    # not evidence that a new concept is grounded.
    historical_datasets = ("Blunt_Fin", "Carotid", "Combustor", "FireFlow", "Kitchen", "NASA_LOx_Post", "Office")
    for dataset_id in historical_datasets:
        required, optional, irrelevant = _legacy_capability_requirements()
        cap_evidence = {"SPATIAL_COORDINATES": "SUPPORTED", "TIME_SEQUENCE": "UNSUPPORTED"}
        cap_result = evaluate_capability_requirements(required, cap_evidence, optional_capabilities=optional, irrelevant_capabilities=irrelevant)
        cells.append(DatasetConceptCapability(
            dataset_id=dataset_id, concept_id="high_speed_region",
            scientific_target="high-speed flow region", candidate_analysis_archetype="THRESHOLD_REGION",
            scientific_entity_type="SPATIAL_REGION", data_observability_status="SUPPORTED",
            context_support_status="SUPPORTED", geometry_support_status="PARTIAL",
            boundary_support_status="PARTIAL", reference_direction_support_status="PARTIAL",
            temporal_support_status="SINGLE_SNAPSHOT_ONLY", temporal_data_status="SINGLE_SNAPSHOT_ONLY", scientific_grounding_status="PROVISIONAL",
            operationalization_plurality_status="MATERIALIZED", finding_verifiability_status="SUPPORTED",
            evidence_ids=("historical_28_case_manifest",), decision="ALREADY_MATERIALIZED",
            **cap_result,
        ))
    # Candidate cells come from the authored screening matrix.  Unsupported
    # rows remain sparse and are never promoted to DATA_SUPPORTED.
    for (dataset_id, concept_id), item in sorted(entry_by_key.items()):
        target, archetype, entity = specs.get(concept_id, (concept_id, "UNSPECIFIED", "SCIENTIFIC_ENTITY"))
        profile = profiles.get(dataset_id, {})
        data_text = str(item.get("data_feasibility", ""))
        context_text = str(item.get("context_feasibility", ""))
        grounding_text = str(item.get("scientific_grounding", ""))
        plurality_text = str(item.get("operationalization_feasibility", ""))
        status = str(item.get("status", "NOT_SUPPORTED"))
        # Decisions are derived from authored screening evidence, never from
        # a concept identifier lookup in the formal evaluator.
        grounding = "PROVISIONAL" if "FAIL" not in grounding_text and grounding_text.strip() else "EVIDENCE_GAP"
        plurality = "SUPPORTED" if "PASS" in plurality_text else "NOT_ASSESSED"
        decision = "SUPPORTED_FOR_GROUNDING_REVIEW" if status in {"STRONG_CANDIDATE", "PROVISIONAL"} and "FAIL" not in data_text and "FAIL" not in context_text else "EVIDENCE_GAP"
        blockers: list[str] = []
        if "FAIL" in context_text or "No approved" in str(profile.get("boundary_information", "")):
            blockers.append("dataset context or boundary evidence is not reviewed")
        if "FAIL" in grounding_text or grounding == "EVIDENCE_GAP":
            blockers.append(str(item.get("major_ambiguity") or "scientific grounding remains unresolved"))
        if "NOT_ASSESSED" in plurality:
            blockers.append("no scientifically defensible operationalization plurality established")
        # Formal candidates consume declarations authored in the concept card
        # or dataset-family record.  No formal decision is keyed by concept ID.
        card = _read(root / "artifacts/reference/concept_expansion_phase1/concept_design_cards" / f"{concept_id}.json", {})
        declaration = card if isinstance(card, Mapping) else {}
        required, optional, irrelevant = _declared_capability_requirements(declaration)
        cap_result = evaluate_capability_requirements(
            required,
            _dataset_capability_evidence(profile),
            optional_capabilities=optional,
            irrelevant_capabilities=irrelevant,
        )
        blockers.extend(
            f"required capability missing: {value}"
            for value in cap_result["missing_required_capabilities"]
        )
        profile_evidence = tuple(
            str(source.get("source_id"))
            for source in (profile.get("source_evidence", []) or ())
            if isinstance(source, Mapping) and source.get("source_id")
        )
        authored_evidence = tuple(
            str(value) for value in str(item.get("scientific_evidence", "")).split(", ")
            if value and not value.lower().startswith("no accepted") and not value.lower().startswith("no ")
        )
        cells.append(DatasetConceptCapability(
            dataset_id=dataset_id, concept_id=concept_id,
            scientific_target=target, candidate_analysis_archetype=archetype, scientific_entity_type=entity,
            data_observability_status="SUPPORTED" if "PASS" in data_text else "PARTIAL",
            context_support_status="SUPPORTED" if "PASS" in context_text else "UNSUPPORTED",
            geometry_support_status=_profile_status(profile, "geometry_information"),
            boundary_support_status=_profile_status(profile, "boundary_information"),
            reference_direction_support_status=_profile_status(profile, "reference_direction_information"),
            temporal_support_status=("SINGLE_SNAPSHOT_ONLY" if str(profile.get("temporal_support", "")).casefold() == "single_snapshot" else "TIME_RESOLVED_SUPPORTED" if str(profile.get("temporal_support", "")).strip() else "UNKNOWN"),
            temporal_data_status=("SINGLE_SNAPSHOT_ONLY" if str(profile.get("temporal_support", "")).casefold() == "single_snapshot" else "TIME_RESOLVED_SUPPORTED" if str(profile.get("temporal_support", "")).strip() else "UNKNOWN"),
            scientific_grounding_status=grounding,
            operationalization_plurality_status=plurality,
            finding_verifiability_status="SUPPORTED" if status in {"STRONG_CANDIDATE", "PROVISIONAL"} else "NOT_ASSESSED",
            evidence_ids=profile_evidence or authored_evidence,
            major_blockers=tuple(dict.fromkeys(blockers)), decision=decision,
            **cap_result,
            scientific_semantic_capabilities={
                str(key): dict(value) for key, value in (profile.get("scientific_semantic_capabilities", {}) or {}).items() if isinstance(value, Mapping)
            },
        ))
    return DatasetConceptCapabilityMap(cells=tuple(cells))


__all__ = ["ALLOWED_DECISIONS", "DatasetConceptCapability", "DatasetConceptCapabilityMap", "build_dataset_concept_capability_map", "evaluate_capability_requirements"]
