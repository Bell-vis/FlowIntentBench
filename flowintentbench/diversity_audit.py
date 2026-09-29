"""Explicit-universe portfolio diversity summaries."""

from __future__ import annotations

import re
from collections import Counter
from enum import Enum
from typing import Any, Mapping, Sequence


class PortfolioUniverse(str, Enum):
    HISTORICAL_CONTROLLED_PILOT = "HISTORICAL_CONTROLLED_PILOT"
    CONCEPT_EXPANSION_CANDIDATES = "CONCEPT_EXPANSION_CANDIDATES"
    COMBINED_CONSTRUCTION_PORTFOLIO = "COMBINED_CONSTRUCTION_PORTFOLIO"
    FORMAL_RELEASE_CANDIDATES = "FORMAL_RELEASE_CANDIDATES"


# Frozen construction diagnostic: a template occurring in at least one
# quarter of a declared case universe is a material case-level dominance
# signal.  The threshold is intentionally independent of family count.
CASE_TEMPLATE_DOMINANCE_THRESHOLD = 0.25
FAMILY_TEMPLATE_DOMINANCE_THRESHOLD = 0.50


def _percentages(counter: Counter[str], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {}
    return {key: round(value / denominator, 6) for key, value in sorted(counter.items())}


def _normalize_text(value: Any, *, strip_dataset: bool = False, dataset_id: Any = None) -> str:
    text = re.sub(r"[*_`~]", "", str(value or "")).casefold()
    if strip_dataset and dataset_id:
        name = re.sub(r"[*_`~]", "", str(dataset_id)).casefold()
        text = re.sub(r"(?<!\w)" + re.escape(name) + r"(?!\w)", "<dataset>", text)
    text = re.sub(r"[-+]?\d+(?:\.\d+)?", "<num>", text)
    text = re.sub(r"\([^)]*\)", "<coords>", text)
    if strip_dataset:
        text = re.sub(r"\b(?:blunt[_ -]?fin|carotid|combustor|fireflow|kitchen|nasa[_ -]?lox[_ -]?post|office)\b", "<dataset>", text)
    return re.sub(r"\s+", " ", text).strip()


def _question_template_class(question: Any, target_class: str, *, dataset_id: Any = None) -> str:
    """Reduce authored question prose to a stable scientific task template."""

    text = _normalize_text(question, strip_dataset=True, dataset_id=dataset_id)
    # Method clauses contain instantiated thresholds/coordinates and are not
    # question-template identity.  Keep the interrogative lead only.
    lead = text.split("?", 1)[0].strip()
    if target_class == "high_speed_flow_region":
        if "where" in lead or "located" in lead:
            return "select_strongest_high_speed_region_and_report_location"
        if "characterize" in lead:
            return "characterize_strongest_high_speed_region"
        return "select_strongest_high_speed_region"
    if target_class == "concentration_heterogeneity":
        return "select_most_heterogeneous_concentration_field"
    if target_class == "turbulence_activity_hotspot":
        return "select_turbulence_activity_hotspot"
    if target_class == "density_feature":
        return "select_density_feature"
    return lead


def _case_metadata(case: Mapping[str, Any]) -> dict[str, Any]:
    """Project explicit family semantics without modifying scientific records."""
    family = case.get("family", {})
    metadata = case.get("metadata", case.get("case_construction_metadata", {}))
    return {**(dict(family) if isinstance(family, Mapping) else {}),
            **(dict(metadata) if isinstance(metadata, Mapping) else {})}


def _infer_target_class(case: Mapping[str, Any]) -> str:
    """Infer a scientific target class without retaining dataset/condition text."""

    metadata = _case_metadata(case)
    target = metadata.get("model_visible_scientific_target")
    if isinstance(target, Mapping) and target.get("canonical_id"):
        return str(target["canonical_id"])
    concept = metadata.get("concept_id")
    if concept:
        return {"high_speed_region": "high_speed_flow_region"}.get(str(concept), str(concept))
    question = case.get("scientific_question")
    if question is None and isinstance(case.get("case_input"), Mapping):
        question = case["case_input"].get("scientific_question")
    semantic_text = f"{metadata.get('scientific_target', '')} {question or ''} {metadata.get('case_family_id', '')}".casefold()
    if "high-speed" in semantic_text or "high speed" in semantic_text or "speed_zones" in semantic_text:
        return "high_speed_flow_region"
    if "density" in semantic_text and "isosurface" in semantic_text:
        return "density_isosurface"
    if "turbulence" in semantic_text and ("hotspot" in semantic_text or "turbulence_activity" in semantic_text):
        return "turbulence_activity_hotspot"
    if "concentration" in semantic_text or "heterogeneity" in semantic_text:
        return "concentration_heterogeneity"
    if "density" in semantic_text and "feature" in semantic_text:
        return "density_feature"
    return _normalize_text(metadata.get("scientific_target"), strip_dataset=True, dataset_id=case.get("dataset_id", metadata.get("dataset_id")))


def _scientific_entity_type(case: Mapping[str, Any], target_class: str) -> str:
    metadata = _case_metadata(case)
    explicit = metadata.get("scientific_entity_type")
    if explicit:
        return str(explicit).upper()
    return {
        "high_speed_flow_region": "SPATIAL_REGION",
        "turbulence_activity_hotspot": "SPATIAL_REGION",
        "concentration_heterogeneity": "FIELD",
        "density_feature": "POINT_FEATURE",
        "density_isosurface": "SURFACE",
    }.get(target_class, "SCIENTIFIC_ENTITY")


def _scientific_archetype(case: Mapping[str, Any], target_class: str) -> str:
    metadata = _case_metadata(case)
    explicit = metadata.get("analysis_archetype")
    if explicit:
        return str(explicit).upper()
    return {
        "high_speed_flow_region": "THRESHOLD_REGION",
        "turbulence_activity_hotspot": "EXTREMUM_FEATURE",
        "concentration_heterogeneity": "FIELD_HETEROGENEITY",
        "density_feature": "GRADIENT_FEATURE",
        "density_isosurface": "LEVEL_SET_GEOMETRY",
    }.get(target_class, "UNSPECIFIED")


def _finding_goal_class(case: Mapping[str, Any], target_class: str) -> str:
    """Normalize F1/F2 wording to the scientific obligation class.

    F2 may ask for additional characterization, but it remains the same
    scientific question template when the target and core finding obligation
    are unchanged.  Protocol-specific F1/F2 labels stay in the condition
    signature.
    """

    metadata = _case_metadata(case)
    contract = metadata.get("finding_requirement_contract", {})
    if isinstance(contract, Mapping) and contract.get("adequate_core_sets"):
        return ";".join(sorted(",".join(sorted(str(role) for role in roles))
                               for roles in contract["adequate_core_sets"]))
    return {
        "high_speed_flow_region": "location_and_strength",
        "turbulence_activity_hotspot": "location_and_activity_quantity",
        "concentration_heterogeneity": "field_identity_and_heterogeneity",
        "density_feature": "location_and_density_quantity",
        "density_isosurface": "location_density_and_extent",
    }.get(target_class, "scientific_finding")


def scientific_question_template_signature(case: Mapping[str, Any]) -> str:
    """Return a scientific-question identity independent of protocol condition.

    Dataset names, O/F labels, instantiated numbers and coordinates are
    deliberately absent.  This is the signature used for scientific
    diversity claims; the condition signature below is kept separate for
    controlled-protocol diagnostics.
    """

    metadata = _case_metadata(case)
    question = case.get("scientific_question")
    if question is None and isinstance(case.get("case_input"), Mapping):
        question = case["case_input"].get("scientific_question")
    target = _infer_target_class(case)
    principal = tuple(sorted(str(x) for x in metadata.get("principal_operationalization_dimensions", case.get("principal_dimensions", [])) or []))
    # F1's explicit location clause and F2's characterization clause are
    # protocol-level finding obligations, not distinct scientific questions.
    intent = {
        "high_speed_flow_region": "select_strongest_high_speed_region",
        "turbulence_activity_hotspot": "select_turbulence_activity_hotspot",
        "concentration_heterogeneity": "select_most_heterogeneous_concentration_field",
        "density_feature": "select_density_feature",
        "density_isosurface": "characterize_density_isosurface",
    }.get(target, _question_template_class(question, target, dataset_id=case.get("dataset_id", metadata.get("dataset_id"))))
    return "|".join((
        target,
        _scientific_entity_type(case, target),
        _scientific_archetype(case, target),
        intent,
        ",".join(principal),
        _finding_goal_class(case, target),
    ))


def _cross_dataset_surface_template_signature(question: Any, *, dataset_id: Any = None) -> str:
    """Normalize question prose for a cross-dataset reuse diagnostic.

    This deliberately operates on the human-facing question, rather than on
    case metadata.  Dataset names, condition labels, instantiated values and
    coordinates are removed, while the scientific verb/object wording is
    retained.  The result is an audit key, not a scientific validity claim.
    """

    text = _normalize_text(question, strip_dataset=True, dataset_id=dataset_id)
    text = re.sub(r"\b(?:o[123])\s*[-/]\s*f[12]\b", "<condition>", text)
    text = re.sub(r"\b(?:condition|case)\s+o[123]\s*[-/]\s*f[12]\b", "<condition>", text)
    # ``strip_dataset=True`` above already replaces dataset names with the
    # placeholder; do not rewrite the placeholder itself (which would yield
    # a confusing ``<<dataset>>`` token).
    # Keep sentence content but remove presentation-only punctuation and
    # whitespace differences so a reused scientific template is visible.
    text = re.sub(r"[^a-z0-9<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cross_dataset_scientific_template_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit scientific-question template reuse across dataset boundaries.

    The audit is intentionally diagnostic-only: a shared scientific
    archetype is not automatically a family-condition leakage failure.  It
    reports exact normalized prose reuse and the datasets participating in
    each cluster so a curator can distinguish legitimate portfolio
    repetition from accidental copy/paste.
    """

    entries: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        question = row.get("scientific_question")
        if question is None:
            primary = row.get("primary_human_facing_question")
            question = primary
        if question is None:
            primary_id = str(row.get("primary_candidate_id") or "")
            for candidate in row.get("question_candidates", ()) or ():
                if not isinstance(candidate, Mapping):
                    continue
                if not primary_id or str(candidate.get("candidate_id") or "") == primary_id:
                    question = candidate.get("model_visible_text") or candidate.get("scientific_question")
                    if question is not None:
                        break
        signature = _cross_dataset_surface_template_signature(question, dataset_id=row.get("dataset_id"))
        if not signature:
            continue
        entries.append(
            {
                "slot_id": str(row.get("slot_id") or row.get("case_id") or ""),
                "dataset_id": str(row.get("dataset_id") or ""),
                "condition": str(row.get("condition") or ""),
                "signature": signature,
                # Build the already-frozen scientific identity signature from
                # the row's projection.  This catches reuse hidden by benign
                # dataset-specific scope wording while adding no new schema.
                "structured_signature": scientific_question_template_signature(
                    {
                        "scientific_question": question,
                        "metadata": {
                            "scientific_target": row.get("scientific_target"),
                            "case_family_id": row.get("family_id"),
                            "principal_operationalization_dimensions": [
                                item.get("dimension_id")
                                for item in row.get("fixed_operationalization", ()) or ()
                                if isinstance(item, Mapping)
                            ],
                        },
                    }
                ),
            }
        )
    clusters: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        clusters.setdefault(entry["signature"], []).append(entry)
    cross_dataset_clusters = []
    for signature, members in sorted(clusters.items()):
        datasets = sorted({member["dataset_id"] for member in members if member["dataset_id"]})
        if len(datasets) < 2:
            continue
        cross_dataset_clusters.append(
            {
                "template_signature": signature,
                "question_count": len(members),
                "dataset_count": len(datasets),
                "datasets": datasets,
                "slots": sorted(
                    members,
                    key=lambda item: (item["dataset_id"], item["condition"], item["slot_id"]),
                ),
            }
        )
    structured_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        structured_groups.setdefault(entry["structured_signature"], []).append(entry)
    structured_clusters = []
    for signature, members in sorted(structured_groups.items()):
        datasets = sorted({member["dataset_id"] for member in members if member["dataset_id"]})
        if len(datasets) < 2:
            continue
        structured_clusters.append(
            {
                "scientific_template_signature": signature,
                "question_count": len(members),
                "dataset_count": len(datasets),
                "datasets": datasets,
                "slots": sorted(
                    [
                        {
                            "slot_id": member["slot_id"],
                            "dataset_id": member["dataset_id"],
                            "condition": member["condition"],
                        }
                        for member in members
                    ],
                    key=lambda item: (item["dataset_id"], item["condition"], item["slot_id"]),
                ),
            }
        )
    dominant = max(clusters.items(), key=lambda item: (len(item[1]), item[0])) if clusters else None
    return {
        "status": "PASS",
        "audit_type": "CROSS_DATASET_SCIENTIFIC_TEMPLATE_DIVERSITY",
        "diagnostic_only": True,
        "question_count": len(entries),
        "dataset_count": len({entry["dataset_id"] for entry in entries if entry["dataset_id"]}),
        "unique_surface_template_count": len(clusters),
        "cross_dataset_reuse_cluster_count": len(cross_dataset_clusters),
        "cross_dataset_reuse_observed": bool(cross_dataset_clusters),
        "diversity_status": "REVIEW" if (cross_dataset_clusters or structured_clusters) else "PASS",
        "interpretation": (
            "Shared normalized scientific wording is reported for curator inspection; "
            "it is not treated as a family-condition leakage failure."
        ),
        "dominant_template_signature": dominant[0] if dominant else None,
        "dominant_template_count": len(dominant[1]) if dominant else 0,
        "dominant_template_fraction": round(len(dominant[1]) / len(entries), 6) if dominant and entries else 0.0,
        "clusters": cross_dataset_clusters,
        "structured_template_count": len(structured_groups),
        "structured_cross_dataset_reuse_cluster_count": len(structured_clusters),
        "structured_cross_dataset_reuse_observed": bool(structured_clusters),
        "structured_clusters": structured_clusters,
    }


def experimental_condition_signature(case: Mapping[str, Any]) -> str:
    """Return a protocol signature retaining O/F and openness information."""

    metadata = case.get("metadata", case.get("case_construction_metadata", {}))
    metadata = metadata if isinstance(metadata, Mapping) else {}
    condition = str(case.get("condition") or "")
    if not condition:
        match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", str(case.get("case_id", "")), re.I)
        condition = f"{match.group(1).upper()}-{match.group(2).upper()}" if match else "UNKNOWN"
    unresolved = tuple(sorted(str(x) for x in metadata.get("unresolved_operationalization_dimensions", case.get("unresolved_dimensions", [])) or []))
    finding = tuple(sorted(str(item.get("category")) for item in metadata.get("explicit_finding_requirements", []) or [] if isinstance(item, Mapping) and item.get("category")))
    return "|".join((condition, ",".join(unresolved), ",".join(finding)))


def _case_signature(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata", case.get("case_construction_metadata", {}))
    if not isinstance(metadata, Mapping):
        metadata = {}
    question = case.get("scientific_question")
    if question is None and isinstance(case.get("case_input"), Mapping):
        question = case["case_input"].get("scientific_question")
    condition = str(case.get("condition") or "")
    if not condition:
        match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", str(case.get("case_id", "")), re.I)
        condition = f"{match.group(1).upper()}-{match.group(2).upper()}" if match else "UNKNOWN"
    principal = tuple(sorted(str(x) for x in metadata.get("principal_operationalization_dimensions", case.get("principal_dimensions", [])) or []))
    unresolved = tuple(sorted(str(x) for x in metadata.get("unresolved_operationalization_dimensions", case.get("unresolved_dimensions", [])) or []))
    requirements = tuple(sorted(str(item.get("category")) for item in metadata.get("explicit_finding_requirements", []) or [] if isinstance(item, Mapping)))
    target_class = _infer_target_class(case)
    constraints = {
        "feature_definition": "OPEN",
        "criterion": "OPEN",
        "property_measure": "OPEN",
        "aggregation_or_representation": "OPEN",
    }
    for item in metadata.get("explicit_method_constraints", []) or []:
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("category", ""))
        if dimension == "analysis_procedure":
            dimension = "aggregation_or_representation"
        if dimension not in constraints or dimension in unresolved:
            continue
        text = str(item.get("statement", "")).casefold()
        if dimension == "feature_definition":
            constraints[dimension] = "CONNECTED_REGION" if any(token in text for token in ("connected", "contiguous", "component")) else "FIELD_OR_FEATURE"
        elif dimension == "criterion":
            constraints[dimension] = "THRESHOLD" if re.search(r"threshold|above|greater|criterion|percentile", text) else "DISTRIBUTIONAL_RULE"
        elif dimension == "property_measure":
            constraints[dimension] = "PEAK" if "peak" in text or "maximum" in text else "MEAN" if "mean" in text or "average" in text else "OTHER_MEASURE"
        else:
            constraints[dimension] = "FIELD_IDENTIFIER" if "identifier" in text or "field" in text and "location" not in text else "CENTROID_OR_MEAN" if "mean" in text or "centroid" in text else "PEAK_POINT" if "peak" in text and "location" in text else "OTHER_REPRESENTATION"
    normalized_question = _question_template_class(question, target_class)
    return "|".join((
        condition,
        target_class,
        ",".join(principal), ",".join(unresolved),
        ";".join(f"{key}={constraints[key]}" for key in sorted(constraints)),
        ",".join(requirements),
        normalized_question,
    ))


def _family_template_signature(item: Mapping[str, Any]) -> str:
    concept = str(item.get("concept_id", ""))
    if concept == "high_speed_region":
        concept = "high_speed_flow_region"
    principal = ",".join(sorted(str(x) for x in item.get("principal_dimensions", []) or []))
    return f"{concept}|{principal}"


def build_diversity_report(
    families: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    *,
    universe: PortfolioUniverse | str | None = None,
    candidate_status_filter: str | None = None,
) -> dict[str, Any]:
    """Summarize one declared portfolio universe without mixing populations."""

    universe_value = (universe.value if isinstance(universe, PortfolioUniverse) else str(universe)) if universe else PortfolioUniverse.COMBINED_CONSTRUCTION_PORTFOLIO.value
    selected_families = list(families)
    selected_cases = list(cases)
    if candidate_status_filter:
        selected_families = [item for item in selected_families if str(item.get("release_status", item.get("status", ""))) == candidate_status_filter]
        # Filter the same scientific population at both levels. Never count
        # held cases in the denominator of an eligible-family summary.
        owners: dict[str, str] = {}
        for family in families:
            for case_id in family.get("controlled_case_ids", ()):
                if case_id in owners and owners[case_id] != family.get("family_id"):
                    raise ValueError(f"case belongs to multiple families: {case_id}")
                owners[str(case_id)] = str(family.get("family_id"))
        retained = {str(family.get("family_id")) for family in selected_families}
        scoped_cases = []
        for case in selected_cases:
            metadata = _case_metadata(case)
            declared = metadata.get("case_family_id") or case.get("family_id") or metadata.get("family_id")
            owner = owners.get(str(case.get("case_id")), declared)
            if owner is None:
                raise ValueError(f"case has no family binding: {case.get('case_id')}")
            if declared and owner != declared:
                raise ValueError(f"case family binding conflicts: {case.get('case_id')}")
            if owner in retained:
                scoped_cases.append(case)
        selected_cases = scoped_cases
    family_ids = sorted({str(item.get("family_id")) for item in selected_families if item.get("family_id")})
    concept_ids = sorted({str(item.get("concept_id", item.get("family_id"))) for item in selected_families if item.get("concept_id", item.get("family_id"))})
    dataset_ids = sorted({str(item.get("dataset_id")) for item in selected_families if item.get("dataset_id")})
    archetypes = Counter(str(item.get("analysis_archetype")) for item in selected_families if item.get("analysis_archetype"))
    principal_signatures = Counter("|".join(sorted(str(x) for x in item.get("principal_dimensions", []) or [])) for item in selected_families)
    unresolved_signatures = Counter("|".join(sorted(str(x) for x in item.get("unresolved_dimensions", item.get("o2_unresolved_dimensions", []) or []))) for item in selected_families)
    family_templates = Counter(_family_template_signature(item) for item in selected_families)
    case_signatures = Counter(_case_signature(case) for case in selected_cases)
    scientific_templates = Counter(scientific_question_template_signature(case) for case in selected_cases)
    condition_templates = Counter(experimental_condition_signature(case) for case in selected_cases)
    family_case_ids = {
        str(case_id): str(family.get("family_id"))
        for family in selected_families
        for case_id in family.get("controlled_case_ids", []) or []
    }
    for case in selected_cases:
        metadata = _case_metadata(case)
        declared = metadata.get("case_family_id") or case.get("family_id") or metadata.get("family_id")
        if declared in family_ids:
            family_case_ids.setdefault(str(case.get("case_id")), str(declared))
    case_families = Counter(family_case_ids[str(case.get("case_id"))] for case in selected_cases if str(case.get("case_id")) in family_case_ids)
    family_dominant_count = max(family_templates.values(), default=0)
    family_dominant_fraction = family_dominant_count / len(selected_families) if selected_families else 0.0
    case_dominant_count = max(case_signatures.values(), default=0)
    case_dominant_fraction = case_dominant_count / len(selected_cases) if selected_cases else 0.0
    scientific_dominant_count = max(scientific_templates.values(), default=0)
    scientific_dominant_fraction = scientific_dominant_count / len(selected_cases) if selected_cases else 0.0
    family_dominance = bool(selected_families) and family_dominant_fraction >= FAMILY_TEMPLATE_DOMINANCE_THRESHOLD
    case_dominance = bool(selected_cases) and case_dominant_fraction >= CASE_TEMPLATE_DOMINANCE_THRESHOLD
    # A family-level archetype count cannot clear dominance when case content
    # is effectively one template.
    release_gate = "NOT_EVALUABLE" if universe_value == PortfolioUniverse.FORMAL_RELEASE_CANDIDATES.value and not selected_families else "REVIEW" if case_dominance else "PASS"
    return {
        "portfolio_universe": universe_value,
        "candidate_status_filter": candidate_status_filter,
        "case_ids": sorted(str(item.get("case_id")) for item in selected_cases if item.get("case_id")),
        "family_ids": family_ids,
        "concept_ids": concept_ids,
        "case_count": len(selected_cases),
        "dataset_count": len(dataset_ids),
        "datasets": dataset_ids,
        "concept_count": len(concept_ids),
        "dataset_family_count": len(family_ids),
        "family_count": len(selected_families),
        "concept_percentages": _percentages(Counter(str(item.get("concept_id", item.get("family_id"))) for item in selected_families), len(selected_families)),
        "family_percentages": _percentages(Counter(str(item.get("family_id")) for item in selected_families), len(selected_families)),
        "case_family_percentages": _percentages(case_families, len(selected_cases)),
        "analysis_archetype_count": len(archetypes),
        "analysis_archetypes": dict(sorted(archetypes.items())),
        "analysis_archetype_percentages": _percentages(archetypes, len(selected_families)),
        "principal_dimension_signatures": dict(sorted(principal_signatures.items())),
        "principal_dimension_percentages": _percentages(principal_signatures, len(selected_families)),
        "o2_unresolved_dimension_signatures": dict(sorted(unresolved_signatures.items())),
        "o2_unresolved_dimension_percentages": _percentages(unresolved_signatures, len(selected_families)),
        "question_template_signatures": dict(sorted(family_templates.items())),
        "case_template_signatures": dict(sorted(case_signatures.items())),
        "scientific_question_template_signatures": dict(sorted(scientific_templates.items())),
        "scientific_question_template_count": len(scientific_templates),
        "dominant_scientific_question_template": scientific_templates.most_common(1)[0][0] if scientific_templates else None,
        "dominant_scientific_question_fraction": round(scientific_dominant_fraction, 6),
        "experimental_condition_signatures": dict(sorted(condition_templates.items())),
        "experimental_condition_template_count": len(condition_templates),
        "condition_balance": _percentages(condition_templates, len(selected_cases)),
        "finding_requirement_signatures": dict(sorted(Counter("|".join(sorted(str(x) for x in (item.get("finding_schema", {}).get("f1_goal", ""), item.get("finding_schema", {}).get("f2_goal", "")))) for item in selected_families).items())),
        "case_template_signature_count": len(case_signatures),
        "family_level_template_dominance": family_dominance,
        "family_level_dominant_template_signature": family_templates.most_common(1)[0][0] if family_templates else None,
        "family_level_dominant_template_count": family_dominant_count,
        "family_level_dominant_fraction": round(family_dominant_fraction, 6),
        "case_level_template_dominance": case_dominance,
        "case_level_dominant_template_signature": case_signatures.most_common(1)[0][0] if case_signatures else None,
        "case_level_dominant_template_count": case_dominant_count,
        "case_level_dominant_fraction": round(case_dominant_fraction, 6),
        "case_template_dominance_threshold": CASE_TEMPLATE_DOMINANCE_THRESHOLD,
        "family_template_dominance_threshold": FAMILY_TEMPLATE_DOMINANCE_THRESHOLD,
        # Compatibility aliases retain their historical meaning but are now
        # explicitly case-level rather than silently copied from families.
        "case_template_dominance": case_dominance,
        "template_dominance_flag": case_dominance,
        "release_gate": release_gate,
    }


__all__ = [
    "CASE_TEMPLATE_DOMINANCE_THRESHOLD",
    "FAMILY_TEMPLATE_DOMINANCE_THRESHOLD",
    "PortfolioUniverse",
    "build_diversity_report",
    "cross_dataset_scientific_template_audit",
    "experimental_condition_signature",
    "scientific_question_template_signature",
]
