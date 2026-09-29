"""Materialize case-grounded scientific portfolio audit artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diversity_audit import PortfolioUniverse, build_diversity_report
from .family_validator import (
    audit_o3_target_alignment_from_cases,
    canonicalize_operationalization_clause,
    evaluate_family_release_eligibility,
    normalize_resolved_clauses,
    resolved_dimension_invariance,
    validate_reference_space_semantics,
    validate_family_definition,
)
from .finding_requirements import FindingRequirementContract, audit_f2_requirements
from .scientific_family import DatasetConceptFamily, GroundingStatus, PortfolioLifecycle, ScientificConcept


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clauses_from_metadata(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    clauses: dict[str, dict[str, Any]] = {}
    for item in metadata.get("explicit_method_constraints", []) or []:
        if not isinstance(item, Mapping):
            continue
        dimension = str(item.get("category", ""))
        if dimension == "analysis_procedure":
            dimension = "aggregation_or_representation"
        if dimension in {"feature_definition", "criterion", "property_measure", "aggregation_or_representation"}:
            clauses[dimension] = canonicalize_operationalization_clause(dimension, str(item.get("statement", "")))
    return clauses


def _f2_contract(family_id: str, concept_id: str | None = None) -> dict[str, Any]:
    """Return a family-scoped adequate-core F2 contract."""
    concept = concept_id or family_id
    if concept == "high_speed_region":
        entity, mandatory = "SPATIAL_REGION", ["principal_feature_location", "defining_strength_evidence"]
        role_by_category = {"location": "principal_feature_location", "quantity": "defining_strength_evidence", "property": "defining_strength_evidence"}
        alternatives = ["spatial_extent", "orientation", "geometric_relation", "morphology", "meaningful_comparison"]
    elif "concentration" in concept:
        entity, mandatory = "FIELD", ["selected_field_identity", "heterogeneity_measure_evidence"]
        role_by_category = {"existence_or_identity": "selected_field_identity", "property": "heterogeneity_measure_evidence", "quantity": "heterogeneity_measure_evidence"}
        alternatives = ["distribution_spread", "comparison_to_other_fields", "distribution_shape", "extreme_behavior", "consequential_distributional_characterization"]
    elif "turbulence" in concept:
        entity, mandatory = "SPATIAL_REGION", ["hotspot_location", "activity_measure_evidence"]
        role_by_category = {"location": "hotspot_location", "property": "activity_measure_evidence", "quantity": "activity_measure_evidence"}
        alternatives = ["spatial_extent", "relation_to_geometry", "comparison_to_secondary_hotspot", "companion_turbulence_quantity", "morphology"]
    else:
        entity, mandatory = "POINT_FEATURE", ["feature_location", "defining_density_or_gradient_evidence"]
        role_by_category = {"location": "feature_location", "property": "defining_density_or_gradient_evidence", "quantity": "defining_density_or_gradient_evidence"}
        alternatives = ["gradient_orientation", "spatial_extent", "contrast", "meaningful_comparison"]
    sets = [mandatory + [role] for role in alternatives]
    return {
        "scientific_entity_type": entity,
        "contract_source": "CONSTRUCTION_DEFINED",
        "role_by_category": role_by_category,
        "mandatory_roles": mandatory,
        "alternative_role_groups": [{"group_id": "characterization", "min_required": 1, "roles": alternatives}],
        "adequate_core_sets": sets,
        "supporting_roles": ["supporting_statistic"],
        "novel_role_policy": "VALID_UNENUMERATED may satisfy one characterization role only after blinded adjudication",
    }


def _reference_anchors(records: Sequence[Mapping[str, Any]], family_id: str, *, legacy: bool) -> list[dict[str, Any]]:
    """Import accepted O branches from actual authored case Ground Truth."""

    anchors: list[dict[str, Any]] = []
    for record in records:
        ground_truth = record.get("ground_truth") or {}
        for index, branch in enumerate(ground_truth.get("acceptable_operationalizations", []) or []):
            if not isinstance(branch, Mapping):
                continue
            branch_id = str(branch.get("operationalization_id") or f"branch-{index + 1}")
            decisions = branch.get("decisions", []) or []
            canonical = {
                str(item.get("dimension")): canonicalize_operationalization_clause(str(item.get("dimension")), str(item.get("statement", "")))
                for item in decisions if isinstance(item, Mapping)
            }
            findings: list[dict[str, Any]] = []
            for gt_branch in ground_truth.get("findings_by_operationalization", []) or []:
                if isinstance(gt_branch, Mapping) and str(gt_branch.get("operationalization_id")) == branch_id:
                    findings = [dict(item) for item in gt_branch.get("findings", []) or [] if isinstance(item, Mapping)]
                    break
            anchors.append({
                "reference_o_id": f"{record.get('case_id')}::{branch_id}",
                "family_id": family_id,
                "source_case_id": record.get("case_id"),
                "dataset_id": record.get("dataset_id"),
                "resolved_operationalization_clauses": canonical,
                "reference_findings": findings,
                "validity_status": "ENUMERATED_VALID",
                "scientific_rationale": "Imported from the authored Ground Truth accepted operationalization branch.",
                "provenance": "LEGACY_GT_REFERENCE" if legacy else "GROUND_TRUTH_BRANCH",
            })
    return anchors


def _manifest_case_directory(repository_root: Path, item: Mapping[str, Any]) -> Path:
    """Resolve a case only from its explicit manifest artifact identity.

    The canonical N=1 manifest is bound to the immutable reference snapshot;
    its rows therefore do not live below ``datasets/<dataset>/construction``.
    Falling back to the historical layout is retained solely for old pilot
    manifests that do not carry an explicit ``case_input_path``.
    """

    explicit = item.get("case_input_path") or item.get("case_input")
    if explicit:
        path = Path(str(explicit))
        path = path if path.is_absolute() else repository_root / path
        path = path.resolve()
        if path.is_file():
            return path.parent
        # Keep a useful error at the actual artifact boundary instead of
        # silently switching to a different case with the same id.
        return path.parent
    case_id = str(item.get("case_id", ""))
    dataset_id = str(item.get("dataset_id", ""))
    return repository_root / "datasets" / dataset_id / "construction" / "cases" / case_id


def _load_historical_case_records(
    repository_root: Path,
    manifest_cases: Sequence[Mapping[str, Any]],
    *,
    case_id_aliases: Mapping[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in manifest_cases:
        source_case_id = str(item.get("case_id"))
        case_id = (
            str((case_id_aliases or {}).get((str(item.get("dataset_id")), str(item.get("condition"))), source_case_id))
        )
        dataset_id = str(item.get("dataset_id"))
        source_item = dict(item)
        source_item["case_id"] = source_case_id
        case_dir = _manifest_case_directory(repository_root, source_item)
        metadata = _read(case_dir / "case_construction_metadata.json")
        case_input = _read(case_dir / "case_input.json")
        gt_path = case_dir / "ground_truth.json"
        ground_truth = _read(gt_path) if gt_path.is_file() else {}
        records.append({
            "case_id": case_id,
            "dataset_id": dataset_id,
            "condition": item.get("condition"),
            "metadata": metadata,
            "case_input": case_input,
            "scientific_question": case_input.get("scientific_question"),
            "ground_truth": ground_truth,
            "resolved_operationalization_clauses": _clauses_from_metadata(metadata),
        })
    return records


def _load_candidate_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_root = root / "artifacts/reference/concept_expansion_phase1/candidate_cases"
    for path in sorted(candidate_root.glob("*/*/case_construction_metadata.json")):
        metadata = _read(path)
        case_dir = path.parent
        question = _read(case_dir / "scientific_question.json") if (case_dir / "scientific_question.json").is_file() else {}
        gt = _read(case_dir / "ground_truth.json") if (case_dir / "ground_truth.json").is_file() else {}
        records.append({
            "case_id": metadata.get("case_id"),
            "dataset_id": metadata.get("dataset_id"),
            "condition": (re.search(r"(?:^|_)(O[123])[_-](F[12])(?:$|_)", str(metadata.get("case_id", "")), re.I).group(1).upper() + "-" + re.search(r"(?:^|_)(O[123])[_-](F[12])(?:$|_)", str(metadata.get("case_id", "")), re.I).group(2).upper() if re.search(r"(?:^|_)(O[123])[_-](F[12])(?:$|_)", str(metadata.get("case_id", "")), re.I) else None),
            "metadata": metadata,
            "case_input": {"scientific_question": question.get("scientific_question")},
            "scientific_question": question.get("scientific_question"),
            "ground_truth": gt,
            "resolved_operationalization_clauses": _clauses_from_metadata(metadata),
        })
    return records


def _family_record(
    *, family_id: str, concept_id: str, dataset_id: str, target: str,
    archetype: str, principal: Sequence[str], cases: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any] | None = None,
    grounding: str = "PROVISIONAL",
    candidate_screening_status: str | None = None,
    required_capabilities: Mapping[str, Any] | None = None,
    optional_capabilities: Mapping[str, Any] | None = None,
    irrelevant_capabilities: Sequence[str] = (),
    scientific_semantic_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    o1 = next((c for c in cases if str(c.get("condition", "")).startswith("O1-F1")), None)
    o2 = next((c for c in cases if str(c.get("condition", "")).startswith("O2-F1")), None)
    baseline = (o1 or {}).get("resolved_operationalization_clauses", {})
    o2_metadata = (o2 or {}).get("metadata", {})
    unresolved = list(o2_metadata.get("unresolved_operationalization_dimensions", []) or [])
    contract = _f2_contract(family_id, concept_id)
    target_canonical = {
        "high_speed_region": "high_speed_flow_region",
        "kitchen_turbulence_activity": "turbulence_activity_hotspot",
        "kitchen_concentration_heterogeneity": "concentration_heterogeneity",
        "combustor_density_features": "density_feature",
    }.get(concept_id, concept_id or target)
    return {
        "family_id": family_id,
        "concept_id": concept_id,
        "dataset_id": dataset_id,
        "scientific_target": target,
        "model_visible_scientific_target": {"canonical_id": target_canonical, "description": target},
        "question_template": target,
        "rationale": "Dataset-specific family derived from actual authored case artifacts.",
        "evidence_bundle": dict(evidence or {}),
        "analysis_archetype": archetype,
        "principal_dimensions": list(principal),
        "unresolved_dimensions": unresolved,
        "o2_unresolved_dimensions": unresolved,
        "baseline_resolved_clauses": baseline,
        "resolved_operationalization_clauses": baseline,
        "candidate_reference_operationalizations": _reference_anchors(cases, family_id, legacy=concept_id == "high_speed_region"),
        "finding_schema": {"f1_goal": str((o1 or {}).get("metadata", {}).get("finding_goal", "")), "f2_goal": "selective characterization of the selected feature"},
        "finding_requirement_contract": contract,
        "controlled_case_ids": [str(c.get("case_id")) for c in cases],
        "controlled_case_records": [{"case_id": c.get("case_id"), "dataset_id": c.get("dataset_id"), "condition": c.get("condition")} for c in cases],
        "grounding_status": grounding,
        "candidate_screening_status": candidate_screening_status,
        "curator_status": "PENDING_EXPERT_REVIEW",
        "release_status": "COMPUTED",
        "lifecycle_status": PortfolioLifecycle.DATA_SUPPORTED.value,
        "required_capabilities": dict(required_capabilities or {}),
        "optional_capabilities": dict(optional_capabilities or {}),
        "irrelevant_capabilities": [str(item) for item in irrelevant_capabilities],
        "scientific_semantic_capabilities": {str(k): dict(v) for k, v in (scientific_semantic_capabilities or {}).items() if isinstance(v, Mapping)},
        # Explicit lifecycle metadata keeps the historical compatibility
        # adapter out of the generic formal evaluator.  Formal families must
        # provide their own finding contract and capability declarations.
        **({
            "portfolio_role": "controlled_pilot",
            "formal_release_eligible": False,
            "migration_status": "CONTROLLED_PILOT_ONLY",
            "legacy_fixed_core": True,
        } if concept_id == "high_speed_region" else {}),
    }


def load_portfolio_inputs(
    repository_root: str | Path,
    *,
    use_reference_bound: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(repository_root)
    cards = [_read(path) for path in sorted((root / "artifacts/reference/concept_expansion_phase1/concept_design_cards").glob("*.json"))]
    manifest = _read(root / "experiments/userstudy/case_manifest.json")
    historical = _load_historical_case_records(root, manifest.get("cases", []))
    # The v4 reference snapshot intentionally contains four dataset-specific
    # semantic templates (including the Combustor provisional family).  The
    # legacy diversity regression is about the archived controlled-pilot
    # wording and must consume that archive rather than the new N=1 authority.
    if (
        not use_reference_bound
        and str(manifest.get("reference_schema_version", "")) in {
            "reference-science-baseline-v4",
            "reference-science-baseline-v5",
            "reference-science-baseline-v6",
            "reference-science-baseline-v7",
        }
    ):
        legacy_manifest_path = root / "artifacts/reference/full_dataset_n1_controlled_pilot/pilot_manifest.json"
        legacy_manifest = _read(legacy_manifest_path) if legacy_manifest_path.is_file() else None
        if isinstance(legacy_manifest, Mapping):
            # Default readers preserve the historical controlled-pilot
            # population, but expose the current manifest's slot IDs so
            # compatibility consumers that filter by the canonical 7x4 table
            # still see one coherent historical template.  The archived bytes
            # remain the source; this alias is never used by the v4 builder.
            current_by_slot = {
                (str(item.get("dataset_id")), str(item.get("condition"))): str(item.get("case_id"))
                for item in manifest.get("cases", [])
                if isinstance(item, Mapping)
            }
            historical = _load_historical_case_records(
                root,
                legacy_manifest.get("cases", []),
                case_id_aliases=current_by_slot,
            )
    candidates = _load_candidate_records(root)
    screening = _read(root / "artifacts/reference/concept_expansion_phase1/concept_screening.json")
    return cards, historical + candidates, screening


def build_scientific_portfolio(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root), Path(output_root)
    cards, all_cases, screening = load_portfolio_inputs(root, use_reference_bound=True)
    historical = [c for c in all_cases if c["case_id"] in {str(i.get("case_id")) for i in _read(root / "experiments/userstudy/case_manifest.json").get("cases", [])}]
    candidates = [c for c in all_cases if c not in historical]
    # Seven dataset-specific high-speed families; no family may mix datasets.
    families: list[dict[str, Any]] = []
    for dataset_id in sorted({str(c["dataset_id"]) for c in historical}):
        cases = [c for c in historical if str(c["dataset_id"]) == dataset_id]
        metadata = cases[0]["metadata"]
        families.append(_family_record(
            family_id=f"{dataset_id.lower()}__high_speed_region",
            concept_id="high_speed_region",
            dataset_id=dataset_id,
            target=str(metadata.get("scientific_target", "high-speed flow regions")),
            archetype="THRESHOLD_REGION",
            principal=metadata.get("principal_operationalization_dimensions", []),
            cases=cases,
            evidence={"source": "historical_28_case_manifest", "grounding_assessment": "formal review pending"},
        ))
    # Concept Expansion candidates are separate dataset families.
    card_by_id = {str(card.get("concept_id") or card.get("family_id")): card for card in cards}
    for concept_id, card in sorted(card_by_id.items()):
        family_cases = [c for c in candidates if str(c["metadata"].get("case_family_id")) == str(card.get("family_id"))]
        if not family_cases:
            continue
        principal = card.get("principal_dimensions", [])
        archetype = {"kitchen_turbulence_activity": "EXTREMUM_FEATURE", "kitchen_concentration_heterogeneity": "FIELD_HETEROGENEITY", "combustor_density_features": "GRADIENT_FEATURE"}.get(concept_id, "EXTREMUM_FEATURE")
        families.append(_family_record(
            family_id=str(card.get("family_id") or concept_id), concept_id=concept_id,
            dataset_id=str(card.get("dataset_id")), target=str(card.get("scientific_question_family") or card.get("scientific_concept") or concept_id),
            archetype=archetype, principal=principal, cases=family_cases,
            evidence={"evidence_ids": card.get("evidence_ids", []), "grounding_assessment": card.get("grounding_assessment"), "source_literature_grounding": card.get("source_literature_grounding"), "observable_data_requirements": card.get("observable_data_requirements")},
            candidate_screening_status=str(card.get("screening_status") or "PROVISIONAL"),
            required_capabilities=card.get("required_capabilities", card.get("capability_requirements", {})),
            optional_capabilities=card.get("optional_capabilities", {}),
            irrelevant_capabilities=card.get("irrelevant_capabilities", ()),
            scientific_semantic_capabilities=card.get("scientific_semantic_capabilities", {}),
        ))
    # Actual O3, F2 and invariance audits for each family.
    o3_audits: dict[str, Any] = {}
    f2_audits: dict[str, Any] = {}
    invariant_audits: dict[str, Any] = {}
    release_audits: dict[str, Any] = {}
    for family in families:
        records = [c for c in all_cases if str(c.get("case_id")) in set(family["controlled_case_ids"])]
        o3 = audit_o3_target_alignment_from_cases(family, records)
        o3_audits[family["family_id"]] = o3
        f2 = audit_f2_requirements(family)
        f2_audits[family["family_id"]] = f2
        case_invariance: dict[str, Any] = {}
        for record in records:
            gt = record.get("ground_truth") or {}
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), Mapping) else {}
            principal = [str(item) for item in metadata.get("principal_operationalization_dimensions", family.get("principal_dimensions", [])) or []]
            unresolved = [str(item) for item in metadata.get("unresolved_operationalization_dimensions", []) or []]
            case_branches: list[dict[str, Any]] = []
            for branch in gt.get("acceptable_operationalizations", []) or []:
                decisions = {
                    str(item.get("dimension")): canonicalize_operationalization_clause(
                        str(item.get("dimension")), str(item.get("statement", ""))
                    )
                    for item in branch.get("decisions", [])
                    if isinstance(item, Mapping) and str(item.get("dimension")) not in set(unresolved)
                }
                case_branches.append({
                    "case_id": record.get("case_id"),
                    "branch_id": branch.get("operationalization_id"),
                    "resolved_operationalization_clauses": decisions,
                    "resolved_dimensions": list(decisions),
                })
            case_id = str(record.get("case_id"))
            case_result = resolved_dimension_invariance(
                case_branches,
                baseline_clauses=family.get("baseline_resolved_clauses"),
                principal_dimensions=principal,
                unresolved_dimensions=unresolved,
                family_id=family.get("family_id"),
                case_id=case_id,
            )
            case_invariance[case_id] = case_result | {
                "case_id": case_id,
                "condition": record.get("condition"),
                "principal_dimensions": principal,
                "unresolved_dimensions": unresolved,
                "required_resolved_dimensions": sorted(set(principal) - set(unresolved)),
                "branch_results": case_result.get("branch_results", []),
            }
        evaluable_cases = [item for item in case_invariance.values() if item.get("status") in {"PASS", "FAIL"}]
        all_drifts = [drift for item in case_invariance.values() for drift in item.get("drifts", [])]
        invariant_audits[family["family_id"]] = {
            "status": "PASS" if evaluable_cases and all(item.get("status") == "PASS" for item in evaluable_cases) else "FAIL" if any(item.get("status") == "FAIL" for item in evaluable_cases) else "NOT_EVALUABLE",
            "family_id": family.get("family_id"),
            "case_count": len(case_invariance),
            "evaluable_case_count": len(evaluable_cases),
            "branch_count": sum(int(item.get("branch_count", 0)) for item in case_invariance.values()),
            "drifts": all_drifts,
            "failure_code": next((item.get("code") for item in all_drifts), None),
            "cases": case_invariance,
        }
        errors = validate_family_definition(family)
        reference_audit = validate_reference_space_semantics({
            "accepted": family.get("candidate_reference_operationalizations", []),
            "explicit_rejected": [],
            "omitted_combinations": "UNENUMERATED",
            "require_anchor_schema": True,
        })
        audits = {
            "grounding_status": family.get("grounding_status"), "curator_status": "CONFIRMED" if family.get("curator_status") == "CONFIRMED" else family.get("curator_status"),
            "controlled_family_validation": "PASS" if not errors else "FAIL",
            "o3_target_alignment": o3.get("status"), "f2_construct_validation": f2.get("status"),
            "resolved_dimension_invariance": invariant_audits[family["family_id"]].get("status"),
            "data_contract_validation": "PASS" if records else "NOT_EVALUABLE", "reference_space_semantics": reference_audit.get("status"),
            "family_lifecycle_state": family.get("lifecycle_status"),
        }
        release_audits[family["family_id"]] = evaluate_family_release_eligibility(
            family, audits, conditions=[str(record["condition"]) for record in records]
        ) if records else evaluate_family_release_eligibility(family, audits)
        release_audits[family["family_id"]]["reference_space_audit"] = reference_audit
        family["release_status"] = release_audits[family["family_id"]]["release_status"]
        family["release_blockers"] = release_audits[family["family_id"]]["blockers"]
    classification = [{"case_id": c["case_id"], "dataset_id": c["dataset_id"], "condition": c.get("condition"), "conceptual_role": "CONTROLLED_PIPELINE_PILOT", "portfolio_role": "controlled_pilot", "scientific_family": "high_speed_region", "concept_id": "high_speed_region", "family_id": f"{str(c['dataset_id']).lower()}__high_speed_region", "formal_release_eligible": False, "migration_status": "CONTROLLED_PILOT_ONLY"} for c in historical]
    historical_families = [f for f in families if f["concept_id"] == "high_speed_region"]
    concept_objects = [ScientificConcept("high_speed_region", "high-speed flow regions", tuple(f["family_id"] for f in historical_families))] + [ScientificConcept(f["concept_id"], f["scientific_target"], (f["family_id"],)) for f in families if f["concept_id"] != "high_speed_region"]
    universes = {
        "historical": build_diversity_report(historical_families, historical, universe=PortfolioUniverse.HISTORICAL_CONTROLLED_PILOT),
        "concept_expansion": build_diversity_report([f for f in families if f["concept_id"] != "high_speed_region"], candidates, universe=PortfolioUniverse.CONCEPT_EXPANSION_CANDIDATES),
        "formal": build_diversity_report([f for f in families if release_audits[f["family_id"]]["release_eligible"]], [], universe=PortfolioUniverse.FORMAL_RELEASE_CANDIDATES),
    }
    formal_families = [f for f in families if release_audits[f["family_id"]]["release_eligible"]]
    _write(output / "historical_28_case_classification.json", {"status": "CONTROLLED_PILOT_ONLY", "case_count": len(classification), "dataset_family_count": len(historical_families), "concept_count": 1, "cases": classification})
    _write(output / "concept_family_inventory.json", {"status": "PORTFOLIO_CONSTRUCTION", "concepts": [c.to_dict() for c in concept_objects], "families": families, "rejected_screening_entries": [e for e in screening.get("entries", []) if e.get("status") == "NOT_SUPPORTED"]})
    _write(output / "scientific_grounding_audit.json", {"status": "REVIEW_REQUIRED", "families": [{"family_id": f["family_id"], "dataset_id": f["dataset_id"], "concept_id": f.get("concept_id"), "grounding_status": f["grounding_status"], "candidate_screening_status": f.get("candidate_screening_status"), "curator_status": f.get("curator_status"), "release_eligible": False, "evidence_bundle": f.get("evidence_bundle", {})} for f in families]})
    _write(output / "portfolio_diversity_report.json", universes["historical"])
    _write(output / "diversity_audit_historical.json", universes["historical"])
    _write(output / "diversity_audit_concept_expansion.json", universes["concept_expansion"])
    _write(output / "diversity_audit_formal_candidates.json", universes["formal"])
    _write(output / "o3_target_alignment_audit.json", {"status": "FAIL" if any(v.get("status") == "FAIL" for v in o3_audits.values()) else "REVIEW", "families": o3_audits})
    _write(output / "f2_construct_validity_audit.json", {"status": "PASS" if f2_audits and all(item.get("status") == "PASS" for item in f2_audits.values()) else "REVIEW", "families": f2_audits})
    _write(output / "f2_adequate_core_audit.json", {"status": "PASS" if f2_audits and all(item.get("status") == "PASS" for item in f2_audits.values()) else "REVIEW", "families": f2_audits})
    _write(output / "resolved_dimension_invariance_audit.json", {"status": "PASS" if invariant_audits and all(item.get("status") == "PASS" for item in invariant_audits.values()) else "REVIEW", "families": invariant_audits})
    _write(output / "lifecycle_release_audit.json", {"status": "NOT_READY", "families": release_audits})
    _write(output / "formal_portfolio_candidates.json", {"status": "EMPTY_PENDING_GROUNDING_AND_CURATOR", "families": formal_families})
    _write(output / "release_readiness.json", {"status": "PORTFOLIO_NOT_READY", "reason": "no family is simultaneously grounded, curator-confirmed, and release eligible", "family_validation_errors": {f["family_id"]: validate_family_definition(f) for f in families}})
    _write(output / "dataset_gap_report.json", {"status": "SPARSE_BY_DESIGN", "missing_cells_are_acceptable": True, "unsupported_concepts_are_not_promoted": True})
    _write(output / "portfolio_universe_audit.json", {"status": "PASS", "universes": universes})
    (output / "portfolio_diversity_report.md").write_text(
        "# Portfolio diversity audit\n\n"
        "The report is split by declared universe; case counts are not used as a proxy for scientific diversity.\n\n"
        + "\n".join(
            f"## {label}\n\n"
            f"- Families: {report['family_count']}\n"
            f"- Concepts: {report['concept_count']}\n"
            f"- Datasets: {report['dataset_count']}\n"
            f"- Analysis archetypes: {report['analysis_archetype_count']}\n"
            f"- Case-template dominance: {report['case_template_dominance']}\n"
            f"- Release gate: {report['release_gate']}\n"
            for label, report in (("Historical controlled pilot", universes["historical"]), ("Concept expansion candidates", universes["concept_expansion"]), ("Formal release candidates", universes["formal"]))
        ) + "\n",
        encoding="utf-8",
    )
    (output / "o3_target_alignment_audit.md").write_text(
        "# O3 target alignment audit\n\n"
        "O3 must keep the scientific target visible and fixed while leaving operationalization dimensions open.\n\n"
        + "\n".join(f"- `{family_id}`: **{audit['status']}**" for family_id, audit in sorted(o3_audits.items()))
        + "\n",
        encoding="utf-8",
    )
    (output / "f2_construct_validity_audit.md").write_text(
        "# F2 construct-validity audit\n\n"
        "F2 uses explicit adequate core sets; historical cases retain the legacy fixed-core designation.\n\n"
        + "\n".join(f"- `{family_id}`: **{audit['status']}**, adequate sets={len(audit.get('adequate_core_sets', []))}" for family_id, audit in sorted(f2_audits.items()))
        + "\n",
        encoding="utf-8",
    )
    (output / "resolved_dimension_invariance_audit.md").write_text(
        "# Resolved-dimension invariance audit\n\n"
        "Canonical clause IDs are compared across every accepted branch; unresolved dimensions may vary.\n\n"
        + "\n".join(f"- `{family_id}`: **{audit['status']}**, branches={audit.get('branch_count', 0)}" for family_id, audit in sorted(invariant_audits.items()))
        + "\n",
        encoding="utf-8",
    )
    (output / "dataset_gap_report.md").write_text(
        "# Dataset capability gaps\n\n"
        "The construction matrix is sparse by design. Unsupported dataset-concept cells are not promoted to satisfy a diversity target.\n\n"
        "Current gaps requiring future evidence or datasets include rotational/vorticity structure, pressure or thermal transport, wall shear, temporal behavior, and multiphase variables.\n",
        encoding="utf-8",
    )
    (output / "reference_space_semantics.md").write_text("# Reference space semantics\n\nReference operationalizations are non-exhaustive; omitted combinations remain UNENUMERATED.\n", encoding="utf-8")
    (output / "template_dominance_audit.md").write_text("# Template dominance audit\n\nDominance is computed from case-level question, target, dimension, and finding signatures; archetype labels alone cannot clear this gate.\n", encoding="utf-8")
    (output / "scientific_grounding_audit.md").write_text("# Scientific grounding audit\n\nAll current families remain provisional pending independent grounding and curator review.\n", encoding="utf-8")
    (output / "redesign_summary.md").write_text("# Scientific portfolio redesign\n\nHistorical cases are controlled-pilot-only; dataset-specific family instances and Concept Expansion candidates are audited separately.\n", encoding="utf-8")
    for family in families:
        directory = output / "families" / family["family_id"]
        _write(directory / "family_definition.json", family)
        _write(directory / "grounding_evidence.json", family.get("evidence_bundle", {}))
        _write(directory / "controlled_cases.json", {"status": "CONTROLLED_PILOT_ONLY" if family["concept_id"] == "high_speed_region" else "CONSTRUCTION_ONLY", "cases": family["controlled_case_ids"]})
        _write(directory / "reference_operationalizations.json", {"accepted": family.get("candidate_reference_operationalizations", []), "explicit_rejected": [], "omitted_combinations": "UNENUMERATED"})
        _write(directory / "finding_requirements.json", family["finding_requirement_contract"])
        (directory / "scientific_family_review.md").write_text("# Family review\n\nStatus: construction candidate pending curator confirmation.\n", encoding="utf-8")
    return {"status": "PORTFOLIO_NOT_READY", "family_count": len(families), "concept_count": len(concept_objects), "historical_case_count": len(historical), "output_root": str(output)}


__all__ = ["build_scientific_portfolio", "load_portfolio_inputs"]
