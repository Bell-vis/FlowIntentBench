"""Sparse seven-dataset scientific-question candidate-matrix construction.

This module is a review harness.  It derives existing scientific identities,
keeps eligibility condition-scoped, and optionally invokes the canonical Flow
Expert for scientific review, wording, and post-rewrite fidelity.  It never
executes SCQ, SRAC, curator confirmation, or tested-model evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .case_repository import load_case_record
from .hccq import audit_family_presentation_leakage, audit_human_presentation
from .human_facing_acceptance import audit_question_responsibility_realization
from .question_presentation import (
    audit_final_flow_expert_wording,
    audit_post_rewrite_semantic_fidelity,
    author_human_questions,
    build_question_presentation_authoring_packet,
    build_semantic_visibility_classification,
    nominate_primary_question_candidate,
    realize_human_questions,
    run_flow_expert_post_rewrite_semantic_fidelity,
    run_flow_expert_final_flow_wording_review,
    run_flow_expert_question_presentation_authoring,
    validate_question_semantic_fidelity,
    validate_o1_f2_effective_o_semantics,
)
from .scientific_run_snapshots import canonical_json_sha256
from .scientific_semantics import (
    build_scientific_semantic_projection,
    semantic_contract_sha256,
)


SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION = (
    "seven-dataset-scientific-question-candidate-matrix-v1"
)
CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")
MATRIX_SLOT_STATUSES = frozenset(
    {
        "CANDIDATE_FOR_HUMAN_REVIEW",
        "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE_REVIEW",
        "NEW_CASE_PENDING_HUMAN_SELECTION",
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED",
    }
)

# These three state spaces deliberately remain independent.  A question can be
# reviewable while its scientific release is blocked, and an unchanged
# historical identity can still have reuse authorization pending selection.
QUESTION_CANDIDATE_STATUSES = frozenset(
    {
        "READY_FOR_HUMAN_REVIEW",
        "NEEDS_SCIENTIFIC_REVISION",
        "NEEDS_EVIDENCE_CONTEXT",
        "NEEDS_TARGET_REDESIGN",
    }
)
SCIENTIFIC_RELEASE_STATUSES = frozenset(
    {
        "ELIGIBLE",
        "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE",
        "NEW_CASE_REQUIRED",
    }
)
ARTIFACT_REUSE_STATUSES = frozenset({"VALID", "STALE", "REBUILD_REQUIRED"})
SUPPORT_DIMENSION_NAMES = (
    "scientific_target_supported",
    "dataset_observable_supported",
    "execution_materialization_supported",
    "evaluation_contract_supported",
    "evidence_complete_for_release",
)

_SUPPORT_VALUES = frozenset(
    {"SUPPORTED", "UNSUPPORTED", "NOT_ESTABLISHED", "NOT_APPLICABLE"}
)
_INVARIANCE_VALUES = frozenset(
    {"SAME_TARGET_SUPPORTED", "TARGET_DRIFT", "NOT_ESTABLISHED", "NOT_APPLICABLE"}
)
_ACTIONS = frozenset(
    {"RETAIN", "REQUEST_EVIDENCE", "REVISE_O_SPACE", "OMIT"}
)
_FORBIDDEN_RESULT_EQUIVALENCE_REASONS = frozenset(
    {
        "G_OF_O_DISAGREEMENT",
        "NUMERICAL_RESULT_DISAGREEMENT",
        "RANKING_DISAGREEMENT",
        "SAME_RESULT_REQUIRED",
        "SELECTED_FEATURE_DISAGREEMENT",
    }
)

ConditionReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
QuestionAuthor = Callable[[Mapping[str, Any]], Mapping[str, Any]]
FidelityReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
FinalReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
CombustorTargetReviewer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


COMBUSTOR_TARGET_COMPARISON_INSTRUCTION = (
    "You are the existing FlowIntentBench Flow Expert reviewing a Combustor "
    "target redesign. Compare every supplied target candidate using only the "
    "visible evidence. Consider physical meaning, target stability, observable "
    "support, meaningful O variation, F1/F2 finding support, and whether the "
    "candidate is reproducibly representable. Maximum density and a density "
    "level-set are different targets unless the evidence establishes otherwise; "
    "do not merge them as alternative O choices. Return exactly one JSON object "
    "with family_id, recommended_target_id (a candidate_id or null), "
    "recommendation_status (RECOMMENDED_PROVISIONAL_TARGET or UNRESOLVED), "
    "human_approval_required=true, evidence_sufficiency (SUFFICIENT or "
    "INSUFFICIENT), candidate_comparisons (one comparison for each candidate), "
    "evidence_ids, and a non-empty rationale. A recommendation is provisional "
    "only: it must never mutate or inherit a case, Ground Truth, or SRAC. If "
    "the evidence cannot support a coherent four-condition family, return null "
    "and UNRESOLVED."
)


DATASET_POLICIES: dict[str, dict[str, Any]] = {
    "Blunt_Fin": {
        "family_id": "blunt_fin__high_speed_region",
        "case_ids": {
            "O1-F1": "blunt_fin_o1_f1",
            "O2-F1": "blunt_fin_o2_f1",
            "O3-F1": "blunt_fin_o3_f1",
            "O1-F2": "blunt_fin_o1_f2",
        },
        "required_target_policy": "high-speed flow regions in the blunt-fin flow field",
        "prohibited_claims": ["separation", "wake direction"],
    },
    "Carotid": {
        "family_id": "carotid__high_speed_region",
        "case_ids": {
            "O1-F1": "carotid_o1_f1",
            "O2-F1": "carotid_o2_f1",
            "O3-F1": "carotid_o3_f1",
            "O1-F2": "carotid_o1_f2",
        },
        "required_target_policy": "velocity or high-speed blood-flow regions",
        "prohibited_claims": [
            "wall shear",
            "centreline profile",
            "anatomical direction",
        ],
    },
    "Combustor": {
        "family_id": "combustor_density_redesign_pending_human_selection",
        "case_ids": {},
        "required_target_policy": "one evidence-supported stable density target",
        "prohibited_claims": ["most prominent density feature"],
        "redesign_required": True,
    },
    "FireFlow": {
        "family_id": "fireflow__high_speed_region",
        "case_ids": {
            "O1-F1": "fireflow_o1_f1",
            "O2-F1": "fireflow_o2_f1",
            "O3-F1": "fireflow_o3_f1",
            "O1-F2": "fireflow_o1_f2",
        },
        "required_target_policy": "velocity or high-speed flow regions",
        "prohibited_claims": ["temperature from t", "species from mfrac"],
    },
    "Kitchen": {
        "family_id": "kitchen_concentration_heterogeneity",
        "case_ids": {
            "O1-F1": "kitchen_concentration_heterogeneity_o1_f1",
            "O2-F1": "kitchen_concentration_heterogeneity_o2_f1",
            "O3-F1": "kitchen_concentration_heterogeneity_o3_f1",
            "O1-F2": "kitchen_concentration_heterogeneity_o1_f2",
        },
        "required_target_policy": "spatial concentration heterogeneity",
        "prohibited_claims": ["mixed concentration-and-turbulence target"],
    },
    "NASA_LOx_Post": {
        "family_id": "nasa_lox_post__high_speed_region",
        "case_ids": {
            "O1-F1": "nasa_lox_post_o1_f1",
            "O2-F1": "nasa_lox_post_o2_f1",
            "O3-F1": "nasa_lox_post_o3_f1",
            "O1-F2": "nasa_lox_post_o1_f2",
        },
        "required_target_policy": "velocity or high-speed flow regions",
        "prohibited_claims": ["wake direction", "separation", "wall-relative flow"],
    },
    "Office": {
        "family_id": "office__high_speed_region",
        "case_ids": {
            "O1-F1": "office_speed_zones_o1_f1",
            "O2-F1": "office_speed_zones_o2_f1",
            "O3-F1": "office_speed_zones_o3_f1",
            "O1-F2": "office_speed_zones_o1_f2",
        },
        "required_target_policy": "ventilation or high-speed flow regions",
        "prohibited_claims": ["unsupported pressure claim", "unsupported direction"],
    },
}


CONDITION_SCIENTIFIC_REVIEW_INSTRUCTION = (
    "You are the existing FlowIntentBench evidence-conditioned Flow Expert. Review exactly "
    "one candidate condition using only the visible evidence. Different scientifically valid "
    "Operationalizations may produce different G(O), rankings, selected features, and numeric "
    "results; that disagreement is not target drift. Target invariance means preservation of "
    "the scientific object and high-level target meaning while allowing consequential O "
    "choices. Return exactly one JSON object with case_id, condition, scientific_target_support, "
    "fixed_o_support, unresolved_o_space_support, finding_responsibility_support, "
    "materialization_support, target_invariance_judgment, evidence_sufficiency, "
    "recommended_action, blocking_reason_codes, evidence_gaps, evidence_ids, rationale, and "
    "the five support dimensions scientific_target_supported, dataset_observable_supported, "
    "execution_materialization_supported, evaluation_contract_supported, and "
    "evidence_complete_for_release (also grouped under support_dimensions), and "
    "result_equivalence_used_as_predicate=false. Support fields use SUPPORTED, UNSUPPORTED, "
    "NOT_ESTABLISHED, or NOT_APPLICABLE. target_invariance_judgment uses "
    "SAME_TARGET_SUPPORTED, TARGET_DRIFT, NOT_ESTABLISHED, or NOT_APPLICABLE. "
    "evidence_sufficiency is SUFFICIENT or INSUFFICIENT. recommended_action is RETAIN, "
    "REQUEST_EVIDENCE, REVISE_O_SPACE, or OMIT. REVISE_O_SPACE is valid only for O2/O3; "
    "an O1 evidence deficiency uses REQUEST_EVIDENCE or OMIT. For O1, set both "
    "unresolved_o_space_support and target_invariance_judgment exactly to NOT_APPLICABLE. "
    "For O3, set fixed_o_support exactly to NOT_APPLICABLE. A null "
    "materialization_declared_available means not declared, not unavailable; judge it as "
    "NOT_ESTABLISHED unless other visible evidence resolves support. For F2, the respondent "
    "selects an adequate scientifically relevant characterization; do not require the fixed O "
    "to preselect an optional characterization role. Judge whether the visible data and selected "
    "feature can support at least one supplied adequate-core route. Do not use Ground Truth or historical model "
    "results as scientific evidence. Do not mutate a case or execute SCQ, SRAC, or evaluation."
)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_inventory(repository_root: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(
        repository_root
        / "artifacts/reference/scientific_portfolio/concept_family_inventory.json",
        {},
    )
    return {
        str(row.get("family_id")): dict(row)
        for row in value.get("families", ())
        if isinstance(row, Mapping) and str(row.get("family_id", "")).strip()
    }


def _capability_profiles(repository_root: Path) -> dict[str, dict[str, Any]]:
    value = _read_json(
        repository_root
        / "artifacts/reference/concept_expansion_phase1/dataset_capability_profiles.json",
        [],
    )
    rows = value.get("profiles", ()) if isinstance(value, Mapping) else value
    return {
        str(row.get("dataset_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("dataset_id", "")).strip()
    }


def _evidence_bundle(
    repository_root: Path, dataset_id: str, family_id: str
) -> dict[str, Any]:
    construction = repository_root / "datasets" / dataset_id / "construction"
    grounding = _read_json(
        repository_root
        / "artifacts/reference/scientific_portfolio/families"
        / family_id
        / "grounding_evidence.json",
        {},
    )
    legacy_grounding = (
        isinstance(grounding, Mapping)
        and str(grounding.get("source", "")) == "historical_28_case_manifest"
    )
    context_evidence = _read_json(construction / "context_evidence.json", [])
    operationalization_evidence = _read_json(
        construction / "operationalization_evidence.json", []
    )
    finding_evidence = _read_json(construction / "finding_evidence.json", [])
    scoped_ids = {
        str(item).strip()
        for item in grounding.get("evidence_ids", ())
        if str(item).strip()
    }
    if scoped_ids and isinstance(operationalization_evidence, list):
        operationalization_evidence = [
            row
            for row in operationalization_evidence
            if isinstance(row, Mapping)
            and str(row.get("evidence_id", "")) in scoped_ids
        ]
    visible_evidence_rows = [
        *(
            context_evidence
            if isinstance(context_evidence, list)
            else []
        ),
        *(
            operationalization_evidence
            if isinstance(operationalization_evidence, list)
            else []
        ),
        *(finding_evidence if isinstance(finding_evidence, list) else []),
    ]
    visible_source_ids = {
        str(row.get("source_id", ""))
        for row in visible_evidence_rows
        if isinstance(row, Mapping) and str(row.get("source_id", "")).strip()
    }
    source_records = _read_json(construction / "sources.json", [])
    if isinstance(source_records, Mapping) and isinstance(
        source_records.get("sources"), list
    ):
        source_records = {
            **dict(source_records),
            "sources": [
                row
                for row in source_records["sources"]
                if isinstance(row, Mapping)
                and str(row.get("source_id", "")) in visible_source_ids
            ],
        }
    return {
        "dataset_context": _read_json(construction / "dataset_context.json", {}),
        "context_evidence": context_evidence,
        "operationalization_evidence": operationalization_evidence,
        "finding_evidence": finding_evidence,
        "source_records": source_records,
        "family_grounding_evidence": grounding,
        "family_scoped_operationalization_evidence_ids": sorted(scoped_ids),
        "family_grounding_admissible_for_scientific_support": not legacy_grounding,
        "legacy_ground_truth_used_as_scientific_evidence": False,
        "candidate_reference_results_used_as_scientific_evidence": False,
    }


def _known_evidence_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "evidence_id" and str(item).strip():
                result.add(str(item).strip())
            result.update(_known_evidence_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_known_evidence_ids(item))
    return result


def _semantic_visibility(
    projection: Mapping[str, Any], dataset_context: Mapping[str, Any]
) -> dict[str, Any]:
    return build_semantic_visibility_classification(
        projection,
        dataset_context=dataset_context,
        presentation_support_placements={
            "dataset_field_mapping": "DATASET_CONTEXT_VISIBLE",
            "exact_coordinate_capture": "BACKEND_ONLY",
            "field_identifier_capture": "BACKEND_ONLY",
        },
    )


def _dataset_context_for_authoring(
    record: Mapping[str, Any], capability: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "case_context": copy.deepcopy(dict(record.get("case_context") or {})),
        "dataset_capability_profile": copy.deepcopy(dict(capability)),
        "presentation_support": {
            "backend_field_names_are_context_not_scientific_choices": True,
            "exact_coordinates_or_identifiers_may_be_recorded_by_the_interface": True,
        },
    }


def build_condition_scientific_review_packets(
    repository_root: str | Path,
) -> list[dict[str, Any]]:
    """Build 24 GT-free, condition-scoped review packets for existing identities."""

    root = Path(repository_root).resolve()
    families = _family_inventory(root)
    capabilities = _capability_profiles(root)
    packets: list[dict[str, Any]] = []
    for dataset_id, policy in DATASET_POLICIES.items():
        if policy.get("redesign_required"):
            continue
        family_id = str(policy["family_id"])
        family = families.get(family_id)
        if not isinstance(family, Mapping):
            raise ValueError(f"scientific family not found: {family_id}")
        for condition in CONDITIONS:
            case_id = str(policy["case_ids"][condition])
            record = load_case_record(root, case_id)
            projection = harden_portfolio_scientific_projection(
                build_scientific_semantic_projection(
                    record, family_metadata=family
                ),
                condition=condition,
            )
            evidence = _evidence_bundle(root, dataset_id, family_id)
            metadata = record.get("metadata") or {}
            representability = metadata.get("evaluation_representability_contract")
            representability = (
                representability if isinstance(representability, Mapping) else {}
            )
            materialization = representability.get("materialization")
            materialization = (
                materialization if isinstance(materialization, Mapping) else {}
            )
            evaluation_contract = representability
            declared_materialization: bool | None = None
            if "available" in materialization:
                declared_materialization = materialization.get("available") is True
            packet = {
                "review_packet_version": SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION,
                "task": "CONDITION_SCIENTIFIC_ELIGIBILITY_ADVISORY",
                "case_id": case_id,
                "dataset_id": dataset_id,
                "family_id": family_id,
                "condition": condition,
                "dataset_policy": {
                    "required_target_policy": policy["required_target_policy"],
                    "prohibited_claims": list(policy["prohibited_claims"]),
                },
                "canonical_semantic_projection": projection,
                "semantic_contract_sha256": semantic_contract_sha256(projection),
                "dataset_capability_profile": capabilities.get(dataset_id, {}),
                "scientific_evidence": evidence,
                "evaluation_contract": copy.deepcopy(evaluation_contract),
                "support_dimension_definitions": {
                    "scientific_target_supported": "The proposed scientific object is grounded by visible evidence.",
                    "dataset_observable_supported": "The supplied dataset exposes the observable needed by the target.",
                    "execution_materialization_supported": "The condition can be materialized into an executable analysis route.",
                    "evaluation_contract_supported": "The existing evaluation representability contract can adjudicate the requested findings.",
                    "evidence_complete_for_release": "The visible evidence packet is sufficient for formal release.",
                },
                "materialization_declared_available": declared_materialization,
                "target_invariance_contract": {
                    "scientific_object_invariance_required": True,
                    "target_meaning_invariance_required": True,
                    "legitimate_consequential_o_choice_allowed": True,
                    "different_g_of_o_allowed": True,
                    "ranking_agreement_required": False,
                    "selected_feature_agreement_required": False,
                    "numerical_agreement_required": False,
                },
                "condition_scoped_review_required": True,
                "family_wide_failure_propagation_forbidden": True,
                "ground_truth_visible": False,
                "historical_ground_truth_is_scientific_evidence": False,
            }
            packets.append(packet)
    if len(packets) != 24:
        raise ValueError(f"expected 24 existing-identity packets, got {len(packets)}")
    return packets


_HIGH_SPEED_DATASETS = {
    "Blunt_Fin",
    "Carotid",
    "FireFlow",
    "NASA_LOx_Post",
    "Office",
}


def _replace_dimension_meaning(
    projection: dict[str, Any], dimension: str, meaning: str
) -> None:
    for row in projection.get("resolved_operationalization", ()):
        if isinstance(row, dict) and row.get("dimension_id") == dimension:
            row["normalized_meaning"] = meaning


def _replace_open_dimension_meaning(
    projection: dict[str, Any], dimension: str, meaning: str
) -> None:
    for row in projection.get("unresolved_operationalization_dimensions", ()):
        if isinstance(row, dict) and row.get("dimension_id") == dimension:
            row["normalized_meaning"] = meaning
            row["must_remain_open"] = True


def harden_portfolio_scientific_projection(
    raw_projection: Mapping[str, Any], *, condition: str
) -> dict[str, Any]:
    """Apply the final, authored scientific corrections for the 28-case portfolio.

    This function is deliberately a construction-time compiler pass.  It does
    not inspect a model response, Ground Truth value, or evaluator result.  Its
    only inputs are the existing semantic projection and controlled condition.
    The pass makes reader/topology facts and executable O boundaries explicit
    before a human-facing question is authored.
    """

    projection = copy.deepcopy(dict(raw_projection))
    projection["scientific_semantic_projection_version"] = (
        "scientific-semantic-projection-v3"
    )
    dataset_id = str(projection.get("dataset_id", ""))
    condition = condition.upper().replace("_", "-")

    if dataset_id in _HIGH_SPEED_DATASETS:
        structured = dataset_id != "FireFlow"
        connectivity = (
            "treat retained grid points adjacent by one structured-grid index "
            "step along exactly one axis as one six-neighbor connected speed region"
            if structured
            else "treat retained points that share an unstructured-mesh cell as "
            "one connected speed region"
        )
        _replace_dimension_meaning(projection, "feature_definition", connectivity)
        _replace_open_dimension_meaning(
            projection,
            "feature_definition",
            "select and state a connectivity definition for thresholded "
            "velocity-magnitude points that is executable on the supplied topology",
        )
        _replace_open_dimension_meaning(
            projection,
            "criterion",
            "select and state an explicit numeric threshold or percentile rule "
            "over finite non-zero speed values",
        )
        _replace_open_dimension_meaning(
            projection,
            "property_measure",
            "select and state the regional strength statistic",
        )
        _replace_open_dimension_meaning(
            projection,
            "aggregation_or_representation",
            "select and state how the selected region's location is represented",
        )
        # A fixed percentile convention is a scientific choice, not a parser
        # default.  Existing high-speed O1 clauses use the deterministic
        # linear convention; expose it in every model-visible projection (not
        # only the historical Blunt_Fin row) so the question, projection and
        # materializer share one explicit authority.
        for row in projection.get("resolved_operationalization", ()):
            if (
                isinstance(row, dict)
                and row.get("dimension_id") == "criterion"
                and "percentile" in str(row.get("normalized_meaning", ""))
            ):
                meaning = str(row["normalized_meaning"]).rstrip(".")
                if "linear quantile interpolation" not in meaning.casefold() and (
                    "nearest-rank" not in meaning.casefold()
                    and "nearest rank" not in meaning.casefold()
                ):
                    meaning += " using linear quantile interpolation"
                row["normalized_meaning"] = meaning
        scope = projection.setdefault("scientific_scope", {})
        conditions = list(scope.get("condition_constraints", ()) or ())
        observable = {
            "Carotid": "speed is the Euclidean magnitude of the stored vectors array",
            "Office": "speed is the Euclidean magnitude of the stored vectors array",
            "FireFlow": "speed is the Euclidean magnitude of the stored uvw array",
            "Blunt_Fin": "speed is the Euclidean magnitude of the reader-exposed Velocity array",
            "NASA_LOx_Post": "speed is the magnitude of the canonical reader-derived Velocity array",
        }[dataset_id]
        conditions.extend(
            [
                observable,
                "operationalizations remain within thresholded connected "
                "velocity-magnitude region analysis on the supplied topology",
            ]
        )
        scope["condition_constraints"] = list(dict.fromkeys(conditions))
        f2 = projection.get("finding_semantics", {}).get("adequate_core_semantics")
        if condition == "O1-F2" and isinstance(f2, dict):
            # F2 has a fixed strength/location core and an open, adjudicated
            # characterization role.  The reference extent is one valid role,
            # not the hidden answer.  Every role listed in the question is
            # eligible only after support, relevance and O/F consistency have
            # been adjudicated.
            characterization_roles = [
                "spatial_extent",
                "geometric_relation",
                "meaningful_comparison",
                "morphology",
                "orientation",
            ]
            f2["adequate_core_sets"] = [
                ["defining_strength_evidence", "principal_feature_location", role]
                for role in characterization_roles
            ]
            f2["alternative_role_groups"] = [{
                "group_id": "characterization",
                "min_required": 1,
                "roles": characterization_roles,
            }]

    if dataset_id == "Kitchen":
        projection["scientific_target"] = (
            "the stored gas-concentration field with the greatest "
            "distributional heterogeneity"
        )
        scope = projection.setdefault("scientific_scope", {})
        conditions = list(scope.get("condition_constraints", ()) or ())
        conditions.append(
            "the supplied c1 and c11-c17 concentration arrays are point-associated"
        )
        scope["condition_constraints"] = list(dict.fromkeys(conditions))
        _replace_dimension_meaning(
            projection,
            "feature_definition",
            "treat each point-associated c1 or c11-c17 scalar array as one "
            "gas-concentration field",
        )
        _replace_dimension_meaning(
            projection,
            "property_measure",
            "compare fields by cell-volume-weighted coefficient of variation, "
            "forming each cell concentration as the arithmetic mean of its vertex "
            "values and weighting positive-volume cells by cell volume",
        )
        _replace_open_dimension_meaning(
            projection,
            "feature_definition",
            "select and state whether heterogeneity is defined over the point "
            "population, derived cells, or structured six-neighbor point pairs",
        )
        _replace_open_dimension_meaning(
            projection,
            "property_measure",
            "select and state a heterogeneity statistic for the chosen population",
        )
        _replace_open_dimension_meaning(
            projection,
            "aggregation_or_representation",
            "select and state point-equal or positive-cell-volume weighting and "
            "report the selected stored field identifier",
        )
        semantics = projection.get("finding_semantics")
        if isinstance(semantics, dict):
            semantics["normalized_demand"] = (
                "identify the requested concentration field and report its stored "
                "identifier and heterogeneity statistic"
            )
            semantics["finding_goal"] = semantics["normalized_demand"]

    return projection


def _unwrap_review_output(raw: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = raw.get("parsed_result")
    if isinstance(parsed, Mapping):
        return parsed
    if "scientific_target_support" in raw or any(
        name in raw for name in SUPPORT_DIMENSION_NAMES
    ) or isinstance(raw.get("support_dimensions"), Mapping):
        return raw
    return None


def _review_invocation(raw: Mapping[str, Any]) -> dict[str, Any]:
    invocation = {
        key: copy.deepcopy(raw.get(key))
        for key in (
            "invocation_status",
            "provider",
            "model_id",
            "execution_mode",
            "live_model_calls",
            "agent_profile_id",
            "agent_profile_sha256",
            "model_family",
            "role",
            "role_profile_sha256",
            "visible_input_sha256",
            "timestamp",
        )
        if raw.get(key) is not None
    }
    invocation["raw_call_sha256"] = _json_sha256(raw)
    return invocation


_AUTHORING_SUCCESS_INVOCATION_STATUSES = frozenset(
    {"SUCCESS", "PASS", "COMPLETED", "COMPLETE"}
)
_AUTHORING_FAILURE_INVOCATION_STATUSES = frozenset(
    {
        "FAILED",
        "ERROR",
        "PROVIDER_ERROR",
        "PROVIDER_TIMEOUT",
        "TOOL_ERROR",
        "PARSE_FAILURE",
        "INPUT_REJECTED",
        "EXCEPTION",
    }
)


def _normalise_question_authoring_invocation(
    authored: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    run_live: bool,
) -> dict[str, Any]:
    """Return an explicit invocation envelope for one authoring attempt.

    ``author_human_questions`` deliberately keeps the author callback's raw
    response untouched.  Live responses already contain transport metadata,
    while small injected reviewers used by tests commonly return only the
    structured ``candidates`` object.  The matrix is the handoff boundary, so
    it normalizes both forms into the same candidate-level provenance contract.

    A successful injected callback is marked ``INJECTED_REVIEWER``.  This is
    still distinct from deterministic fallback (which remains ``NOT_RUN`` and
    can never establish genuine authored provenance), and makes the test path
    auditable without pretending it was a live provider call.
    """

    raw_call = authored.get("authoring_call")
    raw = dict(raw_call) if isinstance(raw_call, Mapping) else {}
    invocation = _review_invocation(raw)
    execution_mode = "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWER"
    invocation.setdefault("execution_mode", execution_mode)
    authored_status = str(authored.get("status", "")).strip().upper()
    raw_invocation_status = str(raw.get("invocation_status", "")).strip().upper()
    transport_status = str(raw.get("status", "")).strip().upper()
    candidates = authored.get("candidates")
    has_candidates = isinstance(candidates, list) and bool(candidates)

    # Never overwrite an explicit transport status.  A provider failure with a
    # malformed/partial parsed payload must remain excluded from genuine counts.
    if raw_invocation_status:
        invocation_status = raw_invocation_status
    elif transport_status in _AUTHORING_FAILURE_INVOCATION_STATUSES:
        invocation_status = transport_status
    elif authored_status == "PASS" and has_candidates:
        # Injected callbacks have no transport envelope.  Record their success
        # explicitly while retaining the non-live execution mode above.
        invocation_status = "SUCCESS"
        invocation["synthetic_invocation_record"] = True
    elif authored_status in {"INFRA_INVALID", "INCOMPLETE"}:
        invocation_status = "NOT_REPORTED"
    else:
        invocation_status = "NOT_REPORTED"
    invocation["invocation_status"] = invocation_status
    invocation["authoring_status"] = authored_status or None
    invocation["authoring_operation"] = "QUESTION_PRESENTATION_AUTHORING"
    invocation["candidate_count"] = len(candidates) if isinstance(candidates, list) else 0
    invocation["invocation_success"] = invocation_status in _AUTHORING_SUCCESS_INVOCATION_STATUSES
    invocation.setdefault("synthetic_invocation_record", False)
    if row.get("case_id") is not None:
        invocation["case_id"] = str(row.get("case_id"))
    if row.get("condition") is not None:
        invocation["condition"] = str(row.get("condition"))
    return invocation


def _support_dimensions(
    output: Mapping[str, Any], packet: Mapping[str, Any]
) -> tuple[dict[str, str], list[str], bool]:
    """Normalize the explicit five-dimensional support contract.

    Older injected reviewers use the original ``*_support`` fields.  They are
    still accepted and deterministically lifted into the new dimensions.  A
    reviewer may provide either the five top-level fields, ``support_dimensions``,
    or both; when both are present they must agree.  No scientific support is
    inferred from a Python value or from a historical Ground Truth artifact.
    """

    errors: list[str] = []
    legacy = {
        "scientific_target_supported": output.get("scientific_target_support"),
        "execution_materialization_supported": output.get("materialization_support"),
    }
    explicit = {
        name: output.get(name)
        for name in SUPPORT_DIMENSION_NAMES
        if name in output
    }
    raw_mapping = output.get("support_dimensions")
    mapping: dict[str, Any] = {}
    if raw_mapping is not None:
        if not isinstance(raw_mapping, Mapping):
            errors.append("SUPPORT_DIMENSIONS_NOT_OBJECT")
        else:
            mapping = dict(raw_mapping)
            if set(mapping) != set(SUPPORT_DIMENSION_NAMES):
                errors.append("SUPPORT_DIMENSIONS_FIELDS_INVALID")

    # Explicit values win.  Fill omitted dimensions from the legacy contract
    # and conservative packet facts so missing execution metadata cannot turn
    # into a target-unsupported judgment.
    target = explicit.get(
        "scientific_target_supported",
        mapping.get(
            "scientific_target_supported", legacy["scientific_target_supported"]
        ),
    )
    if target is None:
        target = "NOT_ESTABLISHED"
    if "dataset_observable_supported" in explicit:
        observable = explicit["dataset_observable_supported"]
    elif "dataset_observable_supported" in mapping:
        observable = mapping["dataset_observable_supported"]
    elif target == "UNSUPPORTED":
        observable = "UNSUPPORTED"
    else:
        observable = "NOT_ESTABLISHED"

    execution = explicit.get(
        "execution_materialization_supported",
        mapping.get(
            "execution_materialization_supported",
            legacy["execution_materialization_supported"],
        ),
    )
    if execution is None:
        execution = "NOT_ESTABLISHED"
    if "evaluation_contract_supported" in explicit:
        evaluation = explicit["evaluation_contract_supported"]
    elif "evaluation_contract_supported" in mapping:
        evaluation = mapping["evaluation_contract_supported"]
    else:
        # The packet can expose an evaluation representability contract through
        # the case metadata, but absence is deliberately unknown rather than
        # an unsupported scientific target.
        evaluation = "NOT_ESTABLISHED"
    evidence_sufficiency = output.get("evidence_sufficiency")
    if "evidence_complete_for_release" in explicit:
        evidence = explicit["evidence_complete_for_release"]
    elif "evidence_complete_for_release" in mapping:
        evidence = mapping["evidence_complete_for_release"]
    elif target == "UNSUPPORTED" or observable == "UNSUPPORTED":
        evidence = "UNSUPPORTED"
    elif evidence_sufficiency == "SUFFICIENT":
        evidence = "SUPPORTED"
    else:
        evidence = "NOT_ESTABLISHED"

    normalized = {
        "scientific_target_supported": target,
        "dataset_observable_supported": observable,
        "execution_materialization_supported": execution,
        "evaluation_contract_supported": evaluation,
        "evidence_complete_for_release": evidence,
    }
    for name, value in normalized.items():
        if value not in _SUPPORT_VALUES:
            errors.append(f"{name.upper()}_INVALID")
    for name, value in mapping.items():
        if name in normalized and name in explicit and explicit[name] != value:
            errors.append(f"{name.upper()}_MISMATCH")
    explicit_contract = bool(explicit or raw_mapping is not None)
    return (
        {name: str(value) for name, value in normalized.items()},
        sorted(set(errors)),
        explicit_contract,
    )


def _scientific_release_status(matrix_status: str) -> str:
    return {
        "CANDIDATE_FOR_HUMAN_REVIEW": "ELIGIBLE",
        "BLOCKED_PENDING_EVIDENCE": "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE_REVIEW": "BLOCKED_PENDING_O_SPACE",
        "NEW_CASE_PENDING_HUMAN_SELECTION": "NEW_CASE_REQUIRED",
        # An omitted identity needs a new scientifically supported target/case;
        # retain the precise omission reason separately on the condition row.
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED": "NEW_CASE_REQUIRED",
    }.get(matrix_status, "BLOCKED_PENDING_EVIDENCE")


def _question_candidate_status(matrix_status: str) -> str:
    return {
        "CANDIDATE_FOR_HUMAN_REVIEW": "READY_FOR_HUMAN_REVIEW",
        # A blocked release may still be a perfectly meaningful question for
        # human review.  Candidate status is therefore independent of release
        # status and is upgraded to READY after wording audits.
        "BLOCKED_PENDING_EVIDENCE": "READY_FOR_HUMAN_REVIEW",
        "BLOCKED_PENDING_O_SPACE_REVIEW": "READY_FOR_HUMAN_REVIEW",
        "NEW_CASE_PENDING_HUMAN_SELECTION": "NEEDS_TARGET_REDESIGN",
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED": "NEEDS_SCIENTIFIC_REVISION",
    }.get(matrix_status, "NEEDS_SCIENTIFIC_REVISION")


def _artifact_reuse_status(matrix_status: str, *, semantic_identity_changed: bool = False) -> str:
    if semantic_identity_changed:
        return "STALE"
    if matrix_status in {
        "NEW_CASE_PENDING_HUMAN_SELECTION",
        "OMITTED_SCIENTIFICALLY_UNSUPPORTED",
    }:
        return "REBUILD_REQUIRED"
    return "VALID"


def combustor_target_question_text(target: Mapping[str, Any]) -> str:
    """Return a natural-language question for an unselected density target.

    The target definition remains available in the candidate metadata and
    analysis context.  Only the principal sentence is deliberately phrased in
    scientific language, so storage names and exact representation details do
    not leak into the question itself.
    """

    supplied = (
        target.get("human_facing_question")
        or target.get("scientific_question")
        or target.get("question_text")
    )
    if isinstance(supplied, str) and supplied.strip():
        return supplied.strip()
    candidate_id = str(target.get("candidate_id", "")).strip().casefold()
    if candidate_id == "density_global_maximum_point":
        return (
            "Which location in the combustor flow has the highest density, "
            "and what density value should be reported there?"
        )
    if candidate_id == "density_level_set_surface":
        return "What spatial structure is revealed by a selected density level in the combustor flow?"
    return "What scientifically meaningful density structure can be characterized in the supplied combustor flow?"


def combustor_target_analysis_context(target: Mapping[str, Any]) -> dict[str, Any]:
    """Describe target-specific implementation obligations outside the main question."""

    return {
        "target_definition": str(target.get("scientific_meaning", "")).strip(),
        "operationalization_dimensions": copy.deepcopy(
            target.get("plausible_o_dimensions") or []
        ),
        "evidence_ids": copy.deepcopy(target.get("evidence_ids") or []),
        "evidence_basis": str(target.get("evidence_basis", "")).strip(),
        "backend_details_are_context_only": True,
    }


def _combustor_provisional_family_contract(
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the proposed four-condition contract for one redesign target.

    The contract is construction material presented to the Flow Expert.  It is
    not a scientific conclusion.  The pointwise global-maximum target is kept
    as a legitimate alternative, but it is explicitly marked unable to carry
    the required consequential O2/O3 variation without changing its target.
    """

    target_id = str(target.get("candidate_id", "")).strip()
    if target_id == "density_global_maximum_point":
        fixed = {
            "feature_definition": {
                "dimension_id": "feature_definition",
                "canonical_id": "provisional_density_grid_point",
                "normalized_meaning": (
                    "treat a finite stored-Density grid sample as one point feature"
                ),
            },
            "criterion": {
                "dimension_id": "criterion",
                "canonical_id": "provisional_global_density_extremum",
                "normalized_meaning": (
                    "select the point feature by the largest finite stored Density value"
                ),
            },
            "aggregation_or_representation": {
                "dimension_id": "aggregation_or_representation",
                "canonical_id": "provisional_exact_point_coordinate_value",
                "normalized_meaning": (
                    "represent the selected point by its stored coordinate and Density value"
                ),
            },
        }
        target_text = (
            "a prominent pointwise stored-Density feature in the annular-combustor flow field"
        )
        finding_entity = "POINT_FEATURE"
        location_role = "feature_location"
        level_role = "defining_density_value"
        selection_text = "identify one scientifically relevant pointwise density feature"
        # The exact global maximum is itself fixed by the criterion.  Opening
        # that criterion for O2/O3 would change the scientific target rather
        # than expose a valid alternative operationalization.  Keep this
        # candidate in the comparison surface, but fail closed if an agent
        # recommends it as the four-condition family target.
        four_condition_constructible = False
        constructibility_reasons = [
            "OPEN_CRITERION_WOULD_CHANGE_EXACT_GLOBAL_MAXIMUM_TARGET"
        ]
    elif target_id == "density_level_set_surface":
        fixed = {
            "feature_definition": {
                "dimension_id": "feature_definition",
                "canonical_id": "provisional_constant_density_connected_surface",
                "normalized_meaning": (
                    "treat each connected component of a constant-Density isosurface "
                    "as one candidate surface"
                ),
            },
            "criterion": {
                "dimension_id": "criterion",
                "canonical_id": "provisional_density_q90_largest_surface",
                "normalized_meaning": (
                    "use the 90th percentile of finite stored Density values with "
                    "linear quantile interpolation as the isovalue"
                ),
            },
            "aggregation_or_representation": {
                "dimension_id": "aggregation_or_representation",
                "canonical_id": "provisional_surface_centroid_area",
                "normalized_meaning": (
                    "select the connected component with the largest surface area and "
                    "represent it by its area-weighted centroid and surface area"
                ),
            },
        }
        target_text = (
            "connected components of a constant-Density isosurface in the "
            "annular-combustor density field"
        )
        finding_entity = "SURFACE_STRUCTURE"
        location_role = "spatial_location"
        level_role = "defining_density_level"
        selection_text = "select and characterize one connected density-isosurface component"
        four_condition_constructible = True
        constructibility_reasons = []
    else:
        return {
            "contract_status": "NOT_CONSTRUCTIBLE_AS_FOUR_CONDITION_FAMILY",
            "blocking_reason_codes": ["UNKNOWN_COMBUSTOR_TARGET_CANDIDATE"],
            "scientific_target": str(target.get("scientific_meaning", "")).strip(),
        }
    return {
        "contract_status": (
            "PROVISIONAL_CONSTRUCTIBLE_PENDING_HUMAN_APPROVAL"
            if four_condition_constructible
            else "NOT_CONSTRUCTIBLE_AS_FOUR_CONDITION_FAMILY"
        ),
        "four_condition_constructible": four_condition_constructible,
        "blocking_reason_codes": constructibility_reasons,
        "target_candidate_id": target_id,
        "family_id": f"combustor_{target_id}_provisional",
        "scientific_target": target_text,
        "scientific_scope": {
            "scope_constraints": ["the annular-combustor flow field"],
            "condition_constraints": [],
            "selection_constraints": [
                selection_text
            ],
        },
        "principal_operationalization_dimensions": [
            "feature_definition",
            "criterion",
            "aggregation_or_representation",
        ],
        "baseline_resolved_operationalization": fixed,
        "condition_open_dimensions": {
            "O1-F1": [],
            "O2-F1": ["criterion"],
            "O3-F1": [
                "feature_definition",
                "criterion",
                "aggregation_or_representation",
            ],
            "O1-F2": [],
        },
        "f1_fixed_finding_requirements": [
            {
                "category": "location",
                "normalized_meaning": (
                    "report the selected structure's representative location"
                ),
            },
            {
                "category": "quantity",
                "normalized_meaning": (
                    "report its defining Density isovalue"
                ),
            },
            {
                "category": "characterization",
                "normalized_meaning": "report its surface area",
            },
        ],
        "f2_adequate_core_semantics": {
            "scientific_entity_type": finding_entity,
            "mandatory_roles": [level_role],
            "supporting_roles": ["supporting_surface_statistic"],
            "adequate_core_sets": [
                [level_role, "surface_extent"],
                [level_role, location_role],
                [level_role, "shape_characterization"],
            ],
            "alternative_role_groups": [
                {
                    "group_id": "characterization",
                    "min_required": 1,
                    "roles": [
                        "surface_extent",
                        location_role,
                        "shape_characterization",
                    ],
                }
            ],
            "role_by_category": {
                "location": location_role,
                "quantity": level_role,
                "characterization": "shape_characterization",
            },
            "novel_role_policy": (
                "A scientifically relevant data-supported characterization may be "
                "reviewed during later formal construction."
            ),
        },
        "scientific_support_status": "PROVISIONAL_PENDING_FLOW_EXPERT_AND_HUMAN_REVIEW",
    }


def _combustor_condition_projection(
    target: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    """Materialize one provisional semantic projection without old artifacts."""

    contract = target.get("provisional_family_contract")
    if not isinstance(contract, Mapping) or contract.get("contract_status") != (
        "PROVISIONAL_CONSTRUCTIBLE_PENDING_HUMAN_APPROVAL"
    ):
        raise ValueError("recommended Combustor target has no constructible family contract")
    open_by_condition = contract.get("condition_open_dimensions")
    fixed_by_dimension = contract.get("baseline_resolved_operationalization")
    if not isinstance(open_by_condition, Mapping) or not isinstance(
        fixed_by_dimension, Mapping
    ):
        raise ValueError("Combustor provisional family contract is incomplete")
    principal = [str(item) for item in contract.get("principal_operationalization_dimensions", ())]
    unresolved = [str(item) for item in open_by_condition.get(condition, ())]
    if not principal or set(unresolved) - set(principal):
        raise ValueError("Combustor provisional O responsibility is invalid")
    resolved = [
        copy.deepcopy(dict(fixed_by_dimension[dimension]))
        for dimension in principal
        if dimension not in unresolved
        and isinstance(fixed_by_dimension.get(dimension), Mapping)
    ]
    if {str(item.get("dimension_id", "")) for item in resolved} != (
        set(principal) - set(unresolved)
    ):
        raise ValueError("Combustor provisional fixed-O contract is incomplete")
    # The target contract is the single source of truth for F1/F2 role names.
    # Keep this lookup local so a projection can never accidentally inherit a
    # role from another target or from a historical case.
    role_by_category = contract.get("f2_adequate_core_semantics", {}).get(
        "role_by_category", {}
    )
    if not isinstance(role_by_category, Mapping):
        role_by_category = {}
    location_role = str(role_by_category.get("location") or "feature_location")
    level_role = str(role_by_category.get("quantity") or "defining_density_value")
    if condition not in CONDITIONS:
        raise ValueError("unknown Combustor condition")
    finding_mode = condition.split("-", 1)[1]
    if finding_mode == "F1":
        finding_semantics = {
            "normalized_demand": (
                "identify the requested density feature and report its "
                "representative location, defining density level, and spatial extent"
            ),
            "finding_goal": (
                "identify and report the requested density feature"
            ),
            "fixed_finding_requirements": copy.deepcopy(
                contract.get("f1_fixed_finding_requirements", [])
            ),
            "required_finding_roles": [
                location_role,
                level_role,
                "surface_extent",
            ],
        }
    else:
        finding_semantics = {
            "normalized_demand": (
                "characterize the selected density feature using an adequate "
                "scientifically relevant set of findings"
            ),
            "finding_goal": (
                "select an adequate characterization of the density feature"
            ),
            "adequate_core_semantics": copy.deepcopy(
                contract.get("f2_adequate_core_semantics", {})
            ),
        }
    projection = {
        "scientific_semantic_projection_version": "provisional-combustor-target-v1",
        "dataset_id": "Combustor",
        "family_id": (
            "combustor_" + str(target.get("candidate_id", "density_target"))
            + "_provisional"
        ),
        "concept_id": (
            "combustor_" + str(target.get("candidate_id", "density_target"))
            + "_provisional"
        ),
        "scientific_target": str(contract.get("scientific_target", "")).strip(),
        "scientific_scope": copy.deepcopy(contract.get("scientific_scope", {})),
        "operationalization_responsibility": (
            "user_specified"
            if not unresolved
            else "partially_specified"
        ),
        "principal_operationalization_dimensions": principal,
        "resolved_operationalization": resolved,
        "unresolved_operationalization_dimensions": [
            {
                "dimension_id": dimension,
                "normalized_meaning": (
                    "select a scientifically defensible "
                    + dimension.replace("_", " ")
                    + " for the selected density target"
                ),
                "must_remain_open": True,
            }
            for dimension in unresolved
        ],
        "finding_responsibility": finding_mode,
        "finding_semantics": finding_semantics,
        "observable_scientific_semantics": {
            "required_capabilities": {
                "SCALAR_FIELD_SEMANTICS:Density": True,
                "SPATIAL_COORDINATES": True,
                "ISOSURFACE_EXTRACTION": True,
            },
            "scientific_semantic_capabilities": {},
        },
        "verification_semantics": {
            "deterministic_verification_required_claim_types": [],
            "semantic_adjudication_required": True,
        },
        "provisional_target_status": "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL",
    }
    # The density-family projection is already authored here rather than
    # derived from a historical Case, so apply its final executable envelope
    # directly.  This remains a semantic compiler pass, not a runtime guess.
    if condition == "O2-F1":
        for row in projection["unresolved_operationalization_dimensions"]:
            if row.get("dimension_id") == "criterion":
                row["normalized_meaning"] = (
                    "select and state a finite stored-Density isovalue rule; "
                    "component selection remains largest surface area"
                )
    elif condition == "O3-F1":
        meanings = {
            "feature_definition": (
                "select and state how constant-Density isosurface components are "
                "formed from the supplied grid"
            ),
            "criterion": "select and state a finite stored-Density isovalue rule",
            "aggregation_or_representation": (
                "select and state a component-ranking rule and representative "
                "location derived from the component geometry"
            ),
        }
        for row in projection["unresolved_operationalization_dimensions"]:
            dimension = str(row.get("dimension_id", ""))
            row["normalized_meaning"] = meanings[dimension]
    projection["scientific_semantic_projection_version"] = (
        "scientific-semantic-projection-v3"
    )
    return projection


def _combustor_question_candidate(
    target: Mapping[str, Any], condition: str, index: int
) -> dict[str, Any]:
    """Render one non-canonical Combustor target proposal for review.

    Target proposals intentionally have no semantic-contract hash or case id:
    selecting one is a human scientific action that must precede any case,
    Ground Truth, or SRAC construction.
    """

    candidate_id = str(target.get("candidate_id", "")).strip()
    meaning = str(target.get("scientific_meaning", "")).strip()
    if not candidate_id or not meaning:
        raise ValueError("Combustor target candidate requires candidate_id and scientific_meaning")
    question = combustor_target_question_text(target)
    analysis_context = combustor_target_analysis_context(target)
    text = "\n".join(
        [
            "Scientific Question",
            question,
            "",
            "Analysis Context",
            f"Candidate target: {meaning}",
            "Use the target-specific criterion and representation described in this context when characterizing the result.",
        ]
    )
    return {
        "candidate_id": f"combustor::{condition}::{candidate_id}",
        "target_candidate_id": candidate_id,
        "style": "TARGET_REDESIGN_PROPOSAL",
        "scientific_question": question,
        "question_text": question,
        "model_visible_text": text,
        "scientific_target": meaning,
        "scientific_scope": "in the supplied Combustor flow data",
        "fixed_operationalization": [],
        "unresolved_operationalization_dimensions": copy.deepcopy(
            target.get("plausible_o_dimensions", [])
        ),
        "finding_responsibility": "PENDING_HUMAN_TARGET_SELECTION",
        "semantic_contract_sha256": None,
        "target_selection_required": True,
        "automatic_adoption": False,
        "evidence_ids": copy.deepcopy(target.get("evidence_ids", [])),
        "analysis_context": analysis_context,
        "model_visible_analysis_context": copy.deepcopy(analysis_context),
        "evidence_status": "TARGET_REDESIGN_EVIDENCE_PENDING_HUMAN_SELECTION",
        "scientific_release_status": "NEW_CASE_REQUIRED",
        "artifact_reuse_status": "REBUILD_REQUIRED",
        "question_candidate_status": "NEEDS_TARGET_REDESIGN",
        "flow_expert_comments": str(target.get("distinct_from_rejected_target", "")),
    }


def _annotate_question_candidates(
    candidates: Sequence[Mapping[str, Any]], row: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach review metadata without changing the frozen semantic projection."""

    result: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        # Keep provenance on the candidate itself.  Older matrix artifacts
        # stored this only on the condition row; copying the explicit envelope
        # here prevents downstream portfolio counting from depending on a
        # lossy row-level join.
        if not isinstance(item.get("question_authoring_invocation"), Mapping):
            row_invocation = row.get("question_authoring_invocation")
            if isinstance(row_invocation, Mapping):
                item["question_authoring_invocation"] = copy.deepcopy(row_invocation)
        if not str(item.get("candidate_source") or "").strip():
            mode = str(item.get("authoring_mode") or item.get("generation_mode") or "").strip().upper()
            if "AUTHORING" in mode:
                item["candidate_source"] = "FLOW_EXPERT_PRESENTATION_AUTHORING"
            elif "FALLBACK" in mode or "REALIZATION" in mode:
                item["candidate_source"] = "DETERMINISTIC_REVIEW_FALLBACK"
        provenance_invocation = item.get("question_authoring_invocation")
        provenance_status = (
            str(provenance_invocation.get("invocation_status", "")).strip().upper()
            if isinstance(provenance_invocation, Mapping)
            else ""
        )
        provenance_mode = (
            str(provenance_invocation.get("execution_mode", "")).strip().upper()
            if isinstance(provenance_invocation, Mapping)
            else ""
        )
        source_upper = str(item.get("candidate_source") or "").strip().upper()
        item.setdefault(
            "authoring_provenance",
            {
                "candidate_source": source_upper or None,
                "authoring_mode": item.get("authoring_mode")
                or item.get("generation_mode"),
                "invocation_status": provenance_status or None,
                "invocation_verified": provenance_status
                in _AUTHORING_SUCCESS_INVOCATION_STATUSES
                and not (
                    provenance_mode.startswith("NOT_RUN")
                    or "DETERMINISTIC" in provenance_mode
                ),
                "deterministic_fallback": "FALLBACK" in source_upper
                or "DETERMINISTIC" in source_upper,
            },
        )
        item.setdefault("question_candidate_status", row.get("QUESTION_CANDIDATE_STATUS"))
        item.setdefault("scientific_target", row.get("scientific_target"))
        item.setdefault("scientific_scope", row.get("scientific_scope"))
        item.setdefault("fixed_operationalization", copy.deepcopy(row.get("fixed_operationalization", [])))
        item.setdefault(
            "unresolved_operationalization_dimensions",
            copy.deepcopy(row.get("unresolved_operationalization_dimensions", [])),
        )
        item.setdefault("finding_responsibility", copy.deepcopy(row.get("finding_responsibility")))
        item.setdefault("flow_expert_comments", copy.deepcopy(row.get("condition_scientific_review", {})))
        item.setdefault("evidence_status", row.get("evidence_sufficiency"))
        item.setdefault("scientific_release_status", row.get("SCIENTIFIC_RELEASE_STATUS"))
        item.setdefault("artifact_reuse_status", row.get("ARTIFACT_REUSE_STATUS"))
        item.setdefault("semantic_contract_sha256", row.get("semantic_contract_sha256"))
        # Annotation is part of the persisted candidate envelope.  Recompute
        # the envelope digest after attaching it; otherwise a later fidelity
        # audit would (correctly) see a stale hash whenever row-level review
        # metadata was added.
        item["presentation_sha256"] = canonical_json_sha256(
            {key: value for key, value in item.items() if key != "presentation_sha256"}
        )
        result.append(item)
    return result


def _deterministic_question_candidates(
    root: Path,
    row: Mapping[str, Any],
    capabilities: Mapping[str, Mapping[str, Any]],
    candidate_count: int,
) -> dict[str, Any]:
    """Produce reviewable wording for a slot without a live author call.

    This path is deliberately presentation-only.  It can expose a blocked or
    unknown scientific condition to a curator, but it never changes release
    eligibility or creates a downstream artifact.
    """

    case_id = str(row.get("case_id") or row.get("_authoring_case_id") or "").strip()
    if not case_id or not isinstance(row.get("canonical_semantic_projection"), Mapping):
        return {"question_candidates": [], "question_candidate_generation_error": "NO_CANONICAL_IDENTITY"}
    if row.get("_authoring_context"):
        context = copy.deepcopy(row.get("_authoring_context"))
    else:
        record = load_case_record(root, case_id)
        context = _dataset_context_for_authoring(
            record, capabilities.get(str(row.get("dataset_id")), {})
        )
    projection = row["canonical_semantic_projection"]
    authored = realize_human_questions(
        projection,
        case_id=case_id,
        condition=str(row.get("condition", "")),
        candidate_count=candidate_count,
        semantic_contract_sha256=str(row.get("semantic_contract_sha256", "")),
    )
    packet = build_question_presentation_authoring_packet(
        projection,
        dataset_context=context,
        candidate_count=candidate_count,
        case_id=case_id,
        condition=str(row.get("condition", "")),
        semantic_visibility=row.get("semantic_visibility"),
        semantic_contract_sha256=str(row.get("semantic_contract_sha256", "")),
    )
    audited: list[dict[str, Any]] = []
    for candidate in authored.get("candidates", ()):
        candidate_copy = copy.deepcopy(dict(candidate))
        # Validate the deterministic realization against the unchanged contract
        # and run the existing human-presentation lint; failures remain visible
        # to the reviewer rather than deleting the candidate.
        fidelity = validate_question_semantic_fidelity(
            projection,
            candidate_copy,
            semantic_contract_sha256=str(row.get("semantic_contract_sha256", "")),
        )
        hccq = audit_human_presentation(_hccq_shape(candidate_copy, packet))
        candidate_copy["post_rewrite_semantic_fidelity"] = fidelity
        candidate_copy["hccq_presentation"] = hccq
        candidate_copy["generation_mode"] = "DETERMINISTIC_SEMANTIC_REALIZATION"
        # Slot-scoped ids avoid accidental joins across conditions while the
        # semantic hash remains exactly the canonical hash.
        candidate_copy["candidate_id"] = (
            f"{case_id}::{candidate_copy.get('candidate_id', 'presentation')}"
        )
        candidate_copy["presentation_sha256"] = canonical_json_sha256(
            {key: value for key, value in candidate_copy.items() if key != "presentation_sha256"}
        )
        audited.append(candidate_copy)
    return {
        "question_candidates": audited,
        "primary_human_facing_question": (
            audited[0].get("model_visible_text") if audited else None
        ),
        "primary_candidate_id": audited[0].get("candidate_id") if audited else None,
        "primary_candidate_nomination": {
            "status": "DETERMINISTIC_CANDIDATE_FOR_HUMAN_REVIEW" if audited else "NO_CANDIDATE",
            "primary_candidate_id": audited[0].get("candidate_id") if audited else None,
            "primary_model_visible_text": audited[0].get("model_visible_text") if audited else None,
            "nomination_only": True,
            "human_final_selection_required": True,
            "human_question_selected": False,
        },
        "question_candidate_generation": "DETERMINISTIC_SEMANTIC_REALIZATION",
        "question_candidate_generation_error": None,
        # The wording-author agent was not invoked on this path.  Keep the
        # historical invocation marker for callers that distinguish authoring
        # transport from deterministic candidate generation.
        "question_authoring_invocation": {
            "execution_mode": "NOT_RUN",
            "invocation_status": "NOT_RUN_SLOT_NOT_READY",
            "raw_call_sha256": None,
        },
    }


def validate_condition_scientific_review(
    raw: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one advisory component review without deriving cross-condition state."""

    output = _unwrap_review_output(raw)
    if output is None:
        return {
            "status": "INCOMPLETE",
            "errors": ["NO_PARSED_CONDITION_REVIEW"],
            "normalized_review": {},
            "review_invocation": _review_invocation(raw),
        }
    required = {
        "case_id",
        "condition",
        "scientific_target_support",
        "fixed_o_support",
        "unresolved_o_space_support",
        "finding_responsibility_support",
        "materialization_support",
        "target_invariance_judgment",
        "evidence_sufficiency",
        "recommended_action",
        "blocking_reason_codes",
        "evidence_gaps",
        "evidence_ids",
        "rationale",
        "result_equivalence_used_as_predicate",
    }
    errors: list[str] = []
    allowed_fields = required | set(SUPPORT_DIMENSION_NAMES) | {"support_dimensions"}
    if not required.issubset(set(output)):
        errors.append("CONDITION_REVIEW_FIELDS_INVALID")
    unknown_fields = set(output) - allowed_fields
    if unknown_fields:
        errors.append(
            "CONDITION_REVIEW_UNKNOWN_FIELDS:" + ",".join(sorted(unknown_fields))
        )
    if str(output.get("case_id", "")) != str(packet.get("case_id", "")):
        errors.append("CASE_ID_MISMATCH")
    condition = str(packet.get("condition", ""))
    if str(output.get("condition", "")) != condition:
        errors.append("CONDITION_MISMATCH")
    support_fields = (
        "scientific_target_support",
        "fixed_o_support",
        "unresolved_o_space_support",
        "finding_responsibility_support",
        "materialization_support",
    )
    for field in support_fields:
        if output.get(field) not in _SUPPORT_VALUES:
            errors.append(f"{field.upper()}_INVALID")
    if output.get("target_invariance_judgment") not in _INVARIANCE_VALUES:
        errors.append("TARGET_INVARIANCE_JUDGMENT_INVALID")
    if output.get("evidence_sufficiency") not in {"SUFFICIENT", "INSUFFICIENT"}:
        errors.append("EVIDENCE_SUFFICIENCY_INVALID")
    if output.get("recommended_action") not in _ACTIONS:
        errors.append("RECOMMENDED_ACTION_INVALID")
    if output.get("result_equivalence_used_as_predicate") is not False:
        errors.append("RESULT_EQUIVALENCE_PREDICATE_FORBIDDEN")
    reason_codes = output.get("blocking_reason_codes")
    gaps = output.get("evidence_gaps")
    evidence_ids = output.get("evidence_ids")
    if not isinstance(reason_codes, list) or not all(
        isinstance(item, str) and item.strip() for item in reason_codes
    ):
        errors.append("BLOCKING_REASON_CODES_INVALID")
        reason_codes = []
    if set(reason_codes) & _FORBIDDEN_RESULT_EQUIVALENCE_REASONS:
        errors.append("RESULT_EQUIVALENCE_REASON_FORBIDDEN")
    if not isinstance(gaps, list) or not all(
        isinstance(item, str) and item.strip() for item in gaps
    ):
        errors.append("EVIDENCE_GAPS_INVALID")
        gaps = []
    if output.get("evidence_sufficiency") == "INSUFFICIENT" and not gaps:
        errors.append("SPECIFIC_EVIDENCE_GAP_REQUIRED")
    if not isinstance(evidence_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in evidence_ids
    ):
        errors.append("EVIDENCE_IDS_INVALID")
        evidence_ids = []
    if output.get("evidence_sufficiency") == "SUFFICIENT" and not evidence_ids:
        errors.append("EVIDENCE_CITATION_REQUIRED_FOR_SUFFICIENT_REVIEW")
    known_ids = _known_evidence_ids(packet.get("scientific_evidence", {}))
    if set(evidence_ids) - known_ids:
        errors.append("UNKNOWN_EVIDENCE_ID")
    rationale = str(output.get("rationale", "")).strip()
    if not rationale:
        errors.append("RATIONALE_REQUIRED")

    support_dimensions, support_errors, support_dimensions_explicit = _support_dimensions(
        output, packet
    )
    errors.extend(support_errors)

    o_mode = condition.split("-", 1)[0]
    if o_mode == "O1":
        if output.get("unresolved_o_space_support") != "NOT_APPLICABLE":
            errors.append("O1_OPEN_O_SUPPORT_MUST_BE_NOT_APPLICABLE")
        if output.get("target_invariance_judgment") != "NOT_APPLICABLE":
            errors.append("O1_TARGET_INVARIANCE_MUST_BE_NOT_APPLICABLE")
        if output.get("recommended_action") == "REVISE_O_SPACE":
            errors.append("O1_REVISE_O_SPACE_NOT_APPLICABLE")
    else:
        if output.get("unresolved_o_space_support") == "NOT_APPLICABLE":
            errors.append("OPEN_O_SUPPORT_REQUIRED")
        if output.get("target_invariance_judgment") == "NOT_APPLICABLE":
            errors.append("TARGET_INVARIANCE_REQUIRED")
    if o_mode == "O3" and output.get("fixed_o_support") != "NOT_APPLICABLE":
        errors.append("O3_FIXED_O_SUPPORT_MUST_BE_NOT_APPLICABLE")

    normalized = {
        field: copy.deepcopy(output.get(field)) for field in sorted(required)
    }
    normalized["blocking_reason_codes"] = list(reason_codes)
    normalized["evidence_gaps"] = list(gaps)
    normalized["evidence_ids"] = list(evidence_ids)
    normalized["rationale"] = rationale
    for name, value in support_dimensions.items():
        normalized[name] = value
    normalized["support_dimensions"] = copy.deepcopy(support_dimensions)
    return {
        "status": "PASS" if not errors else "INCOMPLETE",
        "errors": sorted(set(errors)),
        "normalized_review": normalized,
        "review_invocation": _review_invocation(raw),
        "support_dimensions_explicit": support_dimensions_explicit,
    }


def _not_run_review(packet: Mapping[str, Any]) -> dict[str, Any]:
    o_mode = str(packet["condition"]).split("-", 1)[0]
    # Even without a Flow Expert call, a few engineering facts can be read
    # directly from the visible packet.  Keep scientific support unknown, but
    # record declared materialization/evaluation capability instead of letting
    # a legacy eligibility label stand in for it.
    materialization = packet.get("materialization_declared_available")
    if materialization is True:
        execution_support = "SUPPORTED"
    elif materialization is False:
        execution_support = "UNSUPPORTED"
    else:
        execution_support = "NOT_ESTABLISHED"
    evaluation_contract = packet.get("evaluation_contract")
    evaluation_support = (
        "SUPPORTED"
        if isinstance(evaluation_contract, Mapping) and bool(evaluation_contract)
        else "NOT_ESTABLISHED"
    )
    support_dimensions = {
        name: "NOT_ESTABLISHED" for name in SUPPORT_DIMENSION_NAMES
    }
    support_dimensions["execution_materialization_supported"] = execution_support
    support_dimensions["evaluation_contract_supported"] = evaluation_support
    return {
        "case_id": packet["case_id"],
        "condition": packet["condition"],
        "scientific_target_support": "NOT_ESTABLISHED",
        "fixed_o_support": "NOT_APPLICABLE" if o_mode == "O3" else "NOT_ESTABLISHED",
        "unresolved_o_space_support": (
            "NOT_APPLICABLE" if o_mode == "O1" else "NOT_ESTABLISHED"
        ),
        "finding_responsibility_support": "NOT_ESTABLISHED",
        "materialization_support": "NOT_ESTABLISHED",
        "target_invariance_judgment": (
            "NOT_APPLICABLE" if o_mode == "O1" else "NOT_ESTABLISHED"
        ),
        "evidence_sufficiency": "INSUFFICIENT",
        "recommended_action": (
            "REQUEST_EVIDENCE" if o_mode == "O1" else "REVISE_O_SPACE"
        ),
        "blocking_reason_codes": ["FLOW_EXPERT_NOT_RUN"],
        "evidence_gaps": ["Condition-scoped Flow Expert review has not run."],
        "evidence_ids": [],
        "rationale": "No scientific eligibility judgment is inferred from schema validity.",
        "result_equivalence_used_as_predicate": False,
        **support_dimensions,
        "support_dimensions": support_dimensions,
    }


def _compile_slot_status(
    validation: Mapping[str, Any], condition: str
) -> tuple[str, str, list[str], list[str]]:
    o_mode = condition.split("-", 1)[0]
    if validation.get("status") != "PASS":
        status = (
            "BLOCKED_PENDING_EVIDENCE"
            if o_mode == "O1"
            else "BLOCKED_PENDING_O_SPACE_REVIEW"
        )
        return (
            status,
            "SCIENTIFIC_REVIEW_INCOMPLETE",
            list(validation.get("errors", ())),
            ["A complete condition-scoped scientific review is required."],
        )
    review = validation["normalized_review"]
    reasons = list(review["blocking_reason_codes"])
    gaps = list(review["evidence_gaps"])
    support = review.get("support_dimensions", {})
    if not isinstance(support, Mapping):
        support = {}
    # Only a scientifically unsupported target/observable (or an explicit
    # finding-contract failure) can justify omission.  Missing execution or
    # evaluation metadata is a release blocker, not evidence that the target
    # itself is impossible.
    unsupported = {
        field
        for field in (
            "scientific_target_support",
            "finding_responsibility_support",
        )
        if review.get(field) == "UNSUPPORTED"
    }
    if support.get("scientific_target_supported") == "UNSUPPORTED":
        unsupported.add("scientific_target_supported")
    if support.get("dataset_observable_supported") == "UNSUPPORTED":
        unsupported.add("dataset_observable_supported")
    if o_mode != "O3" and review["fixed_o_support"] == "UNSUPPORTED":
        unsupported.add("fixed_o_support")
    if review["target_invariance_judgment"] == "TARGET_DRIFT":
        unsupported.add("target_invariance_judgment")
    if unsupported or review["recommended_action"] == "OMIT":
        return (
            "OMITTED_SCIENTIFICALLY_UNSUPPORTED",
            "SCIENTIFICALLY_UNSUPPORTED",
            [*reasons, *sorted(f"UNSUPPORTED:{item}" for item in unsupported)],
            gaps,
        )
    if o_mode in {"O2", "O3"} and (
        review["unresolved_o_space_support"] != "SUPPORTED"
        or review["target_invariance_judgment"] != "SAME_TARGET_SUPPORTED"
        or review["recommended_action"] == "REVISE_O_SPACE"
    ):
        return (
            "BLOCKED_PENDING_O_SPACE_REVIEW",
            "OPEN_O_SPACE_NOT_YET_SUPPORTED",
            reasons,
            gaps,
        )
    # Execution/evaluation support are release-context dimensions, not
    # scientific-target predicates.  Missing declarations therefore leave a
    # reviewable question visible, but a formally eligible release still needs
    # an explicit support decision for every applicable dimension.  The legacy
    # reviewer contract predates the five-dimensional fields; for that contract
    # only, its target/materialization fields provide a compatibility lift.  No
    # packet capability or Python value is used to invent support.
    explicit_dimensions = bool(validation.get("support_dimensions_explicit"))
    if explicit_dimensions:
        observable_support = support.get("dataset_observable_supported")
        execution_support = support.get("execution_materialization_supported")
        evaluation_support = support.get("evaluation_contract_supported")
        evidence_support = support.get("evidence_complete_for_release")
    else:
        # Compatibility mapping for the pre-v1 reviewer output only.
        observable_support = review.get("scientific_target_support")
        execution_support = review.get("materialization_support")
        evaluation_support = review.get("materialization_support")
        evidence_support = (
            "SUPPORTED"
            if review.get("evidence_sufficiency") == "SUFFICIENT"
            else "NOT_ESTABLISHED"
        )
    support_for_release = {
        "scientific_target_supported": (
            support.get("scientific_target_supported")
            if explicit_dimensions
            else review.get("scientific_target_support")
        ),
        "dataset_observable_supported": observable_support,
        "execution_materialization_supported": execution_support,
        "evaluation_contract_supported": evaluation_support,
        "evidence_complete_for_release": evidence_support,
    }
    if o_mode != "O3":
        support_for_release["fixed_operationalization"] = review.get("fixed_o_support")
    if any(value == "UNSUPPORTED" for value in support_for_release.values()):
        reasons.extend(
            f"{name.upper()}_UNSUPPORTED"
            for name, value in support_for_release.items()
            if value == "UNSUPPORTED"
        )
    # ``NOT_APPLICABLE`` is an explicit positive declaration that a dimension
    # does not govern this condition (for example fixed O under O3).  Only an
    # unresolved or explicitly unsupported applicable dimension blocks release.
    if any(
        value in {"NOT_ESTABLISHED", "UNSUPPORTED"}
        for value in support_for_release.values()
    ) or review[
        "evidence_sufficiency"
    ] != "SUFFICIENT" or review["recommended_action"] == "REQUEST_EVIDENCE":
        return (
            "BLOCKED_PENDING_EVIDENCE",
            "SCIENTIFIC_SUPPORT_NOT_YET_ESTABLISHED",
            reasons,
            gaps,
        )
    return (
        "CANDIDATE_FOR_HUMAN_REVIEW",
        "SUPPORTED_FOR_CANDIDATE_REVIEW",
        reasons,
        gaps,
    )


def gt_srac_impact_for_slot(
    slot_status: str, *, semantic_identity_changed: bool = False
) -> str:
    """Keep dependency identity separate from scientific review readiness."""

    if slot_status == "NEW_CASE_PENDING_HUMAN_SELECTION":
        return "NEW_CASE_REQUIRED"
    if semantic_identity_changed:
        return "STALE_DUE_TO_SEMANTIC_CHANGE"
    if slot_status == "BLOCKED_PENDING_EVIDENCE":
        return "REUSE_BLOCKED_PENDING_EVIDENCE"
    if slot_status == "BLOCKED_PENDING_O_SPACE_REVIEW":
        return "SRAC_O_SPACE_REVIEW_REQUIRED"
    return "UNCHANGED"


def validate_family_condition_semantics(rows: Sequence[Mapping[str, Any]]) -> None:
    """Enforce target identity and O1/O1-F2 Effective-O parity before authoring."""

    projections = {
        str(row["condition"]): row["canonical_semantic_projection"]
        for row in rows
        if isinstance(row.get("canonical_semantic_projection"), Mapping)
    }
    targets = {
        json.dumps(value["scientific_target"], ensure_ascii=False, sort_keys=True)
        for value in projections.values()
    }
    if len(targets) > 1:
        raise ValueError("O3_TARGET_DRIFT_WITHIN_FAMILY")
    if "O1-F1" in projections and "O1-F2" in projections:
        validation = validate_o1_f2_effective_o_semantics(
            projections["O1-F1"], projections["O1-F2"]
        )
        if validation["status"] != "PASS":
            raise ValueError(
                "O1_F2_EFFECTIVE_O_MISMATCH:"
                + ",".join(validation["failure_codes"])
            )


def _hccq_shape(
    candidate: Mapping[str, Any], authoring_packet: Mapping[str, Any]
) -> dict[str, Any]:
    visibility = {
        str(row.get("information_id", "")): str(row.get("visibility", ""))
        for row in authoring_packet.get("semantic_visibility", {}).get("items", ())
        if isinstance(row, Mapping)
    }
    return {
        **dict(candidate),
        "analysis_constraints": [
            {
                "dimension_id": str(row.get("dimension_id", "")),
                "statement": str(row.get("statement", "")),
            }
            for row in authoring_packet["fixed_operationalization"]
            if isinstance(row, Mapping)
            and visibility.get(
                f"fixed_o:{row.get('dimension_id', '')}", "QUESTION_VISIBLE"
            )
            == "QUESTION_VISIBLE"
        ],
        "open_analysis_choices": list(
            authoring_packet["unresolved_operationalization_dimensions"]
        ),
        "requested_findings": dict(authoring_packet["finding_responsibility"]),
    }


def _authoring_repair_feedback(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Project failed presentation audits into a bounded rewrite request.

    The feedback contains only the previous wording and machine-readable
    presentation defects.  It does not expose Ground Truth, reference
    Findings, release decisions, or a replacement scientific semantics.  A
    second authoring attempt can therefore repair the identified wording
    problem instead of repeating the exact first prompt.
    """

    repair_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        defects: list[str] = []
        fidelity = candidate.get("post_rewrite_semantic_fidelity")
        if isinstance(fidelity, Mapping) and str(
            fidelity.get("status", "")
        ).upper() != "PASS":
            defects.extend(
                str(code).strip()
                for code in fidelity.get("failure_codes", ())
                if str(code).strip()
            )
            defects.extend(
                "ADDITIONAL_SCIENTIFIC_OBLIGATION:" + str(obligation).strip()
                for obligation in fidelity.get(
                    "additional_scientific_obligations", ()
                )
                if str(obligation).strip()
            )
            defects.extend(
                "FIDELITY_VALIDATION:" + str(error).strip()
                for error in fidelity.get("validation_errors", ())
                if str(error).strip()
            )

        hccq = candidate.get("hccq_presentation")
        if isinstance(hccq, Mapping) and str(hccq.get("status", "")).upper() != "PASS":
            for key in (
                "backend_schema_language_leakage",
                "unnecessary_mechanical_burden",
                "responsibility_clarity_signals",
                "natural_readability_signals",
                "protocol_wording_signals",
            ):
                defects.extend(
                    "HCCQ:" + str(item).strip()
                    for item in hccq.get(key, ())
                    if str(item).strip()
                )

        final_review = candidate.get("final_flow_expert_wording_review")
        if isinstance(final_review, Mapping) and str(
            final_review.get("final_flow_expert_review_status")
            or final_review.get("status")
            or ""
        ).upper() != "PASS":
            defects.extend(
                "FINAL_REVIEW:" + str(code).strip()
                for code in final_review.get("failure_codes", ())
                if str(code).strip()
            )
            checks = final_review.get("checks")
            if isinstance(checks, Mapping):
                defects.extend(
                    "FINAL_REVIEW_CHECK:" + str(name)
                    for name, passed in checks.items()
                    if passed is False
                )
            recommendation = str(final_review.get("recommendation", "")).strip()
            if recommendation and recommendation != "KEEP":
                defects.append("FINAL_REVIEW_RECOMMENDATION:" + recommendation)

        responsibility = candidate.get("wording_responsibility_audit")
        if isinstance(responsibility, Mapping) and str(
            responsibility.get("status", "")
        ).upper() not in {"", "PASS", "NOT_ESTABLISHED"}:
            defects.extend(
                "RESPONSIBILITY:" + str(code).strip()
                for code in responsibility.get("failure_codes", ())
                if str(code).strip()
            )

        if not defects:
            # A candidate may fail nomination because a reviewer returned an
            # incomplete record without detailed codes.  Keep that failure
            # explicit rather than silently issuing the identical request.
            defects.append("PRESENTATION_AUDIT_DID_NOT_PASS")
        repair_candidates.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")).strip()
                or "unnamed-candidate",
                "previous_model_visible_text": str(
                    candidate.get("model_visible_text", "")
                ).strip(),
                "identified_defects": sorted(set(defects)),
            }
        )
    if not repair_candidates:
        return None
    return {
        "rewrite_only_identified_defects": True,
        "candidates": repair_candidates,
    }


def _successful_resume_invocation(
    value: Mapping[str, Any] | None, *, require_live: bool
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if str(value.get("invocation_status", "")).strip().upper() not in {
        "SUCCESS",
        "PASS",
        "COMPLETED",
        "COMPLETE",
    }:
        return False
    execution_mode = str(value.get("execution_mode", "")).strip().upper()
    if execution_mode.startswith("NOT_RUN") or "DETERMINISTIC" in execution_mode:
        return False
    if require_live and (
        execution_mode != "LIVE_MODEL_CALL" or value.get("live_model_calls") is not True
    ):
        return False
    return True


def _resume_condition_review_raw(
    row: Mapping[str, Any], *, require_live: bool
) -> dict[str, Any] | None:
    review = row.get("condition_scientific_review")
    if not isinstance(review, Mapping) or str(review.get("status", "")).upper() != "PASS":
        return None
    normalized = review.get("normalized_review")
    invocation = review.get("review_invocation") or row.get(
        "condition_review_invocation"
    )
    if not isinstance(normalized, Mapping) or not _successful_resume_invocation(
        invocation if isinstance(invocation, Mapping) else None,
        require_live=require_live,
    ):
        return None
    return {
        **copy.deepcopy(dict(invocation)),
        "parsed_result": copy.deepcopy(dict(normalized)),
    }


def _resume_authored_row_is_complete(
    row: Mapping[str, Any],
    *,
    candidate_count: int,
    require_live: bool,
    enforce_human_facing_acceptance: bool = False,
) -> bool:
    candidates = row.get("question_candidates")
    if not isinstance(candidates, list) or len(candidates) != candidate_count:
        return False
    primary_id = str(row.get("primary_candidate_id", "")).strip()
    primary = next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("candidate_id", "")).strip() == primary_id
        ),
        None,
    )
    if not primary_id or not isinstance(primary, Mapping):
        return False
    if enforce_human_facing_acceptance:
        projection = row.get("canonical_semantic_projection")
        visibility = row.get("semantic_visibility")
        if not isinstance(projection, Mapping):
            return False
        current_responsibility = audit_question_responsibility_realization(
            projection,
            str(primary.get("model_visible_text") or ""),
            candidate={
                "fixed_operationalization": copy.deepcopy(
                    primary.get(
                        "fixed_operationalization",
                        projection.get("resolved_operationalization", ()),
                    )
                ),
                "unresolved_operationalization_dimensions": copy.deepcopy(
                    primary.get(
                        "unresolved_operationalization_dimensions",
                        projection.get("unresolved_operationalization_dimensions", ()),
                    )
                ),
                "finding_responsibility": str(
                    primary.get(
                        "finding_responsibility",
                        projection.get("finding_responsibility", ""),
                    )
                ),
                "requested_findings": copy.deepcopy(
                    primary.get(
                        "requested_findings",
                        projection.get("finding_responsibility", {}),
                    )
                ),
            },
            semantic_visibility=(visibility if isinstance(visibility, Mapping) else None),
        )
        if current_responsibility.get("status") != "PASS":
            return False
    # All variants must be real authored outputs, but only the nominated
    # primary is required to pass the final fidelity/HCCQ/scientific wording
    # gates.  A rejected secondary variant is retained as audit evidence and
    # must not force a scientifically accepted primary to be regenerated.
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return False
        source = str(candidate.get("candidate_source", "")).strip().upper()
        provenance = candidate.get("authoring_provenance")
        if source != "FLOW_EXPERT_PRESENTATION_AUTHORING" or (
            isinstance(provenance, Mapping)
            and provenance.get("deterministic_fallback") is True
        ):
            return False
        author_invocation = candidate.get("question_authoring_invocation")
        if not _successful_resume_invocation(
            author_invocation if isinstance(author_invocation, Mapping) else None,
            require_live=require_live,
        ):
            return False
    fidelity = primary.get("post_rewrite_semantic_fidelity")
    if not isinstance(fidelity, Mapping) or str(
        fidelity.get("status", "")
    ).upper() != "PASS":
        return False
    if not _successful_resume_invocation(
        fidelity.get("review_call")
        if isinstance(fidelity.get("review_call"), Mapping)
        else None,
        require_live=require_live,
    ):
        return False
    hccq = primary.get("hccq_presentation")
    if not isinstance(hccq, Mapping) or str(hccq.get("status", "")).upper() != "PASS":
        return False
    final_review = primary.get("final_flow_expert_wording_review") or primary.get(
        "final_flow_expert_review"
    )
    if not isinstance(final_review, Mapping) or str(
        final_review.get("final_flow_expert_review_status")
        or final_review.get("status")
        or ""
    ).upper() != "PASS":
        return False
    final_invocation = final_review.get("review_call")
    if not _successful_resume_invocation(
        final_invocation if isinstance(final_invocation, Mapping) else None,
        require_live=require_live,
    ):
        return False
    return True


def build_condition_scientific_reviewer_visible_payload(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one internal matrix packet into the existing reviewer contract."""

    projection = packet.get("canonical_semantic_projection")
    evidence = packet.get("scientific_evidence")
    if not isinstance(projection, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("condition review packet lacks semantic projection or evidence")
    source_value = evidence.get("source_records", {})
    if isinstance(source_value, Mapping):
        source_records = list(source_value.get("sources", ()))
    elif isinstance(source_value, list):
        source_records = list(source_value)
    else:
        source_records = []
    evidence_records = [
        *list(evidence.get("context_evidence", ())),
        *list(evidence.get("operationalization_evidence", ())),
        *list(evidence.get("finding_evidence", ())),
    ]
    finding = {
        "finding_mode": projection.get("finding_responsibility"),
        **dict(projection.get("finding_semantics") or {}),
    }
    condition_view = {
        "case_id": packet.get("case_id"),
        "condition": packet.get("condition"),
        "fixed_operationalization": copy.deepcopy(
            projection.get("resolved_operationalization", [])
        ),
        "unresolved_operationalization_dimensions": copy.deepcopy(
            projection.get("unresolved_operationalization_dimensions", [])
        ),
        "finding_responsibility": copy.deepcopy(finding),
        "materialization_declared_available": packet.get(
            "materialization_declared_available"
        ),
        "target_invariance_contract": copy.deepcopy(
            packet.get("target_invariance_contract", {})
        ),
    }
    return {
        "scientific_review_packet_version": (
            SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION
        ),
        "review_input_status": "CONDITION_SCOPED_ADVISORY",
        "task": packet.get("task"),
        "case_identity": {
            "case_id": packet.get("case_id"),
            "condition": packet.get("condition"),
            "dataset_id": packet.get("dataset_id"),
            "family_id": packet.get("family_id"),
        },
        "semantic_contract_sha256": packet.get("semantic_contract_sha256"),
        "dataset_identity": {"dataset_id": packet.get("dataset_id")},
        "family_identity": {
            "family_id": packet.get("family_id"),
            "concept_id": projection.get("concept_id"),
        },
        "scientific_target": copy.deepcopy(projection.get("scientific_target")),
        "scientific_scope": copy.deepcopy(projection.get("scientific_scope")),
        "fixed_operationalization": copy.deepcopy(
            projection.get("resolved_operationalization", [])
        ),
        "unresolved_operationalization_dimensions": copy.deepcopy(
            projection.get("unresolved_operationalization_dimensions", [])
        ),
        "finding_responsibility": copy.deepcopy(finding),
        "dataset_context": {
            "dataset_context": copy.deepcopy(evidence.get("dataset_context", {})),
            "dataset_capability_profile": copy.deepcopy(
                packet.get("dataset_capability_profile", {})
            ),
            "dataset_policy": copy.deepcopy(packet.get("dataset_policy", {})),
        },
        "observable_semantics": copy.deepcopy(
            projection.get("observable_scientific_semantics", {})
        ),
        "conditions": [condition_view],
        "scientific_claims": [],
        "evidence_records": copy.deepcopy(evidence_records),
        "source_records": copy.deepcopy(source_records),
        "claim_support_links": [],
        "provenance_summary": {
            "family_grounding_evidence": copy.deepcopy(
                evidence.get("family_grounding_evidence", {})
            ),
            "family_scoped_operationalization_evidence_ids": copy.deepcopy(
                evidence.get("family_scoped_operationalization_evidence_ids", [])
            ),
            "historical_results_excluded": True,
        },
        "runtime_contract": {
            "condition_scoped_review_required": True,
            "family_wide_failure_propagation_forbidden": True,
            "different_g_of_o_allowed": True,
        },
    }


def run_flow_expert_condition_scientific_review(
    repository_root: str | Path,
    packet: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Invoke the existing canonical Flow Expert for one condition."""

    from .live_agents import LiveModelCaller
    from .scientific_expert_review import (
        FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    )

    caller = LiveModelCaller(
        Path(repository_root).resolve(),
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        max_output_tokens=FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
    )
    visible_payload = build_condition_scientific_reviewer_visible_payload(packet)
    return caller.call(
        role="scientific_reviewer",
        visible_payload=visible_payload,
        instruction=CONDITION_SCIENTIFIC_REVIEW_INSTRUCTION,
        identity={
            "case_id": packet["case_id"],
            "condition": packet["condition"],
            "review_index": "seven-dataset-condition-scientific-review",
        },
        tool_free=True,
    )


def _combustor_target_candidates(
    repository_root: Path,
    manifest_path: str | Path | None = None,
    *,
    allow_persisted_recommendation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else repository_root
        / "outputs/experiments/scientific_question_candidate_matrix_round2"
        / "targeted_reviews/manifest.json"
    )
    manifest = _read_json(path, {})
    reviews = manifest.get("reviews") if isinstance(manifest, Mapping) else None
    review = (
        reviews.get("combustor_density_features", {})
        if isinstance(reviews, Mapping)
        else {}
    )
    candidates: Sequence[Any] = ()
    candidate_sources = (
        review.get("validated_review", {}).get("normalized_review", {})
        if isinstance(review.get("validated_review"), Mapping)
        else {},
        review.get("parsed_result", {}),
        review.get("call", {}).get("parsed_result", {})
        if isinstance(review.get("call"), Mapping)
        else {},
    )
    for source in candidate_sources:
        if isinstance(source, Mapping) and isinstance(
            source.get("candidate_scientific_targets"), list
        ):
            candidates = source["candidate_scientific_targets"]
            break
    rows = candidates
    # Keep this surface limited to the target-redesign contract.  In
    # particular, do not copy legacy case/GT/SRAC identifiers from an older
    # review artifact into the provisional family.  The whitelist also makes
    # the candidate-comparison packet deterministic and model-visible.
    target_fields = {
        "candidate_id",
        "scientific_meaning",
        "evidence_basis",
        "evidence_ids",
        "plausible_o_dimensions",
        "distinct_from_rejected_target",
        "automatic_adoption",
        "recommended_provisional_target",
        "condition_questions",
        "authored_condition_questions",
    }
    normalized_candidates = []
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("automatic_adoption") is not False:
            continue
        normalized = {
            key: copy.deepcopy(raw[key])
            for key in target_fields
            if key in raw
        }
        normalized["automatic_adoption"] = False
        normalized["provisional_family_contract"] = (
            _combustor_provisional_family_contract(normalized)
        )
        normalized_candidates.append(normalized)

    # A prior, independently completed Flow-Expert target comparison is a
    # legitimate input for a deterministic rebuild.  The recommendation is
    # still advisory (``automatic_adoption`` remains false); this fallback
    # merely carries its provenance forward when the targeted-review manifest
    # was archived separately from the current source matrix.  It is strictly
    # candidate-id based and only accepted when it names exactly one of the
    # evidence-backed candidates above.
    persisted_recommendation_path = (
        repository_root
        / "artifacts/review/current/scientific_question_manifest.json"
    )
    persisted_target_id: str | None = None
    if allow_persisted_recommendation and persisted_recommendation_path.is_file():
        try:
            persisted = _read_json(persisted_recommendation_path, {})
        except (OSError, ValueError, TypeError):
            persisted = {}
        persisted_target = (
            persisted.get("combustor_recommended_provisional_target")
            if isinstance(persisted, Mapping)
            else None
        )
        if isinstance(persisted_target, Mapping):
            candidate_id = str(persisted_target.get("candidate_id") or "").strip()
            if candidate_id and any(
                str(item.get("candidate_id") or "").strip() == candidate_id
                for item in normalized_candidates
            ):
                persisted_target_id = candidate_id
    if persisted_target_id is not None:
        for candidate in normalized_candidates:
            candidate["recommended_provisional_target"] = (
                str(candidate.get("candidate_id") or "").strip()
                == persisted_target_id
            )
    source = {
        "manifest_path": (
            str(path.relative_to(repository_root))
            if path.is_relative_to(repository_root)
            # Preserve an explicitly supplied external manifest path for
            # callers/tests that need to identify the exact source.  The
            # canonical in-repository path above remains portable and does not
            # embed the repository's absolute `/home/...` prefix.
            else str(path)
        ),
        "manifest_present": path.is_file(),
        "targeted_review_status": manifest.get("FLOW_EXPERT_TARGETED_REVIEW_STATUS"),
        "combustor_review_invocation_status": review.get("invocation_status"),
        "candidate_count": len(normalized_candidates),
        "automatic_adoption_performed": False,
        "persisted_recommendation_used": persisted_target_id is not None,
        "persisted_recommendation_candidate_id": persisted_target_id,
    }
    return normalized_candidates, source


def _combustor_target_comparison_packet(
    repository_root: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a GT-free packet for selecting one provisional target.

    This packet intentionally contains alternatives and evidence only.  It is
    not a case construction object and therefore cannot carry a historical
    case identifier or a Ground Truth/SRAC result.
    """

    evidence = _evidence_bundle(
        repository_root, "Combustor", "combustor_density_features"
    )
    target_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_contract = _combustor_provisional_family_contract(candidate)
        target_rows.append(
            {
                key: copy.deepcopy(candidate[key])
                for key in (
                    "candidate_id",
                    "scientific_meaning",
                    "evidence_basis",
                    "evidence_ids",
                    "plausible_o_dimensions",
                "distinct_from_rejected_target",
                "four_condition_constructible",
                "constructibility_blocking_reason_codes",
            )
                if key in candidate
            }
        )
        target_rows[-1]["four_condition_constructible"] = bool(
            candidate_contract.get("four_condition_constructible", False)
        )
        target_rows[-1]["constructibility_blocking_reason_codes"] = copy.deepcopy(
            candidate_contract.get("blocking_reason_codes", [])
        )
    return {
        "task": "COMBUSTOR_TARGET_COMPARISON_ADVISORY",
        "packet_version": SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION,
        "family_id": "combustor_density_features",
        "dataset_id": "Combustor",
        "previous_target": "the previously underdefined most prominent density feature",
        "candidate_targets": target_rows,
        "dataset_context": copy.deepcopy(evidence.get("dataset_context", {})),
        "dataset_capability_profile": copy.deepcopy(
            _capability_profiles(repository_root).get("Combustor", {})
        ),
        "observable_semantics": copy.deepcopy(
            evidence.get("operationalization_evidence", [])
        ),
        "evidence_records": [
            *copy.deepcopy(evidence.get("context_evidence", [])),
            *copy.deepcopy(evidence.get("operationalization_evidence", [])),
            *copy.deepcopy(evidence.get("finding_evidence", [])),
        ],
        "source_records": copy.deepcopy(evidence.get("source_records", [])),
        "selection_contract": {
            "exactly_one_recommended_target": True,
            "recommendation_is_provisional": True,
            "human_approval_required": True,
            "no_case_or_gt_srac_mutation": True,
            "target_candidates_are_not_alternative_o_for_one_target": True,
        },
    }


def _unwrap_combustor_target_review(raw: Mapping[str, Any]) -> Mapping[str, Any] | None:
    parsed = raw.get("parsed_result")
    if isinstance(parsed, Mapping):
        return parsed
    if any(
        key in raw
        for key in (
            "recommended_target_id",
            "recommended_provisional_target_id",
            "recommended_provisional_target",
            "recommendation_status",
            "candidate_comparisons",
        )
    ):
        return raw
    return None


def validate_combustor_target_selection(
    raw: Mapping[str, Any],
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one Flow Expert target comparison without adopting a target.

    The validator accepts a small set of naming aliases for compatibility with
    existing live/injected callers, but it only authorizes a recommendation
    when the model explicitly identifies one candidate and keeps human
    approval required.  An unresolved result is valid evidence of uncertainty,
    not an error that should be converted into a selection.
    """

    output = _unwrap_combustor_target_review(raw)
    invocation = _review_invocation(raw)
    if output is None:
        return {
            "status": "INCOMPLETE",
            "errors": ["NO_PARSED_COMBUSTOR_TARGET_REVIEW"],
            "normalized_review": {},
            "target_review_invocation": invocation,
        }
    candidate_ids = {
        str(item.get("candidate_id", "")).strip()
        for item in packet.get("candidate_targets", ())
        if isinstance(item, Mapping) and str(item.get("candidate_id", "")).strip()
    }
    errors: list[str] = []
    allowed = {
        "family_id",
        "recommended_target_id",
        "recommended_provisional_target_id",
        "recommended_provisional_target",
        "recommendation_status",
        "human_approval_required",
        "evidence_sufficiency",
        "candidate_comparisons",
        "evidence_ids",
        "rationale",
    }
    unknown = sorted(set(output) - allowed)
    if unknown:
        errors.append("UNKNOWN_TARGET_REVIEW_FIELDS:" + ",".join(unknown))
    if str(output.get("family_id", "")).strip() != str(packet.get("family_id", "")):
        errors.append("TARGET_REVIEW_FAMILY_ID_MISMATCH")

    recommendation = output.get("recommended_target_id")
    if recommendation is None:
        recommendation = output.get("recommended_provisional_target_id")
    if recommendation is None:
        alias = output.get("recommended_provisional_target")
        if isinstance(alias, Mapping):
            recommendation = alias.get("candidate_id") or alias.get("target_id")
        elif isinstance(alias, str):
            recommendation = alias
    recommendation_id = str(recommendation or "").strip() or None
    if recommendation_id is not None and recommendation_id not in candidate_ids:
        errors.append("RECOMMENDED_TARGET_ID_UNKNOWN")
    if recommendation_id is not None:
        selected_packet_target = next(
            (
                item
                for item in packet.get("candidate_targets", ())
                if isinstance(item, Mapping)
                and str(item.get("candidate_id", "")).strip() == recommendation_id
            ),
            None,
        )
        if (
            isinstance(selected_packet_target, Mapping)
            and selected_packet_target.get("four_condition_constructible") is False
        ):
            errors.append("RECOMMENDED_TARGET_NOT_FOUR_CONDITION_CONSTRUCTIBLE")
    status = str(output.get("recommendation_status", "")).strip().upper()
    if not status:
        status = (
            "RECOMMENDED_PROVISIONAL_TARGET"
            if recommendation_id is not None
            else "UNRESOLVED"
        )
    if status not in {"RECOMMENDED_PROVISIONAL_TARGET", "UNRESOLVED"}:
        errors.append("INVALID_TARGET_RECOMMENDATION_STATUS")
    if status == "RECOMMENDED_PROVISIONAL_TARGET" and recommendation_id is None:
        errors.append("RECOMMENDED_TARGET_REQUIRED")
    if status == "UNRESOLVED" and recommendation_id is not None:
        errors.append("UNRESOLVED_RECOMMENDATION_MUST_BE_NULL")
    if output.get("human_approval_required") is not True:
        errors.append("TARGET_HUMAN_APPROVAL_MUST_REMAIN_REQUIRED")
    sufficiency = str(output.get("evidence_sufficiency", "")).strip().upper()
    if sufficiency not in {"SUFFICIENT", "INSUFFICIENT"}:
        errors.append("INVALID_TARGET_EVIDENCE_SUFFICIENCY")
    if status == "RECOMMENDED_PROVISIONAL_TARGET" and sufficiency != "SUFFICIENT":
        errors.append("RECOMMENDATION_REQUIRES_SUFFICIENT_EVIDENCE")
    rationale = str(output.get("rationale", "")).strip()
    if not rationale:
        errors.append("TARGET_RECOMMENDATION_RATIONALE_REQUIRED")

    evidence_ids = output.get("evidence_ids")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(item, str) or not item.strip() for item in evidence_ids
    ):
        errors.append("TARGET_EVIDENCE_IDS_INVALID")
        evidence_ids = []
    known_evidence = {
        str(item.get("evidence_id", "")).strip()
        for item in packet.get("evidence_records", ())
        if isinstance(item, Mapping) and str(item.get("evidence_id", "")).strip()
    }
    if set(evidence_ids) - known_evidence:
        errors.append("TARGET_UNKNOWN_EVIDENCE_ID")

    comparisons = output.get("candidate_comparisons")
    normalized_comparisons: list[dict[str, Any]] = []
    if not isinstance(comparisons, list):
        errors.append("TARGET_CANDIDATE_COMPARISONS_REQUIRED")
    else:
        seen: set[str] = set()
        for item in comparisons:
            if not isinstance(item, Mapping):
                errors.append("TARGET_CANDIDATE_COMPARISON_INVALID")
                continue
            cid = str(item.get("candidate_id", "")).strip()
            if not cid or cid in seen or cid not in candidate_ids:
                errors.append("TARGET_CANDIDATE_COMPARISON_COVERAGE_INVALID")
                continue
            seen.add(cid)
            comparison = {
                "candidate_id": cid,
                "physical_meaning": str(item.get("physical_meaning", "")).strip(),
                "target_stability": str(item.get("target_stability", "")).strip(),
                "dataset_support": str(item.get("dataset_support", "")).strip(),
                "observable_semantics": str(item.get("observable_semantics", "")).strip(),
                "o_variation_support": str(item.get("o_variation_support", "")).strip(),
                "finding_support": str(item.get("finding_support", "")).strip(),
                # The instruction requires a judgment and scientific reason
                # but intentionally does not prescribe a candidate-comparison
                # micro-schema.  Accept the two concise aliases produced by
                # the canonical Flow Expert instead of rejecting a complete
                # review on naming alone.
                "judgment": str(
                    item.get("judgment") or item.get("assessment") or ""
                ).strip(),
                "rationale": str(
                    item.get("rationale")
                    or item.get("scientific_rationale")
                    or ""
                ).strip(),
                "evidence_ids": [
                    str(value).strip()
                    for value in item.get("evidence_ids", ())
                    if str(value).strip()
                ],
            }
            # The reviewer instruction fixes the scientific comparison
            # criteria but not a nested micro-schema.  A comparison is
            # substantive when it supplies a judgment/rationale or actually
            # discusses one of those criteria; a bare candidate id remains
            # insufficient.
            if not any(
                comparison[key]
                for key in (
                    "physical_meaning",
                    "target_stability",
                    "dataset_support",
                    "observable_semantics",
                    "o_variation_support",
                    "finding_support",
                    "judgment",
                    "rationale",
                )
            ):
                errors.append("TARGET_CANDIDATE_COMPARISON_RATIONALE_REQUIRED")
            normalized_comparisons.append(comparison)
        if seen != candidate_ids:
            errors.append("TARGET_CANDIDATE_COMPARISON_COVERAGE_INVALID")

    normalized = {
        "family_id": str(packet.get("family_id", "")),
        "recommended_target_id": recommendation_id,
        "recommended_provisional_target_id": recommendation_id,
        "recommendation_status": status,
        "human_approval_required": True,
        "evidence_sufficiency": sufficiency,
        "candidate_comparisons": normalized_comparisons,
        "evidence_ids": [str(item).strip() for item in evidence_ids],
        "rationale": rationale,
    }
    return {
        "status": "PASS" if not errors else "INCOMPLETE",
        "errors": sorted(set(errors)),
        "normalized_review": normalized,
        "target_review_invocation": invocation,
    }


def run_flow_expert_combustor_target_review(
    repository_root: str | Path,
    packet: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Use the existing Flow Expert profile for Combustor target comparison."""

    from .live_agents import LiveModelCaller
    from .scientific_expert_review import (
        FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    )

    caller = LiveModelCaller(
        Path(repository_root).resolve(),
        FLOW_SCIENTIFIC_REVIEWER_PROFILE,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
        max_output_tokens=FLOW_SCIENTIFIC_REVIEWER_MAX_OUTPUT_TOKENS,
    )
    return caller.call(
        role="scientific_reviewer",
        visible_payload=packet,
        instruction=COMBUSTOR_TARGET_COMPARISON_INSTRUCTION,
        identity={
            "family_id": packet.get("family_id"),
            "dataset_id": packet.get("dataset_id"),
            "review_index": "combustor-target-comparison",
        },
        tool_free=True,
    )


def _resolve_combustor_provisional_target(
    repository_root: Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    reviewer: CombustorTargetReviewer | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run/compile the single Combustor target-selection advisory.

    The returned target is still only a proposal.  The helper deliberately
    keeps the comparison packet and validation result alongside it so a later
    portfolio audit can distinguish an actual Flow Expert recommendation from
    a hand-edited ``recommended_provisional_target`` flag.
    """

    packet = _combustor_target_comparison_packet(repository_root, candidates)
    if reviewer is None:
        return None, {
            "status": "NOT_RUN",
            "validation_status": "NOT_RUN",
            "packet": packet,
            "target_review_invocation": {
                "execution_mode": "NOT_RUN",
                "invocation_status": "NOT_RUN",
                "raw_call_sha256": None,
            },
        }
    attempts: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for attempt in range(1, 3):
        try:
            raw = reviewer(copy.deepcopy(packet))
        except Exception as exc:
            # A provider/agent failure is an unresolved scientific review, not
            # permission to synthesize a target or condition questions.
            error_value = {"error_type": type(exc).__name__, "error": str(exc)}
            attempts.append(
                {
                    "attempt": attempt,
                    "validation_status": "INCOMPLETE",
                    "errors": [f"TARGET_REVIEWER_EXCEPTION:{type(exc).__name__}:{exc}"],
                    "invocation": {
                        "execution_mode": "LIVE_FLOW_EXPERT",
                        "invocation_status": "EXCEPTION",
                        "raw_call_sha256": _json_sha256(error_value),
                    },
                }
            )
            validation = {
                "status": "INCOMPLETE",
                "errors": attempts[-1]["errors"],
                "normalized_review": {},
                "target_review_invocation": attempts[-1]["invocation"],
            }
            continue
        if not isinstance(raw, Mapping):
            raw = {"raw_result": raw}
        validation = validate_combustor_target_selection(raw, packet)
        target_invocation = validation.setdefault(
            "target_review_invocation", _review_invocation(raw)
        )
        if isinstance(target_invocation, dict):
            # A returned structured mapping proves that the injected reviewer
            # callback ran.  Live callers already carry their provider
            # envelope; only fill the transport fields omitted by lightweight
            # injected test/replay callbacks.
            target_invocation.setdefault("execution_mode", "INJECTED_REVIEWER")
            target_invocation.setdefault("invocation_status", "SUCCESS")
        attempts.append(
            {
                "attempt": attempt,
                "validation_status": validation.get("status"),
                "errors": list(validation.get("errors", ())),
                "invocation": copy.deepcopy(
                    validation.get("target_review_invocation", {})
                ),
            }
        )
        if validation.get("status") == "PASS":
            break
    normalized = validation.get("normalized_review")
    selected: dict[str, Any] | None = None
    if validation.get("status") == "PASS" and isinstance(normalized, Mapping):
        target_id = str(normalized.get("recommended_target_id") or "").strip()
        for candidate in candidates:
            if str(candidate.get("candidate_id", "")).strip() == target_id:
                selected = copy.deepcopy(dict(candidate))
                selected["recommended_provisional_target"] = True
                selected["automatic_adoption"] = False
                selected["target_selection_review"] = copy.deepcopy(validation)
                selected["provisional_family_contract"] = (
                    _combustor_provisional_family_contract(selected)
                )
                break
    result = {
        "status": "RECOMMENDED_PROVISIONAL_TARGET" if selected else "UNRESOLVED",
        "validation_status": validation.get("status", "INCOMPLETE"),
        "normalized_review": copy.deepcopy(normalized or {}),
        "target_review_invocation": copy.deepcopy(
            validation.get("target_review_invocation", {})
        ),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "packet_sha256": _json_sha256(packet),
        "human_approval_required": True,
    }
    return selected, result


def _explicit_combustor_recommendation(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Accept a prior explicit recommendation only when it is unambiguous.

    This is useful when a separately persisted Flow Expert comparison has
    already run.  The boolean is treated as provenance, never as an automatic
    adoption signal; the returned target remains provisional and requires the
    four-condition authoring path below.
    """

    recommended = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("recommended_provisional_target") is True
    ]
    if len(recommended) != 1:
        return None
    selected = copy.deepcopy(dict(recommended[0]))
    selected["automatic_adoption"] = False
    selected["recommended_provisional_target"] = True
    selected["provisional_family_contract"] = _combustor_provisional_family_contract(
        selected
    )
    return selected


def _historical_dependency_state(
    matrix_status: str, gt_srac_impact: str
) -> dict[str, Any]:
    if matrix_status == "CANDIDATE_FOR_HUMAN_REVIEW" and gt_srac_impact == "UNCHANGED":
        reuse_status = "PENDING_HUMAN_SELECTION"
    elif matrix_status == "BLOCKED_PENDING_EVIDENCE":
        reuse_status = "BLOCKED_PENDING_EVIDENCE"
    elif matrix_status == "BLOCKED_PENDING_O_SPACE_REVIEW":
        reuse_status = "BLOCKED_PENDING_O_SPACE_REVIEW"
    elif matrix_status == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
        reuse_status = "NOT_AUTHORIZED_SCIENTIFICALLY_UNSUPPORTED"
    else:
        reuse_status = "NOT_AUTHORIZED"
    return {
        "historical_gt_srac_identity_preserved": True,
        "historical_gt_srac_reuse_authorized": False,
        "historical_gt_srac_reuse_status": reuse_status,
    }


def _render_review(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Seven-Dataset Scientific Question Candidate Review",
        "",
        "This is a construction-time human review pack. Candidate count is sparse by design; no slot was added merely to reach 28. Flow Expert judgments are advisory.",
        "",
        "## Summary",
        "",
        "| Dataset | O1-F1 | O2-F1 | O3-F1 | O1-F2 | retained count |",
        "|---|---|---|---|---|---:|",
    ]
    for dataset in manifest["datasets"]:
        slots = dataset["condition_slots"]
        lines.append(
            "| {dataset} | {o1} | {o2} | {o3} | {f2} | {count} |".format(
                dataset=dataset["dataset_id"],
                o1=slots["O1-F1"],
                o2=slots["O2-F1"],
                o3=slots["O3-F1"],
                f2=slots["O1-F2"],
                count=dataset["retained_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Counts",
            "",
            *[
                f"- {key}: `{manifest[key]}`"
                for key in (
                    "TOTAL_CANDIDATES_PROPOSED",
                    "TOTAL_CANDIDATES_FOR_HUMAN_REVIEW",
                    "TOTAL_BLOCKED_PENDING_EVIDENCE",
                    "TOTAL_BLOCKED_PENDING_O_SPACE_REVIEW",
                    "TOTAL_NEW_CASE_PENDING_SELECTION",
                    "TOTAL_OMITTED",
                )
            ],
            "",
            "`TOTAL_CANDIDATES_PROPOSED` counts every non-omitted slot still in the review pipeline: ready + blocked pending evidence + blocked pending O-space review + new pending selection. It is not a PASS count.",
            "",
            "## Family Presentation Audit",
            "",
            "This audit is a presentation observation. It does not alter condition-scoped scientific eligibility.",
            "",
            "| Dataset | family presentation status |",
            "|---|---|",
        ]
    )
    for dataset_id in DATASET_POLICIES:
        audit = manifest["family_template_leakage_audits"].get(dataset_id, {})
        lines.append(f"| {dataset_id} | {audit.get('status', 'NOT_RUN')} |")
    lines.append("")
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in manifest["conditions"]:
        by_dataset.setdefault(str(row["dataset_id"]), []).append(row)
    for dataset_id in DATASET_POLICIES:
        lines.extend([f"## {dataset_id}", ""])
        if dataset_id == "Combustor":
            lines.extend(
                [
                    "The previous `most prominent density feature` target is not reused. No redesigned target is automatically selected, and no historical Density GT/SRAC is inherited.",
                    "",
                    "Candidate scientific targets awaiting human selection:",
                    "",
                ]
            )
            targets = manifest["combustor_redesign_target_candidates"]
            if not targets:
                lines.append("- No evidence-grounded target candidate is currently available.")
            for target in targets:
                lines.append(
                    f"- `{target.get('candidate_id')}`: {target.get('scientific_meaning')} Evidence: {target.get('evidence_basis')}"
                )
            lines.append("")
        for row in by_dataset[dataset_id]:
            lines.extend(
                [
                    f"### {row['condition']} - {row['matrix_status']}",
                    "",
                    f"Scientific family / concept: `{row['family_id']}` / `{row.get('concept_id') or 'PENDING'}`",
                    "",
                    f"Scientific target: {row.get('scientific_target') or 'Pending human target selection.'}",
                    "",
                    f"Primary human-facing question: {row.get('primary_human_facing_question') or 'Not nominated for this slot.'}",
                    "",
                    "Fixed O: "
                    + ("; ".join(row.get("fixed_o_display", ())) or "None / pending."),
                    "",
                    "Open O: "
                    + (", ".join(row.get("open_o_display", ())) or "None / pending."),
                    "",
                    f"F responsibility: {row.get('finding_responsibility') or 'Pending.'}",
                    "",
                    f"Flow Expert scientific status: `{row['CONDITION_SCIENTIFIC_STATUS']}`; evidence sufficiency: `{row['evidence_sufficiency']}`.",
                    "",
                    f"Semantic fidelity: `{row['semantic_fidelity_status']}`; HCCQ: `{row['hccq_status']}`.",
                    "",
                    f"GT/SRAC impact: `{row['GT_SRAC_IMPACT']}`.",
                    "",
                    "Unresolved issue: "
                    + ("; ".join(row["evidence_gaps"]) or "None recorded."),
                    "",
                ]
            )
    lines.extend(
        [
            "## Integrity Answers",
            "",
            "- A. Were cases added only to reach 28? **No.**",
            "- B. Did O2/O3 fail solely because valid O choices produced different G(O)? **No.**",
            "- C. Did a family-level issue automatically block another supported condition? **No; every slot is compiled independently.**",
            "- D. Scientifically eligible questions ready for direct human review are exactly those listed as `CANDIDATE_FOR_HUMAN_REVIEW`. Family presentation-leakage observations remain visible above for wording inspection and do not change scientific eligibility.",
            "",
            "## Downstream Gates",
            "",
            "Human curator confirmation, official SCQ, formal SRAC adjudication, evaluator calibration, and tested-model experiments were not executed.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_scientific_question_candidate_matrix(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    condition_reviewer: ConditionReviewer | None = None,
    question_author: QuestionAuthor | None = None,
    fidelity_reviewer: FidelityReviewer | None = None,
    final_reviewer: FinalReviewer | None = None,
    combustor_target_reviewer: CombustorTargetReviewer | None = None,
    run_live: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
    max_workers: int = 7,
    candidate_count: int = 2,
    targeted_review_manifest_path: str | Path | None = None,
    resume_manifest_path: str | Path | None = None,
    enforce_human_facing_acceptance: bool = False,
    max_authoring_attempts: int = 2,
) -> dict[str, Any]:
    """Construct and write the sparse review matrix without crossing later gates."""

    if candidate_count not in {2, 3}:
        raise ValueError("candidate_count must be 2 or 3")
    if max_authoring_attempts < 1 or max_authoring_attempts > 12:
        raise ValueError("max_authoring_attempts must be between 1 and 12")
    if run_live and any(
        item is not None
        for item in (
            condition_reviewer,
            question_author,
            fidelity_reviewer,
            final_reviewer,
            combustor_target_reviewer,
        )
    ):
        raise ValueError("run_live and injected reviewers are mutually exclusive")
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    resume_manifest: Mapping[str, Any] = {}
    resume_rows_by_case: dict[str, Mapping[str, Any]] = {}
    resume_rows_by_slot: dict[str, Mapping[str, Any]] = {}
    resume_manifest_sha256: str | None = None
    if resume_manifest_path is not None:
        resume_path = Path(resume_manifest_path).resolve()
        loaded_resume = _read_json(resume_path, {})
        if not isinstance(loaded_resume, Mapping):
            raise ValueError("RESUME_MANIFEST_NOT_OBJECT")
        resume_manifest = loaded_resume
        resume_rows = resume_manifest.get("conditions")
        if not isinstance(resume_rows, list):
            raise ValueError("RESUME_MANIFEST_CONDITIONS_NOT_LIST")
        for resume_row in resume_rows:
            if not isinstance(resume_row, Mapping):
                continue
            case_key = str(resume_row.get("case_id") or "").strip()
            slot_key = str(resume_row.get("slot_id") or "").strip()
            if case_key:
                resume_rows_by_case[case_key] = resume_row
            if slot_key:
                resume_rows_by_slot[slot_key] = resume_row
        resume_manifest_sha256 = _json_sha256(resume_manifest)
    allowed_outputs = {
        "scientific_question_manifest.json",
        "scientific_question_review.md",
    }
    if out.exists():
        unexpected = sorted(
            path.name
            for path in out.iterdir()
            if path.name not in allowed_outputs or not path.is_file()
        )
        if unexpected:
            raise ValueError(
                "SCIENTIFIC_QUESTION_OUTPUT_DIRECTORY_NOT_SHALLOW:"
                + ",".join(unexpected)
            )
    families = _family_inventory(root)
    capabilities = _capability_profiles(root)
    packets = build_condition_scientific_review_packets(root)

    if run_live:
        condition_reviewer = lambda packet: run_flow_expert_condition_scientific_review(
            root,
            packet,
            config_path=config_path,
            api_key=api_key,
            timeout=timeout,
        )
        question_author = lambda packet: run_flow_expert_question_presentation_authoring(
            str(root),
            packet,
            config_path=str(config_path) if config_path else None,
            api_key=api_key,
            timeout=timeout,
        )
        fidelity_reviewer = lambda packet: run_flow_expert_post_rewrite_semantic_fidelity(
            str(root),
            packet,
            config_path=str(config_path) if config_path else None,
            api_key=api_key,
            timeout=timeout,
        )
        final_reviewer = lambda packet: run_flow_expert_final_flow_wording_review(
            str(root),
            packet,
            config_path=str(config_path) if config_path else None,
            api_key=api_key,
            timeout=timeout,
        )
        combustor_target_reviewer = lambda packet: run_flow_expert_combustor_target_review(
            str(root),
            packet,
            config_path=str(config_path) if config_path else None,
            api_key=api_key,
            timeout=timeout,
        )

    if resume_rows_by_case:
        underlying_condition_reviewer = condition_reviewer

        def condition_reviewer_with_resume(
            packet: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            case_id = str(packet.get("case_id") or "").strip()
            resume_row = resume_rows_by_case.get(case_id)
            if isinstance(resume_row, Mapping) and str(
                resume_row.get("semantic_contract_sha256") or ""
            ) == str(packet.get("semantic_contract_sha256") or ""):
                resumed = _resume_condition_review_raw(
                    resume_row, require_live=run_live
                )
                if resumed is not None:
                    return resumed
            if underlying_condition_reviewer is None:
                return _not_run_review(packet)
            return underlying_condition_reviewer(packet)

        condition_reviewer = condition_reviewer_with_resume

    packet_by_case = {str(packet["case_id"]): packet for packet in packets}

    def review_packet(packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        max_attempts = 2 if condition_reviewer is not None else 1
        validation: dict[str, Any] = {}
        for attempt_number in range(1, max_attempts + 1):
            raw = (
                dict(condition_reviewer(copy.deepcopy(packet)))
                if condition_reviewer is not None
                else _not_run_review(packet)
            )
            validation = validate_condition_scientific_review(raw, packet)
            invocation = validation["review_invocation"]
            invocation.setdefault(
                "execution_mode",
                "LIVE_FLOW_EXPERT"
                if run_live
                else "INJECTED_REVIEWER"
                if condition_reviewer is not None
                else "NOT_RUN",
            )
            invocation.setdefault(
                "invocation_status",
                "NOT_RUN" if condition_reviewer is None else "NOT_REPORTED",
            )
            attempts.append(
                {
                    "attempt": attempt_number,
                    "validation_status": validation["status"],
                    "errors": list(validation.get("errors", ())),
                    "invocation": copy.deepcopy(invocation),
                }
            )
            retryable_invalid_output = validation["status"] != "PASS"
            if validation["status"] == "PASS" or not retryable_invalid_output:
                break
        validation["review_attempt_count"] = len(attempts)
        validation["review_attempts"] = attempts
        return str(packet["case_id"]), validation

    reviews: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(packets)))) as pool:
        futures = {pool.submit(review_packet, packet): packet for packet in packets}
        for future in as_completed(futures):
            packet = futures[future]
            case_id = str(packet["case_id"])
            try:
                result_case_id, validation = future.result()
                reviews[result_case_id] = validation
            except Exception as exc:
                error_value = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                reviews[case_id] = {
                    "status": "INCOMPLETE",
                    "errors": [f"REVIEWER_EXCEPTION:{type(exc).__name__}:{exc}"],
                    "normalized_review": {},
                    "review_invocation": {
                        "execution_mode": (
                            "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWER"
                        ),
                        "invocation_status": "EXCEPTION",
                        "raw_call_sha256": _json_sha256(error_value),
                    },
                }

    rows: list[dict[str, Any]] = []
    source_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    allow_persisted_recommendation = False
    try:
        allow_persisted_recommendation = out.is_relative_to(
            root / "outputs/current"
        )
    except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
        allow_persisted_recommendation = str(out).startswith(
            str(root / "outputs/current")
        )
    combustor_candidates, combustor_candidate_source = _combustor_target_candidates(
        root,
        targeted_review_manifest_path,
        allow_persisted_recommendation=allow_persisted_recommendation,
    )
    combustor_target_review: dict[str, Any]
    recommended_combustor_target: dict[str, Any] | None = None
    resumed_combustor_target_review = False
    effective_combustor_target_reviewer = combustor_target_reviewer
    if len(combustor_candidates) == 2 and resume_manifest:
        persisted_target_review = resume_manifest.get(
            "combustor_target_selection_review"
        )
        if isinstance(persisted_target_review, Mapping):
            normalized_target_review = persisted_target_review.get(
                "normalized_review"
            )
            target_invocation = persisted_target_review.get(
                "target_review_invocation"
            )
            if (
                isinstance(normalized_target_review, Mapping)
                and isinstance(target_invocation, Mapping)
                and _successful_resume_invocation(
                    target_invocation, require_live=run_live
                )
            ):
                persisted_envelope = {
                    **copy.deepcopy(dict(target_invocation)),
                    "parsed_result": copy.deepcopy(dict(normalized_target_review)),
                }
                persisted_validation = validate_combustor_target_selection(
                    persisted_envelope,
                    _combustor_target_comparison_packet(root, combustor_candidates),
                )
                if persisted_validation.get("status") == "PASS":
                    effective_combustor_target_reviewer = (
                        lambda _packet, envelope=copy.deepcopy(persisted_envelope): copy.deepcopy(
                            envelope
                        )
                    )
                    resumed_combustor_target_review = True
    if len(combustor_candidates) != 2:
        combustor_target_review = {
            "status": "UNRESOLVED",
            "validation_status": "INCOMPLETE",
            "errors": ["COMBUSTOR_TARGET_CANDIDATE_COUNT_INVALID"],
            "human_approval_required": True,
            "target_review_invocation": {
                "execution_mode": "NOT_RUN",
                "invocation_status": "NOT_RUN_INVALID_CANDIDATE_SET",
                "raw_call_sha256": None,
            },
        }
    elif effective_combustor_target_reviewer is None:
        recommended_combustor_target = _explicit_combustor_recommendation(
            combustor_candidates
        )
        combustor_target_review = {
            "status": (
                "RECOMMENDED_PROVISIONAL_TARGET"
                if recommended_combustor_target is not None
                else "UNRESOLVED"
            ),
            "validation_status": (
                "SOURCE_RECOMMENDATION"
                if recommended_combustor_target is not None
                else "NOT_RUN"
            ),
            "errors": [],
            "human_approval_required": True,
            "target_review_invocation": {
                "execution_mode": "PERSISTED_REVIEW_ARTIFACT"
                if recommended_combustor_target is not None
                else "NOT_RUN",
                "invocation_status": "SUCCESS"
                if recommended_combustor_target is not None
                else "NOT_RUN",
                "raw_call_sha256": None,
            },
        }
    else:
        recommended_combustor_target, combustor_target_review = (
            _resolve_combustor_provisional_target(
                root,
                combustor_candidates,
                reviewer=effective_combustor_target_reviewer,
            )
        )
    if recommended_combustor_target is not None:
        # Only the selected target carries the recommendation bit.  Existing
        # alternatives remain visible in the audit surface and can never seed
        # a condition row by themselves.
        selected_id = str(recommended_combustor_target.get("candidate_id", ""))
        for candidate in combustor_candidates:
            candidate["recommended_provisional_target"] = (
                str(candidate.get("candidate_id", "")) == selected_id
            )
        recommended_combustor_target["recommended_provisional_target"] = True
        recommended_combustor_target["automatic_adoption"] = False
        recommended_combustor_target["provisional_family_contract"] = (
            _combustor_provisional_family_contract(recommended_combustor_target)
        )
        # Keep the selection provenance with the target itself.  This is an
        # advisory recommendation, never an automatic case/GT/SRAC adoption.
        recommended_combustor_target["target_selection_review"] = copy.deepcopy(
            combustor_target_review
        )
    elif len(combustor_candidates) == 2:
        # Make unresolved selection explicit for both alternatives.  False is
        # an audit state, never an implicit adoption signal.
        for candidate in combustor_candidates:
            candidate["recommended_provisional_target"] = False
    for dataset_id, policy in DATASET_POLICIES.items():
        if policy.get("redesign_required"):
            # A target comparison is a family-level review; condition rows are
            # constructed only after one explicit recommendation is available.
            if recommended_combustor_target is not None and (
                recommended_combustor_target.get("provisional_family_contract", {}).get(
                    "contract_status"
                )
                == "PROVISIONAL_CONSTRUCTIBLE_PENDING_HUMAN_APPROVAL"
            ):
                target_contract = recommended_combustor_target[
                    "provisional_family_contract"
                ]
                evidence = _evidence_bundle(
                    root, "Combustor", "combustor_density_features"
                )
                capability = capabilities.get("Combustor", {})
                selected_combustor_rows: list[dict[str, Any]] = []
                provisional_family_id = str(
                    target_contract.get("family_id")
                    or (
                        "combustor_"
                        + str(recommended_combustor_target.get("candidate_id", "density_target"))
                        + "_provisional"
                    )
                )
                for condition in CONDITIONS:
                    projection = _combustor_condition_projection(
                        recommended_combustor_target, condition
                    )
                    slot_id = (
                        f"{provisional_family_id}_"
                        f"{condition.casefold().replace('-', '_')}"
                    )
                    author_case_id = slot_id
                    authoring_context = _dataset_context_for_authoring(
                        {"case_context": evidence.get("dataset_context", {})},
                        capability,
                    )
                    visibility = _semantic_visibility(
                        projection, authoring_context
                    )
                    support_dimensions = {
                        # The successful evidence-conditioned target review
                        # establishes scientific support for human review.  It
                        # does not constitute human approval or release.
                        "scientific_target_supported": "SUPPORTED",
                        "dataset_observable_supported": "SUPPORTED",
                        "execution_materialization_supported": "NOT_ESTABLISHED",
                        "evaluation_contract_supported": "NOT_ESTABLISHED",
                        "evidence_complete_for_release": "NOT_ESTABLISHED",
                    }
                    row = {
                        "case_id": None,
                        "slot_id": slot_id,
                        "dataset_id": dataset_id,
                        "family_id": provisional_family_id,
                        "concept_id": provisional_family_id,
                        "condition": condition,
                        "matrix_status": "NEW_CASE_PENDING_HUMAN_SELECTION",
                        "QUESTION_CANDIDATE_STATUS": "READY_FOR_HUMAN_REVIEW",
                        "SCIENTIFIC_RELEASE_STATUS": "NEW_CASE_REQUIRED",
                        "FORMAL_CANDIDATE_STATUS": "WITHDRAWN_NEW_CASE_REQUIRED",
                        "ARTIFACT_REUSE_STATUS": "REBUILD_REQUIRED",
                        "CONDITION_SCIENTIFIC_STATUS": "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL",
                        "scientific_target": projection["scientific_target"],
                        "scientific_scope": projection["scientific_scope"],
                        "fixed_operationalization": copy.deepcopy(
                            projection["resolved_operationalization"]
                        ),
                        "fixed_o_display": [
                            f"{item['dimension_id']}: {item['normalized_meaning']}"
                            for item in projection["resolved_operationalization"]
                        ],
                        "unresolved_operationalization_dimensions": copy.deepcopy(
                            projection["unresolved_operationalization_dimensions"]
                        ),
                        "open_o_display": [
                            str(item.get("dimension_id", ""))
                            for item in projection[
                                "unresolved_operationalization_dimensions"
                            ]
                        ],
                        "finding_responsibility": projection[
                            "finding_responsibility"
                        ],
                        "canonical_semantic_projection": projection,
                        # This provisional projection intentionally has its
                        # own version and is not a canonical CaseConstruction
                        # projection.  Hash it as a JSON contract instead of
                        # asking the canonical semantic helper to coerce it
                        # into an old case schema.
                        "semantic_contract_sha256": canonical_json_sha256(
                            projection
                        ),
                        "semantic_visibility": visibility,
                        "historical_semantic_identity_reused": False,
                        "historical_gt_srac_identity_preserved": False,
                        "historical_gt_srac_reuse_authorized": False,
                        "historical_gt_srac_reuse_status": "NEW_CASE_REQUIRED",
                        "condition_review_invocation": {
                            "execution_mode": "NOT_APPLICABLE",
                            "invocation_status": "NOT_RUN_NEW_PROVISIONAL_IDENTITY",
                            "raw_call_sha256": None,
                        },
                        "primary_human_facing_question": None,
                        "question_candidates": [],
                        "target_question_candidates": [],
                        "target_selection_required": True,
                        "question_authoring_invocation": {
                            "execution_mode": "NOT_RUN",
                            "invocation_status": "NOT_RUN_SLOT_NOT_READY",
                            "raw_call_sha256": None,
                        },
                        "semantic_fidelity_status": "NOT_RUN",
                        "hccq_status": "NOT_RUN",
                        "final_flow_expert_review_status": "NOT_RUN",
                        "evidence_sufficiency": "PROVISIONAL",
                        "blocking_reason_codes": [
                            "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL"
                        ],
                        "evidence_gaps": [
                            "Human approval and formal grounding are required before case construction."
                        ],
                        "question_candidate_omission_reason": None,
                        "GT_SRAC_IMPACT": "NEW_CASE_REQUIRED",
                        "support_dimensions": support_dimensions,
                        "provisional_target_status": "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL",
                        "combustor_target_review": copy.deepcopy(
                            combustor_target_review
                        ),
                        # Private construction-only fields.  They are removed
                        # before serialization but let the shared authoring
                        # worker use a new identity without loading old cases.
                        "_authoring_case_id": author_case_id,
                        "_authoring_context": authoring_context,
                        "_scientific_evidence": copy.deepcopy(evidence),
                        "_provisional_target_id": str(
                            recommended_combustor_target.get("candidate_id", "")
                        ),
                    }
                    for support_name in SUPPORT_DIMENSION_NAMES:
                        row[support_name] = support_dimensions[support_name]
                    rows.append(row)
                    source_rows_by_dataset.setdefault(dataset_id, []).append(row)
                    selected_combustor_rows.append(row)
                validate_family_condition_semantics(selected_combustor_rows)
                continue
            for condition in CONDITIONS:
                target_question_candidates = [
                    _combustor_question_candidate(target, condition, index)
                    for index, target in enumerate(combustor_candidates, start=1)
                ]
                rows.append(
                    {
                        "case_id": None,
                        "slot_id": f"combustor_density_redesign_pending_{condition.casefold().replace('-', '_')}",
                        "dataset_id": dataset_id,
                        "family_id": policy["family_id"],
                        "concept_id": None,
                        "condition": condition,
                        "matrix_status": "NEW_CASE_PENDING_HUMAN_SELECTION",
                        "QUESTION_CANDIDATE_STATUS": "NEEDS_TARGET_REDESIGN",
                        "SCIENTIFIC_RELEASE_STATUS": "NEW_CASE_REQUIRED",
                        "FORMAL_CANDIDATE_STATUS": "WITHDRAWN_NEW_CASE_REQUIRED",
                        "ARTIFACT_REUSE_STATUS": "REBUILD_REQUIRED",
                        "CONDITION_SCIENTIFIC_STATUS": (
                            "NEW_SCIENTIFIC_IDENTITY_PENDING_HUMAN_SELECTION"
                        ),
                        "scientific_target": None,
                        "scientific_scope": None,
                        "fixed_operationalization": [],
                        "fixed_o_display": [],
                        "unresolved_operationalization_dimensions": [],
                        "open_o_display": [],
                        "finding_responsibility": None,
                        "semantic_contract_sha256": None,
                        "semantic_visibility": None,
                        "support_dimensions": {
                            name: "NOT_ESTABLISHED" for name in SUPPORT_DIMENSION_NAMES
                        },
                        "historical_semantic_identity_reused": False,
                        "historical_gt_srac_identity_preserved": False,
                        "historical_gt_srac_reuse_authorized": False,
                        "historical_gt_srac_reuse_status": "NEW_CASE_REQUIRED",
                        "condition_review_invocation": {
                            "execution_mode": "NOT_APPLICABLE",
                            "invocation_status": "NOT_RUN_NEW_IDENTITY_PENDING_SELECTION",
                            "raw_call_sha256": None,
                        },
                        "primary_human_facing_question": None,
                        # Target proposals are kept in their own audit surface;
                        # they are not authored condition questions and must
                        # never enter candidate or human-review counters.
                        "question_candidates": [],
                        "target_question_candidates": copy.deepcopy(target_question_candidates),
                        "target_selection_required": True,
                        "question_authoring_invocation": {
                            "execution_mode": "NOT_APPLICABLE",
                            "invocation_status": "NOT_RUN_NEW_IDENTITY_PENDING_SELECTION",
                            "raw_call_sha256": None,
                        },
                        "semantic_fidelity_status": "NOT_APPLICABLE",
                        "hccq_status": "NOT_APPLICABLE",
                        "evidence_sufficiency": "PENDING_HUMAN_TARGET_SELECTION",
                        "blocking_reason_codes": [
                            "PREVIOUS_DENSITY_TARGET_REJECTED",
                            "NEW_TARGET_REQUIRES_HUMAN_SELECTION",
                        ],
                        "evidence_gaps": [
                            "Select one evidence-supported stable density target before case authoring."
                        ],
                        "question_candidate_omission_reason": (
                            "ONE_RECOMMENDED_TARGET_WITH_FOUR_AUTHORED_CONDITION_QUESTIONS_REQUIRED"
                            if target_question_candidates
                            else "NO_EVIDENCE_GROUNDED_TARGET_CANDIDATE_AVAILABLE"
                        ),
                        "GT_SRAC_IMPACT": "NEW_CASE_REQUIRED",
                        "canonical_case_mutated": False,
                        "combustor_target_review": copy.deepcopy(
                            combustor_target_review
                        ),
                    }
                )
                for support_name in SUPPORT_DIMENSION_NAMES:
                    rows[-1][support_name] = rows[-1]["support_dimensions"][support_name]
            continue
        family_id = str(policy["family_id"])
        family = families[family_id]
        dataset_rows: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            case_id = str(policy["case_ids"][condition])
            packet = packet_by_case[case_id]
            projection = packet["canonical_semantic_projection"]
            validation = reviews[case_id]
            matrix_status, scientific_status, reasons, gaps = _compile_slot_status(
                validation, condition
            )
            review = validation.get("normalized_review", {})
            gt_srac_impact = gt_srac_impact_for_slot(matrix_status)
            fixed = projection["resolved_operationalization"]
            unresolved = projection["unresolved_operationalization_dimensions"]
            record = load_case_record(root, case_id)
            authoring_context = _dataset_context_for_authoring(
                record, capabilities.get(dataset_id, {})
            )
            visibility = _semantic_visibility(projection, authoring_context)
            row = {
                "case_id": case_id,
                "slot_id": case_id,
                "dataset_id": dataset_id,
                "family_id": family_id,
                "concept_id": family.get("concept_id"),
                "condition": condition,
                "matrix_status": matrix_status,
                # Public, decoupled state names.  ``matrix_status`` remains as
                # a compatibility alias for callers of the earlier matrix.
                "QUESTION_CANDIDATE_STATUS": _question_candidate_status(matrix_status),
                "SCIENTIFIC_RELEASE_STATUS": _scientific_release_status(matrix_status),
                "ARTIFACT_REUSE_STATUS": _artifact_reuse_status(matrix_status),
                "CONDITION_SCIENTIFIC_STATUS": scientific_status,
                "scientific_target": projection["scientific_target"],
                "scientific_scope": projection["scientific_scope"],
                "fixed_operationalization": fixed,
                "fixed_o_display": [
                    f"{item['dimension_id']}: {item['normalized_meaning']}"
                    for item in fixed
                ],
                "unresolved_operationalization_dimensions": unresolved,
                "open_o_display": [
                    str(item.get("dimension_id", ""))
                    if isinstance(item, Mapping)
                    else str(item)
                    for item in unresolved
                ],
                "finding_responsibility": projection["finding_responsibility"],
                "canonical_semantic_projection": projection,
                "semantic_contract_sha256": packet["semantic_contract_sha256"],
                "semantic_visibility": visibility,
                "historical_semantic_identity_reused": True,
                **_historical_dependency_state(matrix_status, gt_srac_impact),
                "condition_scientific_review": validation,
                "condition_review_invocation": validation.get(
                    "review_invocation", {}
                ),
                "evidence_sufficiency": review.get(
                    "evidence_sufficiency", "NOT_ESTABLISHED"
                ),
                "blocking_reason_codes": reasons,
                "evidence_gaps": gaps,
                "support_dimensions": copy.deepcopy(
                    review.get(
                        "support_dimensions",
                        {name: "NOT_ESTABLISHED" for name in SUPPORT_DIMENSION_NAMES},
                    )
                ),
                "GT_SRAC_IMPACT": gt_srac_impact,
                "primary_human_facing_question": None,
                "question_candidates": [],
                "question_authoring_invocation": {
                    "execution_mode": "NOT_RUN",
                    "invocation_status": "NOT_RUN_SLOT_NOT_READY",
                    "raw_call_sha256": None,
                },
                "semantic_fidelity_status": "NOT_RUN",
                "hccq_status": "NOT_RUN",
                "canonical_case_mutated": False,
            }
            # Keep each dimension addressable without requiring consumers to
            # unpack the convenience mapping.
            for support_name in SUPPORT_DIMENSION_NAMES:
                row[support_name] = row["support_dimensions"].get(
                    support_name, "NOT_ESTABLISHED"
                )
            dataset_rows.append(row)
            rows.append(row)
        validate_family_condition_semantics(dataset_rows)
        source_rows_by_dataset[dataset_id] = dataset_rows

    resumed_authored_row_count = 0
    if resume_rows_by_case or resume_rows_by_slot:
        authored_fields = (
            "question_candidates",
            "primary_human_facing_question",
            "primary_candidate_id",
            "primary_candidate_nomination",
            "human_final_selection_required",
            "human_question_selected",
            "question_authoring_invocation",
            "question_authoring_attempt_count",
            "question_authoring_attempts",
            "semantic_fidelity_status",
            "hccq_status",
            "final_flow_expert_review_status",
            "final_flow_expert_wording_review",
            "question_candidate_generation",
            "question_candidate_generation_error",
        )
        for row in rows:
            resume_row = resume_rows_by_slot.get(
                str(row.get("slot_id") or "").strip()
            ) or resume_rows_by_case.get(str(row.get("case_id") or "").strip())
            if not isinstance(resume_row, Mapping):
                continue
            if str(resume_row.get("semantic_contract_sha256") or "") != str(
                row.get("semantic_contract_sha256") or ""
            ):
                continue
            if not _resume_authored_row_is_complete(
                resume_row,
                candidate_count=candidate_count,
                require_live=run_live,
                enforce_human_facing_acceptance=enforce_human_facing_acceptance,
            ):
                continue
            for field in authored_fields:
                if field in resume_row:
                    row[field] = copy.deepcopy(resume_row[field])
            row["question_authoring_resume"] = {
                "status": "REUSED_VALIDATED_AUTHORING_TRACE",
                "source_manifest_sha256": resume_manifest_sha256,
                "semantic_contract_sha256": row.get("semantic_contract_sha256"),
            }
            row["_resume_complete"] = True
            resumed_authored_row_count += 1

    authorable_rows = [
        row
        for row in rows
        if (
            str(row.get("case_id") or row.get("_authoring_case_id") or "").strip()
        )
        and row.get("matrix_status") != "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
        and (
            row.get("dataset_id") != "Combustor"
            or row.get("_authoring_case_id")
        )
        and not row.get("_resume_complete")
    ]
    if question_author is None or fidelity_reviewer is None:
        # Build-only expansion has an existing deterministic semantic renderer;
        # a live wording author is not a prerequisite for human visibility.
        question_author = None
        fidelity_reviewer = None

    def invoke_final_review(review_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Call the optional final wording reviewer with an explicit envelope.

        Injected reviewers in the unit-test/integration path commonly return a
        bare structured JSON object.  The final-review provenance contract,
        however, needs an explicit successful invocation marker.  Add that
        transport metadata at this boundary; live callers already provide a
        richer envelope and are left unchanged.
        """

        if final_reviewer is None:
            return {}
        raw = final_reviewer(copy.deepcopy(review_packet))
        if not isinstance(raw, Mapping):
            return {"raw_result": raw}
        envelope = dict(raw)
        # ``audit_final_flow_expert_wording`` treats a bare mapping as the
        # model result and therefore rejects transport-only keys such as
        # ``invocation_status`` as unknown output fields.  Put a direct
        # injected result under the same ``parsed_result`` envelope used by
        # the live provider path before adding provenance metadata.
        if "parsed_result" not in envelope and any(
            key in envelope
            for key in (
                "question_meaningful",
                "meaningful",
                "target_stable",
                "target_wording_faithful",
                "recommendation",
            )
        ):
            model_result = dict(envelope)
            envelope = {"parsed_result": model_result}
        envelope.setdefault(
            "execution_mode", "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWER"
        )
        envelope.setdefault("invocation_status", "SUCCESS")
        return envelope

    def nomination_responsibility_gate(candidate: Mapping[str, Any]) -> bool:
        """Reject an explicitly failed responsibility audit before nomination.

        Semantic fidelity and the dedicated final wording review are the
        canonical model-backed checks.  This small guard prevents a producer
        from attaching a contradictory, already-computed responsibility audit
        and then nominating the candidate solely because its top-level status
        says ``PASS``.  Missing audits remain compatible with the build-only
        and legacy injected paths; they are still subject to the normal
        fidelity/final-review gates.
        """

        audit = candidate.get("wording_responsibility_audit")
        if isinstance(audit, Mapping):
            audit_status = str(audit.get("status", "")).strip().upper()
            if audit_status and audit_status not in {"PASS", "NOT_ESTABLISHED"}:
                return False
            failure_codes = audit.get("failure_codes")
            if isinstance(failure_codes, list) and any(
                str(code).strip() for code in failure_codes
            ):
                return False

        fidelity = candidate.get("post_rewrite_semantic_fidelity")
        if isinstance(fidelity, Mapping):
            if fidelity.get("no_additional_scientific_obligation") is False:
                return False
            obligations = fidelity.get("additional_scientific_obligations")
            if isinstance(obligations, list) and obligations:
                return False
            failure_codes = fidelity.get("failure_codes")
            if isinstance(failure_codes, list) and "EXTRA_SCIENTIFIC_OBLIGATION" in {
                str(code).strip().upper() for code in failure_codes
            }:
                return False

        final_review = candidate.get("final_flow_expert_wording_review")
        if isinstance(final_review, Mapping):
            checks = final_review.get("checks")
            if isinstance(checks, Mapping) and checks.get(
                "no_unintended_o_responsibility"
            ) is False:
                return False
            failure_codes = final_review.get("failure_codes")
            if isinstance(failure_codes, list) and "UNDECLARED_O_RESPONSIBILITY" in {
                str(code).strip().upper() for code in failure_codes
            }:
                return False
        return True

    def nominate_audited_candidates(
        audited: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Nominate only candidates that passed every enabled presentation gate."""

        responsibility_clean = [
            candidate for candidate in audited if nomination_responsibility_gate(candidate)
        ]
        if final_reviewer is None:
            nomination = nominate_primary_question_candidate(responsibility_clean)
            nomination["responsibility_rejected_candidate_ids"] = [
                str(candidate.get("candidate_id"))
                for candidate in audited
                if not nomination_responsibility_gate(candidate)
            ]
            return nomination

        # ``nominate_primary_question_candidate`` remains the canonical
        # fidelity/HCCQ selector.  Filtering here adds the independent final
        # wording gate without changing its backwards-compatible API.
        final_eligible = [
            candidate
            for candidate in responsibility_clean
            if str(candidate.get("flow_expert_review_status", "")).upper() == "PASS"
        ]
        nomination = nominate_primary_question_candidate(final_eligible)
        nomination.update(
            {
                "final_flow_expert_review_required": True,
                "final_review_eligible_candidate_ids": [
                    str(candidate.get("candidate_id")) for candidate in final_eligible
                ],
                "final_review_rejected_candidate_ids": [
                    str(candidate.get("candidate_id"))
                    for candidate in audited
                    if candidate not in final_eligible
                ],
                "responsibility_rejected_candidate_ids": [
                    str(candidate.get("candidate_id"))
                    for candidate in audited
                    if not nomination_responsibility_gate(candidate)
                ],
            }
        )
        return nomination

    def author_and_audit(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        case_id = str(row.get("_authoring_case_id") or row.get("case_id"))
        if row.get("_authoring_case_id"):
            # Provisional Combustor identities have no canonical case yet.  Use
            # only the dataset context/evidence explicitly staged for the
            # provisional target; never load or expose an old case/GT/SRAC.
            projection = copy.deepcopy(row["canonical_semantic_projection"])
            context = copy.deepcopy(row.get("_authoring_context") or {})
        else:
            record = load_case_record(root, case_id)
            projection = row["canonical_semantic_projection"]
            context = _dataset_context_for_authoring(
                record, capabilities.get(str(row["dataset_id"]), {})
            )
        visibility = row["semantic_visibility"]
        packet = build_question_presentation_authoring_packet(
            projection,
            dataset_context=context,
            candidate_count=candidate_count,
            case_id=case_id,
            condition=str(row["condition"]),
            semantic_visibility=visibility,
            semantic_contract_sha256=str(row["semantic_contract_sha256"]),
        )
        authoring_attempts: list[dict[str, Any]] = []
        repair_feedback: dict[str, Any] | None = None
        audited: list[dict[str, Any]] = []
        nomination: dict[str, Any] = {
            "status": "NO_ELIGIBLE_PRIMARY_CANDIDATE",
            "primary_candidate_id": None,
            "primary_model_visible_text": None,
        }
        row_invocation: dict[str, Any] = {
            "execution_mode": "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWER",
            "invocation_status": "NOT_RUN",
            "raw_call_sha256": None,
        }
        # The portfolio-closure controller may need more than one bounded
        # rewrite.  Six is an engineering convergence ceiling, not a relaxed
        # scientific threshold.  Three consecutive identical normalized
        # defect sets stop blind rewriting and expose a root-cause diagnosis.
        defect_history: list[tuple[str, ...]] = []
        for authoring_attempt in range(1, max_authoring_attempts + 1):
            feedback_applied = copy.deepcopy(repair_feedback)
            try:
                authored = author_human_questions(
                    projection,
                    dataset_context=context,
                    author=question_author,  # type: ignore[arg-type]
                    candidate_count=candidate_count,
                    case_id=case_id,
                    condition=str(row["condition"]),
                    semantic_visibility=visibility,
                    semantic_contract_sha256=str(row["semantic_contract_sha256"]),
                    repair_feedback=feedback_applied,
                )
            except Exception as exc:
                error_record = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                row_invocation = {
                    "execution_mode": (
                        "LIVE_FLOW_EXPERT" if run_live else "INJECTED_REVIEWER"
                    ),
                    "invocation_status": "EXCEPTION",
                    "raw_call_sha256": _json_sha256(error_record),
                }
                authoring_attempts.append(
                    {
                        "attempt": authoring_attempt,
                        "repair_feedback_applied": feedback_applied,
                        "authoring_invocation": copy.deepcopy(row_invocation),
                        "candidate_outcomes": [],
                        "nomination_status": "AUTHORING_EXCEPTION",
                        "error": error_record,
                    }
                )
                continue
            # Normalize the author transport once per attempt before attaching
            # it to candidates.  This makes every authored candidate
            # independently auditable instead of relying on a row-level
            # invocation field during the later portfolio handoff.
            invocation = _normalise_question_authoring_invocation(
                authored,
                row=row,
                run_live=run_live,
            )
            audited: list[dict[str, Any]] = []
            for candidate in authored.get("candidates", ()):
                try:
                    fidelity = audit_post_rewrite_semantic_fidelity(
                        projection,
                        str(candidate["model_visible_text"]),
                        reviewer=fidelity_reviewer,  # type: ignore[arg-type]
                        dataset_context=context,
                        case_id=case_id,
                        condition=str(row["condition"]),
                        semantic_visibility=visibility,
                        semantic_contract_sha256=str(row["semantic_contract_sha256"]),
                    )
                except Exception as exc:
                    fidelity = {
                        "status": "INFRA_INVALID",
                        "failure_codes": [],
                        "validation_errors": [
                            f"FIDELITY_REVIEW_EXCEPTION:{type(exc).__name__}:{exc}"
                        ],
                        "additional_scientific_obligations": [],
                        "excluded_from_semantic_fidelity_decision": True,
                    }
                try:
                    hccq = audit_human_presentation(_hccq_shape(candidate, packet))
                except Exception as exc:
                    hccq = {
                        "status": "INCOMPLETE",
                        "natural_readability_signals": [
                            f"HCCQ_EXCEPTION:{type(exc).__name__}:{exc}"
                        ],
                    }
                audited_candidate: dict[str, Any] = {
                    **dict(candidate),
                    "post_rewrite_semantic_fidelity": fidelity,
                    "hccq_presentation": hccq,
                    "candidate_source": "FLOW_EXPERT_PRESENTATION_AUTHORING",
                    "question_authoring_invocation": copy.deepcopy(invocation),
                }
                if enforce_human_facing_acceptance:
                    responsibility_audit = audit_question_responsibility_realization(
                        projection,
                        str(candidate["model_visible_text"]),
                        candidate={
                            "fixed_operationalization": copy.deepcopy(
                                projection.get("resolved_operationalization", ())
                            ),
                            "unresolved_operationalization_dimensions": copy.deepcopy(
                                projection.get(
                                    "unresolved_operationalization_dimensions", ()
                                )
                            ),
                        "finding_responsibility": str(
                            projection.get("finding_responsibility", "")
                        ),
                        "requested_findings": copy.deepcopy(
                            candidate.get(
                                "requested_findings",
                                projection.get("finding_responsibility", {}),
                            )
                        ),
                    },
                        semantic_visibility=visibility,
                    )
                    audited_candidate["wording_responsibility_audit"] = (
                        responsibility_audit
                    )
                if final_reviewer is not None:
                    # Use the condition packet's evidence only as visible
                    # context.  It contains no Ground Truth or evaluation
                    # result and therefore cannot leak scientific answers.
                    condition_packet = packet_by_case.get(case_id, {})
                    if not condition_packet:
                        condition_packet = {
                            "scientific_evidence": copy.deepcopy(
                                row.get("_scientific_evidence") or {}
                            )
                        }
                    try:
                        final_review = audit_final_flow_expert_wording(
                            projection,
                            str(candidate["model_visible_text"]),
                            reviewer=invoke_final_review,
                            dataset_context=context,
                            evidence_context=copy.deepcopy(
                                condition_packet.get("scientific_evidence", {})
                                if isinstance(condition_packet, Mapping)
                                else {}
                            ),
                            case_id=case_id,
                            condition=str(row["condition"]),
                            semantic_visibility=visibility,
                            semantic_contract_sha256=str(row["semantic_contract_sha256"]),
                        )
                    except Exception as exc:
                        final_review = {
                            "status": "INCOMPLETE",
                            "final_flow_expert_review_status": "NOT_ESTABLISHED",
                            "failure_codes": [
                                f"FINAL_REVIEW_EXCEPTION:{type(exc).__name__}:{exc}"
                            ],
                            "checks": {},
                            "recommendation": None,
                            "excluded_from_scientific_denominator": True,
                        }
                    final_status = str(
                        final_review.get("final_flow_expert_review_status", "")
                    ).upper() or "NOT_ESTABLISHED"
                    # Keep one canonical nested record plus shallow aliases for
                    # portfolio consumers.  The nested record carries the
                    # reviewed text hash and successful call provenance.
                    audited_candidate.update(
                        {
                            "final_flow_expert_wording_review": final_review,
                            "flow_expert_review_status": final_status,
                            "final_flow_expert_review_status": final_status,
                            "final_flow_expert_review_provenance": copy.deepcopy(
                                final_review.get("provenance", {})
                            ),
                        }
                    )
                audited.append(audited_candidate)
            nomination = nominate_audited_candidates(audited)
            # Authoring success is independent of release readiness.  Keep the
            # actual row-level invocation instead of rewriting a successful
            # call as NOT_RUN merely because scientific release is blocked.
            row_invocation = copy.deepcopy(invocation)
            authoring_attempts.append(
                {
                    "attempt": authoring_attempt,
                    "repair_feedback_applied": feedback_applied,
                    "authoring_invocation": copy.deepcopy(invocation),
                    "candidate_outcomes": [
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "semantic_fidelity_status": candidate.get(
                                "post_rewrite_semantic_fidelity", {}
                            ).get("status"),
                            "hccq_status": candidate.get(
                                "hccq_presentation", {}
                            ).get("status"),
                            "final_flow_expert_review_status": candidate.get(
                                "flow_expert_review_status", "NOT_ESTABLISHED"
                            ),
                        }
                        for candidate in audited
                    ],
                    "nomination_status": nomination["status"],
                }
            )
            if nomination["status"] == "PRIMARY_CANDIDATE_NOMINATED":
                break
            repair_feedback = _authoring_repair_feedback(audited)
            normalized_defects = tuple(
                sorted(
                    {
                        str(defect).strip()
                        for repair_candidate in (
                            repair_feedback or {}
                        ).get("candidates", ())
                        if isinstance(repair_candidate, Mapping)
                        for defect in repair_candidate.get("identified_defects", ())
                        if str(defect).strip()
                    }
                )
            )
            defect_history.append(normalized_defects)
            if (
                len(defect_history) >= 3
                and defect_history[-1]
                and defect_history[-1] == defect_history[-2] == defect_history[-3]
            ):
                authoring_attempts[-1]["root_cause_diagnosis_required"] = True
                authoring_attempts[-1]["convergence_stop_reason"] = (
                    "SAME_NORMALIZED_DEFECT_THREE_CONSECUTIVE_REWRITES"
                )
                break
        if nomination["status"] != "PRIMARY_CANDIDATE_NOMINATED":
            return str(row.get("slot_id") or case_id), {
                "question_candidates": [],
                "rejected_authored_question_candidates": _annotate_question_candidates(
                    audited, row
                ),
                "primary_human_facing_question": None,
                "primary_candidate_id": None,
                "primary_candidate_nomination": nomination,
                "human_final_selection_required": True,
                "human_question_selected": False,
                "question_candidate_generation": "FLOW_EXPERT_AUTHORING",
                "question_candidate_generation_error": (
                    "NO_FIDELITY_HCCQ_FINAL_REVIEW_PASSING_PRIMARY"
                ),
                "question_authoring_invocation": row_invocation,
                "question_authoring_attempt_count": len(authoring_attempts),
                "question_authoring_attempts": authoring_attempts,
                "semantic_fidelity_status": "NOT_ESTABLISHED",
                "hccq_status": "NOT_ESTABLISHED",
                "final_flow_expert_review_status": "NOT_ESTABLISHED",
                "semantic_visibility": visibility,
            }
        primary_candidate = next(
            (
                candidate
                for candidate in audited
                if str(candidate.get("candidate_id", ""))
                == str(nomination.get("primary_candidate_id", ""))
            ),
            None,
        )
        return str(row.get("slot_id") or case_id), {
            "question_candidates": _annotate_question_candidates(audited, row),
            "primary_human_facing_question": nomination[
                "primary_model_visible_text"
            ],
            "primary_candidate_id": nomination["primary_candidate_id"],
            "primary_candidate_nomination": nomination,
            "human_final_selection_required": True,
            "human_question_selected": False,
            "question_authoring_invocation": row_invocation,
            "question_authoring_attempt_count": len(authoring_attempts),
            "question_authoring_attempts": authoring_attempts,
            "semantic_fidelity_status": "PASS",
            "hccq_status": "PASS",
            "final_flow_expert_review_status": (
                str(primary_candidate.get("flow_expert_review_status", "")).upper()
                if isinstance(primary_candidate, Mapping)
                else "NOT_ESTABLISHED"
            ),
            "final_flow_expert_wording_review": (
                copy.deepcopy(primary_candidate.get("final_flow_expert_wording_review"))
                if isinstance(primary_candidate, Mapping)
                and isinstance(
                    primary_candidate.get("final_flow_expert_wording_review"), Mapping
                )
                else None
            ),
            "semantic_visibility": visibility,
        }

    authored_results: dict[str, dict[str, Any]] = {}
    if authorable_rows and question_author is not None and fidelity_reviewer is not None:
        with ThreadPoolExecutor(
            max_workers=max(1, min(max_workers, len(authorable_rows)))
        ) as pool:
            futures = {pool.submit(author_and_audit, row): row for row in authorable_rows}
            for future in as_completed(futures):
                row = futures[future]
                result_key = str(row.get("slot_id") or row.get("case_id") or "")
                try:
                    result_case_id, result = future.result()
                    authored_results[result_case_id] = result
                except Exception as exc:
                    authored_results[result_key] = {
                        "question_candidates": [],
                        "question_candidate_generation": "FLOW_EXPERT_AUTHORING",
                        "question_candidate_generation_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                        # The worker can fail before ``author_and_audit`` has
                        # entered its bounded attempt loop (for example while
                        # materialising a provisional authoring context).  A
                        # marker is therefore needed in addition to the
                        # attempt list: the fallback renderer must remain
                        # recovery-only even when no attempt record exists.
                        "_authoring_pipeline_attempted": True,
                        "question_authoring_invocation": {
                            "execution_mode": "INJECTED_REVIEWER"
                            if not run_live
                            else "LIVE_FLOW_EXPERT",
                            "invocation_status": "EXCEPTION",
                            "raw_call_sha256": _json_sha256(
                                {"error_type": type(exc).__name__, "error": str(exc)}
                            ),
                        },
                    }
        for row in authorable_rows:
            result_key = str(row.get("slot_id") or row.get("case_id") or "")
            result = authored_results.get(result_key)
            if result:
                row.update(result)

    # Every existing-identity slot gets a review surface regardless of release
    # status.  Live/injected authors are used for eligible slots when present;
    # all remaining slots use the deterministic semantic renderer.  An omitted
    # scientifically unsupported identity is kept as an explicit omission; it
    # must not receive a weak wording merely to improve matrix coverage.
    for row in rows:
        if (
            row.get("question_candidates")
            or (
                row.get("dataset_id") == "Combustor"
                and not isinstance(row.get("canonical_semantic_projection"), Mapping)
            )
        ):
            continue
        if row.get("matrix_status") == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
            row["question_candidate_omission_reason"] = (
                "SCIENTIFIC_TARGET_OR_OBSERVABLE_UNSUPPORTED"
            )
            continue
        result = _deterministic_question_candidates(
            root, row, capabilities, candidate_count
        )
        result["question_candidates"] = _annotate_question_candidates(
            result.get("question_candidates", ()), row
        )
        # Deterministic wording is recovery-only.  Keep the failed live
        # authoring trace and its diagnostics intact so a portfolio audit can
        # distinguish provider/audit failure from a slot that was never
        # attempted.  The recovery renderer gets its own explicit fields.
        live_generation_error = row.get("question_candidate_generation_error")
        live_invocation = copy.deepcopy(row.get("question_authoring_invocation"))
        live_attempts = copy.deepcopy(row.get("question_authoring_attempts"))
        live_attempt_count = row.get("question_authoring_attempt_count")

        # A failed real/injected authoring run is not equivalent to a build-only
        # run.  In the former case the deterministic renderer is recovery
        # material only: putting it back in ``question_candidates`` (or making
        # it the primary question) would make a failed live authoring call
        # appear to have produced a genuine human-review question.  Preserve
        # the authored failure trace and keep the fallback in its dedicated
        # audit-only field.  A build-only matrix has no authoring trace and may
        # continue to expose the deterministic renderer for compatibility.
        authoring_attempted = bool(row.get("_authoring_pipeline_attempted")) or (
            isinstance(live_attempts, list) and bool(live_attempts)
        )
        if authoring_attempted:
            row.update(result)
            row["recovery_question_candidates"] = copy.deepcopy(
                result.get("question_candidates", [])
            )
            row["recovery_question_candidate_generation"] = result.get(
                "question_candidate_generation"
            )
            row["recovery_question_candidate_generation_error"] = result.get(
                "question_candidate_generation_error"
            )
            # Never expose recovery wording through the genuine question
            # handles after an authoring attempt has failed.
            row["question_candidates"] = []
            row["primary_human_facing_question"] = None
            row["primary_candidate_id"] = None
            row["primary_candidate_nomination"] = {
                "status": "NO_ELIGIBLE_PRIMARY_CANDIDATE",
                "primary_candidate_id": None,
                "primary_model_visible_text": None,
                "nomination_only": True,
                "human_final_selection_required": True,
                "human_question_selected": False,
            }
            # Keep the actual authoring result/error and invocation envelope;
            # only the recovery renderer is annotated below.
            if live_generation_error:
                row["live_question_candidate_generation_error"] = (
                    live_generation_error
                )
                row["question_candidate_generation_error"] = live_generation_error
            if isinstance(live_invocation, Mapping):
                row["question_authoring_invocation"] = live_invocation
            if isinstance(live_attempts, list):
                row["question_authoring_attempts"] = live_attempts
                row["question_authoring_attempt_count"] = (
                    live_attempt_count
                    if isinstance(live_attempt_count, int)
                    else len(live_attempts)
                )
            # The remaining status normalization below must see an empty
            # genuine candidate list and must not turn recovery prose into a
            # primary handle.
            continue

        row["recovery_question_candidates"] = copy.deepcopy(
            result.get("question_candidates", [])
        )
        row["recovery_question_candidate_generation"] = result.get(
            "question_candidate_generation"
        )
        row["recovery_question_candidate_generation_error"] = result.get(
            "question_candidate_generation_error"
        )
        row.update(result)
        if live_generation_error:
            row["live_question_candidate_generation_error"] = live_generation_error
            row["question_candidate_generation_error"] = live_generation_error
        if isinstance(live_invocation, Mapping) and str(
            live_invocation.get("invocation_status", "")
        ).upper() not in {"", "NOT_RUN", "NOT_RUN_SLOT_NOT_READY"}:
            row["question_authoring_invocation"] = live_invocation
        if isinstance(live_attempts, list):
            row["question_authoring_attempts"] = live_attempts
            row["question_authoring_attempt_count"] = (
                live_attempt_count if isinstance(live_attempt_count, int) else len(live_attempts)
            )
        candidates = row.get("question_candidates", ())
        if candidates:
            fidelity_statuses = {
                str(candidate.get("post_rewrite_semantic_fidelity", {}).get("status", ""))
                for candidate in candidates
                if isinstance(candidate, Mapping)
            }
            hccq_statuses = {
                str(candidate.get("hccq_presentation", {}).get("status", ""))
                for candidate in candidates
                if isinstance(candidate, Mapping)
            }
            row["semantic_fidelity_status"] = (
                "PASS" if fidelity_statuses == {"PASS"} else "REVISE"
            )
            row["hccq_status"] = (
                "PASS" if hccq_statuses == {"PASS"} else "REVISE_HUMAN_INTERFACE"
            )
            if row["semantic_fidelity_status"] != "PASS":
                row["QUESTION_CANDIDATE_STATUS"] = "NEEDS_SCIENTIFIC_REVISION"
        else:
            row["question_candidate_omission_reason"] = str(
                result.get("question_candidate_generation_error")
                or "NO_REVIEWABLE_QUESTION_CANDIDATE"
            )

    # Normalize candidate-level state for both live/injected and deterministic
    # paths.  This is deliberately response-level: release and artifact states
    # are not changed by wording audits.
    for row in rows:
        candidates = [
            item for item in row.get("question_candidates", ())
            if isinstance(item, Mapping)
        ]
        if row.get("dataset_id") == "Combustor" and not row.get("_authoring_case_id"):
            row.setdefault("question_candidate_omission_reason", None)
            continue
        if row.get("matrix_status") == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
            row["QUESTION_CANDIDATE_STATUS"] = "NEEDS_SCIENTIFIC_REVISION"
            row.setdefault(
                "question_candidate_omission_reason",
                "SCIENTIFIC_TARGET_OR_OBSERVABLE_UNSUPPORTED",
            )
            continue
        if not candidates:
            row["QUESTION_CANDIDATE_STATUS"] = (
                "NEEDS_TARGET_REDESIGN"
                if row.get("provisional_target_status")
                == "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL"
                else "NEEDS_EVIDENCE_CONTEXT"
            )
            row.setdefault(
                "question_candidate_omission_reason",
                "PROVISIONAL_TARGET_AUTHORING_INCOMPLETE"
                if row.get("provisional_target_status")
                == "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL"
                else "NO_REVIEWABLE_QUESTION_CANDIDATE",
            )
            continue
        fidelity = {
            str(item.get("post_rewrite_semantic_fidelity", {}).get("status", ""))
            for item in candidates
        }
        hccq = {
            str(item.get("hccq_presentation", {}).get("status", ""))
            for item in candidates
        }
        row["semantic_fidelity_status"] = "PASS" if fidelity == {"PASS"} else "REVISE"
        row["hccq_status"] = "PASS" if hccq == {"PASS"} else "REVISE_HUMAN_INTERFACE"
        if final_reviewer is not None:
            final_status = {
                str(item.get("flow_expert_review_status", "")).upper()
                for item in candidates
            }
            row["final_flow_expert_review_status"] = (
                "PASS" if "PASS" in final_status else "NOT_ESTABLISHED"
            )
            row["QUESTION_CANDIDATE_STATUS"] = (
                "READY_FOR_HUMAN_REVIEW"
                if fidelity == {"PASS"}
                and hccq == {"PASS"}
                and "PASS" in final_status
                else "NEEDS_SCIENTIFIC_REVISION"
            )
        else:
            row["QUESTION_CANDIDATE_STATUS"] = (
                "READY_FOR_HUMAN_REVIEW"
                if fidelity == {"PASS"} and hccq == {"PASS"}
                else "NEEDS_SCIENTIFIC_REVISION"
            )

    # The selected target is a family-level object.  Attach exactly one
    # primary question per condition plus the complete two-candidate authoring
    # audit, so downstream human_review_portfolio can consume the same
    # condition-specific records without reconstructing a lossy join.
    if recommended_combustor_target is not None:
        condition_questions: dict[str, dict[str, Any]] = {}
        condition_question_candidates: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("dataset_id") != "Combustor":
                continue
            condition = str(row.get("condition", ""))
            candidates = [
                copy.deepcopy(item)
                for item in row.get("question_candidates", ())
                if isinstance(item, Mapping)
            ]
            if candidates:
                condition_question_candidates[condition] = candidates
                primary_id = str(row.get("primary_candidate_id", ""))
                primary = next(
                    (
                        item for item in candidates
                        if str(item.get("candidate_id", "")) == primary_id
                    ),
                    candidates[0],
                )
                condition_questions[condition] = primary
        if condition_questions:
            recommended_combustor_target["condition_questions"] = condition_questions
        if condition_question_candidates:
            recommended_combustor_target[
                "condition_question_candidates"
            ] = condition_question_candidates
        # ``recommended_combustor_target`` is a defensive copy returned by
        # the selector; update the corresponding public candidate row as well
        # so the serialized target manifest and the in-memory selection expose
        # one identical family-level record.
        selected_id = str(recommended_combustor_target.get("candidate_id", ""))
        for candidate in combustor_candidates:
            if str(candidate.get("candidate_id", "")) == selected_id:
                candidate.update(
                    {
                        key: copy.deepcopy(recommended_combustor_target[key])
                        for key in (
                            "condition_questions",
                            "condition_question_candidates",
                            "target_selection_review",
                            "provisional_family_contract",
                        )
                        if key in recommended_combustor_target
                    }
                )
                break

    # Private staging values are needed only while the shared authoring worker
    # runs.  Never expose dataset context/evidence staging internals in the
    # portable review artifact.
    for row in rows:
        for key in tuple(row):
            if str(key).startswith("_"):
                row.pop(key, None)

    family_leakage: dict[str, Any] = {}
    for dataset_id, dataset_rows in source_rows_by_dataset.items():
        primary_by_condition = {
            str(row["condition"]): row["primary_human_facing_question"]
            for row in dataset_rows
            if row.get("primary_human_facing_question")
        }
        family_leakage[dataset_id] = audit_family_presentation_leakage(
            primary_by_condition
        )

    counts = {
        status: sum(row["matrix_status"] == status for row in rows)
        for status in MATRIX_SLOT_STATUSES
    }
    question_candidate_coverage = all(
        bool(row.get("question_candidates"))
        or bool(str(row.get("question_candidate_omission_reason", "")).strip())
        for row in rows
    )
    dataset_summaries: list[dict[str, Any]] = []
    for dataset_id in DATASET_POLICIES:
        dataset_rows = [row for row in rows if row["dataset_id"] == dataset_id]
        slots = {str(row["condition"]): str(row["matrix_status"]) for row in dataset_rows}
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "condition_slots": slots,
                "retained_count": sum(
                    status == "CANDIDATE_FOR_HUMAN_REVIEW"
                    for status in slots.values()
                ),
                "non_omitted_slot_count": sum(
                    status != "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
                    for status in slots.values()
                ),
                "human_review_ready_count": sum(
                    status == "CANDIDATE_FOR_HUMAN_REVIEW"
                    for status in slots.values()
                ),
            }
        )
    manifest = {
        "schema_version": SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION,
        "execution_mode": "LIVE_FLOW_EXPERT" if run_live else (
            "INJECTED_REVIEWERS" if condition_reviewer is not None else "BUILD_ONLY"
        ),
        "final_flow_expert_review_enabled": final_reviewer is not None,
        "final_flow_expert_review_contract": (
            "actual_model_visible_text_reviewed_after_authoring_fidelity_and_hccq"
            if final_reviewer is not None
            else "NOT_RUN"
        ),
        "resume_manifest_used": resume_manifest_path is not None,
        "resume_manifest_sha256": resume_manifest_sha256,
        "resumed_validated_authored_row_count": resumed_authored_row_count,
        "combustor_target_review_resumed": resumed_combustor_target_review,
        "maximum_candidate_slots": 28,
        "maximum_is_quota": False,
        "dataset_count": len(DATASET_POLICIES),
        "condition_slot_count": len(rows),
        "candidate_count_definition": (
            "PROPOSED = READY + BLOCKED_PENDING_EVIDENCE + "
            "BLOCKED_PENDING_O_SPACE_REVIEW + NEW_PENDING_SELECTION"
        ),
        "TOTAL_CANDIDATES_PROPOSED": len(rows)
        - counts["OMITTED_SCIENTIFICALLY_UNSUPPORTED"],
        "TOTAL_CANDIDATES_FOR_HUMAN_REVIEW": counts[
            "CANDIDATE_FOR_HUMAN_REVIEW"
        ],
        "TOTAL_BLOCKED_PENDING_EVIDENCE": counts["BLOCKED_PENDING_EVIDENCE"],
        "TOTAL_BLOCKED_PENDING_O_SPACE_REVIEW": counts[
            "BLOCKED_PENDING_O_SPACE_REVIEW"
        ],
        "TOTAL_NEW_CASE_PENDING_SELECTION": counts[
            "NEW_CASE_PENDING_HUMAN_SELECTION"
        ],
        "TOTAL_OMITTED": counts["OMITTED_SCIENTIFICALLY_UNSUPPORTED"],
        "TOTAL_SLOTS_ANALYZED": len(rows),
        "TOTAL_HUMAN_REVIEW_CANDIDATES": sum(
            bool(row.get("question_candidates")) for row in rows
        ),
        "TOTAL_RELEASE_READY": sum(
            row.get("SCIENTIFIC_RELEASE_STATUS") == "ELIGIBLE" for row in rows
        ),
        "TOTAL_PENDING_EVIDENCE": sum(
            row.get("SCIENTIFIC_RELEASE_STATUS") == "BLOCKED_PENDING_EVIDENCE"
            for row in rows
        ),
        "TOTAL_PENDING_O_SPACE": sum(
            row.get("SCIENTIFIC_RELEASE_STATUS") == "BLOCKED_PENDING_O_SPACE"
            for row in rows
        ),
        "TOTAL_NEW_TARGET_REQUIRED": sum(
            row.get("SCIENTIFIC_RELEASE_STATUS") == "NEW_CASE_REQUIRED"
            for row in rows
        ),
        "question_candidate_coverage_invariant_satisfied": question_candidate_coverage,
        "support_dimension_names": list(SUPPORT_DIMENSION_NAMES),
        "candidate_count_invariant_satisfied": (
            len(rows) - counts["OMITTED_SCIENTIFICALLY_UNSUPPORTED"]
            + counts["OMITTED_SCIENTIFICALLY_UNSUPPORTED"]
            == len(rows)
            and len(rows) <= 28
        ),
        "datasets": dataset_summaries,
        "conditions": sorted(
            rows,
            key=lambda row: (
                list(DATASET_POLICIES).index(str(row["dataset_id"])),
                CONDITIONS.index(str(row["condition"])),
            ),
        ),
        "combustor_target_selection_review": copy.deepcopy(combustor_target_review),
        "combustor_recommended_provisional_target": copy.deepcopy(
            recommended_combustor_target
        ),
        "combustor_redesign_target_candidates": combustor_candidates,
        "combustor_redesign_target_candidate_source": combustor_candidate_source,
        "family_template_leakage_audits": family_leakage,
        "added_only_to_reach_28": False,
        "o2_o3_failed_only_due_to_different_g_of_o_count": 0,
        "family_issue_cross_condition_block_count": 0,
        "canonical_case_mutation_count": 0,
        "human_grounding_curator_confirmation_count": 0,
        "official_scq_executed_count": 0,
        "formal_srac_adjudication_executed_count": 0,
        "evaluation_contract_curator_executed_count": 0,
        "judge_calibration_executed_count": 0,
        "formal_tested_model_run_count": 0,
        "historical_ground_truth_used_as_scientific_evidence": False,
        "human_scientific_selection_status": "PENDING",
    }
    _write_json(out / "scientific_question_manifest.json", manifest)
    (out / "scientific_question_review.md").write_text(
        _render_review(manifest), encoding="utf-8"
    )
    return manifest


__all__ = [
    "CONDITIONS",
    "CONDITION_SCIENTIFIC_REVIEW_INSTRUCTION",
    "DATASET_POLICIES",
    "MATRIX_SLOT_STATUSES",
    "QUESTION_CANDIDATE_STATUSES",
    "SCIENTIFIC_RELEASE_STATUSES",
    "ARTIFACT_REUSE_STATUSES",
    "SUPPORT_DIMENSION_NAMES",
    "CombustorTargetReviewer",
    "COMBUSTOR_TARGET_COMPARISON_INSTRUCTION",
    "FinalReviewer",
    "SCIENTIFIC_QUESTION_CANDIDATE_MATRIX_VERSION",
    "build_condition_scientific_reviewer_visible_payload",
    "build_condition_scientific_review_packets",
    "gt_srac_impact_for_slot",
    "run_flow_expert_condition_scientific_review",
    "run_flow_expert_combustor_target_review",
    "run_scientific_question_candidate_matrix",
    "validate_combustor_target_selection",
    "validate_condition_scientific_review",
    "validate_family_condition_semantics",
]
