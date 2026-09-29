"""Unified scientific-sensitivity hardening for the canonical Flow Expert.

This is a validation harness, not a lifecycle stage.  It composes the existing
canonical reviewer, controlled packet mutations, evidence-coverage audits,
repeatability comparison, and blinded human-calibration tooling.  It never
assigns scientific truth or advances curator, SCQ, SRAC, evaluator, or formal
benchmark state.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent_profile import load_agent_profile
from .flow_expert_evidence_coverage import (
    audit_evidence_ablation,
    audit_evidence_ablation_response,
    audit_evidence_coverage,
)
from .flow_expert_hardening import (
    ReviewInvoker,
    execute_flow_expert_validation_task,
)
from .flow_expert_portfolio_validation import (
    build_flow_expert_responsibility_matrix_packets,
)
from .flow_expert_responsibility import (
    audit_responsibility_mutation_reviews,
    build_responsibility_mutation_suite,
    validate_responsibility_mutation_delivery,
)
from .live_agents import build_live_role_system_instruction
from .scientific_expert_review import (
    ADVISORY_RECOMMENDATIONS,
    ATOMIC_OBSERVATION_FIELDS,
    FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    QUESTION_RECOMMENDATIONS,
    SCIENTIFIC_REVIEW_API_VERSION,
    SCIENTIFIC_REVIEW_CONTRACT_VERSION,
    SCIENTIFIC_REVIEW_INSTRUCTION,
    SCIENTIFIC_REVIEW_PACKET_VERSION,
    SCIENTIFIC_REVIEW_RUNTIME_CONTRACT,
    validate_scientific_review_packet,
)
from .scientific_expert_validation import (
    ablate_scientific_evidence,
    audit_semantic_mutation_reviews,
    build_blinded_human_calibration_packet,
    build_semantic_mutation_suite,
    compare_atomic_review_repeatability,
)
from .scientific_run_snapshots import canonical_json_sha256


SENSITIVITY_HARDENING_VERSION = "flow-expert-scientific-sensitivity-v1"
_RETAINED_SEMANTIC_MUTATIONS = frozenset(
    {
        "TARGET_DRIFT",
        "SCIENTIFIC_SCOPE_DRIFT",
        "WRONG_FAMILY_EVIDENCE",
        "FABRICATED_PROVENANCE_ID",
        "UNSUPPORTED_FIXED_O",
        "LEGITIMATE_O2_OPENNESS",
        "O2_UNRESOLVED_DIMENSION_ACCIDENTALLY_FIXED",
        "UNSUPPORTED_OBSERVABLE",
    }
)
_INFRASTRUCTURE_STATUSES = frozenset(
    {"INFRASTRUCTURE_INVALID", "PROVIDER_ERROR", "PROVIDER_TIMEOUT", "TOOL_ERROR"}
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _family_id(packet: Mapping[str, Any]) -> str:
    family = packet.get("family_identity")
    return str(family.get("family_id", "")) if isinstance(family, Mapping) else ""


def _concept_id(packet: Mapping[str, Any]) -> str:
    family = packet.get("family_identity")
    return str(family.get("concept_id", "")) if isinstance(family, Mapping) else ""


def _condition_rows(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = packet.get("conditions")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _condition_id(row: Mapping[str, Any]) -> str:
    return str(row.get("condition", ""))


def _single_condition_packet(
    packet: Mapping[str, Any], condition: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(packet))
    matches = [row for row in _condition_rows(result) if _condition_id(row) == condition]
    if len(matches) != 1:
        raise ValueError(f"packet requires exactly one {condition} condition")
    result["conditions"] = [copy.deepcopy(dict(matches[0]))]
    validation = validate_scientific_review_packet(result)
    if validation.get("status") != "PASS":
        raise ValueError(
            f"single-condition packet is invalid for {condition}: "
            + ", ".join(validation.get("reason_codes", ()))
        )
    return result


def _packet_by_family(
    packets: Sequence[Mapping[str, Any]], family_id: str
) -> Mapping[str, Any]:
    matches = [packet for packet in packets if _family_id(packet) == family_id]
    if len(matches) != 1:
        raise ValueError(f"responsibility matrix requires one packet for {family_id}")
    return matches[0]


def _review_row(
    result: Mapping[str, Any] | None, condition: str
) -> Mapping[str, Any] | None:
    if not isinstance(result, Mapping) or result.get("review_validation_status") != "PASS":
        return None
    review = result.get("parsed_review")
    if not isinstance(review, Mapping):
        return None
    rows = review.get("condition_reviews")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    return next(
        (
            row
            for row in rows
            if isinstance(row, Mapping) and str(row.get("condition", "")) == condition
        ),
        None,
    )


def _atomic_projection(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "review_status": "NOT_AVAILABLE",
            **{field: None for field in ATOMIC_OBSERVATION_FIELDS},
            "eligibility_recommendation": None,
            "scientific_ambiguities": [],
            "evidence_ids": [],
            "source_ids": [],
        }
    return {
        "review_status": "PASS",
        **{field: row.get(field) for field in ATOMIC_OBSERVATION_FIELDS},
        "eligibility_recommendation": row.get("eligibility_recommendation"),
        "scientific_ambiguities": list(row.get("scientific_ambiguities", ()) or ()),
        "evidence_ids": list(row.get("evidence_ids", ()) or ()),
        "source_ids": list(row.get("source_ids", ()) or ()),
    }


def _model_review_projection(
    result: Mapping[str, Any] | None, condition: str
) -> dict[str, Any]:
    row = _review_row(result, condition)
    if row is None:
        return {
            "model_review_status": "NOT_AVAILABLE",
            **{field: None for field in ATOMIC_OBSERVATION_FIELDS},
            "model_uncertainty_explicit": False,
            "eligibility_recommendation": None,
            "scientific_ambiguities": [],
        }
    ambiguities = list(row.get("scientific_ambiguities", ()) or ())
    recommendation = row.get("eligibility_recommendation")
    return {
        "model_review_status": "PASS",
        **{field: row.get(field) for field in ATOMIC_OBSERVATION_FIELDS},
        "model_uncertainty_explicit": (
            recommendation == "MORE_EVIDENCE_REQUIRED" or bool(ambiguities)
        ),
        "eligibility_recommendation": recommendation,
        "scientific_ambiguities": ambiguities,
    }


def _task_state(results: Sequence[Mapping[str, Any]], *, planned: int) -> str:
    if not results:
        return "NOT_RUN"
    if len(results) != planned or any(
        str(row.get("invocation_status", "")) in _INFRASTRUCTURE_STATUSES
        for row in results
    ):
        return "INFRASTRUCTURE_INVALID"
    if any(row.get("invocation_status") != "SUCCESS" for row in results):
        return "OUTPUT_INCOMPLETE"
    if any(row.get("review_validation_status") != "PASS" for row in results):
        return "OUTPUT_INCOMPLETE"
    return "COMPLETE"


def _freeze_manifest(
    repository_root: Path,
    packets: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profile = load_agent_profile(
        FLOW_SCIENTIFIC_REVIEWER_PROFILE, repository_root=repository_root
    )
    system = build_live_role_system_instruction(
        "scientific_reviewer", SCIENTIFIC_REVIEW_INSTRUCTION
    )
    system_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()
    contract = {
        "contract_version": SCIENTIFIC_REVIEW_CONTRACT_VERSION,
        "packet_version": SCIENTIFIC_REVIEW_PACKET_VERSION,
        "atomic_observation_fields": list(ATOMIC_OBSERVATION_FIELDS),
        "advisory_recommendations": sorted(ADVISORY_RECOMMENDATIONS),
        "question_recommendations": sorted(QUESTION_RECOMMENDATIONS),
        "runtime_contract": SCIENTIFIC_REVIEW_RUNTIME_CONTRACT,
        "system_instruction_sha256": system_hash,
    }
    api_path = Path(__file__).with_name("scientific_expert_review.py")
    api_hash = hashlib.sha256(api_path.read_bytes()).hexdigest()
    packet_hashes = {
        _family_id(packet): canonical_json_sha256(packet) for packet in packets
    }
    task_packet_hashes = {
        str(task["item_id"]): canonical_json_sha256(task["packet"]) for task in tasks
    }
    observed_system_hashes = sorted(
        {
            str(result.get("immutable_run", {}).get("profile_identity", {}).get(
                "system_instruction_sha256", ""
            ))
            for result in results
            if result.get("submitted_to_model") is True
        }
        - {"", "UNKNOWN"}
    )
    manifest = {
        "reviewer_profile": profile.agent_id,
        "model_identifier": profile.model_id,
        "model_family": profile.model_family,
        "agent_profile_sha256": profile.profile_sha256,
        "runtime_profile": profile.runtime_profile_id,
        "runtime_profile_sha256": profile.runtime_profile.profile_sha256,
        "system_instruction_sha256": system_hash,
        "review_contract_version": SCIENTIFIC_REVIEW_CONTRACT_VERSION,
        "review_contract_sha256": canonical_json_sha256(contract),
        "canonical_api_version": SCIENTIFIC_REVIEW_API_VERSION,
        "canonical_api_sha256": api_hash,
        "portfolio_evidence_packet_hashes": packet_hashes,
        "portfolio_evidence_packet_set_sha256": canonical_json_sha256(packet_hashes),
        "validation_task_packet_hashes": task_packet_hashes,
        "validation_task_packet_set_sha256": canonical_json_sha256(task_packet_hashes),
        "observed_live_system_instruction_hashes": observed_system_hashes,
        "observed_live_system_instruction_matches": (
            observed_system_hashes == [system_hash] if observed_system_hashes else None
        ),
    }
    manifest["frozen_reviewer_version_sha256"] = canonical_json_sha256(manifest)
    return manifest


def _repeatability(
    task_results: Mapping[str, Mapping[str, Any]], item_ids: Sequence[str]
) -> dict[str, Any]:
    runs = []
    for item_id in item_ids:
        result = task_results.get(item_id)
        if result is None:
            continue
        runs.append(
            {
                "run_metadata": result.get("immutable_run", {}),
                "review": result.get("parsed_review"),
                "review_validation_status": result.get("review_validation_status"),
            }
        )
    return compare_atomic_review_repeatability(runs, required_successful_runs=3)


def _report(
    readiness: Mapping[str, Any],
    cross: Mapping[str, Any],
    responsibility: Mapping[str, Any],
    evidence: Mapping[str, Any],
    repeatability: Mapping[str, Any],
) -> str:
    lines = [
        "# Flow Expert scientific sensitivity hardening",
        "",
        "This validation is advisory and performs no curator, SCQ, SRAC, evaluator, or formal benchmark transition.",
        "",
        "## A. Responsibility sensitivity",
        "",
        "| Mutation | Diff | Result | Finding contract |",
        "|---|---|---|---|",
    ]
    for row in responsibility.get("review_audit", {}).get("rows", ()):
        lines.append(
            f"| `{row.get('mutation_type')}` | `{row.get('packet_diff_verification')}` | "
            f"`{row.get('status')}` | `{row.get('finding_contract_supported')}` |"
        )
    lines.extend(
        [
            "",
            f"- Detected: `{responsibility.get('review_audit', {}).get('detected_count')}` / `{responsibility.get('mutation_count')}`",
            f"- Packet-diff verified: `{responsibility.get('review_audit', {}).get('packet_diff_verified_count')}`",
            f"- Remaining ambiguity: `{responsibility.get('review_audit', {}).get('remaining_ambiguity')}`",
            "",
            "## B. Evidence sensitivity",
            "",
            "| Ablation | Removed | Provenance | Coverage after | Model response | False confidence |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in evidence.get("ablations", ()):
        coverage_after = row.get("coverage_audit", {}).get("coverage_after", {})
        lines.append(
            f"| `{row.get('classification')}` | `{', '.join(row.get('removed_evidence_ids', ()))}` | "
            f"`{row.get('coverage_audit', {}).get('provenance_still_closed')}` | "
            f"`{coverage_after.get('evidence_coverage_status')}` | "
            f"`{row.get('response_audit', {}).get('status')}` | "
            f"`{row.get('response_audit', {}).get('false_confidence')}` |"
        )
    lines.extend(
        [
            "",
            f"- False-confidence count: `{evidence.get('false_confidence_count')}`",
            "",
            "## C. Cross-family coverage",
            "",
            "| Family | Case | Condition | Coverage | Recommendation | Target | Fixed O | Open O | Finding | Materialization | Q target | Q scope |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in cross.get("cases", ()):
        review = row.get("scientific_observation", {})
        lines.append(
            f"| `{row.get('family_id')}` | `{row.get('case_id')}` | `{row.get('condition')}` | "
            f"`{row.get('evidence_coverage_status')}` | `{review.get('eligibility_recommendation')}` | "
            f"`{review.get('scientific_target_supported')}` | `{review.get('fixed_o_supported')}` | "
            f"`{review.get('unresolved_o_is_legitimate_scientific_choice')}` | "
            f"`{review.get('finding_contract_supported')}` | "
            f"`{review.get('materialization_scientifically_meaningful')}` | "
            f"`{review.get('question_target_preserved')}` | "
            f"`{review.get('question_scope_preserved')}` |"
        )
    lines.extend(
        [
            "",
            f"- O-family coverage: `{cross.get('o_family_coverage')}`",
            f"- F-family coverage: `{cross.get('f_family_coverage')}`",
            f"- Scientific outcomes: `{cross.get('scientific_outcome_counts')}`",
            "",
            "## D. Robustness",
            "",
            f"- Generic mutation audit: `{readiness.get('generic_mutation_audit', {}).get('status')}`; detected `{readiness.get('generic_mutation_audit', {}).get('detected_count')}`.",
            f"- Responsibility sensitivity: `{readiness.get('readiness', {}).get('FLOW_EXPERT_RESPONSIBILITY_SENSITIVITY')}`.",
            f"- Evidence sensitivity: `{readiness.get('readiness', {}).get('FLOW_EXPERT_EVIDENCE_SENSITIVITY')}`.",
            f"- Unresolved defects: `{readiness.get('unresolved_sensitivity_defects')}`.",
            "",
            "| Generic mutation | Result | Detected |",
            "|---|---|---|",
        ]
    )
    for row in readiness.get("generic_mutation_audit", {}).get("rows", ()):
        lines.append(
            f"| `{row.get('mutation_type')}` | `{row.get('status')}` | "
            f"`{row.get('detected')}` |"
        )
    lines.extend(
        [
            "",
            "## E. Repeatability",
            "",
            "| Repeated case | Status | Atomic agreement | Citation agreement | Material disagreements |",
            "|---|---|---|---|---|",
        ]
    )
    for name, row in repeatability.get("groups", {}).items():
        lines.append(
            f"| `{name}` | `{row.get('status')}` | `{row.get('atomic_field_agreement')}` | "
            f"`{row.get('citation_agreement')}` | `{row.get('disagreement_fields')}` |"
        )
    frozen = readiness.get("frozen_reviewer", {})
    lines.extend(
        [
            "",
            "No majority vote is used.",
            "",
            "## F. Human calibration",
            "",
            f"- Frozen reviewer: `{frozen.get('reviewer_profile')}` / `{frozen.get('model_identifier')}`.",
            f"- Frozen version hash: `{frozen.get('frozen_reviewer_version_sha256')}`.",
            f"- Review contract hash: `{frozen.get('review_contract_sha256')}`.",
            f"- Canonical API hash: `{frozen.get('canonical_api_sha256')}`.",
            f"- Calibration item count: `{readiness.get('human_calibration', {}).get('item_count')}`.",
            "- Human labels supplied: `false`.",
            "- Calibration status: `PENDING`.",
            "",
            "## G. Final readiness",
            "",
            "| State | Value |",
            "|---|---|",
        ]
    )
    for key in (
        "FLOW_EXPERT_ENGINEERING_READY",
        "FLOW_EXPERT_RESPONSIBILITY_SENSITIVITY",
        "FLOW_EXPERT_EVIDENCE_SENSITIVITY",
        "FLOW_EXPERT_CROSS_FAMILY_VALIDATION",
        "FLOW_EXPERT_ROBUSTNESS_STATUS",
        "FLOW_EXPERT_REPEATABILITY_STATUS",
        "FLOW_EXPERT_HUMAN_CALIBRATION_STATUS",
        "FLOW_EXPERT_SCIENTIFICALLY_VALIDATED",
    ):
        lines.append(f"| `{key}` | `{readiness.get('readiness', {}).get(key)}` |")
    return "\n".join(lines) + "\n"


def run_flow_expert_sensitivity_hardening(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    reviewer: ReviewInvoker | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run the frozen sensitivity plan and prepare blinded human calibration."""

    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    root = Path(repository_root).resolve()
    out = Path(output_root).resolve()
    packets = list(build_flow_expert_responsibility_matrix_packets(root))
    if any(validate_scientific_review_packet(packet).get("status") != "PASS" for packet in packets):
        raise ValueError("responsibility matrix contains an invalid canonical packet")

    high_speed = _packet_by_family(packets, "kitchen_flow_regions")
    turbulence = _packet_by_family(packets, "kitchen_turbulence_activity")
    concentration = _packet_by_family(
        packets, "kitchen_concentration_heterogeneity"
    )
    combustor = _packet_by_family(packets, "combustor_density_features")
    high_speed_o2 = _single_condition_packet(high_speed, "O2-F1")
    turbulence_o2 = _single_condition_packet(turbulence, "O2-F1")
    combustor_o1 = _single_condition_packet(combustor, "O1-F1")

    responsibility_suite = build_responsibility_mutation_suite(high_speed)
    for mutation in responsibility_suite["mutations"]:
        delivery = validate_responsibility_mutation_delivery(
            mutation, mutation["packet"]
        )
        if delivery["status"] != "PASS":
            raise ValueError(
                f"responsibility mutation delivery failed: {mutation['mutation_id']}"
            )

    raw_semantic_suite = build_semantic_mutation_suite(
        high_speed_o2, wrong_family_packet=turbulence_o2
    )
    semantic_mutations = [
        row
        for row in raw_semantic_suite["mutations"]
        if row["mutation_type"] in _RETAINED_SEMANTIC_MUTATIONS
    ]
    semantic_suite = {
        **raw_semantic_suite,
        "mutation_count": len(semantic_mutations),
        "mutations": semantic_mutations,
        "superseded_duplicate_mutations": [
            "F1_F2_RESPONSIBILITY_CONFUSION",
            "MISSING_CRITICAL_EVIDENCE",
        ],
    }
    semantic_by_type = {
        str(row["mutation_type"]): row for row in semantic_mutations
    }

    ablation_specs = (
        (
            "critical_direct",
            "ext_ev_001",
            "Remove dataset-specific direct support used by target, observable, and Finding claims.",
        ),
        (
            "contextual_noncritical",
            "ext_ev_002",
            "Remove contextual support while stronger direct target support remains.",
        ),
        (
            "partial_reduction",
            "ext_ev_004",
            "Remove background support while direct target and Operationalization support remain.",
        ),
    )
    ablations: dict[str, dict[str, Any]] = {}
    for label, evidence_id, reason in ablation_specs:
        artifact = ablate_scientific_evidence(
            high_speed_o2, evidence_id=evidence_id, critical_reason=reason
        )
        artifact["coverage_audit"] = audit_evidence_ablation(
            high_speed_o2, artifact["packet"]
        )
        artifact["baseline_task_id"] = "evidence::baseline"
        artifact["condition"] = "O2-F1"
        ablations[label] = artifact
    combustor_ablation = ablate_scientific_evidence(
        combustor_o1,
        evidence_id="op_002",
        critical_reason=(
            "Remove the linked direct dataset-context evidence in a non-Kitchen "
            "representative family."
        ),
    )
    combustor_ablation["coverage_audit"] = audit_evidence_ablation(
        combustor_o1, combustor_ablation["packet"]
    )
    combustor_ablation["baseline_task_id"] = (
        "cross-family::combustor_density_features"
    )
    combustor_ablation["condition"] = "O1-F1"
    ablations["non_kitchen_critical_direct"] = combustor_ablation

    tasks: list[dict[str, Any]] = []

    def add_task(item_id: str, task_kind: str, packet: Mapping[str, Any]) -> None:
        tasks.append(
            {
                "task_index": len(tasks) + 1,
                "item_id": item_id,
                "task_kind": task_kind,
                "packet": copy.deepcopy(dict(packet)),
            }
        )

    for packet in packets:
        add_task(
            f"cross-family::{_family_id(packet)}", "CROSS_FAMILY_REVIEW", packet
        )
    add_task("evidence::baseline", "EVIDENCE_BASELINE", high_speed_o2)
    for mutation in responsibility_suite["mutations"]:
        add_task(
            f"responsibility::{mutation['mutation_id']}",
            "RESPONSIBILITY_MUTATION",
            mutation["packet"],
        )
    for label, artifact in ablations.items():
        add_task(f"evidence::{label}", "EVIDENCE_ABLATION", artifact["packet"])
    preflight_rejected_semantic: list[dict[str, Any]] = []
    for mutation in semantic_mutations:
        if mutation["preflight_validation"].get("status") != "PASS":
            preflight_rejected_semantic.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "mutation_type": mutation["mutation_type"],
                    "status": "EXPECTED_INPUT_REJECTION",
                    "submitted_to_model": False,
                }
            )
            continue
        add_task(
            f"semantic::{mutation['mutation_id']}",
            "SEMANTIC_MUTATION",
            mutation["packet"],
        )

    repeat_packets = {
        "normal_supported_high_speed": high_speed_o2,
        "legitimate_o2_open": semantic_by_type["LEGITIMATE_O2_OPENNESS"]["packet"],
        "f1_f2_responsibility": responsibility_suite["mutations"][0]["packet"],
        "critical_evidence_ablation": ablations["critical_direct"]["packet"],
        "non_kitchen_combustor": combustor_o1,
    }
    repeat_ids: dict[str, list[str]] = {}
    for group, packet in repeat_packets.items():
        repeat_ids[group] = []
        for trial in range(1, 4):
            item_id = f"repeat::{group}::trial-{trial}"
            add_task(item_id, "REPEATABILITY", packet)
            repeat_ids[group].append(item_id)

    ordered_results: list[dict[str, Any]] = []
    if reviewer is not None:
        by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    execute_flow_expert_validation_task,
                    task,
                    output_root=out,
                    reviewer=reviewer,
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                by_index[int(task["task_index"])] = future.result()
        ordered_results = [by_index[index] for index in sorted(by_index)]
    result_by_id = {str(row["item_id"]): row for row in ordered_results}

    cross_task_results = [
        result_by_id[item_id]
        for item_id in (
            f"cross-family::{_family_id(packet)}" for packet in packets
        )
        if item_id in result_by_id
    ]
    coverage_by_family = {
        _family_id(packet): audit_evidence_coverage(packet) for packet in packets
    }
    cross_cases: list[dict[str, Any]] = []
    o_families = {key: set() for key in ("O1", "O2", "O3")}
    f_families = {key: set() for key in ("F1", "F2")}
    for packet in packets:
        family_id = _family_id(packet)
        task_result = result_by_id.get(f"cross-family::{family_id}")
        coverage = coverage_by_family[family_id]
        for condition in _condition_rows(packet):
            condition_id = _condition_id(condition)
            o_name, f_name = condition_id.split("-", 1)
            o_families[o_name].add(family_id)
            f_families[f_name].add(family_id)
            cross_cases.append(
                {
                    "dataset_id": packet["dataset_identity"]["dataset_id"],
                    "family_id": family_id,
                    "concept_id": _concept_id(packet),
                    "case_id": condition.get("case_id"),
                    "condition": condition_id,
                    "provenance_status": coverage["provenance_status"],
                    "evidence_coverage_status": coverage[
                        "evidence_coverage_status"
                    ],
                    "missing_support_dimension_ids": coverage[
                        "missing_support_dimension_ids"
                    ],
                    "scientific_observation": _atomic_projection(
                        _review_row(task_result, condition_id)
                    ),
                }
            )
    cross_state = _task_state(cross_task_results, planned=len(packets))
    recommendation_counts = Counter(
        row["scientific_observation"].get("eligibility_recommendation")
        for row in cross_cases
        if row["scientific_observation"].get("eligibility_recommendation")
    )
    cross_artifact = {
        "schema_version": SENSITIVITY_HARDENING_VERSION,
        "status": cross_state,
        "family_count": len(packets),
        "case_count": len(cross_cases),
        "o_family_coverage": {
            key: sorted(value) for key, value in o_families.items()
        },
        "f_family_coverage": {
            key: sorted(value) for key, value in f_families.items()
        },
        "coverage_requirement_met": all(
            len(value) >= 2 for value in [*o_families.values(), *f_families.values()]
        ),
        "scientific_outcome_counts": dict(sorted(recommendation_counts.items())),
        "family_evidence_coverage": coverage_by_family,
        "cases": cross_cases,
        "same_canonical_reviewer_contract": True,
        "lifecycle_gate_advanced": False,
    }

    responsibility_attempts = []
    for mutation in responsibility_suite["mutations"]:
        result = result_by_id.get(f"responsibility::{mutation['mutation_id']}")
        if result is None:
            continue
        responsibility_attempts.append(
            {
                "mutation_id": mutation["mutation_id"],
                "invocation_status": result.get("invocation_status"),
                "submitted_packet_sha256": result.get("packet_sha256"),
                "review": result.get("parsed_review"),
            }
        )
    responsibility_audit = audit_responsibility_mutation_reviews(
        responsibility_suite, responsibility_attempts
    )
    responsibility_artifact = {
        "schema_version": responsibility_suite["suite_version"],
        "mutation_count": responsibility_suite["mutation_count"],
        "packet_diff_verification": [
            {
                "mutation_id": row["mutation_id"],
                "mutation_type": row["mutation_type"],
                "source_packet_sha256": row["source_packet_sha256"],
                "mutated_packet_sha256": row["mutated_packet_sha256"],
                "original_finding_contract": row["original_finding_contract"],
                "mutated_finding_contract": row["mutated_finding_contract"],
                "packet_diff_verification": row["packet_diff_verification"],
                "mutation_reaches_validated_canonical_packet": row[
                    "mutation_reaches_validated_canonical_packet"
                ],
            }
            for row in responsibility_suite["mutations"]
        ],
        "review_audit": responsibility_audit,
        "lifecycle_gate_advanced": False,
    }

    baseline_result = result_by_id.get("evidence::baseline")
    evidence_rows: list[dict[str, Any]] = []
    for label, artifact in ablations.items():
        ablated_result = result_by_id.get(f"evidence::{label}")
        item_baseline_result = result_by_id.get(str(artifact["baseline_task_id"]))
        condition = str(artifact["condition"])
        if (
            item_baseline_result is not None
            and ablated_result is not None
            and item_baseline_result.get("review_validation_status") == "PASS"
            and ablated_result.get("review_validation_status") == "PASS"
        ):
            response_audit = audit_evidence_ablation_response(
                artifact["coverage_audit"],
                item_baseline_result["parsed_review"],
                ablated_result["parsed_review"],
                condition=condition,
            )
        else:
            response_audit = {
                "status": "NOT_RUN" if reviewer is None else "NOT_EVALUATED",
                "condition": condition,
                "false_confidence": False,
            }
        evidence_rows.append(
            {
                "ablation_label": label,
                "classification": artifact["coverage_audit"]["classification"],
                "removed_evidence_ids": artifact["coverage_audit"][
                    "removed_evidence_ids"
                ],
                "affected_support_dimension_ids": artifact["coverage_audit"][
                    "affected_support_dimension_ids"
                ],
                "coverage_lost_dimension_ids": artifact["coverage_audit"][
                    "coverage_lost_dimension_ids"
                ],
                "coverage_audit": artifact["coverage_audit"],
                "model_response_before": _atomic_projection(
                    _review_row(item_baseline_result, condition)
                ),
                "model_response_after": _atomic_projection(
                    _review_row(ablated_result, condition)
                ),
                "response_audit": response_audit,
            }
        )
    evidence_task_ids = sorted(
        {
            "evidence::baseline",
            "cross-family::combustor_density_features",
            *(f"evidence::{label}" for label in ablations),
        }
    )
    evidence_task_results = [
        result_by_id[item_id] for item_id in evidence_task_ids if item_id in result_by_id
    ]
    evidence_execution_state = _task_state(
        evidence_task_results, planned=len(evidence_task_ids)
    )
    false_confidence_count = sum(
        row["response_audit"].get("false_confidence") is True
        for row in evidence_rows
    )
    evidence_status = (
        evidence_execution_state
        if evidence_execution_state != "COMPLETE"
        else "FALSE_CONFIDENCE_OBSERVED"
        if false_confidence_count
        else "COMPLETE"
    )
    evidence_artifact = {
        "schema_version": SENSITIVITY_HARDENING_VERSION,
        "status": evidence_status,
        "baseline": {
            "provenance_and_coverage": audit_evidence_coverage(high_speed_o2),
            "model_response": _atomic_projection(
                _review_row(baseline_result, "O2-F1")
            ),
        },
        "representative_non_kitchen_baseline": {
            "family_id": "combustor_density_features",
            "provenance_and_coverage": audit_evidence_coverage(combustor_o1),
            "model_response": _atomic_projection(
                _review_row(
                    result_by_id.get("cross-family::combustor_density_features"),
                    "O1-F1",
                )
            ),
        },
        "ablations": evidence_rows,
        "false_confidence_count": false_confidence_count,
        "provenance_closure_is_not_evidence_coverage": True,
        "contextual_noncritical_semantics": (
            "COVERAGE_PRESERVING_REMOVAL; ALL CURRENT AUTHORED CLAIMS REMAIN "
            "CRITICAL CLAIMS"
        ),
        "scientific_truth_assigned": False,
    }

    semantic_attempts = []
    for mutation in semantic_mutations:
        result = result_by_id.get(f"semantic::{mutation['mutation_id']}")
        if result is None:
            continue
        semantic_attempts.append(
            {
                "mutation_id": mutation["mutation_id"],
                "invocation_status": result.get("invocation_status"),
                "review": result.get("parsed_review"),
                "submitted_to_model": result.get("submitted_to_model"),
            }
        )
    semantic_audit = audit_semantic_mutation_reviews(
        semantic_suite, semantic_attempts
    )

    repeat_groups = {
        group: _repeatability(result_by_id, item_ids)
        for group, item_ids in repeat_ids.items()
    }
    repeat_results = [
        result_by_id[item_id]
        for item_ids in repeat_ids.values()
        for item_id in item_ids
        if item_id in result_by_id
    ]
    repeat_execution_state = _task_state(
        repeat_results, planned=sum(len(value) for value in repeat_ids.values())
    )
    repeat_statuses = {row.get("status") for row in repeat_groups.values()}
    repeat_status = (
        repeat_execution_state
        if repeat_execution_state != "COMPLETE"
        else "FLOW_EXPERT_UNSTABLE"
        if "FLOW_EXPERT_UNSTABLE" in repeat_statuses
        else "STABLE"
        if repeat_statuses == {"STABLE"}
        else "INCOMPLETE"
    )
    repeatability_artifact = {
        "schema_version": SENSITIVITY_HARDENING_VERSION,
        "status": repeat_status,
        "groups": repeat_groups,
        "majority_vote_used": False,
    }

    freeze_manifest = _freeze_manifest(root, packets, tasks, ordered_results)
    calibration_specs = [
        ("SUPPORTED_ORIGINAL", high_speed, "O1-F1", "cross-family::kitchen_flow_regions"),
        ("LEGITIMATE_O2_OPENNESS", high_speed_o2, "O2-F1", f"semantic::{semantic_by_type['LEGITIMATE_O2_OPENNESS']['mutation_id']}"),
        ("O3_RESPONSIBILITY", concentration, "O3-F1", "cross-family::kitchen_concentration_heterogeneity"),
        ("MULTI_CONCEPT_TURBULENCE", turbulence, "O2-F1", "cross-family::kitchen_turbulence_activity"),
        ("MULTI_DATASET_DENSITY", combustor, "O1-F1", "cross-family::combustor_density_features"),
        ("TARGET_DRIFT", semantic_by_type["TARGET_DRIFT"]["packet"], "O2-F1", f"semantic::{semantic_by_type['TARGET_DRIFT']['mutation_id']}"),
        ("F1_PRESENTED_OPEN", responsibility_suite["mutations"][0]["packet"], responsibility_suite["mutations"][0]["condition"], f"responsibility::{responsibility_suite['mutations'][0]['mutation_id']}"),
        ("F2_ADEQUATE_CORE_REMOVED", responsibility_suite["mutations"][-1]["packet"], responsibility_suite["mutations"][-1]["condition"], f"responsibility::{responsibility_suite['mutations'][-1]['mutation_id']}"),
        ("EVIDENCE_INSUFFICIENCY", ablations["critical_direct"]["packet"], "O2-F1", "evidence::critical_direct"),
        ("CONTEXTUAL_EVIDENCE_EDGE", ablations["contextual_noncritical"]["packet"], "O2-F1", "evidence::contextual_noncritical"),
    ]
    calibration_items = [
        {
            "selection_stratum": stratum,
            "condition": condition,
            "review_packet": packet,
            "source_task_id": source_task_id,
        }
        for stratum, packet, condition, source_task_id in calibration_specs
    ]
    calibration_id = (
        "flow-expert-human-calibration-"
        + canonical_json_sha256(
            {
                "frozen_reviewer": freeze_manifest[
                    "frozen_reviewer_version_sha256"
                ],
                "items": [
                    [canonical_json_sha256(item["review_packet"]), item["condition"]]
                    for item in calibration_items
                ],
            }
        )[:12]
    )
    human_packet = build_blinded_human_calibration_packet(
        calibration_items, calibration_packet_id=calibration_id
    )
    human_packet["frozen_reviewer_version_sha256"] = freeze_manifest[
        "frozen_reviewer_version_sha256"
    ]
    model_reviews: dict[str, dict[str, Any]] = {}
    calibration_coordinator: list[dict[str, Any]] = []
    for human_item, calibration_item in zip(
        human_packet["items"], calibration_items, strict=True
    ):
        item_id = str(human_item["calibration_item_id"])
        source_task_id = str(calibration_item["source_task_id"])
        model_reviews[item_id] = _model_review_projection(
            result_by_id.get(source_task_id), str(calibration_item["condition"])
        )
        calibration_coordinator.append(
            {
                "calibration_item_id": item_id,
                "selection_stratum": calibration_item["selection_stratum"],
                "source_task_id": source_task_id,
                "packet_sha256": canonical_json_sha256(
                    calibration_item["review_packet"]
                ),
                "model_review_sha256": canonical_json_sha256(model_reviews[item_id]),
                "not_visible_in_human_packet": True,
            }
        )

    full_execution_state = _task_state(ordered_results, planned=len(tasks))
    responsibility_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if responsibility_audit["infrastructure_invalid_count"]
        else "INCOMPLETE"
        if responsibility_audit["review_incomplete_count"]
        or responsibility_audit["packet_delivery_unverified_count"]
        else "COMPLETE"
        if responsibility_audit["status"] == "PASS"
        else "SENSITIVITY_GAPS_OBSERVED"
    )
    generic_status = semantic_audit["status"]
    unresolved_defects = []
    if responsibility_status == "SENSITIVITY_GAPS_OBSERVED":
        unresolved_defects.append("F1_F2_RESPONSIBILITY_SENSITIVITY")
    if evidence_status == "FALSE_CONFIDENCE_OBSERVED":
        unresolved_defects.append("CRITICAL_EVIDENCE_FALSE_CONFIDENCE")
    if generic_status == "PARTIAL":
        unresolved_defects.append("GENERIC_MUTATION_SENSITIVITY")
    robustness_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if full_execution_state == "INFRASTRUCTURE_INVALID"
        else "INCOMPLETE"
        if full_execution_state == "OUTPUT_INCOMPLETE"
        else "SENSITIVITY_GAPS_OBSERVED"
        if unresolved_defects
        else "REPEATABILITY_GAPS_OBSERVED"
        if repeat_status == "FLOW_EXPERT_UNSTABLE"
        else "COMPLETE"
    )
    responsibility_artifact["status"] = responsibility_status
    readiness_states = {
        "FLOW_EXPERT_ENGINEERING_READY": True,
        "FLOW_EXPERT_LIVE_EXECUTION_STATUS": full_execution_state,
        "FLOW_EXPERT_RESPONSIBILITY_SENSITIVITY": responsibility_status,
        "FLOW_EXPERT_EVIDENCE_SENSITIVITY": evidence_status,
        "FLOW_EXPERT_CROSS_FAMILY_VALIDATION": cross_state,
        "FLOW_EXPERT_ROBUSTNESS_STATUS": robustness_status,
        "FLOW_EXPERT_REPEATABILITY_STATUS": repeat_status,
        "FLOW_EXPERT_HUMAN_CALIBRATION_STATUS": "PENDING",
        "FLOW_EXPERT_SCIENTIFICALLY_VALIDATED": False,
        "LIFECYCLE_PROMOTION_PERFORMED": False,
        "GROUNDING_CURATOR_CONFIRMATION_CREATED": False,
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "SRAC_EXECUTED_COUNT": 0,
        "EVALUATOR_CALIBRATION_EXECUTED": False,
        "FORMAL_MODEL_RUN_COUNT": 0,
    }
    readiness = {
        "schema_version": SENSITIVITY_HARDENING_VERSION,
        "execution_mode": (
            "BUILD_ONLY"
            if reviewer is None
            else "LIVE_MODEL_CALL"
            if any(row.get("submitted_to_model") is True for row in ordered_results)
            else "INJECTED_VALIDATION_REVIEWER"
        ),
        "readiness": readiness_states,
        "task_summary": {
            "planned_callable_task_count": len(tasks),
            "completed_task_count": len(ordered_results),
            "successful_call_count": sum(
                row.get("invocation_status") == "SUCCESS" for row in ordered_results
            ),
            "validated_review_count": sum(
                row.get("review_validation_status") == "PASS"
                for row in ordered_results
            ),
            "infrastructure_invalid_count": sum(
                str(row.get("invocation_status", "")) in _INFRASTRUCTURE_STATUSES
                for row in ordered_results
            ),
            "output_incomplete_count": sum(
                row.get("invocation_status") == "SUCCESS"
                and row.get("review_validation_status") != "PASS"
                for row in ordered_results
            ),
            "preflight_rejected_semantic_mutation_count": len(
                preflight_rejected_semantic
            ),
            "max_workers": max_workers,
        },
        "generic_mutation_audit": semantic_audit,
        "preflight_rejected_semantic_mutations": preflight_rejected_semantic,
        "unresolved_sensitivity_defects": unresolved_defects,
        "unresolved_robustness_findings": (
            ["REPEATABILITY_INSTABILITY"]
            if repeat_status == "FLOW_EXPERT_UNSTABLE"
            else []
        ),
        "frozen_reviewer": freeze_manifest,
        "human_calibration": {
            "packet_id": human_packet["calibration_packet_id"],
            "item_count": human_packet["item_count"],
            "human_labels_supplied": False,
            "status": "PENDING",
            "coordinator": calibration_coordinator,
        },
        "model_reviews": model_reviews,
    }

    _write_json(out / "readiness.json", readiness)
    _write_json(out / "cross_family_review.json", cross_artifact)
    _write_json(out / "responsibility_sensitivity.json", responsibility_artifact)
    _write_json(out / "evidence_sensitivity.json", evidence_artifact)
    _write_json(out / "repeatability.json", repeatability_artifact)
    _write_json(out / "human_calibration_packet.json", human_packet)
    (out / "report.md").write_text(
        _report(
            readiness,
            cross_artifact,
            responsibility_artifact,
            evidence_artifact,
            repeatability_artifact,
        ),
        encoding="utf-8",
    )
    return readiness


__all__ = [
    "SENSITIVITY_HARDENING_VERSION",
    "run_flow_expert_sensitivity_hardening",
]
