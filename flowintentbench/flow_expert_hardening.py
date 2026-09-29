"""End-to-end robustness harness for the canonical Flow Expert interface."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scientific_expert_review import (
    ATOMIC_OBSERVATION_FIELDS,
    FLOW_SCIENTIFIC_REVIEWER_PROFILE,
    validate_scientific_expert_output,
    validate_scientific_review_packet,
)
from .scientific_expert_validation import (
    ablate_scientific_evidence,
    audit_semantic_mutation_reviews,
    build_blinded_human_calibration_packet,
    build_semantic_mutation_suite,
    compare_atomic_review_repeatability,
    compare_evidence_ablation_behavior,
    compare_external_human_labels,
    write_immutable_scientific_expert_run,
)
from .scientific_run_snapshots import canonical_json_sha256


_REQUIRED_PORTFOLIO_IDENTITIES = frozenset(
    {
        ("Kitchen", "kitchen_flow_regions", "high_speed_region"),
        (
            "Kitchen",
            "kitchen_turbulence_activity",
            "kitchen_turbulence_activity",
        ),
        (
            "Kitchen",
            "kitchen_concentration_heterogeneity",
            "kitchen_concentration_heterogeneity",
        ),
        (
            "Combustor",
            "combustor_density_features",
            "combustor_density_features",
        ),
    }
)


ReviewInvoker = Callable[[Mapping[str, Any], str], Mapping[str, Any]]

_CONTROLLED_ABLATION_TARGETS = {
    "kitchen_flow_regions": {
        "evidence_id": "ext_ev_001",
        "reason": (
            "This is the sole visible dataset-specific source that directly documents "
            "the Kitchen velocity values and named arrays; generic VTK sources remain visible."
        ),
    }
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _family_id(packet: Mapping[str, Any]) -> str:
    identity = packet.get("family_identity")
    return str(identity.get("family_id", "")) if isinstance(identity, Mapping) else ""


def _concept_id(packet: Mapping[str, Any]) -> str:
    identity = packet.get("family_identity")
    return str(identity.get("concept_id", "")) if isinstance(identity, Mapping) else ""


def _condition_id(packet: Mapping[str, Any]) -> str:
    conditions = packet.get("conditions")
    if isinstance(conditions, Sequence) and conditions and isinstance(conditions[0], Mapping):
        return str(conditions[0].get("condition", ""))
    return ""


def _controlled_ablation_target(packet: Mapping[str, Any]) -> tuple[str, str]:
    family_id = _family_id(packet)
    target = _CONTROLLED_ABLATION_TARGETS.get(family_id)
    if target is None:
        raise ValueError(f"no authored ablation target for family: {family_id}")
    evidence_id = str(target["evidence_id"])
    evidence_ids = {
        str(item.get("evidence_id", ""))
        for item in packet.get("evidence_records", ())
        if isinstance(item, Mapping)
    }
    if evidence_id not in evidence_ids:
        raise ValueError(
            f"authored ablation evidence is absent from {family_id}: {evidence_id}"
        )
    return evidence_id, str(target["reason"])


def _artifact_review(
    artifact: Mapping[str, Any], packet: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    parsed = artifact.get("parsed_result")
    parsed_review = dict(parsed) if isinstance(parsed, Mapping) and parsed else None
    if parsed_review is None:
        return None, {
            "status": "NOT_EVALUATED",
            "condition_reviews": [],
            "question_reviews": [],
            "reason_codes": ["INVOCATION_NOT_SUCCESSFUL"],
        }
    return parsed_review, validate_scientific_expert_output(parsed_review, packet)


def _identity_from_artifact(artifact: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    call = artifact.get("call")
    call = call if isinstance(call, Mapping) else {}
    return (
        {
            "provider": str(call.get("provider", "UNKNOWN")),
            "model_id": str(call.get("model_id", "UNKNOWN")),
            "model_family": str(call.get("model_family", "UNKNOWN")),
        },
        {
            "agent_profile_id": str(
                call.get("agent_profile_id", FLOW_SCIENTIFIC_REVIEWER_PROFILE)
            ),
            "agent_profile_sha256": str(call.get("agent_profile_sha256", "UNKNOWN")),
            "runtime_profile_id": str(call.get("runtime_profile_id", "flow-tool-free-v1")),
            "runtime_profile_sha256": str(call.get("runtime_profile_sha256", "UNKNOWN")),
            "system_instruction_sha256": str(
                call.get("system_instruction_sha256", "UNKNOWN")
            ),
        },
    )


def _execute_review_task(
    task: Mapping[str, Any],
    *,
    output_root: Path,
    reviewer: ReviewInvoker,
) -> dict[str, Any]:
    packet = task["packet"]
    item_id = str(task["item_id"])
    try:
        artifact = dict(reviewer(packet, item_id))
    except Exception as exc:
        artifact = {
            "status": "FAILED",
            "invocation_status": "INFRASTRUCTURE_INVALID",
            "parsed_result": {},
            "validated_review": {
                "status": "NOT_EVALUATED",
                "condition_reviews": [],
                "question_reviews": [],
                "reason_codes": ["REVIEWER_EXCEPTION"],
            },
            "call": {"error": f"{type(exc).__name__}: {exc}"},
            "live_model_calls": False,
        }
    parsed_review, validated = _artifact_review(artifact, packet)
    call = artifact.get("call")
    call = call if isinstance(call, Mapping) else {}
    model_identity, profile_identity = _identity_from_artifact(artifact)
    immutable = write_immutable_scientific_expert_run(
        output_root,
        review_packet=packet,
        review=(
            parsed_review
            if str(artifact.get("invocation_status", "")).upper() == "SUCCESS"
            else None
        ),
        invocation_status=str(artifact.get("invocation_status", "INFRASTRUCTURE_INVALID")),
        model_identity=model_identity,
        profile_identity=profile_identity,
        invocation_metadata={
            "validation_item_id": item_id,
            "task_index": task["task_index"],
            "task_kind": task["task_kind"],
            "family_id": _family_id(packet),
            "concept_id": _concept_id(packet),
            "condition": _condition_id(packet),
            "review_validation_status": validated.get("status"),
            "invocation_error": str(call.get("error", "")),
        },
    )
    run_directory = Path(str(immutable.pop("run_directory")))
    result = {
        "task_index": task["task_index"],
        "item_id": item_id,
        "task_kind": task["task_kind"],
        "family_id": _family_id(packet),
        "concept_id": _concept_id(packet),
        "condition": _condition_id(packet),
        "packet_sha256": canonical_json_sha256(packet),
        "artifact_status": artifact.get("status"),
        "invocation_status": artifact.get("invocation_status"),
        "review_validation_status": validated.get("status"),
        "submitted_to_model": artifact.get("live_model_calls") is True,
        "invocation_error": str(call.get("error", "")),
        "validated_review": validated,
        "parsed_review": parsed_review,
        "immutable_run": {
            **immutable,
            "run_directory": str(run_directory.relative_to(output_root)),
        },
        "lifecycle_gate_advanced": False,
    }
    return result


def execute_flow_expert_validation_task(
    task: Mapping[str, Any],
    *,
    output_root: str | Path,
    reviewer: ReviewInvoker,
) -> dict[str, Any]:
    """Execute one reusable validation task and persist its immutable attempt."""

    return _execute_review_task(
        task,
        output_root=Path(output_root).resolve(),
        reviewer=reviewer,
    )


def _atomic_row(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {
            **{field: None for field in ATOMIC_OBSERVATION_FIELDS},
            "model_review_status": "NOT_AVAILABLE",
            "model_uncertainty_explicit": False,
            "eligibility_recommendation": None,
            "scientific_ambiguities": [],
        }
    if result.get("review_validation_status") != "PASS":
        return {
            **{field: None for field in ATOMIC_OBSERVATION_FIELDS},
            "model_review_status": "INCOMPLETE",
            "model_uncertainty_explicit": False,
            "eligibility_recommendation": None,
            "scientific_ambiguities": [],
        }
    validated = result.get("validated_review")
    rows = validated.get("condition_reviews") if isinstance(validated, Mapping) else None
    if isinstance(rows, Sequence):
        for row in rows:
            if isinstance(row, Mapping):
                recommendation = row.get("eligibility_recommendation")
                ambiguities = list(row.get("scientific_ambiguities", ()) or ())
                return {
                    **{field: row.get(field) for field in ATOMIC_OBSERVATION_FIELDS},
                    "model_review_status": "PASS",
                    "model_uncertainty_explicit": (
                        recommendation == "MORE_EVIDENCE_REQUIRED" or bool(ambiguities)
                    ),
                    "eligibility_recommendation": recommendation,
                    "scientific_ambiguities": ambiguities,
                }
    return {
        **{field: None for field in ATOMIC_OBSERVATION_FIELDS},
        "model_review_status": "INCOMPLETE",
        "model_uncertainty_explicit": False,
        "eligibility_recommendation": None,
        "scientific_ambiguities": [],
    }


def _repeatability_result(
    task_results: Mapping[str, Mapping[str, Any]], item_ids: Sequence[str]
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for item_id in item_ids:
        result = task_results.get(item_id)
        if result is None:
            continue
        immutable = result["immutable_run"]
        runs.append(
            {
                "run_metadata": immutable,
                "review": result.get("parsed_review"),
                "review_validation_status": result.get("review_validation_status"),
            }
        )
    return compare_atomic_review_repeatability(runs, required_successful_runs=3)


def _markdown_report(readiness: Mapping[str, Any], task_summary: Mapping[str, Any]) -> str:
    lines = [
        "# Flow Expert robustness and human-calibration preparation",
        "",
        "This validation harness does not perform curator review, SCQ, SRAC, evaluator calibration, or benchmark execution.",
        "",
        "| State | Value |",
        "|---|---|",
    ]
    for key in (
        "FLOW_EXPERT_ENGINEERING_READY",
        "FLOW_EXPERT_LIVE_EXECUTION_STATUS",
        "FLOW_EXPERT_CROSS_FAMILY_VALIDATION",
        "FLOW_EXPERT_ROBUSTNESS_STATUS",
        "FLOW_EXPERT_REPEATABILITY_STATUS",
        "FLOW_EXPERT_HUMAN_CALIBRATION_STATUS",
        "FLOW_EXPERT_SCIENTIFICALLY_VALIDATED",
    ):
        lines.append(f"| `{key}` | `{readiness.get(key)}` |")
    lines.extend(
        [
            "",
            f"- Planned callable tasks: `{task_summary.get('planned_callable_task_count')}`",
            f"- Completed successful calls: `{task_summary.get('successful_call_count')}`",
            f"- Preflight-rejected mutations: `{task_summary.get('preflight_rejected_mutation_count')}`",
            f"- Infrastructure-invalid calls: `{task_summary.get('infrastructure_invalid_call_count')}`",
            f"- Lifecycle promotion performed: `{str(readiness.get('LIFECYCLE_PROMOTION_PERFORMED')).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_flow_expert_validation_hardening(
    packets: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    *,
    reviewer: ReviewInvoker | None = None,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Run the frozen WS4/WS5 validation plan without lifecycle promotion."""

    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be between 1 and 32")
    packet_rows = [dict(packet) for packet in packets]
    packet_identities = [
        (
            str(packet.get("dataset_identity", {}).get("dataset_id", "")),
            str(packet.get("family_identity", {}).get("family_id", "")),
            str(packet.get("family_identity", {}).get("concept_id", "")),
        )
        for packet in packet_rows
    ]
    if (
        len(packet_identities) != len(set(packet_identities))
        or set(packet_identities) != _REQUIRED_PORTFOLIO_IDENTITIES
    ):
        raise ValueError(
            "hardening validation requires the four unique frozen portfolio identities"
        )
    packet_validations = [validate_scientific_review_packet(packet) for packet in packet_rows]
    if any(item.get("status") != "PASS" for item in packet_validations):
        raise ValueError("hardening validation received an invalid canonical packet")
    out = Path(output_root).resolve()
    for packet in packet_rows:
        _write_json(out / "00_packets" / f"{_family_id(packet)}.json", packet)

    control = next(
        (packet for packet in packet_rows if _family_id(packet) == "kitchen_flow_regions"),
        None,
    )
    if control is None:
        raise ValueError("hardening validation requires the kitchen_flow_regions control")
    same_dataset_sibling = next(
        (
            packet
            for packet in packet_rows
            if _family_id(packet) == "kitchen_turbulence_activity"
        ),
        None,
    )
    if same_dataset_sibling is None:
        raise ValueError(
            "hardening validation requires a same-dataset sibling evidence control"
        )
    mutation_suite = build_semantic_mutation_suite(
        control, wrong_family_packet=same_dataset_sibling
    )
    by_mutation_type = {
        str(item["mutation_type"]): item for item in mutation_suite["mutations"]
    }
    evidence_id, ablation_reason = _controlled_ablation_target(control)
    ablation = ablate_scientific_evidence(
        control,
        evidence_id=evidence_id,
        critical_reason=ablation_reason,
    )
    _write_json(out / "01_mutations" / "semantic_mutation_suite.json", mutation_suite)
    _write_json(out / "02_evidence_ablation" / "ablation.json", ablation)

    tasks: list[dict[str, Any]] = []

    def add_task(item_id: str, task_kind: str, packet: Mapping[str, Any]) -> None:
        tasks.append(
            {
                "task_index": len(tasks) + 1,
                "item_id": item_id,
                "task_kind": task_kind,
                "packet": dict(packet),
            }
        )

    for packet in packet_rows:
        add_task(f"original::{_family_id(packet)}", "CROSS_FAMILY_ORIGINAL", packet)
    rejected_mutations: list[dict[str, Any]] = []
    for mutation in mutation_suite["mutations"]:
        preflight = mutation["preflight_validation"]
        if preflight.get("status") != "PASS":
            rejected_mutations.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "mutation_type": mutation["mutation_type"],
                    "status": "INPUT_REJECTED",
                    "invocation_status": "NOT_RUN",
                    "preflight_validation": preflight,
                    "submitted_to_model": False,
                }
            )
            continue
        add_task(
            f"mutation::{mutation['mutation_id']}",
            "SEMANTIC_MUTATION",
            mutation["packet"],
        )
    add_task(f"ablation::{ablation['ablation_id']}", "EVIDENCE_ABLATION", ablation["packet"])

    repeat_groups = {
        "high_speed": control,
        "turbulence_or_scalar": packet_rows[1],
        "evidence_ablation": ablation["packet"],
    }
    repeat_item_ids: dict[str, list[str]] = {}
    for group, packet in repeat_groups.items():
        ids: list[str] = []
        for trial in range(1, 4):
            item_id = f"repeat::{group}::trial-{trial}"
            add_task(item_id, "REPEATABILITY", packet)
            ids.append(item_id)
        repeat_item_ids[group] = ids

    _write_json(
        out / "03_execution" / "task_plan.json",
        {
            "max_workers": max_workers,
            "callable_tasks": [
                {key: value for key, value in task.items() if key != "packet"}
                for task in tasks
            ],
            "preflight_rejected_mutations": rejected_mutations,
            "lifecycle_gate_advanced": False,
        },
    )

    ordered_results: list[dict[str, Any]] = []
    if reviewer is not None:
        results_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _execute_review_task,
                    task,
                    output_root=out,
                    reviewer=reviewer,
                ): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                results_by_index[int(task["task_index"])] = future.result()
        ordered_results = [results_by_index[index] for index in sorted(results_by_index)]
        for result in ordered_results:
            _write_json(
                out / "03_execution" / "attempts" / f"{int(result['task_index']):03d}.json",
                result,
            )
    task_results = {str(item["item_id"]): item for item in ordered_results}

    mutation_attempts: list[dict[str, Any]] = []
    for mutation in mutation_suite["mutations"]:
        result = task_results.get(f"mutation::{mutation['mutation_id']}")
        if result is not None:
            mutation_attempts.append(
                {
                    "mutation_id": mutation["mutation_id"],
                    "invocation_status": result.get("invocation_status"),
                    "review": result.get("parsed_review"),
                    "review_validation_status": result.get("review_validation_status"),
                    "submitted_to_model": result.get("submitted_to_model"),
                }
            )
    mutation_audit = audit_semantic_mutation_reviews(
        mutation_suite, mutation_attempts
    )
    _write_json(out / "01_mutations" / "mutation_audit.json", mutation_audit)

    original_control = task_results.get(f"original::{_family_id(control)}")
    ablation_result = task_results.get(f"ablation::{ablation['ablation_id']}")
    if (
        original_control is not None
        and ablation_result is not None
        and isinstance(original_control.get("parsed_review"), Mapping)
        and isinstance(ablation_result.get("parsed_review"), Mapping)
        and original_control.get("review_validation_status") == "PASS"
        and ablation_result.get("review_validation_status") == "PASS"
    ):
        ablation_behavior = compare_evidence_ablation_behavior(
            original_control["parsed_review"],
            ablation_result["parsed_review"],
            condition=_condition_id(control),
        )
    else:
        ablation_behavior = {
            "status": (
                "NOT_RUN"
                if reviewer is None
                else "INFRASTRUCTURE_INVALID"
                if any(
                    isinstance(item, Mapping)
                    and item.get("invocation_status") != "SUCCESS"
                    for item in (original_control, ablation_result)
                )
                else "REVIEW_INCOMPLETE"
            ),
            "false_confidence_case": None,
            "scientific_truth_assigned": False,
        }
    _write_json(
        out / "02_evidence_ablation" / "behavior_comparison.json",
        ablation_behavior,
    )

    repeatability: dict[str, Any] = {}
    for group, item_ids in repeat_item_ids.items():
        repeatability[group] = _repeatability_result(task_results, item_ids)
    _write_json(out / "04_repeatability" / "atomic_stability.json", repeatability)

    calibration_items: list[dict[str, Any]] = [
        {
            "selection_stratum": "SUPPORTED_ORIGINAL",
            "condition": _condition_id(packet_rows[0]),
            "review_packet": packet_rows[0],
            "source_task_id": f"original::{_family_id(packet_rows[0])}",
        },
        *[
            {
                "selection_stratum": "SCIENTIFIC_CONCEPT_DIVERSITY",
                "condition": _condition_id(packet),
                "review_packet": packet,
                "source_task_id": f"original::{_family_id(packet)}",
            }
            for packet in packet_rows[1:4]
        ],
        {
            "selection_stratum": "LEGITIMATE_O2_OPENNESS",
            "condition": by_mutation_type["LEGITIMATE_O2_OPENNESS"]["condition"],
            "review_packet": by_mutation_type["LEGITIMATE_O2_OPENNESS"]["packet"],
            "source_task_id": f"mutation::{by_mutation_type['LEGITIMATE_O2_OPENNESS']['mutation_id']}",
        },
        {
            "selection_stratum": "TARGET_DRIFT",
            "condition": by_mutation_type["TARGET_DRIFT"]["condition"],
            "review_packet": by_mutation_type["TARGET_DRIFT"]["packet"],
            "source_task_id": f"mutation::{by_mutation_type['TARGET_DRIFT']['mutation_id']}",
        },
        {
            "selection_stratum": "EVIDENCE_INSUFFICIENCY",
            "condition": _condition_id(ablation["packet"]),
            "review_packet": ablation["packet"],
            "source_task_id": f"ablation::{ablation['ablation_id']}",
        },
        {
            "selection_stratum": "UNSUPPORTED_OBSERVABLE",
            "condition": by_mutation_type["UNSUPPORTED_OBSERVABLE"]["condition"],
            "review_packet": by_mutation_type["UNSUPPORTED_OBSERVABLE"]["packet"],
            "source_task_id": f"mutation::{by_mutation_type['UNSUPPORTED_OBSERVABLE']['mutation_id']}",
        },
    ]
    unstable_groups = [
        group
        for group, result in repeatability.items()
        if result.get("status") == "FLOW_EXPERT_UNSTABLE"
    ]
    for group in unstable_groups[:4]:
        packet = repeat_groups[group]
        calibration_items.append(
            {
                "selection_stratum": "MODEL_RUN_DISAGREEMENT",
                "condition": _condition_id(packet),
                "review_packet": packet,
                "source_task_id": repeat_item_ids[group][0],
                "model_disagreement_available": True,
                "selected_for_model_disagreement": True,
            }
        )
    human_packet = build_blinded_human_calibration_packet(
        calibration_items,
        calibration_packet_id=f"flow-expert-calibration-{canonical_json_sha256([canonical_json_sha256(item['review_packet']) for item in calibration_items])[:12]}",
    )
    model_reviews: dict[str, dict[str, Any]] = {}
    coordinator_rows: list[dict[str, Any]] = []
    for human_item, calibration_item in zip(
        human_packet["items"], calibration_items, strict=True
    ):
        source_task_id = str(calibration_item["source_task_id"])
        source_result = task_results.get(source_task_id)
        item_id = str(human_item["calibration_item_id"])
        model_reviews[item_id] = _atomic_row(source_result)
        coordinator_rows.append(
            {
                "calibration_item_id": item_id,
                "selection_stratum": calibration_item["selection_stratum"],
                "source_task_id": source_task_id,
                "packet_sha256": canonical_json_sha256(calibration_item["review_packet"]),
                "model_review_sha256": canonical_json_sha256(model_reviews[item_id]),
                "not_in_human_packet": True,
            }
        )
    pending_comparison = compare_external_human_labels(
        human_packet, model_reviews, None
    )
    _write_json(out / "05_human_calibration" / "human_packet.json", human_packet)
    _write_json(
        out / "05_human_calibration" / "coordinator_manifest.json",
        {
            "calibration_packet_id": human_packet["calibration_packet_id"],
            "items": coordinator_rows,
            "human_labels_supplied": False,
        },
    )
    _write_json(
        out / "05_human_calibration" / "model_reviews.json",
        {"model_reviews": model_reviews},
    )
    _write_json(
        out / "05_human_calibration" / "comparison.json", pending_comparison
    )

    original_results = [
        task_results.get(f"original::{_family_id(packet)}") for packet in packet_rows
    ]
    infrastructure_invalid_results = [
        item
        for item in ordered_results
        if item.get("invocation_status") not in {"SUCCESS", "NOT_RUN"}
    ]
    incomplete_output_results = [
        item
        for item in ordered_results
        if item.get("invocation_status") == "SUCCESS"
        and item.get("review_validation_status") != "PASS"
    ]
    cross_family_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if any(
            not isinstance(item, Mapping) or item.get("invocation_status") != "SUCCESS"
            for item in original_results
        )
        else "INCOMPLETE"
        if any(item.get("review_validation_status") != "PASS" for item in original_results)
        else "COMPLETE"
    )
    robustness_task_ids = {
        str(task["item_id"])
        for task in tasks
        if task["task_kind"] in {"SEMANTIC_MUTATION", "EVIDENCE_ABLATION"}
    } | {f"original::{_family_id(control)}"}
    robustness_results = [
        task_results.get(item_id) for item_id in sorted(robustness_task_ids)
    ]
    robustness_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if any(
            not isinstance(item, Mapping) or item.get("invocation_status") != "SUCCESS"
            for item in robustness_results
        )
        else "INCOMPLETE"
        if any(item.get("review_validation_status") != "PASS" for item in robustness_results)
        else "COMPLETE"
        if mutation_audit["status"] == "PASS"
        else "SENSITIVITY_GAPS_OBSERVED"
    )
    repeat_statuses = {str(item.get("status")) for item in repeatability.values()}
    repeatability_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if any(value == "INSUFFICIENT_SUCCESSFUL_RUNS" for value in repeat_statuses)
        and infrastructure_invalid_results
        else "INCOMPLETE"
        if any(value == "INSUFFICIENT_SUCCESSFUL_RUNS" for value in repeat_statuses)
        else "FLOW_EXPERT_UNSTABLE"
        if "FLOW_EXPERT_UNSTABLE" in repeat_statuses
        else "STABLE"
        if repeat_statuses == {"STABLE"}
        else "INCOMPLETE"
    )
    live_execution_status = (
        "NOT_RUN"
        if reviewer is None
        else "INFRASTRUCTURE_INVALID"
        if infrastructure_invalid_results
        else "OUTPUT_INCOMPLETE"
        if incomplete_output_results
        else "COMPLETE"
    )
    readiness = {
        "FLOW_EXPERT_ENGINEERING_READY": True,
        "FLOW_EXPERT_LIVE_EXECUTION_STATUS": live_execution_status,
        "FLOW_EXPERT_CROSS_FAMILY_VALIDATION": cross_family_status,
        "FLOW_EXPERT_ROBUSTNESS_STATUS": robustness_status,
        "FLOW_EXPERT_REPEATABILITY_STATUS": repeatability_status,
        "FLOW_EXPERT_HUMAN_CALIBRATION_STATUS": "PENDING",
        "FLOW_EXPERT_SCIENTIFICALLY_VALIDATED": False,
        "LIFECYCLE_PROMOTION_PERFORMED": False,
        "GROUNDING_CURATOR_CONFIRMATION_CREATED": False,
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "SRAC_EXECUTED_COUNT": 0,
        "EVALUATOR_CALIBRATION_EXECUTED": False,
        "FORMAL_MODEL_RUN_COUNT": 0,
    }
    task_summary = {
        "planned_callable_task_count": len(tasks),
        "completed_task_count": len(ordered_results),
        "successful_call_count": sum(
            item.get("invocation_status") == "SUCCESS" for item in ordered_results
        ),
        "infrastructure_invalid_call_count": sum(
            item.get("invocation_status") not in {"SUCCESS", "NOT_RUN"}
            for item in ordered_results
        ),
        "validated_review_count": sum(
            item.get("review_validation_status") == "PASS" for item in ordered_results
        ),
        "incomplete_output_count": len(incomplete_output_results),
        "preflight_rejected_mutation_count": len(rejected_mutations),
        "results_in_fixed_task_order": True,
        "max_workers": max_workers,
    }
    final = {
        "schema_version": "flow-expert-validation-hardening-v1",
        "execution_mode": "LIVE_OR_INJECTED_REVIEW" if reviewer is not None else "BUILD_ONLY",
        "readiness": readiness,
        "task_summary": task_summary,
        "mutation_audit": mutation_audit,
        "evidence_ablation_behavior": ablation_behavior,
        "repeatability": repeatability,
        "human_calibration": {
            "calibration_item_count": human_packet["item_count"],
            "human_labels_supplied": False,
            "comparison_status": pending_comparison["status"],
        },
    }
    _write_json(out / "readiness.json", final)
    (out / "report.md").write_text(
        _markdown_report(readiness, task_summary), encoding="utf-8"
    )
    return final


__all__ = [
    "ReviewInvoker",
    "execute_flow_expert_validation_task",
    "run_flow_expert_validation_hardening",
]
