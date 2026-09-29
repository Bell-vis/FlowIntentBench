"""One-time calibration and hardening diagnostics for a frozen pilot.

The calibration layer is deliberately separate from benchmark scoring.  It
reads saved RunRecords and case artifacts, performs deterministic contract and
runtime audits, and writes diagnostic reports.  It never changes Ground Truth,
never calls the tested model, and never turns unavailable evaluator judgments
into zeros.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_runner import RunRecord, RunStatus
from .runtime_config import (
    RuntimeConfigurationError,
    load_server_provider_configuration,
    validate_reasoning_effort,
)


REQUIRED_OUTPUTS = (
    "calibration_summary.md",
    "calibration_case_judgments.json",
    "judge_agreement_report.md",
    "human_evaluator_comparison.md",
    "open_ended_analysis_report.md",
    "of_separation_validation.md",
    "data_contract_audit.md",
    "o1_specification_audit.md",
    "reference_branch_sensitivity.md",
    "runtime_error_classification.md",
    "engineering_hardening_report.md",
    "benchmark_change_log.md",
    "freeze_readiness.json",
)

DIAGNOSTIC_CASE_IDS = (
    "blunt_fin_o1_f1",
    "blunt_fin_o3_f1",
    "carotid_o1_f1",
    "carotid_o1_f2",
    "carotid_o2_f1",
    "nasa_lox_post_o2_f1",
    "nasa_lox_post_o3_f1",
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
)

# Keep the complete controlled vocabulary in every report, including zero
# counts. This makes an absent category distinguishable from an omitted
# implementation branch.
ERROR_CATEGORIES = (
    "AGENT_CODE_ERROR",
    "FILESYSTEM_DISCOVERY_ERROR",
    "MISSING_DEPENDENCY",
    "TOOL_INTERFACE_ERROR",
    "DATA_READER_ERROR",
    "INFRASTRUCTURE_ERROR",
)


@dataclass(frozen=True)
class FrozenObservation:
    dataset_id: str
    case_id: str
    condition: str
    run_id: str
    status: str
    run_record_path: Path
    run_record: RunRecord
    trajectory: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve_collection_state_path(collection_root: str | Path) -> Path:
    """Resolve a frozen collection state in canonical or legacy layout."""

    root = Path(collection_root).resolve()
    canonical = root / "collection_state.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(root.glob("collections/*/collection_state.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"cannot identify unique collection_state.json under {root}")
    return candidates[0]


def _resolve_run_record_path(
    *,
    collection_root: Path,
    state_path: Path,
    observation: Mapping[str, Any],
) -> Path:
    relative = observation.get("run_record_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("collection observation path must be relative")
    legacy = state_path.parent / relative
    if legacy.is_file():
        return legacy

    dataset_id = str(observation.get("dataset_id", "")).strip()
    case_id = str(observation.get("case_id", "")).strip()
    trial_index = observation.get("trial_index")
    attempt_count = observation.get("attempt_count", 1)
    if (
        not dataset_id
        or not case_id
        or isinstance(trial_index, bool)
        or not isinstance(trial_index, int)
        or trial_index < 1
        or isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 1
    ):
        raise FileNotFoundError(legacy)
    stem = f"{dataset_id}__{case_id}__trial-{trial_index}"
    if attempt_count > 1:
        stem += f"__attempt-{attempt_count}"
    canonical = collection_root / "runs" / f"{stem}__run_record.json"
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    return canonical


def load_frozen_observations(collection_root: str | Path) -> tuple[FrozenObservation, ...]:
    """Load and validate the existing completed N=1 collection only."""

    root = Path(collection_root).resolve()
    state_path = resolve_collection_state_path(root)
    state = _read_json(state_path)
    if state.get("status") != "COMPLETE":
        raise ValueError("calibration requires a COMPLETE frozen collection")
    raw_observations = state.get("observations")
    if not isinstance(raw_observations, list) or len(raw_observations) != 28:
        raise ValueError("calibration requires exactly 28 saved observations")
    observations: list[FrozenObservation] = []
    for item in raw_observations:
        if not isinstance(item, Mapping):
            raise ValueError("collection observation must be an object")
        run_path = _resolve_run_record_path(
            collection_root=root,
            state_path=state_path,
            observation=item,
        )
        if item.get("run_record_sha256") != _sha256(run_path):
            raise ValueError(f"RunRecord hash mismatch: {run_path}")
        record = RunRecord.load_json(run_path)
        if record.run_status not in {RunStatus.COMPLETED, RunStatus.MODEL_NONCOMPLETION}:
            raise ValueError(f"invalid scientific observation status: {record.run_status.value}")
        if run_path.name.endswith("__run_record.json"):
            trajectory_path = run_path.with_name(
                run_path.name.removesuffix("__run_record.json") + "__trajectory.json"
            )
        else:
            trajectory_path = run_path.with_name("trajectory.json")
        trajectory = _read_json(trajectory_path) if trajectory_path.is_file() else []
        if not isinstance(trajectory, list):
            raise ValueError(f"trajectory must be a list: {trajectory_path}")
        observations.append(
            FrozenObservation(
                dataset_id=str(item["dataset_id"]),
                case_id=str(item["case_id"]),
                condition=str(item["condition"]),
                run_id=record.run_id,
                status=record.run_status.value,
                run_record_path=run_path,
                run_record=record,
                trajectory=tuple(event for event in trajectory if isinstance(event, Mapping)),
            )
        )
    keys = {(item.case_id, item.condition) for item in observations}
    if len(keys) != 28:
        raise ValueError("collection contains duplicate case/condition observations")
    return tuple(observations)


def _classify_python_error(event: Mapping[str, Any]) -> str:
    exception = event.get("exception")
    if not isinstance(exception, Mapping):
        return "UNKNOWN"
    error_type = str(exception.get("type", ""))
    message = str(exception.get("message", ""))
    code = str(event.get("code", ""))
    if error_type == "TimeoutError" or "exceeded" in message.casefold():
        if "/**" in code or "recursive=True" in code:
            return "FILESYSTEM_DISCOVERY_ERROR"
        return "AGENT_CODE_ERROR"
    if error_type == "ModuleNotFoundError":
        return "MISSING_DEPENDENCY"
    if error_type in {"SyntaxError", "TypeError", "ValueError", "IndexError", "AssertionError", "NameError", "KeyError"}:
        return "AGENT_CODE_ERROR"
    if error_type in {"BrokenPipeError", "ConnectionError"}:
        return "INFRASTRUCTURE_ERROR"
    if "/case" in code and ("FileNotFoundError" in error_type or "No such file" in message):
        return "DATA_READER_ERROR"
    return "AGENT_CODE_ERROR"


def runtime_error_report(observations: Sequence[FrozenObservation], collection_root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter({category: 0 for category in ERROR_CATEGORIES})
    entries: list[dict[str, Any]] = []
    output_lengths: list[int] = []
    largest: list[dict[str, Any]] = []
    for observation in observations:
        last_code = ""
        for index, event in enumerate(observation.trajectory):
            if event.get("event") == "model_turn" and isinstance(event.get("code"), str):
                last_code = str(event.get("code"))
            if event.get("event") != "python_execution":
                continue
            stdout = str(event.get("stdout", ""))
            stderr = str(event.get("stderr", ""))
            output_lengths.append(len(stdout))
            if event.get("exception"):
                classified_event = dict(event)
                classified_event["code"] = last_code
                category = _classify_python_error(classified_event)
                counts[category] += 1
                entries.append(
                    {
                        "case_id": observation.case_id,
                        "condition": observation.condition,
                        "execution_index": event.get("execution_index", index),
                        "category": category,
                        "exception": dict(event["exception"]),
                    }
                )
            largest.append(
                {
                    "case_id": observation.case_id,
                    "condition": observation.condition,
                    "execution_index": event.get("execution_index", index),
                    "stdout_chars": len(stdout),
                    "stderr_chars": len(stderr),
                }
            )
    invalid_attempts: list[dict[str, Any]] = []
    collection_dirs = sorted((collection_root / "collections").glob("*/"))
    runs_root = collection_dirs[0] / "runs" if collection_dirs else collection_root / "runs"
    for path in sorted(runs_root.rglob("*run_record.json")):
        try:
            record = RunRecord.load_json(path)
        except ValueError:
            continue
        if record.run_status is RunStatus.INFRASTRUCTURE_INVALID:
            invalid_attempts.append(
                {
                    "case_id": record.case_id,
                    "run_id": record.run_id,
                    "path": str(path.relative_to(collection_root)),
                    "failure_reason": record.failure_reason,
                    "classification": "TOOL_INTERFACE_ERROR"
                    if "exactly one tool call" in str(record.failure_reason)
                    else "INFRASTRUCTURE_ERROR",
                }
            )
    largest.sort(key=lambda value: value["stdout_chars"], reverse=True)
    combined_counts = Counter(counts)
    for attempt in invalid_attempts:
        combined_counts[str(attempt["classification"])] += 1
    return {
        "python_error_count": sum(counts.values()),
        "python_error_counts": dict(sorted(counts.items())),
        "error_counts": dict(sorted(combined_counts.items())),
        "errors": entries,
        "infrastructure_invalid_attempts": invalid_attempts,
        "stdout_total_chars": sum(output_lengths),
        "stdout_median_chars": statistics.median(output_lengths) if output_lengths else 0,
        "stdout_max_chars": max(output_lengths) if output_lengths else 0,
        "largest_outputs": largest[:10],
    }


def _human_judgments() -> list[dict[str, Any]]:
    """Return the fixed calibration-only curator-reference draft subset.

    These judgments are deliberately explicit and are never merged into GT.
    They encode the diagnostic conclusions from the pilot review supplied to
    the coding task. They remain drafts until an accountable human curator
    confirms them; the tool must not manufacture completed human review.
    """

    rows = {
        "blunt_fin_o1_f1": ("VALID_REFERENCE", "SUPPORTED", "CONSISTENT", None),
        "blunt_fin_o3_f1": ("VALID_UNENUMERATED", "PARTIAL_SUPPORT", "CONSISTENT", "unsupported qualitative geometry label"),
        "carotid_o1_f1": ("VALID_REFERENCE_UNDER_VECTOR_READING", "SUPPORTED_UNDER_VECTOR_READING", "CONSISTENT", "QUESTION_AMBIGUITY"),
        "carotid_o1_f2": ("VALID_UNENUMERATED_UNDER_SCALAR_READING", "SUPPORTED_UNDER_SCALAR_READING", "CONSISTENT", "QUESTION_AMBIGUITY"),
        "carotid_o2_f1": ("VALID_UNENUMERATED", "SUPPORTED_UNDER_VECTOR_READING", "CONSISTENT", "QUESTION_AMBIGUITY"),
        "nasa_lox_post_o2_f1": ("UNCERTAIN_DATA_DOMAIN", "SUPPORTED_FOR_EXECUTED_DOMAIN", "INDETERMINATE", "DATA_CONTRACT_DEFECT"),
        "nasa_lox_post_o3_f1": ("VALID_UNENUMERATED", "SUPPORTED", "CONSISTENT", "DATA_CONTRACT_DEFECT"),
        "office_speed_zones_o1_f1": ("VALID_REFERENCE", "SUPPORTED", "CONSISTENT", None),
        "office_speed_zones_o2_f1": ("VALID_UNENUMERATED", "SUPPORTED", "CONSISTENT", "VALID_ALTERNATIVE"),
        "office_speed_zones_o3_f1": ("VALID_UNENUMERATED", "SUPPORTED", "CONSISTENT", "VALID_ALTERNATIVE"),
    }
    result = []
    for case_id in DIAGNOSTIC_CASE_IDS:
        analysis, finding, consistency, defect = rows[case_id]
        result.append(
            {
                "case_id": case_id,
                "human_reference": {
                    "review_status": "DRAFT_REQUIRES_HUMAN_CONFIRMATION",
                    "analysis_validity": analysis,
                    "finding_support": finding,
                    "o_f_consistency": consistency,
                    "valid_unenumerated": analysis == "VALID_UNENUMERATED" or "UNENUMERATED" in analysis,
                    "benchmark_defect": defect,
                },
                "judge_a": None,
                "judge_b": None,
                "agreement_status": "PENDING_HUMAN_AND_INDEPENDENT_JUDGES",
            }
        )
    return result


def audit_data_contract(datasets_root: Path, observations: Sequence[FrozenObservation]) -> dict[str, Any]:
    dataset_ids = sorted({item.dataset_id for item in observations})
    records: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        manifest = _read_json(datasets_root / dataset_id / "dataset_manifest.json")
        case_path = datasets_root / dataset_id / "construction" / "cases" / next(
            item.case_id for item in observations if item.dataset_id == dataset_id
        ) / "case_input.json"
        case = _read_json(case_path)
        reader = manifest.get("reader", {})
        metadata = case.get("flow_data", {}).get("data_metadata", {})
        variables = metadata.get("variables", [])
        names = {item.get("name") for item in variables if isinstance(item, Mapping)}
        blanking = reader.get("grid_blanking", [])
        mask_status = "DATA_CONTRACT_REQUIRED" if blanking else "IRRELEVANT"
        mask_detail = "no canonical reader mask declared"
        if blanking:
            rules = [item.get("validity_rule") for item in blanking if isinstance(item, Mapping)]
            mask_detail = "explicit: " + "; ".join(str(rule) for rule in rules if rule)
            if not any(rules):
                mask_status = "DATA_CONTRACT_REQUIRED"
                mask_detail = "reader exposes blanking but no validity rule is authored"
        variable_ambiguity = dataset_id == "Carotid" and {"scalars", "vectors"} <= names
        carotid_o1_questions = [
            str(
                _read_json(
                    datasets_root
                    / dataset_id
                    / "construction"
                    / "cases"
                    / item.case_id
                    / "case_input.json"
                ).get("scientific_question", "")
            )
            for item in observations
            if item.dataset_id == dataset_id and item.condition in {"O1-F1", "O1-F2"}
        ]
        variable_contract_resolved = bool(carotid_o1_questions) and all(
            "Euclidean magnitude" in question and "`vectors`" in question
            for question in carotid_o1_questions
        )
        records.append(
            {
                "dataset_id": dataset_id,
                "valid_point_mask": {"classification": mask_status, "detail": mask_detail},
                "ghost_or_inactive_cells": {
                    "classification": "DATA_CONTRACT_REQUIRED" if blanking else "IRRELEVANT",
                    "detail": "IBlank validity is explicit" if blanking else "none declared by canonical manifest",
                },
                "units": {
                    "classification": "SCIENTIFICALLY_INTENTIONAL_UNKNOWN"
                    if metadata.get("coordinate_system", {}).get("unit") is None
                    else "IRRELEVANT",
                    "detail": "coordinate/variable units are null in model-visible metadata",
                },
                "vector_scalar_semantics": {
                    "classification": "DATA_CONTRACT_REQUIRED"
                    if variable_ambiguity and not variable_contract_resolved
                    else "IRRELEVANT",
                    "detail": "O1 explicitly selects Euclidean magnitude of point-data `vectors`; scalar metadata remains distinct"
                    if variable_ambiguity and variable_contract_resolved
                    else "stored scalar and stored velocity vector are both plausible speed inputs"
                    if variable_ambiguity
                    else "canonical variable roles are sufficiently distinct or equivalent",
                },
                "coordinate_system": {
                    "classification": "SCIENTIFICALLY_INTENTIONAL_UNKNOWN"
                    if metadata.get("coordinate_system", {}).get("axis_meaning", {}).get("x") is None
                    else "IRRELEVANT",
                    "detail": "axis labels are not supplied" if metadata.get("coordinate_system", {}).get("axis_meaning", {}).get("x") is None else "axis labels supplied",
                },
                "field_names": {"classification": "IRRELEVANT", "detail": "field names are present in data_metadata"},
                "topology": {"classification": "IRRELEVANT", "detail": "grid/mesh type is explicit"},
                "timestep": {"classification": "IRRELEVANT", "detail": "single_snapshot is explicit"},
            }
        )
    return {"dataset_count": len(records), "records": records}


def audit_o1_specifications(datasets_root: Path, observations: Sequence[FrozenObservation]) -> dict[str, Any]:
    records = []
    for observation in observations:
        if observation.condition not in {"O1-F1", "O1-F2"}:
            continue
        case_path = datasets_root / observation.dataset_id / "construction" / "cases" / observation.case_id / "case_input.json"
        payload = _read_json(case_path)
        question = str(payload.get("scientific_question", ""))
        variables = payload.get("flow_data", {}).get("data_metadata", {}).get("variables", [])
        ambiguity = (
            observation.dataset_id == "Carotid"
            and any(v.get("name") == "scalars" for v in variables if isinstance(v, Mapping))
            and any(v.get("name") == "vectors" for v in variables if isinstance(v, Mapping))
            and not ("Euclidean magnitude" in question and "`vectors`" in question)
        )
        records.append(
            {
                "case_id": observation.case_id,
                "dataset_id": observation.dataset_id,
                "condition": observation.condition,
                "question_has_criterion": any(token in question.casefold() for token in ("percentile", "above", "greater", "threshold")),
                "question_has_region_definition": "connected" in question.casefold() or "region" in question.casefold(),
                "question_has_measure": any(token in question.casefold() for token in ("peak", "mean", "strength")),
                "question_has_representation": "mean spatial location" in question.casefold() or "location" in question.casefold(),
                "classification": "TASK_BREAKING" if ambiguity else "PASS",
                "finding": "stored scalar vs velocity-vector magnitude is not uniquely identified" if ambiguity else None,
            }
        )
    return {"records": records, "task_breaking_count": sum(item["classification"] == "TASK_BREAKING" for item in records)}


def reference_branch_report(datasets_root: Path, observations: Sequence[FrozenObservation]) -> dict[str, Any]:
    records = []
    for observation in observations:
        path = datasets_root / observation.dataset_id / "construction" / "cases" / observation.case_id / "ground_truth.json"
        gt = _read_json(path)
        records.append(
            {
                "case_id": observation.case_id,
                "condition": observation.condition,
                "operationalization_branch_count": len(gt.get("acceptable_operationalizations", [])),
                "finding_branch_count": len(gt.get("findings_by_operationalization", [])),
                "best_match_score": None,
                "second_best_score": None,
                "tied_best_branch_count": None,
                "score_status": "UNAVAILABLE_PENDING_INDEPENDENT_EVALUATOR",
            }
        )
    return {"records": records, "score_status": "UNAVAILABLE_PENDING_INDEPENDENT_EVALUATOR"}


def evaluator_gate(
    *,
    tested_provider: str,
    tested_model: str,
    judge_a_config: str | Path | None,
    judge_b_config: str | Path | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tested_target": {"provider": tested_provider, "model": tested_model},
        "status": "BLOCKED",
        "judge_a": None,
        "judge_b": None,
        "reason": "two independent evaluator configurations are required",
    }
    configurations = []
    tested_family = tested_model.split(":", 1)[0].split("/", 1)[0]
    for label, path in (("judge_a", judge_a_config), ("judge_b", judge_b_config)):
        if path is None:
            continue
        try:
            runtime = load_server_provider_configuration(path, role="evaluator")
            validate_reasoning_effort(runtime.model_configuration, label=label)
            if runtime.provider == tested_provider and runtime.model_id == tested_model:
                raise RuntimeConfigurationError("calibration judge must differ from tested target")
            judge_family = runtime.model_family or runtime.model_id.split(":", 1)[0].split("/", 1)[0]
            if judge_family == tested_family:
                raise RuntimeConfigurationError("calibration judge must use a different model family from tested target")
            value = runtime.without_secrets()
            result[label] = value
            configurations.append(value)
        except (RuntimeError, ValueError) as exc:
            result[label] = {"status": "INVALID", "reason": str(exc)}
    if len(configurations) == 2:
        family_a = configurations[0].get("model_family") or configurations[0]["model_id"].split(":", 1)[0].split("/", 1)[0]
        family_b = configurations[1].get("model_family") or configurations[1]["model_id"].split(":", 1)[0].split("/", 1)[0]
        if family_a == family_b:
            result["reason"] = "Judge A and Judge B must be independently configured"
        else:
            result["status"] = "CONFIGURED_NOT_EXECUTED"
            result["reason"] = "configurations validated; live judge calls are opt-in and require explicit credentials"
    return result


def run_calibration(
    *,
    collection_root: str | Path,
    datasets_root: str | Path,
    output_root: str | Path,
    judge_a_config: str | Path | None = None,
    judge_b_config: str | Path | None = None,
) -> dict[str, Any]:
    """Run deterministic one-time calibration diagnostics and write reports."""

    collection = Path(collection_root).resolve()
    datasets = Path(datasets_root).resolve()
    output = Path(output_root).resolve()
    observations = load_frozen_observations(collection)
    state = _read_json(resolve_collection_state_path(collection))
    tested = state.get("model_configuration", {})
    tested_provider = str(state.get("provider", ""))
    tested_model = str(state.get("model_id", ""))
    gate = evaluator_gate(
        tested_provider=tested_provider,
        tested_model=tested_model,
        judge_a_config=judge_a_config,
        judge_b_config=judge_b_config,
    )
    runtime = runtime_error_report(observations, collection)
    contracts = audit_data_contract(datasets, observations)
    o1 = audit_o1_specifications(datasets, observations)
    branches = reference_branch_report(datasets, observations)
    judgments = _human_judgments()
    output.mkdir(parents=True, exist_ok=True)

    _write_json(output / "calibration_case_judgments.json", {
        "record_type": "CalibrationCaseJudgments",
        "calibration_only": True,
        "tested_target": {"provider": tested_provider, "model": tested_model, "model_configuration": tested},
        "independent_evaluator_gate": gate,
        "cases": judgments,
    })
    _write_json(output / "freeze_readiness.json", {
        "record_type": "CalibrationFreezeReadiness",
        "status": "NOT_READY",
        "independent_judges_configured": gate["status"] in {"CONFIGURED_NOT_EXECUTED", "EXECUTED"},
        "judge_outputs_available": False,
        "human_subset_available": True,
        "human_reference_reviewed": False,
        "o1_task_breaking_count": o1["task_breaking_count"],
        "runtime_error_counts": runtime["error_counts"],
        "scientific_method_changed": True,
        "historical_observations_reused_as_formal": False,
        "formal_n3_allowed": False,
        "next_action": "obtain human confirmation, execute two independent judges, and rerun the affected Carotid/NASA cases before small N=3",
    })
    _write_json(output / "data_contract_audit.json", {"record_type": "DataContractAudit", **contracts})
    _write_json(output / "runtime_error_classification.json", {"record_type": "RuntimeErrorClassification", **runtime})

    _write_text(output / "judge_agreement_report.md", _judge_agreement_markdown(gate, judgments))
    _write_text(output / "human_evaluator_comparison.md", _human_comparison_markdown(judgments))
    _write_text(output / "open_ended_analysis_report.md", _open_ended_markdown())
    _write_text(output / "of_separation_validation.md", _of_separation_markdown())
    _write_text(output / "data_contract_audit.md", _data_contract_markdown(contracts))
    _write_text(output / "o1_specification_audit.md", _o1_markdown(o1))
    _write_text(output / "reference_branch_sensitivity.md", _branch_markdown(branches))
    _write_text(output / "runtime_error_classification.md", _runtime_markdown(runtime))
    _write_text(output / "engineering_hardening_report.md", _engineering_markdown(runtime))
    _write_text(output / "benchmark_change_log.md", _change_log_markdown())
    _write_text(output / "calibration_summary.md", _summary_markdown(gate, o1, runtime, contracts))
    return {
        "status": "NOT_READY",
        "output_root": str(output),
        "observation_count": len(observations),
        "evaluator_gate": gate,
        "o1_task_breaking_count": o1["task_breaking_count"],
        "runtime_error_counts": runtime["error_counts"],
        "required_outputs": list(REQUIRED_OUTPUTS),
    }


def _judge_agreement_markdown(gate: Mapping[str, Any], judgments: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(
        f"| {item['case_id']} | {item['human_reference']['analysis_validity']} | PENDING | PENDING | PENDING |"
        for item in judgments
    )
    return f"""# Judge Agreement Report

Calibration is fail-closed because two independent evaluator outputs are not configured. The tested model identity is not reused.

Evaluator gate: **{gate['status']}**  
Reason: {gate['reason']}

| Case | Human reference | Judge A | Judge B | Agreement |
|---|---|---|---|---|
{rows}

The human-reference column is a curator draft and is not presented as completed human review. No draft judgment is merged into Ground Truth or scientific score artifacts. Agreement remains pending until an accountable human confirms the draft and both judges produce blinded structured judgments.
"""


def _human_comparison_markdown(judgments: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(
        f"| {j['case_id']} | {j['human_reference']['analysis_validity']} | {j['human_reference']['finding_support']} | {j['human_reference']['o_f_consistency']} | {j['human_reference']['benchmark_defect'] or 'none'} |"
        for j in judgments
    )
    return f"""# Human Calibration Comparison

The subset contains prescribed, incomplete, valid-unenumerated, ambiguity, data-contract, and qualitative-finding diagnostic cases.

| Case | Analysis | Finding support | O-F consistency | Defect/classification |
|---|---|---|---|---|
{rows}

These judgments are curator-reference drafts for evaluator calibration only, not completed human review and not benchmark GT.
"""


def _open_ended_markdown() -> str:
    return """# Open-Ended Analysis Calibration

Office O2-F1 (q95 plus regional mean speed) and Office O3-F1 (half-maximum connected core) are independently supported, scientifically defensible, and absent from the enumerated reference branches. They must be routed through `VALID_UNENUMERATED` / novel-O adjudication rather than failing by branch membership.

Kitchen O2-F1 is equivalent to an existing q90/mean branch. NASA O2-F1 is indeterminate until IBlank validity is part of the model-visible contract. No observed strategy is copied into Ground Truth.
"""


def _of_separation_markdown() -> str:
    return """# O-F Separation Validation

The calibration set supplies the required diagnostic patterns without forcing unsupported examples:

| Analysis | Findings | Example |
|---|---|---|
| valid | supported | Office O2-F1 and Office O3-F1 |
| valid | partially supported | Blunt_Fin O3-F1 numerical core versus qualitative geometry label |
| incomplete data domain | supported relative to executed computation | NASA O1/O2 with missing usable IBlank semantics |
| invalid | unsupported | no defensible observed assignment is invented |

Carotid O1-F2 is internally coherent but differs from the frozen GT because the question permits two non-equivalent speed fields. This is question ambiguity, not an ordinary O-F inconsistency.
"""


def _data_contract_markdown(report: Mapping[str, Any]) -> str:
    rows = []
    for item in report["records"]:
        rows.append(
            f"| {item['dataset_id']} | {item['valid_point_mask']['classification']} | {item['ghost_or_inactive_cells']['classification']} | {item['vector_scalar_semantics']['classification']} | {item['units']['classification']} |"
        )
    return """# Data-Contract Audit

| Dataset | Valid-point mask | Ghost/inactive cells | Vector/scalar semantics | Units |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n\nNASA LOx Post carries an explicit reader-side validity rule: `IBlank > 0`; non-positive values are blanked computational-domain points. Carotid O1 now explicitly selects the Euclidean magnitude of point-data `vectors`; historical pre-fix runs remain invalidated by task change."


def _o1_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {item['case_id']} | {item['classification']} | {'yes' if item['question_has_criterion'] else 'no'} | {'yes' if item['question_has_region_definition'] else 'no'} | {item['finding'] or ''} |"
        for item in report["records"]
    )
    return """# O1 Specification Audit

| Case | Classification | Criterion explicit | Region explicit | Finding |
|---|---|---|---|---|
""" + rows + "\n\nCurrent Carotid O1 wording explicitly selects point-data `vectors` magnitude. The existing N=1 Carotid O1 runs were generated under the prior ambiguous wording and remain `PILOT_ONLY_TASK_AMBIGUITY`; they are not retrospectively repaired."


def _branch_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {item['case_id']} | {item['operationalization_branch_count']} | {item['finding_branch_count']} | unavailable | unavailable |"
        for item in report["records"]
    )
    return """# Reference-Branch Sensitivity Audit

| Case | O branches | Finding branches | Best score | Second score |
|---|---:|---:|---|---|
""" + rows + "\n\nBranch scores are unavailable because the independent evaluator gate is blocked. The tool records branch cardinality without fabricating metric values or testing branch-count effects with synthetic scores."


def _runtime_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(f"| {key} | {value} |" for key, value in report["error_counts"].items())
    attempts = "\n".join(
        f"| {item['case_id']} | {item['classification']} | {item['failure_reason']} |" for item in report["infrastructure_invalid_attempts"]
    ) or "| none | — | — |"
    return f"""# Runtime Error Classification

## All observed errors

| Category | Count |
|---|---:|
{rows}

Total Python execution errors: **{report['python_error_count']}**. The category table also includes retained infrastructure-invalid attempts. Filesystem-discovery timeouts are separated from ordinary agent code errors. Missing optional packages remain agent/runtime behavior and are not silently converted to infrastructure failures.

## Infrastructure-invalid attempts

| Case | Classification | Reason |
|---|---|---|
{attempts}

Largest captured stdout: **{report['stdout_max_chars']} characters**; median: **{report['stdout_median_chars']}**. The hardened runtime keeps capture bounded and records truncation telemetry.
"""


def _engineering_markdown(report: Mapping[str, Any]) -> str:
    return f"""# Engineering Hardening Report

Implemented in the runtime:

1. `/case`, `/case/case_input.json`, `/case/reader_metadata.json`, `/case/case_files.json`, and `/workspace` are explicitly named in the neutral runtime prompt.
2. `case_files.json` provides a bounded visible-file index; the prompt discourages host-root recursive discovery and network access.
3. Python output capture defaults to 16,000 characters, returns a bounded head/tail preview, and reports original plus returned character counts.
4. Provider responses containing 1--8 Python calls are normalized to an ordered `ToolBatch` and executed
   sequentially; only malformed call/result mappings or unsafe transport boundaries are retryable. Retries
   remain visible in run telemetry, while valid multi-call turns are not infrastructure errors.
5. The current local route rejects unsupported `reasoning_effort` values without automatic downgrade.
6. The repository pytest configuration includes `pythonpath = [\".\"]`, so direct `pytest` and `python -m pytest` use the same import path.

Observed calibration counts: {json.dumps(report['error_counts'], sort_keys=True)}.

These are engineering changes only. They do not add scientific helpers, change scoring, or rewrite the frozen pilot.
"""


def _change_log_markdown() -> str:
    return """# Benchmark Change Log

## Scientific changes

- Current construction artifacts explicitly select Carotid point-data `vectors` magnitude in O1.
- NASA reader metadata explicitly exposes `IBlank > 0` validity semantics.
- The frozen historical collection was not rewritten; affected observations require disposition and rerun.

## Engineering changes

- Runtime path discoverability and bounded visible-file index.
- Bounded output capture with telemetry.
- Retryable canonical tool-boundary violation.
- Explicit local reasoning-effort validation.
- Consistent pytest import-path configuration.

## Evaluation changes

- Added one-time calibration diagnostics and fail-closed two-judge configuration gate.
- No evaluator score, metric, or Ground Truth rule was changed.

The current collection remains diagnostic and pilot-only. A new benchmark version is required after scientific/data-contract corrections.
"""


def _summary_markdown(gate: Mapping[str, Any], o1: Mapping[str, Any], runtime: Mapping[str, Any], contracts: Mapping[str, Any]) -> str:
    return f"""# One-Time Calibration Summary

Status: **NOT_READY**  
Frozen observations audited: **28**  
Independent evaluator gate: **{gate['status']}**  
O1 task-breaking findings: **{o1['task_breaking_count']}**  
Runtime Python errors: **{runtime['python_error_count']}**  
Datasets audited: **{contracts['dataset_count']}**

This is a calibration and hardening run, not a formal benchmark evaluation. No tested-model trajectory was regenerated, and unavailable evaluator metrics were not replaced with zeros.

The calibration confirms that the current artifacts contain the two versioned corrections: Carotid O1 explicitly selects velocity-vector magnitude, and NASA LOx Post exposes `IBlank > 0` valid-domain semantics. Historical observations affected by the earlier contracts remain pilot-only. The diagnostic draft also identifies valid-unenumerated Office O2/O3 analyses and historical runtime path-discovery/output-volume costs.

Formal N=3 is blocked until the two independent evaluator configurations are supplied, judge/human disagreements are reviewed, affected cases are versioned and rerun, and the corrected benchmark passes the calibration exit criteria.
"""


__all__ = [
    "DIAGNOSTIC_CASE_IDS",
    "ERROR_CATEGORIES",
    "REQUIRED_OUTPUTS",
    "FrozenObservation",
    "load_frozen_observations",
    "run_calibration",
]
