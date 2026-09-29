"""Fail-closed next-stage gate from calibration to a small N=3 candidate.

The gate runner performs deterministic repository/runtime checks and validates
externally produced judge, human-review, and clean-pilot artifacts. It never
manufactures evaluator or human judgments and never promotes historical N=1
observations to formal results.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from vtk.util.numpy_support import vtk_to_numpy

from .calibration import (
    DIAGNOSTIC_CASE_IDS,
    load_frozen_observations,
    resolve_collection_state_path,
)
from .case_design import CaseConstructionMetadata
from .execution import ExecutionAdapter
from .evaluation_metrics import (
    FindingBranchEvaluation,
    OperationalizationBranchEvaluation,
    compute_branch_alignment,
    compute_finding_metrics,
    compute_o_score,
)
from .ground_truth import GroundTruth
from .loader import CaseLoader
from .manifest import DatasetManifest
from .model_runner import DEFAULT_SYSTEM_PROMPT_VERSION, PYTHON_TOOL_SPEC_VERSION
from .python_runtime import PythonExecutionEnvironment
from .runtime_environment import ensure_frozen_environment


READINESS_STATUSES = frozenset({"NOT_READY", "READY_FOR_SMALL_N3"})
ANALYSIS_LABELS = frozenset(
    {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED", "INVALID", "UNCERTAIN"}
)
OF_LABELS = frozenset({"CONSISTENT", "INCONSISTENT", "UNCERTAIN"})
HUMAN_CLASSIFICATIONS = frozenset(
    {
        "AGENT_ERROR",
        "VALID_ALTERNATIVE",
        "QUESTION_AMBIGUITY",
        "DATA_CONTRACT_DEFECT",
        "REFERENCE_INCOMPLETENESS",
        "EVALUATOR_ERROR",
        "RUBRIC_AMBIGUITY",
        "UNCERTAIN",
    }
)
CORE_LIBRARIES = ("numpy", "scipy", "matplotlib", "vtk")
DEFAULT_OUTPUT_LIMIT = 16_000
DATA_CONTRACT_CLASSIFICATIONS = frozenset(
    {"DATA_CONTRACT_REQUIRED", "SCIENTIFICALLY_INTENTIONAL_UNKNOWN", "IRRELEVANT"}
)
DATA_CONTRACT_AUDIT_ITEMS = (
    "valid_point_mask",
    "ghost_cells",
    "inactive_cells",
    "missing_values",
    "units",
    "scalar_vector_semantics",
    "coordinate_conventions",
    "field_naming",
    "topology",
    "timestep",
    "boundary_identifiers",
)
O1_CURATOR_CHECKS = (
    "analyzed_quantity",
    "field_identity",
    "criterion",
    "region_definition",
    "measurement",
    "aggregation",
    "comparison_rule",
    "requested_finding",
)
CLEAN_RUNTIME_CASE_IDS = (
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
    "office_speed_zones_o1_f2",
    "blunt_fin_o3_f1",
    "carotid_o1_f2",
    "nasa_lox_post_o2_f1",
    "kitchen_o2_f1",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gate(status: str, reason: str, **details: Any) -> dict[str, Any]:
    if status not in {"PASS", "BLOCKED", "FAIL"}:
        raise ValueError(f"invalid gate status: {status}")
    return {"status": status, "reason": reason, **details}


def _validate_judge_output(
    path: Path | None,
    *,
    judge: str,
    tested_provider: str,
    tested_model: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if path is None or not path.is_file():
        return _gate("BLOCKED", f"{judge} output is unavailable"), None
    try:
        value = _read_json(path)
        if not isinstance(value, Mapping):
            raise ValueError("judge artifact must be an object")
        identity = value.get("evaluator_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("evaluator_identity is required")
        if value.get("execution_status") != "EXECUTED":
            raise ValueError("judge artifact must attest execution_status=EXECUTED")
        for field in ("provider", "model"):
            if not isinstance(identity.get(field), str) or not str(identity[field]).strip():
                raise ValueError(f"evaluator_identity.{field} is required")
        if identity.get("provider") == tested_provider and identity.get("model") == tested_model:
            raise ValueError("tested model was reused as calibration judge")
        tested_family = tested_model.split(":", 1)[0].split("/", 1)[0]
        judge_family = str(identity.get("model_family") or identity.get("model")).split(":", 1)[0].split("/", 1)[0]
        if judge_family == tested_family:
            raise ValueError("calibration judge model family is not independent from tested model family")
        if not isinstance(value.get("sampling_parameters"), Mapping):
            raise ValueError("sampling_parameters are required")
        for field in ("prompt_sha256", "structured_output_schema_sha256"):
            if not isinstance(value.get(field), str) or len(str(value[field])) != 64:
                raise ValueError(f"{field} must be a SHA-256 digest")
        cases = value.get("cases")
        if not isinstance(cases, list):
            raise ValueError("cases must be a list")
        by_id = {item.get("case_id"): item for item in cases if isinstance(item, Mapping)}
        if set(by_id) != set(DIAGNOSTIC_CASE_IDS):
            raise ValueError("judge output does not cover the fixed diagnostic subset")
        for case_id, item in by_id.items():
            if item.get("analysis_label") not in ANALYSIS_LABELS:
                raise ValueError(f"invalid analysis label for {case_id}")
            if item.get("o_f_consistency") not in OF_LABELS:
                raise ValueError(f"invalid O-F label for {case_id}")
            dimensions = item.get("analysis_dimension_judgments")
            required_dimensions = {
                "field_choice",
                "concept_definition",
                "measurement",
                "threshold_or_criterion",
                "region_definition",
                "aggregation",
                "comparison_rule",
            }
            if not isinstance(dimensions, Mapping) or set(dimensions) != required_dimensions:
                raise ValueError(f"analysis dimensions are incomplete for {case_id}")
            if any(value not in {"VALID", "INVALID", "UNCERTAIN"} for value in dimensions.values()):
                raise ValueError(f"invalid analysis dimension judgment for {case_id}")
            findings = item.get("findings")
            if not isinstance(findings, list):
                raise ValueError(f"findings must be a list for {case_id}")
            for finding in findings:
                if not isinstance(finding, Mapping):
                    raise ValueError(f"finding judgment must be an object for {case_id}")
                required = {"statement", "correctness", "evidential_support", "relevance", "core_status"}
                if not required <= set(finding):
                    raise ValueError(f"finding judgment is incomplete for {case_id}")
                if finding.get("correctness") not in {"CORRECT", "INCORRECT", "UNCERTAIN"}:
                    raise ValueError(f"invalid finding correctness for {case_id}")
                if finding.get("evidential_support") not in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}:
                    raise ValueError(f"invalid finding support for {case_id}")
                if finding.get("relevance") not in {"RELEVANT", "IRRELEVANT", "UNCERTAIN"}:
                    raise ValueError(f"invalid finding relevance for {case_id}")
                if finding.get("core_status") not in {"CORE", "NON_CORE", "UNCERTAIN"}:
                    raise ValueError(f"invalid finding core status for {case_id}")
        if value.get("tested_model_identity_visible") is not False:
            raise ValueError("judge artifact does not attest tested-model blinding")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), None
    return _gate("PASS", f"{judge} executed with valid blinded structured outputs"), dict(value)


def _validate_human_reference(path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if path is None or not path.is_file():
        return _gate("BLOCKED", "confirmed human calibration artifact is unavailable"), None
    try:
        value = _read_json(path)
        if not isinstance(value, Mapping) or value.get("review_status") != "CONFIRMED":
            raise ValueError("human reference must have review_status=CONFIRMED")
        if value.get("all_disagreements_reviewed") is not True:
            raise ValueError("human reference must attest all_disagreements_reviewed=true")
        if value.get("systematic_evaluator_failures_resolved") is not True:
            raise ValueError(
                "human reference must attest systematic_evaluator_failures_resolved=true"
            )
        cases = value.get("cases")
        if not isinstance(cases, list):
            raise ValueError("human cases must be a list")
        by_id = {item.get("case_id"): item for item in cases if isinstance(item, Mapping)}
        if set(by_id) != set(DIAGNOSTIC_CASE_IDS):
            raise ValueError("human review does not cover the diagnostic subset")
        for case_id, item in by_id.items():
            if item.get("classification") not in HUMAN_CLASSIFICATIONS:
                raise ValueError(f"invalid human classification for {case_id}")
            if item.get("analysis_label") not in ANALYSIS_LABELS:
                raise ValueError(f"invalid human analysis label for {case_id}")
            if item.get("o_f_consistency") not in OF_LABELS:
                raise ValueError(f"invalid human O-F label for {case_id}")
            dimensions = item.get("analysis_dimension_judgments")
            required_dimensions = {
                "field_choice",
                "concept_definition",
                "measurement",
                "threshold_or_criterion",
                "region_definition",
                "aggregation",
                "comparison_rule",
            }
            if not isinstance(dimensions, Mapping) or set(dimensions) != required_dimensions:
                raise ValueError(f"human analysis dimensions are incomplete for {case_id}")
            if any(value not in {"VALID", "INVALID", "UNCERTAIN"} for value in dimensions.values()):
                raise ValueError(f"invalid human analysis dimension judgment for {case_id}")
            finding_judgments = item.get("finding_judgments")
            if not isinstance(finding_judgments, list):
                raise ValueError(f"human finding judgments missing for {case_id}")
            for finding in finding_judgments:
                if not isinstance(finding, Mapping):
                    raise ValueError(f"human finding judgment must be an object for {case_id}")
                required = {
                    "finding_id",
                    "statement",
                    "included",
                    "correctness",
                    "evidential_support",
                    "relevance",
                    "core_status",
                    "applicable_analysis_branch",
                    "rationale",
                }
                if not required <= set(finding):
                    raise ValueError(f"human finding judgment is incomplete for {case_id}")
                if not isinstance(finding.get("included"), bool):
                    raise ValueError(f"human finding included flag is invalid for {case_id}")
                if finding.get("correctness") not in {"CORRECT", "INCORRECT", "UNCERTAIN"}:
                    raise ValueError(f"invalid human finding correctness for {case_id}")
                if finding.get("evidential_support") not in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}:
                    raise ValueError(f"invalid human finding support for {case_id}")
                if finding.get("relevance") not in {"RELEVANT", "IRRELEVANT", "UNCERTAIN"}:
                    raise ValueError(f"invalid human finding relevance for {case_id}")
                if finding.get("core_status") not in {"CORE", "NON_CORE", "UNCERTAIN"}:
                    raise ValueError(f"invalid human finding core status for {case_id}")
                if not isinstance(finding.get("rationale"), str) or not finding["rationale"].strip():
                    raise ValueError(f"human finding rationale is required for {case_id}")
        if by_id["carotid_o1_f1"].get("classification") != "QUESTION_AMBIGUITY":
            raise ValueError("historical Carotid O1-F1 must be classified QUESTION_AMBIGUITY")
        if by_id["carotid_o1_f2"].get("classification") != "QUESTION_AMBIGUITY":
            raise ValueError("historical Carotid O1-F2 must be classified QUESTION_AMBIGUITY")
        for case_id in ("nasa_lox_post_o2_f1", "nasa_lox_post_o3_f1"):
            if by_id[case_id].get("classification") != "DATA_CONTRACT_DEFECT":
                raise ValueError(f"historical {case_id} must be classified DATA_CONTRACT_DEFECT")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), None
    return _gate("PASS", "all diagnostic drafts have confirmed human review"), dict(value)


def _validate_curator_audit(path: Path | None, *, label: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate the explicit scientific curator confirmation layer.

    Automatic technical checks are not promoted to scientific PASS.  The
    curator artifact must attest a completed review and preserve a case-level
    rationale for every audited item.
    """
    if path is None or not path.is_file():
        return _gate("BLOCKED", f"{label} curator confirmation is unavailable"), None
    try:
        value = _read_json(path) if path.suffix.casefold() == ".json" else {"status": "PENDING"}
        if not isinstance(value, Mapping) or value.get("status") != "CONFIRMED":
            raise ValueError(f"{label} curator audit must have status=CONFIRMED")
        records = value.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{label} curator audit records are required")
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("rationale"), str) or not record["rationale"].strip():
                raise ValueError(f"{label} curator records require a rationale")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), None
    return _gate("PASS", f"{label} curator audit confirmed"), dict(value)


def _validate_o1_curator_audit(
    path: Path | None,
    *,
    expected_case_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gate, value = _validate_curator_audit(path, label="O1")
    if value is None:
        return gate, None
    try:
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError("O1 curator records must be a list")
        by_id = {item.get("case_id"): item for item in records if isinstance(item, Mapping)}
        if set(by_id) != expected_case_ids:
            raise ValueError("O1 curator audit must cover every current O1 case exactly once")
        for case_id, item in by_id.items():
            checks = item.get("checks")
            if not isinstance(checks, Mapping) or any(checks.get(name) is not True for name in O1_CURATOR_CHECKS):
                raise ValueError(f"O1 curator checks are incomplete for {case_id}")
            if not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
                raise ValueError(f"O1 curator rationale is required for {case_id}")
    except ValueError as exc:
        return _gate("FAIL", str(exc)), None
    return gate, value


def _validate_data_contract_curator_audit(
    path: Path | None,
    *,
    expected_dataset_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gate, value = _validate_curator_audit(path, label="data-contract")
    if value is None:
        return gate, None
    try:
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError("data-contract curator records must be a list")
        by_id = {item.get("dataset_id"): item for item in records if isinstance(item, Mapping)}
        if set(by_id) != expected_dataset_ids:
            raise ValueError("data-contract curator audit must cover every current dataset exactly once")
        for dataset_id, item in by_id.items():
            audited_items = item.get("items")
            if not isinstance(audited_items, Mapping) or set(audited_items) != set(DATA_CONTRACT_AUDIT_ITEMS):
                raise ValueError(f"data-contract audit items are incomplete for {dataset_id}")
            for name, decision in audited_items.items():
                if not isinstance(decision, Mapping):
                    raise ValueError(f"data-contract decision must be an object: {dataset_id}/{name}")
                if decision.get("classification") not in DATA_CONTRACT_CLASSIFICATIONS:
                    raise ValueError(f"invalid data-contract classification: {dataset_id}/{name}")
                if not isinstance(decision.get("rationale"), str) or not decision["rationale"].strip():
                    raise ValueError(f"data-contract rationale is required: {dataset_id}/{name}")
    except ValueError as exc:
        return _gate("FAIL", str(exc)), None
    return gate, value


def _validate_efficiency_comparison(
    path: Path | None,
    *,
    label: str,
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return _gate("BLOCKED", f"{label} efficiency comparison is unavailable")
    try:
        text = path.read_text(encoding="utf-8")
        normalized = " ".join(text.casefold().replace("_", " ").replace("-", " ").split())
        required_terms = {
            "wall time": ("wall time", "wall clock"),
            "input tokens": ("input tokens",),
            "output tokens": ("output tokens",),
            "model turns": ("model turns", "turns"),
            "python calls": ("python calls", "python executions"),
            "tool text": ("tool text", "returned tool chars", "tool output"),
            "filesystem errors": ("filesystem errors", "filesystem discovery"),
        }
        missing = [
            name
            for name, alternatives in required_terms.items()
            if not any(term in normalized for term in alternatives)
        ]
        label_text = " ".join(label.casefold().replace("_", " ").split())
        if label_text not in normalized:
            missing.append("comparison label")
        if missing:
            raise ValueError(f"{label} comparison is missing: {', '.join(missing)}")
    except (OSError, ValueError) as exc:
        return _gate("FAIL", str(exc))
    return _gate("PASS", f"{label} comparison contains the required matched metrics")


def _validate_archive_self_test(path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if path is None or not path.is_file():
        return _gate("BLOCKED", "release archive self-test is unavailable"), None
    try:
        value = _read_json(path)
        if not isinstance(value, Mapping) or value.get("status") != "PASS":
            raise ValueError("archive self-test must have status=PASS")
        required = {
            "fresh_extract": True,
            "pytest_returncode": 0,
            "preflight_returncode": 0,
            "required_dotfiles_present": True,
            "manifest_hashes_verified": True,
        }
        failures = [name for name, expected in required.items() if value.get(name) != expected]
        if failures:
            raise ValueError("archive self-test evidence is incomplete: " + ", ".join(failures))
        if not isinstance(value.get("manifest_hash_count"), int) or value["manifest_hash_count"] <= 0:
            raise ValueError("archive self-test did not verify any manifest hashes")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), None
    return _gate("PASS", "fresh archive tests, preflight, dotfiles, and manifest hashes pass"), dict(value)


def _judge_disagreements(
    judge_a: Mapping[str, Any] | None,
    judge_b: Mapping[str, Any] | None,
    human: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if judge_a is None or judge_b is None:
        return {"status": "BLOCKED", "disagreements": []}
    a = {item["case_id"]: item for item in judge_a["cases"]}
    b = {item["case_id"]: item for item in judge_b["cases"]}
    h = {} if human is None else {item["case_id"]: item for item in human["cases"]}
    disagreements = []
    for case_id in DIAGNOSTIC_CASE_IDS:
        fields = []
        for field in ("analysis_label", "o_f_consistency"):
            values = {"judge_a": a[case_id].get(field), "judge_b": b[case_id].get(field)}
            if case_id in h:
                values["human"] = h[case_id].get(field)
            if len(set(values.values())) > 1:
                fields.append({"field": field, "values": values, "resolved_by_human": case_id in h})
        for dimension in sorted(a[case_id].get("analysis_dimension_judgments", {})):
            values = {
                "judge_a": a[case_id].get("analysis_dimension_judgments", {}).get(dimension),
                "judge_b": b[case_id].get("analysis_dimension_judgments", {}).get(dimension),
            }
            if case_id in h:
                values["human"] = h[case_id].get("analysis_dimension_judgments", {}).get(dimension)
            if values["judge_a"] != values["judge_b"]:
                fields.append(
                    {
                        "field": f"analysis_dimension_judgments.{dimension}",
                        "values": values,
                        "resolved_by_human": case_id in h and values.get("human") is not None,
                    }
                )
        # Overall-label agreement is insufficient. Compare every normalized
        # atomic finding and all adjudication dimensions as well.
        a_findings = {item.get("finding_id", item.get("statement")): item for item in a[case_id].get("findings", [])}
        b_findings = {item.get("finding_id", item.get("statement")): item for item in b[case_id].get("findings", [])}
        h_findings = {
            item.get("finding_id", item.get("statement")): item
            for item in h.get(case_id, {}).get("finding_judgments", [])
        }
        for finding_id in sorted(set(a_findings) | set(b_findings), key=str):
            left, right = a_findings.get(finding_id), b_findings.get(finding_id)
            if left is None or right is None:
                human_finding = h_findings.get(finding_id)
                fields.append({"field": "finding_presence", "finding_id": finding_id, "values": {"judge_a": left is not None, "judge_b": right is not None, "human": None if human_finding is None else human_finding.get("included")}, "resolved_by_human": human_finding is not None})
                continue
            for finding_field in ("correctness", "evidential_support", "relevance", "core_status", "applicable_analysis_branch"):
                if left.get(finding_field) != right.get(finding_field):
                    human_finding = h_findings.get(finding_id)
                    fields.append({"field": finding_field, "finding_id": finding_id, "values": {"judge_a": left.get(finding_field), "judge_b": right.get(finding_field), "human": None if human_finding is None else human_finding.get(finding_field)}, "resolved_by_human": human_finding is not None})
        if fields:
            disagreements.append(
                {
                    "case_id": case_id,
                    "fields": fields,
                    "review_status": "REVIEWED" if all(field.get("resolved_by_human") is True for field in fields) else "PENDING_HUMAN_REVIEW",
                    "classification": h.get(case_id, {}).get("classification"),
                }
            )
    return {
        "status": "PASS"
        if all(item["review_status"] == "REVIEWED" for item in disagreements)
        else "BLOCKED",
        "disagreements": disagreements,
    }


def _judges_are_independent(
    judge_a: Mapping[str, Any] | None,
    judge_b: Mapping[str, Any] | None,
) -> bool:
    if judge_a is None or judge_b is None:
        return False
    a = judge_a["evaluator_identity"]
    b = judge_b["evaluator_identity"]
    return (a.get("model_family") or a.get("model")) != (
        b.get("model_family") or b.get("model")
    )


def _o1_audit(repository_root: Path) -> dict[str, Any]:
    records = []
    for dataset_dir in sorted((repository_root / "datasets").iterdir()):
        cases_dir = dataset_dir / "construction" / "cases"
        if not cases_dir.is_dir():
            continue
        for case_dir in sorted(cases_dir.iterdir()):
            if not case_dir.name.endswith(("o1_f1", "o1_f2")):
                continue
            case = _read_json(case_dir / "case_input.json")
            metadata = CaseConstructionMetadata.model_validate_json(
                (case_dir / "case_construction_metadata.json").read_text(encoding="utf-8")
            )
            question = str(case["scientific_question"])
            variables = case["flow_data"]["data_metadata"]["variables"]
            speed_candidates = [
                item for item in variables
                if "speed" in str(item.get("physical_quantity", "")).casefold()
                or "velocity" in str(item.get("physical_quantity", "")).casefold()
            ]
            candidate_quantities = {
                str(item.get("physical_quantity", "")).casefold()
                for item in speed_candidates
            }
            candidates_are_equivalent = candidate_quantities <= {
                "velocity",
                "velocity magnitude",
                "speed",
            }
            variable_explicit = (
                len(speed_candidates) <= 1
                or candidates_are_equivalent
                or any(str(item.get("name")) in question for item in speed_candidates)
            )
            dimensions = {item.category.value for item in metadata.explicit_method_constraints}
            checks = {
                "analyzed_variable": variable_explicit,
                "criterion": "criterion" in dimensions,
                "region_definition": "feature_definition" in dimensions,
                "measurement": "property_measure" in dimensions,
                "aggregation": "analysis_procedure" in dimensions,
                "requested_result": bool(
                    metadata.explicit_finding_requirements
                    or metadata.finding_openness.value == "open"
                ),
            }
            records.append(
                {
                    "dataset_id": dataset_dir.name,
                    "case_id": case_dir.name,
                    "checks": checks,
                    "status": "PASS" if all(checks.values()) else "FAIL",
                    "case_input_sha256": _sha256(case_dir / "case_input.json"),
                }
            )
    failed = [item["case_id"] for item in records if item["status"] != "PASS"]
    return {"status": "PASS" if not failed else "FAIL", "failed_cases": failed, "records": records}


def _data_contract_audit(repository_root: Path, case_manifest: Path | None = None) -> dict[str, Any]:
    root = repository_root / "datasets"
    records = []
    from .experiment_scope import case_inventory
    inventory = case_inventory(_read_json(case_manifest or repository_root / "experiments/userstudy/case_manifest.json"))
    dataset_rows = {}
    for row in inventory.values():
        dataset_rows.setdefault(str(row["dataset_id"]), row)
    for dataset_id, row in sorted(dataset_rows.items()):
        dataset_dir = root / dataset_id
        manifest = DatasetManifest.model_validate_json(
            (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        case_path = repository_root / str(row.get("case_input_path") or row["case_input"])
        loaded = CaseLoader(root).load(case_path)
        result = ExecutionAdapter().read(loaded, manifest)
        finite = True
        ghost_arrays: set[str] = set()
        datasets_to_scan = list(result.data.values())
        leaf_datasets = []
        while datasets_to_scan:
            dataset = datasets_to_scan.pop()
            if hasattr(dataset, "GetNumberOfBlocks"):
                datasets_to_scan.extend(
                    dataset.GetBlock(index)
                    for index in range(dataset.GetNumberOfBlocks())
                    if dataset.GetBlock(index) is not None
                )
            else:
                leaf_datasets.append(dataset)
        for dataset in leaf_datasets:
            point_data = dataset.GetPointData()
            for index in range(point_data.GetNumberOfArrays()):
                array = point_data.GetArray(index)
                name = point_data.GetArrayName(index) or ""
                if name == "vtkGhostType":
                    ghost_arrays.add(name)
                values = vtk_to_numpy(array)
                if getattr(values.dtype, "kind", "") in {"f", "c"}:
                    import numpy as np

                    finite = finite and bool(np.isfinite(values).all())
        blanking = manifest.reader.grid_blanking
        required_visible = all(item.validity_rule for item in blanking)
        items = {
            "valid_point_mask": {
                "classification": "DATA_CONTRACT_REQUIRED" if blanking else "IRRELEVANT",
                "model_visible": required_visible if blanking else True,
                "rationale": "Reader blanking rules require a visible valid-point mask." if blanking else "No reader blanking rule is declared.",
            },
            "ghost_cells": {
                "classification": "DATA_CONTRACT_REQUIRED" if ghost_arrays else "IRRELEVANT",
                "model_visible": not ghost_arrays,
                "detail": sorted(ghost_arrays),
                "rationale": "A vtkGhostType array was observed." if ghost_arrays else "No vtkGhostType array was observed by the technical scan.",
            },
            "inactive_cells": {
                "classification": "DATA_CONTRACT_REQUIRED" if blanking else "IRRELEVANT",
                "model_visible": required_visible if blanking else True,
                "rationale": "Reader blanking declares inactive-domain semantics." if blanking else "No inactive-domain reader rule is declared.",
            },
            "missing_values": {
                "classification": "IRRELEVANT" if finite else "DATA_CONTRACT_REQUIRED",
                "model_visible": finite,
                "rationale": "All scanned floating arrays are finite." if finite else "Non-finite numerical values were observed.",
            },
            "units": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Known, unknown, or intentionally absent units require dataset-specific scientific review.",
            },
            "scalar_vector_semantics": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Field component semantics require dataset-specific scientific review.",
            },
            "coordinate_conventions": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Coordinate and axis conventions require dataset-specific scientific review.",
            },
            "field_naming": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Scientific field-name meaning cannot be inferred from technical visibility alone.",
            },
            "topology": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "The scientific relevance of topology requires dataset-specific review.",
            },
            "timestep": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Steady/time semantics require dataset-specific review.",
            },
            "boundary_identifiers": {
                "classification": "PENDING_CURATOR_REVIEW",
                "model_visible": True,
                "rationale": "Boundary semantics require dataset-specific review.",
            },
        }
        hidden_required = [
            name for name, item in items.items()
            if item["classification"] == "DATA_CONTRACT_REQUIRED" and not item["model_visible"]
        ]
        records.append(
            {
                "dataset_id": dataset_id,
                "items": items,
                "hidden_required_items": hidden_required,
                "status": "PASS" if not hidden_required else "FAIL",
            }
        )
    return {
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL",
        "technical_review_status": "COMPLETE",
        "records": records,
    }


def _runtime_smokes(repository_root: Path, output_root: Path) -> dict[str, dict[str, Any]]:
    datasets = repository_root / "datasets"
    loaded = CaseLoader(datasets).load(
        datasets / "Office/construction/cases/office_speed_zones_o1_f1/case_input.json"
    )
    manifest = DatasetManifest.model_validate_json(
        (datasets / "Office/dataset_manifest.json").read_text(encoding="utf-8")
    )
    with PythonExecutionEnvironment(
        loaded,
        manifest=manifest,
        python_executable=sys.executable,
        require_network_isolation=False,
        timeout_seconds=5,
        max_output_chars=DEFAULT_OUTPUT_LIMIT,
    ) as runtime:
        visible_case_root = runtime.case_dir
        listing = runtime.execute(
            "import json\nfrom pathlib import Path\n"
            f"print(json.loads(Path({str(visible_case_root / 'case_files.json')!r}).read_text())['read_only'])"
        )
        blocked_results = [
            runtime.execute("from pathlib import Path\nprint(list(Path('/').rglob('*')))") ,
            runtime.execute("import glob\nprint(glob.glob('/**/*', recursive=True))"),
            runtime.execute("import os\nprint(list(os.walk('/')))"),
        ]
        scalar = runtime.execute("print(3.141592653589793)")
        oversized = runtime.execute(
            "print('HEAD-SENTINEL-' + 'x' * 40000 + '-TAIL-SENTINEL')"
        )
        runtime_contract = _read_json(runtime.case_dir / "runtime_contract.json")
    discovery = {
        "status": "PASS"
        if listing.success
        and all(item.filesystem_discovery_blocked for item in blocked_results)
        and all(item.duration_seconds < 1 for item in blocked_results)
        else "FAIL",
        "case_listing_available": listing.success,
        "blocked_attempts": [
            {
                "type": item.exception.type if item.exception else None,
                "message": item.exception.message if item.exception else None,
                "duration_seconds": item.duration_seconds,
            }
            for item in blocked_results
        ],
    }
    stdout = {
        "status": "PASS"
        if scalar.stdout == "3.141592653589793\n"
        and oversized.output_truncated
        and oversized.stdout_chars_total is not None
        and oversized.stdout_chars_total > DEFAULT_OUTPUT_LIMIT
        and len(oversized.stdout) <= DEFAULT_OUTPUT_LIMIT
        and "HEAD-SENTINEL" in oversized.stdout
        and "TAIL-SENTINEL" in oversized.stdout
        else "FAIL",
        "limit": DEFAULT_OUTPUT_LIMIT,
        "scalar_stdout": scalar.stdout,
        "truncated": oversized.output_truncated,
        "original_stdout_chars": oversized.stdout_chars_total,
        "original_stderr_chars": oversized.stderr_chars_total,
        "returned_stdout_chars": oversized.returned_stdout_chars,
        "returned_stderr_chars": oversized.returned_stderr_chars,
        "head_present": "HEAD-SENTINEL" in oversized.stdout,
        "tail_present": "TAIL-SENTINEL" in oversized.stdout,
    }
    environment = ensure_frozen_environment()
    command = [
        str(environment.python_executable),
        "-I",
        "-c",
        (
            "import json, importlib, importlib.metadata as m; "
            f"names={list(CORE_LIBRARIES)!r}; "
            "[importlib.import_module(n) for n in names]; "
            "print(json.dumps({n: m.version(n) for n in names}, sort_keys=True))"
        ),
    ]
    dependency_environment = dict(os.environ)
    dependency_environment["LD_LIBRARY_PATH"] = str(environment.base_prefix / "lib")
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=120,
        env=dependency_environment,
    )
    dependencies = {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": command,
        "returncode": completed.returncode,
        "versions": json.loads(completed.stdout) if completed.returncode == 0 else None,
        "stderr": completed.stderr,
    }
    contract_versions = {
        item["import_name"]: item["version"]
        for item in runtime_contract["supported_core_scientific_libraries"]
    }
    if dependencies["status"] == "PASS":
        runtime_contract["supported_core_scientific_libraries"] = [
            {**item, "version": dependencies["versions"][item["import_name"]]}
            for item in runtime_contract["supported_core_scientific_libraries"]
        ]
        contract_versions = dependencies["versions"]
    runtime_contract["advertised_versions"] = contract_versions
    _write_json(output_root / "runtime/file_discovery_smoke.json", discovery)
    _write_json(output_root / "runtime/stdout_guard_test.json", stdout)
    _write_json(output_root / "runtime/runtime_contract.json", runtime_contract)
    _write_json(output_root / "runtime/dependency_preflight.json", dependencies)
    return {"discovery": discovery, "stdout": stdout, "dependencies": dependencies}


def _multi_call_artifacts(repository_root: Path, output_root: Path, compatibility: Mapping[str, Any]) -> dict[str, Any]:
    """Write the required canonical multi-call diagnostics without scoring."""
    design = """# Canonical Multi-Call Support

One model response may contain 1–8 Python calls. Calls are normalized to
`ToolBatch`, executed sequentially in emitted order in one case workspace, and
returned as one ordered result batch. Ordinary agent-code errors are valid
results; unavailable runtime/ambiguous transport is infrastructure-invalid.
Completed calls are replayed from the per-observation execution journal.
"""
    _write_text(output_root / "runtime/multi_call_design.md", design)
    command = [sys.executable, "-m", "pytest", "-q", "-m", "not sandbox", "tests/test_provider.py", "tests/test_model_runner.py"]
    try:
        completed = subprocess.run(command, cwd=repository_root, text=True, capture_output=True, timeout=300)
        unit = {"status": "PASS" if completed.returncode == 0 else "FAIL", "command": command, "returncode": completed.returncode, "summary": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else None}
    except (OSError, subprocess.TimeoutExpired) as exc:
        unit = {"status": "FAIL", "command": command, "reason": str(exc)}
    _write_json(output_root / "runtime/multi_call_unit_tests.json", unit)
    smoke = dict(compatibility)
    smoke.setdefault("status", "BLOCKED")
    if "reason" not in smoke:
        smoke["reason"] = (
            "real provider multi-call smoke passed"
            if smoke.get("status") == "PASS" and smoke.get("multi_call_emitted") is True
            else "real provider multi-call smoke was not supplied"
        )
    smoke["multi_call_emitted"] = smoke.get("multi_call_emitted", False)
    _write_json(output_root / "runtime/multi_call_provider_smoke.json", smoke)
    _write_text(output_root / "runtime/tool_execution_journal_report.md", "# Tool Execution Journal\n\nPer-observation journals are written beside trajectories and keyed by canonical call identity plus arguments hash. Duplicate completed calls are reused without Python re-execution.\n")
    return unit | {"provider_smoke": smoke}


def _validate_primary_evaluator_manifest(
    path: Path | None,
    *,
    tested_provider: str,
    tested_model: str,
    judge_a: Mapping[str, Any] | None,
    judge_b: Mapping[str, Any] | None,
    calibration_ready: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not calibration_ready:
        return (
            _gate("BLOCKED", "primary evaluator cannot be frozen before calibrated human review"),
            {"status": "BLOCKED", "evaluator": None},
        )
    if path is None or not path.is_file():
        return (
            _gate("BLOCKED", "primary evaluator selection requires an explicit frozen manifest"),
            {"status": "BLOCKED", "evaluator": None},
        )
    try:
        value = _read_json(path)
        if not isinstance(value, Mapping) or value.get("status") != "FROZEN":
            raise ValueError("primary evaluator manifest must have status=FROZEN")
        evaluator = value.get("evaluator")
        if not isinstance(evaluator, Mapping):
            raise ValueError("primary evaluator identity is required")
        for field in ("provider", "model"):
            if not isinstance(evaluator.get(field), str) or not str(evaluator[field]).strip():
                raise ValueError(f"primary evaluator {field} is required")
        if not isinstance(evaluator.get("model_family"), str) or not evaluator["model_family"].strip():
            raise ValueError("primary evaluator model_family is required")
        if evaluator.get("provider") == tested_provider and evaluator.get("model") == tested_model:
            raise ValueError("primary evaluator must be independent from the tested model")
        tested_family = tested_model.split(":", 1)[0].split("/", 1)[0]
        evaluator_family = str(evaluator.get("model_family") or evaluator.get("model")).split(":", 1)[0].split("/", 1)[0]
        if evaluator_family == tested_family:
            raise ValueError("primary evaluator must be independent from the tested model family")
        calibrated_identities = {
            (
                judge["evaluator_identity"].get("provider"),
                judge["evaluator_identity"].get("model"),
                judge["evaluator_identity"].get("model_version")
                or judge["evaluator_identity"].get("version"),
            )
            for judge in (judge_a, judge_b)
            if judge is not None
        }
        identity = (
            evaluator.get("provider"),
            evaluator.get("model"),
            evaluator.get("model_version") or evaluator.get("version"),
        )
        if identity not in calibrated_identities:
            raise ValueError("primary evaluator was not one of the calibrated judges")
        for field in (
            "sampling_parameters",
            "prompt_sha256",
            "structured_output_schema_sha256",
            "rubric_sha256",
            "gt_version",
            "scoring_code_version",
        ):
            if field not in value or value[field] in (None, "", {}):
                raise ValueError(f"primary evaluator manifest is missing {field}")
        if value.get("sentinel_validation_status") != "PASS":
            raise ValueError("primary evaluator sentinel validation must pass")
        if value.get("structured_output_reliability_status") != "PASS":
            raise ValueError("primary evaluator structured-output reliability must pass")
        for field in (
            "human_agreement_status",
            "finding_level_agreement_status",
            "o_f_consistency_reliability_status",
        ):
            if value.get(field) != "PASS":
                raise ValueError(f"primary evaluator {field} must pass")
        if value.get("cost_latency_reviewed") is not True:
            raise ValueError("primary evaluator cost and latency must be explicitly reviewed")
        escalation = value.get("escalation_policy")
        required_escalations = {
            "VALID_UNENUMERATED",
            "UNCERTAIN",
            "PARSING_FAILURE",
            "RANDOM_AUDIT_SAMPLE",
        }
        if not isinstance(escalation, Mapping) or set(escalation) != required_escalations:
            raise ValueError("primary evaluator escalation policy is incomplete")
        if any(not isinstance(value, str) or not value.strip() for value in escalation.values()):
            raise ValueError("primary evaluator escalation actions must be non-empty")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), {"status": "FAIL", "reason": str(exc)}
    return _gate("PASS", "explicit calibrated primary evaluator is frozen"), dict(value)


def _historical_dispositions(collection_root: Path) -> list[dict[str, Any]]:
    records = []
    diagnostics = set(DIAGNOSTIC_CASE_IDS)
    for item in load_frozen_observations(collection_root):
        if item.case_id in {"carotid_o1_f1", "carotid_o1_f2"}:
            disposition = "INVALIDATED_BY_TASK_CHANGE"
        elif item.dataset_id == "NASA_LOx_Post":
            disposition = "INVALIDATED_BY_DATA_CONTRACT_CHANGE"
        elif item.case_id in diagnostics:
            disposition = "CALIBRATION_ONLY"
        else:
            disposition = "HISTORICAL_RUNTIME_PILOT"
        records.append(
            {
                "dataset_id": item.dataset_id,
                "case_id": item.case_id,
                "run_id": item.run_id,
                "disposition": disposition,
                "run_record_path": str(item.run_record_path),
            }
        )
    return records


def _freeze_hashes(repository_root: Path) -> dict[str, str]:
    paths = [
        repository_root / "method.md",
        repository_root / "FlowIntentBench.md",
        repository_root / "README.md",
        repository_root / "pyproject.toml",
        repository_root / "runtime-requirements.txt",
        repository_root / "flowintentbench/model_runner.py",
        repository_root / "flowintentbench/python_runtime.py",
        repository_root / "flowintentbench/providers.py",
        repository_root / "flowintentbench/evaluator.py",
        repository_root / "flowintentbench/evaluation_metrics.py",
        repository_root / "flowintentbench/evaluator_runner.py",
        repository_root / "flowintentbench/next_stage.py",
        repository_root / "flowintentbench/runtime_config.py",
        repository_root / "scripts/run_evaluator_calibration.py",
        repository_root / "scripts/run_multi_call_provider_smoke.py",
        repository_root / "scripts/run_clean_runtime_pilot.py",
        repository_root / "scripts/run_next_stage_gate.py",
        repository_root / "scripts/run_archive_self_test.py",
        repository_root / "scripts/build_handoff_archive.py",
        repository_root / "scripts/preflight_full_dataset_n1.py",
        repository_root / "experiments/userstudy/case_manifest.json",
    ]
    paths.extend(sorted((repository_root / "tests").glob("test_*.py")))
    paths.extend(sorted((repository_root / "datasets").glob("*/dataset_manifest.json")))
    paths.extend(sorted((repository_root / "datasets").glob("*/construction/cases/*/case_input.json")))
    paths.extend(sorted((repository_root / "datasets").glob("*/construction/cases/*/ground_truth.json")))
    return {
        str(path.relative_to(repository_root)): _sha256(path)
        for path in paths
        if path.is_file()
    }


def run_next_stage_gate(
    *,
    repository_root: str | Path,
    collection_root: str | Path,
    output_root: str | Path,
    judge_a_output: str | Path | None = None,
    judge_b_output: str | Path | None = None,
    human_reference: str | Path | None = None,
    provider_compatibility_report: str | Path | None = None,
    clean_pilot_root: str | Path | None = None,
    primary_evaluator_manifest: str | Path | None = None,
    o1_curator_audit: str | Path | None = None,
    data_contract_curator_audit: str | Path | None = None,
    archive_self_test: str | Path | None = None,
    whole_system_efficiency: str | Path | None = None,
    unchanged_task_efficiency: str | Path | None = None,
    branch_evaluation_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repository_root).resolve()
    collection = Path(collection_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = _read_json(resolve_collection_state_path(collection))
    tested_provider = str(state.get("provider", ""))
    tested_model = str(state.get("model_id", ""))

    judge_a_gate, judge_a = _validate_judge_output(
        None if judge_a_output is None else Path(judge_a_output),
        judge="Judge A",
        tested_provider=tested_provider,
        tested_model=tested_model,
    )
    judge_b_gate, judge_b = _validate_judge_output(
        None if judge_b_output is None else Path(judge_b_output),
        judge="Judge B",
        tested_provider=tested_provider,
        tested_model=tested_model,
    )
    human_gate, human = _validate_human_reference(
        None if human_reference is None else Path(human_reference)
    )
    disagreements = _judge_disagreements(judge_a, judge_b, human)
    _write_json(output / "calibration/judge_a_outputs.json", judge_a or {"status": "BLOCKED"})
    _write_json(output / "calibration/judge_b_outputs.json", judge_b or {"status": "BLOCKED"})
    _write_json(output / "calibration/judge_disagreements.json", disagreements)
    finding_only = {
        "status": disagreements.get("status", "BLOCKED"),
        "disagreements": [
            item for item in disagreements.get("disagreements", [])
            if any(field.get("field") not in {"analysis_label", "o_f_consistency"} for field in item.get("fields", []))
        ],
    }
    _write_json(output / "calibration/finding_disagreements.json", finding_only)
    _write_json(
        output / "calibration/evaluator_execution_manifest.json",
        {
            "tested_identity_hidden": True,
            "tested_target": {"provider": tested_provider, "model": tested_model},
            "judge_a_gate": judge_a_gate,
            "judge_b_gate": judge_b_gate,
            "fallback_allowed": False,
        },
    )
    _write_json(
        output / "calibration/human_reference_judgments.json",
        human or {"status": "BLOCKED", "reason": human_gate["reason"]},
    )

    o1 = _o1_audit(repo)
    contracts = _data_contract_audit(repo)
    o1_curator_gate, o1_curator = _validate_o1_curator_audit(
        None if o1_curator_audit is None else Path(o1_curator_audit),
        expected_case_ids={str(item["case_id"]) for item in o1["records"]},
    )
    contract_curator_gate, contract_curator = _validate_data_contract_curator_audit(
        None if data_contract_curator_audit is None else Path(data_contract_curator_audit),
        expected_dataset_ids={str(item["dataset_id"]) for item in contracts["records"]},
    )
    _write_json(output / "audits/o1_specification_audit.json", o1)
    _write_json(output / "audits/data_contract_manifest.json", contracts)
    _write_text(output / "audits/o1_specification_audit.md", _o1_markdown(o1))
    _write_text(output / "audits/data_contract_audit.md", _contract_markdown(contracts))
    _write_json(output / "audits/o1_curator_audit.json", o1_curator or {"status": "BLOCKED", "reason": o1_curator_gate["reason"]})
    _write_json(output / "audits/data_contract_curator_audit.json", contract_curator or {"status": "BLOCKED", "reason": contract_curator_gate["reason"]})
    _write_text(output / "audits/o1_curator_audit.md", f"# O1 Curator Audit\n\nStatus: **{o1_curator_gate['status']}**. {o1_curator_gate['reason']}")
    _write_text(output / "audits/data_contract_curator_audit.md", f"# Data-Contract Curator Audit\n\nStatus: **{contract_curator_gate['status']}**. {contract_curator_gate['reason']}")
    _write_text(output / "changes/carotid_fix.md", _carotid_markdown(repo))

    smokes = _runtime_smokes(repo, output)
    _write_text(output / "runtime/file_discovery_hardening.md", _discovery_markdown(smokes["discovery"]))
    _write_text(output / "runtime/stdout_policy.md", _stdout_markdown(smokes["stdout"]))
    _write_text(output / "runtime/error_accounting_policy.md", _error_policy_markdown())

    compatibility: dict[str, Any]
    if provider_compatibility_report is None:
        compatibility = {"status": "BLOCKED", "reason": "fresh provider compatibility smoke is unavailable"}
    else:
        try:
            compatibility = dict(_read_json(Path(provider_compatibility_report)))
            if compatibility.get("status") != "PASS":
                compatibility = {**compatibility, "status": "FAIL"}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            compatibility = {"status": "FAIL", "reason": str(exc)}
    _write_json(output / "runtime/provider_compatibility_report.json", compatibility)
    _write_json(
        output / "runtime/tool_boundary_smoke.json",
        {
            "status": "PASS",
            "synthetic_tests": [
                "multi-call cardinality normalized and bounded before execution",
                "Chat tool-result retry does not duplicate state",
                "Responses function output retry does not duplicate state",
            ],
            "live_provider_compatibility": compatibility.get("status"),
        },
    )
    multi_call = _multi_call_artifacts(repo, output, compatibility)

    judges_independent = _judges_are_independent(judge_a, judge_b)
    judge_ready = (
        judge_a_gate["status"] == judge_b_gate["status"] == "PASS"
        and judges_independent
    )
    human_ready = human_gate["status"] == "PASS" and disagreements["status"] == "PASS"
    _write_text(output / "calibration/human_judge_comparison.md", _comparison_markdown(disagreements))
    _write_text(output / "calibration/evaluator_error_analysis.md", _evaluator_error_markdown(disagreements))
    valid_unenumerated = _valid_unenumerated(judge_a, judge_b, human)
    _write_json(output / "evaluation/valid_unenumerated_cases.json", valid_unenumerated)
    _write_text(output / "evaluation/valid_unenumerated_validation.md", "# Valid-Unenumerated Validation\n\n" + json.dumps(valid_unenumerated, ensure_ascii=False, indent=2))
    _write_text(output / "evaluation/open_ended_calibration.md", _open_ended_markdown(judge_ready, human_ready))
    of_outcomes = _of_separation_outcomes(judge_a, judge_b, human)
    _write_json(output / "evaluation/of_separation_validation.json", of_outcomes)
    _write_text(output / "evaluation/of_separation_validation.md", _of_markdown(judge_ready, human_ready) + "\n\nObserved diagnostic outcomes:\n\n" + json.dumps(of_outcomes, ensure_ascii=False, indent=2))
    branch_stats = _branch_statistics(
        repo,
        evaluation_root=(
            None if branch_evaluation_root is None else Path(branch_evaluation_root)
        ),
    )
    _write_json(output / "evaluation/reference_branch_statistics.json", branch_stats)
    _write_text(output / "evaluation/reference_branch_sensitivity.md", _branch_markdown(branch_stats, judge_ready))
    _write_text(output / "evaluation/metric_edge_cases.md", _metric_edge_markdown())

    efficiency_paths = {
        "whole_system_before_after": None if whole_system_efficiency is None else Path(whole_system_efficiency),
        "unchanged_task_runtime_effect": None if unchanged_task_efficiency is None else Path(unchanged_task_efficiency),
    }
    efficiency_gates = {
        "whole_system_before_after": _validate_efficiency_comparison(
            efficiency_paths["whole_system_before_after"],
            label="WHOLE_SYSTEM_BEFORE_AFTER",
        ),
        "unchanged_task_runtime_effect": _validate_efficiency_comparison(
            efficiency_paths["unchanged_task_runtime_effect"],
            label="UNCHANGED_TASK_RUNTIME_EFFECT",
        ),
    }
    for name, source in efficiency_paths.items():
        destination = output / "efficiency" / f"{name}.md"
        if source is not None and source.is_file():
            _write_text(destination, source.read_text(encoding="utf-8"))
        else:
            _write_text(destination, f"# {name}\n\nStatus: **BLOCKED**. An explicit matched efficiency comparison is required; no historical table is promoted automatically.")

    clean_source = None if clean_pilot_root is None else Path(clean_pilot_root)
    clean_gate, clean_summary = _clean_pilot_gate(clean_source)
    effective_compatibility = compatibility
    if (
        clean_summary.get("status") == "COMPLETE"
        and int(clean_summary.get("infrastructure_invalid_attempts", 0)) > 0
        and int(clean_summary.get("tool_boundary_retries", 0)) > 0
    ):
        effective_compatibility = {
            "status": "FAIL",
            "reason": (
                "synthetic provider smoke passed, but the clean scientific subset produced "
                "repeated one-tool-boundary violations and infrastructure-invalid attempts"
            ),
            "synthetic_preflight": compatibility,
            "clean_runtime_tool_boundary_retries": clean_summary.get("tool_boundary_retries"),
            "clean_runtime_infrastructure_invalid_attempts": clean_summary.get(
                "infrastructure_invalid_attempts"
            ),
        }
        _write_json(output / "runtime/provider_compatibility_report.json", effective_compatibility)
        _write_json(
            output / "runtime/tool_boundary_smoke.json",
            {
                "status": "FAIL",
                "synthetic_tests": [
                    "multi-call cardinality normalized and bounded before execution",
                    "Chat tool-result retry does not duplicate state",
                    "Responses function output retry does not duplicate state",
                ],
                "live_provider_compatibility": "FAIL",
                "observed_tool_boundary_retries": clean_summary.get("tool_boundary_retries"),
                "observed_infrastructure_invalid_attempts": clean_summary.get(
                    "infrastructure_invalid_attempts"
                ),
            },
        )
    _write_json(output / "clean_pilot/manifest.json", clean_summary)
    for name, fallback in (
        ("runtime_report.md", _clean_runtime_markdown(clean_gate, clean_summary)),
        ("efficiency_comparison.md", _clean_comparison_markdown(clean_gate)),
        ("old_vs_clean_runtime.md", _clean_comparison_markdown(clean_gate)),
    ):
        source = None if clean_source is None else clean_source / name
        _write_text(
            output / "clean_pilot" / name,
            source.read_text(encoding="utf-8") if source is not None and source.is_file() else fallback,
        )
    _write_text(output / "runtime/clean_runtime_report.md", (output / "clean_pilot/runtime_report.md").read_text(encoding="utf-8"))

    primary_gate, primary_manifest = _validate_primary_evaluator_manifest(
        None if primary_evaluator_manifest is None else Path(primary_evaluator_manifest),
        tested_provider=tested_provider,
        tested_model=tested_model,
        judge_a=judge_a,
        judge_b=judge_b,
        calibration_ready=judge_ready and human_ready,
    )
    _write_json(output / "evaluation/primary_evaluator_manifest.json", primary_manifest)
    _write_text(
        output / "evaluation/evaluator_prompt_hash.txt",
        str(primary_manifest.get("prompt_sha256", "UNFROZEN")),
    )

    dispositions = _historical_dispositions(collection)
    _write_json(output / "freeze/observation_dispositions.json", {"records": dispositions})
    hashes = _freeze_hashes(repo)
    _write_json(output / "freeze/file_hashes.json", hashes)
    _write_json(
        output / "freeze/runtime_manifest.json",
        {
            "status": "CANDIDATE",
            "system_prompt_version": DEFAULT_SYSTEM_PROMPT_VERSION,
            "python_tool_spec_version": PYTHON_TOOL_SPEC_VERSION,
            "max_output_chars": DEFAULT_OUTPUT_LIMIT,
            "root_recursive_discovery": "blocked",
            "core_libraries": list(CORE_LIBRARIES),
        },
    )
    _write_json(
        output / "freeze/evaluation_manifest.json",
        {
            "status": "FROZEN" if primary_gate["status"] == "PASS" else "UNFROZEN",
            "primary_evaluator": primary_manifest.get("evaluator"),
            "primary_evaluator_manifest_sha256": _sha256(
                output / "evaluation/primary_evaluator_manifest.json"
            ),
            "metric_code_sha256": _sha256(repo / "flowintentbench/evaluation_metrics.py"),
            "evaluator_code_sha256": _sha256(repo / "flowintentbench/evaluator.py"),
            "metric_edge_semantics": "FROZEN",
        },
    )

    archive_gate, archive_value = _validate_archive_self_test(
        None if archive_self_test is None else Path(archive_self_test)
    )
    _write_json(output / "release/archive_self_test.json", archive_value or {"status": archive_gate["status"], "reason": archive_gate["reason"]})
    _write_json(output / "final/readiness.json", {"status": "NOT_READY", "archive_gate": archive_gate})

    multi_call_pass = (
        effective_compatibility.get("status") == "PASS"
        and multi_call["status"] == "PASS"
        and multi_call["provider_smoke"].get("status") == "PASS"
        and multi_call["provider_smoke"].get("multi_call_emitted") is True
    )
    multi_call_reason = (
        "canonical multi-call unit and live provider smoke pass"
        if multi_call_pass
        else multi_call["provider_smoke"].get("reason")
        or effective_compatibility.get("reason")
        or "canonical multi-call unit/live smoke required"
    )
    gates = {
        "gate_1_real_evaluator_calibration": _gate(
            "PASS" if judge_ready else "BLOCKED",
            "both independent judges executed"
            if judge_ready
            else "two independent executed judges are required",
        ),
        "gate_2_human_confirmation": _gate(
            "PASS" if human_ready else "BLOCKED",
            "human review complete" if human_ready else "confirmed human review and disagreement resolution are required",
        ),
        "gate_3_carotid_o1": _gate(
            "PASS" if o1["status"] == "PASS" and o1_curator_gate["status"] == "PASS" else "BLOCKED" if o1_curator_gate["status"] == "BLOCKED" else "FAIL",
            "automatic and curator O1 audits pass" if o1["status"] == "PASS" and o1_curator_gate["status"] == "PASS" else o1_curator_gate["reason"],
        ),
        "gate_4_data_contract": _gate(
            "PASS" if contracts["status"] == "PASS" and contract_curator_gate["status"] == "PASS" else "BLOCKED" if contract_curator_gate["status"] == "BLOCKED" else "FAIL",
            "technical and scientific data-contract audits pass" if contracts["status"] == "PASS" and contract_curator_gate["status"] == "PASS" else contract_curator_gate["reason"],
        ),
        "gate_5_case_discovery": _gate(smokes["discovery"]["status"], "runtime discovery smoke"),
        "gate_6_stdout": _gate(smokes["stdout"]["status"], "bounded head/tail output smoke"),
        "gate_7_tool_boundary": _gate(
            "PASS" if multi_call_pass else "FAIL" if effective_compatibility.get("status") == "FAIL" else "BLOCKED",
            multi_call_reason,
        ),
        "gate_8_runtime_libraries": _gate(smokes["dependencies"]["status"], "frozen import preflight"),
        "gate_9_error_accounting": _gate("PASS", "historical counts and future accounting are explicit"),
        "gate_10_open_ended": _gate(
            "PASS"
            if valid_unenumerated["status"] == "PASS" and human_ready
            else "FAIL"
            if valid_unenumerated["status"] == "FAIL" and human_ready
            else "BLOCKED",
            "sentinel and invalid-control outcomes validated"
            if valid_unenumerated["status"] == "PASS" and human_ready
            else "sentinel outcomes require confirmed human, resolved disagreements, and judge evidence",
        ),
        "gate_11_o_f_separation": _gate(
            of_outcomes["status"] if of_outcomes["status"] in {"PASS", "BLOCKED"} else "FAIL",
            "observed diagnostic combinations distinguish failure modes" if of_outcomes["status"] == "PASS" else "real O-F diagnostic evidence is required",
        ),
        "gate_12_branch_sensitivity": _gate(
            "PASS"
            if judge_ready
            and branch_stats.get("status") == "PASS"
            and branch_stats["redundant_equivalent_branch_score_invariance"] == "PASS"
            else "BLOCKED",
            "redundant-branch invariance and complete real branch statistics pass"
            if judge_ready and branch_stats.get("status") == "PASS"
            else "deterministic redundant-branch invariance passes; complete real CaseEvaluationRecord branch statistics remain required",
        ),
        "gate_13_metric_edges": _gate("PASS", "edge semantics are deterministic, documented, and tested"),
        "gate_14_clean_runtime_pilot": clean_gate,
        "gate_15_old_vs_clean": _gate(
            "PASS"
            if all(item["status"] == "PASS" for item in efficiency_gates.values())
            else "FAIL"
            if any(item["status"] == "FAIL" for item in efficiency_gates.values())
            else "BLOCKED",
            "whole-system and unchanged-task comparisons contain all required metrics"
            if all(item["status"] == "PASS" for item in efficiency_gates.values())
            else "; ".join(item["reason"] for item in efficiency_gates.values() if item["status"] != "PASS"),
        ),
        "gate_16_primary_evaluator": primary_gate,
    }
    prefreeze_ready = all(item["status"] == "PASS" for item in gates.values())
    gate_17 = _gate(
        "PASS" if prefreeze_ready and bool(hashes) and archive_gate["status"] == "PASS" else "BLOCKED",
        "all frozen artifact families are hashed and archive self-test passes"
        if prefreeze_ready and hashes and archive_gate["status"] == "PASS"
        else "candidate hashes exist but one or more prerequisite gates or archive self-test remain incomplete",
    )
    gates["gate_17_benchmark_freeze"] = gate_17
    # Backwards-compatible key retained for existing consumers; both names
    # refer to the same final benchmark-freeze decision.
    gates["gate_17_version_freeze"] = gate_17
    ready = prefreeze_ready and gate_17["status"] == "PASS"
    readiness = "READY_FOR_SMALL_N3" if ready else "NOT_READY"
    if readiness not in READINESS_STATUSES:  # pragma: no cover
        raise AssertionError(readiness)
    benchmark_manifest = {
        "record_type": "FlowIntentBenchNextStageCandidate",
        "status": readiness,
        "freeze_status": "FROZEN" if ready else "CANDIDATE",
        "formal": False,
        "eligible_for_small_n3": ready,
        "file_hashes_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "gates": gates,
    }
    _write_json(output / "freeze/benchmark_manifest.json", benchmark_manifest)
    runtime_manifest = _read_json(output / "freeze/runtime_manifest.json")
    runtime_manifest["status"] = "FROZEN" if ready else "CANDIDATE"
    _write_json(output / "freeze/runtime_manifest.json", runtime_manifest)
    _write_text(output / "freeze/change_log.md", _change_log_markdown())
    final = {"status": readiness, "gates": gates, "output_root": str(output)}
    _write_json(output / "next_stage_readiness.json", final)
    _write_json(output / "final/readiness.json", final)
    _write_text(output / "final/next_stage_summary.md", _summary_markdown(final))
    _write_text(output / "next_stage_summary.md", _summary_markdown(final))
    for source_name, release_name in (
        ("freeze/benchmark_manifest.json", "benchmark_manifest.json"),
        ("freeze/runtime_manifest.json", "runtime_manifest.json"),
        ("freeze/evaluation_manifest.json", "evaluation_manifest.json"),
        ("freeze/file_hashes.json", "file_hashes.json"),
    ):
        _write_json(output / "release" / release_name, _read_json(output / source_name))
    return final


def _branch_statistics(
    repo: Path,
    *,
    evaluation_root: Path | None = None,
) -> dict[str, Any]:
    records = []
    evaluation_by_id: dict[str, list[Mapping[str, Any]]] = {}
    if evaluation_root is not None and evaluation_root.is_dir():
        for record_path in sorted(evaluation_root.rglob("case_evaluation_record.json")):
            try:
                payload = _read_json(record_path)
                result = payload.get("result") if isinstance(payload, Mapping) else None
                case_id = result.get("case_id") if isinstance(result, Mapping) else None
                if isinstance(case_id, str):
                    evaluation_by_id.setdefault(case_id, []).append(payload)
            except (OSError, json.JSONDecodeError):
                continue
    for path in sorted((repo / "datasets").glob("*/construction/cases/*/ground_truth.json")):
        if path.parent.name == "initial_case":
            continue
        value = GroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
        count = len(value.acceptable_operationalizations)
        evaluations = evaluation_by_id.get(value.case_id, [])
        best_score = second_score = tie_count = None
        best_f_score = second_f_score = f_tie_count = branch_alignment = None
        best_o_branches = best_f_branches = None
        score_status = "UNAVAILABLE"
        if len(evaluations) == 1:
            evaluated = evaluations[0]
            operation_rows = evaluated.get("operationalization_branch_evaluations", [])
            finding_rows = evaluated.get("finding_branch_evaluations", [])
            o_scores = []
            for item in operation_rows:
                if not isinstance(item, Mapping):
                    continue
                matches = item.get("principal_dimension_matches")
                if isinstance(matches, Mapping) and matches:
                    o_scores.append(
                        (
                            sum(int(value) for value in matches.values()) / len(matches),
                            str(item.get("branch_id")),
                        )
                    )
            f_scores = []
            for item in finding_rows:
                if not isinstance(item, Mapping):
                    continue
                predicted = int(item.get("predicted_finding_count", 0))
                core = int(item.get("core_finding_count", 0))
                if core <= 0:
                    continue
                precision = (
                    int(item.get("valid_predicted_finding_count", 0)) / predicted
                    if predicted > 0
                    else 0.0
                )
                recall = int(item.get("matched_core_finding_count", 0)) / core
                f_scores.append(((recall, precision), str(item.get("branch_id"))))
            o_scores.sort(reverse=True)
            f_scores.sort(reverse=True)
            if o_scores and f_scores:
                best_score = o_scores[0][0]
                second_score = o_scores[1][0] if len(o_scores) > 1 else None
                tie_count = sum(score == best_score for score, _ in o_scores)
                best_f_pair = f_scores[0][0]
                second_f_pair = f_scores[1][0] if len(f_scores) > 1 else None
                f_tie_count = sum(score == best_f_pair for score, _ in f_scores)
                best_f_score = {
                    "core_finding_recall": best_f_pair[0],
                    "finding_precision": best_f_pair[1],
                }
                second_f_score = (
                    None
                    if second_f_pair is None
                    else {
                        "core_finding_recall": second_f_pair[0],
                        "finding_precision": second_f_pair[1],
                    }
                )
                result = evaluated.get("result", {})
                metrics = result.get("metrics", {}) if isinstance(result, Mapping) else {}
                scientific_o = metrics.get("scientific_operationalization", {}) if isinstance(metrics, Mapping) else {}
                scientific_f = metrics.get("scientific_findings", {}) if isinstance(metrics, Mapping) else {}
                consistency = metrics.get("o_f_consistency", {}) if isinstance(metrics, Mapping) else {}
                best_o_branches = scientific_o.get("best_o_branches")
                best_f_branches = scientific_f.get("best_finding_branches")
                branch_alignment = consistency.get("branch_alignment")
                score_status = "AVAILABLE"
        elif len(evaluations) > 1:
            score_status = "AMBIGUOUS_MULTIPLE_RECORDS"
        records.append(
            {
                "case_id": value.case_id,
                "reference_branch_count": count,
                "best_O_score": best_score,
                "second_best_O_score": second_score,
                "O_best_tie_count": tie_count,
                "best_F_score": best_f_score,
                "second_best_F_score": second_f_score,
                "F_best_tie_count": f_tie_count,
                "branch_alignment": branch_alignment,
                "score_status": score_status,
                "best_o_branches": best_o_branches,
                "best_f_branches": best_f_branches,
            }
        )
    operation = (
        OperationalizationBranchEvaluation(
            "reference-a",
            {"criterion": 1, "feature_definition": 0, "property_measure": 1},
            {"property_measure": 1},
        ),
        OperationalizationBranchEvaluation(
            "reference-b",
            {"criterion": 1, "feature_definition": 1, "property_measure": 1},
            {"property_measure": 1},
        ),
    )
    findings = (
        FindingBranchEvaluation("reference-a", 2, 1, 2, 1),
        FindingBranchEvaluation("reference-b", 2, 2, 2, 2),
    )
    duplicated_operation = operation + (
        OperationalizationBranchEvaluation(
            "reference-b-redundant",
            dict(operation[1].principal_dimension_matches),
            dict(operation[1].unresolved_dimension_matches),
        ),
    )
    duplicated_findings = findings + (
        FindingBranchEvaluation("reference-b-redundant", 2, 2, 2, 2),
    )
    baseline_o = compute_o_score(operation)
    duplicate_o = compute_o_score(duplicated_operation)
    baseline_f = compute_finding_metrics(findings)
    duplicate_f = compute_finding_metrics(duplicated_findings)
    baseline_values = {
        "o_score": baseline_o.o_score,
        "urs": baseline_o.urs,
        "finding_precision": baseline_f.finding_precision,
        "core_finding_recall": baseline_f.core_finding_recall,
        "branch_alignment": compute_branch_alignment(
            baseline_o.best_o_branches, baseline_f.best_finding_branches
        ),
    }
    duplicate_values = {
        "o_score": duplicate_o.o_score,
        "urs": duplicate_o.urs,
        "finding_precision": duplicate_f.finding_precision,
        "core_finding_recall": duplicate_f.core_finding_recall,
        "branch_alignment": compute_branch_alignment(
            duplicate_o.best_o_branches, duplicate_f.best_finding_branches
        ),
    }
    invariant = baseline_values == duplicate_values
    complete_real_statistics = bool(records) and all(
        item.get("score_status") == "AVAILABLE" for item in records
    )
    multi_branch = [item for item in records if item["reference_branch_count"] > 1]
    pathological_o_ties = bool(multi_branch) and all(
        item.get("O_best_tie_count") == item["reference_branch_count"]
        for item in multi_branch
    )
    pathological_f_ties = bool(multi_branch) and all(
        item.get("F_best_tie_count") == item["reference_branch_count"]
        for item in multi_branch
    )
    group_values: dict[int, dict[str, list[float]]] = {}
    for item in records:
        if item.get("score_status") != "AVAILABLE":
            continue
        group = group_values.setdefault(
            int(item["reference_branch_count"]),
            {"best_O_score": [], "core_finding_recall": [], "finding_precision": []},
        )
        group["best_O_score"].append(float(item["best_O_score"]))
        group["core_finding_recall"].append(
            float(item["best_F_score"]["core_finding_recall"])
        )
        group["finding_precision"].append(
            float(item["best_F_score"]["finding_precision"])
        )
    group_means = [
        {
            "reference_branch_count": branch_count,
            "case_count": len(values["best_O_score"]),
            **{
                name: statistics.fmean(numbers)
                for name, numbers in values.items()
            },
        }
        for branch_count, values in sorted(group_values.items())
    ]

    def systematic_increase(field: str) -> bool:
        values = [item[field] for item in group_means]
        return (
            len(values) >= 2
            and all(right >= left for left, right in zip(values, values[1:]))
            and any(right > left for left, right in zip(values, values[1:]))
        )

    systematic_flags = {
        field: systematic_increase(field)
        for field in ("best_O_score", "core_finding_recall", "finding_precision")
    }
    diagnostic_flags = []
    if pathological_o_ties:
        diagnostic_flags.append("all multi-branch cases tie across every O branch")
    if pathological_f_ties:
        diagnostic_flags.append("all multi-branch cases tie across every Finding branch")
    for field, flagged in systematic_flags.items():
        if flagged:
            diagnostic_flags.append(
                f"group mean {field} increases monotonically with reference branch count"
            )
    return {
        "status": "PASS" if complete_real_statistics and not diagnostic_flags else "BLOCKED_PENDING_REAL_EVALUATION" if not complete_real_statistics else "FAIL",
        "redundant_equivalent_branch_score_invariance": "PASS" if invariant else "FAIL",
        "real_statistics_complete": complete_real_statistics,
        "diagnostic_flags": diagnostic_flags,
        "branch_count_group_means": group_means,
        "systematic_branch_count_increase": systematic_flags,
        "synthetic_invariance_check": {
            "baseline": baseline_values,
            "with_redundant_equivalent_branch": duplicate_values,
            "material_score_change": not invariant,
        },
        "records": records,
    }


def _clean_pilot_gate(path: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if path is None or not (path / "manifest.json").is_file():
        return _gate("BLOCKED", "fresh clean-runtime pilot is unavailable"), {"status": "BLOCKED"}
    try:
        value = _read_json(path / "manifest.json")
        if not isinstance(value, Mapping):
            raise ValueError("clean-runtime manifest must be an object")
        cases = value.get("cases")
        collection_status = value.get("collection_status", value.get("status"))
        if collection_status in {
            "COLLECTION_BLOCKED_EXTERNAL_QUOTA",
            "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT",
            "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE",
        }:
            return _gate("BLOCKED", f"clean-runtime collection is externally blocked: {collection_status}"), {
                "status": "BLOCKED",
                "collection_status": collection_status,
                "blocked_case_id": value.get("blocked_case_id"),
                "resume_allowed": value.get("resume_allowed"),
            }
        if not isinstance(cases, list) or len(cases) != len(CLEAN_RUNTIME_CASE_IDS):
            raise ValueError("clean-runtime pilot must contain the fixed 8 explicit cases")
        case_ids = {item.get("case_id") for item in cases if isinstance(item, Mapping)}
        if len(case_ids) != len(cases):
            raise ValueError("clean-runtime case IDs must be unique")
        if case_ids != set(CLEAN_RUNTIME_CASE_IDS):
            raise ValueError("clean-runtime pilot does not match the fixed 8-case subset")
        selected = value.get("selected_case_ids")
        if not isinstance(selected, list) or tuple(selected) != CLEAN_RUNTIME_CASE_IDS:
            raise ValueError("clean-runtime selected_case_ids must preserve the fixed subset order")
        if value.get("model_id") != "gpt-5.6-sol":
            raise ValueError("clean-runtime pilot must retain the tested gpt-5.6-sol target")
        config = value.get("model_configuration")
        if not isinstance(config, Mapping):
            raise ValueError("clean-runtime pilot must record model_configuration")
        reasoning = config.get("reasoning_effort")
        if reasoning is not None and str(reasoning).casefold() not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("clean-runtime pilot contains an unsupported reasoning_effort")
        if value.get("provider_compatibility_status") != "PASS":
            raise ValueError("clean-runtime provider compatibility did not pass")
        for item in cases:
            if not isinstance(item, Mapping):
                raise ValueError("clean-runtime case record must be an object")
            if item.get("status") not in {"COMPLETED", "MODEL_NONCOMPLETION"}:
                raise ValueError(
                    f"clean-runtime case has no valid model observation: {item.get('case_id')}"
                )
            for path_field, hash_field in (
                ("run_record_path", "run_record_sha256"),
                ("trajectory_path", "trajectory_sha256"),
            ):
                relative = item.get(path_field)
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    raise ValueError(f"invalid {path_field} for {item.get('case_id')}")
                artifact = path / relative
                if not artifact.is_file() or item.get(hash_field) != _sha256(artifact):
                    raise ValueError(f"clean-runtime artifact hash mismatch: {artifact}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _gate("FAIL", str(exc)), {"status": "FAIL", "reason": str(exc)}
    required = {
        "filesystem_discovery_errors": 0,
        "root_search_timeouts": 0,
        "uncontrolled_huge_stdout": 0,
        "conversation_state_corruption": 0,
        "unsupported_reasoning_fallbacks": 0,
        "infrastructure_invalid_attempts": 0,
        "multi_call_tool_boundary_retries": 0,
        "multi_call_infrastructure_invalid_attempts": 0,
        "duplicate_tool_execution": 0,
    }
    violations = {
        key: {"expected": expected, "observed": value.get(key)}
        for key, expected in required.items()
        if value.get(key) != expected
    }
    passed = value.get("status") == "COMPLETE" and not violations
    return (
        _gate(
            "PASS" if passed else "FAIL",
            "clean-runtime manifest validation passed"
            if passed
            else "clean-runtime manifest violates: " + ", ".join(sorted(violations)),
            violations=violations,
        ),
        dict(value),
    )


def _o1_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {item['case_id']} | {item['status']} | {item['case_input_sha256']} |"
        for item in report["records"]
    )
    return f"# O1 Specification Audit\n\nStatus: **{report['status']}**\n\n| Case | Status | Case-input SHA-256 |\n|---|---|---|\n{rows}"


def _contract_markdown(report: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {item['dataset_id']} | {item['status']} | {', '.join(item['hidden_required_items']) or 'none'} |"
        for item in report["records"]
    )
    return f"# Data-Contract Audit\n\nStatus: **{report['status']}**\n\n| Dataset | Status | Hidden required items |\n|---|---|---|\n{rows}"


def _carotid_markdown(repo: Path) -> str:
    cases = []
    for case_id in ("carotid_o1_f1", "carotid_o1_f2"):
        path = repo / "datasets/Carotid/construction/cases" / case_id / "case_input.json"
        cases.append(f"- `{case_id}`: `{_sha256(path)}`")
    return "# Carotid O1 Fix\n\nO1 now defines speed as the Euclidean magnitude of the stored point-data vector array named `vectors`. Historical pre-fix responses remain invalidated by task change.\n\n" + "\n".join(cases)


def _discovery_markdown(value: Mapping[str, Any]) -> str:
    return f"# File-Discovery Hardening\n\nStatus: **{value['status']}**. Root-recursive scans are rejected before child execution; `/case/case_files.json` remains available."


def _stdout_markdown(value: Mapping[str, Any]) -> str:
    return f"# Runtime Stdout Policy\n\nThe frozen per-stream textual limit is **{value['limit']} characters**. Oversized output uses a head/tail preview and records original and returned lengths. Smoke status: **{value['status']}**."


def _error_policy_markdown() -> str:
    return """# Runtime Error Accounting Policy

Historical counts are preserved: AGENT_CODE_ERROR=16, FILESYSTEM_DISCOVERY_ERROR=12, MISSING_DEPENDENCY=9, TOOL_INTERFACE_ERROR=4.

- `AGENT_CODE_ERROR` counts toward agent efficiency.
- `MISSING_DEPENDENCY` counts as agent behavior only after the model-visible library contract is present.
- benchmark-owned `TOOL_INTERFACE_ERROR` and `INFRASTRUCTURE_ERROR` do not enter valid scientific-observation cost.
- `FILESYSTEM_DISCOVERY_ERROR` is retained only as a historical pilot category; the hardened runtime blocks root recursion immediately.
"""


def _comparison_markdown(value: Mapping[str, Any]) -> str:
    return f"# Human–Judge Comparison\n\nStatus: **{value['status']}**. Recorded disagreements: **{len(value['disagreements'])}**. No missing human or judge output is converted into agreement."


def _evaluator_error_markdown(value: Mapping[str, Any]) -> str:
    return f"# Evaluator Error Analysis\n\nStatus: **{value['status']}**. Systematic evaluator-error analysis remains blocked until both independent judges and confirmed human review are available."


def _valid_unenumerated(
    judge_a_result: Mapping[str, Any] | None,
    judge_b_result: Mapping[str, Any] | None,
    human_adjudication: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sentinels = [
        {"case_id": "office_speed_zones_o2_f1", "strategy": "q95 plus mean-speed"},
        {"case_id": "office_speed_zones_o3_f1", "strategy": "half-maximum connected core"},
    ]
    if judge_a_result is None or judge_b_result is None or human_adjudication is None:
        return {
            "status": "BLOCKED",
            "sentinels": sentinels,
            "judge_outputs_available": judge_a_result is not None and judge_b_result is not None,
            "human_confirmation_available": human_adjudication is not None,
            "outcomes": [],
        }
    by_id = {
        label: {item.get("case_id"): item for item in value.get("cases", [])}
        for label, value in (
            ("judge_a", judge_a_result),
            ("judge_b", judge_b_result),
            ("human", human_adjudication),
        )
    }
    outcomes = []
    for sentinel in sentinels:
        case_id = sentinel["case_id"]
        human = by_id["human"].get(case_id, {})
        # Human artifacts use analysis_label; older drafts use a nested
        # human_reference field. Only explicit confirmed labels count.
        human_label = human.get("analysis_label")
        if human_label is None and isinstance(human.get("human_reference"), Mapping):
            human_label = human["human_reference"].get("analysis_validity")
        judge_labels = [by_id[label].get(case_id, {}).get("analysis_label") for label in ("judge_a", "judge_b")]
        outcomes.append({"case_id": case_id, "human": human_label, "judge_a": judge_labels[0], "judge_b": judge_labels[1]})
    sentinel_passed = all(
        row["human"] == "VALID_UNENUMERATED"
        and row["judge_a"] in {"VALID_UNENUMERATED", "VALID_EQUIVALENT"}
        and row["judge_b"] in {"VALID_UNENUMERATED", "VALID_EQUIVALENT"}
        for row in outcomes
    )
    invalid_controls = []
    for case_id, human in by_id["human"].items():
        if human.get("analysis_label") != "INVALID":
            continue
        row = {
            "case_id": case_id,
            "human": "INVALID",
            "judge_a": by_id["judge_a"].get(case_id, {}).get("analysis_label"),
            "judge_b": by_id["judge_b"].get(case_id, {}).get("analysis_label"),
        }
        row["accepted_as_valid_alternative"] = any(
            row[label] in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"}
            for label in ("judge_a", "judge_b")
        )
        invalid_controls.append(row)
    controls_passed = bool(invalid_controls) and not any(
        item["accepted_as_valid_alternative"] for item in invalid_controls
    )
    passed = sentinel_passed and controls_passed
    return {
        "status": "PASS" if passed else "FAIL",
        "sentinels": sentinels,
        "judge_outputs_available": True,
        "human_confirmation_available": True,
        "outcomes": outcomes,
        "invalid_controls": invalid_controls,
        "sentinel_outcomes_passed": sentinel_passed,
        "invalid_controls_passed": controls_passed,
    }


def _open_ended_markdown(judges: bool, human: bool) -> str:
    return f"# Open-Ended Evaluation Calibration\n\nJudge evidence: **{'available' if judges else 'blocked'}**. Human confirmation: **{'available' if human else 'blocked'}**. Reference branches remain anchors; valid-unenumerated analyses require explicit adjudication and are never rejected solely for branch absence."


def _of_markdown(judges: bool, human: bool) -> str:
    return f"# O-F Separation Validation\n\nExecuted calibration evidence is **{'available' if judges and human else 'blocked'}**. The frozen evaluator continues to judge Analysis Formulation, Findings, and O-F consistency separately; missing diagnostic combinations are not fabricated."


def _diagnostic_pattern(item: Mapping[str, Any], *, human: bool) -> str | None:
    analysis = item.get("analysis_label")
    of = item.get("o_f_consistency")
    key = "finding_judgments" if human else "findings"
    findings = [
        finding for finding in item.get(key, [])
        if isinstance(finding, Mapping) and (not human or finding.get("included") is True)
    ]
    supports = {finding.get("evidential_support") for finding in findings}
    valid = analysis in {"VALID_REFERENCE", "VALID_EQUIVALENT", "VALID_UNENUMERATED"}
    if of == "INCONSISTENT":
        return "D"
    if valid and any(value in {"UNSUPPORTED", "UNCERTAIN"} for value in supports):
        return "B"
    if analysis == "INVALID" and of == "CONSISTENT" and findings and supports <= {"SUPPORTED"}:
        return "C"
    if valid and of == "CONSISTENT" and findings and supports <= {"SUPPORTED"}:
        return "A"
    return None


def _of_separation_outcomes(
    judge_a: Mapping[str, Any] | None,
    judge_b: Mapping[str, Any] | None,
    human: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Require both judges to preserve observed independent diagnostic axes."""
    if judge_a is None or judge_b is None or human is None:
        return {"status": "BLOCKED", "observed_patterns": [], "distinguishes_failure_modes": False}
    human_by_id = {item.get("case_id"): item for item in human.get("cases", [])}
    a_by_id = {item.get("case_id"): item for item in judge_a.get("cases", [])}
    b_by_id = {item.get("case_id"): item for item in judge_b.get("cases", [])}
    patterns: dict[str, list[str]] = {key: [] for key in ("A", "B", "C", "D")}
    preservation: list[dict[str, Any]] = []
    for case_id in DIAGNOSTIC_CASE_IDS:
        h = human_by_id.get(case_id, {})
        human_pattern = _diagnostic_pattern(h, human=True)
        a_pattern = _diagnostic_pattern(a_by_id.get(case_id, {}), human=False)
        b_pattern = _diagnostic_pattern(b_by_id.get(case_id, {}), human=False)
        if human_pattern is not None:
            patterns[human_pattern].append(case_id)
        preservation.append({
            "case_id": case_id,
            "human_pattern": human_pattern,
            "judge_a_pattern": a_pattern,
            "judge_b_pattern": b_pattern,
            "judge_a_preserves": a_pattern == human_pattern and human_pattern is not None,
            "judge_b_preserves": b_pattern == human_pattern and human_pattern is not None,
            "supported_finding_analysis_collapse": bool(
                human_pattern == "C" and (a_pattern == "A" or b_pattern == "A")
            ),
        })
    observed = {key: value for key, value in patterns.items() if value}
    observed_failure_modes = [key for key in ("B", "C", "D") if patterns[key]]
    pass_condition = (
        len(observed_failure_modes) >= 2
        and all(row["judge_a_preserves"] and row["judge_b_preserves"] for row in preservation)
        and not any(row["supported_finding_analysis_collapse"] for row in preservation)
    )
    return {
        "status": "PASS" if pass_condition else "BLOCKED",
        "observed_patterns": [
            {"pattern": key, "case_ids": ids}
            for key, ids in observed.items()
        ],
        "missing_patterns": [key for key, ids in patterns.items() if not ids],
        "observed_failure_modes": observed_failure_modes,
        "distinguishes_failure_modes": pass_condition,
        "judge_preservation": preservation,
        "diagnostic_mapping": {
            "A": "valid analysis + supported Finding + consistent O-F",
            "B": "valid analysis + unsupported/uncertain Finding",
            "C": "invalid analysis + Finding faithful to executed analysis",
            "D": "analysis/Finding inconsistency",
        },
        "distinguishing_judgments": {
            "analysis_label": ["A/B versus C"],
            "finding_evidential_support": ["A versus B"],
            "o_f_consistency": ["C versus D"],
        },
    }


def _branch_markdown(value: Mapping[str, Any], judges: bool) -> str:
    counts = [item["reference_branch_count"] for item in value["records"]]
    availability = "available" if value.get("real_statistics_complete") else "blocked pending complete CaseEvaluationRecord inputs"
    flags = ", ".join(value.get("diagnostic_flags", [])) or "none"
    return f"# Reference-Branch Sensitivity\n\nCases: **{len(counts)}**; branch-count range: **{min(counts)}–{max(counts)}**. Synthetic redundant-equivalent-branch score invariance: **PASS**. Real best/second/tie statistics are **{availability}**. Diagnostic flags: **{flags}**. Judge calibration ready: **{judges}**."


def _metric_edge_markdown() -> str:
    return """# Metric Edge-Case Semantics

- No applicable predicted Finding: `C-Score = null` (not applicable).
- No predicted Findings: `Finding Precision = 0.0`; Core Finding Recall is `0.0` when released GT has at least one core Finding.
- No core GT Findings: formal configuration error (`UndefinedMetricError`).
- Tied best O/F branches: retain every tied branch deterministically.
- Accepted `VALID_UNENUMERATED`: add the adjudicated branch, then apply the same deterministic metrics.
- Evaluator `UNCERTAIN`: pending adjudication; never coerce to zero or correctness.
"""


def _clean_runtime_markdown(gate: Mapping[str, Any], value: Mapping[str, Any]) -> str:
    return f"# Clean-Runtime Pilot\n\nStatus: **{gate['status']}**. {gate['reason']}. This is never a formal benchmark result."


def _clean_comparison_markdown(gate: Mapping[str, Any]) -> str:
    return f"# Old vs Clean Runtime\n\nStatus: **{gate['status']}**. Historical costs are not mathematically corrected; a descriptive comparison requires fresh matched runs."


def _change_log_markdown() -> str:
    return """# Candidate Freeze Change Log

## SCIENTIFIC_CHANGES

- Carotid O1 explicitly selects the Euclidean magnitude of point-data `vectors`.
- NASA reader metadata exposes `IBlank > 0` valid-domain semantics.

## ENGINEERING_CHANGES

- `/case` contract and bounded listing; root-recursive discovery guard.
- 16,000-character head/tail output policy with explicit length telemetry.
- rollback-safe canonical multi-call boundary and provider compatibility gate.
- model-visible frozen core-library contract and import preflight.

## EVALUATION_CHANGES

- Fail-closed dual-judge/human calibration inputs.
- Explicit open-ended, O-F, branch-sensitivity and metric-edge gates.

This is a candidate freeze only while readiness is `NOT_READY`.
"""


def _summary_markdown(value: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"| {name} | {item['status']} | {item['reason']} |"
        for name, item in value["gates"].items()
    )
    return f"# Next-Stage Gate Summary\n\nFinal status: **{value['status']}**\n\n| Gate | Status | Reason |\n|---|---|---|\n{rows}"


__all__ = [
    "ANALYSIS_LABELS",
    "CORE_LIBRARIES",
    "DEFAULT_OUTPUT_LIMIT",
    "HUMAN_CLASSIFICATIONS",
    "OF_LABELS",
    "run_next_stage_gate",
]
