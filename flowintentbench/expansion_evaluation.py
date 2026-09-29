"""Development-only case adapter for expansion_v1; delegates all scoring."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .case_design import CaseConstructionMetadata
from .evaluation_policy import compile_finding_verification_policy, validate_finding_verification_policy
from .external_file_evaluator import digest, write_json
from .finding_requirements import FindingRequirementContract
from .ground_truth import GroundTruth
from .schema import validate_model_input

VERSION = "expansion-development-evaluation-v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compile_branch_execution_evidence(root, row, gt, *, definition=None, reference_dir=None):
    """Bind original execution outputs, never ReferenceFinding records."""
    dataset_path = root / "datasets" / row["dataset_id"] / "dataset_manifest.json"
    dataset_manifest = read_json(dataset_path)
    input_files = [{"path": item["path"], "sha256": item["checksum"]} for item in dataset_manifest["files"]]
    evidence = {}
    if definition is not None:
        source = root / "datasets/expansion_v1/execution_evidence" / (row["family_id"] + ".json")
        execution = read_json(source)
        authored_digest = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        if execution.get("definition_sha256") != authored_digest:
            raise ValueError(f"construction execution definition changed: {row['case_id']}")
        if execution.get("executor_sha256") != file_sha256(root / "flowintentbench/construction_recipes.py"):
            raise ValueError("construction recipe implementation changed")
        if execution.get("inputs") != input_files or execution.get("status") != "EXECUTED":
            raise ValueError("construction execution inputs/status do not match frozen data")
    for operation in gt.acceptable_operationalizations:
        effective = {d.dimension.value: d.statement for d in operation.decisions}
        if definition is not None:
            name = next((name for name, op in execution["operations"].items() if op["clauses"] == effective), None)
            if name is None or execution["operations"][name] != definition["operations"][name]:
                raise ValueError("construction G(O) cannot be bound to the exact reference O")
            result = execution["results"][name]
            details = {"recipe": execution["operations"][name]["recipe"],
                       "definition_sha256": execution["definition_sha256"],
                       "executor_sha256": execution["executor_sha256"],
                       "binding_basis": "EXACT_AUTHORED_O_AND_EXECUTED_FAMILY_DEFINITION"}
        else:
            source = reference_dir / ("materialization_" + operation.operationalization_id + ".json")
            raw = read_json(source)
            original = raw.get("execution", {})
            if raw.get("operation_id") != operation.operationalization_id or original.get("status") != "MATERIALIZED" or original.get("reproducible") is not True:
                raise ValueError("preserved execution branch/status binding mismatch")
            result = original["G_of_O"]
            provenance = original.get("data_provenance", {})
            details = {"binding_basis": "FROZEN_BRANCH_ID_AND_IDENTICAL_SOURCE_GT",
                       "original_execution_provenance": {key: provenance[key] for key in (
                           "materialization_plan", "compiled_plan_sha256", "executed_plan_sha256",
                           "effective_o_sha256", "reader_format", "reader_configuration") if key in provenance}}
        record = {"case_id": row["case_id"], "dataset_id": row["dataset_id"],
                  "operationalization": effective, "materialized_G_of_O": copy.deepcopy(result),
                  "source_artifact": {"path": str(source.relative_to(root)), "sha256": file_sha256(source)},
                  "execution_provenance": {"status": "MATERIALIZED", "dataset_manifest_sha256": file_sha256(dataset_path),
                                           "data_file_root": dataset_manifest.get("file_root", "datasets"),
                                           "input_files": input_files, "reader": dataset_manifest["reader"], **details}}
        record["evidence_binding_sha256"] = digest(record)
        evidence[operation.operationalization_id] = record
    return evidence


def compile_expansion_evaluation(root: str | Path, output: str | Path) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root / "datasets/expansion_v1/case_manifest.json"
    construction = read_json(source)
    families, cases = [], []
    for row in construction["cases"]:
        for key, value in row.items():
            if key.endswith("_path") and key[:-5] + "_sha256" in row:
                if file_sha256(root / value) != row[key[:-5] + "_sha256"]:
                    raise ValueError(f"stale construction artifact: {value}")
        case_id, dataset_id = row["case_id"], row["dataset_id"]
        metadata = CaseConstructionMetadata.model_validate(read_json(root / row["case_construction_metadata_path"]))
        ground_truth = GroundTruth.model_validate(read_json(root / row["ground_truth_path"]))
        contract = None
        material = {"schema_version": VERSION, "case_id": case_id,
                    "audience": "EVALUATOR_ONLY",
                    "evaluation_mode": "DEVELOPMENT_EVALUATION", "formal_release": False,
                    "human_calibration_status": "NOT_ESTABLISHED",
                    "source_case": row, "finding_recall_mode": "FIXED_REFERENCE_CORE"}
        definition_path = root / "datasets" / dataset_id / "construction/families.json"
        if str(row.get("origin", "")).startswith("FROZEN_REFERENCE_"):
            projection_path = root / row["scientific_semantic_projection_path"]
            projection = read_json(projection_path)
            material["scientific_semantic_projection"] = projection
            reference_dir = root / "artifacts/reference/portfolio_v7_gt_aligned_final/cases" / dataset_id / case_id
            # The semantic projection is authoritative. The old SEC contributes
            # only its per-finding role map after the semantic portions agree.
            if row["condition"].endswith("F2"):
                contract = copy.deepcopy(projection["finding_semantics"]["adequate_core_semantics"])
                sec_path = reference_dir / "scientific_evaluation_contract.json"
                sec_contract = read_json(sec_path)["finding_requirement_contract"]
                for key, value in contract.items():
                    if sec_contract.get(key) != value:
                        raise ValueError(f"reference semantic contract mismatch: {case_id}:{key}")
                contract["reference_finding_role_map"] = sec_contract["reference_finding_role_map"]
                material["role_map_source"] = {"path": str(sec_path.relative_to(root)), "sha256": file_sha256(sec_path)}
            policy_path = reference_dir / "finding_verification_policy.json"
            policy = read_json(policy_path)
            # Exact reference GT identity prevents stale per-finding policy reuse.
            if read_json(reference_dir / "ground_truth.json") != ground_truth.model_dump(mode="json"):
                raise ValueError(f"reference GT changed: {case_id}")
            material["verification_policy_source"] = {"path": str(policy_path.relative_to(root)), "sha256": file_sha256(policy_path)}
            material["branch_execution_evidence"] = compile_branch_execution_evidence(root, row, ground_truth, reference_dir=reference_dir)
            evidence_path = reference_dir / "agent_ready_evidence.json"
            if evidence_path.is_file():
                material["reference_materialization_evidence"] = {"path": str(evidence_path.relative_to(root)), "sha256": file_sha256(evidence_path)}
        else:
            definition = next(f for f in read_json(definition_path)["families"] if f["family_id"] == row["family_id"])
            material["family_definition"] = definition
            material["family_definition_source"] = {"path": str(definition_path.relative_to(root)), "sha256": file_sha256(definition_path)}
            material["branch_execution_evidence"] = compile_branch_execution_evidence(root, row, ground_truth, definition=definition)
            if row["condition"].endswith("F2"):
                contract = copy.deepcopy(definition["finding_requirement_contract"])
                contract["reference_finding_role_map"] = {
                    finding.finding_id: contract["role_by_category"][finding.category.value]
                    for branch in ground_truth.findings_by_operationalization for finding in branch.findings}
            bindings = {"branches": [{"operationalization_id": branch.operationalization_id,
                          "findings": [{"finding_id": finding.finding_id,
                                        "g_of_o_locator": "/" + next(prop["result_key"] for prop in definition["finding_properties"] if finding.finding_id.endswith("_" + prop["property_id"]))}
                                       for finding in branch.findings]}
                         for branch in ground_truth.findings_by_operationalization]}
            policy = compile_finding_verification_policy(
                ground_truth=ground_truth, finding_execution_binding=bindings,
                scientific_target=metadata.scientific_target,
                semantic_contract_sha256=digest({"responsibility_contract": metadata.responsibility_contract.model_dump(mode="json") if hasattr(metadata.responsibility_contract, "model_dump") else metadata.responsibility_contract,
                                                 "family_definition": definition}),
                significant_figures=5)
        validation = validate_finding_verification_policy(policy, ground_truth=ground_truth)
        if validation["status"] != "PASS":
            raise ValueError(f"invalid policy for {case_id}: {validation}")
        if contract is not None:
            normalized = FindingRequirementContract.from_mapping(contract)
            errors = normalized.validate()
            if errors:
                raise ValueError(f"invalid F2 contract {case_id}: {errors}")
            contract = normalized.to_dict()
            material["finding_recall_mode"] = "SEMANTIC_ADEQUATE_CORE"
        material["finding_requirement_contract"] = contract
        material["finding_verification_policy"] = policy
        family = {"family_id": row["family_id"], "dataset_id": dataset_id,
                  "condition": row["condition"], "controlled_case_ids": [case_id],
                  "formal_release_eligible": False}
        if contract is not None:
            family["finding_requirement_contract"] = contract
        families.append(family)
        destination = output / "cases" / case_id / "evaluation_material.json"
        write_json(destination, material)
        cases.append({**row, "evaluation_material_path": str(destination),
                      "evaluation_material_sha256": file_sha256(destination),
                      "finding_recall_mode": material["finding_recall_mode"]})
    manifest = {"schema_version": VERSION, "status": "DEVELOPMENT_EVALUATION_ONLY",
                "formal_release": False, "human_calibration_status": "NOT_ESTABLISHED",
                "source_manifest_sha256": file_sha256(source), "case_count": len(cases),
                "scientific_family_count": construction["family_count"],
                "cases": cases, "families": families}
    write_json(output / "case_manifest.json", manifest)
    return manifest


def load_development_case(root: str | Path, row: dict[str, Any]):
    root = Path(root)
    material_path = root / row["evaluation_material_path"]
    if file_sha256(material_path) != row["evaluation_material_sha256"]:
        raise ValueError("development evaluation material digest mismatch")
    material = read_json(material_path)
    for evidence in material.get("branch_execution_evidence", {}).values():
        source = evidence["source_artifact"]
        if file_sha256(root / source["path"]) != source["sha256"]:
            raise ValueError("bound execution evidence source changed")
    for key in ("case_input", "case_construction_metadata", "ground_truth"):
        if file_sha256(root / row[key + "_path"]) != row[key + "_sha256"]:
            raise ValueError(f"frozen {key} changed")
    return (validate_model_input(read_json(root / row["case_input_path"])),
            CaseConstructionMetadata.model_validate(read_json(root / row["case_construction_metadata_path"])),
            GroundTruth.model_validate(read_json(root / row["ground_truth_path"])), material)
