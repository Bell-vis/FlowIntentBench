"""Fail-closed preflight for the explicit full-dataset N=1 pilot.

This command performs only repository/artifact validation.  It never creates
an adapter and never calls a model.  The optional test subprocess is the same
frozen repository test suite used by the release checklist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    CaseConstructionMetadata,
    CaseConstructionMetadataValidator,
    CaseLoader,
    ContextSelection,
    DatasetManifest,
    EvidenceRecord,
    GroundTruth,
    GroundTruthValidator,
    EvidenceSourceCrossReferenceValidator,
    code_data_version,
    load_source_collection,
    REFERENCE_FREEZE_V6_SCHEMA_VERSION,
    REFERENCE_FREEZE_V7_SCHEMA_VERSION,
    verify_reference_portfolio_freeze_v6,
    verify_reference_portfolio_freeze_v7,
)
from scripts.build_full_dataset_n1_manifest import (  # noqa: E402
    CONDITIONS,
    DATASET_CASES,
    PILOT_LABEL,
    sha256_file,
)
from flowintentbench.reference_authority import (  # noqa: E402
    CURRENT_POINTER_RELATIVE_PATH,
    assert_frozen_question_consistency,
)
from flowintentbench.experiment_scope import case_inventory


FORBIDDEN_MODEL_VISIBLE_TERMS = (
    "ground_truth",
    "evaluator",
    "reference_finding",
    "case_construction_metadata",
    "adjudication",
    "operationalization_evidence",
    "finding_evidence",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual.casefold() != str(expected).casefold():
        raise ValueError(f"{label} SHA256 mismatch: {path}")


def _validate_model_visible_case(case_input_path: Path) -> None:
    payload = _read_json(case_input_path)
    leaked: set[str] = set()

    def visit(value: Any, *, key: str | None = None) -> None:
        if key is not None and any(term in key.casefold() for term in FORBIDDEN_MODEL_VISIBLE_TERMS):
            leaked.add(key)
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, key=str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            normalized = value.casefold()
            # Natural-language statements may use ordinary scientific words;
            # only path-like strings can expose a hidden artifact by name.
            if ("/" in value or "\\" in value or normalized.endswith((".json", ".yaml", ".yml", ".md"))) and any(term in normalized for term in FORBIDDEN_MODEL_VISIBLE_TERMS):
                leaked.update(term for term in FORBIDDEN_MODEL_VISIBLE_TERMS if term in normalized)
    visit(payload)
    if leaked:
        raise ValueError(f"model-visible case input contains forbidden artifact references: {sorted(leaked)}")


def validate_manifest(
    manifest_path: str | Path,
    *,
    repository_root: str | Path = ROOT,
    output_root: str | Path | None = None,
    run_tests: bool = True,
    allow_existing_collection: bool = False,
    host_network_tests: bool = False,
) -> dict[str, Any]:
    repo = Path(repository_root).resolve()
    authority_audit = None
    if (repo / CURRENT_POINTER_RELATIVE_PATH).is_file():
        authority_audit = assert_frozen_question_consistency(repo)
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = repo / manifest_file
    payload = _read_json(manifest_file)
    if not isinstance(payload, Mapping):
        raise ValueError("pilot manifest must be a JSON object")
    if payload.get("manifest_version") != "full-dataset-n1-v1":
        raise ValueError("unexpected full-dataset N=1 manifest version")
    if payload.get("pilot_label") != PILOT_LABEL:
        raise ValueError("pilot label is not the frozen non-formal label")
    if payload.get("N") != 1 or payload.get("formal") is not False:
        raise ValueError("full-dataset N=1 manifest must be N=1 and formal=false")
    canonical_manifest = (repo / "experiments/userstudy/case_manifest.json").resolve()
    if manifest_file.resolve() == canonical_manifest:
        expected_authority = {
            "reference-science-baseline-v4": "IMMUTABLE_REFERENCE_V4",
            "reference-science-baseline-v5": "IMMUTABLE_REFERENCE_V5",
            "reference-science-baseline-v6": "IMMUTABLE_REFERENCE_V6",
            "reference-science-baseline-v7": "IMMUTABLE_REFERENCE_V7",
        }.get(str(payload.get("reference_schema_version", "")))
        if expected_authority is None:
            raise ValueError("canonical N=1 manifest must declare a supported immutable reference schema")
        if payload.get("case_artifact_authority") != expected_authority:
            raise ValueError("canonical N=1 manifest must be bound to its declared immutable reference")
        if not isinstance(payload.get("reference_manifest_path"), str) or not isinstance(payload.get("reference_manifest_sha256"), str):
            raise ValueError("canonical N=1 manifest must bind its reference manifest")
        if payload.get("reference_schema_version") == REFERENCE_FREEZE_V6_SCHEMA_VERSION:
            reference_path = Path(str(payload["reference_manifest_path"]))
            if not reference_path.is_absolute():
                reference_path = repo / reference_path
            verify_reference_portfolio_freeze_v6(repo, reference_path)
        elif payload.get("reference_schema_version") == REFERENCE_FREEZE_V7_SCHEMA_VERSION:
            reference_path = Path(str(payload["reference_manifest_path"]))
            if not reference_path.is_absolute():
                reference_path = repo / reference_path
            verify_reference_portfolio_freeze_v7(repo, reference_path)
    cases = list(case_inventory(payload, require_dataset=True).values())

    actual = [(item.get("dataset_id"), item.get("case_id"), item.get("condition")) for item in cases]
    source_kind = str(payload.get("case_source", "frozen_controlled_pilot_table"))
    if source_kind == "authoritative_source_manifest":
        # Qualified construction is an explicit authority.  It may contain
        # renamed/provisional IDs (for example Combustor), so comparing to the
        # historical directory table would silently validate the wrong cases.
        if sorted({row[0] for row in actual}) != sorted(payload.get("datasets", ())):
            raise ValueError("authoritative manifest dataset set mismatch")
        if any(row[2] not in CONDITIONS for row in actual):
            raise ValueError("authoritative manifest contains an unsupported condition")
        source_manifest_value = payload.get("source_manifest_path")
        source_manifest_digest = payload.get("source_manifest_sha256")
        if not isinstance(source_manifest_value, str) or not isinstance(source_manifest_digest, str):
            raise ValueError("authoritative manifest must bind its source manifest path and SHA256")
        source_manifest_path = Path(source_manifest_value)
        if not source_manifest_path.is_absolute():
            source_manifest_path = repo / source_manifest_path
        _assert_hash(source_manifest_path, source_manifest_digest, "source_manifest")
        # The N=1 development pilot may execute a candidate whose upstream
        # construction status is pending.  That status is snapshotted and
        # reported separately by the runner; preflight only proves that the
        # explicitly named artifacts are coherent.  It must not turn an
        # upstream construction state into a model-run failure.
    else:
        expected = [(dataset, case_id, condition) for dataset, specs in DATASET_CASES for case_id, condition in specs]
        if actual != expected:
            raise ValueError("manifest case order/content does not match the frozen 7x4 table")
    if any(item.get("case_id") == "initial_case" for item in cases):
        raise ValueError("initial_case must not be present in the pilot manifest")

    loader = CaseLoader(repo / "datasets")
    checked = []
    for position, item in enumerate(cases):
        if item.get("trial_index") != 1 or item.get("case_execution_position") != position:
            raise ValueError(f"invalid trial/position for {item.get('case_id')}")
        def path_for(name: str) -> Path:
            value = item.get(f"{name}_path")
            if not isinstance(value, str):
                raise ValueError(f"missing {name}_path")
            path = repo / value
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        case_input_path = path_for("case_input")
        metadata_path = path_for("case_construction_metadata")
        gt_path = path_for("ground_truth")
        context_path = path_for("case_context")
        dataset_manifest_path = path_for("dataset_manifest")
        _assert_hash(case_input_path, item["case_input_sha256"], "case_input")
        _assert_hash(metadata_path, item["case_construction_metadata_sha256"], "case_construction_metadata")
        _assert_hash(gt_path, item["ground_truth_sha256"], "ground_truth")
        _assert_hash(context_path, item["case_context_sha256"], "case_context")
        _assert_hash(dataset_manifest_path, item["dataset_manifest_sha256"], "dataset_manifest")
        if item.get("scientific_evaluation_contract_path"):
            sec_path = repo / item["scientific_evaluation_contract_path"]
            _assert_hash(sec_path, item["scientific_evaluation_contract_sha256"], "scientific_evaluation_contract")
            if (case_input_path.parent / "scientific_artifact_binding.json").is_file():
                from flowintentbench.evaluation_contract import FrozenEvaluationCase
                FrozenEvaluationCase.from_paths(case_input_path.parent).validate()
        _validate_model_visible_case(case_input_path)

        loaded = loader.load(case_input_path)
        dataset_id = str(item["dataset_id"])
        manifest = DatasetManifest.model_validate_json(dataset_manifest_path.read_text(encoding="utf-8"))
        manifest.validate_files_against_loaded_case(loaded)
        metadata = CaseConstructionMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        case_input = _read_json(case_input_path)
        selection_value = item.get("context_selection_path")
        selection_path = (
            repo / str(selection_value)
            if isinstance(selection_value, str) and not Path(selection_value).is_absolute()
            else Path(selection_value)
            if isinstance(selection_value, str)
            else case_input_path.parent / "context_selection.json"
        )
        selection = None
        if selection_path.is_file():
            selection = ContextSelection.model_validate(_read_json(selection_path))
        CaseConstructionMetadataValidator.validate(
            case_input["scientific_question"], metadata, context_selection=selection
        )
        evidence = []
        evidence_value = item.get("evidence_path")
        if isinstance(evidence_value, str):
            evidence_path = repo / evidence_value
            if not evidence_path.is_file():
                raise FileNotFoundError(evidence_path)
            _assert_hash(evidence_path, item["evidence_sha256"], "evidence")
            evidence.extend(
                EvidenceRecord.model_validate(value)
                for value in _read_json(evidence_path)
            )
        else:
            for filename in ("context_evidence.json", "operationalization_evidence.json", "finding_evidence.json"):
                evidence.extend(EvidenceRecord.model_validate(value) for value in _read_json(repo / "datasets" / dataset_id / "construction" / filename))
        sources = load_source_collection(
            repo / "datasets" / dataset_id / "construction" / "sources.json",
            dataset_id=dataset_id,
            require_formal_source=True,
        )
        EvidenceSourceCrossReferenceValidator.validate(dataset_id, sources, evidence)
        GroundTruthValidator.validate(GroundTruth.model_validate_json(gt_path.read_text(encoding="utf-8")), metadata, evidence_records=evidence)
        checked.append({"dataset_id": dataset_id, "case_id": item["case_id"], "condition": item["condition"]})

    if output_root is None:
        output = repo / "outputs" / "full-dataset-n1-pilot-v1"
    else:
        output = Path(output_root)
        if not output.is_absolute():
            output = repo / output
    conflicts = []
    if output.exists():
        if allow_existing_collection:
            allowed = {
                "preflight_report.json",
                "collection_state.json",
                # The collection runner persists the exact model-visible
                # manifest it consumed.  It is an authority snapshot, not a
                # conflicting trial directory, and must survive a same-root
                # resume/evaluation-only invocation.
                "case_manifest.json",
                # Older collections used this name for the same snapshot.
                "collected_case_manifest.json",
                "runs",
                "pilot_manifest.json",
                "development_snapshot.json",
                # Cross-manifest resume preserves prior experiment identities
                # beside the current one; these are orchestration metadata,
                # not trial/model output.
                "experiment_manifest_history",
            }
            conflicts.extend(
                str(path)
                for path in output.iterdir()
                if path.name not in allowed
            )
        else:
            conflicts.extend(str(path) for path in output.rglob("trial-1"))
            if (output / "pilot_manifest.json").exists():
                conflicts.append(str(output / "pilot_manifest.json"))
    if conflicts:
        raise FileExistsError("pilot output root contains conflicting trial-1 data: " + ", ".join(conflicts))

    test_command = [sys.executable, "-m", "pytest", "-q"]
    if host_network_tests:
        test_command.extend(["-m", "not sandbox", "-n", "4"])
    test_result = {
        "command": test_command,
        "returncode": None,
        "summary": "NOT RUN",
    }
    if run_tests:
        test_environment = dict(os.environ)
        if host_network_tests:
            # A regression gate invokes pytest again. Its child must retain
            # the selected host-network condition and never probe namespaces.
            test_environment["PYTEST_ADDOPTS"] = test_environment.get("PYTEST_ADDOPTS", "") + " -m 'not sandbox'"
        completed = subprocess.run(test_command, cwd=repo, text=True, capture_output=True,
                                   **({"env": test_environment} if host_network_tests else {}))
        evidence = repo / "outputs/validation/latest_suite.log"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        output_lines = [
            line.strip()
            for line in (completed.stdout + "\n" + completed.stderr).splitlines()
            if line.strip()
        ]
        raw_summary = output_lines[-1] if output_lines else "NO TEST OUTPUT"
        # Keep the evidence human-readable but stable across repeated runs;
        # wall-clock duration is runtime telemetry, not experiment identity.
        summary = re.sub(r"\s+in\s+[0-9]+(?:\.[0-9]+)?s\s*$", "", raw_summary)
        test_result = {
            "command": test_command,
            "returncode": completed.returncode,
            # Keep the persisted report stable across an otherwise identical
            # resume.  Wall-clock timing belongs to runtime telemetry, not
            # to the experiment identity binding.
            "summary": summary if completed.returncode == 0 else raw_summary,
        }
        if completed.returncode != 0:
            raise RuntimeError("frozen test suite failed during preflight")

    try:
        output_label = str(output.relative_to(repo))
    except ValueError:
        output_label = str(output)
    try:
        manifest_label = str(manifest_file.relative_to(repo))
    except ValueError:
        manifest_label = str(manifest_file)
    return {
        "status": "PASS",
        "pilot_label": PILOT_LABEL,
        "manifest": manifest_label,
        "case_manifest_sha256": sha256_file(manifest_file),
        "benchmark_code_data_version": code_data_version(repo),
        "case_count": len(checked),
        "dataset_count": len(set(item["dataset_id"] for item in checked)),
        "conditions": list(CONDITIONS),
        "model_calls": 0,
        "output_root": output_label,
        "frozen_tests": test_result,
        "question_authority_consistency": authority_audit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/userstudy/case_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/userstudy/preflight")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = validate_manifest(args.manifest, output_root=args.output_root, run_tests=True)
    report_value = args.report or (args.output_root / "preflight_report.json")
    report = report_value if report_value.is_absolute() else ROOT / report_value
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
