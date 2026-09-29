"""Reference Case/GT freeze boundary.

This module creates a small, write-once engineering manifest for the
reference portfolio.  It deliberately freezes only the scientific objects
that are already authored on disk (case identity, semantic projection,
reference materialization, GT, F2 requirements and bindings).  Live route
evidence is recorded as an excluded dependency so a later route repair cannot
silently rewrite reference truth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .case_design import CaseConstructionMetadata
from .context import EvidenceRecord, load_source_collection
from .ground_truth import load_ground_truth
from .evaluation_policy import validate_finding_verification_policy
from .schema import validate_model_input
from .scientific_artifact_binding import (
    artifact_sha256,
    validate_scientific_artifact_binding,
)
from .scientific_semantics import semantic_contract_sha256
REFERENCE_FREEZE_SCHEMA_VERSION = "reference-science-baseline-v2"
# Versioned reference boundary used by the current evaluation pipeline.  The
# v2 API remains the compatibility baseline for existing archives/tests; v3
# is an opt-in freeze that additionally records the SEC and all deterministic
# case-policy identities consumed by the evaluator.
REFERENCE_FREEZE_V3_SCHEMA_VERSION = "reference-science-baseline-v3"
# V4 is the current construction/reference hand-off.  It does not introduce
# a new scientific schema; it only freezes the corrected agent-ready case
# tree produced by the deterministic construction step.
REFERENCE_FREEZE_V4_SCHEMA_VERSION = "reference-science-baseline-v4"
# V5 is the first snapshot produced after the executable semantic-closure
# patch.  It does not alter Case/GT schemas: it records the comparator-aware
# materialization IR and the bounded construction policy as immutable
# provenance alongside the existing v4 boundary.
REFERENCE_FREEZE_V5_SCHEMA_VERSION = "reference-science-baseline-v5"
# V6 is the model-visible semantic-alignment boundary.  It keeps the v5
# Case/GT schema and execution checks, and adds a fail-closed proof that the
# frozen question/projection agree with every compiled materialization plan.
REFERENCE_FREEZE_V6_SCHEMA_VERSION = "reference-science-baseline-v6"
# V7 is the semantic-correctness snapshot.  It keeps the existing Case/GT
# schema and adds fail-closed checks for explicit quantile interpolation and
# FireFlow's vector/topology meaning.  It is an immutable construction
# boundary, not a new scientific ontology.
REFERENCE_FREEZE_V7_SCHEMA_VERSION = "reference-science-baseline-v7"
REQUIRED_CASE_FILES = (
    "case_input.json",
    "case_construction_metadata.json",
    "agent_ready_evidence.json",
    "ground_truth.json",
    "scientific_semantic_projection.json",
    "finding_execution_binding.json",
    "finding_verification_policy.json",
    "scientific_evidence_coverage.json",
    "reference_analysis.md",
)
_ROUTE_ONLY_FILES = (
    "materialization_trace.json",
    "agent_ready.json",
    "agent_readiness.json",
)
_V3_REQUIRED_CASE_FILES = REQUIRED_CASE_FILES + (
    "scientific_evaluation_contract.json",
)
_V4_REQUIRED_CASE_FILES = _V3_REQUIRED_CASE_FILES + (
    "scientific_artifact_binding.json",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_root(repository_root: Path) -> Path:
    return repository_root / "outputs/current/qualified_scientific_cases/agent_ready_cases"


def _snapshot_root(manifest_path: Path) -> Path:
    return manifest_path.parent / "cases"


def _case_condition(metadata: CaseConstructionMetadata) -> str:
    value = metadata.case_id.casefold().replace("_", "-")
    for condition in ("o1-f1", "o2-f1", "o3-f1", "o1-f2"):
        if value.endswith(condition):
            return condition.upper()
    return "UNKNOWN"


def _identity_status(metadata: CaseConstructionMetadata) -> str:
    # Combustor target slots were explicitly kept provisional in the existing
    # construction contract.  Do not promote them merely because their JSON
    # artifacts are structurally complete.
    token = f"{metadata.dataset_id}:{metadata.case_family_id}:{metadata.case_id}".casefold()
    return "PROVISIONAL" if "provisional" in token else "CANONICAL"


def _load_case_evidence(case_dir: Path) -> tuple[EvidenceRecord, ...]:
    values = _read_json(case_dir / "agent_ready_evidence.json")
    if not isinstance(values, list):
        raise ValueError("agent_ready_evidence.json must be an array")
    records = tuple(EvidenceRecord.model_validate(item) for item in values)
    ids = [item.evidence_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate evidence_id in agent_ready_evidence.json")
    return records


def audit_reference_case(case_dir: str | Path, *, repository_root: str | Path) -> dict[str, Any]:
    """Audit one authored case without inspecting live route judgments."""

    directory = Path(case_dir).resolve()
    root = Path(repository_root).resolve()
    case_id = directory.name
    missing = [name for name in REQUIRED_CASE_FILES if not (directory / name).is_file()]
    result: dict[str, Any] = {
        "case_id": case_id,
        "dataset_id": directory.parent.name,
        "condition": "UNKNOWN",
        "identity_status": "UNKNOWN",
        "reference_case_gt_frozen": False,
        "missing_files": missing,
        "errors": [],
        "warnings": [],
        "frozen_artifacts": {},
        "route_evidence_excluded": list(_ROUTE_ONLY_FILES),
    }
    if missing:
        result["errors"].append("REQUIRED_CASE_ARTIFACT_MISSING")
        return result
    try:
        metadata = CaseConstructionMetadata.model_validate(
            _read_json(directory / "case_construction_metadata.json")
        )
        case_input = _read_json(directory / "case_input.json")
        validate_model_input(case_input)
        evidence = _load_case_evidence(directory)
        gt = load_ground_truth(
            directory / "ground_truth.json",
            case_metadata=metadata,
            evidence_records=evidence,
        )
        projection = _read_json(directory / "scientific_semantic_projection.json")
        execution_binding = _read_json(directory / "finding_execution_binding.json")
        verification_policy = _read_json(directory / "finding_verification_policy.json")
        policy_result = validate_finding_verification_policy(
            verification_policy, ground_truth=gt
        )
        if policy_result["status"] != "PASS":
            result["errors"].append("FINDING_VERIFICATION_POLICY_INVALID")
        # The source collection is part of the reference provenance boundary;
        # route traces are never used to fill an absent source.
        source_path = root / "datasets" / metadata.dataset_id / "construction" / "sources.json"
        if source_path.is_file():
            collection = load_source_collection(source_path, dataset_id=metadata.dataset_id)
            source_ids = {source.source_id for source in collection.sources}
            dangling = [
                record.evidence_id
                for record in evidence
                if record.source_id not in source_ids
            ]
            if dangling:
                result["errors"].append("EVIDENCE_SOURCE_CROSS_REFERENCE_FAILURE")
        else:
            result["errors"].append("SOURCE_COLLECTION_MISSING")
        if not execution_binding.get("branches"):
            result["errors"].append("REFERENCE_G_OF_O_MISSING")
        materialization_files = sorted(directory.glob("materialization_*.json"))
        if not materialization_files:
            result["errors"].append("REFERENCE_MATERIALIZATION_MISSING")
        result.update(
            {
                "case_id": metadata.case_id,
                "dataset_id": metadata.dataset_id,
                "condition": _case_condition(metadata),
                "identity_status": _identity_status(metadata),
                "scientific_target": metadata.scientific_target,
                "finding_responsibility": (
                    "F2" if _case_condition(metadata).endswith("F2") else "F1"
                ),
                "frozen_artifacts": {
                    "dataset_manifest_sha256": _sha256_bytes(
                        root / "datasets" / metadata.dataset_id / "dataset_manifest.json"
                    ),
                    "model_visible_case_sha256": artifact_sha256(case_input),
                    "semantic_contract_sha256": artifact_sha256(projection),
                    "reference_materialization_sha256": artifact_sha256(
                        [
                            {"name": path.name, "sha256": _sha256_bytes(path)}
                            for path in materialization_files
                        ]
                    ),
                    "ground_truth_sha256": artifact_sha256(gt),
                    "case_metadata_sha256": _sha256_bytes(
                        directory / "case_construction_metadata.json"
                    ),
                    "reference_evidence_sha256": _sha256_bytes(
                        directory / "agent_ready_evidence.json"
                    ),
                    "finding_execution_binding_sha256": artifact_sha256(
                        execution_binding
                    ),
                    "finding_verification_policy_sha256": artifact_sha256(
                        verification_policy
                    ),
                    "reference_analysis_sha256": _sha256_bytes(
                        directory / "reference_analysis.md"
                    ),
                    "source_collection_sha256": _sha256_bytes(source_path),
                },
                "source_evidence_count": len(evidence),
                "reference_operationalization_count": len(gt.acceptable_operationalizations),
                "reference_finding_count": sum(
                    len(branch.findings) for branch in gt.findings_by_operationalization
                ),
                "verification_policy_status": policy_result["status"],
            }
        )
        if metadata.evaluation_representability_contract is None:
            result["warnings"].append("EVALUATION_REPRESENTABILITY_CONTRACT_NOT_AUTHORED")
        result["reference_case_gt_frozen"] = not result["errors"]
    except Exception as exc:  # preserve a case-local diagnostic, never promote it
        result["errors"].append(f"CASE_AUDIT_ERROR:{type(exc).__name__}:{exc}")
    return result


def _manifest_payload(repository_root: Path, case_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    root = case_root
    if not root.is_dir():
        raise FileNotFoundError(f"reference case root is missing: {root}")
    for case_dir in sorted(path for dataset in root.iterdir() if dataset.is_dir() for path in dataset.iterdir() if path.is_dir()):
        rows.append(audit_reference_case(case_dir, repository_root=repository_root))
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(
        item.get("reference_case_gt_frozen") is True and item.get("identity_status") == "CANONICAL"
        for item in rows
    )
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        "artifact_type": "ReferencePortfolioFreezeManifest",
        "schema_version": REFERENCE_FREEZE_SCHEMA_VERSION,
        "immutable": True,
        "authoritative_input": "cases",
        "mutable_construction_workspace_is_not_authority": True,
        "sec_srac_response_space_is_not_baseline_input": True,
        "route_evidence_is_not_freeze_input": True,
        "route_evidence_excluded_fields": list(_ROUTE_ONLY_FILES),
        "total_case_count": len(rows),
        "reference_case_gt_frozen_count": sum(item.get("reference_case_gt_frozen") is True for item in rows),
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "scientific_grounding_confirmed_count": 0,
        "official_release_ready_count": 0,
        "cases": rows,
    }


def _payload_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_reference_portfolio_freeze(
    repository_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build or verify a write-once reference freeze manifest."""

    root = Path(repository_root).resolve()
    destination = Path(output_path) if output_path is not None else root / "artifacts/archive/reference/portfolio_v2/reference_science_baseline_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        snapshot = _snapshot_root(destination)
        payload = _manifest_payload(root, snapshot)
        if _payload_digest(existing) != _payload_digest(payload):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing
    source_root = _case_root(root)
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    for source_case in sorted(
        path
        for dataset in source_root.iterdir()
        if dataset.is_dir()
        for path in dataset.iterdir()
        if path.is_dir()
    ):
        target = snapshot / source_case.parent.name / source_case.name
        target.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_CASE_FILES:
            shutil.copy2(source_case / name, target / name)
        for materialization in source_case.glob("materialization_*.json"):
            shutil.copy2(materialization, target / materialization.name)
        scientific_question = source_case / "scientific_question.json"
        if scientific_question.is_file():
            shutil.copy2(scientific_question, target / scientific_question.name)
        case_context = source_case / "case_context.json"
        if case_context.is_file():
            shutil.copy2(case_context, target / case_context.name)
    payload = _manifest_payload(root, snapshot)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _manifest_payload_v3(repository_root: Path, case_root: Path) -> dict[str, Any]:
    """Build the content-addressed v3 reference boundary.

    This deliberately composes the existing v2 case audit rather than
    introducing a second Case/GT schema.  V3 only makes the evaluator-facing
    contract explicit and records the SEC bytes as part of the immutable
    reference snapshot.  Runtime route artifacts remain excluded.
    """

    payload = _manifest_payload(repository_root, case_root)
    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        case_dir = case_root / str(row["dataset_id"]) / str(row["case_id"])
        sec = case_dir / "scientific_evaluation_contract.json"
        updated = dict(row)
        errors = list(updated.get("errors", ()))
        if not sec.is_file():
            errors.append("SCIENTIFIC_EVALUATION_CONTRACT_MISSING")
        else:
            artifacts = dict(updated.get("frozen_artifacts", {}))
            # ``materialization_trace.json`` is a route/readiness artifact,
            # not reference truth.  The legacy v2 audit predates this
            # distinction; v3 recomputes the materialization digest with the
            # explicit route exclusion so trace edits cannot drift v3.
            materialization_files = sorted(
                path
                for path in case_dir.glob("materialization_*.json")
                if path.name not in _ROUTE_ONLY_FILES
            )
            artifacts["reference_materialization_sha256"] = artifact_sha256(
                [
                    {"name": path.name, "sha256": _sha256_bytes(path)}
                    for path in materialization_files
                ]
            )
            artifacts["scientific_evaluation_contract_sha256"] = _sha256_bytes(sec)
            updated["frozen_artifacts"] = artifacts
        updated["errors"] = sorted(set(errors))
        updated["reference_case_gt_frozen"] = not updated["errors"]
        rows.append(updated)
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(
        item.get("reference_case_gt_frozen") is True
        and item.get("identity_status") == "CANONICAL"
        for item in rows
    )
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        **{key: value for key, value in payload.items() if key != "cases"},
        "schema_version": REFERENCE_FREEZE_V3_SCHEMA_VERSION,
        "authoritative_input": "reference_v3_case_snapshot",
        "evaluator_consumes_frozen_manifest": True,
        "reference_truth_boundary": {
            "case_identity": True,
            "case_input": True,
            "case_metadata": True,
            "semantic_projection": True,
            "scientific_evaluation_contract": True,
            "ground_truth": True,
            "execution_binding": True,
            "verification_policy": True,
            "evidence": True,
            "materialized_g_of_o": True,
        },
        "excluded_route_artifacts": list(_ROUTE_ONLY_FILES),
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "cases": rows,
    }


def _manifest_payload_v4(repository_root: Path, case_root: Path) -> dict[str, Any]:
    """Build the corrected evaluator-facing reference boundary.

    V4 is intentionally a thin, content-addressed revision of V3.  The
    additional checks close two construction inconsistencies that were
    present in the historical snapshot: open-O dimensions must be identical
    in the top-level and nested responsibility contracts, and every O1-F2
    reference finding must have an explicit role binding.  These are release
    integrity checks, not new scientific semantics.
    """

    payload = _manifest_payload_v3(repository_root, case_root)
    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        case_dir = case_root / str(row["dataset_id"]) / str(row["case_id"])
        updated = dict(row)
        errors = list(updated.get("errors", ()))
        try:
            metadata = _read_json(case_dir / "case_construction_metadata.json")
            condition = str(updated.get("condition", ""))
            if condition in {"O2-F1", "O3-F1"}:
                top = {
                    str(item) for item in metadata.get(
                        "unresolved_operationalization_dimensions", []
                    )
                }
                nested = {
                    str(item.get("dimension_id"))
                    for item in metadata.get("responsibility_contract", {}).get(
                        "unresolved_operationalization_dimensions", []
                    )
                    if isinstance(item, Mapping)
                }
                if top != nested:
                    errors.append("OPEN_O_RESPONSIBILITY_DIMENSION_MISMATCH")
            if condition == "O1-F2":
                sec = _read_json(case_dir / "scientific_evaluation_contract.json")
                role_map = sec.get("finding_requirement_contract", {}).get(
                    "reference_finding_role_map", {}
                )
                finding_ids = {
                    str(finding.get("finding_id"))
                    for branch in _read_json(case_dir / "ground_truth.json").get(
                        "findings_by_operationalization", []
                    )
                    if isinstance(branch, Mapping)
                    for finding in branch.get("findings", [])
                    if isinstance(finding, Mapping)
                }
                if finding_ids != set(role_map):
                    errors.append("F2_REFERENCE_FINDING_ROLE_MAP_INCOMPLETE")
            binding = _read_json(case_dir / "scientific_artifact_binding.json")
            if not isinstance(binding, Mapping):
                errors.append("SCIENTIFIC_ARTIFACT_BINDING_INVALID")
        except Exception as exc:
            errors.append(f"V4_SEMANTIC_AUDIT_ERROR:{type(exc).__name__}")
        updated["errors"] = sorted(set(errors))
        updated["reference_case_gt_frozen"] = not updated["errors"]
        rows.append(updated)
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(
        item.get("reference_case_gt_frozen") is True
        and item.get("identity_status") == "CANONICAL"
        for item in rows
    )
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        **{key: value for key, value in payload.items() if key != "cases"},
        "schema_version": REFERENCE_FREEZE_V4_SCHEMA_VERSION,
        "authoritative_input": "corrected_agent_ready_case_snapshot",
        "evaluator_consumes_frozen_manifest": True,
        "reference_truth_boundary": {
            "case_identity": True,
            "case_input": True,
            "case_metadata": True,
            "semantic_projection": True,
            "scientific_evaluation_contract": True,
            "ground_truth": True,
            "execution_binding": True,
            "verification_policy": True,
            "evidence": True,
            "materialized_g_of_o": True,
        },
        "construction_integrity_checks": [
            "OPEN_O_RESPONSIBILITY_DIMENSION_MATCH",
            "F2_REFERENCE_FINDING_ROLE_MAP_COMPLETE",
        ],
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "cases": rows,
    }


def _copy_reference_cases(source_root: Path, snapshot: Path, required_files: tuple[str, ...]) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"reference case root is missing: {source_root}")
    for source_case in sorted(
        path
        for dataset in source_root.iterdir()
        if dataset.is_dir()
        for path in dataset.iterdir()
        if path.is_dir()
    ):
        missing = [name for name in required_files if not (source_case / name).is_file()]
        if missing:
            raise ValueError(
                f"cannot build reference snapshot: {source_case.name} missing {', '.join(missing)}"
            )
        target = snapshot / source_case.parent.name / source_case.name
        target.mkdir(parents=True, exist_ok=True)
        for name in required_files:
            shutil.copy2(source_case / name, target / name)
        for materialization in source_case.glob("materialization_*.json"):
            shutil.copy2(materialization, target / materialization.name)
        for optional_name in ("scientific_question.json", "case_context.json"):
            optional = source_case / optional_name
            if optional.is_file():
                shutil.copy2(optional, target / optional_name)


def build_reference_portfolio_freeze_v3(
    repository_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify an immutable evaluator-facing reference-v3 snapshot.

    The operation is write-once.  It copies only authored reference artifacts
    plus the materialized ``G(O)`` files; route/readiness traces are excluded.
    No model, curator, or live evaluator is called.
    """

    root = Path(repository_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root
        / "artifacts/archive/reference/portfolio_v3/reference_science_baseline_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        snapshot = _snapshot_root(destination)
        current = _manifest_payload_v3(root, snapshot)
        if _payload_digest(existing) != _payload_digest(current):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing

    source_root = _case_root(root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"reference case root is missing: {source_root}")
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    for source_case in sorted(
        path
        for dataset in source_root.iterdir()
        if dataset.is_dir()
        for path in dataset.iterdir()
        if path.is_dir()
    ):
        missing = [name for name in _V3_REQUIRED_CASE_FILES if not (source_case / name).is_file()]
        if missing:
            raise ValueError(
                f"cannot build reference-v3: {source_case.name} missing {', '.join(missing)}"
            )
        target = snapshot / source_case.parent.name / source_case.name
        target.mkdir(parents=True, exist_ok=True)
        for name in _V3_REQUIRED_CASE_FILES:
            shutil.copy2(source_case / name, target / name)
        for materialization in source_case.glob("materialization_*.json"):
            shutil.copy2(materialization, target / materialization.name)
        for optional_name in ("scientific_question.json", "case_context.json"):
            optional = source_case / optional_name
            if optional.is_file():
                shutil.copy2(optional, target / optional_name)
    payload = _manifest_payload_v3(root, snapshot)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_reference_portfolio_freeze_v4(
    repository_root: str | Path,
    output_path: str | Path | None = None,
    *,
    source_case_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the corrected immutable evaluator reference snapshot.

    ``source_case_root`` is used by the construction pipeline to hand the
    freshly corrected staging tree to the freeze operation.  Omitting it
    keeps the normal repository path and remains convenient for verification.
    Existing manifests are write-once and are always re-derived from their
    physical snapshot before being accepted.
    """

    root = Path(repository_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "artifacts/archive/reference/portfolio_v4/reference_science_baseline_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("schema_version") != REFERENCE_FREEZE_V4_SCHEMA_VERSION:
            raise ValueError(f"reference freeze manifest has unexpected schema version: {destination}")
        current = _manifest_payload_v4(root, _snapshot_root(destination))
        if _payload_digest(existing) != _payload_digest(current):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing
    source_root = Path(source_case_root).resolve() if source_case_root is not None else _case_root(root)
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    _copy_reference_cases(source_root, snapshot, _V4_REQUIRED_CASE_FILES)
    payload = _manifest_payload_v4(root, snapshot)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_reference_portfolio_freeze_v4(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the V4 physical snapshot and its construction integrity checks."""

    root = Path(repository_root).resolve()
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "artifacts/archive/reference/portfolio_v4/reference_science_baseline_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping) or frozen.get("schema_version") != REFERENCE_FREEZE_V4_SCHEMA_VERSION:
        raise ReferencePortfolioDriftError("reference-v4 manifest has an unexpected schema version")
    current = _manifest_payload_v4(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError(
            "authored reference-v4 inputs differ from the immutable portfolio freeze"
        )
    failures = [row for row in frozen.get("cases", []) if row.get("errors")]
    if failures:
        raise ReferencePortfolioDriftError("reference-v4 contains failed case integrity checks")
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "route_evidence_excluded": list(frozen.get("excluded_route_artifacts", ())),
    }


def _manifest_payload_v5(repository_root: Path, case_root: Path) -> dict[str, Any]:
    """Build the v5 boundary with the Phase-1 executable semantics checks."""

    payload = _manifest_payload_v4(repository_root, case_root)
    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        updated = dict(row)
        errors = list(updated.get("errors", ()))
        case_dir = case_root / str(row["dataset_id"]) / str(row["case_id"])
        materializations = sorted(
            path for path in case_dir.glob("materialization_*.json")
            if path.name not in _ROUTE_ONLY_FILES
        )
        if not materializations:
            errors.append("MATERIALIZATION_ARTIFACT_MISSING")
        for path in materializations:
            try:
                value = _read_json(path)
                execution = value.get("execution", {}) if isinstance(value, Mapping) else {}
                plan = execution.get("data_provenance", {}).get("materialization_plan", {})
                dataset = str(row.get("dataset_id", ""))
                if dataset != "Kitchen" and str(plan.get("domain", "")) == "high_speed_region":
                    if str(plan.get("comparator", "")) not in {"GT", "GE", "LT", "LE"}:
                        errors.append(f"EXECUTION_COMPARATOR_MISSING:{path.name}")
                    if str(plan.get("criterion_kind", "")) == "quantile" and plan.get("comparator") != "GE":
                        errors.append(f"QUANTILE_COMPARATOR_INVALID:{path.name}")
                if execution.get("reproducible") is not True or not isinstance(execution.get("G_of_O"), Mapping):
                    errors.append(f"MATERIALIZATION_PROOF_INVALID:{path.name}")
            except Exception as exc:
                errors.append(f"V5_MATERIALIZATION_AUDIT_ERROR:{path.name}:{type(exc).__name__}")
        # v4 hashed the execution-binding sidecar but did not inspect its
        # verdicts.  A v5 reference is frozen only when every ReferenceFinding
        # is attached to the correct materialized G(O) output.
        try:
            binding = _read_json(case_dir / "finding_execution_binding.json")
            branches = binding.get("branches", ()) if isinstance(binding, Mapping) else ()
            if not isinstance(branches, list) or not branches:
                errors.append("FINDING_EXECUTION_BINDING_MISSING_BRANCHES")
                branches = ()
            gt = _read_json(case_dir / "ground_truth.json")
            expected_by_branch = {
                str(branch.get("operationalization_id")): {
                    str(finding.get("finding_id"))
                    for finding in branch.get("findings", ())
                    if isinstance(finding, Mapping)
                }
                for branch in gt.get("findings_by_operationalization", ())
                if isinstance(branch, Mapping)
            }
            seen_branches: set[str] = set()
            for branch in branches:
                if not isinstance(branch, Mapping):
                    errors.append("FINDING_EXECUTION_BINDING_BRANCH_INVALID")
                    continue
                branch_id = str(branch.get("branch_id", ""))
                if not branch_id or branch_id in seen_branches:
                    errors.append("FINDING_EXECUTION_BINDING_BRANCH_ID_INVALID")
                seen_branches.add(branch_id)
                findings = branch.get("findings", ())
                if not isinstance(findings, list):
                    errors.append(f"FINDING_EXECUTION_BINDING_FINDINGS_INVALID:{branch_id or 'unknown'}")
                    findings = ()
                seen_findings: set[str] = set()
                for finding in findings:
                    if not isinstance(finding, Mapping):
                        errors.append("FINDING_EXECUTION_BINDING_FINDING_INVALID")
                        continue
                    finding_id = str(finding.get("finding_id", ""))
                    if not finding_id or finding_id in seen_findings:
                        errors.append("FINDING_EXECUTION_BINDING_FINDING_ID_INVALID")
                    seen_findings.add(finding_id)
                    if finding.get("binding_status") != "PASS":
                        errors.append(
                            "FINDING_EXECUTION_BINDING_NOT_CLOSED:"
                            + (finding_id or "unknown")
                            + ":"
                            + str(finding.get("binding_status", "MISSING"))
                        )
                expected_findings = expected_by_branch.get(branch_id)
                if expected_findings is None:
                    errors.append(f"FINDING_EXECUTION_BINDING_UNKNOWN_BRANCH:{branch_id or 'unknown'}")
                elif seen_findings != expected_findings:
                    errors.append(f"FINDING_EXECUTION_BINDING_FINDING_COVERAGE_MISMATCH:{branch_id}")
            if seen_branches != set(expected_by_branch):
                errors.append("FINDING_EXECUTION_BINDING_BRANCH_COVERAGE_MISMATCH")
        except Exception as exc:
            errors.append(f"V5_EXECUTION_BINDING_AUDIT_ERROR:{type(exc).__name__}")
        updated["errors"] = sorted(set(errors))
        updated["reference_case_gt_frozen"] = not updated["errors"]
        rows.append(updated)
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(item.get("reference_case_gt_frozen") is True and item.get("identity_status") == "CANONICAL" for item in rows)
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        **{key: value for key, value in payload.items() if key != "cases"},
        "schema_version": REFERENCE_FREEZE_V5_SCHEMA_VERSION,
        "authoritative_input": "phase1_semantic_closure_case_snapshot",
        "semantic_closure": {
            "comparator_ir": True,
            "quantile_comparator": "GE",
            "unique_maximum_required": True,
            "explicit_tie_break_required": True,
            "scientific_defaults": "NONE",
            "reader_fallback": "canonical_reader_array_discovery",
            "scientific_judgments_inferred": False,
        },
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "cases": rows,
    }


def build_reference_portfolio_freeze_v5(
    repository_root: str | Path,
    output_path: str | Path | None = None,
    *,
    source_case_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the immutable v5 semantic-closure snapshot."""

    root = Path(repository_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "artifacts/archive/reference/portfolio_v5_final/reference_science_baseline_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = _read_json(destination)
        if existing.get("schema_version") != REFERENCE_FREEZE_V5_SCHEMA_VERSION:
            raise ValueError("reference freeze manifest has unexpected schema version")
        current = _manifest_payload_v5(root, _snapshot_root(destination))
        if _payload_digest(existing) != _payload_digest(current):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing
    source_root = Path(source_case_root).resolve() if source_case_root is not None else _case_root(root)
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    _copy_reference_cases(source_root, snapshot, _V4_REQUIRED_CASE_FILES)
    payload = _manifest_payload_v5(root, snapshot)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_reference_portfolio_freeze_v5(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the physical v5 snapshot and semantic-closure checks."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve() if manifest_path is not None else root / "artifacts/archive/reference/portfolio_v5_final/reference_science_baseline_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = _read_json(path)
    if frozen.get("schema_version") != REFERENCE_FREEZE_V5_SCHEMA_VERSION:
        raise ReferencePortfolioDriftError("reference-v5 manifest has an unexpected schema version")
    current = _manifest_payload_v5(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError("authored reference-v5 inputs differ from the immutable portfolio freeze")
    failures = [row for row in frozen.get("cases", []) if row.get("errors")]
    if failures:
        raise ReferencePortfolioDriftError("reference-v5 contains failed semantic-closure checks")
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "semantic_closure": frozen.get("semantic_closure", {}),
    }


def _semantic_alignment_errors(case_dir: Path, row: Mapping[str, Any]) -> list[str]:
    """Check that model-visible semantics are the semantics actually executed.

    This is intentionally a small invariant layer, not a new scientific
    ontology.  The authored projection remains the authority; materialization
    plans are checked only for details that can otherwise be hidden from a
    model (quantile interpolation, topology, and Kitchen point-to-cell
    averaging).
    """

    errors: list[str] = []
    try:
        case_input = _read_json(case_dir / "case_input.json")
        projection = _read_json(case_dir / "scientific_semantic_projection.json")
        sec = _read_json(case_dir / "scientific_evaluation_contract.json")
        binding = _read_json(case_dir / "finding_execution_binding.json")
    except Exception as exc:
        return [f"MODEL_VISIBLE_SEMANTIC_INPUT_MISSING:{type(exc).__name__}"]
    question = str(case_input.get("scientific_question", "")).strip()
    if not question:
        errors.append("MODEL_VISIBLE_QUESTION_EMPTY")
    if str(projection.get("scientific_semantic_projection_version", "")) != "scientific-semantic-projection-v3":
        errors.append("MODEL_VISIBLE_PROJECTION_VERSION_NOT_V3")
    projection_digest = semantic_contract_sha256(projection)
    if str(sec.get("source_semantic_contract_sha256", "")) != projection_digest:
        errors.append("MODEL_VISIBLE_SEC_SEMANTIC_HASH_MISMATCH")
    if str(binding.get("semantic_contract_sha256", "")) != projection_digest:
        errors.append("MODEL_VISIBLE_BINDING_SEMANTIC_HASH_MISMATCH")

    resolved_meanings = " ".join(
        str(item.get("normalized_meaning", ""))
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping)
    ).casefold()
    question_text = question.casefold()

    def meaning_is_visible(meaning: str) -> bool:
        """Check only explicit, scientifically material wording markers.

        The freeze must not attempt to parse natural language into a second
        ontology.  These small marker groups cover the deterministic choices
        already represented by the projection/materializer IR; they are
        fail-closed guards against silently dropping a fixed choice from the
        model-visible question.
        """

        checks: list[bool] = []
        if "90th percentile" in meaning:
            checks.append("90th percentile" in question_text)
        if "non-zero" in meaning:
            checks.append("non-zero" in question_text or "nonzero" in question_text)
        if "linear quantile interpolation" in meaning:
            checks.append("linear" in question_text and "interpol" in question_text)
        if "peak speed" in meaning:
            checks.append("peak speed" in question_text or "peak-speed" in question_text)
        if "mean spatial location" in meaning:
            checks.append(
                ("mean" in question_text or "average" in question_text)
                and any(token in question_text for token in ("location", "position", "coordinate"))
            )
        if "one structured-grid index step along exactly one axis" in meaning:
            checks.append(
                "six-neighbor" in question_text
                or "six neighbor" in question_text
                or "one grid axis" in question_text
                or "immediate grid" in question_text
            )
        if "arithmetic mean of its vertex" in meaning:
            checks.append("arithmetic mean" in question_text and "vertex" in question_text)
        if "cell-volume-weighted coefficient of variation" in meaning:
            checks.append(
                "cell-volume-weighted" in question_text
                and "coefficient of variation" in question_text
            )
        if "area-weighted centroid" in meaning:
            checks.append("area-weighted" in question_text and "centroid" in question_text)
        if "surface area" in meaning:
            checks.append("surface area" in question_text)
        if "constant-density isosurface" in meaning:
            checks.append("constant-density" in question_text and "isosurface" in question_text)
        return all(checks) if checks else True

    for meaning in (
        str(item.get("normalized_meaning", ""))
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping)
    ):
        if meaning.strip() and not meaning_is_visible(meaning.casefold()):
            errors.append("MODEL_VISIBLE_FIXED_OPERATIONALIZATION_SEMANTICS_OMITTED")
    if "linear quantile interpolation" in resolved_meanings:
        if not ("linear" in question_text and "interpol" in question_text):
            errors.append("MODEL_VISIBLE_QUANTILE_INTERPOLATION_OMITTED")
    if "one structured-grid index step along exactly one axis" in resolved_meanings:
        if not (
            "six-neighbor" in question_text
            or "six neighbor" in question_text
            or "one grid axis" in question_text
            or "immediate grid" in question_text
        ):
            errors.append("MODEL_VISIBLE_STRUCTURED_CONNECTIVITY_OMITTED")
    if "arithmetic mean of its vertex" in resolved_meanings:
        if not ("arithmetic mean" in question_text and "vertex" in question_text):
            errors.append("MODEL_VISIBLE_POINT_TO_CELL_RULE_OMITTED")

    plans: list[Mapping[str, Any]] = []
    for path in sorted(case_dir.glob("materialization_*.json")):
        if path.name in _ROUTE_ONLY_FILES:
            continue
        try:
            value = _read_json(path)
            plan = (
                value.get("execution", {})
                .get("data_provenance", {})
                .get("materialization_plan", {})
            )
            if isinstance(plan, Mapping):
                plans.append(plan)
        except Exception as exc:
            errors.append(f"MATERIALIZATION_PLAN_UNREADABLE:{path.name}:{type(exc).__name__}")
    if not plans:
        errors.append("MODEL_VISIBLE_MATERIALIZATION_PLANS_MISSING")
    if "linear quantile interpolation" in resolved_meanings:
        if any(
            str(plan.get("criterion_kind", "")) in {"quantile", "density_quantile", "density_q90_fraction"}
            and str(plan.get("quantile_method", "")) != "linear"
            for plan in plans
        ):
            errors.append("EXECUTION_QUANTILE_METHOD_NOT_LINEAR")
    if "one structured-grid index step along exactly one axis" in resolved_meanings:
        if any(str(plan.get("connectivity_kind", "")) != "structured_face_6" for plan in plans):
            errors.append("EXECUTION_CONNECTIVITY_NOT_STRUCTURED_FACE_6")
    if "arithmetic mean of its vertex" in resolved_meanings:
        if any(
            str(plan.get("aggregation_kind", "")) != "cell_volume_weighted"
            for plan in plans
        ):
            errors.append("EXECUTION_KITCHEN_POINT_TO_CELL_AGGREGATION_MISMATCH")
    return sorted(set(errors))


def _manifest_payload_v6(repository_root: Path, case_root: Path) -> dict[str, Any]:
    """Build v6 by composing v5 and adding model-visible alignment checks."""

    payload = _manifest_payload_v5(repository_root, case_root)
    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        updated = dict(row)
        errors = list(updated.get("errors", ()))
        case_dir = case_root / str(row["dataset_id"]) / str(row["case_id"])
        alignment_errors = _semantic_alignment_errors(case_dir, row)
        errors.extend(alignment_errors)
        updated["model_visible_semantic_alignment"] = {
            "status": "PASS" if not alignment_errors else "FAIL",
            "checks": [
                "question_contains_authored_fixed_semantics",
                "projection_hash_bound_to_sec_and_binding",
                "materialization_plan_matches_projection",
            ],
        }
        updated["errors"] = sorted(set(errors))
        updated["reference_case_gt_frozen"] = not updated["errors"]
        rows.append(updated)
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(
        item.get("reference_case_gt_frozen") is True
        and item.get("identity_status") == "CANONICAL"
        for item in rows
    )
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        **{key: value for key, value in payload.items() if key != "cases"},
        "schema_version": REFERENCE_FREEZE_V6_SCHEMA_VERSION,
        "authoritative_input": "model_visible_semantic_alignment_case_snapshot",
        "semantic_alignment": {
            "question_projection_execution_bound": True,
            "scientific_defaults": "NONE",
            "verification_semantics_inferred": False,
        },
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "cases": rows,
    }


def build_reference_portfolio_freeze_v6(
    repository_root: str | Path,
    output_path: str | Path | None = None,
    *,
    source_case_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the write-once v6 semantic-alignment snapshot."""

    root = Path(repository_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "artifacts/archive/reference/portfolio_v6_final/reference_science_baseline_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = _read_json(destination)
        if existing.get("schema_version") != REFERENCE_FREEZE_V6_SCHEMA_VERSION:
            raise ValueError("reference freeze manifest has unexpected schema version")
        current = _manifest_payload_v6(root, _snapshot_root(destination))
        if _payload_digest(existing) != _payload_digest(current):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing
    source_root = Path(source_case_root).resolve() if source_case_root is not None else _case_root(root)
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    _copy_reference_cases(source_root, snapshot, _V4_REQUIRED_CASE_FILES)
    payload = _manifest_payload_v6(root, snapshot)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_reference_portfolio_freeze_v6(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the v6 physical snapshot and semantic alignment checks."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve() if manifest_path is not None else root / "artifacts/archive/reference/portfolio_v6_final/reference_science_baseline_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = _read_json(path)
    if frozen.get("schema_version") != REFERENCE_FREEZE_V6_SCHEMA_VERSION:
        raise ReferencePortfolioDriftError("reference-v6 manifest has an unexpected schema version")
    current = _manifest_payload_v6(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError("authored reference-v6 inputs differ from immutable portfolio freeze")
    failures = [row for row in frozen.get("cases", []) if row.get("errors")]
    if failures:
        raise ReferencePortfolioDriftError("reference-v6 contains failed semantic-alignment checks")
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "semantic_alignment": frozen.get("semantic_alignment", {}),
    }


def _semantic_alignment_errors_v7(case_dir: Path, row: Mapping[str, Any]) -> list[str]:
    """Add the final model-visible semantic guards without changing v6 rules."""

    errors = list(_semantic_alignment_errors(case_dir, row))
    try:
        case_input = _read_json(case_dir / "case_input.json")
        projection = _read_json(case_dir / "scientific_semantic_projection.json")
    except Exception as exc:
        return sorted(set(errors + [f"MODEL_VISIBLE_V7_INPUT_MISSING:{type(exc).__name__}"]))
    question = str(case_input.get("scientific_question", "")).casefold()
    scope = projection.get("scientific_scope", {})
    constraints = " ".join(
        str(item).casefold()
        for item in (scope.get("condition_constraints", ()) if isinstance(scope, Mapping) else ())
    )
    if "euclidean magnitude of the stored uvw" in constraints:
        if "euclidean magnitude" not in question or "uvw" not in question:
            errors.append("MODEL_VISIBLE_FIRE_FLOW_VECTOR_MAGNITUDE_OMITTED")
    if "unstructured-mesh cell" in constraints:
        if not any(
            marker in question
            for marker in ("unstructured-mesh cell", "share an unstructured-mesh cell", "same mesh cell")
        ):
            errors.append("MODEL_VISIBLE_FIRE_FLOW_CELL_CONNECTIVITY_OMITTED")

    plans: list[Mapping[str, Any]] = []
    for path in sorted(case_dir.glob("materialization_*.json")):
        try:
            value = _read_json(path)
            plan = value.get("execution", {}).get("data_provenance", {}).get("materialization_plan", {})
            if isinstance(plan, Mapping):
                plans.append(plan)
        except Exception:
            continue
    resolved = " ".join(
        str(item.get("normalized_meaning", "")).casefold()
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping)
    )
    # Only fixed quantile semantics are required to be visible at the
    # projection/question boundary.  O2/O3 may legitimately leave criterion
    # open; their materialized branch records the selected convention in G(O)
    # without pretending that the question fixed it in advance.
    fixed_quantile = "linear quantile interpolation" in resolved or "nearest-rank" in resolved or "nearest rank" in resolved
    for plan in plans:
        kind = str(plan.get("criterion_kind", ""))
        if not fixed_quantile or kind not in {"quantile", "density_quantile", "density_q90_fraction"}:
            continue
        method = str(plan.get("quantile_method", ""))
        if method == "linear":
            if "linear" not in resolved or "interpol" not in resolved:
                errors.append("PROJECTION_QUANTILE_INTERPOLATION_UNSPECIFIED")
            if "linear" not in question or "interpol" not in question:
                errors.append("MODEL_VISIBLE_QUANTILE_INTERPOLATION_OMITTED")
        elif method == "nearest_rank":
            if "nearest-rank" not in resolved and "nearest rank" not in resolved:
                errors.append("PROJECTION_QUANTILE_INTERPOLATION_UNSPECIFIED")
            if "nearest-rank" not in question and "nearest rank" not in question:
                errors.append("MODEL_VISIBLE_NEAREST_RANK_OMITTED")
        else:
            errors.append("EXECUTION_QUANTILE_METHOD_UNSUPPORTED")
    return sorted(set(errors))


def _manifest_payload_v7(repository_root: Path, case_root: Path) -> dict[str, Any]:
    """Build v7 from the v6 payload plus the final semantic guards."""

    payload = _manifest_payload_v6(repository_root, case_root)
    rows: list[dict[str, Any]] = []
    for row in payload["cases"]:
        updated = dict(row)
        errors = list(updated.get("errors", ()))
        case_dir = case_root / str(row["dataset_id"]) / str(row["case_id"])
        errors.extend(_semantic_alignment_errors_v7(case_dir, row))
        updated["model_visible_semantic_alignment"] = {
            "status": "PASS" if not errors else "FAIL",
            "checks": [
                "question_contains_authored_fixed_semantics",
                "projection_hash_bound_to_sec_and_binding",
                "materialization_plan_matches_projection",
                "quantile_interpolation_is_explicit",
                "fireflow_vector_and_topology_semantics_are_visible",
            ],
        }
        updated["errors"] = sorted(set(errors))
        updated["reference_case_gt_frozen"] = not updated["errors"]
        rows.append(updated)
    rows.sort(key=lambda item: (item.get("dataset_id", ""), item.get("case_id", "")))
    canonical = sum(
        item.get("reference_case_gt_frozen") is True
        and item.get("identity_status") == "CANONICAL"
        for item in rows
    )
    provisional = sum(item.get("identity_status") == "PROVISIONAL" for item in rows)
    return {
        **{key: value for key, value in payload.items() if key != "cases"},
        "schema_version": REFERENCE_FREEZE_V7_SCHEMA_VERSION,
        "authoritative_input": "semantic_correctness_case_snapshot",
        "semantic_alignment": {
            "question_projection_execution_bound": True,
            "scientific_defaults": "NONE",
            "verification_semantics_inferred": False,
            "quantile_interpolation_explicit": True,
            "fireflow_vector_and_topology_explicit": True,
        },
        "canonical_identity_frozen_count": canonical,
        "provisional_identity_count": provisional,
        "cases": rows,
    }


def build_reference_portfolio_freeze_v7(
    repository_root: str | Path,
    output_path: str | Path | None = None,
    *,
    source_case_root: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the write-once v7 semantic-correctness snapshot."""

    root = Path(repository_root).resolve()
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "artifacts/reference/portfolio_v7_gt_aligned_final/reference_science_baseline_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        existing = _read_json(destination)
        if existing.get("schema_version") != REFERENCE_FREEZE_V7_SCHEMA_VERSION:
            raise ValueError("reference freeze manifest has unexpected schema version")
        current = _manifest_payload_v7(root, _snapshot_root(destination))
        if _payload_digest(existing) != _payload_digest(current):
            raise ValueError(f"reference freeze manifest is immutable and differs: {destination}")
        return existing
    source_root = Path(source_case_root).resolve() if source_case_root is not None else _case_root(root)
    snapshot = _snapshot_root(destination)
    if snapshot.exists():
        raise ValueError(f"unbound reference snapshot already exists: {snapshot}")
    _copy_reference_cases(source_root, snapshot, _V4_REQUIRED_CASE_FILES)
    payload = _manifest_payload_v7(root, snapshot)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_reference_portfolio_freeze_v7(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the v7 physical snapshot and semantic-correctness checks."""

    root = Path(repository_root).resolve()
    path = Path(manifest_path).resolve() if manifest_path is not None else root / "artifacts/reference/portfolio_v7_gt_aligned_final/reference_science_baseline_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = _read_json(path)
    if frozen.get("schema_version") != REFERENCE_FREEZE_V7_SCHEMA_VERSION:
        raise ReferencePortfolioDriftError("reference-v7 manifest has an unexpected schema version")
    current = _manifest_payload_v7(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError("authored reference-v7 inputs differ from immutable portfolio freeze")
    failures = [row for row in frozen.get("cases", []) if row.get("errors")]
    if failures:
        raise ReferencePortfolioDriftError("reference-v7 contains failed semantic-correctness checks")
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "semantic_alignment": frozen.get("semantic_alignment", {}),
    }


def verify_reference_portfolio_freeze_v3(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a v3 snapshot without consuming mutable route evidence."""

    root = Path(repository_root).resolve()
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "artifacts/archive/reference/portfolio_v3/reference_science_baseline_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping) or frozen.get("schema_version") != REFERENCE_FREEZE_V3_SCHEMA_VERSION:
        raise ReferencePortfolioDriftError("reference-v3 manifest has an unexpected schema version")
    current = _manifest_payload_v3(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError(
            "authored reference-v3 inputs differ from the immutable portfolio freeze"
        )
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "route_evidence_excluded": list(frozen.get("excluded_route_artifacts", ())),
    }


class ReferencePortfolioDriftError(ValueError):
    """Raised when authored reference inputs differ from the write-once freeze."""


def verify_reference_portfolio_freeze(
    repository_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen reference boundary against the current authored inputs.

    Only the objects listed in ``_manifest_payload`` participate in this
    comparison.  Route evidence (materialization traces, readiness reports,
    and other live adjudication outputs) is intentionally excluded, so route
    repair cannot cause reference truth to drift.  A mismatch is fail-closed.
    """

    root = Path(repository_root).resolve()
    path = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "artifacts/archive/reference/portfolio_v2/reference_science_baseline_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"reference freeze manifest is missing: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping):
        raise ReferencePortfolioDriftError("reference freeze manifest must be a JSON object")
    current = _manifest_payload(root, _snapshot_root(path))
    if _payload_digest(frozen) != _payload_digest(current):
        raise ReferencePortfolioDriftError(
            "authored reference inputs differ from the immutable portfolio freeze"
        )
    return {
        "status": "PASS",
        "manifest_path": str(path),
        "manifest_digest": _payload_digest(frozen),
        "total_case_count": frozen.get("total_case_count", 0),
        "reference_case_gt_frozen_count": frozen.get("reference_case_gt_frozen_count", 0),
        "route_evidence_excluded": list(frozen.get("route_evidence_excluded_fields", ())),
    }


__all__ = [
    "REFERENCE_FREEZE_SCHEMA_VERSION",
    "REFERENCE_FREEZE_V3_SCHEMA_VERSION",
    "REFERENCE_FREEZE_V4_SCHEMA_VERSION",
    "REFERENCE_FREEZE_V5_SCHEMA_VERSION",
    "REFERENCE_FREEZE_V6_SCHEMA_VERSION",
    "REFERENCE_FREEZE_V7_SCHEMA_VERSION",
    "REQUIRED_CASE_FILES",
    "audit_reference_case",
    "build_reference_portfolio_freeze",
    "build_reference_portfolio_freeze_v3",
    "build_reference_portfolio_freeze_v4",
    "build_reference_portfolio_freeze_v5",
    "ReferencePortfolioDriftError",
    "verify_reference_portfolio_freeze",
    "verify_reference_portfolio_freeze_v3",
    "verify_reference_portfolio_freeze_v4",
    "verify_reference_portfolio_freeze_v5",
    "build_reference_portfolio_freeze_v6",
    "verify_reference_portfolio_freeze_v6",
    "build_reference_portfolio_freeze_v7",
    "verify_reference_portfolio_freeze_v7",
]
