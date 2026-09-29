"""Authority and consistency helpers for the frozen reference portfolio.

The v7 reference snapshot is the only source of model-visible scientific
questions after freeze.  Review/authoring manifests are derived indexes or
archived compatibility inputs; they never define a question or a case.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REFERENCE_RELATIVE_PATH = "artifacts/reference/portfolio_v7_gt_aligned_final/reference_science_baseline_manifest.json"
CURRENT_POINTER_RELATIVE_PATH = "artifacts/reference/CURRENT.json"
DERIVED_REVIEW_RELATIVE_PATH = "artifacts/review/current"
ARCHIVE_RELATIVE_PATH = "artifacts/question_authoring/archive/pre_v7"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_reference_path(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    pointer = root / CURRENT_POINTER_RELATIVE_PATH
    if not pointer.is_file():
        return root / REFERENCE_RELATIVE_PATH
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("reference"), str):
        raise ValueError("reference CURRENT.json lacks a reference path")
    path = root / str(payload["reference"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = payload.get("manifest_sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError("CURRENT.json reference manifest digest mismatch")
    return path


def load_current_reference(repository_root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = current_reference_path(repository_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reference manifest must be an object")
    if payload.get("schema_version") != "reference-science-baseline-v7":
        raise ValueError("current reference must be reference-science-baseline-v7")
    if payload.get("total_case_count") != 28 or not isinstance(payload.get("cases"), list):
        raise ValueError("current reference must contain 28 cases")
    return path, payload


def _case_payload(root: Path, reference_path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    case_dir = reference_path.parent / "cases" / str(row["dataset_id"]) / str(row["case_id"])
    question_doc = json.loads((case_dir / "scientific_question.json").read_text(encoding="utf-8"))
    case_input = json.loads((case_dir / "case_input.json").read_text(encoding="utf-8"))
    projection = json.loads((case_dir / "scientific_semantic_projection.json").read_text(encoding="utf-8"))
    question = str(question_doc.get("scientific_question", "")).strip()
    if not question or case_input.get("scientific_question") != question:
        raise ValueError(f"frozen question/case_input mismatch for {row.get('case_id')}")
    return {
        "case_id": str(row["case_id"]),
        "dataset_id": str(row["dataset_id"]),
        "condition": str(row["condition"]),
        "slot_id": str(row["case_id"]),
        "scientific_question": question,
        "primary_human_facing_question": question,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "semantic_contract_sha256": str(question_doc.get("semantic_contract_sha256") or _json_hash(projection)),
        "semantic_projection_sha256": file_sha256(case_dir / "scientific_semantic_projection.json"),
        "case_input_sha256": file_sha256(case_dir / "case_input.json"),
        "reference_manifest_path": str(reference_path.relative_to(root)),
        "reference_manifest_sha256": file_sha256(reference_path),
        "reference_case_artifact_authority": "IMMUTABLE_REFERENCE_V7",
        "canonical_semantic_projection": projection,
        "scientific_target": projection.get("scientific_target"),
        "finding_responsibility": projection.get("finding_responsibility"),
        "source": str(reference_path.relative_to(root)),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sync_frozen_question_manifests(repository_root: str | Path) -> dict[str, Any]:
    """Derive review indexes from the immutable v7 case bytes.

    Existing authoring manifests are read only for compatibility fields needed
    by the legacy construction API.  No question, projection, or case path is
    taken from them.
    """

    root = Path(repository_root).resolve()
    reference_path, reference = load_current_reference(root)
    old_question_path = root / "outputs/current/scientific_questions/scientific_question_manifest.json"
    old_closure_path = root / "outputs/current/scientific_questions/human_facing_closure_manifest.json"
    archive_root = root / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions"
    if not old_question_path.is_file():
        old_question_path = archive_root / "scientific_question_manifest.json"
    if not old_closure_path.is_file():
        old_closure_path = archive_root / "human_facing_closure_manifest.json"
    old_question = json.loads(old_question_path.read_text(encoding="utf-8")) if old_question_path.is_file() else {}
    old_closure = json.loads(old_closure_path.read_text(encoding="utf-8")) if old_closure_path.is_file() else {}
    old_q_rows = {str(r.get("case_id")): r for r in old_question.get("conditions", []) if isinstance(r, Mapping)}
    old_c_rows = {str(r.get("case_id")): r for r in old_closure.get("conditions", []) if isinstance(r, Mapping)}

    rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for reference_row in reference["cases"]:
        frozen = _case_payload(root, reference_path, reference_row)
        case_id = frozen["case_id"]
        # Keep only scalar lifecycle labels in the current index.  Candidate
        # text and nested review calls belong to the archived authoring record;
        # copying them here would make a stale presentation look current.
        old_q = old_q_rows.get(case_id, {})
        index_row = {
            key: copy.deepcopy(old_q[key])
            for key in (
                "slot_id", "family_id", "concept_id", "QUESTION_CANDIDATE_STATUS",
                "SCIENTIFIC_RELEASE_STATUS", "FORMAL_CANDIDATE_STATUS", "GT_SRAC_IMPACT",
            )
            if key in old_q
        }
        index_row.update(frozen)
        index_row["authority"] = "FROZEN_REFERENCE_DERIVED"
        index_row["authoring_compatibility_available"] = bool(old_q)
        rows.append(index_row)

        old_c = old_c_rows.get(case_id, {})
        closure_row = {
            key: copy.deepcopy(old_c[key])
            for key in (
                "slot_id", "family_id", "concept_id", "QUESTION_CANDIDATE_STATUS",
                "SCIENTIFIC_RELEASE_STATUS", "FORMAL_CANDIDATE_STATUS", "GT_SRAC_IMPACT",
            )
            if key in old_c
        }
        closure_row.update(frozen)
        closure_row.update(
            {
                "authority": "FROZEN_REFERENCE_DERIVED",
                "semantic_fidelity": "PASS",
                "responsibility_integrity": "PASS",
                "readability": "NOT_ESTABLISHED",
                "surface_diversity": "NOT_ESTABLISHED",
                "review_status": "NOT_ESTABLISHED",
                "audit_basis": "EXACT_FROZEN_CASE_INPUT_AND_SCIENTIFIC_QUESTION_BYTES",
            }
        )
        closure_rows.append(closure_row)

        # The reference snapshot is write-once.  Do not "repair" a frozen
        # question document while deriving indexes: a source/provenance drift
        # must be surfaced by the read-only consistency audit instead of
        # silently changing model-visible bytes during a sync operation.

    reference_digest = file_sha256(reference_path)
    question_manifest = {
        "artifact_type": "FROZEN_SCIENTIFIC_QUESTION_INDEX",
        "schema_version": "frozen-scientific-question-index-v1",
        "authority": "FROZEN_REFERENCE_DERIVED",
        "reference_authority": "IMMUTABLE_REFERENCE_V7",
        "reference_manifest_path": str(reference_path.relative_to(root)),
        "reference_manifest_sha256": reference_digest,
        "case_count": len(rows),
        "conditions": rows,
    }
    closure_manifest = {
        "artifact_type": "FROZEN_SCIENTIFIC_QUESTION_CLOSURE",
        "schema_version": "frozen-scientific-question-closure-v1",
        "authority": "FROZEN_REFERENCE_DERIVED",
        "reference_authority": "IMMUTABLE_REFERENCE_V7",
        "reference_manifest_path": str(reference_path.relative_to(root)),
        "reference_manifest_sha256": reference_digest,
        "HUMAN_FACING_CLOSURE_STATUS": "NOT_ESTABLISHED",
        "TOTAL_QUESTION_SLOTS": len(closure_rows),
        "TOTAL_HUMAN_REVIEW_READY": 0,
        "TOTAL_SEMANTIC_FIDELITY_PASS": len(closure_rows),
        "TOTAL_RESPONSIBILITY_INTEGRITY_PASS": len(closure_rows),
        "TOTAL_RELEASE_READY": 0,
        "conditions": closure_rows,
    }
    output = root / DERIVED_REVIEW_RELATIVE_PATH
    _write_json(output / "scientific_question_manifest.json", question_manifest)
    _write_json(output / "human_facing_closure_manifest.json", closure_manifest)
    (output / "README.md").write_text(
        "# Current frozen question indexes\n\n"
        "These files are derived from `artifacts/reference/portfolio_v7_gt_aligned_final`; "
        "they are not authoring inputs. Historical candidates live under "
        "`artifacts/question_authoring/archive/pre_v7`.\n",
        encoding="utf-8",
    )
    pointer = {
        "artifact_type": "CURRENT_REFERENCE_POINTER",
        "reference": str(reference_path.relative_to(root)),
        "manifest_sha256": reference_digest,
        "case_count": 28,
        "schema_version": reference["schema_version"],
    }
    _write_json(root / CURRENT_POINTER_RELATIVE_PATH, pointer)
    return {
        "status": "PASS",
        "reference_manifest": str(reference_path.relative_to(root)),
        "reference_manifest_sha256": reference_digest,
        "case_count": len(rows),
        "output_root": str(output.relative_to(root)),
    }


def audit_frozen_question_consistency(repository_root: str | Path) -> dict[str, Any]:
    """Fail closed when any question/index/reference byte binding drifts.

    The frozen case tree is the scientific authority.  This audit deliberately
    recomputes the small set of hashes that downstream indexes advertise,
    rather than trusting a copied status flag.  It is read-only and therefore
    safe to run before every collection or evaluator replay.
    """
    root = Path(repository_root).resolve()
    reference_path, reference = load_current_reference(root)
    derived = root / DERIVED_REVIEW_RELATIVE_PATH
    failures: list[str] = []
    checks = 0
    expected_case_ids = {str(row.get("case_id")) for row in reference["cases"]}
    if len(expected_case_ids) != 28:
        failures.append("reference:DUPLICATE_OR_MISSING_CASE_IDS")
    reference_digest = file_sha256(reference_path)
    indexes: dict[str, dict[str, Any]] = {}
    for name in ("scientific_question_manifest.json", "human_facing_closure_manifest.json"):
        path = derived / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            failures.append(f"{name}:UNREADABLE")
            value = {}
        if not isinstance(value, Mapping):
            failures.append(f"{name}:NOT_OBJECT")
            value = {}
        if value.get("reference_manifest_path") != str(reference_path.relative_to(root)):
            failures.append(f"{name}:REFERENCE_PATH_MISMATCH")
        if value.get("reference_manifest_sha256") != reference_digest:
            failures.append(f"{name}:REFERENCE_HASH_MISMATCH")
        rows = value.get("conditions")
        if not isinstance(rows, list) or len(rows) != 28:
            failures.append(f"{name}:CASE_COUNT_MISMATCH")
            rows = []
        indexes[name] = {str(row.get("case_id")): dict(row) for row in rows if isinstance(row, Mapping)}
        if set(indexes[name]) != expected_case_ids:
            failures.append(f"{name}:CASE_INDEX_SET_MISMATCH")
    for row in reference["cases"]:
        case_id = str(row["case_id"])
        case_dir = reference_path.parent / "cases" / str(row["dataset_id"]) / case_id
        try:
            q = json.loads((case_dir / "scientific_question.json").read_text(encoding="utf-8"))
            ci = json.loads((case_dir / "case_input.json").read_text(encoding="utf-8"))
            projection_path = case_dir / "scientific_semantic_projection.json"
            if not projection_path.is_file():
                raise FileNotFoundError(projection_path)
            projection = json.loads(projection_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, FileNotFoundError) as exc:
            failures.append(f"{case_id}:FROZEN_CASE_UNREADABLE:{type(exc).__name__}")
            continue
        checks += 1
        question = str(q.get("scientific_question", "")).strip()
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
        case_input_hash = file_sha256(case_dir / "case_input.json")
        projection_hash = file_sha256(projection_path)
        reference_artifacts = row.get("frozen_artifacts")
        if not isinstance(reference_artifacts, Mapping):
            failures.append(f"{case_id}:REFERENCE_FROZEN_ARTIFACTS_MISSING")
            reference_artifacts = {}
        if row.get("reference_case_gt_frozen") is not True:
            failures.append(f"{case_id}:REFERENCE_CASE_GT_NOT_FROZEN")
        if q.get("authority") != "IMMUTABLE_REFERENCE_V7":
            failures.append(f"{case_id}:QUESTION_AUTHORITY_MISMATCH")
        if reference_artifacts.get("model_visible_case_sha256") != _json_hash(ci):
            failures.append(f"{case_id}:REFERENCE_CASE_INPUT_CANONICAL_HASH_MISMATCH")
        if reference_artifacts.get("semantic_contract_sha256") != _json_hash(projection):
            failures.append(f"{case_id}:REFERENCE_PROJECTION_CANONICAL_HASH_MISMATCH")
        if not question or q.get("scientific_question") != ci.get("scientific_question"):
            failures.append(f"{case_id}:QUESTION_CASE_INPUT_MISMATCH")
        for name in ("scientific_question_manifest.json", "human_facing_closure_manifest.json"):
            item = indexes.get(name, {}).get(case_id)
            if not isinstance(item, Mapping):
                failures.append(f"{case_id}:{name}:MISSING")
                continue
            if item.get("dataset_id") != row.get("dataset_id") or item.get("condition") != row.get("condition"):
                failures.append(f"{case_id}:{name}:IDENTITY_MISMATCH")
            if item.get("scientific_question") != question:
                failures.append(f"{case_id}:{name}:QUESTION_TEXT_MISMATCH")
            if item.get("question_sha256") != question_hash:
                failures.append(f"{case_id}:{name}:QUESTION_HASH_MISMATCH")
            if item.get("case_input_sha256") != case_input_hash:
                failures.append(f"{case_id}:{name}:CASE_INPUT_HASH_MISMATCH")
            if item.get("semantic_projection_sha256") != projection_hash:
                failures.append(f"{case_id}:{name}:PROJECTION_HASH_MISMATCH")
            if item.get("semantic_contract_sha256") != q.get("semantic_contract_sha256"):
                failures.append(f"{case_id}:{name}:SEMANTIC_CONTRACT_HASH_MISMATCH")
            if item.get("reference_manifest_sha256") != reference_digest:
                failures.append(f"{case_id}:{name}:REFERENCE_HASH_MISMATCH")
        if q.get("source") != str(reference_path.relative_to(root)):
            failures.append(f"{case_id}:QUESTION_SOURCE_NOT_FROZEN_REFERENCE")
        if q.get("source_manifest_sha256") != reference_digest:
            failures.append(f"{case_id}:QUESTION_SOURCE_HASH_MISMATCH")
        if not isinstance(projection, Mapping):
            failures.append(f"{case_id}:PROJECTION_NOT_OBJECT")
    return {"status": "PASS" if not failures and checks == 28 else "FAIL", "checks": checks, "failures": failures}


def assert_frozen_question_consistency(repository_root: str | Path) -> dict[str, Any]:
    """Raise an actionable error if the frozen question authority has drifted."""

    result = audit_frozen_question_consistency(repository_root)
    if result["status"] != "PASS":
        detail = ", ".join(str(item) for item in result["failures"][:12])
        suffix = "..." if len(result["failures"]) > 12 else ""
        raise ValueError(f"frozen question authority consistency failed: {detail}{suffix}")
    return result
