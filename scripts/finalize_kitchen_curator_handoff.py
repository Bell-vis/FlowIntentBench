#!/usr/bin/env python3
"""Finalize the existing Kitchen curator handoff without a model call.

This command consumes the authoritative JSON packet produced by the Kitchen
scientific transition. It renders the human review surface, derives the sparse
post-curator route, and records package/test audits. It never creates a curator
decision and never executes SCQ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.kitchen_condition_eligibility import (
    render_grounding_curator_packet_markdown,
)
from flowintentbench.case_qualification import (
    validate_sparse_controlled_family_construction,
)
from flowintentbench.lifecycle import (
    build_o3_revision_lifecycle,
    build_sparse_post_curator_route,
    canonical_curator_packet_sha256,
    derive_sparse_curator_condition_set,
    summarize_grounding_curator_state,
)
from flowintentbench.scientific_run_snapshots import canonical_json_sha256
from scripts.audit_package_parity import audit_package_parity
from scripts.validate_release_archive import validate_release_archive


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _load_sparse_case_records(
    repository_root: Path,
    packet: Mapping[str, Any],
    approved_conditions: list[str],
) -> list[dict[str, Any]]:
    """Load authored case inputs for exactly the approved sparse subset."""

    approved = set(approved_conditions)
    records: list[dict[str, Any]] = []
    for row in packet.get("conditions", ()):
        if not isinstance(row, Mapping) or row.get("condition") not in approved:
            continue
        case_id = str(row.get("case_id", ""))
        case_dir = (
            repository_root
            / "datasets"
            / str(packet.get("dataset_id", ""))
            / "construction/cases"
            / case_id
        )
        records.append({
            "case_id": case_id,
            "metadata": _read_json(case_dir / "case_construction_metadata.json")
            if (case_dir / "case_construction_metadata.json").is_file() else {},
            "case_input": _read_json(case_dir / "case_input.json")
            if (case_dir / "case_input.json").is_file() else {},
            "ground_truth": _read_json(case_dir / "ground_truth.json")
            if (case_dir / "ground_truth.json").is_file() else {},
            "context_selection": _read_json(case_dir / "context_selection.json")
            if (case_dir / "context_selection.json").is_file() else {},
        })
    return records


def _shared_authored_representability_contract(
    records: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    contracts = [
        record.get("metadata", {}).get("evaluation_representability_contract")
        for record in records
        if isinstance(record.get("metadata"), Mapping)
    ]
    if len(contracts) != len(records) or not contracts or any(
        not isinstance(contract, Mapping) for contract in contracts
    ):
        return None
    canonical = {
        json.dumps(contract, sort_keys=True, separators=(",", ":"))
        for contract in contracts
    }
    return contracts[0] if len(canonical) == 1 else None


def _file_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _o3_summary(packet: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        str(item.get("condition")): item
        for item in packet.get("conditions", ())
        if isinstance(item, Mapping)
    }
    o3 = conditions.get("O3-F1", {})
    policy = o3.get("deterministic_condition_eligibility", {})
    candidate = packet.get("o3_revised_question_candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    target = candidate.get("target_fidelity_audit", {})
    responsibility = candidate.get("responsibility_audit", {})
    candidate_policy = candidate.get("eligibility_predicate", {})
    precheck = (
        "PASS"
        if isinstance(target, Mapping)
        and target.get("status") == "PASS"
        and isinstance(responsibility, Mapping)
        and responsibility.get("status") == "PASS"
        and isinstance(candidate_policy, Mapping)
        and candidate_policy.get("eligibility") == "ELIGIBLE"
        else "FAIL"
    )
    return {
        "canonical_o3_status": (
            policy.get("eligibility", "NOT_AVAILABLE")
            if isinstance(policy, Mapping)
            else "NOT_AVAILABLE"
        ),
        "revised_candidate_wording": candidate.get("question"),
        "revised_candidate_deterministic_precheck": precheck,
        "independent_flow_expert_review_of_revised_candidate": "NOT_RUN",
        "canonical_case_mutated": candidate.get("canonical_case_mutated", False),
    }


def _validate_scientific_run_pin(
    transition_root: Path, packet: Mapping[str, Any]
) -> dict[str, Any]:
    run_id = str(packet.get("scientific_review_run_id", ""))
    run_dir = transition_root / "scientific_runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    review_path = run_dir / "atomic_scientific_review.json"
    errors: list[str] = []
    try:
        manifest = _read_json(manifest_path)
        review = _read_json(review_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "FAIL",
            "run_id": run_id,
            "errors": [f"PINNED_RUN_UNREADABLE:{type(exc).__name__}:{exc}"],
        }
    review_hash = canonical_json_sha256(review)
    checks = {
        "run_id_matches": manifest.get("run_id") == run_id,
        "review_run_id_matches": review.get("run_id") == run_id,
        "review_hash_matches_packet": review_hash
        == packet.get("scientific_review_hash"),
        "review_hash_matches_manifest": review_hash
        == manifest.get("artifact_hashes", {}).get("atomic_scientific_review.json"),
        "evidence_binding_hash_matches": manifest.get("evidence_binding_hash")
        == packet.get("evidence_binding_hash"),
        "condition_contract_hash_matches": manifest.get("condition_contract_hash")
        == packet.get("condition_contract_hash"),
        "successful_scientific_review": manifest.get("successful_scientific_review")
        is True,
    }
    errors.extend(key for key, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not errors else "FAIL",
        "run_id": run_id,
        "review_hash": review_hash,
        "checks": checks,
        "errors": errors,
    }


def _run_working_tree_tests(repository_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repository_root,
        text=True,
        capture_output=True,
    )
    lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": "python -m pytest -q",
        "returncode": completed.returncode,
        "summary": lines[-1] if lines else None,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _collect_project_handoff_tests(archive_path: str | Path | None) -> dict[str, Any]:
    if archive_path is None:
        return {"status": "NOT_EVALUATED", "reason": "project handoff ZIP not supplied"}
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        return {"status": "FAIL", "reason": "project handoff ZIP does not exist"}
    with tempfile.TemporaryDirectory(prefix="flowintentbench-project-collect-") as temporary:
        extracted = Path(temporary)
        try:
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
        except (OSError, zipfile.BadZipFile) as exc:
            return {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
        candidates = [
            extracted,
            *sorted(path.parent for path in extracted.glob("*/pyproject.toml")),
        ]
        project_root = next(
            (path for path in candidates if (path / "pyproject.toml").is_file()),
            None,
        )
        if project_root is None:
            return {"status": "FAIL", "reason": "pyproject.toml not found after extraction"}
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                cwd=project_root,
                text=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return {"status": "FAIL", "reason": "pytest collection timed out"}
        lines = [
            line.strip()
            for line in (completed.stdout + "\n" + completed.stderr).splitlines()
            if line.strip()
        ]
        return {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "command": "python -m pytest --collect-only -q",
            "returncode": completed.returncode,
            "summary": lines[-1] if lines else None,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-2000:],
        }


def finalize_kitchen_curator_handoff(
    *,
    repository_root: str | Path,
    transition_root: str | Path,
    output_root: str | Path,
    curator_artifact_path: str | Path | None = None,
    o3_revision_action_path: str | Path | None = None,
    project_handoff_zip: str | Path | None = None,
    formal_release_zip: str | Path | None = None,
    evaluation_representability_contract: Mapping[str, Any] | None = None,
    run_tests: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    transition = Path(transition_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    packet_path = transition / "curator_packet.json"
    packet_bytes_before = packet_path.read_bytes()
    packet = json.loads(packet_bytes_before.decode("utf-8"))
    if not isinstance(packet, Mapping):
        raise ValueError("authoritative curator packet root must be an object")

    markdown = render_grounding_curator_packet_markdown(packet)
    _write_text(output / "curator_packet.md", markdown)
    packet_bytes_after = packet_path.read_bytes()
    packet_unchanged = packet_bytes_before == packet_bytes_after

    pin_audit = _validate_scientific_run_pin(transition, packet)
    packet_integrity = {
        "status": "PASS" if pin_audit["status"] == "PASS" and packet_unchanged else "FAIL",
        "authoritative_json_path": str(packet_path),
        "authoritative_json_byte_sha256": _file_sha256(packet_bytes_before),
        "canonical_curator_packet_sha256": canonical_curator_packet_sha256(packet),
        "authoritative_json_unchanged_by_rendering": packet_unchanged,
        "scientific_run_pin_audit": pin_audit,
        "model_calls": 0,
    }
    _write_json(output / "packet_integrity.json", packet_integrity)

    sparse = derive_sparse_curator_condition_set(packet)
    artifact = (
        _read_json(Path(curator_artifact_path).resolve())
        if curator_artifact_path is not None
        else None
    )
    revision_action = (
        _read_json(Path(o3_revision_action_path).resolve())
        if o3_revision_action_path is not None
        else None
    )
    curator_state = summarize_grounding_curator_state(packet, artifact)
    gate_validation = curator_state["curator_gate_validation"]
    approved_conditions = list(sparse.get("curator_candidate_condition_set", ()))
    gate_confirmed = (
        gate_validation.get("status") == "PASS"
        and gate_validation.get("grounding_curator_gate_status") == "CONFIRMED"
    )
    case_records = (
        _load_sparse_case_records(root, packet, approved_conditions)
        if gate_confirmed else []
    )
    explicit_contract = evaluation_representability_contract
    if explicit_contract is None and gate_confirmed:
        explicit_contract = _shared_authored_representability_contract(case_records)
    controlled_validation = validate_sparse_controlled_family_construction(
        packet,
        gate_validation,
        approved_conditions,
        case_records,
        evaluation_representability_contract=explicit_contract,
    )
    route = build_sparse_post_curator_route(
        packet,
        artifact,
        controlled_family_validation=controlled_validation,
    )
    _write_json(output / "sparse_condition_manifest.json", sparse)
    _write_json(
        output / "controlled_family_validation.json",
        controlled_validation,
    )
    _write_json(output / "post_curator_route.json", route)

    o3 = _o3_summary(packet)
    _write_json(output / "o3_revision_candidate.json", o3)
    o3_lifecycle = build_o3_revision_lifecycle(packet, artifact, revision_action)
    _write_json(
        output / "o3_revision_lifecycle.json", o3_lifecycle
    )

    package_parity = (
        audit_package_parity(
            repository_root=root,
            project_handoff_zip=project_handoff_zip,
            formal_release_zip=formal_release_zip,
        )
        if project_handoff_zip is not None or formal_release_zip is not None
        else {"overall_package_parity_status": "NOT_EVALUATED"}
    )
    formal_validation = (
        validate_release_archive(formal_release_zip)
        if formal_release_zip is not None
        else {"overall_release_status": "NOT_EVALUATED"}
    )
    project_collection = _collect_project_handoff_tests(project_handoff_zip)
    tests = _run_working_tree_tests(root) if run_tests else {"status": "NOT_RUN"}
    _write_json(output / "package_parity.json", package_parity)
    _write_json(output / "package_formal_release_validation.json", formal_validation)
    _write_json(
        output / "package_project_handoff_test_collection.json",
        project_collection,
    )
    _write_json(output / "working_tree_tests.json", tests)

    genuine_artifact = curator_state["GENUINE_CURATOR_ARTIFACT_SUPPLIED"]
    confirmed = route.get("grounding_curator_gate_status") == "CONFIRMED"
    statuses = {
        "CURATOR_HANDOFF_READINESS": packet.get("curator_handoff_readiness"),
        "CURATOR_PACKET_AUTHORED_STATUS": curator_state[
            "CURATOR_PACKET_AUTHORED_STATUS"
        ],
        "ELIGIBLE_CONDITIONS": sparse.get("curator_candidate_condition_set", []),
        "HELD_FOR_REVISION_CONDITIONS": sparse.get("held_for_revision", []),
        "CURATOR_PINNED_SCIENTIFIC_REVIEW_ID": packet.get("scientific_review_run_id"),
        "CURATOR_PINNED_SCIENTIFIC_REVIEW_HASH": packet.get("scientific_review_hash"),
        "GENUINE_CURATOR_ARTIFACT_SUPPLIED": genuine_artifact,
        "GROUNDING_CURATOR_GATE_STATUS": curator_state[
            "GROUNDING_CURATOR_GATE_STATUS"
        ],
        "CONTROLLED_FAMILY_CONSTRUCTION_ROUTE_PERMITTED": bool(
            route.get("controlled_family_construction_candidate_set")
        ),
        "CONTROLLED_FAMILY_VALIDATION_STATUS": controlled_validation.get("status"),
        "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED": controlled_validation.get(
            "controlled_family_construction_validated", False
        ),
        "CONTROLLED_FAMILY_VALIDATED_CONDITION_SET": controlled_validation.get(
            "validated_condition_set", []
        ),
        "CONTROLLED_FAMILY_VALIDATION_BLOCKERS": controlled_validation.get(
            "failed_invariants", []
        ),
        "CONTROLLED_FAMILY_VALIDATION_CURATOR_PACKET_SHA256": controlled_validation.get(
            "curator_packet_sha256"
        ),
        "OFFICIAL_SCQ_ROUTE_PERMITTED": bool(
            route.get("official_scq_route_permitted_condition_set")
        ),
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "LIVE_SRAC_SCIENTIFIC_FAMILY_STATUS": "NOT_RUN",
        "EVALUATION_CONTRACT_CONFIRMED_COUNT": 0,
        "READY_FOR_FORMAL_EVALUATION": False,
        "NEXT_REAL_SCIENTIFIC_ACTION": curator_state[
            "NEXT_REAL_SCIENTIFIC_ACTION"
        ],
        "WORKING_TREE_TEST_STATUS": tests.get("status"),
        "PROJECT_HANDOFF_ZIP_COMPLETENESS": package_parity.get(
            "project_handoff_zip", {}
        ).get("controlled_file_completeness", "NOT_EVALUATED"),
        "PROJECT_HANDOFF_TEST_COLLECTION_STATUS": project_collection.get(
            "status", "NOT_EVALUATED"
        ),
        "FORMAL_RELEASE_ARCHIVE_STATUS": formal_validation.get(
            "overall_release_status", "NOT_EVALUATED"
        ),
        "PACKAGE_PARITY_STATUS": package_parity.get(
            "overall_package_parity_status", "NOT_EVALUATED"
        ),
    }
    required = (
        packet_integrity["status"] == "PASS",
        sparse.get("status") == "PASS",
        route.get("status") == "PASS",
        (not confirmed or controlled_validation.get("status") == "PASS"),
        o3["revised_candidate_deterministic_precheck"] == "PASS",
        o3_lifecycle.get("status")
        in {"HELD_FOR_CURATOR_ACTION", "REVISED_CASE_VERSION_CREATED"},
        tests.get("status") in {"PASS", "NOT_RUN"},
        statuses["PROJECT_HANDOFF_ZIP_COMPLETENESS"] in {"PASS", "NOT_EVALUATED"},
        statuses["PROJECT_HANDOFF_TEST_COLLECTION_STATUS"]
        in {"PASS", "NOT_EVALUATED"},
        statuses["FORMAL_RELEASE_ARCHIVE_STATUS"] in {"PASS", "NOT_EVALUATED"},
        statuses["PACKAGE_PARITY_STATUS"] in {"PASS", "NOT_EVALUATED"},
    )
    statuses["status"] = "PASS" if all(required) else "FAIL"
    _write_json(output / "readiness.json", statuses)

    candidate_wording = o3.get("revised_candidate_wording") or "NONE"
    report = [
        "# Kitchen curator-ready finalization",
        "",
        "## A. Curator handoff",
        "",
        f"- CURATOR_HANDOFF_READINESS: `{statuses['CURATOR_HANDOFF_READINESS']}`.",
        f"- CURATOR_PACKET_AUTHORED_STATUS: `{statuses['CURATOR_PACKET_AUTHORED_STATUS']}`.",
        f"- Eligible conditions: `{', '.join(statuses['ELIGIBLE_CONDITIONS']) or 'NONE'}`.",
        f"- Held for revision: `{', '.join(statuses['HELD_FOR_REVISION_CONDITIONS']) or 'NONE'}`.",
        f"- Pinned scientific review: `{statuses['CURATOR_PINNED_SCIENTIFIC_REVIEW_ID']}` / `{statuses['CURATOR_PINNED_SCIENTIFIC_REVIEW_HASH']}`.",
        "",
        "## B. O3",
        "",
        f"- Canonical O3 status: `{o3['canonical_o3_status']}`.",
        f"- Revised candidate wording: {candidate_wording}",
        f"- Candidate deterministic precheck: `{o3['revised_candidate_deterministic_precheck']}`.",
        f"- Independent Flow Expert review of candidate: `{o3['independent_flow_expert_review_of_revised_candidate']}`.",
        f"- canonical_case_mutated: `{str(o3['canonical_case_mutated']).lower()}`.",
        f"- Curator O3 action: `{o3_lifecycle.get('curator_action') or 'NONE'}`.",
        f"- Revised-case version: `{(o3_lifecycle.get('revised_case_version') or {}).get('version_id', 'NONE')}`.",
        f"- O3 re-entered validation: `{str(o3_lifecycle.get('o3_reentered_validation', False)).lower()}`.",
        "",
        "## C. Curator gate",
        "",
        f"- Genuine curator artifact supplied: `{str(genuine_artifact).lower()}`.",
        f"- Grounding curator gate status: `{statuses['GROUNDING_CURATOR_GATE_STATUS']}`.",
        "",
        "## D. Sparse routing",
        "",
        f"- Curator candidate condition set: `{', '.join(sparse.get('curator_candidate_condition_set', ())) or 'NONE'}`.",
        f"- Held for revision: `{', '.join(sparse.get('held_for_revision', ())) or 'NONE'}`.",
        f"- Controlled-family construction route permitted: `{str(statuses['CONTROLLED_FAMILY_CONSTRUCTION_ROUTE_PERMITTED']).lower()}`.",
        f"- Controlled-family validation: `{statuses['CONTROLLED_FAMILY_VALIDATION_STATUS']}`.",
        f"- Validated condition set: `{', '.join(statuses['CONTROLLED_FAMILY_VALIDATED_CONDITION_SET']) or 'NONE'}`.",
        f"- Validation blockers: `{', '.join(statuses['CONTROLLED_FAMILY_VALIDATION_BLOCKERS']) or 'NONE'}`.",
        f"- Exact curator packet hash: `{statuses['CONTROLLED_FAMILY_VALIDATION_CURATOR_PACKET_SHA256']}`.",
        f"- Official SCQ route permitted: `{str(statuses['OFFICIAL_SCQ_ROUTE_PERMITTED']).lower()}`.",
        f"- Current official SCQ route permitted set: `{', '.join(route.get('official_scq_route_permitted_condition_set', ())) or 'NONE'}`.",
        "- Synthetic lifecycle tests cover exact-hash CONFIRMED routing and PENDING/REVISE/REJECT blocking; no official SCQ is executed by this command.",
        "",
        "## E. Package reproducibility",
        "",
        f"- Working-tree tests: `{statuses['WORKING_TREE_TEST_STATUS']}` ({tests.get('summary', 'not run')}).",
        f"- Project-handoff ZIP controlled-file completeness: `{statuses['PROJECT_HANDOFF_ZIP_COMPLETENESS']}`.",
        f"- Project-handoff full test collection: `{statuses['PROJECT_HANDOFF_TEST_COLLECTION_STATUS']}` ({project_collection.get('summary', 'not evaluated')}).",
        f"- Formal release archive: `{statuses['FORMAL_RELEASE_ARCHIVE_STATUS']}`.",
        f"- Cross-package controlled-file parity: `{statuses['PACKAGE_PARITY_STATUS']}`.",
        "",
        "The project handoff and formal release have distinct roles and are not required to be byte-identical. External numerical assets remain explicitly provisioned outside both packages.",
        "",
        "## Stop gates",
        "",
        "- OFFICIAL_SCQ_EXECUTED_COUNT: `0`.",
        "- LIVE_SRAC_SCIENTIFIC_FAMILY_STATUS: `NOT_RUN`.",
        "- EVALUATION_CONTRACT_CONFIRMED_COUNT: `0`.",
        "- READY_FOR_FORMAL_EVALUATION: `false`.",
        f"- NEXT_REAL_SCIENTIFIC_ACTION: `{statuses['NEXT_REAL_SCIENTIFIC_ACTION']}`.",
    ]
    _write_text(output / "report.md", "\n".join(report))
    return {
        "status": statuses["status"],
        "output_root": str(output),
        "readiness": statuses,
        "packet_integrity": packet_integrity,
        "curator_state": curator_state,
        "sparse_route": route,
        "controlled_family_validation": controlled_validation,
        "o3_revision_lifecycle": o3_lifecycle,
        "package_parity": package_parity,
        "formal_release_validation": formal_validation,
        "project_handoff_test_collection": project_collection,
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--transition-root",
        type=Path,
        default=ROOT / "artifacts/reference/kitchen_scientific_review",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/current/kitchen",
    )
    parser.add_argument("--curator-artifact", type=Path, default=None)
    parser.add_argument("--o3-revision-action", type=Path, default=None)
    parser.add_argument("--project-handoff-zip", type=Path, default=None)
    parser.add_argument("--formal-release-zip", type=Path, default=None)
    parser.add_argument(
        "--evaluation-representability-contract",
        type=Path,
        default=None,
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    result = finalize_kitchen_curator_handoff(
        repository_root=args.repository_root,
        transition_root=args.transition_root,
        output_root=args.output_root,
        curator_artifact_path=args.curator_artifact,
        o3_revision_action_path=args.o3_revision_action,
        project_handoff_zip=args.project_handoff_zip,
        formal_release_zip=args.formal_release_zip,
        evaluation_representability_contract=(
            _read_json(args.evaluation_representability_contract.resolve())
            if args.evaluation_representability_contract is not None
            else None
        ),
        run_tests=args.run_tests,
    )
    print(json.dumps(result["readiness"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
