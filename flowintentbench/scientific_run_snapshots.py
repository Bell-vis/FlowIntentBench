"""Immutable artifact storage for Kitchen scientific review attempts.

This is deliberately an artifact helper, not a benchmark data model.  The
Kitchen transition remains responsible for the scientific contracts and
policy decisions stored in each snapshot.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


SCIENTIFIC_RUNS_DIR = "scientific_runs"
RUN_ARTIFACT_NAMES = (
    "invocation.json",
    "atomic_scientific_review.json",
    "deterministic_eligibility.json",
    "materialization_summary.json",
)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def create_scientific_run(output_root: str | Path) -> tuple[str, Path, str]:
    """Allocate a unique, stable directory for one transition attempt."""

    created_at = datetime.now(timezone.utc).isoformat()
    runs_root = Path(output_root) / SCIENTIFIC_RUNS_DIR
    runs_root.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = f"run_{uuid4().hex}"
        run_dir = runs_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_dir, created_at


def write_immutable_json(path: str | Path, value: Any) -> str:
    """Write a run artifact once and return its canonical content digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise RuntimeError(f"immutable scientific run artifact already exists: {destination}") from exc
    return canonical_json_sha256(value)


def finalize_scientific_run(
    run_dir: str | Path,
    *,
    run_id: str,
    created_at: str,
    attempt_status: str,
    scientific_review_status: str,
    artifact_hashes: Mapping[str, str],
    evidence_binding_hash: str,
    condition_contract_hash: str,
) -> dict[str, Any]:
    """Seal a complete run with a manifest written after all artifacts."""

    missing = sorted(set(RUN_ARTIFACT_NAMES) - set(artifact_hashes))
    if missing:
        raise ValueError(f"scientific run is missing artifacts: {missing}")
    manifest = {
        "run_id": run_id,
        "created_at": created_at,
        "attempt_status": attempt_status,
        "scientific_review_status": scientific_review_status,
        "successful_scientific_review": (
            attempt_status == "SUCCESS" and scientific_review_status == "PASS"
        ),
        "artifact_hash_algorithm": "CANONICAL_JSON_SHA256",
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "evidence_binding_hash": evidence_binding_hash,
        "condition_contract_hash": condition_contract_hash,
    }
    write_immutable_json(Path(run_dir) / "run_manifest.json", manifest)
    return manifest


def scientific_run_status(output_root: str | Path) -> dict[str, Any]:
    """Derive latest-attempt and last-success facts from immutable manifests."""

    runs_root = Path(output_root) / SCIENTIFIC_RUNS_DIR
    manifests: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/run_manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping) and value.get("run_id") == path.parent.name:
            manifests.append(dict(value))
    manifests.sort(key=lambda item: (str(item.get("created_at", "")), str(item.get("run_id", ""))))
    latest = manifests[-1] if manifests else None
    successful = [item for item in manifests if item.get("successful_scientific_review") is True]
    last_successful = successful[-1] if successful else None
    return {
        "LATEST_SCIENTIFIC_ATTEMPT_ID": latest.get("run_id") if latest else None,
        "LATEST_SCIENTIFIC_ATTEMPT_STATUS": latest.get("attempt_status") if latest else "NOT_RUN",
        "LAST_SUCCESSFUL_SCIENTIFIC_REVIEW_ID": last_successful.get("run_id") if last_successful else None,
        "LAST_SUCCESSFUL_SCIENTIFIC_REVIEW_STATUS": "SUCCESS" if last_successful else "NOT_AVAILABLE",
    }


__all__ = [
    "RUN_ARTIFACT_NAMES",
    "SCIENTIFIC_RUNS_DIR",
    "canonical_json_sha256",
    "create_scientific_run",
    "finalize_scientific_run",
    "scientific_run_status",
    "write_immutable_json",
]
