"""Small, non-scientific status projections for model-run pilots.

The pilot runner executes candidate cases, while release closure decides
whether a case is scientifically ready.  These responsibilities are kept
orthogonal here: ``upstream_case_status`` is copied from an explicit closure
row, ``model_run_status`` records response collection, and
``evaluator_status`` records the separate evaluator stage.  This module
never infers scientific validity and never recalculates evaluation metrics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


UPSTREAM_RELEASE_READY = "RELEASE_READY"
UPSTREAM_CASE_CANDIDATE = "CASE_CANDIDATE"
UPSTREAM_PENDING_CASE_IDENTITY = "PENDING_CASE_IDENTITY"
UPSTREAM_PENDING_EVIDENCE = "PENDING_EVIDENCE"
UPSTREAM_PENDING_SRAC = "PENDING_SRAC"
UPSTREAM_PENDING_GT_BINDING = "PENDING_GT_SEMANTIC_BINDING"
UPSTREAM_PENDING_CURATOR = "PENDING_CURATOR"
UPSTREAM_PENDING_RELEASE_AUTHORIZATION = "PENDING_RELEASE_AUTHORIZATION"
UPSTREAM_NOT_EVALUATED = "UPSTREAM_NOT_EVALUATED"
UPSTREAM_INVALID = "UPSTREAM_INVALID"

DEVELOPMENT_SNAPSHOT_VERSION = "pilot-development-snapshot-v1"

_BLOCKER_STATUS = {
    "CASE_IDENTITY_PENDING": UPSTREAM_PENDING_CASE_IDENTITY,
    "CASE_LINEAGE_PENDING": UPSTREAM_PENDING_CASE_IDENTITY,
    "EVIDENCE_CLOSURE_PENDING": UPSTREAM_PENDING_EVIDENCE,
    "SRAC_PENDING": UPSTREAM_PENDING_SRAC,
    "GT_SEMANTIC_BINDING_PENDING": UPSTREAM_PENDING_GT_BINDING,
    "CURATOR_CONFIRMATION_PENDING": UPSTREAM_PENDING_CURATOR,
    "RELEASE_AUTHORIZATION_PENDING": UPSTREAM_PENDING_RELEASE_AUTHORIZATION,
}


def project_upstream_case_status(case_entry: Mapping[str, Any] | None) -> str:
    """Project one explicit release-closure row to a construction status.

    The projection is deliberately fail-closed.  Missing/invalid rows are
    ``UPSTREAM_NOT_EVALUATED``/``UPSTREAM_INVALID``; a candidate is never
    promoted merely because its files exist.  ``formal_release_eligible`` is
    the only release-ready authority accepted here.
    """

    if case_entry is None:
        return UPSTREAM_NOT_EVALUATED
    if not isinstance(case_entry, Mapping):
        return UPSTREAM_INVALID
    if case_entry.get("formal_release_eligible") is True:
        return UPSTREAM_RELEASE_READY
    blockers = case_entry.get("blockers", ())
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes, bytearray)):
        for blocker in blockers:
            status = _BLOCKER_STATUS.get(str(blocker).strip().upper())
            if status:
                return status
    identity = str(case_entry.get("identity_status", "")).strip().upper()
    if identity in {"PENDING", "PENDING_CASE_IDENTITY", "NEW_CASE_REQUIRED"}:
        return UPSTREAM_PENDING_CASE_IDENTITY
    release_status = str(case_entry.get("release_status", "")).strip().upper()
    if release_status in {"RELEASE_READY", "ELIGIBLE"}:
        # A non-authoritative positive label is still a candidate unless the
        # closure explicitly set formal_release_eligible above.
        return UPSTREAM_CASE_CANDIDATE
    if case_entry.get("selected_for_release") is False or release_status in {
        "CONTROLLED_PILOT_ONLY",
        "PENDING",
        "BLOCKED",
    }:
        return UPSTREAM_CASE_CANDIDATE
    return UPSTREAM_NOT_EVALUATED


def load_upstream_case_statuses(
    repository_root: str | Path,
    *,
    closure_path: str | Path | None = None,
    case_ids: Sequence[str] = (),
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load statuses from the existing closure artifact without discovery.

    The returned metadata is suitable for an immutable pilot snapshot.  An
    absent closure is represented explicitly rather than guessed from case
    directories.
    """

    root = Path(repository_root).resolve()
    source = Path(closure_path).resolve() if closure_path else root / "outputs/current/formal_release_closure/latest.json"
    requested = [str(case_id) for case_id in case_ids]
    statuses = {case_id: UPSTREAM_NOT_EVALUATED for case_id in requested}
    metadata: dict[str, Any] = {
        "source_path": str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
        "source_status": "MISSING",
        "source_sha256": None,
    }
    if not source.is_file():
        return statuses, metadata
    raw_bytes = source.read_bytes()
    metadata["source_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        metadata["source_status"] = "INVALID"
        return statuses, metadata
    rows = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        metadata["source_status"] = "INVALID"
        return statuses, metadata
    by_id = {
        str(row.get("case_id")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("case_id") is not None
    }
    for case_id in requested:
        statuses[case_id] = project_upstream_case_status(by_id.get(case_id))
    metadata["source_status"] = "PRESENT"
    metadata["row_count"] = len(rows)
    return statuses, metadata


def build_case_execution_status(
    *,
    case_id: str,
    upstream_case_status: str,
    model_run_status: str | None = None,
    evaluator_status: str | None = None,
) -> dict[str, Any]:
    """Return an orthogonal status record for one pilot case."""

    if not str(case_id).strip():
        raise ValueError("case_id must be non-empty")
    upstream = str(upstream_case_status).strip() or UPSTREAM_NOT_EVALUATED
    model_run = None if model_run_status is None else str(model_run_status).strip() or None
    evaluator = None if evaluator_status is None else str(evaluator_status).strip() or None
    return {
        "case_id": str(case_id),
        "upstream_case_status": upstream,
        "model_run_status": model_run,
        "evaluator_status": evaluator,
        # Scientific metrics are intentionally left to the evaluator layer;
        # model-run collection must not imply eligibility from completion.
        "scientific_metrics_status": "NOT_ASSESSED",
    }


def build_development_snapshot(
    *,
    manifest_sha256: str,
    benchmark_code_data_version: str,
    target_fingerprint: str,
    upstream_statuses: Mapping[str, str],
    upstream_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build deterministic, write-once input metadata for a pilot run."""

    return {
        "record_type": "flowintentbench_development_snapshot",
        "schema_version": DEVELOPMENT_SNAPSHOT_VERSION,
        "case_manifest_sha256": str(manifest_sha256),
        "benchmark_code_data_version": str(benchmark_code_data_version),
        "target_fingerprint": str(target_fingerprint),
        "upstream_source": dict(upstream_source),
        "case_statuses": [
            {"case_id": str(case_id), "upstream_case_status": str(upstream_statuses[case_id])}
            for case_id in sorted(upstream_statuses)
        ],
    }


def write_immutable_development_snapshot(
    path: str | Path,
    payload: Mapping[str, Any],
) -> str:
    """Write a deterministic snapshot once and return its SHA-256.

    Reusing an identical snapshot is allowed on resume.  Any attempted
    replacement fails closed, preventing a pilot from changing upstream
    inputs while retaining old run records.
    """

    destination = Path(path)
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != serialized:
            raise RuntimeError(f"immutable development snapshot mismatch: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(serialized)
    return hashlib.sha256(serialized).hexdigest()
