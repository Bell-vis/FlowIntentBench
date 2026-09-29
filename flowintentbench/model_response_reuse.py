"""Deterministic audit for reusing saved model responses after hidden edits.

This module is deliberately an audit helper, not a lifecycle state machine.
It answers one narrow question: did a construction update change anything the
evaluated model saw?  Hidden Ground Truth, SEC, binding, and adjudication
artifacts are intentionally outside the comparison surface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_runner import RunRecord, RunStatus


REUSE_AUDIT_VERSION = "model-response-reuse-audit-v1"
_MODEL_VISIBLE_CASE_DIGESTS = (
    "case_input_sha256",
    "dataset_manifest_sha256",
)
_RUN_VISIBLE_IDENTITY_FIELDS = (
    "case_presentation_version",
    "system_instruction_sha256",
    "tool_schema_sha256",
    "runtime_profile_sha256",
)


def _read_object(value: Mapping[str, Any] | str | Path, label: str) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(payload), path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError(f"{label}.cases must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not str(row.get("case_id", "")).strip():
            raise ValueError(f"{label} contains an invalid case row")
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"{label} contains duplicate case_id {case_id!r}")
        result[case_id] = row
    return result


def _load_runs(values: Sequence[RunRecord | Mapping[str, Any] | str | Path]) -> dict[str, RunRecord]:
    result: dict[str, RunRecord] = {}
    for value in values:
        if isinstance(value, RunRecord):
            record = value
        elif isinstance(value, Mapping):
            record = RunRecord.from_dict(value)
        else:
            record = RunRecord.load_json(value)
        if record.case_id in result:
            raise ValueError(f"duplicate saved response for case {record.case_id!r}")
        result[record.case_id] = record
    return result


def audit_model_response_reuse(
    *,
    run_records: Sequence[RunRecord | Mapping[str, Any] | str | Path],
    collected_case_manifest: Mapping[str, Any] | str | Path,
    candidate_case_manifest: Mapping[str, Any] | str | Path,
    expected_collected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare the frozen model-visible surface and return a fail-closed audit.

    ``REUSABLE`` means that every saved response is complete, its own visible
    runtime identity is present, its final-answer digest is valid, and the old
    and candidate case manifests agree on the model-visible case/data hashes.
    Missing identity yields ``INDETERMINATE``; an explicit mismatch yields
    ``RERUN_REQUIRED``.
    """

    collected, collected_path = _read_object(collected_case_manifest, "collected case manifest")
    candidate, candidate_path = _read_object(candidate_case_manifest, "candidate case manifest")
    old_rows = _manifest_rows(collected, "collected case manifest")
    new_rows = _manifest_rows(candidate, "candidate case manifest")
    runs = _load_runs(run_records)

    blockers: list[dict[str, Any]] = []
    indeterminate: list[dict[str, Any]] = []
    if expected_collected_manifest_sha256 is not None:
        if collected_path is None:
            indeterminate.append({"code": "COLLECTED_MANIFEST_PATH_REQUIRED_FOR_DIGEST_CHECK"})
        elif _sha256_file(collected_path) != expected_collected_manifest_sha256:
            blockers.append({"code": "COLLECTED_MANIFEST_DIGEST_MISMATCH"})

    all_case_ids = sorted(set(old_rows) | set(new_rows) | set(runs))
    case_results: list[dict[str, Any]] = []
    for case_id in all_case_ids:
        case_blockers: list[str] = []
        case_unknown: list[str] = []
        old = old_rows.get(case_id)
        new = new_rows.get(case_id)
        run = runs.get(case_id)
        if old is None or new is None or run is None:
            case_blockers.append("CASE_SET_MISMATCH")
        else:
            for field in _MODEL_VISIBLE_CASE_DIGESTS:
                old_value = old.get(field)
                new_value = new.get(field)
                if not old_value or not new_value:
                    case_unknown.append(f"MISSING_{field.upper()}")
                elif old_value != new_value:
                    case_blockers.append(f"MODEL_VISIBLE_{field.upper()}_CHANGED")
            # A bounded N=1 collection may legitimately contain a persisted
            # MODEL_NONCOMPLETION observation.  It has no natural-language
            # response to hash, but it is still an accounted trial and can be
            # consumed by the deterministic evaluator as a noncompletion
            # result.  Do not confuse that protocol outcome with an unusable
            # or infrastructure-corrupt record.
            model_noncompletion = run.run_status is RunStatus.MODEL_NONCOMPLETION
            if run.run_status not in {RunStatus.COMPLETED, RunStatus.MODEL_NONCOMPLETION} or (
                run.run_status is RunStatus.COMPLETED and not run.final_response
            ):
                case_blockers.append("SAVED_RESPONSE_NOT_COMPLETE")
            # A MODEL_NONCOMPLETION/failed record is already blocked above and
            # may legitimately have ``final_response=None``.  Do not perform a
            # second, unrelated ``None.encode`` failure: the audit must return
            # a structured RERUN_REQUIRED result rather than crash.
            if isinstance(run.final_response, str) and run.final_response:
                expected_answer = hashlib.sha256(run.final_response.encode("utf-8")).hexdigest()
                if not run.final_answer_sha256:
                    case_unknown.append("MISSING_FINAL_ANSWER_SHA256")
                elif run.final_answer_sha256 != expected_answer:
                    case_blockers.append("FINAL_ANSWER_DIGEST_MISMATCH")
            elif run.final_answer_sha256 and not model_noncompletion:
                case_blockers.append("FINAL_ANSWER_WITHOUT_RESPONSE")
            for field in _RUN_VISIBLE_IDENTITY_FIELDS:
                if not getattr(run, field):
                    case_unknown.append(f"MISSING_{field.upper()}")
        if case_blockers:
            status = "RERUN_REQUIRED"
            blockers.extend({"case_id": case_id, "code": code} for code in case_blockers)
        elif case_unknown:
            status = "INDETERMINATE"
            indeterminate.extend({"case_id": case_id, "code": code} for code in case_unknown)
        else:
            status = "REUSABLE"
        case_results.append({
            "case_id": case_id,
            "status": status,
            "model_run_status": run.run_status.value if run is not None else None,
            "accounted_model_noncompletion": bool(
                run is not None and run.run_status is RunStatus.MODEL_NONCOMPLETION
            ),
            "blockers": case_blockers,
            "indeterminate_reasons": case_unknown,
        })

    status = "RERUN_REQUIRED" if blockers else "INDETERMINATE" if indeterminate else "REUSABLE"
    return {
        "record_type": "ModelResponseReuseAudit",
        "schema_version": REUSE_AUDIT_VERSION,
        "status": status,
        "case_count": len(case_results),
        "reusable_case_count": sum(row["status"] == "REUSABLE" for row in case_results),
        "rerun_required_case_count": sum(row["status"] == "RERUN_REQUIRED" for row in case_results),
        "indeterminate_case_count": sum(row["status"] == "INDETERMINATE" for row in case_results),
        "collected_case_manifest": None if collected_path is None else str(collected_path),
        "candidate_case_manifest": None if candidate_path is None else str(candidate_path),
        "compared_case_digest_fields": list(_MODEL_VISIBLE_CASE_DIGESTS),
        "required_run_identity_fields": list(_RUN_VISIBLE_IDENTITY_FIELDS),
        "cases": case_results,
        "blockers": blockers,
        "indeterminate_reasons": indeterminate,
    }


def require_reusable_model_responses(
    *,
    run_records: Sequence[RunRecord | Mapping[str, Any] | str | Path],
    collected_case_manifest: Mapping[str, Any] | str | Path,
    candidate_case_manifest: Mapping[str, Any] | str | Path,
    expected_collected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless an existing response collection is reusable.

    Replay callers should invoke this gate before handing saved responses to
    the evaluator.  ``RERUN_REQUIRED`` and ``INDETERMINATE`` are both blocked;
    neither may be silently treated as a zero-score or a fresh response.
    """

    audit = audit_model_response_reuse(
        run_records=run_records,
        collected_case_manifest=collected_case_manifest,
        candidate_case_manifest=candidate_case_manifest,
        expected_collected_manifest_sha256=expected_collected_manifest_sha256,
    )
    if audit.get("status") != "REUSABLE":
        raise ValueError(
            "saved model responses are not reusable: "
            f"{audit.get('status', 'UNKNOWN')}"
        )
    return audit


__all__ = [
    "REUSE_AUDIT_VERSION",
    "audit_model_response_reuse",
    "require_reusable_model_responses",
]
