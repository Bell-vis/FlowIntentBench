"""Evaluate explicit saved RunRecords in PILOT or FORMAL mode.

This entrypoint never generates model responses.  PILOT mode writes
inspectable trial records (including pending records) and is explicitly not a
benchmark result.  FORMAL mode additionally validates release membership and
requires the frozen three-trial aggregation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluator_backend import StructuredEvaluatorBackend
from flowintentbench.continuation_checkpoint import dispatch_with_checkpoint

from flowintentbench import (
    BenchmarkReleaseManifest,
    build_agent_ready_novel_o_materializer,
    CaseAggregate,
    CaseConstructionMetadata,
    CaseEvaluator,
    CaseEvaluationResult,
    build_case_evaluator,
    CaseLoader,
    DatasetManifest,
    EvaluationAdjudications,
    SemanticMatchResolution,
    UnitFrameResolution,
    EvaluationTarget,
    EfficiencyObservation,
    GroundTruth,
    OpenAICompatibleEvaluatorBackend,
    PendingCaseEvaluationRecord,
    CaseEvaluationRecord,
    RunRecord,
    RunStatus,
    RunEvaluationAdjudication,
    TrialEvaluation,
    aggregate_cases,
    aggregate_trials_for_case,
    evaluation_manifest_digest,
    load_evaluation_adjudications,
    load_case_evaluation_record,
    load_run_evaluation_adjudication,
    save_case_evaluation_record,
    validate_formal_release_case,
    resolve_provider_configuration,
    resolve_api_key,
    ScientificEvaluationContract,
    EvaluatorBackendError,
    build_pending_adjudication_packet,
    dispatch_pending_continuation,
    PendingContinuationRegistry,
    load_pending_continuation_registry,
    require_reusable_model_responses,
    provider_failure_info,
    route_pending_adjudication,
)
from flowintentbench.evaluation_records import resolve_current_evaluation_record_path
from flowintentbench.case_design import primary_case_type


def _parse_assignments(values: Sequence[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use CASE_ID=PATH")
        case_id, path = value.split("=", 1)
        if not case_id or case_id in result:
            raise ValueError(f"{option} contains duplicate or empty case ID: {case_id!r}")
        result[case_id] = Path(path)
    if not result:
        raise ValueError(f"at least one {option} assignment is required")
    return result


def _parse_run_assignments(values: Sequence[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError("--run must use CASE_ID=RUN_RECORD_JSON")
        case_id, path = value.split("=", 1)
        if not case_id:
            raise ValueError("--run contains an empty case ID")
        result[case_id].append(Path(path))
    if not result:
        raise ValueError("at least one --run assignment is required")
    return dict(result)


def _parse_dataset_assignments(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-id must use CASE_ID=DATASET_ID")
        case_id, dataset_id = value.split("=", 1)
        case_id = case_id.strip()
        dataset_id = dataset_id.strip()
        if not case_id or not dataset_id or case_id in result:
            raise ValueError("--dataset-id contains duplicate or empty case/dataset ID")
        result[case_id] = dataset_id
    return result


def _safe_component(value: object) -> str:
    normalized = str(value).strip().replace("/", "_").replace("\\", "_")
    return normalized if normalized and normalized not in {".", ".."} else "unnamed"


def _dataset_id_for_case(case_dir: Path, datasets_root: Path, explicit: str | None = None) -> str:
    """Resolve dataset identity without relying on a fixed directory depth.

    Qualified cases are staged under ``outputs/.../agent_ready_cases/<dataset>``
    while historical cases use ``datasets/<dataset>/construction/cases``.  The
    dataset manifest is the authority when present; the direct child-name
    fallback is only accepted when that exact manifest exists.
    """

    if explicit is not None:
        dataset_id = str(explicit).strip()
        if not dataset_id:
            raise ValueError("explicit dataset_id must be non-empty")
        manifest_path = datasets_root / dataset_id / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        return dataset_id

    # Standard tree: an ancestor is the dataset directory containing the
    # manifest.  Iterate from the case upward so nested staging directories
    # cannot shadow the authoritative dataset manifest.
    for ancestor in (case_dir, *case_dir.parents):
        manifest_path = ancestor / "dataset_manifest.json"
        if manifest_path.is_file() and ancestor.parent == datasets_root:
            return ancestor.name

    # Qualified output tree: its immediate parent is the dataset label, while
    # data files still resolve from ``datasets_root``.
    candidates = []
    for candidate in (case_dir.parent.name, case_dir.parents[1].name if len(case_dir.parents) > 1 else ""):
        if candidate and (datasets_root / candidate / "dataset_manifest.json").is_file():
            if candidate not in candidates:
                candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"cannot resolve dataset for case directory {case_dir}; "
            "pass an explicit dataset binding"
        )
    raise ValueError(f"ambiguous dataset binding for {case_dir}: {candidates}")


def _resolve_run_adjudication(
    run_record: RunRecord,
    adjudications_by_run_id: Mapping[
        str, EvaluationAdjudications | RunEvaluationAdjudication
    ] | None,
) -> EvaluationAdjudications | None:
    if not adjudications_by_run_id:
        return None
    value = adjudications_by_run_id.get(run_record.run_id)
    if value is None:
        return None
    if isinstance(value, RunEvaluationAdjudication):
        if value.run_id != run_record.run_id or value.case_id != run_record.case_id:
            raise ValueError(
                "run adjudication wrapper does not match the RunRecord run_id/case_id"
            )
        return value.adjudications
    if not isinstance(value, EvaluationAdjudications):
        raise TypeError("adjudication mapping values must be run-bound or base records")
    return value


def _pending_artifact_path(run_dir: Path) -> Path:
    canonical = run_dir / "pending_case_evaluation.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(run_dir.glob("pending_case_evaluation.attempt-*.json"))
    return candidates[-1] if candidates else canonical


def _current_record_path(run_dir: Path) -> Path | None:
    """Resolve the current record through its atomic pointer.

    The pointer is authoritative whenever present.  Legacy directories that
    predate it fall back to the canonical filenames, but ambiguous legacy
    state fails closed instead of guessing between final and pending.
    """

    return resolve_current_evaluation_record_path(run_dir)


def _archive_current_attempt(run_dir: Path) -> Path | None:
    """Copy current generated artifacts to an immutable retry snapshot.

    This function deliberately does *not* rename or remove the current
    canonical files.  A retry may fail between this snapshot and the next
    terminal record; retaining the old current until ``_save_current_record``
    atomically replaces it keeps current selection crash-safe.
    """

    existing = [
        path
        for path in (
            run_dir / "case_evaluation_record.json",
            run_dir / "pending_case_evaluation.json",
            run_dir / "evaluator_backend_failure.json",
        )
        if path.is_file()
    ]
    if not existing:
        return None
    attempts_root = run_dir / "evaluation_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (attempts_root / f"attempt-{index:03d}").exists():
        index += 1
    attempt_dir = attempts_root / f"attempt-{index:03d}"
    attempt_dir.mkdir()
    manifest: dict[str, Any] = {"attempt": index, "immutable": True, "files": {}}
    for source in existing:
        destination = attempt_dir / source.name
        shutil.copy2(source, destination)
        manifest["files"][source.name] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
    (attempt_dir / "attempt_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return attempt_dir


def _save_current_record(
    record: CaseEvaluationRecord | PendingCaseEvaluationRecord,
    path: Path,
    *,
    archive_existing: bool = True,
) -> None:
    """Persist an immutable attempt, then atomically select the new record.

    The short two-pointer protocol matters for retries: before the canonical
    compatibility mirror is replaced, ``selected_file`` points at the fully
    written attempt.  A crash at any point therefore leaves either the old
    canonical record or the new immutable attempt readable; no reader has to
    guess from directory order.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_current = path.is_file() or (path.parent / "pending_case_evaluation.json").is_file()
    fallback_path: Path | None = None
    fallback_digest: str | None = None
    if existing_current and archive_existing:
        archived = _archive_current_attempt(path.parent)
        if archived is not None:
            # The archive stores the bytes that were selected before this
            # write.  It becomes the crash-safe fallback while the canonical
            # mirror is replaced.
            old_name = path.name
            fallback_candidate = archived / old_name
            if not fallback_candidate.is_file():
                alternate = (
                    "pending_case_evaluation.json"
                    if old_name == "case_evaluation_record.json"
                    else "case_evaluation_record.json"
                )
                fallback_candidate = archived / alternate
            if fallback_candidate.is_file():
                fallback_path = fallback_candidate
                fallback_digest = hashlib.sha256(
                    fallback_candidate.read_bytes()
                ).hexdigest()
    pointer = path.parent / "current_evaluation_pointer.json"

    def write_pointer(payload: Mapping[str, Any]) -> None:
        pointer_tmp = pointer.with_name(
            f"{pointer.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        pointer_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(pointer_tmp, pointer)

    if fallback_path is not None:
        # Publish a crash fallback before changing the canonical mirror.
        previous_pointer = {
            "current_file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file()
            else fallback_digest,
            "fallback_file": fallback_path.relative_to(path.parent).as_posix(),
            "fallback_sha256": fallback_digest,
            "immutable_history_dir": "evaluation_attempts",
        }
        write_pointer(previous_pointer)

    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    save_case_evaluation_record(record, temporary)
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Commit the canonical mirror selection only after it is complete.
    write_pointer(
        {
            "current_file": path.name,
            "sha256": digest,
            "immutable_history_dir": "evaluation_attempts",
        }
    )
    # The committed pointer is durable; removing the alternate canonical name
    # cannot create a no-current window.  A crash before this cleanup is safe
    # because every reader follows the pointer first.
    alternate_name = (
        "pending_case_evaluation.json"
        if path.name == "case_evaluation_record.json"
        else "case_evaluation_record.json"
    )
    alternate = path.parent / alternate_name
    if alternate.is_file() and alternate != path:
        alternate.unlink()


def _freeze_evaluation_manifest(
    evaluation_root: Path, evaluator: CaseEvaluator
) -> str:
    """Freeze exactly one evaluator manifest for one evaluation collection.

    A collection cannot silently mix evaluator implementations/spec versions.
    The lock is write-once and is checked before any run is evaluated.
    """

    digest = evaluation_manifest_digest(evaluator.evaluation_manifest)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    lock_path = evaluation_root / "evaluation_manifest.lock.json"
    payload = {
        "evaluation_manifest": evaluator.evaluation_manifest.to_dict(),
        "evaluation_manifest_digest": digest,
        "immutable": True,
    }
    if lock_path.is_file():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing.get("evaluation_manifest_digest") != digest:
            raise ValueError(
                "evaluation collection is locked to a different evaluator manifest"
            )
        return digest
    # Existing records may predate the lock.  Refuse to bless a heterogeneous
    # directory as a new homogeneous collection.
    persisted: set[str] = set()
    for path in evaluation_root.rglob("case_evaluation_record.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest = value.get("evaluation_manifest")
            if isinstance(manifest, Mapping):
                persisted.add(evaluation_manifest_digest(manifest))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    for path in evaluation_root.rglob("pending_case_evaluation*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest = value.get("evaluation_manifest")
            if isinstance(manifest, Mapping):
                persisted.add(evaluation_manifest_digest(manifest))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    if persisted and persisted != {digest}:
        raise ValueError(
            "existing evaluation artifacts use multiple evaluator manifests; "
            "start a new evaluation collection"
        )
    # Sharded replay processes may initialize the same collection concurrently.
    # Use a process-unique temporary name so one writer cannot move another
    # writer's temporary file before its ``os.replace`` call.  The lock itself
    # remains write-once and atomic.
    temporary = lock_path.with_name(
        f"{lock_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, lock_path)
    return digest


def _load_case_scientific_contract(
    case_id: str,
    case_dir: Path,
) -> ScientificEvaluationContract | None:
    path = case_dir / "scientific_evaluation_contract.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(
            f"scientific evaluation contract for {case_id!r} must be a JSON object"
        )
    contract = ScientificEvaluationContract(
        case_id=str(value.get("case_id", "")),
        enumerated_valid_O=tuple(value.get("enumerated_valid_O", ()) or ()),
        explicitly_invalid_O=tuple(value.get("explicitly_invalid_O", ()) or ()),
        unenumerated_policy=str(value.get("unenumerated_policy", "")),
        G_of_O=tuple(value.get("G_of_O", ()) or ()),
        finding_requirement_contract=value.get("finding_requirement_contract", {}),
        adequate_core_sets=tuple(
            tuple(item) for item in value.get("adequate_core_sets", ()) or ()
        ),
        tolerances=value.get("tolerances", {}),
        adjudication_provenance=tuple(
            value.get("adjudication_provenance", ()) or ()
        ),
        status=str(value.get("status", "DRAFT_REQUIRES_HUMAN_CONFIRMATION")),
        adjudicated_findings=tuple(value.get("adjudicated_findings", ()) or ()),
        source_semantic_contract_sha256=value.get(
            "source_semantic_contract_sha256"
        ),
    )
    if contract.case_id != case_id:
        raise ValueError(
            f"scientific evaluation contract case_id mismatch for {case_id!r}"
        )
    return contract


def _configure_case_evaluator(
    *,
    evaluator: CaseEvaluator,
    case_id: str,
    case_dir: Path,
    metadata: CaseConstructionMetadata,
    datasets_root: Path,
    mode: Literal["PILOT", "FORMAL"],
    portfolio_manifest: str | Path | Mapping[str, object] | None,
) -> CaseEvaluator:
    """Bind the case-local SEC and trusted execution route to one evaluator.

    The configured evaluator components retain their own identity.  The
    evaluated model remains identified exclusively by the RunRecord target
    fingerprint; no target configuration is copied into this evaluator.
    """

    contract = _load_case_scientific_contract(case_id, case_dir)
    verification_policy_path = case_dir / "finding_verification_policy.json"
    verification_policy = (
        json.loads(verification_policy_path.read_text(encoding="utf-8"))
        if verification_policy_path.is_file()
        else None
    )
    if mode == "FORMAL" and contract is None:
        raise ValueError(
            f"FORMAL evaluation requires case-local scientific contract for {case_id!r}"
        )

    materializer = evaluator.novel_o_materializer
    evidence_path = case_dir / "agent_ready_evidence.json"
    if materializer is None and evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(evidence, list):
            raise ValueError(
                f"Agent-ready evidence for {case_id!r} must be a JSON array"
            )
        materializer = build_agent_ready_novel_o_materializer(
            datasets_root.parent,
            metadata,
            evidence,
        )

    evaluation_mode = (
        "FORMAL_EVALUATION" if mode == "FORMAL" else "DEVELOPMENT_EVALUATION"
    )
    if contract is None and portfolio_manifest is not None:
        return build_case_evaluator(
            case_id,
            portfolio_manifest,
            evaluator,
            evaluation_manifest=evaluator.evaluation_manifest,
            unit_converter=evaluator.unit_converter,
            scientific_evaluation_contract=None,
            finding_verification_policy=verification_policy,
            evaluation_mode=evaluation_mode,
            novel_o_materializer=materializer,
        )

    return CaseEvaluator(
        evaluator.extractor,
        evaluator.semantic_matcher,
        evaluator.eligibility_judge,
        evaluator.evaluation_manifest,
        unit_converter=evaluator.unit_converter,
        finding_requirement_contract=(
            evaluator.finding_requirement_contract
            if contract is None
            else contract.finding_requirement_contract
        ),
        scientific_evaluation_contract=(
            evaluator.scientific_evaluation_contract
            if contract is None
            else contract
        ),
        finding_verification_policy=verification_policy,
        evaluation_mode=evaluation_mode,
        novel_o_materializer=materializer,
    )


def _save_pending_attempt(record: PendingCaseEvaluationRecord, run_dir: Path) -> None:
    canonical = run_dir / "pending_case_evaluation.json"
    _save_current_record(record, canonical)


def _save_pending_continuation_packet(
    record: PendingCaseEvaluationRecord,
    case_input: Any,
    metadata: CaseConstructionMetadata,
    run_dir: Path,
) -> Path:
    """Persist the production next-owner packet for one pending evaluation."""

    packet = build_pending_adjudication_packet(record, case_input, metadata)
    path = run_dir / "pending_continuation_packet.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _write_pilot_checkpoint(
    summary_path: str | Path,
    *,
    evaluation_root: Path,
    finalized_records: int = 0,
    pending_records: Sequence[PendingCaseEvaluationRecord] = (),
    infrastructure_failure_records: int = 0,
    evaluator: CaseEvaluator,
) -> dict[str, object]:
    """Atomically summarize every currently selected PILOT record.

    Selected-case shards share one immutable evaluator manifest.  Rebuilding
    this summary from atomic current pointers makes interruption/resume safe
    and prevents the last shard from reporting a misleading case_count=1.
    The counter arguments are accepted for call-site clarity but the
    persisted current records are authoritative.
    """

    del finalized_records, pending_records, infrastructure_failure_records
    records: list[CaseEvaluationRecord | PendingCaseEvaluationRecord] = []
    for pointer in sorted(evaluation_root.rglob("current_evaluation_pointer.json")):
        current = resolve_current_evaluation_record_path(pointer.parent)
        if current is not None:
            records.append(load_case_evaluation_record(current))
    pending = [item for item in records if isinstance(item, PendingCaseEvaluationRecord)]
    finalized = [item for item in records if isinstance(item, CaseEvaluationRecord)]
    infrastructure = sum(
        item.result.run_status == RunStatus.INFRASTRUCTURE_INVALID
        for item in finalized
    )
    payload: dict[str, object] = {
        "status": "PILOT / NOT FORMAL BENCHMARK RESULT",
        "mode": "PILOT",
        "case_count": 0,
        "finalized_trial_count": len(finalized),
        "terminal_record_count": len(finalized),
        "persisted_trial_count": len(records),
        "scientifically_finalized_trial_count": len(finalized) - infrastructure,
        "infrastructure_invalid_count": infrastructure,
        "pending_count": len(pending),
        "evaluation_pending_count": len(pending),
        "evaluation_manifest": evaluator.evaluation_manifest.to_dict(),
        "evaluation_manifest_digest": evaluation_manifest_digest(
            evaluator.evaluation_manifest
        ),
        "benchmark_release_id": None,
        "summary": None,
        "case_aggregates": [],
        "pending_records": [item.to_dict() for item in pending],
    }
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Multiple disjoint case shards refresh the same pilot summary.  Keep the
    # final replace atomic while giving each writer its own temporary file.
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return payload


def _save_evaluator_failure(
    *,
    active_evaluator: CaseEvaluator,
    run_record: RunRecord,
    condition: str,
    efficiency: Any,
    run_dir: Path,
    error: EvaluatorBackendError,
) -> CaseEvaluationRecord:
    """Persist an evaluator/provider failure without inventing a score.

    A timeout or malformed evaluator response is an infrastructure event.  A
    status record keeps the N=1 trial table complete while the sidecar marks
    the record as retryable on a later invocation.
    """

    result = CaseEvaluationResult(
        case_id=run_record.case_id,
        condition=condition,
        run_status=RunStatus.INFRASTRUCTURE_INVALID,
        eligible_for_scientific_aggregation=False,
        metrics=None,
    )
    record = active_evaluator._attach_run_identity(  # noqa: SLF001
        active_evaluator._status_record(result, efficiency),  # noqa: SLF001
        run_record,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "case_evaluation_record.json"
    _save_current_record(record, final_path)
    repair_audits: list[Mapping[str, Any]] = []
    seen_components: set[int] = set()
    for component in (
        active_evaluator.extractor,
        active_evaluator.semantic_matcher,
        active_evaluator.eligibility_judge,
    ):
        if id(component) in seen_components:
            continue
        seen_components.add(id(component))
        audit = getattr(component, "repair_audit", None)
        if isinstance(audit, list):
            repair_audits.extend(
                dict(item) for item in audit if isinstance(item, Mapping)
            )
    provider_failure = provider_failure_info(error)
    error_text = str(error).casefold()
    retryable = not (
        isinstance(provider_failure, Mapping)
        and provider_failure.get("category") == "QUOTA"
    ) and "http 403" not in error_text
    (run_dir / "evaluator_backend_failure.json").write_text(
        json.dumps(
            {
                "record_type": "evaluator_backend_failure",
                "case_id": run_record.case_id,
                "run_id": run_record.run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "retryable": retryable,
                "provider_failure": provider_failure,
                # This is the actual raw -> normalized/repaired provenance
                # produced by the frozen evaluator backend.  It contains no
                # provider credential or HTTP header and makes a schema or
                # semantic-preservation rejection reproducible.
                "repair_audit": repair_audits,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return record


def _failure_sidecar_is_retryable(path: Path) -> bool:
    """Read the persisted retry decision without guessing from filenames."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, Mapping) and value.get("retryable") is True


def _status_record_for_evaluator_failure(
    *,
    active_evaluator: CaseEvaluator,
    run_record: RunRecord,
    condition: str,
    efficiency: Any,
) -> CaseEvaluationRecord:
    result = CaseEvaluationResult(
        case_id=run_record.case_id,
        condition=condition,
        run_status=RunStatus.INFRASTRUCTURE_INVALID,
        eligible_for_scientific_aggregation=False,
        metrics=None,
    )
    return active_evaluator._attach_run_identity(  # noqa: SLF001
        active_evaluator._status_record(result, efficiency),  # noqa: SLF001
        run_record,
    )


def _assert_record_matches_run(
    record: CaseEvaluationRecord | PendingCaseEvaluationRecord,
    run_record: RunRecord,
    case_id: str,
) -> None:
    # Pending records carry ``case_id`` directly; finalized records keep it
    # inside their immutable CaseEvaluationResult payload.
    record_case_id = (
        record.case_id
        if isinstance(record, PendingCaseEvaluationRecord)
        else record.result.case_id
    )
    if record_case_id != case_id:
        raise ValueError("stored evaluation case_id does not match the requested case")
    expected = {
        "run_id": run_record.run_id,
        "trial_index": run_record.trial_index,
        "experiment_id": run_record.experiment_id or None,
        "target_fingerprint": run_record.target_fingerprint,
        "benchmark_release_id": run_record.benchmark_release_id or None,
        "formal_mode": run_record.formal_mode,
    }
    for field_name, expected_value in expected.items():
        if getattr(record, field_name) != expected_value:
            raise ValueError(
                f"stored evaluation {field_name} does not match RunRecord"
            )


def evaluate_saved_runs(
    *,
    datasets_root: str | Path,
    case_paths: Mapping[str, str | Path],
    run_record_paths: Mapping[str, Sequence[str | Path]],
    evaluator: CaseEvaluator,
    evaluation_root: str | Path,
    summary_path: str | Path,
    mode: Literal["PILOT", "FORMAL"],
    release_manifest: BenchmarkReleaseManifest | Mapping[str, object] | None = None,
    adjudications_by_run_id: Mapping[
        str, EvaluationAdjudications | RunEvaluationAdjudication
    ] | None = None,
    finalize: bool = False,
    portfolio_manifest: str | Path | Mapping[str, object] | None = None,
    dataset_ids_by_case: Mapping[str, str] | None = None,
    retry_evaluator_failures: bool = False,
    pending_continuation_handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]]
    | PendingContinuationRegistry
    | None = None,
    max_continuation_hops: int = 8,
    selected_case_ids: Sequence[str] | None = None,
    checkpoint_summary: bool = False,
    force_retry_case_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Evaluate saved runs without confusing a pilot with a formal result.

    ``pending_continuation_handlers`` is an optional production hook.  Each
    handler receives only the blinded owner packet and may return an explicit
    ``EvaluationAdjudications`` payload or the next persisted pending record.
    The loop is bounded by ``max_continuation_hops``; no scientific judgment
    is inferred when a handler is absent or the bound is reached.
    """

    if mode not in {"PILOT", "FORMAL"}:
        raise ValueError("mode must be PILOT or FORMAL")
    if isinstance(max_continuation_hops, bool) or not isinstance(max_continuation_hops, int) or max_continuation_hops < 1:
        raise ValueError("max_continuation_hops must be a positive integer")
    if set(case_paths) != set(run_record_paths):
        raise ValueError("case_paths and run_record_paths must contain the same explicit case IDs")
    selected = None if selected_case_ids is None else set(selected_case_ids)
    force_retry = set(force_retry_case_ids or ())
    if selected is not None:
        unknown_selected = selected - set(case_paths)
        if unknown_selected:
            raise ValueError(
                "selected_case_ids contains unknown cases: "
                + ", ".join(sorted(unknown_selected))
            )
    unknown_retry = force_retry - set(case_paths)
    if unknown_retry:
        raise ValueError(
            "force_retry_case_ids contains unknown cases: "
            + ", ".join(sorted(unknown_retry))
        )
    release = (
        None
        if release_manifest is None
        else (
            release_manifest
            if isinstance(release_manifest, BenchmarkReleaseManifest)
            else BenchmarkReleaseManifest.model_validate(release_manifest)
        )
    )
    if mode == "FORMAL" and release is None:
        raise ValueError("FORMAL evaluation requires an explicit BenchmarkReleaseManifest")

    root = Path(datasets_root).resolve()
    continuation_registry = (
        None
        if pending_continuation_handlers is None
        else (
            pending_continuation_handlers
            if isinstance(pending_continuation_handlers, PendingContinuationRegistry)
            else PendingContinuationRegistry(pending_continuation_handlers)
        )
    )
    if continuation_registry is not None:
        continuation_manifest = continuation_registry.execution_manifest
        current_manifest = evaluator.evaluation_manifest.continuation_execution_manifest
        if current_manifest and dict(current_manifest) != continuation_manifest:
            raise ValueError(
                "evaluator manifest is already bound to a different continuation execution manifest"
            )
        if not current_manifest:
            # Bind the executable continuation identity before freezing the
            # collection manifest.  No handler can therefore be swapped in
            # under the same evaluation digest.
            evaluator.evaluation_manifest = replace(
                evaluator.evaluation_manifest,
                continuation_execution_manifest=continuation_manifest,
            )
    evaluation_root = Path(evaluation_root)
    frozen_manifest_digest = _freeze_evaluation_manifest(evaluation_root, evaluator)
    trial_groups: dict[tuple[str, str, str, str, bool], list[TrialEvaluation]] = defaultdict(list)
    aggregates: list[CaseAggregate] = []
    pending_records: list[PendingCaseEvaluationRecord] = []
    seen_run_ids: set[str] = set()
    finalized_records = 0
    infrastructure_failure_records = 0

    if selected is not None and len(selected) == 1:
        # Persist an accounted infrastructure status at process start.  If
        # the evaluator provider hangs or this shard is externally killed,
        # the 28-case report remains complete and explicitly retryable rather
        # than silently omitting the case.  A later pending/final result
        # atomically replaces this provisional status.
        selected_case_id = next(iter(selected))
        selected_run_paths = run_record_paths[selected_case_id]
        if len(selected_run_paths) == 1:
            provisional_run = RunRecord.load_json(selected_run_paths[0])
            provisional_dir = (
                evaluation_root
                / _safe_component(provisional_run.experiment_id or "pilot")
                / _safe_component(provisional_run.target_fingerprint or "no-target")
                / _safe_component(selected_case_id)
                / f"trial-{provisional_run.trial_index}"
                / _safe_component(provisional_run.run_id)
            )
            if _current_record_path(provisional_dir) is None:
                provisional_metadata = CaseConstructionMetadata.model_validate_json(
                    (
                        Path(case_paths[selected_case_id]).resolve()
                        / "case_construction_metadata.json"
                    ).read_text(encoding="utf-8")
                )
                provisional_condition = primary_case_type(provisional_metadata)
                provisional_efficiency = EfficiencyObservation(
                    input_tokens=provisional_run.input_tokens,
                    output_tokens=provisional_run.output_tokens,
                    model_turn_count=provisional_run.model_turn_count,
                    tool_call_count=getattr(
                        provisional_run,
                        "tool_call_count",
                        provisional_run.python_execution_count,
                    ),
                    python_execution_count=provisional_run.python_execution_count,
                    wall_clock_time=provisional_run.wall_clock_time,
                    provider_reported_cost=provisional_run.provider_reported_cost,
                )
                provisional = _status_record_for_evaluator_failure(
                    active_evaluator=evaluator,
                    run_record=provisional_run,
                    condition=provisional_condition,
                    efficiency=provisional_efficiency,
                )
                _save_current_record(
                    provisional,
                    provisional_dir / "case_evaluation_record.json",
                )

    for case_id, case_dir_value in case_paths.items():
        if selected is not None and case_id not in selected:
            continue
        case_dir = Path(case_dir_value).resolve()
        case_input_path = case_dir / "case_input.json"
        metadata_path = case_dir / "case_construction_metadata.json"
        ground_truth_path = case_dir / "ground_truth.json"
        loaded = CaseLoader(root).load(case_input_path)
        dataset_id = _dataset_id_for_case(
            case_dir,
            root,
            None if dataset_ids_by_case is None else dataset_ids_by_case.get(case_id),
        )
        dataset_manifest = DatasetManifest.model_validate_json(
            (root / dataset_id / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        dataset_manifest.validate_against_case(loaded.case)
        metadata = CaseConstructionMetadata.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        active_evaluator = _configure_case_evaluator(
            evaluator=evaluator,
            case_id=case_id,
            case_dir=case_dir,
            metadata=metadata,
            datasets_root=root,
            mode=mode,
            portfolio_manifest=portfolio_manifest,
        )
        ground_truth = GroundTruth.model_validate_json(
            ground_truth_path.read_text(encoding="utf-8")
        )
        (case_dir / "case_context.json").read_text(encoding="utf-8")
        condition, _ = active_evaluator._validate_case(  # noqa: SLF001
            case_id, metadata, ground_truth
        )

        release_case = None
        if release is not None:
            release_case = release.case_for_id(case_id)
            if release_case.dataset_id != dataset_id:
                raise ValueError(f"release dataset mismatch for case {case_id!r}")
            if mode == "FORMAL":
                validate_formal_release_case(release_case, loaded, dataset_manifest)

        records = run_record_paths[case_id]
        if not records:
            raise ValueError(f"no run records supplied for case {case_id!r}")
        for run_path_value in records:
            run_record = RunRecord.load_json(run_path_value)
            # For the userstudy's host-network protocol, numerical defaults
            # may be recovered from successful model code, not guessed from
            # incomplete prose. Formal and historical routes are unchanged.
            if mode == "PILOT" and run_record.runtime_profile_id == "flow-python-host-network-v1":
                evidence_file = case_dir / "agent_ready_evidence.json"
                if evidence_file.is_file():
                    active_evaluator.novel_o_materializer = build_agent_ready_novel_o_materializer(
                        root.parent, metadata, json.loads(evidence_file.read_text()),
                        execution_trajectory=Path(run_path_value).parent / "trajectory.json",
                        open_field_reference=next((record for record in (
                            active_evaluator.scientific_evaluation_contract.G_of_O
                            if active_evaluator.scientific_evaluation_contract is not None else ()
                        ) if isinstance(record, Mapping) and record.get("execution", {}).get("G_of_O", {}).get(
                            "field_selection_policy", {}).get("mode") == "explicit_candidate_field_v1"), None),
                    )
                    from flowintentbench.deterministic_materialization import supplemental_high_speed_evidence
                    from flowintentbench.qualified_scientific_cases import supplemental_density_evidence
                    from flowintentbench.descriptive_statistics import requested_numeric_thresholds, supplemental_kitchen_evidence
                    answer_path = Path(run_path_value).parent / "final_answer.md"
                    thresholds = requested_numeric_thresholds(answer_path.read_text()) if answer_path.is_file() else ()
                    import re
                    sensitivity = bool(re.search(r"stabl|sensitiv|robust|perturb", answer_path.read_text(), re.I)) if answer_path.is_file() else False
                    supplemental_cache = {}
                    def supplemental(record, *, _dataset=metadata.dataset_id, _cache=supplemental_cache, _thresholds=thresholds, _sensitivity=sensitivity):
                        key = json.dumps(record, sort_keys=True)
                        if key not in _cache:
                            _cache[key] = (supplemental_density_evidence(root.parent, _dataset, record, requested_thresholds=_thresholds, include_sensitivity=_sensitivity)
                                           if _dataset == "Combustor" else
                                           supplemental_kitchen_evidence(root.parent, record, requested_thresholds=_thresholds) if _dataset == "Kitchen" else
                                           supplemental_high_speed_evidence(root.parent, _dataset, record, requested_thresholds=_thresholds))
                        return _cache[key]
                    active_evaluator.supplemental_materializer = supplemental
            if run_record.run_id in seen_run_ids:
                raise ValueError(f"duplicate RunRecord run_id: {run_record.run_id}")
            seen_run_ids.add(run_record.run_id)
            if run_record.case_id != case_id:
                raise ValueError(
                    f"run record case_id {run_record.case_id!r} does not match {case_id!r}"
                )
            if mode == "FORMAL":
                if not run_record.formal_mode:
                    raise ValueError("FORMAL evaluation requires RunRecord formal_mode=True")
                if not run_record.benchmark_release_id:
                    raise ValueError("FORMAL evaluation requires benchmark_release_id")
                if not run_record.experiment_id:
                    raise ValueError("FORMAL evaluation requires experiment_id")
                if not run_record.target_fingerprint:
                    raise ValueError("FORMAL evaluation requires target_fingerprint")
                if run_record.benchmark_release_id != release.release_id:  # type: ignore[union-attr]
                    raise ValueError("RunRecord benchmark_release_id does not match release manifest")

            efficiency = EfficiencyObservation(
                input_tokens=run_record.input_tokens,
                output_tokens=run_record.output_tokens,
                model_turn_count=run_record.model_turn_count,
                tool_call_count=getattr(
                    run_record,
                    "tool_call_count",
                    run_record.python_execution_count,
                ),
                python_execution_count=run_record.python_execution_count,
                wall_clock_time=run_record.wall_clock_time,
                provider_reported_cost=run_record.provider_reported_cost,
            )

            run_dir = (
                evaluation_root
                / _safe_component(run_record.experiment_id or "pilot")
                / _safe_component(run_record.target_fingerprint or "no-target")
                / _safe_component(case_id)
                / f"trial-{run_record.trial_index}"
                / _safe_component(run_record.run_id)
            )
            if mode == "PILOT" and run_record.runtime_profile_id == "flow-python-host-network-v1":
                # Retain validated API judgments across infrastructure retries,
                # separately for every subject observation. No cross-round
                # response reuse and no caching of failures or malformed JSON.
                components = (active_evaluator.extractor, active_evaluator.semantic_matcher,
                              active_evaluator.eligibility_judge)
                cache_context = {
                    "evaluation_manifest": evaluation_manifest_digest(active_evaluator.evaluation_manifest),
                    "case_files": {
                        name: hashlib.sha256((case_dir / name).read_bytes()).hexdigest()
                        for name in ("case_input.json", "case_construction_metadata.json", "ground_truth.json",
                                     "scientific_evaluation_contract.json", "finding_verification_policy.json")
                        if (case_dir / name).is_file()
                    },
                }
                for component in {id(c): c for c in components}.values():
                    if isinstance(component, StructuredEvaluatorBackend):
                        component.enable_response_cache(run_dir / "validated_responses.sqlite3", context_identity=cache_context)
            final_path = run_dir / "case_evaluation_record.json"
            failure_path = run_dir / "evaluator_backend_failure.json"
            retry_failure = retry_evaluator_failures and (
                (failure_path.exists() and _failure_sidecar_is_retryable(failure_path))
                or case_id in force_retry
            )
            # Do not archive the current record before invoking a retry.  The
            # writer owns the archive-and-pointer transaction: if the retry
            # provider fails, the old pointer and bytes remain the current
            # selection; if it succeeds, ``_save_current_record`` archives
            # exactly the selected old bytes before committing the new record.
            current_path = None if retry_failure else _current_record_path(run_dir)
            current_record = (
                None
                if current_path is None
                else load_case_evaluation_record(current_path)
            )
            if (
                selected is not None
                and case_id in selected
                and isinstance(current_record, CaseEvaluationRecord)
                and current_record.result.run_status
                == RunStatus.INFRASTRUCTURE_INVALID
            ):
                # A selected shard is an explicit retry request.  This also
                # replaces the provisional interruption-safe status written
                # at shard start, without requiring a failure sidecar that an
                # externally killed process could not have produced.
                retry_failure = True
                current_path = None
                current_record = None
            if isinstance(current_record, CaseEvaluationRecord):
                record = current_record
                _assert_record_matches_run(record, run_record, case_id)
                if record.evaluation_manifest_digest != evaluation_manifest_digest(
                    active_evaluator.evaluation_manifest
                ):
                    raise ValueError(
                        "stored final evaluation uses a different evaluation manifest"
                    )
                # A successful retry is authoritative even if an interrupted
                # writer left the old retryable sidecar behind.  Do not let a
                # stale failure marker force every later replay to call the
                # provider again.
                if (
                    record.result.run_status != RunStatus.INFRASTRUCTURE_INVALID
                    and failure_path.is_file()
                ):
                    failure_path.unlink()
            else:
                adjudications = _resolve_run_adjudication(
                    run_record, adjudications_by_run_id
                )
                pending_path = (
                    current_path
                    if isinstance(current_record, PendingCaseEvaluationRecord)
                    else _pending_artifact_path(run_dir)
                )
                if (
                    failure_path.exists()
                    and not retry_failure
                    and not pending_path.exists()
                ):
                    # A prior invocation already established a retryable
                    # evaluator failure but was interrupted before its status
                    # record could be written.  Rehydrate only the explicit
                    # infrastructure status; never manufacture scientific
                    # metrics from the failure.
                    record = _status_record_for_evaluator_failure(
                        active_evaluator=active_evaluator,
                        run_record=run_record,
                        condition=condition,
                        efficiency=efficiency,
                    )
                    _save_current_record(
                        record, final_path
                    )
                else:
                    try:
                        if isinstance(current_record, PendingCaseEvaluationRecord):
                            pending = current_record
                            if not isinstance(pending, PendingCaseEvaluationRecord):
                                raise ValueError("pending case evaluation file contains a finalized record")
                            _assert_record_matches_run(pending, run_record, case_id)
                            if adjudications is None:
                                record = pending
                            else:
                                record = active_evaluator.finalize_pending_evaluation(
                                    pending,
                                    loaded.case,
                                    metadata,
                                    ground_truth,
                                    adjudications=adjudications,
                                )
                                if isinstance(record, PendingCaseEvaluationRecord):
                                    _save_pending_attempt(record, run_dir)
                                else:
                                    _save_current_record(
                                        record, final_path
                                    )
                        else:
                            record = active_evaluator.evaluate_run_record(
                                run_record,
                                loaded.case,
                                metadata,
                                ground_truth,
                                adjudications=adjudications,
                            )
                            if isinstance(record, PendingCaseEvaluationRecord):
                                _save_pending_attempt(record, run_dir)
                            else:
                                _save_current_record(
                                    record, final_path
                                )
                    except EvaluatorBackendError as exc:
                        record = _save_evaluator_failure(
                            active_evaluator=active_evaluator,
                            run_record=run_record,
                            condition=condition,
                            efficiency=efficiency,
                            run_dir=run_dir,
                            error=exc,
                        )
                    else:
                        if failure_path.exists() and not isinstance(
                            record, PendingCaseEvaluationRecord
                        ):
                            failure_path.unlink()
            if (
                isinstance(record, CaseEvaluationRecord)
                and record.result.run_status == RunStatus.INFRASTRUCTURE_INVALID
            ):
                infrastructure_failure_records += 1
            if isinstance(record, PendingCaseEvaluationRecord):
                continuation_hops = 0
                continuation_cycle_detected = False
                continuation_infrastructure_failure = False
                continuation_events: list[dict[str, Any]] = []
                continuation_states: set[tuple[str, str]] = set()
                # The primitive dispatcher remains one-hop and side-effect
                # free; production orchestration may apply it repeatedly in a
                # bounded loop so novel-O completion -> materialization ->
                # adjudication can finish in one invocation.
                while isinstance(record, PendingCaseEvaluationRecord) and continuation_registry and continuation_hops < max_continuation_hops:
                    route = route_pending_adjudication(record, loaded.case, metadata)
                    request_id = str(
                        record.continuation_context.get("continuation_request_id", "")
                    )
                    # Legacy novel-O records may predate request IDs; retain a
                    # deterministic fallback for those records while all new
                    # pending routes use request-specific identity.
                    if not request_id:
                        # Legacy records have no explicit request identity.
                        # Derive a non-semantic compatibility key from the
                        # complete blinded packet rather than collapsing all
                        # requests of one pending type into one cycle.
                        request_id = "legacy:" + hashlib.sha256(
                            json.dumps(
                                route["packet"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest()
                    state = (str(route["pending_owner"]), request_id)
                    if state in continuation_states:
                        continuation_cycle_detected = True
                        continuation_events.append({
                            "hop": continuation_hops,
                            "pending_type": record.pending_type,
                            "pending_owner": route["pending_owner"],
                            "next_action": route["next_action"],
                            "dispatch_status": "CONTINUATION_CYCLE_DETECTED",
                            "state": list(state),
                        })
                        break
                    continuation_states.add(state)
                    try:
                        if mode == "PILOT" and run_record.runtime_profile_id == "flow-python-host-network-v1":
                            dispatch = dispatch_with_checkpoint(
                                continuation_registry, record, loaded.case, metadata,
                                route=route, path=run_dir / "validated_continuations.sqlite3",
                                context=cache_context,
                            )
                        else:
                            dispatch = continuation_registry.dispatch(record, loaded.case, metadata)
                    except EvaluatorBackendError as exc:
                        # A continuation provider failure is an infrastructure
                        # event, not scientific INVALID and not an uncaught
                        # process error.  Persist the current trial as
                        # retryable so a later shard can resume it safely.
                        failed_pending_type = record.pending_type
                        record = _save_evaluator_failure(
                            active_evaluator=active_evaluator,
                            run_record=run_record,
                            condition=condition,
                            efficiency=efficiency,
                            run_dir=run_dir,
                            error=exc,
                        )
                        continuation_infrastructure_failure = True
                        continuation_events.append({
                            "hop": continuation_hops,
                            "pending_type": failed_pending_type,
                            "dispatch_status": "INFRASTRUCTURE_INVALID",
                            "error": str(exc),
                        })
                        break
                    except Exception as exc:
                        wrapped = EvaluatorBackendError(
                            "continuation handler failed for "
                            f"{type(exc).__name__}: {exc}"
                        )
                        failed_pending_type = record.pending_type
                        record = _save_evaluator_failure(
                            active_evaluator=active_evaluator,
                            run_record=run_record,
                            condition=condition,
                            efficiency=efficiency,
                            run_dir=run_dir,
                            error=wrapped,
                        )
                        continuation_infrastructure_failure = True
                        continuation_events.append({
                            "hop": continuation_hops,
                            "pending_type": failed_pending_type,
                            "dispatch_status": "INFRASTRUCTURE_INVALID",
                            "error": str(wrapped),
                            "error_type": type(exc).__name__,
                        })
                        break
                    continuation_hops += 1
                    output = dispatch.get("continuation_output")
                    continuation_events.append({
                        "hop": continuation_hops,
                        "pending_type": record.pending_type,
                        "pending_owner": dispatch.get("pending_owner"),
                        "next_action": dispatch.get("next_action"),
                        "dispatch_status": dispatch.get("dispatch_status"),
                        "output_type": type(output).__name__ if output is not None else None,
                    })
                    if isinstance(output, PendingCaseEvaluationRecord):
                        try:
                            _assert_record_matches_run(output, run_record, case_id)
                            if output.evaluation_manifest_digest != evaluation_manifest_digest(active_evaluator.evaluation_manifest):
                                raise ValueError("pending continuation returned a different evaluation manifest")
                        except Exception as exc:
                            failed_pending_type = record.pending_type
                            wrapped = EvaluatorBackendError(
                                "pending continuation returned an invalid pending record for "
                                f"{failed_pending_type}: {type(exc).__name__}: {exc}"
                            )
                            record = _save_evaluator_failure(
                                active_evaluator=active_evaluator,
                                run_record=run_record,
                                condition=condition,
                                efficiency=efficiency,
                                run_dir=run_dir,
                                error=wrapped,
                            )
                            continuation_infrastructure_failure = True
                            continuation_events.append({
                                "hop": continuation_hops,
                                "pending_type": failed_pending_type,
                                "dispatch_status": "INFRASTRUCTURE_INVALID",
                                "error": str(wrapped),
                                "error_type": type(exc).__name__,
                            })
                            break
                        record = output
                        _save_pending_attempt(record, run_dir)
                        continue
                    continuation_adjudications: EvaluationAdjudications | None = None
                    if isinstance(output, EvaluationAdjudications):
                        continuation_adjudications = output
                    elif isinstance(output, SemanticMatchResolution):
                        continuation_adjudications = EvaluationAdjudications(
                            semantic_resolutions=(output,)
                        )
                    elif isinstance(output, UnitFrameResolution):
                        continuation_adjudications = EvaluationAdjudications(
                            unit_frame_resolutions=(output,)
                        )
                    elif isinstance(output, Mapping) and set(output).intersection(
                        {
                            "novel_operationalization",
                            "novel_findings",
                            "novel_finding_roles",
                            "semantic_resolutions",
                            "unit_frame_resolutions",
                        }
                    ):
                        try:
                            continuation_adjudications = EvaluationAdjudications.from_dict(output)
                        except Exception as exc:
                            failed_pending_type = record.pending_type
                            wrapped = EvaluatorBackendError(
                                "pending continuation adjudication payload is invalid for "
                                f"{failed_pending_type}: {type(exc).__name__}: {exc}"
                            )
                            record = _save_evaluator_failure(
                                active_evaluator=active_evaluator,
                                run_record=run_record,
                                condition=condition,
                                efficiency=efficiency,
                                run_dir=run_dir,
                                error=wrapped,
                            )
                            continuation_infrastructure_failure = True
                            continuation_events.append({
                                "hop": continuation_hops,
                                "pending_type": failed_pending_type,
                                "dispatch_status": "INFRASTRUCTURE_INVALID",
                                "error": str(wrapped),
                                "error_type": type(exc).__name__,
                            })
                            break
                    if continuation_adjudications is None:
                        break
                    try:
                        continued = active_evaluator.finalize_pending_evaluation(
                            record,
                            loaded.case,
                            metadata,
                            ground_truth,
                            adjudications=continuation_adjudications,
                        )
                    except EvaluatorBackendError as exc:
                        failed_pending_type = record.pending_type
                        record = _save_evaluator_failure(
                            active_evaluator=active_evaluator,
                            run_record=run_record,
                            condition=condition,
                            efficiency=efficiency,
                            run_dir=run_dir,
                            error=exc,
                        )
                        continuation_infrastructure_failure = True
                        continuation_events.append({
                            "hop": continuation_hops,
                            "pending_type": failed_pending_type,
                            "dispatch_status": "INFRASTRUCTURE_INVALID",
                            "error": str(exc),
                        })
                        break
                    except Exception as exc:
                        failed_pending_type = record.pending_type
                        wrapped = EvaluatorBackendError(
                            "pending continuation finalization failed for "
                            f"{failed_pending_type}: {type(exc).__name__}: {exc}"
                        )
                        record = _save_evaluator_failure(
                            active_evaluator=active_evaluator,
                            run_record=run_record,
                            condition=condition,
                            efficiency=efficiency,
                            run_dir=run_dir,
                            error=wrapped,
                        )
                        continuation_infrastructure_failure = True
                        continuation_events.append({
                            "hop": continuation_hops,
                            "pending_type": failed_pending_type,
                            "dispatch_status": "INFRASTRUCTURE_INVALID",
                            "error": str(wrapped),
                            "error_type": type(exc).__name__,
                        })
                        break
                    record = continued
                    if isinstance(record, PendingCaseEvaluationRecord):
                        _save_pending_attempt(record, run_dir)
                if continuation_events:
                    state_path = run_dir / "pending_continuation_state.json"
                    state_tmp = state_path.with_name(state_path.name + ".tmp")
                    state_tmp.write_text(
                        json.dumps({
                            "max_continuation_hops": max_continuation_hops,
                            "hops_executed": continuation_hops,
                            "bounded": continuation_hops >= max_continuation_hops,
                            "cycle_detected": continuation_cycle_detected,
                            "events": continuation_events,
                        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(state_tmp, state_path)
                if isinstance(record, PendingCaseEvaluationRecord):
                    _save_pending_continuation_packet(
                        record, loaded.case, metadata, run_dir
                    )
                    pending_records.append(record)
                    if checkpoint_summary:
                        _write_pilot_checkpoint(
                            summary_path,
                            evaluation_root=evaluation_root,
                            finalized_records=finalized_records,
                            pending_records=pending_records,
                            infrastructure_failure_records=infrastructure_failure_records,
                            evaluator=evaluator,
                        )
                    continue
                if continuation_infrastructure_failure:
                    infrastructure_failure_records += 1
                _save_current_record(record, final_path)
            finalized_records += 1
            if checkpoint_summary:
                _write_pilot_checkpoint(
                    summary_path,
                    evaluation_root=evaluation_root,
                    finalized_records=finalized_records,
                    pending_records=pending_records,
                    infrastructure_failure_records=infrastructure_failure_records,
                    evaluator=evaluator,
                )
            if mode == "FORMAL" or finalize:
                trial = TrialEvaluation.from_evaluation_record(record)
                key = (
                    case_id,
                    trial.experiment_id or "",
                    trial.target_fingerprint or "",
                    trial.benchmark_release_id or "",
                    bool(trial.formal_mode),
                )
                trial_groups[key].append(trial)

    if mode == "FORMAL" and pending_records:
        raise ValueError("FORMAL aggregation is blocked by pending evaluation records")

    if mode == "FORMAL":
        for (case_id, _experiment_id, _target, _release, _formal), trials in sorted(trial_groups.items()):
            aggregates.append(aggregate_trials_for_case(trials))
        summary = aggregate_cases(aggregates)
        status = "FORMAL EVALUATION SUMMARY"
    else:
        # A pilot is intentionally inspectable at N=1.  Three-trial case
        # aggregation is only attempted when the caller explicitly requests it.
        if finalize:
            for (_case_id, _experiment_id, _target, _release, _formal), trials in sorted(trial_groups.items()):
                if len(trials) == 3:
                    aggregates.append(aggregate_trials_for_case(trials))
        summary = None
        status = "PILOT / NOT FORMAL BENCHMARK RESULT"

    payload: dict[str, object] = {
        "status": status,
        "mode": mode,
        "case_count": len(aggregates),
        "finalized_trial_count": finalized_records,
        # Unambiguous status counters.  ``finalized_trial_count`` is retained
        # for backwards-compatible consumers, but includes terminal
        # infrastructure-invalid records; scientific consumers should use
        # these explicit counters instead.
        "terminal_record_count": finalized_records,
        "persisted_trial_count": finalized_records + len(pending_records),
        "scientifically_finalized_trial_count": max(
            0, finalized_records - infrastructure_failure_records
        ),
        "infrastructure_invalid_count": infrastructure_failure_records,
        "pending_count": len(pending_records),
        "evaluation_pending_count": len(pending_records),
        "evaluation_manifest": evaluator.evaluation_manifest.to_dict(),
        "evaluation_manifest_digest": frozen_manifest_digest,
        "benchmark_release_id": None if release is None else release.release_id,
        "summary": summary,
        "case_aggregates": [aggregate.to_dict() for aggregate in aggregates],
        "pending_records": [record.to_dict() for record in pending_records],
    }
    if mode == "PILOT" and checkpoint_summary:
        return _write_pilot_checkpoint(
            summary_path,
            evaluation_root=evaluation_root,
            evaluator=evaluator,
        )
    destination = Path(summary_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("PILOT", "FORMAL"), required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--case", dest="cases", action="append", required=True, help="CASE_ID=CASE_ARTIFACT_DIR")
    parser.add_argument(
        "--only-case",
        dest="only_cases",
        action="append",
        default=[],
        help="evaluate only this supplied case ID; may be repeated for resumable shards",
    )
    parser.add_argument(
        "--checkpoint-summary",
        action="store_true",
        help="atomically refresh the PILOT summary after each selected case",
    )
    parser.add_argument("--run", dest="runs", action="append", required=True, help="CASE_ID=RUN_RECORD_JSON")
    parser.add_argument(
        "--dataset-id",
        dest="dataset_ids",
        action="append",
        default=[],
        help="optional explicit path-independent dataset binding CASE_ID=DATASET_ID",
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--adjudication", dest="adjudications", action="append", default=[], help="RUN_ID=RUN_ADJUDICATION_JSON")
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--evaluator-timeout-seconds",
        type=float,
        default=300.0,
        help="per-request timeout for the separately configured evaluator backend",
    )
    parser.add_argument(
        "--evaluator-transport-attempts",
        type=int,
        default=4,
        help="attempts for transient evaluator transport/429/5xx failures",
    )
    parser.add_argument(
        "--evaluator-retry-backoff-seconds",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--evaluator-rate-limit-backoff-seconds",
        type=float,
        default=15.0,
        help="base delay for HTTP 429; Retry-After takes precedence",
    )
    parser.add_argument(
        "--evaluator-rate-limit-max-backoff-seconds",
        type=float,
        default=60.0,
        help="maximum local exponential delay for HTTP 429",
    )
    parser.add_argument(
        "--evaluator-retry-jitter-seconds",
        type=float,
        default=0.0,
        help="deterministic bounded jitter added to evaluator transport retries",
    )
    parser.add_argument("--finalize-pilot", action="store_true")
    parser.add_argument(
        "--retry-evaluator-failures",
        action="store_true",
        help="retry saved evaluator-backend failure records instead of reusing them",
    )
    parser.add_argument(
        "--force-retry-case",
        dest="force_retry_cases",
        action="append",
        default=[],
        help="retry a selected provisional infrastructure record even if interruption prevented its sidecar",
    )
    parser.add_argument("--portfolio-manifest", type=Path, default=None, help="Frozen family manifest used to auto-load F2 finding contracts")
    parser.add_argument(
        "--require-reusable-responses",
        action="store_true",
        help="fail closed unless saved responses match the two explicit case manifests",
    )
    parser.add_argument("--collected-case-manifest", type=Path, default=None)
    parser.add_argument(
        "--expected-collected-case-manifest-sha256",
        default=None,
        help="optional digest recorded by collection_state.json for the persisted model-visible manifest",
    )
    parser.add_argument("--candidate-case-manifest", type=Path, default=None)
    parser.add_argument(
        "--pending-handlers-json",
        type=Path,
        default=None,
        help="optional JSON object mapping pending owner names to module:function handlers; omitted means fail-closed pending",
    )
    parser.add_argument(
        "--max-continuation-hops",
        type=int,
        default=8,
        help="bounded number of owner continuations to apply to one pending trial (cycle detection is also enforced)",
    )
    args = parser.parse_args()
    if (
        (args.collected_case_manifest is not None or args.candidate_case_manifest is not None)
        and not args.require_reusable_responses
    ):
        parser.error(
            "--collected-case-manifest/--candidate-case-manifest require "
            "--require-reusable-responses"
        )
    case_paths = _parse_assignments(args.cases, option="--case")
    grouped_runs = _parse_run_assignments(args.runs)
    try:
        dataset_ids_by_case = _parse_dataset_assignments(args.dataset_ids)
    except ValueError as exc:
        parser.error(str(exc))
    unknown_dataset_bindings = set(dataset_ids_by_case) - set(case_paths)
    if unknown_dataset_bindings:
        parser.error(
            "--dataset-id contains cases not supplied via --case: "
            + ", ".join(sorted(unknown_dataset_bindings))
        )
    adjudications = {}
    if args.adjudications:
        for run_id, path in _parse_assignments(
            args.adjudications, option="--adjudication"
        ).items():
            wrapper = load_run_evaluation_adjudication(path)
            if wrapper.run_id != run_id:
                raise ValueError("--adjudication key does not match wrapper run_id")
            adjudications[run_id] = wrapper
    release = (
        None
        if args.release_manifest is None
        else BenchmarkReleaseManifest.model_validate_json(
            args.release_manifest.read_text(encoding="utf-8")
        )
    )
    try:
        config = (
            None
            if args.model_config_json is None
            else json.loads(args.model_config_json)
        )
        if config is not None and not isinstance(config, dict):
            raise ValueError("--model-config-json must contain a JSON object")
        runtime = resolve_provider_configuration(
            role="evaluator",
            config_path=args.server_config,
            provider=args.provider,
            base_url=args.base_url,
            model_id=args.model_id,
            model_configuration=config,
        )
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    target = EvaluationTarget(runtime.provider, runtime.model_id, runtime.model_configuration)
    try:
        evaluator_api_key = resolve_api_key(
            runtime,
            explicit_api_key=args.api_key,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    deadline_monotonic = None
    deadline_value = os.environ.get("FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC")
    if deadline_value:
        try:
            deadline_monotonic = float(deadline_value)
        except ValueError:
            parser.error(
                "FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC must be numeric"
            )
        if deadline_monotonic <= time.monotonic():
            parser.error("FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC has expired")
    backend = OpenAICompatibleEvaluatorBackend(
        target,
        api_key=evaluator_api_key,
        api_key_env=runtime.api_key_env,
        base_url=runtime.base_url,
        timeout_seconds=args.evaluator_timeout_seconds,
        max_transport_attempts=args.evaluator_transport_attempts,
        retry_backoff_seconds=args.evaluator_retry_backoff_seconds,
        rate_limit_backoff_seconds=args.evaluator_rate_limit_backoff_seconds,
        rate_limit_max_backoff_seconds=args.evaluator_rate_limit_max_backoff_seconds,
        retry_jitter_seconds=args.evaluator_retry_jitter_seconds,
        deadline_monotonic=deadline_monotonic,
    )
    evaluator = CaseEvaluator(backend, backend, backend, backend.manifest())
    if args.require_reusable_responses:
        if args.collected_case_manifest is None or args.candidate_case_manifest is None:
            parser.error(
                "--require-reusable-responses requires --collected-case-manifest "
                "and --candidate-case-manifest"
            )
        try:
            reuse_audit = require_reusable_model_responses(
                run_records=[path for paths in grouped_runs.values() for path in paths],
                collected_case_manifest=args.collected_case_manifest,
                candidate_case_manifest=args.candidate_case_manifest,
                expected_collected_manifest_sha256=(
                    args.expected_collected_case_manifest_sha256
                ),
            )
        except (OSError, ValueError, TypeError) as exc:
            parser.error(f"saved-response reuse preflight failed: {exc}")
        audit_path = Path(args.evaluation_root) / "model_response_reuse_audit.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(reuse_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    pending_registry = None
    if args.pending_handlers_json is not None:
        try:
            handler_spec = json.loads(args.pending_handlers_json.read_text(encoding="utf-8"))
            pending_registry = load_pending_continuation_registry(handler_spec)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(f"invalid --pending-handlers-json: {exc}")
    portfolio_manifest = args.portfolio_manifest
    if args.mode == "FORMAL" and portfolio_manifest is None:
        candidate_manifest = ROOT / "artifacts/reference/scientific_portfolio/concept_family_inventory.json"
        if candidate_manifest.is_file():
            portfolio_manifest = candidate_manifest
    evaluate_saved_runs(
        datasets_root=args.datasets_root,
        case_paths=case_paths,
        run_record_paths=grouped_runs,
        evaluator=evaluator,
        evaluation_root=args.evaluation_root,
        summary_path=args.summary,
        mode=args.mode,
        release_manifest=release,
        adjudications_by_run_id=adjudications,
        finalize=args.finalize_pilot,
        portfolio_manifest=portfolio_manifest,
        dataset_ids_by_case=dataset_ids_by_case,
        retry_evaluator_failures=args.retry_evaluator_failures,
        pending_continuation_handlers=pending_registry,
        max_continuation_hops=args.max_continuation_hops,
        selected_case_ids=args.only_cases or None,
        checkpoint_summary=args.checkpoint_summary,
        force_retry_case_ids=args.force_retry_cases or None,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
