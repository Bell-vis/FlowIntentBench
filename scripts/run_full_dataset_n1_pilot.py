"""Run the explicit manifest, N=1 integration pilot.

This is an orchestration layer only.  Case loading, Python runtime lifecycle,
provider calls, trajectory capture, and RunRecord semantics remain owned by
``BenchmarkRunner``.  No directory discovery or formal N=3 aggregation is
performed here.
"""

from __future__ import annotations

from typing import Sequence

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    AgentProfile,
    BenchmarkRunner,
    CaseLoader,
    DatasetManifest,
    EvaluationTarget,
    OpenAIChatCompletionsAdapter,
    RunRecord,
    RunStatus,
    ThirdPartyResponsesAdapter,
    code_data_version,
    evaluation_target_fingerprint,
    resolve_provider_configuration,
    resolve_api_key,
    validate_reasoning_effort,
    load_agent_profile,
    effective_agent_config,
    efficiency_observation_from_run_record,
    reject_profile_identity_overrides,
    classify_provider_failure,
    provider_failure_info,
)
from flowintentbench.pilot_readiness import (
    UPSTREAM_NOT_EVALUATED,
    build_case_execution_status,
    build_development_snapshot,
    load_upstream_case_statuses,
    write_immutable_development_snapshot,
)
from flowintentbench.experiment_scope import case_inventory
from scripts.build_full_dataset_n1_manifest import PILOT_LABEL  # noqa: E402
from scripts.preflight_full_dataset_n1 import validate_manifest  # noqa: E402
from scripts.run_real_model_pilot import parse_model_config_json  # noqa: E402


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root_for_manifest(manifest_file: Path) -> Path:
    """Locate the repository independently of the manifest's output folder."""

    for candidate in (manifest_file.parent, *manifest_file.parents):
        if (candidate / "datasets").is_dir() and (candidate / "flowintentbench").is_dir():
            return candidate.resolve()
    return ROOT.resolve()


def _repository_artifact_path(repository_root: Path, value: object) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else repository_root / path).resolve()


def _external_block_status(reason: object) -> str | None:
    value = classify_provider_failure(reason)
    return None if value is None else value.value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small progress artifact without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _pilot_identity(
    *,
    manifest_file: Path,
    repository_root: Path,
    target: EvaluationTarget,
    execution_backend: str,
    wire_api: str,
    provider_base_url: str | None,
    preflight_report_sha256: str,
    case_timeout_seconds: float,
    provider_retries: int,
    provider_retry_backoff_seconds: float,
    provider_rate_limit_backoff_seconds: float,
    provider_rate_limit_max_backoff_seconds: float,
    provider_circuit_breaker_threshold: int,
    provider_circuit_breaker_cooldown_seconds: float,
    agent_profile: AgentProfile | None = None,
) -> dict[str, Any]:
    effective = None if agent_profile is None else effective_agent_config(agent_profile)
    return {
        "case_manifest_sha256": _sha256_file(manifest_file),
        "benchmark_code_data_version": code_data_version(repository_root),
        "provider": target.provider,
        "agent_id": None if agent_profile is None else agent_profile.agent_id,
        "runtime_profile_id": None if agent_profile is None else agent_profile.runtime_profile_id,
        "protocol_version": None if agent_profile is None else "profiled-natural-output-v1",
        "agent_profile_sha256": None if agent_profile is None else agent_profile.profile_sha256,
        "runtime_profile_sha256": None if agent_profile is None else agent_profile.runtime_profile.profile_sha256,
        "system_instruction_sha256": None if effective is None else effective.system_instruction_sha256,
        "tool_schema_sha256": None if effective is None else effective.tool_schema_sha256,
        "model_id": target.model_id,
        "model_configuration": dict(target.model_configuration),
        "reasoning_effort": target.model_configuration.get("reasoning_effort"),
        "execution_backend": execution_backend,
        "wire_api": wire_api,
        "provider_base_url": provider_base_url,
        "target_fingerprint": evaluation_target_fingerprint(target),
        "preflight_report_sha256": preflight_report_sha256,
        "case_timeout_seconds": float(case_timeout_seconds),
        "provider_retries": int(provider_retries),
        "provider_retry_backoff_seconds": float(provider_retry_backoff_seconds),
        "provider_rate_limit_backoff_seconds": float(provider_rate_limit_backoff_seconds),
        "provider_rate_limit_max_backoff_seconds": float(provider_rate_limit_max_backoff_seconds),
        "provider_circuit_breaker_threshold": int(provider_circuit_breaker_threshold),
        "provider_circuit_breaker_cooldown_seconds": float(provider_circuit_breaker_cooldown_seconds),
    }


def _assert_identity(state: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    for field, expected in identity.items():
        if state.get(field) != expected:
            raise RuntimeError(f"collection identity mismatch for {field}")


def _record_observation(
    record: RunRecord,
    *,
    output_root: Path,
    dataset_id: str,
    condition: str,
    case_execution_position: int,
    run_record_path: Path,
    attempt_count: int,
    upstream_case_status: str = UPSTREAM_NOT_EVALUATED,
) -> dict[str, Any]:
    if record.run_status not in {RunStatus.COMPLETED, RunStatus.MODEL_NONCOMPLETION}:
        raise ValueError("only COMPLETED or MODEL_NONCOMPLETION records are valid observations")
    status_layer = build_case_execution_status(
        case_id=record.case_id,
        upstream_case_status=upstream_case_status,
        model_run_status=record.run_status.value,
        evaluator_status="NOT_STARTED",
    )
    efficiency = efficiency_observation_from_run_record(record).to_dict()
    return {
        "case_execution_position": case_execution_position,
        "dataset_id": dataset_id,
        "case_id": record.case_id,
        "condition": condition,
        "trial_index": record.trial_index,
        "run_id": record.run_id,
        "run_record_path": str(run_record_path.relative_to(output_root)),
        "run_record_sha256": _sha256_file(run_record_path),
        "status": record.run_status.value,
        **status_layer,
        "efficiency": efficiency,
        "reliability": _record_reliability(record),
        "tool_batch_count": record.tool_batch_count,
        "attempt_count": attempt_count,
}


def _record_reliability(record: RunRecord) -> dict[str, Any]:
    """Return provider/network telemetry outside scientific efficiency."""

    return {
        "infrastructure_retry_count": record.infrastructure_retry_count,
        "infrastructure_failed_request_time": record.infrastructure_failed_request_time,
        "retry_backoff_time": record.retry_backoff_time,
        "provider_failure": provider_failure_info(record.failure_reason),
    }


def _record_infrastructure_invalid_observation(
    *,
    expected_case: Mapping[str, Any],
    case_execution_position: int,
    dataset_id: str,
    attempt_dir: Path,
    record: RunRecord,
    output_root: Path,
    attempt_count: int,
    upstream_case_status: str = UPSTREAM_NOT_EVALUATED,
) -> dict[str, Any]:
    """Persist an accounted but non-scientific infrastructure slot.

    This is used only by the opt-in resilient N=1 collection path.  The
    record remains visible to the downstream evaluator as
    ``INFRASTRUCTURE_INVALID`` and therefore contributes no scientific score.
    """
    run_path = attempt_dir / "run_record.json"
    return {
        "case_execution_position": case_execution_position,
        "dataset_id": dataset_id,
        "case_id": str(expected_case["case_id"]),
        "condition": str(expected_case["condition"]),
        "trial_index": 1,
        "run_id": record.run_id,
        "run_record_path": str(run_path.relative_to(output_root)),
        "run_record_sha256": _sha256_file(run_path),
        "status": RunStatus.INFRASTRUCTURE_INVALID.value,
        "model_run_status": RunStatus.INFRASTRUCTURE_INVALID.value,
        "upstream_case_status": upstream_case_status,
        "upstream_status_source": "full-dataset-n1-pilot",
        "evaluator_status": "NOT_STARTED",
        "scientific_metrics_status": "NOT_ASSESSED",
        "efficiency": efficiency_observation_from_run_record(record).to_dict(),
        "reliability": _record_reliability(record),
        "tool_batch_count": record.tool_batch_count,
        "attempt_count": attempt_count,
        "failure_reason": record.failure_reason,
        "provider_failure": provider_failure_info(record.failure_reason),
    }


def _assert_run_record_identity(
    record: RunRecord,
    *,
    expected_case: Mapping[str, Any],
    target: EvaluationTarget,
    target_fingerprint: str,
    expected_status: str | None = None,
) -> None:
    """Prove that a persisted record belongs to the exact pilot slot."""

    case_id = str(expected_case["case_id"])
    if (
        record.case_id != case_id
        or record.trial_index != 1
        or record.case_execution_position != expected_case.get("case_execution_position")
        or record.provider != target.provider
        or record.model_id != target.model_id
        or record.model_configuration != dict(target.model_configuration)
        or record.target_fingerprint != target_fingerprint
        or record.experiment_id != "full-dataset-n1-pilot-v1"
        or record.formal_mode is not False
        or (
            expected_status is not None
            and record.run_status.value != expected_status
        )
    ):
        raise RuntimeError(f"saved RunRecord identity mismatch for {case_id}")


def _validate_observation(
    observation: Mapping[str, Any],
    *,
    output_root: Path,
    expected_case: Mapping[str, Any],
    target: EvaluationTarget,
    target_fingerprint: str,
) -> RunRecord:
    case_id = str(expected_case["case_id"])
    if observation.get("case_id") != case_id:
        raise RuntimeError(f"collection state case mismatch for {case_id}")
    if (
        observation.get("dataset_id") != expected_case.get("dataset_id")
        or observation.get("condition") != expected_case.get("condition")
    ):
        raise RuntimeError(f"collection state dataset/condition mismatch for {case_id}")
    if observation.get("case_execution_position") != expected_case.get("case_execution_position"):
        raise RuntimeError(f"collection state position mismatch for {case_id}")
    if observation.get("trial_index") != 1 or observation.get("status") not in {
        RunStatus.COMPLETED.value,
        RunStatus.MODEL_NONCOMPLETION.value,
        RunStatus.INFRASTRUCTURE_INVALID.value,
    }:
        raise RuntimeError(f"collection state observation is not a valid N=1 observation for {case_id}")
    if observation.get("model_run_status") != observation.get("status"):
        raise RuntimeError(f"collection state model-run status mismatch for {case_id}")
    if observation.get("evaluator_status") != "NOT_STARTED":
        raise RuntimeError(f"collection state evaluator status is not NOT_STARTED for {case_id}")
    relative = observation.get("run_record_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"collection state run_record_path is not relative for {case_id}")
    run_path = (output_root / relative).resolve()
    if output_root not in run_path.parents or not run_path.is_file():
        raise RuntimeError(f"saved RunRecord is missing for {case_id}")
    expected_base = (
        output_root
        / "runs"
        / str(expected_case["dataset_id"])
        / case_id
        / target_fingerprint
        / "trial-1"
    ).resolve()
    if run_path.parent != expected_base and run_path.parent.parent != expected_base:
        raise RuntimeError(f"saved RunRecord path is outside the expected case slot for {case_id}")
    if observation.get("run_record_sha256") != _sha256_file(run_path):
        raise RuntimeError(f"saved RunRecord hash mismatch for {case_id}")
    try:
        record = RunRecord.load_json(run_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"saved RunRecord is corrupt for {case_id}") from exc
    _assert_run_record_identity(
        record,
        expected_case=expected_case,
        target=target,
        target_fingerprint=target_fingerprint,
        expected_status=str(observation.get("status")),
    )
    if record.run_id != observation.get("run_id"):
        raise RuntimeError(f"saved RunRecord identity mismatch for {case_id}")
    return record


def _state_counts(observations: Mapping[str, Mapping[str, Any]]) -> tuple[int, int]:
    completed = sum(value.get("status") == RunStatus.COMPLETED.value for value in observations.values())
    noncompletion = sum(value.get("status") == RunStatus.MODEL_NONCOMPLETION.value for value in observations.values())
    return completed, noncompletion


def _write_collection_state(
    path: Path,
    *,
    identity: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]],
    infrastructure_invalid_attempt_count: int,
    aborted_infrastructure_attempt_count: int,
    status: str,
) -> None:
    completed, noncompletion = _state_counts(observations)
    accounted = len(observations)
    infrastructure_invalid = sum(
        value.get("status") == RunStatus.INFRASTRUCTURE_INVALID.value
        for value in observations.values()
    )
    valid_scientific = completed + noncompletion
    total_cases = int(identity.get("case_count", 28))
    slot_accounting_status = "COMPLETE" if accounted == total_cases else "INCOMPLETE"
    scientific_collection_status = (
        "COMPLETE"
        if valid_scientific == total_cases
        else (
            "INCOMPLETE_INFRASTRUCTURE"
            if infrastructure_invalid
            else "INCOMPLETE"
        )
    )
    # A caller may supply a provider-block status, which remains the most
    # specific top-level status.  Never let a generic COMPLETE label mask
    # exhausted infrastructure observations.
    effective_status = status
    if status == "COMPLETE" and scientific_collection_status != "COMPLETE":
        effective_status = scientific_collection_status
    payload = {
        "record_type": "full_dataset_n1_collection_state",
        "status": effective_status,
        **dict(identity),
        "total_cases": total_cases,
        "slot_accounting_status": slot_accounting_status,
        "scientific_collection_status": scientific_collection_status,
        "accounted_observation_count": accounted,
        "valid_observations": valid_scientific,
        "valid_scientific_observation_count": valid_scientific,
        "infrastructure_invalid_observation_count": infrastructure_invalid,
        "completed_count": completed,
        "model_noncompletion_count": noncompletion,
        "infrastructure_invalid_attempt_count": infrastructure_invalid_attempt_count,
        "aborted_infrastructure_attempt_count": aborted_infrastructure_attempt_count,
        "observations": [observations[key] for key in sorted(observations, key=lambda item: observations[item]["case_execution_position"])],
    }
    _atomic_write_json(path, payload)


def _inspect_case_attempt_history(
    case_base: Path,
    *,
    expected_case: Mapping[str, Any],
    target: EvaluationTarget,
    target_fingerprint: str,
    retry_network_noncompletion: bool = False,
) -> dict[str, Any]:
    """Inspect all persisted attempts before any new adapter/session exists."""

    if not case_base.exists():
        return {
            "valid": None,
            "invalid_count": 0,
            "aborted_count": 0,
            "consumed_count": 0,
            "highest_attempt": 0,
            "last_failure_reason": None,
        }
    if not case_base.is_dir():
        raise RuntimeError(f"case attempt root is not a directory: {case_base}")
    directories: list[tuple[int, Path]] = [(1, case_base)]
    for path in case_base.glob("attempt-*"):
        if not path.is_dir():
            raise RuntimeError(f"case attempt entry is not a directory: {path}")
        suffix = path.name.removeprefix("attempt-")
        if not suffix.isdigit() or int(suffix) <= 1:
            raise RuntimeError(f"invalid case attempt directory: {path}")
        directories.append((int(suffix), path))
    directories.sort(key=lambda value: value[0])
    found: dict[str, Any] | None = None
    invalid_count = 0
    aborted_count = 0
    last_failure_reason: str | None = None
    for attempt_number, directory in directories:
        record_path = directory / "run_record.json"
        if not record_path.exists():
            # The exact identity-bound attempt directory already exists, so a
            # provider/session execution may have started before a process
            # crash.  It is never a valid observation, but it consumes one
            # infrastructure attempt and its number must never be reused.
            aborted_count += 1
            continue
        try:
            record = RunRecord.load_json(record_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"saved RunRecord is corrupt: {record_path}") from exc
        _assert_run_record_identity(
            record,
            expected_case=expected_case,
            target=target,
            target_fingerprint=target_fingerprint,
        )
        if record.run_status is RunStatus.INFRASTRUCTURE_INVALID:
            invalid_count += 1
            last_failure_reason = record.failure_reason
        elif record.run_status in {RunStatus.COMPLETED, RunStatus.MODEL_NONCOMPLETION}:
            if retry_network_noncompletion and record.run_status is RunStatus.MODEL_NONCOMPLETION:
                reason = (record.failure_reason or "").casefold()
                if (
                    (record.infrastructure_retry_count or 0) > 0
                    or "during provider request" in reason
                    or "provider transport" in reason
                    or "provider unavailable" in reason
                    or "ssl" in reason
                    or "connection reset" in reason
                ):
                    # This is a prior network-induced pseudo-observation. It
                    # remains in the immutable attempt archive, but is not a
                    # reusable N=1 model observation for the corrected run.
                    continue
            if found is not None:
                raise RuntimeError(f"multiple valid RunRecords found for {expected_case['case_id']}")
            found = {"record": record, "path": record_path, "attempt_count": attempt_number}
        else:
            raise RuntimeError(f"unsupported persisted RunRecord status: {record.run_status.value}")
    return {
        "valid": found,
        "invalid_count": invalid_count,
        "aborted_count": aborted_count,
        "consumed_count": invalid_count + aborted_count,
        "highest_attempt": directories[-1][0],
        "last_failure_reason": last_failure_reason,
    }


def _seed_resume_from_collection(
    source_root: Path,
    *,
    destination_root: Path,
    state_path: Path,
    identity: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    retry_network_noncompletion: bool = False,
    recollect_case_ids: set[str] | None = None,
) -> None:
    """Seed a new immutable collection root with only valid source slots.

    This is the explicit migration used when a retry-policy correction creates
    a new pilot identity.  Invalid provider attempts remain on the source
    archive for audit, while valid model observations are copied unchanged so
    the next run re-collects only failed cases.
    """
    source_state_path = source_root / "collection_state.json"
    if not source_state_path.is_file():
        raise FileNotFoundError(f"resume source collection state is missing: {source_state_path}")
    source_state = _read_json(source_state_path)
    source_manifest_hash = source_state.get("case_manifest_sha256")
    selected_manifest_hash = identity.get("case_manifest_sha256")
    if source_manifest_hash != selected_manifest_hash:
        # Legacy pilots may have sealed collection_state with an obsolete
        # identity digest even though the immutable model-visible manifest
        # bytes are unchanged (the old filename was
        # ``collected_case_manifest.json``). Permit this migration only when
        # the persisted snapshot proves exact model-visible equivalence.
        snapshot_candidates = (
            source_root / "case_manifest.json",
            source_root / "collected_case_manifest.json",
        )
        source_snapshot = next((path for path in snapshot_candidates if path.is_file()), None)
        if source_snapshot is None:
            raise RuntimeError("resume source case manifest does not match current manifest")
        try:
            source_manifest = _read_json(source_snapshot)
            current_manifest = {str(item["case_id"]): item for item in cases}
            source_rows = {
                str(item["case_id"]): item
                for item in source_manifest.get("cases", [])
                if isinstance(item, Mapping) and item.get("case_id")
            }
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("resume source case manifest does not match current manifest") from exc
        visible_fields = ("case_id", "dataset_id", "condition", "case_input_sha256", "dataset_manifest_sha256")
        if set(source_rows) != set(current_manifest) or any(
            any(source_rows[case_id].get(field) != current_manifest[case_id].get(field) for field in visible_fields)
            for case_id in current_manifest
        ):
            raise RuntimeError("resume source case manifest does not match current manifest")
    source_target = source_state.get("target_fingerprint")
    if source_target != identity.get("target_fingerprint"):
        raise RuntimeError("resume source evaluated-model target does not match current target")
    source_by_id = {
        str(item.get("case_id")): item
        for item in source_state.get("observations", [])
        if isinstance(item, Mapping) and item.get("case_id")
    }
    expected_ids = {str(item["case_id"]) for item in cases}
    # A migrated source snapshot may intentionally contain only valid
    # observations (failed infrastructure rows remain in its parent audit
    # archive).  Accept that subset, while still rejecting unknown case IDs;
    # the destination manifest remains the authority for the full 28 slots.
    if not set(source_by_id).issubset(expected_ids):
        raise RuntimeError("resume source contains an unknown case ID")
    valid_statuses = {RunStatus.COMPLETED.value, RunStatus.MODEL_NONCOMPLETION.value}
    observations: dict[str, dict[str, Any]] = {}
    recollect_case_ids = {str(value) for value in (recollect_case_ids or set())}
    for case_id, item in source_by_id.items():
        if case_id in recollect_case_ids:
            continue
        if item.get("status") not in valid_statuses:
            continue
        relative = item.get("run_record_path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError(f"resume source path is invalid for {case_id}")
        source_path = (source_root / relative).resolve()
        destination_path = (destination_root / relative).resolve()
        if not source_path.is_file():
            raise RuntimeError(f"resume source RunRecord is missing for {case_id}")
        source_record = RunRecord.load_json(source_path)
        if retry_network_noncompletion and source_record.run_status is RunStatus.MODEL_NONCOMPLETION:
            reason = str(source_record.failure_reason or "").casefold()
            retry_count = source_record.infrastructure_retry_count or 0
            if (
                retry_count > 0
                or "during provider request" in reason
                or "provider transport" in reason
                or "provider unavailable" in reason
                or "ssl" in reason
                or "connection reset" in reason
            ):
                # Classify from the immutable RunRecord, not a legacy state
                # row that may predate failure_reason/reliability fields.
                continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path.parent, destination_path.parent, dirs_exist_ok=True)
        copied = dict(item)
        copied.setdefault("reliability", _record_reliability(source_record))
        observations[case_id] = copied
    _write_collection_state(
        state_path,
        identity=identity,
        observations=observations,
        # Invalid/aborted attempts are deliberately retained in the source
        # archive, not copied into the new retry-policy identity.  The new
        # state counters therefore start from the attempt history that is
        # actually present under ``destination_root`` and cannot drift from
        # its on-disk audit.
        infrastructure_invalid_attempt_count=0,
        aborted_infrastructure_attempt_count=0,
        status="INCOMPLETE",
    )


def resolve_execution_route(target: EvaluationTarget) -> tuple[str, str]:
    """Return the explicitly declared backend/wire pair or fail closed.

    Chat Completions and third-party Responses are explicit, disjoint routes.
    A CODEX_SUBAGENT identity remains unavailable and cannot be silently
    routed through either HTTP adapter.
    """

    config = dict(target.model_configuration)
    validate_reasoning_effort(config, label="evaluated model")
    backend = config.get("execution_backend") or config.get("backend")
    wire = config.get("wire_api") or config.get("wire_protocol")
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError("model_configuration must explicitly declare execution_backend")
    if not isinstance(wire, str) or not wire.strip():
        raise ValueError("model_configuration must explicitly declare wire_api")
    backend_key = backend.strip().casefold().replace("-", "_")
    wire_key = wire.strip().casefold().replace("-", "_")
    if backend_key in {"codex_subagent", "codex_hosted"}:
        raise RuntimeError(
            "the declared CODEX_SUBAGENT execution route is unavailable in this repository"
        )
    if backend_key in {"third_party_responses", "responses"}:
        if wire_key != "responses":
            raise RuntimeError(f"unsupported execution backend/wire protocol: {backend!r}/{wire!r}")
        return "THIRD_PARTY_RESPONSES", "responses"
    if backend_key in {"openai_chat_completions", "chat_completions"} and wire_key in {
        "chat_completions",
        "chat_completion",
    }:
        return "OPENAI_CHAT_COMPLETIONS", "chat_completions"
    raise RuntimeError(f"unsupported execution backend/wire protocol: {backend!r}/{wire!r}")


def _normalize_provider_base_url(
    target: EvaluationTarget,
    *,
    execution_backend: str,
    provider_base_url: str | None,
) -> str | None:
    if provider_base_url is None:
        return None
    normalized = provider_base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("provider_base_url must be non-empty when supplied")
    return normalized


def build_pilot_adapter(
    target: EvaluationTarget,
    *,
    execution_backend: str,
    provider_base_url: str | None,
    api_key: str | None,
    api_key_env: str = "OPENAI_API_KEY",
    request_timeout_seconds: float | None = None,
) -> object:
    """Construct only the adapter selected by the already-resolved route."""

    if execution_backend == "THIRD_PARTY_RESPONSES":
        if provider_base_url is None:
            raise RuntimeError("third-party Responses base URL is unavailable")
        return ThirdPartyResponsesAdapter(
            target,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=provider_base_url,
            formal_mode=False,
            request_timeout_seconds=request_timeout_seconds,
        )
    if execution_backend == "OPENAI_CHAT_COMPLETIONS":
        if provider_base_url is None:
            raise ValueError("chat_completions execution requires an explicit --base-url")
        return OpenAIChatCompletionsAdapter(
            target,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=provider_base_url,
            formal_mode=False,
            request_timeout_seconds=request_timeout_seconds,
        )
    raise RuntimeError(f"unsupported resolved execution backend: {execution_backend}")


def _assert_preflight_binding(
    report: Mapping[str, Any], *, manifest_file: Path, repository_root: Path
) -> None:
    if report.get("status") != "PASS":
        raise RuntimeError("preflight report is not PASS")
    cases = case_inventory(_read_json(manifest_file), require_dataset=True)
    if (
        report.get("case_count") != len(cases)
        or report.get("dataset_count") != len({row["dataset_id"] for row in cases.values()})
        or report.get("model_calls") != 0
    ):
        raise RuntimeError("preflight report does not match the manifest population or no-model condition")
    frozen_tests = report.get("frozen_tests")
    if not isinstance(frozen_tests, Mapping) or frozen_tests.get("returncode") != 0:
        raise RuntimeError("preflight frozen test suite did not pass")
    if not frozen_tests.get("command") or not frozen_tests.get("summary"):
        raise RuntimeError("preflight frozen test evidence is incomplete")
    expected_manifest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    expected_code = code_data_version(repository_root)
    if report.get("case_manifest_sha256") != expected_manifest:
        raise RuntimeError("preflight case_manifest_sha256 does not match current manifest")
    if report.get("benchmark_code_data_version") != expected_code:
        raise RuntimeError("preflight benchmark_code_data_version does not match current code")


def run_full_dataset_n1_pilot(
    *,
    manifest_path: str | Path,
    datasets_root: str | Path,
    output_root: str | Path,
    target: EvaluationTarget,
    adapter_factory: Callable[[EvaluationTarget], object],
    provider_base_url: str | None = None,
    runner: BenchmarkRunner | None = None,
    infrastructure_attempt_cap: int = 3,
    preflight_report_path: str | Path | None = None,
    run_preflight_tests: bool = True,
    resume: bool = False,
    resume_from: str | Path | None = None,
    agent_profile: AgentProfile | None = None,
    case_timeout_seconds: float = 900.0,
    provider_retries: int = 5,
    provider_retry_backoff_seconds: float = 1.0,
    provider_rate_limit_backoff_seconds: float = 15.0,
    provider_rate_limit_max_backoff_seconds: float = 60.0,
    provider_circuit_breaker_threshold: int = 3,
    provider_circuit_breaker_cooldown_seconds: float = 30.0,
    continue_on_infrastructure_exhaustion: bool = False,
    retry_network_noncompletion: bool = False,
    recollect_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Execute every frozen manifest slot once, with infra-only retries.

    Live collection normally runs a fresh preflight.  The canonical full
    dataset orchestrator may pass an identity-bound preflight report with
    ``run_preflight_tests=False``; exact manifest/code bindings are still
    checked. ``resume=True`` reuses only valid, identity-matching observations
    from ``collection_state.json``.
    """

    recollect_case_ids = {str(value) for value in (recollect_case_ids or set())}
    if infrastructure_attempt_cap <= 0:
        raise ValueError("infrastructure_attempt_cap must be positive")
    if case_timeout_seconds <= 0:
        raise ValueError("case_timeout_seconds must be positive")
    if provider_retries < 0:
        raise ValueError("provider_retries must be non-negative")
    if provider_retry_backoff_seconds < 0:
        raise ValueError("provider_retry_backoff_seconds must be non-negative")
    if provider_rate_limit_backoff_seconds < 0:
        raise ValueError("provider_rate_limit_backoff_seconds must be non-negative")
    if provider_rate_limit_max_backoff_seconds < provider_rate_limit_backoff_seconds:
        raise ValueError(
            "provider_rate_limit_max_backoff_seconds must be at least provider_rate_limit_backoff_seconds"
        )
    if provider_circuit_breaker_threshold <= 0:
        raise ValueError("provider_circuit_breaker_threshold must be positive")
    if provider_circuit_breaker_cooldown_seconds < 0:
        raise ValueError("provider_circuit_breaker_cooldown_seconds must be non-negative")
    if agent_profile is not None and (
        target.provider != agent_profile.provider or target.model_id != agent_profile.model_id
    ):
        raise ValueError("target provider/model does not match immutable agent profile")
    execution_backend, wire_api = resolve_execution_route(target)
    normalized_base_url = _normalize_provider_base_url(
        target,
        execution_backend=execution_backend,
        provider_base_url=provider_base_url,
    )
    manifest_file = Path(manifest_path).resolve()
    repository_root = _repository_root_for_manifest(manifest_file)
    output = Path(output_root).resolve()
    state_path = output / "collection_state.json"
    if output.exists():
        # Resume may follow read-only evaluation/report generation in the
        # same pilot root.  Permit only known pilot artifacts; arbitrary files
        # still fail closed to avoid mixing unrelated runs.
        allowed = {
            "preflight_report.json", "collection_state.json",
            "case_manifest.json",
            "development_snapshot.json", "pilot_manifest.json", "runs",
            "evaluation_records", "evaluation_summary.json", "summary.json",
            "report", "reports", "evaluate_args.json",
            "model_response_reuse_audit.json", "live_collection_diagnostic.json",
            "n1_metrics.json", "collected_case_manifest.json",
        }
        if resume:
            allowed.update({
                "collection_state.json",
                "runs",
                "pilot_manifest.json",
                "development_snapshot.json",
            })
        conflicts = [path for path in output.iterdir() if path.name not in allowed]
        if conflicts:
            raise FileExistsError(
                "refusing to overwrite existing pilot output: "
                + ", ".join(str(path) for path in conflicts)
            )
    output.mkdir(parents=True, exist_ok=True)
    # Preserve the exact model-visible case manifest consumed by this
    # collection.  This snapshot lets a later canonical replay distinguish a
    # hidden evaluator/GT edit (reusable response) from a changed model input
    # (rerun required) without consulting historical command-line arguments.
    manifest_snapshot = output / "case_manifest.json"
    if manifest_snapshot.is_file():
        if _sha256_file(manifest_snapshot) != _sha256_file(manifest_file):
            raise RuntimeError(
                "collection case_manifest.json does not match the selected manifest"
            )
    else:
        temporary = manifest_snapshot.with_name(f".{manifest_snapshot.name}.{os.getpid()}.tmp")
        temporary.write_bytes(manifest_file.read_bytes())
        os.replace(temporary, manifest_snapshot)
    # Repeat the no-model integrity gate immediately before creating any
    # runner, adapter, or session.  The canonical orchestrator may supply its
    # freshly written identity-bound report to avoid running the same suite
    # twice; it cannot bypass manifest/code binding checks.
    # On resume, reuse the already-bound preflight artifact after validating
    # its manifest/code binding.  The human-readable pytest duration in a
    # fresh report is inherently variable; regenerating it would change the
    # report byte hash and make an otherwise valid immutable collection
    # impossible to resume.
    preflight_output = output / "preflight_report.json"
    if preflight_report_path is not None:
        if run_preflight_tests:
            raise ValueError(
                "a supplied preflight report cannot replace the fresh live preflight; "
                "omit preflight_report_path"
            )
        supplied_report_path = Path(preflight_report_path).resolve()
        preflight_report = _read_json(supplied_report_path)
        validate_manifest(
            manifest_file,
            repository_root=repository_root,
            output_root=output,
            run_tests=False,
            allow_existing_collection=resume,
        )
    elif resume and preflight_output.is_file():
        preflight_report = _read_json(preflight_output)
        validate_manifest(
            manifest_file,
            repository_root=repository_root,
            output_root=output,
            run_tests=False,
            allow_existing_collection=True,
        )
    else:
        if not run_preflight_tests:
            raise ValueError(
                "run_preflight_tests=False requires an explicitly supplied bound report "
                "and is not a live-collection mode"
            )
        preflight_report = validate_manifest(
            manifest_file,
            repository_root=repository_root,
            output_root=output,
            run_tests=True,
            allow_existing_collection=resume,
        )
    # A resumed legacy collection may have a candidate manifest different
    # from the model-visible snapshot.  Its preflight report must still bind
    # to the snapshot that was actually consumed; the canonical orchestrator
    # performs the explicit cross-manifest reuse audit before invoking this
    # worker.  Direct pilot callers remain strict and fail closed.
    preflight_binding_manifest = manifest_file
    if resume:
        report_digest = preflight_report.get("case_manifest_sha256")
        snapshot_digest = _sha256_file(manifest_snapshot)
        if report_digest != _sha256_file(manifest_file) and report_digest == snapshot_digest:
            preflight_binding_manifest = manifest_snapshot
    _assert_preflight_binding(
        preflight_report,
        manifest_file=preflight_binding_manifest,
        repository_root=repository_root,
    )
    # Preserve the identity-bound report on resume.  A fresh run writes the
    # report atomically; resume has already validated the existing bytes.
    if not (resume and preflight_output.is_file()):
        _atomic_write_json(preflight_output, preflight_report)
    preflight_report_sha256 = _sha256_file(preflight_output)
    target_fp = evaluation_target_fingerprint(target)
    identity = _pilot_identity(
        manifest_file=manifest_file,
        repository_root=repository_root,
        target=target,
        execution_backend=execution_backend,
        wire_api=wire_api,
        provider_base_url=normalized_base_url,
        preflight_report_sha256=preflight_report_sha256,
        case_timeout_seconds=case_timeout_seconds,
        provider_retries=provider_retries,
        provider_retry_backoff_seconds=provider_retry_backoff_seconds,
        provider_rate_limit_backoff_seconds=provider_rate_limit_backoff_seconds,
        provider_rate_limit_max_backoff_seconds=provider_rate_limit_max_backoff_seconds,
        provider_circuit_breaker_threshold=provider_circuit_breaker_threshold,
        provider_circuit_breaker_cooldown_seconds=provider_circuit_breaker_cooldown_seconds,
        agent_profile=agent_profile,
    )
    manifest = _read_json(manifest_file)
    cases = manifest.get("cases")
    if manifest.get("manifest_version") != "full-dataset-n1-v1" or manifest.get("N") != 1 or manifest.get("formal") is not False:
        raise ValueError("manifest is not the frozen non-formal N=1 manifest")
    cases = list(case_inventory(manifest, require_dataset=True).values())
    if len(cases) != 28:
        identity["case_count"] = len(cases)
    if any(item.get("trial_index") != 1 for item in cases):
        raise ValueError("orchestrator supports trial_index=1 only")

    # Construction/release status is an upstream input snapshot.  For an
    # authoritative staged manifest, use the status bound to each explicit
    # row; otherwise retain the historical closure-file route.  In neither
    # route is status inferred from model output.
    case_ids = [str(item["case_id"]) for item in cases]
    if manifest.get("case_source") == "authoritative_source_manifest":
        upstream_statuses = {
            str(item["case_id"]): str(
                item.get("source_agent_ready_status") or UPSTREAM_NOT_EVALUATED
            )
            for item in cases
        }
        upstream_source = {
            "source_status": "PRESENT",
            "source_kind": "AUTHORITATIVE_CASE_MANIFEST",
            "source_path": manifest.get("source_manifest_path"),
            "source_sha256": manifest.get("source_manifest_sha256"),
            "row_count": len(cases),
        }
    else:
        upstream_statuses, upstream_source = load_upstream_case_statuses(
            repository_root,
            case_ids=case_ids,
        )
    snapshot = build_development_snapshot(
        manifest_sha256=str(identity["case_manifest_sha256"]),
        benchmark_code_data_version=str(identity["benchmark_code_data_version"]),
        target_fingerprint=target_fp,
        upstream_statuses=upstream_statuses,
        upstream_source=upstream_source,
    )
    snapshot_path = output / "development_snapshot.json"
    snapshot_sha256 = write_immutable_development_snapshot(snapshot_path, snapshot)
    identity["development_snapshot_sha256"] = snapshot_sha256

    datasets = Path(datasets_root).resolve()
    if resume_from is not None:
        if not resume:
            raise ValueError("resume_from requires resume=True")
        if state_path.exists():
            raise FileExistsError(
                "resume destination already has collection_state.json; omit resume_from"
            )
        _seed_resume_from_collection(
            Path(resume_from).resolve(),
            destination_root=output,
            state_path=state_path,
            identity=identity,
            cases=cases,
            retry_network_noncompletion=retry_network_noncompletion,
            recollect_case_ids=recollect_case_ids,
        )
    observations: dict[str, dict[str, Any]] = {}
    infrastructure_invalid_attempt_count = 0
    aborted_infrastructure_attempt_count = 0
    if resume:
        if not state_path.is_file():
            raise FileNotFoundError(
                f"resume requested but collection state is missing: {state_path}"
            )
        state = _read_json(state_path)
        if state.get("record_type") != "full_dataset_n1_collection_state":
            raise RuntimeError("collection state has an unexpected record_type")
        _assert_identity(state, identity)
        raw_observations = state.get("observations")
        if not isinstance(raw_observations, list):
            raise RuntimeError("collection state observations must be a list")
        case_by_id = {str(item["case_id"]): item for item in cases}
        for raw in raw_observations:
            if not isinstance(raw, Mapping):
                raise RuntimeError("collection state contains a non-object observation")
            case_id = str(raw.get("case_id"))
            if case_id in recollect_case_ids:
                # Completion acquisition invalidates this selected current
                # observation while preserving every immutable prior attempt.
                continue
            if case_id in observations or case_id not in case_by_id:
                raise RuntimeError(f"collection state contains duplicate/unknown case {case_id}")
            persisted_record = _validate_observation(
                raw,
                output_root=output,
                expected_case=case_by_id[case_id],
                target=target,
                target_fingerprint=target_fp,
            )
            # Infrastructure-invalid observations are accounted audit rows,
            # not valid scientific slots.  On resume they must be retried
            # rather than silently reused as completed observations.
            if persisted_record.run_status is not RunStatus.INFRASTRUCTURE_INVALID and not (
                retry_network_noncompletion
                and persisted_record.run_status is RunStatus.MODEL_NONCOMPLETION
                and (
                    (persisted_record.infrastructure_retry_count or 0) > 0
                    or "during provider request" in str(persisted_record.failure_reason or "").casefold()
                    or "provider transport" in str(persisted_record.failure_reason or "").casefold()
                    or "provider unavailable" in str(persisted_record.failure_reason or "").casefold()
                    or "ssl" in str(persisted_record.failure_reason or "").casefold()
                    or "connection reset" in str(persisted_record.failure_reason or "").casefold()
                )
            ):
                copied_observation = dict(raw)
                copied_observation.setdefault("reliability", _record_reliability(persisted_record))
                observations[case_id] = copied_observation
        infrastructure_invalid_attempt_count = int(state.get("infrastructure_invalid_attempt_count", 0))
        if infrastructure_invalid_attempt_count < 0:
            raise RuntimeError("collection state infrastructure attempt count is invalid")
        aborted_infrastructure_attempt_count = int(
            state.get("aborted_infrastructure_attempt_count", 0)
        )
        if aborted_infrastructure_attempt_count < 0:
            raise RuntimeError("collection state aborted infrastructure attempt count is invalid")
    else:
        if state_path.exists():
            raise FileExistsError(
                f"collection state already exists; pass resume=True to continue: {state_path}"
            )
        _write_collection_state(
            state_path,
            identity=identity,
            observations=observations,
            infrastructure_invalid_attempt_count=0,
            aborted_infrastructure_attempt_count=0,
            status="INCOMPLETE",
        )

    # Inspect every existing case slot before constructing the runner. This
    # closes the crash window where a RunRecord was written but collection
    # state was not updated, and makes the retry cap a per-case property.
    histories: dict[str, dict[str, Any]] = {}
    disk_invalid_attempt_count = 0
    disk_aborted_attempt_count = 0
    for item in cases:
        case_id = str(item["case_id"])
        case_base = output / "runs" / str(item["dataset_id"]) / case_id / target_fp / "trial-1"
        history = _inspect_case_attempt_history(
            case_base,
            expected_case=item,
            target=target,
            target_fingerprint=target_fp,
            retry_network_noncompletion=retry_network_noncompletion,
        )
        histories[case_id] = history
        disk_invalid_attempt_count += int(history["invalid_count"])
        disk_aborted_attempt_count += int(history["aborted_count"])
    if resume:
        if infrastructure_invalid_attempt_count > disk_invalid_attempt_count:
            raise RuntimeError(
                "collection state records more infrastructure-invalid attempts than exist on disk"
            )
        if aborted_infrastructure_attempt_count > disk_aborted_attempt_count:
            raise RuntimeError(
                "collection state records more aborted infrastructure attempts than exist on disk"
            )
        # A crash may leave the state count behind the on-disk count by one.
        # Disk history is authoritative for this diagnostic total.
        infrastructure_invalid_attempt_count = disk_invalid_attempt_count
        aborted_infrastructure_attempt_count = disk_aborted_attempt_count

    runner = runner or BenchmarkRunner(
        formal_mode=False,
        agent_profile=agent_profile,
        case_timeout_seconds=case_timeout_seconds,
        provider_retries=provider_retries,
        provider_retry_backoff_seconds=provider_retry_backoff_seconds,
        provider_rate_limit_backoff_seconds=provider_rate_limit_backoff_seconds,
        provider_rate_limit_max_backoff_seconds=provider_rate_limit_max_backoff_seconds,
    )
    if runner.formal_mode:
        raise ValueError("full-dataset N=1 pilot requires BenchmarkRunner(formal_mode=False)")

    results: list[dict[str, Any]] = []
    consecutive_provider_failures = 0
    for position, item in enumerate(cases):
        case_id = str(item["case_id"])
        dataset_id = str(item["dataset_id"])
        # Resolve this slot before any branch.  The exhausted-infrastructure
        # branch also needs the current case root; relying on a value from a
        # previous iteration can bind two case IDs to one RunRecord path.
        case_base = output / "runs" / dataset_id / case_id / target_fp / "trial-1"
        if case_id in observations:
            continue
        history = histories[case_id]
        case_path = _repository_artifact_path(repository_root, item["case_input_path"])
        # The manifest is repository-relative by contract.  Keeping this
        # explicit prevents accidental discovery of another construction tree.
        if not case_path.is_file():
            raise FileNotFoundError(case_path)
        loaded = CaseLoader(datasets).load(case_path)
        dataset_manifest_path = _repository_artifact_path(
            repository_root, item["dataset_manifest_path"]
        )
        dataset_manifest = DatasetManifest.model_validate_json(dataset_manifest_path.read_text(encoding="utf-8"))
        valid = None if case_id in recollect_case_ids else history["valid"]
        if valid is not None:
            discovered_record = valid["record"]
            discovered_path = valid["path"]
            observations[case_id] = _record_observation(
                discovered_record,
                output_root=output,
                dataset_id=dataset_id,
                condition=str(item["condition"]),
                case_execution_position=position,
                run_record_path=discovered_path,
                attempt_count=int(valid["attempt_count"]),
                upstream_case_status=upstream_statuses.get(case_id, UPSTREAM_NOT_EVALUATED),
            )
            _write_collection_state(
                state_path,
                identity=identity,
                observations=observations,
                infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                status="INCOMPLETE",
            )
            continue

        if int(history["consumed_count"]) >= infrastructure_attempt_cap:
            reason = history.get("last_failure_reason")
            block_status = _external_block_status(reason)
            if block_status:
                _write_collection_state(
                    state_path, identity=identity, observations=observations,
                    infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                    aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                    status=block_status,
                )
            if continue_on_infrastructure_exhaustion:
                record_path = case_base / "run_record.json"
                if not record_path.is_file():
                    record_path = case_base / f"attempt-{history['highest_attempt']}" / "run_record.json"
                invalid_record = RunRecord.load_json(record_path)
                observations[case_id] = _record_infrastructure_invalid_observation(
                    expected_case=item,
                    case_execution_position=position,
                    dataset_id=dataset_id,
                    attempt_dir=record_path.parent,
                    record=invalid_record,
                    output_root=output,
                    attempt_count=int(history["highest_attempt"]),
                    upstream_case_status=upstream_statuses.get(case_id, UPSTREAM_NOT_EVALUATED),
                )
                _write_collection_state(
                    state_path, identity=identity, observations=observations,
                    infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                    aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                    status="INCOMPLETE",
                )
                continue
            raise RuntimeError(
                f"case {case_id} already consumed {history['consumed_count']} infrastructure "
                f"attempts ({history['invalid_count']} invalid, {history['aborted_count']} aborted); "
                f"cap={infrastructure_attempt_cap}, refusing another adapter/session"
            )
        attempt = int(history["highest_attempt"]) + 1 if int(history["highest_attempt"]) else 1
        while True:
            attempt_dir = case_base if attempt == 1 else case_base / f"attempt-{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            record = runner.run_case(
                loaded,
                target=target,
                case_id=case_id,
                trial_index=1,
                ordering_seed=None,
                case_execution_position=position,
                manifest=dataset_manifest,
                trajectory_path=attempt_dir / "trajectory.json",
                adapter_factory=adapter_factory,
                experiment_id="full-dataset-n1-pilot-v1",
                benchmark_release_id=None,
            )
            _assert_run_record_identity(
                record,
                expected_case=item,
                target=target,
                target_fingerprint=target_fp,
            )
            record.write_json(attempt_dir / "run_record.json")
            if record.run_status is RunStatus.INFRASTRUCTURE_INVALID:
                history["invalid_count"] = int(history["invalid_count"]) + 1
                history["consumed_count"] = int(history["consumed_count"]) + 1
                history["highest_attempt"] = attempt
                infrastructure_invalid_attempt_count += 1
                _write_collection_state(
                    state_path,
                    identity=identity,
                    observations=observations,
                    infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                    aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                    status="INCOMPLETE",
                )
                failure_status = _external_block_status(record.failure_reason)
                if failure_status == "COLLECTION_BLOCKED_EXTERNAL_QUOTA":
                    _write_collection_state(
                        state_path,
                        identity=identity,
                        observations=observations,
                        infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                        aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                        status=failure_status,
                    )
                    raise RuntimeError(
                        f"collection blocked by external provider quota for {case_id}: "
                        f"{record.failure_reason or 'quota unavailable'}"
                    )
                if failure_status in {
                    "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT",
                    "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE",
                }:
                    consecutive_provider_failures += 1
                    if consecutive_provider_failures >= provider_circuit_breaker_threshold:
                        cooldown = provider_circuit_breaker_cooldown_seconds
                        _atomic_write_json(
                            attempt_dir / "provider_circuit_breaker.json",
                            {
                                "record_type": "provider_circuit_breaker_pause",
                                "status": "PAUSED",
                                "consecutive_failures": consecutive_provider_failures,
                                "threshold": provider_circuit_breaker_threshold,
                                "cooldown_seconds": cooldown,
                                "provider_failure_status": failure_status,
                                "case_id": case_id,
                            },
                        )
                        if cooldown:
                            import time as _time
                            _time.sleep(cooldown)
                        consecutive_provider_failures = 0
                else:
                    consecutive_provider_failures = 0
                if int(history["consumed_count"]) >= infrastructure_attempt_cap:
                    block_status = _external_block_status(record.failure_reason)
                    if block_status:
                        block = {
                            "record_type": "IncompleteExternalBlock",
                            "collection_status": block_status,
                            "blocked_case_id": case_id,
                            "blocked_dataset_id": dataset_id,
                            "blocked_trial": 1,
                            "blocked_trial_index": 1,
                            "blocked_attempt": attempt,
                            "blocked_attempt_index": attempt,
                            "last_completed_case_id": next(reversed(observations), None) if observations else None,
                            "failure_type": type(record.failure_reason).__name__,
                            "provider_failure_type": type(record.failure_reason).__name__,
                            "failure_message": record.failure_reason,
                            "provider_error_message": record.failure_reason,
                            "provider_failure": provider_failure_info(record.failure_reason),
                            "resume_allowed": True,
                            "case_manifest_sha256": identity["case_manifest_sha256"],
                            "benchmark_code_data_version": identity["benchmark_code_data_version"],
                            "agent_id": identity.get("agent_id"),
                            "agent_profile_sha256": identity.get("agent_profile_sha256"),
                            "runtime_profile_id": identity.get("runtime_profile_id"),
                            "runtime_profile_sha256": identity.get("runtime_profile_sha256"),
                            "system_instruction_sha256": identity.get("system_instruction_sha256"),
                            "tool_schema_sha256": identity.get("tool_schema_sha256"),
                            "protocol_version": identity.get("protocol_version"),
                        }
                        _atomic_write_json(attempt_dir / "incomplete_external_block.json", block)
                        _write_collection_state(
                            state_path, identity=identity, observations=observations,
                            infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                            aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                            status=block_status,
                        )
                    if continue_on_infrastructure_exhaustion:
                        observations[case_id] = _record_infrastructure_invalid_observation(
                            expected_case=item,
                            case_execution_position=position,
                            dataset_id=dataset_id,
                            attempt_dir=attempt_dir,
                            record=record,
                            output_root=output,
                            attempt_count=attempt,
                            upstream_case_status=upstream_statuses.get(case_id, UPSTREAM_NOT_EVALUATED),
                        )
                        _write_collection_state(
                            state_path, identity=identity, observations=observations,
                            infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                            aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                            status="INCOMPLETE",
                        )
                        break
                    raise RuntimeError(
                        f"case {case_id} exhausted infrastructure retry cap: {record.failure_reason or 'unknown'}"
                    )
                attempt = int(history["highest_attempt"]) + 1
                continue
            observations[case_id] = _record_observation(
                record,
                output_root=output,
                dataset_id=dataset_id,
                condition=str(item["condition"]),
                case_execution_position=position,
                run_record_path=attempt_dir / "run_record.json",
                attempt_count=attempt,
                upstream_case_status=upstream_statuses.get(case_id, UPSTREAM_NOT_EVALUATED),
            )
            _write_collection_state(
                state_path,
                identity=identity,
                observations=observations,
                infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
                aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
                status="INCOMPLETE",
            )
            break

    if len(observations) != len(cases):
        raise RuntimeError("collection did not account for the explicit manifest observations")
    _write_collection_state(
        state_path,
        identity=identity,
        observations=observations,
        infrastructure_invalid_attempt_count=infrastructure_invalid_attempt_count,
        aborted_infrastructure_attempt_count=aborted_infrastructure_attempt_count,
        status="COMPLETE",
    )
    collection_state_sha256 = _sha256_file(state_path)

    # The state is authoritative on resume.  Reconstruct the result list in
    # manifest order so a resumed run has the same global manifest shape as a
    # fresh run, without re-running any valid case.
    results = []
    for position, item in enumerate(cases):
        observation = observations[str(item["case_id"])]
        run_path = output / str(observation["run_record_path"])
        results.append(
            {
                "case_execution_position": position,
                "dataset_id": str(item["dataset_id"]),
                "case_id": str(item["case_id"]),
                "condition": item["condition"],
                "trial_index": 1,
                "attempt_count": int(observation.get("attempt_count", 1)),
                "status": observation["status"],
                "upstream_case_status": observation.get("upstream_case_status", UPSTREAM_NOT_EVALUATED),
                "model_run_status": observation.get("model_run_status", observation["status"]),
                "evaluator_status": observation.get("evaluator_status", "NOT_STARTED"),
                "scientific_metrics_status": observation.get("scientific_metrics_status", "NOT_ASSESSED"),
                "efficiency": dict(observation.get("efficiency", {})),
                "tool_batch_count": int(observation.get("tool_batch_count", 0)),
                "run_record_path": observation["run_record_path"],
                "trajectory_path": str(
                    (run_path.parent / "trajectory.json").relative_to(output)
                ),
                "run_id": observation["run_id"],
            }
        )

    pilot_manifest = {
        "record_type": "full_dataset_n1_pilot_manifest",
        "status": "PILOT / NOT FORMAL BENCHMARK RESULT",
        "collection_status": (
            "COMPLETE"
            if _state_counts(observations)[0] + _state_counts(observations)[1] == len(cases)
            else "INCOMPLETE_INFRASTRUCTURE"
            if any(item["status"] == RunStatus.INFRASTRUCTURE_INVALID.value for item in results)
            else "INCOMPLETE"
        ),
        "pilot_label": PILOT_LABEL,
        "manifest_version": manifest["manifest_version"],
        "source_case_manifest_path": str(manifest_file.relative_to(repository_root)) if manifest_file.is_relative_to(repository_root) else str(manifest_file),
        "case_manifest_sha256": identity["case_manifest_sha256"],
        "N": 1,
        "formal": False,
        "case_count": len(cases),
        "trial_index": 1,
        "execution_backend": execution_backend,
        "wire_api": wire_api,
        "provider_base_url": normalized_base_url,
        "preflight_report_path": "preflight_report.json",
        "preflight_report_sha256": preflight_report_sha256,
        "collection_state_sha256": collection_state_sha256,
        "development_snapshot_path": "development_snapshot.json",
        "development_snapshot_sha256": snapshot_sha256,
        "upstream_status_source": dict(upstream_source),
        "upstream_case_status_counts": dict(Counter(upstream_statuses.values())),
        "model_run_status_counts": dict(Counter(item["model_run_status"] for item in results)),
        "evaluator_status_counts": dict(Counter(item["evaluator_status"] for item in results)),
        "benchmark_code_data_version": identity["benchmark_code_data_version"],
        "provider": target.provider,
        "agent_id": identity.get("agent_id"),
        "runtime_profile_id": identity.get("runtime_profile_id"),
        "model_id": target.model_id,
        "model_configuration": dict(target.model_configuration),
        "reasoning_effort": target.model_configuration.get("reasoning_effort"),
        "target_fingerprint": target_fp,
        "total_cases": len(cases),
        "accounted_observation_count": len(observations),
        "valid_observations": _state_counts(observations)[0] + _state_counts(observations)[1],
        "valid_scientific_observation_count": _state_counts(observations)[0] + _state_counts(observations)[1],
        "infrastructure_invalid_observation_count": sum(
            item["status"] == RunStatus.INFRASTRUCTURE_INVALID.value for item in results
        ),
        "completed_count": _state_counts(observations)[0],
        "model_noncompletion_count": _state_counts(observations)[1],
        "infrastructure_invalid_attempt_count": infrastructure_invalid_attempt_count,
        "aborted_infrastructure_attempt_count": aborted_infrastructure_attempt_count,
        "target": {
            "provider": target.provider,
            "model_id": target.model_id,
            "model_configuration": dict(target.model_configuration),
            "target_fingerprint": target_fp,
            "provider_base_url": normalized_base_url,
        },
        "cases": results,
    }
    manifest_out = output / "pilot_manifest.json"
    manifest_out.write_text(json.dumps(pilot_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pilot_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/userstudy/case_manifest.json")
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent", required=True, help="agent profile ID or path")
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=None,
        help=(
            "identity-bound preflight report produced by the canonical experiment "
            "orchestrator; requires --skip-preflight-tests"
        ),
    )
    parser.add_argument(
        "--skip-preflight-tests",
        action="store_true",
        help="consume --preflight-report instead of running a duplicate test suite",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "explicit provider base URL; if omitted, resolve it from server/provider configuration"
        ),
    )
    parser.add_argument("--infrastructure-attempt-cap", type=int, default=4)
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=900.0,
        help="per-case wall-clock timeout passed to BenchmarkRunner (default: 900)",
    )
    parser.add_argument(
        "--provider-retries",
        type=int,
        default=5,
        help=(
            "replay the exact unacknowledged provider request within one case "
            "after a transport failure (default: 5)"
        ),
    )
    parser.add_argument(
        "--provider-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="base exponential backoff between exact provider retries (default: 1s)",
    )
    parser.add_argument(
        "--provider-rate-limit-backoff-seconds",
        type=float,
        default=15.0,
        help="base delay for HTTP 429; Retry-After takes precedence (default: 15s)",
    )
    parser.add_argument(
        "--provider-rate-limit-max-backoff-seconds",
        type=float,
        default=60.0,
        help="maximum local exponential delay for HTTP 429 (default: 60s)",
    )
    parser.add_argument(
        "--provider-circuit-breaker-threshold",
        type=int,
        default=3,
        help="pause after this many consecutive provider timeout/unavailable failures (default: 3)",
    )
    parser.add_argument(
        "--provider-circuit-breaker-cooldown-seconds",
        type=float,
        default=30.0,
        help="cooldown duration for the provider circuit breaker (default: 30s)",
    )
    parser.add_argument(
        "--continue-on-infrastructure-exhaustion",
        action="store_true",
        help="account exhausted infrastructure slots and continue remaining cases",
    )
    parser.add_argument(
        "--retry-network-noncompletion",
        action="store_true",
        help=(
            "recollect prior MODEL_NONCOMPLETION records whose telemetry or reason "
            "shows provider/network failure; preserve genuine model noncompletion"
        ),
    )
    parser.add_argument(
        "--recollect-case-ids",
        default="",
        help="comma-separated case IDs to recollect while preserving prior attempts",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from an identity-bound collection_state.json without resampling valid observations",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="seed a new collection identity from valid observations in an older pilot root",
    )
    args = parser.parse_args()
    try:
        profile = load_agent_profile(args.agent)
        config = (
            None if args.model_config_json is None else parse_model_config_json(args.model_config_json)
        )
        reject_profile_identity_overrides(
            provider=args.provider, model_id=args.model_id, model_configuration=config
        )
        profile_configuration = profile.model_configuration
        profile_configuration.update(config or {})
        config = profile_configuration
        runtime = resolve_provider_configuration(
            role="model",
            config_path=args.server_config,
            provider=profile.provider,
            base_url=args.base_url,
            model_id=profile.model_id,
            model_configuration=config,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.error(str(exc))
    target = EvaluationTarget(runtime.provider, runtime.model_id, model_configuration=runtime.model_configuration)
    if runtime.provider != profile.provider or runtime.model_id != profile.model_id:
        parser.error("resolved provider/model does not match immutable agent profile")
    print(json.dumps({"effective_agent_config": effective_agent_config(profile).to_dict()}, indent=2))
    try:
        execution_backend, _wire_api = resolve_execution_route(target)
        provider_base_url = _normalize_provider_base_url(
            target,
            execution_backend=execution_backend,
            provider_base_url=runtime.base_url,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        provider_api_key = resolve_api_key(runtime, explicit_api_key=args.api_key)
    except RuntimeError as exc:
        parser.error(str(exc))
    def factory(evaluation_target: EvaluationTarget):
        return build_pilot_adapter(
            evaluation_target,
            execution_backend=execution_backend,
            provider_base_url=provider_base_url,
            api_key=provider_api_key,
            api_key_env=runtime.api_key_env,
        )
    recollect_case_ids = {
        value.strip() for value in str(args.recollect_case_ids).split(",") if value.strip()
    }
    result = run_full_dataset_n1_pilot(
        manifest_path=args.manifest,
        datasets_root=args.datasets_root,
        output_root=args.output_root,
        target=target,
        adapter_factory=factory,
        provider_base_url=provider_base_url,
        infrastructure_attempt_cap=args.infrastructure_attempt_cap,
        case_timeout_seconds=args.case_timeout_seconds,
        provider_retries=args.provider_retries,
        provider_retry_backoff_seconds=args.provider_retry_backoff_seconds,
        provider_rate_limit_backoff_seconds=args.provider_rate_limit_backoff_seconds,
        provider_rate_limit_max_backoff_seconds=args.provider_rate_limit_max_backoff_seconds,
        provider_circuit_breaker_threshold=args.provider_circuit_breaker_threshold,
        provider_circuit_breaker_cooldown_seconds=args.provider_circuit_breaker_cooldown_seconds,
        continue_on_infrastructure_exhaustion=args.continue_on_infrastructure_exhaustion,
        retry_network_noncompletion=args.retry_network_noncompletion,
        recollect_case_ids=recollect_case_ids,
        run_preflight_tests=not args.skip_preflight_tests,
        preflight_report_path=args.preflight_report,
        resume=args.resume,
        resume_from=args.resume_from,
        agent_profile=profile,
    )
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "target_fingerprint": result["target"]["target_fingerprint"],
        "output_root": str(args.output_root.resolve()),
        "pilot_manifest": str(args.output_root.resolve() / "pilot_manifest.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
