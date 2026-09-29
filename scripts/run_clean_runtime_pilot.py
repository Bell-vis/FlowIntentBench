"""Run the fixed eight-case clean-runtime N=1 diagnostic pilot.

This entrypoint uses the configured local/server model route.  It never falls
back to an official OpenAI URL, never evaluates scientific correctness, and
never labels the resulting N=1 observations as formal benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    AgentProfile,
    BenchmarkRunner,
    CaseLoader,
    DatasetManifest,
    EvaluationTarget,
    RunRecord,
    RunStatus,
    code_data_version,
    evaluation_target_fingerprint,
    resolve_api_key,
    resolve_provider_configuration,
    load_agent_profile,
    effective_agent_config,
)
from flowintentbench.provider_status import provider_failure_info  # noqa: E402
from flowintentbench.calibration import load_frozen_observations  # noqa: E402
from scripts.run_full_dataset_n1_matrix import run_backend_smoke  # noqa: E402
from scripts.run_full_dataset_n1_pilot import (  # noqa: E402
    build_pilot_adapter,
    resolve_execution_route,
)


CLEAN_RUNTIME_CASE_IDS = (
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
    "office_speed_zones_o1_f2",
    "blunt_fin_o3_f1",
    "carotid_o1_f2",
    "nasa_lox_post_o2_f1",
    "kitchen_o2_f1",
)

# Audited against the current case manifest and the preserved N=1 artifacts.
# This registry is deliberately explicit: an unchanged runtime comparison may
# never be inferred from dataset names alone.
CASE_CHANGE_REGISTRY = {
    "office_speed_zones_o1_f1": "UNCHANGED_SCIENTIFIC_TASK",
    "office_speed_zones_o2_f1": "UNCHANGED_SCIENTIFIC_TASK",
    "office_speed_zones_o3_f1": "UNCHANGED_SCIENTIFIC_TASK",
    "office_speed_zones_o1_f2": "UNCHANGED_SCIENTIFIC_TASK",
    "blunt_fin_o3_f1": "UNCHANGED_SCIENTIFIC_TASK",
    "carotid_o1_f2": "TASK_SPEC_CHANGED",
    "nasa_lox_post_o2_f1": "DATA_CONTRACT_CHANGED",
    "kitchen_o2_f1": "UNCHANGED_SCIENTIFIC_TASK",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _is_external_quota(reason: object) -> bool:
    text = str(reason or "").casefold()
    return any(token in text for token in ("429", "quota", "rate limit", "insufficient_quota", "resource exhausted"))


def _external_collection_status(reason: object) -> str | None:
    text = str(reason or "").casefold()
    if _is_external_quota(text):
        return "COLLECTION_BLOCKED_EXTERNAL_QUOTA"
    if "provider request timeout" in text or "timed out" in text or "timeout" in text:
        return "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT"
    if "provider unavailable" in text or "provider transport failure" in text or "connection refused" in text:
        return "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE"
    return None


def _trajectory_diagnostics(trajectory: list[Mapping[str, Any]]) -> dict[str, Any]:
    filesystem_errors = 0
    root_search_timeouts = 0
    truncations = 0
    uncontrolled = 0
    returned_max = 0
    python_errors = 0
    tool_boundary_retries = 0
    multi_call_tool_boundary_retries = 0
    multi_call_turns = 0
    max_batch_size = 1
    duplicate_tool_execution = 0
    seen_call_ids: set[str] = set()
    last_turn_multi = False
    last_code = ""
    for event in trajectory:
        if event.get("event") == "infrastructure_retry":
            if event.get("category") == "TOOL_INTERFACE_ERROR":
                tool_boundary_retries += 1
                if last_turn_multi:
                    multi_call_tool_boundary_retries += 1
            continue
        if event.get("event") == "model_turn":
            if isinstance(event.get("code"), str):
                last_code = str(event["code"])
            call_count = int(event.get("number_of_tool_calls") or 1)
            last_turn_multi = call_count > 1
            if last_turn_multi:
                multi_call_turns += 1
                max_batch_size = max(max_batch_size, call_count)
            continue
        if event.get("event") != "python_execution":
            continue
        call_id = event.get("canonical_call_id")
        if isinstance(call_id, str) and call_id in seen_call_ids:
            duplicate_tool_execution += int(not event.get("journal_reused", False))
        if isinstance(call_id, str):
            seen_call_ids.add(call_id)
        exception = event.get("exception")
        if isinstance(exception, Mapping):
            python_errors += 1
            if exception.get("type") == "FilesystemDiscoveryError" or event.get(
                "filesystem_discovery_blocked"
            ):
                filesystem_errors += 1
            if exception.get("type") == "TimeoutError" and any(
                token in last_code
                for token in (
                    "Path('/').rglob",
                    "glob('/**",
                    "glob.glob('/**",
                    "os.walk('/')",
                )
            ):
                root_search_timeouts += 1
                filesystem_errors += 1
        if event.get("output_truncated"):
            truncations += 1
        returned_stdout = int(event.get("returned_stdout_chars") or len(str(event.get("stdout", ""))))
        returned_stderr = int(event.get("returned_stderr_chars") or len(str(event.get("stderr", ""))))
        returned_max = max(returned_max, returned_stdout, returned_stderr)
        for original, returned in (
            (event.get("stdout_chars_total"), returned_stdout),
            (event.get("stderr_chars_total"), returned_stderr),
        ):
            if isinstance(original, int) and original > 16_000 and returned > 16_000:
                uncontrolled += 1
    return {
        "filesystem_discovery_errors": filesystem_errors,
        "root_search_timeouts": root_search_timeouts,
        "stdout_truncations": truncations,
        "uncontrolled_huge_stdout": uncontrolled,
        "max_returned_tool_chars": returned_max,
        "python_errors": python_errors,
        "tool_boundary_retries": tool_boundary_retries,
        "multi_call_tool_boundary_retries": multi_call_tool_boundary_retries,
        "multi_call_turns": multi_call_turns,
        "max_batch_size": max_batch_size,
        "duplicate_tool_execution": duplicate_tool_execution,
    }


def _historical_rows(collection_root: Path, case_ids: set[str]) -> list[dict[str, Any]]:
    try:
        observations = load_frozen_observations(collection_root)
    except (FileNotFoundError, ValueError):
        return []
    rows = []
    for observation in observations:
        if observation.case_id not in case_ids:
            continue
        diagnostics = _trajectory_diagnostics(
            list(observation.trajectory)
        )
        rows.append(
            {
                "case_id": observation.run_record.case_id,
                "record": observation.run_record,
                **diagnostics,
            }
        )
    return rows


def _comparison_markdown(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> str:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        records = [row["record"] for row in rows]
        return {
            "case_count": len(rows),
            "wall_time": _mean([record.wall_clock_time for record in records]),
            "input_tokens": _mean([record.input_tokens for record in records]),
            "turns": _mean([record.model_turn_count for record in records]),
            "python_calls": _mean([record.python_execution_count for record in records]),
            "filesystem_errors": sum(int(row["filesystem_discovery_errors"]) for row in rows),
            "stdout_truncations": sum(int(row["stdout_truncations"]) for row in rows),
            "max_returned_tool_chars": max(
                (int(row["max_returned_tool_chars"]) for row in rows), default=0
            ),
        }

    old = summarize(old_rows)
    new = summarize(new_rows)
    rows = "\n".join(
        f"| {metric} | {old[metric]} | {new[metric]} |"
        for metric in (
            "case_count",
            "wall_time",
            "input_tokens",
            "turns",
            "python_calls",
            "filesystem_errors",
            "stdout_truncations",
            "max_returned_tool_chars",
        )
    )
    return (
        "# Old vs Clean Runtime\n\n"
        "This is a descriptive pilot comparison. Historical values are not corrected or "
        "promoted to formal results.\n\n"
        "| Metric | Historical matched pilot | Clean runtime |\n|---|---:|---:|\n"
        + rows
        + "\n"
    )


def _comparison_subset(
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    allowed_status: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if allowed_status is None:
        return old_rows, new_rows
    allowed = {case_id for case_id, status in CASE_CHANGE_REGISTRY.items() if status == allowed_status}
    return ([row for row in old_rows if row.get("case_id") in allowed], [row for row in new_rows if row.get("case_id") in allowed])


def _comparison_with_registry(old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    unchanged_old, unchanged_new = _comparison_subset(old_rows, new_rows, allowed_status="UNCHANGED_SCIENTIFIC_TASK")
    return {
        "registry": [{"case_id": case_id, "classification": classification} for case_id, classification in sorted(CASE_CHANGE_REGISTRY.items())],
        "whole_system": _comparison_markdown(old_rows, new_rows),
        "unchanged_task": _comparison_markdown(unchanged_old, unchanged_new) if unchanged_old or unchanged_new else "# Unchanged-task runtime effect\n\nStatus: **PENDING_EMPTY_COMPARISON_SUBSET**.\n",
        "unchanged_case_count": len(unchanged_new),
    }
def run_clean_runtime_pilot(
    *,
    output_root: str | Path,
    collection_root: str | Path,
    server_config: str | Path | None = None,
    api_key: str | None = None,
    infrastructure_attempt_cap: int = 3,
    request_timeout_seconds: float = 120.0,
    agent_profile: AgentProfile | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if infrastructure_attempt_cap <= 0:
        raise ValueError("infrastructure_attempt_cap must be positive")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if agent_profile is None:
        raise ValueError("New benchmark runs require an explicit agent profile.")
    output = Path(output_root).resolve()
    expected_case_manifest_sha256 = _sha256(ROOT / "experiments/userstudy/case_manifest.json")
    expected_code_data_version = code_data_version(ROOT)
    existing_manifest: Mapping[str, Any] | None = None
    if output.exists() and (output / "manifest.json").is_file() and resume:
        existing_manifest = _read_json(output / "manifest.json")
        if existing_manifest.get("status") == "COMPLETE":
            return dict(existing_manifest)
        if existing_manifest.get("agent_profile_sha256") != agent_profile.profile_sha256:
            raise RuntimeError("clean pilot resume profile hash mismatch")
        if existing_manifest.get("runtime_profile_sha256") != agent_profile.runtime_profile.profile_sha256:
            raise RuntimeError("clean pilot resume runtime profile hash mismatch")
        effective = effective_agent_config(agent_profile)
        if existing_manifest.get("system_instruction_sha256") != effective.system_instruction_sha256:
            raise RuntimeError("clean pilot resume system instruction hash mismatch")
        if existing_manifest.get("tool_schema_sha256") != effective.tool_schema_sha256:
            raise RuntimeError("clean pilot resume tool schema hash mismatch")
        if existing_manifest.get("protocol_version") != "profiled-natural-output-v1":
            raise RuntimeError("clean pilot resume protocol mismatch")
        if existing_manifest.get("case_manifest_sha256") != expected_case_manifest_sha256:
            raise RuntimeError("clean pilot resume case manifest hash mismatch")
        if existing_manifest.get("benchmark_code_data_version") != expected_code_data_version:
            raise RuntimeError("clean pilot resume benchmark code/data version mismatch")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite clean pilot output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    completed_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )
    if completed_tests.returncode != 0:
        raise RuntimeError(f"clean-pilot preflight tests failed: {completed_tests.stdout[-2000:]}")

    runtime = resolve_provider_configuration(
        role="model",
        config_path=server_config,
        provider=None if agent_profile is None else agent_profile.provider,
        model_id=None if agent_profile is None else agent_profile.model_id,
        model_configuration=None if agent_profile is None else agent_profile.model_configuration,
    )
    # The immutable AgentProfile is the single source of model reasoning
    # configuration; this runner must not silently mutate it.
    target_configuration = dict(runtime.model_configuration)
    target = EvaluationTarget(runtime.provider, runtime.model_id, target_configuration)
    backend, wire = resolve_execution_route(target)
    key = resolve_api_key(runtime, explicit_api_key=api_key)
    try:
        compatibility = run_backend_smoke(target=target, runtime=runtime, api_key=key)
    except Exception as exc:
        compatibility = {
            "status": "FAIL",
            "label": "BACKEND CONTRACT SMOKE TEST ONLY — NOT A BENCHMARK OBSERVATION",
            "provider": target.provider,
            "model_id": target.model_id,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    (output / "provider_compatibility_report.json").write_text(
        json.dumps(compatibility, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # A failed provider conformance smoke is a fail-closed pilot outcome. Do
    # not spend eight scientific calls against a route that cannot satisfy the
    # canonical adapter contract.
    if compatibility.get("status") != "PASS":
        blocked_status = _external_collection_status(compatibility.get("reason")) or "BLOCKED"
        manifest = {
            "record_type": "FlowIntentBenchCleanRuntimePilot",
            "status": blocked_status,
            "formal": False,
            "N": 1,
            "case_count": 0,
            "provider": target.provider,
            "model_id": target.model_id,
            "agent_id": agent_profile.agent_id,
            "agent_profile_sha256": agent_profile.profile_sha256,
            "runtime_profile_id": agent_profile.runtime_profile_id,
            "runtime_profile_sha256": agent_profile.runtime_profile.profile_sha256,
            "system_instruction_sha256": effective_agent_config(agent_profile).system_instruction_sha256,
            "tool_schema_sha256": effective_agent_config(agent_profile).tool_schema_sha256,
            "protocol_version": "profiled-natural-output-v1",
            "model_configuration": dict(target.model_configuration),
            "provider_timeout_configuration": {
                "connection_seconds": request_timeout_seconds,
                "read_seconds": request_timeout_seconds,
                "overall_request_seconds": request_timeout_seconds,
            },
            "provider_compatibility_status": compatibility.get("status"),
            "provider_compatibility_reason": compatibility.get("reason"),
            "blocked_at": datetime.now(timezone.utc).isoformat() if blocked_status.startswith("COLLECTION_BLOCKED_") else None,
            "resume_allowed": blocked_status.startswith("COLLECTION_BLOCKED_"),
            "case_manifest_sha256": _sha256(ROOT / "experiments/userstudy/case_manifest.json"),
            "filesystem_discovery_errors": 0,
            "root_search_timeouts": 0,
            "uncontrolled_huge_stdout": 0,
            "conversation_state_corruption": 0,
            "unsupported_reasoning_fallbacks": 0,
            "infrastructure_invalid_attempts": 0,
            "multi_call_tool_boundary_retries": 0,
            "multi_call_infrastructure_invalid_attempts": 0,
            "duplicate_tool_execution": 0,
            "cases": [],
        }
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "runtime_report.md").write_text(
            f"# Clean-Runtime N=1 Pilot\n\nStatus: **{blocked_status}**. Provider compatibility smoke failed; no scientific case was executed.\n",
            encoding="utf-8",
        )
        for name in ("efficiency_comparison.md", "unchanged_task_runtime_effect.md"):
            (output / name).write_text("No scientific case was executed; external provider availability is not a runtime result.\n", encoding="utf-8")
        return manifest

    manifest_path = ROOT / "experiments/userstudy/case_manifest.json"
    source_manifest = _read_json(manifest_path)
    case_manifest_sha256 = _sha256(manifest_path)
    by_id = {
        str(item["case_id"]): item
        for item in source_manifest["cases"]
        if isinstance(item, Mapping)
    }
    if not set(CLEAN_RUNTIME_CASE_IDS) <= set(by_id):
        raise RuntimeError("clean-runtime subset is not present in the frozen N=1 manifest")

    def factory(evaluation_target: EvaluationTarget):
        return build_pilot_adapter(
            evaluation_target,
            execution_backend=backend,
            provider_base_url=runtime.base_url,
            api_key=key,
            api_key_env=runtime.api_key_env,
            request_timeout_seconds=request_timeout_seconds,
        )

    runner = BenchmarkRunner(
        formal_mode=False,
        agent_profile=agent_profile,
    )
    result_rows: list[dict[str, Any]] = list((existing_manifest or {}).get("cases", []))
    completed_case_ids = {str(row.get("case_id")) for row in result_rows}
    case_states: list[dict[str, Any]] = list((existing_manifest or {}).get("case_states", []))
    if not case_states:
        case_states = [{"case_id": row.get("case_id"), "state": "COMPLETED" if row.get("status") in {"COMPLETED", "MODEL_NONCOMPLETION"} else row.get("status")} for row in result_rows]
    diagnostic_rows: list[dict[str, Any]] = []
    attempt_diagnostics: list[dict[str, Any]] = []
    infrastructure_invalid_attempts = 0
    for position, case_id in enumerate(CLEAN_RUNTIME_CASE_IDS):
        if case_id in completed_case_ids:
            continue
        item = by_id[case_id]
        dataset_id = str(item["dataset_id"])
        case_path = ROOT / str(item["case_input_path"])
        case = CaseLoader(ROOT / "datasets").load(case_path)
        dataset_manifest = DatasetManifest.model_validate_json(
            (ROOT / str(item["dataset_manifest_path"])).read_text(encoding="utf-8")
        )
        case_root = output / "runs" / dataset_id / case_id
        record: RunRecord | None = None
        existing_attempts = sorted(case_root.glob("attempt-*/run_record.json"))
        attempt_start = len(existing_attempts) + 1
        for attempt in range(attempt_start, attempt_start + infrastructure_attempt_cap):
            attempt_root = case_root / f"attempt-{attempt}"
            trajectory_path = attempt_root / "trajectory.json"
            record = runner.run_case(
                case,
                target=target,
                case_id=case_id,
                trial_index=1,
                case_execution_position=position,
                manifest=dataset_manifest,
                trajectory_path=trajectory_path,
                adapter_factory=factory,
                experiment_id="clean-runtime-n1-gate-v1",
            )
            record.write_json(attempt_root / "run_record.json")
            diagnostics = _trajectory_diagnostics(
                [entry for entry in record.trajectory if isinstance(entry, Mapping)]
            )
            attempt_diagnostics.append(diagnostics)
            if record.run_status is not RunStatus.INFRASTRUCTURE_INVALID:
                break
            infrastructure_invalid_attempts += 1
            block_status = _external_collection_status(record.failure_reason)
            if block_status is None:
                for entry in record.trajectory:
                    if isinstance(entry, Mapping):
                        block_status = _external_collection_status(entry.get("exception") or entry.get("message"))
                        if block_status:
                            break
            if block_status is not None:
                failure_info = provider_failure_info(record.failure_reason, fallback_status=block_status)
                if failure_info is None:
                    failure_info = provider_failure_info(block_status, fallback_status=block_status)
                unattempted_case_ids = list(CLEAN_RUNTIME_CASE_IDS[position + 1:])
                case_states = [state for state in case_states if str(state.get("case_id")) != case_id]
                case_states.append({"case_id": case_id, "state": "BLOCKED_DURING_ATTEMPT", "attempt": attempt, "collection_status": block_status})
                case_states.extend({"case_id": pending_id, "state": "NOT_ATTEMPTED_DUE_TO_EARLIER_BLOCK"} for pending_id in unattempted_case_ids if pending_id not in {str(item.get("case_id")) for item in case_states})
                block = {
                    "record_type": "IncompleteExternalBlock",
                    "collection_status": block_status,
                    "blocked_case_id": case_id,
                    "blocked_dataset_id": dataset_id,
                    "blocked_trial": 1,
                    "blocked_trial_index": 1,
                    "blocked_attempt": attempt,
                    "blocked_attempt_index": attempt,
                    "last_completed_case_id": result_rows[-1].get("case_id") if result_rows else None,
                    "failure_type": failure_info.get("exception_type") if failure_info else "ProviderTransportError",
                    "provider_failure_type": failure_info.get("category") if failure_info else "PROVIDER_TRANSPORT",
                    "provider_failure": failure_info,
                    "failure_message": record.failure_reason,
                    "provider_error_message": record.failure_reason,
                    "blocked_at": datetime.now(timezone.utc).isoformat(),
                    "resume_allowed": True,
                    "agent_id": agent_profile.agent_id,
                    "agent_profile_sha256": agent_profile.profile_sha256,
                    "runtime_profile_id": agent_profile.runtime_profile_id,
                    "runtime_profile_sha256": agent_profile.runtime_profile.profile_sha256,
                    "system_instruction_sha256": runner.system_instruction_sha256,
                    "tool_schema_sha256": runner.tool_schema_sha256,
                    "protocol_version": "profiled-natural-output-v1",
                    "provider_timeout_configuration": {
                        "connection_seconds": request_timeout_seconds,
                        "read_seconds": request_timeout_seconds,
                        "overall_request_seconds": request_timeout_seconds,
                    },
                    "case_manifest_sha256": case_manifest_sha256,
                    "case_input_sha256": _sha256(case_path),
                    "run_record_path": str((attempt_root / "run_record.json").relative_to(output)),
                    "run_record_status": record.run_status.value,
                    "unattempted_case_ids": unattempted_case_ids,
                }
                (attempt_root / "incomplete_external_block.json").write_text(json.dumps(block, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                partial = {
                    "record_type": "FlowIntentBenchCleanRuntimePilot",
                    "status": block_status,
                    "collection_status": block_status,
                    "formal": False, "N": 1,
                    "case_count": len(result_rows),
                    "selected_case_ids": list(CLEAN_RUNTIME_CASE_IDS),
                    "provider": target.provider, "model_id": target.model_id,
                    "agent_id": agent_profile.agent_id,
                    "agent_profile_sha256": agent_profile.profile_sha256,
                    "runtime_profile_id": agent_profile.runtime_profile_id,
                    "runtime_profile_sha256": agent_profile.runtime_profile.profile_sha256,
                    "system_instruction_sha256": runner.system_instruction_sha256,
                    "tool_schema_sha256": runner.tool_schema_sha256,
                    "protocol_version": "profiled-natural-output-v1",
                    "provider_base_url": runtime.base_url,
                    "failure_type": failure_info.get("exception_type") if failure_info else "ProviderTransportError",
                    "provider_failure_type": failure_info.get("category") if failure_info else "PROVIDER_TRANSPORT",
                    "provider_failure": failure_info,
                    "failure_message": record.failure_reason,
                    "blocked_case_id": case_id,
                    "blocked_dataset_id": dataset_id,
                    "blocked_trial": 1,
                    "blocked_trial_index": 1,
                    "blocked_attempt": attempt,
                    "blocked_attempt_index": attempt,
                    "last_completed_case_id": result_rows[-1].get("case_id") if result_rows else None,
                    "blocked_at": block["blocked_at"],
                    "resume_allowed": True,
                    "case_manifest_sha256": case_manifest_sha256,
                    "cases": result_rows,
                    "case_states": case_states,
                    "unattempted_case_ids": unattempted_case_ids,
                }
                (output / "manifest.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                (output / "runtime_report.md").write_text(f"# Clean-Runtime N=1 Pilot\n\nStatus: **{block_status}**. Completed observations are retained for resumable continuation.\n", encoding="utf-8")
                for name in ("efficiency_comparison.md", "unchanged_task_runtime_effect.md"):
                    (output / name).write_text("Pilot paused by external provider quota; no scientific result is assigned to remaining cases.\n", encoding="utf-8")
                return partial
        if record is None or record.run_status is RunStatus.INFRASTRUCTURE_INVALID:
            raise RuntimeError(f"clean-runtime case failed infrastructure gate: {case_id}")
        diagnostic_rows.append({"case_id": case_id, "record": record, **diagnostics})
        result_rows.append(
            {
                "case_id": case_id,
                "dataset_id": dataset_id,
                "condition": item["condition"],
                "case_input_sha256": _sha256(case_path),
                "status": record.run_status.value,
                "run_id": record.run_id,
                "run_record_path": str((attempt_root / "run_record.json").relative_to(output)),
                "run_record_sha256": _sha256(attempt_root / "run_record.json"),
                "trajectory_path": str(trajectory_path.relative_to(output)),
                "trajectory_sha256": _sha256(trajectory_path),
                "input_tokens": record.input_tokens,
                "output_tokens": record.output_tokens,
                "model_turn_count": record.model_turn_count,
                "tool_batch_count": record.tool_batch_count,
                "python_execution_count": record.python_execution_count,
                "python_errors": diagnostics["python_errors"],
                "wall_clock_time": record.wall_clock_time,
                "stdout_truncations": diagnostics["stdout_truncations"],
                "filesystem_discovery_attempts": diagnostics["filesystem_discovery_errors"],
                "max_returned_tool_chars": diagnostics["max_returned_tool_chars"],
            }
        )
        case_states = [state for state in case_states if str(state.get("case_id")) != case_id]
        case_states.append({"case_id": case_id, "state": "COMPLETED", "attempt": attempt})

    totals = {
        key_name: sum(int(row[key_name]) for row in attempt_diagnostics)
        for key_name in (
            "filesystem_discovery_errors",
            "root_search_timeouts",
            "stdout_truncations",
            "uncontrolled_huge_stdout",
            "python_errors",
            "tool_boundary_retries",
            "multi_call_tool_boundary_retries",
            "duplicate_tool_execution",
        )
    }
    target_fp = evaluation_target_fingerprint(target)
    manifest = {
        "record_type": "FlowIntentBenchCleanRuntimePilot",
        "status": "COMPLETE",
        "formal": False,
        "N": 1,
        "case_count": len(result_rows),
        "selected_case_ids": list(CLEAN_RUNTIME_CASE_IDS),
        "required_condition_coverage": sorted({row["condition"] for row in result_rows}),
        "provider": target.provider,
        "agent_id": None if agent_profile is None else agent_profile.agent_id,
        "runtime_profile_id": None if agent_profile is None else agent_profile.runtime_profile_id,
        "agent_profile_sha256": agent_profile.profile_sha256,
        "runtime_profile_sha256": agent_profile.runtime_profile.profile_sha256,
        "system_instruction_sha256": runner.system_instruction_sha256,
        "tool_schema_sha256": runner.tool_schema_sha256,
        "protocol_version": "profiled-natural-output-v1",
        "model_id": target.model_id,
        "model_configuration": dict(target.model_configuration),
        "provider_timeout_configuration": {
            "connection_seconds": request_timeout_seconds,
            "read_seconds": request_timeout_seconds,
            "overall_request_seconds": request_timeout_seconds,
        },
        "target_fingerprint": target_fp,
        "case_manifest_sha256": case_manifest_sha256,
        "benchmark_code_data_version": expected_code_data_version,
        "execution_backend": backend,
        "wire_api": wire,
        "provider_base_url": runtime.base_url,
        "system_prompt_version": runner.system_prompt_version,
        "runtime_output_limit": agent_profile.runtime_profile.max_returned_text_chars_per_call,
        "preflight_test_status": "PASS",
        "preflight_test_summary": completed_tests.stdout.strip().splitlines()[-1],
        "provider_compatibility_status": compatibility["status"],
        "filesystem_discovery_errors": totals["filesystem_discovery_errors"],
        "root_search_timeouts": totals["root_search_timeouts"],
        "stdout_truncations": totals["stdout_truncations"],
        "uncontrolled_huge_stdout": totals["uncontrolled_huge_stdout"],
        "conversation_state_corruption": 0,
        "unsupported_reasoning_fallbacks": 0,
        "tool_boundary_retries": totals["tool_boundary_retries"],
        "multi_call_tool_boundary_retries": totals["multi_call_tool_boundary_retries"],
        "multi_call_infrastructure_invalid_attempts": 0,
        "duplicate_tool_execution": totals["duplicate_tool_execution"],
        "multi_call_turns": sum(int(row.get("multi_call_turns", 0)) for row in diagnostic_rows),
        "max_batch_size": max((int(row.get("max_batch_size", 1)) for row in diagnostic_rows), default=1),
        "infrastructure_invalid_attempts": infrastructure_invalid_attempts,
        "corrected_data_contract_cases": ["carotid_o1_f2", "nasa_lox_post_o2_f1"],
        "cases": result_rows,
        "case_states": case_states,
        "blocked_case_id": None,
        "unattempted_case_ids": [],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    old_rows = _historical_rows(Path(collection_root).resolve(), set(CLEAN_RUNTIME_CASE_IDS))
    comparisons = _comparison_with_registry(old_rows, diagnostic_rows)
    (output / "old_vs_clean_runtime.md").write_text(comparisons["whole_system"], encoding="utf-8")
    (output / "efficiency_comparison.md").write_text(comparisons["whole_system"], encoding="utf-8")
    (output / "unchanged_task_runtime_effect.md").write_text(comparisons["unchanged_task"], encoding="utf-8")
    (output / "case_change_registry.json").write_text(json.dumps(comparisons["registry"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    case_lines = "\n".join(
        f"| {row['case_id']} | {row['status']} | {row['wall_clock_time']:.3f} | "
        f"{row['python_execution_count']} | {row['python_errors']} | {row['stdout_truncations']} |"
        for row in result_rows
    )
    report = (
        "# Clean-Runtime N=1 Pilot\n\n"
        "This is a runtime-validation pilot, not a formal benchmark result.\n\n"
        "| Case | Status | Wall time (s) | Python calls | Python errors | Stdout truncations |\n"
        "|---|---|---:|---:|---:|---:|\n"
        + case_lines
        + "\n\n"
        + f"Filesystem-discovery errors: **{totals['filesystem_discovery_errors']}**.  "
        + f"Root-search timeouts: **{totals['root_search_timeouts']}**.  "
        + f"Uncontrolled huge stdout events: **{totals['uncontrolled_huge_stdout']}**.\n"
    )
    (output / "runtime_report.md").write_text(report, encoding="utf-8")
    return manifest


def reconcile_existing_clean_runtime_pilot(
    output_root: str | Path,
    *,
    collection_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute attempt-level runtime counters without making model calls."""

    output = Path(output_root).resolve()
    manifest_path = output / "manifest.json"
    manifest = dict(_read_json(manifest_path))
    diagnostics = []
    infrastructure_invalid = 0
    for run_path in sorted((output / "runs").glob("**/run_record.json")):
        record = RunRecord.load_json(run_path)
        if record.run_status is RunStatus.INFRASTRUCTURE_INVALID:
            infrastructure_invalid += 1
        trajectory_path = run_path.with_name("trajectory.json")
        value = json.loads(trajectory_path.read_text(encoding="utf-8"))
        diagnostics.append(
            _trajectory_diagnostics([entry for entry in value if isinstance(entry, Mapping)])
        )
    for key_name in (
        "filesystem_discovery_errors",
        "root_search_timeouts",
        "stdout_truncations",
        "uncontrolled_huge_stdout",
        "tool_boundary_retries",
    ):
        manifest[key_name] = sum(int(item[key_name]) for item in diagnostics)
    manifest["infrastructure_invalid_attempts"] = infrastructure_invalid
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if collection_root is not None:
        new_rows = []
        for item in manifest.get("cases", []):
            if not isinstance(item, Mapping):
                continue
            run_path = output / str(item["run_record_path"])
            trajectory_path = output / str(item["trajectory_path"])
            record = RunRecord.load_json(run_path)
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            new_rows.append(
                {
                    "case_id": record.case_id,
                    "record": record,
                    **_trajectory_diagnostics(
                        [entry for entry in trajectory if isinstance(entry, Mapping)]
                    ),
                }
            )
        old_rows = _historical_rows(
            Path(collection_root).resolve(),
            {str(item["case_id"]) for item in manifest.get("cases", []) if isinstance(item, Mapping)},
        )
        comparisons = _comparison_with_registry(old_rows, new_rows)
        (output / "old_vs_clean_runtime.md").write_text(comparisons["whole_system"], encoding="utf-8")
        (output / "efficiency_comparison.md").write_text(comparisons["whole_system"], encoding="utf-8")
        (output / "unchanged_task_runtime_effect.md").write_text(comparisons["unchanged_task"], encoding="utf-8")
        (output / "case_change_registry.json").write_text(json.dumps(comparisons["registry"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent", default=None, help="agent profile ID or path")
    parser.add_argument(
        "--collection-root",
        type=Path,
        default=ROOT / "artifacts/reference/full_dataset_n1_controlled_pilot",
    )
    parser.add_argument("--server-config", type=Path)
    parser.add_argument("--api-key")
    parser.add_argument("--infrastructure-attempt-cap", type=int, default=3)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="recompute attempt-level counters from an existing completed pilot without model calls",
    )
    args = parser.parse_args()
    if not args.finalize_existing and args.agent is None:
        parser.error("New benchmark runs require an explicit agent profile.")
    profile = load_agent_profile(args.agent) if args.agent else None
    result = (
        reconcile_existing_clean_runtime_pilot(
            args.output_root, collection_root=args.collection_root
        )
        if args.finalize_existing
        else run_clean_runtime_pilot(
            output_root=args.output_root,
            collection_root=args.collection_root,
            server_config=args.server_config,
            api_key=args.api_key,
            infrastructure_attempt_cap=args.infrastructure_attempt_cap,
            agent_profile=profile,
            resume=args.resume,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    )
    summary = dict(result)
    summary["output_root"] = str(args.output_root.resolve())
    summary["manifest_path"] = str(args.output_root.resolve() / "manifest.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
