#!/usr/bin/env python3
"""Collect 96 x 3 blind case runs for GPT-6 Astra and GPT-5.6 Sol.

This entry point intentionally stops at model-run collection.  It does not
import or invoke an evaluator, reviewer, scorer, or Ground Truth loader.  Each
case is presented through the existing ``BenchmarkRunner`` contract, which
stages only the case input and its declared data/geometry files into the
model-visible runtime and exposes only the Python tool.

The GPT-6 label in this repository is the concrete ``gpt-6-astra`` target.
Both targets use the local YiAPI Responses route with ``xhigh`` reasoning. The
default ``flow-python-v1`` runtime is the strict Linux/no-network path; the
explicit ``--windows-local`` path is a development-only native Windows local
process and is recorded as not strictly comparable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import statistics
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import CaseLoader, DatasetManifest, EvaluationTarget  # noqa: E402
from flowintentbench.agent_profile import load_agent_profile  # noqa: E402
from flowintentbench.experiment_scope import case_inventory  # noqa: E402
from flowintentbench.model_runner import (  # noqa: E402
    BenchmarkRunner,
    RunStatus,
    schedule_formal_trials,
)
from flowintentbench.runtime_config import (  # noqa: E402
    resolve_api_key,
    resolve_provider_configuration,
    validate_reasoning_effort,
)
from flowintentbench.providers import ThirdPartyResponsesAdapter  # noqa: E402
from scripts.portable_fcntl import fcntl, process_exists  # noqa: E402
from scripts.yiapi_transport import responses_transport  # noqa: E402


MANIFEST = ROOT / "datasets/expansion_v1/case_manifest.json"
DATASETS_ROOT = ROOT / "datasets"
DEFAULT_OUTPUT = ROOT / "outputs/expansion96_n3_gpt6_sol"
CONFIG_PATH = ROOT / "config.toml"
SEED = 20260914
REPETITIONS = 3
CASE_COUNT = 96
CASE_TIMEOUT_SECONDS = 1800.0
PROVIDER_RETRIES = 5
PROVIDER_RETRY_BACKOFF_SECONDS = 1.0
# YiAPI account-level 503s can outlast the short transport retry window.  Use
# a real cooldown so parallel case workers do not immediately re-trigger the
# same account limit and turn a transient outage into terminal invalid slots.
PROVIDER_RATE_LIMIT_BACKOFF_SECONDS = 45.0
PROVIDER_RATE_LIMIT_MAX_BACKOFF_SECONDS = 300.0
MODELS = ("gpt-6-astra", "gpt-5.6-sol")
PROFILES = {
    "gpt-6-astra": ROOT / "agents/gpt-6-astra-xhigh/agent.yaml",
    "gpt-5.6-sol": ROOT / "agents/gpt-5.6-sol-xhigh/agent.yaml",
}
WINDOWS_PROFILES = {
    "gpt-6-astra": ROOT / "agents/gpt-6-astra-windows-local-xhigh/agent.yaml",
    "gpt-5.6-sol": ROOT / "agents/gpt-5.6-sol-windows-local-xhigh/agent.yaml",
}
TERMINAL = {
    RunStatus.COMPLETED.value,
    RunStatus.MODEL_NONCOMPLETION.value,
    RunStatus.INFRASTRUCTURE_INVALID.value,
}
SCHEMA = "gpt6-sol-case-collection-v1"


class RuntimePreflightError(RuntimeError):
    """The declared case runtime cannot be enforced on this host."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _output_lock(output: Path, name: str = ".collection.lock"):
    output.mkdir(parents=True, exist_ok=True)
    with (output / name).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another collection process owns {output}") from None
        yield


def _profile_paths(windows_local: bool) -> dict[str, Path]:
    return WINDOWS_PROFILES if windows_local else PROFILES


def _profile_data(windows_local: bool = False) -> dict[str, Any]:
    profile_paths = _profile_paths(windows_local)
    result = {}
    for model in MODELS:
        profile = load_agent_profile(profile_paths[model], repository_root=ROOT)
        if profile.model_id != model or profile.provider != "yiapi":
            raise ValueError(f"profile identity mismatch for {model}")
        if profile.reasoning_effort != "xhigh" or profile.wire_api.casefold() != "responses":
            raise ValueError(f"profile must use Responses/xhigh for {model}")
        result[model] = {
            "agent_id": profile.agent_id,
            "profile_path": str(profile_paths[model].relative_to(ROOT)),
            "profile_sha256": profile.profile_sha256,
            "model_id": profile.model_id,
            "provider": profile.provider,
            "model_family": profile.model_family,
            "reasoning_effort": profile.reasoning_effort,
            "wire_api": profile.wire_api,
            "runtime_profile_id": profile.runtime_profile_id,
            "runtime_profile_sha256": profile.runtime_profile.profile_sha256,
            "tools": list(profile.tools),
        }
    return result


def _load_inputs() -> tuple[dict[str, Mapping[str, Any]], dict[str, Any], dict[str, DatasetManifest]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = _read(MANIFEST)
    inventory = case_inventory(manifest, require_dataset=True)
    if len(inventory) != CASE_COUNT:
        raise ValueError(f"expected exactly {CASE_COUNT} cases, found {len(inventory)}")
    loader = CaseLoader(DATASETS_ROOT)
    loaded: dict[str, Any] = {}
    manifests: dict[str, DatasetManifest] = {}
    for case_id, row in inventory.items():
        source = (ROOT / str(row["case_input_path"])).resolve()
        if not source.is_relative_to(DATASETS_ROOT.resolve()):
            raise ValueError(f"case input escapes datasets root: {case_id}")
        if _sha256(source) != row["case_input_sha256"]:
            raise ValueError(f"case input checksum mismatch: {case_id}")
        loaded[case_id] = loader.load(source)
        dataset_id = str(row["dataset_id"])
        if dataset_id not in manifests:
            manifest_path = DATASETS_ROOT / dataset_id / "dataset_manifest.json"
            manifests[dataset_id] = DatasetManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
    return inventory, loaded, manifests


def _configuration(windows_local: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    profiles = _profile_data(windows_local)
    runtime_profile = "flow-python-windows-local-v1" if windows_local else "flow-python-v1"
    config = {
        "schema": SCHEMA,
        "models": list(MODELS),
        "repetitions": REPETITIONS,
        "case_count": CASE_COUNT,
        "base_seed": SEED,
        "reasoning_effort": "xhigh",
        "provider": "yiapi",
        "wire_api": "responses",
        "runtime_profile": runtime_profile,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS,
        "provider_retries": PROVIDER_RETRIES,
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": _sha256(MANIFEST),
        "profiles": profiles,
        "runner": "flowintentbench.model_runner.BenchmarkRunner",
        "collection_only": True,
        "evaluation": "DISABLED_BY_ENTRYPOINT",
        "windows_local": windows_local,
        "security_mode": "WINDOWS_LOCAL_UNISOLATED" if windows_local else "LINUX_BUBBLEWRAP",
        "strictly_comparable": not windows_local,
    }
    return config, profiles, _read(MANIFEST)


def _slot_id(model: str, trial: int, case_id: str) -> str:
    return f"{model}__trial-{trial:02d}__{case_id}"


def _prepare(output: Path, *, windows_local: bool = False) -> dict[str, Any]:
    config, profiles, manifest = _configuration(windows_local)
    inventory, _loaded, _manifests = _load_inputs()
    binding = _digest(config)
    state_path = output / "collection_state.json"
    with _output_lock(output):
        if state_path.exists():
            state = _read(state_path)
            if state.get("configuration_sha256") != binding:
                raise ValueError("existing output was created with a different collection configuration")
            return state
        family_by_id = {
            case_id: str(row.get("family_id", case_id)) for case_id, row in inventory.items()
        }
        schedule = schedule_formal_trials(
            tuple(inventory), repetitions=REPETITIONS, base_seed=SEED,
            case_family_by_id=family_by_id,
        )
        slots = []
        for trial in schedule:
            for position, case_id in enumerate(trial.case_ids):
                row = inventory[case_id]
                for model in MODELS:
                    slots.append({
                        "slot_id": _slot_id(model, trial.trial_index, case_id),
                        "model_id": model,
                        "case_id": case_id,
                        "dataset_id": row["dataset_id"],
                        "trial_index": trial.trial_index,
                        "ordering_seed": trial.ordering_seed,
                        "case_execution_position": position,
                        "status": "PENDING",
                        "run_record_path": None,
                        "failure_reason": None,
                    })
        state = {
            "schema_version": SCHEMA,
            "experiment_id": "gpt6-sol-collection-" + _digest({"config": binding, "epoch": time.time_ns()})[:16],
            "configuration": config,
            "configuration_sha256": binding,
            "manifest": {
                "path": str(MANIFEST.relative_to(ROOT)),
                "sha256": config["manifest_sha256"],
                "case_count": len(inventory),
            },
            "slots": slots,
            "status_counts": {"PENDING": len(slots)},
            "total_slots": len(slots),
            "collection_status": "COLLECTION_INCOMPLETE",
            "scientific_evaluation_status": "NOT_RUN",
            "blinding": {
                "ground_truth_loaded": False,
                "evaluation_material_loaded": False,
                "case_input_only": True,
                "runtime_profile": config["runtime_profile"],
                "network_available_to_model_tool": windows_local,
                "security_mode": config["security_mode"],
                "strictly_comparable": not windows_local,
            },
            "created_epoch": time.time(),
        }
        _write(output / "collection_config.json", config)
        _write(output / "collection_state.json", state)
        _write(output / "efficiency_summary.json", _efficiency_summary(output, state))
        _write(output / "process.json", {"status": "PREPARED", "updated_epoch": time.time()})
        return state


def _status_counts(state: Mapping[str, Any]) -> dict[str, int]:
    return dict(Counter(str(slot.get("status")) for slot in state.get("slots", [])))


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": None, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _efficiency_summary(output: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize response/runtime telemetry without evaluating answers."""

    by_model: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        records: list[dict[str, Any]] = []
        eligible_records: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        for slot in state.get("slots", []):
            if slot.get("model_id") != model or slot.get("status") not in TERMINAL:
                continue
            status = str(slot.get("status"))
            status_counts[status] += 1
            relative = slot.get("run_record_path")
            if not relative:
                continue
            path = (output / str(relative)).resolve()
            try:
                path.relative_to(output.resolve())
                record = _read(path)
            except (OSError, ValueError, json.JSONDecodeError, KeyError):
                continue
            if isinstance(record, dict):
                # Providers may omit usage on one response, causing RunRecord
                # to leave its strict aggregate fields null.  Efficiency is a
                # diagnostic-only view, so recover observed per-turn usage from
                # the immutable trajectory without changing the run status.
                if record.get("input_tokens") is None or record.get("output_tokens") is None:
                    trajectory_path = path.with_name("trajectory.json")
                    try:
                        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        trajectory = []
                    input_values = [
                        event.get("input_tokens") for event in trajectory
                        if isinstance(event, dict)
                        and event.get("event") == "model_turn"
                        and isinstance(event.get("input_tokens"), int)
                    ]
                    output_values = [
                        event.get("output_tokens") for event in trajectory
                        if isinstance(event, dict)
                        and event.get("event") == "model_turn"
                        and isinstance(event.get("output_tokens"), int)
                    ]
                    if record.get("input_tokens") is None and input_values:
                        record["input_tokens"] = sum(input_values)
                    if record.get("output_tokens") is None and output_values:
                        record["output_tokens"] = sum(output_values)
                    if record.get("total_tokens") is None and record.get("input_tokens") is not None and record.get("output_tokens") is not None:
                        record["total_tokens"] = record["input_tokens"] + record["output_tokens"]
                records.append(record)
                if status in {RunStatus.COMPLETED.value, RunStatus.MODEL_NONCOMPLETION.value}:
                    eligible_records.append(record)

        def values(field: str, source: list[dict[str, Any]] = eligible_records) -> list[float]:
            result = []
            for record in source:
                value = record.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    result.append(float(value))
            return result

        wall_clock = values("wall_clock_time")
        total_tokens = values("total_tokens")
        input_tokens = values("input_tokens")
        output_tokens = values("output_tokens")
        turns = values("model_turn_count")
        python_calls = values("python_execution_count")
        python_seconds = values("python_execution_total_time")
        tool_calls = values("tool_call_count")
        throughput = [
            record["total_tokens"] / record["wall_clock_time"]
            for record in eligible_records
            if isinstance(record.get("total_tokens"), (int, float))
            and isinstance(record.get("wall_clock_time"), (int, float))
            and record["wall_clock_time"] > 0
        ]
        by_model[model] = {
            "terminal_status_counts": dict(status_counts),
            "run_record_count": len(records),
            "efficiency_observation_count": len(eligible_records),
            "infrastructure_records_excluded_from_efficiency": len(records) - len(eligible_records),
            "wall_clock_seconds": _numeric_summary(wall_clock),
            "input_tokens": _numeric_summary(input_tokens),
            "output_tokens": _numeric_summary(output_tokens),
            "total_tokens": _numeric_summary(total_tokens),
            "model_turns": _numeric_summary(turns),
            "python_execution_count": _numeric_summary(python_calls),
            "python_execution_seconds": _numeric_summary(python_seconds),
            "tool_call_count": _numeric_summary(tool_calls),
            "tokens_per_wall_clock_second": _numeric_summary(throughput),
        }
    return {
        "schema": "gpt6-sol-case-efficiency-v1",
        "updated_epoch": time.time(),
        "evaluation": "NOT_RUN",
        "runtime_profile": state.get("configuration", {}).get("runtime_profile"),
        "security_mode": state.get("configuration", {}).get("security_mode"),
        "strictly_comparable": state.get("configuration", {}).get("strictly_comparable", True),
        "by_model": by_model,
    }


def _save_state(output: Path, state: dict[str, Any]) -> None:
    state["status_counts"] = _status_counts(state)
    state["total_slots"] = len(state["slots"])
    state["collection_status"] = (
        "COLLECTION_COMPLETE"
        if all(slot.get("status") in TERMINAL for slot in state["slots"])
        else "COLLECTION_INCOMPLETE"
    )
    state["updated_epoch"] = time.time()
    _write(output / "collection_state.json", state)


def _resolve_targets(windows_local: bool = False) -> dict[str, tuple[Any, EvaluationTarget, str, str]]:
    targets = {}
    profile_paths = _profile_paths(windows_local)
    for model in MODELS:
        profile = load_agent_profile(profile_paths[model], repository_root=ROOT)
        runtime = resolve_provider_configuration(
            role="model", config_path=CONFIG_PATH, provider=profile.provider,
            model_id=profile.model_id, model_configuration=profile.model_configuration,
        )
        validate_reasoning_effort(runtime.model_configuration, label=model)
        # Use the same local YiAPI credential as the Codex client.  The
        # repository auth.json is a separate historical account and may have
        # an unrelated quota state.
        key = resolve_api_key(runtime, auth_path=Path.home() / ".codex" / "auth.json")
        target = EvaluationTarget(runtime.provider, runtime.model_id, model_configuration=runtime.model_configuration)
        targets[model] = (profile, target, key, runtime.base_url)
    return targets


def _runtime_preflight(*, windows_local: bool = False) -> dict[str, Any]:
    """Verify the declared runtime before resolving or calling a provider.

    ``flow-python-v1`` is a Linux sandbox.  Native Windows is supported only
    through the explicit development profile, which records that its local
    process is not an OS-level security boundary.
    """

    checks = {
        "platform": sys.platform,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "bubblewrap": shutil.which("bwrap"),
        "af_unix": hasattr(socket, "AF_UNIX"),
        "runtime_profile": "flow-python-windows-local-v1" if windows_local else "flow-python-v1",
        "security_mode": "WINDOWS_LOCAL_UNISOLATED" if windows_local else "LINUX_BUBBLEWRAP",
        "provider_calls": 0,
    }
    if windows_local:
        if os.name != "nt":
            raise RuntimePreflightError(
                "--windows-local is only valid on native Windows; use the strict Linux "
                "profile in WSL/Linux instead. No provider call was made."
            )
        return checks
    if os.name == "nt":
        raise RuntimePreflightError(
            "native Windows cannot run flow-python-v1 safely: this profile requires "
            "a Linux bubblewrap network namespace and an inherited AF_UNIX control "
            "channel. No provider call was made. Run from Linux/WSL with the "
            "benchmark_py3.12 environment and bubblewrap installed; do not switch "
            "to an unisolated local or host-network runtime."
        )
    if not checks["af_unix"]:
        raise RuntimePreflightError(
            "flow-python-v1 requires AF_UNIX support in the active Python runtime; "
            "run from Linux/WSL with a POSIX Python build. No provider call was made."
        )
    if checks["bubblewrap"] is None:
        raise RuntimePreflightError(
            "flow-python-v1 requires bubblewrap for the no-network case runtime. "
            "Install bubblewrap in the Linux/WSL environment and retry. No provider "
            "call was made."
        )
    return checks


def preflight(output: Path, *, windows_local: bool = False) -> int:
    """Validate inputs, credentials, and runtime without making provider calls."""

    state = _prepare(output, windows_local=windows_local)
    checks = _runtime_preflight(windows_local=windows_local)
    _resolve_targets(windows_local=windows_local)
    print(json.dumps({
        "status": "READY",
        "output": str(output),
        "runtime": checks,
        "total_slots": state["total_slots"],
        "provider_calls": 0,
        "evaluation": "NOT_RUN",
        "strictly_comparable": not windows_local,
    }, ensure_ascii=False, indent=2))
    return 0


def _claim(output: Path, slot_id: str, state_lock: threading.Lock) -> dict[str, Any] | None:
    with state_lock:
        state = _read(output / "collection_state.json")
        slot = next(item for item in state["slots"] if item["slot_id"] == slot_id)
        if slot["status"] != "PENDING":
            return None
        slot.update(status="RUNNING", started_epoch=time.time(), worker_pid=os.getpid())
        _save_state(output, state)
        return dict(slot)


def _run_slot(output: Path, slot: Mapping[str, Any], loaded: Mapping[str, Any], manifests: Mapping[str, DatasetManifest], targets: Mapping[str, tuple[Any, EvaluationTarget, str, str]]) -> dict[str, Any]:
    model = str(slot["model_id"])
    profile, target, key, base_url = targets[model]
    case_id = str(slot["case_id"])
    run_dir = output / "runs" / model / f"trial-{int(slot['trial_index']):02d}" / case_id
    if slot.get("attempt_history"):
        run_dir = run_dir / f"attempt-{len(slot['attempt_history']) + 1:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    def adapter_factory(evaluation_target: EvaluationTarget):
        return ThirdPartyResponsesAdapter(
            evaluation_target,
            api_key=key,
            api_key_env="OPENAI_API_KEY",
            base_url=base_url,
            formal_mode=False,
            request_timeout_seconds=CASE_TIMEOUT_SECONDS,
            transport=responses_transport,
        )
    runner = BenchmarkRunner(
        formal_mode=False,
        agent_profile=profile,
        provider_retries=PROVIDER_RETRIES,
        provider_retry_backoff_seconds=PROVIDER_RETRY_BACKOFF_SECONDS,
        provider_rate_limit_backoff_seconds=PROVIDER_RATE_LIMIT_BACKOFF_SECONDS,
        provider_rate_limit_max_backoff_seconds=PROVIDER_RATE_LIMIT_MAX_BACKOFF_SECONDS,
        case_timeout_seconds=CASE_TIMEOUT_SECONDS,
        runtime_kwargs={
            "runtime_profile": profile.runtime_profile,
            "workspace_parent": output / "runtime-work",
        },
    )
    # The model target and endpoint are resolved before the worker starts. The
    # endpoint is passed separately to avoid ever serializing the API key.
    record = runner.run_case(
        loaded[case_id], target=target, case_id=case_id,
        trial_index=int(slot["trial_index"]), ordering_seed=int(slot["ordering_seed"]),
        case_execution_position=int(slot["case_execution_position"]),
        manifest=manifests[str(slot["dataset_id"])],
        trajectory_path=run_dir / "trajectory.json", adapter_factory=adapter_factory,
        experiment_id="gpt6-sol-case-collection", benchmark_release_id=None,
    )
    record.write_json(run_dir / "run_record.json")
    return {
        "slot_id": slot["slot_id"],
        "status": record.run_status.value,
        "run_record_path": str((run_dir / "run_record.json").relative_to(output)),
        "run_record_sha256": _sha256(run_dir / "run_record.json"),
        "failure_reason": record.failure_reason,
        "finished_epoch": time.time(),
    }


def _reset_slot(slot: dict[str, Any], reason: str) -> None:
    history = slot.setdefault("attempt_history", [])
    history.append({
        key: slot.get(key) for key in
        ("status", "run_record_path", "run_record_sha256", "failure_reason", "started_epoch", "finished_epoch")
    } | {"retry_reason": reason, "archived_epoch": time.time()})
    slot.update(status="PENDING", run_record_path=None, run_record_sha256=None, failure_reason=None)
    for key in ("started_epoch", "finished_epoch", "worker_pid"):
        slot.pop(key, None)


def _commit(output: Path, result: Mapping[str, Any], state_lock: threading.Lock, *, stopping: bool = False) -> None:
    with state_lock:
        state = _read(output / "collection_state.json")
        slot = next(item for item in state["slots"] if item["slot_id"] == result["slot_id"])
        slot.update({key: result[key] for key in ("status", "run_record_path", "run_record_sha256", "failure_reason", "finished_epoch")})
        slot.pop("worker_pid", None)
        if stopping and result["status"] == RunStatus.INFRASTRUCTURE_INVALID.value:
            _reset_slot(slot, "infrastructure failure while collector was stopping")
        _save_state(output, state)
        efficiency = _efficiency_summary(output, state)
        _write(output / "efficiency_summary.json", efficiency)
        _write(output / "progress.json", {
            "updated_epoch": time.time(),
            "status_counts": state["status_counts"],
            "completed": sum(value for name, value in state["status_counts"].items() if name in TERMINAL),
            "total": state["total_slots"],
            "collection_status": state["collection_status"],
            "evaluation": "NOT_RUN",
            "efficiency": efficiency,
        })


def _is_quota_exhausted(result: Mapping[str, Any]) -> bool:
    """Recognize account-quota failures that must pause collection, not consume slots."""
    reason = str(result.get("failure_reason") or "").lower()
    return "insufficient_user_quota" in reason or "用户额度不足" in reason


def _recover_stale_running(output: Path, state: dict[str, Any]) -> None:
    running = [slot for slot in state["slots"] if slot.get("status") == "RUNNING"]
    if not running:
        return
    process = _read(output / "process.json") if (output / "process.json").exists() else {}
    pid = int(process.get("pid", 0) or 0)
    if pid and process_exists(pid):
        raise RuntimeError(f"collection process {pid} is still running; do not launch a duplicate")
    # A missing collector process means these claims cannot complete or be
    # committed.  Preserve their metadata in attempt_history and make the
    # slots retryable; this lets a long Windows run resume after a terminal,
    # host restart, or other external interruption without losing prior data.
    reason = f"stale running slot recovered; collector pid {pid or 'unknown'} is not alive"
    for slot in running:
        _reset_slot(slot, reason)
    _save_state(output, state)
    _write(output / "efficiency_summary.json", _efficiency_summary(output, state))
    _write(output / "progress.json", {
        "updated_epoch": time.time(),
        "status_counts": state["status_counts"],
        "completed": sum(value for name, value in state["status_counts"].items() if name in TERMINAL),
        "total": state["total_slots"],
        "collection_status": state["collection_status"],
        "evaluation": "NOT_RUN",
        "efficiency": _efficiency_summary(output, state),
        "recovery": {"recovered_stale_running": len(running), "reason": reason},
    })


def collect(
    output: Path,
    *,
    workers: int,
    max_slots: int | None,
    selected_models: tuple[str, ...],
    windows_local: bool = False,
    retry_infrastructure: bool = False,
) -> int:
    # Hold this lock for the whole collection, including preparation and retries.
    with _output_lock(output, ".collector.lock"):
        return _collect_locked(output, workers=workers, max_slots=max_slots,
                               selected_models=selected_models, windows_local=windows_local,
                               retry_infrastructure=retry_infrastructure)


def _collect_locked(output: Path, *, workers: int, max_slots: int | None,
                    selected_models: tuple[str, ...], windows_local: bool,
                    retry_infrastructure: bool) -> int:
    state = _prepare(output, windows_local=windows_local)
    if retry_infrastructure:
        process = _read(output / "process.json") if (output / "process.json").exists() else {}
        pid = int(process.get("pid", 0) or 0)
        if process.get("status") in {"STOPPED", "STOP_REQUESTED", "STALE", "IDLE"} or not process_exists(pid):
            for slot in state["slots"]:
                if slot["status"] == "RUNNING" and slot["model_id"] in selected_models:
                    _reset_slot(slot, "stale running slot from an interrupted collector")
            _save_state(output, state)
    _recover_stale_running(output, state)
    _runtime_preflight(windows_local=windows_local)
    targets = _resolve_targets(windows_local=windows_local)
    if retry_infrastructure:
        for slot in state["slots"]:
            if slot["status"] == RunStatus.INFRASTRUCTURE_INVALID.value and slot["model_id"] in selected_models:
                _reset_slot(slot, "explicit infrastructure retry")
        _save_state(output, state)
        _write(output / "efficiency_summary.json", _efficiency_summary(output, state))
    inventory, loaded, manifests = _load_inputs()
    del inventory
    state_lock = threading.Lock()
    stop_event = threading.Event()
    process_path = output / "process.json"
    _write(process_path, {"status": "RUNNING", "pid": os.getpid(), "started_epoch": time.time(), "updated_epoch": time.time()})
    if windows_local:
        print(json.dumps({
            "event": "security_warning",
            "message": (
                "native Windows local mode uses staged paths and a local process only; "
                "network and host filesystem access are not OS-isolated, so these runs "
                "are development-only and not strictly comparable to Linux bubblewrap runs"
            ),
        }, ensure_ascii=False), flush=True)
    def request_stop(signum, frame):
        del signum, frame
        stop_event.set()
        _write(process_path, {"status": "STOP_REQUESTED", "pid": os.getpid(), "updated_epoch": time.time()})
        print(json.dumps({"event": "stop_requested", "message": "finishing claimed slots; no new slots will start"}, ensure_ascii=False), flush=True)
    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    launched = 0
    futures = {}
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="case-collector") as pool:
            while not stop_event.is_set() and (max_slots is None or launched < max_slots):
                stop_path = output / "stop_request.json"
                if stop_path.exists() and _read(stop_path).get("pid") == os.getpid():
                    request_stop(None, None)
                    break
                state = _read(output / "collection_state.json")
                pending = [slot for slot in state["slots"] if slot.get("status") == "PENDING" and slot.get("model_id") in selected_models]
                # Finish all cases of the first selected model before the next.
                active_model = next((model for model in selected_models if any(
                    slot["model_id"] == model and slot["status"] in {"PENDING", "RUNNING"}
                    for slot in state["slots"])), None)
                pending = [slot for slot in pending if slot["model_id"] == active_model]
                while pending and len(futures) < workers and not stop_event.is_set() and (max_slots is None or launched + len(futures) < max_slots):
                    candidate = pending.pop(0)
                    claimed = _claim(output, str(candidate["slot_id"]), state_lock)
                    if claimed is None:
                        continue
                    futures[pool.submit(_run_slot, output, claimed, loaded, manifests, targets)] = claimed["slot_id"]
                if not futures:
                    break
                done, _ = wait(tuple(futures), timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    slot_id = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"slot_id": slot_id, "status": RunStatus.INFRASTRUCTURE_INVALID.value,
                                  "run_record_path": None, "run_record_sha256": None,
                                  "failure_reason": f"collector worker exception: {type(exc).__name__}: {exc}",
                                  "finished_epoch": time.time()}
                    if _is_quota_exhausted(result) and not stop_event.is_set():
                        # Account quota is an external pause condition.  Do not
                        # burn the remaining slots as terminal failures; the
                        # stopping commit resets the claimed slot to PENDING.
                        request_stop(None, None)
                    _commit(output, result, state_lock, stopping=stop_event.is_set())
                    launched += 1
                    print(json.dumps({"event": "case_finished", **result}, ensure_ascii=False), flush=True)
            # Drain already claimed work after a stop request; no new slots are claimed.
            while futures:
                done, _ = wait(tuple(futures), timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    slot_id = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"slot_id": slot_id, "status": RunStatus.INFRASTRUCTURE_INVALID.value,
                                  "run_record_path": None, "run_record_sha256": None,
                                  "failure_reason": f"collector worker exception: {type(exc).__name__}: {exc}",
                                  "finished_epoch": time.time()}
                    _commit(output, result, state_lock, stopping=stop_event.is_set())
                    print(json.dumps({"event": "case_finished", **result}, ensure_ascii=False), flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        final_state = _read(output / "collection_state.json")
        final_status = "STOPPED" if stop_event.is_set() else ("COMPLETE" if final_state["collection_status"] == "COLLECTION_COMPLETE" else "IDLE")
        _write(process_path, {"status": final_status, "pid": os.getpid(), "stopped_epoch": time.time(), "updated_epoch": time.time()})
    return 0 if _read(output / "collection_state.json")["collection_status"] == "COLLECTION_COMPLETE" else 2


def stop(output: Path) -> int:
    process = _read(output / "process.json")
    pid = int(process.get("pid", 0) or 0)
    if process.get("status") not in {"RUNNING", "STOP_REQUESTED"} or not process_exists(pid):
        print(json.dumps({"event": "not_running", "pid": pid}), flush=True)
        return 0
    _write(output / "stop_request.json", {"pid": pid, "requested_epoch": time.time()})
    print(json.dumps({"event": "stop_requested", "pid": pid,
                      "message": "current cases will finish; no new cases will start"}), flush=True)
    return 0


def status(output: Path, *, as_json: bool, watch: float) -> int:
    if not (output / "collection_state.json").exists():
        payload = {"status": "NOT_PREPARED", "output": str(output)}
        print(json.dumps(payload, ensure_ascii=False) if as_json else payload)
        return 1
    while True:
        state = _read(output / "collection_state.json")
        process = _read(output / "process.json") if (output / "process.json").exists() else {}
        if process.get("status") == "RUNNING":
            pid = int(process.get("pid", 0) or 0)
            if not process_exists(pid):
                process = {**process, "status": "STALE"}
        counts = _status_counts(state)
        efficiency = _efficiency_summary(output, state)
        payload = {
            "output": str(output),
            "collection_status": state.get("collection_status"),
            "process": process,
            "status_counts": counts,
            "completed": sum(counts.get(name, 0) for name in TERMINAL),
            "total": len(state.get("slots", [])),
            "by_model": {
                model: dict(Counter(slot.get("status") for slot in state.get("slots", []) if slot.get("model_id") == model))
                for model in MODELS
            },
            "evaluation": "NOT_RUN",
            "efficiency": efficiency,
            "runtime_profile": state.get("configuration", {}).get("runtime_profile"),
            "security_mode": state.get("configuration", {}).get("security_mode"),
            "strictly_comparable": state.get("configuration", {}).get("strictly_comparable", True),
            "updated_epoch": state.get("updated_epoch"),
        }
        if as_json:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"{payload['collection_status']}  {payload['completed']}/{payload['total']}")
            for model, counts_for_model in payload["by_model"].items():
                print(f"  {model}: {counts_for_model}")
            print(f"  process: {process.get('status', 'UNKNOWN')} pid={process.get('pid', '-')}")
            print("  evaluation: NOT_RUN")
            print(f"  security: {payload['security_mode']} comparable={payload['strictly_comparable']}")
            for model, summary in efficiency["by_model"].items():
                wall = summary["wall_clock_seconds"]["mean"]
                tokens = summary["total_tokens"]["mean"]
                rate = summary["tokens_per_wall_clock_second"]["mean"]
                print(f"  efficiency {model}: mean_wall_s={wall} mean_tokens={tokens} tokens_per_s={rate}")
            print(flush=True)
        if not watch or state.get("collection_status") == "COLLECTION_COMPLETE" or process.get("status") in {"PREPARED", "STALE", "STOPPED", "STOP_REQUESTED", "IDLE", "COMPLETE"}:
            return 0
        time.sleep(watch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "collect", "status", "stop"))
    parser.add_argument("--retry-infrastructure", action="store_true",
                        help="retry infrastructure-invalid slots, preserving prior attempt artifacts")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-slots", type=int, default=None, help="bounded smoke run; omit for all 576 slots")
    parser.add_argument("--model", dest="models", action="append", choices=MODELS,
                        help="restrict collection to one model; may be repeated")
    parser.add_argument("--watch", type=float, default=0, help="status refresh interval in seconds; 0 prints once")
    parser.add_argument("--json", action="store_true", help="emit machine-readable status JSON")
    parser.add_argument(
        "--windows-local",
        action="store_true",
        help=(
            "use the native Windows staged local runtime; network/filesystem isolation "
            "is not OS-enforced and results are development-only"
        ),
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.max_slots is not None and args.max_slots < 1:
        parser.error("--max-slots must be positive")
    if args.watch < 0:
        parser.error("--watch must be non-negative")
    if args.command == "prepare":
        state = _prepare(output, windows_local=args.windows_local)
        print(json.dumps({
            "output": str(output),
            "total_slots": state["total_slots"],
            "status_counts": state["status_counts"],
            "collection_status": state["collection_status"],
            "evaluation": "NOT_RUN",
            "security_mode": state["configuration"]["security_mode"],
            "strictly_comparable": state["configuration"]["strictly_comparable"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "preflight":
        try:
            return preflight(output, windows_local=args.windows_local)
        except RuntimePreflightError as exc:
            print(json.dumps({
                "status": "BLOCKED_ENVIRONMENT",
                "output": str(output),
                "runtime_profile": "flow-python-windows-local-v1" if args.windows_local else "flow-python-v1",
                "provider_calls": 0,
                "evaluation": "NOT_RUN",
                "message": str(exc),
            }, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
    if args.command == "status":
        return status(output, as_json=args.json, watch=args.watch)
    if args.command == "stop":
        return stop(output)
    selected = tuple(dict.fromkeys(args.models or MODELS))
    try:
        return collect(
            output,
            workers=args.workers,
            max_slots=args.max_slots,
            selected_models=selected,
            windows_local=args.windows_local,
            retry_infrastructure=args.retry_infrastructure,
        )
    except RuntimePreflightError as exc:
        print(json.dumps({
            "status": "BLOCKED_ENVIRONMENT",
            "output": str(output),
            "runtime_profile": "flow-python-windows-local-v1" if args.windows_local else "flow-python-v1",
            "provider_calls": 0,
            "evaluation": "NOT_RUN",
            "message": str(exc),
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
