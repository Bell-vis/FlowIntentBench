#!/usr/bin/env python3
"""Run the canonical full-dataset N=1 experiment orchestration.

The script is deliberately an orchestration boundary, not a second evaluator.
It reads the frozen case manifest, a selected agent profile, an evaluator JSON
configuration, and the production continuation-handler manifest.  It never
reads historical ``replay_args.json`` files and never changes Case/GT or the
scientific scoring rules.

The command is resumable.  Collection and evaluation are separate persisted
roots, evaluator infrastructure failures are retryable, and legitimate
scientific pending values remain null in the final report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluation_records import iter_current_evaluation_records
from flowintentbench.agent_profile import load_agent_profile
from flowintentbench.model_response_reuse import audit_model_response_reuse
from flowintentbench.pending_adjudication import load_pending_continuation_registry
from flowintentbench.runtime_config import (
    resolve_api_key,
    resolve_provider_configuration,
)
from flowintentbench.experiment_scope import case_inventory
from scripts.run_full_dataset_n1_pilot import _assert_preflight_binding
from scripts.preflight_full_dataset_n1 import validate_manifest


def _read_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return dict(value)


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_reliability_summary(
    *,
    output_path: Path,
    collection_state: Mapping[str, Any],
    collection_root: Path,
    evaluation_root: Path,
    report_root: Path,
    evaluator_circuit_breaker_activations: int = 0,
    evaluator_cooldown_seconds: float = 0.0,
) -> None:
    """Persist transport/retry telemetry separately from scientific metrics."""

    rows: list[dict[str, Any]] = []
    csv_path = report_root / "infrastructure_reliability.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    collection_categories: Counter[str] = Counter()
    collection_retry_count = 0
    collection_failed_request_time = 0.0
    collection_retry_backoff_time = 0.0
    collection_request_count = 0
    collection_rate_limit_count = 0
    collection_timeout_count = 0
    collection_unavailable_count = 0
    for observation in collection_state.get("observations", ()):
        if not isinstance(observation, Mapping):
            continue
        reliability = observation.get("reliability")
        failure = reliability.get("provider_failure") if isinstance(reliability, Mapping) else None
        if isinstance(failure, Mapping) and failure.get("category"):
            collection_categories[str(failure["category"])] += 1
        if isinstance(reliability, Mapping):
            retry_count = reliability.get("infrastructure_retry_count")
            failed_time = reliability.get("infrastructure_failed_request_time")
            backoff_time = reliability.get("retry_backoff_time")
            if isinstance(retry_count, int) and not isinstance(retry_count, bool):
                collection_retry_count += retry_count
            if isinstance(failed_time, (int, float)) and not isinstance(failed_time, bool):
                collection_failed_request_time += float(failed_time)
            if isinstance(backoff_time, (int, float)) and not isinstance(backoff_time, bool):
                collection_retry_backoff_time += float(backoff_time)
            run_record_path = observation.get("run_record_path")
            if isinstance(run_record_path, str):
                run_record_file = collection_root / run_record_path
                try:
                    run_record = _json_object(run_record_file, label="collection RunRecord")
                    turns = run_record.get("model_turn_count")
                    if isinstance(turns, int) and not isinstance(turns, bool):
                        collection_request_count += turns
                    trajectory_file = run_record_file.parent / "trajectory.json"
                    if trajectory_file.is_file():
                        trajectory = _read_json(trajectory_file)
                        if isinstance(trajectory, list):
                            for event in trajectory:
                                if not isinstance(event, Mapping) or event.get("event") != "infrastructure_retry":
                                    continue
                                category = str(event.get("category") or "").upper()
                                collection_request_count += 1
                                if category == "RATE_LIMIT":
                                    collection_rate_limit_count += 1
                                elif category == "TIMEOUT":
                                    collection_timeout_count += 1
                                elif category == "UNAVAILABLE":
                                    collection_unavailable_count += 1
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
    # Reliability totals describe immutable attempts, not only the selected
    # current record.  Count each archived attempt exactly once.  A stale
    # root-level sidecar may survive after a successful pending/final record;
    # it is the mutable current marker, not an additional failed request.
    evaluator_failures = 0
    evaluator_categories: Counter[str] = Counter()
    all_failure_sidecars = list(
        evaluation_root.rglob("evaluator_backend_failure.json")
    )
    archived_failure_sidecars = [
        path for path in all_failure_sidecars if "evaluation_attempts" in path.parts
    ]
    current_inventory = _record_inventory(evaluation_root)
    selected_infrastructure_dirs = {
        Path(value["path"]).parent.resolve()
        for value in current_inventory.values()
        if value.get("status") == "INFRASTRUCTURE_INVALID"
        and isinstance(value.get("path"), str)
    }
    selected_failure_sidecars = [
        path
        for path in all_failure_sidecars
        if "evaluation_attempts" not in path.parts
        and path.parent.resolve() in selected_infrastructure_dirs
    ]
    for sidecar in [*archived_failure_sidecars, *selected_failure_sidecars]:
        evaluator_failures += 1
        try:
            failure = _json_object(sidecar, label="evaluator backend failure").get("provider_failure")
            if isinstance(failure, Mapping) and failure.get("category"):
                evaluator_categories[str(failure["category"])] += 1
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    _atomic_json(
        output_path,
        {
            "record_type": "N1ProviderReliabilitySummary",
            "schema_version": "n1-provider-reliability-v1",
            "collection_failure_categories": dict(sorted(collection_categories.items())),
            "collection_infrastructure_retry_count": collection_retry_count,
            "collection_failed_request_time": collection_failed_request_time,
            "collection_retry_backoff_time": collection_retry_backoff_time,
            "collection_provider_request_count": collection_request_count,
            "collection_rate_limit_count": collection_rate_limit_count,
            "collection_timeout_count": collection_timeout_count,
            "collection_unavailable_count": collection_unavailable_count,
            "evaluator_failure_categories": dict(sorted(evaluator_categories.items())),
            "evaluator_failure_sidecar_count": evaluator_failures,
            "evaluator_circuit_breaker_activations": evaluator_circuit_breaker_activations,
            "evaluator_cooldown_seconds": evaluator_cooldown_seconds,
            "rendered_reliability_rows": rows,
            "efficiency_and_scientific_metrics_separated": True,
        },
    )


def _build_experiment_manifest(
    *,
    manifest_path: Path,
    evaluator_config_path: Path,
    continuation_path: Path,
    preflight_path: Path,
    runtime_preflight_path: Path,
    tested_profile: Any,
    evaluator_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Build one immutable authority record for the orchestration run."""

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "record_type": "FullDatasetN1ExperimentManifest",
        "schema_version": "full-dataset-n1-experiment-manifest-v1",
        "case_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        "evaluator_config": {"path": str(evaluator_config_path), "sha256": digest(evaluator_config_path)},
        "continuation_handlers": {"path": str(continuation_path), "sha256": digest(continuation_path)},
        "preflight_report": {"path": str(preflight_path), "sha256": digest(preflight_path)},
        "runtime_environment_preflight": {
            "path": str(runtime_preflight_path),
            "sha256": digest(runtime_preflight_path),
        },
        "tested_model": {
            "agent_id": tested_profile.agent_id,
            "provider": tested_profile.provider,
            "model": tested_profile.model_id,
            "profile_sha256": tested_profile.profile_sha256,
            "runtime_profile_sha256": tested_profile.runtime_profile.profile_sha256,
        },
        "evaluator": {
            "provider": evaluator_config.get("provider"),
            "model": evaluator_config.get("model"),
            "model_configuration": dict(evaluator_config.get("model_configuration") or {}),
        },
        "provider_policy": {
            key: evaluator_config.get(key)
            for key in (
                "timeout_seconds", "transport_attempts", "retry_backoff_seconds",
                "rate_limit_backoff_seconds", "rate_limit_max_backoff_seconds",
                "max_continuation_hops", "evaluator_circuit_breaker_threshold",
                "evaluator_circuit_breaker_cooldown_seconds", "inter_case_delay_seconds",
                "global_evaluation_deadline_seconds", "evaluator_infrastructure_attempt_cap",
            )
            if key in evaluator_config
        },
        "metric_implementation": {
            "evaluator_py_sha256": digest(ROOT / "flowintentbench/evaluator.py"),
            "evaluation_metrics_py_sha256": digest(
                ROOT / "flowintentbench/evaluation_metrics.py"
            ),
        },
        "orchestrator": {
            "path": "scripts/run_full_dataset_n1_experiment.py",
            "sha256": digest(ROOT / "scripts/run_full_dataset_n1_experiment.py"),
        },
        "output_root": str(output_root),
        "historical_replay_args_used": False,
    }


def _run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(subprocess.list2cmdline([str(item)]) for item in command)
    try:
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            env=None if env is None else dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        log_path.write_text(
            f"$ {rendered}\n\n{completed.stdout or ''}\nexit_code={completed.returncode}\n",
            encoding="utf-8",
        )
        return int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        log_path.write_text(
            f"$ {rendered}\n\n{output}\nTIMEOUT seconds={timeout}\n",
            encoding="utf-8",
        )
        return 124


def _cfg(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def _load_authority(
    *,
    manifest_path: Path,
    evaluator_config_path: Path,
    continuation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _json_object(manifest_path, label="case manifest")
    evaluator = _json_object(evaluator_config_path, label="evaluator config")
    continuation = _json_object(
        continuation_path, label="production continuation handlers"
    )
    case_inventory(manifest, require_dataset=True)
    if not continuation:
        raise ValueError("production continuation handler manifest is empty")
    model_configuration = evaluator.get("model_configuration", {})
    if not isinstance(model_configuration, dict):
        raise ValueError("evaluator config model_configuration must be an object")
    structured_mode = str(
        model_configuration.get("structured_output_mode", "")
    ).strip().casefold()
    structured_capability = str(
        model_configuration.get("structured_output_capability", "")
    ).strip().upper()
    allowed_structured_pairs = {
        ("strict_json_schema", "STRICT_JSON_SCHEMA"),
        ("json_schema", "JSON_SCHEMA"),
        ("json_object", "JSON_OBJECT_ONLY"),
    }
    if (structured_mode, structured_capability) not in allowed_structured_pairs:
        raise ValueError(
            "evaluator config must explicitly declare a matching "
            "structured_output_mode/structured_output_capability pair"
        )
    timeout = float(_cfg(evaluator, "timeout_seconds", 180))
    attempts = int(_cfg(evaluator, "transport_attempts", 4))
    case_timeout = float(_cfg(evaluator, "case_process_timeout_seconds", 1800))
    max_hops = int(_cfg(evaluator, "max_continuation_hops", 8))
    continuation_attempts = int(_cfg(evaluator, "continuation_attempts", 4))
    evaluation_deadline = float(
        _cfg(evaluator, "global_evaluation_deadline_seconds", 86400)
    )
    if timeout <= 0 or attempts < 1 or case_timeout <= 0 or max_hops < 1 or continuation_attempts < 1:
        raise ValueError("evaluator timeout/attempt settings must be positive")
    if evaluation_deadline <= 0:
        raise ValueError("global_evaluation_deadline_seconds must be positive")
    transient_backoff = float(_cfg(evaluator, "retry_backoff_seconds", 2))
    rate_backoff = float(_cfg(evaluator, "rate_limit_backoff_seconds", 15))
    rate_backoff_max = float(_cfg(evaluator, "rate_limit_max_backoff_seconds", 60))
    if transient_backoff < 0 or rate_backoff < 0 or rate_backoff_max < rate_backoff:
        raise ValueError("evaluator retry backoff settings must be non-negative and ordered")
    transient_sleep_budget = sum(
        transient_backoff * (2**index) for index in range(max(0, attempts - 1))
    )
    rate_sleep_budget = sum(
        min(rate_backoff * (2**index), rate_backoff_max)
        for index in range(max(0, attempts - 1))
    )
    if case_timeout < timeout * attempts + max(
        transient_sleep_budget, rate_sleep_budget
    ):
        raise ValueError(
            "case_process_timeout_seconds must cover one full evaluator request/retry/backoff budget"
        )
    for owner, implementation in continuation.items():
        if not isinstance(owner, str) or not isinstance(implementation, str):
            raise ValueError("continuation handler manifest must map strings to strings")
    return manifest, evaluator, continuation


def _runtime_environment_preflight(
    *,
    tested_profile: Any,
    evaluator_config: Mapping[str, Any],
    continuation_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate all live runtime identities without issuing a provider call.

    The canonical experiment has three independent runtime roles: the tested
    model, the evaluator, and the continuation model used by model-backed
    owners.  Resolving their local provider configuration and credentials here
    catches missing endpoints/keys before a long replay starts.  Secrets are
    never returned or persisted.
    """

    server_config = evaluator_config.get("server_config")
    model_server_config = evaluator_config.get("model_server_config", server_config)
    evaluator_server_config = evaluator_config.get(
        "evaluator_server_config", server_config
    )
    continuation_server_config = evaluator_config.get(
        "continuation_server_config", server_config
    )

    tested_runtime = resolve_provider_configuration(
        role="model",
        config_path=model_server_config,
        provider=tested_profile.provider,
        model_id=tested_profile.model_id,
        model_configuration=tested_profile.model_configuration,
    )
    resolve_api_key(tested_runtime)

    evaluator_runtime = resolve_provider_configuration(
        role="evaluator",
        config_path=evaluator_server_config,
        provider=evaluator_config.get("provider"),
        model_id=evaluator_config.get("model"),
        model_configuration=evaluator_config.get("model_configuration"),
    )
    resolve_api_key(evaluator_runtime)

    continuation_agent = evaluator_config.get("continuation_agent")
    if not isinstance(continuation_agent, str) or not continuation_agent.strip():
        raise RuntimeError("evaluator config must declare continuation_agent")
    continuation_profile = load_agent_profile(
        continuation_agent, repository_root=ROOT
    )
    continuation_runtime = resolve_provider_configuration(
        role="model",
        config_path=continuation_server_config,
        provider=continuation_profile.provider,
        model_id=continuation_profile.model_id,
        model_configuration=continuation_profile.model_configuration,
    )
    resolve_api_key(continuation_runtime)

    # Handler identity factories consume these two deployment variables.  Set
    # them only for the duration of the no-call capability audit.
    previous_server = os.environ.get("FLOWINTENTBENCH_SERVER_CONFIG")
    previous_agent = os.environ.get("FLOWINTENTBENCH_CONTINUATION_AGENT")
    try:
        if continuation_server_config:
            os.environ["FLOWINTENTBENCH_SERVER_CONFIG"] = str(
                _resolve(str(continuation_server_config), base=ROOT)
            )
        os.environ["FLOWINTENTBENCH_CONTINUATION_AGENT"] = continuation_agent
        registry = load_pending_continuation_registry(continuation_spec)
        issues = list(registry.production_capability_issues)
        if issues:
            raise RuntimeError(
                "production continuation handlers are not ready: "
                + json.dumps(issues, ensure_ascii=False, sort_keys=True)
            )
        execution_manifest = registry.execution_manifest
    finally:
        if previous_server is None:
            os.environ.pop("FLOWINTENTBENCH_SERVER_CONFIG", None)
        else:
            os.environ["FLOWINTENTBENCH_SERVER_CONFIG"] = previous_server
        if previous_agent is None:
            os.environ.pop("FLOWINTENTBENCH_CONTINUATION_AGENT", None)
        else:
            os.environ["FLOWINTENTBENCH_CONTINUATION_AGENT"] = previous_agent

    if evaluator_runtime.model_id == tested_runtime.model_id:
        raise RuntimeError("evaluator model must be independent from tested model")
    if continuation_runtime.model_id == tested_runtime.model_id:
        raise RuntimeError("continuation model must be independent from tested model")
    return {
        "status": "PASS",
        "provider_calls": 0,
        "tested_model": tested_runtime.without_secrets(),
        "evaluator": evaluator_runtime.without_secrets(),
        "continuation": continuation_runtime.without_secrets(),
        "continuation_owner_count": execution_manifest["owner_count"],
        "continuation_required_owner_count": execution_manifest[
            "required_owner_count"
        ],
        "continuation_execution_manifest_sha256": execution_manifest[
            "manifest_sha256"
        ],
        "credentials_resolved": True,
        "model_evaluator_independence": True,
        "model_continuation_independence": True,
    }


def _collection_command(
    *,
    manifest_path: Path,
    collection_root: Path,
    agent: str,
    config: Mapping[str, Any],
    resume: bool,
    recollect_case_ids: Sequence[str] | None = None,
    resume_from: Path | None = None,
    preflight_report: Path | None = None,
    skip_preflight_tests: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_full_dataset_n1_pilot.py"),
        "--manifest",
        str(manifest_path),
        "--datasets-root",
        str(ROOT / "datasets"),
        "--output-root",
        str(collection_root),
        "--agent",
        agent,
        "--infrastructure-attempt-cap",
        str(int(_cfg(config, "collection_infrastructure_attempt_cap", 4))),
        "--case-timeout-seconds",
        str(float(_cfg(config, "collection_case_timeout_seconds", 1800))),
        "--provider-retries",
        str(int(_cfg(config, "collection_provider_retries", 5))),
        "--provider-retry-backoff-seconds",
        str(float(_cfg(config, "collection_provider_retry_backoff_seconds", 1))),
        "--provider-rate-limit-backoff-seconds",
        str(float(_cfg(config, "collection_rate_limit_backoff_seconds", 15))),
        "--provider-rate-limit-max-backoff-seconds",
        str(float(_cfg(config, "collection_rate_limit_max_backoff_seconds", 60))),
        "--provider-circuit-breaker-threshold",
        str(int(_cfg(config, "collection_circuit_breaker_threshold", 3))),
        "--provider-circuit-breaker-cooldown-seconds",
        str(float(_cfg(config, "collection_circuit_breaker_cooldown_seconds", 120))),
        "--continue-on-infrastructure-exhaustion",
    ]
    server_config = config.get("model_server_config", config.get("server_config"))
    if server_config:
        command.extend(["--server-config", str(_resolve(str(server_config), base=ROOT))])
    if config.get("model_provider"):
        command.extend(["--provider", str(config["model_provider"])])
    if config.get("model_id"):
        command.extend(["--model", str(config["model_id"])])
    if resume:
        command.append("--resume")
    selected = sorted({str(value) for value in (recollect_case_ids or ()) if str(value)})
    if selected:
        command.extend(["--recollect-case-ids", ",".join(selected)])
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    if preflight_report is not None:
        command.extend(["--preflight-report", str(preflight_report)])
    if skip_preflight_tests:
        command.append("--skip-preflight-tests")
    if bool(_cfg(config, "retry_network_noncompletion", True)):
        command.append("--retry-network-noncompletion")
    return command


def _case_and_run_assignments(
    manifest: Mapping[str, Any],
    collection_root: Path,
    *,
    tested_model: str,
    manifest_sha256: str | None = None,
    allow_collected_manifest_mismatch: bool = False,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, str]]:
    state_path = collection_root / "collection_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"collection state is missing: {state_path}")
    state = _json_object(state_path, label="collection state")
    observations = state.get("observations")
    if not isinstance(observations, list):
        raise ValueError("collection state observations must be a list")
    if state.get("total_cases") != len(case_inventory(manifest)):
        raise ValueError("collection state does not match the experiment case membership")
    # The collection state records the manifest used when the model response
    # was collected.  A changed manifest is not silently trusted: the caller
    # must run ``audit_model_response_reuse`` against the persisted snapshot
    # before replay.  We allow the mismatch only for that explicit
    # cross-manifest audit path; the old digest remains available below for
    # binding and diagnostics.
    if (
        manifest_sha256
        and state.get("case_manifest_sha256") != manifest_sha256
        and not allow_collected_manifest_mismatch
    ):
        raise ValueError("collection state case manifest digest does not match selected manifest")
    if state.get("model_id") not in {None, tested_model}:
        raise ValueError("collection state model does not match the selected agent profile")
    by_case: dict[str, Mapping[str, Any]] = {}
    for raw in observations:
        if not isinstance(raw, Mapping) or not raw.get("case_id"):
            raise ValueError("collection state contains an invalid observation")
        case_id = str(raw["case_id"])
        if case_id in by_case:
            raise ValueError(f"collection state contains duplicate case: {case_id}")
        by_case[case_id] = raw
    case_paths: dict[str, Path] = {}
    run_paths: dict[str, Path] = {}
    dataset_ids: dict[str, str] = {}
    for item in manifest["cases"]:
        case_id = str(item["case_id"])
        observation = by_case.get(case_id)
        if observation is None:
            continue
        relative = observation.get("run_record_path")
        if not isinstance(relative, str):
            continue
        run_path = (collection_root / relative).resolve()
        case_path = _resolve(str(item["case_input_path"]).rsplit("/", 1)[0])
        if run_path.is_file() and case_path.is_dir():
            case_paths[case_id] = case_path
            run_paths[case_id] = run_path
            dataset_ids[case_id] = str(item["dataset_id"])
    return case_paths, run_paths, dataset_ids


def _resolve_collected_manifest(
    collection_root: Path,
    *,
    selected_manifest: Path,
    selected_manifest_sha256: str,
) -> tuple[Path, str]:
    """Resolve the immutable manifest snapshot written by collection.

    Current collections use ``case_manifest.json`` while older pilots used
    ``collected_case_manifest.json``.  Both names are equivalent authority
    snapshots.  If an old collection predates either snapshot, it may only be
    reconstructed when ``collection_state.json`` proves byte identity with
    the selected manifest; otherwise response reuse is indeterminate and the
    caller must fail closed rather than guess.
    """

    state_path = collection_root / "collection_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"collection state is missing: {state_path}")
    state = _json_object(state_path, label="collection state")
    recorded_digest = state.get("case_manifest_sha256")
    candidates: list[tuple[Path, str]] = []
    for name in ("case_manifest.json", "collected_case_manifest.json"):
        candidate = collection_root / name
        if candidate.is_file():
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            candidates.append((candidate, digest))
            # A migrated collection can temporarily contain both names.  The
            # digest recorded by collection_state is authoritative, so do not
            # let directory/name ordering select the wrong snapshot.
            if recorded_digest and digest == recorded_digest:
                return candidate, digest
    if candidates:
        # Some early pilots wrote ``collection_state.json`` before the final
        # pilot manifest was sealed, leaving its digest stale.  For the legacy
        # filename only, accept the snapshot when the independently persisted
        # pilot manifest binds the same bytes.  A current ``case_manifest``
        # snapshot never takes this compatibility path.
        legacy = next(
            ((path, digest) for path, digest in candidates if path.name == "collected_case_manifest.json"),
            None,
        )
        pilot_manifest = collection_root / "pilot_manifest.json"
        if legacy is not None and pilot_manifest.is_file():
            try:
                pilot_value = _json_object(pilot_manifest, label="legacy pilot manifest")
            except (OSError, ValueError, json.JSONDecodeError):
                pilot_value = {}
            if pilot_value.get("case_manifest_sha256") == legacy[1]:
                return legacy
        raise RuntimeError(
            "collection manifest snapshot digest does not match collection_state.json: "
            + ", ".join(str(path) for path, _ in candidates)
        )
    if recorded_digest != selected_manifest_sha256:
        raise RuntimeError(
            "collection has no manifest snapshot and its recorded digest differs "
            "from the selected manifest; response reuse is indeterminate"
        )
    snapshot = collection_root / "case_manifest.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_name(f".{snapshot.name}.tmp")
    temporary.write_bytes(selected_manifest.read_bytes())
    os.replace(temporary, snapshot)
    return snapshot, selected_manifest_sha256


def _failure_sidecar(evaluation_root: Path, case_id: str, value: Mapping[str, Any]) -> dict[str, Any] | None:
    path = value.get("path")
    if not isinstance(path, str):
        return None
    sidecar = Path(path).parent / "evaluator_backend_failure.json"
    if not sidecar.is_file():
        return None
    try:
        loaded = _json_object(sidecar, label="evaluator backend failure")
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return loaded


def _provider_failure_category(
    evaluation_root: Path, case_id: str, record: Mapping[str, Any] | None
) -> str | None:
    """Return the persisted provider category for one selected case.

    The sidecar is the evaluator's typed transport authority.  Reading it
    immediately after a child process exits closes the small window in which
    a quota response could otherwise allow the next case to start.
    """

    if not isinstance(record, Mapping):
        return None
    sidecar = _failure_sidecar(evaluation_root, case_id, record)
    if not isinstance(sidecar, Mapping):
        return None
    failure = sidecar.get("provider_failure")
    if not isinstance(failure, Mapping):
        return None
    category = failure.get("category")
    return str(category).upper() if category is not None else None


def _collection_accounting(collection_state: Mapping[str, Any]) -> dict[str, int]:
    """Return stable collection counters, including legacy-state fallbacks.

    Early Terra pilots persisted the observation list and model counters but
    did not persist the later ``*_observation_count`` fields.  The
    observation list is the durable slot authority in that case; a missing
    or JSON-null counter must therefore be derived rather than interpreted as
    zero.  Explicit non-negative integer counters remain authoritative for
    current collections.
    """

    observations = collection_state.get("observations")
    rows = [item for item in observations if isinstance(item, Mapping)] if isinstance(observations, list) else []
    derived = Counter(str(item.get("status") or "") for item in rows)

    def counter(name: str, fallback: int) -> int:
        value = collection_state.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
        return int(fallback)

    return {
        "accounted_observation_count": counter("accounted_observation_count", len(rows)),
        "completed_count": counter("completed_count", derived.get("COMPLETED", 0)),
        "model_noncompletion_count": counter(
            "model_noncompletion_count", derived.get("MODEL_NONCOMPLETION", 0)
        ),
        "infrastructure_invalid_observation_count": counter(
            "infrastructure_invalid_observation_count",
            derived.get("INFRASTRUCTURE_INVALID", 0),
        ),
    }


def _retryable_evaluation_cases(
    evaluation_root: Path,
    case_ids: Sequence[str],
    *,
    max_infrastructure_attempts: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return retryable infrastructure cases and permanent quota cases.

    Scientific/evaluability PENDING records are terminal for the current
    replay once their continuation genuinely returned no resolution.  A
    pending record left only because the per-shard continuation-hop budget
    was exhausted after a successful typed handler result is reopened in the
    next bounded round.  Plumbing and unclassified PENDING records are also
    reopened; none of these orchestration states may be silently counted as
    scientific uncertainty.
    """

    inventory = _record_inventory(evaluation_root)
    retryable: list[str] = []
    quota: list[str] = []
    for case_id in case_ids:
        current = inventory.get(case_id)
        if current is None:
            retryable.append(case_id)
            continue
        current_status = current.get("status")
        if current_status == "PENDING":
            record_path = current.get("path")
            continuation_budget_exhausted = False
            if isinstance(record_path, str):
                record_file = Path(record_path)
                if not record_file.is_absolute():
                    record_file = evaluation_root / record_file
                state_path = record_file.parent / "pending_continuation_state.json"
                if state_path.is_file():
                    try:
                        state_value = _json_object(
                            state_path, label="pending continuation state"
                        )
                        events = state_value.get("events")
                        last_status = (
                            str((events[-1] or {}).get("dispatch_status") or "")
                            if isinstance(events, list) and events
                            else ""
                        )
                        continuation_budget_exhausted = bool(
                            state_value.get("bounded") is True
                            and state_value.get("cycle_detected") is not True
                            and last_status == "HANDLER_RETURNED"
                        )
                    except (OSError, ValueError, json.JSONDecodeError):
                        continuation_budget_exhausted = False
            if current.get("pending_category") in {
                "PLUMBING_PENDING",
                "UNCLASSIFIED_PENDING",
            } or continuation_budget_exhausted:
                retryable.append(case_id)
            continue
        if current_status != "INFRASTRUCTURE_INVALID":
            continue
        sidecar = _failure_sidecar(evaluation_root, case_id, current)
        if sidecar:
            provider = sidecar.get("provider_failure")
            category = provider.get("category") if isinstance(provider, Mapping) else None
            if category == "QUOTA":
                quota.append(case_id)
                continue
            if sidecar.get("retryable") is False:
                continue
        if max_infrastructure_attempts is not None:
            record_path = current.get("path")
            if isinstance(record_path, str):
                record_file = Path(record_path)
                if not record_file.is_absolute():
                    record_file = evaluation_root / record_file
                run_dir = record_file.parent
                historical = list((run_dir / "evaluation_attempts").glob("attempt-*"))
                # The selected record counts as one attempt.  Archived
                # attempts are immutable and therefore provide a durable
                # per-case retry budget across process restarts.
                attempt_count = len(historical) + 1
                if attempt_count >= max_infrastructure_attempts:
                    continue
        retryable.append(case_id)
    return retryable, quota


def _assignment_args(values: Mapping[str, Path]) -> list[str]:
    """Render one complete ``CASE_ID=PATH`` argument per mapping row."""

    return [f"{key}={path}" for key, path in values.items()]


def _evaluation_command(
    *,
    manifest_path: Path,
    collected_manifest_path: Path,
    manifest: Mapping[str, Any],
    case_paths: Mapping[str, Path],
    run_paths: Mapping[str, Path],
    dataset_ids: Mapping[str, str],
    evaluation_root: Path,
    summary_path: Path,
    evaluator_config: Mapping[str, Any],
    continuation_path: Path,
    collected_manifest_sha256: str,
    selected_case: str,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_saved_runs.py"),
        "--mode",
        "PILOT",
        "--datasets-root",
        str(ROOT / "datasets"),
        "--evaluation-root",
        str(evaluation_root),
        "--summary",
        str(summary_path),
        "--case",
        _assignment_args(case_paths)[0],
    ]
    command.extend(sum((["--case", value] for value in _assignment_args(case_paths)[1:]), []))
    command.append("--run")
    command.append(_assignment_args(run_paths)[0])
    command.extend(sum((["--run", value] for value in _assignment_args(run_paths)[1:]), []))
    for case_id, dataset_id in dataset_ids.items():
        command.extend(["--dataset-id", f"{case_id}={dataset_id}"])
    command.extend(
        [
            "--only-case",
            selected_case,
            "--checkpoint-summary",
            "--retry-evaluator-failures",
            "--pending-handlers-json",
            str(continuation_path),
            "--require-reusable-responses",
            "--collected-case-manifest",
            str(collected_manifest_path),
            "--expected-collected-case-manifest-sha256",
            str(collected_manifest_sha256),
            "--candidate-case-manifest",
            str(manifest_path),
            "--max-continuation-hops",
            str(int(_cfg(evaluator_config, "max_continuation_hops", 8))),
            "--provider",
            str(_cfg(evaluator_config, "provider", "yiapi")),
            "--model",
            str(_cfg(evaluator_config, "model", "gpt-5.6-sol")),
            "--model-config-json",
            json.dumps(
                dict(_cfg(evaluator_config, "model_configuration", {})),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "--evaluator-timeout-seconds",
            str(float(_cfg(evaluator_config, "timeout_seconds", 180))),
            "--evaluator-transport-attempts",
            str(int(_cfg(evaluator_config, "transport_attempts", 4))),
            "--evaluator-retry-backoff-seconds",
            str(float(_cfg(evaluator_config, "retry_backoff_seconds", 2))),
            "--evaluator-rate-limit-backoff-seconds",
            str(float(_cfg(evaluator_config, "rate_limit_backoff_seconds", 15))),
            "--evaluator-rate-limit-max-backoff-seconds",
            str(float(_cfg(evaluator_config, "rate_limit_max_backoff_seconds", 60))),
            "--evaluator-retry-jitter-seconds",
            str(float(_cfg(evaluator_config, "retry_jitter_seconds", 0))),
        ]
    )
    server_config = evaluator_config.get("evaluator_server_config", evaluator_config.get("server_config"))
    if server_config:
        command.extend(["--server-config", str(_resolve(str(server_config), base=ROOT))])
    portfolio = ROOT / "artifacts/reference/scientific_portfolio/concept_family_inventory.json"
    if portfolio.is_file():
        command.extend(["--portfolio-manifest", str(portfolio)])
    return command


def _record_inventory(evaluation_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in iter_current_evaluation_records(evaluation_root):
        try:
            value = _json_object(path, label="evaluation record")
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        result = value.get("result")
        pending_category = None
        if isinstance(result, Mapping):
            case_id = str(result.get("case_id") or "")
            status = str(result.get("run_status") or "UNKNOWN")
            manifest_digest_value = value.get("evaluation_manifest_digest")
            scientifically_finalized = bool(
                result.get("eligible_for_scientific_aggregation", False)
            )
        else:
            case_id = str(value.get("case_id") or "")
            status = "PENDING"
            scientifically_finalized = False
            manifest_digest_value = value.get("evaluation_manifest_digest")
            pending_type = str(value.get("pending_type") or "")
            # Provider failures are persisted as INFRASTRUCTURE_INVALID.  A
            # remaining PENDING record is therefore scientific/evaluability
            # uncertainty unless its owner/request was not executable.
            if pending_type in {
                "semantic_uncertain",
                "unit_relationship",
                "gt_outside_finding",
                "novel_finding",
                "novel_operationalization_scientific_adjudication",
            }:
                pending_category = "SCIENTIFIC_PENDING"
            elif pending_type == "novel_operationalization_materialization":
                pending_category = "EVALUABILITY_PENDING"
            elif pending_type in {"novel_operationalization", "novel_operationalization_candidate"}:
                pending_category = "SCIENTIFIC_PENDING"
            else:
                pending_category = "UNCLASSIFIED_PENDING"
            state_path = path.parent / "pending_continuation_state.json"
            if state_path.is_file():
                try:
                    state_value = _json_object(state_path, label="pending continuation state")
                    events = state_value.get("events")
                    if isinstance(events, list) and events:
                        last_status = str((events[-1] or {}).get("dispatch_status") or "")
                        if last_status in {
                            "PENDING_OWNER_NOT_EXECUTED",
                            "PENDING_REQUEST_IDENTITY_INVALID",
                            "CONTINUATION_CYCLE_DETECTED",
                        }:
                            pending_category = "PLUMBING_PENDING"
                        elif (
                            state_value.get("bounded") is True
                            and state_value.get("cycle_detected") is not True
                            and last_status == "HANDLER_RETURNED"
                        ):
                            # The handler produced a typed resolution, but the
                            # shard stopped before the evaluator could exhaust
                            # the next request.  This is resumable orchestration
                            # work, not scientific uncertainty.
                            pending_category = "PLUMBING_PENDING"
                except (OSError, ValueError, json.JSONDecodeError):
                    pending_category = "UNCLASSIFIED_PENDING"
        if case_id:
            records[case_id] = {
                "status": status,
                "scientifically_finalized": scientifically_finalized,
                "path": str(path),
                "pending_type": value.get("pending_type"),
                "pending_category": pending_category,
                "evaluation_manifest_digest": manifest_digest_value,
                "continuation_request_id": (
                    value.get("continuation_context") or {}
                ).get("continuation_request_id")
                if isinstance(value.get("continuation_context"), Mapping)
                else None,
            }
    return records


def _record_state(evaluation_root: Path) -> dict[str, str]:
    return {
        case_id: str(value["status"])
        for case_id, value in _record_inventory(evaluation_root).items()
    }


def _write_summary(evaluation_root: Path, summary_path: Path, expected_cases: int) -> dict[str, Any]:
    inventory = _record_inventory(evaluation_root)
    states = {
        case_id: str(value["status"]) for case_id, value in inventory.items()
    }
    status_counts = Counter(states.values())
    pending_count = status_counts.get("PENDING", 0)
    terminal_count = len(states)
    infra = status_counts.get("INFRASTRUCTURE_INVALID", 0)
    scientific_finalized = sum(
        value.get("status") == "COMPLETED"
        and bool(value.get("scientifically_finalized"))
        for value in inventory.values()
    )
    pending_category_counts = Counter(
        str(value.get("pending_category") or "UNCLASSIFIED_PENDING")
        for value in inventory.values()
        if value.get("status") == "PENDING"
    )
    manifest_variants = {
        str(value.get("evaluation_manifest_digest"))
        for value in inventory.values()
        if value.get("evaluation_manifest_digest")
    }
    manifest = None
    manifest_digest = None
    lock = evaluation_root / "evaluation_manifest.lock.json"
    if lock.is_file():
        lock_value = _json_object(lock, label="evaluation manifest lock")
        manifest = lock_value.get("evaluation_manifest")
        manifest_digest = lock_value.get("evaluation_manifest_digest")
    existing: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            candidate = _json_object(summary_path, label="evaluation summary")
            existing = dict(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
    payload: dict[str, Any] = {
        **existing,
        "status": "PILOT / NOT FORMAL BENCHMARK RESULT",
        "mode": "PILOT",
        "case_count": expected_cases,
        "persisted_trial_count": len(states),
        "terminal_record_count": terminal_count - pending_count,
        "finalized_trial_count": terminal_count - pending_count,
        "scientifically_finalized_trial_count": scientific_finalized,
        "model_noncompletion_count": status_counts.get("MODEL_NONCOMPLETION", 0),
        "infrastructure_invalid_count": infra,
        "pending_count": pending_count,
        "evaluation_pending_count": pending_count,
        "scientific_pending_count": pending_category_counts.get("SCIENTIFIC_PENDING", 0),
        "evaluability_pending_count": pending_category_counts.get("EVALUABILITY_PENDING", 0),
        "plumbing_pending_count": pending_category_counts.get("PLUMBING_PENDING", 0),
        "unclassified_pending_count": pending_category_counts.get("UNCLASSIFIED_PENDING", 0),
        "pending_category_counts": dict(sorted(pending_category_counts.items())),
        "evaluator_manifest_variants": len(manifest_variants),
        "evaluator_manifest_digests": sorted(manifest_variants),
        "evaluation_manifest": existing.get("evaluation_manifest", manifest),
        "evaluation_manifest_digest": existing.get(
            "evaluation_manifest_digest", manifest_digest
        ),
        "summary": {
            **(
                existing.get("summary")
                if isinstance(existing.get("summary"), Mapping)
                else {}
            ),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "case_aggregates": existing.get("case_aggregates", []),
        "pending_records": existing.get("pending_records", []),
        "orchestrator_record_state": states,
    }
    _atomic_json(summary_path, payload)
    return payload


def _final_status(
    *,
    collection_state: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    missing_cases: int,
    report_ok: bool,
    metrics_ok: bool,
    evaluation_deadline_exhausted: bool = False,
    provider_quota_blocked: bool = False,
) -> dict[str, Any]:
    collection_counts = _collection_accounting(collection_state)
    accounted = collection_counts["accounted_observation_count"]
    total = int(collection_state.get("total_cases", 28))
    collection_infra = collection_counts["infrastructure_invalid_observation_count"]
    collection_status = str(collection_state.get("status") or "")
    evaluator_infra = int(evaluation_summary.get("infrastructure_invalid_count", 0))
    pending = int(evaluation_summary.get("scientific_pending_count", 0)) + int(
        evaluation_summary.get("evaluability_pending_count", 0)
    )
    plumbing_pending = int(evaluation_summary.get("plumbing_pending_count", 0))
    unclassified_pending = int(
        evaluation_summary.get("unclassified_pending_count", 0)
    )
    manifest_variants = int(evaluation_summary.get("evaluator_manifest_variants", 0))
    evaluation_accounted = int(evaluation_summary.get("persisted_trial_count", 0))
    evaluation_model_noncompletion = int(
        evaluation_summary.get("model_noncompletion_count", 0)
    )
    scientific_accounting_total = (
        int(evaluation_summary.get("scientifically_finalized_trial_count", 0))
        + pending
        + evaluation_model_noncompletion
    )
    quality_ok = (
        accounted == total
        and collection_infra == 0
        and collection_status not in {
            "COLLECTION_BLOCKED_EXTERNAL_QUOTA",
            "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_TIMEOUT",
            "COLLECTION_BLOCKED_EXTERNAL_PROVIDER_UNAVAILABLE",
        }
        and evaluation_accounted == total
        and scientific_accounting_total == total
        and missing_cases == 0
        and evaluator_infra == 0
        and plumbing_pending == 0
        and unclassified_pending == 0
        and manifest_variants == 1
        and report_ok
        and metrics_ok
        and not evaluation_deadline_exhausted
        and not provider_quota_blocked
    )
    return {
        "record_type": "full_dataset_n1_experiment_status",
        "schema_version": "full-dataset-n1-experiment-v1",
        "collection_status": collection_state.get("status"),
        "model_completed": collection_counts["completed_count"],
        "model_noncompletion": collection_counts["model_noncompletion_count"],
        "collection_infrastructure_invalid": collection_counts[
            "infrastructure_invalid_observation_count"
        ],
        "collection_infrastructure_unresolved": collection_infra,
        "collection_accounted": accounted,
        "collection_requested": total,
        "collection_total": total,
        "evaluation_accounted": evaluation_accounted,
        "evaluation_requested": total,
        "missing_case_count": missing_cases,
        "scientifically_finalized": evaluation_summary.get(
            "scientifically_finalized_trial_count", 0
        ),
        "evaluation_model_noncompletion": evaluation_model_noncompletion,
        "scientific_accounting_total": scientific_accounting_total,
        "scientific_pending": int(evaluation_summary.get("scientific_pending_count", 0)),
        "evaluability_pending": int(evaluation_summary.get("evaluability_pending_count", 0)),
        "scientific_pending_count": int(evaluation_summary.get("scientific_pending_count", 0)),
        "evaluability_pending_count": int(evaluation_summary.get("evaluability_pending_count", 0)),
        "evaluation_pending_count": int(evaluation_summary.get("pending_count", 0)),
        "pending_category_counts": evaluation_summary.get("pending_category_counts", {}),
        "plumbing_pending": plumbing_pending,
        "unclassified_pending": unclassified_pending,
        "evaluator_manifest_variants": manifest_variants,
        "evaluator_infrastructure_invalid": evaluator_infra,
        "evaluator_quota_blocked": bool(provider_quota_blocked),
        "evaluation_deadline_exhausted": bool(evaluation_deadline_exhausted),
        "quality_report_status": (
            "COMPLETE_WITH_SCIENTIFIC_NULLS"
            if quality_ok
            else "BLOCKED_PROVIDER_QUOTA"
            if provider_quota_blocked
            else "INCOMPLETE_INFRASTRUCTURE"
        ),
        "efficiency_report_status": "COMPLETE" if report_ok else "INCOMPLETE",
        "metrics_generated": bool(metrics_ok),
        "report_generated": bool(report_ok),
        "final_experiment_status": (
            "COMPLETE_WITH_SCIENTIFIC_NULLS"
            if quality_ok
            else "BLOCKED_PROVIDER_QUOTA"
            if provider_quota_blocked
            else "INCOMPLETE_INFRASTRUCTURE"
        ),
        "legitimate_scientific_pending_allowed": True,
    }


def _format_terminal_status(status: Mapping[str, Any]) -> str:
    """Human-readable final summary for the canonical one-command runner."""

    lines = [
        "FLOWINTENTBENCH N1 EXPERIMENT",
        f"  collection: {status.get('collection_accounted', 0)}/{status.get('collection_requested', 28)}",
        f"  model_completed: {status.get('model_completed', 0)}",
        f"  model_noncompletion: {status.get('model_noncompletion', 0)}",
        f"  collection_infrastructure_unresolved: {status.get('collection_infrastructure_unresolved', 0)}",
        f"  evaluation: {status.get('evaluation_accounted', 0)}/{status.get('evaluation_requested', 28)}",
        f"  scientifically_finalized: {status.get('scientifically_finalized', 0)}",
        f"  scientific_pending: {status.get('scientific_pending', 0)}",
        f"  evaluation_infrastructure_unresolved: {status.get('evaluator_infrastructure_invalid', 0)}",
        f"  plumbing_pending: {status.get('plumbing_pending', 0)}",
        f"  evaluator_quota_blocked: {status.get('evaluator_quota_blocked', False)}",
        f"  response_reuse_status: {status.get('response_reuse_status', 'NOT_RUN')}",
        f"  response_reuse_rerun_required_cases: {len(status.get('response_reuse_rerun_required_cases', []))}",
        f"  metrics_generated: {status.get('metrics_generated', False)}",
        f"  report_generated: {status.get('report_generated', False)}",
        f"FINAL STATUS: {status.get('final_experiment_status', 'UNKNOWN')}",
    ]
    return "\n".join(lines)


def _evaluation_environment(evaluator_config: Mapping[str, Any]) -> dict[str, str]:
    """Use identical continuation identity and retry settings for every entrypoint."""
    evaluation_env = dict(os.environ)
    continuation_server = evaluator_config.get("continuation_server_config", evaluator_config.get("server_config"))
    if continuation_server:
        evaluation_env["FLOWINTENTBENCH_SERVER_CONFIG"] = str(
            _resolve(str(continuation_server), base=ROOT)
        )
    if evaluator_config.get("continuation_agent"):
        evaluation_env["FLOWINTENTBENCH_CONTINUATION_AGENT"] = str(
            evaluator_config["continuation_agent"]
        )
    continuation_env = {
        "FLOWINTENTBENCH_CONTINUATION_TIMEOUT": "continuation_timeout_seconds",
        "FLOWINTENTBENCH_CONTINUATION_ATTEMPTS": "continuation_attempts",
        "FLOWINTENTBENCH_CONTINUATION_RETRY_BACKOFF": "continuation_retry_backoff_seconds",
        "FLOWINTENTBENCH_CONTINUATION_RATE_LIMIT_BACKOFF": "continuation_rate_limit_backoff_seconds",
        "FLOWINTENTBENCH_CONTINUATION_RATE_LIMIT_MAX_BACKOFF": "continuation_rate_limit_max_backoff_seconds",
        "FLOWINTENTBENCH_CONTINUATION_RETRY_JITTER": "continuation_retry_jitter_seconds",
    }
    for env_name, config_key in continuation_env.items():
        if config_key in evaluator_config:
            evaluation_env[env_name] = str(evaluator_config[config_key])
    return evaluation_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True)
    parser.add_argument(
        "--evaluator-config",
        type=Path,
        default=ROOT / "config/evaluator_sol.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments/userstudy/case_manifest.json",
    )
    parser.add_argument(
        "--continuation-handlers",
        type=Path,
        default=ROOT / "config/production_continuation_handlers.json",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--skip-preflight-tests", action="store_true")
    parser.add_argument("--preflight-report", type=Path,
                        help="reuse current manifest/code-bound passing suite evidence")
    parser.add_argument("--max-evaluation-rounds", type=int, default=3)
    args = parser.parse_args()
    if args.max_evaluation_rounds < 1:
        parser.error("--max-evaluation-rounds must be positive")

    output_root = _resolve(args.output_root)
    manifest_path = _resolve(args.manifest)
    evaluator_config_path = _resolve(args.evaluator_config)
    continuation_path = _resolve(args.continuation_handlers)
    manifest, evaluator_config, continuation_spec = _load_authority(
        manifest_path=manifest_path,
        evaluator_config_path=evaluator_config_path,
        continuation_path=continuation_path,
    )
    tested_profile = load_agent_profile(args.agent, repository_root=ROOT)
    tested_model = str(tested_profile.model_id)
    evaluator_model = str(_cfg(evaluator_config, "model", "gpt-5.6-sol"))
    if evaluator_model == tested_model:
        raise RuntimeError(
            "evaluator model must be independent from the tested model: "
            f"{tested_model}"
        )
    continuation_model = evaluator_config.get("continuation_agent")
    if continuation_model:
        try:
            continuation_profile = load_agent_profile(
                str(continuation_model), repository_root=ROOT
            )
        except Exception as exc:
            raise RuntimeError(
                f"cannot resolve continuation agent identity {continuation_model!r}: {exc}"
            ) from exc
        if str(continuation_profile.model_id) == tested_model:
            raise RuntimeError(
                "continuation model must be independent from the tested model: "
                f"{tested_model}"
            )
    collection_root = output_root / "collection"
    evaluation_root = output_root / "evaluation"
    report_root = output_root / "report"
    summary_path = evaluation_root / "summary.json"
    preflight_path = output_root / "preflight_report.json"
    runtime_preflight_path = output_root / "runtime_environment_preflight.json"
    status_path = output_root / "experiment_status.json"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    # Determine a legacy/current manifest mismatch before validating the
    # persisted preflight report.  In a cross-manifest resume the preflight
    # report is correctly bound to the old model-visible snapshot and must be
    # checked against that snapshot, not against the candidate manifest.
    cross_manifest_resume = False
    collected_snapshot_for_preflight: Path | None = None
    collection_state_probe_path = output_root / "collection" / "collection_state.json"
    if (args.resume or args.evaluation_only) and collection_state_probe_path.is_file():
        # Resolve current and legacy authority in one place.  In particular,
        # early Terra collections can have a stale state digest while their
        # independently sealed pilot manifest correctly binds
        # ``collected_case_manifest.json``.  Do not reject that collection
        # before the explicit response-reuse audit can compare model-visible
        # inputs.
        (
            collected_snapshot_for_preflight,
            collected_snapshot_digest,
        ) = _resolve_collected_manifest(
            collection_root,
            selected_manifest=manifest_path,
            selected_manifest_sha256=manifest_sha256,
        )
        cross_manifest_resume = collected_snapshot_digest != manifest_sha256

    if args.skip_preflight_tests and not args.preflight_report and not (
        (args.resume or args.evaluation_only) and preflight_path.is_file()
    ):
        raise RuntimeError(
            "No valid frozen preflight report; run without --skip-preflight-tests."
        )
    if args.preflight_report:
        preflight = _json_object(_resolve(args.preflight_report), label="preflight report")
        _assert_preflight_binding(preflight, manifest_file=manifest_path, repository_root=ROOT)
        validate_manifest(manifest_path, repository_root=ROOT, output_root=collection_root,
                          run_tests=False, allow_existing_collection=args.resume or args.evaluation_only)
        _atomic_json(preflight_path, preflight)
    elif (
        (args.resume or args.evaluation_only)
        and preflight_path.is_file()
        and args.skip_preflight_tests
    ):
        # An explicitly skipped resume must consume already-frozen test
        # evidence.  It may not silently refresh a stale report.
        preflight = _json_object(preflight_path, label="preflight report")
        validate_manifest(
            manifest_path,
            repository_root=ROOT,
            output_root=collection_root,
            run_tests=False,
            allow_existing_collection=True,
        )
    else:
        # A normal resume is also the supported way to refresh preflight
        # evidence after an orchestration-only code correction.  This runs
        # the local suite, performs no provider/model call, and rebinds the
        # experiment manifest to the current code tree.
        preflight = validate_manifest(
            manifest_path,
            repository_root=ROOT,
            output_root=collection_root,
            run_tests=True,
            allow_existing_collection=args.resume or args.evaluation_only,
        )
        _atomic_json(preflight_path, preflight)
    # A resume must validate the persisted preflight bytes themselves, not
    # merely rerun a structural manifest check.  This prevents a stale report
    # from being used to bind a new experiment identity or a new collection
    # subprocess.
    preflight_manifest_for_binding = manifest_path
    if cross_manifest_resume and collected_snapshot_for_preflight is not None:
        report_digest = preflight.get("case_manifest_sha256")
        old_digest = hashlib.sha256(
            collected_snapshot_for_preflight.read_bytes()
        ).hexdigest()
        candidate_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if report_digest == old_digest:
            preflight_manifest_for_binding = collected_snapshot_for_preflight
        elif report_digest != candidate_digest:
            raise RuntimeError(
                "preflight report is bound to neither the selected nor collected manifest"
            )
    _assert_preflight_binding(
        preflight,
        manifest_file=preflight_manifest_for_binding,
        repository_root=ROOT,
    )
    runtime_preflight = _runtime_environment_preflight(
        tested_profile=tested_profile,
        evaluator_config=evaluator_config,
        continuation_spec=continuation_spec,
    )
    _atomic_json(runtime_preflight_path, runtime_preflight)
    experiment_manifest_path = output_root / "experiment_manifest.json"
    experiment_manifest = _build_experiment_manifest(
        manifest_path=manifest_path,
        evaluator_config_path=evaluator_config_path,
        continuation_path=continuation_path,
        preflight_path=preflight_path,
        runtime_preflight_path=runtime_preflight_path,
        tested_profile=tested_profile,
        evaluator_config=evaluator_config,
        output_root=output_root,
    )
    if experiment_manifest_path.is_file():
        existing_manifest = _json_object(
            experiment_manifest_path, label="experiment manifest"
        )
        if existing_manifest != experiment_manifest:
            # A same-authority resume may follow a tested orchestration-only
            # corrective patch (for example accounting or bounded-resume
            # logic).  Preserve the former manifest immutably, then bind the
            # resumed report to the current code/preflight.  Scientific
            # authorities and the tested/evaluator identities must remain
            # byte-for-byte equal; otherwise the caller needs a new root (or
            # the explicit cross-manifest response-reuse path).
            authority_keys = (
                "case_manifest",
                "continuation_handlers",
                "evaluator",
                "evaluator_config",
                "tested_model",
                "metric_implementation",
            )
            same_resume_authority = bool(args.resume or args.evaluation_only) and all(
                existing_manifest.get(key) == experiment_manifest.get(key)
                for key in authority_keys
            )
            if not cross_manifest_resume and not same_resume_authority:
                raise RuntimeError(
                    "experiment manifest differs from the existing output root; "
                    "use --resume with the same authority files or choose a new root"
                )
            # Preserve the old identity under a content-addressed path before
            # publishing the candidate identity.  The collection snapshot and
            # reuse audit remain the evidence for the old model-visible input.
            existing_digest = hashlib.sha256(
                json.dumps(
                    existing_manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            history_path = (
                output_root
                / "experiment_manifest_history"
                / f"manifest-{existing_digest[:16]}.json"
            )
            if not history_path.is_file():
                _atomic_json(history_path, existing_manifest)
            _atomic_json(experiment_manifest_path, experiment_manifest)
    else:
        _atomic_json(experiment_manifest_path, experiment_manifest)

    if not args.evaluation_only and not cross_manifest_resume:
        if collection_root.exists() and not args.resume:
            raise RuntimeError(
                f"collection output already exists; use --resume or choose a new --output-root: {collection_root}"
            )
        collection_cmd = _collection_command(
            manifest_path=manifest_path,
            collection_root=collection_root,
            agent=args.agent,
            config=evaluator_config,
            resume=args.resume,
        )
        collection_cmd.extend(
            [
                "--preflight-report",
                str(preflight_path),
                "--skip-preflight-tests",
            ]
        )
        collection_env = dict(os.environ)
        model_server = evaluator_config.get("model_server_config", evaluator_config.get("server_config"))
        if model_server:
            collection_env["FLOWINTENTBENCH_SERVER_CONFIG"] = str(
                _resolve(str(model_server), base=ROOT)
            )
        code = _run_command(
            collection_cmd,
            log_path=output_root / "logs/collection.log",
            timeout=float(_cfg(evaluator_config, "collection_process_timeout_seconds", 0)) or None,
            env=collection_env,
        )
        if code != 0 and not (collection_root / "collection_state.json").is_file():
            raise RuntimeError(f"model collection failed; see {output_root / 'logs/collection.log'}")

    # Resolve the persisted model-visible collection snapshot before checking
    # assignments.  This supports both current and legacy collection roots;
    # a cross-manifest mismatch is evaluated by the explicit reuse audit below.
    collected_manifest_path, collected_manifest_sha256 = _resolve_collected_manifest(
        collection_root,
        selected_manifest=manifest_path,
        selected_manifest_sha256=manifest_sha256,
    )
    collection_state = _json_object(
        collection_root / "collection_state.json", label="collection state"
    )
    collection_quota_blocked = str(collection_state.get("status") or "") == (
        "COLLECTION_BLOCKED_EXTERNAL_QUOTA"
    )
    case_paths, run_paths, dataset_ids = _case_and_run_assignments(
        manifest,
        collection_root,
        tested_model=tested_model,
        manifest_sha256=manifest_sha256,
        allow_collected_manifest_mismatch=(
            collected_manifest_sha256 != manifest_sha256
        ),
    )
    missing_collection = len(case_inventory(manifest)) - len(run_paths)
    if missing_collection:
        # Do not pass a partial case/run mapping to the evaluator CLI: its
        # explicit assignment contract requires equal case and run sets.  A
        # collection may still be reported as incomplete after this guard.
        evaluation_summary = _write_summary(
            output_root / "evaluation", output_root / "evaluation/summary.json", len(manifest["cases"])
        )
        status = _final_status(
            collection_state=collection_state,
            evaluation_summary=evaluation_summary,
            missing_cases=missing_collection,
            report_ok=False,
            metrics_ok=False,
            provider_quota_blocked=collection_quota_blocked,
        )
        status["authority"] = {
            "case_manifest": str(manifest_path),
            "tested_model": tested_model,
            "evaluator_model": evaluator_model,
            "evaluator_config": str(evaluator_config_path),
            "continuation_handlers": str(continuation_path),
            "historical_replay_args_used": False,
            "experiment_manifest": str(experiment_manifest_path),
        }
        _atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    # A collection-level quota is permanent for this run.  Do not spend
    # evaluator requests on a set of model slots that cannot be trusted to
    # contain complete responses; resume remains available after credentials
    # or provider capacity are repaired.
    if collection_quota_blocked:
        evaluation_summary = _write_summary(evaluation_root, summary_path, len(manifest["cases"]))
        status = _final_status(
            collection_state=collection_state,
            evaluation_summary=evaluation_summary,
            missing_cases=0,
            report_ok=False,
            metrics_ok=False,
            provider_quota_blocked=True,
        )
        status["authority"] = {
            "case_manifest": str(manifest_path),
            "tested_model": tested_model,
            "evaluator_model": evaluator_model,
            "evaluator_config": str(evaluator_config_path),
            "continuation_handlers": str(continuation_path),
            "historical_replay_args_used": False,
            "experiment_manifest": str(experiment_manifest_path),
        }
        status["collection_quota_blocked"] = True
        _atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        print(_format_terminal_status(status))
        return 2

    # Automatic model-response reuse gate.  The canonical collection and the
    # selected manifest are compared before any evaluator call; hidden GT or
    # evaluator changes cannot force a fresh model response.  For an older
    # collection with a different manifest, the audit is authoritative and
    # receives the *old* digest rather than the candidate digest.
    reuse_audit = audit_model_response_reuse(
        run_records=list(run_paths.values()),
        collected_case_manifest=collected_manifest_path,
        candidate_case_manifest=manifest_path,
        expected_collected_manifest_sha256=collected_manifest_sha256,
    )
    reuse_path = output_root / "model_response_reuse_audit.json"
    _atomic_json(reuse_path, reuse_audit)
    if reuse_audit.get("status") != "REUSABLE":
        collection_state = _json_object(
            collection_root / "collection_state.json", label="collection state"
        )
        evaluation_summary = _write_summary(
            evaluation_root, summary_path, expected_cases=len(manifest["cases"])
        )
        status = _final_status(
            collection_state=collection_state,
            evaluation_summary=evaluation_summary,
            missing_cases=0,
            report_ok=False,
            metrics_ok=False,
        )
        status["final_experiment_status"] = "BLOCKED_RESPONSE_REUSE"
        status["quality_report_status"] = "BLOCKED_RESPONSE_REUSE"
        status["response_reuse_audit"] = str(reuse_path)
        status["response_reuse_status"] = reuse_audit.get("status")
        status["response_reuse_rerun_required_cases"] = [
            str(row.get("case_id"))
            for row in reuse_audit.get("cases", [])
            if isinstance(row, Mapping) and row.get("status") == "RERUN_REQUIRED"
        ]
        status["response_reuse_indeterminate_cases"] = [
            str(row.get("case_id"))
            for row in reuse_audit.get("cases", [])
            if isinstance(row, Mapping) and row.get("status") == "INDETERMINATE"
        ]
        recollection_cases = status["response_reuse_rerun_required_cases"]
        recollection_plan_path = output_root / "response_recollection_plan.json"
        _atomic_json(
            recollection_plan_path,
            {
                "record_type": "N1ResponseRecollectionPlan",
                "schema_version": "n1-response-recollection-plan-v1",
                "status": "RECOLLECT_AFFECTED_CASES_ONLY"
                if recollection_cases
                else "BLOCKED_INDETERMINATE",
                "case_ids": recollection_cases,
                "indeterminate_case_ids": status["response_reuse_indeterminate_cases"],
                "source_audit": str(reuse_path),
                "tested_model": tested_model,
                "candidate_case_manifest": str(manifest_path),
                "historical_replay_args_used": False,
            },
        )
        status["response_recollection_plan"] = str(recollection_plan_path)
        status["authority"] = {
            "case_manifest": str(manifest_path),
            "tested_model": tested_model,
            "evaluator_model": str(_cfg(evaluator_config, "model", "gpt-5.6-sol")),
            "evaluator_config": str(evaluator_config_path),
            "continuation_handlers": str(continuation_path),
            "historical_replay_args_used": False,
        }
        _atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    evaluation_env = _evaluation_environment(evaluator_config)
    # A single wall-clock budget covers all evaluator retries and continuation
    # calls.  Child processes receive only the remaining budget, preventing a
    # nested retry schedule from outliving the canonical orchestration run.
    evaluation_deadline_seconds = float(
        _cfg(evaluator_config, "global_evaluation_deadline_seconds", 86400)
    )
    evaluation_deadline = time.monotonic() + evaluation_deadline_seconds
    evaluation_deadline_exhausted = False
    evaluator_failure_streak = 0
    evaluator_circuit_breaker_activations = 0
    evaluator_cooldown_seconds = 0.0
    provider_quota_blocked = False
    evaluator_infrastructure_attempt_cap = int(
        _cfg(
            evaluator_config,
            "evaluator_infrastructure_attempt_cap",
            args.max_evaluation_rounds,
        )
    )
    if evaluator_infrastructure_attempt_cap < 1:
        raise ValueError("evaluator_infrastructure_attempt_cap must be positive")
    breaker_threshold = int(
        _cfg(evaluator_config, "evaluator_circuit_breaker_threshold", 3)
    )
    breaker_cooldown = float(
        _cfg(evaluator_config, "evaluator_circuit_breaker_cooldown_seconds", 120)
    )
    for round_number in range(1, args.max_evaluation_rounds + 1):
        unresolved, quota_cases = _retryable_evaluation_cases(
            evaluation_root,
            list(case_paths),
            max_infrastructure_attempts=evaluator_infrastructure_attempt_cap,
        )
        # A permanent quota/billing rejection must not be retried or hidden as
        # a scientific pending.  Leave the explicit status and stop opening
        # new provider calls for this experiment.
        if quota_cases:
            provider_quota_blocked = True
            break
        if not unresolved:
            break
        for case_id in unresolved:
            remaining = evaluation_deadline - time.monotonic()
            if remaining <= 0:
                evaluation_deadline_exhausted = True
                break
            command = _evaluation_command(
                manifest_path=manifest_path,
                collected_manifest_path=collected_manifest_path,
                manifest=manifest,
                case_paths=case_paths,
                run_paths=run_paths,
                dataset_ids=dataset_ids,
                evaluation_root=evaluation_root,
                summary_path=summary_path,
                evaluator_config=evaluator_config,
                continuation_path=continuation_path,
                collected_manifest_sha256=collected_manifest_sha256,
                selected_case=case_id,
            )
            case_environment = dict(evaluation_env)
            # All evaluator and continuation requests in this child shard
            # share one case-scoped wall-clock budget.  The child backends
            # clip each individual request/retry to this deadline.
            case_environment["FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC"] = str(
                time.monotonic()
                + min(
                    float(_cfg(evaluator_config, "case_process_timeout_seconds", 1800)),
                    remaining,
                )
            )
            code = _run_command(
                command,
                log_path=output_root / "logs" / f"evaluation-round-{round_number}-{case_id}.log",
                timeout=min(
                    float(_cfg(evaluator_config, "case_process_timeout_seconds", 1800)),
                    remaining,
                ),
                env=case_environment,
            )
            # Quota is a permanent provider/account condition.  Detect it
            # from the typed failure sidecar before opening another case, even
            # when the child returned a non-zero exit code or was interrupted.
            current_inventory = _record_inventory(evaluation_root).get(case_id)
            if (
                _provider_failure_category(evaluation_root, case_id, current_inventory)
                == "QUOTA"
            ):
                provider_quota_blocked = True
                break
            if code not in {0, 124} and current_inventory is None:
                raise RuntimeError(
                    "evaluator shard failed without persisting a typed current "
                    f"record for {case_id}; see "
                    f"{output_root / 'logs' / f'evaluation-round-{round_number}-{case_id}.log'}"
                )
            if code == 124:
                evaluator_failure_streak += 1
                if (
                    breaker_threshold > 0
                    and evaluator_failure_streak >= breaker_threshold
                    and breaker_cooldown > 0
                ):
                    pause = min(
                        breaker_cooldown,
                        max(0.0, evaluation_deadline - time.monotonic()),
                    )
                    evaluator_circuit_breaker_activations += 1
                    evaluator_cooldown_seconds += pause
                    time.sleep(pause)
                    evaluator_failure_streak = 0
                continue
            current_status = _record_state(evaluation_root).get(case_id)
            if current_status == "INFRASTRUCTURE_INVALID":
                evaluator_failure_streak += 1
            else:
                evaluator_failure_streak = 0
            if (
                breaker_threshold > 0
                and evaluator_failure_streak >= breaker_threshold
                and breaker_cooldown > 0
            ):
                pause = min(
                    breaker_cooldown,
                    max(0.0, evaluation_deadline - time.monotonic()),
                )
                evaluator_circuit_breaker_activations += 1
                evaluator_cooldown_seconds += pause
                time.sleep(pause)
                evaluator_failure_streak = 0
            pacing = float(_cfg(evaluator_config, "inter_case_delay_seconds", 5))
            if pacing > 0:
                time.sleep(min(pacing, max(0.0, evaluation_deadline - time.monotonic())))
        if provider_quota_blocked:
            break
        if evaluation_deadline_exhausted:
            break

    evaluation_summary = _write_summary(
        evaluation_root, summary_path, expected_cases=len(manifest["cases"])
    )
    missing_evaluation = max(
        0, len(manifest["cases"]) - int(evaluation_summary.get("persisted_trial_count", 0))
    )
    report_ok = False
    metrics_ok = False
    render_cmd = [
        sys.executable,
        str(ROOT / "scripts/render_evaluation_report.py"),
        "--evaluation-root",
        str(evaluation_root),
        "--summary",
        str(summary_path),
        "--output-dir",
        str(report_root),
        "--model",
        tested_model,
        "--collection-state",
        str(collection_root / "collection_state.json"),
    ]
    report_ok = _run_command(
        render_cmd, log_path=output_root / "logs/render-report.log"
    ) == 0
    metrics_cmd = [
        sys.executable,
        str(ROOT / "scripts/collect_n1_metrics_json.py"),
        "--evaluation-root",
        str(evaluation_root),
        "--summary",
        str(summary_path),
        "--report-dir",
        str(report_root),
        "--output",
        str(output_root / "metrics.json"),
        "--model",
        tested_model,
        "--collection-state",
        str(collection_root / "collection_state.json"),
        "--case-manifest",
        str(manifest_path),
    ]
    if report_ok:
        metrics_ok = _run_command(
            metrics_cmd, log_path=output_root / "logs/collect-metrics.log"
        ) == 0

    collection_state = _json_object(
        collection_root / "collection_state.json", label="collection state"
    )
    _write_reliability_summary(
        output_path=output_root / "reliability.json",
        collection_state=collection_state,
        collection_root=collection_root,
        evaluation_root=evaluation_root,
        report_root=report_root,
        evaluator_circuit_breaker_activations=evaluator_circuit_breaker_activations,
        evaluator_cooldown_seconds=evaluator_cooldown_seconds,
    )
    status = _final_status(
        collection_state=collection_state,
        evaluation_summary=evaluation_summary,
        missing_cases=max(missing_collection, missing_evaluation),
        report_ok=report_ok,
        metrics_ok=metrics_ok,
        evaluation_deadline_exhausted=evaluation_deadline_exhausted,
        provider_quota_blocked=provider_quota_blocked,
    )
    status["response_reuse_status"] = reuse_audit.get("status")
    status["response_reuse_reusable_case_count"] = int(
        reuse_audit.get("reusable_case_count", 0)
    )
    status["response_reuse_rerun_required_cases"] = [
        str(row.get("case_id"))
        for row in reuse_audit.get("cases", [])
        if isinstance(row, Mapping) and row.get("status") == "RERUN_REQUIRED"
    ]
    status["response_reuse_indeterminate_cases"] = [
        str(row.get("case_id"))
        for row in reuse_audit.get("cases", [])
        if isinstance(row, Mapping) and row.get("status") == "INDETERMINATE"
    ]
    status["authority"] = {
        "case_manifest": str(manifest_path),
        "tested_model": tested_model,
        "evaluator_model": evaluator_model,
        "evaluator_config": str(evaluator_config_path),
        "continuation_handlers": str(continuation_path),
        "historical_replay_args_used": False,
        "experiment_manifest": str(experiment_manifest_path),
    }
    status["outputs"] = {
        "metrics": str(output_root / "metrics.json"),
        "report": str(report_root / "report.md"),
        "reliability": str(output_root / "reliability.json"),
        "experiment_manifest": str(experiment_manifest_path),
        "experiment_status": str(status_path),
    }
    _atomic_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    print(_format_terminal_status(status))
    return 0 if status["final_experiment_status"] == "COMPLETE_WITH_SCIENTIFIC_NULLS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
