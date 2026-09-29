#!/usr/bin/env python3
"""Selective completion acquisition for an existing canonical N=1 run.

This script is intentionally downstream of ``run_full_dataset_n1_experiment``.
It copies the immutable collection/evaluation snapshot, locks cases whose
applicable metric cells are complete, and only recollects or re-evaluates the
remaining case IDs.  It does not alter Case/GT/scientific metric definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import re
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.experiment_scope import case_inventory
from flowintentbench.completion_acquisition import (  # noqa: E402
    build_completion_plan,
    completion_case_ids,
)
from scripts import run_full_dataset_n1_experiment as canonical  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_matrix(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_sha256(path: Path) -> str | None:
    """Return a deterministic digest for a file or directory.

    Completion manifests bind content, not merely paths.  Directory digests
    are formed from sorted relative filenames and their bytes so that a
    resumed run cannot silently mix two different collection snapshots.
    """

    if not path.exists():
        return None
    if path.is_file():
        return _sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _model_family(model_id: str) -> str:
    value = str(model_id).strip().casefold()
    if "luna" in value:
        return "luna"
    if "terra" in value:
        return "terra"
    return ""


def _default_evaluator_config(model_id: str) -> Path:
    """Select the evaluator profile for the tested model family.

    The profiles intentionally remain ordinary JSON configuration files.  The
    family-based default only prevents the completion command from silently
    falling back to the generic ``evaluator_sol.json`` profile; an explicit
    ``--evaluator-config`` remains supported for controlled replay.
    """

    family = _model_family(model_id)
    if family in {"luna", "terra"}:
        return ROOT / "config" / f"evaluator_for_{family}.json"
    return ROOT / "config" / "evaluator_sol.json"


def _continuation_identity(continuation_path: Path) -> dict[str, Any]:
    value = _read_json(continuation_path)
    agents: dict[str, Any] = {}
    for owner, agent in sorted(value.items()):
        agents[str(owner)] = str(agent)
    return {"path": str(continuation_path), "sha256": _path_sha256(continuation_path), "owners": agents}


def _validate_runtime_identity(
    *, model: str, evaluator_config: Mapping[str, Any], continuation_path: Path
) -> dict[str, Any]:
    """Fail closed when evaluator/continuation identity aliases tested model."""

    evaluator_model = str(evaluator_config.get("model") or "").strip()
    if not evaluator_model:
        raise ValueError("evaluator config must declare model")
    if evaluator_model.casefold() == model.casefold():
        raise ValueError("evaluator model must be independent from tested model")
    continuation = _continuation_identity(continuation_path)
    configured_agent = str(evaluator_config.get("continuation_agent") or "").strip()
    if not configured_agent:
        raise ValueError("evaluator config must declare continuation_agent")
    # Resolve the configured profile where possible.  This is a local check
    # and never issues a provider call.
    try:
        continuation_profile = canonical.load_agent_profile(configured_agent, repository_root=ROOT)
        continuation_model = str(continuation_profile.model_id)
    except Exception as exc:
        raise ValueError(f"cannot resolve continuation agent {configured_agent!r}: {exc}") from exc
    if continuation_model.casefold() == model.casefold():
        raise ValueError("continuation model must be independent from tested model")
    return {
        "tested_model": model,
        "evaluator_model": evaluator_model,
        "continuation_agent": configured_agent,
        "continuation_model": continuation_model,
        "continuation": continuation,
    }


def _completion_manifest(
    *,
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    evaluator_config_path: Path,
    continuation_path: Path,
    evaluator_config: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    """Build the immutable completion authority manifest."""

    def code_digest(name: str) -> dict[str, Any]:
        path = ROOT / name
        return {"path": name, "sha256": _path_sha256(path)}

    source_collection = source_root / "collection"
    source_state = source_collection / "collection_state.json"
    return {
        "record_type": "N1CompletionManifest",
        "schema_version": "n1-completion-manifest-v1",
        "tested_model": model,
        "source_collection": {
            "path": str(source_collection),
            "collection_state_sha256": _path_sha256(source_state),
            "snapshot_sha256": _path_sha256(source_collection),
        },
        "case_manifest": {"path": str(manifest_path), "sha256": _path_sha256(manifest_path)},
        "evaluator_config": {"path": str(evaluator_config_path), "sha256": _path_sha256(evaluator_config_path)},
        "evaluator_identity": {
            "provider": evaluator_config.get("provider"),
            "model": evaluator_config.get("model"),
            "model_configuration": dict(evaluator_config.get("model_configuration") or {}),
        },
        "runtime_roles": {
            "tested_model": model,
            "evaluator_model": runtime_identity.get("evaluator_model"),
            "continuation_agent": runtime_identity.get("continuation_agent"),
            "continuation_model": runtime_identity.get("continuation_model"),
        },
        "continuation": dict(runtime_identity.get("continuation") or {}),
        "implementation": {
            "completion_script": code_digest("scripts/run_full_dataset_n1_completion.py"),
            "completion_planner": code_digest("flowintentbench/completion_acquisition.py"),
            "evaluator": code_digest("flowintentbench/evaluator.py"),
            "metrics": code_digest("flowintentbench/evaluation_metrics.py"),
        },
        "completion_policy": {
            "complete_cell_states": ["NOT_APPLICABLE", "NUMERIC"],
            "recollect_states": ["MODEL_NONCOMPLETION"],
            "reevaluate_states": ["PENDING", "INFRASTRUCTURE_INVALID", "UNEXPLAINED_NULL"],
            "locked_cases_are_immutable": True,
        },
        "historical_replay_args_used": False,
        "output_root": str(output_root),
    }


def _load_plan_numbers(output_root: Path) -> list[int]:
    numbers: list[int] = []
    pattern = re.compile(r"^completion_plan_round_(\d+)\.json$")
    for path in output_root.glob("completion_plan_round_*.json"):
        match = pattern.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(set(numbers))


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write once; an existing artifact must be byte/content identical."""

    if path.is_file():
        existing = _read_json(path)
        if existing != dict(value):
            raise RuntimeError(f"immutable completion artifact conflict: {path}")
        return
    _write_json(path, value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy_initial_snapshot(source_root: Path, output_root: Path, *, copy_collection: bool = False) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"completion output already contains files: {output_root}; use --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    names = ["evaluation", "report"]
    if copy_collection:
        names.insert(0, "collection")
    for name in names:
        source = source_root / name
        if source.is_dir():
            shutil.copytree(source, output_root / name, dirs_exist_ok=True)
    for name in ("runtime_environment_preflight.json", "experiment_manifest.json"):
        source = source_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)


def _progress_fingerprints(
    *,
    metric_rows: list[dict[str, Any]],
    evaluation_root: Path,
    collection_root: Path,
    expected_case_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Create a deterministic pending-progress fingerprint per case.

    The fingerprint binds the actual persisted response/evaluation record,
    evaluator manifest and pending cell states.  It is an orchestration guard;
    it does not infer or alter scientific judgments.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row.get("case_id") or ""), []).append(row)
    inventory = canonical._record_inventory(evaluation_root)
    state_path = collection_root / "collection_state.json"
    collection_state = _read_json(state_path) if state_path.is_file() else {}
    observations = {
        str(item.get("case_id")): item
        for item in collection_state.get("observations", [])
        if isinstance(item, Mapping) and item.get("case_id")
    }
    result: dict[str, dict[str, Any]] = {}
    for case_id in expected_case_ids:
        rows = grouped.get(case_id, [])
        current = inventory.get(case_id, {})
        record_path = current.get("path")
        record_digest = None
        if isinstance(record_path, str):
            path = Path(record_path)
            if not path.is_absolute():
                path = evaluation_root / path
            if path.is_file():
                record_digest = _sha256(path)
        run_record_digest = None
        relative = observations.get(case_id, {}).get("run_record_path")
        if isinstance(relative, str):
            run_path = collection_root / relative
            if run_path.is_file():
                run_record_digest = _sha256(run_path)
        payload = {
            "case_id": case_id,
            "metric_states": sorted(
                (str(row.get("metric") or ""), str(row.get("status") or ""))
                for row in rows
            ),
            "evaluation_manifest_digest": current.get("evaluation_manifest_digest"),
            "pending_type": current.get("pending_type"),
            "pending_category": current.get("pending_category"),
            "record_sha256": record_digest,
            "run_record_sha256": run_record_digest,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        result[case_id] = {"fingerprint": hashlib.sha256(encoded).hexdigest(), "payload": payload}
    return result


def _previous_progress(output_root: Path, round_number: int) -> dict[str, dict[str, Any]]:
    """Load the last post-execution fingerprints for progress detection."""

    if round_number <= 1:
        return {}
    path = output_root / f"completion_plan_round_{round_number - 1:02d}_result.json"
    if not path.is_file():
        return {}
    try:
        value = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    raw = value.get("pending_fingerprints")
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(case_id): dict(item)
        for case_id, item in raw.items()
        if isinstance(item, Mapping) and item.get("fingerprint")
    }


def _render_and_collect(
    *,
    manifest_path: Path,
    collection_root: Path,
    evaluation_root: Path,
    report_root: Path,
    summary_path: Path,
    model: str,
    output_root: Path,
) -> None:
    canonical._write_summary(evaluation_root, summary_path, expected_cases=len(case_inventory(canonical._json_object(manifest_path, label="case manifest"))))
    render = [
        sys.executable,
        str(ROOT / "scripts/render_evaluation_report.py"),
        "--evaluation-root", str(evaluation_root),
        "--summary", str(summary_path),
        "--output-dir", str(report_root),
        "--model", model,
        "--collection-state", str(collection_root / "collection_state.json"),
    ]
    if canonical._run_command(render, log_path=output_root / "logs/render-report.log") != 0:
        raise RuntimeError("evaluation report rendering failed")
    collect = [
        sys.executable,
        str(ROOT / "scripts/collect_n1_metrics_json.py"),
        "--evaluation-root", str(evaluation_root),
        "--summary", str(summary_path),
        "--report-dir", str(report_root),
        "--output", str(output_root / "metrics.json"),
        "--model", model,
        "--collection-state", str(collection_root / "collection_state.json"),
        "--case-manifest", str(manifest_path),
    ]
    if canonical._run_command(collect, log_path=output_root / "logs/collect-metrics.log") != 0:
        raise RuntimeError("metrics JSON collection failed")


def run_completion(
    *,
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    agent: str,
    evaluator_config_path: Path | None,
    continuation_path: Path,
    max_rounds: int,
    resume: bool,
    evaluation_only: bool,
    plan_only: bool = False,
    strict_metrics: bool = False,
) -> dict[str, Any]:
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    if not resume:
        _copy_initial_snapshot(source_root, output_root, copy_collection=evaluation_only)
    collection_root = output_root / "collection"
    evaluation_root = output_root / "evaluation"
    report_root = output_root / "report"
    summary_path = evaluation_root / "summary.json"
    manifest = canonical._json_object(manifest_path, label="case manifest")
    expected_ids = list(case_inventory(manifest))
    profile = canonical.load_agent_profile(agent, repository_root=ROOT)
    model = str(profile.model_id)
    if evaluator_config_path is None:
        evaluator_config_path = _default_evaluator_config(model)
    evaluator_config_path = evaluator_config_path.resolve()
    if not evaluator_config_path.is_file():
        raise FileNotFoundError(f"evaluator config does not exist: {evaluator_config_path}")
    config = canonical._json_object(evaluator_config_path, label="evaluator config")
    runtime_identity = _validate_runtime_identity(
        model=model, evaluator_config=config, continuation_path=continuation_path
    )
    evaluation_env = canonical._evaluation_environment(config)
    deadline = time.monotonic() + float(config.get("global_evaluation_deadline_seconds", 86400))
    completion_manifest = _completion_manifest(
        source_root=source_root,
        output_root=output_root,
        manifest_path=manifest_path,
        evaluator_config_path=evaluator_config_path,
        continuation_path=continuation_path,
        evaluator_config=config,
        runtime_identity=runtime_identity,
        model=model,
    )
    completion_manifest_path = output_root / "completion_manifest.json"
    if resume:
        if not completion_manifest_path.is_file():
            raise RuntimeError(
                "--resume requires completion_manifest.json; refusing to mix an unbound output root"
            )
        existing_manifest = _read_json(completion_manifest_path)
        if existing_manifest != completion_manifest:
            def scientific_identity(value):
                value = dict(value)
                implementation = dict(value.pop("implementation"))
                implementation.pop("completion_script", None)
                implementation.pop("completion_planner", None)
                return value, implementation
            if scientific_identity(existing_manifest) != scientific_identity(completion_manifest):
                raise RuntimeError("completion manifest mismatch; model, config, source or scoring implementation changed")
            history = output_root / "completion_manifest_history" / (_sha256(completion_manifest_path) + ".json")
            _write_immutable_json(history, existing_manifest)
            _write_json(completion_manifest_path, completion_manifest)
    else:
        _write_immutable_json(completion_manifest_path, completion_manifest)
    seed_needed = not (collection_root / "collection_state.json").is_file()
    if seed_needed and not evaluation_only:
        # The full repository suite is run as a separate completion gate.  A
        # seed collection only needs a current, manifest/code-bound report;
        # rerunning pytest inside the model-collection subprocess would add
        # avoidable latency and make a provider timeout more likely.
        preflight = canonical.validate_manifest(
            manifest_path,
            repository_root=ROOT,
            output_root=collection_root,
            run_tests=False,
            allow_existing_collection=True,
        )
        # Preserve the source's completed-suite evidence as descriptive
        # provenance while rebinding manifest/code hashes above.  The suite
        # has already passed in the completion preflight gate; a stale source
        # code-data version must never be copied into the new identity.
        for candidate in (
            source_root / "collection" / "preflight_report.json",
            source_root / "preflight_report.json",
        ):
            if candidate.is_file():
                source_preflight = _read_json(candidate)
                frozen_tests = source_preflight.get("frozen_tests")
                if isinstance(frozen_tests, dict) and frozen_tests.get("returncode") == 0:
                    preflight["frozen_tests"] = dict(frozen_tests)
                break
        _write_json(collection_root / "preflight_report.json", preflight)
    plan_history: list[dict[str, Any]] = []
    provider_quota_blocked = False
    existing_rounds = _load_plan_numbers(output_root) if resume else []
    latest_round = max(existing_rounds, default=0)
    # A plan is immutable.  If the latest plan was written but its matrix has
    # not changed yet (for example the process was interrupted before
    # execution), resume that exact round; otherwise start the next round.
    round_number = 1 if not resume else latest_round + 1
    if resume and latest_round:
        latest_path = output_root / f"completion_plan_round_{latest_round:02d}.json"
        latest_result_path = output_root / f"completion_plan_round_{latest_round:02d}_result.json"
        try:
            latest_matrix = report_root / "metric_matrix.csv"
            latest_digest = _sha256(latest_matrix) if latest_matrix.is_file() else None
            # A post-execution result proves that the round already ran.  Do
            # not reopen its immutable pre-run plan on resume; either finish
            # immediately when it is complete or start the next round so the
            # prior pending fingerprint can be compared.
            if latest_result_path.is_file():
                latest_result = _read_json(latest_result_path)
                if latest_result.get("status") == "FULL_METRIC_COMPLETE":
                    round_number = latest_round
                else:
                    round_number = latest_round + 1
            else:
                latest_plan = _read_json(latest_path)
                if latest_plan.get("metric_matrix_sha256") == latest_digest:
                    # The process stopped after writing the pre-run plan but
                    # before publishing a result snapshot.  Resume the same
                    # round and preserve its immutable routing decision.
                    round_number = latest_round
        except Exception:
            # A malformed historical plan is handled by the immutable write
            # below, which fails closed with a useful conflict error.
            round_number = latest_round + 1
    for _round_index in range(max_rounds):
        matrix_path = report_root / "metric_matrix.csv"
        if not matrix_path.is_file():
            raise FileNotFoundError(f"rendered metric matrix is missing: {matrix_path}")
        current_progress = _progress_fingerprints(
            metric_rows=_read_matrix(matrix_path),
            evaluation_root=evaluation_root,
            collection_root=collection_root,
            expected_case_ids=expected_ids,
        )
        prior_progress = _previous_progress(output_root, round_number)
        # If a process was interrupted after writing the immutable pre-run
        # plan but before executing any work, resume that exact plan.  This
        # preserves the original routing decision and avoids rewriting an
        # immutable artifact merely because resume was requested.
        existing_plan_path = output_root / f"completion_plan_round_{round_number:02d}.json"
        existing_plan = None
        if existing_plan_path.is_file():
            candidate = _read_json(existing_plan_path)
            if candidate.get("metric_matrix_sha256") == _sha256(matrix_path):
                existing_plan = candidate
        if existing_plan is not None:
            plan = existing_plan
        else:
            planner_progress = {
                case_id: {
                    "fingerprint": current_progress.get(case_id, {}).get("fingerprint"),
                    "previous_fingerprint": prior_progress.get(case_id, {}).get("fingerprint"),
                }
                for case_id in expected_ids
            }
            plan = build_completion_plan(
                _read_matrix(matrix_path),
                expected_case_ids=expected_ids,
                round_number=round_number,
                previous_progress=planner_progress,
                strict=strict_metrics,
            )
        plan["tested_model"] = model
        plan["source_root"] = str(source_root)
        plan["metric_matrix_sha256"] = _sha256(matrix_path)
        plan.setdefault("pending_fingerprints", current_progress)
        plan_path = output_root / f"completion_plan_round_{round_number:02d}.json"
        _write_immutable_json(plan_path, plan)
        _write_json(
            output_root / "incomplete_cases.json",
            {
                "record_type": "N1IncompleteCases",
                "schema_version": "n1-incomplete-cases-v1",
                "round": round_number,
                "tested_model": model,
                "case_count": plan["incomplete_case_count"],
                "cases": [
                    item for item in plan["cases"] if item.get("action") != "LOCK"
                ],
            },
        )
        plan_history.append(plan)
        if plan_only:
            break
        if plan["status"] == "FULL_METRIC_COMPLETE":
            break
        recollect_ids = completion_case_ids(plan, "RECOLLECT")
        # Recollection and evaluator replay are different actions.  A
        # MODEL_NONCOMPLETION slot has no valid response to evaluate; it only
        # enters the evaluator after a successful recollection.  In
        # evaluation-only mode it therefore remains unprocessed rather than
        # being accidentally sent to ``evaluate_saved_runs.py``.
        reevaluate_ids = completion_case_ids(plan, "REEVALUATE")
        if not evaluation_only:
            reevaluate_ids = sorted(set(reevaluate_ids) | set(recollect_ids))
        if not evaluation_only and (recollect_ids or (round_number == 1 and seed_needed)):
            collection_cmd = canonical._collection_command(
                manifest_path=manifest_path,
                collection_root=collection_root,
                agent=agent,
                config=config,
                resume=True,
                recollect_case_ids=recollect_ids,
                resume_from=(source_root / "collection") if round_number == 1 and seed_needed else None,
                preflight_report=(collection_root / "preflight_report.json") if not (round_number == 1 and seed_needed) else None,
                skip_preflight_tests=not (round_number == 1 and seed_needed),
            )
            code = canonical._run_command(
                collection_cmd,
                log_path=output_root / "logs" / f"collection-round-{round_number}.log",
                timeout=float(canonical._cfg(config, "collection_process_timeout_seconds", 0)) or None,
            )
            if code != 0 and not (collection_root / "collection_state.json").is_file():
                raise RuntimeError(f"completion collection failed in round {round_number}")
        if reevaluate_ids:
            collected_manifest_path, collected_digest = canonical._resolve_collected_manifest(
                collection_root,
                selected_manifest=manifest_path,
                selected_manifest_sha256=_sha256(manifest_path),
            )
            case_paths, run_paths, dataset_ids = canonical._case_and_run_assignments(
                manifest,
                collection_root,
                tested_model=model,
                manifest_sha256=_sha256(manifest_path),
                allow_collected_manifest_mismatch=collected_digest != _sha256(manifest_path),
            )
            for case_id in reevaluate_ids:
                if case_id not in run_paths:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                command = canonical._evaluation_command(
                    manifest_path=manifest_path,
                    collected_manifest_path=collected_manifest_path,
                    manifest=manifest,
                    case_paths=case_paths,
                    run_paths=run_paths,
                    dataset_ids=dataset_ids,
                    evaluation_root=evaluation_root,
                    summary_path=summary_path,
                    evaluator_config=config,
                    continuation_path=continuation_path,
                    collected_manifest_sha256=collected_digest,
                    selected_case=case_id,
                )
                case_timeout = min(float(canonical._cfg(config, "case_process_timeout_seconds", 1800)), remaining)
                case_env = dict(evaluation_env)
                case_env["FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC"] = str(time.monotonic() + case_timeout)
                code = canonical._run_command(
                    command,
                    log_path=output_root / "logs" / f"evaluation-round-{round_number}-{case_id}.log",
                    timeout=case_timeout,
                    env=case_env,
                )
                # A typed 403/quota rejection is a global provider condition
                # for this completion run.  Stop before opening another case;
                # it is never converted into a scientific pending result.
                inventory = canonical._record_inventory(evaluation_root).get(case_id)
                if (
                    canonical._provider_failure_category(evaluation_root, case_id, inventory)
                    == "QUOTA"
                ):
                    provider_quota_blocked = True
                    break
                if code not in {0, 124}:
                    raise RuntimeError(f"completion evaluator failed for {case_id}")
            if provider_quota_blocked:
                break
        _render_and_collect(
            manifest_path=manifest_path,
            collection_root=collection_root,
            evaluation_root=evaluation_root,
            report_root=report_root,
            summary_path=summary_path,
            model=model,
            output_root=output_root,
        )
        # The pre-execution plan is immutable and records the work selected in
        # this round.  Persist a separate result snapshot after rendering so
        # the final status always reflects the newly produced matrix and never
        # suffers from an off-by-one decision.
        refreshed = build_completion_plan(
            _read_matrix(report_root / "metric_matrix.csv"),
            expected_case_ids=expected_ids,
            round_number=round_number,
            strict=strict_metrics,
        )
        refreshed_progress = _progress_fingerprints(
            metric_rows=_read_matrix(report_root / "metric_matrix.csv"),
            evaluation_root=evaluation_root,
            collection_root=collection_root,
            expected_case_ids=expected_ids,
        )
        refreshed.update(
            {
                "tested_model": model,
                "source_root": str(source_root),
                "metric_matrix_sha256": _sha256(report_root / "metric_matrix.csv"),
                "source_plan": str(plan_path),
                "phase": "post_execution",
                "pending_fingerprints": refreshed_progress,
            }
        )
        result_plan_path = output_root / f"completion_plan_round_{round_number:02d}_result.json"
        _write_immutable_json(result_plan_path, refreshed)
        plan_history[-1] = refreshed
        _write_json(
            output_root / "incomplete_cases.json",
            {
                "record_type": "N1IncompleteCases",
                "schema_version": "n1-incomplete-cases-v1",
                "round": round_number,
                "tested_model": model,
                "case_count": refreshed["incomplete_case_count"],
                "cases": [
                    item for item in refreshed["cases"] if item.get("action") != "LOCK"
                ],
            },
        )
        if refreshed["status"] == "FULL_METRIC_COMPLETE":
            break
        if provider_quota_blocked:
            break
        round_number += 1
    final_plan = plan_history[-1]
    result = {
        "record_type": "N1CompletionAcquisitionResult",
        "schema_version": "n1-completion-result-v1",
        "tested_model": model,
        "round_count": len(plan_history),
        "full_metric_complete": final_plan["status"] == "FULL_METRIC_COMPLETE",
        "status": "BLOCKED_PROVIDER_QUOTA" if provider_quota_blocked else final_plan["status"],
        "provider_quota_blocked": provider_quota_blocked,
        "complete_case_count": final_plan["complete_case_count"],
        "incomplete_case_count": final_plan["incomplete_case_count"],
        "output_root": str(output_root),
        "plans": [str(output_root / f"completion_plan_round_{i:02d}.json") for i in range(1, len(plan_history) + 1)],
    }
    _write_json(output_root / "completion_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/userstudy/case_manifest.json")
    parser.add_argument(
        "--evaluator-config",
        type=Path,
        default=None,
        help=(
            "evaluator profile JSON; when omitted, selects evaluator_for_luna.json "
            "or evaluator_for_terra.json from the tested agent identity"
        ),
    )
    parser.add_argument("--continuation-handlers", type=Path, default=ROOT / "config/production_continuation_handlers.json")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--strict-metrics", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write one deterministic completion plan without model/evaluator calls",
    )
    args = parser.parse_args()
    result = run_completion(
        source_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        manifest_path=args.manifest.resolve(),
        agent=args.agent,
        evaluator_config_path=(
            args.evaluator_config.resolve()
            if args.evaluator_config is not None
            else None
        ),
        continuation_path=args.continuation_handlers.resolve(),
        max_rounds=args.max_rounds,
        resume=args.resume,
        evaluation_only=args.evaluation_only,
        plan_only=args.plan_only,
        strict_metrics=args.strict_metrics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["full_metric_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
