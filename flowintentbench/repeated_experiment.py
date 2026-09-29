"""Strict, read-only completion checks for independent repeated experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from flowintentbench.experiment_scope import case_inventory

from .completion_acquisition import build_completion_plan


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_round(root: Path, manifest: dict, model: str, *, host_network: bool = False) -> dict:
    """Require all requested scientific cells and real execution telemetry.

    A valid low score is complete. Unknown values, noncompletion, duplicate
    rows and truncated matrices cannot be promoted to complete observations.
    Provider-reported cost may be unavailable; it is never estimated as zero.
    """
    expected = {key: row["condition"] for key, row in case_inventory(manifest).items()}
    issues: list[str] = []
    matrix_path = root / "report/metric_matrix.csv"
    if not matrix_path.is_file():
        return {"status": "INCOMPLETE", "complete_case_count": 0,
                "issues": ["metric matrix missing"], "output_root": str(root)}
    with matrix_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if set(r.get("case_id") for r in rows) != set(expected):
        issues.append("metric matrix case membership mismatch")
    for row in rows:
        if row.get("model") != model or row.get("condition") != expected.get(row.get("case_id")):
            issues.append("metric matrix identity mismatch")
        if str(row.get("trial")) != "1":
            issues.append("each independent round must contain exactly one local trial")
    plan = build_completion_plan(rows, expected_case_ids=list(expected), strict=True)
    if plan["status"] != "FULL_METRIC_COMPLETE":
        issues.append("scientific metrics incomplete or invalid")
    bundle_path = root / "metrics.json"
    state_path = root / "collection/collection_state.json"
    record_hashes: list[str] = []
    telemetry = []
    for path in (bundle_path, state_path):
        if not path.is_file():
            issues.append(f"missing {path.name}")
    if bundle_path.is_file():
        bundle = read_json(bundle_path)
        trials = bundle.get("trial_metrics", [])
        if bundle.get("model") != model:
            issues.append("metrics bundle model mismatch")
        if len(trials) != len(expected) or {r.get("case_id") for r in trials} != set(expected):
            issues.append("metrics bundle must contain exactly the requested trials")
        if any(r.get("status") != "COMPLETED" for r in trials):
            issues.append("metrics bundle contains non-finalized trials")
        if bundle.get("evaluation_snapshot", {}).get("scientific_comparability") is not True:
            issues.append("evaluation bundle mixes snapshots or lacks snapshot identity")
        bundle_cells = bundle.get("metric_matrix", [])
        if len(bundle_cells) != len(rows):
            issues.append("JSON/CSV metric matrix size mismatch")
        else:
            by_key = {(r.get("case_id"), r.get("metric")): r for r in bundle_cells}
            for row in rows:
                cell = by_key.get((row.get("case_id"), row.get("metric")), {})
                if cell.get("status") != row.get("status"):
                    issues.append("JSON/CSV metric status mismatch")
                if row.get("status") == "NUMERIC":
                    value = cell.get("value")
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        issues.append("JSON metric value must have a numeric type")
                    else:
                        try:
                            if float(row["value"]) != value:
                                issues.append("JSON/CSV metric value mismatch")
                        except (TypeError, ValueError):
                            issues.append("invalid CSV value")
    if state_path.is_file():
        observations = read_json(state_path).get("observations", [])
        if len(observations) != len(expected) or {o.get("case_id") for o in observations} != set(expected):
            issues.append("collection must have exactly the requested observations")
        for observation in observations:
            value = observation.get("run_record_path")
            if not isinstance(value, str):
                issues.append("observation has no run record")
                continue
            path = (root / "collection" / value).resolve()
            if not path.is_relative_to((root / "collection").resolve()) or not path.is_file():
                issues.append("missing or out-of-root run record")
                continue
            record = read_json(path)
            record_hashes.append(file_hash(path))
            if record.get("case_id") != observation["case_id"] or record.get("model_id") != model:
                issues.append("run record case/model identity mismatch")
            # Field names are those persisted by RunRecord, not filenames
            # reconstructed from historical absolute paths.
            expected_profile = "flow-python-host-network-v1" if host_network else "flow-python-v1"
            if record.get("run_status") != "COMPLETED":
                issues.append(f"run incomplete for {observation['case_id']}")
            if record.get("network_isolation_active") is not (not host_network):
                issues.append(f"network condition mismatch for {observation['case_id']}")
            if host_network and record.get("runtime_profile_id") != expected_profile:
                issues.append(f"host-network runtime profile mismatch for {observation['case_id']}")
            for name in (record.get("final_answer_path") or "final_answer.md", "trajectory.json"):
                artifact = path.parent / name
                if not artifact.is_file() or not artifact.read_text(encoding="utf-8").strip():
                    issues.append(f"missing {name} for {observation['case_id']}")
            values = {k: record.get(k) for k in (
                "input_tokens", "output_tokens", "model_turn_count", "tool_call_count",
                "python_execution_count", "wall_clock_time", "provider_reported_cost",
            )}
            for name, value in values.items():
                if name == "provider_reported_cost" and value is None:
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    issues.append(f"invalid {name} for {observation['case_id']}")
            telemetry.append({"case_id": observation["case_id"], **values,
                              "provider_reported_cost_status": "UNAVAILABLE_FROM_PROVIDER"
                              if values["provider_reported_cost"] is None else "REPORTED"})
    return {"status": "FULL_METRIC_COMPLETE" if not issues else "INCOMPLETE",
            "complete_case_count": plan["complete_case_count"],
            "issues": sorted(set(issues)), "plan": plan, "telemetry": telemetry,
            "record_hashes": record_hashes, "output_root": str(root),
            "metric_matrix_sha256": file_hash(matrix_path),
            "metrics_sha256": file_hash(bundle_path) if bundle_path.is_file() else None}


def run_rounds(*, models: list[str], rounds: int, execute, persist, expected_case_count: int = 28) -> dict:
    """A global barrier: all models finish the requested cases before the next round.

    execute() owns script calls/resumption; persist() atomically checkpoints
    after every model. Completed observations from prior rounds cannot be
    reused as new independent observations.
    """
    if expected_case_count < 1 or rounds < 1 or not models or len(set(models)) != len(models):
        raise ValueError("positive rounds and unique model identities required")
    results = []
    hashes: dict[str, set[str]] = {m: set() for m in models}
    for number in range(1, rounds + 1):
        for model in models:
            try:
                result = dict(execute(number, model))
            except Exception as exc:
                result = {"status": "INCOMPLETE", "complete_case_count": 0,
                          "issues": [f"{type(exc).__name__}: {exc}"]}
            result.update({"round": number, "model": model})
            current = set(result.get("record_hashes", []))
            if current & hashes[model]:
                result["status"] = "INCOMPLETE"
                result.setdefault("issues", []).append("run records reused across independent rounds")
            complete = (result.get("status") == "FULL_METRIC_COMPLETE"
                        and result.get("complete_case_count") == expected_case_count
                        and len(current) == expected_case_count)
            if not complete:
                result["status"] = "INCOMPLETE"
            results.append(result)
            status = {"status": "RUNNING" if complete else "INCOMPLETE",
                      "N": rounds, "models": models, "results": results,
                      "completed_model_rounds": sum(r.get("status") == "FULL_METRIC_COMPLETE" for r in results)}
            persist(status)
            if not complete:
                return status
            hashes[model].update(current)
    status["status"] = "COMPLETE"
    persist(status)
    return status


def run_rounds_parallel(*, models: list[str], rounds: int, execute, persist, expected_case_count: int = 28) -> dict:
    """Run one worker per model, retaining the full-metric barrier per round.

    Workers own distinct model directories. Only the caller writes study
    checkpoints, and every worker in the current round is joined even if its
    peer fails. A partial round never opens the next round.
    """
    from concurrent.futures import ThreadPoolExecutor

    if expected_case_count < 1 or rounds < 1 or not models or len(set(models)) != len(models):
        raise ValueError("positive rounds and unique model identities required")
    results = []
    hashes = {model: set() for model in models}
    with ThreadPoolExecutor(max_workers=len(models)) as workers:
        for number in range(1, rounds + 1):
            futures = {model: workers.submit(execute, number, model) for model in models}
            round_failed = False
            for model in models:
                try:
                    result = dict(futures[model].result())
                except Exception as exc:
                    result = {"status": "INCOMPLETE", "complete_case_count": 0,
                              "issues": [f"{type(exc).__name__}: {exc}"]}
                result.update(round=number, model=model)
                current = set(result.get("record_hashes", []))
                if current & hashes[model]:
                    result["status"] = "INCOMPLETE"
                    result.setdefault("issues", []).append("run records reused across independent rounds")
                complete = (result.get("status") == "FULL_METRIC_COMPLETE"
                            and result.get("complete_case_count") == expected_case_count
                            and len(current) == expected_case_count)
                if complete:
                    hashes[model].update(current)
                else:
                    result["status"] = "INCOMPLETE"
                    round_failed = True
                results.append(result)
                status = {"status": "INCOMPLETE" if round_failed else "RUNNING",
                          "N": rounds, "models": models, "results": list(results),
                          "parallel_models": True,
                          "completed_model_rounds": sum(r.get("status") == "FULL_METRIC_COMPLETE" for r in results)}
                persist(status)
            if round_failed:
                return status
    status["status"] = "COMPLETE"
    persist(status)
    return status


def run_collection_fallback(*, rounds: int, execute, persist, initial_results=(), expected_case_count: int = 28) -> dict:
    """User-authorized Luna advancement while Terra reports provider rate limits.

    This schedules raw collection only. COLLECTION_COMPLETE never means that
    scientific metrics are complete; run_rounds still performs that audit.
    Retry Terra after each completed Luna round, without reusing observations
    across independent rounds or skipping a partially collected Luna round.
    """
    if expected_case_count < 1 or rounds < 1:
        raise ValueError("positive rounds required")
    completed = {"luna": 0, "terra": 0}
    hashes = {"luna": set(), "terra": set()}
    results, attempts = {}, []
    for original in sorted(initial_results, key=lambda r: (r["round"], r["model"])):
        result = dict(original)
        model, number = result["model"], result["round"]
        current = set(result.get("record_hashes", []))
        if (model not in completed or number != completed[model] + 1 or number > rounds
                or result.get("status") != "COLLECTION_COMPLETE"
                or result.get("complete_case_count") != expected_case_count or len(current) != expected_case_count
                or current & hashes[model]):
            raise ValueError("initial collections must be independently audited contiguous complete rounds")
        completed[model] = number
        hashes[model].update(current)
        results[(number, model)] = result

    def attempt(model):
        number = completed[model] + 1
        try:
            result = dict(execute(number, model))
        except Exception as exc:
            result = {"status": "INCOMPLETE", "issues": [f"{type(exc).__name__}: {exc}"]}
        result.update(round=number, model=model)
        current = set(result.get("record_hashes", []))
        if current & hashes[model]:
            result.update(status="INCOMPLETE", failure_category="REUSED_RECORDS")
        complete = result.get("status") == "COLLECTION_COMPLETE" and result.get("complete_case_count") == expected_case_count and len(current) == expected_case_count
        if not complete:
            result["status"] = "INCOMPLETE"
        else:
            hashes[model].update(current)
            completed[model] = number
        results[(number, model)] = result
        attempts.append(dict(result))
        checkpoint("RUNNING" if complete else "INCOMPLETE")
        return result

    def checkpoint(status):
        payload = {"status": status, "N": rounds, "scientific_metrics_complete": False,
                   "completed_collection_rounds": dict(completed),
                   "results": list(results.values()), "attempts": list(attempts)}
        persist(payload)
        return payload

    while completed["terra"] < rounds:
        if completed["luna"] <= completed["terra"]:
            if attempt("luna")["status"] != "COLLECTION_COMPLETE":
                return checkpoint("INCOMPLETE")
        terra = attempt("terra")
        if terra["status"] == "COLLECTION_COMPLETE":
            continue
        if terra.get("failure_category") != "RATE_LIMIT" or completed["luna"] == rounds:
            return checkpoint("INCOMPLETE")
        if attempt("luna")["status"] != "COLLECTION_COMPLETE":
            return checkpoint("INCOMPLETE")
    return checkpoint("COLLECTION_COMPLETE")
