#!/usr/bin/env python3
"""Resumable yiapi rebuild pipeline for the 96-case Luna/Terra experiment.

The pipeline deliberately composes the existing benchmark collector and host
runtime.  It does not change Case, Ground Truth, or metric definitions.

Stages
-------
1. Preflight yiapi and create run-scoped Responses configuration.
2. Register historical collections/metrics in a shared immutable archive.
3. Re-score completed Fable/Sonnet answers with the v2 core scorer.
4. Collect three fresh 96-case rounds, Luna before Terra, with failover.
5. Re-score every fresh completed answer and publish aggregate means.

The command is safe to resume.  Provider failures are persisted as typed
events and do not turn into scientific zeroes.  A malformed or disputed review
gets a bounded second independent review; disagreement is retained and excluded
from the automatic aggregate until inspected.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import (  # noqa: E402
    CORE_SCHEMA_VERSION,
    SCORER_IMPLEMENTATION_VERSION,
    CoreCaseScorer,
    judgment_schema,
    scoring_prompt,
)
from flowintentbench.experiment_scope import case_inventory  # noqa: E402
from flowintentbench.expansion_evaluation import load_development_case, read_json  # noqa: E402
from flowintentbench.ground_truth import GroundTruth  # noqa: E402
from flowintentbench.model_runner import RunRecord  # noqa: E402
from flowintentbench.model_runner import EvaluationTarget  # noqa: E402
from flowintentbench.providers import ThirdPartyResponsesAdapter  # noqa: E402
from flowintentbench.runtime_config import resolve_provider_configuration  # noqa: E402
from flowintentbench.agent_profile import load_agent_profile  # noqa: E402
from flowintentbench.schema import validate_model_input  # noqa: E402
from scripts.run_real_model_pilot import run_real_model_pilot  # noqa: E402
from scripts.run_expansion_file_evaluation import file_sha256  # noqa: E402
from scripts.shared_history import display_path, store_snapshot, tree_sha256  # noqa: E402
from scripts.yiapi_transport import responses_transport, validate_review  # noqa: E402
from flowintentbench.model_runner import ModelAdapterError  # noqa: E402


def _network_open(request: urllib.request.Request, *, timeout: float):
    """Use the repository's explicit direct/proxy policy, never env fallback."""
    from scripts.third_party_benchmark_transport import network_configuration, network_opener
    return network_opener(network_configuration())(request, timeout=timeout)


PIPELINE_SCHEMA = "yiapi-rebuild-pipeline-v1"
BASE_URL = "https://yiapi.ai/v1"
REVIEWER_MODEL = "gpt-6-astra"
TEST_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
MODEL_AGENTS = {
    "gpt-5.6-luna": "luna_responses_agent.yaml",
    "gpt-5.6-terra": "terra_responses_agent.yaml",
}
HISTORICAL_ROOTS = {
    "fable_sonnet": ROOT / "outputs" / "claude_resume_eval",
    "luna_terra": ROOT / "archive" / "outputs" / "expansion96_n3_subagents",
}
HISTORICAL_ARTIFACTS = (
    ROOT / "outputs" / "core_case_comparison_20260920",
    ROOT / "archive" / "outputs" / "expansion96_n3_claude",
    ROOT / "archive" / "outputs" / "expansion96_n3_subagents",
)
HISTORY_OBJECTS = ROOT / "archive" / "history_objects"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return tree_sha256(path)


def _read_auth(path: Path) -> str:
    """Read a key without ever writing or logging its value."""
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[Any] = []
    if isinstance(value, str):
        candidates.append(value)
    elif isinstance(value, Mapping):
        for name in ("YIAPI_API_KEY", "OPENAI_API_KEY", "api_key", "key", "token"):
            candidates.append(value.get(name))
        for nested in value.values():
            if isinstance(nested, Mapping):
                candidates.extend(nested.get(name) for name in ("YIAPI_API_KEY", "OPENAI_API_KEY", "api_key", "key", "token"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise RuntimeError(f"no API key found in {path}; expected YIAPI_API_KEY or OPENAI_API_KEY")


def _require_benchmark() -> None:
    from scripts.benchmark_runtime import require_benchmark_runtime
    require_benchmark_runtime()

def _event(root: Path, event: str, **payload: Any) -> None:
    row = {"epoch": time.time(), "event": event, **payload}
    path = root / "phase_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _state(root: Path, **updates: Any) -> dict[str, Any]:
    path = root / "pipeline_state.json"
    current: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, json.JSONDecodeError):
            pass
    current.update(updates)
    current["schema_version"] = PIPELINE_SCHEMA
    current["updated_epoch"] = time.time()
    _atomic_json(path, current)
    return current


def _display_path(path: Path) -> str:
    return display_path(path, ROOT)


def _shared_snapshot(source: Path) -> dict[str, Any]:
    return store_snapshot(source, repository_root=ROOT, object_root=HISTORY_OBJECTS)


def _health(root: Path, key: str, *, required: Sequence[str] = ()) -> dict[str, Any]:
    """Read-only catalog probe.  No answer-generation request is made here."""
    url = BASE_URL.rstrip("/") + "/models"
    receipt: dict[str, Any] = {"endpoint": url, "method": "GET", "started_epoch": time.time()}
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    try:
        with _network_open(request, timeout=30) as response:
            body = response.read()
            receipt.update(http_status=response.status, content_type=response.headers.get("content-type"), response_sha256=hashlib.sha256(body).hexdigest())
        value = json.loads(body)
        ids = sorted(str(item.get("id")) for item in value.get("data", ()) if isinstance(item, Mapping) and item.get("id"))
        missing = sorted(set(required) - set(ids))
        receipt.update(status="HEALTHY" if not missing else "MISSING_MODELS", model_count=len(ids), model_ids=ids, missing_models=missing)
        # Confirm the declared Responses route without generating an answer:
        # an intentionally incomplete request should return a normal 4xx
        # validation response, while 404/5xx means the route is unusable.
        probe = urllib.request.Request(
            BASE_URL.rstrip("/") + "/responses", data=json.dumps({"model": REVIEWER_MODEL}).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST"
        )
        try:
            with _network_open(probe, timeout=30) as response:
                probe_status = response.status
                response.read(256)
        except urllib.error.HTTPError as exc:
            probe_status = exc.code
            exc.read(256)
        receipt["responses_probe_http_status"] = probe_status
        receipt["responses_route"] = "AVAILABLE" if probe_status in {400, 422} else "UNAVAILABLE"
        if receipt["responses_route"] != "AVAILABLE":
            receipt["status"] = "RESPONSES_ROUTE_UNAVAILABLE"
    except urllib.error.HTTPError as exc:
        body = exc.read(512)
        receipt.update(status="UNAVAILABLE", http_status=exc.code, detail=body.decode("utf-8", "replace")[:300])
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, ValueError) as exc:
        receipt.update(status="UNAVAILABLE", error_type=type(exc).__name__, detail=str(exc)[:300])
    receipt["finished_epoch"] = time.time()
    _atomic_json(root / "health" / "api_health.json", receipt)
    return receipt


def _write_runtime_files(root: Path) -> dict[str, Path]:
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    config = runtime / "yiapi_responses.toml"
    config.write_text(
        'model_provider = "yiapi"\nmodel = "gpt-5.6-luna"\nmodel_reasoning_effort = "xhigh"\n\n'
        '[model_providers.yiapi]\nbase_url = "https://yiapi.ai/v1"\nwire_api = "responses"\n'
        'api_key_env = "OPENAI_API_KEY"\n', encoding="utf-8"
    )
    agents: dict[str, Path] = {}
    for model, filename in MODEL_AGENTS.items():
        short = "luna" if "luna" in model else "terra"
        path = runtime / filename
        path.write_text(
            f"agent_id: gpt-5.6-{short}-xhigh-chat-yiapi-responses\n\n"
            "model:\n  provider: yiapi\n"
            f"  model_id: {model}\n  model_family: gpt-5.6\n  reasoning_effort: xhigh\n\n"
            "interface:\n  wire_api: responses\n\n"
            "runtime_profile: flow-python-host-network-v1\n\n"
            "tools:\n  - python\n", encoding="utf-8"
        )
        agents[model] = path
    evaluator = runtime / "evaluator_gpt6_xhigh.json"
    evaluator_value = {
        "server_config": str(config), "model_server_config": str(config),
        "evaluator_server_config": str(config), "continuation_server_config": str(config),
        "provider": "yiapi", "model": REVIEWER_MODEL,
        "model_configuration": {
            "wire_api": "responses", "execution_backend": "THIRD_PARTY_RESPONSES",
            "reasoning_effort": "xhigh", "max_output_tokens": 16384,
            "structured_output_mode": "json_object_with_local_validation",
            "structured_output_capability": "JSON_OBJECT",
        },
        "continuation_agent": str(agents["gpt-5.6-terra"]),
        "timeout_seconds": 300, "transport_attempts": 3,
        "retry_backoff_seconds": 2, "rate_limit_backoff_seconds": 20,
        "rate_limit_max_backoff_seconds": 120, "retry_jitter_seconds": 1,
        "continuation_timeout_seconds": 180, "continuation_attempts": 3,
        "continuation_retry_backoff_seconds": 2,
        "continuation_rate_limit_backoff_seconds": 20,
        "continuation_rate_limit_max_backoff_seconds": 120,
        "continuation_retry_jitter_seconds": 1, "max_continuation_hops": 8,
        "evaluator_infrastructure_attempt_cap": 3,
        "collection_infrastructure_attempt_cap": 4,
        "collection_case_timeout_seconds": 1800,
        "collection_provider_retries": 5,
        "collection_provider_retry_backoff_seconds": 1,
        "collection_rate_limit_backoff_seconds": 20,
        "collection_rate_limit_max_backoff_seconds": 120,
        "collection_circuit_breaker_threshold": 3,
        "collection_circuit_breaker_cooldown_seconds": 120,
        "retry_network_noncompletion": True,
        "case_process_timeout_seconds": 2100,
        "global_evaluation_deadline_seconds": 172800,
        "inter_case_delay_seconds": 2,
    }
    _atomic_json(evaluator, evaluator_value)
    return {"config": config, "evaluator": evaluator, **{f"agent_{model}": path for model, path in agents.items()}}


def _archive(root: Path) -> dict[str, Any]:
    existing_manifest = root / "archive_manifest.json"
    if existing_manifest.is_file():
        try:
            cached = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if cached.get("schema_version") == "yiapi-archive-v2":
                object_paths = [
                    ROOT / item["object_path"]
                    for item in cached.get("items", ())
                    if item.get("exists")
                ]
                if cached.get("items") and object_paths and all(path.exists() for path in object_paths):
                    _event(root, "archive_reused", item_count=len(cached["items"]))
                    return cached
        except (OSError, json.JSONDecodeError):
            pass
    rows = []
    for role, source in HISTORICAL_ROOTS.items():
        row = _shared_snapshot(source)
        row["role"] = role
        rows.append(row)
    for source in HISTORICAL_ARTIFACTS:
        row = _shared_snapshot(source)
        row["role"] = source.name
        rows.append(row)
    manifest = {
        "schema_version": "yiapi-archive-v2",
        "created_epoch": time.time(),
        "storage_root": _display_path(HISTORY_OBJECTS),
        "items": rows,
    }
    _atomic_json(root / "archive_manifest.json", manifest)
    _event(root, "archive_completed", item_count=len(rows))
    return manifest


def _records(root: Path, model: str | None = None) -> list[tuple[Path, RunRecord]]:
    candidates: dict[tuple[str, int, str, str], tuple[Path, RunRecord]] = {}
    for path in root.rglob("run_record.json") if root.exists() else ():
        try:
            record = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
        if model and record.model_id != model:
            continue
        if record.run_status.value != "COMPLETED" or not record.final_response:
            continue
        # ``trial_index`` restarts at one in each independent round.  Bind the
        # round directory as well so fresh N=3 observations remain distinct,
        # while a retry in the same round replaces its older attempt.
        match = re.search(r"round-\d+", str(path))
        round_scope = match.group(0) if match else "root"
        key = (record.case_id, int(record.trial_index), record.model_id, round_scope)
        old = candidates.get(key)
        if old is None or path.stat().st_mtime_ns > old[0].stat().st_mtime_ns:
            candidates[key] = (path, record)
    return sorted(candidates.values(), key=lambda item: (item[1].model_id, item[1].case_id, item[1].trial_index, str(item[0])))


def _all_record_statuses(root: Path, model: str | None = None) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in root.rglob("run_record.json") if root.exists() else ():
        try:
            record = RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
        if model and record.model_id != model:
            continue
        if str(record.run_id) in seen:
            continue
        seen.add(str(record.run_id))
        status = record.run_status.value
        counts[status] += 1
        if status != "COMPLETED":
            failures.append({"case_id": record.case_id, "trial_index": record.trial_index, "run_id": record.run_id, "status": status, "failure_reason": record.failure_reason, "recommended_action": "recollect_model_answer" if status in {"INFRASTRUCTURE_INVALID", "MODEL_NONCOMPLETION"} else "inspect"})
    return counts, failures


def _status_snapshot(root: Path) -> dict[str, Any]:
    """Return compact, watch-friendly progress counts for a running job."""
    manifest_count = 96
    collection: dict[str, Any] = {}
    for model in TEST_MODELS:
        short = "luna" if "luna" in model else "terra"
        rounds: dict[str, Any] = {}
        completed_total = 0
        for round_number in range(1, 4):
            round_root = root / "answers" / f"round-{round_number:02d}" / short
            state_path = round_root / "collection_state.json"
            latest_path = round_root / "latest_attempt.json"
            state = read_json(state_path) if state_path.is_file() else {}
            latest = read_json(latest_path) if latest_path.is_file() else {}
            completed = int(state.get("completed_case_count", latest.get("completed_count", 0)) or 0)
            completed = min(manifest_count, max(0, completed))
            completed_total += completed
            rounds[str(round_number)] = {
                "completed": completed,
                "expected": manifest_count,
                "pending": max(0, manifest_count - completed),
                "status": state.get("status", latest.get("status", "NOT_STARTED")),
            }
        statuses, failures = _all_record_statuses(root / "answers", model)
        collection[model] = {
            "completed_answers": completed_total,
            "expected_answers": manifest_count * 3,
            "rounds": rounds,
            "record_status_counts": dict(statuses),
            "failed_records": len(failures),
        }

    evaluation: dict[str, Any] = {}
    for base in (root / "new_evaluation", root / "legacy_evaluation"):
        if not base.is_dir():
            continue
        for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            index_path = model_dir / "index.json"
            if not index_path.is_file():
                continue
            index = read_json(index_path)
            counts = index.get("status_counts") or index.get("all_record_status_counts") or {}
            evaluation[f"{base.name}/{model_dir.name}"] = {
                "evaluated": int(index.get("record_count", 0) or 0),
                "core_scored": int(counts.get("CORE_SCORED", 0) or 0),
                "review_errors": int(counts.get("REVIEW_ERROR", 0) or 0),
                "status_counts": counts,
            }
    state_path = root / "pipeline_state.json"
    pipeline = read_json(state_path) if state_path.is_file() else {"status": "NOT_STARTED"}
    return {"pipeline": pipeline, "collection": collection, "evaluation": evaluation, "updated_epoch": time.time()}


def _case_payload(manifest_rows: Mapping[str, Mapping[str, Any]], case_id: str) -> tuple[dict[str, Any], Any]:
    row = manifest_rows[case_id]
    if "evaluation_material_path" in row:
        case_input, metadata, gt, _ = load_development_case(ROOT, row)
        payload = case_input.model_dump(mode="json")
        payload["case_id"] = case_id
        payload["principal_operationalization_dimensions"] = list(getattr(metadata, "principal_operationalization_dimensions", ()))
        payload["unresolved_operationalization_dimensions"] = list(getattr(metadata, "unresolved_operationalization_dimensions", ()))
        payload["finding_goal"] = getattr(metadata, "finding_goal", None)
        return payload, gt
    input_path, gt_path = ROOT / row["case_input_path"], ROOT / row["ground_truth_path"]
    if file_sha256(input_path) != row["case_input_sha256"] or file_sha256(gt_path) != row["ground_truth_sha256"]:
        raise ValueError(f"frozen case/GT digest mismatch: {case_id}")
    return validate_model_input(read_json(input_path)).model_dump(mode="json"), GroundTruth.model_validate(read_json(gt_path))


def _response_text(value: Mapping[str, Any]) -> str:
    if isinstance(value.get("output_text"), str) and value["output_text"].strip():
        return value["output_text"]
    parts: list[str] = []
    for item in value.get("output", ()):
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        for content in item.get("content", ()):
            if isinstance(content, Mapping) and content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    text = "".join(parts).strip()
    if not text:
        raise ValueError("Responses response contained no output text")
    return text


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(value, dict):
        raise ValueError("reviewer output is not a JSON object")
    return value


class YiapiReviewer:
    def __init__(self, root: Path, key: str, *, max_attempts: int = 3):
        self.root, self.key, self.max_attempts = root, key, max_attempts
        self.lock = threading.Lock()
        self.failure_streak = 0
        self.blocked_until = 0.0
        self.fatal_error: str | None = None

    def call(self, prompt: str, schema: Mapping[str, Any], *, case_id: str, attempt: int) -> dict[str, Any]:
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            if time.time() < self.blocked_until:
                raise RuntimeError("review API paused after repeated provider failures")
        # YiAPI exposes the Responses route but its strict JSON-schema subset
        # is narrower than the OpenAI-compatible wire contract.  Use JSON
        # mode by default and validate the parsed object locally; callers can
        # opt back into strict mode for a provider that supports this schema.
        output_mode = os.environ.get("YIAPI_REVIEW_FORMAT", "json_object").strip().lower()
        if output_mode == "strict_json_schema":
            text_format: dict[str, Any] = {
                "type": "json_schema", "name": "flowintentbench_core_case_review",
                "strict": True, "schema": dict(schema),
            }
        else:
            output_mode = "json_object"
            text_format = {"type": "json_object"}
        root_required = ", ".join(schema.get("required", ()))
        dim_schema = schema.get("properties", {}).get("operationalization", {}).get("properties", {}).get("dimensions", {})
        dimensions = ", ".join(dim_schema.get("items", {}).get("properties", {}).get("dimension", {}).get("enum", ()))
        contract = (
            "Root required fields: " + root_required + ". "
            "operationalization.dimensions is an array of objects with dimension, status, "
            "statement, evidence_present, evidence_span, evidence_text, branch_matches. "
            "findings is an array of objects with prediction_id, statement, eligible, "
            "evidence_present, evidence_span, evidence_text, value, unit, matches. "
            "Each match has branch_id, gt_finding_id, semantic_match, numeric_match, "
            "role_match, consistent. Allowed dimensions: " + dimensions + "."
        )
        request_body = {
            "model": REVIEWER_MODEL, "store": False,
            "stream": True,
            "reasoning": {"effort": "xhigh"},
            "instructions": (
                "Return only one JSON object. Do not add commentary. " + contract
            ),
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "text": {"format": text_format},
            # Review JSON is bounded and never needs the full model context;
            # keeping this cap moderate prevents YiAPI's xhigh route from
            # holding an HTTP stream open while reserving an oversized output.
            "max_output_tokens": int(os.environ.get("YIAPI_REVIEW_MAX_OUTPUT_TOKENS", "4096")),
        }
        token = f"{time.time_ns()}-{case_id}-attempt-{attempt}"
        folder = self.root / "review_api" / token
        folder.mkdir(parents=True, exist_ok=True)
        _atomic_json(folder / "request.json", request_body)
        request = urllib.request.Request(
            BASE_URL.rstrip("/") + "/responses", data=json.dumps(request_body, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json", "Connection": "close"}, method="POST"
        )
        started = time.time()
        try:
            raw = responses_transport(request.full_url, dict(request.header_items()), request.data, 300)
            status = 200
            (folder / "response.json").write_bytes(raw)
            payload = json.loads(raw)
            receipt = {"status": "RECEIVED", "http_status": status, "response_sha256": hashlib.sha256(raw).hexdigest(), "elapsed_seconds": time.time() - started, "model": REVIEWER_MODEL, "output_mode": output_mode}
            _atomic_json(folder / "receipt.json", receipt)
            result = validate_review(_parse_json(_response_text(payload)), schema)
            with self.lock:
                self.failure_streak = 0
                self.blocked_until = 0.0
            return result
        except urllib.error.HTTPError as exc:
            body = exc.read(1000)
            receipt = {"status": "HTTP_ERROR", "http_status": exc.code, "detail": body.decode("utf-8", "replace")[:500], "elapsed_seconds": time.time() - started}
            _atomic_json(folder / "receipt.json", receipt)
            if exc.code in {408, 429, 500, 502, 503, 504}:
                with self.lock:
                    self.failure_streak += 1
                    if self.failure_streak >= 3:
                        self.blocked_until = time.time() + 120
            raise RuntimeError(f"review HTTP {exc.code}: {receipt['detail']}") from exc
        except (ModelAdapterError, urllib.error.URLError, socket.timeout, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            _atomic_json(folder / "receipt.json", {"status": "ERROR", "error_type": type(exc).__name__, "detail": str(exc)[:500], "elapsed_seconds": time.time() - started})
            with self.lock:
                if isinstance(exc, ModelAdapterError) and not _transient_review_error(exc):
                    self.fatal_error = str(exc)
                self.failure_streak += 1
                if self.failure_streak >= 3:
                    self.blocked_until = time.time() + 120
            raise RuntimeError(f"review {type(exc).__name__}: {exc}") from exc


def _transient_review_error(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return any(token in text for token in ("429", "408", "500", "502", "503", "504", "timeout", "urlerror", "unavailable", "connection", "reset"))


def _scalar_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-12
    return a == b


def _review_one(path: Path, record: RunRecord, rows: Mapping[str, Mapping[str, Any]], output: Path, reviewer: YiapiReviewer, *, force: bool = False) -> dict[str, Any]:
    digest = file_sha256(path)
    destination = output / "cases" / f"{record.model_id}__trial-{record.trial_index}__{record.case_id}__{record.run_id}.json"
    if destination.is_file() and not force:
        try:
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if cached.get("run_record_sha256") == digest and cached.get("scorer_implementation_version") == SCORER_IMPLEMENTATION_VERSION and cached.get("status") in {"CORE_SCORED", "MODEL_NONCOMPLETION", "REVIEW_DISAGREEMENT"}:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    case_input, gt = _case_payload(rows, record.case_id)
    efficiency = {name: getattr(record, name, None) for name in ("input_tokens", "output_tokens", "model_turn_count", "tool_call_count", "python_execution_count", "wall_clock_time", "provider_reported_cost")}
    gt_packet = gt.model_dump(mode="json") if hasattr(gt, "model_dump") else gt
    dimensions = tuple(item.get("dimension") for branch in gt_packet.get("acceptable_operationalizations", ()) for item in branch.get("decisions", ()) if item.get("dimension"))
    dimensions = tuple(dict.fromkeys(dimensions)) or None
    schema = judgment_schema(dimensions)
    primary: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt in range(1, reviewer.max_attempts + 1):
        try:
            prompt = scoring_prompt(case_input, record.final_response, gt)
            judgment = reviewer.call(prompt, schema, case_id=record.case_id, attempt=attempt)
            primary = CoreCaseScorer().score(case_input, record.final_response, gt, efficiency=efficiency, judgment=judgment)
            primary["review_attempt"] = attempt
            break
        except Exception as exc:
            errors.append(f"attempt-{attempt}: {type(exc).__name__}: {exc}")
            if attempt >= reviewer.max_attempts or not (_transient_review_error(exc) or isinstance(exc, ValueError) or "ValueError" in str(exc)):
                break
            time.sleep(min(60, 2 ** (attempt - 1)))
    if primary is None:
        result = {"schema_version": CORE_SCHEMA_VERSION, "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION, "case_id": record.case_id, "model_id": record.model_id, "run_id": record.run_id, "run_record_sha256": digest, "status": "REVIEW_ERROR", "error": errors}
    else:
        result = dict(primary)
        result.update({"model_id": record.model_id, "trial_index": record.trial_index, "run_id": record.run_id, "run_record_sha256": digest, "review_errors": errors})
        # Partial evidence/ambiguity is a review-detection trigger.  A second
        # reviewer is independent and its full judgment is retained.
        if primary.get("status") == "CORE_SCORED" and primary.get("audit_status") == "PARTIAL":
            try:
                second_judgment = reviewer.call(scoring_prompt(case_input, record.final_response, gt), schema, case_id=record.case_id, attempt=reviewer.max_attempts + 1)
                second = CoreCaseScorer().score(case_input, record.final_response, gt, efficiency=efficiency, judgment=second_judgment)
                result["secondary"] = second
                keys = ("o_score", "finding_precision", "finding_requirement_recall", "numeric_accuracy", "c_score", "evidence_coverage")
                same = all(_scalar_equal(primary.get("metrics", {}).get(key), second.get("metrics", {}).get(key)) for key in keys)
                result["secondary_status"] = "AGREED" if same else "DISAGREEMENT"
                if not same:
                    result["status"] = "REVIEW_DISAGREEMENT"
            except Exception as exc:
                result["secondary_status"] = "SECOND_REVIEW_PENDING"
                result["secondary_error"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "SECOND_REVIEW_PENDING"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, result)
    return result


def _score_collection(collection_root: Path, output: Path, rows: Mapping[str, Mapping[str, Any]], key: str, *, workers: int, model: str | None = None, case_ids: set[str] | None = None, limit: int | None = None) -> dict[str, Any]:
    records = _records(collection_root, model)
    records = [(path, record) for path, record in records if record.case_id in rows]
    if case_ids is not None:
        records = [(path, record) for path, record in records if record.case_id in case_ids]
    if limit is not None:
        records = records[:limit]
    all_status_counts, failed_records = _all_record_statuses(collection_root, model)
    reviewer = YiapiReviewer(output, key)
    if not records:
        summary = {"status": "NO_COMPLETED_RECORDS", "record_count": 0, "all_record_status_counts": dict(all_status_counts), "failed_records": failed_records, "results": []}
        _atomic_json(output / "index.json", summary)
        return summary
    results: list[dict[str, Any]] = []
    _atomic_json(output / "progress.json", {"expected": len(records), "finished": 0, "status_counts": {}, "status": "RUNNING"})
    # Keep only worker-count futures in flight. A schema/auth failure stops
    # admission; a circuit break yields remaining work to other phases.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending = iter(records)
        futures = set()
        while True:
            while len(futures) < max(1, workers) and not reviewer.fatal_error and time.time() >= reviewer.blocked_until:
                item = next(pending, None)
                if item is None:
                    break
                path, record = item
                futures.add(pool.submit(_review_one, path, record, rows, output, reviewer))
            if not futures:
                break
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"status": "REVIEW_ERROR", "error": f"{type(exc).__name__}: {exc}"})
            _atomic_json(output / "progress.json", {"expected": len(records), "finished": len(results), "status_counts": dict(Counter(r.get("status") for r in results)), "updated_epoch": time.time(), "status": "RUNNING"})
    counts = Counter(str(item.get("status")) for item in results)
    metrics: dict[str, list[float]] = {}
    by_trial: dict[str, dict[str, list[float]]] = {}
    for item in results:
        if item.get("status") != "CORE_SCORED":
            continue
        trial_key = str(item.get("trial_index", "all"))
        by_trial.setdefault(trial_key, {})
        for name, value in (item.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics.setdefault(name, []).append(float(value))
                by_trial[trial_key].setdefault(name, []).append(float(value))
    means = {name: sum(values) / len(values) if values else None for name, values in metrics.items()}
    means_by_trial = {trial: {name: sum(values) / len(values) if values else None for name, values in values_by_metric.items()} for trial, values_by_metric in by_trial.items()}
    summary = {"schema_version": CORE_SCHEMA_VERSION, "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION, "reviewer_model": REVIEWER_MODEL, "reviewer_effort": "xhigh", "record_count": len(results), "all_record_status_counts": dict(all_status_counts), "failed_records": failed_records, "status_counts": dict(sorted(counts.items())), "means_core_scored_only": means, "means_by_trial_core_scored_only": means_by_trial, "secondary_review_policy": {"partial_audit_triggers_second_review": True, "disagreement_excluded_from_means": True}, "results": results}
    _atomic_json(output / "index.json", summary)
    summary.update(expected_count=len(records), pending_count=len(records) - len(results), fatal_error=reviewer.fatal_error)
    _atomic_json(output / "index.json", summary)
    _atomic_json(output / "progress.json", {"expected": len(records), "finished": len(results), "status_counts": dict(counts), "updated_epoch": time.time(), "status": "COMPLETE" if counts.get("CORE_SCORED", 0) == len(records) else "INCOMPLETE", "fatal_error": reviewer.fatal_error})
    return summary


def _manifest_and_preflight(root: Path, manifest_path: Path, runtime: Mapping[str, Path]) -> Path:
    if shutil.which("proot") is None:
        raise RuntimeError(
            "the configured flow-python-host-network-v1 profile requires proot; "
            "run `python scripts/install_host_runtime.py` in the benchmark_py3.12 environment first"
        )
    manifest = read_json(manifest_path)
    inventory = case_inventory(manifest, require_dataset=True)
    if len(inventory) != 96:
        raise ValueError(f"the expansion collection requires 96 cases, got {len(inventory)}")
    # The standard full-dataset N=1 preflight intentionally accepts only the
    # separate 28-case integration manifest.  This pipeline uses the existing
    # expansion collector contract instead, while retaining the same frozen
    # input hashes and host runtime checks.
    checked = 0
    for case_id, row in inventory.items():
        input_path = ROOT / row["case_input_path"]
        gt_path = ROOT / row["ground_truth_path"]
        if file_sha256(input_path) != row["case_input_sha256"]:
            raise ValueError(f"case input checksum mismatch: {case_id}")
        if file_sha256(gt_path) != row["ground_truth_sha256"]:
            raise ValueError(f"ground truth checksum mismatch: {case_id}")
        checked += 1
    report_path = root / "runtime" / "preflight_report.json"
    if not report_path.is_file():
        report = {
            "status": "PASS", "provider_calls": 0, "case_count": checked,
            "manifest_sha256": _sha256(manifest_path), "runtime_config_sha256": _sha256(runtime["config"]),
            "runtime_profile": "flow-python-host-network-v1",
            "collector": "scripts.run_real_model_pilot + BenchmarkRunner",
            "note": "96-case expansion manifest; standard 28-case N=1 preflight is not applicable",
        }
        _atomic_json(report_path, report)
    return report_path


def _collection_attempt(root: Path, manifest_path: Path, runtime: Mapping[str, Path], key: str, model: str, round_number: int, *, resume: bool) -> dict[str, Any]:
    model_root = root / "answers" / f"round-{round_number:02d}" / ("luna" if "luna" in model else "terra")
    model_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    inventory = case_inventory(manifest, require_dataset=True)
    existing = {record.case_id for _, record in _records(model_root, model)}
    pending = [case_id for case_id in inventory if case_id not in existing]
    expected = len(inventory)
    if not pending:
        result = {"model": model, "round": round_number, "status": "COMPLETE", "exit_code": 0, "completed_count": expected, "expected_count": expected, "transient_provider_failure": False, "reason": "", "collection_root": str(model_root)}
        _atomic_json(model_root / "latest_attempt.json", result)
        return result

    attempt_number = 1
    for candidate in model_root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-(\d+)", candidate.name)
        if match:
            attempt_number = max(attempt_number, int(match.group(1)) + 1)
    attempt_root = model_root / f"attempt-{attempt_number:02d}"
    profile = load_agent_profile(runtime[f"agent_{model}"], repository_root=ROOT)
    resolved = resolve_provider_configuration(
        role="model", config_path=runtime["config"], provider=profile.provider,
        model_id=profile.model_id, model_configuration=profile.model_configuration,
    )
    target = EvaluationTarget(resolved.provider, resolved.model_id, model_configuration=resolved.model_configuration)
    datasets: dict[str, list[str]] = {}
    for case_id in pending:
        datasets.setdefault(str(inventory[case_id]["dataset_id"]), []).append(case_id)
    records_seen = []
    errors: list[str] = []
    for dataset_id, case_ids in datasets.items():
        def factory(evaluation_target: EvaluationTarget):
            return ThirdPartyResponsesAdapter(
                evaluation_target, api_key=key, api_key_env=resolved.api_key_env,
                base_url=resolved.base_url, formal_mode=False, request_timeout_seconds=1800,
                transport=responses_transport,
            )
        try:
            records_seen.extend(run_real_model_pilot(
                datasets_root=ROOT / "datasets", dataset_id=dataset_id,
                case_ids=case_ids, target=target, output_root=attempt_root,
                adapter_factory=factory, agent_profile=profile, trial_index=1,
                request_timeout_seconds=1800, case_manifest_path=manifest_path,
            ))
        except Exception as exc:
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
            _event(root, "dataset_collection_error", model=model, round=round_number, dataset_id=dataset_id, error=errors[-1])
            # Continue with the next dataset so one gateway failure does not
            # discard already collected cases or block the alternate model.
            continue
    selected = _records(model_root, model)
    completed = len({record.case_id for _, record in selected})
    status = "COMPLETE" if completed >= expected else "INCOMPLETE"
    all_text = " ".join(errors + [str(record.failure_reason or "") for record in records_seen])
    transient = any(token in all_text.casefold() for token in ("rate", "quota", "timeout", "unavailable", "provider", "429", "503", "connection", "reset"))
    result = {"model": model, "round": round_number, "status": status, "exit_code": 0 if not errors else 2, "completed_count": completed, "expected_count": expected, "pending_count": max(0, expected - completed), "transient_provider_failure": transient, "reason": "; ".join(errors), "collection_root": str(model_root), "attempt_root": str(attempt_root)}
    _atomic_json(model_root / "latest_attempt.json", result)
    _atomic_json(model_root / "collection_state.json", {"model_id": model, "round": round_number, "expected_case_count": expected, "completed_case_count": completed, "pending_case_ids": sorted(set(inventory) - {record.case_id for _, record in selected}), "status": status, "updated_epoch": time.time()})
    return result


def _collect(root: Path, manifest_path: Path, runtime: Mapping[str, Path], key: str, *, resume: bool, max_attempts: int) -> dict[str, Any]:
    schedule: list[dict[str, Any]] = []
    for round_number in range(1, 4):
        pending = list(TEST_MODELS)
        attempts = Counter()
        paused_until: dict[str, float] = {}
        while pending:
            progress = False
            for model in tuple(pending):
                if time.time() < paused_until.get(model, 0):
                    continue
                attempts[model] += 1
                _event(root, "collection_attempt_started", model=model, round=round_number, attempt=attempts[model])
                result = _collection_attempt(root, manifest_path, runtime, key, model, round_number, resume=resume or attempts[model] > 1)
                schedule.append(result)
                _event(root, "collection_attempt_finished", **result)
                if result["status"] == "COMPLETE":
                    pending.remove(model)
                    progress = True
                elif result["transient_provider_failure"] and attempts[model] < max_attempts:
                    paused_until[model] = time.time() + min(120, 15 * attempts[model])
                    _event(root, "model_paused_after_provider_failure", model=model, round=round_number, retry_after_epoch=paused_until[model])
                elif attempts[model] >= max_attempts:
                    pending.remove(model)
                    _event(root, "model_collection_attempt_budget_exhausted", model=model, round=round_number)
                else:
                    paused_until[model] = time.time() + 10
            if pending and not progress:
                if all(time.time() < paused_until.get(model, 0) for model in pending):
                    time.sleep(min(30, max(1, min(paused_until[model] for model in pending) - time.time())))
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for row in schedule:
        latest[(int(row["round"]), str(row["model"]))] = row
    expected_keys = {(round_number, model) for round_number in range(1, 4) for model in TEST_MODELS}
    complete = bool(schedule) and expected_keys.issubset(latest) and all(latest[key]["status"] == "COMPLETE" for key in expected_keys)
    summary = {"status": "COMPLETE" if complete else "INCOMPLETE", "attempts": schedule, "latest": list(latest.values())}
    _atomic_json(root / "collection_schedule.json", summary)
    return summary


def _score_new(root: Path, manifest_path: Path, key: str, *, workers: int) -> dict[str, Any]:
    rows = {row["case_id"]: row for row in read_json(manifest_path)["cases"]}
    summaries = {}
    for model in TEST_MODELS:
        collection_root = root / "answers"
        output = root / "new_evaluation" / model
        summaries[model] = _score_collection(collection_root, output, rows, key, workers=workers, model=model)
    _atomic_json(root / "new_evaluation" / "summary.json", {"models": summaries, "reviewer_model": REVIEWER_MODEL, "reviewer_effort": "xhigh"})
    return summaries


def _evaluation_complete(summary: Mapping[str, Any], *, expected: int | None = None) -> bool:
    if not summary.get("record_count") or summary.get("pending_count") or summary.get("fatal_error"):
        return False
    statuses = summary.get("status_counts") or {}
    if statuses.get("REVIEW_ERROR", 0) or statuses.get("SECOND_REVIEW_PENDING", 0) or statuses.get("REVIEW_DISAGREEMENT", 0):
        return False
    if expected is not None:
        return statuses.get("CORE_SCORED", 0) == expected
    return statuses.get("CORE_SCORED", 0) == int(summary.get("record_count", 0))


def _score_legacy(root: Path, manifest_path: Path, key: str, *, workers: int) -> dict[str, Any]:
    rows = {row["case_id"]: row for row in read_json(manifest_path)["cases"]}
    summaries = {}
    for model, source in (("claude-fable-5-1", HISTORICAL_ROOTS["fable_sonnet"]), ("claude-sonnet-5", HISTORICAL_ROOTS["fable_sonnet"])):
        output = root / "legacy_evaluation" / model
        summaries[model] = _score_collection(source, output, rows, key, workers=workers, model=model)
    _atomic_json(root / "legacy_evaluation" / "summary.json", {"models": summaries, "reviewer_model": REVIEWER_MODEL, "reviewer_effort": "xhigh"})
    return summaries


def _prepare(root: Path, manifest_path: Path, auth_path: Path) -> tuple[str, dict[str, Path]]:
    key = _read_auth(auth_path)
    runtime = _write_runtime_files(root)
    receipt = _health(root, key, required=(REVIEWER_MODEL, *TEST_MODELS))
    if receipt.get("status") != "HEALTHY":
        raise RuntimeError(f"yiapi preflight failed; inspect {root / 'health' / 'api_health.json'}")
    env_digest = {"base_url": BASE_URL, "reviewer_model": REVIEWER_MODEL, "tested_models": list(TEST_MODELS), "effort": "xhigh", "runtime_config_sha256": _sha256(runtime["config"]), "evaluator_config_sha256": _sha256(runtime["evaluator"])}
    _atomic_json(root / "runtime" / "runtime_identity.json", env_digest)
    _state(root, status="PREFLIGHT_READY", manifest=str(manifest_path), runtime=env_digest)
    return key, runtime


def _default_root() -> Path:
    return ROOT / "outputs" / f"yiapi_rebuild_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"


def main(argv: Sequence[str] | None = None) -> int:
    _require_benchmark()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "preflight", "archive", "legacy-evaluate", "collect", "evaluate-new", "status"), default="all")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments" / "expansion_v1_development" / "case_manifest.json")
    parser.add_argument("--auth", type=Path, default=ROOT / "auth_yapi.json")
    parser.add_argument("--review-workers", type=int, default=2)
    parser.add_argument("--collection-attempts", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.review_workers < 1 or args.collection_attempts < 1:
        parser.error("worker/attempt limits must be positive")
    if args.run_root is not None and not str(args.run_root).strip():
        parser.error("--run-root 不能为空；请先设置 RUN_ROOT")
    root = (args.run_root or _default_root()).resolve()
    if root == ROOT.resolve():
        parser.error("--run-root 不能是项目根目录，请使用 outputs/yiapi_rebuild_<timestamp>")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, auth_path = args.manifest.resolve(), args.auth.resolve()
    if args.phase == "status":
        print(json.dumps(_status_snapshot(root), ensure_ascii=False, indent=2))
        return 0
    key, runtime = _prepare(root, manifest_path, auth_path)
    if args.phase == "preflight":
        print(json.dumps(json.loads((root / "health" / "api_health.json").read_text()), ensure_ascii=False, indent=2))
        return 0
    if args.phase in {"all", "archive"}:
        _state(root, status="ARCHIVING")
        _archive(root)
        _state(root, status="ARCHIVE_READY")
    if args.phase in {"all", "legacy-evaluate"}:
        _manifest_and_preflight(root, manifest_path, runtime)
        _state(root, status="LEGACY_EVALUATION_RUNNING")
        legacy_summary = _score_legacy(root, manifest_path, key, workers=args.review_workers)
        legacy_ok = all(_evaluation_complete(value) for value in legacy_summary.values())
        _state(root, status="LEGACY_EVALUATION_COMPLETE" if legacy_ok else "LEGACY_EVALUATION_INCOMPLETE")
        if args.phase == "legacy-evaluate":
            return 0
    if args.phase in {"all", "collect"}:
        if not (root / "archive_manifest.json").is_file():
            _state(root, status="ARCHIVING")
            _archive(root)
        _manifest_and_preflight(root, manifest_path, runtime)
        _state(root, status="COLLECTION_RUNNING", collection_order=["luna", "terra"])
        collection_summary = _collect(root, manifest_path, runtime, key, resume=args.resume, max_attempts=args.collection_attempts)
        _state(root, status="COLLECTION_COMPLETE" if collection_summary.get("status") == "COMPLETE" else "COLLECTION_INCOMPLETE")
        if args.phase == "all":
            # A reviewer outage is deliberately allowed to yield to model
            # collection.  Re-open the legacy review queue after collection
            # so transient GPT-6 failures do not leave step 1 incomplete.
            _state(root, status="LEGACY_EVALUATION_RETRY_AFTER_COLLECTION")
            legacy_summary = _score_legacy(root, manifest_path, key, workers=args.review_workers)
        if args.phase == "collect":
            return 0
    if args.phase in {"all", "evaluate-new"}:
        _state(root, status="NEW_EVALUATION_RUNNING")
        new_summary = _score_new(root, manifest_path, key, workers=args.review_workers)
        expected_per_model = len(case_inventory(read_json(manifest_path))) * 3
        new_ok = all(_evaluation_complete(value, expected=expected_per_model) for value in new_summary.values())
        _state(root, status="COMPLETE" if new_ok else "NEW_EVALUATION_INCOMPLETE")
        final_status = "COMPLETE" if new_ok else "INCOMPLETE"
    else:
        final_status = "COMPLETE"
    print(json.dumps({"status": final_status, "run_root": str(root)}, ensure_ascii=False))
    return 0 if final_status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
