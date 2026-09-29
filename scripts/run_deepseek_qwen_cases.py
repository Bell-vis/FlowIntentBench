#!/usr/bin/env python3
"""Collect 96 cases x N=1 per model using DeepSeek-V4.1-Flash and Qwen3.8-Max.

Commands: probe, prepare, preflight, collect, status. Credentials are read only
from the named fields in --auth (default: repository auth.json). The original
96-case inputs, BenchmarkRunner, isolated flow-python-v1 runtime, Python tool,
case timeout and provider retry policy are reused from run_gpt6_sol_cases.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import signal
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.agent_profile import load_agent_profile
from flowintentbench.model_runner import BenchmarkRunner, EvaluationTarget, schedule_formal_trials
from flowintentbench.providers import OpenAIChatCompletionsAdapter
from flowintentbench.python_runtime import PythonExecutionEnvironment
from scripts import run_gpt6_sol_cases as original
from scripts.deepseek_qwen_transport import PROVIDERS, chat_transport
from scripts.benchmark_runtime import require_benchmark_runtime

MODELS = tuple(PROVIDERS)
DEFAULT_OUTPUT = ROOT / "outputs/expansion96_n1_deepseek_qwen"
HOST_NETWORK_OUTPUT = ROOT / "outputs/expansion96_n1_deepseek_qwen_host_network"
SCHEMA = "deepseek-qwen-case-collection-v1"


def credentials(auth_path, models):
    value = json.loads(auth_path.read_text(encoding="utf-8"))
    keys = {}
    for model in models:
        field = PROVIDERS[model]["key_field"]
        key = value.get(field)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"missing credential field: {field}")
        keys[model] = key.strip()
    return keys


def profiles(host_network=False):
    return {model: load_agent_profile(spec["profile"] + ("-host-network" if host_network else ""), repository_root=ROOT)
            for model, spec in PROVIDERS.items()}


def configuration(host_network=False):
    return {
        "schema": SCHEMA, "models": list(MODELS), "repetitions": 1,
        "case_count": 96, "base_seed": original.SEED,
        "manifest_path": str(original.MANIFEST.relative_to(ROOT)),
        "manifest_sha256": original._sha256(original.MANIFEST),
        "runtime_profile": "flow-python-host-network-v1" if host_network else "flow-python-v1",
        "security_mode": "HOST_NETWORK_PROOT" if host_network else "LINUX_BUBBLEWRAP",
        "host_network": host_network, "strictly_comparable_to_isolated_runs": not host_network,
        "case_timeout_seconds": original.CASE_TIMEOUT_SECONDS,
        "provider_retries": original.PROVIDER_RETRIES,
        "provider_retry_backoff_seconds": original.PROVIDER_RETRY_BACKOFF_SECONDS,
        "provider_rate_limit_backoff_seconds": original.PROVIDER_RATE_LIMIT_BACKOFF_SECONDS,
        "provider_rate_limit_max_backoff_seconds": original.PROVIDER_RATE_LIMIT_MAX_BACKOFF_SECONDS,
        "collection_only": True, "evaluation": "NOT_RUN",
        "profiles": {model: {"agent_id": p.agent_id, "profile_sha256": p.profile_sha256,
                     "runtime_profile_sha256": p.runtime_profile.profile_sha256,
                     "provider": p.provider, "model_id": p.model_id, "tools": list(p.tools),
                     "base_url": PROVIDERS[model]["base_url"], "wire_api": p.wire_api,
                     "wire_options": PROVIDERS[model]["options"]}
                     for model, p in profiles(host_network).items()},
    }


def save(output, state):
    state["status_counts"] = dict(Counter(s["status"] for s in state["slots"]))
    state["by_model"] = {m: dict(Counter(s["status"] for s in state["slots"] if s["model_id"] == m)) for m in MODELS}
    state["collection_status"] = ("COLLECTION_COMPLETE" if all(s["status"] == "COMPLETED" for s in state["slots"])
                                  else "COLLECTION_INCOMPLETE")
    state["updated_epoch"] = time.time()
    original._write(output / "collection_state.json", state)
    original._write(output / "progress.json", {k: state[k] for k in
                    ("status_counts", "by_model", "collection_status", "total_slots", "updated_epoch")})


def prepare(output, host_network=False):
    config = configuration(host_network)
    inventory, _, _ = original._load_inputs()
    binding = original._digest(config)
    path = output / "collection_state.json"
    if path.exists():
        state = original._read(path)
        if state["configuration_sha256"] != binding:
            raise ValueError("output configuration differs; use a new --output-root")
        return state
    schedule = schedule_formal_trials(tuple(inventory), repetitions=1, base_seed=original.SEED,
                    case_family_by_id={cid: str(row.get("family_id", cid)) for cid, row in inventory.items()})
    slots = []
    for trial in schedule:
        for position, cid in enumerate(trial.case_ids):
            for model in MODELS:
                slots.append({"slot_id": original._slot_id(model, trial.trial_index, cid),
                    "model_id": model, "case_id": cid, "dataset_id": inventory[cid]["dataset_id"],
                    "trial_index": trial.trial_index, "ordering_seed": trial.ordering_seed,
                    "case_execution_position": position, "status": "PENDING"})
    state = {"schema_version": SCHEMA, "experiment_id": SCHEMA + "-" + str(time.time_ns()),
             "configuration": config, "configuration_sha256": binding, "slots": slots,
             "total_slots": len(slots), "scientific_evaluation_status": "NOT_RUN",
             "blinding": {"ground_truth_loaded": False, "case_input_only": True,
                          "network_available_to_model_tool": host_network}}
    original._write(output / "collection_config.json", config)
    save(output, state)
    return state


def probe_model(model, key):
    """Two-turn API/tool contract probe; never executes model-supplied code."""
    spec = PROVIDERS[model]
    messages = [{"role": "user", "content": 'Call python exactly once with code print(1+1), then reply with exactly 2 after the tool result.'}]
    tool = {"type": "function", "function": {"name": "python", "description": "Execute Python code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}
    started = time.monotonic()
    receipt = {"requested_model": model, "base_url": spec["base_url"], "wire_options": spec["options"],
               "status": "FAIL", "benchmark_observations": 0, "checked_epoch": time.time()}
    def request():
        return json.loads(chat_transport(spec["base_url"] + "/chat/completions",
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json.dumps({"model": model, "messages": messages, "tools": [tool], "tool_choice": "auto",
                        "parallel_tool_calls": False, "max_tokens": 4096}).encode(), 90))
    try:
        first = request()
        msg = first["choices"][0]["message"]
        calls = msg.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "python":
            raise ValueError("probe did not return one Python call")
        # Allow only the prescribed calculation; model-generated code is not evaluated here.
        code = json.loads(calls[0]["function"]["arguments"])["code"]
        if "".join(code.split()) != "print(1+1)":
            raise ValueError("probe returned an unexpected Python calculation")
        messages.extend([msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "2\n"}])
        second = request()
        choice = second["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"].get("content", "").strip() != "2":
            raise ValueError("probe did not finish with the expected answer")
        receipt.update(status="PASS", response_models=[first["model"], second["model"]],
                       response_ids=[first.get("id"), second.get("id")],
                       usage=[first.get("usage"), second.get("usage")], tool_roundtrip=True,
                       reasoning_returned=bool(msg.get("reasoning_content")))
    except Exception as exc:
        receipt["error"] = str(exc).replace(key, "[REDACTED]")
    receipt["seconds"] = round(time.monotonic() - started, 3)
    return receipt


def probe(output, auth, models):
    keys = credentials(auth, models)
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = [pool.submit(probe_model, model, keys[model]) for model in models]
        results = [f.result() for f in futures]
    report = {"status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL", "results": results}
    original._write(output / "api_probe.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def runtime_preflight(output, host_network=False):
    checks = {"status": "FAIL", "provider_calls": 0,
              "runtime_profile": "flow-python-host-network-v1" if host_network else "flow-python-v1"}
    try:
        if host_network:
            checks.update(security_mode="HOST_NETWORK_PROOT", proot=shutil.which("proot"),
                          network_isolation_required=False, strictly_comparable_to_isolated_runs=False)
            if not checks["proot"]:
                raise RuntimeError("host-network runtime requires proot")
        else:
            checks.update(original._runtime_preflight())
        inventory, loaded, manifests = original._load_inputs()
        case_id = next(iter(loaded))
        with PythonExecutionEnvironment(loaded[case_id], manifest=manifests[inventory[case_id]["dataset_id"]],
                runtime_profile=profiles(host_network)[MODELS[0]].runtime_profile, workspace_parent=output / "runtime-work") as runtime:
            result = runtime.execute("import pathlib, numpy, scipy, matplotlib, vtk; assert pathlib.Path('/case/case_input.json').is_file(); print('READY')")
            checks.update(smoke_success=result.success, runtime_metadata=runtime.runtime_metadata)
            if not result.success:
                raise RuntimeError(f"Python runtime smoke failed: {result.exception}")
        checks["status"] = "PASS"
    except Exception as exc:
        checks["error"] = str(exc)
        raise
    finally:
        original._write(output / "runtime_preflight.json", checks)
    return checks


def run_slot(output, slot, loaded, manifests, profile, key, experiment_id):
    model = slot["model_id"]
    spec = PROVIDERS[model]
    directory = output / "runs" / model / f"trial-{slot['trial_index']:02d}" / slot["case_id"]
    directory = directory / f"attempt-{len(slot.get('attempt_history', [])) + 1:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    target = EvaluationTarget(profile.provider, model,
        model_configuration={**profile.model_configuration, **spec["options"], "stream": True})
    request_number = 0
    def audited_transport(url, headers, body, timeout):
        nonlocal request_number
        request_number += 1
        receipt = {"request_number": request_number, "requested_model": model,
                   "started_epoch": time.time(), "status": "REQUEST_RUNNING"}
        original._write(directory / "api_progress.json", receipt)
        try:
            response_bytes = chat_transport(url, headers, body, timeout)
            response = json.loads(response_bytes)
            choice = response["choices"][0]
            receipt.update(status="RESPONSE_RECEIVED", response_id=response.get("id"),
                response_model=response.get("model"), finish_reason=choice.get("finish_reason"),
                tool_names=[c.get("function", {}).get("name") for c in choice["message"].get("tool_calls", [])],
                usage=response.get("usage"), finished_epoch=time.time())
            return response_bytes
        except Exception as exc:
            receipt.update(status="REQUEST_FAILED", error=str(exc).replace(key, "[REDACTED]"), finished_epoch=time.time())
            raise
        finally:
            original._write(directory / "api_progress.json", receipt)
            original._write(directory / "api_receipts" / f"request-{request_number:04d}.json", receipt)
    def factory(target):
        return OpenAIChatCompletionsAdapter(target, api_key=key, base_url=spec["base_url"],
            formal_mode=False, request_timeout_seconds=original.CASE_TIMEOUT_SECONDS, transport=audited_transport)
    runner = BenchmarkRunner(formal_mode=False, agent_profile=profile,
        provider_retries=original.PROVIDER_RETRIES,
        provider_retry_backoff_seconds=original.PROVIDER_RETRY_BACKOFF_SECONDS,
        provider_rate_limit_backoff_seconds=original.PROVIDER_RATE_LIMIT_BACKOFF_SECONDS,
        provider_rate_limit_max_backoff_seconds=original.PROVIDER_RATE_LIMIT_MAX_BACKOFF_SECONDS,
        case_timeout_seconds=original.CASE_TIMEOUT_SECONDS,
        runtime_kwargs={"runtime_profile": profile.runtime_profile, "workspace_parent": output / "runtime-work"})
    record = runner.run_case(loaded[slot["case_id"]], target=target, case_id=slot["case_id"],
        trial_index=slot["trial_index"], ordering_seed=slot["ordering_seed"],
        case_execution_position=slot["case_execution_position"], manifest=manifests[slot["dataset_id"]],
        trajectory_path=directory / "trajectory.json", adapter_factory=factory,
        experiment_id=experiment_id, benchmark_release_id=None)
    record.write_json(directory / "run_record.json")
    return {"status": record.run_status.value, "failure_reason": record.failure_reason,
            "run_record_path": str((directory / "run_record.json").relative_to(output)),
            "run_record_sha256": original._sha256(directory / "run_record.json"), "finished_epoch": time.time()}


def collect(args):
    output = args.output_root
    host_network = getattr(args, "host_network", False)
    mode = {"host_network": True} if host_network else {}
    with original._output_lock(output, ".collector.lock"):
        state = prepare(output, **mode)
        runtime_preflight(output, **mode)
        if probe(output, args.auth, args.models)["status"] != "PASS":
            return 2
        keys, agents = credentials(args.auth, args.models), profiles(host_network)
        _, loaded, manifests = original._load_inputs()
        for slot in state["slots"]:
            if slot["model_id"] in args.models and (slot["status"] == "RUNNING" or
                    (args.retry_infrastructure and slot["status"] == "INFRASTRUCTURE_INVALID")):
                original._reset_slot(slot, "interrupted collection or explicit infrastructure retry")
        save(output, state)
        pending = [s for s in state["slots"] if s["status"] == "PENDING" and s["model_id"] in args.models]
        if args.max_slots is not None:
            pending = pending[:args.max_slots]
        stopping = False
        def stop(signum, frame):
            nonlocal stopping
            stopping = True
            print("Stop requested; finishing current cases.", flush=True)
        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                remaining = deque(pending)
                futures = {}
                def fill_workers():
                    while not stopping and len(futures) < args.workers:
                        if not remaining:
                            break
                        slot = remaining.popleft()
                        slot.update(status="RUNNING", started_epoch=time.time())
                        save(output, state)
                        futures[pool.submit(run_slot, output, dict(slot), loaded, manifests,
                            agents[slot["model_id"]], keys[slot["model_id"]], state["experiment_id"])] = slot
                fill_workers()
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        slot = futures.pop(future)
                        try:
                            slot.update(future.result())
                        except Exception as exc:
                            detail = str(exc)
                            for key in keys.values():
                                detail = detail.replace(key, "[REDACTED]")
                            slot.update(status="INFRASTRUCTURE_INVALID", failure_reason=detail, finished_epoch=time.time())
                        save(output, state)
                        print(json.dumps({"slot_id": slot["slot_id"], "status": slot["status"],
                                          "counts": state["status_counts"]}), flush=True)
                        if slot["status"] == "INFRASTRUCTURE_INVALID":
                            reason = str(slot.get("failure_reason", "")).lower()
                            account_failure = any(token in reason for token in (
                                "http 401", "http 402", "http 403", "invalid_api_key",
                                "insufficient_quota", "insufficient_balance", "arrearage"))
                            if account_failure:
                                stopping = True
                            elif not stopping and len(slot.get("attempt_history", [])) < getattr(args, "max_infrastructure_retries", 1):
                                original._reset_slot(slot, "bounded automatic infrastructure retry")
                                remaining.append(slot)
                                save(output, state)
                            # An exhausted per-case retry does not stop unrelated cases.
                    fill_workers()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        selected = [s for s in state["slots"] if s["model_id"] in args.models]
        return 0 if all(s["status"] == "COMPLETED" for s in selected) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "prepare", "preflight", "collect", "status"))
    parser.add_argument("--auth", type=Path, default=ROOT / "auth.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--host-network", action="store_true", help="use existing no-admin PRoot runtime without network namespaces")
    parser.add_argument("--model", dest="models", action="append", choices=MODELS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-slots", type=int, help="limit this invocation; N remains 1")
    parser.add_argument("--retry-infrastructure", action="store_true")
    parser.add_argument("--max-infrastructure-retries", type=int, default=1,
                        help="automatic retries per logical case; preserve every failed attempt")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or (args.max_slots is not None and args.max_slots < 1):
        parser.error("workers must be 1..8 and max-slots must be positive")
    if not 0 <= args.max_infrastructure_retries <= 3:
        parser.error("max-infrastructure-retries must be 0..3")
    args.output_root = (args.output_root or (HOST_NETWORK_OUTPUT if args.host_network else DEFAULT_OUTPUT)).resolve()
    args.models = tuple(dict.fromkeys(args.models or MODELS))
    require_benchmark_runtime()
    if args.command == "probe":
        return 0 if probe(args.output_root, args.auth, args.models)["status"] == "PASS" else 2
    if args.command == "collect":
        return collect(args)
    if args.command == "status":
        state = original._read(args.output_root / "collection_state.json")
        print(json.dumps({k: state[k] for k in ("collection_status", "total_slots", "status_counts", "by_model")}, indent=2))
        return 0
    with original._output_lock(args.output_root, ".collector.lock"):
        state = prepare(args.output_root, host_network=args.host_network)
        if args.command == "preflight":
            credentials(args.auth, args.models)
            runtime_preflight(args.output_root, host_network=args.host_network)
        print(json.dumps({"status": "READY", "total_slots": state["total_slots"], "provider_calls": 0}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
