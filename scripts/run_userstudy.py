#!/usr/bin/env python3
"""Run the manifest's cases through Luna and Terra APIs, with metric audits for N=3.

Each independent round uses new collection state, answers and trajectories.
Selective retries complete that round; low scores are retained. Infrastructure
or unresolved scientific issues stop advancement and remain resumable.
The optional collect_userstudy_with_fallback.py launcher can collect Luna
rounds ahead of Terra during provider rate limits; these audits still require
every case and never treat collected answers alone as complete results.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.repeated_experiment import file_hash, read_json, run_rounds, run_rounds_parallel, validate_round
from flowintentbench.runtime_environment import ensure_frozen_environment
from scripts import run_full_dataset_n1_experiment as canonical
from flowintentbench.experiment_scope import case_inventory
from scripts.run_full_dataset_n1_completion import run_completion

MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")


def preflight(manifest_path: Path, configs: dict, continuation: Path, *, host_network: bool = True) -> dict:
    issues = []
    checks = {}
    try:
        checks["cases"] = canonical.validate_manifest(manifest_path, run_tests=False)
    except Exception as exc:
        issues.append(f"case inputs: {exc}")
    try:
        environment = ensure_frozen_environment()
        checks["python"] = environment.fingerprint
    except Exception as exc:
        issues.append(f"frozen Python environment: {exc}")
    if host_network:
        try:
            from flowintentbench.loader import CaseLoader
            from flowintentbench.python_runtime import PythonExecutionEnvironment
            from flowintentbench.manifest import DatasetManifest
            row = read_json(manifest_path)["cases"][0]
            loaded = CaseLoader(ROOT / "datasets").load(ROOT / row["case_input_path"])
            data_manifest = DatasetManifest.model_validate_json((ROOT / "datasets" / row["dataset_id"] / "dataset_manifest.json").read_text())
            with PythonExecutionEnvironment(loaded, manifest=data_manifest, runtime_profile="flow-python-host-network-v1") as runtime:
                smoke = runtime.execute(f"import pathlib, numpy, scipy, vtk; assert pathlib.Path('/case/case_input.json').is_file(); assert not pathlib.Path({str(ROOT / 'auth.json')!r}).exists(); print('READY')")
                checks["sandbox"] = {**runtime.runtime_metadata, "smoke_success": smoke.success, "diagnostic": smoke.stderr}
                if not smoke.success:
                    issues.append(f"host-network Python startup: {smoke.exception}")
        except Exception as exc:
            issues.append(f"host-network runtime: {exc}")
    else:
        bwrap = shutil.which("bwrap")
        if bwrap:
            try:
                p = subprocess.run([bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid",
                                    "--unshare-ipc", "--unshare-uts", "--ro-bind", "/", "/",
                                    "--", "/bin/true"], capture_output=True, text=True, timeout=15)
                checks["sandbox"] = {"returncode": p.returncode, "diagnostic": p.stderr.strip()}
                if p.returncode:
                    issues.append("required bubblewrap isolation unavailable: " + p.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                issues.append(f"sandbox probe: {exc}")
        else:
            issues.append("bubblewrap is not installed")
    for model in MODELS:
        try:
            profile = canonical.load_agent_profile(model + "-xhigh-chat" + ("-host-network" if host_network else ""), repository_root=ROOT)
            checks[model] = canonical._runtime_environment_preflight(
                tested_profile=profile, evaluator_config=read_json(configs[model]),
                continuation_spec=read_json(continuation))
        except Exception as exc:
            issues.append(f"{model} API configuration: {exc}")
    return {"status": "READY" if not issues else "BLOCKED_ENVIRONMENT",
            "provider_calls": 0, "checks": checks, "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/userstudy/case_manifest.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/userstudy")
    parser.add_argument("--luna-evaluator-config", type=Path, default=ROOT / "config/evaluator_for_luna.json")
    parser.add_argument("--terra-evaluator-config", type=Path, default=ROOT / "config/evaluator_for_terra.json")
    parser.add_argument("--network-mode", choices=("host", "isolated"), default="host")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--completion-passes", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel-models", action="store_true",
                        help="Run Luna and Terra concurrently within each round; retain the full-metric round barrier")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.completion_passes < 1:
        parser.error("rounds and completion-passes must be positive")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Hold the lock for this invocation, including all collection children.
    # A second launcher must not spend tokens in the same observation slots.
    study_lock = (root / ".run.lock").open("a")
    try:
        fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("this study is already running; monitor its status instead of launching it again")
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    case_count = len(case_inventory(manifest))
    host_network = args.network_mode == "host"
    def agent_id(model):
        return model + "-xhigh-chat" + ("-host-network" if host_network else "")
    configs = {MODELS[0]: args.luna_evaluator_config.resolve(), MODELS[1]: args.terra_evaluator_config.resolve()}
    continuation = ROOT / "config/production_continuation_handlers.json"
    identity = {"schema_version": "userstudy-repeated-v1", "N": args.rounds,
                "models": list(MODELS), "cases_per_model_round": case_count, "formal": False,
                "network_mode": args.network_mode,
                "runtime_profile": "flow-python-host-network-v1" if host_network else "flow-python-v1",
                "case_manifest_sha256": file_hash(manifest_path),
                "configs": {m: {"path": str(p), "sha256": file_hash(p)} for m, p in configs.items()},
                "profiles": {m: canonical.load_agent_profile(agent_id(m), repository_root=ROOT).profile_sha256 for m in MODELS},
                "continuation_sha256": file_hash(continuation),
                "policy": f"audit both models' {case_count} cases before the next round audit; independently collected ahead-of-order rounds remain subject to the same full metric checks"}
    identity_path = root / "study_manifest.json"
    if identity_path.is_file() and read_json(identity_path) != identity:
        if any(root.glob("round-*")):
            raise RuntimeError("existing study has different inputs/models/configs; choose a new output root")
        canonical._atomic_json(root / "preflight_history" / (file_hash(identity_path) + ".json"), read_json(identity_path))
    if not args.resume and any(root.glob("round-*")):
        raise RuntimeError("existing rounds require --resume")
    canonical._atomic_json(identity_path, identity)
    report = preflight(manifest_path, configs, continuation, host_network=host_network)
    canonical._atomic_json(root / "environment_preflight.json", report)
    if report["status"] != "READY" or args.preflight_only:
        status = {"status": report["status"], "N": args.rounds, "models": list(MODELS),
                  "completed_model_rounds": 0, "issues": report["issues"],
                  "requested_observations": case_count * len(MODELS) * args.rounds,
                  "new_provider_calls": 0}
        canonical._atomic_json(root / "last_invocation.json", status)
        if not any(root.glob("round-*")):
            canonical._atomic_json(root / "study_status.json", status)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "READY" else 2
    # One full test suite for all six model-rounds; each child verifies the
    # exact code/input binding before reusing the evidence.
    suite_path = root / "suite_preflight.json"
    suite = None
    if suite_path.is_file():
        try:
            candidate = read_json(suite_path)
            canonical._assert_preflight_binding(candidate, manifest_file=manifest_path, repository_root=ROOT)
            if ("not sandbox" in candidate["frozen_tests"]["command"]) == host_network:
                suite = candidate
        except (RuntimeError, ValueError, KeyError):
            pass
    if suite is None:
        suite = canonical.validate_manifest(manifest_path, run_tests=True, host_network_tests=host_network)
    canonical._atomic_json(suite_path, suite)

    def execute(number: int, model: str) -> dict:
        folder = root / f"round-{number:02d}" / model.removeprefix("gpt-5.6-")
        initial, completed = folder / "initial", folder / "completed"
        if (completed / "completion_manifest.json").is_file():
            audited = validate_round(completed, manifest, model, host_network=host_network)
            if audited["status"] == "FULL_METRIC_COMPLETE":
                return audited
        else:
            audited = validate_round(initial, manifest, model, host_network=host_network)
            if audited["status"] == "FULL_METRIC_COMPLETE":
                return audited
            command = [sys.executable, str(ROOT / "scripts/run_full_dataset_n1_experiment.py"),
                       "--agent", agent_id(model), "--manifest", str(manifest_path),
                       "--evaluator-config", str(configs[model]), "--output-root", str(initial),
                       "--preflight-report", str(suite_path)]
            if (initial / "collection/collection_state.json").is_file():
                command.append("--resume")
                state = read_json(initial / "collection/collection_state.json")
                if state.get("completed_count") == case_count and {o.get("case_id") for o in state.get("observations", [])} == {r["case_id"] for r in manifest["cases"]}:
                    # The canonical evaluator still validates every original
                    # RunRecord and exact model/input bindings. An evaluator
                    # repair must not reopen an already-complete collector or
                    # rewrite its immutable code/environment snapshot.
                    command.append("--evaluation-only")
            code = canonical._run_command(command, log_path=folder / "initial.log")
            audited = validate_round(initial, manifest, model, host_network=host_network)
            if audited["status"] == "FULL_METRIC_COMPLETE":
                return audited
            if not (initial / "report/metric_matrix.csv").is_file():
                audited["issues"].append(f"initial script exited {code}; inspect initial.log and resume after repair")
                return audited
        run_completion(source_root=initial, output_root=completed,
                       manifest_path=manifest_path, agent=agent_id(model),
                       evaluator_config_path=configs[model], continuation_path=continuation,
                       max_rounds=args.completion_passes,
                       resume=(completed / "completion_manifest.json").is_file(),
                       evaluation_only=False, strict_metrics=True)
        return validate_round(completed, manifest, model, host_network=host_network)

    def persist(status: dict) -> None:
        canonical._atomic_json(root / "study_status.json", status)
        print(json.dumps({"status": status["status"], "completed_model_rounds": status["completed_model_rounds"]}), flush=True)

    runner = run_rounds_parallel if args.parallel_models else run_rounds
    status = runner(models=list(MODELS), rounds=args.rounds, expected_case_count=case_count, execute=execute, persist=persist)
    if status["status"] == "COMPLETE":
        # Preserve all six bundles as independent per-round results; aggregate
        # only after every slot passes, without labeling the pilot FORMAL.
        bundles = [{"round": r["round"], "model": r["model"],
                    "metrics": read_json(Path(r["output_root"]) / "metrics.json")}
                   for r in status["results"]]
        canonical._atomic_json(root / "results.json", {"N": args.rounds, "formal": False, "results": bundles})
    return 0 if status["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
