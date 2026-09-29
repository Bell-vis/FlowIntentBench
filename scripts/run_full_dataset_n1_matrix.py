"""Run the explicit N=1 pilot for every selected server model target.

This is deliberately a thin orchestration layer.  It freezes a small,
identity-bound model plan before the first model call, then invokes the
existing single-target pilot sequentially.  It contains no scientific logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    AgentProfile,
    CANONICAL_PYTHON_TOOL_SPEC,
    EvaluationTarget,
    FinalAnswer,
    ModelStartRequest,
    PythonExecutionRequest,
    PythonExecutionResult,
    PythonResultEvent,
    ThirdPartyResponsesAdapter,
    ToolBatch,
    code_data_version,
    evaluation_target_fingerprint,
    load_server_provider_configuration,
    resolve_api_key,
    resolve_provider_configuration,
    load_agent_profile,
    effective_agent_config,
    reject_profile_identity_overrides,
)
from scripts.run_full_dataset_n1_pilot import (  # noqa: E402
    build_pilot_adapter,
    resolve_execution_route,
    run_full_dataset_n1_pilot,
)
from flowintentbench.experiment_scope import case_inventory  # noqa: E402


EXPERIMENT_TYPE = "full-dataset-n1-model-matrix-v1"
MANIFEST_PATH = ROOT / "experiments/userstudy/case_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_backend_smoke(*, target: EvaluationTarget, runtime: Any, api_key: str) -> dict[str, Any]:
    """Run one synthetic tool-turn before any benchmark case for this target."""

    backend, wire = resolve_execution_route(target)
    adapter = build_pilot_adapter(
        target,
        execution_backend=backend,
        provider_base_url=runtime.base_url,
        api_key=api_key,
        api_key_env=runtime.api_key_env,
    )
    request = ModelStartRequest(
        system_prompt=(
            "This is a synthetic backend contract smoke test, not a benchmark case. "
            "Use the supplied Python tool exactly once when asked, then answer the request."
        ),
        system_prompt_version="matrix-backend-smoke-v1",
        user_message="Use the Python tool exactly once to compute 6 * 7, then answer with only the integer result.",
        case_presentation_version="synthetic-v1",
        python_tool_spec=CANONICAL_PYTHON_TOOL_SPEC.to_dict(),
        python_tool_spec_version="python-tool-v1",
    )
    try:
        first = adapter.start_case(request, timeout_seconds=120.0)
        if not isinstance(first.action, (PythonExecutionRequest, ToolBatch)):
            raise RuntimeError(
                f"backend smoke expected a Python tool request, got {type(first.action).__name__}"
            )
        if isinstance(first.action, ToolBatch):
            results = tuple(
                PythonExecutionResult(
                    success=True,
                    stdout="42\n" if index == 0 else "",
                    stderr="",
                    exception=None,
                    duration_seconds=0.0001,
                    execution_index=index + 1,
                )
                for index, _call in enumerate(first.action.calls)
            )
            result_event = PythonResultEvent(results=results, tool_batch=first.action)
        else:
            result_event = PythonResultEvent(
                result=PythonExecutionResult(
                    success=True,
                    stdout="42\n",
                    stderr="",
                    exception=None,
                    duration_seconds=0.0001,
                    execution_index=1,
                )
            )
        second = adapter.continue_case(
            result_event,
            timeout_seconds=120.0,
        )
        if not isinstance(second.action, FinalAnswer):
            raise RuntimeError(
                f"backend smoke expected a final answer, got {type(second.action).__name__}"
            )
        if second.action.text.strip() != "42":
            raise RuntimeError(
                f"backend smoke returned an unexpected final answer: {second.action.text!r}"
            )
        return {
            "status": "PASS",
            "label": "BACKEND CONTRACT SMOKE TEST ONLY — NOT A BENCHMARK OBSERVATION",
            "provider": target.provider,
            "base_url": runtime.base_url,
            "model_id": target.model_id,
            "execution_backend": backend,
            "wire_api": wire,
            "input_tokens": (first.input_tokens or 0) + (second.input_tokens or 0),
            "output_tokens": (first.output_tokens or 0) + (second.output_tokens or 0),
        }
    finally:
        adapter.close()


def _plan_entry(runtime: Any) -> dict[str, Any]:
    target = EvaluationTarget(runtime.provider, runtime.model_id, runtime.model_configuration)
    backend, wire = resolve_execution_route(target)
    return {
        "provider": runtime.provider,
        "base_url": runtime.base_url,
        "model_id": runtime.model_id,
        "model_configuration": dict(runtime.model_configuration),
        "reasoning_effort": runtime.model_configuration.get("reasoning_effort"),
        "execution_backend": backend,
        "wire_api": wire,
        "target_fingerprint": evaluation_target_fingerprint(target),
        "api_key_env": runtime.api_key_env,
    }


def construct_model_plan(
    *,
    server_config: str | Path | None = None,
    output_root: str | Path,
    manifest_path: str | Path = MANIFEST_PATH,
    provider: str | None = None,
    base_url: str | None = None,
    model_id: str | None = None,
    model_configuration: Mapping[str, Any] | None = None,
    agent_profile: AgentProfile | None = None,
) -> dict[str, Any]:
    """Resolve the current server model and freeze it before collection."""

    root = Path(output_root).resolve()
    plan_path = root / "model_plan.json"
    manifest = Path(manifest_path).resolve()
    if agent_profile is not None:
        reject_profile_identity_overrides(
            provider=provider if provider != agent_profile.provider else None,
            model_id=model_id if model_id != agent_profile.model_id else None,
            model_configuration=model_configuration,
        )
        profile_configuration = agent_profile.model_configuration
        profile_configuration.update(model_configuration or {})
        model_configuration = profile_configuration
        provider = provider or agent_profile.provider
        model_id = model_id or agent_profile.model_id
    runtime = resolve_provider_configuration(
        role="model",
        config_path=server_config,
        provider=provider,
        base_url=base_url,
        model_id=model_id,
        model_configuration=model_configuration,
    )
    model = _plan_entry(runtime)
    if agent_profile is not None:
        model["agent_id"] = agent_profile.agent_id
        model["runtime_profile_id"] = agent_profile.runtime_profile_id
        model["protocol_version"] = "profiled-natural-output-v1"
        model["agent_profile_sha256"] = agent_profile.profile_sha256
        model["runtime_profile_sha256"] = agent_profile.runtime_profile.profile_sha256
    try:
        evaluator_runtime = load_server_provider_configuration(server_config, role="evaluator")
    except Exception:
        evaluator_runtime = None
    plan = {
        "experiment_type": EXPERIMENT_TYPE,
        "case_manifest_sha256": _sha256(manifest),
        "benchmark_code_data_version": code_data_version(ROOT),
        "models": [model],
        "evaluator": (
            None
            if evaluator_runtime is None
            else {
                "provider": evaluator_runtime.provider,
                "base_url": evaluator_runtime.base_url,
                "model_id": evaluator_runtime.model_id,
                "model_configuration": dict(evaluator_runtime.model_configuration),
                "reasoning_effort": evaluator_runtime.model_configuration.get("reasoning_effort"),
                "api_key_env": evaluator_runtime.api_key_env,
            }
        ),
        "evaluator_status": "CONFIGURED" if evaluator_runtime is not None else "UNCONFIGURED",
    }
    plan["plan_sha256"] = _canonical_digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    if plan_path.exists():
        existing = _read_json(plan_path)
        if existing != plan:
            raise RuntimeError("model_plan.json already exists and does not match current server configuration")
        return dict(existing)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def _validate_plan(plan: Mapping[str, Any], *, manifest_path: Path) -> None:
    if plan.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("unexpected model plan experiment_type")
    if plan.get("case_manifest_sha256") != _sha256(manifest_path):
        raise RuntimeError("model plan case manifest digest does not match current manifest")
    if plan.get("benchmark_code_data_version") != code_data_version(ROOT):
        raise RuntimeError("model plan benchmark code/data version does not match current source")
    models = plan.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("model plan must contain at least one model")
    fingerprints: set[str] = set()
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("model plan model entry must be an object")
        target = EvaluationTarget(
            str(model["provider"]), str(model["model_id"]), dict(model.get("model_configuration", {}))
        )
        if model.get("target_fingerprint") != evaluation_target_fingerprint(target):
            raise RuntimeError("model plan target fingerprint mismatch")
        resolve_execution_route(target)
        if target.model_id in fingerprints:
            raise ValueError("duplicate model target in model plan")
        fingerprints.add(target.model_id)


def run_matrix(
    *,
    plan_path: str | Path,
    output_root: str | Path,
    manifest_path: str | Path = MANIFEST_PATH,
    datasets_root: str | Path = ROOT / "datasets",
    api_key: str | None = None,
    api_key_env: str | None = None,
    resume: bool = False,
    infrastructure_attempt_cap: int = 3,
) -> dict[str, Any]:
    plan_file = Path(plan_path).resolve()
    matrix_root = Path(output_root).resolve()
    manifest = Path(manifest_path).resolve()
    plan = _read_json(plan_file)
    _validate_plan(plan, manifest_path=manifest)
    cases = case_inventory(_read_json(manifest), require_dataset=True)
    results: list[dict[str, Any]] = []
    for model in plan["models"]:
        model = dict(model)
        agent_profile = (
            load_agent_profile(str(model["agent_id"]), repository_root=ROOT)
            if model.get("agent_id")
            else None
        )
        target = EvaluationTarget(
            str(model["provider"]), str(model["model_id"]), dict(model["model_configuration"])
        )
        runtime = resolve_provider_configuration(
            provider=target.provider,
            base_url=str(model["base_url"]),
            model_id=target.model_id,
            model_configuration=target.model_configuration,
        )
        backend, wire = resolve_execution_route(target)
        target_root = matrix_root / str(model["target_fingerprint"])
        key = resolve_api_key(runtime, explicit_api_key=api_key)
        smoke = run_backend_smoke(
            target=target,
            runtime=runtime,
            api_key=key,
        )
        pilot = run_full_dataset_n1_pilot(
            manifest_path=manifest,
            datasets_root=datasets_root,
            output_root=target_root,
            target=target,
            adapter_factory=lambda evaluated_target, b=backend, r=runtime, k=key: build_pilot_adapter(
                evaluated_target,
                execution_backend=b,
                provider_base_url=r.base_url,
                api_key=k,
                api_key_env=r.api_key_env,
            ),
            provider_base_url=runtime.base_url,
            infrastructure_attempt_cap=infrastructure_attempt_cap,
            resume=resume,
            agent_profile=agent_profile,
        )
        results.append(
            {
                "provider": target.provider,
                "base_url": runtime.base_url,
                "model_id": target.model_id,
                "target_fingerprint": model["target_fingerprint"],
                "execution_backend": backend,
                "wire_api": wire,
                "output_root": str(target_root.relative_to(matrix_root)),
                "collection_status": pilot.get("collection_status"),
                "valid_observations": pilot.get("valid_observations"),
                "completed_count": pilot.get("completed_count"),
                "model_noncompletion_count": pilot.get("model_noncompletion_count"),
                "infrastructure_invalid_attempt_count": pilot.get("infrastructure_invalid_attempt_count"),
                "aborted_infrastructure_attempt_count": pilot.get("aborted_infrastructure_attempt_count"),
                "backend_smoke": smoke,
            }
        )
    manifest_out = {
        "record_type": "full_dataset_n1_model_matrix_manifest",
        "status": "PILOT / NOT FORMAL BENCHMARK RESULT",
        "experiment_type": EXPERIMENT_TYPE,
        "model_plan_path": str(plan_file.relative_to(matrix_root.parent)) if matrix_root.parent in plan_file.parents else str(plan_file),
        "model_plan_sha256": _sha256(plan_file),
        "case_manifest_sha256": plan["case_manifest_sha256"],
        "benchmark_code_data_version": plan["benchmark_code_data_version"],
        "case_count": len(cases),
        "dataset_count": len({row["dataset_id"] for row in cases.values()}),
        "condition_count": len({row["condition"] for row in cases.values()}),
        "N": 1,
        "formal": False,
        "model_count": len(results),
        "expected_observations": len(cases) * len(results),
        "models": results,
        "evaluator_status": plan.get("evaluator_status", "UNCONFIGURED"),
    }
    manifest_out["matrix_manifest_sha256"] = _canonical_digest(manifest_out)
    destination = matrix_root / "matrix_manifest.json"
    if destination.exists() and resume:
        raise FileExistsError("matrix_manifest.json already exists; target outputs are already complete or must be resumed per target")
    matrix_root.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model-config-json", default=None)
    parser.add_argument("--agent", required=True, help="agent profile ID or path")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--infrastructure-attempt-cap", type=int, default=3)
    args = parser.parse_args()
    profile = load_agent_profile(args.agent)
    config_for_validation = None
    if args.model_config_json is not None:
        config_for_validation = json.loads(args.model_config_json)
        if not isinstance(config_for_validation, dict):
            parser.error("--model-config-json must contain a JSON object")
    try:
        reject_profile_identity_overrides(
            provider=args.provider, model_id=args.model_id,
            model_configuration=config_for_validation,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"effective_agent_config": effective_agent_config(profile).to_dict()}, indent=2))
    if not args.plan.exists():
        config = (
            config_for_validation
        )
        if config is not None and not isinstance(config, dict):
            parser.error("--model-config-json must contain a JSON object")
        plan = construct_model_plan(
            server_config=args.server_config,
            output_root=args.plan.parent,
            manifest_path=args.manifest,
            provider=profile.provider,
            base_url=args.base_url,
            model_id=profile.model_id,
            model_configuration=config,
            agent_profile=profile,
        )
        print(json.dumps({"model_plan": str(args.plan), "plan_sha256": plan["plan_sha256"]}, indent=2))
    result = run_matrix(
        plan_path=args.plan,
        output_root=args.output_root,
        manifest_path=args.manifest,
        datasets_root=args.datasets_root,
        api_key=args.api_key,
        resume=args.resume,
        infrastructure_attempt_cap=args.infrastructure_attempt_cap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
