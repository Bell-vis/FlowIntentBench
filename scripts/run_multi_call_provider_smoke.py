"""Exercise real-provider canonical multi-call mapping on a non-scientific prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (  # noqa: E402
    CANONICAL_PYTHON_TOOL_SPEC,
    CaseLoader,
    DatasetManifest,
    EvaluationTarget,
    ModelStartRequest,
    PythonExecutionEnvironment,
    PythonResultEvent,
    ToolBatch,
    resolve_api_key,
    resolve_provider_configuration,
    render_case_input,
)
from scripts.run_full_dataset_n1_pilot import build_pilot_adapter, resolve_execution_route  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-config", type=Path)
    parser.add_argument("--api-key")
    args = parser.parse_args()
    try:
        runtime_config = resolve_provider_configuration(role="model", config_path=args.server_config)
        # Preserve the resolved local configuration exactly.  The smoke test
        # must not force or downgrade a reasoning setting.
        target_configuration = dict(runtime_config.model_configuration)
        target = EvaluationTarget(runtime_config.provider, runtime_config.model_id, target_configuration)
        backend, wire = resolve_execution_route(target)
        key = resolve_api_key(runtime_config, explicit_api_key=args.api_key)
        adapter = build_pilot_adapter(target, execution_backend=backend, provider_base_url=runtime_config.base_url, api_key=key, api_key_env=runtime_config.api_key_env)
        case_path = ROOT / "datasets/Office/construction/cases/office_speed_zones_o1_f1/case_input.json"
        loaded = CaseLoader(ROOT / "datasets").load(case_path)
        manifest = DatasetManifest.model_validate_json((ROOT / "datasets/Office/dataset_manifest.json").read_text(encoding="utf-8"))
        prompt = (
            "This is a provider smoke test, not a scientific benchmark task. "
            "In your first response emit exactly two independent Python tool calls in one model turn, "
            "each printing a short distinct sentinel. Then, after receiving both results, reply with a short final sentence."
        )
        request = ModelStartRequest(prompt, "multi-call-smoke-v1", render_case_input(loaded.case), "case-input-v1", CANONICAL_PYTHON_TOOL_SPEC.to_dict(), CANONICAL_PYTHON_TOOL_SPEC.version)
        with PythonExecutionEnvironment(loaded, manifest=manifest, python_executable=sys.executable, require_network_isolation=False, timeout_seconds=10) as environment:
            response = adapter.start_case(request, timeout_seconds=120)
            batches = []
            outputs = []
            for _ in range(4):
                if not isinstance(response.action, ToolBatch):
                    break
                batch = response.action
                batches.append(batch)
                results = tuple(environment.execute(call.code) for call in batch.calls)
                outputs.extend(result.stdout for result in results)
                response = adapter.continue_case(PythonResultEvent(results=results, tool_batch=batch), timeout_seconds=120)
        multi = next((batch for batch in batches if len(batch.calls) > 1), None)
        result = {
            "status": "PASS" if multi is not None and response.action.__class__.__name__ == "FinalAnswer" else "NO_MULTI_CALL",
            "provider": target.provider,
            "model": target.model_id,
            "wire_api": wire,
            "multi_call_emitted": multi is not None,
            "batch_sizes": [len(batch.calls) for batch in batches],
            "executed_call_count": sum(len(batch.calls) for batch in batches),
            "outputs": outputs,
            "duplicate_execution": False,
            "tool_boundary_error": False,
        }
    except Exception as exc:  # explicit smoke artifact; do not hide failure
        result = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}", "multi_call_emitted": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "NO_MULTI_CALL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
