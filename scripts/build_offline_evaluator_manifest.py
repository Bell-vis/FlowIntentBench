#!/usr/bin/env python3
"""Materialize the deterministic evaluator identity without a model call."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluator import EvaluatorComponentIdentity, save_evaluation_manifest
from flowintentbench.evaluator_backend import (
    OpenAICompatibleEvaluatorBackend,
    StructuredEvaluatorBackend,
)
from flowintentbench.model_runner import EvaluationTarget
from flowintentbench.pending_adjudication import load_pending_continuation_registry
from flowintentbench.runtime_config import resolve_provider_configuration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/userstudy/evaluator_manifest.json")
    parser.add_argument("--server-config", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--model-config-json")
    parser.add_argument("--pending-handlers-json", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--transport-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.server_config or args.provider or args.model:
        config = json.loads(args.model_config_json or "{}")
        if not isinstance(config, dict):
            parser.error("--model-config-json must contain an object")
        runtime = resolve_provider_configuration(
            role="evaluator",
            config_path=args.server_config,
            provider=args.provider,
            model_id=args.model,
            model_configuration=config,
        )
        backend = OpenAICompatibleEvaluatorBackend(
            EvaluationTarget(
                runtime.provider, runtime.model_id, runtime.model_configuration
            ),
            api_key="manifest-only-no-request",
            base_url=runtime.base_url,
            timeout_seconds=args.timeout_seconds,
            max_transport_attempts=args.transport_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
        )
    else:
        backend = StructuredEvaluatorBackend(
            completion=lambda _operation, _payload: {"result": "MATCH"},
            identity=EvaluatorComponentIdentity(
                implementation_id="flowintentbench-structured-evaluator",
                version="offline-contract",
                provider="codex-hosted",
                model="evaluator-contract-only",
            ),
            max_format_repairs=1,
        )
    manifest = backend.manifest()
    if args.pending_handlers_json is not None:
        specifications = json.loads(
            args.pending_handlers_json.read_text(encoding="utf-8")
        )
        registry = load_pending_continuation_registry(specifications)
        if not registry.production_ready:
            parser.error(
                "pending handler registry is not production capable: "
                + json.dumps(registry.production_capability_issues)
            )
        manifest = replace(
            manifest,
            continuation_execution_manifest=registry.execution_manifest,
        )
    save_evaluation_manifest(manifest, args.output)
    print(args.output.resolve())
    print("MODEL_CALLS=0")
    print("EVALUATOR_MANIFEST_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
