#!/usr/bin/env python3
"""Run the semantic/presentation/GT-binding construction audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.scientific_semantic_optimization import (
    run_scientific_semantic_optimization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "outputs/experiments/scientific_semantic_optimization",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the canonical local Codex-hosted Flow Expert advisory audit.",
    )
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-workers", type=int, default=3)
    args = parser.parse_args()
    try:
        result = run_scientific_semantic_optimization(
            args.repository_root,
            args.output_root,
            run_live=args.live,
            config_path=args.server_config,
            api_key=args.api_key,
            timeout=args.timeout,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    print(json.dumps(result["status"], ensure_ascii=False, indent=2))
    required = (
        result["status"]["STATIC_IMPLEMENTATION_STATUS"] == "PASS"
        and result["status"]["SEMANTIC_PROJECTION_STATUS"] == "PASS"
        and result["status"]["PRESENTATION_SEMANTIC_FIDELITY_STATUS"] == "PASS"
    )
    if args.live:
        required = required and result["status"][
            "FLOW_EXPERT_LIVE_EXECUTION_STATUS"
        ] == "COMPLETE"
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
