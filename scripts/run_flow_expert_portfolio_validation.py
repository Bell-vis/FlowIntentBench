#!/usr/bin/env python3
"""Build or run the canonical Flow Expert validation exercises."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.flow_expert_portfolio_validation import (
    run_flow_expert_portfolio_validation,
)
from flowintentbench.flow_expert_sensitivity_hardening import (
    run_flow_expert_sensitivity_hardening,
)
from flowintentbench.scientific_expert_review import run_scientific_expert_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/experiments/flow_expert_portfolio_validation",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Invoke the configured canonical tool-free Flow Expert.",
    )
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--hardening",
        action="store_true",
        help=(
            "Run the unified responsibility/evidence sensitivity, cross-family, "
            "repeatability, and calibration-preparation exercise."
        ),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.hardening:
            def reviewer(packet, item_id):
                return run_scientific_expert_review(
                    args.repository_root,
                    packet,
                    config_path=args.server_config,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    run_id=item_id,
                )

            result = run_flow_expert_sensitivity_hardening(
                args.repository_root,
                args.output_root,
                reviewer=reviewer if args.live else None,
                max_workers=args.max_workers,
            )
        else:
            result = run_flow_expert_portfolio_validation(
                args.repository_root,
                args.output_root,
                run_live=args.live,
                config_path=args.server_config,
                api_key=args.api_key,
                timeout=args.timeout,
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
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.hardening:
        readiness = result["readiness"]
        summary = result["task_summary"]
        engineering_ready = readiness["FLOW_EXPERT_ENGINEERING_READY"] is True
        all_calls_completed = (
            not args.live
            or summary["completed_task_count"] == summary["planned_callable_task_count"]
        )
        live_execution_complete = (
            not args.live
            or readiness.get("FLOW_EXPERT_LIVE_EXECUTION_STATUS") == "COMPLETE"
        )
        return 0 if engineering_ready and all_calls_completed and live_execution_complete else 1
    if args.live:
        return 0 if result["cross_family_validation_status"] == "PASS" else 1
    return 0 if result["packet_construction_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
