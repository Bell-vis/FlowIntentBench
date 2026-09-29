#!/usr/bin/env python3
"""Run the read-only evaluation readiness gate.

This command only consumes frozen artifacts and existing saved records.  It
never creates a model client and never invokes either the evaluated model or
an evaluator model.  The optional v3 reference check is reported alongside
the v2-compatible readiness audit because the latter remains the compatibility
input for historical saved-run replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluation_readiness import (  # noqa: E402
    audit_fresh_n1_model_test_readiness,
    audit_offline_evaluation_readiness,
    save_offline_evaluation_readiness,
)
from flowintentbench.pending_adjudication import load_pending_continuation_registry  # noqa: E402
from flowintentbench.reference_freeze import (  # noqa: E402
    verify_reference_portfolio_freeze_v3,
)


def run(args: argparse.Namespace) -> dict:
    if getattr(args, "fresh_n1", False):
        if args.reference_freeze is None or args.n1_manifest is None:
            raise ValueError("--fresh-n1 requires --reference-freeze and --n1-manifest")
        return audit_fresh_n1_model_test_readiness(
            args.repository_root,
            reference_manifest=args.reference_freeze,
            n1_manifest=args.n1_manifest,
            evaluator_manifest=args.evaluator_manifest,
            pending_continuation_registry=(
                load_pending_continuation_registry(json.loads(args.pending_handlers_json.read_text(encoding="utf-8")))
                if getattr(args, "pending_handlers_json", None) is not None
                else None
            ),
            test_report=getattr(args, "test_report", None),
        )
    report = audit_offline_evaluation_readiness(
        args.repository_root,
        reference_freeze_manifest=args.reference_freeze,
        saved_run_root=args.saved_run_root,
        collected_case_manifest=args.collected_case_manifest,
        candidate_case_manifest=args.candidate_case_manifest,
        existing_evaluation_root=args.existing_evaluation_root,
    )
    report = dict(report)
    report["no_model_calls"] = True
    if args.reference_v3 is not None:
        try:
            report["checks"] = dict(report.get("checks", {}))
            report["checks"]["reference_portfolio_v3"] = verify_reference_portfolio_freeze_v3(
                args.repository_root,
                args.reference_v3,
            )
        except Exception as exc:
            report["checks"] = dict(report.get("checks", {}))
            report["checks"]["reference_portfolio_v3"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["status"] = "BLOCKED"
            report["scientific_metrics_computable"] = False
            report["blockers"] = list(report.get("blockers", ())) + [
                "REFERENCE_PORTFOLIO_V3_INVALID"
            ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--reference-freeze", type=Path, default=None)
    parser.add_argument("--reference-v3", type=Path, default=None)
    parser.add_argument("--saved-run-root", type=Path, default=None)
    parser.add_argument("--collected-case-manifest", type=Path, default=None)
    parser.add_argument("--candidate-case-manifest", type=Path, default=None)
    parser.add_argument("--existing-evaluation-root", type=Path, default=None)
    parser.add_argument(
        "--fresh-n1",
        action="store_true",
        help="run the strict no-model gate for a new reference-bound N=1 test",
    )
    parser.add_argument("--n1-manifest", type=Path, default=None)
    parser.add_argument("--evaluator-manifest", type=Path, default=None)
    parser.add_argument(
        "--pending-handlers-json",
        type=Path,
        default=None,
        help="optional JSON owner-to-module:function configuration for identity preflight",
    )
    parser.add_argument(
        "--test-report",
        type=Path,
        default=None,
        help="optional successful no-model test report to bind to the readiness artifact",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is not None:
        save_offline_evaluation_readiness(report, args.output)
    print(
        json.dumps(
            (
                {
                    "status": report["status"],
                    "no_model_calls": report["no_model_calls"],
                    "scientific_metrics_computable": report["scientific_metrics_computable"],
                    "metric_bearing_evaluation_status": report.get(
                        "metric_bearing_evaluation_status"
                    ),
                    "metric_bearing_scientific_metrics_computable": report.get(
                        "metric_bearing_scientific_metrics_computable"
                    ),
                    "metric_bearing_evaluation_blockers": report.get(
                        "metric_bearing_evaluation_blockers", []
                    ),
                    "model_test_not_run": report.get("model_test_not_run", False),
                    "blockers": report["blockers"],
                }
                if getattr(args, "fresh_n1", False)
                else {
                    "status": report["status"],
                    "no_model_calls": report["no_model_calls"],
                    "scientific_metrics_computable": report["scientific_metrics_computable"],
                    "current_metrics_available": report["current_metrics_available"],
                    "current_metrics_are_complete_28_case_results": report[
                        "current_metrics_are_complete_28_case_results"
                    ],
                    "blockers": report["blockers"],
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    # Fresh readiness has its own success label; map it to a successful shell
    # exit as well as the historical replay label.
    return 0 if report["status"] in {"PASS", "READY_FOR_N1_MODEL_TEST"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
