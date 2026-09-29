#!/usr/bin/env python3
"""Run the frozen 28-question human-facing contract closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.human_facing_closure import (  # noqa: E402
    run_human_facing_question_closure,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    default_matrix = ROOT / "outputs/current/audits/scientific_question_candidate_matrix_v3.json"
    if not default_matrix.is_file():
        default_matrix = ROOT / "outputs/current/scientific_question_matrix/scientific_question_manifest.json"
    parser.add_argument(
        "--source-matrix",
        type=Path,
        default=default_matrix,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts/review/current",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run 28 fresh final-wording reviews with the repository-configured Flow Expert.",
    )
    parser.add_argument("--server-config")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-workers", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_human_facing_question_closure(
        args.repository_root,
        args.source_matrix,
        args.output_root,
        run_live=args.live,
        config_path=args.server_config,
        api_key=args.api_key,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    keys = (
        "HUMAN_FACING_CLOSURE_STATUS",
        "HUMAN_FACING_ACCEPTANCE_CONTRACT_SHA256",
        "TOTAL_QUESTION_SLOTS",
        "TOTAL_GENUINE_AUTHORED_PRIMARY",
        "TOTAL_DETERMINISTIC_PRIMARY_FALLBACK",
        "TOTAL_TARGET_CLARITY_PASS",
        "TOTAL_RESPONSIBILITY_INTEGRITY_PASS",
        "TOTAL_VISIBILITY_INTEGRITY_PASS",
        "TOTAL_SEMANTIC_FIDELITY_PASS",
        "TOTAL_HCCQ_PASS",
        "TOTAL_FLOW_EXPERT_FINAL_REVIEW_PASS",
        "TOTAL_HUMAN_INTERFACE_AUDITED",
        "TOTAL_HUMAN_INTERFACE_PASS",
        "TOTAL_HUMAN_INTERFACE_REVISE",
        "TOTAL_HUMAN_INTERFACE_BLOCKED",
        "TOTAL_HUMAN_FRIENDLY",
        "TOTAL_FAMILY_CONDITION_INTEGRITY_PASS",
        "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS",
        "TOTAL_HUMAN_REVIEW_READY",
        "TOTAL_RELEASE_READY",
        "TOTAL_NEW_TARGET_REQUIRED",
        "TOTAL_PENDING_EVIDENCE",
        "TOTAL_PENDING_O_SPACE",
        "TOTAL_PENDING_SRAC",
        "TOTAL_PROVISIONAL_TARGET",
    )
    payload = {key: result.get(key) for key in keys}
    payload.update(
        {
            "VALIDATOR_SEMANTIC_CHALLENGE_FALSE_ACCEPTS": result[
                "validator_robustness"
            ]["VALIDATOR_SEMANTIC_CHALLENGE_FALSE_ACCEPTS"],
            "VALIDATOR_PRESERVING_CHALLENGE_FALSE_REJECTS": result[
                "validator_robustness"
            ]["VALIDATOR_PRESERVING_CHALLENGE_FALSE_REJECTS"],
            "CROSS_DATASET_TEMPLATE_AUDIT": result.get(
                "portfolio_presentation_audit", {}
            ).get("cross_dataset_scientific_template_audit", {}),
        }
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["HUMAN_FACING_CLOSURE_STATUS"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
