#!/usr/bin/env python3
"""Run one isolated formal-release closure iteration for the frozen 28 slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.release_closure import run_release_closure_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/current/formal_release_closure")
    parser.add_argument(
        "--selected-case-id",
        action="append",
        dest="selected_case_ids",
        help="explicit case_id/slot_id to include in the sparse release denominator; repeatable",
    )
    args = parser.parse_args()
    report = run_release_closure_loop(
        ROOT,
        args.output_root,
        iteration=args.iteration,
        selected_case_ids=args.selected_case_ids,
    )
    print(json.dumps({
        "iteration": report["iteration"],
        "TOTAL_CASES": report["TOTAL_CASES"],
        "CANDIDATE_TOTAL": report["CANDIDATE_TOTAL"],
        "SELECTED_RELEASE_CASE_COUNT": report["SELECTED_RELEASE_CASE_COUNT"],
        "CONTROLLED_PILOT_ONLY": report["CONTROLLED_PILOT_ONLY"],
        "SCIENTIFIC_CASE_IDENTITY_FROZEN": report["SCIENTIFIC_CASE_IDENTITY_FROZEN"],
        "TOTAL_PENDING_CASE_IDENTITY": report["TOTAL_PENDING_CASE_IDENTITY"],
        "TOTAL_GT_SCHEMA_VALID": report["TOTAL_GT_SCHEMA_VALID"],
        "TOTAL_DATA_MEDIATED_PROOF": report["TOTAL_DATA_MEDIATED_PROOF"],
        "TOTAL_EVIDENCE_CLOSURE_COMPLETE": report["TOTAL_EVIDENCE_CLOSURE_COMPLETE"],
        "TOTAL_PENDING_SRAC": report["TOTAL_PENDING_SRAC"],
        "TOTAL_GT_SEMANTIC_BOUND": report["TOTAL_GT_SEMANTIC_BOUND"],
        "TOTAL_CURATOR_CONFIRMED": report["TOTAL_CURATOR_CONFIRMED"],
        "TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE": report["TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE"],
        "TOTAL_SCIENTIFIC_GROUNDING_CONFIRMED": report["TOTAL_SCIENTIFIC_GROUNDING_CONFIRMED"],
        "TOTAL_PENDING_SCIENTIFIC_GROUNDING": report["TOTAL_PENDING_SCIENTIFIC_GROUNDING"],
        "TOTAL_RELEASE_AUTHORIZED": report["TOTAL_RELEASE_AUTHORIZED"],
        "TOTAL_PENDING_RELEASE_AUTHORIZATION": report["TOTAL_PENDING_RELEASE_AUTHORIZATION"],
        "OFFICIAL_RELEASE_READY": report["OFFICIAL_RELEASE_READY"],
        "FORMAL_RELEASE_ELIGIBLE": report["FORMAL_RELEASE_ELIGIBLE"],
        "target_status": report["target_status"],
        "target_status_scope": report["target_status_scope"],
        "formal_release_authority": report["formal_release_authority"],
        "blocker_counts": report["blocker_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
