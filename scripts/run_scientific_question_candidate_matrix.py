#!/usr/bin/env python3
"""Build or live-review the sparse seven-dataset question candidate matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from flowintentbench.scientific_question_candidate_matrix import (  # noqa: E402
    run_scientific_question_candidate_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the sparse 7-dataset x 4-condition scientific-question review "
            "matrix. Without --live this performs a deterministic build-only pass."
        )
    )
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    parser.add_argument(
        "--output-root",
        default=str(
            REPOSITORY_ROOT / "outputs/current/scientific_question_matrix"
        ),
        help=(
            "Source matrix artifact directory (defaults to "
            "outputs/current/scientific_question_matrix)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the repository-configured canonical Flow Expert for all advisory calls.",
    )
    parser.add_argument("--server-config")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-workers", type=int, default=7)
    parser.add_argument("--candidate-count", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--targeted-review-manifest",
        help=(
            "Override the Round-2 targeted-review manifest used only to display "
            "non-adopted Combustor target candidates."
        ),
    )
    parser.add_argument(
        "--resume-manifest",
        help=(
            "Reuse only strictly validated authoring/review traces whose semantic "
            "contract hash still matches, and run the configured reviewer for the "
            "remaining slots."
        ),
    )
    parser.add_argument(
        "--enforce-human-facing-acceptance",
        action="store_true",
        help=(
            "Apply the frozen deterministic HF responsibility audit when "
            "reusing or nominating final question wording."
        ),
    )
    parser.add_argument("--max-authoring-attempts", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run_scientific_question_candidate_matrix(
        args.repository_root,
        args.output_root,
        run_live=args.live,
        config_path=args.server_config,
        api_key=args.api_key,
        timeout=args.timeout,
        max_workers=args.max_workers,
        candidate_count=args.candidate_count,
        targeted_review_manifest_path=args.targeted_review_manifest,
        resume_manifest_path=args.resume_manifest,
        enforce_human_facing_acceptance=args.enforce_human_facing_acceptance,
        max_authoring_attempts=args.max_authoring_attempts,
    )
    summary_keys = (
        "execution_mode",
        "maximum_candidate_slots",
        "condition_slot_count",
        "TOTAL_CANDIDATES_PROPOSED",
        "TOTAL_CANDIDATES_FOR_HUMAN_REVIEW",
        "TOTAL_BLOCKED_PENDING_EVIDENCE",
        "TOTAL_BLOCKED_PENDING_O_SPACE_REVIEW",
        "TOTAL_NEW_CASE_PENDING_SELECTION",
        "TOTAL_OMITTED",
    )
    print(json.dumps({key: manifest[key] for key in summary_keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
