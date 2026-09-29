#!/usr/bin/env python3
"""Build the round-one human scientific-question quality review pack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.scientific_question_quality import (
    repair_incomplete_scientific_question_quality_fidelity,
    run_scientific_question_quality_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/experiments/scientific_question_quality_round1",
    )
    parser.add_argument(
        "--targeted-review-manifest",
        type=Path,
        default=ROOT
        / "outputs/experiments/scientific_question_quality_round1/targeted_reviews/manifest.json",
    )
    parser.add_argument(
        "--generic-review-manifest",
        type=Path,
        default=ROOT
        / "outputs/experiments/scientific_semantic_optimization/manifest.json",
    )
    parser.add_argument("--server-config", type=Path, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument(
        "--resume-incomplete-fidelity",
        action="store_true",
        help="Retry only incomplete saved candidate fidelity audits.",
    )
    args = parser.parse_args()
    try:
        if args.resume_incomplete_fidelity:
            result = repair_incomplete_scientific_question_quality_fidelity(
                args.repository_root,
                args.output_root,
                config_path=args.server_config,
                api_key=args.api_key,
                timeout=args.timeout,
                max_workers=args.max_workers,
                retry_attempts=args.retry_attempts,
            )
        else:
            result = run_scientific_question_quality_review(
                args.repository_root,
                args.output_root,
                targeted_review_manifest=args.targeted_review_manifest,
                generic_review_manifest=args.generic_review_manifest,
                run_live=True,
                config_path=args.server_config,
                api_key=args.api_key,
                timeout=args.timeout,
                max_workers=args.max_workers,
                retry_attempts=args.retry_attempts,
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
    return 0 if (
        result["status"]["STATIC_IMPLEMENTATION_STATUS"] == "PASS"
        and result["status"]["HUMAN_PRESENTATION_GENERATION_STATUS"] == "COMPLETE"
        and result["status"]["POST_REWRITE_SEMANTIC_FIDELITY_STATUS"]
        in {"PASS", "REVISIONS_OBSERVED"}
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
