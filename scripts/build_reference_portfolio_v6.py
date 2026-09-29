#!/usr/bin/env python3
"""Build the v6 model-visible semantic-alignment reference snapshot.

Construction only: this command never calls an evaluated model, evaluator
model, or curator.  It compiles the explicit candidate matrix into fresh case
artifacts and freezes those artifacts under a new write-once authority.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.qualified_scientific_cases import build_agent_ready_scientific_case_portfolio
from flowintentbench.reference_freeze import build_reference_portfolio_freeze_v6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--conditions-manifest", type=Path, default=None)
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    conditions = (args.conditions_manifest or root / "outputs/current/audits/scientific_question_candidate_matrix_v3.json").resolve()
    staging = (args.staging_root or root / "outputs/staging/reference_portfolio_v6").resolve()
    staging.mkdir(parents=True, exist_ok=True)
    build_agent_ready_scientific_case_portfolio(
        root,
        staging,
        conditions_manifest=conditions,
    )
    manifest_path = (
        args.output.resolve()
        if args.output is not None
        else root / "artifacts/archive/reference/portfolio_v6_final/reference_science_baseline_manifest.json"
    )
    manifest = build_reference_portfolio_freeze_v6(
        root,
        manifest_path,
        source_case_root=staging / "agent_ready_cases",
    )
    status = "PASS" if manifest.get("reference_case_gt_frozen_count") == manifest.get("total_case_count") else "BLOCKED"
    print(f"REFERENCE_V6_STATUS={status}")
    print(f"TOTAL_CASES={manifest.get('total_case_count', 0)}")
    print(f"REFERENCE_CASE_GT_FROZEN={manifest.get('reference_case_gt_frozen_count', 0)}")
    print(f"CANONICAL_IDENTITIES={manifest.get('canonical_identity_frozen_count', 0)}")
    print(f"PROVISIONAL_IDENTITIES={manifest.get('provisional_identity_count', 0)}")
    print(f"STAGING_ROOT={staging}")
    print(f"MANIFEST={manifest_path}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
