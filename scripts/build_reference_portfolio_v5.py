#!/usr/bin/env python3
"""Build the v5 immutable reference snapshot after semantic-closure fixes.

The script is construction-only: it never calls an evaluated model, an
evaluator model, or a curator.  It creates a fresh staging tree and then
freezes the resulting Case/GT/materialization artifacts under a new v5
authority, leaving all historical v1-v4 snapshots untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.qualified_scientific_cases import build_agent_ready_scientific_case_portfolio
from flowintentbench.reference_freeze import build_reference_portfolio_freeze_v5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--staging-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    staging = (args.staging_root or root / "outputs/staging/reference_portfolio_v5").resolve()
    staging.mkdir(parents=True, exist_ok=True)
    build_agent_ready_scientific_case_portfolio(root, staging)
    manifest = build_reference_portfolio_freeze_v5(
        root,
        args.output,
        source_case_root=staging / "agent_ready_cases",
    )
    status = "PASS" if manifest.get("reference_case_gt_frozen_count") == manifest.get("total_case_count") else "BLOCKED"
    print(f"REFERENCE_V5_STATUS={status}")
    print(f"TOTAL_CASES={manifest.get('total_case_count', 0)}")
    print(f"REFERENCE_CASE_GT_FROZEN={manifest.get('reference_case_gt_frozen_count', 0)}")
    print(f"CANONICAL_IDENTITIES={manifest.get('canonical_identity_frozen_count', 0)}")
    print(f"PROVISIONAL_IDENTITIES={manifest.get('provisional_identity_count', 0)}")
    print(f"STAGING_ROOT={staging}")
    print(f"MANIFEST={Path(args.output).resolve() if args.output else root / 'artifacts/archive/reference/portfolio_v5_final/reference_science_baseline_manifest.json'}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
