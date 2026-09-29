#!/usr/bin/env python3
"""Build the evidence-first Kitchen scientific grounding inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.kitchen_evidence import build_kitchen_evidence_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/experiments/kitchen_scientific_review",
    )
    args = parser.parse_args()
    result = build_kitchen_evidence_bundle(args.repository_root, args.output_root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "family_id": result["family_id"],
                "dataset_id": result["dataset_id"],
                "output_root": result["output_root"],
                "new_evidence_count": result["new_evidence_inventory"]["new_evidence_count"],
                "unresolved_critical_claim_count": result["unresolved_evidence_gaps"]["unresolved_critical_claim_count"],
                "provenance_status": result["provenance_audit"]["status"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
