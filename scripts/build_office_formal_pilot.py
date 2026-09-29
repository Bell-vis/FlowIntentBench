"""CLI boundary for the Office formal-pilot builder.

Scientific computation lives in :mod:`flowintentbench.office_analysis`;
this script only supplies the repository construction builder and parses CLI
arguments. Historical helper names remain re-exported for compatibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.office_analysis import *  # noqa: F401,F403
from flowintentbench.office_analysis import CASE_IDS, build_case as _build_case
from scripts.build_context_construction import build_dataset


def build_case(case_id: str) -> None:
    """Build one case through the existing Dataset Context builder."""
    _build_case(case_id, dataset_builder=build_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="*", default=list(CASE_IDS))
    args = parser.parse_args()
    unknown = sorted(set(args.case_ids) - set(CASE_IDS))
    if unknown:
        parser.error(f"unsupported Office pilot case(s): {', '.join(unknown)}")
    for case_id in args.case_ids:
        build_case(case_id)


if __name__ == "__main__":
    main()
