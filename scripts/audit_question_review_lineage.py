#!/usr/bin/env python3
"""Audit current question text against live final reviews; write only a report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.review_lineage import audit_question_review_lineage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/userstudy/case_manifest.json"))
    parser.add_argument("--review-root", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = args.review_root or [Path("artifacts/question_authoring"), Path("artifacts/review"), Path("outputs/current")]
    result = audit_question_review_lineage(args.repository_root, args.manifest, roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
