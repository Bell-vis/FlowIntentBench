"""Execute two blinded, independent evaluator calibration judges.

No configuration means a fail-closed BLOCKED result.  This command never
uses the evaluated model as a judge and never computes benchmark metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.evaluator_runner import run_evaluator_calibration  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--judge-a-config", type=Path)
    parser.add_argument("--judge-b-config", type=Path)
    args = parser.parse_args()
    result = run_evaluator_calibration(
        collection_root=args.collection_root,
        datasets_root=args.datasets_root,
        output_root=args.output_root,
        judge_a_config_path=args.judge_a_config,
        judge_b_config_path=args.judge_b_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"EXECUTED", "BLOCKED"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
