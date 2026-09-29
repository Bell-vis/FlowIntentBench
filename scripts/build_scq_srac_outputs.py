#!/usr/bin/env python3
"""Build the frozen SCQ/SRAC preparation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.scq_srac_artifacts import build_input_hardening_outputs, build_scq_srac_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--input-output-root", type=Path, default=ROOT / "outputs/experiments/scq_srac_inputs")
    parser.add_argument("--scq-output-root", type=Path, default=ROOT / "outputs/experiments/scq_srac")
    args = parser.parse_args()
    first = build_input_hardening_outputs(args.repository_root, args.input_output_root)
    second = build_scq_srac_outputs(args.repository_root, args.scq_output_root)
    print(json.dumps({"input_hardening": first, "scq_srac": second}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
