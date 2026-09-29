#!/usr/bin/env python3
"""Compare saved Flow Expert reviews with independent external human labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.scientific_expert_validation import compare_external_human_labels


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-packet", type=Path, required=True)
    parser.add_argument("--model-reviews", type=Path, required=True)
    parser.add_argument("--human-labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    calibration_packet = _read_object(args.calibration_packet, "calibration packet")
    model_payload = _read_object(args.model_reviews, "model reviews")
    model_reviews = model_payload.get("model_reviews", model_payload)
    if not isinstance(model_reviews, Mapping) or any(
        not isinstance(value, Mapping) for value in model_reviews.values()
    ):
        raise ValueError("model reviews must map calibration item IDs to review objects")
    human_labels = (
        _read_object(args.human_labels, "human labels") if args.human_labels else None
    )
    result = compare_external_human_labels(
        calibration_packet,
        {str(key): dict(value) for key, value in model_reviews.items()},
        human_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
