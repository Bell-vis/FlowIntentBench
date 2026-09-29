#!/usr/bin/env python3
"""Rebuild only the O1-F2 construction sidecars in a reference snapshot.

The script is intentionally a narrow integrity operation.  It reruns the
existing deterministic agent-ready constructor into staging, proves that all
21 non-F2 case artifacts are byte-identical to the source snapshot, and then
freezes the corrected tree as a new write-once reference boundary.  No model
or scientific adjudicator is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.qualified_scientific_cases import (
    build_agent_ready_scientific_case_portfolio,
)
from flowintentbench.reference_freeze import build_reference_portfolio_freeze_v4


REFERENCE_FILES = (
    "case_input.json",
    "case_construction_metadata.json",
    "ground_truth.json",
    "scientific_evaluation_contract.json",
    "finding_verification_policy.json",
    "finding_execution_binding.json",
    "scientific_semantic_projection.json",
    "agent_ready_evidence.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_dirs(root: Path):
    return sorted(
        path
        for dataset in root.iterdir()
        if dataset.is_dir()
        for path in dataset.iterdir()
        if path.is_dir()
    )


def repair_snapshot(
    repository_root: str | Path,
    source_manifest: str | Path,
    output_manifest: str | Path,
    *,
    construction_root: str | Path | None = None,
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    constructor_root = (
        root if construction_root is None else Path(construction_root).resolve()
    )
    source = Path(source_manifest).resolve()
    source_cases = source.parent / "cases"
    if not source.is_file() or not source_cases.is_dir():
        raise FileNotFoundError(source)
    stage = Path(tempfile.mkdtemp(prefix="flowintentbench-f2-repair-"))
    build_agent_ready_scientific_case_portfolio(constructor_root, stage)
    fresh_cases = stage / "agent_ready_cases"
    f2_ids: list[str] = []
    unchanged_ids: list[str] = []
    changed_non_f2: list[str] = []
    for old in _case_dirs(source_cases):
        case_id = old.name
        fresh = fresh_cases / old.parent.name / case_id
        condition = "O1-F2" if case_id.endswith("o1_f2") else "OTHER"
        if condition == "O1-F2":
            f2_ids.append(case_id)
            continue
        for name in REFERENCE_FILES:
            if _sha256(old / name) != _sha256(fresh / name):
                changed_non_f2.append(f"{old.parent.name}/{case_id}/{name}")
        if not changed_non_f2 or not any(item.startswith(f"{old.parent.name}/{case_id}/") for item in changed_non_f2):
            unchanged_ids.append(case_id)
    if changed_non_f2:
        raise ValueError(
            "non-F2 reference drift detected; refusing targeted repair: "
            + ", ".join(changed_non_f2)
        )
    payload = build_reference_portfolio_freeze_v4(
        root,
        output_manifest,
        source_case_root=fresh_cases,
    )
    return {
        "status": "PASS",
        "source_manifest": str(source),
        "output_manifest": str(Path(output_manifest).resolve()),
        "affected_case_count": len(f2_ids),
        "affected_case_ids": sorted(f2_ids),
        "unaffected_case_count": len(unchanged_ids),
        "non_f2_artifacts_unchanged": len(changed_non_f2) == 0,
        "reference_case_gt_frozen_count": payload.get("reference_case_gt_frozen_count", 0),
        "total_case_count": payload.get("total_case_count", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-manifest", type=Path, default=None)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    source = args.source_manifest or root / "artifacts/reference/portfolio_v7_gt_aligned_final/reference_science_baseline_manifest.json"
    output = args.output_manifest or root / "artifacts/reference/portfolio_v7_targeted/reference_science_baseline_manifest.json"
    result = repair_snapshot(root, source, output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
