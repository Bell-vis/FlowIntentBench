"""Re-run the 7x4 construction pattern in an isolated metadata staging tree.

Only dataset manifests, construction specifications, and generated case
artifacts are staged.  Large numerical payloads remain in the repository and
are referenced by absolute manifest paths.  The active ``datasets`` tree is
never used as a write target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# Make direct ``python scripts/...`` invocation work from any cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_all_formal_pilots as flow_builder
from scripts import build_context_construction as context_builder
from flowintentbench import office_analysis


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_dataset_tree(stage_root: Path) -> None:
    stage_datasets = stage_root / "datasets"
    stage_datasets.mkdir(parents=True, exist_ok=True)
    for source in sorted(ROOT.joinpath("datasets").iterdir()):
        if not source.is_dir() or not (source / "dataset_manifest.json").is_file():
            continue
        target = stage_datasets / source.name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("dataset_manifest.json", "data_metadata.json"):
            shutil.copy2(source / filename, target / filename)
        shutil.copytree(source / "construction", target / "construction")
        manifest_path = target / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The existing readers resolve ``ROOT / file_root``; an absolute path
        # therefore keeps the raw payload read-only while staging writes stay
        # under stage_root.
        manifest["file_root"] = str(ROOT / "datasets")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def rebuild_pattern_staging(stage_root: str | Path) -> dict[str, object]:
    stage = Path(stage_root).resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copytree(ROOT / "runtime_profiles", stage / "runtime_profiles")
    _prepare_dataset_tree(stage)

    # Existing builders are reused; only their repository root is redirected.
    original_roots = (flow_builder.ROOT, context_builder.ROOT, office_analysis.ROOT)
    flow_builder.ROOT = stage
    context_builder.ROOT = stage
    office_analysis.ROOT = stage
    try:
        for dataset_id in flow_builder.NON_OFFICE_DATASETS:
            config = flow_builder.CONFIGS[dataset_id]
            for suffix in flow_builder.CASE_SUFFIXES:
                flow_builder._materialize_case(config, suffix)
        for case_id in office_analysis.CASE_IDS:
            office_analysis.build_case(case_id, dataset_builder=context_builder.build_dataset)
    finally:
        flow_builder.ROOT, context_builder.ROOT, office_analysis.ROOT = original_roots

    staged = sorted(stage.glob("datasets/*/construction/cases/*/ground_truth.json"))
    comparisons: list[dict[str, object]] = []
    matching = 0
    for path in staged:
        relative = path.relative_to(stage)
        active = ROOT / relative
        staged_hash = _hash(path)
        active_hash = _hash(active) if active.is_file() else None
        equal = staged_hash == active_hash
        matching += int(equal)
        comparisons.append(
            {
                "path": str(relative),
                "staged_sha256": staged_hash,
                "active_sha256": active_hash,
                "matches_active": equal,
            }
        )
    return {
        "status": "PASS" if len(staged) == 28 else "FAIL",
        "source_policy": "PATTERN_ONLY_STAGING",
        "active_case_mutation": False,
        "reconstructed_ground_truth_count": len(staged),
        "matching_active_hash_count": matching,
        "all_hashes_match": matching == len(staged),
        "staging_root": str(stage),
        "cases": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/current/release_migration/pattern_rebuild_stage")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/current/release_migration/pattern_rebuild.json")
    args = parser.parse_args()
    result = rebuild_pattern_staging(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "reconstructed_ground_truth_count", "matching_active_hash_count", "all_hashes_match", "active_case_mutation")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
