#!/usr/bin/env python3
"""Build the reproducible FlowIntentBench formal-release archive.

The authoritative release is self-contained by default: frozen current
portfolio artifacts and numerical dataset payloads are shipped so a fresh
extraction can reproduce the complete repository test suite.  An explicit
``--external-datasets`` mode remains available for a lightweight handoff, but
that mode cannot claim full-suite reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_handoff_archive import REQUIRED_PATHS, build_handoff_archive, iter_archive_files


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "release_manifest.json"
REQUIRED_RELEASE_PATHS = (
    ".gitignore",
    "README.md",
    "docs",
    "pyproject.toml",
    "runtime-requirements.txt",
    "flowintentbench",
    "scripts",
    "tests",
    "runtime_profiles",
    "experiments/userstudy/case_manifest.json",
    "experiments/expansion_v1_development/case_manifest.json",
    "artifacts/reference/office_method_pilot_v1",
    "artifacts/reference/concept_expansion_phase1",
    "artifacts/reference/full_dataset_n1_controlled_pilot",
    "artifacts/reference/kitchen_scientific_review",
    "artifacts/reference/scientific_portfolio",
    "artifacts/reference/scientific_runs/live_agent_validation",
    "artifacts/reference/scientific_validation",
    "artifacts/reference/portfolio_v7_gt_aligned_final",
    "datasets/data_metadata_sources_v1.yaml",
    "docs",
    "outputs/experiments/scientific_semantic_optimization/manifest.json",
    "outputs/experiments/scientific_question_candidate_matrix_round2/targeted_reviews/manifest.json",
)

# These two experiment manifests are immutable inputs to existing regression
# tests (including the Combustor target-proposal and quality-review tests).
# Other ``outputs/experiments`` files are transient and are intentionally not
# shipped in the formal release.
FROZEN_EXPERIMENT_FILES = frozenset(
    {
        "outputs/experiments/scientific_semantic_optimization/manifest.json",
        "outputs/experiments/scientific_question_candidate_matrix_round2/targeted_reviews/manifest.json",
    }
)

# Generated user-study and 96-case runs are intentionally excluded from the
# formal source archive; they remain local evidence under `outputs/`.
RETAINED_EXPERIMENT_DIRECTORIES = ()

# Validation of an archive writes a report containing that archive's own
# SHA256.  Shipping the report would make the next archive include a hash of
# the previous archive, creating a self-referential and non-reproducible
# artifact.  Keep validation outputs local to the workspace instead.
TRANSIENT_OUTPUT_DIRECTORIES = (
    "outputs/staging",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_external_dataset_file(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    # Keep all construction/evaluation metadata in the release.  Numerical
    # payloads above 1 MiB are provisioned externally to avoid a multi-GB ZIP.
    return relative.parts and relative.parts[0] == "datasets" and path.stat().st_size > 1024 * 1024


def _manifest_payload(
    root: Path,
    external_files: list[Path],
    excluded_experiment_files: list[Path],
) -> dict[str, Any]:
    excluded = {MANIFEST_NAME}
    files: list[dict[str, Any]] = []
    for path, relative in iter_archive_files(
        root,
        exclude_paths=(
            "outputs/README.md",
            *TRANSIENT_OUTPUT_DIRECTORIES,
            *excluded_experiment_files,
            *external_files,
        ),
    ):
        if relative == MANIFEST_NAME or relative in excluded:
            continue
        item = {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        if relative.startswith("outputs/current/") or any(
            relative.startswith(directory.rstrip("/") + "/")
            for directory in RETAINED_EXPERIMENT_DIRECTORIES
        ):
            item["reproducibility_evidence"] = True
        if relative in FROZEN_EXPERIMENT_FILES:
            item["reproducibility_evidence"] = True
        files.append(item)
    external: list[dict[str, Any]] = []
    for path in sorted(external_files):
        relative = path.relative_to(root).as_posix()
        dataset_id = relative.split("/", 2)[1] if len(relative.split("/", 2)) > 1 else "unknown"
        external.append({
            "dataset_id": dataset_id,
            "path": relative,
            "required": True,
            "sha256": _sha256(path),
            "availability_status": "EXTERNAL_REQUIRED_ASSET",
            "provisioning": "Provision the dataset payload at this repository-relative path before data-dependent execution.",
        })
    dataset_policy = (
        "EXTERNAL_REQUIRED_ASSET" if external_files else "INCLUDED_IN_RELEASE"
    )
    return {
        "manifest_version": "flowintentbench-release-v1",
        "dataset_policy": dataset_policy,
        "required_release_paths": list(REQUIRED_RELEASE_PATHS),
        # An empty list is intentional and means ``python -m pytest -q`` at
        # the extracted repository root.  This prevents a release from
        # silently proving only a small curated subset of the 749 tests.
        "self_contained_test_paths": [],
        "full_test_suite_required": not bool(external_files),
        "full_test_suite_timeout_seconds": 900 if not external_files else 300,
        "frozen_reproducibility_paths": sorted(
            item["path"] for item in files if item.get("reproducibility_evidence") is True
        ),
        "files": files,
        "external_dataset_assets": external,
        "notes": (
            "All dataset payloads and outputs/current are included; a fresh extraction "
            "must reproduce the complete pytest suite."
            if not external_files
            else "Numerical dataset payloads are external; this lightweight mode cannot claim full-suite reproducibility."
        ),
    }


def build_release_archive(
    *, repository_root: str | Path = ROOT,
    output_path: str | Path | None = None,
    force: bool = True,
    include_external_datasets: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    output = Path(
        output_path
        or root / "outputs/experiments/release_archive/flowintentbench-release.zip"
    ).resolve()
    external_files = []
    if not include_external_datasets:
        external_files = [
            path for path in root.glob("datasets/**/*")
            if path.is_file() and _is_external_dataset_file(path, root)
        ]
    excluded_experiment_files = []
    for path in root.glob("outputs/experiments/**/*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        retained = relative in FROZEN_EXPERIMENT_FILES or any(
            relative.startswith(directory.rstrip("/") + "/")
            for directory in RETAINED_EXPERIMENT_DIRECTORIES
        )
        if not retained:
            excluded_experiment_files.append(path)
    # ``outputs/staging`` is construction scratch space, not a reproducibility
    # input.  It is deliberately excluded as a directory so a prior build
    # cannot leak hundreds of duplicate candidate artifacts into a release.
    excluded_output_directories = [root / value for value in TRANSIENT_OUTPUT_DIRECTORIES]
    manifest = _manifest_payload(root, external_files, excluded_experiment_files)
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    excluded = (
        "outputs/README.md",
        *excluded_output_directories,
        *excluded_experiment_files,
        *external_files,
    )
    built = build_handoff_archive(
        repository_root=root,
        output_path=output,
        force=force,
        exclude_paths=excluded,
        report_path=output.with_name("release_archive_build.json"),
    )
    built.update({
        "release_manifest": MANIFEST_NAME,
        "release_manifest_path": str(manifest_path),
        "dataset_policy": manifest["dataset_policy"],
        "external_dataset_asset_count": len(external_files),
        "self_contained_test_paths": manifest["self_contained_test_paths"],
        "full_test_suite_required": manifest["full_test_suite_required"],
        "full_test_suite_timeout_seconds": manifest["full_test_suite_timeout_seconds"],
        "required_release_paths": list(REQUIRED_RELEASE_PATHS),
    })
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiments/release_archive/flowintentbench-release.zip",
    )
    parser.add_argument(
        "--external-datasets",
        action="store_true",
        help="omit large numerical payloads (lightweight mode; not 749-test reproducible)",
    )
    args = parser.parse_args()
    result = build_release_archive(
        repository_root=args.repository_root,
        output_path=args.output,
        include_external_datasets=not args.external_datasets,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
