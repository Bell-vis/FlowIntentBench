#!/usr/bin/env python3
"""Audit controlled-file completeness across project and release ZIP roles.

This audit compares path presence, not archive bytes. Large numerical dataset
payloads remain external under the formal release policy, and transient output
trees are not treated as repository-controlled test dependencies.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:  # Imported as scripts.audit_package_parity by tests and library callers.
    from scripts.build_handoff_archive import _declared_external_dataset_paths, _excluded
except ModuleNotFoundError:  # Executed directly as python scripts/audit_package_parity.py.
    from build_handoff_archive import _declared_external_dataset_paths, _excluded


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_ROOTS = (
    "flowintentbench",
    "scripts",
    "tests",
    "agents",
    "runtime_profiles",
    "scientific_review_profiles",
    "experiments",
    "artifacts/reference",
    "datasets",
    "docs",
)
CONTROLLED_ROOT_FILES = (
    ".gitignore",
    "README.md",
    "method.md",
    "FlowIntentBench.md",
    "pyproject.toml",
    "runtime-requirements.txt",
)


def _under_controlled_root(relative: Path) -> bool:
    value = relative.as_posix()
    return any(
        value == root or value.startswith(root.rstrip("/") + "/")
        for root in CONTROLLED_ROOTS
    )


def controlled_test_dependencies(repository_root: str | Path) -> tuple[list[str], list[str]]:
    """Return required controlled paths and excluded external dataset paths."""

    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    required: set[str] = set()
    declared_external = set(_declared_external_dataset_paths(root))
    external: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        relative_name = relative.as_posix()
        if _excluded(relative) or "outputs" in relative.parts:
            continue
        if relative_name == "release_manifest.json":
            # This is formal-release metadata, not a shared test dependency.
            continue
        if relative_name not in CONTROLLED_ROOT_FILES and not _under_controlled_root(relative):
            continue
        if relative_name in declared_external:
            external.append(relative_name)
            continue
        required.add(relative_name)
    return sorted(required), sorted(external)


def _normalized_archive_names(names: Iterable[str], required: set[str]) -> tuple[set[str], str | None]:
    files = {name.rstrip("/") for name in names if name and not name.endswith("/")}
    if "pyproject.toml" in files:
        return files, None

    # Uploaded project ZIPs may have one enclosing repository directory. Pick
    # a prefix only when it exposes the canonical pyproject sentinel.
    prefixes = sorted(
        {
            name[: -len("pyproject.toml")]
            for name in files
            if name.endswith("pyproject.toml") and name != "pyproject.toml"
        },
        key=len,
    )
    if not prefixes:
        return files, None
    best_prefix = max(
        prefixes,
        key=lambda prefix: len(required & {name[len(prefix) :] for name in files if name.startswith(prefix)}),
    )
    normalized = {
        name[len(best_prefix) :] for name in files if name.startswith(best_prefix)
    }
    return normalized, best_prefix.rstrip("/") or None


def _audit_archive(
    archive_path: str | Path | None,
    *,
    role: str,
    required_paths: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "role": role,
        "archive_path": None if archive_path is None else str(Path(archive_path).resolve()),
        "availability_status": "NOT_AVAILABLE",
        "controlled_file_completeness": "NOT_EVALUATED",
        "required_controlled_file_count": len(required_paths),
        "present_controlled_file_count": 0,
        "missing_controlled_file_count": None,
        "missing_controlled_files": [],
        "archive_root_prefix": None,
    }
    if archive_path is None:
        result["reason"] = f"{role} ZIP was not supplied"
        return result
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        result["reason"] = f"{role} ZIP does not exist"
        return result
    try:
        with zipfile.ZipFile(archive) as handle:
            names, prefix = _normalized_archive_names(handle.namelist(), set(required_paths))
    except (OSError, zipfile.BadZipFile) as exc:
        result["availability_status"] = "INVALID_ARCHIVE"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    missing = sorted(set(required_paths) - names)
    result.update(
        {
            "availability_status": "AVAILABLE",
            "controlled_file_completeness": "PASS" if not missing else "FAIL",
            "present_controlled_file_count": len(required_paths) - len(missing),
            "missing_controlled_file_count": len(missing),
            "missing_controlled_files": missing,
            "archive_root_prefix": prefix,
        }
    )
    return result


def audit_package_parity(
    *,
    repository_root: str | Path = ROOT,
    project_handoff_zip: str | Path | None = None,
    formal_release_zip: str | Path | None = None,
) -> dict[str, Any]:
    """Compare shared controlled dependencies without requiring byte identity."""

    root = Path(repository_root).resolve()
    required, external = controlled_test_dependencies(root)
    project = _audit_archive(
        project_handoff_zip,
        role="PROJECT_HANDOFF",
        required_paths=required,
    )
    release = _audit_archive(
        formal_release_zip,
        role="FORMAL_RELEASE",
        required_paths=required,
    )
    completeness = {
        project["controlled_file_completeness"],
        release["controlled_file_completeness"],
    }
    overall = (
        "FAIL"
        if "FAIL" in completeness
        else "NOT_EVALUATED"
        if "NOT_EVALUATED" in completeness
        else "PASS"
    )
    return {
        "schema_version": "flowintentbench-package-parity-v1",
        "comparison_semantics": "repository-controlled path presence; byte identity is not required",
        "working_tree": {
            "role": "CONTROLLED_DEPENDENCY_SOURCE",
            "status": "PASS",
            "repository_root": str(root),
            "required_controlled_file_count": len(required),
            "excluded_external_dataset_file_count": len(external),
            "excluded_external_dataset_files": external,
        },
        "project_handoff_zip": project,
        "formal_release_zip": release,
        "overall_package_parity_status": overall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--project-handoff-zip", type=Path, default=None)
    parser.add_argument("--formal-release-zip", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit_package_parity(
        repository_root=args.repository_root,
        project_handoff_zip=args.project_handoff_zip,
        formal_release_zip=args.formal_release_zip,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["overall_package_parity_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
