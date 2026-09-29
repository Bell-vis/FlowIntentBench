#!/usr/bin/env python3
"""Validate one actual FlowIntentBench release ZIP in a fresh extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _member_inventory(
    names: list[str], *, allowed_transient_members: set[str] | None = None
) -> dict[str, Any]:
    """Summarize forbidden members in an emitted archive.

    ZIP archives do not have to contain explicit directory entries, so all
    checks operate on path components.  Counts refer to archive members (not
    source-tree directories), which makes the result auditable and stable.
    """

    pycache_members = [
        name for name in names if "__pycache__" in Path(name).parts
    ]
    pytest_cache_members = [
        name for name in names if ".pytest_cache" in Path(name).parts
    ]
    pyc_members = [name for name in names if name.casefold().endswith(".pyc")]
    allowed_transient_members = allowed_transient_members or set()
    transient_output_members = [
        name
        for name in names
        if "outputs" in Path(name).parts and name not in allowed_transient_members
    ]
    forbidden = sorted(
        set(pycache_members)
        | set(pytest_cache_members)
        | set(pyc_members)
        | set(transient_output_members)
    )
    return {
        "gitignore_present": ".gitignore" in names,
        "pycache_member_count": len(pycache_members),
        "pycache_members": sorted(pycache_members),
        "pyc_member_count": len(pyc_members),
        "pyc_members": sorted(pyc_members),
        "pytest_cache_member_count": len(pytest_cache_members),
        "pytest_cache_members": sorted(pytest_cache_members),
        "transient_output_member_count": len(transient_output_members),
        "transient_output_members": sorted(transient_output_members),
        "forbidden_member_count": len(forbidden),
        "forbidden_members": forbidden,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_markdown(result: dict[str, Any]) -> str:
    rows = ["# Release archive validation", "", "| Check | Status |", "|---|---|"]
    for key in (
        "archive_content_status",
        "manifest_integrity_status",
        "self_contained_tests_status",
        "external_data_status",
        "environment_dependency_status",
        "overall_release_status",
    ):
        rows.append(f"| `{key}` | **{result.get(key)}** |")
    return "\n".join(rows) + "\n"


def validate_release_archive(archive_path: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    archive = Path(archive_path).resolve()
    result: dict[str, Any] = {
        "archive_filename": archive.name,
        "archive_path": str(archive),
        "archive_content_status": "FAIL",
        "manifest_integrity_status": "FAIL",
        "self_contained_tests_status": "NOT_RUN",
        "external_data_status": "NOT_DECLARED",
        "environment_dependency_status": "PASS",
        "overall_release_status": "FAIL",
    }
    if not archive.is_file():
        result["archive_content_reason"] = "archive does not exist"
        return result
    result["archive_sha256"] = _sha256(archive)
    with tempfile.TemporaryDirectory(prefix="flowintentbench-release-validate-") as temporary:
        extracted = Path(temporary)
        try:
            with zipfile.ZipFile(archive) as handle:
                member_names = handle.namelist()
                names = set(member_names)
                # An output is exempt from the transient-output check only
                # when the release manifest explicitly marks that exact path
                # as reproducibility evidence.  This allowlist never applies
                # to cache/bytecode members.
                allowed_transient_members: set[str] = set()
                if "release_manifest.json" in names:
                    try:
                        manifest_probe = json.loads(
                            handle.read("release_manifest.json").decode("utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        manifest_probe = {}
                    if isinstance(manifest_probe, dict):
                        declared = manifest_probe.get("frozen_reproducibility_paths", [])
                        if isinstance(declared, list):
                            allowed_transient_members.update(
                                str(value) for value in declared if isinstance(value, str)
                            )
                        for item in manifest_probe.get("files", []):
                            if (
                                isinstance(item, dict)
                                and item.get("reproducibility_evidence") is True
                                and isinstance(item.get("path"), str)
                            ):
                                allowed_transient_members.add(str(item["path"]))
                result.update(
                    _member_inventory(
                        member_names,
                        allowed_transient_members=allowed_transient_members,
                    )
                )
                handle.extractall(extracted)
            # Keep the historical field for consumers of older validation
            # reports, while exposing the canonical name used by the release
            # contract.
            result["dotfiles_present"] = result["gitignore_present"]
            manifest_path = extracted / "release_manifest.json"
            if not result["gitignore_present"]:
                result["archive_content_reason"] = "required .gitignore is absent"
            elif not manifest_path.is_file():
                result["archive_content_reason"] = "release_manifest.json is absent"
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                required = list(manifest.get("required_release_paths", []))
                missing = [path for path in required if not (extracted / path).exists()]
                result["missing_required_paths"] = missing
                forbidden = result["forbidden_member_count"]
                result["archive_content_status"] = (
                    "PASS"
                    if not missing and not forbidden
                    else "ARCHIVE_CONTENT_FAILURE"
                )
                if manifest.get("dataset_policy") == "EXTERNAL_REQUIRED_ASSET":
                    result["external_data_status"] = "DECLARED_EXTERNAL_REQUIRED_ASSET"
                    result["external_dataset_asset_count"] = len(manifest.get("external_dataset_assets", []))
                else:
                    result["external_data_status"] = "INCLUDED_IN_RELEASE"
                failures: list[dict[str, str]] = []
                for item in manifest.get("files", []):
                    path = extracted / str(item.get("path"))
                    if not path.is_file():
                        failures.append({"path": str(item.get("path")), "reason": "missing"})
                    elif _sha256(path).casefold() != str(item.get("sha256", "")).casefold():
                        failures.append({"path": str(item.get("path")), "reason": "sha256 mismatch"})
                result["manifest_file_count"] = len(manifest.get("files", []))
                result["manifest_hash_failures"] = failures
                result["manifest_integrity_status"] = "PASS" if not failures else "FAIL"
                if manifest.get("dataset_policy") == "EXTERNAL_REQUIRED_ASSET":
                    # A lightweight handoff deliberately omits large numeric
                    # payloads.  Running pytest in that extraction produces
                    # misleading failures from missing data, so this mode is
                    # explicitly not self-contained rather than failed.  The
                    # required external assets and their digests remain in the
                    # manifest and are reported above.
                    result["self_contained_test_command"] = None
                    result["self_contained_test_timeout_seconds"] = None
                    result["self_contained_tests_status"] = "NOT_APPLICABLE_EXTERNAL_DATA"
                    result["self_contained_tests_returncode"] = None
                    result["self_contained_tests_summary"] = (
                        "not run: numerical dataset payloads are declared external"
                    )
                else:
                    tests = [sys.executable, "-m", "pytest", "-q", *manifest.get("self_contained_test_paths", [])]
                    timeout_seconds = manifest.get("full_test_suite_timeout_seconds", 300)
                    try:
                        timeout_seconds = max(1, int(timeout_seconds))
                    except (TypeError, ValueError):
                        timeout_seconds = 300
                    completed = subprocess.run(tests, cwd=extracted, text=True, capture_output=True, timeout=timeout_seconds)
                    result["self_contained_test_command"] = "python -m pytest -q " + " ".join(manifest.get("self_contained_test_paths", []))
                    result["self_contained_test_timeout_seconds"] = timeout_seconds
                    result["self_contained_tests_status"] = "PASS" if completed.returncode == 0 else "FAIL"
                    result["self_contained_tests_returncode"] = completed.returncode
                    output_lines = [line.strip() for line in (completed.stdout + "\n" + completed.stderr).splitlines() if line.strip()]
                    result["self_contained_tests_summary"] = output_lines[-1] if output_lines else None
                # The release preflight is intentionally metadata-only: raw
                # payload checks are represented by external_data_status.
                preflight = subprocess.run(
                    [sys.executable, "-c", "import flowintentbench; print('release import preflight PASS')"],
                    cwd=extracted, text=True, capture_output=True, timeout=60,
                )
                result["release_preflight_status"] = "PASS" if preflight.returncode == 0 else "FAIL"
        except (OSError, zipfile.BadZipFile, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            result["archive_content_reason"] = f"{type(exc).__name__}: {exc}"
    # A malformed/incomplete archive must not accidentally be reported as
    # content-clean merely because manifest processing was skipped.
    if result.get("forbidden_member_count", 0):
        result.setdefault("archive_content_reason", "forbidden transient members are present")
    result["overall_release_status"] = (
        "PASS"
        if result.get("archive_content_status") == "PASS"
        and result.get("manifest_integrity_status") == "PASS"
        and result.get("self_contained_tests_status") in {"PASS", "NOT_APPLICABLE_EXTERNAL_DATA"}
        and result.get("release_preflight_status") == "PASS"
        and result.get("external_data_status") in {"DECLARED_EXTERNAL_REQUIRED_ASSET", "INCLUDED_IN_RELEASE"}
        else "FAIL"
    )
    result["status"] = result["overall_release_status"]
    if output_root is not None:
        output = Path(output_root).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "release_archive_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "release_archive_validation.md").write_text(_status_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    result = validate_release_archive(args.archive, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("overall_release_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
