"""Build and verify the exact repository archive handed to a user.

The builder deliberately walks the repository itself (including dotfiles),
uses a fixed ZIP timestamp for reproducibility, and verifies the archive after
writing it.  The archive and its JSON verification report are separate from
the benchmark outputs and are never benchmark inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    ".gitignore",
    "README.md",
    "docs",
    "pyproject.toml",
    "runtime-requirements.txt",
    "flowintentbench",
    "scripts",
    "tests",
    "agents/gpt-5.6-sol-xhigh/agent.yaml",
    "agents/flow-case-auditor-gpt-5.6-sol/agent.yaml",
    "agents/flow-scientific-reviewer-gpt-5.6-sol/agent.yaml",
    "agents/flow-scientific-adjudicator-gpt-5.6-sol/agent.yaml",
    "agents/proxy-flow-analyst-gpt-5.6-sol/agent.yaml",
    "runtime_profiles/flow-python-v1.yaml",
    "runtime_profiles/flow-tool-free-v1.yaml",
    "scientific_review_profiles/flow-proxy-review-v1.yaml",
    "experiments/userstudy/case_manifest.json",
    "experiments/expansion_v1_development/case_manifest.json",
    "artifacts/reference/office_method_pilot_v1",
    "artifacts/reference/concept_expansion_phase1",
    "artifacts/reference/full_dataset_n1_controlled_pilot",
    "artifacts/reference/kitchen_scientific_review",
    "artifacts/reference/scientific_portfolio",
    "artifacts/reference/scientific_runs/live_agent_validation",
    "artifacts/reference/scientific_validation",
)
EXCLUDED_DIRECTORY_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "tmp"}
EXCLUDED_FILE_NAMES = {".coverage"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".swp", ".swo", ".tmp", "~")
SENSITIVE_FILE_NAMES = {
    ".env",
    ".netrc",
    "credentials.json",
    "auth.json",
    "config.toml",
    "yiapi_api_key",
    "openai_api_key",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}
SAFE_ENV_TEMPLATE_NAMES = {".env.example", ".env.sample", ".env.template"}


def _excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts[:-1]):
        return True
    name = relative.name
    return name in EXCLUDED_FILE_NAMES or name.endswith(EXCLUDED_SUFFIXES)


def _sensitive(relative: Path) -> bool:
    name = relative.name
    if name in SENSITIVE_FILE_NAMES:
        return True
    return name.startswith(".env.") and name not in SAFE_ENV_TEMPLATE_NAMES


def iter_archive_files(
    root: Path,
    *,
    output_path: Path | None = None,
    exclude_paths: Iterable[str | Path] = (),
) -> Iterable[tuple[Path, str]]:
    output_resolved = output_path.resolve() if output_path is not None else None
    excluded_resolved = {
        (root / Path(value)).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        for value in exclude_paths
    }
    files: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _excluded(relative) or _sensitive(relative):
            continue
        if output_resolved is not None and path.resolve() == output_resolved:
            continue
        if path.resolve() in excluded_resolved or any(
            excluded.is_dir() and excluded in path.resolve().parents
            for excluded in excluded_resolved
        ):
            continue
        files.append((path, relative.as_posix()))
    return sorted(files, key=lambda item: item[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_present(names: set[str], required: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for value in required:
        if value.endswith("/"):
            prefix = value
        else:
            prefix = value.rstrip("/") + "/"
        if value not in names and not any(name.startswith(prefix) for name in names):
            missing.append(value)
    return missing


def verify_handoff_archive(
    archive_path: str | Path,
    *,
    required_paths: Iterable[str] = REQUIRED_PATHS,
) -> dict[str, object]:
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive, "r") as handle:
        names = handle.namelist()
        name_set = set(names)
        forbidden = [
            name
            for name in names
            if _excluded(Path(name)) or _sensitive(Path(name))
        ]
        missing = _required_present(name_set, required_paths)
        info = handle.infolist()
        uncompressed_size = sum(item.file_size for item in info)
    if missing:
        raise RuntimeError(f"handoff archive is missing required paths: {missing}")
    if forbidden:
        raise RuntimeError(
            f"handoff archive contains forbidden cache, build, or credential files: {forbidden[:10]}"
        )
    return {
        "archive_filename": archive.name,
        "archive_path": str(archive),
        "archive_format": "zip",
        "archive_sha256": _sha256(archive),
        "file_entry_count": len(names),
        "compressed_size_bytes": archive.stat().st_size,
        "uncompressed_size_bytes": uncompressed_size,
        "required_paths": list(required_paths),
        "missing_required_paths": missing,
        "forbidden_entry_count": len(forbidden),
        "forbidden_entries": forbidden,
    }


def build_handoff_archive(
    *,
    repository_root: str | Path = ROOT,
    output_path: str | Path,
    report_path: str | Path | None = None,
    force: bool = False,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, object]:
    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    exclude_paths = tuple(exclude_paths)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite handoff archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    files = list(iter_archive_files(root, output_path=output, exclude_paths=exclude_paths))
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for source, relative in files:
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                mode = stat.S_IMODE(source.stat().st_mode)
                info.external_attr = ((mode or 0o644) & 0o777) << 16
                with source.open("rb") as source_handle, archive.open(info, "w", force_zip64=True) as archive_handle:
                    shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)
        os.replace(temporary, output)
        os.chmod(output, 0o644)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = verify_handoff_archive(output)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["repository_root"] = str(root)
    result["source_file_count"] = len(files)
    result["excluded_paths"] = [str(value) for value in exclude_paths]
    report = Path(report_path).resolve() if report_path is not None else output.with_name(output.name + ".manifest.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report_fd, report_temporary_name = tempfile.mkstemp(prefix=f".{report.name}.", suffix=".tmp", dir=report.parent)
    os.close(report_fd)
    report_temporary = Path(report_temporary_name)
    try:
        report_temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(report_temporary, report)
        os.chmod(report, 0o644)
    finally:
        if report_temporary.exists():
            report_temporary.unlink()
    result["verification_report"] = str(report)
    return result


def _declared_external_dataset_paths(repository_root: Path) -> list[str]:
    manifest_path = repository_root / "release_manifest.json"
    if not manifest_path.is_file():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release_manifest.json root must be an object")
    paths: list[str] = []
    for item in payload.get("external_dataset_assets", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("external_dataset_assets entries must contain a string path")
        relative = Path(item["path"])
        candidate = (repository_root / relative).resolve()
        try:
            normalized = candidate.relative_to(repository_root).as_posix()
        except ValueError as exc:
            raise ValueError("external dataset path escapes the repository") from exc
        paths.append(normalized)
    return sorted(set(paths))


def build_project_handoff_archive(
    *,
    repository_root: str | Path = ROOT,
    output_path: str | Path,
    report_path: str | Path | None = None,
    force: bool = False,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, object]:
    """Build the normal project handoff without transient or external data.

    The release manifest remains metadata in the project package. Only dataset
    payloads it explicitly declares external are omitted.
    """

    root = Path(repository_root).resolve()
    if not (root / "release_manifest.json").is_file():
        raise FileNotFoundError(
            "project handoff mode requires release_manifest.json so external assets are explicit"
        )
    declared_external = _declared_external_dataset_paths(root)
    exclusions = ("outputs", *declared_external, *tuple(exclude_paths))
    result = build_handoff_archive(
        repository_root=root,
        output_path=output_path,
        report_path=report_path,
        force=force,
        exclude_paths=exclusions,
    )
    result.update(
        {
            "package_role": "PROJECT_HANDOFF",
            "controlled_dependency_policy": "INCLUDE_REPOSITORY_CONTROLLED_TEST_DEPENDENCIES",
            "transient_outputs_included": False,
            "declared_external_dataset_asset_count": len(declared_external),
            "declared_external_dataset_assets": declared_external,
        }
    )
    # Persist the role metadata added by this wrapper, not just the lower-level
    # archive verification fields.
    report = Path(result["verification_report"])
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_archive_self_test(
    *,
    repository_root: str | Path = ROOT,
    archive_path: str | Path,
    report_path: str | Path | None = None,
    file_hash_manifest: str | Path | None = None,
    pytest_timeout: int = 300,
) -> dict[str, object]:
    """Build/extract a fresh archive and run the canonical test command.

    This is a release-integrity check only.  It never runs a model or changes
    benchmark artifacts in the source tree.
    """
    root = Path(repository_root).resolve()
    archive = Path(archive_path).resolve()
    built = build_handoff_archive(repository_root=root, output_path=archive, force=True)
    manifest_relative: Path | None = None
    if file_hash_manifest is not None:
        manifest_path = Path(file_hash_manifest).resolve()
        try:
            manifest_relative = manifest_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("file-hash manifest must be inside the archived repository") from exc
    with tempfile.TemporaryDirectory(prefix="flowintentbench-archive-self-test-") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(archive, "r") as handle:
            handle.extractall(extracted)
        tests = subprocess.run(
            [os.environ.get("PYTHON", "python"), "-m", "pytest", "-q"],
            cwd=extracted,
            text=True,
            capture_output=True,
            timeout=pytest_timeout,
        )
        preflight_report = extracted / "outputs/experiments/archive_self_test_preflight/preflight_report.json"
        preflight = subprocess.run(
            [
                os.environ.get("PYTHON", "python"),
                "scripts/preflight_full_dataset_n1.py",
                "--output-root",
                str(extracted / "outputs/experiments/archive_self_test_preflight"),
                "--report",
                str(preflight_report),
            ],
            cwd=extracted,
            text=True,
            capture_output=True,
            timeout=pytest_timeout,
        )
        required_configuration = (
            ".gitignore",
            "pyproject.toml",
            "runtime-requirements.txt",
                "experiments/userstudy/case_manifest.json",
        )
        required_present = all((extracted / name).exists() for name in required_configuration)
        hash_failures: list[dict[str, str]] = []
        checked_hash_count = 0
        if manifest_relative is not None:
            manifest_value = json.loads((extracted / manifest_relative).read_text(encoding="utf-8"))
            if not isinstance(manifest_value, dict):
                raise ValueError("file-hash manifest root must be an object")
            for relative_name, expected in sorted(manifest_value.items()):
                checked_hash_count += 1
                candidate = extracted / str(relative_name)
                if not candidate.is_file():
                    hash_failures.append({"path": str(relative_name), "reason": "missing"})
                    continue
                actual = _sha256(candidate)
                if actual.casefold() != str(expected).casefold():
                    hash_failures.append(
                        {"path": str(relative_name), "reason": "sha256 mismatch"}
                    )
        hashes_verified = manifest_relative is not None and not hash_failures
        status = (
            "PASS"
            if tests.returncode == 0
            and preflight.returncode == 0
            and required_present
            and hashes_verified
            else "FAIL"
        )
        result = {
            "status": status,
            "archive": built,
            "fresh_extract": True,
            "canonical_test_command": "python -m pytest -q",
            "pytest_returncode": tests.returncode,
            "pytest_summary": tests.stdout.strip().splitlines()[-1] if tests.stdout.strip() else None,
            "pytest_stderr_tail": tests.stderr[-2000:],
            "preflight_command": "python scripts/preflight_full_dataset_n1.py",
            "preflight_returncode": preflight.returncode,
            "preflight_summary": preflight.stdout.strip().splitlines()[-1] if preflight.stdout.strip() else None,
            "preflight_stderr_tail": preflight.stderr[-2000:],
            "required_configuration_paths": list(required_configuration),
            "required_dotfiles_present": required_present,
            "file_hash_manifest": None if manifest_relative is None else manifest_relative.as_posix(),
            "manifest_hashes_verified": hashes_verified,
            "manifest_hash_count": checked_hash_count,
            "manifest_hash_failures": hash_failures,
        }
    report = Path(report_path).resolve() if report_path is not None else archive.with_name("archive_self_test.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_path"] = str(report)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--project-handoff",
        action="store_true",
        help="exclude outputs and dataset assets declared external by release_manifest.json",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="repository-relative path to omit; may be repeated for generated verification files",
    )
    args = parser.parse_args()
    builder = build_project_handoff_archive if args.project_handoff else build_handoff_archive
    result = builder(
        repository_root=args.root,
        output_path=args.output,
        report_path=args.report,
        force=args.force,
        exclude_paths=args.exclude,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
