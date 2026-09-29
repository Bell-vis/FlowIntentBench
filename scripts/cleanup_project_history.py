#!/usr/bin/env python3
"""Move superseded generated runs out of the active project surface.

The command is deliberately conservative: source code, datasets, experiment
manifests, current outputs, and all answer/judgment files are preserved. A
manifest makes every move reversible. Use ``--dry-run`` before ``--apply``.
Duplicate YiAPI history trees can be removed explicitly with
``--prune-duplicate-history`` after their shared content-addressed objects
have been verified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SUPERSEDED = {
    "yiapi_rebuild_20260920_210331": "bootstrap-only YiAPI rebuild superseded by the resumable run",
    "yiapi_rebuild_20260920_210656": "incomplete YiAPI rebuild superseded after the schema fix",
}

CACHE_DIRS = (
    ".pytest_cache", ".ruff_cache", "flowintentbench/__pycache__",
    "scripts/__pycache__", "tests/__pycache__", "flowintentbench.egg-info",
)
HISTORY_OBJECTS = ROOT / "archive" / "history_objects"
LEGACY_28_PATHS = {
    "outputs/userstudy_n3": "compatibility link for the superseded 28-case study",
    "outputs/userstudy_n3_revised": "completed 28-case N=3 study output",
    "archive/outputs/userstudy_n3": "superseded 28-case study output",
    "outputs/experiments/n1_luna_20260910_v8": "legacy 28-case Luna N=1 result",
    "outputs/experiments/n1_terra_20260907": "legacy 28-case Terra N=1 result",
    "outputs/experiments/n1_luna_terra_identity_audit_20260912.json": "legacy 28-case identity audit",
    "outputs/current": "legacy 28-case lifecycle projections",
    "experiments/full_dataset_n1": "superseded standalone 28-case experiment definition",
    "tests/fixtures/legacy_n1": "legacy 28-case regression fixture",
}


def _present(path: Path) -> bool:
    """Detect files, directories, and broken Windows links without following targets."""
    try:
        return path.exists() or path.is_symlink()
    except OSError:
        return os.path.lexists(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list actions without changing files")
    parser.add_argument("--apply", action="store_true", help="perform reversible moves and cache cleanup")
    parser.add_argument("--legacy-only", action="store_true",
                        help="only remove the explicitly superseded 28-case surfaces")
    parser.add_argument("--prune-duplicate-history", action="store_true",
                        help="remove YiAPI local history copies already stored in archive/history_objects")
    parser.add_argument("--archive-root", type=Path, default=None,
                        help="destination for superseded runs (defaults to today's archive)")
    return parser.parse_args()


def plan(archive_root: Path, *, prune_duplicate_history: bool = False) -> list[dict]:
    actions = []
    for run_root in sorted((ROOT / "outputs").glob("yiapi_rebuild_*")):
        manifest_path = run_root / "archive_manifest.json"
        historical = run_root / "archive" / "historical"
        if not manifest_path.is_file() or not historical.is_dir():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        schema_version = manifest.get("schema_version")
        can_prune_v2 = (
            prune_duplicate_history
            and schema_version == "yiapi-archive-v2"
            and any(historical.iterdir())
        )
        if schema_version == "yiapi-archive-v1" or can_prune_v2:
            actions.append({
                "action": "compact_yiapi_history",
                "source": str(run_root.relative_to(ROOT)),
                "item_count": len(manifest.get("items", ())),
                "reason": "replace run-local history copies with shared content-addressed objects",
            })
    for name, reason in SUPERSEDED.items():
        source = ROOT / "outputs" / name
        if not _present(source) or source.is_symlink():
            continue
        destination = archive_root / name
        if destination.exists():
            raise RuntimeError(f"archive destination already exists: {destination}")
        actions.append(dict(action="move", source=str(source.relative_to(ROOT)),
                            destination=str(destination.relative_to(ROOT)), reason=reason))
    for relative in CACHE_DIRS:
        path = ROOT / relative
        if _present(path) and not path.is_symlink():
            actions.append(dict(action="remove_regenerable", source=relative,
                                reason="regenerable cache or build metadata"))
    for relative, reason in LEGACY_28_PATHS.items():
        path = ROOT / relative
        if _present(path):
            actions.append(dict(action="remove_legacy_28", source=relative, reason=reason))
    return actions


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compact_yiapi_history(run_root: Path, *, prune_duplicate_history: bool = False) -> None:
    if prune_duplicate_history:
        raise RuntimeError("History pruning is disabled pending independent content verification.")
    manifest_path = run_root / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in {"yiapi-archive-v1", "yiapi-archive-v2"}:
        return
    historical = run_root / "archive" / "historical"
    for item in manifest.get("items", ()):
        if not item.get("exists"):
            continue
        digest = str(item.get("destination_sha256", ""))
        if not digest or digest != item.get("source_sha256"):
            raise RuntimeError(f"unverified YiAPI history item in {manifest_path}: {item.get('destination', item.get('legacy_destination'))}")
        legacy_destination = str(item.get("destination") or item.get("legacy_destination") or "")
        local_path = historical / Path(legacy_destination).name
        if schema_version == "yiapi-archive-v2" and item.get("object_path"):
            shared_path = ROOT / Path(str(item["object_path"]).replace("\\", "/"))
            directory_object = shared_path if shared_path.is_dir() else HISTORY_OBJECTS / "directories" / digest
            file_object = shared_path if shared_path.is_file() else HISTORY_OBJECTS / "files" / digest
        else:
            directory_object = HISTORY_OBJECTS / "directories" / digest
            file_object = HISTORY_OBJECTS / "files" / digest
        if local_path.exists():
            shared_path = directory_object if local_path.is_dir() else file_object
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            if shared_path.exists():
                reused = True
                if prune_duplicate_history:
                    if local_path.is_dir() and not local_path.is_symlink():
                        shutil.rmtree(local_path)
                    else:
                        local_path.unlink()
            else:
                local_path.replace(shared_path)
                reused = False
        elif directory_object.exists() or file_object.exists():
            shared_path = directory_object if directory_object.exists() else file_object
            reused = True
        else:
            raise RuntimeError(f"missing YiAPI history item and shared object: {local_path}")
        item["legacy_destination"] = legacy_destination
        item.pop("destination", None)
        item["object_path"] = str(shared_path.relative_to(ROOT))
        item["storage"] = "shared_content_addressed"
        item["reused"] = reused
        item.setdefault("role", local_path.name)
    if historical.exists() and not any(historical.iterdir()):
        historical.rmdir()
    local_archive = run_root / "archive"
    if local_archive.exists() and not any(local_archive.iterdir()):
        local_archive.rmdir()
    manifest["schema_version"] = "yiapi-archive-v2"
    manifest["storage_root"] = str(HISTORY_OBJECTS.relative_to(ROOT))
    manifest["compacted_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    if args.prune_duplicate_history:
        raise SystemExit("History pruning is disabled pending independent content verification.")
    if not args.dry_run and not args.apply:
        raise SystemExit("选择 --dry-run 或 --apply；默认不修改文件。")
    archive_root = (
        args.archive_root
        or ROOT / "archive" / "outputs" / f"superseded_{datetime.now(timezone.utc):%Y%m%d}"
    ).resolve()
    actions = plan(
        archive_root,
        prune_duplicate_history=args.prune_duplicate_history,
    )
    if args.legacy_only:
        actions = [item for item in actions if item["action"] == "remove_legacy_28"]
    report = {
        "schema": "project-history-cleanup-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive_root": str(archive_root.relative_to(ROOT)),
        "preserved_active": ["outputs/claude_resume_eval",
                             "outputs/experiments", "experiments/userstudy",
                             "outputs/yiapi_rebuild_20260921_090434"],
        "actions": actions,
        "applied": bool(args.apply),
    }
    if args.apply:
        archive_root.mkdir(parents=True, exist_ok=True)
        for item in actions:
            source = ROOT / item["source"]
            if item["action"] == "compact_yiapi_history":
                _compact_yiapi_history(
                    source,
                    prune_duplicate_history=args.prune_duplicate_history,
                )
            elif item["action"] == "move":
                destination = ROOT / item["destination"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
            elif item["action"] == "remove_legacy_28":
                if source.is_symlink() or source.is_file():
                    source.unlink()
                else:
                    shutil.rmtree(source)
            else:
                shutil.rmtree(source)
        (archive_root / "CLEANUP_MANIFEST.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        (archive_root / "README.md").write_text(
            "# Superseded generated runs\n\n"
            "Moved from `outputs/` by `scripts/cleanup_project_history.py`. "
            "The manifest records the reversible moves; answer, execution and "
            "judgment files are retained.\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
