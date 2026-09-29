"""Content-addressed storage helpers for large immutable history trees."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(tree_sha256(child).encode())
            digest.update(b"\n")
        return digest.hexdigest()
    return "MISSING"


def display_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.relative_to(repository_root))
    except ValueError:
        return str(path)


def store_snapshot(source: Path, *, repository_root: Path, object_root: Path) -> dict[str, Any]:
    """Store one immutable copy per content hash and return its manifest row."""
    digest = tree_sha256(source)
    row = {
        "source": display_path(source, repository_root),
        "source_sha256": digest,
        "exists": source.exists(),
        "storage": "shared_content_addressed",
    }
    if not source.exists():
        return row
    kind = "directories" if source.is_dir() else "files"
    destination = object_root / kind / digest
    row["object_path"] = display_path(destination, repository_root)
    row["destination_sha256"] = digest
    if destination.exists():
        row["reused"] = True
        return row
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary) if temporary.is_dir() else temporary.unlink()
    if source.is_dir():
        shutil.copytree(source, temporary)
    else:
        shutil.copy2(source, temporary)
    try:
        temporary.replace(destination)
    except FileExistsError:
        shutil.rmtree(temporary) if temporary.is_dir() else temporary.unlink()
        row["reused"] = True
        return row
    row["reused"] = False
    return row
