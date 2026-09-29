"""Resolve repository-bound paths preserved by a run moved to another host."""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath


REPOSITORY_NAMES = frozenset({"FlowIntentBench", "FlowIntentBench_optimized"})


@lru_cache(maxsize=16)
def _read_aliases(path: Path, mtime_ns: int, size: int) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "repository-path-aliases-v1":
        raise ValueError(f"Unsupported repository path index: {path}")
    aliases = document.get("aliases", {})
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in aliases.items()
    ):
        raise ValueError(f"Invalid repository path aliases: {path}")
    return aliases


def _repository_target(root: Path, parts: tuple[str, ...]) -> Path:
    relative = PurePosixPath(*parts).as_posix()
    index = root / "config/path_aliases.json"
    if index.is_file():
        stamp = index.stat()
        relative = _read_aliases(index, stamp.st_mtime_ns, stamp.st_size).get(relative, relative)
    normalized = relative.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        raise ValueError("Repository alias must be relative")
    candidate = root.joinpath(*PurePosixPath(normalized).parts).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Repository path escapes the current checkout")
    return candidate


def _normalized_parts(value: str | os.PathLike[str]) -> tuple[str, ...]:
    normalized = os.fspath(value).replace("\\", "/")
    return tuple(part for part in PurePosixPath(normalized).parts if part not in {"/", ""})


def resolve_repository_path(
    value: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    relative_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a path while preserving immutable, old-host provenance strings.

    Repository-bound paths always use this checkout, even when an old checkout
    still exists. Explicit aliases locate frozen files moved during archival;
    source records and their hashes remain unchanged. Relative paths use the
    supplied run directory first, then the repository root, never an ambient cwd.
    """

    root = Path(repository_root).resolve()
    raw = os.path.expanduser(os.fspath(value)).replace("\\", "/")
    path = Path(raw)
    parts = _normalized_parts(raw)
    foreign_absolute = raw.startswith("/") or (
        len(raw) >= 3 and raw[1] == ":" and raw[2] == "/"
    )
    # Prefer the full current-root prefix over a name shared by a parent folder.
    root_parts = _normalized_parts(root)
    if parts[:len(root_parts)] == root_parts:
        return _repository_target(root, parts[len(root_parts):])
    anchors = REPOSITORY_NAMES | {root.name}
    if foreign_absolute:
        for index, part in enumerate(parts):
            if part in anchors:
                return _repository_target(root, parts[index + 1:])
        return path
    if relative_root is not None:
        base = Path(relative_root)
        if not base.is_absolute():
            base = root / base
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
        repository_candidate = _repository_target(root, parts)
        return repository_candidate if repository_candidate.exists() else candidate
    return _repository_target(root, parts)


__all__ = ["resolve_repository_path"]
