"""Single authoritative reader for persisted evaluation records.

Readers follow ``current_evaluation_pointer.json``.  Retry attempts are
immutable and are never selected by a directory glob.  The pointer keeps the
legacy ``current_file`` field for compatibility and may additionally name a
selected immutable attempt through ``selected_file``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def resolve_current_evaluation_record_path(run_dir: str | Path) -> Path | None:
    directory = Path(run_dir)
    pointer = directory / "current_evaluation_pointer.json"
    if pointer.is_file():
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("current evaluation pointer is unreadable") from exc
        if not isinstance(value, dict):
            raise ValueError("current evaluation pointer must be an object")
        filename = value.get("selected_file") or value.get("current_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError("current evaluation pointer has no selected file")
        candidate = (directory / filename).resolve()
        if directory.resolve() not in candidate.parents:
            raise ValueError("current evaluation pointer escapes run directory")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest() if candidate.is_file() else None
        if digest == value.get("sha256"):
            return candidate
        # A retry may have replaced the legacy canonical mirror before its
        # pointer swap (or the old canonical name may have been pending-only).
        # An atomically recorded fallback keeps the prior selected attempt
        # readable through either crash window.
        fallback_name = value.get("fallback_file")
        fallback_hash = value.get("fallback_sha256")
        if isinstance(fallback_name, str) and isinstance(fallback_hash, str):
            fallback = (directory / fallback_name).resolve()
            if directory.resolve() in fallback.parents and fallback.is_file():
                if hashlib.sha256(fallback.read_bytes()).hexdigest() == fallback_hash:
                    return fallback
        # A selected immutable attempt may be removed only after a committed
        # pointer update.  If the pointer is corrupt, fail closed; never fall
        # back to a same-named file with a different digest.
        if not candidate.is_file():
            raise ValueError("current evaluation pointer targets a missing record")
        raise ValueError("current evaluation pointer digest mismatch")
    final = directory / "case_evaluation_record.json"
    pending = directory / "pending_case_evaluation.json"
    if final.is_file() and pending.is_file():
        raise ValueError("legacy evaluation directory has ambiguous current records")
    if final.is_file():
        return final
    if pending.is_file():
        return pending
    return None


def iter_current_evaluation_records(root: str | Path):
    """Yield pointer-selected record paths below *root* in stable order."""
    base = Path(root)
    filenames = (
        "current_evaluation_pointer.json",
        "case_evaluation_record.json",
        "pending_case_evaluation.json",
    )
    records: dict[str, list[Path]] = {name: [] for name in filenames}
    # Retry archives can dwarf the current collection. Walk it once and
    # resolve only attempts explicitly selected by a live pointer.
    for directory, subdirectories, files in os.walk(base):
        subdirectories[:] = [name for name in subdirectories if name != "evaluation_attempts"]
        for name in filenames:
            if name in files:
                records[name].append(Path(directory) / name)
    selected_dirs: set[Path] = set()
    for pointer in sorted(records["current_evaluation_pointer.json"]):
        selected = resolve_current_evaluation_record_path(pointer.parent)
        if selected is not None:
            selected_dirs.add(pointer.parent.resolve())
            yield selected
    # Legacy collections have no pointer.  Keep a compatibility fallback, but
    # never glob archived ``evaluation_attempts`` or silently choose between
    # two canonical records.
    for candidate in sorted(records["case_evaluation_record.json"]):
        directory = candidate.parent.resolve()
        if directory in selected_dirs or "evaluation_attempts" in directory.parts:
            continue
        if (directory / "current_evaluation_pointer.json").is_file():
            continue
        # A legacy retry can leave a stale pending sidecar beside the
        # finalized canonical record.  The terminal record is authoritative
        # for the same run; pointer-based directories remain strict above.
        yield candidate
    for candidate in sorted(records["pending_case_evaluation.json"]):
        directory = candidate.parent.resolve()
        if directory in selected_dirs or "evaluation_attempts" in directory.parts:
            continue
        if (directory / "current_evaluation_pointer.json").is_file():
            continue
        if (directory / "case_evaluation_record.json").is_file():
            continue
        yield candidate


def load_current_evaluation_record(run_dir: str | Path):
    from .evaluator import load_case_evaluation_record

    path = resolve_current_evaluation_record_path(run_dir)
    if path is None:
        return None
    return load_case_evaluation_record(path)


__all__ = [
    "iter_current_evaluation_records",
    "load_current_evaluation_record",
    "resolve_current_evaluation_record_path",
]
