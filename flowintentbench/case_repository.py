"""Stable readers for authored candidate case artifacts.

Historically these readers lived in the SCQ corrective-pass runner.  They are
small repository access helpers, so production code can use them without
depending on a historical audit module's private implementation functions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def condition_from_case_id(case_id: str) -> str | None:
    """Return the canonical ``O[123]-F[12]`` condition encoded in an id."""
    match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", str(case_id), re.I)
    return f"{match.group(1).upper()}-{match.group(2).upper()}" if match else None


def _family_index(root: Path) -> dict[str, dict[str, Any]]:
    inventory = _read(
        root / "artifacts/reference/scientific_portfolio/concept_family_inventory.json",
        {},
    )
    families = {
        str(item.get("family_id")): dict(item)
        for item in inventory.get("families", ())
        if isinstance(item, Mapping) and item.get("family_id")
    }
    for path in sorted(root.glob("datasets/*/construction/families.json")):
        for item in _read(path, {}).get("families", []):
            family_id = str(item["family_id"])
            if family_id in families:
                raise ValueError(f"ambiguous family identity: {family_id}")
            families[family_id] = {**item, "dataset_id": path.parents[1].name,
                                   "concept_id": item.get("concept_id", family_id)}
    return families


def _candidate_case_dirs(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.glob("artifacts/reference/concept_expansion_phase1/candidate_cases/*/*/case_construction_metadata.json")
    )


def _record_from_dir(root: Path, case_dir: Path, families: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metadata = _read(case_dir / "case_construction_metadata.json", {})
    case_id = str(metadata.get("case_id", case_dir.name))
    family_id = str(metadata.get("case_family_id", ""))
    case_input = _read(case_dir / "case_input.json", {})
    case_context = _read(case_dir / "case_context.json", case_input.get("case_context", {}))
    return {
        "case_id": case_id,
        "family_id": family_id,
        "dataset_id": str(metadata.get("dataset_id", "")),
        "condition": condition_from_case_id(case_id),
        "case_dir": str(case_dir),
        "path": str(case_dir.relative_to(root)) if case_dir.is_relative_to(root) else str(case_dir),
        "metadata": metadata,
        "case_input": case_input,
        "case_context": case_context,
        "ground_truth": _read(case_dir / "ground_truth.json", {}),
        "family": dict(families.get(family_id, {})),
    }


def load_candidate_case_records(repository_root: str | Path) -> list[dict[str, Any]]:
    """Load all Concept Expansion candidate case records in stable order."""
    root = Path(repository_root)
    families = _family_index(root)
    return [_record_from_dir(root, case_dir, families) for case_dir in _candidate_case_dirs(root)]


def load_case_record(repository_root: str | Path, case_id: str) -> dict[str, Any]:
    """Load one authored candidate case by id.

    A ``FileNotFoundError`` is raised rather than returning a partial record;
    callers use this at construction/review boundaries where missing input is
    an integrity failure.
    """
    root = Path(repository_root)
    families = _family_index(root)
    for case_dir in _candidate_case_dirs(root):
        record = _record_from_dir(root, case_dir, families)
        if record["case_id"] == str(case_id):
            return record
    # Historical 28-case artifacts use the same layout under datasets/.
    for metadata_path in sorted(root.glob(f"datasets/*/construction/cases/{case_id}/case_construction_metadata.json")):
        return _record_from_dir(root, metadata_path.parent, families)
    # Expanded construction output remains under datasets/. Only explicitly
    # requested identities use this fallback; the historical portfolio's
    # candidate enumeration is not expanded implicitly.
    matches = list(root.glob(f"datasets/*/candidate_cases/*/{case_id}/case_construction_metadata.json"))
    if len(matches) > 1:
        raise ValueError(f"ambiguous expanded case identity: {case_id}")
    if matches:
        return _record_from_dir(root, matches[0].parent, families)
    raise FileNotFoundError(f"case not found: {case_id}")


def family_view_for_case(
    record: Mapping[str, Any],
    *,
    responsibility: Any = ...,
    representability: Any = ...,
) -> dict[str, Any]:
    """Return the family view joined with case-level contracts.

    ``responsibility`` and ``representability`` are optional mutation hooks
    used by negative audits; omitted values preserve the authored metadata.
    """
    family = dict(record.get("family") or {})
    metadata = record.get("metadata") or {}
    family["responsibility_contract"] = (
        metadata.get("responsibility_contract") if responsibility is ... else responsibility
    )
    family["evaluation_representability_contract"] = (
        metadata.get("evaluation_representability_contract") if representability is ... else representability
    )
    return family


__all__ = [
    "condition_from_case_id",
    "family_view_for_case",
    "load_candidate_case_records",
    "load_case_record",
]
