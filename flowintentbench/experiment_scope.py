"""One explicit case population shared by collection, evaluation and replay.

This validates experiment membership only. Scientific admission remains owned
by the case/evidence/GT and lifecycle contracts. No Cartesian product is made.
"""
from __future__ import annotations

from typing import Any, Mapping

PRIMARY_CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")


def case_inventory(manifest: Mapping[str, Any], *, require_dataset: bool = False) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest requires a non-empty explicit cases list")
    declared = manifest.get("case_count", len(rows))
    if type(declared) is not int or declared != len(rows):
        raise ValueError("manifest case_count must equal its explicit case membership")
    inventory: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest cases must be objects")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id != case_id.strip():
            raise ValueError("manifest case IDs must be non-empty normalized strings")
        if case_id in inventory:
            raise ValueError(f"manifest contains duplicate case IDs: {case_id}")
        if row.get("condition") not in PRIMARY_CONDITIONS:
            raise ValueError(f"unsupported condition for {case_id}: {row.get('condition')}")
        if require_dataset and (not isinstance(row.get("dataset_id"), str) or not row["dataset_id"].strip() or row["dataset_id"] != row["dataset_id"].strip()):
            raise ValueError(f"dataset identity missing for {case_id}")
        inventory[case_id] = row
    if require_dataset and "datasets" in manifest:
        declared_datasets = manifest["datasets"]
        actual = {row["dataset_id"] for row in rows}
        if (not isinstance(declared_datasets, list)
                or any(not isinstance(value, str) for value in declared_datasets)
                or len(declared_datasets) != len(set(declared_datasets))
                or set(declared_datasets) != actual):
            raise ValueError("manifest dataset set mismatch")
    return inventory
