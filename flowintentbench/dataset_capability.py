"""Read-only dataset capability audit for scientific-case reselection.

The audit answers only whether a dataset has the basic reader, field, context,
and provenance inputs needed to author a candidate question.  It deliberately
does not promote a candidate to a formal case or infer Ground Truth support.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_RESELECTION_TARGETS: dict[str, dict[str, str]] = {
    "Blunt_Fin": {
        "condition": "O1-F1",
        "target": "the point of highest recorded speed in the airflow around the blunt fin",
        "question": "In the airflow around the blunt fin, where does the recorded speed reach its highest value, and what value is recorded there?",
        "operation": "velocity magnitude argmax with explicit spatial and scalar verification",
    },
    "Carotid": {
        "condition": "O1-F1",
        "target": "the point of highest supplied velocity magnitude in the carotid flow",
        "question": "Within the carotid artery flow, where is the supplied velocity magnitude highest, and what value is recorded at that location?",
        "operation": "Euclidean magnitude of the supplied velocity vectors followed by argmax",
    },
    "Combustor": {
        "condition": "O1-F1",
        "target": "the documented density 0.38 isosurface in the annular combustor",
        "question": "On the documented density isosurface at 0.38 in the annular combustor, where does the velocity magnitude reach its highest value, and what value is recorded there?",
        "operation": "fixed Density=0.38 contour followed by velocity-magnitude argmax on the surface",
    },
    "FireFlow": {
        "condition": "O1-F1",
        "target": "the point of highest supplied speed in the FireFlow room field",
        "question": "In the FireFlow room field, where does the supplied flow speed reach its highest value, and what value is recorded there?",
        "operation": "Euclidean magnitude of the supplied uvw vector followed by argmax",
    },
    "Kitchen": {
        "condition": "O1-F1",
        "target": "the point of highest stored turbulent kinetic energy in the kitchen flow",
        "question": "In the kitchen flow, where does the stored turbulent kinetic energy reach its highest value, and what value is recorded there?",
        "operation": "stored turbulent-kinetic-energy argmax with spatial and scalar verification",
    },
    "NASA_LOx_Post": {
        "condition": "O1-F1",
        "target": "the point of highest valid recorded speed around the LOx post",
        "question": "Around the post in the liquid-oxygen flow, where does the valid recorded speed reach its highest value, and what speed is recorded there?",
        "operation": "IBlank-valid velocity magnitude argmax",
    },
    "Office": {
        "condition": "O1-F1",
        "target": "the point of highest supplied speed in the office ventilation field",
        "question": "Within the office ventilation field, where is the supplied speed highest, and what value is recorded at that location?",
        "operation": "Euclidean magnitude of the supplied velocity vectors followed by argmax",
    },
}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_count(construction: Path) -> int:
    payload = _read_json(construction / "sources.json", {})
    if isinstance(payload, Mapping):
        payload = payload.get("sources", [])
    return len(payload) if isinstance(payload, list) else 0


def audit_dataset_capabilities(
    repository_root: str | Path,
    dataset_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit explicitly selected dataset inputs without modifying any artifact."""

    root = Path(repository_root).resolve()
    selected = tuple(dataset_ids or DATASET_RESELECTION_TARGETS)
    rows: list[dict[str, Any]] = []
    for dataset_id in selected:
        dataset_root = root / "datasets" / dataset_id
        manifest_path = dataset_root / "dataset_manifest.json"
        metadata_path = dataset_root / "data_metadata.json"
        construction = dataset_root / "construction"
        manifest = _read_json(manifest_path, {})
        metadata = _read_json(metadata_path, {})
        reader = manifest.get("reader", {}) if isinstance(manifest, Mapping) else {}
        variables = metadata.get("variables", []) if isinstance(metadata, Mapping) else []
        grid = metadata.get("grid", {}) if isinstance(metadata, Mapping) else {}
        source_urls = (
            manifest.get("provenance", {}).get("source_urls", [])
            if isinstance(manifest, Mapping)
            else []
        )
        files = manifest.get("files", []) if isinstance(manifest, Mapping) else []
        files_present = bool(files) and all(
            isinstance(item, Mapping)
            and (dataset_root / str(item.get("path", "")).split("/", 1)[-1]).is_file()
            for item in files
        )
        nonempty_payloads = bool(files) and all(
            isinstance(item, Mapping)
            and int(item.get("size_bytes", 0) or 0) > 0
            for item in files
        )
        reference_analysis_count = len(
            list(construction.glob("cases/*/reference_analysis.md"))
        )
        vector_variables = [
            str(item.get("name"))
            for item in variables
            if isinstance(item, Mapping) and item.get("type") == "vector"
        ]
        scalar_variables = [
            str(item.get("name"))
            for item in variables
            if isinstance(item, Mapping) and item.get("type") == "scalar"
        ]
        checks = {
            "manifest_present": manifest_path.is_file(),
            "metadata_present": metadata_path.is_file(),
            "canonical_reader_declared": bool(reader.get("canonical_reader")),
            "payload_files_present": files_present,
            "payload_files_nonempty": nonempty_payloads,
            "grid_or_coordinates_declared": isinstance(grid, Mapping) and bool(grid),
            "scalar_or_vector_field_declared": bool(vector_variables or scalar_variables),
            "traceable_source_declared": bool(source_urls) or _source_count(construction) > 0,
            "reference_analysis_pattern_present": reference_analysis_count > 0,
        }
        # Prior case analyses are useful history, not a prerequisite for
        # constructing the first case on a newly acquired dataset.
        eligible = all(value for key, value in checks.items() if key != "reference_analysis_pattern_present")
        target = DATASET_RESELECTION_TARGETS.get(dataset_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "reader_format": reader.get("format"),
                "canonical_reader": reader.get("canonical_reader"),
                "grid": grid,
                "vector_variables": vector_variables,
                "scalar_variables": scalar_variables,
                "source_url_count": len(source_urls),
                "construction_source_count": _source_count(construction),
                "reference_analysis_count": reference_analysis_count,
                "checks": checks,
                "data_capability_status": "PASS" if eligible else "FAIL",
                "candidate_release_status": "CANDIDATE_ONLY",
                "formal_release_status": "NOT_READY_WITHOUT_CASE_GT_EVIDENCE_AND_CURATOR",
                "recommended_condition": target.get("condition"),
                "recommended_target": target.get("target"),
                "recommended_question": target.get("question"),
                "recommended_operation": target.get("operation"),
                "payload_fingerprints": {
                    str(item.get("path")): _sha256(
                        dataset_root / str(item.get("path", "")).split("/", 1)[-1]
                    )
                    for item in files
                    if isinstance(item, Mapping)
                    and (dataset_root / str(item.get("path", "")).split("/", 1)[-1]).is_file()
                },
            }
        )
    return {
        "schema_version": "dataset-capability-audit-v1",
        "artifact_type": "DATASET_CAPABILITY_AUDIT",
        "dataset_count": len(rows),
        "all_data_capability_pass": all(row["data_capability_status"] == "PASS" for row in rows),
        "official_release_ready_count": 0,
        "official_release_note": "Capability PASS authorizes candidate construction only; it does not satisfy evidence, GT binding, SRAC, or accountable curator gates.",
        "datasets": rows,
    }


__all__ = ["DATASET_RESELECTION_TARGETS", "audit_dataset_capabilities"]
