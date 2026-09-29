"""Build the explicit 28-case manifest for the full-dataset N=1 pilot.

The manifest is intentionally boring and explicit: it is generated from a
fixed dataset/case table, never from directory discovery.  It is a freeze
boundary for the upcoming pilot and contains only artifact identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.experiment_scope import case_inventory


DATASET_CASES: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Blunt_Fin", (("blunt_fin_o1_f1", "O1-F1"), ("blunt_fin_o2_f1", "O2-F1"), ("blunt_fin_o3_f1", "O3-F1"), ("blunt_fin_o1_f2", "O1-F2"))),
    ("Carotid", (("carotid_o1_f1", "O1-F1"), ("carotid_o2_f1", "O2-F1"), ("carotid_o3_f1", "O3-F1"), ("carotid_o1_f2", "O1-F2"))),
    ("Combustor", (("combustor_o1_f1", "O1-F1"), ("combustor_o2_f1", "O2-F1"), ("combustor_o3_f1", "O3-F1"), ("combustor_o1_f2", "O1-F2"))),
    ("FireFlow", (("fireflow_o1_f1", "O1-F1"), ("fireflow_o2_f1", "O2-F1"), ("fireflow_o3_f1", "O3-F1"), ("fireflow_o1_f2", "O1-F2"))),
    ("Kitchen", (("kitchen_o1_f1", "O1-F1"), ("kitchen_o2_f1", "O2-F1"), ("kitchen_o3_f1", "O3-F1"), ("kitchen_o1_f2", "O1-F2"))),
    ("NASA_LOx_Post", (("nasa_lox_post_o1_f1", "O1-F1"), ("nasa_lox_post_o2_f1", "O2-F1"), ("nasa_lox_post_o3_f1", "O3-F1"), ("nasa_lox_post_o1_f2", "O1-F2"))),
    ("Office", (("office_speed_zones_o1_f1", "O1-F1"), ("office_speed_zones_o2_f1", "O2-F1"), ("office_speed_zones_o3_f1", "O3-F1"), ("office_speed_zones_o1_f2", "O1-F2"))),
)
CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")
PILOT_LABEL = "FULL-DATASET N=1 INTEGRATION PILOT / NOT FORMAL BENCHMARK RESULT"
REFERENCE_SCHEMAS = {
    "reference-science-baseline-v4": "IMMUTABLE_REFERENCE_V4",
    "reference-science-baseline-v5": "IMMUTABLE_REFERENCE_V5",
    "reference-science-baseline-v6": "IMMUTABLE_REFERENCE_V6",
    "reference-science-baseline-v7": "IMMUTABLE_REFERENCE_V7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, relative: str | Path) -> dict[str, str]:
    """Return a repository-relative (when possible) artifact identity.

    The historical pilot uses paths below ``datasets/`` while the current
    qualification output lives below ``outputs/current``.  Joining an
    absolute path with ``root`` is safe, so this helper can serve both trees
    without silently discovering a different case.
    """

    candidate = Path(relative)
    path = candidate if candidate.is_absolute() else root / candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        display_path = str(path.resolve().relative_to(root))
    except ValueError:
        # External artifacts are allowed only when explicitly named by the
        # source manifest; retain the absolute path so preflight can reject
        # or explicitly bind it rather than guessing a relative location.
        display_path = str(path.resolve())
    return {"path": display_path, "sha256": sha256_file(path)}


def _load_source_manifest(repo: Path, value: str | Path) -> tuple[Path, Mapping[str, Any]]:
    path = Path(value)
    if not path.is_absolute():
        path = repo / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("source manifest must be a JSON object")
    rows = payload.get("cases")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("source manifest must contain a cases list")
    return path, payload


def _source_case_rows(repo: Path, source_manifest: str | Path) -> tuple[Path, list[Mapping[str, Any]]]:
    path, payload = _load_source_manifest(repo, source_manifest)
    rows = list(payload["cases"])
    case_inventory({"cases": rows, "case_count": payload.get("case_count", len(rows))}, require_dataset=True)
    return path, rows  # type: ignore[return-value]


def build_manifest(
    *,
    repository_root: str | Path = ROOT,
    source_manifest: str | Path | None = None,
    require_agent_ready: bool = False,
) -> dict[str, Any]:
    """Build the explicit N=1 manifest.

    With no ``source_manifest`` this preserves the frozen historical 7x4
    controlled-pilot table.  When supplied, every case path and identity is
    taken from that explicit manifest (currently the qualified
    ``agent_ready_manifest.json``), so a renamed/provisional case cannot be
    accidentally replaced by a stale directory-table entry.
    """

    repo = Path(repository_root).resolve()
    cases: list[dict[str, Any]] = []
    position = 0
    source_path: Path | None = None
    source_rows: list[Mapping[str, Any]] | None = None
    source_payload: Mapping[str, Any] | None = None
    if source_manifest is not None:
        source_path, source_payload = _load_source_manifest(repo, source_manifest)
        source_rows = list(source_payload["cases"])
        case_inventory({"cases": source_rows, "case_count": source_payload.get("case_count", len(source_rows))}, require_dataset=True)
        if require_agent_ready:
            blocked = [
                str(row.get("case_id", ""))
                for row in source_rows
                if (
                    (
                        source_payload.get("schema_version") in REFERENCE_SCHEMAS
                        and row.get("reference_case_gt_frozen") is not True
                    )
                    or (
                        source_payload.get("schema_version") not in REFERENCE_SCHEMAS
                        and str(row.get("agent_ready_status", "")).upper() != "AGENT_READY"
                    )
                )
            ]
            if blocked:
                raise ValueError(
                    "authoritative source manifest contains non-agent-ready cases: "
                    + ", ".join(blocked)
                )
        # Preserve the historical 7x4 execution slot order.  This is an
        # explicit compatibility rule for resume/trajectory indexing, not a
        # directory discovery mechanism; the source rows are still the sole
        # authority for every artifact identity.
        source_rows.sort(
            key=lambda row: (
                str(row.get("dataset_id", "")),
                CONDITIONS.index(str(row.get("condition", "")))
                if str(row.get("condition", "")) in CONDITIONS
                else len(CONDITIONS),
                str(row.get("case_id", "")),
            )
        )
        case_specs = tuple(
            (str(row.get("case_id", "")), str(row.get("condition", "")))
            for row in source_rows
        )
        source_items: Sequence[Mapping[str, Any]] = source_rows
    else:
        case_specs = tuple(spec for _, specs in DATASET_CASES for spec in specs)
        source_items = tuple(
            {
                "dataset_id": dataset_id,
                "case_id": case_id,
                "condition": condition,
                "case_path": f"datasets/{dataset_id}/construction/cases/{case_id}",
            }
            for dataset_id, specs in DATASET_CASES
            for case_id, condition in specs
        )
    seen_case_ids: set[str] = set()
    for source_item, (fallback_case_id, fallback_condition) in zip(source_items, case_specs):
        dataset_id = str(source_item.get("dataset_id", "")).strip()
        case_id = str(source_item.get("case_id", fallback_case_id)).strip()
        condition = str(source_item.get("condition", fallback_condition)).strip()
        if not dataset_id or not case_id or not condition:
            raise ValueError("each source manifest case needs dataset_id, case_id, and condition")
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id in source manifest: {case_id}")
        seen_case_ids.add(case_id)
        if condition not in CONDITIONS:
            raise ValueError(f"unsupported condition in source manifest: {condition}")
        source_case_path = source_item.get("case_path")
        if source_case_path is None:
            source_case_path = Path(str(source_item.get("case_input_path", ""))).parent
        # A frozen reference manifest intentionally stores artifact digests,
        # not mutable workspace paths.  Resolve its case under the physical
        # snapshot, never by directory discovery or case-id search.
        if (
            source_payload is not None
            and source_payload.get("schema_version") in REFERENCE_SCHEMAS
            and not source_item.get("case_path")
            and not source_item.get("case_input_path")
        ):
            source_case_path = (
                source_path.parent
                / "cases"
                / dataset_id
                / case_id
            )
        case_dir_path = Path(str(source_case_path))
        if not case_dir_path.is_absolute():
            case_dir_path = (repo / case_dir_path).resolve()
        paths = {
            "case_input": source_item.get("case_input_path") or case_dir_path / "case_input.json",
            "case_construction_metadata": source_item.get("case_construction_metadata_path") or case_dir_path / "case_construction_metadata.json",
            "ground_truth": source_item.get("ground_truth_path") or case_dir_path / "ground_truth.json",
            "case_context": source_item.get("case_context_path") or case_dir_path / "case_context.json",
            "dataset_manifest": source_item.get("dataset_manifest_path") or f"datasets/{dataset_id}/dataset_manifest.json",
        }
        artifacts = {name: _artifact(repo, rel) for name, rel in paths.items()}
        row = {
            "dataset_id": dataset_id,
            "case_id": case_id,
            "condition": condition,
            "case_execution_position": position,
            "trial_index": 1,
            "case_input": artifacts["case_input"]["path"],
            "case_input_path": artifacts["case_input"]["path"],
            "case_input_sha256": artifacts["case_input"]["sha256"],
            "case_construction_metadata": artifacts["case_construction_metadata"]["path"],
            "case_construction_metadata_path": artifacts["case_construction_metadata"]["path"],
            "case_construction_metadata_sha256": artifacts["case_construction_metadata"]["sha256"],
            "ground_truth": artifacts["ground_truth"]["path"],
            "ground_truth_path": artifacts["ground_truth"]["path"],
            "ground_truth_sha256": artifacts["ground_truth"]["sha256"],
            "case_context": artifacts["case_context"]["path"],
            "case_context_path": artifacts["case_context"]["path"],
            "case_context_sha256": artifacts["case_context"]["sha256"],
            "dataset_manifest": artifacts["dataset_manifest"]["path"],
            "dataset_manifest_path": artifacts["dataset_manifest"]["path"],
            "dataset_manifest_sha256": artifacts["dataset_manifest"]["sha256"],
        }
        # Preserve explicit qualification identity and source provenance
        # without making any of it model-visible.
        if source_manifest is not None:
            reference_bound = bool(
                source_payload is not None
                and source_payload.get("schema_version") in REFERENCE_SCHEMAS
            )
            row["case_artifact_authority"] = (
                (REFERENCE_SCHEMAS[source_payload.get("schema_version")] if reference_bound else "AGENT_READY_CONSTRUCTION")
            )
            row.update(
                {
                    "case_source": "authoritative_source_manifest",
                    "source_manifest_path": str(source_path.relative_to(repo))
                    if source_path is not None and source_path.is_relative_to(repo)
                    else str(source_path),
                    "source_manifest_sha256": sha256_file(source_path) if source_path else None,
                    "source_case_path": str(source_item.get("source_case_pattern", "")) or None,
                    "source_agent_ready_status": source_item.get("agent_ready_status"),
                    "source_official_release_ready": source_item.get("official_release_ready"),
                }
            )
            # ContextSelection remains a construction-side artifact.  A
            # qualified case may inherit it from its explicitly recorded
            # source-case pattern; never search by case ID.
            candidates = [case_dir_path / "context_selection.json"]
            source_pattern = str(source_item.get("source_case_pattern", "")).strip()
            if source_pattern:
                candidates.append(repo / source_pattern / "context_selection.json")
            for candidate in candidates:
                if candidate.is_file():
                    context_artifact = _artifact(repo, candidate)
                    row["context_selection_path"] = context_artifact["path"]
                    row["context_selection_sha256"] = context_artifact["sha256"]
                    break
            evidence_candidate = case_dir_path / "agent_ready_evidence.json"
            if evidence_candidate.is_file():
                evidence_artifact = _artifact(repo, evidence_candidate)
                row["evidence_path"] = evidence_artifact["path"]
                row["evidence_sha256"] = evidence_artifact["sha256"]
            sec_path = source_item.get("scientific_evaluation_contract_path")
            if sec_path is None and reference_bound:
                sec_path = case_dir_path / "scientific_evaluation_contract.json"
            if sec_path:
                sec_artifact = _artifact(repo, sec_path)
                row["scientific_evaluation_contract_path"] = sec_artifact["path"]
                row["scientific_evaluation_contract_sha256"] = sec_artifact["sha256"]
        cases.append(row)
        position += 1
    case_inventory({"cases": cases}, require_dataset=True)
    return {
        "manifest_version": "full-dataset-n1-v1",
        "pilot_label": PILOT_LABEL,
        "case_count": len(cases),
        "N": 1,
        "formal": False,
        "case_source": "authoritative_source_manifest" if source_manifest is not None else "frozen_controlled_pilot_table",
        "case_artifact_authority": (
            REFERENCE_SCHEMAS[source_payload.get("schema_version")]
            if source_payload is not None
            and source_payload.get("schema_version") in REFERENCE_SCHEMAS
            else ("AGENT_READY_CONSTRUCTION" if source_manifest is not None else "HISTORICAL_CONTROLLED_PILOT")
        ),
        "reference_manifest_path": (
            str(source_path.relative_to(repo))
            if source_path is not None and source_path.is_relative_to(repo)
            else None
        ) if source_payload is not None and source_payload.get("schema_version") in REFERENCE_SCHEMAS else None,
        "reference_manifest_sha256": (
            sha256_file(source_path) if source_path is not None else None
        ) if source_payload is not None and source_payload.get("schema_version") in REFERENCE_SCHEMAS else None,
        "reference_schema_version": (
            source_payload.get("schema_version") if source_payload is not None else None
        ) if source_payload is not None and source_payload.get("schema_version") in REFERENCE_SCHEMAS else None,
        "source_manifest_path": (
            str(source_path.relative_to(repo))
            if source_path is not None and source_path.is_relative_to(repo)
            else (str(source_path) if source_path is not None else None)
        ),
        "source_manifest_sha256": sha256_file(source_path) if source_path is not None else None,
        "datasets": list(dict.fromkeys(str(item["dataset_id"]) for item in cases)),
        "conditions": list(CONDITIONS),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/userstudy/case_manifest.json")
    parser.add_argument("--force", action="store_true", help="explicitly replace an existing manifest during regeneration")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="explicit authoritative case manifest (for example outputs/current/qualified_scientific_cases/agent_ready_manifest.json)",
    )
    parser.add_argument(
        "--require-agent-ready",
        action="store_true",
        help="fail if an explicit source manifest contains a non-AGENT_READY case",
    )
    args = parser.parse_args()
    payload = build_manifest(
        repository_root=ROOT,
        source_manifest=args.source_manifest,
        require_agent_ready=args.require_agent_ready,
    )
    destination = args.output if args.output.is_absolute() else ROOT / args.output
    if destination.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite frozen manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
