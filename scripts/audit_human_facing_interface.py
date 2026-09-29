#!/usr/bin/env python3
"""Run the deterministic human-facing interface audit without model calls.

This is intentionally not the live Flow Expert closure.  It checks that each
matrix primary preserves the frozen semantic projection and passes the existing
HCCQ/responsibility lint, while leaving live authoring and final expert review
explicitly pending.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.human_facing_acceptance import (  # noqa: E402
    audit_question_responsibility_realization,
)
from flowintentbench.human_facing_closure import audit_actual_hccq  # noqa: E402
from flowintentbench.question_presentation import (  # noqa: E402
    validate_question_semantic_fidelity,
)


def _primary(row: dict) -> dict | None:
    candidates = row.get("question_candidates") or []
    primary_id = row.get("primary_candidate_id") or row.get(
        "accepted_question_candidate_id"
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and (
            not primary_id or candidate.get("candidate_id") == primary_id
        ):
            return candidate
    return None


def run(matrix_path: str | Path, output_path: str | Path | None = None) -> dict:
    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    rows = matrix.get("conditions")
    if not isinstance(rows, list):
        raise ValueError("matrix conditions must be a list")
    audits = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate = _primary(row)
        projection = row.get("canonical_semantic_projection")
        if not isinstance(candidate, dict) or not isinstance(projection, dict):
            audits.append({"slot_id": row.get("slot_id"), "status": "FAIL", "error": "PRIMARY_OR_PROJECTION_MISSING"})
            continue
        finding = candidate.get("requested_findings", projection.get("finding_responsibility", {}))
        responsibility = audit_question_responsibility_realization(
            projection,
            str(candidate.get("model_visible_text", "")),
            candidate={
                "fixed_operationalization": candidate.get("fixed_operationalization", projection.get("resolved_operationalization", [])),
                "unresolved_operationalization_dimensions": candidate.get("unresolved_operationalization_dimensions", projection.get("unresolved_operationalization_dimensions", [])),
                "finding_responsibility": candidate.get("finding_responsibility", projection.get("finding_responsibility")),
                "requested_findings": finding,
            },
            semantic_visibility=row.get("semantic_visibility"),
        )
        hccq = audit_actual_hccq(
            str(candidate.get("model_visible_text", "")),
            semantic_projection=projection,
            semantic_visibility=row.get("semantic_visibility"),
        )
        fidelity = validate_question_semantic_fidelity(
            projection,
            candidate,
            semantic_contract_sha256=str(row.get("semantic_contract_sha256", "")),
        )
        audits.append({
            "slot_id": row.get("slot_id"),
            "dataset_id": row.get("dataset_id"),
            "condition": row.get("condition"),
            "responsibility_status": responsibility.get("status"),
            "hccq_status": hccq.get("status"),
            "semantic_fidelity_status": fidelity.get("status"),
            "failure_codes": sorted(set(responsibility.get("failure_codes", [])) | set(fidelity.get("failure_codes", []))),
        })
    passed = sum(
        row.get("responsibility_status") == "PASS"
        and row.get("hccq_status") == "PASS"
        and row.get("semantic_fidelity_status") == "PASS"
        for row in audits
    )
    result = {
        "record_type": "HumanFacingInterfaceAudit",
        "schema_version": "human-facing-interface-audit-v1",
        "execution_mode": "OFFLINE_DETERMINISTIC_AUDIT",
        "no_model_calls": True,
        "question_slot_count": len(rows),
        "interface_pass_count": passed,
        "interface_failure_count": len(audits) - passed,
        "live_flow_expert_review_status": "NOT_EXECUTED",
        "human_final_selection_status": "PENDING",
        "status": "PASS" if len(rows) == 28 and passed == 28 else "FAIL",
        "conditions": audits,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    default_matrix = ROOT / "outputs/current/audits/scientific_question_candidate_matrix_v3.json"
    if not default_matrix.is_file():
        default_matrix = ROOT / "outputs/current/scientific_question_matrix/scientific_question_manifest.json"
    parser.add_argument("--matrix", type=Path, default=default_matrix)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.matrix, args.output)
    print(json.dumps({key: result[key] for key in ("execution_mode", "question_slot_count", "interface_pass_count", "interface_failure_count", "live_flow_expert_review_status", "status")}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
