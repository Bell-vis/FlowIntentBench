#!/usr/bin/env python3
"""Build the evidence-first scientific diversity and grounding phase bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench.concept_capability import build_dataset_concept_capability_map
from flowintentbench.curator_review import build_curator_review_packets
from flowintentbench.grounding import build_grounding_packets
from flowintentbench.portfolio import build_scientific_portfolio, load_portfolio_inputs
from flowintentbench.scientific_diversity import (
    build_dataset_acquisition_requirements,
    build_scientific_diversity_gap_analysis,
)
from flowintentbench.diversity_audit import build_diversity_report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, value: Mapping[str, Any]) -> None:
    lines = [f"# {title}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _portfolio_inputs(
    root: Path,
    portfolio_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_path = portfolio_root / "concept_family_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
    families = [dict(item) for item in inventory.get("families", []) if isinstance(item, Mapping)]
    _, cases, _ = load_portfolio_inputs(root)
    return families, [dict(item) for item in cases]


def build_phase(repository_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(repository_root), Path(output_root)
    # Refresh only the derived portfolio audit; authored cases and GT remain
    # untouched and all families retain their existing lifecycle labels.
    portfolio_output = output / "scientific_portfolio"
    build_scientific_portfolio(root, portfolio_output)
    families, cases = _portfolio_inputs(root, portfolio_output)
    historical_manifest = json.loads((root / "experiments/userstudy/case_manifest.json").read_text(encoding="utf-8"))
    historical_ids = {str(item.get("case_id")) for item in historical_manifest.get("cases", [])}
    historical_cases = [case for case in cases if str(case.get("case_id")) in historical_ids]
    candidate_cases = [case for case in cases if str(case.get("case_id")) not in historical_ids]
    candidate_family_ids = {str(item.get("family_id")) for item in families if str(item.get("concept_id")) != "high_speed_region"}
    candidate_families = [family for family in families if str(family.get("family_id")) in candidate_family_ids]
    formal_families = [family for family in families if str(family.get("release_status")) == "ELIGIBLE" and str(family.get("lifecycle_status")) == "RELEASE_ELIGIBLE"]

    historical_report = build_diversity_report([f for f in families if str(f.get("concept_id")) == "high_speed_region"], historical_cases, universe="HISTORICAL_CONTROLLED_PILOT")
    candidate_report = build_diversity_report(candidate_families, candidate_cases, universe="CONCEPT_EXPANSION_CANDIDATES")
    formal_report = build_diversity_report(formal_families, [], universe="FORMAL_RELEASE_CANDIDATES")
    baseline = {
        "status": "PROVISIONAL",
        "historical_controlled_pilot": {
            "case_count": len(historical_cases), "dataset_count": historical_report.get("dataset_count", 0),
            "concept_count": 1, "scientific_question_template_count": historical_report.get("scientific_question_template_count", 0),
            "analysis_archetype_count": historical_report.get("analysis_archetype_count", 0), "scientific_homogeneity": True,
        },
        "current_concept_expansion": {
            "case_count": len(candidate_cases), "concept_count": candidate_report.get("concept_count", 0),
            "dataset_count": candidate_report.get("dataset_count", 0), "analysis_archetype_count": candidate_report.get("analysis_archetype_count", 0),
            "status": "PROVISIONAL",
        },
        "formal_portfolio": {"released_family_count": len(formal_families), "status": "PORTFOLIO_NOT_READY" if not formal_families else "PROVISIONAL"},
        "populations_are_separate": True,
    }
    _write_json(output / "current_portfolio_baseline.json", baseline)
    _write_md(output / "current_portfolio_baseline.md", "Current Portfolio Baseline", baseline)

    capability_map = build_dataset_concept_capability_map(root)
    _write_json(output / "dataset_concept_capability_map.json", capability_map.to_dict())
    _write_md(output / "dataset_concept_capability_map.md", "Dataset x Concept Capability Map", capability_map.to_dict())

    gap = build_scientific_diversity_gap_analysis(candidate_families, candidate_cases)
    gap["historical_baseline"] = historical_report
    gap["candidate_portfolio"] = candidate_report
    gap["formal_portfolio"] = formal_report
    gap["DATASET_EXPANSION_REQUIRED"] = True
    _write_json(output / "scientific_diversity_gap_analysis.json", gap)
    _write_md(output / "scientific_diversity_gap_analysis.md", "Scientific Diversity Gap Analysis", gap)

    template_audit = {
        "historical": {key: historical_report.get(key) for key in ("scientific_question_template_count", "dominant_scientific_question_template", "dominant_scientific_question_fraction", "experimental_condition_template_count", "condition_balance")},
        "concept_expansion": {key: candidate_report.get(key) for key in ("scientific_question_template_count", "dominant_scientific_question_template", "dominant_scientific_question_fraction", "experimental_condition_template_count", "condition_balance")},
        "scientific_signatures_exclude_protocol_condition": True,
        "condition_signatures_are_protocol_diagnostics": True,
    }
    _write_json(output / "scientific_template_audit.json", template_audit)
    _write_md(output / "scientific_template_audit.md", "Scientific Question Template Audit", template_audit)

    packets = build_grounding_packets(root)
    for packet in packets:
        value = packet.to_dict()
        _write_json(output / "grounding_packets" / f"{packet.family_id}.json", value)
        _write_md(output / "grounding_packets" / f"{packet.family_id}.md", f"Grounding Packet: {packet.family_id}", value)

    screening_path = root / "artifacts/reference/concept_expansion_phase1/concept_screening.json"
    screening = json.loads(screening_path.read_text(encoding="utf-8")) if screening_path.is_file() else {"entries": []}
    cell_by_key = {(cell.dataset_id, cell.concept_id): cell for cell in capability_map.cells}
    screening_rows = []
    for original in screening.get("entries", []):
        row = dict(original)
        cell = cell_by_key.get((str(row.get("dataset_id")), str(row.get("concept_id"))))
        row["source_status"] = row.get("status")
        row["status"] = cell.decision if cell is not None else "EVIDENCE_GAP"
        row["major_blockers"] = list(cell.major_blockers) if cell is not None else ["capability cell not established"]
        screening_rows.append(row)
    screening_rows.append({"dataset_id": "Combustor", "concept_id": "high_density_threshold_clone", "status": "SCIENTIFICALLY_UNJUSTIFIED", "reason": "would duplicate the historical threshold-region decision structure"})
    screening_payload = {"status": "PROVISIONAL", "entries": screening_rows, "decisions_are_not_release_status": True}
    _write_json(output / "candidate_screening.json", screening_payload)
    _write_md(output / "candidate_screening.md", "Candidate Screening", screening_payload)

    acquisition = {"DATASET_EXPANSION_REQUIRED": True, "requirements": build_dataset_acquisition_requirements(), "reason": "Current seven datasets do not yet support the preferred orthogonal scientific coverage without lowering grounding standards."}
    _write_json(output / "dataset_acquisition_requirements.json", acquisition)
    _write_md(output / "dataset_acquisition_requirements.md", "Dataset Acquisition Requirements", acquisition)

    curator_packets = build_curator_review_packets([packet.to_dict() for packet in packets])
    for packet in curator_packets:
        value = packet.to_dict()
        _write_json(output / "curator_review" / f"{packet.family_id}.json", value)
        _write_md(output / "curator_review" / f"{packet.family_id}.md", f"Curator Review: {packet.family_id}", value)

    readiness = {
        "EVALUATOR_PRECALIBRATION_STATUS": "IMPLEMENTED_NOT_CALIBRATED",
        "SCIENTIFIC_DIVERSITY_CONSTRUCTION_STATUS": "PROVISIONAL_CANDIDATES_PREPARED",
        "GROUNDING_PACKET_STATUS": "PACKETS_PREPARED_PROVISIONAL",
        "READY_FOR_INDEPENDENT_GROUNDING_REVIEW": bool(packets),
        "FORMAL_PORTFOLIO_STATUS": "PORTFOLIO_NOT_READY" if not formal_families else "PROVISIONAL",
        "READY_FOR_JUDGE_CALIBRATION": False,
        "READY_FOR_SMALL_N3": False,
        "curator_status": "PENDING_EXPERT_REVIEW",
        "grounding_recommendations": {packet.family_id: packet.grounding_recommendation for packet in packets},
        "HISTORICAL_28_CASES_CHANGED": False,
        "MODEL_RUNS_PERFORMED": False,
    }
    _write_json(output / "grounding_readiness.json", readiness)
    report = {
        "A_implementation_follow_up": {"atomic_finding_audit": "See artifacts/reference/scientific_validation/atomic_finding_real_semantic_audit.json", "f2_contract_auto_loading": True},
        "B_scientific_template_audit": template_audit,
        "C_existing_candidates": {packet.family_id: {"grounding_recommendation": packet.grounding_recommendation, "unresolved_questions": list(packet.unresolved_questions)} for packet in packets},
        "D_new_candidate_search": [cell.to_dict() for cell in capability_map.cells if cell.concept_id != "high_speed_region"],
        "E_diversity_progress": {"concept_count": candidate_report.get("concept_count", 0), "dataset_count": candidate_report.get("dataset_count", 0), "archetype_count": candidate_report.get("analysis_archetype_count", 0), "entity_type_count": gap.get("scientific_entity_type_count", 0), "dominant_scientific_template_fraction": candidate_report.get("dominant_scientific_question_fraction", 0.0)},
        "F_grounding": readiness["grounding_recommendations"],
        "G_DATASET_EXPANSION_REQUIRED": True,
        "H_final_status": readiness,
    }
    _write_json(output / "report.json", report)
    lines = ["# Scientific Diversity Expansion and Grounding Preparation", "", "This is a provisional construction and evidence-preparation bundle; it does not perform model runs or curator confirmation.", ""]
    for section, value in report.items():
        lines.extend([f"## {section}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    (output / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "PASS", "output_root": str(output), "readiness": readiness}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/scientific_diversity_grounding")
    args = parser.parse_args()
    print(json.dumps(build_phase(args.repository_root, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
