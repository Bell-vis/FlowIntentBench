"""Freeze one original-answer bridge packet using the prior outcome review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import _dimensions, judgment_schema, scoring_prompt
from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import RunRecord
from scripts.evaluate_answered_outcomes import sha_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/original_metric_bridge_test")
    args = parser.parse_args()
    source = args.source.resolve()
    selection = read_json(source / "selection.json")
    selected = selection["answers"][0]
    result = read_json(source / "cases" / (selected["run_id"] + ".json"))
    request_id = result.get("request_id") or result.get("baseline_request_id")
    prior = read_json(source / "judgments" / (request_id + ".json"))["judgment"]
    manifest = {row["case_id"]: row for row in read_json(args.manifest)["cases"]}
    case_input, metadata, gt, _material = load_development_case(ROOT, manifest[selected["case_id"]])
    record_path = Path(selected["run_path"])
    record = RunRecord.from_dict(read_json(record_path))
    case_payload = case_input.model_dump(mode="json")
    case_payload.update(case_id=record.case_id,
                        principal_operationalization_dimensions=list(getattr(metadata, "principal_operationalization_dimensions", ())),
                        unresolved_operationalization_dimensions=list(getattr(metadata, "unresolved_operationalization_dimensions", ())))
    gt_packet = {"operationalizations": [{"branch_id": b.operationalization_id,
                                           "decisions": [{"dimension": getattr(d.dimension, "value", d.dimension)} for d in b.decisions]}
                                          for b in gt.acceptable_operationalizations], "branches": []}
    schema = judgment_schema(_dimensions(gt_packet))
    prompt = scoring_prompt(case_payload, record.final_response, gt)
    prefill = {"method_ratings": [item for item in prior.get("ratings", []) if item["item_id"].startswith("method:")],
               "branch_relations": prior.get("branch_relations", []), "claims": prior.get("claims", []),
               "claim_verification": result.get("verification"), "reviewer_source": result.get("reviewer_source")}
    prompt += ("\n\nPRIOR OUTCOME EVIDENCE (reuse as extraction aid only; do not map its labels to original booleans):\n"
               + json.dumps(prefill, ensure_ascii=False, separators=(",", ":")))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "packet.json", {"run_id": selected["run_id"], "case_id": selected["case_id"],
        "answer": record.final_response, "answer_sha256": selected["answer_sha256"], "prompt": prompt,
        "schema": schema, "source_run_sha256": sha_file(record_path), "prior_request_id": request_id,
        "prior_judgment_sha256": result["judgment_sha256"], "case_payload": case_payload,
        "gt": gt.model_dump(mode="json")})
    print(json.dumps({"run_id": selected["run_id"], "case_id": selected["case_id"],
                      "packet": str(args.output / "packet.json"), "schema_dimensions": _dimensions(gt_packet)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
