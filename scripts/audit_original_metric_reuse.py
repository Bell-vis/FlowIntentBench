"""Audit whether outcome-rubric judgments can be reused for frozen O/URS/F/C metrics.

This intentionally produces no converted scores. It records reusable evidence
and the missing fields required by the original metric contract.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.expansion_evaluation import read_json
from flowintentbench.external_file_evaluator import write_json
from scripts.evaluate_answered_outcomes import sha_file
from scripts.benchmark_runtime import require_benchmark_runtime

METRICS = (
    "o_score", "urs", "resolved_o_compliance", "finding_precision",
    "core_finding_recall", "finding_requirement_recall", "adequate_core_complete",
    "c_score", "branch_alignment",
)


def audit(source):
    selection = read_json(source / "selection.json")
    rows = []
    for selected in selection["answers"]:
        result = read_json(source / "cases" / (selected["run_id"] + ".json"))
        request_id = result.get("request_id") or result.get("baseline_request_id")
        request = read_json(source / "requests" / (request_id + ".json"))
        judgment = read_json(source / "judgments" / (request_id + ".json"))["judgment"]
        method_items = [item for item in judgment["ratings"] if item["item_id"].startswith("method:")]
        claims = judgment.get("claims", [])
        relations = judgment.get("branch_relations", [])
        reusable = {
            "answer_sha256": selected["answer_sha256"],
            "source_judgment_sha256": result["judgment_sha256"],
            "method_item_verdicts": {item["item_id"][len("method:"):]: item["verdict"] for item in method_items},
            "method_evidence": {item["item_id"][len("method:"):]: item.get("evidence_text", "") for item in method_items},
            "branch_relations": relations,
            "claims": claims,
            "claim_verification": result["verification"],
        }
        missing = {
            "o_score": "per-branch binary match for every principal dimension; outcome ratings are not per-branch and allow PARTIAL",
            "urs": "per-branch binary match for every unresolved dimension",
            "resolved_o_compliance": "per-branch binary match for every resolved dimension",
            "finding_precision": "original eligibility plus one-to-one role/semantic assignment for every extracted finding",
            "core_finding_recall": "original branch-specific core finding assignment",
            "finding_requirement_recall": "mandatory/alternative role assignment under the original finding contract",
            "adequate_core_complete": "complete adequate-core-set evaluation under F2 contract",
            "c_score": "per-finding consistency judgments after selecting the original O branch",
            "branch_alignment": "original best-O and best-finding branch sets",
        }
        rows.append({"run_id": selected["run_id"], "case_id": selected["case_id"],
                     "condition": selected["condition"], "model_id": selected["model_id"],
                     "reviewer_source": result.get("reviewer_source"),
                     "reusable_evidence": reusable, "exact_metric_reuse": {metric: False for metric in METRICS},
                     "required_additional_fields": missing})
    report = {
        "protocol": "original-metric-reuse-audit-v1",
        "source": str(source),
        "source_selection_sha256": sha_file(source / "selection.json"),
        "answers": len(rows),
        "exact_conversion": False,
        "reason": "Outcome-rubric-v2 does not contain the original branch-level binary, eligibility, assignment and consistency judgments.",
        "metric_contract": "unchanged; flowintentbench.evaluation_metrics + flowintentbench.core_case_scoring",
        "reusable_evidence_counts": {
            "answer_identity": len(rows),
            "method_verdict_sets": sum(bool(row["reusable_evidence"]["method_item_verdicts"]) for row in rows),
            "complete_branch_relations": sum(bool(row["reusable_evidence"]["branch_relations"]) for row in rows),
            "extracted_claims": sum(len(row["reusable_evidence"]["claims"]) for row in rows),
            "claim_verification_records": sum(sum(row["reusable_evidence"]["claim_verification"][scope]["total"]
                                                  for scope in ("PRIMARY", "SUPPLEMENTAL", "ALL")) for row in rows),
        },
        "exact_metric_reuse_counts": {metric: 0 for metric in METRICS},
        "rows": rows,
    }
    write_json(source / "reports/original_metric_reuse_audit.json", report)
    # Prefill is evidence only; no original metric value is assigned.
    write_json(source / "reports/original_metric_review_prefill.json", {
        "protocol": "original-metric-review-prefill-v1", "metric_values": {},
        "source_audit": str(source / "reports/original_metric_reuse_audit.json"),
        "rows": [{"run_id": row["run_id"], "case_id": row["case_id"],
                  "answer_sha256": row["reusable_evidence"]["answer_sha256"],
                  "claims": row["reusable_evidence"]["claims"],
                  "branch_relations": row["reusable_evidence"]["branch_relations"],
                  "method_evidence": row["reusable_evidence"]["method_evidence"]} for row in rows],
    })
    return {key: value for key, value in report.items() if key not in {"rows"}}


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    args = parser.parse_args()
    print(json.dumps(audit(args.source.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
