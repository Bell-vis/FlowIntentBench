#!/usr/bin/env python3
"""Synthetic GT-derived wiring self-check. Never a model observation or judge calibration."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.evaluator import (build_case_evaluator, ExtractedPrediction,
    ExtractedOperationalization, ExtractedOperationalizationDecision, PredictedAtomicFinding,
    FindingEligibility, SemanticMatchResult, PendingCaseEvaluationRecord,
    EfficiencyObservation, EvaluatorComponentIdentity)
from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import FileJudgmentTransport, build_file_backend, write_json


class ReferenceFixture:
    """Declared exact reference-positive controls; not a natural-language judge."""
    def __init__(self, gt, manifest):
        self.gt = gt
        self.fixture_manifest = replace(manifest, evaluator_backend=EvaluatorComponentIdentity(
            implementation_id="SYNTHETIC_GT_DERIVED_SELF_CHECK", version="1"))
        self.known_statements = {d.statement for o in gt.acceptable_operationalizations for d in o.decisions}
        self.known_statements.update(f.statement for b in gt.findings_by_operationalization for f in b.findings)

    def manifest(self):
        return self.fixture_manifest

    def extract(self, request):
        raise AssertionError("Reference fixture must not extract any model response")

    def judge(self, request):
        assert request.finding.statement in self.known_statements
        return FindingEligibility(True, True, True)

    def match(self, request):
        assert request.predicted_statement in self.known_statements
        assert request.reference_statement in self.known_statements
        return (SemanticMatchResult.MATCH if request.predicted_statement == request.reference_statement
                else SemanticMatchResult.NO_MATCH)


def reference_self_check(manifest, output, *, isolate_primary_branch=False):
    base_manifest = build_file_backend(FileJudgmentTransport(Path(output) / "unused", context={})).manifest()
    rows = []
    for row in manifest["cases"]:
        case_input, metadata, gt, material = load_development_case(ROOT, row)
        operation = gt.acceptable_operationalizations[0]
        branch = next(b for b in gt.findings_by_operationalization if b.operationalization_id == operation.operationalization_id)
        if isolate_primary_branch:
            # A separate, explicitly scoped self-identity control. Source GT
            # and policies stay untouched; no off-branch scientific verdicts
            # are fabricated to make the full-GT fixture look resolved.
            gt = gt.model_copy(update={"acceptable_operationalizations": [operation],
                                       "findings_by_operationalization": [branch]})
        prediction = ExtractedPrediction(
            ExtractedOperationalization(tuple(ExtractedOperationalizationDecision(
                d.dimension, "EXTRACTED", d.statement, d.statement) for d in operation.decisions)),
            tuple(PredictedAtomicFinding(f.finding_id, f.statement, f.value, f.unit) for f in branch.findings))
        evaluator = build_case_evaluator(row["case_id"], manifest, ReferenceFixture(gt, base_manifest),
                                         finding_verification_policy=material["finding_verification_policy"],
                                         branch_execution_evidence=material["branch_execution_evidence"])
        try:
            record = evaluator.evaluate_prediction_record(metadata, gt, prediction,
                scientific_question=case_input.scientific_question,
                case_context=case_input.case_context.model_dump(mode="json"),
                efficiency=EfficiencyObservation(None, None, None, None, None))
            if isinstance(record, PendingCaseEvaluationRecord):
                result = {"status": "PENDING", "pending_type": record.pending_type,
                          "pending_reason": record.pending_reason}
            else:
                metrics = record.result.metrics
                values = metrics.to_dict()
                finding = values["scientific_findings"]
                ok = (values["scientific_operationalization"]["o_score"] == 1.0
                      and finding["finding_precision"] == 1.0
                      and finding["finding_requirement_recall"] == 1.0
                      and (not row["condition"].endswith("F2") or finding["adequate_core_complete"] is True))
                result = {"status": "PASS" if ok else "FAIL", "metrics": values}
        except Exception as exc:
            result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        rows.append({"case_id": row["case_id"], "condition": row["condition"],
                     "evaluation_material_sha256": row["evaluation_material_sha256"], **result})
    report = {"artifact_type": "SYNTHETIC_GT_DERIVED_SELF_CHECK", "model_observation": False,
              "included_in_576_model_trials": False, "judge_calibration": False,
              "api_calls": 0, "data_recomputed": False,
              "reference_scope": "PRIMARY_BRANCH_IN_MEMORY_PROJECTION" if isolate_primary_branch else "FULL_UNMODIFIED_GT",
              "fixture_rule": "Use the first frozen reference O and its GT findings directly as a controlled prediction. Exact original statements define the reference-positive fixture; no general-purpose string matcher is installed.",
              "case_count": len(rows), "pass_count": sum(r["status"] == "PASS" for r in rows),
              "pending_count": sum(r["status"] == "PENDING" for r in rows),
              "fail_count": sum(r["status"] == "FAIL" for r in rows),
              "status": "FAIL" if any(r["status"] == "FAIL" for r in rows) else "PENDING" if any(r["status"] == "PENDING" for r in rows) else "PASS",
              "cases": rows}
    name = "reference_primary_branch_self_check.json" if isolate_primary_branch else "reference_self_check.json"
    write_json(Path(output) / name, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation/expansion_file_evaluation")
    parser.add_argument("--isolate-primary-branch", action="store_true")
    args = parser.parse_args()
    report = reference_self_check(read_json(args.manifest), args.output, isolate_primary_branch=args.isolate_primary_branch)
    print(f"SYNTHETIC_GT_DERIVED_SELF_CHECK {report['status']}: {report['pass_count']}/{report['case_count']}; API_CALLS=0; MODEL_OBSERVATIONS=0")
    if report["status"] != "PASS":
        for row in report["cases"]:
            if row["status"] != "PASS":
                print(row)
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
