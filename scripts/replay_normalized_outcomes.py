#!/usr/bin/env python3
"""Replay frozen outcome reviews with explicit-unit normalization, without API calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.answer_normalization import POLICY, PROTOCOL, score_normalized_outcome
from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import RunRecord
from flowintentbench.outcome_scoring import OutcomeJudgment, build_rubric, digest, review_prompt, score_outcome
from scripts.evaluate_answered_outcomes import sha_file, write_reports
from scripts.benchmark_runtime import require_benchmark_runtime


def replay(source, output, manifest_path):
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("use a separate normalization output directory")
    selection = read_json(source / "selection.json")
    if sha_file(manifest_path) != selection["manifest_sha256"]:
        raise ValueError("source case manifest changed")
    manifest = read_json(manifest_path)
    cases = {row["case_id"]: row for row in manifest["cases"]}
    scorer_hash = sha_file(ROOT / "flowintentbench/outcome_scoring.py")
    normalization_hash = sha_file(ROOT / "flowintentbench/answer_normalization.py")
    verifier_hash = sha_file(ROOT / "flowintentbench/evaluator.py")
    normalized_selection = {**selection, "protocol": PROTOCOL, "policy_sha256": digest(POLICY),
                            "baseline_selection_sha256": digest(selection)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists() and read_json(output / "selection.json") != normalized_selection:
        raise ValueError("normalization selection changed; use a new directory")
    write_json(output / "selection.json", normalized_selection)
    write_json(output / "policy.json", POLICY)
    results = []
    transitions = Counter()
    conversions = 0
    for row in selection["answers"]:
        result_path = source / "cases" / (row["run_id"] + ".json")
        if not result_path.exists():
            continue
        original = read_json(result_path)
        if original["status"] != "SCORED":
            results.append({**row, "status": original["status"], "error": original.get("error")})
            continue
        if sha_file(Path(row["run_path"])) != row["run_sha256"]:
            raise ValueError("source answer changed: " + row["run_id"])
        if original["scorer_sha256"] != scorer_hash or original["verifier_sha256"] != verifier_hash:
            raise ValueError("baseline scoring implementation changed")
        request_id = original["request_id"]
        request = read_json(source / "requests" / (request_id + ".json"))
        cached = read_json(source / "judgments" / (request_id + ".json"))
        if request["run_id"] != row["run_id"] or cached["request_id"] != request_id:
            raise ValueError("review provenance mismatch")
        judgment = cached["judgment"]
        if digest(judgment) != original["judgment_sha256"]:
            raise ValueError("review content changed")
        answer = RunRecord.from_dict(read_json(Path(row["run_path"]))).final_response
        if hashlib.sha256(answer.encode()).hexdigest() != row["answer_sha256"]:
            raise ValueError("answer identity mismatch")
        case_input, metadata, gt, material = load_development_case(ROOT, cases[row["case_id"]])
        rubric = build_rubric(metadata, material)
        if (request["rubric"] != rubric
                or request["prompt"] != review_prompt(case_input, metadata, material, gt, answer, rubric)
                or request["schema"] != OutcomeJudgment.model_json_schema()
                or request["scorer_sha256"] != scorer_hash or request["request_id"] != request_id):
            raise ValueError("frozen review input changed")
        baseline = score_outcome(answer, rubric, gt, material["finding_verification_policy"], judgment)
        if any(original[key] != value for key, value in baseline.items()):
            raise ValueError("complete baseline score could not be reproduced")
        result = score_normalized_outcome(answer, request["rubric"], gt,
                                          material["finding_verification_policy"], judgment)
        if result["baseline_verification"] != original["verification"]:
            raise ValueError("baseline verification could not be reproduced")
        result.update(row, baseline_request_id=request_id, baseline_result_sha256=sha_file(result_path),
                      scorer_sha256=scorer_hash, normalization_sha256=normalization_hash,
                      verifier_sha256=verifier_hash, reviewer_model=original["reviewer_model"],
                      reasoning_effort=original["reasoning_effort"],
                      reviewer_receipts=original["reviewer_receipts"], new_api_calls=0)
        for key in ("reviewer_source", "agent_review_provenance", "imported_from", "reused_from"):
            if key in original:
                result[key] = original[key]
        originals = {claim["claim_id"]: claim["status"] for claim in original["claim_checks"]}
        for claim in result["claim_checks"]:
            transitions[originals[claim["claim_id"]] + " -> " + claim["status"]] += 1
        conversions += len(result["normalization_ledger"])
        write_json(output / "cases" / (row["run_id"] + ".json"), result)
        results.append(result)
    report = write_reports(output, normalized_selection, results, protocol=PROTOCOL, policy=POLICY)
    comparison = {"protocol": PROTOCOL, "status": report["status"],
                  "expected": report["expected"], "status_counts": report["status_counts"],
                  "unit_conversions": conversions, "claim_transitions": dict(transitions),
                  "new_api_calls": 0, "rules": POLICY["normalization_rules"]}
    write_json(output / "reports/normalization_comparison.json", comparison)
    return comparison


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_v2_177")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_outcomes_normalized_177")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    args = parser.parse_args()
    import json
    result = replay(args.source.resolve(), args.output.resolve(), args.manifest.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
