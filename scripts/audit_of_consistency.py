#!/usr/bin/env python3
"""Frozen-reference O/F cross-swap study, with no LLM or model ranking.

Every pair holds a valid reference O and an independently correct reference F.
Off-diagonal pairs test what is lost by joining them. The numerical host is
tested with oracle extraction; this is not end-to-end judge validation.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.evaluator import PredictedAtomicFinding, verify_finding_value, EvaluationPendingAdjudication
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.evaluation_policy import policy_index
from flowintentbench.of_consistency import bound_result_group

OUT = ROOT / "outputs/model_metric_calibration/of_consistency"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_key(reference, policy):
    identity = policy.get("policy_identity", {})
    locator = identity.get("g_of_o_locator")
    if not locator:
        raise ValueError("reference has no authored result locator")
    return reference.category.value + ":" + locator


def main():
    started = time.monotonic()
    manifest_path = ROOT / "experiments/expansion_v1_development/case_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, case_audits, skipped = [], [], []
    for row in manifest["cases"]:
        if row["condition"] not in {"O2-F1", "O3-F1"}:
            continue
        _, _, gt, material = load_development_case(ROOT, row)
        policies = policy_index(material["finding_verification_policy"])
        branches = {}
        for branch in gt.findings_by_operationalization:
            refs = {}
            for ref in branch.findings:
                if ref.importance.value != "core":
                    continue
                policy = policies[(branch.operationalization_id, ref.finding_id)]
                key = role_key(ref, policy)
                if key in refs:
                    raise ValueError("authored locator is not unique within a branch")
                refs[key] = (ref, policy)
            branches[branch.operationalization_id] = refs
        role_sets = {tuple(sorted(v)) for v in branches.values()}
        if len(role_sets) != 1:
            skipped.append({"case_id": row["case_id"], "reason": "BRANCHES_HAVE_DIFFERENT_REQUIRED_ROLES"})
            continue
        # Each F column is correct under its own authored method. Matching under
        # another method is tested with its frozen tolerance and unit policy.
        matrix = {}
        for f_id, observed in branches.items():
            support = {}
            for o_id, target in branches.items():
                support[o_id] = {}
                for role, (ref, policy) in target.items():
                    source = observed[role][0]
                    prediction = PredictedAtomicFinding(source.finding_id, source.statement, source.value, source.unit, "Authored reference control")
                    try:
                        verdict = verify_finding_value(prediction, ref, explicit_policy=policy)
                    except EvaluationPendingAdjudication:
                        verdict = None
                    support[o_id][role] = verdict
            matrix[f_id] = support
        case_rows = []
        for declared, generated in itertools.product(branches, repeat=2):
            result = bound_result_group(matrix[generated], [declared])
            entry = {"case_id": row["case_id"], "condition": row["condition"],
                "declared_o": declared, "source_f": generated,
                "control_kind": "MATCHED" if declared == generated else "CROSS_SWAPPED",
                "o_valid_by_construction": True, "f_valid_somewhere_by_construction": True,
                "result": result}
            rows.append(entry)
            case_rows.append(entry)
        swapped = [r for r in case_rows if r["control_kind"] == "CROSS_SWAPPED"]
        case_audits.append({"case_id": row["case_id"], "condition": row["condition"], "family_id": row["family_id"],
            "branch_n": len(branches), "ordered_cross_swap_n": len(swapped),
            "numerically_distinguishable_cross_swap_n": sum(r["result"]["binding_gap"]["lower"] > 0 for r in swapped),
            "unresolved_cross_swap_n": sum(r["result"]["binding_gap"]["value"] is None for r in swapped),
            "gt_sha256": row["ground_truth_sha256"], "evaluation_material_sha256": row["evaluation_material_sha256"]})
    matched = [r for r in rows if r["control_kind"] == "MATCHED"]
    swapped = [r for r in rows if r["control_kind"] == "CROSS_SWAPPED"]
    summary = {"matched_n": len(matched),
        "scientific_family_n": len({r["family_id"] for r in case_audits}),
        "matched_conditional_support_one_n": sum(r["result"]["conditional_required_support"]["value"] == 1 for r in matched),
        "cross_swapped_n": len(swapped),
        "cross_swapped_identified_gap_n": sum(r["result"]["binding_gap"]["lower"] > 0 for r in swapped),
        "cross_swapped_zero_gap_n": sum(r["result"]["binding_gap"]["value"] == 0 for r in swapped),
        "cross_swapped_unknown_gap_n": sum(r["result"]["binding_gap"]["value"] is None for r in swapped),
        "cross_swapped_conditional_support_values": dict(Counter(str(r["result"]["conditional_required_support"]["value"]) for r in swapped)),
        "cross_swapped_alignment_values": dict(Counter(str(r["result"]["informative_branch_alignment"]["value"]) for r in swapped))}
    output = {"protocol": "of-consistency-reference-controls-v1", "manifest_sha256": fingerprint(manifest_path),
        "scorer_sha256": fingerprint(ROOT / "flowintentbench/of_consistency.py"),
        "script_sha256": fingerprint(__file__), "api_calls": 0, "seconds": time.monotonic() - started,
        "scope": "All existing O2-F1/O3-F1 cases; oracle extraction; frozen GT values and tolerances; no model score changes.",
        "summary": summary, "case_audits": case_audits, "skipped": skipped, "rows": rows,
        "limits": ["Reference cross-swaps test numerical distinguishability, not natural model error frequency.",
            "No LLM extraction, result grouping or novel-method adjudication is validated by this study.",
            "A zero cross-swap gap can reflect identical or tolerance-indistinguishable outcomes; it is not successful attribution.",
            "Quality rankings and end-to-end consistency validity require independently labeled natural answers."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference_controls.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["case_id", "condition", "declared_o", "source_f", "control_kind", "conditional_support", "best_support", "binding_gap", "informative_alignment", "alignment_reason"]
    with (OUT / "reference_controls.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            d = r["result"]
            writer.writerow({**{k: r[k] for k in columns[:5]},
                "conditional_support": d["conditional_required_support"]["value"],
                "best_support": d["best_required_support"]["value"], "binding_gap": d["binding_gap"]["value"],
                "informative_alignment": d["informative_branch_alignment"]["value"],
                "alignment_reason": d["informative_branch_alignment"]["reason"]})
    print(json.dumps({"summary": summary, "cases": len(case_audits), "skipped": skipped, "seconds": output["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
