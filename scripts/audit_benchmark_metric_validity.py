#!/usr/bin/env python3
"""Report reference coverage, ceiling effects, and identification without ranking models."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/model_metric_calibration/reports"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    manifest_path = ROOT / "experiments/expansion_v1_development/case_manifest.json"
    manifest = read(manifest_path)
    branches, findings, values, spatial, conditions = (Counter() for _ in range(5))
    for row in manifest["cases"]:
        gt = read(ROOT / row["ground_truth_path"])
        material = read(row["evaluation_material_path"])
        branches[len(gt["acceptable_operationalizations"])] += 1
        conditions[row["condition"]] += 1
        for branch in gt["findings_by_operationalization"]:
            findings[len(branch["findings"])] += 1
            values.update(type(r.get("value")).__name__ for r in branch["findings"])
        for policy in material["finding_verification_policy"]["policies"]:
            if policy["verification_mode"] == "spatial_euclidean":
                spatial[str(policy["verification_parameters"].get("spatial_tolerance"))] += 1
    report_path = OUT / "experiment_report.json"
    report = read(report_path)
    per_metric = {}
    for key, d in report["metric_diagnostics"].items():
        flags = []
        if d["point_n"] and d["ceiling_point_n"] == d["point_n"]:
            flags.append("ALL_IDENTIFIED_POINTS_AT_CEILING")
        if d["interval_n"]:
            flags.append("UNRESOLVED_VALUES_PRESENT")
        if not d["identified_model_mean_pairs"]:
            flags.append("NO_IDENTIFIED_MODEL_MEAN_ORDER")
        per_metric[key] = {"flags": flags, "applicable_n": d["applicable_n"],
            "point_n": d["point_n"], "ceiling_point_n": d["ceiling_point_n"],
            "interval_n": d["interval_n"], "identified_model_mean_pairs": d["identified_model_mean_pairs"]}
    result = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "case_count": len(manifest["cases"]), "conditions": dict(conditions),
        "release_status": manifest.get("status"), "human_calibration_status": manifest.get("human_calibration_status"),
        "branch_count_histogram": dict(branches), "findings_per_branch_histogram": dict(findings),
        "reference_value_type_histogram": dict(values), "spatial_tolerance_histogram": dict(spatial),
        "branch_alignment_structurally_trivial_case_n": branches[1],
        "reference_branches_with_at_most_two_findings": sum(n for k, n in findings.items() if k <= 2),
        "reference_branch_total": sum(findings.values()),
        "raw_blind_grade_histogram": dict(Counter(r["quality"] for r in report["blind_ratings"])),
        "metric_diagnostics": per_metric,
        "performance_rank_supported": False,
        "conclusion": "This pilot supports a ceiling/coverage diagnosis, not a calibrated model ranking. Numerical agreement alone does not establish scientific quality.",
        "change_constraints": ["Never choose thresholds to obtain a desired model order.",
            "Use the same declared rules for every model and trial.",
            "Keep scientific correctness, completion, and evaluator uncertainty separate.",
            "Publish changed rubrics/references as a new version and replay all models."]}
    (OUT / "benchmark_validity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("case_count", "branch_count_histogram",
        "reference_branches_with_at_most_two_findings", "reference_branch_total", "performance_rank_supported")}))


if __name__ == "__main__":
    main()
