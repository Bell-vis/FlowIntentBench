"""Export each outcome metric separately, with explicit valid denominators."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.external_file_evaluator import write_json
from flowintentbench.outcome_scoring import POLICY
from scripts.evaluate_answered_outcomes import sha_file
from scripts.benchmark_runtime import require_benchmark_runtime


def summarize(rows):
    metrics = []

    def add(name, values, lower=None, upper=None):
        known = [value for value in values if value is not None]
        metrics.append({"metric": name, "mean": statistics.mean(known) if known else None,
                        "n_valid": len(known), "n_applicable": len(values),
                        "mean_lower": statistics.mean(lower) if lower else None,
                        "mean_upper": statistics.mean(upper) if upper else None})

    for group in POLICY["groups"]:
        values = [row["metrics"][group] for row in rows]
        add(group + ".score", [value["score"] for value in values],
            [value["lower"] for value in values], [value["upper"] for value in values])
        add(group + ".assessment_coverage", [value["assessed"] / value["total"] for value in values])
        for key in ("assessed", "total"):
            add(group + "." + key, [value[key] for value in values])
    for scope in ("PRIMARY", "SUPPLEMENTAL", "ALL"):
        for key in ("coverage", "accuracy_lower", "accuracy_upper", "VERIFIED", "REFUTED", "UNVERIFIED", "total"):
            add("verification." + scope + "." + key, [row["verification"][scope][key] for row in rows])
        selected = [row["verification"][scope] for row in rows]
        add("verification." + scope + ".verified_accuracy",
            [value["VERIFIED"] / (value["VERIFIED"] + value["REFUTED"])
             if value["VERIFIED"] + value["REFUTED"] else None for value in selected])
    add("rubric.assessment_coverage", [sum(value["assessed"] for value in row["metrics"].values()) /
                                      sum(value["total"] for value in row["metrics"].values()) for row in rows])
    add("audit.complete_rate", [int(row["audit_status"] == "COMPLETE") for row in rows])
    add("score.point_available_rate", [int(row["rubric_score"] is not None) for row in rows])
    add("review.extraction_limitations_rate", [int(bool(row["extraction_limitations"])) for row in rows])
    add("review.unbound_rubric_items", [len(row["unbound_rubric_items"]) for row in rows])
    add("normalization.unit_conversions", [len(row["normalization_ledger"]) for row in rows])

    items = defaultdict(list)
    for row in rows:
        for rating in row["ratings"]:
            item_id = rating["item_id"]
            # Ordinal requirements describe different things in different cases.
            if item_id.startswith("requirement:explicit:"):
                item_id = row["case_id"] + ":" + item_id
            items[item_id].append(POLICY["ratings"].get(rating["verdict"]))
    for item_id, values in sorted(items.items()):
        add("item." + item_id, values, [value if value is not None else 0 for value in values],
            [value if value is not None else 1 for value in values])
    return metrics


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_normalized_177")
    args = parser.parse_args()
    source = args.source.resolve()
    report_path = source / "reports/experiment_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["results"]
    if (report["status"] != "COMPLETE" or len(rows) != report["expected"]
            or len({row["run_id"] for row in rows}) != report["expected"]
            or any(row["status"] != "SCORED" for row in rows)):
        raise ValueError("a complete unique answer cohort is required")
    cohorts = {"ALL": rows}
    for key in ("model_id", "reviewer_source"):
        for value in sorted({row[key] for row in rows}):
            cohorts[key + ":" + value] = [row for row in rows if row[key] == value]
    output = {"source_report_sha256": sha_file(report_path), "expected": report["expected"],
              "aggregation": "Unweighted answer means within each cohort; repeated answers remain separate",
              "unknown": "mean excludes nulls and reports n_valid; lower/upper means include unknown bounds",
              "empty_claim_scope": "coverage follows stored protocol (0 for no claims); accuracy bounds are null",
              "verified_accuracy": "Diagnostic mean over answers with verified or refuted claims only; not full accuracy",
              "items": "Conditional on item applicability; absent items are not zero; ordinal requirements are case-specific",
              "cohorts": {name: {"answers": len(selected), "metrics": summarize(selected)}
                          for name, selected in cohorts.items()}}
    write_json(source / "reports/metric_means.json", output)
    csv_path = source / "reports/metric_means.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cohort", "metric", "mean", "n_valid", "n_applicable",
                                                   "mean_lower", "mean_upper"])
        writer.writeheader()
        for cohort, block in output["cohorts"].items():
            writer.writerows({"cohort": cohort, **metric} for metric in block["metrics"])
    for cohort, block in output["cohorts"].items():
        if cohort.startswith("reviewer_source:"):
            continue
        print(json.dumps({"cohort": cohort, "answers": block["answers"], "metrics": [metric for metric in block["metrics"]
              if not metric["metric"].startswith("item.") or metric["metric"].startswith("item.method:")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
