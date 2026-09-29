#!/usr/bin/env python3
"""Report all 576 requested collaboration slots without invoking evaluators."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowintentbench.quality_efficiency import quality_status
from flowintentbench.subagent_reporting import report_subagent_experiment


def render_markdown(result):
    """Render the validated report without filling unknown scores or telemetry."""

    def number(value):
        return "-" if value is None else f"{value:.3f}"

    lines = [
        "# 96-case x 2-model x N=3 experiment report",
        "",
        f"Status: {result['status']}. Scientifically terminal trials: "
        f"{result['scientifically_terminal_slot_count']}/{result['requested_slot_count']}.",
        "",
        "Incomplete collection or evaluation is not promoted to a full result. "
        "A dash means unknown or not applicable; exact denominators are in the JSON report.",
        "",
        "Run status: "
        + ", ".join(f"{key}={value}" for key, value in result["status_counts"].items())
        + ".",
        "",
        "## Scientific metrics by O/F condition",
        "",
        "Each case is aggregated across three trials before case-level macro means are computed. "
        "Only scientifically finalized N=3 cases contribute values.",
        "",
        "| Model | Condition | Finalized N=3 cases | O | URS | Resolved O compliance | "
        "F precision | F requirement recall | F2 adequate core rate | C |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metric_keys = (
        "mean_o_score",
        "mean_urs",
        "mean_resolved_o_compliance",
        "mean_finding_precision",
        "mean_finding_requirement_recall",
        "adequate_core_complete_rate",
        "mean_c_score",
    )
    for model, groups in result["scientific_by_model"].items():
        for condition, group in groups["by_condition"].items():
            cells = [
                model,
                condition,
                f"{group['finalized_n3_case_count']}/{group['requested_case_count']}",
            ]
            cells.extend(number(group[key]) for key in metric_keys)
            lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Quality indicators and corresponding efficiency",
            "",
            "The quality system reports the original indicators (O-score, URS, resolved-O compliance, "
            "finding precision, finding requirement recall, adequate-core completeness, and C-score) "
            "with condition-specific applicability and denominators. PASS/FAIL/PENDING describes "
            "whether the required indicators for that case are complete; it is not an additional "
            "quality metric. Efficiency is reported separately and includes only passing trials in "
            "the qualified view. Wall time includes scheduling; Python counts cover only executions "
            "recorded by the journal helper.",
            "",
            "| Model | Condition | Resolved/eligible | Passing trials | Pass rate | "
            "Passing wall seconds | Passing Python calls |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, groups in result["quality_efficiency"]["by_model"].items():
        for condition, group in groups["by_condition"].items():
            efficiency = group["quality_qualified_efficiency"]
            cells = [
                model,
                condition,
                f"{group['quality_resolved_n']}/{group['quality_eligible_n']}",
                str(group["quality_status_counts"]["PASS"]),
                number(group["quality_pass_rate"]),
                number(efficiency["wall_clock_time"]["case_macro_mean"]),
                number(efficiency["python_execution_count"]["case_macro_mean"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")

    answered = result.get("answered_evaluation")
    solver_efficiency = result.get("solver_efficiency")
    if answered and solver_efficiency:
        lines.extend(
            [
                "",
                "## Existing answered-trial coverage and solver-only efficiency",
                "",
                f"Existing-answer status: {answered['status']}. Quality complete: "
                f"{answered['quality_complete_count']}/{answered['answered_slot_count']}; "
                f"efficiency audited: {answered['efficiency_audited_count']}/"
                f"{answered['answered_slot_count']}; primary efficiency available: "
                f"{answered['primary_efficiency_available_count']}/"
                f"{answered['answered_slot_count']}.",
                "",
                "The primary view uses successful target Claude sessions only. GPT-6 review, "
                "host scoring, local code/tool time, failed API attempts, and known "
                "transport-affected runs are excluded.",
                "",
                "| Model | Answered | Quality complete | Efficiency audited | Primary available | "
                "Mean target tokens | Mean API seconds | Quality-pass sessions |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model, coverage in answered.get("by_model", {}).items():
            efficiency = solver_efficiency.get("by_model", {}).get(model, {})
            metrics = efficiency.get("all_eligible_sessions", {})
            tokens = metrics.get("total_tokens", {}).get("observation_mean")
            elapsed = metrics.get("model_api_time_seconds", {}).get("observation_mean")
            lines.append(
                f"| {model} | {coverage['answered_slot_count']} | "
                f"{coverage['quality_complete_count']} | {coverage['efficiency_audited_count']} | "
                f"{coverage['primary_efficiency_available_count']} | {number(tokens)} | "
                f"{number(elapsed)} | {efficiency.get('quality_qualified_session_count', 0)} |"
            )
        lines.append(
            "API seconds come from Claude Code `duration_api_ms`. Provider-internal queueing is "
            "not separately observable, so this is not pure inference time."
        )

    if result.get("primary_efficiency_policy"):
        lines.extend(
            [
                "",
                "For mixed transports, raw end-to-end wall time is retained but must not be used for "
                "cross-transport ranking. Transport-affected observations remain separate, and successful "
                "request queue time is unknown.",
                "",
                "| Transport | Model | Observed | Transport affected | Passing in primary view | "
                "Passing wall seconds by case |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for transport, block in result.get("transport_efficiency", {}).items():
            primary = block.get("primary")
            if primary is None:
                lines.append(
                    f"| {transport} | - | {block['observed_count']} | "
                    f"{block['affected_count']} | 0 | - |"
                )
                continue
            for model, groups in primary["by_model"].items():
                overall = groups["overall"]
                elapsed = overall["quality_qualified_efficiency"]["wall_clock_time"][
                    "case_macro_mean"
                ]
                lines.append(
                    f"| {transport} | {model} | {block['observed_count']} | "
                    f"{block['affected_count']} | {overall['quality_status_counts']['PASS']} | "
                    f"{number(elapsed)} |"
                )

    diagnostics = result.get("metric_validity_diagnostics")
    if diagnostics:
        lines.extend(
            [
                "",
                "## Metric validity diagnostics (no new model calls)",
                "",
                "URS measures valid completion of unspecified operationalization dimensions. C measures "
                "scientific support for the selected O. Diagnostics cover finalized answers only; "
                "all-three-pass rates are publishable only after every eligible case is complete.",
                "",
                "| Model | Condition | URS full/applicable | C equals precision/applicable | "
                "Resolved N3 cases | All-three-pass cases |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model, conditions in diagnostics["by_model"].items():
            for condition, group in conditions.items():
                urs = group["urs"]
                consistency = group["c"]
                reliability = group["reliability_n3"]
                lines.append(
                    f"| {model} | {condition} | {urs['full_score_count']}/{urs['n']} | "
                    f"{consistency['equals_precision_n']}/{consistency['applicable_n']} | "
                    f"{reliability['resolved_cases']} | {reliability['all_three_pass_cases']} |"
                )
        lines.append(
            "The JSON report retains dimension-level URS completion, C applicability, out-of-GT "
            "counts, and validation diagnostics; missing results are not filled with zero."
        )

    completed_trials = [
        row
        for row in result["trial_rows"]
        if row.get("evaluation_status") == "SCORED"
        and row["status"] in {"COMPLETED", "MODEL_NONCOMPLETION"}
    ]
    lines.extend(
        [
            "",
            "## Finalized individual trials",
            "",
            "This table is a progress diagnostic, not a complete or N=3 aggregate and not a model "
            "ranking. Unresolved values are not filled with zero; formal summaries still require "
            "three trials per case.",
            "",
        ]
    )
    if completed_trials:
        lines.extend(
            [
                "| Model | Case | Trial | Condition | O | URS | Resolved O | F precision | "
                "F recall | F2 adequate | C | Quality | Wall seconds | Python calls |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|",
            ]
        )
        for row in completed_trials:
            cells = [row["model"], row["case_id"], str(row["trial"]), row["condition"]]
            cells.extend(
                number(row.get(key))
                for key in (
                    "o_score",
                    "urs",
                    "resolved_o_compliance",
                    "finding_precision",
                    "finding_requirement_recall",
                )
            )
            adequate = row.get("adequate_core_complete")
            cells.extend(
                [
                    "-" if adequate is None else ("yes" if adequate else "no"),
                    number(row.get("c_score")),
                    quality_status(row),
                    number(row.get("wall_clock_time")),
                    "-"
                    if row.get("python_execution_count") is None
                    else str(row["python_execution_count"]),
                ]
            )
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("No successful trial has complete scientific adjudication yet.")

    lines.extend(
        [
            "",
            "This is a development evaluation. Model and scoring-script API calls recorded by the "
            f"report: {result['api_calls']}. Missing token, cost, and turn telemetry remains unknown.",
            "",
        ]
    )
    if result.get("mixed_evaluator_snapshots"):
        lines.extend(
            [
                "",
                "## Evaluator versions and continuation",
                "",
                "Historical scores retain their original version. Infrastructure failures do not become "
                "scientific scores; different scientific evaluator versions are summarized separately "
                "and never form a mixed-version N=3. See `by_evaluator_snapshot` in the JSON report.",
                "",
                "| Evaluator snapshot | Stored evaluations | Scientifically terminal trials |",
                "|---|---:|---:|",
            ]
        )
        for snapshot, counts in result["evaluation_snapshots"].items():
            lines.append(
                f"| {snapshot} | {counts['scored_record_count']} | "
                f"{counts['scientifically_terminal_slot_count']} |"
            )
        if result.get("mixed_scientific_evaluator_snapshots"):
            lines.append(
                "Cross-version scientific means and quality-qualified efficiency are withheld; "
                "use `by_evaluator_snapshot` for version-specific results."
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report_subagent_experiment(args.collection_root, args.evaluation_root)
    if (args.collection_root / "hybrid_protocol.json").exists():
        from scripts.benchmark_hybrid import api_accounting, annotate_transport_efficiency

        result["api_transport_usage"] = api_accounting(args.collection_root)
        result["api_calls"] = result["api_transport_usage"]["http_attempts"]
        state = json.loads((args.collection_root / "collection_state.json").read_text())
        annotate_transport_efficiency(result, state, args.collection_root)
    output = args.output or args.collection_root / "reports" / "experiment_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output.with_suffix(".md").write_text(render_markdown(result))
    print(
        json.dumps(
            {
                "report": str(output),
                "status": result["status"],
                "requested_slot_count": result["requested_slot_count"],
                "scientifically_terminal_slot_count": result[
                    "scientifically_terminal_slot_count"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
