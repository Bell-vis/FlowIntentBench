#!/usr/bin/env python3
"""Score the same existing answer cases for four models and summarize means.

This script only reads completed ``RunRecord`` answers.  It never invokes a
solver or creates a model answer.  Missing reviewer judgments are obtained by
the bounded core reviewer (one call per model/case); completed judgments and
results are content-addressed and reused on ``--resume``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import (  # noqa: E402
    CORE_SCHEMA_VERSION,
    SCORER_IMPLEMENTATION_VERSION,
    CoreCaseScorer,
    judgment_schema,
    score_judgment,
    scoring_prompt,
)
from flowintentbench.expansion_evaluation import load_development_case, read_json  # noqa: E402
from flowintentbench.model_runner import RunRecord  # noqa: E402
from flowintentbench.subagent_reporting import current_slot_run_paths  # noqa: E402
from flowintentbench.schema import validate_model_input  # noqa: E402
from flowintentbench.ground_truth import GroundTruth  # noqa: E402
from scripts.run_core_case_scoring import _api_completion, _load_case, _parse_json  # noqa: E402
from scripts.run_expansion_file_evaluation import file_sha256  # noqa: E402


MODEL_SOURCES = {
    "claude-fable-5-1": ROOT / "outputs/claude_resume_eval",
    "claude-sonnet-5": ROOT / "outputs/claude_resume_eval",
    "gpt-5.6-luna": ROOT / "outputs/expansion96_n3_subagents",
    "gpt-5.6-terra": ROOT / "outputs/expansion96_n3_subagents",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _answer_sha(answer: str) -> str:
    return hashlib.sha256(answer.encode()).hexdigest()


def _completed_records(model_id: str) -> dict[str, tuple[Path, RunRecord]]:
    """Return one deterministic completed answer per case, preferring trial 1."""
    root = MODEL_SOURCES[model_id]
    candidates: dict[str, list[tuple[Path, RunRecord]]] = defaultdict(list)
    for path in current_slot_run_paths(root):
        record = RunRecord.from_dict(read_json(path))
        if record.model_id != model_id or record.run_status.value != "COMPLETED" or not record.final_response:
            continue
        candidates[record.case_id].append((path, record))
    selected = {}
    for case_id, rows in candidates.items():
        rows.sort(key=lambda item: (item[1].trial_index, item[1].run_id, str(item[0])))
        selected[case_id] = rows[0]
    return selected


def _select_cases(records_by_model: Mapping[str, Mapping[str, tuple[Path, RunRecord]]], manifest: Mapping[str, Any], limit: int) -> list[str]:
    common = set.intersection(*(set(rows) for rows in records_by_model.values()))
    by_id = {row["case_id"]: row for row in manifest["cases"]}
    # Keep the comparison paired and reasonably balanced across the frozen
    # O/F conditions.  The manifest order is the only tie breaker.
    targets = {"O1-F1": 3, "O2-F1": 3, "O3-F1": 2, "O1-F2": 2}
    selected: list[str] = []
    for condition, target in targets.items():
        for case_id in (row["case_id"] for row in manifest["cases"] if row.get("condition") == condition):
            if case_id in common and case_id in by_id and case_id not in selected:
                selected.append(case_id)
                if sum(by_id[c]["condition"] == condition for c in selected) >= target:
                    break
    if len(selected) < limit:
        for row in manifest["cases"]:
            if row["case_id"] in common and row["case_id"] not in selected:
                selected.append(row["case_id"])
                if len(selected) >= limit:
                    break
    return selected[:limit]


def _judgment_has_v3_evidence(value: Mapping[str, Any]) -> bool:
    try:
        dimensions = value["operationalization"]["dimensions"]
        findings = value["findings"]
        return all("evidence_span" in item and "evidence_text" in item for item in (*dimensions, *findings))
    except (KeyError, TypeError):
        return False


def _load_reusable_judgments(cache_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read only judgments that already contain the current evidence contract."""
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    for receipt in cache_root.rglob("receipt.json"):
        if receipt.parent.parent.name != "api":
            continue
        try:
            payload = json.loads(receipt.read_text())
            if not payload.get("completed"):
                continue
            judgment = _parse_json(payload.get("final_text", ""))
            if not _judgment_has_v3_evidence(judgment):
                continue
            prompt = receipt.parent / "prompt.txt"
            # The answer hash is carried in the corresponding result when
            # available; otherwise this judgment is intentionally not reused.
            found[("", "", _sha(judgment))] = judgment
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return found


def _find_cached_for_result(output: Path, model_id: str, case_id: str, answer_sha: str) -> dict[str, Any] | None:
    judgment_path = output / "judgments" / f"{model_id}__{case_id}__{answer_sha}.json"
    if not judgment_path.exists():
        return None
    try:
        judgment = json.loads(judgment_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return judgment if _judgment_has_v3_evidence(judgment) else None


def _case_payload(row: Mapping[str, Any]):
    case_input, metadata, gt, _material = _load_case(row)
    payload = case_input.model_dump(mode="json")
    payload["case_id"] = row["case_id"]
    if metadata is not None:
        payload["principal_operationalization_dimensions"] = list(getattr(metadata, "principal_operationalization_dimensions", ()))
        payload["unresolved_operationalization_dimensions"] = [
            getattr(item, "value", item) for item in getattr(metadata, "unresolved_operationalization_dimensions", ())
        ]
        payload["finding_goal"] = getattr(metadata, "finding_goal", None)
    return payload, gt


def _efficiency(record: RunRecord) -> dict[str, Any]:
    return {key: getattr(record, key, None) for key in (
        "input_tokens", "output_tokens", "model_turn_count", "tool_call_count",
        "python_execution_count", "wall_clock_time", "provider_reported_cost",
    )}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _write_summary(output: Path, selected: list[str], results: list[dict[str, Any]], manifest: Mapping[str, Any]) -> None:
    metrics = ("o_score", "urs", "finding_precision", "core_finding_recall", "finding_requirement_recall", "numeric_accuracy", "c_score", "branch_alignment", "evidence_coverage")
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_model[result["model_id"]].append(result)
    summary: dict[str, Any] = {
        "schema_version": CORE_SCHEMA_VERSION,
        "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION,
        "selection_case_ids": selected,
        "selection_sha256": _sha(selected),
        "selection_n": len(selected),
        "models": {},
        "pairwise": {},
    }
    for model_id, rows in sorted(by_model.items()):
        block: dict[str, Any] = {
            "requested_n": len(selected),
            "result_n": len(rows),
            "core_scored_n": sum(row.get("status") == "CORE_SCORED" for row in rows),
            "review_error_n": sum(row.get("status") == "REVIEW_ERROR" for row in rows),
            "model_noncompletion_n": sum(row.get("status") == "MODEL_NONCOMPLETION" for row in rows),
            "metrics": {},
        }
        for metric in metrics:
            values = [row.get("metrics", {}).get(metric) for row in rows]
            values = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
            block["metrics"][metric] = {"mean": _mean(values), "sd": _sd(values), "n": len(values), "missing_n": len(rows) - len(values)}
        summary["models"][model_id] = block
    for left in sorted(by_model):
        for right in sorted(by_model):
            if left >= right:
                continue
            pair = {}
            left_rows = {row["case_id"]: row for row in by_model[left]}
            right_rows = {row["case_id"]: row for row in by_model[right]}
            for metric in metrics:
                pairs = []
                for case_id in selected:
                    lv = left_rows.get(case_id, {}).get("metrics", {}).get(metric)
                    rv = right_rows.get(case_id, {}).get("metrics", {}).get(metric)
                    if isinstance(lv, (int, float)) and isinstance(rv, (int, float)) and not isinstance(lv, bool) and not isinstance(rv, bool):
                        pairs.append(float(lv) - float(rv))
                pair[metric] = {"mean_difference_left_minus_right": _mean(pairs), "paired_n": len(pairs), "positive_n": sum(value > 0 for value in pairs), "negative_n": sum(value < 0 for value in pairs)}
            summary["pairwise"][f"{left}__vs__{right}"] = pair
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Four-model core scoring comparison", "", f"Scorer: `{SCORER_IMPLEMENTATION_VERSION}`; paired cases: **{len(selected)}**.", "", "| Model | Core scored | O | Finding precision | Core recall | Numeric accuracy | C | Evidence coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_id, block in sorted(summary["models"].items()):
        m = block["metrics"]
        def fmt(name: str) -> str:
            item = m[name]
            return "—" if item["mean"] is None else f"{item['mean']:.3f} (n={item['n']})"
        lines.append(f"| {model_id} | {block['core_scored_n']}/{block['requested_n']} | {fmt('o_score')} | {fmt('finding_precision')} | {fmt('core_finding_recall')} | {fmt('numeric_accuracy')} | {fmt('c_score')} | {fmt('evidence_coverage')} |")
    lines += ["", "The comparison is descriptive: it uses one already-collected answer per model and case, with reviewer coverage shown explicitly. Missing or failed reviews are not converted to zero.", ""]
    (output / "summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/core_case_comparison_20260920")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--reviewer-model", default="gpt-6-astra")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="parallel reviewer calls; bounded by the shared admission limit")
    parser.add_argument("--offline", action="store_true", help="do not submit missing judgments")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    rows = {row["case_id"]: row for row in manifest["cases"]}
    model_ids = tuple(MODEL_SOURCES)
    records_by_model = {model_id: _completed_records(model_id) for model_id in model_ids}
    selected = _select_cases(records_by_model, manifest, args.limit)
    if len(selected) < args.limit:
        raise SystemExit(f"only {len(selected)} common completed cases are available; need {args.limit}")
    (output / "selection.json").write_text(json.dumps({"models": model_ids, "case_ids": selected, "criteria": "common completed answers; condition-stratified manifest order; trial 1 preferred", "case_manifest_sha256": file_sha256(args.manifest)}, indent=2, ensure_ascii=False) + "\n")

    controller = None
    if not args.offline:
        from scripts.run_claude_benchmark import ClaudeController
        controller = ClaudeController(output, ROOT / "settings.json", None, ROOT / "config/yiapi.toml", 0, api_judge_workers=4, api_shared_concurrency=4)
        controller.agent.structured_only = True
        if not controller.judge_governor.check_health(startup=True):
            raise SystemExit("GPT-6 reviewer preflight failed; no model answers were touched")

    work_items = [(model_id, case_id) for model_id in model_ids for case_id in selected]

    def process_one(item: tuple[str, str]) -> dict[str, Any] | None:
        model_id, case_id = item
        path, record = records_by_model[model_id][case_id]
        answer_sha = _answer_sha(record.final_response or "")
        destination = output / "cases" / f"{model_id}__{case_id}.json"
        if args.resume and destination.exists():
            try:
                cached_result = json.loads(destination.read_text())
            except (OSError, json.JSONDecodeError):
                cached_result = None
            if (isinstance(cached_result, dict)
                    and cached_result.get("run_record_sha256") == file_sha256(path)
                    and cached_result.get("scorer_implementation_version") == SCORER_IMPLEMENTATION_VERSION
                    and cached_result.get("status") in {"CORE_SCORED", "MODEL_NONCOMPLETION"}):
                return cached_result
        case_input, gt = _case_payload(rows[case_id])
        cached_judgment = _find_cached_for_result(output, model_id, case_id, answer_sha)
        judgment = cached_judgment
        error = None
        result = None
        if judgment is None and not args.offline:
            api_completion = _api_completion(controller, output, args.reviewer_model, case_id)
            captured: dict[str, Any] = {}
            def completion(prompt, schema):
                value = api_completion(prompt, schema)
                captured["judgment"] = value
                return value
            try:
                # CoreCaseScorer supplies the exact dimension enum for
                # this case (some cases have an additional criterion).
                result = CoreCaseScorer(completion).score(
                    case_input, record.final_response or "", gt,
                    efficiency=_efficiency(record),
                )
                judgment = captured.get("judgment")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if judgment is not None:
            (output / "judgments").mkdir(parents=True, exist_ok=True)
            (output / "judgments" / f"{model_id}__{case_id}__{answer_sha}.json").write_text(json.dumps(judgment, indent=2, ensure_ascii=False) + "\n")
            try:
                result = score_judgment(case_input, record.final_response or "", gt, judgment, efficiency=_efficiency(record))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if error or result is None:
            result = {"schema_version": CORE_SCHEMA_VERSION, "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION, "case_id": case_id, "status": "REVIEW_ERROR", "error": error or "offline judgment unavailable", "metrics": {}}
        result.update({"model_id": model_id, "run_id": record.run_id, "run_record_sha256": file_sha256(path)})
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result

    results: list[dict[str, Any]] = []
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must be between 1 and 4")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_one, work_items):
            if result is not None:
                results.append(result)
    results.sort(key=lambda item: (item.get("model_id", ""), item.get("case_id", "")))
    index = {"schema_version": CORE_SCHEMA_VERSION, "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION, "reviewer_model": None if args.offline else args.reviewer_model, "results": results}
    (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    _write_summary(output, selected, results, manifest)
    print(json.dumps({"status": "OK", "cases": len(selected), "results": len(results), "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
