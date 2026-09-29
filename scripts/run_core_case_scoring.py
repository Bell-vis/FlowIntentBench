#!/usr/bin/env python3
"""Run the fast one-call core scorer for existing model answers.

This entrypoint never runs a solver.  It accepts current ledger-bound
RunRecords, sends at most one structured review call per selected case, and
writes a self-contained metric record.  ``--judgment-json`` is provided for
offline replay and tests; normal runs use GPT-6 Astra through the project's
explicit proxy transport.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import (
    CORE_SCHEMA_VERSION,
    SCORER_IMPLEMENTATION_VERSION,
    CoreCaseScorer,
    judgment_schema,
)
from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.schema import validate_model_input
from flowintentbench.ground_truth import GroundTruth
from flowintentbench.model_runner import RunRecord
from flowintentbench.subagent_reporting import current_slot_run_paths
from scripts.run_expansion_file_evaluation import file_sha256


def require_benchmark() -> None:
    from scripts.benchmark_runtime import require_benchmark_runtime
    require_benchmark_runtime()

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3].rstrip()
    # A complete decimal token such as 1. has the exact numeric meaning 1.0.
    # Only normalize tokens at JSON value positions, outside quoted strings.
    # Do not complete containers, quotes, exponents, or missing judgments.
    decimal_tokens = re.compile(r'"(?:\\.|[^"\\])*"|(?<=[\[:,])(?P<number>\s*-?(?:0|[1-9]\d*)\.)(?=\s*[,}\]])', re.S)
    text = decimal_tokens.sub(lambda m: m['number']+'0' if m['number'] is not None else m[0], text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        # Gateways may concatenate commentary and final JSON. Recover one
        # complete object only; multiple judgments are ambiguous. Raw receipts
        # remain unchanged and typed/evidence validation is still required.
        decoder, candidates, cursor = json.JSONDecoder(), [], 0
        while cursor < len(text):
            positions = [p for p in (text.find('{',cursor),text.find('[',cursor)) if p>=0]
            if not positions:
                break
            start = min(positions)
            try:
                candidate, end = decoder.raw_decode(text,start)
            except json.JSONDecodeError:
                raise ValueError(f"reviewer did not return unambiguous JSON: {exc}") from exc
            candidates.append(candidate)
            cursor = end
        if len(candidates)!=1:
            raise ValueError('reviewer must return exactly one complete JSON object') from exc
        value = candidates[0]
    if not isinstance(value, dict):
        raise ValueError("reviewer JSON must be an object")
    return value


def _load_case(row):
    """Load both expansion-development and full-dataset manifest rows."""
    if "evaluation_material_path" in row:
        return load_development_case(ROOT, row)
    input_path = ROOT / row["case_input_path"]
    gt_path = ROOT / row["ground_truth_path"]
    if file_sha256(input_path) != row["case_input_sha256"] or file_sha256(gt_path) != row["ground_truth_sha256"]:
        raise ValueError(f"frozen case/GT digest mismatch for {row['case_id']}")
    case_input = validate_model_input(read_json(input_path))
    gt = GroundTruth.model_validate(read_json(gt_path))
    return case_input, None, gt, {}


def _api_completion(controller, output: Path, model: str, case_id: str):
    def completion(prompt, schema):
        last_error = None
        # A malformed SSE body is a transport observation, not a scientific
        # judgment. Allow one bounded retry after a fresh health probe; never
        # loop on a provider or schema failure.
        for attempt in range(2):
            if attempt:
                controller.judge_governor.check_health(startup=True)
            token = f"{time.time_ns()}-{case_id}-try{attempt + 1}"
            work = output / "work" / token
            audit = output / "api" / token
            receipt = controller.agent(prompt, model, work, audit, timeout_seconds=240,
                                       output_schema=schema, reasoning_effort="medium")
            if receipt.get("completed"):
                return _parse_json(receipt.get("final_text", ""))
            last_error = receipt.get("error") or "core reviewer did not complete"
            if last_error not in {"invalid_stream_response", "api_preflight_required",
                                  "gpt_api_preflight_unavailable", "provider_unavailable"}:
                break
        raise RuntimeError(last_error)
    return completion


def main() -> int:
    require_benchmark()
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, default=ROOT / "outputs/claude_resume_eval")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_resume_eval/core_scoring")
    parser.add_argument("--model-id", action="append", help="only this answer model; repeatable")
    parser.add_argument("--case-id", action="append", help="only this case; repeatable")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse an unchanged core result without another API call")
    parser.add_argument("--judgment-json", type=Path, help="offline judgment object")
    parser.add_argument("--reviewer-model", default="gpt-6-astra")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.offline and args.judgment_json:
        parser.error("--judgment-json requires --offline")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest)
    rows = {row["case_id"]: row for row in manifest["cases"]}
    records = []
    for path in current_slot_run_paths(args.collection_root):
        record = RunRecord.from_dict(read_json(path))
        if args.model_id and record.model_id not in set(args.model_id):
            continue
        if args.case_id and record.case_id not in set(args.case_id):
            continue
        if record.case_id in rows and record.final_response:
            records.append((path, record))
    records.sort(key=lambda item: (item[1].model_id, item[1].case_id, str(item[0])))
    records = records[:args.limit]
    if not records:
        raise SystemExit("没有符合筛选条件的已完成答案")

    controller = None
    if not args.offline:
        from scripts.run_claude_benchmark import ClaudeController
        controller = ClaudeController(output, ROOT / "settings.json", None, ROOT / "config/yiapi.toml", 0,
                                      api_judge_workers=1, api_shared_concurrency=4)
        controller.agent.structured_only = True
        if not controller.judge_governor.check_health(startup=True):
            raise SystemExit("GPT-6 reviewer preflight failed; no case was submitted")
        completion = _api_completion(controller, output, args.reviewer_model, records[0][1].case_id)
        scorer = CoreCaseScorer(completion)
    else:
        if not args.judgment_json:
            raise SystemExit("offline mode requires --judgment-json")
        judgment_payload = _parse_json(args.judgment_json.read_text())
        # A single judgment is safe only for one selected answer.  For a
        # batch, require an explicit case-id keyed mapping so one review can
        # never be silently applied to another model/case.
        if len(records) == 1 and "answer_completed" in judgment_payload:
            offline_judgments = {records[0][1].case_id: judgment_payload}
        else:
            offline_judgments = judgment_payload.get("by_case")
            if not isinstance(offline_judgments, dict) or any(
                not isinstance(value, dict) for value in offline_judgments.values()
            ):
                raise SystemExit(
                    "offline batch judgment JSON must be {'by_case': {case_id: judgment}}"
                )
        scorer = CoreCaseScorer()

    results = []
    for path, record in records:
        row = rows[record.case_id]
        source_digest = file_sha256(path)
        destination = output / "cases" / f"{record.model_id}__{record.case_id}.json"
        if args.resume and destination.is_file():
            try:
                cached = json.loads(destination.read_text())
            except (OSError, json.JSONDecodeError):
                cached = None
            if (isinstance(cached, dict) and cached.get("run_record_sha256") == source_digest
                    and cached.get("schema_version") == CORE_SCHEMA_VERSION
                    and cached.get("scorer_implementation_version") == SCORER_IMPLEMENTATION_VERSION
                    and cached.get("status") in {"CORE_SCORED", "MODEL_NONCOMPLETION"}):
                results.append(cached)
                continue
        case_input, metadata, gt, _material = _load_case(row)
        case_payload = case_input.model_dump(mode="json")
        case_payload["case_id"] = record.case_id
        if metadata is not None:
            case_payload["principal_operationalization_dimensions"] = list(
                getattr(metadata, "principal_operationalization_dimensions", ())
            )
            case_payload["unresolved_operationalization_dimensions"] = list(
                getattr(metadata, "unresolved_operationalization_dimensions", ())
            )
            case_payload["finding_goal"] = getattr(metadata, "finding_goal", None)
        efficiency = {key: getattr(record, key, None) for key in (
            "input_tokens", "output_tokens", "model_turn_count", "tool_call_count",
            "python_execution_count", "wall_clock_time", "provider_reported_cost")}
        try:
            result = scorer.score(case_payload, record.final_response, gt,
                                  efficiency=efficiency,
                                  judgment=(offline_judgments.get(record.case_id)
                                            if args.offline else None))
        except Exception as exc:
            result = {"schema_version": CORE_SCHEMA_VERSION, "case_id": record.case_id,
                      "status": "REVIEW_ERROR", "error": f"{type(exc).__name__}: {exc}",
                      "answer_sha256": __import__("hashlib").sha256(record.final_response.encode()).hexdigest()}
        result.update({"model_id": record.model_id, "run_id": record.run_id,
                       "run_record_sha256": source_digest})
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        results.append(result)
    index = {"schema_version": CORE_SCHEMA_VERSION,
             "scorer_implementation_version": SCORER_IMPLEMENTATION_VERSION,
             "reviewer_model": None if args.offline else args.reviewer_model,
             "one_call_per_case": not args.offline, "results": results}
    (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "OK", "cases": len(results),
                      "core_scored": sum(item.get("status") == "CORE_SCORED" for item in results),
                      "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
