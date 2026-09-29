"""Validate separately attributed GPT-6 agent reviews against frozen API packets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.expansion_evaluation import load_development_case, read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.model_runner import RunRecord
from flowintentbench.outcome_scoring import POLICY, OutcomeJudgment, build_rubric, digest, review_prompt, score_outcome
from scripts.evaluate_answered_outcomes import sha_file, write_reports
from scripts.benchmark_runtime import require_benchmark_runtime


def merge(source, reviews, output, manifest_path):
    if source == output or reviews == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("use a separate merged output directory")
    selection = read_json(source / "selection.json")
    agent_selection = read_json(reviews / "selection.json")
    orchestration = read_json(reviews / "orchestration.json")
    if sha_file(source / "selection.json") != agent_selection["source_selection_sha256"]:
        raise ValueError("agent source selection changed")
    if sha_file(manifest_path) != selection["manifest_sha256"]:
        raise ValueError("case manifest changed")
    assignments = {}
    for group in orchestration["groups"]:
        if group["requested_model"] != "gpt-6-astra":
            raise ValueError("agent model must be explicitly selected as gpt-6-astra")
        group_path = reviews / group["assignment_file"]
        if sha_file(group_path) != group["assignment_sha256"]:
            raise ValueError("agent assignment changed")
        for item in read_json(group_path):
            if item["run_id"] in assignments:
                raise ValueError("duplicate agent assignment")
            assignments[item["run_id"]] = (group, item)
    if set(assignments) != {item["run_id"] for item in agent_selection["pending"]}:
        raise ValueError("agent assignment coverage mismatch")
    rows = selection["answers"]
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate selected answers")
    manifest = {row["case_id"]: row for row in read_json(manifest_path)["cases"]}
    scorer_hash = sha_file(ROOT / "flowintentbench/outcome_scoring.py")
    verifier_hash = sha_file(ROOT / "flowintentbench/evaluator.py")
    merged_selection = {**selection, "review_sources": ["HTTP_API", "EXPLICIT_GPT6_AGENT"],
                        "source_selection_sha256": sha_file(source / "selection.json"),
                        "orchestration_sha256": sha_file(reviews / "orchestration.json")}
    existing_selection = output / "selection.json"
    if existing_selection.exists() and read_json(existing_selection) != merged_selection:
        raise ValueError("merged selection changed; use a new output directory")
    write_json(existing_selection, merged_selection)
    write_json(output / "policy.json", POLICY)
    results = []
    for row in rows:
        original_path = source / "cases" / (row["run_id"] + ".json")
        original = read_json(original_path)
        request = read_json(source / "requests" / (original["request_id"] + ".json"))
        if sha_file(Path(row["run_path"])) != row["run_sha256"]:
            raise ValueError("source answer changed: " + row["run_id"])
        answer = RunRecord.from_dict(read_json(Path(row["run_path"]))).final_response
        if hashlib.sha256(answer.encode()).hexdigest() != row["answer_sha256"]:
            raise ValueError("answer identity mismatch")
        case_input, metadata, gt, material = load_development_case(ROOT, manifest[row["case_id"]])
        rubric = build_rubric(metadata, material)
        prompt = review_prompt(case_input, metadata, material, gt, answer, rubric)
        if (request["prompt"] != prompt or request["rubric"] != rubric
                or request["schema"] != OutcomeJudgment.model_json_schema()
                or request["scorer_sha256"] != scorer_hash or request["run_id"] != row["run_id"]):
            raise ValueError("frozen review input changed: " + row["run_id"])
        if original["status"] == "SCORED":
            cached = read_json(source / "judgments" / (original["request_id"] + ".json"))
            if (cached["request_id"] != original["request_id"]
                    or digest(cached["judgment"]) != original["judgment_sha256"]
                    or original["reviewer_model"] != "gpt-6-astra"
                    or original["scorer_sha256"] != scorer_hash or original["verifier_sha256"] != verifier_hash):
                raise ValueError("API review provenance mismatch")
            replayed = score_outcome(answer, rubric, gt, material["finding_verification_policy"], cached["judgment"])
            if any(original[key] != value for key, value in replayed.items()):
                raise ValueError("API score could not be reproduced")
            result = {**original, "reviewer_source": "HTTP_API",
                      "imported_from": {"result_path": str(original_path), "sha256": sha_file(original_path)}}
        else:
            group, item = assignments[row["run_id"]]
            packet_path = Path(item["packet_path"])
            packet = read_json(packet_path)
            if (packet["source_request_sha256"] != sha_file(Path(packet["source_request_path"]))
                    or packet["prompt"] != prompt or packet["schema"] != request["schema"]
                    or packet["rubric"] != rubric or packet["run_id"] != row["run_id"]
                    or packet["run_sha256"] != row["run_sha256"]):
                raise ValueError("agent packet changed")
            raw_path = Path(item["judgment_path"])
            if not raw_path.exists():
                result = {**row, "status": "NOT_REVIEWED", "reviewer_source": "EXPLICIT_GPT6_AGENT"}
                write_json(output / "cases" / (row["run_id"] + ".json"), result)
                results.append(result)
                continue
            try:
                judgment = read_json(raw_path)
                result = score_outcome(answer, rubric, gt, material["finding_verification_policy"], judgment)
            except (ValueError, TypeError, KeyError) as exc:
                result = {**row, "status": "REVIEW_ERROR", "reviewer_source": "EXPLICIT_GPT6_AGENT",
                          "error": f"{type(exc).__name__}: {exc}"}
                write_json(output / "cases" / (row["run_id"] + ".json"), result)
                results.append(result)
                continue
            provenance = {"source": "EXPLICIT_GPT6_AGENT", "agent_task": group["agent_task"],
                          "requested_model": group["requested_model"], "reasoning_effort": group["reasoning_effort"],
                          "model_evidence": "Codex spawn_agent explicit model selection; no HTTP response model receipt",
                          "review_profile": "frozen-packet-agent-v1", "packet_path": str(packet_path),
                          "packet_sha256": sha_file(packet_path), "raw_judgment_path": str(raw_path),
                          "raw_judgment_sha256": sha_file(raw_path),
                          "orchestration_sha256": sha_file(reviews / "orchestration.json")}
            identity = digest({"source_request": request["request_id"], "policy": POLICY,
                               "verification_policy": material["finding_verification_policy"],
                               "provenance": provenance, "scorer_sha256": scorer_hash})
            request = {**request, "request_id": identity, "agent_review_provenance": provenance}
            cached = {"request_id": identity, "judgment": judgment, "reviewer_receipts": [],
                      "agent_review_provenance": provenance}
            result.update(row, request_id=identity, scorer_sha256=scorer_hash, verifier_sha256=verifier_hash,
                          reviewer_model="gpt-6-astra", reasoning_effort=group["reasoning_effort"],
                          reviewer_source="EXPLICIT_GPT6_AGENT", reviewer_receipts=[],
                          agent_review_provenance=provenance)
        write_json(output / "requests" / (result["request_id"] + ".json"), request)
        write_json(output / "judgments" / (result["request_id"] + ".json"), cached)
        write_json(output / "cases" / (row["run_id"] + ".json"), result)
        results.append(result)
    return write_reports(output, merged_selection, results)


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_v2_responses_177")
    parser.add_argument("--reviews", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_agent_reviews")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--monitor-seconds", type=float, default=60)
    args = parser.parse_args()
    while True:
        cycle_started = time.monotonic()
        report = merge(args.source.resolve(), args.reviews.resolve(), args.output.resolve(), args.manifest.resolve())
        snapshot = {"utc": datetime.now(timezone.utc).isoformat(), "status": report["status"],
                    "expected": report["expected"], "status_counts": report["status_counts"],
                    "source_counts": report["reviewer"]["source_counts"]}
        with (args.output / "monitor.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        print(json.dumps(snapshot), flush=True)
        if not args.watch or report["status"] == "COMPLETE":
            return 0 if report["status"] == "COMPLETE" else 2
        time.sleep(max(1, args.monitor_seconds - (time.monotonic() - cycle_started)))


if __name__ == "__main__":
    raise SystemExit(main())
