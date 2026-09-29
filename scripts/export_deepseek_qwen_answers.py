#!/usr/bin/env python3
"""Validate and export the 192 collected answers without evaluating them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_deepseek_qwen_sequence as sequence


def export(output, *, allow_partial=False):
    output = output.resolve()
    collector = sequence.collector
    state = collector.original._read(output / "collection_state.json")
    complete = {model: sequence.model_complete(output, model) for model in collector.MODELS}
    for model in collector.MODELS:
        if not complete[model] and not allow_partial:
            raise ValueError(f"{model}: 96 complete, hash-verified answers are required")
    summary = {"status": "COMPLETE" if all(complete.values()) else "PARTIAL", "models": {}, "repetitions": 1,
               "target_case_count_per_model": 96, "target_total_answers": 192, "total_answers": 0,
               "runtime_profile": state["configuration"]["runtime_profile"],
               "security_mode": state["configuration"]["security_mode"],
               "scientific_evaluation_status": "NOT_RUN"}
    for model in collector.MODELS:
        rows = []
        for slot in sorted(state["slots"], key=lambda slot: slot["case_id"]):
            if slot["model_id"] != model or slot["status"] != "COMPLETED":
                continue
            path = (output / slot["run_record_path"]).resolve()
            if (not path.is_relative_to(output) or not path.is_file()
                    or collector.original._sha256(path) != slot.get("run_record_sha256")):
                raise ValueError(f"invalid completed record: {slot['slot_id']}")
            record = collector.original._read(path)
            answer = path.with_name("final_answer.md")
            if (record.get("model_id") != model or record.get("case_id") != slot["case_id"]
                    or record.get("trial_index") != 1 or record.get("run_status") != "COMPLETED"
                    or not answer.is_file() or not answer.read_text(encoding="utf-8").strip()
                    or collector.original._sha256(answer) != record.get("final_answer_sha256")):
                raise ValueError(f"invalid completed answer: {slot['slot_id']}")
            trajectory = path.with_name("trajectory.json")
            if not trajectory.is_file() or not isinstance(json.loads(trajectory.read_text()), list):
                raise ValueError(f"missing or invalid trajectory: {slot['slot_id']}")
            rows.append({"model_id": model, "case_id": slot["case_id"], "dataset_id": slot["dataset_id"],
                "trial_index": slot["trial_index"], "final_answer": path.with_name("final_answer.md").read_text(encoding="utf-8"),
                "final_answer_sha256": record["final_answer_sha256"], "run_record_path": slot["run_record_path"],
                "run_record_sha256": slot["run_record_sha256"], "attempt_count": len(slot.get("attempt_history", [])) + 1,
                "wall_clock_time": record.get("wall_clock_time"), "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"), "python_execution_count": record.get("python_execution_count"),
                "network_isolation_active": record.get("network_isolation_active")})
        suffix = "" if complete[model] else ".partial"
        destination = output / f"{model}_answers{suffix}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(destination)
        summary["total_answers"] += len(rows)
        summary["models"][model] = {"status": "COMPLETE" if complete[model] else "PARTIAL",
            "answer_count": len(rows), "answers_file": destination.name,
            "sha256": collector.original._sha256(destination),
            "cases_with_retries": sum(row["attempt_count"] > 1 for row in rows),
            "prior_attempts_preserved": sum(row["attempt_count"] - 1 for row in rows),
            "slot_status_counts": state["by_model"][model]}
    summary_name = "completion_summary.json" if summary["status"] == "COMPLETE" else "partial_summary.json"
    collector.original._write(output / summary_name, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=sequence.collector.HOST_NETWORK_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly export only completed cases, with PARTIAL labels")
    args = parser.parse_args()
    print(json.dumps(export(args.output_root, allow_partial=args.allow_partial), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
