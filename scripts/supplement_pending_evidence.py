#!/usr/bin/env python3
"""Create content-bound continuations for pending requests with answer evidence.

This is a transport/cache maintenance pass.  It does not score, infer units,
execute a recipe, or contact a model.  Historical request files remain
immutable; only a new request is created when the original answer contains a
single explicit ``## Operationalization`` section.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.external_file_evaluator import FileJudgmentTransport, digest, adjudication_schema
from scripts.run_expansion_file_evaluation import operationalization_source_section


def require_benchmark() -> None:
    from scripts.benchmark_runtime import require_benchmark_runtime
    require_benchmark_runtime()

def main() -> int:
    require_benchmark()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/claude_resume_eval")
    args = parser.parse_args()
    exchange = args.output / "exchange"

    # The final response is invocation-bound by its digest in each request.
    responses_by_hash = {}
    for path in args.output.rglob("run_record.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        response = record.get("final_response")
        if isinstance(response, str):
            responses_by_hash[digest(response)] = response

    created = []
    skipped = {"not_pending": 0, "not_unit": 0, "no_answer": 0,
               "no_unique_operationalization": 0, "already_enriched": 0}
    for path in sorted((exchange / "requests").glob("*.json")):
        try:
            request = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if request.get("status") != "PENDING":
            skipped["not_pending"] += 1
            continue
        payload = request.get("input") or {}
        context = payload.get("unit_review_context") or {}
        if request.get("operation") != "adjudication" or payload.get("pending_type") != "unit_relationship" or not context:
            skipped["not_unit"] += 1
            continue
        if context.get("operationalization_source_section") is not None:
            skipped["already_enriched"] += 1
            continue
        answer = responses_by_hash.get((request.get("context") or {}).get("response_sha256"))
        if answer is None:
            skipped["no_answer"] += 1
            continue
        section = operationalization_source_section(answer)
        if section is None:
            skipped["no_unique_operationalization"] += 1
            continue
        enriched = json.loads(json.dumps(payload, ensure_ascii=False))
        enriched.setdefault("unit_review_context", {})["operationalization_source_section"] = section
        transport = FileJudgmentTransport(exchange, context=request["context"])
        new_id = None
        try:
            transport.request("adjudication", enriched, output_schema=adjudication_schema())
        except Exception as exc:
            # A new request normally remains PENDING, which is the expected
            # outcome.  Any other failure is surfaced in the report.
            name = type(exc).__name__
            if name != "EvaluationPendingAdjudication":
                raise
            new_id = exc.continuation_context.get("external_request_id")
        if not new_id:
            raise RuntimeError("evidence supplement did not publish a request id")
        if new_id == path.stem:
            raise RuntimeError("evidence supplement did not change request identity")
        created.append({"old_request_id": path.stem, "new_request_id": new_id,
                        "case_id": (request.get("context") or {}).get("case_id")})

    report = {"status": "PASS", "created_count": len(created),
              "created": created, "skipped": skipped,
              "source": "original_run_record.final_response",
              "api_calls": 0}
    report_path = args.output / "supplement_pending_evidence.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "created_count": len(created),
                      "skipped": skipped, "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
