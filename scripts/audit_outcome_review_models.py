"""Audit scored review identities without confusing requested and returned models."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.expansion_evaluation import read_json
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.outcome_scoring import digest
from scripts.evaluate_answered_outcomes import sha_file
from scripts.run_core_case_scoring import _parse_json
from scripts.benchmark_runtime import require_benchmark_runtime


def response_judgment(response):
    if "output" in response:
        content = "\n".join(part["text"] for item in response["output"] if item.get("type") == "message"
                            for part in item.get("content", []) if part.get("type") == "output_text")
    else:
        content = response["choices"][0]["message"]["content"]
    return _parse_json(content)


def audit(source):
    rows = read_json(source / "selection.json")["answers"]
    records = []
    for row in rows:
        result = read_json(source / "cases" / (row["run_id"] + ".json"))
        if result["status"] != "SCORED":
            continue
        if result["reviewer_model"] != "gpt-6-astra":
            raise ValueError("unexpected scored reviewer model")
        record = {"run_id": row["run_id"], "reviewer_source": result.get("reviewer_source", "HTTP_API")}
        if record["reviewer_source"] == "EXPLICIT_GPT6_AGENT":
            provenance = result["agent_review_provenance"]
            if (provenance["requested_model"] != "gpt-6-astra" or result["reviewer_receipts"]
                    or sha_file(Path(provenance["raw_judgment_path"])) != provenance["raw_judgment_sha256"]
                    or digest(read_json(Path(provenance["raw_judgment_path"]))) != result["judgment_sha256"]):
                raise ValueError("invalid agent review identity")
            record.update(evidence="EXPLICIT_AGENT_SELECTION", actual_http_model=None,
                          requested_model=provenance["requested_model"], agent_task=provenance["agent_task"])
        else:
            matching = []
            for location in result["reviewer_receipts"]:
                path = Path(location)
                receipt = read_json(path)
                if not receipt.get("completed"):
                    continue
                try:
                    matches = digest(_parse_json(receipt["final_text"])) == result["judgment_sha256"]
                except (ValueError, TypeError):
                    matches = False
                if not matches:
                    continue
                if receipt.get("transport") == "responses":
                    response = read_json(path.parent / "response.json")
                    if receipt.get("response_model") != "gpt-6-astra" or response.get("model") != "gpt-6-astra":
                        raise ValueError("HTTP model mismatch in scored review")
                    if (response.get("status") != "completed" or not response.get("id")
                            or response["id"] != receipt.get("response_id")
                            or digest(response_judgment(response)) != result["judgment_sha256"]):
                        raise ValueError("actual response body or id does not match scored judgment")
                    matching.append({"receipt_path": str(path), "sha256": sha_file(path),
                                     "response_path": str(path.parent / "response.json"),
                                     "response_sha256": sha_file(path.parent / "response.json")})
                else:
                    accepted = []
                    actual_judgment_matched = False
                    for http_path in (path.parent / "http").glob("*/http_receipt.json"):
                        http = read_json(http_path)
                        if http.get("status") != "ACCEPTED":
                            continue
                        response_path = http_path.parent / "response.json"
                        response = read_json(response_path)
                        if (http.get("response_model") != "gpt-6-astra" or response.get("model") != "gpt-6-astra"
                                or http.get("response_sha256") != sha_file(response_path)
                                or not response.get("id") or response["id"] != http.get("response_id")):
                            raise ValueError("legacy HTTP model or response hash mismatch")
                        try:
                            actual_judgment_matched |= digest(response_judgment(response)) == result["judgment_sha256"]
                        except (ValueError, TypeError, KeyError, IndexError):
                            pass
                        accepted.append({"receipt_path": str(http_path), "sha256": sha_file(http_path),
                                         "response_path": str(response_path), "response_sha256": sha_file(response_path)})
                    if accepted and actual_judgment_matched:
                        matching.append({"receipt_path": str(path), "sha256": sha_file(path), "http": accepted})
            if not matching:
                raise ValueError("no actual-model receipt bound to judgment: " + row["run_id"])
            record.update(evidence="HTTP_RESPONSE_MODEL", actual_http_model="gpt-6-astra", receipts=matching)
        records.append(record)
    report = {"expected": len(rows), "scored_audited": len(records),
              "evidence_counts": dict(Counter(row["evidence"] for row in records)), "records": records}
    write_json(source / "reports/review_identity_audit.json", report)
    return {key: value for key, value in report.items() if key != "records"}


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    args = parser.parse_args()
    import json
    print(json.dumps(audit(args.source.resolve()), indent=2))


if __name__ == "__main__":
    main()
