#!/usr/bin/env python3
"""Audit reuse of an identical visible judging task after an evaluator repair.

This does not produce judgments. Every reused answer retains its actual reviewer
and original request/response hashes. Changed evidence or conflicting prior
judgments require review.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowintentbench.external_file_evaluator import digest, write_json
from flowintentbench.expansion_evaluation import file_sha256

POLICY = "identical-visible-judgment-task-v1"
IDENTITY_KEYS = ("schema_version", "context", "operation", "input", "prompt",
                 "visible_input_sha256", "output_schema")


def reusable_verdict(request, answer):
    """Validate an existing verdict; unresolved evidence is not a retry license."""
    if answer.get('request_id') != request['request_id']:
        raise ValueError('Stored judgment identity mismatch')
    reviewer = answer.get('reviewer_id')
    if (not isinstance(reviewer, str) or not reviewer.strip()
            or reviewer == 'INDEPENDENT_REVIEWER_ID'):
        raise ValueError('Stored judgment lacks reviewer provenance')
    status = answer.get('status')
    if status == 'RESOLVED' and isinstance(answer.get('response'), dict):
        return dict(status=status, response=answer['response'])
    if (status == 'PENDING' and 'response' in answer and answer['response'] is None
            and isinstance(answer.get('reason'), str) and answer['reason'].strip()):
        return dict(status=status, response=None, reason=answer['reason'])
    raise ValueError('Stored judgment lacks a valid resolved or pending verdict')


def visible_task_signature(request):
    if request.get("schema_version") != "external-judgment-v1":
        raise ValueError("unsupported judgment request schema")
    if request["visible_input_sha256"] != digest(request["input"]):
        raise ValueError("judgment visible-input hash mismatch")
    if request["request_id"] != digest({key: request[key] for key in IDENTITY_KEYS}):
        raise ValueError("judgment request identity mismatch")
    normalized = deepcopy(request)
    for key in ("request_id", "status", "response_template"):
        normalized.pop(key, None)
    snapshot = normalized["context"].pop("evaluation_manifest_sha256", None)
    if not isinstance(snapshot, str) or len(snapshot) != 64:
        raise ValueError("judgment requires an evaluator snapshot binding")
    # All context apart from this version digest remains in the signature,
    # including case/material/answer hashes. Prompt, schema, role, full input,
    # review rules and even additional request fields must match exactly.
    return digest(normalized)


class JudgmentReuseIndex:
    """Cache parsing/validation only; stat changes invalidate immutable-file entries."""
    def __init__(self):
        self.entries = {}
        self.parsed_files = 0

    def load(self, path, *, request=False):
        from scripts.console_evaluation_cache import file_stamp
        key = (str(path.resolve()), request)
        stamp = file_stamp(path)
        old = self.entries.get(key)
        if old is not None and old[0] == stamp:
            return old[1]
        value = json.loads(path.read_text())
        result = (value, visible_task_signature(value)) if request else value
        self.entries[key] = (stamp, result)
        self.parsed_files += 1
        return result


def carry_forward(exchange, inventory, *, apply=False, cache=None):
    cache = cache if cache is not None else JudgmentReuseIndex()
    exchange = Path(exchange)
    active = json.loads(Path(inventory).read_text())
    target_ids = {row["request_id"] for group in active["by_operation"].values() for row in group}
    previous = defaultdict(list)
    for path in sorted((exchange / "requests").glob("*.json")):
        request, signature = cache.load(path, request=True)
        response_path = exchange / "responses" / (request["request_id"] + ".json")
        if not response_path.exists():
            continue
        response = cache.load(response_path)
        if response.get("request_id") != request["request_id"]:
            raise ValueError("stored judgment response identity mismatch")
        if response.get("status") not in {"RESOLVED", "PENDING"}:
            continue
        reusable_verdict(request, response)
        previous[signature].append((request, response, path, response_path))
    outcomes = []
    for request_id in sorted(target_ids):
        request_path = exchange / "requests" / (request_id + ".json")
        request, signature = cache.load(request_path, request=True)
        if request["request_id"] != request_id:
            raise ValueError("active request filename identity mismatch")
        target = exchange / "responses" / (request_id + ".json")
        if target.exists():
            outcomes.append({"request_id": request_id, "status": "EXISTING_RESPONSE_PRESERVED"})
            continue
        candidates = [row for row in previous[signature]
                      if row[0]["context"]["evaluation_manifest_sha256"]
                      != request["context"]["evaluation_manifest_sha256"]]
        if not candidates:
            outcomes.append({"request_id": request_id, "status": "REQUIRES_REVIEW"})
            continue
        if len({digest(reusable_verdict(row[0], row[1])) for row in candidates}) != 1:
            outcomes.append({"request_id": request_id, "status": "CONFLICT_REQUIRES_REVIEW"})
            continue
        source, answer, source_path, answer_path = candidates[0]
        provenance = {"policy": POLICY, "visible_task_signature": signature,
                      "source_request_id": source["request_id"],
                      "source_request_sha256": file_sha256(source_path),
                      "source_response_sha256": file_sha256(answer_path),
                      "source_evaluation_manifest_sha256": source["context"]["evaluation_manifest_sha256"],
                      "target_evaluation_manifest_sha256": request["context"]["evaluation_manifest_sha256"],
                      "target_request_sha256": file_sha256(request_path)}
        envelope = {"request_id": request_id, "reviewer_id": answer["reviewer_id"],
                    "status": answer["status"], "response": answer["response"], "carry_forward": provenance}
        if 'provenance' in answer:
            envelope['provenance'] = deepcopy(answer['provenance'])
        if 'reason' in answer:
            envelope['reason'] = answer['reason']
        if apply:
            # Migration is run with reviewers stopped. Exclusive creation also
            # prevents overwriting a response if a reviewer finishes meanwhile.
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as handle:
                json.dump(envelope, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            write_json(exchange / "carry_forward" / (request_id + ".json"),
                       {**provenance, "target_request_id": request_id,
                        "target_response_sha256": file_sha256(target),
                        "original_response": answer})
        outcomes.append({"request_id": request_id, "source_request_id": source["request_id"],
                         "status": "CARRIED_FORWARD" if apply else "IDENTICAL_TASK_REUSABLE"})
    return {"policy": POLICY, "applied": apply, "results": outcomes,
            "new_scientific_judgments": 0, "api_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(carry_forward(args.exchange, args.inventory, apply=args.apply), ensure_ascii=False))
