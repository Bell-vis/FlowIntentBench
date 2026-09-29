#!/usr/bin/env python3
"""Budgeted, resumable evaluation with local uncertainty and a full slot ledger."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.core_case_scoring import _reference_packet
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.answer_collections import collect
from flowintentbench.trusted_scoring import from_outcome  # historical import compatibility only
from flowintentbench.rubric_scoring import (VERSION, METRICS, Judgment, digest, empty_metrics, review_schema,
                                          interval, review_prompt, score)
from flowintentbench.reference_packages import load_supplements, apply_supplement, package
from scripts.path_migration import resolve_repository_path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validated_cache(path, identity, prompt, schema, model, *, effort=None, max_output_tokens=None, structured_output=False):
    stored = read(path)
    request = read(path.parent.parent / "requests" / (identity + ".json"))
    return validate_review_record(stored, request, identity, prompt, schema, model,
                                 effort=effort, max_output_tokens=max_output_tokens, structured_output=structured_output)


def validate_review_record(stored, request, identity, prompt, schema, model, *, effort=None, max_output_tokens=None, structured_output=False):
    from scripts.run_core_case_scoring import _parse_json
    if stored["request_id"] != identity or stored["judgment_sha256"] != digest(stored["judgment"]):
        raise ValueError("trusted cache identity mismatch")
    if request["prompt"] != prompt or request["schema"] != schema:
        raise ValueError("trusted cache prompt/schema mismatch")
    if bool(request.get('structured_output')) != structured_output:
        raise ValueError('trusted cache structured output mode mismatch')
    if structured_output:
        from jsonschema import Draft202012Validator
        Draft202012Validator(schema).validate(stored['judgment'])
    receipt = read(resolve(stored["receipt"]))
    if receipt.get('transport') == 'codex_exec_chatgpt_login':
        from scripts.outcome_codex_transport import audit_cli
        cli_payload = read(resolve(stored['receipt']).with_name('request.json'))
        if cli_payload.get('schema') != schema:
            raise ValueError('CLI review schema differs from current request')
        audit_cli(receipt, cli_payload)
    if (not receipt.get("completed") or receipt.get("response_model") != model
            or _parse_json(receipt["final_text"]) != stored["judgment"]):
        raise ValueError("trusted cache is not bound to a completed reviewer response")
    if effort is not None:
        payload = read(resolve(stored["receipt"]).with_name("request.json"))
        if (receipt.get("reasoning_effort") != effort or payload.get("max_output_tokens") != max_output_tokens
                or payload.get("input", [{}])[0].get("content") != prompt):
            raise ValueError("cached reviewer payload/effort/budget differs from current request")
        if structured_output and payload.get('text',{}).get('format') != {
                'type':'json_schema','name':'answer_evaluation','schema':schema,'strict':True}:
            raise ValueError('cached reviewer structured schema differs from current request')
    return stored


def replay_record(directory, identity, prompt, schema, model, effort, budget, structured_output=False):
    original = directory / "judgments" / (identity + ".json")
    if original.exists():
        stored = validated_cache(original, identity, prompt, schema, model, effort=effort, max_output_tokens=budget, structured_output=structured_output)
        stored["host_replay_source"] = {"judgment": str(original), "sha256": sha(original)}
        return stored
    request_path = directory / "requests" / (identity + ".json")
    if not request_path.exists():
        return None
    from scripts.run_core_case_scoring import _parse_json
    # A completed API response may have failed the old host's validation.
    # Recover that exact response; never parse a truncated streaming draft.
    for receipt_path in sorted((directory / "api").glob(identity + "-*/receipt.json")):
        receipt = read(receipt_path)
        if not receipt.get("completed"):
            continue
        judgment = _parse_json(receipt["final_text"])
        stored = {"request_id": identity, "judgment": judgment, "judgment_sha256": digest(judgment),
                  "receipt": str(receipt_path)}
        validate_review_record(stored, read(request_path), identity, prompt, schema, model,
                               effort=effort, max_output_tokens=budget, structured_output=structured_output)
        stored["host_replay_source"] = {"completed_receipt": str(receipt_path), "sha256": sha(receipt_path)}
        return stored
    return None


def prior_review_index(directories):
    """Index old extraction evidence by identical scientific input, not scores.

    v2 adds optional auxiliary checks. A v1 response can be rescored with those
    checks absent; it must not be relabeled as a response to the new prompt.
    """
    index = defaultdict(list)
    for directory in directories:
        if directory is None:
            continue
        for path in sorted((Path(directory) / "requests").glob("*.json")):
            request = read(path)
            if set(request.get("schema", {}).get("properties", {})) != {"dimensions", "findings", "extraction_complete", "limitations"}:
                continue
            try:
                packet = json.loads(request["prompt"].split("\n\n", 1)[1])
            except (KeyError, IndexError, ValueError):
                continue
            packet.pop("auxiliary_verification_catalog", None)
            packet.pop("frozen_auxiliary_catalog", None)
            for folder in ("judgments", "reviews"):
                cached_path = Path(directory) / folder / path.name
                if cached_path.exists():
                    index[digest(packet)].append((cached_path, request))
    return index


def prior_review(index, prompt, model, *, effort="medium", max_output_tokens=6000):
    from scripts.run_core_case_scoring import _parse_json
    try:
        packet = json.loads(prompt.split("\n\n", 1)[1])
    except (IndexError, ValueError):
        return None
    packet.pop("auxiliary_verification_catalog", None)
    packet.pop("frozen_auxiliary_catalog", None)
    candidates = index.get(digest(packet), [])
    # Prefer the richer extraction schema, independently of answers or scores.
    candidates = sorted(candidates, key=lambda item: (
        '"frozen_auxiliary_catalog"' in item[1].get("prompt", ""),
        "numeric_checks" in item[1].get("schema", {}).get("$defs", {}).get("Finding", {}).get("properties", {})), reverse=True)
    for path, request in candidates:
        stored = read(path)
        identity = stored.get("request_id", stored.get("id"))
        receipt = read(resolve(stored["receipt"]))
        if receipt.get("response_model") != model or receipt.get("reasoning_effort") != effort:
            continue
        payload_path = resolve(stored["receipt"]).with_name("request.json")
        if not payload_path.exists():
            continue
        payload = read(payload_path)
        if payload.get("max_output_tokens") != max_output_tokens:
            continue
        if payload.get("input", [{}])[0].get("content") != request["prompt"]:
            raise ValueError("prior extraction request payload changed")
        if (identity != path.stem or digest(stored["judgment"]) != stored["judgment_sha256"]
                or not receipt.get("completed") or _parse_json(receipt["final_text"]) != stored["judgment"]):
            raise ValueError("prior extraction differs from completed reviewer response")
        return path, request, stored
    return None


def resolve(path, relative=None):
    return resolve_repository_path(path, repository_root=ROOT, relative_root=relative)


def load_source(selected, donor, case_input, gt):
    """Bind a saved judgment and recover only hash-identical original answer text."""
    prior_path = donor / "cases" / (selected["run_id"] + ".json")
    prior = read(prior_path)
    rid = prior.get("request_id") or prior.get("baseline_request_id")
    request_path, judgment_path = donor / "requests" / (rid + ".json"), donor / "judgments" / (rid + ".json")
    request, stored = read(request_path), read(judgment_path)
    old_judgment = stored["judgment"]
    if digest(old_judgment) != prior["judgment_sha256"]:
        raise ValueError("donor judgment digest mismatch")
    packet = json.loads(request["prompt"].split("\n\n", 1)[1])
    if (packet["reference_examples"] != _reference_packet(gt) or packet["question"] != case_input.scientific_question
            or packet["case_context"] != case_input.case_context.model_dump(mode="json")
            or packet["public_metadata"] != case_input.flow_data.data_metadata.model_dump(mode="json")):
        raise ValueError("donor question/reference differs from current frozen task")
    answer = packet["answer"]
    if hashlib.sha256(answer.encode()).hexdigest() != selected["answer_sha256"]:
        raise ValueError("cached prompt answer digest mismatch")
    if prior["answer_sha256"] != selected["answer_sha256"] or prior["run_id"] != selected["run_id"]:
        raise ValueError("donor answer identity mismatch")
    run_path = resolve(selected["run_path"])
    if run_path.exists():
        if sha(run_path) != selected["run_sha256"] or read(run_path)["final_response"] != answer:
            raise ValueError("original RunRecord digest/answer mismatch")
    return answer, old_judgment, {"source": "CACHED_OUTCOME_EVIDENCE", "request_id": rid,
        "judgment_path": str(judgment_path), "judgment_file_sha256": sha(judgment_path),
        "request_file_sha256": sha(request_path), "answer_source": str(request_path),
        "original_run_file_available": run_path.exists(),
        "reviewer_source": prior.get("reviewer_source"), "reviewer_receipts": prior.get("reviewer_receipts", []),
        "reasoning_effort": prior.get("reasoning_effort"),
        "reuse_limit": "No MET/PARTIAL to original metric conversion; unsupported fields remain unknown"}


def load_run_source(slot, collection, experiment_id):
    """New answers need no donor review, but must belong to the frozen ledger."""
    path = resolve(slot["run_record_path"], collection)
    if sha(path) != slot["run_record_sha256"]:
        raise ValueError("run/ledger digest mismatch")
    run = read(path)
    for key, expected in (("case_id", slot["case_id"]), ("model_id", slot["model_id"]),
                          ("trial_index", slot["trial_index"]), ("experiment_id", slot.get("run_experiment_id", experiment_id)),
                          ("run_status", "COMPLETED")):
        if run.get(key) != expected:
            raise ValueError(f"run/ledger {key} mismatch")
    answer = run.get("final_response")
    if not isinstance(answer, str):
        raise ValueError("completed run has no textual answer")
    if run.get("final_answer_sha256") and run["final_answer_sha256"] != hashlib.sha256(answer.encode()).hexdigest():
        raise ValueError("final answer digest mismatch")
    return answer, None, {"source": "ORIGINAL_RUN", "answer_source": str(path),
                          "run_sha256": sha(path), "run_id": run["run_id"],
                          "collection_experiment_id": experiment_id, "run_experiment_id": run.get("experiment_id")}


def optional_mean_bounds(values):
    """Sharp mean bounds when some entries may be inapplicable.

    An optional contribution enters both numerator and denominator. For a
    fixed optional count, the smallest lower endpoints / largest upper
    endpoints attain the extremes. Condition on a nonempty applicable set;
    callers separately retain unknown applicability.
    """
    definite = [v for v in values if not v.get('applicability_unknown')]
    optional = [v for v in values if v.get('applicability_unknown')]
    if not values:
        return None, None
    bounds = []
    for key, descending, choose in (('lower', False, min), ('upper', True, max)):
        endpoint = lambda v: v[key] if v.get(key) is not None else (0.0 if key == 'lower' else 1.0)
        total = sum(endpoint(v) for v in definite)
        count = len(definite)
        candidates = [total / count] if count else []
        for value in sorted((endpoint(v) for v in optional), reverse=descending):
            total += value
            count += 1
            candidates.append(total / count)
        bounds.append(choose(candidates))
    return tuple(bounds)


def summarize(rows, metric_names=METRICS):
    output = {}
    for name in metric_names:
        applicable = [r for r in rows if r["metrics"][name]["applicable"]
                      or r["metrics"][name].get('applicability_unknown')]
        by_case = defaultdict(list)
        for r in applicable:
            by_case[r["case_id"]].append(r["metrics"][name])
        groups = []
        for group in by_case.values():
            lo, hi = optional_mean_bounds(group)
            groups.append({'lower': lo, 'upper': hi,
                           'applicability_unknown': all(v.get('applicability_unknown') for v in group)})
        bounds = optional_mean_bounds(groups)
        known = sum(r["metrics"][name]["status"] == "POINT" and not r["metrics"][name].get("applicability_unknown") for r in applicable)
        output[name] = {"case_macro_lower": bounds[0], "case_macro_upper": bounds[1],
            "value": bounds[0] if known == len(applicable) and applicable else None,
            "applicable_trials": len(applicable), "resolved_trials": known,
            "coverage": known / len(applicable) if applicable else None,
            "applicable_cases": len(by_case)}
        output[name]["unknown_applicability_trials"] = sum(r["metrics"][name].get("applicability_unknown", False) for r in applicable)
        output[name]["supported_value"] = bounds[0]
        output[name]["value_kind"] = ("NOT_APPLICABLE" if not applicable else
            "EXACT" if output[name]["value"] is not None else "EVIDENCE_LOWER_BOUND")
    return output


def paired_comparison(rows):
    models = sorted({r["model_id"] for r in rows})
    result = []
    for ai, left in enumerate(models):
        for right in models[ai + 1:]:
            a = {(r["case_id"], r["trial"]): r for r in rows if r["model_id"] == left and r["collection_status"] == "COMPLETED"}
            b = {(r["case_id"], r["trial"]): r for r in rows if r["model_id"] == right and r["collection_status"] == "COMPLETED"}
            keys = sorted(a.keys() & b.keys())
            metrics = {}
            for name in METRICS:
                pairs = [(a[k]["metrics"][name], b[k]["metrics"][name], k[0]) for k in keys
                         if a[k]["metrics"][name]["applicable"] and b[k]["metrics"][name]["applicable"]]
                grouped = defaultdict(list)
                for x, y, c in pairs:
                    grouped[c].append((x["lower"] - y["upper"], x["upper"] - y["lower"]))
                bounds = [statistics.fmean(statistics.fmean(p[i] for p in group) for group in grouped.values())
                          if grouped else None for i in (0, 1)]
                metrics[name] = {"pairs": len(pairs), "cases": len(grouped),
                    "left_minus_right_bounds": bounds,
                    "left_definite_wins": sum(x["lower"] > y["upper"] for x, y, _ in pairs),
                    "right_definite_wins": sum(y["lower"] > x["upper"] for x, y, _ in pairs),
                    "unresolved_or_tied": sum(not (x["lower"] > y["upper"] or y["lower"] > x["upper"]) for x, y, _ in pairs)}
            result.append({"left": left, "right": right, "matched_answer_pairs": len(keys), "metrics": metrics})
    return result


def write_report(output, rows, costs, source_identity):
    if costs.get("api_calls") or costs.get('cli_calls'):
        write_json(output / "cost_history" / (digest(costs) + ".json"), costs)
    history = [read(p) for p in sorted((output / "cost_history").glob("*.json"))]
    cumulative = {key: sum(c.get(key, 0) for c in history)
                  for key in ("api_calls", "cli_calls", "input_tokens", "output_tokens", "usage_missing_calls", "api_errors", "cli_errors", "output_limit_exceeded_calls")}
    cumulative["token_totals_complete"] = cumulative["usage_missing_calls"] == 0
    answered = [r for r in rows if r["collection_status"] == "COMPLETED"]
    models = sorted({r["model_id"] for r in rows})
    identified = all(all(m["status"] != "INTERVAL" and not m.get("applicability_unknown")
                         for m in r["metrics"].values()) for r in rows)
    reviewed = all(r["status"] in {"SCORED", "MODEL_NONCOMPLETION"} for r in rows)
    for row in rows:
        row["metric_values"] = {k: m.get("supported_value", m["lower"]) for k, m in row["metrics"].items()}
        row["metric_value_kinds"] = {k: m.get("value_kind") for k, m in row["metrics"].items()}
        row["metric_values_interpretation"] = "EXACT or EVIDENCE_LOWER_BOUND, never imputed true quality; see bounds and applicability"
        if row.get("output_id"):
            write_json(output / "cases" / (row["output_id"] + ".json"), row)
    report = {"protocol": VERSION, "updated_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity, "requested_slots": len(rows), "answered_slots": len(answered),
        "status": ("COMPLETE" if identified else "COMPLETE_WITH_BOUNDS") if reviewed else "INCOMPLETE",
        "scientific_identification_status": "COMPLETE" if reviewed and identified else "PARTIAL",
        "evaluation_boundary": "ANSWER_AND_FROZEN_EVIDENCE_ONLY; NO_SOLVING_OR_RAW_DATA_ACCESS",
        "published_values": "supported_value is the lower identification bound; exact value remains null when evidence is unresolved. Neither bound coverage nor review completion is judge accuracy.",
        "answered_review_status": "COMPLETE" if answered and all(r["status"] in {"SCORED", "MODEL_NONCOMPLETION"} for r in answered) else "INCOMPLETE",
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "aggregation": "Equal case weight; trials averaged within case; all requested missing applicable trials contribute [0,1]. No N3 closure filter.",
        "uncertainty": "Identification bounds, not confidence intervals. C applicability can be unknown for missing answers.",
        "model_performance_correlation": {"status": "NOT_ESTABLISHED", "reason": "No independent human/model-performance reference labels supplied; use paired task results and controlled sensitivity tests."},
        "models": {m: {"all_requested": summarize([r for r in rows if r["model_id"] == m]),
                       "answered_only": summarize([r for r in answered if r["model_id"] == m]),
                       "by_condition": {c: summarize([r for r in rows if r["model_id"] == m and r["condition"] == c])
                                        for c in sorted({r["condition"] for r in rows})}}
                   for m in models}, "paired_comparisons": paired_comparison(rows),
        "verified_error_counts": {m: sum(len(r.get("error_diagnostics", {}).get("independent_contradicted_finding_ids", []))
                                         for r in rows if r["model_id"] == m) for m in models},
        "evaluation_cost": costs, "cumulative_new_review_cost": cumulative, "rows": rows}
    report["branch_alignment_diagnostics"] = {}
    report["of_consistency_diagnostics"] = {}
    for model in models:
        selected = [r for r in answered if r["model_id"] == model]
        informative = [r for r in selected if r.get("branch_alignment_diagnostic", {}).get("informative")]
        values = [r["metrics"]["branch_alignment"]["value"] for r in informative]
        report["branch_alignment_diagnostics"][model] = {
            "kinds": dict(Counter(r.get("branch_alignment_diagnostic", {}).get("kind", "UNREVIEWED") for r in selected)),
            "informative_trials": len(values),
            "informative_mean": statistics.fmean(values) if values else None,
            "informative_case_macro": summarize(informative, ("branch_alignment",))["branch_alignment"],
            "informative_trial_coverage": len(values) / len(selected) if selected else None,
            "interpretation": "Single-branch and all-branch ties excluded; raw alignment is retained for reproducibility."}
        diagnostic_names = ("conditional_required_support", "binding_gap", "informative_branch_alignment")
        diagnostic_rows = [{"case_id": r["case_id"], "metrics": {
            name: r["of_consistency"][name] for name in diagnostic_names}}
            for r in selected if r.get("of_consistency")]
        report["of_consistency_diagnostics"][model] = {
            "evaluated_trials": len(diagnostic_rows), "answered_trials": len(selected),
            "demonstrated_mismatch_trials": sum(r.get("of_consistency", {}).get("mismatch_demonstrated", False) for r in selected),
            "case_macro": summarize(diagnostic_rows, diagnostic_names),
            "interpretation": "Conditional diagnostics over reviewed applicable answers; absent reviews are counted separately."}
    report["review_failures"] = {"source_errors": sum(r["status"] == "SOURCE_ERROR" for r in rows),
        "review_errors": sum(bool(r.get("review_error")) or r["status"] == "REVIEW_ERROR" for r in rows),
        "unreviewed_answers": sum(r["status"] not in {"SCORED", "MODEL_NONCOMPLETION"} for r in answered)}
    report["rubric_diagnostics"] = {model: {
        "item_states": dict(Counter(x["state"] for r in answered if r["model_id"] == model for x in r.get("rubric_items", []))),
        "unresolved_reasons": dict(Counter(x["reason"] for r in answered if r["model_id"] == model for x in r.get("reference_gaps", []))),
        "result_groups": sum(len(r.get("result_groups", [])) for r in answered if r["model_id"] == model),
        "extraction_incomplete": sum(r.get("extraction_complete") is False for r in answered if r["model_id"] == model),
    } for model in models}
    gaps = {}
    for row in rows:
        for gap in row.get("reference_gaps", []):
            key = (row["case_id"], gap["dedup_key"])
            entry = gaps.setdefault(key, {"case_id": row["case_id"], **gap, "affected_answers": []})
            entry["affected_answers"].append({"model_id": row["model_id"], "trial": row["trial"], "output_id": row["output_id"]})
    write_json(output / "reports" / "reference_gaps.json", {"protocol": VERSION,
        "note": "Do not repeatedly judge missing numeric truth. Build and freeze references separately.",
        "items": list(gaps.values())})
    report_dir = output / "reports"
    write_json(report_dir / "experiment_report.json", report)
    columns = ["run_id", "model_id", "case_id", "trial", "condition", "collection_status", "status"]
    columns += [name + "_" + key for name in METRICS for key in ("value", "supported_value", "lower", "upper", "status", "value_kind")]
    with (report_dir / "trial_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            line = {k: r.get(k) for k in columns[:7]}
            line.update({name + "_" + key: v[key] for name, v in r["metrics"].items() for key in ("value", "supported_value", "lower", "upper", "status", "value_kind")})
            writer.writerow(line)
    lines = ["# Trusted metric evaluation", "", f"Requested slots: {len(rows)}; completed answers: {len(answered)}.",
        f"Metric completion: {report['status']}; answered review disposition: {report['answered_review_status']}.", "",
        "Bounds represent unresolved evidence. They are not confidence intervals. Model ranking is not established.",
        "supported_value publishes the evidenced lower bound, not an imputed quality score. N/A remains N/A.", "",
        "| Model | Metric | Answered case macro bounds | Resolved/applicable trials |", "|---|---|---|---|"]
    for model in models:
        for name, v in report["models"][model]["answered_only"].items():
            bounds = "N/A" if v["case_macro_lower"] is None else f"[{v['case_macro_lower']:.4f}, {v['case_macro_upper']:.4f}]"
            lines.append(f"| {model} | {name} | {bounds} | {v['resolved_trials']}/{v['applicable_trials']} |")
    lines += ["", "Independently contradicted auxiliary findings (zero means none established, not error-free):", "",
              "| Model | Contradicted findings |", "|---|---:|"]
    lines += [f"| {m} | {n} |" for m, n in report["verified_error_counts"].items()]
    lines += ["", "Evaluation cost for this invocation:", "", "```json", json.dumps(costs, ensure_ascii=False, indent=2), "```"]
    (report_dir / "experiment_report.md").write_text("\n".join(lines) + "\n")
    return report


def choose_paired(rows, limit):
    """Stable, condition-stratified sampling without looking at scores or model rank."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["trial"])].append(row)
    pairs = [sorted(g, key=lambda r: r["model_id"]) for g in groups.values() if len(g) > 1]
    conditions = defaultdict(list)
    for pair in sorted(pairs, key=lambda g: (g[0]["case_id"], g[0]["trial"])):
        conditions[pair[0]["condition"]].append(pair)
    chosen = []
    while conditions and len(chosen) < limit:
        for c in sorted(list(conditions)):
            group = conditions[c].pop(0)
            if len(group) <= limit - len(chosen):
                chosen.extend(group)
            if not conditions[c]:
                del conditions[c]
    return chosen


def prioritize_unattempted_reviews(rows, request_ids, directories):
    """Defer repeated failures behind untouched pairs without dropping a slot.

    Called only for requests lacking a usable review. Attempt counts affect
    dispatch order, never scoring, model ranking, or eligibility. Exact request
    IDs allow a host replay to retain retry history without importing verdicts.
    """
    attempts = Counter()
    seen = set()
    for directory in directories:
        if directory is None:
            continue
        for path in (Path(directory) / 'api').glob('*/receipt.json'):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            request_id = path.parent.name.split('-', 1)[0]
            if len(request_id) == 64 and all(c in '0123456789abcdef' for c in request_id):
                attempts[request_id] += 1
    pair_attempts = {}
    for row in rows:
        pair = row['case_id'], row['trial']
        pair_attempts[pair] = max(pair_attempts.get(pair, 0), attempts[request_ids[row['output_id']]])
    return sorted(rows, key=lambda row: pair_attempts[(row['case_id'], row['trial'])])


def fatal_provider_error(receipt):
    error = str(receipt.get("error") or "").lower()
    return receipt.get("http_status") in {401, 403} or any(
        word in error for word in ("insufficient_user_quota", "insufficient_quota", "额度不足", "invalid_api_key"))


class ProviderCircuit:
    """Drain in-flight work after repeated transport failures; retain all slots."""
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._streak = 0
        self.reason = None

    def is_set(self):
        return self._event.is_set()

    def observe(self, receipt):
        error = str(receipt.get('error') or '').lower()
        transport_failure = not receipt.get('completed') and any(term in error for term in (
            'upstream_error', 'upstream failed', 'servers are currently overloaded', 'service is busy',
            'unexpected_eof_while_reading', 'stream ended before response.completed', 'stream_read_error',
            'remotedisconnected', 'remote end closed connection without response', 'connection reset by peer'))
        with self._lock:
            self._streak = self._streak + 1 if transport_failure else 0
            if fatal_provider_error(receipt):
                self.reason = 'CREDENTIAL_OR_QUOTA_ERROR'
                self._event.set()
            elif self._streak >= 3 and not self.is_set():
                self.reason = 'THREE_CONSECUTIVE_TRANSPORT_FAILURES'
                self._event.set()


def run(args):
    if (args.max_api_calls < 0 or args.workers < 1 or args.shared_concurrency < 1
            or args.max_wall_seconds <= 0 or args.timeout <= 0
            or args.max_output_tokens < 1 or args.max_prompt_chars < 1):
        raise ValueError("invalid review budget")
    started = time.monotonic()
    output = args.output.resolve()
    donor = args.donor.resolve() if args.donor else None
    if args.reuse_trusted_from:
        raise ValueError("v4 requires fresh reviews; cross-run/legacy reviewer imports are disabled")
    replay = getattr(args, "replay_from", None)
    if replay:
        replay = replay.resolve()
        if read(replay / "evaluation_contract.json").get("protocol") != VERSION:
            raise ValueError("host replay requires the identical current review protocol")
    additional_replays=[p.resolve() for p in getattr(args,'additional_replay_from',[]) or []]
    for candidate in additional_replays:
        if read(candidate / 'evaluation_contract.json').get('protocol') != VERSION:
            raise ValueError('host replay requires the identical current review protocol')
    # Resume only this protocol and these frozen inputs. Never overwrite v3
    # results or quietly combine protocols after a code/reference change.
    previous_report = output / "reports/experiment_report.json"
    if previous_report.exists() and read(previous_report).get("protocol") != VERSION:
        raise ValueError("output contains another evaluation protocol; use a new output directory")
    manifest = args.manifest.resolve()
    slots, cases, sources = collect(args.collection, args.answers, manifest,
        models=args.models, case_ids=args.cases, trials=args.trials)
    selection_path = donor / "selection.json" if donor else None
    selection = read(selection_path) if selection_path and selection_path.exists() else {"answers": []}
    if selection_path and selection_path.exists() and selection["manifest_sha256"] != sha(manifest):
        raise ValueError("donor and evaluation manifests differ")
    selected = {r["run_id"]: r for r in selection["answers"]}
    source_identity = {"inputs": sources, "manifest_sha256": sha(manifest),
        "donor_selection_sha256": sha(selection_path) if selection_path and selection_path.exists() else None,
        "scoring_dependencies_sha256": {p: sha(ROOT / p) for p in (
            "scripts/evaluate_model_answers.py", "scripts/run_core_case_scoring.py",
            "scripts/outcome_responses_transport.py", "scripts/yiapi_transport.py", "flowintentbench/trusted_scoring.py",
            "scripts/outcome_codex_transport.py", "scripts/codex_console_transport.py",
            "flowintentbench/rubric_scoring.py", "flowintentbench/reference_packages.py",
            "flowintentbench/review_repairs.py",
            "flowintentbench/answer_evidence.py", "flowintentbench/answer_collections.py",
            "flowintentbench/frozen_evidence.py", "flowintentbench/evaluator.py",
            "flowintentbench/evaluation_policy.py", "flowintentbench/finding_requirements.py",
            "flowintentbench/answer_normalization.py")}}
    from flowintentbench.frozen_evidence import load_bank
    bank = load_bank(args.evidence_bank, ROOT) if args.evidence_bank else {}
    supplements = load_supplements(args.reference_package, ROOT) if args.reference_package else {}
    source_identity["evidence_bank_sha256"] = sha(args.evidence_bank) if args.evidence_bank else None
    source_identity["reference_package_sha256"] = sha(args.reference_package) if args.reference_package else None
    if replay:
        source_identity["host_replay"] = {"source": str(replay),
            "source_contract_sha256": sha(replay / "evaluation_contract.json"),
            "compatibility": "EXACT_PROMPT_SCHEMA_MODEL_EFFORT_BUDGET; SCORING_CODE_MAY_CHANGE"}
    if additional_replays:
        source_identity['additional_host_replays']=[{'source':str(p),'source_contract_sha256':sha(p/'evaluation_contract.json')}
                                                   for p in additional_replays]
    repair_dir = getattr(args, "review_repairs", None)
    if repair_dir:
        source_identity["review_repairs"] = {str(p.resolve()): sha(p) for p in sorted(repair_dir.glob("*.json"))}
    session_contract = {"protocol": VERSION, "source_identity": source_identity,
        "reviewer": {"model": args.reviewer_model, "effort": args.effort, "max_output_tokens": args.max_output_tokens}}
    cli_backend = getattr(args, 'reviewer_backend', 'responses') == 'codex-cli'
    if cli_backend:
        session_contract['reviewer'].update(new_review_backend='codex-cli',
            output_budget_enforcement='PROMPT_TARGET_ONLY', replay_transports='PRESERVED_PER_RECEIPT')
    structured_output = bool(getattr(args,'structured_output',False))
    if structured_output:
        session_contract['reviewer']['structured_output']=True
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "evaluation_contract.json"
    if contract_path.exists() and read(contract_path) != session_contract:
        raise ValueError("evaluation inputs/code/reviewer changed; preserve this run and use a new output directory")
    if not contract_path.exists() and any((output / folder).exists() and any((output / folder).iterdir())
                                           for folder in ("judgments", "requests", "cases", "api")):
        raise ValueError("unversioned existing evaluation artifacts; start v4 in a new output directory")
    write_json(contract_path, session_contract)
    if previous_report.exists():
        previous_cost = read(previous_report).get("evaluation_cost", {})
        if previous_cost.get("api_calls") or previous_cost.get('cli_calls'):
            write_json(output / "cost_history" / (digest(previous_cost) + ".json"), previous_cost)
    work, rows, local_seconds = {}, [], []
    loaded, original_ground_truth, reference_failures = {}, {}, {}
    for slot in slots:
        condition = cases[slot["case_id"]]["condition"]
        row = {"run_id": slot["slot_id"], "model_id": slot["model_id"], "case_id": slot["case_id"],
               "trial": slot["trial_index"], "condition": condition, "collection_status": slot["status"],
               "output_id": slot["output_id"], "protocol": VERSION,
               "status": "NOT_STARTED", "metrics": empty_metrics(condition, "ANSWER_NOT_AVAILABLE")}
        rows.append(row)
        if slot["status"] == "MODEL_NONCOMPLETION":
            row.update(status="MODEL_NONCOMPLETION")
            row["metrics"] = {k: interval(0, 0, applicable=v["applicable"] and k not in {"c_score", "branch_alignment"})
                              for k, v in row["metrics"].items()}
            continue
        if slot["status"] != "COMPLETED":
            continue
        t0 = time.monotonic()
        preparation_stage = 'REFERENCE'
        try:
            if row["case_id"] not in loaded:
                ci, meta, gt0, mat = load_development_case(ROOT, cases[row["case_id"]])
                original_ground_truth[row["case_id"]] = gt0
                gt0, mat = apply_supplement(ci, meta, gt0, mat, supplements.get(row["case_id"]))
                mat["frozen_auxiliary_evidence"] = bank.get(row["case_id"], [])
                loaded[row["case_id"]] = (ci, meta, gt0, mat)
                write_json(output / "reference_packages" / (digest(row["case_id"]) + ".json"), package(ci, meta, gt0, mat))
            case_input, metadata, gt, material = loaded[row["case_id"]]
            preparation_stage = 'ANSWER'
            if "export_source" in slot:
                answer, old = slot["answer"], None
                provenance = {"source": "ANSWER_EXPORT", "answer_source": slot["export_source"],
                              "export_sha256": slot["export_sha256"]}
            elif row["run_id"] in selected:
                src = selected[row["run_id"]]
                if any(src[k] != row[k] for k in ("case_id", "model_id", "trial", "condition")):
                    raise ValueError("donor slot identity mismatch")
                if src["run_sha256"] != slot.get("run_record_sha256"):
                    raise ValueError("donor run digest differs from current collection slot")
                answer, old, provenance = load_source(src, donor, case_input, original_ground_truth[row["case_id"]])
            else:
                answer, old, provenance = load_run_source(slot, Path(slot["collection"]), slot["experiment_id"])
            if not answer.strip():
                row.update(status="MODEL_NONCOMPLETION", answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                           provenance=provenance, reason="EMPTY_COMPLETED_ANSWER")
                row["metrics"] = {k: interval(0, 0, applicable=v["applicable"] and k not in {"c_score", "branch_alignment"})
                                  for k, v in row["metrics"].items()}
                write_json(output / "cases" / (row["output_id"] + ".json"), row)
                local_seconds.append(time.monotonic() - t0)
                continue
            prompt = review_prompt(case_input, metadata, gt, material, answer)
            schema = review_schema(metadata, gt, structured_output=structured_output)
            if structured_output:
                from jsonschema import Draft202012Validator
                Draft202012Validator.check_schema(schema)
            request_identity = {"prompt": prompt, "schema": schema, "model": args.reviewer_model,
                "effort": args.effort, "max_output_tokens": args.max_output_tokens, "protocol": VERSION}
            if structured_output:request_identity['structured_output']=True
            identity = digest(request_identity)
            row.update(answer_sha256=hashlib.sha256(answer.encode()).hexdigest(), provenance=provenance)
            work[row["output_id"]] = (row, answer, metadata, gt, material, prompt, identity)
            # Donors may recover hash-identical original answer text; their
            # scientific judgments never enter a fresh protocol evaluation.
            j = None
            cache = output / "judgments" / (identity + ".json")
            if not cache.exists():
                for candidate in ([replay] if replay else [])+additional_replays:
                    stored = replay_record(candidate, identity, prompt, schema, args.reviewer_model, args.effort, args.max_output_tokens, structured_output)
                    if stored:
                        write_json(output / "requests" / (identity + ".json"), read(candidate / "requests" / (identity + ".json")))
                        write_json(cache, stored)
                        break
            if cache.exists():
                stored = validated_cache(cache, identity, prompt, schema, args.reviewer_model,
                                         effort=args.effort, max_output_tokens=args.max_output_tokens, structured_output=structured_output)
                j = stored["judgment"]
                provenance = {"source": "TRUSTED_REVIEW_CACHE", "request_id": identity, "receipt": stored.get("receipt")}
                if stored.get("host_replay_source"):
                    provenance["host_replay_source"] = stored["host_replay_source"]
                if repair_dir and (repair_dir / (identity + ".json")).exists():
                    from flowintentbench.review_repairs import apply_repair
                    repair_path = repair_dir / (identity + ".json")
                    j = apply_repair(stored["judgment"], prompt, read(repair_path))
                    provenance["review_repair"] = {"path": str(repair_path.resolve()), "sha256": sha(repair_path)}
            result = score(answer, metadata, gt, material, j, condition=condition, case_input=case_input) if j is not None else {
                "status": "NOT_REVIEWED", "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "metrics": empty_metrics(condition, "REVIEW_NOT_YET_AVAILABLE")}
            row.update(result, provenance=provenance)
            work[row["output_id"]] = (row, answer, metadata, gt, material, prompt, identity)
            write_json(output / "cases" / (row["output_id"] + ".json"), row)
        except Exception as exc:
            if preparation_stage == 'REFERENCE':
                reference_failures[row['case_id']] = f'{type(exc).__name__}: {exc}'
            if row["output_id"] in work:
                row.update(status="REVIEW_ERROR", review_error=f"{type(exc).__name__}: {exc}")
            else:
                row.update(status="SOURCE_ERROR", error=f"{type(exc).__name__}: {exc}")
        local_seconds.append(time.monotonic() - t0)
    # Online calls are optional, bounded and resume only unreviewed answers.
    costs = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "usage_missing_calls": 0, "output_limit_exceeded_calls": 0,
             "api_errors": 0, "local_answer_count": len(local_seconds),
             "local_seconds_mean": statistics.fmean(local_seconds) if local_seconds else None,
             "local_seconds_total": sum(local_seconds), "new_review_seconds": [],
             "max_api_calls": args.max_api_calls, "max_output_tokens_per_call": args.max_output_tokens,
             "max_prompt_chars": args.max_prompt_chars, "effort": args.effort}
    if cli_backend:
        costs.update(cli_calls=0,cli_errors=0,reviewer_backend='codex-cli',
                     output_budget_enforcement='PROMPT_TARGET_ONLY')
    if reference_failures:
        # A malformed rubric/reference is a run configuration error. Do not
        # spend calls on a partially prepared comparison and discover the
        # invalid reference only after paying for the other answers.
        costs.update(startup_error='REFERENCE_PREFLIGHT_FAILED',
                     reference_preflight_errors=reference_failures,
                     wall_seconds=time.monotonic()-started)
        report=write_report(output,rows,costs,source_identity)
        print(json.dumps({'status':report['status'],'startup_error':costs['startup_error'],
                          'reference_preflight_errors':reference_failures,'api_calls':0}),flush=True)
        return report
    needed = [r for r in rows if r["output_id"] in work and r["provenance"]["source"] != "TRUSTED_REVIEW_CACHE"]
    needed_pairs = {(r["case_id"], r["trial"]) for r in needed}
    candidates = [r for r in rows if r["output_id"] in work and (r["case_id"], r["trial"]) in needed_pairs]
    partial_pairs = {(r["case_id"], r["trial"]) for r in candidates
                     if r["provenance"]["source"] == "TRUSTED_REVIEW_CACHE"}
    review_rows = choose_paired([r for r in candidates if (r["case_id"], r["trial"]) in partial_pairs], args.max_api_calls)
    review_rows += choose_paired([r for r in candidates if (r["case_id"], r["trial"]) not in partial_pairs],
                                args.max_api_calls - len(review_rows))
    review_rows = [r for r in review_rows if r["provenance"]["source"] != "TRUSTED_REVIEW_CACHE"]
    chosen_ids = {r["output_id"] for r in review_rows}
    review_rows.extend(r for r in needed if r["output_id"] not in chosen_ids)
    unique_requests = {}
    for row in review_rows:
        unique_requests.setdefault(work[row["output_id"]][-1], row)
    review_rows = prioritize_unattempted_reviews(list(unique_requests.values()),
        {row['output_id']: work[row['output_id']][-1] for row in unique_requests.values()},
        [output, replay, *additional_replays])[:args.max_api_calls]
    costs['dispatch_policy'] = 'FEWEST_PRIOR_ATTEMPTS_FIRST_BY_CASE_TRIAL; NO_SLOT_DROPPED'
    write_report(output, rows, costs, source_identity)
    if not args.offline and args.max_api_calls and needed:
        from scripts.evaluate_answered_outcomes import configure_review_credentials
        from scripts.outcome_responses_transport import ResponsesOutcomeReviewer
        from scripts.run_core_case_scoring import _parse_json
        try:
            if cli_backend:
                from scripts.outcome_codex_transport import CodexOutcomeReviewer
                reviewer = CodexOutcomeReviewer()
            else:
                configure_review_credentials(args.auth_path, args.api_config)
                reviewer = ResponsesOutcomeReviewer(args.api_config, concurrency=args.shared_concurrency)
        except Exception as exc:
            costs["startup_error"] = f"{type(exc).__name__}: {exc}"
            costs["wall_seconds"] = time.monotonic() - started
            return write_report(output, rows, costs, source_identity)
        deadline = time.monotonic() + args.max_wall_seconds
        circuit = ProviderCircuit()
        def evaluate(row):
            _, answer, metadata, gt, material, prompt, identity = work[row["output_id"]]
            schema = review_schema(metadata, gt, structured_output=structured_output)
            if row["provenance"]["source"] == "TRUSTED_REVIEW_CACHE":
                return None
            if circuit.is_set():
                row["review_error"] = "PROVIDER_CIRCUIT_OPEN; no request sent"
                return None
            if len(prompt) > args.max_prompt_chars:
                row.update(status="REVIEW_ERROR", review_error="PROMPT_BUDGET_EXCEEDED; full answer retained, no truncation")
                return None
            remaining = min(args.timeout, deadline - time.monotonic())
            if remaining <= 0:
                return None
            api_dir = output / "api" / (identity + "-" + str(time.time_ns()))
            write_json(output / "requests" / (identity + ".json"), {"request_id": identity, "prompt": prompt,
                "schema": schema, "answer_sha256": row["answer_sha256"],
                **({'structured_output':True} if structured_output else {})})
            t0 = time.monotonic()
            try:
                receipt = reviewer(prompt, args.reviewer_model, output / "work", api_dir,
                    timeout_seconds=remaining, output_schema=schema, reasoning_effort=args.effort,
                    max_output_tokens=args.max_output_tokens, **({'structured_output':True} if structured_output else {}))
            except Exception as exc:
                receipt = {"completed": False, "error": f"{type(exc).__name__}: {exc}", "usage": {}}
                write_json(api_dir / "receipt.json", receipt)
            duration = time.monotonic() - t0
            circuit.observe(receipt)
            try:
                if not receipt.get("completed"):
                    raise ValueError(receipt.get("error") or "review incomplete")
                j = _parse_json(receipt["final_text"])
                if structured_output:
                    from jsonschema import Draft202012Validator
                    Draft202012Validator(schema).validate(j)
                result = score(answer, metadata, gt, material, j, condition=row["condition"], case_input=loaded[row["case_id"]][0])
                write_json(output / "judgments" / (identity + ".json"), {"request_id": identity, "judgment": j,
                    "judgment_sha256": digest(j), "receipt": str(api_dir / "receipt.json")})
                row.update(result, provenance={"source": "TRUSTED_CODEX_CLI_REVIEW" if cli_backend else "TRUSTED_HTTP_REVIEW", "request_id": identity,
                    "receipt": str(api_dir / "receipt.json"), "effort": args.effort})
                row.pop("review_error", None)
                for duplicate, *_rest, duplicate_identity in work.values():
                    if duplicate is row or duplicate_identity != identity:
                        continue
                    duplicate.update(deepcopy(result), provenance={"source": "TRUSTED_REVIEW_CACHE",
                        "request_id": identity, "receipt": str(api_dir / "receipt.json"),
                        "same_run_identical_request": True})
                    duplicate.pop("review_error", None)
                    write_json(output / "cases" / (duplicate["output_id"] + ".json"), duplicate)
            except Exception as exc:
                row["review_error"] = f"{type(exc).__name__}: {exc}"
                if row["status"] != "SCORED":
                    row["status"] = "REVIEW_ERROR"
            row["evaluation_cost"] = {"seconds": duration, "usage": receipt.get("usage", {}),
                                      "api_calls": int(not cli_backend), "cli_calls": int(cli_backend)}
            write_json(output / "cases" / (row["output_id"] + ".json"), row)
            return row
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate, row) for row in review_rows]
            for future in as_completed(futures):
                row = future.result()
                if row is None:
                    continue
                costs['cli_calls' if cli_backend else 'api_calls'] += 1
                usage = row["evaluation_cost"]["usage"]
                costs["output_limit_exceeded_calls"] += int(usage.get("output_tokens", 0) > args.max_output_tokens)
                if not all(k in usage for k in ("input_tokens", "output_tokens")):
                    costs["usage_missing_calls"] += 1
                for key in ("input_tokens", "output_tokens"):
                    costs[key] += usage.get(key, 0)
                costs['cli_errors' if cli_backend else 'api_errors'] += int("review_error" in row)
                costs["new_review_seconds"].append(row["evaluation_cost"]["seconds"])
                # Publish progress after each completed review without recording
                # partial invocation costs as separate billable invocations.
                write_json(output / "review_progress.json", {"updated_utc": datetime.now(timezone.utc).isoformat(),
                    "cost": costs, "remaining_selected": len(futures) - sum(f.done() for f in futures)})
                print(json.dumps({"reviewed": costs["api_calls"] + costs.get('cli_calls',0), "max_calls": args.max_api_calls,
                    "case_id": row["case_id"], "model_id": row["model_id"], "review_error": row.get("review_error")}), flush=True)
        costs["provider_circuit_open"] = circuit.is_set()
        costs['provider_circuit_reason'] = circuit.reason
    costs["wall_seconds"] = time.monotonic() - started
    costs["token_totals_complete"] = costs["usage_missing_calls"] == 0
    report = write_report(output, rows, costs, source_identity)
    print(json.dumps({k: report[k] for k in ("requested_slots", "answered_slots", "status", "answered_review_status", "status_counts", "evaluation_cost")}, ensure_ascii=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", type=Path, action="append", help="Repeat for independent collections")
    p.add_argument("--answers", type=Path, action="append", help="JSON/JSONL answer export or directory")
    p.add_argument("--cases", nargs="+")
    p.add_argument("--trials", nargs="+", type=int)
    p.add_argument("--evidence-bank", type=Path, help="Frozen supplementary reference evidence; never computed by evaluation")
    p.add_argument("--reference-package", type=Path, help="Task-bound, reviewed reference additions/reporting policies")
    p.add_argument("--donor", type=Path, help="Optional legacy exact-answer recovery source")
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/model_answer_evaluation_v4")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-identified", action="store_true", help="Exit 3 when review is complete but any applicable metric remains an interval")
    p.add_argument("--reuse-trusted-from", type=Path, action="append", help="Historical option; v4 rejects cross-run review reuse")
    p.add_argument("--replay-from", type=Path, help="Explicit host repair replay of identical current-protocol requests; optional budgeted calls only for missing reviews")
    p.add_argument('--additional-replay-from',type=Path,action='append',default=[],
                   help='Additional exact-request replay sources in declared priority order, e.g. validated pilots; never choose by score')
    p.add_argument("--review-repairs", type=Path, help="Source-bound API repairs of method citations/result-group attribution")
    p.add_argument("--max-api-calls", "--max-review-calls", type=int, default=8)
    p.add_argument('--reviewer-backend',choices=('responses','codex-cli'),default='responses',
                   help='Backend for new reviews only; explicit replay receipts retain their original transport')
    p.add_argument("--max-wall-seconds", type=float, default=300)
    p.add_argument("--max-output-tokens", type=int, default=6000)
    p.add_argument("--max-prompt-chars", type=int, default=60000)
    p.add_argument("--timeout", type=float, default=150)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shared-concurrency", type=int, default=4)
    p.add_argument("--models", nargs="+")
    p.add_argument("--reviewer-model", default="gpt-6-astra")
    p.add_argument("--effort", default="medium", choices=("low", "medium", "high"))
    p.add_argument('--structured-output',action='store_true',help='Use an explicitly bound strict JSON response schema; requires provider support')
    p.add_argument("--api-config", type=Path, default=ROOT / "config/yiapi_direct.toml")
    p.add_argument("--auth-path", type=Path, default=ROOT / "auth_yapi.json")
    return p


def main():
    args = parser().parse_args()
    if args.max_api_calls < 0 or args.workers < 1 or args.max_wall_seconds <= 0 or args.max_output_tokens < 1:
        raise SystemExit("invalid review budget")
    from scripts.portable_fcntl import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".trusted_evaluation.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = run(args)
    if args.require_identified and report["status"] == "COMPLETE_WITH_BOUNDS":
        return 3
    return 0 if report["status"] in {"COMPLETE", "COMPLETE_WITH_BOUNDS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
