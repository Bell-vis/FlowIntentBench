#!/usr/bin/env python3
"""Budgeted evidence-citation/group repair, preserving original API judgments."""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.rubric_scoring import digest
from flowintentbench.review_repairs import needs_repair, repair_prompt, repair_schema, apply_repair, citation_targets, citation_description, finding_citation_targets
from scripts.evaluate_model_answers import validated_cache, read
from scripts.run_core_case_scoring import _parse_json


def citation_preflight(judgment, mode='CITATIONS_ONLY_V2'):
    """Reject layouts the narrow citation prompt cannot faithfully describe.

    V2 accepts an explicit method description in `choice` as well as rationale.
    An empty target must not consume a call or imply absent answer evidence.
    Historical V1 prompts remain unchanged for receipt verification.
    """
    targets = citation_targets(judgment)
    if not targets:
        return 'no uncited explicit dimensions'
    for gi, di in targets.values():
        dimension = judgment['result_groups'][gi]['dimensions'][di]
        description = dimension.get('rationale')
        if mode == 'CITATIONS_ONLY_V2':
            description = citation_description(dimension)
        if not isinstance(description, str) or not description.strip():
            return 'missing method rationale in citation layout; use GROUPS repair'
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-api-calls", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shared-concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--max-output-tokens", type=int, default=4000)
    p.add_argument("--request-ids", nargs="+", help="Diagnosed requests requiring an explicit focused audit")
    p.add_argument("--extraction", action="store_true", help="Also audit claim extraction and explicit method attribution")
    p.add_argument("--method-scope", action="store_true", help="Audit overclaimed method equivalence and mixed-method result groups")
    p.add_argument('--citations-only',action='store_true',help='Locate method citations without changing any decisions or findings')
    p.add_argument('--finding-citations',action='store_true',help='Locate missing finding/unit citations; preserve all claims and decisions')
    p.add_argument('--structured-output', action='store_true',
                   help='Enforce the citation-only JSON schema through the Responses API')
    p.add_argument("--citation-donors", type=Path, help="Validated previous group repairs; fill only omitted citations for identical semantic decisions")
    p.add_argument('--prior-repairs', type=Path, help='Apply a source-bound prior repair before this focused audit; preserve the verified chain')
    p.add_argument("--api-config", type=Path, default=ROOT / "config/yiapi_direct.toml")
    p.add_argument("--auth-path", type=Path, default=ROOT / "auth_yapi.json")
    args = p.parse_args()
    if args.max_api_calls < 0 or args.workers < 1 or args.shared_concurrency < 1 or args.timeout <= 0 or args.max_output_tokens < 1:
        p.error("invalid repair budget")
    args.output = args.output.resolve()
    contract = read(args.source / "evaluation_contract.json")
    reviewer_info = contract["reviewer"]
    jobs = []
    if sum((args.extraction,args.method_scope,args.citations_only,args.finding_citations))>1:
        p.error('extraction, method-scope, citations-only and finding-citations are mutually exclusive')
    mode = ('FINDING_CITATIONS' if args.finding_citations else 'CITATIONS_ONLY_V2' if args.citations_only else
            "EXTRACTION" if args.extraction else "METHOD_SCOPE_BLIND" if args.method_scope else "GROUPS")
    if args.structured_output and mode not in {'FINDING_CITATIONS', 'CITATIONS_ONLY_V2'}:
        p.error('--structured-output currently requires --citations-only or --finding-citations')
    for path in sorted((args.source / "judgments").glob("*.json")):
        if args.request_ids and path.stem not in args.request_ids:
            continue
        request = read(args.source / "requests" / path.name)
        stored = validated_cache(path, path.stem, request["prompt"], request["schema"], reviewer_info["model"],
                                 effort=reviewer_info["effort"], max_output_tokens=reviewer_info["max_output_tokens"])
        original_judgment = stored['judgment']
        prior_binding = None
        prior = args.prior_repairs / path.name if args.prior_repairs else None
        if prior and prior.exists():
            stored = {**stored, 'judgment':apply_repair(original_judgment, request['prompt'], read(prior))}
            prior_binding = {'path':str(prior.resolve()), 'sha256':hashlib.sha256(prior.read_bytes()).hexdigest()}
        if args.citations_only:
            citation_problem = citation_preflight(stored['judgment'])
            if citation_problem:
                if args.request_ids:
                    raise ValueError('selected citation repair: '+citation_problem+': '+path.stem)
                continue
        if args.finding_citations and not finding_citation_targets(stored['judgment'],request['prompt']):
            if args.request_ids:
                raise ValueError('selected finding citation repair has no unbound source/unit targets: '+path.stem)
            continue
        reasons = (["EXPLICIT_FINDING_CITATION_AUDIT"] if args.finding_citations else ["EXPLICIT_CITATION_AUDIT"] if args.citations_only else
                   ["EXPLICIT_METHOD_SCOPE_AUDIT"] if args.method_scope else ["EXPLICIT_EXTRACTION_AUDIT"] if args.extraction else
                   ["EXPLICIT_GROUP_AUDIT"] if args.request_ids else needs_repair(stored["judgment"]))
        if reasons:
            dest = args.output / (path.stem + ".json")
            if not dest.exists():
                for receipt_path in sorted((args.output / "api").glob(path.stem + "-*/receipt.json")):
                    receipt = read(receipt_path)
                    if not receipt.get("completed"):
                        continue
                    record = {"base_judgment_sha256": digest(stored["judgment"]), "base_prompt_sha256": digest(request["prompt"]),
                        "receipt": str(receipt_path), "model": reviewer_info["model"], "effort": "medium",
                        "max_output_tokens": args.max_output_tokens, "mode": mode, "repair": _parse_json(receipt["final_text"])}
                    if prior_binding:
                        record['prior_repair'] = prior_binding
                    wire = read(receipt_path.with_name('request.json'))
                    if wire.get('text', {}).get('format', {}).get('type') == 'json_schema':
                        record['structured_output'] = True
                    try:
                        apply_repair(original_judgment, request["prompt"], record)
                    except (ValueError, KeyError, TypeError):
                        continue
                    write_json(dest, record)
                    break
            if dest.exists():
                apply_repair(original_judgment, request["prompt"], read(dest))
            else:
                jobs.append((path.stem, stored["judgment"], request["prompt"], reasons, original_judgment, prior_binding))
    from scripts.evaluate_answered_outcomes import configure_review_credentials
    from scripts.outcome_responses_transport import ResponsesOutcomeReviewer
    configure_review_credentials(args.auth_path, args.api_config)
    reviewer = ResponsesOutcomeReviewer(args.api_config, concurrency=args.shared_concurrency)
    def run(job):
        identity, judgment, original_prompt, reasons, original_judgment, prior_binding = job
        api_dir = args.output / "api" / (identity + "-" + str(time.time_ns()))
        t0 = time.monotonic()
        receipt = reviewer(repair_prompt(original_prompt, judgment, mode), reviewer_info["model"], args.output / "work", api_dir,
            timeout_seconds=args.timeout, output_schema=repair_schema(mode), reasoning_effort="medium",
            max_output_tokens=args.max_output_tokens, structured_output=args.structured_output)
        result = {"identity": identity, "reasons": reasons, "seconds": time.monotonic() - t0,
                  "usage": receipt.get("usage", {}), "completed": False}
        try:
            if not receipt.get("completed"):
                raise ValueError(receipt.get("error", "repair incomplete"))
            record = {"base_judgment_sha256": digest(judgment), "base_prompt_sha256": digest(original_prompt),
                "receipt": str(api_dir / "receipt.json"), "model": reviewer_info["model"], "effort": "medium",
                "max_output_tokens": args.max_output_tokens, "mode": mode, "repair": _parse_json(receipt["final_text"])}
            if prior_binding:
                record['prior_repair'] = prior_binding
            if args.structured_output:
                record['structured_output'] = True
            donor = args.citation_donors / (identity + ".json") if args.citation_donors else None
            if mode == "EXTRACTION" and donor and donor.exists():
                record["citation_donor"] = {"path": str(donor.resolve()), "sha256": hashlib.sha256(donor.read_bytes()).hexdigest()}
            apply_repair(original_judgment, original_prompt, record)
            write_json(args.output / (identity + ".json"), record)
            result["completed"] = True
        except Exception as exc:
            result["error"] = str(exc)
        write_json(api_dir / "repair_result.json", result)
        print(json.dumps({**result, "usage": {k: result["usage"].get(k) for k in ("input_tokens", "output_tokens")}}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, jobs[:args.max_api_calls]))
    write_json(args.output / "reports" / "repair_report.json", {"requested": len(jobs), "attempted": len(results),
        "rows": results, "remaining": max(0, len(jobs)-len(results))})
    return 0 if all(r["completed"] for r in results) and len(jobs) <= len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
