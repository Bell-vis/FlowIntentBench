"""Live model-call ownership for FlowIntentBench role agents.

This module owns provider invocation, response parsing, and status/label
normalizers shared by the auditor, analyst, and adjudicator.  It deliberately
does not contain scientific routing or Ground Truth logic.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .agent_profile import effective_agent_config, load_agent_profile
from .model_runner import CANONICAL_PYTHON_TOOL_SPEC, EvaluationTarget, FinalAnswer, ModelStartRequest, ToolBatch
from .providers import OpenAIChatCompletionsAdapter, ProviderRequestTimeoutError, ProviderTransportError, ProviderUnavailableError, ProviderRateLimitError, ProviderQuotaError, ThirdPartyResponsesAdapter
from .provider_status import provider_failure_info
from .proxy_expert import build_firewalled_payload
from .runtime_config import resolve_api_key, resolve_provider_configuration
from .case_repository import load_case_record

LIVE_EXECUTION_MODE = "LIVE_MODEL_CALL"
AUDITOR_PROFILE = "flow-case-auditor-gpt-5.6-sol"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object while retaining the raw model text."""
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate.lstrip())
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_live_role_system_instruction(role: str, instruction: str) -> str:
    """Return the exact system instruction sent by a live role call."""

    return (
        f"You are the FlowIntentBench {role}. Return exactly one JSON object and no "
        "markdown or chain-of-thought. Give concise scientific rationale only. Never "
        "infer information that is not in the visible payload.\n\n"
        f"{instruction}"
    )


def _normalize_auditor_verdict(raw: Any) -> tuple[str, str, bool]:
    value = str(raw or "").strip().upper()
    if value in {"PASS", "REVISE", "UNCERTAIN"}:
        return value, "CANONICAL", False
    compatibility = {"FAIL": "REVISE", "PARTIAL": "UNCERTAIN", "CONDITIONAL": "UNCERTAIN"}
    if value in compatibility:
        return compatibility[value], "LEGACY_MODEL_OUTPUT_COMPATIBILITY", False
    return "UNCERTAIN", "PARSE_FAILURE", True


def _normalize_finding_label(kind: str, raw: Any) -> str:
    value = str(raw or "").strip().upper()
    positive = {"support": "SUPPORTED", "relevance": "RELEVANT", "consistency": "CONSISTENT"}[kind]
    negative = {"support": "UNSUPPORTED", "relevance": "IRRELEVANT", "consistency": "INCONSISTENT"}[kind]
    if value in {positive, "TRUE", "YES", "PASS", "VALID"}:
        return positive
    if value in {negative, "FALSE", "NO", "FAIL", "INVALID", "REJECT"}:
        return negative
    return "UNCERTAIN"


class LiveModelCaller:
    """One independent role call using the repository's configured local route."""

    def __init__(self, root: Path, profile_id: str, *, config_path: str | Path | None = None, api_key: str | None = None, timeout: float = 90.0, max_output_tokens: int = 1200):
        self.root = root
        self.profile = load_agent_profile(profile_id, repository_root=root)
        self.runtime = resolve_provider_configuration(role="model", config_path=config_path, model_id=self.profile.model_id, model_configuration=self.profile.model_configuration)
        token_key = ("max_output_tokens" if self.profile.wire_api.casefold() == "responses"
                     else "max_completion_tokens")
        self.runtime.model_configuration.setdefault(token_key, max_output_tokens)
        if not self.profile.tools:
            self.runtime.model_configuration[token_key] = max_output_tokens
        self.api_key = resolve_api_key(self.runtime, explicit_api_key=api_key)
        self.timeout = timeout

    @property
    def profile_metadata(self) -> dict[str, Any]:
        return {
            "agent_profile_id": self.profile.agent_id,
            "agent_profile_sha256": self.profile.profile_sha256,
            "provider": self.runtime.provider,
            "model_id": self.runtime.model_id,
            "model_family": self.profile.model_family,
            "reasoning_effort": self.runtime.model_configuration.get("reasoning_effort"),
            "wire_api": self.runtime.model_configuration.get("wire_api", self.profile.wire_api),
            "runtime_profile_id": self.profile.runtime_profile_id,
            "runtime_profile_sha256": self.profile.runtime_profile.profile_sha256,
            "effective_agent_config": effective_agent_config(self.profile).to_dict(),
        }

    def call(
        self,
        *,
        role: str,
        visible_payload: Mapping[str, Any],
        instruction: str,
        identity: Mapping[str, Any],
        tool_free: bool = False,
    ) -> dict[str, Any]:
        """Invoke one role with its declared capability boundary.

        ``tool_free`` is used for evidence-conditioned advisory reviews.  It
        is an invocation-level capability switch, not a scientific schema
        field: an empty tool specification makes provider requests omit the
        Python tool entirely.
        """
        payload = build_firewalled_payload(role, visible_payload)
        system = build_live_role_system_instruction(role, instruction)
        profile_has_python = "python" in self.profile.tools
        tool_free = tool_free or not profile_has_python
        python_tool_spec = {} if tool_free else CANONICAL_PYTHON_TOOL_SPEC.to_dict()
        tool_spec_version = "tool-free-v1" if tool_free else CANONICAL_PYTHON_TOOL_SPEC.version
        request = ModelStartRequest(system_prompt=system, system_prompt_version="live-scientific-agent-validation-v1", user_message=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), case_presentation_version="live-role-view-v1", python_tool_spec=python_tool_spec, python_tool_spec_version=tool_spec_version)
        target = EvaluationTarget(self.runtime.provider, self.runtime.model_id, self.runtime.model_configuration)
        wire = str(self.runtime.model_configuration.get("wire_api", self.profile.wire_api)).casefold()
        backend = str(self.runtime.model_configuration.get("execution_backend", "")).casefold()
        if wire == "responses" or backend in {"responses", "third_party_responses"}:
            adapter: Any = ThirdPartyResponsesAdapter(target, base_url=self.runtime.base_url, api_key=self.api_key, formal_mode=False, request_timeout_seconds=self.timeout)
        else:
            adapter = OpenAIChatCompletionsAdapter(target, base_url=self.runtime.base_url, api_key=self.api_key, formal_mode=False, request_timeout_seconds=self.timeout)
        started = time.perf_counter()
        raw_text, usage, error, invocation_status = "", {}, None, "SUCCESS"
        error_exception: BaseException | None = None
        try:
            response = adapter.start_case(request, timeout_seconds=self.timeout)
            if isinstance(response.action, ToolBatch):
                error, invocation_status = "MODEL_REQUESTED_UNSUPPORTED_TOOL_IN_ROLE_CALL", "TOOL_ERROR"
            elif isinstance(response.action, FinalAnswer):
                raw_text, usage = response.action.text, dict(response.raw_usage)
            else:
                error, invocation_status = "MODEL_RETURNED_UNSUPPORTED_ACTION", "TOOL_ERROR"
        except ProviderRequestTimeoutError as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_TIMEOUT"
        except ProviderRateLimitError as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_RATE_LIMIT"
        except ProviderQuotaError as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_QUOTA"
        except (ProviderTransportError, ProviderUnavailableError) as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_ERROR"
        except TimeoutError as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_TIMEOUT"
        except Exception as exc:
            error_exception = exc
            error, invocation_status = f"{type(exc).__name__}: {exc}", "PROVIDER_ERROR"
        finally:
            adapter.close()
        parsed = _json_object(raw_text) if raw_text else None
        if invocation_status == "SUCCESS" and parsed is None:
            invocation_status = "PARSE_FAILURE"
        failure = provider_failure_info(error)
        if failure is not None and error_exception is not None:
            if hasattr(error_exception, "retry_after_seconds"):
                failure["retry_after_seconds"] = getattr(error_exception, "retry_after_seconds")
        return {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, **dict(identity), "role": role, **self.profile_metadata, "tool_free": tool_free, "visible_input_sha256": _digest(payload), "role_profile_sha256": self.profile.profile_sha256, "system_instruction_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(), "timestamp": _now(), "latency_seconds": round(time.perf_counter() - started, 6), "raw_structured_output": raw_text, "parsed_result": parsed, "raw_usage": usage, "error": error, "provider_failure": failure, "invocation_status": invocation_status, "excluded_from_scientific_denominator": invocation_status != "SUCCESS", "status": "PASS" if parsed is not None and error is None else "FAILED"}


def _review_from_model(call: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    invocation_status = str(call.get("invocation_status", "PARSE_FAILURE"))
    review_base = {"case_id": case_id, "execution_mode": LIVE_EXECUTION_MODE, "reviewer_agent_id": call.get("agent_profile_id"), "reviewer_model_family": call.get("model_family"), "proxy_only": True, "live_model_calls": True, "call_status": call.get("status"), "invocation_status": invocation_status, "excluded_from_scientific_denominator": invocation_status != "SUCCESS", "call_error": call.get("error")}
    parsed = call.get("parsed_result")
    if not isinstance(parsed, Mapping):
        return {**review_base, "overall_proxy_opinion": "UNCERTAIN", "raw_overall_proxy_opinion": "", "normalized_overall_proxy_opinion": "UNCERTAIN", "normalization_reason": "PARSE_FAILURE", **{name: {"verdict": "UNCERTAIN", "raw_verdict": "", "normalized_verdict": "UNCERTAIN", "normalization_reason": "PARSE_FAILURE", "failure_codes": [], "scientific_rationale": "No parseable structured review was returned.", "evidence_ids": []} for name in ("q1", "q2", "q3")}}

    def q(name: str) -> dict[str, Any]:
        item = parsed.get(name, {})
        if not isinstance(item, Mapping):
            item = {}
        raw_verdict = item.get("verdict", "")
        normalized, reason, parse_failure = _normalize_auditor_verdict(raw_verdict)
        codes = item.get("failure_codes", item.get("failure_code", []))
        if isinstance(codes, str):
            codes = [codes]
        if parse_failure:
            codes = [*codes, "PARSE_FAILURE"]
        return {"verdict": normalized, "raw_verdict": str(raw_verdict), "normalized_verdict": normalized, "normalization_reason": reason, "failure_codes": sorted({str(x) for x in codes or ()}), "scientific_rationale": str(item.get("scientific_rationale", item.get("rationale", ""))), "evidence_ids": [str(x) for x in item.get("evidence_ids", item.get("supporting_evidence_ids", [])) or ()]}

    qs = {name: q(name) for name in ("q1", "q2", "q3")}
    raw_overall = parsed.get("overall_proxy_opinion", "")
    overall, overall_reason, overall_parse_failure = _normalize_auditor_verdict(raw_overall)
    if overall_parse_failure:
        overall = "REVISE" if any(item["verdict"] == "REVISE" for item in qs.values()) else "UNCERTAIN" if any(item["verdict"] == "UNCERTAIN" for item in qs.values()) else "PASS"
        overall_reason = "DERIVED_FROM_Q_VERDICTS"
    if overall_parse_failure or any("PARSE_FAILURE" in item.get("failure_codes", ()) for item in qs.values()):
        review_base["invocation_status"], review_base["excluded_from_scientific_denominator"] = "PARSE_FAILURE", True
    return {**review_base, **qs, "overall_proxy_opinion": overall, "raw_overall_proxy_opinion": str(raw_overall), "normalized_overall_proxy_opinion": overall, "normalization_reason": overall_reason, "scientific_questions_or_risks": sorted({code for item in qs.values() for code in item["failure_codes"]})}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_live_case_auditor(root: Path, out: Path, *, config_path: str | Path | None = None, api_key: str | None = None, timeout: float = 90.0) -> dict[str, Any]:
    """Run live original/mutation auditor calls and write role artifacts."""
    from .grounding import build_grounding_packets
    from .scientific_expert_constraint_validation import _build_auditor_input, _mutations

    caller = LiveModelCaller(root, AUDITOR_PROFILE, config_path=config_path, api_key=api_key, timeout=timeout)
    cases = [load_case_record(root, cid) for cid in ("office_speed_zones_o2_f1", "office_speed_zones_o1_f2", "kitchen_turbulence_activity_o2_f1")]
    packets = {packet.family_id: packet for packet in build_grounding_packets(root)}
    originals: list[dict[str, Any]] = []
    mutated: list[dict[str, Any]] = []
    firewall_rows: list[dict[str, Any]] = []
    tasks: list[tuple[int, Mapping[str, Any], Mapping[str, Any], str, Mapping[str, Any] | None]] = []
    index = 0
    for case in cases:
        packet = packets.get(str(case["metadata"].get("case_family_id")))
        visible = _build_auditor_input(root, case, packet)
        forbidden = {"mutation_id", "mutation_type", "expected", "model_answer", "model_score", "accepted_o", "reference_finding", "ground_truth"}
        hits = [key for key in visible if str(key).casefold() in forbidden]
        firewall_rows.append({"case_id": case["case_id"], "visible_keys": sorted(visible), "forbidden_keys": hits, "passed": not hits})
        tasks.append((index, case, visible, "original", None)); index += 1
        for mutation in _mutations(case):
            mutated_visible = dict(visible); mutated_visible["question"] = mutation["question"]
            hits = [key for key in mutated_visible if str(key).casefold() in forbidden]
            firewall_rows.append({"case_id": case["case_id"], "mutation_id": mutation["mutation_id"], "visible_keys": sorted(mutated_visible), "forbidden_keys": hits, "passed": not hits})
            tasks.append((index, case, mutated_visible, "mutation", mutation)); index += 1
    instruction = "Assess Q1 scientific target/capability, Q2 responsibility and unresolved dimensions, and Q3 actionability/representability. Each q1/q2/q3 verdict MUST be exactly one of: PASS, REVISE, UNCERTAIN. Return q1/q2/q3 objects with verdict, failure_codes, scientific_rationale, evidence_ids, plus overall_proxy_opinion using the same three canonical values."

    def execute(task: tuple[int, Mapping[str, Any], Mapping[str, Any], str, Mapping[str, Any] | None]) -> tuple[int, dict[str, Any], Mapping[str, Any] | None]:
        task_index, case, visible, kind, mutation = task
        identity: dict[str, Any] = {"case_id": case["case_id"], "input_kind": kind}
        if mutation is not None:
            identity["mutation_id"] = mutation["mutation_id"]
        call = caller.call(role="auditor", visible_payload=visible, instruction=instruction, identity=identity)
        review = _review_from_model(call, case["case_id"]); review["call"] = call
        return task_index, review, mutation

    with ThreadPoolExecutor(max_workers=min(8, len(tasks) or 1)) as pool:
        completed = list(pool.map(execute, tasks))
    for _index, review, mutation in sorted(completed, key=lambda item: item[0]):
        if mutation is None:
            originals.append(review)
        else:
            detected = review.get("overall_proxy_opinion") == "REVISE" or bool(review.get("scientific_questions_or_risks", []))
            mutated.append({"case_id": review["case_id"], "mutation_id": mutation["mutation_id"], "mutation_type": mutation["mutation_type"], "expected_failure": mutation["expected"], "expected_failure_detected": detected, "review": review, "call": review.get("call", {})})
    successful_originals = [item for item in originals if item.get("invocation_status") == "SUCCESS"]
    successful_mutations = [item for item in mutated if item.get("review", {}).get("invocation_status") == "SUCCESS"]
    infra_originals = [item for item in originals if item.get("invocation_status") in {"PROVIDER_TIMEOUT", "PROVIDER_ERROR", "TOOL_ERROR"}]
    infra_mutations = [item for item in mutated if item.get("review", {}).get("invocation_status") in {"PROVIDER_TIMEOUT", "PROVIDER_ERROR", "TOOL_ERROR"}]
    detection = sum(bool(item["expected_failure_detected"]) for item in successful_mutations)
    summary = {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "attempted_original_reviews": len(originals), "successful_original_reviews": len(successful_originals), "infra_invalid_original_reviews": len(infra_originals), "attempted_mutation_reviews": len(mutated), "successful_mutation_reviews": len(successful_mutations), "infra_invalid_mutation_reviews": len(infra_mutations), "parse_failure_original_reviews": sum(item.get("invocation_status") == "PARSE_FAILURE" for item in originals), "parse_failure_mutation_reviews": sum(item.get("review", {}).get("invocation_status") == "PARSE_FAILURE" for item in mutated), "detected_mutations": detection, "scientific_false_negative_count": len(successful_mutations) - detection, "mutation_detection_count": detection, "mutation_false_negative_count": len(successful_mutations) - detection, "live_original_cases_reviewed": len(successful_originals), "live_mutated_cases_reviewed": len(successful_mutations), "original_pass_count": sum(item["overall_proxy_opinion"] == "PASS" for item in successful_originals), "original_revise_count": sum(item["overall_proxy_opinion"] == "REVISE" for item in successful_originals), "original_uncertain_count": sum(item["overall_proxy_opinion"] == "UNCERTAIN" for item in successful_originals), "historical_positive_control_status": "NOT_APPLICABLE", "original_false_reject_count": "NOT_EVALUABLE", "excluded_from_scientific_denominator": len(originals) - len(successful_originals) + len(mutated) - len(successful_mutations)}
    _write_json(out / "manifest.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "role": "auditor", "profile": caller.profile_metadata, "case_ids": [case["case_id"] for case in cases], "mutation_labels_hidden_from_model": True})
    _write_json(out / "original_reviews.json", originals)
    _write_json(out / "mutated_reviews.json", mutated)
    _write_json(out / "sensitivity_summary.json", summary)
    _write_json(out / "firewall_audit.json", {"status": "PASS" if all(row["passed"] for row in firewall_rows) else "FAIL", "execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "rows": firewall_rows})
    return {"summary": summary, "originals": originals, "mutated": mutated, "firewall": firewall_rows, "status": "RUN" if successful_originals or successful_mutations else "FAILED"}


__all__ = ["LIVE_EXECUTION_MODE", "LiveModelCaller", "build_live_role_system_instruction", "run_live_case_auditor", "_digest", "_json_object", "_now", "_normalize_auditor_verdict", "_normalize_finding_label", "_review_from_model"]
