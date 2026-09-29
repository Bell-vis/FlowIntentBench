"""Production pending-evaluation continuations.

This module implements the six already-frozen continuation owners without
adding a workflow layer.  Deterministic owners only transform explicit typed
requests.  Scientific owners invoke a separately configured, tool-free model
through the repository's local compatible provider route and receive only a
blinded scientific view; they never receive Ground Truth or reference-space
membership.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .evaluator import (
    AdjudicationStatus,
    EvaluationAdjudications,
    NovelOperationalizationAdjudication,
    SemanticMatchRequest,
    SemanticMatchResolution,
    SemanticMatchResult,
    UnitFrameResolution,
    continuation_request_id,
)
from .evaluator_backend import EvaluatorBackendError
from .ground_truth import OperationalizationBundle
from .live_agents import LiveModelCaller
from .proxy_expert import AdjudicationCandidateView, build_adjudication_candidate_view
from .provider_retry import ProviderRetryPolicy, failure_category
from .provider_status import provider_failure_info


SCHEMA_VERSION = "production-continuation-v1"
DEFAULT_PROFILE = "gpt-5.6-terra-xhigh-chat"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _config_path() -> str | None:
    return os.environ.get("FLOWINTENTBENCH_SERVER_CONFIG")


def _profile_id() -> str:
    return os.environ.get(
        "FLOWINTENTBENCH_CONTINUATION_AGENT", DEFAULT_PROFILE
    )


def _declared_model_identity() -> dict[str, Any]:
    """Resolve the exact model/profile identity without making a model call."""

    from .agent_profile import load_agent_profile
    from .runtime_config import resolve_provider_configuration

    root = Path(__file__).resolve().parents[1]
    profile = load_agent_profile(_profile_id(), repository_root=root)
    runtime = resolve_provider_configuration(
        role="model",
        config_path=_config_path(),
        provider=profile.provider,
        model_id=profile.model_id,
        model_configuration=profile.model_configuration,
    )
    return {
        "provider": runtime.provider,
        "model": runtime.model_id,
        "model_configuration": {
            **dict(runtime.model_configuration),
            "agent_profile": profile.agent_id,
            "agent_profile_sha256": profile.profile_sha256,
            "tool_free": True,
        },
    }


def _caller() -> LiveModelCaller:
    configured_timeout = float(
        os.environ.get("FLOWINTENTBENCH_CONTINUATION_TIMEOUT", "300")
    )
    deadline_value = os.environ.get("FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC")
    timeout = configured_timeout
    if deadline_value:
        try:
            remaining = float(deadline_value) - time.monotonic()
        except ValueError as exc:
            raise ValueError(
                "FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC must be numeric"
            ) from exc
        if remaining <= 0:
            raise EvaluatorBackendError(
                "continuation case deadline timeout before provider request"
            )
        timeout = min(timeout, remaining)
    return LiveModelCaller(
        Path(__file__).resolve().parents[1],
        _profile_id(),
        config_path=_config_path(),
        timeout=timeout,
        max_output_tokens=int(
            os.environ.get("FLOWINTENTBENCH_CONTINUATION_MAX_OUTPUT_TOKENS", "8192")
        ),
    )


def _call_json(
    *, role: str, payload: Mapping[str, Any], instruction: str
) -> Mapping[str, Any] | None:
    attempts = int(os.environ.get("FLOWINTENTBENCH_CONTINUATION_ATTEMPTS", "4"))
    if attempts < 1:
        raise ValueError("FLOWINTENTBENCH_CONTINUATION_ATTEMPTS must be positive")
    policy = ProviderRetryPolicy(
        transient_backoff_seconds=float(
            os.environ.get("FLOWINTENTBENCH_CONTINUATION_RETRY_BACKOFF", "2")
        ),
        rate_limit_backoff_seconds=float(
            os.environ.get("FLOWINTENTBENCH_CONTINUATION_RATE_LIMIT_BACKOFF", "15")
        ),
        rate_limit_max_backoff_seconds=float(
            os.environ.get("FLOWINTENTBENCH_CONTINUATION_RATE_LIMIT_MAX_BACKOFF", "60")
        ),
        jitter_seconds=float(
            os.environ.get("FLOWINTENTBENCH_CONTINUATION_RETRY_JITTER", "0")
        ),
    )
    last_error: EvaluatorBackendError | None = None
    for attempt in range(attempts):
        deadline_value = os.environ.get("FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC")
        if deadline_value:
            try:
                if float(deadline_value) <= time.monotonic():
                    raise EvaluatorBackendError(
                        "continuation case deadline timeout before provider retry"
                    )
            except ValueError as exc:
                raise ValueError(
                    "FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC must be numeric"
                ) from exc
        call = _caller().call(
            role=role,
            visible_payload=payload,
            instruction=instruction,
            identity={"continuation_schema": SCHEMA_VERSION},
            tool_free=True,
        )
        parsed = call.get("parsed_result")
        if call.get("invocation_status") == "SUCCESS" and isinstance(parsed, Mapping):
            return dict(parsed)
        # A model response that cannot be parsed is a bounded format failure,
        # not scientific uncertainty.  Transport errors are likewise raised
        # after retries so evaluate_saved_runs persists INFRASTRUCTURE_INVALID
        # instead of silently retaining a scientific PENDING record.
        status = str(call.get("invocation_status") or "PARSE_FAILURE")
        message = str(call.get("error") or status)
        last_error = EvaluatorBackendError(
            f"continuation {role} {status} after attempt {attempt + 1}: {message}"
        )
        provider_failure = call.get("provider_failure")
        if isinstance(provider_failure, Mapping):
            setattr(last_error, "provider_failure", dict(provider_failure))
            if provider_failure.get("http_status") is not None:
                setattr(last_error, "http_status", provider_failure.get("http_status"))
            if provider_failure.get("retry_after_seconds") is not None:
                setattr(
                    last_error,
                    "retry_after_seconds",
                    provider_failure.get("retry_after_seconds"),
                )
        if status == "PARSE_FAILURE" and attempt + 1 >= attempts:
            break
        if provider_failure_info(message, fallback_status=None) and (
            isinstance(provider_failure, Mapping)
            and provider_failure.get("category") == "QUOTA"
        ):
            # Quota/billing errors are not recoverable by sleeping.
            break
        if attempt + 1 < attempts:
            delay = policy.delay(attempt, last_error)
            deadline_value = os.environ.get("FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC")
            if deadline_value:
                try:
                    remaining = float(deadline_value) - time.monotonic()
                except ValueError as exc:
                    raise ValueError(
                        "FLOWINTENTBENCH_CASE_DEADLINE_MONOTONIC must be numeric"
                    ) from exc
                if remaining <= 0:
                    raise EvaluatorBackendError(
                        "continuation case deadline timeout during provider retry"
                    ) from last_error
                time.sleep(min(delay, remaining))
            else:
                time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise EvaluatorBackendError(f"continuation {role} returned no usable result")


def _claims(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    value = packet.get("model_claims")
    if not isinstance(value, Mapping):
        raise ValueError("pending continuation packet lacks model_claims")
    return value


def _candidate_operationalization(
    packet: Mapping[str, Any], *, status: AdjudicationStatus
) -> EvaluationAdjudications:
    raw = _claims(packet).get("candidate_operationalization")
    if not isinstance(raw, Mapping):
        raise ValueError("novel O continuation lacks candidate_operationalization")
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("novel O continuation lacks explicit O decisions")
    normalized = []
    for item in decisions:
        if not isinstance(item, Mapping):
            raise ValueError("novel O decision must be an object")
        statement = item.get("statement", item.get("normalized_statement"))
        if not isinstance(statement, str) or not statement.strip():
            # Completion cannot invent a scientific choice that the evaluated
            # response did not state explicitly.
            return EvaluationAdjudications()
        normalized.append(
            {"dimension": item.get("dimension"), "statement": statement.strip()}
        )
    operationalization = OperationalizationBundle.model_validate(
        {
            "operationalization_id": "runtime_o_" + _digest(normalized)[:20],
            "decisions": normalized,
            "evidence_ids": [],
        }
    )
    return EvaluationAdjudications(
        novel_operationalization=NovelOperationalizationAdjudication(
            status=status,
            operationalization=operationalization,
        )
    )


def novel_o_completion(packet: Mapping[str, Any]) -> EvaluationAdjudications | None:
    """Complete only structure that was explicit in the extracted response."""

    result = _candidate_operationalization(
        packet, status=AdjudicationStatus.UNRESOLVED
    )
    return result if result.novel_operationalization is not None else None


def novel_o_materializer(_packet: Mapping[str, Any]) -> None:
    """The case-local trusted materializer is owned by ``CaseEvaluator``.

    Reaching this owner means the exact O is outside that deterministic route;
    returning ``None`` correctly preserves a scientific/data execution
    pending instead of fabricating G(O).
    """

    return None


def blinded_scientific_adjudicator(
    packet: Mapping[str, Any],
) -> EvaluationAdjudications | None:
    claims = _claims(packet)
    candidate = claims.get("candidate_operationalization")
    materialized = claims.get("G_of_O")
    materialization_id = claims.get("materialization_id")
    materialization_digest = claims.get("G_of_O_sha256")
    if not (
        isinstance(candidate, Mapping)
        and isinstance(materialized, Mapping)
        and isinstance(materialization_id, str)
        and isinstance(materialization_digest, str)
    ):
        return None
    view = AdjudicationCandidateView(
        question=str(packet.get("scientific_question") or ""),
        context=packet.get("case_context"),
        scientific_target=packet.get("finding_goal"),
        candidate_scientific_O=candidate,
        materialized_G_of_O=materialized,
        structured_scientific_evidence={
            "execution_status": claims.get("execution_status"),
            "materialization_id": materialization_id,
            "G_of_O_sha256": materialization_digest,
        },
    )
    visible = build_adjudication_candidate_view(view)
    parsed = _call_json(
        role="adjudicator",
        payload=visible,
        instruction=(
            "Judge only whether the candidate Operationalization is scientifically "
            "coherent for the visible question/context and whether the actual "
            "materialized G(O) supports executing it. You do not know reference "
            "membership. Return {\"scientific_validity\": "
            "\"SCIENTIFICALLY_VALID\"|\"SCIENTIFICALLY_INVALID\"|\"UNCERTAIN\", "
            "\"rationale\": \"...\"}."
        ),
    )
    if parsed is None:
        return None
    label = str(parsed.get("scientific_validity", "UNCERTAIN")).upper()
    if label == "SCIENTIFICALLY_VALID":
        status = AdjudicationStatus.ACCEPTED
    elif label == "SCIENTIFICALLY_INVALID":
        status = AdjudicationStatus.REJECTED
    else:
        return None
    operationalization = OperationalizationBundle.model_validate(candidate)
    return EvaluationAdjudications(
        novel_operationalization=NovelOperationalizationAdjudication(
            status=status,
            operationalization=operationalization,
            adjudicated_materialization_id=materialization_id,
            adjudicated_g_of_o_sha256=materialization_digest,
        )
    )


def blinded_finding_adjudicator(
    packet: Mapping[str, Any],
) -> EvaluationAdjudications | None:
    claims = _claims(packet)
    request = claims.get("continuation_request")
    branch_id = claims.get("branch_id")
    if not isinstance(request, Mapping) or not isinstance(branch_id, str):
        return None
    prediction_id = request.get("prediction_id")
    finding = request.get("finding")
    if not isinstance(prediction_id, str) or not isinstance(finding, Mapping):
        return None
    # Multiple reported findings share the same independently executed G(O).
    # Judge each separately in one bounded request; never pool across branches.
    candidates = [{**dict(finding), "finding_id": prediction_id}]
    seen = {prediction_id}
    for item in claims.get("candidate_findings", []):
        if not isinstance(item, Mapping):
            continue
        identifier = item.get("prediction_id")
        if not isinstance(identifier, str) or identifier in seen:
            continue
        candidates.append({"finding_id": identifier, **{k: item[k] for k in ("statement", "value", "unit") if k in item}})
        seen.add(identifier)
        if len(candidates) == 32:
            break
    view = AdjudicationCandidateView(
        question=str(packet.get("scientific_question") or ""),
        context=packet.get("case_context"),
        scientific_target=packet.get("finding_goal"),
        candidate_scientific_O=claims.get("candidate_operationalization"),
        materialized_G_of_O=claims.get("materialized_G_of_O"),
        fixed_evidence_context={
            "branch_id": branch_id,
            "allowed_finding_roles": claims.get("allowed_finding_roles", []),
        },
        candidate_F=candidates,
    )
    parsed = _call_json(
        role="adjudicator",
        payload=build_adjudication_candidate_view(view),
        instruction=(
            'Judge each reported Finding independently against this one executed G(O), '
            'without knowing Ground Truth or reference membership. Return {"findings": ['
            '{"finding_id": "exact input ID", "support": "SUPPORTED|UNSUPPORTED|UNCERTAIN", '
            '"relevance": "RELEVANT|IRRELEVANT|UNCERTAIN", '
            '"consistency": "CONSISTENT|INCONSISTENT|UNCERTAIN", '
            '"finding_role": "allowed role or null", "rationale": "brief evidence-based reason"}]}. '
            'Return exactly one item per input Finding. Interpret the vertical bars as choices, '
            'not literal output values. Scientific credit requires all three positive judgments. '
            'Use the model-authored Operationalization to interpret references such as selected structure; '
            'it is context, not reference membership or a requirement to match a reference method. '
            'Do not transfer support between Findings. Use the supplied bounding box and region '
            'rankings when judging extents or comparisons; preserve uncertainty when evidence is absent. '
            'Explicit numerical evidence that the claimed object is absent or its requested quantity is '
            'undefined refutes claims assigning that quantity a value or stability. This is negative '
            'evidence, distinct from a missing or unexecuted test.'
        ),
    )
    if parsed is None:
        return None
    verdicts = parsed.get("findings")
    if verdicts is None:
        # Read legacy single-Finding responses and deterministic test fixtures.
        verdicts = [{"finding_id": prediction_id, **dict(parsed)}]
    if not isinstance(verdicts, list):
        raise ValueError("Finding adjudication must return a findings list")
    statuses, roles, returned = {}, {}, set()
    allowed_roles = {str(item) for item in claims.get("allowed_finding_roles", []) if str(item).strip()}
    for verdict in verdicts:
        if not isinstance(verdict, Mapping):
            raise ValueError("Finding verdict must be an object")
        identifier = verdict.get("finding_id")
        if not isinstance(identifier, str) or identifier not in seen or identifier in returned:
            raise ValueError("Finding verdict has an unknown or duplicate ID")
        returned.add(identifier)
        values = tuple(str(verdict.get(k, "")).upper() for k in ("support", "relevance", "consistency"))
        accepted = values == ("SUPPORTED", "RELEVANT", "CONSISTENT")
        rejected = any(v in {"UNSUPPORTED", "IRRELEVANT", "INCONSISTENT"} for v in values)
        role = verdict.get("finding_role")
        if accepted and allowed_roles and str(role or "").strip() not in allowed_roles:
            continue
        if not accepted and not rejected:
            continue
        statuses[(branch_id, identifier)] = AdjudicationStatus.ACCEPTED if accepted else AdjudicationStatus.REJECTED
        if accepted and isinstance(role, str) and role.strip():
            roles[(branch_id, identifier)] = role.strip()
    return EvaluationAdjudications(novel_findings=statuses, novel_finding_roles=roles) if statuses else None


def semantic_match_resolver(
    packet: Mapping[str, Any],
) -> SemanticMatchResolution | None:
    raw = _claims(packet).get("semantic_match_request")
    if not isinstance(raw, Mapping):
        return None
    request = SemanticMatchRequest.from_dict(raw)
    parsed = _call_json(
        role="adjudicator",
        payload=AdjudicationCandidateView(
            question=request.scientific_question,
            context=request.case_context,
            scientific_target=request.finding_goal,
            candidate_scientific_O=_claims(packet).get("candidate_operationalization"),
            fixed_evidence_context={"other_model_findings_for_referent_context": _claims(packet).get("candidate_findings", [])},
            candidate_F=[
                {
                    "finding_id": request.predicted_id,
                    "predicted_statement": request.predicted_statement,
                    "reference_statement": request.reference_statement,
                    "predicted_value": request.predicted_value, "reference_value": request.reference_value,
                    "predicted_unit": request.predicted_unit, "reference_unit": request.reference_unit,
                    "purpose": request.purpose.value,
                    "verification_mode": request.verification_mode.value,
                }
            ],
        ).to_dict(),
        instruction=(
            "Judge whether the two visible statements have the same consequential "
            "scientific meaning. For deterministic value-verification modes, ignore "
            "numeric disagreement because it is checked later. Compare the actual properties and "
            "relationships asserted, not just a shared category. A field identifier and a statistic "
            "about that field are different properties; a qualitative extent or boundary identity "
            "does not by itself assert a numerical centroid. These distinctions are NO_MATCH, "
            "not numerical disagreements. Use the model context only to resolve referents; do not "
            "merge separate Findings or borrow another Finding's numerical value. Reserve UNCERTAIN "
            "for unresolved semantic ambiguity. Return "
            "{\"result\": \"MATCH\"|\"NO_MATCH\"|\"UNCERTAIN\"}."
        ),
    )
    if parsed is None:
        return None
    result = str(parsed.get("result", "UNCERTAIN")).upper()
    if result not in {"MATCH", "NO_MATCH"}:
        return None
    return SemanticMatchResolution(
        continuation_request_id("semantic_match", request.to_dict()),
        request,
        SemanticMatchResult(result),
    )


def unit_frame_resolver(packet: Mapping[str, Any]) -> UnitFrameResolution:
    raw = _claims(packet).get("unit_frame_request")
    if not isinstance(raw, Mapping):
        raise ValueError("unit/frame continuation lacks a typed request")
    request = dict(raw)
    predicted_unit = request.get("predicted_unit")
    reference_unit = request.get("reference_unit")
    authority = request.get("unit_frame_authority")
    normalized = " ".join(str(predicted_unit or "").casefold().split())
    coordinate_aliases = {
        "dataset coordinates",
        "dataset coordinate units",
        "dataset coordinate frame",
        "dataset's coordinate frame",
        "stored cartesian coordinates",
        "cartesian coordinates",
    }
    resolved = False
    canonical_value = None
    canonical_unit = None
    rule = "explicit_no_match"
    if predicted_unit == reference_unit:
        resolved = True
        canonical_value = request.get("predicted_value")
        canonical_unit = reference_unit
        rule = "identical_explicit_unit"
    elif (
        reference_unit is None
        and normalized in coordinate_aliases
        and isinstance(authority, Mapping)
        and authority.get("coordinate_or_numeric_scale") == "CASE_READER_METADATA"
    ):
        # Authorization is bound to the compiled case policy; the label alone
        # never creates coordinate semantics.
        resolved = True
        canonical_value = request.get("predicted_value")
        rule = "case_reader_metadata_coordinate_frame"
    return UnitFrameResolution(
        request_id=continuation_request_id("unit_frame", request),
        request=request,
        status="RESOLVED" if resolved else "NO_MATCH",
        canonical_value=canonical_value,
        canonical_unit=canonical_unit,
        rule=rule,
        provenance=(
            {
                "authority": "explicit_case_verification_policy",
                "schema_version": SCHEMA_VERSION,
            }
            if resolved
            else {}
        ),
    )


def _identity(
    handler: Any, *, capability: str, model_backed: bool = False
) -> Any:
    marker = f"{SCHEMA_VERSION}:{handler.__name__}"
    identity: dict[str, Any] = {
        "execution_capability": capability,
        "prompt_sha256": _digest(marker + ":prompt"),
        "schema_sha256": _digest(marker + ":schema"),
    }
    if model_backed:
        # Do not read the host provider configuration while importing this
        # module.  The registry resolves this factory only when it freezes a
        # production execution manifest or audits production capability.
        handler.__flowintentbench_identity_factory__ = lambda: {
            **identity,
            **_declared_model_identity(),
        }
    handler.__flowintentbench_identity__ = identity
    return handler


novel_o_completion = _identity(
    novel_o_completion, capability="PRODUCTION_DETERMINISTIC"
)
novel_o_materializer = _identity(
    novel_o_materializer, capability="PRODUCTION_DETERMINISTIC"
)
unit_frame_resolver = _identity(
    unit_frame_resolver, capability="PRODUCTION_DETERMINISTIC"
)
semantic_match_resolver = _identity(
    semantic_match_resolver, capability="PRODUCTION_MODEL", model_backed=True
)
blinded_scientific_adjudicator = _identity(
    blinded_scientific_adjudicator,
    capability="PRODUCTION_MODEL",
    model_backed=True,
)
blinded_finding_adjudicator = _identity(
    blinded_finding_adjudicator,
    capability="PRODUCTION_MODEL",
    model_backed=True,
)


HANDLERS = {
    "NOVEL_O_COMPLETION": novel_o_completion,
    "NOVEL_O_MATERIALIZER": novel_o_materializer,
    "BLINDED_SCIENTIFIC_ADJUDICATOR": blinded_scientific_adjudicator,
    "BLINDED_FINDING_ADJUDICATOR": blinded_finding_adjudicator,
    "SEMANTIC_MATCH_RESOLVER": semantic_match_resolver,
    "UNIT_FRAME_RESOLVER": unit_frame_resolver,
}

HANDLER_SPEC = {
    owner: f"flowintentbench.production_continuation:{handler.__name__}"
    for owner, handler in HANDLERS.items()
}


__all__ = ["HANDLERS", "HANDLER_SPEC"]
