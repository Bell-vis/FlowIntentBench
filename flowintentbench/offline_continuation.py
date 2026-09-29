"""Deterministic, no-model continuation handlers used by the offline gate.

These handlers are protocol fixtures, not scientific judges.  They are useful
for proving that the production registry can load all owner identities and
that typed semantic/unit requests can be resolved without a provider call.
The two adjudicator owners intentionally keep a pending record: no offline
fixture is allowed to invent a scientific verdict or Ground Truth binding.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .evaluator import (
    SemanticMatchRequest,
    SemanticMatchResolution,
    SemanticMatchResult,
    UnitFrameResolution,
    continuation_request_id,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").casefold().split()).rstrip(".")


def semantic_match_resolver(packet: Mapping[str, Any]) -> SemanticMatchResolution:
    request_value = (packet.get("model_claims") or {}).get("semantic_match_request")
    if not isinstance(request_value, Mapping):
        raise ValueError("offline semantic resolver requires semantic_match_request")
    request = SemanticMatchRequest.from_dict(request_value)
    result = (
        SemanticMatchResult.MATCH
        if _normalize(request.predicted_statement) == _normalize(request.reference_statement)
        else SemanticMatchResult.NO_MATCH
    )
    return SemanticMatchResolution(
        continuation_request_id("semantic_match", request.to_dict()),
        request,
        result,
    )


def unit_frame_resolver(packet: Mapping[str, Any]) -> UnitFrameResolution:
    request = (packet.get("model_claims") or {}).get("unit_frame_request")
    if not isinstance(request, Mapping):
        raise ValueError("offline unit resolver requires unit_frame_request")
    request = dict(request)
    predicted_unit = str(request.get("predicted_unit") or "").strip().casefold()
    reference_unit = str(request.get("reference_unit") or "").strip().casefold()
    predicted_value = request.get("predicted_value")
    reference_value = request.get("reference_value")
    canonical_value = None
    canonical_unit = request.get("reference_unit")
    rule = "explicit_no_match"
    resolved = predicted_unit == reference_unit and bool(reference_unit)
    if resolved:
        canonical_value = predicted_value
        rule = "same_explicit_unit"
    elif predicted_unit == "mm" and reference_unit == "m" and isinstance(predicted_value, (int, float)):
        canonical_value = float(predicted_value) / 1000.0
        rule = "millimetre_to_metre"
        resolved = True
    elif predicted_unit == "m" and reference_unit == "mm" and isinstance(predicted_value, (int, float)):
        canonical_value = float(predicted_value) * 1000.0
        rule = "metre_to_millimetre"
        resolved = True
    status = "RESOLVED" if resolved else "NO_MATCH"
    provenance = (
        {
            "authority": "offline_explicit_unit_fixture",
            "reference_value_observed": reference_value,
            "rule_version": "offline-unit-fixture-v1",
        }
        if resolved
        else {}
    )
    return UnitFrameResolution(
        continuation_request_id("unit_frame", request),
        request,
        status,
        canonical_value,
        canonical_unit if resolved else None,
        rule,
        provenance,
    )


def _scientific_owner_pending(_packet: Mapping[str, Any]) -> None:
    """Keep scientific adjudication pending in a no-model run."""

    return None


def novel_o_completion(_packet: Mapping[str, Any]) -> None:
    return None


def novel_o_materializer(_packet: Mapping[str, Any]) -> None:
    return None


def blinded_scientific_adjudicator(_packet: Mapping[str, Any]) -> None:
    return None


def blinded_finding_adjudicator(_packet: Mapping[str, Any]) -> None:
    return None


def _identity(handler, *, model_backed: bool = False):
    marker = f"offline-continuation:{handler.__name__}:v1"
    identity = {
        "provider": "offline-fixture",
        "model": "deterministic-no-model" if model_backed else "not-applicable",
        "model_configuration": {"fixture": True, "model_calls": 0},
        "execution_capability": "PROTOCOL_ONLY",
        "prompt_sha256": _digest(marker + ":prompt"),
        "schema_sha256": _digest(marker + ":schema"),
    }
    handler.__flowintentbench_identity__ = identity
    return handler


semantic_match_resolver = _identity(semantic_match_resolver)
unit_frame_resolver = _identity(unit_frame_resolver)
novel_o_completion = _identity(novel_o_completion)
novel_o_materializer = _identity(novel_o_materializer)
blinded_scientific_adjudicator = _identity(blinded_scientific_adjudicator, model_backed=True)
blinded_finding_adjudicator = _identity(blinded_finding_adjudicator, model_backed=True)


HANDLERS = {
    "SEMANTIC_MATCH_RESOLVER": semantic_match_resolver,
    "UNIT_FRAME_RESOLVER": unit_frame_resolver,
    "NOVEL_O_COMPLETION": novel_o_completion,
    "NOVEL_O_MATERIALIZER": novel_o_materializer,
    "BLINDED_SCIENTIFIC_ADJUDICATOR": blinded_scientific_adjudicator,
    "BLINDED_FINDING_ADJUDICATOR": blinded_finding_adjudicator,
}

HANDLER_SPEC = {
    owner: f"flowintentbench.offline_continuation:{name}"
    for owner, name in {
        "SEMANTIC_MATCH_RESOLVER": "semantic_match_resolver",
        "UNIT_FRAME_RESOLVER": "unit_frame_resolver",
        "NOVEL_O_COMPLETION": "novel_o_completion",
        "NOVEL_O_MATERIALIZER": "novel_o_materializer",
        "BLINDED_SCIENTIFIC_ADJUDICATOR": "blinded_scientific_adjudicator",
        "BLINDED_FINDING_ADJUDICATOR": "blinded_finding_adjudicator",
    }.items()
}


__all__ = ["HANDLERS", "HANDLER_SPEC"]
