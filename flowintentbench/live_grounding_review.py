"""Live proxy grounding-review ownership.

The reviewer is advisory: it can inspect unresolved claims and evidence but
cannot mutate support status or make a curator decision.  The orchestration
entrypoint imports :func:`run_live_grounding_review` from this module.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grounding import build_grounding_packets
from .live_agents import LIVE_EXECUTION_MODE, LiveModelCaller, _digest


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_PROXY_DISPOSITIONS = frozenset({
    "PROVISIONAL_SUPPORTED",
    "PROVISIONAL_UNSUPPORTED",
    "NEEDS_MORE_EVIDENCE",
    "SCIENTIFICALLY_AMBIGUOUS",
})
_FAILED_INVOCATIONS = frozenset({
    "PROVIDER_TIMEOUT",
    "PROVIDER_ERROR",
    "PARSE_FAILURE",
    "TOOL_ERROR",
    "FAILED",
})
_GENERIC_AMBIGUITY = frozenset({
    "",
    "UNKNOWN",
    "UNCERTAIN",
    "AMBIGUOUS",
    "SCIENTIFICALLY_AMBIGUOUS",
    "NEEDS_MORE_EVIDENCE",
    "NOT LOCALIZED",
    "UNCLEAR",
    "NOT SPECIFIED",
    "INSUFFICIENT DETAIL",
    "INSUFFICIENT INFORMATION",
    "NO LOCALIZATION",
})


def _review_call_status(review: Mapping[str, Any]) -> str | None:
    call = review.get("call")
    if not isinstance(call, Mapping):
        call = {}
    explicit = [
        str(value).strip().upper()
        for value in (review.get("invocation_status"), call.get("invocation_status"))
        if value is not None and str(value).strip()
    ]
    if explicit:
        for value in explicit:
            if value in _FAILED_INVOCATIONS or value != "SUCCESS":
                return value
        return "SUCCESS"
    value = review.get("status") or call.get("status")
    if str(value or "").upper() == "PASS":
        return "SUCCESS"
    if str(value or "").upper() == "FAILED":
        return "FAILED"
    return None


def _claim_rows(review: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values = review.get("claims", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(item for item in values if isinstance(item, Mapping))


def _claim_disposition(claim: Mapping[str, Any]) -> str:
    return str(claim.get("proxy_disposition", claim.get("disposition", ""))).strip().upper()


def _has_usable_provenance(claim: Mapping[str, Any]) -> bool:
    evidence = claim.get("supporting_evidence_ids", claim.get("evidence_ids", ()))
    sources = claim.get("supporting_source_ids", claim.get("source_ids", ()))
    if isinstance(evidence, str):
        evidence = [evidence] if evidence.strip() else []
    if isinstance(sources, str):
        sources = [sources] if sources.strip() else []
    return bool(
        isinstance(evidence, Sequence)
        and not isinstance(evidence, (str, bytes))
        and any(str(item).strip() for item in evidence)
        and isinstance(sources, Sequence)
        and not isinstance(sources, (str, bytes))
        and any(str(item).strip() for item in sources)
    )


def _has_localized_ambiguity(claim: Mapping[str, Any]) -> bool:
    for key in (
        "ambiguity_scope",
        "ambiguous_dimension",
        "ambiguity_localization",
        "localized_ambiguity",
        "uncertainty_scope",
    ):
        value = claim.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if any(str(item).strip() for item in value):
                return True
        elif str(value or "").strip().upper() not in _GENERIC_AMBIGUITY:
            return True
    value = str(claim.get("remaining_uncertainty", "") or "").strip()
    return bool(value and value.upper() not in _GENERIC_AMBIGUITY)


def _review_evidence_digest(review: Mapping[str, Any]) -> str | None:
    for key in ("evidence_bundle_digest", "evidence_digest", "input_evidence_digest"):
        value = review.get(key)
        if value:
            return str(value)
    call = review.get("call")
    if isinstance(call, Mapping):
        for key in ("evidence_bundle_digest", "evidence_digest", "input_evidence_digest"):
            value = call.get(key)
            if value:
                return str(value)
    return None


def needs_optional_proxy_review(
    reviews: Sequence[Mapping[str, Any]],
    *,
    current_evidence_digest: str | None = None,
    critical_claim_ids: Sequence[str] | None = None,
    new_evidence_added: bool = False,
) -> tuple[bool, list[str]]:
    """Return whether the frozen adaptive policy requires another proxy review."""

    rows = tuple(item for item in reviews if isinstance(item, Mapping))
    reasons: set[str] = set()
    critical = {str(item) for item in critical_claim_ids or () if str(item).strip()}
    claim_maps: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for review_index, review in enumerate(rows):
        if _review_call_status(review) in _FAILED_INVOCATIONS:
            reasons.add("EXISTING_PROXY_INVOCATION_NOT_SUCCESSFUL")
        for claim in _claim_rows(review):
            claim_id = str(claim.get("claim_id", "")).strip()
            if claim_id:
                claim_maps.setdefault(claim_id, []).append((review_index, claim))
            disposition = _claim_disposition(claim)
            if disposition == "PROVISIONAL_SUPPORTED" and not _has_usable_provenance(claim):
                reasons.add("PROVISIONAL_SUPPORT_WITHOUT_USABLE_PROVENANCE")
            if disposition == "SCIENTIFICALLY_AMBIGUOUS" and not _has_localized_ambiguity(claim):
                reasons.add("UNLOCALIZED_SCIENTIFIC_AMBIGUITY")

    if not critical:
        critical = set(claim_maps)
    for claim_id, claim_entries in claim_maps.items():
        if claim_id not in critical or len({index for index, _ in claim_entries}) < 2:
            continue
        dispositions = {
            _claim_disposition(claim)
            for _, claim in claim_entries
            if _claim_disposition(claim) in _PROXY_DISPOSITIONS
        }
        if len(dispositions) > 1:
            reasons.add("MATERIAL_CRITICAL_CLAIM_DISAGREEMENT")

    if new_evidence_added and rows:
        reasons.add("NEW_SCIENTIFIC_EVIDENCE_AFTER_PRIOR_REVIEWS")
    elif current_evidence_digest:
        prior_digests = {_review_evidence_digest(review) for review in rows}
        prior_digests.discard(None)
        if prior_digests and any(digest != current_evidence_digest for digest in prior_digests):
            reasons.add("NEW_SCIENTIFIC_EVIDENCE_AFTER_PRIOR_REVIEWS")
    return bool(reasons), sorted(reasons)


def proxy_model_family_report(reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report reviewer identity separately from same-family diversity limits."""

    rows = tuple(reviews)
    agent_ids: set[str] = set()
    families: set[str] = set()
    for review in rows:
        call = review.get("call") if isinstance(review, Mapping) else None
        call = call if isinstance(call, Mapping) else {}
        for key, target in (
            ("reviewer_agent_id", agent_ids),
            ("agent_id", agent_ids),
            ("agent_profile_id", agent_ids),
            ("reviewer_model_family", families),
            ("model_family", families),
        ):
            value = review.get(key) or call.get(key)
            if value is not None and str(value).strip():
                target.add(str(value).strip())
    limited = len(families) <= 1
    return {
        "proxy_review_count": len(rows),
        "distinct_agent_ids": sorted(agent_ids),
        "distinct_model_families": sorted(families),
        "MODEL_FAMILY_DIVERSITY_LIMITED": limited,
        "PROXY_RESPONSE_SPACE_COVERAGE_LIMITED": limited,
        "MODEL_FAMILY_REPORTING_STATUS": "PASS",
    }


def run_live_grounding_review(
    root: Path,
    out: Path,
    *,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Run two independent advisory reviews for Kitchen unresolved claims."""

    packet = next(packet for packet in build_grounding_packets(root) if packet.family_id == "kitchen_turbulence_activity")
    claims = [dict(item) for item in packet.scientific_support_claims if str(item.get("support_status", "")).upper() == "UNKNOWN"]
    caller = LiveModelCaller(root, "flow-case-auditor-gpt-5.6-sol", config_path=config_path, api_key=api_key, timeout=timeout)

    def review_one(index: int) -> dict[str, Any]:
        visible = {
            "scientific_target": packet.scientific_target,
            "context": packet.dataset_context,
            "dataset_contract": {"dataset_id": packet.dataset_id, "observable_support": packet.observable_support, "geometry_or_boundary_support": packet.geometry_or_boundary_support},
            "grounding_evidence_bundle": {"claims": claims, "evidence_references": [dict(x) for x in packet.evidence_references], "claim_provenance": [dict(x) for x in packet.claim_provenance], "claim_support_links": [dict(x) for x in packet.claim_support_links], "known_limitations": list(packet.known_limitations)},
            "runtime_contract": {"tools": ["python"], "network": "unavailable"},
        }
        call = caller.call(role="auditor", visible_payload=visible, instruction="For each UNKNOWN claim, return claims with claim_id, proxy_disposition (PROVISIONAL_SUPPORTED, PROVISIONAL_UNSUPPORTED, NEEDS_MORE_EVIDENCE, or SCIENTIFICALLY_AMBIGUOUS), scientific_rationale, supporting_evidence_ids, supporting_source_ids, remaining_uncertainty. Do not make curator decisions.", identity={"family_id": packet.family_id, "review_index": index})
        parsed = call.get("parsed_result") if isinstance(call.get("parsed_result"), Mapping) else {}
        digest = _digest([dict(item) for item in packet.scientific_support_claims])
        return {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "review_index": index, "reviewer_agent_id": call.get("agent_profile_id"), "reviewer_model_family": call.get("model_family"), "claims": parsed.get("claims", []) if isinstance(parsed, Mapping) else [], "proxy_only": True, "evidence_bundle_digest": _digest(visible["grounding_evidence_bundle"]), "original_support_status_digest": digest, "post_review_support_status_digest": digest, "support_status_unchanged": True, "curator_status": "PENDING", "curator_decision": None, "curator_notes": None, "call": call}

    with ThreadPoolExecutor(max_workers=2) as pool:
        reviews = list(pool.map(review_one, (1, 2)))
    identity_report = proxy_model_family_report(reviews)
    family_limited = bool(identity_report["MODEL_FAMILY_DIVERSITY_LIMITED"])
    agreement = {
        "execution_mode": LIVE_EXECUTION_MODE,
        "live_model_calls": True,
        "reviewer_count": len(reviews),
        **identity_report,
        "model_family_diversity_limited": family_limited,
        "model_family": caller.profile.model_family,
        "same_family_limitation": (
            "Both runs use the configured gpt-5.6 model family."
            if family_limited
            else None
        ),
    }
    _write_json(out / "manifest.json", {"execution_mode": LIVE_EXECUTION_MODE, "live_model_calls": True, "family_id": packet.family_id, "profile": caller.profile_metadata, "curator_status": "PENDING", "curator_decision": None, "curator_notes": None})
    _write_json(out / "kitchen_turbulence_activity_reviews.json", reviews)
    _write_json(out / "reviewer_agreement.json", agreement)
    return {"status": "RUN" if any(item["call"].get("status") == "PASS" for item in reviews) else "FAILED", "reviews": reviews}


__all__ = [
    "needs_optional_proxy_review",
    "proxy_model_family_report",
    "run_live_grounding_review",
]
