"""Fail-closed lifecycle and two independent curator gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class LifecycleState(str, Enum):
    DISCOVERED = "DISCOVERED"
    DATA_SUPPORTED = "DATA_SUPPORTED"
    SCIENTIFICALLY_GROUNDED = "SCIENTIFICALLY_GROUNDED"
    GROUNDING_CURATOR_CONFIRMED = "GROUNDING_CURATOR_CONFIRMED"
    CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED = "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED"
    SCQ_PASS = "SCQ_PASS"
    SRAC = "SRAC"
    EVALUATION_CONTRACT_CONFIRMED = "EVALUATION_CONTRACT_CONFIRMED"
    EVALUATOR_CALIBRATED = "EVALUATOR_CALIBRATED"
    RELEASE_ELIGIBLE = "RELEASE_ELIGIBLE"


class GateDecision(str, Enum):
    CONFIRMED = "CONFIRMED"
    PENDING = "PENDING"
    REVISE = "REVISE"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CuratorGateArtifact:
    gate: str
    reviewer_id: str | None
    artifact_sha256: str | None
    timestamp: str | None
    decision: str
    revision_notes: str | None = None
    proxy_generated: bool = False

    def __post_init__(self) -> None:
        if self.gate not in {"GROUNDING_CURATOR_CONFIRMED", "EVALUATION_CONTRACT_CONFIRMED"}:
            raise ValueError("unknown curator gate")
        if self.decision not in {item.value for item in GateDecision}:
            raise ValueError("unknown gate decision")
        if self.decision == "CONFIRMED":
            if self.proxy_generated:
                raise ValueError("proxy agents cannot confirm curator gates")
            if not self.reviewer_id or not self.artifact_sha256 or not self.timestamp:
                raise ValueError("confirmed curator gate requires reviewer, hash, and timestamp")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "reviewer_id": self.reviewer_id,
            "artifact_sha256": self.artifact_sha256,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "revision_notes": self.revision_notes,
            "proxy_generated": self.proxy_generated,
        }


_CONDITION_DISPOSITIONS = frozenset({
    "ELIGIBLE",
    "REVISE_QUESTION",
    "NOT_ELIGIBLE",
    "MORE_EVIDENCE_REQUIRED",
})
_CURATOR_PACKET_PIN_FIELDS = (
    "scientific_review_run_id",
    "scientific_review_hash",
    "evidence_binding_hash",
    "condition_contract_hash",
)

_NEXT_REAL_SCIENTIFIC_ACTION = {
    "PENDING": "HUMAN_GROUNDING_CURATOR_REVIEW",
    "CONFIRMED": "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATION",
    "REVISE": "APPLY_CURATOR_REQUESTED_REVISION",
    "REJECT": "FAMILY_STOPPED_BY_CURATOR",
    "INVALID": "RESOLVE_INVALID_CURATOR_ARTIFACT",
}

_TARGET_APPROVAL_DECISIONS = frozenset({"APPROVED", "CONFIRMED", "PENDING", "REJECTED"})


def validate_target_approval_lineage(
    lineage: Mapping[str, Any] | None,
    approval: Mapping[str, Any] | None = None,
    *,
    required: bool = False,
    expected_target_id: str | None = None,
    target_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the identity-authoring approval embedded in a lineage record.

    Target selection is deliberately a construction/identity prerequisite.  It
    is not a ``GROUNDING_CURATOR_CONFIRMED`` artifact and this function never
    changes the normative lifecycle.  A missing approval is therefore reported
    as ``PENDING`` only when the caller marks it as required (for example, the
    provisional Combustor slots); other datasets receive ``NOT_REQUIRED``.

    The accepted record is intentionally small and transport-neutral: an
    explicit decision, target identifier, accountable reviewer, timestamp, and
    a non-proxy marker.  Additional lineage fields remain opaque to this
    validator.  This keeps target approval separate from scientific grounding
    while still preventing a proxy or an unbound target from authoring identity.
    """

    # The optional second positional argument is a compatibility convenience
    # for callers that keep review and approval as separate objects.  When the
    # first object clearly has target-review fields and no approval was given,
    # interpret it as the review and leave the approval pending.
    looks_like_target_review = isinstance(lineage, Mapping) and any(
        key in lineage
        for key in (
            "recommendation_status",
            "recommended_target_id",
            "recommended_provisional_target_id",
            "candidate_targets",
        )
    )
    if target_review is None and looks_like_target_review:
        target_review = lineage
        lineage = None if approval is None else {"target_approval": approval}
        approval = None
    if approval is not None:
        if lineage is None or not isinstance(lineage, Mapping):
            lineage = {"target_approval": approval}
        elif "target_approval" not in lineage and "human_target_approval" not in lineage:
            lineage = {**lineage, "target_approval": approval}

    review_hash: str | None = None
    review_target_id: str | None = None
    review_family_id: str | None = None
    review_errors: list[str] = []
    if target_review is not None:
        if not isinstance(target_review, Mapping):
            review_errors.append("TARGET_REVIEW_MUST_BE_OBJECT")
        else:
            try:
                review_hash = canonical_target_review_sha256(target_review)
            except (TypeError, ValueError):
                review_errors.append("TARGET_REVIEW_NOT_CANONICALIZABLE")
            recommendation = target_review.get("recommended_target_id")
            if recommendation is None:
                recommendation = target_review.get("recommended_provisional_target_id")
            if recommendation is None:
                alias = target_review.get("recommended_provisional_target")
                if isinstance(alias, Mapping):
                    recommendation = alias.get("candidate_id") or alias.get("target_id")
                elif isinstance(alias, str):
                    recommendation = alias
            review_target_id = str(recommendation or "").strip() or None
            review_family_id = str(target_review.get("family_id", "")).strip() or None
            if (
                str(target_review.get("recommendation_status", "")).strip().upper()
                != "RECOMMENDED_PROVISIONAL_TARGET"
            ):
                review_errors.append("TARGET_REVIEW_RECOMMENDATION_NOT_APPROVED_FOR_LINEAGE")
            if review_target_id is None:
                review_errors.append("TARGET_REVIEW_RECOMMENDED_TARGET_REQUIRED")
            if review_family_id is None:
                review_errors.append("TARGET_REVIEW_FAMILY_ID_REQUIRED")
            if target_review.get("human_approval_required") is True:
                required = True
            if expected_target_id is not None and review_target_id not in {
                None,
                expected_target_id,
            }:
                review_errors.append("TARGET_REVIEW_TARGET_ID_MISMATCH")

    if lineage is None or not isinstance(lineage, Mapping):
        status = "INVALID" if review_errors else ("PENDING" if required else "NOT_REQUIRED")
        return {
            "status": status,
            "target_approval_status": status,
            "approved": False,
            "proxy_generated": False,
            "target_review_sha256": review_hash,
            "errors": [*review_errors, *(["TARGET_APPROVAL_PENDING"] if required else [])],
        }

    # Accept either the nested construction-lineage shape or a direct approval
    # object, without introducing a second schema.
    approval = lineage.get("target_approval")
    if not isinstance(approval, Mapping):
        approval = lineage.get("human_target_approval")
    if not isinstance(approval, Mapping):
        # A construction lineage record normally has ``status=AUTHORED`` but
        # no target decision.  That is not an invalid approval for datasets
        # where target selection is not a separate prerequisite.
        direct_keys = {
            "decision",
            "target_id",
            "approved_target_id",
            "candidate_id",
            "reviewer_id",
            "approver_id",
            "approved_at",
            "proxy_generated",
        }
        if not any(key in lineage for key in direct_keys):
            return {
                "status": "PENDING" if required else "NOT_REQUIRED",
                "target_approval_status": "PENDING" if required else "NOT_REQUIRED",
                "approved": False,
                "proxy_generated": False,
                "errors": ["TARGET_APPROVAL_PENDING"] if required else [],
            }
        approval = lineage

    decision = str(approval.get("decision", approval.get("status", ""))).strip().upper()
    errors: list[str] = list(review_errors)
    approval_type = str(
        approval.get("approval_type", approval.get("artifact_type", ""))
    ).strip().upper()
    if approval_type and approval_type != "HUMAN_TARGET_APPROVAL":
        errors.append("TARGET_APPROVAL_TYPE_MUST_BE_HUMAN_TARGET_APPROVAL")
    if str(approval.get("gate", "")).strip().upper() == "GROUNDING_CURATOR_CONFIRMED":
        errors.append("GROUNDING_CURATOR_GATE_CANNOT_APPROVE_TARGET")
    if decision not in _TARGET_APPROVAL_DECISIONS:
        errors.append("TARGET_APPROVAL_DECISION_MISSING_OR_INVALID")
    proxy_generated = bool(approval.get("proxy_generated", False))
    if decision in {"APPROVED", "CONFIRMED"}:
        if "proxy_generated" not in approval:
            errors.append("TARGET_APPROVAL_PROXY_MARKER_REQUIRED")
        elif approval.get("proxy_generated") is not False:
            errors.append("PROXY_TARGET_APPROVAL_FORBIDDEN")
    target_id = str(
        approval.get("target_id")
        or approval.get("approved_target_id")
        or approval.get("candidate_id")
        or ""
    ).strip()
    if decision in {"APPROVED", "CONFIRMED"} and not target_id:
        errors.append("TARGET_APPROVAL_TARGET_ID_MISSING")
    if expected_target_id and target_id and target_id != expected_target_id:
        errors.append("TARGET_APPROVAL_TARGET_ID_MISMATCH")
    if review_target_id and target_id and target_id != review_target_id:
        errors.append("TARGET_APPROVAL_TARGET_ID_MISMATCH")
    family_id = str(approval.get("family_id", "")).strip() or None
    if review_family_id and family_id != review_family_id:
        errors.append("TARGET_APPROVAL_FAMILY_ID_MISMATCH")
    supplied_review_hash = str(
        approval.get(
            "target_review_sha256",
            approval.get(
                "target_selection_review_hash",
                approval.get("target_review_hash", ""),
            ),
        )
    ).strip().lower()
    if review_hash is not None:
        if not supplied_review_hash:
            errors.append("TARGET_APPROVAL_REVIEW_HASH_REQUIRED")
        elif supplied_review_hash != review_hash:
            errors.append("TARGET_APPROVAL_REVIEW_HASH_MISMATCH")
    reviewer_id = str(
        approval.get("reviewer_id") or approval.get("approver_id") or ""
    ).strip()
    timestamp = str(approval.get("timestamp") or approval.get("approved_at") or "").strip()
    if decision in {"APPROVED", "CONFIRMED"} and not reviewer_id:
        errors.append("TARGET_APPROVAL_REVIEWER_ID_MISSING")
    if decision in {"APPROVED", "CONFIRMED"} and not timestamp:
        errors.append("TARGET_APPROVAL_TIMESTAMP_MISSING")

    if decision in {"PENDING", "REJECTED"} and not errors:
        status = decision
    elif errors:
        status = "INVALID"
    else:
        status = "APPROVED"
    return {
        "status": "PASS" if status == "APPROVED" else status,
        "target_approval_status": status,
        "approved": status == "APPROVED",
        "target_id": target_id or None,
        "approved_target_id": target_id or None,
        "family_id": family_id,
        "reviewer_id": reviewer_id or None,
        "timestamp": timestamp or None,
        "proxy_generated": proxy_generated,
        "target_review_sha256": review_hash,
        "errors": sorted(set(errors)),
    }


def canonical_curator_packet_sha256(packet: Mapping[str, Any]) -> str:
    """Return the canonical digest used by an external curator artifact.

    The hash binds the curator decision to the complete authoritative JSON
    packet, including its selected scientific-run and construction hashes.
    It is intentionally not a second curator packet schema.
    """

    return hashlib.sha256(
        json.dumps(
            dict(packet),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_target_review_sha256(target_review: Mapping[str, Any]) -> str:
    """Return the digest used to bind a target approval to its review.

    Target approval is a construction decision and is intentionally separate
    from the Grounding Curator gate.  This helper only canonicalizes the
    existing target-selection review mapping; it does not introduce a new
    persisted schema or lifecycle state.
    """

    if not isinstance(target_review, Mapping):
        raise TypeError("target review must be a mapping")
    return hashlib.sha256(
        json.dumps(
            dict(target_review),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _condition_disposition(condition: Mapping[str, Any]) -> str:
    deterministic = condition.get("deterministic_condition_eligibility")
    if not isinstance(deterministic, Mapping):
        deterministic = condition.get("deterministic_eligibility")
    if isinstance(deterministic, Mapping):
        value = deterministic.get("eligibility")
        if value is not None:
            return str(value).strip().upper()
    return str(
        condition.get("eligibility_recommendation", condition.get("eligibility", ""))
    ).strip().upper()


def derive_sparse_curator_condition_set(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Derive condition candidates from the packet's frozen dispositions.

    Family review is sparse: a held or ineligible condition does not force an
    otherwise valid condition out of the candidate set.  This artifact is a
    lifecycle transition view and does not mutate case or Dataset records.
    """

    values = packet.get("conditions")
    conditions = (
        values
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes))
        else ()
    )
    dispositions: dict[str, str] = {}
    errors: list[str] = []
    for value in conditions:
        if not isinstance(value, Mapping):
            errors.append("INVALID_CONDITION_RECORD")
            continue
        condition = str(value.get("condition", "")).strip()
        if not condition:
            errors.append("MISSING_CONDITION_ID")
            continue
        if condition in dispositions:
            errors.append(f"DUPLICATE_CONDITION_ID:{condition}")
            continue
        disposition = _condition_disposition(value)
        if disposition not in _CONDITION_DISPOSITIONS:
            errors.append(f"UNKNOWN_CONDITION_DISPOSITION:{condition}:{disposition or 'MISSING'}")
        dispositions[condition] = disposition

    if not dispositions:
        errors.append("EMPTY_CONDITION_SET")
    candidates = [
        condition for condition, disposition in dispositions.items()
        if disposition == "ELIGIBLE"
    ]
    held = [
        condition for condition, disposition in dispositions.items()
        if disposition == "REVISE_QUESTION"
    ]
    blocked = [
        condition for condition, disposition in dispositions.items()
        if disposition in {"NOT_ELIGIBLE", "MORE_EVIDENCE_REQUIRED"}
    ]
    return {
        "status": "PASS" if not errors else "FAIL",
        "curator_candidate_condition_set": candidates,
        "held_for_revision": held,
        "blocked_condition_set": blocked,
        "condition_dispositions": dispositions,
        "errors": errors,
        "dataset_schema_mutated": False,
    }


def validate_grounding_curator_gate_artifact(
    packet: Mapping[str, Any],
    artifact: CuratorGateArtifact | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Ingest an existing gate artifact and validate exact packet binding.

    No artifact is generated here.  ``None`` means the real human gate is
    still pending.  A supplied non-pending decision must be attributable and
    bound to the exact canonical packet hash; a mismatch fails closed.
    """

    packet_hash = canonical_curator_packet_sha256(packet)
    missing_pins = [
        key for key in _CURATOR_PACKET_PIN_FIELDS
        if not str(packet.get(key, "")).strip()
    ]
    errors: list[str] = []
    if missing_pins:
        errors.extend(f"MISSING_CURATOR_PACKET_PIN:{key}" for key in missing_pins)
    if str(packet.get("curator_handoff_readiness", "")) != "READY":
        errors.append("CURATOR_PACKET_NOT_READY")
    if str(packet.get("curator_status", "")).strip().upper() != "PENDING":
        errors.append("CURATOR_PACKET_AUTHORED_STATUS_NOT_PENDING")
    if packet.get("curator_decision") is not None:
        errors.append("CURATOR_PACKET_AUTHORED_DECISION_NOT_NULL")

    if artifact is None:
        return {
            "status": "PENDING" if not errors else "INVALID",
            "grounding_curator_gate_status": "PENDING",
            "curator_artifact_supplied": False,
            "packet_sha256": packet_hash,
            "artifact_sha256": None,
            "packet_hash_matches": False,
            "packet_pins_complete": not missing_pins,
            "errors": errors,
        }

    try:
        gate = artifact if isinstance(artifact, CuratorGateArtifact) else CuratorGateArtifact(
            gate=str(artifact.get("gate", "")),
            reviewer_id=artifact.get("reviewer_id"),
            artifact_sha256=artifact.get("artifact_sha256"),
            timestamp=artifact.get("timestamp"),
            decision=str(artifact.get("decision", "")),
            revision_notes=artifact.get("revision_notes"),
            proxy_generated=bool(artifact.get("proxy_generated", False)),
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "INVALID",
            "grounding_curator_gate_status": "INVALID",
            "curator_artifact_supplied": True,
            "packet_sha256": packet_hash,
            "artifact_sha256": None,
            "packet_hash_matches": False,
            "packet_pins_complete": not missing_pins,
            "errors": [*errors, f"INVALID_CURATOR_ARTIFACT:{exc}"],
        }

    if gate.gate != "GROUNDING_CURATOR_CONFIRMED":
        errors.append("INVALID_GROUNDING_CURATOR_GATE")
    if gate.proxy_generated:
        errors.append("PROXY_CURATOR_ARTIFACT_FORBIDDEN")
    if gate.decision != "PENDING":
        if not str(gate.reviewer_id or "").strip():
            errors.append("MISSING_CURATOR_REVIEWER_ID")
        if not str(gate.timestamp or "").strip():
            errors.append("MISSING_CURATOR_TIMESTAMP")
        if not str(gate.artifact_sha256 or "").strip():
            errors.append("MISSING_CURATOR_PACKET_HASH")
    hash_matches = gate.artifact_sha256 == packet_hash
    if gate.decision != "PENDING" and not hash_matches:
        errors.append("CURATOR_PACKET_HASH_MISMATCH")

    return {
        "status": "PASS" if not errors else "INVALID",
        "grounding_curator_gate_status": gate.decision if not errors else "INVALID",
        "curator_artifact_supplied": True,
        "packet_sha256": packet_hash,
        "artifact_sha256": gate.artifact_sha256,
        "packet_hash_matches": hash_matches,
        "packet_pins_complete": not missing_pins,
        "reviewer_id": gate.reviewer_id,
        "timestamp": gate.timestamp,
        "proxy_generated": gate.proxy_generated,
        "errors": errors,
    }


def next_real_scientific_action(grounding_curator_gate_status: str) -> str:
    """Return the next lifecycle action implied by a validated gate state.

    The authored packet status is deliberately not accepted as an input here:
    it records the immutable pre-review state, whereas progression is governed
    by the independently supplied and validated external curator artifact.
    """

    status = str(grounding_curator_gate_status).strip().upper()
    return _NEXT_REAL_SCIENTIFIC_ACTION.get(
        status,
        "RESOLVE_INVALID_CURATOR_ARTIFACT",
    )


def summarize_grounding_curator_state(
    packet: Mapping[str, Any],
    artifact: CuratorGateArtifact | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep immutable packet authorship separate from the external gate state."""

    gate = validate_grounding_curator_gate_artifact(packet, artifact)
    genuine_artifact = bool(
        gate.get("curator_artifact_supplied")
        and gate.get("status") == "PASS"
        and gate.get("proxy_generated") is False
        and gate.get("grounding_curator_gate_status")
        in {"CONFIRMED", "REVISE", "REJECT"}
    )
    return {
        "CURATOR_PACKET_AUTHORED_STATUS": str(
            packet.get("curator_status", "PENDING")
        ).strip().upper(),
        "GROUNDING_CURATOR_GATE_STATUS": gate["grounding_curator_gate_status"],
        "GENUINE_CURATOR_ARTIFACT_SUPPLIED": genuine_artifact,
        "NEXT_REAL_SCIENTIFIC_ACTION": next_real_scientific_action(
            gate["grounding_curator_gate_status"]
        ),
        "curator_gate_validation": gate,
    }


def o3_revision_candidate_sha256(packet: Mapping[str, Any]) -> str | None:
    """Digest the exact O3 wording candidate exposed to the curator."""

    candidate = packet.get("o3_revised_question_candidate")
    if not isinstance(candidate, Mapping):
        return None
    question = candidate.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    return hashlib.sha256(
        json.dumps(
            {"condition": "O3-F1", "question": question},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def build_o3_revision_lifecycle(
    packet: Mapping[str, Any],
    artifact: CuratorGateArtifact | Mapping[str, Any] | None,
    revision_action: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive an O3 revision version only from an explicit curator action.

    ``revision_notes`` is deliberately ignored because free text cannot safely
    authorize a canonical case change.  The optional action is a transition
    input, not a new scientific schema.  It must identify the exact candidate
    digest already bound into the immutable curator packet.
    """

    conditions = {
        str(item.get("condition", "")): item
        for item in packet.get("conditions", ())
        if isinstance(item, Mapping)
    }
    canonical = conditions.get("O3-F1", {})
    canonical_disposition = _condition_disposition(canonical)
    candidate = packet.get("o3_revised_question_candidate")
    candidate = candidate if isinstance(candidate, Mapping) else {}
    candidate_digest = o3_revision_candidate_sha256(packet)
    gate = validate_grounding_curator_gate_artifact(packet, artifact)
    base = {
        "status": "HELD_FOR_CURATOR_ACTION",
        "condition": "O3-F1",
        "canonical_case_id": canonical.get("case_id"),
        "canonical_status": canonical_disposition or "NOT_AVAILABLE",
        "curator_gate_status": gate["grounding_curator_gate_status"],
        "curator_action": None,
        "revision_action_sha256": None,
        "candidate_question_sha256": candidate_digest,
        "revised_case_version": None,
        "o3_reentered_validation": False,
        "controlled_family_validation_status": "NOT_RUN",
        "official_scq_route_permitted": False,
        "canonical_case_mutated": False,
        "errors": [],
    }
    if revision_action is None:
        return base
    if not isinstance(revision_action, Mapping):
        return {
            **base,
            "status": "INVALID_REVISION_ACTION",
            "errors": ["O3_REVISION_ACTION_MUST_BE_OBJECT"],
        }

    action = str(revision_action.get("action", "")).strip().upper()
    condition = str(revision_action.get("condition", "")).strip().upper()
    supplied_digest = str(
        revision_action.get("candidate_question_sha256", "")
    ).strip().lower()
    action_digest = hashlib.sha256(
        json.dumps(
            dict(revision_action),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    errors: list[str] = []
    if action != "ACCEPT_PROPOSED_WORDING":
        errors.append("UNSUPPORTED_O3_REVISION_ACTION")
    if condition != "O3-F1":
        errors.append("O3_REVISION_CONDITION_MISMATCH")
    if not candidate_digest or supplied_digest != candidate_digest:
        errors.append("O3_REVISION_CANDIDATE_HASH_MISMATCH")
    if canonical_disposition != "REVISE_QUESTION":
        errors.append("CANONICAL_O3_NOT_HELD_FOR_REVISION")
    if not (
        gate.get("status") == "PASS"
        and gate.get("curator_artifact_supplied") is True
        and gate.get("packet_hash_matches") is True
        and gate.get("proxy_generated") is False
        and gate.get("grounding_curator_gate_status") in {"CONFIRMED", "REVISE"}
    ):
        errors.append("VALIDATED_HUMAN_CURATOR_ACTION_REQUIRED")
    if revision_action.get("proxy_generated") is not False:
        errors.append("PROXY_O3_REVISION_ACTION_FORBIDDEN")
    if revision_action.get("reviewer_id") != gate.get("reviewer_id"):
        errors.append("O3_REVISION_REVIEWER_BINDING_MISMATCH")
    if revision_action.get("timestamp") != gate.get("timestamp"):
        errors.append("O3_REVISION_TIMESTAMP_BINDING_MISMATCH")
    if revision_action.get("curator_packet_sha256") != gate.get("packet_sha256"):
        errors.append("O3_REVISION_PACKET_BINDING_MISMATCH")

    target = candidate.get("target_fidelity_audit")
    responsibility = candidate.get("responsibility_audit")
    eligibility = candidate.get("eligibility_predicate")
    target_pass = isinstance(target, Mapping) and target.get("status") == "PASS"
    responsibility_pass = (
        isinstance(responsibility, Mapping)
        and responsibility.get("status") == "PASS"
    )
    materialization_pass = (
        isinstance(eligibility, Mapping)
        and eligibility.get("execution_materialization_status") == "MATERIALIZED"
        and eligibility.get("materialization_support") is True
        and eligibility.get("scientific_materialization_support") is True
    )
    if not target_pass:
        errors.append("O3_REVISED_TARGET_FIDELITY_FAILED")
    if not responsibility_pass:
        errors.append("O3_REVISED_RESPONSIBILITY_INTEGRITY_FAILED")
    if not materialization_pass:
        errors.append("O3_REVISED_MATERIALIZATION_COMPATIBILITY_FAILED")
    if errors:
        return {
            **base,
            "status": "INVALID_REVISION_ACTION",
            "curator_action": action or None,
            "revision_action_sha256": action_digest,
            "errors": errors,
        }

    prior_case_id = str(canonical.get("case_id", "kitchen_o3_f1"))
    version_id = f"{prior_case_id}@revision-{candidate_digest[:12]}"
    revised_version = {
        "version_id": version_id,
        "case_id": prior_case_id,
        "condition": "O3-F1",
        "question": candidate["question"],
        "disposition": "PENDING_CONTROLLED_FAMILY_VALIDATION",
        "eligibility_inherited_from_precheck": False,
        "validation_sequence": {
            "target_fidelity": "PASS",
            "responsibility_integrity": "PASS",
            "deterministic_materialization_compatibility": "PASS",
            "controlled_family_validation": "NOT_RUN",
            "official_scq": "BLOCKED",
        },
        "provenance": {
            "prior_case_id": prior_case_id,
            "prior_question_sha256": hashlib.sha256(
                str(canonical.get("question", "")).encode("utf-8")
            ).hexdigest(),
            "curator_packet_sha256": gate["packet_sha256"],
            "accepted_candidate_sha256": candidate_digest,
            "curator_reviewer_id": gate.get("reviewer_id"),
            "curator_timestamp": gate.get("timestamp"),
            "revision_action_sha256": action_digest,
        },
    }
    return {
        **base,
        "status": "REVISED_CASE_VERSION_CREATED",
        "curator_action": action,
        "revision_action_sha256": action_digest,
        "revised_case_version": revised_version,
        "o3_reentered_validation": True,
        "errors": [],
    }


def can_enter_official_scq(
    condition: str,
    *,
    curator_gate_validation: Mapping[str, Any],
    condition_disposition: str,
    controlled_family_validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the single fail-closed official-SCQ entry decision.

    Curator confirmation is necessary but never sufficient. The controlled
    construction artifact must bind the same packet and contain a passing row
    for this exact eligible condition, including canonical-case and explicit
    representability checks.
    """

    blockers: list[str] = []
    packet_sha256 = curator_gate_validation.get("packet_sha256")
    gate_confirmed = (
        curator_gate_validation.get("status") == "PASS"
        and curator_gate_validation.get("grounding_curator_gate_status") == "CONFIRMED"
        and curator_gate_validation.get("packet_hash_matches") is True
        and curator_gate_validation.get("artifact_sha256") == packet_sha256
        and curator_gate_validation.get("proxy_generated") is False
    )
    if not gate_confirmed:
        blockers.append("GROUNDING_CURATOR_EXACT_CONFIRMATION_REQUIRED")
    if str(condition_disposition).strip().upper() != "ELIGIBLE":
        blockers.append(f"CONDITION_DISPOSITION_{str(condition_disposition).strip().upper() or 'MISSING'}")

    validation = controlled_family_validation if isinstance(controlled_family_validation, Mapping) else {}
    if not validation:
        blockers.append("CONTROLLED_FAMILY_VALIDATION_NOT_RUN")
    elif validation.get("status") != "PASS" or validation.get("controlled_family_construction_validated") is not True:
        blockers.append(f"CONTROLLED_FAMILY_VALIDATION_{str(validation.get('status', 'INVALID')).upper()}")
    if validation and validation.get("curator_packet_sha256") != packet_sha256:
        blockers.append("CONTROLLED_FAMILY_VALIDATION_PACKET_HASH_MISMATCH")

    approved = {
        str(item) for item in validation.get("approved_condition_ids", ())
    } if validation else set()
    validated = {
        str(item) for item in validation.get("validated_condition_set", ())
    } if validation else set()
    if validation and condition not in approved:
        blockers.append("CONDITION_NOT_IN_APPROVED_VALIDATION_SET")
    if validation and condition not in validated:
        blockers.append("CONDITION_NOT_CONTROLLED_FAMILY_VALIDATED")
    row = next((
        item for item in validation.get("validation_results", ())
        if isinstance(item, Mapping) and str(item.get("condition")) == condition
    ), None) if validation else None
    if validation and not isinstance(row, Mapping):
        blockers.append("CONDITION_VALIDATION_RESULT_MISSING")
    elif isinstance(row, Mapping):
        if row.get("status") != "PASS":
            blockers.append("CONDITION_VALIDATION_FAILED")
        if row.get("canonical_case_exists") is not True:
            blockers.append("CANONICAL_CASE_MISSING")
        representability = row.get("evaluation_representability")
        if not isinstance(representability, Mapping) or representability.get("status") != "PASS" or representability.get("contract_present") is not True:
            blockers.append("EVALUATION_REPRESENTABILITY_CONTRACT_REQUIRED")

    blockers = list(dict.fromkeys(blockers))
    return {
        "condition": condition,
        "permitted": not blockers,
        "status": "SCQ_ENTRY_PERMITTED" if not blockers else "OFFICIAL_SCQ_ROUTE_BLOCKED",
        "blockers": blockers,
        "curator_packet_sha256": packet_sha256,
    }


def build_sparse_post_curator_route(
    packet: Mapping[str, Any],
    artifact: CuratorGateArtifact | Mapping[str, Any] | None,
    controlled_family_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed post-curator route without executing SCQ.

    Exact packet confirmation permits only packet conditions already marked
    ``ELIGIBLE`` to proceed to controlled-construction validation. Official
    SCQ remains blocked until the separate controlled-family artifact passes.
    """

    sparse = derive_sparse_curator_condition_set(packet)
    gate = validate_grounding_curator_gate_artifact(packet, artifact)
    confirmed = (
        sparse["status"] == "PASS"
        and gate["status"] == "PASS"
        and gate["grounding_curator_gate_status"] == "CONFIRMED"
    )
    permitted_conditions = list(
        sparse["curator_candidate_condition_set"] if confirmed else ()
    )
    permitted = set(permitted_conditions)
    official_conditions: list[str] = []
    rows: list[dict[str, Any]] = []
    for condition, disposition in sparse["condition_dispositions"].items():
        construction_permitted = condition in permitted
        scq_entry = can_enter_official_scq(
            condition,
            curator_gate_validation=gate,
            condition_disposition=disposition,
            controlled_family_validation=controlled_family_validation,
        )
        official_permitted = construction_permitted and scq_entry["permitted"]
        if official_permitted:
            official_conditions.append(condition)
            reason = "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED"
        elif construction_permitted:
            reason = scq_entry["blockers"][0]
        elif confirmed:
            reason = f"CONDITION_DISPOSITION_{disposition}_BLOCKS_ROUTE"
        elif gate["grounding_curator_gate_status"] == "INVALID":
            reason = "INVALID_GROUNDING_CURATOR_ARTIFACT"
        else:
            reason = f"GROUNDING_CURATOR_{gate['grounding_curator_gate_status']}"
        rows.append({
            "condition": condition,
            "packet_disposition": disposition,
            "controlled_family_construction_route_permitted": construction_permitted,
            "official_scq_route_permitted": official_permitted,
            "official_scq_entry": scq_entry,
            "route_status": "PERMITTED" if official_permitted else "BLOCKED",
            "reason": reason,
        })

    return {
        "status": "PASS" if sparse["status"] == "PASS" and gate["status"] in {"PASS", "PENDING"} else "FAIL",
        "grounding_curator_gate_status": gate["grounding_curator_gate_status"],
        "curator_gate_validation": gate,
        "curator_candidate_condition_set": sparse["curator_candidate_condition_set"],
        "held_for_revision": sparse["held_for_revision"],
        "blocked_condition_set": sparse["blocked_condition_set"],
        "controlled_family_construction_candidate_set": permitted_conditions,
        "controlled_family_validation_status": (
            str(controlled_family_validation.get("status", "NOT_RUN"))
            if isinstance(controlled_family_validation, Mapping)
            else "NOT_RUN"
        ),
        "CONTROLLED_FAMILY_CONSTRUCTION_VALIDATED": bool(
            isinstance(controlled_family_validation, Mapping)
            and controlled_family_validation.get("status") == "PASS"
            and controlled_family_validation.get("controlled_family_construction_validated") is True
        ),
        "official_scq_route_permitted_condition_set": official_conditions,
        "condition_routes": rows,
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "official_scq_executed": False,
        "dataset_schema_mutated": False,
    }


def validate_gate_separation(
    grounding_gate: Mapping[str, Any] | None,
    evaluation_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate independent gate artifacts without allowing substitution."""

    def status(value: Mapping[str, Any] | None, expected_gate: str) -> str:
        if not isinstance(value, Mapping):
            return "PENDING"
        if str(value.get("gate", expected_gate)) != expected_gate:
            return "INVALID_GATE"
        return str(value.get("decision", "PENDING"))

    grounding = status(grounding_gate, "GROUNDING_CURATOR_CONFIRMED")
    evaluation = status(evaluation_gate, "EVALUATION_CONTRACT_CONFIRMED")
    return {
        "grounding_gate_status": grounding,
        "evaluation_contract_gate_status": evaluation,
        "gates_are_distinct": grounding != "INVALID_GATE" and evaluation != "INVALID_GATE",
        "grounding_does_not_confirm_evaluation": not (grounding == "CONFIRMED" and evaluation != "CONFIRMED") or evaluation != "CONFIRMED",
        "evaluation_does_not_confirm_grounding": not (evaluation == "CONFIRMED" and grounding != "CONFIRMED") or grounding != "CONFIRMED",
        "both_confirmed": grounding == "CONFIRMED" and evaluation == "CONFIRMED",
    }


def validate_lifecycle_order(states: list[str]) -> list[str]:
    order = {state.value: index for index, state in enumerate(LifecycleState)}
    errors: list[str] = []
    for previous, current in zip(states, states[1:]):
        if previous not in order or current not in order:
            errors.append("UNKNOWN_LIFECYCLE_STATE")
        elif order[current] < order[previous]:
            errors.append(f"LIFECYCLE_REGRESSION:{previous}->{current}")
    return errors


__all__ = [
    "CuratorGateArtifact",
    "GateDecision",
    "LifecycleState",
    "build_o3_revision_lifecycle",
    "build_sparse_post_curator_route",
    "can_enter_official_scq",
    "canonical_curator_packet_sha256",
    "canonical_target_review_sha256",
    "derive_sparse_curator_condition_set",
    "next_real_scientific_action",
    "o3_revision_candidate_sha256",
    "summarize_grounding_curator_state",
    "validate_target_approval_lineage",
    "validate_gate_separation",
    "validate_grounding_curator_gate_artifact",
    "validate_lifecycle_order",
]
