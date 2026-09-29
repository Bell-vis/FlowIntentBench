"""Curator review packets; generation never confirms a family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


CURATOR_DECISION_KEYS = (
    "question_meaningful",
    "observables_correctly_interpreted",
    "answerable_from_data",
    "operationalizations_defensible",
    "same_target_alternatives",
    "o2_dimensions_appropriate",
    "o3_target_fixed_and_visible",
    "finding_requirements_appropriate",
    "reference_findings_correct",
    "evidence_sources_sufficient",
    "ambiguities_disclosed",
    "recommendation",
)
CURATOR_RECOMMENDATIONS = frozenset({"CONFIRM", "REVISE", "REJECT", "MORE_EVIDENCE_REQUIRED"})
GROUNDING_PACKET_STATUS = "PENDING"


@dataclass(frozen=True)
class CuratorReviewPacket:
    family_id: str
    dataset_id: str
    concept_id: str
    decisions: Mapping[str, Any]
    curator_status: str = "PENDING_EXPERT_REVIEW"
    curator_decision: str | None = None
    curator_notes: str | None = None
    construction_recommendation: str = "MORE_EVIDENCE_REQUIRED"

    def __post_init__(self) -> None:
        missing = set(CURATOR_DECISION_KEYS) - set(self.decisions)
        if missing:
            raise ValueError(f"curator packet missing decisions: {sorted(missing)}")
        if self.curator_status != "PENDING_EXPERT_REVIEW":
            raise ValueError("generated curator packets must remain PENDING_EXPERT_REVIEW")
        if self.curator_decision is not None and self.curator_decision not in CURATOR_RECOMMENDATIONS:
            raise ValueError(f"invalid curator decision: {self.curator_decision}")
        if self.construction_recommendation not in {"READY_FOR_GROUNDING_CURATOR_REVIEW", "READY_FOR_EVALUATION_CONTRACT_REVIEW", "MORE_EVIDENCE_REQUIRED", "SCIENTIFICALLY_AMBIGUOUS", "REJECT"}:
            raise ValueError(f"invalid construction recommendation: {self.construction_recommendation}")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decisions"] = dict(self.decisions)
        return value


@dataclass(frozen=True)
class GroundingCuratorReviewPacket:
    """Independent grounding gate input.

    This packet deliberately uses ``PENDING`` and null decision fields.  The
    construction recommendation is evidence-derived input, not a curator
    decision and cannot promote a family in the lifecycle.
    """

    family_id: str
    dataset_id: str
    concept_id: str
    grounding_packet_sha256: str
    construction_recommendation: str
    curator_status: str = GROUNDING_PACKET_STATUS
    curator_decision: str | None = None
    curator_notes: str | None = None

    def __post_init__(self) -> None:
        if not self.family_id.strip() or not self.dataset_id.strip() or not self.concept_id.strip():
            raise ValueError("grounding curator packet identity is required")
        if self.curator_status != GROUNDING_PACKET_STATUS:
            raise ValueError("generated grounding curator packets must remain PENDING")
        if self.curator_decision is not None:
            raise ValueError("generated grounding curator packets cannot contain a decision")
        if self.curator_notes is not None:
            raise ValueError("generated grounding curator packets cannot contain curator notes")
        if not self.grounding_packet_sha256.strip():
            raise ValueError("grounding packet hash is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_curator_review_packets(packets: Sequence[Mapping[str, Any] | Any]) -> tuple[CuratorReviewPacket, ...]:
    result: list[CuratorReviewPacket] = []
    for packet in packets:
        value = packet.to_dict() if hasattr(packet, "to_dict") else dict(packet)
        recommendation = str(value.get("grounding_predicate", {}).get("recommendation") or value.get("grounding_recommendation", "MORE_EVIDENCE_REQUIRED"))
        decisions = {key: None for key in CURATOR_DECISION_KEYS}
        decisions["recommendation"] = None
        result.append(CuratorReviewPacket(
            family_id=str(value.get("family_id", "")),
            dataset_id=str(value.get("dataset_id", "")),
            concept_id=str(value.get("concept_id", "")),
            decisions=decisions,
            construction_recommendation=recommendation,
        ))
    return tuple(result)


def build_grounding_curator_packets(packets: Sequence[Mapping[str, Any] | Any]) -> tuple[GroundingCuratorReviewPacket, ...]:
    """Create hash-bound, pending grounding packets without preselection."""

    import hashlib
    import json

    result: list[GroundingCuratorReviewPacket] = []
    for packet in packets:
        value = packet.to_dict() if hasattr(packet, "to_dict") else dict(packet)
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        recommendation = str(value.get("construction_recommendation") or value.get("grounding_predicate", {}).get("recommendation", "MORE_EVIDENCE_REQUIRED"))
        result.append(GroundingCuratorReviewPacket(
            family_id=str(value.get("family_id", "")),
            dataset_id=str(value.get("dataset_id", "")),
            concept_id=str(value.get("concept_id", "")),
            grounding_packet_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            construction_recommendation=recommendation,
        ))
    return tuple(result)


__all__ = ["CURATOR_DECISION_KEYS", "CURATOR_RECOMMENDATIONS", "GROUNDING_PACKET_STATUS", "CuratorReviewPacket", "GroundingCuratorReviewPacket", "build_curator_review_packets", "build_grounding_curator_packets"]
