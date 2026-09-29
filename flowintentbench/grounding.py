"""Scientific grounding packets prepared for independent expert review."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .context import EvidenceRecord, SourceRecord


GROUNDING_RECOMMENDATIONS = frozenset({
    "READY_FOR_GROUNDING_CURATOR_REVIEW",
    "MORE_EVIDENCE_REQUIRED",
    "SCIENTIFICALLY_AMBIGUOUS",
    "REJECT",
})


EVIDENCE_SOURCE_TYPES = frozenset({
    "DATASET_DOCUMENTATION",
    "DATASET_PUBLICATION",
    "INDEPENDENT_SCIENTIFIC_REFERENCE",
    "DATA_FILE_METADATA",
    "CURATOR_NOTE",
})

SUPPORT_STATUSES = frozenset({"SUPPORTED", "UNSUPPORTED", "UNKNOWN"})
SUPPORT_RELATIONS = frozenset({"DIRECT", "CONTEXTUAL", "BACKGROUND"})
SUPPORT_CLAIM_TYPES = frozenset({
    "SCIENTIFIC_TARGET_VALIDITY", "OBSERVABLE_SEMANTICS", "DATASET_CONTEXT",
    "OPERATIONALIZATION_RATIONALE", "SAME_TARGET_RATIONALE", "FINDING_VERIFIABILITY",
})


class ScientificEvidenceBindingError(ValueError):
    """Raised when unbound evidence is requested for an expert-visible view."""

_SOURCE_TYPE_MAP = {
    "original_publication": "DATASET_PUBLICATION",
    "direct_analysis_publication": "DATASET_PUBLICATION",
    "dataset_documentation": "DATASET_DOCUMENTATION",
    "example_source_code": "DATASET_DOCUMENTATION",
    "relevant_scientific_reference": "INDEPENDENT_SCIENTIFIC_REFERENCE",
}


@dataclass(frozen=True)
class EvidenceProvenanceResolution:
    """One evidence record resolved through its declared source reference."""

    evidence_record_id: str
    source_id: str | None
    source_type: str | None
    title: str | None
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    external_identifier: str | None = None
    supported_claim_ids: tuple[str, ...] = ()
    resolution_status: str = "UNRESOLVED_SOURCE"
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authors"] = list(self.authors)
        # Family-scoped binding expresses support through ClaimSupportLink.
        # Preserve this historical compatibility field only when a caller
        # explicitly supplied claim IDs; never emit an empty parallel map.
        if self.supported_claim_ids:
            value["supported_claim_ids"] = list(self.supported_claim_ids)
        else:
            value.pop("supported_claim_ids", None)
        return value


def resolve_evidence_provenance(
    evidence_record_id: str,
    evidence_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    *,
    supported_claim_ids: Sequence[str] = (),
) -> EvidenceProvenanceResolution:
    """Resolve evidence -> declared source -> canonical source, fail closed."""

    evidence = next((item for item in evidence_records if str(item.get("evidence_id", "")) == str(evidence_record_id)), None)
    if not isinstance(evidence, Mapping):
        return EvidenceProvenanceResolution(str(evidence_record_id), None, None, None, supported_claim_ids=tuple(supported_claim_ids))
    declared_source_id = str(evidence.get("source_id", "")).strip()
    if not declared_source_id or declared_source_id in {str(evidence_record_id), "pending_source", "unknown_source"}:
        return EvidenceProvenanceResolution(str(evidence_record_id), declared_source_id or None, None, None, supported_claim_ids=tuple(supported_claim_ids), resolution_status="INVALID_SOURCE_RECORD")
    source = next((item for item in source_records if str(item.get("source_id", "")) == declared_source_id), None)
    if not isinstance(source, Mapping):
        return EvidenceProvenanceResolution(str(evidence_record_id), declared_source_id, None, None, supported_claim_ids=tuple(supported_claim_ids))
    source_type = _SOURCE_TYPE_MAP.get(str(source.get("source_type", "")).strip(), str(source.get("source_type", "")).strip() or None)
    external_identifier = source.get("doi") or source.get("report_id")
    return EvidenceProvenanceResolution(
        evidence_record_id=str(evidence_record_id), source_id=declared_source_id,
        source_type=source_type, title=str(source.get("title", "")) or None,
        venue=str(source.get("venue")) if source.get("venue") else None,
        authors=tuple(str(item) for item in source.get("authors", ()) or ()),
        year=source.get("year"), doi=str(source.get("doi")) if source.get("doi") else None,
        external_identifier=str(external_identifier) if external_identifier else None,
        supported_claim_ids=tuple(str(item) for item in supported_claim_ids), resolution_status="RESOLVED",
    )


def resolve_dataset_provenance(
    repository_root: str | Path,
    dataset_id: str,
    evidence_record_ids: Sequence[str],
    *,
    supported_claim_ids: Sequence[str] = (),
) -> tuple[EvidenceProvenanceResolution, ...]:
    """Load the repository's construction records and resolve all requested IDs."""

    root = Path(repository_root) / "datasets" / str(dataset_id) / "construction"
    evidence_payload = _read(root / "operationalization_evidence.json", [])
    context_payload = _read(root / "context_evidence.json", [])
    finding_payload = _read(root / "finding_evidence.json", [])
    evidence_records = [item for item in (evidence_payload if isinstance(evidence_payload, list) else []) + (context_payload if isinstance(context_payload, list) else []) + (finding_payload if isinstance(finding_payload, list) else []) if isinstance(item, Mapping)]
    source_payload = _read(root / "sources.json", {})
    source_records = source_payload.get("sources", []) if isinstance(source_payload, Mapping) else []
    return tuple(resolve_evidence_provenance(item, evidence_records, source_records, supported_claim_ids=supported_claim_ids) for item in evidence_record_ids)


def _family_claim_prefix(claim_id: str) -> str:
    """Return the authored family portion of a construction claim ID.

    Construction cards use ``<family_id>_<claim-kind>`` identifiers.  Claim
    kinds are deliberately kept as a small suffix set so this check does not
    infer scientific semantics from the claim statement itself.
    """

    suffixes = tuple(sorted(
        (
            "scientific_target_validity",
            "observable_semantics",
            "dataset_context",
            "operationalization_rationale",
            "same_target_rationale",
            "finding_verifiability",
            "same_target",
            "operationalization",
            "observable",
            "findings",
            "context",
            "target",
        ),
        key=len,
        reverse=True,
    ))
    value = str(claim_id).strip()
    for suffix in suffixes:
        marker = "_" + suffix
        if value.endswith(marker):
            return value[: -len(marker)]
    return value.rsplit("_", 1)[0] if "_" in value else value


def resolve_family_scientific_evidence(
    dataset_id: str,
    family_id: str,
    concept_id: str,
    source_records: Sequence[Mapping[str, Any]] = (),
    evidence_records: Sequence[Mapping[str, Any]] = (),
    scientific_support_claims: Sequence[Mapping[str, Any] | ScientificSupportClaim] = (),
    claim_support_links: Sequence[Mapping[str, Any] | ClaimSupportLink] = (),
    *,
    evidence_family_id: str | None = None,
    evidence_concept_id: str | None = None,
) -> FamilyScientificEvidenceResolution:
    """Resolve a family/concept-scoped evidence bundle through frozen IDs.

    This is a binding/audit function, not a new evidence model.  Input and
    successful output rows are the existing ``SourceRecord``,
    ``EvidenceRecord``, ``ScientificSupportClaim`` and ``ClaimSupportLink``
    contracts.  It fails closed when a claim is authored for another family,
    when dataset identity differs, or when any claim link cannot close through
    ``claim -> evidence -> source``.
    """

    requested_dataset = str(dataset_id).strip()
    requested_family = str(family_id).strip()
    requested_concept = str(concept_id).strip()
    if not requested_dataset or not requested_family or not requested_concept:
        return FamilyScientificEvidenceResolution.from_mapping({
            "status": "FAIL",
            "reason_code": "EVIDENCE_FAMILY_MISMATCH",
            "dataset_id": requested_dataset,
            "family_id": requested_family,
            "concept_id": requested_concept,
            "bound_family_id": str(evidence_family_id or ""),
            "bound_concept_id": str(evidence_concept_id or ""),
            "source_records": [],
            "evidence_records": [],
            "scientific_support_claims": [],
            "claim_support_links": [],
            "provenance_resolutions": [],
            "unresolved_claim_ids": [],
        })

    # Validate and normalize only the frozen models. Pydantic's extra=forbid
    # catches URLs and any accidental parallel schema fields at this boundary.
    sources: list[SourceRecord] = []
    evidence: list[EvidenceRecord] = []
    claims: list[ScientificSupportClaim] = []
    links: list[ClaimSupportLink] = []
    try:
        # Accept the existing SourceCollection JSON envelope as a convenience,
        # while retaining the same frozen rows internally. Other collections
        # may likewise be supplied as their existing named list envelope.
        if isinstance(source_records, Mapping) and "sources" in source_records:
            source_records = source_records["sources"]
        if isinstance(evidence_records, Mapping) and "evidence_records" in evidence_records:
            evidence_records = evidence_records["evidence_records"]
        if isinstance(scientific_support_claims, Mapping) and "scientific_support_claims" in scientific_support_claims:
            scientific_support_claims = scientific_support_claims["scientific_support_claims"]
        if isinstance(claim_support_links, Mapping) and "claim_support_links" in claim_support_links:
            claim_support_links = claim_support_links["claim_support_links"]
        sources = [
            item if isinstance(item, SourceRecord) else SourceRecord.model_validate(item)
            for item in source_records
        ]
        evidence = [
            item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
            for item in evidence_records
        ]
        claim_fields = {
            "claim_id",
            "claim_type",
            "statement",
            "support_status",
            "supporting_source_ids",
            "supporting_evidence_record_ids",
            "critical",
        }
        link_fields = {
            "claim_id",
            "evidence_record_id",
            "source_id",
            "support_relation",
            "supported_statement",
        }
        for item in scientific_support_claims:
            if isinstance(item, Mapping) and not set(item).issubset(claim_fields):
                raise ValueError(
                    "ScientificSupportClaim fields must match the frozen contract"
                )
        for item in claim_support_links:
            if isinstance(item, Mapping) and set(item) != link_fields:
                raise ValueError("ClaimSupportLink fields must match the frozen contract")
        claims = [
            item if isinstance(item, ScientificSupportClaim) else ScientificSupportClaim.from_mapping(item)
            for item in scientific_support_claims
        ]
        links = [
            item if isinstance(item, ClaimSupportLink) else ClaimSupportLink.from_mapping(item)
            for item in claim_support_links
        ]
    except (TypeError, ValueError, KeyError) as exc:
        return FamilyScientificEvidenceResolution.from_mapping({
            "status": "FAIL",
            "reason_code": "EVIDENCE_PROVENANCE_UNRESOLVED",
            "dataset_id": requested_dataset,
            "family_id": requested_family,
            "concept_id": requested_concept,
            "bound_family_id": str(evidence_family_id or ""),
            "bound_concept_id": str(evidence_concept_id or ""),
            "source_records": [],
            "evidence_records": [],
            "scientific_support_claims": [],
            "claim_support_links": [],
            "provenance_resolutions": [],
            "unresolved_claim_ids": [],
            "error": f"{type(exc).__name__}: {exc}",
        })

    source_by_id: dict[str, SourceRecord] = {}
    duplicate_source_ids: list[str] = []
    for source in sources:
        if source.source_id in source_by_id:
            duplicate_source_ids.append(source.source_id)
        source_by_id[source.source_id] = source
    evidence_by_id: dict[str, EvidenceRecord] = {}
    duplicate_evidence_ids: list[str] = []
    for record in evidence:
        if record.evidence_id in evidence_by_id:
            duplicate_evidence_ids.append(record.evidence_id)
        evidence_by_id[record.evidence_id] = record
    claim_by_id: dict[str, ScientificSupportClaim] = {}
    duplicate_claim_ids: list[str] = []
    for claim in claims:
        if claim.claim_id in claim_by_id:
            duplicate_claim_ids.append(claim.claim_id)
        claim_by_id[claim.claim_id] = claim

    claim_prefixes = {_family_claim_prefix(claim.claim_id) for claim in claims}
    bound_family = str(evidence_family_id or (next(iter(claim_prefixes)) if len(claim_prefixes) == 1 else "")).strip()
    bound_concept = str(evidence_concept_id or bound_family).strip()
    mismatch_claims = [
        claim.claim_id
        for claim in claims
        if _family_claim_prefix(claim.claim_id) != bound_family
    ]
    dataset_mismatch_evidence = [
        record.evidence_id
        for record in evidence
        if record.dataset_id != requested_dataset
    ]
    link_errors: list[dict[str, str]] = []
    for link in links:
        record = evidence_by_id.get(link.evidence_record_id)
        source = source_by_id.get(link.source_id)
        if link.claim_id not in claim_by_id:
            link_errors.append({"claim_support_link": link.claim_id, "reason": "CLAIM_NOT_FOUND"})
        if record is None:
            link_errors.append({"claim_support_link": link.claim_id, "reason": "EVIDENCE_NOT_FOUND", "evidence_id": link.evidence_record_id})
        if source is None:
            link_errors.append({"claim_support_link": link.claim_id, "reason": "SOURCE_NOT_FOUND", "source_id": link.source_id})
        if record is not None and record.source_id != link.source_id:
            link_errors.append({"claim_support_link": link.claim_id, "reason": "EVIDENCE_SOURCE_MISMATCH", "evidence_id": link.evidence_record_id, "source_id": link.source_id})

    unresolved_claims: list[str] = []
    for claim in claims:
        expected_evidence = set(claim.supporting_evidence_record_ids)
        expected_sources = set(claim.supporting_source_ids)
        claim_links = [link for link in links if link.claim_id == claim.claim_id]
        linked_evidence = {link.evidence_record_id for link in claim_links}
        linked_sources = {link.source_id for link in claim_links}
        if not expected_evidence or not expected_sources:
            unresolved_claims.append(claim.claim_id)
            continue
        if not expected_evidence.issubset(linked_evidence) or not expected_sources.issubset(linked_sources):
            unresolved_claims.append(claim.claim_id)
            continue
        if any(item not in evidence_by_id for item in expected_evidence) or any(item not in source_by_id for item in expected_sources):
            unresolved_claims.append(claim.claim_id)

    identity_mismatch = requested_family != bound_family or requested_concept != bound_concept
    mismatch = identity_mismatch or bool(mismatch_claims)
    unresolved = bool(
        not sources
        or not evidence
        or not claims
        or not links
        or duplicate_source_ids
        or duplicate_evidence_ids
        or duplicate_claim_ids
        or dataset_mismatch_evidence
        or link_errors
        or unresolved_claims
    )
    resolutions = [
        resolve_evidence_provenance(
            record.evidence_id,
            [item.model_dump(mode="json") for item in evidence],
            [item.model_dump(mode="json") for item in sources],
        ).to_dict()
        for record in evidence
    ]
    if mismatch:
        reason_code = "EVIDENCE_FAMILY_MISMATCH"
    elif unresolved:
        reason_code = "EVIDENCE_PROVENANCE_UNRESOLVED"
    else:
        reason_code = "NONE"
    return FamilyScientificEvidenceResolution.from_mapping({
        "status": "PASS" if not mismatch and not unresolved else "FAIL",
        "reason_code": reason_code,
        "dataset_id": requested_dataset,
        "family_id": requested_family,
        "concept_id": requested_concept,
        "bound_family_id": bound_family,
        "bound_concept_id": bound_concept,
        "source_records": [item.model_dump(mode="json") for item in sources],
        "evidence_records": [item.model_dump(mode="json") for item in evidence],
        "scientific_support_claims": [item.to_dict() for item in claims],
        "claim_support_links": [item.to_dict() for item in links],
        "provenance_resolutions": resolutions,
        "unresolved_claim_ids": sorted(set(unresolved_claims)),
        "mismatch_claim_ids": sorted(set(mismatch_claims)),
        "dataset_mismatch_evidence_ids": sorted(set(dataset_mismatch_evidence)),
        "link_errors": link_errors,
        "duplicate_source_ids": sorted(set(duplicate_source_ids)),
        "duplicate_evidence_ids": sorted(set(duplicate_evidence_ids)),
        "duplicate_claim_ids": sorted(set(duplicate_claim_ids)),
        "family_scope_verified": not mismatch,
        "provenance_closure_verified": not unresolved,
    })


@dataclass(frozen=True)
class EvidenceReference:
    source_id: str
    source_type: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    url_or_identifier: str | None = None
    dataset_specific: bool = False
    supports_claims: tuple[str, ...] = ()
    evidence_location: str | None = None
    verification_status: str = "RESOLVED"
    venue: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.title.strip():
            raise ValueError("EvidenceReference source_id and title are required")
        if self.source_type not in EVIDENCE_SOURCE_TYPES:
            raise ValueError(f"invalid evidence source_type: {self.source_type}")
        if self.verification_status not in {"RESOLVED", "PENDING", "UNRESOLVED"}:
            raise ValueError("invalid evidence verification_status")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authors"] = list(self.authors)
        value["supports_claims"] = list(self.supports_claims)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceReference":
        payload = dict(value)
        payload["authors"] = tuple(str(x) for x in payload.get("authors", ()) or ())
        payload["supports_claims"] = tuple(str(x) for x in payload.get("supports_claims", ()) or ())
        return cls(**payload)


@dataclass(frozen=True)
class ScientificClaimProvenance:
    claim_id: str
    claim: str
    supporting_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.claim.strip() or not self.supporting_source_ids:
            raise ValueError("critical scientific claims require source IDs")

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "claim": self.claim, "supporting_source_ids": list(self.supporting_source_ids)}


@dataclass(frozen=True)
class ScientificSupportClaim:
    """Explicit, reviewable scientific support assertion.

    The statement is explanatory text only; support is established by the
    structured status and its evidence/source references.  In particular, a
    non-empty statement never implies support.
    """

    claim_id: str
    claim_type: str
    statement: str
    support_status: str = "UNKNOWN"
    supporting_source_ids: tuple[str, ...] = ()
    supporting_evidence_record_ids: tuple[str, ...] = ()
    critical: bool = True

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.claim_type.strip():
            raise ValueError("ScientificSupportClaim requires claim_id and claim_type")
        if self.support_status not in SUPPORT_STATUSES:
            raise ValueError(f"invalid support_status: {self.support_status}")
        if self.claim_type not in SUPPORT_CLAIM_TYPES:
            raise ValueError(f"invalid claim_type: {self.claim_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "statement": self.statement,
            "support_status": self.support_status,
            "supporting_source_ids": list(self.supporting_source_ids),
            "supporting_evidence_record_ids": list(self.supporting_evidence_record_ids),
            "critical": bool(self.critical),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScientificSupportClaim":
        return cls(
            claim_id=str(value.get("claim_id", "")),
            claim_type=str(value.get("claim_type", "")).upper(),
            statement=str(value.get("statement", value.get("claim", ""))),
            support_status=str(value.get("support_status", "UNKNOWN")).upper(),
            supporting_source_ids=tuple(str(x) for x in value.get("supporting_source_ids", ()) or ()),
            supporting_evidence_record_ids=tuple(str(x) for x in value.get("supporting_evidence_record_ids", value.get("evidence_record_ids", ())) or ()),
            critical=bool(value.get("critical", True)),
        )


@dataclass(frozen=True)
class ClaimSupportLink:
    """One claim -> evidence record -> canonical source support relation."""

    claim_id: str
    evidence_record_id: str
    source_id: str
    support_relation: str
    supported_statement: str

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.evidence_record_id.strip() or not self.source_id.strip():
            raise ValueError("ClaimSupportLink identifiers are required")
        if self.support_relation not in SUPPORT_RELATIONS:
            raise ValueError(f"invalid support_relation: {self.support_relation}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "evidence_record_id": self.evidence_record_id,
            "source_id": self.source_id,
            "support_relation": self.support_relation,
            "supported_statement": self.supported_statement,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimSupportLink":
        return cls(
            claim_id=str(value.get("claim_id", "")),
            evidence_record_id=str(value.get("evidence_record_id", value.get("evidence_id", ""))),
            source_id=str(value.get("source_id", "")),
            support_relation=str(value.get("support_relation", "BACKGROUND")).upper(),
            supported_statement=str(value.get("supported_statement", "")),
        )


@dataclass(frozen=True)
class FamilyScientificEvidenceResolution(Mapping[str, Any]):
    """Typed result of family-scoped evidence binding and provenance closure."""

    dataset_id: str
    family_id: str
    concept_id: str
    bound_family_id: str
    bound_concept_id: str
    status: str
    reason_code: str
    source_records: tuple[SourceRecord, ...] = ()
    evidence_records: tuple[EvidenceRecord, ...] = ()
    scientific_support_claims: tuple[ScientificSupportClaim, ...] = ()
    claim_support_links: tuple[ClaimSupportLink, ...] = ()
    provenance_resolutions: tuple[EvidenceProvenanceResolution, ...] = ()
    unresolved_claim_ids: tuple[str, ...] = ()
    mismatch_claim_ids: tuple[str, ...] = ()
    dataset_mismatch_evidence_ids: tuple[str, ...] = ()
    link_errors: tuple[Mapping[str, str], ...] = ()
    duplicate_source_ids: tuple[str, ...] = ()
    duplicate_evidence_ids: tuple[str, ...] = ()
    duplicate_claim_ids: tuple[str, ...] = ()
    error: str | None = None

    @property
    def family_binding_status(self) -> str:
        return "PASS" if self.reason_code != "EVIDENCE_FAMILY_MISMATCH" else "EVIDENCE_FAMILY_MISMATCH"

    @property
    def provenance_status(self) -> str:
        return "PASS" if self.reason_code == "NONE" else (
            "EVIDENCE_PROVENANCE_UNRESOLVED"
            if self.reason_code == "EVIDENCE_PROVENANCE_UNRESOLVED"
            else "NOT_EVALUATED_FAMILY_MISMATCH"
        )

    @property
    def expert_review_permitted(self) -> bool:
        return self.status == "PASS"

    def require_expert_evidence(self) -> dict[str, Any]:
        """Return only frozen expert-visible rows, or fail before invocation."""

        if not self.expert_review_permitted:
            raise ScientificEvidenceBindingError(
                f"{self.reason_code}: evidence is not bound to "
                f"{self.dataset_id}/{self.family_id}/{self.concept_id}"
            )
        return {
            "source_records": [item.model_dump(mode="json") for item in self.source_records],
            "evidence_records": [item.model_dump(mode="json") for item in self.evidence_records],
            "scientific_support_claims": [item.to_dict() for item in self.scientific_support_claims],
            "claim_support_links": [item.to_dict() for item in self.claim_support_links],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "dataset_id": self.dataset_id,
            "family_id": self.family_id,
            "concept_id": self.concept_id,
            "bound_family_id": self.bound_family_id,
            "bound_concept_id": self.bound_concept_id,
            "family_binding_status": self.family_binding_status,
            "provenance_status": self.provenance_status,
            "expert_review_permitted": self.expert_review_permitted,
            "source_records": [item.model_dump(mode="json") for item in self.source_records],
            "evidence_records": [item.model_dump(mode="json") for item in self.evidence_records],
            "scientific_support_claims": [item.to_dict() for item in self.scientific_support_claims],
            "claim_support_links": [item.to_dict() for item in self.claim_support_links],
            "provenance_resolutions": [item.to_dict() for item in self.provenance_resolutions],
            "unresolved_claim_ids": list(self.unresolved_claim_ids),
            "mismatch_claim_ids": list(self.mismatch_claim_ids),
            "dataset_mismatch_evidence_ids": list(self.dataset_mismatch_evidence_ids),
            "link_errors": [dict(item) for item in self.link_errors],
            "duplicate_source_ids": list(self.duplicate_source_ids),
            "duplicate_evidence_ids": list(self.duplicate_evidence_ids),
            "duplicate_claim_ids": list(self.duplicate_claim_ids),
            "error": self.error,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FamilyScientificEvidenceResolution":
        return cls(
            dataset_id=str(value.get("dataset_id", "")),
            family_id=str(value.get("family_id", "")),
            concept_id=str(value.get("concept_id", "")),
            bound_family_id=str(value.get("bound_family_id", "")),
            bound_concept_id=str(value.get("bound_concept_id", "")),
            status=str(value.get("status", "FAIL")),
            reason_code=str(value.get("reason_code", "EVIDENCE_PROVENANCE_UNRESOLVED")),
            source_records=tuple(SourceRecord.model_validate(item) for item in value.get("source_records", ()) or ()),
            evidence_records=tuple(EvidenceRecord.model_validate(item) for item in value.get("evidence_records", ()) or ()),
            scientific_support_claims=tuple(ScientificSupportClaim.from_mapping(item) for item in value.get("scientific_support_claims", ()) or ()),
            claim_support_links=tuple(ClaimSupportLink.from_mapping(item) for item in value.get("claim_support_links", ()) or ()),
            provenance_resolutions=tuple(EvidenceProvenanceResolution(**item) for item in value.get("provenance_resolutions", value.get("resolutions", ())) or ()),
            unresolved_claim_ids=tuple(str(item) for item in value.get("unresolved_claim_ids", ()) or ()),
            mismatch_claim_ids=tuple(str(item) for item in value.get("mismatch_claim_ids", ()) or ()),
            dataset_mismatch_evidence_ids=tuple(str(item) for item in value.get("dataset_mismatch_evidence_ids", ()) or ()),
            link_errors=tuple(dict(item) for item in value.get("link_errors", ()) or () if isinstance(item, Mapping)),
            duplicate_source_ids=tuple(str(item) for item in value.get("duplicate_source_ids", ()) or ()),
            duplicate_evidence_ids=tuple(str(item) for item in value.get("duplicate_evidence_ids", ()) or ()),
            duplicate_claim_ids=tuple(str(item) for item in value.get("duplicate_claim_ids", ()) or ()),
            error=str(value.get("error")) if value.get("error") else None,
        )


@dataclass(frozen=True)
class GroundingReadinessInput:
    capability_requirements_satisfied: bool = False
    scientific_target_support: bool = False
    observable_semantics_support: bool = False
    dataset_context_support: bool = False
    operationalization_plurality_support: bool = False
    alternative_o_same_target_support: bool = False
    finding_verifiability_support: bool = False
    evidence_provenance_complete: bool = False
    critical_scientific_ambiguities: tuple[Any, ...] = ()
    fundamental_data_target_mismatch: bool = False
    resolvable_evidence_gaps: tuple[Any, ...] = ()
    support_claims: tuple[Any, ...] = ()
    scientific_support_claims: tuple[Any, ...] = ()
    claim_support_links: tuple[Any, ...] = ()
    resolved_source_ids: tuple[str, ...] = ()


def _normalize_issue(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        issue = dict(value)
        issue.setdefault("issue_id", "issue-unknown")
        issue.setdefault("issue_type", "EVIDENCE_GAP")
        issue.setdefault("related_claim_ids", list(issue.get("claim_id", "") and [issue["claim_id"]] or []))
        issue.setdefault("description", "")
        issue.setdefault("severity", "CRITICAL")
        issue.setdefault("blocking", True)
        issue.setdefault("resolution_requirement", "")
        explicit = str(issue.get("issue_resolution_class", "")).upper()
        issue["issue_resolution_class"] = explicit if explicit in {"EVIDENCE_GAP", "SCIENTIFIC_AMBIGUITY", "FUNDAMENTAL_INVALIDITY"} else _issue_class_for_type(str(issue.get("issue_type", "")))
        return issue
    return {
        "issue_id": "issue-legacy",
        "issue_type": "EVIDENCE_GAP",
        "related_claim_ids": [],
        "description": str(value),
        "severity": "CRITICAL",
        "blocking": True,
        "resolution_requirement": "provide supporting evidence",
        "issue_resolution_class": "EVIDENCE_GAP",
    }


def _issue_class_for_type(issue_type: str) -> str:
    """Map authored issue types to exactly one resolution responsibility."""

    return {
        "DATA_TARGET_MISMATCH": "FUNDAMENTAL_INVALIDITY",
        "FUNDAMENTAL_INVALIDITY": "FUNDAMENTAL_INVALIDITY",
        "SCIENTIFIC_AMBIGUITY": "SCIENTIFIC_AMBIGUITY",
        "SAME_TARGET_RATIONALE_UNRESOLVED": "SCIENTIFIC_AMBIGUITY",
        "FIELD_COMPARABILITY_UNRESOLVED": "SCIENTIFIC_AMBIGUITY",
        "OBSERVABLE_SEMANTICS_AMBIGUOUS": "SCIENTIFIC_AMBIGUITY",
    }.get(issue_type.upper(), "EVIDENCE_GAP")


def _normalize_claims(values: Sequence[Any]) -> list[ScientificSupportClaim]:
    if isinstance(values, Mapping):
        values = (values,)
    claims: list[ScientificSupportClaim] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, ScientificSupportClaim) and isinstance(value, Mapping):
            value = ScientificSupportClaim.from_mapping(value)
        if isinstance(value, ScientificSupportClaim) and value.claim_id not in seen:
            claims.append(value)
            seen.add(value.claim_id)
    return claims


def _normalize_links(values: Sequence[Any]) -> list[ClaimSupportLink]:
    if isinstance(values, Mapping):
        values = (values,)
    links: list[ClaimSupportLink] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, ClaimSupportLink) and isinstance(value, Mapping):
            value = ClaimSupportLink.from_mapping(value)
        if isinstance(value, ClaimSupportLink):
            key = (value.claim_id, value.evidence_record_id, value.source_id)
            if key not in seen:
                links.append(value)
                seen.add(key)
    return links


_CLAIM_FIELD_MAP = {
    "SCIENTIFIC_TARGET_VALIDITY": "scientific_target_support",
    "OBSERVABLE_SEMANTICS": "observable_semantics_support",
    "DATASET_CONTEXT": "dataset_context_support",
    "OPERATIONALIZATION_RATIONALE": "operationalization_plurality_support",
    "SAME_TARGET_RATIONALE": "alternative_o_same_target_support",
    "FINDING_VERIFIABILITY": "finding_verifiability_support",
}


def _issue_descriptions(values: Sequence[Any]) -> list[str]:
    return [str(_normalize_issue(item).get("description", "")).strip() for item in values if str(_normalize_issue(item).get("description", "")).strip()]


def evaluate_grounding_readiness(value: GroundingReadinessInput | Mapping[str, Any]) -> dict[str, Any]:
    """Family-agnostic construction recommendation predicate."""

    if isinstance(value, GroundingReadinessInput):
        inputs = value
    else:
        inputs = GroundingReadinessInput(
            capability_requirements_satisfied=bool(value.get("capability_requirements_satisfied", False)),
            scientific_target_support=bool(value.get("scientific_target_support", False)),
            observable_semantics_support=bool(value.get("observable_semantics_support", False)),
            dataset_context_support=bool(value.get("dataset_context_support", False)),
            operationalization_plurality_support=bool(value.get("operationalization_plurality_support", False)),
            alternative_o_same_target_support=bool(value.get("alternative_o_same_target_support", value.get("alternative_O_same_target_support", False))),
            finding_verifiability_support=bool(value.get("finding_verifiability_support", False)),
            evidence_provenance_complete=bool(value.get("evidence_provenance_complete", False)),
            critical_scientific_ambiguities=tuple(value.get("critical_scientific_ambiguities", ()) or ()),
            fundamental_data_target_mismatch=bool(value.get("fundamental_data_target_mismatch", False)),
            resolvable_evidence_gaps=tuple(value.get("resolvable_evidence_gaps", ()) or ()),
            support_claims=tuple(value.get("support_claims", value.get("scientific_support_claims", ())) or ()),
            claim_support_links=tuple(value.get("claim_support_links", ()) or ()),
            resolved_source_ids=tuple(str(x) for x in value.get("resolved_source_ids", ()) or ()),
        )
    claims = _normalize_claims(inputs.support_claims or inputs.scientific_support_claims)
    links = _normalize_links(inputs.claim_support_links)
    fields = {
        key: bool(getattr(inputs, key)) for key in (
            "capability_requirements_satisfied", "scientific_target_support", "observable_semantics_support",
            "dataset_context_support", "operationalization_plurality_support", "alternative_o_same_target_support",
            "finding_verifiability_support", "evidence_provenance_complete",
        )
    }
    # Once explicit claims are present they are the sole source of scientific
    # support state.  Legacy booleans remain accepted only when no claims were
    # supplied, preserving compatibility with historical fixtures.
    if claims:
        by_type: dict[str, list[ScientificSupportClaim]] = {}
        for claim in claims:
            by_type.setdefault(claim.claim_type, []).append(claim)
        for claim_type, field_name in _CLAIM_FIELD_MAP.items():
            relevant = by_type.get(claim_type, [])
            fields[field_name] = bool(relevant) and all(
                claim.support_status == "SUPPORTED"
                and bool(claim.supporting_source_ids)
                for claim in relevant if claim.critical
            )
            if not any(claim.critical for claim in relevant):
                fields[field_name] = bool(relevant) and any(claim.support_status == "SUPPORTED" and claim.supporting_source_ids for claim in relevant)
        critical_claims = [claim for claim in claims if claim.critical]
        resolved_ids = set(inputs.resolved_source_ids)
        provenance_complete = True
        for claim in critical_claims:
            if claim.support_status != "SUPPORTED":
                provenance_complete = False
                continue
            matching = [link for link in links if link.claim_id == claim.claim_id and link.source_id in set(claim.supporting_source_ids) and (not claim.supporting_evidence_record_ids or link.evidence_record_id in set(claim.supporting_evidence_record_ids))]
            if not matching or (resolved_ids and not all(link.source_id in resolved_ids for link in matching)):
                provenance_complete = False
        fields["evidence_provenance_complete"] = provenance_complete
    # Preserve the frozen contract spelling while retaining the Pythonic
    # lowercase alias used by older artifacts.
    fields["alternative_O_same_target_support"] = fields["alternative_o_same_target_support"]
    all_issues = [_normalize_issue(item) for item in inputs.critical_scientific_ambiguities] + [_normalize_issue(item) for item in inputs.resolvable_evidence_gaps]
    # One issue ID has one primary resolution class.  Duplicated records are
    # normalized into the class explicitly authored (or deterministically
    # mapped from issue_type), never emitted in two blocker buckets.
    issues_by_id: dict[str, dict[str, Any]] = {}
    for issue in all_issues:
        issue_id = str(issue.get("issue_id", "issue-unknown"))
        existing = issues_by_id.get(issue_id)
        if existing is None:
            issues_by_id[issue_id] = issue
        elif existing.get("issue_resolution_class") != issue.get("issue_resolution_class"):
            # Conflicting authoring is itself an ambiguity, with one stable
            # record retained for deterministic downstream adjudication.
            existing["issue_resolution_class"] = "SCIENTIFIC_AMBIGUITY"
            existing["issue_type"] = "SCIENTIFIC_AMBIGUITY"
    normalized_issues = list(issues_by_id.values())
    critical_issues = [item for item in normalized_issues if item.get("issue_resolution_class") == "SCIENTIFIC_AMBIGUITY"]
    gap_issues = [item for item in normalized_issues if item.get("issue_resolution_class") == "EVIDENCE_GAP"]
    invalid_issues = [item for item in normalized_issues if item.get("issue_resolution_class") == "FUNDAMENTAL_INVALIDITY"]
    blocking_critical = [item for item in critical_issues if bool(item.get("blocking", True))]
    if inputs.fundamental_data_target_mismatch or any(bool(item.get("blocking", True)) for item in invalid_issues):
        recommendation = "REJECT"
    elif blocking_critical:
        recommendation = "SCIENTIFICALLY_AMBIGUOUS"
    elif gap_issues or not all(fields.values()):
        recommendation = "MORE_EVIDENCE_REQUIRED"
    else:
        recommendation = "READY_FOR_GROUNDING_CURATOR_REVIEW"
    blockers = [key for key, satisfied in fields.items() if not satisfied]
    blockers.extend(issue.get("issue_id") for issue in gap_issues)
    blockers.extend(issue.get("issue_id") for issue in blocking_critical)
    blockers.extend(issue.get("issue_id") for issue in invalid_issues if issue.get("blocking", True))
    if inputs.fundamental_data_target_mismatch:
        blockers.append("fundamental_data_target_mismatch")
    return {
        "recommendation": recommendation,
        "inputs": fields,
        "critical_scientific_ambiguities": critical_issues,
        "resolvable_evidence_gaps": gap_issues,
        "fundamental_invalidity_issues": invalid_issues,
        "scientific_support_claims": [claim.to_dict() for claim in claims],
        "claim_support_links": [link.to_dict() for link in links],
        "critical_scientific_issue_descriptions": _issue_descriptions(critical_issues),
        "resolvable_evidence_gap_descriptions": _issue_descriptions(gap_issues),
        "blockers": list(dict.fromkeys(blockers)),
        "family_id_used": False,
        "sets_grounding_curator_confirmed": False,
    }


@dataclass(frozen=True)
class ScientificGroundingPacket:
    family_id: str
    dataset_id: str
    concept_id: str
    scientific_question: str
    scientific_target: str
    scientific_entity_type: str
    analysis_archetype: str
    scientific_motivation: str
    dataset_context: str
    observable_support: str
    geometry_or_boundary_support: str
    operationalization_options: tuple[str, ...]
    operationalization_rationales: tuple[str, ...]
    operationalization_consequentiality: str
    reference_findings: tuple[Mapping[str, Any], ...]
    finding_requirement_contract: Mapping[str, Any]
    independent_scientific_sources: tuple[str, ...]
    dataset_specific_sources: tuple[str, ...]
    evidence_provenance: Mapping[str, Any]
    known_limitations: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    curator_questions: tuple[str, ...]
    grounding_recommendation: str
    curator_status: str = "PENDING_EXPERT_REVIEW"
    lifecycle_status: str = "DATA_SUPPORTED"
    evidence_references: tuple[Mapping[str, Any], ...] = ()
    claim_provenance: tuple[Mapping[str, Any], ...] = ()
    scientific_support_claims: tuple[Mapping[str, Any], ...] = ()
    claim_support_links: tuple[Mapping[str, Any], ...] = ()
    grounding_predicate: Mapping[str, Any] = field(default_factory=dict)
    construction_recommendation: str = "MORE_EVIDENCE_REQUIRED"

    def __post_init__(self) -> None:
        for name in ("family_id", "dataset_id", "concept_id", "scientific_question", "scientific_target", "scientific_entity_type", "analysis_archetype"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.grounding_recommendation not in GROUNDING_RECOMMENDATIONS:
            raise ValueError(f"invalid grounding recommendation: {self.grounding_recommendation}")
        if self.grounding_recommendation in {"REJECT", "MORE_EVIDENCE_REQUIRED"} and not self.unresolved_questions:
            raise ValueError("a blocked grounding packet must disclose unresolved questions")
        if self.curator_status == "CONFIRMED" or self.lifecycle_status in {"SCIENTIFICALLY_GROUNDED", "CURATOR_CONFIRMED", "RELEASE_ELIGIBLE"}:
            raise ValueError("generated grounding packets cannot self-certify grounding or curator confirmation")
        if not self.independent_scientific_sources or not self.dataset_specific_sources:
            raise ValueError("grounding packet requires independent and dataset-specific sources")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("operationalization_options", "operationalization_rationales", "known_limitations", "unresolved_questions", "curator_questions", "independent_scientific_sources", "dataset_specific_sources"):
            value[name] = list(getattr(self, name))
        value["reference_findings"] = [dict(item) for item in self.reference_findings]
        value["evidence_references"] = [dict(item) for item in self.evidence_references]
        value["claim_provenance"] = [dict(item) for item in self.claim_provenance]
        value["scientific_support_claims"] = [dict(item) for item in self.scientific_support_claims]
        value["claim_support_links"] = [dict(item) for item in self.claim_support_links]
        value["grounding_predicate"] = dict(self.grounding_predicate)
        value["construction_recommendation"] = self.construction_recommendation
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScientificGroundingPacket":
        payload = dict(value)
        for name in ("operationalization_options", "operationalization_rationales", "known_limitations", "unresolved_questions", "curator_questions", "independent_scientific_sources", "dataset_specific_sources"):
            payload[name] = tuple(str(item) for item in payload.get(name, ()) or ())
        payload["reference_findings"] = tuple(item for item in payload.get("reference_findings", ()) or () if isinstance(item, Mapping))
        payload["evidence_references"] = tuple(item for item in payload.get("evidence_references", ()) or () if isinstance(item, Mapping))
        payload["claim_provenance"] = tuple(item for item in payload.get("claim_provenance", ()) or () if isinstance(item, Mapping))
        payload["scientific_support_claims"] = tuple(item for item in payload.get("scientific_support_claims", ()) or () if isinstance(item, Mapping))
        payload["claim_support_links"] = tuple(item for item in payload.get("claim_support_links", ()) or () if isinstance(item, Mapping))
        return cls(**payload)


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def build_grounding_packets(repository_root: str | Path) -> tuple[ScientificGroundingPacket, ...]:
    root = Path(repository_root)
    cards_root = root / "artifacts/reference/concept_expansion_phase1/concept_design_cards"
    inventory = _read(
        root / "artifacts/reference/scientific_portfolio/concept_family_inventory.json",
        {},
    )
    families = {str(item.get("family_id")): item for item in inventory.get("families", []) if isinstance(item, Mapping)}
    # Capability status is supplied by the requirement-aware map.  Grounding
    # packet construction consumes this frozen result; it does not infer a
    # family decision from an identifier.
    try:
        from .concept_capability import build_dataset_concept_capability_map
        capability_cells = {
            (cell.dataset_id, cell.concept_id): cell
            for cell in build_dataset_concept_capability_map(root).cells
        }
    except Exception:
        capability_cells = {}
    packets: list[ScientificGroundingPacket] = []
    for path in sorted(cards_root.glob("*.json")):
        card = _read(path, {})
        if not isinstance(card, Mapping):
            continue
        concept_id = str(card.get("concept_id", ""))
        family_id = str(card.get("family_id", concept_id))
        family = families.get(family_id, {})
        limitations = tuple(str(x) for x in (str(card.get("known_construction_risks", "")).split("; ") if card.get("known_construction_risks") else ()))
        declared_issues = card.get("scientific_issues", ())
        if not isinstance(declared_issues, Sequence) or isinstance(declared_issues, (str, bytes)):
            declared_issues = ()
        # Legacy cards without structured issue records are conservatively
        # treated as resolvable evidence gaps, never classified by keywords.
        gap_issues = tuple(declared_issues) if declared_issues else tuple({
            "issue_id": f"{family_id}_evidence_gap",
            "issue_type": "PROVENANCE_INCOMPLETE",
            "related_claim_ids": [f"{family_id}_target"],
            "description": item,
            "severity": "CRITICAL",
            "blocking": True,
            "resolution_requirement": "independent scientific evidence is required",
        } for item in limitations if item.strip())
        critical_issues = tuple(item for item in gap_issues if isinstance(item, Mapping) and str(item.get("issue_type", "")).upper() in {"DATA_TARGET_MISMATCH", "OBSERVABLE_SEMANTICS_MISSING", "SAME_TARGET_RATIONALE_UNRESOLVED"})
        unresolved = tuple(_issue_descriptions(gap_issues))
        questions = (
            "Is the scientific question meaningful in this physical context?",
            "Are the named observables correctly interpreted for this dataset?",
            "Are the proposed O alternatives defensible alternatives of one target?",
            "Are reference findings and their verification rules adjudicable?",
        )
        evidence_ids = tuple(str(x) for x in card.get("evidence_ids", []) or ())
        source_lit = str(card.get("source_literature_grounding") or "")
        # Source resolution and scientific support are separate records.  A
        # source is never implicitly attached to the family target claim.
        raw_claims = card.get("scientific_support_claims", card.get("support_claims", ()))
        claims = _normalize_claims(raw_claims if isinstance(raw_claims, Sequence) and not isinstance(raw_claims, (str, bytes)) else ())
        if not claims:
            claims = [ScientificSupportClaim(
                claim_id=f"{family_id}_{claim_type.casefold()}", claim_type=claim_type,
                statement="", support_status="UNKNOWN", critical=True,
            ) for claim_type in SUPPORT_CLAIM_TYPES]
        raw_links = card.get("claim_support_links", ())
        links = _normalize_links(raw_links if isinstance(raw_links, Sequence) and not isinstance(raw_links, (str, bytes)) else ())
        resolutions = resolve_dataset_provenance(root, str(card.get("dataset_id", "")), evidence_ids)
        resolved_source_ids = tuple(str(item.source_id) for item in resolutions if item.resolution_status == "RESOLVED" and item.source_id)
        links_by_source: dict[str, list[str]] = {}
        for link in links:
            links_by_source.setdefault(link.source_id, []).append(link.claim_id)
        evidence_references = tuple(
            EvidenceReference(
                source_id=str(resolution.source_id or f"unresolved:{resolution.evidence_record_id}"),
                source_type=resolution.source_type or "DATASET_DOCUMENTATION",
                title=resolution.title or "Unresolved source record",
                authors=resolution.authors,
                year=resolution.year,
                venue=resolution.venue,
                doi=resolution.doi,
                url_or_identifier=resolution.external_identifier,
                dataset_specific=(resolution.source_type in {"DATASET_DOCUMENTATION", "DATASET_PUBLICATION", "DATA_FILE_METADATA"}),
                supports_claims=tuple(links_by_source.get(str(resolution.source_id or ""), ())),
                verification_status=resolution.resolution_status,
            ).to_dict()
            for resolution in resolutions
        )
        if not evidence_references:
            evidence_references = (EvidenceReference(
                source_id="unresolved:missing", source_type="DATASET_DOCUMENTATION",
                title="Unresolved source record", dataset_specific=True, verification_status="UNRESOLVED",
            ).to_dict(),)
        claim_provenance = tuple({
            "claim_id": claim.claim_id,
            "claim": claim.statement,
            "supporting_source_ids": list(claim.supporting_source_ids),
        } for claim in claims if claim.support_status == "SUPPORTED" and claim.supporting_source_ids)
        capability_cell = capability_cells.get((str(card.get("dataset_id", "")), concept_id))
        capability_status = getattr(capability_cell, "capability_status", "REQUIRED_CAPABILITY_UNKNOWN")
        predicate = evaluate_grounding_readiness({
            "capability_requirements_satisfied": capability_status == "REQUIREMENTS_SATISFIED",
            "support_claims": [claim.to_dict() for claim in claims],
            "claim_support_links": [link.to_dict() for link in links],
            "resolved_source_ids": list(resolved_source_ids),
            "critical_scientific_ambiguities": critical_issues,
            "resolvable_evidence_gaps": gap_issues,
        })
        predicate_recommendation = str(predicate["recommendation"])
        recommendation = predicate_recommendation
        reference_findings = tuple(
            item for item in (family.get("candidate_reference_operationalizations", []) or ())
            if isinstance(item, Mapping)
        )
        packets.append(ScientificGroundingPacket(
            family_id=family_id, dataset_id=str(card.get("dataset_id", "")), concept_id=concept_id,
            scientific_question=str(card.get("scientific_question_family", "")),
            scientific_target=str(card.get("scientific_concept", card.get("concept", ""))),
            scientific_entity_type={"kitchen_turbulence_activity": "SPATIAL_REGION", "kitchen_concentration_heterogeneity": "FIELD", "combustor_density_features": "POINT_FEATURE"}.get(concept_id, "SCIENTIFIC_ENTITY"),
            analysis_archetype={"kitchen_turbulence_activity": "EXTREMUM_FEATURE", "kitchen_concentration_heterogeneity": "FIELD_HETEROGENEITY", "combustor_density_features": "GRADIENT_FEATURE"}.get(concept_id, "UNSPECIFIED"),
            scientific_motivation=str(card.get("why_scientifically_meaningful", "")),
            dataset_context=str(card.get("available_dataset_support", "")),
            observable_support=str(card.get("observable_data_requirements", "")),
            geometry_or_boundary_support=str(card.get("available_dataset_support", "")),
            operationalization_options=tuple(str(x) for x in card.get("candidate_accepted_o_branches", []) or ()),
            operationalization_rationales=(str(card.get("defensible_choices_per_dimension", "")), str(card.get("o_role_coherence", ""))),
            operationalization_consequentiality=str(card.get("expected_o_to_g_divergence", card.get("branch_divergence", ""))),
            reference_findings=reference_findings,
            finding_requirement_contract=dict(family.get("finding_requirement_contract", {}) or {}),
            independent_scientific_sources=tuple(str(item.get("source_id")) for item in evidence_references if item.get("source_type") == "INDEPENDENT_SCIENTIFIC_REFERENCE") or ("UNRESOLVED_INDEPENDENT_SOURCE",),
            dataset_specific_sources=tuple(str(item.get("source_id")) for item in evidence_references if item.get("source_type") in {"DATASET_DOCUMENTATION", "DATASET_PUBLICATION", "DATA_FILE_METADATA"}) or ("UNRESOLVED_DATASET_SOURCE",),
            evidence_provenance={"evidence_ids": list(evidence_ids), "card_path": str(path.relative_to(root)), "source_literature": source_lit, "resolutions": [resolution.to_dict() for resolution in resolutions], "resolved_source_count": sum(item.resolution_status == "RESOLVED" for item in resolutions), "unresolved_source_count": sum(item.resolution_status != "RESOLVED" for item in resolutions), "placeholder_source_count": sum(str(item.source_id or "").startswith(("pending", "unknown")) for item in resolutions), "claim_support_links": [link.to_dict() for link in links], "critical_claim_ids": [claim.claim_id for claim in claims if claim.critical], "resolved_source_ids": list(resolved_source_ids)},
            known_limitations=limitations or ("formal release review pending",),
            unresolved_questions=unresolved, curator_questions=questions,
            grounding_recommendation=recommendation,
            evidence_references=evidence_references,
            claim_provenance=claim_provenance,
            scientific_support_claims=tuple(claim.to_dict() for claim in claims),
            claim_support_links=tuple(link.to_dict() for link in links),
            grounding_predicate=predicate,
            construction_recommendation=predicate_recommendation,
        ))
    return tuple(packets)


__all__ = ["EVIDENCE_SOURCE_TYPES", "GROUNDING_RECOMMENDATIONS", "SUPPORT_STATUSES", "SUPPORT_RELATIONS", "SUPPORT_CLAIM_TYPES", "ScientificEvidenceBindingError", "EvidenceReference", "EvidenceProvenanceResolution", "FamilyScientificEvidenceResolution", "resolve_evidence_provenance", "resolve_dataset_provenance", "resolve_family_scientific_evidence", "ScientificClaimProvenance", "ScientificSupportClaim", "ClaimSupportLink", "GroundingReadinessInput", "ScientificGroundingPacket", "build_grounding_packets", "evaluate_grounding_readiness"]
