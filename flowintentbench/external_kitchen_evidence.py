"""Construction-side Kitchen evidence inventory.

External sources reviewed during construction are serialized with the same
frozen contracts used by authored dataset evidence. This module deliberately
does not implement a retrieval schema: URLs and search metadata are neither
part of a ``SourceRecord`` nor persisted in milestone artifacts. The inventory
is a candidate bundle only; it never mutates canonical construction records or
promotes a support status.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .context import EvidenceRecord, SourceCollection, SourceRecord
from .grounding import ClaimSupportLink, ScientificSupportClaim


FAMILY_ID = "kitchen_turbulence_activity"
DATASET_ID = "Kitchen"
ACQUISITION_TIMESTAMP = datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def _source_records() -> tuple[SourceRecord, ...]:
    """Return reviewed sources using the frozen eight-field SourceRecord."""

    return (
        SourceRecord(
            source_id="ext_src_001",
            source_type="dataset_documentation",
            title="LAB1: Visualisation of a room",
            authors=["Kai-Mikael Jää-Aro", "Lars Kjelldahl"],
            year=2006,
            venue="KTH DD2257 course materials",
            doi=None,
            report_id="KTH-DD2257-LAB1-2006",
        ),
        SourceRecord(
            source_id="ext_src_002",
            source_type="dataset_documentation",
            title="Kitchen",
            authors=[],
            year=None,
            venue="VTK Examples",
            doi=None,
            report_id="VTK-EXAMPLES-KITCHEN",
        ),
        SourceRecord(
            source_id="ext_src_003",
            source_type="dataset_documentation",
            title="VTK File Formats",
            authors=[],
            year=None,
            venue="VTK Documentation",
            doi=None,
            report_id="VTK-FILE-FORMATS",
        ),
        SourceRecord(
            source_id="ext_src_004",
            source_type="dataset_documentation",
            title="VTK Book Chapter 6: Visualization Algorithms",
            authors=[],
            year=None,
            venue="VTK Book",
            doi=None,
            report_id="VTK-BOOK-CH6",
        ),
    )


def _evidence_records() -> tuple[EvidenceRecord, ...]:
    """Return evidence records with only the frozen EvidenceRecord fields."""

    return (
        EvidenceRecord(
            evidence_id="ext_ev_001",
            dataset_id=DATASET_ID,
            evidence_type="operationalization",
            statement=(
                "The KTH laboratory handout identifies kitchen.vtk as a "
                "structured-grid room dataset, describes velocity values at "
                "grid points, names ke as turbulent kinetic energy and ep as "
                "turbulent dissipation rate, and separately lists concentration arrays."
            ),
            source_id="ext_src_001",
            locator={"page": 1, "section": "The data"},
            eligible_for_context=False,
            context_facts=[],
        ),
        EvidenceRecord(
            evidence_id="ext_ev_002",
            dataset_id=DATASET_ID,
            evidence_type="context",
            statement=(
                "The official VTK Kitchen example documents a small kitchen "
                "flow scene, convection currents, and streamline-based "
                "inspection of flow-field features."
            ),
            source_id="ext_src_002",
            locator={"section": "Description"},
            eligible_for_context=False,
            context_facts=[],
        ),
        EvidenceRecord(
            evidence_id="ext_ev_003",
            dataset_id=DATASET_ID,
            evidence_type="operationalization",
            statement=(
                "VTK format documentation specifies named point and cell data "
                "arrays and distinguishes point-associated attributes from "
                "cell-associated attributes."
            ),
            source_id="ext_src_003",
            locator={"section": "Dataset Attribute Format"},
            eligible_for_context=False,
            context_facts=[],
        ),
        EvidenceRecord(
            evidence_id="ext_ev_004",
            dataset_id=DATASET_ID,
            evidence_type="operationalization",
            statement=(
                "The VTK Book chapter describes streamlines as a visualization "
                "of vector-field structure and flow features in structured data."
            ),
            source_id="ext_src_004",
            locator={"section": "Streamline visualization"},
            eligible_for_context=False,
            context_facts=[],
        ),
    )


def _support_claims(*, family_id: str = FAMILY_ID) -> tuple[ScientificSupportClaim, ...]:
    """Represent unresolved construction claims without changing support status."""

    if family_id == "kitchen_flow_regions":
        prefix = family_id
        return (
            ScientificSupportClaim(
                claim_id=f"{prefix}_target",
                claim_type="SCIENTIFIC_TARGET_VALIDITY",
                statement="High-speed flow-region characterization is a scientifically meaningful target for the Kitchen velocity field.",
                support_status="UNKNOWN",
                supporting_source_ids=("ext_src_001", "ext_src_002", "ext_src_004"),
                supporting_evidence_record_ids=("ext_ev_001", "ext_ev_002", "ext_ev_004"),
                critical=True,
            ),
            ScientificSupportClaim(
                claim_id=f"{prefix}_observable",
                claim_type="OBSERVABLE_SEMANTICS",
                statement="The Kitchen dataset exposes point-associated velocity vectors from which speed and connected regions can be computed.",
                support_status="UNKNOWN",
                supporting_source_ids=("ext_src_001", "ext_src_003"),
                supporting_evidence_record_ids=("ext_ev_001", "ext_ev_003"),
                critical=True,
            ),
            ScientificSupportClaim(
                claim_id=f"{prefix}_operationalization",
                claim_type="OPERATIONALIZATION_RATIONALE",
                statement="Thresholded face-connected regions and explicit speed measures are reproducible operationalizations of high-speed flow regions.",
                support_status="UNKNOWN",
                supporting_source_ids=("ext_src_003", "ext_src_004"),
                supporting_evidence_record_ids=("ext_ev_003", "ext_ev_004"),
                critical=True,
            ),
            ScientificSupportClaim(
                claim_id=f"{prefix}_findings",
                claim_type="FINDING_VERIFIABILITY",
                statement="Region locations and strength values can be deterministically verified from the named Kitchen velocity array and grid coordinates.",
                support_status="UNKNOWN",
                supporting_source_ids=("ext_src_001", "ext_src_003"),
                supporting_evidence_record_ids=("ext_ev_001", "ext_ev_003"),
                critical=True,
            ),
        )
    return (
        ScientificSupportClaim(
            claim_id="kitchen_turbulence_activity_target",
            claim_type="SCIENTIFIC_TARGET_VALIDITY",
            statement=(
                "Ranking a local turbulence-activity hotspot is a scientifically "
                "valid target for the Kitchen dataset."
            ),
            support_status="UNKNOWN",
            supporting_source_ids=("ext_src_002", "ext_src_004"),
            supporting_evidence_record_ids=("ext_ev_002", "ext_ev_004"),
            critical=True,
        ),
        ScientificSupportClaim(
            claim_id="kitchen_turbulence_activity_observable",
            claim_type="OBSERVABLE_SEMANTICS",
            statement=(
                "The Kitchen dataset exposes interpretable ke and ep observables "
                "for the proposed analysis."
            ),
            support_status="UNKNOWN",
            supporting_source_ids=("ext_src_001",),
            supporting_evidence_record_ids=("ext_ev_001",),
            critical=True,
        ),
        ScientificSupportClaim(
            claim_id="kitchen_turbulence_activity_same_target",
            claim_type="SAME_TARGET_RATIONALE",
            statement=(
                "Maximum ke and maximum ep are alternative operationalizations of "
                "one broad turbulence-activity target."
            ),
            support_status="UNKNOWN",
            supporting_source_ids=(),
            supporting_evidence_record_ids=(),
            critical=True,
        ),
        ScientificSupportClaim(
            claim_id="kitchen_turbulence_activity_findings",
            claim_type="FINDING_VERIFIABILITY",
            statement=(
                "Findings based on named Kitchen arrays and their coordinates can "
                "be scientifically and deterministically verified."
            ),
            support_status="UNKNOWN",
            supporting_source_ids=("ext_src_001", "ext_src_003"),
            supporting_evidence_record_ids=("ext_ev_001", "ext_ev_003"),
            critical=True,
        ),
    )


def _support_links(*, family_id: str = FAMILY_ID) -> tuple[ClaimSupportLink, ...]:
    if family_id == "kitchen_flow_regions":
        prefix = family_id
        return (
            ClaimSupportLink(claim_id=f"{prefix}_target", evidence_record_id="ext_ev_001", source_id="ext_src_001", support_relation="DIRECT", supported_statement="The KTH handout documents the Kitchen velocity field used by the target."),
            ClaimSupportLink(claim_id=f"{prefix}_target", evidence_record_id="ext_ev_002", source_id="ext_src_002", support_relation="CONTEXTUAL", supported_statement="The official Kitchen example documents the room flow scene and flow-field features."),
            ClaimSupportLink(claim_id=f"{prefix}_target", evidence_record_id="ext_ev_004", source_id="ext_src_004", support_relation="BACKGROUND", supported_statement="The VTK Book describes vector-field flow features in structured data."),
            ClaimSupportLink(claim_id=f"{prefix}_observable", evidence_record_id="ext_ev_001", source_id="ext_src_001", support_relation="DIRECT", supported_statement="The KTH handout describes grid-point velocity data."),
            ClaimSupportLink(claim_id=f"{prefix}_observable", evidence_record_id="ext_ev_003", source_id="ext_src_003", support_relation="CONTEXTUAL", supported_statement="VTK format documentation distinguishes point-associated arrays."),
            ClaimSupportLink(claim_id=f"{prefix}_operationalization", evidence_record_id="ext_ev_003", source_id="ext_src_003", support_relation="DIRECT", supported_statement="Named point arrays and structured-grid connectivity are executable from the VTK representation."),
            ClaimSupportLink(claim_id=f"{prefix}_operationalization", evidence_record_id="ext_ev_004", source_id="ext_src_004", support_relation="BACKGROUND", supported_statement="The VTK Book provides scientific background for structured flow-feature analysis."),
            ClaimSupportLink(claim_id=f"{prefix}_findings", evidence_record_id="ext_ev_001", source_id="ext_src_001", support_relation="DIRECT", supported_statement="The handout documents the grid-point velocity data used for deterministic findings."),
            ClaimSupportLink(claim_id=f"{prefix}_findings", evidence_record_id="ext_ev_003", source_id="ext_src_003", support_relation="CONTEXTUAL", supported_statement="VTK documentation defines point-array and coordinate access."),
        )
    return (
        ClaimSupportLink(
            claim_id="kitchen_turbulence_activity_target",
            evidence_record_id="ext_ev_002",
            source_id="ext_src_002",
            support_relation="CONTEXTUAL",
            supported_statement=(
                "The official example documents a kitchen flow scene and flow-field features; "
                "it does not establish the hotspot target."
            ),
        ),
        ClaimSupportLink(
            claim_id="kitchen_turbulence_activity_target",
            evidence_record_id="ext_ev_004",
            source_id="ext_src_004",
            support_relation="BACKGROUND",
            supported_statement=(
                "The VTK Book explains streamline visualization, but does not establish "
                "turbulence-activity hotspot validity."
            ),
        ),
        ClaimSupportLink(
            claim_id="kitchen_turbulence_activity_observable",
            evidence_record_id="ext_ev_001",
            source_id="ext_src_001",
            support_relation="DIRECT",
            supported_statement=(
                "The KTH handout names ke and ep and describes grid-point velocity data."
            ),
        ),
        ClaimSupportLink(
            claim_id="kitchen_turbulence_activity_findings",
            evidence_record_id="ext_ev_001",
            source_id="ext_src_001",
            support_relation="DIRECT",
            supported_statement=(
                "The handout documents structured-grid/grid-point data and named ke/ep arrays."
            ),
        ),
        ClaimSupportLink(
            claim_id="kitchen_turbulence_activity_findings",
            evidence_record_id="ext_ev_003",
            source_id="ext_src_003",
            support_relation="CONTEXTUAL",
            supported_statement=(
                "VTK format documentation distinguishes named point and cell arrays."
            ),
        ),
    )


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def acquire_kitchen_external_evidence(
    repository_root: str | Path,
    output_root: str | Path | None = None,
    *,
    family_id: str = FAMILY_ID,
    concept_id: str | None = None,
) -> dict[str, Any]:
    """Write a schema-frozen candidate evidence inventory.

    The function returns a small in-memory compatibility summary for existing
    callers, while every newly generated JSON artifact contains only frozen
    construction objects. Existing canonical dataset files are read solely to
    report identifier collisions and are never modified.
    """

    root = Path(repository_root).resolve()
    out = (
        Path(output_root).resolve()
        if output_root
        else root / "outputs/experiments/kitchen_external_evidence"
    )
    out.mkdir(parents=True, exist_ok=True)
    # Remove only obsolete files from this generated candidate bundle. These
    # names belonged to the former retrieval/URL artifact and are not part of
    # the frozen evidence inventory. Canonical dataset files are untouched.
    for obsolete_name in (
        "source_search_log.json",
        "source_urls.json",
        "source_targets.json",
    ):
        obsolete = out / obsolete_name
        if obsolete.exists():
            obsolete.unlink()

    sources = _source_records()
    evidence = _evidence_records()
    claims = _support_claims(family_id=family_id)
    links = _support_links(family_id=family_id)

    # Validate every object before writing. This catches accidental additions
    # to any frozen model immediately.
    source_collection = SourceCollection(dataset_id=DATASET_ID, sources=list(sources))
    for record in evidence:
        EvidenceRecord.model_validate(record)
    for claim in claims:
        ScientificSupportClaim.from_mapping(claim.to_dict())
    for link in links:
        ClaimSupportLink.from_mapping(link.to_dict())

    construction_root = root / "datasets" / DATASET_ID / "construction"
    source_payload = _read_json(construction_root / "sources.json", {})
    existing_source_ids = (
        {
            str(item.get("source_id"))
            for item in source_payload.get("sources", [])
            if isinstance(item, Mapping)
        }
        if isinstance(source_payload, Mapping)
        else set()
    )
    existing_evidence_ids: set[str] = set()
    for evidence_path in (
        "operationalization_evidence.json",
        "context_evidence.json",
        "finding_evidence.json",
    ):
        rows = _read_json(construction_root / evidence_path, [])
        if isinstance(rows, list):
            existing_evidence_ids.update(
                str(item.get("evidence_id"))
                for item in rows
                if isinstance(item, Mapping)
            )

    source_rows = [item.model_dump(mode="json") for item in sources]
    evidence_rows = [item.model_dump(mode="json") for item in evidence]
    claim_rows = [item.to_dict() for item in claims]
    link_rows = [item.to_dict() for item in links]

    # Exact existing contracts, not a new ExternalEvidenceRecord.
    _write_json(out / "acquired_sources.json", source_collection.model_dump(mode="json"))
    _write_json(out / "acquired_evidence.json", evidence_rows)
    # Condition-milestone aliases are still raw frozen object collections;
    # they make the inventory consumable without introducing another schema.
    _write_json(out / "source_records.json", source_rows)
    _write_json(out / "evidence_records.json", evidence_rows)
    _write_json(out / "claim_support_links.json", link_rows)
    _write_json(
        out / "evidence_inventory.json",
        {
            "evidence_records": evidence_rows,
            "scientific_support_claims": claim_rows,
            "claim_support_links": link_rows,
        },
    )
    _write_json(
        out / "claim_evidence_map.json",
        {
            "scientific_support_claims": claim_rows,
            "claim_support_links": link_rows,
        },
    )
    unresolved_rows = [
        item
        for item in claim_rows
        if str(item.get("support_status", "UNKNOWN")) != "SUPPORTED"
    ]
    _write_json(out / "unresolved_after_external_search.json", unresolved_rows)

    # Same-target alternatives are construction reasoning, retained only in
    # the return value for old callers; they are not persisted as a second
    # evidence schema.
    same_target_options = {
        "A": {
            "label": "ke_and_ep_are_alternative_observables_of_one_broad_target",
            "assessment": "NOT_ESTABLISHED",
            "rationale": "The reviewed sources do not link the two extrema to one turbulence-activity construct.",
        },
        "B": {
            "label": "ke_and_ep_are_related_but_scientifically_distinct_constructs",
            "assessment": "NOT_ESTABLISHED",
            "rationale": "The reviewed sources do not provide enough dataset-specific or solver-specific semantics to select this revision.",
        },
        "C": {
            "label": "evidence_insufficient",
            "assessment": "SELECTED",
            "rationale": "Keep the same-target question open until dataset-specific scientific evidence or domain review is available.",
        },
        "selected_branch": "C",
    }
    claim_evidence_map = {
        "scientific_support_claims": claim_rows,
        "claim_support_links": link_rows,
        "same_target_options": same_target_options,
    }
    unresolved = {
        "family_id": family_id,
        "dataset_id": DATASET_ID,
        "status": "EVIDENCE_GAPS_REMAIN",
        "unresolved_claim_ids": [item["claim_id"] for item in unresolved_rows],
        "same_target_branch": "C_INSUFFICIENT",
        "next_action": "Obtain dataset-specific scientific/domain review before curator gate; do not promote support statuses.",
        "support_status_mutated": False,
    }
    acquired = {
        "dataset_id": DATASET_ID,
        # Construction-side binding metadata is consumed by the resolver and
        # is not persisted in any frozen SourceRecord/EvidenceRecord artifact.
        "family_id": family_id,
        "concept_id": concept_id or family_id,
        "sources": source_rows,
        # Compatibility alias for the live expert adapter. This is an
        # in-memory list of SourceRecord values and is never written as a new
        # source schema.
        "source_records": source_rows,
        "evidence_records": evidence_rows,
        "scientific_support_claims": claim_rows,
        "claim_support_links": link_rows,
        "canonical_source_id_collision_count": len(
            existing_source_ids & {item.source_id for item in sources}
        ),
        "evidence_id_collision_count": len(
            existing_evidence_ids & {item.evidence_id for item in evidence}
        ),
    }
    return {
        "status": "PASS",
        "output_root": str(out),
        "source_count": len(sources),
        "evidence_count": len(evidence),
        "claim_evidence_map": claim_evidence_map,
        "unresolved": unresolved,
        "acquired_sources": acquired,
        "acquisition_digest": _digest(
            {
                "sources": source_rows,
                "evidence": evidence_rows,
                "claims": claim_rows,
                "links": link_rows,
            }
        ),
    }


__all__ = ["FAMILY_ID", "DATASET_ID", "acquire_kitchen_external_evidence"]
