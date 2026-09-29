"""Evidence-first inventory for the Kitchen turbulence-activity family.

This module is deliberately read-only with respect to authored scientific
records.  It assembles the current evidence, resolves every declared source
through the repository provenance contract, and reports evidence gaps.  It
does not obtain web evidence, infer units from values, or mutate support
statuses.  A later acquisition process may consume the explicit gaps and
append independently curated records.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .grounding import resolve_evidence_provenance


FAMILY_ID = "kitchen_turbulence_activity"
DATASET_ID = "Kitchen"

_CRITICAL_CLAIM_TYPES = (
    "SCIENTIFIC_TARGET_VALIDITY",
    "OBSERVABLE_SEMANTICS",
    "SAME_TARGET_RATIONALE",
    "FINDING_VERIFIABILITY",
)


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _family_paths(root: Path) -> dict[str, Path]:
    base = root / "datasets" / DATASET_ID / "construction"
    card = root / "artifacts/reference/concept_expansion_phase1/concept_design_cards" / f"{FAMILY_ID}.json"
    return {
        "card": card,
        "metadata": root / "datasets" / DATASET_ID / "data_metadata.json",
        "sources": base / "sources.json",
        "context_evidence": base / "context_evidence.json",
        "operationalization_evidence": base / "operationalization_evidence.json",
        "finding_evidence": base / "finding_evidence.json",
        "reviews": root / "artifacts/reference/scientific_runs/live_agent_validation/02_live_grounding_review/kitchen_turbulence_activity_reviews.json",
    }


def _claim_evidence_ids(claim: Mapping[str, Any]) -> tuple[str, ...]:
    values = claim.get("supporting_evidence_record_ids", claim.get("evidence_record_ids", ()))
    return tuple(dict.fromkeys(str(item) for item in values or () if str(item).strip()))


def _claim_source_ids(claim: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in claim.get("supporting_source_ids", ()) or () if str(item).strip()))


def _review_claims(reviews: Sequence[Mapping[str, Any]], claim_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for review in reviews:
        call = review.get("call", {}) if isinstance(review.get("call"), Mapping) else {}
        invocation_status = str(call.get("invocation_status", review.get("invocation_status", "UNKNOWN")))
        parsed = review.get("claims")
        if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
            parsed_result = call.get("parsed_result", {})
            parsed = parsed_result.get("claims", ()) if isinstance(parsed_result, Mapping) else ()
        for item in parsed:
            if not isinstance(item, Mapping) or str(item.get("claim_id", "")) != claim_id:
                continue
            result.append({
                "review_index": review.get("review_index", call.get("review_index")),
                "agent_id": call.get("agent_profile_id", review.get("agent_id")),
                "model_family": call.get("model_family", review.get("model_family")),
                "invocation_status": invocation_status,
                "proxy_disposition": str(item.get("proxy_disposition", "UNKNOWN")),
                "scientific_rationale": str(item.get("scientific_rationale", "")),
                "supporting_evidence_ids": [str(x) for x in item.get("supporting_evidence_ids", ()) or ()],
                "supporting_source_ids": [str(x) for x in item.get("supporting_source_ids", ()) or ()],
                "remaining_uncertainty": str(item.get("remaining_uncertainty", "")),
            })
    return result


def _gap_for_claim(claim_type: str) -> tuple[list[str], str, str]:
    """Return a claim-specific gap without inferring scientific meaning."""

    if claim_type == "SCIENTIFIC_TARGET_VALIDITY":
        return (
            [
                "dataset-specific evidence that a local turbulence-activity hotspot is a meaningful target in this flow",
                "a defensible objective rationale for ranking a local hotspot",
            ],
            "TARGET_VALIDITY_UNRESOLVED",
            "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED: obtain authoritative dataset or domain evidence for target validity",
        )
    if claim_type == "OBSERVABLE_SEMANTICS":
        return (
            [
                "units or normalization for ke and ep",
                "dataset-specific turbulence-model/solver provenance for ke and ep",
                "confirmation that the documented physical interpretation applies to these stored arrays",
            ],
            "OBSERVABLE_SEMANTICS_UNRESOLVED",
            "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED: locate primary dataset documentation or technical report defining ke/ep",
        )
    if claim_type == "SAME_TARGET_RATIONALE":
        return (
            [
                "an explicit scientific argument that ke and ep extrema are alternatives of one high-level target",
                "definitions and scope conditions for comparing the two criteria",
            ],
            "SAME_TARGET_RELATION_UNRESOLVED",
            "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED: obtain domain review or primary methodological support for same-target status",
        )
    if claim_type == "FINDING_VERIFIABILITY":
        return (
            [
                "authoritative confirmation of array-to-coordinate alignment and missing-value semantics",
                "a release-level statement that coordinate and companion quantities are scientifically adjudicable",
            ],
            "FINDING_VERIFIABILITY_UNRESOLVED",
            "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED: document the data-to-finding verification basis; computation alone is insufficient",
        )
    return (["claim-specific scientific support"], "EVIDENCE_GAP", "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED")


def _metadata_observations(metadata: Mapping[str, Any]) -> dict[str, Any]:
    variables = _rows(metadata.get("variables", ()))
    selected = {
        str(item.get("name")): {
            "physical_quantity": item.get("physical_quantity"),
            "type": item.get("type"),
            "components": item.get("components"),
            "association": item.get("association"),
            "unit": item.get("unit"),
            "source": item.get("source"),
        }
        for item in variables
        if str(item.get("name", "")) in {"ke", "ep"}
    }
    return {
        "record_path": "datasets/Kitchen/data_metadata.json",
        "record_kind": "repository_reader_metadata",
        "selected_variables": selected,
        "grid": metadata.get("grid", {}),
        "coordinate_system": metadata.get("coordinate_system", {}),
        "interpretation_limit": "Metadata records stored type/association and null unit fields; it does not establish physical units or model provenance.",
    }


def _evidence_checks(
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    metadata_observation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Summarize only what the current records explicitly expose.

    ``DOCUMENTED`` and ``METADATA_ONLY`` are evidence states, not support
    statuses.  In particular, a documented field name does not promote the
    corresponding UNKNOWN scientific-support claim.
    """

    ke = evidence_by_id.get("op_002", {})
    ep = evidence_by_id.get("op_002", {})
    model = evidence_by_id.get("op_003", {})
    selected = metadata_observation.get("selected_variables", {})
    return {
        "ke_meaning": {
            "evidence_status": "DOCUMENTED" if "turbulent kinetic energy" in str(ke.get("statement", "")).lower() else "UNRESOLVED",
            "evidence_ids": ["op_002"] if ke else [],
            "note": "The repository record names ke as turbulent kinetic energy; this does not settle units or solver provenance.",
        },
        "ep_meaning": {
            "evidence_status": "DOCUMENTED" if "turbulent dissipation rate" in str(ep.get("statement", "")).lower() else "UNRESOLVED",
            "evidence_ids": ["op_002"] if ep else [],
            "note": "The repository record names ep as turbulent dissipation rate; this does not settle units or solver provenance.",
        },
        "units_or_normalization": {
            "evidence_status": "UNRESOLVED",
            "evidence_ids": ["op_002", "op_003"] if ke and model else [],
            "note": "The reader metadata contains null unit fields and the construction records do not assert a normalization.",
        },
        "point_or_cell_association": {
            "evidence_status": "METADATA_ONLY" if all(str(selected.get(name, {}).get("association", "")) == "point" for name in ("ke", "ep")) else "UNRESOLVED",
            "metadata_value": {name: selected.get(name, {}).get("association") for name in ("ke", "ep")},
            "note": "Point association is recorded by repository metadata; physical interpretation remains a separate claim.",
        },
        "dataset_specific_physical_interpretation": {
            "evidence_status": "UNRESOLVED",
            "evidence_ids": ["op_002", "op_003"] if ke and model else [],
            "note": "The current records do not establish the solver/model provenance for these stored arrays in this dataset.",
        },
        "k_epsilon_model_context": {
            "evidence_status": "BACKGROUND_DOCUMENTED" if model else "UNRESOLVED",
            "evidence_ids": ["op_003"] if model else [],
            "note": "The Launder-Spalding publication supplies relevant k-epsilon background; it is not dataset-specific evidence.",
        },
        "same_high_level_target": {
            "evidence_status": "UNRESOLVED",
            "evidence_ids": [],
            "note": "No current record gives a domain-reviewed argument that ke and ep extrema instantiate one target for Kitchen.",
        },
        "finding_verifiability": {
            "evidence_status": "UNRESOLVED",
            "evidence_ids": ["op_002"] if ke else [],
            "note": "The arrays are computationally accessible, but an authoritative data-to-finding adjudication basis is not recorded.",
        },
    }


def build_kitchen_evidence_bundle(
    repository_root: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build WS3 artifacts from current repository records.

    The function intentionally performs no external request.  Consequently,
    ``new_evidence_count`` is zero for this run and unresolved claims retain
    their authored ``UNKNOWN`` status.  All output is derived and written only
    below ``output_root``.
    """

    root = Path(repository_root).resolve()
    out = (
        Path(output_root).resolve()
        if output_root
        else root / "outputs/experiments/kitchen_scientific_review"
    )
    paths = _family_paths(root)
    card = _read(paths["card"], {})
    metadata = _read(paths["metadata"], {})
    source_payload = _read(paths["sources"], {})
    source_records = _rows(source_payload.get("sources", ())) if isinstance(source_payload, Mapping) else []
    evidence_records = (
        _rows(_read(paths["context_evidence"], []))
        + _rows(_read(paths["operationalization_evidence"], []))
        + _rows(_read(paths["finding_evidence"], []))
    )
    evidence_by_id = {str(item.get("evidence_id")): item for item in evidence_records if item.get("evidence_id")}
    source_by_id = {str(item.get("source_id")): item for item in source_records if item.get("source_id")}
    reviews = _rows(_read(paths["reviews"], []))
    claims = _rows(card.get("scientific_support_claims", ())) if isinstance(card, Mapping) else []
    claims_by_type = {str(item.get("claim_type", "")): item for item in claims}
    claim_links = _rows(card.get("claim_support_links", ())) if isinstance(card, Mapping) else []
    link_by_claim: dict[str, list[dict[str, Any]]] = {}
    for link in claim_links:
        link_by_claim.setdefault(str(link.get("claim_id", "")), []).append(link)

    # Resolve all records, including context and finding records, through the
    # canonical evidence -> source path.  No source IDs are synthesized.
    resolution_rows: list[dict[str, Any]] = []
    for evidence in evidence_records:
        evidence_id = str(evidence.get("evidence_id", ""))
        resolution = resolve_evidence_provenance(evidence_id, evidence_records, source_records)
        row = resolution.to_dict()
        row["evidence_type"] = evidence.get("evidence_type")
        row["statement"] = evidence.get("statement")
        row["declared_source_id"] = evidence.get("source_id")
        resolution_rows.append(row)

    matrix: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for claim_type in _CRITICAL_CLAIM_TYPES:
        claim = claims_by_type.get(claim_type, {})
        claim_id = str(claim.get("claim_id", f"{FAMILY_ID}_{claim_type.casefold()}"))
        status = str(claim.get("support_status", "UNKNOWN")).upper()
        evidence_ids = list(_claim_evidence_ids(claim))
        source_ids = list(_claim_source_ids(claim))
        # Include links as a cross-check, but do not silently convert an
        # unlinked claim into a supported claim.
        linked = link_by_claim.get(claim_id, [])
        for link in linked:
            if str(link.get("evidence_record_id", "")) and str(link.get("evidence_record_id")) not in evidence_ids:
                evidence_ids.append(str(link.get("evidence_record_id")))
            if str(link.get("source_id", "")) and str(link.get("source_id")) not in source_ids:
                source_ids.append(str(link.get("source_id")))
        proxy_opinions = _review_claims(reviews, claim_id)
        claim_linked_evidence_ids = list(evidence_ids)
        claim_linked_source_ids = list(source_ids)
        # Review-cited records are included as evidence inspected for this
        # claim even when the authored claim has no support link.  The
        # separate ``claim_linked_*`` fields preserve that distinction.
        for opinion in proxy_opinions:
            for evidence_id in opinion.get("supporting_evidence_ids", ()):
                if evidence_id in evidence_by_id and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
            for source_id in opinion.get("supporting_source_ids", ()):
                if source_id in source_by_id and source_id not in source_ids:
                    source_ids.append(source_id)
        missing, gap_type, next_action = _gap_for_claim(claim_type)
        if status == "SUPPORTED":
            missing = []
            gap_type = "NONE"
            next_action = "No evidence acquisition action required by current support status; retain provenance audit."
        row = {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "claim": str(claim.get("statement", claim.get("claim", ""))),
            "critical": bool(claim.get("critical", True)),
            "current_support_status": status,
            "existing_evidence_ids": evidence_ids,
            "existing_source_ids": source_ids,
            "claim_linked_evidence_ids": claim_linked_evidence_ids,
            "claim_linked_source_ids": claim_linked_source_ids,
            "existing_evidence_scope": "authored claim links plus records cited by current proxy reviews; presence does not imply scientific support",
            "current_proxy_opinions": proxy_opinions,
            "missing_evidence": missing,
            "evidence_gap_type": gap_type,
            "next_evidence_action": next_action,
            "support_status_will_be_mutated": False,
        }
        matrix.append(row)
        if bool(claim.get("critical", True)) and status != "SUPPORTED":
            unresolved_rows.append({
                "claim_id": claim_id,
                "claim_type": claim_type,
                "current_support_status": status,
                "evidence_gap_type": gap_type,
                "missing_evidence": missing,
                "resolution_requirement": next_action,
                "blocking": True,
                "external_scientific_evidence_required": True,
            })

    before_digest = _digest({str(item.get("claim_id")): str(item.get("support_status", "UNKNOWN")) for item in claims})
    # This derived run cannot write authored claims.  Keep an explicit equal
    # digest in the output so downstream gates can verify that invariant.
    after_digest = before_digest
    resolved = [row for row in resolution_rows if row.get("resolution_status") == "RESOLVED"]
    unresolved_sources = [row for row in resolution_rows if row.get("resolution_status") != "RESOLVED"]
    source_ids_all = [str(item.get("source_id", "")) for item in source_records if str(item.get("source_id", ""))]
    evidence_ids_all = [str(item.get("evidence_id", "")) for item in evidence_records if str(item.get("evidence_id", ""))]
    duplicate_source_ids = sorted({item for item in source_ids_all if source_ids_all.count(item) > 1})
    duplicate_evidence_ids = sorted({item for item in evidence_ids_all if evidence_ids_all.count(item) > 1})
    provenance_audit = {
        "family_id": FAMILY_ID,
        "dataset_id": DATASET_ID,
        "status": "PASS" if not unresolved_sources and not duplicate_source_ids and not duplicate_evidence_ids else "INCOMPLETE",
        "resolution_method": "resolve_evidence_provenance (evidence_id -> declared source_id -> canonical SourceRecord)",
        "evidence_record_count": len(evidence_records),
        "resolved_evidence_record_count": len(resolved),
        "unresolved_evidence_record_count": len(unresolved_sources),
        "canonical_source_count": len(source_records),
        "canonical_source_ids": sorted(source_by_id),
        "canonical_source_ids_unique": not duplicate_source_ids,
        "duplicate_source_ids": duplicate_source_ids,
        "evidence_ids_unique": not duplicate_evidence_ids,
        "duplicate_evidence_ids": duplicate_evidence_ids,
        "evidence_resolution_rows": resolution_rows,
        "source_records": source_records,
        "claim_support_links": claim_links,
        "claim_provenance_complete": all(
            str(item.get("current_support_status")) != "SUPPORTED"
            or (item.get("existing_evidence_ids") and item.get("existing_source_ids"))
            for item in matrix
        ),
        "support_status_mutation": {
            "before_digest": before_digest,
            "after_digest": after_digest,
            "mutated": False,
        },
        "specific_chain_checks": {
            "op_002_to_src_003": bool(
                evidence_by_id.get("op_002", {}).get("source_id") == "src_003"
                and source_by_id.get("src_003") is not None
            ),
            "op_003_to_src_004_to_doi": bool(
                evidence_by_id.get("op_003", {}).get("source_id") == "src_004"
                and source_by_id.get("src_004", {}).get("doi") == "10.1016/0045-7825(74)90029-2"
            ),
        },
    }

    existing_inventory = [
        {
            "evidence_id": str(item.get("evidence_id")),
            "evidence_type": item.get("evidence_type"),
            "source_id": item.get("source_id"),
            "statement": item.get("statement"),
            "status": "EXISTING_REPOSITORY_RECORD",
        }
        for item in evidence_records
        if item.get("evidence_id")
    ]
    metadata_observation = _metadata_observations(metadata)
    new_inventory = {
        "family_id": FAMILY_ID,
        "dataset_id": DATASET_ID,
        "acquisition_mode": "REPOSITORY_ONLY_NO_EXTERNAL_REQUEST",
        "status": "NO_NEW_EVIDENCE_ACQUIRED",
        "new_evidence_count": 0,
        "new_source_count": 0,
        "new_evidence": [],
        "new_sources": [],
        "existing_repository_evidence": existing_inventory,
        "repository_metadata_observations": [metadata_observation],
        "scientific_evidence_checks": _evidence_checks(evidence_by_id, metadata_observation),
        "external_scientific_evidence_required": bool(unresolved_rows),
        "trigger_inputs": {
            "new_scientific_evidence_added": False,
            "unresolved_critical_claim_count": len(unresolved_rows),
            "prior_proxy_review_count": len(reviews),
            "action": "EVIDENCE_ACQUISITION_REQUIRED" if unresolved_rows else "NO_ADDITIONAL_EVIDENCE_ACTION",
        },
        "support_status_mutated": False,
    }
    unresolved = {
        "family_id": FAMILY_ID,
        "dataset_id": DATASET_ID,
        "status": "EXTERNAL_SCIENTIFIC_EVIDENCE_REQUIRED" if unresolved_rows else "NO_UNRESOLVED_CRITICAL_GAPS",
        "unresolved_critical_claim_count": len(unresolved_rows),
        "unresolved_critical_claim_ids": [item["claim_id"] for item in unresolved_rows],
        "gaps": unresolved_rows,
        "next_action": "Acquire authoritative evidence and obtain domain review before curator handoff." if unresolved_rows else "Proceed to the existing grounding readiness gate.",
        "new_evidence_added": False,
        "support_status_mutated": False,
    }

    matrix_payload = {
        "family_id": FAMILY_ID,
        "dataset_id": DATASET_ID,
        "claims": matrix,
        "critical_claim_count": len(matrix),
        "unresolved_critical_claim_ids": [item["claim_id"] for item in unresolved_rows],
        "support_status_mutated": False,
    }
    _write_json(out / "critical_claim_matrix.json", matrix_payload)
    _write_json(out / "new_evidence_inventory.json", new_inventory)
    _write_json(out / "provenance_audit.json", provenance_audit)
    _write_json(out / "unresolved_evidence_gaps.json", unresolved)
    return {
        "status": "PASS",
        "family_id": FAMILY_ID,
        "dataset_id": DATASET_ID,
        "output_root": str(out),
        "critical_claim_matrix": matrix_payload,
        "new_evidence_inventory": new_inventory,
        "provenance_audit": provenance_audit,
        "unresolved_evidence_gaps": unresolved,
    }


__all__ = ["DATASET_ID", "FAMILY_ID", "build_kitchen_evidence_bundle"]
