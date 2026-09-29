"""Derived evidence-coverage audits for the canonical Flow Expert packet.

The records consumed here remain the frozen ``ScientificSupportClaim``,
``ClaimSupportLink``, ``EvidenceRecord`` and ``SourceRecord`` projections.
This module does not assign scientific truth.  It keeps three questions
separate:

* whether every declared identifier closes through claim -> evidence -> source;
* whether every critical, authored support dimension retains the relation tier
  visible in the baseline packet; and
* whether a model response changes after controlled evidence removal.

``DIRECT > CONTEXTUAL > BACKGROUND`` is used only to preserve the authored
support tier during an ablation.  It is not a confidence score and does not
claim that one source is scientifically more correct than another.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SUPPORT_RELATION_STRENGTH = {
    "BACKGROUND": 1,
    "CONTEXTUAL": 2,
    "DIRECT": 3,
}

CLAIM_TYPE_TO_REVIEW_FIELDS = {
    "SCIENTIFIC_TARGET_VALIDITY": ("scientific_target_supported",),
    "DATASET_CONTEXT": (
        "scientific_target_supported",
        "materialization_scientifically_meaningful",
    ),
    "OBSERVABLE_SEMANTICS": (
        "fixed_o_supported",
        "materialization_scientifically_meaningful",
    ),
    "OPERATIONALIZATION_RATIONALE": (
        "fixed_o_supported",
        "unresolved_o_is_legitimate_scientific_choice",
        "materialization_scientifically_meaningful",
    ),
    "SAME_TARGET_RATIONALE": (
        "unresolved_o_is_legitimate_scientific_choice",
    ),
    "FINDING_VERIFIABILITY": (
        "finding_contract_supported",
        "materialization_scientifically_meaningful",
    ),
}


def _object_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    dataset = packet.get("dataset_identity")
    family = packet.get("family_identity")
    dataset = dataset if isinstance(dataset, Mapping) else {}
    family = family if isinstance(family, Mapping) else {}
    conditions = _object_rows(packet.get("conditions"))
    return {
        "dataset_id": str(dataset.get("dataset_id", "")),
        "family_id": str(family.get("family_id", "")),
        "concept_id": str(family.get("concept_id", "")),
        "conditions": sorted(str(item.get("condition", "")) for item in conditions),
    }


def _unique_rows(
    rows: Sequence[Mapping[str, Any]], field: str, collection: str
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        identifier = str(row.get(field, "")).strip()
        if not identifier:
            errors.append(
                {
                    "code": "IDENTIFIER_MISSING",
                    "field": f"{collection}[{index}].{field}",
                }
            )
            continue
        if identifier in indexed:
            errors.append(
                {
                    "code": "DUPLICATE_IDENTIFIER",
                    "field": f"{collection}.{identifier}",
                }
            )
        indexed[identifier] = row
    return indexed, errors


def audit_evidence_provenance_closure(
    review_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit identifier closure without treating closure as sufficiency."""

    sources = _object_rows(review_packet.get("source_records"))
    evidence = _object_rows(review_packet.get("evidence_records"))
    claims = _object_rows(review_packet.get("scientific_claims"))
    links = _object_rows(review_packet.get("claim_support_links"))
    source_by_id, source_errors = _unique_rows(sources, "source_id", "source_records")
    evidence_by_id, evidence_errors = _unique_rows(
        evidence, "evidence_id", "evidence_records"
    )
    claim_by_id, claim_errors = _unique_rows(
        claims, "claim_id", "scientific_claims"
    )
    errors = [*source_errors, *evidence_errors, *claim_errors]
    link_keys: set[tuple[str, str, str]] = set()
    closed_links: list[dict[str, Any]] = []

    for index, link in enumerate(links):
        claim_id = str(link.get("claim_id", "")).strip()
        evidence_id = str(link.get("evidence_record_id", "")).strip()
        source_id = str(link.get("source_id", "")).strip()
        relation = str(link.get("support_relation", "")).strip().upper()
        key = (claim_id, evidence_id, source_id)
        if key in link_keys:
            errors.append(
                {
                    "code": "DUPLICATE_CLAIM_SUPPORT_LINK",
                    "field": f"claim_support_links[{index}]",
                }
            )
        link_keys.add(key)
        if relation not in SUPPORT_RELATION_STRENGTH:
            errors.append(
                {
                    "code": "SUPPORT_RELATION_INVALID",
                    "field": f"claim_support_links[{index}].support_relation",
                }
            )
            continue
        if claim_id not in claim_by_id:
            errors.append(
                {
                    "code": "CLAIM_NOT_FOUND",
                    "field": f"claim_support_links[{index}].claim_id",
                }
            )
            continue
        record = evidence_by_id.get(evidence_id)
        if record is None:
            errors.append(
                {
                    "code": "EVIDENCE_NOT_FOUND",
                    "field": f"claim_support_links[{index}].evidence_record_id",
                }
            )
            continue
        if source_id not in source_by_id:
            errors.append(
                {
                    "code": "SOURCE_NOT_FOUND",
                    "field": f"claim_support_links[{index}].source_id",
                }
            )
            continue
        if str(record.get("source_id", "")).strip() != source_id:
            errors.append(
                {
                    "code": "EVIDENCE_SOURCE_MISMATCH",
                    "field": f"claim_support_links[{index}]",
                }
            )
            continue
        closed_links.append(
            {
                "claim_id": claim_id,
                "evidence_record_id": evidence_id,
                "source_id": source_id,
                "support_relation": relation,
            }
        )

    closed_by_claim: dict[str, list[dict[str, Any]]] = {}
    for link in closed_links:
        closed_by_claim.setdefault(link["claim_id"], []).append(link)
    for claim_id, claim in claim_by_id.items():
        linked = closed_by_claim.get(claim_id, [])
        linked_evidence = {item["evidence_record_id"] for item in linked}
        linked_sources = {item["source_id"] for item in linked}
        declared_evidence = {
            str(item) for item in claim.get("evidence_ids", ()) or ()
        }
        declared_sources = {str(item) for item in claim.get("source_ids", ()) or ()}
        if declared_evidence != linked_evidence or declared_sources != linked_sources:
            errors.append(
                {
                    "code": "CLAIM_LINK_CLOSURE_MISMATCH",
                    "field": claim_id,
                }
            )

    linked_evidence_ids = {item["evidence_record_id"] for item in closed_links}
    return {
        "status": "PROVENANCE_CLOSED" if not errors else "PROVENANCE_NOT_CLOSED",
        "errors": errors,
        "closed_links": closed_links,
        "source_count": len(source_by_id),
        "evidence_count": len(evidence_by_id),
        "claim_count": len(claim_by_id),
        "unlinked_evidence_ids": sorted(set(evidence_by_id) - linked_evidence_ids),
    }


def derive_evidence_coverage_contract(
    review_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze required support dimensions from one authored baseline packet.

    Each critical claim is one required dimension.  Its strongest visible,
    provenance-closed relation is retained as the minimum relation for
    controlled ablations.  Claims with no closed link remain explicitly
    uncovered instead of being silently removed from the denominator.
    """

    provenance = audit_evidence_provenance_closure(review_packet)
    links_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for link in provenance["closed_links"]:
        links_by_claim.setdefault(str(link["claim_id"]), []).append(link)
    dimensions: list[dict[str, Any]] = []
    for claim in _object_rows(review_packet.get("scientific_claims")):
        if claim.get("critical") is not True:
            continue
        claim_id = str(claim.get("claim_id", "")).strip()
        linked = links_by_claim.get(claim_id, [])
        strongest = max(
            (
                str(link["support_relation"])
                for link in linked
            ),
            key=lambda relation: SUPPORT_RELATION_STRENGTH[relation],
            default=None,
        )
        dimensions.append(
            {
                "dimension_id": claim_id,
                "claim_type": str(claim.get("claim_type", "")),
                "minimum_support_relation": strongest,
                "baseline_evidence_ids": sorted(
                    {str(link["evidence_record_id"]) for link in linked}
                ),
                "baseline_source_ids": sorted(
                    {str(link["source_id"]) for link in linked}
                ),
            }
        )
    return {
        "status": (
            "COVERAGE_CONTRACT_DERIVED"
            if provenance["status"] == "PROVENANCE_CLOSED"
            else "COVERAGE_CONTRACT_INVALID_PROVENANCE"
        ),
        "identity": _identity(review_packet),
        "dimensions": dimensions,
        "required_support_dimension_ids": sorted(
            item["dimension_id"] for item in dimensions
        ),
        "provenance_status": provenance["status"],
    }


def audit_evidence_coverage(
    review_packet: Mapping[str, Any],
    *,
    coverage_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit required scientific-support coverage independently of judgment."""

    contract = (
        dict(coverage_contract)
        if isinstance(coverage_contract, Mapping)
        else derive_evidence_coverage_contract(review_packet)
    )
    provenance = audit_evidence_provenance_closure(review_packet)
    errors: list[dict[str, str]] = []
    if contract.get("identity") != _identity(review_packet):
        errors.append(
            {
                "code": "COVERAGE_CONTRACT_BINDING_MISMATCH",
                "field": "identity",
            }
        )
    current_claims = {
        str(item.get("claim_id", "")): item
        for item in _object_rows(review_packet.get("scientific_claims"))
    }
    links_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for link in provenance["closed_links"]:
        links_by_claim.setdefault(str(link["claim_id"]), []).append(link)

    dimension_rows: list[dict[str, Any]] = []
    for dimension in _object_rows(contract.get("dimensions")):
        dimension_id = str(dimension.get("dimension_id", ""))
        claim = current_claims.get(dimension_id)
        if claim is None or str(claim.get("claim_type", "")) != str(
            dimension.get("claim_type", "")
        ) or claim.get("critical") is not True:
            errors.append(
                {
                    "code": "COVERAGE_DIMENSION_DRIFT",
                    "field": dimension_id,
                }
            )
        required_relation = dimension.get("minimum_support_relation")
        linked = links_by_claim.get(dimension_id, [])
        available_relations = sorted(
            {str(item["support_relation"]) for item in linked},
            key=lambda relation: SUPPORT_RELATION_STRENGTH[relation],
            reverse=True,
        )
        strongest_available = available_relations[0] if available_relations else None
        if required_relation not in SUPPORT_RELATION_STRENGTH:
            coverage_status = "NO_AUTHORED_SUPPORT"
            covered = False
        elif strongest_available is None:
            coverage_status = "NO_VISIBLE_SUPPORT"
            covered = False
        elif (
            SUPPORT_RELATION_STRENGTH[strongest_available]
            < SUPPORT_RELATION_STRENGTH[str(required_relation)]
        ):
            coverage_status = "REQUIRED_RELATION_MISSING"
            covered = False
        else:
            coverage_status = "COVERED"
            covered = True
        dimension_rows.append(
            {
                "dimension_id": dimension_id,
                "claim_type": str(dimension.get("claim_type", "")),
                "minimum_support_relation": required_relation,
                "visible_support_relations": available_relations,
                "visible_evidence_ids": sorted(
                    {str(item["evidence_record_id"]) for item in linked}
                ),
                "visible_source_ids": sorted(
                    {str(item["source_id"]) for item in linked}
                ),
                "coverage_status": coverage_status,
                "covered": covered,
            }
        )

    missing = sorted(
        item["dimension_id"] for item in dimension_rows if not item["covered"]
    )
    complete = (
        provenance["status"] == "PROVENANCE_CLOSED"
        and not errors
        and bool(dimension_rows)
        and not missing
    )
    return {
        "provenance_status": provenance["status"],
        "evidence_coverage_status": (
            "EVIDENCE_COVERAGE_COMPLETE"
            if complete
            else "EVIDENCE_COVERAGE_INCOMPLETE"
        ),
        "required_support_dimension_ids": sorted(
            item["dimension_id"] for item in dimension_rows
        ),
        "covered_support_dimension_ids": sorted(
            item["dimension_id"] for item in dimension_rows if item["covered"]
        ),
        "missing_support_dimension_ids": missing,
        "dimensions": dimension_rows,
        "unlinked_evidence_ids": provenance["unlinked_evidence_ids"],
        "errors": [*provenance["errors"], *errors],
        "scientific_judgment_assigned": False,
    }


def audit_evidence_ablation(
    baseline_packet: Mapping[str, Any],
    ablated_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare evidence coverage before and after a controlled removal."""

    contract = derive_evidence_coverage_contract(baseline_packet)
    before = audit_evidence_coverage(
        baseline_packet, coverage_contract=contract
    )
    after = audit_evidence_coverage(ablated_packet, coverage_contract=contract)
    baseline_evidence = {
        str(item.get("evidence_id", ""))
        for item in _object_rows(baseline_packet.get("evidence_records"))
    }
    ablated_evidence = {
        str(item.get("evidence_id", ""))
        for item in _object_rows(ablated_packet.get("evidence_records"))
    }
    removed = sorted(baseline_evidence - ablated_evidence)
    added = sorted(ablated_evidence - baseline_evidence)
    before_dimensions = {
        item["dimension_id"]: item for item in before["dimensions"]
    }
    after_dimensions = {
        item["dimension_id"]: item for item in after["dimensions"]
    }
    lost = sorted(
        dimension_id
        for dimension_id, row in before_dimensions.items()
        if row["covered"] and not after_dimensions.get(dimension_id, {}).get("covered")
    )
    removed_links = [
        link
        for link in audit_evidence_provenance_closure(baseline_packet)["closed_links"]
        if link["evidence_record_id"] in set(removed)
    ]
    affected = sorted({str(item["claim_id"]) for item in removed_links})
    removed_relations = {str(item["support_relation"]) for item in removed_links}
    if added:
        classification = "INVALID_EVIDENCE_ADDITION"
    elif lost and any(
        before_dimensions[item]["minimum_support_relation"] == "DIRECT"
        for item in lost
    ):
        classification = "CRITICAL_DIRECT"
    elif lost:
        classification = "CRITICAL_REQUIRED_SUPPORT"
    elif not affected:
        classification = "UNLINKED_EVIDENCE_REMOVAL"
    elif removed_relations == {"CONTEXTUAL"}:
        classification = "CONTEXTUAL_NONCRITICAL"
    else:
        classification = "PARTIAL_REDUCTION"
    return {
        "classification": classification,
        "removed_evidence_ids": removed,
        "added_evidence_ids": added,
        "affected_support_dimension_ids": affected,
        "coverage_lost_dimension_ids": lost,
        "provenance_still_closed": after["provenance_status"]
        == "PROVENANCE_CLOSED",
        "coverage_before": before,
        "coverage_after": after,
        "scientific_truth_assigned": False,
    }


def enumerate_evidence_ablation_candidates(
    review_packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Classify every one-record removal from the authored coverage contract."""

    contract = derive_evidence_coverage_contract(review_packet)
    baseline = audit_evidence_coverage(review_packet, coverage_contract=contract)
    dimension_by_id = {
        str(item["dimension_id"]): item for item in baseline["dimensions"]
    }
    provenance = audit_evidence_provenance_closure(review_packet)
    evidence_ids = sorted(
        str(item.get("evidence_id", ""))
        for item in _object_rows(review_packet.get("evidence_records"))
    )
    rows: list[dict[str, Any]] = []
    for evidence_id in evidence_ids:
        removed_links = [
            item
            for item in provenance["closed_links"]
            if item["evidence_record_id"] == evidence_id
        ]
        affected = sorted({str(item["claim_id"]) for item in removed_links})
        lost: list[str] = []
        for dimension_id in affected:
            baseline_dimension = dimension_by_id.get(dimension_id, {})
            if not baseline_dimension.get("covered"):
                continue
            required_relation = baseline_dimension.get("minimum_support_relation")
            remaining_relations = [
                str(item["support_relation"])
                for item in provenance["closed_links"]
                if item["claim_id"] == dimension_id
                and item["evidence_record_id"] != evidence_id
            ]
            strongest_remaining = max(
                remaining_relations,
                key=lambda relation: SUPPORT_RELATION_STRENGTH[relation],
                default=None,
            )
            if strongest_remaining is None or (
                required_relation in SUPPORT_RELATION_STRENGTH
                and SUPPORT_RELATION_STRENGTH[strongest_remaining]
                < SUPPORT_RELATION_STRENGTH[str(required_relation)]
            ):
                lost.append(dimension_id)
        removed_relations = {
            str(item["support_relation"]) for item in removed_links
        }
        if lost and any(
            dimension_by_id[item].get("minimum_support_relation") == "DIRECT"
            for item in lost
        ):
            classification = "CRITICAL_DIRECT"
        elif lost:
            classification = "CRITICAL_REQUIRED_SUPPORT"
        elif not affected:
            classification = "UNLINKED_EVIDENCE_REMOVAL"
        elif removed_relations == {"CONTEXTUAL"}:
            classification = "CONTEXTUAL_NONCRITICAL"
        else:
            classification = "PARTIAL_REDUCTION"
        rows.append(
            {
                "evidence_id": evidence_id,
                "classification": classification,
                "affected_support_dimension_ids": affected,
                "coverage_lost_dimension_ids": sorted(lost),
                "removed_support_relations": sorted(removed_relations),
            }
        )
    return rows


def select_representative_evidence_ablations(
    review_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the three frozen sensitivity roles without family-specific IDs."""

    candidates = enumerate_evidence_ablation_candidates(review_packet)
    required = (
        "CRITICAL_DIRECT",
        "CONTEXTUAL_NONCRITICAL",
        "PARTIAL_REDUCTION",
    )
    selected: dict[str, dict[str, Any]] = {}
    for classification in required:
        row = next(
            (
                item
                for item in candidates
                if item["classification"] == classification
            ),
            None,
        )
        if row is not None:
            selected[classification] = row
    missing = [item for item in required if item not in selected]
    return {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "selected": selected,
        "missing_classifications": missing,
        "candidates": candidates,
        "scientific_truth_assigned": False,
    }


def _condition_review(
    review: Mapping[str, Any], condition: str
) -> Mapping[str, Any] | None:
    for row in _object_rows(review.get("condition_reviews")):
        if str(row.get("condition", "")) == condition:
            return row
    return None


def audit_evidence_ablation_response(
    ablation_audit: Mapping[str, Any],
    baseline_review: Mapping[str, Any],
    ablated_review: Mapping[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    """Detect confident equivalence after a required coverage loss.

    This is a response-sensitivity observation, not a scientific verdict.  It
    deliberately checks only atomic fields related to the claim types whose
    coverage was lost, so an unrelated question-wording change cannot hide
    false confidence.
    """

    baseline = _condition_review(baseline_review, condition)
    ablated = _condition_review(ablated_review, condition)
    if baseline is None or ablated is None:
        return {
            "status": "NOT_EVALUATED",
            "condition": condition,
            "false_confidence": False,
            "reason": "CONDITION_REVIEW_MISSING",
        }
    lost = {
        str(item)
        for item in ablation_audit.get("coverage_lost_dimension_ids", ()) or ()
    }
    before_dimensions = {
        str(item.get("dimension_id", "")): item
        for item in _object_rows(
            ablation_audit.get("coverage_before", {}).get("dimensions", ())
            if isinstance(ablation_audit.get("coverage_before"), Mapping)
            else ()
        )
    }
    relevant_fields = sorted(
        {
            field
            for dimension_id in lost
            for field in CLAIM_TYPE_TO_REVIEW_FIELDS.get(
                str(before_dimensions.get(dimension_id, {}).get("claim_type", "")),
                (),
            )
        }
    )
    changed = [
        field for field in relevant_fields if baseline.get(field) != ablated.get(field)
    ]
    baseline_positive = (
        baseline.get("eligibility_recommendation") == "ELIGIBLE"
        and all(baseline.get(field) is True for field in relevant_fields)
    )
    new_ambiguities = {
        str(item)
        for item in ablated.get("scientific_ambiguities", ()) or ()
    } - {
        str(item)
        for item in baseline.get("scientific_ambiguities", ()) or ()
    }
    ablated_positive = (
        ablated.get("eligibility_recommendation") == "ELIGIBLE"
        and all(ablated.get(field) is True for field in relevant_fields)
    )
    # Generic new ambiguities do not negate an otherwise fully positive
    # scientific-support judgment. Required coverage loss is acknowledged only
    # by a relevant atomic change or a non-ELIGIBLE recommendation.
    false_confidence = bool(lost) and baseline_positive and ablated_positive
    if not lost:
        status = "NO_REQUIRED_COVERAGE_LOSS"
    elif false_confidence:
        status = "FALSE_CONFIDENCE"
    elif not baseline_positive:
        status = "BASELINE_NOT_CONFIDENT"
    else:
        status = "RESPONDED_TO_REQUIRED_COVERAGE_LOSS"
    return {
        "status": status,
        "condition": condition,
        "coverage_lost_dimension_ids": sorted(lost),
        "relevant_atomic_fields": relevant_fields,
        "changed_relevant_atomic_fields": changed,
        "baseline_recommendation": baseline.get("eligibility_recommendation"),
        "ablated_recommendation": ablated.get("eligibility_recommendation"),
        "new_scientific_ambiguities": sorted(new_ambiguities),
        "false_confidence": false_confidence,
        "scientific_truth_assigned": False,
    }


__all__ = [
    "CLAIM_TYPE_TO_REVIEW_FIELDS",
    "SUPPORT_RELATION_STRENGTH",
    "audit_evidence_ablation",
    "audit_evidence_ablation_response",
    "audit_evidence_coverage",
    "audit_evidence_provenance_closure",
    "derive_evidence_coverage_contract",
    "enumerate_evidence_ablation_candidates",
    "select_representative_evidence_ablations",
]
