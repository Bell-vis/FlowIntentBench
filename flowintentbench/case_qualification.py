"""Scientific Case Qualification (SCQ) and pre-SCQ construction validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .case_design import (
    CaseConstructionMetadata,
    CaseConstructionMetadataValidator,
    ControlledCaseFamilyValidator,
    normalize_case_text,
    primary_case_type,
)
from .family_validator import canonicalize_operationalization_clause, resolved_dimension_invariance, validate_family_definition, validate_reference_space_semantics
from .representability import validate_representability_contract


SCQ_STATUSES = frozenset({"SCQ_PASS", "SCQ_REVISE", "SCQ_REJECT"})


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _question(case_input: Any) -> str:
    return str(_value(case_input, "scientific_question", ""))


def _condition(case_id: str) -> str | None:
    match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", case_id, re.I)
    return f"{match.group(1).upper()}-{match.group(2).upper()}" if match else None


def _target(value: Any) -> str:
    text = normalize_case_text(str(value or ""))
    text = re.sub(r"\b(?:in|for|within|of)\s+(?:the\s+)?[a-z0-9_ -]+\b", "", text)
    text = re.sub(r"\b(?:o[123]|f[12])\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _question_mentions_concrete_clause(question: str, statement: str) -> bool:
    """Detect a concrete fixed choice without requiring wording identity."""

    q = normalize_case_text(question).casefold()
    s = normalize_case_text(statement).casefold()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if numbers and any(number in q for number in numbers):
        return True
    salient = [token for token in re.findall(r"[a-z][a-z0-9_-]{3,}", s) if token not in {"the", "this", "that", "with", "from", "into", "using", "stored", "selected", "region", "locations"}]
    return len([token for token in salient if token in q]) >= 2


def _responsibility_clauses(family: Mapping[str, Any], metadata: CaseConstructionMetadata) -> dict[str, Any]:
    """Normalize the frozen semantic responsibility contract."""

    # The canonical production contract is authoritative.  Legacy aliases are
    # parsed only for historical synthetic fixtures and are normalized here.
    contract = family.get("responsibility_contract")
    if "responsibility_contract" not in family:
        contract = metadata.model_dump(mode="json").get("responsibility_contract")
    contract = contract or {}
    if not isinstance(contract, Mapping):
        contract = {}
    def clauses(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, Mapping):
            # Accept either one clause object or the compact
            # ``{dimension_id: clause}`` representation.
            if any(key in value for key in ("dimension_id", "dimension", "canonical_id", "normalized_meaning", "statement")):
                value = [dict(value)]
            else:
                value = [dict(item, dimension_id=str(dimension)) if isinstance(item, Mapping) else {"dimension_id": str(dimension), "normalized_meaning": str(item)} for dimension, item in value.items()]
        result = []
        for item in value or ():
            if not isinstance(item, Mapping):
                continue
            result.append({
                "dimension_id": str(item.get("dimension_id", item.get("dimension", ""))),
                "canonical_id": str(item.get("canonical_id", item.get("canonical", ""))),
                "normalized_meaning": normalize_case_text(str(item.get("normalized_meaning", item.get("statement", item.get("meaning", ""))))),
                "question_visibility": str(item.get("question_visibility", item.get("visible_expression_policy", {}).get("mode", "REQUIRED"))).upper() if isinstance(item.get("visible_expression_policy", {}), Mapping) else str(item.get("question_visibility", "REQUIRED")).upper(),
                "must_remain_open": bool(item.get("must_remain_open", False)),
                "required_semantic_anchors": [normalize_case_text(str(x)) for x in item.get("required_semantic_anchors", ()) or ()],
                "allowed_visible_realizations": [normalize_case_text(str(x)) for x in item.get("allowed_visible_realizations", ()) or ()],
            })
        return result
    target = contract.get("scientific_target", contract.get("target_clause", {}))
    if not isinstance(target, Mapping):
        target = {"normalized_meaning": str(target)}
    finding = contract.get("finding_responsibility", contract.get("finding_demand_clause", {}))
    if not isinstance(finding, Mapping):
        finding = {"normalized_meaning": str(finding)}
    return {
        "target_clause": {
            "canonical_id": str(target.get("canonical_id", target.get("canonical", ""))),
            "normalized_meaning": normalize_case_text(str(target.get("normalized_meaning", target.get("statement", target.get("meaning", ""))))),
            "required_semantic_anchors": [normalize_case_text(str(x)) for x in target.get("required_semantic_anchors", ()) or ()],
            "allowed_visible_realizations": [normalize_case_text(str(x)) for x in target.get("allowed_visible_realizations", ()) or ()],
        },
        "resolved_operationalization_clauses": clauses(contract.get("resolved_operationalization_clauses", contract.get("resolved_clauses", ()))),
        "unresolved_operationalization_dimensions": [
            str(x.get("dimension_id", x.get("dimension", ""))) if isinstance(x, Mapping) else str(x)
            for x in contract.get("unresolved_operationalization_dimensions", ()) or ()
        ],
        "unresolved_operationalization_clauses": clauses(contract.get("unresolved_operationalization_dimensions", contract.get("unresolved_operationalization_clauses", contract.get("unresolved_clauses", ())))),
        "finding_demand_clause": {
            "canonical_id": str(finding.get("canonical_id", finding.get("canonical", ""))),
            "normalized_meaning": normalize_case_text(str(
                finding.get("normalized_demand", finding.get("normalized_meaning", finding.get("statement", finding.get("meaning", ""))))
            )),
            "finding_mode": str(finding.get("finding_mode", "")),
            "required_responsibility_mode": (str(finding.get("required_responsibility_mode")) if finding.get("required_responsibility_mode") else None),
            "forbidden_responsibility_modes": [str(x) for x in finding.get("forbidden_responsibility_modes", ()) or ()],
            "required_semantic_anchors": [normalize_case_text(str(x)) for x in finding.get("required_semantic_anchors", ()) or ()],
            "allowed_visible_realizations": [normalize_case_text(str(x)) for x in finding.get("allowed_visible_realizations", ()) or ()],
        },
    }


def _question_realizes(question: str, clause: Mapping[str, Any]) -> bool:
    q = normalize_case_text(question).casefold()
    candidates = [str(clause.get("normalized_meaning", "")), *[str(x) for x in clause.get("allowed_visible_realizations", ()) or ()]]
    anchors = [normalize_case_text(str(item)).casefold() for item in clause.get("required_semantic_anchors", ()) or ()]
    # Canonical contracts carry their own semantic anchors.  Once present,
    # they are authoritative and we do not fall back to a loose token-count
    # heuristic that could be satisfied by unrelated clauses in the question.
    if anchors:
        # Finding-responsibility anchors are scoped to the reporting demand.
        # A question may mention a coordinate while defining an
        # operationalization, but that must not satisfy F1's requirement to
        # report the coordinate.  Keep this rule contract-driven: it applies
        # only when the frozen anchor list explicitly starts with a reporting
        # verb and does not infer a value type from the prose.
        reporting_anchor = next((anchor for anchor in anchors if anchor in {"report", "characterize"}), None)
        if reporting_anchor:
            marker = q.find(reporting_anchor)
            if marker < 0:
                return False
            demand_tail = q[marker:]
            return all(anchor and anchor in demand_tail for anchor in anchors)
        return all(anchor and anchor in q for anchor in anchors)
    for candidate in candidates:
        candidate = normalize_case_text(candidate).casefold()
        if candidate and (candidate in q or _question_mentions_concrete_clause(question, candidate)):
            return True
    return False


def _validate_responsibility_contract_schema(value: Any, metadata: CaseConstructionMetadata) -> tuple[bool, list[str], bool]:
    """Validate the canonical contract without inferring semantics from prose.

    The final boolean identifies the explicitly supported legacy shape used by
    old synthetic tests.  Candidate artifacts must use the canonical shape.
    """

    if not isinstance(value, Mapping):
        return False, ["RESPONSIBILITY_CONTRACT_MISSING"], False
    canonical_keys = {"scientific_target", "resolved_operationalization_clauses", "unresolved_operationalization_dimensions", "finding_responsibility"}
    legacy = "target_clause" in value or "finding_demand_clause" in value
    if not canonical_keys.issubset(value):
        return (True, [], True) if legacy else (False, ["RESPONSIBILITY_CONTRACT_INVALID"], False)
    failures: list[str] = []
    target = value.get("scientific_target")
    if not isinstance(target, Mapping) or not target.get("canonical_id") or not target.get("normalized_meaning") or not isinstance(target.get("required_semantic_anchors"), list):
        failures.append("RESPONSIBILITY_CONTRACT_INVALID")
    resolved = value.get("resolved_operationalization_clauses")
    unresolved = value.get("unresolved_operationalization_dimensions")
    finding = value.get("finding_responsibility")
    if not isinstance(resolved, list) or not isinstance(unresolved, list) or not isinstance(finding, Mapping):
        failures.append("RESPONSIBILITY_CONTRACT_INVALID")
    else:
        seen: set[str] = set()
        for clause in resolved:
            if not isinstance(clause, Mapping) or not clause.get("dimension_id") or not clause.get("canonical_id") or not clause.get("normalized_meaning") or clause.get("question_visibility") not in {"REQUIRED", "OPTIONAL", "IMPLICIT_ALLOWED"} or not isinstance(clause.get("required_semantic_anchors"), list):
                failures.append("RESPONSIBILITY_CONTRACT_INVALID")
            else:
                dimension = str(clause["dimension_id"])
                if dimension in seen:
                    failures.append("RESPONSIBILITY_CONTRACT_INVALID")
                seen.add(dimension)
        unresolved_ids: set[str] = set()
        for clause in unresolved:
            if not isinstance(clause, Mapping) or not clause.get("dimension_id") or not clause.get("normalized_meaning") or clause.get("must_remain_open") is not True:
                failures.append("RESPONSIBILITY_CONTRACT_INVALID")
            else:
                unresolved_ids.add(str(clause["dimension_id"]))
        if seen & unresolved_ids:
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
        if finding.get("finding_mode") not in {"F1", "F2"} or not finding.get("normalized_demand") or not isinstance(finding.get("required_semantic_anchors"), list):
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
        allowed_modes = {"OPEN_FINDING_SELECTION", "SELECT_PRINCIPAL_FINDINGS", "EXHAUSTIVE_FIXED_FINDING_LIST"}
        forbidden_modes = finding.get("forbidden_responsibility_modes", [])
        required_mode = finding.get("required_responsibility_mode")
        if not isinstance(forbidden_modes, list) or any(str(item) not in allowed_modes for item in forbidden_modes):
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
        if required_mode is not None and (not isinstance(required_mode, str) or required_mode not in allowed_modes):
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
        if finding.get("finding_mode") == "F1" and "OPEN_FINDING_SELECTION" not in forbidden_modes:
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
        if finding.get("finding_mode") == "F2" and (required_mode != "SELECT_PRINCIPAL_FINDINGS" or "EXHAUSTIVE_FIXED_FINDING_LIST" not in forbidden_modes):
            failures.append("RESPONSIBILITY_CONTRACT_INVALID")
    return not failures, list(dict.fromkeys(failures)), False


def _question_fixes_unresolved_dimension(question: str, dimension: str) -> bool:
    """Minimal semantic diagnostic for an unresolved choice being fixed.

    The contract remains the authority; these anchors only detect a concrete
    choice that contradicts ``must_remain_open`` and are not used to infer a
    resolved operationalization.
    """

    q = normalize_case_text(question).casefold()
    if dimension == "criterion":
        return any(token in q for token in ("maximum", "maximizing", "minimum", "minimizing", "peak", "percentile", "threshold", "criterion of", "criterion:") )
    if dimension in {"property_measure", "comparison_or_normalization"}:
        # ``mean spatial location`` is a resolved representation clause, not
        # a choice of property measure.  Restrict this diagnostic to phrases
        # that actually name a scalar/vector measure so an O2 question is not
        # rejected merely because it reports a centroid/mean location.
        return any(token in q for token in (
            "coefficient of variation", "interdecile", "mean speed",
            "mean velocity", "mean density", "mean turbulent",
            "standard deviation", "normalized by", "relative spread",
            "peak speed", "peak velocity", "peak density",
        ))
    return False


def _question_has_open_finding_selection(question: str) -> bool:
    """Detect an explicitly open/additional finding responsibility."""

    q = normalize_case_text(question).casefold()
    return bool(re.search(
        r"\b(?:any|whatever|additional|other)\s+(?:scientifically\s+relevant\s+)?findings?\b"
        r"|\bfindings?\s+you\s+(?:consider|deem)\s+(?:important|relevant)\b",
        q,
    ))


def _question_has_exhaustive_fixed_finding_list(question: str) -> bool:
    """Detect a question that closes an F2 demand to an exhaustive list."""

    q = normalize_case_text(question).casefold()
    return bool(re.search(
        r"\bexactly\s+(?:these\s+)?(?:required\s+)?findings?\b"
        r"|\breport\s+no\s+other\s+findings?\b"
        r"|\breport\s+only\s+(?:the\s+)?(?:following|these)\b"
        r"|\bno\s+other\s+findings?\b",
        q,
    ))


@dataclass(frozen=True)
class ControlledConstructionAudit:
    status: str
    case_id: str
    condition: str | None
    checks: Mapping[str, str]
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blockers"] = list(self.blockers)
        return value


def validate_controlled_family_construction(
    case_input: Mapping[str, Any] | Any,
    metadata: CaseConstructionMetadata | Mapping[str, Any],
    ground_truth: Mapping[str, Any] | Any,
    family: Mapping[str, Any],
) -> ControlledConstructionAudit:
    """Run the deterministic construction gate that must precede SCQ."""

    metadata_obj = metadata if isinstance(metadata, CaseConstructionMetadata) else CaseConstructionMetadata.model_validate(metadata)
    case_id = str(metadata_obj.case_id)
    condition = _condition(case_id)
    checks: dict[str, str] = {}
    blockers: list[str] = []
    if str(metadata_obj.dataset_id) != str(family.get("dataset_id")):
        checks["dataset_identity"] = "FAIL"; blockers.append("DATASET_IDENTITY_MISMATCH")
    else:
        checks["dataset_identity"] = "PASS"
    family_errors = validate_family_definition(family)
    checks["family_definition"] = "PASS" if not family_errors else "FAIL"
    blockers.extend(family_errors)
    gt = ground_truth if isinstance(ground_truth, Mapping) else ground_truth.model_dump(mode="json")
    branches = gt.get("acceptable_operationalizations", []) if isinstance(gt, Mapping) else []
    principal = [item.value for item in metadata_obj.principal_operationalization_dimensions]
    unresolved = [item.value for item in metadata_obj.unresolved_operationalization_dimensions]
    branch_maps = []
    for branch in branches:
        decisions = branch.get("decisions", []) if isinstance(branch, Mapping) else []
        clauses: dict[str, Any] = {}
        for item in decisions:
            if not isinstance(item, Mapping):
                continue
            dimension = str(item.get("dimension", ""))
            if not dimension:
                continue
            if dimension in unresolved:
                continue
            clause = item.get("resolved_clause", item.get("clause"))
            if not isinstance(clause, Mapping):
                statement = item.get("statement", item.get("description", item.get("value", "")))
                clause = canonicalize_operationalization_clause(dimension, str(statement))
            else:
                clause = dict(clause)
            clauses[dimension] = clause
        branch_maps.append({"branch_id": branch.get("operationalization_id") if isinstance(branch, Mapping) else None, "resolved_operationalization_clauses": clauses, "resolved_dimensions": list(clauses)})
    invariance = resolved_dimension_invariance(branch_maps, principal_dimensions=principal, unresolved_dimensions=unresolved, family_id=str(family.get("family_id")), case_id=case_id)
    checks["resolved_clause_invariance"] = "PASS" if invariance.get("status") == "PASS" else "FAIL"
    if checks["resolved_clause_invariance"] != "PASS":
        blockers.append("RESOLVED_CLAUSE_INVARIANCE")
        if invariance.get("failure_code"):
            blockers.append(str(invariance["failure_code"]))
    # Unresolved dimensions are intentionally instantiated per accepted O
    # branch in the GT; their openness is checked against the question text
    # in SCQ-Q2, not by rejecting the branch materialization itself.
    checks["unresolved_dimension_openness"] = "PASS"
    reference = validate_reference_space_semantics({"accepted": family.get("candidate_reference_operationalizations", []), "omitted_combinations": "UNENUMERATED", "require_anchor_schema": bool(family.get("candidate_reference_operationalizations"))})
    checks["reference_space_schema"] = "PASS" if reference.get("status") in {"PASS", "NOT_EVALUABLE_NO_REFERENCE_ANCHOR"} else "FAIL"
    if checks["reference_space_schema"] == "FAIL": blockers.append("REFERENCE_SPACE_SCHEMA")
    checks["finding_requirement_schema"] = "PASS" if gt.get("findings_by_operationalization") else "FAIL"
    if checks["finding_requirement_schema"] == "FAIL": blockers.append("FINDING_REQUIREMENT_SCHEMA")
    expected = primary_case_type(metadata_obj)
    if condition and expected and condition != expected:
        checks["condition_contract"] = "FAIL"; blockers.append("CONDITION_CONFOUND")
    else:
        checks["condition_contract"] = "PASS"
    status = "PASS" if not blockers else "FAIL"
    return ControlledConstructionAudit(status, case_id, condition, checks, tuple(dict.fromkeys(blockers)))


def validate_sparse_controlled_family_construction(
    packet: Mapping[str, Any],
    curator_gate_validation: Mapping[str, Any],
    approved_condition_ids: Sequence[str],
    case_records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    evaluation_representability_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate only the exact curator-approved controlled-family subset.

    This is a derived execution artifact, not a case or Dataset schema. The
    function requires an explicit representability contract and an already
    validated curator gate. It never fills either input from a default and it
    does not execute SCQ.
    """

    from .kitchen_condition_eligibility import (
        validate_materialized_gt_consistency,
        validate_o1_f2_effective_o,
    )
    from .lifecycle import canonical_curator_packet_sha256

    packet_sha256 = canonical_curator_packet_sha256(packet)
    gate_confirmed = (
        curator_gate_validation.get("status") == "PASS"
        and curator_gate_validation.get("grounding_curator_gate_status") == "CONFIRMED"
        and curator_gate_validation.get("packet_hash_matches") is True
        and curator_gate_validation.get("packet_sha256") == packet_sha256
        and curator_gate_validation.get("artifact_sha256") == packet_sha256
        and curator_gate_validation.get("proxy_generated") is False
    )
    if not gate_confirmed:
        return {
            "dataset_id": packet.get("dataset_id"),
            "family_id": packet.get("family_id"),
            "curator_packet_sha256": packet_sha256,
            "approved_condition_ids": [],
            "validated_condition_set": [],
            "validation_results": [],
            "failed_invariants": ["GROUNDING_CURATOR_EXACT_CONFIRMATION_REQUIRED"],
            "overall_status": "NOT_RUN",
            "status": "NOT_RUN",
            "controlled_family_construction_validated": False,
            "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
            "dataset_schema_mutated": False,
        }

    packet_rows = {
        str(row.get("condition", "")).strip().upper(): row
        for row in packet.get("conditions", ())
        if isinstance(row, Mapping) and str(row.get("condition", "")).strip()
    }
    requested = [str(item).strip().upper() for item in approved_condition_ids]
    requested_set = set(requested)
    failed: list[str] = []
    if not requested:
        failed.append("EMPTY_APPROVED_CONDITION_SET")
    if len(requested) != len(requested_set):
        failed.append("DUPLICATE_APPROVED_CONDITION_ID")
    for condition in requested:
        row = packet_rows.get(condition)
        eligibility = (
            row.get("deterministic_condition_eligibility", {})
            if isinstance(row, Mapping)
            else {}
        )
        if not isinstance(row, Mapping):
            failed.append(f"{condition}:CONDITION_ABSENT_FROM_CURATOR_PACKET")
        elif not isinstance(eligibility, Mapping) or str(
            eligibility.get("eligibility", "")
        ).strip().upper() != "ELIGIBLE":
            failed.append(f"{condition}:CONDITION_NOT_CURATOR_ELIGIBLE")

    if isinstance(case_records, Mapping):
        records = [value for value in case_records.values() if isinstance(value, Mapping)]
    else:
        records = [value for value in case_records if isinstance(value, Mapping)]
    record_by_case_id = {
        str((record.get("metadata") or {}).get("case_id", record.get("case_id", ""))): record
        for record in records
        if isinstance(record.get("metadata") or {}, Mapping)
    }

    ordered_conditions = [condition for condition in packet_rows if condition in requested_set]
    selected_rows = [packet_rows[condition] for condition in ordered_conditions]
    selected_records: list[tuple[str, Mapping[str, Any], Mapping[str, Any] | None]] = []
    for condition, row in zip(ordered_conditions, selected_rows):
        case_id = str(row.get("case_id", ""))
        selected_records.append((condition, row, record_by_case_id.get(case_id)))

    metadata_rows = [
        record.get("metadata", {})
        for _, _, record in selected_records
        if isinstance(record, Mapping) and isinstance(record.get("metadata"), Mapping)
    ]
    principal_signatures = {
        tuple(str(item) for item in metadata.get("principal_operationalization_dimensions", ()))
        for metadata in metadata_rows
    }
    target_signatures = {
        normalize_case_text(str(metadata.get("scientific_target", "")))
        for metadata in metadata_rows
        if str(metadata.get("scientific_target", "")).strip()
    }
    packet_target = normalize_case_text(str(packet.get("scientific_target", "")))
    principal_status = "PASS" if len(metadata_rows) == len(requested) and len(principal_signatures) == 1 else "FAIL"
    target_status = "PASS" if (
        len(metadata_rows) == len(requested)
        and len(target_signatures) == 1
        and (not packet_target or target_signatures == {packet_target})
    ) else "FAIL"
    if principal_status != "PASS":
        failed.append("PRINCIPAL_DIMENSION_INVARIANCE")
    if target_status != "PASS":
        failed.append("SCIENTIFIC_TARGET_INVARIANCE")

    reference_anchors: list[dict[str, Any]] = []
    for _condition_id, row, _record in selected_records:
        case_id = str(row.get("case_id", ""))
        for anchor in row.get("authored_reference_o_anchors", ()) or ():
            if not isinstance(anchor, Mapping) or not str(anchor.get("operationalization_id", "")).strip():
                continue
            reference_anchors.append({
                "reference_o_id": f"{case_id}::{anchor['operationalization_id']}",
                "family_id": str(packet.get("family_id", "")),
                "source_case_id": case_id,
                "provenance": "AUTHORED_GT_REFERENCE",
                "validity_status": "ENUMERATED_VALID",
            })
    reference_validation = validate_reference_space_semantics({
        "accepted": reference_anchors,
        "explicit_rejected": [],
        "omitted_combinations": "UNENUMERATED",
        "require_anchor_schema": True,
    })
    if not reference_anchors or reference_validation.get("status") != "PASS":
        failed.append("REFERENCE_SPACE_SEMANTICS")

    principal_dimensions = list(next(iter(principal_signatures), ()))
    o2_metadata = next((
        record.get("metadata", {})
        for condition, _row, record in selected_records
        if condition == "O2-F1" and isinstance(record, Mapping)
    ), {})
    family = {
        "family_id": packet.get("family_id"),
        "dataset_id": packet.get("dataset_id"),
        "concept_id": packet.get("concept_id"),
        "scientific_target": packet.get("scientific_target"),
        "principal_dimensions": principal_dimensions,
        "o2_unresolved_dimensions": list(
            o2_metadata.get("unresolved_operationalization_dimensions", ())
        ) if isinstance(o2_metadata, Mapping) else [],
        "candidate_reference_operationalizations": reference_anchors,
        "controlled_case_records": [
            {
                "case_id": row.get("case_id"),
                "condition": condition,
                "dataset_id": packet.get("dataset_id"),
            }
            for condition, row, _record in selected_records
        ],
    }
    family_errors = validate_family_definition(family)
    failed.extend(f"FAMILY_DEFINITION:{error}" for error in family_errors)

    representability = validate_representability_contract(
        evaluation_representability_contract,
        allow_default=False,
    )
    if representability.get("status") != "PASS":
        failed.extend(
            f"EVALUATION_REPRESENTABILITY:{code}"
            for code in representability.get("failure_codes", ())
        )

    validation_results: list[dict[str, Any]] = []
    validated_conditions: list[str] = []
    for condition, packet_row, record in selected_records:
        case_id = str(packet_row.get("case_id", ""))
        metadata = record.get("metadata") if isinstance(record, Mapping) else None
        case_input = record.get("case_input") if isinstance(record, Mapping) else None
        ground_truth = record.get("ground_truth") if isinstance(record, Mapping) else None
        context_selection = record.get("context_selection") if isinstance(record, Mapping) else None
        canonical_exists = all(isinstance(value, Mapping) and bool(value) for value in (
            metadata, case_input, ground_truth, context_selection,
        )) and str(metadata.get("case_id", "")) == case_id
        row_blockers: list[str] = []
        if canonical_exists:
            try:
                CaseConstructionMetadataValidator.validate(
                    str(case_input.get("scientific_question", "")),
                    metadata,
                    context_selection=context_selection,
                )
                metadata_validation = {"status": "PASS", "errors": []}
            except (TypeError, ValueError) as exc:
                metadata_validation = {
                    "status": "FAIL",
                    "errors": [str(exc)],
                }
                row_blockers.append("CASE_CONSTRUCTION_METADATA")
            construction = validate_controlled_family_construction(
                case_input, metadata, ground_truth, family
            ).to_dict()
        else:
            metadata_validation = {
                "status": "FAIL",
                "errors": ["canonical case artifacts are incomplete"],
            }
            construction = {
                "status": "FAIL",
                "case_id": case_id,
                "condition": condition,
                "checks": {},
                "blockers": ["CANONICAL_CASE_MISSING"],
            }
        if construction["status"] != "PASS":
            row_blockers.extend(str(item) for item in construction.get("blockers", ()))

        packet_target_audit = packet_row.get("deterministic_condition_eligibility", {})
        packet_target_audit = (
            packet_target_audit.get("target_fidelity", {})
            if isinstance(packet_target_audit, Mapping)
            else {}
        )
        condition_target_status = (
            "PASS" if isinstance(packet_target_audit, Mapping)
            and packet_target_audit.get("status") == "PASS" else "FAIL"
        )
        if condition_target_status != "PASS":
            row_blockers.append("TARGET_FIDELITY")

        materialization = packet_row.get("materialized_g_of_o_summary", {})
        gt_consistency = (
            validate_materialized_gt_consistency(materialization, ground_truth)
            if canonical_exists and isinstance(materialization, Mapping)
            else {"status": "FAIL", "failures": ["MATERIALIZATION_OR_GT_MISSING"]}
        )
        if gt_consistency.get("status") != "PASS":
            row_blockers.append("DATA_GROUND_TRUTH_CONSISTENCY")
        if not canonical_exists:
            row_blockers.append("CANONICAL_CASE_MISSING")
        if representability.get("status") != "PASS":
            row_blockers.append("EVALUATION_REPRESENTABILITY_CONTRACT")

        row_status = "PASS" if not row_blockers else "FAIL"
        if row_status == "PASS":
            validated_conditions.append(condition)
        else:
            failed.extend(f"{condition}:{item}" for item in row_blockers)
        validation_results.append({
            "condition": condition,
            "case_id": case_id,
            "status": row_status,
            "canonical_case_exists": canonical_exists,
            "target_fidelity": condition_target_status,
            "case_construction_metadata_validation": metadata_validation,
            "construction_audit": construction,
            "gt_materialization_consistency": gt_consistency,
            "evaluation_representability": representability,
            "blockers": list(dict.fromkeys(row_blockers)),
        })

    if set(ordered_conditions) != requested_set:
        failed.append("APPROVED_CONDITION_SET_NOT_FULLY_RESOLVED")

    records_by_condition = {
        condition: record
        for condition, _row, record in selected_records
        if isinstance(record, Mapping)
    }
    pairwise_results: list[dict[str, Any]] = []
    for left_condition, right_condition, axis in (
        ("O1-F1", "O2-F1", "operationalization"),
        ("O1-F1", "O1-F2", "finding"),
    ):
        if not {left_condition, right_condition} <= requested_set:
            continue
        left_record = records_by_condition.get(left_condition, {})
        right_record = records_by_condition.get(right_condition, {})
        try:
            ControlledCaseFamilyValidator.validate_pair(
                left_record.get("metadata", {}),
                right_record.get("metadata", {}),
                axis=axis,
                left_context_selection=left_record.get("context_selection"),
                right_context_selection=right_record.get("context_selection"),
            )
            pairwise_results.append({
                "left_condition": left_condition,
                "right_condition": right_condition,
                "axis": axis,
                "status": "PASS",
                "errors": [],
            })
        except (TypeError, ValueError) as exc:
            code = f"CONTROLLED_{axis.upper()}_PAIR_DRIFT"
            failed.append(code)
            pairwise_results.append({
                "left_condition": left_condition,
                "right_condition": right_condition,
                "axis": axis,
                "status": "FAIL",
                "errors": [str(exc)],
            })

    o1_rows = {condition: row for condition, row, _record in selected_records}
    if {"O1-F1", "O1-F2"} <= requested_set:
        o1_f2_inheritance = validate_o1_f2_effective_o(
            o1_rows["O1-F1"].get("materialized_g_of_o_summary", {}),
            o1_rows["O1-F2"].get("materialized_g_of_o_summary", {}),
        )
        if o1_f2_inheritance.get("status") != "PASS":
            failed.extend(
                f"O1_F2_INHERITANCE:{item}"
                for item in o1_f2_inheritance.get("failures", ())
            )
    else:
        o1_f2_inheritance = {"status": "NOT_APPLICABLE", "failures": []}

    failed = list(dict.fromkeys(failed))
    overall = "PASS" if not failed and set(validated_conditions) == requested_set else "FAIL"
    return {
        "dataset_id": packet.get("dataset_id"),
        "family_id": packet.get("family_id"),
        "curator_packet_sha256": packet_sha256,
        "approved_condition_ids": ordered_conditions,
        "validated_condition_set": validated_conditions,
        "validation_results": validation_results,
        "family_definition_validation": {
            "status": "PASS" if not family_errors else "FAIL",
            "errors": family_errors,
        },
        "principal_dimension_invariance": principal_status,
        "scientific_target_invariance": target_status,
        "reference_space_validation": reference_validation,
        "controlled_family_pairwise_validation": pairwise_results,
        "o1_f2_inheritance": o1_f2_inheritance,
        "evaluation_representability": representability,
        "failed_invariants": failed,
        "overall_status": overall,
        "status": overall,
        "controlled_family_construction_validated": overall == "PASS",
        "OFFICIAL_SCQ_EXECUTED_COUNT": 0,
        "dataset_schema_mutated": False,
    }


@dataclass(frozen=True)
class SCQResult:
    case_id: str
    status: str
    q1_case_concept_fidelity: Mapping[str, Any]
    q2_responsibility_integrity: Mapping[str, Any]
    q3_evaluation_representability: Mapping[str, Any]
    construction_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in SCQ_STATUSES:
            raise ValueError(f"invalid SCQ status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qualify_case(
    case_input: Mapping[str, Any] | Any,
    metadata: CaseConstructionMetadata | Mapping[str, Any],
    family: Mapping[str, Any],
    *,
    capability_result: Mapping[str, Any] | None = None,
    ground_truth: Mapping[str, Any] | Any | None = None,
) -> SCQResult:
    """Evaluate the exactly-three SCQ checks without model answers or scores."""

    metadata_obj = metadata if isinstance(metadata, CaseConstructionMetadata) else CaseConstructionMetadata.model_validate(metadata)
    case_id = str(metadata_obj.case_id)
    gt = ground_truth or {"acceptable_operationalizations": family.get("candidate_reference_operationalizations", []), "findings_by_operationalization": family.get("finding_schema", {})}
    construction = validate_controlled_family_construction(case_input, metadata_obj, gt, family)
    q1_blockers: list[str] = []
    if str(metadata_obj.case_family_id) != str(family.get("family_id")) and str(metadata_obj.case_family_id) != str(family.get("concept_id")):
        q1_blockers.append("CASE_CONCEPT_DRIFT")
    if str(metadata_obj.dataset_id) != str(family.get("dataset_id")):
        q1_blockers.append("CASE_DATA_MISMATCH")
    if capability_result and capability_result.get("capability_status") in {"REQUIRED_CAPABILITY_MISSING", "REQUIRED_CAPABILITY_UNKNOWN"}:
        q1_blockers.append("CASE_DATA_MISMATCH")
    q1 = {"status": "PASS" if not q1_blockers else "FAIL", "check": "Q1_CASE_CONCEPT_FIDELITY", "failure_codes": sorted(set(q1_blockers))}
    q2_blockers: list[str] = []
    responsibility_contract = family.get("responsibility_contract")
    if "responsibility_contract" not in family:
        responsibility_contract = metadata_obj.responsibility_contract
    responsibility_contract_present = isinstance(responsibility_contract, Mapping)
    if not responsibility_contract_present:
        q2_blockers.append("RESPONSIBILITY_CONTRACT_MISSING")
    contract_schema_valid, contract_schema_failures, legacy_contract = _validate_responsibility_contract_schema(responsibility_contract, metadata_obj)
    q2_blockers.extend(contract_schema_failures)
    family_target = family.get("model_visible_scientific_target") or family.get("scientific_target")
    if (not responsibility_contract_present) and family_target and _target(metadata_obj.scientific_target) not in _target(family_target) and _target(family_target) not in _target(metadata_obj.scientific_target):
        q2_blockers.append("TARGET_DRIFT")
    if construction.condition and primary_case_type(metadata_obj) and construction.condition != primary_case_type(metadata_obj):
        q2_blockers.append("CONDITION_CONFOUND")
    question = _question(case_input)
    responsibility = _responsibility_clauses(
        {**family, "responsibility_contract": responsibility_contract}, metadata_obj
    )
    canonical_target = responsibility["target_clause"]
    if (not legacy_contract and canonical_target.get("normalized_meaning")
            and normalize_case_text(str(canonical_target["normalized_meaning"]))
            != normalize_case_text(metadata_obj.scientific_target)):
        q2_blockers.append("TARGET_DRIFT")
    if canonical_target.get("normalized_meaning") and not _question_realizes(question, canonical_target):
        # A contract with a target clause is authoritative; lexical target
        # checks below are retained only for legacy families without it.
        q2_blockers.append("TARGET_DRIFT")
    unresolved_contract_dimensions = set(responsibility["unresolved_operationalization_dimensions"])
    if contract_schema_valid and not legacy_contract:
        principal_dimensions = {item.value for item in metadata_obj.principal_operationalization_dimensions}
        expected_unresolved = {item.value for item in metadata_obj.unresolved_operationalization_dimensions}
        resolved_clauses = responsibility.get("resolved_operationalization_clauses", [])
        resolved_by_dimension = {str(item.get("dimension_id")): item for item in resolved_clauses}
        for dimension in sorted(principal_dimensions - expected_unresolved):
            clause = resolved_by_dimension.get(dimension)
            if clause is None:
                q2_blockers.append("RESOLVED_RESPONSIBILITY_MISSING")
                continue
            if clause.get("question_visibility") == "REQUIRED" and not _question_realizes(question, clause):
                q2_blockers.append("RESOLVED_DIMENSION_DRIFT")
        # A canonical clause whose authored value no longer corresponds to the
        # model-visible question is a drift, not an unresolved-dimension leak.
        expected_constraints = {
            str(getattr(_value(item, "category", ""), "value", _value(item, "category", ""))): str(_value(item, "statement", ""))
            for item in metadata_obj.explicit_method_constraints
        }
        dimension_aliases = {"aggregation_or_representation": "analysis_procedure"}
        for dimension, clause in resolved_by_dimension.items():
            expected_statement = expected_constraints.get(dimension_aliases.get(dimension, dimension), "")
            if expected_statement and normalize_case_text(str(clause.get("normalized_meaning", ""))) != normalize_case_text(expected_statement):
                q2_blockers.append("RESOLVED_DIMENSION_DRIFT")
    # A resolved clause is required to be visible in the question.  Its
    # presence is therefore evidence of fidelity, not a leakage failure.
    contract_unresolved = unresolved_contract_dimensions
    if contract_unresolved and metadata_obj.operationalization_responsibility.value in {"partially_specified", "model_selected"}:
        unresolved_clause_mentions = any(
            clause.get("dimension_id") in contract_unresolved and _question_realizes(question, clause)
            for clause in responsibility.get("unresolved_operationalization_clauses", [])
        )
        if unresolved_clause_mentions:
            q2_blockers.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")
        for clause in responsibility.get("unresolved_operationalization_clauses", []):
            if clause.get("must_remain_open") and _question_fixes_unresolved_dimension(
                question, str(clause.get("dimension_id", ""))
            ):
                q2_blockers.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")
    finding_clause = responsibility["finding_demand_clause"]
    expected_finding_mode = "F2" if metadata_obj.finding_openness.value == "open" else "F1"
    if not legacy_contract and finding_clause.get("finding_mode") != expected_finding_mode:
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    if (not legacy_contract and finding_clause.get("normalized_meaning")
            and normalize_case_text(str(finding_clause["normalized_meaning"]))
            != normalize_case_text(metadata_obj.finding_goal)):
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    if finding_clause.get("normalized_meaning") and not _question_realizes(question, finding_clause):
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    forbidden_modes = {str(item) for item in finding_clause.get("forbidden_responsibility_modes", ()) or ()}
    if "OPEN_FINDING_SELECTION" in forbidden_modes and _question_has_open_finding_selection(question):
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    if "EXHAUSTIVE_FIXED_FINDING_LIST" in forbidden_modes and _question_has_exhaustive_fixed_finding_list(question):
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    if finding_clause.get("required_responsibility_mode") == "SELECT_PRINCIPAL_FINDINGS" and _question_has_exhaustive_fixed_finding_list(question):
        q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    unresolved_dimensions = {item.value for item in metadata_obj.unresolved_operationalization_dimensions}
    resolved_constraints = {
        str(item.get("category")): str(item.get("statement", ""))
        for item in metadata_obj.explicit_method_constraints
        if isinstance(item, Mapping)
    }
    for dimension in unresolved_dimensions:
        statement = resolved_constraints.get(dimension, "")
        if statement and _question_mentions_concrete_clause(question, statement):
            q2_blockers.append("RESOLVED_DIMENSION_LEAK")
    # An open responsibility must remain visibly open in the question; a
    # concrete method clause for an unresolved dimension is a hidden fix.
    if unresolved_dimensions and metadata_obj.operationalization_responsibility.value in {"partially_specified", "model_selected"}:
        if any(token in question.casefold() for token in ("90th percentile", "75th percentile", "peak speed", "mean speed", "strictly greater", "strictly less")):
            q2_blockers.append("UNRESOLVED_DIMENSION_PREMATURELY_FIXED")
    finding_goal = normalize_case_text(metadata_obj.finding_goal).casefold()
    if metadata_obj.finding_openness.value == "open" and any(token in question.casefold() for token in ("must report", "report exactly", "only report")):
        if finding_goal and not any(token in finding_goal for token in ("report", "characterize", "identify")):
            q2_blockers.append("FINDING_RESPONSIBILITY_DRIFT")
    q2 = {"status": "PASS" if not q2_blockers else "FAIL", "check": "Q2_RESPONSIBILITY_INTEGRITY", "failure_codes": sorted(set(q2_blockers))}
    q3_blockers: list[str] = []
    contract = family.get("finding_requirement_contract")
    accepted_o = family.get("candidate_reference_operationalizations", family.get("acceptable_operationalizations", ()))
    if not metadata_obj.principal_operationalization_dimensions:
        q3_blockers.append("O_SCHEMA_UNAVAILABLE")
    representability_contract = family.get("evaluation_representability_contract")
    if representability_contract is None:
        representability_contract = metadata_obj.evaluation_representability_contract
    representability = validate_representability_contract(
        representability_contract,
        allow_default=False,
    )
    q3_blockers.extend(representability["failure_codes"])
    if not contract and metadata_obj.finding_openness.value == "open":
        q3_blockers.append("FINDING_REQUIREMENT_CONTRACT_MISSING")
    if not accepted_o and metadata_obj.operationalization_responsibility.value != "model_selected":
        q3_blockers.append("O_SCHEMA_UNAVAILABLE")
    # The contract is a deterministic registry, not a calibration result.
    # Keep a compact trace of each required route in the SCQ artifact.
    q3 = {"status": "PASS" if not q3_blockers else "FAIL", "check": "Q3_EVALUATION_REPRESENTABILITY", "failure_codes": sorted(set(q3_blockers),), "representability_contract": representability}
    all_blockers = q1_blockers + q2_blockers + q3_blockers + list(construction.blockers)
    status = "SCQ_PASS" if not all_blockers else "SCQ_REJECT" if any(code in {"CASE_DATA_MISMATCH", "UNSUPPORTED_CLAIM_SCOPE", "EVALUATION_NOT_REPRESENTABLE"} for code in all_blockers) else "SCQ_REVISE"
    return SCQResult(case_id, status, q1, q2, q3, construction.to_dict())


__all__ = [
    "SCQResult",
    "ControlledConstructionAudit",
    "qualify_case",
    "validate_controlled_family_construction",
    "validate_sparse_controlled_family_construction",
]
