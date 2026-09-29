"""Complete human-review question portfolio construction.

The scientific-question candidate matrix intentionally reports *release* state
and only authors wording for slots which have already passed its advisory
eligibility gate.  This module is the presentation-layer handoff for human
review.  It consumes that matrix, preserves every scientific judgment, and
materializes a review candidate (or an explicit omission) for each of the
seven-by-four slots.  It never changes a semantic contract, selects a target,
or promotes a case into the benchmark lifecycle.

The three status axes in the output are deliberately independent:

``QUESTION_CANDIDATE_STATUS``
    Whether a human can inspect the wording/responsibility now.
``SCIENTIFIC_RELEASE_STATUS``
    Whether the underlying condition is currently eligible for the benchmark
    lifecycle.
``ARTIFACT_REUSE_STATUS``
    Whether old GT/SRAC artifacts are semantically reusable (authorization is
    still withheld until human selection and downstream gates).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hccq import audit_family_presentation_leakage
from .question_presentation import realize_human_questions
from .scientific_question_candidate_matrix import (
    CONDITIONS,
    DATASET_POLICIES,
    combustor_target_analysis_context,
    combustor_target_question_text,
)


HUMAN_REVIEW_PORTFOLIO_VERSION = "human-review-scientific-question-expansion-v1"
PORTFOLIO_ARTIFACT_NAMES = frozenset(
    {
        "scientific_question_manifest.json",
        "scientific_question_review.md",
        "human_facing_closure_manifest.json",
        "human_facing_closure_report.md",
    }
)

QUESTION_CANDIDATE_STATUSES = frozenset(
    {
        "READY_FOR_HUMAN_REVIEW",
        "NEEDS_SCIENTIFIC_REVISION",
        "NEEDS_EVIDENCE_CONTEXT",
        "NEEDS_TARGET_REDESIGN",
    }
)
SCIENTIFIC_RELEASE_STATUSES = frozenset(
    {
        "ELIGIBLE",
        "BLOCKED_PENDING_EVIDENCE",
        "BLOCKED_PENDING_O_SPACE",
        "NEW_CASE_REQUIRED",
    }
)
ARTIFACT_REUSE_STATUSES = frozenset({"VALID", "STALE", "REBUILD_REQUIRED"})
SUPPORT_DIMENSION_NAMES = (
    "scientific_target_supported",
    "dataset_observable_supported",
    "execution_materialization_supported",
    "evaluation_contract_supported",
    "evidence_complete_for_release",
)
SUPPORT_DIMENSION_VALUES = frozenset(
    {"SUPPORTED", "UNSUPPORTED", "NOT_ESTABLISHED", "NOT_APPLICABLE"}
)

_FALLBACK_SOURCE_MARKERS = frozenset(
    {"DETERMINISTIC_REVIEW_FALLBACK", "DETERMINISTIC_FALLBACK"}
)
_AUTHORED_SOURCE_MARKERS = frozenset(
    {
        "MATRIX_AUTHORED",
        "FLOW_EXPERT_PRESENTATION_AUTHORING",
        "LIVE_FLOW_EXPERT_AUTHORING",
        "FLOW_EXPERT_TARGET_REVIEW_AUTHORING",
    }
)
_SUCCESS_INVOCATION_STATUSES = frozenset(
    {"SUCCESS", "PASS", "COMPLETED", "COMPLETE"}
)
_QUESTION_AUTHORING_OPERATION = "QUESTION_PRESENTATION_AUTHORING"


def _authoring_mode_is_explicit(value: Any) -> bool:
    """Return whether a generation-mode label denotes real question authoring.

    A substring check such as ``"AUTHORING" in mode`` accepts values like
    ``NOT_AUTHORING``.  Mode labels are provenance hints only, but accepting a
    negated label would incorrectly upgrade an otherwise unverified candidate.
    """

    mode = _text(value).upper()
    if not mode or mode.startswith(("NOT_", "NO_", "DETERMINISTIC", "FALLBACK")):
        return False
    return mode in _AUTHORED_SOURCE_MARKERS or mode.endswith("AUTHORING")

# A post-rewrite fidelity result is deliberately *not* a final scientific
# review.  Only one of these explicit records can establish that the final
# model-visible wording was reviewed for meaning, target/scope, O/F
# responsibility, answerability, and evidence limits.  The aliases keep the
# handoff compatible with the small number of historical review artifacts
# without treating a bare status flag as provenance.
_FINAL_FLOW_EXPERT_REVIEW_KEYS = (
    "final_flow_expert_wording_review",
    "final_flow_expert_review",
    "final_question_review",
    "final_wording_review",
    "flow_expert_final_review",
)


def _read_json(path: str | Path) -> Any:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_manifest(
    value: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    loaded = _read_json(value)
    if not isinstance(loaded, Mapping):
        raise ValueError("matrix manifest must be a JSON object")
    return copy.deepcopy(dict(loaded))


def _is_portfolio_manifest(value: Mapping[str, Any]) -> bool:
    """Identify a final portfolio so it cannot be consumed as its own matrix."""

    artifact_type = _text(value.get("artifact_type")).upper()
    schema_version = _text(value.get("schema_version")).lower()
    return (
        artifact_type in {
            "HUMAN_REVIEW_SCIENTIFIC_QUESTION_PORTFOLIO",
            "SCIENTIFIC_QUESTION_HUMAN_REVIEW_PORTFOLIO",
        }
        or schema_version.startswith("human-review-scientific-question-expansion-")
    )


def _default_matrix_path(root: Path) -> Path | None:
    """Resolve the staged source matrix without selecting a final portfolio."""

    candidates = (
        root / "outputs/current/scientific_question_matrix/scientific_question_manifest.json",
        root / "outputs/current/scientific_question_source/scientific_question_manifest.json",
        # Compatibility locations used before the source/final split.
        root / "outputs/current/scientific_question_review/scientific_question_manifest.json",
        root / "outputs/current/scientific_questions/scientific_question_manifest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(value, Mapping) and not _is_portfolio_manifest(value):
            return path
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _condition_key(dataset_id: str, condition: str) -> tuple[str, str]:
    return (_text(dataset_id), _text(condition))


def _expected_slots() -> list[tuple[str, str]]:
    return [
        (dataset_id, condition)
        for dataset_id in DATASET_POLICIES
        for condition in CONDITIONS
    ]


def _matrix_status(row: Mapping[str, Any]) -> str:
    return _text(row.get("matrix_status")) or _text(
        row.get("scientific_release_status")
    )


def _release_status(row: Mapping[str, Any]) -> str:
    explicit = _text(
        row.get("SCIENTIFIC_RELEASE_STATUS")
        or row.get("scientific_release_status")
    )
    if explicit in SCIENTIFIC_RELEASE_STATUSES:
        return explicit
    status = _matrix_status(row)
    if status == "CANDIDATE_FOR_HUMAN_REVIEW":
        return "ELIGIBLE"
    if status == "BLOCKED_PENDING_EVIDENCE":
        return "BLOCKED_PENDING_EVIDENCE"
    if status in {"BLOCKED_PENDING_O_SPACE_REVIEW", "BLOCKED_PENDING_O_SPACE"}:
        return "BLOCKED_PENDING_O_SPACE"
    if status in {
        "NEW_CASE_PENDING_HUMAN_SELECTION",
        "NEW_CASE_PENDING_SELECTION",
        "NEW_CASE_REQUIRED",
    }:
        return "NEW_CASE_REQUIRED"
    if status == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
        # The release vocabulary has no OMITTED value.  A scientifically
        # omitted slot cannot enter the lifecycle without a new target/case.
        return "NEW_CASE_REQUIRED"
    return "BLOCKED_PENDING_EVIDENCE"


def _support_dimensions_are_explicit(row: Mapping[str, Any]) -> bool:
    """Return whether this row carries the current condition support contract."""

    raw = row.get("support_dimensions")
    if isinstance(raw, Mapping):
        return True
    review = row.get("condition_scientific_review")
    if isinstance(review, Mapping):
        normalized = review.get("normalized_review")
        if isinstance(normalized, Mapping) and isinstance(
            normalized.get("support_dimensions"), Mapping
        ):
            return True
        if isinstance(review.get("support_dimensions"), Mapping):
            return True
    return any(name in row for name in SUPPORT_DIMENSION_NAMES)


def _strict_release_status(row: Mapping[str, Any]) -> str:
    """Compile release state from current support fields, never stale labels.

    Historical matrix artifacts predate the five-dimensional contract.  They
    remain readable for compatibility, but their old ``ELIGIBLE`` label is
    explicitly non-authoritative and therefore cannot produce a strict release
    result until a current support contract is present.
    """

    legacy = _release_status(row)
    if _text(row.get("dataset_id")) == "Combustor":
        return "NEW_CASE_REQUIRED"
    support = _support_dimensions(row)
    if not _support_dimensions_are_explicit(row):
        return "BLOCKED_PENDING_EVIDENCE" if legacy == "ELIGIBLE" else legacy
    if support.get("scientific_target_supported") == "UNSUPPORTED" or support.get(
        "dataset_observable_supported"
    ) == "UNSUPPORTED":
        return "NEW_CASE_REQUIRED"
    if legacy == "BLOCKED_PENDING_O_SPACE":
        return legacy
    applicable = [
        value
        for value in support.values()
        if value != "NOT_APPLICABLE"
    ]
    if any(value != "SUPPORTED" for value in applicable):
        return "BLOCKED_PENDING_EVIDENCE"
    return legacy


def _question_status(row: Mapping[str, Any], release_status: str) -> str:
    explicit = _text(
        row.get("QUESTION_CANDIDATE_STATUS")
        or row.get("question_candidate_status")
    )
    if explicit in QUESTION_CANDIDATE_STATUSES:
        return explicit
    status = _matrix_status(row)
    if status == "CANDIDATE_FOR_HUMAN_REVIEW" or release_status == "ELIGIBLE":
        return "READY_FOR_HUMAN_REVIEW"
    # A blocked condition can still carry a complete wording candidate.  If
    # the source matrix exposes one, retain it as reviewable and use the
    # explicit four-state status only when supplied by the producer.
    if status in {"BLOCKED_PENDING_EVIDENCE", "BLOCKED_PENDING_O_SPACE_REVIEW"}:
        if isinstance(row.get("question_candidates"), list) and row.get("question_candidates"):
            return "READY_FOR_HUMAN_REVIEW"
    if status == "BLOCKED_PENDING_O_SPACE_REVIEW" or release_status == "BLOCKED_PENDING_O_SPACE":
        return "NEEDS_SCIENTIFIC_REVISION"
    if status in {
        "NEW_CASE_PENDING_HUMAN_SELECTION",
        "NEW_CASE_PENDING_SELECTION",
        "NEW_CASE_REQUIRED",
    }:
        return "NEEDS_TARGET_REDESIGN"
    if status == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
        return "NEEDS_TARGET_REDESIGN"
    return "NEEDS_EVIDENCE_CONTEXT"


def _reuse_status(row: Mapping[str, Any], release_status: str) -> str:
    explicit = _text(
        row.get("ARTIFACT_REUSE_STATUS") or row.get("artifact_reuse_status")
    )
    if explicit in ARTIFACT_REUSE_STATUSES:
        return explicit
    impact = _text(row.get("GT_SRAC_IMPACT"))
    legacy = _text(row.get("historical_gt_srac_reuse_status"))
    if "STALE" in impact.upper() or "STALE" in legacy.upper():
        return "STALE"
    historical_identity = _text(row.get("historical_semantic_identity_reused")).casefold() in {
        "true",
        "1",
    }
    # ``PENDING_HUMAN_SELECTION`` means the identity is valid but not yet
    # authorized.  Explicit evidence/O-space blocks are conservatively marked
    # for rebuild/re-adjudication instead of being presented as reusable.
    if historical_identity and legacy.upper() in {
        "PENDING_HUMAN_SELECTION",
        "UNCHANGED",
    } and not bool(row.get("canonical_case_mutated", False)):
        return "VALID"
    if historical_identity and release_status == "ELIGIBLE" and not bool(
        row.get("canonical_case_mutated", False)
    ):
        return "VALID"
    # A blocked or newly redesigned identity cannot safely reuse old output.
    return "REBUILD_REQUIRED"


def _pending_srac_required(row: Mapping[str, Any]) -> bool:
    """Return whether this condition explicitly carries an SRAC dependency."""

    release_status = _text(
        row.get("SCIENTIFIC_RELEASE_STATUS")
        or row.get("scientific_release_status")
    ).upper()
    impact = _text(row.get("GT_SRAC_IMPACT")).upper()
    return release_status in {"BLOCKED_PENDING_SRAC", "PENDING_SRAC"} or (
        "SRAC" in impact and "REQUIRED" in impact
    )


def _review_comments(row: Mapping[str, Any]) -> dict[str, Any]:
    review = row.get("condition_scientific_review")
    if not isinstance(review, Mapping):
        review = {}
    normalized = review.get("normalized_review")
    if not isinstance(normalized, Mapping):
        normalized = review
    return {
        "rationale": _text(normalized.get("rationale")),
        "blocking_reason_codes": [
            _text(item)
            for item in (
                normalized.get("blocking_reason_codes")
                or row.get("blocking_reason_codes")
                or []
            )
            if _text(item)
        ],
        "evidence_gaps": [
            _text(item)
            for item in (
                normalized.get("evidence_gaps")
                or row.get("evidence_gaps")
                or []
            )
            if _text(item)
        ],
        "evidence_ids": [
            _text(item)
            for item in (
                normalized.get("evidence_ids")
                or row.get("evidence_ids")
                or []
            )
            if _text(item)
        ],
        "review_status": _text(review.get("status")) or "NOT_REPORTED",
    }


def _support_dimensions(row: Mapping[str, Any]) -> dict[str, str]:
    """Expose the matrix's five independent support dimensions verbatim."""

    raw = row.get("support_dimensions")
    if not isinstance(raw, Mapping):
        review = row.get("condition_scientific_review")
        if isinstance(review, Mapping):
            normalized = review.get("normalized_review")
            if isinstance(normalized, Mapping):
                raw = normalized.get("support_dimensions")
            if not isinstance(raw, Mapping):
                raw = review.get("support_dimensions")
    result: dict[str, str] = {}
    for name in SUPPORT_DIMENSION_NAMES:
        value = raw.get(name) if isinstance(raw, Mapping) else row.get(name)
        if value is None:
            value = row.get(name)
        result[name] = _text(value) or "NOT_ESTABLISHED"
    return result


def _open_operationalization_dimensions(value: Mapping[str, Any]) -> list[Any]:
    """Read the canonical open-O field, with a legacy alias fallback.

    ``unresolved_operationalization_dimensions`` is the frozen matrix field.
    ``open_operationalization`` is retained for older artifacts only.  Key
    presence, rather than truthiness, is intentional: a canonical empty list
    means that no O dimension is open and must not be replaced by a stale
    non-empty compatibility alias.
    """

    if "unresolved_operationalization_dimensions" in value:
        raw = value.get("unresolved_operationalization_dimensions")
    else:
        raw = value.get("open_operationalization")
    return copy.deepcopy(raw) if isinstance(raw, list) else []


def _analysis_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the non-semantic analysis context shown beside the question."""

    # This is deliberately a placement object.  The canonical projection and
    # its hash remain the sole scientific source of truth.
    raw_visibility = row.get("semantic_visibility")
    visibility = copy.deepcopy(raw_visibility)
    if isinstance(raw_visibility, Mapping):
        # A review/model-facing context may carry QUESTION_VISIBLE and
        # DATASET_CONTEXT_VISIBLE placement information, but must not surface
        # entries explicitly classified BACKEND_ONLY.
        visibility = {
            **{
                key: copy.deepcopy(value)
                for key, value in raw_visibility.items()
                if key != "items"
            },
            "items": [
                copy.deepcopy(item)
                for item in raw_visibility.get("items", ())
                if isinstance(item, Mapping)
                and _text(item.get("visibility")) != "BACKEND_ONLY"
            ],
        }
    return {
        "dataset_id": _text(row.get("dataset_id")),
        "family_id": _text(row.get("family_id")),
        "scientific_scope": copy.deepcopy(row.get("scientific_scope")),
        "fixed_operationalization": copy.deepcopy(
            row.get("fixed_operationalization") or []
        ),
        "open_operationalization": _open_operationalization_dimensions(row),
        "finding_responsibility": copy.deepcopy(row.get("finding_responsibility")),
        "semantic_visibility": visibility,
        "evidence_status": _text(row.get("evidence_sufficiency"))
        or "NOT_ESTABLISHED",
        "support_dimensions": _support_dimensions(row),
        "backend_details_are_context_only": True,
    }


def _candidate_status_from_fidelity(candidate: Mapping[str, Any]) -> str:
    fidelity = candidate.get("post_rewrite_semantic_fidelity")
    if isinstance(fidelity, Mapping):
        status = _text(fidelity.get("status")).upper()
        if status:
            return status
    status = _text(candidate.get("semantic_fidelity_status")).upper()
    return status or "NOT_ESTABLISHED"


def _successful_invocation_record(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Return a verifiable successful invocation record, if one is present.

    ``status=PASS`` on a candidate is a result, not evidence that a model was
    called.  The provenance boundary therefore requires an explicit
    ``invocation_status`` on a call record.  Deterministic/NOT_RUN execution
    markers are never accepted as live authoring provenance.
    """

    if not isinstance(value, Mapping):
        return None
    status = _text(value.get("invocation_status")).upper()
    if status not in _SUCCESS_INVOCATION_STATUSES:
        return None
    execution_mode = _text(value.get("execution_mode")).upper()
    if execution_mode.startswith("NOT_RUN") or "DETERMINISTIC" in execution_mode:
        return None
    # New records identify the operation explicitly.  Keep compatibility with
    # older minimal fixtures that omit it, but reject a contradictory record
    # (for example a final-review call attached as authoring evidence).
    operation = _text(value.get("authoring_operation")).upper()
    if operation and operation != _QUESTION_AUTHORING_OPERATION:
        return None
    return value


def _candidate_authoring_invocation(
    candidate: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Locate the explicit call record for a wording authoring operation."""

    for key in (
        "question_authoring_invocation",
        "authoring_invocation",
        "authoring_call",
    ):
        record = _successful_invocation_record(candidate.get(key))
        if record is not None:
            return record
    return None


def _combustor_target_selection_is_qualified(
    target: Mapping[str, Any],
) -> bool:
    """Return whether a provisional target has review-backed provenance.

    A hand-edited ``recommended_provisional_target=true`` bit is not a Flow
    Expert recommendation.  The target may enter the human-review portfolio
    only when its target-selection review passed validation and carries a
    successful, non-deterministic invocation record.
    """

    if target.get("recommended_provisional_target") is not True:
        return False
    review = target.get("target_selection_review")
    if not isinstance(review, Mapping):
        return False
    validation_status = _text(
        review.get("validation_status") or review.get("status")
    ).upper()
    if validation_status != "PASS":
        return False
    review_status = _text(review.get("status")).upper()
    if review_status not in {"PASS", "RECOMMENDED_PROVISIONAL_TARGET"}:
        return False
    if review.get("human_approval_required") is not True:
        return False
    normalized = review.get("normalized_review")
    if not isinstance(normalized, Mapping):
        return False
    recommended_id = _text(
        normalized.get("recommended_target_id")
        or normalized.get("recommended_provisional_target_id")
    )
    if not recommended_id or recommended_id != _text(target.get("candidate_id")):
        return False
    invocation = next(
        (
            review.get(key)
            for key in (
                "target_review_invocation",
                "review_invocation",
                "invocation",
                "review_call",
                "call",
            )
            if isinstance(review.get(key), Mapping)
        ),
        None,
    )
    return _successful_invocation_record(invocation) is not None


def _qualified_combustor_recommended_targets(
    targets: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select only unambiguous, review-backed provisional recommendations."""

    return [
        target
        for target in targets
        if isinstance(target, Mapping)
        and _combustor_target_selection_is_qualified(target)
    ]


def _review_record_status(record: Mapping[str, Any]) -> str:
    """Read a final-review verdict without interpreting free text."""

    for key in ("status", "review_status", "verdict", "recommendation"):
        value = _text(record.get(key)).upper()
        if value:
            return value
    return ""


def _candidate_final_flow_expert_review(
    candidate: Mapping[str, Any],
) -> tuple[str, str]:
    """Resolve a *dedicated* final wording review and its provenance.

    The historical ``post_rewrite_semantic_fidelity`` record is intentionally
    excluded: that call checks semantic preservation only and declares
    ``scientific_qualification_performed=false``.  A bare
    ``flow_expert_review_status=PASS`` is likewise insufficient.  A final
    review must carry both a PASS verdict and an explicit successful
    invocation record (directly or under ``review_call``/``invocation``).
    """

    records: list[tuple[str, Mapping[str, Any]]] = []
    for key in _FINAL_FLOW_EXPERT_REVIEW_KEYS:
        value = candidate.get(key)
        if isinstance(value, Mapping):
            records.append((key, value))

    # Some producers keep the verdict and invocation as sibling fields.
    explicit_status = _text(
        candidate.get("flow_expert_review_status")
        or candidate.get("final_flow_expert_review_status")
    ).upper()
    sibling_invocation = None
    for key in (
        "flow_expert_review_invocation",
        "final_flow_expert_review_invocation",
        "final_review_invocation",
    ):
        if isinstance(candidate.get(key), Mapping):
            sibling_invocation = candidate[key]
            break
    if explicit_status == "PASS" and sibling_invocation is not None:
        records.append(
            ("FLOW_EXPERT_REVIEW_STATUS_AND_INVOCATION", {
                "status": explicit_status,
                "invocation": sibling_invocation,
            })
        )

    for key, record in records:
        verdict = _review_record_status(record)
        # When the canonical audit record carries the bound text/hash, make
        # sure the reviewer actually saw this candidate's final wording.  The
        # lightweight compatibility fixtures may omit these optional fields;
        # the explicit successful invocation requirement still applies.
        reviewed_text = record.get("reviewed_model_visible_text")
        candidate_text = _question_text(candidate)
        if isinstance(reviewed_text, str) and reviewed_text.strip():
            if candidate_text and reviewed_text.strip() != candidate_text:
                continue
        reviewed_hash = _text(record.get("reviewed_model_visible_text_sha256"))
        if reviewed_hash and candidate_text:
            expected_hash = _stable_digest({"model_visible_text": candidate_text})
            if reviewed_hash != expected_hash:
                continue
        invocation_values: list[Mapping[str, Any]] = []
        for invocation_key in ("review_call", "invocation", "call"):
            invocation = record.get(invocation_key)
            if isinstance(invocation, Mapping):
                invocation_values.append(invocation)
        # ``audit_final_flow_expert_wording`` stores the transport envelope in
        # ``review_call`` and also promotes its normalized invocation status to
        # the result object.  Check the result itself as a fallback so an
        # injected reviewer that omitted an envelope status is still recorded
        # with the explicit SUCCESS set by the audit function.
        invocation_values.append(record)
        if verdict == "PASS" and any(
            _successful_invocation_record(invocation)
            for invocation in invocation_values
        ):
            return "PASS", key

    return "NOT_ESTABLISHED", "NO_FINAL_FLOW_EXPERT_REVIEW_RECORD"


def _candidate_flow_expert_review_status(candidate: Mapping[str, Any]) -> tuple[str, str]:
    """Backward-compatible alias for the dedicated final-review resolver."""

    return _candidate_final_flow_expert_review(candidate)


def _candidate_wording_responsibility_audit(
    candidate: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    """Check condition responsibility at the presentation boundary.

    This is intentionally a small lexical guard.  Scientific semantics remain
    in the canonical projection and the independent fidelity reviewer.
    """

    text = _text(
        candidate.get("scientific_question")
        or candidate.get("question_text")
        or candidate.get("model_visible_text")
    )
    condition = _text(row.get("condition")).upper()
    lower = text.casefold()
    fixed = row.get("fixed_operationalization") or []
    open_dims = _open_operationalization_dimensions(row)
    open_ids = {
        _text(item.get("dimension_id") if isinstance(item, Mapping) else item)
        for item in open_dims
        if _text(item.get("dimension_id") if isinstance(item, Mapping) else item)
    }
    findings = _text(row.get("finding_responsibility")).upper()
    codes: list[str] = []
    if condition in {"O2-F1", "O3-F1"} and open_ids and not re.search(
        r"\b(?:appropriate|choose|choice|defensible|select|decide|your)\b", lower
    ):
        codes.append("OPEN_O_RESPONSIBILITY_NOT_VISIBLE")
    if condition == "O1-F2" and not re.search(
        r"\b(?:characteri[sz]e|relevant|other|additional|findings?|meaningful)\b", lower
    ):
        codes.append("F2_FINDING_RESPONSIBILITY_NOT_VISIBLE")
    if condition == "O1-F1" and fixed and re.search(
        r"\b(?:choose|appropriate|defensible|your\s+(?:analysis|choice|method))\b", lower
    ):
        codes.append("O1_FIXED_O_PRESENTED_AS_OPEN")
    # Observable/field selection is an O dimension only when declared.  Merely
    # asking the respondent to report which field was used is not a choice.
    if re.search(
        r"\b(?:choose|select|decide)\s+(?:the\s+)?(?:flow\s+)?(?:field|observable|variable)\b",
        lower,
    ) and not ({"observable_selection", "field_selection", "observable"} & open_ids):
        codes.append("UNDECLARED_OBSERVABLE_SELECTION")
    if findings == "F1" and condition.endswith("-F1") and re.search(
        r"\b(?:choose|select)\s+(?:any|which|the)\s+(?:finding|result)", lower
    ):
        codes.append("F1_FINDING_SELECTION_PRESENTED_AS_OPEN")
    return {
        "status": "PASS" if not codes else "REVISE",
        "failure_codes": sorted(set(codes)),
    }


def _annotate_candidate_provenance(
    candidate: Mapping[str, Any], row: Mapping[str, Any], *, candidate_role: str = "AUTHORED_QUESTION"
) -> dict[str, Any]:
    value = dict(candidate)
    source = _text(value.get("candidate_source")).upper()
    fallback = source in _FALLBACK_SOURCE_MARKERS or "FALLBACK" in source
    if candidate_role == "TARGET_PROPOSAL" or source == "COMBUSTOR_TARGET_REDESIGN_REVIEW":
        candidate_role = "TARGET_PROPOSAL"
    authoring_mode = _text(value.get("authoring_mode") or value.get("generation_mode")).upper()
    if not source:
        source = "UNSPECIFIED"
    authored_source = source in _AUTHORED_SOURCE_MARKERS or _authoring_mode_is_explicit(
        authoring_mode
    )
    # A source marker is necessary but not sufficient: an authored question
    # must carry the frozen target and O/F responsibility context as well.
    target = value.get("scientific_target") or row.get("scientific_target")
    fixed_o = value.get("fixed_operationalization")
    if fixed_o is None:
        fixed_o = row.get("fixed_operationalization") or []
    # The matrix row is the semantic source of truth.  A candidate-level
    # canonical field may be used when present (for hand-authored fixtures),
    # but a candidate's legacy alias must never override a row-level canonical
    # value, especially an authoritative empty list for O1.
    if "unresolved_operationalization_dimensions" in value:
        open_o = _open_operationalization_dimensions(value)
    elif "unresolved_operationalization_dimensions" in row:
        open_o = _open_operationalization_dimensions(row)
    elif "open_operationalization" in value:
        open_o = _open_operationalization_dimensions(value)
    else:
        open_o = _open_operationalization_dimensions(row)
    finding = value.get("finding_responsibility")
    if finding is None:
        finding = row.get("finding_responsibility")
    contract_present = target not in (None, "") and isinstance(fixed_o, list) and isinstance(open_o, list) and finding not in (None, "")
    authoring_invocation = _candidate_authoring_invocation(value)
    if authoring_invocation is None:
        # Matrix rows produced by older authoring paths stored one invocation
        # record at row level.  It remains valid provenance for each candidate
        # only when that explicit record itself reports success.
        for key in ("question_authoring_invocation", "authoring_invocation"):
            row_record = row.get(key)
            authoring_invocation = _successful_invocation_record(row_record)
            if authoring_invocation is not None:
                break
    # A source marker or authoring_mode is only a label.  Strict portfolio
    # counts require the corresponding successful call record as well.
    authoring_invocation_verified = authoring_invocation is not None
    authoring_status = (
        "PASS"
        if authored_source
        and not fallback
        and candidate_role == "AUTHORED_QUESTION"
        and contract_present
        and authoring_invocation_verified
        else "NOT_ESTABLISHED"
    )
    fidelity_status = _candidate_status_from_fidelity(value)
    flow_review_status, flow_review_provenance = _candidate_flow_expert_review_status(value)
    hccq = value.get("hccq_presentation")
    hccq_status = _text(
        value.get("hccq_status")
        or (hccq.get("status") if isinstance(hccq, Mapping) else "")
    ).upper() or "NOT_ESTABLISHED"
    responsibility = _candidate_wording_responsibility_audit(value, row)
    if responsibility["status"] != "PASS" and fidelity_status == "PASS":
        fidelity_status = "REVISE"
    genuine = bool(
        authoring_status == "PASS"
        and _text(value.get("scientific_question") or value.get("question_text") or value.get("model_visible_text"))
    )
    value.update(
        {
            "candidate_source": source,
            "candidate_role": candidate_role,
            "authoring_status": authoring_status,
            "authoring_provenance": {
                "candidate_source": source,
                "authoring_mode": authoring_mode or None,
                "invocation_status": _text(
                    value.get("question_authoring_invocation", {}).get("invocation_status")
                    if isinstance(value.get("question_authoring_invocation"), Mapping)
                    else row.get("question_authoring_invocation", {}).get("invocation_status")
                    if isinstance(row.get("question_authoring_invocation"), Mapping)
                    else ""
                ).upper() or None,
                "invocation_verified": authoring_invocation_verified,
                "deterministic_fallback": fallback,
                "contract_present": contract_present,
                "genuine_authored_question": genuine,
            },
            "genuine_authored_question": genuine,
            "flow_expert_review_status": flow_review_status,
            "flow_expert_review_provenance": flow_review_provenance,
            "semantic_fidelity_status": fidelity_status,
            # Keep fidelity and final scientific review as separate explicit
            # fields.  Consumers must not infer one from the other.
            "post_rewrite_semantic_fidelity_status": fidelity_status,
            "hccq_status": hccq_status,
            "wording_responsibility_audit": responsibility,
            "family_template_leakage_status": _text(value.get("family_template_leakage_status")).upper() or "NOT_ESTABLISHED",
        }
    )
    return value


def _normalise_existing_candidate(
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    source: str = "UNSPECIFIED",
) -> dict[str, Any]:
    """Expose a stable question_text alias without rewriting authored text."""

    value = copy.deepcopy(dict(candidate))
    nested = value.get("candidate")
    if isinstance(nested, Mapping):
        # Older quality-review artifacts wrap the authored object.  Keep the
        # audit fields while flattening only the presentation aliases.
        for key in ("model_visible_text", "scientific_question", "candidate_id", "style"):
            if key not in value and key in nested:
                value[key] = copy.deepcopy(nested[key])
    # Keep the principal scientific sentence separate from the full model
    # presentation.  Deterministic renderers expose ``scientific_question``;
    # authored model output may only expose ``model_visible_text``.
    text = _text(
        value.get("scientific_question")
        or value.get("question_text")
        or value.get("model_visible_text")
    )
    if text:
        value["question_text"] = text
        value.setdefault("scientific_question", text)
        value.setdefault("model_visible_text", text)
    if not _text(value.get("candidate_source")):
        authoring_mode = _text(value.get("authoring_mode") or value.get("generation_mode"))
        if _authoring_mode_is_explicit(authoring_mode):
            source = "FLOW_EXPERT_PRESENTATION_AUTHORING"
        elif "FALLBACK" in authoring_mode.upper() or "REALIZATION" in authoring_mode.upper():
            source = "DETERMINISTIC_REVIEW_FALLBACK"
        value["candidate_source"] = source
    else:
        value["candidate_source"] = _text(value.get("candidate_source"))
    value.setdefault("automatic_adoption", False)
    value.setdefault("dataset_id", _text(row.get("dataset_id")))
    value.setdefault("condition", _text(row.get("condition")))
    # Keep each candidate self-describing for reviewers.  Live author calls
    # intentionally return only presentation fields; the frozen row remains
    # the source of truth for these scientific/release annotations.  Copying
    # them here makes the required per-candidate review record complete while
    # leaving the authored wording and semantic hash untouched.
    value.setdefault("scientific_target", copy.deepcopy(row.get("scientific_target")))
    value.setdefault("scientific_scope", copy.deepcopy(row.get("scientific_scope")))
    value.setdefault(
        "fixed_operationalization",
        copy.deepcopy(row.get("fixed_operationalization") or []),
    )
    row_open_o = _open_operationalization_dimensions(row)
    if "unresolved_operationalization_dimensions" in row:
        # Canonical row metadata wins over any stale alias carried by an older
        # candidate producer.  Keep both public spellings synchronized for the
        # downstream presentation contract.
        value["unresolved_operationalization_dimensions"] = copy.deepcopy(row_open_o)
        value["open_operationalization"] = copy.deepcopy(row_open_o)
    else:
        value.setdefault("open_operationalization", copy.deepcopy(row_open_o))
        value.setdefault(
            "unresolved_operationalization_dimensions", copy.deepcopy(row_open_o)
        )
    value.setdefault("finding_responsibility", copy.deepcopy(row.get("finding_responsibility")))
    value.setdefault("flow_expert_comments", copy.deepcopy(_review_comments(row)))
    value.setdefault(
        "question_authoring_invocation",
        copy.deepcopy(row.get("question_authoring_invocation")),
    )
    # A final wording review may be emitted once per row by the matrix
    # producer.  Copy it onto each candidate so candidate-level counters can
    # verify that the actual text (rather than only slot metadata) was
    # reviewed.  Fidelity records are intentionally not used for this field.
    for review_key in _FINAL_FLOW_EXPERT_REVIEW_KEYS + (
        "flow_expert_review_invocation",
        "final_flow_expert_review_invocation",
        "final_review_invocation",
    ):
        if review_key not in value and review_key in row:
            value[review_key] = copy.deepcopy(row.get(review_key))
    value.setdefault(
        "evidence_status",
        _text(row.get("evidence_sufficiency") or row.get("evidence_status"))
        or "NOT_ESTABLISHED",
    )
    value.setdefault("evidence_ids", copy.deepcopy(row.get("evidence_ids") or []))
    value.setdefault(
        "scientific_release_status",
        _text(row.get("SCIENTIFIC_RELEASE_STATUS") or row.get("scientific_release_status"))
        or _release_status(row),
    )
    value.setdefault(
        "artifact_reuse_status",
        _text(row.get("ARTIFACT_REUSE_STATUS") or row.get("artifact_reuse_status"))
        or _reuse_status(row, _release_status(row)),
    )
    value.setdefault(
        "question_candidate_status",
        _text(row.get("QUESTION_CANDIDATE_STATUS") or row.get("question_candidate_status"))
        or _question_status(row, _release_status(row)),
    )
    if row.get("semantic_contract_sha256") and value.get("semantic_contract_sha256") is None:
        value["semantic_contract_sha256"] = row.get("semantic_contract_sha256")
    return value


def _fallback_candidates(row: Mapping[str, Any], *, candidate_count: int) -> list[dict[str, Any]]:
    projection = row.get("canonical_semantic_projection")
    if not isinstance(projection, Mapping):
        return []
    try:
        realization = realize_human_questions(
            projection,
            case_id=row.get("case_id"),
            condition=row.get("condition"),
            candidate_count=candidate_count,
            semantic_contract_sha256=(
                _text(row.get("semantic_contract_sha256")) or None
            ),
        )
    except (TypeError, ValueError, KeyError):
        return []
    result: list[dict[str, Any]] = []
    for candidate in realization.get("candidates", ()):
        if not isinstance(candidate, Mapping):
            continue
        normalized = _normalise_existing_candidate(
            candidate,
            row,
            source="DETERMINISTIC_REVIEW_FALLBACK",
        )
        # Mark deterministic fallbacks as review candidates, not as approved
        # authoring output.  A curator still decides whether wording is fit.
        normalized["question_review_status"] = "NEEDS_HUMAN_REVIEW"
        result.append(normalized)
    return result


def _combustor_candidates(
    row: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Use only an explicitly authored four-condition provisional family.

    Target-review records commonly contain alternative target proposals and a
    single generic target sentence.  Those records are not four condition
    questions.  Until a human/Flow Expert author supplies one recommended
    target with condition-specific wording, this function fails closed and the
    row carries an explicit hard-failure reason.
    """

    condition = _text(row.get("condition"))
    result: list[dict[str, Any]] = []
    # A target-review record may contain several alternatives.  Only an
    # explicit boolean recommendation can seed a provisional four-condition
    # family; status strings are descriptive/audit metadata and are not an
    # authorization signal.
    recommended_targets = _qualified_combustor_recommended_targets(targets)
    if len(recommended_targets) != 1:
        return result
    for target in recommended_targets:
        if not isinstance(target, Mapping):
            continue
        target_id = _text(target.get("candidate_id")) or "provisional-target"
        meaning = _text(target.get("scientific_meaning"))
        questions = target.get("condition_questions") or target.get("authored_condition_questions")
        # A recommended target is a family-level provisional proposal.  Do not
        # expose a partial target (for example, only O1-F1) as if it were a
        # valid Combustor condition; all four condition-specific questions must
        # be available before any one slot can enter the review surface.
        if not isinstance(questions, Mapping) or any(
            not isinstance(questions.get(required_condition), Mapping)
            or not _text(
                questions.get(required_condition, {}).get("scientific_question")
                or questions.get(required_condition, {}).get("question_text")
                or questions.get(required_condition, {}).get("model_visible_text")
            )
            for required_condition in CONDITIONS
        ):
            continue
        authored = questions.get(condition)
        if not isinstance(authored, Mapping):
            continue
        candidate = _normalise_existing_candidate(
            authored,
            row,
            source=_text(authored.get("candidate_source"))
            or "FLOW_EXPERT_PRESENTATION_AUTHORING",
        )
        canonical_target = row.get("scientific_target")
        if canonical_target is None:
            canonical_target = candidate.get("scientific_target")
        candidate.update(
            {
                "candidate_id": _text(candidate.get("candidate_id"))
                or f"{row['slot_id']}::{target_id}",
                "target_candidate_id": target_id,
                "dataset_id": "Combustor",
                "condition": condition,
                # The row projection is the canonical semantic contract.  The
                # target-review ``scientific_meaning`` is explanatory text and
                # must not overwrite the target against which this wording was
                # authored and fidelity-reviewed.
                "scientific_target": copy.deepcopy(canonical_target),
                "target_candidate_scientific_meaning": meaning or None,
                "scientific_scope": candidate.get("scientific_scope")
                or "in the supplied combustor flow data",
                "candidate_role": "AUTHORED_QUESTION",
                "target_selection_required": True,
                "provisional_target_status": "PROVISIONAL_TARGET_PENDING_HUMAN_APPROVAL",
                "automatic_adoption": False,
                "evidence_ids": copy.deepcopy(target.get("evidence_ids", [])),
                "evidence_basis": _text(target.get("evidence_basis")),
                "semantic_contract_sha256": candidate.get("semantic_contract_sha256"),
                "scientific_release_status": "NEW_CASE_REQUIRED",
                "artifact_reuse_status": "REBUILD_REQUIRED",
                "question_candidate_status": "READY_FOR_HUMAN_REVIEW",
            }
        )
        result.append(_annotate_candidate_provenance(candidate, row))
    return result


def _row_from_matrix(
    row: Mapping[str, Any],
    *,
    target_candidates: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> dict[str, Any]:
    dataset_id = _text(row.get("dataset_id"))
    condition = _text(row.get("condition"))
    # A legacy matrix's ELIGIBLE label is audit-only.  The portfolio requires
    # an explicit current five-dimensional support contract before retaining
    # that release conclusion.
    release_status = _strict_release_status(row)
    question_status = _question_status(row, release_status)
    reuse_status = _reuse_status(row, release_status)
    # Only the explicitly redesigned Combustor family is target-selection
    # driven.  A missing row for another dataset is an omission, never an
    # opportunity to borrow Combustor target candidates.
    is_combustor = dataset_id == "Combustor"

    if is_combustor:
        candidates = _combustor_candidates(row, target_candidates)
    else:
        raw_candidates = row.get("question_candidates")
        candidates = [
            _annotate_candidate_provenance(_normalise_existing_candidate(item, row), row)
            for item in (raw_candidates if isinstance(raw_candidates, list) else [])
            if isinstance(item, Mapping)
        ]
        if (
            not candidates
            and question_status != "NEEDS_TARGET_REDESIGN"
            and _matrix_status(row) != "OMITTED_SCIENTIFICALLY_UNSUPPORTED"
        ):
            candidates = [
                _annotate_candidate_provenance(candidate, row)
                for candidate in _fallback_candidates(row, candidate_count=candidate_count)
            ]

    all_candidates = candidates
    genuine_candidates = [
        candidate
        for candidate in all_candidates
        if isinstance(candidate, Mapping)
        and candidate.get("genuine_authored_question") is True
    ]
    recovery_candidates = [
        candidate
        for candidate in all_candidates
        if isinstance(candidate, Mapping)
        and candidate.get("genuine_authored_question") is not True
    ]
    omission_reason = ""
    if not all_candidates:
        if is_combustor:
            omission_reason = (
                "No evidence-grounded Combustor target candidate is available; "
                "human target selection is required before wording can be authored."
            )
        elif _matrix_status(row) == "OMITTED_SCIENTIFICALLY_UNSUPPORTED":
            omission_reason = (
                "The condition was scientifically omitted by the condition-scoped "
                "review; no weak question is synthesized."
            )
        else:
            omission_reason = (
                "The canonical semantic projection could not be materialized for "
                "human review; scientific omission must be adjudicated explicitly."
            )
    genuine_present = bool(genuine_candidates)
    genuine_omission_reason = None
    if not genuine_present:
        genuine_omission_reason = (
            "No genuinely authored question is available for this slot."
            if not all_candidates
            else "Only deterministic/recovery or otherwise unverified wording is available; it is excluded from genuine human-review counts."
        )
        if not omission_reason:
            omission_reason = genuine_omission_reason

    comments = _review_comments(row)
    support_dimensions = _support_dimensions(row)
    analysis_context = _analysis_context(row)
    value: dict[str, Any] = {
        "slot_id": _text(row.get("slot_id"))
        or (_text(row.get("case_id")) or f"{dataset_id}::{condition}"),
        "case_id": row.get("case_id"),
        "dataset_id": dataset_id,
        "family_id": _text(row.get("family_id")),
        "concept_id": row.get("concept_id"),
        "condition": condition,
        # Preserve the source authoring nomination so the family-level
        # presentation audit evaluates the actual proposed primary wording,
        # rather than whichever generated variant happens to be first.
        "source_primary_candidate_id": _text(row.get("primary_candidate_id")) or None,
        "QUESTION_CANDIDATE_STATUS": question_status,
        "SCIENTIFIC_RELEASE_STATUS": release_status,
        "FORMAL_CANDIDATE_STATUS": (
            "WITHDRAWN_NEW_CASE_REQUIRED"
            if release_status == "NEW_CASE_REQUIRED"
            else "RETAINED_FOR_REVIEW"
        ),
        "ARTIFACT_REUSE_STATUS": reuse_status,
        "release_status_legacy": _matrix_status(row),
        # Preserve the condition-scoped dependency impact explicitly so the
        # portfolio can report pending SRAC work without consulting the source
        # matrix again or collapsing it into generic evidence blocking.
        "GT_SRAC_IMPACT": _text(row.get("GT_SRAC_IMPACT")) or "NOT_ESTABLISHED",
        "scientific_target": copy.deepcopy(row.get("scientific_target")),
        "scientific_scope": copy.deepcopy(row.get("scientific_scope")),
        "fixed_operationalization": copy.deepcopy(
            row.get("fixed_operationalization") or []
        ),
        "open_operationalization": _open_operationalization_dimensions(row),
        "finding_responsibility": copy.deepcopy(row.get("finding_responsibility")),
        "flow_expert_comments": comments,
        "support_dimensions": support_dimensions,
        **support_dimensions,
        "evidence_status": _text(row.get("evidence_sufficiency"))
        or "NOT_ESTABLISHED",
        "evidence_gaps": copy.deepcopy(comments["evidence_gaps"]),
        "evidence_ids": copy.deepcopy(comments["evidence_ids"]),
        "semantic_contract_sha256": row.get("semantic_contract_sha256"),
        "semantic_visibility": copy.deepcopy(row.get("semantic_visibility")),
        "analysis_context": analysis_context,
        # Explicit alias for model-facing consumers.  This is a placement
        # object only; neither spelling participates in the semantic hash.
        "model_visible_analysis_context": copy.deepcopy(analysis_context),
        # ``question_candidates`` is the genuine authored review surface.  A
        # recovery renderer is kept separately below so it cannot accidentally
        # become the singular/primary question handle.
        "question_candidates": [copy.deepcopy(candidate) for candidate in genuine_candidates],
            "recovery_question_candidates": [
                copy.deepcopy(candidate)
                for candidate in recovery_candidates
            ],
        # ``question_candidate`` is the stable singular genuine-question
        # handle.  It is intentionally null when only recovery prose exists.
        "question_candidate": copy.deepcopy(
            next(
                (
                    candidate for candidate in genuine_candidates
                ),
                None,
            )
        ),
        # Populated during portfolio quality gating.  Keeping the field
        # explicit avoids consumers mistaking an authored-but-unreviewed
        # candidate for an accepted primary.
        "accepted_question_candidate": None,
        "accepted_primary_question_candidate": None,
        "accepted_question_candidate_id": None,
        "question_candidate_omission_reason": omission_reason or None,
        "genuine_question_omission_reason": genuine_omission_reason,
        "canonical_case_mutated": bool(row.get("canonical_case_mutated", False)),
        "artifact_reuse_authorized": False,
        "human_question_selected": False,
        "official_scq_executed": False,
        "final_flow_expert_review_status": "NOT_ESTABLISHED",
        "semantic_fidelity_status": "NOT_ESTABLISHED",
        "hccq_status": "NOT_ESTABLISHED",
        "family_template_leakage_status": "NOT_ESTABLISHED",
        "genuine_authored_question": False,
        "question_candidate_provenance": {
            "candidate_count": len(candidates),
            "genuine_authored_count": sum(
                bool(candidate.get("genuine_authored_question"))
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ),
            "deterministic_fallback_count": sum(
                bool(candidate.get("authoring_provenance", {}).get("deterministic_fallback"))
                for candidate in candidates
                if isinstance(candidate, Mapping)
            ),
        },
    }
    # Keep only a digest of the source row.  Copying the full matrix row here
    # would make the review artifact needlessly large and could accidentally
    # carry a future downstream/GT field into the human-facing package.
    value["source_matrix_row_sha256"] = _stable_digest(row)
    return value


def _stable_digest(value: Any) -> str:
    import hashlib

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _question_text(candidate: Mapping[str, Any]) -> str:
    return _text(
        candidate.get("scientific_question")
        or candidate.get("question_text")
        or candidate.get("model_visible_text")
    )


def _genuine_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return authored question records eligible for strict portfolio counts.

    Recovery prose is intentionally kept in the artifact for debugging and
    human inspection, but it never enters a human-review question count.
    """

    return [
        candidate
        for candidate in (row.get("question_candidates") or ())
        if isinstance(candidate, Mapping)
        and candidate.get("genuine_authored_question") is True
    ]


def _accepted_candidates(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return authored candidates that passed every presentation gate.

    This is intentionally stricter than ``_genuine_candidates``: authored
    provenance proves that wording came from the authoring path, while an
    accepted primary additionally needs the dedicated final Flow Expert
    review, semantic fidelity, HCCQ, and family-leakage PASS verdicts.
    """

    return [
        candidate
        for candidate in _genuine_candidates(row)
        if _text(candidate.get("authoring_status")).upper() == "PASS"
        and _text(candidate.get("flow_expert_review_status")).upper() == "PASS"
        and _candidate_flow_expert_review_status(candidate)[0] == "PASS"
        and _text(candidate.get("semantic_fidelity_status")).upper() == "PASS"
        and _text(candidate.get("hccq_status")).upper() == "PASS"
        and _text(candidate.get("family_template_leakage_status")).upper() == "PASS"
    ]


def _candidate_passes_pre_family_gates(candidate: Mapping[str, Any]) -> bool:
    """Return whether a nominated wording is eligible for family comparison."""

    gate_fields_present = any(
        key in candidate
        for key in (
            "authoring_status",
            "semantic_fidelity_status",
            "hccq_status",
            "flow_expert_review_status",
            "final_flow_expert_wording_review",
        )
    )
    # Compatibility for direct family-audit callers that provide only text
    # and an explicit nomination.  Portfolio-normalized candidates always
    # carry the gate fields below and therefore take the strict path.
    if not gate_fields_present:
        return candidate.get("genuine_authored_question") is True
    return bool(
        candidate.get("genuine_authored_question") is True
        and _text(candidate.get("authoring_status")).upper() == "PASS"
        and _candidate_flow_expert_review_status(candidate)[0] == "PASS"
        and _text(candidate.get("semantic_fidelity_status")).upper() == "PASS"
        and _text(candidate.get("hccq_status")).upper() == "PASS"
    )


def _nominated_genuine_candidate(
    row: Mapping[str, Any],
    *,
    require_pre_family_gates: bool = False,
) -> Mapping[str, Any] | None:
    """Resolve the source producer's explicit primary-candidate nomination.

    Candidate ordering is not a nomination contract.  When an explicit
    primary id is present but missing or ineligible, fail closed rather than
    silently auditing the first generated variant.
    """

    nominated_id = _text(
        row.get("source_primary_candidate_id") or row.get("primary_candidate_id")
    )
    if not nominated_id:
        return None
    selected = next(
        (
            candidate
            for candidate in _genuine_candidates(row)
            if _text(candidate.get("candidate_id")) == nominated_id
        ),
        None,
    )
    if selected is None:
        return None
    if require_pre_family_gates and not _candidate_passes_pre_family_gates(selected):
        return None
    return selected


def _recovery_candidate_count(row: Mapping[str, Any]) -> int:
    return sum(
        1
        for candidate in (
            list(row.get("question_candidates") or ())
            + list(row.get("recovery_question_candidates") or ())
        )
        if isinstance(candidate, Mapping)
        and bool(
            (candidate.get("authoring_provenance") or {}).get(
                "deterministic_fallback"
            )
        )
    )


def _effective_o_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash the O portion of a row without including Finding responsibility."""

    # Matrix rows use the canonical name
    # ``unresolved_operationalization_dimensions``.  The shorter
    # ``open_operationalization`` spelling is retained only as a compatibility
    # alias in the portfolio projection.  Reading the alias alone would turn
    # every unresolved O space into ``[]`` and allow an O1-F2 mutation to pass
    # the exact-Effective-O inheritance check unnoticed.
    unresolved = _open_operationalization_dimensions(row)
    fixed = row.get("fixed_operationalization")
    if fixed is None:
        fixed = row.get("resolved_operationalization")

    return _stable_digest(
        {
            "scientific_target": row.get("scientific_target"),
            "scientific_scope": row.get("scientific_scope"),
            "fixed_operationalization": fixed or [],
            "open_operationalization": unresolved or [],
        }
    )


def _has_complete_public_semantic_contract(row: Mapping[str, Any]) -> bool:
    """Whether a portfolio row has enough public fields for family checks.

    The final human-review artifact intentionally does not copy the complete
    backend ``canonical_semantic_projection``.  Family invariants still need
    to be checked there, so use the frozen public target/scope/O/F fields as a
    compact contract.  Sparse placeholder rows (for example a missing source
    matrix slot) remain ``NOT_ESTABLISHED`` rather than being interpreted as
    semantic drift.
    """

    projection = row.get("canonical_semantic_projection")
    if isinstance(projection, Mapping):
        return True
    target = row.get("scientific_target")
    scope = row.get("scientific_scope")
    finding = row.get("finding_responsibility")
    fixed = row.get("fixed_operationalization")
    unresolved = _open_operationalization_dimensions(row)
    return bool(
        _text(target)
        and _text(scope)
        and finding not in (None, "")
        and isinstance(fixed, list)
        and isinstance(unresolved, list)
    )


def _apply_family_quality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Attach condition-family wording and O-inheritance audits to rows."""

    audits: dict[str, Any] = {}
    for dataset_id in DATASET_POLICIES:
        family_rows = [row for row in rows if _text(row.get("dataset_id")) == dataset_id]
        by_condition = {
            _text(row.get("condition")): row for row in family_rows
        }
        # Recovery/fallback prose is deliberately excluded from the family
        # audit.  It is useful for debugging but cannot establish that a
        # condition-specific authored family is free of template leakage.
        primary: dict[str, Mapping[str, Any]] = {}
        for condition, row in by_condition.items():
            selected = _nominated_genuine_candidate(
                row, require_pre_family_gates=True
            )
            if isinstance(selected, Mapping) and _question_text(selected):
                primary[condition] = selected
        presentation_inputs = {
            condition: {
                "model_visible_text": _question_text(candidate),
                "style": candidate.get("style"),
            }
            for condition, candidate in primary.items()
        }
        try:
            audit = audit_family_presentation_leakage(presentation_inputs)
        except ValueError as exc:
            audit = {
                "status": "INCOMPLETE",
                "error": f"{type(exc).__name__}: {exc}",
                "condition_identity_guessable_from_surface_templates": None,
            }
        # A family contract can only be judged when all of the required
        # comparison inputs are present.  A missing source-matrix row (or a
        # row with no canonical projection yet) means that the invariant is
        # currently untestable, not that it has drifted.  This distinction is
        # important for sparse/incomplete portfolios: explicit FAIL remains a
        # hard validation error, while NOT_ESTABLISHED remains an honest
        # pending state.
        o_status = "NOT_ESTABLISHED"
        o_failure: list[str] = []
        o1_f1 = by_condition.get("O1-F1")
        o1_f2 = by_condition.get("O1-F2")
        if (
            isinstance(o1_f1, Mapping)
            and isinstance(o1_f2, Mapping)
            and _has_complete_public_semantic_contract(o1_f1)
            and _has_complete_public_semantic_contract(o1_f2)
        ):
            left = _effective_o_fingerprint(o1_f1)
            right = _effective_o_fingerprint(o1_f2)
            if left == right:
                o_status = "PASS"
            else:
                o_status = "FAIL"
                o_failure.append("O1_F2_EFFECTIVE_O_DRIFT")
        target_conditions = ("O1-F1", "O2-F1", "O3-F1")
        target_rows = [by_condition.get(condition) for condition in target_conditions]
        target_inputs_complete = all(
            isinstance(item, Mapping)
            and item.get("scientific_target") is not None
            for item in target_rows
        )
        target_values = {
            _stable_digest(item.get("scientific_target"))
            for item in target_rows
            if isinstance(item, Mapping)
        }
        target_status = (
            "PASS"
            if target_inputs_complete and len(target_values) == 1
            else "FAIL"
            if target_inputs_complete and len(target_values) > 1
            else "NOT_ESTABLISHED"
        )
        target_failure = "TARGET_INVARIANCE_DRIFT" if target_status == "FAIL" else None
        for row in family_rows:
            selected_primary = primary.get(_text(row.get("condition")))
            selected_primary_id = (
                _text(selected_primary.get("candidate_id"))
                if isinstance(selected_primary, Mapping)
                else ""
            )
            row["family_target_invariance_status"] = target_status
            row["effective_o_inheritance_status"] = o_status
            if audit.get("status") == "PASS":
                family_status = "PASS"
            elif audit.get("status") == "PRESENTATION_LEAKAGE_OBSERVED":
                family_status = "REVISE"
            else:
                family_status = "NOT_ESTABLISHED"
            row["family_template_leakage_status"] = family_status
            row["family_presentation_audit"] = copy.deepcopy(audit)
            for candidate in row.get("question_candidates", ()):
                if not isinstance(candidate, dict):
                    continue
                # Only a genuine authored candidate can receive a family
                # quality verdict, and the family verdict belongs only to the
                # nominated primary wording that was actually compared.
                # Unselected variants remain inspectable but cannot borrow
                # the nominated question's leakage PASS.
                candidate["family_template_leakage_status"] = (
                    row["family_template_leakage_status"]
                    if candidate.get("genuine_authored_question") is True
                    and selected_primary_id
                    and _text(candidate.get("candidate_id")) == selected_primary_id
                    else "NOT_ESTABLISHED"
                )
                candidate["family_target_invariance_status"] = target_status
                candidate["effective_o_inheritance_status"] = o_status
                family_failures = []
                if target_failure:
                    family_failures.append(target_failure)
                if o_failure:
                    family_failures.extend(o_failure)
                if family_failures:
                    audit_record = candidate.setdefault(
                        "wording_responsibility_audit", {}
                    )
                    audit_record["failure_codes"] = sorted(
                        set(
                            [
                                *audit_record.get("failure_codes", []),
                                *family_failures,
                            ]
                        )
                    )
            # ``question_candidate`` is a stable handle for an authored
            # question only. Recovery/fallback prose remains available in the
            # plural list for diagnostics but must never become primary.
            if isinstance(selected_primary, Mapping):
                row["question_candidate"] = copy.deepcopy(selected_primary)
                row["genuine_question_candidate"] = copy.deepcopy(selected_primary)
            else:
                row["question_candidate"] = None
                row["genuine_question_candidate"] = None
        audits[dataset_id] = {
            **audit,
            "target_invariance_status": target_status,
            "target_invariance_failure": target_failure,
            "o1_f2_effective_o_status": o_status,
            "o1_f2_effective_o_failure_codes": o_failure,
        }
    return audits


def validate_human_review_portfolio_manifest(
    manifest: Mapping[str, Any], *, expected_slot_count: int = 28
) -> dict[str, Any]:
    """Validate coverage and status separation without judging science."""

    errors: list[str] = []
    rows = manifest.get("conditions")
    if not isinstance(rows, list):
        return {"status": "INVALID", "errors": ["conditions must be a list"]}
    if len(rows) != expected_slot_count:
        errors.append(f"EXPECTED_{expected_slot_count}_SLOTS_GOT_{len(rows)}")
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"ROW_{index}_NOT_OBJECT")
            continue
        key = _condition_key(row.get("dataset_id", ""), row.get("condition", ""))
        if key in seen:
            errors.append(f"DUPLICATE_SLOT:{key[0]}:{key[1]}")
        seen.add(key)
        q_status = _text(row.get("QUESTION_CANDIDATE_STATUS"))
        r_status = _text(row.get("SCIENTIFIC_RELEASE_STATUS"))
        a_status = _text(row.get("ARTIFACT_REUSE_STATUS"))
        if q_status not in QUESTION_CANDIDATE_STATUSES:
            errors.append(f"ROW_{index}_QUESTION_STATUS_INVALID")
        if r_status not in SCIENTIFIC_RELEASE_STATUSES:
            errors.append(f"ROW_{index}_RELEASE_STATUS_INVALID")
        if a_status not in ARTIFACT_REUSE_STATUSES:
            errors.append(f"ROW_{index}_REUSE_STATUS_INVALID")
        support = row.get("support_dimensions")
        if not isinstance(support, Mapping):
            errors.append(f"ROW_{index}_SUPPORT_DIMENSIONS_NOT_OBJECT")
            support = {}
        for name in SUPPORT_DIMENSION_NAMES:
            value = _text(support.get(name) or row.get(name))
            if value not in SUPPORT_DIMENSION_VALUES:
                errors.append(f"ROW_{index}_{name.upper()}_INVALID")
            if name in row and _text(row.get(name)) != value:
                errors.append(f"ROW_{index}_{name.upper()}_ALIAS_MISMATCH")
        analysis_context = row.get("analysis_context")
        if not isinstance(analysis_context, Mapping):
            errors.append(f"ROW_{index}_ANALYSIS_CONTEXT_NOT_OBJECT")
        elif not isinstance(analysis_context.get("support_dimensions"), Mapping):
            errors.append(f"ROW_{index}_ANALYSIS_CONTEXT_SUPPORT_MISSING")
        model_context = row.get("model_visible_analysis_context")
        if not isinstance(model_context, Mapping):
            errors.append(f"ROW_{index}_MODEL_VISIBLE_ANALYSIS_CONTEXT_NOT_OBJECT")
        elif isinstance(analysis_context, Mapping) and _stable_digest(model_context) != _stable_digest(analysis_context):
            errors.append(f"ROW_{index}_ANALYSIS_CONTEXT_ALIAS_MISMATCH")
        candidates = row.get("question_candidates")
        if not isinstance(candidates, list):
            errors.append(f"ROW_{index}_QUESTION_CANDIDATES_NOT_LIST")
            candidates = []
        omission = _text(row.get("question_candidate_omission_reason"))
        singular = row.get("question_candidate")
        genuine = _genuine_candidates(row)
        # The primary handle may point to any candidate that passed all gates,
        # not necessarily the first generated variant.  This matters when a
        # bounded authoring pass retains a rejected first variant and selects
        # the second one.  Resolve the accepted set before validating aliases;
        # a non-accepted authored row still uses its first variant as the
        # diagnostic handle unless the source matrix explicitly nominated a
        # different genuine variant.  Family-quality compilation uses that
        # same nominated text, so incomplete-family portfolios must not be
        # rejected merely because the nominated variant is not list element 0.
        accepted_for_handle = _accepted_candidates(row)
        nominated_for_handle = _nominated_genuine_candidate(row)
        expected_handle = (
            accepted_for_handle[0]
            if accepted_for_handle
            else nominated_for_handle
            if isinstance(nominated_for_handle, Mapping)
            else genuine[0]
            if genuine
            else None
        )
        if genuine and not isinstance(singular, Mapping):
            errors.append(f"ROW_{index}_SINGULAR_QUESTION_CANDIDATE_MISSING")
        if genuine and isinstance(singular, Mapping) and isinstance(expected_handle, Mapping):
            if _stable_digest(singular) != _stable_digest(expected_handle):
                errors.append(f"ROW_{index}_SINGULAR_QUESTION_CANDIDATE_MISMATCH")
        if not genuine and singular is not None:
            errors.append(f"ROW_{index}_SINGULAR_QUESTION_CANDIDATE_UNEXPECTED")
        genuine_handle = row.get("genuine_question_candidate")
        if genuine and not isinstance(genuine_handle, Mapping):
            errors.append(f"ROW_{index}_GENUINE_QUESTION_CANDIDATE_MISSING")
        if not genuine and genuine_handle is not None:
            errors.append(f"ROW_{index}_GENUINE_QUESTION_CANDIDATE_UNEXPECTED")
        if genuine and isinstance(genuine_handle, Mapping) and isinstance(expected_handle, Mapping):
            if _stable_digest(genuine_handle) != _stable_digest(expected_handle):
                errors.append(f"ROW_{index}_GENUINE_QUESTION_CANDIDATE_MISMATCH")
        # An accepted primary is a separate, stricter handle.  It may only
        # reference a candidate that passed authoring provenance, dedicated
        # final Flow Expert review, fidelity, HCCQ, and family leakage gates.
        accepted = accepted_for_handle
        accepted_handle = row.get("accepted_question_candidate")
        if accepted and not isinstance(accepted_handle, Mapping):
            errors.append(f"ROW_{index}_ACCEPTED_QUESTION_CANDIDATE_MISSING")
        if not accepted and accepted_handle is not None:
            errors.append(f"ROW_{index}_ACCEPTED_QUESTION_CANDIDATE_UNEXPECTED")
        if accepted and isinstance(accepted_handle, Mapping):
            if _stable_digest(accepted_handle) != _stable_digest(accepted[0]):
                errors.append(f"ROW_{index}_ACCEPTED_QUESTION_CANDIDATE_MISMATCH")
            accepted_id = _text(row.get("accepted_question_candidate_id"))
            if accepted_id and accepted_id != _text(accepted[0].get("candidate_id")):
                errors.append(f"ROW_{index}_ACCEPTED_QUESTION_CANDIDATE_ID_MISMATCH")
            if row.get("genuine_question_ready") is not True:
                errors.append(f"ROW_{index}_ACCEPTED_PRIMARY_NOT_READY")
            if isinstance(row.get("question_candidate"), Mapping) and _stable_digest(
                row["question_candidate"]
            ) != _stable_digest(accepted[0]):
                errors.append(f"ROW_{index}_PRIMARY_NOT_ACCEPTED_CANDIDATE")
        elif row.get("genuine_question_ready") is True:
            errors.append(f"ROW_{index}_READY_WITHOUT_ACCEPTED_PRIMARY")
        accepted_alias = row.get("accepted_primary_question_candidate")
        if accepted_alias is not None:
            if not isinstance(accepted_alias, Mapping) or not isinstance(
                accepted_handle, Mapping
            ):
                errors.append(f"ROW_{index}_ACCEPTED_PRIMARY_ALIAS_INVALID")
            elif _stable_digest(accepted_alias) != _stable_digest(accepted_handle):
                errors.append(f"ROW_{index}_ACCEPTED_PRIMARY_ALIAS_MISMATCH")
        if not candidates and not omission:
            errors.append(f"ROW_{index}_MISSING_CANDIDATE_OR_OMISSION_REASON")
        genuine_omission = _text(row.get("genuine_question_omission_reason"))
        if not _genuine_candidates(row) and not genuine_omission:
            errors.append(f"ROW_{index}_MISSING_GENUINE_QUESTION_OMISSION_REASON")
        semantic_hash = _text(row.get("semantic_contract_sha256"))
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                errors.append(f"ROW_{index}_CANDIDATE_{candidate_index}_NOT_OBJECT")
                continue
            if not _text(
                candidate.get("scientific_question")
                or candidate.get("question_text")
                or candidate.get("model_visible_text")
            ):
                errors.append(f"ROW_{index}_CANDIDATE_{candidate_index}_TEXT_MISSING")
            source = _text(candidate.get("candidate_source")).upper()
            is_fallback = source in _FALLBACK_SOURCE_MARKERS or "FALLBACK" in source
            if is_fallback and candidate.get("genuine_authored_question") is True:
                errors.append(
                    f"ROW_{index}_CANDIDATE_{candidate_index}_FALLBACK_MARKED_GENUINE"
                )
            if candidate.get("genuine_authored_question") is True and _text(
                candidate.get("authoring_status")
            ).upper() != "PASS":
                errors.append(
                    f"ROW_{index}_CANDIDATE_{candidate_index}_GENUINE_AUTHORING_UNPROVEN"
                )
            if candidate.get("genuine_authored_question") is True:
                provenance = candidate.get("authoring_provenance")
                if not isinstance(provenance, Mapping) or provenance.get(
                    "invocation_verified"
                ) is not True:
                    errors.append(
                        f"ROW_{index}_CANDIDATE_{candidate_index}_AUTHORING_INVOCATION_UNVERIFIED"
                    )
            # A final-review PASS must be backed by the dedicated review
            # record.  In particular, a fidelity-only call cannot be relabeled
            # as a scientific Flow Expert review by editing a shallow status.
            derived_review_status, _ = _candidate_flow_expert_review_status(candidate)
            declared_review_status = _text(
                candidate.get("flow_expert_review_status")
            ).upper()
            if declared_review_status == "PASS" and derived_review_status != "PASS":
                errors.append(
                    f"ROW_{index}_CANDIDATE_{candidate_index}_FINAL_REVIEW_PROVENANCE_INVALID"
                )
            if (
                candidate.get("genuine_authored_question") is True
                and declared_review_status == "PASS"
                and derived_review_status != declared_review_status
            ):
                errors.append(
                    f"ROW_{index}_CANDIDATE_{candidate_index}_FINAL_REVIEW_STATUS_MISMATCH"
                )
            candidate_hash = _text(candidate.get("semantic_contract_sha256"))
            if semantic_hash and candidate_hash and semantic_hash != candidate_hash:
                errors.append(f"ROW_{index}_CANDIDATE_{candidate_index}_SEMANTIC_HASH_DRIFT")
        if _text(row.get("dataset_id")) == "Combustor":
            if row.get("case_id") is not None:
                errors.append(f"ROW_{index}_COMBUSTOR_CASE_ID_INHERITED")
            for candidate in candidates:
                if isinstance(candidate, Mapping) and candidate.get("automatic_adoption") is not False:
                    errors.append(f"ROW_{index}_COMBUSTOR_AUTOMATIC_ADOPTION")
        if row.get("canonical_case_mutated") is not False:
            errors.append(f"ROW_{index}_CANONICAL_CASE_MUTATED")
        # These are deterministic family-contract checks, not advisory
        # wording opinions.  A source matrix that changes a fixed clause or
        # target across conditions must fail closed instead of being rendered
        # as a merely incomplete portfolio.
        if _text(row.get("effective_o_inheritance_status")).upper() == "FAIL":
            errors.append(f"ROW_{index}_O1_F2_EFFECTIVE_O_DRIFT")
        if _text(row.get("family_target_invariance_status")).upper() == "FAIL":
            errors.append(f"ROW_{index}_TARGET_INVARIANCE_DRIFT")

    expected = set(_expected_slots())
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing:
            errors.append("MISSING_SLOTS:" + ",".join(f"{d}/{c}" for d, c in missing))
        if extra:
            errors.append("UNEXPECTED_SLOTS:" + ",".join(f"{d}/{c}" for d, c in extra))

    counter_specs = {
        "TOTAL_SLOTS_ANALYZED": len(rows),
        "TOTAL_HUMAN_REVIEW_CANDIDATES": sum(
            bool(_genuine_candidates(row))
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_RELEASE_READY": sum(
            _text(row.get("SCIENTIFIC_RELEASE_STATUS")) == "ELIGIBLE"
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_PENDING_EVIDENCE": sum(
            _text(row.get("SCIENTIFIC_RELEASE_STATUS")) == "BLOCKED_PENDING_EVIDENCE"
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_PENDING_O_SPACE": sum(
            _text(row.get("SCIENTIFIC_RELEASE_STATUS")) == "BLOCKED_PENDING_O_SPACE"
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_PENDING_SRAC": sum(
            _pending_srac_required(row)
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_NEW_TARGET_REQUIRED": sum(
            _text(row.get("SCIENTIFIC_RELEASE_STATUS")) == "NEW_CASE_REQUIRED"
            for row in rows
            if isinstance(row, Mapping)
        ),
        "TOTAL_OMITTED": sum(
            not _genuine_candidates(row)
            for row in rows
            if isinstance(row, Mapping)
        ),
    }
    # The strict counters are intentionally validated from candidate-level
    # provenance.  This prevents a stale source-matrix counter from turning
    # fallback prose into a successful 28-question portfolio.
    counter_specs.update(
        {
            "TOTAL_QUESTION_SLOTS": len(rows),
            "TOTAL_GENUINE_AUTHORED_QUESTIONS": sum(
                bool(_genuine_candidates(row))
                for row in rows
                if isinstance(row, Mapping)
            ),
            "TOTAL_FLOW_EXPERT_REVIEWED": sum(
                any(
                    _text(item.get("flow_expert_review_status")).upper() == "PASS"
                    for item in _genuine_candidates(row)
                )
                for row in rows
                if isinstance(row, Mapping)
            ),
            "TOTAL_SEMANTIC_FIDELITY_PASS": sum(
                any(
                    _text(item.get("semantic_fidelity_status")).upper() == "PASS"
                    for item in _genuine_candidates(row)
                )
                for row in rows
                if isinstance(row, Mapping)
            ),
            "TOTAL_HCCQ_PASS": sum(
                any(
                    _text(item.get("hccq_status")).upper() == "PASS"
                    for item in _genuine_candidates(row)
                )
                for row in rows
                if isinstance(row, Mapping)
            ),
            "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS": sum(
                any(
                    _text(item.get("family_template_leakage_status")).upper()
                    == "PASS"
                    for item in _genuine_candidates(row)
                )
                for row in rows
                if isinstance(row, Mapping)
            ),
            "TOTAL_HUMAN_REVIEW_READY": sum(
                bool(row.get("genuine_question_ready"))
                for row in rows
                if isinstance(row, Mapping)
            ),
        }
    )
    # The long names are the canonical portfolio counters.  The shorter names
    # are retained as explicit aliases for consumers of the source matrix.
    counter_specs.update(
        {
            "TOTAL_CANDIDATES_PROPOSED": counter_specs[
                "TOTAL_HUMAN_REVIEW_CANDIDATES"
            ],
            "TOTAL_CANDIDATES_FOR_HUMAN_REVIEW": counter_specs[
                "TOTAL_HUMAN_REVIEW_CANDIDATES"
            ],
            "TOTAL_BLOCKED_PENDING_EVIDENCE": counter_specs[
                "TOTAL_PENDING_EVIDENCE"
            ],
            "TOTAL_BLOCKED_PENDING_O_SPACE_REVIEW": counter_specs[
                "TOTAL_PENDING_O_SPACE"
            ],
            "TOTAL_NEW_CASE_PENDING_SELECTION": counter_specs[
                "TOTAL_NEW_TARGET_REQUIRED"
            ],
        }
    )
    for key, expected_value in counter_specs.items():
        if manifest.get(key) != expected_value:
            errors.append(f"COUNTER_MISMATCH:{key}")
    return {
        "status": "PASS" if not errors else "INVALID",
        "errors": sorted(set(errors)),
        "checked_slot_count": len(rows),
    }


def _dataset_summary(rows: Sequence[Mapping[str, Any]], dataset_id: str) -> dict[str, Any]:
    selected = [row for row in rows if _text(row.get("dataset_id")) == dataset_id]
    return {
        "dataset_id": dataset_id,
        "condition_slots": {
            _text(row.get("condition")): {
                "QUESTION_CANDIDATE_STATUS": row.get("QUESTION_CANDIDATE_STATUS"),
                "SCIENTIFIC_RELEASE_STATUS": row.get("SCIENTIFIC_RELEASE_STATUS"),
                "ARTIFACT_REUSE_STATUS": row.get("ARTIFACT_REUSE_STATUS"),
            }
            for row in selected
        },
        # Strict count: fallback/recovery prose is never a human-review
        # question candidate.  Keep its count separate for diagnostics.
        "human_review_candidate_count": sum(
            bool(_genuine_candidates(row)) for row in selected
        ),
        "recovery_candidate_count": sum(
            _recovery_candidate_count(row) for row in selected
        ),
        "human_review_ready_count": sum(
            bool(row.get("genuine_question_ready")) for row in selected
        ),
        "release_ready_count": sum(
            row.get("SCIENTIFIC_RELEASE_STATUS") == "ELIGIBLE" for row in selected
        ),
    }


def render_human_review_markdown(manifest: Mapping[str, Any]) -> str:
    """Render an inspection-only Markdown surface from existing artifact data."""

    lines = [
        "# Human-Review Scientific Question Portfolio",
        "",
        "This artifact separates wording review from benchmark release. It does not select a question, mutate a case, or approve GT/SRAC reuse.",
        "",
        "## Status Axes",
        "",
        "- `QUESTION_CANDIDATE_STATUS` answers whether a human can inspect the question now.",
        "- `SCIENTIFIC_RELEASE_STATUS` answers whether the condition can enter the benchmark lifecycle.",
        "- `FORMAL_CANDIDATE_STATUS` records explicit withdrawal when a new canonical case is required; it never authorizes automatic case creation.",
        "- `ARTIFACT_REUSE_STATUS` describes semantic reuse only; authorization remains withheld.",
        "",
        "## Summary",
        "",
        f"- Slots analyzed: **{manifest.get('TOTAL_SLOTS_ANALYZED', 0)}**",
        f"- Genuine authored questions: **{manifest.get('TOTAL_GENUINE_AUTHORED_QUESTIONS', 0)}**",
        f"- Flow Expert reviewed: **{manifest.get('TOTAL_FLOW_EXPERT_REVIEWED', 0)}**",
        f"- Semantic fidelity PASS: **{manifest.get('TOTAL_SEMANTIC_FIDELITY_PASS', 0)}**",
        f"- HCCQ PASS: **{manifest.get('TOTAL_HCCQ_PASS', 0)}**",
        f"- Family leakage PASS: **{manifest.get('TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS', 0)}**",
        f"- Human-review ready: **{manifest.get('TOTAL_HUMAN_REVIEW_READY', 0)}**",
        f"- Recovery/fallback candidates (not genuine): **{manifest.get('TOTAL_RECOVERY_CANDIDATES', 0)}**",
        f"- Portfolio status: **{manifest.get('HUMAN_REVIEW_PORTFOLIO_STATUS', 'NOT_REPORTED')}**",
        f"- Release eligible: **{manifest.get('TOTAL_RELEASE_READY', 0)}**",
        f"- Pending evidence: **{manifest.get('TOTAL_PENDING_EVIDENCE', 0)}**",
        f"- Pending O-space review: **{manifest.get('TOTAL_PENDING_O_SPACE', 0)}**",
        f"- New target/case required: **{manifest.get('TOTAL_NEW_TARGET_REQUIRED', 0)}**",
        f"- Explicitly omitted: **{manifest.get('TOTAL_OMITTED', 0)}**",
        "",
        "## Complete Matrix",
        "",
        "The legacy phrase `Question review status` refers to the canonical human-review question status below.",
        "",
        "| Dataset | Condition | Human-review question status | Scientific support | Release status | Formal candidate disposition | Artifact reuse status | Question candidates |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in manifest.get("conditions", ()):
        if not isinstance(row, Mapping):
            continue
        candidates = row.get("question_candidates") or []
        primary_candidate = row.get("genuine_question_candidate")
        question = _question_text(primary_candidate) if isinstance(primary_candidate, Mapping) else ""
        if not question:
            question = (
                _text(row.get("genuine_question_omission_reason"))
                or _text(row.get("question_candidate_omission_reason"))
                or "(no genuine authored question)"
            )
        question = question.replace("|", "\\|").replace("\n", " ")
        support = row.get("support_dimensions")
        if isinstance(support, Mapping):
            support_summary = "; ".join(
                f"{_text(key)}={_text(support.get(key))}"
                for key in SUPPORT_DIMENSION_NAMES
            )
        else:
            support_summary = "NOT_ESTABLISHED"
        support_summary = support_summary.replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| {dataset} | {condition} | {qstatus} | {support} | {release} | {formal} | {reuse} | {question} |".format(
                dataset=_text(row.get("dataset_id")),
                condition=_text(row.get("condition")),
                qstatus=_text(row.get("QUESTION_CANDIDATE_STATUS")),
                support=support_summary,
                release=_text(row.get("SCIENTIFIC_RELEASE_STATUS")),
                formal=_text(row.get("FORMAL_CANDIDATE_STATUS")) or (
                    "WITHDRAWN_NEW_CASE_REQUIRED"
                    if _text(row.get("SCIENTIFIC_RELEASE_STATUS")) == "NEW_CASE_REQUIRED"
                    else "RETAINED_FOR_REVIEW"
                ),
                reuse=_text(row.get("ARTIFACT_REUSE_STATUS")),
                question=question,
            )
        )
    lines.extend(["", "## Candidate Details", ""])
    for row in manifest.get("conditions", ()):
        if not isinstance(row, Mapping):
            continue
        heading = f"### {_text(row.get('dataset_id'))} / {_text(row.get('condition'))}"
        lines.extend(
            [
                heading,
                "",
                f"- Question status: `{_text(row.get('QUESTION_CANDIDATE_STATUS'))}`",
                f"- Release status: `{_text(row.get('SCIENTIFIC_RELEASE_STATUS'))}`",
                f"- Formal candidate disposition: `{_text(row.get('FORMAL_CANDIDATE_STATUS')) or ('WITHDRAWN_NEW_CASE_REQUIRED' if _text(row.get('SCIENTIFIC_RELEASE_STATUS')) == 'NEW_CASE_REQUIRED' else 'RETAINED_FOR_REVIEW')}`",
                f"- Artifact reuse status: `{_text(row.get('ARTIFACT_REUSE_STATUS'))}` (authorization: `false`)",
                f"- Scientific target: {_text(row.get('scientific_target')) or 'Pending target selection.'}",
                f"- Fixed O: {json.dumps(row.get('fixed_operationalization') or [], ensure_ascii=False, sort_keys=True)}",
                f"- Open O: {json.dumps(row.get('open_operationalization') or [], ensure_ascii=False, sort_keys=True)}",
                f"- Support dimensions: {json.dumps(row.get('support_dimensions') or {}, ensure_ascii=False, sort_keys=True)}",
                f"- Analysis context: {json.dumps(row.get('analysis_context') or {}, ensure_ascii=False, sort_keys=True)}",
                f"- F responsibility: {json.dumps(row.get('finding_responsibility'), ensure_ascii=False, sort_keys=True)}",
                f"- Evidence status: `{_text(row.get('evidence_status'))}`",
                f"- Flow Expert comments: {_text((row.get('flow_expert_comments') or {}).get('rationale')) or 'None recorded.'}",
                f"- Final wording review: `{_text(row.get('final_flow_expert_review_status')) or 'NOT_ESTABLISHED'}`",
                f"- Semantic fidelity: `{_text(row.get('semantic_fidelity_status')) or 'NOT_ESTABLISHED'}`; HCCQ: `{_text(row.get('hccq_status')) or 'NOT_ESTABLISHED'}`; family leakage: `{_text(row.get('family_template_leakage_status')) or 'NOT_ESTABLISHED'}`",
                f"- GT/SRAC impact: {_text(row.get('GT_SRAC_IMPACT')) or 'NOT_ESTABLISHED'}",
                "",
            ]
        )
        candidates = row.get("question_candidates") or []
        genuine = _genuine_candidates(row)
        recovery = row.get("recovery_question_candidates") or [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and bool(
                (candidate.get("authoring_provenance") or {}).get(
                    "deterministic_fallback"
                )
            )
        ]
        if genuine:
            lines.append("Primary authored question:")
            lines.append("")
            for candidate in genuine[:1]:
                if not isinstance(candidate, Mapping):
                    continue
                lines.extend(
                    [
                        f"- `{_text(candidate.get('candidate_id'))}`: {_text(candidate.get('scientific_question') or candidate.get('question_text') or candidate.get('model_visible_text'))}",
                        f"  - source: `{_text(candidate.get('candidate_source'))}`; automatic adoption: `{candidate.get('automatic_adoption') is True}`",
                    ]
                )
        if recovery:
            lines.append("Recovery candidates (not genuine; excluded from counts):")
            lines.append("")
            for candidate in recovery:
                if not isinstance(candidate, Mapping):
                    continue
                lines.append(
                    f"- `{_text(candidate.get('candidate_id'))}`: {_question_text(candidate)}"
                )
            lines.append("")
        if not genuine:
            lines.extend(
                [
                    "No genuine authored question is proposed for this slot.",
                    f"Omission reason: {_text(row.get('genuine_question_omission_reason')) or _text(row.get('question_candidate_omission_reason')) or 'Not recorded.'}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Human Review Questions",
            "",
            "1. Is the scientific question meaningful?",
            "2. Is the target clear?",
            "3. Is the O responsibility appropriate?",
            "4. Would a flow scientist understand what is being asked?",
            "5. Should this condition proceed to benchmark construction?",
            "",
            "Human selection, target adoption, official SCQ, SRAC, curator confirmation, and model evaluation are downstream gates and were not performed by this artifact.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_human_review_portfolio(
    matrix_manifest: Mapping[str, Any] | str | Path,
    output_root: str | Path,
    *,
    candidate_count: int = 2,
) -> dict[str, Any]:
    """Build the complete 28-slot review artifact from a matrix manifest."""

    if candidate_count not in {2, 3}:
        raise ValueError("candidate_count must be 2 or 3")
    matrix = _load_manifest(matrix_manifest)
    raw_rows = matrix.get("conditions")
    if not isinstance(raw_rows, list):
        raise ValueError("matrix manifest must contain a conditions list")
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_keys: set[tuple[str, str]] = set()
    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        key = _condition_key(row.get("dataset_id", ""), row.get("condition", ""))
        if key in by_key:
            duplicate_keys.add(key)
        else:
            by_key[key] = row
    if duplicate_keys:
        labels = ",".join(f"{dataset}/{condition}" for dataset, condition in sorted(duplicate_keys))
        raise ValueError("DUPLICATE_SOURCE_MATRIX_SLOTS:" + labels)
    target_candidates = matrix.get("combustor_redesign_target_candidates")
    if not isinstance(target_candidates, list):
        target_candidates = []
    rows: list[dict[str, Any]] = []
    for dataset_id, condition in _expected_slots():
        source = by_key.get((dataset_id, condition))
        if source is None:
            source = {
                "slot_id": f"{dataset_id}::{condition}",
                "dataset_id": dataset_id,
                "condition": condition,
                "matrix_status": "OMITTED_SCIENTIFICALLY_UNSUPPORTED",
                "scientific_target": None,
                "canonical_case_mutated": False,
                "blocking_reason_codes": ["MATRIX_SLOT_MISSING"],
                "evidence_gaps": ["No source matrix row was available."],
            }
        rows.append(
            _row_from_matrix(
                source,
                target_candidates=[
                    item for item in target_candidates if isinstance(item, Mapping)
                ],
                candidate_count=candidate_count,
            )
        )

    family_quality = _apply_family_quality(rows)
    # A question is successful only when the actual authored candidate (not a
    # deterministic renderer or target placeholder) passed every presentation
    # review.  Keep row-level status explicit even when a blocked condition is
    # still useful for human inspection.
    for row in rows:
        candidates = [
            item for item in row.get("question_candidates", ())
            if isinstance(item, Mapping)
        ]
        genuine = [
            item for item in candidates
            if item.get("genuine_authored_question") is True
        ]
        recovery = [
            item
            for item in row.get("recovery_question_candidates", ())
            if bool(
                (item.get("authoring_provenance") or {}).get(
                    "deterministic_fallback"
                )
            )
        ]
        accepted = [
            item
            for item in genuine
            if _text(item.get("authoring_status")).upper() == "PASS"
            and _candidate_flow_expert_review_status(item)[0] == "PASS"
            and _text(item.get("semantic_fidelity_status")).upper() == "PASS"
            and _text(item.get("hccq_status")).upper() == "PASS"
            and _text(item.get("family_template_leakage_status")).upper() == "PASS"
        ]
        row["genuine_authored_question_count"] = len(genuine)
        row["flow_expert_reviewed_count"] = sum(
            _candidate_flow_expert_review_status(item)[0] == "PASS"
            for item in genuine
        )
        row["semantic_fidelity_pass_count"] = sum(
            _text(item.get("semantic_fidelity_status")).upper() == "PASS"
            for item in genuine
        )
        row["hccq_pass_count"] = sum(
            _text(item.get("hccq_status")).upper() == "PASS" for item in genuine
        )
        row["family_template_leakage_pass_count"] = sum(
            _text(item.get("family_template_leakage_status")).upper() == "PASS"
            for item in genuine
        )
        # Keep an explicit accepted-only handle.  Authored but failed wording
        # remains inspectable in ``question_candidates`` yet cannot be exposed
        # as the selected/primary question.
        nominated_primary = _nominated_genuine_candidate(row)
        nomination_id = _text(
            row.get("source_primary_candidate_id")
            or row.get("primary_candidate_id")
        )
        accepted_primary = next(
            (
                item
                for item in accepted
                if isinstance(nominated_primary, Mapping)
                and _text(item.get("candidate_id"))
                == _text(nominated_primary.get("candidate_id"))
            ),
            None,
        )
        if not nomination_id and accepted_primary is None and accepted:
            # Compatibility for older artifacts that predate explicit
            # nomination.  New matrix producers always provide an id.
            accepted_primary = accepted[0]
        row["genuine_question_ready"] = accepted_primary is not None
        row["accepted_question_candidate"] = (
            copy.deepcopy(accepted_primary) if accepted_primary else None
        )
        row["accepted_primary_question_candidate"] = (
            copy.deepcopy(accepted_primary) if accepted_primary else None
        )
        row["accepted_question_candidate_id"] = (
            _text(accepted_primary.get("candidate_id"))
            if isinstance(accepted_primary, Mapping)
            else None
        )
        # ``question_candidate`` / ``genuine_question_candidate`` are legacy
        # review handles.  When a candidate has passed every gate they point
        # to that accepted primary; before completion they retain the first
        # authored variant for human diagnosis, while the explicit accepted
        # handles above remain null.
        review_handle = accepted_primary or nominated_primary
        if review_handle is None and not nomination_id and genuine:
            review_handle = genuine[0]
        row["genuine_question_candidate"] = (
            copy.deepcopy(review_handle) if review_handle else None
        )
        row["question_candidate"] = (
            copy.deepcopy(review_handle) if review_handle else None
        )
        row["recovery_question_candidates"] = copy.deepcopy(recovery)
        # Mirror the primary authored candidate's explicit review statuses at
        # row level for shallow consumers.  Missing/failed gates remain visible
        # rather than being collapsed into a generic READY/PENDING label.
        # Recovery candidates are intentionally not promoted to the singular
        # primary question handle.
        # Row-level quality fields describe the candidate exposed as primary;
        # use the accepted candidate when available, otherwise keep the first
        # authored variant's failed/unknown status visible for diagnosis.
        primary = review_handle
        if isinstance(primary, Mapping):
            row["genuine_authored_question"] = bool(genuine)
            row["final_flow_expert_review_status"] = (
                _text(primary.get("flow_expert_review_status")).upper()
                or "NOT_ESTABLISHED"
            )
            row["semantic_fidelity_status"] = (
                _text(primary.get("semantic_fidelity_status")).upper()
                or "NOT_ESTABLISHED"
            )
            row["hccq_status"] = (
                _text(primary.get("hccq_status")).upper() or "NOT_ESTABLISHED"
            )
        else:
            row["genuine_authored_question"] = False
        row["question_candidate_provenance"] = {
            **(row.get("question_candidate_provenance") or {}),
            "genuine_authored_count": len(genuine),
            "accepted_genuine_count": len(accepted),
            "deterministic_fallback_count": sum(
                bool(item.get("authoring_provenance", {}).get("deterministic_fallback"))
                for item in row.get("recovery_question_candidates", ())
            ),
        }
        if accepted:
            # Human-review readiness is a presentation/semantic-quality axis,
            # independent of scientific release.  A source-matrix revision
            # label must not survive after an actually authored candidate has
            # passed every required gate.
            row["QUESTION_CANDIDATE_STATUS"] = "READY_FOR_HUMAN_REVIEW"
        elif _text(row.get("dataset_id")) == "Combustor":
            row["QUESTION_CANDIDATE_STATUS"] = "NEEDS_TARGET_REDESIGN"
        else:
            # A fallback can remain visible as an infrastructure recovery
            # surface, but it is never a successful human-review question.
            row["QUESTION_CANDIDATE_STATUS"] = (
                "NEEDS_SCIENTIFIC_REVISION" if genuine else "NEEDS_EVIDENCE_CONTEXT"
            )

    recommended_targets = _qualified_combustor_recommended_targets(
        [item for item in target_candidates if isinstance(item, Mapping)]
    )
    manifest: dict[str, Any] = {
        "schema_version": HUMAN_REVIEW_PORTFOLIO_VERSION,
        "artifact_type": "HUMAN_REVIEW_SCIENTIFIC_QUESTION_PORTFOLIO",
        "source_matrix_manifest": (
            Path(matrix_manifest).name
            if isinstance(matrix_manifest, (str, Path))
            else "IN_MEMORY"
        ),
        "source_matrix_manifest_sha256": _stable_digest(matrix),
        "maximum_candidate_slots": 28,
        "question_generation_decoupled_from_release": True,
        "primary_question_source_contract": (
            "genuine_question_candidate; recovery_question_candidates are audit-only"
        ),
        "human_final_selection_required": True,
        "human_question_selected": False,
        "automatic_target_adoption_performed": False,
        "canonical_case_mutation_count": 0,
        "conditions": rows,
        "family_template_leakage_audits": family_quality,
        # Keep redesign alternatives visible for accountable human review,
        # while never treating them as an adopted scientific identity.
        "combustor_redesign_target_candidates": copy.deepcopy(target_candidates),
        "combustor_recommended_provisional_target": copy.deepcopy(
            recommended_targets[0]
            if len(recommended_targets) == 1
            else None
        ),
    }
    manifest.update(
        {
            "TOTAL_SLOTS_ANALYZED": len(rows),
            # This is a strict human-review count.  Recovery/fallback prose is
            # exposed separately below and cannot inflate this value.
            "TOTAL_HUMAN_REVIEW_CANDIDATES": sum(
                bool(_genuine_candidates(row)) for row in rows
            ),
            "TOTAL_RELEASE_READY": sum(
                row["SCIENTIFIC_RELEASE_STATUS"] == "ELIGIBLE" for row in rows
            ),
            "TOTAL_PENDING_EVIDENCE": sum(
                row["SCIENTIFIC_RELEASE_STATUS"] == "BLOCKED_PENDING_EVIDENCE"
                for row in rows
            ),
            "TOTAL_PENDING_O_SPACE": sum(
                row["SCIENTIFIC_RELEASE_STATUS"] == "BLOCKED_PENDING_O_SPACE"
                for row in rows
            ),
            "TOTAL_NEW_TARGET_REQUIRED": sum(
                row["SCIENTIFIC_RELEASE_STATUS"] == "NEW_CASE_REQUIRED"
                for row in rows
            ),
            # A slot with only recovery prose is omitted from the genuine
            # human-review portfolio even when a debug candidate is present.
            "TOTAL_OMITTED": sum(not _genuine_candidates(row) for row in rows),
            "TOTAL_QUESTION_VARIANTS": sum(
                len(_genuine_candidates(row)) for row in rows
            ),
            "TOTAL_RECOVERY_CANDIDATES": sum(
                _recovery_candidate_count(row) for row in rows
            ),
            "TOTAL_EMPTY_SLOTS": sum(
                not row["question_candidates"] for row in rows
            ),
            "datasets": [
                _dataset_summary(rows, dataset_id) for dataset_id in DATASET_POLICIES
            ],
            "downstream": {
                "official_scq_executed_count": 0,
                "formal_srac_adjudication_executed_count": 0,
                "human_grounding_curator_confirmation_count": 0,
                "formal_model_experiment_executed_count": 0,
            },
        }
    )
    # Keep both the portfolio vocabulary and the source-matrix compatibility
    # vocabulary explicit.  ``PROPOSED`` counts non-omitted slots; it is not a
    # count of semantic PASS results.
    manifest.update(
        {
            "TOTAL_CANDIDATES_PROPOSED": manifest["TOTAL_HUMAN_REVIEW_CANDIDATES"],
            "TOTAL_CANDIDATES_FOR_HUMAN_REVIEW": manifest[
                "TOTAL_HUMAN_REVIEW_CANDIDATES"
            ],
            "TOTAL_BLOCKED_PENDING_EVIDENCE": manifest["TOTAL_PENDING_EVIDENCE"],
            "TOTAL_BLOCKED_PENDING_O_SPACE_REVIEW": manifest[
                "TOTAL_PENDING_O_SPACE"
            ],
            "TOTAL_NEW_CASE_PENDING_SELECTION": manifest[
                "TOTAL_NEW_TARGET_REQUIRED"
            ],
            # Strict quality counters are derived from candidate provenance;
            # compatibility counters above intentionally retain their old
            # slot-oriented meanings for existing consumers.
            "TOTAL_QUESTION_SLOTS": len(rows),
            "TOTAL_GENUINE_AUTHORED_QUESTIONS": sum(
                int(row.get("genuine_authored_question_count", 0)) > 0
                for row in rows
            ),
            "TOTAL_FLOW_EXPERT_REVIEWED": sum(
                int(row.get("flow_expert_reviewed_count", 0)) > 0
                for row in rows
            ),
            "TOTAL_SEMANTIC_FIDELITY_PASS": sum(
                int(row.get("semantic_fidelity_pass_count", 0)) > 0
                for row in rows
            ),
            "TOTAL_HCCQ_PASS": sum(
                int(row.get("hccq_pass_count", 0)) > 0 for row in rows
            ),
            "TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS": sum(
                int(row.get("family_template_leakage_pass_count", 0)) > 0
                for row in rows
            ),
            "TOTAL_HUMAN_REVIEW_READY": sum(
                bool(row.get("genuine_question_ready")) for row in rows
            ),
            "TOTAL_PENDING_SRAC": sum(_pending_srac_required(row) for row in rows),
            # The provisional target is a family-level redesign decision, not
            # one target per Combustor condition slot.
            "TOTAL_PROVISIONAL_TARGET": len(recommended_targets),
            "TOTAL_GENUINE_OMITTED": sum(
                not bool(_genuine_candidates(row)) for row in rows
            ),
            "HUMAN_REVIEW_PORTFOLIO_STATUS": (
                "COMPLETE"
                if sum(bool(row.get("genuine_question_ready")) for row in rows) == len(rows)
                else "INCOMPLETE_PENDING_AUTHORED_QUESTIONS"
            ),
            "TOTAL_PROVISIONAL_TARGETS": len(recommended_targets),
        }
    )
    # Keep the successful-slot contract explicit.  A portfolio with unresolved
    # authoring/review gates is a valid audit artifact, but it is not a
    # completed 28-question deliverable and must never be reported as PASS.
    manifest["HUMAN_REVIEW_PORTFOLIO_STATUS"] = (
        "COMPLETE"
        if (
            manifest["TOTAL_QUESTION_SLOTS"] == 28
            and manifest["TOTAL_GENUINE_AUTHORED_QUESTIONS"] == 28
            and manifest["TOTAL_FLOW_EXPERT_REVIEWED"] == 28
            and manifest["TOTAL_SEMANTIC_FIDELITY_PASS"] == 28
            and manifest["TOTAL_HCCQ_PASS"] == 28
            and manifest["TOTAL_FAMILY_TEMPLATE_LEAKAGE_PASS"] == 28
            and manifest["TOTAL_HUMAN_REVIEW_READY"] == 28
        )
        else "INCOMPLETE_PENDING_AUTHORED_QUESTIONS"
    )
    validation = validate_human_review_portfolio_manifest(manifest)
    manifest["validation"] = validation
    if validation["status"] != "PASS":
        raise ValueError("invalid human-review portfolio: " + "; ".join(validation["errors"]))

    out = Path(output_root).resolve()
    # The canonical portfolio is intentionally shallow.  Refuse to overwrite
    # unrelated files left by an earlier run: silently retaining stale
    # artifacts would make the directory look like one coherent portfolio
    # while its contents came from different source matrices.
    if out.exists() and not out.is_dir():
        raise ValueError(
            "HUMAN_REVIEW_PORTFOLIO_OUTPUT_ROOT_NOT_DIRECTORY:" + str(out)
        )
    if out.exists():
        unexpected = sorted(
            path.name
            for path in out.iterdir()
            if path.name not in PORTFOLIO_ARTIFACT_NAMES
        )
        if unexpected:
            raise ValueError(
                "HUMAN_REVIEW_PORTFOLIO_OUTPUT_DIRECTORY_NOT_SHALLOW:"
                + ",".join(unexpected)
            )
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "scientific_question_manifest.json", manifest)
    (out / "scientific_question_review.md").write_text(
        render_human_review_markdown(manifest), encoding="utf-8"
    )
    return manifest


def run_human_review_scientific_question_expansion(
    repository_root: str | Path,
    output_root: str | Path,
    *,
    matrix_manifest_path: str | Path | None = None,
    matrix_output_root: str | Path | None = None,
    refresh_matrix: bool = False,
    run_live_matrix: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 240.0,
    max_workers: int = 7,
    candidate_count: int = 2,
) -> dict[str, Any]:
    """Resolve/optionally build the source matrix, then write review artifacts."""

    root = Path(repository_root).resolve()
    matrix_path = (
        Path(matrix_manifest_path).resolve()
        if matrix_manifest_path is not None
        else _default_matrix_path(root)
    )
    if matrix_manifest_path is None and matrix_path is None:
        # Let the rebuild branch below create a source matrix when no prior
        # matrix exists.  Never default to the final portfolio directory.
        matrix_path = root / "outputs/current/scientific_question_matrix/scientific_question_manifest.json"
    # The command-line discovery helper refuses to select a final portfolio
    # as a source matrix.  Keep the Python API compatible with historical
    # callers that explicitly pass the old in-place manifest: those callers
    # have already selected their input and the builder remains byte-local and
    # does not mutate the supplied file.  New callers should use the staged
    # v3 candidate matrix (the CLI default).
    if matrix_manifest_path is not None and matrix_path.is_file() and not refresh_matrix:
        try:
            explicit_value = _read_json(matrix_path)
        except (OSError, ValueError, TypeError):
            explicit_value = None
        # A complete legacy in-place portfolio remains a readable compatibility
        # fixture.  A portfolio-shaped object without source rows is always an
        # accidental self-reference and must fail with the actionable error.
        if (
            isinstance(explicit_value, Mapping)
            and _is_portfolio_manifest(explicit_value)
            and not isinstance(explicit_value.get("conditions"), list)
        ):
            raise ValueError(
                "matrix manifest points to a final human-review portfolio; "
                "provide the source candidate matrix or use refresh_matrix"
            )
    if matrix_path.is_file() and not refresh_matrix:
        source: Mapping[str, Any] | str | Path = matrix_path
    else:
        from .scientific_question_candidate_matrix import (
            run_scientific_question_candidate_matrix,
        )
        import tempfile

        temporary_output = Path(matrix_output_root).resolve() if matrix_output_root else None
        if temporary_output is None:
            temporary_output = Path(tempfile.mkdtemp(prefix="flowintentbench-matrix-"))
        matrix = run_scientific_question_candidate_matrix(
            root,
            temporary_output,
            run_live=run_live_matrix,
            config_path=config_path,
            api_key=api_key,
            timeout=timeout,
            max_workers=max_workers,
            candidate_count=candidate_count,
        )
        source = matrix
    return build_human_review_portfolio(
        source,
        output_root,
        candidate_count=candidate_count,
    )


# Friendly aliases for callers that use the shorter task name.
build_human_review_question_portfolio = build_human_review_portfolio
run_human_review_question_expansion = run_human_review_scientific_question_expansion
validate_human_review_portfolio = validate_human_review_portfolio_manifest


__all__ = [
    "ARTIFACT_REUSE_STATUSES",
    "HUMAN_REVIEW_PORTFOLIO_VERSION",
    "QUESTION_CANDIDATE_STATUSES",
    "SCIENTIFIC_RELEASE_STATUSES",
    "SUPPORT_DIMENSION_NAMES",
    "SUPPORT_DIMENSION_VALUES",
    "build_human_review_portfolio",
    "build_human_review_question_portfolio",
    "render_human_review_markdown",
    "run_human_review_question_expansion",
    "run_human_review_scientific_question_expansion",
    "validate_human_review_portfolio",
    "validate_human_review_portfolio_manifest",
]
