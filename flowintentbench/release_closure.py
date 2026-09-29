"""Iterative, fail-closed construction loop for the formal 28-case portfolio.

The loop is intentionally an orchestration/audit layer.  It does not invent
scientific evidence, infer a Ground Truth rule, or create a curator decision.
Each run starts from the frozen human-facing 28-slot manifest and writes an
isolated staging portfolio.  External construction outputs can be added to a
staged/source case and the same iteration can then be rerun to advance only
the gates supported by those artifacts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context import EvidenceRecord
from .ground_truth import GroundTruth, GroundTruthValidator
from .lifecycle import (
    CuratorGateArtifact,
    LifecycleState,
    validate_grounding_curator_gate_artifact,
    validate_target_approval_lineage,
)
from .scientific_family import normative_lifecycle_state
from .reference_authority import DERIVED_REVIEW_RELATIVE_PATH, ARCHIVE_RELATIVE_PATH


TOTAL_REQUIRED_CASES = 28
CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")
_EVIDENCE_FILES = ("context_evidence.json", "operationalization_evidence.json", "finding_evidence.json")
FORMAL_RELEASE_LIFECYCLE_STATE = LifecycleState.RELEASE_ELIGIBLE.value
FORMAL_RELEASE_STATUS = "ELIGIBLE"


def _read_json(path: Path, default: Any = None) -> Any:
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


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_path(root: Path) -> Path:
    # Release-closure is the compatibility qualification stage.  It consumes
    # the explicitly archived pre-v7 lifecycle rows while frozen v7 case bytes
    # remain the sole model/evaluation authority.
    path = root / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions" / "human_facing_closure_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"frozen 28-slot manifest is missing: {path}")
    return path


def _target_review_manifest_path(root: Path) -> Path:
    """Return the optional upstream target-review artifact path.

    The human-facing closure manifest intentionally carries slot identity and
    question presentation only.  Combustor target selection is authored in the
    existing scientific-question review/matrix artifact, so the closure loop
    may use that record to bind a later human approval without promoting the
    review itself to a gate.
    """

    return root / DERIVED_REVIEW_RELATIVE_PATH / "scientific_question_manifest.json"


def _target_review_manifest_paths(root: Path) -> tuple[Path, ...]:
    """Return current and staged target-review artifacts in precedence order."""

    return (
        _target_review_manifest_path(root),
        root / "outputs/current/audits/scientific_question_candidate_matrix_v3.json",
        root / "outputs/current/scientific_question_review/scientific_question_manifest.json",
        # Compatibility authoring indexes retain the normalized target review
        # that predates the frozen v7 question index.  They are consulted only
        # to expose a review digest for the pending Combustor identity gate;
        # they never define the model-visible question or case bytes.
        root / "outputs/current/scientific_question_matrix/scientific_question_manifest.json",
        root / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions" / "scientific_question_manifest.json",
    )


def _target_review_for_row(root: Path, row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Load an existing Combustor target review for one slot, if available."""

    def unwrap(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        # The existing review pipeline stores the canonical recommendation in
        # ``normalized_review`` and wraps it with invocation/provenance fields.
        # Consume that projection without treating the wrapper as a new schema.
        nested = value.get("normalized_review")
        if isinstance(nested, Mapping):
            return nested
        nested = value.get("target_selection_review")
        if isinstance(nested, Mapping):
            return unwrap(nested)
        nested = value.get("combustor_target_review")
        if isinstance(nested, Mapping):
            return unwrap(nested)
        if any(
            key in value
            for key in (
                "recommendation_status",
                "recommended_target_id",
                "recommended_provisional_target_id",
            )
        ):
            return value
        return None

    direct = unwrap(
        row.get("target_selection_review")
        or row.get("combustor_target_review")
        or row.get("combustor_target_selection_review")
    )
    if direct is not None:
        return direct
    if str(row.get("dataset_id", "")).strip().lower() != "combustor":
        return None
    wanted_condition = str(row.get("condition", "")).strip().upper()
    for manifest_path in _target_review_manifest_paths(root):
        payload = _read_json(manifest_path, None)
        if isinstance(payload, Mapping):
            review = unwrap(
                payload.get("target_selection_review")
                or payload.get("combustor_target_review")
                or payload.get("combustor_target_selection_review")
            )
            if review is not None:
                return review
            # Candidate-matrix builds store the family-level review under the
            # recommended provisional target.  It is shared by all four
            # Combustor slots, but remains only a source recommendation until
            # a construction-lineage approval is supplied.
            recommended = payload.get("combustor_recommended_provisional_target")
            if isinstance(recommended, Mapping):
                review = unwrap(recommended.get("target_selection_review"))
                if review is not None:
                    return review
        rows = payload.get("conditions") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            continue
        for candidate in rows:
            if not isinstance(candidate, Mapping):
                continue
            if (
                str(candidate.get("dataset_id", "")).strip().lower() == "combustor"
                and str(candidate.get("condition", "")).strip().upper() == wanted_condition
            ):
                review = unwrap(
                    candidate.get("target_selection_review")
                    or candidate.get("combustor_target_review")
                    or candidate.get("combustor_target_selection_review")
                )
                if review is not None:
                    return review
    return None


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(_manifest_path(root), {})
    rows = manifest.get("conditions") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or len(rows) != TOTAL_REQUIRED_CASES:
        raise ValueError("frozen human-facing manifest must contain exactly 28 conditions")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest conditions must be objects")
        dataset_id = str(row.get("dataset_id", "")).strip()
        condition = str(row.get("condition", "")).strip().upper()
        if not dataset_id or condition not in CONDITIONS:
            raise ValueError(f"invalid manifest slot: {row!r}")
        key = (dataset_id, condition)
        if key in seen:
            raise ValueError(f"duplicate manifest slot: {key!r}")
        seen.add(key)
        result.append(dict(row))
    if len(seen) != TOTAL_REQUIRED_CASES:
        raise ValueError("manifest does not cover 28 unique dataset/condition slots")
    return result


def _candidate_case_dir(root: Path, dataset_id: str, condition: str) -> Path | None:
    candidate_root = root / "artifacts/reference/concept_expansion_phase1/candidate_cases" / dataset_id
    if not candidate_root.is_dir():
        return None
    suffix = condition.lower().replace("-", "_")
    matches = sorted(
        path.parent
        for path in candidate_root.glob(f"*_{suffix}/case_construction_metadata.json")
    )
    return matches[0] if matches else None


def _source_case(root: Path, row: Mapping[str, Any]) -> tuple[Path | None, str, str | None]:
    dataset_id = str(row["dataset_id"])
    condition = str(row["condition"]).upper()
    case_id = str(row.get("case_id") or "").strip()
    if case_id:
        directory = root / "datasets" / dataset_id / "construction" / "cases" / case_id
        if directory.is_dir():
            return directory, "PATTERN_RECONSTRUCTION_BASELINE", case_id
        # Some already-frozen rows are represented by the concept-expansion
        # candidate tree rather than an active dataset directory.  Preserve
        # their manifest identity while still recording that they are staged.
        candidate_root = root / "artifacts/reference/concept_expansion_phase1/candidate_cases" / dataset_id
        exact = candidate_root / case_id
        if (exact / "case_construction_metadata.json").is_file():
            return exact, "CANDIDATE_IDENTITY_FROZEN", case_id
    candidate = _candidate_case_dir(root, dataset_id, condition)
    if candidate is not None:
        metadata = _read_json(candidate / "case_construction_metadata.json", {})
        candidate_id = str(metadata.get("case_id") or candidate.name).strip()
        return candidate, "NEW_CANONICAL_IDENTITY_CANDIDATE", candidate_id
    return None, "MISSING_CASE_ARTIFACT", None


def _load_evidence(root: Path, dataset_id: str) -> tuple[list[EvidenceRecord], list[str]]:
    construction = root / "datasets" / dataset_id / "construction"
    records: list[EvidenceRecord] = []
    errors: list[str] = []
    for filename in _EVIDENCE_FILES:
        payload = _read_json(construction / filename, [])
        if not isinstance(payload, list):
            errors.append(f"{filename}:NOT_A_LIST")
            continue
        for index, item in enumerate(payload):
            try:
                records.append(EvidenceRecord.model_validate(item))
            except Exception as exc:  # pydantic error text is useful in the audit artifact
                errors.append(f"{filename}[{index}]:{type(exc).__name__}:{exc}")
    ids = [record.evidence_id for record in records]
    if len(ids) != len(set(ids)):
        errors.append("DUPLICATE_EVIDENCE_ID")
    return records, errors


def _finding_evidence_complete(gt: GroundTruth, records: Sequence[EvidenceRecord]) -> bool:
    known = {record.evidence_id: record for record in records}
    findings = [finding for branch in gt.findings_by_operationalization for finding in branch.findings]
    if not findings:
        return False
    return all(
        finding.evidence_ids
        and all(known.get(evidence_id) is not None and known[evidence_id].evidence_type == "finding" for evidence_id in finding.evidence_ids)
        for finding in findings
    )


def _operationalization_evidence_complete(gt: GroundTruth, records: Sequence[EvidenceRecord]) -> bool:
    known = {record.evidence_id: record for record in records}
    bundles = gt.acceptable_operationalizations
    return bool(bundles) and all(
        bundle.evidence_ids
        and all(known.get(evidence_id) is not None and known[evidence_id].evidence_type == "operationalization" for evidence_id in bundle.evidence_ids)
        for bundle in bundles
    )


def _external_curator_gate(case_dir: Path, root: Path, case_id: str) -> tuple[bool, str | None]:
    candidates = (
        case_dir / "grounding_curator_gate.json",
        root / "outputs/current/curator_gates" / f"{case_id}.json",
    )
    for path in candidates:
        if path.is_file():
            payload = _read_json(path, None)
            if not isinstance(payload, Mapping):
                return False, "CURATOR_ARTIFACT_INVALID_JSON"
            packet_candidates = (
                case_dir / "curator_packet.json",
                root / "outputs/current/curator_packets" / f"{case_id}.json",
                root / "outputs/current/kitchen/curator_packet.json" if case_id.startswith("kitchen_") else case_dir / "__no_packet__",
            )
            packet_path = next((item for item in packet_candidates if item.is_file()), None)
            if packet_path is None:
                return False, "CURATOR_PACKET_MISSING"
            packet = _read_json(packet_path, None)
            if not isinstance(packet, Mapping):
                return False, "CURATOR_PACKET_INVALID_JSON"
            validation = validate_grounding_curator_gate_artifact(packet, payload)
            if validation.get("status") != "PASS" or validation.get("grounding_curator_gate_status") != "CONFIRMED":
                errors = validation.get("errors") or ["CURATOR_GATE_NOT_CONFIRMED"]
                return False, str(errors[0])
            try:
                artifact = CuratorGateArtifact(
                    gate=str(payload.get("gate", "GROUNDING_CURATOR_CONFIRMED")),
                    reviewer_id=payload.get("reviewer_id"),
                    artifact_sha256=payload.get("artifact_sha256"),
                    timestamp=payload.get("timestamp"),
                    decision=str(payload.get("decision", "")),
                    revision_notes=payload.get("revision_notes"),
                    proxy_generated=bool(payload.get("proxy_generated", False)),
                )
            except (TypeError, ValueError) as exc:
                return False, f"CURATOR_ARTIFACT_INVALID:{type(exc).__name__}"
            if artifact.decision == "CONFIRMED":
                return True, None
            return False, f"CURATOR_DECISION_{artifact.decision}"
    return False, "CURATOR_CONFIRMATION_PENDING"


def _copy_case(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


def _lineage_state(
    path: Path,
    *,
    case_id: str,
    dataset_id: str,
    condition: str,
    target_approval_required: bool = False,
    expected_target_id: str | None = None,
    target_review: Mapping[str, Any] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Validate the explicit construction handoff marker, if supplied."""

    if not path.is_file():
        approval = validate_target_approval_lineage(
            None,
            required=target_approval_required,
            expected_target_id=expected_target_id,
            target_review=target_review,
        )
        return False, "CASE_LINEAGE_PENDING", approval
    payload = _read_json(path, None)
    if not isinstance(payload, Mapping):
        approval = validate_target_approval_lineage(
            None,
            required=target_approval_required,
            expected_target_id=expected_target_id,
            target_review=target_review,
        )
        return False, "CASE_LINEAGE_INVALID", approval
    approval = validate_target_approval_lineage(
        payload,
        required=target_approval_required,
        expected_target_id=expected_target_id,
        target_review=target_review,
    )
    if str(payload.get("case_id", "")) != case_id:
        return False, "CASE_LINEAGE_CASE_ID_MISMATCH", approval
    if str(payload.get("dataset_id", "")) != dataset_id:
        return False, "CASE_LINEAGE_DATASET_ID_MISMATCH", approval
    if str(payload.get("condition", "")).upper() != condition:
        return False, "CASE_LINEAGE_CONDITION_MISMATCH", approval
    if payload.get("semantic_identity_reauthored") is not True:
        return False, "CASE_LINEAGE_NOT_REAUTHORED", approval
    if str(payload.get("status", "")).upper() not in {"AUTHORED", "READY_FOR_FORMAL_RELEASE"}:
        return False, "CASE_LINEAGE_NOT_FORMAL", approval
    if target_approval_required and not approval["approved"]:
        errors = approval.get("errors") or [
            f"TARGET_APPROVAL_{approval['target_approval_status']}"
        ]
        return False, str(errors[0]), approval
    return True, None, approval


def project_formal_release_eligibility(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project explicit family lifecycle authority for one closure row.

    ``OFFICIAL_RELEASE_READY`` is retained as a Phase-2 closure counter for
    migration compatibility.  It is deliberately not a formal manifest
    authority.  Formal authority must be supplied explicitly by the upstream
    scientific-family lifecycle as ``RELEASE_ELIGIBLE`` together with the
    family release status ``ELIGIBLE``; absent fields never get inferred from
    a Phase-2 counter or from candidate status.
    """

    raw_lifecycle = (
        row.get("family_lifecycle_state")
        or row.get("lifecycle_status")
        or row.get("formal_release_lifecycle_state")
        or ""
    )
    try:
        # Normalize both the normative enum and the explicitly supported legacy
        # projection through one lifecycle authority.  Unknown values still
        # fail closed below.
        lifecycle = normative_lifecycle_state(raw_lifecycle).value
    except (TypeError, ValueError):
        lifecycle = str(getattr(raw_lifecycle, "value", raw_lifecycle) or "").strip().upper()
    raw_release_status = (
        row.get("formal_release_status")
        or row.get("family_release_status")
        or row.get("release_status")
        or ""
    )
    release_status = str(
        getattr(raw_release_status, "value", raw_release_status) or ""
    ).strip().upper()
    blockers: list[str] = []
    if lifecycle != FORMAL_RELEASE_LIFECYCLE_STATE:
        blockers.append("FORMAL_RELEASE_LIFECYCLE_NOT_RELEASE_ELIGIBLE")
    if release_status != FORMAL_RELEASE_STATUS:
        blockers.append("FORMAL_RELEASE_STATUS_NOT_ELIGIBLE")
    return {
        "formal_release_eligible": not blockers,
        "formal_release_status": (
            FORMAL_RELEASE_LIFECYCLE_STATE if not blockers else "NOT_ELIGIBLE"
        ),
        "lifecycle_status": lifecycle or None,
        "release_status": release_status or None,
        "blockers": blockers,
    }


def _case_audit(root: Path, stage_case: Path, row: Mapping[str, Any], source_kind: str, source_case_id: str | None) -> dict[str, Any]:
    metadata_path = stage_case / "case_construction_metadata.json"
    gt_path = stage_case / "ground_truth.json"
    case_id = str(source_case_id or stage_case.name)
    metadata = _read_json(metadata_path, None)
    gt_payload = _read_json(gt_path, None)
    metadata_errors: list[str] = []
    gt_errors: list[str] = []
    metadata_model = None
    gt_model = None
    try:
        from .case_design import CaseConstructionMetadata

        metadata_model = CaseConstructionMetadata.model_validate(metadata)
    except Exception as exc:
        metadata_errors.append(f"{type(exc).__name__}:{exc}")
    records, evidence_errors = _load_evidence(root, str(row["dataset_id"]))
    if metadata_model is not None:
        try:
            gt_model = GroundTruthValidator.validate(gt_payload, metadata_model, evidence_records=records)
        except Exception as exc:
            gt_errors.append(f"{type(exc).__name__}:{exc}")
    else:
        try:
            gt_model = GroundTruth.model_validate(gt_payload)
        except Exception as exc:
            gt_errors.append(f"{type(exc).__name__}:{exc}")

    gt_schema_valid = not metadata_errors and not gt_errors and gt_model is not None
    reference_analysis_present = (stage_case / "reference_analysis.md").is_file()
    data_mediated_proof = bool(
        gt_schema_valid
        and reference_analysis_present
        and gt_model.acceptable_operationalizations
        and all(branch.findings for branch in gt_model.findings_by_operationalization)
    )
    op_evidence = bool(gt_schema_valid and _operationalization_evidence_complete(gt_model, records))
    finding_evidence = bool(gt_schema_valid and _finding_evidence_complete(gt_model, records))
    evidence_complete = bool(op_evidence and finding_evidence and not evidence_errors)
    condition = str(row["condition"]).upper()
    contract_path = root / "outputs/current/qualified_scientific_cases/scientific_evaluation_contracts" / f"{case_id}.json"
    binding_path = root / "outputs/current/qualified_scientific_cases/scientific_artifact_bindings" / f"{case_id}.json"
    srac_required = condition in {"O2-F1", "O3-F1"}
    srac_ready = bool(contract_path.is_file()) if srac_required else True
    semantic_bound = bool(binding_path.is_file() and gt_schema_valid and evidence_complete)
    curator_confirmed, curator_reason = _external_curator_gate(stage_case, root, case_id)
    manifest_release_status = str(
        row.get("SCIENTIFIC_RELEASE_STATUS") or row.get("scientific_release_status") or ""
    ).strip().upper()
    grounding_eligible = manifest_release_status == "ELIGIBLE"
    grounding_confirmed = bool(curator_confirmed and evidence_complete)
    # Historical rows already have a frozen slot identity in the question
    # manifest.  The four Combustor rows deliberately remain pending because
    # the manifest has no canonical case_id for their replacement target.
    identity_status = "FROZEN" if source_kind in {"PATTERN_RECONSTRUCTION_BASELINE", "CANDIDATE_IDENTITY_FROZEN"} else "CANDIDATE_PENDING_IDENTITY_REVIEW"
    target_approval_required = str(row["dataset_id"]).strip().lower() == "combustor"
    # The frozen 28-slot row intentionally carries a slot/family identity, not
    # a human target decision.  Do not guess a candidate target ID from either
    # field; an external construction lineage must bind its own explicit
    # approval (and may additionally bind a target-review digest).
    expected_target_id = (
        str(
            row.get("target_candidate_id")
            or row.get("approved_target_id")
            or row.get("target_id")
            or ""
        ).strip()
        or None
    )
    target_review = _target_review_for_row(root, row)
    lineage_ready, lineage_reason, target_approval = _lineage_state(
        stage_case / "construction_lineage.json",
        case_id=case_id,
        dataset_id=str(row["dataset_id"]),
        condition=condition,
        target_approval_required=target_approval_required,
        expected_target_id=expected_target_id,
        target_review=target_review,
    )
    blockers: list[str] = []
    if identity_status != "FROZEN":
        blockers.append("CASE_IDENTITY_PENDING")
    if not lineage_ready:
        blockers.append(lineage_reason or "CASE_LINEAGE_PENDING")
    if target_approval_required and target_approval["target_approval_status"] != "APPROVED":
        blockers.append(f"TARGET_APPROVAL_{target_approval['target_approval_status']}")
    if not gt_schema_valid:
        blockers.append("GT_SCHEMA_INVALID" if metadata_model is not None else "GT_SCHEMA_PENDING")
    if not data_mediated_proof:
        blockers.append("DATA_MEDIATED_PROOF_PENDING")
    if not evidence_complete:
        blockers.append("EVIDENCE_CLOSURE_PENDING")
    if srac_required and not srac_ready:
        blockers.append("SRAC_PENDING")
    if not semantic_bound:
        blockers.append("GT_SEMANTIC_BINDING_PENDING")
    if not curator_confirmed:
        blockers.append(curator_reason or "CURATOR_CONFIRMATION_PENDING")
    if not grounding_confirmed:
        blockers.append("SCIENTIFIC_GROUNDING_PENDING")
    blockers = list(dict.fromkeys(blockers))
    phase2_closure_complete = not blockers
    formal_release = project_formal_release_eligibility(row)
    # Formal manifest authority is intentionally stricter than the existing
    # Phase-2 closure counter: explicit lifecycle eligibility must be present
    # in addition to all case-level closure checks.
    formal_release_eligible = bool(
        phase2_closure_complete and formal_release["formal_release_eligible"]
    )
    return {
        "case_id": case_id,
        "dataset_id": str(row["dataset_id"]),
        "condition": condition,
        "source_kind": source_kind,
        "source_case_id": str(row.get("case_id") or "") or None,
        "staged_case_path": str(stage_case),
        "identity_status": identity_status,
        "lineage_ready": lineage_ready,
        "lineage_reason": lineage_reason,
        "target_approval_required": target_approval_required,
        "target_approval_status": target_approval["target_approval_status"],
        "target_approval": target_approval,
        # Target selection authors identity only; it never substitutes for the
        # later, independently validated grounding curator gate.
        "target_approval_confers_grounding_curator_confirmation": False,
        "gt_schema_status": "VALID" if gt_schema_valid else ("INVALID" if metadata_model is not None else "PENDING"),
        "gt_schema_valid": gt_schema_valid,
        "reference_analysis_present": reference_analysis_present,
        "data_mediated_proof": data_mediated_proof,
        "operationalization_evidence_complete": op_evidence,
        "finding_evidence_complete": finding_evidence,
        "evidence_complete": evidence_complete,
        "evidence_errors": evidence_errors,
        "srac_required": srac_required,
        "srac_status": "NOT_REQUIRED" if not srac_required else ("READY" if srac_ready else "PENDING"),
        "gt_semantic_bound": semantic_bound,
        "curator_confirmed": curator_confirmed,
        "curator_reason": curator_reason,
        "scientific_grounding_status": "CONFIRMED" if grounding_confirmed else ("ELIGIBLE_FOR_CURATOR_REVIEW" if grounding_eligible else "PENDING"),
        "scientific_grounding_eligible": grounding_eligible,
        "scientific_grounding_confirmed": grounding_confirmed,
        # Compatibility field: this means only that the Phase-2 closure
        # blockers are empty.  It is not formal manifest authority.
        "phase2_closure_complete": phase2_closure_complete,
        "phase2_status": "OFFICIAL_RELEASE_READY" if phase2_closure_complete else "PENDING",
        "release_authorized": formal_release_eligible,
        "formal_release_eligibility": formal_release,
        "formal_release_eligible": formal_release_eligible,
        "release_status": "RELEASE_ELIGIBLE" if formal_release_eligible else "PENDING",
        "blockers": blockers,
        "validation_errors": metadata_errors + gt_errors,
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Formal release closure loop",
        "",
        "This is an isolated construction audit. It never fabricates scientific evidence or curator confirmation.",
        "",
        f"- Iteration: **{report['iteration']}**",
        f"- Input fingerprint: `{report['input_fingerprint']}`",
        f"- Candidate portfolio: **{report.get('CANDIDATE_TOTAL', report['TOTAL_CASES'])}**",
        f"- Selected release subset: **{report.get('SELECTED_RELEASE_CASE_COUNT', report['TOTAL_CASES'])}**",
        f"- Phase-2 `OFFICIAL_RELEASE_READY` counter: **{report['OFFICIAL_RELEASE_READY']}/{report.get('SELECTED_RELEASE_CASE_COUNT', report['TOTAL_CASES'])}**",
        f"- Formal manifest authority (`RELEASE_ELIGIBLE`): **{report.get('FORMAL_RELEASE_ELIGIBLE', 0)}/{report.get('SELECTED_RELEASE_CASE_COUNT', report['TOTAL_CASES'])}**",
        f"- Controlled-pilot only: **{report.get('CONTROLLED_PILOT_ONLY', 0)}**",
        f"- Target: **{report['target_status']}**",
        "",
        "## Counters",
        "",
        "| Counter | Value |",
        "|---|---:|",
    ]
    for key in (
        "CANDIDATE_TOTAL", "SELECTED_RELEASE_CASE_COUNT", "CONTROLLED_PILOT_ONLY",
        "PHASE2_CLOSURE_COMPLETE", "FORMAL_RELEASE_ELIGIBLE",
        "SCIENTIFIC_CASE_IDENTITY_FROZEN", "TOTAL_PENDING_CASE_IDENTITY", "TOTAL_GT_SCHEMA_VALID",
        "TOTAL_GT_SCHEMA_PENDING", "TOTAL_GT_SCHEMA_INVALID", "TOTAL_DATA_MEDIATED_PROOF",
        "TOTAL_DATA_MEDIATED_PROOF_PENDING", "TOTAL_EVIDENCE_CLOSURE_COMPLETE", "TOTAL_PENDING_EVIDENCE",
        "TOTAL_SRAC_READY", "TOTAL_PENDING_SRAC", "TOTAL_GT_SEMANTIC_BOUND", "TOTAL_PENDING_GT_SEMANTIC_BINDING",
        "TOTAL_CURATOR_CONFIRMED", "TOTAL_PENDING_CURATOR", "TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE",
        "TOTAL_SCIENTIFIC_GROUNDING_CONFIRMED", "TOTAL_PENDING_SCIENTIFIC_GROUNDING",
        "TOTAL_RELEASE_AUTHORIZED", "TOTAL_PENDING_RELEASE_AUTHORIZATION", "OFFICIAL_RELEASE_READY",
    ):
        lines.append(f"| `{key}` | {report.get(key, 0)} |")
    lines.extend([
        "",
        "## Case blockers",
        "",
        "| Case | Dataset | Condition | Phase-2 status | Formal status | Blockers |",
        "|---|---|---|---|---|---|",
    ])
    for case in report["cases"]:
        blockers = ", ".join(f"`{item}`" for item in case["blockers"]) or "none"
        lines.append(
            f"| `{case['case_id']}` | `{case['dataset_id']}` | {case['condition']} | "
            f"**{case.get('phase2_status', 'PENDING')}** | "
            f"**{case.get('release_status', 'PENDING')}** | {blockers} |"
        )
    lines.extend([
        "",
        "## Iteration protocol",
        "",
        "1. Rebuild/reauthor a case in staging and add `construction_lineage.json` only after its semantic identity is frozen.",
        "2. Add traceable source/evidence records and bind every finding that needs scientific support.",
        "3. Add the existing SRAC/semantic-binding artifacts where required.",
        "4. Supply an accountable, non-proxy curator gate bound to the authoritative packet.",
        "5. Rerun this loop with the next iteration number; only supported gates advance.",
        "",
        "Historical active cases remain controlled-pilot material until a complete replacement portfolio is validated.",
    ])
    return "\n".join(lines) + "\n"


def run_release_closure_loop(
    repository_root: str | Path,
    output_root: str | Path | None = None,
    *,
    iteration: int = 1,
    selected_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one deterministic closure iteration and write isolated artifacts.

    The frozen 28-row manifest remains the candidate audit surface.  When
    ``selected_case_ids`` is supplied, it is an explicit sparse release
    selection (case IDs or stable slot IDs); unselected rows remain visible as
    controlled-pilot material and cannot affect the selected release
    denominator.  No scientific gate is weakened by this option.
    """

    if iteration < 1:
        raise ValueError("iteration must be >= 1")
    root = Path(repository_root).resolve()
    out = Path(output_root) if output_root is not None else root / "outputs/current/formal_release_closure"
    out = out.resolve()
    rows = _manifest_rows(root)
    manifest_payload = _read_json(_manifest_path(root), {})
    available_ids = {
        str(row.get("case_id") or row.get("slot_id") or "").strip()
        for row in rows
    }
    if selected_case_ids is None:
        selected_ids = set(available_ids)
    else:
        selected_ids = {str(value).strip() for value in selected_case_ids if str(value).strip()}
        if not selected_ids:
            raise ValueError("selected_case_ids must contain at least one case or slot ID")
        unknown = sorted(selected_ids - available_ids)
        if unknown:
            raise ValueError(f"selected_case_ids contains unknown IDs: {unknown}")
    input_fingerprint = _sha256_json({
        "manifest": manifest_payload,
        # The optional review is an upstream input to Combustor identity
        # binding.  Include it in the fingerprint so a changed review cannot
        # masquerade as the same closure iteration.
        "target_review_manifest": _read_json(_target_review_manifest_path(root), None),
        "iteration": iteration,
        "selected_case_ids": sorted(selected_ids),
    })
    stage_root = out / f"iteration-{iteration:03d}"
    cases_root = stage_root / "cases"
    case_reports: list[dict[str, Any]] = []
    for row in rows:
        source, source_kind, source_case_id = _source_case(root, row)
        case_id = source_case_id or f"{row['dataset_id'].lower()}_{str(row['condition']).lower().replace('-', '_')}"
        stage_case = cases_root / case_id
        if source is not None:
            _copy_case(source, stage_case)
        else:
            stage_case.mkdir(parents=True, exist_ok=True)
        lineage = {
            "schema_version": "formal-release-case-lineage-v1",
            "case_id": case_id,
            "dataset_id": str(row["dataset_id"]),
            "condition": str(row["condition"]).upper(),
            "source_kind": source_kind,
            "source_case_id": str(row.get("case_id") or "") or None,
            "semantic_identity_reauthored": False,
            "status": "PENDING_EXTERNAL_CONSTRUCTION",
        }
        _write_json(stage_case / "closure_lineage.json", lineage)
        case_report = _case_audit(root, stage_case, row, source_kind, source_case_id)
        stable_id = str(row.get("case_id") or row.get("slot_id") or "").strip()
        case_report["selected_for_release"] = stable_id in selected_ids
        if stable_id not in selected_ids:
            case_report["release_authorized"] = False
            case_report["phase2_closure_complete"] = False
            case_report["phase2_status"] = "CONTROLLED_PILOT_ONLY"
            case_report["formal_release_eligible"] = False
            case_report["release_status"] = "CONTROLLED_PILOT_ONLY"
            case_report["blockers"] = [*case_report["blockers"], "NOT_SELECTED_FOR_RELEASE"]
        case_reports.append(case_report)

    counters: dict[str, int] = {
        "TOTAL_CASES": len(case_reports),
        "CANDIDATE_TOTAL": len(case_reports),
        "SELECTED_RELEASE_CASE_COUNT": len(selected_ids),
        "CONTROLLED_PILOT_ONLY": len(case_reports) - len(selected_ids),
        "PHASE2_CLOSURE_COMPLETE": sum(
            item["phase2_closure_complete"] and item.get("selected_for_release", False)
            for item in case_reports
        ),
        "FORMAL_RELEASE_ELIGIBLE": sum(
            item["formal_release_eligible"] and item.get("selected_for_release", False)
            for item in case_reports
        ),
        "SCIENTIFIC_CASE_IDENTITY_FROZEN": sum(item["identity_status"] == "FROZEN" for item in case_reports),
        "TOTAL_PENDING_CASE_IDENTITY": sum(item["identity_status"] != "FROZEN" for item in case_reports),
        "TOTAL_GT_SCHEMA_VALID": sum(item["gt_schema_status"] == "VALID" for item in case_reports),
        "TOTAL_GT_SCHEMA_PENDING": sum(item["gt_schema_status"] == "PENDING" for item in case_reports),
        "TOTAL_GT_SCHEMA_INVALID": sum(item["gt_schema_status"] == "INVALID" for item in case_reports),
        "TOTAL_DATA_MEDIATED_PROOF": sum(item["data_mediated_proof"] for item in case_reports),
        "TOTAL_DATA_MEDIATED_PROOF_PENDING": sum(not item["data_mediated_proof"] for item in case_reports),
        "TOTAL_EVIDENCE_CLOSURE_COMPLETE": sum(item["evidence_complete"] for item in case_reports),
        "TOTAL_PENDING_EVIDENCE": sum(not item["evidence_complete"] for item in case_reports),
        "TOTAL_SRAC_READY": sum(item["srac_status"] == "READY" for item in case_reports),
        "TOTAL_PENDING_SRAC": sum(item["srac_status"] == "PENDING" for item in case_reports),
        "TOTAL_GT_SEMANTIC_BOUND": sum(item["gt_semantic_bound"] for item in case_reports),
        "TOTAL_PENDING_GT_SEMANTIC_BINDING": sum(not item["gt_semantic_bound"] for item in case_reports),
        "TOTAL_CURATOR_CONFIRMED": sum(item["curator_confirmed"] for item in case_reports),
        "TOTAL_PENDING_CURATOR": sum(not item["curator_confirmed"] for item in case_reports),
        "TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE": sum(item["scientific_grounding_eligible"] for item in case_reports),
        "TOTAL_SCIENTIFIC_GROUNDING_CONFIRMED": sum(item["scientific_grounding_confirmed"] for item in case_reports),
        "TOTAL_PENDING_SCIENTIFIC_GROUNDING": sum(not item["scientific_grounding_confirmed"] for item in case_reports),
        "TOTAL_RELEASE_AUTHORIZED": sum(
            item["release_authorized"] and item.get("selected_for_release", False)
            for item in case_reports
        ),
        "TOTAL_PENDING_RELEASE_AUTHORIZATION": sum(
            not item["release_authorized"] and item.get("selected_for_release", False)
            for item in case_reports
        ),
    }
    # Backward-compatible Phase-2 counter.  Formal manifest authority is the
    # separate ``FORMAL_RELEASE_ELIGIBLE`` counter above.
    counters["OFFICIAL_RELEASE_READY"] = sum(
        item["phase2_closure_complete"]
        and item.get("selected_for_release", False)
        for item in case_reports
    )
    blockers: dict[str, int] = {}
    for item in case_reports:
        for blocker in item["blockers"]:
            blockers[blocker] = blockers.get(blocker, 0) + 1
    report: dict[str, Any] = {
        "schema_version": "formal-release-closure-loop-v1",
        "artifact_type": "FORMAL_RELEASE_CLOSURE_LOOP",
        "iteration": iteration,
        "target_total_cases": TOTAL_REQUIRED_CASES,
        "target_status": (
            "OFFICIAL_RELEASE_READY"
            if counters["OFFICIAL_RELEASE_READY"] == len(selected_ids)
            else "PENDING_BLOCKERS"
        ),
        "target_status_scope": "PHASE2_CLOSURE_COUNTER",
        "formal_release_authority": (
            FORMAL_RELEASE_LIFECYCLE_STATE
            if counters["FORMAL_RELEASE_ELIGIBLE"] == len(selected_ids)
            else "PENDING_RELEASE_ELIGIBILITY"
        ),
        "release_selection": {
            "selected_case_ids": sorted(selected_ids),
            "selected_case_count": len(selected_ids),
            "candidate_total": len(case_reports),
        },
        "input_fingerprint": input_fingerprint,
        "cases": case_reports,
        "blocker_counts": blockers,
        **counters,
    }
    _write_json(stage_root / "closure.json", report)
    (stage_root / "closure.md").write_text(_render_markdown(report), encoding="utf-8")
    _write_json(out / "latest.json", report)
    (out / "latest.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "FORMAL_RELEASE_LIFECYCLE_STATE",
    "FORMAL_RELEASE_STATUS",
    "TOTAL_REQUIRED_CASES",
    "project_formal_release_eligibility",
    "run_release_closure_loop",
]
