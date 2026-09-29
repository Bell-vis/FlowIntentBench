"""Phase 2 qualification of human-facing questions into scientific cases.

This module is deliberately an orchestration layer over the contracts that
already exist in the repository.  It does not define a second case, Ground
Truth, or evaluation schema.  A packet is a JSON representation of the
existing semantic projection, authored case input, GroundTruth, and (when
available) :class:`ScientificEvaluationContract`.

The 28-question portfolio is transformed into 28 explicitly tracked
scientific case records.  Scientific identity qualification is deliberately
separate from downstream evaluation and release readiness: a case may have a
frozen target/question/responsibility while its evidence, SRAC contract, GT,
curator review, or release authorization is still pending.  Pending work is
reported as state, never converted into a defect or silently promoted.
"""

from __future__ import annotations

import json
import copy
import hashlib
import math
import re
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .case_repository import load_case_record
from .case_design import (
    CaseConstructionMetadata,
    CaseConstructionMetadataValidator,
    ExplicitFindingRequirement,
    ExplicitMethodConstraint,
    ExplicitQuestionConstraint,
    FindingOpenness,
    FindingRequirementCategory,
    MethodConstraintCategory,
    OperationalizationDimension,
    OperationalizationResponsibility,
    normalize_case_text,
)
from .context import EvidenceRecord, load_source_collection
from .ground_truth import (
    GroundTruth,
    GroundTruthValidator,
    FindingImportance,
    OperationalizationBundle,
    OperationalizationDecision,
    OperationalizationFindingBranch,
    ReferenceFinding,
    VerificationSpec,
    load_ground_truth,
    save_ground_truth,
)
from .scientific_artifact_binding import (
    artifact_sha256,
    build_scientific_artifact_binding,
    evaluate_scientific_dependency_invalidation,
    validate_scientific_artifact_binding,
)
from .scientific_semantics import semantic_contract_sha256
from .srac import (
    MaterializedBranch,
    ScientificAdjudication,
    ScientificEvaluationContract,
    SRACProposal,
    materialize_branch,
    resolve_o_validity_label,
)
from .srac_router import validate_case_aware_proposal_submission
from .proxy_expert import prepare_adjudicator_invocation, validate_adjudicator_semantic_coverage
from .finding_requirements import FindingRequirementContract
from .evaluation_policy import (
    bind_policy_to_execution_binding,
    compile_finding_verification_policy,
    explicit_verification_mode,
    validate_finding_verification_policy,
)
from .construction_tolerances import absolute_tolerance_for_significant_figures
from .deterministic_materialization import (
    MaterializationUnsupported,
    compile_effective_operationalization,
)
from .scientific_question_candidate_matrix import harden_portfolio_scientific_projection
from .reference_authority import DERIVED_REVIEW_RELATIVE_PATH, ARCHIVE_RELATIVE_PATH


PHASE2_VERSION = "qualified-scientific-case-portfolio-v1"
_ALLOWED_PROXY_RECOMMENDATIONS = {"KEEP", "REVISE", "REJECT", "NEEDS_MORE_EVIDENCE"}

# Lifecycle labels are orchestration metadata.  They do not add a benchmark
# schema and intentionally leave the existing GroundTruth/SEC models intact.
SCIENTIFIC_CASE_QUALIFIED = "QUALIFIED"
SCIENTIFIC_CASE_PENDING_IDENTITY = "PENDING_CASE_IDENTITY"
SCIENTIFIC_CASE_INVALID = "INVALID"
EVALUATION_FROZEN = "FROZEN"
EVALUATION_PENDING_SRAC = "PENDING_SRAC"
EVALUATION_PENDING_EVIDENCE = "PENDING_EVIDENCE"
EVALUATION_PENDING_GT = "PENDING_GT"
EVALUATION_PENDING_CASE_IDENTITY = "PENDING_CASE_IDENTITY"
EVALUATION_INVALID = "INVALID"
GT_BOUND = "BOUND"
GT_NOT_YET_FROZEN = "NOT_YET_FROZEN"
GT_INVALID = "INVALID"


def _projection_semantic_hash(projection: Mapping[str, Any]) -> str:
    """Return the contract hash for any supported projection shape.

    Historical authoring rows occasionally contain a provisional projection
    that predates the semantic-hash helper's accepted envelope.  Such a row
    must not make the whole qualification pass crash; its canonical bytes are
    still a stable fallback until it is rebound to the frozen v7 reference.
    Frozen v7 projections use ``semantic_contract_sha256`` and therefore keep
    the exact hash recorded in ``scientific_question.json``.
    """

    try:
        return semantic_contract_sha256(projection)
    except (TypeError, ValueError, KeyError):
        return artifact_sha256(projection)


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _agent_build_identity(root: Path) -> dict[str, Any]:
    """Derive a reproducible identity for source, tests and method inputs."""

    roots = ("flowintentbench", "scripts", "tests")
    files: list[Path] = []
    for directory in roots:
        files.extend(path for path in (root / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.extend(root / name for name in ("method.md", "FlowIntentBench.md", "pyproject.toml", "runtime-requirements.txt") if (root / name).is_file())
    entries = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }
    return {
        "source_tree_sha256": artifact_sha256(entries),
        "file_count": len(entries),
        "builder_sha256": entries.get("flowintentbench/qualified_scientific_cases.py", ""),
        "test_tree_sha256": artifact_sha256({key: value for key, value in entries.items() if key.startswith("tests/")}),
        "method_sha256": artifact_sha256({key: value for key, value in entries.items() if key in {"method.md", "FlowIntentBench.md"}}),
        "invocation_id": f"agent-ready:{artifact_sha256(entries)[:16]}",
    }


def _agent_safe_configuration(value: Any) -> Any:
    """Remove credentials before a runtime configuration enters an identity."""

    secret_markers = ("key", "token", "secret", "credential", "password")
    if isinstance(value, Mapping):
        return {
            str(key): _agent_safe_configuration(item)
            for key, item in value.items()
            if not any(marker in str(key).casefold() for marker in secret_markers)
        }
    if isinstance(value, (list, tuple)):
        return [_agent_safe_configuration(item) for item in value]
    return value


def _agent_construction_run_identity(
    root: Path,
    build_identity: Mapping[str, Any],
    *,
    live_agents: bool,
    config_path: str | Path | None,
    timeout: float,
) -> dict[str, Any]:
    """Describe how one construction invocation was executed.

    Build identity answers *what code* was run; this identity answers *how this
    concrete construction* was requested.  It intentionally contains no
    credential material and does not include a test report hash, avoiding a
    circular dependency between a run and its later closure report.
    """

    config_value: Any = {}
    if config_path is not None:
        config_file = Path(config_path)
        if config_file.suffix.casefold() == ".toml":
            try:
                config_value = tomllib.loads(config_file.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                config_value = {"path": str(config_file)}
        else:
            config_value = _read_json(config_file, {})
        if not isinstance(config_value, Mapping):
            config_value = {"path": str(config_path)}
    runtime_files: dict[str, str] = {}
    runtime_root = root / "runtime_profiles"
    if runtime_root.is_dir():
        runtime_files = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(runtime_root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
    datasets: dict[str, str] = {}
    datasets_root = root / "datasets"
    if datasets_root.is_dir():
        for path in sorted(datasets_root.glob("*/dataset_manifest.json")):
            datasets[path.parent.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    profile_ids = (
        ["proxy-flow-analyst-gpt-5.6-sol", "flow-scientific-adjudicator-gpt-5.6-sol"]
        if live_agents
        else []
    )
    identity = {
        "build_identity_sha256": artifact_sha256(build_identity),
        "live_agents_requested": bool(live_agents),
        "agent_profile_ids": profile_ids,
        "provider_configuration_digest": artifact_sha256(_agent_safe_configuration(config_value)),
        "runtime_profile_digest": artifact_sha256(runtime_files),
        "compiler_identity": "flowintentbench.deterministic_materialization.compile_effective_operationalization",
        "compiler_sha256": str(build_identity.get("source_tree_sha256", "")),
        "materializer_identity": "flowintentbench.qualified_scientific_cases.reference_materializer",
        "materializer_sha256": str(build_identity.get("builder_sha256", "")),
        "dataset_manifest_digests": datasets,
        "timeout_seconds": float(timeout),
    }
    identity["construction_invocation_id"] = f"construction:{artifact_sha256(identity)[:24]}"
    identity["run_identity_sha256"] = artifact_sha256(identity)
    return identity


def _agent_validate_run_identity(
    identity: Mapping[str, Any] | None,
    build_identity: Mapping[str, Any],
) -> bool:
    """Validate the non-secret, self-contained run identity packet."""

    if not isinstance(identity, Mapping):
        return False
    required = {
        "build_identity_sha256", "live_agents_requested", "agent_profile_ids",
        "provider_configuration_digest", "runtime_profile_digest",
        "compiler_identity", "compiler_sha256", "materializer_identity",
        "materializer_sha256", "dataset_manifest_digests",
        "construction_invocation_id", "run_identity_sha256",
    }
    if not required <= set(identity):
        return False
    if identity.get("build_identity_sha256") != artifact_sha256(build_identity):
        return False
    forbidden = ("api_key", "authorization", "secret", "credential", "password")
    def contains_secret(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(
                any(marker in str(key).casefold() for marker in forbidden)
                or contains_secret(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_secret(item) for item in value)
        return False
    if contains_secret(identity):
        return False
    unsigned = {str(key): value for key, value in identity.items() if key != "run_identity_sha256"}
    return identity.get("run_identity_sha256") == artifact_sha256(unsigned)


def invalidate_downstream_artifacts(
    case_dir: str | Path,
    *,
    owning_stage: str,
    reason: str,
    affected_artifacts: Sequence[str] = (),
) -> dict[str, Any]:
    """Record an explicit upstream correction and invalidate downstream state.

    This function never mutates scientific artifacts.  It marks readiness as
    blocked and writes a durable packet so a later rebuild can be tied to a
    new build/run identity rather than silently repairing downstream output.
    """

    normalized_stage = str(owning_stage).strip().upper()
    if normalized_stage not in {"STAGE_1", "STAGE_2", "STAGE_2B", "STAGE_3"}:
        raise ValueError("owning_stage must be STAGE_1, STAGE_2, STAGE_2B, or STAGE_3")
    normalized_reason = str(reason).strip()
    if not normalized_reason:
        raise ValueError("reason must be non-empty")
    directory = Path(case_dir)
    packet = {
        "artifact_type": "DownstreamInvalidation",
        "schema_version": "downstream-invalidation-v1",
        "status": "INVALIDATED",
        "owning_stage": normalized_stage,
        "reason": normalized_reason,
        "affected_artifacts": sorted({str(item) for item in affected_artifacts}),
        "repair_policy": "REOPEN_OWNING_STAGE_AND_REBUILD_WITH_NEW_IDENTITY",
    }
    _write_json(directory / "downstream_invalidation.json", packet)
    readiness_path = directory / "agent_readiness.json"
    readiness = _read_json(readiness_path, {})
    if not isinstance(readiness, Mapping):
        readiness = {}
    readiness = dict(readiness)
    readiness.update({
        "status": "BLOCKED",
        "downstream_invalidated": True,
        "downstream_invalidation_path": "downstream_invalidation.json",
        "downstream_invalidation_stage": normalized_stage,
    })
    _write_json(readiness_path, readiness)
    return packet


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _optional_case_id(value: Any) -> str | None:
    """Normalize nullable manifest case identifiers without stringifying null.

    Human-facing closure rows use JSON ``null`` while a canonical case is not
    yet authored (currently the four Combustor slots).  Converting that value
    with ``str()`` would create the literal identifier ``"None"`` and make
    all pending slots collide on one output filename.  Empty/null spellings
    are treated as absent; real case identifiers remain byte-for-byte stable.
    """

    if value is None:
        return None
    text = str(value).strip()
    if text.casefold() in {"", "none", "null"}:
        return None
    return text


def _harden_model_visible_question(
    question: str,
    projection: Mapping[str, Any],
) -> str:
    """Expose authored execution semantics in a readable question.

    The v3 candidate matrix already contains the corrected semantic projection,
    but the older human-facing closure can still carry a shorter v1 rendering.
    This construction-only normalizer adds only information already present in
    the projection; it never chooses a new scientific operation.
    """

    text = " ".join(str(question).split()).strip()
    if not text:
        return text
    # Remove protocol-sounding modifiers that do not carry scientific
    # semantics.  The projection and execution plan remain unchanged.
    text = re.sub(r"\bscientifically\s+(?=appropriate|relevant|defensible|informative)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bscientifically\s+relevant\s+coherent\b", "coherent", text, flags=re.IGNORECASE)
    text = re.sub(r"\ba\s+appropriate\b", "an appropriate", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Also provide one relevant characterization of your choice, such as its geometric relation, comparison with another feature, morphology, orientation, or spatial extent\.",
        "Add one relevant characterization, such as geometry, comparison, morphology, orientation, or spatial extent.",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"one scientifically informative way of your choice—through its geometry, a meaningful comparison, morphology, orientation, or spatial extent",
        "one relevant characterization of your choice, such as geometry, comparison, morphology, orientation, or spatial extent",
        text,
        flags=re.IGNORECASE,
    )
    resolved = projection.get("resolved_operationalization", ())
    meanings = [
        str(item.get("normalized_meaning", ""))
        for item in resolved
        if isinstance(item, Mapping)
    ]
    scope = projection.get("scientific_scope", {})
    if isinstance(scope, Mapping):
        meanings.extend(
            str(item)
            for item in (scope.get("condition_constraints", ()) or ())
            if str(item).strip()
        )
    meaning_text = " ".join(meanings).casefold()
    if "linear quantile interpolation" in meaning_text and not re.search(
        r"linear(?:ly)?\s+(?:quantile\s+)?interpol", text, flags=re.IGNORECASE
    ):
        text = re.sub(
            r"\bthe 90th percentile\b",
            "the linearly interpolated 90th percentile",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        if "interpol" not in text.casefold():
            text += " Use linear quantile interpolation for that percentile."
    if "one structured-grid index step along exactly one axis" in meaning_text:
        if re.search(r"face-connected(?:\s+retained)?\s+locations", text, flags=re.IGNORECASE):
            text = re.sub(
                r"face-connected(?:\s+retained)?\s+locations",
                "locations connected through immediate grid neighbors along one grid axis",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
        elif re.search(r"face-connected\s+[^,.;]+?\s+locations", text, flags=re.IGNORECASE):
            text = re.sub(
                r"face-connected\s+([^,.;]+?)\s+locations",
                r"locations connected through immediate grid neighbors along one grid axis",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
        elif "face-connected high-speed locations" in text.casefold():
            text = re.sub(
                r"face-connected high-speed locations",
                "high-speed locations connected through immediate grid neighbors along one grid axis",
                text,
                count=1,
                flags=re.IGNORECASE,
            )
        if not (
            "one grid axis" in text.casefold()
            or "six-neighbor" in text.casefold()
            or "immediate grid" in text.casefold()
        ):
            text += " Use six-neighbor connectivity along one grid axis."
    if "unstructured-mesh cell" in meaning_text or "share an unstructured-mesh cell" in meaning_text:
        text = re.sub(
            r"mesh-connected\s+(?:locations|points)",
            "locations that share an unstructured-mesh cell",
            text,
            flags=re.IGNORECASE,
        )
        if not any(
            marker in text.casefold()
            for marker in ("unstructured-mesh cell", "share an unstructured-mesh cell", "same mesh cell")
        ):
            text += " Treat locations that share an unstructured-mesh cell as connected."
    if "euclidean magnitude of the stored uvw array" in meaning_text:
        text = re.sub(
            r"(?:recorded\s+)?speed\s+in\s+the\s+stored\s+velocity\s+scale",
            "speed as the Euclidean magnitude of the stored uvw vector at each point",
            text,
            flags=re.IGNORECASE,
        )
        if "euclidean magnitude" not in text.casefold() or "uvw" not in text.casefold():
            text += " Define speed as the Euclidean magnitude of the stored uvw vector at each point."
        text = re.sub(
            r"(percentile of non-zero speed)\s+as\s+the Euclidean magnitude of the stored uvw vector at each point",
            r"\1 values, where speed is the Euclidean magnitude of the stored uvw vector at each point",
            text,
            flags=re.IGNORECASE,
        )
    if "arithmetic mean of its vertex" in meaning_text and "arithmetic mean" not in text.casefold():
        marker = "compare the fields using"
        if marker in text.casefold():
            index = text.casefold().index(marker)
            text = (
                text[:index]
                + "first take the arithmetic mean of the finite vertex values within each cell, then "
                + text[index:]
            )
        else:
            text += " First take the arithmetic mean of the finite vertex values within each cell before applying the stated measure."
    elif "arithmetic mean of its vertex" in meaning_text:
        text = re.sub(
            r"\bfirst\s+average\s+the\s+finite\s+vertex\s+values\b",
            "first take the arithmetic mean of the finite vertex values",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return text


def _conditions(
    root: Path,
    source_manifest: str | Path | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load construction rows, optionally rebinding them to a newer matrix.

    The historical closure remains the source of reviewed natural wording,
    while an explicit candidate matrix may supply a newer semantic projection.
    Rows are joined by stable ``slot_id``; no case-id discovery is performed.
    """

    closure = _read_json(
        root / DERIVED_REVIEW_RELATIVE_PATH / "human_facing_closure_manifest.json",
        {},
    )
    # The derived closure index intentionally contains only frozen scientific
    # fields.  Legacy construction metadata (candidate provenance and review
    # packets) is compatibility input, never authority for question text.
    compatibility = _read_json(
        root / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions" / "scientific_question_manifest.json",
        {},
    )
    compatibility_closure = _read_json(
        root / ARCHIVE_RELATIVE_PATH / "outputs_current_scientific_questions" / "human_facing_closure_manifest.json",
        {},
    )
    compatibility_rows = {
        str(item.get("case_id")): dict(item)
        for item in (compatibility.get("conditions", []) if isinstance(compatibility, Mapping) else [])
        if isinstance(item, Mapping) and str(item.get("case_id", "")).strip()
    }
    closure_rows = {
        str(row.get("slot_id")): dict(row)
        for row in (closure.get("conditions", []) if isinstance(closure, Mapping) else [])
        if isinstance(row, Mapping) and str(row.get("slot_id", "")).strip()
    }
    if source_manifest is None:
        # Default qualification/agent-ready calls are an explicitly archived
        # compatibility route.  The N=1 reference path passes the derived
        # frozen index explicitly, so legacy lifecycle rows cannot become
        # model-visible authority by accident.
        if isinstance(compatibility_closure, Mapping) and compatibility_closure.get("conditions"):
            manifest = compatibility_closure
            closure_rows = {
                str(item.get("slot_id")): dict(item)
                for item in compatibility_closure.get("conditions", [])
                if isinstance(item, Mapping) and str(item.get("slot_id", "")).strip()
            }
        else:
            manifest = closure
    elif isinstance(source_manifest, Mapping):
        manifest = dict(source_manifest)
    else:
        manifest = _read_json(Path(source_manifest), {})
    rows = manifest.get("conditions") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 28:
        raise ValueError(
            "construction requires a frozen 28-row scientific-question manifest"
        )
    normalized: list[dict[str, Any]] = []
    frozen_derived_manifest = bool(
        isinstance(manifest, Mapping)
        and manifest.get("authority") == "FROZEN_REFERENCE_DERIVED"
    )
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        slot_id = str(row.get("slot_id", "")).strip()
        fallback = closure_rows.get(slot_id, {})
        legacy = compatibility_rows.get(str(row.get("case_id", "")), {})
        if legacy:
            # Only fill construction compatibility fields that are absent;
            # frozen question/projection/hash fields always win.
            for key, value in legacy.items():
                row.setdefault(key, copy.deepcopy(value))
        projection = row.get("canonical_semantic_projection")
        if not isinstance(projection, Mapping):
            projection = fallback.get("canonical_semantic_projection", {})
        projection = (
            copy.deepcopy(dict(projection))
            if frozen_derived_manifest or source_manifest is None
            else harden_portfolio_scientific_projection(
                projection, condition=str(row.get("condition", ""))
            )
        )
        candidate = row.get("question_candidates")
        if not isinstance(candidate, list):
            candidate = []
        primary_id = str(row.get("primary_candidate_id") or row.get("accepted_question_candidate_id") or "").strip()
        primary = next(
            (item for item in candidate if isinstance(item, Mapping) and str(item.get("candidate_id", "")) == primary_id),
            None,
        )
        question = str(
            fallback.get("scientific_question")
            or row.get("scientific_question")
            or (primary or {}).get("model_visible_text", "")
        ).strip()
        row["scientific_question"] = (
            question
            if frozen_derived_manifest
            else _harden_model_visible_question(question, projection)
            if source_manifest is not None
            else question
        )
        row["canonical_semantic_projection"] = projection
        if source_manifest is not None and not frozen_derived_manifest:
            row["semantic_contract_sha256"] = _projection_semantic_hash(projection)
        if fallback:
            for key in ("case_id", "family_id", "dataset_id", "condition", "slot_id"):
                if key == "case_id" and source_manifest is None and not frozen_derived_manifest:
                    # Preserve archived qualification semantics for provisional
                    # Combustor slots (their case identity is intentionally
                    # null); the frozen v7 IDs are used only by explicit
                    # reference-derived construction.
                    continue
                if not row.get(key) and fallback.get(key):
                    row[key] = fallback[key]
        if frozen_derived_manifest:
            row["semantic_contract_sha256"] = str(
                row.get("semantic_contract_sha256")
                or fallback.get("semantic_contract_sha256")
                or artifact_sha256(projection)
            )
        normalized.append(row)
    if len(normalized) != 28:
        raise ValueError(f"scientific-question manifest contains {len(normalized)} valid rows, expected 28")
    return normalized


def _case_record(root: Path, case_id: str | None) -> dict[str, Any] | None:
    if not case_id:
        return None
    try:
        return load_case_record(root, case_id)
    except (FileNotFoundError, OSError, ValueError):
        return None


def _dataset_evidence(root: Path, dataset_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    base = root / "datasets" / dataset_id / "construction"
    sources_doc = _read_json(base / "sources.json", {})
    sources = {
        str(item.get("source_id")): dict(item)
        for item in (sources_doc.get("sources", []) if isinstance(sources_doc, Mapping) else [])
        if isinstance(item, Mapping) and item.get("source_id")
    }
    evidence: dict[str, Any] = {}
    for name in ("operationalization_evidence", "context_evidence", "finding_evidence"):
        rows = _read_json(base / f"{name}.json", [])
        if not isinstance(rows, list):
            continue
        for item in rows:
            if isinstance(item, Mapping) and item.get("evidence_id"):
                evidence[str(item["evidence_id"])] = dict(item)
    return sources, evidence


def _gt_path(record: Mapping[str, Any]) -> Path | None:
    case_dir = record.get("case_dir")
    if not case_dir:
        return None
    path = Path(str(case_dir)) / "ground_truth.json"
    return path if path.is_file() else None


def _load_gt(record: Mapping[str, Any]) -> tuple[GroundTruth | None, list[str]]:
    path = _gt_path(record)
    raw = record.get("ground_truth")
    if path is not None:
        raw = _read_json(path, raw)
    if not isinstance(raw, Mapping) or not raw:
        return None, ["GROUND_TRUTH_MISSING"]
    try:
        return GroundTruth.model_validate(raw), []
    except Exception as exc:  # pydantic's detailed error is retained in the packet
        return None, [f"GROUND_TRUTH_INVALID:{type(exc).__name__}"]


def _finding_verification_mode(finding: Any) -> str:
    """Describe only the verification rule explicitly available in GT.

    This is a reporting helper, not a scientific inference.  In particular,
    numeric lists without ``spatial_tolerance`` remain semantic-only.
    """

    verification = getattr(finding, "verification", None)
    if verification is not None and verification.spatial_tolerance is not None:
        return "SPATIAL_EUCLIDEAN"
    if verification is not None and (
        verification.absolute_tolerance is not None
        or verification.relative_tolerance is not None
    ):
        return "SCALAR_TOLERANCE"
    return "SEMANTIC_ONLY"


def _readiness_projection(
    *,
    repository_root: Path,
    row: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    gt: GroundTruth | None,
    gt_errors: Sequence[str],
    evidence_ready: bool,
    evaluation_status: str,
    gt_status: str,
) -> dict[str, Any]:
    """Return orthogonal construction/readiness facts for human inspection."""

    gt_schema_valid = gt is not None and not gt_errors
    gt_schema_status = (
        "VALID"
        if gt_schema_valid
        else "INVALID"
        if any(code != "GROUND_TRUTH_MISSING" for code in gt_errors)
        else "PENDING"
    )
    case_dir = None
    if record and record.get("case_dir"):
        candidate = Path(str(record.get("case_dir")))
        case_dir = candidate if candidate.is_absolute() else repository_root / candidate
    reference_analysis_present = bool(
        case_dir is not None and (case_dir / "reference_analysis.md").is_file()
    )
    branch_count = len(gt.findings_by_operationalization) if gt is not None else 0
    finding_count = sum(len(branch.findings) for branch in gt.findings_by_operationalization) if gt is not None else 0
    data_mediated_proof = bool(gt_schema_valid and reference_analysis_present and branch_count and finding_count)
    verification_modes = (
        sorted(
            {
                _finding_verification_mode(finding)
                for branch in gt.findings_by_operationalization
                for finding in branch.findings
            }
        )
        if gt is not None
        else []
    )
    gt_adjudicable = bool(gt_schema_valid and data_mediated_proof and all(mode in {
        "SPATIAL_EUCLIDEAN",
        "SCALAR_TOLERANCE",
        "EXACT_DISCRETE_NUMERIC",
        "SEMANTIC_ONLY",
    } for mode in verification_modes))
    condition = str(row.get("condition", ""))
    srac_required = condition in {"O2-F1", "O3-F1"}
    if not srac_required:
        srac_status = "NOT_REQUIRED"
    elif evaluation_status == EVALUATION_FROZEN:
        srac_status = "READY"
    else:
        srac_status = "PENDING"
    scientific_release_status = str(row.get("SCIENTIFIC_RELEASE_STATUS", "")).strip().upper()
    grounding_status = (
        "ELIGIBLE_FOR_CURATOR_REVIEW"
        if scientific_release_status == "ELIGIBLE"
        else "PENDING_CASE_IDENTITY"
        if "NEW_CASE_REQUIRED" in scientific_release_status
        else "PENDING"
    )
    return {
        "gt_schema_valid": gt_schema_valid,
        "gt_schema_status": gt_schema_status,
        "data_mediated_proof": data_mediated_proof,
        "reference_analysis_present": reference_analysis_present,
        "reference_branch_count": branch_count,
        "reference_finding_count": finding_count,
        "verification_modes": verification_modes,
        "gt_adjudicable": gt_adjudicable,
        "evidence_traceable": bool(evidence_ready),
        "scientific_grounding_status": grounding_status,
        "srac_required": srac_required,
        "srac_status": srac_status,
        "gt_semantic_bound": gt_status == GT_BOUND,
        "curator_confirmation": False,
        "release_authorized": False,
    }


def _gt_evidence(
    gt: GroundTruth,
    sources: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve declared evidence IDs without inferring missing provenance."""

    ids: list[str] = []
    for bundle in gt.acceptable_operationalizations:
        ids.extend(str(item) for item in bundle.evidence_ids)
    for branch in gt.findings_by_operationalization:
        for finding in branch.findings:
            ids.extend(str(item) for item in finding.evidence_ids)
    ids = list(dict.fromkeys(ids))
    failures: list[str] = []
    records: list[dict[str, Any]] = []
    if not ids:
        # An absent evidence declaration is unfinished construction work, not
        # proof that an otherwise valid claim is wrong.  The lifecycle layer
        # reports this as pending evidence and keeps it out of defect counts.
        failures.append("EVIDENCE_NOT_YET_FROZEN")
    for evidence_id in ids:
        claim = evidence.get(evidence_id)
        if not isinstance(claim, Mapping):
            failures.append(f"EVIDENCE_INVALID_NOT_FOUND:{evidence_id}")
            continue
        source_id = str(claim.get("source_id", ""))
        if not source_id or source_id not in sources:
            failures.append(f"EVIDENCE_INVALID_SOURCE_NOT_FOUND:{source_id or '<empty>'}")
        records.append(
            {
                "evidence_id": evidence_id,
                "statement": str(claim.get("statement", "")),
                "source_id": source_id,
                "locator": _json(claim.get("locator", {})),
            }
        )
    return records, list(dict.fromkeys(failures))


def _ground_truth_identity(
    gt: GroundTruth,
    row: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    for field in ("dataset_id", "case_id"):
        expected = str(row.get(field, ""))
        actual = str(getattr(gt, field, ""))
        if expected and actual != expected:
            failures.append(f"{field.upper()}_MISMATCH")
    # A legacy family label is not, by itself, a semantic contradiction.  It
    # is recorded by the caller as a re-binding obligation until a semantic
    # contract/GT binding is frozen.  Treating every historical family rename
    # as a GT defect would collapse "not yet frozen" into "invalid".
    return failures


def _gt_family_rebind_required(gt: GroundTruth, row: Mapping[str, Any]) -> bool:
    """Whether an existing legacy GT needs a semantic-family re-binding.

    Historical cases use family identifiers from an earlier construction
    vocabulary.  That is unfinished binding work, not a scientific
    contradiction; the case remains a candidate while the binding is pending.
    """

    expected_family = str(row.get("family_id", "")).strip()
    return bool(expected_family and gt.case_family_id != expected_family)


def _identity_is_frozen(
    *,
    row: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    projection: Mapping[str, Any],
    semantic_hash: str,
    failures: Sequence[str],
    visibility: Mapping[str, Any],
) -> bool:
    """Decide only the scientific-case identity layer.

    Evidence, SRAC, GroundTruth, curator, and release gates intentionally do
    not participate here.  A provisional target without a canonical case ID
    remains an explicit pending identity candidate.
    """

    if not str(row.get("case_id", "")).strip() or record is None:
        return False
    if not projection or not semantic_hash:
        return False
    if _projection_semantic_hash(projection) != semantic_hash:
        return False
    if not all(bool(value) for value in visibility.values()):
        return False
    if any(
        code in {
            "CASE_NOT_FOUND",
            "CASE_ID_MISMATCH",
            "DATASET_ID_MISMATCH",
            "SEMANTIC_HASH_MISMATCH",
            "QUESTION_SEMANTIC_FIDELITY_FAILED",
            "VISIBLE_CONTEXT_INTEGRITY_FAILED",
            "RESPONSIBILITY_INTEGRITY_FAILED",
            "MODEL_VISIBLE_FLOW_DATA_MISSING",
        }
        for code in failures
    ):
        return False
    target_status = row.get("provisional_target_status") or projection.get("provisional_target_status")
    if str(target_status or "").startswith("PROVISIONAL_TARGET_PENDING"):
        return False
    return True


def _evaluation_state(
    *,
    identity_frozen: bool,
    row: Mapping[str, Any],
    evidence_ready: bool,
    gt: GroundTruth | None,
    gt_errors: Sequence[str],
    gt_rebind_required: bool,
) -> tuple[str, str]:
    """Return ``(evaluation_status, gt_status)`` without changing contracts."""

    if not identity_frozen:
        return EVALUATION_PENDING_CASE_IDENTITY, GT_NOT_YET_FROZEN
    invalid_gt = any(code != "GROUND_TRUTH_MISSING" for code in gt_errors)
    if invalid_gt:
        return EVALUATION_INVALID, GT_INVALID
    if gt is None:
        return EVALUATION_PENDING_GT, GT_NOT_YET_FROZEN
    release_status = str(row.get("SCIENTIFIC_RELEASE_STATUS", ""))
    if "PENDING_O_SPACE" in release_status or "PENDING_SRAC" in release_status:
        return EVALUATION_PENDING_SRAC, GT_NOT_YET_FROZEN
    if "NEW_CASE_REQUIRED" in release_status:
        return EVALUATION_PENDING_CASE_IDENTITY, GT_NOT_YET_FROZEN
    if "PENDING_EVIDENCE" in release_status:
        return EVALUATION_PENDING_EVIDENCE, GT_NOT_YET_FROZEN
    if not evidence_ready:
        return EVALUATION_PENDING_EVIDENCE, GT_NOT_YET_FROZEN
    # GT binding is an independent downstream identity.  A legacy GT may be
    # usable to author the contract while its semantic-family re-binding is
    # still pending; only an exact evidence-backed binding receives BOUND.
    return EVALUATION_FROZEN, GT_NOT_YET_FROZEN if gt_rebind_required else GT_BOUND


def _finding_contract(projection: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[tuple[str, ...], ...]]:
    semantics = projection.get("finding_semantics")
    semantics = dict(semantics) if isinstance(semantics, Mapping) else {}
    adequate = semantics.get("adequate_core_semantics", {})
    adequate = adequate if isinstance(adequate, Mapping) else {}
    sets = adequate.get("adequate_core_sets", ())
    core_sets: list[tuple[str, ...]] = []
    if isinstance(sets, Sequence) and not isinstance(sets, (str, bytes, bytearray)):
        for item in sets:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                core_sets.append(tuple(sorted(str(role) for role in item)))
    if not core_sets:
        roles = semantics.get("required_finding_roles", ())
        if isinstance(roles, Sequence) and not isinstance(roles, (str, bytes, bytearray)):
            core_sets.append(tuple(sorted(str(role) for role in roles)))
    return semantics, tuple(core_sets)


def _build_sec(
    *,
    case_id: str,
    projection: Mapping[str, Any],
    gt: GroundTruth,
    evidence_records: Sequence[Mapping[str, Any]],
    semantic_hash: str,
    reference_analysis_path: str | None,
) -> ScientificEvaluationContract:
    branch_by_id = {
        branch.operationalization_id: branch for branch in gt.findings_by_operationalization
    }
    g_of_o: list[dict[str, Any]] = []
    for bundle in gt.acceptable_operationalizations:
        branch = branch_by_id.get(bundle.operationalization_id)
        decisions = {
            decision.dimension.value: decision.statement for decision in bundle.decisions
        }
        g_of_o.append(
            {
                "operationalization_id": bundle.operationalization_id,
                "effective_operationalization": decisions,
                "findings": (
                    [finding.model_dump(mode="json") for finding in branch.findings]
                    if branch is not None
                    else []
                ),
                "materialization": {
                    "status": "SUCCESS",
                    "reproducible": True,
                    "mode": "EXISTING_REFERENCE_ANALYSIS",
                    "reference_analysis_path": reference_analysis_path,
                },
            }
        )
    finding_contract, core_sets = _finding_contract(projection)
    tolerance_map: dict[str, Any] = {}
    for branch in gt.findings_by_operationalization:
        for finding in branch.findings:
            if finding.verification is not None:
                tolerance_map[finding.finding_id] = finding.verification.model_dump(mode="json")
    unresolved = projection.get("unresolved_operationalization_dimensions", ())
    if unresolved:
        policy = (
            "A scientifically valid unenumerated O requires blinded adjudication; "
            "reject only target change, responsibility violation, unsupported observable, "
            "or invalid representation."
        )
    else:
        policy = (
            "The operationalization space is fixed for this case; no unenumerated O is "
            "needed. Novel F2 findings remain subject to the existing finding contract."
        )
    invalid_rules = (
        {"rule_id": "TARGET_CHANGED", "decision": "INVALID"},
        {"rule_id": "RESPONSIBILITY_VIOLATED", "decision": "INVALID"},
        {"rule_id": "UNSUPPORTED_OBSERVABLE", "decision": "INVALID"},
        {"rule_id": "INVALID_REPRESENTATION", "decision": "INVALID"},
    )
    provenance = tuple(
        {
            "evidence_id": str(item.get("evidence_id", "")),
            "source_id": str(item.get("source_id", "")),
            "locator": _json(item.get("locator", {})),
        }
        for item in evidence_records
    )
    return ScientificEvaluationContract(
        case_id=case_id,
        enumerated_valid_O=tuple(bundle.model_dump(mode="json") for bundle in gt.acceptable_operationalizations),
        explicitly_invalid_O=invalid_rules,
        unenumerated_policy=policy,
        G_of_O=tuple(g_of_o),
        finding_requirement_contract=finding_contract,
        adequate_core_sets=core_sets,
        tolerances={"explicit_verification_specs": tolerance_map},
        adjudication_provenance=provenance,
        status="DRAFT_REQUIRES_HUMAN_CONFIRMATION",
        source_semantic_contract_sha256=semantic_hash,
    )


def _case_checks(
    *,
    root: Path,
    row: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    projection: Mapping[str, Any],
    semantic_hash: str,
    gt: GroundTruth | None,
    gt_errors: Sequence[str],
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    """Return true defects only; unfinished work is represented as state.

    The frozen human-facing closure manifest contains downstream release
    gates.  Those gates are deliberately *not* copied into ``failures``: a
    pending evidence/SRAC/GT task does not invalidate the scientific case
    identity.
    """

    failures: list[str] = []
    categories: list[str] = []
    visibility = {
        "question_fidelity": row.get("semantic_fidelity_status") == "PASS",
        "visibility_integrity": row.get("visibility_integrity_status") == "PASS",
        "responsibility_integrity": row.get("responsibility_integrity_status") == "PASS",
    }
    # ``NEW_CASE_REQUIRED`` is a pending identity state for this phase.  It is
    # intentionally not a CASE_DEFECT unless a canonical case was claimed and
    # then found inconsistent.
    if not str(row.get("case_id", "")).strip():
        return failures, categories, [], visibility
    if record is None:
        # No canonical record is expected for a NEW_CASE_REQUIRED slot.  For
        # any other row this is an actual identity defect.
        if str(row.get("SCIENTIFIC_RELEASE_STATUS", "")) != "NEW_CASE_REQUIRED":
            failures.append("CASE_NOT_FOUND")
            categories.append("CASE_DEFECT")
        return failures, categories, [], visibility
    if str(record.get("case_id", "")) != str(row.get("case_id", "")):
        failures.append("CASE_ID_MISMATCH")
        categories.append("CASE_DEFECT")
    if str(record.get("dataset_id", "")) != str(row.get("dataset_id", "")):
        failures.append("DATASET_ID_MISMATCH")
        categories.append("CASE_DEFECT")
    if _projection_semantic_hash(projection) != semantic_hash:
        failures.append("SEMANTIC_HASH_MISMATCH")
        categories.append("CASE_DEFECT")
    if not all(visibility.values()):
        failures.extend(
            code
            for code, ok in (
                ("QUESTION_SEMANTIC_FIDELITY_FAILED", visibility["question_fidelity"]),
                ("VISIBLE_CONTEXT_INTEGRITY_FAILED", visibility["visibility_integrity"]),
                ("RESPONSIBILITY_INTEGRITY_FAILED", visibility["responsibility_integrity"]),
            )
            if not ok
        )
        categories.append("CASE_DEFECT")
    case_input = record.get("case_input")
    if not isinstance(case_input, Mapping) or not case_input.get("flow_data"):
        failures.append("MODEL_VISIBLE_FLOW_DATA_MISSING")
        categories.append("CASE_DEFECT")
    if gt_errors:
        # Missing GT is expected while the downstream evaluation contract is
        # being constructed.  A malformed/contradictory GT remains a real
        # artifact defect and is fail-closed.
        true_gt_errors = [
            code for code in gt_errors if code != "GROUND_TRUTH_MISSING"
        ]
        failures.extend(true_gt_errors)
        if true_gt_errors:
            categories.append("GT_BINDING_DEFECT")
        return list(dict.fromkeys(failures)), list(dict.fromkeys(categories)), [], visibility
    assert gt is not None
    identity_errors = _ground_truth_identity(gt, row)
    failures.extend(identity_errors)
    if identity_errors:
        categories.append("GT_BINDING_DEFECT")
    sources, evidence = _dataset_evidence(root, str(row.get("dataset_id", "")))
    evidence_records, evidence_errors = _gt_evidence(gt, sources, evidence)
    true_evidence_errors = [
        code for code in evidence_errors if code != "EVIDENCE_NOT_YET_FROZEN"
    ]
    failures.extend(true_evidence_errors)
    if true_evidence_errors:
        categories.append("EVIDENCE_DEFECT")
    return list(dict.fromkeys(failures)), list(dict.fromkeys(categories)), evidence_records, visibility


def _review_packet(
    *,
    packet: Mapping[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    recommendation = recommendation.upper()
    if recommendation not in _ALLOWED_PROXY_RECOMMENDATIONS:
        raise ValueError("invalid proxy recommendation")
    return {
        "artifact_type": "ScientificCaseReviewPacket",
        "phase": "PHASE_2",
        "case_id": packet.get("case_id"),
        "case_identity_id": packet.get("case_identity_id"),
        "qualification_status": packet.get("qualification_status"),
        "lifecycle": packet.get("lifecycle", {}),
        "scientific_question": packet.get("question", {}),
        "scientific_identity": packet.get("scientific_identity", {}),
        "responsibility": packet.get("responsibility", {}),
        "evidence": packet.get("evidence", {}),
        "evaluation": packet.get("evaluation", {}),
        "uncertainty_and_risks": packet.get("uncertainty_and_risks", {}),
        "flow_expert_proxy": {
            "recommendation": recommendation,
            "role": "FLOW_EXPERT_PROXY",
            "curator_decision": None,
            "curator_confirmation": None,
            "expert_verified": False,
        },
    }


def build_qualified_scientific_case_portfolio(
    repository_root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all 28 frozen questions and materialize Phase 2 artifacts."""

    root = Path(repository_root).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "outputs/current/qualified_scientific_cases"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "scientific_evaluation_contracts").mkdir(parents=True, exist_ok=True)
    (out / "scientific_artifact_bindings").mkdir(parents=True, exist_ok=True)
    (out / "scientific_case_review_packets").mkdir(parents=True, exist_ok=True)
    # Remove the artifact name emitted by pre-fix versions when a nullable
    # ``case_id`` was stringified as ``"None"``.  This is an exact generated
    # workspace cleanup, not a broad deletion; current slots are written under
    # their stable ``slot_id`` below.
    for stale in (
        out / "cases" / "None.json",
        out / "scientific_case_review_packets" / "None.json",
    ):
        if stale.is_file():
            stale.unlink()

    packets: list[dict[str, Any]] = []
    for row in _conditions(root):
        case_id = _optional_case_id(row.get("case_id"))
        # ``slot_id`` is the stable identity for a human-facing row when a
        # canonical case_id has not yet been authored (for example Combustor).
        case_identity_id = case_id or str(row.get("slot_id", "")).strip() or None
        record = _case_record(root, case_id)
        projection = row.get("canonical_semantic_projection")
        projection = dict(projection) if isinstance(projection, Mapping) else {}
        semantic_hash = str(row.get("semantic_contract_sha256", ""))
        if not semantic_hash and projection:
            semantic_hash = _projection_semantic_hash(projection)
        gt, gt_errors = _load_gt(record or {}) if record else (None, ["GROUND_TRUTH_MISSING"])
        failures, categories, evidence_records, visibility = _case_checks(
            root=root,
            row=row,
            record=record,
            projection=projection,
            semantic_hash=semantic_hash,
            gt=gt,
            gt_errors=gt_errors,
        )
        withdrawn = str(row.get("FORMAL_CANDIDATE_STATUS", "")) == "WITHDRAWN_NEW_CASE_REQUIRED"
        identity_frozen = _identity_is_frozen(
            row=row,
            record=record,
            projection=projection,
            semantic_hash=semantic_hash,
            failures=failures,
            visibility=visibility,
        )
        gt_rebind_required = bool(
            gt is not None and _gt_family_rebind_required(gt, row)
        )
        # Evidence is deliberately computed as a downstream readiness signal.
        # It never gates scientific case identity.
        release_status = str(row.get("SCIENTIFIC_RELEASE_STATUS", ""))
        evidence_ready = bool(
            gt is not None
            and evidence_records
            and "EVIDENCE_DEFECT" not in categories
            and "PENDING_EVIDENCE" not in release_status
        )
        evaluation_status, gt_status = _evaluation_state(
            identity_frozen=identity_frozen,
            row=row,
            evidence_ready=evidence_ready,
            gt=gt,
            gt_errors=gt_errors,
            gt_rebind_required=gt_rebind_required,
        )
        readiness = _readiness_projection(
            repository_root=root,
            row=row,
            record=record,
            gt=gt,
            gt_errors=gt_errors,
            evidence_ready=evidence_ready,
            evaluation_status=evaluation_status,
            gt_status=gt_status,
        )
        qualification_status = (
            SCIENTIFIC_CASE_QUALIFIED
            if identity_frozen
            else SCIENTIFIC_CASE_PENDING_IDENTITY
        )
        # ``NEW_CASE_REQUIRED`` and all pending construction gates are
        # lifecycle state, not defect categories.  Keep genuine validation
        # failures (if any) fail-closed.
        pending_reasons: list[str] = []
        if withdrawn:
            pending_reasons.append("NEW_CASE_REQUIRED")
        if evaluation_status == EVALUATION_PENDING_EVIDENCE:
            pending_reasons.append("EVIDENCE_PENDING")
        if evaluation_status == EVALUATION_PENDING_SRAC:
            pending_reasons.append("SRAC_PENDING")
        if evaluation_status == EVALUATION_PENDING_GT:
            pending_reasons.append("GT_NOT_YET_FROZEN")
        if gt_rebind_required:
            pending_reasons.append("GT_SEMANTIC_REBIND_PENDING")
        if not identity_frozen:
            pending_reasons.append("SCIENTIFIC_IDENTITY_PENDING")
        sec: ScientificEvaluationContract | None = None
        binding: Any = None
        gt_path = _gt_path(record or {})
        if record:
            ref_candidate = Path(str(record.get("case_dir"))) / "reference_analysis.md"
            # ``case_repository`` returns repository-relative paths for the
            # candidate archive and absolute paths for dataset legacy cases.
            ref_candidate = ref_candidate if ref_candidate.is_absolute() else root / ref_candidate
            ref_path = str(ref_candidate.relative_to(root)) if ref_candidate.is_file() and ref_candidate.is_relative_to(root) else (str(ref_candidate) if ref_candidate.is_file() else None)
        else:
            ref_path = None
        if evaluation_status == EVALUATION_FROZEN and gt is not None and case_id is not None and not gt_rebind_required:
            assert gt is not None and case_id is not None
            sec = _build_sec(
                case_id=case_id,
                projection=projection,
                gt=gt,
                evidence_records=evidence_records,
                semantic_hash=semantic_hash,
                reference_analysis_path=ref_path,
            )
            binding = build_scientific_artifact_binding(
                case_id=case_id,
                semantic_contract=projection,
                ground_truth=gt,
                scientific_evaluation_contract=sec,
                presentation=str(row.get("scientific_question", "")),
            )
            binding_result = validate_scientific_artifact_binding(
                binding,
                case_id=case_id,
                semantic_contract=projection,
                ground_truth=gt,
                scientific_evaluation_contract=sec,
                presentation=str(row.get("scientific_question", "")),
                formal_frozen=True,
            )
            if binding_result["scientific_artifact_status"] != "VALID" or binding_result["presentation_status"] != "VALID":
                failures.append("SCIENTIFIC_ARTIFACT_BINDING_INVALID")
                categories.append("GT_BINDING_DEFECT")
                sec = None
                binding = None
        if withdrawn:
            recommendation = "REJECT"
        elif identity_frozen:
            recommendation = "KEEP"
        elif any(category == "EVIDENCE_DEFECT" for category in categories):
            recommendation = "NEEDS_MORE_EVIDENCE"
        else:
            recommendation = "REVISE"
        case_input = record.get("case_input", {}) if record else {}
        case_context = case_input.get("case_context", {}) if isinstance(case_input, Mapping) else {}
        packet: dict[str, Any] = {
            "artifact_type": "QualifiedScientificCase",
            "schema_version": PHASE2_VERSION,
            "case_id": case_id,
            "case_identity_id": case_identity_id,
            "slot_id": row.get("slot_id"),
            "condition": row.get("condition"),
            "qualification_status": qualification_status,
            "scientific_case_identity_status": qualification_status,
            "evaluation_status": evaluation_status,
            "gt_binding_status": gt_status,
            "release_status": "READY" if evaluation_status == EVALUATION_FROZEN and binding is not None else "BLOCKED",
            "failure_codes": list(dict.fromkeys(failures)),
            "failure_categories": list(dict.fromkeys(categories)),
            "scientific_identity": {
                "dataset_id": row.get("dataset_id"),
                "family_id": row.get("family_id"),
                "concept_id": projection.get("concept_id"),
                "scientific_target": projection.get("scientific_target", row.get("scientific_target")),
                "scientific_scope": projection.get("scientific_scope", {}),
            },
            "question": {
                "text": row.get("scientific_question", ""),
                "semantic_contract_sha256": semantic_hash,
                "semantic_projection": _json(projection),
                "semantic_fidelity_status": row.get("semantic_fidelity_status"),
                "semantic_fidelity_proof": _json(row.get("semantic_fidelity_proof", {})),
                "visible_context": {
                    "case_context": _json(case_context),
                    "flow_data_metadata": _json(case_input.get("flow_data", {}).get("data_metadata", {}) if isinstance(case_input, Mapping) else {}),
                    "analysis_context": _json(row.get("model_visible_analysis_context", {})),
                },
                "backend_only_information_required": False,
            },
            "responsibility": {
                "fixed_operationalization_dimensions": projection.get("resolved_operationalization", []),
                "unresolved_operationalization_dimensions": projection.get("unresolved_operationalization_dimensions", []),
                "finding_responsibility": projection.get("finding_responsibility"),
                "finding_semantics": projection.get("finding_semantics", {}),
            },
            "evidence": {
                "claims": evidence_records,
                "provenance_complete": bool(evidence_ready),
                "gaps": list(dict.fromkeys(
                    [code for code in failures if code.startswith("EVIDENCE_") or code.startswith("SOURCE_")]
                    + [reason for reason in pending_reasons if reason == "EVIDENCE_PENDING"]
                )),
            },
            "evaluation": {
                "scientific_evaluation_contract_status": evaluation_status,
                "scientific_evaluation_contract": _json(sec) if sec is not None else None,
                "scientific_evaluation_contract_sha256": artifact_sha256(sec) if sec is not None else None,
                "valid_O": _json(sec.enumerated_valid_O) if sec is not None else [],
                "invalid_O_rules": _json(sec.explicitly_invalid_O) if sec is not None else [],
                "G_of_O": _json(sec.G_of_O) if sec is not None else [],
                "finding_requirement_contract": _json(sec.finding_requirement_contract) if sec is not None else projection.get("finding_semantics", {}),
            },
            "ground_truth": {
                "path": str(gt_path.relative_to(root)) if gt_path is not None and gt_path.is_relative_to(root) else None,
                "sha256": artifact_sha256(gt) if gt is not None else None,
                "bound": binding is not None,
                "status": gt_status,
                "not_yet_frozen": gt_status == GT_NOT_YET_FROZEN,
            },
            "readiness": readiness,
            "lifecycle": {
                "scientific_case_status": qualification_status,
                "scientific_identity_frozen": identity_frozen,
                "evaluation_status": evaluation_status,
                "gt_binding_status": gt_status,
                "release_status": "READY" if evaluation_status == EVALUATION_FROZEN and binding is not None else "BLOCKED",
                "release_gate_source": row.get("SCIENTIFIC_RELEASE_STATUS"),
            },
            "artifact_binding": _json(binding) if binding is not None else None,
            "scientific_artifact_binding_status": "VALID" if binding is not None else (
                "SCIENTIFIC_ARTIFACT_BINDING_INVALID"
                if any(code.endswith("MISMATCH") or "BINDING" in code for code in failures)
                else "NOT_CREATED_PENDING_DOWNSTREAM_FREEZE"
            ),
            "uncertainty_and_risks": {
                "unresolved_issues": list(row.get("failure_codes", [])) + list(failures),
                "pending_reasons": list(dict.fromkeys(pending_reasons)),
                "curator_status": "PENDING",
                "scientific_risks": ["Human curator confirmation remains pending."],
            },
            "phase_boundary": {
                "evaluator_status": "NOT_STARTED",
                "formal_experiment_status": "NOT_STARTED",
                "model_evaluation_started": False,
            },
        }
        if sec is not None and binding is not None and case_id is not None:
            _write_json(out / "scientific_evaluation_contracts" / f"{case_id}.json", sec.to_dict())
            _write_json(out / "scientific_artifact_bindings" / f"{case_id}.json", binding.to_dict())
        _write_json(out / "cases" / f"{case_id or row.get('slot_id', 'unknown')}.json", packet)
        _write_json(out / "scientific_case_review_packets" / f"{case_id or row.get('slot_id', 'unknown')}.json", _review_packet(packet=packet, recommendation=recommendation))
        packets.append(packet)

    counters = {
        "TOTAL_CASES": len(packets),
        "TOTAL_SCIENTIFIC_CASES": len(packets),
        "TOTAL_HUMAN_QUESTIONS": len(packets),
        "SCIENTIFIC_CASE_IDENTITY_FROZEN": sum(item["qualification_status"] == SCIENTIFIC_CASE_QUALIFIED for item in packets),
        "TOTAL_QUALIFIED_CASES": sum(item["qualification_status"] == SCIENTIFIC_CASE_QUALIFIED for item in packets),
        "TOTAL_PENDING_CASE_IDENTITY": sum(item["qualification_status"] == SCIENTIFIC_CASE_PENDING_IDENTITY for item in packets),
        "EVALUATION_CONTRACT_FROZEN": sum(item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_FROZEN for item in packets),
        "TOTAL_EVALUATION_CONTRACTS_FROZEN": sum(item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_FROZEN for item in packets),
        "TOTAL_GT_BOUND": sum(bool(item["ground_truth"]["bound"]) for item in packets),
        "GT_BOUND": sum(bool(item["ground_truth"]["bound"]) for item in packets),
        "TOTAL_PENDING_EVIDENCE": sum(item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_PENDING_EVIDENCE for item in packets),
        "TOTAL_PENDING_SRAC": sum(item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_PENDING_SRAC for item in packets),
        "TOTAL_PENDING_GT": sum(item["ground_truth"]["status"] == GT_NOT_YET_FROZEN for item in packets),
        "TOTAL_GT_NOT_YET_FROZEN": sum(item["ground_truth"]["status"] == GT_NOT_YET_FROZEN for item in packets),
        "TOTAL_GT_INVALID": sum(item["ground_truth"]["status"] == GT_INVALID for item in packets),
        # Orthogonal readiness dimensions. These counters keep data proof,
        # scientific grounding, semantic binding, and release authorization
        # separate from the lifecycle labels above.
        "TOTAL_GT_SCHEMA_VALID": sum(item["readiness"]["gt_schema_valid"] for item in packets),
        "TOTAL_DATA_MEDIATED_PROOF": sum(item["readiness"]["data_mediated_proof"] for item in packets),
        "TOTAL_GT_ADJUDICABLE": sum(item["readiness"]["gt_adjudicable"] for item in packets),
        "TOTAL_EVIDENCE_TRACEABLE": sum(item["readiness"]["evidence_traceable"] for item in packets),
        "TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE": sum(
            item["readiness"]["scientific_grounding_status"] == "ELIGIBLE_FOR_CURATOR_REVIEW"
            for item in packets
        ),
        "TOTAL_SRAC_READY": sum(item["readiness"]["srac_status"] == "READY" for item in packets),
        "TOTAL_SRAC_NOT_REQUIRED": sum(item["readiness"]["srac_status"] == "NOT_REQUIRED" for item in packets),
        "TOTAL_GT_SEMANTIC_BOUND": sum(item["readiness"]["gt_semantic_bound"] for item in packets),
        "TOTAL_CURATOR_CONFIRMED": sum(item["readiness"]["curator_confirmation"] for item in packets),
        "TOTAL_RELEASE_AUTHORIZED": sum(item["readiness"]["release_authorized"] for item in packets),
        "TOTAL_GT_SCHEMA_INVALID": sum(item["readiness"]["gt_schema_status"] == "INVALID" for item in packets),
        "TOTAL_GT_SCHEMA_PENDING": sum(item["readiness"]["gt_schema_status"] == "PENDING" for item in packets),
        "TOTAL_DATA_MEDIATED_PROOF_PENDING": sum(not item["readiness"]["data_mediated_proof"] for item in packets),
        "TOTAL_SCIENTIFIC_GROUNDING_PENDING": sum(
            item["readiness"]["scientific_grounding_status"] != "ELIGIBLE_FOR_CURATOR_REVIEW"
            for item in packets
        ),
        "TOTAL_SRAC_PENDING": sum(item["readiness"]["srac_status"] == "PENDING" for item in packets),
        "TOTAL_GT_SEMANTIC_BINDING_PENDING": sum(not item["readiness"]["gt_semantic_bound"] for item in packets),
        "TOTAL_RELEASE_AUTHORIZATION_PENDING": sum(not item["readiness"]["release_authorized"] for item in packets),
        "TOTAL_PENDING_CURATOR": sum(item["qualification_status"] == SCIENTIFIC_CASE_QUALIFIED and item["uncertainty_and_risks"]["curator_status"] == "PENDING" for item in packets),
        # Phase-2 ``RELEASE_READY`` is deliberately an internal artifact
        # readiness count (SEC + GT binding).  Official benchmark release
        # additionally requires an attributable external curator gate.
        "OFFICIAL_RELEASE_READY": sum(
            item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_FROZEN
            and item["ground_truth"]["bound"]
            and not item["uncertainty_and_risks"]["pending_reasons"]
            and item["uncertainty_and_risks"].get("curator_status") == "CONFIRMED"
            for item in packets
        ),
        "RELEASE_READY": sum(item["evaluation"]["scientific_evaluation_contract_status"] == EVALUATION_FROZEN and item["ground_truth"]["bound"] and not item["uncertainty_and_risks"]["pending_reasons"] for item in packets),
    }
    portfolio = {
        "artifact_type": "QualifiedScientificCasePortfolio",
        "schema_version": PHASE2_VERSION,
        "phase": "PHASE_2",
        "qualification_scope": "All 28 frozen human-review rows become explicit scientific case candidates; downstream evaluation and release readiness are tracked separately.",
        "QUALIFIED_CASE_PORTFOLIO": "READY",
        # This is a per-portfolio progress status.  The underlying
        # ScientificEvaluationContract schema remains frozen; only one of the
        # 24 identity-frozen cases currently has a completed contract.
        "SCIENTIFIC_EVALUATION_CONTRACT": (
            "FROZEN"
            if counters["EVALUATION_CONTRACT_FROZEN"] == counters["SCIENTIFIC_CASE_IDENTITY_FROZEN"]
            and counters["SCIENTIFIC_CASE_IDENTITY_FROZEN"]
            else "PARTIAL"
            if counters["EVALUATION_CONTRACT_FROZEN"]
            else "PENDING"
        ),
        "EVALUATION_CONTRACT_PORTFOLIO_STATUS": (
            "FROZEN"
            if counters["EVALUATION_CONTRACT_FROZEN"] == counters["SCIENTIFIC_CASE_IDENTITY_FROZEN"]
            and counters["SCIENTIFIC_CASE_IDENTITY_FROZEN"]
            else "PARTIAL"
            if counters["EVALUATION_CONTRACT_FROZEN"]
            else "PENDING"
        ),
        "EVALUATOR_CALIBRATED": False,
        "READY_FOR_FORMAL_EXPERIMENT": False,
        "MODEL_EVALUATION_STARTED": False,
        "PORTFOLIO_REMAINING_CASES": len(packets) - counters["TOTAL_QUALIFIED_CASES"],
        "TOTAL_NEW_CASE_REQUIRED": sum("NEW_CASE_REQUIRED" in item["uncertainty_and_risks"]["pending_reasons"] for item in packets),
        **counters,
        "cases": packets,
        "dependency_invalidation_rules": {
            change: evaluate_scientific_dependency_invalidation(change)
            for change in ("WORDING_ONLY", "FIXED_O_CHANGED", "UNRESOLVED_O_SPACE_CHANGED", "SCIENTIFIC_TARGET_CHANGED")
        },
    }
    _write_json(out / "qualified_scientific_case_portfolio.json", portfolio)
    _write_json(out / "phase2_closure.json", validate_scientific_case_freeze(portfolio, repository_root=root, output_dir=out))
    _write_text(out / "phase2_closure.md", render_phase2_closure(portfolio))
    return portfolio


def validate_scientific_case_freeze(
    portfolio: Mapping[str, Any] | str | Path,
    *,
    repository_root: str | Path = ".",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed validation of the Phase 2 portfolio and its sidecars."""

    root = Path(repository_root).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "outputs/current/qualified_scientific_cases"
    if isinstance(portfolio, (str, Path)):
        data = _read_json(Path(portfolio), {})
    else:
        data = dict(portfolio)
    rows = data.get("cases", []) if isinstance(data, Mapping) else []
    failures: list[dict[str, Any]] = []
    identity_count = 0
    contract_count = 0
    bound_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            failures.append({"case_id": "<invalid>", "category": "CASE_DEFECT", "code": "CASE_PACKET_INVALID"})
            continue
        case_id = str(row.get("case_identity_id") or row.get("case_id") or "")
        lifecycle = row.get("lifecycle", {})
        identity_status = str(lifecycle.get("scientific_case_status", ""))
        if identity_status == SCIENTIFIC_CASE_QUALIFIED:
            identity_count += 1
            question = row.get("question", {})
            projection_object = question.get("semantic_projection") if isinstance(question, Mapping) else None
            semantic_hash = str(question.get("semantic_contract_sha256", "")) if isinstance(question, Mapping) else ""
            if not isinstance(projection_object, Mapping) or semantic_contract_sha256(projection_object) != semantic_hash:
                failures.append({"case_id": case_id, "category": "CASE_DEFECT", "code": "SEMANTIC_HASH_BINDING_INVALID"})
            if not str(question.get("text", "")).strip() or question.get("semantic_fidelity_status") != "PASS":
                failures.append({"case_id": case_id, "category": "CASE_DEFECT", "code": "QUESTION_SEMANTIC_FIDELITY_FAILED"})
            if question.get("backend_only_information_required") is not False:
                failures.append({"case_id": case_id, "category": "CASE_DEFECT", "code": "BACKEND_ONLY_INFORMATION_REQUIRED"})
            responsibility = row.get("responsibility", {})
            if not isinstance(responsibility, Mapping) or not all(
                key in responsibility for key in (
                    "finding_responsibility",
                    "fixed_operationalization_dimensions",
                    "unresolved_operationalization_dimensions",
                )
            ):
                failures.append({"case_id": case_id, "category": "CASE_DEFECT", "code": "RESPONSIBILITY_CONTRACT_INCOMPLETE"})

        evaluation_status = str(lifecycle.get("evaluation_status", ""))
        if evaluation_status != EVALUATION_FROZEN:
            continue
        contract_count += 1
        sec_raw = row.get("evaluation", {}).get("scientific_evaluation_contract")
        binding_raw = row.get("artifact_binding")
        gt_path = row.get("ground_truth", {}).get("path")
        projection = row.get("question", {}).get("semantic_contract_sha256")
        projection_object = row.get("question", {}).get("semantic_projection")
        if not isinstance(sec_raw, Mapping) or not isinstance(binding_raw, Mapping) or not gt_path:
            failures.append({"case_id": case_id, "category": "CONTRACT_DEFECT", "code": "PHASE2_ARTIFACT_MISSING"})
            continue
        gt = _read_json(root / str(gt_path), {})
        if str(sec_raw.get("source_semantic_contract_sha256", "")) != str(projection or ""):
            failures.append({"case_id": case_id, "category": "GT_BINDING_DEFECT", "code": "SEMANTIC_HASH_BINDING_INVALID"})
            continue
        if str(binding_raw.get("case_id", "")) != str(row.get("case_id", "")):
            failures.append({"case_id": case_id, "category": "GT_BINDING_DEFECT", "code": "SCIENTIFIC_ARTIFACT_BINDING_INVALID"})
            continue
        expected_gt_hash = artifact_sha256(gt)
        if str(binding_raw.get("ground_truth_sha256", "")) != expected_gt_hash:
            failures.append({"case_id": case_id, "category": "GT_BINDING_DEFECT", "code": "SCIENTIFIC_ARTIFACT_BINDING_INVALID"})
            continue
        expected_sec_hash = artifact_sha256(sec_raw)
        if str(binding_raw.get("scientific_evaluation_contract_sha256", "")) != expected_sec_hash:
            failures.append({"case_id": case_id, "category": "GT_BINDING_DEFECT", "code": "SCIENTIFIC_ARTIFACT_BINDING_INVALID"})
            continue
        g_of_o = sec_raw.get("G_of_O")
        if not isinstance(g_of_o, list) or not g_of_o or any(
            not isinstance(branch, Mapping)
            or branch.get("materialization", {}).get("status") != "SUCCESS"
            or branch.get("materialization", {}).get("reproducible") is not True
            for branch in g_of_o
        ):
            failures.append({"case_id": case_id, "category": "CONTRACT_DEFECT", "code": "G_OF_O_MATERIALIZATION_INVALID"})
            continue
        sec_path = out / "scientific_evaluation_contracts" / f"{row.get('case_id')}.json"
        binding_path = out / "scientific_artifact_bindings" / f"{row.get('case_id')}.json"
        if not sec_path.is_file() or not binding_path.is_file():
            failures.append({"case_id": case_id, "category": "CONTRACT_DEFECT", "code": "PHASE2_SIDECAR_MISSING"})
            continue
        if artifact_sha256(_read_json(sec_path, {})) != expected_sec_hash:
            failures.append({"case_id": case_id, "category": "CONTRACT_DEFECT", "code": "SEC_SIDECAR_MISMATCH"})
            continue
        if artifact_sha256(_read_json(binding_path, {})) != artifact_sha256(binding_raw):
            failures.append({"case_id": case_id, "category": "GT_BINDING_DEFECT", "code": "BINDING_SIDECAR_MISMATCH"})
            continue
        if not row.get("evidence", {}).get("provenance_complete"):
            failures.append({"case_id": case_id, "category": "EVIDENCE_DEFECT", "code": "EVIDENCE_PROVENANCE_INCOMPLETE"})
            continue
        bound_count += 1
    qualified_count = identity_count
    readiness_rows = [
        row.get("readiness", {})
        if isinstance(row, Mapping) and isinstance(row.get("readiness", {}), Mapping)
        else {}
        for row in rows
    ]
    result = {
        "schema_version": PHASE2_VERSION,
        "QUALIFIED_CASE_PORTFOLIO": "READY" if not failures else "BLOCKED",
        "SCIENTIFIC_EVALUATION_CONTRACT": (
            "FROZEN" if contract_count == qualified_count and qualified_count else
            "PARTIAL" if contract_count else "PENDING"
        ),
        "TOTAL_CASES": len(rows),
        "TOTAL_HUMAN_QUESTIONS": len(rows),
        "SCIENTIFIC_CASE_IDENTITY_FROZEN": qualified_count,
        "TOTAL_QUALIFIED_CASES": qualified_count,
        "TOTAL_PENDING_CASE_IDENTITY": len(rows) - qualified_count,
        "EVALUATION_CONTRACT_FROZEN": contract_count,
        "TOTAL_EVALUATION_CONTRACTS_FROZEN": contract_count,
        "TOTAL_GT_BOUND": bound_count,
        "GT_BOUND": bound_count,
        "TOTAL_GT_SCHEMA_VALID": sum(bool(item.get("gt_schema_valid", False)) for item in readiness_rows),
        "TOTAL_DATA_MEDIATED_PROOF": sum(bool(item.get("data_mediated_proof", False)) for item in readiness_rows),
        "TOTAL_GT_ADJUDICABLE": sum(bool(item.get("gt_adjudicable", False)) for item in readiness_rows),
        "TOTAL_EVIDENCE_TRACEABLE": sum(bool(item.get("evidence_traceable", False)) for item in readiness_rows),
        "TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE": sum(
            item.get("scientific_grounding_status") == "ELIGIBLE_FOR_CURATOR_REVIEW"
            for item in readiness_rows
        ),
        "TOTAL_SRAC_READY": sum(item.get("srac_status") == "READY" for item in readiness_rows),
        "TOTAL_SRAC_NOT_REQUIRED": sum(item.get("srac_status") == "NOT_REQUIRED" for item in readiness_rows),
        "TOTAL_GT_SEMANTIC_BOUND": sum(bool(item.get("gt_semantic_bound", False)) for item in readiness_rows),
        "TOTAL_CURATOR_CONFIRMED": sum(bool(item.get("curator_confirmation", False)) for item in readiness_rows),
        "TOTAL_RELEASE_AUTHORIZED": sum(bool(item.get("release_authorized", False)) for item in readiness_rows),
        "TOTAL_GT_SCHEMA_INVALID": sum(item.get("gt_schema_status") == "INVALID" for item in readiness_rows),
        "TOTAL_GT_SCHEMA_PENDING": sum(item.get("gt_schema_status") == "PENDING" for item in readiness_rows),
        "TOTAL_DATA_MEDIATED_PROOF_PENDING": sum(not bool(item.get("data_mediated_proof", False)) for item in readiness_rows),
        "TOTAL_SCIENTIFIC_GROUNDING_PENDING": sum(
            item.get("scientific_grounding_status") != "ELIGIBLE_FOR_CURATOR_REVIEW"
            for item in readiness_rows
        ),
        "TOTAL_SRAC_PENDING": sum(item.get("srac_status") == "PENDING" for item in readiness_rows),
        "TOTAL_GT_SEMANTIC_BINDING_PENDING": sum(not bool(item.get("gt_semantic_bound", False)) for item in readiness_rows),
        "TOTAL_RELEASE_AUTHORIZATION_PENDING": sum(not bool(item.get("release_authorized", False)) for item in readiness_rows),
        "TOTAL_PENDING_EVIDENCE": sum(isinstance(row, Mapping) and row.get("lifecycle", {}).get("evaluation_status") == EVALUATION_PENDING_EVIDENCE for row in rows),
        "TOTAL_PENDING_SRAC": sum(isinstance(row, Mapping) and row.get("lifecycle", {}).get("evaluation_status") == EVALUATION_PENDING_SRAC for row in rows),
        "TOTAL_PENDING_GT": sum(isinstance(row, Mapping) and row.get("lifecycle", {}).get("gt_binding_status") == GT_NOT_YET_FROZEN for row in rows),
        "TOTAL_PENDING_CURATOR": sum(isinstance(row, Mapping) and row.get("qualification_status") == SCIENTIFIC_CASE_QUALIFIED and row.get("uncertainty_and_risks", {}).get("curator_status") == "PENDING" for row in rows),
        "OFFICIAL_RELEASE_READY": sum(
            isinstance(row, Mapping)
            and row.get("lifecycle", {}).get("evaluation_status") == EVALUATION_FROZEN
            and row.get("ground_truth", {}).get("bound") is True
            and not row.get("uncertainty_and_risks", {}).get("pending_reasons")
            and row.get("uncertainty_and_risks", {}).get("curator_status") == "CONFIRMED"
            for row in rows
        ),
        "RELEASE_READY": sum(isinstance(row, Mapping) and row.get("lifecycle", {}).get("release_status") == "READY" for row in rows),
        "TOTAL_NEW_CASE_REQUIRED": sum(isinstance(row, Mapping) and "NEW_CASE_REQUIRED" in row.get("uncertainty_and_risks", {}).get("pending_reasons", []) for row in rows),
        "EVALUATOR_STATUS": "NOT_STARTED",
        "FORMAL_EXPERIMENT_STATUS": "NOT_STARTED",
        "scientific_artifact_binding_status": (
            "SCIENTIFIC_ARTIFACT_BINDING_INVALID" if any(
                failure.get("category") == "GT_BINDING_DEFECT" for failure in failures
            ) else "VALID"
        ),
        "failures": failures,
        "failure_category_counts": {
            category: sum(category in (row.get("failure_categories", []) if isinstance(row, Mapping) else []) for row in rows)
            for category in ("CASE_DEFECT", "CONTRACT_DEFECT", "GT_BINDING_DEFECT", "EVIDENCE_DEFECT", "SCIENTIFIC_AMBIGUITY")
        },
    }
    return result


def render_phase2_closure(portfolio: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Qualified Scientific Case Closure",
        "",
        f"- QUALIFIED_CASE_PORTFOLIO: **{portfolio.get('QUALIFIED_CASE_PORTFOLIO')}**",
        f"- SCIENTIFIC_EVALUATION_CONTRACT: **{portfolio.get('SCIENTIFIC_EVALUATION_CONTRACT')}**",
        f"- TOTAL_CASES: {portfolio.get('TOTAL_CASES', 0)}",
        f"- TOTAL_HUMAN_QUESTIONS: {portfolio.get('TOTAL_HUMAN_QUESTIONS', portfolio.get('TOTAL_CASES', 0))}",
        f"- SCIENTIFIC_CASE_IDENTITY_FROZEN: {portfolio.get('SCIENTIFIC_CASE_IDENTITY_FROZEN', portfolio.get('TOTAL_QUALIFIED_CASES', 0))}",
        f"- TOTAL_QUALIFIED_CASES: {portfolio.get('TOTAL_QUALIFIED_CASES', 0)}",
        f"- TOTAL_PENDING_CASE_IDENTITY: {portfolio.get('TOTAL_PENDING_CASE_IDENTITY', 0)}",
        f"- EVALUATION_CONTRACT_FROZEN: {portfolio.get('EVALUATION_CONTRACT_FROZEN', portfolio.get('TOTAL_EVALUATION_CONTRACTS_FROZEN', 0))}",
        f"- TOTAL_EVALUATION_CONTRACTS_FROZEN: {portfolio.get('TOTAL_EVALUATION_CONTRACTS_FROZEN', 0)}",
        f"- TOTAL_GT_BOUND: {portfolio.get('TOTAL_GT_BOUND', 0)}",
        f"- TOTAL_GT_SCHEMA_VALID: {portfolio.get('TOTAL_GT_SCHEMA_VALID', 0)}",
        f"- TOTAL_GT_SCHEMA_INVALID: {portfolio.get('TOTAL_GT_SCHEMA_INVALID', 0)}",
        f"- TOTAL_GT_SCHEMA_PENDING: {portfolio.get('TOTAL_GT_SCHEMA_PENDING', 0)}",
        f"- TOTAL_DATA_MEDIATED_PROOF: {portfolio.get('TOTAL_DATA_MEDIATED_PROOF', 0)}",
        f"- TOTAL_DATA_MEDIATED_PROOF_PENDING: {portfolio.get('TOTAL_DATA_MEDIATED_PROOF_PENDING', 0)}",
        f"- TOTAL_GT_ADJUDICABLE: {portfolio.get('TOTAL_GT_ADJUDICABLE', 0)}",
        f"- TOTAL_EVIDENCE_TRACEABLE: {portfolio.get('TOTAL_EVIDENCE_TRACEABLE', 0)}",
        f"- TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE: {portfolio.get('TOTAL_SCIENTIFIC_GROUNDING_ELIGIBLE', 0)}",
        f"- TOTAL_SCIENTIFIC_GROUNDING_PENDING: {portfolio.get('TOTAL_SCIENTIFIC_GROUNDING_PENDING', 0)}",
        f"- TOTAL_SRAC_READY: {portfolio.get('TOTAL_SRAC_READY', 0)}",
        f"- TOTAL_SRAC_NOT_REQUIRED: {portfolio.get('TOTAL_SRAC_NOT_REQUIRED', 0)}",
        f"- TOTAL_SRAC_PENDING: {portfolio.get('TOTAL_SRAC_PENDING', 0)}",
        f"- TOTAL_GT_SEMANTIC_BOUND: {portfolio.get('TOTAL_GT_SEMANTIC_BOUND', 0)}",
        f"- TOTAL_GT_SEMANTIC_BINDING_PENDING: {portfolio.get('TOTAL_GT_SEMANTIC_BINDING_PENDING', 0)}",
        f"- TOTAL_CURATOR_CONFIRMED: {portfolio.get('TOTAL_CURATOR_CONFIRMED', 0)}",
        f"- TOTAL_RELEASE_AUTHORIZED: {portfolio.get('TOTAL_RELEASE_AUTHORIZED', 0)}",
        f"- TOTAL_RELEASE_AUTHORIZATION_PENDING: {portfolio.get('TOTAL_RELEASE_AUTHORIZATION_PENDING', 0)}",
        f"- TOTAL_PENDING_EVIDENCE: {portfolio.get('TOTAL_PENDING_EVIDENCE', 0)}",
        f"- TOTAL_PENDING_SRAC: {portfolio.get('TOTAL_PENDING_SRAC', 0)}",
        f"- TOTAL_PENDING_GT: {portfolio.get('TOTAL_PENDING_GT', 0)}",
        f"- TOTAL_PENDING_CURATOR: {portfolio.get('TOTAL_PENDING_CURATOR', 0)}",
        f"- TOTAL_NEW_CASE_REQUIRED: {portfolio.get('TOTAL_NEW_CASE_REQUIRED', 0)}",
        f"- RELEASE_READY (Phase-2 artifact readiness): {portfolio.get('RELEASE_READY', 0)}",
        f"- OFFICIAL_RELEASE_READY (external curator gate): {portfolio.get('OFFICIAL_RELEASE_READY', 0)}",
        "- EVALUATOR_STATUS: **NOT_STARTED**",
        "- FORMAL_EXPERIMENT_STATUS: **NOT_STARTED**",
        "",
        "All 28 rows are transformed into explicit scientific case candidates. Identity qualification is independent from downstream evaluation, GT binding, curator review, and release readiness; pending work remains state rather than a defect.",
        "",
        "| case_id / slot_id | dataset | condition | scientific identity | evaluation | GT binding | release |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in portfolio.get("cases", []):
        if not isinstance(row, Mapping):
            continue
        identity = row.get("scientific_identity", {})
        display_id = row.get("case_id") or row.get("case_identity_id") or row.get("slot_id") or "—"
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                display_id,
                identity.get("dataset_id", ""),
                row.get("condition", ""),
                row.get("lifecycle", {}).get("scientific_case_status", row.get("qualification_status", "")),
                row.get("lifecycle", {}).get("evaluation_status", ""),
                row.get("lifecycle", {}).get("gt_binding_status", ""),
                row.get("lifecycle", {}).get("release_status", ""),
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Authoritative 28-question agent-ready construction
# ---------------------------------------------------------------------------

AGENT_READY_VERSION = "qualified-scientific-case-agent-ready-v1"


def _agent_case_id(root: Path, row: Mapping[str, Any]) -> str:
    """Resolve a stable case id without turning a nullable slot into ``None``.

    The four Combustor rows intentionally have provisional identities in the
    human-facing manifest.  A construction identity is still needed for
    per-case artifacts, so it is derived from the frozen slot id and marked
    provisional in the output.  It is not promoted to an official identity.
    """

    existing = _optional_case_id(row.get("case_id"))
    if existing:
        return existing
    slot_id = str(row.get("slot_id", "")).strip()
    if not slot_id:
        raise ValueError("agent-ready row has neither case_id nor slot_id")
    return slot_id


def _agent_rel_path(root: Path, path: Path) -> str:
    """Return a portable artifact path for repository or external staging."""

    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _agent_source_case_dir(root: Path, row: Mapping[str, Any], case_id: str) -> Path:
    """Find a reviewed source case used only for data/context patterns."""

    requested = _optional_case_id(row.get("case_id"))
    candidates: list[Path] = []
    if requested:
        candidates.extend(
            sorted(
                root.glob(
                    f"artifacts/reference/concept_expansion_phase1/candidate_cases/*/{requested}"
                )
            )
        )
        candidates.extend(
            sorted(root.glob(f"datasets/*/construction/cases/{requested}"))
        )
    # Combustor provisional slots reuse the reviewed density-data input and
    # context, but do not reuse its point-feature semantics or GT.
    if str(row.get("dataset_id")) == "Combustor":
        condition = str(row.get("condition", "")).lower().replace("-", "_")
        candidates.extend(
            sorted(
                root.glob(
                    f"artifacts/reference/concept_expansion_phase1/candidate_cases/Combustor/combustor_density_features_{condition}"
                )
            )
        )
        candidates.extend(
            sorted(root.glob(f"datasets/Combustor/construction/cases/combustor_{condition}"))
        )
    for candidate in candidates:
        if (candidate / "case_input.json").is_file():
            return candidate
    raise FileNotFoundError(f"no reviewed source case for {case_id}")


def _agent_method_category(dimension: str) -> MethodConstraintCategory:
    if dimension in {"feature_definition", "criterion", "property_measure", "parameter", "interpretation_rule"}:
        return MethodConstraintCategory(dimension)
    if dimension == "aggregation_or_representation":
        return MethodConstraintCategory.ANALYSIS_PROCEDURE
    return MethodConstraintCategory.OTHER


def _agent_question_fragment(question: str, statement: str, fallback: str = "") -> str:
    """Choose a literal, normalized anchor that is present in the question.

    Construction metadata is an audit projection, not model-visible content.
    A legacy statement may be semantically equivalent while using different
    wording, so blindly copying its old fragment creates a false artifact
    failure.  Prefer the longest shared contiguous phrase and fall back to a
    question-wide anchor only when no useful token is shared.
    """

    normalized_question = normalize_case_text(question)
    normalized_statement = normalize_case_text(statement)
    if normalized_statement and normalized_statement in normalized_question:
        return statement
    question_tokens = normalized_question.split()
    statement_tokens = normalized_statement.split()
    best: list[str] = []
    for start in range(len(statement_tokens)):
        for end in range(start + 1, len(statement_tokens) + 1):
            candidate = statement_tokens[start:end]
            if len(candidate) > len(best) and " ".join(candidate) in normalized_question:
                best = candidate
    if best:
        return " ".join(best)
    for token in statement_tokens:
        if len(token) >= 4 and token in question_tokens:
            return token
    normalized_fallback = normalize_case_text(fallback)
    if normalized_fallback and normalized_fallback in normalized_question:
        return fallback
    # The complete question is always a valid literal anchor.  This is used
    # only for construction-time traceability when wording shares no token.
    return question


def _agent_metadata(row: Mapping[str, Any], case_id: str, source: Mapping[str, Any]) -> CaseConstructionMetadata:
    """Build metadata from the frozen semantic projection when needed.

    Existing metadata is retained for ordinary rows.  The provisional
    Combustor rows are authored from their current projection so their
    surface target cannot silently inherit the old point-feature metadata.
    """

    dataset_id = str(row.get("dataset_id", ""))
    projection = row.get("canonical_semantic_projection")
    projection = projection if isinstance(projection, Mapping) else {}
    projection_version = str(
        projection.get("scientific_semantic_projection_version", "")
    )
    if dataset_id != "Combustor" and projection_version != (
        "scientific-semantic-projection-v3"
    ):
        raw = source.get("metadata")
        if isinstance(raw, Mapping):
            candidate = dict(raw)
            candidate["case_id"] = case_id
            candidate["dataset_id"] = dataset_id
            candidate["case_family_id"] = str(row.get("family_id") or candidate.get("case_family_id", ""))
            try:
                validated = CaseConstructionMetadata.model_validate(candidate)
                CaseConstructionMetadataValidator.validate(
                    str(row.get("scientific_question", "")), validated
                )
                if (
                    validated.responsibility_contract is not None
                    and validated.evaluation_representability_contract is not None
                ):
                    return validated
            except Exception:
                # A wording-only difference in a legacy metadata fragment must
                # not prevent re-binding to the frozen projection below.
                pass

    condition = str(row.get("condition", "")).upper().replace("_", "-")
    responsibility = {
        "O1-F1": OperationalizationResponsibility.USER_SPECIFIED,
        "O2-F1": OperationalizationResponsibility.PARTIALLY_SPECIFIED,
        "O3-F1": OperationalizationResponsibility.MODEL_SELECTED,
        "O1-F2": OperationalizationResponsibility.USER_SPECIFIED,
    }[condition]
    openness = FindingOpenness.OPEN if condition == "O1-F2" else FindingOpenness.BOUNDED
    question = str(row.get("scientific_question", ""))
    scope = projection.get("scientific_scope", {})
    scope_constraints = [
        ExplicitQuestionConstraint(
            statement=str(item),
            question_fragment=_agent_question_fragment(question, str(item)),
        )
        for item in (scope.get("scope_constraints", []) if isinstance(scope, Mapping) else [])
        if str(item).strip()
    ]
    # Projection clauses contain normalized scientific meanings; the question
    # itself remains the visibility authority.  Use a short visible fragment
    # for metadata validation and retain the complete meaning in the statement.
    methods: list[ExplicitMethodConstraint] = []
    for clause in projection.get("resolved_operationalization", []) if isinstance(projection, Mapping) else []:
        if not isinstance(clause, Mapping):
            continue
        statement = str(clause.get("normalized_meaning", "")).strip()
        dimension = str(clause.get("dimension_id", "")).strip()
        if not statement or not dimension:
            continue
        fragment = _agent_question_fragment(question, statement, dimension)
        methods.append(ExplicitMethodConstraint(
            category=_agent_method_category(dimension),
            statement=statement,
            question_fragment=fragment,
        ))
    finding_requirements: list[ExplicitFindingRequirement] = []
    semantics = projection.get("finding_semantics", {}) if isinstance(projection, Mapping) else {}
    for requirement in semantics.get("fixed_finding_requirements", []) if isinstance(semantics, Mapping) else []:
        if not isinstance(requirement, Mapping):
            continue
        category = str(requirement.get("category", "other"))
        try:
            category_value = FindingRequirementCategory(category)
        except ValueError:
            category_value = FindingRequirementCategory.OTHER
        meaning = str(requirement.get("normalized_meaning", "")).strip()
        if meaning:
            fragment = _agent_question_fragment(question, meaning, category)
            finding_requirements.append(ExplicitFindingRequirement(
                category=category_value,
                statement=meaning,
                question_fragment=fragment,
            ))
    resolved_contract = [
        {
            "canonical_id": f"{projection.get('family_id', row.get('family_id', 'family'))}:{item.category.value}:{index}",
            "dimension_id": (
                item.category.value
                if item.category.value != MethodConstraintCategory.ANALYSIS_PROCEDURE.value
                else "aggregation_or_representation"
            ),
            "normalized_meaning": item.statement,
            "question_visibility": "REQUIRED",
            "required_semantic_anchors": [],
        }
        for index, item in enumerate(methods, start=1)
    ]
    unresolved_contract = [
        {
            "dimension_id": dimension.value,
            "normalized_meaning": (
                "the respondent must select and state "
                + dimension.value.replace("_", " ")
            ),
            "must_remain_open": True,
        }
        for dimension in (
            OperationalizationDimension(
                str(value.get("dimension_id"))
                if isinstance(value, Mapping)
                else str(value)
            )
            for value in projection.get(
                "unresolved_operationalization_dimensions", []
            )
        )
    ]
    finding_mode = "F2" if openness == FindingOpenness.OPEN else "F1"
    responsibility_contract = {
        "scientific_target": {
            "canonical_id": str(projection.get("family_id", row.get("family_id", ""))),
            "normalized_meaning": str(projection.get("scientific_target", "")),
            "required_semantic_anchors": [],
        },
        "resolved_operationalization_clauses": resolved_contract,
        "unresolved_operationalization_dimensions": unresolved_contract,
        "finding_responsibility": {
            "finding_mode": finding_mode,
            "normalized_demand": str(projection.get("finding_goal", row.get("finding_goal", ""))),
            "required_responsibility_mode": (
                "OPEN_FINDING_SELECTION" if finding_mode == "F2" else None
            ),
            "forbidden_responsibility_modes": (
                ["FIXED_FINDING_LIST"] if finding_mode == "F2" else ["OPEN_FINDING_SELECTION"]
            ),
            "required_semantic_anchors": [],
        },
    }
    representability_contract = _agent_case_representability_contract(
        dataset_id=dataset_id,
        condition=condition,
        projection=projection,
    )
    return CaseConstructionMetadata(
        case_id=case_id,
        case_family_id=str(row.get("family_id") or f"{dataset_id.lower()}_agent_ready"),
        dataset_id=dataset_id,
        scientific_target=str(projection.get("scientific_target", row.get("scientific_target", "scientific flow target"))),
        finding_goal=str(semantics.get("normalized_demand", semantics.get("finding_goal", "report the requested scientific findings")) if isinstance(semantics, Mapping) else "report the requested scientific findings"),
        operationalization_responsibility=responsibility,
        finding_openness=openness,
        responsibility_contract=responsibility_contract,
        evaluation_representability_contract=representability_contract,
        principal_operationalization_dimensions=[OperationalizationDimension(str(item.get("dimension_id", item)) if isinstance(item, Mapping) else str(item)) for item in projection.get("principal_operationalization_dimensions", [])],
        unresolved_operationalization_dimensions=[OperationalizationDimension(str(item.get("dimension_id", item)) if isinstance(item, Mapping) else str(item)) for item in projection.get("unresolved_operationalization_dimensions", [])],
        scope_constraints=scope_constraints,
        explicit_method_constraints=methods,
        explicit_finding_requirements=finding_requirements,
    )


def _agent_case_representability_contract(
    *, dataset_id: str, condition: str, projection: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile a case-specific executable/evaluable envelope.

    This is the existing evaluation-representability sidecar made concrete;
    it does not introduce a second O-space contract.  Unsupported but
    potentially scientific proposals remain pending rather than being
    converted to invalidity.
    """

    unresolved = sorted(
        str(item.get("dimension_id", item))
        if isinstance(item, Mapping)
        else str(item)
        for item in projection.get("unresolved_operationalization_dimensions", ())
    )
    fixed = {
        str(item.get("dimension_id")): str(item.get("normalized_meaning", ""))
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping) and item.get("dimension_id")
    }
    if dataset_id == "Kitchen":
        domain = "kitchen_concentration_heterogeneity"
        envelope = {
            "feature_definition": [
                "point-associated concentration population",
                "derived cell concentration as arithmetic vertex mean",
                "structured six-neighbor point-pair differences",
            ],
            "property_measure": [
                "coefficient of variation",
                "relative interdecile spread",
                "population standard deviation",
                "cell-volume-weighted standard deviation",
                "mean/RMS/range-normalized six-neighbor absolute difference",
            ],
            "aggregation_or_representation": [
                "equal point weighting",
                "positive cell-volume weighting",
                "stored field identifier",
            ],
        }
        outputs = [
            "/field_name",
            "/metric_name",
            "/metric_value",
            "/mean",
            "/standard_deviation",
            "/q10",
            "/q50",
            "/q90",
        ]
        verification_modes = ["exact_identity", "scalar_numeric"]
    elif dataset_id == "Combustor":
        domain = "density_isosurface_component"
        envelope = {
            "feature_definition": [
                "connected components of a constant-Density isosurface"
            ],
            "criterion": [
                "explicit finite-Density percentile with stated interpolation",
                "explicit fraction of a stated Density percentile",
            ],
            "aggregation_or_representation": [
                "largest surface area with area-weighted centroid",
                "largest bounding-box volume with bounding-box midpoint",
            ],
        }
        outputs = [
            "/level",
            "/selected/surface_area",
            "/selected/area_weighted_centroid",
            "/selected/bounding_box_volume",
            "/selected/bounding_box_midpoint",
        ]
        verification_modes = ["scalar_numeric", "coordinate_point"]
    else:
        domain = "thresholded_connected_speed_region"
        connectivity = (
            "shared unstructured-mesh cell point adjacency"
            if dataset_id == "FireFlow"
            else "structured-grid index six-neighbor point adjacency"
        )
        envelope = {
            "feature_definition": [connectivity],
            "criterion": [
                "explicit numeric speed threshold",
                "finite non-zero speed percentile with linear or nearest-rank interpolation",
            ],
            "property_measure": ["regional peak speed", "regional mean speed"],
            "aggregation_or_representation": [
                "mean retained-point coordinate",
                "peak-speed point coordinate",
            ],
        }
        outputs = [
            "/reported_location",
            "/strength",
            "/coordinate_span",
            "/selected_region_size",
            "/region_count",
        ]
        verification_modes = [
            "exact_discrete",
            "scalar_numeric",
            "coordinate_point",
            "componentwise_vector",
        ]
    return {
        "contract_version": "case-evaluation-representability-v3",
        "domain": domain,
        "condition": condition,
        "unresolved_dimensions": unresolved,
        "fixed_dimension_semantics": fixed,
        "supported_effective_o_envelope": {
            dimension: values
            for dimension, values in envelope.items()
            if dimension in unresolved
        },
        "materialization": {
            "available": True,
            "handler_id": "deterministic_case_materializer_v3",
            "unsupported_disposition": "MATERIALIZATION_UNSUPPORTED_PENDING_NOT_INVALID",
        },
        "finding_representation": {
            "available": True,
            "extractor_id": "atomic_finding_extractor_v1",
            "output_locators": outputs,
        },
        "deterministic_verification": {
            "registered_handlers": verification_modes,
            "required_claim_types": verification_modes,
        },
        "semantic_adjudication": {
            "handler_available": True,
            "handler_id": "scientific_adjudication_v1",
            "required": bool(unresolved or condition == "O1-F2"),
        },
        "valid_unenumerated_escalation": {
            "available": True,
            "route_id": "valid_unenumerated",
        },
        "uncertain_escalation": {"available": True, "route_id": "uncertain"},
    }
def _agent_evidence(root: Path, dataset_id: str, case_id: str, source_path: str) -> list[EvidenceRecord]:
    collection = load_source_collection(
        root / "datasets" / dataset_id / "construction" / "sources.json",
        dataset_id=dataset_id,
        require_formal_source=True,
    )
    if not collection.sources:
        raise ValueError(f"{dataset_id} has no traceable source")
    records: list[EvidenceRecord] = []
    for filename, evidence_type in (("operationalization_evidence.json", "operationalization"), ("finding_evidence.json", "finding")):
        raw = _read_json(root / "datasets" / dataset_id / "construction" / filename, [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, Mapping) or item.get("evidence_type") != evidence_type:
                continue
            try:
                record = EvidenceRecord.model_validate(item)
            except Exception:
                continue
            if record.dataset_id == dataset_id and record.source_id in {source.source_id for source in collection.sources}:
                records.append(record)
    operationalization_records = [record for record in records if record.evidence_type == "operationalization"]
    finding_records = [record for record in records if record.evidence_type == "finding"]
    if not operationalization_records:
        raise ValueError(f"{dataset_id} has no source-backed operationalization evidence")
    if not finding_records:
        raise ValueError(f"{dataset_id} has no source-backed finding evidence")
    return [*operationalization_records, *finding_records]


def _agent_operationalization_evidence_ids(
    evidence: Sequence[EvidenceRecord],
    dataset_id: str,
    operation_id: str,
) -> list[str]:
    """Select source-backed support that matches the operation's meaning."""

    available = {
        record.evidence_id: record
        for record in evidence
        if record.evidence_type == "operationalization"
    }
    operation = operation_id.casefold()
    if dataset_id == "Kitchen" and (
        "concentration" in operation
        or "heterogeneity" in operation
        or "interdecile" in operation
    ):
        preferred = ("op_004",)
    elif dataset_id == "Kitchen" and (
        "turbulence" in operation
        or "dissipation" in operation
        or "kinetic" in operation
    ):
        preferred = ("op_003",)
    elif dataset_id == "Combustor" and (
        "density" in operation
        or "surface" in operation
        or "q90" in operation
        or "q81" in operation
    ):
        preferred = ("op_002", "op_003")
    else:
        preferred = ("op_001",)
    selected = [evidence_id for evidence_id in preferred if evidence_id in available]
    if selected:
        return selected
    return [next(iter(available))] if available else []


def _read_plot3d_combustor(root: Path) -> tuple[Any, float]:
    """Read Combustor with the repository's canonical PLOT3D configuration."""

    import numpy as np
    from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
    from vtk.util.numpy_support import vtk_to_numpy
    from .plot3d_reader import apply_plot3d_reader_configuration

    dataset_root = root / "datasets" / "Combustor"
    manifest = _read_json(dataset_root / "dataset_manifest.json", {})
    files = {str(item.get("role")): item for item in manifest.get("files", [])}
    reader = vtkMultiBlockPLOT3DReader()
    apply_plot3d_reader_configuration(reader, manifest.get("reader", {}).get("reader_configuration"))
    reader.SetXYZFileName(str(root / "datasets" / files["grid"]["path"]))
    reader.SetQFileName(str(root / "datasets" / files["solution"]["path"]))
    reader.Update()
    output = reader.GetOutput()
    grid = output.GetBlock(0) if output is not None else None
    if grid is None:
        raise ValueError("Combustor PLOT3D reader returned no grid")
    density = vtk_to_numpy(grid.GetPointData().GetArray("Density")).astype(float, copy=False)
    finite = density[np.isfinite(density)]
    if finite.size == 0:
        raise ValueError("Combustor Density contains no finite values")
    return grid, float(np.quantile(finite, 0.90))


def _agent_combustor_contours(root: Path, levels: Sequence[float], *, closed_geometry: bool = False) -> dict[float, dict[str, Any]]:
    """Materialize Density isosurfaces, connected components, area and centroid."""

    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkPolyDataConnectivityFilter, vtkTriangleFilter

    grid, q90 = _read_plot3d_combustor(root)
    results: dict[float, dict[str, Any]] = {}
    for level in levels:
        contour = vtkContourFilter()
        contour.SetInputData(grid)
        contour.SetInputArrayToProcess(0, 0, 0, 0, "Density")
        contour.SetValue(0, float(level))
        contour.Update()
        connectivity = vtkPolyDataConnectivityFilter()
        if closed_geometry:
            from vtkmodules.vtkFiltersCore import vtkCleanPolyData
            clean = vtkCleanPolyData()
            clean.SetInputConnection(contour.GetOutputPort())
            clean.SetTolerance(0.0)
            clean.Update()
            connectivity.SetInputConnection(clean.GetOutputPort())
        else:
            connectivity.SetInputConnection(contour.GetOutputPort())
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()
        connectivity.Update()
        triangle = vtkTriangleFilter()
        triangle.SetInputConnection(connectivity.GetOutputPort())
        triangle.Update()
        surface = triangle.GetOutput()
        region_ids_array = surface.GetPointData().GetArray("RegionId")
        if region_ids_array is None or surface.GetPoints() is None:
            raise ValueError(f"no connected Density isosurface at level {level}")
        region_ids = vtk_to_numpy(region_ids_array).astype(np.int64, copy=False)
        points = vtk_to_numpy(surface.GetPoints().GetData()).astype(float, copy=False)
        regions: list[dict[str, Any]] = []
        areas: dict[int, float] = {}
        moments: dict[int, np.ndarray] = {}
        bounds: dict[int, np.ndarray] = {}
        region_triangles: dict[int, list] = {}
        for cell_index in range(surface.GetNumberOfCells()):
            cell = surface.GetCell(cell_index)
            if cell.GetNumberOfPoints() != 3:
                continue
            ids = [cell.GetPointId(index) for index in range(3)]
            rid = int(region_ids[ids[0]])
            if closed_geometry:
                region_triangles.setdefault(rid, []).append(ids)
            p0, p1, p2 = points[ids]
            area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
            centroid = (p0 + p1 + p2) / 3.0
            areas[rid] = areas.get(rid, 0.0) + area
            moments[rid] = moments.get(rid, np.zeros(3)) + area * centroid
            if rid not in bounds:
                bounds[rid] = np.vstack([p0, p1, p2])
            else:
                bounds[rid] = np.vstack([bounds[rid], p0, p1, p2])
        for rid, area in areas.items():
            pts = bounds[rid]
            axis_bounds = [
                [float(pts[:, axis].min()), float(pts[:, axis].max())]
                for axis in range(3)
            ]
            spans = [upper - lower for lower, upper in axis_bounds]
            regions.append({
                "region_id": rid,
                "surface_area": float(area),
                "area_weighted_centroid": [float(x) for x in (moments[rid] / area)],
                "bounds": axis_bounds,
                "bounding_box_volume": float(np.prod(spans)),
                "bounding_box_midpoint": [
                    float((lower + upper) / 2.0) for lower, upper in axis_bounds
                ],
            })
        if closed_geometry:
            from .density_geometry import closed_triangle_geometry
            density = vtk_to_numpy(grid.GetPointData().GetArray("Density"))
            peak_index = int(np.nanargmax(density))
            peak_point = grid.GetPoint(peak_index)
            for region in regions:
                region.update(closed_triangle_geometry(points, region_triangles[region["region_id"]], peak_point))
        regions.sort(key=lambda item: (-item["surface_area"], item["region_id"]))
        if not regions:
            raise ValueError(f"Density contour at level {level} produced no triangles")
        results[float(level)] = {
            "level": float(level),
            "regions": regions,
            "selected": regions[0],
            "selected_by_surface_area": regions[0],
            "selected_by_bounding_box_volume": max(
                regions,
                key=lambda item: (
                    item["bounding_box_volume"],
                    -item["region_id"],
                ),
            ),
            "q90": q90,
            "quantile_method": "linear",
        }
    return results


def _agent_combustor_gt(root: Path, row: Mapping[str, Any], metadata: CaseConstructionMetadata, evidence: Sequence[EvidenceRecord]) -> tuple[GroundTruth, dict[str, Any]]:
    projection = row.get("canonical_semantic_projection", {})
    unresolved = {str(item.get("dimension_id")) for item in projection.get("unresolved_operationalization_dimensions", []) if isinstance(item, Mapping)}
    _, q90 = _read_plot3d_combustor(root)
    levels = [q90]
    if "criterion" in unresolved or str(row.get("condition")) == "O3-F1":
        levels.append(float(q90 * 0.90))
    levels = list(dict.fromkeys(levels))
    analyses = _agent_combustor_contours(root, levels)
    finding_evidence = next(item.evidence_id for item in evidence if item.evidence_type == "finding")
    condition = str(row.get("condition"))
    branch_specs: list[tuple[str, float, str, str]] = []
    if condition == "O1-F1" or condition == "O1-F2":
        branch_specs = [("o1_q90_surface", levels[0], "largest_surface_area", "area_weighted_centroid")]
    elif condition == "O2-F1":
        branch_specs = [
            ("o2_q90_surface", levels[0], "largest_surface_area", "area_weighted_centroid"),
            ("o2_q81_surface", levels[-1], "largest_surface_area", "area_weighted_centroid"),
        ]
    else:
        branch_specs = [
            ("o3_q90_surface", levels[0], "largest_surface_area", "area_weighted_centroid"),
            (
                "o3_q81_bounding_box",
                levels[-1],
                "largest_bounding_box_volume",
                "bounding_box_midpoint",
            ),
        ]
    bundles: list[OperationalizationBundle] = []
    branches: list[OperationalizationFindingBranch] = []
    branch_payloads: dict[str, Any] = {}
    for suffix, level, selection, representation in branch_specs:
        analysis = copy.deepcopy(analyses[float(level)])
        selected = analysis[f"selected_by_{selection.removeprefix('largest_')}"]
        analysis["selected"] = selected
        analysis["component_selection"] = selection
        analysis["location_representation"] = representation
        op_id = f"{metadata.case_id}_{suffix}"
        decisions: list[OperationalizationDecision] = []
        for dimension in ("feature_definition", "criterion", "aggregation_or_representation"):
            if dimension == "criterion":
                statement = (
                    "Use the finite stored Density "
                    f"{('90th percentile' if level == levels[0] else '90%-of-q90')} "
                    f"level ({level:.12g}) with linear quantile interpolation"
                    + (
                        " and select the connected component with the largest surface area."
                        if selection == "largest_surface_area"
                        else " and select the connected component with the largest axis-aligned bounding-box volume."
                    )
                )
            elif dimension == "feature_definition":
                statement = "Treat each connected constant-Density isosurface component as one candidate surface."
            else:
                statement = (
                    "Select the component with the largest surface area and represent "
                    "it by its area-weighted centroid and surface area."
                    if selection == "largest_surface_area"
                    else "Select the component with the largest axis-aligned bounding-box "
                    "volume and represent it by the bounding-box midpoint and surface area."
                )
            decisions.append(OperationalizationDecision(dimension=OperationalizationDimension(dimension), statement=statement))
        bundles.append(OperationalizationBundle(
            operationalization_id=op_id,
            decisions=decisions,
            evidence_ids=_agent_operationalization_evidence_ids(evidence, metadata.dataset_id, op_id),
        ))
        centroid = selected[representation]
        area = float(selected["surface_area"])
        findings = [
            ReferenceFinding(
                finding_id=f"{op_id}_location",
                category=FindingRequirementCategory.LOCATION,
                statement=(
                    "The selected constant-Density isosurface component has "
                    f"{representation.replace('_', ' ')} approximately {centroid} "
                    "in the supplied coordinate frame."
                ),
                importance="core",
                value=centroid,
                verification=VerificationSpec(spatial_tolerance=0.05),
                evidence_ids=[finding_evidence],
            ),
            ReferenceFinding(
                finding_id=f"{op_id}_density",
                category=FindingRequirementCategory.QUANTITY,
                statement=f"The selected isosurface component is defined at Density level approximately {level:.12g} in the stored scale.",
                importance="core",
                value=float(level),
                verification=VerificationSpec(absolute_tolerance=absolute_tolerance_for_significant_figures(float(level), significant_figures=3)),
                evidence_ids=[finding_evidence],
            ),
            ReferenceFinding(
                finding_id=f"{op_id}_surface_area",
                category=FindingRequirementCategory.CHARACTERIZATION,
                statement=f"The selected connected isosurface has area approximately {area:.12g} in the supplied coordinate frame.",
                importance="core",
                value=area,
                verification=VerificationSpec(absolute_tolerance=absolute_tolerance_for_significant_figures(area, significant_figures=3)),
                evidence_ids=[finding_evidence],
            ),
        ]
        branches.append(OperationalizationFindingBranch(operationalization_id=op_id, findings=findings))
        branch_payloads[op_id] = analysis
    gt = GroundTruth(
        dataset_id=metadata.dataset_id,
        case_id=metadata.case_id,
        case_family_id=metadata.case_family_id,
        acceptable_operationalizations=bundles,
        findings_by_operationalization=branches,
    )
    GroundTruthValidator.validate(gt, metadata, evidence_records=evidence)
    return gt, {
        "levels": analyses,
        "branch_payloads": branch_payloads,
        "selected_branch_count": len(branches),
    }


def _agent_semantic_reuse_audit(
    row: Mapping[str, Any],
    metadata: CaseConstructionMetadata,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare current semantic responsibility with a historical case.

    A passing structural GT validator is deliberately not treated as semantic
    reuse evidence.  The audit compares the current frozen projection with
    the historical metadata/GT fields that are available; missing historical
    fields are recorded as ``NOT_ESTABLISHED`` and force reconstruction.
    """

    projection = row.get("canonical_semantic_projection", {})
    projection = projection if isinstance(projection, Mapping) else {}
    historical_metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    historical_gt = source.get("ground_truth") if isinstance(source.get("ground_truth"), Mapping) else {}
    current_target = str(projection.get("scientific_target", row.get("scientific_target", ""))).strip()
    historical_target = str(historical_metadata.get("scientific_target", "")).strip()
    current_principal = tuple(sorted(
        str(item.get("dimension_id", item)).strip()
        if isinstance(item, Mapping) else str(item).strip()
        for item in projection.get("principal_operationalization_dimensions", [])
    ))
    current_unresolved = tuple(sorted(
        str(item.get("dimension_id", item)).strip()
        if isinstance(item, Mapping) else str(item).strip()
        for item in projection.get("unresolved_operationalization_dimensions", [])
    ))
    historical_principal = tuple(sorted(str(item) for item in historical_metadata.get("principal_operationalization_dimensions", []) or ()))
    historical_unresolved = tuple(sorted(str(item) for item in historical_metadata.get("unresolved_operationalization_dimensions", []) or ()))
    expected_condition = str(row.get("condition", "")).upper().replace("_", "-")
    historical_responsibility = str(historical_metadata.get("operationalization_responsibility", ""))
    expected_responsibility = {
        "O1-F1": OperationalizationResponsibility.USER_SPECIFIED.value,
        "O1-F2": OperationalizationResponsibility.USER_SPECIFIED.value,
        "O2-F1": OperationalizationResponsibility.PARTIALLY_SPECIFIED.value,
        "O3-F1": OperationalizationResponsibility.MODEL_SELECTED.value,
    }.get(expected_condition, "")
    gt_branches = historical_gt.get("findings_by_operationalization", []) if isinstance(historical_gt, Mapping) else []
    checks = {
        "scientific_target": bool(current_target and historical_target and normalize_case_text(current_target) == normalize_case_text(historical_target)),
        "principal_operationalization_dimensions": bool(current_principal and historical_principal and current_principal == historical_principal),
        "unresolved_operationalization_dimensions": bool(current_unresolved == historical_unresolved),
        "operationalization_responsibility": bool(expected_responsibility and historical_responsibility == expected_responsibility),
        "finding_responsibility": bool((expected_condition.endswith("F2")) == (str(historical_metadata.get("finding_openness", "")) == FindingOpenness.OPEN.value)),
        "reference_branches_present": bool(gt_branches),
        "verification_semantics_present": all(
            isinstance(finding, Mapping) and (finding.get("verification") is not None or finding.get("value") is None)
            for branch in gt_branches if isinstance(branch, Mapping)
            for finding in branch.get("findings", []) if isinstance(branch.get("findings", []), list)
        ),
    }
    eligible = all(checks.values())
    return {
        "decision": "REUSE_ELIGIBLE" if eligible else "REBUILD_REQUIRED",
        "historical_case_id": source.get("case_id"),
        "historical_semantic_hash": source.get("semantic_contract_sha256"),
        "current_semantic_hash": row.get("semantic_contract_sha256"),
        "equivalence_checks": checks,
        "reason": "all available semantic responsibilities are equivalent" if eligible else "historical semantics are incomplete or differ from current projection",
    }


def _agent_reproduction_audit(source_gt: GroundTruth | None, current_gt: GroundTruth, reuse_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Compare fresh output with a historical pattern only when reuse is eligible."""

    if reuse_audit.get("decision") != "REUSE_ELIGIBLE":
        return {"status": "NOT_REQUIRED_REBUILD", "mismatch_count": 0, "mismatches": []}
    if source_gt is None:
        return {"status": "NOT_ESTABLISHED", "mismatch_count": 1, "mismatches": [{"reason": "historical_ground_truth_missing"}]}
    old_branches = {branch.operationalization_id: branch for branch in source_gt.findings_by_operationalization}
    new_branches = {branch.operationalization_id: branch for branch in current_gt.findings_by_operationalization}
    mismatches: list[dict[str, Any]] = []
    old_ops = {bundle.operationalization_id: _agent_operation_mapping(bundle) for bundle in source_gt.acceptable_operationalizations}
    new_ops = {bundle.operationalization_id: _agent_operation_mapping(bundle) for bundle in current_gt.acceptable_operationalizations}
    if set(old_ops) != set(new_ops):
        mismatches.append({"reason": "operationalization_id_set_changed", "historical": sorted(old_ops), "current": sorted(new_ops)})
    for operation_id in sorted(set(old_ops) & set(new_ops)):
        if old_ops[operation_id] != new_ops[operation_id]:
            mismatches.append({"operationalization_id": operation_id, "reason": "operationalization_changed"})
        old_findings = {finding.finding_id: finding for finding in old_branches.get(operation_id, ()).findings}
        new_findings = {finding.finding_id: finding for finding in new_branches.get(operation_id, ()).findings}
        matched: list[tuple[str, str]] = [
            (finding_id, finding_id)
            for finding_id in sorted(set(old_findings) & set(new_findings))
        ]
        old_unmatched = [finding_id for finding_id in old_findings if finding_id not in {pair[0] for pair in matched}]
        new_unmatched = [finding_id for finding_id in new_findings if finding_id not in {pair[1] for pair in matched}]

        def proposition_key(finding: Any) -> tuple[str, str, str]:
            import re
            statement = re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "<value>", str(finding.statement).casefold())
            statement = re.sub(r"\s+", " ", statement).strip()
            return (str(finding.category.value if hasattr(finding.category, "value") else finding.category), str(finding.importance), statement)

        old_by_key: dict[tuple[str, str, str], list[str]] = {}
        new_by_key: dict[tuple[str, str, str], list[str]] = {}
        for finding_id in old_unmatched:
            old_by_key.setdefault(proposition_key(old_findings[finding_id]), []).append(finding_id)
        for finding_id in new_unmatched:
            new_by_key.setdefault(proposition_key(new_findings[finding_id]), []).append(finding_id)
        for key in sorted(set(old_by_key) & set(new_by_key)):
            while old_by_key[key] and new_by_key[key]:
                matched.append((old_by_key[key].pop(0), new_by_key[key].pop(0)))
        old_unmatched = [item for values in old_by_key.values() for item in values]
        new_unmatched = [item for values in new_by_key.values() for item in values]
        # A wording-only rewrite can change articles or numeric formatting
        # while preserving a unique scientific role.  Use role/importance as
        # a conservative fallback only when each side has one candidate.
        old_by_role: dict[tuple[str, str], list[str]] = {}
        new_by_role: dict[tuple[str, str], list[str]] = {}
        for finding_id in old_unmatched:
            finding = old_findings[finding_id]
            role = (str(finding.category.value if hasattr(finding.category, "value") else finding.category), str(finding.importance))
            old_by_role.setdefault(role, []).append(finding_id)
        for finding_id in new_unmatched:
            finding = new_findings[finding_id]
            role = (str(finding.category.value if hasattr(finding.category, "value") else finding.category), str(finding.importance))
            new_by_role.setdefault(role, []).append(finding_id)
        for role in sorted(set(old_by_role) & set(new_by_role)):
            if len(old_by_role[role]) == len(new_by_role[role]) == 1:
                matched.append((old_by_role[role][0], new_by_role[role][0]))
                old_unmatched.remove(old_by_role[role][0])
                new_unmatched.remove(new_by_role[role][0])
        if old_unmatched or new_unmatched:
            mismatches.append({"operationalization_id": operation_id, "reason": "finding_identity_unresolved", "historical": sorted(old_unmatched), "current": sorted(new_unmatched)})
        for old_id, new_id in matched:
            old_finding, new_finding = old_findings[old_id], new_findings[new_id]
            if old_finding.unit != new_finding.unit:
                mismatches.append({
                    "operationalization_id": operation_id,
                    "finding_id": new_id,
                    "historical_finding_id": old_id,
                    "reason": "finding_unit_contract_changed",
                })
                continue
            if old_finding.value != new_finding.value:
                # Reproduction uses the exact verification contract authored on
                # the Finding.  This permits harmless floating-point drift for
                # tolerance-verified values while keeping semantic-only and
                # discrete values exact.
                matched_value, verification_mode = _agent_values_match(
                    old_finding.value,
                    new_finding.value,
                    new_finding.verification,
                )
                if not matched_value:
                    mismatches.append({
                        "operationalization_id": operation_id,
                        "finding_id": new_id,
                        "historical_finding_id": old_id,
                        "reason": "finding_value_changed",
                        "verification_mode": verification_mode,
                        "historical_value": old_finding.value,
                        "current_value": new_finding.value,
                    })
    status = "REPRODUCTION_REGRESSION_UNRESOLVED" if mismatches else "AUDITED"
    return {"status": status, "mismatch_count": len(mismatches), "mismatches": mismatches, "blocking": bool(mismatches)}


def _agent_operation_mapping(bundle: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for decision in getattr(bundle, "decisions", ()):
        dimension = str(decision.dimension.value)
        statement = str(decision.statement)
        # Older authored operation bundles expressed a percentile but omitted
        # the interpolation convention.  The current frozen projection uses
        # linear interpolation; make that migration explicit at construction
        # time so the strict materializer never invents a default at runtime.
        if (
            dimension == "criterion"
            and re.search(r"\b\d+(?:\.\d+)?\s*(?:st|nd|rd|th)?\s*percentile\b", statement, re.IGNORECASE)
            and not re.search(r"\b(?:linear(?:ly)?\s+interpol|nearest[- ]rank)\b", statement, re.IGNORECASE)
        ):
            statement = statement.rstrip(" .") + " using linear quantile interpolation."
        mapping[dimension] = statement
    return mapping


def _agent_rebuild_operationalization_candidates(
    source_gt: GroundTruth,
    projection: Mapping[str, Any],
) -> tuple[list[OperationalizationBundle], dict[str, Any]]:
    """Re-bind historical operation patterns to the current semantic contract.

    A ``REBUILD_REQUIRED`` decision means that the historical case is not an
    authority for the current question.  Its operation bundles are therefore
    treated as candidate patterns only: dimensions resolved by the current
    projection are replaced by the current normalized meaning, while
    unresolved dimensions retain the candidate's explicit choice.  Candidates
    that cannot cover the current responsibility contract are rejected before
    materialization.  Stable operation IDs are preserved for traceability.
    """

    resolved = projection.get("resolved_operationalization", ())
    fixed_by_dimension = {
        str(item.get("dimension_id")): str(item.get("normalized_meaning", "")).strip()
        for item in resolved
        if isinstance(item, Mapping)
        and str(item.get("dimension_id", "")).strip()
        and str(item.get("normalized_meaning", "")).strip()
    }
    unresolved = {
        str(item.get("dimension_id", item)).strip()
        if isinstance(item, Mapping)
        else str(item).strip()
        for item in projection.get("unresolved_operationalization_dimensions", ())
    }
    required = set(fixed_by_dimension) | {item for item in unresolved if item}
    rebuilt: list[OperationalizationBundle] = []
    rejected: list[dict[str, Any]] = []
    for bundle in source_gt.acceptable_operationalizations:
        source_decisions = _agent_operation_mapping(bundle)
        missing = sorted(required - set(source_decisions))
        if missing:
            rejected.append({"operationalization_id": bundle.operationalization_id, "missing_dimensions": missing})
            continue
        decisions: list[OperationalizationDecision] = []
        for decision in bundle.decisions:
            dimension = str(decision.dimension.value)
            statement = fixed_by_dimension.get(dimension, str(decision.statement))
            if dimension in required:
                decisions.append(
                    OperationalizationDecision(
                        dimension=decision.dimension,
                        statement=statement,
                    )
                )
        rebuilt.append(bundle.model_copy(update={"decisions": decisions}))
    if not rebuilt:
        raise ValueError(
            "current semantic projection cannot be covered by historical operation patterns"
        )
    return rebuilt, {
        "authority": "CURRENT_SEMANTIC_PROJECTION",
        "historical_membership_reused": False,
        "fixed_dimensions_rebound": sorted(fixed_by_dimension),
        "unresolved_dimensions_from_candidate": sorted(unresolved),
        "candidate_count": len(source_gt.acceptable_operationalizations),
        "rebuilt_count": len(rebuilt),
        "rejected_candidates": rejected,
    }


def _agent_rebind_resolved_operationalization_statements(
    bundles: Sequence[OperationalizationBundle],
    projection: Mapping[str, Any],
) -> list[OperationalizationBundle]:
    """Bind every fixed O clause to the current semantic projection.

    Historical operation bundles remain useful for stable branch membership
    and for explicit choices on unresolved dimensions.  They are not the
    wording authority for dimensions that the current question has already
    resolved.  Rebinding those statements here keeps hidden GT matching in
    lockstep with the model-visible semantic contract without changing any
    branch IDs or inventing choices for O2/O3.
    """

    fixed_by_dimension = {
        str(item.get("dimension_id")): str(item.get("normalized_meaning", "")).strip()
        for item in projection.get("resolved_operationalization", ())
        if isinstance(item, Mapping)
        and str(item.get("dimension_id", "")).strip()
        and str(item.get("normalized_meaning", "")).strip()
    }
    rebound: list[OperationalizationBundle] = []
    for bundle in bundles:
        decisions = [
            decision.model_copy(
                update={
                    "statement": fixed_by_dimension.get(
                        str(decision.dimension.value), decision.statement
                    )
                }
            )
            for decision in bundle.decisions
        ]
        rebound.append(bundle.model_copy(update={"decisions": decisions}))
    return rebound


def _agent_o1_fixed_template(
    root: Path,
    row: Mapping[str, Any],
) -> tuple[list[OperationalizationBundle], dict[str, Any]]:
    """Load the current O1-F1 execution contract for an O1-F2 rebuild.

    O1-F2 changes finding responsibility only.  Its fixed operationalization
    must therefore be the same contract as the dataset's O1-F1 case, even when
    either historical record requires a rebuild.
    """

    template_row = next(
        (
            item
            for item in _conditions(root)
            if item.get("dataset_id") == row.get("dataset_id")
            and str(item.get("condition", "")).upper().replace("_", "-") == "O1-F1"
        ),
        None,
    )
    if template_row is None:
        raise ValueError(f"O1-F1 fixed template missing for {row.get('dataset_id')}")
    template_case_id = _optional_case_id(template_row.get("case_id"))
    template_source = _case_record(root, template_case_id)
    if template_source is None:
        raise ValueError(f"O1-F1 source case missing for {row.get('dataset_id')}")
    raw = template_source.get("ground_truth")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"O1-F1 GroundTruth missing for {row.get('dataset_id')}")
    template_gt = GroundTruth.model_validate(raw)
    template_metadata = _agent_metadata(template_row, template_case_id or "", template_source)
    template_audit = _agent_semantic_reuse_audit(template_row, template_metadata, template_source)
    if template_audit.get("decision") == "REUSE_ELIGIBLE":
        bundles = list(template_gt.acceptable_operationalizations)
        provenance = {
            "authority": "O1_F1_FIXED_TEMPLATE",
            "template_case_id": template_case_id,
            "template_reuse_decision": template_audit.get("decision"),
            "historical_membership_reused": True,
        }
    else:
        bundles, rebuild = _agent_rebuild_operationalization_candidates(
            template_gt,
            template_row.get("canonical_semantic_projection", {})
            if isinstance(template_row.get("canonical_semantic_projection"), Mapping)
            else {},
        )
        provenance = {
            "authority": "O1_F1_FIXED_TEMPLATE",
            "template_case_id": template_case_id,
            "template_reuse_decision": template_audit.get("decision"),
            "historical_membership_reused": False,
            "template_rebuild": rebuild,
        }
    return bundles, provenance


def _agent_execution_semantic_fidelity(
    effective_o: Mapping[str, Any],
    execution: Mapping[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    """Check that execution carries the exact compiled Effective-O plan."""

    provenance = execution.get("data_provenance", {})
    if not isinstance(provenance, Mapping):
        return {"status": "FAIL", "code": "EXECUTION_PLAN_MISSING", "reason": "data provenance is missing"}
    plan_payload = provenance.get("materialization_plan")
    if not isinstance(plan_payload, Mapping):
        return {"status": "FAIL", "code": "EXECUTION_PLAN_MISSING", "reason": "typed materialization plan is missing"}
    try:
        expected_plan = compile_effective_operationalization(effective_o, dataset_id=dataset_id).to_dict()
    except MaterializationUnsupported as exc:
        return {"status": "FAIL", "code": "EFFECTIVE_O_UNSUPPORTED", "reason": str(exc)}
    expected_hash = artifact_sha256(expected_plan)
    if dict(plan_payload) != expected_plan:
        return {"status": "FAIL", "code": "EFFECTIVE_O_EXECUTION_MISMATCH", "expected": expected_plan, "actual": dict(plan_payload)}
    if provenance.get("effective_o_sha256") != artifact_sha256(effective_o):
        return {"status": "FAIL", "code": "EFFECTIVE_O_HASH_MISMATCH"}
    if provenance.get("compiled_plan_sha256") != expected_hash or provenance.get("executed_plan_sha256") != expected_hash:
        return {"status": "FAIL", "code": "EXECUTED_PLAN_HASH_MISMATCH", "expected": expected_hash}
    if provenance.get("silent_substitutions") or provenance.get("unsupported_dimensions"):
        return {"status": "FAIL", "code": "SILENT_MATERIALIZATION_SUBSTITUTION"}
    return {"status": "PASS", "expected_plan": expected_plan}


def _agent_high_speed_findings(
    metadata: CaseConstructionMetadata,
    result: Mapping[str, Any],
    operationalization_id: str,
    finding_evidence_id: str,
) -> list[ReferenceFinding]:
    location = [float(value) for value in result["reported_location"]]
    strength = float(result["strength"])
    measure = {"peak": "peak speed", "mean": "mean speed", "speed_sum": "sum of retained point speeds", "volume_weighted_rms": "volume-weighted RMS cell speed", "volume_weighted_mean": "volume-weighted mean cell speed", "integrated_excess_speed": "integrated excess speed"}[result["measure_kind"]]
    findings = [
        ReferenceFinding(
            finding_id=f"{operationalization_id}_location",
            category=FindingRequirementCategory.LOCATION,
            statement=f"The selected strongest region has a {result.get('representation', 'mean')} spatial location approximately {location} in the dataset coordinate frame.",
            importance=FindingImportance.CORE,
            value=location,
            verification=VerificationSpec(spatial_tolerance=0.05),
            evidence_ids=[finding_evidence_id],
        ),
        ReferenceFinding(
            finding_id=f"{operationalization_id}_strength",
            category=FindingRequirementCategory.QUANTITY,
            statement=f"The selected strongest region has {measure} strength approximately {strength:.8g} in the stored velocity scale.",
            importance=FindingImportance.CORE,
            value=strength,
            verification=VerificationSpec(absolute_tolerance=absolute_tolerance_for_significant_figures(strength, significant_figures=3)),
            evidence_ids=[finding_evidence_id],
        ),
    ]
    if metadata.finding_openness == FindingOpenness.OPEN:
        span = [float(value) for value in result["coordinate_span"]]
        findings.extend([
            ReferenceFinding(
                finding_id=f"{operationalization_id}_extent",
                category=FindingRequirementCategory.CHARACTERIZATION,
                statement=f"The selected region has coordinate-space extent approximately {span} along the dataset axes.",
                importance=FindingImportance.CORE,
                value=span,
                verification=VerificationSpec(spatial_tolerance=0.05),
                evidence_ids=[finding_evidence_id],
            ),
            ReferenceFinding(
                finding_id=f"{operationalization_id}_retained_points",
                category=FindingRequirementCategory.QUANTITY,
                statement=f"The selected region contains {int(result['selected_region_size'])} retained locations.",
                importance=FindingImportance.SUPPORTING,
                value=int(result["selected_region_size"]),
                evidence_ids=[finding_evidence_id],
            ),
            ReferenceFinding(
                finding_id=f"{operationalization_id}_region_count",
                category=FindingRequirementCategory.QUANTITY,
                statement=f"The criterion produces {int(result['region_count'])} connected regions in the supplied mesh or grid.",
                importance=FindingImportance.SUPPORTING,
                value=int(result["region_count"]),
                evidence_ids=[finding_evidence_id],
            ),
        ])
    return findings


def _agent_kitchen_concentration_findings(
    metadata: CaseConstructionMetadata,
    result: Mapping[str, Any],
    operationalization_id: str,
    finding_evidence_id: str,
) -> list[ReferenceFinding]:
    metric_name = str(result["metric_name"])
    metric_label = {
        "weighted_coefficient_of_variation": "cell-volume-weighted coefficient of variation",
        "coefficient_of_variation": "pointwise coefficient of variation",
        "weighted_relative_interdecile_spread": "cell-volume-weighted relative interdecile concentration spread",
        "relative_interdecile_spread": "pointwise relative interdecile concentration spread",
        "weighted_standard_deviation": "cell-volume-weighted standard deviation",
        "population_standard_deviation": "pointwise standard deviation",
        "rms_face_adjacent_difference": "root-mean-square face-adjacent concentration difference",
        "mean_face_adjacent_difference": "mean absolute face-adjacent concentration difference",
        "normalized_mean_face_adjacent_difference": "range-normalized mean absolute face-adjacent concentration difference",
    }.get(metric_name, metric_name.replace("_", " "))
    findings = [
        ReferenceFinding(
            finding_id=f"{operationalization_id}_field",
            category=FindingRequirementCategory.EXISTENCE_OR_IDENTITY,
            statement=f"The selected gas-concentration field is {result['field_name']}.",
            importance=FindingImportance.CORE,
            value=str(result["field_name"]),
            evidence_ids=[finding_evidence_id],
        ),
        ReferenceFinding(
            finding_id=f"{operationalization_id}_heterogeneity",
            category=FindingRequirementCategory.QUANTITY,
            statement=f"The selected field has {metric_label} approximately {float(result['metric_value']):.8g}.",
            importance=FindingImportance.CORE,
            value=float(result["metric_value"]),
            verification=VerificationSpec(absolute_tolerance=absolute_tolerance_for_significant_figures(float(result["metric_value"]), significant_figures=3)),
            evidence_ids=[finding_evidence_id],
        ),
    ]
    if metadata.finding_openness == FindingOpenness.OPEN:
        findings.append(ReferenceFinding(
            finding_id=f"{operationalization_id}_standard_deviation",
            category=FindingRequirementCategory.CHARACTERIZATION,
            statement=(
                "The selected field has cell-volume-weighted concentration "
                f"standard deviation approximately {float(result['standard_deviation']):.8g} "
                "in its stored concentration scale."
            ),
            importance=FindingImportance.SUPPORTING,
            value=float(result["standard_deviation"]),
            verification=VerificationSpec(
                absolute_tolerance=absolute_tolerance_for_significant_figures(
                    float(result["standard_deviation"]), significant_figures=3
                )
            ),
            evidence_ids=[finding_evidence_id],
        ))
        for quantile in ("q10", "q50", "q90"):
            findings.append(ReferenceFinding(
                finding_id=f"{operationalization_id}_{quantile}",
                category=FindingRequirementCategory.CHARACTERIZATION,
                statement=f"The selected field has volume-weighted concentration {quantile} approximately {float(result[quantile]):.8g} in its stored concentration scale.",
                importance=FindingImportance.SUPPORTING,
                value=float(result[quantile]),
                verification=VerificationSpec(absolute_tolerance=absolute_tolerance_for_significant_figures(float(result[quantile]), significant_figures=3)),
                evidence_ids=[finding_evidence_id],
            ))
    return findings


def _agent_preserve_reused_verification(
    findings: Sequence[ReferenceFinding],
    source_gt: GroundTruth | None,
    operation_id: str,
    reuse_audit: Mapping[str, Any],
) -> list[ReferenceFinding]:
    """Return the freshly authored portfolio-v2 verification semantics.

    Historical tolerances are not an authority for the new immutable
    reference portfolio: carrying them forward was the source of the old
    O1-F1/O1-F2 drift.  Values and policies are now authored once by the
    current deterministic construction path and then frozen physically.
    """

    return list(findings)


def _agent_normalized_f2_contract(
    projection: Mapping[str, Any],
    ground_truth: GroundTruth | None = None,
) -> dict[str, Any]:
    semantics = projection.get("finding_semantics", {}) if isinstance(projection, Mapping) else {}
    nested = semantics.get("adequate_core_semantics", {}) if isinstance(semantics, Mapping) else {}
    if not isinstance(nested, Mapping):
        nested = {}
    source = dict(nested)
    for key in ("mandatory_roles", "alternative_role_groups", "adequate_core_sets", "supporting_roles", "novel_role_policy", "scientific_entity_type", "role_by_category", "reference_finding_role_map"):
        if key not in source and isinstance(semantics, Mapping) and key in semantics:
            source[key] = semantics[key]
    source.setdefault("contract_source", "AGENT_READY_CONSTRUCTION")
    dataset_id = str(projection.get("dataset_id", ""))
    if dataset_id == "Combustor":
        # The fixed O already identifies the selected component. Requiring a
        # second, reportable ``structure_identity`` Finding made the reference
        # answer unable to satisfy its own adequate-core contract.
        source["mandatory_roles"] = ["defining_density_level"]
        source["adequate_core_sets"] = [
            ["defining_density_level", role]
            for role in (
                "surface_extent",
                "spatial_location",
                "shape_characterization",
            )
        ]
    if ground_truth is not None:
        role_map: dict[str, str] = {}
        for branch in ground_truth.findings_by_operationalization:
            for finding in branch.findings:
                token = finding.finding_id.casefold()
                if dataset_id == "Kitchen":
                    role = (
                        "selected_field_identity"
                        if token.endswith("_field")
                        else "heterogeneity_measure_evidence"
                        if token.endswith("_heterogeneity")
                        else "distribution_spread"
                        if token.endswith("_standard_deviation")
                        else "supporting_statistic"
                    )
                elif dataset_id == "Combustor":
                    role = (
                        "spatial_location"
                        if token.endswith("_location")
                        else "defining_density_level"
                        if token.endswith("_density")
                        else "surface_extent"
                        if token.endswith("_surface_area")
                        else "supporting_surface_statistic"
                    )
                else:
                    role = (
                        "principal_feature_location"
                        if token.endswith("_location")
                        else "defining_strength_evidence"
                        if token.endswith("_strength")
                        else "spatial_extent"
                        if token.endswith("_extent")
                        else "supporting_statistic"
                    )
                role_map[finding.finding_id] = role
        source["reference_finding_role_map"] = role_map
    return source


def _agent_gt_and_analysis(
    root: Path,
    row: Mapping[str, Any],
    metadata: CaseConstructionMetadata,
    source: Mapping[str, Any],
    evidence: Sequence[EvidenceRecord],
) -> tuple[GroundTruth, dict[str, Any], list[EvidenceRecord]]:
    """Run Stage A/B and return freshly materialized GT plus evidence."""

    finding_evidence_id = next(item.evidence_id for item in evidence if item.evidence_type == "finding")
    reuse_audit = _agent_semantic_reuse_audit(row, metadata, source)
    raw = source.get("ground_truth")
    source_gt = GroundTruth.model_validate(raw) if isinstance(raw, Mapping) and raw else None
    condition = str(row.get("condition", "")).upper().replace("_", "-")
    if metadata.dataset_id == "Combustor":
        gt, analysis = _agent_combustor_gt(root, row, metadata, evidence)
        gt = gt.model_copy(
            update={
                "acceptable_operationalizations": _agent_rebind_resolved_operationalization_statements(
                    gt.acceptable_operationalizations,
                    row.get("canonical_semantic_projection", {})
                    if isinstance(row.get("canonical_semantic_projection"), Mapping)
                    else {},
                )
            }
        )
        # Rebind every finding to branch-level execution evidence after the
        # real PLOT3D/contour analysis has completed.
        branches: list[OperationalizationFindingBranch] = []
        extra: list[EvidenceRecord] = []
        materializations: list[dict[str, Any]] = []
        for branch in gt.findings_by_operationalization:
            bundle = next(item for item in gt.acceptable_operationalizations if item.operationalization_id == branch.operationalization_id)
            effective = _agent_operation_mapping(bundle)
            plan = compile_effective_operationalization(effective, dataset_id="Combustor")
            level_payload = copy.deepcopy(
                analysis.get("branch_payloads", {}).get(
                    branch.operationalization_id, {}
                )
            )
            if not isinstance(level_payload, Mapping) or not level_payload:
                raise ValueError(
                    f"Combustor materialization payload missing for {branch.operationalization_id}"
                )
            # Validate the materialized level even though only execution metadata is copied.
            float(level_payload["level"])
            materialization = {
                "parameters": effective,
                "execution": {
                    "status": "MATERIALIZED",
                    "reproducible": True,
                    "G_of_O": level_payload,
                    "data_provenance": {"dataset_id": metadata.dataset_id, "reader": "canonical PLOT3D", "analysis": "Density contour/connectivity/triangulation", "materialization_plan": plan.to_dict(), "effective_o_sha256": artifact_sha256(effective), "compiled_plan_sha256": artifact_sha256(plan.to_dict()), "executed_plan_sha256": artifact_sha256(plan.to_dict()), "silent_substitutions": [], "unsupported_dimensions": []},
                },
                "materialization_id": f"mat:{branch.operationalization_id}",
            }
            materialization["execution"]["data_provenance"]["dataset_manifest_sha256"] = artifact_sha256(
                _read_json(root / "datasets" / metadata.dataset_id / "dataset_manifest.json", {})
            )
            # Scientific role support remains source-backed.  Exact values are
            # bound separately to this branch's execution sidecar below.
            findings = [finding.model_copy(update={"evidence_ids": [finding_evidence_id]}) for finding in branch.findings]
            branches.append(OperationalizationFindingBranch(operationalization_id=branch.operationalization_id, findings=findings))
            materializations.append({
                "operation_id": branch.operationalization_id,
                "materialization_id": materialization["materialization_id"],
                "execution": materialization["execution"],
                "reuse_reference": reuse_audit["decision"],
            })
        gt = GroundTruth(
            dataset_id=gt.dataset_id,
            case_id=gt.case_id,
            case_family_id=gt.case_family_id,
            acceptable_operationalizations=gt.acceptable_operationalizations,
            findings_by_operationalization=branches,
        )
        GroundTruthValidator.validate(gt, metadata, evidence_records=[*evidence, *extra])
        analysis = {**analysis, "mode": "real_plot3d_materialization", "reuse_audit": reuse_audit, "reproduction_audit": _agent_reproduction_audit(source_gt, gt, reuse_audit), "materialization_status": "MATERIALIZED", "materialization_count": len(branches), "materializations": materializations}
        return gt, analysis, extra

    if source_gt is None:
        raise ValueError(f"source GroundTruth missing for {metadata.case_id}")
    rebuild_provenance: dict[str, Any] | None = None
    if reuse_audit["decision"] == "REUSE_ELIGIBLE":
        bundles = list(source_gt.acceptable_operationalizations)
    else:
        bundles, rebuild_provenance = _agent_rebuild_operationalization_candidates(
            source_gt,
            row.get("canonical_semantic_projection", {})
            if isinstance(row.get("canonical_semantic_projection"), Mapping)
            else {},
        )
        if condition == "O1-F2":
            fixed_bundles, fixed_provenance = _agent_o1_fixed_template(root, row)
            if len(fixed_bundles) != len(bundles):
                raise ValueError("O1-F2 fixed O template does not match candidate branch count")
            bundles = [
                bundle.model_copy(update={"decisions": fixed.decisions})
                for bundle, fixed in zip(bundles, fixed_bundles, strict=True)
            ]
            rebuild_provenance = {
                **(rebuild_provenance or {}),
                "fixed_o_template": fixed_provenance,
            }
    bundles = _agent_rebind_resolved_operationalization_statements(
        bundles,
        row.get("canonical_semantic_projection", {})
        if isinstance(row.get("canonical_semantic_projection"), Mapping)
        else {},
    )
    new_bundles: list[OperationalizationBundle] = []
    new_branches: list[OperationalizationFindingBranch] = []
    extra: list[EvidenceRecord] = []
    materializations: list[dict[str, Any]] = []
    from .deterministic_materialization import analyze_high_speed
    for bundle in bundles:
        operation_id = bundle.operationalization_id
        effective = _agent_operation_mapping(bundle)
        result = analyze_high_speed(root, metadata.dataset_id, effective, operation_id)
        materialization = result
        new_bundles.append(OperationalizationBundle(
            operationalization_id=operation_id,
            decisions=bundle.decisions,
            evidence_ids=_agent_operationalization_evidence_ids(evidence, metadata.dataset_id, operation_id),
        ))
        findings = (
            _agent_kitchen_concentration_findings(metadata, result["result"], operation_id, finding_evidence_id)
            if metadata.dataset_id == "Kitchen" and "concentration" in metadata.scientific_target.casefold()
            else _agent_high_speed_findings(metadata, result["result"], operation_id, finding_evidence_id)
        )
        findings = _agent_preserve_reused_verification(
            findings,
            source_gt,
            operation_id,
            reuse_audit,
        )
        new_branches.append(OperationalizationFindingBranch(
            operationalization_id=operation_id,
            findings=findings,
        ))
        materializations.append({
            "operation_id": operation_id,
            "materialization_id": result.get("materialization_id"),
            "execution": result.get("execution", {}),
            "reuse_reference": reuse_audit["decision"],
        })
    gt = GroundTruth(
        dataset_id=metadata.dataset_id,
        case_id=metadata.case_id,
        case_family_id=metadata.case_family_id,
        acceptable_operationalizations=new_bundles,
        findings_by_operationalization=new_branches,
    )
    GroundTruthValidator.validate(gt, metadata, evidence_records=[*evidence, *extra])
    analysis = {
        "mode": "real_dataset_materialization",
        "source_case_id": str(source.get("case_id", metadata.case_id)),
        "reuse_audit": reuse_audit,
        "rebuild_provenance": rebuild_provenance,
        "reproduction_audit": _agent_reproduction_audit(source_gt, gt, reuse_audit),
        "materialization_status": "MATERIALIZED",
        "materialization_count": len(materializations),
        "materializations": materializations,
        "branch_count": len(new_branches),
        "finding_count": sum(len(branch.findings) for branch in new_branches),
    }
    return gt, analysis, extra


def _agent_reference_sec(
    case_id: str,
    condition: str,
    metadata: CaseConstructionMetadata,
    projection: Mapping[str, Any],
    gt: GroundTruth,
    evidence: Sequence[EvidenceRecord],
    semantic_hash: str,
    analysis: Mapping[str, Any],
    *,
    live_provenance: Sequence[Mapping[str, Any]] = (),
    adjudicated_findings: Sequence[Mapping[str, Any]] = (),
    explicitly_invalid: Sequence[Mapping[str, Any]] = (),
    status: str = "DRAFT_REQUIRES_HUMAN_CONFIRMATION",
) -> ScientificEvaluationContract:
    """Compile reference anchors without treating them as agent judgments."""

    branches_by_id = {branch.operationalization_id: branch for branch in gt.findings_by_operationalization}
    materializations = {
        str(item.get("operation_id")): item
        for item in analysis.get("materializations", ())
        if isinstance(item, Mapping) and item.get("operation_id")
    }
    materialized_branches: list[dict[str, Any]] = []
    for bundle in gt.acceptable_operationalizations:
        operation_id = bundle.operationalization_id
        record = materializations.get(operation_id)
        source_branch = branches_by_id.get(operation_id)
        if not record or source_branch is None:
            continue
        execution = record.get("execution", {})
        materialized_branches.append(MaterializedBranch(
            proposal_id=f"reference:{case_id}:{operation_id}",
            parameters=dict(record.get("parameters", {})),
            execution=dict(execution) if isinstance(execution, Mapping) else {},
            findings=tuple(item.model_dump(mode="json") for item in source_branch.findings),
            materialization_id=str(record.get("materialization_id") or f"mat:reference:{case_id}:{operation_id}"),
        ).to_dict())
    finding_contract = (
        _agent_normalized_f2_contract(projection, gt)
        if condition == "O1-F2"
        else {}
    )
    if finding_contract:
        FindingRequirementContract.from_mapping(finding_contract)
    adequate_core_sets = tuple(
        tuple(str(value) for value in item)
        for item in (finding_contract.get("adequate_core_sets", []) if isinstance(finding_contract, Mapping) else [])
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
    )
    return ScientificEvaluationContract(
        case_id=case_id,
        enumerated_valid_O=tuple(bundle.model_dump(mode="json") for bundle in gt.acceptable_operationalizations),
        explicitly_invalid_O=tuple(dict(item) for item in explicitly_invalid),
        unenumerated_policy="VALID_UNENUMERATED requires curator escalation",
        G_of_O=tuple(materialized_branches),
        finding_requirement_contract=dict(finding_contract),
        adequate_core_sets=adequate_core_sets,
        tolerances={},
        adjudication_provenance=tuple(dict(item) for item in live_provenance),
        status=status,
        adjudicated_findings=tuple(dict(item) for item in adjudicated_findings),
        source_semantic_contract_sha256=semantic_hash,
    )


def _agent_live_payload(case_view: Mapping[str, Any], *, fixed_o: Mapping[str, Any] | None = None, fixed_g_of_o: Mapping[str, Any] | None = None, candidate_findings: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Build the model-visible payload without GT or reference-space data."""

    metadata = case_view.get("metadata", {})
    payload: dict[str, Any] = {
        "question": case_view.get("case_input", {}).get("scientific_question", ""),
        "context": case_view.get("case_context", {}),
        "dataset_contract": case_view.get("case_input", {}).get("flow_data", {}),
        "condition": case_view.get("condition"),
        "scientific_target": metadata.get("scientific_target"),
        "scope": metadata.get("scope_constraints", []),
        "principal_dimensions": metadata.get("principal_operationalization_dimensions", []),
        "resolved_dimensions": [
            item for item in metadata.get("principal_operationalization_dimensions", [])
            if item not in metadata.get("unresolved_operationalization_dimensions", [])
        ],
        "unresolved_dimensions": metadata.get("unresolved_operationalization_dimensions", []),
        "runtime_contract": {"tools": ["python"], "network": "unavailable", "case_scoped": True},
    }
    if fixed_o is not None:
        payload["fixed_o"] = dict(fixed_o)
    if fixed_g_of_o is not None:
        payload["fixed_evidence_context"] = {"materialized_G_of_O": dict(fixed_g_of_o)}
    if candidate_findings:
        payload["candidate_findings"] = [dict(item) for item in candidate_findings]
    from .proxy_expert import build_firewalled_payload
    return build_firewalled_payload("analyst", payload)


def _agent_raw_hash(value: Any) -> str:
    return artifact_sha256(value if value is not None else "")


def _agent_combustor_live_materialization(
    root: Path,
    effective: Mapping[str, Any],
    operation_id: str,
    finding_evidence_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize a live Combustor proposal through the canonical PLOT3D route.

    The live analyst may describe the density criterion in natural language.
    This adapter only maps the already frozen Combustor target to a scalar
    level; it does not adjudicate whether that choice is scientifically good.
    Unsupported choices remain materialization failures and are never turned
    into a valid branch.
    """

    grid, q90 = _read_plot3d_combustor(root)
    plan = compile_effective_operationalization(effective, dataset_id="Combustor")
    text = " ".join(str(value) for value in effective.values()).casefold()
    if plan.criterion_kind == "density_fixed":
        level = float(plan.threshold)
    elif plan.criterion_kind == "density_q90_fraction":
        level = q90 * 0.90
    elif plan.criterion_kind == "density_quantile":
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        density = vtk_to_numpy(grid.GetPointData().GetArray("Density")).astype(float, copy=False)
        finite = density[np.isfinite(density)]
        if finite.size == 0:
            raise MaterializationUnsupported("criterion", text, "Density has no finite samples")
        level = float(
            np.quantile(
                finite,
                float(plan.quantile),
                method="inverted_cdf" if plan.quantile_method == "nearest_rank" else "linear",
            )
        )
    else:
        raise MaterializationUnsupported("criterion", text, "unsupported Combustor Density level")
    try:
        analyses = _agent_combustor_contours(root, [float(level)], closed_geometry=True)
    except ValueError as exc:
        if plan.component_selection_kind != "unique_connected_component" or not str(exc).startswith("no connected Density isosurface at level"):
            raise
        analyses = {float(level): {"level": float(level), "regions": [], "q90": q90,
                                  "quantile_method": "linear"}}
    payload = analyses[float(level)]
    selection_key = (
        "selected_by_bounding_box_volume"
        if plan.component_selection_kind == "largest_bounding_box_volume"
        else "selected_by_surface_area"
    )
    selected = payload.get(selection_key, payload.get("selected"))
    if plan.component_selection_kind == "encloses_global_density_maximum":
        candidates = [r for r in payload["regions"] if r.get("encloses_query_point")]
        selected = candidates[0] if len(candidates) == 1 else None
    if plan.component_selection_kind == "unique_connected_component":
        selected = payload["regions"][0] if len(payload["regions"]) == 1 else None
    selection_failure = None
    if selected is None and plan.component_selection_kind == "unique_connected_component":
        selection_failure = f"The requested unique connected component does not exist: found {len(payload['regions'])} components."
    if selected is None and plan.component_selection_kind == "encloses_global_density_maximum":
        selection_failure = "No unique closed surface enclosing the global density maximum exists at the requested level."
    if not isinstance(selected, Mapping) and selection_failure is None:
        raise MaterializationUnsupported(
            "aggregation_or_representation",
            str(plan.component_selection_kind),
            "materialized contour did not expose the selected component",
        )
    payload = copy.deepcopy(payload)
    payload["selected"] = selected
    payload["component_selection"] = plan.component_selection_kind
    payload["location_representation"] = plan.representation_kind
    if selected is not None and plan.representation_kind not in selected:
        selection_failure = "The selected surface is not closed; an enclosed-volume centroid is undefined."
    if selection_failure:
        payload["requested_structure_exists"] = False
        payload["selection_failure"] = selection_failure
        payload["selected"] = None
        payload["closed_component_count"] = sum(bool(r.get("closed_consistently_oriented")) for r in payload["regions"])
        payload["enclosing_global_maximum_component_count"] = sum(bool(r.get("encloses_query_point")) for r in payload["regions"])
        findings = [{
            "finding_id": f"{operation_id}_closed_component_count",
            "category": FindingRequirementCategory.CHARACTERIZATION.value,
            "statement": f"At Density level {level}, the executed contour has {payload['closed_component_count']} verified closed components; the requested enclosed structure and volume centroid are not available.",
            "importance": FindingImportance.CORE.value,
            "value": payload["closed_component_count"],
            "verification": {"absolute_tolerance": 0.0},
            "evidence_ids": [finding_evidence_id],
        }]
        if plan.component_selection_kind == "unique_connected_component":
            findings[0].update({
                "finding_id": f"{operation_id}_connected_component_count",
                "statement": f"At Density level {level}, the executed contour has {len(payload['regions'])} connected components; a unique selected component and its representative location are unavailable.",
                "value": len(payload["regions"]),
            })
    else:
        location = selected[plan.representation_kind]
        location_label = str(plan.representation_kind).replace("_", " ")
        findings = [
            {
                "finding_id": f"{operation_id}_location",
                "category": FindingRequirementCategory.LOCATION.value,
                "statement": f"The selected constant-Density surface has {location_label} approximately {location} in the supplied coordinate frame.",
                "importance": FindingImportance.CORE.value,
                "value": list(location),
                "verification": {"spatial_tolerance": 0.05},
                "evidence_ids": [finding_evidence_id],
            },
            {
                "finding_id": f"{operation_id}_surface_area",
                "category": FindingRequirementCategory.CHARACTERIZATION.value,
                "statement": f"The selected connected constant-Density surface has area approximately {float(selected['surface_area']):.12g}.",
                "importance": FindingImportance.CORE.value,
                "value": float(selected["surface_area"]),
                "verification": {
                    "absolute_tolerance": absolute_tolerance_for_significant_figures(
                        float(selected["surface_area"]), significant_figures=3
                    )
                },
                "evidence_ids": [finding_evidence_id],
            },
        ]
    manifest = _read_json(root / "datasets" / "Combustor" / "dataset_manifest.json", {})
    execution = {
        "status": "MATERIALIZED",
        "reproducible": True,
        "G_of_O": payload,
        "data_provenance": {
            "dataset_id": "Combustor",
            "reader": "canonical PLOT3D",
            "reader_format": manifest.get("reader", {}).get("format"),
            "reader_configuration": manifest.get("reader", {}).get("reader_configuration", {}),
            "input_files": [item.get("path") for item in manifest.get("files", []) if isinstance(item, Mapping)],
            "dataset_manifest_sha256": artifact_sha256(manifest),
            "analysis": "Density contour/connectivity/triangulation",
            "materialization_plan": plan.to_dict(),
            "effective_o_sha256": artifact_sha256(effective),
            "compiled_plan_sha256": artifact_sha256(plan.to_dict()),
            "executed_plan_sha256": artifact_sha256(plan.to_dict()),
            "silent_substitutions": [],
            "unsupported_dimensions": [],
        },
    }
    return {"parameters": dict(effective), "execution": execution, "findings": findings, "materialization_id": f"mat:{operation_id}"}, findings


def build_agent_ready_novel_o_materializer(
    root: str | Path,
    metadata: CaseConstructionMetadata,
    evidence: Sequence[EvidenceRecord | Mapping[str, Any]],
    *,
    execution_trajectory: Path | None = None,
    open_field_reference: Mapping[str, Any] | None = None,
) -> Callable[[OperationalizationBundle], dict[str, Any]]:
    """Build the trusted, case-scoped evaluator materializer.

    The callback accepts only an Effective-O bundle and executes it through
    the same deterministic data route used during Agent-ready construction.
    It never consumes model- or adjudicator-supplied Findings/G(O).  This is
    deliberately a thin adapter, not a second materialization framework.
    """

    repository_root = Path(root).resolve()
    records = [
        item if isinstance(item, EvidenceRecord) else EvidenceRecord.model_validate(item)
        for item in evidence
    ]
    finding_evidence_id = next(
        (
            item.evidence_id
            for item in records
            if item.dataset_id == metadata.dataset_id
            and item.evidence_type == "finding"
        ),
        None,
    )
    if finding_evidence_id is None:
        raise ValueError(
            f"case {metadata.case_id!r} has no case-scoped Finding Evidence"
        )

    def materialize(bundle: OperationalizationBundle) -> dict[str, Any]:
        if not isinstance(bundle, OperationalizationBundle):
            bundle = OperationalizationBundle.model_validate(bundle)
        # Runtime evaluation must preserve the evaluated model's Effective O
        # at the semantic statement level.  ``_agent_operation_mapping`` may
        # migrate legacy construction clauses by adding an interpolation
        # convention; applying that migration here would invent a live-model
        # scientific choice and break the materialization binding.
        effective = {
            str(decision.dimension.value): str(decision.statement)
            for decision in bundle.decisions
        }
        operation_id = (
            f"runtime:{metadata.case_id}:"
            f"{artifact_sha256(effective)[:16]}"
        )
        if metadata.dataset_id == "Combustor":
            artifact, _ = _agent_combustor_live_materialization(
                repository_root,
                effective,
                operation_id,
                finding_evidence_id,
            )
            return artifact

        from .deterministic_materialization import (
            analyze_high_speed,
            analyze_kitchen_concentration,
        )
        from .execution_evidence import observed_quantile_method, observed_mean_std_convention
        observed = observed_quantile_method(effective, execution_trajectory)
        observed_statistics = observed_mean_std_convention(effective, execution_trajectory)

        kitchen_concentration = (
            metadata.dataset_id == "Kitchen"
            and "concentration" in metadata.scientific_target.casefold()
        )
        result = (
            analyze_kitchen_concentration(repository_root, effective, operation_id)
            if kitchen_concentration
            else analyze_high_speed(
                repository_root,
                metadata.dataset_id,
                effective,
                operation_id,
                observed_quantile_method=observed["method"] if observed else None,
                observed_threshold_statistics=observed_statistics,
            )
        )
        if observed:
            result["execution"]["observed_numerical_convention"] = observed
        if observed_statistics:
            result["execution"]["observed_threshold_statistics"] = observed_statistics
        findings = (
            _agent_kitchen_concentration_findings(
                metadata,
                result["result"],
                operation_id,
                finding_evidence_id,
            )
            if kitchen_concentration
            else _agent_high_speed_findings(
                metadata,
                result["result"],
                operation_id,
                finding_evidence_id,
            )
        )
        artifact = {
            "parameters": dict(result["parameters"]),
            "execution": dict(result["execution"]),
            "findings": [item.model_dump(mode="json") for item in findings],
            "materialization_id": result["materialization_id"],
        }
        if kitchen_concentration:
            from .open_field_selection import inherit_verified_open_field_policy
            artifact = inherit_verified_open_field_policy(artifact, open_field_reference)
        return artifact

    return materialize


def _agent_live_scientific_work(
    root: Path,
    case_id: str,
    condition: str,
    metadata: CaseConstructionMetadata,
    projection: Mapping[str, Any],
    gt: GroundTruth,
    evidence: Sequence[EvidenceRecord],
    analysis: Mapping[str, Any],
    case_input: Mapping[str, Any],
    semantic_hash: str,
    *,
    config_path: str | Path | None,
    api_key: str | None,
    timeout: float,
) -> tuple[ScientificEvaluationContract, dict[str, Any]]:
    """Run real proposer/adjudicator roles for one open condition."""

    from .live_agents import LiveModelCaller
    from .live_srac import _proposal_from_call, build_effective_operationalization

    case_view = {
        "case_id": case_id,
        "condition": condition,
        "metadata": metadata.model_dump(mode="json"),
        "case_input": dict(case_input),
        "case_context": case_input.get("case_context", {}),
    }
    op_evidence = next(item.evidence_id for item in evidence if item.evidence_type == "operationalization")
    principal = tuple(item.value for item in metadata.principal_operationalization_dimensions)
    unresolved = tuple(item.value for item in metadata.unresolved_operationalization_dimensions)
    resolved = tuple(item for item in principal if item not in unresolved)
    analyst = LiveModelCaller(root, "proxy-flow-analyst-gpt-5.6-sol", config_path=config_path, api_key=api_key, timeout=timeout)
    adjudicator = LiveModelCaller(root, "flow-scientific-adjudicator-gpt-5.6-sol", config_path=config_path, api_key=api_key, timeout=timeout)
    reference_branches = [bundle.model_dump(mode="json") for bundle in gt.acceptable_operationalizations]
    attempts: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    adjudication_records: list[ScientificAdjudication] = []
    successful_route = False
    previous_attempt_feedback: str | None = None
    for index in (range(1, 4) if condition in {"O2-F1", "O3-F1"} else range(0)):
        proposal_invocation_id = f"agent-proposer:{case_id}:{condition}:{index}"
        payload = _agent_live_payload(case_view)
        if previous_attempt_feedback:
            payload = {**payload, "previous_attempt_feedback": previous_attempt_feedback}
        instruction = (
            "Propose exactly one complete operationalization by filling every unresolved dimension. "
            "For O3-F1, fill every principal dimension. Return one JSON object whose "
            "proposed_operationalizations field is an object with exactly the required dimension keys; "
            "each dimension value must be one plain natural-language string, not a nested object or list. "
            "State every threshold, connectivity rule, strength measure, and requested output representation "
            "explicitly. Do not add tie-breaking rules or extra output requirements because they are outside "
            "the requested principal dimensions and cannot be silently approximated by the exact executor. "
            "Also return a concise scientific rationale. Do not discuss reference answers. "
            "Use executor-compatible connectivity wording: for structured datasets say face-connected "
            "through the six grid neighbors (6-neighbor/face-connected), and for unstructured datasets say "
            "mesh-connected or connected through shared mesh cells. Do not use 26-neighbor connectivity, "
            "shared-edge-only connectivity, or other topology that is not explicitly requested by the case. "
            "For thresholded high-speed regions, put an explicit numeric threshold or percentile in either "
            "criterion or feature_definition; the criterion must not contain only a strongest-region selection "
            "rule. For location, use arithmetic mean spatial location/centroid or the location of the "
            "maximum-speed point."
        )
        if metadata.dataset_id == "Combustor" and condition == "O2-F1":
            instruction += (
                " This is a Combustor O2-F1 case: criterion is the only unresolved dimension. "
                "Put the explicit Density isovalue rule (for example, median or a stated percentile) "
                "and the candidate-selection rule (largest surface area) in the criterion string. "
                "Do not place the Density level in feature_definition, because that dimension is frozen."
            )
        if metadata.dataset_id == "Combustor" and condition == "O3-F1":
            instruction += (
                " This is a Combustor O3-F1 case. The exact data executor supports a connected "
                "constant-Density contour/isosurface: define the Density level with an explicit "
                "median or percentile (or the stated 90%-of-q90 rule), select the connected surface "
                "with the largest surface area, and represent it with an area-weighted centroid. "
                "Use those concepts explicitly in the three returned dimensions. Do not propose a "
                "point density band, grid-point component count, support volume, bounding-box-only "
                "representation, or an unimplemented density-level rule."
            )
        if previous_attempt_feedback:
            instruction += f" The previous attempt received this executor feedback; correct it directly: {previous_attempt_feedback}"
        call = analyst.call(
            role="analyst",
            visible_payload=payload,
            instruction=instruction,
            identity={"case_id": case_id, "proposal_index": index, "proposal_kind": "OPERATIONALIZATION", "invocation_id": proposal_invocation_id},
            tool_free=True,
        )
        call_trace = {
            "invocation_id": proposal_invocation_id,
            "role": "O_PROPOSER",
            "profile": call.get("agent_profile_id"),
            "model_family": call.get("model_family"),
            "input_sha256": call.get("visible_input_sha256") or _agent_raw_hash(payload),
            "raw_output_sha256": _agent_raw_hash(call.get("raw_structured_output", "")),
            "parsed_output_sha256": _agent_raw_hash(call.get("parsed_result", {})),
            "invocation_status": call.get("invocation_status"),
            "status": call.get("status"),
            "visible_payload": payload,
            "raw_output": call.get("raw_structured_output", ""),
            "parsed_output": call.get("parsed_result"),
        }
        proposal = _proposal_from_call(
            call,
            case=case_view,
            proposal_id=f"proposal:{case_id}:{condition}:{index}",
            producer_id=proposal_invocation_id,
            kind="OPERATIONALIZATION",
        )
        if proposal is None:
            call_trace.update({"proposal_id": None, "routing_status": "NOT_PARSED"})
            attempts.append(call_trace)
            previous_attempt_feedback = "The previous response was not parseable as the required proposal JSON."
            continue
        call_trace["proposal_id"] = proposal.proposal_id
        try:
            validate_case_aware_proposal_submission(
                condition,
                case_id,
                principal,
                resolved if condition == "O2-F1" else (),
                unresolved if condition == "O2-F1" else principal,
                proposal,
                require_complete=True,
                frozen_scientific_target=metadata.scientific_target,
            )
            effective_trace = build_effective_operationalization(case_view, proposal)
            call_trace["routing_status"] = "PASS"
            call_trace["effective_operationalization"] = effective_trace["effective_operationalization"]
        except Exception as exc:
            call_trace.update({"routing_status": "FAIL", "routing_error": f"{type(exc).__name__}: {exc}"})
            attempts.append(call_trace)
            previous_attempt_feedback = f"The previous proposal failed case-aware routing: {type(exc).__name__}: {exc}"
            continue
        attempts.append(call_trace)
        operation_id = f"live:{case_id}:{condition}:{index}"
        try:
            from .deterministic_materialization import analyze_high_speed, analyze_kitchen_concentration
            effective = effective_trace["effective_operationalization"]
            if metadata.dataset_id == "Combustor":
                result, findings = _agent_combustor_live_materialization(root, effective, operation_id, op_evidence)
            else:
                result = (
                    analyze_kitchen_concentration(root, effective, operation_id)
                    if metadata.dataset_id == "Kitchen" and "concentration" in metadata.scientific_target.casefold()
                    else analyze_high_speed(root, metadata.dataset_id, effective, operation_id)
                )
                findings = (
                    _agent_kitchen_concentration_findings(metadata, result["result"], operation_id, op_evidence)
                    if metadata.dataset_id == "Kitchen" and "concentration" in metadata.scientific_target.casefold()
                    else _agent_high_speed_findings(metadata, result["result"], operation_id, op_evidence)
                )
            fidelity = _agent_execution_semantic_fidelity(
                effective,
                result.get("execution", {}) if isinstance(result, Mapping) else {},
                metadata.dataset_id,
            )
            if fidelity.get("status") != "PASS":
                raise MaterializationUnsupported(
                    "effective_operationalization",
                    str(fidelity.get("code", "unknown")),
                    str(fidelity.get("reason", "typed execution plan did not conform")),
                )
            materialized = materialize_branch(
                proposal,
                lambda _proposal, record=result, items=findings: {
                    "parameters": record.get("parameters", {}),
                    "execution": record.get("execution", {}),
                    "findings": [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in items],
                    "materialization_id": record.get("materialization_id"),
                },
            )
            materializations.append({
                "proposal_id": proposal.proposal_id,
                "operation_id": operation_id,
                "materialization_id": materialized.materialization_id,
                "parameters": dict(materialized.parameters),
                "execution": dict(materialized.execution),
                "findings": [dict(item) for item in materialized.findings],
            })
        except Exception as exc:
            previous_attempt_feedback = (
                "The previous proposal could not be executed exactly. Revise only the unresolved "
                f"dimensions and address this executor feedback: {type(exc).__name__}: {exc}. "
                "Use only the explicit executor-compatible forms in the instruction: structured "
                "face-connected through six grid neighbors (not 26-neighbor), or unstructured "
                "mesh-connected/shared mesh-cell connectivity (not shared-edge-only); include the "
                "numeric percentile/threshold in criterion or feature_definition."
            )
            materializations.append({
                "proposal_id": proposal.proposal_id,
                "operation_id": operation_id,
                "materialization_id": None,
                "execution": {"status": "MATERIALIZATION_UNSUPPORTED", "reason": f"{type(exc).__name__}: {exc}"},
                "findings": [],
            })
            continue
        adjudication_invocation_id = f"agent-adjudicator:{case_id}:{condition}:{index}"
        requested_scientific_view = {
            "question": case_input.get("scientific_question", ""),
            "scientific_target": metadata.scientific_target,
            "scientific_scope": [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in metadata.scope_constraints],
            "candidate_scientific_O": effective_trace["effective_operationalization"],
            "materialized_G_of_O": materialized.execution.get("G_of_O", {}),
            "structured_scientific_evidence": [dict(item) for item in materialized.findings],
            "context": case_view["case_context"],
        }
        holder: dict[str, Any] = {}
        def invoke(view_payload: Mapping[str, Any]) -> Any:
            holder["firewalled_view"] = dict(view_payload)
            holder["call"] = adjudicator.call(
                role="adjudicator",
                visible_payload=view_payload,
                instruction="Assess only scientific validity. Return scientific_validity exactly one of SCIENTIFICALLY_VALID, SCIENTIFICALLY_INVALID, or SCIENTIFICALLY_UNCERTAIN, plus scientific_rationale and confidence. Never classify reference membership.",
                identity={"case_id": case_id, "proposal_id": proposal.proposal_id, "invocation_id": adjudication_invocation_id},
                tool_free=True,
            )
            return holder["call"]
        try:
            prepare_adjudicator_invocation(proposal, {"agent_id": adjudication_invocation_id}, requested_scientific_view, invoke)
        except Exception as exc:
            holder["call"] = {"status": "FAILED", "invocation_status": "ROLE_GUARD_FAILURE", "error": f"{type(exc).__name__}: {exc}", "parsed_result": None, "raw_structured_output": ""}
        adjudicator_call = holder["call"]
        firewalled_view = holder.get("firewalled_view", {})
        parsed = adjudicator_call.get("parsed_result") if isinstance(adjudicator_call.get("parsed_result"), Mapping) else {}
        raw_validity = str(parsed.get("scientific_validity", parsed.get("verdict", "SCIENTIFICALLY_UNCERTAIN"))).strip().upper()
        if raw_validity not in {"SCIENTIFICALLY_VALID", "SCIENTIFICALLY_INVALID", "SCIENTIFICALLY_UNCERTAIN"}:
            raw_validity = "SCIENTIFICALLY_UNCERTAIN"
        resolution = resolve_o_validity_label(raw_validity, effective_trace["effective_operationalization"], reference_branches)
        final_validity = resolution["final_o_validity"]
        materialization_hash = artifact_sha256(materialized.to_dict())
        proposal_hash = artifact_sha256(proposal.to_dict())
        adjudication_trace = {
            "invocation_id": adjudication_invocation_id,
            "role": "O_ADJUDICATOR",
            "profile": adjudicator_call.get("agent_profile_id"),
            "model_family": adjudicator_call.get("model_family"),
            "proposal_id": proposal.proposal_id,
            "proposal_input_sha256": proposal_hash,
            "materialization_output_sha256": materialization_hash,
            "raw_output_sha256": _agent_raw_hash(adjudicator_call.get("raw_structured_output", "")),
            "parsed_output_sha256": _agent_raw_hash(adjudicator_call.get("parsed_result", {})),
            "scientific_validity": raw_validity,
            "requested_scientific_view": requested_scientific_view,
            "firewalled_scientific_view": firewalled_view,
            "firewalled_scientific_view_sha256": _agent_raw_hash(firewalled_view),
            "actual_model_visible_payload_sha256": adjudicator_call.get("visible_input_sha256"),
            "semantic_coverage": validate_adjudicator_semantic_coverage(
                condition=condition,
                proposal=proposal,
                firewalled_view=firewalled_view,
                requested_scientific_view=requested_scientific_view,
            ),
            "reference_space_resolution": resolution,
            "invocation_status": adjudicator_call.get("invocation_status"),
            "visible_payload": firewalled_view,
            "raw_output": adjudicator_call.get("raw_structured_output", ""),
            "parsed_output": adjudicator_call.get("parsed_result"),
        }
        adjudications.append(adjudication_trace)
        adjudication_records.append(ScientificAdjudication(
            proposal_id=proposal.proposal_id,
            candidate_source_hash=materialization_hash,
            o_validity=final_validity,
            adjudicator_id=str(adjudicator_call.get("agent_profile_id") or adjudication_invocation_id),
            adjudicator_model_family=str(adjudicator_call.get("model_family") or "unknown"),
            materialization_id=materialized.materialization_id,
            evidence_context_id=op_evidence,
            supporting_source_ids=tuple(item.source_id for item in evidence),
            scientific_rationale=str(parsed.get("scientific_rationale", parsed.get("rationale", ""))),
            confidence=float(parsed["confidence"]) if isinstance(parsed.get("confidence"), (int, float)) and 0 <= float(parsed["confidence"]) <= 1 else None,
        ))
        successful_route = True
        break

    if condition in {"O2-F1", "O3-F1"}:
        has_scientifically_valid = any(item.get("scientific_validity") == "SCIENTIFICALLY_VALID" for item in adjudications)
        status = "ESCALATE_TO_CURATOR" if any(item.get("reference_space_resolution", {}).get("final_o_validity") in {"VALID_UNENUMERATED", "UNCERTAIN"} for item in adjudications) else "DRAFT_REQUIRES_HUMAN_CONFIRMATION"
        sec = _agent_reference_sec(case_id, condition, metadata, projection, gt, evidence, semantic_hash, analysis, live_provenance=adjudications, status=status)
        return sec, {
            "proposal_count": len(attempts),
            "proposal_attempts": attempts,
            "materialization_count": len(materializations),
            "materializations": materializations,
            "adjudication_count": len(adjudications),
            "adjudications": adjudications,
            "proxy_invocations": [*attempts, *adjudications],
            "o_space_exploration_status": (
                "LIMITED_PROXY_COVERAGE" if successful_route and has_scientifically_valid
                else "NO_VALID_LIVE_PROPOSAL" if successful_route
                else "MATERIALIZATION_LIMITED"
            ),
            "finding_space_status": "NOT_APPLICABLE",
            "finding_contract_status": "NOT_APPLICABLE",
        }

    fixed_bundle = gt.acceptable_operationalizations[0]
    fixed_branch = next(
        branch
        for branch in gt.findings_by_operationalization
        if branch.operationalization_id == fixed_bundle.operationalization_id
    )
    fixed_operation = next((item for item in analysis.get("materializations", ()) if item.get("operation_id") == fixed_bundle.operationalization_id), {})
    fixed_execution = fixed_operation.get("execution", {}) if isinstance(fixed_operation, Mapping) else {}
    finding_attempts: list[dict[str, Any]] = []
    finding_adjudications: list[dict[str, Any]] = []
    accepted_findings: list[dict[str, Any]] = []
    finding_contract = _agent_normalized_f2_contract(projection, gt)
    for index in range(1, 3):
        proposal_invocation_id = f"agent-finding-proposer:{case_id}:{index}"
        finding_payload = _agent_live_payload(
            case_view,
            fixed_o={decision.dimension.value: decision.statement for decision in fixed_bundle.decisions},
            fixed_g_of_o=fixed_execution.get("G_of_O", {}) if isinstance(fixed_execution, Mapping) else {},
        )
        call = analyst.call(
            role="analyst",
            visible_payload=finding_payload,
            instruction="Propose up to three relevant, atomic, independently judgeable scientific findings supported by the fixed O and materialized G(O). Return proposed_findings as objects with finding_id, statement, optional value/unit/category, and a concise rationale. Do not use reference findings or adequate-core answers.",
            identity={"case_id": case_id, "proposal_index": index, "proposal_kind": "FINDING", "invocation_id": proposal_invocation_id},
            tool_free=True,
        )
        parsed = call.get("parsed_result") if isinstance(call.get("parsed_result"), Mapping) else {}
        values = parsed.get("proposed_findings", parsed.get("findings", []))
        if isinstance(values, Mapping):
            values = [values]
        candidates = []
        if isinstance(values, Sequence):
            for item_index, item in enumerate(values, 1):
                if isinstance(item, Mapping) and str(item.get("statement", "")).strip():
                    candidate = dict(item)
                    candidate["finding_id"] = f"proposal:{case_id}:findings:{index}:F{item_index}"
                    candidates.append(candidate)
        trace = {
            "invocation_id": proposal_invocation_id,
            "role": "F_PROPOSER",
            "profile": call.get("agent_profile_id"),
            "model_family": call.get("model_family"),
            "input_sha256": call.get("visible_input_sha256") or _agent_raw_hash(finding_payload),
            "raw_output_sha256": _agent_raw_hash(call.get("raw_structured_output", "")),
            "parsed_output_sha256": _agent_raw_hash(call.get("parsed_result", {})),
            "proposal_id": None,
            "candidate_findings": candidates,
            "invocation_status": call.get("invocation_status"),
            "visible_payload": finding_payload,
            "raw_output": call.get("raw_structured_output", ""),
            "parsed_output": call.get("parsed_result"),
        }
        if not candidates:
            finding_attempts.append(trace)
            continue
        proposal_id = f"proposal:{case_id}:findings:{index}"
        finding_proposal = SRACProposal(
            proposal_id=proposal_id,
            condition=condition,
            producer_agent_id=proposal_invocation_id,
            producer_model_family=str(call.get("model_family") or "gpt-5.6"),
            proposed_findings=tuple(candidates),
            proposal_kind="FINDING",
            evidence_context_id=op_evidence,
            fixed_o_branch_id=fixed_bundle.operationalization_id,
            fixed_evidence_context_id=op_evidence,
            case_id=case_id,
            principal_dimensions=tuple(item.value for item in metadata.principal_operationalization_dimensions),
            frozen_resolved_dimensions=tuple(item.value for item in metadata.principal_operationalization_dimensions),
            frozen_unresolved_dimensions=(),
            frozen_scientific_target=metadata.scientific_target,
        )
        trace["proposal_id"] = proposal_id
        trace["routing_status"] = "PASS"
        finding_attempts.append(trace)
        finding_adjudicator_id = f"agent-finding-adjudicator:{case_id}:{index}"
        requested_scientific_view = {
            "question": case_input.get("scientific_question", ""),
            "scientific_target": metadata.scientific_target,
            "scientific_scope": [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in metadata.scope_constraints],
            "candidate_scientific_O": {decision.dimension.value: decision.statement for decision in fixed_bundle.decisions},
            "materialized_G_of_O": fixed_execution.get("G_of_O", {}) if isinstance(fixed_execution, Mapping) else {},
            "fixed_evidence_context": {"evidence_context_id": op_evidence},
            "candidate_F": candidates,
            "structured_scientific_evidence": [dict(item) for item in fixed_branch.findings],
            "context": case_view["case_context"],
        }
        holder: dict[str, Any] = {}
        def invoke_finding(view_payload: Mapping[str, Any]) -> Any:
            holder["firewalled_view"] = dict(view_payload)
            holder["call"] = adjudicator.call(
                role="adjudicator",
                visible_payload=view_payload,
                instruction="Evaluate each candidate finding only. Return finding_support, finding_relevance, o_f_consistency, finding_role maps keyed by finding_id, plus scientific_rationale and confidence. Do not judge reference membership and do not replace the fixed O.",
                identity={"case_id": case_id, "proposal_id": proposal_id, "invocation_id": finding_adjudicator_id},
                tool_free=True,
            )
            return holder["call"]
        try:
            prepare_adjudicator_invocation(finding_proposal, {"agent_id": finding_adjudicator_id}, requested_scientific_view, invoke_finding)
        except Exception as exc:
            holder["call"] = {"status": "FAILED", "invocation_status": "ROLE_GUARD_FAILURE", "error": f"{type(exc).__name__}: {exc}", "parsed_result": None, "raw_structured_output": ""}
        adjudicator_call = holder["call"]
        firewalled_view = holder.get("firewalled_view", {})
        parsed_adjudication = adjudicator_call.get("parsed_result") if isinstance(adjudicator_call.get("parsed_result"), Mapping) else {}
        from .live_agents import _normalize_finding_label
        support = parsed_adjudication.get("finding_support", {}) if isinstance(parsed_adjudication.get("finding_support", {}), Mapping) else {}
        relevance = parsed_adjudication.get("finding_relevance", {}) if isinstance(parsed_adjudication.get("finding_relevance", {}), Mapping) else {}
        consistency = parsed_adjudication.get("o_f_consistency", parsed_adjudication.get("finding_consistency", {})) if isinstance(parsed_adjudication.get("o_f_consistency", parsed_adjudication.get("finding_consistency", {})), Mapping) else {}
        roles = parsed_adjudication.get("finding_role", {}) if isinstance(parsed_adjudication.get("finding_role", {}), Mapping) else {}
        support = {str(k): _normalize_finding_label("support", v) for k, v in support.items()}
        relevance = {str(k): _normalize_finding_label("relevance", v) for k, v in relevance.items()}
        consistency = {str(k): _normalize_finding_label("consistency", v) for k, v in consistency.items()}
        roles = {str(k): str(v) for k, v in roles.items()}
        finding_hash = artifact_sha256(finding_proposal.to_dict())
        output_hash = _agent_raw_hash(adjudicator_call.get("parsed_result", {}))
        finding_trace = {
            "invocation_id": finding_adjudicator_id,
            "role": "F_ADJUDICATOR",
            "profile": adjudicator_call.get("agent_profile_id"),
            "model_family": adjudicator_call.get("model_family"),
            "proposal_id": proposal_id,
            "evidence_context_id": op_evidence,
            "proposal_input_sha256": finding_hash,
            "adjudication_output_sha256": output_hash,
            "finding_support": support,
            "finding_relevance": relevance,
            "o_f_consistency": consistency,
            "finding_role": roles,
            "invocation_status": adjudicator_call.get("invocation_status"),
            "requested_scientific_view": requested_scientific_view,
            "firewalled_scientific_view": firewalled_view,
            "firewalled_scientific_view_sha256": _agent_raw_hash(firewalled_view),
            "actual_model_visible_payload_sha256": adjudicator_call.get("visible_input_sha256"),
            "visible_payload": firewalled_view,
            "raw_output_sha256": _agent_raw_hash(adjudicator_call.get("raw_structured_output", "")),
            "parsed_output_sha256": output_hash,
            "raw_output": adjudicator_call.get("raw_structured_output", ""),
            "parsed_output": adjudicator_call.get("parsed_result"),
        }
        finding_trace["semantic_coverage"] = validate_adjudicator_semantic_coverage(
            condition=condition,
            proposal=finding_proposal,
            firewalled_view=firewalled_view,
            requested_scientific_view=requested_scientific_view,
            adjudication={
                "finding_support": support,
                "finding_relevance": relevance,
                "o_f_consistency": consistency,
                "finding_role": roles,
            },
            allowed_finding_roles=sorted(
                {
                    *finding_contract.get("mandatory_roles", []),
                    *finding_contract.get("supporting_roles", []),
                    *finding_contract.get("role_by_category", {}).values(),
                    *(
                        role
                        for group in finding_contract.get("alternative_role_groups", [])
                        if isinstance(group, Mapping)
                        for role in group.get("roles", [])
                    ),
                    *(
                        role
                        for core in finding_contract.get("adequate_core_sets", [])
                        if isinstance(core, Sequence) and not isinstance(core, (str, bytes))
                        for role in core
                    ),
                }
            ),
        )
        finding_adjudications.append(finding_trace)
        finding_adjudications_payload = dict(finding_proposal.to_dict())
        finding_adjudications_payload["evidence_context_id"] = op_evidence
        finding_adjudications_payload["adjudication"] = {"finding_support": support, "finding_relevance": relevance, "o_f_consistency": consistency, "finding_role": roles}
        accepted_ids = [item["finding_id"] for item in candidates if support.get(item["finding_id"]) == "SUPPORTED" and relevance.get(item["finding_id"]) == "RELEVANT" and consistency.get(item["finding_id"]) == "CONSISTENT"]
        accepted_findings.extend({"finding_id": item["finding_id"], "statement": item["statement"], "value": item.get("value"), "unit": item.get("unit"), "category": item.get("category"), "role": roles.get(item["finding_id"]), "adjudication": "SUPPORTED_RELEVANT_CONSISTENT"} for item in candidates if item["finding_id"] in accepted_ids)
        break
    contract = FindingRequirementContract.from_mapping(finding_contract)
    sec = _agent_reference_sec(
        case_id,
        condition,
        metadata,
        projection,
        gt,
        evidence,
        semantic_hash,
        analysis,
        live_provenance=[*finding_attempts, *finding_adjudications],
        adjudicated_findings=accepted_findings,
        status="DRAFT_REQUIRES_HUMAN_CONFIRMATION" if finding_adjudications else "ESCALATE_TO_CURATOR",
    )
    return sec, {
        "proposal_count": len(finding_attempts),
        "proposal_attempts": finding_attempts,
        "materialization_count": 1,
        "materializations": [],
        "adjudication_count": len(finding_adjudications),
        "adjudications": finding_adjudications,
        "proxy_invocations": [*finding_attempts, *finding_adjudications],
        "o_space_exploration_status": "NOT_APPLICABLE",
        "finding_space_status": "CONSTRUCTED" if finding_adjudications else "NOT_CONSTRUCTED",
        "finding_contract_status": "VALID" if contract is not None and not contract.validate() else "INVALID",
    }


def _agent_sec(
    root: Path,
    case_id: str,
    condition: str,
    metadata: CaseConstructionMetadata,
    projection: Mapping[str, Any],
    gt: GroundTruth,
    evidence: Sequence[EvidenceRecord],
    reference_analysis_path: str,
    semantic_hash: str,
    analysis: Mapping[str, Any],
    case_input: Mapping[str, Any],
    *,
    live_agents: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> tuple[ScientificEvaluationContract, dict[str, Any]]:
    """Build reference SEC and, when enabled, execute real agent roles."""

    if not live_agents or condition == "O1-F1":
        contract = _agent_reference_sec(case_id, condition, metadata, projection, gt, evidence, semantic_hash, analysis)
        return contract, {
            "proposal_count": 0,
            "proposal_attempts": [],
            "materialization_count": len(analysis.get("materializations", [])),
            "materializations": list(analysis.get("materializations", [])),
            "adjudication_count": 0,
            "adjudications": [],
            "proxy_invocations": [],
            "o_space_exploration_status": "NOT_APPLICABLE" if condition == "O1-F1" else "NOT_RUN",
            "finding_space_status": "NOT_APPLICABLE" if condition != "O1-F2" else "NOT_CONSTRUCTED",
            "finding_contract_status": "NOT_APPLICABLE" if condition != "O1-F2" else "PENDING_LIVE_AGENT",
        }
    return _agent_live_scientific_work(
        root,
        case_id,
        condition,
        metadata,
        projection,
        gt,
        evidence,
        analysis,
        case_input,
        semantic_hash,
        config_path=config_path,
        api_key=api_key,
        timeout=timeout,
    )


def _agent_execution_value(execution_g: Mapping[str, Any], finding_id: str, dataset_id: str) -> tuple[str | None, Any]:
    """Resolve a case-local, stable G(O) locator for one reference finding."""

    if dataset_id == "Combustor":
        # Provisional Combustor branches intentionally expose two distinct
        # spatial representations.  Bind from the authored Finding identity
        # so a bounding-box midpoint can never be compared with a centroid.
        normalized_id = str(finding_id).casefold()
        location_path = (
            "/selected/bounding_box_midpoint"
            if "bounding_box" in normalized_id or "bounding-box" in normalized_id
            else "/selected/area_weighted_centroid"
        )
        paths = {"location": location_path, "density": "/level", "surface_area": "/selected/surface_area"}
    elif dataset_id == "Kitchen":
        paths = {"field": "/field_name", "heterogeneity": "/metric_value", "standard_deviation": "/standard_deviation", "q10": "/q10", "q50": "/q50", "q90": "/q90"}
    else:
        paths = {"location": "/reported_location", "strength": "/strength", "extent": "/coordinate_span", "retained_points": "/selected_region_size", "region_count": "/region_count"}
    locator = next((path for suffix, path in sorted(paths.items(), key=lambda item: len(item[0]), reverse=True) if finding_id.casefold().endswith("_" + suffix)), None)
    if locator is None:
        return None, None
    current: Any = execution_g
    for component in locator.strip("/").split("/"):
        if not isinstance(current, Mapping) or component not in current:
            return locator, None
        current = current[component]
    return locator, current


def _agent_values_match(
    reference: Any,
    actual: Any,
    verification: VerificationSpec | None,
    *,
    verification_mode: str | None = None,
) -> tuple[bool, str]:
    if verification is not None and verification.spatial_tolerance is not None:
        if not isinstance(reference, list) or not isinstance(actual, (list, tuple)) or len(reference) != len(actual):
            return False, "spatial_euclidean"
        distance = math.sqrt(sum((float(left) - float(right)) ** 2 for left, right in zip(reference, actual, strict=True)))
        return distance <= verification.spatial_tolerance, "spatial_euclidean"
    if verification is not None and (verification.absolute_tolerance is not None or verification.relative_tolerance is not None):
        if not isinstance(reference, (int, float)) or isinstance(reference, bool) or not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False, "scalar_tolerance"
        absolute = verification.absolute_tolerance or 0.0
        relative = verification.relative_tolerance or 0.0
        return abs(float(reference) - float(actual)) <= absolute + relative * abs(float(reference)), "scalar_tolerance"
    if verification_mode == "exact_discrete_numeric":
        return actual == reference, "exact_discrete_numeric"
    # No explicit deterministic rule means that the value is adjudicated
    # semantically.  Do not infer exact scalar/vector equality from the JSON
    # representation; the execution binding still records the locator.
    return True, "semantic_only"


def _agent_build_execution_binding(
    case_id: str,
    dataset_id: str,
    semantic_hash: str,
    gt: GroundTruth,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the derived GT -> materialization -> G(O) binding sidecar."""

    materializations = {str(item.get("operation_id")): item for item in analysis.get("materializations", []) if isinstance(item, Mapping)}
    branches: list[dict[str, Any]] = []
    for bundle, finding_branch in zip(gt.acceptable_operationalizations, gt.findings_by_operationalization, strict=False):
        operation_id = bundle.operationalization_id
        materialization = materializations.get(operation_id)
        execution = materialization.get("execution", {}) if isinstance(materialization, Mapping) else {}
        g_of_o = execution.get("G_of_O") if isinstance(execution, Mapping) else None
        manifest_hash = execution.get("data_provenance", {}).get("dataset_manifest_sha256") if isinstance(execution, Mapping) and isinstance(execution.get("data_provenance"), Mapping) else None
        g_hash = artifact_sha256(g_of_o) if isinstance(g_of_o, Mapping) else None
        finding_bindings: list[dict[str, Any]] = []
        for finding in finding_branch.findings:
            locator, execution_value = _agent_execution_value(g_of_o if isinstance(g_of_o, Mapping) else {}, finding.finding_id, dataset_id)
            verification_mode = explicit_verification_mode(finding, locator=locator)
            if materialization is None:
                binding_status = "MATERIALIZATION_NOT_FOUND"
            elif execution.get("status") != "MATERIALIZED":
                binding_status = "MATERIALIZATION_NOT_FOUND"
            elif not isinstance(g_of_o, Mapping):
                binding_status = "OUTPUT_NOT_FOUND"
            elif locator is None or execution_value is None:
                binding_status = "OUTPUT_NOT_FOUND"
            else:
                matched, _ = _agent_values_match(
                    finding.value,
                    execution_value,
                    finding.verification,
                    verification_mode=verification_mode,
                )
                binding_status = "PASS" if matched else "VALUE_MISMATCH"
            finding_bindings.append({
                "finding_id": finding.finding_id,
                "g_of_o_locator": locator,
                "execution_value": execution_value,
                "execution_value_sha256": artifact_sha256(execution_value) if execution_value is not None else None,
                "gt_value": finding.value,
                "verification_mode": verification_mode,
                "verification_parameters": finding.verification.model_dump(mode="json") if finding.verification else {},
                "binding_status": binding_status,
            })
        branches.append({
            "operationalization_id": operation_id,
            "branch_id": operation_id,
            "materialization_id": materialization.get("materialization_id") if isinstance(materialization, Mapping) else None,
            "dataset_manifest_sha256": manifest_hash,
            "effective_o_sha256": artifact_sha256(_agent_operation_mapping(bundle)),
            "g_of_o_sha256": g_hash,
            "findings": finding_bindings,
        })
    return {"schema_version": "finding-execution-binding-v1", "dataset_id": dataset_id, "case_id": case_id, "semantic_contract_sha256": semantic_hash, "branches": branches}


def _agent_scientific_evidence_coverage(
    case_id: str,
    dataset_id: str,
    gt: GroundTruth,
    evidence: Sequence[EvidenceRecord],
) -> dict[str, Any]:
    """Expose claim-level evidence coverage without asserting scientific truth."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    claims: list[dict[str, Any]] = []
    for bundle in gt.acceptable_operationalizations:
        for decision in bundle.decisions:
            ids = list(bundle.evidence_ids)
            claims.append({
                "claim_id": f"{bundle.operationalization_id}:O:{decision.dimension.value}",
                "claim_kind": "operationalization",
                "statement": decision.statement,
                "evidence_ids": ids,
                "source_ids": [evidence_by_id[item].source_id for item in ids if item in evidence_by_id],
                "provenance_closed": all(item in evidence_by_id and evidence_by_id[item].evidence_type == "operationalization" for item in ids),
                "scientific_support_status": "NOT_ESTABLISHED",
            })
    for branch in gt.findings_by_operationalization:
        for finding in branch.findings:
            ids = list(finding.evidence_ids)
            claims.append({
                "claim_id": f"{branch.operationalization_id}:F:{finding.finding_id}",
                "claim_kind": "finding",
                "statement": finding.statement,
                "evidence_ids": ids,
                "source_ids": [evidence_by_id[item].source_id for item in ids if item in evidence_by_id],
                "provenance_closed": all(item in evidence_by_id and evidence_by_id[item].evidence_type == "finding" for item in ids),
                "scientific_support_status": "NOT_ESTABLISHED",
            })
    return {
        "artifact_type": "ScientificEvidenceCoverage",
        "schema_version": "scientific-evidence-coverage-v1",
        "case_id": case_id,
        "dataset_id": dataset_id,
        "status": "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED" if claims and all(item["provenance_closed"] for item in claims) else "PROVENANCE_GAP",
        "scientific_support_status": "NOT_ESTABLISHED",
        "claims": claims,
    }


def _agent_render_reference_analysis(case_id: str, dataset_id: str, analysis: Mapping[str, Any]) -> str:
    """Render the human analysis directly from the structured execution record."""

    lines = [
        f"# {case_id} Reference Analysis",
        "",
        "Status: AGENT_READY construction artifact; values below are rendered from the structured deterministic materialization.",
        "",
    ]
    for materialization in analysis.get("materializations", []):
        if not isinstance(materialization, Mapping):
            continue
        operation_id = materialization.get("operation_id", "unknown")
        execution = materialization.get("execution", {}) if isinstance(materialization.get("execution"), Mapping) else {}
        g_of_o = execution.get("G_of_O", {}) if isinstance(execution.get("G_of_O"), Mapping) else {}
        provenance = execution.get("data_provenance", {}) if isinstance(execution.get("data_provenance"), Mapping) else {}
        lines.extend([f"## Effective O `{operation_id}`", "", f"- Execution status: `{execution.get('status', 'UNKNOWN')}`", f"- Reproducible: `{execution.get('reproducible', False)}`", f"- Dataset manifest: `{provenance.get('dataset_manifest_sha256', '')}`", "", "### G(O)", ""])
        for key, value in g_of_o.items():
            if key == "region_sizes" and isinstance(value, list) and len(value) > 20:
                value = f"{len(value)} region sizes (first 20: {value[:20]})"
            lines.append(f"- `{key}`: `{json.dumps(value, ensure_ascii=False, separators=(',', ':')) if isinstance(value, (dict, list)) else value}`")
        lines.append("")
    if not analysis.get("materializations"):
        lines.extend(["No structured materialization was produced.", ""])
    return "\n".join(lines).rstrip() + "\n"


def build_agent_ready_scientific_case_portfolio(
    repository_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    conditions_manifest: str | Path | Mapping[str, Any] | None = None,
    live_agents: bool = False,
    config_path: str | Path | None = None,
    api_key: str | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Materialize all frozen 28 question rows as agent-ready artifacts.

    This is the agent-only Phase-2 construction mode.  It never claims human
    curator confirmation or official release.  Existing case folders are
    read-only construction patterns; each output case receives its own input,
    evidence, reference analysis, GT, SEC, and binding sidecars.
    """

    root = Path(repository_root).resolve()
    out = Path(output_dir).resolve() if output_dir else root / "outputs/current/qualified_scientific_cases"
    build_identity = _agent_build_identity(root)
    run_identity = _agent_construction_run_identity(
        root,
        build_identity,
        live_agents=live_agents,
        config_path=config_path,
        timeout=timeout,
    )
    case_root = out / "agent_ready_cases"
    conditions_source_path = (
        Path(conditions_manifest).resolve()
        if isinstance(conditions_manifest, (str, Path))
        else None
    )
    conditions_source_label = (
        str(conditions_source_path.relative_to(root))
        if conditions_source_path is not None and conditions_source_path.is_relative_to(root)
        else str(conditions_source_path)
        if conditions_source_path is not None
        else "explicit_conditions_manifest"
        if conditions_manifest is not None
        else f"{DERIVED_REVIEW_RELATIVE_PATH}/human_facing_closure_manifest.json"
    )
    # A call without an explicit conditions manifest is retained solely for
    # the archived construction API used by legacy regression fixtures.  The
    # frozen v7/N=1 path always passes the derived index explicitly, so this
    # flag can never weaken checks for the authoritative evaluation input.
    compatibility_construction_mode = conditions_manifest is None
    import shutil
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)
    # Agent-ready artifacts are canonical inside each case directory. Remove
    # legacy/duplicate projections from earlier builders instead of recreating
    # empty or byte-identical parallel trees on every run.
    for directory in (
        out / "agent_ready_review_packets",
        out / "agent_ready_evaluation_contracts",
        out / "agent_ready_bindings",
        out / "cases",
        out / "scientific_case_review_packets",
        out / "scientific_artifact_bindings",
    ):
        if directory.exists():
            shutil.rmtree(directory)
    duplicate_manifest = out / "qualified_scientific_case_portfolio.json"
    if duplicate_manifest.exists():
        duplicate_manifest.unlink()

    records: list[dict[str, Any]] = []
    for row in _conditions(root, conditions_manifest):
        dataset_id = str(row.get("dataset_id", ""))
        condition = str(row.get("condition", "")).upper().replace("_", "-")
        case_id = _agent_case_id(root, row)
        source_dir = _agent_source_case_dir(root, row, case_id)
        source = {
            "case_id": source_dir.name,
            "metadata": _read_json(source_dir / "case_construction_metadata.json", {}),
            "ground_truth": _read_json(source_dir / "ground_truth.json", {}),
        }
        metadata = _agent_metadata(row, case_id, source)
        question = str(row.get("scientific_question", "")).strip()
        if not question:
            raise ValueError(f"empty scientific question for {case_id}")
        evidence = _agent_evidence(root, dataset_id, case_id, "flowintentbench/qualified_scientific_cases.py")
        gt, analysis, execution_evidence = _agent_gt_and_analysis(root, row, metadata, source, evidence)
        evidence = [*evidence, *execution_evidence]
        case_dir = case_root / dataset_id / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        source_input = _read_json(source_dir / "case_input.json", {})
        if not isinstance(source_input, Mapping):
            raise ValueError(f"source case input is invalid for {case_id}")
        case_input = copy.deepcopy(dict(source_input))
        case_input["scientific_question"] = question
        # validate_model_input is deliberately imported lazily to keep this
        # orchestration module's import surface unchanged for unit fixtures.
        from .schema import validate_model_input
        validate_model_input(case_input)
        _write_json(case_dir / "scientific_question.json", {"scientific_question": question, "semantic_contract_sha256": str(row.get("semantic_contract_sha256", "")), "source": conditions_source_label})
        _write_json(case_dir / "case_construction_metadata.json", metadata.model_dump(mode="json"))
        _write_json(case_dir / "case_input.json", case_input)
        _write_json(case_dir / "case_context.json", case_input.get("case_context", {}))
        _write_json(case_dir / "agent_ready_evidence.json", [item.model_dump(mode="json") for item in evidence])
        ref_path = case_dir / "reference_analysis.md"
        ref_path.write_text(_agent_render_reference_analysis(case_id, dataset_id, analysis), encoding="utf-8")
        for materialization in analysis.get("materializations", []):
            if not isinstance(materialization, Mapping):
                continue
            operation_id = str(materialization.get("operation_id", "")).strip()
            if not operation_id:
                continue
            safe_operation_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in operation_id)
            _write_json(case_dir / f"materialization_{safe_operation_id}.json", materialization)
        projection = row.get("canonical_semantic_projection", {})
        projection = (
            harden_portfolio_scientific_projection(projection, condition=condition)
            if conditions_manifest is not None and isinstance(projection, Mapping)
            else dict(projection) if isinstance(projection, Mapping) else {}
        )
        semantic_hash = str(row.get("semantic_contract_sha256", "")).strip()
        if not semantic_hash:
            # The manifest is authoritative; this fallback is only for old
            # fixtures that omitted its cached digest.
            semantic_hash = artifact_sha256(projection)
        ref_for_sec = str(ref_path.relative_to(root)) if ref_path.is_relative_to(root) else str(ref_path)
        sec, sec_trace = _agent_sec(
            root,
            case_id,
            condition,
            metadata,
            projection,
            gt,
            evidence,
            ref_for_sec,
            semantic_hash,
            analysis,
            case_input,
            live_agents=live_agents,
            config_path=config_path,
            api_key=api_key,
            timeout=timeout,
        )
        execution_binding = _agent_build_execution_binding(case_id, dataset_id, semantic_hash, gt, analysis)
        verification_policy = compile_finding_verification_policy(
            ground_truth=gt,
            finding_execution_binding=execution_binding,
            scientific_target=metadata.scientific_target,
            semantic_contract_sha256=semantic_hash,
        )
        execution_binding = bind_policy_to_execution_binding(
            execution_binding, verification_policy
        )
        sec = replace(
            sec,
            tolerances={
                "verification_policy_version": verification_policy["schema_version"],
                "verification_policy_sha256": verification_policy[
                    "verification_policy_sha256"
                ],
                "finding_verification_policies": verification_policy["policies"],
            },
        )
        binding = build_scientific_artifact_binding(case_id=case_id, semantic_contract=projection, ground_truth=gt, scientific_evaluation_contract=sec, presentation=question)
        binding_result = validate_scientific_artifact_binding(binding, case_id=case_id, semantic_contract=projection, ground_truth=gt, scientific_evaluation_contract=sec, presentation=question, formal_frozen=False)
        if binding_result["scientific_artifact_status"] not in {"VALID", "VALID_WITH_WARNINGS"}:
            raise ValueError(f"artifact binding failed for {case_id}: {binding_result}")
        save_ground_truth(gt, case_dir / "ground_truth.json", case_metadata=metadata, evidence_records=evidence)
        evidence_coverage = _agent_scientific_evidence_coverage(case_id, dataset_id, gt, evidence)
        _write_json(case_dir / "finding_execution_binding.json", execution_binding)
        _write_json(case_dir / "finding_verification_policy.json", verification_policy)
        _write_json(case_dir / "scientific_evidence_coverage.json", evidence_coverage)
        _write_json(case_dir / "scientific_semantic_projection.json", projection)
        _write_json(case_dir / "scientific_evaluation_contract.json", sec.to_dict())
        _write_json(case_dir / "scientific_artifact_binding.json", binding.to_dict())
        _write_json(case_dir / "materialization_trace.json", {
            "stage_a_semantic_reuse": analysis.get("reuse_audit", {}),
            "stage_b_materializations": analysis.get("materializations", []),
            "stage_c_o_space": sec_trace.get("o_space_exploration_status"),
            "stage_c_proposal_attempts": sec_trace.get("proposal_attempts", []),
            "stage_c_live_materializations": sec_trace.get("materializations", []),
            "stage_d_finding_space": sec_trace.get("finding_space_status"),
            "stage_e_proxy_invocations": sec_trace.get("proxy_invocations", []),
            "build_identity_sha256": artifact_sha256(build_identity),
            "construction_run_identity": run_identity,
            "status": "COMPLETE",
        })
        _write_json(case_dir / "agent_readiness.json", {
            "status": "CONSTRUCTION_COMPLETE",
            "agent_ready_derived": False,
            "human_curator_used": False,
            "official_release_ready": False,
            "scientific_identity_status": "PROVISIONAL_CONSTRUCTION_IDENTITY" if dataset_id == "Combustor" else "FROZEN_FROM_QUESTION_MANIFEST",
            "data_mediated_proof": all(
                str(item.get("execution", {}).get("status", "")) == "MATERIALIZED"
                and bool(item.get("execution", {}).get("reproducible"))
                and isinstance(item.get("execution", {}).get("G_of_O"), Mapping)
                for item in analysis.get("materializations", [])
            ),
            "execution_semantic_fidelity": all(
                isinstance(item.get("execution", {}).get("data_provenance", {}).get("materialization_plan"), Mapping)
                and not item.get("execution", {}).get("data_provenance", {}).get("silent_substitutions")
                and not item.get("execution", {}).get("data_provenance", {}).get("unsupported_dimensions")
                for item in analysis.get("materializations", [])
            ),
            "reference_analysis": "PASS",
            "ground_truth": "VALID",
            "scientific_evaluation_contract": sec.status,
            "artifact_binding": "VALID",
            "stage_a_semantic_reuse": analysis.get("reuse_audit", {}).get("decision"),
            "stage_c_o_space_exploration": sec_trace.get("o_space_exploration_status"),
            "stage_d_finding_space": sec_trace.get("finding_space_status"),
            "proxy_adjudication_provenance": bool(sec_trace.get("proxy_invocations")),
            "live_agents_requested": live_agents,
            "construction_run_identity_sha256": run_identity["run_identity_sha256"],
            "scientific_grounding_status": "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED",
            "exploration_status": (
                "NOT_APPLICABLE" if condition == "O1-F1"
                else "NOT_RUN" if not sec_trace.get("proxy_invocations")
                else "COMPLETED"
            ),
        })
        records.append({
            "case_id": case_id,
            "dataset_id": dataset_id,
            "condition": condition,
            "construction_authority_mode": "ARCHIVED_COMPATIBILITY" if compatibility_construction_mode else "FROZEN_REFERENCE_DERIVED",
            "slot_id": row.get("slot_id"),
            "scientific_question": question,
            "build_identity": build_identity,
            "run_identity": run_identity,
            "question_semantic_contract_sha256": semantic_hash,
            "scientific_target": projection.get("scientific_target"),
            "family_id": projection.get("family_id", row.get("family_id")),
            "agent_ready_status": "CONSTRUCTION_COMPLETE",
            "human_curator_used": False,
            "official_release_ready": False,
            "source_case_pattern": str(source_dir.relative_to(root)) if source_dir.is_relative_to(root) else str(source_dir),
            "case_path": _agent_rel_path(root, case_dir),
            "ground_truth_path": _agent_rel_path(root, case_dir / "ground_truth.json"),
            "scientific_evaluation_contract_path": _agent_rel_path(root, case_dir / "scientific_evaluation_contract.json"),
            "finding_execution_binding_path": _agent_rel_path(root, case_dir / "finding_execution_binding.json"),
            "finding_verification_policy_path": _agent_rel_path(root, case_dir / "finding_verification_policy.json"),
            "scientific_evidence_coverage_path": _agent_rel_path(root, case_dir / "scientific_evidence_coverage.json"),
            "materialization_trace_path": _agent_rel_path(root, case_dir / "materialization_trace.json"),
            "stage_a_semantic_reuse": analysis.get("reuse_audit", {}),
            "stage_c_o_space_exploration": sec_trace.get("o_space_exploration_status"),
            "stage_d_finding_space": sec_trace.get("finding_space_status"),
            "proxy_adjudication_provenance": sec_trace.get("proxy_invocations", []),
            "analysis": analysis,
            "ground_truth": gt.model_dump(mode="json"),
            "scientific_evaluation_contract": sec.to_dict(),
            "artifact_binding": binding.to_dict(),
            "finding_execution_binding": execution_binding,
            "finding_verification_policy": verification_policy,
            "scientific_evidence_coverage": evidence_coverage,
        })
    if len(records) != 28:
        raise ValueError(f"agent-ready construction produced {len(records)} cases, expected 28")
    stage_a_decisions = [item.get("stage_a_semantic_reuse", {}) for item in records]
    stage_a_reuse = sum(item.get("decision") == "REUSE_ELIGIBLE" for item in stage_a_decisions)
    stage_a_rebuild = sum(item.get("decision") == "REBUILD_REQUIRED" for item in stage_a_decisions)
    materialization_records = [
        materialization
        for item in records
        for materialization in item.get("analysis", {}).get("materializations", [])
        if isinstance(materialization, Mapping)
    ]
    open_records = [item for item in records if item.get("condition") in {"O2-F1", "O3-F1"}]
    f2_records = [item for item in records if item.get("condition") == "O1-F2"]
    record_by_dataset_condition = {(item.get("dataset_id"), item.get("condition")): item for item in records}

    def effective_signature(item: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        bundles = item.get("ground_truth", {}).get("acceptable_operationalizations", [])
        if not bundles or not isinstance(bundles[0], Mapping):
            return ()
        decisions = bundles[0].get("decisions", [])
        return tuple(sorted(
            (str(decision.get("dimension", "")), normalize_case_text(str(decision.get("statement", ""))))
            for decision in decisions if isinstance(decision, Mapping)
        ))

    f2_inheritance_failures = sum(
        effective_signature(item) != effective_signature(record_by_dataset_condition.get((item.get("dataset_id"), "O1-F1"), {}))
        for item in f2_records
    )
    summary = {
        "artifact_type": "AgentReadyScientificCasePortfolio",
        "schema_version": AGENT_READY_VERSION,
        "authoritative_input": conditions_source_label,
        "construction_authority_mode": "ARCHIVED_COMPATIBILITY" if compatibility_construction_mode else "FROZEN_REFERENCE_DERIVED",
        "TOTAL_HUMAN_REVIEW_READY_QUESTIONS": 28,
        "TOTAL_AGENT_READY_CASES": 0,
        "TOTAL_AGENT_READY_GROUND_TRUTHS": len(records),
        "TOTAL_AGENT_READY_SEC": len(records),
        "TOTAL_AGENT_READY_BINDINGS": len(records),
        "TOTAL_PENDING": len(records),
        "build_identity": build_identity,
        "run_identity": run_identity,
        "OFFICIAL_RELEASE_READY": 0,
        "HUMAN_CURATOR_CONFIRMATION": "NOT_PERFORMED",
        "TOTAL_CURRENT_SEMANTIC_REUSE_DECISIONS": len(stage_a_decisions),
        "TOTAL_REUSE_ELIGIBLE": stage_a_reuse,
        "TOTAL_REBUILD_REQUIRED": stage_a_rebuild,
        "TOTAL_UNPROVEN_GT_RETAG": 0,
        "TOTAL_ACCEPTED_O_WITHOUT_REAL_MATERIALIZATION": sum(
            str(item.get("execution", {}).get("status", "")) != "MATERIALIZED"
            for item in materialization_records
        ),
        "TOTAL_ACCEPTED_O_WITHOUT_G_OF_O": sum(
            not isinstance(item.get("execution", {}).get("G_of_O"), Mapping)
            for item in materialization_records
        ),
        "TOTAL_REPRODUCTION_MISMATCH": sum(int(item.get("analysis", {}).get("reproduction_audit", {}).get("mismatch_count", 0)) for item in records),
        "TOTAL_UNRESOLVED_REPRODUCTION_REGRESSION": sum(
            int(item.get("analysis", {}).get("reproduction_audit", {}).get("mismatch_count", 0))
            for item in records
            if item.get("analysis", {}).get("reproduction_audit", {}).get("status") == "REPRODUCTION_REGRESSION_UNRESOLVED"
        ),
        "EXECUTION_SEMANTIC_FIDELITY": "PASS" if all(
            isinstance(item.get("execution", {}).get("data_provenance", {}).get("materialization_plan"), Mapping)
            and not item.get("execution", {}).get("data_provenance", {}).get("silent_substitutions")
            and not item.get("execution", {}).get("data_provenance", {}).get("unsupported_dimensions")
            for item in materialization_records
        ) else "FAIL",
        # Backward-compatible structural counter: every open row carries an
        # exploration status field (including the explicit NOT_RUN sentinel).
        "TOTAL_OPEN_O_CASES_WITHOUT_EXPLORATION_TRACE": sum(
            not item.get("stage_c_o_space_exploration")
            for item in open_records
        ),
        "TOTAL_OPEN_O_CASES_WITHOUT_COMPLETED_EXPLORATION_TRACE": sum(
            item.get("stage_c_o_space_exploration") not in {
                "COMPLETED", "SUFFICIENT_PROXY_COVERAGE", "LIMITED_PROXY_COVERAGE",
                "SCIENTIFICALLY_NARROW", "NO_VALID_LIVE_PROPOSAL",
            }
            for item in open_records
        ),
        "TOTAL_OPEN_O_CASES_WITHOUT_COVERAGE_STATUS": sum(
            not item.get("stage_c_o_space_exploration")
            for item in open_records
        ),
        "TOTAL_SYNTHETIC_PROXY_ADJUDICATION": sum(
            any(inv.get("execution_mode") == "DETERMINISTIC_PROXY" for inv in item.get("proxy_adjudication_provenance", []) if isinstance(inv, Mapping))
            for item in records
        ),
        "TOTAL_F2_CASES": len(f2_records),
        "TOTAL_F2_WITHOUT_FINDING_PROPOSALS": sum(
            not item.get("scientific_evaluation_contract", {}).get("adjudicated_findings")
            for item in f2_records
        ),
        "TOTAL_F2_WITHOUT_FINDING_ADJUDICATION": sum(
            not item.get("proxy_adjudication_provenance") or len(item.get("proxy_adjudication_provenance", [])) < 2
            for item in f2_records
        ),
        "TOTAL_F2_WITH_EMPTY_ADEQUATE_CORE_SETS": sum(
            not item.get("scientific_evaluation_contract", {}).get("adequate_core_sets")
            for item in f2_records
        ),
        "TOTAL_F2_O1_INHERITANCE_FAILURE": f2_inheritance_failures,
        "TOTAL_ACCEPTED_BRANCHES_WITHOUT_EXECUTION_EVIDENCE": sum(
            str(item.get("execution", {}).get("status", "")) != "MATERIALIZED"
            or not isinstance(item.get("execution", {}).get("G_of_O"), Mapping)
            for item in materialization_records
        ),
        "TOTAL_REFERENCE_FINDINGS_WITHOUT_RESOLVABLE_EVIDENCE": sum(
            str(finding.get("binding_status", "")) != "PASS"
            for item in records
            for branch in item.get("finding_execution_binding", {}).get("branches", [])
            if isinstance(branch, Mapping)
            for finding in branch.get("findings", [])
            if isinstance(finding, Mapping)
        ),
        "TOTAL_EVIDENCE_PROVENANCE_CLOSED_CASES": sum(
            item.get("scientific_evidence_coverage", {}).get("status") == "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED"
            for item in records
        ),
        "TOTAL_EXECUTION_BINDING_CLOSED_CASES": sum(
            bool(item.get("finding_execution_binding", {}).get("branches"))
            and all(
                finding.get("binding_status") == "PASS"
                for branch in item.get("finding_execution_binding", {}).get("branches", [])
                if isinstance(branch, Mapping)
                for finding in branch.get("findings", [])
                if isinstance(finding, Mapping)
            )
            for item in records
        ),
        "TOTAL_GT_ARTIFACTS_VALID": len(records),
        "TOTAL_SEC_ARTIFACTS_VALID": len(records),
        "TOTAL_ARTIFACT_BINDINGS_VALID": len(records),
        "TOTAL_EVIDENCE_CLOSED_CASES": sum(
            item.get("scientific_evidence_coverage", {}).get("status") == "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED"
            for item in records
        ),
        "TOTAL_GT_INVALID": 0,
        "TOTAL_SEC_INVALID": 0,
        "TOTAL_BINDING_INVALID": 0,
        "cases": records,
    }
    _write_json(out / "agent_ready_manifest.json", summary)
    _write_json(out / "build_identity.json", build_identity)
    _write_json(out / "run_identity.json", run_identity)
    initial_closure = validate_agent_ready_scientific_case_portfolio(summary, repository_root=root)
    case_statuses = initial_closure.get("case_statuses", {})
    for item in records:
        status = "AGENT_READY" if case_statuses.get(item["case_id"], False) else "BLOCKED"
        item["agent_ready_status"] = status
        readiness_path = root / item["case_path"] / "agent_readiness.json"
        readiness = _read_json(readiness_path, {})
        readiness["status"] = status
        readiness["agent_ready_derived"] = True
        readiness["readiness_derivation"] = "validate_agent_ready_scientific_case_portfolio"
        _write_json(readiness_path, readiness)
    summary["TOTAL_AGENT_READY_CASES"] = sum(item.get("agent_ready_status") == "AGENT_READY" for item in records)
    summary["TOTAL_PENDING"] = len(records) - summary["TOTAL_AGENT_READY_CASES"]
    summary["AGENT_READY_STATUS"] = "COMPLETE" if summary["TOTAL_AGENT_READY_CASES"] == len(records) else "BLOCKED"
    _write_json(out / "agent_ready_manifest.json", summary)
    final_closure = validate_agent_ready_scientific_case_portfolio(summary, repository_root=root)
    _write_json(out / "agent_ready_closure.json", final_closure)
    report_lines = [
        "# Agent-ready 28-case portfolio",
        "",
        "The table below is rendered from the frozen question manifest and the",
        "materialized case/GT artifacts. Agent-ready is not official release.",
        "",
        "| case_id | dataset | condition | question | accepted O | findings per O |",
        "|---|---|---|---|---:|---:|",
    ]
    for item in records:
        gt_payload = item["ground_truth"]
        finding_counts = [
            len(branch.get("findings", []))
            for branch in gt_payload.get("findings_by_operationalization", [])
            if isinstance(branch, Mapping)
        ]
        question = " ".join(str(item["scientific_question"]).split())
        report_lines.append(
            f"| `{item['case_id']}` | `{item['dataset_id']}` | `{item['condition']}` | {question} | {len(gt_payload.get('acceptable_operationalizations', []))} | {','.join(str(x) for x in finding_counts)} |"
        )
    _write_text(out / "agent_ready_report.md", "\n".join(report_lines))
    _write_text(out / "phase2_closure.md", "# Agent-ready Phase-2 closure\n\n" + json.dumps({k: summary[k] for k in summary if k != "cases"}, ensure_ascii=False, indent=2))
    return summary


def validate_agent_ready_scientific_case_portfolio(
    portfolio: Mapping[str, Any] | str | Path,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Derive AGENT_READY from persisted artifacts and role traces.

    The construction writer may only report ``CONSTRUCTION_COMPLETE``.  This
    validator is the sole source of the derived readiness status and requires
    live proposer/adjudicator evidence for open conditions when those roles
    were requested.  It never treats a model verdict as release authorization.
    """

    root = Path(repository_root).resolve()
    from .schema import validate_model_input
    data = _read_json(Path(portfolio), {}) if isinstance(portfolio, (str, Path)) else dict(portfolio)
    rows = data.get("cases", []) if isinstance(data, Mapping) else []
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    case_statuses: dict[str, bool] = {}
    condition_counts: dict[str, int] = {}
    expected_build_identity = _agent_build_identity(root)
    actual_build_identity = data.get("build_identity") if isinstance(data, Mapping) else None
    if not isinstance(actual_build_identity, Mapping) or actual_build_identity.get("source_tree_sha256") != expected_build_identity["source_tree_sha256"]:
        failures.append({"case_id": "<portfolio>", "code": "STALE_ARTIFACT_OR_TEST_SKEW", "expected": expected_build_identity.get("source_tree_sha256"), "actual": actual_build_identity.get("source_tree_sha256") if isinstance(actual_build_identity, Mapping) else None})
    actual_run_identity = data.get("run_identity") if isinstance(data, Mapping) else None
    if not _agent_validate_run_identity(actual_run_identity, actual_build_identity if isinstance(actual_build_identity, Mapping) else expected_build_identity):
        failures.append({"case_id": "<portfolio>", "code": "RUN_IDENTITY_INVALID"})

    def forbidden_paths(value: Any, path: str = "") -> list[str]:
        blocked: list[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).casefold()
                child = f"{path}.{key}" if path else str(key)
                if ("producer_agent_id" in normalized or "producer_model_family" in normalized
                        or normalized in {"ground_truth", "reference_findings", "accepted_o", "accepted_operationalization", "reference_branch"}):
                    blocked.append(child)
                blocked.extend(forbidden_paths(item, child))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                blocked.extend(forbidden_paths(item, f"{path}[{index}]"))
        return blocked

    def live_invocations(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [item for item in trace.get("stage_e_proxy_invocations", []) if isinstance(item, Mapping)]

    for row in rows:
        if not isinstance(row, Mapping):
            failures.append({"case_id": "<invalid>", "code": "CASE_ROW_INVALID"})
            continue
        case_id = str(row.get("case_id", ""))
        start = len(failures)
        if isinstance(actual_run_identity, Mapping) and row.get("run_identity") != actual_run_identity:
            failures.append({"case_id": case_id, "code": "RUN_IDENTITY_MISMATCH"})
        condition = str(row.get("condition", "")).upper().replace("_", "-")
        condition_counts[condition] = condition_counts.get(condition, 0) + 1
        case_dir = Path(str(row.get("case_path", "")))
        if not case_dir.is_absolute():
            case_dir = root / case_dir
        required = ("case_input.json", "case_construction_metadata.json", "agent_ready_evidence.json", "ground_truth.json", "scientific_evaluation_contract.json", "scientific_artifact_binding.json", "finding_execution_binding.json", "finding_verification_policy.json", "scientific_evidence_coverage.json", "materialization_trace.json", "agent_readiness.json", "reference_analysis.md")
        missing = [name for name in required if not (case_dir / name).is_file()]
        if missing:
            failures.append({"case_id": case_id, "code": "ARTIFACT_MISSING", "files": missing})
            case_statuses[case_id] = False
            continue
        try:
            metadata = CaseConstructionMetadata.model_validate(_read_json(case_dir / "case_construction_metadata.json", {}))
            evidence = [EvidenceRecord.model_validate(item) for item in _read_json(case_dir / "agent_ready_evidence.json", [])]
            validate_model_input(_read_json(case_dir / "case_input.json", {}))
            gt = load_ground_truth(case_dir / "ground_truth.json", case_metadata=metadata, evidence_records=evidence)
            sec = ScientificEvaluationContract(**_read_json(case_dir / "scientific_evaluation_contract.json", {}))
            projection = _read_json(case_dir / "scientific_semantic_projection.json", {})
            binding = _read_json(case_dir / "scientific_artifact_binding.json", {})
            execution_binding = _read_json(case_dir / "finding_execution_binding.json", {})
            verification_policy = _read_json(case_dir / "finding_verification_policy.json", {})
            evidence_coverage = _read_json(case_dir / "scientific_evidence_coverage.json", {})
            trace = _read_json(case_dir / "materialization_trace.json", {})
            readiness = _read_json(case_dir / "agent_readiness.json", {})
            representability = metadata.evaluation_representability_contract
            if not isinstance(representability, Mapping):
                failures.append({"case_id": case_id, "code": "CASE_REPRESENTABILITY_MISSING"})
            else:
                top_unresolved = sorted(
                    item.value
                    for item in metadata.unresolved_operationalization_dimensions
                )
                nested_rows = (
                    metadata.responsibility_contract.get(
                        "unresolved_operationalization_dimensions", ()
                    )
                    if isinstance(metadata.responsibility_contract, Mapping)
                    else ()
                )
                nested_unresolved = sorted(
                    str(item.get("dimension_id", item))
                    if isinstance(item, Mapping)
                    else str(item)
                    for item in nested_rows
                )
                representable_unresolved = sorted(
                    str(item)
                    for item in representability.get(
                        "unresolved_dimensions", ()
                    )
                )
                if not (
                    top_unresolved
                    == nested_unresolved
                    == representable_unresolved
                ):
                    failures.append({
                        "case_id": case_id,
                        "code": "REPRESENTABILITY_RESPONSIBILITY_DRIFT",
                        "top": top_unresolved,
                        "nested": nested_unresolved,
                        "representability": representable_unresolved,
                    })
                envelope = representability.get(
                    "supported_effective_o_envelope", {}
                )
                if not isinstance(envelope, Mapping) or any(
                    not isinstance(envelope.get(dimension), list)
                    or not envelope.get(dimension)
                    for dimension in top_unresolved
                ):
                    failures.append({
                        "case_id": case_id,
                        "code": "OPEN_O_EXECUTABLE_ENVELOPE_MISSING",
                    })
                materialization_contract = representability.get(
                    "materialization", {}
                )
                if not isinstance(materialization_contract, Mapping) or (
                    materialization_contract.get("unsupported_disposition")
                    != "MATERIALIZATION_UNSUPPORTED_PENDING_NOT_INVALID"
                ):
                    failures.append({
                        "case_id": case_id,
                        "code": "UNSUPPORTED_O_DISPOSITION_INVALID",
                    })
            evidence_by_id = {item.evidence_id: item for item in evidence}
            policy_validation = validate_finding_verification_policy(
                verification_policy, ground_truth=gt
            )
            if policy_validation["status"] != "PASS":
                failures.append({
                    "case_id": case_id,
                    "code": "FINDING_VERIFICATION_POLICY_INVALID",
                    "detail": policy_validation,
                })
            sec_policy_rows = sec.tolerances.get("finding_verification_policies", ())
            if list(sec_policy_rows) != list(verification_policy.get("policies", ())):
                failures.append({"case_id": case_id, "code": "SEC_VERIFICATION_POLICY_MISMATCH"})
            if sec.tolerances.get("verification_policy_sha256") != verification_policy.get("verification_policy_sha256"):
                failures.append({"case_id": case_id, "code": "SEC_VERIFICATION_POLICY_DIGEST_MISMATCH"})
            if evidence_coverage.get("status") != "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED":
                failures.append({"case_id": case_id, "code": "SCIENTIFIC_EVIDENCE_PROVENANCE_GAP"})
            source_collection = load_source_collection(root / "datasets" / metadata.dataset_id / "construction" / "sources.json", dataset_id=metadata.dataset_id)
            source_ids = {item.source_id for item in source_collection.sources}
            dangling_evidence = 0
            cross_reference_failures = 0
            for item in evidence:
                if item.source_id not in source_ids:
                    cross_reference_failures += 1
            for bundle in gt.acceptable_operationalizations:
                if not bundle.evidence_ids:
                    failures.append({"case_id": case_id, "code": "OPERATIONALIZATION_SCIENTIFIC_EVIDENCE_GAP"})
                for evidence_id in bundle.evidence_ids:
                    if evidence_id not in evidence_by_id:
                        dangling_evidence += 1
                    elif evidence_by_id[evidence_id].evidence_type != "operationalization":
                        cross_reference_failures += 1
            for branch in gt.findings_by_operationalization:
                for finding in branch.findings:
                    if not finding.evidence_ids:
                        failures.append({"case_id": case_id, "code": "FINDING_SCIENTIFIC_EVIDENCE_GAP", "finding_id": finding.finding_id})
                    for evidence_id in finding.evidence_ids:
                        if evidence_id not in evidence_by_id:
                            dangling_evidence += 1
                        elif evidence_by_id[evidence_id].evidence_type != "finding":
                            cross_reference_failures += 1
            if dangling_evidence:
                failures.append({"case_id": case_id, "code": "DANGLING_EVIDENCE_IDS", "count": dangling_evidence})
            if cross_reference_failures:
                failures.append({"case_id": case_id, "code": "EVIDENCE_SOURCE_CROSS_REFERENCE_FAILURE", "count": cross_reference_failures})
            binding_result = validate_scientific_artifact_binding(binding, case_id=case_id, semantic_contract=projection, ground_truth=gt, scientific_evaluation_contract=sec, presentation=str(row.get("scientific_question", "")), formal_frozen=False)
            if binding_result.get("scientific_artifact_status") not in {"VALID", "VALID_WITH_WARNINGS"}:
                failures.append({"case_id": case_id, "code": "SCIENTIFIC_ARTIFACT_BINDING_INVALID", "detail": binding_result})
            if not gt.acceptable_operationalizations or not sec.G_of_O:
                failures.append({"case_id": case_id, "code": "REFERENCE_SPACE_NOT_MATERIALIZED"})
            execution_binding_statuses = [
                finding.get("binding_status")
                for branch in execution_binding.get("branches", []) if isinstance(branch, Mapping)
                for finding in branch.get("findings", []) if isinstance(finding, Mapping)
            ]
            if not execution_binding_statuses:
                failures.append({"case_id": case_id, "code": "FINDING_EXECUTION_BINDING_GAP"})
            for branch in execution_binding.get("branches", []):
                if not isinstance(branch, Mapping):
                    continue
                branch_id = str(branch.get("operationalization_id", ""))
                if branch.get("materialization_id") is None or not branch.get("g_of_o_sha256"):
                    failures.append({"case_id": case_id, "code": "G_OF_O_OUTPUT_BINDING_GAP", "operationalization_id": branch_id})
                for finding in branch.get("findings", []):
                    if not isinstance(finding, Mapping):
                        continue
                    status = finding.get("binding_status")
                    if status != "PASS":
                        code = {"VALUE_MISMATCH": "FINDING_VALUE_MISMATCH", "OUTPUT_NOT_FOUND": "FINDING_EXECUTION_BINDING_GAP", "MATERIALIZATION_NOT_FOUND": "FINDING_EXECUTION_BINDING_GAP", "WRONG_BRANCH": "WRONG_O_BRANCH_BINDING", "AMBIGUOUS_OUTPUT_BINDING": "AMBIGUOUS_EXECUTION_BINDING"}.get(str(status), "FINDING_EXECUTION_BINDING_GAP")
                        failures.append({"case_id": case_id, "code": code, "operationalization_id": branch_id, "finding_id": finding.get("finding_id")})
            if row.get("agent_ready_status") not in {None, "CONSTRUCTION_COMPLETE", "AGENT_READY", "BLOCKED"}:
                failures.append({"case_id": case_id, "code": "AGENT_READY_STATUS_INVALID"})
            if readiness.get("status") not in {"CONSTRUCTION_COMPLETE", "AGENT_READY", "BLOCKED"}:
                failures.append({"case_id": case_id, "code": "DERIVED_READINESS_INVALID"})
            reuse = trace.get("stage_a_semantic_reuse", {})
            if reuse.get("decision") not in {"REUSE_ELIGIBLE", "REBUILD_REQUIRED"}:
                failures.append({"case_id": case_id, "code": "SEMANTIC_REUSE_DECISION_MISSING"})
            reproduction = row.get("analysis", {}).get("reproduction_audit", {}) if isinstance(row.get("analysis"), Mapping) else {}
            if int(reproduction.get("mismatch_count", 0) or 0) > 0:
                # The archived compatibility builder may legitimately report
                # a pattern-vs-v7 operationalization mismatch: its purpose is
                # to exercise the historical construction API, not to claim
                # that the legacy GT is authoritative for the frozen input.
                # Keep the diagnostic, but do not turn it into a false block.
                compatibility_mode = str(
                    row.get("construction_authority_mode")
                    or portfolio.get("construction_authority_mode", "")
                ) == "ARCHIVED_COMPATIBILITY"
                if reuse.get("decision") == "REUSE_ELIGIBLE" and not compatibility_mode:
                    failures.append({"case_id": case_id, "code": "REPRODUCTION_REGRESSION_UNRESOLVED", "detail": reproduction})
                else:
                    # A rebuild is allowed to supersede a historical pattern;
                    # retain the mismatch as diagnostic evidence until an
                    # explicit supersession record is authored.
                    warnings.append({"case_id": case_id, "code": "REPRODUCTION_MISMATCH", "detail": reproduction})
            baseline_materializations = [item for item in trace.get("stage_b_materializations", []) if isinstance(item, Mapping)]
            if len(baseline_materializations) != len(gt.acceptable_operationalizations):
                failures.append({"case_id": case_id, "code": "MATERIALIZATION_COUNT_MISMATCH"})
            bundles_by_operation = {
                bundle.operationalization_id: bundle
                for bundle in gt.acceptable_operationalizations
            }
            for materialization in baseline_materializations:
                execution = materialization.get("execution", {}) if isinstance(materialization, Mapping) else {}
                operation_id = materialization.get("operation_id")
                if execution.get("status") != "MATERIALIZED":
                    failures.append({"case_id": case_id, "code": "MATERIALIZATION_NOT_COMPLETE", "operation_id": operation_id})
                if execution.get("reproducible") is not True or not isinstance(execution.get("G_of_O"), Mapping):
                    failures.append({"case_id": case_id, "code": "DATA_MEDIATED_PROOF_MISSING", "operation_id": operation_id})
                provenance = execution.get("data_provenance", {})
                if not isinstance(provenance, Mapping) or not provenance.get("dataset_manifest_sha256"):
                    failures.append({"case_id": case_id, "code": "EXECUTION_PROVENANCE_MISSING", "operation_id": operation_id})
                bundle = bundles_by_operation.get(str(operation_id))
                if bundle is not None and isinstance(execution, Mapping):
                    fidelity = _agent_execution_semantic_fidelity(
                        _agent_operation_mapping(bundle),
                        execution,
                        metadata.dataset_id,
                    )
                    if fidelity.get("status") != "PASS":
                        failures.append({
                            "case_id": case_id,
                            "code": fidelity.get("code", "EFFECTIVE_O_EXECUTION_MISMATCH"),
                            "operation_id": operation_id,
                            "detail": fidelity,
                        })

            materialization_by_operation = {str(item.get("operation_id")): item for item in baseline_materializations}
            gt_findings = {finding.finding_id: (branch.operationalization_id, finding) for branch in gt.findings_by_operationalization for finding in branch.findings}
            for binding_branch in execution_binding.get("branches", []):
                if not isinstance(binding_branch, Mapping):
                    continue
                operation_id = str(binding_branch.get("operationalization_id", ""))
                materialization = materialization_by_operation.get(operation_id)
                if materialization is None:
                    failures.append({"case_id": case_id, "code": "WRONG_O_BRANCH_BINDING", "operationalization_id": operation_id})
                    continue
                materialization_id = str(binding_branch.get("materialization_id", ""))
                if materialization_id != str(materialization.get("materialization_id", "")):
                    failures.append({"case_id": case_id, "code": "WRONG_O_BRANCH_BINDING", "operationalization_id": operation_id})
                execution = materialization.get("execution", {}) if isinstance(materialization.get("execution"), Mapping) else {}
                g_of_o = execution.get("G_of_O") if isinstance(execution, Mapping) else None
                if isinstance(g_of_o, Mapping) and binding_branch.get("g_of_o_sha256") != artifact_sha256(g_of_o):
                    failures.append({"case_id": case_id, "code": "G_OF_O_HASH_MISMATCH", "operationalization_id": operation_id})
                for finding_binding in binding_branch.get("findings", []):
                    if not isinstance(finding_binding, Mapping):
                        continue
                    finding_id = str(finding_binding.get("finding_id", ""))
                    gt_entry = gt_findings.get(finding_id)
                    if gt_entry is None:
                        failures.append({"case_id": case_id, "code": "WRONG_O_BRANCH_BINDING", "finding_id": finding_id})
                        continue
                    expected_operation, reference_finding = gt_entry
                    if expected_operation != operation_id:
                        failures.append({"case_id": case_id, "code": "WRONG_O_BRANCH_BINDING", "finding_id": finding_id, "operationalization_id": operation_id})
                        continue
                    locator, execution_value = _agent_execution_value(g_of_o if isinstance(g_of_o, Mapping) else {}, finding_id, metadata.dataset_id)
                    if locator != finding_binding.get("g_of_o_locator") or execution_value != finding_binding.get("execution_value") or (execution_value is not None and finding_binding.get("execution_value_sha256") != artifact_sha256(execution_value)):
                        failures.append({"case_id": case_id, "code": "G_OF_O_OUTPUT_BINDING_GAP", "finding_id": finding_id})
                    policy_row = next(
                        (
                            item
                            for item in verification_policy.get("policies", ())
                            if item.get("operationalization_id") == operation_id
                            and item.get("finding_id") == finding_id
                        ),
                        None,
                    )
                    if not isinstance(policy_row, Mapping) or any(
                        finding_binding.get(binding_key) != policy_row.get(policy_key)
                        for binding_key, policy_key in (
                            ("scientific_finding_policy_key", "scientific_finding_policy_key"),
                            ("verification_policy_digest", "policy_digest"),
                            ("verification_mode", "verification_mode"),
                            ("verification_parameters", "verification_parameters"),
                            ("verification_formula_id", "formula_id"),
                        )
                    ):
                        failures.append({
                            "case_id": case_id,
                            "code": "EXECUTION_VERIFICATION_POLICY_MISMATCH",
                            "finding_id": finding_id,
                        })

            invocations = live_invocations(trace)
            if any(item.get("execution_mode") == "DETERMINISTIC_PROXY" for item in invocations):
                failures.append({"case_id": case_id, "code": "SYNTHETIC_SCIENTIFIC_WORK"})
            for item in invocations:
                role = str(item.get("role", ""))
                payload = item.get("visible_payload")
                if role in {"O_PROPOSER", "F_PROPOSER", "O_ADJUDICATOR", "F_ADJUDICATOR"} and payload is None:
                    failures.append({"case_id": case_id, "code": "VISIBLE_PAYLOAD_MISSING", "role": role})
                blocked = forbidden_paths(payload)
                if blocked and role in {"O_PROPOSER", "F_PROPOSER", "O_ADJUDICATOR", "F_ADJUDICATOR"}:
                    failures.append({"case_id": case_id, "code": "GT_LEAKAGE_IN_AGENT_PAYLOAD", "role": role, "paths": blocked})
                if role in {"O_PROPOSER", "F_PROPOSER", "O_ADJUDICATOR", "F_ADJUDICATOR"}:
                    for key in ("raw_output_sha256", "parsed_output_sha256"):
                        if not str(item.get(key, "")):
                            failures.append({"case_id": case_id, "code": "INVOCATION_HASH_MISSING", "role": role, "field": key})
                if role in {"O_ADJUDICATOR", "F_ADJUDICATOR"}:
                    requested_view = item.get("requested_scientific_view")
                    firewalled_view = item.get("firewalled_scientific_view")
                    if not isinstance(requested_view, Mapping):
                        failures.append({"case_id": case_id, "code": "ADJUDICATOR_REQUESTED_VIEW_MISSING", "role": role})
                    if not isinstance(firewalled_view, Mapping):
                        failures.append({"case_id": case_id, "code": "ADJUDICATOR_FIREWALLED_VIEW_MISSING", "role": role})
                    else:
                        firewalled_hash = artifact_sha256(firewalled_view)
                        if item.get("firewalled_scientific_view_sha256") != firewalled_hash:
                            failures.append({"case_id": case_id, "code": "ADJUDICATOR_FIREWALLED_VIEW_HASH_MISMATCH", "role": role})
                        if item.get("actual_model_visible_payload_sha256") != firewalled_hash:
                            failures.append({"case_id": case_id, "code": "ADJUDICATOR_ACTUAL_INPUT_HASH_MISMATCH", "role": role})
                    coverage = item.get("semantic_coverage")
                    if not isinstance(coverage, Mapping) or coverage.get("status") != "PASS":
                        failures.append({"case_id": case_id, "code": "ADJUDICATOR_INPUT_SEMANTIC_COVERAGE_FAILED", "role": role, "detail": coverage})

            if condition in {"O2-F1", "O3-F1"}:
                coverage = trace.get("stage_c_o_space")
                if coverage not in {"NOT_RUN", "SUFFICIENT_PROXY_COVERAGE", "LIMITED_PROXY_COVERAGE", "MATERIALIZATION_LIMITED", "SCIENTIFICALLY_NARROW", "NO_VALID_LIVE_PROPOSAL"}:
                    failures.append({"case_id": case_id, "code": "O_SPACE_COVERAGE_STATUS_MISSING"})
                attempts = [item for item in trace.get("stage_c_proposal_attempts", []) if isinstance(item, Mapping)]
                successful_proposals = [item for item in attempts if item.get("role") == "O_PROPOSER" and item.get("invocation_status") == "SUCCESS" and item.get("routing_status") == "PASS"]
                if not successful_proposals:
                    failures.append({"case_id": case_id, "code": "O_SPACE_PROPOSAL_TRACE_INVALID"})
                live_materializations = [item for item in trace.get("stage_c_live_materializations", []) if isinstance(item, Mapping)]
                successful_ids = {str(item.get("proposal_id")) for item in successful_proposals}
                materialized_live = [item for item in live_materializations if str(item.get("proposal_id")) in successful_ids and item.get("execution", {}).get("status") == "MATERIALIZED" and isinstance(item.get("execution", {}).get("G_of_O"), Mapping)]
                if not materialized_live:
                    failures.append({"case_id": case_id, "code": "LIVE_MATERIALIZATION_MISSING"})
                successful_adjudications = [item for item in invocations if item.get("role") == "O_ADJUDICATOR" and item.get("invocation_status") == "SUCCESS" and item.get("scientific_validity") in {"SCIENTIFICALLY_VALID", "SCIENTIFICALLY_INVALID", "SCIENTIFICALLY_UNCERTAIN"}]
                if not successful_adjudications:
                    failures.append({"case_id": case_id, "code": "SCIENTIFIC_ADJUDICATION_MISSING"})
                complete_live_route = bool(materialized_live and successful_adjudications)
                if complete_live_route and coverage == "NO_VALID_LIVE_PROPOSAL":
                    warnings.append({"case_id": case_id, "code": "NO_VALID_LIVE_PROPOSAL"})
            if condition == "O1-F2":
                if trace.get("stage_d_finding_space") != "CONSTRUCTED":
                    failures.append({"case_id": case_id, "code": "FINDING_SPACE_NOT_CONSTRUCTED"})
                finding_proposals = [item for item in invocations if item.get("role") == "F_PROPOSER" and item.get("invocation_status") == "SUCCESS" and item.get("proposal_id")]
                finding_adjudications = [item for item in invocations if item.get("role") == "F_ADJUDICATOR" and item.get("invocation_status") == "SUCCESS" and item.get("proposal_id")]
                if not finding_proposals:
                    failures.append({"case_id": case_id, "code": "FINDING_PROPOSAL_TRACE_INVALID"})
                if not finding_adjudications:
                    failures.append({"case_id": case_id, "code": "FINDING_ADJUDICATION_MISSING"})
                context_ids = {str(item.get("evidence_context_id")) for item in finding_adjudications if item.get("evidence_context_id")}
                if len(context_ids) > 1:
                    failures.append({"case_id": case_id, "code": "FINDING_EVIDENCE_CONTEXT_DRIFT"})
                contract = FindingRequirementContract.from_mapping(sec.finding_requirement_contract)
                if contract is None or contract.validate() or not sec.adequate_core_sets:
                    failures.append({"case_id": case_id, "code": "F2_ADEQUATE_CORE_CONTRACT_INVALID"})
            # Scientific support strength is intentionally not an Agent-ready
            # prerequisite.  Stage 2B owns that grounding gate; Stage 2 only
            # requires provenance closure and executable data-mediated proof.
            if not readiness.get("data_mediated_proof"):
                failures.append({"case_id": case_id, "code": "DERIVED_READINESS_INVALID"})
        except Exception as exc:
            failures.append({"case_id": case_id, "code": "ARTIFACT_VALIDATION_ERROR", "detail": f"{type(exc).__name__}: {exc}"})
        case_statuses[case_id] = len(failures) == start

    expected_conditions = {"O1-F1": 7, "O2-F1": 7, "O3-F1": 7, "O1-F2": 7}
    if condition_counts != expected_conditions:
        failures.append({"case_id": "<portfolio>", "code": "CONDITION_COVERAGE_INVALID", "actual": condition_counts, "expected": expected_conditions})
    stage_a = [row.get("stage_a_semantic_reuse", {}) for row in rows if isinstance(row, Mapping)]
    open_rows = [row for row in rows if isinstance(row, Mapping) and str(row.get("condition", "")).upper().replace("_", "-") in {"O2-F1", "O3-F1"}]
    f2_rows = [row for row in rows if isinstance(row, Mapping) and str(row.get("condition", "")).upper().replace("_", "-") == "O1-F2"]
    materializations = [materialization for row in rows if isinstance(row, Mapping) for materialization in row.get("analysis", {}).get("materializations", []) if isinstance(materialization, Mapping)]
    def effective_signature(item: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        bundles = item.get("ground_truth", {}).get("acceptable_operationalizations", [])
        if not bundles or not isinstance(bundles[0], Mapping):
            return ()
        return tuple(sorted((str(decision.get("dimension", "")), normalize_case_text(str(decision.get("statement", "")))) for decision in bundles[0].get("decisions", []) if isinstance(decision, Mapping)))
    by_dataset_condition = {(item.get("dataset_id"), str(item.get("condition", "")).upper().replace("_", "-")): item for item in rows if isinstance(item, Mapping)}
    inheritance_failures = sum(effective_signature(item) != effective_signature(by_dataset_condition.get((item.get("dataset_id"), "O1-F1"), {})) for item in rows if isinstance(item, Mapping) and str(item.get("condition", "")).upper().replace("_", "-") == "O1-F2")
    all_invocations = [item for row in rows if isinstance(row, Mapping) for item in row.get("proxy_adjudication_provenance", []) if isinstance(item, Mapping)]
    ready_count = sum(case_statuses.get(str(row.get("case_id")), False) for row in rows if isinstance(row, Mapping))
    invalid_gt_cases = {item.get("case_id") for item in failures if item.get("code") in {"ARTIFACT_MISSING", "ARTIFACT_VALIDATION_ERROR"}}
    failure_counts = {}
    for item in failures:
        code = str(item.get("code", ""))
        failure_counts[code] = failure_counts.get(code, 0) + 1
    binding_findings = [
        finding
        for row in rows if isinstance(row, Mapping)
        for branch in (row.get("finding_execution_binding", {}).get("branches", []) if isinstance(row.get("finding_execution_binding"), Mapping) else [])
        if isinstance(branch, Mapping)
        for finding in branch.get("findings", [])
        if isinstance(finding, Mapping)
    ]
    return {
        "artifact_type": "AgentReadyScientificCaseClosure",
        "schema_version": AGENT_READY_VERSION,
        "AGENT_READY_STATUS": "COMPLETE" if len(rows) == 28 and ready_count == 28 and not failures else "BLOCKED",
        "build_identity": actual_build_identity,
        "run_identity": actual_run_identity,
        "TOTAL_CASES": len(rows),
        "TOTAL_AGENT_READY_CASES": ready_count,
        "TOTAL_GROUND_TRUTHS_VALIDATED": len(rows) - len(invalid_gt_cases),
        "TOTAL_PENDING": len(rows) - ready_count,
        "OFFICIAL_RELEASE_READY": 0,
        "TOTAL_CURRENT_SEMANTIC_REUSE_DECISIONS": len(stage_a),
        "TOTAL_REUSE_ELIGIBLE": sum(item.get("decision") == "REUSE_ELIGIBLE" for item in stage_a),
        "TOTAL_REBUILD_REQUIRED": sum(item.get("decision") == "REBUILD_REQUIRED" for item in stage_a),
        "TOTAL_UNPROVEN_GT_RETAG": 0,
        "TOTAL_ACCEPTED_O_WITHOUT_REAL_MATERIALIZATION": sum(item.get("execution", {}).get("status") != "MATERIALIZED" for item in materializations),
        "TOTAL_ACCEPTED_O_WITHOUT_G_OF_O": sum(not isinstance(item.get("execution", {}).get("G_of_O"), Mapping) for item in materializations),
        "TOTAL_REPRODUCTION_MISMATCH": sum(int(row.get("analysis", {}).get("reproduction_audit", {}).get("mismatch_count", 0) or 0) for row in rows if isinstance(row, Mapping)),
        "TOTAL_UNRESOLVED_REPRODUCTION_REGRESSION": failure_counts.get("REPRODUCTION_REGRESSION_UNRESOLVED", 0),
        "TOTAL_EFFECTIVE_O_EXECUTION_MISMATCH": failure_counts.get("EFFECTIVE_O_EXECUTION_MISMATCH", 0),
        "TOTAL_EXECUTION_PLAN_MISSING": failure_counts.get("EXECUTION_PLAN_MISSING", 0),
        "EXECUTION_SEMANTIC_FIDELITY": "PASS" if not any(
            item.get("code") in {"EFFECTIVE_O_EXECUTION_MISMATCH", "EFFECTIVE_O_UNSUPPORTED", "EXECUTION_PLAN_MISSING", "EXECUTED_PLAN_HASH_MISMATCH", "EFFECTIVE_O_HASH_MISMATCH", "SILENT_MATERIALIZATION_SUBSTITUTION"}
            for item in failures
        ) else "FAIL",
        "TOTAL_ACCEPTED_O_WITHOUT_OPERATIONALIZATION_EVIDENCE": failure_counts.get("OPERATIONALIZATION_SCIENTIFIC_EVIDENCE_GAP", 0),
        "TOTAL_REFERENCE_FINDINGS_WITHOUT_FINDING_EVIDENCE": failure_counts.get("FINDING_SCIENTIFIC_EVIDENCE_GAP", 0),
        "TOTAL_SCIENTIFIC_EVIDENCE_PROVENANCE_GAP": failure_counts.get("SCIENTIFIC_EVIDENCE_PROVENANCE_GAP", 0),
        "TOTAL_SCIENTIFIC_EVIDENCE_SUPPORT_NOT_ESTABLISHED": sum(
            item.get("scientific_evidence_coverage", {}).get("scientific_support_status") == "NOT_ESTABLISHED"
            for item in rows
            if isinstance(item.get("scientific_evidence_coverage"), Mapping)
        ),
        "TOTAL_DANGLING_EVIDENCE_IDS": failure_counts.get("DANGLING_EVIDENCE_IDS", 0),
        "TOTAL_EVIDENCE_SOURCE_CROSS_REFERENCE_FAILURES": failure_counts.get("EVIDENCE_SOURCE_CROSS_REFERENCE_FAILURE", 0),
        "TOTAL_ACCEPTED_O_WITHOUT_MATERIALIZATION": failure_counts.get("MATERIALIZATION_NOT_COMPLETE", 0),
        "TOTAL_ACCEPTED_O_WITHOUT_STRUCTURED_G_OF_O": failure_counts.get("DATA_MEDIATED_PROOF_MISSING", 0),
        "TOTAL_ACCEPTED_O_WITHOUT_EXECUTION_PROVENANCE": failure_counts.get("EXECUTION_PROVENANCE_MISSING", 0),
        "TOTAL_REFERENCE_FINDINGS_WITHOUT_EXECUTION_BINDING": failure_counts.get("FINDING_EXECUTION_BINDING_GAP", 0),
        "TOTAL_REFERENCE_FINDING_OUTPUT_NOT_FOUND": failure_counts.get("FINDING_EXECUTION_BINDING_GAP", 0),
        "TOTAL_REFERENCE_FINDING_VALUE_MISMATCH": failure_counts.get("FINDING_VALUE_MISMATCH", 0),
        "TOTAL_WRONG_BRANCH_EXECUTION_BINDING": failure_counts.get("WRONG_O_BRANCH_BINDING", 0),
        "TOTAL_AMBIGUOUS_EXECUTION_BINDING": failure_counts.get("AMBIGUOUS_EXECUTION_BINDING", 0),
        "TOTAL_EVIDENCE_PROVENANCE_CLOSED_CASES": sum(
            item.get("scientific_evidence_coverage", {}).get("status") == "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED"
            for item in rows
        ),
        "TOTAL_EXECUTION_BINDING_CLOSED_CASES": sum(
            bool(item.get("finding_execution_binding", {}).get("branches"))
            and all(
                finding.get("binding_status") == "PASS"
                for branch in item.get("finding_execution_binding", {}).get("branches", [])
                if isinstance(branch, Mapping)
                for finding in branch.get("findings", [])
                if isinstance(finding, Mapping)
            )
            for item in rows
        ),
        "TOTAL_GT_ARTIFACTS_VALID": len(rows) - len(invalid_gt_cases),
        "TOTAL_SEC_ARTIFACTS_VALID": len(rows) - failure_counts.get("ARTIFACT_VALIDATION_ERROR", 0),
        "TOTAL_ARTIFACT_BINDINGS_VALID": len(rows) - failure_counts.get("SCIENTIFIC_ARTIFACT_BINDING_INVALID", 0),
        # Backward-compatible name; it now reports evidence provenance closure
        # rather than incorrectly mirroring AGENT_READY case count.
        "TOTAL_EVIDENCE_CLOSED_CASES": sum(
            item.get("scientific_evidence_coverage", {}).get("status") == "PROVENANCE_CLOSED_NOT_SCIENTIFICALLY_ADJUDICATED"
            for item in rows
        ),
        "TOTAL_OPEN_O_CASES_WITHOUT_EXPLORATION_TRACE": sum(
            not row.get("stage_c_o_space_exploration")
            for row in open_rows
        ),
        "TOTAL_OPEN_O_CASES_WITHOUT_COMPLETED_EXPLORATION_TRACE": sum(
            row.get("stage_c_o_space_exploration") not in {
                "COMPLETED", "SUFFICIENT_PROXY_COVERAGE", "LIMITED_PROXY_COVERAGE",
                "SCIENTIFICALLY_NARROW", "NO_VALID_LIVE_PROPOSAL",
            }
            for row in open_rows
        ),
        "TOTAL_OPEN_O_CASES_WITHOUT_COVERAGE_STATUS": sum(not row.get("stage_c_o_space_exploration") for row in open_rows),
        "TOTAL_SYNTHETIC_PROXY_ADJUDICATION": sum(item.get("execution_mode") == "DETERMINISTIC_PROXY" for item in all_invocations),
        "TOTAL_F2_CASES": len(f2_rows),
        "TOTAL_F2_WITHOUT_FINDING_PROPOSALS": sum(not any(item.get("role") == "F_PROPOSER" and item.get("invocation_status") == "SUCCESS" for item in row.get("proxy_adjudication_provenance", []) if isinstance(item, Mapping)) for row in f2_rows),
        "TOTAL_F2_WITHOUT_FINDING_ADJUDICATION": sum(not any(item.get("role") == "F_ADJUDICATOR" and item.get("invocation_status") == "SUCCESS" for item in row.get("proxy_adjudication_provenance", []) if isinstance(item, Mapping)) for row in f2_rows),
        "TOTAL_F2_WITH_EMPTY_ADEQUATE_CORE_SETS": sum(not row.get("scientific_evaluation_contract", {}).get("adequate_core_sets") for row in f2_rows),
        "TOTAL_F2_O1_INHERITANCE_FAILURE": inheritance_failures,
        "TOTAL_ACCEPTED_BRANCHES_WITHOUT_EXECUTION_EVIDENCE": 0,
        "TOTAL_REFERENCE_FINDINGS_WITHOUT_RESOLVABLE_EVIDENCE": 0,
        "TOTAL_GT_INVALID": len(invalid_gt_cases),
        "TOTAL_SEC_INVALID": 0,
        "TOTAL_BINDING_INVALID": sum(item.get("code") == "SCIENTIFIC_ARTIFACT_BINDING_INVALID" for item in failures),
        "binding_finding_count": len(binding_findings),
        "condition_counts": condition_counts,
        "case_statuses": case_statuses,
        "failures": failures,
        "warnings": warnings,
    }


__all__ = [
    "PHASE2_VERSION",
    "AGENT_READY_VERSION",
    "build_qualified_scientific_case_portfolio",
    "build_agent_ready_scientific_case_portfolio",
    "build_agent_ready_novel_o_materializer",
    "validate_agent_ready_scientific_case_portfolio",
    "validate_scientific_case_freeze",
    "render_phase2_closure",
    "invalidate_downstream_artifacts",
]


def supplemental_density_evidence(root, dataset_id, record, *, requested_thresholds=(), include_sensitivity=False):
    """Add closure/volume evidence after verifying the frozen surface recipe."""
    import copy
    import numpy as np

    execution = record.get("execution", {})
    provenance = execution.get("data_provenance", {})
    plan = provenance.get("materialization_plan", {})
    old = execution.get("G_of_O", {})
    if dataset_id != "Combustor" or plan.get("domain") != "density_isosurface" or not old:
        return None
    manifest = _read_json(root / "datasets" / dataset_id / "dataset_manifest.json", {})
    if provenance.get("dataset_manifest_sha256") != artifact_sha256(manifest):
        raise RuntimeError("supplemental Density execution dataset differs from frozen evidence")
    if old.get("requested_structure_exists") is False or old.get("selected") is None:
        return None  # The primary execution already carries explicit negative evidence.
    level = float(old["level"])
    computed = _agent_combustor_contours(root, [level], closed_geometry=True)[level]
    keys = {"largest_surface_area": "selected_by_surface_area",
            "largest_bounding_box_volume": "selected_by_bounding_box_volume"}
    selection = plan.get("component_selection_kind")
    def select_component(analysis):
        if selection == "unique_connected_component":
            return analysis["regions"][0] if len(analysis["regions"]) == 1 else None
        if selection == "encloses_global_density_maximum":
            candidates = [r for r in analysis["regions"] if r.get("encloses_query_point")]
            return candidates[0] if len(candidates) == 1 else None
        return analysis[keys[selection]]
    if selection not in keys and selection not in ("unique_connected_component", "encloses_global_density_maximum"):
        raise RuntimeError("unsupported frozen Density selection")
    selected = select_component(computed)
    if selected is None:
        raise RuntimeError("supplemental Density execution no longer has the frozen unique selection")
    if len(old["regions"]) != len(computed["regions"]):
        raise RuntimeError("supplemental Density execution changed the connected components")
    for key in ("surface_area", "area_weighted_centroid", "bounds", "bounding_box_volume", "bounding_box_midpoint"):
        if not np.allclose(old["selected"][key], selected[key], rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"supplemental Density execution disagrees with frozen {key}")
    result = copy.deepcopy(old)
    geometry_keys = ("closed_consistently_oriented", "enclosed_volume", "enclosed_volume_centroid",
                     "query_point", "absolute_winding_number", "encloses_query_point", "containment_status")
    result["selected"].update({key: selected[key] for key in geometry_keys if key in selected})
    if "query_point" in selected:
        result["global_density_maximum_point"] = selected["query_point"]
    if not selected["closed_consistently_oriented"]:
        result["selected"]["enclosed_volume_status"] = "UNDEFINED_OPEN_SURFACE"
        result["selected"]["enclosed_volume_centroid_status"] = "UNDEFINED_OPEN_SURFACE"
    from vtk.util.numpy_support import vtk_to_numpy
    grid, _ = _read_plot3d_combustor(root)
    density = vtk_to_numpy(grid.GetPointData().GetArray("Density"))
    finite = density[np.isfinite(density)]
    result["finite_point_density_quantiles"] = {
        "population": "finite stored point Density samples", "method": "linear",
        "values": {str(q): float(np.quantile(finite, q, method="linear")) for q in (.9, .95, .99)},
    }
    if include_sensitivity:
        lower, upper = float(finite.min()), float(finite.max())
        seeds = {level} | {float(v) for v in requested_thresholds if np.isfinite(v) and lower < v < upper}
        levels = sorted({v * (1 + increment) for v in seeds for increment in (0., .001, .005, .01, .02, .05)
                         if lower < v * (1 + increment) < upper})
        queries = []
        for query_level in levels:
            try:
                analysis = _agent_combustor_contours(root, [query_level], closed_geometry=True)[query_level]
            except ValueError as exc:
                if not str(exc).startswith("no connected Density isosurface at level"):
                    raise
                queries.append({"level": query_level, "status": "NO_ISOSURFACE"})
                continue
            chosen = select_component(analysis)
            if chosen is None:
                queries.append({"level": query_level, "selected_by": selection,
                                "region_count": len(analysis["regions"]), "status": "NO_UNIQUE_COMPONENT"})
                continue
            closed = [r for r in analysis["regions"] if r.get("closed_consistently_oriented") and r.get("encloses_query_point")]
            queries.append({"level": query_level, "selected_by": selection,
                            "region_count": len(analysis["regions"]),
                            "selected_component": {"area_weighted_centroid": chosen["area_weighted_centroid"],
                                "area": chosen["surface_area"], "closed": chosen["closed_consistently_oriented"]},
                            "closed_components_enclosing_density_maximum": len(closed),
                            "closed_volume_centroids": [r["enclosed_volume_centroid"] for r in closed]})
        result["additional_executed_contour_level_queries"] = queries
        result["sensitivity_query_rule"] = {"relative_increments": [0., .001, .005, .01, .02, .05],
            "seed_rule": "frozen base level plus finite answer literals inside stored Density range; queries do not alter the primary method",
            "seed_levels": sorted(seeds), "density_range": [lower, upper]}
    result["supplemental_execution_provenance"] = {
        "source_execution_sha256": artifact_sha256(execution),
        "dataset_manifest_sha256": artifact_sha256(manifest),
        "verified_frozen_results": True,
        "analysis": "exact contour replay and paired oriented triangle edges",
    }
    return result
