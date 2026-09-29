"""Blinded continuation packets for persisted pending evaluations.

The packet is an orchestration artifact, not a scientific judgment.  It is
deliberately derived from the pending record and model-visible case context;
Ground Truth, accepted reference membership, benchmark scores, and evaluator
identity are never copied into it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import importlib
import hashlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .evaluator import (
    EvaluationAdjudications,
    AdjudicationStatus,
    PendingCaseEvaluationRecord,
    SemanticMatchResolution,
    UnitFrameResolution,
    continuation_request_id,
)
from .schema import BenchmarkCaseInput
from .case_design import CaseConstructionMetadata


# Pending records are a persisted hand-off, not an implicit workflow engine.
# This small table is the single owner map used by readiness audits and keeps
# the materialize-first order for novel operationalizations explicit.
PENDING_OWNER_BY_TYPE: dict[str, str] = {
    "novel_operationalization_candidate": "NOVEL_O_COMPLETION",
    "novel_operationalization": "NOVEL_O_COMPLETION",
    "novel_operationalization_materialization": "NOVEL_O_MATERIALIZER",
    "novel_operationalization_scientific_adjudication": "BLINDED_SCIENTIFIC_ADJUDICATOR",
    "gt_outside_finding": "BLINDED_FINDING_ADJUDICATOR",
    "novel_finding": "BLINDED_FINDING_ADJUDICATOR",
    "semantic_uncertain": "SEMANTIC_MATCH_RESOLVER",
    "unit_relationship": "UNIT_FRAME_RESOLVER",
}

# Registration is intentionally separate from the owner map.  The map is a
# scientific routing contract; this registry describes which executable
# implementation a caller has explicitly installed for this process.
REQUIRED_PENDING_OWNERS = frozenset(PENDING_OWNER_BY_TYPE.values())
# Handler registration and production capability are deliberately separate.
# A protocol fixture may implement the packet boundary while being forbidden
# from claiming that it can produce a scientific continuation.
EXECUTION_CAPABILITY_PROTOCOL_ONLY = "PROTOCOL_ONLY"
EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC = "PRODUCTION_DETERMINISTIC"
EXECUTION_CAPABILITY_PRODUCTION_MODEL = "PRODUCTION_MODEL"
EXECUTION_CAPABILITIES = frozenset(
    {
        EXECUTION_CAPABILITY_PROTOCOL_ONLY,
        EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC,
        EXECUTION_CAPABILITY_PRODUCTION_MODEL,
    }
)
# The values describe the minimum capability needed for a production route;
# they are not a new workflow or scientific schema.
REQUIRED_OWNER_CAPABILITIES: dict[str, frozenset[str]] = {
    "NOVEL_O_COMPLETION": frozenset(
        {EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC, EXECUTION_CAPABILITY_PRODUCTION_MODEL}
    ),
    "NOVEL_O_MATERIALIZER": frozenset({EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC}),
    "BLINDED_SCIENTIFIC_ADJUDICATOR": frozenset({EXECUTION_CAPABILITY_PRODUCTION_MODEL}),
    "BLINDED_FINDING_ADJUDICATOR": frozenset({EXECUTION_CAPABILITY_PRODUCTION_MODEL}),
    "SEMANTIC_MATCH_RESOLVER": frozenset(
        {EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC, EXECUTION_CAPABILITY_PRODUCTION_MODEL}
    ),
    "UNIT_FRAME_RESOLVER": frozenset({EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC}),
}
MODEL_BACKED_PENDING_OWNERS = frozenset(
    {
        "BLINDED_SCIENTIFIC_ADJUDICATOR",
        "BLINDED_FINDING_ADJUDICATOR",
    }
)
PENDING_OWNER_MANIFEST_VERSION = "pending-continuation-owner-manifest-v1"
CONTINUATION_EXECUTION_MANIFEST_VERSION = "continuation-execution-manifest-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUEST_IDENTITY_SPEC: dict[str, tuple[str, str]] = {
    "novel_operationalization_candidate": (
        "operationalization_completion",
        "operationalization_completion_request",
    ),
    "novel_operationalization": (
        "operationalization_completion",
        "operationalization_completion_request",
    ),
    "novel_operationalization_materialization": (
        "operationalization_materialization",
        "continuation_request",
    ),
    "novel_operationalization_scientific_adjudication": (
        "operationalization_adjudication",
        "continuation_request",
    ),
    "gt_outside_finding": ("finding_adjudication", "continuation_request"),
    "novel_finding": ("finding_adjudication", "continuation_request"),
    "semantic_uncertain": ("semantic_match", "semantic_match_request"),
    "unit_relationship": ("unit_frame", "unit_frame_request"),
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolved_handler_identity(handler: Callable[..., Any]) -> dict[str, Any]:
    """Resolve an optional handler identity only when it is consumed.

    Production model-backed continuations depend on deployment configuration,
    but importing their module is also required by collection-free tests and
    static tooling.  A handler may therefore expose a lazy identity factory;
    loading the Python module remains configuration-free, while building a
    production execution manifest still fails closed if the identity cannot
    be resolved completely.
    """

    declared = getattr(handler, "__flowintentbench_identity__", None)
    identity = dict(declared) if isinstance(declared, Mapping) else {}
    factory = getattr(handler, "__flowintentbench_identity_factory__", None)
    if factory is None:
        return identity
    if not callable(factory):
        raise ValueError("continuation handler identity factory must be callable")
    try:
        resolved = factory()
    except Exception as exc:
        raise ValueError(
            "continuation handler identity could not be resolved: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(resolved, Mapping):
        raise ValueError("continuation handler identity factory must return an object")
    identity.update(dict(resolved))
    return identity


def _callable_identity(handler: Callable[..., Any]) -> dict[str, Any]:
    """Return a content-addressed identity for one installed owner.

    The continuation registry is intentionally generic, so this records the
    concrete callable rather than guessing whether it is model- or
    deterministic-based.  A source hash is included when the implementation
    is backed by a readable Python file; otherwise the stable module and
    qualified name still make the unresolved implementation explicit.
    """

    module = str(getattr(handler, "__module__", "") or "")
    qualname = str(
        getattr(handler, "__qualname__", getattr(handler, "__name__", "")) or ""
    )
    try:
        source_file = inspect.getsourcefile(handler) or inspect.getfile(handler)
    except (OSError, TypeError):
        source_file = None
    source_path = None if source_file is None else str(Path(source_file).resolve())
    source_sha256 = None
    if source_path:
        try:
            source_sha256 = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
        except OSError:
            source_sha256 = None
    descriptor: dict[str, Any] = {
        "implementation": f"{module}:{qualname}" if module and qualname else qualname,
        "module": module,
        "qualname": qualname,
        "source_sha256": source_sha256,
    }
    declared_identity = _resolved_handler_identity(handler)
    if declared_identity:
        # Optional provider/model/prompt identity supplied by a live owner.
        # It is metadata only; no secret or callable object can enter the
        # JSON manifest because the value is validated as JSON below.
        try:
            descriptor["declared_identity"] = json.loads(
                json.dumps(dict(declared_identity), ensure_ascii=False, allow_nan=False)
            )
            capability = descriptor["declared_identity"].get("execution_capability")
            if capability is not None:
                if capability not in EXECUTION_CAPABILITIES:
                    raise ValueError(
                        "continuation handler execution_capability must be one of: "
                        + ", ".join(sorted(EXECUTION_CAPABILITIES))
                    )
        except (TypeError, ValueError) as exc:
            raise ValueError("continuation handler identity must be JSON-compatible") from exc
    descriptor["implementation_sha256"] = _canonical_sha256(descriptor)
    return descriptor


def build_continuation_execution_manifest(
    handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> dict[str, Any]:
    """Build the immutable identity of explicitly installed continuations.

    This is configuration evidence, not a scientific result.  It is safe to
    persist in an evaluator manifest because it contains only executable
    identity and source hashes, never prompts, Ground Truth, or judgments.
    """

    if not isinstance(handlers, Mapping):
        raise ValueError("continuation handlers must be an object")
    unknown = set(handlers) - set(REQUIRED_PENDING_OWNERS)
    if unknown:
        raise ValueError(f"unknown pending continuation owner(s): {sorted(unknown)}")
    owners: dict[str, Any] = {}
    for owner, handler in sorted(handlers.items()):
        if not callable(handler):
            raise ValueError(f"continuation owner {owner!r} is not callable")
        descriptor = _callable_identity(handler)
        declared = descriptor.get("declared_identity")
        declared_capability = (
            declared.get("execution_capability")
            if isinstance(declared, Mapping)
            else None
        )
        if owner in MODEL_BACKED_PENDING_OWNERS or declared_capability == EXECUTION_CAPABILITY_PRODUCTION_MODEL:
            if not isinstance(declared, Mapping):
                raise ValueError(
                    f"model-backed continuation owner {owner!r} requires declared_identity"
                )
            required = ("provider", "model", "model_configuration", "prompt_sha256", "schema_sha256")
            missing = [key for key in required if key not in declared]
            if missing:
                raise ValueError(
                    f"model-backed continuation owner {owner!r} identity is missing: "
                    + ", ".join(missing)
                )
            for key in ("provider", "model"):
                if not isinstance(declared[key], str) or not declared[key].strip():
                    raise ValueError(
                        f"model-backed continuation owner {owner!r} identity field {key!r} must be non-empty"
                    )
            if not isinstance(declared["model_configuration"], Mapping):
                raise ValueError(
                    f"model-backed continuation owner {owner!r} model_configuration must be an object"
                )
            for key in ("prompt_sha256", "schema_sha256"):
                if not isinstance(declared[key], str) or not _SHA256_RE.fullmatch(declared[key]):
                    raise ValueError(
                        f"model-backed continuation owner {owner!r} {key} must be a SHA-256 digest"
                    )
        owners[str(owner)] = descriptor
    payload = {
        "schema_version": CONTINUATION_EXECUTION_MANIFEST_VERSION,
        "owners": owners,
        "owner_count": len(owners),
        "required_owner_count": len(REQUIRED_PENDING_OWNERS),
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def continuation_execution_manifest_digest(
    value: Mapping[str, Any],
) -> str:
    """Return the canonical digest of a continuation execution manifest."""

    if not isinstance(value, Mapping):
        raise ValueError("continuation execution manifest must be an object")
    payload = dict(value)
    if payload.get("schema_version") != CONTINUATION_EXECUTION_MANIFEST_VERSION:
        raise ValueError("unsupported continuation execution manifest schema version")
    owners = payload.get("owners")
    if not isinstance(owners, Mapping):
        raise ValueError("continuation execution manifest owners must be an object")
    if payload.get("owner_count") != len(owners):
        raise ValueError("continuation execution manifest owner_count is inconsistent")
    if payload.get("required_owner_count") != len(REQUIRED_PENDING_OWNERS):
        raise ValueError("continuation execution manifest required_owner_count is inconsistent")
    unknown = set(str(key) for key in owners) - set(REQUIRED_PENDING_OWNERS)
    if unknown:
        raise ValueError(
            "continuation execution manifest contains unknown owner(s): "
            + ", ".join(sorted(unknown))
        )
    for owner, descriptor in owners.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"continuation owner {owner!r} identity must be an object")
        implementation_digest = descriptor.get("implementation_sha256")
        if not isinstance(implementation_digest, str):
            raise ValueError(f"continuation owner {owner!r} lacks implementation_sha256")
        implementation_payload = dict(descriptor)
        implementation_payload.pop("implementation_sha256", None)
        if implementation_digest != _canonical_sha256(implementation_payload):
            raise ValueError(
                f"continuation owner {owner!r} implementation digest mismatch"
            )
        declared = descriptor.get("declared_identity")
        declared_capability = (
            declared.get("execution_capability")
            if isinstance(declared, Mapping)
            else None
        )
        if owner in MODEL_BACKED_PENDING_OWNERS or declared_capability == EXECUTION_CAPABILITY_PRODUCTION_MODEL:
            if not isinstance(declared, Mapping):
                raise ValueError(
                    f"model-backed continuation owner {owner!r} lacks declared_identity"
                )
            for key in ("provider", "model"):
                if not isinstance(declared.get(key), str) or not declared[key].strip():
                    raise ValueError(
                        f"model-backed continuation owner {owner!r} identity field {key!r} is invalid"
                    )
            if not isinstance(declared.get("model_configuration"), Mapping):
                raise ValueError(
                    f"model-backed continuation owner {owner!r} model_configuration is invalid"
                )
            for key in ("prompt_sha256", "schema_sha256"):
                if not isinstance(declared.get(key), str) or not _SHA256_RE.fullmatch(declared[key]):
                    raise ValueError(
                        f"model-backed continuation owner {owner!r} {key} is invalid"
                    )
    recorded = payload.pop("manifest_sha256", None)
    digest = _canonical_sha256(payload)
    if recorded is not None and recorded != digest:
        raise ValueError("continuation execution manifest digest mismatch")
    return digest


def validate_pending_owner_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an explicit owner implementation declaration.

    This is a *configuration* artifact, not a claim that an implementation
    performed a scientific adjudication.  Keeping the declaration separate
    from :class:`PendingContinuationRegistry` prevents a string in JSON from
    being mistaken for an installed callable.
    """

    if not isinstance(value, Mapping):
        raise ValueError("pending owner manifest must be a JSON object")
    if value.get("schema_version") != PENDING_OWNER_MANIFEST_VERSION:
        raise ValueError("unsupported pending owner manifest schema version")
    owners = value.get("owners")
    if not isinstance(owners, Mapping):
        raise ValueError("pending owner manifest must contain an owners object")
    normalized = {str(key).strip(): str(item).strip() for key, item in owners.items()}
    if any(not key or not item for key, item in normalized.items()):
        raise ValueError("pending owner manifest contains an empty owner or implementation")
    unknown = set(normalized) - REQUIRED_PENDING_OWNERS
    missing = REQUIRED_PENDING_OWNERS - set(normalized)
    if unknown:
        raise ValueError("unknown pending continuation owner(s): " + ", ".join(sorted(unknown)))
    if missing:
        raise ValueError("pending owner manifest is missing: " + ", ".join(sorted(missing)))
    return {
        "record_type": "PendingContinuationOwnerManifest",
        "schema_version": PENDING_OWNER_MANIFEST_VERSION,
        "owners": {key: normalized[key] for key in sorted(normalized)},
        "owner_count": len(normalized),
        "implementation_declarations_only": True,
    }


def load_pending_owner_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load pending owner manifest: {source}") from exc
    return validate_pending_owner_manifest(value)


def load_pending_continuation_registry(
    specifications: Mapping[str, str],
) -> "PendingContinuationRegistry":
    """Load explicitly configured owner handlers from ``module:function`` refs.

    The loader is intentionally opt-in and generic.  It never supplies a
    default scientific judge and never turns an owner declaration into a
    callable implicitly; every configured owner must resolve to a callable.
    """

    if not isinstance(specifications, Mapping):
        raise ValueError("pending handler specifications must be an object")
    handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {}
    for owner, reference in specifications.items():
        owner_name = str(owner).strip()
        ref = str(reference).strip()
        if ":" not in ref:
            raise ValueError(f"pending handler {owner_name!r} must use module:function")
        module_name, function_name = ref.rsplit(":", 1)
        if not module_name or not function_name:
            raise ValueError(f"invalid pending handler reference: {ref!r}")
        try:
            handler = getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot load pending handler {owner_name!r}: {ref}") from exc
        if not callable(handler):
            raise ValueError(f"pending handler {owner_name!r} is not callable: {ref}")
        handlers[owner_name] = handler
    return PendingContinuationRegistry(handlers)


@dataclass(frozen=True)
class PendingContinuationRegistry:
    """Small production registry for owner-specific pending continuations.

    The registry is deliberately not a workflow engine: it only binds the
    already-frozen owner names to callables and dispatches one packet at a
    time.  Scientific state transitions remain in ``CaseEvaluator``.
    """

    handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]]

    def __post_init__(self) -> None:
        if not isinstance(self.handlers, Mapping):
            raise ValueError("continuation handlers must be an object")
        unknown = set(self.handlers) - set(PENDING_OWNER_BY_TYPE.values())
        if unknown:
            raise ValueError(f"unknown pending continuation owner(s): {sorted(unknown)}")
        invalid = sorted(owner for owner, handler in self.handlers.items() if not callable(handler))
        if invalid:
            raise ValueError(
                "continuation owner handlers must be callable: " + ", ".join(invalid)
            )

    def missing_for_types(self, pending_types: Sequence[str]) -> tuple[str, ...]:
        owners = {pending_owner_for_type(str(item)) for item in pending_types}
        return tuple(sorted(owner for owner in owners if owner not in self.handlers))

    @property
    def installed_owners(self) -> tuple[str, ...]:
        return tuple(sorted(self.handlers))

    @property
    def execution_manifest(self) -> dict[str, Any]:
        """Content-addressed identity of the installed owner callables."""

        return build_continuation_execution_manifest(self.handlers)

    @property
    def production_capability_issues(self) -> tuple[dict[str, Any], ...]:
        """Return fail-closed issues for production continuation readiness.

        Presence of a callable is intentionally insufficient.  Missing
        owners, undeclared capability, and ``PROTOCOL_ONLY`` fixtures are all
        explicit issues.  The result is diagnostic configuration evidence and
        never a scientific judgment.
        """

        issues: list[dict[str, Any]] = []
        for owner in sorted(REQUIRED_PENDING_OWNERS):
            handler = self.handlers.get(owner)
            if handler is None:
                issues.append({"owner": owner, "reason": "MISSING_HANDLER"})
                continue
            try:
                declared = _resolved_handler_identity(handler)
            except ValueError as exc:
                issues.append(
                    {
                        "owner": owner,
                        "reason": "IDENTITY_UNRESOLVED",
                        "error": str(exc),
                    }
                )
                continue
            capability = declared.get("execution_capability")
            allowed = REQUIRED_OWNER_CAPABILITIES.get(owner, frozenset())
            if capability is None:
                issues.append({"owner": owner, "reason": "CAPABILITY_UNDECLARED"})
            elif capability not in allowed:
                issues.append(
                    {
                        "owner": owner,
                        "reason": "CAPABILITY_INSUFFICIENT",
                        "execution_capability": capability,
                        "required_capabilities": sorted(allowed),
                    }
                )
        return tuple(issues)

    @property
    def production_ready(self) -> bool:
        return not self.production_capability_issues

    def require_for_types(self, pending_types: Sequence[str]) -> None:
        missing = self.missing_for_types(pending_types)
        if missing:
            raise ValueError(
                "pending continuation handlers are not installed: "
                + ", ".join(missing)
            )

    def dispatch(
        self,
        record: PendingCaseEvaluationRecord,
        case_input: BenchmarkCaseInput | Mapping[str, Any],
        case_metadata: CaseConstructionMetadata | Mapping[str, Any],
    ) -> dict[str, Any]:
        return dispatch_pending_continuation(
            record,
            case_input,
            case_metadata,
            self.handlers,
        )


def pending_owner_for_type(pending_type: str, *, next_action: str | None = None) -> str:
    """Return the explicit continuation owner for a persisted pending type.

    ``MATERIALIZATION_CONTEXT_REQUIRED`` is intentionally owned by the
    materializer even when a legacy record was labelled as a scientific
    adjudication: no judge may run before an actual ``G(O)`` proof exists.
    Unknown types fail closed rather than being routed to a generic guesser.
    """

    normalized = str(pending_type or "").strip()
    if next_action == "MATERIALIZATION_CONTEXT_REQUIRED":
        return "NOVEL_O_MATERIALIZER"
    try:
        return PENDING_OWNER_BY_TYPE[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown pending continuation type: {normalized!r}") from exc


def _prediction_payload(record: PendingCaseEvaluationRecord) -> dict[str, Any]:
    prediction = record.extracted_prediction
    if prediction is None:
        return {"operationalization": None, "findings": []}
    return {
        "operationalization": {
            "decisions": [
                {
                    "dimension": decision.dimension.value,
                    "status": decision.status.value,
                    "normalized_statement": decision.normalized_statement,
                }
                for decision in prediction.operationalization.decisions
            ]
        },
        "findings": [
            {
                "prediction_id": finding.prediction_id,
                "statement": finding.statement,
                "value": finding.value,
                "unit": finding.unit,
            }
            for finding in prediction.findings
        ],
    }


def build_pending_adjudication_packet(
    record: PendingCaseEvaluationRecord,
    case_input: BenchmarkCaseInput | Mapping[str, Any],
    case_metadata: CaseConstructionMetadata | Mapping[str, Any],
) -> dict[str, Any]:
    """Build the next-action packet without copying GT or evaluator outputs.

    ``gt_outside_finding`` and novel-O pending routes expose only the response
    claims requiring scientific review.  Other pending types carry a compact
    diagnostic and are routed to their owning evaluator component rather than
    being guessed by this helper.
    """

    if not isinstance(record, PendingCaseEvaluationRecord):
        raise TypeError("record must be PendingCaseEvaluationRecord")
    if isinstance(case_input, BenchmarkCaseInput):
        model_input = case_input
    else:
        model_input = BenchmarkCaseInput.model_validate(case_input)
    if isinstance(case_metadata, CaseConstructionMetadata):
        metadata = case_metadata
    else:
        metadata = CaseConstructionMetadata.model_validate(case_metadata)
    if metadata.case_id != record.case_id:
        raise ValueError("pending packet case_id does not match case metadata")

    pending_type = record.pending_type
    continuation = dict(record.continuation_context)
    if record.continuation_request_id and "continuation_request_id" not in continuation:
        continuation["continuation_request_id"] = record.continuation_request_id
    if pending_type in {
        "novel_operationalization_candidate",
        "novel_operationalization",
    }:
        next_action = "COMPLETE_NOVEL_OPERATIONALIZATION"
        payload = {
            "candidate_operationalization": _prediction_payload(record)[
                "operationalization"
            ],
            "candidate_findings": [],
            "continuation_request": continuation.get("continuation_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    elif pending_type == "novel_operationalization_materialization":
        next_action = "MATERIALIZE_NOVEL_OPERATIONALIZATION"
        payload = {
            "candidate_operationalization": continuation.get(
                "candidate_operationalization",
                _prediction_payload(record)["operationalization"],
            ),
            "candidate_findings": [],
            "continuation_request": continuation.get("continuation_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    elif pending_type == "novel_operationalization_scientific_adjudication":
        # Blind adjudication is legal only after the route has persisted the
        # actual execution result.  A hand-built/legacy pending record without
        # that proof is routed back to materialization rather than allowing a
        # judge to reason about an unexecuted O.
        has_materialization = (
            continuation.get("execution_status") == "MATERIALIZED"
            and isinstance(continuation.get("G_of_O"), Mapping)
            and bool(continuation.get("materialization_id"))
        )
        next_action = (
            "BLINDED_SCIENTIFIC_ADJUDICATION"
            if has_materialization
            else "MATERIALIZATION_CONTEXT_REQUIRED"
        )
        payload = {
            "candidate_operationalization": continuation.get(
                "candidate_operationalization",
                _prediction_payload(record)["operationalization"],
            ),
            "materialization_id": continuation.get("materialization_id"),
            "execution_status": continuation.get("execution_status"),
            "G_of_O": continuation.get("G_of_O") if has_materialization else None,
            "G_of_O_sha256": continuation.get("G_of_O_sha256") if has_materialization else None,
            "candidate_findings": [],
            "continuation_request": continuation.get("continuation_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    elif pending_type in {"gt_outside_finding", "novel_finding"}:
        next_action = "BLINDED_FINDING_ADJUDICATION"
        payload = {
            "candidate_operationalization": _prediction_payload(record)["operationalization"],
            # Reusing a completed independent verdict must not expose it to
            # fresh stochastic adjudication while another Finding is pending.
            "candidate_findings": [
                item for item in _prediction_payload(record)["findings"]
                if record.accumulated_adjudications.novel_findings.get(
                    (continuation.get("branch_id"), item["prediction_id"])
                ) not in {AdjudicationStatus.ACCEPTED, AdjudicationStatus.REJECTED}
            ],
            "branch_id": continuation.get("branch_id"),
            "materialized_G_of_O": continuation.get("materialized_G_of_O"),
            "allowed_finding_roles": continuation.get(
                "allowed_finding_roles", []
            ),
            "continuation_request": continuation.get("continuation_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    elif pending_type == "semantic_uncertain":
        next_action = "RESOLVE_SEMANTIC_MATCH"
        payload = {
            "candidate_operationalization": _prediction_payload(record)["operationalization"],
            "candidate_findings": _prediction_payload(record)["findings"],
            "semantic_match_request": continuation.get("semantic_match_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    elif pending_type == "unit_relationship":
        next_action = "RESOLVE_UNIT_FRAME"
        payload = {
            "candidate_operationalization": None,
            "candidate_findings": [],
            "unit_frame_request": continuation.get("unit_frame_request"),
            "continuation_request_id": continuation.get("continuation_request_id"),
        }
    else:
        next_action = "OWNER_SPECIFIC_EVALUATOR_CONTINUATION"
        payload = {
            "candidate_operationalization": None,
            "candidate_findings": [],
        }

    pending_owner = pending_owner_for_type(pending_type, next_action=next_action)

    request_identity_status = "PRESENT"
    request_spec = _REQUEST_IDENTITY_SPEC.get(pending_type)
    if request_spec is not None:
        request_kind, request_key = request_spec
        request_value = continuation.get(request_key)
        # Completion records historically used a named request key, while
        # all newer materialization/adjudication records use the generic
        # continuation_request field.  Never synthesize either from claims.
        if request_value is None and request_key != "continuation_request":
            request_value = continuation.get("continuation_request")
        request_value_id = continuation.get("continuation_request_id")
        if not isinstance(request_value, Mapping) or not isinstance(request_value_id, str):
            # Historical pending artifacts remain readable for archive/routing
            # audits, but cannot be resumed as scientific input until a fresh
            # evaluation writes the structured request.
            request_identity_status = "LEGACY_MISSING"
        else:
            expected_id = continuation_request_id(request_kind, request_value)
            if request_value_id != expected_id:
                request_identity_status = "INVALID"

    # A persisted record produced by an older evaluator may lack request
    # identity.  New semantic/unit/finding pending records are required to
    # carry it; expose the absence explicitly so production orchestration can
    # fail closed instead of treating unrelated requests as one cycle.
    continuation_request = continuation.get("continuation_request_id")

    return {
        "packet_type": "pending_evaluation_adjudication_v1",
        "case_id": record.case_id,
        "condition": record.condition,
        "pending_type": pending_type,
        "pending_reason": record.pending_reason,
        "next_action": next_action,
        "pending_owner": pending_owner,
        "scientific_score_blocked": True,
        "scientific_question": model_input.scientific_question,
        "case_context": model_input.case_context.model_dump(mode="json"),
        "finding_goal": metadata.finding_goal,
        "model_claims": payload,
        "continuation_context": continuation,
        "continuation_request_id": continuation_request,
        "continuation_request_identity_status": request_identity_status,
    "forbidden_fields": [
            "ground_truth",
            "reference_findings",
            "accepted_operationalizations",
            "reference_membership",
            "benchmark_scores",
            "evaluator_identity",
        ],
    }


def route_pending_adjudication(
    record: PendingCaseEvaluationRecord,
    case_input: BenchmarkCaseInput | Mapping[str, Any],
    case_metadata: CaseConstructionMetadata | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic hand-off envelope without invoking a model.

    The function deliberately performs no materialization, adjudication, or
    mutation.  It only records who owns the next action and whether the
    scientific score remains blocked.
    """

    packet = build_pending_adjudication_packet(record, case_input, case_metadata)
    return {
        "route_status": "ROUTABLE",
        "pending_owner": packet["pending_owner"],
        "next_action": packet["next_action"],
        "scientific_score_blocked": True,
        "packet": packet,
    }


def dispatch_pending_continuation(
    record: PendingCaseEvaluationRecord,
    case_input: BenchmarkCaseInput | Mapping[str, Any],
    case_metadata: CaseConstructionMetadata | Mapping[str, Any],
    handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]],
) -> dict[str, Any]:
    """Dispatch one pending packet to its declared owner.

    This is intentionally a bounded, single-hop dispatch primitive.  It does
    not loop, mutate records, select a model, or infer a scientific verdict.
    A handler receives only the blinded packet and may return an explicit
    continuation payload (normally ``EvaluationAdjudications``); callers
    persist that payload and invoke the existing evaluator finalizer.  Missing
    handlers leave the record pending and are reported as such.
    """

    route = route_pending_adjudication(record, case_input, case_metadata)
    owner = route["pending_owner"]
    handler = handlers.get(owner)
    if handler is None:
        return {
            **route,
            "dispatch_status": "PENDING_OWNER_NOT_EXECUTED",
            "continuation_output": None,
        }
    packet = route["packet"]
    request_status = packet.get("continuation_request_identity_status")
    if request_status in {"LEGACY_MISSING", "INVALID"}:
        # Historical packets remain readable for archive audits, but a
        # production continuation may not hand an unbound request to an
        # owner.  A fresh evaluation must recreate the typed request first.
        return {
            **route,
            "dispatch_status": "PENDING_REQUEST_IDENTITY_INVALID",
            "continuation_output": None,
        }
    output = handler(route["packet"])
    if output is not None and not isinstance(
        output,
        (
            Mapping,
            PendingCaseEvaluationRecord,
            EvaluationAdjudications,
            SemanticMatchResolution,
            UnitFrameResolution,
        ),
    ):
        raise TypeError(
            "pending continuation handlers must return a mapping, "
            "EvaluationAdjudications-compatible or typed-resolution payload, or None"
        )
    return {
        **route,
        "dispatch_status": "HANDLER_RETURNED" if output is not None else "PENDING",
        "continuation_output": output,
    }


__all__ = [
    "PENDING_OWNER_BY_TYPE",
    "REQUIRED_PENDING_OWNERS",
    "MODEL_BACKED_PENDING_OWNERS",
    "EXECUTION_CAPABILITY_PROTOCOL_ONLY",
    "EXECUTION_CAPABILITY_PRODUCTION_DETERMINISTIC",
    "EXECUTION_CAPABILITY_PRODUCTION_MODEL",
    "EXECUTION_CAPABILITIES",
    "REQUIRED_OWNER_CAPABILITIES",
    "PENDING_OWNER_MANIFEST_VERSION",
    "CONTINUATION_EXECUTION_MANIFEST_VERSION",
    "PendingContinuationRegistry",
    "validate_pending_owner_manifest",
    "load_pending_owner_manifest",
    "build_continuation_execution_manifest",
    "continuation_execution_manifest_digest",
    "load_pending_continuation_registry",
    "build_pending_adjudication_packet",
    "pending_owner_for_type",
    "route_pending_adjudication",
    "dispatch_pending_continuation",
]
