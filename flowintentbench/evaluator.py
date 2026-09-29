"""Scientific evaluator orchestration for FlowIntentBench.

The evaluator keeps natural-language extraction and semantic judgment behind
replaceable, versioned interfaces.  Everything after those judgments is
deterministic: status scoring, branch matching, value verification, one-to-one
Finding credit, O-conditioned consistency, and the explicit trial -> case ->
benchmark aggregation hierarchy.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import statistics
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .case_design import (
    CaseConstructionMetadata,
    OperationalizationDimension,
    primary_case_type,
)
from .evaluation_metrics import (
    BenchmarkMetricValues,
    CaseAggregateMetricResult,
    CaseMetricInput,
    CaseMetricResult,
    ConsistencyEvaluation,
    ConsistencyMetricResult,
    EFFICIENCY_METRIC_NAMES,
    EfficiencyObservation,
    EfficiencyAggregate,
    FindingBranchEvaluation,
    FindingMetricResult,
    OperationalizationBranchEvaluation,
    OperationalizationMetricResult,
    benchmark_summary,
    compute_case_metrics,
)
from .ground_truth import (
    FindingImportance,
    FindingValue,
    GroundTruth,
    GroundTruthValidator,
    OperationalizationBundle,
    OperationalizationFindingBranch,
    ReferenceFinding,
)
from .finding_requirements import FindingRequirementContract, evaluate_adequate_core_sets
from .evaluation_policy import policy_index
from .srac import ScientificEvaluationContract
from .experiment import FORMAL_TRIAL_COUNT
from .model_runner import RunRecord, RunStatus
from .schema import BenchmarkCaseInput


class EvaluationConfigurationError(ValueError):
    """The benchmark/evaluator inputs cannot produce a formal score."""


class EvaluationPendingAdjudication(RuntimeError):
    """A formal score cannot be finalized until a judgment is resolved."""

    def __init__(
        self,
        message: str,
        *,
        pending_type: str = "adjudication",
        continuation_context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.pending_type = pending_type
        # Context is an optional, model-visible continuation artifact.  It is
        # never a scientific verdict and must not contain GT/reference fields.
        self.continuation_context = (
            {} if continuation_context is None else dict(continuation_context)
        )


class ExtractionStatus(str, Enum):
    EXTRACTED = "EXTRACTED"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"


class SemanticMatchResult(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    UNCERTAIN = "UNCERTAIN"


class AdjudicationStatus(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class SemanticMatchPurpose(str, Enum):
    OPERATIONALIZATION = "operationalization"
    FINDING = "finding"
    FINDING_DEDUPLICATION = "finding_deduplication"


class FindingVerificationMode(str, Enum):
    """Explicit GT value rule exposed to Finding semantic matching."""

    SEMANTIC_ONLY = "semantic_only"
    EXACT_DISCRETE_NUMERIC = "exact_discrete_numeric"
    SCALAR_TOLERANCE = "scalar_tolerance"
    SPATIAL_EUCLIDEAN = "spatial_euclidean"
    COMPONENTWISE_VECTOR = "componentwise_vector"
    EXACT_IDENTITY = "exact_identity"


EvidenceSpan = str | tuple[int, int] | None


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_evidence_span(value: EvidenceSpan) -> None:
    if value is None or isinstance(value, str):
        return
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= value[0]
    ):
        raise ValueError("evidence_span must be text, a non-empty offset range, or None")


@dataclass(frozen=True)
class ExtractedOperationalizationDecision:
    dimension: OperationalizationDimension
    status: ExtractionStatus
    normalized_statement: str | None = None
    evidence_span: EvidenceSpan = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self, "dimension", OperationalizationDimension(self.dimension)
            )
            object.__setattr__(self, "status", ExtractionStatus(self.status))
        except ValueError as exc:
            raise ValueError("invalid O dimension or extraction status") from exc
        if self.status == ExtractionStatus.EXTRACTED:
            _text(self.normalized_statement, "normalized_statement")
        elif self.normalized_statement is not None:
            _text(self.normalized_statement, "normalized_statement")
        _validate_evidence_span(self.evidence_span)


@dataclass(frozen=True)
class ExtractedOperationalization:
    decisions: tuple[ExtractedOperationalizationDecision, ...]

    def __post_init__(self) -> None:
        dimensions = [decision.dimension for decision in self.decisions]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("extracted O decisions must have unique dimensions")


@dataclass(frozen=True)
class PredictedAtomicFinding:
    prediction_id: str
    statement: str
    value: FindingValue = None
    unit: str | None = None
    evidence_span: EvidenceSpan = None

    def __post_init__(self) -> None:
        _text(self.prediction_id, "prediction_id")
        _text(self.statement, "statement")
        object.__setattr__(
            self, "value", _validate_predicted_finding_value(self.value)
        )
        if self.unit is not None:
            _text(self.unit, "unit")
        _validate_evidence_span(self.evidence_span)


def _validate_predicted_finding_value(value: FindingValue) -> FindingValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ValueError("PredictedAtomicFinding.value does not accept bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("PredictedAtomicFinding.value floats must be finite")
        return value
    if not isinstance(value, list):
        raise ValueError(
            "PredictedAtomicFinding.value must be str, int, finite float, "
            "list[finite numeric values], list[str], or None"
        )
    if all(isinstance(item, str) for item in value):
        return value
    if all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in value
    ):
        return [float(item) for item in value]
    raise ValueError(
        "PredictedAtomicFinding.value lists must contain only strings or finite numeric values"
    )


@dataclass(frozen=True)
class FindingEligibility:
    """Response-level judgments used to construct the frozen F_app set."""

    scientifically_interpretable: bool
    relevant_to_finding_goal: bool
    in_principle_verifiable: bool

    def __post_init__(self) -> None:
        for field_name in (
            "scientifically_interpretable",
            "relevant_to_finding_goal",
            "in_principle_verifiable",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be bool")

    @property
    def in_f_app(self) -> bool:
        return (
            self.scientifically_interpretable
            and self.relevant_to_finding_goal
            and self.in_principle_verifiable
        )


@dataclass(frozen=True)
class ExtractedPrediction:
    operationalization: ExtractedOperationalization
    findings: tuple[PredictedAtomicFinding, ...]

    def __post_init__(self) -> None:
        prediction_ids = [finding.prediction_id for finding in self.findings]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("prediction_id values must be unique before deduplication")


@dataclass(frozen=True)
class ExtractionRequest:
    scientific_question: str
    case_context: Mapping[str, Any]
    principal_operationalization_dimensions: tuple[OperationalizationDimension, ...]
    unresolved_operationalization_dimensions: tuple[OperationalizationDimension, ...]
    final_response: str
    # Reader/data metadata is explicit authority for coordinate-frame labels;
    # it is never inferred from a Python value or natural-language phrase.
    coordinate_semantics: Mapping[str, Any] | None = None


class PredictionExtractor(Protocol):
    """GT-blind natural-language extraction component."""

    def extract(self, request: ExtractionRequest) -> ExtractedPrediction:
        """Extract explicit O decisions and atomic Findings from one response."""


@dataclass(frozen=True)
class FindingEligibilityRequest:
    scientific_question: str
    case_context: Mapping[str, Any]
    finding_goal: str
    finding: PredictedAtomicFinding


class FindingEligibilityJudge(Protocol):
    """Judge response-level F_app eligibility independently of extraction."""

    def judge(self, request: FindingEligibilityRequest) -> FindingEligibility:
        """Return the three frozen response-level eligibility judgments."""


@dataclass(frozen=True)
class SemanticMatchRequest:
    purpose: SemanticMatchPurpose
    predicted_statement: str
    reference_statement: str
    predicted_id: str
    reference_id: str
    branch_id: str | None = None
    dimension: OperationalizationDimension | None = None
    scientific_question: str = ""
    case_context: Mapping[str, Any] = field(default_factory=dict)
    finding_goal: str | None = None
    finding_category: str | None = None
    predicted_value: FindingValue = None
    predicted_unit: str | None = None
    reference_value: FindingValue = None
    reference_unit: str | None = None
    verification_mode: FindingVerificationMode = FindingVerificationMode.SEMANTIC_ONLY
    predicted_evidence_span: EvidenceSpan = None

    def __post_init__(self) -> None:
        _validate_evidence_span(self.predicted_evidence_span)
        object.__setattr__(
            self, "purpose", SemanticMatchPurpose(self.purpose)
        )
        object.__setattr__(
            self,
            "verification_mode",
            FindingVerificationMode(self.verification_mode),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize one semantic request for a typed continuation hand-off.

        This is an evaluator request, not a scientific finding.  Keeping the
        complete request in the pending artifact lets a resolver prove that
        its answer belongs to exactly the unresolved comparison that created
        the pending record.
        """

        result = {
            "purpose": self.purpose.value,
            "predicted_statement": self.predicted_statement,
            "reference_statement": self.reference_statement,
            "predicted_id": self.predicted_id,
            "reference_id": self.reference_id,
            "branch_id": self.branch_id,
            "dimension": None if self.dimension is None else self.dimension.value,
            "scientific_question": self.scientific_question,
            "case_context": _json_value(self.case_context, "case_context"),
            "finding_goal": self.finding_goal,
            "finding_category": self.finding_category,
            "predicted_value": self.predicted_value,
            "predicted_unit": self.predicted_unit,
            "reference_value": self.reference_value,
            "reference_unit": self.reference_unit,
            "verification_mode": self.verification_mode.value,
        }
        # Omit absent evidence so persisted legacy request identities remain
        # stable. Preserve exact text or raw offsets; no source is bound here
        # from which an offset range could safely be resolved into text.
        if self.predicted_evidence_span is not None:
            result["predicted_evidence_span"] = (
                list(self.predicted_evidence_span)
                if isinstance(self.predicted_evidence_span, tuple)
                else self.predicted_evidence_span
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticMatchRequest":
        request = dict(value)
        if isinstance(request.get("predicted_evidence_span"), list):
            request["predicted_evidence_span"] = tuple(request["predicted_evidence_span"])
        request["purpose"] = SemanticMatchPurpose(request["purpose"])
        if request.get("dimension") is not None:
            request["dimension"] = OperationalizationDimension(request["dimension"])
        request["verification_mode"] = FindingVerificationMode(
            request.get("verification_mode", FindingVerificationMode.SEMANTIC_ONLY)
        )
        return cls(**request)


@dataclass(frozen=True)
class SemanticMatchResolution:
    """Typed answer for one persisted ``semantic_uncertain`` request."""

    request_id: str
    request: SemanticMatchRequest
    result: SemanticMatchResult

    def __post_init__(self) -> None:
        _text(self.request_id, "semantic resolution request_id")
        object.__setattr__(self, "result", SemanticMatchResult(self.result))
        if self.result not in {
            SemanticMatchResult.MATCH,
            SemanticMatchResult.NO_MATCH,
        }:
            raise EvaluationConfigurationError(
                "semantic continuation result must resolve to MATCH or NO_MATCH"
            )
        expected = continuation_request_id("semantic_match", self.request.to_dict())
        if self.request_id != expected:
            raise EvaluationConfigurationError(
                "semantic resolution request_id does not match its request"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "result": self.result.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticMatchResolution":
        request = value.get("request")
        if not isinstance(request, Mapping):
            raise EvaluationConfigurationError(
                "semantic resolution must contain a request object"
            )
        return cls(
            request_id=str(value.get("request_id", "")),
            request=SemanticMatchRequest.from_dict(request),
            result=SemanticMatchResult(value.get("result")),
        )


@dataclass(frozen=True)
class UnitFrameResolution:
    """Typed answer for one persisted ``unit_relationship`` request.

    ``canonical_value`` is already expressed in the reference representation
    when ``status`` is ``RESOLVED``.  The runtime never infers a conversion
    from a Python value or an unbound unit label.
    """

    request_id: str
    request: Mapping[str, Any]
    status: str
    canonical_value: FindingValue = None
    canonical_unit: str | None = None
    rule: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.request_id, "unit/frame resolution request_id")
        normalized_status = _text(self.status, "unit/frame resolution status").upper()
        if normalized_status not in {"RESOLVED", "NO_MATCH"}:
            raise EvaluationConfigurationError(
                "unit/frame resolution status must be RESOLVED or NO_MATCH"
            )
        object.__setattr__(self, "status", normalized_status)
        object.__setattr__(
            self,
            "request",
            _json_value(dict(self.request), "unit/frame resolution request"),
        )
        expected = continuation_request_id("unit_frame", self.request)
        if self.request_id != expected:
            raise EvaluationConfigurationError(
                "unit/frame resolution request_id does not match its request"
            )
        object.__setattr__(
            self,
            "canonical_value",
            _validate_predicted_finding_value(self.canonical_value),
        )
        object.__setattr__(
            self,
            "provenance",
            _json_value(dict(self.provenance), "unit/frame resolution provenance"),
        )
        if self.canonical_unit is not None:
            _text(self.canonical_unit, "unit/frame resolution canonical_unit")
        if self.status == "RESOLVED" and not str(self.rule).strip():
            raise EvaluationConfigurationError(
                "resolved unit/frame result requires an explicit rule"
            )
        if self.status == "RESOLVED" and self.canonical_value is None:
            raise EvaluationConfigurationError(
                "resolved unit/frame result requires canonical_value"
            )
        if self.status == "RESOLVED" and not self.provenance:
            raise EvaluationConfigurationError(
                "resolved unit/frame result requires provenance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request": dict(self.request),
            "status": self.status,
            "canonical_value": self.canonical_value,
            "canonical_unit": self.canonical_unit,
            "rule": self.rule,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnitFrameResolution":
        request = value.get("request")
        if not isinstance(request, Mapping):
            raise EvaluationConfigurationError(
                "unit/frame resolution must contain a request object"
            )
        return cls(
            request_id=str(value.get("request_id", "")),
            request=request,
            status=str(value.get("status", "")),
            canonical_value=value.get("canonical_value"),
            canonical_unit=value.get("canonical_unit"),
            rule=str(value.get("rule", "")),
            provenance=value.get("provenance", {}),
        )


class SemanticMatcher(Protocol):
    """One semantic-equivalence interface shared by O and Findings."""

    def match(self, request: SemanticMatchRequest) -> SemanticMatchResult:
        """Return MATCH, NO_MATCH, or UNCERTAIN for one semantic pair."""


UnitConverter = Callable[[FindingValue, str, str], FindingValue | None]


def _json_value(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


def _canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the same canonical form as execution artifacts."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


CONTINUATION_REQUEST_ID_VERSION = "continuation-request-v1"


def continuation_request_id(kind: str, payload: Mapping[str, Any]) -> str:
    """Return the stable identity of one unresolved continuation request.

    Request identity is deliberately derived from the concrete unresolved
    comparison, not merely from its pending type.  This prevents two semantic
    or unit requests in the same case from being mistaken for a cycle.
    """

    normalized_kind = _text(kind, "continuation request kind")
    canonical = {
        "schema_version": CONTINUATION_REQUEST_ID_VERSION,
        "kind": normalized_kind,
        "payload": _json_value(dict(payload), "continuation request payload"),
    }
    return f"{normalized_kind}:{_canonical_json_sha256(canonical)}"


@dataclass(frozen=True)
class EvaluatorComponentIdentity:
    implementation_id: str
    version: str
    provider: str | None = None
    model: str | None = None
    model_configuration: Mapping[str, Any] = field(default_factory=dict)
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        _text(self.implementation_id, "implementation_id")
        _text(self.version, "version")
        for field_name in ("provider", "model", "prompt_version"):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        object.__setattr__(
            self,
            "model_configuration",
            _json_value(dict(self.model_configuration), "model_configuration"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_id": self.implementation_id,
            "version": self.version,
            "provider": self.provider,
            "model": self.model,
            "model_configuration": dict(self.model_configuration),
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluatorComponentIdentity":
        return cls(**dict(value))


@dataclass(frozen=True)
class EvaluationManifest:
    """One fixed evaluator backend and one fixed specification per experiment."""

    evaluator_backend: EvaluatorComponentIdentity
    extraction_spec_id: str
    eligibility_spec_id: str
    semantic_match_spec_id: str
    unit_converter_version: str | None = None
    # Content-addressed deterministic evaluator semantics.  This is kept as
    # an optional mapping for loading legacy manifests, while every newly
    # created backend manifest records the implementation contract explicitly.
    deterministic_contract_hashes: Mapping[str, Any] = field(default_factory=dict)
    # Optional content-addressed identity of explicitly installed pending
    # continuation implementations.  Empty preserves legacy/offline
    # manifests; a live continuation run must bind this before evaluation.
    continuation_execution_manifest: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator_backend, EvaluatorComponentIdentity):
            object.__setattr__(
                self,
                "evaluator_backend",
                EvaluatorComponentIdentity.from_dict(self.evaluator_backend),
            )
        for field_name in (
            "extraction_spec_id",
            "eligibility_spec_id",
            "semantic_match_spec_id",
        ):
            _text(getattr(self, field_name), field_name)
        if self.unit_converter_version is not None:
            _text(self.unit_converter_version, "unit_converter_version")
        object.__setattr__(
            self,
            "deterministic_contract_hashes",
            _json_value(dict(self.deterministic_contract_hashes), "deterministic_contract_hashes"),
        )
        object.__setattr__(
            self,
            "continuation_execution_manifest",
            _json_value(
                dict(self.continuation_execution_manifest),
                "continuation_execution_manifest",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "evaluator_backend": self.evaluator_backend.to_dict(),
            "extraction_spec_id": self.extraction_spec_id,
            "eligibility_spec_id": self.eligibility_spec_id,
            "semantic_match_spec_id": self.semantic_match_spec_id,
            "unit_converter_version": self.unit_converter_version,
        }
        # Keep legacy manifest byte identity when no content-addressed
        # contract map was authored; new backend manifests always populate it.
        if self.deterministic_contract_hashes:
            value["deterministic_contract_hashes"] = dict(
                self.deterministic_contract_hashes
            )
        if self.continuation_execution_manifest:
            value["continuation_execution_manifest"] = dict(
                self.continuation_execution_manifest
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationManifest":
        return cls(
            evaluator_backend=EvaluatorComponentIdentity.from_dict(
                value["evaluator_backend"]
            ),
            extraction_spec_id=value["extraction_spec_id"],
            eligibility_spec_id=value["eligibility_spec_id"],
            semantic_match_spec_id=value["semantic_match_spec_id"],
            unit_converter_version=value.get("unit_converter_version"),
            deterministic_contract_hashes=value.get("deterministic_contract_hashes", {}),
            continuation_execution_manifest=value.get(
                "continuation_execution_manifest", {}
            ),
        )


def evaluation_manifest_digest(
    manifest: EvaluationManifest | Mapping[str, Any]
) -> str:
    """Return the canonical identity of one evaluator configuration."""

    validated = (
        manifest
        if isinstance(manifest, EvaluationManifest)
        else EvaluationManifest.from_dict(manifest)
    )
    payload = json.dumps(
        validated.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_evaluation_manifest(
    manifest: EvaluationManifest, path: str | Path
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_evaluation_adjudications(
    adjudications: EvaluationAdjudications | Mapping[str, Any],
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> EvaluationAdjudications:
    validated = (
        adjudications
        if isinstance(adjudications, EvaluationAdjudications)
        else EvaluationAdjudications.from_dict(adjudications)
    )
    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite adjudications: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validated


def load_evaluation_adjudications(path: str | Path) -> EvaluationAdjudications:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load adjudications from {source}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("adjudications root must be an object")
    return EvaluationAdjudications.from_dict(value)


def save_run_evaluation_adjudication(
    adjudication: RunEvaluationAdjudication | Mapping[str, Any],
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> RunEvaluationAdjudication:
    validated = (
        adjudication
        if isinstance(adjudication, RunEvaluationAdjudication)
        else RunEvaluationAdjudication.from_dict(adjudication)
    )
    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite run adjudication: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validated


def load_run_evaluation_adjudication(
    path: str | Path,
) -> RunEvaluationAdjudication:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load run adjudication from {source}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("run adjudication root must be an object")
    return RunEvaluationAdjudication.from_dict(value)


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evaluation manifest from {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("evaluation manifest root must be an object")
    return EvaluationManifest.from_dict(value)


@dataclass(frozen=True)
class SemanticMatchJudgment:
    request: SemanticMatchRequest
    result: SemanticMatchResult

    def to_dict(self) -> dict[str, Any]:
        request = self.request
        return {
            "request": {
                "purpose": request.purpose.value,
                "predicted_statement": request.predicted_statement,
                "reference_statement": request.reference_statement,
                "predicted_id": request.predicted_id,
                "reference_id": request.reference_id,
                "branch_id": request.branch_id,
                "dimension": (
                    None if request.dimension is None else request.dimension.value
                ),
                "scientific_question": request.scientific_question,
                "case_context": _json_value(request.case_context, "case_context"),
                "finding_goal": request.finding_goal,
                "finding_category": request.finding_category,
                "predicted_value": request.predicted_value,
                "predicted_unit": request.predicted_unit,
                "reference_value": request.reference_value,
                "reference_unit": request.reference_unit,
                "verification_mode": request.verification_mode.value,
            },
            "result": self.result.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticMatchJudgment":
        return cls(
            request=SemanticMatchRequest.from_dict(value["request"]),
            result=SemanticMatchResult(value["result"]),
        )


@dataclass(frozen=True)
class DeduplicationJudgment:
    prediction_id: str
    representative_prediction_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "prediction_id": self.prediction_id,
            "representative_prediction_id": self.representative_prediction_id,
        }


@dataclass(frozen=True)
class ValueVerificationJudgment:
    branch_id: str
    prediction_id: str
    reference_id: str
    rule: str
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "prediction_id": self.prediction_id,
            "reference_id": self.reference_id,
            "rule": self.rule,
            "verified": self.verified,
        }


@dataclass
class _EvaluationTrace:
    extracted_prediction: ExtractedPrediction | None = None
    semantic_matches: list[SemanticMatchJudgment] = field(default_factory=list)
    value_verifications: list[ValueVerificationJudgment] = field(default_factory=list)
    deduplication_mapping: tuple[DeduplicationJudgment, ...] = ()
    finding_eligibility: dict[str, FindingEligibility] = field(default_factory=dict)
    novel_finding_adjudications: list[tuple[str, str, AdjudicationStatus]] = field(
        default_factory=list
    )
    continuation_context: dict[str, Any] = field(default_factory=dict)


def _operationalization_request_payload(
    operationalization: ExtractedOperationalization | OperationalizationBundle,
) -> dict[str, Any]:
    decisions = operationalization.decisions
    return {
        "decisions": [
            {
                "dimension": decision.dimension.value,
                "statement": (
                    decision.normalized_statement
                    if isinstance(decision, ExtractedOperationalizationDecision)
                    else decision.statement
                ),
            }
            for decision in decisions
        ]
    }


def _canonical_statement_identity(value: str) -> str:
    """Canonicalize a statement for the completion self-consistency check.

    A completed novel-O candidate is derived from the extracted model response.
    When the candidate statement is byte-equivalent (modulo whitespace) to the
    extracted statement, it is already structurally proven to describe the
    same model choice.  Sending this exact self-comparison through the
    semantic backend can introduce an avoidable model refusal/false mismatch.
    Non-identical statements still use the normal semantic matcher below.
    """

    return " ".join(str(value).split()).casefold()


class _CachedSemanticMatcher:
    """Reuse persisted semantic judgments and delegate only new requests."""

    def __init__(
        self,
        judgments: Sequence[SemanticMatchJudgment],
        delegate: SemanticMatcher,
        resolutions: Sequence[SemanticMatchResolution] = (),
    ) -> None:
        self._judgments = tuple(judgments)
        self._delegate = delegate
        self._resolutions = tuple(resolutions)

    def match(self, request: SemanticMatchRequest) -> SemanticMatchResult:
        request_id = continuation_request_id("semantic_match", request.to_dict())
        for resolution in self._resolutions:
            if resolution.request_id == request_id:
                return resolution.result
        for judgment in self._judgments:
            if judgment.request == request:
                return judgment.result
        return self._delegate.match(request)


class _CachedEligibilityJudge:
    """Reuse persisted response-level eligibility judgments."""

    def __init__(
        self,
        judgments: Mapping[str, FindingEligibility],
        delegate: FindingEligibilityJudge,
    ) -> None:
        self._judgments = dict(judgments)
        self._delegate = delegate

    def judge(self, request: FindingEligibilityRequest) -> FindingEligibility:
        judgment = self._judgments.get(request.finding.prediction_id)
        if judgment is not None:
            return judgment
        return self._delegate.judge(request)


def _append_semantic_judgment(
    trace: _EvaluationTrace,
    judgment: SemanticMatchJudgment,
) -> None:
    if judgment not in trace.semantic_matches:
        trace.semantic_matches.append(judgment)


def _append_value_verification(
    trace: _EvaluationTrace,
    judgment: ValueVerificationJudgment,
) -> None:
    if judgment not in trace.value_verifications:
        trace.value_verifications.append(judgment)


@dataclass(frozen=True)
class AdjudicatedNovelBranch:
    operationalization: OperationalizationBundle
    findings: OperationalizationFindingBranch
    # Formal evaluation may carry the exact runtime materialization proof.
    # These fields are optional for the historical development adapter.
    materialization_id: str | None = None
    execution_provenance: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operationalization, OperationalizationBundle):
            object.__setattr__(
                self,
                "operationalization",
                OperationalizationBundle.model_validate(self.operationalization),
            )
        if not isinstance(self.findings, OperationalizationFindingBranch):
            object.__setattr__(
                self,
                "findings",
                OperationalizationFindingBranch.model_validate(self.findings),
            )
        if (
            self.operationalization.operationalization_id
            != self.findings.operationalization_id
        ):
            raise ValueError("novel O and G(O) must use the same operationalization_id")
        if self.materialization_id is not None and not str(self.materialization_id).strip():
            raise ValueError("materialization_id must be non-empty when supplied")


@dataclass(frozen=True)
class NovelOperationalizationAdjudication:
    status: AdjudicationStatus
    branch: AdjudicatedNovelBranch | None = None
    # Materialization-first continuations complete only the scientific O.
    # They must not fabricate a provisional G(O)/ReferenceFinding merely to
    # satisfy the historical branch-shaped transport.  ``branch`` remains a
    # backwards-compatible input for archived adjudications; new production
    # continuations use this O-only field and the trusted case materializer
    # supplies the actual G(O).
    operationalization: OperationalizationBundle | None = None
    adjudicated_materialization_id: str | None = None
    adjudicated_g_of_o_sha256: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", AdjudicationStatus(self.status))
        except ValueError as exc:
            raise ValueError("invalid novel O adjudication status") from exc
        if self.operationalization is not None and not isinstance(
            self.operationalization, OperationalizationBundle
        ):
            object.__setattr__(
                self,
                "operationalization",
                OperationalizationBundle.model_validate(self.operationalization),
            )
        if self.branch is not None and self.operationalization is not None:
            if self.branch.operationalization != self.operationalization:
                raise ValueError(
                    "novel O branch and operationalization fields disagree"
                )
        if self.status in {
            AdjudicationStatus.ACCEPTED,
            AdjudicationStatus.REJECTED,
        } and self.branch is None and self.operationalization is None:
            raise ValueError("resolved novel O requires a candidate Operationalization")
        for field_name in (
            "adjudicated_materialization_id",
            "adjudicated_g_of_o_sha256",
        ):
            value = getattr(self, field_name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{field_name} must be non-empty when supplied")
        if self.adjudicated_g_of_o_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.adjudicated_g_of_o_sha256
        ):
            raise ValueError("adjudicated_g_of_o_sha256 must be a SHA-256 digest")


@dataclass(frozen=True)
class EvaluationAdjudications:
    novel_operationalization: NovelOperationalizationAdjudication | None = None
    novel_findings: Mapping[tuple[str, str], AdjudicationStatus] = field(
        default_factory=dict
    )
    novel_finding_roles: Mapping[tuple[str, str], str] = field(default_factory=dict)
    semantic_resolutions: tuple[SemanticMatchResolution, ...] = ()
    unit_frame_resolutions: tuple[UnitFrameResolution, ...] = ()

    def __post_init__(self) -> None:
        normalized: dict[tuple[str, str], AdjudicationStatus] = {}
        for (branch_id, prediction_id), raw_status in self.novel_findings.items():
            _text(branch_id, "novel Finding branch_id")
            _text(prediction_id, "novel Finding prediction_id")
            try:
                status = AdjudicationStatus(raw_status)
            except ValueError as exc:
                raise ValueError("invalid novel Finding adjudication status") from exc
            normalized[(branch_id, prediction_id)] = status
        object.__setattr__(self, "novel_findings", normalized)
        role_map = {
            (str(branch_id), str(prediction_id)): _text(role, "novel Finding role")
            for (branch_id, prediction_id), role in self.novel_finding_roles.items()
        }
        object.__setattr__(self, "novel_finding_roles", role_map)
        object.__setattr__(
            self,
            "semantic_resolutions",
            tuple(
                item
                if isinstance(item, SemanticMatchResolution)
                else SemanticMatchResolution.from_dict(item)
                for item in self.semantic_resolutions
            ),
        )
        object.__setattr__(
            self,
            "unit_frame_resolutions",
            tuple(
                item
                if isinstance(item, UnitFrameResolution)
                else UnitFrameResolution.from_dict(item)
                for item in self.unit_frame_resolutions
            ),
        )
        for field_name in ("semantic_resolutions", "unit_frame_resolutions"):
            values = getattr(self, field_name)
            request_ids = [item.request_id for item in values]
            if len(request_ids) != len(set(request_ids)):
                raise EvaluationConfigurationError(
                    f"{field_name} contains duplicate continuation request_id"
                )

    def semantic_resolution_for(
        self, request: SemanticMatchRequest
    ) -> SemanticMatchResolution | None:
        request_id = continuation_request_id("semantic_match", request.to_dict())
        return next(
            (item for item in self.semantic_resolutions if item.request_id == request_id),
            None,
        )

    def unit_frame_resolution_for(
        self,
        predicted: PredictedAtomicFinding,
        reference: ReferenceFinding,
        explicit_policy: Mapping[str, Any] | None,
    ) -> UnitFrameResolution | None:
        request = _unit_frame_request_payload(predicted, reference, explicit_policy)
        request_id = continuation_request_id("unit_frame", request)
        return next(
            (item for item in self.unit_frame_resolutions if item.request_id == request_id),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        novel = self.novel_operationalization
        novel_payload: dict[str, Any] | None = None
        if novel is not None:
            novel_payload = {
                "status": novel.status.value,
                "adjudicated_materialization_id": novel.adjudicated_materialization_id,
                "adjudicated_g_of_o_sha256": novel.adjudicated_g_of_o_sha256,
            }
            if novel.operationalization is not None:
                novel_payload["operationalization"] = (
                    novel.operationalization.model_dump(mode="json")
                )
            if novel.branch is not None:
                novel_payload["branch"] = {
                    "operationalization": novel.branch.operationalization.model_dump(mode="json"),
                    "findings": novel.branch.findings.model_dump(mode="json"),
                }
                if novel.branch.materialization_id is not None:
                    novel_payload["branch"]["materialization_id"] = novel.branch.materialization_id
                if novel.branch.execution_provenance is not None:
                    novel_payload["branch"]["execution_provenance"] = dict(novel.branch.execution_provenance)
        return {
            "novel_operationalization": novel_payload,
            "novel_findings": [
                {
                    "branch_id": branch_id,
                    "prediction_id": prediction_id,
                    "status": status.value,
                }
                for (branch_id, prediction_id), status in sorted(self.novel_findings.items())
            ],
            "novel_finding_roles": [
                {"branch_id": branch_id, "prediction_id": prediction_id, "role": role}
                for (branch_id, prediction_id), role in sorted(self.novel_finding_roles.items())
            ],
            "semantic_resolutions": [
                item.to_dict() for item in self.semantic_resolutions
            ],
            "unit_frame_resolutions": [
                item.to_dict() for item in self.unit_frame_resolutions
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationAdjudications":
        novel_value = value.get("novel_operationalization")
        novel: NovelOperationalizationAdjudication | None = None
        if novel_value is not None:
            branch_value = novel_value.get("branch")
            branch = None
            if branch_value is not None:
                branch = AdjudicatedNovelBranch(
                    operationalization=OperationalizationBundle.model_validate(
                        branch_value["operationalization"]
                    ),
                    findings=OperationalizationFindingBranch.model_validate(
                        branch_value["findings"]
                    ),
                    materialization_id=branch_value.get("materialization_id"),
                    execution_provenance=branch_value.get("execution_provenance"),
                )
            novel = NovelOperationalizationAdjudication(
                status=novel_value["status"],
                branch=branch,
                operationalization=(
                    None
                    if novel_value.get("operationalization") is None
                    else OperationalizationBundle.model_validate(
                        novel_value["operationalization"]
                    )
                ),
                adjudicated_materialization_id=novel_value.get(
                    "adjudicated_materialization_id"
                ),
                adjudicated_g_of_o_sha256=novel_value.get(
                    "adjudicated_g_of_o_sha256"
                ),
            )
        novel_findings = {
            (item["branch_id"], item["prediction_id"]): item["status"]
            for item in value.get("novel_findings", ())
        }
        novel_finding_roles = {
            (item["branch_id"], item["prediction_id"]): item["role"]
            for item in value.get("novel_finding_roles", ())
        }
        return cls(
            novel_operationalization=novel,
            novel_findings=novel_findings,
            novel_finding_roles=novel_finding_roles,
            semantic_resolutions=tuple(value.get("semantic_resolutions", ())),
            unit_frame_resolutions=tuple(value.get("unit_frame_resolutions", ())),
        )


@dataclass(frozen=True)
class RunEvaluationAdjudication:
    """Run-bound wrapper for one explicit adjudication input artifact."""

    run_id: str
    case_id: str
    adjudications: EvaluationAdjudications

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        _text(self.case_id, "case_id")
        if not isinstance(self.adjudications, EvaluationAdjudications):
            object.__setattr__(
                self,
                "adjudications",
                EvaluationAdjudications.from_dict(self.adjudications),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "run_evaluation_adjudication",
            "run_id": self.run_id,
            "case_id": self.case_id,
            "adjudications": self.adjudications.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunEvaluationAdjudication":
        if value.get("record_type") not in {
            None,
            "run_evaluation_adjudication",
        }:
            raise ValueError("invalid run evaluation adjudication record type")
        return cls(
            run_id=value["run_id"],
            case_id=value["case_id"],
            adjudications=EvaluationAdjudications.from_dict(value["adjudications"]),
        )


@dataclass(frozen=True)
class FindingBranchDiagnostic:
    branch_id: str
    matched_pairs: tuple[tuple[str, str], ...]
    accepted_gt_outside_prediction_ids: tuple[str, ...]
    consistent_applicable_prediction_ids: tuple[str, ...]
    predicted_finding_count: int
    valid_predicted_finding_count: int
    core_finding_count: int
    matched_core_finding_count: int
    adequate_core_evaluation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "matched_pairs": [
                {"prediction_id": prediction_id, "finding_id": finding_id}
                for prediction_id, finding_id in self.matched_pairs
            ],
            "accepted_gt_outside_prediction_ids": list(
                self.accepted_gt_outside_prediction_ids
            ),
            "consistent_applicable_prediction_ids": list(
                self.consistent_applicable_prediction_ids
            ),
            "predicted_finding_count": self.predicted_finding_count,
            "valid_predicted_finding_count": self.valid_predicted_finding_count,
            "core_finding_count": self.core_finding_count,
            "matched_core_finding_count": self.matched_core_finding_count,
            "adequate_core_evaluation": self.adequate_core_evaluation,
        }


@dataclass(frozen=True)
class ConsistencyBranchDiagnostic:
    branch_id: str
    applicable_finding_count: int
    consistent_finding_count: int
    c_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "applicable_finding_count": self.applicable_finding_count,
            "consistent_finding_count": self.consistent_finding_count,
            "c_score": self.c_score,
        }


@dataclass(frozen=True)
class CaseEvaluationResult:
    case_id: str
    condition: str
    run_status: RunStatus
    eligible_for_scientific_aggregation: bool
    metrics: CaseMetricResult | None
    deduplicated_prediction_ids: tuple[str, ...] = ()
    f_app_prediction_ids: tuple[str, ...] = ()
    compatible_o_branches: tuple[str, ...] = ()
    best_c_branches: tuple[str, ...] = ()
    finding_branch_diagnostics: tuple[FindingBranchDiagnostic, ...] = ()
    consistency_branch_diagnostics: tuple[ConsistencyBranchDiagnostic, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "run_status": self.run_status.value,
            "eligible_for_scientific_aggregation": self.eligible_for_scientific_aggregation,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "deduplicated_prediction_ids": list(self.deduplicated_prediction_ids),
            "f_app_prediction_ids": list(self.f_app_prediction_ids),
            "compatible_o_branches": list(self.compatible_o_branches),
            "best_c_branches": list(self.best_c_branches),
            "finding_branch_diagnostics": [
                diagnostic.to_dict() for diagnostic in self.finding_branch_diagnostics
            ],
            "consistency_branch_diagnostics": [
                diagnostic.to_dict()
                for diagnostic in self.consistency_branch_diagnostics
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaseEvaluationResult":
        metrics_value = value.get("metrics")
        metrics = None if metrics_value is None else _case_metric_result_from_dict(
            metrics_value
        )
        return cls(
            case_id=value["case_id"],
            condition=value["condition"],
            run_status=RunStatus(value["run_status"]),
            eligible_for_scientific_aggregation=value[
                "eligible_for_scientific_aggregation"
            ],
            metrics=metrics,
            deduplicated_prediction_ids=tuple(
                value.get("deduplicated_prediction_ids", ())
            ),
            f_app_prediction_ids=tuple(value.get("f_app_prediction_ids", ())),
            compatible_o_branches=tuple(value.get("compatible_o_branches", ())),
            best_c_branches=tuple(value.get("best_c_branches", ())),
            finding_branch_diagnostics=tuple(
                FindingBranchDiagnostic(
                    branch_id=item["branch_id"],
                    matched_pairs=tuple(
                        (pair["prediction_id"], pair["finding_id"])
                        for pair in item["matched_pairs"]
                    ),
                    accepted_gt_outside_prediction_ids=tuple(
                        item["accepted_gt_outside_prediction_ids"]
                    ),
                    consistent_applicable_prediction_ids=tuple(
                        item["consistent_applicable_prediction_ids"]
                    ),
                    predicted_finding_count=item["predicted_finding_count"],
                    valid_predicted_finding_count=item[
                        "valid_predicted_finding_count"
                    ],
                    core_finding_count=item["core_finding_count"],
                    matched_core_finding_count=item["matched_core_finding_count"],
                    adequate_core_evaluation=item.get("adequate_core_evaluation"),
                )
                for item in value.get("finding_branch_diagnostics", ())
            ),
            consistency_branch_diagnostics=tuple(
                ConsistencyBranchDiagnostic(**item)
                for item in value.get("consistency_branch_diagnostics", ())
            ),
        )


@dataclass(frozen=True)
class TrialEvaluation:
    """One trial-level evaluation supplied to the formal case aggregator.

    Infrastructure-invalid attempts may be included for auditability but do
    not occupy a formal trial slot.  A pending marker represents an unresolved
    adjudication and blocks case finalization; it is deliberately not coerced
    to a zero or substituted by another observation.
    """

    trial_index: int
    result: CaseEvaluationResult | None = None
    pending: bool = False
    pending_reason: str | None = None
    experiment_id: str | None = None
    target_fingerprint: str | None = None
    benchmark_release_id: str | None = None
    formal_mode: bool | None = None
    evaluation_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 1
        ):
            raise ValueError("trial_index must be a positive integer")
        if not isinstance(self.pending, bool):
            raise ValueError("pending must be bool")
        if self.pending and self.result is not None:
            raise ValueError("pending trial cannot contain a scored result")
        if not self.pending and self.result is None:
            raise ValueError("non-pending trial requires a CaseEvaluationResult")
        if self.pending_reason is not None:
            _text(self.pending_reason, "pending_reason")
        for field_name in (
            "experiment_id",
            "target_fingerprint",
            "benchmark_release_id",
            "evaluation_manifest_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        if self.formal_mode is not None and not isinstance(self.formal_mode, bool):
            raise ValueError("formal_mode must be bool or None")

    @property
    def is_infrastructure_invalid(self) -> bool:
        return (
            self.result is not None
            and self.result.run_status == RunStatus.INFRASTRUCTURE_INVALID
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "pending": self.pending,
            "pending_reason": self.pending_reason,
            "experiment_id": self.experiment_id,
            "target_fingerprint": self.target_fingerprint,
            "benchmark_release_id": self.benchmark_release_id,
            "formal_mode": self.formal_mode,
            "evaluation_manifest_digest": self.evaluation_manifest_digest,
            "result": None if self.result is None else self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialEvaluation":
        raw_result = value.get("result")
        return cls(
            trial_index=value["trial_index"],
            result=(
                None
                if raw_result is None
                else CaseEvaluationResult.from_dict(raw_result)
            ),
            pending=bool(value.get("pending", False)),
            pending_reason=value.get("pending_reason"),
            experiment_id=value.get("experiment_id"),
            target_fingerprint=value.get("target_fingerprint"),
            benchmark_release_id=value.get("benchmark_release_id"),
            formal_mode=value.get("formal_mode"),
            evaluation_manifest_digest=value.get("evaluation_manifest_digest"),
        )

    @classmethod
    def from_evaluation_record(cls, record: "CaseEvaluationRecord") -> "TrialEvaluation":
        """Carry run identity from an evaluation record without manual copying."""

        if record.trial_index is None:
            raise EvaluationConfigurationError(
                "evaluation record has no originating trial_index"
            )
        return cls(
            trial_index=record.trial_index,
            result=record.result,
            experiment_id=record.experiment_id,
            target_fingerprint=record.target_fingerprint,
            benchmark_release_id=record.benchmark_release_id,
            formal_mode=record.formal_mode,
            evaluation_manifest_digest=record.evaluation_manifest_digest,
        )


@dataclass(frozen=True)
class CaseAggregate:
    """A finalized case estimator built from exactly three valid trials."""

    case_id: str
    condition: str
    trial_evaluations: tuple[TrialEvaluation, ...]
    metrics: CaseAggregateMetricResult
    experiment_id: str | None = None
    target_fingerprint: str | None = None
    benchmark_release_id: str | None = None
    formal_mode: bool | None = None
    evaluation_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        _text(self.case_id, "case_id")
        _text(self.condition, "condition")
        if self.metrics.case_id != self.case_id or self.metrics.condition != self.condition:
            raise EvaluationConfigurationError(
                "case aggregate identity must match its metric identity"
            )
        if any(trial.pending for trial in self.trial_evaluations):
            raise EvaluationPendingAdjudication(
                "a pending trial cannot be stored in a finalized CaseAggregate"
            )
        if self.trial_count != FORMAL_TRIAL_COUNT:
            raise EvaluationConfigurationError(
                "a formal CaseAggregate requires exactly three valid model trials"
            )
        valid_trials = self.valid_trials
        if {trial.trial_index for trial in valid_trials} != set(
            range(1, FORMAL_TRIAL_COUNT + 1)
        ):
            raise EvaluationConfigurationError(
                "a finalized CaseAggregate requires trial indices {1, 2, 3}"
            )
        if any(
            trial.result is None
            or not trial.result.eligible_for_scientific_aggregation
            or trial.result.metrics is None
            for trial in valid_trials
        ):
            raise EvaluationConfigurationError(
                "a finalized CaseAggregate may contain only fully scored valid trials"
            )
        if any(
            trial.result.case_id != self.case_id or trial.result.condition != self.condition
            for trial in valid_trials
            if trial.result is not None
        ):
            raise EvaluationConfigurationError(
                "all trials in a CaseAggregate must share case_id and condition"
            )
        experiment_ids = {
            trial.experiment_id
            for trial in valid_trials
            if trial.experiment_id is not None
        }
        target_fingerprints = {
            trial.target_fingerprint
            for trial in valid_trials
            if trial.target_fingerprint is not None
        }
        release_ids = {
            trial.benchmark_release_id
            for trial in valid_trials
            if trial.benchmark_release_id is not None
        }
        formal_modes = {trial.formal_mode for trial in valid_trials if trial.formal_mode is not None}
        for field_name, values in (
            ("experiment_id", [trial.experiment_id for trial in valid_trials]),
            ("target_fingerprint", [trial.target_fingerprint for trial in valid_trials]),
            ("benchmark_release_id", [trial.benchmark_release_id for trial in valid_trials]),
            ("formal_mode", [trial.formal_mode for trial in valid_trials]),
            (
                "evaluation_manifest_digest",
                [trial.evaluation_manifest_digest for trial in valid_trials],
            ),
        ):
            if any(value is None for value in values) and any(value is not None for value in values):
                raise EvaluationConfigurationError(
                    f"all trials in a CaseAggregate must either provide or omit {field_name} identity"
                )
        for field_name in (
            "experiment_id",
            "target_fingerprint",
            "benchmark_release_id",
            "formal_mode",
            "evaluation_manifest_digest",
        ):
            aggregate_value = getattr(self, field_name)
            trial_values = [getattr(trial, field_name) for trial in self.trial_evaluations]
            if aggregate_value is None:
                if any(value is not None for value in trial_values):
                    raise EvaluationConfigurationError(
                        f"case aggregate {field_name} is missing trial identity"
                    )
            elif any(value != aggregate_value for value in trial_values):
                raise EvaluationConfigurationError(
                    f"case aggregate {field_name} does not match trial identities"
                )
        if len(experiment_ids) > 1 or len(target_fingerprints) > 1:
            raise EvaluationConfigurationError(
                "all trials in a CaseAggregate must share experiment and target identity"
            )
        if len(release_ids) > 1 or len(formal_modes) > 1:
            raise EvaluationConfigurationError(
                "all trials in a CaseAggregate must share release and formal-mode identity"
            )
        for field_name in (
            "experiment_id",
            "target_fingerprint",
            "benchmark_release_id",
            "evaluation_manifest_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        if self.formal_mode is not None and not isinstance(self.formal_mode, bool):
            raise ValueError("formal_mode must be bool or None")

    @property
    def valid_trials(self) -> tuple[TrialEvaluation, ...]:
        return tuple(
            trial
            for trial in self.trial_evaluations
            if trial.result is not None and not trial.is_infrastructure_invalid
        )

    @property
    def trial_count(self) -> int:
        return len(self.valid_trials)

    @property
    def infrastructure_invalid_trial_count(self) -> int:
        return sum(trial.is_infrastructure_invalid for trial in self.trial_evaluations)

    @property
    def model_noncompletion_trial_count(self) -> int:
        return sum(
            trial.result is not None
            and trial.result.run_status == RunStatus.MODEL_NONCOMPLETION
            for trial in self.trial_evaluations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "condition": self.condition,
            "trial_count": self.trial_count,
            "experiment_id": self.experiment_id,
            "target_fingerprint": self.target_fingerprint,
            "benchmark_release_id": self.benchmark_release_id,
            "formal_mode": self.formal_mode,
            "evaluation_manifest_digest": self.evaluation_manifest_digest,
            "trial_evaluations": [
                trial.to_dict() for trial in self.trial_evaluations
            ],
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaseAggregate":
        metrics_value = value["metrics"]
        efficiency = metrics_value["efficiency"]
        finding_recall_mode = metrics_value.get(
            "finding_recall_mode", "FIXED_REFERENCE_CORE"
        )
        core_recall = metrics_value.get("core_finding_recall")
        # F1's public requirement recall is a compatibility alias for its
        # fixed core recall.  F2 must never be reconstructed from the core
        # field: its core field is intentionally null and requirement recall
        # is the adequate-core score.
        requirement_recall = metrics_value.get("finding_requirement_recall")
        if requirement_recall is None and finding_recall_mode == "FIXED_REFERENCE_CORE":
            requirement_recall = core_recall
        metrics = CaseAggregateMetricResult(
            case_id=metrics_value["case_id"],
            condition=metrics_value["condition"],
            o_score=metrics_value["o_score"],
            urs=metrics_value["urs"],
            finding_precision=metrics_value["finding_precision"],
            core_finding_recall=core_recall,
            resolved_o_compliance=metrics_value.get("resolved_o_compliance"),
            c_score=metrics_value["c_score"],
            branch_alignment=metrics_value["branch_alignment"],
            finding_requirement_recall=requirement_recall,
            finding_recall_mode=finding_recall_mode,
            adequate_core_complete=metrics_value.get("adequate_core_complete"),
            adequate_core_complete_rate=metrics_value.get("adequate_core_complete_rate"),
            metric_trial_denominators=metrics_value.get("metric_trial_denominators", {}),
            determinate_o_trial_count=metrics_value.get("determinate_o_trial_count"),
            determinate_o_trial_rate=metrics_value.get("determinate_o_trial_rate"),
            efficiency=EfficiencyAggregate(
                input_tokens=efficiency["input_tokens"],
                output_tokens=efficiency["output_tokens"],
                model_turn_count=efficiency["model_turn_count"],
                tool_call_count=efficiency.get("tool_call_count"),
                python_execution_count=efficiency["python_execution_count"],
                wall_clock_time=efficiency["wall_clock_time"],
                provider_reported_cost=efficiency.get("provider_reported_cost"),
            ),
        )
        return cls(
            case_id=value["case_id"],
            condition=value["condition"],
            trial_evaluations=tuple(
                TrialEvaluation.from_dict(item)
                for item in value["trial_evaluations"]
            ),
            metrics=metrics,
            experiment_id=value.get("experiment_id"),
            target_fingerprint=value.get("target_fingerprint"),
            benchmark_release_id=value.get("benchmark_release_id"),
            formal_mode=value.get("formal_mode"),
            evaluation_manifest_digest=value.get("evaluation_manifest_digest"),
        )


def _case_metric_result_from_dict(value: Mapping[str, Any]) -> CaseMetricResult:
    operationalization = value["scientific_operationalization"]
    findings = value["scientific_findings"]
    consistency = value["o_f_consistency"]
    efficiency = value["efficiency"]
    finding_recall_mode = findings.get(
        "finding_recall_mode", "FIXED_REFERENCE_CORE"
    )
    core_recall = findings.get("core_finding_recall")
    requirement_recall = findings.get("finding_requirement_recall")
    if requirement_recall is None and finding_recall_mode == "FIXED_REFERENCE_CORE":
        requirement_recall = core_recall
    return CaseMetricResult(
        case_id=value["case_id"],
        condition=value["condition"],
        scientific_operationalization=OperationalizationMetricResult(
            o_score=operationalization["o_score"],
            best_o_branches=tuple(operationalization["best_o_branches"]),
            urs=operationalization["urs"],
            resolved_o_compliance=operationalization.get("resolved_o_compliance"),
        ),
        scientific_findings=FindingMetricResult(
            finding_precision=findings["finding_precision"],
            core_finding_recall=core_recall,
            best_finding_branches=(
                None
                if findings["best_finding_branches"] is None
                else tuple(findings["best_finding_branches"])
            ),
            finding_requirement_recall=requirement_recall,
            finding_recall_mode=finding_recall_mode,
            adequate_core_complete=findings.get("adequate_core_complete"),
        ),
        o_f_consistency=ConsistencyMetricResult(
            c_score=consistency["c_score"],
            branch_alignment=consistency["branch_alignment"],
            operationalization_determinate=consistency.get("operationalization_determinate"),
            unavailable_reason=consistency.get("unavailable_reason"),
        ),
        efficiency=EfficiencyObservation(
            input_tokens=efficiency["input_tokens"],
            output_tokens=efficiency["output_tokens"],
            model_turn_count=efficiency["model_turn_count"],
            tool_call_count=efficiency.get("tool_call_count"),
            python_execution_count=efficiency["python_execution_count"],
            wall_clock_time=efficiency["wall_clock_time"],
            provider_reported_cost=efficiency.get("provider_reported_cost"),
        ),
    )


def _extracted_prediction_to_dict(value: ExtractedPrediction) -> dict[str, Any]:
    def evidence_span(span: EvidenceSpan) -> str | list[int] | None:
        return list(span) if isinstance(span, tuple) else span

    return {
        "operationalization": {
            "decisions": [
                {
                    "dimension": decision.dimension.value,
                    "status": decision.status.value,
                    "normalized_statement": decision.normalized_statement,
                    "evidence_span": evidence_span(decision.evidence_span),
                }
                for decision in value.operationalization.decisions
            ]
        },
        "findings": [
            {
                "prediction_id": finding.prediction_id,
                "statement": finding.statement,
                "value": finding.value,
                "unit": finding.unit,
                "evidence_span": evidence_span(finding.evidence_span),
            }
            for finding in value.findings
        ],
    }


def _extracted_prediction_from_dict(value: Mapping[str, Any]) -> ExtractedPrediction:
    def evidence_span(span: Any) -> EvidenceSpan:
        return tuple(span) if isinstance(span, list) else span

    return ExtractedPrediction(
        operationalization=ExtractedOperationalization(
            tuple(
                ExtractedOperationalizationDecision(
                    dimension=item["dimension"],
                    status=item["status"],
                    normalized_statement=item.get("normalized_statement"),
                    evidence_span=evidence_span(item.get("evidence_span")),
                )
                for item in value["operationalization"]["decisions"]
            )
        ),
        findings=tuple(
            PredictedAtomicFinding(
                prediction_id=item["prediction_id"],
                statement=item["statement"],
                value=item.get("value"),
                unit=item.get("unit"),
                evidence_span=evidence_span(item.get("evidence_span")),
            )
            for item in value["findings"]
        ),
    )


@dataclass(frozen=True)
class CaseEvaluationRecord:
    """Serializable judgments sufficient for deterministic score replay."""

    record_version: str
    evaluation_manifest: EvaluationManifest
    result: CaseEvaluationResult
    extracted_prediction: ExtractedPrediction | None
    deduplication_mapping: tuple[DeduplicationJudgment, ...]
    finding_eligibility: Mapping[str, FindingEligibility]
    semantic_matches: tuple[SemanticMatchJudgment, ...]
    value_verifications: tuple[ValueVerificationJudgment, ...]
    operationalization_semantic_matches: Mapping[
        str, Mapping[str, SemanticMatchResult]
    ]
    operationalization_branch_evaluations: tuple[
        OperationalizationBranchEvaluation, ...
    ]
    finding_branch_evaluations: tuple[FindingBranchEvaluation, ...]
    consistency_evaluation: ConsistencyEvaluation
    efficiency: EfficiencyObservation
    novel_operationalization_adjudication: Mapping[str, Any] | None = None
    novel_finding_adjudications: tuple[
        tuple[str, str, AdjudicationStatus], ...
    ] = ()
    run_id: str | None = None
    trial_index: int | None = None
    experiment_id: str | None = None
    target_fingerprint: str | None = None
    benchmark_release_id: str | None = None
    formal_mode: bool | None = None
    evaluation_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        _text(self.record_version, "record_version")
        object.__setattr__(
            self,
            "novel_operationalization_adjudication",
            (
                None
                if self.novel_operationalization_adjudication is None
                else _json_value(
                    self.novel_operationalization_adjudication,
                    "novel_operationalization_adjudication",
                )
            ),
        )
        if self.run_id is not None:
            _text(self.run_id, "run_id")
        if self.trial_index is not None and (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 1
        ):
            raise ValueError("trial_index must be a positive integer or None")
        for field_name in ("experiment_id", "target_fingerprint"):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        if self.benchmark_release_id is not None:
            _text(self.benchmark_release_id, "benchmark_release_id")
        if self.formal_mode is not None and not isinstance(self.formal_mode, bool):
            raise ValueError("formal_mode must be bool or None")
        expected_digest = evaluation_manifest_digest(self.evaluation_manifest)
        if self.evaluation_manifest_digest is None:
            object.__setattr__(self, "evaluation_manifest_digest", expected_digest)
        elif self.evaluation_manifest_digest != expected_digest:
            raise EvaluationConfigurationError(
                "evaluation_manifest_digest does not match evaluation_manifest"
            )

    def replay(self) -> CaseEvaluationResult:
        """Recalculate deterministic metrics without invoking semantic components."""

        if self.result.run_status != RunStatus.COMPLETED:
            return self.result
        recomputed = compute_case_metrics(
            CaseMetricInput(
                case_id=self.result.case_id,
                condition=self.result.condition,
                operationalization_branches=self.operationalization_branch_evaluations,
                finding_branches=self.finding_branch_evaluations,
                consistency=self.consistency_evaluation,
                efficiency=self.efficiency,
            )
        )
        if self.result.metrics is None or recomputed.to_dict() != self.result.metrics.to_dict():
            raise EvaluationConfigurationError(
                "persisted evaluation judgments do not reproduce the stored metrics"
            )
        return replace(self.result, metrics=recomputed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_version": self.record_version,
            "evaluation_manifest": self.evaluation_manifest.to_dict(),
            "result": self.result.to_dict(),
            "extracted_prediction": (
                None
                if self.extracted_prediction is None
                else _extracted_prediction_to_dict(self.extracted_prediction)
            ),
            "deduplication_mapping": [
                item.to_dict() for item in self.deduplication_mapping
            ],
            "finding_eligibility": {
                prediction_id: {
                    "scientifically_interpretable": judgment.scientifically_interpretable,
                    "relevant_to_finding_goal": judgment.relevant_to_finding_goal,
                    "in_principle_verifiable": judgment.in_principle_verifiable,
                }
                for prediction_id, judgment in self.finding_eligibility.items()
            },
            "semantic_matches": [item.to_dict() for item in self.semantic_matches],
            "value_verifications": [
                item.to_dict() for item in self.value_verifications
            ],
            "operationalization_semantic_matches": {
                branch_id: {
                    dimension: result.value
                    for dimension, result in matches.items()
                }
                for branch_id, matches in self.operationalization_semantic_matches.items()
            },
            "operationalization_branch_evaluations": [
                {
                    "branch_id": item.branch_id,
                    "principal_dimension_matches": dict(
                        item.principal_dimension_matches
                    ),
                    "unresolved_dimension_matches": dict(
                        item.unresolved_dimension_matches
                    ),
                }
                for item in self.operationalization_branch_evaluations
            ],
            "finding_branch_evaluations": [
                {
                    "branch_id": item.branch_id,
                    "predicted_finding_count": item.predicted_finding_count,
                    "valid_predicted_finding_count": item.valid_predicted_finding_count,
                    "core_finding_count": item.core_finding_count,
                    "matched_core_finding_count": item.matched_core_finding_count,
                    "scoring_mode": item.scoring_mode,
                    "mandatory_role_count": item.mandatory_role_count,
                    "matched_mandatory_role_count": item.matched_mandatory_role_count,
                    "adequate_set_evaluations": list(item.adequate_set_evaluations),
                    "best_adequate_set_ids": list(item.best_adequate_set_ids),
                    "best_adequate_set_id": item.best_adequate_set_id,
                    "best_adequate_set_recall": item.best_adequate_set_recall,
                    "novel_role_matches": list(item.novel_role_matches),
                    "role_match_provenance": dict(item.role_match_provenance),
                }
                for item in self.finding_branch_evaluations
            ],
            "consistency_evaluation": dict(
                self.consistency_evaluation.finding_results
            ),
            "consistency_operationalization_determinate": self.consistency_evaluation.operationalization_determinate,
            "consistency_unavailable_reason": self.consistency_evaluation.unavailable_reason,
            "efficiency": self.efficiency.to_dict(),
            "novel_operationalization_adjudication": self.novel_operationalization_adjudication,
            "novel_finding_adjudications": [
                {
                    "branch_id": branch_id,
                    "prediction_id": prediction_id,
                    "status": status.value,
                }
                for branch_id, prediction_id, status in self.novel_finding_adjudications
            ],
            "run_id": self.run_id,
            "trial_index": self.trial_index,
            "experiment_id": self.experiment_id,
            "target_fingerprint": self.target_fingerprint,
            "benchmark_release_id": self.benchmark_release_id,
            "formal_mode": self.formal_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CaseEvaluationRecord":
        efficiency = value["efficiency"]
        return cls(
            record_version=value["record_version"],
            evaluation_manifest=EvaluationManifest.from_dict(
                value["evaluation_manifest"]
            ),
            result=CaseEvaluationResult.from_dict(value["result"]),
            extracted_prediction=(
                None
                if value.get("extracted_prediction") is None
                else _extracted_prediction_from_dict(value["extracted_prediction"])
            ),
            deduplication_mapping=tuple(
                DeduplicationJudgment(**item)
                for item in value.get("deduplication_mapping", ())
            ),
            finding_eligibility={
                prediction_id: FindingEligibility(**judgment)
                for prediction_id, judgment in value.get(
                    "finding_eligibility", {}
                ).items()
            },
            semantic_matches=tuple(
                SemanticMatchJudgment.from_dict(item)
                for item in value.get("semantic_matches", ())
            ),
            value_verifications=tuple(
                ValueVerificationJudgment(**item)
                for item in value.get("value_verifications", ())
            ),
            operationalization_semantic_matches={
                branch_id: {
                    dimension: SemanticMatchResult(result)
                    for dimension, result in matches.items()
                }
                for branch_id, matches in value.get(
                    "operationalization_semantic_matches", {}
                ).items()
            },
            operationalization_branch_evaluations=tuple(
                OperationalizationBranchEvaluation(**item)
                for item in value.get(
                    "operationalization_branch_evaluations", ()
                )
            ),
            finding_branch_evaluations=tuple(
                FindingBranchEvaluation(**item)
                for item in value.get("finding_branch_evaluations", ())
            ),
            consistency_evaluation=ConsistencyEvaluation(
                value.get("consistency_evaluation", {}),
                operationalization_determinate=value.get("consistency_operationalization_determinate"),
                unavailable_reason=value.get("consistency_unavailable_reason"),
            ),
            efficiency=EfficiencyObservation(
                input_tokens=efficiency["input_tokens"],
                output_tokens=efficiency["output_tokens"],
                model_turn_count=efficiency["model_turn_count"],
                tool_call_count=efficiency.get("tool_call_count"),
                python_execution_count=efficiency["python_execution_count"],
                wall_clock_time=efficiency["wall_clock_time"],
                provider_reported_cost=efficiency.get("provider_reported_cost"),
            ),
            novel_operationalization_adjudication=value.get(
                "novel_operationalization_adjudication"
            ),
            novel_finding_adjudications=tuple(
                (
                    item["branch_id"],
                    item["prediction_id"],
                    AdjudicationStatus(item["status"]),
                )
                for item in value.get("novel_finding_adjudications", ())
            ),
            run_id=value.get("run_id"),
            trial_index=value.get("trial_index"),
            experiment_id=value.get("experiment_id"),
            target_fingerprint=value.get("target_fingerprint"),
            benchmark_release_id=value.get("benchmark_release_id"),
            formal_mode=value.get("formal_mode"),
            evaluation_manifest_digest=value.get("evaluation_manifest_digest"),
        )


@dataclass(frozen=True)
class PendingCaseEvaluationRecord:
    """Persisted partial evaluation state awaiting one explicit adjudication."""

    record_version: str
    evaluation_manifest: EvaluationManifest
    case_id: str
    condition: str
    pending_type: str
    pending_reason: str
    extracted_prediction: ExtractedPrediction | None
    deduplication_mapping: tuple[DeduplicationJudgment, ...] = ()
    finding_eligibility: Mapping[str, FindingEligibility] = field(default_factory=dict)
    semantic_matches: tuple[SemanticMatchJudgment, ...] = ()
    value_verifications: tuple[ValueVerificationJudgment, ...] = ()
    run_id: str | None = None
    trial_index: int | None = None
    experiment_id: str | None = None
    target_fingerprint: str | None = None
    benchmark_release_id: str | None = None
    formal_mode: bool | None = None
    evaluation_manifest_digest: str | None = None
    efficiency: EfficiencyObservation | None = None
    continuation_context: Mapping[str, Any] = field(default_factory=dict)
    continuation_request_id: str | None = None
    accumulated_adjudications: EvaluationAdjudications = field(
        default_factory=EvaluationAdjudications
    )

    def __post_init__(self) -> None:
        _text(self.record_version, "record_version")
        _text(self.case_id, "case_id")
        _text(self.condition, "condition")
        _text(self.pending_type, "pending_type")
        _text(self.pending_reason, "pending_reason")
        if not isinstance(self.evaluation_manifest, EvaluationManifest):
            object.__setattr__(
                self,
                "evaluation_manifest",
                EvaluationManifest.from_dict(self.evaluation_manifest),
            )
        for field_name in (
            "run_id",
            "experiment_id",
            "target_fingerprint",
            "benchmark_release_id",
            "evaluation_manifest_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _text(value, field_name)
        if self.formal_mode is not None and not isinstance(self.formal_mode, bool):
            raise ValueError("formal_mode must be bool or None")
        object.__setattr__(
            self,
            "continuation_context",
            _json_value(dict(self.continuation_context), "continuation_context"),
        )
        if not isinstance(self.accumulated_adjudications, EvaluationAdjudications):
            object.__setattr__(
                self,
                "accumulated_adjudications",
                EvaluationAdjudications.from_dict(self.accumulated_adjudications),
            )
        context_request_id = self.continuation_context.get("continuation_request_id")
        if context_request_id is not None:
            _text(context_request_id, "continuation_context.continuation_request_id")
        if self.continuation_request_id is None:
            object.__setattr__(self, "continuation_request_id", context_request_id)
        elif context_request_id is not None and self.continuation_request_id != context_request_id:
            raise EvaluationConfigurationError(
                "continuation_request_id does not match continuation_context"
            )
        elif self.continuation_request_id is not None:
            _text(self.continuation_request_id, "continuation_request_id")
        if self.efficiency is not None and not isinstance(
            self.efficiency, EfficiencyObservation
        ):
            raw_efficiency = dict(self.efficiency)
            object.__setattr__(
                self,
                "efficiency",
                EfficiencyObservation(
                    **{
                        field_name: raw_efficiency.get(field_name)
                        for field_name in (
                            "input_tokens",
                            "output_tokens",
                            "model_turn_count",
                            "tool_call_count",
                            "python_execution_count",
                            "wall_clock_time",
                            "provider_reported_cost",
                        )
                    }
                ),
            )
        expected_digest = evaluation_manifest_digest(self.evaluation_manifest)
        if self.evaluation_manifest_digest is None:
            object.__setattr__(self, "evaluation_manifest_digest", expected_digest)
        elif self.evaluation_manifest_digest != expected_digest:
            raise EvaluationConfigurationError(
                "evaluation_manifest_digest does not match evaluation_manifest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "pending_case_evaluation",
            "record_version": self.record_version,
            "evaluation_manifest": self.evaluation_manifest.to_dict(),
            "case_id": self.case_id,
            "condition": self.condition,
            "pending": True,
            "pending_type": self.pending_type,
            "pending_reason": self.pending_reason,
            "extracted_prediction": (
                None
                if self.extracted_prediction is None
                else _extracted_prediction_to_dict(self.extracted_prediction)
            ),
            "deduplication_mapping": [
                item.to_dict() for item in self.deduplication_mapping
            ],
            "finding_eligibility": {
                prediction_id: {
                    "scientifically_interpretable": judgment.scientifically_interpretable,
                    "relevant_to_finding_goal": judgment.relevant_to_finding_goal,
                    "in_principle_verifiable": judgment.in_principle_verifiable,
                }
                for prediction_id, judgment in self.finding_eligibility.items()
            },
            "semantic_matches": [item.to_dict() for item in self.semantic_matches],
            "value_verifications": [item.to_dict() for item in self.value_verifications],
            "run_id": self.run_id,
            "trial_index": self.trial_index,
            "experiment_id": self.experiment_id,
            "target_fingerprint": self.target_fingerprint,
            "benchmark_release_id": self.benchmark_release_id,
            "formal_mode": self.formal_mode,
            "evaluation_manifest_digest": self.evaluation_manifest_digest,
            "efficiency": (
                None if self.efficiency is None else self.efficiency.to_dict()
            ),
            "continuation_context": dict(self.continuation_context),
            "continuation_request_id": self.continuation_request_id,
            "accumulated_adjudications": self.accumulated_adjudications.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PendingCaseEvaluationRecord":
        return cls(
            record_version=value["record_version"],
            evaluation_manifest=EvaluationManifest.from_dict(value["evaluation_manifest"]),
            case_id=value["case_id"],
            condition=value["condition"],
            pending_type=value["pending_type"],
            pending_reason=value["pending_reason"],
            extracted_prediction=(
                None
                if value.get("extracted_prediction") is None
                else _extracted_prediction_from_dict(value["extracted_prediction"])
            ),
            deduplication_mapping=tuple(
                DeduplicationJudgment(**item)
                for item in value.get("deduplication_mapping", ())
            ),
            finding_eligibility={
                prediction_id: FindingEligibility(**judgment)
                for prediction_id, judgment in value.get("finding_eligibility", {}).items()
            },
            semantic_matches=tuple(
                SemanticMatchJudgment.from_dict(item)
                for item in value.get("semantic_matches", ())
            ),
            value_verifications=tuple(
                ValueVerificationJudgment(**item)
                for item in value.get("value_verifications", ())
            ),
            run_id=value.get("run_id"),
            trial_index=value.get("trial_index"),
            experiment_id=value.get("experiment_id"),
            target_fingerprint=value.get("target_fingerprint"),
            benchmark_release_id=value.get("benchmark_release_id"),
            formal_mode=value.get("formal_mode"),
            evaluation_manifest_digest=value.get("evaluation_manifest_digest"),
            efficiency=(
                None
                if value.get("efficiency") is None
                else EfficiencyObservation(
                    **{
                        field_name: value["efficiency"].get(field_name)
                        for field_name in (
                            "input_tokens",
                            "output_tokens",
                            "model_turn_count",
                            "tool_call_count",
                            "python_execution_count",
                            "wall_clock_time",
                            "provider_reported_cost",
                        )
                    }
                )
            ),
            continuation_context=value.get("continuation_context", {}),
            continuation_request_id=value.get("continuation_request_id"),
            accumulated_adjudications=EvaluationAdjudications.from_dict(
                value.get("accumulated_adjudications", {})
            ),
        )

    def to_trial_evaluation(self) -> "TrialEvaluation":
        if self.trial_index is None:
            raise EvaluationConfigurationError(
                "pending evaluation has no originating trial_index"
            )
        return TrialEvaluation(
            trial_index=self.trial_index,
            pending=True,
            pending_reason=self.pending_reason,
            experiment_id=self.experiment_id,
            target_fingerprint=self.target_fingerprint,
            benchmark_release_id=self.benchmark_release_id,
            formal_mode=self.formal_mode,
            evaluation_manifest_digest=self.evaluation_manifest_digest,
        )


def save_case_evaluation_record(
    record: CaseEvaluationRecord | PendingCaseEvaluationRecord,
    path: str | Path,
    *,
    refuse_overwrite: bool = True,
) -> None:
    destination = Path(path)
    if refuse_overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite evaluation record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_case_evaluation_record(
    path: str | Path,
) -> CaseEvaluationRecord | PendingCaseEvaluationRecord:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load evaluation record from {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("evaluation record root must be an object")
    if value.get("record_type") == "pending_case_evaluation":
        return PendingCaseEvaluationRecord.from_dict(value)  # type: ignore[return-value]
    return CaseEvaluationRecord.from_dict(value)


def _match(
    matcher: SemanticMatcher,
    request: SemanticMatchRequest,
    trace: _EvaluationTrace | None = None,
    *,
    allow_uncertain: bool = False,
) -> SemanticMatchResult:
    try:
        result = SemanticMatchResult(matcher.match(request))
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigurationError(
            "semantic matcher returned an invalid result"
        ) from exc
    if _explicit_representation_conflict(request):
        result = SemanticMatchResult.NO_MATCH
    if trace is not None:
        _append_semantic_judgment(
            trace, SemanticMatchJudgment(request, result)
        )
    if result == SemanticMatchResult.UNCERTAIN and not allow_uncertain:
        raise EvaluationPendingAdjudication(
            f"semantic match remains UNCERTAIN for {request.predicted_id!r} and "
            f"{request.reference_id!r}",
            pending_type="semantic_uncertain",
            continuation_context={
                "route_owner": "SEMANTIC_MATCH_RESOLVER",
                "required_next_action": "RESOLVE_SEMANTIC_MATCH",
                "continuation_request_id": continuation_request_id(
                    "semantic_match", request.to_dict()
                ),
                "continuation_request": request.to_dict(),
                "semantic_match_request": request.to_dict(),
            },
        )
    return result


def _explicit_representation_conflict(request: SemanticMatchRequest) -> bool:
    """Guard against an unqualified peak/mean representation mismatch."""

    if request.purpose != SemanticMatchPurpose.FINDING:
        return False
    left = request.predicted_statement.casefold()
    right = request.reference_statement.casefold()
    location_terms = ("location", "coordinate", "centroid", "point", "position")
    if not any(term in left or term in right for term in location_terms):
        return False
    peak_left = any(term in left for term in ("peak", "maximum", "maximizing", "argmax"))
    peak_right = any(term in right for term in ("peak", "maximum", "maximizing", "argmax"))
    mean_left = any(term in left for term in ("mean", "average", "centroid"))
    mean_right = any(term in right for term in ("mean", "average", "centroid"))
    return (peak_left and mean_right and not (peak_right and mean_left)) or (
        peak_right and mean_left and not (peak_left and mean_right)
    )


def _dedup_values_do_not_conflict(
    finding: PredictedAtomicFinding,
    representative: PredictedAtomicFinding,
    unit_converter: UnitConverter | None,
) -> bool:
    left = finding.value
    right = representative.value
    if left is None or right is None:
        return True
    if isinstance(left, (str, list)) and isinstance(right, (str, list)):
        if isinstance(left, str) and isinstance(right, str):
            return True
        if (
            isinstance(left, list)
            and isinstance(right, list)
            and all(isinstance(item, str) for item in left)
            and all(isinstance(item, str) for item in right)
        ):
            return True
    numeric_left = _finite_number(left) or _numeric_list(left)
    numeric_right = _finite_number(right) or _numeric_list(right)
    if not (numeric_left and numeric_right):
        return False
    normalized_left = left
    if finding.unit != representative.unit:
        if (
            finding.unit is None
            or representative.unit is None
            or unit_converter is None
        ):
            return False
        normalized_left = unit_converter(left, finding.unit, representative.unit)
        if normalized_left is None:
            return False
    if _finite_number(normalized_left) and _finite_number(right):
        return float(normalized_left) == float(right)
    if _numeric_list(normalized_left) and _numeric_list(right):
        return len(normalized_left) == len(right) and all(
            float(left_item) == float(right_item)
            for left_item, right_item in zip(normalized_left, right, strict=True)
        )
    return False


def _deduplicate_findings(
    prediction: ExtractedPrediction,
    matcher: SemanticMatcher,
    *,
    scientific_question: str,
    case_context: Mapping[str, Any],
    finding_goal: str,
    unit_converter: UnitConverter | None,
    trace: _EvaluationTrace,
) -> tuple[tuple[PredictedAtomicFinding, ...], tuple[DeduplicationJudgment, ...]]:
    representatives: list[PredictedAtomicFinding] = []
    mapping: list[DeduplicationJudgment] = []
    for finding in prediction.findings:
        duplicate_of: PredictedAtomicFinding | None = None
        first_external_pending = None
        for representative in representatives:
            # Numerically conflicting equal-unit values, or incompatible
            # values without an installed converter, can never merge under
            # the fixed conjunction below. Skip only such impossible edges.
            if (finding.unit == representative.unit or unit_converter is None) and not _dedup_values_do_not_conflict(
                finding, representative, unit_converter
            ):
                continue
            request = SemanticMatchRequest(
                purpose=SemanticMatchPurpose.FINDING_DEDUPLICATION,
                predicted_statement=finding.statement,
                reference_statement=representative.statement,
                predicted_id=finding.prediction_id,
                reference_id=representative.prediction_id,
                scientific_question=scientific_question,
                case_context=case_context,
                finding_goal=finding_goal,
                predicted_value=finding.value,
                predicted_unit=finding.unit,
                reference_value=representative.value,
                reference_unit=representative.unit,
            )
            try:
                result = _match(matcher, request, trace, allow_uncertain=True)
            except EvaluationPendingAdjudication as exc:
                if exc.pending_type != "external_semantic_match":
                    raise
                if first_external_pending is None:
                    first_external_pending = exc
                continue
            if result == SemanticMatchResult.MATCH and _dedup_values_do_not_conflict(
                finding, representative, unit_converter
            ):
                duplicate_of = representative
                break
        # Earlier unresolved comparisons can change the chosen representative.
        # Do not merge or discover later findings with a guessed representative set.
        if first_external_pending is not None:
            raise first_external_pending
        if duplicate_of is None:
            representatives.append(finding)
            mapping.append(
                DeduplicationJudgment(finding.prediction_id, finding.prediction_id)
            )
        else:
            mapping.append(
                DeduplicationJudgment(
                    finding.prediction_id, duplicate_of.prediction_id
                )
            )
    return tuple(representatives), tuple(mapping)


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


def _numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all(_finite_number(item) for item in value)


def _unit_frame_request_payload(
    predicted: PredictedAtomicFinding,
    reference: ReferenceFinding,
    explicit_policy: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    policy_id = (
        explicit_policy.get("scientific_finding_policy_key")
        if isinstance(explicit_policy, Mapping)
        else None
    )
    return {
        "prediction_id": predicted.prediction_id,
        "reference_id": reference.finding_id,
        "predicted_value": predicted.value,
        "predicted_unit": predicted.unit,
        "reference_value": reference.value,
        "reference_unit": reference.unit,
        "verification_policy_id": policy_id,
        "unit_frame_authority": (
            explicit_policy.get("unit_frame_authority")
            if isinstance(explicit_policy, Mapping)
            else None
        ),
    }


def _normalize_unit(
    predicted: PredictedAtomicFinding,
    reference: ReferenceFinding,
    unit_converter: UnitConverter | None,
    explicit_policy: Mapping[str, Any] | None = None,
) -> FindingValue:
    # A typed continuation may resolve one previously ambiguous relationship.
    # It is accepted only when the request identity was derived from this
    # exact prediction/reference pair; no unit alias is inferred here.
    resolution = (
        explicit_policy.get("unit_frame_resolution")
        if isinstance(explicit_policy, Mapping)
        else None
    )
    if isinstance(resolution, Mapping):
        expected_request = _unit_frame_request_payload(
            predicted, reference, explicit_policy
        )
        expected_request_id = continuation_request_id("unit_frame", expected_request)
        # Always parse the persisted response through the typed contract.  A
        # plain mapping is not allowed to bypass provenance/status/value
        # validation merely because it happens to contain a request_id.
        typed_resolution = UnitFrameResolution.from_dict(resolution)
        if typed_resolution.request_id != expected_request_id or typed_resolution.request != expected_request:
            raise EvaluationConfigurationError(
                "unit/frame resolution does not match the unresolved request"
            )
        if typed_resolution.status == "RESOLVED":
            return typed_resolution.canonical_value
        if typed_resolution.status == "NO_MATCH":
            return None
    if predicted.unit == reference.unit:
        return predicted.value
    # A coordinate-frame label is meaningful only when the compiled case
    # policy explicitly delegates frame interpretation to reader metadata.
    # This is deliberately case-aware: no global alias or Python value shape
    # can manufacture equivalence with a null GT unit.
    frame_authority = (
        explicit_policy.get("unit_frame_authority", {})
        if isinstance(explicit_policy, Mapping)
        else {}
    )
    coordinate_scale = (
        frame_authority.get("coordinate_or_numeric_scale")
        if isinstance(frame_authority, Mapping)
        else None
    )
    normalized_predicted_unit = " ".join(str(predicted.unit or "").casefold().split())
    if (
        reference.unit is None
        and coordinate_scale == "CASE_READER_METADATA"
        and normalized_predicted_unit
        in {
            "dataset coordinates",
            "dataset coordinate units",
            "dataset coordinate frame",
            "dataset's coordinate frame",
            "stored cartesian coordinates",
            "cartesian coordinates",
        }
    ):
        return predicted.value
    if predicted.unit is None or reference.unit is None or unit_converter is None:
        raise EvaluationPendingAdjudication(
            f"unit relationship is unresolved for prediction {predicted.prediction_id!r} "
            f"and reference {reference.finding_id!r}",
            pending_type="unit_relationship",
            continuation_context={
                "route_owner": "UNIT_FRAME_RESOLVER",
                "required_next_action": "RESOLVE_UNIT_FRAME",
                "continuation_request_id": continuation_request_id(
                    "unit_frame",
                    _unit_frame_request_payload(predicted, reference, explicit_policy),
                ),
                "continuation_request": _unit_frame_request_payload(
                    predicted, reference, explicit_policy
                ),
                "unit_frame_request": _unit_frame_request_payload(
                    predicted, reference, explicit_policy
                ),
            },
        )
    converted = unit_converter(predicted.value, predicted.unit, reference.unit)
    if converted is None:
        raise EvaluationPendingAdjudication(
            f"unit conversion is unresolved for prediction {predicted.prediction_id!r} "
            f"and reference {reference.finding_id!r}",
            pending_type="unit_relationship",
            continuation_context={
                "route_owner": "UNIT_FRAME_RESOLVER",
                "required_next_action": "RESOLVE_UNIT_FRAME",
                "continuation_request_id": continuation_request_id(
                    "unit_frame",
                    _unit_frame_request_payload(predicted, reference, explicit_policy),
                ),
                "continuation_request": _unit_frame_request_payload(
                    predicted, reference, explicit_policy
                ),
                "unit_frame_request": _unit_frame_request_payload(
                    predicted, reference, explicit_policy
                ),
            },
        )
    return converted


def verify_finding_value(
    predicted: PredictedAtomicFinding,
    reference: ReferenceFinding,
    *,
    unit_converter: UnitConverter | None = None,
    explicit_policy: Mapping[str, Any] | str | None = None,
) -> bool:
    """Execute only the deterministic rule explicitly supported by the GT.

    Semantic matching is authoritative when no applicable deterministic rule is
    present.  In particular, string values and untyped numeric lists do not
    acquire hidden exact or spatial semantics from their Python representation.
    """

    verified, _ = _verify_finding_value(
        predicted,
        reference,
        unit_converter=unit_converter,
        explicit_policy=explicit_policy,
    )
    return verified


def finding_verification_mode(
    reference: ReferenceFinding,
    explicit_policy: Mapping[str, Any] | str | None = None,
) -> FindingVerificationMode:
    """Expose the explicit GT verifier rule to Finding semantic matching."""

    declared_mode = (
        explicit_policy.get("verification_mode")
        if isinstance(explicit_policy, Mapping)
        else explicit_policy
    )
    if declared_mode is not None:
        try:
            mode = FindingVerificationMode(str(declared_mode))
        except ValueError as exc:
            raise EvaluationConfigurationError(
                f"unsupported explicit Finding verification mode: {declared_mode!r}"
            ) from exc
        parameters = (
            explicit_policy.get("verification_parameters")
            if isinstance(explicit_policy, Mapping)
            else None
        )
        if parameters is not None:
            expected = (
                reference.verification.model_dump(mode="json")
                if reference.verification is not None
                else {"absolute_tolerance": None, "relative_tolerance": None, "spatial_tolerance": None}
            )
            if dict(parameters) != expected:
                raise EvaluationConfigurationError(
                    "explicit Finding verification policy conflicts with GroundTruth"
                )
        if mode == FindingVerificationMode.EXACT_DISCRETE_NUMERIC and not (
            isinstance(reference.value, int) and not isinstance(reference.value, bool)
        ):
            raise EvaluationConfigurationError(
                "exact_discrete_numeric requires an integer reference value"
            )
        if mode == FindingVerificationMode.EXACT_IDENTITY and not isinstance(
            reference.value, str
        ):
            raise EvaluationConfigurationError(
                "exact_identity requires a string reference value"
            )
        if mode == FindingVerificationMode.COMPONENTWISE_VECTOR and not (
            _numeric_list(reference.value) and bool(reference.value)
        ):
            raise EvaluationConfigurationError(
                "componentwise_vector requires a non-empty numeric vector reference value"
            )
        verification = reference.verification
        if mode == FindingVerificationMode.SCALAR_TOLERANCE and (
            verification is None
            or (verification.absolute_tolerance is None and verification.relative_tolerance is None)
        ):
            raise EvaluationConfigurationError("scalar_tolerance requires an explicit GT scalar tolerance")
        if mode in {FindingVerificationMode.SPATIAL_EUCLIDEAN, FindingVerificationMode.COMPONENTWISE_VECTOR} and (
            verification is None or verification.spatial_tolerance is None
        ):
            raise EvaluationConfigurationError("spatial verification requires an explicit GT spatial tolerance")
        return mode
    verification = reference.verification
    if verification is not None and verification.spatial_tolerance is not None:
        return FindingVerificationMode.SPATIAL_EUCLIDEAN
    if verification is not None and (
        verification.absolute_tolerance is not None
        or verification.relative_tolerance is not None
    ):
        return FindingVerificationMode.SCALAR_TOLERANCE
    return FindingVerificationMode.SEMANTIC_ONLY


def _verify_finding_value(
    predicted: PredictedAtomicFinding,
    reference: ReferenceFinding,
    *,
    unit_converter: UnitConverter | None = None,
    explicit_policy: Mapping[str, Any] | str | None = None,
) -> tuple[bool, str]:
    """Return ``(verified, applied_rule)`` after semantic MATCH."""

    reference_value = reference.value
    if reference_value is None:
        return True, "semantic_only"

    if isinstance(reference_value, bool):  # GroundTruth rejects this; fail closed.
        raise EvaluationConfigurationError("boolean ReferenceFinding values are invalid")
    verification = reference.verification
    mode = finding_verification_mode(reference, explicit_policy)

    if mode == FindingVerificationMode.EXACT_DISCRETE_NUMERIC:
        if predicted.value is None:
            return False, "exact_discrete_numeric"
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        return (
            _finite_number(predicted_value)
            and float(predicted_value) == reference_value,
            "exact_discrete_numeric",
        )

    if mode == FindingVerificationMode.EXACT_IDENTITY:
        if predicted.value is None:
            return False, "exact_identity"
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        return predicted_value == reference_value, "exact_identity"

    if mode == FindingVerificationMode.SCALAR_TOLERANCE:
        if not _finite_number(reference_value):
            raise EvaluationConfigurationError(
                "scalar tolerance is incompatible with ReferenceFinding.value"
            )
        if predicted.value is None:
            return False, "scalar_tolerance"
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _finite_number(predicted_value):
            return False, "scalar_tolerance"
        assert verification is not None
        absolute = (
            0.0
            if verification.absolute_tolerance is None
            else verification.absolute_tolerance
        )
        relative = (
            0.0
            if verification.relative_tolerance is None
            else verification.relative_tolerance
        )
        return (
            abs(float(predicted_value) - float(reference_value))
            <= absolute + relative * abs(float(reference_value)),
            "scalar_tolerance",
        )

    if mode == FindingVerificationMode.SPATIAL_EUCLIDEAN:
        if not _numeric_list(reference_value) or not reference_value:
            raise EvaluationConfigurationError(
                "spatial tolerance is incompatible with ReferenceFinding.value"
            )
        if predicted.value is None:
            return False, "spatial_euclidean"
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _numeric_list(predicted_value) or len(predicted_value) != len(reference_value):
            return False, "spatial_euclidean"
        distance = math.sqrt(
            sum(
                (float(predicted_item) - float(reference_item)) ** 2
                for predicted_item, reference_item in zip(
                    predicted_value, reference_value, strict=True
                )
            )
        )
        assert verification is not None and verification.spatial_tolerance is not None
        return distance <= verification.spatial_tolerance, "spatial_euclidean"

    if mode == FindingVerificationMode.COMPONENTWISE_VECTOR:
        if not _numeric_list(reference_value) or not reference_value:
            raise EvaluationConfigurationError(
                "componentwise vector tolerance is incompatible with ReferenceFinding.value"
            )
        if predicted.value is None:
            return False, "componentwise_vector"
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _numeric_list(predicted_value) or len(predicted_value) != len(reference_value):
            return False, "componentwise_vector"
        assert verification is not None and verification.spatial_tolerance is not None
        tolerance = float(verification.spatial_tolerance)
        return (
            all(
                abs(float(predicted_item) - float(reference_item)) <= tolerance
                for predicted_item, reference_item in zip(
                    predicted_value, reference_value, strict=True
                )
            ),
            "componentwise_vector",
        )

    return True, "semantic_only"


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, cost: int) -> int:
    forward_index = len(graph[source])
    reverse_index = len(graph[target])
    graph[source].append(_FlowEdge(target, reverse_index, 1, cost))
    graph[target].append(_FlowEdge(source, forward_index, 0, -cost))
    return forward_index


def _maximum_weight_assignment(
    prediction_ids: Sequence[str],
    references: Sequence[ReferenceFinding],
    verified_edges: set[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    """Lexicographically maximize core matches, then all valid matches."""

    if not prediction_ids or not references or not verified_edges:
        return ()
    prediction_count = len(prediction_ids)
    reference_count = len(references)
    source = 0
    prediction_offset = 1
    reference_offset = prediction_offset + prediction_count
    sink = reference_offset + reference_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    tracked: list[tuple[str, str, int, int]] = []
    reference_by_id = {reference.finding_id: reference for reference in references}
    core_bonus = min(prediction_count, reference_count) + 1

    for prediction_index, prediction_id in enumerate(prediction_ids):
        prediction_node = prediction_offset + prediction_index
        _add_flow_edge(graph, source, prediction_node, 0)
        _add_flow_edge(graph, prediction_node, sink, 0)
        for reference_index, reference in enumerate(references):
            if (prediction_id, reference.finding_id) not in verified_edges:
                continue
            weight = 1 + (
                core_bonus if reference.importance == FindingImportance.CORE else 0
            )
            edge_index = _add_flow_edge(
                graph,
                prediction_node,
                reference_offset + reference_index,
                -weight,
            )
            tracked.append(
                (prediction_id, reference.finding_id, prediction_node, edge_index)
            )
    for reference_index in range(reference_count):
        _add_flow_edge(graph, reference_offset + reference_index, sink, 0)

    for _ in range(prediction_count):
        distance = [math.inf] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distance[source] = 0
        for _relaxation in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] == math.inf:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distance[node] + edge.cost
                    if candidate < distance[edge.to]:
                        distance[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            raise EvaluationConfigurationError("Finding assignment graph is infeasible")
        node = sink
        while node != source:
            prior_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior_node

    selected = [
        (prediction_id, finding_id)
        for prediction_id, finding_id, node, edge_index in tracked
        if graph[node][edge_index].capacity == 0
    ]
    selected.sort(key=lambda pair: (prediction_ids.index(pair[0]), pair[1]))
    # Guard against an invalid reference identifier being introduced above.
    if any(finding_id not in reference_by_id for _, finding_id in selected):
        raise EvaluationConfigurationError("Finding assignment produced an unknown reference")
    return tuple(selected)


class CaseEvaluator:
    """Deterministic orchestration around versioned semantic components."""

    def __init__(
        self,
        extractor: PredictionExtractor,
        semantic_matcher: SemanticMatcher,
        eligibility_judge: FindingEligibilityJudge,
        evaluation_manifest: EvaluationManifest,
        *,
        unit_converter: UnitConverter | None = None,
        finding_requirement_contract: FindingRequirementContract | Mapping[str, Any] | None = None,
        scientific_evaluation_contract: ScientificEvaluationContract | Mapping[str, Any] | None = None,
        finding_verification_policy: Mapping[str, Any] | None = None,
        evaluation_mode: str = "DEVELOPMENT_EVALUATION",
        novel_o_materializer: Callable[[OperationalizationBundle], Any] | None = None,
        branch_execution_evidence: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(evaluation_manifest, EvaluationManifest):
            evaluation_manifest = EvaluationManifest.from_dict(evaluation_manifest)
        self.extractor = extractor
        self.semantic_matcher = semantic_matcher
        self.eligibility_judge = eligibility_judge
        self.evaluation_manifest = evaluation_manifest
        self.unit_converter = unit_converter
        self.evaluation_mode = str(evaluation_mode).strip().upper()
        if self.evaluation_mode not in {"DEVELOPMENT_EVALUATION", "FORMAL_EVALUATION"}:
            raise EvaluationConfigurationError(
                "evaluation_mode must be DEVELOPMENT_EVALUATION or FORMAL_EVALUATION"
            )
        if self.evaluation_mode == "FORMAL_EVALUATION" and scientific_evaluation_contract is None:
            raise EvaluationConfigurationError(
                "FORMAL_EVALUATION requires the frozen ScientificEvaluationContract"
            )
        # A novel O is never allowed to supply its own G(O).  The callback sees
        # only the complete OperationalizationBundle, not the adjudicated
        # branch's claimed Findings.  It is deliberately optional so callers
        # without an executable route receive a pending record instead of a
        # silently trusted score.
        if novel_o_materializer is not None and not callable(novel_o_materializer):
            raise EvaluationConfigurationError(
                "novel_o_materializer must be callable when supplied"
            )
        self.novel_o_materializer = novel_o_materializer
        if branch_execution_evidence is not None and self.evaluation_mode != "DEVELOPMENT_EVALUATION":
            raise EvaluationConfigurationError("branch_execution_evidence is development-only; formal execution authority remains the SEC")
        self.branch_execution_evidence = _json_value(dict(branch_execution_evidence or {}), "branch_execution_evidence")
        self.supplemental_materializer = None
        if scientific_evaluation_contract is not None and not isinstance(
            scientific_evaluation_contract, ScientificEvaluationContract
        ):
            value = dict(scientific_evaluation_contract)
            scientific_evaluation_contract = ScientificEvaluationContract(
                case_id=str(value.get("case_id", "")),
                enumerated_valid_O=tuple(value.get("enumerated_valid_O", ()) or ()),
                explicitly_invalid_O=tuple(value.get("explicitly_invalid_O", ()) or ()),
                unenumerated_policy=str(value.get("unenumerated_policy", "")),
                G_of_O=tuple(value.get("G_of_O", ()) or ()),
                finding_requirement_contract=value.get("finding_requirement_contract", {}),
                adequate_core_sets=tuple(tuple(item) for item in value.get("adequate_core_sets", ()) or ()),
                tolerances=value.get("tolerances", {}),
                adjudication_provenance=tuple(value.get("adjudication_provenance", ()) or ()),
                status=str(value.get("status", "DRAFT_REQUIRES_HUMAN_CONFIRMATION")),
                adjudicated_findings=tuple(value.get("adjudicated_findings", ()) or ()),
                source_semantic_contract_sha256=value.get("source_semantic_contract_sha256"),
            )
        self.scientific_evaluation_contract = scientific_evaluation_contract
        policy_source: Mapping[str, Any] | None = finding_verification_policy
        if policy_source is None and scientific_evaluation_contract is not None:
            policy_source = {
                "policies": scientific_evaluation_contract.tolerances.get(
                    "finding_verification_policies", ()
                )
            }
        self.finding_verification_policy = policy_source
        self.finding_verification_policy_index = policy_index(policy_source)
        if self.evaluation_mode == "FORMAL_EVALUATION":
            finding_requirement_contract = scientific_evaluation_contract.finding_requirement_contract
        self.finding_requirement_contract = (
            finding_requirement_contract
            if isinstance(finding_requirement_contract, FindingRequirementContract)
            else FindingRequirementContract.from_mapping(finding_requirement_contract)
        )
        if unit_converter is not None and evaluation_manifest.unit_converter_version is None:
            raise EvaluationConfigurationError(
                "a nontrivial unit converter requires unit_converter_version"
            )
        if unit_converter is None and evaluation_manifest.unit_converter_version is not None:
            raise EvaluationConfigurationError(
                "unit_converter_version requires a configured unit converter"
            )

    def evaluate_run(
        self,
        run_record: RunRecord,
        case_input: BenchmarkCaseInput,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        *,
        adjudications: EvaluationAdjudications | None = None,
    ) -> CaseEvaluationResult:
        record = self.evaluate_run_record(
            run_record,
            case_input,
            case_metadata,
            ground_truth,
            adjudications=adjudications,
        )
        if isinstance(record, PendingCaseEvaluationRecord):
            raise EvaluationPendingAdjudication(
                record.pending_reason, pending_type=record.pending_type
            )
        return record.result

    def evaluate_run_record(
        self,
        run_record: RunRecord,
        case_input: BenchmarkCaseInput,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        *,
        adjudications: EvaluationAdjudications | None = None,
    ) -> CaseEvaluationRecord | PendingCaseEvaluationRecord:
        condition, validated_gt = self._validate_case(
            run_record.case_id, case_metadata, ground_truth
        )
        efficiency = EfficiencyObservation(
            input_tokens=run_record.input_tokens,
            output_tokens=run_record.output_tokens,
            model_turn_count=run_record.model_turn_count,
            tool_call_count=getattr(
                run_record,
                "tool_call_count",
                run_record.python_execution_count,
            ),
            python_execution_count=run_record.python_execution_count,
            wall_clock_time=run_record.wall_clock_time,
            provider_reported_cost=run_record.provider_reported_cost,
        )
        if run_record.run_status == RunStatus.INFRASTRUCTURE_INVALID:
            result = CaseEvaluationResult(
                case_id=run_record.case_id,
                condition=condition,
                run_status=run_record.run_status,
                eligible_for_scientific_aggregation=False,
                metrics=None,
            )
            return self._attach_run_identity(
                self._status_record(result, efficiency), run_record
            )
        if run_record.run_status == RunStatus.MODEL_NONCOMPLETION:
            result = self._noncompletion_result(
                run_record.case_id, condition, case_metadata, efficiency
            )
            return self._attach_run_identity(
                self._status_record(result, efficiency), run_record
            )
        if not run_record.final_response or not run_record.final_response.strip():
            raise EvaluationConfigurationError(
                "COMPLETED RunRecord must contain a non-empty final_response"
            )
        return self._attach_run_identity(
            self.evaluate_response_record(
            case_input,
            case_metadata,
            validated_gt,
            final_response=run_record.final_response,
            efficiency=efficiency,
            adjudications=adjudications,
            ),
            run_record,
        )

    def evaluate_response(
        self,
        case_input: BenchmarkCaseInput,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        *,
        final_response: str,
        efficiency: EfficiencyObservation,
        adjudications: EvaluationAdjudications | None = None,
    ) -> CaseEvaluationResult:
        record = self.evaluate_response_record(
            case_input,
            case_metadata,
            ground_truth,
            final_response=final_response,
            efficiency=efficiency,
            adjudications=adjudications,
        )
        if isinstance(record, PendingCaseEvaluationRecord):
            raise EvaluationPendingAdjudication(
                record.pending_reason, pending_type=record.pending_type
            )
        return record.result

    def evaluate_response_record(
        self,
        case_input: BenchmarkCaseInput,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        *,
        final_response: str,
        efficiency: EfficiencyObservation,
        adjudications: EvaluationAdjudications | None = None,
    ) -> CaseEvaluationRecord | PendingCaseEvaluationRecord:
        condition, validated_gt = self._validate_case(
            case_metadata.case_id, case_metadata, ground_truth
        )
        response = _text(final_response, "final_response")
        extraction = self.extractor.extract(
            ExtractionRequest(
                scientific_question=case_input.scientific_question,
                case_context=case_input.case_context.model_dump(mode="json"),
                principal_operationalization_dimensions=tuple(
                    case_metadata.principal_operationalization_dimensions
                ),
                unresolved_operationalization_dimensions=tuple(
                    case_metadata.unresolved_operationalization_dimensions
                ),
                final_response=response,
                coordinate_semantics=case_input.flow_data.data_metadata.coordinate_system.model_dump(
                    mode="json"
                ),
            )
        )
        if not isinstance(extraction, ExtractedPrediction):
            raise EvaluationConfigurationError(
                "extractor must return ExtractedPrediction"
            )
        return self.evaluate_prediction_record(
            case_metadata,
            validated_gt,
            extraction,
            scientific_question=case_input.scientific_question,
            case_context=case_input.case_context.model_dump(mode="json"),
            efficiency=efficiency,
            adjudications=adjudications,
        )

    def evaluate_prediction(
        self,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        prediction: ExtractedPrediction,
        *,
        scientific_question: str,
        case_context: Mapping[str, Any],
        efficiency: EfficiencyObservation,
        adjudications: EvaluationAdjudications | None = None,
    ) -> CaseEvaluationResult:
        record = self.evaluate_prediction_record(
            case_metadata,
            ground_truth,
            prediction,
            scientific_question=scientific_question,
            case_context=case_context,
            efficiency=efficiency,
            adjudications=adjudications,
        )
        if isinstance(record, PendingCaseEvaluationRecord):
            raise EvaluationPendingAdjudication(
                record.pending_reason, pending_type=record.pending_type
            )
        return record.result

    def evaluate_prediction_record(
        self,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        prediction: ExtractedPrediction,
        *,
        scientific_question: str,
        case_context: Mapping[str, Any],
        efficiency: EfficiencyObservation,
        adjudications: EvaluationAdjudications | None = None,
    ) -> "CaseEvaluationRecord | PendingCaseEvaluationRecord":
        """Evaluate an extracted prediction, persisting unresolved judgments as a draft."""

        trace = _EvaluationTrace()
        trace.extracted_prediction = prediction
        try:
            return self._evaluate_prediction_record(
                case_metadata,
                ground_truth,
                prediction,
                scientific_question=scientific_question,
                case_context=case_context,
                efficiency=efficiency,
                adjudications=adjudications,
                _trace=trace,
            )
        except EvaluationPendingAdjudication as exc:
            condition, _ = self._validate_case(
                case_metadata.case_id, case_metadata, ground_truth
            )
            return PendingCaseEvaluationRecord(
                record_version="1",
                evaluation_manifest=self.evaluation_manifest,
                case_id=case_metadata.case_id,
                condition=condition,
                pending_type=exc.pending_type,
                pending_reason=str(exc),
                extracted_prediction=prediction,
                deduplication_mapping=trace.deduplication_mapping,
                finding_eligibility=dict(trace.finding_eligibility),
                semantic_matches=tuple(trace.semantic_matches),
                value_verifications=tuple(trace.value_verifications),
                efficiency=efficiency,
                continuation_context=(
                    dict(exc.continuation_context)
                    if exc.continuation_context
                    else dict(trace.continuation_context)
                ),
                continuation_request_id=(
                    (dict(exc.continuation_context) if exc.continuation_context else dict(trace.continuation_context)).get(
                        "continuation_request_id"
                    )
                ),
            )

    def finalize_pending_evaluation(
        self,
        pending_record: PendingCaseEvaluationRecord,
        case_input: BenchmarkCaseInput,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        *,
        adjudications: EvaluationAdjudications,
    ) -> CaseEvaluationRecord | PendingCaseEvaluationRecord:
        """Continue one persisted evaluation without repeating extraction.

        The persisted semantic, value, deduplication, and eligibility judgments
        are reused as the authoritative prefix.  Only comparisons not present
        in that prefix may reach the configured components.
        """

        if not isinstance(pending_record, PendingCaseEvaluationRecord):
            raise EvaluationConfigurationError(
                "finalize_pending_evaluation requires a PendingCaseEvaluationRecord"
            )
        if pending_record.case_id != case_metadata.case_id:
            raise EvaluationConfigurationError(
                "pending evaluation case_id does not match case metadata"
            )
        if pending_record.evaluation_manifest_digest != evaluation_manifest_digest(
            self.evaluation_manifest
        ):
            raise EvaluationConfigurationError(
                "pending evaluation was produced by a different evaluation manifest"
            )
        if pending_record.extracted_prediction is None:
            raise EvaluationConfigurationError(
                "pending evaluation has no extracted prediction to continue"
            )
        if pending_record.efficiency is None:
            raise EvaluationConfigurationError(
                "pending evaluation has no persisted efficiency observation"
            )
        if not isinstance(adjudications, EvaluationAdjudications):
            adjudications = EvaluationAdjudications.from_dict(adjudications)
        persisted = pending_record.accumulated_adjudications
        # Each owner returns only the decision it owns.  Accumulate those
        # typed decisions across hops; replacing the prior object would make
        # a later semantic/Finding continuation forget a completed novel O.
        adjudications = EvaluationAdjudications(
            novel_operationalization=(
                adjudications.novel_operationalization
                or persisted.novel_operationalization
            ),
            novel_findings={
                **persisted.novel_findings,
                **adjudications.novel_findings,
            },
            novel_finding_roles={
                **persisted.novel_finding_roles,
                **adjudications.novel_finding_roles,
            },
            semantic_resolutions=tuple(
                {
                    item.request_id: item
                    for item in (
                        *persisted.semantic_resolutions,
                        *adjudications.semantic_resolutions,
                    )
                }.values()
            ),
            unit_frame_resolutions=tuple(
                {
                    item.request_id: item
                    for item in (
                        *persisted.unit_frame_resolutions,
                        *adjudications.unit_frame_resolutions,
                    )
                }.values()
            ),
        )
        context = case_input.case_context.model_dump(mode="json")
        trace = _EvaluationTrace(
            extracted_prediction=pending_record.extracted_prediction,
            semantic_matches=list(pending_record.semantic_matches),
            value_verifications=list(pending_record.value_verifications),
            deduplication_mapping=pending_record.deduplication_mapping,
            finding_eligibility=dict(pending_record.finding_eligibility),
            continuation_context=dict(pending_record.continuation_context),
        )
        matcher = _CachedSemanticMatcher(
            pending_record.semantic_matches,
            self.semantic_matcher,
            adjudications.semantic_resolutions,
        )
        eligibility_judge = _CachedEligibilityJudge(
            pending_record.finding_eligibility, self.eligibility_judge
        )
        try:
            record = self._evaluate_prediction_record(
                case_metadata,
                ground_truth,
                pending_record.extracted_prediction,
                scientific_question=case_input.scientific_question,
                case_context=context,
                efficiency=pending_record.efficiency,
                adjudications=adjudications,
                _trace=trace,
                _semantic_matcher=matcher,
                _eligibility_judge=eligibility_judge,
            )
        except EvaluationPendingAdjudication as exc:
            condition, _ = self._validate_case(
                case_metadata.case_id, case_metadata, ground_truth
            )
            return replace(
                pending_record,
                condition=condition,
                pending_type=exc.pending_type,
                pending_reason=str(exc),
                deduplication_mapping=trace.deduplication_mapping,
                finding_eligibility=dict(trace.finding_eligibility),
                semantic_matches=tuple(trace.semantic_matches),
                value_verifications=tuple(trace.value_verifications),
                continuation_context=(
                    dict(exc.continuation_context)
                    if exc.continuation_context
                    else dict(trace.continuation_context)
                ),
                continuation_request_id=(
                    (dict(exc.continuation_context) if exc.continuation_context else dict(trace.continuation_context)).get(
                        "continuation_request_id"
                    )
                ),
                accumulated_adjudications=adjudications,
            )
        return replace(
            record,
            run_id=pending_record.run_id,
            trial_index=pending_record.trial_index,
            experiment_id=pending_record.experiment_id,
            target_fingerprint=pending_record.target_fingerprint,
            benchmark_release_id=pending_record.benchmark_release_id,
            formal_mode=pending_record.formal_mode,
        )

    def _evaluate_prediction_record(
        self,
        case_metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
        prediction: ExtractedPrediction,
        *,
        scientific_question: str,
        case_context: Mapping[str, Any],
        efficiency: EfficiencyObservation,
        adjudications: EvaluationAdjudications | None = None,
        _trace: _EvaluationTrace | None = None,
        _semantic_matcher: SemanticMatcher | None = None,
        _eligibility_judge: FindingEligibilityJudge | None = None,
    ) -> CaseEvaluationRecord:
        condition, validated_gt = self._validate_case(
            case_metadata.case_id, case_metadata, ground_truth
        )
        scientific_question = _text(scientific_question, "scientific_question")
        case_context = _json_value(case_context, "case_context")
        adjudications = adjudications or EvaluationAdjudications()
        trace = _trace or _EvaluationTrace()
        semantic_matcher = _semantic_matcher or self.semantic_matcher
        eligibility_judge = _eligibility_judge or self.eligibility_judge
        principal = tuple(case_metadata.principal_operationalization_dimensions)
        unresolved = set(case_metadata.unresolved_operationalization_dimensions)
        supplied_decisions = {
            decision.dimension: decision
            for decision in prediction.operationalization.decisions
        }
        unexpected_dimensions = set(supplied_decisions) - set(principal)
        if unexpected_dimensions:
            raise EvaluationConfigurationError(
                "extractor returned non-principal O dimensions: "
                + ", ".join(sorted(item.value for item in unexpected_dimensions))
            )
        decisions = {
            dimension: supplied_decisions.get(
                dimension,
                ExtractedOperationalizationDecision(
                    dimension=dimension,
                    status=ExtractionStatus.MISSING,
                ),
            )
            for dimension in principal
        }

        bundles = list(validated_gt.acceptable_operationalizations)
        finding_branches = list(validated_gt.findings_by_operationalization)
        o_evaluations, o_matches = self._evaluate_o_bundles(
            bundles,
            decisions,
            unresolved,
            scientific_question=scientific_question,
            case_context=case_context,
            finding_goal=case_metadata.finding_goal,
            trace=trace,
            semantic_matcher=semantic_matcher,
        )
        coherent = all(
            decision.status == ExtractionStatus.EXTRACTED
            for decision in decisions.values()
        )
        has_complete_match = any(
            all(value == 1 for value in evaluation.principal_dimension_matches.values())
            for evaluation in o_evaluations
        )
        novel_o_record: Mapping[str, Any] | None = None
        consistency_only_branch = None
        if coherent and not has_complete_match:
            novel = adjudications.novel_operationalization
            candidate_operationalization = (
                None
                if novel is None
                else (
                    novel.operationalization
                    if novel.operationalization is not None
                    else (
                        None
                        if novel.branch is None
                        else novel.branch.operationalization
                    )
                )
            )
            if novel is None or candidate_operationalization is None:
                raise EvaluationPendingAdjudication(
                    "coherent GT-outside Operationalization requires a complete candidate O",
                    pending_type="novel_operationalization_candidate",
                    continuation_context={
                        "route_owner": "NOVEL_O_COMPLETION",
                        "required_next_action": "COMPLETE_NOVEL_OPERATIONALIZATION",
                        "continuation_request_id": continuation_request_id(
                            "operationalization_completion",
                            _operationalization_request_payload(prediction.operationalization),
                        ),
                        "continuation_request": _operationalization_request_payload(
                            prediction.operationalization
                        ),
                        "operationalization_completion_request": _operationalization_request_payload(
                            prediction.operationalization
                        ),
                    },
                )
            self._validate_novel_operationalization(
                candidate_operationalization, principal, bundles
            )
            # Validate the complete candidate against the extracted response;
            # this does not consume any scientific-validity verdict.
            candidate_evaluations, _ = self._evaluate_o_bundles(
                (candidate_operationalization,),
                decisions,
                unresolved,
                scientific_question=scientific_question,
                case_context=case_context,
                finding_goal=case_metadata.finding_goal,
                trace=trace,
                semantic_matcher=semantic_matcher,
                allow_exact_self_match=True,
            )
            if not all(
                value == 1
                for value in candidate_evaluations[0].principal_dimension_matches.values()
            ):
                raise EvaluationConfigurationError(
                    "novel O candidate does not match the extracted model O"
                )

            # Frozen authority order: complete O -> materialize -> actual
            # G(O) -> blind scientific adjudication -> reference resolution.
            materialized_novel = self._materialize_novel_operationalization(
                candidate_operationalization,
                expected_dataset_id=case_metadata.dataset_id,
            )
            self._validate_novel_branch(
                materialized_novel,
                principal,
                bundles,
                require_materialization=True,
            )
            assert materialized_novel.execution_provenance is not None
            actual_g_digest = _canonical_json_sha256(
                materialized_novel.execution_provenance["G_of_O"]
            )
            if novel.status == AdjudicationStatus.UNRESOLVED:
                raise EvaluationPendingAdjudication(
                    "materialized GT-outside Operationalization requires blind scientific adjudication",
                    pending_type="novel_operationalization_scientific_adjudication",
                    continuation_context={
                        "route_owner": "NOVEL_O_SCIENTIFIC_ADJUDICATION",
                        "required_next_action": "BLINDED_SCIENTIFIC_ADJUDICATION",
                        "continuation_request_id": continuation_request_id(
                            "operationalization_adjudication",
                            {
                                "operationalization": candidate_operationalization.model_dump(mode="json"),
                                "materialization_id": materialized_novel.materialization_id,
                                "G_of_O_sha256": actual_g_digest,
                            },
                        ),
                        "continuation_request": {
                            "operationalization": candidate_operationalization.model_dump(mode="json"),
                            "materialization_id": materialized_novel.materialization_id,
                            "G_of_O_sha256": actual_g_digest,
                        },
                        "candidate_operationalization": candidate_operationalization.model_dump(mode="json"),
                        "materialization_id": materialized_novel.materialization_id,
                        "execution_status": "MATERIALIZED",
                        "G_of_O": dict(materialized_novel.execution_provenance["G_of_O"]),
                        "G_of_O_sha256": actual_g_digest,
                        "execution_provenance": {
                            "status": "MATERIALIZED", "reproducible": True,
                            "data_provenance": {
                                key: materialized_novel.execution_provenance["data_provenance"][key]
                                for key in ("dataset_id", "dataset_manifest_sha256", "effective_o_sha256",
                                            "materialization_plan", "compiled_plan_sha256", "executed_plan_sha256")
                            },
                        },
                    },
                )
            if (
                novel.adjudicated_materialization_id
                != materialized_novel.materialization_id
                or novel.adjudicated_g_of_o_sha256 != actual_g_digest
            ):
                raise EvaluationConfigurationError(
                    "novel O scientific adjudication is not bound to the actual materialized G(O)"
                )
            branch_id = materialized_novel.operationalization.operationalization_id
            novel_o_record = {
                "status": novel.status.value,
                "branch_id": branch_id,
                "materialization_id": materialized_novel.materialization_id,
                "G_of_O_sha256": actual_g_digest,
                "execution_status": "MATERIALIZED",
            }
            if novel.status == AdjudicationStatus.ACCEPTED:
                novel_evaluations, novel_matches = self._evaluate_o_bundles(
                    (materialized_novel.operationalization,),
                    decisions,
                    unresolved,
                    scientific_question=scientific_question,
                    case_context=case_context,
                    finding_goal=case_metadata.finding_goal,
                    trace=trace,
                    semantic_matcher=semantic_matcher,
                )
                novel_evaluation = novel_evaluations[0]
                if not all(
                    value == 1
                    for value in novel_evaluation.principal_dimension_matches.values()
                ):
                    raise EvaluationConfigurationError(
                        "trusted materialized novel O does not match the extracted model O"
                    )
                bundles.append(materialized_novel.operationalization)
                finding_branches.append(materialized_novel.findings)
                o_matches.update(novel_matches)
                o_evaluations.append(novel_evaluation)
            else:
                # Scientific rejection affects O validity, not whether the
                # reported Findings agree with the executed method. Its G(O)
                # is available for consistency but earns no O/F validity credit.
                consistency_only_branch = materialized_novel.findings

        deduplicated, deduplication_mapping = _deduplicate_findings(
            prediction,
            semantic_matcher,
            scientific_question=scientific_question,
            case_context=case_context,
            finding_goal=case_metadata.finding_goal,
            unit_converter=self.unit_converter,
            trace=trace,
        )
        trace.deduplication_mapping = deduplication_mapping
        eligibility: dict[str, FindingEligibility] = {}
        first_eligibility_pending = None
        for finding in deduplicated:
            try:
                judgment = eligibility_judge.judge(
                    FindingEligibilityRequest(
                        scientific_question=scientific_question,
                        case_context=case_context,
                        finding_goal=case_metadata.finding_goal,
                        finding=finding,
                    )
                )
            except EvaluationPendingAdjudication as exc:
                if exc.pending_type != 'external_eligibility':
                    raise
                if first_eligibility_pending is None:
                    first_eligibility_pending = exc
                continue
            if not isinstance(judgment, FindingEligibility):
                raise EvaluationConfigurationError(
                    "eligibility judge must return FindingEligibility"
                )
            eligibility[finding.prediction_id] = judgment
        trace.finding_eligibility = dict(eligibility)
        if first_eligibility_pending is not None:
            # Discover independent GT-blind tasks together; no partial F_app
            # denominator or score may be constructed until every item resolves.
            raise first_eligibility_pending
        f_app = tuple(
            finding
            for finding in deduplicated
            if eligibility[finding.prediction_id].in_f_app
        )
        branch_diagnostics: list[FindingBranchDiagnostic] = []
        finding_metric_inputs: list[FindingBranchEvaluation] = []
        for branch in (*finding_branches, *((consistency_only_branch,) if consistency_only_branch else ())):
            materialized_g_of_o = None
            execution_evidence_provenance = None
            branch_operation = next((bundle for bundle in bundles if bundle.operationalization_id == branch.operationalization_id), None)
            candidate_operationalization = ({d.dimension.value: d.statement for d in branch_operation.decisions}
                                           if branch_operation is not None else None)
            if (
                novel_o_record is not None
                and novel_o_record.get("branch_id") == branch.operationalization_id
                and "materialized_novel" in locals()
            ):
                materialized_g_of_o = dict(
                    materialized_novel.execution_provenance["G_of_O"]  # type: ignore[index]
                )
                candidate_operationalization = {d.dimension.value: d.statement for d in materialized_novel.operationalization.decisions}
                execution_evidence_provenance = materialized_novel.execution_provenance.get("data_provenance")
                if self.supplemental_materializer is not None:
                    supplemental = self.supplemental_materializer({
                        "execution": materialized_novel.execution_provenance,
                        "parameters": {
                            decision.dimension.value: decision.statement
                            for decision in materialized_novel.operationalization.decisions
                        },
                    })
                    if supplemental is not None:
                        materialized_g_of_o = supplemental
            elif self.scientific_evaluation_contract is not None:
                for execution_record in self.scientific_evaluation_contract.G_of_O:
                    if not isinstance(execution_record, Mapping):
                        continue
                    proposal_id = str(execution_record.get("proposal_id", ""))
                    if proposal_id.rsplit(":", 1)[-1] != branch.operationalization_id:
                        continue
                    execution = execution_record.get("execution")
                    if isinstance(execution, Mapping) and isinstance(
                        execution.get("G_of_O"), Mapping
                    ):
                        materialized_g_of_o = dict(execution["G_of_O"])
                        execution_evidence_provenance = execution.get("data_provenance")
                        if self.supplemental_materializer is not None:
                            supplemental = self.supplemental_materializer(execution_record)
                            if supplemental is not None:
                                materialized_g_of_o = supplemental
                        break
            elif self.evaluation_mode == "DEVELOPMENT_EVALUATION":
                evidence = self.branch_execution_evidence.get(branch.operationalization_id)
                if evidence is not None:
                    if not isinstance(evidence, Mapping):
                        raise EvaluationConfigurationError("branch execution evidence must be an object")
                    bound = dict(evidence)
                    supplied_digest = bound.pop("evidence_binding_sha256", None)
                    if supplied_digest != _canonical_json_sha256(bound):
                        raise EvaluationConfigurationError("branch execution evidence digest mismatch")
                    if (bound.get("case_id") != case_metadata.case_id
                            or bound.get("dataset_id") != case_metadata.dataset_id
                            or bound.get("operationalization") != candidate_operationalization):
                        raise EvaluationConfigurationError("branch execution evidence does not bind the current case and O")
                    if not isinstance(bound.get("materialized_G_of_O"), Mapping) or not bound["materialized_G_of_O"]:
                        raise EvaluationConfigurationError("branch execution evidence requires actual nonempty G(O)")
                    provenance = bound.get("execution_provenance")
                    if not isinstance(provenance, Mapping) or provenance.get("status") != "MATERIALIZED":
                        raise EvaluationConfigurationError("branch execution evidence is not materialized")
                    materialized_g_of_o = dict(bound["materialized_G_of_O"])
                    execution_evidence_provenance = {**dict(provenance),
                        "source_artifact_sha256": bound.get("source_artifact", {}).get("sha256"),
                        "evidence_binding_sha256": supplied_digest}
            from .open_field_selection import bind_open_field_selection
            branch, materialized_g_of_o = bind_open_field_selection(
                branch, f_app, materialized_g_of_o
            )
            metric_input, diagnostic = self._evaluate_finding_branch(
                branch,
                deduplicated,
                f_app,
                adjudications,
                scientific_question=scientific_question,
                case_context=case_context,
                finding_goal=case_metadata.finding_goal,
                trace=trace,
                semantic_matcher=semantic_matcher,
                requirement_contract=(
                    self.finding_requirement_contract
                    if condition.endswith("-F2") and branch is not consistency_only_branch
                    else None
                ),
                materialized_g_of_o=materialized_g_of_o,
                candidate_operationalization=candidate_operationalization,
                execution_evidence_provenance=execution_evidence_provenance,
            )
            if branch is not consistency_only_branch:
                finding_metric_inputs.append(metric_input)
            branch_diagnostics.append(diagnostic)

        conflicting = any(
            decision.status == ExtractionStatus.CONFLICTING
            for decision in decisions.values()
        )
        compatible = () if conflicting else tuple(
            bundle.operationalization_id
            for bundle in bundles
            if all(
                decisions[dimension].status
                in {ExtractionStatus.MISSING, ExtractionStatus.AMBIGUOUS}
                or o_matches[bundle.operationalization_id][dimension]
                == SemanticMatchResult.MATCH
                for dimension in principal
            )
        )
        if coherent and consistency_only_branch is not None:
            compatible += (consistency_only_branch.operationalization_id,)
        diagnostics_by_branch = {
            diagnostic.branch_id: diagnostic for diagnostic in branch_diagnostics
        }
        consistency_diagnostics: list[ConsistencyBranchDiagnostic] = []
        for branch_id in compatible if coherent else ():
            diagnostic = diagnostics_by_branch[branch_id]
            consistent_count = len(diagnostic.consistent_applicable_prediction_ids)
            c_score = 0.0 if not f_app else consistent_count / len(f_app)
            consistency_diagnostics.append(
                ConsistencyBranchDiagnostic(
                    branch_id,
                    len(f_app),
                    consistent_count,
                    c_score,
                )
            )
        if consistency_diagnostics:
            best_c_value = max(item.c_score for item in consistency_diagnostics)
            best_c = tuple(
                item.branch_id
                for item in consistency_diagnostics
                if item.c_score == best_c_value
            )
            selected_consistency = diagnostics_by_branch[best_c[0]]
            consistent_ids = set(
                selected_consistency.consistent_applicable_prediction_ids
            )
            consistency_input = ConsistencyEvaluation(
                {
                    finding.prediction_id: int(finding.prediction_id in consistent_ids)
                    for finding in f_app
                }
            )
        else:
            best_c = ()
            consistency_input = ConsistencyEvaluation({})

        consistency_input = replace(
            consistency_input,
            operationalization_determinate=coherent,
            unavailable_reason=(
                "INDETERMINATE_OPERATIONALIZATION" if not coherent
                else "NO_EXECUTABLE_O_EVIDENCE" if not compatible
                else "NO_APPLICABLE_FINDINGS" if not f_app
                else None
            ),
        )
        metric_input = CaseMetricInput(
            case_id=case_metadata.case_id,
            condition=condition,
            operationalization_branches=tuple(o_evaluations),
            finding_branches=tuple(finding_metric_inputs),
            consistency=consistency_input,
            efficiency=efficiency,
        )
        metrics = compute_case_metrics(metric_input)
        result = CaseEvaluationResult(
            case_id=case_metadata.case_id,
            condition=condition,
            run_status=RunStatus.COMPLETED,
            eligible_for_scientific_aggregation=True,
            metrics=metrics,
            deduplicated_prediction_ids=tuple(
                finding.prediction_id for finding in deduplicated
            ),
            f_app_prediction_ids=tuple(finding.prediction_id for finding in f_app),
            compatible_o_branches=compatible,
            best_c_branches=best_c,
            finding_branch_diagnostics=tuple(branch_diagnostics),
            consistency_branch_diagnostics=tuple(consistency_diagnostics),
        )
        return CaseEvaluationRecord(
            record_version="1",
            evaluation_manifest=self.evaluation_manifest,
            result=result,
            extracted_prediction=prediction,
            deduplication_mapping=deduplication_mapping,
            finding_eligibility=eligibility,
            semantic_matches=tuple(trace.semantic_matches),
            value_verifications=tuple(trace.value_verifications),
            operationalization_semantic_matches={
                branch_id: {
                    dimension.value: result
                    for dimension, result in matches.items()
                }
                for branch_id, matches in o_matches.items()
            },
            operationalization_branch_evaluations=metric_input.operationalization_branches,
            finding_branch_evaluations=metric_input.finding_branches,
            consistency_evaluation=metric_input.consistency,
            efficiency=efficiency,
            novel_operationalization_adjudication=novel_o_record,
            novel_finding_adjudications=tuple(trace.novel_finding_adjudications),
        )

    def _status_record(
        self,
        result: CaseEvaluationResult,
        efficiency: EfficiencyObservation,
    ) -> CaseEvaluationRecord:
        return CaseEvaluationRecord(
            record_version="1",
            evaluation_manifest=self.evaluation_manifest,
            result=result,
            extracted_prediction=None,
            deduplication_mapping=(),
            finding_eligibility={},
            semantic_matches=(),
            value_verifications=(),
            operationalization_semantic_matches={},
            operationalization_branch_evaluations=(),
            finding_branch_evaluations=(),
            consistency_evaluation=ConsistencyEvaluation({}),
            efficiency=efficiency,
        )

    @staticmethod
    def _attach_run_identity(
        record: CaseEvaluationRecord | PendingCaseEvaluationRecord, run_record: RunRecord
    ) -> CaseEvaluationRecord | PendingCaseEvaluationRecord:
        return replace(
            record,
            run_id=run_record.run_id,
            trial_index=run_record.trial_index,
            experiment_id=run_record.experiment_id or None,
            target_fingerprint=run_record.target_fingerprint,
            benchmark_release_id=run_record.benchmark_release_id or None,
            formal_mode=run_record.formal_mode,
        )

    def _evaluate_o_bundles(
        self,
        bundles: Sequence[OperationalizationBundle],
        decisions: Mapping[OperationalizationDimension, ExtractedOperationalizationDecision],
        unresolved: set[OperationalizationDimension],
        *,
        scientific_question: str,
        case_context: Mapping[str, Any],
        finding_goal: str,
        trace: _EvaluationTrace,
        semantic_matcher: SemanticMatcher,
        allow_exact_self_match: bool = False,
    ) -> tuple[
        list[OperationalizationBranchEvaluation],
        dict[str, dict[OperationalizationDimension, SemanticMatchResult]],
    ]:
        evaluations: list[OperationalizationBranchEvaluation] = []
        match_table: dict[str, dict[OperationalizationDimension, SemanticMatchResult]] = {}
        for bundle in bundles:
            reference_by_dimension = {
                decision.dimension: decision for decision in bundle.decisions
            }
            branch_matches: dict[OperationalizationDimension, SemanticMatchResult] = {}
            dimension_scores: dict[str, int] = {}
            unresolved_scores: dict[str, int] = {}
            for dimension, extracted in decisions.items():
                if extracted.status == ExtractionStatus.EXTRACTED:
                    predicted_statement = extracted.normalized_statement or ""
                    reference_statement = reference_by_dimension[dimension].statement
                    # A novel-O completion is derived from the extracted
                    # response.  If it preserves the exact statement (up to
                    # whitespace/case), that identity is deterministic and
                    # should not be sent through a second model semantic
                    # judgment that could spuriously reject a self-match.
                    if allow_exact_self_match and _canonical_statement_identity(predicted_statement) == _canonical_statement_identity(reference_statement):
                        result = SemanticMatchResult.MATCH
                    else:
                        result = _match(
                            semantic_matcher,
                            SemanticMatchRequest(
                                purpose=SemanticMatchPurpose.OPERATIONALIZATION,
                                predicted_statement=predicted_statement,
                                predicted_evidence_span=extracted.evidence_span,
                                reference_statement=reference_statement,
                                predicted_id=dimension.value,
                                reference_id=dimension.value,
                                branch_id=bundle.operationalization_id,
                                dimension=dimension,
                                scientific_question=scientific_question,
                                case_context=case_context,
                                finding_goal=finding_goal,
                            ),
                            trace,
                        )
                else:
                    result = SemanticMatchResult.NO_MATCH
                branch_matches[dimension] = result
                score = int(
                    extracted.status == ExtractionStatus.EXTRACTED
                    and result == SemanticMatchResult.MATCH
                )
                dimension_scores[dimension.value] = score
                if dimension in unresolved:
                    unresolved_scores[dimension.value] = score
            match_table[bundle.operationalization_id] = branch_matches
            evaluations.append(
                OperationalizationBranchEvaluation(
                    bundle.operationalization_id,
                    dimension_scores,
                    unresolved_scores,
                )
            )
        return evaluations, match_table

    def _evaluate_finding_branch(
        self,
        branch: OperationalizationFindingBranch,
        predictions: Sequence[PredictedAtomicFinding],
        f_app: Sequence[PredictedAtomicFinding],
        adjudications: EvaluationAdjudications,
        *,
        scientific_question: str,
        case_context: Mapping[str, Any],
        finding_goal: str,
        trace: _EvaluationTrace,
        semantic_matcher: SemanticMatcher,
        requirement_contract: FindingRequirementContract | None = None,
        materialized_g_of_o: Mapping[str, Any] | None = None,
        candidate_operationalization: Mapping[str, Any] | None = None,
        execution_evidence_provenance: Mapping[str, Any] | None = None,
    ) -> tuple[FindingBranchEvaluation, FindingBranchDiagnostic]:
        references = tuple(branch.findings)
        core_count = sum(
            reference.importance == FindingImportance.CORE for reference in references
        )
        if core_count == 0:
            raise EvaluationConfigurationError(
                f"branch {branch.operationalization_id!r} has no core ReferenceFinding"
            )
        # F_app is selected once for the response.  Branch evaluation may
        # classify these predictions differently, but may not change the
        # denominator or admit out-of-scope predictions.
        f_app_ids = {finding.prediction_id for finding in f_app}
        applicable_predictions = tuple(
            prediction
            for prediction in predictions
            if prediction.prediction_id in f_app_ids
        )
        semantic_candidate_ids: set[str] = set()
        verified_edges: set[tuple[str, str]] = set()
        verified_prediction_ids: set[str] = set()
        matched_pairs = []
        first_external_pending = None
        for prediction in applicable_predictions:
            for reference in references:
                explicit_policy = self.finding_verification_policy_index.get(
                    (branch.operationalization_id, reference.finding_id)
                )
                effective_policy: Mapping[str, Any] | None = explicit_policy
                unit_resolution = adjudications.unit_frame_resolution_for(
                    prediction, reference, explicit_policy
                )
                if unit_resolution is not None:
                    effective_policy = dict(explicit_policy or {})
                    effective_policy["unit_frame_resolution"] = unit_resolution.to_dict()
                verification_mode = finding_verification_mode(reference, effective_policy)
                # A qualitative claim cannot satisfy a numeric requirement.
                # Do not spend a semantic call on this impossible edge, then
                # mistake the missing value for a false scientific statement.
                # Keep it in F_app and send it through the existing blinded
                # finding adjudication, with no automatic correctness credit.
                if prediction.value is None and verification_mode in {
                    FindingVerificationMode.SCALAR_TOLERANCE,
                    FindingVerificationMode.SPATIAL_EUCLIDEAN,
                    FindingVerificationMode.COMPONENTWISE_VECTOR,
                    FindingVerificationMode.EXACT_DISCRETE_NUMERIC,
                }:
                    continue
                request = SemanticMatchRequest(
                    purpose=SemanticMatchPurpose.FINDING,
                    predicted_statement=prediction.statement,
                    reference_statement=reference.statement,
                    predicted_id=prediction.prediction_id,
                    reference_id=reference.finding_id,
                    branch_id=branch.operationalization_id,
                    scientific_question=scientific_question,
                    case_context=case_context,
                    finding_goal=finding_goal,
                    finding_category=reference.category.value,
                    predicted_value=prediction.value,
                    predicted_unit=prediction.unit,
                    reference_value=reference.value,
                    reference_unit=reference.unit,
                    verification_mode=verification_mode,
                )
                try:
                    result = _match(semantic_matcher, request, trace)
                except EvaluationPendingAdjudication as exc:
                    # File review can discover independent pairs together.
                    # Missing judgments are never interpreted as NO_MATCH,
                    # and no assignments or scores may use this partial set.
                    if exc.pending_type != "external_semantic_match":
                        raise
                    if first_external_pending is None:
                        first_external_pending = exc
                    continue
                if result != SemanticMatchResult.MATCH:
                    continue
                matched_pairs.append((prediction, reference, effective_policy))
        if first_external_pending is not None:
            raise first_external_pending
        for prediction, reference, effective_policy in matched_pairs:
            semantic_candidate_ids.add(prediction.prediction_id)
            cached_verification = next(
                (
                    item
                    for item in trace.value_verifications
                    if item.branch_id == branch.operationalization_id
                    and item.prediction_id == prediction.prediction_id
                    and item.reference_id == reference.finding_id
                ),
                None,
            )
            if cached_verification is None:
                verified, rule = _verify_finding_value(
                    prediction,
                    reference,
                    unit_converter=self.unit_converter,
                    explicit_policy=effective_policy,
                )
                cached_verification = ValueVerificationJudgment(
                    branch_id=branch.operationalization_id,
                    prediction_id=prediction.prediction_id,
                    reference_id=reference.finding_id,
                    rule=rule,
                    verified=verified,
                )
                _append_value_verification(trace, cached_verification)
            verified = cached_verification.verified
            if verified:
                verified_edges.add(
                    (prediction.prediction_id, reference.finding_id)
                )
                verified_prediction_ids.add(prediction.prediction_id)
        assignment = _maximum_weight_assignment(
            [prediction.prediction_id for prediction in applicable_predictions],
            references,
            verified_edges,
        )
        assigned_prediction_ids = {prediction_id for prediction_id, _ in assignment}
        assigned_reference_ids = {finding_id for _, finding_id in assignment}
        accepted_novel: set[str] = set()
        for prediction in applicable_predictions:
            prediction_id = prediction.prediction_id
            if prediction_id in semantic_candidate_ids or prediction_id not in f_app_ids:
                continue
            status = adjudications.novel_findings.get(
                (branch.operationalization_id, prediction_id)
            )
            if status is None or status == AdjudicationStatus.UNRESOLVED:
                raise EvaluationPendingAdjudication(
                    f"GT-outside Finding {prediction_id!r} requires adjudication for "
                    f"branch {branch.operationalization_id!r}",
                    pending_type="gt_outside_finding",
                    continuation_context={
                        "route_owner": "BLINDED_FINDING_ADJUDICATOR",
                        "required_next_action": "BLINDED_FINDING_ADJUDICATION",
                        # This is an opaque case-local join key.  It identifies
                        # the Effective-O branch to the continuation owner but
                        # does not expose reference membership or GT content.
                        "branch_id": branch.operationalization_id,
                        "scientific_question": scientific_question,
                        "case_context": dict(case_context),
                        "finding_goal": finding_goal,
                        "candidate_operationalization": candidate_operationalization,
                        "execution_evidence_provenance": execution_evidence_provenance,
                        # Actual execution evidence is scientific input, not
                        # reference membership.  A blinded Finding judge needs
                        # it to decide data support; GT findings remain hidden.
                        "materialized_G_of_O": (
                            None
                            if materialized_g_of_o is None
                            else dict(materialized_g_of_o)
                        ),
                        "allowed_finding_roles": (
                            []
                            if requirement_contract is None
                            else sorted(requirement_contract.allowed_roles)
                        ),
                        "continuation_request_id": continuation_request_id(
                            "finding_adjudication",
                            {
                                "branch_id": branch.operationalization_id,
                                "prediction_id": prediction_id,
                                "finding": {
                                    "statement": prediction.statement,
                                    "value": prediction.value,
                                    "unit": prediction.unit,
                                },
                            },
                        ),
                        "continuation_request": {
                            "branch_id": branch.operationalization_id,
                            "prediction_id": prediction_id,
                            "finding": {
                                "statement": prediction.statement,
                                "value": prediction.value,
                                "unit": prediction.unit,
                            },
                        },
                        # Host-only discovery: these judgments depend on the
                        # same finalized matching/eligibility state and O.
                        # No verdict or partial score is inferred for them.
                        "independent_finding_requests": [
                            {"branch_id": branch.operationalization_id,
                             "prediction_id": item.prediction_id,
                             "finding": {"statement": item.statement,
                                         "value": item.value, "unit": item.unit}}
                            for item in applicable_predictions
                            if item.prediction_id not in semantic_candidate_ids
                            and adjudications.novel_findings.get(
                                (branch.operationalization_id, item.prediction_id))
                            in (None, AdjudicationStatus.UNRESOLVED)
                        ],
                    },
                )
            trace.novel_finding_adjudications.append(
                (branch.operationalization_id, prediction_id, status)
            )
            if status == AdjudicationStatus.ACCEPTED:
                accepted_novel.add(prediction_id)
        if assigned_prediction_ids & accepted_novel:
            raise EvaluationConfigurationError(
                "matched GT and accepted GT-outside Finding credits must be disjoint"
            )
        reference_by_id = {reference.finding_id: reference for reference in references}
        matched_core = sum(
            reference_by_id[finding_id].importance == FindingImportance.CORE
            for finding_id in assigned_reference_ids
        )
        consistent_applicable = tuple(
            prediction.prediction_id
            for prediction in f_app
            if prediction.prediction_id in verified_prediction_ids
            or prediction.prediction_id in accepted_novel
        )
        valid_count = len(assigned_prediction_ids) + len(accepted_novel)
        adequate_core_evaluation = None
        scoring_mode = "LEGACY_FIXED_CORE"
        mandatory_role_count = 0
        matched_mandatory_role_count = 0
        adequate_set_evaluations: tuple[Mapping[str, Any], ...] = ()
        best_adequate_set_ids: tuple[str, ...] = ()
        best_adequate_set_recall: float | None = None
        novel_role_matches: list[Mapping[str, Any]] = []
        role_match_provenance: dict[str, str] = {}
        if requirement_contract is not None and requirement_contract.adequate_core_sets:
            scoring_mode = "SEMANTIC_ADEQUATE_CORE"
            role_by_category = dict(requirement_contract.role_by_category or {})
            reference_role_map = dict(
                requirement_contract.reference_finding_role_map or {}
            )
            matched_roles = {
                reference_role_map.get(
                    finding_id,
                    role_by_category.get(
                        reference_by_id[finding_id].category.value,
                        reference_by_id[finding_id].category.value,
                    ),
                )
                for finding_id in assigned_reference_ids
            }
            for role in matched_roles:
                role_match_provenance[role] = "ENUMERATED"
            allowed_roles = requirement_contract.allowed_roles
            for prediction_id in accepted_novel:
                role = adjudications.novel_finding_roles.get((branch.operationalization_id, prediction_id))
                if role is None:
                    continue
                if (
                    role == "selected_field_identity"
                    and materialized_g_of_o is not None
                    and materialized_g_of_o.get("field_selection_policy", {}).get("mode") == "explicit_candidate_field_v1"
                    and materialized_g_of_o.get("selection_resolution", {}).get("status") != "EXPLICIT_VALID_CANDIDATE"
                ):
                    # An adjudicated true description of a candidate cannot
                    # repair a missing/invalid declaration of which was chosen.
                    continue
                if role not in allowed_roles:
                    raise EvaluationConfigurationError(
                        f"adjudicated novel Finding role {role!r} is not eligible for the F2 contract"
                    )
                matched_roles.add(role)
                novel_role_matches.append({
                    "prediction_id": prediction_id,
                    "role": role,
                    "role_match_provenance": "ADJUDICATED_VALID_UNENUMERATED",
                })
                role_match_provenance[role] = "ADJUDICATED_VALID_UNENUMERATED"
            adequate_core_evaluation = evaluate_adequate_core_sets(matched_roles, requirement_contract)
            mandatory_role_count = len(requirement_contract.mandatory_roles)
            matched_mandatory_role_count = len(set(requirement_contract.mandatory_roles) & matched_roles)
            adequate_set_evaluations = tuple(adequate_core_evaluation.get("sets", ()))
            best_adequate_set_ids = tuple(adequate_core_evaluation.get("best_set_ids", ()))
            best_adequate_set_recall = adequate_core_evaluation.get("best_recall")
        metric_input = FindingBranchEvaluation(
            branch.operationalization_id,
            len(f_app),
            valid_count,
            core_count,
            matched_core,
            scoring_mode=scoring_mode,
            mandatory_role_count=mandatory_role_count,
            matched_mandatory_role_count=matched_mandatory_role_count,
            adequate_set_evaluations=adequate_set_evaluations,
            best_adequate_set_ids=best_adequate_set_ids,
            best_adequate_set_recall=best_adequate_set_recall,
            novel_role_matches=tuple(novel_role_matches),
            role_match_provenance=role_match_provenance,
        )
        diagnostic = FindingBranchDiagnostic(
            branch_id=branch.operationalization_id,
            matched_pairs=assignment,
            accepted_gt_outside_prediction_ids=tuple(
                prediction.prediction_id
                for prediction in applicable_predictions
                if prediction.prediction_id in accepted_novel
            ),
            consistent_applicable_prediction_ids=consistent_applicable,
            predicted_finding_count=len(f_app),
            valid_predicted_finding_count=valid_count,
            core_finding_count=core_count,
            matched_core_finding_count=matched_core,
            adequate_core_evaluation=adequate_core_evaluation,
        )
        return metric_input, diagnostic

    @staticmethod
    def _validate_novel_operationalization(
        operationalization: OperationalizationBundle,
        principal: Sequence[OperationalizationDimension],
        existing: Sequence[OperationalizationBundle],
    ) -> None:
        branch_id = operationalization.operationalization_id
        if branch_id in {bundle.operationalization_id for bundle in existing}:
            raise EvaluationConfigurationError(
                "novel O branch_id must not collide with frozen GT"
            )
        dimensions = {
            decision.dimension for decision in operationalization.decisions
        }
        if dimensions != set(principal):
            raise EvaluationConfigurationError(
                "accepted novel O must cover every principal dimension exactly once"
            )

    @staticmethod
    def _validate_novel_branch(
        novel: AdjudicatedNovelBranch,
        principal: Sequence[OperationalizationDimension],
        existing: Sequence[OperationalizationBundle],
        *,
        require_materialization: bool = False,
        require_core_findings: bool = True,
    ) -> None:
        CaseEvaluator._validate_novel_operationalization(
            novel.operationalization, principal, existing
        )
        if require_core_findings and not any(
            finding.importance == FindingImportance.CORE
            for finding in novel.findings.findings
        ):
            raise EvaluationConfigurationError(
                "accepted novel G(O) must contain a core Finding"
            )
        if require_materialization:
            provenance = novel.execution_provenance
            if not novel.materialization_id:
                raise EvaluationConfigurationError(
                    "accepted novel O requires exact materialization_id"
                )
            if not isinstance(provenance, Mapping):
                raise EvaluationConfigurationError(
                    "accepted novel O requires execution provenance"
                )
            if str(provenance.get("status", "")).upper() != "MATERIALIZED":
                raise EvaluationConfigurationError(
                    "accepted novel O execution provenance must be MATERIALIZED"
                )
            if not isinstance(provenance.get("G_of_O"), Mapping):
                raise EvaluationConfigurationError(
                    "accepted novel O requires structured G(O) execution output"
                )

    def _materialize_novel_branch(
        self,
        novel: AdjudicatedNovelBranch,
        *,
        expected_dataset_id: str,
    ) -> AdjudicatedNovelBranch:
        """Materialize an accepted novel O through the trusted execution hook.

        ``novel`` is an adjudication payload and its Findings/provenance are
        untrusted.  The callback receives only the OperationalizationBundle;
        its returned artifact is the sole source used to build the temporary
        G(O) branch.  Without a callback, a coherent novel O remains pending.
        """

        return self._materialize_novel_operationalization(
            novel.operationalization,
            expected_dataset_id=expected_dataset_id,
        )

    def _materialize_novel_operationalization(
        self,
        operationalization: OperationalizationBundle,
        *,
        expected_dataset_id: str,
    ) -> AdjudicatedNovelBranch:
        """Execute an O-only candidate and construct G(O) from trusted output.

        The separation prevents a completion/adjudication model from having
        to invent a dummy ReferenceFinding before the data route is run.
        """

        callback = self.novel_o_materializer
        if callback is None:
            raise EvaluationPendingAdjudication(
                "accepted novel O requires a trusted materializer callback",
                pending_type="novel_operationalization_materialization",
                continuation_context={
                    "route_owner": "NOVEL_O_MATERIALIZATION",
                    "required_next_action": "MATERIALIZE_NOVEL_OPERATIONALIZATION",
                    "continuation_request_id": continuation_request_id(
                        "operationalization_materialization",
                        {
                            "operationalization": operationalization.model_dump(mode="json"),
                        },
                    ),
                    "continuation_request": {
                        "operationalization": operationalization.model_dump(mode="json"),
                    },
                    "candidate_operationalization": operationalization.model_dump(mode="json"),
                },
            )
        try:
            raw = callback(operationalization)
        except EvaluationPendingAdjudication:
            raise
        except Exception as exc:
            raise EvaluationPendingAdjudication(
                "trusted novel O materialization did not complete: "
                f"{type(exc).__name__}: {exc}",
                pending_type="novel_operationalization_materialization",
                continuation_context={
                    "route_owner": "NOVEL_O_MATERIALIZATION",
                    "required_next_action": "MATERIALIZE_NOVEL_OPERATIONALIZATION",
                    "continuation_request_id": continuation_request_id(
                        "operationalization_materialization",
                        {
                            "operationalization": operationalization.model_dump(mode="json"),
                        },
                    ),
                    "continuation_request": {
                        "operationalization": operationalization.model_dump(mode="json"),
                    },
                    "candidate_operationalization": operationalization.model_dump(mode="json"),
                },
            ) from exc
        if isinstance(raw, Mapping):
            artifact = dict(raw)
        else:
            to_dict = getattr(raw, "to_dict", None)
            if not callable(to_dict):
                raise EvaluationConfigurationError(
                    "trusted novel O materializer must return a mapping artifact"
                )
            converted = to_dict()
            if not isinstance(converted, Mapping):
                raise EvaluationConfigurationError(
                    "trusted novel O materializer returned a non-object artifact"
                )
            artifact = dict(converted)

        expected_parameters = {
            decision.dimension.value: decision.statement
            for decision in operationalization.decisions
        }
        parameters = artifact.get("parameters")
        if not isinstance(parameters, Mapping):
            raise EvaluationConfigurationError(
                "trusted novel O materialization must include parameters"
            )
        if _json_value(dict(parameters), "materialization.parameters") != _json_value(
            expected_parameters, "effective_operationalization"
        ):
            raise EvaluationConfigurationError(
                "trusted novel O materialization parameters do not match Effective O"
            )

        execution = artifact.get("execution")
        if not isinstance(execution, Mapping):
            raise EvaluationConfigurationError(
                "trusted novel O materialization must include execution"
            )
        status = str(execution.get("status", "")).strip().upper()
        if status != "MATERIALIZED":
            # Unsupported/failed routes are not scientific INVALID results.
            raise EvaluationPendingAdjudication(
                "accepted novel O has no materialized G(O): "
                f"{status or 'UNKNOWN'}",
                pending_type="novel_operationalization_materialization",
                continuation_context={
                    "route_owner": "NOVEL_O_MATERIALIZATION",
                    "required_next_action": "MATERIALIZE_NOVEL_OPERATIONALIZATION",
                    "continuation_request_id": continuation_request_id(
                        "operationalization_materialization",
                        {
                            "operationalization": operationalization.model_dump(mode="json"),
                        },
                    ),
                    "continuation_request": {
                        "operationalization": operationalization.model_dump(mode="json"),
                    },
                    "candidate_operationalization": operationalization.model_dump(mode="json"),
                    "execution_status": status or "UNKNOWN",
                },
            )
        if execution.get("reproducible") is not True:
            raise EvaluationConfigurationError(
                "trusted novel O execution must declare reproducible=true"
            )
        g_of_o = execution.get("G_of_O")
        if not isinstance(g_of_o, Mapping) or not g_of_o:
            raise EvaluationConfigurationError(
                "trusted novel O execution must contain non-empty structured G(O)"
            )

        provenance = execution.get("data_provenance")
        if not isinstance(provenance, Mapping):
            raise EvaluationConfigurationError(
                "trusted novel O execution requires data provenance"
            )
        dataset_id = provenance.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise EvaluationConfigurationError(
                "trusted novel O data provenance requires dataset_id"
            )
        if dataset_id != expected_dataset_id:
            raise EvaluationConfigurationError(
                "trusted novel O data provenance belongs to a different dataset"
            )
        manifest_digest = str(provenance.get("dataset_manifest_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
            raise EvaluationConfigurationError(
                "trusted novel O data provenance requires dataset_manifest_sha256"
            )
        expected_o_digest = _canonical_json_sha256(expected_parameters)
        if provenance.get("effective_o_sha256") != expected_o_digest:
            raise EvaluationConfigurationError(
                "trusted novel O execution Effective O hash does not match"
            )
        plan = provenance.get("materialization_plan")
        if not isinstance(plan, Mapping):
            raise EvaluationConfigurationError(
                "trusted novel O execution requires materialization_plan"
            )
        plan_digest = _canonical_json_sha256(dict(plan))
        for key in ("compiled_plan_sha256", "executed_plan_sha256"):
            if provenance.get(key) != plan_digest:
                raise EvaluationConfigurationError(
                    f"trusted novel O execution {key} does not match materialization_plan"
                )
        for key in ("silent_substitutions", "unsupported_dimensions"):
            if provenance.get(key) not in ([], None):
                raise EvaluationConfigurationError(
                    f"trusted novel O execution contains {key}"
                )

        materialization_id = str(artifact.get("materialization_id", "")).strip()
        if not materialization_id:
            raise EvaluationConfigurationError(
                "trusted novel O materialization requires materialization_id"
            )
        raw_findings = artifact.get("findings")
        if not isinstance(raw_findings, Sequence) or isinstance(
            raw_findings, (str, bytes, bytearray)
        ):
            raise EvaluationConfigurationError(
                "trusted novel O materialization requires findings"
            )
        try:
            findings = tuple(
                item
                if isinstance(item, ReferenceFinding)
                else ReferenceFinding.model_validate(item)
                for item in raw_findings
            )
            finding_branch = OperationalizationFindingBranch(
                operationalization_id=operationalization.operationalization_id,
                findings=list(findings),
            )
        except Exception as exc:
            raise EvaluationConfigurationError(
                "trusted novel O materializer returned invalid findings"
            ) from exc
        return AdjudicatedNovelBranch(
            operationalization=operationalization,
            findings=finding_branch,
            materialization_id=materialization_id,
            execution_provenance=dict(execution),
        )

    @staticmethod
    def _validate_case(
        case_id: str,
        metadata: CaseConstructionMetadata,
        ground_truth: GroundTruth,
    ) -> tuple[str, GroundTruth]:
        if case_id != metadata.case_id:
            raise EvaluationConfigurationError(
                f"case_id mismatch: run={case_id!r}, metadata={metadata.case_id!r}"
            )
        validated = GroundTruthValidator.validate(ground_truth, metadata)
        condition = primary_case_type(metadata)
        if condition is None:
            raise EvaluationConfigurationError(
                "evaluator supports the frozen O1-F1/O2-F1/O3-F1/O1-F2 conditions"
            )
        return condition, validated

    @staticmethod
    def _noncompletion_result(
        case_id: str,
        condition: str,
        metadata: CaseConstructionMetadata,
        efficiency: EfficiencyObservation,
    ) -> CaseEvaluationResult:
        metrics = CaseMetricResult(
            case_id=case_id,
            condition=condition,
            scientific_operationalization=OperationalizationMetricResult(
                o_score=0.0,
                best_o_branches=(),
                urs=(
                    0.0
                    if metadata.unresolved_operationalization_dimensions
                    else None
                ),
            ),
            scientific_findings=FindingMetricResult(
                finding_precision=0.0,
                core_finding_recall=None if condition.endswith("-F2") else 0.0,
                best_finding_branches=(),
                finding_recall_mode=(
                    "SEMANTIC_ADEQUATE_CORE" if condition.endswith("-F2")
                    else "FIXED_REFERENCE_CORE"
                ),
                finding_requirement_recall=0.0,
                adequate_core_complete=False if condition.endswith("-F2") else None,
            ),
            o_f_consistency=ConsistencyMetricResult(
                c_score=None,
                branch_alignment=None,
                operationalization_determinate=False,
                unavailable_reason="MODEL_NONCOMPLETION",
            ),
            efficiency=efficiency,
        )
        return CaseEvaluationResult(
            case_id=case_id,
            condition=condition,
            run_status=RunStatus.MODEL_NONCOMPLETION,
            eligible_for_scientific_aggregation=True,
            metrics=metrics,
        )


def _coerce_trial_evaluation(value: TrialEvaluation | Mapping[str, Any]) -> TrialEvaluation:
    if isinstance(value, TrialEvaluation):
        return value
    if isinstance(value, Mapping):
        return TrialEvaluation.from_dict(value)
    raise EvaluationConfigurationError(
        "formal trial aggregation requires TrialEvaluation values"
    )


def _required_trial_mean(values: Sequence[float | None], field_name: str) -> float:
    if len(values) != FORMAL_TRIAL_COUNT or any(value is None for value in values):
        raise EvaluationConfigurationError(
            f"formal trial metric {field_name} is not fully defined for all three trials"
        )
    return statistics.fmean(value for value in values if value is not None)


def _optional_trial_mean(
    values: Sequence[float | None], field_name: str
) -> float | None:
    if len(values) != FORMAL_TRIAL_COUNT:
        raise EvaluationConfigurationError(
            f"formal trial metric {field_name} must have exactly three values"
        )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise EvaluationConfigurationError(
            f"formal trial metric {field_name} is inconsistently applicable across trials"
        )
    return statistics.fmean(value for value in values if value is not None)


def _aggregate_efficiency(trials: Sequence[TrialEvaluation]) -> EfficiencyAggregate:
    values: dict[str, float | None] = {}
    for field_name in EFFICIENCY_METRIC_NAMES:
        raw_values = [
            getattr(trial.result.metrics.efficiency, field_name)  # type: ignore[union-attr]
            for trial in trials
        ]
        values[field_name] = (
            None
            if any(value is None for value in raw_values)
            else statistics.fmean(value for value in raw_values if value is not None)
        )
    return EfficiencyAggregate(**values)


def aggregate_trials_for_case(
    trials: Iterable[TrialEvaluation | Mapping[str, Any]],
) -> CaseAggregate:
    """Finalize one case from the frozen three-trial formal protocol.

    ``INFRASTRUCTURE_INVALID`` attempts are retained in the aggregate audit
    record but do not occupy a trial slot.  A pending trial raises immediately;
    callers must resolve it before a formal case result can be emitted.
    """

    values = tuple(_coerce_trial_evaluation(value) for value in trials)
    if not values:
        raise EvaluationConfigurationError("at least one trial evaluation is required")
    pending = tuple(value for value in values if value.pending)
    if pending:
        reasons = tuple(value.pending_reason or "unresolved adjudication" for value in pending)
        raise EvaluationPendingAdjudication(
            "case aggregation is blocked by pending trial adjudication: "
            + "; ".join(reasons)
        )

    valid = tuple(
        value
        for value in values
        if value.result is not None and not value.is_infrastructure_invalid
    )
    if len(valid) != FORMAL_TRIAL_COUNT:
        raise EvaluationConfigurationError(
            "formal case finalization requires exactly three valid model trials; "
            f"received {len(valid)}"
        )
    trial_indices = tuple(value.trial_index for value in valid)
    if len(set(trial_indices)) != FORMAL_TRIAL_COUNT or set(trial_indices) != set(
        range(1, FORMAL_TRIAL_COUNT + 1)
    ):
        raise EvaluationConfigurationError(
            "valid formal trials must have unique trial_index values {1, 2, 3}"
        )

    results = tuple(value.result for value in valid)
    if any(result is None for result in results):  # pragma: no cover - guarded above
        raise EvaluationConfigurationError("valid trial is missing its result")
    concrete_results: tuple[CaseEvaluationResult, ...] = tuple(
        result for result in results if result is not None
    )
    case_ids = {result.case_id for result in concrete_results}
    conditions = {result.condition for result in concrete_results}
    if len(case_ids) != 1 or len(conditions) != 1:
        raise EvaluationConfigurationError(
            "all trials in a case aggregate must share case_id and condition"
        )
    if any(
        not result.eligible_for_scientific_aggregation or result.metrics is None
        for result in concrete_results
    ):
        raise EvaluationConfigurationError(
            "only fully scored eligible model observations may enter a case estimator"
        )

    case_id = next(iter(case_ids))
    condition = next(iter(conditions))
    identities = {
        value.experiment_id for value in valid if value.experiment_id is not None
    }
    target_fingerprints = {
        value.target_fingerprint
        for value in valid
        if value.target_fingerprint is not None
    }
    release_ids = {
        value.benchmark_release_id
        for value in valid
        if value.benchmark_release_id is not None
    }
    formal_modes = {value.formal_mode for value in valid if value.formal_mode is not None}
    evaluator_digests = {
        value.evaluation_manifest_digest
        for value in valid
        if value.evaluation_manifest_digest is not None
    }
    for field_name, identity_values in (
        ("experiment_id", [value.experiment_id for value in valid]),
        ("target_fingerprint", [value.target_fingerprint for value in valid]),
        ("benchmark_release_id", [value.benchmark_release_id for value in valid]),
        ("formal_mode", [value.formal_mode for value in valid]),
        (
            "evaluation_manifest_digest",
            [value.evaluation_manifest_digest for value in valid],
        ),
    ):
        if any(item is None for item in identity_values) and any(item is not None for item in identity_values):
            raise EvaluationConfigurationError(
                f"all trials in a case aggregate must either provide or omit {field_name} identity"
            )
    if (
        len(identities) > 1
        or len(target_fingerprints) > 1
        or len(release_ids) > 1
        or len(formal_modes) > 1
        or len(evaluator_digests) > 1
    ):
        raise EvaluationConfigurationError(
            "all trials in a case aggregate must share experiment, target, release and formal-mode identity"
        )

    metrics = tuple(result.metrics for result in concrete_results)
    o_score = _required_trial_mean(
        [metric.scientific_operationalization.o_score for metric in metrics],
        "o_score",
    )
    urs = _optional_trial_mean(
        [metric.scientific_operationalization.urs for metric in metrics],
        "urs",
    )
    resolved_values = [metric.scientific_operationalization.resolved_o_compliance for metric in metrics]
    # This diagnostic is not applicable to non-completion trials.  Preserve a
    # null case-level value rather than treating the missing observation as a
    # zero or rejecting an otherwise valid three-trial aggregate.
    resolved_o_compliance = (
        None if any(value is None for value in resolved_values)
        else _optional_trial_mean(resolved_values, "resolved_o_compliance")
    )
    finding_precision = _required_trial_mean(
        [metric.scientific_findings.finding_precision for metric in metrics],
        "finding_precision",
    )
    finding_recall_modes = {
        metric.scientific_findings.finding_recall_mode for metric in metrics
    }
    if len(finding_recall_modes) != 1:
        raise EvaluationConfigurationError(
            "all trials in a case aggregate must share finding_recall_mode"
        )
    finding_recall_mode = next(iter(finding_recall_modes))
    core_values = [metric.scientific_findings.core_finding_recall for metric in metrics]
    core_finding_recall = (
        _required_trial_mean(core_values, "core_finding_recall")
        if finding_recall_mode == "FIXED_REFERENCE_CORE"
        else _optional_trial_mean(core_values, "core_finding_recall")
    )
    finding_requirement_recall = _required_trial_mean(
        [metric.scientific_findings.finding_requirement_recall for metric in metrics],
        "finding_requirement_recall",
    )
    adequate_values = [
        metric.scientific_findings.adequate_core_complete
        for metric in metrics
    ]
    adequate_core_complete = (
        all(value is True for value in adequate_values)
        if finding_recall_mode == "SEMANTIC_ADEQUATE_CORE"
        else None
    )
    adequate_core_complete_rate = (
        statistics.fmean(float(value is True) for value in adequate_values)
        if finding_recall_mode == "SEMANTIC_ADEQUATE_CORE"
        else None
    )
    # C and alignment are conditional estimates. Keep every model trial in
    # the case and expose the selected denominator instead of zero filling or
    # rejecting a case whose responses differ in O completeness.
    consistency_values = [metric.o_f_consistency for metric in metrics]
    metric_trial_denominators = {}
    conditional_means = {}
    for name in ("c_score", "branch_alignment"):
        finalized = [getattr(value, name) for value in consistency_values
                     if getattr(value, name) is not None]
        applicable_n = sum(
            value.operationalization_determinate is not False
            and value.unavailable_reason != "NO_APPLICABLE_FINDINGS"
            for value in consistency_values
        )
        conditional_means[name] = statistics.fmean(finalized) if finalized else None
        metric_trial_denominators[name] = {
            "eligible_trial_n": len(valid),
            "applicable_trial_n": applicable_n,
            "finalized_trial_n": len(finalized),
            "not_applicable_trial_n": len(valid) - applicable_n,
            "unavailable_trial_n": applicable_n - len(finalized),
            "coverage": len(finalized) / len(valid),
            "applicability_coverage": applicable_n / len(valid),
        }
    determinate_flags = [value.operationalization_determinate for value in consistency_values]
    determinate_o_trial_count = (
        None if any(value is None for value in determinate_flags)
        else sum(value is True for value in determinate_flags)
    )
    c_score = conditional_means["c_score"]
    branch_alignment = conditional_means["branch_alignment"]
    case_metrics = CaseAggregateMetricResult(
        case_id=case_id,
        condition=condition,
        o_score=o_score,
        urs=urs,
        resolved_o_compliance=resolved_o_compliance,
        finding_precision=finding_precision,
        core_finding_recall=core_finding_recall,
        finding_requirement_recall=finding_requirement_recall,
        finding_recall_mode=finding_recall_mode,
        adequate_core_complete=adequate_core_complete,
        adequate_core_complete_rate=adequate_core_complete_rate,
        metric_trial_denominators=metric_trial_denominators,
        determinate_o_trial_count=determinate_o_trial_count,
        determinate_o_trial_rate=(
            None if determinate_o_trial_count is None
            else determinate_o_trial_count / len(valid)
        ),
        c_score=c_score,
        branch_alignment=branch_alignment,
        efficiency=_aggregate_efficiency(valid),
    )
    return CaseAggregate(
        case_id=case_id,
        condition=condition,
        trial_evaluations=values,
        metrics=case_metrics,
        experiment_id=next(iter(identities), None),
        target_fingerprint=next(iter(target_fingerprints), None),
        benchmark_release_id=next(iter(release_ids), None),
        formal_mode=next(iter(formal_modes), None),
        evaluation_manifest_digest=next(iter(evaluator_digests), None),
    )


def aggregate_cases(
    aggregates: Iterable[CaseAggregate],
    *,
    report_conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Macro-average finalized case estimators, never raw trial rows."""

    values = tuple(aggregates)
    if any(not isinstance(value, CaseAggregate) for value in values):
        raise EvaluationConfigurationError(
            "benchmark aggregation requires finalized CaseAggregate values"
        )
    experiment_ids = {value.experiment_id for value in values}
    target_fingerprints = {value.target_fingerprint for value in values}
    release_ids = {value.benchmark_release_id for value in values}
    formal_modes = {value.formal_mode for value in values}
    evaluator_digests = {
        value.evaluation_manifest_digest for value in values
    }
    if (
        not values
        or None in experiment_ids
        or None in target_fingerprints
        or None in release_ids
        or None in evaluator_digests
        or formal_modes != {True}
        or len(experiment_ids) != 1
        or len(target_fingerprints) != 1
        or len(release_ids) != 1
        or len(evaluator_digests) != 1
    ):
        raise EvaluationConfigurationError(
            "formal benchmark aggregation requires one non-null experiment_id "
            ", target_fingerprint, benchmark_release_id and formal_mode=True "
            "shared by every CaseAggregate"
        )
    case_ids = [value.case_id for value in values]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationConfigurationError(
            "each case may contribute exactly one finalized CaseAggregate"
        )

    def collect(selected: Sequence[CaseAggregate]) -> BenchmarkMetricValues:
        efficiency: dict[str, tuple[Real, ...]] = {
            field_name: tuple(
                value
                for aggregate in selected
                for value in [
                    getattr(aggregate.metrics.efficiency, field_name)
                ]
                if value is not None
            )
            for field_name in EFFICIENCY_METRIC_NAMES
        }
        # Applicability is derived from the frozen case condition, while
        # finalized counts come only from the already-finalized CaseAggregate
        # values below.  No scientific judgment is recomputed here.
        urs_applicable_n = sum(
            aggregate.condition.startswith(("O2", "O3"))
            for aggregate in selected
        )
        roc_applicable_n = sum(
            not aggregate.condition.startswith("O3")
            for aggregate in selected
        )
        adequate_applicable_n = sum(
            aggregate.metrics.finding_recall_mode == "SEMANTIC_ADEQUATE_CORE"
            for aggregate in selected
        )
        return BenchmarkMetricValues(
            o_scores=tuple(
                aggregate.metrics.o_score
                for aggregate in selected
            ),
            resolved_o_compliance_scores=tuple(
                aggregate.metrics.resolved_o_compliance
                for aggregate in selected
                if aggregate.metrics.resolved_o_compliance is not None
            ),
            urs_scores=tuple(
                aggregate.metrics.urs
                for aggregate in selected
                if aggregate.metrics.urs is not None
            ),
            finding_precision_scores=tuple(
                aggregate.metrics.finding_precision
                for aggregate in selected
            ),
            core_finding_recall_scores=tuple(
                aggregate.metrics.core_finding_recall
                for aggregate in selected
                if aggregate.metrics.finding_recall_mode == "FIXED_REFERENCE_CORE"
                and aggregate.metrics.core_finding_recall is not None
            ),
            finding_requirement_recall_scores=tuple(
                aggregate.metrics.finding_requirement_recall
                for aggregate in selected
                if aggregate.metrics.finding_requirement_recall is not None
            ),
            c_scores=tuple(
                aggregate.metrics.c_score
                for aggregate in selected
                if aggregate.metrics.c_score is not None
            ),
            branch_alignment_scores=tuple(
                aggregate.metrics.branch_alignment
                for aggregate in selected
                if aggregate.metrics.branch_alignment is not None
            ),
            adequate_core_complete_scores=tuple(
                float(
                    aggregate.metrics.adequate_core_complete_rate
                    if aggregate.metrics.adequate_core_complete_rate is not None
                    else aggregate.metrics.adequate_core_complete is True
                )
                for aggregate in selected
                if aggregate.metrics.finding_recall_mode == "SEMANTIC_ADEQUATE_CORE"
                and (
                    aggregate.metrics.adequate_core_complete_rate is not None
                    or aggregate.metrics.adequate_core_complete is not None
                )
            ),
            efficiency_values=efficiency,
            metric_applicable_counts={
                "o_score": len(selected),
                "urs": urs_applicable_n,
                "resolved_o_compliance": roc_applicable_n,
                "finding_precision": len(selected),
                "core_finding_recall": sum(
                    aggregate.metrics.finding_recall_mode == "FIXED_REFERENCE_CORE"
                    for aggregate in selected
                ),
                "finding_requirement_recall": len(selected),
                "adequate_core_complete": adequate_applicable_n,
                # A case can supply a conditional estimate once at least one
                # trial has explicit O. Keep legacy unknown coverage in scope.
                "c_score": sum(
                    aggregate.metrics.metric_trial_denominators.get("c_score", {}).get("applicable_trial_n", 1) > 0
                    for aggregate in selected
                ),
                "branch_alignment": sum(
                    aggregate.metrics.metric_trial_denominators.get("branch_alignment", {}).get("applicable_trial_n", 1) > 0
                    for aggregate in selected
                ),
            },
        )

    existing_conditions = {aggregate.condition for aggregate in values}
    requested_conditions = set(report_conditions or existing_conditions)
    # Formal reporting never manufactures an N=0 condition.  An explicit
    # condition filter may select a subset, but only conditions represented by
    # finalized cases are emitted.
    conditions = sorted(existing_conditions & requested_conditions)
    grouped = {
        condition: tuple(
            aggregate for aggregate in values if aggregate.condition == condition
        )
        for condition in conditions
    }
    grouped_modes = {
        mode: tuple(
            aggregate for aggregate in values
            if aggregate.metrics.finding_recall_mode == mode
        )
        for mode in sorted({
            aggregate.metrics.finding_recall_mode for aggregate in values
        })
    }
    summary = benchmark_summary(
        collect(values),
        by_condition={
            condition: collect(condition_values)
            for condition, condition_values in grouped.items()
        },
        by_finding_recall_mode={
            mode: collect(mode_values)
            for mode, mode_values in grouped_modes.items()
        },
    )
    for block, selected in (
        (summary["overall"], values),
        *((summary["by_condition"][key], group) for key, group in grouped.items()),
        *((summary["by_finding_recall_mode"][key], group) for key, group in grouped_modes.items()),
    ):
        # Trial coverage accompanies case-macro means; do not reinterpret a
        # mean over two determinate trials as three observed consistency scores.
        block["metric_trial_denominators"] = {}
        for name in ("c_score", "branch_alignment"):
            records = [item.metrics.metric_trial_denominators.get(name) for item in selected]
            if any(record is None for record in records):
                block["metric_trial_denominators"][name] = None
                continue
            counts = {
                key: sum(record[key] for record in records)
                for key in ("eligible_trial_n", "applicable_trial_n", "finalized_trial_n",
                            "not_applicable_trial_n", "unavailable_trial_n")
            }
            total = counts["eligible_trial_n"]
            counts["coverage"] = counts["finalized_trial_n"] / total if total else None
            counts["applicability_coverage"] = counts["applicable_trial_n"] / total if total else None
            block["metric_trial_denominators"][name] = counts
    summary["overall"].update(
        {
            "eligible_case_count": len(values),
            "infrastructure_invalid_trial_count": sum(
                aggregate.infrastructure_invalid_trial_count for aggregate in values
            ),
            "model_noncompletion_trial_count": sum(
                aggregate.model_noncompletion_trial_count for aggregate in values
            ),
            "benchmark_release_id": next(iter(release_ids)),
            "experiment_id": next(iter(experiment_ids)),
            "target_fingerprint": next(iter(target_fingerprints)),
            "evaluation_manifest_digest": next(iter(evaluator_digests)),
        }
    )
    for condition, condition_values in grouped.items():
        summary["by_condition"][condition].update(
            {
                "eligible_case_count": len(condition_values),
                "infrastructure_invalid_trial_count": sum(
                    aggregate.infrastructure_invalid_trial_count
                    for aggregate in condition_values
                ),
                "model_noncompletion_trial_count": sum(
                    aggregate.model_noncompletion_trial_count
                    for aggregate in condition_values
                ),
            }
        )
    return summary


def aggregate_formal_trial_groups(
    trial_groups: Iterable[Iterable[TrialEvaluation | Mapping[str, Any]]],
    *,
    report_conditions: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Finalize each explicit trial group, then aggregate those case values."""

    aggregates = tuple(aggregate_trials_for_case(group) for group in trial_groups)
    return aggregate_cases(aggregates, report_conditions=report_conditions)


def _load_portfolio_manifest(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    """Load a frozen portfolio/family manifest without accepting hidden state."""

    if isinstance(value, Mapping):
        return value
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError(f"cannot load portfolio manifest: {path}") from exc
    if not isinstance(payload, Mapping):
        raise EvaluationConfigurationError("portfolio manifest must be a JSON object")
    return payload


def _is_historical_controlled_pilot_family(family: Mapping[str, Any]) -> bool:
    """Select the explicitly marked historical compatibility adapter.

    This is lifecycle routing, not scientific capability policy.  Formal
    release candidates never carry these markers and therefore cannot enter
    the legacy fixed-core path.
    """

    return bool(
        family.get("legacy_fixed_core") is True
        or (
            str(family.get("portfolio_role", "")).casefold() == "controlled_pilot"
            and family.get("formal_release_eligible") is False
        )
    )


def build_case_evaluator(
    case_id: str,
    portfolio_manifest: Mapping[str, Any] | str | Path,
    evaluator_backend: Any,
    evaluation_manifest: EvaluationManifest | Mapping[str, Any] | None = None,
    *,
    unit_converter: UnitConverter | None = None,
    scientific_evaluation_contract: ScientificEvaluationContract | Mapping[str, Any] | None = None,
    finding_verification_policy: Mapping[str, Any] | None = None,
    evaluation_mode: str = "DEVELOPMENT_EVALUATION",
    novel_o_materializer: Callable[[OperationalizationBundle], Any] | None = None,
    branch_execution_evidence: Mapping[str, Any] | None = None,
) -> CaseEvaluator:
    """Construct the evaluator for one formal case from its family manifest.

    Development evaluation preserves the historical family-manifest adapter.
    Formal evaluation takes its Finding contract only from the frozen SEC;
    family metadata is never used as a formal scientific authority.  Accepted
    novel O branches in either mode require the supplied trusted materializer;
    self-reported G(O) payloads are never consumed.
    """

    case_id = _text(case_id, "case_id")
    # ``evaluate_saved_runs`` receives a configured CaseEvaluator so that the
    # extractor/matcher/eligibility components remain one immutable bundle.
    # Accepting that bundle here avoids treating the CaseEvaluator itself as a
    # backend (it has no ``manifest()`` method) while preserving the public
    # backend-oriented API for existing callers.
    template_evaluator = evaluator_backend if isinstance(evaluator_backend, CaseEvaluator) else None
    backend = (
        template_evaluator.extractor
        if template_evaluator is not None
        else evaluator_backend
    )
    manifest = _load_portfolio_manifest(portfolio_manifest)
    families = manifest.get("families", manifest.get("concept_families", ()))
    if isinstance(families, Mapping):
        families = list(families.values())
    if not isinstance(families, Sequence) or isinstance(families, (str, bytes)):
        raise EvaluationConfigurationError("portfolio manifest must contain a families list")
    family = next(
        (
            item for item in families
            if isinstance(item, Mapping)
            and case_id in {
                *{str(value) for value in item.get("controlled_case_ids", ()) or ()},
                *{
                    str(record.get("case_id"))
                    for record in item.get("controlled_case_records", ()) or ()
                    if isinstance(record, Mapping) and record.get("case_id")
                },
            }
        ),
        None,
    )
    if family is None:
        raise EvaluationConfigurationError(f"case {case_id!r} is absent from the portfolio manifest")
    condition_match = re.search(r"(?:^|[_-])(O[123])[_-](F[12])(?:$|[_-])", case_id, re.I)
    condition = (
        f"{condition_match.group(1).upper()}-{condition_match.group(2).upper()}"
        if condition_match else str(family.get("condition", ""))
    )
    legacy = _is_historical_controlled_pilot_family(family)
    formal = str(evaluation_mode).strip().upper() == "FORMAL_EVALUATION"
    contract_value = family.get("finding_requirement_contract")
    if formal:
        if scientific_evaluation_contract is None:
            raise EvaluationConfigurationError("FORMAL_EVALUATION requires scientific_evaluation_contract")
        if isinstance(scientific_evaluation_contract, ScientificEvaluationContract):
            contract_value = scientific_evaluation_contract.finding_requirement_contract
        elif isinstance(scientific_evaluation_contract, Mapping):
            contract_value = scientific_evaluation_contract.get("finding_requirement_contract")
        else:
            raise EvaluationConfigurationError("scientific_evaluation_contract must be a mapping or ScientificEvaluationContract")
    if condition.endswith("-F2") and (formal or not legacy) and not contract_value:
        raise EvaluationConfigurationError("MISSING_FINDING_REQUIREMENT_CONTRACT")
    if evaluation_manifest is None:
        if template_evaluator is not None:
            evaluation_manifest = template_evaluator.evaluation_manifest
        elif not hasattr(evaluator_backend, "manifest"):
            raise EvaluationConfigurationError("evaluation_manifest is required when backend has no manifest()")
        else:
            evaluation_manifest = evaluator_backend.manifest()
    if template_evaluator is not None:
        return CaseEvaluator(
            template_evaluator.extractor,
            template_evaluator.semantic_matcher,
            template_evaluator.eligibility_judge,
            evaluation_manifest,
            unit_converter=template_evaluator.unit_converter,
            finding_requirement_contract=None if (legacy and not formal) else contract_value,
            scientific_evaluation_contract=scientific_evaluation_contract
            if scientific_evaluation_contract is not None
            else template_evaluator.scientific_evaluation_contract,
            finding_verification_policy=(
                finding_verification_policy
                if finding_verification_policy is not None
                else template_evaluator.finding_verification_policy
            ),
            evaluation_mode=evaluation_mode,
            novel_o_materializer=novel_o_materializer or template_evaluator.novel_o_materializer,
            branch_execution_evidence=(branch_execution_evidence if branch_execution_evidence is not None
                                       else template_evaluator.branch_execution_evidence or None),
        )
    return CaseEvaluator(
        backend,
        backend,
        backend,
        evaluation_manifest,
        unit_converter=unit_converter,
        finding_requirement_contract=None if (legacy and not formal) else contract_value,
        scientific_evaluation_contract=scientific_evaluation_contract,
        finding_verification_policy=finding_verification_policy,
        evaluation_mode=evaluation_mode,
        novel_o_materializer=novel_o_materializer,
        branch_execution_evidence=branch_execution_evidence,
    )


__all__ = [
    "AdjudicatedNovelBranch",
    "AdjudicationStatus",
    "CaseAggregate",
    "CaseEvaluationRecord",
    "CaseEvaluationResult",
    "CaseEvaluator",
    "build_case_evaluator",
    "ConsistencyBranchDiagnostic",
    "DeduplicationJudgment",
    "EvaluationAdjudications",
    "EvaluationConfigurationError",
    "EvaluationManifest",
    "EvaluationPendingAdjudication",
    "EvaluatorComponentIdentity",
    "ExtractedOperationalization",
    "ExtractedOperationalizationDecision",
    "ExtractedPrediction",
    "ExtractionRequest",
    "ExtractionStatus",
    "FindingVerificationMode",
    "FindingBranchDiagnostic",
    "FindingEligibility",
    "FindingEligibilityJudge",
    "FindingEligibilityRequest",
    "NovelOperationalizationAdjudication",
    "PredictedAtomicFinding",
    "PendingCaseEvaluationRecord",
    "PredictionExtractor",
    "RunEvaluationAdjudication",
    "SemanticMatchPurpose",
    "SemanticMatchRequest",
    "SemanticMatchResolution",
    "SemanticMatchResult",
    "SemanticMatchJudgment",
    "SemanticMatcher",
    "FORMAL_TRIAL_COUNT",
    "TrialEvaluation",
    "UnitConverter",
    "UnitFrameResolution",
    "continuation_request_id",
    "CONTINUATION_REQUEST_ID_VERSION",
    "ValueVerificationJudgment",
    "aggregate_cases",
    "aggregate_formal_trial_groups",
    "aggregate_trials_for_case",
    "evaluation_manifest_digest",
    "load_case_evaluation_record",
    "load_evaluation_adjudications",
    "load_evaluation_manifest",
    "load_run_evaluation_adjudication",
    "save_case_evaluation_record",
    "save_evaluation_adjudications",
    "save_evaluation_manifest",
    "save_run_evaluation_adjudication",
    "verify_finding_value",
    "finding_verification_mode",
]
