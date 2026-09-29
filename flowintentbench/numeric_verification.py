"""Deterministic numerical verification under frozen finding policies."""
from __future__ import annotations
import json
import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Callable, Mapping
from .ground_truth import FindingValue, ReferenceFinding

class EvaluationConfigurationError(ValueError):
    """The benchmark/evaluator inputs cannot produce a formal score."""

class EvaluationPendingAdjudication(RuntimeError):
    """A formal score cannot be finalized until a judgment is resolved."""

    def __init__(self, message: str, *, pending_type: str='adjudication', continuation_context: Mapping[str, Any] | None=None) -> None:
        super().__init__(message)
        self.pending_type = pending_type
        self.continuation_context = {} if continuation_context is None else dict(continuation_context)

class FindingVerificationMode(str, Enum):
    """Explicit GT value rule exposed to Finding semantic matching."""
    SEMANTIC_ONLY = 'semantic_only'
    EXACT_DISCRETE_NUMERIC = 'exact_discrete_numeric'
    SCALAR_TOLERANCE = 'scalar_tolerance'
    SPATIAL_EUCLIDEAN = 'spatial_euclidean'
    COMPONENTWISE_VECTOR = 'componentwise_vector'
    EXACT_IDENTITY = 'exact_identity'
EvidenceSpan = str | tuple[int, int] | None

def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string')
    return value.strip()

def _validate_evidence_span(value: EvidenceSpan) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, tuple) or len(value) != 2 or any((isinstance(item, bool) or not isinstance(item, int) for item in value)) or (value[0] < 0) or (value[1] <= value[0]):
        raise ValueError('evidence_span must be text, a non-empty offset range, or None')

@dataclass(frozen=True)
class PredictedAtomicFinding:
    prediction_id: str
    statement: str
    value: FindingValue = None
    unit: str | None = None
    evidence_span: EvidenceSpan = None

    def __post_init__(self) -> None:
        _text(self.prediction_id, 'prediction_id')
        _text(self.statement, 'statement')
        object.__setattr__(self, 'value', _validate_predicted_finding_value(self.value))
        if self.unit is not None:
            _text(self.unit, 'unit')
        _validate_evidence_span(self.evidence_span)

def _validate_predicted_finding_value(value: FindingValue) -> FindingValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise ValueError('PredictedAtomicFinding.value does not accept bool')
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('PredictedAtomicFinding.value floats must be finite')
        return value
    if not isinstance(value, list):
        raise ValueError('PredictedAtomicFinding.value must be str, int, finite float, list[finite numeric values], list[str], or None')
    if all((isinstance(item, str) for item in value)):
        return value
    if all((isinstance(item, (int, float)) and (not isinstance(item, bool)) and math.isfinite(float(item)) for item in value)):
        return [float(item) for item in value]
    raise ValueError('PredictedAtomicFinding.value lists must contain only strings or finite numeric values')

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
    rule: str = ''
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.request_id, 'unit/frame resolution request_id')
        normalized_status = _text(self.status, 'unit/frame resolution status').upper()
        if normalized_status not in {'RESOLVED', 'NO_MATCH'}:
            raise EvaluationConfigurationError('unit/frame resolution status must be RESOLVED or NO_MATCH')
        object.__setattr__(self, 'status', normalized_status)
        object.__setattr__(self, 'request', _json_value(dict(self.request), 'unit/frame resolution request'))
        expected = continuation_request_id('unit_frame', self.request)
        if self.request_id != expected:
            raise EvaluationConfigurationError('unit/frame resolution request_id does not match its request')
        object.__setattr__(self, 'canonical_value', _validate_predicted_finding_value(self.canonical_value))
        object.__setattr__(self, 'provenance', _json_value(dict(self.provenance), 'unit/frame resolution provenance'))
        if self.canonical_unit is not None:
            _text(self.canonical_unit, 'unit/frame resolution canonical_unit')
        if self.status == 'RESOLVED' and (not str(self.rule).strip()):
            raise EvaluationConfigurationError('resolved unit/frame result requires an explicit rule')
        if self.status == 'RESOLVED' and self.canonical_value is None:
            raise EvaluationConfigurationError('resolved unit/frame result requires canonical_value')
        if self.status == 'RESOLVED' and (not self.provenance):
            raise EvaluationConfigurationError('resolved unit/frame result requires provenance')

    def to_dict(self) -> dict[str, Any]:
        return {'request_id': self.request_id, 'request': dict(self.request), 'status': self.status, 'canonical_value': self.canonical_value, 'canonical_unit': self.canonical_unit, 'rule': self.rule, 'provenance': dict(self.provenance)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'UnitFrameResolution':
        request = value.get('request')
        if not isinstance(request, Mapping):
            raise EvaluationConfigurationError('unit/frame resolution must contain a request object')
        return cls(request_id=str(value.get('request_id', '')), request=request, status=str(value.get('status', '')), canonical_value=value.get('canonical_value'), canonical_unit=value.get('canonical_unit'), rule=str(value.get('rule', '')), provenance=value.get('provenance', {}))
UnitConverter = Callable[[FindingValue, str, str], FindingValue | None]

def _json_value(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be JSON-serializable') from exc

def _canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the same canonical form as execution artifacts."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
CONTINUATION_REQUEST_ID_VERSION = 'continuation-request-v1'

def continuation_request_id(kind: str, payload: Mapping[str, Any]) -> str:
    """Return the stable identity of one unresolved continuation request.

    Request identity is deliberately derived from the concrete unresolved
    comparison, not merely from its pending type.  This prevents two semantic
    or unit requests in the same case from being mistaken for a cycle.
    """
    normalized_kind = _text(kind, 'continuation request kind')
    canonical = {'schema_version': CONTINUATION_REQUEST_ID_VERSION, 'kind': normalized_kind, 'payload': _json_value(dict(payload), 'continuation request payload')}
    return f'{normalized_kind}:{_canonical_json_sha256(canonical)}'

def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))

def _numeric_list(value: Any) -> bool:
    return isinstance(value, list) and all((_finite_number(item) for item in value))

def _unit_frame_request_payload(predicted: PredictedAtomicFinding, reference: ReferenceFinding, explicit_policy: Mapping[str, Any] | str | None) -> dict[str, Any]:
    policy_id = explicit_policy.get('scientific_finding_policy_key') if isinstance(explicit_policy, Mapping) else None
    return {'prediction_id': predicted.prediction_id, 'reference_id': reference.finding_id, 'predicted_value': predicted.value, 'predicted_unit': predicted.unit, 'reference_value': reference.value, 'reference_unit': reference.unit, 'verification_policy_id': policy_id, 'unit_frame_authority': explicit_policy.get('unit_frame_authority') if isinstance(explicit_policy, Mapping) else None}

def _normalize_unit(predicted: PredictedAtomicFinding, reference: ReferenceFinding, unit_converter: UnitConverter | None, explicit_policy: Mapping[str, Any] | None=None) -> FindingValue:
    resolution = explicit_policy.get('unit_frame_resolution') if isinstance(explicit_policy, Mapping) else None
    if isinstance(resolution, Mapping):
        expected_request = _unit_frame_request_payload(predicted, reference, explicit_policy)
        expected_request_id = continuation_request_id('unit_frame', expected_request)
        typed_resolution = UnitFrameResolution.from_dict(resolution)
        if typed_resolution.request_id != expected_request_id or typed_resolution.request != expected_request:
            raise EvaluationConfigurationError('unit/frame resolution does not match the unresolved request')
        if typed_resolution.status == 'RESOLVED':
            return typed_resolution.canonical_value
        if typed_resolution.status == 'NO_MATCH':
            return None
    if predicted.unit == reference.unit:
        return predicted.value
    frame_authority = explicit_policy.get('unit_frame_authority', {}) if isinstance(explicit_policy, Mapping) else {}
    coordinate_scale = frame_authority.get('coordinate_or_numeric_scale') if isinstance(frame_authority, Mapping) else None
    normalized_predicted_unit = ' '.join(str(predicted.unit or '').casefold().split())
    if reference.unit is None and coordinate_scale == 'CASE_READER_METADATA' and (normalized_predicted_unit in {'dataset coordinates', 'dataset coordinate units', 'dataset coordinate frame', "dataset's coordinate frame", 'stored cartesian coordinates', 'cartesian coordinates'}):
        return predicted.value
    if predicted.unit is None or reference.unit is None or unit_converter is None:
        raise EvaluationPendingAdjudication(f'unit relationship is unresolved for prediction {predicted.prediction_id!r} and reference {reference.finding_id!r}', pending_type='unit_relationship', continuation_context={'route_owner': 'UNIT_FRAME_RESOLVER', 'required_next_action': 'RESOLVE_UNIT_FRAME', 'continuation_request_id': continuation_request_id('unit_frame', _unit_frame_request_payload(predicted, reference, explicit_policy)), 'continuation_request': _unit_frame_request_payload(predicted, reference, explicit_policy), 'unit_frame_request': _unit_frame_request_payload(predicted, reference, explicit_policy)})
    converted = unit_converter(predicted.value, predicted.unit, reference.unit)
    if converted is None:
        raise EvaluationPendingAdjudication(f'unit conversion is unresolved for prediction {predicted.prediction_id!r} and reference {reference.finding_id!r}', pending_type='unit_relationship', continuation_context={'route_owner': 'UNIT_FRAME_RESOLVER', 'required_next_action': 'RESOLVE_UNIT_FRAME', 'continuation_request_id': continuation_request_id('unit_frame', _unit_frame_request_payload(predicted, reference, explicit_policy)), 'continuation_request': _unit_frame_request_payload(predicted, reference, explicit_policy), 'unit_frame_request': _unit_frame_request_payload(predicted, reference, explicit_policy)})
    return converted

def verify_finding_value(predicted: PredictedAtomicFinding, reference: ReferenceFinding, *, unit_converter: UnitConverter | None=None, explicit_policy: Mapping[str, Any] | str | None=None) -> bool:
    """Execute only the deterministic rule explicitly supported by the GT.

    Semantic matching is authoritative when no applicable deterministic rule is
    present.  In particular, string values and untyped numeric lists do not
    acquire hidden exact or spatial semantics from their Python representation.
    """
    verified, _ = _verify_finding_value(predicted, reference, unit_converter=unit_converter, explicit_policy=explicit_policy)
    return verified

def finding_verification_mode(reference: ReferenceFinding, explicit_policy: Mapping[str, Any] | str | None=None) -> FindingVerificationMode:
    """Expose the explicit GT verifier rule to Finding semantic matching."""
    declared_mode = explicit_policy.get('verification_mode') if isinstance(explicit_policy, Mapping) else explicit_policy
    if declared_mode is not None:
        try:
            mode = FindingVerificationMode(str(declared_mode))
        except ValueError as exc:
            raise EvaluationConfigurationError(f'unsupported explicit Finding verification mode: {declared_mode!r}') from exc
        parameters = explicit_policy.get('verification_parameters') if isinstance(explicit_policy, Mapping) else None
        if parameters is not None:
            expected = reference.verification.model_dump(mode='json') if reference.verification is not None else {'absolute_tolerance': None, 'relative_tolerance': None, 'spatial_tolerance': None}
            if dict(parameters) != expected:
                raise EvaluationConfigurationError('explicit Finding verification policy conflicts with GroundTruth')
        if mode == FindingVerificationMode.EXACT_DISCRETE_NUMERIC and (not (isinstance(reference.value, int) and (not isinstance(reference.value, bool)))):
            raise EvaluationConfigurationError('exact_discrete_numeric requires an integer reference value')
        if mode == FindingVerificationMode.EXACT_IDENTITY and (not isinstance(reference.value, str)):
            raise EvaluationConfigurationError('exact_identity requires a string reference value')
        if mode == FindingVerificationMode.COMPONENTWISE_VECTOR and (not (_numeric_list(reference.value) and bool(reference.value))):
            raise EvaluationConfigurationError('componentwise_vector requires a non-empty numeric vector reference value')
        verification = reference.verification
        if mode == FindingVerificationMode.SCALAR_TOLERANCE and (verification is None or (verification.absolute_tolerance is None and verification.relative_tolerance is None)):
            raise EvaluationConfigurationError('scalar_tolerance requires an explicit GT scalar tolerance')
        if mode in {FindingVerificationMode.SPATIAL_EUCLIDEAN, FindingVerificationMode.COMPONENTWISE_VECTOR} and (verification is None or verification.spatial_tolerance is None):
            raise EvaluationConfigurationError('spatial verification requires an explicit GT spatial tolerance')
        return mode
    verification = reference.verification
    if verification is not None and verification.spatial_tolerance is not None:
        return FindingVerificationMode.SPATIAL_EUCLIDEAN
    if verification is not None and (verification.absolute_tolerance is not None or verification.relative_tolerance is not None):
        return FindingVerificationMode.SCALAR_TOLERANCE
    return FindingVerificationMode.SEMANTIC_ONLY

def _verify_finding_value(predicted: PredictedAtomicFinding, reference: ReferenceFinding, *, unit_converter: UnitConverter | None=None, explicit_policy: Mapping[str, Any] | str | None=None) -> tuple[bool, str]:
    """Return ``(verified, applied_rule)`` after semantic MATCH."""
    reference_value = reference.value
    if reference_value is None:
        return (True, 'semantic_only')
    if isinstance(reference_value, bool):
        raise EvaluationConfigurationError('boolean ReferenceFinding values are invalid')
    verification = reference.verification
    mode = finding_verification_mode(reference, explicit_policy)
    if mode == FindingVerificationMode.EXACT_DISCRETE_NUMERIC:
        if predicted.value is None:
            return (False, 'exact_discrete_numeric')
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        return (_finite_number(predicted_value) and float(predicted_value) == reference_value, 'exact_discrete_numeric')
    if mode == FindingVerificationMode.EXACT_IDENTITY:
        if predicted.value is None:
            return (False, 'exact_identity')
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        return (predicted_value == reference_value, 'exact_identity')
    if mode == FindingVerificationMode.SCALAR_TOLERANCE:
        if not _finite_number(reference_value):
            raise EvaluationConfigurationError('scalar tolerance is incompatible with ReferenceFinding.value')
        if predicted.value is None:
            return (False, 'scalar_tolerance')
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _finite_number(predicted_value):
            return (False, 'scalar_tolerance')
        assert verification is not None
        absolute = 0.0 if verification.absolute_tolerance is None else verification.absolute_tolerance
        relative = 0.0 if verification.relative_tolerance is None else verification.relative_tolerance
        return (abs(float(predicted_value) - float(reference_value)) <= absolute + relative * abs(float(reference_value)), 'scalar_tolerance')
    if mode == FindingVerificationMode.SPATIAL_EUCLIDEAN:
        if not _numeric_list(reference_value) or not reference_value:
            raise EvaluationConfigurationError('spatial tolerance is incompatible with ReferenceFinding.value')
        if predicted.value is None:
            return (False, 'spatial_euclidean')
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _numeric_list(predicted_value) or len(predicted_value) != len(reference_value):
            return (False, 'spatial_euclidean')
        distance = math.sqrt(sum(((float(predicted_item) - float(reference_item)) ** 2 for predicted_item, reference_item in zip(predicted_value, reference_value, strict=True))))
        assert verification is not None and verification.spatial_tolerance is not None
        return (distance <= verification.spatial_tolerance, 'spatial_euclidean')
    if mode == FindingVerificationMode.COMPONENTWISE_VECTOR:
        if not _numeric_list(reference_value) or not reference_value:
            raise EvaluationConfigurationError('componentwise vector tolerance is incompatible with ReferenceFinding.value')
        if predicted.value is None:
            return (False, 'componentwise_vector')
        predicted_value = _normalize_unit(predicted, reference, unit_converter, explicit_policy if isinstance(explicit_policy, Mapping) else None)
        if not _numeric_list(predicted_value) or len(predicted_value) != len(reference_value):
            return (False, 'componentwise_vector')
        assert verification is not None and verification.spatial_tolerance is not None
        tolerance = float(verification.spatial_tolerance)
        return (all((abs(float(predicted_item) - float(reference_item)) <= tolerance for predicted_item, reference_item in zip(predicted_value, reference_value, strict=True))), 'componentwise_vector')
    return (True, 'semantic_only')
