"""Source binding, one-to-one matching, and frozen numeric checks."""
from __future__ import annotations
import hashlib
import json
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt
from .answer_evidence import bind_quote, bind_value, unit_is_bound, is_native_scale_unit, is_coordinate_extent_statement, dimensionless_unit_proof, implicit_dimensionless_scale, rounding_radius, quantity_match_conflict
from .numeric_verification import EvaluationPendingAdjudication, PredictedAtomicFinding, verify_finding_value
from .frozen_evidence import NumericCheck

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Dimension(Strict):
    dimension: str
    status: Literal['EXTRACTED', 'MISSING', 'AMBIGUOUS', 'CONFLICTING']
    evidence_text: str
    matches: dict[str, StrictBool | None]

class Match(Strict):
    branch_id: str
    finding_id: str

class Finding(Strict):
    finding_id: str
    statement: str
    evidence_text: str
    value: StrictFloat | StrictInt | str | list[StrictFloat | StrictInt] | None
    unit: str | None
    unit_evidence_text: str = ''
    eligible: StrictBool | None
    matches: list[Match]
    numeric_checks: list[NumericCheck] = Field(default_factory=list, max_length=12)

def interval(lower=0.0, upper=1.0, *, applicable=True, reason=None):
    if not applicable:
        return {'status': 'NOT_APPLICABLE', 'value': None, 'lower': None, 'upper': None, 'applicable': False, 'reason': reason, 'supported_value': None, 'value_kind': 'NOT_APPLICABLE'}
    lower, upper = (float(lower), float(upper))
    if not 0 <= lower <= upper <= 1:
        raise ValueError(f'invalid metric bounds: {lower}, {upper}')
    exact = abs(upper - lower) < 1e-12
    return {'status': 'POINT' if exact else 'INTERVAL', 'value': lower if exact else None, 'lower': lower, 'upper': upper, 'applicable': True, 'reason': reason if not exact else None, 'supported_value': lower, 'value_kind': 'EXACT' if exact else 'EVIDENCE_LOWER_BOUND'}

def _assignment(edges, refs):
    """Maximum cardinality, preferring core references for tied matchings."""
    assigned = {}

    def visit(pid, seen):
        for rid in sorted(edges.get(pid, ()), key=lambda r: (refs[r].importance.value != 'core', r)):
            if rid in seen:
                continue
            seen.add(rid)
            if rid not in assigned or visit(assigned[rid], seen):
                assigned[rid] = pid
                return True
        return False
    for pid in sorted(edges):
        visit(pid, set())
    return assigned

def _verify(answer, claim, reference, policy, reference_method=''):
    binding = bind_value(answer, claim.evidence_text, claim.value, coordinate_extent=is_coordinate_extent_statement(claim.statement) and is_coordinate_extent_statement(reference.statement))
    if binding['status'] != 'BOUND':
        return (None, binding['reason'])
    conflict = quantity_match_conflict(claim.statement, binding['source']['text'], reference.statement)
    if conflict:
        return (None, conflict)
    unit_quote = bind_quote(answer, claim.unit_evidence_text) if claim.unit_evidence_text else None
    unit_text = binding['source']['text'] + ('\n' + unit_quote['text'] if unit_quote else '')
    proof = dimensionless_unit_proof(claim.value, binding['source']['text'], reference.statement + ' ' + reference_method, reference.unit, context=answer)
    if proof is None and claim.unit in {None, '1', 'dimensionless', 'unitless'}:
        proof = implicit_dimensionless_scale(claim.value, binding['source']['text'], reference.unit, unit_declaration=unit_quote['text'] if unit_quote else '')
    if not unit_is_bound(claim.unit, unit_text) and proof is None:
        return (None, 'UNIT_NOT_IN_SOURCE')
    if not policy:
        return (None, 'MISSING_VERIFICATION_POLICY')
    mode = policy.get('verification_mode')
    if mode == 'semantic_only':
        return (None, 'QUALITATIVE_SUPPORT_REQUIRES_REVIEW')
    if claim.value is None:
        return (None, 'NONNUMERIC_CLAIM_MATCHED_TO_NUMERIC_REFERENCE')
    value, unit = (claim.value, claim.unit)
    if reference.unit is None and (is_native_scale_unit(unit) or unit in {'stored velocity scale', 'stored velocity units', 'stored coordinate units', 'grid coordinate units', 'dataset coordinate units', 'dataset spatial coordinate units', "dataset's spatial coordinate units", 'dataset’s spatial coordinate units', 'stored units', 'native units', 'field units', 'velocity units', 'stored y-coordinate units', 'stored x-coordinate units', 'stored z-coordinate units', 'stored coordinates', 'stored coordinate-area units', 'square coordinate units', 'snapshot’s stored density–velocity units', 'length^3 in stored units', 'stored coordinate-volume units', 'coordinate-volume units', 'cubic coordinate units', 'mesh coordinate-volume units', 'mesh-volume units', 'volume units', 'stored field scales', "dataset's stored flux-density units", 'dataset’s stored flux-density units'}):
        unit = None
    if proof is not None and (unit is None or unit in {'1', 'dimensionless', 'unitless', '%'}):
        unit = proof['source_unit']
    from .answer_normalization import convert_explicit_units
    conversion = convert_explicit_units(value, unit, reference.unit)
    if conversion:
        value, unit = (conversion['converted_value'], reference.unit)
    predicted = PredictedAtomicFinding(claim.finding_id, claim.statement, value, unit, claim.evidence_text)
    try:
        verdict = verify_finding_value(predicted, reference, explicit_policy=policy)
        if verdict is False and mode in {'scalar_tolerance', 'spatial_euclidean', 'componentwise_vector'}:
            radius = rounding_radius(claim.value, binding['source']['text'], allow_integer=True, min_digits=1)
            if radius is not None and isinstance(reference.value, (int, float)) and (not isinstance(reference.value, bool)):
                parameters = policy.get('verification_parameters') or {}
                tolerance = (parameters.get('absolute_tolerance') or 0) + (parameters.get('relative_tolerance') or 0) * abs(reference.value)
                if abs(value - reference.value) <= radius * abs(conversion['factor'] if conversion else 1) + tolerance:
                    return (None, 'SOURCE_ROUNDING_UNCERTAINTY; strict host tolerance not satisfied')
            if isinstance(value, list) and isinstance(reference.value, list) and (len(value) == len(reference.value)) and (mode in {'spatial_euclidean', 'componentwise_vector'}):
                radii = [rounding_radius(v, binding['source']['text'], allow_integer=True, min_digits=1) or 0.0 for v in claim.value]
                factor = abs(conversion['factor'] if conversion else 1)
                residual = [max(0.0, abs(v - r) - rad * factor) for v, r, rad in zip(value, reference.value, radii)]
                tolerance = (policy.get('verification_parameters') or {}).get('spatial_tolerance') or 0.0
                distance = math.sqrt(sum((v * v for v in residual))) if mode == 'spatial_euclidean' else max(residual)
                if any(radii) and distance <= tolerance:
                    return (None, 'SOURCE_ROUNDING_UNCERTAINTY; strict host vector tolerance not satisfied')
        reason = 'FROZEN_HOST_POLICY; ' + proof['rule'] if proof else 'FROZEN_HOST_POLICY'
        if binding.get('derivation'):
            reason += '; ' + binding['derivation']['rule']
        return (verdict, reason)
    except EvaluationPendingAdjudication as exc:
        return (None, 'UNIT_OR_POLICY_UNRESOLVED: ' + str(exc))
