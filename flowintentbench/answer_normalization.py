"""Conservative, auditable representation rules for extracted answer claims."""
from __future__ import annotations

from copy import deepcopy
import math

from .outcome_scoring import POLICY as BASE_POLICY, digest, score_outcome, source_binding
from .answer_evidence import bind_value

PROTOCOL = "outcome-rubric-v3-normalized-v2"
RULES = {
    "N01": "Equivalent terminology preserves scientific method and scope",
    "N02": "Explicit compatible units may be converted without widening tolerances",
    "N03": "Missing units or scientific choices are never inferred from reference agreement",
    "N04": "Fixed constraints are mandatory; open choices may have valid alternatives",
    "N05": "Mandatory findings and authored alternative groups define coverage",
    "N06": "Primary and supplemental claims are assessed separately",
    "N07": "Method adequacy, information coverage and internal consistency are separate from truth",
    "N08": "Evaluator uncertainty is local and contributes score bounds",
    "N09": "Missing answer content earns deductions; invalid review schemas remain errors",
    "N10": "Every normalization retains original evidence, values and provenance",
}
POLICY = {**BASE_POLICY, "protocol": PROTOCOL, "normalization_rules": RULES,
          "normalization_scope": "Explicit unit representations in already matched claims; no new semantic matches"}

# Symbols are case-sensitive: Pa must never be confused with pA.
_UNITS = {}
for dimension, scale, names in [
    ("ratio", 1.0, ("1", "dimensionless", "unitless", "fraction")),
    ("ratio", 0.01, ("%", "percent", "percentage")),
    ("length", 1.0, ("m", "meter", "meters", "metre", "metres")),
    ("length", 0.01, ("cm", "centimeter", "centimeters")),
    ("length", 0.001, ("mm", "millimeter", "millimeters")),
    ("time", 1.0, ("s", "second", "seconds")),
    ("time", 0.001, ("ms", "millisecond", "milliseconds")),
    ("pressure", 1.0, ("Pa", "pascal", "pascals")),
    ("pressure", 1000.0, ("kPa", "kilopascal", "kilopascals")),
    ("pressure", 100000.0, ("bar",)),
    ("speed", 1.0, ("m/s", "m s^-1")),
    ("speed", 0.01, ("cm/s", "cm s^-1")),
    ("speed", 0.001, ("mm/s", "mm s^-1")),
    ("rate", 1.0, ("1/s", "s^-1", "s^{-1}", "s\u207b\u00b9")),
    ("viscosity", 1.0, ("Pa s", "Pa*s", "Pa.s", "Pa\u00b7s")),
    ("viscosity", 0.001, ("mPa s", "mPa*s", "mPa\u00b7s", "cP")),
    ("angle", 1.0, ("rad", "radian", "radians")),
    ("angle", math.pi / 180.0, ("degree", "degrees", "deg", "\u00b0")),
]:
    for name in names:
        _UNITS[name] = (dimension, scale)


def convert_explicit_units(value, source, target):
    """Return a conversion record, or None when representation is not established."""
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    left = _UNITS.get(" ".join(source.split()))
    right = _UNITS.get(" ".join(target.split()))
    if left is None or right is None or left[0] != right[0]:
        return None
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                         or not math.isfinite(v) for v in values):
        return None
    factor = left[1] / right[1]
    converted = [v * factor for v in values]
    if not all(math.isfinite(v) for v in converted):
        return None
    return {"rule": "N02", "source_unit": source, "target_unit": target,
            "original_value": value, "factor": factor,
            "converted_value": converted if isinstance(value, list) else converted[0]}


def normalize_judgment(answer, gt, judgment):
    """Normalize only already-equivalent references sharing one explicit target unit."""
    normalized = deepcopy(judgment)
    equivalent = {item["branch_id"] for item in judgment["branch_relations"]
                  if item["relation"] == "EQUIVALENT"}
    references = {(branch.operationalization_id, finding.finding_id): finding
                  for branch in gt.findings_by_operationalization for finding in branch.findings}
    ledger = []
    for claim in normalized["claims"]:
        if bind_value(answer, claim["evidence_text"], claim["value"])["status"] != "BOUND":
            continue
        keys = [(match["branch_id"], match["finding_id"]) for match in claim["reference_matches"]]
        if not keys or any(key not in references or key[0] not in equivalent for key in keys):
            continue
        units = {references[key].unit for key in keys}
        if len(units) != 1:
            continue
        target = next(iter(units))
        if claim["unit"] == target:
            continue
        conversion = convert_explicit_units(claim["value"], claim["unit"], target)
        if conversion is not None:
            ledger.append({"claim_id": claim["claim_id"], **conversion})
            claim.update(value=conversion["converted_value"], unit=target)
    return normalized, ledger


def score_normalized_outcome(answer, rubric, gt, verification_policy, judgment):
    # Validate the original review before applying any representation changes.
    baseline = score_outcome(answer, rubric, gt, verification_policy, judgment)
    normalized, ledger = normalize_judgment(answer, gt, judgment)
    result = score_outcome(answer, rubric, gt, verification_policy, normalized, normalizations=ledger)
    result.update(protocol=PROTOCOL, policy_sha256=digest(POLICY),
                  original_judgment_sha256=digest(judgment), normalization_ledger=ledger,
                  baseline_verification=baseline["verification"],
                  normalization_scope=POLICY["normalization_scope"])
    return result
