"""Compare supplied answers against frozen supplementary evidence; never solve.

No mesh readers, recipe execution, subprocesses or network access belong here.
The separate numeric_diagnostics utility is offline reference-construction code.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt
from .answer_evidence import bind_quote, bind_value, rounding_radius, value_is_bound, numeric_notation

VERSION = "frozen-auxiliary-evidence-v1"
class NumericCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    branch_id: str
    statistic: Literal["mean_x", "mean_y", "std_x", "std_y", "covariance", "pearson",
                       "r_squared", "regression_slope", "regression_intercept", "conditional_mean_y", "conditional_max_abs_y",
                       "layer_pearson_min", "layer_pearson_max", "layer_pearson_median"]
    # Components refer to the extracted value, not to a mesh field component.
    value_index: StrictInt | None = None
    axis: Literal["x", "y", "z"] | None = None
    lower_x: StrictFloat | StrictInt | None = None
    upper_x: StrictFloat | StrictInt | None = None
    method_evidence_text: str
    scope_evidence_text: str = ""
    scope_status: Literal["EXPLICIT", "UNRESOLVED"] = "UNRESOLVED"



def _pointer(value, pointer):
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError("invalid reference JSON pointer")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def load_bank(path, root):
    """Validate every supplied reference against its preserved JSON artifact.

    A checksum proves identity, not correctness: evidence authority is reported
    and remains a responsibility of benchmark construction/calibration.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("protocol") != "FROZEN_AUXILIARY_REFERENCES_V1":
        raise ValueError("unsupported supplementary evidence protocol")
    result, seen, loaded = {}, set(), {}
    for entry in payload["entries"]:
        query = entry["query"]
        if set(query) != {"branch_id", "statistic", "axis", "lower_x", "upper_x"}:
            raise ValueError("incomplete frozen evidence query")
        if not entry.get("authority") or not entry.get("branch_evidence_sha256"):
            raise ValueError("supplementary reference lacks authority/context")
        key = (entry["case_id"], json.dumps(query, sort_keys=True))
        if key in seen:
            raise ValueError("duplicate supplementary reference")
        seen.add(key)
        source = entry["source"]
        artifact = (Path(root) / source["path"]).resolve()
        if artifact.suffix != ".json" or not artifact.is_relative_to(Path(root).resolve()):
            raise ValueError("supplementary reference must be a local frozen JSON artifact")
        if artifact not in loaded:
            raw = artifact.read_bytes()
            loaded[artifact] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
        checksum, data = loaded[artifact]
        if checksum != source["sha256"]:
            raise ValueError("frozen supplementary evidence checksum mismatch")
        expected = entry["expected"]
        if type(expected) not in {int, float} or not math.isfinite(expected):
            raise ValueError("supplementary expected value must be finite")
        if _pointer(data, source["pointer"]) != expected:
            raise ValueError("supplementary value differs from preserved source")
        result.setdefault(entry["case_id"], []).append(entry)
    return result


def catalog(material):
    entries = material.get("frozen_auxiliary_evidence", [])
    if not entries:
        return None
    return {"population": "Complete rectilinear cells; arithmetic vertex means for point fields; volume weights.",
        "queries": [r["query"] for r in entries],
        "binding": "Bind only identical quantity, population, fields, weighting and method; quote method AND local scope. Restricted regions are not whole-domain evidence. Do not invent selectors or checks."}

def verify_checks(material, answer, claim, equivalent_branches, coordinate_axes=None):
    """Return local scientific truth plus complete provenance for a claim.

False is conclusive if one correctly bound scalar is contradicted. True needs
every extracted scalar component verified; partial success remains unknown.
    Auxiliary values have an explicit policy: reproduce the reported decimal
    precision (at least two significant digits), independent of frozen core GT.
"""
    queries = claim.numeric_checks
    if not queries:
        return None, []
    if len({q.branch_id for q in queries}) != 1:
        return None, [{"query": q.model_dump(mode="json"), "verdict": None,
                       "reason": "AUXILIARY_CHECK_UNRESOLVED: split different result groups before binding"} for q in queries]
    binding = bind_value(answer, claim.evidence_text, claim.value)
    values = claim.value if isinstance(claim.value, list) else [claim.value]
    rows, covered = [], set()
    for query in queries:
        row = {"query": query.model_dump(mode="json"), "verdict": None, "protocol": VERSION}
        try:
            if binding["status"] != "BOUND":
                raise ValueError(binding["reason"])
            if query.branch_id not in equivalent_branches:
                raise ValueError("method not independently bound to a supported branch")
            method = bind_quote(answer, query.method_evidence_text)
            if not method:
                raise ValueError("method quotation unbound")
            if query.scope_status != "EXPLICIT" or not bind_quote(answer, query.scope_evidence_text):
                raise ValueError("population scope unresolved or unbound")
            quoted_method = numeric_notation(query.method_evidence_text + "\n" + query.scope_evidence_text).casefold()
            if re.search(r"\b(?:unweighted|equal[- ]weight(?:ed)?|point[- ]level|point values directly|demeaned)\b", quoted_method):
                raise ValueError("METHOD_REFERENCE_MISMATCH: frozen reference uses volume-weighted cell values")
            if (not query.statistic.startswith("conditional_") and
                    re.search(r"\b(?:restricted|subset|within[- ](?:the[- ])?(?:band|region)|interface[- ]band|density[- ]bin)\b", quoted_method)):
                raise ValueError("SCOPE_REFERENCE_MISMATCH: regional statistic has no whole-domain reference")
            index = query.value_index if isinstance(claim.value, list) else 0
            axis = query.axis
            repairs = []
            # Recover omitted JSON selectors only from explicit source labels,
            # never from which independently computed number happens to match.
            if query.statistic.startswith("layer_"):
                source = numeric_notation(claim.evidence_text)
                if isinstance(claim.value, list) and index is None:
                    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                    limits = re.search(r"\brange\s+"+number+r"\s+to\s+"+number, source, re.I)
                    median = re.search(r"\bmedian\s+"+number, source, re.I)
                    value_from_label = None
                    if limits and query.statistic in {"layer_pearson_min", "layer_pearson_max"}:
                        value_from_label = (min if query.statistic.endswith("min") else max)(float(v) for v in limits.groups())
                    elif median and query.statistic == "layer_pearson_median":
                        value_from_label = float(median[1])
                    positions = [i for i, value in enumerate(values) if value == value_from_label]
                    if value_from_label is not None and len(positions) == 1:
                        index = positions[0]
                        repairs.append("VALUE_INDEX_FROM_EXPLICIT_RANGE_OR_MEDIAN_LABEL")
                axis_source = query.scope_evidence_text + '\n' + query.method_evidence_text
                axes = set(re.findall(r"\b([xyz])[- ](?:planes?|layers?|levels?)\b", axis_source, re.I))
                if re.search(r"\bvertical (?:planes?|layers?|levels?)\b", axis_source, re.I):
                    axes.update(a for a, meaning in (coordinate_axes or {}).items() if isinstance(meaning, str) and re.search(r"\bvertical\b", meaning, re.I))
                axes = {a.lower() for a in axes}
                if len(axes) == 1:
                    source_axis = next(iter(axes))
                    if axis is not None and axis != source_axis:
                        raise ValueError("axis contradicts quoted scope and public coordinate metadata")
                    if axis is None:
                        axis = source_axis
                        repairs.append("AXIS_FROM_QUOTED_SCOPE_AND_PUBLIC_COORDINATES")
            row.update(resolved_value_index=index, resolved_axis=axis, selector_repairs=repairs)
            if (isinstance(claim.value, list) and index is None
                    or not isinstance(claim.value, list) and query.value_index is not None
                    or type(index) is not int or not 0 <= index < len(values)):
                raise ValueError("invalid scalar component")
            value = values[index]
            if type(value) not in {int, float}:
                raise ValueError("nonnumeric extracted component")
            if claim.unit is not None:
                dimensionless = query.statistic in {"pearson", "r_squared"} or query.statistic.startswith("layer_")
                allowed = {"1", "dimensionless", "unitless"} if dimensionless else {
                    "stored units", "native units", "stored units as applicable", "stored velocity units"}
                if claim.unit not in allowed:
                    raise ValueError("supplemental unit conversion not established")
            conditional = query.statistic.startswith("conditional_")
            if query.statistic.startswith("layer_") != (axis is not None):
                raise ValueError("axis required only for layer statistics")
            if conditional != (query.lower_x is not None or query.upper_x is not None):
                raise ValueError("conditional statistics require explicit bounds only")
            if query.lower_x is not None and query.upper_x is not None and query.lower_x >= query.upper_x:
                raise ValueError("invalid condition bounds")
            for threshold in (query.lower_x, query.upper_x):
                if threshold is not None and not value_is_bound(threshold, method["text"] + '\n' + claim.evidence_text):
                    raise ValueError("condition threshold not present in source")
            entry = material["branch_execution_evidence"][query.branch_id]
            provenance = entry["execution_provenance"]
            recipe = provenance["recipe"]
            if recipe.get("kind") != "association" or recipe.get("measure") != "pearson":
                raise ValueError("unsupported branch recipe")
            signature = {"branch_id": query.branch_id, "statistic": query.statistic,
                         "axis": axis, "lower_x": query.lower_x, "upper_x": query.upper_x}
            references = [r for r in material.get("frozen_auxiliary_evidence", []) if r["query"] == signature]
            if len(references) != 1:
                raise ValueError("no unique frozen supplementary reference")
            reference = references[0]
            if reference["branch_evidence_sha256"] != entry["evidence_binding_sha256"]:
                raise ValueError("supplementary reference belongs to another frozen method/data context")
            expected = reference["expected"]
            if not math.isfinite(expected):
                raise ValueError("nonfinite independent statistic")
            radius = rounding_radius(value, binding["source"]["text"], allow_integer=True, min_digits=2)
            residual = abs(value-expected)
            numerical_epsilon = 1e-10 * max(abs(expected), 1e-8)
            verdict = residual <= numerical_epsilon + (radius or 0.0)
            row.update(verdict=verdict, expected=expected, reported=value, rounding_radius=radius,
                       reason="INDEPENDENT_REPORTED_PRECISION_MATCH" if verdict else "INDEPENDENT_NUMERIC_CONTRADICTION",
                       verification_policy="AUXILIARY_REPORTED_PRECISION_V1; two or more significant digits; no alteration of frozen core tolerance",
                       reference_source=reference["source"], recipe=recipe, evidence_origin="FROZEN_SUPPLIED_REFERENCE",
                       semantic_binding="REVIEWER_QUOTED_METHOD; not proved by numerical agreement")
            if verdict is True:
                covered.add(index)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row["reason"] = "AUXILIARY_CHECK_UNRESOLVED: " + str(exc)
        rows.append(row)
    if any(r["verdict"] is False for r in rows):
        return False, rows
    return (True if len(covered) == len(values) else None), rows
