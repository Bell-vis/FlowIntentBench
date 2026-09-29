"""Frozen engineering acceptance contract for the 28-question closure run.

This module does not add a scientific schema or a lifecycle gate.  It turns
the already-frozen target/O/F/visibility contracts into deterministic
engineering checks, and supplies the compact sentinel challenge suite required
before a human-facing portfolio can be called closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from itertools import combinations
from typing import Any, Mapping, Sequence

from .hccq import audit_family_presentation_leakage, audit_human_presentation


HUMAN_FACING_ACCEPTANCE_CONTRACT_VERSION = "human-facing-acceptance-contract-v2"
REQUIRED_CONDITIONS = ("O1-F1", "O2-F1", "O3-F1", "O1-F2")


_FIXED_CLAUSE_PATTERNS: dict[str, tuple[str, ...]] = {
    "face_connected_region": (
        r"face[- ]connected",
        r"connected through shared (?:grid )?faces",
        r"shared (?:grid )?faces",
        r"six[- ]neighbor connected",
        r"adjacent by one structured[- ]grid index step",
    ),
    "legacy_feature_definition": (
        r"mesh[- ]connected",
        r"connected (?:mesh )?(?:locations|points|cells)",
        r"share an unstructured[- ]mesh cell",
    ),
    "unstructured_cell_connected_region": (
        r"share an unstructured[- ]mesh cell",
        r"unstructured[- ]mesh cell",
    ),
    "mean_region_coordinate": (
        r"mean spatial location",
        r"mean (?:location|coordinate)",
        r"centroid",
    ),
    "peak_speed": (r"peak speed",),
    "speed_gt_0_20": (
        r"(?:above|greater than|exceeding)\s*0\.20",
        r"0\.20\s*(?:threshold|cutoff)",
    ),
    "speed_q90_nonzero": (
        r"90th percentile",
        r"top ten percent",
    ),
    "kitchen_concentration_heterogeneity:aggregation_or_representation": (
        r"which (?:gas[- ]?)?concentration field",
        r"selected (?:gas[- ]?)?concentration field",
        r"field (?:was )?selected",
        r"selected concentration field",
    ),
    "kitchen_concentration_heterogeneity:feature_definition": (
        r"(?:every|all|available) (?:stored )?gas[- ]?concentration (?:scalar|field)",
        r"gas[- ]?concentration fields? as (?:the )?candidates?",
        r"available gas[- ]?concentration field",
    ),
    "kitchen_concentration_heterogeneity:property_measure": (
        r"cell[- ]volume[- ]weighted coefficient of variation",
        r"volume[- ]weighted coefficient of variation",
    ),
    "provisional_constant_density_connected_surface": (
        r"connected (?:component|part)s? of (?:a |the )?constant[- ]density isosurface",
        r"connected constant[- ]density isosurface",
    ),
    "provisional_density_q90_largest_surface": (
        r"90th percentile.*(?:density|Density)",
        r"largest surface area",
        r"largest[- ]area (?:surface|component)",
    ),
    "provisional_surface_centroid_area": (
        r"area[- ]weighted centroid",
        r"surface area",
    ),
}


HUMAN_FACING_ACCEPTANCE_CONTRACT: dict[str, Any] = {
    "version": HUMAN_FACING_ACCEPTANCE_CONTRACT_VERSION,
    "required_slot_count": 28,
    "required_dataset_count": 7,
    "required_conditions": list(REQUIRED_CONDITIONS),
    "required_hf_invariants": [f"HF{index}" for index in range(1, 11)],
    "required_primary_gates": [
        "genuine_authored_primary",
        "target_clarity",
        "responsibility_integrity",
        "visibility_integrity",
        "semantic_fidelity",
        "hccq",
        "fresh_flow_expert_final_review",
        "family_condition_integrity",
        "family_template_integrity",
    ],
    # Sentinel mutations test the auditor, not the scientific membership or
    # human-interface outcome of any original case.
    "auditor_sentinel": {
        "reported_separately": True,
        "may_change_core_membership": False,
    },
    "semantic_challenges": [f"M{index}" for index in range(1, 13)],
    "preserving_challenges": [f"P{index}" for index in range(1, 6)],
    "failure_mapping": {
        "PRESENTATION_DEFECT": [
            "BACKEND_SCHEMA_LEAKAGE",
            "MECHANICAL_TRANSCRIPTION_BURDEN",
            "PROTOCOL_HEAVY_WORDING",
            "FAMILY_TEMPLATE_LEAKAGE",
        ],
        "RESPONSIBILITY_REALIZATION_DEFECT": [
            "UNDECLARED_RESPONDENT_CHOICE",
            "MISSING_FIXED_O",
            "OPEN_O_PREMATURELY_FIXED",
            "FIXED_OPEN_O_CONTRADICTION",
            "FINDING_RESPONSIBILITY_DRIFT",
            "O1_F2_EFFECTIVE_O_DRIFT",
        ],
        "VALIDATOR_DEFECT": ["SENTINEL_FALSE_ACCEPT", "SENTINEL_FALSE_REJECT"],
        "CONTEXT_DEFECT": ["REQUIRED_INFORMATION_BACKEND_ONLY"],
        "SCIENTIFIC_CONTRACT_DEFECT": ["TARGET_DRIFT", "TARGET_INVARIANCE_DRIFT"],
        "RELEASE_ONLY_BLOCKER": [
            "BLOCKED_PENDING_EVIDENCE",
            "BLOCKED_PENDING_O_SPACE",
            "NEW_CASE_REQUIRED",
        ],
    },
    "fixed_clause_patterns": {
        key: list(value) for key, value in sorted(_FIXED_CLAUSE_PATTERNS.items())
    },
    "thresholds": {
        "semantic_false_accepts": 0,
        "preserving_false_rejects": 0,
        "required_valid_audit_records": 28,
        "same_defect_root_cause_trigger": 3,
    },
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def human_facing_acceptance_contract_sha256() -> str:
    return canonical_json_sha256(HUMAN_FACING_ACCEPTANCE_CONTRACT)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalized_text(value: Any) -> str:
    text = _text(value).casefold().replace("’", "'")
    text = re.sub(r"[_/]", " ", text)
    text = re.sub(r"[^a-z0-9.\-']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dimension_ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        dimension = (
            _text(item.get("dimension_id") or item.get("dimension"))
            if isinstance(item, Mapping)
            else _text(item)
        )
        if dimension:
            result.append(dimension)
    return result


def _fixed_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _projection(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("canonical_semantic_projection")
    return value if isinstance(value, Mapping) else row


def _finding_mode(projection: Mapping[str, Any]) -> str:
    finding = projection.get("finding_responsibility")
    if isinstance(finding, Mapping):
        return _text(finding.get("finding_mode")).upper()
    return _text(finding).upper()


def _combined_visible_text(question: str, visible_context: str | None) -> str:
    return "\n".join(
        item for item in (_text(question), _text(visible_context)) if item
    )


def _target_anchor_groups(target: str) -> list[tuple[str, ...]]:
    value = _normalized_text(target)
    if "high speed" in value or "high-speed" in value:
        return [(r"\bhigh[- ]speed\b",)]
    if "spatial heterogeneity" in value:
        return [
            (r"\bheterogene(?:ity|ous)\b",),
            (r"\b(?:gas[- ]?)?concentration\b",),
        ]
    if "constant density" in value or "constant-density" in value:
        return [
            (r"\bconstant[- ]density\b", r"\bdensity isosurface\b"),
            (r"\b(?:structure|surface|isosurface)\b",),
        ]
    if "airflow region" in value or "ventilation" in value:
        return [
            (r"\b(?:airflow|ventilation|high[- ]speed)\b",),
            (r"\b(?:region|structure)\b",),
        ]
    keywords = [
        token
        for token in re.findall(r"[a-z][a-z-]+", value)
        if token
        not in {
            "the",
            "a",
            "an",
            "in",
            "of",
            "and",
            "flow",
            "field",
            "strongest",
            "scientifically",
            "relevant",
        }
        and len(token) >= 4
    ]
    return [(rf"\b{re.escape(token)}\b",) for token in keywords[:2]]


def _target_is_clear(target: str, text: str) -> bool:
    normalized = _normalized_text(text)
    groups = _target_anchor_groups(target)
    return bool(groups) and all(
        any(re.search(pattern, normalized, re.I) for pattern in group)
        for group in groups
    )


def _fixed_clause_is_visible(row: Mapping[str, Any], text: str) -> bool:
    canonical_id = _text(row.get("canonical_id"))
    patterns = _FIXED_CLAUSE_PATTERNS.get(canonical_id)
    if patterns:
        return any(re.search(pattern, text, re.I | re.S) for pattern in patterns)
    meaning = _normalized_text(row.get("normalized_meaning"))
    tokens = [
        token
        for token in re.findall(r"[a-z0-9.]+", meaning)
        if len(token) >= 5
        and token
        not in {
            "scientifically",
            "selected",
            "strongest",
            "represent",
            "locations",
            "location",
            "measure",
        }
    ]
    present = sum(bool(re.search(rf"\b{re.escape(token)}\b", text, re.I)) for token in tokens)
    return present >= min(2, len(tokens)) if tokens else True


_FIELD_CHOICE_PATTERNS = (
    r"\b(?:choose|select|decide|determine)\s+(?:which|an?|the)?\s*(?:available\s+)?(?:velocity[- ]related\s+|flow\s+)?field\b",
    r"\bwhich\s+(?:available\s+)?(?:velocity[- ]related\s+|flow\s+)?field\s+(?:should|will|was|is)\b",
    r"\breport\s+(?:which|the)\s+(?:available\s+)?field\s+(?:was\s+)?selected\b",
    r"\bstate\s+which\s+(?:available\s+)?field\s+(?:you\s+)?selected\b",
)


_OPEN_DIMENSION_FIXED_MARKERS: dict[str, tuple[str, ...]] = {
    "criterion": (
        r"\b90th percentile\b",
        r"\babove\s*0\.20\b",
        r"\bgreater than\s*0\.20\b",
    ),
    "property_measure": (
        r"\bpeak speed\b",
        r"\bcoefficient of variation\b",
    ),
    "aggregation_or_representation": (
        r"\bmean spatial location\b",
        r"\barea[- ]weighted centroid\b",
    ),
    "feature_definition": (
        r"\bface[- ]connected\b",
        r"\bmesh[- ]connected\b",
        r"\bshare an unstructured[- ]mesh cell\b",
        r"\bconnected constant[- ]density isosurface\b",
    ),
    "observable_selection": (
        r"\bvelocity magnitude\b",
        r"\bstored velocity\b",
    ),
}


_OPEN_DIMENSION_CHOICE_MARKERS: dict[str, tuple[str, ...]] = {
    "criterion": (r"\b(?:choose|select|define|specify).*\bcriterion\b",),
    "property_measure": (
        r"\b(?:choose|select|specify|use).*\b(?:measure|metric)\b",
    ),
    "aggregation_or_representation": (
        r"\b(?:choose|select|specify|define).*\b(?:representation|aggregation|represent)\b",
    ),
    "feature_definition": (
        r"\b(?:choose|select|define|specify).*\b(?:feature|region|structure definition)\b",
        r"\bdefine what constitutes\b",
    ),
    "observable_selection": _FIELD_CHOICE_PATTERNS,
}


_BACKEND_WORDING_PATTERNS = (
    r"\bc[- ]prefixed\b",
    r"\binternal (?:identifier|schema|field id)\b",
    r"\bexact stored grid[- ]point coordinate\b",
    r"\bstored array name\b",
    r"\bvalidator\b",
    r"\bsemantic_contract_sha256\b",
)


_UNSUPPORTED_ASSERTION_PATTERNS = (
    r"\bthis (?:definitively |conclusively )?(?:proves|demonstrates|establishes) that\b",
    r"\bguarantees? that\b",
    r"\bis caused by\b",
    r"\bconfirms the mechanism\b",
)

# A case may report a count as a supporting observation, but a question whose
# only scientific request is to count/enumerate objects is outside the
# human-facing FlowIntentBench target policy.  This is deliberately a small
# presentation diagnostic; it does not infer Finding semantics or alter the
# O/F contracts.
_COUNT_ONLY_PATTERNS = (
    r"\bhow many\b",
    r"\b(?:the )?(?:number|count) of\b",
    r"\b(?:count|enumerate|list)\s+(?:all\s+)?(?:the\s+)?(?:regions?|components?|objects?|cells?|features?)\b",
)
_NON_COUNT_SCIENTIFIC_REQUEST_PATTERNS = (
    r"\bwhere\b|\blocation\b|\bposition\b",
    r"\b(?:speed|velocity|pressure|temperature|magnitude|strength|direction|gradient)\b",
    r"\b(?:trajectory|path|streamline|shape|surface|profile|extent|span)\b",
    r"\b(?:strongest|largest|weakest|compare|characteri[sz]e|describe|identify|select|report)\b",
)


def is_count_only_scientific_question(question: str) -> bool:
    """Return whether *question* has a pure count/enumeration objective.

    The check intentionally does not reject a physical question that also asks
    for a supporting count.  It only fires when a count marker is present and
    no non-count scientific property, relation, or representation is
    requested.
    """

    visible = _text(question)
    if not visible or not any(
        re.search(pattern, visible, re.IGNORECASE | re.DOTALL)
        for pattern in _COUNT_ONLY_PATTERNS
    ):
        return False
    return not any(
        re.search(pattern, visible, re.IGNORECASE | re.DOTALL)
        for pattern in _NON_COUNT_SCIENTIFIC_REQUEST_PATTERNS
    )


def audit_question_responsibility_realization(
    semantic_projection: Mapping[str, Any],
    model_visible_text: str,
    *,
    candidate: Mapping[str, Any] | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
    model_visible_context: str | None = None,
) -> dict[str, Any]:
    """Deterministically audit the structured and visible O/F realization.

    The checker deliberately uses only frozen structured contracts and compact
    maintainable lexical sentinels.  It supplements, rather than replaces, the
    independent Flow Expert semantic review.
    """

    projection = _projection(semantic_projection)
    candidate = dict(candidate or {})
    visible = _combined_visible_text(model_visible_text, model_visible_context)
    expected_fixed = _fixed_rows(
        projection.get("resolved_operationalization")
        if projection.get("resolved_operationalization") is not None
        else projection.get("fixed_operationalization")
    )
    expected_fixed_ids = _dimension_ids(expected_fixed)
    expected_open = _dimension_ids(
        projection.get("unresolved_operationalization_dimensions")
        if projection.get("unresolved_operationalization_dimensions") is not None
        else projection.get("open_operationalization")
    )
    expected_finding = _finding_mode(projection)

    actual_fixed = _fixed_rows(
        candidate.get("fixed_operationalization", expected_fixed)
    )
    actual_fixed_ids = _dimension_ids(actual_fixed)
    actual_open = _dimension_ids(
        candidate.get(
            "unresolved_operationalization_dimensions",
            candidate.get("open_operationalization", expected_open),
        )
    )
    actual_finding = _text(
        candidate.get("finding_responsibility", expected_finding)
    ).upper()
    failures: list[str] = []
    checks: dict[str, bool] = {}

    count_only = is_count_only_scientific_question(visible)
    if count_only:
        failures.append("COUNT_ONLY_SCIENTIFIC_TARGET")

    target = _text(projection.get("scientific_target"))
    checks["target_clarity"] = bool(target) and _target_is_clear(target, visible)
    if not checks["target_clarity"]:
        failures.append("TARGET_DRIFT")

    if set(actual_fixed_ids) != set(expected_fixed_ids):
        missing = set(expected_fixed_ids) - set(actual_fixed_ids)
        extra = set(actual_fixed_ids) - set(expected_fixed_ids)
        if missing:
            failures.append("MISSING_FIXED_O")
        if extra & set(expected_open):
            failures.append("OPEN_O_PREMATURELY_FIXED")
        elif extra:
            failures.append("UNDECLARED_RESPONDENT_CHOICE")
    if set(actual_open) != set(expected_open):
        if set(expected_open) - set(actual_open):
            failures.append("OPEN_O_PREMATURELY_FIXED")
        if set(actual_open) - set(expected_open):
            failures.append("UNDECLARED_RESPONDENT_CHOICE")
    if set(actual_fixed_ids) & set(actual_open):
        failures.append("FIXED_OPEN_O_CONTRADICTION")

    for fixed_row in expected_fixed:
        if not _fixed_clause_is_visible(fixed_row, visible):
            failures.append("MISSING_FIXED_O")
            failures.append(
                "MISSING_FIXED_O:" + _text(fixed_row.get("dimension_id"))
            )

    finding_semantics = projection.get("finding_semantics")
    roles = {
        _text(role)
        for role in (
            finding_semantics.get("required_finding_roles", ())
            if isinstance(finding_semantics, Mapping)
            else ()
        )
    }
    if isinstance(finding_semantics, Mapping):
        adequate = finding_semantics.get("adequate_core_semantics")
        if isinstance(adequate, Mapping):
            roles.update(
                _text(role) for role in adequate.get("mandatory_roles", ())
            )
    observable_choice_allowed = (
        "observable_selection" in set(expected_open)
        or "selected_field_identity" in roles
    )
    if not observable_choice_allowed and any(
        re.search(pattern, visible, re.I | re.S) for pattern in _FIELD_CHOICE_PATTERNS
    ):
        failures.append("UNDECLARED_RESPONDENT_CHOICE")
        failures.append("UNDECLARED_RESPONDENT_CHOICE:observable_selection")

    for dimension in expected_open:
        fixed_markers = _OPEN_DIMENSION_FIXED_MARKERS.get(dimension, ())
        choice_markers = _OPEN_DIMENSION_CHOICE_MARKERS.get(dimension, ())
        fixed_visible = any(
            re.search(pattern, visible, re.I | re.S) for pattern in fixed_markers
        )
        choice_visible = any(
            re.search(pattern, visible, re.I | re.S) for pattern in choice_markers
        )
        # A dimension may contain a shared phrase (e.g. a largest-area
        # selection) while its actual unresolved clause remains open.  Treat
        # an explicit choice cue as authoritative; shared fixed wording alone
        # is not premature closure.
        if fixed_visible and not choice_visible:
            failures.append("OPEN_O_PREMATURELY_FIXED")
            failures.append("OPEN_O_PREMATURELY_FIXED:" + dimension)
        if fixed_visible and choice_visible:
            failures.append("FIXED_OPEN_O_CONTRADICTION")
            failures.append("FIXED_OPEN_O_CONTRADICTION:" + dimension)

    if actual_finding != expected_finding:
        failures.append("FINDING_RESPONSIBILITY_DRIFT")
        if expected_finding == "F1":
            failures.append("F1_BECAME_OPEN")
        elif expected_finding == "F2":
            failures.append("F2_BECAME_HIDDEN_F1")
    if expected_finding == "F1" and re.search(
        r"\b(?:report anything|whatever findings|any findings you consider|anything important)\b",
        visible,
        re.I,
    ):
        failures.extend(("FINDING_RESPONSIBILITY_DRIFT", "F1_BECAME_OPEN"))
    if expected_finding == "F2":
        open_finding_language = re.search(
            r"\b(?:of your choice|chosen by you|choose (?:one|a)|scientifically relevant (?:findings?|characterization))\b",
            visible,
            re.I,
        )
        # A structured F2 contract can be explicit about mandatory roles and
        # alternatives while still leaving the respondent's Finding selection
        # open.  The renderer's "select and report ... supported by the data"
        # wording is therefore sufficient; only an explicit exhaustive list
        # or an absence of any open-selection cue is rejected.
        structured_open = (
            isinstance(candidate.get("requested_findings"), Mapping)
            and str(candidate["requested_findings"].get("finding_mode", "")).upper() == "F2"
        )
        if not (open_finding_language or structured_open) or re.search(
            r"\b(?:exactly these findings|report all of the following|no other findings)\b",
            visible,
            re.I,
        ):
            failures.extend(("FINDING_RESPONSIBILITY_DRIFT", "F2_BECAME_HIDDEN_F1"))

    visibility = semantic_visibility or projection.get("semantic_visibility")
    visibility_failures: list[str] = []
    if isinstance(visibility, Mapping):
        for item in visibility.get("items", ()):
            if not isinstance(item, Mapping):
                continue
            role = _text(item.get("semantic_role"))
            placement = _text(item.get("visibility"))
            if role in {
                "SCIENTIFIC_TARGET",
                "SCIENTIFIC_SCOPE",
                "FIXED_OPERATIONALIZATION",
                "UNRESOLVED_OPERATIONALIZATION",
                "FINDING_RESPONSIBILITY",
            } and placement == "BACKEND_ONLY":
                visibility_failures.append(_text(item.get("information_id")))
    if visibility_failures:
        failures.append("REQUIRED_INFORMATION_BACKEND_ONLY")

    if any(re.search(pattern, visible, re.I) for pattern in _BACKEND_WORDING_PATTERNS):
        failures.append("BACKEND_SCHEMA_LEAKAGE")
    if any(
        re.search(pattern, visible, re.I) for pattern in _UNSUPPORTED_ASSERTION_PATTERNS
    ):
        failures.append("UNSUPPORTED_PHYSICAL_ASSERTION")

    unique = sorted(set(failures))
    checks.update(
        {
            "fixed_o_exact": set(actual_fixed_ids) == set(expected_fixed_ids),
            "open_o_exact": set(actual_open) == set(expected_open),
            "finding_mode_exact": actual_finding == expected_finding,
            "visibility_integrity": not visibility_failures,
            "no_undeclared_respondent_choice": not any(
                item.startswith("UNDECLARED_RESPONDENT_CHOICE") for item in unique
            ),
            "fixed_o_visible": not any(
                item.startswith("MISSING_FIXED_O") for item in unique
            ),
            "open_o_remains_open": not any(
                item.startswith("OPEN_O_PREMATURELY_FIXED") for item in unique
            ),
            "no_fixed_open_contradiction": not any(
                item.startswith("FIXED_OPEN_O_CONTRADICTION") for item in unique
            ),
            "finding_responsibility_exact": not any(
                item in {"FINDING_RESPONSIBILITY_DRIFT", "F1_BECAME_OPEN", "F2_BECAME_HIDDEN_F1"}
                for item in unique
            ),
            "non_count_scientific_target": not count_only,
        }
    )
    return {
        "status": "PASS" if not unique else "REVISE",
        "failure_codes": unique,
        "checks": checks,
        "expected_fixed_dimensions": sorted(expected_fixed_ids),
        "expected_open_dimensions": sorted(expected_open),
        "expected_finding_mode": expected_finding,
        "visible_text_sha256": canonical_json_sha256({"text": visible}),
        "scientific_target_validity_judged": False,
        "core_membership_affected": False,
    }


def _mutated_candidate(
    projection: Mapping[str, Any],
    *,
    fixed: Sequence[Mapping[str, Any]] | None = None,
    opened: Sequence[Any] | None = None,
    finding: str | None = None,
) -> dict[str, Any]:
    expected_fixed = _fixed_rows(projection.get("resolved_operationalization"))
    expected_open = projection.get("unresolved_operationalization_dimensions", [])
    return {
        "fixed_operationalization": copy.deepcopy(
            list(fixed) if fixed is not None else expected_fixed
        ),
        "unresolved_operationalization_dimensions": copy.deepcopy(
            list(opened) if opened is not None else expected_open
        ),
        "finding_responsibility": finding or _finding_mode(projection),
    }


def run_validator_sentinel_challenges(
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the frozen M1–M12 and P1–P5 validator challenge corpus."""

    by_condition = {
        (_text(row.get("dataset_id")), _text(row.get("condition"))): row
        for row in source_rows
    }
    high_o1 = next(
        row
        for row in source_rows
        if _text(row.get("condition")) == "O1-F1"
        and "high-speed" in _text(_projection(row).get("scientific_target")).casefold()
    )
    high_o2 = next(
        row
        for row in source_rows
        if _text(row.get("condition")) == "O2-F1"
        and "high-speed" in _text(_projection(row).get("scientific_target")).casefold()
    )
    f2 = next(row for row in source_rows if _text(row.get("condition")) == "O1-F2")
    p1 = _projection(high_o1)
    p2 = _projection(high_o2)
    pf2 = _projection(f2)

    o1_text = (
        "Identify the strongest high-speed region. Define high-speed locations at the "
        "90th percentile, group face-connected locations, use peak speed for strength, "
        "represent the region by its mean spatial location, and report location and strength."
    )
    o2_text = (
        "Identify the strongest high-speed region. Treat face-connected locations as one "
        "region and use their mean spatial location. Choose the high-speed criterion and "
        "strength measure, then report the region's location and strength."
    )
    f2_text = (
        "Identify the strongest high-speed region using the fixed 90th-percentile, "
        "face-connected, peak-speed, mean-location analysis. Report its strength and location, "
        "plus one scientifically relevant characterization of your choice."
    )

    semantic_cases: list[tuple[str, dict[str, Any], set[str]]] = []
    semantic_cases.append(
        (
            "M1",
            audit_question_responsibility_realization(
                p2,
                o2_text + " Choose which field represents velocity.",
                candidate=_mutated_candidate(p2),
            ),
            {"UNDECLARED_RESPONDENT_CHOICE"},
        )
    )
    moved_open = list(_dimension_ids(p2.get("unresolved_operationalization_dimensions")))
    prematurely_fixed = [
        *_fixed_rows(p2.get("resolved_operationalization")),
        {"dimension_id": moved_open[0], "canonical_id": "sentinel-fixed-open"},
    ]
    semantic_cases.append(
        (
            "M2",
            audit_question_responsibility_realization(
                p2,
                o2_text,
                candidate=_mutated_candidate(
                    p2,
                    fixed=prematurely_fixed,
                    opened=moved_open[1:],
                ),
            ),
            {"OPEN_O_PREMATURELY_FIXED"},
        )
    )
    semantic_cases.append(
        (
            "M3",
            audit_question_responsibility_realization(
                p1,
                o1_text,
                candidate=_mutated_candidate(
                    p1, fixed=_fixed_rows(p1.get("resolved_operationalization"))[1:]
                ),
            ),
            {"MISSING_FIXED_O"},
        )
    )
    overlap_open = _dimension_ids(p2.get("unresolved_operationalization_dimensions"))
    overlap_fixed = [
        *_fixed_rows(p2.get("resolved_operationalization")),
        {"dimension_id": overlap_open[0], "canonical_id": "sentinel-overlap"},
    ]
    semantic_cases.append(
        (
            "M4",
            audit_question_responsibility_realization(
                p2,
                o2_text + " Use the 90th percentile but also choose the criterion.",
                candidate=_mutated_candidate(p2, fixed=overlap_fixed, opened=overlap_open),
            ),
            {"FIXED_OPEN_O_CONTRADICTION"},
        )
    )
    semantic_cases.append(
        (
            "M5",
            audit_question_responsibility_realization(
                p1,
                o1_text.replace("high-speed", "generic flow"),
                candidate=_mutated_candidate(p1),
            ),
            {"TARGET_DRIFT"},
        )
    )
    semantic_cases.append(
        (
            "M6",
            audit_question_responsibility_realization(
                pf2,
                f2_text.replace(
                    "plus one scientifically relevant characterization of your choice",
                    "and exactly these findings with no other findings",
                ),
                candidate=_mutated_candidate(pf2, finding="F1"),
            ),
            {"F2_BECAME_HIDDEN_F1"},
        )
    )
    semantic_cases.append(
        (
            "M7",
            audit_question_responsibility_realization(
                p1,
                "Identify the strongest high-speed region and report anything important.",
                candidate=_mutated_candidate(p1, finding="F2"),
            ),
            {"F1_BECAME_OPEN"},
        )
    )
    semantic_cases.append(
        (
            "M9",
            audit_question_responsibility_realization(
                p2,
                o2_text,
                candidate=_mutated_candidate(p2),
                semantic_visibility={
                    "items": [
                        {
                            "information_id": "unresolved_o:criterion",
                            "semantic_role": "UNRESOLVED_OPERATIONALIZATION",
                            "visibility": "BACKEND_ONLY",
                        }
                    ]
                },
            ),
            {"REQUIRED_INFORMATION_BACKEND_ONLY"},
        )
    )
    semantic_cases.append(
        (
            "M10",
            audit_question_responsibility_realization(
                p1,
                o1_text + " Copy the exact stored grid-point coordinate and semantic_contract_sha256.",
                candidate=_mutated_candidate(p1),
            ),
            {"BACKEND_SCHEMA_LEAKAGE"},
        )
    )
    semantic_cases.append(
        (
            "M12",
            audit_question_responsibility_realization(
                p1,
                o1_text + " This conclusively proves that the structure is caused by separation.",
                candidate=_mutated_candidate(p1),
            ),
            {"UNSUPPORTED_PHYSICAL_ASSERTION"},
        )
    )

    # M8 is a family-level Effective-O mutation.
    o1_family = by_condition[
        (_text(f2.get("dataset_id")), "O1-F1")
    ]
    mutated_f2 = copy.deepcopy(dict(f2))
    mutated_projection = copy.deepcopy(dict(_projection(mutated_f2)))
    fixed_rows = _fixed_rows(mutated_projection.get("resolved_operationalization"))
    fixed_rows[0]["canonical_id"] = fixed_rows[0].get("canonical_id", "") + "-mutated"
    mutated_projection["resolved_operationalization"] = fixed_rows
    m8_fail = canonical_json_sha256(
        {
            "target": _projection(o1_family).get("scientific_target"),
            "scope": _projection(o1_family).get("scientific_scope"),
            "fixed": _fixed_rows(_projection(o1_family).get("resolved_operationalization")),
            "open": _dimension_ids(
                _projection(o1_family).get("unresolved_operationalization_dimensions")
            ),
        }
    ) != canonical_json_sha256(
        {
            "target": mutated_projection.get("scientific_target"),
            "scope": mutated_projection.get("scientific_scope"),
            "fixed": _fixed_rows(mutated_projection.get("resolved_operationalization")),
            "open": _dimension_ids(
                mutated_projection.get("unresolved_operationalization_dimensions")
            ),
        }
    )

    near_template = {
        condition: {
            "model_visible_text": (
                "Identify the strongest high-speed region and report its location and strength "
                + condition
            )
        }
        for condition in REQUIRED_CONDITIONS
    }
    m11_audit = audit_family_presentation_leakage(near_template)

    semantic_results: list[dict[str, Any]] = []
    for challenge_id, audit, expected in semantic_cases:
        observed = set(audit.get("failure_codes", ()))
        accepted = audit.get("status") == "PASS"
        expected_seen = any(
            any(code == wanted or code.startswith(wanted + ":") for code in observed)
            for wanted in expected
        )
        semantic_results.append(
            {
                "challenge_id": challenge_id,
                "expected": "FAIL",
                "status": "PASS" if not accepted and expected_seen else "FAIL",
                "expected_failure_codes": sorted(expected),
                "observed_failure_codes": sorted(observed),
                "validator_accepted": accepted,
            }
        )
    semantic_results.extend(
        [
            {
                "challenge_id": "M8",
                "expected": "FAIL",
                "status": "PASS" if m8_fail else "FAIL",
                "expected_failure_codes": ["O1_F2_EFFECTIVE_O_DRIFT"],
                "observed_failure_codes": ["O1_F2_EFFECTIVE_O_DRIFT"] if m8_fail else [],
                "validator_accepted": not m8_fail,
            },
            {
                "challenge_id": "M11",
                "expected": "FAIL",
                "status": (
                    "PASS"
                    if m11_audit.get("status") == "PRESENTATION_LEAKAGE_OBSERVED"
                    else "FAIL"
                ),
                "expected_failure_codes": ["FAMILY_TEMPLATE_LEAKAGE"],
                "observed_failure_codes": (
                    ["FAMILY_TEMPLATE_LEAKAGE"]
                    if m11_audit.get("status") == "PRESENTATION_LEAKAGE_OBSERVED"
                    else []
                ),
                "validator_accepted": m11_audit.get("status") == "PASS",
            },
        ]
    )

    preserving_texts = {
        "P1": o1_text.replace("Identify", "Please identify"),
        "P2": o1_text.replace(
            "Define high-speed locations at the 90th percentile, ", ""
        ),
        "P3": o1_text.replace("mean spatial location", "centroid"),
        "P4": o1_text.replace("region. Define", "region; define"),
        "P5": o1_text.replace(
            "report location and strength", "report strength and location"
        ),
    }
    preserving_contexts = {
        "P2": "High-speed locations are defined at the 90th percentile.",
    }
    preserving_results: list[dict[str, Any]] = []
    for challenge_id, text in preserving_texts.items():
        audit = audit_question_responsibility_realization(
            p1,
            text,
            candidate=_mutated_candidate(p1),
            model_visible_context=preserving_contexts.get(challenge_id),
        )
        preserving_results.append(
            {
                "challenge_id": challenge_id,
                "expected": "PASS",
                "status": "PASS" if audit.get("status") == "PASS" else "FAIL",
                "observed_failure_codes": list(audit.get("failure_codes", ())),
                "validator_accepted": audit.get("status") == "PASS",
            }
        )

    false_accepts = sum(item["status"] != "PASS" for item in semantic_results)
    false_rejects = sum(item["status"] != "PASS" for item in preserving_results)
    return {
        "status": "PASS" if false_accepts == 0 and false_rejects == 0 else "FAIL",
        "contract_sha256": human_facing_acceptance_contract_sha256(),
        "semantic_challenges": semantic_results,
        "preserving_challenges": preserving_results,
        "VALIDATOR_SEMANTIC_CHALLENGE_FALSE_ACCEPTS": false_accepts,
        "VALIDATOR_PRESERVING_CHALLENGE_FALSE_REJECTS": false_rejects,
    }


def _hccq_finding_contract(projection: Mapping[str, Any]) -> dict[str, Any]:
    mode = _finding_mode(projection)
    semantics = projection.get("finding_semantics")
    semantics = dict(semantics) if isinstance(semantics, Mapping) else {}
    if mode == "F1":
        requirements = []
        for item in semantics.get("fixed_finding_requirements", ()) or ():
            if isinstance(item, Mapping):
                statement = _text(
                    item.get("normalized_meaning")
                    or item.get("statement")
                    or item.get("category")
                )
            else:
                statement = _text(item)
            if statement:
                requirements.append(statement)
        if not requirements:
            requirements = [
                _text(item)
                for item in semantics.get("required_finding_roles", ()) or ()
                if _text(item)
            ]
        return {
            "finding_mode": "F1",
            "fixed_requirements": requirements,
            "mandatory_roles": [],
            "alternative_role_groups": [],
            "adequate_core_sets": [],
        }

    adequate = semantics.get("adequate_core_semantics")
    adequate = dict(adequate) if isinstance(adequate, Mapping) else {}
    return {
        "finding_mode": "F2",
        "fixed_requirements": [],
        "mandatory_roles": list(adequate.get("mandatory_roles", ()) or ()),
        "alternative_role_groups": copy.deepcopy(
            list(adequate.get("alternative_role_groups", ()) or ())
        ),
        "adequate_core_sets": copy.deepcopy(
            list(adequate.get("adequate_core_sets", ()) or ())
        ),
    }


def audit_actual_hccq(
    question: str,
    visible_context: str | None = None,
    *,
    semantic_projection: Mapping[str, Any] | None = None,
    semantic_visibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute HCCQ from current text plus its structured O/F contract.

    HCCQ's responsibility checks intentionally require structured fixed, open,
    and Finding declarations.  Supplying text alone makes every responsibility
    appear absent, so the closure runner must rebuild this presentation shape
    rather than trusting a stale stored HCCQ boolean.
    """

    presentation: dict[str, Any] = {
        "model_visible_text": _combined_visible_text(question, visible_context)
    }
    if semantic_projection is not None:
        projection = _projection(semantic_projection)
        visibility = {}
        if isinstance(semantic_visibility, Mapping):
            visibility = {
                _text(item.get("information_id")): _text(item.get("visibility"))
                for item in semantic_visibility.get("items", ()) or ()
                if isinstance(item, Mapping)
            }
        presentation["analysis_constraints"] = [
            {
                "dimension_id": _text(item.get("dimension_id")),
                "statement": _text(
                    item.get("normalized_meaning") or item.get("statement")
                ),
            }
            for item in _fixed_rows(
                projection.get("resolved_operationalization")
                if projection.get("resolved_operationalization") is not None
                else projection.get("fixed_operationalization")
            )
            if visibility.get(
                f"fixed_o:{_text(item.get('dimension_id'))}", "QUESTION_VISIBLE"
            )
            == "QUESTION_VISIBLE"
        ]
        presentation["open_analysis_choices"] = _dimension_ids(
            projection.get("unresolved_operationalization_dimensions")
            if projection.get("unresolved_operationalization_dimensions") is not None
            else projection.get("open_operationalization")
        )
        presentation["requested_findings"] = _hccq_finding_contract(projection)
    try:
        return audit_human_presentation(presentation)
    except ValueError as exc:
        return {
            "status": "REVISE_HUMAN_INTERFACE",
            "backend_schema_language_leakage": [str(exc)],
            "unnecessary_mechanical_burden": [],
            "protocol_wording_signals": [],
            "responsibility_clarity_signals": [],
        }


__all__ = [
    "HUMAN_FACING_ACCEPTANCE_CONTRACT",
    "HUMAN_FACING_ACCEPTANCE_CONTRACT_VERSION",
    "REQUIRED_CONDITIONS",
    "audit_actual_hccq",
    "audit_question_responsibility_realization",
    "canonical_json_sha256",
    "human_facing_acceptance_contract_sha256",
    "is_count_only_scientific_question",
    "run_validator_sentinel_challenges",
]
