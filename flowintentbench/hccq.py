"""Optional Human-Comparable Case Qualification (HCCQ).

HCCQ evaluates presentation and human-comparability only. It cannot decide a
scientific target, valid O, Ground Truth, evaluation-contract content, release
eligibility, or core benchmark membership.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Mapping, Sequence


HCCQ_OUTCOMES = frozenset(
    {
        "HUMAN_COMPARABLE",
        "REVISE_HUMAN_INTERFACE",
        "NOT_HUMAN_COMPARABLE_CURRENT_INTERFACE",
    }
)

_FORBIDDEN_SCIENTIFIC_KEYS = (
    "accepted_o",
    "benchmark_membership",
    "curator_decision",
    "g_of_o",
    "ground_truth",
    "reference_answer",
    "reference_finding",
    "release_eligible",
    "scientific_evaluation_contract",
    "scientific_target_validity",
    "valid_operationalization",
)

_BACKEND_PATTERNS = (
    ("C_PREFIXED_SCHEMA_LANGUAGE", r"\bc[- ]?prefixed\b"),
    (
        "STORED_ARRAY_IDENTIFIER_LANGUAGE",
        r"\b(?:stored\s+)?(?:array|field)\s+(?:identifier|id|name)\b",
    ),
    ("BACKEND_IDENTIFIER_LANGUAGE", r"\bbackend\s+(?:identifier|id|schema)\b"),
    ("VTK_SCHEMA_LANGUAGE", r"\bvtk(?:dataarray|pointdata|celldata|dataset)?\b"),
    (
        "EXACT_GRID_STORAGE_LANGUAGE",
        r"\bexact\s+stored\s+grid[- ]point\s+coordinate\b",
    ),
    (
        "IMPLEMENTATION_HANDLER_LANGUAGE",
        r"\b(?:handler_id|extractor_id|canonical_id|array_index)\b",
    ),
    (
        "POINT_OR_CELL_DATA_SCHEMA_LANGUAGE",
        r"\b(?:point|cell)[ _-]?(?:data|array)\b|\bdataarray\b",
    ),
    (
        "RAW_STORAGE_LAYOUT_LANGUAGE",
        r"\b(?:raw|stored)\s+(?:tuple|component|array|index)\b",
    ),
    (
        "SERIALIZATION_SCHEMA_LANGUAGE",
        r"\b(?:json|schema)\s+(?:field|key|property|name)\b",
    ),
    (
        "STORED_GRID_ENTITY_LANGUAGE",
        r"\bstored\s+grid[- ]?(?:point|cell)\b",
    ),
    (
        "MACHINE_KEY_LANGUAGE",
        r"\b(?:machine[- ]readable|internal)\s+(?:identifier|id|key|field)\b",
    ),
)

_MECHANICAL_PATTERNS = (
    ("MANUAL_TRANSCRIPTION_BURDEN", r"\b(?:manually\s+)?(?:copy|transcribe)\b"),
    (
        "EXHAUSTIVE_COORDINATE_COPY_BURDEN",
        r"\b(?:list|copy|reproduce)\s+(?:every|all|the exact)\s+coordinates?\b",
    ),
    (
        "BACKEND_IDENTIFIER_COPY_BURDEN",
        r"\b(?:copy|reproduce|transcribe)\s+(?:the\s+)?(?:array|field|backend)\s+(?:identifier|id|name)\b",
    ),
    (
        "MANUAL_CALCULATION_BURDEN",
        r"\b(?:(?:manually|by hand)\s+(?:calculate|compute|derive|evaluate)|"
        r"(?:calculate|compute|derive|evaluate)\s+(?:every|each|all)\b)",
    ),
    (
        "EXACT_COORDINATE_ENTRY_BURDEN",
        r"\b(?:copy|enter|provide|record|report|reproduce|transcribe|type|write\s+down)\b"
        r"[^.\n]{0,60}\b(?:exact|raw|stored)\b[^.\n]{0,40}"
        r"\b(?:coordinate|grid[- ]?point|index|location)\b",
    ),
    (
        "IDENTIFIER_ENTRY_BURDEN",
        r"\b(?:copy|enter|provide|record|report|reproduce|transcribe|type)\b"
        r"[^.\n]{0,60}\b(?:array|backend|field|internal|machine[- ]readable)\b"
        r"[^.\n]{0,30}\b(?:identifier|id|key|name)\b",
    ),
    (
        "EXHAUSTIVE_SAMPLE_REPORTING_BURDEN",
        r"\b(?:enumerate|list|report|return)\s+(?:each|every|all)\s+"
        r"(?:cell|coordinate|grid[- ]?point|index|sample|tuple|value)s?\b",
    ),
)

_REQUEST_CUE = re.compile(
    r"(?:\?|\b(?:analy[sz]e|assess|characteri[sz]e|compare|describe|determine|"
    r"estimate|evaluate|examine|find|identify|investigate|locate|quantify|report|"
    r"show|study|what|where|which|how)\b)",
    re.I,
)

_FIXED_RESPONSIBILITY_CUE = re.compile(
    r"\b(?:analysis constraints?|according to|based on|classify|compare|define|"
    r"defined by|fixed|held constant|maximi[sz](?:e|ed|ing)|measure|prescribed|"
    r"rank|regard|represent|retain|select(?:ed|ing)? by|set|threshold|treat|"
    r"treating|use|using)\b",
    re.I,
)

_OPEN_RESPONSIBILITY_CUE = re.compile(
    r"\b(?:analysis choices?|appropriate|choose|choice|decide|defensible|open|"
    r"select|suitable|your (?:analysis|choice|method|operationalization))\b",
    re.I,
)

_REPORTING_CUE = re.compile(
    r"\b(?:characteri[sz]e|compare|describe|finding|give|identify|include|locate|"
    r"provide|quantify|report|return|state|summari[sz]e)\b",
    re.I,
)

_NON_SUBSTANTIVE_CLAUSES = frozenset(
    {
        "analysis constraints",
        "open analysis choices",
        "requested findings",
        "scientific question",
        "scientific scope",
    }
)

_CONDITION_TEMPLATE_MARKERS = {
    "O1-F1": (
        r"\bfor this analysis, define\b",
        r"\buse the following fixed (?:definition|procedure|constraints?)\b",
    ),
    "O2-F1": (
        r"\busing an appropriate\b",
        r"\bchoose an appropriate\b",
    ),
    "O3-F1": (
        r"\busing a scientifically defensible\b",
        r"\bchoose a scientifically defensible\b",
    ),
    "O1-F2": (
        r"\bany other scientifically relevant findings?\b",
        r"\bwhatever findings? you (?:consider|deem) relevant\b",
    ),
}


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _presentation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        preferred = value.get("model_visible_text")
        if isinstance(preferred, str) and preferred.strip():
            return preferred
        return "\n".join(
            str(value.get(key, ""))
            for key in ("scientific_question", "scientific_scope")
            if str(value.get(key, "")).strip()
        )
    return str(value or "")


def _forbidden_paths(value: Any, path: str = "presentation") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            normalized = str(key).strip().casefold()
            if any(fragment in normalized for fragment in _FORBIDDEN_SCIENTIFIC_KEYS):
                hits.append(child)
            hits.extend(_forbidden_paths(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            hits.extend(_forbidden_paths(item, f"{path}[{index}]"))
    return hits


def _pattern_hits(text: str, patterns: Sequence[tuple[str, str]]) -> list[str]:
    return [code for code, pattern in patterns if re.search(pattern, text, re.I)]


def _readability_audit(text: str) -> tuple[bool, list[str], dict[str, int | float]]:
    words = re.findall(r"\b[\w'-]+\b", text)
    segments = [
        re.findall(r"\b[\w'-]+\b", segment)
        for segment in re.split(r"[.!?\n]+", text)
        if segment.strip()
    ]
    substantive = [segment for segment in segments if len(segment) >= 3]
    max_segment_words = max((len(segment) for segment in substantive), default=0)
    alpha_count = len(re.findall(r"[A-Za-z]", text))
    nonspace_count = len(re.findall(r"\S", text))
    alpha_ratio = alpha_count / nonspace_count if nonspace_count else 0.0
    normalized_segments = [" ".join(word.casefold() for word in row) for row in substantive]

    signals: list[str] = []
    if not words:
        signals.append("EMPTY_PRESENTATION")
    if len(words) > 350:
        signals.append("EXCESSIVE_TOTAL_LENGTH")
    if max_segment_words > 90:
        signals.append("EXCESSIVE_SENTENCE_LENGTH")
    if nonspace_count and alpha_ratio < 0.45:
        signals.append("LOW_NATURAL_LANGUAGE_CONTENT")
    if words and not _REQUEST_CUE.search(text):
        signals.append("NO_CLEAR_SCIENTIFIC_REQUEST_CUE")
    if normalized_segments and len(normalized_segments) - len(set(normalized_segments)) >= 2:
        signals.append("REPETITIVE_PRESENTATION")

    return (
        not signals,
        signals,
        {
            "word_count": len(words),
            "max_sentence_or_line_word_count": max_segment_words,
            "alphabetic_character_ratio": round(alpha_ratio, 4),
        },
    )


def _protocol_signals(text: str) -> list[str]:
    signals: list[str] = []
    if re.search(r"\b(?:step\s*[1-9]|protocol)\b", text, re.I):
        signals.append("STEPWISE_PROTOCOL_LANGUAGE")
    if re.search(
        r"\b(?:follow (?:these instructions|exactly)|must exactly|strictly follow)\b",
        text,
        re.I,
    ):
        signals.append("STRICT_COMPLIANCE_LANGUAGE")
    if len(re.findall(r"\bmust\b", text, re.I)) >= 4:
        signals.append("EXCESSIVE_MANDATORY_CLAUSES")
    if len(re.findall(r"^\s*[-*]\s+", text, re.M)) > 10:
        signals.append("EXCESSIVE_BULLET_PROTOCOL")
    sequence_markers = len(
        re.findall(r"\b(?:first|second|third|next|then|finally)\b", text, re.I)
    )
    if sequence_markers >= 4:
        signals.append("PROCEDURAL_SEQUENCE_LANGUAGE")
    if re.search(
        r"\b(?:return|output|respond with)\s+(?:a\s+)?(?:json|yaml|csv|xml)\b",
        text,
        re.I,
    ):
        signals.append("OUTPUT_SERIALIZATION_PROTOCOL")
    return signals


def _meaningful_item(value: Any, *, keys: Sequence[str]) -> bool:
    if isinstance(value, Mapping):
        return any(str(value.get(key, "")).strip() for key in keys)
    return bool(str(value or "").strip())


def _responsibility_clarity(
    presentation: Any, text: str
) -> tuple[dict[str, bool | None], list[str]]:
    if not isinstance(presentation, Mapping):
        return {
            "benchmark_fixed_choices_clear": None,
            "respondent_choices_clear": None,
            "requested_reporting_clear": None,
        }, []

    signals: list[str] = []
    constraints = presentation.get("analysis_constraints")
    fixed_clear = isinstance(constraints, list)
    if fixed_clear:
        fixed_clear = all(
            _meaningful_item(item, keys=("statement", "description", "meaning"))
            for item in constraints
        )
        dimensions = [
            str(item.get("dimension_id", item.get("dimension", ""))).strip()
            for item in constraints
            if isinstance(item, Mapping)
        ]
        fixed_clear = fixed_clear and len(dimensions) == len(set(dimensions))
        if constraints:
            fixed_clear = fixed_clear and bool(_FIXED_RESPONSIBILITY_CUE.search(text))
    if not fixed_clear:
        signals.append("BENCHMARK_FIXED_CHOICES_UNCLEAR")

    choices = presentation.get("open_analysis_choices")
    open_clear = isinstance(choices, list)
    if open_clear:
        open_clear = all(
            _meaningful_item(item, keys=("dimension_id", "dimension", "statement"))
            for item in choices
        )
        if choices:
            open_clear = open_clear and bool(_OPEN_RESPONSIBILITY_CUE.search(text))
    if not open_clear:
        signals.append("RESPONDENT_SELECTED_CHOICES_UNCLEAR")

    findings = presentation.get("requested_findings")
    reporting_clear = isinstance(findings, Mapping)
    if reporting_clear:
        mode = str(findings.get("finding_mode", "")).upper()
        if mode == "F1":
            requirements = findings.get("fixed_requirements")
            reporting_clear = isinstance(requirements, list) and bool(requirements)
            if reporting_clear:
                reporting_clear = all(
                    _meaningful_item(item, keys=("statement", "description", "role"))
                    for item in requirements
                )
        elif mode == "F2":
            mandatory = findings.get("mandatory_roles")
            alternatives = findings.get("alternative_role_groups")
            adequate = findings.get("adequate_core_sets")
            reporting_clear = (
                isinstance(mandatory, list)
                and isinstance(alternatives, list)
                and isinstance(adequate, list)
                and bool(mandatory or alternatives or adequate)
            )
        else:
            reporting_clear = False
        reporting_clear = reporting_clear and bool(_REPORTING_CUE.search(text))
    if not reporting_clear:
        signals.append("REQUESTED_FINDINGS_UNCLEAR")

    return {
        "benchmark_fixed_choices_clear": fixed_clear,
        "respondent_choices_clear": open_clear,
        "requested_reporting_clear": reporting_clear,
    }, signals


def _surface_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _surface_clauses(text: str) -> list[list[str]]:
    clauses: list[list[str]] = []
    for raw in re.split(r"[.!?;\n]+|\b(?:and then|then|and)\b", text, flags=re.I):
        tokens = _surface_tokens(re.sub(r"^\s*[-*\d.)]+\s*", "", raw))
        normalized = " ".join(tokens)
        if len(tokens) >= 2 and normalized not in _NON_SUBSTANTIVE_CLAUSES:
            clauses.append(tokens)
    return clauses


def _pairwise_surface_similarity(left: str, right: str) -> dict[str, Any]:
    left_tokens = _surface_tokens(left)
    right_tokens = _surface_tokens(right)
    sequence_similarity = SequenceMatcher(None, left_tokens, right_tokens).ratio()
    left_set, right_set = set(left_tokens), set(right_tokens)
    token_jaccard = (
        len(left_set & right_set) / len(left_set | right_set)
        if left_set or right_set
        else 1.0
    )

    left_clauses = _surface_clauses(left)
    right_clauses = _surface_clauses(right)
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_clause in enumerate(left_clauses):
        for right_index, right_clause in enumerate(right_clauses):
            ratio = SequenceMatcher(None, left_clause, right_clause).ratio()
            if ratio >= 0.86:
                candidates.append((ratio, left_index, right_index))
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    for _, left_index, right_index in sorted(candidates, reverse=True):
        if left_index not in matched_left and right_index not in matched_right:
            matched_left.add(left_index)
            matched_right.add(right_index)
    shared_clause_count = len(matched_left)
    clause_overlap = (
        2 * shared_clause_count / (len(left_clauses) + len(right_clauses))
        if left_clauses or right_clauses
        else 1.0
    )
    minimum_words = min(len(left_tokens), len(right_tokens))
    near_identical = minimum_words >= 8 and (
        (sequence_similarity >= 0.84 and token_jaccard >= 0.68)
        or (shared_clause_count >= 2 and clause_overlap >= 0.55)
    )
    return {
        "sequence_similarity": round(sequence_similarity, 4),
        "token_jaccard": round(token_jaccard, 4),
        "clause_overlap": round(clause_overlap, 4),
        "shared_clause_count": shared_clause_count,
        "near_identical": near_identical,
    }


def audit_human_presentation(presentation: Any) -> dict[str, Any]:
    """Audit human-comparability signals without scientific qualification."""

    forbidden = _forbidden_paths(presentation)
    if forbidden:
        raise ValueError(
            "HCCQ presentation crosses the scientific firewall: " + ", ".join(forbidden)
        )
    text = _presentation_text(presentation)
    backend = _pattern_hits(text, _BACKEND_PATTERNS)
    mechanical = _pattern_hits(text, _MECHANICAL_PATTERNS)
    readability, readability_signals, readability_metrics = _readability_audit(text)
    protocol_signals = _protocol_signals(text)
    excessive_protocol = bool(protocol_signals)
    responsibility_clarity, responsibility_signals = _responsibility_clarity(
        presentation, text
    )
    responsibilities_clear = not any(
        value is False for value in responsibility_clarity.values()
    )
    return {
        "status": (
            "PASS"
            if readability
            and not backend
            and not mechanical
            and not excessive_protocol
            and responsibilities_clear
            else "REVISE_HUMAN_INTERFACE"
        ),
        "readability": readability,
        "natural_readability_signals": readability_signals,
        "readability_metrics": readability_metrics,
        "backend_schema_language_leakage": backend,
        "unnecessary_mechanical_burden": mechanical,
        "excessive_protocol_like_wording": excessive_protocol,
        "protocol_wording_signals": protocol_signals,
        "responsibility_clarity": responsibility_clarity,
        "responsibility_clarity_signals": responsibility_signals,
        "presentation_sha256": _digest(presentation),
        "scientific_target_validity_judged": False,
        "scientific_qualification_performed": False,
        "core_membership_affected": False,
        "firewall_status": "PASS",
    }


def audit_family_presentation_leakage(
    presentations: Mapping[str, Any],
    *, expected_conditions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Detect obvious condition-specific surface templates within one family."""

    allowed = {"O1-F1", "O2-F1", "O3-F1", "O1-F2"}
    required = allowed if expected_conditions is None else {str(item).upper().replace("_", "-") for item in expected_conditions}
    if not required or not required <= allowed:
        raise ValueError("invalid expected family conditions")
    supplied = {str(key).upper() for key in presentations}
    if supplied != required:
        return {
            "status": "INCOMPLETE",
            "condition_identity_guessable_from_surface_templates": None,
            "missing_conditions": sorted(required - supplied),
            "unexpected_conditions": sorted(supplied - required),
            "leakage_signals": [],
            "similarity_signals": [],
            "pairwise_similarity": [],
            "scientific_qualification_performed": False,
            "scientific_target_validity_judged": False,
            "core_membership_affected": False,
            "firewall_status": "PASS",
        }
    normalized = {str(key).upper(): value for key, value in presentations.items()}
    for condition, value in normalized.items():
        forbidden = _forbidden_paths(value, f"presentations.{condition}")
        if forbidden:
            raise ValueError(
                "HCCQ family presentation crosses the scientific firewall: "
                + ", ".join(forbidden)
            )
    signals: list[dict[str, Any]] = []
    for condition, value in normalized.items():
        text = _presentation_text(value).casefold()
        if re.search(r"\b(?:o[123][-_ ]?f[12]|condition\s+o[123])\b", text, re.I):
            signals.append({"condition": condition, "code": "CONDITION_LABEL_VISIBLE"})
        for pattern in _CONDITION_TEMPLATE_MARKERS[condition]:
            if re.search(pattern, text, re.I):
                signals.append(
                    {
                        "condition": condition,
                        "code": "CONDITION_SPECIFIC_TEMPLATE_MARKER",
                        "pattern": pattern,
                    }
                )
    styles = {
        condition: str(value.get("style", ""))
        for condition, value in normalized.items()
        if isinstance(value, Mapping) and str(value.get("style", "")).strip()
    }
    style_coupling = len(styles) >= 3 and len(styles) == len(required) and len(set(styles.values())) == len(required)
    pairwise_similarity: list[dict[str, Any]] = []
    similarity_signals: list[dict[str, Any]] = []
    compared_conditions = tuple(condition for condition in ("O1-F1", "O2-F1", "O3-F1", "O1-F2") if condition in required)
    for left, right in combinations(compared_conditions, 2):
        similarity = _pairwise_surface_similarity(
            _presentation_text(normalized[left]),
            _presentation_text(normalized[right]),
        )
        row = {"conditions": [left, right], **similarity}
        pairwise_similarity.append(row)
        if similarity["near_identical"]:
            similarity_signals.append(
                {
                    **row,
                    "code": "NEAR_IDENTICAL_CONDITION_SURFACE",
                }
            )
    style_signals = (
        [
            {
                "code": "POTENTIAL_STYLE_CONDITION_COUPLING",
                "condition_styles": dict(styles),
            }
        ]
        if style_coupling
        else []
    )
    guessable = bool(signals or similarity_signals or style_signals)
    return {
        "status": "PRESENTATION_LEAKAGE_OBSERVED" if guessable else "PASS",
        "condition_identity_guessable_from_surface_templates": guessable,
        "leakage_signals": signals,
        "similarity_signals": similarity_signals,
        "pairwise_similarity": pairwise_similarity,
        "style_leakage_signals": style_signals,
        "potential_style_condition_coupling": style_coupling,
        "condition_styles": styles,
        "scientific_qualification_performed": False,
        "scientific_target_validity_judged": False,
        "core_membership_affected": False,
        "firewall_status": "PASS",
    }


@dataclass(frozen=True)
class HCCQRecord:
    case_id: str
    human_interface_profile_id: str
    human_interface_sha256: str
    human_backend_profile_id: str
    human_backend_sha256: str
    qualification: str
    auditor_id: str
    audit_version: str
    actionability: bool = True
    fairness: bool = True
    feasibility: str = "UNKNOWN"
    readability: bool = True
    responsibility_clarity: bool = True
    backend_schema_language_leakage: tuple[str, ...] = ()
    unnecessary_mechanical_burden: tuple[str, ...] = ()
    excessive_protocol_like_wording: bool = False
    presentation_audit_sha256: str = ""
    scientific_target_validity_judged: bool = False
    scientific_qualification_performed: bool = False
    core_membership_affected: bool = False
    firewall_status: str = "PASS"

    def __post_init__(self) -> None:
        if self.qualification not in HCCQ_OUTCOMES:
            raise ValueError("HCCQ cannot emit a scientific release rejection")
        if (
            self.scientific_target_validity_judged
            or self.scientific_qualification_performed
            or self.core_membership_affected
        ):
            raise ValueError("HCCQ cannot perform scientific qualification or membership gating")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["backend_schema_language_leakage"] = list(
            self.backend_schema_language_leakage
        )
        value["unnecessary_mechanical_burden"] = list(
            self.unnecessary_mechanical_burden
        )
        return value

    def is_stale(
        self,
        *,
        interface_profile: Any,
        backend_profile: Any,
        presentation: Any | None = None,
    ) -> bool:
        profile_stale = self.human_interface_sha256 != _digest(
            interface_profile
        ) or self.human_backend_sha256 != _digest(backend_profile)
        if presentation is None:
            return profile_stale
        return profile_stale or self.presentation_audit_sha256 != _digest(
            audit_human_presentation(presentation)
        )


def qualify_human_interface(
    case_id: str,
    interface_profile: Any,
    backend_profile: Any,
    *,
    actionability: bool,
    fairness: bool,
    feasibility: str,
    presentation: Any | None = None,
    auditor_id: str = "pending",
    audit_version: str = "hccq-v2",
) -> HCCQRecord:
    """Qualify only the optional human interface; never gate core membership."""

    presentation_audit = (
        audit_human_presentation(presentation)
        if presentation is not None
        else {
            "status": "PASS",
            "readability": True,
            "backend_schema_language_leakage": [],
            "unnecessary_mechanical_burden": [],
            "excessive_protocol_like_wording": False,
            "responsibility_clarity": {
                "benchmark_fixed_choices_clear": None,
                "respondent_choices_clear": None,
                "requested_reporting_clear": None,
            },
        }
    )
    clarity_values = presentation_audit["responsibility_clarity"].values()
    clarity = not any(value is False for value in clarity_values)
    comparable = (
        actionability
        and fairness
        and feasibility in {"LOW", "MODERATE", "HIGH"}
        and presentation_audit["status"] == "PASS"
        and clarity
    )
    qualification = "HUMAN_COMPARABLE" if comparable else "REVISE_HUMAN_INTERFACE"
    return HCCQRecord(
        str(case_id),
        str(interface_profile.get("profile_id", "interface"))
        if isinstance(interface_profile, Mapping)
        else "interface",
        _digest(interface_profile),
        str(backend_profile.get("profile_id", "backend"))
        if isinstance(backend_profile, Mapping)
        else "backend",
        _digest(backend_profile),
        qualification,
        auditor_id,
        audit_version,
        actionability=bool(actionability),
        fairness=bool(fairness),
        feasibility=str(feasibility),
        readability=bool(presentation_audit["readability"]),
        responsibility_clarity=clarity,
        backend_schema_language_leakage=tuple(
            presentation_audit["backend_schema_language_leakage"]
        ),
        unnecessary_mechanical_burden=tuple(
            presentation_audit["unnecessary_mechanical_burden"]
        ),
        excessive_protocol_like_wording=bool(
            presentation_audit["excessive_protocol_like_wording"]
        ),
        presentation_audit_sha256=_digest(presentation_audit),
    )


__all__ = [
    "HCCQRecord",
    "HCCQ_OUTCOMES",
    "audit_family_presentation_leakage",
    "audit_human_presentation",
    "qualify_human_interface",
]
