"""Deterministic, identity-blind inputs shared by calibration evaluators."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import DatasetManifest
from .python_runtime import reader_metadata_for_manifest
from .family_validator import canonicalize_operationalization_clause


@dataclass(frozen=True)
class AtomicFinding:
    finding_id: str
    statement: str
    source_section: str = "Finding"
    source_span: Mapping[str, int] | None = None
    source_text: str = ""
    extraction_rule: str = ""
    validation: Mapping[str, Any] | None = None
    # Semantic continuation metadata is explicit so downstream audits do not
    # infer claim boundaries from Markdown formatting or value shape.
    parent_required: bool = False
    continuation_of: str | None = None
    independently_judgeable: bool = True
    canonical_statement: str = ""
    dependency_status: str = "INDEPENDENT"
    resolved_subject_id: str | None = None
    resolved_subject: Mapping[str, Any] | None = None
    dependency_resolution_method: str | None = "NONE"
    source_spans: tuple[Mapping[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.dependency_status not in {"INDEPENDENT", "DEPENDENT_RESOLVED", "DEPENDENT_UNRESOLVED"}:
            raise ValueError(f"invalid dependency_status: {self.dependency_status}")
        allowed_methods = {None, "NONE", "PREVIOUS_ENTITY", "PREVIOUS_SENTENCE_ENTITY", "PARENT_FINDING", "PARENT_CLAUSE", "STRUCTURED_BLOCK", "STRUCTURED_BLOCK_PARENT", "PRONOUN_RESOLUTION", "OTHER_DETERMINISTIC"}
        if self.dependency_resolution_method not in allowed_methods:
            raise ValueError(f"invalid dependency_resolution_method: {self.dependency_resolution_method}")
        if not self.canonical_statement:
            object.__setattr__(self, "canonical_statement", self.statement)
        if self.resolved_subject is None:
            if self.resolved_subject_id or self.dependency_status == "DEPENDENT_RESOLVED":
                object.__setattr__(self, "resolved_subject", {
                    "entity_id": self.resolved_subject_id or self.finding_id,
                    "canonical_subject": None,
                    "source_finding_id": self.resolved_subject_id,
                })
            else:
                object.__setattr__(self, "resolved_subject", {
                    "entity_id": self.finding_id,
                    "canonical_subject": None,
                    "source_finding_id": None,
                })
        if not self.source_spans and self.source_span:
            object.__setattr__(self, "source_spans", (dict(self.source_span),))
        if self.dependency_status == "DEPENDENT_UNRESOLVED":
            object.__setattr__(self, "independently_judgeable", False)
        elif self.dependency_status == "DEPENDENT_RESOLVED":
            object.__setattr__(self, "independently_judgeable", True)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_spans"] = [dict(span) for span in self.source_spans]
        return value

    # ``section`` and the string span were used by early pilot artifacts.  Keep
    # read compatibility without exposing the legacy representation in new
    # serialized findings.
    @property
    def section(self) -> str:
        return self.source_section

    @property
    def raw_source_text(self) -> str:
        """Compatibility name for the untouched source slice."""

        return self.source_text

    @property
    def raw_start(self) -> int | None:
        return self.source_span.get("start_char") if self.source_span else None

    @property
    def raw_end(self) -> int | None:
        return self.source_span.get("end_char") if self.source_span else None


@dataclass(frozen=True)
class AtomicFindingSet:
    status: str
    findings: tuple[AtomicFinding, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [item.to_dict() for item in self.findings],
            "scoring_findings": [
                item.to_dict() for item in self.findings
                if item.independently_judgeable and item.dependency_status != "DEPENDENT_UNRESOLVED"
            ],
            "excluded_dependency_findings": [
                item.finding_id for item in self.findings
                if item.dependency_status == "DEPENDENT_UNRESOLVED" or not item.independently_judgeable
            ],
            "reason": self.reason,
        }


def _finding_normalize(value: str) -> str:
    value = re.sub(r"[*_`~]", "", value)
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value.rstrip(".!?;:")


def _finding_is_operationalization(value: str) -> bool:
    """Conservative filter: method prose is never silently counted as a claim."""

    return bool(re.match(
        r"^(?:using|use|we use|we used|computed|calculate|calculat(?:e|ed|ing)|"
        r"the (?:criterion|threshold|operationalization|method)|operationalization)\b",
        value.strip(), flags=re.IGNORECASE,
    ))


def _finding_is_format_only(value: str) -> bool:
    """Reject Markdown/LaTeX delimiters and bare numeric debris."""

    cleaned = re.sub(r"[*_`~]", "", value).strip()
    if not cleaned or re.fullmatch(r"[\W_]+", cleaned):
        return True
    if re.fullmatch(r"(?:[-+]?\d+(?:\.\d+)?|[()[\]{},.;:\\\s\"']+)+", cleaned):
        if re.search(r"[,，].*[,，]", cleaned) and re.search(r"[-+]?\d", cleaned):
            return False
        return True
    if len(cleaned.split()) <= 8 and not re.search(r"\d|=|\b(?:is|are|has|contains|spans|located|occurs|measured|reported|shows)\b", cleaned, re.I):
        return True
    if cleaned.endswith(":") and not re.search(r"\b(?:is|are|has|contains|equals|spans?|extent|range|coordinates?|at|near|from|to)\b", cleaned, re.I):
        return True
    return False


def _finding_is_dependency_fragment(value: str) -> bool:
    """Identify continuation prose that has no independent scientific claim."""

    stripped = value.strip()
    if re.match(r"^(?:mean|peak|retained|spatial|cartesian|velocity|strength)\b.*\b(?:is|are|was|were|equals?)\b", stripped, flags=re.IGNORECASE):
        return False
    if re.match(r"^(?:[xyz]\s*[:=]|\([a-z,\s]+\)\s*=|\|.*\||\\frac)", stripped, flags=re.IGNORECASE):
        return True
    # A full quantitative clause introduced by ``At the peak point`` is still
    # an independently reportable vector/value Finding; only its location
    # anchor is inherited from the preceding claim.
    if re.match(r"^at\s+(?:the\s+)?(?:peak|selected)\s+(?:point|location)\b", value.strip(), flags=re.IGNORECASE):
        return False
    return bool(re.match(
        r"^(?:and|or|but|while|whereas|which|that|with|above|below|at|from|to|spans?|extends?|peaks?|"
        r"a\s+(?:speed[- ]weighted|weighted)\s+center|(?:mean|peak|retained|spatial|cartesian|velocity|strength|point\s+count|region\s+coordinate)\b|"
        r"it|its|this\s+(?:component|region|core|point|quantity)|"
        r"this\b|[xyz]\s*[:=]|\([xyz,\s]+\)\s*=|\|.*\||\\frac\b|"
        r"approximately|respectively|as shown|corresponding to)\b",
        value.strip(), flags=re.IGNORECASE,
    ))


def _finding_is_pronoun_dependency(value: str) -> bool:
    """Return true for a dependent continuation worth retaining separately."""

    return bool(re.match(
        r"^(?:it|its|this(?:\s+(?:component|region|core|point|quantity))?|"
        r"the\s+(?:former|latter))\b",
        value.strip(), flags=re.IGNORECASE,
    ))


def _extract_subject(statement: str) -> str | None:
    """Extract an explicit grammatical subject without adding domain meaning."""

    text = _clean_finding_markup(statement).strip().rstrip(".!?;:")
    match = re.match(
        r"^(?P<subject>.+?)\s+(?:is|are|was|were|has|have|contains|contain|spans?|"
        r"lies?|occurs?|shows?|reports?|reaches?|measures?|equals?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        subject = re.sub(r"\s+", " ", match.group("subject")).strip(" ,")
        return subject if subject else None
    return None


def _resolved_subject_text(finding: AtomicFinding | None) -> str | None:
    """Return a previously established subject, including restored metadata."""

    if finding is None:
        return None
    direct = _extract_subject(finding.canonical_statement or finding.statement)
    if direct:
        return direct
    if isinstance(finding.resolved_subject, Mapping):
        value = finding.resolved_subject.get("canonical_subject")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_dependency_statement(statement: str, previous: AtomicFinding | None) -> tuple[str, str, str | None, str | None]:
    """Resolve only an explicit pronoun to the immediately preceding entity."""

    if not _finding_is_pronoun_dependency(statement):
        return statement, "INDEPENDENT", None, "NONE"
    if previous is None:
        return statement, "DEPENDENT_UNRESOLVED", None, None
    subject = _resolved_subject_text(previous)
    if not subject:
        return statement, "DEPENDENT_UNRESOLVED", None, None
    text = statement.strip()
    if re.match(r"^its\b", text, flags=re.IGNORECASE):
        remainder = re.sub(r"^its\s+", "", text, count=1, flags=re.IGNORECASE)
        property_match = re.match(r"(.+?)\s+is\s+(.+)$", remainder, flags=re.IGNORECASE)
        canonical = f"{subject} has {property_match.group(1)} {property_match.group(2)}" if property_match else f"{subject} has {remainder}"
    elif re.match(r"^it\b", text, flags=re.IGNORECASE):
        remainder = re.sub(r"^it\s+", "", text, count=1, flags=re.IGNORECASE)
        previous_text = previous.canonical_statement or previous.statement
        # A qualitative continuation of a coordinate/extent block (for
        # example ``It is concentrated near ...``) has no independently
        # adjudicable quantity or identity in the response contract.  Keep it
        # auditable but unresolved; explicit comparative claims such as the
        # Office O2 maximum-region sentence below remain resolvable.
        if re.match(r"^is\s+concentrated\b", remainder, re.I) and re.search(r"\b(?:spans?|extent|coordinates?)\b", previous_text, re.I):
            return statement, "DEPENDENT_UNRESOLVED", None, None
        # A maximum that occurs in a smaller region denotes that region as
        # the grammatical referent in the following comparison.  This is a
        # deterministic lexical antecedent, not a scientific inference.
        if re.search(r"\bmaximum\b", previous_text, re.I) and re.search(r"\bregion\b", previous_text, re.I):
            canonical = f"The region containing the absolute single-point maximum {remainder}"
        else:
            canonical = f"{subject} {remainder}"
    else:
        remainder = re.sub(r"^this(?:\s+(?:component|region|core|point|quantity))?\s*", "", text, count=1, flags=re.IGNORECASE)
        canonical = f"{subject} {remainder}" if remainder != text else text
    return canonical, "DEPENDENT_RESOLVED", previous.finding_id, "PRONOUN_RESOLUTION"


def _finding_is_syntactic_continuation(value: str) -> bool:
    """Return true for a split clause whose subject is inherited from a parent."""

    stripped = value.strip()
    if re.match(r"^(?:[xyz]\s*[:=]|\([a-z,\s]+\)\s*=|\|.*\||\\frac)", stripped, flags=re.IGNORECASE):
        return True
    return bool(re.match(
        r"^(?:spans?|extends?|has\s+(?:a\s+)?(?:center|location|mean|peak|extent|speed)|"
        r"peaks?\s+at|with\s+(?:a\s+)?(?:speed[- ]weighted|weighted|mean)?\s*(?:center|location)|"
        r"a\s+(?:speed[- ]weighted|weighted)\s+center|(?:mean|peak|retained|spatial|cartesian|velocity|strength|point\s+count|region\s+coordinate)\b|"
        r"[xyz]\s*[:=]|\([xyz,\s]+\)\s*=|\|.*\||\\frac\b|"
        r"and\s+(?:has|contains?|spans?))\b",
        value.strip(), flags=re.IGNORECASE,
    ))


def _select_dependency_parent(statement: str, findings: Sequence[AtomicFinding]) -> AtomicFinding | None:
    """Select a deterministic antecedent matching the pronoun's entity noun.

    A response often reports a region, then its coordinates/vector, and only
    afterwards says ``This component ...``.  Choosing the nearest sentence
    would incorrectly make the vector the component.  The noun in the
    pronoun is an explicit grammatical constraint, so prefer the latest
    compatible region/component/core/point finding and otherwise fall back to
    the latest subject-bearing finding.
    """

    text = statement.casefold()
    noun = next((item for item in ("component", "region", "core", "point", "quantity") if re.search(rf"\bthis\s+{item}\b", text)), None)
    candidates: list[tuple[AtomicFinding, str]] = []
    for item in reversed(findings):
        subject = _resolved_subject_text(item)
        if subject:
            candidates.append((item, subject.casefold()))
    if not candidates:
        return None
    if noun:
        entity_tokens = {
            "component": ("region", "component", "core", "feature", "structure"),
            "region": ("region", "component", "core", "feature"),
            "core": ("core", "region", "component"),
            "point": ("point", "location", "coordinate"),
            "quantity": ("quantity", "measure", "speed", "value"),
        }[noun]
        for item, subject in candidates:
            if any(re.search(rf"\b{re.escape(token)}\b", subject) for token in entity_tokens):
                return item
    return candidates[0][0]


def _merge_coordinate_continuations(
    text: str, units: Sequence[tuple[str, int, int, str]]
) -> list[tuple[str, int, int, str]]:
    """Keep a lead-in and its coordinate/range list in one source span.

    Markdown list items such as ``x=...``, ``y=...`` and ``z=...`` are not
    independently meaningful Findings.  They are continuations of the
    preceding ``spans approximately:`` claim.  The merge is performed using
    source offsets, so the resulting span remains an exact slice of ``text``.
    """

    merged: list[tuple[str, int, int, str]] = []
    index = 0
    coordinate_item = re.compile(
        r"^(?:\\\[|\$\$|\\begin\{|[xyz]\s*=|(?:x|y|z)\s*:\s*|[-+]?\d+(?:\.\d+)?\s*(?:[-–]\s*)|[-+]?(?:\d|\.\d))",
        re.IGNORECASE,
    )
    quantitative_item = re.compile(
        r"^(?:[xyz]\s*=|(?:x|y|z)\s*:\s*|[-+]?(?:\d|\.\d)|[\[\(\{]|"
        r"(?:n|count|number|value|magnitude|extent|mean|peak|strength|location|vector|speed)\b\s*[:=]?)",
        re.IGNORECASE,
    )
    while index < len(units):
        statement, start, end, rule = units[index]
        lead_end = end
        stripped = statement.strip()
        coordinate_lead = bool(re.search(r"\b(?:spans?|extent|range|coordinates?)\b", stripped, re.IGNORECASE)) and (
            stripped.endswith(":") or bool(re.search(r"\bspans?\s+approximately$", stripped, re.IGNORECASE))
        )
        quantitative_lead = stripped.endswith(":") and bool(re.search(
            r"\b(?:has|contains|located\s+at|equals?|value|magnitude|quantit(?:y|ies))\b",
            stripped, re.IGNORECASE,
        ))
        axis_lead = bool(re.match(r"^x\s*[:=]", stripped, re.IGNORECASE)) and index + 1 < len(units) and bool(re.match(r"^y\s*[:=]", units[index + 1][0].strip(), re.IGNORECASE))
        if coordinate_lead or quantitative_lead or axis_lead:
            next_index = index + 1
            while next_index < len(units):
                next_statement = units[next_index][0].strip()
                next_statement = re.sub(r"^[-*+]\s+", "", next_statement)
                matcher = coordinate_item if (coordinate_lead or axis_lead) else quantitative_item
                if not matcher.match(next_statement):
                    break
                end = units[next_index][2]
                next_index += 1
            if next_index > index + 1:
                # Keep compact legacy two-item blocks intact, but split a
                # labelled multi-property block (count/location/mean/peak)
                # into independently judgeable units.
                block_items = [units[item][0].strip() for item in range(index + 1, next_index)]
                labelled_properties = sum(
                    bool(re.search(r"\b(?:count|points?|location|mean|peak|strength|speed)\b", item, re.I))
                    for item in block_items
                )
                if quantitative_lead and not coordinate_lead and not axis_lead and len(block_items) < 3 and labelled_properties < 3:
                    raw = text[start:end]
                    normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", raw)
                    normalized = _clean_finding_markup(normalized)
                    merged.append((normalized, start, end, rule + ":quantitative_block"))
                    index = next_index
                    continue
                if quantitative_lead and not coordinate_lead and not axis_lead:
                    # Drop only the syntactic lead-in.  The following units
                    # retain their exact source spans and are emitted below.
                    raw_statement = text[start:lead_end]
                    prefix_match = re.search(r"(?P<prefix>.+?)\s+and\s+has\s*:\s*$", raw_statement, flags=re.IGNORECASE | re.DOTALL)
                    if prefix_match:
                        prefix = _clean_finding_markup(prefix_match.group("prefix")).strip()
                        if prefix:
                            merged.append((prefix, start, start + prefix_match.end("prefix"), rule + ":quantitative_lead"))
                    index += 1
                    continue
                raw = text[start:end]
                # Remove list markers only for the normalized statement; the
                # source span continues to point at the untouched raw text.
                normalized = re.sub(r"(?m)^\s*[-*+]\s+", "", raw)
                normalized = _clean_finding_markup(normalized)
                merged.append((normalized, start, end, rule + ":coordinate_continuation"))
                index = next_index
                continue
        merged.append((statement, start, end, rule))
        index += 1
    return merged


def _clean_finding_markup(value: str) -> str:
    value = re.sub(r"\\(?:\[|\]|\(|\))", "", value)
    value = value.replace("$$", "")
    value = re.sub(r"\\\\\s*", " ", value)
    value = re.sub(r"[*_`~]", "", value)
    return re.sub(r"\s+", " ", value).strip()


# Numeric tokens are deliberately lexical.  This catches decimal/scientific
# notation and percentages without assigning any scientific meaning to them.
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?:\s*[xX×]\s*10\^?[-+]?\d+)?%?"
)


def numeric_tokens(text: str | None) -> tuple[str, ...]:
    """Return normalized numeric lexemes for source/statement comparison."""

    if not isinstance(text, str):
        return ()
    return tuple(re.sub(r"\s+", "", match.group(0)).casefold() for match in _NUMERIC_TOKEN_RE.finditer(text))


def validate_numeric_preservation(source_text: str | None, statement: str | None) -> dict[str, Any]:
    """Check that extraction did not drop or alter numeric content."""

    source = numeric_tokens(source_text)
    extracted = numeric_tokens(statement)
    missing = list(source)
    for token in extracted:
        if token in missing:
            missing.remove(token)
    extra = list(extracted)
    for token in source:
        if token in extra:
            extra.remove(token)
    valid = not missing and not extra
    return {
        "valid": valid,
        "status": "PASS" if valid else "FAIL",
        "failure_code": None if valid else "NUMERIC_CONTENT_CORRUPTION",
        "source_numeric_tokens": list(source),
        "statement_numeric_tokens": list(extracted),
        "missing_tokens": missing,
        "extra_tokens": extra,
    }


def validate_source_fidelity(finding: AtomicFinding | Mapping[str, Any], original_response: str | None = None) -> dict[str, Any]:
    """Validate that a finding's span and source text agree with its claim."""

    source_text = finding.source_text if isinstance(finding, AtomicFinding) else finding.get("source_text", "")
    statement = finding.statement if isinstance(finding, AtomicFinding) else finding.get("statement", "")
    span = finding.source_span if isinstance(finding, AtomicFinding) else finding.get("source_span")
    valid_span = isinstance(span, Mapping) and isinstance(span.get("start_char"), int) and isinstance(span.get("end_char"), int) and span["start_char"] <= span["end_char"]
    if valid_span and isinstance(original_response, str):
        valid_span = original_response[span["start_char"]:span["end_char"]] == source_text
    nonempty = bool(str(source_text).strip()) and bool(str(statement).strip())
    numeric = validate_numeric_preservation(source_text, statement)
    valid = valid_span and nonempty and numeric["valid"]
    return {
        "valid": valid,
        "status": "PASS" if valid else "FAIL",
        "failure_code": None if valid else ("SOURCE_SPAN_MISMATCH" if not valid_span or not nonempty else numeric["failure_code"]),
        "numeric_preservation": numeric,
    }


def _protected_same_length(value: str) -> str:
    """Mask math and markup while preserving every character offset."""

    value = re.sub(r"(?:\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$))", lambda match: "M" * len(match.group(0)), value, flags=re.S)
    return re.sub(r"[*_`~]", " ", value)


def _scan_source_segments(masked: str) -> list[tuple[int, int]]:
    """Scan boundary punctuation without excluding decimal points from matches."""

    segments: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(masked):
        if character not in ";.!?":
            continue
        previous = masked[index - 1] if index else ""
        following = masked[index + 1] if index + 1 < len(masked) else ""
        decimal_or_exponent = character == "." and previous.isdigit() and following.isdigit()
        if decimal_or_exponent:
            continue
        segments.append((start, index + 1))
        start = index + 1
    if start < len(masked):
        segments.append((start, len(masked)))
    return segments


def _split_finding_units(text: str) -> list[tuple[str, int, int, str]]:
    """Split Markdown claims while protecting display-math blocks."""

    units: list[tuple[str, int, int, str]] = []
    # Group lines belonging to one LaTeX display before segmentation.  This
    # keeps ``\\[ ... \\]`` and multiline coordinate tuples as one span.
    groups: list[tuple[str, int, int]] = []
    lines = list(re.finditer(r"[^\n]+", text))
    index = 0
    while index < len(lines):
        first = lines[index]
        line = first.group(0)
        if not line.strip():
            index += 1
            continue
        value = line
        end = first.end()
        if re.search(r"\\\[|\$\$|\\begin\{", line):
            while index + 1 < len(lines) and not re.search(r"\\\]|\$\$|\\end\{", value):
                index += 1
                value += " " + lines[index].group(0)
                end = lines[index].end()
        groups.append((value, first.start(), end))
        index += 1
    for line, line_start, line_end in groups:
        if not line.strip() or re.match(r"^\s*#{1,6}\s+", line):
            continue
        leading = len(line) - len(line.lstrip())
        marker = re.match(r"(?:[-*+]\s+|\d+[.)]\s+)", line.lstrip())
        marker_len = len(marker.group(0)) if marker else 0
        value = line.lstrip()[marker_len:].strip()
        if not value:
            continue
        start = line_start + leading + marker_len
        end = start + len(value)
        rule = "bullet_or_numbered_item" if marker else "paragraph_sentence"
        # Protect math blocks from sentence punctuation and internal commas;
        # equal-length masking keeps spans indexable in the original response.
        safe_value = _protected_same_length(value)
        for piece_start, piece_end in _scan_source_segments(safe_value):
            raw_fragment = value[piece_start:piece_end]
            left_trim = len(raw_fragment) - len(raw_fragment.lstrip())
            right_trim = len(raw_fragment.rstrip())
            fragment_start = start + piece_start + left_trim
            fragment_end = start + piece_start + right_trim
            raw_fragment = value[piece_start + left_trim:piece_start + right_trim]
            fragment = _clean_finding_markup(raw_fragment)
            if not fragment:
                continue
            if _finding_is_format_only(fragment):
                continue
            # Split independently judgeable quantitative propositions while
            # retaining exact source spans for each proposition.  This covers
            # the recurring ``mean ..., with weighted center ...``,
            # ``contains N points and spans ...`` and ``count ... and peak``
            # forms without splitting ordinary scientific prose.
            separators = list(re.finditer(
                r"(?:,\s+with\s+(?=(?:a\s+)?(?:speed[- ]weighted|weighted|mean)\s+(?:center|location))|"
                r"\s+and\s+(?=(?:has|contains?|spans?|extends?|reports?|peak|maximum|a\s+peak|the\s+peak)\b))",
                raw_fragment, flags=re.IGNORECASE,
            ))
            if separators:
                pieces: list[tuple[int, int]] = []
                cursor = 0
                for separator in separators:
                    left = _clean_finding_markup(raw_fragment[cursor:separator.start()]).strip()
                    if left and re.search(r"\b(?:count|points?|mean|peak|speed|center|location|extent|spans?|contains?)\b", left, re.I):
                        pieces.append((cursor, separator.start()))
                    cursor = separator.end()
                right = _clean_finding_markup(raw_fragment[cursor:]).strip()
                if right and pieces:
                    pieces.append((cursor, len(raw_fragment)))
                for piece_start, piece_end in pieces:
                    raw_piece = raw_fragment[piece_start:piece_end]
                    piece = _clean_finding_markup(raw_piece).strip()
                    if not piece:
                        continue
                    leading_piece = len(raw_piece) - len(raw_piece.lstrip())
                    trailing_piece = len(raw_piece.rstrip())
                    units.append((piece, fragment_start + piece_start + leading_piece, fragment_start + piece_start + trailing_piece, "compound_quantitative_clause"))
            else:
                units.append((fragment, fragment_start, fragment_end, rule))
    return _merge_coordinate_continuations(text, units)


def extract_atomic_findings(final_response: str | None) -> AtomicFindingSet:
    """Freeze duplicate-free, claim-oriented Findings without judging them."""

    if not isinstance(final_response, str) or not final_response.strip():
        return AtomicFindingSet("EXTRACTION_UNCERTAIN", (), "final response is absent")
    match = re.search(r"(?im)^##\s+Finding\s*$", final_response)
    mode = "PROFILED_FINDING_SECTION"
    if match is None:
        # Legacy pilot responses may use a plain ``Finding:`` marker.  This is
        # explicitly a compatibility path and never changes the profiled rule.
        match = re.search(r"(?im)^\s*Finding\s*:\s*", final_response)
        mode = "LEGACY_FINDING_FALLBACK"
    if match is None:
        return AtomicFindingSet("EXTRACTION_UNCERTAIN", (), "required Finding section is absent")
    section_start = match.end()
    if mode == "PROFILED_FINDING_SECTION":
        next_heading = re.search(r"(?m)^#{1,2}\s+\S.*$", final_response[section_start:])
        section_end = section_start + next_heading.start() if next_heading else len(final_response)
    else:
        section_end = len(final_response)
    section = final_response[section_start:section_end]
    if not section.strip():
        return AtomicFindingSet("EXTRACTION_UNCERTAIN", (), "Finding section is empty")
    seen: set[str] = set()
    findings: list[AtomicFinding] = []
    for statement, start, end, rule in _split_finding_units(section):
        raw_statement = statement.strip()
        # A syntactic ``has:``/``contains:`` lead-in is not itself a Finding;
        # the following labelled properties carry the scientific claims.
        if raw_statement.endswith(":") and re.search(r"\b(?:has|contains|located\s+at)\s*:$", raw_statement, re.I):
            continue
        statement = raw_statement.rstrip(".!?;:")
        if not statement or _finding_is_operationalization(statement):
            continue
        absolute_start = section_start + start
        absolute_end = section_start + end
        key = _finding_normalize(statement)
        if not key or key in seen:
            continue
        seen.add(key)
        finding = AtomicFinding(
            finding_id=f"P{len(findings) + 1}",
            statement=statement,
            source_section="Finding",
            source_span={"start_char": absolute_start, "end_char": absolute_end},
            source_text=final_response[absolute_start:absolute_end],
            extraction_rule=f"{mode}:{rule}",
            resolved_subject={"entity_id": f"P{len(findings) + 1}", "canonical_subject": _extract_subject(statement), "source_finding_id": None},
        )
        if _finding_is_dependency_fragment(statement):
            # Pronoun continuations remain explicit records.  A reliable
            # antecedent is canonicalized and remains scoreable; an absent or
            # ambiguous antecedent is retained but excluded from scoring.
            if _finding_is_pronoun_dependency(statement):
                previous = findings[-1] if findings else None
                subject_parent = _select_dependency_parent(statement, findings) or previous
                canonical, dependency_status, subject_id, resolution_method = _resolve_dependency_statement(statement, subject_parent)
                dependent = AtomicFinding(
                        finding_id=f"P{len(findings) + 1}",
                        statement=statement,
                        source_section="Finding",
                        source_span={"start_char": absolute_start, "end_char": absolute_end},
                        source_text=final_response[absolute_start:absolute_end],
                        extraction_rule=f"{mode}:{rule}:dependent_continuation",
                        parent_required=True,
                        continuation_of=previous.finding_id if previous else None,
                        independently_judgeable=dependency_status == "DEPENDENT_RESOLVED",
                        canonical_statement=canonical,
                        dependency_status=dependency_status,
                        resolved_subject_id=subject_id,
                        resolved_subject={"entity_id": subject_id, "canonical_subject": _extract_subject(canonical), "source_finding_id": subject_id},
                        dependency_resolution_method=resolution_method,
                )
                object.__setattr__(dependent, "validation", validate_source_fidelity(dependent, final_response))
                findings.append(dependent)
                continue
            if _finding_is_syntactic_continuation(statement):
                previous = findings[-1] if findings else None
                subject = _resolved_subject_text(previous)
                if subject:
                    if re.match(
                        r"^(?:mean|peak|retained|spatial|cartesian|velocity|strength|point\s+count|region\s+coordinate|a\s+(?:speed[- ]weighted|weighted)\s+center|[xyz]\s*[:=]|\([a-z,\s]+\)\s*=|\|.*\||\\frac)",
                        statement,
                        flags=re.IGNORECASE,
                    ):
                        canonical = f"{subject} has {statement}"
                    else:
                        canonical = f"{subject} {statement}"
                    dependent = AtomicFinding(
                        finding_id=f"P{len(findings) + 1}",
                        statement=statement,
                        source_section="Finding",
                        source_span={"start_char": absolute_start, "end_char": absolute_end},
                        source_text=final_response[absolute_start:absolute_end],
                        extraction_rule=f"{mode}:{rule}:parent_continuation",
                        parent_required=True,
                        continuation_of=previous.finding_id,
                        independently_judgeable=True,
                        canonical_statement=canonical,
                        # The restored canonical statement is a complete,
                        # independently judgeable proposition.  Keep the
                        # parent link for auditability while reserving the
                        # dependency state for pronouns whose source wording
                        # is itself referential.
                        dependency_status="INDEPENDENT",
                        resolved_subject_id=previous.finding_id,
                        resolved_subject={"entity_id": previous.finding_id, "canonical_subject": subject, "source_finding_id": previous.finding_id},
                        dependency_resolution_method="STRUCTURED_BLOCK_PARENT" if "coordinate" in rule else "PARENT_CLAUSE",
                    )
                    object.__setattr__(dependent, "validation", validate_source_fidelity(dependent, final_response))
                    findings.append(dependent)
                    continue
            # Preserve the source as part of the preceding claim rather than
            # emitting a semantically incomplete non-pronoun fragment.
            if findings:
                previous = findings[-1]
                previous_span = previous.source_span or {}
                merged_start = min(int(previous_span.get("start_char", absolute_start)), absolute_start)
                merged_end = max(int(previous_span.get("end_char", absolute_end)), absolute_end)
                merged = AtomicFinding(
                    finding_id=previous.finding_id,
                    statement=_clean_finding_markup(final_response[merged_start:merged_end]).rstrip(".!?;:"),
                    source_section=previous.source_section,
                    source_span={"start_char": merged_start, "end_char": merged_end},
                    source_text=final_response[merged_start:merged_end],
                    extraction_rule=previous.extraction_rule + "+dependency_fragment",
                    parent_required=False,
                    continuation_of=None,
                    independently_judgeable=True,
                    canonical_statement=_clean_finding_markup(final_response[merged_start:merged_end]).rstrip(".!?;:"),
                    dependency_status="INDEPENDENT",
                    resolved_subject=previous.resolved_subject,
                )
                object.__setattr__(merged, "validation", validate_source_fidelity(merged, final_response))
                findings[-1] = merged
            continue
        object.__setattr__(finding, "validation", validate_source_fidelity(finding, final_response))
        findings.append(finding)
    if not findings:
        return AtomicFindingSet("EXTRACTION_UNCERTAIN", (), "no atomic claim was found")
    return AtomicFindingSet("EXTRACTED", tuple(findings))


def sanitize_evaluation_evidence(
    trajectory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Preserve scientific execution evidence while removing identity/efficiency telemetry."""

    call_ids: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    next_id = 1
    for event in trajectory:
        event_type = event.get("event")
        if event_type == "model_turn" and isinstance(event.get("tool_batch"), Mapping):
            for call in event["tool_batch"].get("calls", []):
                if not isinstance(call, Mapping) or not isinstance(call.get("arguments"), Mapping):
                    continue
                code = call["arguments"].get("code")
                if not isinstance(code, str):
                    continue
                original = str(call.get("canonical_call_id") or f"anonymous-{next_id}")
                neutral = call_ids.setdefault(original, f"T{next_id}")
                if neutral == f"T{next_id}":
                    next_id += 1
                events.append({"event": "python_request", "tool_call_id": neutral, "code": code})
        elif event_type == "model_turn" and isinstance(event.get("code"), str):
            neutral = f"T{next_id}"
            next_id += 1
            events.append({"event": "python_request", "tool_call_id": neutral, "code": event["code"]})
        elif event_type == "python_execution":
            original = str(event.get("canonical_call_id") or f"execution-{event.get('execution_index', next_id)}")
            neutral = call_ids.get(original)
            if neutral is None:
                neutral = f"T{next_id}"
                next_id += 1
                call_ids[original] = neutral
            exception = event.get("exception")
            clean_exception = None
            if isinstance(exception, Mapping):
                clean_exception = {
                    "type": exception.get("type"),
                    "message": exception.get("message"),
                    "traceback": exception.get("traceback"),
                }
            events.append({
                "event": "python_result",
                "tool_call_id": neutral,
                "execution_index": event.get("execution_index"),
                "success": event.get("success"),
                "stdout": event.get("stdout", ""),
                "stderr": event.get("stderr", ""),
                "exception": clean_exception,
                "output_truncated": bool(event.get("output_truncated", False)),
            })
    return {"schema_version": "evaluation-evidence-v1", "events": events}


def build_evaluator_data_contract(
    case_input: Mapping[str, Any], manifest: DatasetManifest,
    case_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge all model-visible interpretation metadata without exposing GT resources."""

    flow_data = case_input.get("flow_data", {})
    metadata = flow_data.get("data_metadata", {}) if isinstance(flow_data, Mapping) else {}
    reader = reader_metadata_for_manifest(manifest)
    validity_rules = [
        {
            "name": item.get("name"),
            "association": item.get("association"),
            "validity_rule": item.get("validity_rule"),
        }
        for item in reader.get("grid_blanking", [])
        if item.get("validity_rule")
    ]
    source_metadata = case_metadata if isinstance(case_metadata, Mapping) else metadata
    principal = source_metadata.get("principal_operationalization_dimensions")
    if not isinstance(principal, list):
        principal = source_metadata.get("specified_analysis_dimensions", [])
    if not isinstance(principal, list):
        principal = []
    principal = [
        item.get("dimension", item.get("name")) if isinstance(item, Mapping) else item
        for item in principal
    ]
    principal = [str(item) for item in principal if item is not None]
    unresolved = source_metadata.get("unresolved_operationalization_dimensions", [])
    if not isinstance(unresolved, list):
        unresolved = []
    unresolved = [str(item.get("dimension", item.get("name")) if isinstance(item, Mapping) else item) for item in unresolved]
    resolved = source_metadata.get("resolved_operationalization_dimensions")
    if not isinstance(resolved, list):
        resolved = [item for item in principal if item not in set(unresolved)]
    resolved = [str(item.get("dimension", item.get("name")) if isinstance(item, Mapping) else item) for item in resolved]
    authored_clauses = source_metadata.get("resolved_operationalization_clauses")
    if not isinstance(authored_clauses, Mapping):
        authored_clauses = {}
        for item in source_metadata.get("explicit_method_constraints", []) or []:
            if not isinstance(item, Mapping):
                continue
            dimension = str(item.get("category", ""))
            if dimension == "analysis_procedure":
                dimension = "aggregation_or_representation"
            if dimension in set(resolved):
                authored_clauses[dimension] = canonicalize_operationalization_clause(dimension, str(item.get("statement", "")))
    else:
        authored_clauses = {
            str(dimension): dict(clause) if isinstance(clause, Mapping) else {"description": str(clause)}
            for dimension, clause in authored_clauses.items()
            if str(dimension) in set(resolved)
        }
    return {
        "schema_version": "evaluator-data-contract-v1",
        "flow_data": {
            "data_files": flow_data.get("data_files", []) if isinstance(flow_data, Mapping) else [],
            "data_metadata": metadata,
        },
        "reader_metadata": reader,
        "validity_rules": validity_rules,
        "validity_rule": "; ".join(
            str(item["validity_rule"]) for item in validity_rules
            if item.get("validity_rule")
        ) or None,
        "coordinate_semantics": metadata.get("coordinate_system") if isinstance(metadata, Mapping) else None,
        "field_semantics": metadata.get("variables", []) if isinstance(metadata, Mapping) else [],
        "case_context": case_input.get("case_context"),
        "operationalization_dimensions": {
            "principal_dimensions": principal,
            "resolved_dimensions": resolved,
            "unresolved_dimensions": unresolved,
        },
        "resolved_operationalization_clauses": authored_clauses,
    }


def load_evaluator_data_contract(case_dir: str | Path) -> dict[str, Any]:
    import json

    directory = Path(case_dir)
    case_input = json.loads((directory / "case_input.json").read_text(encoding="utf-8"))
    dataset_dir = directory.parents[2]
    manifest = DatasetManifest.model_validate_json(
        (dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    metadata_path = directory / "case_construction_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    return build_evaluator_data_contract(case_input, manifest, metadata)


__all__ = [
    "AtomicFinding", "AtomicFindingSet", "build_evaluator_data_contract",
    "extract_atomic_findings", "load_evaluator_data_contract", "sanitize_evaluation_evidence",
    "numeric_tokens", "validate_numeric_preservation", "validate_source_fidelity",
    "_finding_is_dependency_fragment",
]
