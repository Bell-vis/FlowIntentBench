"""Deterministic source checks for quoted, extracted values.

These checks establish literal support, not semantic correctness. Scientific
quantity/method alignment remains a separately recorded reviewer judgment.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re


_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)")


def bind_quote(answer: str, quotation: str | None):
    if not isinstance(quotation, str) or not quotation.strip():
        return None
    start = answer.find(quotation)
    if start >= 0:
        return {"start": start, "end": start + len(quotation), "text": quotation, "mode": "EXACT"}
    spans = list(re.finditer(r"\S+", answer))
    wanted = quotation.split()
    tokens = [m.group() for m in spans]
    positions = [i for i in range(len(tokens) - len(wanted) + 1) if tokens[i:i + len(wanted)] == wanted]
    if len(positions) == 1:
        start, end = spans[positions[0]].start(), spans[positions[0] + len(wanted) - 1].end()
        return {"start": start, "end": end, "text": answer[start:end], "mode": "WHITESPACE_ONLY"}
    # Reviewers sometimes omit presentation markup. Preserve an offset map and
    # require one exact rendered span; never fuzzy-match scientific wording.
    def rendered(text):
        chars, offsets, i = [], [], 0
        while i < len(text):
            if text.startswith("**", i):
                i += 2
                continue
            if text[i] == "`":
                i += 1
                continue
            c = " " if text[i].isspace() else text[i]
            if c != " " or not chars or chars[-1] != " ":
                chars.append(c)
                offsets.append(i)
            i += 1
        return "".join(chars), offsets
    source, offsets = rendered(answer)
    target, _ = rendered(quotation)
    target = target.strip()
    start = source.find(target)
    mode = 'PRESENTATION_MARKUP_ONLY'
    # A copied sentence fragment may capitalize its initial function word
    # ("Its R-squared is" versus "its R-squared is"). Only that first character
    # may vary, and only for an explicit function-word list. Scientific names,
    # units, variable case and the remainder of the quotation stay exact.
    if start < 0 and re.match(r'(?i)(?:the|its|this|these|those|using|for|with|from)\s+(?=[A-Za-z])', target):
        variant = target[0].swapcase() + target[1:]
        at = source.find(variant)
        if at >= 0:
            target, start, mode = variant, at, 'INITIAL_FUNCTION_WORD_CASE_ONLY'
    if len(target) < 12 or start < 0 or source.find(target, start+1) >= 0:
        return None
    lo, hi = offsets[start], offsets[start+len(target)-1]+1
    return {"start": lo, "end": hi, "text": answer[lo:hi], "mode": mode}


def numeric_notation(text: str):
    # Unicode minus is a representation difference, not a numerical change.
    text = text.replace("−", "-").replace("–", "-").replace("‑", "-")
    # A prose-labelled range can use the TeX double hyphen as its separator.
    # Require the explicit range/stratum label; arithmetic a--b and signed
    # vectors must retain their signs. A negative upper bound uses three dashes.
    number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    text = re.sub(r'(\b(?:range|interval|stratum)\s*(?:(?:is|of)\s*)?\(?\s*'+number+r')\s*--\s*('+number+r')',
                  r'\1 to \2', text, flags=re.I)
    text = re.sub(r"(?<![\w.])(-)\s+(?=\d)", r"\1", text)
    superscripts = str.maketrans("⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "+-0123456789")
    text = re.sub(r"10([⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^{" + m[1].translate(superscripts) + "}", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:×|\\times|\*)\s*10\s*\^\s*\{?\(?([-+]?\d+)\)?\}?",
                  r"\1e\2", text)
    # Space-grouped cardinalities occur in scientific prose ("3 844 points").
    # Require an explicit count label; a whitespace-separated vector must not
    # become an invented scalar. Do not bridge newlines or decimal tokens.
    grouped = re.compile(r'(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?![\d.])')
    count_label = r'(?:points?|cells?|regions?|components?|locations?|speeds?|count|size)'
    def cardinality(match):
        before, after = text[:match.start()], text[match.end():]
        prefix = re.search(count_label+r'\s*:\s*$',before,re.I)
        suffix = re.match(r'\s+(?:(?:non[- ]zero|positive|finite|high[- ]speed|grid|mesh|retained)\s+)*'+count_label+r'\b',after,re.I)
        return re.sub(r'[ \u00a0\u202f]','',match[0]) if prefix or suffix else match[0]
    text = grouped.sub(cardinality,text)
    return text


def numeric_string(value):
    """Canonicalize only a complete numeric literal, never interpret prose."""
    if not isinstance(value, str):
        return value
    text = numeric_notation(value.strip())
    if _NUMBER.fullmatch(text):
        result = float(Decimal(text.replace(",", "")))
        if math.isfinite(result):
            return result
    return value


def numeric_literals(text: str):
    text = numeric_notation(text)
    for match in _NUMBER.finditer(text):
        try:
            yield Decimal(match.group().replace(",", ""))
        except InvalidOperation:
            continue


def value_is_bound(value, quotation: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return bool(value.strip()) and value.strip() in quotation
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        return False
    # Cardinalities are often written in words ("Six regions"). This is a
    # literal representation change, not scientific calculation or tolerance.
    words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")
    if not isinstance(value, list) and value == int(value) and 0 <= value < len(words):
        pattern = r"(?<![\w-])" + words[int(value)] + r"\b(?![ -](?:hundred|thousand|million|billion|one|two|three|four|five|six|seven|eight|nine)\b)"
        if re.search(pattern, quotation, re.I):
            return True
    available = list(numeric_literals(quotation))
    # Preserve order and multiplicity for vectors. Never match an arbitrary
    # number elsewhere in the answer or use a tolerance to repair extraction.
    offset = 0
    for item in values:
        target = Decimal(str(item))
        # JSON/Pydantic floating values are binary64. A decimal source token
        # may have more digits than its round-trip Python spelling. Equality
        # of the parsed float is a representation check, never an isclose
        # tolerance; integers retain exact decimal identity.
        found = next((i for i in range(offset, len(available)) if available[i] == target
                      or type(item) is float and float(available[i]) == item), None)
        if found is None:
            return False
        offset = found + 1
    return True


def is_coordinate_extent_statement(statement: str) -> bool:
    return (bool(re.search(r'\b(?:coordinate|spatial)[-\w ]{0,40}\bextents?\b|\bextents?\b[-\w ]{0,40}\b(?:coordinate|spatial)\b', statement, re.I))
            and not re.search(r'\bindex\b|\bindices\b', statement, re.I))


def coordinate_extent_derivation(value, quotation):
    """Prove a supplied extent from three explicitly printed coordinate bounds.

    This uses only answer literals, never mesh data or reference numbers. It
    does not infer a region, coordinate system, hidden precision or a bound.
    The caller must establish that the requested quantity is a spatial extent.
    """
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in {int, float} or not math.isfinite(v) for v in value):
        return None
    # Preserve an explicit range dash before normalizing numeric notation.
    text = numeric_notation(quotation.replace('–', '__RANGE__').replace('—', '__RANGE__'))
    n = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    prefix = r'(?<!\w)([xyz])\s*(?:=|∈|\\in|:)\s*'
    bracket = re.compile(prefix + r'\[\s*(' + n + r')\s*,\s*(' + n + r')\s*\]', re.I)
    ranged = re.compile(prefix + r'(' + n + r')\s*(?:__RANGE__|\bto\b)\s*(' + n + r')', re.I)
    bounds = {axis: set() for axis in 'xyz'}
    for pattern in (bracket, ranged):
        for m in pattern.finditer(text):
            bounds[m[1].lower()].add((Decimal(m[2]), Decimal(m[3])))
    if any(len(v) != 1 for v in bounds.values()):
        return None  # Missing or competing region bounds cannot be selected.
    pairs = [next(iter(bounds[a])) for a in 'xyz']
    if any(hi < lo for lo, hi in pairs):
        return None
    widths = [hi-lo for lo, hi in pairs]
    if any(w != Decimal(str(v)) and not (type(v) is float and float(w) == v) for w, v in zip(widths, value)):
        return None
    return {'rule': 'PRINTED_COORDINATE_UPPER_MINUS_LOWER',
            'axis_bounds': {a: [str(v) for v in pair] for a, pair in zip('xyz', pairs)},
            'derived_extent': [str(w) for w in widths]}


def bind_value(answer: str, quotation: str | None, value, *, coordinate_extent=False):
    bound = bind_quote(answer, quotation)
    if bound is None:
        return {"status": "UNBOUND", "reason": "SOURCE_QUOTATION_UNLOCATABLE", "source": None}
    if not value_is_bound(value, bound["text"]):
        derivation = coordinate_extent_derivation(value, bound['text']) if coordinate_extent else None
        if derivation:
            return {'status': 'BOUND', 'reason': None, 'source': bound, 'derivation': derivation}
        return {"status": "UNBOUND", "reason": "VALUE_NOT_IN_SOURCE", "source": bound}
    return {"status": "BOUND", "reason": None, "source": bound}


def unit_is_bound(unit, quotation: str) -> bool:
    """Require a printed unit declaration; never infer SI or dimensionlessness."""
    if unit is None:
        return True  # The numerical verifier decides whether absence is usable.
    def normalize(text):
        text = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r" \1 ", text)
        text = (text.replace("$", "").replace("{", "").replace("}", "")
                .replace("²", "^2").replace("³", "^3").replace("⁻¹", "^-1"))
        text = text.replace("‑", "-").replace(r"\times", " × ")
        text = re.sub(r"\b(file|mesh|dataset|grid)(?:['’]s)\b", r"\1", text, flags=re.I)
        text = re.sub(r"\(\s*(undeclared)\s*\)", r"\1", text, flags=re.I)
        return " ".join(text.split())
    text, unit = normalize(quotation), normalize(unit.strip())
    # Possessive punctuation is a spelling variation of an explicit native
    # coordinate declaration. It never establishes a physical conversion.
    native_coordinate = r"\bdataset(?:['’]s)?\s+(?:spatial\s+)?coordinate units\b"
    if re.fullmatch(native_coordinate, unit, re.I):
        return bool(re.search(native_coordinate, text, re.I))
    # Equivalent spellings of the SAME explicit reciprocal-second unit.
    if unit in {"1/s", "s^-1"}:
        return bool(re.search(r"(?<!\w)(?:1\s*/\s*s|s\s*\^\s*-1)(?!\w)", text))
    if unit in {"1", "dimensionless", "unitless", "fraction"}:
        return bool(re.search(r"\b(?:dimensionless|unitless|fraction)\b|\[1\]|\(1\)|units?\s*[:=]\s*1\b", text, re.I))
    if unit in {"%", "percent", "percentage"}:
        return "%" in text or bool(re.search(r"\bpercent(?:age)?\b", text, re.I))
    return bool(re.search(r"(?<!\w)" + re.escape(unit) + r"(?!\w)", text))


def is_native_scale_unit(unit) -> bool:
    """Recognize declarations of the supplied numerical scale, not SI units.

    This does not establish source support, quantity identity or a conversion.
    The caller must separately bind the declaration and match the quantity.
    Only named generic field/coordinate scales are accepted; arbitrary words
    such as 'stored cm units' must not erase a physical conversion requirement.
    """
    if not isinstance(unit, str):
        return False
    text = unit.strip().lower().replace('²', '^2').replace('³', '^3').replace('‑', '-')
    text = ' '.join(text.split())
    native = r"(?:stored|native|file|dataset(?:['’]s)?|mesh|grid|code|data|simulation|supplied)"
    quantity = (r"(?:[xyz](?:-coordinate)?|coordinate(?:[- ](?:area|volume))?|"
                r"(?:mass[- ])?flux(?:[- ]density)?|current[- ]density|density|pressure|speed|velocity|field|flow|length|area|volume|"
                r"density\s*(?:×|\*|times|[-–])\s*velocity)")
    suffix = r"(?:undeclared\s+)?(?:units?|scale)(?:\s*\^\s*[23])?"
    patterns = (
        native + r"(?:[- /]" + native + r")?(?:[- ]" + quantity + r")?\s+" + suffix,
        quantity + r"[- ]" + suffix + r"\s+as\s+(?:stored|supplied)",
        r"(?:stored[- ])?(?:[xyz][- ])?coordinate[- ]" + suffix,
        r"(?:stored[- ])?coordinate[- ](?:area|volume)[- ]" + suffix,
        r"(?:square(?:d)?|cubic|cubed)\s+" + native + r"[- ]coordinate[- ]" + suffix,
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def quantity_match_conflict(statement: str, quotation: str, reference_statement: str):
    """Reject a demonstrated reviewer confusion without judging the claim false.

    A printed threshold/percentile is not the regional mean. Ambiguous labels
    remain a reviewer responsibility; this guard only widens uncertainty.
    """
    source = statement + "\n" + quotation
    if (re.search(r"\b(?:threshold|percentile|quantile)\b", source, re.I)
            and not re.search(r"\b(?:mean|average|peak|maximum|centroid)\b", source, re.I)
            and re.search(r"\b(?:mean|average|peak|maximum|centroid)\b", reference_statement, re.I)):
        return "QUANTITY_MATCH_CONFLICT: threshold/percentile is not a mean, extremum, or centroid"
    return None


def _adjacent_result_declaration(context: str, quotation: str) -> str:
    """Bind a split result heading without borrowing another result's context.

    Only whitespace/display delimiters may intervene. A numbered list, another
    quantity, or an intervening assertion stops the lookup. The declaration
    must explicitly end in a colon or 'is'; no arbitrary surrounding paragraph
    is used as a unit or quantity declaration.
    """
    bound = bind_quote(context, quotation)
    if bound is None:
        return quotation
    before = context[:bound['start']]
    lines = before.splitlines()
    for line in reversed(lines[-6:]):
        stripped = line.strip()
        if not stripped or stripped in {r'\[', r'\(', '$$', '$'}:
            continue
        plain = re.sub(r'[*`]', '', stripped)
        if (re.search(r'(?:\bis\s*:?|:)\s*$', plain, re.I)
                and re.search(r'coefficient|correlation|cosine', plain, re.I)):
            return plain + '\n' + quotation
        break
    return quotation


def dimensionless_unit_proof(value, quotation: str, reference_statement: str, reference_unit, *, context=""):
    """Named mathematical ratios cancel physical units; presentation scale does not.

    The unit belongs to the source quantity, independently of which candidate
    reference is being checked. This proves no method/quantity equivalence;
    that remains the separate semantic mapping. This rule never consults the
    reference number and never infers an SI data unit.
    """
    if reference_unit not in {"1", "dimensionless", "unitless"} or isinstance(value, (list, str, bool)) or value is None:
        return None
    quantity_text = _adjacent_result_declaration(context, quotation) if context else quotation
    cv_pattern = r"coefficient of variation|\bCV\b|\bCoV\b|standard deviation[^.]*divided by[^.]*mean"
    patterns = [("COEFFICIENT_OF_VARIATION", cv_pattern),
                ("PEARSON_CORRELATION", r"Pearson|correlation|\br(?:_[a-z])?\s*(?:\([^)]*\))?\s*(?:=|≈)"),
                ("COSINE_ALIGNMENT", r"signed alignment coefficient|local cosines?|cosine (?:alignment|coefficient)")]
    quantity = next((name for name, source_pattern in patterns if re.search(source_pattern, quantity_text, re.I)), None)
    if (quantity is None and context and re.search(cv_pattern, reference_statement, re.I)
            and re.search(r'\b(?:nonuniformity|heterogeneity) coefficient\b', quantity_text, re.I)
            and re.search(r'\bcoefficient of variation\b', context, re.I)):
        quantity = 'SOURCE_NAMED_NONUNIFORMITY_CV'
    if quantity is None and context and re.search(cv_pattern, reference_statement, re.I):
        # The standard-deviation/absolute-mean notation states a dimensionless
        # ratio. Require the reported symbol and exact numeric literal too;
        # the nearby mean and standard deviation do not inherit that unit.
        definitions = re.finditer(
            r'(?<![\w])([A-Za-z](?:_[A-Za-z])?)\s*=\s*\\sigma\s*/\s*\|\s*\\mu\s*\|', context)
        for declaration in definitions:
            result = re.search(r'(?<![\w])' + re.escape(declaration[1]) + r'\s*=\s*', numeric_notation(quotation))
            if result:
                literal = _NUMBER.match(numeric_notation(quotation), result.end())
                if literal and Decimal(literal.group().replace(',', '')) == Decimal(str(value)):
                    quantity = 'SOURCE_DEFINED_STD_OVER_MEAN_SYMBOL'
                    break
    if quantity is None and context and re.search(cv_pattern, reference_statement, re.I):
        # A named CV may be defined on an earlier line and reported by symbol.
        # Bind the SAME symbol to an explicit sigma/mean definition, and only
        # accept the last result of its equality chain (not either operand).
        definition = re.compile(
            r"(?:coefficient of variation|\bCV\b|\bCoV\b)[^\n.;]{0,60}?"
            r"\b([A-Za-z](?:_[A-Za-z])?)\s*=\s*\\sigma_([A-Za-z])\s*/\s*\|\\(?:mu|bar)_?\2\|", re.I)
        for declared in definition.finditer(context):
            result = re.search(r"(?<![\w])" + re.escape(declared[1]) +
                               r"\s*=\s*([0-9eE+.,/ =−\-]+)", quotation)
            if result:
                final = result[1].rsplit("=", 1)[-1].strip()
                if _NUMBER.fullmatch(final) and Decimal(final.replace(",", "")) == Decimal(str(value)):
                    quantity = "SOURCE_DEFINED_CV_SYMBOL"
                    break
    if quantity is None:
        # A directly printed sigma_X / |mean_X| ratio has cancelling units.
        # Require the same field/subscript and this exact reported value after
        # the equality; a neighboring mean or a ratio of different fields is
        # not granted a unit by the presence of the formula elsewhere.
        ratio = re.compile(r"\\frac\s*\{\s*\\sigma_\{([^{}]+)\}\s*\}\s*\{\s*\|\s*\\bar(\\[A-Za-z]+|[A-Za-z])(?:_\{([^{}]+)\}|_([A-Za-z]+))?\s*\|\s*\}\s*=\s*")
        normalized = numeric_notation(quotation)
        # Braces around the mean symbol and numeric font commands are TeX
        # presentation only. Keep the field/subscript and literal checks below.
        normalized = re.sub(r'\\bar\{(\\[A-Za-z]+|[A-Za-z])\}',
                            lambda m: r'\bar' + m[1], normalized)
        normalized = re.sub(r'\\(?:mathbf|mathrm|mathsf)\{([^{}]+)\}',
                            lambda m: m[1] if _NUMBER.fullmatch(m[1]) else m[0], normalized)
        for match in ratio.finditer(normalized):
            numerator = re.sub(r"[^A-Za-z0-9]", "", match[1])
            denominator = re.sub(r"[^A-Za-z0-9]", "", match[2] + (match[3] or match[4] or ""))
            literal = _NUMBER.match(normalized, match.end())
            if numerator == denominator and literal and Decimal(literal.group().replace(",", "")) == Decimal(str(value)):
                quantity = "SAME_FIELD_STD_OVER_MEAN_FORMULA"
                break
    if quantity is None:
        return None
    # A quantity name or formula does not override a printed physical unit,
    # or two occurrences of the same value with incompatible scales.
    scale_proof = implicit_dimensionless_scale(value, quotation, reference_unit)
    if scale_proof is None:
        return None
    text = quotation.replace("−", "-")
    scales = []
    for match in _NUMBER.finditer(text):
        if Decimal(match.group().replace(",", "")) != Decimal(str(value)):
            continue
        suffix = text[match.end():].lstrip(" *$_}")
        scales.append("%" if suffix.startswith("%") else "1")
    if not scales or len(set(scales)) != 1:
        return None
    return {"source_unit": scale_proof['source_unit'], "rule": quantity + "_DIMENSIONLESS_FROM_SOURCE",
            "reference_unit": reference_unit}


def implicit_dimensionless_scale(value, quotation: str, reference_unit, *, unit_declaration=""):
    """Omitted units on a semantically matched dimensionless quantity mean 1.

    This is a unit convention, not a numerical correctness or semantic match
    decision. Percent suffixes still determine scale. Physical unit suffixes,
    ambiguous occurrences and separate ambiguous unit declarations prevent
    this fallback. No physical unit can be inherited from a reference.
    """
    if reference_unit not in {'1','dimensionless','unitless'} or type(value) not in {int,float}:
        return None
    if unit_declaration and not unit_is_bound('1',unit_declaration):
        return None
    text=numeric_notation(quotation)
    text=re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}',r' \1 ',text)
    scales=[]
    physical=r'(?:Pa|kPa|MPa|bar|m|mm|cm|km|s|ms|kg|g|K|A|V|W|J|Hz)\b'
    for match in _NUMBER.finditer(text):
        if Decimal(match.group().replace(',',''))!=Decimal(str(value)):
            continue
        suffix=text[match.end():].lstrip(' *$_}')
        suffix=re.sub(r'^(?:\\[,;! ]\s*)+','',suffix)
        if suffix.startswith('%') or suffix.startswith(r'\%'):
            scales.append('%')
        elif re.match(physical,suffix) or re.match(r'\b(?:percent|percentage|per\s+cent)\b',suffix,re.I):
            # Written "percent" is also an explicit scale, not an omission.
            if re.match(r'\b(?:percent|percentage|per\s+cent)\b',suffix,re.I):scales.append('%')
            else:return None
        else:
            scales.append('1')
    if len(set(scales))!=1:
        return None
    return {'source_unit':scales[0],'reference_unit':reference_unit,
            'rule':'IMPLICIT_DIMENSIONLESS_UNIT_WITH_EXPLICIT_SOURCE_SCALE'}


def operationalization_match_uncertainty(quotation: str, reference_statement: str):
    """Withhold equivalence when the reference leaves a decisive detail open.

    Finite values can contain zeros. Neither numerical agreement nor a judge's
    assertion establishes equivalence to a nonzero-only quantile. This guard
    does not decide whether the open choice is valid or inspect dataset values.
    Explicit zero exclusion is left to the semantic reviewer.
    """
    # Averaging vector norms and taking the norm of an averaged vector do not
    # commute. The generic vertex-values clause does not specify that order.
    # An explicit mean of vertex magnitudes cannot establish equivalence from
    # this reference wording alone. This is uncertainty, not method invalidity.
    if (re.search(r'arithmetic means? of vertex values for point fields', reference_statement, re.I)
            and re.search(r'\b(?:mean|average)\s+(?:of|over)\s+(?:its\s+|the\s+)?'
                          r'(?:(?:eight|\d+)\s+)?(?:vertex|point)\s+magnitudes\b', quotation, re.I)
            and not re.search(r'\b(?:norm|magnitude)\b', reference_statement, re.I)):
        return 'VECTOR_REDUCTION_ORDER_UNRESOLVED'
    quantile = r"\b(?:quantile|percentile|decile)\b"
    positive = r"\b(?:non[- ]?zero|positive)[- ]+(?:recorded[- ]+)?(?:speed|values?|samples?|observations?)\b"
    if not (re.search(quantile, reference_statement, re.I)
            and re.search(positive, reference_statement, re.I)
            and re.search(quantile, quotation, re.I)):
        return None
    finite_population = re.search(
        r"\b(?:all\s+)?finite\s+(?:recorded\s+)?(?:speed\s+)?(?:values?|samples?|observations?)\b",
        quotation, re.I)
    explicit_exclusion = (re.search(positive, quotation, re.I)
        or re.search(r"\b(?:exclud\w*|remov\w*|discard\w*|without)\s+(?:the\s+)?zeros?\b", quotation, re.I)
        or re.search(r"\b(?:speed|values?)\s*>\s*0\b", quotation, re.I))
    if finite_population and not explicit_exclusion:
        return "FINITE_QUANTILE_POPULATION_NOT_ESTABLISHED_AS_NONZERO"
    return None


def operationalization_match_conflict(quotation: str, reference_statement: str):
    """Reject an explicit method conflict before numeric grading.

    This is a representation guard, not an assessment of whether either method
    is scientifically valid. The open-method rubric judges validity separately.
    """
    if ("absolute signed cell volume from VTK" in reference_statement
            and re.search(r"absolute\s+(?:tetrahedral|tetrahedron|tetrahedra)\s+volumes?", quotation, re.I)
            and not re.search(r"vtkCellSizeFilter|cross.check", quotation, re.I)):
        return "TETRA_VOLUME_VS_ORIGINAL_VTK_CELL_VOLUME"
    if re.search(r'\b(?:six|6)[ -]neighbor\b', reference_statement, re.I):
        # Six-neighbor graph edges may use any of the three axes. Holding two
        # coordinates fixed throughout a component instead produces 1-D runs.
        # Do not mistake the usual "each edge differs along one axis" wording
        # for this explicit restriction of an entire connected component.
        if re.search(r'\b(?:six|6)[ -]neighbor\b', quotation, re.I):
            # A quote describing both alternatives needs semantic attribution;
            # a keyword guard cannot decide which one supplied the result.
            return None
        held = re.search(r'\bneighbors?\s+while\s+holding\s+([xyz])\s+and\s+([xyz])\s+fixed\b',
                         quotation, re.I)
        separate = (re.search(r'\bconnectivity\s+separately\s+along\s+each\s+(?:single\s+)?grid\s+axis\b', quotation, re.I)
                    and re.search(r'\bno\s+(?:diagonal\s+or\s+)?cross-axis\s+connections\b', quotation, re.I))
        if (held and held[1].lower() != held[2].lower()) or separate:
            return "SINGLE_AXIS_RUNS_VS_SIX_NEIGHBOR_COMPONENTS"
    return None


def rounding_radius(value, quotation: str, *, allow_integer=False, min_digits=3):
    """Uncertainty of a printed decimal, not permission to widen GT tolerance."""
    if isinstance(value, (list, str, bool)) or value is None:
        return None
    radii = []
    text = numeric_notation(quotation)
    for match in _NUMBER.finditer(text):
        literal = match.group().replace(",", "")
        number = Decimal(literal)
        if number != Decimal(str(value)) or len(number.as_tuple().digits) < min_digits or (not allow_integer and "." not in literal and "e" not in literal.lower()):
            continue
        # Exactness must qualify this number. A sentence such as "This is
        # exactly why absolute weights are used" says nothing about precision.
        before = re.sub(r'[*`$]', '', text[:match.start()])
        after = re.sub(r'[*`$]', '', text[match.end():])
        if (re.search(r'\bexact(?:ly)?\s*(?:(?:value|result|sum|total|coefficient)\s*)?(?:(?:is|of|equals)\s*|[=:]\s*)?$', before, re.I)
                or re.match(r'\s*(?:\(exact(?:ly)?\)|exact(?:ly)?\b)', after, re.I)):
            return None
        radii.append(float(Decimal(5).scaleb(number.as_tuple().exponent - 1)))
    return min(radii) if radii else None
