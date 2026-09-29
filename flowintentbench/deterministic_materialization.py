"""Deterministic, data-mediated materialization for agent-ready construction.

This module contains only numerical execution helpers.  It deliberately does
not decide scientific validity or reference membership; those decisions stay
in the SRAC/GT contracts.  A materializer receives a complete Effective O,
reads the dataset through the manifest reader contract, and returns a plain
execution record consumed by :func:`flowintentbench.srac.materialize_branch`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .vtk_reader import apply_legacy_vtk_reader_configuration
from .plot3d_reader import apply_plot3d_reader_configuration


VECTOR_NAMES = {
    "Blunt_Fin": "Velocity",
    "Carotid": "vectors",
    "FireFlow": "uvw",
    "Kitchen": "velocity",
    "NASA_LOx_Post": "Velocity",
    "Office": "vectors",
}


class MaterializationUnsupported(ValueError):
    """Raised when an Effective O cannot be executed without guessing."""

    def __init__(self, dimension: str, requested: str, reason: str | None = None) -> None:
        self.dimension = dimension
        self.requested = requested
        self.reason = reason or "no exact deterministic handler is registered"
        super().__init__(f"{dimension}: {requested} ({self.reason})")


@dataclass(frozen=True)
class MaterializationPlan:
    """Internal, typed execution plan compiled from an Effective O.

    This is deliberately an implementation IR.  It is not serialized into
    GroundTruth or the benchmark case schema.  The executor receives this
    plan only after every scientific choice has been recognized exactly.
    """

    domain: str
    criterion_kind: str | None = None
    # Comparator is part of the executable IR.  It is intentionally absent
    # for operations (for example an isosurface level) where the criterion is
    # not a point-wise inequality.  A materializer must never replace an
    # authored comparator with a default.
    comparator: str | None = None
    quantile: float | None = None
    quantile_method: str = "linear"
    quantile_population: str = "positive"
    threshold: float | None = None
    connectivity_kind: str | None = None
    measure_kind: str | None = None
    representation_kind: str | None = None
    metric_kind: str | None = None
    aggregation_kind: str | None = None
    component_selection_kind: str | None = None
    standard_deviation_multiplier: float | None = None
    standard_deviation_ddof: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve archived plan identities when the new statistical
        # criterion is not used.
        for key in ("standard_deviation_multiplier", "standard_deviation_ddof"):
            if value[key] is None:
                value.pop(key)
        return value


def _flatten_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        values: list[str] = []
        for key, item in value.items():
            values.extend(_flatten_values(key))
            values.extend(_flatten_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_values(item))
        return values
    return [str(value)]


def _effective_text(effective_o: Mapping[str, Any]) -> str:
    return " ".join(_flatten_values(effective_o)).casefold()


def _has_explicit_tie_break(text: str) -> bool:
    """Detect authored tie semantics that encounter-order cannot replace."""

    return bool(
        re.search(r"\b(?:break(?:ing)?|resolve)\b.{0,24}\bties?\b", text)
        or re.search(r"\bif\b.{0,32}\bties?\b", text)
        or re.search(r"\btie[_ -]?break(?:er|ing)?\b", text)
    )


def _parse_quantile(text: str, *, observed_method: str | None = None) -> tuple[float, str] | None:
    """Parse an authored quantile and require an explicit interpolation rule.

    A percentile is a scientific operationalization choice.  Falling back to
    NumPy's default interpolation when the question merely says ``90th
    percentile`` would make the materializer choose an answer that was not
    authored.  ``median`` remains the conventional 0.5 quantile for legacy
    density cases; all other percentile/quantile forms must name either linear
    interpolation or nearest-rank.
    """
    explicit_linear = bool(
        re.search(r"\blinear(?:ly)?\b.{0,32}\binterpol(?:ation|ated|ate)\b", text)
        or re.search(r"\binterpol(?:ation|ated|ate)\b.{0,32}\blinear(?:ly)?\b", text)
    )
    explicit_nearest = bool(
        re.search(r"\bnearest[- ]rank\b", text)
        or re.search(r"\bnearest\s+rank\b", text)
    )
    upper_volume = re.search(r"(?:upper|top)\s*(\d+(?:\.\d+)?)\s*%\s*(?:of\s+(?:the\s+)?)?(?:domain\s+)?volume", text)
    percentile = re.search(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", text)
    if percentile is None:
        percentile = re.search(r"\bp_?\{?(\d+(?:\.\d+)?)\}?(?![\d.])", text)
    if upper_volume:
        value = 1 - float(upper_volume.group(1)) / 100
    elif percentile:
        value = float(percentile.group(1)) / 100.0
    else:
        quantile = re.search(r"\bq\s*([0-1](?:\.\d+)?|\d{1,3}(?:\.\d+)?)\b", text)
        if quantile:
            raw = float(quantile.group(1))
            value = raw / 100.0 if raw > 1 else raw
        elif re.search(r"\bmedian\b", text):
            value = 0.5
        else:
            quantile = re.search(r"\bquantile\b[^0-9]*(0?\.\d+|\d+(?:\.\d+)?)", text)
            if not quantile:
                return None
            raw = float(quantile.group(1))
            value = raw / 100.0 if raw > 1 else raw
    if not 0 < value < 1:
        raise MaterializationUnsupported("criterion", str(value), "quantile must lie strictly between zero and one")
    if explicit_nearest and explicit_linear:
        raise MaterializationUnsupported(
            "criterion", text, "quantile interpolation cannot specify both linear and nearest-rank"
        )
    if explicit_nearest:
        method = "nearest_rank"
    elif explicit_linear:
        method = "linear"
    elif re.search(r"\bmedian\b", text):
        # Median is retained as the established 0.5 convention for the
        # density-surface cases.  Percentiles and general q/quantile clauses
        # do not receive this compatibility default.
        method = "linear"
    elif observed_method in {"linear", "nearest_rank"}:
        method = observed_method
    else:
        raise MaterializationUnsupported(
            "criterion", text, "quantile interpolation must be explicitly linear or nearest-rank"
        )
    return value, method


def _parse_speed_population(text: str, *, required: bool = True) -> str | None:
    """Bind declared speed samples, never a percentile's reported cutoff.

    ``finite`` means the reader's valid points, including zero speeds.  A
    positive/nonzero restriction is equivalent for speed magnitudes.  Match
    sample nouns so unrelated positive quantities (e.g. density) cannot
    silently change the population.  Consume restricted noun phrases before
    looking for unrestricted ones: "all finite nonzero speeds" is one
    positive population, whereas "all finite speeds and nonzero speeds"
    declares two incompatible populations.
    """
    sample = (
        r"(?:(?:stored|grid[- ]point|grid|point|nodal)\s+)*"
        r"(?:speeds?|velocity(?:[- ]?magnitude)?s?|values?|samples?|locations?|points?|nodes?)\b"
    )
    if re.search(
        r"\b(?:in|within|inside|outside|from|over|restricted to|limited to)\s+"
        r"(?:(?:the|a|an)\s+)?(?:(?:selected|specified|custom)\s+)?"
        r"(?:roi|region of interest|region|mask|subdomain|subset)\b"
        r"|\b(?:custom|unknown|user[- ]defined)\s+(?:support|population)\b"
        r"|\b(?:volume|area|density)[- ]weighted\b"
        r"|\b(?:masked|unmasked|fluid[- ]only|interior|boundary)\s+"
        + r"(?:(?:finite|valid)\s+)*" + sample,
        text,
    ):
        raise MaterializationUnsupported("criterion", text, "requested speed population support has no exact executor")
    modifiers = r"(?:(?:all|the|finite|valid|stored)\s+)*"
    positive_phrase = re.compile(
        r"\b" + modifiers + r"(?:non[- ]?zero|(?:strictly\s+)?positive)\s+"
        + modifiers + sample
        + r"|\b(?:non[- ]?zero|positive)\s+(?:(?:global|empirical)\s+)?"
        + r"(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?\s+percentile|p\d+(?:\.\d+)?\b)"
        + r"|\b" + modifiers + sample
        + r"\s+(?:excluding|omitting|without)\s+(?:the\s+)?zero(?:[- ]speed)?(?:\s+(?:values|samples|points|locations|speeds))?\b"
    )
    positive = bool(positive_phrase.search(text))
    remaining = positive_phrase.sub(" ", text)
    # Explicit exclusion may be a separate population clause.
    zero_samples = r"zero(?:s|(?:[- ]speed)?(?:\s+(?:speeds?|values?|samples?|points?|locations?))?)\b"
    if re.search(r"\b(?:not|never)\s+(?:includ\w*|exclud\w*)\s+" + zero_samples
                 + r"|\b" + zero_samples + r"\s+(?:(?:are|is)\s+)?not\s+(?:included|excluded)\b", text):
        raise MaterializationUnsupported("criterion", text, "negated speed population support has no exact executor")
    excludes_zero = bool(re.search(
        r"\b(?:exclud\w*|omitt?\w*|without)\s+(?:the\s+)?" + zero_samples
        + r"|\b" + zero_samples + r"\s+(?:(?:are|is)\s+)?(?:excluded|omitted)\b", remaining))
    includes_zero = bool(re.search(
        r"\binclud\w*\s+(?:the\s+)?" + zero_samples
        + r"|\b" + zero_samples + r"\s+(?:(?:are|is)\s+)?included\b", remaining))
    finite = bool(
        re.search(r"\b(?:finite|valid)\s+" + sample, remaining)
        or re.search(r"\b(?:all|every)\s+" + modifiers + sample, remaining)
        or re.search(r"\bdomain[- ]wide\s+" + modifiers + sample, remaining)
        or includes_zero
    )
    if (positive and finite) or ((positive or excludes_zero) and includes_zero):
        raise MaterializationUnsupported("criterion", text, "conflicting speed population specifications")
    if positive or excludes_zero:
        return "positive"
    if finite:
        return "finite"
    if required:
        raise MaterializationUnsupported(
            "criterion", text,
            "speed population must explicitly specify valid/all finite samples or nonzero/positive samples",
        )
    return None


def compile_effective_operationalization(
    effective_o: Mapping[str, Any],
    *,
    dataset_id: str | None = None,
    observed_quantile_method: str | None = None,
    observed_threshold_statistics: Mapping[str, Any] | None = None,
) -> MaterializationPlan:
    """Compile a complete Effective O without silent semantic substitution."""

    text = _effective_text(effective_o)
    by_dimension = {
        str(key).casefold(): _effective_text({"value": value})
        for key, value in effective_o.items()
    }
    criterion_text = by_dimension.get("criterion", text)
    feature_text = by_dimension.get("feature_definition", text)
    property_text = by_dimension.get("property_measure", "")
    representation_text = by_dimension.get("aggregation_or_representation", text)
    measure_text = " ".join(
        part for part in (criterion_text, property_text, representation_text) if part
    )
    representation_search_text = " ".join(
        part for part in (representation_text, property_text, criterion_text) if part
    )
    if dataset_id == "Combustor" and "density" in text and "surface" in text:
        density_criterion_text = " ".join((criterion_text, feature_text, representation_text))
        density_quantile, density_quantile_method, density_level = None, "linear", None
        fixed_levels = re.findall(r"(?:\brho|ρ|\bdensity(?:\s+isosurface)?(?:\s+level)?)\s*(?:=|of|at)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", density_criterion_text)
        if fixed_levels:
            levels = {float(item) for item in fixed_levels}
            if len(levels) != 1:
                raise MaterializationUnsupported("criterion", criterion_text, "conflicting fixed Density levels")
            criterion_kind, density_level = "density_fixed", levels.pop()
        elif "90%-of-q90" in text or "90% of q90" in text:
            criterion_kind = "density_q90_fraction"
            density_quantile, density_quantile_method = 0.90, "linear"
        elif parsed_density_quantile := _parse_quantile(density_criterion_text, observed_method=observed_quantile_method):
            criterion_kind = "density_quantile"
            density_quantile, density_quantile_method = parsed_density_quantile
        else:
            raise MaterializationUnsupported("criterion", criterion_text, "Density level is not explicit")
        if not any(token in text for token in ("connected", "shared-edge", "shared edge", "triangles sharing an edge")):
            raise MaterializationUnsupported("feature_definition", feature_text, "isosurface connectivity is not explicit")
        if re.search(r"(?:surround|enclos|contain).{0,60}global density maximum", text):
            component_selection = "encloses_global_density_maximum"
        elif re.search(
            r"(?:largest|greatest|maximum).{0,32}(?:bounding[- ]box volume|bounds volume)",
            text,
        ):
            component_selection = "largest_bounding_box_volume"
        elif re.search(
            # Inside the already-established constant-Density surface domain,
            # "largest area" is an explicit surface-area ranking, not an
            # inferred scientific choice.  Current construction projections
            # use both phrasings and must compile to the same exact plan.
            r"(?:largest|greatest|maximum).{0,32}(?:surface )?area",
            text,
        ):
            component_selection = "largest_surface_area"
        elif re.search(r"\b(?:unique|single|only|one)\s+(?:connected\s+)?(?:component|contour|surface)\b", text):
            component_selection = "unique_connected_component"
        else:
            raise MaterializationUnsupported(
                "aggregation_or_representation",
                representation_text,
                "component selection must explicitly specify a supported ranking, containment, or unique connected component",
            )
        if re.search(r"enclosed[- ]volume centroid|volume[- ]weighted centroid|volume centroid", text):
            representation = "enclosed_volume_centroid"
        elif any(token in text for token in ("area-weighted centroid", "area weighted centroid", "area-weighted triangle-centroid")):
            representation = "area_weighted_centroid"
        elif any(
            token in text
            for token in (
                "bounding-box midpoint",
                "bounding box midpoint",
                "bounds midpoint",
                "box center",
                "box centre",
            )
        ):
            representation = "bounding_box_midpoint"
        else:
            raise MaterializationUnsupported(
                "aggregation_or_representation",
                representation_text,
                "representative location must explicitly use area-weighted centroid or bounding-box midpoint",
            )
        if _has_explicit_tie_break(text):
            raise MaterializationUnsupported(
                "criterion",
                criterion_text,
                "the requested tie-breaking rule is not implemented by this exact materializer",
            )
        return MaterializationPlan(
            domain="density_isosurface",
            criterion_kind=criterion_kind,
            threshold=density_level,
            comparator=None,
            quantile=density_quantile,
            quantile_method=density_quantile_method,
            connectivity_kind="polydata_connected",
            measure_kind=(
                "surface_area"
                if component_selection == "largest_surface_area"
                else "global_density_maximum_containment"
                if component_selection == "encloses_global_density_maximum"
                else "connected_component_count"
                if component_selection == "unique_connected_component"
                else "bounding_box_volume"
            ),
            representation_kind=representation,
            aggregation_kind=component_selection,
            component_selection_kind=component_selection,
        )
    if dataset_id == "Kitchen" and any(token in text for token in ("concentration", "heterogeneity", "gas-concentration")):
        select_minimum = bool(re.search(r"\b(?:smallest|lowest|minimum|least)\b.{0,60}\b(?:cv|coefficient|heterogeneity|variation|deviation)\b", text) or "most spatially homogeneous" in text)
        field_selection = "minimum_metric" if select_minimum else None
        if re.search(r"total[- ]variation", text):
            if not ("normalized" in text and "trilinear" in text
                    and any(t in text for t in ("standard deviation", "sigma", "σ"))
                    and any(t in text for t in ("characteristic", "cube root", "1/3", "⅓"))
                    and "volume" in text):
                raise MaterializationUnsupported("property_measure", text, "normalized total variation requires explicit volume, standard-deviation, domain-length normalization and trilinear interpolation")
            return MaterializationPlan(domain="kitchen_concentration",
                                       metric_kind="normalized_total_variation",
                                       aggregation_kind="trilinear_volume_integral",
                                       component_selection_kind=field_selection)
        if "gradient" in property_text or "∇" in property_text:
            raise MaterializationUnsupported("property_measure", property_text, "the requested gradient statistic has no exact executor")
        if "coefficient of variation" in text or re.search(r"\bcv\b", text):
            metric = "weighted_coefficient_of_variation" if "cell-volume" in text or "volume-weighted" in text else "coefficient_of_variation"
        elif "relative interdecile" in text or "interdecile" in text:
            metric = "weighted_relative_interdecile_spread" if "cell-volume" in text or "volume-weighted" in text else "relative_interdecile_spread"
        elif (
            ("root-mean-square" in text or "root mean square" in text or re.search(r"\brms\b", text))
            and any(token in text for token in ("adjacent", "neighbor", "neighbour", "face"))
        ):
            metric = "rms_face_adjacent_difference"
        elif any(token in text for token in ("normalized mean absolute neighbor", "normalized mean absolute neighbouring", "normalized mean absolute neighboring", "normalized mean adjacent", "normalized adjacent", "neighbor contrast", "neighbour contrast", "adjacent contrast")):
            metric = "normalized_mean_face_adjacent_difference"
        elif any(token in text for token in ("mean absolute neighbor", "mean absolute neighbouring", "mean absolute neighboring", "mean adjacent difference", "mean absolute adjacent", "mean of the edgewise absolute", "mean of edgewise absolute", "mean absolute concentration difference", "mean absolute concentration contrast", "mean adjacent concentration difference", "average absolute difference", "mean over all 6-connected", "mean over all 6-connected grid-edge", "mean over all included 6-neighbor", "mean over all included face", "mean edgewise absolute")):
            metric = "mean_face_adjacent_difference"
        elif "standard deviation" in text or re.search(r"\bstd\b", text):
            metric = "weighted_standard_deviation" if "cell-volume" in text or "volume-weighted" in text else "population_standard_deviation"
        else:
            raise MaterializationUnsupported("property_measure", text, "Kitchen heterogeneity measure is not supported")
        aggregation = "cell_volume_weighted" if "cell-volume" in text or "volume-weighted" in text else "point_equal_weight"
        return MaterializationPlan(domain="kitchen_concentration", metric_kind=metric, aggregation_kind=aggregation,
                                   component_selection_kind=field_selection)

    cell_speed_requested = any(t in text for t in ("cell-averaged speed", "cell averaged speed", "cell mean speed", "cell speed", "cell-speed", "cell-centered speed", "cell-centred speed", "cell-centered velocity", "cell-centred velocity"))
    if cell_speed_requested:
        vector_mean = bool(re.search(r"(?:averag|mean).{0,65}(?:corner|nodal|point).{0,35}(?:velocity\s+vectors?|vectors?)", text) and re.search(r"norm|magnitude", text))
        scalar_mean = bool(re.search(r"(?:point|nodal)[- ](?:speed|velocity magnitude)", text) and any(t in text for t in ("averag", "mean")))
        if re.search(r"volume[- ]weighted.{0,30}(?:rms|root[- ]mean[- ]square)", text):
            cell_measure = "volume_weighted_rms"
        elif re.search(r"volume[- ]weighted.{0,20}(?:mean|average).{0,20}speed", text):
            cell_measure = "volume_weighted_mean"
        else:
            cell_measure = None
        if not (any(t in text for t in ("face-connected", "face connected", "shared faces", "shared-face"))
                and cell_measure is not None
                and (any(t in text for t in ("volume centroid", "volume-weighted centroid", "volume weighted centroid"))
                     or ("centroid" in representation_text and "volume" in representation_text and "weight" in representation_text))
                and (vector_mean or scalar_mean)):
            raise MaterializationUnsupported("feature_definition", text, "cell-speed execution requires an explicit corner speed/vector aggregation, shared-face connectivity, volume-weighted RMS or mean, and volume centroid")
        quantile = _parse_quantile(" ".join((criterion_text, feature_text)), observed_method=observed_quantile_method)
        if quantile is None:
            raise MaterializationUnsupported("criterion", criterion_text, "cell-speed percentile must be explicit")
        volume_quantile = bool(re.search(r"(?:upper|top)\s*\d+(?:\.\d+)?\s*%\s*(?:of\s+(?:the\s+)?)?(?:domain\s+)?volume", criterion_text) or re.search(r"volume[- ]weighted\s+(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?\s+)?percentile", criterion_text))
        if volume_quantile and quantile[1] != "nearest_rank":
            raise MaterializationUnsupported("criterion", criterion_text, "volume-weighted percentile requires an evidenced inverse-ECDF convention")
        if re.search(r"at or above|>=|≥|at least", criterion_text):
            comparator = "GE"
        elif re.search(r"exceed|above|greater than|>", criterion_text):
            comparator = "GT"
        else:
            raise MaterializationUnsupported("criterion", criterion_text, "cell-speed comparator must be explicit")
        return MaterializationPlan(domain="cell_speed_region", criterion_kind="quantile",
                                   comparator=comparator, quantile=quantile[0], quantile_method=quantile[1],
                                   quantile_population="volume_weighted" if volume_quantile else "positive" if "non-zero" in text or "nonzero" in text else "finite",
                                   connectivity_kind="cell_shared_face", measure_kind=cell_measure,
                                   representation_kind="volume_centroid", aggregation_kind="magnitude_of_mean_point_velocity" if vector_mean else "mean_of_point_speeds",
                                   component_selection_kind="max_" + cell_measure)

    # A complete Effective O may state the threshold in the feature
    # definition (for example, "regions whose speed is at the 95th
    # percentile") and the component-selection rule in ``criterion``.  Both
    # are explicit parts of the same proposal; looking only at one field
    # would incorrectly reject an otherwise executable proposal.  We still
    # require an unambiguous numeric/quantile expression and never invent a
    # default threshold.
    criterion_search_text = " ".join((criterion_text, feature_text))
    # Normalize only mathematical comparison glyphs; these are exact textual
    # aliases, not inferred scientific choices.
    criterion_search_text = (
        criterion_search_text.replace("≥", " at or above ")
        .replace("≤", " at or below ")
        .replace("≧", " at or above ")
        .replace("≦", " at or below ")
    )
    criterion_search_text = re.sub(r"\\+ge(?:q)?\b", " >= ", criterion_search_text)
    formula_text = criterion_search_text.replace("\\", "")
    mean_std = re.search(r"(?:mu(?:_\{?[a-z]\}?)?|μ|mean)\s*\+\s*(\d+(?:\.\d+)?)\s*\*?\s*(?:sigma(?:_\{?[a-z]\}?)?|σ|standard deviation)", formula_text)
    # Prefer an explicitly authored numeric cutoff even when the sentence
    # also explains that it corresponds to a percentile.  The negative
    # lookahead prevents "at or above the 95th percentile" from being
    # misread as the fixed scalar threshold 95.
    threshold_match = re.search(
        r"(?P<operator>strictly\s+greater\s+than|greater\s+than\s+or\s+equal\s+to|"
        r"greater\s+than|at\s+or\s+above|"
        r"at\s+least|above|>=|strictly\s+less\s+than|less\s+than|at\s+or\s+below|"
        r"at\s+most|below|<=|>|<)\s*(?:the\s+)?(?P<value>-?\d+(?:\.\d+)?)"
        r"(?!\d)(?!(?:st|nd|rd|th)?\s*percentile)",
        criterion_search_text,
    )
    std_multiplier, std_ddof = None, None
    if mean_std:
        std_multiplier = float(mean_std.group(1))
        proof = observed_threshold_statistics
        if (not proof or proof.get("multiplier") != std_multiplier
                or proof.get("ddof") not in {0, 1} or proof.get("population") != "all_speed_points"):
            raise MaterializationUnsupported("criterion", criterion_text, "mean-plus-standard-deviation requires executed ddof and population evidence")
        if _parse_speed_population(criterion_search_text, required=False) == "positive":
            raise MaterializationUnsupported("criterion", criterion_text, "declared statistical population conflicts with full-array execution evidence")
        if not any(token in criterion_search_text for token in (">=", "at or above", "at least")):
            raise MaterializationUnsupported("criterion", criterion_text, "statistical threshold comparison is not explicit")
        criterion_kind, criterion_quantile, quantile_method, threshold, comparator = "mean_plus_std", None, "linear", None, "GE"
        std_ddof = int(proof["ddof"])
    elif threshold_match:
        criterion_kind = "fixed_threshold"
        criterion_quantile, quantile_method, threshold = None, "linear", float(threshold_match.group("value"))
        operator = re.sub(r"\s+", " ", threshold_match.group("operator").casefold()).strip()
        comparator = {
            "strictly greater than": "GT",
            "greater than or equal to": "GE",
            "greater than": "GT",
            "above": "GT",
            ">=": "GE",
            ">": "GT",
            "<": "LT",
            "at or above": "GE",
            "at least": "GE",
            "strictly less than": "LT",
            "less than": "LT",
            "below": "LT",
            "<=": "LE",
            "at or below": "LE",
            "at most": "LE",
        }[operator]
    else:
        quantile = _parse_quantile(criterion_search_text, observed_method=observed_quantile_method)
        if not quantile:
            raise MaterializationUnsupported("criterion", criterion_text, "criterion is not explicit")
        criterion_kind = "quantile"
        criterion_quantile, quantile_method = quantile
        threshold = None
        comparator = "GE"

    connectivity_text = " ".join((feature_text, criterion_text, representation_text))
    if re.search(r"(?<!\d)26[- ](?:neighbou?r|connected|connectivity)\b", feature_text):
        connectivity = "structured_full_26"
    elif re.search(r"(?<!\d)18[- ](?:neighbou?r|connected|connectivity)\b", feature_text):
        connectivity = "structured_edge_18"
    elif (
        re.search(r"(?<!\d)6-neighbou?r\b", feature_text)
        or re.search(r"\bsix-neighbou?r\b", feature_text)
        or re.search(r"(?<!\d)6-connected\b", feature_text)
        or re.search(r"(?<!\d)6-connectivity\b", feature_text)
        or "face-connected" in feature_text
        or "face-sharing" in feature_text
        or "shared grid face" in feature_text
        or "shared grid faces" in feature_text
        or "shared face" in feature_text
        or "shared faces" in feature_text
        or "face adjacency" in feature_text
        or "face-adjacency" in feature_text
    ):
        connectivity = "structured_face_6"
    elif any(
        token in connectivity_text
        for token in (
            "mesh-connected",
            "mesh connected",
            "mesh connectivity",
            "mesh adjacency",
            "cell-connected",
            "shared unstructured-mesh cells",
            "same unstructured-mesh cell",
            "same unstructured mesh cell",
            "same mesh cell",
            "cell-sharing",
            "cell sharing",
            "shared mesh cell",
            "shared unstructured-mesh cell",
            "shared unstructured mesh cell",
            "share an unstructured-mesh cell",
            "share an unstructured mesh cell",
            "sharing an unstructured-mesh cell",
            "sharing an eight-node hexahedral mesh cell",
            "unstructured-mesh cell adjacency",
            "unstructured mesh cell adjacency",
            "cell adjacency",
        )
    ):
        connectivity = "mesh_cell_point"
    else:
        raise MaterializationUnsupported("feature_definition", feature_text, "connectivity semantics are not explicit")

    if any(token in property_text for token in ("integrated excess", "excess-speed integral", "excess speed integral", "above-threshold speed integral", "above threshold speed integral")):
        measure = "integrated_excess_speed"
    elif re.search(r"\bsum\s+of\s+(?:the\s+)?(?:retained\s+)?speeds\b", property_text):
        measure = "speed_sum"
    elif any(
        token in measure_text
        for token in (
            "peak speed",
            "maximum speed",
            "max speed",
            "within-region maximum",
            "maximum velocitymagnitude",
            "maximum velocity magnitude",
            "greatest peak",
            "largest peak",
            "peak value",
            "region maximum",
            "largest pointwise",
            "maximum pointwise flow speed",
            "maximum q value",
            "largest maximum",
            "maximum-speed",
            "maximum speed point",
            "maximum speed node",
            "max speed point",
            "max speed node",
            "maximal speed",
        )
    ) or bool(re.search(r"\bmax(?:imum)?\b.{0,32}\b(?:speed|velocitymagnitude|velocity magnitude)\b", measure_text)):
        measure = "peak"
    elif any(
        token in measure_text
        for token in (
            "mean speed",
            "mean velocity magnitude",
            "mean of velocity magnitudes",
            "mean of speed",
            "mean of pointwise speed",
            "average speed",
            "arithmetic mean",
            "largest mean",
            "greatest mean",
            "mean-speed",
        )
    ):
        measure = "mean"
    else:
        raise MaterializationUnsupported("property_measure", measure_text, "region strength measure is not explicit")

    # Prefer an explicit representation clause over words inherited from the
    # property-measure clause.  For example, "maximum speed" can select the
    # region while "mean spatial coordinates" still specifies its location.
    if "excess" in representation_text and "weighted" in representation_text and any(t in representation_text for t in ("centroid", "center", "centre")):
        representation = "excess_speed_weighted_centroid"
    elif (re.search(r"(?:speed|velocity[- ]magnitude)[- ]weighted", representation_text)
          or ("velocity-weighted" in representation_text and ("velocity magnitude" in text or "velocitymagnitude" in text))) and any(t in representation_text for t in ("centroid", "center", "centre")):
        representation = "speed_weighted_centroid"
    elif re.search(r"\bweighted\b", representation_text) and any(t in representation_text for t in ("centroid", "center", "centre")):
        raise MaterializationUnsupported("aggregation_or_representation", representation_text, "weighted location convention is not implemented")
    elif (
        bool(re.search(r"\b(?:mean|average)\b.{0,70}\b(?:cartesian |spatial |physical )?(?:coordinates|positions|locations)\b", representation_text))
        or "mean spatial location" in representation_text
        or "mean spatial position" in representation_text
        or "mean spatial coordinates" in representation_text
        or "mean cartesian coordinate" in representation_text
        or "mean coordinate" in representation_text
        or "centroid" in representation_text
    ):
        representation = "mean"
    elif any(token in representation_search_text for token in ("peak-point", "peak point", "peak location", "peak grid index", "maximizing point", "argmax", "peak-speed location", "point attaining", "corresponding grid-point", "corresponding grid point", "maximum-speed point", "maximum speed point", "maximum-speed node", "maximum speed node", "max-speed point", "max speed point", "max-speed node", "max speed node", "location of the maximum", "location of its maximum", "location where the maximum", "coordinates where the maximum", "coordinates from", "grid-index location")) or bool(re.search(r"\b(?:cartesian )?coordinates?\b.{0,48}\b(?:maximum|maximizing|argmax|peak)\b", representation_search_text)):
        representation = "peak"
    elif (
        "mean spatial location" in representation_search_text
        or "mean spatial coordinates" in representation_search_text
        or "mean cartesian coordinate" in representation_search_text
        or "mean coordinate" in representation_search_text
        or "centroid" in representation_search_text
    ):
        representation = "mean"
    else:
        raise MaterializationUnsupported("aggregation_or_representation", representation_text, "location representation is not explicit")

    if _has_explicit_tie_break(text):
        raise MaterializationUnsupported(
            "criterion",
            criterion_text,
            "the requested tie-breaking rule is not implemented by this exact materializer",
        )

    return MaterializationPlan(
        domain="high_speed_region",
        criterion_kind=criterion_kind,
        comparator=comparator,
        quantile=criterion_quantile,
        quantile_method=quantile_method,
        quantile_population=(
            _parse_speed_population(criterion_search_text)
            if criterion_kind == "quantile"
            else "finite" if criterion_kind == "mean_plus_std" else "positive"
        ),
        threshold=threshold,
        connectivity_kind=connectivity,
        measure_kind=measure,
        representation_kind=representation,
        standard_deviation_multiplier=std_multiplier,
        standard_deviation_ddof=std_ddof,
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reader_dataset(root: Path, dataset_id: str) -> tuple[Any, dict[str, Any]]:
    """Read one dataset using the repository's declared reader settings."""

    from vtkmodules.vtkIOParallel import vtkMultiBlockPLOT3DReader
    from vtkmodules.vtkIOLegacy import vtkDataSetReader
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

    dataset_root = root / "datasets" / dataset_id
    manifest = _json(dataset_root / "dataset_manifest.json")
    file_root = root / str(manifest.get("file_root", "datasets"))
    config = manifest.get("reader", {}).get("reader_configuration", {})
    fmt = str(manifest.get("reader", {}).get("format", ""))
    if fmt == "VTK":
        reader = vtkDataSetReader()
        apply_legacy_vtk_reader_configuration(reader, config)
        flow = next(item for item in manifest["files"] if item.get("role") == "flow_field")
        reader.SetFileName(str(file_root / flow["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif fmt == "VTU":
        reader = vtkXMLUnstructuredGridReader()
        flow = next(item for item in manifest["files"] if item.get("role") == "flow_field")
        reader.SetFileName(str(file_root / flow["path"]))
        reader.Update()
        dataset = reader.GetOutput()
    elif fmt == "PLOT3D":
        reader = vtkMultiBlockPLOT3DReader()
        apply_plot3d_reader_configuration(reader, config)
        grid = next(item for item in manifest["files"] if item.get("role") == "grid")
        solution = next(item for item in manifest["files"] if item.get("role") == "solution")
        reader.SetXYZFileName(str(file_root / grid["path"]))
        reader.SetQFileName(str(file_root / solution["path"]))
        reader.Update()
        output = reader.GetOutput()
        dataset = output.GetBlock(0) if output is not None else None
    else:
        raise ValueError(f"unsupported deterministic materialization format: {fmt}")
    if dataset is None:
        raise RuntimeError(f"{dataset_id} canonical reader returned no dataset")
    return dataset, manifest


def _flow_arrays(root: Path, dataset_id: str) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, tuple[int, int, int] | None]:
    from vtk.util.numpy_support import vtk_to_numpy

    dataset, manifest = _reader_dataset(root, dataset_id)
    point_data = dataset.GetPointData()
    preferred = VECTOR_NAMES.get(dataset_id)
    vectors = point_data.GetArray(preferred) if preferred else None
    if vectors is None:
        vectors = point_data.GetVectors()
    if vectors is None:
        for index in range(point_data.GetNumberOfArrays()):
            candidate = point_data.GetArray(index)
            if candidate is not None and candidate.GetNumberOfComponents() == 3:
                vectors = candidate
                break
    if vectors is None:
        raise RuntimeError(f"{dataset_id} has no three-component velocity array")
    values = vtk_to_numpy(vectors).astype(np.float64, copy=False)
    speed = np.linalg.norm(values, axis=1)
    valid = np.isfinite(speed)
    iblank = point_data.GetArray("IBlank")
    if iblank is not None and iblank.GetNumberOfTuples() == len(valid):
        valid &= vtk_to_numpy(iblank) > 0
    dimensions: tuple[int, int, int] | None = None
    # The VTK object is authoritative for executable topology.  Manifest
    # metadata remains provenance and is checked when it makes a claim, but a
    # missing ``grid.type`` must not erase structured-grid semantics.
    is_structured = any(
        bool(getattr(dataset, "IsA", lambda _name: False)(name))
        for name in ("vtkStructuredGrid", "vtkStructuredPoints", "vtkImageData", "vtkRectilinearGrid")
    )
    manifest_grid_type = manifest.get("grid", {}).get("type") if isinstance(manifest.get("grid"), Mapping) else None
    if manifest_grid_type == "structured" and not is_structured:
        raise RuntimeError(f"{dataset_id} manifest declares structured data but reader returned {dataset.GetClassName()}")
    if is_structured:
        values3 = [0, 0, 0]
        dataset.GetDimensions(values3)
        dimensions = tuple(int(value) for value in values3)
        if int(np.prod(dimensions)) != len(speed):
            raise RuntimeError(f"{dataset_id} dimensions do not match point count")
    return dataset, manifest, speed, valid, dimensions


def _adjacency(dataset: Any, count: int) -> list[list[int]]:
    import vtk

    result: list[set[int]] = [set() for _ in range(count)]
    point_ids = vtk.vtkIdList()
    for cell_index in range(dataset.GetNumberOfCells()):
        dataset.GetCellPoints(cell_index, point_ids)
        points = [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]
        for left_index, left in enumerate(points):
            for right in points[left_index + 1 :]:
                result[left].add(right)
                result[right].add(left)
    return [sorted(items) for items in result]


def _regions(
    mask: np.ndarray,
    dimensions: tuple[int, int, int] | None,
    dataset: Any,
    connectivity_kind: str,
) -> list[np.ndarray]:
    visited = np.zeros(mask.size, dtype=bool)
    regions: list[np.ndarray] = []
    if connectivity_kind == "structured_face_6" and dimensions is None:
        raise MaterializationUnsupported("feature_definition", connectivity_kind, "dataset has no verifiable structured topology")
    adjacency = _adjacency(dataset, mask.size) if connectivity_kind == "mesh_cell_point" else None
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    if connectivity_kind in {"structured_full_26", "structured_edge_18"}:
        from itertools import product
        max_distance = 3 if connectivity_kind == "structured_full_26" else 2
        offsets = tuple(delta for delta in product((-1, 0, 1), repeat=3)
                        if 0 < sum(abs(v) for v in delta) <= max_distance)

    def neighbors(index: int):
        if connectivity_kind == "mesh_cell_point":
            yield from adjacency[index]  # type: ignore[index]
            return
        if dimensions is None:
            raise MaterializationUnsupported("feature_definition", connectivity_kind, "structured dimensions are unavailable")
        nx, ny, nz = dimensions
        z, remainder = divmod(index, nx * ny)
        y, x = divmod(remainder, nx)
        for dx, dy, dz in offsets:
            xx, yy, zz = x + dx, y + dy, z + dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                yield xx + nx * (yy + ny * zz)

    for start in np.flatnonzero(mask):
        start = int(start)
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        region: list[int] = []
        while stack:
            current = stack.pop()
            region.append(current)
            for neighbor in neighbors(current):
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        regions.append(np.asarray(region, dtype=int))
    return regions


def _point_scalar(dataset: Any, name: str) -> np.ndarray:
    from vtk.util.numpy_support import vtk_to_numpy

    array = dataset.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"expected Kitchen scalar array {name!r} is unavailable")
    return vtk_to_numpy(array).astype(float, copy=False)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(np.interp(quantile, cumulative, values[order]))


def _select_unique_max(scores: np.ndarray, *, dimension: str) -> int:
    """Select a unique maximum, failing closed on an un-authored tie.

    Encounter order is not a scientific tie rule.  If the authored Effective
    O did not specify one (explicit tie wording is rejected at compile time),
    an exact tie must remain unsupported rather than silently selecting the
    first component/field.
    """

    if scores.size == 0:
        raise MaterializationUnsupported(dimension, "empty candidate set", "no candidates are available")
    maximum = np.max(scores)
    winners = np.flatnonzero(scores == maximum)
    if len(winners) != 1:
        raise MaterializationUnsupported(
            dimension,
            "maximum",
            "multiple candidates share the maximum and no authored tie rule is available",
        )
    return int(winners[0])


def _kitchen_cell_values(dataset: Any, scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from vtk.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersVerdict import vtkCellSizeFilter

    size_filter = vtkCellSizeFilter()
    size_filter.SetInputData(dataset)
    size_filter.SetComputeVolume(True)
    size_filter.Update()
    volumes = vtk_to_numpy(size_filter.GetOutput().GetCellData().GetArray("Volume")).astype(float)
    means = np.empty(dataset.GetNumberOfCells(), dtype=float)
    for cell_id in range(dataset.GetNumberOfCells()):
        ids = dataset.GetCell(cell_id).GetPointIds()
        means[cell_id] = float(np.mean([scalar[ids.GetId(index)] for index in range(ids.GetNumberOfIds())]))
    valid = np.isfinite(volumes) & (volumes > 0) & np.isfinite(means)
    return volumes[valid], means[valid]


def _kitchen_point_metric(dataset: Any, scalar: np.ndarray, metric_kind: str, dimensions: tuple[int, int, int] | None) -> float:
    values = np.asarray(scalar, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise MaterializationUnsupported("property_measure", metric_kind, "field has no finite values")
    if metric_kind == "population_standard_deviation":
        return float(np.std(values, ddof=0))
    if metric_kind in {
        "rms_face_adjacent_difference",
        "mean_face_adjacent_difference",
        "normalized_mean_face_adjacent_difference",
    }:
        if dimensions is None or int(np.prod(dimensions)) != scalar.size:
            raise MaterializationUnsupported("property_measure", metric_kind, "face-adjacent metric requires structured point topology")
        nx, ny, nz = dimensions
        grid = np.asarray(scalar, dtype=float).reshape((nz, ny, nx))
        differences: list[np.ndarray] = []
        for axis in range(3):
            difference = np.diff(grid, axis=axis)
            differences.append(difference[np.isfinite(difference)])
        finite_differences = np.concatenate(differences) if differences else np.asarray([], dtype=float)
        if finite_differences.size == 0:
            raise MaterializationUnsupported("property_measure", metric_kind, "no finite adjacent differences")
        absolute_differences = np.abs(finite_differences)
        if metric_kind == "rms_face_adjacent_difference":
            return float(np.sqrt(np.mean(np.square(finite_differences))))
        mean_absolute = float(np.mean(absolute_differences))
        if metric_kind == "normalized_mean_face_adjacent_difference":
            finite_values = values[np.isfinite(values)]
            value_range = float(np.max(finite_values) - np.min(finite_values))
            return 0.0 if value_range == 0 else mean_absolute / value_range
        return mean_absolute
    raise MaterializationUnsupported("property_measure", metric_kind, "point metric is not supported")


def analyze_kitchen_concentration(root: Path, effective_o: Mapping[str, Any], operation_id: str, *, _compiled_plan: MaterializationPlan | None = None) -> dict[str, Any]:
    """Execute the reviewed Kitchen FIELD heterogeneity operation."""

    dataset, manifest = _reader_dataset(root, "Kitchen")
    plan = _compiled_plan or compile_effective_operationalization(effective_o, dataset_id="Kitchen")
    fields = ("c1", "c11", "c12", "c13", "c14", "c15", "c16", "c17")
    if plan.metric_kind == "normalized_total_variation":
        from .trilinear_variation import materialize_trilinear_variation
        return materialize_trilinear_variation(dataset, manifest, fields, effective_o, plan, operation_id)
    metrics: list[dict[str, Any]] = []
    dimensions: tuple[int, int, int] | None = None
    if any(bool(getattr(dataset, "IsA", lambda _name: False)(name)) for name in ("vtkStructuredGrid", "vtkStructuredPoints", "vtkImageData", "vtkRectilinearGrid")):
        values3 = [0, 0, 0]
        dataset.GetDimensions(values3)
        dimensions = tuple(int(value) for value in values3)
    for field_name in fields:
        volumes, values = _kitchen_cell_values(dataset, _point_scalar(dataset, field_name))
        mean = float(np.average(values, weights=volumes))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=volumes)))
        q10 = _weighted_quantile(values, volumes, 0.10)
        q50 = _weighted_quantile(values, volumes, 0.50)
        q90 = _weighted_quantile(values, volumes, 0.90)
        scalar = _point_scalar(dataset, field_name)
        point_metric = _kitchen_point_metric(dataset, scalar, plan.metric_kind, dimensions) if plan.metric_kind in {"population_standard_deviation", "rms_face_adjacent_difference", "mean_face_adjacent_difference", "normalized_mean_face_adjacent_difference"} else None
        if plan.metric_kind == "weighted_coefficient_of_variation":
            metric_value = std / max(abs(mean), 1e-30)
        elif plan.metric_kind == "coefficient_of_variation":
            finite_scalar = scalar[np.isfinite(scalar)]
            metric_value = float(np.std(finite_scalar, ddof=0) / max(abs(float(np.mean(finite_scalar))), 1e-30))
        elif plan.metric_kind == "weighted_relative_interdecile_spread":
            metric_value = (q90 - q10) / max(abs(mean), 1e-30)
        elif plan.metric_kind == "relative_interdecile_spread":
            finite_scalar = scalar[np.isfinite(scalar)]
            metric_value = float((np.quantile(finite_scalar, 0.90) - np.quantile(finite_scalar, 0.10)) / max(abs(float(np.mean(finite_scalar))), 1e-30))
        elif plan.metric_kind == "weighted_standard_deviation":
            metric_value = std
        else:
            metric_value = float(point_metric)
        metrics.append({
            "field_name": field_name,
            "mean": mean,
            "standard_deviation": std,
            "coefficient_of_variation": std / max(abs(mean), 1e-30),
            "relative_interdecile_spread": (q90 - q10) / max(abs(mean), 1e-30),
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "cell_count": int(len(values)),
            "metric_value": metric_value,
        })
    selected_index = _select_unique_max(
        np.asarray([item["metric_value"] for item in metrics], dtype=float) * (-1 if plan.component_selection_kind == "minimum_metric" else 1),
        dimension="property_measure",
    )
    selected = metrics[selected_index]
    result = {
        "operation_id": operation_id,
        "field_name": selected["field_name"],
        "metric_name": plan.metric_kind,
        "metric_value": float(selected["metric_value"]),
        "mean": float(selected["mean"]),
        "standard_deviation": float(selected["standard_deviation"]),
        "q10": float(selected["q10"]),
        "q50": float(selected["q50"]),
        "q90": float(selected["q90"]),
        "cell_count": int(selected["cell_count"]),
        "all_field_metrics": metrics,
    }
    return {
        "parameters": dict(effective_o),
        "result": result,
        "execution": {
            "status": "MATERIALIZED",
            "reproducible": True,
            "G_of_O": result,
            "data_provenance": {
                "dataset_id": "Kitchen",
                "dataset_manifest_sha256": _digest(manifest),
                "reader_format": manifest.get("reader", {}).get("format"),
            "reader_configuration": manifest.get("reader", {}).get("reader_configuration", {}),
            "input_files": [item.get("path") for item in manifest.get("files", []) if isinstance(item, Mapping)],
            "scalar_fields": list(fields),
            "aggregation": plan.aggregation_kind,
            "point_to_cell_rule": (
                "arithmetic mean of each cell's finite vertex values"
                if plan.aggregation_kind == "cell_volume_weighted"
                else None
            ),
            "weighted_quantile_rule": (
                "linear interpolation over cumulative positive cell-volume weights"
                if plan.aggregation_kind == "cell_volume_weighted"
                else None
            ),
            "materialization_plan": plan.to_dict(),
            "effective_o_sha256": _digest(effective_o),
            "compiled_plan_sha256": _digest(plan.to_dict()),
            "executed_plan_sha256": _digest(plan.to_dict()),
            "silent_substitutions": [],
            "unsupported_dimensions": [],
        },
        },
        "materialization_id": f"mat:{operation_id}",
    }


def analyze_high_speed(root: Path, dataset_id: str, effective_o: Mapping[str, Any], operation_id: str, *, observed_quantile_method: str | None = None, observed_threshold_statistics: Mapping[str, Any] | None = None, _compiled_plan: MaterializationPlan | None = None, _extra_thresholds=()) -> dict[str, Any]:
    """Execute one high-speed region Effective O on real dataset arrays."""

    if dataset_id == "Kitchen" and "concentration" in _effective_text(effective_o):
        return analyze_kitchen_concentration(root, effective_o, operation_id)

    plan = _compiled_plan or compile_effective_operationalization(effective_o, dataset_id=dataset_id, observed_quantile_method=observed_quantile_method, observed_threshold_statistics=observed_threshold_statistics)
    if plan.domain == "cell_speed_region":
        from .cell_speed_regions import analyze_cell_speed_regions
        return analyze_cell_speed_regions(root, dataset_id, effective_o, operation_id, plan)
    dataset, manifest, speed, valid, dimensions = _flow_arrays(root, dataset_id)
    if plan.domain != "high_speed_region":
        raise MaterializationUnsupported("feature_definition", plan.domain, "Effective O is not a high-speed region operation")
    if plan.connectivity_kind == "structured_face_6" and dimensions is None:
        raise MaterializationUnsupported("feature_definition", "structured_face_6", "reader did not expose structured dimensions")
    if plan.criterion_kind != "fixed_threshold":
        if plan.quantile_population not in {"finite", "positive"}:
            raise MaterializationUnsupported("criterion", str(plan.quantile_population), "unknown speed population in execution plan")
        if plan.criterion_kind == "mean_plus_std" and plan.quantile_population != "finite":
            raise MaterializationUnsupported("criterion", str(plan.quantile_population), "statistical threshold requires evidenced full-array population")
        population = speed[valid] if plan.quantile_population == "finite" else speed[valid & (speed > 0)]
        if population.size == 0:
            raise RuntimeError(f"{dataset_id} has no {plan.quantile_population} speed values")
    threshold_statistics = None
    if plan.criterion_kind == "mean_plus_std":
        if plan.standard_deviation_ddof not in {0, 1} or plan.standard_deviation_multiplier is None:
            raise MaterializationUnsupported("criterion", plan.criterion_kind, "incomplete statistical threshold plan")
        if not bool(np.all(valid)):
            raise MaterializationUnsupported("criterion", plan.criterion_kind, "executed full-array mean/std has no authorized nonfinite-value omission")
        mean = float(np.mean(population))
        std = float(np.std(population, ddof=plan.standard_deviation_ddof))
        threshold = mean + plan.standard_deviation_multiplier * std
        criterion = f">= mean + {plan.standard_deviation_multiplier:g} standard deviations ({threshold:.12g})"
        threshold_statistics = {"mean": mean, "standard_deviation": std, "ddof": plan.standard_deviation_ddof,
                                "population_count": int(population.size), "population": plan.quantile_population}
        comparator = plan.comparator or "GE"
        if comparator != "GE":
            raise MaterializationUnsupported("criterion", comparator, "statistical criterion supports explicit GE only")
        mask = valid & (speed >= threshold)
    elif plan.criterion_kind == "fixed_threshold":
        threshold = float(plan.threshold)  # type: ignore[arg-type]
        comparator = plan.comparator or "GT"
        symbol = {"GT": ">", "GE": ">=", "LT": "<", "LE": "<="}[comparator]
        criterion = f"{symbol} {threshold:.12g}"
        if comparator == "GT":
            selected_mask = speed > threshold
        elif comparator == "GE":
            selected_mask = speed >= threshold
        elif comparator == "LT":
            selected_mask = speed < threshold
        elif comparator == "LE":
            selected_mask = speed <= threshold
        else:  # pragma: no cover - plan construction constrains this value
            raise MaterializationUnsupported("criterion", comparator, "unknown comparator")
        mask = valid & selected_mask
    else:
        threshold = float(np.quantile(population, plan.quantile, method="inverted_cdf" if plan.quantile_method == "nearest_rank" else "linear"))  # type: ignore[arg-type]
        percentile = float(plan.quantile) * 100.0  # type: ignore[arg-type]
        population_label = "finite" if plan.quantile_population == "finite" else "non-zero"
        criterion = f">= {population_label} {percentile:g}th percentile ({threshold:.12f})"
        comparator = plan.comparator or "GE"
        if comparator != "GE":
            raise MaterializationUnsupported("criterion", comparator, "quantile comparator must be GE")
        mask = valid & (speed >= threshold)
    regions = _regions(mask, dimensions, dataset, plan.connectivity_kind or "structured_face_6")
    if not regions:
        raise RuntimeError(f"{dataset_id} operation {operation_id} retained no regions")
    point_volume = None
    if plan.measure_kind == "peak":
        scores = np.asarray([speed[region].max() for region in regions])
    elif plan.measure_kind == "speed_sum":
        scores = np.asarray([speed[region].sum() for region in regions])
    elif plan.measure_kind == "integrated_excess_speed":
        if not getattr(dataset, "IsA", lambda _: False)("vtkImageData"):
            raise MaterializationUnsupported("property_measure", plan.measure_kind, "excess-speed integral requires uniform point-grid volume")
        point_volume = float(abs(np.prod(dataset.GetSpacing())))
        scores = np.asarray([np.sum(speed[region] - threshold) * point_volume for region in regions])
    else:
        scores = np.asarray([speed[region].mean() for region in regions])
    selected_index = _select_unique_max(scores, dimension="component_selection")
    selected = regions[selected_index]
    coords = np.asarray([dataset.GetPoint(int(index)) for index in selected], dtype=float)
    peak_index = int(selected[np.argmax(speed[selected])])
    peak_point = list(dataset.GetPoint(peak_index))
    mean_location = coords.mean(axis=0).tolist()
    reported_location = peak_point if plan.representation_kind == "peak" else mean_location
    if plan.representation_kind == "speed_weighted_centroid":
        if np.any(speed[selected] < 0) or speed[selected].sum() <= 0:
            raise MaterializationUnsupported("aggregation_or_representation", plan.representation_kind, "speed weights must have a positive sum")
        reported_location = np.average(coords, axis=0, weights=speed[selected]).tolist()
    if plan.representation_kind == "excess_speed_weighted_centroid":
        excess = speed[selected] - threshold
        if np.any(excess < 0) or excess.sum() <= 0:
            raise MaterializationUnsupported("aggregation_or_representation", plan.representation_kind, "excess-speed centroid has no positive weight")
        reported_location = np.average(coords, axis=0, weights=excess).tolist()
    result = {
        "operation_id": operation_id,
        "criterion_kind": plan.criterion_kind,
        "criterion": criterion,
        "comparator": plan.comparator,
        "threshold": float(threshold),
        "measure_kind": plan.measure_kind,
        "representation": plan.representation_kind,
        "retained_points": int(mask.sum()),
        "region_count": len(regions),
        "region_sizes": [int(len(region)) for region in regions],
        "selected_region_size": int(len(selected)),
        "reported_location": [float(value) for value in reported_location],
        "mean_spatial_location": [float(value) for value in mean_location],
        "peak_point": [float(value) for value in peak_point],
        "coordinate_span": [float(value) for value in (coords.max(axis=0) - coords.min(axis=0))],
        "strength": float(scores[selected_index]),
        "peak_speed": float(speed[selected].max()),
        "mean_speed": float(speed[selected].mean()),
        "bounding_box": {axis: [float(coords[:, i].min()), float(coords[:, i].max())]
                         for i, axis in enumerate(("x", "y", "z"))},
        "region_rankings": [
            {"rank": rank + 1, "strength": float(scores[i]), "peak_speed": float(speed[regions[i]].max()),
             "mean_speed": float(speed[regions[i]].mean()), "point_count": int(len(regions[i]))}
            for rank, i in enumerate(np.argsort(-scores, kind="stable")[:32])
        ],
    }
    from .descriptive_statistics import region_statistics
    vectors = None
    if hasattr(dataset, "GetPointData"):
        from vtk.util.numpy_support import vtk_to_numpy
        array = dataset.GetPointData().GetArray(VECTOR_NAMES.get(dataset_id, ""))
        if array is not None:
            vectors = vtk_to_numpy(array)
    result["descriptive_statistics"] = region_statistics(speed, valid, selected, mask, vectors=vectors, thresholds=_extra_thresholds)
    if point_volume is not None:
        result["point_volume"] = point_volume
    if threshold_statistics is not None:
        result["threshold_statistics"] = threshold_statistics
    source_files = [item.get("path") for item in manifest.get("files", []) if isinstance(item, Mapping)]
    return {
        "parameters": dict(effective_o),
        "result": result,
        "execution": {
            "status": "MATERIALIZED",
            "reproducible": True,
            "G_of_O": result,
            "data_provenance": {
                "dataset_id": dataset_id,
                "dataset_manifest_sha256": _digest(manifest),
                "reader_format": manifest.get("reader", {}).get("format"),
                "reader_configuration": manifest.get("reader", {}).get("reader_configuration", {}),
            "input_files": source_files,
            "vector_array": VECTOR_NAMES.get(dataset_id),
            "materialization_plan": plan.to_dict(),
            "effective_o_sha256": _digest(effective_o),
            "compiled_plan_sha256": _digest(plan.to_dict()),
            "executed_plan_sha256": _digest(plan.to_dict()),
            "silent_substitutions": [],
            "unsupported_dimensions": [],
        },
        },
        "materialization_id": f"mat:{operation_id}",
    }


def supplemental_high_speed_evidence(root: Path, dataset_id: str, record: Mapping[str, Any], *, requested_thresholds=()) -> dict[str, Any] | None:
    """Re-execute a frozen recipe to supply extra facts, preserving frozen GT.

    The original dataset identity and all principal stored results must
    agree before the new bounding-box/ranking evidence is usable.
    """
    execution = record.get("execution", {})
    provenance = execution.get("data_provenance", {})
    plan_data = provenance.get("materialization_plan", {})
    old = execution.get("G_of_O", {})
    if plan_data.get("domain") != "high_speed_region" or not old:
        return None
    manifest = _json(root / "datasets" / dataset_id / "dataset_manifest.json")
    if provenance.get("dataset_manifest_sha256") != _digest(manifest):
        raise RuntimeError("supplemental execution dataset differs from frozen evidence")
    plan = MaterializationPlan(**plan_data)
    computed = analyze_high_speed(root, dataset_id, record.get("parameters", {}),
                                  str(old["operation_id"]), _compiled_plan=plan, _extra_thresholds=requested_thresholds)["result"]
    for key in ("threshold", "strength", "peak_speed", "reported_location", "mean_spatial_location",
                "retained_points", "selected_region_size", "region_count"):
        if key not in old or not np.allclose(old[key], computed[key], rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"supplemental execution disagrees with frozen {key}")
    return {**old, "bounding_box": computed["bounding_box"], "region_rankings": computed["region_rankings"],
            "descriptive_statistics": computed["descriptive_statistics"],
            "supplemental_execution_provenance": {"source_execution_sha256": _digest(execution),
                                                  "dataset_manifest_sha256": _digest(manifest),
                                                  "verified_frozen_results": True}}


__all__ = [
    "MaterializationPlan",
    "MaterializationUnsupported",
    "compile_effective_operationalization",
    "analyze_high_speed",
    "analyze_kitchen_concentration",
]
