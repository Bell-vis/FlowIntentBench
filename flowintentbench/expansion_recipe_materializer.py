"""Bind supported construction recipes to CaseEvaluator's existing O callback.

Exact authored clauses select their already declared recipe. Other wording or
parameters need an independent O-only recipe binding. Executed G(O) still goes
through CaseEvaluator's separate scientific adjudication; execution is never
scientific acceptance. Unrepresentable choices remain coverage gaps.
"""
from __future__ import annotations

import copy
import math
import importlib.metadata
from pathlib import Path
from types import SimpleNamespace

from .construction_recipes import computed_findings
from .runtime_construction_recipes import (
    _normalize_point_mixture_range_recipe, _validate_point_statistics_controls,
    execute_runtime_recipes as execute_recipes)
from .deterministic_materialization import _reader_dataset
from .evaluator import EvaluationPendingAdjudication
from .external_file_evaluator import digest, write_json
from .expansion_evaluation import file_sha256, read_json
from .recipe_binding_validation import recipe_from_binding, request_supported_recipe_binding


def cached_recipe_result(directory, identity, compute):
    """Reuse only independently executed numeric facts, never scientific verdicts."""
    from scripts.portable_fcntl import fcntl
    import json
    identity = json.loads(json.dumps(identity, allow_nan=False))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = digest(identity)
    path = directory / (key + '.json')
    with (directory / (key + '.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            cached = read_json(path)
            checksum = cached.pop('artifact_sha256', None)
            if checksum != digest(cached) or cached.get('identity') != identity:
                raise ValueError('Numeric execution cache identity or digest mismatch')
            return cached['result']
        result = compute()
        cached = dict(identity=identity, result=result)
        cached['artifact_sha256'] = digest(cached)
        write_json(path, cached)
        return result

OVERRIDES = {
    "dispersion": {"measure": ("cv", "relative_interdecile"),
                   "sampling": ("cell_volume", "point_equal"), "point_domain": ("all", "finite"),
                   "ddof": (0, 1), "cv_denominator": ("mean", "absolute_mean"),
                   "relative_denominator": ("median", "absolute_median"),
                   "quantile_method": ("linear", "nearest_rank")},
    "association": {"measure": ("pearson", "spearman"),
                    "sampling": ("cell_volume", "point_equal"), "point_domain": ("all", "finite")},
    "upper_tail": {"quantile": "unit_interval", "quantile_weighting": ("cell_volume", "unweighted"),
                   "quantile_method": ("linear", "nearest_rank", "from_execution")},
    "weighted_width": {"axis": (0, 1, 2), "measure": ("rms", "interdecile", "range"),
                       "sampling": ("cell_volume", "point_equal"),
                       "endmembers": "endmember_rule_or_finite_increasing_pair",
                       "tail_planes": "positive_integer", "tail_bounds": "finite_increasing_pair",
                       "clip_mixture": "boolean", "mixture_mask": "unit_interval_increasing_pair",
                       "center_weight": ("mixture", "density", "uniform"), "center_domain": ("all", "selected")},
    "advective_flux": {"product_rule": ("cell_then_product", "point_then_cell")},
    "vector_alignment": {"measure": ("mean_cosine", "normalized_dot")},
    "surface": {"level": "finite_number"},
}

OVERRIDE_SEMANTICS = {
    "point_equal": "Association, dispersion, or weighted_width range on stored point fields only: equal point weights, no cell averaging, reconstruction, interpolation, ROI, subsampling, or face extrapolation. Vector components or magnitudes are computed at each stored point before any statistics. Field specifications remain fixed; a source finite_cell_fields domain is unsupported. For weighted_width range, explicit endmembers, inclusive mixture_mask, center_weight and center_domain are required. clip_mixture must be explicit except when clipping is provably inactive under the center rule below; in that case omit the unspecified control, do not invent a clipping choice.",
    "point_statistics": "For association/dispersion only, point_domain=all (default) requires every stored sample finite; point_domain=finite explicitly selects finite reduced scalar samples, with one joint finite-pair mask for association before ranking or correlation. Both cover the complete supplied point mesh and require finite coordinates. Pearson uses ordinary equal-weight correlation (matching sample or population covariance/std normalization); Spearman uses ordinary average midranks for ties then Pearson. Constant fields are undefined. Do not bind ROI, cell-to-point reconstruction, alternative ranks, or nonuniform weights.",
    "point_dispersion": "For point_equal CV, ddof must explicitly be 0 (population) or 1 (sample), and cv_denominator explicitly mean or absolute_mean. For relative_interdecile, quantile_method must explicitly be linear (NumPy linear interpolation) or nearest_rank (left empirical cumulative inverse), and relative_denominator explicitly median or absolute_median. Compute (q90-q10)/denominator using that same estimator for the median. Never infer missing denominator, ddof, or quantile conventions. Controls for the other measure or cell sampling are unsupported; zero denominators are undefined.",
    "endmembers": "sample_extrema, tail_plane_medians (pool points on the lowest/highest tail_planes distinct coordinates along axis), tail_coordinate_medians (pool points at coordinate <= tail_bounds[0] and >= tail_bounds[1]), or an explicit increasing [low, high] pair. Tail medians are computed from data, not supplied result numbers. Fixed candidate normalization constants may use the explicit pair; do not substitute reported findings for choices.",
    "center": "Range is max-minus-min stored coordinate in the inclusive mixture_mask. Center uses independent center_weight (mixture=4c(1-c), density=stored scalar, uniform=1) and center_domain (all points or selected mask). Clipping occurs before mask and mixture weights. Never infer center domain or weight. Only when 0 < mixture_mask[0] < mixture_mask[1] < 1, center_weight is density or uniform, and center_domain is selected, omitted clip_mixture is supported: finite normalization has the same selected points and center with either clipping choice. Execution uses the unclipped formula and records this inactive-control normalization. Boundary masks, all-domain centers, mixture centers, nonfinite normalization and unknown controls do not qualify.",
    "unweighted_percentile": "quantile_weighting=unweighted changes selection only; absolute/signed volume policy and volume-normalized summaries remain the source recipe's. quantile_method must be candidate-explicit linear/nearest_rank or from_execution with supplied authenticated numerical-convention evidence. Reported threshold numbers are not evidence of interpolation and never replace percentile recomputation.",
}


def _valid_override(value, rule):
    if isinstance(rule, tuple):
        return not isinstance(value, bool) and value in rule
    if rule == "boolean":
        return isinstance(value, bool)
    if rule == "endmember_rule_or_finite_increasing_pair" and isinstance(value, str):
        return value in {"sample_extrema", "tail_plane_medians", "tail_coordinate_medians"}
    if rule in {"finite_increasing_pair", "unit_interval_increasing_pair", "endmember_rule_or_finite_increasing_pair"}:
        return (isinstance(value, (list, tuple)) and len(value) == 2
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
                and value[0] < value[1]
                and (rule != "unit_interval_increasing_pair" or 0 <= value[0] < value[1] <= 1))
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if rule == "positive_integer":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    return valid and (rule != "unit_interval" or 0 <= value <= 1)


def _validate_recipe_controls(recipe):
    """Reject ignored controls and incomplete opt-in scientific choices."""
    if recipe["kind"] in {"association", "dispersion"}:
        _validate_point_statistics_controls(recipe)
    if recipe["kind"] == "upper_tail":
        if recipe.get("quantile_weighting", "cell_volume") == "unweighted":
            if recipe.get("quantile_method") not in {"linear", "nearest_rank"}:
                raise ValueError("unweighted percentile requires an explicit or evidenced estimator")
        elif recipe.get("quantile_method", "nearest_rank") != "nearest_rank":
            raise ValueError("volume percentile requires the left cumulative inverse")
    if recipe["kind"] == "weighted_width":
        point_controls = {"tail_planes", "tail_bounds", "clip_mixture", "mixture_mask", "center_weight", "center_domain"}
        if recipe.get("sampling", "cell_volume") != "point_equal":
            if point_controls & recipe.keys() or recipe["measure"] == "range":
                raise ValueError("point mask/range controls require point_equal sampling")
            if isinstance(recipe.get("endmembers"), str) and recipe["endmembers"] != "sample_extrema":
                raise ValueError("tail endmembers require point_equal sampling")
        else:
            _normalize_point_mixture_range_recipe(recipe)


def coverage_gap(bundle, reason):
    raise EvaluationPendingAdjudication(reason, pending_type="evaluator_coverage_gap",
        continuation_context={"execution_status": "EVALUATOR_COVERAGE_GAP",
                              "candidate_operationalization": bundle.model_dump(mode="json"),
                              "required_next_action": "ADD_AND_VERIFY_SUPPORTED_EXECUTION_ROUTE"})


def build_expansion_recipe_materializer(root, metadata, material, transport, *, execution_trajectory=None):
    root = Path(root)
    definition = material.get("family_definition")

    def materialize(bundle):
        if not definition:
            evidence_source = material.get("reference_materialization_evidence")
            if evidence_source:
                evidence_path = root / evidence_source["path"]
                if file_sha256(evidence_path) != evidence_source["sha256"]:
                    coverage_gap(bundle, "Preserved reference execution evidence changed")
                from .qualified_scientific_cases import build_agent_ready_novel_o_materializer
                from .deterministic_materialization import MaterializationUnsupported
                try:
                    return build_agent_ready_novel_o_materializer(root, metadata, read_json(evidence_path),
                                                                execution_trajectory=execution_trajectory)(bundle)
                except MaterializationUnsupported as exc:
                    coverage_gap(bundle, str(exc))
            coverage_gap(bundle, "No expansion recipe binding for this preserved reference family")
        effective = {d.dimension.value: d.statement for d in bundle.decisions}
        exact = next((name for name, operation in definition["operations"].items()
                      if operation["clauses"] == effective), None)
        if exact is not None:
            binding = {"source_operation": exact, "parameter_overrides": {},
                       "semantic_binding_rationale": "Exact authored operation clauses"}
            try:
                recipe, convention_evidence = recipe_from_binding(definition["operations"], binding)
            except ValueError as exc:
                coverage_gap(bundle, str(exc))
        else:
            from .execution_evidence import observed_quantile_method
            numerical_evidence = observed_quantile_method(effective, execution_trajectory)
            payload = {
                "candidate_operationalization": bundle.model_dump(mode="json"),
                "scientific_target": metadata.scientific_target,
                "operations": definition["operations"], "supported_overrides": OVERRIDES,
                "override_semantics": OVERRIDE_SEMANTICS,
                "observed_quantile_method": numerical_evidence,
                "domain_policy": {key: definition[key] for key in ("cell_volume_convention", "finite_cell_fields") if key in definition},
                "instruction": "Bind only if a source recipe with supported overrides exactly implements every explicit candidate O clause. Never fill missing model decisions. Fields and domain policy stay fixed; sampling and numerical conventions may change only through documented supported overrides. Preserve candidate normalization constants when they are fixed choices; recompute data-derived estimates from their specified procedure. An unweighted percentile estimator must be explicit in O or use from_execution with supplied numerical evidence. Reported thresholds and Findings are never execution evidence. Unknown support or ambiguous choices are PENDING, not scientific INVALID. Return source_operation, parameter_overrides, semantic_binding_rationale; no findings or G(O)."}
            schema = {"type": "object", "additionalProperties": False,
                    "required": ["source_operation", "parameter_overrides", "semantic_binding_rationale"],
                    "properties": {"source_operation": {"type": "string", "enum": list(definition["operations"])},
                        "parameter_overrides": {"type": "object"}, "semantic_binding_rationale": {"type": "string"}}}
            try:
                recipe, convention_evidence = request_supported_recipe_binding(transport, payload, schema)
            except ValueError as exc:
                coverage_gap(bundle, str(exc))
        runtime_definition = copy.deepcopy(definition)
        runtime_definition["operations"] = {"runtime": {"recipe": recipe, "clauses": effective}}
        manifest_path = root / "datasets" / metadata.dataset_id / "dataset_manifest.json"
        dataset_manifest = read_json(manifest_path)
        plan = {"adapter": "expansion-recipe-materializer-v1", "recipe": recipe,
                "domain_policy": {key: definition[key] for key in ("cell_volume_convention", "finite_cell_fields") if key in definition},
                "recipe_implementation_sha256": file_sha256(Path(__file__).with_name("construction_recipes.py")),
                "runtime_recipe_implementation_sha256": file_sha256(Path(__file__).with_name("runtime_construction_recipes.py"))}
        if (recipe["kind"] == "weighted_width" and recipe.get("sampling") == "point_equal"
                and "clip_mixture" not in recipe):
            plan["inactive_control_normalization"] = {
                "control": "clip_mixture", "execution_value": False,
                "reason": "strictly interior mask and selected density/uniform center are clipping-invariant",
                "runtime_requirement": "finite mixture normalization"}
        if convention_evidence:
            plan["numerical_convention_evidence"] = convention_evidence
        identity = {"case_id": metadata.case_id, "bundle_id": bundle.operationalization_id,
                    "family_definition_sha256": digest(definition),
                    "effective_o": effective, "plan": plan,
                    "dataset_manifest_sha256": file_sha256(manifest_path)}
        cache_id = digest(identity)
        cache = transport.directory / "materializations" / f"{cache_id}.json"
        for item in dataset_manifest["files"]:
            if item.get("checksum_algorithm") != "sha256" or not item.get("checksum"):
                coverage_gap(bundle, "Dataset bytes lack a supported frozen checksum")
            path = root / dataset_manifest.get("file_root", "datasets") / item["path"]
            if file_sha256(path) != item["checksum"]:
                coverage_gap(bundle, "Dataset bytes differ from frozen manifest")
        if cache.is_file():
            artifact = read_json(cache)
            body = dict(artifact)
            supplied = body.pop("artifact_sha256", None)
            if supplied != digest(body):
                raise ValueError("materialization cache digest mismatch")
            return artifact
        try:
            numeric_identity = dict(
                version='independent-recipe-numerics-v1',
                dataset_manifest_sha256=file_sha256(manifest_path),
                definition=runtime_definition, plan=plan,
                reader_sha256=file_sha256(Path(__file__).with_name('deterministic_materialization.py')),
                materializer_sha256=file_sha256(Path(__file__)),
                packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'vtk')})

            def compute():
                dataset, _ = _reader_dataset(root, metadata.dataset_id)
                return execute_recipes(dataset, runtime_definition)['runtime']

            result = cached_recipe_result(transport.directory / 'numerical_materializations',
                                         numeric_identity, compute)
            findings, _ = computed_findings(SimpleNamespace(definition=runtime_definition),
                                             metadata, bundle.operationalization_id, result)
        except Exception as exc:
            coverage_gap(bundle, f"Supported recipe did not materialize: {type(exc).__name__}: {exc}")
        artifact = {"parameters": effective,
                    "materialization_id": "expansion-recipe:" + cache_id,
                    "findings": [f.model_dump(mode="json") for f in findings],
                    "execution": {"status": "MATERIALIZED", "reproducible": True,
                        "G_of_O": result, "data_provenance": {
                            "dataset_id": metadata.dataset_id,
                            "dataset_manifest_sha256": file_sha256(manifest_path),
                            "effective_o_sha256": digest(effective),
                            "materialization_plan": plan,
                            "compiled_plan_sha256": digest(plan), "executed_plan_sha256": digest(plan),
                            "silent_substitutions": [], "unsupported_dimensions": []}}}
        artifact["artifact_sha256"] = digest(artifact)
        write_json(cache, artifact)
        return artifact

    return materialize
