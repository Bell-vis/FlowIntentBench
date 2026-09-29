"""Frozen, task-bound evaluation references. No analysis execution or network.

Supplement records are construction artifacts, not judge outputs. Hashes prove
identity; the named author/reviewer remains responsible for scientific validity.
"""
from __future__ import annotations

from copy import deepcopy
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .ground_truth import OperationalizationBundle, OperationalizationFindingBranch, ReferenceFinding
from .evaluation_policy import policy_index, VERIFICATION_MODES
from .finding_requirements import FindingRequirementContract
from .frozen_evidence import _pointer

VERSION = "evaluation-reference-package-v1"

# These are already computed construction outputs, never new analyses. The
# whitelist states the scientific identity; arbitrary JSON numbers are excluded.
MATERIALIZED_SUPPORT = {
    "threshold": ("Numerical selection threshold in the stored field scale", "quantity", "FIELD"),
    "cutoff": ("Numerical field threshold defining the selected region", "quantity", "FIELD"),
    "fraction": ("Fraction of total mesh volume occupied by the selected region", "quantity", "1"),
    "mean_speed": ("Mean speed of the selected high-speed region", "quantity", "FIELD"),
    "peak_point": ("Coordinates of the peak-speed point in the selected region", "location", None),
    "region_count": ("Number of connected regions in the retained mask", "quantity", "count"),
    "retained_points": ("Total number of retained high-speed grid points in all regions", "quantity", "count"),
    "selected_region_size": ("Number of retained grid points in the selected strongest region", "quantity", "count"),
}


def materialized_support_record(case_input, metadata, gt, material):
    """Author a supplement from typed, hash-checked construction material.

    The caller must use load_development_case, which checks the source hashes.
    No raw arrays, solver code, or answer values enter this preparation.
    """
    record = {"status": "VERIFIED", "authority": "FROZEN_CONSTRUCTION_MATERIALIZATION",
        "reviewed_by": "materialized-support-field-map-v1",
        "rationale": "Copy only predefined scientific fields from the case's frozen materialized_G_of_O; preserve original core obligations.",
        "task_sha256": task_identity(case_input, metadata, gt, material), "supporting_findings": []}
    for branch in gt.findings_by_operationalization:
        bid = branch.operationalization_id
        entry = material.get("branch_execution_evidence", {}).get(bid, {})
        values = entry.get("materialized_G_of_O", {})
        if not entry.get("source_artifact") or not entry.get("evidence_binding_sha256"):
            continue
        quantity = next((f for f in branch.findings if f.category.value == "quantity"), None)
        field_unit = quantity.unit if quantity else None
        for key, (label, category, unit) in MATERIALIZED_SUPPORT.items():
            value = values.get(key)
            if value is None:
                continue
            numbers = value if isinstance(value, list) else [value]
            if not numbers or any(type(v) not in {int, float} or not math.isfinite(v) for v in numbers):
                raise ValueError("invalid materialized scientific value: " + key)
            if unit == "count" and (len(numbers) != 1 or int(value) != value):
                raise ValueError("materialized count must be an integer")
            rid = bid + "_support_" + key
            peers = [f for f in branch.findings if f.category.value == category and f.verification]
            tolerance_key = "spatial_tolerance" if category == "location" else "absolute_tolerance"
            inherited = next((getattr(f.verification, tolerance_key) for f in peers
                              if getattr(f.verification, tolerance_key) is not None), None)
            tolerance = (0 if unit == "count" else inherited if inherited is not None and unit != "1"
                         else 1e-8 * max(abs(v) for v in numbers))
            spec = {"absolute_tolerance": None, "relative_tolerance": None, "spatial_tolerance": None}
            spec["spatial_tolerance" if category == "location" else "absolute_tolerance"] = tolerance
            ref = {"finding_id": rid, "category": category, "importance": "supporting", "statement": label,
                "value": value, "unit": field_unit if unit == "FIELD" else unit, "verification": spec}
            policy = {"operationalization_id": bid, "finding_id": rid,
                "verification_mode": "spatial_euclidean" if category == "location" else "scalar_tolerance",
                "verification_parameters": spec}
            record["supporting_findings"].append({"branch_id": bid, "finding": ref, "policy": policy})
    return record


def reporting_precision_rules(gt, supporting_findings=()):
    """Uniform three-significant-digit reporting contract for numeric outputs.

    Compatibility with the literal printed rounding interval is still required.
    This does not forgive arbitrary deviations within a relative tolerance.
    Counts/identifiers remain exact. Frozen GT values are never changed.
    """
    refs = [(b.operationalization_id, plain(f)) for b in gt.findings_by_operationalization for f in b.findings]
    refs += [(r["branch_id"], r["finding"]) for r in supporting_findings]
    rules = []
    for bid, ref in refs:
        v = ref.get("value")
        if ref.get("unit") == "count" or type(v) is int or v is None or isinstance(v, str):
            continue
        values = v if isinstance(v, list) else [v]
        if not values or any(type(x) not in {int, float} for x in values):
            continue
        radius = max(max(abs(x) for x in values) * .005, 1e-12)
        rules.append({"branch_id": bid, "finding_id": ref["finding_id"],
                      "min_significant_digits": 3, "max_rounding_radius": radius})
    return rules


def plain(value):
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, SimpleNamespace):
        return plain(vars(value))
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def task_identity(case_input, metadata, gt, material):
    return fingerprint({"input": plain(case_input), "metadata": plain(metadata), "gt": plain(gt),
        "verification": material.get("finding_verification_policy"),
        "requirements": material.get("finding_requirement_contract")})


def package(case_input, metadata, gt, material):
    """Expose existing scientific obligations, without adding solver duties."""
    dimensions = [d.value for d in metadata.principal_operationalization_dimensions]
    free = {d.value for d in metadata.unresolved_operationalization_dimensions}
    policies = policy_index(material.get("finding_verification_policy") or {})
    bundles = {b.operationalization_id: b for b in gt.acceptable_operationalizations}
    result = {"protocol": VERSION, "task_sha256": task_identity(case_input, metadata, gt, material),
        "dimensions": [{"dimension": d, "kind": "OPEN" if d in free else "FIXED",
            "criterion": ("An explicit, coherent scientific choice satisfying the original question; reference methods are examples."
                          if d in free else "Satisfies the original question's specified constraint.")}
                       for d in dimensions],
        "finding_goal": getattr(metadata, "finding_goal", ""),
        "adequate_role_contract": material.get("finding_requirement_contract"),
        "branches": [], "supplement_sources": material.get("reference_supplement_sources", [])}
    for branch in gt.findings_by_operationalization:
        bid = branch.operationalization_id
        result["branches"].append({"branch_id": bid,
            "decisions": plain(getattr(bundles[bid], "decisions", [])),
            "findings": [{**plain(f), "host_policy": policies.get((bid, f.finding_id)),
                "reporting_policy": material.get("reporting_policies", {}).get(bid, {}).get(f.finding_id)}
                         for f in branch.findings]})
        interpretation = material.get("branch_interpretations", {}).get(bid)
        if interpretation:
            result["branches"][-1]["input_population"] = deepcopy(interpretation)
    if material.get('supporting_finding_dependencies'):
        result['host_method_dependencies'] = deepcopy(material['supporting_finding_dependencies'])
    result["package_sha256"] = fingerprint(result)
    return result


def load_supplements(path, root):
    """Each source pointer binds an entire reviewed supplement, not just a value."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != VERSION:
        raise ValueError("unsupported reference package")
    out = {}
    for entry in payload.get("entries", []):
        if set(entry) != {"case_id", "task_sha256", "source"}:
            raise ValueError("reference entry requires case, task hash and source")
        src = entry["source"]
        source = (Path(root) / src["path"]).resolve()
        if not source.is_relative_to(Path(root).resolve()) or source.suffix != ".json":
            raise ValueError("reference source must be a frozen JSON artifact within the repository")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != src["sha256"]:
            raise ValueError("reference source checksum mismatch")
        record = _pointer(json.loads(raw), src["pointer"])
        if (record.get("status") != "VERIFIED" or not record.get("authority")
                or not record.get("reviewed_by") or not record.get("rationale")
                or record.get("task_sha256") != entry["task_sha256"]):
            raise ValueError("supplement needs task-bound scientific review and rationale")
        if entry["case_id"] in out:
            raise ValueError("duplicate case supplement; consolidate before freezing")
        allowed = {"status", "authority", "reviewed_by", "rationale", "task_sha256",
                   "new_branches", "supporting_findings", "reporting_policies", "branch_interpretations"}
        if set(record) - allowed:
            raise ValueError("unknown supplement fields")
        out[entry["case_id"]] = {"record": record, "source": src}
    return out


def apply_supplement(case_input, metadata, gt, material, supplement):
    if not supplement:
        return gt, material
    record = supplement["record"]
    if task_identity(case_input, metadata, gt, material) != record["task_sha256"]:
        raise ValueError("reference supplement belongs to another task version")
    gt, material = deepcopy(gt), deepcopy(material)
    policies = material.setdefault("finding_verification_policy", {}).setdefault("policies", [])
    dims = {d.value for d in metadata.principal_operationalization_dimensions}
    fixed = dims - {d.value for d in metadata.unresolved_operationalization_dimensions}
    original_bundles = list(gt.acceptable_operationalizations)
    original_refs = {b.operationalization_id: {f.finding_id: f for f in b.findings}
                     for b in gt.findings_by_operationalization}
    roles = (material.get("finding_requirement_contract") or {}).get("reference_finding_role_map", {})
    for addition in record.get("new_branches", []):
        bundle = OperationalizationBundle.model_validate(addition["operationalization"])
        branch = OperationalizationFindingBranch.model_validate(addition["findings"])
        bid = bundle.operationalization_id
        if bid in {b.operationalization_id for b in gt.acceptable_operationalizations} or branch.operationalization_id != bid:
            raise ValueError("duplicate or mismatched supplemental branch")
        decisions = {d.dimension.value: d.statement for d in bundle.decisions}
        if set(decisions) != dims:
            raise ValueError("supplement must specify every principal dimension")
        if not any(all(decisions[d] == {x.dimension.value: x.statement for x in old.decisions}.get(d)
                       for d in fixed) for old in original_bundles):
            raise ValueError("supplement changes fixed task constraints")
        # All new core requirements must correspond to the existing task. Never
        # add obligations after seeing an answer or let a shorter core inflate recall.
        base = original_refs[addition["requirement_branch_id"]]
        mapping = addition["requirement_map"]
        core = {f.finding_id for f in branch.findings if f.importance.value == "core"}
        old_core = {rid for rid, f in base.items() if f.importance.value == "core"}
        if set(mapping) != core or set(mapping.values()) != old_core or len(mapping) != len(old_core):
            raise ValueError("supplement must preserve all original core obligations one-to-one")
        for rid, old_rid in mapping.items():
            if old_rid in roles:
                if rid in roles and roles[rid] != roles[old_rid]:
                    raise ValueError("conflicting supplemental role ID")
                roles[rid] = roles[old_rid]
        gt.acceptable_operationalizations.append(bundle)
        gt.findings_by_operationalization.append(branch)
        _add_policies(policies, bid, branch.findings, addition["policies"])
    for addition in record.get("supporting_findings", []):
        bid = addition["branch_id"]
        branch = next((b for b in gt.findings_by_operationalization if b.operationalization_id == bid), None)
        ref = ReferenceFinding.model_validate(addition["finding"])
        if branch is None or ref.importance.value != "supporting" or any(f.finding_id == ref.finding_id for f in branch.findings):
            raise ValueError("invalid supplemental supporting finding")
        branch.findings.append(ref)
        _add_policies(policies, bid, [ref], [addition["policy"]])
        dependencies = addition.get('method_dimensions')
        if dependencies is not None:
            if (not isinstance(dependencies, list) or not dependencies
                    or any(not isinstance(d, str) for d in dependencies)
                    or len(set(dependencies)) != len(dependencies) or not set(dependencies) <= dims):
                raise ValueError('supporting reference needs explicit nonempty valid method dimensions')
            material.setdefault('supporting_finding_dependencies', {}).setdefault(bid, {})[ref.finding_id] = dependencies
    for bid, interpretation in record.get("branch_interpretations", {}).items():
        if (bid not in {b.operationalization_id for b in gt.acceptable_operationalizations}
                or set(interpretation) != {"dimension", "description", "publicly_specified"}
                or interpretation["dimension"] not in dims
                or not isinstance(interpretation["description"], str) or not interpretation["description"].strip()
                or type(interpretation["publicly_specified"]) is not bool):
            raise ValueError("invalid branch input-population interpretation")
        material.setdefault("branch_interpretations", {})[bid] = deepcopy(interpretation)
    for rule in record.get("reporting_policies", []):
        bid, rid = rule["branch_id"], rule["finding_id"]
        if not any(b.operationalization_id == bid and any(f.finding_id == rid for f in b.findings)
                   for b in gt.findings_by_operationalization):
            raise ValueError("reporting policy has unknown reference")
        if set(rule) != {"branch_id", "finding_id", "min_significant_digits", "max_rounding_radius"}:
            raise ValueError("reporting policy must explicitly bound accepted precision")
        if type(rule["min_significant_digits"]) is not int or rule["min_significant_digits"] < 2:
            raise ValueError("reporting precision needs at least two significant digits")
        bound = rule["max_rounding_radius"]
        if type(bound) not in {int, float} or not math.isfinite(bound) or bound <= 0:
            raise ValueError("reporting precision requires finite positive radius")
        rules = material.setdefault("reporting_policies", {}).setdefault(bid, {})
        if rid in rules:
            raise ValueError("duplicate reporting policy")
        rules[rid] = rule
    material["reference_supplement_sources"] = [{"source": supplement["source"],
        "authority": record["authority"], "reviewed_by": record["reviewed_by"], "rationale": record["rationale"]}]
    contract = FindingRequirementContract.from_mapping(material.get("finding_requirement_contract"))
    if contract and contract.validate():
        raise ValueError("invalid supplemental role contract")
    return gt, material


def _add_policies(policies, bid, findings, added):
    from .evaluator import finding_verification_mode
    index = policy_index({"policies": added})
    if set(index) != {(bid, f.finding_id) for f in findings}:
        raise ValueError("supplement needs one explicit policy per finding")
    if len(index) != len(added) or any(p.get("verification_mode") not in VERIFICATION_MODES for p in added):
        raise ValueError("supplement cannot install an executable verifier")
    for p in added:
        for value in (p.get("verification_parameters") or {}).values():
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
                raise ValueError("invalid supplemental numerical tolerance")
    for ref in findings:
        finding_verification_mode(ref, index[(bid, ref.finding_id)])
    policies.extend(deepcopy(added))
