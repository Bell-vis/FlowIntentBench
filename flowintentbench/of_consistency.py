"""Candidate O–F diagnostics for a single explicitly linked result group.

Inputs are independently verified, per-role support under each method. This
module does not infer O from the reported numbers, identify primary results,
accept novel methods, or replace the frozen C-score. Those are upstream gates.
"""
from __future__ import annotations

from .trusted_scoring import interval


def bound_result_group(support_by_method, declared_methods, *, binding_status="RESOLVED"):
    """Compute conditional support and mismatch evidence without best-O rescue.

    support_by_method: method ID -> {required role: True / False / None}.
    False includes a verified absent required result. None is evaluator unknown.
    More than one possible declared method yields an interval over all choices.
    All methods must concern the same required roles in the same result group.
    """
    if not support_by_method:
        raise ValueError("at least one independently evaluated method is required")
    methods = set(support_by_method)
    roles = set(next(iter(support_by_method.values())))
    if not roles or any(set(v) != roles for v in support_by_method.values()):
        raise ValueError("all methods must cover exactly the same nonempty required roles")
    if any(v is not None and type(v) is not bool for row in support_by_method.values() for v in row.values()):
        raise ValueError("role support must be True, False, or None")
    declared = set(declared_methods)
    if not declared <= methods:
        raise ValueError("a declared method lacks an independent support record")
    if binding_status not in {"RESOLVED", "AMBIGUOUS", "MISSING", "UNVERIFIED_ALTERNATIVE"}:
        raise ValueError("unknown binding status")
    if binding_status == "RESOLVED" and len(declared) != 1:
        raise ValueError("resolved binding requires exactly one declared method")
    by_method = {m: interval(sum(v is True for v in rr.values()) / len(roles),
                            sum(v is not False for v in rr.values()) / len(roles),
                            reason="UNVERIFIED_ROLE_SUPPORT") for m, rr in support_by_method.items()}
    best = interval(max(v["lower"] for v in by_method.values()), max(v["upper"] for v in by_method.values()))
    if binding_status in {"MISSING", "UNVERIFIED_ALTERNATIVE"} or not declared:
        conditional = interval(reason=binding_status)
        gap = interval(reason=binding_status)
    else:
        # min/max over possible O bindings; never pick the branch with the best F.
        conditional = interval(min(by_method[m]["lower"] for m in declared),
                               max(by_method[m]["upper"] for m in declared), reason="METHOD_OR_ROLE_UNRESOLVED")
        gap = interval(max(0, best["lower"] - conditional["upper"]),
                       max(0, best["upper"] - conditional["lower"]), reason="METHOD_OR_ROLE_UNRESOLVED")
    alignment = None
    reason = "UNRESOLVED_METHOD_OR_ROLE"
    if len(methods) == 1:
        reason = "SINGLE_METHOD_NO_CONTRAST"
    elif binding_status == "RESOLVED" and all(v["value"] is not None for v in by_method.values()):
        supported = {m for m, v in by_method.items() if v["value"] == 1}
        if not supported:
            reason = "NO_FULLY_SUPPORTED_FINDING_BRANCH"
        elif len(supported) > 1:
            reason = "OBSERVATIONALLY_AMBIGUOUS_FINDING_BRANCH"
        else:
            alignment = float(bool(supported & declared))
            reason = "UNIQUE_SUPPORTED_BRANCH"
    return {"protocol": "of-consistency-candidate-v1", "scope": "ONE_EXPLICIT_RESULT_GROUP",
        "required_roles": sorted(roles), "binding_status": binding_status,
        "declared_methods": sorted(declared), "support_by_method": by_method,
        "conditional_required_support": conditional, "best_required_support": best,
        "binding_gap": gap,
        "informative_branch_alignment": {"value": alignment, "assessable": alignment is not None, "reason": reason},
        "mismatch_demonstrated": binding_status == "RESOLVED" and gap["lower"] > 0,
        "interpretation": "Conditional numerical support and cross-method mismatch diagnostic; not global answer quality."}
