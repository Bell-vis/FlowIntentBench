"""Bind an explicitly selected field to independently frozen references.

The policy is opt-in scientific input. It never selects by the predicted
numeric value, and never changes any metric formula or finding role.
"""
import re


POLICY = "explicit_candidate_field_v1"


def inherit_verified_open_field_policy(artifact, reference_record):
    """Apply the case's choice policy after exact independent numeric replay.

    Equivalent runtime O descriptions can create a temporary branch. When its
    entire candidate computation equals the frozen one, the same candidate
    references and finding roles apply. O validity is still judged separately.
    Different computations never inherit this reference policy.
    """
    import copy
    if not reference_record:
        return artifact
    reference = reference_record.get("execution", {}).get("G_of_O", {})
    policy = reference.get("field_selection_policy", {})
    computed = artifact.get("execution", {}).get("G_of_O", {})
    if (policy.get("mode") != POLICY or not reference.get("all_field_metrics")
            or computed.get("metric_name") != reference.get("metric_name")
            or computed.get("all_field_metrics") != reference["all_field_metrics"]):
        return artifact
    result = copy.deepcopy(artifact)
    result["execution"]["G_of_O"]["field_selection_policy"] = copy.deepcopy(policy)
    result["findings"] = copy.deepcopy(reference_record["findings"])
    result["execution"]["data_provenance"]["open_field_policy_replay"] = {
        "candidate_metric_rows_equal": True,
        "scope": "same frozen field-choice policy and finding roles after independent execution; O scientific validity remains subject to adjudication",
    }
    return result


def bind_open_field_selection(branch, predictions, materialized):
    if not materialized or materialized.get("field_selection_policy", {}).get("mode") != POLICY:
        return branch, materialized
    policy = materialized["field_selection_policy"]
    refs = policy["reference_ids_by_field"]
    rows = {row["field_name"]: row for row in materialized["all_field_metrics"]}
    if set(refs) != set(rows):
        raise ValueError("open field references and computed candidate table differ")
    selected = set()
    for finding in predictions:
        # A table mentioning a field is not a declaration that it was chosen.
        statement = finding.statement.casefold()
        if not re.search(r"\b(selected|chosen|chose|choose|select|selection|focus)\b", statement):
            continue
        if re.search(r"\b(?:not|never)\s+(?:the\s+)?(?:selected|chosen|select|choose)\b", statement):
            continue
        value = finding.value
        named = set(re.findall(r"\bc\d+\b", statement))
        if isinstance(value, str) and re.fullmatch(r"c\d+", value.strip().casefold()):
            literal = value.strip().casefold()
            if named and literal not in named:
                selected.update(named | {literal})
                continue
            named = {literal}
        if len(named) == 1:
            selected.update(named)
    result = dict(materialized)
    result["selection_resolution"] = {
        "policy": POLICY, "declared_fields": sorted(selected),
        "source": "explicit authored selection in extracted findings; never numeric-value matching",
    }
    if len(selected) == 1 and next(iter(selected)) in rows:
        name = next(iter(selected))
        keep = set(refs[name])
        findings = [f for f in branch.findings if f.finding_id in keep]
        if len(findings) != len(keep):
            raise ValueError("open field reference IDs are incomplete")
        result.update(rows[name])
        result["selection_resolution"]["status"] = "EXPLICIT_VALID_CANDIDATE"
    else:
        # No unique valid selection: descriptive claims can still be true,
        # but they cannot earn the selected-field-identity requirement.
        identity_ids = set(policy["identity_reference_ids"])
        findings = [f for f in branch.findings if f.finding_id not in identity_ids]
        result["selection_resolution"]["status"] = "NO_UNIQUE_VALID_SELECTION"
        for key in ("field_name", "metric_value", "mean", "standard_deviation", "q10", "q50", "q90", "cell_count"):
            result.pop(key, None)
    return branch.model_copy(update={"findings": findings}), result
