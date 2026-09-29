import pytest

from flowintentbench.of_consistency import bound_result_group


def test_valid_f_for_other_valid_o_exposes_information_missing_from_marginals():
    support = {"a": {"value": False}, "b": {"value": True}}
    matched = bound_result_group(support, ["b"])
    swapped = bound_result_group(support, ["a"])
    assert matched["best_required_support"]["value"] == swapped["best_required_support"]["value"] == 1
    assert matched["conditional_required_support"]["value"] == 1
    assert matched["binding_gap"]["value"] == 0
    assert swapped["conditional_required_support"]["value"] == 0
    assert swapped["binding_gap"]["value"] == 1
    assert swapped["informative_branch_alignment"]["value"] == 0


def test_numerical_failure_is_not_misattributed_to_another_branch():
    r = bound_result_group({"a": {"value": False}, "b": {"value": False}}, ["a"])
    assert r["conditional_required_support"]["value"] == 0
    assert r["binding_gap"]["value"] == 0
    assert not r["mismatch_demonstrated"]
    assert r["informative_branch_alignment"]["value"] is None


def test_partly_wrong_pairing_gives_partial_credit():
    r = bound_result_group({"a": {"location": True, "strength": False},
                            "b": {"location": True, "strength": True}}, ["a"])
    assert r["conditional_required_support"]["value"] == .5
    assert r["binding_gap"]["value"] == .5


def test_indistinguishable_outcomes_do_not_award_alignment():
    r = bound_result_group({"a": {"value": True}, "b": {"value": True}}, ["a"])
    assert r["conditional_required_support"]["value"] == 1
    assert r["binding_gap"]["value"] == 0
    assert r["informative_branch_alignment"]["value"] is None
    assert r["informative_branch_alignment"]["reason"] == "OBSERVATIONALLY_AMBIGUOUS_FINDING_BRANCH"


def test_ambiguous_o_cannot_select_best_result_branch_for_full_credit():
    r = bound_result_group({"a": {"value": False}, "b": {"value": True}}, ["a", "b"], binding_status="AMBIGUOUS")
    assert r["conditional_required_support"]["lower"] == 0
    assert r["conditional_required_support"]["upper"] == 1
    assert r["binding_gap"]["value"] is None
    assert r["informative_branch_alignment"]["value"] is None


def test_single_method_conditional_support_survives_without_alignment():
    r = bound_result_group({"a": {"value": False}}, ["a"])
    assert r["conditional_required_support"]["value"] == 0
    assert r["informative_branch_alignment"]["reason"] == "SINGLE_METHOD_NO_CONTRAST"


@pytest.mark.parametrize("status", ["MISSING", "UNVERIFIED_ALTERNATIVE"])
def test_missing_or_unverified_o_is_unknown_not_inferred_from_best_f(status):
    r = bound_result_group({"a": {"value": True}}, [], binding_status=status)
    assert r["conditional_required_support"]["value"] is None
    assert r["conditional_required_support"]["lower"] == 0
    assert r["conditional_required_support"]["upper"] == 1


def test_unknown_role_keeps_identification_bounds():
    r = bound_result_group({"a": {"centroid": None, "mean": True},
                            "b": {"centroid": False, "mean": False}}, ["a"])
    assert r["conditional_required_support"]["lower"] == .5
    assert r["conditional_required_support"]["upper"] == 1
    assert r["informative_branch_alignment"]["value"] is None
    with pytest.raises(ValueError, match="same nonempty required roles"):
        bound_result_group({"a": {"mean": True}, "b": {"peak": True}}, ["a"])
