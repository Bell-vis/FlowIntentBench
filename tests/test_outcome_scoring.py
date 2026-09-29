from copy import deepcopy
from types import SimpleNamespace

import pytest

from flowintentbench.ground_truth import ReferenceFinding
from flowintentbench.outcome_scoring import build_rubric, score_outcome, source_binding
from flowintentbench.answer_normalization import convert_explicit_units, score_normalized_outcome
from scripts.evaluate_answered_outcomes import write_reports, configure_review_credentials, review_effort_for_row


ANSWER = "The method is explicit. The required value is 0.5. The conclusion follows the method."
RUBRIC = [{"item_id": name, "group": name} for name in ("method", "requirements", "consistency")]


def test_retry_effort_retains_successful_identity_and_changes_only_failed_rows(tmp_path):
    import json
    row = {"run_id": "run"}
    (tmp_path / "cases").mkdir()
    path = tmp_path / "cases/run.json"
    assert review_effort_for_row(tmp_path, row, "medium", "high") == "medium"
    path.write_text(json.dumps({"status": "REVIEW_ERROR"}))
    assert review_effort_for_row(tmp_path, row, "medium", "high") == "high"
    path.write_text(json.dumps({"status": "SCORED", "reasoning_effort": "medium"}))
    assert review_effort_for_row(tmp_path, row, "medium", "high") == "medium"
    path.write_text(json.dumps({"status": "SCORED", "reasoning_effort": "high"}))
    assert review_effort_for_row(tmp_path, row, "medium", "high") == "high"


def test_review_credentials_use_explicit_file_over_ambient_key(tmp_path, monkeypatch):
    import os
    import flowintentbench.runtime_config as runtime
    monkeypatch.setattr(runtime, "resolve_provider_configuration", lambda **kwargs:
                        SimpleNamespace(api_key_env="TEST_REVIEW_KEY", base_url="https://example.invalid/v1"))
    monkeypatch.setenv("TEST_REVIEW_KEY", "ambient-key")
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY":" selected-test-key "}', encoding="utf-8")
    record = configure_review_credentials(auth)
    assert os.environ["TEST_REVIEW_KEY"] == "selected-test-key"
    assert record["auth_path"] == str(auth.resolve())
    assert "selected-test-key" not in str(record)
    auth.write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="no nonempty OPENAI_API_KEY"):
        configure_review_credentials(auth)


def fixture():
    ref = ReferenceFinding.model_validate({"finding_id": "f", "category": "quantity",
        "importance": "core", "statement": "value", "value": 0.5,
        "verification": {"absolute_tolerance": 0.01}})
    gt = SimpleNamespace(acceptable_operationalizations=[SimpleNamespace(operationalization_id="o")],
        findings_by_operationalization=[SimpleNamespace(operationalization_id="o", findings=[ref])])
    policy = {"policies": [{"operationalization_id": "o", "finding_id": "f",
        "verification_mode": "scalar_tolerance", "verification_parameters": ref.verification.model_dump()}]}
    judgment = {"ratings": [{"item_id": item["item_id"], "verdict": "MET",
        "evidence_text": ANSWER, "reason": "explicit"} for item in RUBRIC],
        "branch_relations": [{"branch_id": "o", "relation": "EQUIVALENT", "reason": "same method"}],
        "claims": [{"claim_id": "c", "scope": "PRIMARY", "statement": "value is 0.5",
            "evidence_text": "The required value is 0.5.", "value": 0.5, "unit": None,
            "reference_matches": [{"branch_id": "o", "finding_id": "f"}]}],
        "extraction_limitations": ""}
    return gt, policy, judgment


def test_complete_rubric_and_real_host_verification():
    gt, policy, judgment = fixture()
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["status"] == "SCORED"
    assert result["rubric_score"] == 1
    assert result["verification"]["ALL"]["VERIFIED"] == 1


def test_wrong_number_is_refuted_without_erasing_other_scores():
    gt, policy, judgment = fixture()
    judgment["claims"][0]["value"] = 0.9
    wrong_answer = ANSWER.replace("0.5", "0.9")
    judgment["claims"][0]["evidence_text"] = "The required value is 0.9."
    for item in judgment["ratings"]:
        item["evidence_text"] = wrong_answer
    result = score_outcome(wrong_answer, RUBRIC, gt, policy, judgment)
    assert result["status"] == "SCORED"
    assert result["verification"]["ALL"]["REFUTED"] == 1


def test_fabricated_extracted_value_is_unknown_even_with_real_quotation():
    gt, policy, judgment = fixture()
    wrong_answer = ANSWER.replace("0.5", "0.9")
    judgment["claims"][0]["evidence_text"] = "The required value is 0.9."
    result = score_outcome(wrong_answer, RUBRIC, gt, policy, judgment)
    assert result["verification"]["ALL"]["VERIFIED"] == 0
    assert result["verification"]["ALL"]["UNVERIFIED"] == 1
    assert result["claim_checks"][0]["value_binding"]["reason"] == "VALUE_NOT_IN_SOURCE"


def test_valid_alternative_not_forced_to_reference_numbers():
    gt, policy, judgment = fixture()
    judgment["branch_relations"][0]["relation"] = "DIFFERENT"
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["rubric_score"] == 1
    assert result["audit_status"] == "PARTIAL"
    assert result["verification"]["ALL"]["accuracy_lower"] == 0
    assert result["verification"]["ALL"]["accuracy_upper"] == 1


def test_unknown_rubric_is_an_interval_not_a_zero_or_a_pass():
    gt, policy, judgment = fixture()
    judgment["ratings"][0]["verdict"] = "NOT_ASSESSABLE"
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["status"] == "SCORED"
    assert result["rubric_score"] is None
    assert result["rubric_score_interval"] == pytest.approx([2 / 3, 1])


def test_missing_answer_content_gets_a_terminal_deduction():
    gt, policy, judgment = fixture()
    judgment["ratings"][0].update(verdict="NOT_MET", evidence_text="")
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["rubric_score"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "branch", "reference"])
def test_invalid_reviews_never_become_scored(mutation):
    gt, policy, judgment = fixture()
    if mutation == "omit":
        judgment["ratings"].pop()
    elif mutation == "duplicate":
        judgment["ratings"].append(deepcopy(judgment["ratings"][0]))
    elif mutation == "branch":
        judgment["branch_relations"][0]["branch_id"] = "foreign"
    else:
        judgment["claims"][0]["reference_matches"][0]["finding_id"] = "foreign"
    with pytest.raises(ValueError):
        score_outcome(ANSWER, RUBRIC, gt, policy, judgment)


def test_unlocatable_evidence_never_earns_credit_or_blocks_other_items():
    gt, policy, judgment = fixture()
    judgment["ratings"][0]["evidence_text"] = "invented evidence"
    judgment["claims"][0]["evidence_text"] = "invented number"
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["status"] == "SCORED"
    assert result["rubric_score"] is None
    assert result["rubric_score_interval"] == pytest.approx([2 / 3, 1])
    assert result["verification"]["ALL"]["VERIFIED"] == 0
    assert result["verification"]["ALL"]["UNVERIFIED"] == 1
    assert result["ratings"][0]["reviewer_verdict"] == "MET"
    assert result["ratings"][0]["verdict"] == "NOT_ASSESSABLE"


def test_supplemental_unknown_does_not_block_primary_score():
    gt, policy, judgment = fixture()
    extra = dict(judgment["claims"][0], claim_id="extra", scope="SUPPLEMENTAL", reference_matches=[])
    judgment["claims"].append(extra)
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["rubric_score"] == 1
    assert result["verification"]["PRIMARY"]["coverage"] == 1
    assert result["verification"]["ALL"]["coverage"] == 0.5


def test_report_retains_missing_trials_and_does_not_overwrite_repetitions(tmp_path):
    selection = {"answers": [{"run_id": str(i), "slot_id": str(i), "model_id": "m",
        "case_id": "same", "trial": i, "condition": "O1-F1"} for i in range(3)]}
    gt, policy, judgment = fixture()
    result = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    result.update(selection["answers"][0])
    report = write_reports(tmp_path, selection, [result])
    assert report["status"] == "INCOMPLETE"
    assert report["status_counts"] == {"SCORED": 1, "NOT_REVIEWED": 2}
    assert len((tmp_path / "reports/trial_metrics.csv").read_text().splitlines()) == 4


def test_report_separates_agent_and_api_review_sources(tmp_path):
    rows = [{"run_id": str(i), "slot_id": str(i), "model_id": "m",
             "case_id": "case", "trial": i, "condition": "O1-F1"} for i in range(2)]
    gt, policy, judgment = fixture()
    api = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    api.update(rows[0], reviewer_source="HTTP_API", reasoning_effort="high")
    agent = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    agent.update(rows[1], reviewer_source="EXPLICIT_GPT6_AGENT", reasoning_effort="medium")
    report = write_reports(tmp_path, {"answers": rows}, [api, agent])
    assert report["reviewer"]["source_counts"] == {"HTTP_API": 1, "EXPLICIT_GPT6_AGENT": 1}
    assert report["reviewer"]["source_metrics"]["HTTP_API"]["reasoning_efforts"] == {"high": 1}
    assert "no HTTP model receipt" in report["reviewer"]["source_metrics"]["EXPLICIT_GPT6_AGENT"]["model_evidence"]
    assert report["status"] == "COMPLETE"


def test_whitespace_binding_preserves_original_and_rejects_changed_content():
    answer = "Value is\n  0.5, in native units."
    bound = source_binding(answer, "Value is 0.5, in native units.")
    assert bound["text"] == answer
    assert bound["mode"] == "WHITESPACE_ONLY"
    assert source_binding(answer, "Value is 0.6, in native units.") is None
    assert source_binding(answer, "Value is ... in native units.") is None


def test_f2_rubric_preserves_mandatory_and_alternative_roles():
    metadata = SimpleNamespace(principal_operationalization_dimensions=["criterion"],
        unresolved_operationalization_dimensions=["criterion"], explicit_finding_requirements=[], finding_goal="goal")
    contract = {"mandatory_roles": ["location"], "alternative_role_groups": [
        {"group_id": "characterization", "min_required": 1, "roles": ["extent", "comparison"]}],
        "adequate_core_sets": [["location", "extent"], ["location", "comparison"]]}
    rubric = build_rubric(metadata, {"finding_requirement_contract": contract})
    requirements = [item for item in rubric if item["group"] == "requirements"]
    assert len(requirements) == 2
    assert requirements[1]["description"]["min_required"] == 1
    assert rubric[0]["open_choice"] is True


@pytest.mark.parametrize("value,source,target,expected", [
    (50, "%", "1", 0.5), (0.5, "fraction", "percent", 50),
    (1, "bar", "Pa", 100000), (2, "kPa", "bar", 0.02),
    ([100, 200], "cm", "m", [1, 2]), (4, "Pa\u00b7s", "Pa s", 4),
    (4, "s\u207b\u00b9", "1/s", 4), (1000, "ms", "seconds", 1),
])
def test_explicit_unit_rules(value, source, target, expected):
    converted = convert_explicit_units(value, source, target)
    assert converted["converted_value"] == pytest.approx(expected)
    assert converted["original_value"] == value
    assert converted["rule"] == "N02"


@pytest.mark.parametrize("value,source,target", [
    (1, "1", None), (1, None, "1"), (1, "stored units", "m"),
    (1, "Pa", "pA"), (1, "Pa", "m"), (True, "%", "1"),
    (float("nan"), "%", "1"), ("50%", "%", "1"),
])
def test_normalization_does_not_invent_scientific_equivalence(value, source, target):
    assert convert_explicit_units(value, source, target) is None


@pytest.mark.parametrize("percent,expected", [(50, "VERIFIED"), (90, "REFUTED")])
def test_normalized_replay_resolves_units_without_relaxing_correctness(percent, expected):
    gt, policy, judgment = fixture()
    gt.findings_by_operationalization[0].findings[0].unit = "1"
    answer = ANSWER.replace("0.5", str(percent) + "%")
    for rating in judgment["ratings"]:
        rating["evidence_text"] = answer
    judgment["claims"][0].update(value=percent, unit="%", evidence_text=answer)
    original = deepcopy(judgment)
    result = score_normalized_outcome(answer, RUBRIC, gt, policy, judgment)
    assert result["status"] == "SCORED"
    assert result["verification"]["ALL"][expected] == 1
    assert result["baseline_verification"]["ALL"]["UNVERIFIED"] == 1
    assert result["normalization_ledger"][0]["original_value"] == percent
    assert judgment == original


def test_unit_normalization_cannot_override_an_alternative_method_or_bad_source():
    gt, policy, judgment = fixture()
    gt.findings_by_operationalization[0].findings[0].unit = "1"
    judgment["claims"][0].update(value=50, unit="%")
    judgment["branch_relations"][0]["relation"] = "DIFFERENT"
    result = score_normalized_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["normalization_ledger"] == []
    assert result["verification"]["ALL"]["UNVERIFIED"] == 1
    judgment["branch_relations"][0]["relation"] = "EQUIVALENT"
    judgment["claims"][0]["evidence_text"] = "not in the answer"
    result = score_normalized_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    assert result["normalization_ledger"] == []
    assert result["verification"]["ALL"]["UNVERIFIED"] == 1


@pytest.mark.parametrize("tamper", ["rubric", "original_score"])
def test_normalized_replay_rejects_changed_rubric_or_original_score(tmp_path, monkeypatch, tamper):
    import hashlib
    import scripts.replay_normalized_outcomes as runner
    from flowintentbench.external_file_evaluator import write_json
    from flowintentbench.outcome_scoring import OutcomeJudgment

    gt, policy, judgment = fixture()
    source = tmp_path / "source"
    output = tmp_path / "normalized"
    manifest = tmp_path / "manifest.json"
    run = tmp_path / "run.json"
    write_json(manifest, {"cases": [{"case_id": "case"}]})
    write_json(run, {"final_response": ANSWER})
    row = {"run_id": "run", "slot_id": "run", "case_id": "case", "model_id": "m", "trial": 1,
           "condition": "O1-F1", "run_path": str(run), "run_sha256": runner.sha_file(run),
           "answer_sha256": hashlib.sha256(ANSWER.encode()).hexdigest()}
    write_json(source / "selection.json", {"answers": [row], "manifest_sha256": runner.sha_file(manifest)})
    scorer_hash = runner.sha_file(runner.ROOT / "flowintentbench/outcome_scoring.py")
    original = score_outcome(ANSWER, RUBRIC, gt, policy, judgment)
    original.update(row, request_id="review", scorer_sha256=scorer_hash,
                    verifier_sha256=runner.sha_file(runner.ROOT / "flowintentbench/evaluator.py"),
                    reviewer_model="gpt-6-astra", reasoning_effort="medium", reviewer_receipts=[])
    request = {"run_id": "run", "request_id": "review", "prompt": "frozen prompt",
               "rubric": deepcopy(RUBRIC), "schema": OutcomeJudgment.model_json_schema(), "scorer_sha256": scorer_hash}
    if tamper == "rubric":
        request["rubric"][1]["group"] = "consistency"
    else:
        original["rubric_score"] = 0.25
    write_json(source / "cases/run.json", original)
    write_json(source / "requests/review.json", request)
    write_json(source / "judgments/review.json", {"request_id": "review", "judgment": judgment})
    monkeypatch.setattr(runner.RunRecord, "from_dict", lambda _: SimpleNamespace(final_response=ANSWER))
    monkeypatch.setattr(runner, "load_development_case", lambda *_: (None, None, gt, {"finding_verification_policy": policy}))
    monkeypatch.setattr(runner, "build_rubric", lambda *_: deepcopy(RUBRIC))
    monkeypatch.setattr(runner, "review_prompt", lambda *_: "frozen prompt")
    with pytest.raises(ValueError, match="frozen review input changed|complete baseline score"):
        runner.replay(source, output, manifest)
