"""Regressions for model-agnostic ingestion and the evaluator/solver boundary."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from flowintentbench.answer_collections import collect
from flowintentbench.frozen_evidence import load_bank
from flowintentbench.trusted_scoring import METRICS, score
from scripts import evaluate_model_answers as runner
from test_trusted_scoring import fixture


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_retry_dispatch_preserves_pairs_and_defers_persistent_failures(tmp_path):
    ids = {'failed': 'a' * 64, 'once': 'b' * 64, 'fresh': 'c' * 64, 'paired': 'd' * 64}
    for i in range(3):
        write(tmp_path / 'old' / 'api' / (ids['failed'] + '-' + str(i)) / 'receipt.json', {'completed': False})
    write(tmp_path / 'current' / 'api' / (ids['once'] + '-1') / 'receipt.json', {'completed': False})
    rows = [{'output_id': key, 'case_id': case, 'trial': 1}
            for key, case in [('failed', 'a'), ('paired', 'a'), ('once', 'b'), ('fresh', 'c')]]
    before = deepcopy(rows)
    result = runner.prioritize_unattempted_reviews(rows, ids,
        [tmp_path / 'current', tmp_path / 'old', tmp_path / 'old', None])
    assert [x['output_id'] for x in result] == ['fresh', 'once', 'failed', 'paired']
    assert rows == before
    assert {x['output_id'] for x in result} == set(ids)
    # Without history the established paired-review order is unchanged.
    assert runner.prioritize_unattempted_reviews(rows, ids, []) == rows


def test_provider_circuit_requires_consecutive_transport_failures_and_drains():
    failure = {'completed': False, 'error': 'ProviderUnavailableError: upstream_error: upstream failed'}
    circuit = runner.ProviderCircuit()
    circuit.observe(failure); circuit.observe(failure)
    assert not circuit.is_set()
    circuit.observe({'completed': True})
    circuit.observe(failure); circuit.observe(failure)
    assert not circuit.is_set()
    circuit.observe({'completed': False, 'error': 'URLError: [SSL: UNEXPECTED_EOF_WHILE_READING]'})
    assert circuit.is_set()
    assert circuit.reason == 'THREE_CONSECUTIVE_TRANSPORT_FAILURES'
    circuit.observe({'completed': True})  # In-flight success cannot restart this batch.
    assert circuit.is_set()


def test_provider_circuit_counts_mixed_stream_and_remote_disconnect_failures():
    circuit = runner.ProviderCircuit()
    errors = ['ProviderUnavailableError: Responses stream ended before response.completed',
              'RemoteDisconnected: Remote end closed connection without response',
              'ConnectionResetError: [Errno 104] Connection reset by peer']
    for index, error in enumerate(errors):
        circuit.observe({'completed': False, 'error': error})
        assert circuit.is_set() == (index == 2)
    assert circuit.reason == 'THREE_CONSECUTIVE_TRANSPORT_FAILURES'


def test_provider_circuit_keeps_answer_validation_and_slow_answers_local():
    circuit = runner.ProviderCircuit()
    for _ in range(5):
        circuit.observe({'completed': False, 'error': 'TimeoutError: deadline'})
        circuit.observe({'completed': True, 'error': 'Invalid judgment schema'})
    assert not circuit.is_set()
    circuit.observe({'completed': False, 'http_status': 401})
    assert circuit.is_set() and circuit.reason == 'CREDENTIAL_OR_QUOTA_ERROR'


def test_optional_case_cannot_erase_a_known_error_in_macro_upper_bound():
    from flowintentbench.rubric_scoring import interval
    scores = [interval(.12,.96), interval(0,1), interval(.38,1), interval(.10,1)]
    scores[1]['applicability_unknown'] = True
    rows = [{'case_id':str(i),'metrics':{'c_score':s}} for i,s in enumerate(scores)]
    summary = runner.summarize(rows, ('c_score',))['c_score']
    assert summary['case_macro_upper'] == pytest.approx(.99)
    assert summary['case_macro_lower'] == pytest.approx(.15)
    assert summary['value'] is None


def test_hierarchical_optional_bounds_match_exhaustive_applicability_assignments():
    from itertools import product
    from statistics import mean
    from flowintentbench.rubric_scoring import interval
    endpoints = [(0,.1),(.4,.7),(.8,1),(.2,.9)]
    for optional_flags in product((False,True), repeat=4):
        rows=[]; possibilities=[]
        for i, ((lo,hi), optional) in enumerate(zip(endpoints,optional_flags)):
            metric=interval(lo,hi)
            metric['applicability_unknown']=optional
            rows.append({'case_id':str(i//2),'metrics':{'c_score':metric}})
            possibilities.append((None,lo,hi) if optional else (lo,hi))
        possible_means=[]
        for assignment in product(*possibilities):
            cases=[[x for x in assignment[i:i+2] if x is not None] for i in (0,2)]
            means=[mean(xs) for xs in cases if xs]
            if means:possible_means.append(mean(means))
        got=runner.summarize(rows,('c_score',))['c_score']
        assert got['case_macro_lower']==pytest.approx(min(possible_means))
        assert got['case_macro_upper']==pytest.approx(max(possible_means))


def test_unknown_applicability_is_retained_even_when_current_flag_is_false():
    from flowintentbench.rubric_scoring import interval
    metric=interval(applicable=False)
    metric['applicability_unknown']=True
    got=runner.summarize([{'case_id':'x','metrics':{'c_score':metric}}],('c_score',))['c_score']
    assert (got['case_macro_lower'],got['case_macro_upper'])==(0,1)
    assert got['unknown_applicability_trials']==1
    assert got['value'] is None


def test_review_json_recovery_accepts_commentary_but_rejects_competing_or_truncated_objects():
    from scripts.run_core_case_scoring import _parse_json
    assert _parse_json('I will map the supplied evidence.\n{"result":{"x":1}}\nDone.')=={'result':{'x':1}}
    assert _parse_json('```json\n{"x":1}\n```')=={'x':1}
    for text in ('{"x":1}\n{"x":2}', 'Preface\n{"outer":{"x":1}', 'Preface\n[{"x":1}]'):
        with pytest.raises(ValueError):_parse_json(text)


def test_strict_review_cache_is_bound_to_actual_api_format(tmp_path):
    schema={'type':'object','properties':{},'required':[],'additionalProperties':False}
    receipt=write(tmp_path/'receipt.json',{'completed':True,'response_model':'judge','reasoning_effort':'medium','final_text':'{}'})
    payload={'input':[{'content':'prompt'}],'max_output_tokens':100,
             'text':{'format':{'type':'json_schema','name':'answer_evaluation','strict':True,'schema':schema}}}
    write(tmp_path/'request.json',payload)
    stored={'request_id':'id','judgment':{},'judgment_sha256':runner.digest({}),'receipt':str(receipt)}
    request={'prompt':'prompt','schema':schema,'structured_output':True}
    kwargs=dict(effort='medium',max_output_tokens=100,structured_output=True)
    assert runner.validate_review_record(stored,request,'id','prompt',schema,'judge',**kwargs)==stored
    payload['text']['format']['strict']=False;write(tmp_path/'request.json',payload)
    with pytest.raises(ValueError,match='structured schema'):
        runner.validate_review_record(stored,request,'id','prompt',schema,'judge',**kwargs)


def test_generic_collections_support_n1_seven_models_and_preserve_unstarted(tmp_path):
    manifest = write(tmp_path / "manifest.json", {"cases": [{"case_id": "c", "condition": "O1-F1"}]})
    models = [f"arbitrary-model-{i}" for i in range(7)]
    config = {"models": models, "repetitions": 1, "manifest_sha256": runner.sha(manifest)}
    slots = [{"slot_id": m, "model_id": m, "case_id": "c", "trial_index": 1,
              "status": "PENDING" if i else "MODEL_NONCOMPLETION"} for i, m in enumerate(models)]
    write(tmp_path / "collection_state.json", {"configuration": config, "slots": slots, "experiment_id": "e"})
    actual, _, _ = collect([tmp_path], None, manifest)
    assert len(actual) == 7 and actual[1]["status"] == "PENDING"
    assert all(len(s["output_id"]) == 64 for s in actual)
    slots.append(slots[0])
    write(tmp_path / "collection_state.json", {"configuration": config, "slots": slots, "experiment_id": "e"})
    with pytest.raises(ValueError, match="duplicate, missing"):
        collect([tmp_path], None, manifest)


def test_exports_bind_hashes_reject_duplicates_and_do_not_trust_run_paths(tmp_path):
    manifest = write(tmp_path / "manifest.json", {"cases": [{"case_id": "c", "condition": "O1-F1"}]})
    row = {"model_id": "../../model", "case_id": "c", "trial": 1, "answer": "answer", "run_id": "../escape"}
    path = write(tmp_path / "answers.json", [row])
    actual, _, _ = collect(None, [path], manifest)
    assert "/" not in actual[0]["output_id"]
    with pytest.raises(ValueError, match="duplicate model"):
        collect(None, [path, path], manifest)
    row["answer_sha256"] = "wrong"
    write(path, [row])
    with pytest.raises(ValueError, match="answer hash"):
        collect(None, [path], manifest)


def test_production_scoring_cannot_recompute_answers_even_with_raw_root(tmp_path, monkeypatch):
    import flowintentbench.numeric_diagnostics as offline
    def forbidden(*args, **kwargs):
        raise AssertionError("production evaluator tried to solve")
    monkeypatch.setattr(offline, "_arrays", forbidden)
    monkeypatch.setattr(offline, "_compute", forbidden)
    data = fixture()
    extra = {"finding_id": "extra", "statement": "mean X", "evidence_text": "Mean X is 9.",
             "value": 9, "unit": None, "eligible": True, "matches": [],
             "numeric_checks": [{"branch_id": "o", "statistic": "mean_x", "method_evidence_text": "Mean X is 9.",
                                 "scope_evidence_text": "Mean X is 9.", "scope_status": "EXPLICIT"}]}
    answer, meta, gt, material, review = data
    material["branch_execution_evidence"] = {"o": {"execution_provenance": {"recipe": {"kind": "association", "measure": "pearson"}},
                                                   "evidence_binding_sha256": "context"}}
    material["frozen_auxiliary_evidence"] = [{"query": {"branch_id": "o", "statistic": "mean_x", "axis": None,
        "lower_x": None, "upper_x": None}, "expected": 7, "branch_evidence_sha256": "context",
        "source": {"authority": "analytic fixture"}}]
    review["findings"].append(extra)
    result = score(answer + " Mean X is 9.", meta, gt, material, review, condition="O2-F1", root=tmp_path)
    assert result["metrics"]["finding_precision"]["value"] == .5
    assert result["metrics"]["c_score"]["value"] == .5
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    material.pop("frozen_auxiliary_evidence")
    result = score(answer + " Mean X is 9.", meta, gt, material, review, condition="O2-F1", root=tmp_path)
    assert result["metrics"]["finding_precision"]["value"] is None
    assert result["metrics"]["finding_precision"]["supported_value"] == .5


@pytest.mark.parametrize("condition", ["O1-F1", "O2-F1", "O3-F1", "O1-F2"])
def test_complete_applicable_metric_values_follow_condition(condition):
    answer, meta, gt, material, review = fixture()
    meta.unresolved_operationalization_dimensions = ([] if condition.startswith("O1") else
        meta.principal_operationalization_dimensions if condition.startswith("O3") else meta.unresolved_operationalization_dimensions)
    material["finding_requirement_contract"] = {"mandatory_roles": ["measure"], "adequate_core_sets": [["measure"]],
                                               "reference_finding_role_map": {"cv": "measure"}}
    result = score(answer, meta, gt, material, review, condition=condition)
    assert set(result["metrics"]) == set(METRICS)
    for value in result["metrics"].values():
        assert value["supported_value"] == (1 if value["applicable"] else None)
    assert result["metrics"]["urs"]["applicable"] == (not condition.startswith("O1"))
    assert result["metrics"]["resolved_o_compliance"]["applicable"] == (not condition.startswith("O3"))
    assert result["metrics"]["core_finding_recall"]["applicable"] == condition.endswith("F1")
    assert result["metrics"]["adequate_core_complete"]["applicable"] == condition.endswith("F2")


def test_frozen_evidence_source_tamper_is_not_scored(tmp_path):
    source = write(tmp_path / "reference.json", {"mean": 7})
    entry = {"case_id": "c", "query": {"branch_id": "o", "statistic": "mean_x", "axis": None, "lower_x": None, "upper_x": None},
             "expected": 7, "branch_evidence_sha256": "frozen", "authority": "fixture",
             "source": {"path": source.name, "sha256": runner.sha(source), "pointer": "/mean"}}
    bank = write(tmp_path / "bank.json", {"protocol": "FROZEN_AUXILIARY_REFERENCES_V1", "entries": [entry]})
    assert load_bank(bank, tmp_path)["c"][0]["expected"] == 7
    write(source, {"mean": 9})
    with pytest.raises(ValueError, match="checksum"):
        load_bank(bank, tmp_path)


def test_of_gap_exposes_method_result_mismatch_while_missing_and_wrong_differ():
    answer, meta, gt, material, review = fixture()
    gt.acceptable_operationalizations.append(SimpleNamespace(operationalization_id="b"))
    ref = gt.findings_by_operationalization[0].findings[0].model_copy(update={"finding_id": "b_cv"})
    gt.findings_by_operationalization.append(SimpleNamespace(operationalization_id="b", findings=[ref]))
    material["finding_verification_policy"]["policies"].append({**material["finding_verification_policy"]["policies"][0],
        "operationalization_id": "b", "finding_id": "b_cv"})
    for dimension in review["dimensions"]:
        dimension["matches"]["b"] = False
    review["findings"][0]["matches"] = [{"branch_id": "b", "finding_id": "b_cv"}]
    result = score(answer, meta, gt, material, review, condition="O2-F1")
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["of_consistency"]["conditional_required_support"]["value"] == 0
    assert result["of_consistency"]["binding_gap"]["value"] == 1
    assert result["of_consistency"]["mismatch_demonstrated"]
    assert result["of_consistency"]["informative_branch_alignment"]["value"] == 0
    answer, meta, gt, material, review = fixture()
    answer = answer.replace("0.5", "0.9")
    review["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    result = score(answer, meta, gt, material, review, condition="O2-F1")
    assert result["answer_issues"]["missing_core_by_declared_branch"] == {"o": []}
    assert len(result["answer_issues"]["reference_contradictions_under_declared_o"]) == 1


def test_all_unknown_extra_preserves_full_numeric_bounds_without_claiming_exactness(tmp_path):
    answer, meta, gt, material, review = fixture()
    review["findings"].append({"finding_id": "extra", "statement": "new", "evidence_text": "new", "value": None,
                               "unit": None, "eligible": True, "matches": []})
    result = score(answer + " new", meta, gt, material, review, condition="O2-F1")
    row = {"run_id": "r", "output_id": "r", "case_id": "c", "model_id": "any", "trial": 1,
           "condition": "O2-F1", "collection_status": "COMPLETED", **result}
    report = runner.write_report(tmp_path, [row], {"api_calls": 0}, {})
    assert report["status"] == "COMPLETE_WITH_BOUNDS"
    assert report["scientific_identification_status"] == "PARTIAL"
    assert row["metric_values"]["finding_precision"] == .5
    assert row["metrics"]["finding_precision"]["value"] is None
    assert report["branch_alignment_diagnostics"]["any"]["informative_mean"] is None


def test_legacy_import_uses_same_canonical_runner():
    from scripts import evaluate_trusted_metrics as old
    assert old.run is runner.run


def test_distinct_statistics_cannot_hide_three_errors_in_one_vector():
    answer, meta, gt, material, review = fixture()
    quote = "Layer range is 0.80 to 0.90 and median is 0.85 over all z layers."
    statistics = ["layer_pearson_min", "layer_pearson_max", "layer_pearson_median"]
    queries = [{"branch_id": "o", "statistic": statistic, "value_index": i, "axis": "z",
                "method_evidence_text": quote, "scope_evidence_text": quote, "scope_status": "EXPLICIT"}
               for i, statistic in enumerate(statistics)]
    review["findings"].append({"finding_id": "layers", "statement": "layer correlations", "evidence_text": quote,
        "value": [.80, .90, .85], "unit": None, "eligible": True, "matches": [], "numeric_checks": queries})
    material["branch_execution_evidence"] = {"o": {"evidence_binding_sha256": "context",
        "execution_provenance": {"recipe": {"kind": "association", "measure": "pearson"}}}}
    material["frozen_auxiliary_evidence"] = [{"query": {"branch_id": "o", "statistic": statistic, "axis": "z",
        "lower_x": None, "upper_x": None}, "expected": expected, "branch_evidence_sha256": "context", "source": {}}
        for statistic, expected in zip(statistics, [.2, .4, .3])]
    result = score(answer + " " + quote, meta, gt, material, review, condition="O2-F1")
    assert result["applicable_finding_count_upper"] == 4
    assert result["metrics"]["finding_precision"]["value"] == .25
    assert result["metrics"]["c_score"]["value"] == .25
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert len(result["error_diagnostics"]["independent_contradicted_finding_ids"]) == 3


def test_explicit_tetra_volume_is_not_mistaken_for_original_vtk_signed_volume():
    answer, meta, gt, material, review = fixture()
    quote = "I used absolute tetrahedral volumes."
    meta.unresolved_operationalization_dimensions = meta.principal_operationalization_dimensions
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(
        dimension=SimpleNamespace(value="property_measure"),
        statement="Weight by the absolute signed cell volume from VTK and normalize by the included volume.")]
    review["dimensions"][1]["evidence_text"] = quote
    result = score(answer + " " + quote, meta, gt, material, review, condition="O3-F1")
    assert result["dimension_matches"]["o"]["property_measure"] is False
    assert result["answer_issues"]["novel_method_unverified"]
    assert result["metrics"]["o_score"]["upper"] == 1
    assert result["metrics"]["finding_precision"]["upper"] == 1
    assert result["finding_checks"][0]["reason"] == "UNVERIFIED_ALTERNATIVE_METHOD_REFERENCE_MISMATCH"


def test_reviewer_trailing_decimal_tokens_preserve_values_and_quoted_text():
    from scripts.run_core_case_scoring import _parse_json
    assert _parse_json('{"value":[1.,-2.,0.],"quote":"range [1.,2.]"}') == {
        'value':[1.,-2.,0.], 'quote':'range [1.,2.]'}
    assert _parse_json('{"value": 1. ,"quoted":"escaped \\\"[2.,3.]\\\""}') == {
        'value':1., 'quoted':'escaped "[2.,3.]"'}
    for text in ('{"value":1.','{"value":01.}','{"value":1.e}','{"value":+1.}',
                 '{"value":[1.},{"extraction_complete":true}'):
        with pytest.raises(ValueError):_parse_json(text)
