"""Scientific scoring regressions with explicit, controlled evidence labels.

These are implementation tests, not a measurement of real judge accuracy.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from flowintentbench import rubric_scoring as rubric
from flowintentbench.reference_packages import package, fingerprint, load_supplements, apply_supplement, task_identity
from flowintentbench.ground_truth import ReferenceFinding
from scripts import evaluate_model_answers as runner
from test_trusted_scoring import fixture


def example():
    answer, meta, gt, material, old = fixture()
    review = {"protocol": rubric.VERSION, "result_groups": [{"group_id": "main", "purpose": "PRIMARY",
        "evidence_text": "The method uses pressure.", "dimensions": [dict(d, adequacy="MET",
        reason="SATISFIES_CONSTRAINT", rationale="Supplied evidence satisfies the task.") for d in old["dimensions"]]}],
        "findings": [dict(f, group_id="main", mapping_complete=True) for f in old["findings"]],
        "extraction_complete": True, "limitations": ""}
    return answer, meta, gt, material, review


def evaluate(data, condition="O2-F1"):
    return rubric.score(*data, condition=condition)


def test_empty_limitations_list_preserves_scores_and_nonempty_warning_is_not_erased():
    data=example();baseline=evaluate(deepcopy(data));data[-1]['limitations']=[]
    assert evaluate(data)['metrics']==baseline['metrics']
    data[-1]['limitations']=['Only catalogued scientific findings were extracted.']
    result=evaluate(data)
    assert result['extraction_complete'] is False
    data[-1]['limitations']=[{'invented':'object'}]
    with pytest.raises(ValueError,match='text only'):evaluate(data)


@pytest.mark.parametrize('value', [.5, .9])
def test_explicit_extracted_value_alias_is_lossless_and_conflicts_are_rejected(value):
    data = list(example())
    data[0] = data[0].replace('0.5', str(value))
    finding = data[-1]['findings'][0]
    finding.update(value=value, evidence_text=f'CV is {value}.')
    expected = evaluate(deepcopy(data))
    finding['extracted_value'] = finding.pop('value')
    original = deepcopy(data[-1])
    got = evaluate(data)
    assert got['metrics'] == expected['metrics']
    assert data[-1] == original
    finding['value'] = value + 1
    with pytest.raises(ValueError, match='conflicting extracted value aliases'):
        evaluate(data)


@pytest.mark.parametrize('also_purpose', [False, True])
def test_explicit_group_role_alias_preserves_scoring_and_rejects_conflicts(also_purpose):
    data = example()
    expected = evaluate(deepcopy(data))
    group = data[-1]['result_groups'][0]
    group['role'] = group['purpose']
    if not also_purpose:
        group.pop('purpose')
    original = deepcopy(data[-1])
    assert evaluate(data)['metrics'] == expected['metrics']
    assert data[-1] == original
    group.update(purpose='PRIMARY', role='SUPPLEMENTARY')
    with pytest.raises(ValueError, match='conflicting group purpose aliases'):
        evaluate(data)


def two_branches():
    answer, meta, gt, material, review = example()
    gt.acceptable_operationalizations.append(SimpleNamespace(operationalization_id="b"))
    ref = gt.findings_by_operationalization[0].findings[0].model_copy(update={"finding_id": "b_cv", "value": .8})
    gt.findings_by_operationalization.append(SimpleNamespace(operationalization_id="b", findings=[ref]))
    material["finding_verification_policy"]["policies"].append({**material["finding_verification_policy"]["policies"][0],
        "operationalization_id": "b", "finding_id": "b_cv"})
    for d in review["result_groups"][0]["dimensions"]:
        d["matches"]["b"] = False
    review["findings"][0]["matches"].append({"branch_id": "b", "finding_id": "b_cv"})
    return answer, meta, gt, material, review


def test_extracted_dimensions_alias_preserves_evidence_and_rejects_conflicts():
    data = example()
    expected = evaluate(deepcopy(data))
    group = data[-1]['result_groups'][0]
    group['extracted_dimensions'] = group.pop('dimensions')
    original = deepcopy(data[-1])
    assert evaluate(data)['metrics'] == expected['metrics']
    assert data[-1] == original
    group['dimensions'] = []
    with pytest.raises(ValueError, match='conflicting extracted dimension aliases'):
        evaluate(data)


def test_dimension_assessment_alias_does_not_supply_missing_branch_matches():
    data = example()
    expected = evaluate(deepcopy(data))
    dimension = data[-1]['result_groups'][0]['dimensions'][0]
    dimension['assessment'] = dimension.pop('adequacy')
    assert evaluate(data)['metrics'] == expected['metrics']
    dimension.pop('matches')
    assert evaluate(data)['metrics']['o_score']['value'] is None
    dimension['adequacy'] = 'NOT_MET'
    with pytest.raises(ValueError, match='conflicting dimension assessment aliases'):
        evaluate(data)


@pytest.mark.parametrize('label', [True, False, None])
def test_finding_equivalence_alias_preserves_unknown_and_false_edges(label):
    data = example()
    claim = data[-1]['findings'][0]
    claim['matches'][0]['equivalent'] = label
    expected = evaluate(deepcopy(data))
    edge = claim['matches'][0]
    edge['equivalence'] = edge.pop('equivalent')
    if label is not True:
        claim['matches'].append({'branch_id':None,'finding_id':None,'equivalence':None})
    original = deepcopy(data[-1])
    got = evaluate(data)
    assert got['metrics'] == expected['metrics']
    assert data[-1] == original
    if label is not True:
        assert got['metrics']['finding_precision']['value'] is None
    edge['equivalent'] = not label
    with pytest.raises(ValueError, match='conflicting finding equivalence aliases'):
        evaluate(data)


def test_annotated_unit_line_requires_literal_text_on_the_named_line():
    answer, meta, gt, material, review = example()
    answer += '\nValues are in file coordinate units.'
    review['findings'][0].update(unit='file coordinate units',
                                unit_evidence_text='L2: Values are in file coordinate units.')
    assert evaluate((answer,meta,gt,material,review))['metrics']['finding_requirement_recall']['value']==1
    review['findings'][0]['unit_evidence_text']='L1: Values are in file coordinate units.'
    result=evaluate((answer,meta,gt,material,review))
    assert result['metrics']['finding_requirement_recall']['value'] is None
    review['findings'][0]['unit_evidence_text']='L2: Values are in meters.'
    assert evaluate((answer,meta,gt,material,review))['metrics']['finding_requirement_recall']['value'] is None


def test_auxiliary_statistic_uses_its_own_definition_inside_another_method_branch():
    answer, meta, gt, material, review = example()
    finding = rubric.Finding.model_validate(review['findings'][0])
    finding = finding.model_copy(update={'statement':'Coefficient of variation', 'evidence_text':'CV is 0.5.'})
    branch = SimpleNamespace(decisions=[SimpleNamespace(statement='Use the interdecile range divided by the median.')])
    ref = gt.findings_by_operationalization[0].findings[0]
    assert rubric._incompatible_reference_mapping(answer, finding, ref, branch)
    support = ref.model_copy(update={'importance':'supporting', 'statement':'Unweighted coefficient of variation'})
    # Revalidate enum after model_copy, which intentionally does not validate.
    support = ReferenceFinding.model_validate(support.model_dump())
    assert not rubric._incompatible_reference_mapping(answer, finding, support, branch)
    other = support.model_copy(update={'statement':'Relative interdecile pressure range'})
    assert rubric._incompatible_reference_mapping(answer, finding, other, branch)


@pytest.mark.parametrize('claimed', [.5, .9])
@pytest.mark.parametrize('source_kind,reference_kind', [('point', 'cell'), ('cell', 'point')])
def test_distinct_field_array_sources_neither_earn_credit_nor_become_numeric_errors(claimed, source_kind, reference_kind):
    answer, meta, gt, material, review = example()
    quote = f'{source_kind.capitalize()}-field coefficient is {claimed}.'
    answer += '\n' + quote
    review['findings'][0].update(evidence_text=quote, value=claimed)
    material['branch_execution_evidence'] = {'o': {'evidence_binding_sha256': 'frozen',
        'execution_provenance': {'status': 'MATERIALIZED', 'recipe': {'kind': 'association',
            'field': {'association': reference_kind}, 'other_field': {'association': reference_kind}}}}}
    result = evaluate((answer, meta, gt, material, review))
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []
    assert result['metrics']['finding_precision']['lower'] == 0
    assert result['metrics']['finding_precision']['upper'] == 1
    assert result['metrics']['finding_requirement_recall']['lower'] == 0
    assert result['metrics']['finding_requirement_recall']['upper'] == 1
    assert any(c['reason'] == 'FIELD_ASSOCIATION_REFERENCE_GAP' for c in result['finding_checks'])
    # An unbound reviewer label cannot decide the source array, and a matching
    # source still receives the ordinary numeric comparison.
    for label in ('Unspecified', reference_kind.capitalize(), 'Point-field and cell'):
        source = f'{label}-field coefficient is {claimed}.'
        review['findings'][0]['evidence_text'] = source
        checked = evaluate((answer + '\n' + source, meta, gt, material, review))
        assert checked['metrics']['finding_precision']['value'] == (1 if claimed == .5 else 0)


@pytest.mark.parametrize('claimed,expected_errors',[(.5,[]),(.9,['p'])])
def test_one_false_compatible_reference_does_not_make_a_verified_answer_error(claimed,expected_errors):
    answer,meta,gt,material,review=two_branches()
    for d in review['result_groups'][0]['dimensions']:
        d['matches']['b']=True
    answer=answer.replace('0.5',str(claimed))
    review['findings'][0].update(value=claimed,evidence_text=f'CV is {claimed}.')
    result=evaluate((answer,meta,gt,material,review))
    assert result['error_diagnostics']['independent_contradicted_finding_ids']==expected_errors
    assert result['metrics']['c_score']['value']==(1 if claimed==.5 else 0)


def test_compact_support_catalog_retains_ids_without_leaking_numeric_targets():
    answer,meta,gt,mat,review=example()
    ref=gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings.append(ReferenceFinding.model_validate(
        ref.model_copy(update={'finding_id':'extra','importance':'supporting','value':123.456789,
                              'statement':'Additional scalar statistic'}).model_dump()))
    mat['supporting_finding_dependencies']={'o':{'extra':['feature_definition']}}
    mat['finding_verification_policy']['policies'].append({
        **mat['finding_verification_policy']['policies'][0],'finding_id':'extra'})
    empty=SimpleNamespace(model_dump=lambda **_: {})
    ci=SimpleNamespace(scientific_question='A synthetic question',case_context=empty,
                       flow_data=SimpleNamespace(data_metadata=empty))
    prompt=rubric.review_prompt(ci,meta,gt,mat,answer)
    assert '123.456789' not in prompt
    packet=json.loads(prompt.split('\n\n',1)[1])
    r=packet['reference_package'];shared=r['shared_supporting_findings']
    assert shared[0]['reference_ids']=={'B0':'R1'}
    assert r['branches'][0]['supporting_reference_ids']==['R1']
    assert 'value' not in r['branches'][0]['findings'][0]
    assert package(ci,meta,gt,mat)['branches'][0]['findings'][1]['value']==123.456789


@pytest.mark.parametrize('unit', ['length^3 in stored units', 'mesh-volume units', 'volume units'])
def test_explicit_stored_coordinate_volume_units_remain_native_not_si(unit):
    answer, meta, gt, material, review = example()
    quote = f'Included volume is 0.5 {unit}.'
    answer += ' ' + quote
    ref = gt.findings_by_operationalization[0].findings[0].model_copy(update={'unit':None, 'statement':'Included volume'})
    finding = rubric.Finding.model_validate(review['findings'][0]).model_copy(update={
        'statement':'Included volume', 'evidence_text':quote, 'unit':unit})
    policy = material['finding_verification_policy']['policies'][0]
    from flowintentbench.trusted_scoring import _verify
    assert _verify(answer, finding, ref, policy)[0] is True
    assert _verify(answer, finding.model_copy(update={'unit':'m^3'}), ref, policy)[0] is None


@pytest.mark.parametrize("condition", ["O1-F1", "O2-F1", "O3-F1", "O1-F2"])
def test_complete_answer_has_all_applicable_points_and_correct_denominators(condition):
    answer, meta, gt, material, review = example()
    meta.unresolved_operationalization_dimensions = ([] if condition.startswith("O1") else
        meta.principal_operationalization_dimensions if condition.startswith("O3") else meta.unresolved_operationalization_dimensions)
    material["finding_requirement_contract"] = {"mandatory_roles": ["measure"], "adequate_core_sets": [["measure"]],
        "reference_finding_role_map": {"cv": "measure"}}
    result = evaluate((answer, meta, gt, material, review), condition)
    assert set(result["metrics"]) == set(rubric.METRICS)
    assert all(m["value"] == 1 if m["applicable"] else m["value"] is None for m in result["metrics"].values())
    assert result["metrics"]["urs"]["applicable"] == (not condition.startswith("O1"))
    assert result["metrics"]["resolved_o_compliance"]["applicable"] == (not condition.startswith("O3"))
    assert result["metrics"]["core_finding_recall"]["applicable"] == condition.endswith("F1")


def test_f2_supporting_mean_cannot_satisfy_required_coefficient_by_generic_category():
    answer,meta,gt,mat,review=example()
    answer+=' Mean pressure is 7 bar.'
    ref=gt.findings_by_operationalization[0].findings[0]
    support=ReferenceFinding.model_validate(ref.model_copy(update={'finding_id':'mean','importance':'supporting',
        'statement':'Mean pressure','value':7,'unit':'bar'}).model_dump())
    gt.findings_by_operationalization[0].findings.append(support)
    mat['finding_verification_policy']['policies'].append({**mat['finding_verification_policy']['policies'][0],'finding_id':'mean'})
    mat['finding_requirement_contract']={'mandatory_roles':[], 'adequate_core_sets':[['coefficient']],
        'reference_finding_role_map':{'cv':'coefficient'},'role_by_category':{'quantity':'coefficient'}}
    review['findings']=[dict(review['findings'][0],finding_id='mean',statement='Mean pressure',value=7,unit='bar',
        evidence_text='Mean pressure is 7 bar.',matches=[{'branch_id':'o','finding_id':'mean'}])]
    r=evaluate((answer,meta,gt,mat,review),'O1-F2')
    assert r['metrics']['finding_precision']['value']==1
    assert r['metrics']['finding_requirement_recall']['value']==0
    assert r['metrics']['adequate_core_complete']['value']==0
    # An independently authored explicit role is still honored.
    mat['finding_requirement_contract']['reference_finding_role_map']['mean']='coefficient'
    assert evaluate((answer,meta,gt,mat,review),'O1-F2')['metrics']['adequate_core_complete']['value']==1


def test_missing_and_wrong_required_results_are_distinct_local_failures():
    data = example()
    data[-1]["findings"] = []
    result = evaluate(data)
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["core_finding_recall"]["value"] == 0
    assert result["metrics"]["c_score"]["status"] == "NOT_APPLICABLE"
    assert result["requirement_items"][0]["reason"] == "REQUIRED_RESULT_MISSING"
    answer, meta, gt, material, review = example()
    answer = answer.replace("0.5", "0.9")
    review["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    result = evaluate((answer, meta, gt, material, review))
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["finding_precision"]["value"] == 0
    assert result["metrics"]["c_score"]["value"] == 0
    assert result["requirement_items"][0]["reason"] == "REQUIRED_RESULT_INCORRECT"
    assert result["branch_alignment_diagnostic"]["informative"] is False


def test_missing_judge_eligibility_does_not_block_independent_metrics_or_award_credit():
    data = example()
    del data[-1]['findings'][0]['eligible']
    result = evaluate(data)
    assert result['status'] == 'SCORED'
    assert result['metrics']['o_score']['value'] == 1
    assert result['metrics']['finding_precision']['lower'] == 0
    assert result['metrics']['finding_precision']['upper'] == 1
    assert result['metrics']['core_finding_recall']['lower'] == 0
    assert any(r['reason'] == 'MISSING_ELIGIBILITY_UNKNOWN' for r in result['review_repairs'])


def test_explicit_nested_review_layout_recovers_same_semantics_without_new_api():
    data=example();review=data[-1];group=review['result_groups'][0]
    group['group_type']=group.pop('purpose')
    group['findings']=review.pop('findings')
    for f in group['findings']:
        f['eligibility']=f.pop('eligible');f.pop('group_id');f.pop('mapping_complete')
    for d in group['dimensions']:
        d['matches']=[{'branch_id':b,'equivalent':v} for b,v in d['matches'].items()]
        d['adequacy']='ADEQUATE';d.pop('reason');d['choice']='Explicit method description'
    review['mapping_complete']=True;review.pop('protocol');review.pop('limitations')
    r=evaluate(data)
    assert r['metrics']['o_score']['value']==1
    assert r['metrics']['finding_precision']['value']==1
    assert r['metrics']['c_score']['value']==1
    group['dimensions'][0]['matches']=[{'branch_id':'o','finding_id':'cv'}]
    r=evaluate(data)
    assert r['metrics']['o_score']['value'] is None
    assert any(x['reason']=='MISSING_PROTOCOL_FROM_CURRENT_REQUEST' for x in r['review_repairs'])
    group['purpose']='SUPPLEMENTARY'
    with pytest.raises(ValueError,match='conflicting group purpose'):
        evaluate(data)


@pytest.mark.parametrize('source_value,expected',[(50,1),(90,0)])
def test_judge_fraction_restored_only_from_explicit_source_percentage(source_value,expected):
    answer,meta,gt,material,review=example()
    refs=gt.findings_by_operationalization[0].findings
    refs[0]=refs[0].model_copy(update={'unit':'1'})
    quote=f'CV is {source_value}%.'
    answer=answer.replace('CV is 0.5.',quote)
    review['findings'][0].update(evidence_text=quote,value=source_value/100,unit='1')
    r=evaluate((answer,meta,gt,material,review))
    assert r['metrics']['finding_precision']['value']==expected
    assert any(x['reason']=='RESTORED_LITERAL_PERCENT' for x in r['review_repairs'])
    answer=answer.replace('%',' bar');review['findings'][0]['evidence_text']=quote.replace('%',' bar')
    r=evaluate((answer,meta,gt,material,review))
    assert r['metrics']['finding_precision']['value'] is None


def test_missing_o_cannot_be_filled_by_reference_and_does_not_erase_findings():
    data = example()
    data[-1]["result_groups"][0]["dimensions"][0].update(status="MISSING", evidence_text="", adequacy="NOT_MET", reason="MISSING")
    result = evaluate(data)
    assert result["metrics"]["o_score"]["value"] == .5
    assert result["rubric_summaries"]["o_score"] == "PARTIAL"
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert not result["metrics"]["c_score"]["applicable"]


@pytest.mark.parametrize('status', ['AMBIGUOUS', 'MISSING'])
@pytest.mark.parametrize('condition', ['O1-F1', 'O2-F1'])
def test_reviewer_uncertainty_does_not_establish_answer_omission_or_erase_c(status, condition):
    data = example()
    if condition.startswith('O1'):
        data[1].unresolved_operationalization_dimensions = []
    dimension = data[-1]['result_groups'][0]['dimensions'][0]
    dimension.update(status=status, adequacy='UNVERIFIABLE', reason='REVIEW_UNCERTAIN',
                     matches={'o': None})
    result = evaluate(data, condition)
    assert result['metrics']['o_score']['value'] is None
    assert result['metrics']['o_score']['upper'] == 1
    assert result['metrics']['c_score']['applicable']
    assert result['metrics']['c_score']['value'] is None
    assert result['metrics']['c_score']['upper'] == 1
    assert result['result_groups'][0]['method_determinacy_unresolved']
    assert not result['result_groups'][0]['method_determinate']
    assert not result['of_consistency']['mismatch_demonstrated']
    # A reviewer explicitly establishing answer ambiguity/omission remains
    # a failed rubric item; it cannot be rescued by this uncertainty handling.
    dimension.update(adequacy='NOT_MET', reason='ANSWER_AMBIGUOUS' if status=='AMBIGUOUS' else 'MISSING')
    result = evaluate(data, condition)
    assert result['metrics']['o_score']['value'] == .5
    assert not result['metrics']['c_score']['applicable']


def test_valid_alternative_o_gets_credit_without_inventing_its_numeric_truth():
    data = example()
    data[-1]["result_groups"][0]["dimensions"][1].update(matches={"o": False}, reason="VALID_ALTERNATIVE")
    result = evaluate(data)
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["urs"]["value"] == 1
    assert result["metrics"]["finding_precision"]["status"] == "INTERVAL"
    assert result["reference_gaps"][0]["action"] == "REFERENCE_CONSTRUCTION"
    data[-1]["findings"] = []
    assert evaluate(data)["metrics"]["core_finding_recall"]["value"] == 0


@pytest.mark.parametrize('fixed', [True, False])
def test_connectivity_conflict_is_separate_from_open_method_validity(fixed):
    answer, meta, gt, material, review = example()
    quote = 'I used z neighbors while holding x and y fixed.'
    answer += ' ' + quote
    dimension = review['result_groups'][0]['dimensions'][0]
    dimension.update(evidence_text=quote, matches={'o': True}, adequacy='MET')
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(
        dimension=meta.principal_operationalization_dimensions[0],
        statement='Use six-neighbor connected components.')]
    if not fixed:
        meta.unresolved_operationalization_dimensions = meta.principal_operationalization_dimensions[:]
        dimension['reason'] = 'VALID_ALTERNATIVE'
    result = evaluate((answer, meta, gt, material, review), 'O2-F1' if fixed else 'O3-F1')
    assert result['metrics']['o_score']['value'] == (.5 if fixed else 1)
    assert result['rubric_items'][0]['reference_equivalence_conflicts'] == {
        'o':'SINGLE_AXIS_RUNS_VS_SIX_NEIGHBOR_COMPONENTS'}
    if not fixed:
        assert result['result_groups'][0]['novel_method']
        assert result['metrics']['c_score']['value'] is None


@pytest.mark.parametrize('feature_match,claimed,expected', [(True,7,(.5,1)), (True,9,(0,.5)),
                                                          (False,7,(0,1)), (None,7,(0,1))])
def test_independent_support_does_not_depend_on_unrelated_open_measure(feature_match, claimed, expected):
    answer, meta, gt, material, review = example()
    answer += f' Mean pressure is {claimed} bar.'
    dimensions = review['result_groups'][0]['dimensions']
    dimensions[0]['matches']['o'] = feature_match
    dimensions[1].update(matches={'o':False}, reason='VALID_ALTERNATIVE')
    ref = gt.findings_by_operationalization[0].findings[0]
    support = ReferenceFinding.model_validate(ref.model_copy(update={
        'finding_id':'mean', 'statement':'Mean pressure', 'value':7, 'unit':'bar', 'importance':'supporting'}).model_dump())
    gt.findings_by_operationalization[0].findings.append(support)
    material['finding_verification_policy']['policies'].append({
        **material['finding_verification_policy']['policies'][0], 'finding_id':'mean'})
    material['supporting_finding_dependencies'] = {'o':{'mean':[dimensions[0]['dimension']]}}
    review['findings'].append(dict(review['findings'][0], finding_id='mean', statement='Mean pressure',
        value=claimed, unit='bar', evidence_text=f'Mean pressure is {claimed} bar.',
        matches=[{'branch_id':'o','finding_id':'mean'}]))
    r = evaluate((answer,meta,gt,material,review))
    for n in ['finding_precision','c_score']:
        target = ((1,1) if n == 'finding_precision' else (0,0)) if feature_match is False else expected
        assert (r['metrics'][n]['lower'],r['metrics'][n]['upper']) == target
    assert r['metrics']['core_finding_recall']['value'] == (1 if feature_match is False else None)
    assert ('mean' in r['error_diagnostics']['independent_contradicted_finding_ids']) == (feature_match is True and claimed==9)


def test_fixed_constraint_cannot_be_rescued_by_judge_calling_it_alternative():
    data = example()
    data[-1]["result_groups"][0]["dimensions"][0].update(matches={"o": False}, reason="VALID_ALTERNATIVE")
    result = evaluate(data)
    assert result["metrics"]["resolved_o_compliance"]["value"] == 0
    assert result["metrics"]["o_score"]["value"] == .5


def test_public_fixed_constraint_does_not_require_private_reference_implementation():
    data = example()
    dimension = data[-1]['result_groups'][0]['dimensions'][0]
    name = dimension['dimension']
    data[3]['branch_interpretations'] = {'o':{'dimension':name,
        'description':'Reference-only discrete integration convention', 'publicly_specified':False}}
    dimension.update(status='AMBIGUOUS', matches={'o':None}, adequacy='MET', reason='SATISFIES_CONSTRAINT')
    result = evaluate(data)
    assert result['metrics']['resolved_o_compliance']['value'] == 1
    assert result['metrics']['c_score']['applicable']
    assert result['metrics']['c_score']['value'] is None
    assert result['metrics']['c_score']['lower'] == 0
    # Absence of an implementation match does not verify its numerical result.
    assert result['result_groups'][0]['method_determinate']
    dimension.update(status='EXTRACTED',matches={'o':False})
    result = evaluate(data)
    assert result['metrics']['resolved_o_compliance']['value'] == 1
    assert result['result_groups'][0]['novel_method']
    assert result['metrics']['c_score']['value'] is None
    dimension.update(status='EXTRACTED', adequacy='NOT_MET', reason='VIOLATES_CONSTRAINT')
    assert evaluate(data)['metrics']['resolved_o_compliance']['value'] == 0
    dimension.update(adequacy='MET',reason='SATISFIES_CONSTRAINT',evidence_text='Invented quote')
    assert evaluate(data)['metrics']['resolved_o_compliance']['value'] is None
    dimension.update(evidence_text=data[0],matches={'o':False},reason='VALID_ALTERNATIVE')
    result = evaluate(data)
    assert result['metrics']['resolved_o_compliance']['value'] == 1
    assert result['result_groups'][0]['novel_method']
    assert result['metrics']['c_score']['value'] is None
    assert not result['of_consistency']['mismatch_demonstrated']


@pytest.mark.parametrize('condition', ['O1-F1', 'O2-F1'])
@pytest.mark.parametrize('publicly_specified', [True, False])
def test_private_implementation_alternative_label_does_not_create_false_of_mismatch(
        condition, publicly_specified):
    data = example()
    if condition.startswith('O1'):
        data[1].unresolved_operationalization_dimensions = []
    dimension = data[-1]['result_groups'][0]['dimensions'][0]
    data[3]['branch_interpretations'] = {'o': {'dimension': dimension['dimension'],
        'description': 'Cell coordinate implementation', 'publicly_specified': publicly_specified}}
    dimension.update(matches={'o': False}, adequacy='MET', reason='SATISFIES_CONSTRAINT')
    expected = evaluate(deepcopy(data), condition)
    dimension['reason'] = 'VALID_ALTERNATIVE'
    actual = evaluate(data, condition)
    assert actual['metrics'] == expected['metrics']
    assert actual['of_consistency'] == expected['of_consistency']
    if not publicly_specified:
        assert actual['metrics']['o_score']['value'] == 1
        assert actual['metrics']['c_score']['value'] is None
        assert actual['of_consistency']['binding_gap']['lower'] == 0
        assert not actual['of_consistency']['mismatch_demonstrated']
    else:
        assert actual['metrics']['resolved_o_compliance']['value'] < 1


@pytest.mark.parametrize('relation', [False, None])
def test_unmatched_open_numeric_result_is_unverified_not_missing(relation):
    answer, meta, gt, material, review = example()
    review['result_groups'][0]['dimensions'][1].update(matches={'o': relation}, reason='VALID_ALTERNATIVE')
    review['findings'][0]['matches'] = []
    result = evaluate((answer, meta, gt, material, review))
    recall = result['metrics']['core_finding_recall']
    assert (recall['lower'], recall['upper']) == (0, 1)
    assert result['requirement_items'][0]['reason'] == 'REQUIREMENT_UNVERIFIED'
    # Genuine omission remains a failure, including prose with no result value.
    review['findings'][0]['value'] = None
    assert evaluate((answer, meta, gt, material, review))['metrics']['core_finding_recall']['value'] == 0
    # A catalogued fixed method receives no such alternative-method allowance.
    meta.unresolved_operationalization_dimensions = []
    review['findings'][0]['value'] = .5
    for d in review['result_groups'][0]['dimensions']:
        d['matches'] = {'o': True}
    assert evaluate((answer, meta, gt, material, review), 'O1-F1')['metrics']['core_finding_recall']['value'] == 0


def test_two_correct_methods_are_scored_in_their_own_result_groups():
    answer, meta, gt, material, review = two_branches()
    quote = "Sensitivity uses the weighted method. CV is 0.8."
    group = deepcopy(review["result_groups"][0])
    group.update(group_id="sensitivity", purpose="SUPPLEMENTARY", evidence_text=quote)
    for d in group["dimensions"]:
        d.update(matches={"o": False, "b": True}, evidence_text=quote)
    review["result_groups"].append(group)
    claim = deepcopy(review["findings"][0])
    claim.update(group_id="sensitivity", finding_id="s", value=.8, evidence_text="CV is 0.8.")
    review["findings"].append(claim)
    result = evaluate((answer + " " + quote, meta, gt, material, review))
    for name in ("finding_precision", "core_finding_recall", "c_score"):
        assert result["metrics"][name]["value"] == 1
    assert result["of_consistency"]["method_binding_coverage"] == 1


def test_o_a_with_f_b_reduces_consistency_not_best_branch_finding_credit():
    answer, meta, gt, material, review = two_branches()
    answer = answer.replace("0.5", "0.8")
    review["findings"][0].update(value=.8, evidence_text="CV is 0.8.")
    result = evaluate((answer, meta, gt, material, review))
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["metrics"]["c_score"]["value"] == 0
    assert result["metrics"]["branch_alignment"]["value"] == 0
    assert result["of_consistency"]["binding_gap"]["value"] == 1


def test_branch_ties_do_not_create_a_per_claim_frankenstein_score():
    answer, meta, gt, material, review = two_branches()
    quote = "The second reported CV is 0.8."
    second = deepcopy(review["findings"][0])
    second.update(finding_id="second", evidence_text=quote, value=.8)
    review["findings"].append(second)
    result = evaluate((answer + " " + quote, meta, gt, material, review))
    assert result["metrics"]["finding_precision"]["value"] == .5
    assert result["metrics"]["c_score"]["value"] == .5
    assert result["branch_alignment_diagnostic"]["kind"] == "ALL_BRANCH_TIE_TRIVIAL"


def test_unknown_extra_does_not_block_core_and_cannot_be_silently_ignored():
    answer, meta, gt, material, review = example()
    extra = deepcopy(review["findings"][0])
    extra.update(finding_id="extra", statement="new scientific quantity", evidence_text="Extra is 7.", value=7, matches=[])
    review["findings"].append(extra)
    result = evaluate((answer + " Extra is 7.", meta, gt, material, review))
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["finding_precision"]["lower"] == .5
    assert result["metrics"]["finding_precision"]["upper"] == 1
    assert result["reference_gaps"][0]["reason"] == "REFERENCE_GAP"


def test_reference_semantic_support_requires_quoted_reference_evidence():
    answer, meta, gt, material, review = example()
    ref = ReferenceFinding.model_validate({"finding_id": "qual", "category": "quantity", "importance": "supporting",
        "statement": "The pressure is nonuniform.", "value": None})
    gt.findings_by_operationalization[0].findings.append(ref)
    material["finding_verification_policy"]["policies"].append({"operationalization_id": "o", "finding_id": "qual", "verification_mode": "semantic_only"})
    extra = deepcopy(review["findings"][0])
    extra.update(finding_id="extra", evidence_text="The pressure is nonuniform.", statement=ref.statement, value=None,
        matches=[{"branch_id": "o", "finding_id": "qual", "semantic_verdict": "SUPPORTED", "reference_evidence_text": ref.statement}])
    review["findings"].append(extra)
    data = (answer + " " + ref.statement, meta, gt, material, review)
    assert evaluate(data)["metrics"]["finding_precision"]["value"] == 1
    extra["matches"][0]["reference_evidence_text"] = "This was never in the reference."
    assert evaluate(data)["metrics"]["finding_precision"]["value"] is None


def test_rounding_acceptance_requires_reviewed_policy_and_minimum_precision():
    answer, meta, gt, material, review = example()
    ref = gt.findings_by_operationalization[0].findings[0]
    ref = ref.model_copy(update={"value": .512345, "verification": ref.verification.model_copy(update={"absolute_tolerance": 1e-6})})
    gt.findings_by_operationalization[0].findings[0] = ref
    material["finding_verification_policy"]["policies"][0]["verification_parameters"] = ref.verification.model_dump()
    review["findings"][0].update(value=.512, evidence_text="CV is 0.512.")
    data = (answer.replace("0.5", "0.512"), meta, gt, material, review)
    assert evaluate(data)["metrics"]["finding_precision"]["value"] is None
    material["reporting_policies"] = {"o": {"cv": {"min_significant_digits": 3, "max_rounding_radius": .0005}}}
    assert evaluate(data)["metrics"]["finding_precision"]["value"] == 1
    material["reporting_policies"]["o"]["cv"]["max_rounding_radius"] = .00005
    assert evaluate(data)["metrics"]["finding_precision"]["value"] is None


def test_old_reviews_rejected_and_schema_failure_not_a_wrong_model_answer():
    with pytest.raises(ValueError):
        evaluate(fixture())
    data = example()
    data[-1]["findings"][0]["group_id"] = "invented"
    with pytest.raises(ValueError, match="unknown result group"):
        evaluate(data)


def test_prose_reason_repair_preserves_explicit_verdict_and_wrong_numbers():
    answer, meta, gt, material, review = example()
    review['result_groups'][0]['dimensions'][0]['reason'] = 'Uses the specified stored pressure.'
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['o_score']['value'] == 1
    assert any(r['reason'] == 'PROSE_REASON_CANONICALIZED' for r in result['review_repairs'])
    review['findings'][0].update(value=.9, evidence_text='CV is 0.9.')
    result = evaluate((answer.replace('0.5', '0.9'), meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] == 0


def test_judge_cannot_override_wrong_numeric_reference_with_semantic_supported():
    answer, meta, gt, material, review = example()
    review["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    review["findings"][0]["matches"][0].update(semantic_verdict="SUPPORTED", reference_evidence_text="CV")
    assert evaluate((answer.replace("0.5", "0.9"), meta, gt, material, review))["metrics"]["finding_precision"]["value"] == 0


def test_reference_supplement_bound_to_whole_record_and_task(tmp_path):
    answer, meta, gt, material, review = example()
    record = {"status": "VERIFIED", "authority": "controlled analytic fixture", "reviewed_by": "test author",
        "rationale": "Known analytic quantity.", "task_sha256": task_identity(None, meta, gt, material),
        "supporting_findings": [{"branch_id": "o", "finding": {"finding_id": "extra", "category": "quantity",
            "importance": "supporting", "statement": "Mean", "value": 7, "verification": {"absolute_tolerance": .01}},
            "policy": {"operationalization_id": "o", "finding_id": "extra", "verification_mode": "scalar_tolerance",
            "verification_parameters": {"absolute_tolerance": .01, "relative_tolerance": None, "spatial_tolerance": None}}}]}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(record))
    path = tmp_path / "package.json"
    path.write_text(json.dumps({"protocol": "evaluation-reference-package-v1", "entries": [{"case_id": "c",
        "task_sha256": record["task_sha256"], "source": {"path": source.name, "sha256": runner.sha(source), "pointer": ""}}]}))
    supplement = load_supplements(path, tmp_path)["c"]
    augmented, mat = apply_supplement(None, meta, gt, material, supplement)
    assert len(augmented.findings_by_operationalization[0].findings) == 2
    assert len(gt.findings_by_operationalization[0].findings) == 1
    assert package(None, meta, augmented, mat)["supplement_sources"]
    with pytest.raises(ValueError, match="another task"):
        apply_supplement(None, meta, augmented, mat, supplement)
    record["supporting_findings"][0]["finding"]["value"] = 99
    source.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        load_supplements(path, tmp_path)


def test_population_interpretation_is_bound_to_known_dimension_and_reference():
    _, meta, gt, material, _ = example()
    note = {'dimension':'feature_definition', 'description':'All finite stored points.', 'publicly_specified':False}
    record = {'task_sha256':task_identity(None, meta, gt, material), 'authority':'controlled construction',
              'reviewed_by':'test','rationale':'input scope omitted','branch_interpretations':{'o':note}}
    gt2, mat = apply_supplement(None, meta, gt, material, {'record':record,'source':{}})
    assert package(None, meta, gt2, mat)['branches'][0]['input_population'] == note
    assert 'branch_interpretations' not in material
    bad = deepcopy(record)
    bad['branch_interpretations']['o']['dimension'] = 'new_solver_obligation'
    with pytest.raises(ValueError, match='input-population'):
        apply_supplement(None, meta, gt, material, {'record':bad,'source':{}})


def test_new_output_cannot_resume_v3_and_import_flags_are_rejected(tmp_path):
    args = runner.parser().parse_args(["--output", str(tmp_path), "--offline"])
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/experiment_report.json").write_text(json.dumps({"protocol": "trusted-original-metrics-v3-evidence-only"}))
    with pytest.raises(ValueError, match="another evaluation protocol"):
        runner.run(args)
    args.reuse_trusted_from = [tmp_path]
    with pytest.raises(ValueError, match="fresh reviews"):
        runner.run(args)


def test_production_v4_never_calls_scientific_execution(monkeypatch):
    from flowintentbench import numeric_diagnostics
    def forbidden(*a, **k):
        raise AssertionError("evaluation attempted to solve")
    monkeypatch.setattr(numeric_diagnostics, "_compute", forbidden)
    monkeypatch.setattr(numeric_diagnostics, "_arrays", forbidden)
    assert evaluate(example())["metrics"]["finding_precision"]["value"] == 1


def test_local_extraction_and_mapping_errors_do_not_erase_other_metrics():
    data = example()
    data[-1]["findings"][0]["value"] = [0.5, {"unparsed": "number"}]
    result = evaluate(data)
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["core_finding_recall"]["status"] == "INTERVAL"
    assert result["review_repairs"][0]["reason"] == "INVALID_EXTRACTED_VALUE_SCHEMA"
    data = example()
    data[-1]["findings"][0]["matches"][0]["finding_id"] = "invented"
    result = evaluate(data)
    assert result["metrics"]["core_finding_recall"]["status"] == "INTERVAL"
    assert result["reference_gaps"][0]["action"] == "TARGETED_REVIEW"
    data = example()
    data[-1]["findings"][0]["value"] = "0.5"
    assert evaluate(data)["metrics"]["finding_precision"]["value"] == 1


def test_missing_branch_review_is_not_a_claimed_novel_method():
    data = example()
    data[-1]["result_groups"][0]["dimensions"][1]["matches"] = {}
    result = evaluate(data)
    assert result["metrics"]["o_score"]["value"] == 1
    assert not result["result_groups"][0]["novel_method"]
    assert result["metrics"]["c_score"]["status"] == "INTERVAL"
    assert any(x["reason"] == "REVIEW_O_BINDING_UNRESOLVED" and x["action"] == "TARGETED_REVIEW"
               for x in result["reference_gaps"])


def test_line_citations_restore_exact_original_text_and_aliases_without_value_inference():
    answer, meta, gt, material, review = example()
    answer = answer.replace(" CV is", "\nCV is")
    group = review["result_groups"][0]
    group.update(evidence_text="", source_start_line=2, source_end_line=2)
    for d in group["dimensions"]:
        d.update(evidence_text="", source_start_line=1, source_end_line=1, matches={"B0": True})
    review["findings"][0].update(evidence_text="", source_start_line=2, source_end_line=2,
        matches=[{"branch_id": "B0", "finding_id": "R0"}])
    result = evaluate((answer, meta, gt, material, review))
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["c_score"]["value"] == 1
    review["findings"][0]["value"] = .9
    assert evaluate((answer, meta, gt, material, review))["metrics"]["finding_precision"]["value"] is None
    schema = rubric.review_schema(meta, gt)
    assert schema["$defs"]["Match"]["properties"]["finding_id"]["enum"] == ["R0"]


def test_native_scale_labels_and_unit_line_citations_are_representation_only():
    answer, meta, gt, material, review = example()
    review["findings"][0].update(unit="stored velocity scale")
    assert evaluate((answer, meta, gt, material, review))["metrics"]["finding_precision"]["value"] == 1
    review["findings"][0].update(unit="m/s")
    assert evaluate((answer, meta, gt, material, review))["metrics"]["finding_precision"]["value"] is None
    review["findings"][0].update(unit="bar", unit_evidence_text="L2")
    gt.findings_by_operationalization[0].findings[0] = gt.findings_by_operationalization[0].findings[0].model_copy(update={"unit": "bar"})
    assert evaluate((answer + "\nAll values in bar.", meta, gt, material, review))["metrics"]["finding_precision"]["value"] == 1


def test_spelled_cardinalities_are_literal_without_parsing_larger_numbers():
    from flowintentbench.answer_evidence import value_is_bound
    assert value_is_bound(6, "Six disjoint regions were found.")
    assert not value_is_bound(6, "Six hundred regions were found.")
    assert not value_is_bound(20, "Twenty-three regions were found.")


def test_single_group_bad_heading_does_not_erase_independently_bound_results():
    data = example()
    data[-1]["result_groups"][0]["evidence_text"] = "A paraphrased heading."
    result = evaluate(data)
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["metrics"]["c_score"]["value"] == 1


def test_unbound_group_with_multiple_methods_keeps_f_but_flags_c_uncertainty():
    answer, meta, gt, material, review = two_branches()
    other = deepcopy(review["result_groups"][0])
    other.update(group_id="supplement", purpose="SUPPLEMENTARY", evidence_text="No such heading.")
    review["result_groups"].append(other)
    claim = deepcopy(review["findings"][0])
    claim.update(group_id="supplement", finding_id="extra")
    review["findings"].append(claim)
    result = evaluate((answer, meta, gt, material, review))
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["metrics"]["c_score"]["lower"] == .5
    assert result["metrics"]["c_score"]["upper"] == 1


def test_group_inheritance_is_explicit_and_preserves_source_bound_choices():
    data = example()
    data[-1]["result_groups"].append({"group_id": "s", "purpose": "SUPPLEMENTARY",
        "evidence_text": "The method uses pressure.", "inherits_group_id": "main", "dimensions": []})
    extra = dict(data[-1]["findings"][0], group_id="s", finding_id="s_cv")
    data[-1]["findings"].append(extra)
    assert evaluate(data)["metrics"]["c_score"]["value"] == 1
    data[-1]["result_groups"][1]["inherits_group_id"] = "s"
    with pytest.raises(ValueError, match="inherit only"):
        evaluate(data)


def test_precomputed_support_is_supplementary_and_exact_counts_are_checkable():
    from flowintentbench.reference_packages import materialized_support_record, reporting_precision_rules
    answer, meta, gt, material, review = example()
    material["branch_execution_evidence"] = {"o": {"materialized_G_of_O": {"selected_region_size": 7},
        "source_artifact": {"path": "frozen.json", "sha256": "checked by loader"}, "evidence_binding_sha256": "bound"}}
    record = materialized_support_record(None, meta, gt, material)
    assert record["supporting_findings"][0]["finding"]["importance"] == "supporting"
    record["reporting_policies"] = reporting_precision_rules(gt, record["supporting_findings"])
    assert not any("selected_region_size" in p["finding_id"] for p in record["reporting_policies"])
    augmented, mat = apply_supplement(None, meta, gt, material, {"record": record, "source": {}})
    review["findings"].append(dict(review["findings"][0], finding_id="size", statement="selected region point count",
        evidence_text="Region contains 7 points.", value=7, unit="points", matches=[{"branch_id": "o", "finding_id": "o_support_selected_region_size"}]))
    result = evaluate((answer + " Region contains 7 points.", meta, augmented, mat, review))
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["metrics"]["core_finding_recall"]["value"] == 1


def test_reference_supplement_accepts_explicit_null_f1_role_contract():
    from flowintentbench.reference_packages import materialized_support_record
    _, meta, gt, mat, _ = example()
    mat["finding_requirement_contract"] = None
    record = materialized_support_record(None, meta, gt, mat)
    augmented, material = apply_supplement(None, meta, gt, mat, {"record": record, "source": {}})
    assert material["finding_requirement_contract"] is None
    assert len(augmented.findings_by_operationalization) == 1


def test_f2_missing_numeric_role_is_not_replaced_by_prose_or_extra_findings():
    answer, meta, gt, material, review = example()
    second = gt.findings_by_operationalization[0].findings[0].model_copy(update={
        "finding_id": "mean", "statement": "Mean pressure", "value": 7})
    gt.findings_by_operationalization[0].findings.append(second)
    material["finding_verification_policy"]["policies"].append({
        **material["finding_verification_policy"]["policies"][0], "finding_id": "mean"})
    material["finding_requirement_contract"] = {"mandatory_roles": ["dispersion", "level"],
        "adequate_core_sets": [["dispersion", "level"]], "reference_finding_role_map": {"cv": "dispersion", "mean": "level"}}
    quote = "Mean pressure is high."
    review["findings"].append(dict(review["findings"][0], finding_id="mean_prose", statement=quote,
        evidence_text=quote, value=None, matches=[{"branch_id": "o", "finding_id": "mean"}]))
    result = evaluate((answer + " " + quote, meta, gt, material, review), "O1-F2")
    assert result["metrics"]["finding_requirement_recall"]["value"] == .5
    assert result["metrics"]["adequate_core_complete"]["value"] == 0
    assert any(x["reason"] == "REQUIRED_NUMERIC_RESULT_MISSING" for x in result["requirement_items"])
    assert result["metrics"]["finding_precision"]["lower"] == .5


def test_reviewed_new_method_closes_numeric_gap_without_changing_original_requirements():
    from flowintentbench.ground_truth import OperationalizationBundle
    answer, meta, gt, material, review = example()
    gt.acceptable_operationalizations[0] = OperationalizationBundle.model_validate({"operationalization_id": "o",
        "decisions": [{"dimension": "feature_definition", "statement": "Use pressure."},
                      {"dimension": "property_measure", "statement": "Uniform weighting."}]})
    ref = gt.findings_by_operationalization[0].findings[0].model_dump(mode="json")
    ref.update(finding_id="new_cv", value=.8)
    addition = {"operationalization": {"operationalization_id": "new", "decisions": [
        {"dimension": "feature_definition", "statement": "Use pressure."},
        {"dimension": "property_measure", "statement": "Alternative weighting."}]},
        "findings": {"operationalization_id": "new", "findings": [ref]},
        "requirement_branch_id": "o", "requirement_map": {"new_cv": "cv"},
        "policies": [{**material["finding_verification_policy"]["policies"][0],
                      "operationalization_id": "new", "finding_id": "new_cv"}]}
    supplement = {"record": {"task_sha256": task_identity(None, meta, gt, material), "new_branches": [addition],
        "authority": "analytic fixture", "reviewed_by": "test author", "rationale": "Known analytic values."}, "source": {}}
    augmented, mat = apply_supplement(None, meta, gt, material, supplement)
    answer = answer.replace("uniform", "alternative").replace("0.5", "0.8")
    review["result_groups"][0]["dimensions"][0]["matches"] = {"o": True, "new": True}
    review["result_groups"][0]["dimensions"][1].update(matches={"o": False, "new": True}, reason="VALID_ALTERNATIVE")
    review["findings"][0].update(value=.8, evidence_text="CV is 0.8.", matches=[{"branch_id": "new", "finding_id": "new_cv"}])
    result = evaluate((answer, meta, augmented, mat, review))
    assert all(result["metrics"][k]["value"] == 1 for k in ("o_score", "finding_precision", "core_finding_recall", "c_score"))
    assert len(gt.acceptable_operationalizations) == 1
    bad = deepcopy(supplement)
    bad["record"]["new_branches"][0]["requirement_map"] = {}
    with pytest.raises(ValueError, match="core obligations"):
        apply_supplement(None, meta, gt, material, bad)
    bad = deepcopy(supplement)
    bad["record"]["new_branches"][0]["operationalization"]["decisions"][0]["statement"] = "Use velocity."
    with pytest.raises(ValueError, match="fixed task constraints"):
        apply_supplement(None, meta, gt, material, bad)


def test_optional_correct_auxiliary_claim_cannot_inflate_known_precision(monkeypatch):
    answer, meta, gt, material, review = example()
    answer = answer.replace("0.5", "0.9") + " Extra is 7."
    review["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    extra = dict(review["findings"][0], finding_id="extra", value=7, evidence_text="Extra is 7.",
        eligible=None, matches=[])
    review["findings"].append(extra)
    monkeypatch.setattr(rubric, "verify_checks", lambda mat, ans, c, *args:
        (True, [{"query": {"branch_id": "o"}, "verdict": True}]) if c.finding_id == "extra" else (None, []))
    result = evaluate((answer, meta, gt, material, review))
    assert result["metrics"]["finding_precision"]["lower"] == 0
    assert result["metrics"]["c_score"]["lower"] == 0


def test_reference_failure_prevents_paid_calls_even_when_other_cases_are_valid(tmp_path,monkeypatch):
    from flowintentbench.external_file_evaluator import write_json
    from scripts import evaluate_answered_outcomes as credentials
    answer,meta,gt,material,_=example()
    export=tmp_path/'answers.json';manifest=tmp_path/'manifest.json'
    write_json(export,[{'model_id':'m','case_id':c,'trial':1,'answer':answer} for c in ('good','bad')])
    write_json(manifest,{'cases':[{'case_id':c,'condition':'O2-F1'} for c in ('good','bad')]})
    def load(root,item):
        if item['case_id']=='bad':raise ValueError('conflicting reference definitions')
        return None,deepcopy(meta),deepcopy(gt),deepcopy(material)
    def paid_start(*args):raise AssertionError('reference failure must stop before API configuration')
    monkeypatch.setattr(runner,'load_development_case',load)
    monkeypatch.setattr(runner,'review_prompt',lambda *args:'controlled prompt')
    monkeypatch.setattr(credentials,'configure_review_credentials',paid_start)
    args=runner.parser().parse_args(['--answers',str(export),'--manifest',str(manifest),'--output',str(tmp_path/'out')])
    report=runner.run(args)
    assert report['evaluation_cost']['api_calls']==0
    assert report['evaluation_cost']['startup_error']=='REFERENCE_PREFLIGHT_FAILED'
    assert set(report['evaluation_cost']['reference_preflight_errors'])=={'bad'}
    assert report['status']=='INCOMPLETE'


def test_complete_generic_run_deduplicates_current_requests_resumes_and_freezes_identity(tmp_path, monkeypatch):
    from flowintentbench.external_file_evaluator import write_json
    from scripts import evaluate_answered_outcomes as credentials, outcome_responses_transport as transport
    answer, meta, gt, material, review = example()
    export = tmp_path / "answers.json"
    write_json(export, [{"model_id": model, "case_id": "c", "trial": 1, "answer": answer} for model in ("luna", "terra")])
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"cases": [{"case_id": "c", "condition": "O2-F1"}]})
    monkeypatch.setattr(runner, "load_development_case", lambda *_: (None, deepcopy(meta), deepcopy(gt), deepcopy(material)))
    monkeypatch.setattr(runner, "review_prompt", lambda *args: "controlled unit-test packet " + args[-1])
    monkeypatch.setattr(credentials, "configure_review_credentials", lambda *_: None)
    calls = []
    class Reviewer:
        def __init__(self, *a, **k):
            pass
        def __call__(self, prompt, model, work, api_dir, **kwargs):
            calls.append(prompt)
            write_json(api_dir / "request.json", {"input": [{"content": prompt}], "max_output_tokens": kwargs["max_output_tokens"]})
            receipt = {"completed": True, "response_model": model, "reasoning_effort": kwargs["reasoning_effort"],
                "final_text": json.dumps(review), "usage": {"input_tokens": 100, "output_tokens": 100}}
            write_json(api_dir / "receipt.json", receipt)
            return receipt
    monkeypatch.setattr(transport, "ResponsesOutcomeReviewer", Reviewer)
    args = runner.parser().parse_args(["--answers", str(export), "--manifest", str(manifest),
        "--output", str(tmp_path / "out"), "--max-api-calls", "2"])
    report = runner.run(args)
    assert report["status"] == "COMPLETE" and len(calls) == 1
    assert report["evaluation_cost"]["api_calls"] == 1
    assert all(r["metrics"]["finding_precision"]["value"] == 1 for r in report["rows"])
    args.offline = True
    resumed = runner.run(args)
    assert resumed["status"] == "COMPLETE" and len(calls) == 1
    assert resumed["evaluation_cost"]["api_calls"] == 0
    assert resumed["cumulative_new_review_cost"]["api_calls"] == 1
    args.replay_from = args.output
    args.output = tmp_path / "host-replay"
    args.max_api_calls = 0
    replayed = runner.run(args)
    assert replayed["status"] == "COMPLETE" and len(calls) == 1
    assert replayed["evaluation_cost"]["api_calls"] == 0
    assert all(r["provenance"].get("host_replay_source") for r in replayed["rows"])
    args.additional_replay_from=[args.output]
    args.replay_from=None
    args.output=tmp_path/'additional-source-replay'
    extra=runner.run(args)
    assert extra['status']=='COMPLETE' and len(calls)==1
    assert extra['evaluation_cost']['api_calls']==0
    assert all(r['metrics']['finding_precision']['value']==1 for r in extra['rows'])
    args.effort = "low"
    with pytest.raises(ValueError, match="inputs/code/reviewer changed"):
        runner.run(args)


def test_v4_keeps_auxiliary_scalar_errors_separate_and_core_credit_intact():
    answer, meta, gt, material, review = example()
    quote = "Layer range is 0.80 to 0.90 and median is 0.85 over all z layers."
    stats = ["layer_pearson_min", "layer_pearson_max", "layer_pearson_median"]
    review["findings"].append({"finding_id": "layers", "group_id": "main", "mapping_complete": True,
        "statement": "layer correlations", "evidence_text": quote, "value": [.8, .9, .85], "unit": None,
        "eligible": True, "matches": [], "numeric_checks": [{"branch_id": "o", "statistic": stat, "value_index": i,
            "axis": "z", "method_evidence_text": quote, "scope_evidence_text": quote, "scope_status": "EXPLICIT"} for i, stat in enumerate(stats)]})
    material["branch_execution_evidence"] = {"o": {"evidence_binding_sha256": "context",
        "execution_provenance": {"recipe": {"kind": "association", "measure": "pearson"}}}}
    material["frozen_auxiliary_evidence"] = [{"query": {"branch_id": "o", "statistic": stat, "axis": "z",
        "lower_x": None, "upper_x": None}, "expected": expected, "branch_evidence_sha256": "context", "source": {}}
        for stat, expected in zip(stats, [.2, .4, .3])]
    result = evaluate((answer + " " + quote, meta, gt, material, review))
    assert result["applicable_finding_count_upper"] == 4
    assert result["metrics"]["finding_precision"]["value"] == .25
    assert result["metrics"]["c_score"]["value"] == .25
    assert result["metrics"]["core_finding_recall"]["value"] == 1


def test_explicit_numerical_convention_difference_is_not_equivalent_gt():
    answer, meta, gt, material, review = example()
    quote = "I used absolute tetrahedral volumes."
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(dimension=SimpleNamespace(value="property_measure"),
        statement="Weight by the absolute signed cell volume from VTK and normalize by the included volume.")]
    review["result_groups"][0]["dimensions"][1]["evidence_text"] = quote
    result = evaluate((answer + " " + quote, meta, gt, material, review))
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["finding_precision"]["value"] is None
    assert result["result_groups"][0]["novel_method"]


def test_units_normalization_and_duplicate_restating_do_not_change_scores():
    answer, meta, gt, material, review = example()
    ref = gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings[0] = ref.model_copy(update={"statement": "coefficient of variation", "unit": "1"})
    review["findings"][0].update(value=50, evidence_text="CV is 50%.")
    other = deepcopy(review["findings"][0])
    other["finding_id"] = "duplicate"
    review["findings"].append(other)
    result = evaluate((answer.replace("CV is 0.5.", "CV is 50%."), meta, gt, material, review))
    assert result["metrics"]["finding_precision"]["value"] == 1
    assert result["duplicate_finding_ids"] == ["duplicate"]


def test_supplementary_method_outside_fixed_o_is_unknown_not_wrong():
    answer, meta, gt, material, review = example()
    meta.unresolved_operationalization_dimensions = []
    supplement = deepcopy(review['result_groups'][0])
    supplement.update(group_id='sensitivity', purpose='SUPPLEMENTARY', evidence_text='Supplementary CV is 0.8.')
    for d in supplement['dimensions']:
        d['matches']['o'] = False
    review['result_groups'].append(supplement)
    f = deepcopy(review['findings'][0])
    f.update(finding_id='extra', group_id='sensitivity', evidence_text='Supplementary CV is 0.8.', value=.8)
    review['findings'].append(f)
    result = evaluate((answer + ' Supplementary CV is 0.8.', meta, gt, material, review), 'O1-F1')
    assert result['metrics']['o_score']['value'] == 1
    assert result['metrics']['core_finding_recall']['value'] == 1
    assert result['metrics']['c_score']['lower'] == .5
    assert result['metrics']['c_score']['upper'] == 1
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []


@pytest.mark.parametrize('publicly_specified,match,unknown', [
    (False, None, True), (True, None, False), (False, True, False),
])
def test_fixed_public_o_unknown_private_implementation_does_not_prove_numeric_error(
        publicly_specified, match, unknown):
    answer, meta, gt, material, review = example()
    meta.unresolved_operationalization_dimensions = []
    dimension = review['result_groups'][0]['dimensions'][0]
    dimension['matches']['o'] = match
    material['branch_interpretations'] = {'o': {'dimension': dimension['dimension'],
        'description': 'Frozen implementation detail', 'publicly_specified': publicly_specified}}
    review['findings'][0].update(value=.8, evidence_text='CV is 0.8.')
    result = evaluate((answer.replace('CV is 0.5.', 'CV is 0.8.'), meta, gt, material, review), 'O1-F1')
    if unknown:
        assert result['metrics']['o_score']['value'] == 1
        for name in ('finding_precision', 'core_finding_recall', 'finding_requirement_recall', 'c_score'):
            assert result['metrics'][name]['value'] is None
            assert result['metrics'][name]['upper'] == 1
        assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []
        assert any(c['reason']=='ALTERNATIVE_METHOD_REFERENCE_GAP' for c in result['finding_checks'])
    else:
        assert result['metrics']['finding_precision']['value'] == 0


@pytest.mark.parametrize('scope,method', [
    ('band-restricted', 'volume-weighted correlation'),
    ('all cells', 'unweighted cell correlation'),
    ('all stored points', 'point-level correlation'),
])
def test_incompatible_auxiliary_population_is_not_refuted_by_global_reference(scope, method):
    from flowintentbench.frozen_evidence import verify_checks
    from types import SimpleNamespace
    from flowintentbench.frozen_evidence import NumericCheck
    quote = 'The value is 0.9.'
    claim = SimpleNamespace(evidence_text=quote, value=.9, unit=None,
        numeric_checks=[NumericCheck(branch_id='o', statistic='pearson', method_evidence_text=method,
                                     scope_evidence_text=scope, scope_status='EXPLICIT')])
    material = {'branch_execution_evidence': {'o': {'evidence_binding_sha256': 'context',
        'execution_provenance': {'recipe': {'kind': 'association', 'measure': 'pearson'}}}},
        'frozen_auxiliary_evidence': [{'query': {'branch_id': 'o', 'statistic': 'pearson', 'axis': None,
            'lower_x': None, 'upper_x': None}, 'expected': .2, 'branch_evidence_sha256': 'context', 'source': {}}]}
    value, checks = verify_checks(material, ' '.join([quote, scope, method]), claim, {'o'})
    assert value is None
    assert 'REFERENCE_MISMATCH' in checks[0]['reason']


def test_core_recall_floor_tightens_precision_and_extra_unknown_does_not_block_alignment():
    answer, meta, gt, material, review = two_branches()
    extra = deepcopy(review['findings'][0])
    extra.update(finding_id='extra', matches=[], evidence_text='Extra is 7.', value=7)
    review['findings'].append(extra)
    result = evaluate((answer + ' Extra is 7.', meta, gt, material, review))
    assert result['metrics']['finding_precision']['lower'] == .5
    assert result['metrics']['branch_alignment']['value'] == 1
    assert result['branch_alignment_diagnostic']['informative']


def test_duplicate_extraction_row_ids_are_repaired_without_dropping_claims():
    answer, meta, gt, material, review = example()
    extra = deepcopy(review['findings'][0])
    extra.update(matches=[], evidence_text='Extra is 7.', value=7)
    review['findings'].append(extra)
    result = evaluate((answer + ' Extra is 7.', meta, gt, material, review))
    assert result['applicable_finding_count_upper'] == 2
    assert result['metrics']['finding_precision']['lower'] == .5
    assert any(r['reason'] == 'DUPLICATE_ROW_ID_RENAMED' for r in result['review_repairs'])


def test_native_dimensionless_description_preserves_stored_value_but_not_si_conversion():
    answer, meta, gt, material, review = example()
    quote = 'Stored units are dimensionless.'
    review['findings'][0].update(unit='dimensionless', unit_evidence_text=quote)
    assert evaluate((answer + ' ' + quote, meta, gt, material, review))['metrics']['core_finding_recall']['value'] == 1
    review['findings'][0].update(unit='m/s', unit_evidence_text='SI units are m/s.')
    assert evaluate((answer + ' SI units are m/s.', meta, gt, material, review))['metrics']['core_finding_recall']['value'] is None


def test_wrong_quantity_reference_edges_do_not_make_core_unknown_or_refute_mean():
    answer, meta, gt, material, review = two_branches()
    for b in gt.findings_by_operationalization:
        b.findings[0] = b.findings[0].model_copy(update={'unit': '1'})
    gt.acceptable_operationalizations[0].decisions = [SimpleNamespace(dimension=SimpleNamespace(value='property_measure'),
        statement='Use the volume-weighted standard deviation divided by the absolute volume-weighted mean.')]
    gt.acceptable_operationalizations[1].decisions = [SimpleNamespace(dimension=SimpleNamespace(value='property_measure'),
        statement='Use the 90th-minus-10th percentile range divided by the median.')]
    review['findings'][0].update(statement='Coefficient of variation CV')
    extra = deepcopy(review['findings'][0])
    extra.update(finding_id='mean', statement='Mean pressure', evidence_text='Mean pressure is 12 bar.', value=12, unit='bar')
    review['findings'].append(extra)
    result = evaluate((answer + ' Mean pressure is 12 bar.', meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] == 1
    assert result['metrics']['finding_precision']['lower'] == .5
    assert result['metrics']['branch_alignment']['value'] == 1
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == []


@pytest.mark.parametrize('mode', ['GROUPS', 'METHOD_SCOPE', 'METHOD_SCOPE_BLIND'])
def test_review_repair_preserves_claims_and_rejects_unbound_receipt(tmp_path, mode):
    from flowintentbench.review_repairs import repair_prompt, repair_schema, apply_repair
    from flowintentbench.external_file_evaluator import write_json
    _, _, _, _, judgment = example()
    prompt = 'Anonymized original task and answer'
    if mode == 'METHOD_SCOPE_BLIND':
        prompt += '\n\n' + json.dumps({'answer':'L1: Mean 7.',
            'reference_package':{'dimensions':[], 'branches':[]}})
    patch = {'result_groups': deepcopy(judgment['result_groups']), 'finding_groups': {}}
    request = {'input': [{'content': repair_prompt(prompt, judgment, mode)}], 'max_output_tokens': 1000,
        'instructions': 'Return only JSON matching the supplied schema. No tools.\n' + json.dumps(repair_schema(mode), ensure_ascii=False)}
    receipt = {'completed': True, 'response_model': 'reviewer', 'reasoning_effort': 'medium', 'final_text': json.dumps(patch)}
    write_json(tmp_path/'request.json', request)
    write_json(tmp_path/'receipt.json', receipt)
    record = {'base_judgment_sha256': rubric.digest(judgment), 'base_prompt_sha256': rubric.digest(prompt),
        'receipt': str(tmp_path/'receipt.json'), 'model': 'reviewer', 'effort': 'medium', 'max_output_tokens': 1000, 'repair': patch, 'mode': mode}
    assert apply_repair(judgment, prompt, record)['findings'] == judgment['findings']
    altered = deepcopy(judgment)
    altered['findings'][0]['value'] = 9
    with pytest.raises(ValueError, match='another answer/review'):
        apply_repair(altered, prompt, record)
    receipt['completed'] = False
    write_json(tmp_path/'receipt.json', receipt)
    with pytest.raises(ValueError, match='completed'):
        apply_repair(judgment, prompt, record)


def test_blind_method_audit_does_not_repeat_old_verdict_or_numeric_reference():
    from flowintentbench.review_repairs import repair_prompt
    _, _, _, _, judgment = example()
    judgment['result_groups'][0]['dimensions'][0]['rationale'] = 'UNSUPPORTED_OLD_VERDICT'
    packet = {'answer':'L1: VTK cell centers; average vertex velocity vectors.',
        'reference_package':{'dimensions':[], 'branches':[{'branch_id':'B0', 'decisions':[],
            'input_population':{'description':'VTK parametric center', 'publicly_specified':False},
            'findings':[{'value':987654321}]}], 'shared_supporting_findings':[{'value':123456789}]},
        'frozen_auxiliary_catalog':{'secret_value':246813579}}
    prompt = repair_prompt('Instructions\n\n'+json.dumps(packet), judgment, 'METHOD_SCOPE_BLIND')
    assert 'UNSUPPORTED_OLD_VERDICT' not in prompt
    assert all(str(n) not in prompt for n in (987654321,123456789,246813579))
    bound = json.loads(prompt.rsplit('\n\n',1)[1])
    assert bound['answer'] == packet['answer']
    assert bound['reference_package']['branches'][0]['input_population']['publicly_specified'] is False
    assert 'averaging vertex FIELD values does not imply averaging vertex COORDINATES' in prompt


def test_blind_method_audit_preserves_legacy_nested_claims_without_inventing_role():
    from flowintentbench.review_repairs import repair_prompt
    legacy = {'result_groups': [{'group_id': 'g', 'findings': [
        {'finding_id': 'f', 'statement': 'Reported peak', 'value': 123456789,
         'source_start_line': 1, 'source_end_line': 1}],
        'matches': [{'choice': 'UNSUPPORTED_OLD_VERDICT'}]}]}
    original = deepcopy(legacy)
    packet = {'answer': 'L1: Reported peak is 2.',
              'reference_package': {'dimensions': [], 'branches': []}}
    prompt = repair_prompt('Instructions\n\n' + json.dumps(packet), legacy, 'METHOD_SCOPE_BLIND')
    data = json.loads(prompt.rsplit('\n\n', 1)[1])
    assert data['existing_groups'] == [{'group_id': 'g'}]
    assert data['finding_attribution'] == [{'finding_id': 'f', 'group_id': 'g',
        'statement': 'Reported peak', 'source_start_line': 1, 'source_end_line': 1}]
    assert '123456789' not in prompt and 'UNSUPPORTED_OLD_VERDICT' not in prompt
    assert legacy == original


def test_chained_repairs_preserve_verified_extraction_and_reject_mutated_history(tmp_path):
    from flowintentbench.review_repairs import repair_prompt, repair_schema, apply_repair
    from flowintentbench.external_file_evaluator import write_json
    import hashlib
    _, _, _, _, original = example()
    prompt = 'Original source-bound task'
    extracted = deepcopy(original)
    extracted['findings'][0]['statement'] = 'Corrected extraction label'
    prior = None
    current = original
    for i, (mode, patch) in enumerate([
        ('EXTRACTION', extracted),
        ('GROUPS', {'result_groups':deepcopy(extracted['result_groups']), 'finding_groups':{}})]):
        folder = tmp_path/str(i)
        write_json(folder/'request.json', {'input':[{'content':repair_prompt(prompt,current,mode)}],
            'max_output_tokens':1000, 'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(repair_schema(mode))})
        write_json(folder/'receipt.json', {'completed':True, 'response_model':'reviewer',
            'reasoning_effort':'medium', 'final_text':json.dumps(patch)})
        record = {'base_judgment_sha256':rubric.digest(current), 'base_prompt_sha256':rubric.digest(prompt),
            'receipt':str(folder/'receipt.json'), 'model':'reviewer','effort':'medium',
            'max_output_tokens':1000,'mode':mode,'repair':patch}
        if prior:
            record['prior_repair'] = {'path':str(prior),'sha256':hashlib.sha256(prior.read_bytes()).hexdigest()}
        current = apply_repair(original,prompt,record)
        write_json(folder/'repair.json',record)
        prior = folder/'repair.json'
    assert current['findings'] == extracted['findings']
    (tmp_path/'0/repair.json').write_text('{}')
    with pytest.raises(ValueError,match='prior repair checksum'):
        apply_repair(original,prompt,record)


@pytest.mark.parametrize('request_protocol,raw_protocol,valid',[
    (rubric.VERSION,'ABSENT',True),('wrong-version','ABSENT',False),
    (rubric.VERSION,None,False),(rubric.VERSION,'wrong-version',False)])
@pytest.mark.parametrize('patch_protocol', ['PRESENT', 'ABSENT', 'wrong-version', None])
def test_extraction_repair_missing_protocol_uses_only_bound_original_request(tmp_path,request_protocol,raw_protocol,valid,patch_protocol):
    from flowintentbench.review_repairs import repair_prompt,repair_schema,apply_repair
    from flowintentbench.external_file_evaluator import write_json
    _,_,_,_,patch=example()
    legacy={'result_groups':deepcopy(patch['result_groups'])}
    legacy['result_groups'][0]['findings']=deepcopy(patch['findings'])
    if raw_protocol!='ABSENT':legacy['protocol']=raw_protocol
    expected = deepcopy(patch)
    if patch_protocol == 'ABSENT':
        patch.pop('protocol')
    elif patch_protocol != 'PRESENT':
        patch['protocol'] = patch_protocol
    valid = valid and patch_protocol in {'PRESENT','ABSENT'}
    prompt='Original instructions\n\n'+json.dumps({'protocol':request_protocol,'answer':'Original answer'})
    write_json(tmp_path/'request.json',{'input':[{'content':repair_prompt(prompt,legacy,'EXTRACTION')}],
        'max_output_tokens':1000,'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(repair_schema('EXTRACTION'))})
    write_json(tmp_path/'receipt.json',{'completed':True,'response_model':'reviewer',
        'reasoning_effort':'medium','final_text':json.dumps(patch)})
    record={'base_judgment_sha256':rubric.digest(legacy),'base_prompt_sha256':rubric.digest(prompt),
        'receipt':str(tmp_path/'receipt.json'),'model':'reviewer','effort':'medium',
        'max_output_tokens':1000,'mode':'EXTRACTION','repair':patch}
    if valid:
        assert apply_repair(legacy,prompt,record)==expected
        if patch_protocol == 'ABSENT':assert 'protocol' not in patch
        assert 'protocol' not in legacy
    else:
        with pytest.raises(ValueError,match='protocol mismatch'):apply_repair(legacy,prompt,record)


def test_frozen_layer_reference_can_check_scoped_inherited_method():
    answer, meta, gt, material, review = example()
    from types import SimpleNamespace
    meta.principal_operationalization_dimensions.append(SimpleNamespace(value='aggregation_or_representation'))
    parent = review['result_groups'][0]
    parent['dimensions'].append(dict(parent['dimensions'][0], dimension='aggregation_or_representation'))
    quote = 'All z layers have volume-weighted correlation minimum -0.63.'
    group = {'group_id': 'layers', 'purpose': 'SUPPLEMENTARY', 'evidence_text': quote,
        'inherits_group_id': 'main', 'dimensions': [dict(parent['dimensions'][-1], matches={'o': False},
            evidence_text=quote, adequacy='MET', reason='VALID_ALTERNATIVE')]}
    review['result_groups'].append(group)
    review['findings'].append({'finding_id': 'layer_min', 'group_id': 'layers', 'mapping_complete': True,
        'statement': 'layer minimum', 'evidence_text': quote, 'value': -.63, 'unit': None, 'eligible': True, 'matches': [],
        'numeric_checks': [{'branch_id': 'o', 'statistic': 'layer_pearson_min', 'axis': 'z',
            'method_evidence_text': 'volume-weighted correlation', 'scope_evidence_text': 'All z layers', 'scope_status': 'EXPLICIT'}]})
    material['branch_execution_evidence'] = {'o': {'evidence_binding_sha256': 'context',
        'execution_provenance': {'recipe': {'kind': 'association', 'measure': 'pearson'}}}}
    material['frozen_auxiliary_evidence'] = [{'query': {'branch_id': 'o', 'statistic': 'layer_pearson_min', 'axis': 'z',
        'lower_x': None, 'upper_x': None}, 'expected': -.9258, 'branch_evidence_sha256': 'context', 'source': {}}]
    result = evaluate((answer + ' ' + quote, meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] == 1
    assert result['metrics']['finding_precision']['value'] == .5
    assert result['metrics']['c_score']['value'] == .5
    assert result['error_diagnostics']['independent_contradicted_finding_ids'] == ['layer_min']


def test_missing_supplementary_dimension_localizes_review_error():
    answer, meta, gt, material, review = example()
    quote = 'The separate diagnostic is 7.'
    review['result_groups'].append({'group_id': 'diagnostic', 'purpose': 'SUPPLEMENTARY', 'evidence_text': quote, 'dimensions': []})
    extra = deepcopy(review['findings'][0])
    extra.update(finding_id='extra', group_id='diagnostic', matches=[], evidence_text=quote, value=7)
    review['findings'].append(extra)
    result = evaluate((answer + ' ' + quote, meta, gt, material, review))
    assert result['status'] == 'SCORED'
    assert result['metrics']['o_score']['value'] == 1
    assert result['metrics']['core_finding_recall']['value'] == 1
    assert result['metrics']['c_score']['lower'] == .5
    assert result['metrics']['c_score']['upper'] == 1


def test_unbound_open_method_cannot_be_refuted_solely_by_gt_disagreement():
    answer, meta, gt, material, review = example()
    answer = answer.replace('0.5', '0.9')
    review['findings'][0].update(evidence_text='CV is 0.9.', value=.9)
    for d in review['result_groups'][0]['dimensions']:
        d['evidence_text'] = ''
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['finding_precision']['upper'] == 1
    assert not result['error_diagnostics']['independent_contradicted_finding_ids']


def test_proven_best_branch_precision_is_not_diluted_by_possible_unknown_ties():
    answer, meta, gt, material, review = two_branches()
    gt.findings_by_operationalization[1].findings[0] = gt.findings_by_operationalization[1].findings[0].model_copy(update={'value':.5})
    ref = ReferenceFinding.model_validate({'finding_id':'b_extra','category':'quantity','importance':'supporting',
        'statement':'Additional empirical quantity','value':7,'verification':{'absolute_tolerance':.01}})
    gt.findings_by_operationalization[1].findings.append(ref)
    material['finding_verification_policy']['policies'].append({'operationalization_id':'b','finding_id':'b_extra',
        'verification_mode':'scalar_tolerance','verification_parameters':ref.verification.model_dump()})
    for d in review['result_groups'][0]['dimensions']:
        d['matches'] = {'o':False,'b':True}
    answer += ' Additional empirical quantity is 7.'
    review['findings'].append({'finding_id':'extra','statement':'Additional empirical quantity','value':7,'unit':None,
        'eligible':True,'evidence_text':'Additional empirical quantity is 7.','group_id':'main','mapping_complete':True,
        'matches':[{'branch_id':'b','finding_id':'b_extra'}]})
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['finding_precision']['value'] == 1
    assert result['metrics']['c_score']['value'] == 1
    assert result['metrics']['branch_alignment']['value'] == 1
    assert result['branch_alignment_diagnostic']['kind'] == 'IDENTIFIED_WITH_UNRESOLVED_TIES'
    assert result['branch_alignment_diagnostic']['informative'] is False
    assert not result['reference_gaps']
    # A contradicted extra cannot receive the guaranteed precision floor.
    review['findings'][1].update(value=9, evidence_text='Additional empirical quantity is 9.')
    result = evaluate((answer.replace('quantity is 7.', 'quantity is 9.'), meta, gt, material, review))
    assert result['metrics']['finding_precision']['value'] is None
    assert result['metrics']['c_score']['value'] == .5


def test_labeled_finding_matches_preserve_only_explicit_relations():
    answer, meta, gt, material, review = example()
    claim = review['findings'][0]
    claim['matches'] = {'true':claim['matches'], 'false':[], 'null':[]}
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] == 1
    claim['matches'] = {'true':[], 'false':[], 'null':[{'branch_id':'o','finding_id':'cv'}]}
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] is None


def test_prose_reason_without_matches_does_not_invent_equivalence():
    answer, meta, gt, material, review = example()
    for d in review['result_groups'][0]['dimensions']:
        d.pop('matches')
        d['reason'] = 'The declared method is coherent.'
    review['result_groups'][0]['dimensions'][1]['classification'] = 'VALID_ALTERNATIVE'
    result = evaluate((answer, meta, gt, material, review))
    # Independent F may still be correct; conditional O-F cannot be inferred.
    assert result['metrics']['c_score']['value'] is None
    assert result['metrics']['o_score']['value'] is None
    assert result['status'] == 'SCORED'


@pytest.mark.parametrize('literal,expected', [('50%',1),('90%',0)])
def test_literal_percent_string_is_normalized_then_source_and_truth_checked(literal, expected):
    answer, meta, gt, material, review = example()
    ref = gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings[0] = ref.model_copy(update={'unit':'1'})
    quote = f'CV is {literal}.'
    review['findings'][0].update(value=literal,unit='1',evidence_text=quote)
    result = evaluate((answer.replace('CV is 0.5.',quote), meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] == expected
    result = evaluate((answer, meta, gt, material, review))
    assert result['metrics']['core_finding_recall']['value'] is None


def test_reference_filtered_extraction_cannot_claim_complete_precision():
    answer, meta, gt, material, review = example()
    review['limitations'] = 'Only catalogued scientific quantities are represented; unreferenced claims are omitted.'
    result = evaluate((answer, meta, gt, material, review))
    assert result['extraction_complete'] is False
    assert result['metrics']['finding_precision']['value'] is None
    assert result['metrics']['core_finding_recall']['value'] == 1
    assert result['metrics']['o_score']['value'] == 1
    review['limitations'] = 'Unreferenced quantities are included and remain unverified.'
    assert evaluate((answer, meta, gt, material, review))['extraction_complete'] is True


def test_missing_adequacy_is_local_unknown_and_preserves_fixed_equivalence():
    answer, meta, gt, material, review = example()
    for d in review['result_groups'][0]['dimensions']:
        d.pop('adequacy')
    r=evaluate((answer,meta,gt,material,review))
    assert r['status']=='SCORED'
    assert r['metrics']['urs']['value'] is None
    assert r['metrics']['resolved_o_compliance']['value']==1
    assert r['metrics']['core_finding_recall']['value']==1


def test_conflicting_reviewer_dimensions_do_not_block_the_primary_answer():
    answer,meta,gt,material,review=example()
    group=deepcopy(review['result_groups'][0]);group.update(group_id='extra_group',purpose='SUPPLEMENTARY')
    duplicate=deepcopy(group['dimensions'][1]);duplicate['matches']={'o':False}
    group['dimensions'].append(duplicate);review['result_groups'].append(group)
    claim=deepcopy(review['findings'][0]);claim.update(finding_id='extra',group_id='extra_group',value=.7,evidence_text='Extra is 0.7.')
    review['findings'].append(claim)
    r=evaluate((answer+' Extra is 0.7.',meta,gt,material,review))
    assert r['status']=='SCORED' and r['metrics']['o_score']['value']==1
    assert r['metrics']['core_finding_recall']['value']==1
    assert any(x['reason']=='CONFLICTING_REVIEW_DIMENSIONS_UNKNOWN' for x in r['review_repairs'])
    assert r['metrics']['c_score']['value'] is None
    # Literal duplicates carry no additional scientific judgment.
    answer,meta,gt,material,review=example()
    review['result_groups'][0]['dimensions'].append(deepcopy(review['result_groups'][0]['dimensions'][0]))
    assert evaluate((answer,meta,gt,material,review))['metrics']['o_score']['value']==1


def test_nested_explicit_role_and_completeness_labels_are_recoverable():
    answer,meta,gt,material,review=example()
    g=review['result_groups'][0];g.pop('purpose');g['group_id']='PRIMARY'
    g['findings']=review.pop('findings');g['mapping_complete']=True
    for f in g['findings']:f.pop('group_id');f.pop('mapping_complete')
    g['extraction_complete']=review.pop('extraction_complete')
    r=evaluate((answer,meta,gt,material,review))
    assert r['metrics']['finding_precision']['value']==1 and r['extraction_complete']
    g['extraction_complete']=False
    assert evaluate((answer,meta,gt,material,review))['metrics']['finding_precision']['value'] is None


def test_missing_finding_row_id_is_not_a_scientific_failure():
    data=example();baseline=evaluate(deepcopy(data))
    data[-1]['findings'][0].pop('finding_id')
    result=evaluate(data)
    assert result['metrics']==baseline['metrics']
    assert any(x['reason']=='MISSING_ROW_ID_ASSIGNED' for x in result['review_repairs'])


def test_explicit_choices_alias_with_null_diagnostic_preserves_judgments():
    data=example();baseline=evaluate(deepcopy(data))
    g=data[-1]['result_groups'][0];g['choices']=g.pop('dimensions')
    for d in g['choices']:d['reason']=None
    assert evaluate(data)['metrics']==baseline['metrics']


def test_missing_group_link_uses_only_a_unique_declared_result_group():
    data=example();baseline=evaluate(deepcopy(data));data[-1]['findings'][0].pop('group_id')
    assert evaluate(data)['metrics']==baseline['metrics']
    other=deepcopy(data[-1]['result_groups'][0]);other.update(group_id='extra',purpose='SUPPLEMENTARY')
    data[-1]['result_groups'].append(other)
    with pytest.raises(ValueError,match='group_id'):
        evaluate(data)


def test_dictionary_choices_and_summary_verdicts_never_replace_item_evidence():
    data=example();g=data[-1]['result_groups'][0]
    g['choices']={d['dimension']:{k:v for k,v in d.items() if k!='dimension'} for d in g.pop('dimensions')}
    for d in g['choices'].values():d['status']='MET'
    data[-1]['adequacy']={'status':'MET'};data[-1]['reason']=None;data[-1]['auxiliary_numeric_checks']=[]
    g['branch_equivalence']={'o':True}
    assert evaluate(data)['metrics']['finding_precision']['value']==1
    for d in g['choices'].values():d['evidence_text']=''
    r=evaluate(data)
    assert r['metrics']['o_score']['value'] is None
    assert any(x['reason']=='GROUP_SUMMARY_MATCH_RETAINED' for x in r['review_repairs'])


def test_empty_supplementary_alias_does_not_reject_valid_review_or_hide_findings():
    data=example();baseline=evaluate(deepcopy(data))
    data[-1]['supplementary']=[]
    assert evaluate(data)['metrics']==baseline['metrics']
    data[-1]['supplementary']=[{'statement':'unmapped extra scientific claim'}]
    with pytest.raises(ValueError,match='supplementary'):
        evaluate(data)


def test_fixed_diagnostic_alias_preserves_explicit_source_bound_matches_only():
    data=example();baseline=evaluate(deepcopy(data))
    for d in data[-1]['result_groups'][0]['dimensions']:d['reason']='FIXED'
    assert evaluate(data)['metrics']==baseline['metrics']
    for d in data[-1]['result_groups'][0]['dimensions']:d['evidence_text']=''
    assert evaluate(data)['metrics']['o_score']['value'] is None


@pytest.mark.parametrize('label',['range','extrema','minimum and maximum'])
def test_range_is_scored_as_two_labeled_endpoints_and_wrong_endpoint_is_not_hidden(label):
    answer,meta,gt,mat,review=example()
    answer+=f' Other-region peak-speed {label} is 2.785 to 2.856.'
    template=gt.findings_by_operationalization[0].findings[0]
    for name,value in [('minimum',2.785),('maximum',2.856)]:
        ref=ReferenceFinding.model_validate(template.model_copy(update={'finding_id':name,'importance':'supporting',
            'statement':name.capitalize()+' of other-region peak speeds','value':value,'unit':None}).model_dump())
        gt.findings_by_operationalization[0].findings.append(ref)
        mat['finding_verification_policy']['policies'].append({**mat['finding_verification_policy']['policies'][0],'finding_id':name})
    review['findings'].append(dict(review['findings'][0],finding_id='range',statement=f'Other-region peak-speed {label}',
        value=[2.785,2.856],unit=None,evidence_text=f'Other-region peak-speed {label} is 2.785 to 2.856.',
        matches=[{'branch_id':'o','finding_id':name} for name in ('maximum','minimum')]))
    without=deepcopy(review)
    without['findings'][-1]['matches']=[]
    before=evaluate((answer,meta,gt,mat,without))
    after=evaluate((answer,meta,gt,mat,review))
    assert before['applicable_finding_count_upper']==after['applicable_finding_count_upper']==3
    assert before['metrics']['finding_precision']['lower']==pytest.approx(1/3)
    refs={b.operationalization_id:{f.finding_id:f for f in b.findings} for b in gt.findings_by_operationalization}
    prepared,_,_=rubric._prepare_review(without,refs,gt,answer)
    assert evaluate((answer,meta,gt,mat,prepared.model_dump(mode='json')))['metrics']==before['metrics']
    assert evaluate((answer,meta,gt,mat,review))['metrics']['finding_precision']['value']==1
    answer=answer.replace('2.856','3.856')
    review['findings'][-1].update(value=[2.785,3.856],evidence_text=f'Other-region peak-speed {label} is 2.785 to 3.856.')
    r=evaluate((answer,meta,gt,mat,review))
    assert r['metrics']['finding_precision']['value']==pytest.approx(2/3)
    assert any(c['finding_id']=='range::maximum' and c['verdict'] is False for c in r['finding_checks'])


@pytest.mark.parametrize('label',['95% confidence range','central range','interquartile range','coordinate centroid','local extrema'])
def test_range_atomization_preserves_interval_and_vector_results(label):
    answer,meta,gt,mat,review=example()
    review['findings'][0].update(statement=label,value=[.1,.9],matches=[])
    refs={b.operationalization_id:{f.finding_id:f for f in b.findings} for b in gt.findings_by_operationalization}
    prepared,_,_=rubric._prepare_review(review,refs,gt,answer)
    assert len(prepared.findings)==1
    assert prepared.findings[0].value==[.1,.9]


def test_range_mapped_to_scalar_width_does_not_turn_mapping_error_into_model_error():
    answer,meta,gt,mat,review=example()
    answer+=' Range is 0.2 to 0.8.'
    review['findings'][0].update(statement='Observed range',value=[.2,.8],evidence_text='Range is 0.2 to 0.8.')
    result=evaluate((answer,meta,gt,mat,review))
    assert result['applicable_finding_count_upper']==2
    assert result['metrics']['core_finding_recall']['upper']==1
    assert result['error_diagnostics']['independent_contradicted_finding_ids']==[]


def test_unspecified_vector_to_scalar_mapping_is_unknown_not_a_false_numeric_error():
    answer,meta,gt,mat,review=example()
    answer+=' Values are 0.5 and 0.6.'
    review['findings'][0].update(value=[.5,.6],evidence_text='Values are 0.5 and 0.6.')
    r=evaluate((answer,meta,gt,mat,review))
    assert r['metrics']['finding_precision']['value'] is None
    assert r['finding_checks'][0]['reason']=='REVIEW_VALUE_SHAPE_MISMATCH'


def test_space_grouped_counts_require_cardinality_context_and_never_join_vectors():
    from flowintentbench.answer_evidence import value_is_bound
    for literal,value in [('Zero-speed points: 2 520, exactly the wall.',2520),
            ('leaving 38 440 non-zero speeds.',38440),('3 844 points retained.',3844),
            ('Size: 3 831 of the selected points.',3831)]:
        assert value_is_bound(value,literal)
    assert not value_is_bound(3831400,'Coordinates: 3 831 400')
    assert value_is_bound([3,831,400],'Coordinates: 3 831 400')
    assert not value_is_bound(3844,'3\n844 points')


def test_extra_group_and_finding_labels_do_not_supply_scientific_credit():
    data=example();baseline=evaluate(deepcopy(data));g=data[-1]['result_groups'][0]
    g.update(heading='Summary heading label',adequacy='MET',reason='Reviewer summary')
    data[-1]['findings'][0].update(category='quantity',role='principal_feature_location')
    for d in g['dimensions']:d['population_choice']='A descriptive label'
    assert evaluate(data)['metrics']==baseline['metrics']
    g['dimensions']=[]
    data[-1]['dimension_choices']=[{'dimension':'feature_definition','selected_branch_id':'o'}]
    assert evaluate(data)['metrics']['o_score']['value'] is None


def test_explicit_dimension_equivalence_and_extraction_aliases_preserve_source_checks():
    data=example();baseline=evaluate(deepcopy(data))
    for d in data[-1]['result_groups'][0]['dimensions']:
        d['branch_equivalence']=d.pop('matches');d['extraction_status']=d.pop('status')
    assert evaluate(data)['metrics']==baseline['metrics']
    for d in data[-1]['result_groups'][0]['dimensions']:
        d.pop('extraction_status');d['evidence_text']=''
    assert evaluate(data)['metrics']['o_score']['value'] is None


def test_verifiable_label_is_not_itself_a_positive_judgment():
    data=example()
    for d in data[-1]['result_groups'][0]['dimensions']:
        d['status']='MET';d['adequacy']='VERIFIABLE'
    assert evaluate(data)['metrics']['o_score']['value']==1
    for d in data[-1]['result_groups'][0]['dimensions']:d['evidence_text']=''
    assert evaluate(data)['metrics']['o_score']['value'] is None


def test_strict_wire_schema_requires_all_fields_and_uses_line_citations_without_copied_prose():
    _,meta,gt,_,_=example()
    schema=rubric.review_schema(meta,gt,structured_output=True)
    def check(node):
        if isinstance(node,dict):
            assert 'default' not in node
            if node.get('type')=='object':
                assert set(node['required'])==set(node['properties'])
                assert node['additionalProperties'] is False
            for child in node.values():check(child)
        elif isinstance(node,list):
            for child in node:check(child)
    check(schema)
    assert schema['properties']['protocol']['type']=='string'
    assert schema['properties']['protocol']['enum']==[rubric.VERSION]
    dim=schema['$defs']['Dimension']['properties']
    assert 'rationale' not in dim and 'evidence_text' not in dim
    assert 'source_start_line' in dim and 'matches' in dim and 'adequacy' in dim


def test_dimension_map_and_title_alias_do_not_invent_evidence():
    data=example();baseline=evaluate(deepcopy(data));g=data[-1]['result_groups'][0]
    g['dimensions']={d['dimension']:{k:v for k,v in d.items() if k!='dimension'} for d in g['dimensions']}
    g['title']='A reviewer title'
    assert evaluate(data)['metrics']==baseline['metrics']


def test_duplicate_other_region_count_with_total_scope_mismatch_is_not_two_claims_or_an_error():
    answer,meta,gt,mat,review=example();answer+=' The other five regions are smaller.'
    template=gt.findings_by_operationalization[0].findings[0]
    for fid,value,text in [('total',6,'Number of connected regions in the retained mask'),
                           ('other',5,'Number of other retained connected regions')]:
        ref=ReferenceFinding.model_validate(template.model_copy(update={'finding_id':fid,'importance':'supporting',
            'statement':text,'value':value,'unit':'count',
            'verification':template.verification.model_copy(update={'absolute_tolerance':0})}).model_dump())
        gt.findings_by_operationalization[0].findings.append(ref)
        mat['finding_verification_policy']['policies'].append({'operationalization_id':'o','finding_id':fid,
            'verification_mode':'scalar_tolerance','verification_parameters':ref.verification.model_dump()})
        review['findings'].append(dict(review['findings'][0],finding_id=fid,statement='Other retained regions',value=5,
            unit='count',evidence_text='The other five regions are smaller.',matches=[{'branch_id':'o','finding_id':fid}]))
    r=evaluate((answer,meta,gt,mat,review))
    assert r['metrics']['finding_precision']['value']==1
    assert r['applicable_finding_count_upper']==2
    assert r['duplicate_finding_ids']==['other']
    assert not r['error_diagnostics']['independent_contradicted_finding_ids']
    # Even a generic reviewer label cannot turn a quoted "other five" into
    # an empirical claim that all regions total five.
    review['findings'][-2]['statement']='Number of connected regions'
    r=evaluate((answer,meta,gt,mat,review))
    assert not r['error_diagnostics']['independent_contradicted_finding_ids']
    ref=next(f for f in gt.findings_by_operationalization[0].findings if f.finding_id=='total')
    f=rubric.Finding.model_validate(review['findings'][-2]).model_copy(update={
        'statement':'Retained high-speed locations','value':3831,
        'evidence_text':'There are 3831 retained locations in the selected region.'})
    assert rubric._incompatible_reference_mapping(f.evidence_text,f,ref,gt.acceptable_operationalizations[0])


def test_top_level_dimensions_require_explicit_group_and_preserve_scores():
    data=example();baseline=evaluate(deepcopy(data));judgment=data[-1]
    group=judgment['result_groups'][0]
    judgment['dimensions']=[dict(d,group_id=group['group_id']) for d in group.pop('dimensions')]
    assert evaluate(data)['metrics']==baseline['metrics']
    for invalid in (None,'missing_group'):
        changed=deepcopy(data)
        if invalid is None:changed[-1]['dimensions'][0].pop('group_id')
        else:changed[-1]['dimensions'][0]['group_id']=invalid
        with pytest.raises(ValueError,match='explicit known group'):
            evaluate(changed)


def test_dimension_comment_does_not_replace_item_verdict():
    data=example();baseline=evaluate(deepcopy(data));d=data[-1]['result_groups'][0]['dimensions'][0]
    d['criterion']='Reviewer explanatory text, not source evidence.'
    result=evaluate(data)
    assert result['metrics']==baseline['metrics']
    assert any(r['reason']=='DIMENSION_COMMENT_RETAINED' for r in result['review_repairs'])


@pytest.mark.parametrize('layout',['anonymous','nested','incomplete_nested','top_dimensions'])
def test_group_repair_preserves_legacy_claims_and_completeness(tmp_path,layout):
    from flowintentbench.review_repairs import repair_prompt,repair_schema,apply_repair
    from flowintentbench.external_file_evaluator import write_json
    data=example();judgment=data[-1];patch={'result_groups':deepcopy(judgment['result_groups']),'finding_groups':{}}
    group=judgment['result_groups'][0]
    if layout=='anonymous':
        judgment['findings'][0].pop('finding_id');judgment['findings'][0].pop('group_id')
    elif layout in {'nested','incomplete_nested'}:
        group['findings']=judgment.pop('findings')
        group['findings'][0].pop('group_id')
        group['findings'][0].pop('mapping_complete')
        group['mapping_complete']=True
        group['extraction_complete']=layout!='incomplete_nested'
        judgment.pop('extraction_complete')
    else:
        judgment['dimensions']=[dict(d,group_id=group['group_id']) for d in group.pop('dimensions')]
    before=evaluate(deepcopy(data));prompt='Original request'
    write_json(tmp_path/'request.json',{'input':[{'content':repair_prompt(prompt,judgment)}],'max_output_tokens':1000,
        'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(repair_schema(),ensure_ascii=False)})
    write_json(tmp_path/'receipt.json',{'completed':True,'response_model':'reviewer','reasoning_effort':'medium','final_text':json.dumps(patch)})
    record={'base_judgment_sha256':rubric.digest(judgment),'base_prompt_sha256':rubric.digest(prompt),
        'receipt':str(tmp_path/'receipt.json'),'model':'reviewer','effort':'medium','max_output_tokens':1000,'repair':patch}
    repaired=apply_repair(judgment,prompt,record)
    assert len(repaired['findings'])==1
    assert repaired['findings'][0]['value']==data[-1].get('findings',group.get('findings'))[0]['value']
    after=evaluate((*data[:-1],repaired))
    assert after['metrics']==before['metrics']
    assert after['extraction_complete']==before['extraction_complete']


def test_preserved_findings_reject_conflicting_or_ambiguous_group_identity():
    from flowintentbench.review_repairs import preserved_findings
    with pytest.raises(ValueError,match='ambiguous'):
        preserved_findings({'result_groups':[{'group_id':'a'},{'group_id':'b'}],'findings':[{'value':1}]})
    with pytest.raises(ValueError,match='conflicting'):
        preserved_findings({'result_groups':[{'group_id':'a','findings':[{'group_id':'b','value':1}]}]})


def test_optional_eligibility_cannot_hide_a_mandatory_wrong_finding():
    claims=[SimpleNamespace(eligible=e) for e in (True,True,None,None)]
    bound=rubric._ratio_bounds([(1,1),(0,0),(0,0),(0,1)],claims)
    assert bound['lower']==.25
    assert bound['upper']==pytest.approx(2/3)
    # Whole-branch count restrictions still apply; per-item possibilities may
    # not assemble an answer out of incompatible branch choices.
    bound=rubric._count_bounds(1,1,claims,items=[(1,1),(0,0),(0,0),(0,1)])
    assert bound['upper']==.5


def test_optional_eligibility_bounds_contain_every_possible_completion():
    from itertools import product
    for eligible in product((True,None),repeat=3):
        claims=[SimpleNamespace(eligible=e) for e in eligible]
        for truth in product((True,False,None),repeat=3):
            items=[(float(v is True and e is True),float(v is not False)) for v,e in zip(truth,eligible)]
            bound=rubric._ratio_bounds(items,claims)
            completions=[]
            for presence in product(*[(True,) if e is True else (True,False) for e in eligible]):
                if not any(presence):continue
                for values in product(*[(False,True) if v is None else (v,) for v in truth]):
                    completions.append(sum(v for v,p in zip(values,presence) if p)/sum(presence))
            assert all(bound['lower']<=v<=bound['upper']+1e-12 for v in completions)
            if True in eligible:
                assert bound['upper']==pytest.approx(max(completions))
