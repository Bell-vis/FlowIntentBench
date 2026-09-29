def fixture():
    answer = "The method uses pressure. The weighting is uniform. CV is 0.5."
    names = ["feature_definition", "property_measure"]
    metadata = SimpleNamespace(principal_operationalization_dimensions=[SimpleNamespace(value=n) for n in names],
                               unresolved_operationalization_dimensions=[SimpleNamespace(value=names[1])])
    ref = ReferenceFinding.model_validate({"finding_id": "cv", "category": "quantity", "importance": "core",
        "statement": "CV", "value": 0.5, "verification": {"absolute_tolerance": 0.01}})
    gt = SimpleNamespace(acceptable_operationalizations=[SimpleNamespace(operationalization_id="o")],
        findings_by_operationalization=[SimpleNamespace(operationalization_id="o", findings=[ref])])
    material = {"finding_verification_policy": {"policies": [{"operationalization_id": "o", "finding_id": "cv",
        "verification_mode": "scalar_tolerance", "verification_parameters": ref.verification.model_dump()}]}}
    judgment = {"dimensions": [{"dimension": n, "status": "EXTRACTED", "evidence_text": "The method uses pressure.",
        "matches": {"o": True}} for n in names], "findings": [{"finding_id": "p", "statement": "CV",
        "evidence_text": "CV is 0.5.", "value": 0.5, "unit": None, "eligible": True,
        "matches": [{"branch_id": "o", "finding_id": "cv"}]}], "extraction_complete": True, "limitations": ""}
    return answer, metadata, gt, material, judgment


from copy import deepcopy


import json


from types import SimpleNamespace


import pytest


from flowintentbench import finding_scoring as rubric


from flowintentbench.reference_packages import package, fingerprint, load_supplements, apply_supplement, task_identity


from flowintentbench.ground_truth import ReferenceFinding


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
    from flowintentbench.evidence_scoring import _verify
    assert _verify(answer, finding, ref, policy)[0] is True
    assert _verify(answer, finding.model_copy(update={'unit':'m^3'}), ref, policy)[0] is None


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


def test_judge_cannot_override_wrong_numeric_reference_with_semantic_supported():
    answer, meta, gt, material, review = example()
    review["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    review["findings"][0]["matches"][0].update(semantic_verdict="SUPPORTED", reference_evidence_text="CV")
    assert evaluate((answer.replace("0.5", "0.9"), meta, gt, material, review))["metrics"]["finding_precision"]["value"] == 0


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


def test_production_v4_never_calls_scientific_execution(monkeypatch):
    from flowintentbench import numeric_diagnostics
    def forbidden(*a, **k):
        raise AssertionError("evaluation attempted to solve")
    monkeypatch.setattr(numeric_diagnostics, "_compute", forbidden)
    monkeypatch.setattr(numeric_diagnostics, "_arrays", forbidden)
    assert evaluate(example())["metrics"]["finding_precision"]["value"] == 1


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


def test_empty_supplementary_alias_does_not_reject_valid_review_or_hide_findings():
    data=example();baseline=evaluate(deepcopy(data))
    data[-1]['supplementary']=[]
    assert evaluate(data)['metrics']==baseline['metrics']
    data[-1]['supplementary']=[{'statement':'unmapped extra scientific claim'}]
    with pytest.raises(ValueError,match='supplementary'):
        evaluate(data)


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

