from copy import deepcopy
import json

import pytest

from flowintentbench.reference_extension_review import additions,request,apply
from scripts.extend_evaluation_references import source_answer


def fixture():
    answer='Mean density is 2.0. All cells use volume weights.'
    claim={'finding_id':'f','group_id':'g','statement':'mean density','value':2.0,'unit':None,
           'evidence_text':answer,'eligible':True,'matches':[], 'mapping_complete':True}
    judgment={'result_groups':[{'group_id':'g','purpose':'PRIMARY','dimensions':[], 'evidence_text':answer}],
              'findings':[claim]}
    old={'protocol':'p','dimensions':[],'finding_goal':'goal','adequate_role_contract':{},
         'branches':[{'branch_id':'b','decisions':[], 'findings':[{'finding_id':'core','importance':'core','value':1,
            'host_policy':{'tolerance':.01}}]}]}
    new=deepcopy(old)
    new['branches'][0]['findings'].append({'finding_id':'mean','importance':'supporting','statement':'Volume-weighted mean density over all cells',
                                          'value':987654.321,'unit':None})
    return answer,judgment,old,new


@pytest.mark.parametrize('change',[
    lambda p:p['branches'][0]['findings'][0].update(value=2),
    lambda p:p['branches'][0]['findings'][0]['host_policy'].update(tolerance=1),
    lambda p:p['branches'][0].update(decisions=['new method']),
    lambda p:p['branches'][0].update(input_population={'publicly_specified':False}),
    lambda p:p['adequate_role_contract'].update(mandatory_roles=['new role']),
    lambda p:p['branches'][0]['findings'][-1].update(importance='core'),
    lambda p:p['branches'][0]['findings'].pop(0),
    lambda p:p.update(host_method_dependencies={'b':{'core':['feature_definition']}}),
    lambda p:p['branches'].append(deepcopy(p['branches'][0])),
])
def test_extensions_cannot_change_existing_scientific_obligations(change):
    _,_,old,new=fixture();change(new)
    with pytest.raises(ValueError):additions(old,new)


def test_matching_prompt_blinds_reference_numbers_and_preserves_source():
    answer,j,old,new=fixture();prompt,schema,aliases,binding=request(answer,j,old,new,{'f'})
    assert '987654.321' not in prompt
    assert answer in prompt
    assert aliases=={'N0':('b','mean')}
    assert 'model_id' not in prompt
    assert binding['old_reference_sha256']!=binding['new_reference_sha256']
    assert schema['properties']['mappings']['items']['properties']['reference_ids']['items']['enum']==['N0']


def test_compact_mapping_request_preserves_full_answer_quantities_and_schema():
    answer,j,old,new=fixture()
    prompt,schema,aliases,binding=request(answer,j,old,new,{'f'})
    compact,cs,ca,cb=request(answer,j,old,new,{'f'},compact_findings=True)
    p=json.loads(prompt.rsplit('\n\n',1)[1]);c=json.loads(compact.rsplit('\n\n',1)[1])
    assert c['answer']==p['answer']
    assert c['new_references']==p['new_references']
    assert c['findings']==[{k:v for k,v in f.items() if k!='evidence_text'} for f in p['findings']]
    assert (cs,ca)==(schema,aliases)
    assert cb.pop('compact_findings') is True
    assert cb==binding
    assert len(compact)<len(prompt)


def test_extensions_add_semantic_edges_without_modifying_claims_or_methods():
    answer,j,old,new=fixture();_,_,aliases,_=request(answer,j,old,new,{'f'})
    payload={'mappings':[{'finding_id':'f','status':'MATCHED','reference_ids':['N0'],'source_start_line':1,'source_end_line':1}]}
    result,audit=apply(answer,j,aliases,payload,{'f'})
    assert result['findings'][0]['matches']==[{'branch_id':'b','finding_id':'mean'}]
    result['findings'][0]['matches']=[]
    assert result==j
    assert j['findings'][0]['matches']==[]
    assert audit==[{**payload['mappings'][0],'applied':True,'reason':'BOUND_MAPPING'}]


@pytest.mark.parametrize('change',[
    lambda p:p['mappings'].append(deepcopy(p['mappings'][0])),
    lambda p:p['mappings'][0].update(reference_ids=['core']),
    lambda p:p['mappings'][0].update(status='NO_MATCH'),
    lambda p:p['mappings'][0].update(reference_ids=['N0','N0']),
    lambda p:p['mappings'][0].update(finding_id='invented'),
])
def test_bad_mapping_receipts_are_rejected(change):
    answer,j,old,new=fixture();_,_,aliases,_=request(answer,j,old,new,{'f'})
    payload={'mappings':[{'finding_id':'f','status':'MATCHED','reference_ids':['N0'],'source_start_line':1,'source_end_line':1}]}
    change(payload)
    with pytest.raises(ValueError):apply(answer,j,aliases,payload,{'f'})


def test_no_match_does_not_turn_unknown_claim_into_wrong_or_ineligible():
    answer,j,old,new=fixture();_,_,aliases,_=request(answer,j,old,new,{'f'})
    payload={'mappings':[{'finding_id':'f','status':'NO_MATCH','reference_ids':[],'source_start_line':None,'source_end_line':None}]}
    result,_=apply(answer,j,aliases,payload,{'f'})
    assert result==j


def test_partition_count_mismatch_is_reviewer_uncertainty_not_a_model_error():
    answer,j,old,new=fixture()
    payload={'mappings':[{'finding_id':'f','status':'MATCHED','reference_ids':['N0'],
                         'source_start_line':1,'source_end_line':1,'partition_bin_count':5}]}
    result,audit=apply(answer,j,{'N0':('b','mean')},payload,{'f'},
                       partition_scopes={'N0':{'bin_count':10}})
    assert result==j and audit[0]['reason']=='PARTITION_BIN_COUNT_MISMATCH'
    assert not audit[0]['applied']
    result,audit=apply(answer,j,{'N0':('b','mean')},payload,{'f'},
                       partition_scopes={'N0':{'bin_count':5}})
    assert audit[0]['applied']


def test_partition_prompt_requires_independent_count_and_keeps_values_hidden():
    from flowintentbench.reference_extension_review import partition_scope
    answer,j,old,new=fixture()
    statement='Volume-weighted mean; bin 1 of 5 in increasing order; 5 equal-width density bins spanning its complete cell minimum to maximum'
    new['branches'][0]['findings'][-1]['statement']=statement
    p,s,_,b=request(answer,j,old,new,{'f'},partition_aware=True)
    assert partition_scope(statement)=={'scheme':'equal_width','bin_count':5,'selected_bin':1}
    assert '987654.321' not in p
    assert 'partition_bin_count' in s['properties']['mappings']['items']['required']
    assert b['partition_aware']


def test_one_unbound_mapping_does_not_block_other_findings():
    answer,j,old,new=fixture();_,_,aliases,_=request(answer,j,old,new,{'f'})
    j['findings'].append({**deepcopy(j['findings'][0]),'finding_id':'other'})
    good={'finding_id':'f','status':'MATCHED','reference_ids':['N0'],'source_start_line':1,'source_end_line':1}
    bad={**good,'finding_id':'other','source_start_line':2,'source_end_line':2}
    result,audit=apply(answer,j,aliases,{'mappings':[good,bad]},{'f','other'})
    assert result['findings'][0]['matches']==[{'branch_id':'b','finding_id':'mean'}]
    assert result['findings'][1]['matches']==[]
    assert audit[1]['applied'] is False
    assert audit[1]['reason']=='MAPPING_SOURCE_LINES_UNBOUND'


def test_answer_recovery_requires_exact_hash_including_final_newline():
    import hashlib
    prompt='Instructions\n\n'+json.dumps({'answer':'L1: hello\nL2: world'})
    for answer in ('hello\nworld','hello\nworld\n','hello\r\nworld\r\n'):
        assert source_answer(prompt,hashlib.sha256(answer.encode()).hexdigest())==answer
    with pytest.raises(ValueError):source_answer(prompt,hashlib.sha256(b'other').hexdigest())


def test_original_export_restores_control_characters_and_rejects_source_drift(tmp_path):
    import hashlib
    from flowintentbench.answer_collections import collect
    from scripts.extend_evaluation_references import original_answer
    from scripts.evaluate_model_answers import sha
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'cases':[{'case_id':'c','condition':'O1-F1'}]}))
    export=tmp_path/'answer.json'
    answer='Text with a TeX form feed: \frac{1}{2}\n'
    export.write_text(json.dumps({'case_id':'c','model_id':'m','trial':1,'run_id':'r','answer':answer}))
    slots,_,sources=collect(None,[export],manifest)
    row={'case_id':'c','model_id':'m','trial':1,'run_id':'r','output_id':slots[0]['output_id'],
         'answer_sha256':hashlib.sha256(answer.encode()).hexdigest()}
    contract={'source_identity':{'inputs':sources}}
    assert original_answer(row,contract,manifest)==answer
    export.write_text('{}')
    with pytest.raises(ValueError,match='collection changed'):original_answer(row,contract,manifest)


def test_extension_replay_keeps_one_authoritative_receipt_and_checks_external_hash(tmp_path):
    from scripts.extend_evaluation_references import replay_receipt
    from scripts.evaluate_model_answers import sha
    original=tmp_path/'original';original.mkdir()
    receipt=original/'receipt.json';receipt.write_text('{"completed":true}')
    replay=tmp_path/'replay';replay.mkdir()
    (replay/'extension_report.json').write_text(json.dumps({'rows':[{'request_id':'i',
        'receipt':str(receipt),'receipt_sha256':sha(receipt)}]}))
    assert replay_receipt(replay,'i')==receipt
    assert replay_receipt(replay,'other') is None
    assert not (replay/'api').exists()
    receipt.write_text('{}')
    with pytest.raises(ValueError,match='lineage changed'):replay_receipt(replay,'i')


def test_retry_preserves_failed_receipt_and_replay_prefers_bound_success(tmp_path):
    from scripts.extend_evaluation_references import replay_receipt,retry_receipt_path
    from scripts.evaluate_model_answers import sha
    original=tmp_path/'api/i/receipt.json';original.parent.mkdir(parents=True)
    failure={'completed':False,'finished_epoch':1,'error':'provider overload'}
    original.write_text(json.dumps(failure));old_hash=sha(original)
    retry=retry_receipt_path(tmp_path,'i',failure)
    retry.parent.mkdir(parents=True);retry.write_text('{"completed":true}')
    (tmp_path/'extension_report.json').write_text(json.dumps({'rows':[{'request_id':'i',
        'receipt':str(retry),'receipt_sha256':sha(retry)}]}))
    assert replay_receipt(tmp_path,'i')==retry
    assert sha(original)==old_hash
    assert retry_receipt_path(tmp_path,'i',failure).parent.name=='attempt-2'


@pytest.mark.parametrize('receipt',[{'completed':True,'finished_epoch':1},{'completed':False}])
def test_retry_rejects_successful_or_still_running_attempt(tmp_path,receipt):
    from scripts.extend_evaluation_references import retry_receipt_path
    with pytest.raises(ValueError,match='finished incomplete'):
        retry_receipt_path(tmp_path,'i',receipt)


def test_number_blind_prompt_replay_allows_only_hidden_new_reference_change(tmp_path):
    from scripts.extend_evaluation_references import identical_prompt_receipt
    from flowintentbench.rubric_scoring import digest
    answer,j,old,new=fixture();p,s,_,b=request(answer,j,old,new,{'f'})
    b.update(model='judge',effort='medium',max_output_tokens=2000,prompt_sha256=digest(p),schema_sha256=digest(s))
    rid=digest(b);rp=tmp_path/'requests'/(rid+'.json');rp.parent.mkdir()
    rp.write_text(json.dumps({'prompt':p,'schema':s,'binding':b}))
    receipt=tmp_path/'api'/rid/'receipt.json';receipt.parent.mkdir(parents=True)
    receipt.write_text('{"completed":true}')
    changed={**b,'new_reference_sha256':'new independently frozen numerical policy'}
    path,lineage=identical_prompt_receipt(tmp_path,p,s,changed)
    assert path==receipt and lineage['request_id']==rid
    assert identical_prompt_receipt(tmp_path,p+'changed scope',s,changed) is None
    assert identical_prompt_receipt(tmp_path,p,{**s,'description':'changed'},changed) is None
    for key in ['judgment_sha256','old_reference_sha256','answer_sha256','model','max_output_tokens']:
        assert identical_prompt_receipt(tmp_path,p,s,{**changed,key:'different'}) is None


@pytest.mark.parametrize('changed_content', [False, True])
def test_reference_repackaging_reuses_only_identical_case_contents(tmp_path, changed_content):
    from copy import deepcopy
    from scripts.extend_evaluation_references import provenance_only_reference_alias
    from flowintentbench.rubric_scoring import digest
    from flowintentbench.external_file_evaluator import write_json
    base = tmp_path/'base'
    original = {'package_sha256': 'old', 'branches': [{'value': .5, 'tolerance': .01}],
        'supplement_sources': [{'authority': 'frozen', 'source': {'path': 'old.json',
            'sha256': 'old-file-hash', 'pointer': '/case'}}]}
    write_json(base/'reference_packages'/(digest('case')+'.json'), original)
    write_json(tmp_path/'reports'/'experiment_report.json',
        {'source_identity': {'base_evaluation': str(base)}})
    current = deepcopy(original);current['package_sha256']='new'
    current['supplement_sources'][0]['source'].update(path='new.json',sha256='new-file-hash')
    if changed_content:current['branches'][0]['tolerance']=.02
    assert provenance_only_reference_alias(tmp_path,'case',current) == (None if changed_content else digest(original))


def test_provenance_alias_does_not_bypass_prompt_or_judgment_checks(tmp_path):
    from scripts.extend_evaluation_references import identical_prompt_receipt
    from flowintentbench.rubric_scoring import digest
    from flowintentbench.external_file_evaluator import write_json
    answer,j,old,new=fixture();p,s,_,b=request(answer,j,old,new,{'f'})
    b.update(model='judge',effort='medium',max_output_tokens=2000,prompt_sha256=digest(p),schema_sha256=digest(s))
    rid=digest(b)
    write_json(tmp_path/'requests'/(rid+'.json'),{'prompt':p,'schema':s,'binding':b})
    write_json(tmp_path/'api'/rid/'receipt.json',{'completed':True})
    changed={**b,'old_reference_sha256':'repacked'}
    assert identical_prompt_receipt(tmp_path,p,s,changed) is None
    assert identical_prompt_receipt(tmp_path,p,s,changed,old_reference_alias=b['old_reference_sha256'])
    assert identical_prompt_receipt(tmp_path,p+'x',s,changed,old_reference_alias=b['old_reference_sha256']) is None
    changed['judgment_sha256']='different'
    assert identical_prompt_receipt(tmp_path,p,s,changed,old_reference_alias=b['old_reference_sha256']) is None


@pytest.mark.parametrize('claimed,expected_precision',[(5.,1.),(9.,.5)])
def test_added_reference_exposes_wrong_extra_without_changing_required_recall(claimed,expected_precision):
    from test_rubric_evaluation import example,evaluate
    from flowintentbench.ground_truth import ReferenceFinding
    answer,meta,gt,material,judgment=example()
    quote=f'Mean pressure is {claimed}.'
    answer+='\n'+quote
    base=deepcopy(judgment['findings'][0])
    base.update(finding_id='extra',statement='mean pressure',value=claimed,evidence_text=quote,matches=[])
    judgment['findings'].append(base)
    ref=gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings.append(ReferenceFinding.model_validate(
        ref.model_copy(update={'finding_id':'mean','importance':'supporting','statement':'Mean pressure', 'value':5.}).model_dump()))
    material['finding_verification_policy']['policies'].append({
        **material['finding_verification_policy']['policies'][0],'finding_id':'mean'})
    before=evaluate((answer,meta,gt,material,judgment))
    assert before['metrics']['finding_precision']['value'] is None
    mapping={'mappings':[{'finding_id':'extra','status':'MATCHED','reference_ids':['N0'],
                         'source_start_line':2,'source_end_line':2}]}
    extended,_=apply(answer,judgment,{'N0':('o','mean')},mapping,{'extra'})
    after=evaluate((answer,meta,gt,material,extended))
    assert after['metrics']['finding_precision']['value']==expected_precision
    assert after['metrics']['c_score']['value']==expected_precision
    for metric in ('o_score','finding_requirement_recall'):
        assert before['metrics'][metric]==after['metrics'][metric]


def test_prior_mapping_replays_evidence_and_rejects_changed_source(tmp_path):
    from scripts.extend_evaluation_references import carry_prior_mapping
    from scripts.evaluate_model_answers import sha
    from flowintentbench.rubric_scoring import digest
    answer,j,old,new=fixture()
    p,s,_,b=request(answer,j,old,new,{'f'},compact_findings=True)
    reviewer={'model':'judge','effort':'medium','max_output_tokens':2000}
    b.update(**reviewer,prompt_sha256=digest(p),schema_sha256=digest(s))
    rid=digest(b);rp=tmp_path/'requests'/(rid+'.json');rp.parent.mkdir()
    rp.write_text(json.dumps({'prompt':p,'schema':s,'binding':b}))
    receipt=tmp_path/'api'/rid/'receipt.json';receipt.parent.mkdir(parents=True)
    payload={'mappings':[{'finding_id':'f','status':'MATCHED','reference_ids':['N0'],
                         'source_start_line':1,'source_end_line':1}]}
    receipt.write_text(json.dumps({'completed':True,'response_model':'judge','reasoning_effort':'medium',
                                   'final_text':json.dumps(payload)}))
    api_request={'input':[{'role':'user','content':p}],
        'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(s,ensure_ascii=False),
        'max_output_tokens':2000}
    receipt.with_name('request.json').write_text(json.dumps(api_request))
    row={'request_id':rid,'receipt':str(receipt),'receipt_sha256':sha(receipt),
         'after_metrics':{'invented_score':999}}
    result,proof=carry_prior_mapping(tmp_path,row,answer,j,old,new,reviewer)
    assert result['findings'][0]['matches']==[{'branch_id':'b','finding_id':'mean'}]
    assert proof['receipt_sha256']==sha(receipt)
    assert 'invented_score' not in str(result)
    with pytest.raises(ValueError,match='bind current source'):
        carry_prior_mapping(tmp_path,row,answer+' changed',j,old,new,reviewer)
    altered=deepcopy(new);altered['branches'][0]['findings'][-1]['value']=2
    with pytest.raises(ValueError,match='bind current source'):
        carry_prior_mapping(tmp_path,row,answer,j,old,altered,reviewer)
    api_request['max_output_tokens']=999
    receipt.with_name('request.json').write_text(json.dumps(api_request))
    with pytest.raises(ValueError,match='does not bind'):
        carry_prior_mapping(tmp_path,row,answer,j,old,new,reviewer)
