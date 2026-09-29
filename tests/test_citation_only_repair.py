from copy import deepcopy
import json

import pytest
from jsonschema import ValidationError

from flowintentbench.external_file_evaluator import write_json
from flowintentbench.review_repairs import apply_repair, repair_prompt, repair_schema
from flowintentbench.rubric_scoring import digest


def setup(tmp_path, patch):
    judgment = {'result_groups':[{'group_id':'main','purpose':'PRIMARY','dimensions':[
        {'dimension':'criterion','status':'EXTRACTED','matches':{'B0':False},
         'adequacy':'MET','reason':'VALID_ALTERNATIVE','rationale':'Use upper ten percent'},
        {'dimension':'property_measure','status':'EXTRACTED','matches':{'B0':True},
         'evidence_text':'Known original citation','adequacy':'MET'}],
         'findings':[{'value':99,'eligible':None}]}], 'findings':[]}
    prompt = 'Original request\n\n'+json.dumps({'question':'Describe the region',
        'answer':'L1: I selected the upper ten percent.\nL2: \nL3: Result is 99.',
        'reference_package':{'secret_reference_value':0}})
    mode = 'CITATIONS_ONLY'
    request = {'input':[{'content':repair_prompt(prompt,judgment,mode)}],
        'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(repair_schema(mode)),
        'max_output_tokens':500}
    write_json(tmp_path/'request.json',request)
    write_json(tmp_path/'receipt.json',{'completed':True,'response_model':'judge',
        'reasoning_effort':'medium','final_text':json.dumps(patch)})
    record = {'base_judgment_sha256':digest(judgment),'base_prompt_sha256':digest(prompt),
        'receipt':str(tmp_path/'receipt.json'),'model':'judge','effort':'medium',
        'max_output_tokens':500,'mode':mode,'repair':patch}
    return judgment,prompt,record


def citation(status='BOUND', start=1, end=1):
    return {'target_id':'T0','status':status,'source_start_line':start,'source_end_line':end}


def test_citation_only_preserves_every_semantic_decision_and_nested_finding(tmp_path):
    j,p,r=setup(tmp_path,{'citations':[citation()]})
    got=apply_repair(j,p,r)
    expected=deepcopy(j)
    expected['result_groups'][0]['dimensions'][0].update(evidence_text='I selected the upper ten percent.',
        source_start_line=1,source_end_line=1)
    assert got==expected
    assert 'secret_reference_value' not in repair_prompt(p,j,'CITATIONS_ONLY')
    assert len(json.loads(repair_prompt(p,j,'CITATIONS_ONLY').rsplit('\n\n',1)[1])['targets'])==1


def test_structured_citation_receipt_binds_wire_schema_and_preserves_decisions(tmp_path):
    j, p, r = setup(tmp_path, {'citations': [citation()]})
    expected = apply_repair(j, p, r)
    r.update(mode='CITATIONS_ONLY_V2', structured_output=True)
    request = json.loads((tmp_path / 'request.json').read_text())
    request['input'][0]['content'] = repair_prompt(p, j, r['mode'])
    request['instructions'] = 'Return only JSON matching the supplied response schema. No tools. Use the required line citations; do not add commentary.'
    request['text'] = {'format': {'type': 'json_schema', 'strict': True,
        'schema': repair_schema(r['mode'])}}
    write_json(tmp_path / 'request.json', request)
    assert apply_repair(j, p, r) == expected
    for changed in ('strict', 'schema', 'record_flag'):
        wire, record = deepcopy(request), deepcopy(r)
        if changed == 'strict':
            wire['text']['format']['strict'] = False
        elif changed == 'schema':
            wire['text']['format']['schema']['additionalProperties'] = True
        else:
            record.pop('structured_output')
        write_json(tmp_path / 'request.json', wire)
        with pytest.raises(ValueError, match='schema'):
            apply_repair(j, p, record)


@pytest.mark.parametrize('status',['NOT_FOUND','UNCERTAIN'])
def test_unresolved_citation_remains_unchanged(tmp_path,status):
    j,p,r=setup(tmp_path,{'citations':[citation(status,None,None)]})
    assert apply_repair(j,p,r)==j


@pytest.mark.parametrize('rows',[
    [],[citation(),citation()], [dict(citation(),target_id='invented')],
    [citation(start=1,end=4)], [citation(start=2,end=2)],
    [citation('UNCERTAIN',1,1)], [dict(citation(),matches={'B0':True})],
])
def test_invalid_or_expanded_citation_patch_is_rejected(tmp_path,rows):
    j,p,r=setup(tmp_path,{'citations':rows})
    with pytest.raises((ValueError,ValidationError)):apply_repair(j,p,r)


def test_already_cited_review_is_not_a_citation_job(tmp_path):
    from flowintentbench.review_repairs import citation_targets
    j,p,_=setup(tmp_path,{'citations':[citation()]})
    j['result_groups'][0]['dimensions'][0]['source_start_line']=1
    assert citation_targets(j)=={}
    with pytest.raises(ValueError,match='uncited explicit'):
        repair_prompt(p,j,'CITATIONS_ONLY')


@pytest.mark.parametrize('description', [None, '', '  '])
def test_citation_preflight_rejects_empty_descriptions_before_api(tmp_path, description):
    from scripts.repair_evaluation_reviews import citation_preflight
    j,p,_=setup(tmp_path,{'citations':[citation()]})
    dimension=j['result_groups'][0]['dimensions'][0]
    assert citation_preflight(j) is None
    dimension['choice']=dimension.pop('rationale')
    dimension['rationale']=description
    assert 'GROUPS' in citation_preflight(j, 'CITATIONS_ONLY')
    # Old receipts still validate against the historical prompt, rather than
    # silently substituting the newly discovered `choice` field.
    packet=json.loads(repair_prompt(p,j,'CITATIONS_ONLY').rsplit('\n\n',1)[1])
    assert packet['targets'][0]['description_to_locate']==description
    assert citation_preflight(j) is None
    packet_v2=json.loads(repair_prompt(p,j,'CITATIONS_ONLY_V2').rsplit('\n\n',1)[1])
    assert packet_v2['protocol']=='method-citation-repair-v2'
    assert packet_v2['targets'][0]['description_to_locate']==dimension['choice']
    dimension.pop('choice')
    assert 'GROUPS' in citation_preflight(j)


def test_v2_choice_citation_binds_without_changing_legacy_decisions(tmp_path):
    j,p,r=setup(tmp_path,{'citations':[citation()]})
    dimension=j['result_groups'][0]['dimensions'][0]
    dimension['choice']=dimension.pop('rationale')
    r.update(base_judgment_sha256=digest(j),mode='CITATIONS_ONLY_V2')
    request=json.loads((tmp_path/'request.json').read_text())
    request['input'][0]['content']=repair_prompt(p,j,r['mode'])
    write_json(tmp_path/'request.json',request)
    got=apply_repair(j,p,r)
    expected=deepcopy(j)
    expected['result_groups'][0]['dimensions'][0].update(evidence_text='I selected the upper ten percent.',
        source_start_line=1,source_end_line=1)
    assert got==expected
