from copy import deepcopy
import json

import pytest

from flowintentbench.answer_evidence import bind_value
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.review_repairs import apply_repair, finding_citation_targets, repair_prompt, repair_schema
from flowintentbench.rubric_scoring import digest


def setup(tmp_path, citations):
    judgment={'result_groups':[{'group_id':'primary','purpose':'PRIMARY','dimensions':[],
        'findings':[{'finding_id':'area','statement':'total surface area','value':9,
            'unit':'stored coordinate-area units','source_start_line':4,'source_end_line':4,
            'unit_evidence_text':'L4: in stored coordinate-area units.',
            'eligible':True,'matches':[{'branch_id':'B0','finding_id':'R0'}]}]}],
        'findings':[{'finding_id':'count','group_id':'primary','statement':'number of regions',
            'value':2,'unit':None,'source_start_line':1,'source_end_line':1},
            {'finding_id':'center','group_id':'primary','statement':'centroid coordinates',
             'value':[3,4],'unit':None,'eligible':True,'matches':[]}],
        'extraction_complete':True}
    prompt='Original\n\n'+json.dumps({'question':'Report region geometry',
        'answer':'L1: Two regions.\nL2: \nL3: Centroid = (3, 4).\nL4: Total area = 9.\nL5: in stored coordinate-area units.',
        'reference_package':{'private_truth':999}})
    mode='FINDING_CITATIONS';patch={'citations':citations}
    write_json(tmp_path/'request.json',{'input':[{'content':repair_prompt(prompt,judgment,mode)}],
        'instructions':'Return only JSON matching the supplied schema. No tools.\n'+json.dumps(repair_schema(mode)),
        'max_output_tokens':500})
    write_json(tmp_path/'receipt.json',{'completed':True,'response_model':'judge','reasoning_effort':'medium',
        'final_text':json.dumps(patch)})
    record={'base_judgment_sha256':digest(judgment),'base_prompt_sha256':digest(prompt),
        'receipt':str(tmp_path/'receipt.json'),'model':'judge','effort':'medium','max_output_tokens':500,
        'mode':mode,'repair':patch}
    return judgment,prompt,record


def citation(key,line,status='BOUND'):
    return {'target_id':key,'status':status,'source_start_line':line,'source_end_line':line}


def test_finding_and_unit_citations_preserve_all_claims_and_decisions(tmp_path):
    j,p,r=setup(tmp_path,[citation('T0',3),citation('T1',5)])
    targets=finding_citation_targets(j,p)
    assert [(x['kind'],x['statement']) for x in targets.values()]==[
        ('SOURCE','centroid coordinates'),('UNIT','total surface area')]
    packet=json.loads(repair_prompt(p,j,'FINDING_CITATIONS').rsplit('\n\n',1)[1])
    assert 'private_truth' not in repair_prompt(p,j,'FINDING_CITATIONS')
    assert all('path' not in x for x in packet['targets'])
    got=apply_repair(j,p,r);expected=deepcopy(j)
    expected['findings'][1].update(evidence_text='Centroid = (3, 4).',source_start_line=3,source_end_line=3)
    expected['result_groups'][0]['findings'][0]['unit_evidence_text']='in stored coordinate-area units.'
    assert got==expected
    assert finding_citation_targets(got,p)=={}


def test_unverified_finding_citation_does_not_change_values_or_award_correctness(tmp_path):
    j,p,r=setup(tmp_path,[citation('T0',1),citation('T1',None,'NOT_FOUND')])
    got=apply_repair(j,p,r)
    claim=got['findings'][1]
    assert claim['value']==[3,4] and claim['matches']==[]
    assert bind_value('Two regions.',claim['evidence_text'],claim['value'])['reason']=='VALUE_NOT_IN_SOURCE'
    assert got['result_groups']==j['result_groups']


@pytest.mark.parametrize('rows',[
    [citation('T0',3)],
    [citation('T0',3),citation('T0',5)],
    [citation('T0',3),citation('bad',5)],
    [citation('T0',2),citation('T1',5)],
    [citation('T0',9),citation('T1',5)],
    [citation('T0',3),citation('T1',5,'UNCERTAIN')],
])
def test_finding_citation_patch_rejects_ambiguous_or_invalid_evidence(tmp_path,rows):
    j,p,r=setup(tmp_path,rows)
    with pytest.raises(ValueError):apply_repair(j,p,r)
