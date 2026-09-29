from copy import deepcopy
from statistics import stdev
from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

import pytest
from flowintentbench.paper_evaluation import (load_cases,score_answer,operationalization_scores,
    AttributeGrade,aggregate,review_schema,validate_grades,METRICS)
from flowintentbench.evidence_scoring import interval
from tests.paper_fixtures import reference_review
from scripts.evaluate import contract_for,digest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def cases():return load_cases()


def test_atomic_inventory_matches_paper(cases):
    attrs=[a for c in cases.values() for a in c[-1]['attributes']]
    assert len(cases)==96 and len(attrs)==972
    assert sum(a['responsibility']=='fixed' for a in attrs)==694
    assert sum(a['responsibility']=='open' for a in attrs)==278


@pytest.mark.parametrize('cid',list(json.loads((ROOT/'evaluation/task_cards.json').read_text())['cases']))
def test_all_cases_score_reference_fixture(cases,cid):
    answer,review=reference_review(cases[cid])
    result=score_answer(answer,review,cases[cid])
    assert result['metrics']['o_score']['value']==1
    assert set(result['metrics'])==set(METRICS)
    urs=result['metrics']['urs']
    assert urs['value']==(None if cases[cid][-1]['condition'].startswith('O1') else 1)
    # All findings in the selected reference branch are individually source-bound.
    assert result['metrics']['finding_precision']['value']==1
    assert result['metrics']['c_score']['value']==1
    assert result['metrics']['finding_recall']['value']==1


def test_mixed_dimension_urs_uses_fixed_and_open_attributes(cases):
    case=cases['radiative_vertical_advective_flux_o2_f1'];_,review=reference_review(case)
    grades=[AttributeGrade.model_validate(g) for g in review['attributes']]
    parameters=next(g for g in grades if g.attribute_id=='property_measure.parameters')
    parameters.grade=2
    metrics,_=operationalization_scores(case[-1],grades)
    assert metrics['urs']['value']==pytest.approx(5/6)
    next(g for g in grades if g.attribute_id=='property_measure.quantity').grade=None
    next(g for g in grades if g.attribute_id=='property_measure.quantity').status='UNASSESSABLE'
    metrics,_=operationalization_scores(case[-1],grades)
    assert metrics['urs']['value'] is None
    assert metrics['urs']['lower']==pytest.approx(.5)
    assert metrics['urs']['upper']==pytest.approx(5/6)


def test_unknown_outside_o2_designated_set_does_not_erase_urs(cases):
    case=cases['blunt_fin_o2_f1'];answer,r=reference_review(case)
    g=next(g for g in r['attributes'] if g['attribute_id'].split('.')[0] not in case[-1]['designated_dimensions'])
    g.update(status='UNASSESSABLE',grade=None,basis='UNCERTAIN',quotation='',source_start_line=0,source_end_line=0)
    result=score_answer(answer,r,case)
    assert result['metrics']['o_score']['value'] is None
    assert result['metrics']['urs']['value']==1


@pytest.mark.parametrize('mutation',['missing','duplicate','float','bad_quote','bad_line','open_inheritance','missing_dimension'])
def test_invalid_reviews_rejected(cases,mutation):
    case=cases['blunt_fin_o3_f1'];answer,r=reference_review(case)
    if mutation=='missing':r['attributes'].pop()
    if mutation=='duplicate':r['attributes'].append(deepcopy(r['attributes'][0]))
    if mutation=='float':r['attributes'][0]['grade']=3.5
    if mutation=='bad_quote':r['attributes'][0]['quotation']='fabricated source text'
    if mutation=='bad_line':r['attributes'][0]['source_end_line']=999
    if mutation=='open_inheritance':
        g=next(g for g in r['attributes'] if next(a for a in case[-1]['attributes'] if a['id']==g['attribute_id'])['responsibility']=='open')
        g.update(basis='QUESTION',source_start_line=0,source_end_line=0,quotation=case[0].scientific_question)
    if mutation=='missing_dimension':r['findings']['result_groups'][0]['dimensions'].pop()
    with pytest.raises(ValueError):score_answer(answer,r,case)


def test_missing_open_choice_receives_zero_not_null(cases):
    case=cases['blunt_fin_o3_f1'];answer,r=reference_review(case)
    r['attributes'][0].update(grade=0,basis='MISSING',source_start_line=0,source_end_line=0,quotation='')
    result=score_answer(answer,r,case)
    assert result['metrics']['o_score']['value'] is not None
    assert result['metrics']['o_score']==result['metrics']['urs']


def test_condition_balance_and_sample_sd():
    rows=[]
    for trial,values in [(1,[0,0,0,1]),(2,[1,1,1,0]),(3,[1,1,1,1])]:
        for i,value in enumerate(values):
            rows.append({'model_id':'M','case_id':str(i),'trial':trial,'condition':'O1-F1' if i<3 else 'O2-F1',
                         'metrics':{m:interval(value,value) for m in METRICS}})
    result=aggregate(rows)['M']['o_score']
    assert result['trials']['1']['mean']==.5
    assert result['mean']==pytest.approx(2/3)
    assert result['sample_sd']==pytest.approx(stdev([.5,.5,1]))
    with pytest.raises(ValueError):aggregate(rows+[rows[0]])


def test_unknown_and_inapplicable_are_excluded_from_point_means():
    rows=[{'model_id':'M','case_id':str(i),'trial':1,'condition':'O1-F1','metrics':{m:v for m in METRICS}}
          for i,v in enumerate([interval(.5,.5),interval(),interval(applicable=False)])]
    result=aggregate(rows)['M']['o_score']
    assert result['mean']==.5
    assert result['coverage']=={'requested':3,'applicable':2,'resolved':1}


def test_offline_cli_reproduces_validated_scores(cases,tmp_path):
    cid='blunt_fin_o1_f1';answer,review=reference_review(cases[cid])
    row={'model_id':'reference-fixture','case_id':cid,'trial':1,'answer':answer}
    path=tmp_path/'answers.jsonl';path.write_text(json.dumps(row)+'\n')
    binding=contract_for(row,cases[cid]);reviews=tmp_path/'reviews';reviews.mkdir()
    (reviews/(digest(binding)+'.json')).write_text(json.dumps({'binding':binding,'review':review}))
    output=tmp_path/'scored'
    run=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'score','--answers',str(path),
                        '--reviews',str(reviews),'--output',str(output)],cwd=tmp_path,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    report=json.loads((output/'report.json').read_text())
    assert report['api_calls']==0 and report['reviewed_answers']==1
    assert report['summary']['reference-fixture']['finding_recall']['mean']==1
    # A scientific-grade edit without rebinding input must still be validated.
    envelope=json.loads((reviews/(digest(binding)+'.json')).read_text())
    envelope['binding']['answer_sha256']='different'
    (reviews/(digest(binding)+'.json')).write_text(json.dumps(envelope))
    retry=subprocess.run(run.args[:-1]+[str(tmp_path/'tampered')],cwd=tmp_path,capture_output=True,text=True)
    assert retry.returncode!=0 and 'different answer' in retry.stderr


def test_finding_source_and_reference_ids_are_validated(cases):
    case=cases['blunt_fin_o1_f1'];answer,review=reference_review(case)
    bad=deepcopy(review);bad['findings']['findings'][0]['matches'][0]['finding_id']='invented'
    with pytest.raises(ValueError,match='unknown reference'):score_answer(answer,bad,case)
    bad=deepcopy(review);bad['findings']['findings'][0]['source_end_line']=9999
    with pytest.raises(ValueError,match='source-line'):score_answer(answer,bad,case)


def test_live_driver_retries_once_then_replays_without_calls(cases,tmp_path,monkeypatch):
    from scripts import evaluate as cli
    case=cases['blunt_fin_o1_f1'];answer,review=reference_review(case)
    row={'model_id':'test','case_id':'blunt_fin_o1_f1','trial':1,'answer':answer}
    path=tmp_path/'answers.json';path.write_text(json.dumps([row]))
    args=SimpleNamespace(command='evaluate',answers=path,output=tmp_path/'run',
        max_api_calls=2,max_prompt_chars=200000,timeout=350,max_output_tokens=10000,
        base_url='https://example.invalid/v1',reviewer_model='test-reviewer',effort='max')
    calls=[]
    def fake(prompt,args):
        calls.append(prompt)
        payload={'broken':True} if len(calls)==1 else review
        return {'status':'completed','model':'test-reviewer','output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(payload)}]}]}
    monkeypatch.setattr(cli,'api_review',fake)
    assert cli.run(args)==0 and len(calls)==2
    assert 'Correct the structural/transport error' in calls[1]
    assert cli.run(args)==0 and len(calls)==2
    assert json.loads((args.output/'report.json').read_text())['api_calls']==0


def test_two_failed_reviews_produce_pending_not_default_scores(cases,tmp_path,monkeypatch):
    from scripts import evaluate as cli
    answer,_=reference_review(cases['blunt_fin_o1_f1'])
    path=tmp_path/'answers.json';path.write_text(json.dumps([{'model_id':'test','case_id':'blunt_fin_o1_f1','trial':1,'answer':answer}]))
    args=SimpleNamespace(command='evaluate',answers=path,output=tmp_path/'run',
        max_api_calls=2,max_prompt_chars=200000,timeout=350,max_output_tokens=10000,
        base_url='https://example.invalid/v1',reviewer_model='test-reviewer',effort='max')
    monkeypatch.setattr(cli,'api_review',lambda *_:{'status':'incomplete'})
    assert cli.run(args)==2
    report=json.loads((args.output/'report.json').read_text())
    assert report['api_calls']==2 and report['reviewed_answers']==0
    assert report['summary']['test']['o_score']['mean'] is None
    assert len(list((args.output/'requests').glob('*/*/receipt.json')))==2
