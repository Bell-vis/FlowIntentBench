#!/usr/bin/env python3
"""Prepare, review, and reproduce FlowIntentBench paper-protocol scores."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.paper_evaluation import (
    PROTOCOL, METRICS, load_cases, review_prompt, review_schema, score_answer, aggregate,
)
from flowintentbench.evidence_scoring import interval


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def answers(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get('answers', [value])
    if not isinstance(value, list) or not value:
        raise ValueError('answers must contain a nonempty JSON array or JSONL records')
    seen = set()
    for row in value:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('model_id','case_id','answer')):
            raise ValueError('model_id, case_id, and answer must be nonempty strings')
        if type(row.get('trial')) is not int or row['trial'] < 1:
            raise ValueError('trial must be a positive integer')
        key = row['model_id'], row['case_id'], row['trial']
        if key in seen:
            raise ValueError('duplicate model/case/trial answer')
        seen.add(key)
    return value


def contract_for(row, case):
    ci, meta, gt, material, card = case
    return {'protocol': PROTOCOL, 'model_id': row['model_id'], 'case_id': row['case_id'], 'trial': row['trial'],
            'answer_sha256': hashlib.sha256(row['answer'].encode()).hexdigest(),
            'reference_sha256': digest([ci.model_dump(mode='json'), meta.model_dump(mode='json'), gt.model_dump(mode='json'), material, card])}


def parse_response(response):
    if response.get('status') != 'completed':
        raise ValueError('reviewer response did not complete')
    if any(item.get('type') not in {'message','reasoning'} for item in response.get('output', [])):
        raise ValueError('unexpected reviewer tool output')
    text = '\n'.join(part['text'] for item in response.get('output',[]) if item.get('type')=='message'
                     for part in item.get('content',[]) if part.get('type')=='output_text')
    if text.strip().startswith('```'):
        text = '\n'.join(text.strip().splitlines()[1:-1])
    return json.loads(text)


def api_review(prompt, args):
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        raise ValueError('set OPENAI_API_KEY before live evaluation')
    payload = {'model': args.reviewer_model, 'reasoning': {'effort': args.effort},
               'max_output_tokens': args.max_output_tokens, 'store': False,
               'instructions': 'Assess the answer without tools. Return JSON matching the supplied schema.',
               'input': prompt}
    request = urllib.request.Request(args.base_url.rstrip('/')+'/responses',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer '+key, 'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        # Do not print server response bodies containing credentials or prompts.
        raise RuntimeError(f'reviewer HTTP {exc.code}') from None
    if value.get('model') != args.reviewer_model:
        raise ValueError('reviewer response model differs from requested model')
    return value


def run(args):
    cases = load_cases()
    if args.command == 'prepare':
        selected = args.cases or list(cases)
        for cid in selected:
            if cid not in cases:
                raise ValueError(f'unknown case: {cid}')
        inventory = {'protocol':PROTOCOL, 'case_count':len(selected),
                     'attribute_count':sum(len(cases[c][-1]['attributes']) for c in selected),
                     'by_condition': {condition:sum(cases[c][-1]['condition']==condition for c in selected)
                                      for condition in ('O1-F1','O2-F1','O3-F1','O1-F2')}}
        write(args.output/'inventory.json', inventory)
        write(args.output/'review_schema.json', review_schema())
        print(json.dumps(inventory)); return 0
    rows = answers(args.answers)
    if any(r['case_id'] not in cases for r in rows):
        raise ValueError('answer references an unknown case')
    if args.command == 'evaluate' and (args.max_api_calls<0 or args.max_prompt_chars<1 or args.timeout<=0 or args.max_output_tokens<1):
        raise ValueError('invalid review budget')
    bindings = [contract_for(row,cases[row['case_id']]) for row in rows]
    code_files = sorted((ROOT/'flowintentbench').glob('*.py')) + [Path(__file__).resolve()]
    contract = {'protocol':PROTOCOL, 'answers':bindings,
                'implementation':digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files})}
    if args.command == 'evaluate':
        contract['reviewer'] = {key:getattr(args,key) for key in ('base_url','reviewer_model','effort','max_output_tokens','max_prompt_chars','timeout')}
    contract_path = args.output/'evaluation_contract.json'
    if contract_path.exists() and read(contract_path)!=contract:
        raise ValueError('output contains a different evaluation contract; select a new directory')
    write(contract_path, contract)
    scores, failed, calls = [], [], 0
    for row,binding in zip(rows,bindings):
        case = cases[row['case_id']]
        rid = digest(binding)
        envelope_path = (args.reviews if args.command=='score' else args.output/'reviews')/(rid+'.json')
        value, error = None, None
        if envelope_path.exists():
            envelope = read(envelope_path)
            if envelope.get('binding')!=binding:
                raise ValueError('review is bound to a different answer or reference')
            value = score_answer(row['answer'],envelope['review'],case)
        elif args.command == 'evaluate':
            original_prompt = review_prompt(case,row['answer'])
            prompt = original_prompt
            for attempt in range(2):
                if calls >= args.max_api_calls:
                    error = 'REVIEW_BUDGET_EXHAUSTED'; break
                if len(prompt)>args.max_prompt_chars:
                    error = 'PROMPT_LENGTH_LIMIT'; break
                # Every attempt has its own immutable request/response record.
                attempt_number = len(list((args.output/'requests'/rid).glob('*/request.json')))+1
                folder = args.output/'requests'/rid/str(attempt_number)
                write(folder/'request.json', {'binding':binding,'reviewer':contract['reviewer'],'prompt':prompt})
                started = time.monotonic()
                calls += 1
                try:
                    response = api_review(prompt,args)
                    write(folder/'response.json',response)
                    parsed = parse_response(response)
                    value = score_answer(row['answer'],parsed,case)
                    write(envelope_path,{'binding':binding,'review':parsed})
                    write(folder/'receipt.json',{'status':'ACCEPTED','elapsed_seconds':time.monotonic()-started,'usage':response.get('usage')})
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    write(folder/'receipt.json',{'status':'ERROR','error':error,'elapsed_seconds':time.monotonic()-started})
                    if any(x in error for x in ('HTTP 401','HTTP 403','HTTP 402','HTTP 404')):
                        raise RuntimeError(error) from None
                    prompt = original_prompt+'\nCorrect the structural/transport error and return the full review, preserving scientific judgments: '+error[:700]
        if value is None:
            failed.append({'model_id':row['model_id'],'case_id':row['case_id'],'trial':row['trial'],'reason':error or 'REVIEW_MISSING'})
            value = {'protocol':PROTOCOL,'case_id':row['case_id'],'condition':case[-1]['condition'],
                     'family_id':case[-1]['family_id'],'metrics':{m:interval(applicable=m!='urs' or not case[-1]['condition'].startswith('O1')) for m in METRICS}}
        value.update(model_id=row['model_id'],trial=row['trial'])
        scores.append(value)
        write(args.output/'scores'/(rid+'.json'),value)
    report={'protocol':PROTOCOL,'requested_answers':len(rows),'reviewed_answers':len(rows)-len(failed),
            'api_calls':calls,'pending':failed,'summary':aggregate(scores),'scores':scores}
    write(args.output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('protocol','requested_answers','reviewed_answers','api_calls')}))
    return 2 if failed else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare',help='validate 96 cases and export the reviewer schema')
    prepare.add_argument('--cases',nargs='+')
    prepare.add_argument('--output',type=Path,required=True)
    for command in ('score','evaluate'):
        p=sub.add_parser(command)
        p.add_argument('--answers',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command=='score':
            p.add_argument('--reviews',type=Path,required=True)
        else:
            p.add_argument('--base-url',default=os.environ.get('FLOWINTENT_API_BASE','https://api.openai.com/v1'))
            p.add_argument('--reviewer-model',default='gpt-6-astra')
            p.add_argument('--effort',default='max')
            p.add_argument('--max-output-tokens',type=int,default=10000)
            p.add_argument('--timeout',type=float,default=350)
            p.add_argument('--max-prompt-chars',type=int,default=200000)
            p.add_argument('--max-api-calls',type=int,default=8)
    return run(parser.parse_args())


if __name__=='__main__':
    raise SystemExit(main())
