#!/usr/bin/env python3
"""Budgeted additive support-reference review of already scored answers.

No re-extraction, answer execution or dataset loading. Original reviews remain
immutable. Separate receipts authorize only additional semantic reference edges.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.reference_extension_review import request, apply, additions, partition_scope, semantic_reference_texts
from flowintentbench.reference_packages import load_supplements, apply_supplement, package
from flowintentbench.rubric_scoring import _prepare_review, _atomize_ranges, score, digest
from flowintentbench.frozen_evidence import load_bank
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.review_repairs import apply_repair
from scripts.evaluate_model_answers import read, sha, validated_cache, write_report
from scripts.run_core_case_scoring import _parse_json


def source_answer(prompt, expected_hash):
    packet=json.loads(prompt.rsplit('\n\n',1)[1])
    lines=packet['answer'].splitlines()
    if any(not line.startswith(f'L{i}: ') for i,line in enumerate(lines,1)):
        raise ValueError('invalid numbered source answer')
    text='\n'.join(line.split(': ',1)[1] for line in lines)
    for candidate in (text,text+'\n',text.replace('\n','\r\n'),(text+'\n').replace('\n','\r\n')):
        if hashlib.sha256(candidate.encode()).hexdigest()==expected_hash:return candidate
    raise ValueError('answer hash does not match preserved request')


def original_answer(row,contract,manifest):
    """Recover original bytes when splitlines removed e.g. a form-feed in TeX."""
    from flowintentbench.answer_collections import collect
    from scripts.evaluate_model_answers import load_run_source
    inputs=contract['source_identity']['inputs']
    for entry in inputs:
        if sha(Path(entry['path']))!=entry['sha256']:raise ValueError('original answer collection changed')
    collections=[Path(x['path']).parent for x in inputs if x['kind']=='COLLECTION']
    exports=[Path(x['path']) for x in inputs if x['kind']=='ANSWER_EXPORT']
    slots,_,_=collect(collections or None,exports or None,manifest,models=[row['model_id']],
                      case_ids=[row['case_id']],trials=[row['trial']])
    selected=[s for s in slots if s['output_id']==row['output_id'] and s.get('run_id',s.get('slot_id'))==row['run_id']]
    if len(selected)!=1:raise ValueError('original answer identity ambiguous')
    slot=selected[0]
    answer=slot['answer'] if 'export_source' in slot else load_run_source(slot,Path(slot['collection']),slot['experiment_id'])[0]
    if hashlib.sha256(answer.encode()).hexdigest()!=row['answer_sha256']:raise ValueError('original answer hash differs')
    return answer


def replay_receipt(directory, request_id):
    report_path=directory/'extension_report.json'
    if report_path.exists():
        rows=[r for r in read(report_path)['rows'] if r.get('request_id')==request_id and r.get('receipt')]
        if len(rows)>1:raise ValueError('ambiguous extension receipt lineage')
        if rows:
            path=Path(rows[0]['receipt'])
            if sha(path)!=rows[0]['receipt_sha256']:raise ValueError('extension receipt lineage changed')
            return path
    direct=directory/'api'/request_id/'receipt.json'
    if direct.exists():return direct
    return None


def retry_receipt_path(directory, request_id, receipt):
    """Allocate an immutable attempt only for a terminal transport failure."""
    if receipt.get('completed') or not receipt.get('finished_epoch'):
        raise ValueError('only a finished incomplete API attempt can be retried')
    attempt=1
    while (directory/'api'/request_id/f'attempt-{attempt}').exists():
        attempt+=1
    return directory/'api'/request_id/f'attempt-{attempt}'/'receipt.json'


def provenance_only_reference_alias(directory, case_id, current):
    """Authorize a former hash only when reference contents are unchanged.

    Package hashes include the containing source file. Editing another case in
    that file must not require a new semantic call for this unchanged case.
    Authority, task, branch descriptions, values, policies and pointers remain
    part of the comparison; only source location/checksum and the derived
    package fingerprint are excluded.
    """
    report_path = directory/'reports'/'experiment_report.json'
    if not report_path.exists():
        return None
    base = read(report_path).get('source_identity', {}).get('base_evaluation')
    if not isinstance(base, str):
        return None
    path = Path(base)/'reference_packages'/(digest(case_id)+'.json')
    if not path.exists():
        return None
    previous = read(path)
    def content(package):
        result = deepcopy(package)
        result.pop('package_sha256', None)
        for item in result.get('supplement_sources', []):
            source = item.get('source', {})
            source.pop('path', None)
            source.pop('sha256', None)
        return result
    return digest(previous) if content(previous) == content(current) else None


def identical_prompt_receipt(directory,prompt,schema,binding, *, old_reference_alias=None):
    """Reuse a number-blind mapping only when its entire API input is identical.

    The new hidden reference package may differ. The original answer, extracted
    judgment, old references, selected claims, protocol and API settings may not.
    The new scoring contract still records the new reference package separately.
    """
    def semantic_binding(b):
        result = {k:v for k,v in b.items() if k!='new_reference_sha256'}
        if old_reference_alias and result.get('old_reference_sha256') == old_reference_alias:
            result['old_reference_sha256'] = binding['old_reference_sha256']
        return result
    for path in sorted((directory/'requests').glob('*.json')):
        old=read(path)
        if (old['prompt']!=prompt or old['schema']!=schema
                or semantic_binding(old['binding'])!=semantic_binding(binding)):
            continue
        if path.stem!=digest(old['binding']):raise ValueError('replay request binding changed')
        receipt=replay_receipt(directory,path.stem)
        if receipt and read(receipt).get('completed'):
            return receipt,{'request_id':path.stem,'request':str(path.resolve()),'request_sha256':sha(path),
                'reason':'IDENTICAL_NUMBER_BLIND_API_INPUT; NEW_HOST_REFERENCE_CONTRACT',
                'provenance_only_old_reference_alias':old_reference_alias}
    return None


def validated_extension(receipt_path,prompt,schema,reviewer):
    receipt=read(receipt_path);payload=read(receipt_path.with_name('request.json'))
    instructions='Return only JSON matching the supplied schema. No tools.\n'+json.dumps(schema,ensure_ascii=False)
    if not receipt.get('completed'):
        raise ValueError('extension API incomplete: ' + str(receipt.get('error') or 'no terminal response'))
    if (receipt.get('response_model')!=reviewer['model']
        or receipt.get('reasoning_effort')!=reviewer['effort']
        or payload.get('input')!=[{'role':'user','content':prompt}]
        or payload.get('instructions')!=instructions or payload.get('max_output_tokens')!=reviewer['max_output_tokens']):
        raise ValueError('extension receipt does not bind the request')
    extension=_parse_json(receipt['final_text'])
    from jsonschema import Draft202012Validator
    Draft202012Validator(schema).validate(extension)
    return extension


def carry_prior_mapping(directory,row,answer,judgment,old,prior,reviewer):
    """Replay the exact earlier mapping; never trust its reported numeric score."""
    rid=row['request_id'];saved=read(directory/'requests'/(rid+'.json'));bound=saved['binding']
    if digest(bound)!=rid:raise ValueError('prior mapping request identity changed')
    ids=set(bound['finding_ids'])
    prompt,schema,aliases,binding=request(answer,judgment,old,prior,ids,
        compact_findings=bound.get('compact_findings',False),partition_aware=bound.get('partition_aware',False),
        semantic_support=bound.get('semantic_support',False))
    expected={**binding,**reviewer,'prompt_sha256':digest(prompt),'schema_sha256':digest(schema)}
    if bound!=expected or saved['prompt']!=prompt or saved['schema']!=schema:
        raise ValueError('prior mapping does not bind current source and references')
    receipt=Path(row['receipt'])
    if sha(receipt)!=row['receipt_sha256']:raise ValueError('prior mapping receipt changed')
    payload=validated_extension(receipt,prompt,schema,reviewer)
    extra=additions(old,prior)
    scopes={a:partition_scope(extra[key]['statement']) for a,key in aliases.items()} if bound.get('partition_aware') else None
    semantic_texts = semantic_reference_texts(extra, aliases) if bound.get('semantic_support') else None
    result,audit=apply(answer,judgment,aliases,payload,ids,partition_scopes=scopes,semantic_texts=semantic_texts)
    return result,{'request_id':rid,'request':str((directory/'requests'/(rid+'.json')).resolve()),
        'request_sha256':sha(directory/'requests'/(rid+'.json')),'receipt':str(receipt),
        'receipt_sha256':sha(receipt),'mapping_audit':audit}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference-package',type=Path,required=True)
    p.add_argument('--evidence-bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+')
    p.add_argument('--models',nargs='+')
    p.add_argument('--trials',type=int,nargs='+')
    p.add_argument('--max-api-calls',type=int,default=1)
    p.add_argument('--max-output-tokens',type=int,default=2000)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--shared-concurrency',type=int,default=5)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--replay-from',type=Path,help='Reuse only identical extension request receipts')
    p.add_argument('--require-identified',action='store_true')
    p.add_argument('--compact-new-requests',action='store_true',help='Omit repeated finding quotations for new requests; retain the complete numbered answer')
    p.add_argument('--retry-failed-receipts',action='store_true',help='Retry terminal incomplete API attempts within the call budget, retaining the original receipt')
    p.add_argument('--replay-identical-prompts',action='store_true',help='Reuse a number-blind mapping across hidden reference changes only if the full API input and original judgment are identical')
    p.add_argument('--partition-aware',action='store_true',help='Require source-declared bin counts for conditional reference mappings')
    p.add_argument('--semantic-support',action='store_true',help='Separately review truth of frozen textual references, with verbatim reference evidence; numeric truth stays host-only')
    p.add_argument('--prior-extension',type=Path,help='Carry validated earlier matching edges before requesting only newly added references')
    p.add_argument('--prior-reference-package',type=Path,help='Exact frozen package used by --prior-extension')
    p.add_argument('--auth-path',type=Path,default=ROOT/'auth_yapi.json')
    p.add_argument('--api-config',type=Path,default=ROOT/'config/yiapi_direct.toml')
    args=p.parse_args()
    if args.max_api_calls<0 or min(args.max_output_tokens,args.timeout,args.shared_concurrency)<=0:p.error('invalid budget')
    if bool(args.prior_extension)!=bool(args.prior_reference_package):p.error('prior extension and reference package are required together')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    report=read(args.source/'reports/experiment_report.json');contract=read(args.source/'evaluation_contract.json')
    supplements=load_supplements(args.reference_package,ROOT);bank=load_bank(args.evidence_bank,ROOT)
    manifest=ROOT/'experiments/expansion_v1_development/case_manifest.json'
    if sha(manifest)!=contract['source_identity']['manifest_sha256']:raise ValueError('manifest identity changed')
    if sha(args.evidence_bank)!=contract['source_identity']['evidence_bank_sha256']:raise ValueError('auxiliary bank changed')
    cases={r['case_id']:r for r in read(manifest)['cases']};reviewer_info=contract['reviewer']
    rows=[r for r in report['rows'] if r['status']=='SCORED'
          and (not args.cases or r['case_id'] in args.cases) and (not args.models or r['model_id'] in args.models)
          and (not args.trials or r['trial'] in args.trials)]
    identity={'source_report_sha256':sha(args.source/'reports/experiment_report.json'),
        'source_contract_sha256':sha(args.source/'evaluation_contract.json'),
        'reference_package_sha256':sha(args.reference_package),
        'code_sha256':{name:sha(ROOT/name) for name in sorted(set(contract['source_identity']['scoring_dependencies_sha256']) |
            {'scripts/extend_evaluation_references.py','flowintentbench/reference_extension_review.py'})},
        'selected_output_ids':[r['output_id'] for r in rows],
        'reviewer':{'model':reviewer_info['model'],'effort':'medium','max_output_tokens':args.max_output_tokens}}
    if args.replay_from:identity['receipt_replay_source']=str(args.replay_from.resolve())
    if args.compact_new_requests:identity['compact_new_requests']=True
    if args.retry_failed_receipts:identity['retry_failed_receipts']=True
    if args.replay_identical_prompts:identity['replay_identical_prompts']=True
    if args.partition_aware:identity['partition_aware']=True
    if args.semantic_support:identity['semantic_support']=True
    prior_rows={};prior_packages={};prior_contract=None
    if args.prior_extension:
        prior_contract=read(args.prior_extension/'extension_contract.json')
        if (prior_contract['source_report_sha256']!=identity['source_report_sha256']
            or prior_contract['source_contract_sha256']!=identity['source_contract_sha256']
            or prior_contract['reference_package_sha256']!=sha(args.prior_reference_package)):
            raise ValueError('prior extension source/reference identity changed')
        selected_prior=[r for r in read(args.prior_extension/'extension_report.json')['rows'] if r['status']=='SCORED']
        prior_rows={r['output_id']:r for r in selected_prior}
        if len(prior_rows)!=len(selected_prior):raise ValueError('duplicate prior extension output identity')
        prior_packages=load_supplements(args.prior_reference_package,ROOT)
        identity['prior_extension']={'path':str(args.prior_extension.resolve()),
            'report_sha256':sha(args.prior_extension/'extension_report.json'),
            'contract_sha256':sha(args.prior_extension/'extension_contract.json'),
            'reference_package_sha256':sha(args.prior_reference_package)}
    cp=args.output/'extension_contract.json'
    if cp.exists() and read(cp)!=identity:raise ValueError('extension inputs changed; use a new output directory')
    write_json(cp,identity)
    jobs=[];skipped=[];prompt_replays={};scopes_by_request={};semantic_by_request={};prior_baselines={}
    for row in rows:
        ci,meta,gt,material=load_development_case(ROOT,cases[row['case_id']])
        gt,material=apply_supplement(ci,meta,gt,material,supplements.get(row['case_id']))
        material['frozen_auxiliary_evidence']=bank.get(row['case_id'],[])
        new=package(ci,meta,gt,material)
        old=read(args.source/'reference_packages'/(digest(row['case_id'])+'.json'))
        extra=additions(old,new)
        if not extra:continue
        partition_aware=args.partition_aware or any(partition_scope(f['statement']) for f in extra.values())
        rid=row['provenance']['request_id'];rp=args.source/'requests'/(rid+'.json');req=read(rp)
        stored=validated_cache(args.source/'judgments'/(rid+'.json'),rid,req['prompt'],req['schema'],reviewer_info['model'],
            effort=reviewer_info['effort'],max_output_tokens=reviewer_info['max_output_tokens'],
            structured_output=reviewer_info.get('structured_output',False))
        try:
            answer=source_answer(req['prompt'],row['answer_sha256'])
        except ValueError:
            answer=original_answer(row,contract,manifest)
        raw=stored['judgment'];repair=row['provenance'].get('review_repair')
        if repair:
            if sha(Path(repair['path']))!=repair['sha256']:raise ValueError('source repair changed')
            raw=apply_repair(raw,req['prompt'],read(repair['path']))
        old_gt=deepcopy(gt)
        by_branch={b.operationalization_id:b for b in old_gt.findings_by_operationalization}
        for b in old['branches']:
            current=by_branch[b['branch_id']];by_id={f.finding_id:f for f in current.findings}
            current.findings=[by_id[f['finding_id']] for f in b['findings']]
        old_gt.findings_by_operationalization=[by_branch[b['branch_id']] for b in old['branches']]
        refs={b.operationalization_id:{f.finding_id:f for f in b.findings} for b in old_gt.findings_by_operationalization}
        # Retain the original claim representation in the semantic request so
        # source-bound receipts remain reusable. Scoring always applies the
        # current reference-independent endpoint atomization.
        prepared,_,invalid=_prepare_review(raw,refs,old_gt,answer,atomize_ranges=False)
        if invalid:
            # Keep this answer's original bounds. Normalizing an invalid value
            # to None and rescoring would confuse extraction failure with an
            # omitted answer. Other answers can still receive their extensions.
            skipped.append({'output_id':row['output_id'],'case_id':row['case_id'],
                'model_id':row['model_id'],'trial':row['trial'],'status':'SOURCE_EXTRACTION_REPAIR_REQUIRED',
                'invalid_finding_ids':sorted(invalid)})
            continue
        judgment=prepared.model_dump(mode='json')
        baseline=score(answer,meta,old_gt,material,raw,condition=row['condition'],case_input=ci)
        normalized=score(answer,meta,old_gt,material,judgment,condition=row['condition'],case_input=ci)
        if any(baseline.get(k)!=normalized.get(k) for k in ('metrics','error_diagnostics','extraction_complete')):
            skipped.append({'output_id':row['output_id'],'case_id':row['case_id'],
                'model_id':row['model_id'],'trial':row['trial'],'status':'SOURCE_NORMALIZATION_REPAIR_REQUIRED'})
            continue
        if row['output_id'] in prior_rows:
            pci,pmeta,pgt,pmaterial=load_development_case(ROOT,cases[row['case_id']])
            pgt,pmaterial=apply_supplement(pci,pmeta,pgt,pmaterial,prior_packages.get(row['case_id']))
            pmaterial['frozen_auxiliary_evidence']=bank.get(row['case_id'],[])
            prior_package=package(pci,pmeta,pgt,pmaterial)
            judgment,proof=carry_prior_mapping(args.prior_extension,prior_rows[row['output_id']],answer,
                judgment,old,prior_package,prior_contract['reviewer'])
            old=prior_package;old_gt=pgt;extra=additions(old,new)
            partition_aware=args.partition_aware or any(partition_scope(f['statement']) for f in extra.values())
            refs={b.operationalization_id:{f.finding_id:f for f in b.findings} for b in old_gt.findings_by_operationalization}
            prepared,_,invalid=_prepare_review(judgment,refs,old_gt,answer,atomize_ranges=False)
            if invalid:raise ValueError('validated prior mapping produced invalid extraction')
            judgment=prepared.model_dump(mode='json')
            baseline=score(answer,meta,old_gt,pmaterial,judgment,condition=row['condition'],case_input=ci)
            row={**row,**baseline,'provenance':{**row['provenance'],'prior_support_review':proof}}
            prior_baselines[row['output_id']]=row
        known={f['finding_id'] for f in judgment['findings']}
        gaps={g['finding_id'] for g in baseline.get('reference_gaps',[]) if g['reason']=='REFERENCE_GAP'}
        gaps|={c['finding_id'] for c in baseline.get('finding_checks',[]) if c['reason']=='REVIEW_VALUE_SHAPE_MISMATCH'}
        range_repairs=[]
        _atomize_ranges(prepared.findings,refs,range_repairs)
        finding_ids=(gaps & known) | {r['finding_id'] for r in range_repairs if gaps & set(r['child_finding_ids'])}
        if not finding_ids or not extra:continue
        prompt,schema,aliases,binding=request(answer,judgment,old,new,finding_ids,partition_aware=partition_aware,
                                            semantic_support=args.semantic_support)
        bound={**binding,**identity['reviewer'],'prompt_sha256':digest(prompt),'schema_sha256':digest(schema)}
        eid=digest(bound)
        old_alias=(provenance_only_reference_alias(args.replay_from,row['case_id'],old)
                   if args.replay_from and args.replay_identical_prompts else None)
        reusable=(identical_prompt_receipt(args.replay_from,prompt,schema,bound,old_reference_alias=old_alias)
                  if args.replay_from and args.replay_identical_prompts else None)
        if (args.compact_new_requests and not replay_receipt(args.output,eid)
                and not (args.replay_from and replay_receipt(args.replay_from,eid)) and not reusable):
            prompt,schema,aliases,binding=request(answer,judgment,old,new,finding_ids,compact_findings=True,partition_aware=partition_aware,
                                                semantic_support=args.semantic_support)
            bound={**binding,**identity['reviewer'],'prompt_sha256':digest(prompt),'schema_sha256':digest(schema)}
            eid=digest(bound)
            reusable=(identical_prompt_receipt(args.replay_from,prompt,schema,bound,old_reference_alias=old_alias)
                      if args.replay_from and args.replay_identical_prompts else None)
        if reusable:prompt_replays[eid]=reusable
        if partition_aware:
            scopes_by_request[eid]={alias:partition_scope(extra[key]['statement']) for alias,key in aliases.items()}
        if args.semantic_support:
            semantic_by_request[eid] = semantic_reference_texts(extra, aliases)
        write_json(args.output/'requests'/(eid+'.json'),{'prompt':prompt,'schema':schema,'binding':bound})
        jobs.append((row,ci,meta,gt,material,answer,judgment,prompt,schema,aliases,finding_ids,eid))
    costs={'api_calls':0,'input_tokens':0,'output_tokens':0,'usage_missing_calls':0,'errors':0}
    results=list(skipped);reviewer=None
    combined={r['output_id']:deepcopy(r) for r in report['rows']}
    combined.update(prior_baselines)
    # Snapshot before incremental report writes replace its previous rows.
    cached_receipts={job[-1]:replay_receipt(args.output,job[-1]) for job in jobs}
    for row,ci,meta,gt,material,answer,judgment,prompt,schema,aliases,finding_ids,eid in jobs:
        receipt_path=cached_receipts[eid] or args.output/'api'/eid/'receipt.json'
        if args.replay_from and not receipt_path.exists():
            previous=replay_receipt(args.replay_from,eid)
            if previous:
                # Keep the authoritative receipt at its original path so
                # cross-run cost audits cannot count a copied call twice.
                receipt_path=previous
        if not receipt_path.exists() and eid in prompt_replays:
            receipt_path=prompt_replays[eid][0]
        receipt=read(receipt_path) if receipt_path.exists() else None
        result={'output_id':row['output_id'],'case_id':row['case_id'],'model_id':row['model_id'],'trial':row['trial'],
                'request_id':eid,'selected_findings':len(finding_ids),'new_references':len(aliases),
                'before_metrics':row['metrics'],'status':'NOT_REVIEWED'}
        if eid in prompt_replays and receipt_path==prompt_replays[eid][0]:
            result['identical_prompt_replay']=prompt_replays[eid][1]
        try:
            if (receipt and not receipt.get('completed') and receipt.get('finished_epoch')
                    and args.retry_failed_receipts and not args.offline and costs['api_calls']<args.max_api_calls):
                result['previous_failed_receipt']={'path':str(receipt_path),'sha256':sha(receipt_path)}
                receipt_path=retry_receipt_path(args.output,eid,receipt)
                receipt=None
            if receipt is None and not args.offline and costs['api_calls']<args.max_api_calls:
                if reviewer is None:
                    from scripts.evaluate_answered_outcomes import configure_review_credentials
                    from scripts.outcome_responses_transport import ResponsesOutcomeReviewer
                    configure_review_credentials(args.auth_path,args.api_config)
                    reviewer=ResponsesOutcomeReviewer(args.api_config,concurrency=args.shared_concurrency)
                t=time.monotonic()
                receipt=reviewer(prompt,reviewer_info['model'],args.output/'work',receipt_path.parent,
                    timeout_seconds=args.timeout,output_schema=schema,reasoning_effort='medium',max_output_tokens=args.max_output_tokens)
                costs['api_calls']+=1
                for key in ('input_tokens','output_tokens'):costs[key]+=receipt.get('usage',{}).get(key,0)
                costs['usage_missing_calls']+=int(not all(k in receipt.get('usage',{}) for k in ('input_tokens','output_tokens')))
                result['new_seconds']=time.monotonic()-t
            if receipt:
                result.update(receipt=str(receipt_path),receipt_sha256=sha(receipt_path))
                extension=validated_extension(receipt_path,prompt,schema,identity['reviewer'])
                extended,audit=apply(answer,judgment,aliases,extension,finding_ids,partition_scopes=scopes_by_request.get(eid),
                                     semantic_texts=semantic_by_request.get(eid))
                scored=score(answer,meta,gt,material,extended,condition=row['condition'],case_input=ci)
                result.update(status='SCORED',after_metrics=scored['metrics'],mapping_audit=audit,
                              receipt=str(receipt_path),receipt_sha256=sha(receipt_path))
                combined[row['output_id']]={**row,**scored,
                    'provenance':{**row['provenance'],'additive_support_review':{'request_id':eid,
                        'receipt':str(receipt_path),'receipt_sha256':sha(receipt_path),
                        'identical_prompt_replay':result.get('identical_prompt_replay'),
                        'base_evaluation':str(args.source.resolve()),'extension_contract_sha256':sha(cp)}}}
        except Exception as exc:
            costs['errors']+=1;result.update(status='REVIEW_ERROR',error=f'{type(exc).__name__}: {exc}')
        results.append(result)
        write_json(args.output/'extension_report.json',{'contract':identity,'cost':costs,'rows':results,
                   'remaining_jobs':len(jobs)-len(results)+len(skipped),'note':'Additional supporting mappings only; no re-extraction or solver execution.'})
        print(json.dumps({k:result[k] for k in ('case_id','model_id','trial','status','selected_findings','new_references')}),flush=True)
    for row in results:
        combined[row['output_id']]['support_extension_status']=row['status']
    # A valid run may need no new review after carrying prior mappings. Keep a
    # complete ledger even in that zero-call case.
    write_json(args.output/'extension_report.json',{'contract':identity,'cost':costs,'rows':results,
               'remaining_jobs':0,'note':'Additional supporting mappings only; no re-extraction or solver execution.'})
    summary=write_report(args.output,list(combined.values()),costs,{
        'base_evaluation':str(args.source.resolve()),'extension_contract':identity,
        'scope':'Only additive supporting references; unreviewed extensions retain original identification bounds.',
        'extension_review_complete':all(r['status']=='SCORED' for r in results)})
    print(json.dumps(costs),flush=True)
    if any(r['status']!='SCORED' for r in results):return 2
    if args.require_identified and summary['status']!='COMPLETE':return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
