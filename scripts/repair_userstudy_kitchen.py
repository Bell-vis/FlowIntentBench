#!/usr/bin/env python3
"""Apply the user-authorized open field selection reference, without rewording Q.

Only the mutable userstudy case is repaired. Immutable baseline artifacts and
metric formulas are preserved. No subject answer supplies reference numbers.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASE = "kitchen_concentration_heterogeneity_o1_f2"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n")


def repair(root=ROOT):
    from flowintentbench.evaluation_contract import FrozenEvaluationCase
    from flowintentbench.scientific_artifact_binding import artifact_sha256, build_scientific_artifact_binding
    from flowintentbench.deterministic_materialization import analyze_kitchen_concentration, MaterializationPlan
    from flowintentbench.ground_truth import GroundTruth
    from flowintentbench.srac import ScientificEvaluationContract
    from flowintentbench.evaluation_policy import compile_finding_verification_policy, bind_policy_to_execution_binding
    from flowintentbench.construction_tolerances import absolute_tolerance_for_significant_figures
    from flowintentbench.open_field_selection import POLICY
    folder = root/'experiments/userstudy/cases/Kitchen'/CASE
    manifest_path = root/'experiments/userstudy/case_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from flowintentbench.reference_authority import current_reference_path
    baseline = current_reference_path(root).parent/'cases'/'Kitchen'/CASE
    def read(name, base=folder):
        return json.loads((base/name).read_text())
    original_question = (folder/'case_input.json').read_bytes()
    backup = root/'outputs/validation/kitchen_open_selection_before'
    if not backup.exists():
        backup.mkdir(parents=True)
        for p in folder.glob('*.json'):
            (backup/p.name).write_bytes(p.read_bytes())
        (backup/'case_manifest.json').write_bytes(manifest_path.read_bytes())
    gt = read('ground_truth.json', baseline)
    sec = read('scientific_evaluation_contract.json', baseline)
    old_materialization = sec['G_of_O'][0]
    operation_id = gt['acceptable_operationalizations'][0]['operationalization_id']
    plan = MaterializationPlan(**old_materialization['execution']['data_provenance']['materialization_plan'])
    effective_o = {d['dimension']:d['statement'] for d in gt['acceptable_operationalizations'][0]['decisions']}
    computed = analyze_kitchen_concentration(root, effective_o, operation_id, _compiled_plan=plan)
    rows = computed['result']['all_field_metrics']
    original_refs = gt['findings_by_operationalization'][0]['findings']
    mappings = {'field':'field_name', 'heterogeneity':'metric_value', 'standard_deviation':'standard_deviation',
                'q10':'q10', 'q50':'q50', 'q90':'q90'}
    all_refs, execution_findings, by_field, identities = [], [], {}, []
    old_roles = sec['finding_requirement_contract']['reference_finding_role_map']
    new_roles = {}
    for i, row in enumerate(rows):
        name = row['field_name']; by_field[name] = []
        for reference in original_refs:
            ref = copy.deepcopy(reference)
            suffix = reference['finding_id'][len(operation_id)+1:]
            key = mappings[suffix]; value = row[key]
            ref['finding_id'] = reference['finding_id']+'__'+name
            ref['statement'] = (f'The selected gas-concentration field is {name}.' if suffix == 'field' else
                                f'For gas-concentration field {name}, the cell-volume-weighted '+
                                ('coefficient of variation' if suffix == 'heterogeneity' else suffix.replace('_',' '))+
                                f' is approximately {value:.10g}.')
            ref['value'] = value
            if suffix == 'field':
                identities.append(ref['finding_id'])
            else:
                tolerance = absolute_tolerance_for_significant_figures(float(value), significant_figures=3) if value else 0.0005
                ref['verification'] = {'absolute_tolerance':tolerance,'relative_tolerance':None,'spatial_tolerance':None}
            all_refs.append(ref); by_field[name].append(ref['finding_id'])
            new_roles[ref['finding_id']] = old_roles[reference['finding_id']]
            execution_findings.append({'finding_id':ref['finding_id'], 'g_of_o_locator':f'/all_field_metrics/{i}/{key}',
                'execution_value':value,'execution_value_sha256':artifact_sha256(value),'gt_value':value,
                'verification_mode':'exact_identity' if suffix == 'field' else 'scalar_tolerance',
                'verification_parameters':ref['verification'] or {},'binding_status':'PASS'})
    gt['findings_by_operationalization'][0]['findings'] = all_refs
    ground_truth = GroundTruth.model_validate(gt)
    target = 'a scientifically justified selection among the compared stored gas-concentration fields'
    metadata = read('case_construction_metadata.json')
    metadata['scientific_target'] = target
    metadata['responsibility_contract']['scientific_target']['normalized_meaning'] = target
    metadata['finding_goal'] = 'compare candidate fields by the specified volume-weighted CV, identify the selected field, and support its statistic and an additional characterization'
    projection = read('scientific_semantic_projection.json')
    projection['scientific_target'] = target
    projection['finding_semantics']['finding_goal'] = metadata['finding_goal']
    projection['finding_semantics']['normalized_demand'] = metadata['finding_goal']
    semantic_hash = artifact_sha256(projection)
    result = computed['result']
    result['field_selection_policy'] = {'mode':POLICY,'reference_ids_by_field':by_field,'identity_reference_ids':identities,
        'default_anchor_is_not_required_selection':True,'selection_rule':'one explicitly selected candidate; no required extremum'}
    computed['execution']['G_of_O'] = result
    record = copy.deepcopy(old_materialization)
    record['execution'] = computed['execution'];record['findings'] = all_refs;record['parameters'] = effective_o
    sec['G_of_O'] = [record]
    sec['source_semantic_contract_sha256'] = semantic_hash
    sec['finding_requirement_contract']['reference_finding_role_map'] = new_roles
    binding = read('finding_execution_binding.json', baseline)
    binding['semantic_contract_sha256'] = semantic_hash
    binding['branches'][0].update(findings=execution_findings,g_of_o_sha256=artifact_sha256(result),
        effective_o_sha256=computed['execution']['data_provenance']['effective_o_sha256'])
    policy = compile_finding_verification_policy(ground_truth=ground_truth,finding_execution_binding=binding,
        scientific_target=target,semantic_contract_sha256=semantic_hash)
    binding = bind_policy_to_execution_binding(binding,policy)
    sec['tolerances'] = {'verification_policy_version':policy['schema_version'],
        'verification_policy_sha256':policy['verification_policy_sha256'],'finding_verification_policies':policy['policies']}
    contract = ScientificEvaluationContract(**sec)
    artifact_binding = build_scientific_artifact_binding(case_id=CASE,semantic_contract=projection,
        ground_truth=ground_truth,scientific_evaluation_contract=contract,presentation=json.loads(original_question)['scientific_question'])
    for name, value in [('ground_truth.json',ground_truth.model_dump(mode='json')),
        ('scientific_evaluation_contract.json',contract.to_dict()),('scientific_artifact_binding.json',artifact_binding.to_dict()),
        ('case_construction_metadata.json',metadata),('scientific_semantic_projection.json',projection),
        ('finding_execution_binding.json',binding),('finding_verification_policy.json',policy),
        ('materialization_'+operation_id+'.json',computed)]:
        write(folder/name,value)
    from flowintentbench.qualified_scientific_cases import _agent_scientific_evidence_coverage
    from flowintentbench.context import EvidenceRecord
    evidence = read('agent_ready_evidence.json')
    for item in evidence:
        if item['evidence_id'] == 'finding_001':
            item['statement'] = 'Independent cell-volume-weighted computation of CV and distribution statistics for all eight candidate concentration fields; no maximum-only selection requirement.'
    write(folder/'agent_ready_evidence.json',evidence)
    coverage = _agent_scientific_evidence_coverage(CASE,'Kitchen',ground_truth,[EvidenceRecord.model_validate(e) for e in evidence])
    write(folder/'scientific_evidence_coverage.json',coverage)
    question = read('scientific_question.json');question['semantic_contract_sha256'] = semantic_hash
    write(folder/'scientific_question.json',question)
    write(folder/'materialization_trace.json',{'repair':'open field selection from unchanged ZIP wording',
        'stage_b_materializations':[computed], 'previous_trace':str(backup/'materialization_trace.json'),
        'reference_numbers_source':'independent dataset execution; no model response consulted'})
    FrozenEvaluationCase.from_paths(folder).validate()
    assert (folder/'case_input.json').read_bytes() == original_question
    row = next(r for r in manifest['cases'] if r['case_id'] == CASE)
    for key, value in list(row.items()):
        if key.endswith('_path') and isinstance(value,str) and (root/value).parent == folder:
            row[key[:-5]+'_sha256'] = hashlib.sha256((root/value).read_bytes()).hexdigest()
    # Preserve the original ZIP provenance; record this explicit correction separately.
    report = {'status':'REPAIRED','case_id':CASE,'question_unchanged':True,'metric_formulas_unchanged':True,
        'candidate_count':len(rows),'reference_finding_count':len(all_refs),'forced_recollection_models':['gpt-5.6-luna','gpt-5.6-terra'],
        'selection_policy':POLICY,'previous_artifacts':str(backup),'field_metrics':rows}
    write(folder/'open_selection_repair.json',report)
    write(manifest_path,manifest)
    return report


if __name__ == '__main__':
    result=repair()
    print(json.dumps({k:v for k,v in result.items() if k != 'field_metrics'},ensure_ascii=False,indent=2))
