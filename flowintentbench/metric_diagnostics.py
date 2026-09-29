"""Evidence-only metric diagnostics: no model calls, I/O, or new score verdicts."""
from collections import Counter, defaultdict
from statistics import fmean

from .quality_efficiency import quality_status

DIAGNOSTIC_VERSION = 'metric-validity-diagnostics-v1'


def record_diagnostics(record):
    result = record.get('result', {})
    metrics = result.get('metrics')
    if not metrics or result.get('run_status') != 'COMPLETED':
        return None
    branches = record.get('operationalization_branch_evaluations', [])
    unresolved = branches[0]['unresolved_dimension_matches'] if branches else {}
    urs_branch = None
    if unresolved:
        urs_branch = min(branches, key=lambda b: (
            -sum(b['unresolved_dimension_matches'].values()),
            -sum(b['principal_dimension_matches'].values()), b['branch_id']))
    decisions = record.get('extracted_prediction', {}).get('operationalization', {}).get('decisions', [])
    statuses = {d['dimension']: d['status'] for d in decisions}
    f = metrics['scientific_findings']; c = metrics['o_f_consistency']
    best_f = set(f.get('best_finding_branches') or [])
    diagnostic = next((b for b in result.get('finding_branch_diagnostics', [])
                       if b['branch_id'] in best_f), None)
    findings = record.get('extracted_prediction', {}).get('findings', [])
    # Flag legacy routing for audit, never infer a corrected verdict.
    null_ids = {p['prediction_id'] for p in findings if p.get('value') is None}
    supported = set(diagnostic['consistent_applicable_prediction_ids']) if diagnostic else set()
    legacy_flags = sorted({v['prediction_id'] for v in record.get('value_verifications', [])
                          if v['prediction_id'] in null_ids and v['prediction_id'] not in supported
                          and v['branch_id'] in best_f and not v['verified']
                          and v['rule'] in {'scalar_tolerance', 'spatial_euclidean',
                                           'componentwise_vector', 'exact_discrete_numeric'}})
    return dict(version=DIAGNOSTIC_VERSION,
        urs_scope='Valid completion of unspecified O dimensions; not clarification or rationale quality',
        unresolved_dimension_count=len(unresolved),
        urs_diagnostic_branch=None if urs_branch is None else urs_branch['branch_id'],
        unresolved_dimension_credits=None if urs_branch is None else dict(urs_branch['unresolved_dimension_matches']),
        unresolved_extraction_statuses={d: statuses.get(d, 'MISSING') for d in unresolved},
        f_app_count=len(result.get('f_app_prediction_ids', [])),
        gt_outside_accepted_count=None if diagnostic is None else len(diagnostic['accepted_gt_outside_prediction_ids']),
        reference_core_count=None if diagnostic is None else diagnostic['core_finding_count'],
        matched_reference_core_count=None if diagnostic is None else diagnostic['matched_core_finding_count'],
        finding_recall_mode=f.get('finding_recall_mode'),
        c_scope='Scientific support conditional on declared O; not independent procedural compliance',
        c_applicable=c.get('c_score') is not None,
        c_unavailable_reason=c.get('unavailable_reason'),
        c_equals_precision=None if c.get('c_score') is None else c['c_score'] == f['finding_precision'],
        branch_alignment=c.get('branch_alignment'),
        legacy_qualitative_numeric_failure_ids=legacy_flags)


def _group(rows):
    scored = [r for r in rows if r.get('status') == 'COMPLETED'
              and r.get('evaluation_status') == 'SCORED']
    urs = [r for r in scored if r.get('urs') is not None]
    cs = [r for r in scored if r.get('c_score') is not None]
    dims = defaultdict(list)
    for r in scored:
        diagnostic = r.get('metric_diagnostics') or {}
        for d, value in (diagnostic.get('unresolved_dimension_credits') or {}).items():
            dims[d].append(value)
    cases = defaultdict(list)
    for r in rows:
        cases[r['case_id']].append(r)
    resolved, all_pass, infra = 0, 0, 0
    for trials in cases.values():
        if len(trials) != 3 or {r['trial'] for r in trials} != {1, 2, 3}:
            continue
        states = [quality_status(r) for r in trials]
        if 'INFRASTRUCTURE_INVALID' in states:
            infra += 1
        elif all(s in {'PASS', 'FAIL'} for s in states):
            resolved += 1
            all_pass += int(all(s == 'PASS' for s in states))
    eligible = len(cases)-infra
    return dict(completed_scored_answers=len(scored),
        urs=dict(n=len(urs), distinct_cases=len({r['case_id'] for r in urs}),
            mean=fmean(r['urs'] for r in urs) if urs else None,
            full_score_count=sum(r['urs']==1 for r in urs),
            dimension_count_distribution=dict(Counter(str((r.get('metric_diagnostics') or {}).get('unresolved_dimension_count', 'unknown')) for r in urs)),
            by_dimension={d: dict(n=len(v),mean=fmean(v)) for d,v in sorted(dims.items())}),
        c=dict(applicable_n=len(cs), unavailable_n=len(scored)-len(cs),
            equals_precision_n=sum(r['c_score']==r.get('finding_precision') for r in cs),
            branch_alignment_counts=dict(Counter(str(r.get('branch_alignment')) for r in cs))),
        legacy_routing_flagged_answers=sum(bool((r.get('metric_diagnostics') or {}).get('legacy_qualitative_numeric_failure_ids')) for r in scored),
        reliability_n3=dict(requested_cases=len(cases),infrastructure_excluded_cases=infra,
            resolved_cases=resolved,all_three_pass_cases=all_pass,
            all_three_pass_rate=all_pass/eligible if eligible and resolved==eligible else None,
            resolved_only_rate_diagnostic=all_pass/resolved if resolved else None,
            definition='All three frozen trials meet the unchanged quality gate; not best-of-three. Incomplete cases remain unresolved.'))


def metric_validity_summary(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r['model']][r['condition']].append(r)
    return dict(version=DIAGNOSTIC_VERSION,scope='Completed-answer diagnostics are selected-subset observations, not N3 rankings',
        new_model_calls=0,by_model={model:{condition:_group(group) for condition,group in sorted(conditions.items())}
                                  for model,conditions in sorted(groups.items())})
