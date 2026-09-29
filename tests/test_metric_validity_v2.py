"""Controlled scientific defects and reporting diagnostics, with no model calls."""
from dataclasses import replace
from copy import deepcopy
import pytest
from tests.test_evaluator import (_prediction,_evaluator,_metadata,_case_input,_ground_truth,
                                 _adjudications,StubMatcher)
from flowintentbench.evaluator import (ExtractionStatus,SemanticMatchPurpose,
    EvaluationPendingAdjudication,AdjudicationStatus,EvaluationAdjudications,SemanticMatchResult)
from flowintentbench.evaluation_metrics import EfficiencyObservation,compute_o_score
from flowintentbench.metric_diagnostics import metric_validity_summary,record_diagnostics


def evaluate(prediction, *, matcher=None, adjudications=None):
    evaluator,_ = _evaluator(prediction,matcher)
    result=evaluator.evaluate_response(_case_input(),_metadata(),_ground_truth(),
        final_response='synthetic',efficiency=EfficiencyObservation(1,1,1,0,1),
        adjudications=_adjudications() if adjudications is None else adjudications)
    return result


def qualitative_fixture():
    p=_prediction();p=replace(p,findings=tuple(replace(f,statement='The mean strength is low.') if f.prediction_id=='extra' else f for f in p.findings))
    class RouteMatcher(StubMatcher):
        def match(self, request):
            if (request.purpose==SemanticMatchPurpose.FINDING
                    and request.predicted_id=='extra' and request.reference_id=='mean_strength'):
                self.requests.append(request)
                return SemanticMatchResult.MATCH
            return super().match(request)
    matcher=RouteMatcher(equivalent_pairs=[('The strongest region is B.','Region B is the strongest.')])
    return p,matcher


def test_qualitative_summary_uses_existing_adjudication_without_numeric_semantic_call():
    p,m=qualitative_fixture();r=evaluate(p,matcher=m)
    assert r.metrics.scientific_findings.finding_precision==1
    assert r.metrics.scientific_findings.finding_requirement_recall==1
    assert not any(x.purpose==SemanticMatchPurpose.FINDING and x.predicted_id=='extra'
                   and x.verification_mode.value=='scalar_tolerance' for x in m.requests)


def test_true_qualitative_claim_cannot_replace_required_numerical_finding():
    p,m=qualitative_fixture();p=replace(p,findings=tuple(f for f in p.findings if f.prediction_id!='strength'))
    r=evaluate(p,matcher=m)
    assert r.metrics.scientific_findings.finding_precision==1
    assert r.metrics.scientific_findings.finding_requirement_recall==.5


def test_qualitative_claim_without_scientific_evidence_stays_pending():
    p,m=qualitative_fixture();a=_adjudications();values=dict(a.novel_findings);values.pop(('branch_mean','extra'))
    with pytest.raises(EvaluationPendingAdjudication) as exc:
        evaluate(p,matcher=m,adjudications=EvaluationAdjudications(novel_findings=values))
    assert exc.value.pending_type=='gt_outside_finding'


def test_false_qualitative_claim_is_not_auto_accepted():
    p,m=qualitative_fixture();r=evaluate(p,matcher=m,adjudications=_adjudications(accept_extra_mean=False))
    assert r.metrics.scientific_findings.finding_precision==pytest.approx(2/3)


def test_incorrect_explicit_number_cannot_escape_into_novel_finding_credit():
    p,m=qualitative_fixture();p=replace(p,findings=tuple(replace(f,value=999.) if f.prediction_id=='strength' else f for f in p.findings))
    a=_adjudications();values=dict(a.novel_findings);values['branch_mean','strength']=AdjudicationStatus.ACCEPTED
    r=evaluate(p,matcher=m,adjudications=EvaluationAdjudications(novel_findings=values))
    assert r.metrics.scientific_findings.finding_precision==pytest.approx(2/3)
    assert r.metrics.scientific_findings.finding_requirement_recall==.5
    assert r.metrics.scientific_operationalization.urs==1


@pytest.mark.parametrize('status',[ExtractionStatus.MISSING,ExtractionStatus.AMBIGUOUS,ExtractionStatus.CONFLICTING])
def test_urs_detects_missing_or_unresolved_choice(status):
    r=evaluate(_prediction(property_status=status))
    assert r.metrics.scientific_operationalization.urs==0
    assert r.metrics.o_f_consistency.c_score is None


def test_fixed_constraint_defect_does_not_lower_urs():
    p=_prediction();ds=tuple(replace(d,status=ExtractionStatus.MISSING,normalized_statement=None) if d.dimension=='feature_definition' else d for d in p.operationalization.decisions)
    r=evaluate(replace(p,operationalization=replace(p.operationalization,decisions=ds)))
    assert r.metrics.scientific_operationalization.urs==1
    assert r.metrics.scientific_operationalization.resolved_o_compliance==0


def test_c_detects_correct_findings_for_different_declared_method():
    p=_prediction();ds=tuple(replace(d,normalized_statement='Use peak strength.') if d.dimension=='property_measure' else d for d in p.operationalization.decisions)
    r=evaluate(replace(p,operationalization=replace(p.operationalization,decisions=ds)))
    assert r.metrics.scientific_operationalization.urs==1
    assert r.metrics.scientific_findings.finding_precision==1
    assert r.metrics.o_f_consistency.c_score==0
    assert r.metrics.o_f_consistency.branch_alignment==0


def test_urs_cannot_cherry_pick_dimensions_from_incompatible_branches():
    b=[dict(branch_id='a',principal_dimension_matches={'x':1,'y':0},unresolved_dimension_matches={'x':1,'y':0}),
       dict(branch_id='b',principal_dimension_matches={'x':0,'y':1},unresolved_dimension_matches={'x':0,'y':1})]
    assert compute_o_score(b).urs==.5


def rows(statuses):
    return [dict(model='m',case_id='c',trial=i+1,condition='O2-F1',status=s,
                 evaluation_status='SCORED' if s in ('COMPLETED','MODEL_NONCOMPLETION') else 'PENDING',
                 o_score=1,urs=1,finding_precision=1,finding_requirement_recall=1,c_score=1)
            for i,s in enumerate(statuses)]


@pytest.mark.parametrize('statuses,expected',[(['COMPLETED']*3,1.),
    (['COMPLETED','COMPLETED','MODEL_NONCOMPLETION'],0.),
    (['COMPLETED','COMPLETED','PENDING'],None),
    (['COMPLETED','COMPLETED','INFRASTRUCTURE_INVALID'],None)])
def test_n3_reliability_not_best_of_three_or_pending_zero(statuses,expected):
    s=metric_validity_summary(rows(statuses))['by_model']['m']['O2-F1']
    assert s['reliability_n3']['all_three_pass_rate']==expected


def test_one_complete_case_does_not_hide_another_pending_case():
    rs=rows(['COMPLETED']*3)+[dict(r,case_id='d') for r in rows(['PENDING']*3)]
    s=metric_validity_summary(rs)['by_model']['m']['O2-F1']['reliability_n3']
    assert s['all_three_pass_rate'] is None and s['resolved_only_rate_diagnostic']==1


def test_record_diagnostics_preserve_source_and_c_applicability():
    evaluator,_=_evaluator(_prediction())
    record=evaluator.evaluate_response_record(_case_input(),_metadata(),_ground_truth(),
        final_response='synthetic',efficiency=EfficiencyObservation(1,1,1,0,1),
        adjudications=_adjudications()).to_dict()
    before=deepcopy(record);d=record_diagnostics(record)
    assert record==before
    assert d['unresolved_dimension_count']==1
    assert d['unresolved_dimension_credits']=={'property_measure':1}
    assert d['c_applicable'] is True and d['legacy_qualitative_numeric_failure_ids']==[]
    record['result']['metrics']['o_f_consistency'].update(c_score=None,
        unavailable_reason='INDETERMINATE_OPERATIONALIZATION')
    d=record_diagnostics(record)
    assert d['c_applicable'] is False and d['c_equals_precision'] is None
    assert d['c_unavailable_reason']=='INDETERMINATE_OPERATIONALIZATION'
