import csv

from flowintentbench.external_file_evaluator import write_json
from scripts.audit_answered_claude_metrics import audit


def test_every_answer_has_metrics_and_blockers_without_counting_unanswered(tmp_path):
    report = tmp_path / 'reports'
    write_json(report / 'experiment_report.json', dict(trial_rows=[
        dict(slot_id='one', run_id='one', model='claude', case_id='case', trial=1,
             collection_status='COMPLETED', evaluation_status='PENDING', status='PENDING'),
        dict(slot_id='two', collection_status='NOT_STARTED')]))
    path = tmp_path / 'evaluation/one.json'
    write_json(path, dict(pending_record=dict(pending_reason='Unsupported numeric route')))
    write_json(tmp_path / 'evaluation/evaluation_index.json', dict(results=[dict(
        run_id='one', evaluation_path=str(path), pending_type='evaluator_coverage_gap',
        active_request_ids=['held', 'missing'])]))
    write_json(tmp_path / 'exchange/responses/held.json', dict(status='PENDING', reason='Missing units'))
    with (report / 'solver_efficiency.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['slot_id', 'included', 'total_tokens'])
        writer.writeheader()
        writer.writerow(dict(slot_id='one', included=True, total_tokens=100))
    result = audit(tmp_path)
    assert result['answered_slot_count'] == result['efficiency_disposition_count'] == 1
    assert result['quality_complete_count'] == 0 and result['status'] == 'INCOMPLETE'
    assert result['answers'][0]['pending_reason'] == 'Unsupported numeric route'
    assert result['answers'][0]['held_pending'][0]['reason'] == 'Missing units'
    assert result['answers'][0]['unanswered_request_ids'] == ['missing']
    assert result['answers'][0]['solver_efficiency']['total_tokens'] == '100'
