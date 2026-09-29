"""Read-only completeness audit; optionally export a separate diagnostic JSON."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.path_migration import resolve_repository_path
from scripts.benchmark_runtime import require_benchmark_runtime

QUALITY_FIELDS = ('o_score', 'urs', 'resolved_o_compliance', 'finding_precision',
                  'core_finding_recall', 'finding_requirement_recall',
                  'adequate_core_complete', 'c_score', 'branch_alignment')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(output):
    output = Path(output).resolve()
    report = read(output / 'reports/experiment_report.json')
    index = read(output / 'evaluation/evaluation_index.json')
    evaluations = {row['run_id']: row for row in index['results']}
    with (output / 'reports/solver_efficiency.csv').open(encoding='utf-8', newline='') as stream:
        efficiency = {row['slot_id']: row for row in csv.DictReader(stream)}
    answers = []
    seen = set()
    for row in report['trial_rows']:
        if row.get('collection_status') != 'COMPLETED':
            continue
        slot = row['slot_id']
        if slot in seen:
            raise ValueError('Duplicate answered slot in report: ' + slot)
        seen.add(slot)
        result = evaluations.get(row.get('run_id'), {})
        evaluation = {}
        if result.get('evaluation_path'):
            evaluation = read(resolve_repository_path(result['evaluation_path'], repository_root=ROOT))
        held = []
        unanswered = []
        for request_id in result.get('active_request_ids', []):
            path = output / 'exchange/responses' / (request_id + '.json')
            if not path.exists():
                unanswered.append(request_id)
                continue
            answer = read(path)
            if answer.get('status') == 'PENDING':
                held.append(dict(request_id=request_id, reason=answer.get('reason')))
        complete = (row.get('evaluation_status') == 'SCORED' and row.get('status') == 'COMPLETED')
        answers.append(dict(slot_id=slot, run_id=row.get('run_id'), model=row['model'],
            case_id=row['case_id'], trial=row['trial'], quality_complete=complete,
            quality_metrics={field: row.get(field) for field in QUALITY_FIELDS},
            solver_efficiency=efficiency.get(slot),
            pending_type=result.get('pending_type'),
            pending_reason=evaluation.get('pending_record', {}).get('pending_reason'),
            unanswered_request_ids=unanswered, held_pending=held,
            evaluation_path=result.get('evaluation_path')))
    quality_count = sum(row['quality_complete'] for row in answers)
    efficiency_count = sum(bool(row['solver_efficiency']) for row in answers)
    primary_count = sum((row['solver_efficiency'] or {}).get('included') == 'True' for row in answers)
    return dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), output=str(output),
        status='COMPLETE' if answers and quality_count == efficiency_count == len(answers) else 'INCOMPLETE',
        answered_slot_count=len(answers), answered_case_count=len({row['case_id'] for row in answers}),
        quality_complete_count=quality_count, efficiency_disposition_count=efficiency_count,
        primary_efficiency_count=primary_count,
        pending_type_counts=dict(Counter(row['pending_type'] for row in answers if not row['quality_complete'])),
        scope='Existing completed answers only; no new judgments or API calls. Null quality values are not scores.',
        efficiency_scope='Original Claude solver records; reviewer/local evaluator costs are separate. Upstream queueing is not observable.',
        answers=answers)


def main():
    require_benchmark_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/claude_resume_eval')
    parser.add_argument('--report', type=Path, help='optional separate diagnostic export, never an evaluation artifact')
    args = parser.parse_args()
    if args.report:
        target = args.report.resolve()
        if target.drive.lower() == 'c:':
            parser.error('Diagnostic output on C: is disabled')
        if target.exists():
            parser.error('Diagnostic report already exists; choose a new filename')
    result = audit(args.output)
    if args.report:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'answers'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
