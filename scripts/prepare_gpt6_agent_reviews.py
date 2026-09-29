"""Freeze failed answer packets for explicitly selected GPT-6 review agents."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from scripts.evaluate_answered_outcomes import sha_file

SOURCE = ROOT / 'outputs/claude_outcomes_v2_responses_177'
OUTPUT = ROOT / 'outputs/claude_outcomes_gpt6_agent_reviews'


def main():
    selection = json.loads((SOURCE / 'selection.json').read_text(encoding='utf-8'))
    pending = []
    for row in selection['answers']:
        result = json.loads((SOURCE / 'cases' / (row['run_id'] + '.json')).read_text(encoding='utf-8'))
        if result['status'] == 'SCORED':
            continue
        request_path = SOURCE / 'requests' / (result['request_id'] + '.json')
        request = json.loads(request_path.read_text(encoding='utf-8'))
        packet = {**row, 'source_request_path': str(request_path), 'source_request_sha256': sha_file(request_path),
                  'reviewer_model': 'gpt-6-astra', 'reviewer_source': 'EXPLICIT_GPT6_AGENT',
                  'prompt': request['prompt'], 'schema': request['schema'], 'rubric': request['rubric']}
        packet_path = OUTPUT / 'packets' / (row['run_id'] + '.json')
        write_json(packet_path, packet)
        pending.append({'run_id': row['run_id'], 'packet_path': str(packet_path),
                        'judgment_path': str(OUTPUT / 'judgments' / (row['run_id'] + '.json'))})
    for i in range(3):
        write_json(OUTPUT / f'group_{i + 1}.json', pending[i::3])
    write_json(OUTPUT / 'selection.json', {'source': str(SOURCE), 'source_selection_sha256': sha_file(SOURCE / 'selection.json'),
                                        'reviewer_model': 'gpt-6-astra', 'reviewer_source': 'EXPLICIT_GPT6_AGENT',
                                        'pending': pending})
    print(json.dumps({'pending': len(pending), 'groups': [len(pending[i::3]) for i in range(3)]}))


if __name__ == '__main__':
    main()
