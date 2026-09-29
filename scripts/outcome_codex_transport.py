"""Answer-only rubric reviews through Codex CLI's existing ChatGPT login."""
import hashlib
import json
import os
from pathlib import Path

from flowintentbench.external_file_evaluator import write_json
from scripts.codex_console_transport import run_codex, command, parse_events


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cli_prompt(prompt, schema, budget):
    return ('Evaluate only the supplied answer and frozen reference evidence. No tools, files, '
            'network requests, code execution, recomputation, or solving the task. '
            'Return only the requested JSON judgment. Cite exact answer lines for EVERY extracted '
            'dimension. Keep your response concise; target at most ' + str(budget) +
            ' output tokens. Unknown scientific truth must remain unknown.\n\n' +
            'OUTPUT_SCHEMA\n' + json.dumps(schema, ensure_ascii=False) + '\n\n' + prompt)


def audit_cli(receipt, payload):
    """Revalidate the actual CLI transcript, command and exact stdin on replay."""
    folder = Path(receipt['cli_directory'])
    for name, checksum in receipt['cli_artifact_sha256'].items():
        if file_sha(folder/name) != checksum:
            raise ValueError('CLI artifact checksum mismatch: ' + name)
    raw = json.loads((folder/'receipt.json').read_text())
    if not raw.get('completed') or raw.get('returncode') != 0:
        raise ValueError('CLI turn did not complete')
    expected = cli_prompt(payload['input'][0]['content'], payload['schema'], payload['max_output_tokens'])
    if (folder/'prompt.txt').read_text() != expected:
        raise ValueError('CLI prompt does not match rubric request')
    schema_path = folder/'output_schema.json' if raw.get('structured_output_enforced') else None
    expected_command = command(payload['model'], payload['workdir'], schema_path,
                               reasoning_effort=payload['reasoning']['effort'], no_tools=True)
    if json.loads((folder/'command.json').read_text()) != expected_command:
        raise ValueError('CLI command/model/effort differs from declared review')
    events = parse_events(folder/'events.jsonl')
    if not events['completed'] or events['error'] or events['final_text'] != receipt['final_text']:
        raise ValueError('CLI response is not bound to completed transcript')
    if (events.get('usage') or {}) != receipt['usage']:
        raise ValueError('CLI usage differs from transcript')
    for line in (folder/'events.jsonl').read_text().split('\n'):
        if not line.strip():continue
        event = json.loads(line)
        if event.get('type','').startswith('item.'):
            if event.get('item',{}).get('type') not in {'agent_message','reasoning'}:
                raise ValueError('Tool or other unauthorized item in answer-only CLI review')
    return True


class CodexOutcomeReviewer:
    def __init__(self):
        home = Path(os.environ.get('CODEX_HOME', Path.home()/'.codex'))
        auth = json.loads((home/'auth.json').read_text())
        if auth.get('auth_mode') != 'chatgpt' or not auth.get('tokens'):
            raise ValueError('Codex reviewer requires an existing ChatGPT login, not an API key')

    def __call__(self, prompt, model, workdir, output_dir, timeout_seconds=300,
                 output_schema=None, *, reasoning_effort='medium', max_output_tokens=6000,
                 structured_output=False):
        audit = Path(output_dir).resolve()
        # Isolated empty working directory: no project, answers or raw arrays.
        work = audit/'empty_workdir'
        work.mkdir(parents=True, exist_ok=True)
        payload = {'model':model, 'reasoning':{'effort':reasoning_effort},
                   'input':[{'role':'user','content':prompt}], 'schema':output_schema,
                   'max_output_tokens':max_output_tokens, 'workdir':str(work),
                   'budget_enforcement':'PROMPT_TARGET_ONLY; CLI has no exposed hard output cap'}
        if structured_output:
            payload['text']={'format':{'type':'json_schema','name':'answer_evaluation',
                                      'schema':output_schema,'strict':True}}
        write_json(audit/'request.json', payload)
        result = run_codex(cli_prompt(prompt,output_schema,max_output_tokens),model,work,audit/'cli',
                           timeout_seconds=timeout_seconds,output_schema=output_schema,
                           reasoning_effort=reasoning_effort,no_tools=True)
        receipt = dict(result, usage=result.get('usage') or {}, response_model=model,
                       model_identity_source='EXPLICIT_CLI_MODEL_ARGUMENT', cli_directory=str(audit/'cli'),
                       cli_artifact_sha256={n:file_sha(audit/'cli'/n) for n in
                           ('receipt.json','events.jsonl','prompt.txt','command.json','output_schema.json')},
                       output_budget_enforcement=payload['budget_enforcement'])
        if receipt['completed']:
            try:audit_cli(receipt,payload)
            except Exception as exc:receipt.update(completed=False,error=str(exc))
        write_json(audit/'receipt.json', receipt)
        return receipt
