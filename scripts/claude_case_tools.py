"""Minimal stdio MCP server: one assigned case, journaled Python, final answer.

No filesystem browsing, shell, web, other agents or evaluation tool is exposed.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PYTHON_OUTPUT_LIMITS = (8000, 4000)


def _preview(value, limit):
    value = str(value or '')
    if len(value) <= limit:
        return value
    marker = '\n...[tool output truncated; full output is in execution_journal.jsonl]...\n'
    available = max(0, limit - len(marker))
    head = available // 2
    return value[:head] + marker + value[-(available - head):]


TOOLS = [
    dict(name='read_case', description='Read the assigned scientific question and declared data paths.',
         inputSchema={'type': 'object', 'properties': {}, 'additionalProperties': False}),
    dict(name='python', description='Execute analysis Python in a fresh interpreter, with assigned data read-only. '
         'Relative files persist in scratch/. At most 48 executions and 720 seconds per answer. '
         'Use this tool directly; do not invoke any execution helper.',
         inputSchema={'type': 'object', 'properties': {'code': {'type': 'string'}},
                      'required': ['code'], 'additionalProperties': False}),
    dict(name='submit_answer', description='Submit the final answer with nonempty ## Operationalization and ## Finding sections.',
         inputSchema={'type': 'object', 'properties': {'answer': {'type': 'string'}},
                      'required': ['answer'], 'additionalProperties': False}),
]


class CaseTools:
    def __init__(self, work):
        self.work = Path(work).resolve()
        self.scope = json.loads((self.work / 'blind_scope.json').read_text())
        self.case_read = False
        self.python_hashes = set()

    def call(self, name, arguments):
        budget = json.loads((self.work / 'execution_budget.json').read_text())
        if time.time() >= budget['deadline_epoch'] or (self.work / 'finished.json').exists():
            raise ValueError('Case deadline exceeded or already finished')
        if name == 'read_case' and arguments == {}:
            if self.case_read:
                return json.dumps({'error': 'case_already_read',
                                   'instruction': 'Reuse the case content already in context; do not call read_case again.'})
            self.case_read = True
            return Path(self.scope['case_input']).read_text()
        if name == 'python' and set(arguments) == {'code'} and isinstance(arguments['code'], str):
            if (self.work / 'answer.md').exists():
                raise ValueError('Answer already submitted')
            code_hash = hashlib.sha256(arguments['code'].encode()).hexdigest()
            if code_hash in self.python_hashes:
                return json.dumps({'returncode': 1, 'error': 'identical_code_repeated',
                                   'instruction': 'Do not rerun identical code; reuse the prior output or change the analysis.'})
            self.python_hashes.add(code_hash)
            source = self.work / 'analysis.py'
            source.write_text(arguments['code'])
            result = subprocess.run([sys.executable, str(self.work / 'run_python.py'), str(source)],
                                    cwd=self.work, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                    timeout=max(1, budget['deadline_epoch']-time.time()+5))
            # Full execution evidence remains in the journal. Return compact
            # previews so a verbose or repeatedly failing tool cannot consume
            # the whole conversation context.
            stderr = result.stderr
            environment_failure = any(marker in stderr for marker in (
                'Landlock ABI', 'requires Linux x86_64 Landlock',
                'EXECUTION_SUPPORT_UNAVAILABLE'))
            return json.dumps(dict(returncode=result.returncode,
                                   stdout=_preview(result.stdout, PYTHON_OUTPUT_LIMITS[0]),
                                   stderr=_preview(stderr, PYTHON_OUTPUT_LIMITS[1]),
                                   output_truncated=len(result.stdout)>PYTHON_OUTPUT_LIMITS[0]
                                   or len(stderr)>PYTHON_OUTPUT_LIMITS[1],
                                   environment_failure=environment_failure,
                                   instruction=('The analysis sandbox is unavailable in this environment; '
                                                'do not repeat Python calls. Submit only if you can answer '
                                                'from already verified evidence.') if environment_failure else None))
        if name == 'submit_answer' and set(arguments) == {'answer'} and isinstance(arguments['answer'], str):
            answer = arguments['answer']
            if not answer.startswith('## Operationalization\n') or '\n## Finding\n' not in answer:
                raise ValueError('Required sections: ## Operationalization then ## Finding')
            first, second = answer[len('## Operationalization\n'):].split('\n## Finding\n', 1)
            if not first.strip() or not second.strip():
                raise ValueError('Both sections must be nonempty')
            with (self.work / 'answer.md').open('x') as stream:
                stream.write(answer)
            return json.dumps({'submitted': True, 'sha256': hashlib.sha256(answer.encode()).hexdigest()})
        raise ValueError('Tool or arguments outside the assigned case contract')


def main():
    # The CLI needs credentials, but neither this server nor its analysis children do.
    clean = {k: os.environ[k] for k in ('PATH', 'LANG', 'LD_LIBRARY_PATH') if k in os.environ}
    os.environ.clear()
    os.environ.update(clean)
    server = CaseTools(sys.argv[1])
    for line in sys.stdin:
        request = json.loads(line)
        if 'id' not in request:
            continue
        method = request['method']
        try:
            if method == 'initialize':
                result = {'protocolVersion': request['params']['protocolVersion'],
                          'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'flowintentbench-case', 'version': '1.0'}}
            elif method == 'tools/list':
                result = {'tools': TOOLS}
            elif method == 'tools/call':
                params = request['params']
                try:
                    value = server.call(params['name'], params.get('arguments', {}))
                    result = {'content': [{'type': 'text', 'text': value}]}
                except Exception as exc:
                    result = {'isError': True, 'content': [{'type': 'text', 'text': f'{type(exc).__name__}: {exc}'}]}
            elif method == 'ping':
                result = {}
            else:
                raise ValueError('Unsupported MCP method')
            reply = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
        except Exception:
            reply = {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32601, 'message': 'Unsupported request'}}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    main()
