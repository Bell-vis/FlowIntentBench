"""Local completion of file dependencies, without scientific inference or APIs.

Reuse only identical reviewer-visible tasks. Candidate O completion is literal
transcription of already extracted decisions, never an acceptance judgment.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import re
from pathlib import Path
import time

from flowintentbench.external_file_evaluator import digest
from scripts.carry_forward_file_judgments import visible_task_signature, reusable_verdict, POLICY
from scripts.collect_subagent_runs import file_hash


class ReplayJudgmentCompletion:
    def __init__(self, exchange):
        self.exchange = Path(exchange)
        self.index = None
        self.stats = dict(reused=0, candidates_transcribed=0, indexed_responses=0,
                          index_seconds=0.0, conflicts=0)

    def build_index(self, *, workers=16, progress=None):
        started = time.monotonic()
        if not 1 <= workers <= 32:
            raise ValueError('Historical index workers must be between 1 and 32')
        paths = sorted((self.exchange / 'responses').glob('*.json'))
        def validated_source(path):
            answer = json.loads(path.read_text(encoding='utf-8'))
            if answer.get('status') not in {'RESOLVED', 'PENDING'}:
                return None
            request_path = self.exchange / 'requests' / path.name
            try:
                request = json.loads(request_path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                return None
            signature = visible_task_signature(request)
            self.validate_source(request, answer)
            return signature, request_path, path

        # Read independent files concurrently; merge in filename order so the
        # complete conflict set and deterministic source selection are unchanged.
        index = {}
        indexed = 0
        reported = started
        if progress:
            progress(0, len(paths))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='history-index') as pool:
            for count, source in enumerate(pool.map(validated_source, paths), 1):
                if source is not None:
                    signature, request_path, path = source
                    index.setdefault(signature, []).append((request_path, path))
                    indexed += 1
                now = time.monotonic()
                if progress and (now - reported >= 1 or count == len(paths)):
                    progress(count, len(paths))
                    reported = now
        self.index = index
        self.stats['indexed_responses'] = indexed
        self.stats['index_seconds'] = time.monotonic() - started

    def save_index(self, path, *, workers=16, progress=None):
        """Snapshot historical sources once per exclusive runner invocation."""
        from flowintentbench.external_file_evaluator import write_json
        if self.index is None:
            self.build_index(workers=workers, progress=progress)
        body = dict(exchange=str(self.exchange.resolve()), source_sha256=file_hash(Path(__file__)),
            index={signature:[a.name for _, a in sources] for signature,sources in self.index.items()},
            stats=self.stats)
        body['sha256'] = digest(body)
        write_json(Path(path), body)

    def load_index(self, path):
        body = json.loads(Path(path).read_text())
        if (body.get('sha256') != digest({k:v for k,v in body.items() if k != 'sha256'})
                or body['exchange'] != str(self.exchange.resolve())
                or body['source_sha256'] != file_hash(Path(__file__))):
            raise ValueError('Historical reuse index binding mismatch')
        self.index = {}
        for signature, names in body['index'].items():
            if not re.fullmatch('[0-9a-f]{64}', signature) or any(
                    not re.fullmatch('[0-9a-f]{64}\\.json', name) for name in names):
                raise ValueError('Invalid historical reuse index path')
            self.index[signature] = [(self.exchange/'requests'/name, self.exchange/'responses'/name) for name in names]
        self.stats['indexed_responses'] = body['stats']['indexed_responses']

    @staticmethod
    def validate_source(request, answer):
        reusable_verdict(request, answer)

    def __call__(self, request):
        from scripts.run_codex_file_judgments import _exclusive_json
        target = self.exchange / 'responses' / (request['request_id'] + '.json')
        if target.exists():
            return
        signature = visible_task_signature(request)
        if self.index is None:
            self.build_index()
        sources = []
        for request_path, answer_path in self.index.get(signature, []):
            # Revalidate the actual bytes at use, not a stale cached decision.
            source = json.loads(request_path.read_text())
            answer = json.loads(answer_path.read_text())
            if visible_task_signature(source) != signature:
                raise ValueError('Reuse source changed after indexing')
            if answer.get('status') not in {'RESOLVED', 'PENDING'}:
                continue
            self.validate_source(source, answer)
            if source['context']['evaluation_manifest_sha256'] != request['context']['evaluation_manifest_sha256']:
                sources.append((source, answer, request_path, answer_path))
        if sources:
            if len({digest(reusable_verdict(source, answer)) for source, answer, _, _ in sources}) != 1:
                self.stats['conflicts'] += 1
                return  # A conflict is never permission to choose a convenient verdict.
            source, answer, source_path, answer_path = sources[0]
            provenance = dict(policy=POLICY, visible_task_signature=signature,
                source_request_id=source['request_id'], source_request_sha256=file_hash(source_path),
                source_response_sha256=file_hash(answer_path),
                source_evaluation_manifest_sha256=source['context']['evaluation_manifest_sha256'],
                target_evaluation_manifest_sha256=request['context']['evaluation_manifest_sha256'],
                target_request_sha256=file_hash(self.exchange/'requests'/(request['request_id']+'.json')))
            envelope = dict(request_id=request['request_id'], reviewer_id=answer['reviewer_id'],
                status=answer['status'], response=answer['response'], carry_forward=provenance)
            for key in ('provenance', 'reason'):
                if key in answer:
                    envelope[key] = deepcopy(answer[key])
            try:
                _exclusive_json(target, envelope)
            except FileExistsError:
                return
            _exclusive_json(self.exchange/'carry_forward'/(request['request_id']+'.json'), dict(
                **provenance, target_request_id=request['request_id'],
                target_response_sha256=file_hash(target), original_response=answer))
            self.stats['reused'] += 1
            return
        candidate = literal_candidate_response(request)
        if candidate is not None:
            try:
                _exclusive_json(target, candidate)
            except FileExistsError:
                return
            self.stats['candidates_transcribed'] += 1


def literal_candidate_response(request):
    if (request['operation'] != 'adjudication'
            or request['input'].get('pending_type') != 'novel_operationalization_candidate'):
        return None
    from flowintentbench.ground_truth import OperationalizationBundle
    from flowintentbench.evaluator import continuation_request_id
    from scripts.run_codex_file_judgments import validate_response
    data = request['input']
    context = data['continuation_context']
    supplied = context.get('continuation_request')
    if not supplied or set(supplied) != {'decisions'}:
        return None
    if context.get('continuation_request_id') != continuation_request_id('operationalization_completion', supplied):
        raise ValueError('Candidate completion identity mismatch')
    if context.get('operationalization_completion_request', supplied) != supplied:
        raise ValueError('Candidate completion packets disagree')
    decisions = supplied['decisions']
    if (not isinstance(decisions, list) or not decisions
            or any(not isinstance(d, dict) or set(d) != {'dimension', 'statement'}
                   or not isinstance(d['statement'], str) or not d['statement'].strip() for d in decisions)):
        return None
    bundle = OperationalizationBundle.model_validate(dict(
        operationalization_id='literal-o-' + digest(supplied), decisions=decisions, evidence_ids=[]))
    # Do not silently normalize whitespace or any scientific wording.
    if bundle.model_dump(mode='json')['decisions'] != decisions:
        return None
    response = deepcopy(data['accumulated_adjudications'])
    if response.get('novel_operationalization') is not None:
        return None
    response['novel_operationalization'] = dict(status='UNRESOLVED',
        operationalization=bundle.model_dump(mode='json'))
    validate_response(request, response)
    return dict(request_id=request['request_id'], reviewer_id='host:literal-o-transcription-v1',
        status='RESOLVED', response=response,
        reason='Exact transcription of complete extracted O decisions; scientific status remains UNRESOLVED.',
        provenance=dict(policy='literal-o-transcription-v1', source_sha256=file_hash(Path(__file__)),
            input_sha256=digest(supplied), scientific_judgment=False, api_calls=0,
            decisions_unchanged=True))
