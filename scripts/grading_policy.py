"""Versioned reviewer execution policy; never changes solver budgets or scores."""
from flowintentbench.external_file_evaluator import digest

POLICY = {
    'version': 'case-scoped-medium-review-v7',
    'text_effort': 'medium',
    'eligibility_effort': 'medium',
    'extraction_effort': 'medium',
    'scientific_effort': 'medium',
    'batch_size': 4,
    'batch_format_repairs': 1,
    'dependency_refresh_min_interval_seconds': 2,
    'report_min_interval_seconds': 30,
    'batch_max_prompt_bytes': 48000,
    'batch_roles': ['semantic_match'],
    'scientific_batch_pending_types': ['gt_outside_finding'],
    'scientific_batch_size': 4,
    'scientific_batch_timeout_seconds': 900,
    'scientific_evidence_scope': 'Same answer, Effective-O branch, public inputs and materialization; independent verdict for each original request',
    'legacy_prompt_roles': ['eligibility'],
    'reuse': 'Existing exact content-bound responses remain authoritative; provenance retains policy version',
}
POLICY_SHA256 = digest(POLICY)


def text_only(request):
    return request['operation'] in {'extraction', 'eligibility', 'semantic_match'}


def effort(request):
    if request['operation'] in {'eligibility', 'extraction'}:
        return POLICY[request['operation'] + '_effort']
    return POLICY['text_effort'] if text_only(request) else POLICY['scientific_effort']


def batch_key(request):
    scientific = scientific_batch(request)
    operation = request['operation']
    pending_type = request.get('input', {}).get('pending_type')
    batch_role = ('unit_relationship' if operation == 'adjudication'
                  and pending_type == 'unit_relationship' else operation)
    if batch_role not in POLICY['batch_roles'] and not scientific:
        return None
    context = request.get('context', {})
    # Absence of an authenticated answer/case scope is never permission to pool.
    if not all(context.get(k) for k in ('case_id', 'response_sha256', 'evaluation_manifest_sha256')):
        return None
    data = request['input']
    if scientific:
        continuation = data.get('continuation_context', {})
        evidence = data.get('finding_review_context', {})
        if (not continuation.get('branch_id') or not evidence.get('raw_data')
                or evidence.get('prediction_binding_status') != 'BOUND'):
            return None
        return digest({'context': context, 'operation': request['operation'],
            'prompt': request['prompt'], 'schema': request['output_schema'],
            'scope': {k: v for k, v in continuation.items()
                      if k not in ('continuation_request', 'continuation_request_id')},
            'evidence': {k: v for k, v in evidence.items() if k != 'prediction'}})
    if POLICY.get('batch_answer_scope'):
        # Every item retains its complete input; only transport is combined.
        # Never combine answers, operations, instructions or response schemas.
        return digest({'context': context, 'operation': request['operation'],
                       'prompt': request['prompt'], 'schema': request['output_schema']})
    return digest({'context': context, 'operation': request['operation'],
                   'prompt': request['prompt'], 'schema': request['output_schema'],
                   'scope': {k: data.get(k) for k in (
                       'scientific_question', 'case_context', 'finding_goal',
                   'purpose', 'dimension', 'finding_category', 'verification_mode')}})


def scientific_batch(request):
    return (request['operation'] == 'adjudication' and request['input'].get('pending_type')
            in POLICY['scientific_batch_pending_types'])


TEXT_INSTRUCTIONS = (
    'This is a text-only review. Return the requested JSON directly with a concise '
    'evidence-based reason. Use only the supplied packet. Do not call tools, write '
    'scripts, reproduce input files or calculate dataset statistics. The host '
    'preserves the input, hashes, response and reviewer provenance automatically. '
    'Set evidence_files to []. Eligibility concerns interpretability, relevance '
    'and in-principle verifiability, not factual correctness. Semantic matching '
    'does not authorize independent numerical verification. Preserve uncertainty '
    'using the published contract; never invent missing evidence. '
)

STRUCTURED_INSTRUCTIONS = (
    'Your role is to convert supplied answer evidence into the requested typed JSON. '
    'No tools or code execution are available. The host computes materializations, '
    'numeric comparisons and final scores. Extract explicit methods, quantities, '
    'units and exact source excerpts; apply only the supplied semantic contracts. '
    'Do not solve the original question, invent source declarations, simulate '
    'computations, or claim numerical correctness from plausible wording or '
    'agreement with a reference value. For adjudication use only supplied host '
    'materialization evidence where it actually establishes the requested claim. '
    'Use a two-step closure rule: first check whether the supplied answer text, '
    'typed contract, metadata, or host-provided evidence directly establishes '
    'support or contradiction. If it does, return the corresponding terminal '
    'typed judgment, including NO_MATCH or REJECTED where the schema permits it. '
    'Return PENDING only when a fact indispensable to that judgment is actually '
    'absent. Do not return PENDING merely because the answer is not a verbatim '
    'reference, because a qualitative mismatch is inconvenient, or because '
    'numerical execution would be preferable when supplied evidence already '
    'establishes the result. If new numerical execution or missing evidence is '
    'genuinely required, return PENDING with response null and the precise '
    'missing computation/evidence. '
    'For unit_relationship requests, judge only the supplied prediction/reference '
    'pair. If a scalar or vector reference is compared with a multi-component or '
    'mixed-unit prediction and the packet supplies no explicit component projection, '
    'return a terminal RESOLVED unit_frame_resolution with status NO_MATCH; do not '
    'coerce the whole tuple to one reference unit, and do not reopen the same '
    'relationship merely to search for a projection. '
    'For semantic_match requests, compare the supplied predicted and reference '
    'statements as an evidence entailment decision. If they have different '
    'subjects, quantities, units, directions, or explicit numerical targets, '
    'and the packet provides no bridge that makes them equivalent, return a '
    'terminal NO_MATCH semantic resolution. Do not use PENDING as a cautious '
    'substitute for an explicit contradiction or unrelated reference. A match '
    'requires the prediction to entail the requested reference claim; lexical '
    'overlap alone is not a match. For eligibility requests, return the terminal '
    'negative verdict when the supplied finding lacks a locatable method, scope, '
    'or in-principle verification path; reserve PENDING for a genuinely missing '
    'field needed to decide that contract. '
    'For adjudication continuations, copy the exact supplied continuation request_id '
    'and target request object; resolve only that one request and do not echo, add, '
    'remove, or rewrite resolutions belonging to another request. Set evidence_files '
    'to []. Return concise JSON only. '
)
