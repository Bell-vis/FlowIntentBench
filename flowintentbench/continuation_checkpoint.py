"""Commit typed continuation decisions before potentially long finalization."""
from pathlib import Path
from typing import Mapping

from .evaluator import EvaluationAdjudications, SemanticMatchResolution, UnitFrameResolution
from .evaluator_response_cache import ValidatedResponseCache, canonical, digest


def _adjudications(output):
    if isinstance(output, EvaluationAdjudications):
        return output
    if isinstance(output, SemanticMatchResolution):
        return EvaluationAdjudications(semantic_resolutions=(output,))
    if isinstance(output, UnitFrameResolution):
        return EvaluationAdjudications(unit_frame_resolutions=(output,))
    if isinstance(output, Mapping) and set(output).intersection({
        'novel_operationalization', 'novel_findings', 'novel_finding_roles',
        'semantic_resolutions', 'unit_frame_resolutions',
    }):
        return EvaluationAdjudications.from_dict(output)
    return None


def dispatch_with_checkpoint(registry, record, case_input, metadata, *, route, path, context):
    """Replay the same typed decision after interruption, including rejections.

    The complete blinded request, evaluator/handler identity and case hashes
    bind each checkpoint. A fresh database is used for every observation.
    Pending-record outputs are persisted by the existing orchestration path;
    failures, absent handlers and malformed adjudications are never cached.
    """
    namespace = digest(canonical({
        'implementation_sha256': digest(Path(__file__).read_text()),
        'context': context,
        'continuation_execution_manifest': registry.execution_manifest,
    }))
    cache = ValidatedResponseCache(Path(path), namespace)

    def parse(value):
        if set(value) != {'dispatch', 'adjudications'}:
            raise ValueError('invalid continuation checkpoint envelope')
        return {**value['dispatch'],
                'continuation_output': EvaluationAdjudications.from_dict(value['adjudications'])}

    def compute(validate):
        result = registry.dispatch(record, case_input, metadata)
        typed = _adjudications(result.get('continuation_output'))
        if typed is None:
            return result
        return validate({'dispatch': {k: v for k, v in result.items() if k != 'continuation_output'},
                         'adjudications': typed.to_dict()})

    # resolve() commits on return, before the caller enters finalization.
    return cache.resolve('continuation_adjudication', route, parse, compute, [])
