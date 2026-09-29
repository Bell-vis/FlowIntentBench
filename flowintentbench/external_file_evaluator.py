"""File-backed transport for the existing structured evaluator, without an API.

No scientific decisions are made here. Independent reviewers fill content-bound
response envelopes, and StructuredEvaluatorBackend validates their JSON using
the same parsers as the online transport. Missing answers always remain pending.
"""
from __future__ import annotations

import hashlib
import ast
import inspect
import json
import textwrap
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .evaluator import EvaluationPendingAdjudication, EvaluatorComponentIdentity
from .evaluator_backend import StructuredEvaluatorBackend, OpenAICompatibleEvaluatorBackend, _OUTPUT_SCHEMAS


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Atomic replacement prevents parallel independent reviewers from reading
    # a half-written request. Requests themselves are content addressed.
    import tempfile
    import os
    import time
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as handle:
        handle.write(text)
        temporary = handle.name
    try:
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers may briefly hold a handle without delete
                # sharing. Keep the old complete file until replacement works.
                if os.name != 'nt' or attempt == 6:
                    raise
                time.sleep(0.01 * 2 ** attempt)
    finally:
        Path(temporary).unlink(missing_ok=True)


@lru_cache(maxsize=1)
def operation_guidance():
    declarations = {}
    source = textwrap.dedent(inspect.getsource(OpenAICompatibleEvaluatorBackend._complete_json))
    for node in ast.parse(source).body[0].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"instructions", "schema_hints"}:
            declarations[node.targets[0].id] = ast.literal_eval(node.value)
    return declarations


class FileJudgmentTransport:
    def __init__(self, directory: str | Path, *, context: Mapping[str, Any]):
        self.directory = Path(directory)
        self.context = dict(context)
        self.used_response_ids: list[str] = []
        self.pending_request_ids: list[str] = []
        # Optional host-only dependency completion. It cannot consult the API
        # and must publish the same content-bound, validated response envelope.
        self.response_reuser = None
        # Set by the host replay for evidence-aware continuation requests.
        # Kept out of ``context`` so the public content-bound identity remains
        # the response digest plus the exact visible request payload.
        self.response_text: str | None = None

    @staticmethod
    def _operationalization_source_section(response: str) -> Mapping[str, Any] | None:
        headers = list(re.finditer(r'^ {0,3}(#{1,6})[ \t]+([^\n]+)$', response, re.MULTILINE))
        sections = [h for h in headers if h.group(2).strip().rstrip('#').strip().lower() == 'operationalization']
        if len(sections) != 1:
            return None
        header = sections[0]
        end = next((h.start() for h in headers if h.start() > header.start()
                    and len(h.group(1)) <= len(header.group(1))), len(response))
        excerpt = response[header.start():end]
        return {"evidence_status": "BOUND", "evidence_excerpt": excerpt,
                "evidence_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
                "evidence_char_start": header.start(), "evidence_char_end": end}

    def request(self, operation: str, payload: Mapping[str, Any], *,
                output_schema: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        # Old PENDING unit packets often predate source-span propagation. If
        # the exact answer contains one explicit Operationalization section,
        # bind that literal excerpt into a new request before hashing. This is
        # evidence transfer only: units, recipes and scientific conclusions
        # remain the reviewer's responsibility.
        if (operation == "adjudication" and self.response_text
                and payload.get("pending_type") == "unit_relationship"):
            unit_context = payload.get("unit_review_context") or {}
            if ("operationalization_source_section" not in unit_context
                    and unit_context.get("prediction_binding_status") == "BOUND"):
                section = self._operationalization_source_section(self.response_text)
                if section is not None:
                    payload = json.loads(json.dumps(payload, ensure_ascii=False))
                    payload["unit_review_context"]["operationalization_source_section"] = section
        # These are literal dictionaries in the existing online transport.
        # Read the declarations, never instantiate/call that transport. This
        # keeps the offline judge's scientific instructions exactly identical.
        declarations = operation_guidance()
        prompt = declarations.get("instructions", {}).get(operation) or payload.get("instruction")
        if operation in _OUTPUT_SCHEMAS and not prompt:
            raise ValueError("existing evaluator operation prompt could not be loaded")
        identity = {"schema_version": "external-judgment-v1", "context": self.context,
                    "operation": operation, "input": dict(payload), "prompt": prompt,
                    "visible_input_sha256": digest(payload), "output_schema": output_schema}
        request_id = digest(identity)
        request = {**identity, "request_id": request_id, "status": "PENDING",
                   "contract_shape": declarations.get("schema_hints", {}).get(operation),
                   "instructions_source": "flowintentbench/evaluator_backend.py:OpenAICompatibleEvaluatorBackend._complete_json",
                   "review_rule": "Apply the existing operation instructions. Extraction and eligibility are GT-blind; never infer O from results. Incorrect but relevant and verifiable findings remain eligible. Semantic judgment is not string equality. Do not invent missing evidence.",
                   "response_template": {"request_id": request_id, "reviewer_id": "INDEPENDENT_REVIEWER_ID",
                                         "status": "RESOLVED", "response": None}}
        request_path = self.directory / "requests" / f"{request_id}.json"
        if request_path.exists():
            if digest(json.loads(request_path.read_text())) != digest(request):
                raise ValueError("content-addressed external request was modified")
        else:
            write_json(request_path, request)
        answer_path = self.directory / "responses" / f"{request_id}.json"
        if not answer_path.exists() and self.response_reuser is not None:
            self.response_reuser(request)
        if answer_path.is_file():
            answer = json.loads(answer_path.read_text())
            if answer.get("request_id") != request_id:
                raise ValueError("external judgment request_id mismatch")
            if answer.get("status") == "RESOLVED":
                if not str(answer.get("reviewer_id", "")).strip() or answer.get("reviewer_id") == "INDEPENDENT_REVIEWER_ID":
                    raise ValueError("resolved judgment requires reviewer provenance")
                if not isinstance(answer.get("response"), Mapping):
                    raise ValueError("resolved judgment response must be a JSON object")
                self.used_response_ids.append(request_id)
                return answer["response"]
            if answer.get("status") != "PENDING":
                raise ValueError("external judgment status must be PENDING or RESOLVED")
        if request_id not in self.pending_request_ids:
            self.pending_request_ids.append(request_id)
        raise EvaluationPendingAdjudication(
            f"Independent {operation} judgment is unavailable: {request_path}",
            pending_type=f"external_{operation}",
            continuation_context={"external_request_id": request_id,
                                  "external_request_path": str(request_path),
                                  "required_next_action": "FILL_RESPONSE_AND_REPLAY"})

    def __call__(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        schema = json.loads(json.dumps(_OUTPUT_SCHEMAS[operation]))
        if operation == "extraction":
            dimensions = list(payload["principal_operationalization_dimensions"])
            decisions = schema["properties"]["operationalization"]["properties"]["decisions"]
            decisions["minItems"] = decisions["maxItems"] = len(dimensions)
            if dimensions:
                decisions["items"]["properties"]["dimension"]["enum"] = dimensions
        return self.request(operation, payload, output_schema=schema)


def build_file_backend(transport: FileJudgmentTransport) -> StructuredEvaluatorBackend:
    return FileStructuredEvaluatorBackend(
        completion=transport,
        identity=EvaluatorComponentIdentity(implementation_id="flowintentbench-file-judgments",
                                             version="1", provider="external-independent-review",
                                             model="file-replay-no-api"),
        max_format_repairs=0,
    )


class FileStructuredEvaluatorBackend(StructuredEvaluatorBackend):
    def manifest(self, **kwargs):
        from .quality_efficiency import QUALITY_POLICY_SHA256
        manifest = super().manifest(**kwargs)
        return replace(manifest, deterministic_contract_hashes={
            **manifest.deterministic_contract_hashes,
            "quality_efficiency_policy_sha256": QUALITY_POLICY_SHA256,
            # Bind the actual extraction/semantic routing and wire prompts,
            # including evidence propagation, beyond the selected formula hashes.
            "evaluator_source_sha256": hashlib.sha256(Path(__file__).with_name("evaluator.py").read_bytes()).hexdigest(),
            "structured_backend_source_sha256": hashlib.sha256(Path(__file__).with_name("evaluator_backend.py").read_bytes()).hexdigest(),
            "file_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "file_evaluation_workflow_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "scripts" / "run_expansion_file_evaluation.py").read_bytes()).hexdigest(),
            "local_dependency_completion_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "scripts" / "replay_judgment_completion.py").read_bytes()).hexdigest(),
            "expansion_case_compiler_sha256": hashlib.sha256(Path(__file__).with_name("expansion_evaluation.py").read_bytes()).hexdigest(),
            "expansion_recipe_materializer_sha256": hashlib.sha256(Path(__file__).with_name("expansion_recipe_materializer.py").read_bytes()).hexdigest(),
            "recipe_binding_validation_sha256": hashlib.sha256(Path(__file__).with_name("recipe_binding_validation.py").read_bytes()).hexdigest(),
            "supplemental_association_evidence_sha256": hashlib.sha256(Path(__file__).with_name("supplemental_association_evidence.py").read_bytes()).hexdigest(),
            "runtime_construction_recipes_sha256": hashlib.sha256(Path(__file__).with_name("runtime_construction_recipes.py").read_bytes()).hexdigest(),
        })


def adjudication_schema():
    """Transport shape only; original EvaluationAdjudications validates meaning."""
    from .ground_truth import OperationalizationBundle
    status = {"type": "string", "enum": ["UNRESOLVED", "ACCEPTED", "REJECTED"]}
    bundle_schema = OperationalizationBundle.model_json_schema()
    definitions = bundle_schema.pop("$defs", {})
    return {"type": "object", "additionalProperties": False, "$defs": definitions,
        "required": ["novel_operationalization", "novel_findings", "novel_finding_roles", "semantic_resolutions", "unit_frame_resolutions"],
        "properties": {
            "novel_operationalization": {"anyOf": [{"type": "null"}, {"type": "object",
                "required": ["status", "operationalization"], "additionalProperties": False,
                "properties": {"status": status, "operationalization": bundle_schema,
                    "adjudicated_materialization_id": {"type": ["string", "null"]},
                    "adjudicated_g_of_o_sha256": {"type": ["string", "null"]}}}]},
            "novel_findings": {"type": "array", "items": {"type": "object",
                "required": ["branch_id", "prediction_id", "status"], "additionalProperties": False,
                "properties": {"branch_id": {"type": "string"}, "prediction_id": {"type": "string"}, "status": status}}},
            "novel_finding_roles": {"type": "array", "items": {"type": "object",
                "required": ["branch_id", "prediction_id", "role"], "additionalProperties": False,
                "properties": {key: {"type": "string"} for key in ("branch_id", "prediction_id", "role")}}},
            "semantic_resolutions": {"type": "array", "items": {"type": "object",
                "required": ["request_id", "request", "result"], "additionalProperties": False,
                "properties": {"request_id": {"type": "string"}, "request": {"type": "object"},
                               "result": {"type": "string", "enum": ["MATCH", "NO_MATCH"]}}}},
            "unit_frame_resolutions": {"type": "array", "items": {"type": "object",
                "required": ["request_id", "request", "status"],
                "properties": {"request_id": {"type": "string"}, "request": {"type": "object"},
                               "status": {"type": "string", "enum": ["RESOLVED", "NO_MATCH"]}}}}}}
