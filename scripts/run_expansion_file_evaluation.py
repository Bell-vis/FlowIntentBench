#!/usr/bin/env python3
"""Prepare and replay development evaluations using independent JSON judgments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.evaluator import (build_case_evaluator, EvaluationAdjudications,
    EvaluationPendingAdjudication, PendingCaseEvaluationRecord, EfficiencyObservation,
    evaluation_manifest_digest, continuation_request_id)
from flowintentbench.expansion_evaluation import (compile_expansion_evaluation,
    load_development_case, read_json, file_sha256)
from flowintentbench.external_file_evaluator import (FileJudgmentTransport,
    build_file_backend, digest, write_json, adjudication_schema)
from flowintentbench.model_runner import RunRecord
from flowintentbench.expansion_recipe_materializer import build_expansion_recipe_materializer
from flowintentbench.execution_evidence import run_bound_execution_trajectory
from flowintentbench.loader import CaseLoader


_ADJUDICATION_ROLE_FIELDS = {
    "novel_operationalization_candidate": ("novel_operationalization",),
    "novel_operationalization_scientific_adjudication": ("novel_operationalization",),
    "gt_outside_finding": ("novel_findings", "novel_finding_roles"),
    "unit_relationship": ("unit_frame_resolutions",),
    "semantic_uncertain": ("semantic_resolutions",),
}


def adjudication_visible_state(adjudications, pending_type, continuation_context):
    """Project prior state for one role; hidden resolutions stay server-side."""
    prior = adjudications.to_dict()
    # Semantic comparison is expressly reference-aware. Preserve its existing
    # input contract while preventing other roles from inheriting its history.
    if pending_type == "semantic_uncertain":
        return deepcopy(prior)
    visible = EvaluationAdjudications().to_dict()
    for field in _ADJUDICATION_ROLE_FIELDS.get(pending_type, ()):
        visible[field] = deepcopy(prior[field])
    if "novel_operationalization" in _ADJUDICATION_ROLE_FIELDS.get(pending_type, ()):
        novel = visible["novel_operationalization"]
        if novel is not None and "branch" in novel:
            # Archived branch-shaped state can contain findings. Only its O
            # belongs in the O reviewer view, even for legacy continuations.
            branch = novel.pop("branch")
            novel.setdefault("operationalization", branch["operationalization"])
    if pending_type == "gt_outside_finding":
        target = continuation_context.get("continuation_request", {})
        for field in _ADJUDICATION_ROLE_FIELDS[pending_type]:
            visible[field] = [item for item in visible[field]
                              if all(item[key] == target.get(key)
                                     for key in ("branch_id", "prediction_id"))]
    if pending_type == "unit_relationship":
        visible["unit_frame_resolutions"] = [
            item for item in visible["unit_frame_resolutions"]
            if item["request_id"] == continuation_context.get("continuation_request_id")]
    return visible


def _validate_adjudication_reply_shape(value, schema=None, root_schema=None):
    """Validate the published transport schema without an optional dependency."""
    if schema is None:
        schema = adjudication_schema()
    if root_schema is None:
        root_schema = schema
    if "$ref" in schema:
        resolved = root_schema
        for key in schema["$ref"].removeprefix("#/").split("/"):
            resolved = resolved[key]
        return _validate_adjudication_reply_shape(value, resolved, root_schema)
    if "anyOf" in schema:
        for alternative in schema["anyOf"]:
            try:
                _validate_adjudication_reply_shape(value, alternative, root_schema)
                return
            except ValueError:
                pass
        raise ValueError("adjudication reply does not match the output schema")
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {"null": value is None, "object": isinstance(value, Mapping),
               "array": isinstance(value, list), "string": isinstance(value, str)}
    if types and not any(matches.get(kind, False) for kind in types):
        raise ValueError("adjudication reply has an invalid field type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("adjudication reply has an invalid enum value")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value):
            raise ValueError("adjudication reply is missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("adjudication reply contains forbidden fields")
        for key, child in properties.items():
            if key in value:
                _validate_adjudication_reply_shape(value[key], child, root_schema)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError("adjudication reply has too few items")
        for item in value:
            _validate_adjudication_reply_shape(item, schema.get("items", {}), root_schema)
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise ValueError("adjudication reply has an empty required string")


def merge_adjudication_reply(adjudications, pending_type, continuation_context, answer):
    """Validate role authority, then replace only owned state on the server."""
    _validate_adjudication_reply_shape(answer)
    # Reviewers may echo only the request fields needed for their explanation.
    # Bind abbreviated echoes to the authenticated continuation packet; extra
    # or conflicting fields remain integrity failures.
    normalized_answer = deepcopy(answer)
    visible = adjudication_visible_state(adjudications, pending_type, continuation_context)
    owned = _ADJUDICATION_ROLE_FIELDS.get(pending_type, ())
    for field in visible:
        if field not in owned and answer[field] != visible[field]:
            raise ValueError(f"adjudication reply attempted an out-of-role write: {field}")
    if "novel_operationalization" in owned:
        novel = answer["novel_operationalization"]
        if novel is not None:
            if pending_type == "novel_operationalization_candidate":
                if (novel["status"] != "UNRESOLVED"
                        or novel.get("adjudicated_materialization_id") is not None
                        or novel.get("adjudicated_g_of_o_sha256") is not None):
                    raise ValueError("O completion cannot adjudicate scientific acceptance")
            else:
                previous = visible["novel_operationalization"]
                echoed = deepcopy(novel["operationalization"])
                # The schema permits omission of an empty evidence list only.
                echoed.setdefault("evidence_ids", [])
                if previous is None or echoed != previous["operationalization"]:
                    raise ValueError("scientific adjudication cannot rewrite the candidate O")
                normalized_answer["novel_operationalization"]["operationalization"] = echoed
        elif visible["novel_operationalization"] is not None:
            raise ValueError("adjudication reply cannot remove the candidate O")
    if pending_type == "gt_outside_finding":
        target = continuation_context.get("continuation_request", {})
        for field in owned:
            items = answer[field]
            if len(items) > 1 or any(
                    any(item[key] != target.get(key) for key in ("branch_id", "prediction_id"))
                    for item in items):
                raise ValueError("finding adjudication can resolve only the supplied finding")
            if visible[field] and not items:
                raise ValueError("finding adjudication cannot remove a previous resolution")
    if pending_type in {"semantic_uncertain", "unit_relationship"}:
        field = owned[0]
        previous = visible[field]
        items = answer[field]
        if items[:len(previous)] != previous or len(items) > len(previous) + 1:
            raise ValueError("adjudication reply cannot rewrite previous resolutions")
        target = continuation_context.get(
            "semantic_match_request" if pending_type == "semantic_uncertain" else "unit_frame_request",
            continuation_context.get("continuation_request"))
        for item in items[len(previous):]:
            supplied = item["request"]
            if not isinstance(supplied, Mapping) or not isinstance(target, Mapping):
                raise ValueError("adjudication reply can resolve only the supplied request")
            supplied_for_compare = {
                key: value for key, value in supplied.items() if key != "pending_type"
            }
            if (item["request_id"] != continuation_context.get("continuation_request_id")
                    or ("pending_type" in supplied and supplied["pending_type"] != pending_type)
                    or not set(supplied_for_compare).issubset(target)
                    or any(supplied_for_compare[key] != target[key] for key in supplied_for_compare)):
                raise ValueError("adjudication reply can resolve only the supplied request")
        for item in normalized_answer[field][len(previous):]:
            item["request"] = deepcopy(target)
    parsed = EvaluationAdjudications.from_dict(normalized_answer)
    updates = {}
    for field in owned:
        value = getattr(parsed, field)
        if (field == "novel_operationalization" and value is not None
                and adjudications.novel_operationalization is not None
                and adjudications.novel_operationalization.branch is not None):
            value = replace(value, branch=adjudications.novel_operationalization.branch)
        elif field in {"novel_findings", "novel_finding_roles"}:
            value = {**getattr(adjudications, field), **value}
        elif field == "unit_frame_resolutions":
            prior_ids = {item.request_id for item in adjudications.unit_frame_resolutions}
            value = adjudications.unit_frame_resolutions + tuple(
                item for item in value if item.request_id not in prior_ids)
        elif field == "semantic_resolutions":
            value = adjudications.semantic_resolutions + value[len(visible[field]):]
        updates[field] = value
    # Do not reparse hidden state: its typed objects and all prior resolutions
    # survive exactly, independent of empty placeholders in the wire reply.
    return replace(adjudications, **updates)


def semantic_review_context(result, case_input, response):
    """Preserve O evidence across extraction's dimension boundaries for review."""
    context = {"data_context": case_input.flow_data.data_metadata.model_dump(mode="json")}
    request = result.continuation_context.get("semantic_match_request", {})
    prediction = result.extracted_prediction
    if request.get("purpose") == "operationalization" and prediction is not None:
        decisions = []
        for decision in prediction.operationalization.decisions:
            span = decision.evidence_span
            # The response is run-bound above; offsets can be resolved here,
            # unlike an isolated semantic request without its source text.
            evidence = span if isinstance(span, str) else None
            if isinstance(span, tuple) and 0 <= span[0] < span[1] <= len(response):
                evidence = response[span[0]:span[1]]
            decisions.append({"dimension": decision.dimension.value,
                              "status": decision.status.value,
                              "normalized_statement": decision.normalized_statement,
                              "evidence_text": evidence})
        context["extracted_operationalization_context"] = decisions
    return context


def _elided_quote_offsets(span, response):
    """Recover a literal paragraph excerpt between unique long quote anchors."""
    import re
    wrapped = re.fullmatch(
        r'[A-Za-z][A-Za-z /_-]{0,79}:[ \t]*(?:"([^"\n]+)"|\u201c([^\u201c\u201d\n]+)\u201d)',
        span.strip(),
    )
    if wrapped is None:
        return None
    quoted = wrapped.group(1) or wrapped.group(2)
    anchors = re.split(r'\s*(?:\.{3}|\u2026)\s*', quoted)
    if len(anchors) < 2 or any(len(part) < 20 or response.count(part) != 1 for part in anchors):
        return None
    positions = [response.index(part) for part in anchors]
    if any(right < left + len(part) for left, part, right
           in zip(positions, anchors, positions[1:])):
        return None
    start, end = positions[0], positions[-1] + len(anchors[-1])
    if end - start > 2000 or re.search(r'\n[ \t]*\n', response[start:end]):
        return None
    return start, end


def _bound_unit_evidence(span, response):
    """Expose only an exact excerpt from the already run-bound response."""
    if span is None:
        return {"evidence_text": None, "evidence_status": "MISSING", "source": None}
    if isinstance(span, str):
        start = response.find(span) if span else -1
        end = start + len(span)
        if start < 0:
            # Some historical extractions wrap a literal quote in a section
            # label. Strip only that wrapper, never paraphrases or Markdown.
            import re
            wrapped = re.fullmatch(
                r'(?:Finding[ \t]*:[ \t]*)?(?:"([^"\n]+)"|\u201c([^\u201c\u201d\n]+)\u201d)',
                span.strip(),
            )
            if wrapped:
                excerpt = wrapped.group(1) or wrapped.group(2)
                if response.count(excerpt) == 1:
                    start = response.index(excerpt)
                    end = start + len(excerpt)
            if start < 0:
                located = _elided_quote_offsets(span, response)
                if located is not None:
                    start, end = located
    elif (isinstance(span, tuple) and len(span) == 2
          and all(type(offset) is int for offset in span)):
        start, end = span
    else:
        start, end = -1, -1
    if not 0 <= start < end <= len(response):
        return {"evidence_text": None, "evidence_status": "UNLOCATABLE", "source": None}
    return {"evidence_text": response[start:end], "evidence_status": "EXACT_RESPONSE_EXCERPT",
            "source": {"kind": "run_bound_response", "response_sha256": digest(response),
                       "character_offsets": [start, end]}}


def candidate_unit_definition(exchange, adjudications):
    """Expose symbolic units of a validated candidate recipe, never GT values."""
    import re
    from fractions import Fraction

    novel = adjudications.to_dict().get("novel_operationalization")
    if not novel or novel.get("status") != "ACCEPTED":
        return None
    identity = novel.get("adjudicated_materialization_id") or ""
    if not re.fullmatch(r"expansion-recipe:[0-9a-f]{64}", identity):
        return None
    path = Path(exchange) / "materializations" / (identity.split(":", 1)[1] + ".json")
    if not path.is_file():
        return None
    artifact = read_json(path)
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != digest(body) or artifact.get("materialization_id") != identity:
        raise ValueError("Candidate unit definition materialization digest mismatch")
    execution = artifact.get("execution", {})
    provenance = execution.get("data_provenance", {})
    parameters = {item["dimension"]: item["statement"]
                  for item in novel["operationalization"]["decisions"]}
    if (execution.get("status") != "MATERIALIZED"
            or digest(execution.get("G_of_O")) != novel.get("adjudicated_g_of_o_sha256")
            or artifact.get("parameters") != parameters
            or provenance.get("effective_o_sha256") != digest(parameters)):
        raise ValueError("Candidate unit definition does not bind the reviewed method")
    plan = provenance.get("materialization_plan", {})
    if (provenance.get("compiled_plan_sha256") != digest(plan)
            or provenance.get("executed_plan_sha256") != digest(plan)
            or provenance.get("silent_substitutions") != []
            or provenance.get("unsupported_dimensions") != []):
        raise ValueError("Candidate unit definition has an unverified execution plan")
    recipe = plan.get("recipe", {})
    if (recipe.get("kind") != "association" or recipe.get("measure") != "pearson"
            or recipe.get("sampling") not in {"cell_volume", "point_equal"}):
        return None
    for field, filename in (("recipe_implementation_sha256", "construction_recipes.py"),
                            ("runtime_recipe_implementation_sha256", "runtime_construction_recipes.py")):
        if plan.get(field) != file_sha256(ROOT / "flowintentbench" / filename):
            return None  # An older implementation needs its own definition audit.
    # Normalized weighted means retain the operand's dimensions. Correlation
    # divides covariance by sqrt(variance_x * variance_y), for arbitrary X/Y.
    covariance, variance_x, variance_y = (1, 1), (2, 0), (0, 2)
    denominator = tuple(Fraction(a + b, 2) for a, b in zip(variance_x, variance_y))
    powers = [int(a - b) for a, b in zip(covariance, denominator)]
    assert powers == [0, 0]
    return {"source": "host_candidate_recipe_dimension_analysis",
            "materialization_id": identity, "artifact_sha256": artifact["artifact_sha256"],
            "executed_plan_sha256": digest(plan), "recipe": deepcopy(recipe),
            "candidate_method": parameters,
            "definition": "mean_w(dx*dy) / sqrt(mean_w(dx*dx)*mean_w(dy*dy)); mean_w(a)=sum(w*a)/sum(w)",
            "dimension_basis": ["arbitrary_input_X_unit", "arbitrary_input_Y_unit"],
            "covariance_exponents": list(covariance),
            "variance_exponents": [list(variance_x), list(variance_y)],
            "denominator_exponents": [int(item) for item in denominator],
            "result_exponents": powers, "unscaled_result_unit": "1",
            "applicability": "This proves the unit of the implemented unscaled Pearson statistic, not the unit of every finding. "
                             "Independently bind the exact prediction source to this definition. Percent, scaled, transformed, "
                             "covariance, variance, or other statistics do not inherit its unit. Do not use reference values "
                             "or numerical agreement; unresolved source binding remains PENDING."}


def unit_review_context(result, case_input, response, *, computed_definition=None):
    """Supply prediction-side declarations without inferring units or exposing GT."""
    request = result.continuation_context.get("unit_frame_request", {})
    prediction = result.extracted_prediction
    finding = next((item for item in prediction.findings
                    if item.prediction_id == request.get("prediction_id")), None) if prediction else None
    binding_status = "MISSING"
    finding_context = None
    if finding is not None:
        binding_status = "BOUND" if (finding.value == request.get("predicted_value")
                                    and finding.unit == request.get("predicted_unit")) else "REQUEST_MISMATCH"
        if binding_status == "BOUND":
            finding_context = {
                "prediction_id": finding.prediction_id,
                "extracted_statement": finding.statement,
                "extracted_value": finding.value,
                "extracted_unit": finding.unit,
                **_bound_unit_evidence(finding.evidence_span, response),
            }
    # A stale/mismatched request must not borrow declarations from another
    # prediction. O spans are evidence, not an inferred unit assignment.
    decisions = []
    if binding_status == "BOUND":
        for decision in prediction.operationalization.decisions:
            decisions.append({"dimension": decision.dimension.value,
                              "extraction_status": decision.status.value,
                              **_bound_unit_evidence(decision.evidence_span, response)})
    context = {
        "scientific_question": case_input.scientific_question,
        "case_context": case_input.case_context.model_dump(mode="json"),
        "prediction_binding_status": binding_status,
        "prediction": finding_context,
        "operationalization_declarations": decisions,
        "public_data_context": {
            "source": "case_input.flow_data.data_metadata",
            "declarations": case_input.flow_data.data_metadata.model_dump(mode="json"),
        },
        "evidence_rule": "Extracted statements and labels are not independent source declarations. "
                         "Use exact response excerpts and public data declarations to assess this "
                         "prediction's unit or scale. O excerpts apply only when they explicitly "
                         "define this quantity. A null extracted unit remains unknown unless the "
                         "supplied source evidence establishes it. Do not infer units from numerical "
                         "agreement, the reference unit, or another finding. Missing or unlocatable "
                         "evidence must remain unresolved; do not invent a declaration.",
    }
    if (computed_definition is not None and binding_status == "BOUND"
            and finding_context["evidence_status"] == "EXACT_RESPONSE_EXCERPT"):
        context["candidate_computational_definition"] = computed_definition
    return {"unit_review_context": context}


def operationalization_source_section(response):
    """Preserve literal O text even when an old extraction supplied a paraphrase.

    An explicit section boundary is required. Never guess a span from a number
    or unit, copy the Finding section, or replace an invalid extraction span.
    """
    import re
    headers = list(re.finditer(r'^ {0,3}(#{1,6})[ \t]+([^\n]+)$', response, re.MULTILINE))
    sections = [h for h in headers if h.group(2).strip().rstrip('#').strip().lower()
                == 'operationalization']
    if len(sections) != 1:
        return None
    header = sections[0]
    end = next((h.start() for h in headers if h.start() > header.start()
                and len(h.group(1)) <= len(header.group(1))), len(response))
    return _bound_unit_evidence((header.start(), end), response)


def request_adjudication(transport, payload, response):
    """Request one immutable adjudication packet.

    Structured unit reviews must include all already-authenticated answer
    evidence on the first request.  Reopening the same relationship with a
    larger packet after a PENDING reply creates a new content address and can
    keep a trial permanently active without adding new evidence.
    """
    payload = deepcopy(payload)
    from scripts.grading_policy import POLICY
    context = payload.get('unit_review_context', {})
    structured_unit = (
        POLICY.get('structured_only')
        and payload.get('pending_type') == 'unit_relationship'
        and context.get('prediction_binding_status') == 'BOUND'
        and context.get('single_packet_evidence') is True
    )
    if structured_unit:
        section = operationalization_source_section(response)
        if (section is not None and any(
                item.get('evidence_status') in {'MISSING', 'UNLOCATABLE'}
                for item in context.get('operationalization_declarations', [])
        )):
            context['operationalization_source_section'] = section
        # candidate_computational_definition is already host-derived and
        # answer-bound. Keep it in the first packet instead of using a second
        # content-addressed continuation after a PENDING response.
        return transport.request('adjudication', payload,
                                 output_schema=adjudication_schema())

    # Legacy/non-structured callers retain the historical bounded enrichment
    # path; the xhigh evaluator above never enters it.
    definition = context.pop('candidate_computational_definition', None)
    try:
        return transport.request('adjudication', payload, output_schema=adjudication_schema())
    except EvaluationPendingAdjudication as exc:
        section = operationalization_source_section(response)
        enriched = deepcopy(payload)
        pending = exc
        if (section is not None and any(item.get('evidence_status') in {'MISSING', 'UNLOCATABLE'}
                                      for item in context.get('operationalization_declarations', []))):
            enriched['unit_review_context']['operationalization_source_section'] = section
            rid = pending.continuation_context.get('external_request_id')
            if rid in transport.pending_request_ids:
                transport.pending_request_ids.remove(rid)
            try:
                return transport.request('adjudication', enriched, output_schema=adjudication_schema())
            except EvaluationPendingAdjudication as section_pending:
                pending = section_pending
        if definition is None:
            raise pending
        # Only the enriched request is active. Historical replies, including
        # PENDING, stay intact; additional evidence creates a new identity.
        enriched['unit_review_context']['candidate_computational_definition'] = definition
        rid = pending.continuation_context.get('external_request_id')
        if rid in transport.pending_request_ids:
            transport.pending_request_ids.remove(rid)
        return transport.request('adjudication', enriched, output_schema=adjudication_schema())


def stage_declared_review_data(root, row, case_input, exchange):
    """Copy only hash-verified public inputs; never stage evaluation resources."""
    import os
    import shutil
    import tempfile

    root = Path(root).resolve()
    data_root = (root / "datasets").resolve()
    if not data_root.is_relative_to(root):
        raise ValueError("review datasets directory escapes repository root")
    case_path = (root / row["case_input_path"]).resolve()
    if not case_path.is_relative_to(data_root):
        raise ValueError("review case input escapes datasets root")
    if file_sha256(case_path) != row["case_input_sha256"]:
        raise ValueError("review case input digest mismatch")
    loaded = CaseLoader(data_root).load(case_path)
    if loaded.case.model_dump(mode="json") != case_input.model_dump(mode="json"):
        raise ValueError("review public case does not match bound case input")
    dataset_id = row["dataset_id"]
    if not dataset_id or Path(dataset_id).name != dataset_id or dataset_id in {".", ".."}:
        raise ValueError("invalid review dataset identity")
    manifest_path = (data_root / dataset_id / "dataset_manifest.json").resolve()
    if not manifest_path.is_relative_to(data_root):
        raise ValueError("review dataset manifest escapes datasets root")
    manifest = read_json(manifest_path)
    if manifest.get("dataset_id") != dataset_id or manifest.get("file_root") != "datasets":
        raise ValueError("review dataset manifest identity mismatch")
    allowed = {}
    for kind, entries in (("data", manifest.get("files", [])),
                          ("geometry", manifest.get("auxiliary_assets", []))):
        for item in entries:
            key = (kind, item["path"])
            if key in allowed:
                raise ValueError("duplicate review dataset manifest file")
            allowed[key] = item
    verified = []
    for kind, entries in (("data", loaded.data_files), ("geometry", loaded.geometry_assets)):
        for item in entries:
            source = item.path.resolve()
            spec = item.specification
            if not source.is_relative_to(data_root):
                raise ValueError("review data file escapes datasets root")
            expected = allowed.get((kind, spec.path))
            if expected is None:
                raise ValueError("declared review input lacks dataset hash authority")
            if expected.get("role") != (spec.role if kind == "data" else "geometry"):
                raise ValueError("review dataset file role mismatch")
            checksum = expected.get("checksum", "")
            if (expected.get("checksum_algorithm") != "sha256" or len(checksum) != 64
                    or any(char not in "0123456789abcdef" for char in checksum)):
                raise ValueError("review input lacks a valid SHA256 checksum")
            size = source.stat().st_size
            if size != expected.get("size_bytes") or file_sha256(source) != checksum:
                raise ValueError("review input data hash or size mismatch")
            verified.append((kind, spec, source, checksum, size))
    exchange_root = Path(exchange).resolve()
    destination_root = exchange_root / "review_data"
    if destination_root.is_symlink():
        raise ValueError("review staging directory must not be a symlink")
    destination_root.mkdir(parents=True, exist_ok=True)
    files = []
    for kind, spec, source, checksum, size in verified:
        destination = destination_root / (checksum + ".bin")
        if destination.is_symlink():
            raise ValueError("review staged file must not be a symlink")
        if not destination.resolve().is_relative_to(exchange_root):
            raise ValueError("review staged file escapes exchange root")
        if destination.exists():
            if not destination.is_file() or destination.stat().st_size != size or file_sha256(destination) != checksum:
                raise ValueError("review staged file hash or size mismatch")
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=destination_root, delete=False) as handle:
                    temporary = Path(handle.name)
                    with source.open("rb") as origin:
                        shutil.copyfileobj(origin, handle, 1024 * 1024)
                if temporary.stat().st_size != size or file_sha256(temporary) != checksum:
                    raise ValueError("review source changed during staging")
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        if (destination.stat().st_mode & 0o777) != 0o444:
            destination.chmod(0o444)
        files.append({"kind": kind, "declared_path": spec.path,
                      "role": spec.role if kind == "data" else "geometry",
                      "path": str(destination), "sha256": checksum, "size_bytes": size})
    return {"files": files, "files_sha256": digest(files),
            "source": "declared_public_case_inputs_verified_against_dataset_manifest"}


def finding_review_context(result, case_input, response, root, row, exchange, *, execution_trajectory=None):
    """Give the current finding reviewer reproducible, GT-free source inputs."""
    request = result.continuation_context.get("continuation_request", {})
    claim = request.get("finding", {})
    projected = replace(result, continuation_context={"unit_frame_request": {
        "prediction_id": request.get("prediction_id"),
        "predicted_value": claim.get("value"), "predicted_unit": claim.get("unit")}})
    context = unit_review_context(projected, case_input, response)["unit_review_context"]
    finding = context["prediction"]
    if finding is not None and finding["extracted_statement"] != claim.get("statement"):
        context.update(prediction_binding_status="REQUEST_MISMATCH", prediction=None,
                       operationalization_declarations=[])
    context.pop("evidence_rule")
    context["raw_data"] = (stage_declared_review_data(root, row, case_input, exchange)
                           if context["prediction_binding_status"] == "BOUND" else None)
    if context["prediction_binding_status"] == "BOUND":
        from flowintentbench.supplemental_association_evidence import materialize_association_diagnostics, method_sources
        computation = materialize_association_diagnostics(
            root, row, exchange, result.continuation_context.get("execution_evidence_provenance"))
        if computation is not None:
            context["independent_association_diagnostics"] = computation
            declarations = method_sources(execution_trajectory)
            if declarations is not None:
                context["answer_method_declarations"] = declarations
    context["review_rule"] = (
        "Review only the current claim against the public scientific question and explicit O. "
        "Staged files are verified raw/geometry inputs, not a scientific judgment. When supplied "
        "filenames end in .bin, use declared_path/role and public format metadata to select the "
        "reader; .bin is an opaque staging suffix, not the source data format. When supplied "
        "G(O) does not cover a supplemental statistic, use supplied independent host diagnostics only "
        "when their explicit conventions match the submitted method, or independently recompute it from these files "
        "using the declared numerical conventions. Preserve the recomputation source, input hashes, "
        "parameters and outputs as audit artifacts and identify their paths/hashes in external "
        "judgment provenance outside the schema-constrained response. "
        "Do not execute candidate code, consult GT or other answers, infer omitted scientific "
        "choices, or accept a claim because it was staged or self-reported. If evidence or exact "
        "execution support is insufficient, retain PENDING with the specific gap.")
    return {"finding_review_context": context}


def independent_adjudication_requests(result):
    """Expand host discovery into original single-finding reviewer contracts."""
    context = dict(result.continuation_context)
    targets = context.pop("independent_finding_requests", None)
    if result.pending_type != "gt_outside_finding" or not targets:
        return [result]
    requests = []
    for target in targets:
        if target["branch_id"] != context["branch_id"]:
            raise ValueError("Independent finding discovery crossed O branch")
        item = dict(context, continuation_request=target,
                    continuation_request_id=continuation_request_id("finding_adjudication", target))
        requests.append(replace(result, continuation_context=item,
                                continuation_request_id=item['continuation_request_id']))
    return requests


def adjudication_payload(result, case_input, metadata, response, root, row, exchange, adjudications, *, execution_trajectory=None):
    return {
                "pending_type": result.pending_type,
                **({"stage_contract": (
                    "This operation ONLY completes the candidate OperationalizationBundle from explicit extracted decisions. "
                    "When that bundle is complete, return outer status RESOLVED with a typed response containing "
                    "novel_operationalization.status=UNRESOLVED and its operationalization bundle; branch may remain null. "
                    "Outer RESOLVED means candidate transcription complete, NOT scientific acceptance. "
                    "No materialization_id or G(O) is required at this stage: the host executes the completed "
                    "candidate next, then requests separate blinded scientific adjudication. Do not invent choices, "
                    "findings or execution proof. Return outer PENDING only when the candidate itself cannot be completed."
                )} if result.pending_type == "novel_operationalization_candidate" else {}),
                "continuation_context": dict(result.continuation_context),
                **({"scientific_context": {
                    "scientific_question": case_input.scientific_question,
                    "case_context": case_input.case_context.model_dump(mode="json"),
                    "finding_goal": metadata.finding_goal,
                    "coordinate_semantics": case_input.flow_data.data_metadata.coordinate_system.model_dump(mode="json"),
                    "raw_data": stage_declared_review_data(root, row, case_input, exchange),
                    "execution_evidence_rule": (
                        "The supplied execution_provenance is host-validated materializer output, not candidate self-report. "
                        "Review its plan, explicit candidate choices and G(O); use verified raw inputs if additional "
                        "numerical checks are needed. Scientific acceptance is still an independent judgment bound "
                        "to the exact materialization_id and G_of_O_sha256. Do not infer acceptance from execution alone."),
                }} if result.pending_type == "novel_operationalization_scientific_adjudication" else {}),
                **(semantic_review_context(result, case_input, response)
                   if result.pending_type == "semantic_uncertain" else {}),
                **({"unit_review_context": dict(
                    unit_review_context(
                        result, case_input, response,
                        computed_definition=candidate_unit_definition(exchange, adjudications),
                    )["unit_review_context"],
                    single_packet_evidence=True,
                )} if result.pending_type == "unit_relationship" else {}),
                **(finding_review_context(result, case_input, response, root, row, exchange,
                                          execution_trajectory=execution_trajectory)
                   if result.pending_type == "gt_outside_finding" else {}),
                "accumulated_adjudications": adjudication_visible_state(
                    adjudications, result.pending_type, result.continuation_context),
                "instruction": "Complete EvaluationAdjudications.to_dict() JSON, retaining visible earlier resolutions. Prior state is role-scoped: leave every out-of-role field exactly as shown, including empty placeholders; the server preserves hidden resolutions. Change only the supplied request's role-owned field: novel_operationalization for O completion/scientific review, novel_findings and novel_finding_roles for finding review, semantic_resolutions for semantic review, or unit_frame_resolutions for unit/frame review. Resolve only the supplied request from its visible evidence. For O completion, preserve explicit extracted decisions, never invent missing scientific choices, and keep status UNRESOLVED until actual materialization. For scientific O adjudication, acceptance must be bound to the exact supplied materialization_id and G(O) digest. A novel finding must be scientifically supported, relevant and consistent with the candidate O; F2 roles must be independently supported, not assigned to fill a checklist. Unknown scientific or execution support remains PENDING. No self-reported findings or G(O) may replace deterministic execution. Do not consult frozen GT, other stages, model identity, or score effects."}


def evaluate_submission(root, manifest, row, response, run_record, exchange, output, *, execution_trajectory=None,
                        response_reuser=None):
    case_input, metadata, gt, material = load_development_case(root, row)
    # Judge-visible context never contains model/provider names, trial index,
    # run paths, or the evaluation model's configuration.
    transport = FileJudgmentTransport(exchange, context={
        "case_id": row["case_id"],
        "evaluation_material_sha256": row["evaluation_material_sha256"],
        "response_sha256": digest(response),
    })
    transport.response_text = response
    backend = build_file_backend(transport)
    if response_reuser is not None:
        transport.response_reuser = response_reuser
    if run_record is not None and digest(run_record.final_response) != digest(response):
        raise ValueError("evaluation response does not match its run record")
    execution_trajectory = run_bound_execution_trajectory(
        root, run_record, fallback=execution_trajectory)
    transport.context["evaluation_manifest_sha256"] = evaluation_manifest_digest(backend.manifest())
    evaluator = build_case_evaluator(row["case_id"], manifest, backend,
        finding_verification_policy=material["finding_verification_policy"],
        branch_execution_evidence=material["branch_execution_evidence"],
        novel_o_materializer=build_expansion_recipe_materializer(root, metadata, material, transport,
                                                               execution_trajectory=execution_trajectory),
        evaluation_mode="DEVELOPMENT_EVALUATION")
    adjudications = EvaluationAdjudications()
    result = None
    try:
        for _ in range(100):
            if isinstance(result, PendingCaseEvaluationRecord):
                result = evaluator.finalize_pending_evaluation(result, case_input, metadata, gt,
                                                                 adjudications=adjudications)
            elif run_record is not None:
                result = evaluator.evaluate_run_record(run_record, case_input, metadata, gt,
                                                       adjudications=adjudications)
            else:
                result = evaluator.evaluate_response_record(case_input, metadata, gt,
                    final_response=response, efficiency=EfficiencyObservation(
                        input_tokens=None, output_tokens=None, model_turn_count=None,
                        python_execution_count=None, wall_clock_time=None),
                    adjudications=adjudications)
            if not isinstance(result, PendingCaseEvaluationRecord):
                payload = {"status": "SCORED", "evaluation_mode": "DEVELOPMENT_EVALUATION",
                           "formal_release": False, "evaluation_record": result.to_dict()}
                break
            if result.pending_type.startswith("external_") or result.pending_type == "evaluator_coverage_gap":
                payload = {"status": "PENDING", "pending_type": result.pending_type,
                           "pending_record": result.to_dict(), "scores": None}
                break
            first_pending = None
            prior = adjudications
            for item in independent_adjudication_requests(result):
                try:
                    answer = request_adjudication(transport, adjudication_payload(
                        item, case_input, metadata, response, root, row, exchange, prior,
                        execution_trajectory=execution_trajectory),
                        response)
                    updated = merge_adjudication_reply(
                        adjudications, item.pending_type, item.continuation_context, answer)
                    if updated.to_dict() == adjudications.to_dict():
                        raise EvaluationPendingAdjudication("Adjudication made no progress",
                                                           pending_type=item.pending_type)
                    adjudications = updated
                except EvaluationPendingAdjudication as exc:
                    if first_pending is None:
                        first_pending = exc
            if first_pending is not None:
                # Every independent request is now discoverable. Missing replies
                # remain pending; finalization never consumes a partial score.
                raise first_pending

        else:
            raise EvaluationPendingAdjudication("Continuation limit reached", pending_type="continuation_limit")
    except EvaluationPendingAdjudication as exc:
        payload = {"status": "PENDING", "pending_type": exc.pending_type,
                   "pending_reason": str(exc), "continuation_context": exc.continuation_context,
                   "scores": None}
        if isinstance(result, PendingCaseEvaluationRecord):
            payload["pending_record"] = result.to_dict()
    payload.update({"case_id": row["case_id"], "formal_release": False,
                    "evaluation_mode": "DEVELOPMENT_EVALUATION",
                    "evaluation_material_sha256": row["evaluation_material_sha256"],
                    "evaluation_manifest": backend.manifest().to_dict(),
                    "evaluation_manifest_digest": evaluation_manifest_digest(backend.manifest()),
                    "response_sha256": digest(response),
                    "used_external_response_ids": transport.used_response_ids,
                    "accumulated_adjudications": adjudications.to_dict(), "api_calls": 0})
    if run_record:
        payload["run_id"] = run_record.run_id
        payload["trial_index"] = run_record.trial_index
    pending_context = payload.get("continuation_context") or payload.get("pending_record", {}).get("continuation_context", {})
    request_id = pending_context.get("external_request_id")
    payload["active_request_ids"] = list(dict.fromkeys(
        ([request_id] if request_id else []) + list(getattr(transport, "pending_request_ids", []))))
    write_json(output, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, default=ROOT / "experiments/expansion_v1_development")
    replay = sub.add_parser("replay")
    replay.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    replay.add_argument("--run-record", type=Path, action="append", default=[])
    replay.add_argument("--collection-root", type=Path)
    replay.add_argument("--case-id")
    replay.add_argument("--response", type=Path)
    replay.add_argument("--exchange", type=Path, required=True)
    replay.add_argument("--reuse-index", type=Path, help="Invocation-bound historical source index; source bytes revalidated at use")
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = compile_expansion_evaluation(ROOT, args.output)
        write_json(args.output / "evaluator_manifest.json", build_file_backend(
            FileJudgmentTransport(args.output / "unused_exchange", context={})).manifest().to_dict())
        print(json.dumps({"status": "PREPARED_DEVELOPMENT", "cases": manifest["case_count"], "api_calls": 0}))
        return 0
    manifest = read_json(args.manifest)
    rows = {row["case_id"]: row for row in manifest["cases"]}
    paths = list(args.run_record)
    if args.collection_root:
        from flowintentbench.subagent_reporting import current_slot_run_paths
        current_paths = current_slot_run_paths(args.collection_root)
        if (args.collection_root / "collection_state.json").exists():
            allowed = {path.resolve() for path in current_paths}
            if any(path.resolve() not in allowed for path in paths):
                raise ValueError("explicit run record is not a current collection-ledger attempt")
        paths.extend(current_paths)
    submissions = []
    for path in dict.fromkeys(paths):
        record = RunRecord.from_dict(read_json(path))
        if record.case_id not in rows:
            raise ValueError(f"Unknown case in {path}: {record.case_id}")
        submissions.append((record.case_id, record.final_response, record, digest({"run_id": record.run_id, "source_sha256": file_sha256(path)})))
    if args.response:
        if args.case_id not in rows:
            parser.error("--response requires a known --case-id")
        response = args.response.read_text()
        submissions.append((args.case_id, response, None, digest({"case_id": args.case_id, "response": response})))
    if not submissions:
        parser.error("provide --run-record, --collection-root, or --case-id with --response")
    from scripts.replay_judgment_completion import ReplayJudgmentCompletion
    response_reuser = ReplayJudgmentCompletion(args.exchange)
    if args.reuse_index is not None:
        response_reuser.load_index(args.reuse_index)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")

    def evaluate_one(item):
        case_id, response, record, opaque_id = item
        destination = args.output / opaque_id / "evaluation.json"
        try:
            result = evaluate_submission(ROOT, manifest, rows[case_id], response, record, args.exchange, destination,
                                         response_reuser=response_reuser)
        except Exception as exc:
            result = {"status": "EVALUATOR_ERROR", "case_id": case_id, "scores": None,
                      "error": f"{type(exc).__name__}: {exc}", "api_calls": 0}
            write_json(destination, result)
        return {"case_id": case_id, "status": result["status"], "evaluation_path": str(destination),
                "run_id": record.run_id if record is not None else None,
                "active_request_ids": result.get("active_request_ids", []),
                "pending_type": result.get("pending_type")}

    # Each submission has a distinct destination and content-addressed request
    # identities. Bounded parallelism shortens local dependency refresh while
    # preserving deterministic result order and atomic exchange writes.
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="host-replay") as pool:
        results = list(pool.map(evaluate_one, submissions))
    summary = {"evaluation_mode": "DEVELOPMENT_EVALUATION", "formal_release": False,
               "api_calls": 0, "submission_count": len(results), "results": results}
    summary['local_dependency_completion'] = response_reuser.stats
    write_json(args.output / "evaluation_index.json", summary)
    active_ids = sorted({request_id for result in results for request_id in result["active_request_ids"]})
    by_operation = {}
    for request_id in active_ids:
        request_path = args.exchange / "requests" / (request_id + ".json")
        request = read_json(request_path)
        by_operation.setdefault(request["operation"], []).append({"request_id": request_id,
                                                                  "request_path": str(request_path),
                                                                  "pending_type": request["input"].get("pending_type")})
    # This inventory is phase-scoped and contains no model/run identity. Old
    # requests stay cached but cannot become active merely by lacking answers.
    write_json(args.output / "pending_requests.json", {"active_request_count": len(active_ids),
                                                        "by_operation": by_operation})
    print(json.dumps({"submissions": len(results), "statuses": {status: sum(r["status"] == status for r in results) for status in sorted({r["status"] for r in results})}, "api_calls": 0}))
    return int(any(r["status"] == "EVALUATOR_ERROR" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
