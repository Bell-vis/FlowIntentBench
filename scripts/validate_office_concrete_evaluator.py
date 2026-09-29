"""Validate one frozen Office response set with a single concrete evaluator.

This utility deliberately separates a concrete evaluator transcript from the
reviewed replay.  The transcript contains GT-blind extraction/eligibility
responses and semantic-match responses collected under one version-pinned
evaluator specification.  It is replayed through the production
``CaseEvaluator``; no benchmark metric, Ground Truth, or evaluator prompt is
modified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import (
    AdjudicationStatus,
    CaseEvaluator,
    CaseConstructionMetadata,
    EvaluationAdjudications,
    EvaluationManifest,
    EvaluatorComponentIdentity,
    ExtractionRequest,
    FindingEligibility,
    GroundTruth,
    NovelOperationalizationAdjudication,
    RunEvaluationAdjudication,
    RunRecord,
    SemanticMatchPurpose,
    SemanticMatchResult,
    evaluation_manifest_digest,
    load_run_evaluation_adjudication,
    save_run_evaluation_adjudication,
)
from flowintentbench.evaluator_backend import StructuredEvaluatorBackend
from scripts.evaluate_saved_runs import evaluate_saved_runs
from scripts.render_evaluation_report import render_evaluation_report


CASE_IDS = (
    "office_speed_zones_o1_f1",
    "office_speed_zones_o2_f1",
    "office_speed_zones_o3_f1",
    "office_speed_zones_o1_f2",
)
REFERENCE_ROOT = ROOT / "artifacts" / "reference" / "office_method_pilot_v1"
TRANSCRIPT_ROOT = REFERENCE_ROOT / "transcripts"
TRANSCRIPT_PROVENANCE_FILENAME = "transcript_provenance.json"
RUN_RECORDS = {
    case_id: REFERENCE_ROOT / "runs" / case_id / "run_record.json"
    for case_id in CASE_IDS
}
REVIEWED_ROOT = REFERENCE_ROOT / "reviewed"
O2_REVIEWED_ADJUDICATION = REFERENCE_ROOT / "adjudications" / "office_speed_zones_o2_f1.json"

# This is the frozen specification for the concrete evaluator transcript.  The
# source provenance file carries the same canonical digest and is checked
# before replay; a label such as "current" is deliberately not accepted as an
# identity claim.
TRANSCRIPT_EVALUATOR_SPEC = {
    "implementation_id": "codex-hosted-structured-evaluator-transcript",
    "implementation_version": "2",
    "provider_runtime": "codex-hosted",
    "model": "gpt-5.6-sol",
    "model_configuration": {
        "reasoning_effort": "high",
        "validation_mode": "frozen_response",
    },
    "extraction_spec_id": "flowintentbench-extraction-v1",
    "eligibility_spec_id": "flowintentbench-eligibility-v1",
    "semantic_match_spec_id": "flowintentbench-semantic-match-v1",
    "transcript_format": "office-concrete-evaluator-transcript.v2",
}


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


TRANSCRIPT_EVALUATOR_SPEC_DIGEST = _canonical_digest(TRANSCRIPT_EVALUATOR_SPEC)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _json_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_response_hash(run: RunRecord) -> str:
    if run.final_response is None:
        raise ValueError(f"frozen RunRecord {run.run_id} has no final_response")
    return hashlib.sha256(run.final_response.encode("utf-8")).hexdigest()


def _evaluator_identity() -> EvaluatorComponentIdentity:
    return EvaluatorComponentIdentity(
        implementation_id=TRANSCRIPT_EVALUATOR_SPEC["implementation_id"],
        version=TRANSCRIPT_EVALUATOR_SPEC["implementation_version"],
        provider=TRANSCRIPT_EVALUATOR_SPEC["provider_runtime"],
        model=TRANSCRIPT_EVALUATOR_SPEC["model"],
        model_configuration=TRANSCRIPT_EVALUATOR_SPEC["model_configuration"],
        prompt_version=(
            "office-concrete-evaluator-transcript.v2:"
            f"{TRANSCRIPT_EVALUATOR_SPEC_DIGEST}"
        ),
    )


def _load_and_validate_transcript_provenance(
    transcript_root: Path,
    runs: Mapping[str, RunRecord],
) -> dict[str, Any]:
    """Fail closed unless the transcript is bound to this exact replay input."""

    provenance = _load_json(transcript_root / TRANSCRIPT_PROVENANCE_FILENAME)
    if provenance.get("record_type") != "office_concrete_evaluator_transcript_provenance":
        raise ValueError("invalid concrete evaluator transcript provenance record_type")
    if provenance.get("evaluator_spec") != TRANSCRIPT_EVALUATOR_SPEC:
        raise ValueError("transcript evaluator specification does not match the frozen specification")
    if provenance.get("evaluator_spec_digest") != TRANSCRIPT_EVALUATOR_SPEC_DIGEST:
        raise ValueError("transcript evaluator specification digest mismatch")
    if provenance.get("evaluator_identity") != _evaluator_identity().to_dict():
        raise ValueError("transcript evaluator identity does not match the frozen specification")

    bindings = provenance.get("frozen_response_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(CASE_IDS):
        raise ValueError("transcript provenance must bind exactly the four Office responses")
    for case_id, run in runs.items():
        binding = bindings[case_id]
        if not isinstance(binding, Mapping):
            raise ValueError(f"invalid transcript provenance binding for {case_id}")
        expected = {
            "run_id": run.run_id,
            "run_record_path": str(RUN_RECORDS[case_id].relative_to(ROOT)),
            "run_record_sha256": _json_hash(RUN_RECORDS[case_id]),
            "final_response_sha256": _final_response_hash(run),
        }
        for field_name, value in expected.items():
            if binding.get(field_name) != value:
                raise ValueError(
                    f"transcript provenance {field_name} mismatch for {case_id}"
                )
    return provenance


@dataclass
class TranscriptConcreteEvaluator(StructuredEvaluatorBackend):
    """Production contract parser backed by a frozen concrete transcript.

    ``StructuredEvaluatorBackend`` still parses extraction and eligibility
    responses.  Match lookup is overridden only because the generic completion
    payload intentionally omits branch/prediction identifiers, while persisted
    transcript rows need those identifiers for auditability.
    """

    transcripts: Mapping[str, Mapping[str, Any]] | None = None
    question_to_case: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.transcripts is None or self.question_to_case is None:
            raise ValueError("transcripts and question_to_case are required")

    def _case_id(self, scientific_question: str) -> str:
        try:
            return self.question_to_case[scientific_question]  # type: ignore[index]
        except KeyError as exc:
            raise ValueError("unexpected scientific question in evaluator transcript") from exc

    def _completion(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        case_id = self._case_id(str(payload["scientific_question"]))
        transcript = self.transcripts[case_id]  # type: ignore[index]
        if operation == "extraction":
            return transcript["extraction"]
        if operation == "eligibility":
            prediction_id = payload["finding"]["prediction_id"]
            return transcript["eligibility"][prediction_id]
        raise ValueError(f"unexpected completion operation {operation!r}")

    def match(self, request):  # type: ignore[override]
        case_id = self._case_id(request.scientific_question)
        transcript = self.transcripts[case_id]  # type: ignore[index]
        semantic = transcript["semantic"]
        if request.purpose == SemanticMatchPurpose.FINDING_DEDUPLICATION:
            value = semantic["deduplication_default"]
        elif request.purpose == SemanticMatchPurpose.OPERATIONALIZATION:
            key = f"{request.branch_id}|{request.dimension.value}"
            value = semantic["operationalization"].get(
                key, {"result": "NO_MATCH"}
            )
        else:
            key = f"{request.branch_id}|{request.predicted_id}|{request.reference_id}"
            value = semantic["finding_matches"].get(
                key, semantic["finding_default"]
            )
        if not isinstance(value, Mapping) or set(value) != {"result"}:
            raise ValueError(f"invalid semantic transcript judgment for {case_id}: {value!r}")
        return SemanticMatchResult(value["result"])


def _load_transcripts(transcript_root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for case_id in CASE_IDS:
        stage1 = _load_json(transcript_root / "stage1" / f"{case_id}.json")
        stage2 = _load_json(transcript_root / "stage2" / f"{case_id}.json")
        output[case_id] = {**stage1, "semantic": stage2}
    return output


def _accepted_o2_novel_adjudication() -> NovelOperationalizationAdjudication:
    reviewed = load_run_evaluation_adjudication(O2_REVIEWED_ADJUDICATION)
    novel = reviewed.adjudications.novel_operationalization
    if novel is None or novel.status != AdjudicationStatus.ACCEPTED or novel.branch is None:
        raise ValueError("the frozen O2 temporary branch is not an accepted executed branch")
    return novel


# This is a curator-authored decision ledger for this one frozen Office replay,
# not a semantic interpretation table.  Branch meaning is read below from the
# actual stored O bundle and G(O) bundle.  Omitting a required entry is an
# unresolved adjudication and intentionally produces a Pending record.
EXPLICIT_FINDING_ADJUDICATIONS: dict[tuple[str, str, str], AdjudicationStatus] = {
    **{
        ("office_speed_zones_o1_f2", "o1_speed_zones_open", prediction_id): AdjudicationStatus.ACCEPTED
        for prediction_id in ("f3", "f6", "f7", "f10", "f11", "f12")
    },
    **{
        ("office_speed_zones_o2_f1", branch_id, prediction_id): AdjudicationStatus.REJECTED
        for branch_id in ("o2_peak_speed", "o2_mean_speed", "o2_q90_mean_speed")
        for prediction_id in (
            "strongest_region_identity",
            "strongest_region_mean_location",
            "strongest_region_mean_speed",
            "strongest_region_rms_speed",
            "strongest_region_peak_speed",
            "opposing_wall_region_mean_speed",
            "upper_boundary_region_mean_speed",
        )
    },
    **{
        ("office_speed_zones_o2_f1", "temporary_o2_gte_030_mean_speed", prediction_id): AdjudicationStatus.ACCEPTED
        for prediction_id in (
            "strongest_region_identity",
            "strongest_region_rms_speed",
            "strongest_region_peak_speed",
            "opposing_wall_region_mean_speed",
            "upper_boundary_region_mean_speed",
        )
    },
    **{
        ("office_speed_zones_o3_f1", "o3_fixed_peak_speed", prediction_id): status
        for prediction_id, status in {
            "finding_1": AdjudicationStatus.ACCEPTED,
            "finding_2": AdjudicationStatus.REJECTED,
            "finding_3": AdjudicationStatus.REJECTED,
            "finding_4": AdjudicationStatus.ACCEPTED,
            "finding_5": AdjudicationStatus.ACCEPTED,
            "finding_7": AdjudicationStatus.REJECTED,
            "finding_8": AdjudicationStatus.REJECTED,
            "finding_9": AdjudicationStatus.REJECTED,
        }.items()
    },
    **{
        ("office_speed_zones_o3_f1", "o3_fixed_mean_speed", prediction_id): AdjudicationStatus.REJECTED
        for prediction_id in ("finding_1", "finding_2", "finding_3", "finding_4", "finding_5", "finding_6", "finding_7", "finding_9")
    },
    **{
        ("office_speed_zones_o3_f1", "o3_q90_mean_speed", prediction_id): AdjudicationStatus.REJECTED
        for prediction_id in ("finding_1", "finding_2", "finding_3", "finding_4", "finding_5", "finding_6", "finding_7", "finding_9")
    },
    **{
        ("office_speed_zones_o3_f1", "o3_peak_point_speed", prediction_id): status
        for prediction_id, status in {
            "finding_1": AdjudicationStatus.ACCEPTED,
            "finding_2": AdjudicationStatus.REJECTED,
            "finding_3": AdjudicationStatus.REJECTED,
            "finding_5": AdjudicationStatus.ACCEPTED,
            "finding_7": AdjudicationStatus.REJECTED,
            "finding_8": AdjudicationStatus.REJECTED,
            "finding_9": AdjudicationStatus.REJECTED,
        }.items()
    },
}


def _semantic_match_prediction_ids(
    semantic: Mapping[str, Any], branch_id: str
) -> set[str]:
    matched: set[str] = set()
    prefix = f"{branch_id}|"
    for key, result in semantic["finding_matches"].items():
        if not key.startswith(prefix) or result.get("result") != "MATCH":
            continue
        _, prediction_id, _ = key.split("|", 2)
        matched.add(prediction_id)
    return matched


def _branch_contexts(
    ground_truth: GroundTruth,
    novel: NovelOperationalizationAdjudication | None,
) -> dict[str, dict[str, Any]]:
    """Expose the stored O and G(O) bundles used for an adjudication audit."""

    operationalizations = {
        bundle.operationalization_id: bundle
        for bundle in ground_truth.acceptable_operationalizations
    }
    contexts = {
        branch.operationalization_id: {
            "source": "frozen_ground_truth",
            "operationalization": operationalizations[branch.operationalization_id].model_dump(mode="json"),
            "reference_findings": branch.model_dump(mode="json"),
        }
        for branch in ground_truth.findings_by_operationalization
    }
    if novel is not None:
        if novel.status != AdjudicationStatus.ACCEPTED or novel.branch is None:
            raise ValueError("Office temporary O branch must be accepted and executed")
        branch = novel.branch
        contexts[branch.findings.operationalization_id] = {
            "source": "accepted_executed_novel_operationalization",
            "operationalization": branch.operationalization.model_dump(mode="json"),
            "reference_findings": branch.findings.model_dump(mode="json"),
        }
    return contexts


def _adjudication_for(
    case_id: str,
    prediction,
    ground_truth,
    eligibility: Mapping[str, FindingEligibility],
    semantic: Mapping[str, Any],
    *,
    ledger: Mapping[tuple[str, str, str], AdjudicationStatus] = EXPLICIT_FINDING_ADJUDICATIONS,
) -> tuple[EvaluationAdjudications, list[dict[str, Any]]]:
    branches = [
        branch.operationalization_id
        for branch in ground_truth.findings_by_operationalization
    ]
    novel: NovelOperationalizationAdjudication | None = None
    if case_id == "office_speed_zones_o2_f1":
        novel = _accepted_o2_novel_adjudication()
        assert novel.branch is not None
        branches.append(novel.branch.findings.operationalization_id)

    contexts = _branch_contexts(ground_truth, novel)
    statuses: dict[tuple[str, str], AdjudicationStatus] = {}
    rationale: list[dict[str, Any]] = []
    for branch_id in branches:
        matched_ids = _semantic_match_prediction_ids(semantic, branch_id)
        for finding in prediction.findings:
            if not eligibility[finding.prediction_id].in_f_app or finding.prediction_id in matched_ids:
                continue
            status = ledger.get((case_id, branch_id, finding.prediction_id))
            if status is None:
                rationale.append(
                    {
                        "branch_id": branch_id,
                        "prediction_id": finding.prediction_id,
                        "status": "PENDING",
                        "rationale": "No explicit GT-outside adjudication is available; the evaluator must not infer acceptance from eligibility or data verifiability.",
                        "branch_context": contexts[branch_id],
                    }
                )
                continue
            statuses[(branch_id, finding.prediction_id)] = status
            rationale.append(
                {
                    "branch_id": branch_id,
                    "prediction_id": finding.prediction_id,
                    "status": status.value,
                    "rationale": "Explicit curator adjudication against the stored OperationalizationBundle and G(O) bundle; eligibility alone receives no credit.",
                    "branch_context": contexts[branch_id],
                }
            )
    return EvaluationAdjudications(novel_operationalization=novel, novel_findings=statuses), rationale


def _reviewed_record(case_id: str) -> Mapping[str, Any]:
    paths = sorted(REVIEWED_ROOT.glob(f"{case_id}/case_evaluation_record.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one reviewed oracle record for {case_id}, found {len(paths)}")
    return _load_json(paths[0])


def _concrete_record(evaluation_root: Path, case_id: str) -> Mapping[str, Any]:
    paths = sorted(evaluation_root.glob(f"**/{case_id}/trial-1/**/case_evaluation_record.json"))
    paths.extend(
        sorted(evaluation_root.glob(f"**/{case_id}/trial-1/**/pending_case_evaluation*.json"))
    )
    if len(paths) != 1:
        raise ValueError(f"expected one concrete final or pending record for {case_id}, found {len(paths)}")
    return _load_json(paths[0])


COMPARISON_CLASSIFICATIONS = {
    "office_speed_zones_o1_f1": "acceptable semantic variation: the four atomic claims and O roles agree; wording differs.",
    "office_speed_zones_o2_f1": "acceptable semantic variation: the 13 claims and their independent identity/location atomicity agree; wording differs.",
    "office_speed_zones_o3_f1": "acceptable semantic variation: both preserve nine claims and property_measure=MISSING; identity/extent wording is grouped differently without losing a proposition.",
    "office_speed_zones_o1_f2": "acceptable semantic variation: the concrete evaluator keeps a separate strongest-region identity claim, so it has 12 rather than 11 findings; no asserted result is lost and the 3D extent remains one Finding.",
}


def _semantic_match_summary(record: Mapping[str, Any]) -> str:
    counts: dict[str, int] = {}
    for item in record["semantic_matches"]:
        purpose = item["request"]["purpose"]
        if item["result"] == "MATCH":
            counts[purpose] = counts.get(purpose, 0) + 1
    return ", ".join(f"{purpose}={count}" for purpose, count in sorted(counts.items())) or "none"


def _o_decision_summary(prediction: Mapping[str, Any]) -> str:
    decisions = {
        item["dimension"]: item["status"]
        for item in prediction["operationalization"]["decisions"]
    }
    return "; ".join(
        f"{dimension}={decisions[dimension]}" for dimension in sorted(decisions)
    )


def _comparison_markdown(
    transcripts: Mapping[str, Mapping[str, Any]],
    evaluation_root: Path,
    destination: Path,
) -> None:
    lines = ["# Concrete Evaluator vs Reviewed Reference", ""]
    for case_id in CASE_IDS:
        concrete = transcripts[case_id]["extraction"]
        concrete_record = _concrete_record(evaluation_root, case_id)
        reviewed_record = _reviewed_record(case_id)
        reviewed = reviewed_record["extracted_prediction"]
        reviewed_findings = reviewed["findings"]
        concrete_findings = concrete["findings"]
        concrete_statements = {item["statement"] for item in concrete_findings}
        reviewed_statements = {item["statement"] for item in reviewed_findings}
        lines.extend((f"## {case_id}", "", "| Item | Concrete Evaluator | Reviewed Reference | Agreement |", "|---|---|---|---|"))
        lines.append(f"| Finding count | {len(concrete_findings)} | {len(reviewed_findings)} | {'MATCH' if len(concrete_findings) == len(reviewed_findings) else 'DIFFERENT'} |")
        lines.append(f"| O extraction | {_o_decision_summary(concrete)} | {_o_decision_summary(reviewed)} | {'MATCH' if _o_decision_summary(concrete) == _o_decision_summary(reviewed) else 'DIFFERENT'} |")
        lines.append("| Finding atomicity | response-level atomic Findings | reviewed replay atomic Findings | review below |")
        concrete_result = concrete_record.get("result")
        if isinstance(concrete_result, Mapping):
            concrete_eligibility = f"{len(concrete_result['f_app_prediction_ids'])}/{len(concrete_findings)} in F_app"
            concrete_matches = _semantic_match_summary(concrete_record)
            concrete_status = str(concrete_result["run_status"])
        else:
            concrete_eligibility = "PENDING; F_app not finalized"
            concrete_matches = _semantic_match_summary(concrete_record)
            concrete_status = "PENDING"
        lines.append(f"| Eligibility | {concrete_eligibility} | {sum(all(item.values()) for item in reviewed_record['finding_eligibility'].values())}/{len(reviewed_findings)} in F_app | independently judged |")
        lines.append(f"| Semantic MATCHes | {concrete_matches} | {_semantic_match_summary(reviewed_record)} | separately auditable |")
        lines.append(f"| Final status | {concrete_status} | {reviewed_record['result']['run_status']} | {'MATCH' if concrete_status == reviewed_record['result']['run_status'] else 'DIFFERENT'} |")
        lines.append("")
        lines.append(f"Classification: {COMPARISON_CLASSIFICATIONS[case_id]}")
        lines.append("")
        lines.append("Concrete evaluator statements:")
        lines.extend(f"- {item['statement']}" for item in concrete_findings)
        missed = reviewed_statements - concrete_statements
        extra = concrete_statements - reviewed_statements
        if missed:
            lines.append("Reviewed-only statements:")
            lines.extend(f"- {statement}" for statement in sorted(missed))
        if extra:
            lines.append("Concrete-only statements:")
            lines.extend(f"- {statement}" for statement in sorted(extra))
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def _diagnostics_markdown(evaluation_root: Path, destination: Path) -> None:
    lines = ["# Per-Case Evaluator Diagnostics", ""]
    paths = sorted(evaluation_root.rglob("case_evaluation_record.json"))
    paths.extend(sorted(evaluation_root.rglob("pending_case_evaluation*.json")))
    for path in paths:
        record = _load_json(path)
        prediction = record["extracted_prediction"]
        eligibility = record["finding_eligibility"]
        result = record.get("result")
        if not isinstance(result, Mapping):
            findings = [] if prediction is None else prediction["findings"]
            lines.extend((
                f"## {record['case_id']}",
                "",
                f"- Status: PENDING ({record['pending_type']})",
                f"- Pending reason: {record['pending_reason']}",
                f"- Extracted Findings: {len(findings)}",
                f"- Eligibility judgments: {len(eligibility)}",
                "- Scientific metrics: not computed while adjudication remains pending.",
                "",
            ))
            continue
        selected = result["best_c_branches"] or result["metrics"]["scientific_findings"]["best_finding_branches"]
        diagnostics = result["finding_branch_diagnostics"]
        selected_diagnostic = next(
            (item for item in diagnostics if item["branch_id"] in selected),
            None,
        )
        matched_ids = set() if selected_diagnostic is None else {item["prediction_id"] for item in selected_diagnostic["matched_pairs"]}
        accepted_ids = set() if selected_diagnostic is None else set(selected_diagnostic["accepted_gt_outside_prediction_ids"])
        rejected_count = len(set(result["f_app_prediction_ids"]) - matched_ids - accepted_ids)
        lines.extend((f"## {result['case_id']}", "", "- Status: COMPLETED", f"- Extracted O decisions: {len(prediction['operationalization']['decisions'])}", f"- Extracted Findings: {len(prediction['findings'])}", f"- Findings after deduplication: {len(result['deduplicated_prediction_ids'])}", f"- Eligible Findings: {len(result['f_app_prediction_ids'])}", f"- Compatible O branches: `{', '.join(result['compatible_o_branches']) or 'none'}`", f"- Finding diagnostic branch(es): `{', '.join(selected) or 'none'}`", f"- GT-matched Findings: {0 if selected_diagnostic is None else len(selected_diagnostic['matched_pairs'])}", f"- Accepted GT-outside Findings: {0 if selected_diagnostic is None else len(selected_diagnostic['accepted_gt_outside_prediction_ids'])}", f"- Rejected GT-outside Findings: {rejected_count}", f"- Deterministic verification failures: {sum(not item['verified'] for item in record['value_verifications'])}", "- Unresolved pending items: 0", "", "Deduplicated Finding statements:"))
        lines.extend(f"- {item['statement']}" for item in prediction["findings"])
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")


def validate(output_root: Path, transcript_root: Path) -> dict[str, Path]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {output_root}")
    runs = {case_id: RunRecord.load_json(path) for case_id, path in RUN_RECORDS.items()}
    _load_and_validate_transcript_provenance(transcript_root, runs)
    transcripts = _load_transcripts(transcript_root)
    dataset_dir = ROOT / "datasets" / "Office"
    case_paths = {case_id: dataset_dir / "construction" / "cases" / case_id for case_id in CASE_IDS}
    questions = {
        case_id: _load_json(case_path / "case_input.json")["scientific_question"]
        for case_id, case_path in case_paths.items()
    }
    identity = _evaluator_identity()
    backend = TranscriptConcreteEvaluator(
        completion=lambda operation, payload: {},
        identity=identity,
        transcripts=transcripts,
        question_to_case={question: case_id for case_id, question in questions.items()},
    )
    backend.completion = backend._completion
    manifest = backend.manifest()
    evaluator = CaseEvaluator(backend, backend, backend, manifest)
    output_root.mkdir(parents=True)
    shutil.copytree(transcript_root, output_root / "transcripts", dirs_exist_ok=False)
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps({"manifest": manifest.to_dict(), "digest": evaluation_manifest_digest(manifest)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    adjudications = {}
    rationale: list[dict[str, Any]] = []
    for case_id, run in runs.items():
        metadata = CaseConstructionMetadata.model_validate(
            _load_json(case_paths[case_id] / "case_construction_metadata.json")
        )
        prediction = backend.extract(
            ExtractionRequest(
                scientific_question=questions[case_id],
                case_context=_load_json(case_paths[case_id] / "case_input.json")["case_context"],
                principal_operationalization_dimensions=tuple(metadata.principal_operationalization_dimensions),
                unresolved_operationalization_dimensions=tuple(metadata.unresolved_operationalization_dimensions),
                final_response=run.final_response or "",
            )
        )
        ground_truth = GroundTruth.model_validate(
            _load_json(case_paths[case_id] / "ground_truth.json")
        )
        eligibility = {
            finding_id: FindingEligibility(**value)
            for finding_id, value in transcripts[case_id]["eligibility"].items()
        }
        decision, details = _adjudication_for(
            case_id,
            prediction,
            ground_truth,
            eligibility,
            transcripts[case_id]["semantic"],
        )
        rationale.extend({"case_id": case_id, **item} for item in details)
        wrapper = RunEvaluationAdjudication(run.run_id, case_id, decision)
        adjudications[run.run_id] = wrapper
        (output_root / "adjudications").mkdir(exist_ok=True)
        save_run_evaluation_adjudication(wrapper, output_root / "adjudications" / f"{case_id}.json")
    (output_root / "adjudication_rationale.json").write_text(json.dumps(rationale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "frozen_response_references.json").write_text(
        json.dumps(
            {
                case_id: {
                    "path": str(path.relative_to(ROOT)),
                    "run_id": runs[case_id].run_id,
                    "run_record_sha256": _json_hash(path),
                    "final_response_sha256": _final_response_hash(runs[case_id]),
                }
                for case_id, path in RUN_RECORDS.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation_root = output_root / "evaluation"
    summary_path = evaluation_root / "summary.json"
    evaluate_saved_runs(
        datasets_root=ROOT / "datasets",
        case_paths=case_paths,
        run_record_paths={case_id: [path] for case_id, path in RUN_RECORDS.items()},
        evaluator=evaluator,
        evaluation_root=evaluation_root,
        summary_path=summary_path,
        mode="PILOT",
        adjudications_by_run_id=adjudications,
    )
    report_dir = output_root / "report"
    render_evaluation_report(evaluation_root=evaluation_root, summary_path=summary_path, output_dir=report_dir, model="gpt-5.6-sol")
    _comparison_markdown(transcripts, evaluation_root, output_root / "evaluator_comparison.md")
    _diagnostics_markdown(evaluation_root, output_root / "diagnostics.md")
    return {"summary": summary_path, "report": report_dir / "report.md", "trial_metrics": report_dir / "trial_metrics.csv", "comparison": output_root / "evaluator_comparison.md", "diagnostics": output_root / "diagnostics.md"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-root", type=Path, default=TRANSCRIPT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "office-concrete-evaluator-validation",
    )
    args = parser.parse_args()
    for name, path in validate(args.output_root, args.transcript_root).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
