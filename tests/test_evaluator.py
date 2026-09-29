from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from flowintentbench.case_design import CaseConstructionMetadata
from flowintentbench.evaluation_metrics import EfficiencyObservation
from flowintentbench.evaluator import (
    AdjudicatedNovelBranch,
    AdjudicationStatus,
    CaseEvaluator,
    CaseEvaluationResult,
    EvaluationConfigurationError,
    EvaluationAdjudications,
    EvaluationManifest,
    EvaluationPendingAdjudication,
    EvaluatorComponentIdentity,
    ExtractedOperationalization,
    ExtractedOperationalizationDecision,
    ExtractedPrediction,
    ExtractionStatus,
    FindingVerificationMode,
    FindingEligibility,
    NovelOperationalizationAdjudication,
    PredictedAtomicFinding,
    PendingCaseEvaluationRecord,
    RunEvaluationAdjudication,
    SemanticMatchPurpose,
    SemanticMatchRequest,
    SemanticMatchResult,
    TrialEvaluation,
    aggregate_cases,
    aggregate_trials_for_case,
    evaluation_manifest_digest,
    finding_verification_mode,
    load_case_evaluation_record,
    load_evaluation_manifest,
    load_evaluation_adjudications,
    load_run_evaluation_adjudication,
    save_case_evaluation_record,
    save_evaluation_manifest,
    save_evaluation_adjudications,
    save_run_evaluation_adjudication,
    verify_finding_value,
    _match,
)
from flowintentbench.ground_truth import (
    GroundTruth,
    OperationalizationBundle,
    ReferenceFinding,
)
from flowintentbench.model_runner import RunRecord, RunStatus
from flowintentbench.schema import BenchmarkCaseInput


def _case_input() -> BenchmarkCaseInput:
    return BenchmarkCaseInput.model_validate(
        {
            "scientific_question": "Identify the strongest region and report its strength.",
            "flow_data": {
                "data_files": [{"path": "flow.vtu", "role": "flow_field"}],
                "data_metadata": {
                    "format": {"type": "VTU", "details": {}},
                    "grid": {"type": "unstructured"},
                    "variables": [
                        {
                            "name": "velocity",
                            "physical_quantity": "velocity",
                            "type": "vector",
                            "components": 3,
                            "association": "point",
                            "unit": None,
                            "source": "stored",
                        }
                    ],
                    "coordinate_system": {
                        "type": "cartesian",
                        "axis_meaning": {},
                        "unit": None,
                    },
                    "temporal": {"type": "single_snapshot"},
                },
            },
            "case_context": {"physical_setting": "test flow"},
        }
    )


def _metadata(
    *,
    case_id: str = "case_a",
    responsibility: str = "partially_specified",
) -> CaseConstructionMetadata:
    unresolved = (
        ["property_measure"]
        if responsibility == "partially_specified"
        else (
            ["feature_definition", "property_measure"]
            if responsibility == "model_selected"
            else []
        )
    )
    return CaseConstructionMetadata.model_validate(
        {
            "case_id": case_id,
            "case_family_id": "family_a",
            "dataset_id": "dataset_a",
            "scientific_target": "flow regions",
            "finding_goal": "identify and characterize the strongest region",
            "operationalization_responsibility": responsibility,
            "finding_openness": "bounded",
            "principal_operationalization_dimensions": [
                "feature_definition",
                "property_measure",
            ],
            "unresolved_operationalization_dimensions": unresolved,
            "explicit_method_constraints": (
                []
                if responsibility == "model_selected"
                else [
                    {
                        "category": "feature_definition",
                        "statement": "use connected regions",
                        "question_fragment": "region",
                    }
                ]
            ),
            "explicit_finding_requirements": [
                {
                    "category": "existence_or_identity",
                    "statement": "report the strongest region",
                    "question_fragment": "strongest region",
                },
                {
                    "category": "quantity",
                    "statement": "report its strength",
                    "question_fragment": "strength",
                },
            ],
        }
    )


def _ground_truth(*, case_id: str = "case_a") -> GroundTruth:
    return GroundTruth.model_validate(
        {
            "dataset_id": "dataset_a",
            "case_id": case_id,
            "case_family_id": "family_a",
            "acceptable_operationalizations": [
                {
                    "operationalization_id": "branch_peak",
                    "decisions": [
                        {
                            "dimension": "feature_definition",
                            "statement": "Use connected regions.",
                        },
                        {
                            "dimension": "property_measure",
                            "statement": "Use peak strength.",
                        },
                    ],
                },
                {
                    "operationalization_id": "branch_mean",
                    "decisions": [
                        {
                            "dimension": "feature_definition",
                            "statement": "Use connected regions.",
                        },
                        {
                            "dimension": "property_measure",
                            "statement": "Use mean strength.",
                        },
                    ],
                },
            ],
            "findings_by_operationalization": [
                {
                    "operationalization_id": "branch_peak",
                    "findings": [
                        {
                            "finding_id": "peak_identity",
                            "category": "existence_or_identity",
                            "statement": "The strongest region is A.",
                            "importance": "core",
                            "value": "A",
                        },
                        {
                            "finding_id": "peak_strength",
                            "category": "quantity",
                            "statement": "Its peak strength is 4.0.",
                            "importance": "core",
                            "value": 4.0,
                            "verification": {"absolute_tolerance": 0.1},
                        },
                    ],
                },
                {
                    "operationalization_id": "branch_mean",
                    "findings": [
                        {
                            "finding_id": "mean_identity",
                            "category": "existence_or_identity",
                            "statement": "The strongest region is B.",
                            "importance": "core",
                            "value": "B",
                        },
                        {
                            "finding_id": "mean_strength",
                            "category": "quantity",
                            "statement": "Its mean strength is 2.0.",
                            "importance": "core",
                            "value": 2.0,
                            "verification": {"absolute_tolerance": 0.1},
                        },
                    ],
                },
            ],
        }
    )


class StaticExtractor:
    def __init__(self, prediction: ExtractedPrediction) -> None:
        self.prediction = prediction
        self.requests = []

    def extract(self, request):
        self.requests.append(request)
        return self.prediction


class StubMatcher:
    def __init__(self, equivalent_pairs=(), uncertain_pairs=()) -> None:
        self.requests = []
        self.equivalent_pairs = {
            frozenset(self._normalize(item) for item in pair)
            for pair in equivalent_pairs
        }
        self.uncertain_pairs = {
            frozenset(self._normalize(item) for item in pair)
            for pair in uncertain_pairs
        }

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split()).rstrip(".")

    def match(self, request):
        self.requests.append(request)
        pair = frozenset(
            {
                self._normalize(request.predicted_statement),
                self._normalize(request.reference_statement),
            }
        )
        if pair in self.uncertain_pairs:
            return SemanticMatchResult.UNCERTAIN
        if (
            self._normalize(request.predicted_statement)
            == self._normalize(request.reference_statement)
            or pair in self.equivalent_pairs
        ):
            return SemanticMatchResult.MATCH
        return SemanticMatchResult.NO_MATCH


class StubEligibilityJudge:
    def __init__(self, *, irrelevant_ids=()) -> None:
        self.irrelevant_ids = set(irrelevant_ids)
        self.requests = []

    def judge(self, request):
        self.requests.append(request)
        return FindingEligibility(
            True,
            request.finding.prediction_id not in self.irrelevant_ids,
            True,
        )


def _manifest(*, unit_converter_version=None) -> EvaluationManifest:
    component = EvaluatorComponentIdentity("test.stub", "1")
    return EvaluationManifest(
        evaluator_backend=component,
        extraction_spec_id="extraction-v1",
        eligibility_spec_id="eligibility-v1",
        semantic_match_spec_id="semantic-match-v1",
        unit_converter_version=unit_converter_version,
    )


def _prediction(
    *,
    property_status: ExtractionStatus = ExtractionStatus.EXTRACTED,
) -> ExtractedPrediction:
    findings = (
        PredictedAtomicFinding("identity", "The strongest region is B.", "B"),
        PredictedAtomicFinding("strength", "Its mean strength is 2.0.", 2.05),
        PredictedAtomicFinding("duplicate", "Region B is the strongest.", "B"),
        PredictedAtomicFinding("extra", "The selected region is elongated."),
    )
    return ExtractedPrediction(
        operationalization=ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use connected regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure",
                    property_status,
                    "Use mean strength." if property_status == ExtractionStatus.EXTRACTED else None,
                ),
            )
        ),
        findings=findings,
    )


def _adjudications(*, accept_extra_mean: bool = True) -> EvaluationAdjudications:
    return EvaluationAdjudications(
        novel_findings={
            ("branch_peak", "identity"): AdjudicationStatus.REJECTED,
            ("branch_peak", "strength"): AdjudicationStatus.REJECTED,
            ("branch_peak", "extra"): AdjudicationStatus.REJECTED,
            ("branch_mean", "extra"): (
                AdjudicationStatus.ACCEPTED
                if accept_extra_mean
                else AdjudicationStatus.REJECTED
            ),
        }
    )


def _evaluator(
    prediction: ExtractedPrediction,
    matcher: StubMatcher | None = None,
    *,
    irrelevant_ids=(),
    novel_o_materializer=None,
    finding_requirement_contract=None,
):
    extractor = StaticExtractor(prediction)
    matcher = matcher or StubMatcher(
        equivalent_pairs=[
            ("The strongest region is B.", "Region B is the strongest.")
        ]
    )
    return (
        CaseEvaluator(
            extractor,
            matcher,
            StubEligibilityJudge(irrelevant_ids=irrelevant_ids),
            _manifest(),
            novel_o_materializer=novel_o_materializer,
            finding_requirement_contract=finding_requirement_contract,
        ),
        extractor,
    )


def _trusted_novel_materializer(bundle):
    """Explicit execution fixture for accepted novel-O tests."""

    # The callback contract deliberately receives only O; untrusted
    # adjudication Findings/provenance are not visible at this boundary.
    assert not hasattr(bundle, "findings")
    parameters = {
        decision.dimension.value: decision.statement
        for decision in bundle.decisions
    }
    plan = {"domain": "test_novel_o", "dimensions": sorted(parameters)}
    canonical = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    plan_hash = hashlib.sha256(canonical(plan)).hexdigest()
    o_hash = hashlib.sha256(canonical(parameters)).hexdigest()
    return {
        "parameters": parameters,
        "materialization_id": "mat:trusted:novel",
        "execution": {
            "status": "MATERIALIZED",
            "reproducible": True,
            "G_of_O": {"source": "trusted-fixture", "result": "C"},
            "data_provenance": {
                "dataset_id": "dataset_a",
                "dataset_manifest_sha256": "0" * 64,
                "effective_o_sha256": o_hash,
                "materialization_plan": plan,
                "compiled_plan_sha256": plan_hash,
                "executed_plan_sha256": plan_hash,
                "silent_substitutions": [],
                "unsupported_dimensions": [],
            },
        },
        "findings": [
            {
                "finding_id": "trusted_identity",
                "category": "existence_or_identity",
                "statement": "The strongest watershed region is C.",
                "importance": "core",
                "value": "C",
            }
        ],
    }


def _novel_fixture():
    metadata = _metadata(responsibility="model_selected")
    prediction = ExtractedPrediction(
        operationalization=ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use watershed regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use median strength."
                ),
            )
        ),
        findings=(
            PredictedAtomicFinding(
                "novel_identity", "The strongest watershed region is C.", "C"
            ),
        ),
    )
    branch = AdjudicatedNovelBranch(
        operationalization={
            "operationalization_id": "novel_branch",
            "decisions": [
                {"dimension": "feature_definition", "statement": "Use watershed regions."},
                {"dimension": "property_measure", "statement": "Use median strength."},
            ],
        },
        findings={
            "operationalization_id": "novel_branch",
            "findings": [
                {
                    "finding_id": "injected_identity",
                    "category": "existence_or_identity",
                    "statement": "INJECTED G(O) MUST NOT BE TRUSTED.",
                    "importance": "core",
                    "value": "injected",
                }
            ],
        },
        materialization_id="mat:injected",
        execution_provenance={
            "status": "MATERIALIZED",
            "G_of_O": {"source": "injected"},
        },
    )
    return metadata, prediction, branch


def test_deterministic_evaluator_core_keeps_gt_blind_extraction_and_scores_best_branch():
    evaluator, extractor = _evaluator(_prediction())
    result = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="A natural-language response.",
        efficiency=EfficiencyObservation(100, 20, 2, 1, 3.0),
        adjudications=_adjudications(),
    )
    payload = result.to_dict()
    metrics = payload["metrics"]
    assert metrics["scientific_operationalization"] == {
        "o_score": 1.0,
        "best_o_branches": ["branch_mean"],
        "urs": 1.0,
        "resolved_o_compliance": 1.0,
    }
    assert metrics["scientific_findings"] == {
        "finding_precision": 1.0,
        "core_finding_recall": 1.0,
        "best_finding_branches": ["branch_mean"],
        "finding_requirement_recall": 1.0,
        "finding_recall_mode": "FIXED_REFERENCE_CORE",
        "adequate_core_complete": None,
    }
    assert metrics["o_f_consistency"] == {
        "c_score": 1.0,
        "branch_alignment": 1,
        "operationalization_determinate": True,
    }
    assert result.deduplicated_prediction_ids == ("identity", "strength", "extra")
    assert result.f_app_prediction_ids == ("identity", "strength", "extra")
    assert result.compatible_o_branches == ("branch_mean",)
    request = extractor.requests[0]
    assert request.final_response == "A natural-language response."
    assert not hasattr(request, "acceptable_operationalizations")


def test_all_independent_eligibility_requests_discovered_before_pending():
    evaluator, _ = _evaluator(_prediction())
    called = []

    class PendingEligibility:
        def judge(self, request):
            called.append(request.finding.prediction_id)
            raise EvaluationPendingAdjudication('Awaiting independent reviewer',
                                                pending_type='external_eligibility')

    evaluator.eligibility_judge = PendingEligibility()
    pending = evaluator.evaluate_response_record(
        _case_input(), _metadata(), _ground_truth(), final_response='Answer',
        efficiency=EfficiencyObservation(100, 20, 2, 1, 3.0), adjudications=_adjudications())
    assert isinstance(pending, PendingCaseEvaluationRecord)
    assert called == ['identity', 'strength', 'extra']
    assert pending.finding_eligibility == {}
    evaluator.eligibility_judge = StubEligibilityJudge()
    resumed = evaluator.finalize_pending_evaluation(pending, _case_input(), _metadata(),
                                                    _ground_truth(), adjudications=_adjudications())
    direct, _ = _evaluator(_prediction())
    expected = direct.evaluate_response_record(_case_input(), _metadata(), _ground_truth(),
        final_response='Answer', efficiency=EfficiencyObservation(100, 20, 2, 1, 3.0),
        adjudications=_adjudications())
    assert resumed.to_dict() == expected.to_dict()


def test_response_level_f_app_never_shrinks_by_branch_and_differs_from_precision():
    evaluator, _ = _evaluator(_prediction())
    result = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=_adjudications(accept_extra_mean=False),
    )
    assert result.metrics is not None
    assert result.metrics.scientific_findings.finding_precision == pytest.approx(2 / 3)
    assert result.metrics.o_f_consistency.c_score == pytest.approx(2 / 3)
    assert all(
        diagnostic.applicable_finding_count == 3
        for diagnostic in result.consistency_branch_diagnostics
    )

    irrelevant_prediction = _prediction()
    evaluator, _ = _evaluator(irrelevant_prediction, irrelevant_ids=("extra",))
    irrelevant_result = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_findings={
                ("branch_peak", "identity"): AdjudicationStatus.REJECTED,
                ("branch_peak", "strength"): AdjudicationStatus.REJECTED,
            }
        ),
    )
    assert irrelevant_result.metrics is not None
    assert irrelevant_result.metrics.scientific_findings.finding_precision == pytest.approx(1.0)
    assert irrelevant_result.metrics.o_f_consistency.c_score == 1.0
    assert irrelevant_result.f_app_prediction_ids == ("identity", "strength")


def test_missing_and_conflicting_o_have_frozen_o_urs_and_c_semantics():
    missing_evaluator, _ = _evaluator(
        _prediction(property_status=ExtractionStatus.MISSING)
    )
    missing = missing_evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=_adjudications(),
    )
    assert missing.metrics is not None
    assert missing.metrics.scientific_operationalization.o_score == 0.5
    assert missing.metrics.scientific_operationalization.urs == 0.0
    assert missing.compatible_o_branches == ("branch_peak", "branch_mean")
    assert missing.metrics.o_f_consistency.c_score is None
    assert missing.metrics.o_f_consistency.branch_alignment is None
    assert missing.metrics.o_f_consistency.operationalization_determinate is False
    assert missing.metrics.scientific_findings.finding_precision == 1.0
    assert missing.consistency_branch_diagnostics == ()
    assert all(
        diagnostic.applicable_finding_count == 3
        for diagnostic in missing.consistency_branch_diagnostics
    )

    conflicting_evaluator, _ = _evaluator(
        _prediction(property_status=ExtractionStatus.CONFLICTING)
    )
    conflicting = conflicting_evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=_adjudications(),
    )
    assert conflicting.metrics is not None
    assert conflicting.compatible_o_branches == ()
    assert conflicting.metrics.o_f_consistency.c_score is None
    assert conflicting.metrics.o_f_consistency.branch_alignment is None


@pytest.mark.parametrize("include_supplement", [False, True])
def test_accepted_novel_o_adds_temporary_coherent_branch_without_mutating_gt(include_supplement):
    metadata = _metadata(responsibility="model_selected")
    frozen_gt = _ground_truth()
    prediction = ExtractedPrediction(
        operationalization=ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use watershed regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use median strength."
                ),
            )
        ),
        findings=(
            PredictedAtomicFinding("novel_identity", "The strongest watershed region is C.", "C"),
        ),
    )
    novel = AdjudicatedNovelBranch(
        operationalization={
            "operationalization_id": "novel_branch",
            "decisions": [
                {"dimension": "feature_definition", "statement": "Use watershed regions."},
                {"dimension": "property_measure", "statement": "Use median strength."},
            ],
        },
        findings={
            "operationalization_id": "novel_branch",
            "findings": [
                {
                    "finding_id": "novel_identity_ref",
                    "category": "existence_or_identity",
                    "statement": "The strongest watershed region is C.",
                    "importance": "core",
                    "value": "C",
                }
            ],
        },
    )
    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=_trusted_novel_materializer,
    )
    supplemental_calls = []
    if include_supplement:
        def supplement(record):
            supplemental_calls.append(record)
            return {**record["execution"]["G_of_O"], "independent_extra_statistic": 42}
        evaluator.supplemental_materializer = supplement
    result = evaluator.evaluate_response(
        _case_input(),
        metadata,
        frozen_gt,
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_operationalization=NovelOperationalizationAdjudication(
                AdjudicationStatus.ACCEPTED,
                novel,
                adjudicated_materialization_id="mat:trusted:novel",
                adjudicated_g_of_o_sha256=hashlib.sha256(
                    json.dumps(
                        {"source": "trusted-fixture", "result": "C"},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            novel_findings={
                ("branch_peak", "novel_identity"): AdjudicationStatus.REJECTED,
                ("branch_mean", "novel_identity"): AdjudicationStatus.REJECTED,
            },
        ),
    )
    assert result.metrics is not None
    assert result.metrics.scientific_operationalization.best_o_branches == (
        "novel_branch",
    )
    assert result.metrics.scientific_operationalization.urs == 1.0
    assert result.metrics.scientific_findings.best_finding_branches == (
        "novel_branch",
    )
    assert result.compatible_o_branches == ("novel_branch",)
    assert result.metrics.o_f_consistency.c_score == 1.0
    assert len(frozen_gt.acceptable_operationalizations) == 2
    if include_supplement:
        assert len(supplemental_calls) == 1
        assert supplemental_calls[0]["parameters"]["property_measure"] == "Use median strength."
        assert supplemental_calls[0]["execution"]["G_of_O"] == {"source": "trusted-fixture", "result": "C"}


def test_development_novel_o_ignores_injected_g_of_o_without_trusted_materializer():
    metadata, prediction, branch = _novel_fixture()
    evaluator, _ = _evaluator(prediction, StubMatcher())
    with pytest.raises(EvaluationPendingAdjudication, match="trusted materializer"):
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.ACCEPTED, branch
                )
            ),
        )


def test_novel_o_is_materialized_before_scientific_verdict_is_requested():
    metadata, prediction, branch = _novel_fixture()
    calls = []

    def materializer(bundle):
        calls.append(bundle.operationalization_id)
        return _trusted_novel_materializer(bundle)

    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=materializer,
    )
    with pytest.raises(
        EvaluationPendingAdjudication,
        match="blind scientific adjudication",
    ) as exc_info:
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.UNRESOLVED,
                    branch,
                )
            ),
        )
    assert calls == ["novel_branch"]
    assert exc_info.value.pending_type == "novel_operationalization_scientific_adjudication"


def test_novel_o_resolved_verdict_requires_actual_g_of_o_binding():
    metadata, prediction, branch = _novel_fixture()
    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=_trusted_novel_materializer,
    )
    with pytest.raises(EvaluationConfigurationError, match="not bound"):
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.ACCEPTED,
                    branch,
                )
            ),
        )


def test_novel_o_materializer_requires_execution_provenance():
    metadata, prediction, branch = _novel_fixture()

    def missing_provenance(bundle):
        artifact = _trusted_novel_materializer(bundle)
        artifact["execution"].pop("data_provenance")
        return artifact

    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=missing_provenance,
    )
    with pytest.raises(EvaluationConfigurationError, match="data provenance"):
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.ACCEPTED, branch
                )
            ),
        )


def test_novel_o_unsupported_materialization_stays_pending():
    metadata, prediction, branch = _novel_fixture()

    def unsupported(bundle):
        artifact = _trusted_novel_materializer(bundle)
        artifact["execution"]["status"] = "MATERIALIZATION_UNSUPPORTED"
        return artifact

    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=unsupported,
    )
    with pytest.raises(EvaluationPendingAdjudication, match=r"no materialized G\(O\)"):
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.ACCEPTED, branch
                )
            ),
        )


def test_uncertain_semantic_match_blocks_formal_finalization():
    prediction = _prediction()
    matcher = StubMatcher(
        equivalent_pairs=[
            ("The strongest region is B.", "Region B is the strongest.")
        ],
        uncertain_pairs=[("Use connected regions.", "Use connected regions.")],
    )
    evaluator, _ = _evaluator(prediction, matcher)
    with pytest.raises(EvaluationPendingAdjudication, match="UNCERTAIN"):
        evaluator.evaluate_response(
            _case_input(),
            _metadata(),
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=_adjudications(),
        )


def test_empty_findings_score_zero_and_unmatched_findings_require_adjudication():
    empty_prediction = ExtractedPrediction(
        operationalization=ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use connected regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use mean strength."
                ),
            )
        ),
        findings=(),
    )
    evaluator, _ = _evaluator(empty_prediction, StubMatcher())
    empty = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="Only a method was reported.",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
    )
    assert empty.metrics is not None
    assert empty.metrics.scientific_findings.finding_precision == 0
    assert empty.metrics.scientific_findings.core_finding_recall == 0
    assert empty.metrics.o_f_consistency.c_score is None
    assert empty.metrics.o_f_consistency.branch_alignment is None

    evaluator, _ = _evaluator(_prediction())
    with pytest.raises(EvaluationPendingAdjudication, match="GT-outside Finding"):
        evaluator.evaluate_response(
            _case_input(),
            _metadata(),
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        )


def test_value_verification_uses_tolerance_spatial_shape_and_units():
    scalar = ReferenceFinding.model_validate(
        {
            "finding_id": "scalar",
            "category": "quantity",
            "statement": "value",
            "importance": "core",
            "value": 10.0,
            "unit": "m/s",
            "verification": {"absolute_tolerance": 0.1, "relative_tolerance": 0.01},
        }
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p", "value", 10.15, "m/s"), scalar
    )
    with pytest.raises(ValueError, match="finite"):
        PredictedAtomicFinding("p", "value", float("nan"), "m/s")
    spatial = ReferenceFinding.model_validate(
        {
            "finding_id": "location",
            "category": "location",
            "statement": "location",
            "importance": "core",
            "value": [1.0, 2.0, 3.0],
            "verification": {"spatial_tolerance": 0.1},
        }
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p", "location", [1.02, 2.01, 3.0]), spatial
    )
    assert not verify_finding_value(
        PredictedAtomicFinding("p", "location", [1.0, 2.0]), spatial
    )
    with pytest.raises(EvaluationPendingAdjudication, match="unit relationship"):
        verify_finding_value(PredictedAtomicFinding("p", "value", 10.0), scalar)
    assert verify_finding_value(
        PredictedAtomicFinding("p", "value", 36.0, "km/h"),
        scalar,
        unit_converter=lambda value, source, target: (
            float(value) / 3.6 if (source, target) == ("km/h", "m/s") else None
        ),
    )

    integer_valued_continuous = ReferenceFinding.model_validate(
        {
            "finding_id": "integer-valued-continuous",
            "category": "quantity",
            "statement": "a continuous quantity",
            "importance": "core",
            "value": 10,
            "verification": {"absolute_tolerance": 0.1},
        }
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p", "a continuous quantity", 10.05),
        integer_valued_continuous,
    )


def test_predicted_finding_value_boundary_is_closed():
    for value in (True, {}, [[1.0]], [1, "2"], float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PredictedAtomicFinding("invalid", "claim", value)
    assert PredictedAtomicFinding("int-list", "claim", [1, 2, 3]).value == [1.0, 2.0, 3.0]
    assert PredictedAtomicFinding("mixed-numeric-list", "claim", [1.0, 2, 3.0]).value == [1.0, 2.0, 3.0]
    assert PredictedAtomicFinding("float-list", "claim", [1.0, 2.0]).value == [1.0, 2.0]
    assert PredictedAtomicFinding("string-list", "claim", ["A", "B"]).value == ["A", "B"]


def test_finding_verification_mode_separates_semantic_and_numeric_judgment():
    evaluator, _ = _evaluator(_prediction())
    record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=_adjudications(),
    )
    strength = next(
        item
        for item in record.semantic_matches
        if item.request.reference_id == "mean_strength"
        and item.request.predicted_id == "strength"
    )
    assert strength.result is SemanticMatchResult.MATCH
    assert strength.request.verification_mode is FindingVerificationMode.SCALAR_TOLERANCE
    verification = next(
        item for item in record.value_verifications if item.reference_id == "mean_strength"
    )
    assert verification.verified is True


def test_numeric_mismatch_remains_semantic_match_and_is_not_gt_outside():
    prediction = replace(
        _prediction(),
        findings=(PredictedAtomicFinding("strength", "Its mean strength is 2.0.", 9.0),),
    )
    evaluator, _ = _evaluator(prediction)
    record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=_adjudications(),
    )
    semantic = next(
        item
        for item in record.semantic_matches
        if item.request.reference_id == "mean_strength"
        and item.request.predicted_id == "strength"
    )
    assert semantic.result is SemanticMatchResult.MATCH
    verification = next(
        item for item in record.value_verifications if item.reference_id == "mean_strength"
    )
    assert verification.verified is False
    assert record.result.finding_branch_diagnostics[1].accepted_gt_outside_prediction_ids == ()


def test_pending_evaluation_record_is_persisted_and_round_trips(tmp_path):
    evaluator, _ = _evaluator(_prediction())
    pending = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
    )
    assert isinstance(pending, PendingCaseEvaluationRecord)
    assert pending.pending_type == "gt_outside_finding"
    path = tmp_path / "nested" / "pending.json"
    save_case_evaluation_record(pending, path)
    restored = load_case_evaluation_record(path)
    assert isinstance(restored, PendingCaseEvaluationRecord)
    assert restored.to_dict() == pending.to_dict()


def test_pending_finalization_reuses_extraction_and_stored_judgments():
    evaluator, extractor = _evaluator(_prediction())
    pending = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
    )
    assert isinstance(pending, PendingCaseEvaluationRecord)
    matcher = evaluator.semantic_matcher
    before = len(matcher.requests)
    finalized = evaluator.finalize_pending_evaluation(
        pending,
        _case_input(),
        _metadata(),
        _ground_truth(),
        adjudications=_adjudications(),
    )
    assert finalized.result.run_status.value == "COMPLETED"
    assert len(extractor.requests) == 1
    assert len(matcher.requests) > before
    # The continuation may compare a newly admitted GT branch, but must not
    # invoke the concrete matcher again for the persisted prefix.
    old_requests = [judgment.request for judgment in pending.semantic_matches]
    assert not any(request in old_requests for request in matcher.requests[before:])


def test_run_adjudication_wrapper_round_trip_and_identity(tmp_path):
    wrapper = RunEvaluationAdjudication("run-1", "case_a", _adjudications())
    path = tmp_path / "run-adjudication.json"
    save_run_evaluation_adjudication(wrapper, path)
    assert load_run_evaluation_adjudication(path) == wrapper


def test_adjudications_have_serializable_json_contract(tmp_path):
    path = tmp_path / "adjudications" / "case.json"
    save_evaluation_adjudications(_adjudications(), path)
    assert load_evaluation_adjudications(path) == _adjudications()


def _run(status: RunStatus, *, final_response: str | None) -> RunRecord:
    return RunRecord(
        run_id="run",
        case_id="case_a",
        trial_index=1,
        ordering_seed=1,
        case_execution_position=0,
        provider="test",
        model_id="test",
        model_configuration={},
        target_fingerprint="target",
        experiment_id="experiment",
        benchmark_release_id="release",
        system_prompt_version="prompt",
        case_presentation_version="case",
        python_tool_spec_version="tool",
        runner_version="runner",
        runtime_environment_fingerprint={},
        formal_mode=True,
        network_isolation_active=True,
        run_status=status,
        final_response=final_response,
        input_tokens=10,
        output_tokens=2,
        model_turn_count=1,
        python_execution_count=0,
        wall_clock_time=0.5,
    )


def test_run_status_scoring():
    evaluator, extractor = _evaluator(_prediction())
    noncompletion = evaluator.evaluate_run(
        _run(RunStatus.MODEL_NONCOMPLETION, final_response=None),
        _case_input(),
        _metadata(),
        _ground_truth(),
    )
    assert noncompletion.eligible_for_scientific_aggregation
    assert noncompletion.metrics is not None
    assert noncompletion.metrics.scientific_operationalization.o_score == 0
    assert noncompletion.metrics.scientific_operationalization.urs == 0
    assert noncompletion.metrics.scientific_findings.finding_precision == 0
    assert noncompletion.metrics.o_f_consistency.branch_alignment is None
    assert extractor.requests == []

    infrastructure = evaluator.evaluate_run(
        _run(RunStatus.INFRASTRUCTURE_INVALID, final_response=None),
        _case_input(),
        _metadata(),
        _ground_truth(),
    )
    assert not infrastructure.eligible_for_scientific_aggregation
    assert infrastructure.metrics is None

def test_run_identity_is_preserved_into_evaluation_and_trial_conversion():
    evaluator, _ = _evaluator(_prediction())
    record = evaluator.evaluate_run_record(
        _run(RunStatus.COMPLETED, final_response="response"),
        _case_input(),
        _metadata(),
        _ground_truth(),
        adjudications=_adjudications(),
    )
    assert record.run_id == "run"
    assert record.trial_index == 1
    assert record.experiment_id == "experiment"
    assert record.target_fingerprint == "target"
    assert record.benchmark_release_id == "release"
    assert record.formal_mode is True
    trial = TrialEvaluation.from_evaluation_record(record)
    assert trial.trial_index == 1
    assert trial.experiment_id == "experiment"
    assert trial.target_fingerprint == "target"
    assert trial.benchmark_release_id == "release"
    assert trial.formal_mode is True


def test_formal_trial_aggregation_uses_three_valid_trials_and_case_level_efficiency():
    evaluator, _ = _evaluator(_prediction())
    completed = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    noncompletion = evaluator.evaluate_run(
        _run(RunStatus.MODEL_NONCOMPLETION, final_response=None),
        _case_input(),
        _metadata(),
        _ground_truth(),
    )
    infrastructure = evaluator.evaluate_run(
        _run(RunStatus.INFRASTRUCTURE_INVALID, final_response=None),
        _case_input(),
        _metadata(),
        _ground_truth(),
    )

    aggregate = aggregate_trials_for_case(
        (
            TrialEvaluation(1, completed, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True),
            TrialEvaluation(2, noncompletion, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True),
            TrialEvaluation(3, completed, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True),
            # A retry for trial 1 is retained in the audit trail, but is not a
            # fourth scientific observation.
            TrialEvaluation(1, infrastructure, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True),
        )
    )
    assert aggregate.trial_count == 3
    assert aggregate.infrastructure_invalid_trial_count == 1
    assert aggregate.model_noncompletion_trial_count == 1
    assert aggregate.metrics.o_score == pytest.approx(2 / 3)
    assert aggregate.metrics.urs == pytest.approx(2 / 3)
    assert aggregate.metrics.c_score == 1.0
    assert aggregate.metrics.metric_trial_denominators["c_score"]["finalized_trial_n"] == 2
    assert aggregate.metrics.metric_trial_denominators["c_score"]["coverage"] == pytest.approx(2 / 3)
    assert aggregate.metrics.determinate_o_trial_count == 2
    assert aggregate.metrics.branch_alignment == 1.0
    assert aggregate.metrics.efficiency.input_tokens == pytest.approx(50 / 3)
    assert aggregate.metrics.efficiency.wall_clock_time == pytest.approx(7 / 6)
    restored = type(aggregate).from_dict(aggregate.to_dict())
    assert restored.to_dict() == aggregate.to_dict()


def test_formal_trial_aggregation_keeps_o1_urs_not_applicable():
    evaluator, _ = _evaluator(_prediction())
    base_gt = _ground_truth()
    o1_ground_truth = base_gt.model_copy(
        update={
            "acceptable_operationalizations": (base_gt.acceptable_operationalizations[1],),
            "findings_by_operationalization": (base_gt.findings_by_operationalization[1],),
        }
    )
    one_trial = evaluator.evaluate_response(
        _case_input(),
        _metadata(responsibility="user_specified"),
        o1_ground_truth,
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    aggregate = aggregate_trials_for_case(
        tuple(TrialEvaluation(index, one_trial) for index in (1, 2, 3))
    )
    assert aggregate.condition == "O1-F1"
    assert aggregate.metrics.urs is None


@pytest.mark.parametrize("contract", [{}, {"adequate_core_sets": [["width"]], "supporting_roles": ["center"]}])
def test_empty_finding_requirement_contract_keeps_fixed_core_scoring(contract):
    """An F1 SEC serializes a non-applicable F2 contract as an empty object."""

    evaluator, _ = _evaluator(
        _prediction(),
        finding_requirement_contract=contract,
    )
    result = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )

    assert result.metrics.scientific_findings.finding_recall_mode == "FIXED_REFERENCE_CORE"
    assert result.metrics.scientific_findings.core_finding_recall is not None
    assert (
        result.metrics.scientific_findings.finding_requirement_recall
        == result.metrics.scientific_findings.core_finding_recall
    )
    assert result.metrics.scientific_findings.adequate_core_complete is None


def test_f2_n3_adequate_core_is_a_rate_not_all_boolean():
    evaluator, _ = _evaluator(_prediction())
    completed = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    f2_findings = replace(
        completed.metrics.scientific_findings,
        core_finding_recall=None,
        finding_recall_mode="SEMANTIC_ADEQUATE_CORE",
        finding_requirement_recall=1.0,
        adequate_core_complete=True,
    )
    f2_metrics = replace(
        completed.metrics,
        condition="O1-F2",
        scientific_findings=f2_findings,
    )
    f2_result = replace(completed, condition="O1-F2", metrics=f2_metrics)
    incomplete = replace(
        f2_result,
        metrics=replace(f2_metrics, scientific_findings=replace(f2_findings, adequate_core_complete=False)),
    )
    aggregate = aggregate_trials_for_case(
        (
            TrialEvaluation(1, f2_result),
            TrialEvaluation(2, incomplete),
            TrialEvaluation(3, f2_result),
        )
    )
    assert aggregate.metrics.core_finding_recall is None
    assert aggregate.metrics.finding_requirement_recall == 1.0
    assert aggregate.metrics.adequate_core_complete is False
    assert aggregate.metrics.adequate_core_complete_rate == pytest.approx(2 / 3)


def test_formal_trial_aggregation_requires_complete_non_pending_group():
    evaluator, _ = _evaluator(_prediction())
    completed = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    with pytest.raises(EvaluationConfigurationError, match="exactly three valid"):
        aggregate_trials_for_case(
            (TrialEvaluation(1, completed), TrialEvaluation(2, completed))
        )
    with pytest.raises(EvaluationPendingAdjudication, match="pending trial"):
        aggregate_trials_for_case(
            (
                TrialEvaluation(1, completed),
                TrialEvaluation(2, completed),
                TrialEvaluation(3, pending=True, pending_reason="UNCERTAIN match"),
            )
        )
    with pytest.raises(EvaluationConfigurationError, match="unique trial_index"):
        aggregate_trials_for_case(
            (
                TrialEvaluation(1, completed),
                TrialEvaluation(1, completed),
                TrialEvaluation(2, completed),
            )
        )


def test_formal_case_aggregation_weights_cases_not_raw_trials():
    evaluator, _ = _evaluator(_prediction())
    digest = evaluation_manifest_digest(evaluator.evaluation_manifest)
    completed = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    noncompletion = evaluator.evaluate_run(
        _run(RunStatus.MODEL_NONCOMPLETION, final_response=None),
        _case_input(),
        _metadata(),
        _ground_truth(),
    )
    first_case = aggregate_trials_for_case(
        (
            TrialEvaluation(1, completed, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True, evaluation_manifest_digest=digest),
            TrialEvaluation(2, noncompletion, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True, evaluation_manifest_digest=digest),
            TrialEvaluation(3, completed, experiment_id="exp", target_fingerprint="target", benchmark_release_id="release", formal_mode=True, evaluation_manifest_digest=digest),
        )
    )
    second_case_trials = tuple(
        TrialEvaluation(
            index,
            replace(completed, case_id="case_b", metrics=replace(completed.metrics, case_id="case_b")),
            experiment_id="exp",
            target_fingerprint="target",
                benchmark_release_id="release",
                formal_mode=True,
                evaluation_manifest_digest=digest,
            )
        for index in (1, 2, 3)
    )
    second_case = aggregate_trials_for_case(second_case_trials)
    summary = aggregate_cases((first_case, second_case))
    assert summary["overall"]["eligible_case_count"] == 2
    assert summary["overall"]["mean_o_score"] == pytest.approx(5 / 6)
    assert summary["overall"]["mean_c_score"] == 1.0
    coverage = summary["overall"]["metric_trial_denominators"]["c_score"]
    assert coverage["eligible_trial_n"] == 6
    assert coverage["applicable_trial_n"] == coverage["finalized_trial_n"] == 5
    assert coverage["coverage"] == pytest.approx(5 / 6)
    assert "O1-F1" not in summary["by_condition"]
    assert summary["by_condition"]["O2-F1"]["eligible_case_count"] == 2
    assert summary["overall"]["efficiency"]["input_tokens"]["median"] == pytest.approx(
        ((50 / 3) + 20) / 2
    )
    assert summary["overall"]["efficiency"]["input_tokens"]["mean"] == pytest.approx(
        ((50 / 3) + 20) / 2
    )


@pytest.mark.parametrize("identity_field", ["target_fingerprint", "experiment_id"])
def test_formal_case_aggregation_rejects_mixed_model_identity(identity_field):
    evaluator, _ = _evaluator(_prediction())
    digest = evaluation_manifest_digest(evaluator.evaluation_manifest)
    completed = evaluator.evaluate_response(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(20, 4, 2, 1, 1.5),
        adjudications=_adjudications(),
    )
    first = aggregate_trials_for_case(
        tuple(
                TrialEvaluation(index, completed, experiment_id="exp-a", target_fingerprint="target-a", benchmark_release_id="release", formal_mode=True, evaluation_manifest_digest=digest)
            for index in (1, 2, 3)
        )
    )
    second = aggregate_trials_for_case(
        tuple(
            TrialEvaluation(
                index,
                replace(completed, case_id="case_b", metrics=replace(completed.metrics, case_id="case_b")),
                experiment_id=("exp-b" if identity_field == "experiment_id" else "exp-a"),
                target_fingerprint=("target-b" if identity_field == "target_fingerprint" else "target-a"),
                benchmark_release_id="release",
                formal_mode=True,
                evaluation_manifest_digest=digest,
            )
            for index in (1, 2, 3)
        )
    )
    with pytest.raises(EvaluationConfigurationError, match="formal benchmark aggregation"):
        aggregate_cases((first, second))


def test_one_to_one_assignment_protects_core_credit():
    metadata = _metadata(responsibility="user_specified")
    gt = GroundTruth.model_validate(
        {
            "dataset_id": "dataset_a",
            "case_id": "case_a",
            "case_family_id": "family_a",
            "acceptable_operationalizations": [
                {
                    "operationalization_id": "only",
                    "decisions": [
                        {"dimension": "feature_definition", "statement": "Use connected regions."},
                        {"dimension": "property_measure", "statement": "Use mean strength."},
                    ],
                }
            ],
            "findings_by_operationalization": [
                {
                    "operationalization_id": "only",
                    "findings": [
                        {
                            "finding_id": "core",
                            "category": "property",
                            "statement": "The region has the reported property.",
                            "importance": "core",
                        },
                        {
                            "finding_id": "supporting",
                            "category": "property",
                            "statement": "The region has a supporting property.",
                            "importance": "supporting",
                        },
                    ],
                }
            ],
        }
    )
    prediction = ExtractedPrediction(
        ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use connected regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use mean strength."
                ),
            )
        ),
        (PredictedAtomicFinding("p", "A broad property statement."),),
    )
    matcher = StubMatcher(
        equivalent_pairs=[
            ("A broad property statement.", "The region has the reported property."),
            ("A broad property statement.", "The region has a supporting property."),
        ]
    )
    evaluator, _ = _evaluator(prediction, matcher)
    result = evaluator.evaluate_response(
        _case_input(),
        metadata,
        gt,
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
    )
    diagnostic = result.finding_branch_diagnostics[0]
    assert diagnostic.matched_pairs == (("p", "core"),)
    assert diagnostic.matched_core_finding_count == 1
    assert diagnostic.valid_predicted_finding_count == 1


def test_deduplication_preserves_conflicting_values_and_collapses_true_repeat():
    base = _prediction()
    conflicting = replace(
        base,
        findings=(
            PredictedAtomicFinding("v1", "Its mean strength is 2.0.", 0.73),
            PredictedAtomicFinding("v2", "Its mean strength is 2.0.", 0.91),
        ),
    )
    evaluator, _ = _evaluator(conflicting, StubMatcher())
    record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_findings={
                ("branch_peak", "v1"): AdjudicationStatus.REJECTED,
                ("branch_peak", "v2"): AdjudicationStatus.REJECTED,
            }
        ),
    )
    assert record.result.deduplicated_prediction_ids == ("v1", "v2")
    assert tuple(
        item.representative_prediction_id for item in record.deduplication_mapping
    ) == ("v1", "v2")
    assert len(evaluator.eligibility_judge.requests) == 2

    repeated = replace(
        base,
        findings=(
            PredictedAtomicFinding("v1", "Its mean strength is 2.0.", 2.05),
            PredictedAtomicFinding("v2", "Its mean strength is 2.0.", 2.05),
        ),
    )
    evaluator, _ = _evaluator(repeated, StubMatcher())
    repeated_record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_findings={
                ("branch_peak", "v1"): AdjudicationStatus.REJECTED
            }
        ),
    )
    assert repeated_record.result.deduplicated_prediction_ids == ("v1",)
    assert repeated_record.deduplication_mapping[1].representative_prediction_id == "v1"
    assert len(evaluator.eligibility_judge.requests) == 1


def test_accepted_novel_o_payload_must_match_extracted_decisions():
    metadata = _metadata(responsibility="model_selected")
    prediction = ExtractedPrediction(
        ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use watershed regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use median strength."
                ),
            )
        ),
        (),
    )
    mismatched = AdjudicatedNovelBranch(
        operationalization={
            "operationalization_id": "novel_mismatch",
            "decisions": [
                {"dimension": "feature_definition", "statement": "Use connected regions."},
                {"dimension": "property_measure", "statement": "Use median strength."},
            ],
        },
        findings={
            "operationalization_id": "novel_mismatch",
            "findings": [
                {
                    "finding_id": "core",
                    "category": "property",
                    "statement": "A valid result exists.",
                    "importance": "core",
                }
            ],
        },
    )
    evaluator, _ = _evaluator(prediction, StubMatcher())
    with pytest.raises(EvaluationConfigurationError, match="does not match"):
        evaluator.evaluate_response(
            _case_input(),
            metadata,
            _ground_truth(),
            final_response="response",
            efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
            adjudications=EvaluationAdjudications(
                novel_operationalization=NovelOperationalizationAdjudication(
                    AdjudicationStatus.ACCEPTED, mismatched
                )
            ),
        )


def test_novel_o_completion_exactly_preserving_extracted_statements_is_self_consistent():
    metadata = _metadata(responsibility="model_selected")
    prediction = ExtractedPrediction(
        ExtractedOperationalization(
            (
                ExtractedOperationalizationDecision(
                    "feature_definition", ExtractionStatus.EXTRACTED, "Use watershed regions."
                ),
                ExtractedOperationalizationDecision(
                    "property_measure", ExtractionStatus.EXTRACTED, "Use median strength."
                ),
            )
        ),
        (),
    )
    completed = OperationalizationBundle.model_validate(
        {
            "operationalization_id": "runtime_exact_completion",
            "decisions": [
                {"dimension": "feature_definition", "statement": "Use watershed regions."},
                {"dimension": "property_measure", "statement": "Use median strength."},
            ],
        }
    )
    # The semantic backend deliberately rejects all pairs.  The completion
    # check must nevertheless recognize that its statements are the exact
    # extracted response, while ordinary GT comparisons still use the backend.
    evaluator, _ = _evaluator(
        prediction,
        StubMatcher(),
        novel_o_materializer=_trusted_novel_materializer,
    )
    record = evaluator.evaluate_response_record(
        _case_input(),
        metadata,
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_operationalization=NovelOperationalizationAdjudication(
                AdjudicationStatus.UNRESOLVED,
                operationalization=completed,
            )
        ),
    )
    # Reaching materialization/adjudication proves the exact candidate was not
    # spuriously rejected by a second semantic self-comparison.
    assert isinstance(record, PendingCaseEvaluationRecord)
    assert record.pending_type == "novel_operationalization_scientific_adjudication"


def test_textual_value_is_semantic_and_matcher_receives_complete_case_context():
    prediction = replace(
        _prediction(),
        findings=(
            PredictedAtomicFinding(
                "identity", "Region B is the strongest.", "Region B"
            ),
        ),
    )
    matcher = StubMatcher(
        equivalent_pairs=[("Region B is the strongest.", "The strongest region is B.")]
    )
    evaluator, _ = _evaluator(prediction, matcher)
    record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
        adjudications=EvaluationAdjudications(
            novel_findings={
                ("branch_peak", "identity"): AdjudicationStatus.REJECTED
            }
        ),
    )
    identity_match = next(
        item
        for item in record.semantic_matches
        if item.request.reference_id == "mean_identity"
    )
    assert identity_match.result == SemanticMatchResult.MATCH
    assert identity_match.request.predicted_value == "Region B"
    assert identity_match.request.reference_value == "B"
    assert identity_match.request.scientific_question == _case_input().scientific_question
    assert identity_match.request.case_context["physical_setting"] == "test flow"
    assert identity_match.request.finding_goal == _metadata().finding_goal
    assert not hasattr(identity_match.request, "acceptable_operationalizations")
    verification = next(
        item
        for item in record.value_verifications
        if item.reference_id == "mean_identity"
    )
    assert verification.rule == "semantic_only"
    assert verification.verified


def test_peak_and_mean_location_representations_are_not_silently_equivalent():
    request = SemanticMatchRequest(
        purpose=SemanticMatchPurpose.FINDING,
        predicted_statement="The peak-speed point is at (0.2, 1.4, 0.2).",
        reference_statement="The mean spatial location is (0.5, 2.0, 0.5).",
        predicted_id="p",
        reference_id="r",
    )
    # A permissive matcher cannot override this explicit representation guard.
    class Permissive:
        def match(self, _request):
            return SemanticMatchResult.MATCH
    assert _match(Permissive(), request) is SemanticMatchResult.NO_MATCH


def test_numeric_list_without_spatial_tolerance_does_not_infer_euclidean_rule():
    structured = ReferenceFinding.model_validate(
        {
            "finding_id": "extent",
            "category": "characterization",
            "statement": "The coordinate span has the reported characterization.",
            "importance": "core",
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p", "Equivalent characterization.", [9.0, 9.0, 9.0]),
        structured,
    )
    continuous_without_rule = ReferenceFinding.model_validate(
        {
            "finding_id": "scalar_semantic",
            "category": "quantity",
            "statement": "The scalar has the reported qualitative relation.",
            "importance": "core",
            "value": 1.5,
        }
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p2", "Equivalent relation.", 99.0),
        continuous_without_rule,
    )


def test_integer_without_explicit_policy_does_not_infer_exact_discrete_rule():
    count = ReferenceFinding.model_validate(
        {
            "finding_id": "count",
            "category": "quantity",
            "statement": "The region count is reported.",
            "importance": "supporting",
            "value": 8,
        }
    )
    assert finding_verification_mode(count) is FindingVerificationMode.SEMANTIC_ONLY
    assert verify_finding_value(
        PredictedAtomicFinding("p", "An equivalent semantic count claim.", 9), count
    )
    policy = {
        "verification_mode": "exact_discrete_numeric",
        "verification_parameters": {
            "absolute_tolerance": None,
            "relative_tolerance": None,
            "spatial_tolerance": None,
        },
    }
    assert not verify_finding_value(
        PredictedAtomicFinding("p", "The region count is 9.", 9),
        count,
        explicit_policy=policy,
    )
    assert verify_finding_value(
        PredictedAtomicFinding("p", "The region count is 8.", 8),
        count,
        explicit_policy=policy,
    )


def test_explicit_policy_conflicting_with_gt_tolerance_fails_closed():
    scalar = ReferenceFinding.model_validate(
        {
            "finding_id": "strength",
            "category": "quantity",
            "statement": "The strength is reported.",
            "importance": "core",
            "value": 1.0,
            "verification": {"absolute_tolerance": 0.05},
        }
    )
    with pytest.raises(EvaluationConfigurationError, match="conflicts with GroundTruth"):
        verify_finding_value(
            PredictedAtomicFinding("p", "The strength is 1.0.", 1.0),
            scalar,
            explicit_policy={
                "verification_mode": "scalar_tolerance",
                "verification_parameters": {
                    "absolute_tolerance": 0.1,
                    "relative_tolerance": None,
                    "spatial_tolerance": None,
                },
            },
        )


def test_extent_componentwise_verification_is_explicit_and_not_euclidean():
    extent = ReferenceFinding.model_validate(
        {
            "finding_id": "extent",
            "category": "quantity",
            "statement": "The axis-aligned extent is reported.",
            "importance": "core",
            "value": [1.0, 2.0, 3.0],
            "verification": {"spatial_tolerance": 0.1},
        }
    )
    policy = {
        "verification_mode": "componentwise_vector",
        "verification_parameters": {
            "absolute_tolerance": None,
            "relative_tolerance": None,
            "spatial_tolerance": 0.1,
        },
    }
    assert finding_verification_mode(extent, policy) is FindingVerificationMode.COMPONENTWISE_VECTOR
    # Each component is within tolerance even though the L2 distance is above
    # 0.1.  The explicit extent rule therefore must not use Euclidean distance.
    assert verify_finding_value(
        PredictedAtomicFinding("p", "The extent is reported.", [1.09, 2.09, 3.09]),
        extent,
        explicit_policy=policy,
    )
    assert not verify_finding_value(
        PredictedAtomicFinding("p", "The extent is reported.", [1.11, 2.0, 3.0]),
        extent,
        explicit_policy=policy,
    )


def test_coordinate_frame_alias_requires_case_reader_authority():
    location = ReferenceFinding.model_validate(
        {
            "finding_id": "location",
            "category": "location",
            "statement": "The location is reported in the dataset frame.",
            "importance": "core",
            "value": [1.0, 2.0, 3.0],
            "verification": {"spatial_tolerance": 0.01},
        }
    )
    prediction = PredictedAtomicFinding(
        "p", "The location is reported.", [1.0, 2.0, 3.0], "dataset coordinate frame"
    )
    policy = {
        "verification_mode": "spatial_euclidean",
        "verification_parameters": {
            "absolute_tolerance": None,
            "relative_tolerance": None,
            "spatial_tolerance": 0.01,
        },
        "unit_frame_authority": {"coordinate_or_numeric_scale": "CASE_READER_METADATA"},
    }
    assert verify_finding_value(prediction, location, explicit_policy=policy)
    with pytest.raises(EvaluationPendingAdjudication, match="unit relationship"):
        verify_finding_value(
            prediction,
            location,
            explicit_policy={
                **policy,
                "unit_frame_authority": {"coordinate_or_numeric_scale": "REFERENCE_FINDING_UNIT"},
            },
        )


def test_evaluation_record_round_trip_replays_without_semantic_components(tmp_path):
    evaluator, _ = _evaluator(_prediction())
    record = evaluator.evaluate_response_record(
        _case_input(),
        _metadata(),
        _ground_truth(),
        final_response="response",
        efficiency=EfficiencyObservation(100, 20, 2, 1, 3.0),
        adjudications=_adjudications(),
    )
    matcher_request_count = len(evaluator.semantic_matcher.requests)
    eligibility_request_count = len(evaluator.eligibility_judge.requests)
    path = tmp_path / "evaluation_record.json"
    save_case_evaluation_record(record, path)
    loaded = load_case_evaluation_record(path)
    replayed = loaded.replay()
    assert loaded.to_dict() == record.to_dict()
    assert replayed.to_dict() == record.result.to_dict()
    assert len(evaluator.semantic_matcher.requests) == matcher_request_count
    assert len(evaluator.eligibility_judge.requests) == eligibility_request_count
    assert loaded.evaluation_manifest.evaluator_backend.implementation_id == "test.stub"
    assert loaded.operationalization_semantic_matches["branch_mean"][
        "property_measure"
    ] == SemanticMatchResult.MATCH
    manifest_path = tmp_path / "evaluation_manifest.json"
    save_evaluation_manifest(record.evaluation_manifest, manifest_path)
    assert load_evaluation_manifest(manifest_path) == record.evaluation_manifest


def _f2_fixture():
    metadata = CaseConstructionMetadata.model_validate({
        **_metadata(responsibility="user_specified").model_dump(mode="json"),
        "finding_openness": "open",
        "explicit_finding_requirements": [],
    })
    gt = _ground_truth()
    gt = gt.model_copy(update={
        "acceptable_operationalizations": (gt.acceptable_operationalizations[1],),
        "findings_by_operationalization": (gt.findings_by_operationalization[1],),
    })
    contract = {
        "adequate_core_sets": [["width"]],
        "supporting_roles": ["center"],
        "reference_finding_role_map": {"mean_identity": "center", "mean_strength": "center"},
    }
    return metadata, gt, contract


def test_f2_novel_role_only_in_adequate_set_is_projected_and_can_complete_n3():
    from flowintentbench.finding_requirements import FindingRequirementContract
    metadata, gt, contract = _f2_fixture()
    normalized = FindingRequirementContract.from_mapping(contract)
    assert normalized.validate() == []
    assert normalized.allowed_roles == {"width", "center"}
    evaluator, _ = _evaluator(_prediction(), finding_requirement_contract=contract)
    pending = evaluator.evaluate_response_record(
        _case_input(), metadata, gt, final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0),
    )
    assert isinstance(pending, PendingCaseEvaluationRecord)
    assert pending.continuation_context["allowed_finding_roles"] == ["center", "width"]
    adjudications = replace(
        _adjudications(), novel_finding_roles={("branch_mean", "extra"): "width"}
    )
    record = evaluator.evaluate_response_record(
        _case_input(), metadata, gt, final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0), adjudications=adjudications,
    )
    completed = record.result
    assert record.replay().to_dict() == completed.to_dict()
    assert type(record).from_dict(record.to_dict()).replay().to_dict() == completed.to_dict()
    assert completed.metrics.scientific_findings.core_finding_recall is None
    assert completed.metrics.scientific_findings.finding_requirement_recall == 1.0
    assert completed.metrics.scientific_findings.adequate_core_complete is True
    failed = evaluator.evaluate_run(
        _run(RunStatus.MODEL_NONCOMPLETION, final_response=None), _case_input(), metadata, gt,
    )
    assert failed.metrics.scientific_findings.finding_recall_mode == "SEMANTIC_ADEQUATE_CORE"
    assert failed.metrics.scientific_findings.core_finding_recall is None
    assert failed.metrics.scientific_findings.finding_requirement_recall == 0
    assert failed.metrics.scientific_findings.adequate_core_complete is False
    aggregate = aggregate_trials_for_case(tuple(
        TrialEvaluation(index, result)
        for index, result in enumerate((completed, failed, completed), start=1)
    ))
    assert aggregate.trial_count == 3
    assert aggregate.model_noncompletion_trial_count == 1
    assert aggregate.metrics.finding_requirement_recall == pytest.approx(2 / 3)
    assert aggregate.metrics.adequate_core_complete_rate == pytest.approx(2 / 3)
    assert aggregate.metrics.core_finding_recall is None
    assert aggregate.metrics.c_score == 1
    assert aggregate.metrics.metric_trial_denominators["c_score"]["coverage"] == pytest.approx(2 / 3)
    assert type(aggregate).from_dict(aggregate.to_dict()).to_dict() == aggregate.to_dict()


@pytest.mark.parametrize("status", [ExtractionStatus.MISSING, ExtractionStatus.AMBIGUOUS, ExtractionStatus.CONFLICTING])
def test_incomplete_o_consistency_stays_null_through_replay_and_mixed_trials(status):
    evaluator, _ = _evaluator(_prediction(property_status=status))
    record = evaluator.evaluate_response_record(
        _case_input(), _metadata(), _ground_truth(), final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0), adjudications=_adjudications(),
    )
    assert type(record).from_dict(record.to_dict()).replay().metrics.o_f_consistency.c_score is None
    assert record.result.metrics.scientific_findings.finding_requirement_recall == 1
    completed_evaluator, _ = _evaluator(_prediction())
    complete = completed_evaluator.evaluate_response(
        _case_input(), _metadata(), _ground_truth(), final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0), adjudications=_adjudications(),
    )
    aggregate = aggregate_trials_for_case(tuple(
        TrialEvaluation(index, result)
        for index, result in enumerate((complete, record.result, complete), start=1)
    ))
    assert aggregate.metrics.c_score == 1
    assert aggregate.metrics.determinate_o_trial_rate == pytest.approx(2 / 3)
    assert aggregate.metrics.metric_trial_denominators["c_score"] == {
        "eligible_trial_n": 3, "applicable_trial_n": 2, "finalized_trial_n": 2,
        "not_applicable_trial_n": 1, "unavailable_trial_n": 0,
        "coverage": pytest.approx(2 / 3), "applicability_coverage": pytest.approx(2 / 3),
    }


@pytest.mark.parametrize("finding_value, expected_consistency", [("C", 1.0), ("wrong", 0.0)])
def test_scientifically_rejected_explicit_o_keeps_execution_based_consistency(finding_value, expected_consistency):
    metadata, prediction, branch = _novel_fixture()
    prediction = replace(prediction, findings=(replace(
        prediction.findings[0], value=finding_value,
        statement=f"The strongest watershed region is {finding_value}.",
    ),))
    evaluator, _ = _evaluator(prediction, StubMatcher(), novel_o_materializer=_trusted_novel_materializer)
    adjudications = EvaluationAdjudications(
        novel_operationalization=NovelOperationalizationAdjudication(
            AdjudicationStatus.REJECTED, branch,
            adjudicated_materialization_id="mat:trusted:novel",
            adjudicated_g_of_o_sha256=hashlib.sha256(json.dumps(
                {"source": "trusted-fixture", "result": "C"}, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
        ),
        novel_findings={
            ("branch_peak", "novel_identity"): AdjudicationStatus.REJECTED,
            ("branch_mean", "novel_identity"): AdjudicationStatus.REJECTED,
            ("novel_branch", "novel_identity"): AdjudicationStatus.REJECTED,
        },
    )
    record = evaluator.evaluate_response_record(
        _case_input(), metadata, _ground_truth(), final_response="response",
        efficiency=EfficiencyObservation(1, 1, 1, 0, 1.0), adjudications=adjudications,
    )
    metrics = record.result.metrics
    assert metrics.scientific_operationalization.o_score == 0
    assert metrics.scientific_findings.finding_precision == 0
    assert metrics.o_f_consistency.operationalization_determinate is True
    assert metrics.o_f_consistency.c_score == expected_consistency
    assert metrics.o_f_consistency.unavailable_reason is None
    assert record.replay().to_dict() == record.result.to_dict()
