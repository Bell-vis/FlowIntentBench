from copy import deepcopy
from types import SimpleNamespace

import pytest

from flowintentbench.answer_evidence import bind_value, value_is_bound, dimensionless_unit_proof
from flowintentbench.ground_truth import ReferenceFinding
from flowintentbench.trusted_scoring import score, from_outcome, empty_metrics, interval
from scripts.evaluate_trusted_metrics import summarize, paired_comparison, choose_paired, load_run_source, validated_cache, fatal_provider_error


def fixture():
    answer = "The method uses pressure. The weighting is uniform. CV is 0.5."
    names = ["feature_definition", "property_measure"]
    metadata = SimpleNamespace(principal_operationalization_dimensions=[SimpleNamespace(value=n) for n in names],
                               unresolved_operationalization_dimensions=[SimpleNamespace(value=names[1])])
    ref = ReferenceFinding.model_validate({"finding_id": "cv", "category": "quantity", "importance": "core",
        "statement": "CV", "value": 0.5, "verification": {"absolute_tolerance": 0.01}})
    gt = SimpleNamespace(acceptable_operationalizations=[SimpleNamespace(operationalization_id="o")],
        findings_by_operationalization=[SimpleNamespace(operationalization_id="o", findings=[ref])])
    material = {"finding_verification_policy": {"policies": [{"operationalization_id": "o", "finding_id": "cv",
        "verification_mode": "scalar_tolerance", "verification_parameters": ref.verification.model_dump()}]}}
    judgment = {"dimensions": [{"dimension": n, "status": "EXTRACTED", "evidence_text": "The method uses pressure.",
        "matches": {"o": True}} for n in names], "findings": [{"finding_id": "p", "statement": "CV",
        "evidence_text": "CV is 0.5.", "value": 0.5, "unit": None, "eligible": True,
        "matches": [{"branch_id": "o", "finding_id": "cv"}]}], "extraction_complete": True, "limitations": ""}
    return answer, metadata, gt, material, judgment


def calculate(data, condition="O2-F1"):
    return score(*data, condition=condition)


@pytest.mark.parametrize('font', ['mathbf', 'mathrm', 'mathsf'])
def test_braced_mean_and_numeric_font_preserve_ratio_units_only(font):
    formula = r'Mean 52.45; \frac{\sigma_{\rho,V}}{|\bar{\rho}_V|}=' + '\\' + font + '{0.9359}.'
    proof = dimensionless_unit_proof(.9359, formula, 'CV', '1')
    assert proof['source_unit'] == '1'
    assert proof['rule'] == 'SAME_FIELD_STD_OVER_MEAN_FORMULA_DIMENSIONLESS_FROM_SOURCE'
    assert dimensionless_unit_proof(.9359, formula, 'report the numerical coefficient', '1') == proof
    assert dimensionless_unit_proof(.9359, formula, 'density', 'kg/m3') is None
    assert dimensionless_unit_proof(52.45, formula, 'CV', '1') is None
    assert dimensionless_unit_proof(.9359, formula.replace(r'\bar{\rho}',r'\bar{p}'), 'CV', '1') is None
    assert dimensionless_unit_proof(.9359, formula.replace('{0.9359}', '{0.9359 Pa}'), 'CV', '1') is None


@pytest.mark.parametrize('value,expected', [(0.5, 1.0), (0.9, 0.0)])
def test_explicit_ratio_proof_survives_no_si_declaration_without_granting_correctness(value, expected):
    data = list(fixture())
    formula = r'\frac{\sigma_{\rho,V}}{|\bar{\rho}_V|}=' + '\\mathbf{' + str(value) + '}.'
    data[0] = data[0].replace('CV is 0.5.', formula + ' No SI units were assigned.')
    ref = data[2].findings_by_operationalization[0].findings[0]
    data[2].findings_by_operationalization[0].findings[0] = ref.model_copy(update={'unit':'1'})
    data[-1]['findings'][0].update(value=value,evidence_text=formula,
        unit_evidence_text='No SI units were assigned.')
    assert calculate(data)['metrics']['finding_precision']['value'] == expected


def test_value_source_binding_signs_vectors_exponents_and_wrong_values():
    assert value_is_bound([-0.552, -1.0, 17.2], "centroid (−0.552, −1.0, 17.2)")
    assert value_is_bound(1000, "Count: 1,000.")
    assert value_is_bound(0.002, "Value: 2e-3.")
    assert value_is_bound(0.002, "Value: 2 × 10^{-3}.")
    assert not value_is_bound(0.5, "Value: 0.9.")
    assert not value_is_bound([1, 1], "value 1")
    assert not value_is_bound([2, 1], "values 1, 2")
    assert bind_value("CV is 0.9.", "CV is 0.9.", 0.5)["reason"] == "VALUE_NOT_IN_SOURCE"


def test_decimal_source_binding_respects_binary64_representation_without_tolerance():
    assert value_is_bound(-0.03416195401846506, 'r=-0.034161954018465057')
    assert value_is_bound([0.001975950277636181, 0.05590228354481149],
                          '(0.0019759502776361812, 0.055902283544811492)')
    assert not value_is_bound(.5, 'value 0.5000000000000001')
    assert not value_is_bound(9007199254740992, 'count 9007199254740993')


def test_printed_coordinate_bounds_can_prove_extent_but_not_other_quantities():
    from flowintentbench.answer_evidence import coordinate_extent_derivation, is_coordinate_extent_statement
    quote = 'Coordinate bounds x=[-7.816, 14.362], y=[0, 8.328], z=[0.331, 5.724].'
    value = [22.178, 8.328, 5.393]
    assert bind_value(quote, quote, value)['status'] == 'UNBOUND'
    bound = bind_value(quote, quote, value, coordinate_extent=True)
    assert bound['derivation']['derived_extent'] == ['22.178', '8.328', '5.393']
    assert coordinate_extent_derivation([75,48,44], 'x=100–175, y=80–128, z=1–45')['axis_bounds']['x'] == ['100','175']
    assert coordinate_extent_derivation([.24,.6,.24], 'x=0.01–0.25, y=1.95–2.55, z=2.25–2.49')
    assert coordinate_extent_derivation([22,8.328,5.393], quote) is None
    assert coordinate_extent_derivation(value, quote + ' Other region x=[0,1].') is None
    assert coordinate_extent_derivation(value, quote.replace('y=[0, 8.328]', '')) is None
    assert is_coordinate_extent_statement('Coordinate-space extent')
    assert is_coordinate_extent_statement('spatial extents')
    assert not is_coordinate_extent_statement('coordinate centroid')
    assert not is_coordinate_extent_statement('spatial index extent')


def test_native_possessive_and_undeclared_unit_markup_preserves_scale():
    from flowintentbench.answer_evidence import unit_is_bound, is_native_scale_unit
    assert unit_is_bound('file coordinate units', "in the file’s coordinate units")
    assert unit_is_bound('mesh stored coordinate units', "the mesh's stored coordinate units")
    assert unit_is_bound('stored undeclared units', 'in the stored (undeclared) units')
    assert is_native_scale_unit('stored undeclared units')
    assert not unit_is_bound('m', 'in the stored (undeclared) units')


def test_extent_arithmetic_is_gated_by_claim_and_reference_quantity():
    answer, meta, gt, material, j = fixture()
    quote = 'Coordinate bounds x=[1,3], y=[4,7], z=[8,12].'
    answer = answer.replace('CV is 0.5.', quote)
    original = gt.findings_by_operationalization[0].findings[0]
    ref = original.model_copy(update={'statement':'Coordinate-space extent', 'value':[2.,3.,4.],
        'verification':original.verification.model_copy(update={'absolute_tolerance':None,'spatial_tolerance':1e-6})})
    gt.findings_by_operationalization[0].findings[0] = ref
    material['finding_verification_policy']['policies'][0].update(
        verification_mode='spatial_euclidean', verification_parameters=ref.verification.model_dump())
    j['findings'][0].update(statement='Spatial extent', value=[2.,3.,4.], evidence_text=quote)
    result=calculate((answer,meta,gt,material,j))
    assert result['metrics']['core_finding_recall']['value']==1
    assert 'PRINTED_COORDINATE_UPPER_MINUS_LOWER' in result['finding_checks'][0]['reason']
    gt.findings_by_operationalization[0].findings[0]=ref.model_copy(update={'statement':'Spatial centroid'})
    assert calculate((answer,meta,gt,material,j))['metrics']['core_finding_recall']['value'] is None


def test_quote_initial_function_word_case_preserves_scientific_case_and_offsets():
    from flowintentbench.answer_evidence import bind_quote
    answer='A volume-weighted fit gives y = 2 x; its R-squared is 0.81.'
    bound=bind_quote(answer,'Its R-squared is')
    assert bound['mode']=='INITIAL_FUNCTION_WORD_CASE_ONLY'
    assert answer[bound['start']:bound['end']]=='its R-squared is'
    assert bind_quote(answer,'Its r-squared is') is None
    assert bind_quote('Pressure in pA is 3.0.','Pressure in Pa is 3.0.') is None
    assert bind_quote('a = 3.0 is the coefficient.','A = 3.0 is the coefficient.') is None
    assert bind_quote('its R-squared is 0.1; its R-squared is 0.2.','Its R-squared is') is None


def test_named_dimensionless_quantities_do_not_require_unknown_physical_units():
    proof = dimensionless_unit_proof(0.4548, "CoV = σ_V / |μ_V| ≈ 0.4548 (≈ 45.5%)",
                                     "coefficient of variation", "1")
    assert proof["source_unit"] == "1"
    assert dimensionless_unit_proof(45.5, "CV = 45.5%", "coefficient of variation", "1")["source_unit"] == "%"
    assert dimensionless_unit_proof(0.5, "stored pressure 0.5", "coefficient of variation", "1") is None
    assert dimensionless_unit_proof(0.5, "CV = 0.5", "coefficient of variation", "Pa") is None
    # A competing normalized-dot reference cannot make a source cosine acquire
    # unknown units. This does not grant semantic equivalence or numeric credit.
    assert dimensionless_unit_proof(.0035, 'The signed alignment coefficient is 0.0035.',
        'Divide the mean dot product by the product of RMS magnitudes.', '1')['source_unit'] == '1'


def test_split_result_declaration_binds_units_but_not_an_intervening_quantity():
    quote = r'\boxed{C=+0.0035}.'
    context = 'The signed alignment coefficient is\n\n\\[\n' + quote
    ref = 'Average the local cosine of the angle.'
    assert dimensionless_unit_proof(.0035, quote, ref, '1', context=context)['source_unit'] == '1'
    unrelated = context.replace(r'\[', 'The pressure is')
    assert dimensionless_unit_proof(.0035, quote, ref, '1', context=unrelated) is None
    assert dimensionless_unit_proof(.004, quote, ref, '1', context=context) is None
    context = 'Use the coefficient of variation of density.\nThe density nonuniformity coefficient is:\n\n0.9359'
    assert dimensionless_unit_proof(.9359, '0.9359', 'CV', '1', context=context)['source_unit'] == '1'
    assert dimensionless_unit_proof(.9359, '0.9359', 'CV', '1', context=context.replace('coefficient of variation', 'standard deviation')) is None


def test_std_mean_symbol_proof_does_not_bind_operands_or_different_fields():
    context = r'Report C=\sigma/|\mu|. The nonuniformity coefficient is C = 2.4.'
    quote = 'The mean is 10.0 and the standard deviation is 24.0, so C = 2.4.'
    assert dimensionless_unit_proof(2.4, quote, 'CV', '1', context=context)['source_unit'] == '1'
    assert dimensionless_unit_proof(10.0, quote, 'CV', '1', context=context) is None
    assert dimensionless_unit_proof(24.0, quote, 'CV', '1', context=context) is None
    assert dimensionless_unit_proof(2.4, quote.replace('C =', 'D ='), 'CV', '1', context=context) is None
    assert dimensionless_unit_proof(2.4, quote, 'CV', '1', context=context.replace(r'\mu', r'\mu_p')) is None
    assert dimensionless_unit_proof(2.4, quote, 'CV', 'Pa', context=context) is None


def test_explicit_percent_cv_is_verified_without_a_new_judge_call():
    answer, meta, gt, material, j = fixture()
    gt.findings_by_operationalization[0].findings[0] = gt.findings_by_operationalization[0].findings[0].model_copy(
        update={"statement": "coefficient of variation", "unit": "1"})
    answer = answer.replace("CV is 0.5.", "CV is 50%.")
    j["findings"][0].update(evidence_text="CV is 50%.", value=50, unit=None)
    result = calculate((answer, meta, gt, material, j))
    assert result["metrics"]["core_finding_recall"]["value"] == 1


@pytest.mark.parametrize('value,quote,expected',[(.5,'The result is 0.5.',1),(.9,'The result is 0.9.',0),
    (50,r'The result is 50\%.',1),(90,'The result is 90%.',0)])
def test_dimensionless_result_does_not_require_redundant_unit_word(value,quote,expected):
    answer,meta,gt,material,j=fixture()
    ref=gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings[0]=ref.model_copy(update={'unit':'1'})
    answer=answer.replace('CV is 0.5.',quote)
    j['findings'][0].update(value=value,evidence_text=quote,unit=None)
    result=calculate((answer,meta,gt,material,j))
    assert result['metrics']['core_finding_recall']['value']==expected


def test_implicit_dimensionless_unit_rejects_physical_units_and_ambiguous_scales():
    from flowintentbench.answer_evidence import implicit_dimensionless_scale
    for quote in ('0.5 Pa','0.5 m/s',r'0.5\,\mathrm{m/s}','First 0.5, then 0.5%.'):
        assert implicit_dimensionless_scale(.5,quote,'1') is None
    assert implicit_dimensionless_scale(.5,'0.5','Pa') is None
    assert implicit_dimensionless_scale(.5,'0.5','1',unit_declaration='Reported in percent') is None
    assert implicit_dimensionless_scale(50,'50 percent','1')['source_unit']=='%'
    answer,meta,gt,material,j=fixture()
    ref=gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings[0]=ref.model_copy(update={'unit':'1'})
    answer=answer.replace('CV is 0.5.','CV is 0.5 Pa.')
    j['findings'][0].update(evidence_text='CV is 0.5 Pa.',unit=None)
    assert calculate((answer,meta,gt,material,j))['metrics']['core_finding_recall']['value'] is None


def test_cv_symbol_definition_binds_result_but_not_numerator_or_neighbor():
    context = r"coefficient of variation (CV), (C_p=\sigma_p/|\mu_p|)."
    quote = r"C_p = 43.8817/96.4927 = 0.45477 \quad (45.48\%)."
    proof = dimensionless_unit_proof(.45477, quote, "coefficient of variation", "1", context=context)
    assert proof["source_unit"] == "1"
    assert dimensionless_unit_proof(43.8817, quote, "coefficient of variation", "1", context=context) is None
    assert dimensionless_unit_proof(.45477, quote.replace('C_p', 'C_q'), "CV", "1", context=context) is None
    assert dimensionless_unit_proof(.45477, quote, "CV", "1", context=context.replace(r'\mu_p', r'\mu_q')) is None


@pytest.mark.parametrize("unit", ["field units", "velocity units", "stored y-coordinate units",
                                  "stored coordinates", "stored coordinate-area units", "square coordinate units",
                                  "snapshot’s stored density–velocity units", "density units as stored",
                                  "coordinate-units²", "stored-x units", "file speed units", "file coordinate units",
                                  "stored flux units", "stored density × velocity units", "stored/data units",
                                  "stored coordinate scale", "stored speed scale", "stored volume scale",
                                  "stored current-density units", "squared stored-coordinate units",
                                  "cubic native coordinate units"])
def test_native_unit_spellings_require_literal_source_and_never_infer_si(unit):
    answer, meta, gt, material, j = fixture()
    quote = f"Result = 0.5 {unit}."
    answer = answer.replace("CV is 0.5.", quote)
    j['findings'][0].update(evidence_text=quote, unit=unit)
    assert calculate((answer, meta, gt, material, j))['metrics']['core_finding_recall']['value'] == 1
    gt.findings_by_operationalization[0].findings[0] = gt.findings_by_operationalization[0].findings[0].model_copy(update={'unit': 'm'})
    assert calculate((answer, meta, gt, material, j))['metrics']['core_finding_recall']['value'] is None


def test_native_scale_recognition_never_erases_physical_units_or_missing_source():
    from flowintentbench.answer_evidence import is_native_scale_unit, unit_is_bound
    for unit in ('m', 'cm', 'stored cm units', 'stored cm scale', 'stored kg/m^3 scale',
                 'Pa', 'stored kg/m^3 units', '1', '%', 'velocity in SI units',
                 'squared stored cm units', 'stored current-density A/m^2 units'):
        assert not is_native_scale_unit(unit)
    assert unit_is_bound('stored density × velocity units', r'0.5 \text{stored density}\times\text{velocity units}')
    answer, meta, gt, material, j = fixture()
    j['findings'][0].update(unit='file speed units')
    assert calculate((answer, meta, gt, material, j))['finding_checks'][0]['reason'] == 'UNIT_NOT_IN_SOURCE'
    quote = 'Result = 0.9 file speed units.'
    answer = answer.replace('CV is 0.5.', quote)
    j['findings'][0].update(evidence_text=quote, value=.9)
    assert calculate((answer, meta, gt, material, j))['metrics']['core_finding_recall']['value'] == 0


def test_original_metrics_are_exact_for_grounded_reference_answer():
    result = calculate(fixture())
    for key in ("o_score", "urs", "resolved_o_compliance", "finding_precision", "core_finding_recall", "c_score", "branch_alignment"):
        assert result["metrics"][key]["value"] == 1.0
    assert result["metrics"]["adequate_core_complete"]["status"] == "NOT_APPLICABLE"


def test_qualitative_interpretation_is_not_false_or_a_missing_numeric_answer():
    answer, meta, gt, material, j = fixture()
    quote = "The pressure is spatially nonuniform."
    answer = answer.replace("CV is 0.5.", quote)
    j["findings"][0].update(statement="nonuniformity", value=None, evidence_text=quote)
    result = calculate((answer, meta, gt, material, j))
    assert result["finding_checks"][0]["verdict"] is None
    assert result["metrics"]["core_finding_recall"]["value"] == 0
    assert result["metrics"]["finding_precision"]["upper"] == 1
    assert result["metrics"]["finding_precision"]["lower"] == 0


def test_local_malformed_extra_does_not_erase_known_precision_lower_bound():
    data = fixture()
    extra = deepcopy(data[-1]["findings"][0])
    extra.update(finding_id="bad_parse", value=[.5, "unparsed interval"], matches=[])
    data[-1]["findings"].append(extra)
    result = calculate(data)
    assert result["extraction_complete"] is True
    assert result["metrics"]["finding_precision"]["lower"] == .5
    assert result["metrics"]["core_finding_recall"]["value"] == 1


def test_presentation_markup_and_units_preserve_literal_evidence():
    from flowintentbench.answer_evidence import bind_quote, unit_is_bound
    assert bind_quote("Velocity is the **z component**.", "Velocity is the z component.")["mode"] == "PRESENTATION_MARKUP_ONLY"
    assert bind_quote("Velocity is the **x component**.", "Velocity is the z component.") is None
    assert unit_is_bound("1/s", r"\boxed{452.09\ \text{1/s}}")
    assert unit_is_bound("1/s", "530.1 s⁻¹")
    assert value_is_bound(-.051, "w = 0.048 - 0.051 rho")


def test_unicode_scientific_notation_and_numeric_string_remain_source_bound():
    from flowintentbench.answer_evidence import numeric_string
    assert numeric_string("4.7×10²") == 470
    assert value_is_bound(470, "mean = 4.7×10² s⁻¹")
    assert numeric_string("about 0.5") == "about 0.5"
    data = fixture()
    data[-1]["findings"][0]["value"] = "0.5"
    result = calculate(data)
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["review_repairs"][0]["reason"] == "NUMERIC_STRING_CANONICALIZATION"
    data[-1]["findings"][0]["value"] = "0.4"
    assert calculate(data)["finding_checks"][0]["reason"] == "VALUE_NOT_IN_SOURCE"


def test_empty_dimension_matches_preserves_finding_credit_and_unknown_method():
    data = fixture()
    data[-1]["dimensions"][1]["matches"] = {}
    result = calculate(data)
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["o_score"]["lower"] == .5
    assert result["metrics"]["o_score"]["upper"] == 1


def test_invalid_reference_is_local_unknown_and_duplicate_match_is_audited():
    data = fixture()
    data[-1]["findings"][0]["matches"] += [
        {"branch_id": "o", "finding_id": "cv"}, {"branch_id": "o", "finding_id": ""}]
    result = calculate(data)
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["o_score"]["value"] == 1
    assert {r["reason"] for r in result["review_repairs"]} == {
        "DUPLICATE_MATCH_REMOVED", "INVALID_REFERENCE_ID_UNKNOWN"}


def test_invalid_mixed_value_does_not_become_model_error_or_hide_valid_finding():
    data = fixture()
    bad = deepcopy(data[-1]["findings"][0])
    bad.update(finding_id="mixed", value=[.5, "unparsed text"])
    data[-1]["findings"].append(bad)
    result = calculate(data)
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert next(c for c in result["finding_checks"] if c["finding_id"] == "mixed")["verdict"] is None
    assert result["metrics"]["finding_precision"]["value"] is None


def test_native_unit_annotation_requires_source_evidence():
    answer, meta, gt, material, j = fixture()
    answer += " Values use stored coordinate units."
    j["findings"][0].update(unit="stored coordinate units", unit_evidence_text="Values use stored coordinate units.")
    assert calculate((answer, meta, gt, material, j))["finding_checks"][0]["verdict"] is True
    j["findings"][0]["unit_evidence_text"] = ""
    assert calculate((answer, meta, gt, material, j))["finding_checks"][0]["reason"] == "UNIT_NOT_IN_SOURCE"


def test_vector_rounding_retains_strict_failure_without_inventing_false_science():
    answer, meta, gt, material, j = fixture()
    reference = gt.findings_by_operationalization[0].findings[0]
    reference = reference.model_copy(update={"category": "location", "value": [1.23456, 2.34567, 3.45678],
        "verification": reference.verification.model_copy(update={"absolute_tolerance": None, "spatial_tolerance": 1e-5})})
    gt.findings_by_operationalization[0].findings[0] = reference
    policy = material["finding_verification_policy"]["policies"][0]
    policy.update(verification_mode="spatial_euclidean", tolerance_authority="REPORTING_PRECISION_DERIVED",
        verification_parameters=reference.verification.model_dump())
    quote = "Centroid is (1.235, 2.346, 3.457)."
    answer += " " + quote
    j["findings"][0].update(statement="Centroid", value=[1.235, 2.346, 3.457], evidence_text=quote)
    assert calculate((answer, meta, gt, material, j))["finding_checks"][0]["verdict"] is None
    policy["tolerance_authority"] = "DOMAIN_AUTHORED"
    assert calculate((answer, meta, gt, material, j))["finding_checks"][0]["verdict"] is None


def test_wrong_number_lowers_findings_without_erasing_method_scores():
    answer, meta, gt, material, j = fixture()
    answer = answer.replace("0.5", "0.9")
    j["findings"][0].update(value=0.9, evidence_text="CV is 0.9.")
    result = calculate((answer, meta, gt, material, j))
    assert result["metrics"]["o_score"]["value"] == 1.0
    for name in ("finding_precision", "core_finding_recall", "c_score"):
        assert result["metrics"][name]["value"] == 0.0
    assert result["metrics"]["branch_alignment"]["value"] == 1.0
    assert result["branch_alignment_diagnostic"]["kind"] == "SINGLE_BRANCH_TRIVIAL"
    assert result["branch_alignment_diagnostic"]["informative"] is False


def test_alignment_detects_cross_branch_conflict_without_full_precision_closure():
    answer, meta, gt, material, j = fixture()
    gt.acceptable_operationalizations.append(SimpleNamespace(operationalization_id="b"))
    ref = gt.findings_by_operationalization[0].findings[0].model_copy(update={"finding_id": "b_cv"})
    gt.findings_by_operationalization.append(SimpleNamespace(operationalization_id="b", findings=[ref]))
    material["finding_verification_policy"]["policies"].append({
        **material["finding_verification_policy"]["policies"][0], "operationalization_id": "b", "finding_id": "b_cv"})
    for d in j["dimensions"]:
        d["matches"]["b"] = False
    j["findings"][0]["matches"] = [{"branch_id": "b", "finding_id": "b_cv"}]
    result = calculate((answer, meta, gt, material, j))
    assert result["metrics"]["branch_alignment"]["value"] == 0
    assert result["branch_alignment_diagnostic"]["informative"] is True


def test_false_extraction_cannot_get_verified_credit():
    answer, meta, gt, material, j = fixture()
    answer = answer.replace("0.5", "0.9")
    j["findings"][0]["evidence_text"] = "CV is 0.9."
    result = calculate((answer, meta, gt, material, j))
    assert result["finding_checks"][0]["verdict"] is None
    assert result["finding_checks"][0]["reason"] == "VALUE_NOT_IN_SOURCE"
    assert result["metrics"]["core_finding_recall"]["value"] is None
    assert result["metrics"]["core_finding_recall"]["lower"] == 0


def test_reviewer_matching_a_threshold_to_mean_is_unknown_even_if_numbers_agree():
    answer, meta, gt, material, j = fixture()
    answer = answer.replace("CV is 0.5.", "The 90th percentile threshold is 0.5.")
    j["findings"][0].update(statement="The 90th percentile threshold", evidence_text="The 90th percentile threshold is 0.5.")
    gt.findings_by_operationalization[0].findings[0] = gt.findings_by_operationalization[0].findings[0].model_copy(
        update={"statement": "Report the regional mean."})
    result = calculate((answer, meta, gt, material, j))
    assert result["finding_checks"][0]["reason"].startswith("QUANTITY_MATCH_CONFLICT")
    assert result["finding_checks"][0]["verdict"] is None
    assert result["metrics"]["core_finding_recall"]["value"] is None
    assert result["metrics"]["o_score"]["value"] == 1


def test_one_unknown_extra_does_not_block_independent_metrics():
    data = fixture()
    data[-1]["findings"].append({"finding_id": "extra", "statement": "new observation",
        "evidence_text": "unlocatable", "value": 2.0, "unit": None, "eligible": True, "matches": []})
    result = calculate(data)
    assert result["metrics"]["core_finding_recall"]["value"] == 1
    assert result["metrics"]["o_score"]["value"] == 1
    assert result["metrics"]["finding_precision"]["lower"] == 0.5
    assert result["metrics"]["finding_precision"]["upper"] == 1


def test_missing_dimension_is_terminal_deduction_and_c_inapplicable():
    data = fixture()
    data[-1]["dimensions"][1].update(status="MISSING", evidence_text="", matches={"o": None})
    r = calculate(data)["metrics"]
    assert r["o_score"]["value"] == 0.5
    assert r["urs"]["value"] == 0
    assert r["resolved_o_compliance"]["value"] == 1
    assert r["c_score"]["status"] == "NOT_APPLICABLE"
    assert r["core_finding_recall"]["value"] == 1


def test_valid_but_unverified_alternative_gets_bounds_not_zero():
    data = fixture()
    data[1].unresolved_operationalization_dimensions = data[1].principal_operationalization_dimensions
    for d in data[-1]["dimensions"]:
        d["matches"]["o"] = False
    data[-1]["findings"][0]["matches"] = []
    r = calculate(data)["metrics"]
    assert r["o_score"]["lower"] == 0 and r["o_score"]["upper"] == 1
    assert r["finding_requirement_recall"]["value"] is None


def test_novel_method_reference_misassignment_cannot_create_false_quality_deduction():
    answer, meta, gt, material, j = fixture()
    meta.unresolved_operationalization_dimensions = meta.principal_operationalization_dimensions
    for dimension in j["dimensions"]:
        dimension["matches"]["o"] = False
    answer = answer.replace("0.5", "0.9")
    j["findings"][0].update(value=.9, evidence_text="CV is 0.9.")
    result = calculate((answer, meta, gt, material, j))
    assert result["finding_checks"][0]["reason"] == "UNVERIFIED_ALTERNATIVE_METHOD_REFERENCE_MISMATCH"
    assert result["metrics"]["finding_precision"]["upper"] == 1
    assert result["metrics"]["finding_precision"]["lower"] == 0


def test_f2_authored_adequate_set_not_number_of_gt_findings():
    data = fixture()
    data[3]["finding_requirement_contract"] = {"mandatory_roles": ["measure"],
        "adequate_core_sets": [["measure"]], "reference_finding_role_map": {"cv": "measure"}}
    r = calculate(data, "O1-F2")["metrics"]
    assert r["finding_requirement_recall"]["value"] == 1
    assert r["adequate_core_complete"]["value"] == 1
    assert r["core_finding_recall"]["status"] == "NOT_APPLICABLE"


def test_review_schema_errors_are_not_model_failures():
    data = fixture()
    data[-1]["dimensions"].pop()
    with pytest.raises(ValueError, match="every principal dimension"):
        calculate(data)


def test_outcome_met_is_not_implicitly_converted_into_o_match():
    _, _, gt, _, _ = fixture()
    old = {"branch_relations": [{"branch_id": "o", "relation": "DIFFERENT"}],
        "ratings": [{"item_id": "method:feature_definition", "verdict": "MET", "evidence_text": "alternate"}],
        "claims": [], "extraction_limitations": ""}
    assert from_outcome(old, gt, ["feature_definition"])["dimensions"][0]["matches"] == {"o": None}


def test_aggregation_keeps_missing_trials_and_uses_case_weights():
    def row(case, metrics):
        return {"case_id": case, "metrics": metrics}
    m = empty_metrics("O1-F1", "missing")
    known = deepcopy(m)
    known["o_score"] = interval(1, 1)
    result = summarize([row("a", known), row("a", m), row("a", m), row("b", known)])
    assert result["o_score"]["case_macro_lower"] == pytest.approx(2 / 3)
    assert result["o_score"]["case_macro_upper"] == 1
    assert result["o_score"]["coverage"] == 0.5


def test_pairing_excludes_unmatched_cases_without_changing_full_denominator():
    rows = [{"model_id": m, "case_id": c, "trial": 1, "condition": "O1-F1", "collection_status": "COMPLETED",
             "metrics": empty_metrics("O1-F1", "unknown")}
            for m, c in [("a", "x"), ("b", "x"), ("a", "y")]]
    assert paired_comparison(rows)[0]["matched_answer_pairs"] == 1
    assert len(choose_paired(rows, 2)) == 2
    assert len(choose_paired(rows, 1)) == 0


def test_rounding_is_unknown_only_when_it_can_explain_reporting_tolerance_failure():
    answer, meta, gt, material, j = fixture()
    ref = gt.findings_by_operationalization[0].findings[0]
    gt.findings_by_operationalization[0].findings[0] = ref.model_copy(update={"value": 0.454766,
        "verification": ref.verification.model_copy(update={"absolute_tolerance": 0.000005})})
    policy = material["finding_verification_policy"]["policies"][0]
    policy.update(tolerance_authority="REPORTING_PRECISION_DERIVED",
                  verification_parameters=gt.findings_by_operationalization[0].findings[0].verification.model_dump())
    answer = answer.replace("0.5", "0.4548")
    j["findings"][0].update(value=0.4548, evidence_text="CV is 0.4548.")
    data = (answer, meta, gt, material, j)
    assert calculate(data)["finding_checks"][0]["reason"].startswith("SOURCE_ROUNDING")
    policy["tolerance_authority"] = "DOMAIN_AUTHORED"
    assert calculate(data)["finding_checks"][0]["verdict"] is None
    policy["tolerance_authority"] = "REPORTING_PRECISION_DERIVED"
    answer = answer.replace("0.4548", "0.9999")
    j["findings"][0].update(value=0.9999, evidence_text="CV is 0.9999.")
    assert calculate((answer, meta, gt, material, j))["finding_checks"][0]["verdict"] is False


def test_optional_incorrect_claim_cannot_force_upper_precision_below_valid_completion():
    answer, meta, gt, material, j = fixture()
    answer += " Another CV is 0.9."
    extra = deepcopy(j["findings"][0])
    extra.update(finding_id="optional", value=0.9, evidence_text="Another CV is 0.9.", eligible=None)
    j["findings"].append(extra)
    result = calculate((answer, meta, gt, material, j))["metrics"]
    assert result["finding_precision"]["lower"] == 0.5
    assert result["finding_precision"]["upper"] == 1
    assert result["c_score"]["upper"] == 1


def test_raw_run_has_no_dependency_on_historical_judgment_and_rejects_identity_mismatch(tmp_path):
    from flowintentbench.external_file_evaluator import write_json
    from scripts.evaluate_trusted_metrics import sha
    run = dict(run_id="r", final_response="answer", case_id="c", model_id="m", trial_index=1,
               experiment_id="e", run_status="COMPLETED")
    path = tmp_path / "run.json"
    write_json(path, run)
    slot = dict(case_id="c", model_id="m", trial_index=1, run_record_path=str(path), run_record_sha256=sha(path))
    answer, old, provenance = load_run_source(slot, tmp_path, "e")
    assert answer == "answer" and old is None and provenance["source"] == "ORIGINAL_RUN"
    slot["model_id"] = "different"
    with pytest.raises(ValueError, match="model_id"):
        load_run_source(slot, tmp_path, "e")


def test_review_cache_cannot_replace_actual_provider_response(tmp_path):
    from flowintentbench.external_file_evaluator import write_json
    from flowintentbench.trusted_scoring import digest
    path = tmp_path / "judgments" / "id.json"
    receipt = tmp_path / "receipt.json"
    write_json(receipt, {"completed": True, "response_model": "m", "final_text": '{"a":1}'})
    write_json(path, {"request_id": "id", "judgment": {"a": 1}, "judgment_sha256": digest({"a": 1}), "receipt": str(receipt)})
    write_json(tmp_path / "requests/id.json", {"prompt": "p", "schema": {}})
    assert validated_cache(path, "id", "p", {}, "m")["judgment"] == {"a": 1}
    write_json(receipt, {"completed": True, "response_model": "m", "final_text": '{"a":2}'})
    with pytest.raises(ValueError, match="completed reviewer response"):
        validated_cache(path, "id", "p", {}, "m")
    assert fatal_provider_error({"http_status": 403, "error": "quota"})
    assert fatal_provider_error({"error": "insufficient_user_quota"})
    assert not fatal_provider_error({"error": "HTTP timeout"})


def test_prior_protocol_reuse_preserves_input_receipt_and_effort(tmp_path):
    import json
    from flowintentbench.external_file_evaluator import write_json
    from flowintentbench.trusted_scoring import digest, Judgment
    from scripts.evaluate_trusted_metrics import prior_review, prior_review_index
    prompt = 'old protocol\n\n' + json.dumps({"answer": "CV is 0.5", "reference": 0.5})
    new_prompt = 'new protocol\n\n' + json.dumps({"answer": "CV is 0.5", "reference": 0.5, "auxiliary_verification_catalog": {}})
    receipt = tmp_path / 'api/receipt.json'
    write_json(receipt, {"completed": True, "response_model": "m", "reasoning_effort": "medium", "final_text": '{"a":1}'})
    write_json(receipt.with_name('request.json'), {"input": [{"content": prompt}], "max_output_tokens": 6000})
    write_json(tmp_path / 'reviews/id.json', {"id": "id", "judgment": {"a": 1}, "judgment_sha256": digest({"a": 1}), "receipt": str(receipt)})
    write_json(tmp_path / 'requests/id.json', {"prompt": prompt, "schema": Judgment.model_json_schema()})
    index = prior_review_index([tmp_path])
    assert prior_review(index, new_prompt, 'm')[0].stem == 'id'
    assert prior_review(index, new_prompt, 'm', effort='low') is None
    assert prior_review(index, new_prompt.replace('CV is 0.5', 'CV is 0.6'), 'm') is None
    assert prior_review(index, new_prompt.replace('"reference": 0.5', '"reference": 0.6'), 'm') is None


def test_unknown_applicability_does_not_narrow_conditional_c_average():
    certain, optional = empty_metrics("O1-F1", ""), empty_metrics("O1-F1", "")
    certain["c_score"] = interval(1, 1)
    optional["c_score"] = {**interval(0, 0), "applicability_unknown": True}
    result = summarize([{"case_id": "a", "metrics": certain}, {"case_id": "b", "metrics": optional}])
    assert result["c_score"]["case_macro_lower"] == 0.5
    assert result["c_score"]["case_macro_upper"] == 1
    assert result["c_score"]["value"] is None


def test_new_answer_pipeline_can_review_without_donor_and_keeps_api_error_local(tmp_path, monkeypatch):
    import scripts.evaluate_model_answers as runner
    import scripts.evaluate_answered_outcomes as credentials
    import scripts.outcome_responses_transport as transport
    from flowintentbench.external_file_evaluator import write_json
    answer, meta, gt, material, _ = fixture()
    run_path = tmp_path / "run.json"
    write_json(run_path, dict(run_id="new", final_response=answer, case_id="c", model_id="m", trial_index=1,
        experiment_id="e", run_status="COMPLETED"))
    slot = dict(slot_id="new", case_id="c", model_id="m", trial_index=1, status="COMPLETED",
                run_record_path=str(run_path), run_record_sha256=runner.sha(run_path))
    state = dict(experiment_id="e", slots=[slot], configuration={"models": ["m"]})
    cases = {"c": {"case_id": "c", "condition": "O2-F1"}}
    write_json(tmp_path / "collection_state.json", state)
    write_json(tmp_path / "manifest.json", {"cases": list(cases.values())})
    monkeypatch.setattr(runner, "collect", lambda *a, **k: ([{**slot, "output_id": "test", "collection": str(tmp_path), "experiment_id": "e"}], cases, []))
    monkeypatch.setattr(runner, "load_development_case", lambda *_: (None, meta, gt, material))
    monkeypatch.setattr(runner, "review_prompt", lambda *_: "bound fixture request")
    monkeypatch.setattr(credentials, "configure_review_credentials", lambda *_: None)
    class FailedReviewer:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return {"completed": False, "http_status": 403, "error": "insufficient_quota"}
    monkeypatch.setattr(transport, "ResponsesOutcomeReviewer", FailedReviewer)
    args = runner.parser().parse_args(["--collection", str(tmp_path), "--manifest", str(tmp_path / "manifest.json"),
        "--output", str(tmp_path / "output"), "--donor", str(tmp_path / "absent-donor"), "--max-api-calls", "1"])
    result = runner.run(args)
    row, = result["rows"]
    assert row["status"] == "REVIEW_ERROR" and row["provenance"]["source"] == "ORIGINAL_RUN"
    assert row["metrics"]["o_score"]["upper"] == 1 and row["metrics"]["o_score"]["value"] is None
    assert result["evaluation_cost"]["api_errors"] == 1
    assert result["evaluation_cost"]["provider_circuit_open"] is True
    assert result["answered_review_status"] == "INCOMPLETE"


def test_signed_cosine_and_explicit_same_field_cv_formula_are_dimensionless():
    from flowintentbench.answer_evidence import dimensionless_unit_proof
    cosine = dimensionless_unit_proof(.0035, 'The signed alignment coefficient is 0.0035.',
        'Average the local cosine of the angle between vectors.', '1')
    assert cosine and cosine['source_unit'] == '1'
    formula = r'Mean 52.45; \frac{\sigma_{\rho,V}}{|\bar\rho_V|}=0.9358868 (93.59%).'
    reference = 'Use the volume-weighted standard deviation divided by the absolute volume-weighted mean.'
    assert dimensionless_unit_proof(.9358868, formula, reference, '1')['source_unit'] == '1'
    assert dimensionless_unit_proof(52.45, formula, reference, '1') is None
    assert dimensionless_unit_proof(.9358868, formula.replace(r'\bar\rho_V', r'\bar p_V'), reference, '1') is None


def test_native_dataset_coordinate_possessive_is_bound_without_si_inference():
    from flowintentbench.answer_evidence import unit_is_bound
    for possessive in ("dataset's", 'dataset’s'):
        quote = f'Result = 0.5 in the {possessive} spatial coordinate units.'
        assert unit_is_bound('dataset spatial coordinate units', quote)
        assert not unit_is_bound('m', quote)
        answer, meta, gt, material, j = fixture()
        j['findings'][0].update(evidence_text=quote, unit='dataset spatial coordinate units')
        args = (answer.replace('CV is 0.5.', quote), meta, gt, material, j)
        assert calculate(args)['metrics']['core_finding_recall']['value'] == 1
        ref = gt.findings_by_operationalization[0].findings[0]
        gt.findings_by_operationalization[0].findings[0] = ref.model_copy(update={'unit':'m'})
        assert calculate(args)['metrics']['core_finding_recall']['value'] is None
    assert not unit_is_bound('dataset spatial coordinate units', 'Result = 0.5 meters.')


def test_rounding_exactness_applies_to_the_number_not_unrelated_prose():
    from flowintentbench.answer_evidence import rounding_radius
    assert rounding_radius(1.86, 'The total is approximately 1.86. This is exactly why absolute weights are used.') == .005
    assert rounding_radius(1.86, 'Exactly three methods agree. The total is 1.86.') == .005
    for quote in ('Total is exactly 1.86.', 'The exact total is 1.86.', 'Total is 1.86 (exact).'):
        assert rounding_radius(1.86, quote) is None


@pytest.mark.parametrize('quote,value',[
    ('The lowest stratum (0.88235--0.90588; 47.97% of volume).',[.88235,.90588]),
    ('The highest stratum (0.97647--1.00000; 47.85% of volume).',[.97647,1.]),
    ('Observed range is -0.9---0.4.',[-.9,-.4]),
    ('The interval (1e-3--2e-3).',[.001,.002]),
])
def test_explicit_prose_double_hyphen_range_preserves_endpoint_signs(quote,value):
    from flowintentbench.answer_evidence import bind_value
    assert bind_value(quote,quote,value)['status']=='BOUND'
    assert bind_value(quote,quote,[-value[0],value[1]])['status']=='UNBOUND'


def test_arithmetic_double_minus_is_not_reinterpreted_as_prose_range():
    from flowintentbench.answer_evidence import numeric_notation
    for text in ('x=1--2','(1--2)','values [1, -2]','range is not known: x=1--2'):
        assert numeric_notation(text)==text
