from copy import deepcopy
from types import SimpleNamespace

import pytest

from flowintentbench.external_file_evaluator import digest, write_json
from scripts.run_expansion_file_evaluation import ROOT, candidate_unit_definition, file_sha256


def fixture_definition(tmp_path):
    identity = "expansion-recipe:" + "a" * 64
    parameters = {"property_measure": "Use the unscaled volume-weighted Pearson correlation."}
    plan = {
        "recipe": {"kind": "association", "measure": "pearson", "sampling": "cell_volume"},
        "recipe_implementation_sha256": file_sha256(ROOT / "flowintentbench/construction_recipes.py"),
        "runtime_recipe_implementation_sha256": file_sha256(ROOT / "flowintentbench/runtime_construction_recipes.py"),
    }
    artifact = {"materialization_id": identity, "parameters": parameters,
                "findings": [{"value": "REFERENCE_VALUE_MUST_NOT_LEAK", "unit": "SECRET"}],
                "execution": {"status": "MATERIALIZED", "G_of_O": {"value": 0.12345},
                              "data_provenance": {
                                  "effective_o_sha256": digest(parameters),
                                  "materialization_plan": plan,
                                  "compiled_plan_sha256": digest(plan), "executed_plan_sha256": digest(plan),
                                  "silent_substitutions": [], "unsupported_dimensions": []}}}
    novel = {"status": "ACCEPTED", "adjudicated_materialization_id": identity,
             "adjudicated_g_of_o_sha256": digest(artifact["execution"]["G_of_O"]),
             "operationalization": {"decisions": [
                 {"dimension": key, "statement": value} for key, value in parameters.items()]}}
    path = tmp_path / "materializations" / ("a" * 64 + ".json")
    def save():
        artifact["artifact_sha256"] = digest({k: v for k, v in artifact.items() if k != "artifact_sha256"})
        write_json(path, artifact)
    save()
    adjudications = SimpleNamespace(to_dict=lambda: {"novel_operationalization": deepcopy(novel)})
    return artifact, novel, adjudications, save, path


def test_candidate_definition_is_symbolic_bound_and_contains_no_values(tmp_path):
    artifact, novel, adjudications, _, _ = fixture_definition(tmp_path)
    proof = candidate_unit_definition(tmp_path, adjudications)
    assert proof["result_exponents"] == [0, 0]
    assert proof["denominator_exponents"] == [1, 1]
    assert proof["unscaled_result_unit"] == "1"
    assert proof["artifact_sha256"] == artifact["artifact_sha256"]
    assert proof["materialization_id"] == novel["adjudicated_materialization_id"]
    for forbidden in ("REFERENCE_VALUE_MUST_NOT_LEAK", "SECRET", "0.12345"):
        assert forbidden not in str(proof)


@pytest.mark.parametrize("changed", ["digest", "method", "g_of_o", "plan", "substitution"])
def test_candidate_definition_rejects_invalid_bindings(tmp_path, changed):
    artifact, novel, adjudications, save, path = fixture_definition(tmp_path)
    if changed == "digest":
        artifact["parameters"]["property_measure"] = "tampered"
        write_json(path, artifact)
    else:
        if changed == "method":
            novel["operationalization"]["decisions"][0]["statement"] = "different method"
        elif changed == "g_of_o":
            novel["adjudicated_g_of_o_sha256"] = "b" * 64
        elif changed == "plan":
            artifact["execution"]["data_provenance"]["executed_plan_sha256"] = "b" * 64
        elif changed == "substitution":
            artifact["execution"]["data_provenance"]["silent_substitutions"] = ["changed measure"]
        save()
    with pytest.raises(ValueError):
        candidate_unit_definition(tmp_path, adjudications)


@pytest.mark.parametrize("changed", ["not_accepted", "unrecognized_id", "missing", "unsupported", "old_code"])
def test_candidate_definition_does_not_infer_unavailable_proof(tmp_path, changed):
    artifact, novel, adjudications, save, path = fixture_definition(tmp_path)
    if changed == "not_accepted":
        novel["status"] = "UNRESOLVED"
    elif changed == "unrecognized_id":
        novel["adjudicated_materialization_id"] = "../../outside"
    elif changed == "missing":
        path.unlink()
    else:
        provenance = artifact["execution"]["data_provenance"]
        plan = provenance["materialization_plan"]
        if changed == "unsupported":
            plan["recipe"]["measure"] = "covariance"
        else:
            plan["recipe_implementation_sha256"] = "b" * 64
        provenance["compiled_plan_sha256"] = provenance["executed_plan_sha256"] = digest(plan)
        save()
    assert candidate_unit_definition(tmp_path, adjudications) is None
