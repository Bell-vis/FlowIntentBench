import hashlib
from pathlib import Path

from flowintentbench.external_file_evaluator import FileJudgmentTransport, build_file_backend


def test_file_manifest_binds_semantic_routing_and_prompt_source(tmp_path, monkeypatch):
    backend = build_file_backend(FileJudgmentTransport(tmp_path, context={}))
    before = backend.manifest()
    package = Path(__file__).resolve().parents[1] / "flowintentbench"
    for key, name in (("evaluator_source_sha256", "evaluator.py"),
                      ("structured_backend_source_sha256", "evaluator_backend.py")):
        assert before.deterministic_contract_hashes[key] == hashlib.sha256((package / name).read_bytes()).hexdigest()

    original = Path.read_bytes

    def changed_prompt_source(path):
        value = original(path)
        return value + b"\n# changed semantic evidence guidance\n" if path == package / "evaluator_backend.py" else value

    monkeypatch.setattr(Path, "read_bytes", changed_prompt_source)
    after = backend.manifest()
    assert after != before
    assert after.deterministic_contract_hashes["structured_backend_source_sha256"] != before.deterministic_contract_hashes["structured_backend_source_sha256"]


def test_file_manifest_binds_blind_adjudication_workflow(tmp_path, monkeypatch):
    backend = build_file_backend(FileJudgmentTransport(tmp_path, context={}))
    before = backend.manifest()
    workflow = Path(__file__).resolve().parents[1] / "scripts" / "run_expansion_file_evaluation.py"
    assert before.deterministic_contract_hashes["file_evaluation_workflow_sha256"] == hashlib.sha256(workflow.read_bytes()).hexdigest()
    original = Path.read_bytes

    def changed_workflow(path):
        data = original(path)
        return data + b"\n# changed role visibility\n" if path == workflow else data

    monkeypatch.setattr(Path, "read_bytes", changed_workflow)
    after = backend.manifest()
    assert after != before
    assert after.deterministic_contract_hashes["file_evaluation_workflow_sha256"] != before.deterministic_contract_hashes["file_evaluation_workflow_sha256"]
