import os

import pytest

from flowintentbench.external_file_evaluator import write_json


@pytest.mark.skipif(os.name != 'nt', reason='Windows sharing violation retry')
def test_atomic_json_retries_transient_reader_lock(tmp_path, monkeypatch):
    import json
    original = os.replace
    calls = []
    def replace(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError('temporary sharing violation')
        return original(source, target)
    monkeypatch.setattr(os, 'replace', replace)
    path = tmp_path / 'progress.json'
    write_json(path, {'completed': 10})
    assert json.loads(path.read_text()) == {'completed': 10}
    assert len(calls) == 3
    assert list(tmp_path.iterdir()) == [path]


def test_permanent_write_failure_preserves_previous_json(tmp_path, monkeypatch):
    path = tmp_path / 'progress.json'
    write_json(path, {'completed': 9})
    original = path.read_bytes()
    def fail(*args):
        raise PermissionError('permanent error')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(PermissionError):
        write_json(path, {'completed': 10})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_progress_write_failure_cannot_change_evaluation_result(tmp_path, monkeypatch):
    import sys
    from flowintentbench import external_file_evaluator as files
    from scripts import replay_existing_with_progress as wrapper
    from scripts import run_expansion_file_evaluation as replay
    def fail(*args):
        raise PermissionError('monitor is reading')
    monkeypatch.setattr(files, 'write_json', fail)
    monkeypatch.setattr(replay, 'evaluate_submission', lambda: {'status': 'PENDING'})
    def main():
        assert replay.evaluate_submission() == {'status': 'PENDING'}
        return 0
    monkeypatch.setattr(replay, 'main', main)
    monkeypatch.setattr(sys, 'argv', ['wrapper', 'replay', '--output', str(tmp_path),
                                     '--run-record', 'one.json'])
    assert wrapper.main() == 0
