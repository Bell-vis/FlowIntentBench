"""Generic, read-only answer ingestion. No dependence on model names or N=3."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(value):
    return [] if value is None else [value] if isinstance(value, (str, Path)) else value


def collect(collections, exports, manifest, *, models=None, case_ids=None, trials=None):
    """Return all requested observations, never just scientifically closed rows.

    Export format: a row, list of rows, {answers: [...]} or JSONL. Required:
    model_id, case_id, trial and answer (for COMPLETED). Supplied hashes are
    checked. Collections additionally bind the original solver task manifest.
    """
    frozen = read(manifest)
    cases = {r["case_id"]: r for r in frozen["cases"]}
    if len(cases) != len(frozen["cases"]):
        raise ValueError("duplicate evaluator case ID")
    rows, sources, keys = [], [], set()

    def add(row):
        key = (row["model_id"], row["case_id"], row["trial_index"])
        if not isinstance(key[0], str) or not key[0] or type(key[2]) is not int or key[2] < 1:
            raise ValueError("invalid model/trial identity")
        if key[1] not in cases:
            raise ValueError(f"unknown case: {key[1]}")
        if row.get("condition", cases[key[1]]["condition"]) != cases[key[1]]["condition"]:
            raise ValueError("answer condition differs from frozen case")
        if models and key[0] not in models or case_ids and key[1] not in case_ids or trials and key[2] not in trials:
            return
        if key in keys:
            raise ValueError(f"duplicate model/case/trial: {key}; select one authoritative source")
        keys.add(key)
        # File names are hashes; imported model/run IDs cannot escape output.
        row["output_id"] = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        rows.append(row)

    for directory in paths(collections):
        directory = Path(directory).resolve()
        state_path = directory / "collection_state.json"
        state = read(state_path)
        config = state["configuration"]
        # The two shipped collectors use default vs compact JSON separators.
        # Accept those documented serializations, not arbitrary mutated config.
        config_hashes = {hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                                 separators=separators).encode()).hexdigest()
                         for separators in (None, (",", ":"))}
        if state.get("configuration_sha256") is not None and state["configuration_sha256"] not in config_hashes:
            raise ValueError("collection configuration hash mismatch")
        if config["manifest_sha256"] not in {sha(manifest), frozen.get("source_manifest_sha256")}:
            raise ValueError("collection uses a different frozen task manifest")
        model_ids = config["models"]
        count = config["repetitions"]
        if len(set(model_ids)) != len(model_ids) or type(count) is not int or count < 1:
            raise ValueError("invalid collection models/repetitions")
        requested_cases = config.get("cases") or list(cases)
        if not set(requested_cases) <= set(cases):
            raise ValueError("collection contains foreign cases")
        expected = {(m, c, t) for m in model_ids for c in requested_cases for t in range(1, count + 1)}
        actual = [(s["model_id"], s["case_id"], s["trial_index"]) for s in state["slots"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("duplicate, missing or foreign requested collection slot")
        sources.append({"path": str(state_path), "sha256": sha(state_path), "kind": "COLLECTION"})
        # This historical collector writes a fixed run experiment ID and a
        # unique ledger ID (run_gpt6_sol_cases.py::run_slot). Adapt its schema,
        # without relaxing model/case/trial/file-hash checks for any model.
        run_experiment = ("gpt6-sol-case-collection" if
            state.get("schema_version") == config.get("schema") == "gpt6-sol-case-collection-v1"
            else state["experiment_id"])
        for slot in state["slots"]:
            add({**slot, "collection": str(directory), "experiment_id": state["experiment_id"],
                 "run_experiment_id": run_experiment})

    for source in paths(exports):
        source = Path(source).resolve()
        files = sorted(source.glob("*.json")) if source.is_dir() else [source]
        for path in files:
            data = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    if path.suffix == ".jsonl" else read(path))
            records = data if isinstance(data, list) else data.get("answers", [data])
            sources.append({"path": str(path), "sha256": sha(path), "kind": "ANSWER_EXPORT"})
            for record in records:
                trial = record.get("trial", record.get("trial_index"))
                status = record.get("collection_status", record.get("run_status", "COMPLETED"))
                answer = record.get("answer", record.get("final_response"))
                if status == "COMPLETED":
                    if not isinstance(answer, str):
                        raise ValueError("completed answer export has no text")
                    digest = hashlib.sha256(answer.encode()).hexdigest()
                    if record.get("answer_sha256", record.get("final_answer_sha256", digest)) != digest:
                        raise ValueError("exported answer hash mismatch")
                add({**record, "trial_index": trial, "status": status, "answer": answer,
                     "slot_id": record.get("run_id", f'{record["model_id"]}__{record["case_id"]}__{trial}'),
                     "export_source": str(path), "export_sha256": sha(path)})
    if not rows:
        raise ValueError("no selected answers/slots; provide --collection or --answers")
    if models and set(models) != {r["model_id"] for r in rows}:
        raise ValueError("a requested model has no selected slots")
    if case_ids and not set(case_ids) <= {r["case_id"] for r in rows}:
        raise ValueError("a requested case has no selected slots")
    return rows, cases, sources
