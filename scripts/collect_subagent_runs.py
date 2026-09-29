"""Prepare/start/finish/import blind collaboration slots; never call a model API.

The orchestrator calls collaboration.spawn_agent with the returned spawn fields.
Collection completion is distinct from scientific evaluation completion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

try:
    from scripts.portable_fcntl import fcntl
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from portable_fcntl import fcntl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowintentbench import CaseLoader, EvaluationTarget
from flowintentbench.experiment_scope import case_inventory
from flowintentbench.model_runner import schedule_formal_trials, valid_final_answer_sections
from flowintentbench.quality_efficiency import QUALITY_POLICY, QUALITY_POLICY_SHA256
from scripts.run_real_model_pilot import import_collaboration_result

MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
TERMINAL = {"COMPLETED", "MODEL_NONCOMPLETION", "INFRASTRUCTURE_INVALID"}
HELPER = Path(__file__).with_name("subagent_run_python.py")
PYTHON = Path(os.environ.get("FLOWINTENT_PYTHON", sys.executable))
SOLVER_TIMEOUT_SECONDS = 720
SOLVER_MAX_PYTHON_EXECUTIONS = 48


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@contextmanager
def locked(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield output


def save_state(output, state):
    counts = dict(Counter(slot["status"] for slot in state["slots"]))
    state["status_counts"] = counts
    state["total_slots"] = len(state["slots"])
    state["collection_status"] = (
        "COLLECTION_COMPLETE" if all(s["status"] in TERMINAL for s in state["slots"])
        else "COLLECTION_INCOMPLETE")
    state["scientific_evaluation_status"] = "NOT_EVALUATED"
    write(output / "collection_state.json", state)


def prompt_for(slot, public):
    bundle = public / "bundles" / slot["slot_id"]
    work = public / "work" / slot["slot_id"]
    allowed = "\n".join(str(public / "raw" / path) for path in slot["raw_files"])
    return f"""Solve the scientific question in your assigned case_input.json independently.
Assigned bundle: {bundle / 'case_input.json'}
Data root for the relative paths in case_input.json: {public / 'raw'}
Only these explicitly declared data/geometry files are authorized:
{allowed}
Your independent work directory: {work}

Read only your assigned bundle and the explicitly listed data/geometry files.
Do not inspect parent directories, repository files, other bundles, other agents'
work, ground truth, reviews, metadata outside case_input.json, or prior answers.
Do not search the host for benchmark material. Do not reuse any other answer.
Use local Python only. No network, browser, APIs, other agents, or installations.
Local installed Python libraries may be imported normally. All data analysis
must run through the supplied journal helper: write a .py analysis file inside
your work directory, then execute `{PYTHON} {work / 'run_python.py'} {work / 'analysis.py'}`.
The absolute Python interpreter above is installed and must be used. Each helper call starts a fresh
interpreter; persist intermediate computations only in your work directory.
Read `case_input.json` once, make a short plan, and prefer one or two batched
Python calls that load the data and compute all required quantities. Reuse
intermediate files instead of rerunning identical code. Do not print arrays,
dataframes, or long diagnostics; print only compact values needed to validate
the Finding. Stop as soon as the answer has sufficient quantitative evidence.
The helper records exact source, stdout, stderr, elapsed time, and failures.
Do not edit the helper, execution_budget.json, or execution journal. Do not run
analysis Python directly outside the helper. You have 720 seconds from start
and at most 48 Python executions; stop when either is exhausted.

Write your final answer to {work / 'answer.md'} with exactly these two nonempty
Markdown sections, in this order:
## Operationalization
Explain your concrete interpretation and computational method.
## Finding
Report the resulting scientific answer with quantitative evidence where possible.
Leave the analysis source code in your work directory. Your final chat message
should only state that answer.md was written, or explain inability to finish.
This is an instruction-based blind run with fresh context and staged files;
the shared host tools do not provide kernel filesystem/network isolation.
"""


def prepare(*, manifest_path, datasets_root, output, seed=0, reasoning_effort="high"):
    manifest_path, datasets_root = Path(manifest_path).resolve(), Path(datasets_root).resolve()
    inventory = case_inventory(read(manifest_path), require_dataset=True)
    if len(inventory) != 96:
        raise ValueError("This protocol requires the explicit 96-case manifest")
    loaded, cases, raw = {}, {}, {}
    for case_id, row in inventory.items():
        source = (datasets_root.parent / row["case_input_path"]).resolve()
        if not source.is_relative_to(datasets_root):
            raise ValueError("case input escapes datasets root")
        if file_hash(source) != row["case_input_sha256"]:
            raise ValueError(f"case input checksum mismatch: {case_id}")
        case = CaseLoader(datasets_root).load(source)
        loaded[case_id] = case
        files = []
        for asset in (*case.data_files, *case.geometry_assets):
            name = asset.specification.path
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("unsafe declared asset path")
            files.append(name)
            if name not in raw:
                raw[name] = {"source": str(asset.path), "sha256": file_hash(asset.path)}
        cases[case_id] = {"source": str(source), "sha256": file_hash(source),
                          "raw_files": files, "dataset_id": row["dataset_id"]}
    config = {"protocol": "COLLABORATION_STAGED_INSTRUCTIONS_V1",
              "models": MODELS, "repetitions": 3, "base_seed": seed,
              "reasoning_effort": reasoning_effort, "timeout_seconds": SOLVER_TIMEOUT_SECONDS,
              "max_python_executions": SOLVER_MAX_PYTHON_EXECUTIONS, "manifest_path": str(manifest_path),
              "quality_policy": QUALITY_POLICY, "quality_policy_sha256": QUALITY_POLICY_SHA256,
              "manifest_sha256": file_hash(manifest_path),
              "datasets_root": str(datasets_root), "cases": cases, "raw_files": raw,
              "helper_sha256": file_hash(HELPER),
              "python_executable": str(PYTHON),
              "collector_sha256": file_hash(Path(__file__)),
              "importer_sha256": file_hash(Path(__file__).with_name("run_real_model_pilot.py"))}
    binding = digest(config)
    with locked(output) as output:
        state_path = output / "collection_state.json"
        if state_path.exists():
            state = read(state_path)
            if state["configuration_sha256"] != binding:
                raise ValueError("resume configuration/input hashes differ; choose a new output")
            verify(state)
            return state
        runtime_cache = Path(
            os.environ.get("FLOWINTENTBENCH_RUNTIME_CACHE", ROOT / ".runtime-cache")
        ).resolve()
        runtime_cache.mkdir(parents=True, exist_ok=True)
        public = Path(
            tempfile.mkdtemp(prefix="flowintentbench_blind_", dir=runtime_cache)
        ) / "public"
        public.mkdir()
        for name, entry in raw.items():
            destination = public / "raw" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry["source"], destination)
            if os.name != "nt":
                destination.chmod(0o444)
            if file_hash(destination) != entry["sha256"]:
                raise ValueError("raw input changed during staging")
        slots = []
        for trial in schedule_formal_trials(tuple(inventory), repetitions=3, base_seed=seed):
            for position, case_id in enumerate(trial.case_ids):
                for model in MODELS:
                    slot = {"slot_id": uuid4().hex, "model_id": model,
                            "case_id": case_id, "dataset_id": cases[case_id]["dataset_id"],
                            "trial_index": trial.trial_index, "ordering_seed": trial.ordering_seed,
                            "case_execution_position": position, "status": "PENDING",
                            "raw_files": cases[case_id]["raw_files"], "agent_id": None,
                            "run_record_path": None}
                    bundle = public / "bundles" / slot["slot_id"]
                    bundle.mkdir(parents=True)
                    shutil.copyfile(cases[case_id]["source"], bundle / "case_input.json")
                    if os.name != "nt":
                        (bundle / "case_input.json").chmod(0o444)
                    work = public / "work" / slot["slot_id"]
                    work.mkdir(parents=True)
                    shutil.copyfile(HELPER, work / "run_python.py")
                    prompt = prompt_for(slot, public)
                    slot["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                    prompt_path = output / "prompts" / (slot["slot_id"] + ".txt")
                    prompt_path.parent.mkdir(exist_ok=True)
                    prompt_path.write_text(prompt)
                    slots.append(slot)
        state = {"experiment_id": "collaboration-" + uuid4().hex,
                 "configuration": config, "configuration_sha256": binding,
                 "public_root": str(public), "slots": slots,
                 "blinding": {"fresh_context": True, "staged_raw_copies": True,
                              "instruction_scoped_access": True,
                              "kernel_filesystem_isolation": False,
                              "kernel_network_isolation": False},
                 "telemetry_unavailable": ["input_tokens", "output_tokens", "model_turn_count",
                                           "tool_batch_count", "provider_reported_cost",
                                           "all_collaboration_tool_calls"]}
        verify(state)
        save_state(output, state)
        return state


def verify(state, slot=None):
    config = state["configuration"]
    if digest(config) != state["configuration_sha256"]:
        raise ValueError("configuration binding mismatch")
    if config["quality_policy_sha256"] != QUALITY_POLICY_SHA256 or config["quality_policy"] != QUALITY_POLICY:
        raise ValueError("frozen quality policy changed")
    for name, path in (("helper_sha256", HELPER), ("collector_sha256", Path(__file__)),
                       ("importer_sha256", Path(__file__).with_name("run_real_model_pilot.py"))):
        if config[name] != file_hash(path):
            raise ValueError("frozen collection implementation changed")
    public = Path(state["public_root"])
    cases = [slot] if slot is not None else state["slots"]
    raw_names = {name for row in cases for name in row["raw_files"]}
    for name in raw_names:
        path = public / "raw" / name
        if path.is_symlink() or file_hash(path) != config["raw_files"][name]["sha256"]:
            raise ValueError("staged raw data hash mismatch")
    for row in cases:
        case = public / "bundles" / row["slot_id"] / "case_input.json"
        helper = public / "work" / row["slot_id"] / "run_python.py"
        if case.is_symlink() or file_hash(case) != config["cases"][row["case_id"]]["sha256"]:
            raise ValueError("staged case input hash mismatch")
        if helper.is_symlink() or file_hash(helper) != config["helper_sha256"]:
            raise ValueError("execution helper hash mismatch")


def get_slot(state, slot_id):
    matches = [slot for slot in state["slots"] if slot["slot_id"] == slot_id]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate opaque slot ID")
    return matches[0]


def start(output, slot_id, agent_id=None):
    with locked(output) as output:
        state = read(output / "collection_state.json")
        slot = get_slot(state, slot_id)
        verify(state, slot)
        public = Path(state["public_root"])
        work = public / "work" / slot_id
        if slot["status"] == "PENDING":
            if (work / "answer.md").exists() or (work / "execution_journal.jsonl").exists():
                raise ValueError("pending slot contains prior answer/execution")
            slot.update(status="RUNNING", started_epoch=time.time(), agent_id=agent_id)
            write(work / "execution_budget.json", {
                "started_epoch": slot["started_epoch"],
                "deadline_epoch": slot["started_epoch"] + SOLVER_TIMEOUT_SECONDS,
                "max_python_executions": SOLVER_MAX_PYTHON_EXECUTIONS})
        elif slot["status"] != "RUNNING":
            raise ValueError("only pending/running slots can start")
        elif agent_id:
            if slot["agent_id"] not in (None, agent_id):
                raise ValueError("slot already assigned to another agent")
            slot["agent_id"] = agent_id
        if agent_id and "agent_attached_epoch" not in slot:
            slot["agent_attached_epoch"] = time.time()
            slot["dispatch_and_attachment_delay_seconds"] = slot["agent_attached_epoch"] - slot["started_epoch"]
        prompt = (output / "prompts" / (slot_id + ".txt")).read_text()
        if hashlib.sha256(prompt.encode()).hexdigest() != slot["prompt_sha256"]:
            raise ValueError("prompt hash mismatch")
        save_state(output, state)
        return {"task_name": "blind_" + slot_id, "model": slot["model_id"],
                "reasoning_effort": state["configuration"]["reasoning_effort"],
                "fork_turns": "none", "message": prompt,
                "deadline_epoch": slot["started_epoch"] + SOLVER_TIMEOUT_SECONDS}


def finish(output, slot_id, *, outcome="COMPLETED", reason=None, agent_id=None):
    if outcome not in TERMINAL:
        raise ValueError("invalid terminal outcome")
    with locked(output) as output:
        state = read(output / "collection_state.json")
        slot = get_slot(state, slot_id)
        if slot["status"] in TERMINAL or slot["status"] == "FINISHED":
            return slot
        if slot["status"] != "RUNNING":
            raise ValueError("cannot finish a slot that was not started")
        if agent_id:
            if slot["agent_id"] not in (None, agent_id):
                raise ValueError("agent identity mismatch")
            slot["agent_id"] = agent_id
        if not slot["agent_id"]:
            raise ValueError("record the actual collaboration agent ID before finish")
        work = Path(state["public_root"]) / "work" / slot_id
        with (work / ".execution.lock").open("a") as execution_lock:
            fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ended = time.time()
            snapshot = output / "collected" / slot_id
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in ("answer.md", "execution_journal.jsonl"):
                if (work / name).exists():
                    if (work / name).is_symlink():
                        raise ValueError("answer/journal must be regular slot files, not symlinks")
                    shutil.copyfile(work / name, snapshot / name)
            if (work / "execution_sources").exists():
                shutil.copytree(work / "execution_sources", snapshot / "execution_sources", dirs_exist_ok=True)
            slot.update(status="FINISHED", finished_epoch=ended, requested_outcome=outcome,
                        failure_reason=reason,
                        collected_hashes={str(p.relative_to(snapshot)): file_hash(p)
                                          for p in snapshot.rglob("*") if p.is_file()})
            write(work / "finished.json", {"finished_epoch": ended})
        save_state(output, state)
        return slot


def import_slot(output, slot_id):
    with locked(output) as output:
        state = read(output / "collection_state.json")
        slot = get_slot(state, slot_id)
        if slot["status"] in TERMINAL:
            path = output / slot["run_record_path"]
            if file_hash(path) != slot["run_record_sha256"]:
                raise ValueError("imported run record changed")
            return slot
        if slot["status"] != "FINISHED":
            raise ValueError("finish and freeze artifacts before import")
        verify(state, slot)
        snapshot = output / "collected" / slot_id
        for name, expected in slot["collected_hashes"].items():
            if file_hash(snapshot / name) != expected:
                raise ValueError("collected artifact hash mismatch")
        answer_path, journal_path = snapshot / "answer.md", snapshot / "execution_journal.jsonl"
        answer = answer_path.read_text() if answer_path.exists() else None
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()] if journal_path.exists() else []
        outcome, reason = slot["requested_outcome"], slot["failure_reason"]
        elapsed = slot["finished_epoch"] - slot["started_epoch"]
        invalid_journal = any(row.get("execution_index") != index or row.get("finished_epoch") is None
                              or row.get("code_sha256") != hashlib.sha256(row.get("code", "").encode()).hexdigest()
                              or row.get("duration_seconds", -1) < 0
                              for index, row in enumerate(journal, 1))
        if invalid_journal:
            outcome, reason = "INFRASTRUCTURE_INVALID", "incomplete or invalid execution journal"
        elif outcome == "COMPLETED" and (elapsed > SOLVER_TIMEOUT_SECONDS or len(journal) > SOLVER_MAX_PYTHON_EXECUTIONS):
            outcome, reason = "MODEL_NONCOMPLETION", "720s/48 Python execution budget exceeded"
        elif outcome == "COMPLETED" and not valid_final_answer_sections(answer):
            outcome, reason = "MODEL_NONCOMPLETION", "missing/nonconforming answer.md"
        config = {"transport": "collaboration", "api_equivalent": False,
                  "reasoning_effort": state["configuration"]["reasoning_effort"],
                  "timeout_seconds": SOLVER_TIMEOUT_SECONDS, "max_python_executions": SOLVER_MAX_PYTHON_EXECUTIONS}
        target = EvaluationTarget("collaboration", slot["model_id"], model_configuration=config)
        public = Path(state["public_root"])
        loaded = CaseLoader(public / "raw").load(public / "bundles" / slot_id / "case_input.json")
        run_dir = output / "runs" / slot_id
        provenance = {"experiment_id": state["experiment_id"], "transport": "collaboration",
                      "api_equivalent": False, "collaboration_agent_id": slot["agent_id"],
                      "fork_turns": "none", "blinding": state["blinding"],
                      "configuration_sha256": state["configuration_sha256"],
                      "case_input_sha256": state["configuration"]["cases"][slot["case_id"]]["sha256"],
                      "prompt_sha256": slot["prompt_sha256"],
                      "telemetry_unavailable": state["telemetry_unavailable"],
                      "tool_call_count_scope": "recorded Python helper executions only; all collaboration calls unavailable",
                      "quality_policy_sha256": state["configuration"]["quality_policy_sha256"],
                      "started_epoch": slot["started_epoch"], "finished_epoch": slot["finished_epoch"],
                      "dispatch_and_attachment_delay_seconds": slot.get("dispatch_and_attachment_delay_seconds"),
                      "wall_clock_scope": "orchestrator start through finish; includes dispatch latency"}
        record = import_collaboration_result(
            loaded_case=loaded, target=target, slot=slot, output_dir=run_dir,
            answer=answer, journal=journal, elapsed_seconds=elapsed,
            status=outcome, failure_reason=reason, provenance=provenance)
        slot.update(status=record.run_status.value, failure_reason=reason,
                    run_record_path=str((run_dir / "run_record.json").relative_to(output)),
                    run_record_sha256=file_hash(run_dir / "run_record.json"))
        save_state(output, state)
        return slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "status", "start", "finish", "import"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--slot")
    parser.add_argument("--agent-id")
    parser.add_argument("--outcome", choices=sorted(TERMINAL), default="COMPLETED")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.case_manifest:
            parser.error("prepare requires --case-manifest")
        result = prepare(manifest_path=args.case_manifest, datasets_root=args.datasets_root,
                         output=args.output_root, seed=args.seed, reasoning_effort=args.reasoning_effort)
    elif args.command == "status":
        result = read(args.output_root / "collection_state.json")
    else:
        if not args.slot:
            parser.error("start/finish/import require --slot")
        if args.command == "start":
            result = start(args.output_root, args.slot, args.agent_id)
        elif args.command == "finish":
            result = finish(args.output_root, args.slot, outcome=args.outcome,
                            reason=args.reason, agent_id=args.agent_id)
        else:
            result = import_slot(args.output_root, args.slot)
    if args.command in {"prepare", "status"}:
        result = {key: result[key] for key in ("experiment_id", "configuration_sha256", "public_root",
                                              "total_slots", "status_counts", "collection_status",
                                              "scientific_evaluation_status")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
