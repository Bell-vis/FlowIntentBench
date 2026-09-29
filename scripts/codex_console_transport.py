"""Fresh Codex CLI conversations using the existing ChatGPT login, never an SDK.

This is a console transport, not the in-conversation collaboration tool. Inputs,
events, final messages and completion receipt are kept for reproducibility.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cli_schema_compatible(schema):
    """Conservatively recognize closed schemas accepted by strict CLI output.

    Existing adjudication schemas include optional/open maps. Their exact schema
    remains in the prompt and is validated locally; do not mutate that contract
    just to fit the hosted structured-output subset.
    """
    if isinstance(schema, list):
        return all(cli_schema_compatible(value) for value in schema)
    if not isinstance(schema, dict):
        return True
    kinds = schema.get("type", [])
    if kinds == "object" or isinstance(kinds, list) and "object" in kinds:
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("properties", {})) != set(schema.get("required", [])):
            return False
    return all(cli_schema_compatible(value) for value in schema.values())


def command(model, workdir, output_schema=None, *, reasoning_effort='xhigh', no_tools=False):
    if reasoning_effort not in {'low', 'medium', 'high', 'xhigh'}:
        raise ValueError('Unsupported reasoning effort')
    args = ["codex", "-a", "never", "exec", "--ignore-user-config", "--ignore-rules",
            "--skip-git-repo-check", "--ephemeral", "-s", "read-only" if no_tools else "danger-full-access",
            "-C", str(Path(workdir).resolve()), "-m", model,
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
            "-c", "project_doc_max_bytes=0", "-c", "features.memories=false",
            "-c", "features.multi_agent=false", "-c", 'web_search="disabled"',
            "--json"]
    if no_tools:
        for feature in ('shell_tool', 'apps', 'multi_agent', 'browser_use', 'computer_use'):
            args += ['-c', f'features.{feature}=false']
    if output_schema is not None:
        args += ["--output-schema", str(Path(output_schema).resolve())]
    return args + ["-"]


def parse_events(path):
    result = dict(completed=False, thread_id=None, final_text="", usage=None, error=None)
    # JSONL is delimited by LF. str.splitlines() also splits literal U+0085,
    # U+2028 etc. inside otherwise valid JSON strings (e.g. binary previews).
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            result["error"] = "Malformed CLI event stream"
            continue
        if not isinstance(event, dict):
            result["error"] = "Malformed CLI event object"
            continue
        kind = event.get("type")
        if kind == "thread.started":
            result["thread_id"] = event.get("thread_id")
        elif kind == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                result["error"] = "Malformed CLI item"
            elif item.get("type") == "agent_message":
                if not isinstance(item.get("text"), str):
                    result["error"] = "Malformed CLI final text"
                else:
                    result["final_text"] = item["text"]
        elif kind == "turn.completed":
            result["completed"] = True
            result["usage"] = event.get("usage")
        elif kind in {"turn.failed", "error"}:
            result["error"] = str(event.get("error") or event.get("message") or event)
    return result


def _stop(process):
    # Stop this CLI process group only. No namespaces or system settings change.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    # The leader may exit before one of its descendants. Kill the remaining
    # group as well, rather than leaving an analysis running after the budget.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()


def run_codex(prompt, model, workdir, output_dir, timeout_seconds=900, output_schema=None,
              *, reasoning_effort='xhigh', no_tools=False):
    workdir, output_dir = Path(workdir).resolve(), Path(output_dir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    events, errors = output_dir / "events.jsonl", output_dir / "stderr.log"
    if any((output_dir / name).exists() for name in ("events.jsonl", "receipt.json", "prompt.txt")):
        raise ValueError("CLI attempt already exists; never overwrite or resume a blind conversation")
    if timeout_seconds <= 0:
        raise ValueError("Positive remaining timeout required")
    schema_path = None
    if output_schema is not None:
        if isinstance(output_schema, dict):
            schema_path = output_dir / "output_schema.json"
            schema_path.write_text(json.dumps(output_schema))
        else:
            schema_path = Path(output_schema)
    schema_value = json.loads(schema_path.read_text()) if schema_path else None
    enforce_schema = schema_path is not None and cli_schema_compatible(schema_value)
    args = command(model, workdir, schema_path if enforce_schema else None,
                   reasoning_effort=reasoning_effort, no_tools=no_tools)
    (output_dir / "prompt.txt").write_text(prompt)
    (output_dir / "command.json").write_text(json.dumps(args, indent=2))
    started = time.time()
    timed_out = False
    process = None
    launch_error = None
    try:
        with events.open("x") as stdout, errors.open("x") as stderr:
            try:
                environment = os.environ.copy()
                if no_tools:
                    for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CHATGPT_BASE_URL'):
                        environment.pop(name, None)
                process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                           text=True, start_new_session=True, env=environment)
            except OSError as exc:
                launch_error = f"CLI launch failed: {type(exc).__name__}: {exc}"
                stderr.write(launch_error)
                process = None
            try:
                if process is not None:
                    process.communicate(prompt, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop(process)
            except BaseException:
                _stop(process)
                raise
    finally:
        if process is not None and process.poll() is None:
            _stop(process)
    result = parse_events(events)
    if launch_error:
        result["error"] = launch_error
    result.update(returncode=process.returncode if process is not None else None, timed_out=timed_out,
                  started_epoch=started, finished_epoch=time.time(), model=model,
                  events_path=str(events), transport="codex_exec_chatgpt_login",
                  structured_output_enforced=enforce_schema, reasoning_effort=reasoning_effort)
    result["completed"] = bool(result["completed"] and not result["error"]
                               and result["returncode"] == 0 and not timed_out)
    (output_dir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
