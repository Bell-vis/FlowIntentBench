"""Recover numerical conventions from successful model tool calls, never GT."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Any


def successful_code_projection(trajectory: Path) -> tuple[bytes, list[dict], str | None]:
    """Normalize existing trajectory or checked anonymous journal code only."""
    data = trajectory.read_bytes()
    payload = json.loads(data)
    if isinstance(payload, Mapping) and "code_only_execution_projection" in payload:
        source_sha = payload.get("source_journal_sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise ValueError("code-only projection requires source journal SHA-256")
        events = []
        for row in payload["code_only_execution_projection"]:
            code = row.get("code", "")
            if hashlib.sha256(code.encode()).hexdigest() != row.get("code_sha256"):
                raise ValueError("execution convention code digest mismatch")
            if row.get("returncode") != 0 or row.get("finished_epoch") is None:
                continue
            call_id = "execution:" + str(row["execution_index"])
            events.extend([{"event": "model_turn", "tool_batch": {"calls": [
                {"canonical_call_id": call_id, "arguments": {"code": code}}]}},
                {"event": "python_execution", "canonical_call_id": call_id, "success": True}])
        return data, events, source_sha
    if not isinstance(payload, list):
        raise ValueError("execution conventions require a trajectory array or anonymous code projection")
    # import_collaboration_result persists the helper journal rows directly,
    # rather than model_turn/tool_batch events used by the API runner.
    if payload and payload[0].get("event") == "collaboration_provenance":
        events = []
        for index, row in enumerate(payload[1:], 1):
            code = row.get("code", "")
            if (row.get("event") != "python_execution" or row.get("execution_index") != index
                    or hashlib.sha256(code.encode()).hexdigest() != row.get("code_sha256")):
                raise ValueError("collaboration execution journal binding mismatch")
            if row.get("returncode") != 0 or row.get("finished_epoch") is None or row.get("timed_out"):
                continue
            call_id = "execution:" + str(index)
            events.extend([{"event": "model_turn", "tool_batch": {"calls": [
                {"canonical_call_id": call_id, "arguments": {"code": code}}]}},
                {"event": "python_execution", "canonical_call_id": call_id, "success": True}])
        return data, events, None
    return data, payload, None


def run_bound_execution_trajectory(root: Path, run_record, *, fallback: Path | None = None) -> Path | None:
    """Prefer the imported run's trajectory; sidecar labels confer no authority.

    A response-only caller may explicitly supply independently verified
    evidence. With a run record, missing primary evidence stays unavailable.
    """
    if run_record is None:
        return fallback
    if not run_record.trajectory_path:
        return None
    from scripts.path_migration import resolve_repository_path
    path = resolve_repository_path(run_record.trajectory_path,
                                   repository_root=root, relative_root=root)
    if not path.is_file():
        return None
    if run_record.provider == "collaboration":
        payload = json.loads(path.read_bytes())
        expected = {"event": "collaboration_provenance", **run_record.runtime_environment_fingerprint}
        if not isinstance(payload, list) or not payload or payload[0] != expected:
            raise ValueError("trajectory provenance does not match the imported run record")
        # Validate code hashes even if this O needs no statistical convention.
        successful_code_projection(path)
    return path


def observed_quantile_method(effective: Mapping[str, Any], trajectory: Path | None) -> dict | None:
    """Accept an unambiguous, actually executed top-level NumPy quantile call.

    Unexecuted branches, failed calls, dynamic method arguments, reassigned
    NumPy functions and conflicting conventions supply no evidence. The
    default is the recorded NumPy API invocation, not an evaluator choice.
    """
    if trajectory is None or not trajectory.is_file():
        return None
    text = str(effective.get("criterion", "")).casefold()
    if (re.search(r"volume[- ]weighted\s+(?:\d+(?:\.\d+)?(?:st|nd|rd|th)?\s+)?percentile", text)
            or re.search(r"(?:upper|top)\s*\d+(?:\.\d+)?\s*%\s*(?:of\s+(?:the\s+)?)?(?:domain\s+)?volume", text)):
        from .weighted_quantile_evidence import observed_weighted_quantile
        return observed_weighted_quantile(effective, trajectory)
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", text)
    if not match:
        match = re.search(r"\bp_?\{?(\d+(?:\.\d+)?)\}?(?![\d.])", text)
    if not match:
        return None
    target = float(match.group(1)) / 100
    data, events, source_sha = successful_code_projection(trajectory)
    successful = {e.get("canonical_call_id") for e in events
                  if e.get("event") == "python_execution" and e.get("success") is True}
    proofs = []
    aliases = set()
    functions = {"percentile", "quantile", "nanpercentile", "nanquantile"}
    for event in events:
        for call in event.get("tool_batch", {}).get("calls", []):
            if call.get("canonical_call_id") not in successful:
                continue
            code = call.get("arguments", {}).get("code", "")
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            # Do not treat altered library functions as the NumPy API.
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and node.attr in functions:
                    return None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"setattr", "exec"}:
                    return None
            # Only literal scalar assignments in this successful code block
            # may supply a named quantile argument. Control flow invalidates
            # names it may rebind; its nested calls are never presumed executed.
            constants = {}
            for statement in tree.body:
                assigned = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                for name in assigned:
                    constants.pop(name, None)
                    aliases.discard(name)
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    literal = getattr(statement, "value", None)
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, (float, int)) and not isinstance(literal.value, bool):
                        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                        for target_node in targets:
                            if isinstance(target_node, ast.Name):
                                constants[target_node.id] = literal.value
                if isinstance(statement, ast.Import):
                    aliases.update(a.asname or a.name for a in statement.names if a.name == "numpy")
                value = getattr(statement, "value", None)
                if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)) or not isinstance(value, ast.Call):
                    continue
                # A scalar cast does not change the quantile interpolation.
                # Accept only the unrebound builtin float with one argument.
                if (isinstance(value.func, ast.Name) and value.func.id == "float"
                        and len(value.args) == 1 and not value.keywords
                        and not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id == "float" for n in ast.walk(tree))):
                    value = value.args[0]
                    if not isinstance(value, ast.Call):
                        continue
                fn = value.func
                if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                        and fn.value.id in aliases and fn.attr in functions):
                    continue
                if len(value.args) < 2:
                    continue
                argument = value.args[1]
                q = argument.value if isinstance(argument, ast.Constant) else constants.get(argument.id) if isinstance(argument, ast.Name) else None
                if isinstance(q, bool) or not isinstance(q, (float, int)):
                    continue
                q = q / 100 if "percentile" in fn.attr else q
                if abs(q - target) > 1e-12:
                    continue
                keywords = {k.arg: k.value for k in value.keywords}
                if None in keywords or len(value.args) > 2:
                    return None
                method_node = keywords.get("method", keywords.get("interpolation"))
                method = "linear" if method_node is None else method_node.value if isinstance(method_node, ast.Constant) else None
                if method not in {"linear", "inverted_cdf"}:
                    return None
                proofs.append({"canonical_call_id": call["canonical_call_id"], "function": f"numpy.{fn.attr}",
                               "quantile_argument_source": "literal" if isinstance(argument, ast.Constant) else "top_level_literal_assignment",
                               "quantile": q, "method": "nearest_rank" if method == "inverted_cdf" else method,
                               "method_source": "executed_numpy_default" if method_node is None else "executed_explicit_argument",
                               "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    if not proofs or len({p["method"] for p in proofs}) != 1:
        return None
    return {"method": proofs[0]["method"], "trajectory_sha256": hashlib.sha256(data).hexdigest(),
            **({"source_journal_sha256": source_sha} if source_sha else {}),
            "source": "successful_model_python_execution", "calls": proofs}


def observed_mean_std_convention(effective: Mapping[str, Any], trajectory: Path | None) -> dict | None:
    """Recover ddof/population only from a successful explicit mean+k*std call.

    The speed variable must originate at top level in NumPy's vector norm,
    and both statistics must consume that same unsliced array. No generated
    numeric values, printed results, failed cells, or unexecuted branches are
    used as execution evidence.
    """
    criterion = str(effective.get("criterion", "")).casefold()
    if not any(token in criterion for token in ("sigma", "σ", "standard deviation")) or trajectory is None or not trajectory.is_file():
        return None
    data, events, source_sha = successful_code_projection(trajectory)
    successful = {e.get("canonical_call_id") for e in events if e.get("event") == "python_execution" and e.get("success") is True}
    proofs = []
    for event in events:
        for call in event.get("tool_batch", {}).get("calls", []):
            if call.get("canonical_call_id") not in successful:
                continue
            code = call.get("arguments", {}).get("code", "")
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"exec", "setattr"} for n in ast.walk(tree)):
                return None
            if any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and n.attr in {"norm", "mean", "std"} for n in ast.walk(tree)):
                return None
            aliases, speed_arrays = set(), set()
            for statement in tree.body:
                assigned = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                aliases -= assigned
                speed_arrays -= assigned
                if isinstance(statement, ast.Import):
                    aliases.update(a.asname or a.name for a in statement.names if a.name == "numpy")
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                value = statement.value
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                if isinstance(value, ast.Call):
                    fn = value.func
                    if (isinstance(fn, ast.Attribute) and fn.attr == "norm" and isinstance(fn.value, ast.Attribute)
                            and fn.value.attr == "linalg" and isinstance(fn.value.value, ast.Name)
                            and fn.value.value.id in aliases and len(value.args) == 1
                            and isinstance(value.args[0], ast.Name)
                            and {k.arg: getattr(k.value, "value", None) for k in value.keywords} == {"axis": 1}):
                        speed_arrays.update(t.id for t in targets if isinstance(t, ast.Name))
                if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add)
                        and isinstance(value.right, ast.BinOp) and isinstance(value.right.op, ast.Mult)):
                    continue
                mean, multiplier, std = value.left, value.right.left, value.right.right
                if not (isinstance(multiplier, ast.Constant) and isinstance(multiplier.value, (int, float))
                        and not isinstance(multiplier.value, bool)):
                    continue
                def method_array(node, name):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == name and isinstance(node.func.value, ast.Name)
                            and node.func.value.id in speed_arrays and not node.args):
                        return node.func.value.id
                    return None
                variable = method_array(mean, "mean")
                if variable is None or variable != method_array(std, "std") or mean.keywords:
                    continue
                if any(k.arg != "ddof" for k in std.keywords):
                    continue
                ddof = 0 if not std.keywords else getattr(std.keywords[0].value, "value", None)
                if isinstance(ddof, bool) or ddof not in {0, 1}:
                    continue
                proofs.append({"canonical_call_id": call["canonical_call_id"], "line": statement.lineno,
                    "multiplier": float(multiplier.value), "ddof": ddof, "population": "all_speed_points",
                    "ddof_source": "executed_numpy_default" if not std.keywords else "executed_explicit_argument",
                    "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    if not proofs or len({(p["ddof"], p["multiplier"], p["population"]) for p in proofs}) != 1:
        return None
    return {"ddof": proofs[0]["ddof"], "multiplier": proofs[0]["multiplier"], "population": proofs[0]["population"],
            "trajectory_sha256": hashlib.sha256(data).hexdigest(),
            **({"source_journal_sha256": source_sha} if source_sha else {}),
            "source": "successful_model_python_execution", "calls": proofs}
