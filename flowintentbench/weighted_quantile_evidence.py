"""Recognize an executed inverse weighted-ECDF quantile closure."""
import ast
import hashlib
import json
import re


def observed_weighted_quantile(effective, trajectory):
    text = str(effective.get("criterion", "")).casefold()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:st|nd|rd|th)?\s*percentile", text)
    upper = re.search(r"(?:upper|top)\s*(\d+(?:\.\d+)?)\s*%\s*(?:of\s+(?:the\s+)?)?(?:domain\s+)?volume", text)
    if not (match or upper) or trajectory is None or not trajectory.is_file():
        return None
    target = 1 - float(upper.group(1))/100 if upper else float(match.group(1)) / 100
    data = trajectory.read_bytes()
    events = json.loads(data)
    success = {e.get("canonical_call_id") for e in events if e.get("event") == "python_execution" and e.get("success") is True}
    proofs = []
    for event in events:
        for call in event.get("tool_batch", {}).get("calls", []):
            if call.get("canonical_call_id") not in success:
                continue
            code = call.get("arguments", {}).get("code", "")
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            aliases, bindings, functions = set(), {}, {}
            if any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                   or isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"exec", "setattr"}
                   for n in ast.walk(tree)):
                continue

            def numpy_call(node, name):
                return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases
                        and node.func.attr == name)

            def recognized(function):
                if function.decorator_list or len(function.args.args) != 1 or len(function.body) != 1:
                    return False
                body = function.body[0]
                if not isinstance(body, ast.Return) or not isinstance(body.value, ast.Subscript):
                    return False
                result, search = body.value.value, body.value.slice
                if not isinstance(result, ast.Name) or not numpy_call(search, "searchsorted") or len(search.args) != 2:
                    return False
                if any(k.arg != "side" or not isinstance(k.value, ast.Constant) or k.value.value != "left" for k in search.keywords):
                    return False
                cumulative, cutoff = search.args
                if not isinstance(cumulative, ast.Name) or not isinstance(cutoff, ast.BinOp) or not isinstance(cutoff.op, ast.Mult):
                    return False
                qname = function.args.args[0].arg
                operands = [cutoff.left, cutoff.right]
                expected_last = ast.parse(f"{cumulative.id}[-1]", mode="eval").body
                if not (any(isinstance(n, ast.Name) and n.id == qname for n in operands)
                        and any(ast.dump(n) == ast.dump(expected_last) for n in operands)):
                    return False
                cumsum = bindings.get(cumulative.id)
                sorted_values = bindings.get(result.id)
                if not numpy_call(cumsum, "cumsum") or len(cumsum.args) != 1 or not isinstance(cumsum.args[0], ast.Name):
                    return False
                sorted_weights = bindings.get(cumsum.args[0].id)
                if not all(isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and isinstance(n.slice, ast.Name)
                           for n in (sorted_values, sorted_weights)):
                    return False
                if sorted_values.slice.id != sorted_weights.slice.id:
                    return False
                order = bindings.get(sorted_values.slice.id)
                return (numpy_call(order, "argsort") and len(order.args) == 1
                        and isinstance(order.args[0], ast.Name) and order.args[0].id == sorted_values.value.id)

            def resolved(node, seen=frozenset()):
                if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
                    return resolved(bindings[node.id], seen | {node.id})
                return node

            def same(a, b):
                return ast.dump(resolved(a)) == ast.dump(resolved(b))

            def direct_inverse(node):
                # values[order[searchsorted(cumsum(weights[order]), q*weights.sum())]]
                if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Subscript):
                    return False
                order_node, search = node.slice.value, node.slice.slice
                order = resolved(order_node)
                if not numpy_call(order, "argsort") or len(order.args) != 1 or not same(order.args[0], node.value):
                    return False
                if not numpy_call(search, "searchsorted") or len(search.args) != 2:
                    return False
                if any(k.arg != "side" or not isinstance(k.value, ast.Constant) or k.value.value != "left" for k in search.keywords):
                    return False
                cumulative, cutoff = resolved(search.args[0]), resolved(search.args[1])
                normalizer = None
                if isinstance(cumulative, ast.BinOp) and isinstance(cumulative.op, ast.Div):
                    normalizer, cumulative = cumulative.right, resolved(cumulative.left)
                if not numpy_call(cumulative, "cumsum") or len(cumulative.args) != 1:
                    return False
                weighted = resolved(cumulative.args[0])
                if not isinstance(weighted, ast.Subscript) or not same(weighted.slice, order_node):
                    return False
                weights = resolved(weighted.value)
                # Ravel preserves the same total weight.
                if isinstance(weights, ast.Call) and isinstance(weights.func, ast.Attribute) and weights.func.attr == "ravel" and not weights.args and not weights.keywords:
                    weights = resolved(weights.func.value)
                if normalizer is None:
                    if not isinstance(cutoff, ast.BinOp) or not isinstance(cutoff.op, ast.Mult):
                        return False
                    q, normalizer = cutoff.left, cutoff.right
                else:
                    q = cutoff
                if not isinstance(q, ast.Constant) or isinstance(q.value, bool) or not isinstance(q.value, (int,float)) or abs(q.value-target)>1e-12:
                    return False
                normalizer = resolved(normalizer)
                return (isinstance(normalizer, ast.Call) and isinstance(normalizer.func, ast.Attribute)
                        and normalizer.func.attr == "sum" and not normalizer.args and not normalizer.keywords
                        and same(normalizer.func.value, weights))

            for statement in tree.body:
                assigned = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                for name in assigned:
                    bindings.pop(name, None)
                    functions.pop(name, None)
                    aliases.discard(name)
                if isinstance(statement, ast.Import):
                    aliases.update(n.asname or n.name for n in statement.names if n.name == "numpy")
                if isinstance(statement, ast.FunctionDef):
                    functions[statement.name] = statement
                value = getattr(statement, "value", None)
                if isinstance(statement, ast.Assign):
                    for node in statement.targets:
                        if isinstance(node, ast.Name):
                            bindings[node.id] = value
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)) and direct_inverse(value):
                    proofs.append({"canonical_call_id": call["canonical_call_id"], "quantile": target,
                                   "method": "inverse_weighted_ecdf_left", "executed_expression": ast.unparse(value),
                                   "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
                if not isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)) or not isinstance(value, ast.Call):
                    continue
                if not isinstance(value.func, ast.Name) or value.func.id not in functions or len(value.args) != 1 or value.keywords:
                    continue
                argument = value.args[0]
                if not isinstance(argument, ast.Constant) or isinstance(argument.value, bool) or not isinstance(argument.value, (int, float)):
                    continue
                if abs(argument.value - target) > 1e-12:
                    continue
                if not recognized(functions[value.func.id]):
                    if any(numpy_call(node, name) for node in ast.walk(functions[value.func.id])
                           for name in ("searchsorted", "interp", "quantile", "percentile")):
                        return None
                    continue
                proofs.append({"canonical_call_id": call["canonical_call_id"], "quantile": target,
                               "method": "inverse_weighted_ecdf_left", "executed_function": value.func.id,
                               "code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    if not proofs:
        return None
    return {"method": "nearest_rank", "weighting": "cell_volume", "source": "successful_model_python_execution",
            "trajectory_sha256": hashlib.sha256(data).hexdigest(), "calls": proofs}
