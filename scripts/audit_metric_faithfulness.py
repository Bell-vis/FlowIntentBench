#!/usr/bin/env python3
"""Replay real cached answers; audit corrected host rules and auxiliary claims.

Auxiliary semantic bindings below are manually audited, not automatic reviewer
accuracy evidence. The same numerical verifier is used by the production path.
The original 28-answer report/reviews are preserved as the before comparison.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.trusted_scoring import score, digest, VERSION


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# These are source-finding locators, not model-specific grading rules. All seven
# answers receive the same available global moment checks, plus explicitly
# bound auxiliary assertions. Unstated bins/bootstraps/precision remain unknown.
BINDINGS = {
    "gpt-6-astra": {"density_mean": "mean_x", "density_standard_deviation": "std_x",
        "vertical_velocity_mean": "mean_y", "vertical_velocity_standard_deviation": "std_y",
        "density_vertical_velocity_covariance": "covariance", "squared_correlation": "r_squared"},
    "gpt-5.6-luna": {"answer_density_weighted_mean": "mean_x", "answer_density_weighted_standard_deviation": "std_x",
        "answer_vertical_velocity_weighted_mean": "mean_y", "answer_vertical_velocity_weighted_standard_deviation": "std_y",
        "answer_weighted_covariance": "covariance", "answer_squared_correlation": "r_squared"},
    "gpt-5.6-terra": {"f2": "covariance", "f3": "mean_x", "f4": "std_x", "f5": "mean_y", "f6": "std_y",
        "f7": ["regression_intercept", "regression_slope"], "f8": "r_squared"},
    "claude-fable-5-1": {"f2": "covariance", "f3": ["std_x", "std_y"], "f4": ["mean_x", "mean_y"]},
    "claude-sonnet-5": {"f_mean_density": "mean_x", "f_density_std": "std_x",
        "f_mean_vertical_velocity": "mean_y", "f_vertical_velocity_std": "std_y",
        "f_covariance": "covariance", "f_r_squared": "r_squared"},
    "deepseek-flash": {"f2": "r_squared", "f3": "covariance", "f5": "mean_x", "f6": "mean_y", "f7": "std_x", "f8": "std_y", "f9": "regression_slope",
        "f18": [{"statistic": "layer_pearson_min", "axis": "z"}, {"statistic": "layer_pearson_max", "axis": "z"}],
        "f19": {"statistic": "layer_pearson_median", "axis": "z"}},
    "qwen3.8-max": {"f_global_moments": ["mean_x", "mean_y", "std_x", "std_y", "covariance"],
        "f_global_slope": "regression_slope", "f_global_r_squared": "r_squared"},
}


def annotate(row, judgment):
    changed = deepcopy(judgment)
    if row["case_id"] != "rt_density_vertical_motion_o1_f2":
        return changed, []
    by_id = {f["finding_id"]: f for f in changed["findings"]}
    additions = []
    # Full O text is retained only in this offline audit. Live reviews provide a
    # short exact method quote and existing dimension evidence binds the base O.
    method = row["answer"].split("## Finding")[0].strip()
    # All seven O1 answers explicitly declare the prescribed cells, vertex
    # averaging, Pearson and volume weighting (checked against full source O).
    # The old cached reviewer omitted some match maps. Record their closure as
    # an audited semantic annotation, never as a host-inferred equivalence.
    bid = "rt_density_vertical_motion_o1_f2_pearson"
    for dimension in changed["dimensions"]:
        dimension.update(status="EXTRACTED", evidence_text=method, matches={bid: True})
    additions.append({"dimension_evidence_and_matches": deepcopy(changed["dimensions"]),
                      "basis": "MANUALLY_AUDITED_EXPLICIT_O1_DECLARATIONS"})
    for fid, specifications in BINDINGS[row["model_id"]].items():
        finding = by_id[fid]
        vector = isinstance(specifications, list)
        queries = []
        for index, spec in enumerate(specifications if vector else [specifications]):
            query = {"statistic": spec} if isinstance(spec, str) else dict(spec)
            query.update(branch_id="rt_density_vertical_motion_o1_f2_pearson", value_index=index if vector else None,
                         method_evidence_text=method, scope_status="EXPLICIT",
                         scope_evidence_text=finding["evidence_text"] if query["statistic"].startswith("layer_") else method)
            queries.append(query)
        finding["numeric_checks"] = queries
        additions.append({"finding_id": fid, "numeric_checks": queries})
    return changed, additions


def audit_scope_ambiguity():
    """Show why the Qwen plume claim cannot use a whole-mesh negative label.

    Candidate masks are sensitivity diagnostics, never selected for scoring.
    The exact mask was not printed, so even a reproducing candidate earns no
    positive finding credit.
    """
    import numpy as np
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from scripts.audit_open_answer_methods import load, array
    mesh, provenance = load("Rayleigh_Taylor")
    nz, ny, nx = tuple(reversed(mesh.GetDimensions()))
    rho = array(mesh.GetPointData(), "density").reshape(nz, ny, nx)
    velocity = array(mesh.GetPointData(), "velocity").reshape(nz, ny, nx, 3)[..., 2]
    def cell_mean(values):
        return sum(values[k:k+nz-1, j:j+ny-1, i:i+nx-1] for k in (0, 1) for j in (0, 1) for i in (0, 1))/8
    rho, velocity = cell_mean(rho), cell_mean(velocity)
    axes = [vtk_to_numpy(g()).astype(float) for g in (mesh.GetXCoordinates, mesh.GetYCoordinates, mesh.GetZCoordinates)]
    dx, dy, dz = (np.diff(axis) for axis in axes)
    volume = dz[:, None, None]*dy[None, :, None]*dx[None, None, :]
    z = (axes[2][1:]+axes[2][:-1])/2
    candidates = []
    for lo, hi in ((0., 1.), (.23, .72), (.225, .725)):
        band = np.broadcast_to(((z >= lo) & (z <= hi))[:, None, None], rho.shape)
        values = []
        for condition in (rho > .99, rho < .895):
            mask = band & condition
            values.append(float(np.average(velocity[mask], weights=volume[mask])))
        candidates.append({"z_cell_center_range": [lo, hi], "conditional_mean_y": values})
    return {"model": "qwen3.8-max", "finding_id": "f_plume_asymmetry", "data": provenance,
        "reason": "Preceding Inside that band scopes the bullet list. Whole-mesh comparison is invalid. Printed approximate band endpoints do not identify its exact mask; retain unknown.",
        "candidate_scopes_not_scored": candidates,
        "interpretation": "Do not choose an unprinted band based on numerical agreement to award credit."}


def run(source, output):
    started = time.monotonic()
    manifest = ROOT / "experiments/expansion_v1_development/case_manifest.json"
    manifest_rows = {r["case_id"]: r for r in read(manifest)["cases"]}
    cases, rows, annotations = {}, [], []
    for path in sorted((source / "cases").glob("*.json")):
        before = read(path)
        answer = read(source / "answers" / path.name)
        if hashlib.sha256(answer["answer"].encode()).hexdigest() != before["answer_sha256"]:
            raise ValueError("answer source changed")
        stored = read(source / "reviews" / (before["request_id"] + ".json"))
        judgment = stored["judgment"]
        if digest(judgment) != before["judgment_sha256"]:
            raise ValueError("cached judgment source changed")
        if answer["case_id"] not in cases:
            cases[answer["case_id"]] = load_development_case(ROOT, manifest_rows[answer["case_id"]])
        ci, meta, gt, material = cases[answer["case_id"]]
        host_only = score(answer["answer"], meta, gt, material, judgment, condition=answer["condition"], root=ROOT, case_input=ci)
        annotated, additions = annotate(answer, judgment)
        after = score(answer["answer"], meta, gt, material, annotated, condition=answer["condition"], root=ROOT, case_input=ci)
        record = {k: answer[k] for k in ("model_id", "case_id", "trial", "condition", "answer_sha256")}
        record.update(before=before, host_only=host_only, after=after,
                      binding_origin="OFFLINE_SEMANTIC_AUDIT" if additions else "UNCHANGED_CACHED_REVIEW")
        write_json(output / "cases" / path.name, record)
        annotations.append({"answer_sha256": answer["answer_sha256"], "judgment_sha256": digest(judgment),
            "model_id": answer["model_id"], "case_id": answer["case_id"], "bindings": additions})
        rows.append(record)
    result = {"protocol": VERSION, "answers": len(rows), "new_api_calls": 0, "new_tokens": 0,
        "wall_seconds": time.monotonic()-started, "source_report_sha256": hashlib.sha256((source / "reports/experiment_report.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256((ROOT / "flowintentbench/trusted_scoring.py").read_bytes()).hexdigest(),
        "evidence_checker_sha256": hashlib.sha256((ROOT / "flowintentbench/answer_evidence.py").read_bytes()).hexdigest(),
        "auxiliary_semantic_binding": "MANUALLY_AUDITED_REAL_CACHED_FINDINGS; automatic new-review binding not calibrated",
        "binding_count": sum(len(a["bindings"]) for a in annotations),
        "excluded_candidate": audit_scope_ambiguity(), "rows": rows}
    result["wall_seconds"] = time.monotonic()-started
    write_json(output / "auxiliary_bindings.json", annotations)
    write_json(output / "report.json", result)
    def bounds(m):
        return "N/A" if not m["applicable"] else f"[{m['lower']:.4f}, {m['upper']:.4f}]"
    lines = ["# 真实答案与指标忠实性回放", "", f"共 {len(rows)} 份既有答案；新增 API/token = 0/0；离线总耗时 {result['wall_seconds']:.2f}s。",
        "原报告保持不变。host_only 是原评审的规则修复回放；after 另加入明确标注的人工语义绑定，数值由主机独立复算。",
        "附加检查不新增必答项、不修改原 GT、不按模型预期排名调分。", "", "## RT 附加数值断言", "",
        "| 模型 | Precision 原值 → 复核后 | C 原值 → 复核后 | 复算确定错误的 finding |", "|---|---|---|---|"]
    for row in rows:
        if row["case_id"] != "rt_density_vertical_motion_o1_f2":
            continue
        before, after = row["before"]["metrics"], row["after"]["metrics"]
        lines.append(f"| {row['model_id']} | {bounds(before['finding_precision'])} → {bounds(after['finding_precision'])} | {bounds(before['c_score'])} → {bounds(after['c_score'])} | "
                     + ", ".join(row["after"]["error_diagnostics"]["independent_contradicted_finding_ids"]) + " |")
    lines += ["", "## 每个确定错误的证据", ""]
    for row in rows:
        for fid, checks in row["after"]["auxiliary_checks"].items():
            for check in checks:
                if check["verdict"] is False:
                    lines.append(f"- {row['model_id']} / {fid} / {check['query']['statistic']}: 报告 {check['reported']}；复算 {check['expected']}；显示舍入半径 {check['rounding_radius']}。")
    lines += ["", "## 原评审不变时的主机规则修复", "",
        "| 答卷 | 指标 | 原值 → host_only |", "|---|---|---|"]
    for row in rows:
        for metric in ("o_score", "finding_precision", "finding_requirement_recall", "c_score"):
            old, new = row["before"]["metrics"][metric], row["host_only"]["metrics"][metric]
            if (old["lower"], old["upper"]) != (new["lower"], new["upper"]):
                lines.append(f"| {row['model_id']} / {row['case_id']} | {metric} | {bounds(old)} → {bounds(new)} |")
    lines += ["", "## 解释边界", "", "- 复算正确的附加结论可提高 precision/C 下界；错误结论降低其上界。必答项 recall 单独计算。",
              "- Qwen 羽流均速处于 Inside that band 的段落范围内。全域比较会误判；精确 band mask 未声明，因此保持未知，未计错误。",
              "- 人工核对过语义绑定，不能把此回放当作新 judge 抽取准确率。未来单次评审可以使用同一 numeric_checks 字段。",
              "- 未验证的新方法、非数值解释、未声明的 bootstrap/分组仍未知；本报告不是全量可信点值或模型排名。"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))


def live_probe(source, output):
    """Two bounded real calls, without the offline semantic annotations."""
    from types import SimpleNamespace
    from flowintentbench.trusted_scoring import review_prompt, Judgment
    from scripts.run_model_metric_calibration import review_tasks, cached
    case_id = "rt_density_vertical_motion_o1_f2"
    manifest = read(ROOT / "experiments/expansion_v1_development/case_manifest.json")
    row = next(r for r in manifest["cases"] if r["case_id"] == case_id)
    cases = {case_id: load_development_case(ROOT, row)}
    args = SimpleNamespace(output=output / "live", offline=False, reviewer_model="gpt-6-astra",
        auth_path=ROOT / "auth_yapi.json", api_config=ROOT / "config/yiapi_direct.toml",
        max_api_calls=2, max_wall_seconds=180, workers=2, timeout=150)
    tasks = []
    for model in ("deepseek-flash", "qwen3.8-max"):
        answer = read(source / "answers" / f"{model}__{case_id}.json")
        prompt, schema = review_prompt(*cases[case_id], answer["answer"]), Judgment.model_json_schema()
        identity = digest({"prompt": prompt, "schema": schema, "model": args.reviewer_model,
            "effort": "medium", "max_output_tokens": 6000, "protocol": VERSION})
        task = {"id": identity, "kind": "metric", "case_id": case_id, "row": answer,
            "prompt": prompt, "schema": schema, "max_output_tokens": 6000}
        write_json(args.output / "requests" / f"{identity}.json", {k: v for k, v in task.items() if k != "row"})
        tasks.append(task)
    review_tasks(args, tasks, cases)
    rows = [{"model_id": t["row"]["model_id"], "request_id": t["id"], "result": cached(t, args, cases)} for t in tasks]
    attempts = [read(p) for p in sorted((args.output / "attempts").glob("*.json"))]
    result = {"binding_origin": "NEW_SINGLE_PASS_REVIEW; no manual corrections", "rows": rows,
        "cumulative_probe_calls": sum(a["api_calls"] for a in attempts),
        "input_tokens": sum(a.get("usage", {}).get("input_tokens", 0) for a in attempts),
        "output_tokens": sum(a.get("usage", {}).get("output_tokens", 0) for a in attempts),
        "attempts": attempts}
    result["scorer_sha256"] = hashlib.sha256((ROOT / "flowintentbench/trusted_scoring.py").read_bytes()).hexdigest()
    result["numeric_verifier_sha256"] = hashlib.sha256((ROOT / "flowintentbench/numeric_diagnostics.py").read_bytes()).hexdigest()
    write_json(args.output / "report.json", result)
    lines = ["# 新单次评审的真实答卷验证", "", "未加入离线人工语义注释；原始响应和真实 usage 保留在本目录。宿主修复后直接回放响应，没有再次请求模型。", "",
        "| 模型 | Precision | C | 确认的错误 finding |", "|---|---|---|---|"]
    for row in rows:
        scored = row["result"]
        if scored is None:
            lines.append(f"| {row['model_id']} | 未评审 | 未评审 | — |")
            continue
        def fmt(metric):
            return f"[{metric['lower']:.4f}, {metric['upper']:.4f}]"
        errors = ', '.join(scored["error_diagnostics"]["independent_contradicted_finding_ids"]) or "未确认"
        lines.append(f"| {row['model_id']} | {fmt(scored['metrics']['finding_precision'])} | {fmt(scored['metrics']['c_score'])} | {errors} |")
    lines += ["", "DeepSeek 的 layer 范围/中位数被评审合并成一个 finding；离线旧评审拆成两个，因此两套分母不同，不能混报精确错误率。",
              "Qwen 的混合层羽流均速未按全域公式扣分。它的范围尚未精确定义，当前也未给该 finding 正确分。",
              "DeepSeek 主 Pearson 只给三位小数，严格原 GT 容差仍未闭合；其 requirement recall 仍为区间。", "",
              f"累计新增调用 {result['cumulative_probe_calls']} 次；输入 {result['input_tokens']} token，输出 {result['output_tokens']} token（含 reasoning）。",
              "各调用时长："+', '.join(f"{a['seconds']:.1f}s" for a in attempts)+"。",
              "单次评审仍是主要成本；本次没有证明比旧链路更快。两个针对性样本不能估计整体准确率或证明模型排名。"]
    (args.output / "report.md").write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "attempts"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/model_metric_calibration")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/model_metric_calibration/faithfulness_audit")
    parser.add_argument("--live-probe", action="store_true", help="Make at most two new judge calls; never call solvers")
    args = parser.parse_args()
    run(args.source, args.output)
    if args.live_probe:
        live_probe(args.source, args.output)
