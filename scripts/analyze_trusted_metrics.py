#!/usr/bin/env python3
"""Offline evidence/cost audit. Controlled corruptions never enter model scores."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowintentbench.answer_evidence import _NUMBER
from flowintentbench.expansion_evaluation import load_development_case
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.trusted_scoring import digest, from_outcome, score
from scripts.evaluate_trusted_metrics import read, sha, load_source, paired_comparison


def cost_profile(output):
    # Receipts describe actual attempts, including failures with missing usage.
    receipts = [read(p) for p in sorted((output / "api").glob("*/receipt.json"))]
    report = read(output / "reports/experiment_report.json")
    history = [read(p) for p in (output / "cost_history").glob("*.json")]
    durations = [s for h in history for s in h.get("new_review_seconds", [])]
    successful = [r for r in receipts if r.get("completed")]
    tokens = {k: sum(r.get("usage", {}).get(k, 0) for r in receipts) for k in ("input_tokens", "output_tokens")}
    return {"actual_attempts": len(receipts), "completed_responses": len(successful),
        "failed_attempts": len(receipts) - len(successful), **tokens,
        "usage_missing_calls": sum(not all(k in r.get("usage", {}) for k in tokens) for r in receipts),
        "known_tokens_per_completed_response": (sum(tokens.values()) / len(successful)) if successful else None,
        "seconds_mean_including_failures": statistics.fmean(durations) if durations else None,
        "seconds_median_including_failures": statistics.median(durations) if durations else None,
        "source_report_sha256": sha(output / "reports/experiment_report.json"),
        "reported_cumulative": report["cumulative_new_review_cost"]}


def sensitivity(report, donor, manifest, output, limit=20):
    """Hold extraction/matching fixed to isolate the host's numeric gate."""
    selected = {x["run_id"]: x for x in read(donor / "selection.json")["answers"]}
    cases = {x["case_id"]: x for x in read(manifest)["cases"]}
    results = []
    for row in sorted(report["rows"], key=lambda x: (x["case_id"], x["model_id"], x["trial"])):
        if row["status"] != "SCORED" or row["run_id"] not in selected:
            continue
        if not any(c["verdict"] is True for c in row.get("finding_checks", [])):
            continue
        ci, meta, gt, material = load_development_case(ROOT, cases[row["case_id"]])
        answer, old, _ = load_source(selected[row["run_id"]], donor, ci, gt)
        provenance = row["provenance"]
        if provenance["source"].startswith("TRUSTED"):
            judgment = read(output / "judgments" / (provenance["request_id"] + ".json"))["judgment"]
        else:
            judgment = from_outcome(old, gt, [d.value for d in meta.principal_operationalization_dimensions])
        baseline = score(answer, meta, gt, material, judgment, condition=row["condition"])
        findings = {f["finding_id"]: f for f in judgment["findings"]}
        for check in baseline["finding_checks"]:
            if check["verdict"] is not True:
                continue
            claim = findings[check["finding_id"]]
            value = claim["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            quote = claim["evidence_text"]
            literal = next((m.group() for m in _NUMBER.finditer(quote)
                            if float(m.group().replace(",", "")) == value), None)
            if literal is None or quote not in answer:
                continue
            wrong = float(value) + max(abs(float(value)) * 2, 100.0)
            changed_quote = quote.replace(literal, str(wrong), 1)
            changed_answer = answer.replace(quote, changed_quote, 1)
            changed = deepcopy(judgment)
            for f in changed["findings"]:
                if f["finding_id"] == claim["finding_id"]:
                    f.update(value=wrong, evidence_text=changed_quote)
            for d in changed["dimensions"]:
                if quote in d["evidence_text"]:
                    d["evidence_text"] = d["evidence_text"].replace(quote, changed_quote, 1)
            result = score(changed_answer, meta, gt, material, changed, condition=row["condition"])
            target = next(c for c in result["finding_checks"] if all(c[k] == check[k] for k in ("finding_id", "branch_id", "reference_id")))
            result_row = {"run_id": row["run_id"], "case_id": row["case_id"], "model_id": row["model_id"],
                "condition": row["condition"], "answer_sha256": baseline["answer_sha256"],
                "original_value": value, "injected_value": wrong, "before": check, "after": target,
                "wrong_value_rejected": target["verdict"] is False,
                "metrics_before": baseline["metrics"], "metrics_after": result["metrics"]}
            result_row["no_quality_lower_bound_increased"] = all(
                result["metrics"][k]["lower"] <= baseline["metrics"][k]["lower"]
                for k in ("finding_precision", "finding_requirement_recall", "c_score")
                if baseline["metrics"][k]["applicable"] and result["metrics"][k]["applicable"])
            results.append(result_row)
            break
        if len(results) >= limit:
            break
    return {"kind": "HOST_ONLY_CONTROLLED_CORRUPTION", "new_api_calls": 0,
        "interpretation": "Actual answer excerpts, numeric corruption, fixed extraction/matching. Tests host sensitivity, not end-to-end judge accuracy or model ranking.",
        "examples": len(results), "wrong_value_rejected": sum(x["wrong_value_rejected"] for x in results),
        "no_quality_lower_bound_increased": sum(x["no_quality_lower_bound_increased"] for x in results), "rows": results}


def effort_comparison(medium, low):
    a, b = read(medium / "reports/experiment_report.json"), read(low / "reports/experiment_report.json")
    rows = {r["run_id"]: r for r in a["rows"] if r.get("provenance", {}).get("source", "").startswith("TRUSTED")}
    comparisons = []
    for row in b["rows"]:
        if row["run_id"] not in rows or not row.get("provenance", {}).get("source", "").startswith("TRUSTED"):
            continue
        entries = []
        for root, item in ((medium, rows[row["run_id"]]), (low, row)):
            stored = read(root / "judgments" / (item["provenance"]["request_id"] + ".json"))
            receipt = read(Path(stored["receipt"]))
            entries.append({"finding_count": len(stored["judgment"]["findings"]),
                            "usage": receipt.get("usage"), "metrics": item["metrics"],
                            "seconds": receipt["finished_epoch"] - receipt["started_epoch"]})
        comparisons.append({"run_id": row["run_id"], "case_id": row["case_id"], "model_id": row["model_id"],
                            "medium": entries[0], "low": entries[1]})
    return comparisons


def run(args):
    report = read(args.output / "reports/experiment_report.json")
    fresh = [r for r in report["rows"] if r.get("provenance", {}).get("source", "").startswith("TRUSTED")]
    reasons = Counter(c["reason"].split(":")[0].split(";")[0] for r in report["rows"]
                      for c in r.get("finding_checks", []) if c["verdict"] is None)
    result = {"source_report_sha256": sha(args.output / "reports/experiment_report.json"),
        "full_slots": report["requested_slots"], "answers": report["answered_slots"],
        "freshly_reviewed_answers": len(fresh), "fresh_paired_results": paired_comparison(fresh),
        "all_paired_results": report["paired_comparisons"], "unresolved_finding_check_reasons": dict(reasons),
        "controlled_sensitivity": sensitivity(report, args.donor, args.manifest, args.output),
        "cost": {"medium": cost_profile(args.medium), "low": cost_profile(args.low)},
        "effort_comparison": effort_comparison(args.medium, args.low),
        "performance_correlation": {"status": "NOT_ESTABLISHED", "independent_reference_labels": 0,
            "reason": "Model labels are not ground-truth ability ranks. Numeric sensitivity is insufficient to validate semantic extraction or open-answer completeness."}}
    write_json(args.output / "reports/validation_report.json", result)
    t = result["controlled_sensitivity"]
    lines = ["# 评估链路验证报告", "", "## 当前结论", "",
        f"覆盖 {report['requested_slots']} 个计划槽位，其中 {report['answered_slots']} 份已有回答已离线重算；"
        f"{len(fresh)} 份回答完成了新协议真实评审。全量点估计状态：`{report['status']}`。",
        "目前可以输出每项指标的已知值、未知区间和覆盖率；尚不能给出可信的全量模型排序，也没有独立标签可用于验证性能相关性。",
        "区间是证据不足对应的识别界，不是统计置信区间。它们依赖语义抽取/匹配判断正确；不覆盖 judge 的未知错误。", "",
        "## 改动与数据约束", "",
        "- 局部未决只影响相关指标；明确缺失的必答内容扣分；API 错误保留为评审错误。",
        "- 全部计划槽位保留，每个 case 等权、case 内 trial 等权；未采集的适用指标为 [0,1]。",
        "- 原始数字、单位和引文必须绑定原答案。GT 之外的科学结论保留未知，不靠措辞合理性直接判真。",
        "- 冻结容差不变。若打印精度可以解释超差，标记舍入不确定性；不能据此直接判对。",
        "- 旧 rubric 的 MET/PARTIAL 不转换为原始 O/F/C 得分。新旧语义证据来源分别记录。",
        "- 原上传快照缺少 RunRecord 原文件时，从历史请求恢复 SHA 一致的回答；这不能恢复未上传的执行证据。", "",
        "## 新评审的同题对照", "",
        "| 指标 | 配对数 | Fable − Sonnet 的 case 宏平均差值界 | Fable 确定胜 / Sonnet 确定胜 |",
        "|---|---:|---|---|" ]
    for pair in result["fresh_paired_results"]:
        for name, v in pair["metrics"].items():
            lo, hi = v["left_minus_right_bounds"]
            bound = "N/A" if lo is None else f"[{lo:.4f}, {hi:.4f}]"
            lines.append(f"| {name} | {v['pairs']} | {bound} | {v['left_definite_wins']} / {v['right_definite_wins']} |")
    lines += ["", "这些样本按条件和 case 标识选取，没有按得分或预期模型顺序选取；5 个 case 的结果不能外推至全部任务。",
        "两模型历史已回答集合不相同，报告的各自 answered-only 均值也不能直接作为公平性能比较。", "",
        "## 可验证的计分敏感性", "",
        f"对 {t['examples']} 份真实回答中的已验证数值构造明显错误值，固定抽取/匹配，"
        f"主机拒绝错误值 {t['wrong_value_rejected']}/{t['examples']}；"
        f"相关质量下界均未上升 {t['no_quality_lower_bound_increased']}/{t['examples']}。新增 API 调用为 0。",
        "这验证数值计分阶段能响应错误。它没有验证 judge 在重新阅读错误答案时是否会正确抽取，也没有建立模型能力的相关系数。",
        "测试变体单独保存为验证结果，没有写入模型评估行。", "",
        "## 时间与 token", "",
        "| 配置 | 实际请求 | 成功响应 | 已知 input / output token | 请求平均耗时（含失败） |",
        "|---|---:|---:|---|---:|"]
    for effort, c in result["cost"].items():
        lines.append(f"| {effort} | {c['actual_attempts']} | {c['completed_responses']} | {c['input_tokens']} / {c['output_tokens']} | {c['seconds_mean_including_failures']:.2f} 秒 |")
    lines += ["", "medium 的一次超时没有 usage 回执，上述 token 是已知下限。重试由下一次显式运行触发，无自动无限重试。",
        "两档对同一压力问题给出的 finding 数量不同：", "",
        "| 模型 | medium finding 数 | low finding 数 |", "|---|---:|---:|"]
    for c in result["effort_comparison"]:
        lines.append(f"| {c['model_id']} | {c['medium']['finding_count']} | {c['low']['finding_count']} |")
    if result["effort_comparison"]:
        shared = result["effort_comparison"]
        shared_tokens = {e: sum(sum(c[e]["usage"].get(k, 0) for k in ("input_tokens", "output_tokens")) for c in shared)
                         for e in ("medium", "low")}
        shared_seconds = {e: statistics.fmean(c[e]["seconds"] for c in shared) for e in ("medium", "low")}
        lines += ["", f"同两份答案的公平成本对照：medium 共 {shared_tokens['medium']} token、平均 {shared_seconds['medium']:.2f} 秒；"
            f"low 共 {shared_tokens['low']} token、平均 {shared_seconds['low']:.2f} 秒。"
            f"low 节省 {1-shared_tokens['low']/shared_tokens['medium']:.1%} token，"
            f"耗时降低 {1-shared_seconds['low']/shared_seconds['medium']:.1%}。该配置同时调整 effort 和输出上限，不将差异归因于单一因素。"]
    lines += ["", "因此暂时保留 medium 默认，low 可选。降低 effort 可省成本，但这 2 份回答未证明其科学等价性。",
        "离线重算 177 份已有证据不调用 API；新评审每份一次，默认每轮最多 8 次、输出最多 6000 token、2 个并发工作线程。",
        "这些上限控制单轮开销，不能保证每份都能在上限内闭合。完整长答案超出提示预算时保留未知，不静默截断。", "",
        "## 尚未解决的科学覆盖问题", "",
        "1. 398 个槽位没有模型回答；评估器不能生成这些答案的质量点估计。",
        "2. 原始 GT 只覆盖少量 finding。新增合理科学结论缺少独立参考/执行证据，precision 与 C 仍可能有较宽区间。",
        "3. 新评审实测出现数量语义误配、无法定位的引文及 mixed-unit tuple；已增加 threshold/percentile 对 mean 的冲突检查并保留未知，其他语义错误仍需校准。",
        "4. 全量相关性判断还需要盲法独立标注：至少覆盖四种条件、两模型、已知和未知项；先核验匹配/完整性，再比较配对指标。",
        "5. 不应为了得到两个模型有差距的单个数字，放宽 GT 容差、把未知当错或按模型身份调整打分。", "",
        "完整逐项数值、真实请求回执和主机变体结果见同目录 JSON。"]
    (args.output / "reports/validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"fresh_answers": len(fresh), "controlled_examples": t["examples"],
        "controlled_rejections": t["wrong_value_rejected"], "cost": result["cost"]}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/claude_resume_eval/trusted_metrics")
    p.add_argument("--donor", type=Path, default=ROOT / "outputs/claude_outcomes_gpt6_complete_177")
    p.add_argument("--manifest", type=Path, default=ROOT / "experiments/expansion_v1_development/case_manifest.json")
    p.add_argument("--medium", type=Path, default=ROOT / "outputs/claude_trusted_metrics_yiapi")
    p.add_argument("--low", type=Path, default=ROOT / "outputs/claude_trusted_metrics_low_pilot")
    run(p.parse_args())
