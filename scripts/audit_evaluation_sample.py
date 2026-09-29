#!/usr/bin/env python3
"""Summarize same-question model evaluations and enforce a scientific readiness gate.

Offline only. Receipts account for actual calls, including failed attempts and
focused repairs. No assumed model ranking or scientific reference construction.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowintentbench.external_file_evaluator import write_json
from flowintentbench.rubric_scoring import METRICS


def number(lo, hi):
    if lo is None:
        return "N/A"
    return f"{lo:.3f}" if lo == hi else f"[{lo:.3f}, {hi:.3f}]"


def discrimination(rows):
    """Describe observed scores without assigning capability labels to models."""
    result = {}
    for name in METRICS:
        applicable = [r['metrics'][name] for r in rows if r['metrics'][name]['applicable']]
        points = [m['value'] for m in applicable if m['value'] is not None and not m.get('applicability_unknown')]
        result[name] = {'applicable':len(applicable), 'identified':len(points),
            'identified_min':min(points, default=None), 'identified_max':max(points, default=None),
            'identified_at_one':sum(v == 1 for v in points),
            'all_applicable_identified_and_equal':bool(applicable) and len(points) == len(applicable) and len(set(points)) == 1,
            'observed_point_variation':len(set(points)) > 1,
            'interpretation':'Descriptive sample diagnostic; constant points or overlapping bounds do not establish equal model ability.'}
    return result


def identified_contrasts(rows):
    """Same-question differences that hold for every value within both bounds."""
    counts=Counter((r['model_id'],r['case_id'],r['trial']) for r in rows)
    comparisons=[]
    for left in rows:
        if left['status']!='SCORED' or counts[(left['model_id'],left['case_id'],left['trial'])]!=1:continue
        for right in rows:
            if (right['status']!='SCORED' or left['model_id']==right['model_id']
                or (left['case_id'],left['trial'])!=(right['case_id'],right['trial'])
                or counts[(right['model_id'],right['case_id'],right['trial'])]!=1):continue
            for name in METRICS:
                a,b=left['metrics'][name],right['metrics'][name]
                if (not a['applicable'] or not b['applicable'] or a.get('applicability_unknown')
                    or b.get('applicability_unknown') or a['lower'] is None or b['upper'] is None):continue
                if a['lower']>b['upper']:
                    comparisons.append({'case_id':left['case_id'],'trial':left['trial'],'metric':name,
                        'higher_model':left['model_id'],'lower_model':right['model_id'],
                        'higher_lower_bound':a['lower'],'lower_upper_bound':b['upper'],
                        'minimum_gap':a['lower']-b['upper']})
    return comparisons


def audit(report, receipt_roots=()):
    rows = report["rows"]
    slots = {model: {(r["case_id"], r["trial"]) for r in rows if r["model_id"] == model}
             for model in report["models"]}
    slot_counts = Counter((r['model_id'], r['case_id'], r['trial']) for r in rows)
    duplicate_slots = [list(slot) for slot, count in slot_counts.items() if count > 1]
    same_questions = (bool(slots) and all(slots.values()) and not duplicate_slots
                      and set(slots) == {r['model_id'] for r in rows}
                      and len({tuple(sorted(v)) for v in slots.values()}) == 1)
    intervals = [(r["model_id"], r["case_id"], name) for r in rows for name, m in r["metrics"].items()
                 if m.get("applicability_unknown") or (m["applicable"] and m["value"] is None)]
    metrics = {name: {"applicable": sum(r["metrics"][name]["applicable"] for r in rows),
                     "identified": sum(r["metrics"][name]["applicable"] and r["metrics"][name]["value"] is not None
                                       and not r["metrics"][name].get("applicability_unknown") for r in rows)}
               for name in METRICS}
    receipts = {}
    for folder in receipt_roots:
        for path in Path(folder).rglob("receipt.json"):
            receipts[path.resolve()] = json.loads(path.read_text())
    not_sent = {p for p, r in receipts.items()
                if "Shared API admission deadline exceeded before sending" in str(r.get("error", ""))}
    timings = [r["finished_epoch"]-r["started_epoch"] for r in receipts.values()
               if r.get("finished_epoch") and r.get("started_epoch")]
    cost = {"review_attempts": len(receipts), "local_admission_failures": len(not_sent),
        "api_calls": len(receipts) - len(not_sent), "completed_calls": sum(bool(r.get("completed")) for r in receipts.values()),
        "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in receipts.values()),
        "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in receipts.values()),
        "usage_missing_calls": sum(p not in not_sent and not all(k in r.get("usage", {}) for k in ("input_tokens", "output_tokens")) for p, r in receipts.items()),
        "seconds_mean": statistics.fmean(timings) if timings else None,
        "seconds_median": statistics.median(timings) if timings else None,
        "seconds_max": max(timings, default=None), "requested_output_limit_exceeded_calls": 0}
    for path, receipt in receipts.items():
        request = json.loads(path.with_name("request.json").read_text())
        limit = request.get("max_output_tokens")
        cost["requested_output_limit_exceeded_calls"] += int(bool(limit and receipt.get("usage", {}).get("output_tokens", 0) > limit))
    # Receipt roots can cover only a supplement or repair stage. Dividing that
    # cost by every retained report row would understate the cost per reviewed
    # answer (for example, seven RT supplements in a 28-answer report).
    known_tokens = cost["input_tokens"] + cost["output_tokens"]
    cost["scope"] = "SUPPLIED_RECEIPT_ROOTS_ONLY"
    cost["token_totals_complete"] = cost["usage_missing_calls"] == 0
    cost["known_tokens_per_api_call"] = known_tokens / cost["api_calls"] if cost["api_calls"] else None
    cost["known_tokens_amortized_per_report_answer"] = known_tokens / len(rows) if rows else None
    cost["production_tokens_per_answer"] = None
    base_complete = bool(rows) and all(r["status"] in {"SCORED", "MODEL_NONCOMPLETION"} for r in rows)
    extension_complete = report.get('source_identity', {}).get('extension_review_complete', True)
    complete = base_complete and extension_complete is True
    return {"protocol": report["protocol"], "answers": len(rows), "models": len(slots), "same_questions": same_questions,
        "status_counts": dict(Counter(r["status"] for r in rows)), "review_complete": complete,
        "base_review_complete": base_complete, "extension_review_complete": extension_complete,
        "duplicate_slots": duplicate_slots,
        "full_point_comparison_ready": complete and same_questions and not intervals,
        "model_performance_correlation": "NOT_ESTABLISHED: no independent capability labels or human calibration",
        "metric_identification": metrics, "unidentified_metric_slots": intervals,
        "alignment_kinds": dict(Counter(r.get("branch_alignment_diagnostic", {}).get("kind", "UNREVIEWED") for r in rows)),
        "natural_of_mismatches_demonstrated": sum(r.get("of_consistency", {}).get("mismatch_demonstrated", False) for r in rows),
        "metric_discrimination": discrimination(rows),
        "identified_same_question_contrasts":identified_contrasts(rows),
        "metric_discrimination_by_condition": {c:discrimination([r for r in rows if r.get('condition') == c])
            for c in sorted({r['condition'] for r in rows if r.get('condition')})},
        "source_binding_failures": sum(len({x['finding_id'] for x in r.get('finding_checks', [])
            if x.get('reason') in {'VALUE_NOT_IN_SOURCE', 'QUOTE_NOT_FOUND', 'UNIT_NOT_IN_SOURCE'}}) for r in rows),
        "reference_gap_reasons": dict(Counter(g["reason"] for r in rows for g in r.get("reference_gaps", []))),
        "receipt_roots": sorted({str(Path(p).resolve()) for p in receipt_roots}),
        "evaluation_cost": cost}


def publish(report_path, output, receipt_roots=()):
    report = json.loads(report_path.read_text())
    result = audit(report, receipt_roots)
    result["source_report"] = str(report_path.resolve())
    write_json(output / "sample_audit.json", result)
    lines = ["# 七模型同题小样本评估核查" if result["models"] == 7 else "# 同题小样本评估核查", "",
        f"模型 {result['models']} 个，答案 {result['answers']} 份；状态：{result['status_counts']}。", "",
        f"评审完成：{result['review_complete']}；全量点值比较门槛通过：{result['full_point_comparison_ready']}。", "",
        f"主评审完成：{result['base_review_complete']}；所选补评完成：{result['extension_review_complete']}。未完成的补评保留此前的分数区间，不代表补评成功。", "",
        "区间是证据不足时的识别界；下界不是估计的真实质量，不能按下界给模型排名。N/A 不计入该指标分母。", "",
        "## 各模型指标（case 等权；每题先平均 trials）", "",
        "| 模型 | " + " | ".join(METRICS) + " |",
        "|---|" + "---|" * len(METRICS)]
    for model, summary in report["models"].items():
        metrics = summary["answered_only"]
        lines.append("| " + model + " | " + " | ".join(number(metrics[n]["case_macro_lower"], metrics[n]["case_macro_upper"]) for n in METRICS) + " |")
        write_json(output / "by_model" / (model + ".json"), {"model_id": model, "summary": summary,
            "rows": [r for r in report["rows"] if r["model_id"] == model]})
    lines += ["", "## 点值覆盖率", "", "| 指标 | 已确定 / 适用答案 |", "|---|---|"]
    lines += [f"| {name} | {v['identified']} / {v['applicable']} |" for name, v in result["metric_identification"].items()]
    lines += ['', '## 分数变化与天花板检查', '', '| 指标 | 已确定分数范围 | 得 1 的答案 / 已确定答案 | 全部适用答案同分 |', '|---|---|---|---|']
    for name, v in result['metric_discrimination'].items():
        lines.append(f"| {name} | {number(v['identified_min'], v['identified_max'])} | {v['identified_at_one']} / {v['identified']} | {v['all_applicable_identified_and_equal']} |")
    lines += ['', '只统计已识别点值；区间没有被删除后当作点值参与比较。同分、区间重叠都不能证明模型能力相同。']
    lines += ['', '## 可确定的同题差异', '',
        '仅当一个答案的下界严格高于另一个答案的上界时列出；这不把下界当作真实分数，也不构成模型综合能力排名。', '',
        '| 题目 / trial | 指标 | 较高答案 | 较低答案 | 已保证的最小差距 |', '|---|---|---|---|---|']
    for c in result['identified_same_question_contrasts']:
        lines.append(f"| {c['case_id']} / {c['trial']} | {c['metric']} | {c['higher_model']} | {c['lower_model']} | {c['minimum_gap']:.4f} |")
    if not result['identified_same_question_contrasts']:lines.append('| — | — | 无可确定差异 | — | — |')
    lines += ["", "## 必答结果（逐题）", "", "| 模型 | 题目 | O | 必答 recall | precision | C | alignment 类型 |", "|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        values = [number(row["metrics"][n]["lower"], row["metrics"][n]["upper"]) for n in ("o_score", "finding_requirement_recall", "finding_precision", "c_score")]
        lines.append("| " + " | ".join([row["model_id"], row["case_id"], *values, row.get("branch_alignment_diagnostic", {}).get("kind", "UNREVIEWED")]) + " |")
    lines += ["", "## 诊断与成本", "", f"- alignment 分类：`{result['alignment_kinds']}`。",
        f"- 已证明的自然 O–F 方法错配：{result['natural_of_mismatches_demonstrated']} 份；不能用受控单元测试代替自然样本证据。",
        f"- 已确定参考矛盾数（逐模型）：`{report['verified_error_counts']}`。",
        f"- 真实调用及修复成本：`{json.dumps(result['evaluation_cost'], ensure_ascii=False)}`。", "",
        "成本只覆盖显式传入的回执目录。按报告答案数分摊的成本不是单份答案的生产评审成本；补评可能只涉及部分答案，缺失用量也未计入已知 tokens。", "",
        "本报告不证明模型综合能力排序。少量同题答案可以发现评分错误、数值差异和参考缺口，不能替代独立人工标注、更多题目及重复试验。"]
    (output / "sample_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt-root", type=Path, action="append", default=[])
    p.add_argument("--require-identified", action="store_true")
    args = p.parse_args()
    result = publish(args.report, args.output, args.receipt_root)
    print(json.dumps({k: result[k] for k in ("answers", "models", "status_counts", "review_complete", "full_point_comparison_ready")}))
    return 3 if args.require_identified and not result["full_point_comparison_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
