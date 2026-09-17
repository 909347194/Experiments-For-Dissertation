# -*- coding: utf-8 -*-
"""表格生成：Markdown + LaTeX。"""

import numpy as np

from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, SCENARIO_LABELS
from .stats import mannwhitney_test, p_mark, summarize_llm_decisions


# ── LaTeX ─────────────────────────────────────────────────

def generate_latex_table1(all_stats: dict) -> str:
    """解质量对比表（按场景分组，最优值加粗）。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{消融实验结果：解质量对比（最优值加粗）}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"场景 & 配置 & Best & Mean $\pm$ Std & Median \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        group = []
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if stats:
                group.append((c_key, stats))
        if not group:
            continue
        best_min = min(s["best"] for _, s in group)
        mean_min = min(s["mean"] for _, s in group)
        median_min = min(s["median"] for _, s in group)
        first_row = True
        for c_key, stats in group:
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            best_s = f"{stats['best']:.2f}"
            mean_s = f"{stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
            median_s = f"{stats['median']:.2f}"
            if len(group) > 1:
                if stats["best"] == best_min:
                    best_s = r"\textbf{" + best_s + r"}"
                if stats["mean"] == mean_min:
                    mean_s = mean_s.replace(
                        f"{stats['mean']:.2f}",
                        r"\textbf{" + f"{stats['mean']:.2f}" + r"}", 1,
                    )
                if stats["median"] == median_min:
                    median_s = r"\textbf{" + median_s + r"}"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {best_s} & {mean_s} & {median_s} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {best_s} & {mean_s} & {median_s} \\\\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_table2(all_stats: dict) -> str:
    """Mann-Whitney U 显著性检验表。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{统计显著性检验（Mann-Whitney U，A3 vs 基线）}",
        r"\label{tab:mannwhitney}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"场景 & A3 vs A0 & A3 vs A1 & A3 vs A2 \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        a3 = all_stats.get(f"{s_key}_A3", {}).get("fitness_array", np.array([]))
        row = [SCENARIO_LABELS[s_key]]
        for c_key in ["A0", "A1", "A2"]:
            other = all_stats.get(f"{s_key}_{c_key}", {}).get("fitness_array", np.array([]))
            p = mannwhitney_test(a3, other)
            row.append("---" if p < 0 else f"{p:.4f} {p_mark(p)}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"",
        r"\footnotesize{*$p < 0.05$，**$p < 0.01$，n.s. 不显著}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table3(all_stats: dict) -> str:
    """计算时间对比表。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{计算时间对比（分口径）}", r"\label{tab:time_breakdown}",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"场景 & 配置 & 总耗时(s) & DMDE(s) & LLM总耗时(s) & LLM Init(s) & LLM CR(s) \\",
        r"\midrule",
    ]
    for s_key in SCENARIOS:
        first_row = True
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                continue
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            total = f"{stats['total_time_mean']:.1f}"
            dmde = f"{stats['dmde_time_mean']:.1f}"
            llm = f"{stats['llm_time_mean']:.1f}" if stats['llm_time_mean'] > 0 else "---"
            llm_init = f"{stats['llm_init_time_mean']:.1f}" if stats.get('llm_init_time_mean', 0) > 0 else "---"
            llm_cr = f"{stats['llm_cr_time_mean']:.1f}" if stats.get('llm_cr_time_mean', 0) > 0 else "---"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {total} & {dmde} & {llm} & {llm_init} & {llm_cr} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {total} & {dmde} & {llm} & {llm_init} & {llm_cr} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Markdown ──────────────────────────────────────────────

def generate_markdown_table(all_stats: dict) -> str:
    lines = [
        "| 场景 | 配置 | Best | Mean ± Std | Median | 总耗时(s) | DMDE(s) | LLM总耗时(s) | LLM Init(s) | LLM CR(s) | LLM调用次数 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                lines.append(f"| {s_key} | {c_key} | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                continue
            llm_t = f"{stats['llm_time_mean']:.1f}" if stats['llm_time_mean'] > 0 else "---"
            llm_init = f"{stats['llm_init_time_mean']:.1f}" if stats.get('llm_init_time_mean', 0) > 0 else "---"
            llm_cr = f"{stats['llm_cr_time_mean']:.1f}" if stats.get('llm_cr_time_mean', 0) > 0 else "---"
            llm_c = f"{stats['llm_calls_mean']:.0f}" if stats['llm_calls_mean'] > 0 else "---"
            dmde_t = f"{stats['dmde_time_mean']:.1f}" if stats.get('dmde_time_mean', 0) > 0 else "---"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {stats['best']:.2f} | {stats['mean']:.2f} ± {stats['std']:.2f} "
                f"| {stats['median']:.2f} | {stats['total_time_mean']:.1f} "
                f"| {dmde_t} | {llm_t} | {llm_init} | {llm_cr} | {llm_c} |"
            )
    return "\n".join(lines)


def generate_llm_decision_summary(all_stats: dict) -> str:
    lines = [
        "## LLM 决策分析\n",
        "| 场景 | 配置 | LLM模块 | 调用次数 | 平均耗时(s) | 总耗时(s) | CR均值 | CR标准差 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A1", "A2", "A3"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            summary = summarize_llm_decisions(raw)
            if not summary:
                lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | --- | --- | --- | --- | --- | --- |")
                continue
            for mod_name, mod_data in summary.items():
                cr_mean = f"{mod_data.get('cr_mean', 0):.4f}" if "cr_mean" in mod_data else "---"
                cr_std = f"{mod_data.get('cr_std', 0):.4f}" if "cr_std" in mod_data else "---"
                lines.append(
                    f"| {s_key} | {CONFIG_LABELS[c_key]} | {mod_name} "
                    f"| {mod_data['total_calls']} | {mod_data['avg_duration']:.3f} "
                    f"| {mod_data['total_duration']:.1f} | {cr_mean} | {cr_std} |"
                )
    return "\n".join(lines)