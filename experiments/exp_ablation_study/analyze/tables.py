# -*- coding: utf-8 -*-
"""表格生成：Markdown + LaTeX。"""

import numpy as np

from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, SCENARIO_LABELS
from .stats import (
    mannwhitney_test, p_mark, summarize_llm_decisions,
    compute_f_stats, compute_gmr_stats, compute_parameter_coupling,
)


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
        r"\caption{统计显著性检验（Mann-Whitney U，A1 vs A0）}",
        r"\label{tab:mannwhitney}",
        r"\begin{tabular}{lc}", r"\toprule",
        r"场景 & A1 vs A0 \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        a3 = all_stats.get(f"{s_key}_A1", {}).get("fitness_array", np.array([]))
        row = [SCENARIO_LABELS[s_key]]
        for c_key in ["A0"]:
            other = all_stats.get(f"{s_key}_{c_key}", {}).get("fitness_array", np.array([]))
            p = mannwhitney_test(a3, other)
            row.append("---" if p < 0 else f"{p:.4f} {p_mark(p)}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"",
        r"\footnotesize{$*\, p<0.05$，$**\, p<0.01$，n.s. 不显著}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table3(all_stats: dict) -> str:
    """计算时间对比表。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{计算时间对比（分口径）}", r"\label{tab:time_breakdown}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"场景 & 配置 & 总耗时(s) & DMDE(s) & LLM总耗时(s) & LLM CR(s) \\",
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
            llm_cr = f"{stats['llm_cr_time_mean']:.1f}" if stats.get('llm_cr_time_mean', 0) > 0 else "---"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {total} & {dmde} & {llm} & {llm_cr} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {total} & {dmde} & {llm} & {llm_cr} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── Markdown ──────────────────────────────────────────────

def generate_markdown_table(all_stats: dict) -> str:
    lines = [
        "| 场景 | 配置 | Best | Mean ± Std | Median | 总耗时(s) | DMDE(s) | LLM总耗时(s) | LLM CR(s) | LLM调用次数 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                lines.append(f"| {s_key} | {c_key} | --- | --- | --- | --- | --- | --- | --- | --- |")
                continue
            llm_t = f"{stats['llm_time_mean']:.1f}" if stats['llm_time_mean'] > 0 else "---"
            llm_cr = f"{stats['llm_cr_time_mean']:.1f}" if stats.get('llm_cr_time_mean', 0) > 0 else "---"
            llm_c = f"{stats['llm_calls_mean']:.0f}" if stats['llm_calls_mean'] > 0 else "---"
            dmde_t = f"{stats['dmde_time_mean']:.1f}" if stats.get('dmde_time_mean', 0) > 0 else "---"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {stats['best']:.2f} | {stats['mean']:.2f} ± {stats['std']:.2f} "
                f"| {stats['median']:.2f} | {stats['total_time_mean']:.1f} "
                f"| {dmde_t} | {llm_t} | {llm_cr} | {llm_c} |"
            )
    return "\n".join(lines)


def generate_llm_decision_summary(all_stats: dict) -> str:
    lines = [
        "## LLM 决策分析\n",
        "| 场景 | 配置 | LLM模块 | 调用次数 | 平均耗时(s) | 总耗时(s) | CR均值 | CR标准差 | F均值 | F标准差 | GMR模式分布 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A1"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            summary = summarize_llm_decisions(raw)
            if not summary:
                lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                continue
            for mod_name, mod_data in summary.items():
                cr_mean = f"{mod_data.get('cr_mean', 0):.4f}" if "cr_mean" in mod_data else "---"
                cr_std = f"{mod_data.get('cr_std', 0):.4f}" if "cr_std" in mod_data else "---"
                f_mean = f"{mod_data.get('f_mean', 0):.4f}" if "f_mean" in mod_data else "---"
                f_std = f"{mod_data.get('f_std', 0):.4f}" if "f_std" in mod_data else "---"
                # GMR 模式分布
                gmr_pcts = mod_data.get("gmr_mode_pcts", {})
                if gmr_pcts:
                    gmr_str = ", ".join(
                        f"{m}:{p:.0f}%" for m, p in sorted(gmr_pcts.items())
                    )
                else:
                    gmr_str = "---"
                lines.append(
                    f"| {s_key} | {CONFIG_LABELS[c_key]} | {mod_name} "
                    f"| {mod_data['total_calls']} | {mod_data['avg_duration']:.3f} "
                    f"| {mod_data['total_duration']:.1f} | {cr_mean} | {cr_std} "
                    f"| {f_mean} | {f_std} | {gmr_str} |"
                )
    return "\n".join(lines)


def generate_convergence_speed_table(conv_gens: dict, thresholds: list[float] = None) -> str:
    """收敛速度量化表（Markdown）。"""
    if thresholds is None:
        thresholds = [0.90, 0.95, 0.99]
    from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, SCENARIO_LABELS
    header = "| 场景 | 配置 | " + " | ".join(f"达到{int(t*100)}%改进" for t in thresholds) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(thresholds)) + "|"
    lines = [header, sep]
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            data = conv_gens.get(f"{s_key}_{c_key}", {})
            if not data:
                continue
            vals = " | ".join(
                f"Gen {data.get(t, -1)}" if data.get(t, -1) >= 0 else "---"
                for t in thresholds
            )
            lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | {vals} |")
    return "\n".join(lines)


def generate_latex_convergence_speed_table(conv_gens: dict, thresholds: list[float] = None) -> str:
    """收敛速度量化表（LaTeX）。"""
    if thresholds is None:
        thresholds = [0.90, 0.95, 0.99]
    from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, SCENARIO_LABELS
    cols = "l" + "l" + "c" * len(thresholds)
    th_headers = " & ".join(f"达到 {int(t*100)}\\% 改进" for t in thresholds)
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{收敛速度对比：达到 X\% 改进所需的代数（cost 场景下取中位数）}",
        r"\label{tab:convergence_speed}",
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        f"场景 & 配置 & {th_headers} \\\\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        first_row = True
        for c_key in CONFIGS:
            data = conv_gens.get(f"{s_key}_{c_key}", {})
            if not data:
                continue
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            vals = " & ".join(
                str(data.get(t, -1)) if data.get(t, -1) >= 0 else "---"
                for t in thresholds
            )
            if first_row:
                lines.append(f"{s_label} & {c_label} & {vals} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {vals} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_f_stats_table(all_stats: dict) -> str:
    """F 值统计表（Markdown）。

    展示各配置中 F 值的分布特征，验证 LLM 独立控制 F 的效果。
    """
    lines = [
        "## F (Scale Factor) 统计分析\n",
        "| 场景 | 配置 | F均值 | F标准差 | F最小值 | F最大值 | F中位数 | LLM覆写次数 | LLM覆写F均值 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            f_stats = compute_f_stats(raw)
            if not f_stats:
                lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | --- | --- | --- | --- | --- | --- | --- |")
                continue
            override_count = f_stats.get("f_override_count", 0)
            override_mean = f"{f_stats.get('f_override_mean', 0):.4f}" if override_count > 0 else "---"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {f_stats['f_mean']:.4f} | {f_stats['f_std']:.4f} "
                f"| {f_stats['f_min']:.4f} | {f_stats['f_max']:.4f} "
                f"| {f_stats['f_median']:.4f} | {override_count} | {override_mean} |"
            )
    return "\n".join(lines)


def generate_gmr_stats_table(all_stats: dict) -> str:
    """GMR 模式统计表（Markdown）。

    展示 LLM 在 search_controller 中对 GMR 模式的选择分布。
    """
    lines = [
        "## GMR (Extinction) 模式分析\n",
        "| 场景 | 配置 | 总决策次数 | 模式分布 | 模式切换次数 |",
        "|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A1"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            gmr_stats = compute_gmr_stats(raw)
            if not gmr_stats:
                lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | --- | --- | --- |")
                continue
            pcts = gmr_stats.get("mode_pcts", {})
            mode_str = ", ".join(f"{m}: {p:.0f}%" for m, p in sorted(pcts.items()))
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {gmr_stats['total_decisions']} "
                f"| {mode_str} "
                f"| {gmr_stats['mode_switches']} |"
            )
    return "\n".join(lines)


def generate_parameter_coupling_table(all_stats: dict) -> str:
    """参数耦合分析表（Markdown）。

    分析 CR、F、GMR 三个参数的独立性和相关性，
    验证解耦控制的有效性。
    """
    lines = [
        "## 参数解耦分析\n",
        "| 场景 | 配置 | CR-F相关系数 | CR均值±标准差 | F均值±标准差 | GMR模式分布 | 样本数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A0", "A1"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            coupling = compute_parameter_coupling(raw)
            if not coupling:
                lines.append(f"| {s_key} | {CONFIG_LABELS[c_key]} | --- | --- | --- | --- | --- |")
                continue
            corr = coupling.get("cr_f_correlation", 0)
            # 相关系数解释
            if abs(corr) < 0.1:
                corr_label = "≈ 0 (独立)"
            elif abs(corr) < 0.3:
                corr_label = f"{corr:+.4f} (弱相关)"
            elif abs(corr) < 0.7:
                corr_label = f"{corr:+.4f} (中等相关)"
            else:
                corr_label = f"{corr:+.4f} (强相关)"
            cr_str = f"{coupling['cr_mean']:.4f} ± {coupling['cr_std']:.4f}"
            f_str = f"{coupling['f_mean']:.4f} ± {coupling['f_std']:.4f}"
            gmr_counts = coupling.get("gmr_mode_counts", {})
            gmr_str = ", ".join(f"{m}: {c}" for m, c in sorted(gmr_counts.items())) if gmr_counts else "---"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {corr_label} | {cr_str} | {f_str} | {gmr_str} | {coupling['n_samples']} |"
            )
    return "\n".join(lines)
