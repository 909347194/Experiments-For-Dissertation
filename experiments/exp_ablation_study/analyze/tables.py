# -*- coding: utf-8 -*-
"""表格生成：Markdown + LaTeX。"""

import numpy as np

from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, SCENARIO_LABELS
from .stats import (
    mannwhitney_test, p_mark, summarize_llm_decisions,
    compute_f_stats, compute_gmr_stats, compute_parameter_coupling,
    compute_decoupling_effect,
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
        r"\footnotesize{$*\, p<0.05$，$**\, p<0.01$，n.s. 不显著}",
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
        "| 场景 | 配置 | LLM模块 | 调用次数 | 平均耗时(s) | 总耗时(s) | CR均值 | CR标准差 | F均值 | F标准差 | GMR模式分布 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A1", "A2", "A3"]:
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


def generate_synergy_table(synergy: dict) -> str:
    """协同效应分析表。"""
    lines = [
        "| 场景 | A0 均值 | ΔA1 (CR) | ΔA2 (PopInit) | ΔA3 (Full) | ΔA1+ΔA2 | 协同比 | 结论 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s_key, d in synergy.items():
        ratio = d["synergy_ratio"]
        if ratio is None or d.get("is_meaningful") is False:
            conclusion = "无显著差异"
        elif ratio > 1.05:
            conclusion = "协同增益"
        elif ratio >= 0.95:
            conclusion = "近似加性"
        else:
            conclusion = "存在冗余"
        ratio_s = f"{ratio:.3f}" if ratio is not None else "n/a"
        lines.append(
            f"| {s_key} | {d['A0_mean']:.2f} "
            f"| {d['delta_A1']:+.2f} | {d['delta_A2']:+.2f} "
            f"| {d['delta_A3']:+.2f} | {d['sum_delta']:+.2f} "
            f"| {ratio_s} | {conclusion} |"
        )
    return "\n".join(lines)


def generate_latex_synergy_table(synergy: dict) -> str:
    """LaTeX 协同效应表。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{协同效应分析：A3 增益 vs A1+A2 增益之和}",
        r"\label{tab:synergy}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"场景 & $\Delta A_1$ (CR) & $\Delta A_2$ (PopInit) & $\Delta A_3$ (Full) "
        r"& $\Delta A_1 + \Delta A_2$ & 协同比 & 结论 \\", r"\midrule",
    ]
    for s_key, d in synergy.items():
        ratio = d["synergy_ratio"]
        if ratio is None or d.get("is_meaningful") is False:
            conclusion = "无显著差异"
        elif ratio > 1.05:
            conclusion = "协同增益"
        elif ratio >= 0.95:
            conclusion = "近似加性"
        else:
            conclusion = "存在冗余"
        ratio_s = f"{ratio:.3f}" if ratio is not None else "n/a"
        lines.append(
            f"${s_key}$ & {d['delta_A1']:+.2f} & {d['delta_A2']:+.2f} "
            f"& {d['delta_A3']:+.2f} & {d['sum_delta']:+.2f} "
            f"& {ratio_s} & {conclusion} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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


def generate_initial_pop_table(init_pop: dict) -> str:
    """初始种群质量对比表（Markdown）。

    说明：仅展示 A0/A2/A3，因为 A1（仅 CR Control）不修改初始种群，
    与 A0 数值完全一致，省略以避免冗余。
    """
    from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS
    lines = [
        "| 场景 | 配置 | 初始种群 Mean ± Std | 初始种群 Median |",
        "|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in ["A0", "A2", "A3"]:
            data = init_pop.get(f"{s_key}_{c_key}", {})
            if not data:
                continue
            if not np.isfinite(data["mean"]):
                ms = "N/A (不可行)"
                md = "N/A"
            else:
                ms = f"{data['mean']:.2f} ± {data['std']:.2f}"
                md = f"{data['median']:.2f}"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {ms} "
                f"| {md} |"
            )
    return "\n".join(lines)


def generate_latex_initial_pop_table(init_pop: dict) -> str:
    """初始种群质量对比表（LaTeX）。

    说明：仅展示 A0/A2/A3（A1 不影响初始种群，省略）。
    """
    from .constants import SCENARIOS, CONFIG_LABELS, SCENARIO_LABELS
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{初始种群质量对比（PopInit 效果验证；A1+CR Control 不影响初始种群，故省略）}",
        r"\label{tab:initial_pop}",
        r"\begin{tabular}{llcc}", r"\toprule",
        r"场景 & 配置 & 初始 Mean $\pm$ Std & 初始 Median \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        first_row = True
        for c_key in ["A0", "A2", "A3"]:
            data = init_pop.get(f"{s_key}_{c_key}", {})
            if not data:
                continue
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            if not np.isfinite(data["mean"]):
                ms = r"N/A \textit{(不可行)}"
                md = "N/A"
            else:
                ms = f"{data['mean']:.2f} $\\pm$ {data['std']:.2f}"
                md = f"{data['median']:.2f}"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {ms} & {md} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {ms} & {md} \\\\")
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
        for c_key in ["A1", "A3"]:
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
        for c_key in ["A0", "A3"]:
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


def generate_latex_decoupled_table(all_stats: dict) -> str:
    """解耦参数控制 LaTeX 表。

    展示 LLM 独立控制 CR、F、GMR 的统计证据。
    """
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{解耦参数控制分析：LLM 独立决定 CR、F、GMR 的效果}",
        r"\label{tab:decoupled_params}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"场景 & 配置 & CR-F 相关系数 & CR 均值 & F 均值 & GMR 模式 \\",
        r"\midrule",
    ]
    for s_key in SCENARIOS:
        first_row = True
        for c_key in ["A0", "A3"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            coupling = compute_parameter_coupling(raw)
            if not coupling:
                continue
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            corr = coupling.get("cr_f_correlation", 0)
            if c_key == "A0":
                corr_str = "---"  # A0 无 LLM 控制
            else:
                corr_str = f"{corr:+.4f}"
            cr_str = f"{coupling['cr_mean']:.3f}"
            f_str = f"{coupling['f_mean']:.3f}"
            gmr_counts = coupling.get("gmr_mode_counts", {})
            if gmr_counts:
                gmr_str = ", ".join(f"{m}: {c}" for m, c in sorted(gmr_counts.items()))
            else:
                gmr_str = "formula"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {corr_str} & {cr_str} & {f_str} & {gmr_str} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {corr_str} & {cr_str} & {f_str} & {gmr_str} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_decoupling_table(decoupling: dict) -> str:
    """解耦效应分析表（Markdown）。

    核心论点支撑：在 PopInit 条件相同（均无）的前提下，
    解耦 LLM 独立控制 CR/F/GMR 是否优于耦合公式。
    """
    lines = [
        "## 解耦效应分析 (Decoupling Effect)\n",
        "**核心论点**：将原本耦合的 CR→F→GMR 控制链解耦，",
        "让 LLM 在诊断推理框架下独立决定 CR、F、GMR，效果优于耦合公式。\n",
        "| 指标 | S1 (balanced) | S2 (srp) |",
        "|---|---|---|",
    ]

    def _row(label, key, fmt="{:.2f}", suffix=""):
        vals = []
        for s_key in ["S1", "S2"]:
            d = decoupling.get(s_key, {})
            v = d.get(key)
            if v is None:
                vals.append("---")
            elif isinstance(v, float) and not np.isfinite(v):
                vals.append("N/A")
            else:
                vals.append(fmt.format(v) + suffix)
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(_row("A1 均值 (耦合)", "A1_mean"))
    lines.append(_row("A3 均值 (解耦)", "A3_mean"))
    lines.append(_row("**A3 vs A1 增益**", "delta_A3_A1", fmt="{:+.2f}"))
    lines.append(_row("PopInit 单独贡献", "delta_A2_A0", fmt="{:+.2f}"))
    lines.append(_row("**纯解耦效应**", "pure_decoupling_effect", fmt="{:+.2f}"))

    # 显著性
    sig_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        p = d.get("p_value_A3_vs_A1", -1)
        sig = d.get("significant", False)
        if p < 0:
            sig_vals.append("---")
        else:
            mark = "✓" if sig else "✗"
            sig_vals.append(f"p={p:.4f} {mark}")
    lines.append(f"| 显著性 (Mann-Whitney U) | " + " | ".join(sig_vals) + " |")

    # 收敛速度
    conv_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        a1g = d.get("A1_conv_gen_95", -1)
        a3g = d.get("A3_conv_gen_95", -1)
        sp = d.get("conv_speedup")
        if a1g < 0 or a3g < 0:
            conv_vals.append("---")
        else:
            sp_str = f"快 {sp:.1f}%" if sp and sp > 0 else f"慢 {abs(sp):.1f}%" if sp else "---"
            conv_vals.append(f"A1=Gen{a1g} → A3=Gen{a3g} ({sp_str})")
    lines.append(f"| 收敛速度 (95%改进) | " + " | ".join(conv_vals) + " |")

    # 参数解耦指标
    corr_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        a1c = d.get("A1_cr_f_corr")
        a3c = d.get("A3_cr_f_corr")
        parts = []
        if a1c is not None:
            parts.append(f"A1={a1c:+.3f}")
        if a3c is not None:
            parts.append(f"A3={a3c:+.3f}")
        corr_vals.append(" → ".join(parts) if parts else "---")
    lines.append(f"| CR-F 相关系数 | " + " | ".join(corr_vals) + " |")

    # GMR 模式
    gmr_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        a1m = d.get("A1_gmr_modes", {})
        a3m = d.get("A3_gmr_modes", {})
        a1_str = ", ".join(f"{m}:{c}" for m, c in sorted(a1m.items())) if a1m else "formula"
        a3_str = ", ".join(f"{m}:{c}" for m, c in sorted(a3m.items())) if a3m else "---"
        gmr_vals.append(f"A1=[{a1_str}] → A3=[{a3_str}]")
    lines.append(f"| GMR 模式分布 | " + " | ".join(gmr_vals) + " |")

    return "\n".join(lines)


def generate_latex_decoupling_table(decoupling: dict) -> str:
    """解耦效应分析表（LaTeX）。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{解耦效应分析：$A_3$（解耦 CR/F/GMR）vs $A_1$（仅 LLM 控 CR，F/GMR 耦合）}",
        r"\label{tab:decoupling_effect}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"指标 & S1 (balanced) & S2 (srp) \\", r"\midrule",
    ]

    def _row(label, key, fmt="{:.2f}", bold=False):
        vals = []
        for s_key in ["S1", "S2"]:
            d = decoupling.get(s_key, {})
            v = d.get(key)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                vals.append("---")
            else:
                s = fmt.format(v)
                if bold:
                    s = r"\textbf{" + s + "}"
                vals.append(s)
        return f"{label} & " + " & ".join(vals) + r" \\"

    lines.append(_row(r"$A_1$ 均值 (耦合)", "A1_mean"))
    lines.append(_row(r"$A_3$ 均值 (解耦)", "A3_mean"))
    lines.append(_row(r"$\Delta_{A_3-A_1}$ (总增益)", "delta_A3_A1", fmt="{:+.2f}", bold=True))
    lines.append(_row(r"$\Delta_{A_2-A_0}$ (PopInit)", "delta_A2_A0", fmt="{:+.2f}"))
    lines.append(_row(r"纯解耦效应", "pure_decoupling_effect", fmt="{:+.2f}", bold=True))

    # 显著性
    sig_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        p = d.get("p_value_A3_vs_A1", -1)
        sig = d.get("significant", False)
        if p < 0:
            sig_vals.append("---")
        else:
            mark = r"$^{*}$" if sig else r"n.s."
            sig_vals.append(f"{p:.4f} {mark}")
    lines.append(r"$p$-value (Mann-Whitney U) & " + " & ".join(sig_vals) + r" \\")

    # 收敛速度
    conv_vals = []
    for s_key in ["S1", "S2"]:
        d = decoupling.get(s_key, {})
        a1g = d.get("A1_conv_gen_95", -1)
        a3g = d.get("A3_conv_gen_95", -1)
        if a1g < 0 or a3g < 0:
            conv_vals.append("---")
        else:
            conv_vals.append(f"Gen {a1g} $\\to$ Gen {a3g}")
    lines.append(r"收敛速度 (95\% 改进) & " + " & ".join(conv_vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
