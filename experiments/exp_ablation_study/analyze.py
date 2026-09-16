# -*- coding: utf-8 -*-
"""analyze.py — 消融实验结果分析

读取 8 组实验结果，生成 Markdown + LaTeX 表格。

用法：python analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SCENARIOS = {
    "S1": "S1_balanced_N10_M10",
    "S2": "S2_srp_N10_M20",
}
CONFIGS = {
    "A0": "A0_dmde",
    "A1": "A1_cr_control",
    "A2": "A2_pop_init",
    "A3": "A3_full",
}
CONFIG_LABELS = {
    "A0": "Vanilla DMDE",
    "A1": "+CR Control",
    "A2": "+PopInit",
    "A3": "Full LLM-DMDE",
}
SCENARIO_LABELS = {
    "S1": r"balanced ($N{=}M{=}10$)",
    "S2": r"srp ($N{=}10, M{=}20$)",
}


def load_results(scenario_dir: str, config_dir: str) -> list[dict]:
    path = SCRIPT_DIR / scenario_dir / config_dir / "results" / "ablation_results.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def compute_stats(results: list[dict]) -> dict:
    if not results:
        return {}
    fitness = np.array([r["best_fitness"] for r in results])
    times = np.array([r["total_time"] for r in results])
    llm_times = np.array([r.get("llm_time", 0) for r in results])
    llm_calls = np.array([r.get("llm_call_count", 0) for r in results])
    return {
        "n_runs": len(results),
        "best": round(float(fitness.min()), 2),
        "mean": round(float(fitness.mean()), 2),
        "std": round(float(fitness.std()), 2),
        "median": round(float(np.median(fitness)), 2),
        "total_time_mean": round(float(times.mean()), 2),
        "llm_time_mean": round(float(llm_times.mean()), 2),
        "llm_calls_mean": round(float(llm_calls.mean()), 1),
        "fitness_array": fitness,
    }


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> float:
    """双侧 Wilcoxon 秩和检验，返回 p-value。"""
    if not HAS_SCIPY:
        return -1.0
    if len(a) < 5 or len(b) < 5:
        return -1.0
    try:
        _, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        return round(float(p), 4)
    except Exception:
        return -1.0


def p_mark(p: float) -> str:
    if p < 0:
        return "---"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ── LaTeX 输出 ───────────────────────────────────────────────

def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def generate_latex_table1(all_stats: dict) -> str:
    """表 1：消融实验主结果（解质量）。"""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{消融实验结果：解质量对比}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"场景 & 配置 & Best & Mean $\pm$ Std & Median \\",
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
            best = f"{stats['best']:.2f}"
            mean_std = f"{stats['mean']:.2f} $\pm$ {stats['std']:.2f}"
            median = f"{stats['median']:.2f}"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {best} & {mean_std} & {median} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {best} & {mean_std} & {median} \\\\")
        lines.append(r"\midrule")
    lines.pop()  # remove last midrule
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table2(all_stats: dict) -> str:
    """表 2：统计显著性（Wilcoxon p-value）。"""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{统计显著性检验（Wilcoxon 秩和检验，A3 vs 基线）}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"场景 & A3 vs A0 & A3 vs A1 & A3 vs A2 \\",
        r"\midrule",
    ]
    for s_key in SCENARIOS:
        a3 = all_stats.get(f"{s_key}_A3", {}).get("fitness_array", np.array([]))
        row = [SCENARIO_LABELS[s_key]]
        for c_key in ["A0", "A1", "A2"]:
            other = all_stats.get(f"{s_key}_{c_key}", {}).get("fitness_array", np.array([]))
            p = wilcoxon_test(a3, other)
            mark = p_mark(p)
            if p < 0:
                row.append("---")
            else:
                row.append(f"{p:.4f} {mark}")
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\footnotesize{*$p < 0.05$，**$p < 0.01$，n.s. 不显著}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_latex_table3(all_stats: dict) -> str:
    """表 3：计算时间对比。"""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{计算时间对比}",
        r"\label{tab:time_breakdown}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"场景 & 配置 & 总耗时(s) & LLM耗时(s) & LLM调用次数 \\",
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
            llm = f"{stats['llm_time_mean']:.1f}" if stats['llm_time_mean'] > 0 else "---"
            calls = f"{stats['llm_calls_mean']:.0f}" if stats['llm_calls_mean'] > 0 else "---"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {total} & {llm} & {calls} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {total} & {llm} & {calls} \\\\")
        lines.append(r"\midrule")
    lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Markdown 输出 ────────────────────────────────────────────

def generate_markdown_table(all_stats: dict) -> str:
    lines = [
        "| 场景 | 配置 | Best | Mean ± Std | Median | 总耗时(s) | LLM耗时(s) | LLM调用次数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s_key in SCENARIOS:
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                lines.append(f"| {s_key} | {c_key} | --- | --- | --- | --- | --- | --- |")
                continue
            llm_t = f"{stats['llm_time_mean']:.1f}" if stats['llm_time_mean'] > 0 else "---"
            llm_c = f"{stats['llm_calls_mean']:.0f}" if stats['llm_calls_mean'] > 0 else "---"
            lines.append(
                f"| {s_key} | {CONFIG_LABELS[c_key]} "
                f"| {stats['best']:.2f} "
                f"| {stats['mean']:.2f} ± {stats['std']:.2f} "
                f"| {stats['median']:.2f} "
                f"| {stats['total_time_mean']:.1f} "
                f"| {llm_t} | {llm_c} |"
            )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────

def main():
    all_stats = {}

    for s_key, s_dir in SCENARIOS.items():
        for c_key, c_dir in CONFIGS.items():
            results = load_results(s_dir, c_dir)
            stats = compute_stats(results)
            all_stats[f"{s_key}_{c_key}"] = stats

    # Markdown
    md_table = generate_markdown_table(all_stats)
    print("\n## 消融实验结果\n")
    print(md_table)

    md_out = FIGURES_DIR / "summary_table.md"
    with open(md_out, "w") as f:
        f.write("## 消融实验结果\n\n")
        f.write(md_table)
        f.write("\n")

    # CSV
    csv_out = FIGURES_DIR / "summary_table.csv"
    with open(csv_out, "w") as f:
        f.write("scenario,config,best,mean,std,total_time,llm_time,llm_calls\n")
        for s_key in SCENARIOS:
            for c_key in CONFIGS:
                stats = all_stats.get(f"{s_key}_{c_key}", {})
                if not stats:
                    continue
                f.write(f"{s_key},{c_key},{stats['best']},{stats['mean']},"
                        f"{stats['std']},{stats['total_time_mean']},"
                        f"{stats['llm_time_mean']},{stats['llm_calls_mean']}\n")

    # LaTeX
    tex_content = r"""% 消融实验结果表格 — 自动生成
% 编译: pdflatex ablation_tables.tex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}

\begin{document}

\section*{消融实验结果}

""" + generate_latex_table1(all_stats) + "\n\n" + \
    generate_latex_table3(all_stats) + "\n\n" + \
    generate_latex_table2(all_stats) + "\n\n" + \
    r"\end{document}" + "\n"

    tex_out = FIGURES_DIR / "ablation_tables.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(tex_content)

    print(f"\n已保存:")
    print(f"  Markdown: {md_out}")
    print(f"  CSV:      {csv_out}")
    print(f"  LaTeX:    {tex_out}")
    if HAS_SCIPY:
        print(f"  (scipy 可用，Wilcoxon 检验已计算)")
    else:
        print(f"  (scipy 不可用，Wilcoxon 检验跳过)")


if __name__ == "__main__":
    main()