# -*- coding: utf-8 -*-
"""消融实验分析主入口。

用法：python -m analyze
"""
import json
import sys
from pathlib import Path

from .constants import SCENARIOS, CONFIGS
from .stats import compute_stats
from .tables import (
    generate_markdown_table,
    generate_latex_table1,
    generate_latex_table2,
    generate_latex_table3,
    generate_llm_decision_summary,
)
from .plots import (
    plot_convergence_comparison,
    plot_cr_comparison,
    plot_boxplot,
    plot_time_breakdown,
    HAS_MPL,
)

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCRIPT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from core.results import load_results


def main():
    all_stats = {}

    for s_key, s_dir in SCENARIOS.items():
        scenario_dir = SCRIPT_DIR / s_dir
        for c_key, c_dir in CONFIGS.items():
            results = load_results(scenario_dir, c_dir)
            all_stats[f"{s_key}_{c_key}"] = compute_stats(results)

    # ── Markdown ──
    md_table = generate_markdown_table(all_stats)
    print("\n## 消融实验结果\n")
    print(md_table)

    llm_summary = generate_llm_decision_summary(all_stats)
    print(f"\n{llm_summary}")

    md_out = FIGURES_DIR / "summary_table.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("## 消融实验结果\n\n")
        f.write(md_table)
        f.write(f"\n\n{llm_summary}\n")

    # ── CSV ──
    csv_out = FIGURES_DIR / "summary_table.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("scenario,config,best,mean,std,median,total_time,dmde_time,llm_time,"
                "llm_init_time,llm_cr_time,llm_calls\n")
        for s_key in SCENARIOS:
            for c_key in CONFIGS:
                stats = all_stats.get(f"{s_key}_{c_key}", {})
                if not stats:
                    continue
                f.write(f"{s_key},{c_key},{stats['best']},{stats['mean']},{stats['std']},"
                        f"{stats['median']},{stats['total_time_mean']},"
                        f"{stats['dmde_time_mean']},{stats['llm_time_mean']},"
                        f"{stats['llm_init_time_mean']},{stats['llm_cr_time_mean']},"
                        f"{stats['llm_calls_mean']}\n")

    # ── LaTeX ──
    tex_content = (
        r"% 消融实验结果表格 — 自动生成" "\n"
        r"% 编译: pdflatex ablation_tables.tex" "\n"
        r"\documentclass{article}" "\n"
        r"\usepackage[utf8]{inputenc}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\begin{document}" "\n\n"
        r"\section*{消融实验结果}" "\n\n"
        + generate_latex_table1(all_stats) + "\n\n"
        + generate_latex_table3(all_stats) + "\n\n"
        + generate_latex_table2(all_stats) + "\n\n"
        r"\end{document}" "\n"
    )
    tex_out = FIGURES_DIR / "ablation_tables.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # ── 可视化 ──
    if HAS_MPL:
        print("\n生成可视化图表...")
        for s_key in SCENARIOS:
            plot_convergence_comparison(all_stats, s_key, FIGURES_DIR)
            plot_cr_comparison(all_stats, s_key, FIGURES_DIR)
            plot_boxplot(all_stats, s_key, FIGURES_DIR)
        plot_time_breakdown(all_stats, FIGURES_DIR)
    else:
        print("\n(matplotlib 不可用，跳过图表)")

    # ── LLM 决策日志 ──
    for s_key in SCENARIOS:
        for c_key in ["A1", "A2", "A3"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            for r in stats.get("_raw", []):
                decisions = r.get("llm_decisions", [])
                if decisions:
                    seed = r.get("seed", 0)
                    log_out = FIGURES_DIR / f"llm_decisions_{s_key}_{c_key}_seed{seed}.json"
                    with open(log_out, "w", encoding="utf-8") as f:
                        json.dump(decisions, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n已保存:")
    print(f"  Markdown: {md_out}")
    print(f"  CSV:      {csv_out}")
    print(f"  LaTeX:    {tex_out}")


if __name__ == "__main__":
    main()