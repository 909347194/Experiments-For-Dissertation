# -*- coding: utf-8 -*-
"""消融实验分析主入口。

用法：python -m analyze
"""
import json
import sys
from pathlib import Path

from .constants import SCENARIOS, CONFIGS
from .stats import (
    compute_stats,
    compute_convergence_gens,
)
from .tables import (
    generate_markdown_table,
    generate_latex_table1,
    generate_latex_table2,
    generate_latex_table3,
    generate_llm_decision_summary,
    generate_convergence_speed_table,
    generate_latex_convergence_speed_table,
    generate_f_stats_table,
    generate_gmr_stats_table,
    generate_parameter_coupling_table,
)
from .plots import (
    plot_convergence_comparison,
    plot_cr_comparison,
    plot_boxplot,
    plot_time_breakdown,
    plot_f_comparison,
    plot_gmr_mode_distribution,
    plot_parameter_control_overview,
    plot_preset_distribution,
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

PROJECT_ROOT = SCRIPT_DIR.parents[1]

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from core.results import load_results


def main():
    all_stats = {}

    for s_key, s_dir in SCENARIOS.items():
        scenario_dir = SCRIPT_DIR / s_dir
        for c_key, c_dir in CONFIGS.items():
            results = load_results(scenario_dir, c_dir)
            all_stats[f"{s_key}_{c_key}"] = compute_stats(results)

    # ═══════════════════════════════════════════════════════════
    # 1. 主结果表
    # ═══════════════════════════════════════════════════════════
    md_table = generate_markdown_table(all_stats)
    print("\n## 消融实验结果\n")
    print(md_table)

    llm_summary = generate_llm_decision_summary(all_stats)
    print(f"\n{llm_summary}")

    # ═══════════════════════════════════════════════════════════
    # 2. 收敛速度量化
    # ═══════════════════════════════════════════════════════════
    conv_gens = compute_convergence_gens(all_stats)
    conv_md = generate_convergence_speed_table(conv_gens)
    print("\n## 收敛速度对比\n")
    print(conv_md)

    # ═══════════════════════════════════════════════════════════
    # 3. 解耦参数控制分析（CR / F / GMR 独立控制）
    # ═══════════════════════════════════════════════════════════
    f_stats_md = generate_f_stats_table(all_stats)
    print(f"\n{f_stats_md}")

    gmr_stats_md = generate_gmr_stats_table(all_stats)
    print(f"\n{gmr_stats_md}")

    coupling_md = generate_parameter_coupling_table(all_stats)
    print(f"\n{coupling_md}")

    # ═══════════════════════════════════════════════════════════
    # 4. 保存 Markdown
    # ═══════════════════════════════════════════════════════════
    md_out = FIGURES_DIR / "summary_table.md"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("## 消融实验结果\n\n")
        f.write(md_table)
        f.write("\n\n## 收敛速度对比\n\n")
        f.write(conv_md)
        f.write(f"\n\n{llm_summary}\n")
        f.write("\n\n## F (Scale Factor) 统计\n\n")
        f.write(f_stats_md)
        f.write("\n\n## GMR (Extinction) 模式分析\n\n")
        f.write(gmr_stats_md)
        f.write("\n\n## 参数解耦分析\n\n")
        f.write(coupling_md)
        f.write("\n")

    # ═══════════════════════════════════════════════════════════
    # 5. CSV
    # ═══════════════════════════════════════════════════════════
    csv_out = FIGURES_DIR / "summary_table.csv"
    with open(csv_out, "w", encoding="utf-8") as f:
        f.write("scenario,config,best,mean,std,median,"
                "total_time_mean,dmde_time_mean,llm_time_mean,"
                "llm_cr_time_mean,llm_calls_mean\n")
        for s_key in SCENARIOS:
            for c_key in CONFIGS:
                stats = all_stats.get(f"{s_key}_{c_key}", {})
                if not stats:
                    continue
                f.write(f"{s_key},{c_key},{stats['best']},{stats['mean']},{stats['std']},"
                        f"{stats['median']},{stats['total_time_mean']},"
                        f"{stats['dmde_time_mean']},{stats['llm_time_mean']},"
                        f"{stats['llm_cr_time_mean']},"
                        f"{stats['llm_calls_mean']}\n")

    # ═══════════════════════════════════════════════════════════
    # 6. LaTeX
    # ═══════════════════════════════════════════════════════════
    tex_content = (
        r"% 消融实验结果表格 — 自动生成" "\n"
        r"% 编译: xelatex ablation_tables.tex   (pdflatex 无法渲染中文)" "\n"
        r"\documentclass{article}" "\n"
        r"\usepackage[UTF8]{ctex}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{amsmath}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\begin{document}" "\n\n"
        r"\section*{消融实验结果}" "\n\n"
        + generate_latex_table1(all_stats) + "\n\n"
        + generate_latex_table3(all_stats) + "\n\n"
        + generate_latex_table2(all_stats) + "\n\n"
        r"\section*{收敛速度对比}" "\n\n"
        + generate_latex_convergence_speed_table(conv_gens) + "\n\n"
        r"\end{document}" "\n"
    )
    tex_out = FIGURES_DIR / "ablation_tables.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # ═══════════════════════════════════════════════════════════
    # 7. 可视化
    # ═══════════════════════════════════════════════════════════
    if HAS_MPL:
        print("\n生成可视化图表...")
        for s_key in SCENARIOS:
            plot_convergence_comparison(all_stats, s_key, FIGURES_DIR)
            plot_cr_comparison(all_stats, s_key, FIGURES_DIR)
            plot_boxplot(all_stats, s_key, FIGURES_DIR)
            plot_f_comparison(all_stats, s_key, FIGURES_DIR)
            plot_gmr_mode_distribution(all_stats, s_key, FIGURES_DIR)
            plot_parameter_control_overview(all_stats, s_key, FIGURES_DIR)
            plot_preset_distribution(all_stats, s_key, FIGURES_DIR)
        plot_time_breakdown(all_stats, FIGURES_DIR)
    else:
        print("\n(matplotlib 不可用，跳过图表)")

    # ═══════════════════════════════════════════════════════════
    # 8. LLM 决策日志
    # ═══════════════════════════════════════════════════════════
    for s_key in SCENARIOS:
        for c_key in ["A1"]:
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