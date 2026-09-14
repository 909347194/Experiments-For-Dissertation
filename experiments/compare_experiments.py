# -*- coding: utf-8 -*-
"""compare_experiments.py — DMDE 基线 vs LLM-DMDE 实验对比

读取两个实验的结果 JSON，生成：
1. results/comparison.md              — 对比表格
2. results/figures/comparison_boxplot.png  — fitness 箱线图对比
3. results/figures/comparison_convergence.png — 收敛曲线对比

用法：
    cd experiments
    python compare_experiments.py

    # 或指定自定义路径
    python compare_experiments.py --baseline exp_dmde/exp_dmde_01/results/exp_dmde_01_data.json \
                                   --llm exp_llm_enhanced_dmde/exp_llm_dmde_01/results/exp_llm_dmde_01_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 路径解析：兼容从项目根目录或 experiments/ 目录运行
_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "exp_dmde").is_dir():
    # 从 experiments/ 目录运行
    _BASE = _SCRIPT_DIR
elif (_SCRIPT_DIR / "experiments" / "exp_dmde").is_dir():
    # 从项目根目录运行
    _BASE = _SCRIPT_DIR / "experiments"
else:
    _BASE = _SCRIPT_DIR

DEFAULT_BASELINE = _BASE / "exp_dmde" / "exp_dmde_01" / "results" / "exp_dmde_01_data.json"
DEFAULT_LLM = _BASE / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "results" / "exp_llm_dmde_01_data.json"
OUTPUT_DIR = _BASE / "comparison_results"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_run_stats(data: dict) -> dict:
    """从实验 JSON 中提取关键统计信息。"""
    scenario = data["scenarios"][0]
    meta = data.get("meta", {})
    metrics = scenario["metrics"]
    runs = scenario["runs"]

    fitness_values = [r["best_fitness"] for r in runs]
    feasible_count = sum(1 for r in runs if r["extra"].get("is_feasible", False))
    times = [r.get("elapsed_seconds", 0) for r in runs]
    cost_histories = [r.get("cost_history", []) for r in runs]

    return {
        "name": meta.get("description", scenario["name"]),
        "n_runs": len(runs),
        "best_fitness": metrics["best_fitness"],
        "mean_fitness": metrics["mean_fitness"],
        "std_fitness": metrics["std_fitness"],
        "worst_fitness": metrics["worst_fitness"],
        "median_fitness": metrics.get("median_fitness", np.median(fitness_values)),
        "fitness_values": fitness_values,
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / max(len(runs), 1),
        "mean_time": np.mean(times),
        "times": times,
        "cost_histories": cost_histories,
        "solver_params": meta.get("solver_params", {}),
        "llm_config": meta.get("llm_config", {}),
    }


def gen_comparison_table(baseline: dict, llm: dict) -> str:
    """生成 Markdown 对比表。"""

    def pct_change(base, improved):
        if base == 0:
            return "—"
        change = (improved - base) / abs(base) * 100
        if change > 0:
            return f"+{change:.2f}%"
        elif change < 0:
            return f"{change:.2f}%"
        else:
            return "0.00%"

    # Fitness: lower is better, so improvement = baseline - llm
    def fitness_change(base, improved):
        if base == 0:
            return "—"
        change = (base - improved) / abs(base) * 100
        if change > 0:
            return f"+{change:.2f}%"
        elif change < 0:
            return f"{change:.2f}%"
        else:
            return "0.00%"

    lines = [
        "# 实验对比：DMDE 基线 vs LLM-DMDE\n",
        "| 指标 | DMDE 基线 | LLM-DMDE | 变化 |",
        "|------|-----------|----------|------|",
        f"| Best fitness | {baseline['best_fitness']:.2f} | {llm['best_fitness']:.2f} "
        f"| {fitness_change(baseline['best_fitness'], llm['best_fitness'])} |",
        f"| Mean fitness | {baseline['mean_fitness']:.2f} ± {baseline['std_fitness']:.2f} "
        f"| {llm['mean_fitness']:.2f} ± {llm['std_fitness']:.2f} "
        f"| {fitness_change(baseline['mean_fitness'], llm['mean_fitness'])} |",
        f"| 可行解率 | {baseline['feasible_rate']*100:.0f}% ({baseline['feasible_count']}/{baseline['n_runs']}) "
        f"| {llm['feasible_rate']*100:.0f}% ({llm['feasible_count']}/{llm['n_runs']}) | — |",
        f"| 平均耗时 | {baseline['mean_time']:.1f}s | {llm['mean_time']:.1f}s | — |",
    ]

    # LLM-specific stats
    if llm.get("llm_config"):
        lines.append(f"| LLM 模型 | — | {llm['llm_config'].get('model', '—')} | — |")

    lines.append("")
    return "\n".join(lines)


def _setup_sci_font():
    """配置 SCI 论文常用字体（Times New Roman + Arial）。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        # 字体
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        # 字号
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # 坐标轴
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        # 刻度朝内
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # 图例
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        # 保存
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_comparison_boxplot(baseline: dict, llm: dict, output_path: Path):
    """生成 fitness 箱线图对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_sci_font()

    fig, ax = plt.subplots(figsize=(8, 5))

    data_to_plot = [baseline["fitness_values"], llm["fitness_values"]]
    labels = ["DMDE 基线", "LLM-DMDE"]

    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                    widths=0.5, showmeans=True, meanprops=dict(marker="D", markerfacecolor="red"))

    colors = ["#4ECDC4", "#FF6B6B"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title("Fitness Distribution: DMDE vs LLM-DMDE", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add individual data points
    for i, (d, c) in enumerate(zip(data_to_plot, colors), 1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color=c, edgecolors="black", linewidths=0.5, s=40, zorder=5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 箱线图: {output_path}")


def plot_comparison_convergence(baseline: dict, llm: dict, output_path: Path):
    """生成收敛曲线对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_sci_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#4ECDC4", "#FF6B6B"]
    labels = ["DMDE 基线", "LLM-DMDE"]

    for histories, color, label in zip(
        [baseline["cost_histories"], llm["cost_histories"]], colors, labels
    ):
        if not histories:
            continue

        # Find the best run (shortest final fitness)
        best_idx = min(range(len(histories)), key=lambda i: histories[i][-1] if histories[i] else float("inf"))
        best_history = histories[best_idx]

        # Plot individual runs (lighter)
        for i, h in enumerate(histories):
            if h:
                gens = list(range(len(h)))
                ax.plot(gens, h, color=color, alpha=0.2, linewidth=0.8)

        # Plot best run (bold)
        gens = list(range(len(best_history)))
        ax.plot(gens, best_history, color=color, linewidth=2.5, label=f"{label} (best run)")

    # Mark LLM decision points
    llm_gens = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if llm["cost_histories"]:
        best_history = llm["cost_histories"][min(range(len(llm["cost_histories"])),
                                                   key=lambda i: llm["cost_histories"][i][-1]
                                                   if llm["cost_histories"][i] else float("inf"))]
        for g in llm_gens:
            if g < len(best_history):
                ax.axvline(x=g, color="#FF6B6B", alpha=0.15, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title("Convergence Curves: DMDE vs LLM-DMDE", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 收敛曲线: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="对比 DMDE 基线和 LLM-DMDE 实验结果")
    parser.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE),
                        help="DMDE 基线结果 JSON 路径")
    parser.add_argument("--llm", type=str, default=str(DEFAULT_LLM),
                        help="LLM-DMDE 结果 JSON 路径")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    llm_path = Path(args.llm)

    # Check files exist
    if not baseline_path.exists():
        print(f"❌ DMDE 基线结果不存在: {baseline_path}")
        print("请先运行: cd exp_dmde/exp_dmde_01 && python run.py")
        sys.exit(1)
    if not llm_path.exists():
        print(f"❌ LLM-DMDE 结果不存在: {llm_path}")
        print("请先运行: cd exp_llm_enhanced_dmde/exp_llm_dmde_01 && python run.py")
        sys.exit(1)

    print("=" * 60)
    print("实验对比：DMDE 基线 vs LLM-DMDE")
    print("=" * 60)

    # Load data
    baseline_data = load_data(baseline_path)
    llm_data = load_data(llm_path)

    baseline_stats = extract_run_stats(baseline_data)
    llm_stats = extract_run_stats(llm_data)

    print(f"\n  DMDE 基线: {baseline_stats['n_runs']} runs, "
          f"best={baseline_stats['best_fitness']:.2f}")
    print(f"  LLM-DMDE:  {llm_stats['n_runs']} runs, "
          f"best={llm_stats['best_fitness']:.2f}")

    # 1. Generate comparison table
    print("\n[1] 生成对比表...")
    table = gen_comparison_table(baseline_stats, llm_stats)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_md = OUTPUT_DIR / "comparison.md"
    comparison_md.write_text(table, encoding="utf-8")
    print(f"  ✅ 对比表: {comparison_md}")

    # 2. Generate plots
    print("\n[2] 生成对比图...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_comparison_boxplot(
        baseline_stats, llm_stats,
        FIGURES_DIR / "comparison_boxplot.png",
    )
    plot_comparison_convergence(
        baseline_stats, llm_stats,
        FIGURES_DIR / "comparison_convergence.png",
    )

    print("\n✅ 对比完成。")


if __name__ == "__main__":
    main()
