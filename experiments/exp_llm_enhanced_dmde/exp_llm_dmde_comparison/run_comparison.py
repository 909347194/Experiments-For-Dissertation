# -*- coding: utf-8 -*-
"""run_comparison.py — DMDE 基线 vs LLM-DMDE 对比实验（增强版）

同时运行两个实验，生成对比报告，结果保存在本目录的 results/ 中。

用法：
    # N=M 场景（默认），30 次运行
    python run_comparison.py

    # 指定场景
    python run_comparison.py --scenario nm    # N=M
    python run_comparison.py --scenario ngt   # N>M
    python run_comparison.py --scenario nlt   # N<M
    python run_comparison.py --scenarios all  # 全部场景

    # 多规模
    python run_comparison.py --sizes small medium large

    # 生成 LaTeX 表格
    python run_comparison.py --scenarios all --sizes small medium large --format latex

    # 消融实验
    python run_comparison.py --ablation

    # 自定义 solver 参数（大规模需要更大种群）
    python run_comparison.py --sizes large --solver-params '{"pop_size":150,"max_generations":2000}'

    # 只跑一个
    python run_comparison.py --only baseline
    python run_comparison.py --only llm

    # 跳过对比生成
    python run_comparison.py --no-compare
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent.parent  # experiments/
PROJECT_ROOT = EXPERIMENTS_DIR.parent

# 场景映射：scenario → (dmde_dir, llm_dmde_dir)
SCENARIO_MAP = {
    "nm":  ("exp_dmde/exp_dmde_01", "exp_llm_enhanced_dmde/exp_llm_dmde_01"),
    "ngt": ("exp_dmde/exp_dmde_02", "exp_llm_enhanced_dmde/exp_llm_dmde_02"),
    "nlt": ("exp_dmde/exp_dmde_03", "exp_llm_enhanced_dmde/exp_llm_dmde_03"),
}

SCENARIO_LABELS = {
    "nm": "N=M Balanced Assignment",
    "ngt": "N>M Many-to-One",
    "nlt": "N<M Group Patrol",
}

SCALES = ["small", "medium", "large"]

# 配色
COLORS = {"DMDE": "#4ECDC4", "LLM-DMDE": "#FF6B6B"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(run_py: Path, n_runs: int, label: str, *,
                   scale: str | None = None,
                   solver_params: str | None = None,
                   ablation: bool = False) -> bool:
    """运行单个实验。"""
    print(f"\n{'='*60}")
    print(f"▶ {label}")
    print(f"  Script: {run_py}")
    print(f"  Runs: {n_runs}")
    if scale:
        print(f"  Scale: {scale}")
    if solver_params:
        print(f"  Solver params: {solver_params}")
    if ablation:
        print(f"  Ablation: enabled")
    print(f"{'='*60}\n")

    env = {**os.environ, "EXP_N_RUNS": str(n_runs)}
    if scale:
        env["EXP_SCALE"] = scale
    if solver_params:
        env["EXP_SOLVER_PARAMS"] = solver_params
    if ablation:
        env["EXP_MODULES"] = json.dumps({
            "population_init": {"enabled": False},
            "search_controller": {"enabled": False},
        })

    t0 = time.time()
    result = subprocess.run([sys.executable, str(run_py)], env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ {label} failed (exit={result.returncode})")
        return False
    print(f"\n✅ {label} done ({elapsed:.1f}s)")
    return True


def find_result_json(experiment_dir: Path) -> Path | None:
    """自动查找实验结果 JSON。"""
    results_dir = experiment_dir / "results"
    if not results_dir.exists():
        return None
    jsons = sorted(results_dir.glob("*_data.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsons[0] if jsons else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_stats(data: dict, label: str) -> dict:
    """从实验 JSON 提取统计。"""
    sc = data["scenarios"][0]
    meta = data.get("meta", {})
    metrics = sc["metrics"]
    runs = sc["runs"]

    fitness_values = [r["best_fitness"] for r in runs]
    feasible_count = sum(1 for r in runs if r["extra"].get("is_feasible", False))
    times = [r.get("elapsed_seconds", 0) for r in runs]
    cost_histories = [r.get("cost_history", []) for r in runs]

    return {
        "label": label,
        "n_runs": len(runs),
        "best_fitness": metrics["best_fitness"],
        "mean_fitness": metrics["mean_fitness"],
        "std_fitness": metrics["std_fitness"],
        "worst_fitness": metrics["worst_fitness"],
        "median_fitness": float(np.median(fitness_values)),
        "fitness_values": fitness_values,
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / max(len(runs), 1),
        "mean_time": float(np.mean(times)),
        "cost_histories": cost_histories,
        "llm_config": meta.get("llm_config", {}),
    }


def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, bool]:
    """Wilcoxon 符号秩检验。"""
    try:
        from scipy.stats import wilcoxon
        diffs = [x - y for x, y in zip(a, b) if x != y]
        if len(diffs) < 3:
            return 1.0, False
        _, p = wilcoxon(diffs, alternative="two-sided")
        return float(p), p < 0.05
    except ImportError:
        return 1.0, False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图表绘制（统一配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _setup_font():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Find available serif fonts with CJK fallback
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif_fonts = ["Times New Roman", "DejaVu Serif", "Noto Serif CJK SC",
                   "SimSun", "AR PL UMing CN"]
    chosen_serif = [f for f in serif_fonts if f in available]
    if not chosen_serif:
        chosen_serif = ["DejaVu Serif"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": chosen_serif,
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 14,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.unicode_minus": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def _save_fig(fig, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  ✅ Figure: {output}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图表：基础对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_boxplot(dmde: dict, llm: dict, output: Path, scenario: str):
    """Box plot + violin + scatter overlay for fitness distribution."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, ax = plt.subplots(figsize=(8, 5))
    data_list = [dmde["fitness_values"], llm["fitness_values"]]
    colors = [COLORS["DMDE"], COLORS["LLM-DMDE"]]
    labels = ["DMDE", "LLM-DMDE"]

    # Violin plot (background) — skip if data is constant (KDE fails)
    try:
        parts = ax.violinplot(data_list, positions=[1, 2], showmeans=False,
                              showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.25)
    except Exception:
        pass  # all-constant data → violin impossible, box+scatter still works

    # Box plot (middle layer, whis=[5, 95] for extended whiskers)
    bp = ax.boxplot(data_list, positions=[1, 2], tick_labels=labels,
                    patch_artist=True, widths=0.3, whis=[5, 95],
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Scatter (foreground) with jitter
    for i, (d, c) in enumerate(zip(data_list, colors), 1):
        x = np.random.normal(i, 0.06, size=len(d))
        ax.scatter(x, d, alpha=0.7, color=c, edgecolors="black",
                   linewidths=0.5, s=35, zorder=5)

    # Mean +/- std annotation
    for i, (d, c) in enumerate(zip(data_list, colors), 1):
        mu, sigma = np.mean(d), np.std(d)
        ax.annotate(f"{mu:.2f}\n$\\pm${sigma:.2f}",
                    xy=(i, mu), xytext=(i + 0.35, mu),
                    fontsize=9, color=c, fontweight="bold",
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.8))

    # Zoom y-axis to data range with margin
    all_vals = dmde["fitness_values"] + llm["fitness_values"]
    vmin, vmax = min(all_vals), max(all_vals)
    margin = (vmax - vmin) * 0.15 if vmax > vmin else abs(vmax) * 0.05
    ax.set_ylim(vmin - margin, vmax + margin)

    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Fitness Distribution \u2014 {scenario.upper()}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, output)


def plot_convergence(dmde: dict, llm: dict, output: Path, scenario: str):
    """Convergence curves: mean fitness per generation with std band + best-so-far reference."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    _setup_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    line_styles = [("-", 2.0), ("--", 2.0)]  # solid for DMDE, dashed for LLM-DMDE
    max_gen = 200  # truncate display

    for (stats, color, (ls, lw)) in zip(
            [dmde, llm], [COLORS["DMDE"], COLORS["LLM-DMDE"]], line_styles):
        histories = stats["cost_histories"]
        if not histories:
            continue

        # Align all histories to the same length, truncate to max_gen
        min_len = min(len(h) for h in histories if h)
        n_gen = min(min_len, max_gen)

        # Compute per-generation mean and std
        stacked = np.array([h[:n_gen] for h in histories if len(h) >= n_gen])
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        gens = np.arange(n_gen)

        # Mean fitness with std band
        ax.plot(gens, mean_curve, color=color, linestyle=ls, linewidth=lw,
                label=f"{stats['label']} (mean)")
        ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                        color=color, alpha=0.15)

        # Best-so-far reference (thin dotted)
        best_curve = np.minimum.accumulate(mean_curve)
        ax.plot(gens, best_curve, color=color, linestyle=":", linewidth=1.0,
                alpha=0.6)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"Convergence \u2014 {scenario.upper()}")
    ax.set_xlim(0, max_gen)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Inset axes: zoom into first 50 generations
    axins = inset_axes(ax, width="40%", height="35%", loc="center right")
    inset_gen = 50
    for (stats, color, (ls, lw)) in zip(
            [dmde, llm], [COLORS["DMDE"], COLORS["LLM-DMDE"]], line_styles):
        histories = stats["cost_histories"]
        if not histories:
            continue
        min_len = min(len(h) for h in histories if h)
        n_gen = min(min_len, inset_gen)
        stacked = np.array([h[:n_gen] for h in histories if len(h) >= n_gen])
        mean_curve = np.mean(stacked, axis=0)
        std_curve = np.std(stacked, axis=0)
        gens = np.arange(n_gen)
        axins.plot(gens, mean_curve, color=color, linestyle=ls, linewidth=lw)
        axins.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                           color=color, alpha=0.15)
    axins.set_xlim(0, inset_gen)
    axins.set_xlabel("Gen", fontsize=9)
    axins.set_ylabel("Fitness", fontsize=9)
    axins.tick_params(labelsize=8)
    axins.grid(alpha=0.3)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

    plt.tight_layout()
    _save_fig(fig, output)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 图表：新增对比图表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_improvement_bar(dmde: dict, llm: dict, output: Path, scenario: str):
    """改进幅度柱状图（best/mean fitness 改进百分比）。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
    mean_imp = (dmde["mean_fitness"] - llm["mean_fitness"]) / abs(dmde["mean_fitness"]) * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Best Fitness", "Mean Fitness"], [best_imp, mean_imp],
                  color=[COLORS["LLM-DMDE"]] * 2, edgecolor="black", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, [best_imp, mean_imp]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:+.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Improvement (%)")
    ax.set_title(f"LLM-DMDE Improvement — {scenario.upper()}")
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, output)


def plot_feasibility_bar(dmde: dict, llm: dict, output: Path, scenario: str):
    """可行解率对比图。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, ax = plt.subplots(figsize=(6, 5))
    rates = [dmde["feasible_rate"] * 100, llm["feasible_rate"] * 100]
    bars = ax.bar(["DMDE", "LLM-DMDE"], rates,
                  color=[COLORS["DMDE"], COLORS["LLM-DMDE"]],
                  edgecolor="black", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Feasible Rate (%)")
    ax.set_title(f"Feasibility — {scenario.upper()}")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, output)


def plot_time_bar(dmde: dict, llm: dict, output: Path, scenario: str):
    """耗时对比图。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, ax = plt.subplots(figsize=(6, 5))
    times = [dmde["mean_time"], llm["mean_time"]]
    bars = ax.bar(["DMDE", "LLM-DMDE"], times,
                  color=[COLORS["DMDE"], COLORS["LLM-DMDE"]],
                  edgecolor="black", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean Time (s)")
    ax.set_title(f"Computation Time — {scenario.upper()}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, output)


def plot_scaling_curve(all_results: dict[str, dict[str, dict[str, dict]]],
                       output: Path, metric: str = "best_fitness"):
    """多规模缩放曲线图。

    Args:
        all_results: {scenario: {scale: {"DMDE": stats, "LLM-DMDE": stats}}}
        metric: "best_fitness" 或 "mean_time"
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    y_label = "Best Fitness" if metric == "best_fitness" else "Mean Time (s)"

    for idx, (scenario, scale_data) in enumerate(all_results.items()):
        ax = axes[idx]
        sizes = sorted(scale_data.keys())
        for algo in ["DMDE", "LLM-DMDE"]:
            values = []
            for s in sizes:
                stats = scale_data[s].get(algo)
                if stats:
                    values.append(stats[metric])
                else:
                    values.append(None)
            valid_sizes = [s for s, v in zip(sizes, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                color = COLORS[algo]
                ax.plot(valid_sizes, valid_values, "o-", color=color,
                        label=algo, linewidth=2, markersize=8)

        ax.set_xlabel("Scale")
        ax.set_ylabel(y_label)
        ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, output)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Markdown 报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_table(dmde: dict, llm: dict, scenario: str) -> str:
    """生成 Markdown 对比表。"""
    p, sig = wilcoxon_test(dmde["fitness_values"], llm["fitness_values"])
    dagger = "†" if sig else ""

    best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
    mean_imp = (dmde["mean_fitness"] - llm["mean_fitness"]) / abs(dmde["mean_fitness"]) * 100

    lines = [
        f"# Comparison Results: {scenario.upper()}\n",
        "> 30 independent runs. \u2020 Wilcoxon signed-rank test significant (p<0.05)\n",
        "| Metric | DMDE | LLM-DMDE | Improvement |",
        "|--------|------|----------|-------------|",
        f"| Best fitness | {dmde['best_fitness']:.2f} | {llm['best_fitness']:.2f} | {best_imp:+.2f}% |",
        f"| Mean\u00b1std | {dmde['mean_fitness']:.2f}\u00b1{dmde['std_fitness']:.2f} | {llm['mean_fitness']:.2f}\u00b1{llm['std_fitness']:.2f}{dagger} | {mean_imp:+.2f}% |",
        f"| Median | {dmde['median_fitness']:.2f} | {llm['median_fitness']:.2f} | |",
        f"| Feasible rate | {dmde['feasible_rate']*100:.0f}% ({dmde['feasible_count']}/{dmde['n_runs']}) | {llm['feasible_rate']*100:.0f}% ({llm['feasible_count']}/{llm['n_runs']}) | |",
        f"| Mean time | {dmde['mean_time']:.1f}s | {llm['mean_time']:.1f}s | |",
    ]
    if llm.get("llm_config"):
        lines.append(f"| LLM model | — | {llm['llm_config'].get('model', '—')} | |")
    lines.append("")
    return "\n".join(lines)


def gen_markdown_report(all_results: dict[str, dict[str, dict[str, dict]]]) -> str:
    """生成汇总 Markdown 报告。

    Args:
        all_results: {scenario: {scale: {"DMDE": stats, "LLM-DMDE": stats}}}
    """
    lines = [
        "# DMDE vs LLM-DMDE Comparison Report\n",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    for scenario, scale_data in all_results.items():
        label = SCENARIO_LABELS.get(scenario, scenario)
        lines.append(f"## {label} ({scenario.upper()})\n")

        for scale in sorted(scale_data.keys()):
            data = scale_data[scale]
            dmde = data.get("DMDE")
            llm = data.get("LLM-DMDE")
            if not dmde or not llm:
                continue

            p, sig = wilcoxon_test(dmde["fitness_values"], llm["fitness_values"])
            dagger = "†" if sig else ""
            best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
            mean_imp = (dmde["mean_fitness"] - llm["mean_fitness"]) / abs(dmde["mean_fitness"]) * 100

            lines.append(f"### Scale: {scale}\n")
            lines.append("| Metric | DMDE | LLM-DMDE | Improvement |")
            lines.append("|--------|------|----------|-------------|")
            lines.append(f"| Best fitness | {dmde['best_fitness']:.2f} | {llm['best_fitness']:.2f} | {best_imp:+.2f}% |")
            lines.append(f"| Mean±std | {dmde['mean_fitness']:.2f}±{dmde['std_fitness']:.2f} | {llm['mean_fitness']:.2f}±{llm['std_fitness']:.2f}{dagger} | {mean_imp:+.2f}% |")
            lines.append(f"| Feasible rate | {dmde['feasible_rate']*100:.0f}% | {llm['feasible_rate']*100:.0f}% | |")
            lines.append(f"| Mean time | {dmde['mean_time']:.1f}s | {llm['mean_time']:.1f}s | |")
            lines.append(f"| p-value | — | — | {p:.4f}{'\u2020' if sig else ''} |")
            lines.append("")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LaTeX 表格生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _latex_escape(text: str) -> str:
    """转义 LaTeX 特殊字符。"""
    for ch in ("&", "%", "#", "_"):
        text = text.replace(ch, f"\\{ch}")
    return text


def gen_latex_summary_table(all_results: dict[str, dict[str, dict[str, dict]]],
                            output: Path):
    """生成汇总 LaTeX 表格（所有场景×所有规模）。"""
    lines = [
        "% Auto-generated LaTeX table — directly \\input{} into main document",
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Comparison of DMDE and LLM-DMDE}",
        "  \\label{tab:dmde_comparison}",
        "  \\begin{tabular}{llrrrrrrr}",
        "    \\toprule",
        "    Scenario & Scale & \\multicolumn{2}{c}{Best Fitness} & \\multicolumn{2}{c}{Mean$\\pm$Std} & Feasible(\\%) & Time(s) & Improve(\\%) \\\\",
        "    \\cmidrule(lr){3-4} \\cmidrule(lr){5-6}",
        "    & & DMDE & LLM-DMDE & DMDE & LLM-DMDE & & & \\\\",
        "    \\midrule",
    ]

    for scenario, scale_data in all_results.items():
        label = SCENARIO_LABELS.get(scenario, scenario)
        first_row = True
        for scale in sorted(scale_data.keys()):
            data = scale_data[scale]
            dmde = data.get("DMDE")
            llm = data.get("LLM-DMDE")
            if not dmde or not llm:
                continue

            best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
            _, sig = wilcoxon_test(dmde["fitness_values"], llm["fitness_values"])
            best_str = f"{llm['best_fitness']:.2f}" + ("$^\\dagger$" if sig else "")
            mean_str = f"{llm['mean_fitness']:.2f}\\pm{llm['std_fitness']:.2f}" + ("$^\\dagger$" if sig else "")

            scenario_col = _latex_escape(label) if first_row else ""
            first_row = False

            # DMDE feasible as string
            dmde_feas = f"{dmde['feasible_rate']*100:.0f}"
            llm_feas = f"{llm['feasible_rate']*100:.0f}"

            lines.append(
                f"    {scenario_col} & {scale} "
                f"& {dmde['best_fitness']:.2f} & {best_str} "
                f"& {dmde['mean_fitness']:.2f}\\pm{dmde['std_fitness']:.2f} & {mean_str} "
                f"& {dmde_feas}/{llm_feas} "
                f"& {dmde['mean_time']:.1f}/{llm['mean_time']:.1f} "
                f"& {best_imp:+.2f} \\\\"
            )
        lines.append("    \\midrule")

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ LaTeX summary table: {output}")


def gen_latex_scenario_table(scenario: str, scale_data: dict[str, dict[str, dict]],
                             output: Path):
    """生成单场景 LaTeX 表格。"""
    label = SCENARIO_LABELS.get(scenario, scenario)
    lines = [
        "% Auto-generated LaTeX table — " + label,
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Comparison of DMDE and LLM-DMDE under {label}}}",
        f"  \\label{{tab:dmde_{scenario}}}",
        "  \\begin{tabular}{lrrrrrrrr}",
        "    \\toprule",
        "    Scale & \\multicolumn{2}{c}{Best Fitness} & \\multicolumn{2}{c}{Mean$\\pm$Std} & Feasible(\\%) & Time(s) & Improve(\\%) \\\\",
        "    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
        "    & DMDE & LLM-DMDE & DMDE & LLM-DMDE & & & \\\\",
        "    \\midrule",
    ]

    for scale in sorted(scale_data.keys()):
        data = scale_data[scale]
        dmde = data.get("DMDE")
        llm = data.get("LLM-DMDE")
        if not dmde or not llm:
            continue

        best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
        _, sig = wilcoxon_test(dmde["fitness_values"], llm["fitness_values"])
        best_str = f"{llm['best_fitness']:.2f}" + ("$^\\dagger$" if sig else "")
        mean_str = f"{llm['mean_fitness']:.2f}\\pm{llm['std_fitness']:.2f}" + ("$^\\dagger$" if sig else "")

        dmde_feas = f"{dmde['feasible_rate']*100:.0f}"
        llm_feas = f"{llm['feasible_rate']*100:.0f}"

        lines.append(
            f"    {scale} "
            f"& {dmde['best_fitness']:.2f} & {best_str} "
            f"& {dmde['mean_fitness']:.2f}\\pm{dmde['std_fitness']:.2f} & {mean_str} "
            f"& {dmde_feas}/{llm_feas} "
            f"& {dmde['mean_time']:.1f}/{llm['mean_time']:.1f} "
            f"& {best_imp:+.2f} \\\\"
        )

    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
        "",
    ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ LaTeX scenario table ({scenario}): {output}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="DMDE vs LLM-DMDE Comparison Experiment")
    parser.add_argument("--scenario", default="nm", choices=list(SCENARIO_MAP.keys()),
                        help="Scenario: nm/ngt/nlt (default: nm)")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Multiple scenarios or 'all' (overrides --scenario)")
    parser.add_argument("--runs", type=int, default=30, help="Number of runs (default: 30)")
    parser.add_argument("--sizes", nargs="+", default=None, choices=SCALES,
                        help="Scales: small medium large (default: medium only)")
    parser.add_argument("--format", choices=["md", "latex", "both"], default="md",
                        help="Output format: md/latex/both (default: md)")
    parser.add_argument("--only", choices=["baseline", "llm"], help="Run only specified experiment")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparison generation")
    parser.add_argument("--baseline-data", type=str, default=None,
                        help="Directly specify baseline result JSON (skip running)")
    parser.add_argument("--llm-data", type=str, default=None,
                        help="Directly specify LLM result JSON (skip running)")
    parser.add_argument("--solver-params", type=str, default=None,
                        help='Custom solver params JSON, e.g. \'{"pop_size":150}\'')
    parser.add_argument("--ablation", action="store_true",
                        help="Ablation experiment mode (disable LLM modules)")
    args = parser.parse_args()

    # 解析场景列表
    if args.scenarios:
        if "all" in args.scenarios:
            scenarios = list(SCENARIO_MAP.keys())
        else:
            scenarios = args.scenarios
    else:
        scenarios = [args.scenario]

    # 解析规模列表（默认仅 medium）
    sizes = args.sizes if args.sizes else ["medium"]

    # 结果存储：{scenario: {scale: {"DMDE": stats, "LLM-DMDE": stats}}}
    all_results: dict[str, dict[str, dict[str, dict]]] = {}

    print(f"{'='*60}")
    print(f"DMDE vs LLM-DMDE Comparison Experiment")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Scales: {', '.join(sizes)}")
    print(f"  Runs: {args.runs}")
    print(f"  Format: {args.format}")
    if args.solver_params:
        print(f"  Solver params: {args.solver_params}")
    if args.ablation:
        print(f"  Ablation: enabled")
    print(f"{'='*60}")

    results_dir = SCRIPT_DIR / "results"
    figures_dir = results_dir / "figures"
    latex_dir = results_dir / "latex"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 遍历场景×规模 ──────────────────────────────────────
    for scenario in scenarios:
        dmde_dir, llm_dir = SCENARIO_MAP[scenario]
        dmde_path = EXPERIMENTS_DIR / dmde_dir
        llm_path = EXPERIMENTS_DIR / llm_dir

        all_results[scenario] = {}

        for scale in sizes:
            print(f"\n{'#'*60}")
            print(f"# Scenario: {scenario.upper()} | Scale: {scale}")
            print(f"{'#'*60}")

            all_results[scenario][scale] = {}

            # ── 运行实验 ──────────────────────────────────
            baseline_ok, llm_ok = False, False

            if args.only != "llm" and not args.baseline_data:
                baseline_ok = run_experiment(
                    dmde_path / "run.py", args.runs,
                    f"DMDE Baseline ({scenario}, {scale})",
                    scale=scale, solver_params=args.solver_params,
                )
            elif args.baseline_data:
                baseline_ok = True

            if args.only != "baseline" and not args.llm_data:
                llm_ok = run_experiment(
                    llm_path / "run.py", args.runs,
                    f"LLM-DMDE ({scenario}, {scale})",
                    scale=scale, solver_params=args.solver_params,
                    ablation=args.ablation,
                )
            elif args.llm_data:
                llm_ok = True

            # ── 查找结果 ──────────────────────────────────
            if args.baseline_data:
                dmde_json = Path(args.baseline_data)
            else:
                dmde_json = find_result_json(dmde_path)

            if args.llm_data:
                llm_json = Path(args.llm_data)
            else:
                llm_json = find_result_json(llm_path)

            if not dmde_json or not dmde_json.exists():
                print(f"\n⚠️  DMDE result not found: {dmde_json}, skipping {scenario}/{scale}")
                continue
            if not llm_json or not llm_json.exists():
                print(f"\n⚠️  LLM-DMDE result not found: {llm_json}, skipping {scenario}/{scale}")
                continue

            # ── 生成对比 ──────────────────────────────────
            if args.no_compare:
                print("\nSkipping comparison generation.")
                continue

            dmde_data = load_data(dmde_json)
            llm_data = load_data(llm_json)
            dmde_stats = extract_stats(dmde_data, "DMDE")
            llm_stats = extract_stats(llm_data, "LLM-DMDE")

            all_results[scenario][scale]["DMDE"] = dmde_stats
            all_results[scenario][scale]["LLM-DMDE"] = llm_stats

            # 对比表
            table = gen_table(dmde_stats, llm_stats, scenario)
            suffix = f"_{scenario}_{scale}" if len(sizes) > 1 else f"_{scenario}"
            table_path = results_dir / f"comparison{suffix}.md"
            table_path.write_text(table, encoding="utf-8")
            print(f"  ✅ Comparison table: {table_path}")

            # 图表
            fig_suffix = f"_{scenario}_{scale}" if len(sizes) > 1 else f"_{scenario}"
            plot_boxplot(dmde_stats, llm_stats,
                         figures_dir / f"boxplot{fig_suffix}.png", scenario)
            plot_convergence(dmde_stats, llm_stats,
                             figures_dir / f"convergence{fig_suffix}.png", scenario)
            plot_improvement_bar(dmde_stats, llm_stats,
                                 figures_dir / f"improvement{fig_suffix}.png", scenario)
            plot_feasibility_bar(dmde_stats, llm_stats,
                                 figures_dir / f"feasibility{fig_suffix}.png", scenario)
            plot_time_bar(dmde_stats, llm_stats,
                          figures_dir / f"time{fig_suffix}.png", scenario)

            # 保存元数据
            best_imp = (dmde_stats["best_fitness"] - llm_stats["best_fitness"]) \
                        / abs(dmde_stats["best_fitness"]) * 100
            meta = {
                "scenario": scenario,
                "scale": scale,
                "n_runs": args.runs,
                "baseline_json": str(dmde_json),
                "llm_json": str(llm_json),
                "dmde_best": dmde_stats["best_fitness"],
                "llm_best": llm_stats["best_fitness"],
                "improvement_pct": best_imp,
            }
            meta_path = results_dir / f"meta{fig_suffix}.json"
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 多规模缩放曲线图 ──────────────────────────────────────
    if len(sizes) > 1 and not args.no_compare:
        print(f"\n{'='*60}")
        print("Generating multi-scale scaling curves")
        print(f"{'='*60}")

        plot_scaling_curve(all_results,
                           figures_dir / "scaling_fitness.png",
                           metric="best_fitness")
        plot_scaling_curve(all_results,
                           figures_dir / "scaling_time.png",
                           metric="mean_time")

    # ── 生成汇总报告 ──────────────────────────────────────────
    if not args.no_compare and all_results:
        # Markdown 汇总报告
        md_report = gen_markdown_report(all_results)
        md_path = results_dir / "summary_report.md"
        md_path.write_text(md_report, encoding="utf-8")
        print(f"\n  ✅ Markdown summary report: {md_path}")

        # LaTeX 表格
        if args.format in ("latex", "both"):
            latex_dir.mkdir(parents=True, exist_ok=True)

            gen_latex_summary_table(all_results,
                                    latex_dir / "summary_table.tex")

            for scenario, scale_data in all_results.items():
                gen_latex_scenario_table(scenario, scale_data,
                                         latex_dir / f"table_{scenario}.tex")

    # ── 最终汇总 ──────────────────────────────────────────────
    if all_results:
        print(f"\n{'='*60}")
        print("Comparison Experiment Summary")
        print(f"{'='*60}")
        for scenario, scale_data in all_results.items():
            label = SCENARIO_LABELS.get(scenario, scenario)
            for scale, data in scale_data.items():
                dmde = data.get("DMDE")
                llm = data.get("LLM-DMDE")
                if dmde and llm:
                    imp = (dmde["best_fitness"] - llm["best_fitness"]) \
                          / abs(dmde["best_fitness"]) * 100
                    print(f"  {label} ({scale}): "
                          f"DMDE={dmde['best_fitness']:.2f}, "
                          f"LLM-DMDE={llm['best_fitness']:.2f}, "
                          f"Improvement={imp:+.2f}%")
        print(f"\n  Output: {results_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
