# -*- coding: utf-8 -*-
"""run_comparison.py — DMDE 基线 vs LLM-DMDE 对比实验

同时运行两个实验，生成对比报告，结果保存在本目录的 results/ 中。

用法：
    # N=M 场景（默认）
    python run_comparison.py

    # 指定场景
    python run_comparison.py --scenario nm    # N=M
    python run_comparison.py --scenario ngt   # N>M
    python run_comparison.py --scenario nlt   # N<M

    # 指定运行次数
    python run_comparison.py --runs 30

    # 只跑一个
    python run_comparison.py --only baseline
    python run_comparison.py --only llm

    # 跳过对比生成
    python run_comparison.py --no-compare
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

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

# 配色
COLORS = {"DMDE": "#4ECDC4", "LLM-DMDE": "#FF6B6B"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(run_py: Path, n_runs: int, label: str) -> bool:
    """运行单个实验。"""
    print(f"\n{'='*60}")
    print(f"▶ {label}")
    print(f"  脚本: {run_py}")
    print(f"  运行次数: {n_runs}")
    print(f"{'='*60}\n")

    import os
    env = {**os.environ, "EXP_N_RUNS": str(n_runs)}

    t0 = time.time()
    result = subprocess.run([sys.executable, str(run_py)], env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ {label} 失败 (exit={result.returncode})")
        return False
    print(f"\n✅ {label} 完成 ({elapsed:.1f}s)")
    return True


def find_result_json(experiment_dir: Path) -> Path | None:
    """自动查找实验结果 JSON。"""
    results_dir = experiment_dir / "results"
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
# 生成报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def gen_table(dmde: dict, llm: dict, scenario: str) -> str:
    """生成 Markdown 对比表。"""
    p, sig = wilcoxon_test(dmde["fitness_values"], llm["fitness_values"])
    dagger = "†" if sig else ""

    best_imp = (dmde["best_fitness"] - llm["best_fitness"]) / abs(dmde["best_fitness"]) * 100
    mean_imp = (dmde["mean_fitness"] - llm["mean_fitness"]) / abs(dmde["mean_fitness"]) * 100

    lines = [
        f"# 对比实验结果：{scenario.upper()}\n",
        "> 30次独立运行。† Wilcoxon 秩和检验显著 (p<0.05)\n",
        "| 指标 | DMDE | LLM-DMDE | 改进 |",
        "|------|------|----------|------|",
        f"| Best fitness | {dmde['best_fitness']:.2f} | {llm['best_fitness']:.2f} | {best_imp:+.2f}% |",
        f"| Mean±std | {dmde['mean_fitness']:.2f}±{dmde['std_fitness']:.2f} | {llm['mean_fitness']:.2f}±{llm['std_fitness']:.2f}{dagger} | {mean_imp:+.2f}% |",
        f"| Median | {dmde['median_fitness']:.2f} | {llm['median_fitness']:.2f} | |",
        f"| 可行解率 | {dmde['feasible_rate']*100:.0f}% ({dmde['feasible_count']}/{dmde['n_runs']}) | {llm['feasible_rate']*100:.0f}% ({llm['feasible_count']}/{llm['n_runs']}) | |",
        f"| 平均耗时 | {dmde['mean_time']:.1f}s | {llm['mean_time']:.1f}s | |",
    ]
    if llm.get("llm_config"):
        lines.append(f"| LLM 模型 | — | {llm['llm_config'].get('model', '—')} | |")
    lines.append("")
    return "\n".join(lines)


def _setup_font():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 14,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.unicode_minus": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def plot_boxplot(dmde: dict, llm: dict, output: Path, scenario: str):
    """箱线图对比。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([dmde["fitness_values"], llm["fitness_values"]],
                    tick_labels=["DMDE", "LLM-DMDE"], patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    for patch, color in zip(bp["boxes"], [COLORS["DMDE"], COLORS["LLM-DMDE"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (d, c) in enumerate(zip([dmde["fitness_values"], llm["fitness_values"]],
                                     [COLORS["DMDE"], COLORS["LLM-DMDE"]]), 1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color=c, edgecolors="black", linewidths=0.5, s=40, zorder=5)

    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Fitness Distribution — {scenario.upper()}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 箱线图: {output}")


def plot_convergence(dmde: dict, llm: dict, output: Path, scenario: str):
    """收敛曲线对比。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    for stats, color in [(dmde, COLORS["DMDE"]), (llm, COLORS["LLM-DMDE"])]:
        histories = stats["cost_histories"]
        if not histories:
            continue
        best_idx = min(range(len(histories)),
                       key=lambda i: histories[i][-1] if histories[i] else float("inf"))
        for h in histories:
            if h:
                ax.plot(range(len(h)), h, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(range(len(histories[best_idx])), histories[best_idx],
                color=color, linewidth=2.5, label=stats["label"])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title(f"Convergence — {scenario.upper()}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 收敛曲线: {output}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="DMDE vs LLM-DMDE 对比实验")
    parser.add_argument("--scenario", default="nm", choices=list(SCENARIO_MAP.keys()),
                        help="场景: nm/n_gt/n_lt (默认: nm)")
    parser.add_argument("--runs", type=int, default=2, help="运行次数 (默认: 2)")
    parser.add_argument("--only", choices=["baseline", "llm"], help="只运行指定实验")
    parser.add_argument("--no-compare", action="store_true", help="跳过对比生成")
    parser.add_argument("--baseline-data", type=str, default=None,
                        help="直接指定基线结果 JSON（跳过运行）")
    parser.add_argument("--llm-data", type=str, default=None,
                        help="直接指定 LLM 结果 JSON（跳过运行）")
    args = parser.parse_args()

    dmde_dir, llm_dir = SCENARIO_MAP[args.scenario]
    dmde_path = EXPERIMENTS_DIR / dmde_dir
    llm_path = EXPERIMENTS_DIR / llm_dir

    print(f"{'='*60}")
    print(f"对比实验：{args.scenario.upper()}")
    print(f"  DMDE:     {dmde_dir}")
    print(f"  LLM-DMDE: {llm_dir}")
    print(f"  运行次数: {args.runs}")
    print(f"{'='*60}")

    # ── 运行实验 ──────────────────────────────────────────────
    baseline_ok, llm_ok = False, False

    if args.only != "llm" and not args.baseline_data:
        baseline_ok = run_experiment(dmde_path / "run.py", args.runs, f"DMDE 基线 ({args.scenario})")
    elif args.baseline_data:
        baseline_ok = True

    if args.only != "baseline" and not args.llm_data:
        llm_ok = run_experiment(llm_path / "run.py", args.runs, f"LLM-DMDE ({args.scenario})")
    elif args.llm_data:
        llm_ok = True

    # ── 查找结果 ──────────────────────────────────────────────
    if args.baseline_data:
        dmde_json = Path(args.baseline_data)
    else:
        dmde_json = find_result_json(dmde_path)

    if args.llm_data:
        llm_json = Path(args.llm_data)
    else:
        llm_json = find_result_json(llm_path)

    if not dmde_json or not dmde_json.exists():
        print(f"\n❌ DMDE 结果不存在: {dmde_json}")
        sys.exit(1)
    if not llm_json or not llm_json.exists():
        print(f"\n❌ LLM-DMDE 结果不存在: {llm_json}")
        sys.exit(1)

    # ── 生成对比 ──────────────────────────────────────────────
    if args.no_compare:
        print("\n跳过对比生成。")
        return

    print(f"\n{'='*60}")
    print(f"生成对比报告")
    print(f"{'='*60}")

    dmde_data = load_data(dmde_json)
    llm_data = load_data(llm_json)
    dmde_stats = extract_stats(dmde_data, "DMDE")
    llm_stats = extract_stats(llm_data, "LLM-DMDE")

    results_dir = SCRIPT_DIR / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 对比表
    table = gen_table(dmde_stats, llm_stats, args.scenario)
    table_path = results_dir / f"comparison_{args.scenario}.md"
    table_path.write_text(table, encoding="utf-8")
    print(f"  ✅ 对比表: {table_path}")

    # 图表
    plot_boxplot(dmde_stats, llm_stats,
                 figures_dir / f"boxplot_{args.scenario}.png", args.scenario)
    plot_convergence(dmde_stats, llm_stats,
                     figures_dir / f"convergence_{args.scenario}.png", args.scenario)

    # 保存元数据
    meta = {
        "scenario": args.scenario,
        "n_runs": args.runs,
        "baseline_json": str(dmde_json),
        "llm_json": str(llm_json),
        "dmde_best": dmde_stats["best_fitness"],
        "llm_best": llm_stats["best_fitness"],
        "improvement_pct": (dmde_stats["best_fitness"] - llm_stats["best_fitness"])
                           / abs(dmde_stats["best_fitness"]) * 100,
    }
    meta_path = results_dir / f"meta_{args.scenario}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"对比完成")
    print(f"  DMDE best:     {dmde_stats['best_fitness']:.2f}")
    print(f"  LLM-DMDE best: {llm_stats['best_fitness']:.2f}")
    print(f"  改进:          {meta['improvement_pct']:+.2f}%")
    print(f"  输出目录:      {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
