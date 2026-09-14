# -*- coding: utf-8 -*-
"""compare_experiments.py — 多算法实验对比

支持模式：
  1v1 模式（兼容旧版）：
    python compare.py --baseline xxx --llm yyy

  多算法模式（推荐）：
    python compare.py --experiments dmde_01 ga_01 llm_dmde_01
    python compare.py --experiments dmde_01 ablation_sc ablation_pi ablation_vanilla

  自动发现模式：
    python compare.py                          # 索引中所有 baseline + llm
    python compare.py --tags 10u10t            # 按标签筛选
    python compare.py --list                   # 列出索引

输出（experiments/comparison_results/）：
  comparison.md                 — 多算法对比表（均值±标准差，Wilcoxon †）
  figures/comparison_boxplot.png    — 箱线图
  figures/comparison_convergence.png — 收敛曲线
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
# 脚本位于 experiments/exp_llm_enhanced_dmde/exp_llm_dmde_01/
# _BASE 指向 experiments/ 目录
_BASE = _SCRIPT_DIR.parent.parent

OUTPUT_DIR = _SCRIPT_DIR / "comparison_results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# 配色方案（最多支持8个算法）
ALGO_COLORS = [
    "#4ECDC4",  # 青 — baseline1
    "#FF6B6B",  # 红 — llm
    "#45B7D1",  # 蓝 — baseline2
    "#96CEB4",  # 绿 — baseline3
    "#FFEAA7",  # 黄 — ablation1
    "#DDA0DD",  # 紫 — ablation2
    "#FF8C00",  # 橙 — ablation3
    "#87CEEB",  # 天蓝
]


# ── 数据加载 ──────────────────────────────────────────────────

def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_run_stats(data: dict, label: str | None = None) -> dict:
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
        "label": label or meta.get("description", scenario["name"]),
        "n_runs": len(runs),
        "best_fitness": metrics["best_fitness"],
        "mean_fitness": metrics["mean_fitness"],
        "std_fitness": metrics["std_fitness"],
        "worst_fitness": metrics["worst_fitness"],
        "median_fitness": metrics.get("median_fitness", float(np.median(fitness_values))),
        "fitness_values": fitness_values,
        "feasible_count": feasible_count,
        "feasible_rate": feasible_count / max(len(runs), 1),
        "mean_time": float(np.mean(times)),
        "times": times,
        "cost_histories": cost_histories,
        "solver_params": meta.get("solver_params", {}),
        "llm_config": meta.get("llm_config", {}),
    }


# ── 统计检验 ──────────────────────────────────────────────────

def wilcoxon_test(baseline_values: list[float], test_values: list[float]) -> tuple[float, bool]:
    """Wilcoxon 符号秩检验。

    Returns:
        (p_value, significant): p 值和是否显著 (p < 0.05)。
    """
    try:
        from scipy.stats import wilcoxon
        diffs = [b - t for b, t in zip(baseline_values, test_values) if b != t]
        if len(diffs) < 3:
            return 1.0, False
        stat, p = wilcoxon(diffs, alternative="two-sided")
        return float(p), p < 0.05
    except ImportError:
        # scipy 不可用，跳过检验
        return 1.0, False


# ── 从索引发现 ────────────────────────────────────────────────

def resolve_from_registry(
    experiment_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """从索引发现实验结果。

    Returns:
        [{"id": "dmde_01", "path": Path, "label": "...", "tags": [...]}, ...]
    """
    try:
        from registry import get_results, list_experiments
    except ImportError:
        sys.path.insert(0, str(_SCRIPT_DIR))
        from registry import get_results, list_experiments

    if experiment_ids:
        results = []
        for eid in experiment_ids:
            matches = get_results(experiment_id=eid)
            if matches:
                results.append(matches[0])
            else:
                print(f"  ⚠️ 索引中未找到实验: {eid}")
        return results

    return get_results(tags=tags)


# ── 生成对比表 ────────────────────────────────────────────────

def gen_multi_algo_table(stats_list: list[dict], baseline_idx: int = 0) -> str:
    """生成多算法对比 Markdown 表。

    格式：论文级表格，含 Wilcoxon 显著性标记。
    """
    baseline = stats_list[baseline_idx]

    lines = [
        "# 多算法对比实验结果\n",
        "> 所有结果基于30次独立运行。† 表示与基线相比 Wilcoxon 秩和检验显著 (p<0.05)。\n",
    ]

    # 表头
    header = "| 指标 |"
    separator = "|------|"
    for s in stats_list:
        header += f" {s['label']} |"
        separator += "------|"
    lines.append(header)
    lines.append(separator)

    # Best fitness
    row = "| Best fitness |"
    best_val = min(s["best_fitness"] for s in stats_list)
    for i, s in enumerate(stats_list):
        marker = " **" if s["best_fitness"] == best_val else " "
        end = "** |" if s["best_fitness"] == best_val else " |"
        row += f"{marker}{s['best_fitness']:.2f}{end}"
    lines.append(row)

    # Mean ± std（含 Wilcoxon 检验）
    row = "| Mean fitness |"
    for i, s in enumerate(stats_list):
        if i == baseline_idx:
            row += f" {s['mean_fitness']:.2f}±{s['std_fitness']:.2f} |"
        else:
            p, sig = wilcoxon_test(baseline["fitness_values"], s["fitness_values"])
            dagger = "†" if sig else ""
            row += f" {s['mean_fitness']:.2f}±{s['std_fitness']:.2f}{dagger} |"
    lines.append(row)

    # Median
    row = "| Median fitness |"
    for s in stats_list:
        row += f" {s['median_fitness']:.2f} |"
    lines.append(row)

    # 可行解率
    row = "| 可行解率 |"
    for s in stats_list:
        row += f" {s['feasible_rate']*100:.0f}% ({s['feasible_count']}/{s['n_runs']}) |"
    lines.append(row)

    # 平均耗时
    row = "| 平均耗时(s) |"
    for s in stats_list:
        row += f" {s['mean_time']:.1f} |"
    lines.append(row)

    # LLM 模型（如有）
    has_llm = any(s.get("llm_config") for s in stats_list)
    if has_llm:
        row = "| LLM 模型 |"
        for s in stats_list:
            model = s.get("llm_config", {}).get("model", "—")
            row += f" {model} |"
        lines.append(row)

    # 改进幅度（相对基线）
    if len(stats_list) > 1:
        lines.append("")
        lines.append("### 相对基线改进\n")
        lines.append("| 算法 | Best 改进 | Mean 改进 |")
        lines.append("|------|-----------|-----------|")
        for i, s in enumerate(stats_list):
            if i == baseline_idx:
                lines.append(f"| {s['label']} | — (基线) | — (基线) |")
            else:
                best_imp = (baseline["best_fitness"] - s["best_fitness"]) / abs(baseline["best_fitness"]) * 100
                mean_imp = (baseline["mean_fitness"] - s["mean_fitness"]) / abs(baseline["mean_fitness"]) * 100
                lines.append(f"| {s['label']} | {best_imp:+.2f}% | {mean_imp:+.2f}% |")

    lines.append("")
    return "\n".join(lines)


# ── 可视化 ────────────────────────────────────────────────────

def _setup_sci_font():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_multi_boxplot(stats_list: list[dict], output_path: Path):
    """多算法 fitness 箱线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_sci_font()

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(stats_list)), 5))

    data_to_plot = [s["fitness_values"] for s in stats_list]
    labels = [s["label"] for s in stats_list]
    colors = ALGO_COLORS[:len(stats_list)]

    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                    widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 散点
    for i, (d, c) in enumerate(zip(data_to_plot, colors), 1):
        x = np.random.normal(i, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, color=c, edgecolors="black", linewidths=0.5, s=40, zorder=5)

    ax.set_ylabel("Best Fitness")
    ax.set_title("Fitness Distribution Comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 箱线图: {output_path}")


def plot_multi_convergence(stats_list: list[dict], output_path: Path):
    """多算法收敛曲线对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _setup_sci_font()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ALGO_COLORS[:len(stats_list)]

    for s, color in zip(stats_list, colors):
        histories = s["cost_histories"]
        if not histories:
            continue

        # 找最优 run
        best_idx = min(range(len(histories)),
                       key=lambda i: histories[i][-1] if histories[i] else float("inf"))

        # 所有 run（淡色）
        for h in histories:
            if h:
                ax.plot(range(len(h)), h, color=color, alpha=0.15, linewidth=0.8)

        # 最优 run（粗线）
        best_h = histories[best_idx]
        ax.plot(range(len(best_h)), best_h, color=color, linewidth=2.5, label=s["label"])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Convergence Curves Comparison")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 收敛曲线: {output_path}")


# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多算法实验对比")
    parser.add_argument("--experiments", type=str, nargs="*", default=None,
                        help="实验 ID 列表（从索引读取），如 --experiments dmde_01 ga_01 llm_dmde_01")
    parser.add_argument("--tags", type=str, nargs="*", default=None,
                        help="按标签筛选实验")
    parser.add_argument("--baseline", type=str, default=None,
                        help="1v1 模式：基线结果路径")
    parser.add_argument("--llm", type=str, default=None,
                        help="1v1 模式：LLM 结果路径")
    parser.add_argument("--baseline-idx", type=int, default=0,
                        help="多算法模式：基线索引（默认第0个）")
    parser.add_argument("--list", action="store_true",
                        help="列出索引中的所有实验")
    args = parser.parse_args()

    # ── 列出实验 ──────────────────────────────────────────────
    if args.list:
        try:
            from registry import list_experiments
        except ImportError:
            sys.path.insert(0, str(_SCRIPT_DIR))
            from registry import list_experiments
        experiments = list_experiments()
        if not experiments:
            print("索引为空。请先运行实验。")
        else:
            print(f"\n索引中共 {len(experiments)} 个实验：")
            for exp in experiments:
                tags_str = ", ".join(exp.get("tags", []))
                print(f"  [{exp['id']}] {exp['label']}")
                print(f"    路径: {exp['result_path']}")
                print(f"    标签: {tags_str}")
        return

    # ── 收集实验数据 ──────────────────────────────────────────
    stats_list: list[dict] = []

    # 1v1 兼容模式
    if args.baseline and args.llm:
        baseline_path = Path(args.baseline)
        llm_path = Path(args.llm)
        if not baseline_path.exists():
            print(f"❌ 基线结果不存在: {baseline_path}"); sys.exit(1)
        if not llm_path.exists():
            print(f"❌ LLM 结果不存在: {llm_path}"); sys.exit(1)
        stats_list.append(extract_run_stats(load_data(baseline_path), "DMDE 基线"))
        stats_list.append(extract_run_stats(load_data(llm_path), "LLM-DMDE"))

    # 多算法模式
    elif args.experiments:
        registry_entries = resolve_from_registry(experiment_ids=args.experiments)
        for entry in registry_entries:
            path = _BASE / entry["result_path"]
            if not path.exists():
                print(f"  ⚠️ 结果文件不存在: {path}，跳过")
                continue
            stats_list.append(extract_run_stats(load_data(path), entry["label"]))
        if not stats_list:
            print("❌ 无有效实验结果"); sys.exit(1)

    # 自动发现模式
    else:
        registry_entries = resolve_from_registry(tags=args.tags)
        for entry in registry_entries:
            path = _BASE / entry["result_path"]
            if not path.exists():
                continue
            stats_list.append(extract_run_stats(load_data(path), entry["label"]))
        if len(stats_list) < 2:
            print("❌ 索引中至少需要2个实验结果。用 --list 查看。")
            sys.exit(1)

    # ── 打印摘要 ──────────────────────────────────────────────
    n = len(stats_list)
    print(f"\n{'='*60}")
    print(f"多算法对比：{n} 个实验")
    print(f"{'='*60}")
    for i, s in enumerate(stats_list):
        marker = " ← 基线" if i == args.baseline_idx else ""
        print(f"  [{i}] {s['label']}: {s['n_runs']} runs, "
              f"best={s['best_fitness']:.2f}, mean={s['mean_fitness']:.2f}{marker}")

    # ── 生成输出 ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1] 生成对比表...")
    table = gen_multi_algo_table(stats_list, baseline_idx=args.baseline_idx)
    table_path = OUTPUT_DIR / "comparison.md"
    table_path.write_text(table, encoding="utf-8")
    print(f"  ✅ 对比表: {table_path}")

    print("\n[2] 生成对比图...")
    plot_multi_boxplot(stats_list, FIGURES_DIR / "comparison_boxplot.png")
    plot_multi_convergence(stats_list, FIGURES_DIR / "comparison_convergence.png")

    print(f"\n✅ 对比完成。输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
