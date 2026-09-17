# -*- coding: utf-8 -*-
"""analyze.py — 消融实验结果分析

读取 8 组实验结果，生成：
- 汇总表（Markdown / LaTeX / CSV）
- 收敛曲线对比图
- CR 变化轨迹图
- 解质量箱线图
- 时间开销堆叠柱状图
- LLM 决策日志摘要
- Mann-Whitney U 统计检验

用法：python analyze.py

依赖：numpy（必需）、matplotlib（可选，图表）、scipy（可选，统计检验）。
缺失的可选依赖会优雅降级，文本输出仍可用。
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

from core.results import load_results

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
CONFIG_COLORS = {
    "A0": "#1f77b4",
    "A1": "#ff7f0e",
    "A2": "#2ca02c",
    "A3": "#d62728",
}
SCENARIO_LABELS = {
    "S1": r"balanced ($N=M=10$)",
    "S2": r"srp ($N=10, M=20$)",
}


# ── 统计计算 ──────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    if not results:
        return {}
    fitness = np.array([r["best_fitness"] for r in results])
    times = np.array([r["total_time"] for r in results])
    llm_times = np.array([r.get("llm_time", 0) for r in results])
    llm_init_times = np.array([r.get("llm_init_time", 0) for r in results])
    llm_cr_times = np.array([r.get("llm_cr_time", 0) for r in results])
    llm_calls = np.array([r.get("llm_call_count", 0) for r in results])
    dmde_times = np.array([r.get("dmde_time", 0) for r in results])
    # ddof=1 (样本标准差)；n=1 时返回 0.0 而非 nan
    std_val = float(fitness.std(ddof=1)) if len(fitness) > 1 else 0.0
    return {
        "n_runs": len(results),
        "best": round(float(fitness.min()), 2),
        "mean": round(float(fitness.mean()), 2),
        "std": round(std_val, 2),
        "median": round(float(np.median(fitness)), 2),
        "total_time_mean": round(float(times.mean()), 2),
        "dmde_time_mean": round(float(dmde_times.mean()), 2),
        "llm_time_mean": round(float(llm_times.mean()), 2),
        "llm_init_time_mean": round(float(llm_init_times.mean()), 2),
        "llm_cr_time_mean": round(float(llm_cr_times.mean()), 2),
        "llm_calls_mean": round(float(llm_calls.mean()), 1),
        "fitness_array": fitness,
        # 保留原始结果用于进一步分析
        "_raw": results,
    }


def mannwhitney_test(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U 秩和检验（非配对双侧）。

    注意：文献里常被误称为"Wilcoxon 秩和检验"，两者等价时
    此处用 Mann-Whitney U 实现（适用非配对样本）。
    当样本量 < 5 时返回 -1.0 表示"未检验"。
    """
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
    """根据 p 值返回显著性标记；p < 0 表示未检验。"""
    if p < 0: return "---"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."


# ── 收敛曲线分析 ──────────────────────────────────────────

def extract_convergence_curves(results: list[dict]) -> list[list[float]]:
    """提取所有 run 的收敛曲线。"""
    curves = []
    for r in results:
        curve = r.get("convergence_curve", [])
        if curve:
            curves.append(curve)
    return curves


def extract_cr_histories(results: list[dict]) -> list[list[float]]:
    """提取所有 run 的 CR 变化轨迹。"""
    histories = []
    for r in results:
        cr = r.get("cr_history", [])
        if cr:
            histories.append(cr)
    return histories


def compute_convergence_stats(curves: list[list[float]]) -> dict:
    """计算收敛曲线的均值±标准差。"""
    if not curves:
        return {}
    min_len = min(len(c) for c in curves)
    arr = np.array([c[:min_len] for c in curves])
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "n_runs": len(curves),
        "length": min_len,
    }


def find_convergence_gen(curve: list[float], threshold: float = 0.95) -> int:
    """找到达到 95% 最终最优解的代数。

    注：当前未被调用，预留给后续单 run 收敛速度分析模块。
    """
    if not curve:
        return -1
    final_best = curve[-1]
    target = final_best * threshold
    for i, v in enumerate(curve):
        if v <= target:
            return i
    return len(curve) - 1


# ── LLM 决策分析 ──────────────────────────────────────────

def extract_llm_decisions(results: list[dict]) -> list[dict]:
    """提取所有 run 的 LLM 决策记录。"""
    all_decisions = []
    for r in results:
        decisions = r.get("llm_decisions", [])
        for d in decisions:
            d["_run_seed"] = r.get("seed")
            all_decisions.append(d)
    return all_decisions


def summarize_llm_decisions(results: list[dict]) -> dict:
    """汇总 LLM 决策统计。"""
    decisions = extract_llm_decisions(results)
    if not decisions:
        return {}

    modules = {}
    for d in decisions:
        mod = d.get("module", "unknown")
        if mod not in modules:
            modules[mod] = {"count": 0, "durations": [], "cr_values": []}
        modules[mod]["count"] += 1
        modules[mod]["durations"].append(d.get("duration", 0))
        if mod == "search_controller":
            cr = d.get("parsed_decision", {}).get("cr", 0)
            if cr:
                modules[mod]["cr_values"].append(cr)
        elif mod == "population_init":
            pass  # PopInit 不产生 CR

    summary = {}
    for mod, data in modules.items():
        durs = np.array(data["durations"])
        summary[mod] = {
            "total_calls": data["count"],
            "avg_duration": round(float(durs.mean()), 3) if len(durs) else 0,
            "total_duration": round(float(durs.sum()), 2),
        }
        if data["cr_values"]:
            crs = np.array(data["cr_values"])
            summary[mod]["cr_mean"] = round(float(crs.mean()), 4)
            summary[mod]["cr_std"] = round(float(crs.std()), 4)
            summary[mod]["cr_min"] = round(float(crs.min()), 4)
            summary[mod]["cr_max"] = round(float(crs.max()), 4)

    return summary


# ── 可视化 ────────────────────────────────────────────────

def plot_convergence_comparison(all_stats: dict, scenario_key: str):
    """绘制单场景的收敛曲线对比图。"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for config_key in CONFIGS:
        stats = all_stats.get(f"{scenario_key}_{config_key}", {})
        raw = stats.get("_raw", [])
        curves = extract_convergence_curves(raw)
        if not curves:
            continue

        conv = compute_convergence_stats(curves)
        if not conv:
            continue

        x = np.arange(conv["length"])
        mean = np.array(conv["mean"])
        std = np.array(conv["std"])
        color = CONFIG_COLORS[config_key]

        ax.plot(x, mean, label=CONFIG_LABELS[config_key], color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Convergence Comparison — {SCENARIO_LABELS.get(scenario_key, scenario_key)}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / f"convergence_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  收敛曲线: {out}")


def plot_cr_comparison(all_stats: dict, scenario_key: str):
    """绘制单场景的 CR 变化轨迹对比图。"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for config_key in ["A0", "A1", "A3"]:  # A2 无 CR Control
        stats = all_stats.get(f"{scenario_key}_{config_key}", {})
        raw = stats.get("_raw", [])
        histories = extract_cr_histories(raw)
        if not histories:
            continue

        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.arange(min_len)
        color = CONFIG_COLORS[config_key]

        ax.plot(x, mean, label=CONFIG_LABELS[config_key], color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Crossover Rate (CR)", fontsize=12)
    ax.set_title(f"CR Trajectory — {SCENARIO_LABELS.get(scenario_key, scenario_key)}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out = FIGURES_DIR / f"cr_trajectory_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  CR 轨迹: {out}")


def plot_boxplot(all_stats: dict, scenario_key: str):
    """绘制解质量箱线图。"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    colors = []
    for config_key in CONFIGS:
        stats = all_stats.get(f"{scenario_key}_{config_key}", {})
        arr = stats.get("fitness_array", np.array([]))
        if len(arr) > 0:
            data.append(arr)
            labels.append(CONFIG_LABELS[config_key])
            colors.append(CONFIG_COLORS[config_key])

    if not data:
        plt.close(fig)
        return

    # matplotlib >=3.9 用 tick_labels；旧版用 labels
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Solution Quality — {SCENARIO_LABELS.get(scenario_key, scenario_key)}", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    out = FIGURES_DIR / f"boxplot_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  箱线图: {out}")


def plot_time_breakdown(all_stats: dict):
    """绘制时间开销堆叠柱状图。"""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (s_key, s_dir) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        config_keys = []
        dmde_vals = []
        llm_init_vals = []
        llm_cr_vals = []

        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                continue
            config_keys.append(c_key)
            dmde_vals.append(stats.get("dmde_time_mean", 0))
            llm_init_vals.append(stats.get("llm_init_time_mean", 0))
            llm_cr_vals.append(stats.get("llm_cr_time_mean", 0))

        x = np.arange(len(config_keys))
        labels = [CONFIG_LABELS[k] for k in config_keys]

        ax.bar(x, dmde_vals, label="DMDE", color="#1f77b4", alpha=0.8)
        ax.bar(x, llm_init_vals, bottom=dmde_vals, label="LLM Init", color="#ff7f0e", alpha=0.8)
        bottom2 = [d + li for d, li in zip(dmde_vals, llm_init_vals)]
        ax.bar(x, llm_cr_vals, bottom=bottom2, label="LLM CR", color="#2ca02c", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Time (s)", fontsize=11)
        ax.set_title(SCENARIO_LABELS.get(s_key, s_key), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Time Breakdown by Configuration", fontsize=14, y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "time_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  时间分解: {out}")


# ── LaTeX 表格 ───────────────────────────────────────────

def generate_latex_table1(all_stats: dict) -> str:
    """解质量对比表（按场景分组，每组自动加粗最优 Best/Mean/Median）。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{消融实验结果：解质量对比（最优值加粗）}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"场景 & 配置 & Best & Mean $\pm$ Std & Median \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        # 收集本场景的所有有效 stats，用于找最优
        group = []
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if stats:
                group.append((c_key, stats))
        if not group:
            continue
        best_vals = [s["best"] for _, s in group]
        mean_vals = [s["mean"] for _, s in group]
        median_vals = [s["median"] for _, s in group]
        best_min = min(best_vals)
        mean_min = min(mean_vals)
        median_min = min(median_vals)

        first_row = True
        for c_key, stats in group:
            s_label = SCENARIO_LABELS[s_key] if first_row else ""
            c_label = CONFIG_LABELS[c_key]
            best_s = f"{stats['best']:.2f}"
            mean_s = f"{stats['mean']:.2f} $\\pm$ {stats['std']:.2f}"
            median_s = f"{stats['median']:.2f}"
            # 最优值加粗（仅当 group size > 1）
            if len(group) > 1:
                if stats["best"] == best_min:
                    best_s = r"\textbf{" + best_s + r"}"
                if stats["mean"] == mean_min:
                    mean_s = mean_s.replace(f"{stats['mean']:.2f}",
                                            r"\textbf{" + f"{stats['mean']:.2f}" + r"}", 1)
                if stats["median"] == median_min:
                    median_s = r"\textbf{" + median_s + r"}"
            if first_row:
                lines.append(f"{s_label} & {c_label} & {best_s} & {mean_s} & {median_s} \\\\")
                first_row = False
            else:
                lines.append(f" & {c_label} & {best_s} & {mean_s} & {median_s} \\\\")
        lines.append(r"\midrule")
    # 移除最后多出的 \midrule
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_table2(all_stats: dict) -> str:
    """Mann-Whitney U 显著性检验表（A3 vs 各基线）。"""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{统计显著性检验（Mann-Whitney U 秩和检验，A3 vs 基线）}",
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
    lines += [r"\bottomrule", r"\end{tabular}", r"",
              r"\footnotesize{*$p < 0.05$，**$p < 0.01$，n.s. 不显著；样本量 $n<5$ 时未检验}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_table3(all_stats: dict) -> str:
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{计算时间对比（分口径）}", r"\label{tab:time_breakdown}",
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"场景 & 配置 & 总耗时(s) & DMDE(s) & LLM总耗时(s) & LLM Init(s) & LLM CR(s) \\", r"\midrule",
    ]
    for s_key in SCENARIOS:
        first_row = True
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats: continue
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


# ── Markdown 表格 ─────────────────────────────────────────

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
    """生成 LLM 决策摘要 Markdown。"""
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


# ── 主入口 ────────────────────────────────────────────────

def main():
    all_stats = {}

    for s_key, s_dir in SCENARIOS.items():
        scenario_dir = SCRIPT_DIR / s_dir
        for c_key, c_dir in CONFIGS.items():
            results = load_results(scenario_dir, c_dir)
            all_stats[f"{s_key}_{c_key}"] = compute_stats(results)

    # ── Markdown 汇总表 ──
    md_table = generate_markdown_table(all_stats)
    print("\n## 消融实验结果\n")
    print(md_table)

    # ── LLM 决策摘要 ──
    llm_summary = generate_llm_decision_summary(all_stats)
    print(f"\n{llm_summary}")

    # ── 保存 Markdown ──
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
                if not stats: continue
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
            plot_convergence_comparison(all_stats, s_key)
            plot_cr_comparison(all_stats, s_key)
            plot_boxplot(all_stats, s_key)
        plot_time_breakdown(all_stats)
    else:
        print("\n(matplotlib 不可用，跳过图表生成)")

    # ── 保存 LLM 决策日志 ──
    for s_key in SCENARIOS:
        for c_key in ["A1", "A2", "A3"]:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            raw = stats.get("_raw", [])
            for r in raw:
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
    if HAS_SCIPY:
        print(f"  (scipy 可用，Mann-Whitney U 检验已计算)")
    else:
        print(f"  (scipy 不可用，统计检验跳过；表格显示 ---)")


if __name__ == "__main__":
    main()