# -*- coding: utf-8 -*-
"""可视化：收敛曲线、CR 轨迹、箱线图、时间分解。"""

from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .constants import (
    SCENARIOS, CONFIGS, CONFIG_LABELS, CONFIG_COLORS,
    SCENARIO_LABELS, GMR_MODE_COLORS, GMR_MODE_LABELS,
)
from .stats import (
    extract_convergence_curves,
    extract_cr_histories,
    extract_f_histories,
    extract_f_override_histories,
    extract_gmr_histories,
    compute_convergence_stats,
    compute_gmr_stats,
    curve_gens,
)


def plot_convergence_comparison(all_stats: dict, scenario_key: str, figures_dir: Path):
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
        # 用真实代数做横轴：A0 逐代(0..1000)，LLM 配置每 10 代(0,10,...,1000)。
        # 若误用 np.arange(len) 会把 LLM 曲线横向压缩 10 倍。
        x = np.array(conv["gens"])
        mean = np.array(conv["mean"])
        std = np.array(conv["std"])
        color = CONFIG_COLORS[config_key]
        ax.plot(x, mean, label=CONFIG_LABELS[config_key], color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Convergence — {SCENARIO_LABELS.get(scenario_key, scenario_key)}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / f"convergence_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  收敛曲线: {out}")


def plot_cr_comparison(all_stats: dict, scenario_key: str, figures_dir: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for config_key in ["A0", "A1"]:
        stats = all_stats.get(f"{scenario_key}_{config_key}", {})
        raw = stats.get("_raw", [])
        if not raw:
            continue
        # cr_history[i] 与 generation_records[i]['gen'] 对齐；逐代/每 10 代粒度不同，
        # 必须用真实代数做横轴，否则 LLM 曲线会被横向压缩 10 倍。
        series = []
        for r in raw:
            cr = r.get("cr_history", [])
            if cr:
                series.append((curve_gens(r)[: len(cr)], list(cr)))
        if not series:
            continue
        min_len = min(len(c) for _, c in series)
        arr = np.array([c[:min_len] for _, c in series])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.array(series[0][0][:min_len])
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
    out = figures_dir / f"cr_trajectory_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  CR 轨迹: {out}")


def plot_boxplot(all_stats: dict, scenario_key: str, figures_dir: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels, colors = [], [], []
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
    out = figures_dir / f"boxplot_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  箱线图: {out}")


def plot_time_breakdown(all_stats: dict, figures_dir: Path):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (s_key, s_dir) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        config_keys, dmde_vals, llm_cr_vals = [], [], []
        for c_key in CONFIGS:
            stats = all_stats.get(f"{s_key}_{c_key}", {})
            if not stats:
                continue
            config_keys.append(c_key)
            dmde_vals.append(stats.get("dmde_time_mean", 0))
            llm_cr_vals.append(stats.get("llm_cr_time_mean", 0))
        x = np.arange(len(config_keys))
        labels = [CONFIG_LABELS[k] for k in config_keys]
        ax.bar(x, dmde_vals, label="DMDE", color="#1f77b4", alpha=0.8)
        ax.bar(x, llm_cr_vals, bottom=dmde_vals, label="LLM", color="#ff7f0e", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Time (s)", fontsize=11)
        ax.set_title(SCENARIO_LABELS.get(s_key, s_key), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Time Breakdown by Configuration", fontsize=14, y=1.02)
    fig.tight_layout()
    out = figures_dir / "time_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  时间分解: {out}")


def plot_f_comparison(all_stats: dict, scenario_key: str, figures_dir: Path):
    """F 值轨迹对比图：展示各配置的 F 值随代数变化。

    仅绘制 A1（含 LLM 覆写）和 A0（公式推导基线）。
    """
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for config_key in ["A0", "A1"]:
        stats = all_stats.get(f"{scenario_key}_{config_key}", {})
        raw = stats.get("_raw", [])
        if not raw:
            continue
        series = []
        for r in raw:
            recs = r.get("generation_records", [])
            if not recs:
                continue
            gens = [rec["gen"] for rec in recs]
            # 优先 f_override，回退 f_scale
            f_vals = [
                rec.get("f_override") if rec.get("f_override") is not None
                else rec.get("f_scale", 0.5)
                for rec in recs
            ]
            series.append((gens, f_vals))
        if not series:
            continue
        min_len = min(len(f) for _, f in series)
        arr = np.array([f[:min_len] for _, f in series])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.array(series[0][0][:min_len])
        color = CONFIG_COLORS[config_key]
        ax.plot(x, mean, label=CONFIG_LABELS[config_key], color=color, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Scale Factor (F)", fontsize=12)
    ax.set_title(f"F Trajectory — {SCENARIO_LABELS.get(scenario_key, scenario_key)}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2)
    fig.tight_layout()
    out = figures_dir / f"f_trajectory_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  F 轨迹: {out}")


def plot_gmr_mode_distribution(all_stats: dict, scenario_key: str, figures_dir: Path):
    """GMR 模式分布图：展示 LLM 在 search_controller 决策中选择的 GMR 模式分布。

    仅对 A1（含 search_controller）有意义。
    数据来源：llm_decisions（仅 search_controller 模块的决策），
    而非 generation_records（后者 99% 是默认 "auto"，会严重膨胀饼图）。
    """
    if not HAS_MPL:
        return
    stats = all_stats.get(f"{scenario_key}_A1", {})
    raw = stats.get("_raw", [])
    if not raw:
        return
    # 使用 compute_gmr_stats 从 llm_decisions 统计（仅 search_controller 决策点）
    gmr_stats = compute_gmr_stats(raw)
    if not gmr_stats:
        return
    mode_counts = gmr_stats.get("mode_counts", {})
    total = gmr_stats.get("total_decisions", 0)
    if total == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = []
    sizes = []
    colors = []
    for mode in sorted(mode_counts.keys()):
        labels.append(GMR_MODE_LABELS.get(mode, mode))
        sizes.append(mode_counts[mode])
        colors.append(GMR_MODE_COLORS.get(mode, "#999999"))
    _, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title(
        f"GMR Mode Distribution — {SCENARIO_LABELS.get(scenario_key, scenario_key)}\n"
        f"(A1, {len(raw)} runs, {total} LLM decisions)",
        fontsize=12,
    )
    fig.tight_layout()
    out = figures_dir / f"gmr_distribution_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  GMR 分布: {out}")


def plot_parameter_control_overview(all_stats: dict, scenario_key: str, figures_dir: Path):
    """参数控制总览：展示 A1 中 CR、F、GMR 的独立控制效果。

    三行子图：CR 轨迹、F 轨迹、GMR 模式时间线。
    """
    if not HAS_MPL:
        return
    stats = all_stats.get(f"{scenario_key}_A1", {})
    raw = stats.get("_raw", [])
    if not raw:
        return
    # 选取第一个 run 作为示例
    r = raw[0]
    recs = r.get("generation_records", [])
    if not recs:
        return
    gens = [rec["gen"] for rec in recs]
    cr_vals = [rec.get("cr", 0.5) for rec in recs]
    f_vals = [
        rec.get("f_override") if rec.get("f_override") is not None
        else rec.get("f_scale", 0.5)
        for rec in recs
    ]
    gmr_modes = [rec.get("gmr_mode", "auto") for rec in recs]
    # 将 GMR 模式转为数值
    gmr_num_map = {"auto": 0, "off": 1, "on": 2}
    gmr_nums = [gmr_num_map.get(m, 0) for m in gmr_modes]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # CR
    axes[0].plot(gens, cr_vals, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("CR", fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        f"Parameter Control Overview — {SCENARIO_LABELS.get(scenario_key, scenario_key)}\n"
        f"(A1, seed={r.get('seed', '?')})",
        fontsize=12,
    )
    axes[0].grid(True, alpha=0.3)
    # F
    axes[1].plot(gens, f_vals, color="#ff7f0e", linewidth=1.2)
    axes[1].set_ylabel("F (Scale Factor)", fontsize=11)
    axes[1].set_ylim(0, 2)
    axes[1].grid(True, alpha=0.3)
    # GMR
    # 用散点+阶梯线表示离散模式
    axes[2].step(gens, gmr_nums, where="post", color="#2ca02c", linewidth=1.2, alpha=0.7)
    # 标记模式切换点
    for i in range(1, len(gmr_modes)):
        if gmr_modes[i] != gmr_modes[i - 1]:
            axes[2].axvline(x=gens[i], color="red", alpha=0.3, linestyle="--", linewidth=0.8)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Auto", "Off", "On"], fontsize=9)
    axes[2].set_ylabel("GMR Mode", fontsize=11)
    axes[2].set_xlabel("Generation", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    out = figures_dir / f"param_control_overview_{scenario_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  参数控制总览: {out}")
