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

from .constants import SCENARIOS, CONFIGS, CONFIG_LABELS, CONFIG_COLORS, SCENARIO_LABELS
from .stats import extract_convergence_curves, extract_cr_histories, compute_convergence_stats


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
        x = np.arange(conv["length"])
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
    for config_key in ["A0", "A1", "A3"]:
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
        config_keys, dmde_vals, llm_init_vals, llm_cr_vals = [], [], [], []
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
    out = figures_dir / "time_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  时间分解: {out}")