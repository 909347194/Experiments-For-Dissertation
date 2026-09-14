# -*- coding: utf-8 -*-
"""plot_llm_decisions.py — LLM 决策过程可视化。

生成图表：
1. 决策时间线：每代的策略选择和 CR 值变化
2. 决策汇总：策略频率饼图 + CR 分布直方图 + 调用耗时箱线图
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ._common import COLORS, _sanitize_filename


class LLMDecisionPlotter:
    """LLM 决策可视化器。"""

    def __init__(self, output_dir: str | Path, dpi: int = 300, figsize: tuple = (12, 8)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    def plot_decision_timeline(self, llm_decisions: dict[int, list[dict]], scenario_name: str = "") -> Path:
        """绘制 LLM 决策时间线。

        上图：策略选择随代数变化（rand/1=0, best/2=1）
        下图：CR 值随代数变化
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        fig.suptitle(f"LLM Decision Timeline — {scenario_name}", fontsize=14)

        strategy_map = {"rand/1": 0, "best/2": 1, "default": 0.5}
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, max(1, len(llm_decisions)))))

        for run_idx, decisions in llm_decisions.items():
            if not decisions:
                continue
            gens = [d["generation"] for d in decisions]
            strategies = [strategy_map.get(d["decision"].get("strategy", "default"), 0.5) for d in decisions]
            crs = [d["decision"].get("cr", 0.5) for d in decisions]

            color = colors[run_idx % len(colors)]
            ax1.plot(gens, strategies, "o-", color=color, markersize=4, alpha=0.7, label=f"Run {run_idx}")
            ax2.plot(gens, crs, "o-", color=color, markersize=4, alpha=0.7)

        ax1.set_ylabel("Strategy")
        ax1.set_yticks([0, 0.5, 1])
        ax1.set_yticklabels(["rand/1", "default", "best/2"])
        ax1.legend(fontsize=8, ncol=3, loc="upper right")
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Crossover Rate (CR)")
        ax2.set_xlabel("Generation")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fname = f"llm_decision_timeline_{_sanitize_filename(scenario_name)}.png"
        out = self.output_dir / fname
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_decision_summary(self, llm_decisions: dict[int, list[dict]], scenario_name: str = "") -> Path:
        """绘制 LLM 决策汇总：策略饼图 + CR 直方图 + 耗时箱线图 + 每轮调用次数。"""
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        fig.suptitle(f"LLM Decision Summary — {scenario_name}", fontsize=14)

        all_strategies, all_crs, all_durations = [], [], []
        for decisions in llm_decisions.values():
            for d in decisions:
                dec = d.get("decision", {})
                all_strategies.append(dec.get("strategy", "default"))
                all_crs.append(dec.get("cr", 0.5))
                all_durations.append(d.get("duration", 0))

        # 策略频率饼图
        ax1 = fig.add_subplot(gs[0, 0])
        if all_strategies:
            from collections import Counter
            counts = Counter(all_strategies)
            labels, sizes = list(counts.keys()), list(counts.values())
            colors_pie = [COLORS.get(l, COLORS["default"]) for l in labels]
            ax1.pie(sizes, labels=labels, autopct="%1.0f%%", colors=colors_pie, startangle=90)
            ax1.set_title("Strategy Frequency")

        # CR 分布直方图
        ax2 = fig.add_subplot(gs[0, 1])
        if all_crs:
            ax2.hist(all_crs, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                     color=COLORS["llm_dmde"], edgecolor="white", alpha=0.8)
            ax2.set_xlabel("Crossover Rate (CR)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("CR Distribution")

        # 调用耗时箱线图
        ax3 = fig.add_subplot(gs[1, 0])
        if all_durations:
            ax3.boxplot(all_durations, vert=True)
            ax3.set_ylabel("Duration (seconds)")
            ax3.set_title(f"LLM Call Duration (n={len(all_durations)})")
            ax3.set_xticklabels(["LLM Calls"])

        # 每轮调用次数
        ax4 = fig.add_subplot(gs[1, 1])
        runs = sorted(llm_decisions.keys())
        call_counts = [len(llm_decisions[r]) for r in runs]
        if call_counts:
            ax4.bar(range(len(runs)), call_counts, color=COLORS["llm_dmde"], alpha=0.8)
            ax4.set_xlabel("Run Index")
            ax4.set_ylabel("LLM Calls")
            ax4.set_title("LLM Calls per Run")

        fig.tight_layout()
        fname = f"llm_decision_summary_{_sanitize_filename(scenario_name)}.png"
        out = self.output_dir / fname
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return out

    def plot_all(self, llm_decisions: dict[int, list[dict]], scenario_name: str = "") -> list[Path]:
        """生成所有 LLM 决策图表。"""
        saved = []
        p = self.plot_decision_timeline(llm_decisions, scenario_name)
        saved.append(p)
        print(f"  ✓ LLM 决策时间线: {p.name}")
        p = self.plot_decision_summary(llm_decisions, scenario_name)
        saved.append(p)
        print(f"  ✓ LLM 决策汇总: {p.name}")
        return saved
