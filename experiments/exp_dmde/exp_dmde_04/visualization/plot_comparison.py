# -*- coding: utf-8 -*-
"""plot_comparison.py — 场景对比图 + 箱线图绘制。"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._common import COLORS, _PlotBase


class ComparisonPlotter(_PlotBase):
    """场景对比与指标箱线图绘制器。"""

    def plot_scenario_comparison(self, scenarios: list[dict[str, Any]]) -> Path:
        metrics_to_plot = ['best_fitness', 'mean_fitness', 'feasible_rate', 'mean_time']
        n = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]

        names = [s['name'] for s in scenarios]
        colors = [COLORS.get(s['model_type'], '#666666') for s in scenarios]
        labels = {
            'best_fitness': '最优适应度', 'mean_fitness': '平均适应度',
            'feasible_rate': '可行解比例', 'mean_time': '平均耗时(s)',
        }

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [getattr(s['metrics'], metric, 0) for s in scenarios]
            bars = ax.bar(range(len(scenarios)), values, color=colors, alpha=0.8, edgecolor='white')

            for bar, v in zip(bars, values):
                txt = f'{v:.0%}' if metric == 'feasible_rate' else (f'{v:.2f}s' if metric == 'mean_time' else f'{v:.1f}')
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.02,
                        txt, ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax.set_ylabel(labels.get(metric, metric), fontsize=12)
            ax.set_title(labels.get(metric, metric), fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
            ax.set_facecolor(COLORS['background'])
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        fig.suptitle('三种模型场景对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self._save(fig, "scenario_comparison.png")

    def plot_metrics_boxplot(self, scenarios: list[dict[str, Any]]) -> Path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = [s['name'] for s in scenarios]
        colors = [COLORS.get(s['model_type'], '#666666') for s in scenarios]

        # 适应度
        fitness_data = [[r.best_fitness for r in s['results']] for s in scenarios]
        bp1 = axes[0].boxplot(fitness_data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp1['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        axes[0].set_ylabel('适应度值', fontsize=12)
        axes[0].set_title('适应度分布', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        axes[0].tick_params(axis='x', rotation=15)

        # 耗时
        time_data = [[r.elapsed_seconds for r in s['results']] for s in scenarios]
        bp2 = axes[1].boxplot(time_data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp2['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        axes[1].set_ylabel('耗时(秒)', fontsize=12)
        axes[1].set_title('求解耗时分布', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        axes[1].tick_params(axis='x', rotation=15)

        # 约束违背
        violation_data = []
        for s in scenarios:
            vlist = []
            for r in s['results']:
                v = r.extra.get('total_violation', 0.0)
                vlist.append(v / r.best_fitness * 100 if r.best_fitness > 0 else 0.0)
            violation_data.append(vlist)
        bp3 = axes[2].boxplot(violation_data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp3['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        axes[2].set_ylabel('约束违背率(%)', fontsize=12)
        axes[2].set_title('约束违背分布', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        axes[2].tick_params(axis='x', rotation=15)

        fig.suptitle('多次运行指标统计', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self._save(fig, "metrics_boxplot.png")
