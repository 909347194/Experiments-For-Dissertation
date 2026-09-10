# -*- coding: utf-8 -*-
"""plot_convergence.py — 收敛曲线绘制。"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import SolverResult
from ._common import COLORS, _sanitize_filename, _PlotBase


class ConvergencePlotter(_PlotBase):
    """收敛曲线绘制器。"""

    def plot(
        self,
        results: list[SolverResult],
        scenario_name: str = "",
        show_mean: bool = True,
        show_std: bool = True,
        log_scale: bool = False,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figsize)

        histories = [r.cost_history for r in results]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        generations = np.arange(min_len)

        for i, history in enumerate(histories):
            ax.plot(generations, history, alpha=0.3, linewidth=0.8,
                    color=COLORS['balanced'], label=f'Run {i+1}' if i < 5 else "")

        if show_mean or show_std:
            arr = np.array(histories)
            mean_h = np.mean(arr, axis=0)
            std_h = np.std(arr, axis=0)
            if show_mean:
                ax.plot(generations, mean_h, color=COLORS['best'], linewidth=2, label='平均收敛曲线')
            if show_std:
                ax.fill_between(generations, mean_h - std_h, mean_h + std_h,
                                alpha=0.2, color=COLORS['best'], label='±1 标准差')

        best_idx = int(np.argmin([r.best_fitness for r in results]))
        ax.plot(generations, histories[best_idx], color=COLORS['best'],
                linewidth=2.5, linestyle='--', label=f'最优运行 (Run {best_idx+1})')

        ax.set_xlabel('迭代代数', fontsize=12)
        ax.set_ylabel('适应度值', fontsize=12)
        title = 'DMDE 算法收敛曲线'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        if log_scale:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='-', color=COLORS['grid'])
        ax.legend(loc='upper right', fontsize=10)
        ax.set_facecolor(COLORS['background'])

        stats_text = (
            f'运行次数: {len(results)}\n'
            f'最优值: {results[best_idx].best_fitness:.2f}\n'
            f'最终代数: {min_len}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        filename = f"convergence_{_sanitize_filename(scenario_name)}.png"
        return self._save(fig, filename)
