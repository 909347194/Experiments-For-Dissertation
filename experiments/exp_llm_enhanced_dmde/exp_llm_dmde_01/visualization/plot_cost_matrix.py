# -*- coding: utf-8 -*-
"""plot_cost_matrix.py — 代价矩阵热力图绘制。"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._common import COLORS, _sanitize_filename, _PlotBase


class CostMatrixPlotter(_PlotBase):
    """代价矩阵热力图绘制器。"""

    def plot(
        self,
        cost_matrix: np.ndarray,
        scenario_name: str = "",
        uav_ids: list[int] | None = None,
        target_ids: list[int] | None = None,
        highlight_assignment: list[tuple[int, int]] | None = None,
    ) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('代价值', fontsize=12)

        n_rows, n_cols = cost_matrix.shape
        if uav_ids is None:
            uav_ids = list(range(n_rows))
        if target_ids is None:
            target_ids = list(range(n_cols))

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([f'T{tid}' for tid in target_ids], fontsize=10)
        ax.set_yticklabels([f'U{uid}' for uid in uav_ids], fontsize=10)

        for i in range(n_rows):
            for j in range(n_cols):
                v = cost_matrix[i, j]
                tc = 'white' if v > np.max(cost_matrix) * 0.6 else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center', color=tc, fontsize=8, fontweight='bold')

        if highlight_assignment:
            for ui, ti in highlight_assignment:
                if ui < n_rows and ti < n_cols:
                    rect = plt.Rectangle((ti - 0.5, ui - 0.5), 1, 1,
                                         fill=False, edgecolor='blue', linewidth=3)
                    ax.add_patch(rect)

        title = '代价矩阵热力图'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('目标编号', fontsize=12)
        ax.set_ylabel('UAV 编号', fontsize=12)

        filename = f"cost_matrix_{_sanitize_filename(scenario_name)}.png"
        return self._save(fig, filename)
