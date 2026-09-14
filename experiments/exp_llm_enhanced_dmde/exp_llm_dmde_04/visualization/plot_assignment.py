# -*- coding: utf-8 -*-
"""plot_assignment.py — 2D 分配方案可视化。"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._common import COLORS, _sanitize_filename, _PlotBase


class AssignmentPlotter(_PlotBase):
    """2D 分配方案可视化绘制器。"""

    def plot(
        self,
        uavs: list[Any],
        targets: list[Any],
        assignment: list[tuple[int, int]],
        scenario_name: str = "",
        cost_matrix: np.ndarray | None = None,
    ) -> Path:
        fig, ax = plt.subplots(figsize=self.figsize)

        uav_pos = np.array([u.start_pos[:2] for u in uavs])
        tgt_pos = np.array([t.position[:2] for t in targets])

        ax.scatter(uav_pos[:, 0], uav_pos[:, 1], c=COLORS['balanced'], s=200,
                   marker='^', label='UAV', zorder=5, edgecolors='white', linewidth=2)
        ax.scatter(tgt_pos[:, 0], tgt_pos[:, 1], c=COLORS['best'], s=200,
                   marker='o', label='目标', zorder=5, edgecolors='white', linewidth=2)

        for i, (x, y) in enumerate(uav_pos):
            ax.annotate(f'U{i}', (x, y), textcoords="offset points", xytext=(0, 15),
                        ha='center', fontsize=10, fontweight='bold', color=COLORS['balanced'])
        for i, (x, y) in enumerate(tgt_pos):
            ax.annotate(f'T{i}', (x, y), textcoords="offset points", xytext=(0, 15),
                        ha='center', fontsize=10, fontweight='bold', color=COLORS['best'])

        # SRP 检测
        uav_ids = [a[0] for a in assignment]
        is_srp = len(uav_ids) > len(set(uav_ids))

        if is_srp:
            routes: dict[int, list[int]] = {}
            for uid, tid in assignment:
                routes.setdefault(uid, []).append(tid)
            route_colors = plt.cm.Set1(np.linspace(0, 1, max(len(routes), 1)))
            for idx, (uid, tgt_list) in enumerate(routes.items()):
                c = route_colors[idx % len(route_colors)]
                if uid < len(uavs) and tgt_list:
                    up = uavs[uid].start_pos[:2]
                    fp = targets[tgt_list[0]].position[:2]
                    ax.annotate('', xy=fp, xytext=up, arrowprops=dict(arrowstyle='->', color=c, lw=2.5, connectionstyle='arc3,rad=0.1'))
                    for i in range(len(tgt_list) - 1):
                        p1 = targets[tgt_list[i]].position[:2]
                        p2 = targets[tgt_list[i+1]].position[:2]
                        ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle='->', color=c, lw=2.0, connectionstyle='arc3,rad=0.1', linestyle='--'))
        else:
            for uid, tid in assignment:
                if uid < len(uavs) and tid < len(targets):
                    up = uavs[uid].start_pos[:2]
                    tp = targets[tid].position[:2]
                    ax.annotate('', xy=tp, xytext=up, arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, connectionstyle='arc3,rad=0.1'))
                    if cost_matrix is not None and uid < cost_matrix.shape[0] and tid < cost_matrix.shape[1]:
                        mx, my = (up[0]+tp[0])/2, (up[1]+tp[1])/2
                        ax.annotate(f'({cost_matrix[uid, tid]:.0f})', (mx, my), fontsize=8, ha='center', va='center',
                                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        title = 'UAV-目标分配方案'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])

        total_cost = 0
        if cost_matrix is not None:
            total_cost = sum(cost_matrix[u, t] for u, t in assignment if u < cost_matrix.shape[0] and t < cost_matrix.shape[1])
        stats = f'分配数: {len(assignment)}\n总代价: {total_cost:.0f}' if total_cost > 0 else f'分配数: {len(assignment)}'
        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        filename = f"assignment_{_sanitize_filename(scenario_name)}.png"
        return self._save(fig, filename)
