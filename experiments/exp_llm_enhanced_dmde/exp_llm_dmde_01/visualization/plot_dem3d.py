# -*- coding: utf-8 -*-
"""plot_dem3d.py — DEM 三维可视化。"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ._common import COLORS, _sanitize_filename, _PlotBase


class Dem3DPlotter(_PlotBase):
    """DEM 三维分配方案可视化绘制器。"""

    def plot(
        self,
        dem_terrain: Any,
        uavs: list[Any],
        targets: list[Any],
        assignment: list[tuple[int, int]],
        scenario_name: str = "",
        cost_matrix: np.ndarray | None = None,
        view_elev: float = 45,
        view_azim: float = -60,
    ) -> Path:
        """在 DEM 地形上三维可视化分配方案。

        改进：
        - 高斯平滑消除尖锐噪点
        - 相对高程（减去最低点）避免 Z 轴从 3600m 起始
        - stride=1 但网格已降采样，面片大小适中
        - cmap 直接映射 + shade=True 增加立体感
        """
        from utils.utils_dmde.coord_transform import WGS84Transformer

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        elevation = dem_terrain.elevation.copy()
        bounds = dem_terrain.bounds
        left, bottom, right, top = bounds

        # NaN 填补
        nan_mask = np.isnan(elevation)
        fill_val = np.nanmin(elevation) if not np.all(nan_mask) else 0
        elevation[nan_mask] = fill_val

        # 高斯平滑 σ=3
        elev_smooth = self._gaussian_filter(elevation, sigma=3)

        # 降采样
        h, w = elev_smooth.shape
        target_grid = 120
        step = max(1, min(h, w) // target_grid)
        elev_ds = elev_smooth[::step, ::step]

        # 相对高程
        z_min = np.nanmin(elev_ds)
        elev_rel = elev_ds - z_min

        # 坐标
        rows, cols = elev_ds.shape
        xs_deg = np.linspace(left, right, cols)
        ys_deg = np.linspace(top, bottom, rows)
        xx_deg, yy_deg = np.meshgrid(xs_deg, ys_deg)

        ref_lon = (left + right) / 2
        ref_lat = (top + bottom) / 2
        transformer = WGS84Transformer.from_lonlat(ref_lon, ref_lat)
        xx_m, yy_m = transformer.to_xy_batch(xx_deg, yy_deg)

        # 绘制 DEM 表面
        ax.plot_surface(
            xx_m, yy_m, elev_rel,
            cmap='terrain', alpha=0.85,
            rstride=1, cstride=1,
            shade=True, antialiased=False,
            linewidth=0, edgecolor='none',
        )

        def _z(ev):
            return ev - z_min

        # UAV
        uav_xy, uav_z = [], []
        for u in uavs:
            x, y = transformer.to_xy(u.start_pos[0], u.start_pos[1])
            z = _z(u.start_pos[2])
            uav_xy.append((x, y)); uav_z.append(z)
            ax.scatter([x], [y], [z], c=COLORS['balanced'], s=180, marker='^',
                       edgecolors='white', linewidth=1.5, depthshade=False, zorder=10)
            ax.text(x, y, z + 150, f'U{u.id}', fontsize=8, fontweight='bold',
                    color=COLORS['balanced'], ha='center')

        # Target
        tgt_xy, tgt_z = [], []
        for t in targets:
            x, y = transformer.to_xy(t.position[0], t.position[1])
            z = _z(t.position[2])
            tgt_xy.append((x, y)); tgt_z.append(z)
            ax.scatter([x], [y], [z], c=COLORS['best'], s=180, marker='o',
                       edgecolors='white', linewidth=1.5, depthshade=False, zorder=10)
            ax.text(x, y, z + 150, f'T{t.id}', fontsize=8, fontweight='bold',
                    color=COLORS['best'], ha='center')

        # 分配连线
        for uid, tid in assignment:
            if uid < len(uavs) and tid < len(targets):
                ux, uy = uav_xy[uid]; uz = uav_z[uid]
                tx, ty = tgt_xy[tid]; tz = tgt_z[tid]
                ts = np.linspace(0, 1, 20)
                ax.plot(ux + (tx-ux)*ts, uy + (ty-uy)*ts,
                        uz + (tz-uz)*ts + np.sin(ts*np.pi) * max(abs(tz-uz), 100) * 0.3,
                        color='gray', linewidth=1.2, alpha=0.7, zorder=5)
                if cost_matrix is not None and uid < cost_matrix.shape[0] and tid < cost_matrix.shape[1]:
                    mid = 10
                    cv = cost_matrix[uid, tid]
                    ax.text(ux + (tx-ux)*0.5, uy + (ty-uy)*0.5,
                            uz + (tz-uz)*0.5 + 80, f'{cv/1000:.1f}km',
                            fontsize=6, color='#333', ha='center',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.7, edgecolor='none'))

        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_xlabel('东向 (m)', fontsize=11, labelpad=10)
        ax.set_ylabel('北向 (m)', fontsize=11, labelpad=10)
        ax.set_zlabel('相对高程 (m)', fontsize=11, labelpad=10)

        title = 'DEM 地形 + UAV-目标分配方案'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        total_cost = 0
        if cost_matrix is not None:
            total_cost = sum(cost_matrix[u, t] for u, t in assignment if u < cost_matrix.shape[0] and t < cost_matrix.shape[1])
        stats = f'UAV: {len(uavs)} | Target: {len(targets)} | 分配: {len(assignment)} | 总代价: {total_cost/1000:.1f}km'
        ax.text2D(0.02, 0.02, stats, transform=ax.transAxes, fontsize=9,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['balanced'], markersize=10, label='UAV'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['best'], markersize=10, label='Target'),
            Line2D([0], [0], color='gray', linewidth=1.5, label='分配连线'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        plt.tight_layout()

        filename = f"dem3d_{_sanitize_filename(scenario_name)}.png"
        return self._save(fig, filename)

    @staticmethod
    def _gaussian_filter(data: np.ndarray, sigma: float = 3) -> np.ndarray:
        """简易高斯平滑（纯 numpy）。"""
        size = int(sigma * 4) + 1
        x = np.arange(size) - size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=data)
        smoothed = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode='same'), axis=0, arr=smoothed)
        return smoothed
