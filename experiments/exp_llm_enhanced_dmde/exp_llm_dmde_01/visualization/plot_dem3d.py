# -*- coding: utf-8 -*-
"""plot_dem3d.py — DEM 三维可视化（学术论文风格绿色立体 DEM 地形图）。"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects

from ._common import COLORS, _sanitize_filename, _PlotBase


class Dem3DPlotter(_PlotBase):
    """DEM 三维分配方案可视化绘制器。"""

    # 自定义绿色 terrain colormap（学术论文风格，低饱和度）
    _TERRAIN_COLORS = [
        (0x0B / 255, 0x54 / 255, 0x28 / 255),  # 低海拔: 深绿
        (0x16 / 255, 0x80 / 255, 0x3A / 255),  # 中低: 绿
        (0x35 / 255, 0xA8 / 255, 0x52 / 255),  # 中: 亮绿
        (0x70 / 255, 0xC8 / 255, 0x5B / 255),  # 中高: 浅绿
        (0xA9 / 255, 0xD6 / 255, 0x6B / 255),  # 高: 黄绿
        (0xD5 / 255, 0xE5 / 255, 0x8A / 255),  # 最高: 淡黄绿
    ]

    def plot(
        self,
        dem_terrain: Any,
        uavs: list[Any],
        targets: list[Any],
        assignment: list[tuple[int, int]],
        scenario_name: str = "",
        cost_matrix: np.ndarray | None = None,
        view_elev: float = 32,
        view_azim: float = -55,
        vertical_exaggeration: float = 4.0,
    ) -> Path:
        """在 DEM 地形上三维可视化分配方案。

        坐标说明：
        - 真实数据坐标：dem_terrain.elevation 中的高程为真实值（米），
          uavs/targets 的位置为 WGS84 经纬度 + 真实高程。
        - 可视化坐标：XY 转换为 km，Z 应用 vertical_exaggeration 后转换为 km，
          仅用于绘图展示，不影响真实数据。
        """
        from utils.utils_dmde.coord_transform import WGS84Transformer

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- 真实数据 ---
        elevation = dem_terrain.elevation.copy()
        bounds = dem_terrain.bounds
        left, bottom, right, top = bounds

        # NaN 填补（用 nanmin 替换）
        nan_mask = np.isnan(elevation)
        fill_val = np.nanmin(elevation) if not np.all(nan_mask) else 0
        elevation[nan_mask] = fill_val

        # 高斯平滑 σ=0.8（保留山脊和沟谷细节）
        elev_smooth = self._gaussian_filter(elevation, sigma=0.8)

        # 降采样到目标网格 150
        h, w = elev_smooth.shape
        target_grid = 150
        step = max(1, min(h, w) // target_grid)
        elev_ds = elev_smooth[::step, ::step]

        # 相对高程（真实值，单位：米）
        z_min = np.nanmin(elev_ds)
        elev_rel = elev_ds - z_min

        # --- 坐标转换（真实数据坐标 → 米） ---
        rows, cols = elev_ds.shape
        xs_deg = np.linspace(left, right, cols)
        ys_deg = np.linspace(top, bottom, rows)
        xx_deg, yy_deg = np.meshgrid(xs_deg, ys_deg)

        ref_lon = (left + right) / 2
        ref_lat = (top + bottom) / 2
        transformer = WGS84Transformer.from_lonlat(ref_lon, ref_lat)
        xx_m, yy_m = transformer.to_xy_batch(xx_deg, yy_deg)

        # --- 可视化坐标（仅用于绘图） ---
        x_plot = xx_m / 1000                          # XY → km
        y_plot = yy_m / 1000
        z_plot = elev_rel / 1000 * vertical_exaggeration  # Z → km，应用垂直夸张

        # 自定义绿色 terrain colormap
        terrain_cmap = LinearSegmentedColormap.from_list(
            'academic_green', self._TERRAIN_COLORS, N=256
        )

        # Hillshade 光照
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(z_plot, cmap=terrain_cmap, vert_exag=1,
                       blend_mode="soft", fraction=1.2)

        # 绘制 DEM 表面（facecolors 来自 hillshade，关闭网格线）
        ax.plot_surface(
            x_plot, y_plot, z_plot,
            facecolors=rgb, shade=False,
            rstride=1, cstride=1,
            antialiased=False,
            linewidth=0, edgecolor='none',
        )

        # --- UAV 和 Target（Z 坐标也应用 vertical_exaggeration，仅绘图用） ---
        def _z_vis(ev):
            """将真实高程转换为可视化 Z 坐标（km + 垂直夸张）。"""
            return (ev - z_min) / 1000 * vertical_exaggeration

        uav_xy, uav_z = [], []
        for u in uavs:
            # 真实坐标 → 米 → km（仅绘图）
            x, y = transformer.to_xy(u.start_pos[0], u.start_pos[1])
            x_km, y_km = x / 1000, y / 1000
            z_km = _z_vis(u.start_pos[2])
            uav_xy.append((x_km, y_km))
            uav_z.append(z_km)
            ax.scatter([x_km], [y_km], [z_km],
                       c=COLORS['balanced'], s=130, marker='^',
                       edgecolors='white', linewidth=1.5,
                       depthshade=False, zorder=10)
            ax.text(x_km, y_km, z_km + 0.15, f'U{u.id}',
                    fontsize=8, fontweight='bold',
                    color=COLORS['balanced'], ha='center')

        tgt_xy, tgt_z = [], []
        for t in targets:
            x, y = transformer.to_xy(t.position[0], t.position[1])
            x_km, y_km = x / 1000, y / 1000
            z_km = _z_vis(t.position[2])
            tgt_xy.append((x_km, y_km))
            tgt_z.append(z_km)
            ax.scatter([x_km], [y_km], [z_km],
                       c=COLORS['best'], s=130, marker='o',
                       edgecolors='white', linewidth=1.5,
                       depthshade=False, zorder=10)
            ax.text(x_km, y_km, z_km + 0.15, f'T{t.id}',
                    fontsize=8, fontweight='bold',
                    color=COLORS['best'], ha='center')

        # --- 分配连线（三维弧线，Z 坐标为 km） ---
        for uid, tid in assignment:
            if uid < len(uavs) and tid < len(targets):
                ux, uy = uav_xy[uid]
                uz = uav_z[uid]
                tx, ty = tgt_xy[tid]
                tz = tgt_z[tid]
                ts = np.linspace(0, 1, 20)
                # 中间弧度（Z 已是 km，弧度按 km 尺度调整）
                arc = np.sin(ts * np.pi) * max(abs(tz - uz), 0.5) * 0.3
                ax.plot(
                    ux + (tx - ux) * ts,
                    uy + (ty - uy) * ts,
                    uz + (tz - uz) * ts + arc,
                    color='#444444', linewidth=1.5, alpha=0.8, zorder=5,
                )
                # 代价标注（简洁小字，白色文字+黑色描边，无背景框）
                if (cost_matrix is not None
                        and uid < cost_matrix.shape[0]
                        and tid < cost_matrix.shape[1]):
                    cv = cost_matrix[uid, tid]
                    mid_x = ux + (tx - ux) * 0.5
                    mid_y = uy + (ty - uy) * 0.5
                    mid_z = uz + (tz - uz) * 0.5 + max(abs(tz - uz), 0.5) * 0.3 * np.sin(0.5 * np.pi)
                    ax.text(mid_x, mid_y, mid_z + 0.08,
                            f'{cv / 1000:.1f}km',
                            fontsize=6, color='white', ha='center',
                            fontweight='bold',
                            path_effects=[
                                patheffects.withStroke(
                                    linewidth=1.5, foreground='black')
                            ])

        # --- 视角与坐标轴 ---
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_xlabel('X / km', fontsize=11, labelpad=10)
        ax.set_ylabel('Y / km', fontsize=11, labelpad=10)
        ax.set_zlabel('Z / km', fontsize=11, labelpad=10)

        # 无标题（论文图通常在 caption 中写标题）

        # 简洁图例
        legend_elements = [
            Line2D([0], [0], marker='^', color='w',
                   markerfacecolor=COLORS['balanced'], markersize=10, label='UAV'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=COLORS['best'], markersize=10, label='Target'),
            Line2D([0], [0], color='#444444', linewidth=1.5, label='Assignment'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  framealpha=0.8, edgecolor='#cccccc')

        # 白色背景，干净布局
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        plt.tight_layout()

        filename = f"dem3d_{_sanitize_filename(scenario_name)}.png"
        return self._save(fig, filename)

    @staticmethod
    def _gaussian_filter(data: np.ndarray, sigma: float = 0.8) -> np.ndarray:
        """简易高斯平滑（纯 numpy）。"""
        size = int(sigma * 4) + 1
        if size < 3:
            size = 3
        x = np.arange(size) - size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=data)
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode='same'), axis=0, arr=smoothed)
        return smoothed
