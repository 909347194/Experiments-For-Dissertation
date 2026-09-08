# -*- coding: utf-8 -*-
"""radar_threat.py — 雷达威胁半球计算

职责：
    本模块负责管理和计算雷达威胁区域。根据论文公式 (2-29)，
    雷达侦测范围在空间各方向上相同，以雷达最大侦测距离为半径
    的半球区域可表示为：

        Z = sqrt(r² - (x - x0)² - (y - y0)²) + z0

    其中 (x0, y0, z0) 为雷达中心位置，r 为侦测半径。

对应论文：
    第 2 章 2.4.1 节 —— 基于垂直切面的航程代价估算方法。
    雷达威胁区域是影响 UAV 航程代价估算的关键环境因素。

主要功能：
    1. 定义雷达威胁源的几何模型（半球体）。
    2. 判断给定点是否在雷达威胁范围内。
    3. 计算垂直切面上雷达威胁区与地形的叠加高程。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadarThreat:
    """单个雷达威胁源。

    Attributes:
        x0:  雷达中心 X 坐标（经度 / 东向）。
        y0:  雷达中心 Y 坐标（纬度 / 北向）。
        z0:  雷达中心 Z 坐标（高程，通常为地面高程）。
        radius: 侦测半径（球半径）。
        penalty: 进入威胁区的惩罚系数（用于代价计算）。
    """

    x0: float
    y0: float
    z0: float
    radius: float
    penalty: float = 10.0


# ---------------------------------------------------------------------------
# 核心类
# ---------------------------------------------------------------------------

class RadarThreatField:
    """雷达威胁场管理器。

    管理多个雷达威胁源，提供威胁判断和高程叠加功能。

    使用方式::

        radars = [
            RadarThreat(x0=91.1, y0=29.6, z0=3600, radius=15000),
            RadarThreat(x0=91.3, y0=29.5, z0=3700, radius=12000),
        ]
        field = RadarThreatField(radars)
        threat_z = field.threat_elevation_profile(xs, ys, base_elevations)
    """

    def __init__(self, radars: Sequence[RadarThreat] = ()) -> None:
        self._radars = list(radars)

    @property
    def radars(self) -> list[RadarThreat]:
        """所有雷达威胁源列表。"""
        return list(self._radars)

    def add(self, radar: RadarThreat) -> None:
        """添加一个雷达威胁源。"""
        self._radars.append(radar)

    # ---- 威胁判断 ----------------------------------------------------------

    def is_threatened(self, x: float, y: float, z: float) -> bool:
        """判断指定三维点是否在任一雷达威胁范围内。

        Args:
            x, y, z: 空间点坐标。

        Returns:
            True 表示该点在至少一个雷达的侦测范围内。
        """
        for r in self._radars:
            dist_sq = (x - r.x0) ** 2 + (y - r.y0) ** 2 + (z - r.z0) ** 2
            if dist_sq <= r.radius ** 2:
                return True
        return False

    def threat_level(self, x: float, y: float, z: float) -> float:
        """计算指定点的综合威胁等级。

        威胁等级为各雷达在该点威胁贡献的叠加。
        当点在雷达半球内时，贡献为 penalty * (1 - dist/radius)；
        当点在半球外时，贡献为 0。

        Returns:
            综合威胁等级（非负值）。
        """
        level = 0.0
        for r in self._radars:
            dist = np.sqrt((x - r.x0) ** 2 + (y - r.y0) ** 2 + (z - r.z0) ** 2)
            if dist < r.radius:
                level += r.penalty * (1.0 - dist / r.radius)
        return level

    # ---- 半球高程计算 -------------------------------------------------------

    def threat_elevation(
        self, x: float, y: float, radar: RadarThreat
    ) -> float:
        """计算单个雷达在 (x, y) 处的半球上表面高程。

        对应论文公式 (2-29)：
            Z = sqrt(r² - (x - x0)² - (y - y0)²) + z0

        若 (x, y) 超出雷达水平投影范围，返回 NaN。

        Args:
            x, y: 水平坐标。
            radar: 雷达威胁源。

        Returns:
            半球上表面高程值，越界返回 NaN。
        """
        dx = x - radar.x0
        dy = y - radar.y0
        r_sq = radar.radius ** 2
        d_sq = dx ** 2 + dy ** 2

        if d_sq > r_sq:
            return np.nan

        return float(np.sqrt(r_sq - d_sq) + radar.z0)

    def threat_elevations_batch(
        self, xs: np.ndarray, ys: np.ndarray, radar: RadarThreat
    ) -> np.ndarray:
        """批量计算单个雷达的半球上表面高程（矢量化）。

        Args:
            xs, ys: 水平坐标数组。

        Returns:
            高程数组，shape 与 xs 相同。越界位置为 NaN。
        """
        dx = xs - radar.x0
        dy = ys - radar.y0
        r_sq = radar.radius ** 2
        d_sq = dx ** 2 + dy ** 2

        inside = d_sq <= r_sq
        z = np.where(inside, np.sqrt(np.maximum(r_sq - d_sq, 0.0)) + radar.z0, np.nan)
        return z

    def combined_threat_elevation(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        base_elevations: np.ndarray,
    ) -> np.ndarray:
        """计算地形与雷达威胁叠加后的综合高程剖面。

        对于每个采样点，取 base_elevation 和所有雷达半球高程中的
        最大值，作为 UAV 需要规避的"等效地形高程"。

        对应论文中的做法：加入威胁区域后，估计航程的策略按照新的
        威胁与地形共同的高程值为基准进行调整。

        Args:
            xs:              水平 X 坐标数组。
            ys:              水平 Y 坐标数组。
            base_elevations: 对应位置的地形高程数组。

        Returns:
            综合高程数组（地形与威胁取最大值）。
        """
        combined = base_elevations.copy()

        for r in self._radars:
            threat_z = self.threat_elevations_batch(xs, ys, r)
            # 仅在威胁半球内部且高程高于地形时叠加
            valid = ~np.isnan(threat_z)
            combined = np.where(valid & (threat_z > combined), threat_z, combined)

        return combined

    # ---- 可视化辅助 ---------------------------------------------------------

    def plot_radars_on_ax(self, ax, color: str = "red", alpha: float = 0.3):
        """在 matplotlib Axes 上绘制雷达威胁区域的投影圆。

        Args:
            ax:    matplotlib Axes。
            color: 填充颜色。
            alpha: 透明度。

        Returns:
            ax 对象。
        """
        from matplotlib.patches import Circle

        for r in self._radars:
            circle = Circle(
                (r.x0, r.y0),
                r.radius,
                fill=True,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=1.5,
                label=f"Radar ({r.x0:.2f}, {r.y0:.2f})",
            )
            ax.add_patch(circle)

        return ax
