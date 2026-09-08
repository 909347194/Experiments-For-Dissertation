# -*- coding: utf-8 -*-
"""cost_estimator.py — 基于垂直切面的三维航程代价估算器

职责：
    本模块实现了论文第 2 章 2.4.1 节提出的"基于垂直切面的航程代价
    估算方法"。该方法是 DMDE 环境模块的核心，用于估算 UAV 到目标
    点之间的三维航程代价。

算法流程（对应论文 Step1 ~ Step5）：
    Step1: 取初始点 S 和目标点 T，以及两点的连线 L(S,T)。
    Step2: 在 L(S,T) 上取一点做垂直于 XOY 平面的垂线，过 S、T、M
           做平面 STM 即为垂直于 XOY 平面的切割面。
    Step3: 求出切割面与地形的交点，将切面映射成二维坐标面
           （L(S,T) 为 x 轴，Z 为 y 轴）。
    Step4: 在二维坐标系上，按照地形跟随和飞行高度策略生成估计航迹：
           - 初始飞行高度比初始点高 mx；
           - 当飞行器与地形距离 < mx 时，逐渐爬升；
           - 当距离在 [mx, my] 时，保持平飞；
           - 当距离 > my 时，逐渐下降。
    Step5: 计算估计航迹长度 L，航程代价 D(i,j) = L(S,T) * W(T)。

对应论文公式：
    公式 (2-28): D(i,j) = L(S,T) * W(T)
    公式 (2-29): 雷达半球 Z = sqrt(r² - (x-x0)² - (y-y0)²) + z0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dem_terrain import DEMTerrain
    from .radar_threat import RadarThreatField


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostEstimationResult:
    """单次航程代价估算结果。

    Attributes:
        distance:       估算航程长度（三维曲线长度）。
        cost:           航程代价 = distance * weight。
        profile_xs:     切面坐标系下的水平距离序列。
        profile_zs:     切面坐标系下的高程序列（地形+威胁叠加）。
        path_xs:        估计航迹的水平距离序列。
        path_zs:        估计航迹的高程序列。
        flight_altitude: UAV 相对地形的保持高度 (mx)。
    """

    distance: float
    cost: float
    profile_xs: np.ndarray
    profile_zs: np.ndarray
    path_xs: np.ndarray
    path_zs: np.ndarray
    flight_altitude: float


@dataclass(frozen=True)
class CostMatrixResult:
    """代价矩阵构建结果。

    Attributes:
        matrix: 代价矩阵，shape = (n_uavs, n_targets) 或更大（SRP 情况）。
        uav_ids: UAV 标识列表。
        target_ids: 目标标识列表。
        details: 每个 (uav, target) 对应的 CostEstimationResult 详情。
    """

    matrix: np.ndarray
    uav_ids: list[int]
    target_ids: list[int]
    details: dict[tuple[int, int], CostEstimationResult]


# ---------------------------------------------------------------------------
# 飞行策略常量
# ---------------------------------------------------------------------------

# 默认飞行高度参数（单位与 DEM 一致，通常为米）
DEFAULT_MIN_CLEARANCE = 50.0   # mx: 最小离地高度
DEFAULT_MAX_CLEARANCE = 300.0  # my: 最大离地高度
DEFAULT_CLIMB_RATE = 0.1       # 爬升/下降斜率（弧度）


# ---------------------------------------------------------------------------
# 核心估算器
# ---------------------------------------------------------------------------

class VerticalSectionCostEstimator:
    """基于垂直切面的三维航程代价估算器。

    对应论文算法：
        取初始点 S 和目标点 T 的连线做垂直水平面的切面，
        在切面上利用地形信息跟随和 UAV 飞行保持策略获取
        近似三维航程，作为航程代价的估计值。

    使用方式::

        estimator = VerticalSectionCostEstimator(
            dem_terrain=dem,
            radar_field=radar_field,
            min_clearance=50,
            max_clearance=300,
        )
        result = estimator.estimate(
            start=(91.0, 29.5, 3600),
            end=(91.3, 29.7, 3700),
            weight=1.5,
        )
    """

    def __init__(
        self,
        dem_terrain: "DEMTerrain",
        radar_field: "RadarThreatField | None" = None,
        min_clearance: float = DEFAULT_MIN_CLEARANCE,
        max_clearance: float = DEFAULT_MAX_CLEARANCE,
        climb_rate: float = DEFAULT_CLIMB_RATE,
        num_samples: int = 200,
    ) -> None:
        self._dem = dem_terrain
        self._radar = radar_field
        self._mx = min_clearance
        self._my = max_clearance
        self._climb_rate = climb_rate
        self._num_samples = num_samples

    # ---- 单次估算 ----------------------------------------------------------

    def estimate(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        weight: float = 1.0,
    ) -> CostEstimationResult:
        """估算从 start 到 end 的航程代价。

        对应论文 Step1 ~ Step5。

        Args:
            start: 起点 (x, y, z)。
            end:   终点 (x, y, z)。
            weight: 目标权重 W(T)，用于计算代价值 D(i,j) = L * W。

        Returns:
            CostEstimationResult 实例。
        """
        sx, sy, sz = start
        ex, ey, ez = end

        # Step1: 取初始点 S 和目标点 T，以及两点的连线 L(S,T)
        # Step2 & Step3: 在切面上提取地形剖面并映射到二维坐标系
        profile = self._dem.extract_profile(
            start=(sx, sy),
            end=(ex, ey),
            num_samples=self._num_samples,
        )

        # 切面坐标系：水平距离为 x 轴，高程为 y 轴
        xs_2d = profile.distances
        zs_terrain = profile.elevations

        # 处理 NaN 高程（用线性插值填补）
        zs_terrain = self._fill_nan(zs_terrain)

        # 叠加雷达威胁高程
        if self._radar is not None and len(self._radar.radars) > 0:
            xs_3d = profile.coords_xy[:, 0]
            ys_3d = profile.coords_xy[:, 1]
            zs_combined = self._radar.combined_threat_elevation(
                xs_3d, ys_3d, zs_terrain
            )
        else:
            zs_combined = zs_terrain

        # Step4: 在二维坐标系上按飞行策略生成估计航迹
        path_xs, path_zs = self._generate_flight_path(
            xs_2d, zs_combined, start_z=sz, end_z=ez
        )

        # Step5: 计算估计航迹的三维长度
        distance = self._compute_path_length(path_xs, path_zs)
        cost = distance * weight

        return CostEstimationResult(
            distance=distance,
            cost=cost,
            profile_xs=xs_2d,
            profile_zs=zs_combined,
            path_xs=path_xs,
            path_zs=path_zs,
            flight_altitude=self._mx,
        )

    # ---- 代价矩阵构建 -------------------------------------------------------

    def build_cost_matrix(
        self,
        uav_positions: list[tuple[float, float, float]],
        target_positions: list[tuple[float, float, float]],
        target_weights: list[float] | None = None,
    ) -> CostMatrixResult:
        """构建 UAV-Target 代价矩阵。

        对应论文公式 (2-30)（N=M 方阵情况）。

        Args:
            uav_positions:    UAV 起飞位置列表 [(x, y, z), ...]。
            target_positions: 目标位置列表 [(x, y, z), ...]。
            target_weights:   目标权重列表 [w1, w2, ...]，默认全为 1.0。

        Returns:
            CostMatrixResult 实例。
        """
        n_uavs = len(uav_positions)
        n_targets = len(target_positions)

        if target_weights is None:
            target_weights = [1.0] * n_targets

        matrix = np.zeros((n_uavs, n_targets))
        details: dict[tuple[int, int], CostEstimationResult] = {}

        for i, uav_pos in enumerate(uav_positions):
            for j, tgt_pos in enumerate(target_positions):
                result = self.estimate(
                    start=uav_pos,
                    end=tgt_pos,
                    weight=target_weights[j],
                )
                matrix[i, j] = result.cost
                details[(i, j)] = result

        return CostMatrixResult(
            matrix=matrix,
            uav_ids=list(range(n_uavs)),
            target_ids=list(range(n_targets)),
            details=details,
        )

    def build_srp_cost_matrix(
        self,
        uav_positions: list[tuple[float, float, float]],
        target_positions: list[tuple[float, float, float]],
        target_weights: list[float] | None = None,
    ) -> CostMatrixResult:
        """构建 SRP（群巡游）代价矩阵。

        对应论文公式 (2-32)（N<M 情况）。
        矩阵上半部分为 UAV-Target 代价，下半部分为 Target-Target 代价。

        Args:
            uav_positions:    UAV 起飞位置列表。
            target_positions: 目标位置列表。
            target_weights:   目标权重列表。

        Returns:
            CostMatrixResult 实例（矩阵 shape = (n_uavs + n_targets, n_targets)）。
        """
        n_uavs = len(uav_positions)
        n_targets = len(target_positions)

        if target_weights is None:
            target_weights = [1.0] * n_targets

        # 构建完整 SRP 矩阵
        matrix = np.zeros((n_uavs + n_targets, n_targets))
        details: dict[tuple[int, int], CostEstimationResult] = {}

        # 上半部分：UAV -> Target
        for i, uav_pos in enumerate(uav_positions):
            for j, tgt_pos in enumerate(target_positions):
                result = self.estimate(
                    start=uav_pos,
                    end=tgt_pos,
                    weight=target_weights[j],
                )
                matrix[i, j] = result.cost
                details[(i, j)] = result

        # 下半部分：Target -> Target（左对角线为 0）
        for i, tgt_i in enumerate(target_positions):
            for j, tgt_j in enumerate(target_positions):
                if i == j:
                    matrix[n_uavs + i, j] = 0.0
                else:
                    result = self.estimate(
                        start=tgt_i,
                        end=tgt_j,
                        weight=target_weights[j],
                    )
                    matrix[n_uavs + i, j] = result.cost
                    details[(n_uavs + i, j)] = result

        return CostMatrixResult(
            matrix=matrix,
            uav_ids=list(range(n_uavs)),
            target_ids=list(range(n_targets)),
            details=details,
        )

    # ---- 内部方法 -----------------------------------------------------------

    def _generate_flight_path(
        self,
        xs: np.ndarray,
        terrain_zs: np.ndarray,
        start_z: float,
        end_z: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按飞行策略在切面上生成估计航迹。

        对应论文 Step4：
        - 初始飞行高度为 start_z（UAV 起飞高程）。
        - 当 UAV 与地形距离 < mx 时，逐渐爬升。
        - 当距离在 [mx, my] 时，保持平飞。
        - 当距离 > my 时，逐渐下降。

        Args:
            xs: 切面水平距离序列。
            terrain_zs: 地形（+威胁）高程序列。
            start_z: UAV 起始高程。
            end_z: 目标点高程。

        Returns:
            (path_xs, path_zs) 估计航迹坐标。
        """
        n = len(xs)
        path_zs = np.zeros(n)
        path_zs[0] = start_z

        for i in range(1, n):
            clearance = path_zs[i - 1] - terrain_zs[i]

            if clearance < self._mx:
                # 爬升
                path_zs[i] = path_zs[i - 1] + self._climb_rate * (xs[i] - xs[i - 1])
            elif clearance > self._my:
                # 下降
                path_zs[i] = path_zs[i - 1] - self._climb_rate * (xs[i] - xs[i - 1])
            else:
                # 平飞
                path_zs[i] = path_zs[i - 1]

            # 确保不低于地形 + 最小离地高度
            min_z = terrain_zs[i] + self._mx
            if path_zs[i] < min_z:
                path_zs[i] = min_z

        # 最后一点平滑过渡到目标高程
        path_zs[-1] = end_z

        return xs.copy(), path_zs

    @staticmethod
    def _compute_path_length(xs: np.ndarray, zs: np.ndarray) -> float:
        """计算二维航迹的总长度。

        Args:
            xs: 水平距离序列。
            zs: 高程序列。

        Returns:
            航迹总长度。
        """
        dx = np.diff(xs)
        dz = np.diff(zs)
        return float(np.sum(np.sqrt(dx ** 2 + dz ** 2)))

    @staticmethod
    def _fill_nan(arr: np.ndarray) -> np.ndarray:
        """用线性插值填补数组中的 NaN 值。

        Args:
            arr: 输入数组。

        Returns:
            填补后的数组。
        """
        result = arr.copy()
        nans = np.isnan(result)
        if nans.all():
            return np.zeros_like(result)
        if not nans.any():
            return result

        valid_idx = np.where(~nans)[0]
        nan_idx = np.where(nans)[0]
        result[nan_idx] = np.interp(nan_idx, valid_idx, result[valid_idx])
        return result

    # ---- 可视化辅助 ---------------------------------------------------------

    def plot_profile(
        self,
        result: CostEstimationResult,
        ax=None,
        title: str = "Vertical Section Profile",
    ):
        """绘制垂直切面剖面图和估计航迹。

        Args:
            result: 估算结果。
            ax: matplotlib Axes，为 None 时自动创建。
            title: 图标题。

        Returns:
            (fig, ax) 元组。
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 5))
        else:
            fig = ax.figure

        # 绘制地形剖面
        ax.fill_between(
            result.profile_xs,
            result.profile_zs,
            alpha=0.3,
            color="saddlebrown",
            label="Terrain + Threat",
        )
        ax.plot(
            result.profile_xs,
            result.profile_zs,
            color="saddlebrown",
            linewidth=1.5,
        )

        # 绘制估计航迹
        ax.plot(
            result.path_xs,
            result.path_zs,
            color="blue",
            linewidth=2,
            linestyle="--",
            label=f"Estimated Flight Path (d={result.distance:.1f})",
        )

        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax
