# -*- coding: utf-8 -*-
"""context.py — 环境运行上下文（仅物理环境参数）。

职责：
    封装实验运行时的物理环境参数：DEM 地形、雷达威胁场、
    代价估算器配置。UAV/目标配置已移至 models.model_dmde.entities。

对应论文：
    第 2 章 2.4 节 —— 基于垂直切面的航程代价和协同约束。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dem_terrain import DEMTerrain
from .radar_threat import RadarThreatField
from .cost_estimator import VerticalSectionCostEstimator


@dataclass
class EnvironmentContext:
    """物理环境运行上下文。

    仅包含环境层的物理数据，不包含任务模型信息。

    Attributes:
        dem_terrain:   DEM 地形管理器。
        radar_field:   雷达威胁场管理器。
        estimator:     航程代价估算器。
        min_clearance: 最小离地高度 (mx)。
        max_clearance: 最大离地高度 (my)。
        num_samples:   剖面采样点数。
    """

    dem_terrain: DEMTerrain
    radar_field: RadarThreatField
    estimator: VerticalSectionCostEstimator
    min_clearance: float = 50.0
    max_clearance: float = 300.0
    num_samples: int = 200

    @classmethod
    def from_dem_file(
        cls,
        dem_path: str | Path,
        min_clearance: float = 50.0,
        max_clearance: float = 300.0,
        num_samples: int = 200,
    ) -> "EnvironmentContext":
        """从 DEM 文件创建环境上下文。

        Args:
            dem_path: DEM 文件路径。
            min_clearance: 最小离地高度。
            max_clearance: 最大离地高度。
            num_samples: 剖面采样点数。

        Returns:
            EnvironmentContext 实例。
        """
        dem = DEMTerrain.from_file(dem_path)
        radar = RadarThreatField()
        estimator = VerticalSectionCostEstimator(
            dem_terrain=dem,
            radar_field=radar,
            min_clearance=min_clearance,
            max_clearance=max_clearance,
            num_samples=num_samples,
        )
        return cls(
            dem_terrain=dem,
            radar_field=radar,
            estimator=estimator,
            min_clearance=min_clearance,
            max_clearance=max_clearance,
            num_samples=num_samples,
        )
