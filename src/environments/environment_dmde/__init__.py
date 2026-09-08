# -*- coding: utf-8 -*-
"""DMDE 环境模块 —— 基于 DEM 的三维物理环境。

职责：
    管理物理环境数据：DEM 地形、雷达威胁、航程代价物理估算。
    任务模型（UAV/目标配置、代价矩阵、约束评估）在 models.model_dmde 中。

模块结构：
    DEMTerrain               DEM 高程地图加载与地形剖面提取
    RadarThreatField         雷达威胁半球模型与威胁场管理
    VerticalSectionCostEstimator  垂直切面航程代价估算器
    EnvironmentContext       物理环境运行上下文
    DMDETransitionManager    环境状态转移管理器（动态实验）

核心数据流：
    DEM 数据 → DEMTerrain → VerticalSectionCostEstimator → 航程代价
                                                ↑
                                    RadarThreatField (威胁叠加)

对应论文：
    赵明. 多无人机系统的协同目标分配和航迹规划方法研究[D].
    哈尔滨工业大学, 2016. 第 2 章 2.4 节。
"""

from .context import EnvironmentContext
from .cost_estimator import (
    VerticalSectionCostEstimator,
    CostEstimationResult,
)
from .dem_terrain import DEMTerrain, DEMProfile, DEMMeta
from .radar_threat import RadarThreat, RadarThreatField
from .transition import DMDETransitionManager, EnvironmentEvent

__all__ = [
    "EnvironmentContext",
    "DEMTerrain",
    "DEMMeta",
    "DEMProfile",
    "RadarThreat",
    "RadarThreatField",
    "VerticalSectionCostEstimator",
    "CostEstimationResult",
    "DMDETransitionManager",
    "EnvironmentEvent",
]
