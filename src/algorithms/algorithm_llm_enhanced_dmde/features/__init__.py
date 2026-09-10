# -*- coding: utf-8 -*-
"""搜索特征提取层 —— 为 LLM 决策提供搜索状态信息。

负责从优化过程中提取各类特征，供 LLM 理解当前搜索状态：
- population_features.py: 种群多样性、基因方差
- convergence_features.py: 收敛速度、停滞检测
- constraint_features.py: 可行解比例、违反分布
- trajectory_collector.py: 优化轨迹收集与管理
"""

from .population_features import compute_diversity, compute_gene_variance
from .convergence_features import compute_convergence_speed, detect_stagnation
from .constraint_features import compute_feasible_ratio, compute_violation_distribution
from .trajectory_collector import TrajectoryCollector

__all__ = [
    "compute_diversity",
    "compute_gene_variance",
    "compute_convergence_speed",
    "detect_stagnation",
    "compute_feasible_ratio",
    "compute_violation_distribution",
    "TrajectoryCollector",
]
