# -*- coding: utf-8 -*-
"""搜索特征提取层 —— 为 LLM 决策提供搜索状态信息。

负责从优化过程中提取各类特征，供 LLM 理解当前搜索状态：
- population_features.py: 种群多样性、基因方差
- convergence_features.py: 收敛速度、停滞检测
- constraint_features.py: 可行解比例、违反分布
"""

from .population_features import (
    compute_diversity, compute_gene_variance, compute_diversity_quantiles,
)
from .convergence_features import compute_convergence_speed, detect_stagnation
from .constraint_features import compute_feasible_ratio, compute_violation_distribution
from .trigger import evaluate_trigger, stagnation_tier, DEFAULT_STAG_TIERS

__all__ = [
    "compute_diversity",
    "compute_gene_variance",
    "compute_diversity_quantiles",
    "compute_convergence_speed",
    "detect_stagnation",
    "compute_feasible_ratio",
    "compute_violation_distribution",
    "evaluate_trigger",
    "stagnation_tier",
    "DEFAULT_STAG_TIERS",
]
