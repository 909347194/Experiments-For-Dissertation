# -*- coding: utf-8 -*-
"""DMDE 算法模块 —— 离散映射差分进化算法。

实现了论文第 3 章提出的 DMDE 算法，包括：
- 基因表征层（representation）：编码、映射、反映射、修复规则
- 差分进化算子（operators）：动态交叉率、缩放因子、混合变异、灭绝
- 求解器层（solvers）：DMDE 主求解器 + 对比基线算法

核心数据流：
    encoder (离散三元组) → mapper.phi (连续代价值)
    → DE 算子进化 → inverse_mapper.phi' (离散三元组)
    → evaluator (适应度) → 贪婪选择 → 下一代

对应论文：
    赵明. 多无人机系统的协同目标分配和航迹规划方法研究[D].
    哈尔滨工业大学, 2016. 第 3 章.
    Ming et al. Improved discrete mapping DE for multi-UAVs. IJMLC, 2017.
"""

from .representation.encoder import PopulationEncoder, Individual, Gene
from .representation.mapper import phi, phi_batch
from .representation.inverse_mapper import inverse_phi
from .operators.crossover import dynamic_crossover_rate
from .operators.scale_factor import dynamic_scale_factor
from .operators.mutation import mutate_population
from .operators.extinction import gmr_rate, should_extinct, apply_extinction
from .solvers.dmde_solver import DMDESolver, DMDEConfig
from .base.base_optimizer import BaseOptimizer, SolverResult

__all__ = [
    "PopulationEncoder",
    "Individual",
    "Gene",
    "phi",
    "phi_batch",
    "inverse_phi",
    "dynamic_crossover_rate",
    "dynamic_scale_factor",
    "mutate_population",
    "gmr_rate",
    "should_extinct",
    "apply_extinction",
    "DMDESolver",
    "DMDEConfig",
    "BaseOptimizer",
    "SolverResult",
]
