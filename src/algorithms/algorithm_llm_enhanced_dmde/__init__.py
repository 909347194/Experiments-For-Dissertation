# -*- coding: utf-8 -*-
"""LLM 增强 DMDE 算法模块。

对齐论文: Zhang et al. (2025) "LLM-MOEA"
核心思路: LLM 作为高层搜索控制器，DMDE 作为底层执行引擎。
职责边界: LLM 决定"从哪搜、怎么搜"，DMDE 负责"具体搜"。

架构:
    llm/                LLM 交互层（客户端 + 可插拔模块）
    features/           状态特征提取（种群/收敛/约束）
    trajectory/         优化轨迹记录
    representation/     基因表征（DMDE 核心）
    operators/          差分进化算子（DMDE 核心）
    solvers/            求解器（编排 LLM + DMDE）
    base/               基类
"""

# 新架构导出
from .llm.llm_client import LLMClient
from .llm.base_module import BaseLLMModule, ModuleState
from .llm.modules.population_init import LLMPopulationInitModule
from .llm.modules.operator_selection import LLMOperatorSelectionModule
from .llm.modules.cr_control import LLMCRControlModule
from .solvers.llm_enhanced_dmde_solver import (
    LLMEnhancedDMDEConfig,
    LLMEnhancedDMDESolver,
)
from .features.population_features import compute_diversity, compute_gene_variance
from .features.convergence_features import compute_convergence_speed, detect_stagnation
from .features.constraint_features import compute_feasible_ratio, compute_violation_distribution
from .trajectory.optimization_trajectory import OptimizationTrajectory, TrajectoryEntry

__all__ = [
    # LLM 客户端
    "LLMClient",
    # LLM 模块基类
    "BaseLLMModule",
    "ModuleState",
    # LLM 可插拔模块
    "LLMPopulationInitModule",
    "LLMOperatorSelectionModule",
    "LLMCRControlModule",
    # 求解器
    "LLMEnhancedDMDEConfig",
    "LLMEnhancedDMDESolver",
    # 特征提取
    "compute_diversity",
    "compute_gene_variance",
    "compute_convergence_speed",
    "detect_stagnation",
    "compute_feasible_ratio",
    "compute_violation_distribution",
    # 轨迹
    "OptimizationTrajectory",
    "TrajectoryEntry",
]
