# -*- coding: utf-8 -*-
"""LLM-Enhanced DMDE 算法模块 —— 模块化、可插拔、可消融实验友好的 LLM 增强 DMDE。

架构：
    ┌─────────────────────────┐
    │       LLM Modules       │
    │  PopInit | OpSel | CR   │  ← 可插拔、可消融
    └───────────┬─────────────┘
                │ inject(state)
                ▼
    ┌─────────────────────────┐
    │    DMDE Core Loop       │
    │  Encode → Map → Mutate  │
    │  → InvMap → Eval → Sel  │
    └─────────────────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │  OptimizationTrajectory │  ← 完整记录
    └─────────────────────────┘

消融实验示例：
    # Vanilla DMDE
    cfg = LLMEnhancedDMDEConfig(modules={})
    # 仅 LLM 算子选择
    cfg = LLMEnhancedDMDEConfig(modules={"operator_selection": {"enabled": True, "interval": 50}})
    # 仅 LLM CR 控制
    cfg = LLMEnhancedDMDEConfig(modules={"cr_control": {"enabled": True, "interval": 10}})
    # 全部启用
    cfg = LLMEnhancedDMDEConfig(modules={
        "population_init": {"enabled": True},
        "operator_selection": {"enabled": True, "interval": 50},
        "cr_control": {"enabled": True, "interval": 10},
    })
"""

from .representation.encoder import PopulationEncoder, Individual, Gene
from .representation.mapper import phi, phi_batch
from .representation.inverse_mapper import inverse_phi
from .operators.crossover import dynamic_crossover_rate
from .operators.scale_factor import dynamic_scale_factor
from .operators.mutation import mutate_population
from .operators.extinction import gmr_rate, should_extinct, apply_extinction
from .solvers.llm_enhanced_dmde_solver import LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig
from .base.base_optimizer import BaseOptimizer, SolverResult
from .trajectory.optimization_trajectory import OptimizationTrajectory, TrajectoryEntry
from .llm.base_module import BaseLLMModule, ModuleState

__all__ = [
    # 核心数据结构
    "PopulationEncoder",
    "Individual",
    "Gene",
    # 映射
    "phi",
    "phi_batch",
    "inverse_phi",
    # 算子
    "dynamic_crossover_rate",
    "dynamic_scale_factor",
    "mutate_population",
    "gmr_rate",
    "should_extinct",
    "apply_extinction",
    # 求解器
    "LLMEnhancedDMDESolver",
    "LLMEnhancedDMDEConfig",
    "BaseOptimizer",
    "SolverResult",
    # 轨迹
    "OptimizationTrajectory",
    "TrajectoryEntry",
    # 模块接口
    "BaseLLMModule",
    "ModuleState",
]
