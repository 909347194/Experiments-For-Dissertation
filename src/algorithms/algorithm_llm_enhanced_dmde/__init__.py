# -*- coding: utf-8 -*-
"""LLM-Enhanced DMDE 算法模块 —— 大语言模型增强的离散映射差分进化算法。

在 DMDE（第 3 章）基础上，引入 LLM 驱动的自适应算子选择机制：
- 基因表征层（representation）：编码、映射、反映射、修复规则（与 DMDE 一致）
- 差分进化算子（operators）：动态交叉率、缩放因子、混合变异、灭绝（与 DMDE 一致）
- 搜索特征层（features）：种群多样性、收敛速度、约束满足度、轨迹收集
- LLM 决策层（llm）：提示构建、LLM 调用、响应解析
- 求解器层（solvers）：LLM 增强 DMDE 主求解器

核心创新：
    每隔 p 代，提取搜索状态特征 + 优化轨迹，构建提示发给 LLM，
    由 LLM 决定算子策略（DE/rand/1 vs best/2）、交叉率调整、
    温度调整、是否触发灭绝等，实现搜索过程的智能自适应。

数据流：
    encoder (离散三元组) → mapper.phi (连续代价值)
    → DE 算子进化 → inverse_mapper.phi' (离散三元组)
    → evaluator (适应度) → 贪婪选择 → 下一代
    [每 p 代] → features → trajectory → prompt → LLM → decision → 调整算子参数

对应论文扩展：
    在赵明 (2016) DMDE 基础上，引入 LLM 辅助决策的自适应进化策略。
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
    "LLMEnhancedDMDESolver",
    "LLMEnhancedDMDEConfig",
    "BaseOptimizer",
    "SolverResult",
]
