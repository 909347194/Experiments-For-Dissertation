# -*- coding: utf-8 -*-
"""LLM 增强 DMDE 算法模块。

对齐论文: Zhang et al. (2025) "LLM-MOEA"
核心思路: LLM 作为算子选择器，不生成解、不修复解。

子模块:
    llm_advisor         LLM API 封装
    operator_selector   LLM 算子选择器（核心，对齐 LLM-MOEA）
    seed_generator      LLM 引导初始种群
    result_interpreter  LLM 结果解读
    solver              LLM 增强 DMDE 求解器
"""

from .llm_advisor import LLMAdvisor, LLMConfig
from .operator_selector import (
    LLMOperatorSelector,
    OperatorChoice,
    OptimizationState,
    extract_state,
    OPERATOR_POOL,
)
from .seed_generator import LLMSeedGenerator
from .result_interpreter import LLMResultInterpreter
from .solver import LLMEnhancedDMDESolver

__all__ = [
    "LLMAdvisor",
    "LLMConfig",
    "LLMOperatorSelector",
    "OperatorChoice",
    "OptimizationState",
    "extract_state",
    "OPERATOR_POOL",
    "LLMSeedGenerator",
    "LLMResultInterpreter",
    "LLMEnhancedDMDESolver",
]
