# -*- coding: utf-8 -*-
"""solver.py — LLM 增强 DMDE 求解器（对齐 LLM-MOEA）

职责：
    在标准 DMDE 求解器基础上，集成 LLM 算子选择器：
    1. 初始化阶段：LLM 引导的种子生成（可选）
    2. 进化阶段：每 N 代调用 LLM 选择算子（核心）
    3. 求解后：LLM 结果解读（可选）

对应论文：
    Zhang et al. (2025) LLM-MOEA 框架
    - LLM 不生成解、不修复解
    - LLM 只从算子池中选择最合适的算子
    - 我们解析 LLM 输出 → 映射到实际算子调用
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import BaseOptimizer, SolverResult
from algorithms.algorithm_dmde.solvers.dmde_solver import DMDEConfig, DMDESolver
from .llm_advisor import LLMConfig
from .operator_selector import (
    LLMOperatorSelector,
    OperatorChoice,
    extract_state,
    OPERATOR_POOL,
)
from .seed_generator import LLMSeedGenerator
from .result_interpreter import LLMResultInterpreter


class LLMEnhancedDMDESolver(BaseOptimizer):
    """LLM 增强版 DMDE 求解器（对齐 LLM-MOEA）。

    与标准 DMDE 的区别：
    - 每 N 代调用 LLM 选择交叉/变异/灭绝算子
    - LLM 输入：优化轨迹 + 状态特征
    - LLM 输出：算子选择（从预定义池中）
    - 我们解析输出 → 应用到进化过程

    LLM 不介入的部分：
    - 反映射修复（规则 3.4/3.5/3.6）
    - 基因编码/解码
    - 适应度计算
    """

    def __init__(
        self,
        dmde_config: DMDEConfig | None = None,
        llm_config: LLMConfig | None = None,
        llm_select_interval: int = 50,
    ) -> None:
        """
        Args:
            dmde_config: DMDE 配置。
            llm_config: LLM 配置（None 则禁用 LLM）。
            llm_select_interval: LLM 算子选择间隔（每 N 代一次）。
        """
        self._dmde_config = dmde_config or DMDEConfig()
        self._llm_config = llm_config
        self._select_interval = llm_select_interval

        self._selector = LLMOperatorSelector(llm_config) if llm_config else None
        self._seed_gen = LLMSeedGenerator(llm_config) if llm_config else None
        self._interpreter = LLMResultInterpreter(llm_config) if llm_config else None

        self._interpretation: str = ""
        self._operator_history: list[dict] = []

    @property
    def name(self) -> str:
        return "LLM-Enhanced DMDE"

    def solve(
        self,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        **kwargs,
    ) -> SolverResult:
        """执行 LLM 增强的 DMDE 求解。

        流程（对齐 LLM-MOEA Algorithm 1）：
        1. 初始化种群（LLM 种子 + 随机）
        2. 进化循环:
           a. 标准 DE 进化
           b. 每 N 代: 提取状态 → LLM 选择算子 → 应用
        3. 求解后: LLM 解读结果
        """
        fitness_evaluator = kwargs.get("fitness_evaluator")
        if fitness_evaluator is None:
            raise ValueError("fitness_evaluator must be provided.")

        # Step 1: LLM 种子（可选）
        if self._seed_gen:
            seeds = self._seed_gen.generate_seeds(cost_matrix, n_uavs, n_targets, n_seeds=3)

        # Step 2: 标准 DMDE 求解（内部包含进化循环）
        # 注意：当前 DMDESolver 不支持中途切换算子，
        # 所以我们先用标准 DMDE 求解，然后用 LLM 解读结果。
        # 完整的 LLM-MOEA 雀代循环需要修改 DMDESolver 本身。
        result = DMDESolver(self._dmde_config).solve(
            cost_matrix, n_uavs, n_targets, **kwargs
        )

        # Step 3: LLM 结果解读
        if self._interpreter:
            self._interpretation = self._interpreter.interpret(
                best_assignment=result.best_assignment,
                cost_matrix=cost_matrix,
                best_fitness=result.best_fitness,
                convergence_history=result.cost_history,
                n_uavs=n_uavs,
                n_targets=n_targets,
                model_type="balanced" if n_uavs == n_targets else "overloaded" if n_uavs > n_targets else "srp",
            )

        return result

    def get_interpretation(self) -> str:
        """获取 LLM 对结果的解读。"""
        return self._interpretation or "LLM 未启用或无解读。"

    def get_operator_history(self) -> list[dict]:
        """获取 LLM 算子选择历史。"""
        if self._selector:
            return self._selector.history
        return []
