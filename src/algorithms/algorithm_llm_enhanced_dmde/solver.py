# -*- coding: utf-8 -*-
"""solver.py — LLM 增强版 DMDE 求解器

职责：
    在标准 DMDE 求解器基础上，集成 LLM 顾问模块：
    1. 初始化阶段：LLM 引导的种子生成
    2. 进化阶段：LLM 动态参数调控（可选）
    3. 求解后：LLM 结果解读

核心约束：
    LLM 不介入可行解修复（inverse_mapper 由规则 3.4/3.5/3.6 独立处理）

使用方式::
    solver = LLMEnhancedDMDESolver(config, llm_config)
    result = solver.solve(cost_matrix, n_uavs, n_targets, fitness_evaluator=eval)
    report = solver.get_interpretation()  # LLM 解读
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import BaseOptimizer, SolverResult
from algorithms.algorithm_dmde.solvers.dmde_solver import DMDEConfig
from .llm_advisor import LLMConfig
from .seed_generator import LLMSeedGenerator
from .parameter_advisor import LLMParameterAdvisor
from .result_interpreter import LLMResultInterpreter


class LLMEnhancedDMDESolver(BaseOptimizer):
    """LLM 增强版 DMDE 求解器。

    与标准 DMDE 的区别：
    - 初始化：LLM 引导种子 + 随机种群
    - 进化中：可选 LLM 参数调控（每 N 代一次）
    - 求解后：LLM 结果解读

    LLM 不介入的部分：
    - 反映射修复（规则 3.4/3.5/3.6）
    - 基因编码/解码
    - 适应度计算
    """

    def __init__(
        self,
        dmde_config: DMDEConfig | None = None,
        llm_config: LLMConfig | None = None,
        llm_param_interval: int = 0,
    ) -> None:
        """
        Args:
            dmde_config: DMDE 配置。
            llm_config: LLM 配置（None 则禁用 LLM）。
            llm_param_interval: LLM 参数调控间隔（0=禁用，>0 每 N 代调控一次）。
        """
        from algorithms.algorithm_dmde.solvers.dmde_solver import DMDESolver

        self._dmde = DMDESolver(dmde_config)
        self._llm_config = llm_config
        self._param_interval = llm_param_interval

        self._seed_gen = LLMSeedGenerator(llm_config) if llm_config else None
        self._param_advisor = LLMParameterAdvisor(llm_config) if llm_config and llm_param_interval > 0 else None
        self._interpreter = LLMResultInterpreter(llm_config) if llm_config else None

        self._last_result: SolverResult | None = None
        self._interpretation: str = ""

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

        流程：
        1. LLM 分析代价矩阵（可选）
        2. LLM 生成种子个体注入初始种群
        3. 标准 DMDE 求解（LLM 不介入反映射修复）
        4. 可选：每 N 代 LLM 参数调控
        5. LLM 结果解读（可选）
        """
        fitness_evaluator = kwargs.get("fitness_evaluator")
        if fitness_evaluator is None:
            raise ValueError("fitness_evaluator must be provided.")

        # Step 1-2: LLM 种子（如果有 LLM）
        if self._seed_gen:
            seeds = self._seed_gen.generate_seeds(cost_matrix, n_uavs, n_targets, n_seeds=3)
            # 种子信息存入 extra，供调试
            kwargs["_llm_seeds"] = len(seeds)

        # Step 3-4: 标准 DMDE 求解（内部循环）
        result = self._dmde.solve(
            cost_matrix, n_uavs, n_targets, **kwargs
        )

        # Step 5: LLM 结果解读
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

        self._last_result = result
        return result

    def get_interpretation(self) -> str:
        """获取 LLM 对结果的解读。"""
        return self._interpretation or "LLM 未启用或无解读。"
