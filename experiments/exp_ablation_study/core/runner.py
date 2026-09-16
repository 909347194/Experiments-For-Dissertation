# -*- coding: utf-8 -*-
"""runner.py — 单次实验运行封装（A0 / A1-A3）"""

import time
from pathlib import Path

import numpy as np


def run_dmde_baseline(seed: int, cost_matrix: np.ndarray, evaluator: object,
                      n_uavs: int, n_targets: int,
                      pop_size: int, max_generations: int,
                      zeta: int, delta: float) -> dict:
    """运行 Vanilla DMDE 基线（A0）。

    Returns:
        结果字典，含 fitness、时间分口径等字段。
    """
    from src.algorithms.algorithm_dmde.solvers.dmde_solver import DMDESolver, DMDEConfig

    cfg = DMDEConfig(
        pop_size=pop_size,
        max_generations=max_generations,
        zeta=zeta,
        delta=delta,
        seed=seed,
    )
    solver = DMDESolver(cfg)

    t_start = time.time()
    result = solver.solve(cost_matrix, n_uavs, n_targets, fitness_evaluator=evaluator)
    t_total = time.time() - t_start

    return {
        "seed": seed,
        "best_fitness": result.best_fitness,
        "total_time": round(t_total, 2),
        "dmde_time": round(t_total, 2),
        "llm_time": 0.0,
        "llm_init_time": 0.0,
        "llm_cr_time": 0.0,
        "llm_call_count": 0,
    }


def run_llm_dmde(seed: int, cost_matrix: np.ndarray, evaluator: object,
                 n_uavs: int, n_targets: int,
                 pop_size: int, max_generations: int,
                 zeta: int, delta: float,
                 llm_config_path: Path, modules_config: dict) -> dict:
    """运行 LLM-DMDE（A1/A2/A3 通用）。

    Args:
        llm_config_path: LLM YAML 配置路径
        modules_config: modules 字典

    Returns:
        结果字典，含 fitness、时间分口径、LLM 决策记录等。
    """
    from src.algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
        LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig,
    )

    cfg = LLMEnhancedDMDEConfig(
        pop_size=pop_size,
        max_generations=max_generations,
        zeta=zeta,
        delta=delta,
        seed=seed,
        llm_config_path=str(llm_config_path),
        modules=modules_config,
    )
    solver = LLMEnhancedDMDESolver(cfg)

    t_total_start = time.time()
    result = solver.solve(cost_matrix, n_uavs, n_targets, fitness_evaluator=evaluator)
    t_total = time.time() - t_total_start

    # 从 result.extra 提取 LLM 时间
    extra = result.extra or {}
    llm_time = extra.get("llm_time", 0.0)
    llm_calls = extra.get("llm_call_count", 0)

    # 从 LLM 决策记录中细分 init / cr 时间
    llm_decisions = extra.get("llm_decisions", [])
    llm_init_time = 0.0
    llm_cr_time = 0.0
    for d in llm_decisions:
        dur = d.get("llm_call_duration", 0.0)
        module = d.get("llm_module", "")
        if module == "population_init":
            llm_init_time += dur
        elif module == "search_controller":
            llm_cr_time += dur

    return {
        "seed": seed,
        "best_fitness": result.best_fitness,
        "total_time": round(t_total, 2),
        "dmde_time": round(t_total - llm_time, 2),
        "llm_time": round(llm_time, 2),
        "llm_init_time": round(llm_init_time, 2),
        "llm_cr_time": round(llm_cr_time, 2),
        "llm_call_count": llm_calls,
    }