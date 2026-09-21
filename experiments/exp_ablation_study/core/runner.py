# -*- coding: utf-8 -*-
"""runner.py — 单次实验运行封装（A0 / A1）

集成 RunRecorder 记录完整过程数据。
"""

import time
from pathlib import Path

import numpy as np

from .recorder import RunRecorder


def run_dmde_baseline(seed: int, cost_matrix: np.ndarray, evaluator: object,
                      n_uavs: int, n_targets: int,
                      pop_size: int, max_generations: int,
                      zeta: int, delta: float) -> dict:
    """运行 Vanilla DMDE 基线（A0），记录每代 CR/F/fitness。"""
    from src.algorithms.algorithm_dmde.solvers.dmde_solver import DMDESolver, DMDEConfig
    from .solver_wrappers import RecordingDMDESolver

    cfg = DMDEConfig(
        pop_size=pop_size,
        max_generations=max_generations,
        zeta=zeta,
        delta=delta,
        seed=seed,
    )
    solver = DMDESolver(cfg)
    recorder = RunRecorder(seed)
    wrapped = RecordingDMDESolver(solver, recorder)

    t_start = time.time()
    result = wrapped.solve(cost_matrix, n_uavs, n_targets, fitness_evaluator=evaluator)
    t_total = time.time() - t_start

    recorder.set_times(total=t_total, dmde=t_total)
    return recorder.finalize()


def run_llm_dmde(seed: int, cost_matrix: np.ndarray, evaluator: object,
                 n_uavs: int, n_targets: int,
                 pop_size: int, max_generations: int,
                 zeta: int, delta: float,
                 llm_config_path: Path, modules_config: dict,
                 model: str | None = None,
                 fallback_model: str | None = None) -> dict:
    """运行 LLM-DMDE（A1），记录 LLM 决策详情 + 每代轨迹。"""
    from src.algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
        LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig,
    )
    from .solver_wrappers import RecordingLLMDESolver

    cfg = LLMEnhancedDMDEConfig(
        pop_size=pop_size,
        max_generations=max_generations,
        zeta=zeta,
        delta=delta,
        seed=seed,
        llm_config_path=str(llm_config_path),
        modules=modules_config,
    )

    # 模型覆盖：写入临时 YAML，优先级高于原始配置
    if model or fallback_model:
        import yaml, tempfile, os
        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_cfg = yaml.safe_load(f) or {}
        if model:
            llm_cfg["model"] = model
        if fallback_model:
            llm_cfg["fallback_model"] = fallback_model
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        )
        yaml.dump(llm_cfg, tmp, allow_unicode=True, default_flow_style=False)
        tmp.close()
        cfg.llm_config_path = tmp.name
        try:
            result = _run_llm_dmde_inner(cfg, seed, cost_matrix, evaluator,
                                         n_uavs, n_targets)
        finally:
            os.unlink(tmp.name)
        return result

    return _run_llm_dmde_inner(cfg, seed, cost_matrix, evaluator,
                               n_uavs, n_targets)


def _run_llm_dmde_inner(cfg, seed, cost_matrix, evaluator,
                        n_uavs, n_targets) -> dict:
    from src.algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
        LLMEnhancedDMDESolver,
    )
    from .solver_wrappers import RecordingLLMDESolver

    solver = LLMEnhancedDMDESolver(cfg)
    recorder = RunRecorder(seed)
    wrapped = RecordingLLMDESolver(solver, recorder)

    t_total_start = time.time()
    result = wrapped.solve(cost_matrix, n_uavs, n_targets, fitness_evaluator=evaluator)
    t_total = time.time() - t_total_start

    # 从 result.extra 提取 LLM 时间细分
    extra = result.extra or {}
    llm_time = extra.get("llm_time", 0.0)
    llm_call_count = extra.get("llm_call_count", 0)
    llm_decisions = recorder._llm_decisions

    llm_cr_time = sum(d.duration for d in llm_decisions if d.module == "search_controller")

    recorder.set_times(
        total=t_total,
        dmde=t_total - llm_time,
        llm=llm_time,
        llm_cr=llm_cr_time,
    )
    # 覆盖 llm_call_count（从 solver 获取准确值）
    if llm_call_count:
        recorder.set_llm_call_count(llm_call_count)
    return recorder.finalize()