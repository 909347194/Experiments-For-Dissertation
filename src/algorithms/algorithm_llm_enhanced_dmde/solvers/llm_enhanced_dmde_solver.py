# -*- coding: utf-8 -*-
"""llm_enhanced_dmde_solver.py — 模块化 LLM 增强 DMDE 求解器

Architecture:
    ┌─────────────────────────────┐
    │       LLM Modules           │
    │  PopInit | SearchController  │  ← 可插拔、可消融
    └───────────┬─────────────────┘
                │ inject(state)
                ▼
    ┌─────────────────────────────┐
    │      DMDE Core Loop         │
    │  Encode → Map → Mutate      │
    │  → InvMap → Eval → Sel      │
    └─────────────────────────────┘
                │
                ▼
    ┌─────────────────────────────┐
    │   OptimizationTrajectory    │  ← 完整记录
    └─────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..base.base_optimizer import BaseOptimizer, SolverResult
from ..representation.encoder import PopulationEncoder, Individual
from ..representation.inverse_mapper import inverse_phi
from ..operators.crossover import dynamic_crossover_rate
from ..operators.scale_factor import dynamic_scale_factor_batch
from ..operators.mutation import mutate_population
from ..operators.extinction import should_extinct, apply_extinction
from ..features.population_features import compute_diversity, compute_gene_variance
from ..features.convergence_features import compute_convergence_speed, detect_stagnation
from ..features.constraint_features import compute_feasible_ratio, compute_violation_distribution
from ..trajectory.optimization_trajectory import OptimizationTrajectory, TrajectoryEntry
from ..llm.base_module import BaseLLMModule, ModuleState
from ..llm.llm_client import create_llm_client, create_llm_client_from_config

logger = logging.getLogger(__name__)


@dataclass
class LLMEnhancedDMDEConfig:
    """模块化 LLM 增强 DMDE 配置。

    支持通过 modules 配置字典启用/禁用各个 LLM 模块，
    实现灵活的消融实验。

    Attributes:
        # DMDE 核心参数
        pop_size:        种群大小。
        max_generations: 最大迭代代数。
        zeta:            动态交叉率曲率指数 ζ。
        delta:           灭绝临界值 δ。
        seed:            随机种子。
        verbose:         是否输出日志。
        log_interval:    日志间隔。

        # LLM 配置
        llm_config_path: LLM 配置文件路径（YAML 格式，含 provider/model/api_key 等）。
                         未设置时默认使用 DeepSeek。

        # 模块配置（消融实验的核心）
        modules: 各模块配置字典。
            格式: {"module_name": {"enabled": bool, "interval": int, ...}}
            可用模块名: "population_init", "search_controller"

        # 轨迹
        save_trajectory: 是否保存轨迹到 extra。
    """
    # DMDE 参数
    pop_size: int = 50
    max_generations: int = 1000
    zeta: int = 3
    delta: float = 0.3
    seed: int | None = None
    verbose: bool = False
    log_interval: int = 100

    # LLM 配置
    llm_config_path: str | None = None

    # 模块配置
    modules: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "population_init": {"enabled": False},
        "search_controller": {"enabled": True, "interval": 50},
    })

    # 轨迹
    save_trajectory: bool = True
    trajectory_window: int = 20


class LLMEnhancedDMDESolver(BaseOptimizer):
    """模块化 LLM 增强 DMDE 求解器。

    主循环尽量清晰、可重复：
    1. 初始化种群（可选 LLM 种群初始化模块）
    2. 每代进化：
       a. LLM CR 控制模块调整 CR/F
       b. 标准 DMDE 进化步骤
       c. LLM 算子选择模块决定策略
       d. 记录轨迹
    3. 输出结果 + 完整轨迹

    消融实验示例::

        # Vanilla DMDE（无 LLM）
        cfg = LLMEnhancedDMDEConfig(modules={})

        # 仅搜索控制器（策略 + CR 联合决策）
        cfg = LLMEnhancedDMDEConfig(modules={
            "search_controller": {"enabled": True, "interval": 50}
        })

        # 全部启用
        cfg = LLMEnhancedDMDEConfig(modules={
            "population_init": {"enabled": True},
            "search_controller": {"enabled": True, "interval": 50},
        })
    """

    def __init__(self, config: LLMEnhancedDMDEConfig | None = None) -> None:
        self._cfg = config or LLMEnhancedDMDEConfig()
        self._modules: list[BaseLLMModule] = []
        self._trajectory: OptimizationTrajectory | None = None

    @property
    def name(self) -> str:
        active = [m.name for m in self._modules if m.enabled]
        if not active:
            return "LLM-DMDE(vanilla)"
        return f"LLM-DMDE({'+'.join(active)})"

    @property
    def trajectory(self) -> OptimizationTrajectory | None:
        """获取优化轨迹（solve 之后可用）。"""
        return self._trajectory

    def solve(self, cost_matrix, n_uavs, n_targets, **kwargs) -> SolverResult:
        cfg = self._cfg
        fitness_evaluator = kwargs.get("fitness_evaluator")
        if fitness_evaluator is None:
            raise ValueError("fitness_evaluator must be provided.")

        rng = np.random.default_rng(cfg.seed)

        # 模型类型
        if n_uavs == n_targets:
            model_type = "balanced"
        elif n_uavs > n_targets:
            model_type = "overloaded"
        else:
            model_type = "srp"

        # 初始化 LLM 模块
        self._init_modules(cfg)

        # 初始化轨迹
        self._trajectory = OptimizationTrajectory()

        # ---- Step 1: 种群初始化 ----
        encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
        population = encoder.generate(cfg.pop_size, seed=cfg.seed)

        # 评估初始种群
        best_idx = 0
        for i, ind in enumerate(population):
            ind.fitness = self._evaluate(ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs)
            if ind.fitness < population[best_idx].fitness:
                best_idx = i

        best_individual = population[best_idx].copy()
        cost_history = [best_individual.fitness]

        # ---- LLM 种群初始化模块 (hook: after_init) ----
        pop_init_module = self._get_module("population_init")
        if pop_init_module and pop_init_module.enabled:
            state = self._build_state(
                0, cfg.max_generations, population, best_idx,
                cost_matrix, n_uavs, n_targets, model_type,
                cost_history, 0.5, 0.5, 1.0,
            )
            state.extra["pop_size"] = cfg.pop_size
            decision = pop_init_module.inject(state)
            self._record_decision(0, "population_init", decision, state)

            # Apply population changes from LLM init module
            if state.extra.get("llm_init_applied") and state.extra.get("llm_init_n_modified", 0) > 0:
                population = list(state.population)  # Copy back modified population
                # Re-evaluate modified individuals (marked with inf fitness)
                for i, ind in enumerate(population):
                    if ind.fitness == float("inf"):
                        ind.fitness = self._evaluate(ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs)
                    if ind.fitness < population[best_idx].fitness:
                        best_idx = i
                best_individual = population[best_idx].copy()
                cost_history = [best_individual.fitness]

            if cfg.verbose and decision:
                strategy = decision.get("init_strategy", "N/A")
                n_mod = state.extra.get("llm_init_n_modified", 0)
                print(f"  [LLM PopInit] {strategy} (modified {n_mod} individuals)")

        # LLM 决策状态缓存
        llm_cr = None          # None = 未被 LLM 设置，使用公式 3-9

        t_start = time.time()

        # ---- Step 2: DMDE 进化迭代 ----
        for gen in range(1, cfg.max_generations + 1):

            # ---- [Hook: before_mutation] 统一搜索控制器 ----
            sc_module = self._get_module("search_controller")
            if sc_module and sc_module.enabled and gen % sc_module.interval == 0:
                # 使用 LLM 的实际决策值（如有），否则用公式 3-9
                actual_cr = llm_cr if llm_cr is not None else dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
                # 计算实际的 F 值（取种群平均值作为代表值）
                actual_f_values = dynamic_scale_factor_batch(actual_cr, cfg.pop_size, rng)
                actual_f_mean = float(np.mean(actual_f_values))

                state = self._build_state(
                    gen, cfg.max_generations, population, best_idx,
                    cost_matrix, n_uavs, n_targets, model_type,
                    cost_history, actual_cr, actual_f_mean, 1.0 - gen / cfg.max_generations,
                )
                state.trajectory_recent = self._trajectory.get_recent(cfg.trajectory_window)

                decision = sc_module.inject(state)
                self._record_decision(gen, "search_controller", decision, state)

                if decision and "cr" in decision:
                    llm_cr = decision["cr"]

            # CR 来源：LLM 决定 or 公式 3-9（与纯 DMDE 一致）
            if llm_cr is not None:
                cr = llm_cr
            else:
                cr = dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
            # F 从 CR 按公式 3-11 批量计算（每个个体独立 F 值）
            f_values = dynamic_scale_factor_batch(cr, cfg.pop_size, rng)

            # 计算温度
            temperature = 1.0 - gen / cfg.max_generations

            # 提取代价值矩阵和适应度
            cost_vectors = np.array([ind.cost_vector for ind in population])
            fitness_values = np.array([ind.fitness for ind in population])

            # 混合变异产生试验向量
            trial_vectors = mutate_population(
                cost_vectors, best_idx, gen, cfg.max_generations, cfg.zeta, rng,
                cr=cr, f_scale=f_values,
            )

            # 对每个个体执行反映射 + 评估 + 贪婪选择
            for i in range(cfg.pop_size):
                child = inverse_phi(
                    trial_vectors[i], cost_matrix, n_uavs, n_targets, model_type,
                    rng=rng, temperature=temperature,
                )
                child.fitness = self._evaluate(child, fitness_evaluator, cost_matrix, n_uavs=n_uavs)
                if child.fitness < population[i].fitness:
                    population[i] = child
                    if child.fitness < best_individual.fitness:
                        best_individual = child.copy()
                        best_idx = i

            # GMR 灭绝判断
            if should_extinct(cr, cfg.delta, rng):
                fitness_arr = np.array([ind.fitness for ind in population])
                new_cv, survived = apply_extinction(
                    fitness_arr, cost_vectors, best_idx,
                    cost_matrix, n_uavs, n_targets, model_type, rng=rng,
                )
                for i in range(cfg.pop_size):
                    if i not in survived:
                        population[i] = inverse_phi(
                            new_cv[i], cost_matrix, n_uavs, n_targets, model_type, rng=rng,
                        )
                        population[i].fitness = self._evaluate(
                            population[i], fitness_evaluator, cost_matrix, n_uavs=n_uavs,
                        )

            cost_history.append(best_individual.fitness)

            # 记录常规轨迹点
            if cfg.save_trajectory and gen % max(1, cfg.max_generations // 100) == 0:
                fitness_values = np.array([ind.fitness for ind in population])
                self._trajectory.record(TrajectoryEntry(
                    generation=gen,
                    strategy=None,
                    cr=cr,
                    f_scale=float(np.mean(f_values)),
                    temperature=temperature,
                    fitness_best=best_individual.fitness,
                    fitness_mean=float(np.mean(fitness_values)),
                    fitness_worst=float(np.max(fitness_values)),
                    diversity=compute_diversity(population),
                    gene_variance=compute_gene_variance(cost_vectors),
                    convergence_speed=compute_convergence_speed(cost_history),
                    stagnation_count=detect_stagnation(cost_history),
                    feasible_ratio=compute_feasible_ratio(population),
                ))

            # 日志
            if cfg.verbose and gen % cfg.log_interval == 0:
                print(
                    f"  Gen {gen}/{cfg.max_generations}: "
                    f"best={best_individual.fitness:.2f}, "
                    f"CR={cr:.4f}, F={float(np.mean(f_values)):.4f}"
                )

        elapsed = time.time() - t_start

        # 构建结果
        extra = {
            "model_type": model_type,
            "pop_size": cfg.pop_size,
            "zeta": cfg.zeta,
            "delta": cfg.delta,
            "active_modules": [m.name for m in self._modules if m.enabled],
            "llm_decisions": self._trajectory.get_llm_decisions() if cfg.save_trajectory else [],
        }
        if cfg.save_trajectory:
            extra["trajectory_entries"] = len(self._trajectory)

        return SolverResult(
            best_assignment=best_individual.assignment,
            best_fitness=best_individual.fitness,
            cost_history=cost_history,
            total_generations=cfg.max_generations,
            elapsed_seconds=elapsed,
            solver_name=self.name,
            extra=extra,
        )

    def _init_modules(self, cfg: LLMEnhancedDMDEConfig) -> None:
        """根据配置初始化 LLM 模块。"""
        if not cfg.modules:
            self._modules = []
            return

        from ..llm.modules import create_module

        # 优先从配置文件加载，否则默认使用 DeepSeek
        if cfg.llm_config_path:
            try:
                llm_client = create_llm_client_from_config(cfg.llm_config_path)
            except Exception as e:
                if cfg.verbose:
                    print(f"  [Warning] Failed to load LLM config: {e}")
                llm_client = create_llm_client(provider="deepseek")
        else:
            llm_client = create_llm_client(provider="deepseek")

        self._modules = []
        for module_name, module_cfg in cfg.modules.items():
            try:
                # 注入全局种子到模块配置，保证可复现
                merged_cfg = {**module_cfg, "seed": cfg.seed}
                module = create_module(module_name, llm_client, merged_cfg)
                self._modules.append(module)
            except ValueError as e:
                if cfg.verbose:
                    print(f"  [Warning] {e}")

    def _get_module(self, name: str) -> BaseLLMModule | None:
        for m in self._modules:
            if m.name == name:
                return m
        return None

    def _build_state(
        self, gen, max_gen, population, best_idx,
        cost_matrix, n_uavs, n_targets, model_type,
        cost_history, cr, f_scale, temperature,
    ) -> ModuleState:
        fitness_values = np.array([ind.fitness for ind in population])
        cost_vectors = np.array([ind.cost_vector for ind in population])
        violation = compute_violation_distribution(population)

        return ModuleState(
            generation=gen,
            max_generations=max_gen,
            population=population,
            cost_vectors=cost_vectors,
            best_idx=best_idx,
            best_fitness=population[best_idx].fitness,
            mean_fitness=float(np.mean(fitness_values)),
            diversity=compute_diversity(population),
            gene_variance=compute_gene_variance(cost_vectors),
            convergence_speed=compute_convergence_speed(cost_history),
            stagnation_count=detect_stagnation(cost_history),
            feasible_ratio=compute_feasible_ratio(population),
            violation_mean=violation.get("mean", 0.0),
            violation_max=violation.get("max", 0.0),
            cr=cr,
            f_scale=f_scale,
            temperature=temperature,
            cost_matrix=cost_matrix,
            n_uavs=n_uavs,
            n_targets=n_targets,
            model_type=model_type,
            cost_history=list(cost_history),
        )

    def _record_decision(self, gen, module_name, decision, state):
        if self._trajectory and decision:
            clean = {k: v for k, v in decision.items() if not k.startswith("_")}
            # LLM 调用失败时保留错误信息，避免决策记录看起来像“空决策”
            if "_error" in decision:
                clean.setdefault("error", decision["_error"])
                logger.warning(
                    "[%s @ gen %d] LLM 调用失败，回退默认参数: %s",
                    module_name, gen, decision["_error"],
                )
            self._trajectory.record_llm_decision(
                generation=gen,
                llm_module=module_name,
                llm_decision=clean,
                llm_reasoning=decision.get("_llm_reasoning", ""),
                llm_raw_output=decision.get("_llm_raw_output", ""),
                llm_input=decision.get("_llm_input", {}),
                llm_call_duration=decision.get("_llm_call_duration", 0.0),
                fitness_best=state.best_fitness,
                diversity=state.diversity,
                feasible_ratio=state.feasible_ratio,
                convergence_speed=state.convergence_speed,
                stagnation_count=state.stagnation_count,
            )

    @staticmethod
    def _evaluate(individual, fitness_evaluator, cost_matrix, n_uavs=None):
        assignment = individual.assignment
        if not assignment:
            return 1e12
        result = fitness_evaluator.evaluate(assignment, cost_matrix, n_uavs=n_uavs)
        return result.fitness
