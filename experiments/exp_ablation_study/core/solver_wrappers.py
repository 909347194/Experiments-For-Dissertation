# -*- coding: utf-8 -*-
"""solver_wrappers.py — 求解器包装器（无侵入式过程记录）

通过继承 + hook 方式捕获每代 CR/F/fitness，不修改原始 solver 代码。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .recorder import RunRecorder


class RecordingDMDESolver:
    """包装 DMDESolver，在进化过程中记录每代 CR/F/fitness。"""

    def __init__(self, solver, recorder: RunRecorder):
        self._solver = solver
        self._rec = recorder
        self._cfg = solver._cfg

    def solve(self, cost_matrix, n_uavs, n_targets, **kwargs):
        """代理 solve，注入每代记录逻辑。"""
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

        from src.algorithms.algorithm_dmde.representation.encoder import PopulationEncoder
        from src.algorithms.algorithm_dmde.representation.inverse_mapper import inverse_phi
        from src.algorithms.algorithm_dmde.operators.crossover import dynamic_crossover_rate
        from src.algorithms.algorithm_dmde.operators.mutation import mutate_population
        from src.algorithms.algorithm_dmde.operators.extinction import should_extinct, apply_extinction

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

        # 记录初始代
        fitness_arr = np.array([ind.fitness for ind in population])
        self._rec.record_generation(
            gen=0, fitness_best=best_individual.fitness,
            fitness_mean=float(np.mean(fitness_arr)),
            cr=0.0, f_scale=0.0,
        )

        t_start = time.time()

        for gen in range(1, cfg.max_generations + 1):
            cr = dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
            cost_vectors = np.array([ind.cost_vector for ind in population])

            # F 值
            from src.algorithms.algorithm_dmde.operators.scale_factor import dynamic_scale_factor
            f_value = dynamic_scale_factor(cr, rng)

            trial_vectors = mutate_population(
                cost_vectors, best_idx, gen, cfg.max_generations, cfg.zeta, rng,
            )
            temperature = 1.0 - gen / cfg.max_generations

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

            # 记录每代数据
            fitness_arr = np.array([ind.fitness for ind in population])
            self._rec.record_generation(
                gen=gen, fitness_best=best_individual.fitness,
                fitness_mean=float(np.mean(fitness_arr)),
                cr=float(cr), f_scale=float(f_value),
            )

        elapsed = time.time() - t_start

        from src.algorithms.algorithm_dmde.base.base_optimizer import SolverResult
        return SolverResult(
            best_assignment=best_individual.assignment,
            best_fitness=best_individual.fitness,
            cost_history=cost_history,
            total_generations=cfg.max_generations,
            elapsed_seconds=elapsed,
            solver_name="DMDE",
            extra={"model_type": model_type},
        )

    @staticmethod
    def _evaluate(individual, fitness_evaluator, cost_matrix, n_uavs=None):
        assignment = individual.assignment
        if not assignment:
            return 1e12
        result = fitness_evaluator.evaluate(assignment, cost_matrix, n_uavs=n_uavs)
        return result.fitness


class RecordingLLMDESolver:
    """包装 LLMEnhancedDMDESolver，记录 LLM 决策详情。"""

    def __init__(self, solver, recorder: RunRecorder):
        self._solver = solver
        self._rec = recorder

    def solve(self, cost_matrix, n_uavs, n_targets, **kwargs):
        """代理 solve，从 trajectory 提取 LLM 决策记录。"""
        result = self._solver.solve(cost_matrix, n_uavs, n_targets, **kwargs)

        # 从 solver 的 trajectory 提取 LLM 决策
        trajectory = self._solver.trajectory
        if trajectory:
            llm_decisions = trajectory.get_llm_decisions()
            for d in llm_decisions:
                self._rec.record_llm_decision(
                    generation=d.get("generation", 0),
                    module=d.get("module", ""),
                    prompt_messages=d.get("llm_input", {}).get("messages", []),
                    model=d.get("llm_input", {}).get("model", "unknown"),
                    raw_output=d.get("llm_raw_output", ""),
                    parsed_decision=d.get("decision", {}),
                    reasoning=d.get("reasoning", ""),
                    duration=d.get("duration", 0.0),
                    temperature=d.get("llm_input", {}).get("temperature", 0.0),
                )

            # 从 trajectory 提取每代记录
            for entry in trajectory.get_all():
                self._rec.record_generation(
                    gen=entry.generation,
                    fitness_best=entry.fitness_best,
                    fitness_mean=entry.fitness_mean,
                    cr=entry.cr,
                    f_scale=entry.f_scale,
                    diversity=entry.diversity,
                    feasible_ratio=entry.feasible_ratio,
                    llm_module=entry.llm_module,
                    llm_decision=entry.llm_decision,
                    llm_reasoning=entry.llm_reasoning,
                    llm_call_duration=entry.llm_call_duration,
                )

        return result

    def __getattr__(self, name):
        return getattr(self._solver, name)