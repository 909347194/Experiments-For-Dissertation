# -*- coding: utf-8 -*-
"""llm_enhanced_dmde_solver.py — LLM 增强离散映射差分求解器

职责：
    实现 DMDE 算法的完整求解流程，并在每隔 p 代注入 LLM 决策，
    动态调整算子策略、交叉率、温度、灭绝参数等。

对应论文扩展：
    在算法 3.1（DMDE）基础上，引入 LLM 驱动的自适应算子选择。
    每隔 llm_interval 代：
    1. 提取搜索状态特征（features/*.py）
    2. 收集最近 p 代优化轨迹（trajectory_collector.py）
    3. 构建决策提示（prompt_builder.py）
    4. 调用 LLM（llm_client.py）
    5. 解析响应（response_parser.py）
    6. 应用 LLM 决策（更新 CR/F 策略、温度、灭绝参数）

算法流程：
    01-06: 初始化种群（encoder 生成离散三元组基因）
    07-32: DMDE 进化迭代（与 dmde_solver.py 一致）
        [每 llm_interval 代] 插入 LLM 决策步骤：
        → 提取特征 → 构建提示 → 调用 LLM → 解析 → 应用决策
    灭绝操作: GMR 判断（可被 LLM 决策覆盖）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..base.base_optimizer import BaseOptimizer, SolverResult
from ..representation.encoder import PopulationEncoder, Individual, Gene
from ..representation.mapper import phi
from ..representation.inverse_mapper import inverse_phi
from ..operators.crossover import dynamic_crossover_rate
from ..operators.scale_factor import dynamic_scale_factor
from ..operators.mutation import mutate_population
from ..operators.extinction import should_extinct, apply_extinction
from ..features.population_features import compute_diversity, compute_gene_variance
from ..features.convergence_features import compute_convergence_speed, detect_stagnation
from ..features.constraint_features import compute_feasible_ratio, compute_violation_distribution
from ..features.trajectory_collector import TrajectoryCollector
from ..llm.llm_client import LLMClient
from ..llm.prompt_builder import PromptBuilder
from ..llm.response_parser import ResponseParser, LLMDecision


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class LLMEnhancedDMDEConfig:
    """LLM 增强 DMDE 求解器配置。

    继承 DMDE 的全部参数，并新增 LLM 相关参数。

    Attributes:
        pop_size:        种群大小。
        max_generations: 最大迭代代数。
        zeta:            动态交叉率曲率指数 ζ（默认 3）。
        delta:           灭绝临界值 δ（默认 0.3）。
        alpha:           时间代价缩放因子 α。
        beta:            约束违背惩罚缩放因子 β。
        seed:            随机种子（可选）。
        verbose:         是否输出迭代日志。
        log_interval:    日志输出间隔（代数）。
        llm_interval:    调用 LLM 的间隔代数（默认 50）。
        llm_config_path: LLM 配置文件路径（YAML 格式）。
        enable_llm:      是否启用 LLM（关闭则退化为标准 DMDE）。
        trajectory_window: 轨迹收集窗口大小（代数）。
    """

    # DMDE 参数
    pop_size: int = 50
    max_generations: int = 1000
    zeta: int = 3
    delta: float = 0.3
    alpha: float = 2.5
    beta: float = 1.5
    seed: int | None = None
    verbose: bool = False
    log_interval: int = 100

    # LLM 参数
    llm_interval: int = 50
    llm_config_path: str | None = None
    enable_llm: bool = True
    trajectory_window: int = 10


# ---------------------------------------------------------------------------
# 主求解器
# ---------------------------------------------------------------------------

class LLMEnhancedDMDESolver(BaseOptimizer):
    """LLM 增强离散映射差分求解器。

    在标准 DMDE 基础上，每隔 llm_interval 代调用 LLM，
    根据搜索状态特征和优化轨迹动态调整算子参数。

    使用方式::

        solver = LLMEnhancedDMDESolver(
            config=LLMEnhancedDMDEConfig(
                pop_size=50,
                max_generations=1000,
                llm_interval=50,
                enable_llm=True,
            )
        )
        result = solver.solve(
            cost_matrix=cm,
            n_uavs=8,
            n_targets=8,
            fitness_evaluator=evaluator,
        )
    """

    def __init__(self, config: LLMEnhancedDMDEConfig | None = None) -> None:
        self._cfg = config or LLMEnhancedDMDEConfig()
        self._llm_client: LLMClient | None = None
        self._prompt_builder: PromptBuilder | None = None
        self._response_parser: ResponseParser | None = None
        self._trajectory_collector: TrajectoryCollector | None = None

    @property
    def name(self) -> str:
        return "LLM-Enhanced-DMDE"

    def solve(
        self,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        **kwargs,
    ) -> SolverResult:
        """执行 LLM 增强 DMDE 求解。

        每隔 llm_interval 代：
        1. 提取搜索状态特征（种群多样性、收敛速度、约束满足度等）。
        2. 收集最近 p 代的优化轨迹。
        3. 构建决策提示并调用 LLM。
        4. 解析 LLM 响应，获取算子策略调整建议。
        5. 应用 LLM 决策（调整 CR/F 策略、温度、灭绝参数等）。

        Args:
            cost_matrix: 代价矩阵。
            n_uavs:      UAV 数量。
            n_targets:   目标数量。
            **kwargs:
                fitness_evaluator: 适应度评估器（必须）。
                uavs: UAV 列表（可选，用于评估器）。
                targets: 目标列表（可选，用于评估器）。

        Returns:
            SolverResult 实例。
        """
        cfg = self._cfg
        fitness_evaluator = kwargs.get("fitness_evaluator")

        if fitness_evaluator is None:
            raise ValueError("fitness_evaluator must be provided.")

        # 随机数生成器
        rng = np.random.default_rng(cfg.seed)

        # 确定模型类型
        if n_uavs == n_targets:
            model_type = "balanced"
        elif n_uavs > n_targets:
            model_type = "overloaded"
        else:
            model_type = "srp"

        # 初始化 LLM 组件（如果启用）
        if cfg.enable_llm:
            self._init_llm_components(cfg)

        # 初始化轨迹收集器
        self._trajectory_collector = TrajectoryCollector(
            max_history=cfg.trajectory_window
        )

        # ---- Step 1: 初始化种群 (算法 3.1 行 01-06) ----
        encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
        population = encoder.generate(cfg.pop_size, seed=cfg.seed)

        # 评估初始种群
        best_idx = 0
        for i, ind in enumerate(population):
            ind.fitness = self._evaluate(
                ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs
            )
            if ind.fitness < population[best_idx].fitness:
                best_idx = i

        best_individual = population[best_idx].copy()
        cost_history = [best_individual.fitness]

        # LLM 决策状态（可被 LLM 动态调整）
        llm_decision = LLMDecision(
            operator_strategy="default",
            cr_adjustment=None,
            temperature_adjustment=None,
            extinction_trigger=None,
            reasoning="",
        )

        t_start = time.time()

        # ---- Step 2: DMDE 进化迭代 (算法 3.1 行 07-32) ----
        for gen in range(1, cfg.max_generations + 1):
            # 动态交叉率（可被 LLM 调整）
            cr = dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
            if llm_decision.cr_adjustment is not None:
                cr = np.clip(cr + llm_decision.cr_adjustment, 0.0, 1.0)

            # 提取代价值矩阵
            cost_vectors = np.array([ind.cost_vector for ind in population])

            # 混合变异产生试验向量
            trial_vectors = mutate_population(
                cost_vectors, best_idx, gen, cfg.max_generations, cfg.zeta, rng
            )

            # 温度：线性衰减 1.0 → 0.0（可被 LLM 调整）
            temperature = 1.0 - gen / cfg.max_generations
            if llm_decision.temperature_adjustment is not None:
                temperature = np.clip(
                    temperature + llm_decision.temperature_adjustment, 0.0, 1.0
                )

            # 对每个个体执行反映射 + 评估 + 贪婪选择
            for i in range(cfg.pop_size):
                # 反映射 (公式 3-7, 规则 3.4/3.5/3.6)
                child = inverse_phi(
                    trial_vectors[i], cost_matrix, n_uavs, n_targets,
                    model_type, rng=rng, temperature=temperature,
                )

                # 评估适应度
                child.fitness = self._evaluate(
                    child, fitness_evaluator, cost_matrix, n_uavs=n_uavs
                )

                # 贪婪选择 (算法 3.1 行 28-30)
                if child.fitness < population[i].fitness:
                    population[i] = child
                    if child.fitness < best_individual.fitness:
                        best_individual = child.copy()
                        best_idx = i

            # GMR 灭绝判断 (公式 3-12)
            # LLM 可以强制触发或禁止灭绝
            do_extinct = should_extinct(cr, cfg.delta, rng)
            if llm_decision.extinction_trigger is not None:
                do_extinct = llm_decision.extinction_trigger

            if do_extinct:
                fitness_arr = np.array([ind.fitness for ind in population])
                new_cost_vectors, survived = apply_extinction(
                    fitness_arr, cost_vectors, best_idx,
                    cost_matrix, n_uavs, n_targets, model_type, rng=rng,
                )
                # 用新代价值重建种群
                for i in range(cfg.pop_size):
                    if i not in survived:
                        population[i] = inverse_phi(
                            new_cost_vectors[i], cost_matrix,
                            n_uavs, n_targets, model_type, rng=rng,
                        )
                        population[i].fitness = self._evaluate(
                            population[i], fitness_evaluator,
                            cost_matrix, n_uavs=n_uavs,
                        )

            # 记录收敛曲线
            cost_history.append(best_individual.fitness)

            # ---- LLM 决策注入 ----
            if cfg.enable_llm and gen % cfg.llm_interval == 0:
                llm_decision = self._inject_llm_decision(
                    population=population,
                    cost_history=cost_history,
                    gen=gen,
                    best_idx=best_idx,
                    cost_matrix=cost_matrix,
                    n_uavs=n_uavs,
                    n_targets=n_targets,
                    rng=rng,
                )

                if cfg.verbose and llm_decision.reasoning:
                    print(f"  [LLM @ gen {gen}] {llm_decision.reasoning[:100]}...")

            # 日志
            if cfg.verbose and gen % cfg.log_interval == 0:
                print(
                    f"  Gen {gen}/{cfg.max_generations}: "
                    f"best_fitness={best_individual.fitness:.2f}, "
                    f"CR={cr:.4f}, T={temperature:.4f}"
                )

        elapsed = time.time() - t_start

        return SolverResult(
            best_assignment=best_individual.assignment,
            best_fitness=best_individual.fitness,
            cost_history=cost_history,
            total_generations=cfg.max_generations,
            elapsed_seconds=elapsed,
            solver_name=self.name,
            extra={
                "model_type": model_type,
                "pop_size": cfg.pop_size,
                "zeta": cfg.zeta,
                "delta": cfg.delta,
                "llm_interval": cfg.llm_interval,
                "enable_llm": cfg.enable_llm,
                "llm_decisions_count": gen // cfg.llm_interval if cfg.enable_llm else 0,
            },
        )

    def _init_llm_components(self, cfg: LLMEnhancedDMDEConfig) -> None:
        """初始化 LLM 客户端、提示构建器和响应解析器。

        从 llm_config_path 加载 LLM 配置（API base、key、model 等），
        如果配置文件不存在，使用默认值。

        Args:
            cfg: 求解器配置。
        """
        import yaml
        from pathlib import Path

        llm_params = {}
        if cfg.llm_config_path and Path(cfg.llm_config_path).exists():
            with open(cfg.llm_config_path, "r", encoding="utf-8") as f:
                llm_params = yaml.safe_load(f) or {}

        self._llm_client = LLMClient(
            api_base=llm_params.get("api_base", "https://api.openai.com/v1"),
            api_key=llm_params.get("api_key", ""),
            model=llm_params.get("model", "gpt-4"),
            temperature=llm_params.get("temperature", 0.7),
            max_tokens=llm_params.get("max_tokens", 1024),
        )
        self._prompt_builder = PromptBuilder()
        self._response_parser = ResponseParser()

    def _inject_llm_decision(
        self,
        population: list[Individual],
        cost_history: list[float],
        gen: int,
        best_idx: int,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        rng: np.random.Generator,
    ) -> LLMDecision:
        """提取特征、构建提示、调用 LLM、解析并返回决策。

        Args:
            population:   当前种群。
            cost_history: 收敛曲线。
            gen:          当前代数。
            best_idx:     最优个体索引。
            cost_matrix:  代价矩阵。
            n_uavs:       UAV 数量。
            n_targets:    目标数量。
            rng:          随机数生成器。

        Returns:
            LLMDecision 实例，包含算子策略调整建议。
        """
        # 1. 提取搜索状态特征
        fitness_values = np.array([ind.fitness for ind in population])
        cost_vectors = np.array([ind.cost_vector for ind in population])

        features = {
            "generation": gen,
            "best_fitness": population[best_idx].fitness,
            "mean_fitness": float(np.mean(fitness_values)),
            "diversity": compute_diversity(population),
            "gene_variance": compute_gene_variance(cost_vectors),
            "convergence_speed": compute_convergence_speed(cost_history),
            "stagnation_count": detect_stagnation(cost_history),
            "feasible_ratio": compute_feasible_ratio(population),
            "violation_distribution": compute_violation_distribution(population),
        }

        # 2. 收集优化轨迹
        self._trajectory_collector.record(
            generation=gen,
            features=features,
            decision=None,  # 上一次的决策
            fitness=population[best_idx].fitness,
        )
        trajectory = self._trajectory_collector.get_trajectory()

        # 3. 构建提示
        available_operators = [
            "rand/1", "best/1", "best/2", "current-to-pbest/1",
            "rand/2", "rand-to-best/1",
        ]
        messages = self._prompt_builder.build_decision_prompt(
            features=features,
            trajectory=trajectory,
            available_operators=available_operators,
        )

        # 4. 调用 LLM
        try:
            llm_output = self._llm_client.chat(messages)
        except Exception as e:
            # LLM 调用失败时返回默认决策（不影响 DMDE 主循环）
            return LLMDecision(
                operator_strategy="default",
                cr_adjustment=None,
                temperature_adjustment=None,
                extinction_trigger=None,
                reasoning=f"LLM call failed: {e}",
            )

        # 5. 解析响应
        decision = self._response_parser.parse(llm_output)

        # 6. 记录决策到轨迹
        self._trajectory_collector.record(
            generation=gen,
            features=features,
            decision=decision,
            fitness=population[best_idx].fitness,
        )

        return decision

    @staticmethod
    def _evaluate(
        individual: Individual,
        fitness_evaluator: Any,
        cost_matrix: np.ndarray,
        n_uavs: int | None = None,
    ) -> float:
        """评估个体适应度。"""
        assignment = individual.assignment
        if not assignment:
            return 1e12  # 空方案给极大惩罚

        result = fitness_evaluator.evaluate(assignment, cost_matrix, n_uavs=n_uavs)
        return result.fitness
