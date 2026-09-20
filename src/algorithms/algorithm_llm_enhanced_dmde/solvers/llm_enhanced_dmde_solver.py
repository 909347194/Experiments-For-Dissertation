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
from ..features.population_features import (
    compute_diversity, compute_gene_variance, compute_diversity_quantiles,
)
from ..features.convergence_features import compute_convergence_speed, detect_stagnation
from ..features.constraint_features import compute_feasible_ratio, compute_violation_distribution
from ..features.trigger import evaluate_trigger, stagnation_tier, DEFAULT_STAG_TIERS
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

            search_controller v2 闭环控制配置（可选）:
                trigger:
                    mode: "event"        # 事件触发（缺省 = 旧版固定 interval）
                    first_call: 50       # 首次决策最早代数
                    min_interval: 20     # 两次决策最小间隔（防抖）
                    max_interval: 100    # 最大间隔（fallback 定时器）
                    df_threshold: 0.05   # Δf 事件阈值（%）
                    dd_threshold: 0.02   # ΔD 事件阈值（噪声参考量级）
                    stag_tiers: [5, 10, 20, 40, 80, 160]  # 停滞跨档事件边界
                shadow:
                    enabled: true         # 影子对照种群（固定 CR 归因基线）
                    cr: 0.5              # 影子种群固定 CR
                actions:
                    restart_choices: [0.0, 0.1, 0.2, 0.3]  # 重启比例候选值
                freeze:
                    enabled: true        # 无证据时冻结 CR（默认 False，向后兼容）
                    df_noise: 0.05       # |df| 噪声阈值（%）
                    shadow_contrast: 0.05  # |df - df_shadow| 无差异阈值（%）

            v3 CR 动作空间：LLM 输出 {"cr_action": "hold"|"set", "cr": <值或 null>}。
            hold 是默认路径，解析失败/字段缺失/格式非法一律回退 hold；
            freeze 守卫触发时 solver 强制 hold（CR 通道本轮无信息）。

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

    # LLM 种群初始化参数（v2）
    # ⚠️ 以下参数均为实验调参项
    llm_init_ratio: float = 0.2           # α，LLM 候选注入比例（K = ceil(α × P)）
    llm_init_k_min: int = 3               # K_min，LLM 最少生成候选数
    llm_init_k_max: int = 10              # K_max，LLM 最多生成候选数
    llm_init_max_retries: int = 3         # LLM 生成失败时的重试次数
    llm_init_diversity_threshold: float = 0.1  # 多样性过滤阈值（0~1）
    llm_init_preference_top_k: int = 3    # prompt 中每行/列的 top-k 最小代价统计


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
        # 新流程（v2）:
        #   1. LLM pop_init module 生成候选 assignments（hook: before_init）
        #   2. AssignmentConverter 转换为 Individuals
        #   3. CandidateFilter 过滤（quality + diversity）
        #   4. DMDE 随机初始化补齐剩余个体
        #   5. 合并为初始种群
        #   6. 评估所有个体
        population = self._initialize_population(
            cost_matrix, n_uavs, n_targets, model_type,
            fitness_evaluator, cfg, rng,
        )

        # 评估初始种群（LLM 注入的个体已在 _initialize_population 中评估）
        best_idx = 0
        for i, ind in enumerate(population):
            if ind.fitness == float("inf"):
                ind.fitness = self._evaluate(ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs)
            if ind.fitness < population[best_idx].fitness:
                best_idx = i

        best_individual = population[best_idx].copy()
        cost_history = [best_individual.fitness]

        # 显式记录 gen=0 轨迹点（真实初始种群 fitness），
        # 避免 LLM 决策条目（fitness_best=inf）污染收敛曲线。
        if cfg.save_trajectory:
            fitness_arr = np.array([ind.fitness for ind in population])
            self._trajectory.record(TrajectoryEntry(
                generation=0,
                fitness_best=best_individual.fitness,
                fitness_mean=float(np.mean(fitness_arr)),
                fitness_worst=float(np.max(fitness_arr)),
                diversity=compute_diversity(population),
                feasible_ratio=compute_feasible_ratio(population),
            ))

        # LLM 决策状态缓存
        llm_cr = None          # None = 未被 LLM 设置，使用公式 3-9
        llm_cr_prev_fitness = best_individual.fitness  # 上次 LLM CR 决定时的 best fitness
        llm_cr_prev_diversity = 0.0                     # 上次 LLM CR 决定时的 diversity
        llm_cr_prev_gen = 0                             # 上次 LLM CR 决定时的代数
        llm_restart_prev = 0.0                          # 上次决策的重启比例
        llm_f_override = None    # None = 使用公式 3-11，float = LLM 直接指定 F
        llm_gmr_mode = "auto"    # "auto" = 公式 3-12，"on" = 强制灭绝，"off" = 禁止灭绝

        # Stage 级历史记录（闭环控制用）
        # 每次 LLM 调用 = 一个 stage，记录 stage 结束时的 fitness/diversity/CR
        stage_history: list[dict] = []
        stage_fitness = best_individual.fitness   # 当前 stage 起始 fitness
        stage_diversity = compute_diversity(population)  # 当前 stage 起始 diversity

        # Stage 级过程统计（可观测性：携带 stage 内搜索动态）
        stage_accepted = 0        # stage 内被贪婪选择接受的子代数
        stage_offspring = 0       # stage 内产生的子代总数
        stage_improvements = 0    # stage 内 best 被刷新的次数
        stage_best_curve: list[float] = []  # stage 内逐代 best fitness
        last_improve_gen = 0      # 上次 best 改进发生的代数
        tier_at_decision = 0      # 上次决策时的停滞档位

        # ---- v2 闭环控制配置（从 search_controller 模块配置读取） ----
        sc_cfg = (cfg.modules or {}).get("search_controller", {}) or {}
        trigger_cfg = sc_cfg.get("trigger", {}) or {}
        event_mode = trigger_cfg.get("mode") == "event"
        tc = {
            "first_call": int(trigger_cfg.get("first_call", 50)),
            "min_interval": int(trigger_cfg.get("min_interval", 20)),
            "max_interval": int(trigger_cfg.get("max_interval", 100)),
            "df_threshold": float(trigger_cfg.get("df_threshold", 0.05)),
            "dd_threshold": float(trigger_cfg.get("dd_threshold", 0.02)),
            "tiers": trigger_cfg.get("stag_tiers", DEFAULT_STAG_TIERS),
        }
        shadow_cfg = sc_cfg.get("shadow", {}) or {}
        shadow_enabled = bool(shadow_cfg.get("enabled", False))
        shadow_cr = float(shadow_cfg.get("cr", 0.5))

        # 无证据冻结守卫：本 stage 的 CR 通道若被判定为无信息（无改进且与
        # 影子对照无差异），则冻结 CR，LLM 本轮只能决定 restart_fraction。
        # 目的：不让控制器继续操作一个因果效应为零的通道（实测
        # corr(CR_t, CR_{t-1}) = -0.967，即无条件翻转）。
        freeze_cfg = sc_cfg.get("freeze", {}) or {}
        freeze_enabled = bool(freeze_cfg.get("enabled", False))
        freeze_df_noise = float(freeze_cfg.get("df_noise", tc["df_threshold"]))
        freeze_contrast = float(freeze_cfg.get("shadow_contrast", 0.05))
        # 上一轮执行了 restart 时，df vs df_shadow 被重启动作混淆（影子没有重启），
        # 不能再作为 CR 的证据 → 冻结 CR。
        freeze_on_confound = bool(freeze_cfg.get("confound", True))
        # confound 衰减：restart 后连续 confound_decay_stages 个 stage 未再重启，
        # 则认为 restart 的混淆效应已衰减，解除 confound 冻结。
        # 解决死锁：restart 是 LLM 唯一活跃动作 → 每次 restart → 永久冻结。
        confound_decay_stages = int(freeze_cfg.get("confound_decay_stages", 2))

        # 影子对照种群：固定 CR，从每个 stage 起点与主种群同源演化，
        # 为 LLM 的 CR 决策提供"如果不调整会怎样"的归因基线。
        # 使用独立 rng 流，不干扰主搜索的随机数序列。
        rng_shadow = np.random.default_rng((cfg.seed if cfg.seed is not None else 0) + 123456)
        shadow_pop: list[Individual] | None = None
        shadow_best_fitness = float("inf")
        shadow_best_idx = 0
        restart_counter = 0        # 已执行的重启次数（用于派生可复现随机种子）
        restart_encoder = None     # 惰性创建的重启个体生成器
        stages_since_restart = 0   # 距上次 restart 的 stage 数（confound 衰减计数）

        t_start = time.time()

        # ---- Step 2: DMDE 进化迭代 ----
        for gen in range(1, cfg.max_generations + 1):

            # ---- [Hook: before_mutation] 统一搜索控制器 ----
            sc_module = self._get_module("search_controller")
            fire_decision = False
            trigger_reason = ""
            if sc_module and sc_module.enabled:
                if event_mode:
                    # 事件触发：只在搜索状态发生值得注意的变化时咨询 LLM
                    # 前馈补偿：上次决策执行了 restart 时，屏蔽 diversity_move 事件
                    # （restart 注入随机个体必然移动 diversity，那是执行器自身的
                    #  效应，不应回灌触发器形成 restart → ΔD 事件 → 决策的自锁）
                    div_now = compute_diversity(population)
                    stag_now = detect_stagnation(cost_history)
                    fire_decision, trigger_reason = evaluate_trigger(
                        gens_since_decision=gen - llm_cr_prev_gen,
                        is_first_call=llm_cr is None,
                        best_fitness_now=best_individual.fitness,
                        stage_start_fitness=stage_fitness,
                        diversity_now=div_now,
                        stage_start_diversity=stage_diversity,
                        stagnation_now=stag_now,
                        tier_at_last_decision=tier_at_decision,
                        tiers=tc["tiers"],
                        first_call=tc["first_call"],
                        min_interval=tc["min_interval"],
                        max_interval=tc["max_interval"],
                        df_threshold=tc["df_threshold"],
                        dd_threshold=tc["dd_threshold"],
                        diversity_event_enabled=(llm_restart_prev <= 0),
                    )
                else:
                    # 旧模式：固定 interval（向后兼容）
                    fire_decision = gen % sc_module.interval == 0
                    trigger_reason = f"fixed_interval: {sc_module.interval}"

            if fire_decision:
                # 使用 LLM 的实际决策值（如有），否则用公式 3-9
                actual_cr = llm_cr if llm_cr is not None else dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
                # 计算实际的 F 值：LLM override 优先，否则公式 3-11
                if llm_f_override is not None:
                    actual_f_mean = llm_f_override
                else:
                    actual_f_values = dynamic_scale_factor_batch(actual_cr, cfg.pop_size, rng)
                    actual_f_mean = float(np.mean(actual_f_values))

                state = self._build_state(
                    gen, cfg.max_generations, population, best_idx,
                    cost_matrix, n_uavs, n_targets, model_type,
                    cost_history, actual_cr, actual_f_mean, 1.0 - gen / cfg.max_generations,
                )
                state.trajectory_recent = self._trajectory.get_recent(cfg.trajectory_window)

                # ---- Δ 趋势信号（闭环控制核心） ----
                current_fitness = best_individual.fitness
                current_diversity = compute_diversity(population)

                # 第一次 LLM 调用：无上次决策，delta 标记为 None
                is_first_call = llm_cr is None

                if is_first_call:
                    delta_fitness = None
                    delta_diversity = None
                    prev_delta_fitness = None
                    prev_delta_diversity = None
                else:
                    # Δf_t: 当前 stage 的 fitness 变化（%）
                    if stage_fitness > 0 and np.isfinite(stage_fitness):
                        delta_fitness = (stage_fitness - current_fitness) / stage_fitness * 100.0
                    else:
                        delta_fitness = 0.0
                    # ΔD_t: 当前 stage 的多样性变化
                    delta_diversity = current_diversity - stage_diversity

                    # Δf_{t-1}, ΔD_{t-1}: 上个 stage 的变化
                    if len(stage_history) >= 1:
                        prev = stage_history[-1]
                        prev_delta_fitness = prev.get("delta_fitness")
                        prev_delta_diversity = prev.get("delta_diversity")
                    else:
                        prev_delta_fitness = None
                        prev_delta_diversity = None

                # 注入 Δ 信号到 state
                state.delta_fitness = round(delta_fitness, 4) if delta_fitness is not None else None
                state.delta_diversity = round(delta_diversity, 4) if delta_diversity is not None else None
                state.prev_delta_fitness = round(prev_delta_fitness, 4) if prev_delta_fitness is not None else None
                state.prev_delta_diversity = round(prev_delta_diversity, 4) if prev_delta_diversity is not None else None
                state.prev_action = llm_cr  # None on first call
                state.stage_history = stage_history[-5:]  # 最近 5 个 stage

                # ---- 可观测性扩展：stage 级过程统计 ----
                state.stage_length = gen - llm_cr_prev_gen
                state.acceptance_rate = (
                    stage_accepted / stage_offspring if stage_offspring > 0 else None
                )
                state.improvements_in_stage = stage_improvements
                state.gens_since_last_improvement = gen - last_improve_gen
                state.stagnation_raw = detect_stagnation(cost_history)
                state.diversity_p25, state.diversity_p75 = compute_diversity_quantiles(population)
                state.stage_best_curve = self._downsample_curve(stage_best_curve, max_points=12)
                state.trigger_reason = trigger_reason
                state.prev_action_restart = llm_restart_prev

                # ---- 影子对照（可归因）：同一 stage 窗口内固定 CR 的基线 ----
                if shadow_pop is not None:
                    state.shadow_cr = shadow_cr
                    if stage_fitness > 0 and np.isfinite(stage_fitness):
                        state.shadow_delta_fitness = (
                            stage_fitness - shadow_best_fitness
                        ) / stage_fitness * 100.0
                    else:
                        state.shadow_delta_fitness = 0.0
                    state.shadow_delta_diversity = (
                        compute_diversity(shadow_pop) - stage_diversity
                    )

                # ---- 无证据守卫：判定 CR 通道本轮是否可操作 ----
                cr_frozen = False
                cr_frozen_reason = ""
                if freeze_enabled and not is_first_call:
                    df_abs = abs(delta_fitness) if delta_fitness is not None else 0.0
                    contrast = None
                    if delta_fitness is not None and state.shadow_delta_fitness is not None:
                        contrast = abs(delta_fitness - state.shadow_delta_fitness)

                    if freeze_on_confound and llm_restart_prev > 0 and stages_since_restart < confound_decay_stages:
                        # 重启动作混淆了 CR 归因：本轮 df 的改善可能来自注入个体
                        # confound_decay_stages 后自动衰减，允许 LLM 重新操作 CR
                        cr_frozen = True
                        cr_frozen_reason = (
                            f"restart_fraction={llm_restart_prev:.1f} applied "
                            f"{stages_since_restart} stage(s) ago (< decay "
                            f"threshold {confound_decay_stages}): "
                            f"df vs df_shadow confounded"
                        )
                    elif df_abs < freeze_df_noise and (contrast is None or contrast < freeze_contrast):
                        cr_frozen = True
                        if contrast is None:
                            cr_frozen_reason = (
                                f"df={delta_fitness:+.2f}% within noise "
                                f"(<{freeze_df_noise}%), no shadow baseline"
                            )
                        else:
                            cr_frozen_reason = (
                                f"df={delta_fitness:+.2f}% within noise "
                                f"(<{freeze_df_noise}%) and |df-df_shadow|="
                                f"{contrast:.2f}% (<{freeze_contrast}%)"
                            )
                state.cr_frozen = cr_frozen
                state.cr_frozen_reason = cr_frozen_reason

                # 旧版反馈字段保留兼容
                state.extra["previous_llm_cr"] = llm_cr
                state.extra["previous_interval_gens"] = gen - llm_cr_prev_gen
                state.extra["fitness_change_since_last"] = None if is_first_call else llm_cr_prev_fitness - current_fitness
                state.extra["diversity_change_since_last"] = None if is_first_call else current_diversity - llm_cr_prev_diversity

                decision = sc_module.inject(state)

                if decision:
                    # ---- CR 动作解析：hold 为默认，set 才取值 ----
                    cr_action = decision.get("cr_action", "hold")
                    cr_requested = decision.get("cr")   # 模型原始请求（审计用）
                    if cr_frozen:
                        # 守卫强制保持：CR 通道本轮无信息，不采纳任何改动
                        cr_action = "hold"
                        decision["cr"] = None
                    new_cr = decision.get("cr")
                    if new_cr is None or not str(cr_action).startswith("set"):
                        # hold（或无法解析）：保持当前 CR；首次则回退到 0.5
                        new_cr = llm_cr if llm_cr is not None else 0.5

                    # 记录生效后的决策（含守卫判定与模型原始请求），
                    # 保证审计记录与实际执行一致 —— 否则冻结时日志仍显示模型的
                    # 原始 set 值，会误导归因分析。
                    decision["cr_action"] = cr_action
                    decision["cr_requested"] = cr_requested
                    decision["cr_effective"] = new_cr
                    decision["cr_frozen"] = cr_frozen
                    decision["cr_frozen_reason"] = cr_frozen_reason
                    decision["f_effective"] = llm_f_override
                    decision["gmr_mode_effective"] = llm_gmr_mode
                    self._record_decision(gen, "search_controller", decision, state)

                    shadow_df_val = state.shadow_delta_fitness
                    shadow_dd_val = state.shadow_delta_diversity
                    # 记录 stage 结束时的快照到 stage_history
                    # CR 记录 actual_cr（产生 outcome 的 CR），不是 new_cr（刚选的 CR）
                    stage_history.append({
                        "stage": len(stage_history) + 1,
                        "gen_start": llm_cr_prev_gen,
                        "gen_end": gen,
                        "stage_length": gen - llm_cr_prev_gen,
                        "cr": round(actual_cr, 4),
                        "restart_fraction": round(llm_restart_prev, 2),
                        "best_fitness": round(current_fitness, 2),
                        "delta_fitness": round(delta_fitness, 4) if delta_fitness is not None else None,
                        "diversity": round(current_diversity, 4),
                        "delta_diversity": round(delta_diversity, 4) if delta_diversity is not None else None,
                        "shadow_delta_fitness": (
                            round(shadow_df_val, 4) if shadow_df_val is not None else None
                        ),
                        "shadow_delta_diversity": (
                            round(shadow_dd_val, 4) if shadow_dd_val is not None else None
                        ),
                        "acceptance_rate": (
                            round(state.acceptance_rate, 4)
                            if state.acceptance_rate is not None else None
                        ),
                        "trigger_reason": trigger_reason,
                        # v3：CR 动作是否被采纳、通道是否冻结
                        "cr_action": cr_action,
                        "cr_requested": (
                            round(decision.get("cr"), 4)
                            if decision.get("cr") is not None else None
                        ),
                        "cr_frozen": cr_frozen,
                        # v4：LLM 独立控制的 F 和 GMR
                        "f_scale": round(llm_f_override, 4) if llm_f_override is not None else "auto",
                        "gmr_mode": llm_gmr_mode,
                    })

                    # ---- 应用 restart_fraction（能控性：收敛后的有效动作） ----
                    new_restart = float(decision.get("restart_fraction", 0.0) or 0.0)
                    if new_restart > 0:
                        if restart_encoder is None:
                            restart_encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
                        restart_counter += 1
                        population, best_individual, best_idx = self._apply_restart(
                            population, best_individual, best_idx,
                            new_restart, restart_encoder, restart_counter,
                            fitness_evaluator, cost_matrix, n_uavs, cfg,
                        )
                        stages_since_restart = 0  # restart 执行，重置衰减计数
                        if cfg.verbose:
                            print(
                                f"  [SearchController @ gen {gen}] restart applied: "
                                f"{new_restart:.0%} of worst individuals replaced"
                            )
                    else:
                        stages_since_restart += 1  # 无 restart，衰减计数递增

                    # 更新 stage 起始快照（重启之后，使下一 stage 的 Δ 反映新策略起点）
                    current_fitness = best_individual.fitness
                    current_diversity = compute_diversity(population)
                    stage_fitness = current_fitness
                    stage_diversity = current_diversity
                    stage_accepted = 0
                    stage_offspring = 0
                    stage_improvements = 0
                    stage_best_curve = []
                    tier_at_decision = stagnation_tier(state.stagnation_raw, tc["tiers"])
                    # 更新 LLM 决策缓存
                    llm_cr = new_cr
                    llm_restart_prev = new_restart
                    llm_cr_prev_fitness = current_fitness
                    llm_cr_prev_diversity = current_diversity
                    llm_cr_prev_gen = gen
                    # F: 从 decision 读取 LLM 指定值
                    llm_f_val = decision.get("f", None)
                    if llm_f_val is not None:
                        llm_f_override = float(llm_f_val)
                    # GMR: 从 decision 读取 LLM 指定模式
                    llm_gmr_mode = decision.get("gmr_mode", "auto")

                    # ---- 影子种群重置为当前主种群（下一 stage 的对照起点） ----
                    if shadow_enabled:
                        shadow_pop = [ind.copy() for ind in population]
                        shadow_fitness_arr = [ind.fitness for ind in shadow_pop]
                        shadow_best_idx = int(np.argmin(shadow_fitness_arr))
                        shadow_best_fitness = shadow_pop[shadow_best_idx].fitness

            # CR 来源：LLM 决定 or 公式 3-9（与纯 DMDE 一致）
            if llm_cr is not None:
                cr = llm_cr
            else:
                cr = dynamic_crossover_rate(gen, cfg.max_generations, cfg.zeta)
            # F 来源：LLM 直接指定 or 公式 3-11
            if llm_f_override is not None:
                f_values = np.full(cfg.pop_size, llm_f_override)
            else:
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
                    stage_accepted += 1
                    if child.fitness < best_individual.fitness:
                        best_individual = child.copy()
                        best_idx = i
                        stage_improvements += 1
                        last_improve_gen = gen
            stage_offspring += cfg.pop_size

            # GMR 灭绝判断（LLM 可覆写）
            trigger_extinction = False
            if llm_gmr_mode == "on":
                trigger_extinction = True
            elif llm_gmr_mode == "off":
                trigger_extinction = False
            else:  # "auto"
                trigger_extinction = should_extinct(cr, cfg.delta, rng)

            if trigger_extinction:
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

            # ---- 影子对照种群进化（固定 CR，stage 级归因基线） ----
            # 与主种群同一 stage 起点、同一算子链，唯一差异是 CR 固定，
            # 因此下一决策点的主/影子 Δf 之差可归因于 LLM 的 CR 选择。
            if shadow_pop is not None:
                sc_cost_vectors = np.array([ind.cost_vector for ind in shadow_pop])
                sf_values = dynamic_scale_factor_batch(shadow_cr, cfg.pop_size, rng_shadow)
                sc_trial = mutate_population(
                    sc_cost_vectors, shadow_best_idx, gen, cfg.max_generations, cfg.zeta,
                    rng_shadow, cr=shadow_cr, f_scale=sf_values,
                )
                for i in range(cfg.pop_size):
                    sc_child = inverse_phi(
                        sc_trial[i], cost_matrix, n_uavs, n_targets, model_type,
                        rng=rng_shadow, temperature=temperature,
                    )
                    sc_child.fitness = self._evaluate(
                        sc_child, fitness_evaluator, cost_matrix, n_uavs=n_uavs,
                    )
                    if sc_child.fitness < shadow_pop[i].fitness:
                        shadow_pop[i] = sc_child
                        if sc_child.fitness < shadow_best_fitness:
                            shadow_best_fitness = sc_child.fitness
                            shadow_best_idx = i
                if should_extinct(shadow_cr, cfg.delta, rng_shadow):
                    sc_fitness_arr = np.array([ind.fitness for ind in shadow_pop])
                    sc_new_cv, sc_survived = apply_extinction(
                        sc_fitness_arr, sc_cost_vectors, shadow_best_idx,
                        cost_matrix, n_uavs, n_targets, model_type, rng=rng_shadow,
                    )
                    for i in range(cfg.pop_size):
                        if i not in sc_survived:
                            shadow_pop[i] = inverse_phi(
                                sc_new_cv[i], cost_matrix, n_uavs, n_targets, model_type,
                                rng=rng_shadow,
                            )
                            shadow_pop[i].fitness = self._evaluate(
                                shadow_pop[i], fitness_evaluator, cost_matrix, n_uavs=n_uavs,
                            )
                    sc_fits = [ind.fitness for ind in shadow_pop]
                    shadow_best_idx = int(np.argmin(sc_fits))
                    shadow_best_fitness = sc_fits[shadow_best_idx]

            cost_history.append(best_individual.fitness)
            stage_best_curve.append(best_individual.fitness)

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
        # 计算 LLM 总耗时
        llm_decisions_data = self._trajectory.get_llm_decisions() if cfg.save_trajectory else []
        llm_total_time = sum(d.get("duration", 0.0) for d in llm_decisions_data)

        extra = {
            "model_type": model_type,
            "pop_size": cfg.pop_size,
            "zeta": cfg.zeta,
            "delta": cfg.delta,
            "active_modules": [m.name for m in self._modules if m.enabled],
            "llm_decisions": llm_decisions_data,
            "llm_time": llm_total_time,
            "llm_call_count": len(llm_decisions_data),
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

    def _initialize_population(
        self, cost_matrix, n_uavs, n_targets, model_type,
        fitness_evaluator, cfg, rng,
    ) -> list[Individual]:
        """初始化种群（v2 流程）。

        流程：
        1. 如果 population_init 模块启用，调用 LLM 生成候选 assignments
        2. AssignmentConverter 转换为 Individuals
        3. CandidateFilter 过滤（quality + diversity）
        4. DMDE 随机初始化补齐剩余个体
        5. 合并为初始种群

        如果 LLM 模块禁用或调用失败，fallback 到标准 DMDE 初始化。
        """
        from ..llm.modules.assignment_converter import AssignmentConverter
        from ..llm.modules.candidate_filter import CandidateFilter

        encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
        pop_size = cfg.pop_size

        # 检查 population_init 模块是否启用
        pop_init_module = self._get_module("population_init")
        if not (pop_init_module and pop_init_module.enabled):
            # 模块禁用 → 标准 DMDE 初始化（与当前行为完全一致）
            return encoder.generate(pop_size, seed=cfg.seed)

        # ---- LLM 种群初始化 (hook: before_init) ----
        # K = min(K_max, max(K_min, ceil(α × P)))
        alpha = getattr(cfg, "llm_init_ratio", 0.2)
        k_min = getattr(cfg, "llm_init_k_min", 3)
        k_max = getattr(cfg, "llm_init_k_max", 10)
        k = min(k_max, max(k_min, int(np.ceil(alpha * pop_size))))

        # 构建 before_init 状态（此时还没有种群）
        state = self._build_state_before_init(
            cost_matrix, n_uavs, n_targets, model_type,
        )
        state.extra["pop_size"] = pop_size
        state.extra["alpha"] = alpha
        state.extra["k_min"] = k_min
        state.extra["k_max"] = k_max
        state.extra["preference_top_k"] = getattr(cfg, "llm_init_preference_top_k", 3)

        # state.extra 已包含 pop_size, alpha, k_min, k_max（见上方 _build_state_before_init）
        # build_prompt 会从 state.extra 读取这些值，无需直接修改模块私有配置

        # 带重试的 LLM 调用
        max_retries = getattr(cfg, "llm_init_max_retries", 3)
        candidate_solutions = []
        decision = {}

        for attempt in range(max_retries):
            try:
                decision = pop_init_module.inject(state)
                self._record_decision(0, "population_init", decision, state)
                candidate_solutions = state.extra.get("candidate_solutions", [])
                if candidate_solutions:
                    break
                logger.info(
                    "[PopInit] 第 %d 次尝试未生成有效 solutions，重试...",
                    attempt + 1,
                )
            except Exception as e:
                logger.warning(
                    "[PopInit] 第 %d 次 LLM 调用失败: %s",
                    attempt + 1, e,
                )
                decision = {"_error": str(e)}

        if not candidate_solutions:
            # LLM 调用全部失败 → fallback 到标准 DMDE 初始化
            logger.info("[PopInit] LLM 调用失败，fallback 到标准 DMDE 初始化")
            if cfg.verbose:
                print("  [LLM PopInit] Failed, falling back to standard DMDE init")
            return encoder.generate(pop_size, seed=cfg.seed)

        # ---- 转换 assignments → Individuals ----
        converter = AssignmentConverter(cost_matrix, n_uavs, n_targets)
        candidates, convert_stats = converter.convert_batch(candidate_solutions)

        # 记录 LLM 种群初始化质量统计
        n_total = len(candidate_solutions)
        n_raw = convert_stats["n_raw_valid"]
        n_rep = convert_stats["n_repaired"]
        n_fail = convert_stats["n_failed"]
        dists = convert_stats["repair_distances"]
        raw_costs = convert_stats["raw_costs"]
        rep_costs = convert_stats["repaired_costs"]
        avg_dist = float(np.mean(dists)) if dists else 0.0
        avg_raw = float(np.mean(raw_costs)) if raw_costs else float("inf")
        avg_rep = float(np.mean(rep_costs)) if rep_costs else float("inf")
        logger.info(
            "[PopInit] LLM %d 解: %d 直接可行, %d 修复, %d 失败 "
            "| 平均修复距离=%.1f | 平均 cost: 原始=%.1f 修复后=%.1f",
            n_total, n_raw, n_rep, n_fail, avg_dist, avg_raw, avg_rep,
        )
        if cfg.verbose:
            print(
                f"  [LLM PopInit] {n_total} solutions: "
                f"{n_raw} raw valid, {n_rep} repaired (avg dist={avg_dist:.1f}), "
                f"{n_fail} failed"
            )

        if not candidates:
            logger.info("[PopInit] 所有 assignments 转换失败，fallback 到标准 DMDE 初始化")
            return encoder.generate(pop_size, seed=cfg.seed)

        # 评估候选个体
        for ind in candidates:
            ind.fitness = self._evaluate(ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs)

        # ---- Quality + Diversity 过滤 ----
        diversity_threshold = getattr(cfg, "llm_init_diversity_threshold", 0.1)
        candidate_filter = CandidateFilter(
            fitness_evaluator=fitness_evaluator,
            cost_matrix=cost_matrix,
            n_uavs=n_uavs,
            n_targets=n_targets,
            diversity_threshold=diversity_threshold,
            seed=cfg.seed,
        )
        selected = candidate_filter.filter(candidates, k)

        # ---- 合并：LLM 候选 + 随机补齐 ----
        n_random = pop_size - len(selected)
        if n_random > 0:
            random_pop = encoder.generate(n_random, seed=cfg.seed)
            population = list(selected) + random_pop
        else:
            population = list(selected[:pop_size])

        # 日志
        logger.info(
            "[PopInit] LLM 生成 %d 候选解 → 过滤后 %d 注入 + %d 随机 = %d 总种群",
            len(candidate_solutions), len(selected), n_random, len(population),
        )
        if cfg.verbose:
            llm_fitness = [ind.fitness for ind in selected]
            best_llm = min(llm_fitness) if llm_fitness else float("inf")
            print(
                f"  [LLM PopInit] {len(candidate_solutions)} solutions → "
                f"{len(selected)} selected (best={best_llm:.2f}) + "
                f"{n_random} random = {len(population)} total"
            )

        return population

    def _build_state_before_init(
        self, cost_matrix, n_uavs, n_targets, model_type,
    ) -> ModuleState:
        """构建 before_init 阶段的 ModuleState。

        此时还没有种群，只提供问题结构和代价矩阵信息。
        """
        return ModuleState(
            generation=0,
            max_generations=0,
            population=None,
            cost_vectors=None,
            best_idx=0,
            best_fitness=float("inf"),
            mean_fitness=float("inf"),
            diversity=0.0,
            gene_variance=0.0,
            convergence_speed=0.0,
            stagnation_count=0,
            feasible_ratio=0.0,
            violation_mean=0.0,
            violation_max=0.0,
            cr=0.5,
            f_scale=0.5,
            temperature=1.0,
            cost_matrix=cost_matrix,
            n_uavs=n_uavs,
            n_targets=n_targets,
            model_type=model_type,
            cost_history=[],
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
                # 必须无条件下告警：静默回退会让 llm_config.yaml 中的
                # provider/model/max_tokens/thinking 设置全部失效，
                # 实验实际跑的是另一个模型和默认预算（1024 tokens）。
                print(
                    f"  [ERROR] 加载 LLM 配置失败，已回退 DeepSeek 默认客户端：{e}\n"
                    f"  [ERROR] {cfg.llm_config_path} 中的 provider/model/"
                    f"max_tokens/thinking 设置全部失效！"
                )
                llm_client = create_llm_client(provider="deepseek")
        else:
            print("  [ERROR] 未提供 llm_config_path，回退 DeepSeek 默认客户端")
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
        if self._trajectory is not None and decision:
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
    def _downsample_curve(curve: list[float], max_points: int = 12) -> list[list[float]]:
        """将 stage 内逐代 best 曲线降采样为 [gen_offset, Δ%] 点列。

        Δ% 相对 stage 起点归一化，避免在 prompt 中塞入大数值绝对 fitness，
        同时保留 stage 内的收敛动态（何时改进、改进多少）。
        """
        if not curve:
            return []
        n = len(curve)
        if n <= max_points:
            idx = list(range(n))
        else:
            idx = [int(i) for i in np.linspace(0, n - 1, max_points)]
        start = curve[0]
        pts = []
        for i in idx:
            if abs(start) > 1e-12:
                delta_pct = round((start - curve[i]) / abs(start) * 100.0, 3)
            else:
                delta_pct = 0.0
            pts.append([i + 1, delta_pct])
        return pts

    def _apply_restart(
        self, population, best_individual, best_idx,
        restart_fraction: float, encoder, restart_counter: int,
        fitness_evaluator, cost_matrix, n_uavs, cfg,
    ):
        """执行 LLM 决策的重启动作：用新鲜随机个体替换最差比例的个体。

        种子由 (cfg.seed, restart_counter) 派生，保证同 seed 可复现、
        不同次重启产生不同个体。best 个体不受影响（只替换最差端）。

        Returns:
            (population, best_individual, best_idx) 重启后的元组。
        """
        k = int(round(restart_fraction * len(population)))
        if k <= 0:
            return population, best_individual, best_idx

        # encoder.generate(seed=None) 使用全局 numpy 随机流；
        # 每次重启前播种派生种子 → 可复现且每次不同
        np.random.seed((cfg.seed if cfg.seed is not None else 0) * 100003 + 17 * restart_counter)

        fresh = encoder.generate(k)
        worst_order = np.argsort([ind.fitness for ind in population])[::-1]

        for j in range(k):
            ind = fresh[j]
            ind.fitness = self._evaluate(ind, fitness_evaluator, cost_matrix, n_uavs=n_uavs)
            pos = int(worst_order[j])
            population[pos] = ind
            if ind.fitness < best_individual.fitness:
                best_individual = ind.copy()
                best_idx = pos

        # 保险：重算 best_idx（重启理论上只动最差端，但以防万一）
        fits = [ind.fitness for ind in population]
        best_idx = int(np.argmin(fits))
        return population, best_individual, best_idx

    @staticmethod
    def _evaluate(individual, fitness_evaluator, cost_matrix, n_uavs=None):
        assignment = individual.assignment
        if not assignment:
            return 1e12
        result = fitness_evaluator.evaluate(assignment, cost_matrix, n_uavs=n_uavs)
        return result.fitness
