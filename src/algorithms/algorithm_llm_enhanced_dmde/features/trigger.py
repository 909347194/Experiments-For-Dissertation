# -*- coding: utf-8 -*-
"""trigger.py — LLM 决策的事件触发器

闭环控制的"可触发"环节：不再按固定 interval 定时调用 LLM，
而是只在搜索状态发生值得注意的事件时触发决策：

事件类型（任一满足即可触发）：
    1. fitness_move : 自上次决策以来 best fitness 相对变化 |Δf| ≥ df_threshold
    2. diversity_move: 自上次决策以来多样性变化 |ΔD| ≥ dd_threshold
    3. stagnation_deepen: 真实停滞代数跨入更高档位（tier）

保护约束：
    - min_interval: 两次决策之间的最小间隔（防止抖动/频繁调用）
    - max_interval: 最大间隔（fallback，保证 LLM 周期性被咨询）
    - first_call:   首次决策的最早代数（保证有初始历史可看）

设计为纯函数，便于单元测试与离线回放分析。
"""

from __future__ import annotations

# 默认事件档位：停滞代数跨过这些边界视为事件
DEFAULT_STAG_TIERS: list[int] = [5, 10, 20, 40, 80, 160]


def stagnation_tier(stagnation_count: int, tiers: list[int] | None = None) -> int:
    """将停滞代数映射到档位索引（0 = 未停滞区域）。

    tier = 满足 stagnation_count >= boundary 的边界个数。
    例：tiers=[5,10,20], stag=15 → tier 2（已跨过 5 和 10）。
    """
    if tiers is None:
        tiers = DEFAULT_STAG_TIERS
    return sum(1 for b in tiers if stagnation_count >= b)


def evaluate_trigger(
    gens_since_decision: int,
    is_first_call: bool,
    best_fitness_now: float,
    stage_start_fitness: float,
    diversity_now: float,
    stage_start_diversity: float,
    stagnation_now: int,
    tier_at_last_decision: int,
    tiers: list[int] | None = None,
    first_call: int = 50,
    min_interval: int = 20,
    max_interval: int = 100,
    df_threshold: float = 0.05,
    dd_threshold: float = 0.02,
    diversity_event_enabled: bool = True,
) -> tuple[bool, str]:
    """评估当前代是否应触发 LLM 决策。

    Args:
        gens_since_decision:      距上次决策的代数（首次调用时 = 当前代数）。
        is_first_call:            是否尚无任何 LLM 决策。
        best_fitness_now:         当前 best fitness。
        stage_start_fitness:      当前 stage 起始 best fitness。
        diversity_now:            当前多样性。
        stage_start_diversity:    当前 stage 起始多样性。
        stagnation_now:           当前真实停滞代数（未封顶）。
        tier_at_last_decision:    上次决策时的停滞档位。
        tiers:                    停滞档位边界列表。
        first_call / min_interval / max_interval: 触发保护约束（代数）。
        df_threshold:             Δf 事件阈值（%）。
        dd_threshold:             ΔD 事件阈值。
        diversity_event_enabled:  是否启用 diversity_move 事件。
                                  上次决策执行了 restart 时应置 False（前馈补偿）：
                                  restart 注入随机个体必然使 diversity 移动，
                                  该移动是执行器自身的效应而非搜索状态变化，
                                  不应作为事件回灌触发器（否则形成
                                  restart → ΔD 事件 → 决策 → restart 的自锁回路）。

    Returns:
        (should_fire, reason) — reason 为人类可读的触发原因，
        会注入 prompt（"为什么现在找你"），空字符串表示不触发。
    """
    # 首次调用：等到有足够历史
    if is_first_call:
        if gens_since_decision >= first_call:
            return True, f"first_call: initial decision at gen {gens_since_decision}"
        return False, ""

    # fallback：太久没有事件也要周期性咨询
    if gens_since_decision >= max_interval:
        return True, (
            f"fallback_timer: no trigger event for {gens_since_decision} gens "
            f"(max_interval={max_interval})"
        )

    # 防抖：最小间隔内不触发
    if gens_since_decision < min_interval:
        return False, ""

    # 事件 1：fitness 显著移动（改进或退化）
    if stage_start_fitness > 0 and abs(stage_start_fitness) > 1e-12:
        df_pct = (stage_start_fitness - best_fitness_now) / abs(stage_start_fitness) * 100.0
    else:
        df_pct = 0.0
    if abs(df_pct) >= df_threshold:
        return True, (
            f"event_fitness_move: best fitness moved {df_pct:+.2f}% "
            f"since last decision (threshold {df_threshold}%)"
        )

    # 事件 2：多样性显著移动（前馈补偿：restart 引起的移动不算事件）
    if diversity_event_enabled:
        dd = diversity_now - stage_start_diversity
        if abs(dd) >= dd_threshold:
            return True, (
                f"event_diversity_move: diversity changed {dd:+.4f} "
                f"since last decision (threshold {dd_threshold})"
            )

    # 事件 3：停滞深化跨档
    tier_now = stagnation_tier(stagnation_now, tiers)
    if tier_now > tier_at_last_decision:
        return True, (
            f"event_stagnation_deepen: stagnation reached {stagnation_now} gens "
            f"(tier {tier_at_last_decision} -> {tier_now})"
        )

    return False, ""
