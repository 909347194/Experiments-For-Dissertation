# -*- coding: utf-8 -*-
"""v2 闭环控制改造的单元测试。

覆盖：
- 事件触发器 evaluate_trigger / stagnation_tier 的全部触发分支
- detect_stagnation 去饱和（真实停滞代数，不再封顶在 10）
- compute_diversity_quantiles 的空间结构刻画
- search_controller 的 restart_fraction 解析与钳位
- solver 的 _downsample_curve / _apply_restart 辅助逻辑
"""
import numpy as np
import pytest

from algorithms.algorithm_llm_enhanced_dmde.features.trigger import (
    evaluate_trigger, stagnation_tier, DEFAULT_STAG_TIERS,
)
from algorithms.algorithm_llm_enhanced_dmde.features.convergence_features import detect_stagnation
from algorithms.algorithm_llm_enhanced_dmde.features.population_features import (
    compute_diversity_quantiles,
)


# ── stagnation_tier ─────────────────────────────────────────

class TestStagnationTier:
    def test_tier_boundaries(self):
        assert stagnation_tier(0) == 0
        assert stagnation_tier(4) == 0
        assert stagnation_tier(5) == 1
        assert stagnation_tier(10) == 2
        assert stagnation_tier(15) == 2
        assert stagnation_tier(160) == 6
        assert stagnation_tier(5000) == 6  # 超出最高档不再升级（档数有限）

    def test_custom_tiers(self):
        assert stagnation_tier(7, tiers=[3, 7]) == 2
        assert stagnation_tier(2, tiers=[3, 7]) == 0


# ── evaluate_trigger ────────────────────────────────────────

class TestEvaluateTrigger:
    """触发器分支覆盖。公共基线状态：无事件、stage 内稳定。"""

    BASE = dict(
        gens_since_decision=50,
        is_first_call=False,
        best_fitness_now=100.0,
        stage_start_fitness=100.0,
        diversity_now=0.60,
        stage_start_diversity=0.60,
        stagnation_now=10,
        tier_at_last_decision=2,
    )

    def test_first_call_waits_for_history(self):
        fire, reason = evaluate_trigger(**{**self.BASE, "is_first_call": True,
                                           "gens_since_decision": 30})
        assert not fire
        fire, reason = evaluate_trigger(**{**self.BASE, "is_first_call": True,
                                           "gens_since_decision": 50})
        assert fire and "first_call" in reason

    def test_fallback_timer(self):
        fire, reason = evaluate_trigger(**{**self.BASE, "gens_since_decision": 100})
        assert fire and "fallback" in reason

    def test_min_interval_debounce(self):
        # 事件已发生（fitness 移动 10%）但间隔不足 → 不触发
        fire, _ = evaluate_trigger(**{**self.BASE, "gens_since_decision": 10,
                                      "best_fitness_now": 90.0})
        assert not fire

    def test_fitness_move_event(self):
        fire, reason = evaluate_trigger(**{**self.BASE, "best_fitness_now": 95.0})
        assert fire and "fitness_move" in reason and "+5.00%" in reason
        # 退化也应触发（fitness 变差）
        fire, reason = evaluate_trigger(**{**self.BASE, "best_fitness_now": 110.0})
        assert fire and "fitness_move" in reason

    def test_fitness_below_threshold_is_no_event(self):
        # Δf = 0.04% < 0.05 阈值 → 不触发（这正是旧设计里被当作信号的噪声）
        fire, _ = evaluate_trigger(**{**self.BASE, "best_fitness_now": 99.96})
        assert not fire

    def test_diversity_move_event(self):
        fire, reason = evaluate_trigger(**{**self.BASE, "diversity_now": 0.65})
        assert fire and "diversity_move" in reason
        # 0.01 < 0.02 噪声 → 不触发
        fire, _ = evaluate_trigger(**{**self.BASE, "diversity_now": 0.61})
        assert not fire

    def test_stagnation_tier_deepen_event(self):
        # 停滞从 tier 2 深化到 tier 3（跨过 20）→ 触发
        fire, reason = evaluate_trigger(**{**self.BASE, "stagnation_now": 22})
        assert fire and "stagnation_deepen" in reason
        # 同档内增长（10 → 15，都在 tier 2）→ 不触发
        fire, _ = evaluate_trigger(**{**self.BASE, "stagnation_now": 15})
        assert not fire

    def test_quiet_state_no_fire(self):
        # 收敛后的典型状态：Δf=0、ΔD≈0、停滞同档 → 不触发（这正是旧设计空转的场景）
        fire, reason = evaluate_trigger(**self.BASE)
        assert not fire

    def test_diversity_event_masked_after_restart(self):
        # 前馈补偿：restart 后 diversity 大幅移动也不触发
        # （否则 restart → ΔD 事件 → 决策 → restart 形成自锁回路）
        fire, _ = evaluate_trigger(
            **{**self.BASE, "diversity_now": 0.80},
            diversity_event_enabled=False,
        )
        assert not fire
        # fitness 事件不受 mask 影响
        fire, reason = evaluate_trigger(
            **{**self.BASE, "best_fitness_now": 90.0},
            diversity_event_enabled=False,
        )
        assert fire and "fitness_move" in reason


# ── detect_stagnation 去饱和 ────────────────────────────────

class TestDetectStagnationUnsaturated:
    def test_no_cap_at_10(self):
        flat = [100.0] * 500
        assert detect_stagnation(flat) == 499  # 真实值，旧版会封顶在 10

    def test_reset_on_improvement(self):
        hist = [100.0] * 50 + [99.0] + [99.0] * 30
        assert detect_stagnation(hist) == 30


# ── diversity quantiles ─────────────────────────────────────

class _FakeInd:
    def __init__(self, v):
        self.cost_vector = np.array(v, dtype=float)


class TestDiversityQuantiles:
    def test_clustered_vs_outlier(self):
        # 9 个聚成一团 + 1 个远端离群个体
        clustered = [_FakeInd([0.0, 0.0]) for _ in range(9)]
        outlier = _FakeInd([10.0, 0.0])
        p25, p90 = compute_diversity_quantiles(clustered + [outlier], q=(0.25, 0.9))
        # 36/45 的成对距离为 0（团内），9/45 为 1（离群对）：
        # p25 应为 0（团内结构），p90 被离群对撑到 1
        assert p25 == 0.0
        assert p90 == 1.0

    def test_uniform(self):
        pop = [_FakeInd([float(i), 0.0]) for i in range(10)]
        p25, p75 = compute_diversity_quantiles(pop)
        assert 0.0 <= p25 <= 1.0 and 0.0 <= p75 <= 1.0
        assert p75 > p25  # 均匀分布上分位高于下分位

    def test_single_individual(self):
        assert compute_diversity_quantiles([_FakeInd([1.0])]) == (0.0, 0.0)


# ── search_controller 解析 ─────────────────────────────────

class TestSearchControllerParse:
    def _make_module(self):
        from algorithms.algorithm_llm_enhanced_dmde.llm.modules.search_controller import (
            LLMSearchControllerModule,
        )
        return LLMSearchControllerModule(llm_client=object(), config={})

    def test_parse_full_decision(self):
        m = self._make_module()
        d = m.parse_response('{"cr": 0.65, "restart_fraction": 0.15, "reasoning": "r"}')
        assert d["cr"] == 0.6 or d["cr"] == 0.7  # 钳位到最近候选
        assert d["restart_fraction"] in (0.1, 0.2)
        # 旧 schema 只给 cr → 隐式 set（向后兼容）
        assert d["cr_action"] == "set_implicit"

    def test_parse_missing_restart_defaults_zero(self):
        m = self._make_module()
        d = m.parse_response('{"cr": 0.5, "reasoning": "keep"}')
        assert d["restart_fraction"] == 0.0

    def test_parse_invalid_json_defaults_to_hold(self):
        """v3 契约：解析失败回退 hold（cr=None），不再默认给一个新 CR 值。

        旧版回退 0.5 等于默认翻转 —— 那正是实测到的 corr(CR_t, CR_{t-1})=-0.967
        无条件翻转缺陷的组成部分。
        """
        m = self._make_module()
        d = m.parse_response("not json at all")
        assert d["cr_action"] == "hold"
        assert d["cr"] is None
        assert d["restart_fraction"] == 0.0


# ── solver 辅助逻辑 ─────────────────────────────────────────

class TestSolverHelpers:
    def _solver(self):
        from algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
            LLMEnhancedDMDESolver,
        )
        return LLMEnhancedDMDESolver()

    def test_downsample_curve(self):
        curve = [100.0] + [100.0 - i * 0.1 for i in range(50)]
        pts = self._solver()._downsample_curve(curve, max_points=12)
        assert len(pts) <= 12
        assert pts[0] == [1, 0.0]
        # 单调改进 → Δ% 递增
        deltas = [p[1] for p in pts]
        assert all(b >= a for a, b in zip(deltas, deltas[1:]))

    def test_downsample_empty(self):
        assert self._solver()._downsample_curve([]) == []

    def test_apply_restart(self):
        from algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
            LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig,
        )
        solver = LLMEnhancedDMDESolver(LLMEnhancedDMDEConfig(seed=42))

        class _Ind:
            def __init__(self, fit, cv):
                self.fitness = fit
                self.cost_vector = cv
                self.assignment = {"u": 0}

        class _Enc:
            def generate(self, k, seed=None):
                return [_Ind(200.0 + i, np.zeros(3)) for i in range(k)]

        class _Ev:
            def evaluate(self, assignment, cost_matrix, n_uavs=None):
                class R: fitness = 200.0
                return R()

        # best=10（个体 0），最差为 100/99/98...
        pop = [_Ind(10.0 if i == 0 else 100.0 - i, np.zeros(3)) for i in range(10)]
        best = pop[0]
        pop2, best2, idx2 = solver._apply_restart(
            pop, best, 0, 0.2, _Enc(), 1, _Ev(), None, 1, LLMEnhancedDMDEConfig(seed=42),
        )
        assert len(pop2) == 10
        # 最差 2 个（fitness 99, 98 → 个体 1、2 附近的两个）被替换为 200
        replaced = sum(1 for ind in pop2 if ind.fitness == 200.0)
        assert replaced == 2
        # best 不受影响
        assert best2.fitness == 10.0 and idx2 == 0
