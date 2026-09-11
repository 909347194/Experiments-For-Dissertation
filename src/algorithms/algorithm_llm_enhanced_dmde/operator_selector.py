# -*- coding: utf-8 -*-
"""operator_selector.py — LLM 算子选择器（对齐 LLM-MOEA）

职责：
    根据优化轨迹和状态特征，让 LLM 从预定义算子池中选择
    最合适的交叉/变异/灭绝算子。

对应论文：
    Zhang et al. (2025) "Leveraging Large Language Models for
    Dynamic Multi-Objective Optimization in UAV Sensor-Target Assignment"
    Section 3: LLM-MOEA 框架 — LLM 作为算子选择器。

核心思路：
    LLM 不生成解、不修复解，只从有限算子池中"选择"。
    这是 ReAct 模式：LLM 观察状态 → 思考 → 选择行动（算子）。
    我们负责解析 LLM 输出 → 映射到实际算子调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .llm_advisor import LLMAdvisor, LLMConfig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 算子池定义（有限集合，LLM 只能从中选择）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPERATOR_POOL = {
    "crossover": {
        "rand_1": {
            "name": "DE/rand/1",
            "desc": "随机策略，强探索性",
            "formula": "v = x_r1 + F*(x_r2 - x_r3)",
        },
        "best_2": {
            "name": "DE/best/2",
            "desc": "最优策略，强开发性",
            "formula": "v = x_best + F*(x_r1 + x_r2 - x_r3 - x_r4)",
        },
        "rand_to_best_1": {
            "name": "DE/rand-to-best/1",
            "desc": "平衡策略，兼顾探索与开发",
            "formula": "v = x_i + k*(x_best - x_i) + F*(x_r1 - x_r2)",
        },
    },
    "mutation": {
        "dynamic": {
            "name": "动态 F",
            "desc": "F 随 CR 动态变化（公式 3-11）",
        },
        "constant_08": {
            "name": "常数 F=0.8",
            "desc": "经典固定缩放因子",
        },
        "adaptive": {
            "name": "自适应 F",
            "desc": "根据种群多样性自适应调整",
        },
    },
    "extinction": {
        "gmr": {
            "name": "GMR 灭绝",
            "desc": "代间变异率灭绝（公式 3-12）",
        },
        "none": {
            "name": "不灭绝",
            "desc": "正常进化，不触发灭绝",
        },
        "random_restart": {
            "name": "随机重启",
            "desc": "保留最优，其余全部随机重置",
        },
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 状态特征提取
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OptimizationState:
    """优化状态特征（LLM 的观察输入）。

    对应论文 Fig.2 中的 "optimization trajectory" + "state features"。
    """

    current_gen: int
    total_gens: int
    best_fitness: float
    cost_history: list[float]
    stagnation_gens: int         # 连续无改善代数
    improvement_rate: float      # 近期改善率
    population_diversity: float  # 种群多样性（0~1）
    constraint_violation: float  # 约束违背率
    current_crossover: str       # 当前使用的交叉策略
    current_mutation: str        # 当前使用的变异策略

    def to_prompt_dict(self) -> dict:
        """转为 LLM prompt 可用的 dict。"""
        return {
            "current_gen": self.current_gen,
            "total_gens": self.total_gens,
            "progress_pct": round(self.current_gen / self.total_gens * 100, 1),
            "best_fitness": round(self.best_fitness, 2),
            "stagnation_gens": self.stagnation_gens,
            "improvement_rate_pct": round(self.improvement_rate * 100, 3),
            "population_diversity": round(self.population_diversity, 3),
            "constraint_violation": round(self.constraint_violation, 4),
            "current_crossover": self.current_crossover,
            "current_mutation": self.current_mutation,
        }


def extract_state(
    cost_history: list[float],
    population_vectors: np.ndarray,
    current_gen: int,
    total_gens: int,
    current_crossover: str = "rand_1",
    current_mutation: str = "dynamic",
    total_violation: float = 0.0,
) -> OptimizationState:
    """从进化过程中提取状态特征。

    Args:
        cost_history: 最优适应度历史。
        population_vectors: 当前种群代价值矩阵。
        current_gen: 当前代数。
        total_gens: 总代数。
        current_crossover: 当前交叉策略名。
        current_mutation: 当前变异策略名。
        total_violation: 约束违背总量。

    Returns:
        OptimizationState 实例。
    """
    # 改善率
    if len(cost_history) >= 20:
        recent = cost_history[-20:]
        improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
    else:
        improvement = 0.0

    # 停滞代数
    stagnation = 0
    if len(cost_history) >= 2:
        for i in range(len(cost_history) - 1, 0, -1):
            if abs(cost_history[i] - cost_history[i - 1]) < 1e-6:
                stagnation += 1
            else:
                break

    # 种群多样性：各行之间的平均标准差
    if population_vectors.ndim == 2 and population_vectors.shape[0] > 1:
        diversity = float(np.mean(np.std(population_vectors, axis=0)))
        # 归一化到 [0, 1]
        range_val = float(np.max(population_vectors) - np.min(population_vectors))
        diversity = diversity / (range_val + 1e-10)
    else:
        diversity = 0.0

    return OptimizationState(
        current_gen=current_gen,
        total_gens=total_gens,
        best_fitness=cost_history[-1] if cost_history else float("inf"),
        cost_history=cost_history[-50:],  # 只发最近50代
        stagnation_gens=stagnation,
        improvement_rate=improvement,
        population_diversity=diversity,
        constraint_violation=total_violation,
        current_crossover=current_crossover,
        current_mutation=current_mutation,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM 算子选择器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OperatorChoice:
    """LLM 的算子选择结果。

    Attributes:
        crossover: 选择的交叉算子名（算子池中的 key）。
        mutation:  选择的变异算子名。
        extinction: 选择的灭绝策略名。
        reason:    选择理由。
        confidence: LLM 的置信度（0~1）。
    """

    crossover: str = "rand_1"
    mutation: str = "dynamic"
    extinction: str = "gmr"
    reason: str = ""
    confidence: float = 0.5

    def validate(self) -> bool:
        """验证选择是否在算子池内。"""
        return (
            self.crossover in OPERATOR_POOL["crossover"]
            and self.mutation in OPERATOR_POOL["mutation"]
            and self.extinction in OPERATOR_POOL["extinction"]
        )


class LLMOperatorSelector:
    """LLM 算子选择器（对齐 LLM-MOEA 框架）。

    核心流程（ReAct 模式）：
        1. 观察：提取优化状态特征
        2. 思考：LLM 分析状态，推理最佳算子
        3. 行动：从算子池中选择算子
        4. 解析：我们提取 JSON，验证合法性
        5. 执行：映射到实际算子调用

    使用方式::
        selector = LLMOperatorSelector(llm_config)
        choice = selector.select(state)
        # 将 choice 映射到 DMDE 算子
    """

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self._advisor = LLMAdvisor(llm_config)
        self._history: list[dict] = []  # 记录每次选择

    def select(self, state: OptimizationState) -> OperatorChoice:
        """让 LLM 选择算子。

        Args:
            state: 当前优化状态。

        Returns:
            OperatorChoice 实例。
        """
        state_dict = state.to_prompt_dict()

        # 构建算子池描述
        pool_desc = self._format_operator_pool()

        prompt = f"""你是差分进化算法的算子选择器。根据当前优化状态，从算子池中选择最合适的算子。

## 当前优化状态
{state_dict}

## 算子池
{pool_desc}

## 选择规则
- 收敛停滞 (stagnation > 10代): 增大探索性 → 选 rand_1, 建议触发灭绝
- 多样性低 (diversity < 0.1): 增大探索性 → 选 rand_1, 选 random_restart
- 接近尾声 (progress > 80%): 增大开发性 → 选 best_2, 不灭绝
- 约束违背高 (violation > 0.1): 增大探索性 → 选 rand_1
- 正常情况: 选 rand_to_best_1 平衡策略

请用 JSON 格式回答（只选算子池中有的 key）:
{{"crossover": "rand_1|best_2|rand_to_best_1", "mutation": "dynamic|constant_08|adaptive", "extinction": "gmr|none|random_restart", "reason": "选择理由", "confidence": 0.0~1.0}}"""

        result = self._advisor.ask_json(
            prompt,
            system="你是差分进化算法算子选择专家。只从给定算子池中选择，不要创造新算子。"
        )

        choice = self._parse_choice(result)
        self._history.append({
            "state": state_dict,
            "choice": {
                "crossover": choice.crossover,
                "mutation": choice.mutation,
                "extinction": choice.extinction,
                "reason": choice.reason,
            },
        })
        return choice

    def _parse_choice(self, result: dict) -> OperatorChoice:
        """解析 LLM 输出，验证合法性。"""
        if not result:
            return OperatorChoice(reason="LLM 未返回有效结果")

        choice = OperatorChoice(
            crossover=result.get("crossover", "rand_1"),
            mutation=result.get("mutation", "dynamic"),
            extinction=result.get("extinction", "gmr"),
            reason=result.get("reason", ""),
            confidence=float(result.get("confidence", 0.5)),
        )

        # 验证：不在算子池内的 key 回退到默认值
        if choice.crossover not in OPERATOR_POOL["crossover"]:
            choice.crossover = "rand_1"
        if choice.mutation not in OPERATOR_POOL["mutation"]:
            choice.mutation = "dynamic"
        if choice.extinction not in OPERATOR_POOL["extinction"]:
            choice.extinction = "gmr"

        return choice

    def _format_operator_pool(self) -> str:
        """格式化算子池描述。"""
        lines = []
        for category, operators in OPERATOR_POOL.items():
            lines.append(f"### {category}")
            for key, info in operators.items():
                lines.append(f"  - {key}: {info['name']} — {info['desc']}")
        return "\n".join(lines)

    @property
    def history(self) -> list[dict]:
        """选择历史。"""
        return self._history
