# -*- coding: utf-8 -*-
"""parameter_advisor.py — LLM 动态参数调控

职责：
    根据进化过程中的收敛状态，让 LLM 建议参数调整策略。

LLM 介入点：
    - 分析收敛曲线趋势（是否停滞、是否过快收敛）
    - 建议 F/CR/种群规模的调整方向
    - 建议是否触发灭绝操作

LLM 不介入：
    - 不直接修改种群个体
    - 不参与反映射修复
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .llm_advisor import LLMAdvisor, LLMConfig


@dataclass
class ParameterSuggestion:
    """LLM 的参数调整建议。

    Attributes:
        scale_factor_delta: F 调整量（正=增大，负=减小）。
        crossover_rate_delta: CR 调整量。
        trigger_extinction: 是否建议触发灭绝。
        reason: 建议理由。
    """

    scale_factor_delta: float = 0.0
    crossover_rate_delta: float = 0.0
    trigger_extinction: bool = False
    reason: str = ""


class LLMParameterAdvisor:
    """LLM 参数调控顾问。

    使用方式::
        advisor = LLMParameterAdvisor(llm_config)
        suggestion = advisor.suggest(cost_history, current_gen, total_gens)
        # 根据 suggestion 调整 DMDE 参数
    """

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self._advisor = LLMAdvisor(llm_config)

    def suggest(
        self,
        cost_history: list[float],
        current_gen: int,
        total_gens: int,
        current_cr: float,
        current_f: float,
    ) -> ParameterSuggestion:
        """根据收敛状态建议参数调整。

        Args:
            cost_history: 迄今为止的最优适应度历史。
            current_gen: 当前代数。
            total_gens: 总代数。
            current_cr: 当前交叉率。
            current_f: 当前缩放因子。

        Returns:
            ParameterSuggestion 实例。
        """
        # 计算收敛指标
        if len(cost_history) < 10:
            return ParameterSuggestion(reason="代数不足，暂不调整")

        recent = cost_history[-20:]
        improvement = (recent[0] - recent[-1]) / (abs(recent[0]) + 1e-10)
        stagnation_count = sum(1 for i in range(1, len(recent)) if abs(recent[i] - recent[i-1]) < 1e-6)

        stats = {
            "current_gen": current_gen,
            "total_gens": total_gens,
            "progress_pct": round(current_gen / total_gens * 100, 1),
            "recent_improvement_pct": round(improvement * 100, 3),
            "stagnation_generations": stagnation_count,
            "current_cr": round(current_cr, 4),
            "current_f": round(current_f, 4),
            "best_fitness": round(cost_history[-1], 2),
        }

        prompt = f"""分析 DMDE 算法的收敛状态，建议参数调整。

当前状态:
{stats}

规则:
- 如果收敛停滞 (>15代无改善)，建议增大 F 或触发灭绝
- 如果收敛过快 (可能陷入局部最优)，建议增大 CR 提高探索性
- 如果接近尾声 (>80%代数)，建议减小 F 精细搜索

请用 JSON 格式回答:
{{"scale_factor_delta": float, "crossover_rate_delta": float, "trigger_extinction": bool, "reason": "str"}}"""

        result = self._advisor.ask_json(prompt, system="你是差分进化算法调参专家。")
        if not result:
            return ParameterSuggestion(reason="LLM 未返回有效建议")

        return ParameterSuggestion(
            scale_factor_delta=float(result.get("scale_factor_delta", 0)),
            crossover_rate_delta=float(result.get("crossover_rate_delta", 0)),
            trigger_extinction=bool(result.get("trigger_extinction", False)),
            reason=result.get("reason", ""),
        )
