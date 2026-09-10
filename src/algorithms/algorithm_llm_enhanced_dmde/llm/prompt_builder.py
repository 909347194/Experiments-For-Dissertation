# -*- coding: utf-8 -*-
"""prompt_builder.py — LLM 决策提示组装器

职责：
    将搜索状态特征（种群多样性、收敛速度等）和优化轨迹
    组装为结构化的 Chat Completions 提示，供 LLM 做出
    算子策略选择决策。

提示结构：
    1. System prompt: 角色设定 + 决策规范 + 输出格式要求
    2. User prompt: 当前搜索特征 + 优化轨迹 + 可用算子列表
    3. 要求 LLM 输出 JSON 格式的决策

使用方式::

    builder = PromptBuilder()
    messages = builder.build_decision_prompt(
        features={"diversity": 0.5, "convergence_speed": 0.1, ...},
        trajectory=[{"generation": 1, "fitness": 100.0, ...}, ...],
        available_operators=["rand/1", "best/2", ...],
    )
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# System prompt 模板
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert in evolutionary optimization algorithms, specifically Differential Evolution (DE) \
for combinatorial optimization problems (UAV-target assignment).

Your role: Given the current search state features and optimization trajectory, decide which \
DE operator strategy and parameters to use for the next interval of generations.

## Available Operator Strategies
{available_operators}

## Decision Output Format
You MUST respond with a valid JSON object (no markdown, no extra text) containing:
{{
    "operator_strategy": "<strategy name from available list>",
    "cr_adjustment": <float or null, adjustment to crossover rate in [-0.3, 0.3]>,
    "temperature_adjustment": <float or null, adjustment to temperature in [-0.3, 0.3]>,
    "extinction_trigger": <true/false/null, whether to force trigger extinction>,
    "reasoning": "<brief explanation of your decision>"
}}

## Guidelines
- If diversity is low and stagnation is high, favor exploratory strategies (rand/1, rand/2)
  and consider triggering extinction.
- If convergence is slow but diversity is high, favor exploitative strategies (best/1, best/2)
  and reduce temperature.
- CR adjustment: positive = more exploration, negative = more exploitation.
- Temperature adjustment: positive = more random matching, negative = more greedy matching.
- Only adjust parameters when the current state clearly warrants a change.
"""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """LLM 决策提示组装器。

    将搜索状态特征和优化轨迹组装为 Chat Completions 格式的消息列表。
    """

    def build_decision_prompt(
        self,
        features: dict[str, Any],
        trajectory: list[dict[str, Any]],
        available_operators: list[str],
    ) -> list[dict[str, str]]:
        """构建 LLM 决策提示。

        Args:
            features:            当前搜索状态特征字典。
                                 期望包含: generation, best_fitness, mean_fitness,
                                 diversity, gene_variance, convergence_speed,
                                 stagnation_count, feasible_ratio 等。
            trajectory:          最近 p 代的优化轨迹列表。
                                 每个元素: {"generation": int, "features": dict,
                                 "decision": LLMDecision|None, "fitness": float}。
            available_operators: 可用的 DE 算子策略名称列表。

        Returns:
            Chat Completions 消息列表 [system_msg, user_msg]。
        """
        system_content = _SYSTEM_PROMPT.format(
            available_operators=", ".join(available_operators),
        )

        user_content = self._build_user_message(features, trajectory)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_user_message(
        self,
        features: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> str:
        """构建用户消息内容。

        Args:
            features:   当前搜索状态特征。
            trajectory: 优化轨迹。

        Returns:
            格式化的用户消息文本。
        """
        sections = []

        # 当前搜索特征
        sections.append("## Current Search State Features")
        sections.append(json.dumps(features, indent=2, default=str))

        # 优化轨迹摘要
        if trajectory:
            sections.append("\n## Optimization Trajectory (last {} generations)".format(
                len(trajectory)
            ))
            for entry in trajectory[-5:]:  # 最多展示最近 5 条
                summary = {
                    "generation": entry.get("generation"),
                    "fitness": entry.get("fitness"),
                }
                if entry.get("decision"):
                    d = entry["decision"]
                    summary["operator"] = getattr(d, "operator_strategy", None)
                    summary["reasoning"] = getattr(d, "reasoning", "")[:100]
                sections.append(json.dumps(summary, default=str))
        else:
            sections.append("\n## Optimization Trajectory")
            sections.append("No trajectory data yet (first LLM call).")

        # 决策请求
        sections.append("\n## Task")
        sections.append(
            "Based on the above search state and trajectory, "
            "decide the optimal operator strategy and parameter adjustments "
            "for the next interval. Respond with JSON only."
        )

        return "\n".join(sections)
