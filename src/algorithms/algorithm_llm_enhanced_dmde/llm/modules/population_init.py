# -*- coding: utf-8 -*-
"""population_init.py — LLM 种群初始化模块

职责：
    在种群初始化阶段，由 LLM 基于代价矩阵特征建议初始种群策略，
    或对随机生成的初始种群进行智能优化。

注入点：after_init（种群初始化后，优化初始种群）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from ..base_module import BaseLLMModule, ModuleState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert in evolutionary optimization initialization.

Your task: Given the cost matrix structure and problem characteristics, \
suggest an initialization strategy to improve the initial population quality.

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "init_strategy": "<greedy|diverse|hybrid|random>",
    "temperature": <float in [0.0, 1.0], or null to keep default>,
    "reasoning": "<brief explanation>"
}}

## Strategies
- "greedy": Bias toward low-cost assignments (exploitative start)
- "diverse": Maximize initial diversity (exploratory start)
- "hybrid": Mix of greedy and diverse individuals
- "random": Standard random initialization (no change)

## Guidelines
- Small problem (N<=10) → greedy often works well
- Large problem (N>20) → diverse or hybrid preferred
- Many constraints → hybrid to ensure feasible individuals
"""


class LLMPopulationInitModule(BaseLLMModule):
    """LLM 种群初始化模块。"""

    @property
    def name(self) -> str:
        return "population_init"

    @property
    def hook_point(self) -> str:
        return "after_init"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # 代价矩阵统计
        cm = state.cost_matrix
        cm_stats = {}
        if cm is not None:
            finite_vals = cm[np.isfinite(cm)]
            cm_stats = {
                "shape": list(cm.shape),
                "min": float(finite_vals.min()) if len(finite_vals) else 0,
                "max": float(finite_vals.max()) if len(finite_vals) else 0,
                "mean": float(finite_vals.mean()) if len(finite_vals) else 0,
                "n_infinite": int(np.sum(~np.isfinite(cm))),
            }

        features = {
            "n_uavs": state.n_uavs,
            "n_targets": state.n_targets,
            "model_type": state.model_type,
            "pop_size": state.extra.get("pop_size", 50),
            "cost_matrix_stats": cm_stats,
            "initial_diversity": state.diversity,
            "initial_best_fitness": state.best_fitness,
            "initial_feasible_ratio": state.feasible_ratio,
        }

        user = (
            f"## Problem Characteristics\n{json.dumps(features, indent=2)}\n\n"
            f"## Task\nSuggest the best initialization strategy. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {"init_strategy": "random", "reasoning": "Parse failed"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"init_strategy": "random", "reasoning": "Invalid JSON"}

        strategy = data.get("init_strategy", "random")
        valid = {"greedy", "diverse", "hybrid", "random"}
        if strategy not in valid:
            strategy = "random"

        temp = data.get("temperature")
        if temp is not None:
            try:
                temp = max(0.0, min(1.0, float(temp)))
            except (ValueError, TypeError):
                temp = None

        return {
            "init_strategy": strategy,
            "temperature": temp,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        # 种群初始化模块的 apply 是在 solver 中特殊处理的
        # 这里只记录决策，实际应用在 solver 的初始化阶段
        state.extra["llm_init_strategy"] = decision.get("init_strategy", "random")
        state.extra["llm_init_temperature"] = decision.get("temperature")
        return state

    @staticmethod
    def _extract_json(text: str) -> str | None:
        text = text.strip()
        if text.startswith("{"):
            return text
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return None
