# -*- coding: utf-8 -*-
"""search_controller.py — 统一 LLM 搜索控制器模块

职责：
    在一次 LLM 调用中，同时决定差分变异策略（DE/rand/1 或 DE/best/2）
    和交叉率 CR。替代原有的分离式 operator_selection + cr_control 模块。

    对应设计规格：
        "LLM jointly determines the differential mutation strategy and
         crossover rate (CR) in a single invocation."

注入点：before_mutation（变异前，决定本轮使用的策略和 CR）
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..base_module import BaseLLMModule, ModuleState

logger = logging.getLogger(__name__)

# 仅允许的两种策略
AVAILABLE_STRATEGIES = {"rand/1", "best/2"}

# 预定义 CR 候选值
CR_CHOICES = [0.1, 0.3, 0.5, 0.7, 0.9]

_SYSTEM_PROMPT = """\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: In a SINGLE invocation, jointly determine:
1. The differential mutation strategy: DE/rand/1 or DE/best/2
2. The crossover rate (CR) from {cr_choices}

The scaling factor F will be automatically computed from your chosen CR \
using the DMDE parameter relationship (formula 3-11).

## Strategy Descriptions
- **rand/1** (exploratory): trial = x_r1 + F * (x_r2 - x_r3)
  Best when population diversity is LOW or stagnation is HIGH.
  Generates diverse offspring by combining random individuals.
- **best/2** (exploitative): trial = best + F * (x_r1 + x_r2 - x_r3 - x_r4)
  Best when population diversity is HIGH but convergence is SLOW.
  Accelerates convergence toward the current best solution.

## CR Selection Guidelines
- Low CR (0.1): More exploitation — smaller perturbations, fine-tuning
- Mid CR (0.5): Balanced exploration/exploitation
- High CR (0.9): More exploration — larger perturbations, broader search

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "strategy": "rand/1" or "best/2",
    "cr": <one of {cr_choices}>,
    "reasoning": "<brief explanation>"
}}
""".format(cr_choices=str(CR_CHOICES))


class LLMSearchControllerModule(BaseLLMModule):
    """统一 LLM 搜索控制器：一次调用决定策略 + CR。"""

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "best_fitness": state.best_fitness,
            "mean_fitness": state.mean_fitness,
            "diversity": round(state.diversity, 4),
            "gene_variance": round(state.gene_variance, 2),
            "convergence_speed": f"{state.convergence_speed:.6f}",
            "stagnation_count": state.stagnation_count,
            "feasible_ratio": round(state.feasible_ratio, 4),
            "current_cr": round(state.cr, 4),
            "current_f": round(state.f_scale, 4),
            "current_strategy": state.extra.get("llm_strategy", "default (CR-based switching)"),
        }

        # 轨迹摘要
        trajectory_text = "No trajectory data yet."
        if state.trajectory_recent:
            lines = []
            for e in state.trajectory_recent[-5:]:
                line = (
                    f"  gen={e.generation}: fitness={e.fitness_best:.1f}, "
                    f"div={e.diversity:.3f}, stag={e.stagnation_count}"
                )
                if e.llm_module:
                    line += f", last_decision={e.llm_module}"
                lines.append(line)
            trajectory_text = "\n".join(lines)

        user = (
            f"## Current Search State\n{json.dumps(features, indent=2)}\n\n"
            f"## Recent Trajectory\n{trajectory_text}\n\n"
            f"## Task\nJointly select the best DE strategy and CR for the next interval. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {
                "strategy": "rand/1",
                "cr": 0.5,
                "reasoning": "Failed to parse LLM output, using defaults",
            }

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "strategy": "rand/1",
                "cr": 0.5,
                "reasoning": "Invalid JSON from LLM, using defaults",
            }

        # 验证策略
        strategy = data.get("strategy", "rand/1")
        if strategy not in AVAILABLE_STRATEGIES:
            strategy = "rand/1"

        # 验证 CR（取最近的合法值）
        cr = data.get("cr", 0.5)
        try:
            cr = float(cr)
        except (ValueError, TypeError):
            cr = 0.5
        # 钳位到最近的合法候选
        cr = min(CR_CHOICES, key=lambda c: abs(c - cr))

        return {
            "strategy": strategy,
            "cr": cr,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将决策应用到搜索状态。

        直接设置 state.cr（不是偏移量），设置 state.extra["llm_strategy"]。
        F 不在此处计算，由求解器根据 LLM 的 CR 通过公式 3-11 计算。
        """
        state.cr = float(decision.get("cr", 0.5))
        state.extra["llm_strategy"] = decision.get("strategy", "rand/1")
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
