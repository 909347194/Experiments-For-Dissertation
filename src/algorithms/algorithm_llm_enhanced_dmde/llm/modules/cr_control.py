# -*- coding: utf-8 -*-
"""cr_control.py — LLM 交叉率/缩放因子控制模块

职责：
    根据搜索状态，由 LLM 动态调整交叉率 CR 和缩放因子 F 的偏移量。
    与 DMDE 的公式 3-9/3-11 叠加使用。

注入点：before_evolve（每代进化前，调整本轮 CR/F）
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
You are an expert in Differential Evolution parameter control.

Your task: Adjust the crossover rate (CR) and scale factor (F) offsets \
based on the current search state.

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "cr_offset": <float in [-0.3, 0.3], or null to keep default>,
    "f_offset": <float in [-0.3, 0.3], or null to keep default>,
    "reasoning": "<brief explanation>"
}}

## Guidelines
- High stagnation + low diversity → increase CR (more exploration)
- Converging well → decrease CR slightly (more exploitation)
- Large fitness variance → increase F (bigger steps)
- Near convergence → decrease F (fine-tuning)
- Only adjust when the state clearly warrants a change (prefer null)
"""


class LLMCRControlModule(BaseLLMModule):
    """LLM 交叉率/缩放因子控制模块。"""

    @property
    def name(self) -> str:
        return "cr_control"

    @property
    def hook_point(self) -> str:
        return "before_evolve"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "current_cr": round(state.cr, 4),
            "current_f": round(state.f_scale, 4),
            "diversity": round(state.diversity, 4),
            "convergence_speed": f"{state.convergence_speed:.6f}",
            "stagnation_count": state.stagnation_count,
            "feasible_ratio": round(state.feasible_ratio, 4),
            "fitness_improvement": state.extra.get("fitness_improvement", 0.0),
        }

        user = (
            f"## Current Search State\n{json.dumps(features, indent=2)}\n\n"
            f"## Task\nAdjust CR and F offsets for the next generation. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {"cr_offset": None, "f_offset": None, "reasoning": "Parse failed"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"cr_offset": None, "f_offset": None, "reasoning": "Invalid JSON"}

        cr_off = self._clamp(data.get("cr_offset"), -0.3, 0.3)
        f_off = self._clamp(data.get("f_offset"), -0.3, 0.3)

        return {
            "cr_offset": cr_off,
            "f_offset": f_off,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        cr_off = decision.get("cr_offset")
        f_off = decision.get("f_offset")

        if cr_off is not None:
            state.cr = float(np.clip(state.cr + cr_off, 0.0, 1.0))
        if f_off is not None:
            state.f_scale = float(np.clip(state.f_scale + f_off, 0.0, 2.0))

        state.extra["llm_cr_offset"] = cr_off
        state.extra["llm_f_offset"] = f_off
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

    @staticmethod
    def _clamp(value: Any, min_val: float, max_val: float) -> float | None:
        if value is None:
            return None
        try:
            return max(min_val, min(max_val, float(value)))
        except (ValueError, TypeError):
            return None
