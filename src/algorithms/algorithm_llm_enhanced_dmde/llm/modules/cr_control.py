# -*- coding: utf-8 -*-
"""cr_control.py — LLM 交叉率/缩放因子控制模块

职责：
    根据搜索状态 + 上次决策反馈，由 LLM 动态调整 CR/F。
    闭环控制：每次决策后记录 CR 和效果，下次决策时提供反馈。

注入点：before_evolve（每代进化前，按 interval 触发）
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from ..base_module import BaseLLMModule, ModuleState
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


class LLMCRControlModule(BaseLLMModule):
    """LLM 交叉率/缩放因子控制模块（闭环反馈版）。"""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        prompt_path = self._config.get("system_prompt_path")
        self._system_prompt: str = get_prompt(
            "cr_control",
            prompt_path=prompt_path,
        )

    @property
    def name(self) -> str:
        return "cr_control"

    @property
    def hook_point(self) -> str:
        return "before_evolve"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # 当前搜索状态
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

        # 上次 LLM CR 决策的反馈（闭环控制核心）
        prev_cr = state.extra.get("previous_llm_cr")
        feedback = {
            "previous_llm_cr": round(prev_cr, 4) if prev_cr is not None else None,
            "previous_interval_gens": state.extra.get("previous_interval_gens"),
            "fitness_change": round(state.extra.get("fitness_change_since_last", 0.0), 2),
            "diversity_change": round(state.extra.get("diversity_change_since_last", 0.0), 4),
        }

        full_state = {**features, "previous_decision_feedback": feedback}

        from llm.prompts import get_prompt
        user = get_prompt(
            "cr_control",
            prompt_type="user",
            state_json=json.dumps(full_state, indent=2),
        )

        return [
            {"role": "system", "content": self._system_prompt},
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
    def _clamp(value: Any, min_val: float, max_val: float) -> float | None:
        if value is None:
            return None
        try:
            return max(min_val, min(max_val, float(value)))
        except (ValueError, TypeError):
            return None