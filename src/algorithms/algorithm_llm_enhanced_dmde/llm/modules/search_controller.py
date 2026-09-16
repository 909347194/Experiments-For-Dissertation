# -*- coding: utf-8 -*-
"""search_controller.py — LLM 搜索控制器模块

职责：
    在一次 LLM 调用中，决定交叉率 CR。LLM 不直接覆盖变异策略，
    CR 天然影响公式 3-10 中 rand/1 vs best/2 的比例（高 CR → 更多 rand/1），
    因此 LLM 的策略思考隐含在 CR 选择里。

注入点：before_mutation（变异前，决定本轮使用的 CR）
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..base_module import BaseLLMModule, ModuleState
from ..prompts import get_prompt

logger = logging.getLogger(__name__)

# 预定义 CR 候选值
CR_CHOICES = [0.1, 0.3, 0.5, 0.7, 0.9]


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器：决定 CR。"""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        # 从配置加载自定义 system prompt，支持外部文件覆盖和动态参数
        prompt_path = self._config.get("system_prompt_path")
        cr_choices = self._config.get("cr_choices", CR_CHOICES)
        self._system_prompt: str = get_prompt(
            "search_controller",
            prompt_type="system",
            prompt_path=prompt_path,
            cr_choices=cr_choices,
        )

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
            "temperature": round(state.temperature, 4),
            "model_type": state.model_type,
            "n_uavs": state.n_uavs,
            "n_targets": state.n_targets,
            "best_fitness": state.best_fitness,
            "mean_fitness": state.mean_fitness,
            "diversity": round(state.diversity, 4),
            "gene_variance": round(state.gene_variance, 2),
            "convergence_speed": f"{state.convergence_speed:.6f}",
            "stagnation_count": state.stagnation_count,
            "feasible_ratio": round(state.feasible_ratio, 4),
            "violation_mean": round(state.violation_mean, 6),
            "violation_max": round(state.violation_max, 6),
            "current_cr": round(state.cr, 4),
            "current_f": round(state.f_scale, 4),
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

        # 使用统一的 user prompt 模板
        user = get_prompt(
            "search_controller",
            prompt_type="user",
            state_json=json.dumps(features, indent=2),
            trajectory_text=trajectory_text,
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {
                "cr": 0.5,
                "reasoning": "Failed to parse LLM output, using defaults",
            }

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "cr": 0.5,
                "reasoning": "Invalid JSON from LLM, using defaults",
            }

        # 验证 CR（取最近的合法值）
        cr = data.get("cr", 0.5)
        try:
            cr = float(cr)
        except (ValueError, TypeError):
            cr = 0.5
        # 钳位到最近的合法候选
        cr = min(CR_CHOICES, key=lambda c: abs(c - cr))

        return {
            "cr": cr,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将决策应用到搜索状态。

        直接设置 state.cr（不是偏移量）。
        F 不在此处计算，由求解器根据 LLM 的 CR 通过公式 3-11 计算。
        LLM 不直接覆盖 strategy；CR 天然影响公式 3-10 中 rand/1 vs best/2 的比例。
        """
        state.cr = float(decision.get("cr", 0.5))
        return state


