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
CR_CHOICES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器：决定 CR。"""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")
        self._cr_choices = self._config.get("cr_choices", CR_CHOICES)
        self._system_prompt: str | None = None  # 缓存，按 model_type 分别解析

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # ---- State: Δ 趋势信号 ----
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "temperature": round(state.temperature, 4),
            "model_type": state.model_type,
            # 当前水平
            "best_fitness": round(state.best_fitness, 2),
            "diversity": round(state.diversity, 4),
            "stagnation_count": state.stagnation_count,
            "feasible_ratio": round(state.feasible_ratio, 4),
            # Δ 趋势信号
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
            # 上次决策及效果
            "prev_action_cr": state.prev_action,
            "prev_delta_fitness_pct": state.prev_delta_fitness,
            "prev_delta_diversity": state.prev_delta_diversity,
        }

        # ---- Stage 级轨迹表格（最近 5 个 stage） ----
        trajectory_text = "No stage history yet (first decision)."
        stage_hist = state.stage_history
        if stage_hist:
            lines = ["Stage | CR | Best Fitness | Δf(%) | Diversity | ΔD"]
            for s in stage_hist[-5:]:
                lines.append(
                    f"{s['stage']:5d} | {s['cr']:.1f} | "
                    f"{s['best_fitness']:12.1f} | "
                    f"{s['delta_fitness']:+6.2f}% | "
                    f"{s['diversity']:.4f} | {s['delta_diversity']:+.4f}"
                )
            trajectory_text = "\n".join(lines)

        # ---- Action-Outcome 反馈 ----
        feedback = ""
        if state.prev_action is not None:
            fitness_desc = (
                "improved" if state.delta_fitness > 0.1
                else "stagnated" if abs(state.delta_fitness) <= 0.1
                else "degraded"
            )
            div_desc = (
                "increased" if state.prev_delta_diversity > 0.001
                else "stable" if abs(state.prev_delta_diversity) <= 0.001
                else "decreased"
            )
            # 从 stage_history 取上次决策的代数
            prev_gen_str = ""
            if stage_hist:
                prev_gen_str = f" at gen {stage_hist[-1].get('gen_end', '?')}"
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"You chose CR={state.prev_action:.1f}{prev_gen_str}\n"
                f"Outcome: fitness {fitness_desc} ({state.delta_fitness:+.2f}%), "
                f"diversity {div_desc} ({state.prev_delta_diversity:+.4f})\n"
                f"Interpretation: {'Exploration was effective — consider maintaining or increasing CR.' if state.delta_fitness > 0.1 and state.prev_delta_diversity > 0 else 'Exploitation paid off — consider maintaining low CR.' if state.delta_fitness > 0.1 else 'Current strategy stalled — try a different CR to shift the balance.'}"
            )

        user = get_prompt(
            "search_controller",
            prompt_type="user",
            state_json=json.dumps(features, indent=2),
            trajectory_text=trajectory_text + feedback,
        )

        if self._system_prompt is None:
            self._system_prompt = get_prompt(
                "search_controller",
                prompt_type="system",
                prompt_path=self._prompt_path,
                model_type=state.model_type,
                cr_choices=self._cr_choices,
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


