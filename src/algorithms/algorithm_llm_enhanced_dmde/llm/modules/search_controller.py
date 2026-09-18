# -*- coding: utf-8 -*-
"""search_controller.py — LLM 搜索控制器模块

职责：
    在一次 LLM 调用中，决定交叉率 CR 和重启比例 restart_fraction。
    F 完全由 DMDE 公式 3-11 从 CR 自动推导，LLM 不控制 F。

    v2 闭环控制改造（可观测 / 可归因 / 可触发 / 能控性）：
    - prompt 注入 stage 级过程统计（接受率、改进次数、停滞真实值、
      多样性分位数、stage 内 best 曲线），不再只喂两个聚合标量；
    - prompt 注入影子对照（固定 CR 的影子种群在同一 stage 的 Δf/ΔD），
      使 LLM 的动作效果可以与"什么都不调"的基线分离 —— 可归因；
    - prompt 注入 trigger_reason（为什么现在被咨询）；
    - 动作空间扩展 restart_fraction：收敛后 CR 失效时，
      允许 LLM 重启最差个体比例，恢复搜索能力 —— 能控性。

注入点：before_mutation（变异前，由 solver 的事件触发器决定何时调用）
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
# 预定义重启比例候选值（替换最差个体的比例）
RESTART_CHOICES = [0.0, 0.1, 0.2, 0.3]


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器：决定 CR 与 restart_fraction。"""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")
        self._cr_choices = self._config.get("cr_choices", CR_CHOICES)
        self._restart_choices = self._config.get(
            "restart_choices",
            self._config.get("actions", {}).get("restart_choices", RESTART_CHOICES),
        )
        self._system_prompt: str | None = None  # 缓存，按 model_type 分别解析

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # ---- State: S_t（可观测性扩展版） ----
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "model_type": state.model_type,
            # D_t: 当前多样性水平 + 空间结构分位数
            "diversity": round(state.diversity, 4),
            "diversity_p25": round(state.diversity_p25, 4),
            "diversity_p75": round(state.diversity_p75, 4),
            # Δf_t / ΔD_t: 当前 stage 的趋势
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
            # stage 内过程统计
            "stage_length": state.stage_length,
            "acceptance_rate": (
                round(state.acceptance_rate, 4)
                if state.acceptance_rate is not None else None
            ),
            "improvements_in_stage": state.improvements_in_stage,
            "gens_since_last_improvement": state.gens_since_last_improvement,
            # 停滞（真实值，未封顶）
            "stagnation_count": state.stagnation_count,
            "stagnation_raw": state.stagnation_raw,
            # 上次动作及其效果
            "prev_action_cr": state.prev_action,
            "prev_action_restart": state.prev_action_restart,
            "prev_delta_fitness_pct": state.prev_delta_fitness,
            "prev_delta_diversity": state.prev_delta_diversity,
            # stage 内逐代 best 曲线（降采样，携带 stage 内动态）
            "stage_best_curve": state.stage_best_curve,
            # 本次被触发的原因
            "trigger_reason": state.trigger_reason or None,
        }

        # ---- 影子对照（可归因锚点） ----
        if state.shadow_cr is not None:
            features["shadow_control"] = {
                "shadow_cr": round(state.shadow_cr, 2),
                "shadow_delta_fitness_pct": (
                    round(state.shadow_delta_fitness, 4)
                    if state.shadow_delta_fitness is not None else None
                ),
                "shadow_delta_diversity": (
                    round(state.shadow_delta_diversity, 4)
                    if state.shadow_delta_diversity is not None else None
                ),
                "note": (
                    "A shadow population evolved from the SAME starting point "
                    "with FIXED CR. Deltas are over the same stage window."
                ),
            }

        # ---- Stage 级轨迹表格（最近 5 个 stage） ----
        trajectory_text = "No stage history yet (first decision)."
        stage_hist = state.stage_history
        if stage_hist:
            lines = [
                "Stage | Len | CR | Restart | Best Fitness | "
                "df(%) | df_shadow(%) | dD | Accept%"
            ]
            for s in stage_hist[-5:]:
                df = (
                    f"{s['delta_fitness']:+7.2f}%"
                    if s.get("delta_fitness") is not None else "     N/A"
                )
                dfs = (
                    f"{s.get('shadow_delta_fitness', float('nan')):+7.2f}%"
                    if s.get("shadow_delta_fitness") is not None else "     N/A"
                )
                dd = (
                    f"{s['delta_diversity']:+.4f}"
                    if s.get("delta_diversity") is not None else "  N/A"
                )
                acc = (
                    f"{s['acceptance_rate'] * 100:5.1f}"
                    if s.get("acceptance_rate") is not None else "  N/A"
                )
                rst = s.get("restart_fraction", 0.0)
                lines.append(
                    f"{s['stage']:5d} | {s.get('stage_length', '?'):3} | "
                    f"{s['cr']:.1f} | {rst:<7.1f} | "
                    f"{s['best_fitness']:12.1f} | "
                    f"{df} | {dfs} | {dd} | {acc}"
                )
            trajectory_text = "\n".join(lines)

        # ---- Action-Outcome 反馈（含影子对照，可归因） ----
        feedback = ""
        if state.prev_action is None:
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"First decision: no previous LLM action/outcome available.\n"
                f"No prior CR context to evaluate. Choose based on current state."
            )
        else:
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"You set CR={state.prev_action:.1f}, "
                f"restart_fraction={state.prev_action_restart:.1f}. "
                f"The observed stage outcome was "
                f"df={state.delta_fitness:+.2f}%, "
                f"dD={state.delta_diversity:+.4f}."
            )
            if state.shadow_delta_fitness is not None:
                feedback += (
                    f"\nOver the same stage, the fixed-CR shadow achieved "
                    f"df_shadow={state.shadow_delta_fitness:+.2f}%, "
                    f"dD_shadow={state.shadow_delta_diversity:+.4f}. "
                    f"The difference (yours minus shadow) is the effect "
                    f"attributable to your CR choice."
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
                restart_choices=self._restart_choices,
                shadow_cr=state.shadow_cr,
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
                "restart_fraction": 0.0,
                "reasoning": "Failed to parse LLM output, using defaults",
            }

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "cr": 0.5,
                "restart_fraction": 0.0,
                "reasoning": "Invalid JSON from LLM, using defaults",
            }

        # 验证 CR（钳位到最近的合法候选）
        cr = data.get("cr", 0.5)
        try:
            cr = float(cr)
        except (ValueError, TypeError):
            cr = 0.5
        cr = min(self._cr_choices, key=lambda c: abs(c - cr))

        # 验证 restart_fraction（钳位到最近的合法候选）
        rf = data.get("restart_fraction", 0.0)
        try:
            rf = float(rf)
        except (ValueError, TypeError):
            rf = 0.0
        rf = min(self._restart_choices, key=lambda r: abs(r - rf))

        return {
            "cr": cr,
            "restart_fraction": rf,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将决策应用到搜索状态。

        直接设置 state.cr（不是偏移量）。
        F 不在此处计算，由求解器根据 LLM 的 CR 通过公式 3-11 计算。
        restart_fraction 通过 state.extra 传递给 solver 执行（替换最差个体）。
        """
        state.cr = float(decision.get("cr", 0.5))
        state.extra["llm_cr"] = decision.get("cr")
        state.extra["llm_restart_fraction"] = decision.get("restart_fraction", 0.0)
        return state
