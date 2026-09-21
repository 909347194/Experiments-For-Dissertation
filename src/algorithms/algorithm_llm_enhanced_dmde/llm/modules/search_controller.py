# -*- coding: utf-8 -*-
"""search_controller.py — LLM 搜索控制器模块（v3: 语义档位版）

职责：
    在一次 LLM 调用中，从预定义的语义策略档位中选择一个，
    或保持当前档位不变（hold）。
    每个档位 = {mutation_strategy, CR, F, GMR} 的固定组合，
    语义清晰，消除标量微调的无效探索。

    动作空间（v3 档位版）：
        preset: "hold" | "explore" | "balanced" | "exploit" |
                "recover" | "rand-1" | "best-1"
        override_cr:  null | float  （可选，覆盖档位的 CR）
        override_f:   null | float  （可选，覆盖档位的 F）

    保留的工程特性：
        - 影子对照（shadow control，可归因）
        - 事件触发（event-based trigger）
        - 冻结守卫（CR/GMR/F freeze guard）
        - hold/set 二级决策（保守默认）

注入点：before_mutation（变异前，由 solver 的事件触发器决定何时调用）
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from ..base_module import BaseLLMModule, ModuleState
from ..prompts import get_prompt
from ..presets import (
    PRESETS, HOLD_PRESET_NAME, StrategyPreset,
    get_preset, list_preset_names, format_presets_compact,
)

logger = logging.getLogger(__name__)


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器 v3：语义档位选择。

    LLM 从预定义的策略档位中选择一个，每个档位包含：
    - mutation_strategy: "rand/1" | "best/1" | "mixed"
    - CR: 交叉率
    - F: 缩放因子
    - GMR: 灭绝模式

    可选地通过 override_cr / override_f 覆盖档位的默认值。
    """

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")

        # ---- 观测口径阈值（与 solver 的守卫判据同源）----
        _freeze_cfg = self._config.get("freeze", {}) or {}
        _trigger_cfg = self._config.get("trigger", {}) or {}
        self._df_noise_pct = float(_freeze_cfg.get("df_noise", 0.05))
        self._shadow_contrast = float(_freeze_cfg.get("shadow_contrast", 0.05))
        self._dd_noise = float(_trigger_cfg.get("dd_threshold", 0.05))

        # ---- 耦合对照模式 ----
        self._coupled = bool(self._config.get("coupled", False))
        self._no_cr = bool(self._config.get("no_cr", False))

        if self._coupled:
            logger.info("search_controller: COUPLED mode — LLM controls CR only "
                        "(F/GMR derived via formulas 3-11/3-12)")
        if self._no_cr:
            logger.info("search_controller: NO_CR mode — CR locked by solver, "
                        "LLM controls F/GMR only")

        # 系统提示词按 (model_type, 冻结态) 缓存
        self._system_prompt_cache: dict[tuple, str] = {}

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # ---- State: S_t (可观测性扩展版) ----
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "model_type": state.model_type,
            # 当前档位信息
            "current_preset": state.extra.get("current_preset", "balanced"),
            "current_mutation_strategy": state.extra.get("current_mutation_strategy", "mixed"),
            # 适应度（LLM 决策的核心参考）
            "best_fitness": round(state.best_fitness, 2) if np.isfinite(state.best_fitness) else None,
            "mean_fitness": round(state.mean_fitness, 2) if np.isfinite(state.mean_fitness) else None,
            # 约束可行性
            "feasible_ratio": round(state.feasible_ratio, 4),
            "violation_mean": round(state.violation_mean, 4),
            "violation_max": round(state.violation_max, 4),
            # 多样性
            "diversity": round(state.diversity, 4),
            "diversity_p25": round(state.diversity_p25, 4),
            "diversity_p75": round(state.diversity_p75, 4),
            # 趋势
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
            # stage 内过程统计
            "stage_length": state.stage_length,
            "acceptance_rate": (
                round(state.acceptance_rate, 4)
                if state.acceptance_rate is not None else None
            ),
            "gens_since_last_improvement": state.gens_since_last_improvement,
            # 停滞
            "stagnation_raw": state.stagnation_raw,
            # 上次动作
            "prev_preset": state.extra.get("prev_preset", None),
            # stage 内逐代 best 曲线
            "stage_best_curve": state.stage_best_curve,
            # 触发原因
            "trigger_reason": state.trigger_reason or None,
            # 冻结状态
            "cr_frozen": state.cr_frozen,
            "cr_frozen_reason": state.cr_frozen_reason or None,
            "gmr_frozen": state.gmr_frozen,
            "gmr_frozen_reason": state.gmr_frozen_reason or None,
            "f_frozen": state.f_frozen,
            "f_frozen_reason": state.f_frozen_reason or None,
            # 当前 DE 参数
            "current_cr": round(state.cr, 4),
            "current_f": round(state.f_scale, 4),
            "current_gmr_mode": state.gmr_mode,
        }

        # ---- 影子对照 ----
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

        # ---- Stage 级轨迹表格 ----
        trajectory_text = "No stage history yet (first decision)."
        stage_hist = state.stage_history
        if stage_hist:
            lines = [
                "Stage | Len | Preset | CR | F | GMR | Best Fitness | "
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
                preset = s.get("preset", "?")
                f_val = s.get("f_scale", "?")
                gmr = s.get("gmr_mode", "?")
                lines.append(
                    f"{s['stage']:5d} | {s.get('stage_length', '?'):3} | "
                    f"{preset:<8} | {s['cr']:.1f} | {f_val} | {gmr:<3} | "
                    f"{s['best_fitness']:12.1f} | "
                    f"{df} | {dfs} | {dd} | {acc}"
                )
            trajectory_text = "\n".join(lines)

        # ---- Action-Outcome 反馈 ----
        feedback = ""
        if state.prev_action is None:
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"First decision: no previous action/outcome. "
                f"Choose from the current state only."
            )
        else:
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"You chose preset='{state.extra.get('prev_preset', 'unknown')}'. "
                f"Its outcome is the last row of the stage history table "
                f"(df / dD / df_shadow columns)."
            )
            if state.shadow_delta_fitness is not None:
                feedback += (
                    f"\nAttribution gate: |df - df_shadow| must exceed "
                    f"{self._shadow_contrast:.2f}% to count as an effect of your "
                    f"choices. Below that it is indistinguishable from the "
                    f"fixed-CR baseline and is NOT evidence for changing anything."
                )

        user = get_prompt(
            "search_controller",
            prompt_type="user",
            state_json=json.dumps(features, indent=2),
            trajectory_text=trajectory_text + feedback,
        )

        # 缓存键
        cache_key = (
            state.model_type,
            bool(state.cr_frozen),
            bool(state.gmr_frozen),
            bool(state.f_frozen),
            state.shadow_cr,
            self._coupled,
            self._no_cr,
        )
        system_prompt = self._system_prompt_cache.get(cache_key)
        if system_prompt is None:
            system_prompt = get_prompt(
                "search_controller",
                prompt_type="system",
                prompt_path=self._prompt_path,
                model_type=state.model_type,
                shadow_cr=state.shadow_cr,
                cr_frozen=state.cr_frozen,
                gmr_frozen=state.gmr_frozen,
                f_frozen=state.f_frozen,
                df_noise_pct=self._df_noise_pct,
                dd_noise=self._dd_noise,
                shadow_contrast=self._shadow_contrast,
                coupled=self._coupled,
                no_cr=self._no_cr,
            )
            self._system_prompt_cache[cache_key] = system_prompt

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        """解析 LLM 输出。

        v3 输出格式：
            {
                "evidence_read": "...",
                "preset": "hold" | "explore" | "balanced" | ...,
                "override_cr": null | float,
                "override_f": null | float,
                "reasoning": "..."
            }

        默认路径是 hold — 解析失败、字段缺失、格式非法一律返回 hold。

        向后兼容：如果 LLM 输出旧格式（cr_action/f_action/gmr_mode），
        自动转换为最接近的 preset。
        """
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return self._hold_decision("Failed to parse LLM output, holding by default")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return self._hold_decision("Invalid JSON from LLM, holding by default")

        if not isinstance(data, dict):
            return self._hold_decision("LLM output is not a JSON object, holding by default")

        evidence_read = str(data.get("evidence_read", ""))[:500]

        # ---- 优先解析 v3 preset 格式 ----
        raw_preset = data.get("preset", None)
        if raw_preset is not None:
            return self._parse_preset_decision(data, evidence_read)

        # ---- 向后兼容：旧格式 cr_action/f_action/gmr_mode → 映射到最接近的 preset ----
        if "cr_action" in data or "cr" in data or "f_action" in data:
            return self._parse_legacy_decision(data, evidence_read)

        # ---- 什么都没有 → hold ----
        return self._hold_decision(
            "No preset or legacy fields in LLM output; holding by default",
            evidence_read,
        )

    def _parse_preset_decision(self, data: dict, evidence_read: str) -> dict[str, Any]:
        """解析 v3 preset 格式。"""
        raw_preset = str(data.get("preset", "hold")).strip().lower().replace("-", "_")

        # 归一化别名
        alias_map = {
            "rand_1": "rand-1",
            "best_1": "best-1",
            "rand1": "rand-1",
            "best1": "best-1",
            "rand": "rand-1",
            "best": "best-1",
            "explore": "explore",
            "balanced": "balanced",
            "exploit": "exploit",
            "recover": "recover",
            "recovery": "recover",
            "hold": "hold",
            "keep": "hold",
            "unchanged": "hold",
        }
        preset_name = alias_map.get(raw_preset, raw_preset)

        # hold
        if preset_name == "hold":
            return {
                "preset": "hold",
                "mutation_strategy": None,
                "override_cr": None,
                "override_f": None,
                "evidence_read": evidence_read,
                "reasoning": data.get("reasoning", ""),
            }

        # 验证 preset 是否存在
        if preset_name not in PRESETS:
            return self._hold_decision(
                f"Unknown preset '{preset_name}', holding by default",
                evidence_read,
            )

        # 可选覆盖
        override_cr = self._parse_optional_float(data.get("override_cr"))
        override_f = self._parse_optional_float(data.get("override_f"))

        preset = PRESETS[preset_name]
        return {
            "preset": preset_name,
            "mutation_strategy": preset.mutation_strategy,
            "override_cr": override_cr,
            "override_f": override_f,
            "evidence_read": evidence_read,
            "reasoning": data.get("reasoning", ""),
        }

    def _parse_legacy_decision(self, data: dict, evidence_read: str) -> dict[str, Any]:
        """将旧格式（cr_action/cr/f_action/f/gmr_mode）映射到最接近的 preset。"""
        # 提取旧格式值
        cr_action = str(data.get("cr_action", "hold")).strip().lower()
        cr_val = self._parse_optional_float(data.get("cr"))
        f_action = str(data.get("f_action", "hold")).strip().lower()
        f_val = self._parse_optional_float(data.get("f"))
        gmr_raw = str(data.get("gmr_mode", "auto")).strip().lower()

        # hold 一切
        if cr_action == "hold" and f_action == "hold" and gmr_raw == "auto":
            return {
                "preset": "hold",
                "override_cr": None,
                "override_f": None,
                "evidence_read": evidence_read,
                "reasoning": f"[legacy compat] {data.get('reasoning', '')}",
            }

        # 有 GMR=on → recover
        if gmr_raw in ("on", "force", "extinct"):
            return {
                "preset": "recover",
                "override_cr": cr_val if cr_action == "set" else None,
                "override_f": f_val if f_action == "set" else None,
                "evidence_read": evidence_read,
                "reasoning": f"[legacy→recover] {data.get('reasoning', '')}",
            }

        # CR 高 + F 高 → explore
        actual_cr = cr_val if cr_action == "set" and cr_val else None
        actual_f = f_val if f_action == "set" and f_val else None
        if actual_cr and actual_cr >= 0.7:
            return {
                "preset": "explore",
                "override_cr": actual_cr,
                "override_f": actual_f,
                "evidence_read": evidence_read,
                "reasoning": f"[legacy→explore] {data.get('reasoning', '')}",
            }

        # CR 低 → exploit
        if actual_cr and actual_cr <= 0.35:
            return {
                "preset": "exploit",
                "override_cr": actual_cr,
                "override_f": actual_f,
                "evidence_read": evidence_read,
                "reasoning": f"[legacy→exploit] {data.get('reasoning', '')}",
            }

        # 其他 → balanced
        return {
            "preset": "balanced",
            "override_cr": actual_cr,
            "override_f": actual_f,
            "evidence_read": evidence_read,
            "reasoning": f"[legacy→balanced] {data.get('reasoning', '')}",
        }

    @staticmethod
    def _parse_optional_float(val: Any) -> float | None:
        """解析可选浮点值，无效则返回 None。"""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _hold_decision(self, reason: str, evidence_read: str = "") -> dict[str, Any]:
        """构造一个 hold 决策（兜底路径）。"""
        return {
            "preset": "hold",
            "override_cr": None,
            "override_f": None,
            "evidence_read": evidence_read,
            "reasoning": reason,
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将决策应用到搜索状态。

        preset="hold" 时不改动任何参数。
        其他 preset → 解包为 CR/F/GMR/mutation_strategy 写入 state。
        override_cr/override_f 可覆盖 preset 的默认值。
        """
        preset_name = decision.get("preset", "hold")
        state.extra["llm_preset"] = preset_name
        state.extra["llm_override_cr"] = decision.get("override_cr")
        state.extra["llm_override_f"] = decision.get("override_f")

        if preset_name == "hold":
            # 不改动任何参数
            state.extra["llm_cr_action"] = "hold"
            state.extra["llm_f_action"] = "hold"
            return state

        preset = get_preset(preset_name)
        if preset is None:
            logger.warning("Unknown preset '%s' in apply_decision, holding", preset_name)
            return state

        # 解包 preset 参数
        override_cr = decision.get("override_cr")
        new_cr = override_cr if override_cr is not None else preset.cr
        override_f = decision.get("override_f")
        new_f = override_f if override_f is not None else preset.f

        state.cr = float(new_cr)
        state.f_override = float(new_f)
        state.gmr_mode = preset.gmr_mode
        state.mutation_strategy = preset.mutation_strategy

        state.extra["llm_cr"] = new_cr
        state.extra["llm_cr_action"] = "set"
        state.extra["llm_f"] = new_f
        state.extra["llm_f_action"] = "set"
        state.extra["llm_gmr_mode"] = preset.gmr_mode
        state.extra["llm_mutation_strategy"] = preset.mutation_strategy

        return state