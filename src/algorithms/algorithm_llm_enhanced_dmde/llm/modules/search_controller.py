# -*- coding: utf-8 -*-
<<<<<<< HEAD
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
=======
"""search_controller.py — LLM 策略选择控制器

方法：LLM 根据进化状态和搜索轨迹，从预定义策略中选择一个。
每个策略映射到一组 (CR, F, GMR) 参数。LLM 不做数值优化，只做策略决策。

策略定义（基于 CR Response Landscape 实验结论）：

| 策略     | CR   | F    | GMR  | 适用状态 |
|----------|------|------|------|----------|
| explore  | 0.8  | 0.9  | on   | 多样性低、停滞 |
| balanced | 0.5  | 0.5  | auto | 正常搜索 |
| exploit  | 0.3  | 0.3  | off  | 收敛中、需要精细调整 |
| recover  | 0.5  | 0.7  | on   | 深度停滞、需要多样性注入 |
| hold     | 不变 | 不变 | 不变 | 搜索活跃、当前策略有效 |
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

LLM 可通过 override_cr / override_f 微调，但默认使用策略默认值。
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

<<<<<<< HEAD

class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器 v3：语义档位选择。

    LLM 从预定义的策略档位中选择一个，每个档位包含：
    - mutation_strategy: "rand/1" | "best/1" | "mixed"
    - CR: 交叉率
    - F: 缩放因子
    - GMR: 灭绝模式

    可选地通过 override_cr / override_f 覆盖档位的默认值。
    """
=======
# ── 策略定义 ─────────────────────────────────────────────
# CR=0.3 是相变点（GMR 停止），explore 需要 CR>0.3 避免 GMR 干扰
# recover 需要 GMR=on 强制注入多样性
STRATEGIES: dict[str, dict[str, Any]] = {
    "hold":     {"cr": None, "f": None, "gmr": "auto"},
    "explore":  {"cr": 0.8,  "f": 0.9,  "gmr": "on"},
    "balanced": {"cr": 0.5,  "f": 0.5,  "gmr": "auto"},
    "exploit":  {"cr": 0.3,  "f": 0.3,  "gmr": "off"},
    "recover":  {"cr": 0.5,  "f": 0.7,  "gmr": "on"},
}

STRATEGY_CHOICES = list(STRATEGIES.keys())


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 策略选择控制器：根据搜索状态选择策略。"""
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")
<<<<<<< HEAD

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
=======
        self._strategies = self._config.get("strategies", STRATEGIES)
        self._system_prompt: str | None = None
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
<<<<<<< HEAD
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
=======
        # ── 搜索状态（数据，不是指令） ──
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "diversity": round(state.diversity, 4),
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)
            "acceptance_rate": (
                round(state.acceptance_rate, 4)
                if state.acceptance_rate is not None else None
            ),
<<<<<<< HEAD
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
=======
            "stagnation_raw": state.stagnation_raw,
            "gens_since_last_improvement": state.gens_since_last_improvement,
            "improvements_in_stage": state.improvements_in_stage,
            "current_strategy": state.extra.get("current_strategy", "hold"),
            "trigger_reason": state.trigger_reason or None,
        }

        # Shadow 归因
        if state.shadow_delta_fitness is not None:
            features["shadow_df_pct"] = round(state.shadow_delta_fitness, 4)
            features["df_vs_shadow"] = (
                round(state.delta_fitness - state.shadow_delta_fitness, 4)
                if state.delta_fitness is not None else None
            )

        # Stage History（最近 5 个 stage）
        history_text = "No history yet."
        if state.stage_history:
            lines = ["Stage | Strategy | df(%) | df_shadow(%) | Acc% | Stag"]
            for s in state.stage_history[-5:]:
                df = f"{s.get('delta_fitness', 0):+7.2f}%" if s.get("delta_fitness") is not None else "    N/A"
                dfs = f"{s.get('shadow_delta_fitness', 0):+7.2f}%" if s.get("shadow_delta_fitness") is not None else "    N/A"
                acc = f"{s.get('acceptance_rate', 0)*100:5.1f}" if s.get("acceptance_rate") is not None else "  N/A"
                stag = s.get("stagnation_raw", "?")
                strat = s.get("strategy", s.get("preset", "?"))
                lines.append(f"{s.get('stage', '?'):5} | {strat:<9} | {df} | {dfs} | {acc} | {stag}")
            history_text = "\n".join(lines)
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

        user = get_prompt(
            "search_controller",
            prompt_type="user",
            state_json=json.dumps(features, indent=2),
            history_text=history_text,
        )

<<<<<<< HEAD
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
=======
        if self._system_prompt is None:
            self._system_prompt = get_prompt(
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)
                "search_controller",
                prompt_type="system",
                prompt_path=self._prompt_path,
                model_type=state.model_type,
<<<<<<< HEAD
                shadow_cr=state.shadow_cr,
                cr_frozen=state.cr_frozen,
                gmr_frozen=state.gmr_frozen,
                f_frozen=state.f_frozen,
                df_noise_pct=self._df_noise_pct,
                dd_noise=self._dd_noise,
                shadow_contrast=self._shadow_contrast,
                coupled=self._coupled,
                no_cr=self._no_cr,
=======
                strategies=self._strategies,
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)
            )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
<<<<<<< HEAD
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
=======
        """解析 LLM 输出为策略选择。"""
        _fallback = {"strategy": "hold", "override_cr": None, "override_f": None,
                     "cr": None, "f": None, "gmr_mode": "auto", "reasoning": "parse failed"}
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return dict(_fallback)
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
<<<<<<< HEAD
            return self._hold_decision("Invalid JSON from LLM, holding by default")

        if not isinstance(data, dict):
            return self._hold_decision("LLM output is not a JSON object, holding by default")
=======
            return dict(_fallback, reasoning="invalid JSON")

        if not isinstance(data, dict):
            return dict(_fallback, reasoning="not a JSON object")
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

        # 策略选择
        raw_strategy = str(data.get("strategy", "hold")).strip().lower()
        strategy = raw_strategy if raw_strategy in self._strategies else "hold"

<<<<<<< HEAD
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
=======
        # 解析策略配置到具体 CR/F/GMR（供 solver 直接使用）
        strategy_cfg = self._strategies.get(strategy, self._strategies["hold"])
        effective_cr = strategy_cfg["cr"]
        effective_f = strategy_cfg["f"]
        effective_gmr = strategy_cfg["gmr"]

        # 可选 override
        override_cr = None
        override_f = None
        try:
            v = data.get("override_cr")
            if v is not None:
                override_cr = max(0.0, min(1.0, float(v)))
                effective_cr = override_cr
        except (ValueError, TypeError):
            pass
        try:
            v = data.get("override_f")
            if v is not None:
                override_f = max(0.0, min(2.0, float(v)))
                effective_f = override_f
        except (ValueError, TypeError):
            pass
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)

        preset = PRESETS[preset_name]
        return {
<<<<<<< HEAD
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
=======
            "strategy": strategy,
            "override_cr": override_cr,
            "override_f": override_f,
            "cr": effective_cr,          # solver 读这个
            "f": effective_f,            # solver 读这个
            "gmr_mode": effective_gmr,   # solver 读这个
            "evidence_read": str(data.get("evidence_read", ""))[:500],
            "reasoning": str(data.get("reasoning", ""))[:500],
>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将策略决策应用到搜索状态。"""
        strategy = decision.get("strategy", "hold")
        cfg = self._strategies.get(strategy, self._strategies["hold"])

<<<<<<< HEAD
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

=======
        # CR
        if cfg["cr"] is not None:
            state.cr = cfg["cr"]
        if decision.get("override_cr") is not None:
            state.cr = decision["override_cr"]

        # F
        if cfg["f"] is not None:
            state.f_override = cfg["f"]
        if decision.get("override_f") is not None:
            state.f_override = decision["override_f"]

        # GMR
        state.gmr_mode = cfg["gmr"]

        # 记录当前策略供下次 prompt 使用
        state.extra["current_strategy"] = strategy

>>>>>>> 16cd2af (refactor: LLM 策略选择架构 — 精简 prompt 12800→2600 chars)
        return state