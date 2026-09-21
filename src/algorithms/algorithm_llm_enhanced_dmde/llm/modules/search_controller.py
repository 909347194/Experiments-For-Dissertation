# -*- coding: utf-8 -*-
"""search_controller.py — LLM 搜索控制器模块

职责：
    在一次 LLM 调用中，独立决定 CR、F 和 GMR。
    三个参数完全解耦，LLM 可以分别控制。

    动作空间（解耦设计）：
        cr_action: hold | set   → 是否改变 CR
        cr:        离散候选值   → 具体 CR 值
        f_action:  hold | set   → 是否改变 F
        f:         离散候选值   → 具体 F 值
        gmr_mode:  auto|on|off → 灭绝模式

    v2 闭环控制改造（可观测 / 可归因 / 可触发）：
    - prompt 注入 stage 级过程统计（接受率、改进次数、停滞真实值、
      多样性分位数、stage 内 best 曲线），不再只喂两个聚合标量；
    - prompt 注入影子对照（固定参数的影子种群在同一 stage 的 Δf/ΔD），
      使 LLM 的动作效果可以与“什么都不调”的基线分离 —— 可归因；
    - prompt 注入 trigger_reason（为什么现在被咨询）。

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
# 预定义 F 候选值（与 CR 解耦，独立控制变异步长）
F_CHOICES = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]


class LLMSearchControllerModule(BaseLLMModule):
    """LLM 搜索控制器:独立决定 CR、F、GMR。

    动作空间（解耦设计）：
        cr_action: hold | set   → 是否改变 CR
        cr:        离散候选值   → 具体 CR 值（仅 set 时有效）
        f_action:  hold | set   → 是否改变 F
        f:         离散候选值   → 具体 F 值（仅 set 时有效）
        gmr_mode:  auto|on|off → 灭绝模式
    """

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")
        self._cr_choices = self._config.get("cr_choices", CR_CHOICES)
        self._f_choices = self._config.get("f_choices", F_CHOICES)

        # ---- 观测口径阈值（与 solver 的守卫判据同源，避免 prompt 与 solver 口径漂移）----
        _freeze_cfg = self._config.get("freeze", {}) or {}
        _trigger_cfg = self._config.get("trigger", {}) or {}
        self._df_noise_pct = float(_freeze_cfg.get("df_noise", 0.05))
        self._shadow_contrast = float(_freeze_cfg.get("shadow_contrast", 0.05))
        self._dd_noise = float(_trigger_cfg.get("dd_threshold", 0.05))

        # ---- 耦合对照模式 ----
        # True: 模型只控制 CR，F/GMR 由公式 3-11/3-12 推导（原 DMDE 手工耦合关系）。
        # 用于检验核心主张"解耦优于手工耦合"：与解耦条件共享观测集、触发机制、
        # 冻结守卫，唯一差异是动作空间。
        self._coupled = bool(self._config.get("coupled", False))
        # no_cr：CR 锁定（部分解耦），只把 F/GMR 交给模型。
        # solver 侧另有 cr_lock 负责真正锁死 CR 值，此处只负责动作空间。
        self._no_cr = bool(self._config.get("no_cr", False))
        if self._coupled:
            logger.info("search_controller: COUPLED mode — LLM controls CR only "
                        "(F/GMR derived via formulas 3-11/3-12)")
        if self._no_cr:
            logger.info("search_controller: NO_CR mode — CR locked by solver, "
                        "LLM controls F/GMR only")

        # 系统提示词按 (model_type, 三通道冻结态) 缓存：
        # 冻结段随每轮状态变化，若只缓存首轮结果，冻结提示将永不注入。
        self._system_prompt_cache: dict[tuple, str] = {}

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # ---- State: S_t(可观测性扩展版) ----
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "model_type": state.model_type,
            # ---- 约束可行性读数 ----
            # GMR 重启与 F 步长是决定可行性的主要杠杆；此前 solver 已计算并写入
            # ModuleState，但从未进入 prompt，属观测缺口。
            "feasible_ratio": round(state.feasible_ratio, 4),
            "violation_mean": round(state.violation_mean, 4),
            "violation_max": round(state.violation_max, 4),
            # D_t: 当前多样性水平 + 空间结构分位数
            "diversity": round(state.diversity, 4),
            "diversity_p25": round(state.diversity_p25, 4),
            "diversity_p75": round(state.diversity_p75, 4),
            # Δf_t / ΔD_t: 当前 stage 的趋势
            # 注意: Δf 未按 stage_length 归一,跨 stage 比较须结合历史表 Len 列
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
            # stage 内过程统计(已去重:improvements_in_stage 与 acceptance_rate 同源)
            "stage_length": state.stage_length,
            "acceptance_rate": (
                round(state.acceptance_rate, 4)
                if state.acceptance_rate is not None else None
            ),
            "gens_since_last_improvement": state.gens_since_last_improvement,
            # 停滞(真实值,未封顶)
            # stagnation_count 与 stagnation_raw 同为 detect_stagnation() 的返回值,已去重
            "stagnation_raw": state.stagnation_raw,
            # 上次动作(其效果见 stage_history 最后一行,不再重复给出)
            "prev_action_cr": state.prev_action,
            # stage 内逐代 best 曲线(降采样,携带 stage 内动态)
            "stage_best_curve": state.stage_best_curve,
            # 本次被触发的原因
            "trigger_reason": state.trigger_reason or None,
            # CR 通道是否被冻结(无证据守卫)
            "cr_frozen": state.cr_frozen,
            "cr_frozen_reason": state.cr_frozen_reason or None,
            # GMR / F 通道是否被冻结(连续无效动作守卫)
            "gmr_frozen": state.gmr_frozen,
            "gmr_frozen_reason": state.gmr_frozen_reason or None,
            "f_frozen": state.f_frozen,
            "f_frozen_reason": state.f_frozen_reason or None,
            # 当前 DE 参数
            # current_f / current_gmr_mode 是**状态观测**，两种模式都保留：
            # 耦合模式下模型需要它们来预判改 CR 会连带把 F/GMR 带到哪里。
            # f_choices 是**可选项列表**，耦合模式下不存在该选择权，故不注入。
            "current_f": round(state.f_scale, 4),
            "current_gmr_mode": state.gmr_mode,
        }
        if not self._coupled:
            features["f_choices"] = self._f_choices

        # ---- 影子对照(可归因锚点) ----
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

        # ---- Stage 级轨迹表格(最近 5 个 stage) ----
        trajectory_text = "No stage history yet (first decision)."
        stage_hist = state.stage_history
        if stage_hist:
            lines = [
                "Stage | Len | CR | F | GMR | Best Fitness | "
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
                f_val = s.get("f_scale", "?")
                gmr = s.get("gmr_mode", "?")
                lines.append(
                    f"{s['stage']:5d} | {s.get('stage_length', '?'):3} | "
                    f"{s['cr']:.1f} | {f_val} | {gmr:<3} | "
                    f"{s['best_fitness']:12.1f} | "
                    f"{df} | {dfs} | {dd} | {acc}"
                )
            trajectory_text = "\n".join(lines)

        # ---- Action-Outcome 反馈(含影子对照,可归因) ----
        feedback = ""
        if state.prev_action is None:
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"First decision: no previous action/outcome. "
                f"Choose from the current state only."
            )
        else:
            # 压缩:df / dD / df_shadow 已在 features 与 stage_history 末行给出,
            # 这里只补上「归因判据的量化尺度」——此前模型从未被告知多大差值才算它的效果。
            feedback = (
                f"\n\n## Last Decision Feedback\n"
                f"You set CR={state.prev_action:.1f}. Its outcome is the last row "
                f"of the stage history table (df / dD / df_shadow columns)."
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

        # 缓存键必须包含三通道冻结态:冻结段随每轮状态变化,
        # 若沿用首轮(全 False)的结果,_SC_FROZEN / _SC_FROZEN_GMR / _SC_FROZEN_F
        # 将永不注入,模型会持续请求已被守卫强制忽略的动作。
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
                cr_choices=self._cr_choices,
                f_choices=self._f_choices,
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

        动作空间为两级:先决定 cr_action(hold / set),再决定具体 CR 值。
        **默认路径是 hold** -- 解析失败、字段缺失或格式非法时一律返回 hold,
        而不是返回一个新的 CR 值。这是修复"无条件翻转"的关键:
        旧版 schema 强制模型每次输出一个 CR 数字,模型在证据无区分度时
        退化为 CR_t = flip(CR_{t-1})(实测 corr = -0.967,翻转率 0.98)。
        """
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return self._hold_decision("Failed to parse LLM output, holding CR by default")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return self._hold_decision("Invalid JSON from LLM, holding CR by default")

        if not isinstance(data, dict):
            return self._hold_decision("LLM output is not a JSON object, holding CR by default")

        evidence_read = str(data.get("evidence_read", ""))[:500]

        # ---- 第 1 级决策:是否改动 CR ----
        raw_cr = data.get("cr", None)
        cr_num = None
        try:
            cr_num = float(raw_cr)
        except (ValueError, TypeError):
            cr_num = None

        if "cr_action" in data:
            action = self._normalize_action(data.get("cr_action"))
        elif cr_num is not None:
            # 旧 schema(只给 cr、不给 cr_action):视为隐式 set,保持向后兼容
            action = "set_implicit"
        else:
            # 既没说要改、也没给值 → hold(本 bug 的核心修复路径)
            return self._hold_decision(
                "No cr_action and no usable cr in LLM output; holding CR by default",
                evidence_read,
            )

        # ---- 第 2 级决策:具体 CR 值(仅 set 时有效) ----
        cr = None
        if action in ("set", "set_implicit"):
            if cr_num is None:
                # 声明要改却没给合法值 → 回退到 hold(不猜一个值)
                return self._hold_decision(
                    f"cr_action='set' but cr={raw_cr!r} is not a number; "
                    f"holding CR by default", evidence_read,
                )
            cr = min(self._cr_choices, key=lambda c: abs(c - cr_num))

        # 防御:cr_action=hold 时忽略模型可能仍填的 cr 值
        if action == "hold":
            cr = None

        # ---- F: 独立变异步长（与 CR 对称的 hold/set 二级决策） ----
        f_action_raw = data.get("f_action", None)
        f_action = self._normalize_action(f_action_raw)

        f_raw = data.get("f", None)
        f_val = None
        try:
            f_val = float(f_raw)
        except (ValueError, TypeError):
            f_val = None

        if f_action == "set":
            if f_val is None:
                # 声明要改却没给合法值 → 回退到 hold
                f_action = "hold"
            else:
                # 钳位到最近的候选值
                f_val = min(self._f_choices, key=lambda f: abs(f - f_val))
        elif f_action == "hold":
            f_val = None
        else:
            # 未声明 f_action 但给了 f 值 → 隐式 set（向后兼容）
            if f_val is not None:
                f_action = "set"
                f_val = min(self._f_choices, key=lambda f: abs(f - f_val))
            else:
                f_action = "hold"

        # ---- GMR: 灭绝模式 ----
        gmr_raw = data.get("gmr_mode", "auto")
        gmr_mode = "auto"
        if gmr_raw is not None:
            s = str(gmr_raw).strip().lower()
            if s in {"on", "force", "extinct"}:
                gmr_mode = "on"
            elif s in {"off", "none", "skip"}:
                gmr_mode = "off"
            else:
                gmr_mode = "auto"

        # ---- CR 锁定：CR 通道不存在 ----
        if self._no_cr:
            action, cr = "hold", None

        # ---- 耦合对照：F / GMR 通道不存在 ----
        # 模型即使（受旧格式污染）输出了 f_action/gmr_mode，也一律忽略，
        # 强制走公式 3-11（F 由 CR 推导）与 3-12（GMR 由 CR 推导）。
        if self._coupled:
            f_action, f_val, gmr_mode = "hold", None, "auto"

        return {
            "cr_action": action,
            "cr": cr,                      # None = 保持当前 CR
            "f_action": f_action,
            "f": f_val,                     # None = 保持当前 F
            "gmr_mode": gmr_mode,           # "auto" | "on" | "off"
            "evidence_read": evidence_read,
            "reasoning": data.get("reasoning", ""),
        }

    @staticmethod
    def _normalize_action(raw: Any) -> str:
        """把模型对 cr_action 的各种说法归一化为 "hold" / "set"。

        未知/缺失/非法一律归为 hold(保守默认)。
        """
        if raw is None:
            return "hold"
        s = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        if s in {"set", "change", "adjust", "update", "modify", "true", "yes", "1"}:
            return "set"
        # hold / keep / unchanged / none / false / no / 0 / freeze ...
        return "hold"

    def _hold_decision(self, reason: str, evidence_read: str = "") -> dict[str, Any]:
        """构造一个 hold 决策(兜底路径)。"""
        return {
            "cr_action": "hold",
            "cr": None,
            "f_action": "hold",
            "f": None,                 # 不覆写 F
            "gmr_mode": "auto",        # 不改变 GMR
            "evidence_read": evidence_read,
            "reasoning": reason,
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """将决策应用到搜索状态。

        CR:  cr_action == "hold" 时不改动 state.cr。
        F:   f_action == "set" 时覆写 state.f_override(覆盖公式 3-11)。
        GMR: "auto" 保持公式 3-12,"on" 强制灭绝,"off" 禁止灭绝。
        """
        # CR
        new_cr = decision.get("cr", None)
        if new_cr is not None and decision.get("cr_action") == "set":
            state.cr = float(new_cr)
        state.extra["llm_cr"] = new_cr
        state.extra["llm_cr_action"] = decision.get("cr_action", "hold")

        # F: 独立覆写（与 CR 对称的 hold/set 决策）
        f_val = decision.get("f", None)
        f_action = decision.get("f_action", "hold")
        if f_action == "set" and f_val is not None:
            state.f_override = float(f_val)
        # hold: 保持 state.f_override 不变
        state.extra["llm_f"] = f_val
        state.extra["llm_f_action"] = f_action

        # GMR: 模式覆写
        gmr_mode = decision.get("gmr_mode", "auto")
        state.gmr_mode = gmr_mode
        state.extra["llm_gmr_mode"] = gmr_mode

        return state
