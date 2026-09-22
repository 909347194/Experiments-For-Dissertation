# -*- coding: utf-8 -*-
"""search_controller.py - LLM strategy selector.

LLM reads search state and trajectory, then chooses ONE strategy.
Each strategy maps to a fixed (CR, F, GMR) config.
LLM does NOT optimize numeric values - it selects strategy semantics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..base_module import BaseLLMModule, ModuleState
from ..prompts import get_prompt

logger = logging.getLogger(__name__)

# Strategy definitions (based on CR Response Landscape findings)
# CR=0.3 is the phase transition (GMR stops below delta=0.3)
STRATEGIES: dict[str, dict[str, Any]] = {
    "hold":     {"cr": None, "f": None, "gmr": "auto"},
    "explore":  {"cr": 0.8,  "f": 0.9,  "gmr": "off"},
    "balanced": {"cr": 0.5,  "f": 0.5,  "gmr": "auto"},
    "exploit":  {"cr": 0.3,  "f": 0.3,  "gmr": "off"},
    "recover":  {"cr": 0.5,  "f": 0.7,  "gmr": "on"},
}

STRATEGY_CHOICES = list(STRATEGIES.keys())


class LLMSearchControllerModule(BaseLLMModule):
    """LLM strategy selector: chooses strategy based on search state."""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        self._prompt_path = self._config.get("system_prompt_path")
        self._strategies = self._config.get("strategies", STRATEGIES)
        self._system_prompt: str | None = None

    @property
    def name(self) -> str:
        return "search_controller"

    @property
    def hook_point(self) -> str:
        return "before_mutation"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        # Search state (data, not instructions)
        features = {
            "generation": state.generation,
            "max_generations": state.max_generations,
            "diversity": round(state.diversity, 4),
            "delta_fitness_pct": state.delta_fitness,
            "delta_diversity": state.delta_diversity,
            "acceptance_rate": (
                round(state.acceptance_rate, 4)
                if state.acceptance_rate is not None else None
            ),
            "stagnation_raw": state.stagnation_raw,
            "gens_since_last_improvement": state.gens_since_last_improvement,
            "improvements_in_stage": state.improvements_in_stage,
            "current_strategy": state.extra.get("current_strategy", "hold"),
            "trigger_reason": state.trigger_reason or None,
        }

        # Shadow attribution
        if state.shadow_delta_fitness is not None:
            features["shadow_df_pct"] = round(state.shadow_delta_fitness, 4)
            features["df_vs_shadow"] = (
                round(state.delta_fitness - state.shadow_delta_fitness, 4)
                if state.delta_fitness is not None else None
            )

        # Stage history (last 5 stages)
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

        user = get_prompt(
            "search_controller",
            prompt_type="user",
            state_json=json.dumps(features, indent=2),
            history_text=history_text,
        )

        if self._system_prompt is None:
            self._system_prompt = get_prompt(
                "search_controller",
                prompt_type="system",
                prompt_path=self._prompt_path,
                model_type=state.model_type,
                strategies=self._strategies,
            )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        """Parse LLM output into strategy selection with resolved CR/F/GMR."""
        _fallback = {
            "strategy": "hold", "override_cr": None, "override_f": None,
            "cr": None, "f": None, "gmr_mode": "auto", "reasoning": "parse failed",
        }
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return dict(_fallback)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return dict(_fallback, reasoning="invalid JSON")

        if not isinstance(data, dict):
            return dict(_fallback, reasoning="not a JSON object")

        # Strategy selection
        raw_strategy = str(data.get("strategy", "hold")).strip().lower()
        strategy = raw_strategy if raw_strategy in self._strategies else "hold"

        # Resolve strategy config to effective CR/F/GMR (for solver)
        strategy_cfg = self._strategies.get(strategy, self._strategies["hold"])
        effective_cr = strategy_cfg["cr"]
        effective_f = strategy_cfg["f"]
        effective_gmr = strategy_cfg["gmr"]

        # Optional overrides
        override_cr = self._parse_optional_float(data.get("override_cr"))
        override_f = self._parse_optional_float(data.get("override_f"))
        if override_cr is not None:
            effective_cr = override_cr
        if override_f is not None:
            effective_f = override_f

        return {
            "strategy": strategy,
            "override_cr": override_cr,
            "override_f": override_f,
            "cr": effective_cr,
            "f": effective_f,
            "gmr_mode": effective_gmr,
            "evidence_read": str(data.get("evidence_read", ""))[:500],
            "reasoning": str(data.get("reasoning", ""))[:500],
        }

    @staticmethod
    def _parse_optional_float(raw: Any) -> float | None:
        if raw is None:
            return None
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        """Apply strategy decision to search state."""
        strategy = decision.get("strategy", "hold")
        cfg = self._strategies.get(strategy, self._strategies["hold"])

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

        # Record current strategy for next prompt
        state.extra["current_strategy"] = strategy

        return state

    def screen_strategy(self, decision: dict[str, Any], state: ModuleState) -> dict[str, Any]:
        """决策护栏：抑制 recover 的滥用。

        `recover` 在物理上等于强制全局灭绝（apply_extinction）——保留最优 30%、重置
        其余约 70% 种群、丢弃当前盆地。诊断表明，原协议把约 2/3 的决策浪费在 recover
        上，因为 prompt 把"stagnation>50"误判为失败并触发全局重置，反复打断收敛，
        既拖慢速度也停在次优盆地。

        本护栏只允许 recover 在搜索真正陷入绝境时放行：
          - 多样性正在显著塌缩（delta_diversity < -0.06，而非单纯的停滞），且
          - 长期停滞（stagnation_raw >= 60），
          - 且近期（最近 3 个 stage）没有用过 recover（防止连续重置）。
        不满足则降级为 `exploit`（在当前盆地精修，不重置），并在 decision 中
        记录 `recover_suppressed` 与原因，保证事后分析透明。

        Returns:
            可能被动过的 decision（recover 被降级时重写 strategy/cr/f/gmr_mode）。
        """
        strategy = decision.get("strategy", "hold")
        if strategy != "recover":
            return decision

        dd = state.delta_diversity
        stag = state.stagnation_raw
        genuine_trap = (
            dd is not None and dd < -0.06
            and (stag is None or stag >= 60)
        )
        recent_recover = bool(state.stage_history) and any(
            s.get("strategy") == "recover"
            for s in state.stage_history[-3:]
        )

        if not genuine_trap or recent_recover:
            cfg = self._strategies["exploit"]
            decision["strategy"] = "exploit"
            decision["cr"] = cfg["cr"]
            decision["f"] = cfg["f"]
            decision["gmr_mode"] = cfg["gmr"]
            decision["recover_suppressed"] = True
            decision["recover_suppress_reason"] = (
                f"recover blocked: not a genuine trap "
                f"(delta_diversity={dd}, stagnation_raw={stag}, "
                f"recent_recover={recent_recover}) → downgraded to exploit"
            )

        return decision