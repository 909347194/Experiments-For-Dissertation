# -*- coding: utf-8 -*-
"""operator_selection.py — LLM 算子策略选择模块

职责：
    根据搜索状态特征和优化轨迹，由 LLM 选择最适合的 DE 算子策略。
    可选策略：DE/rand/1, DE/best/1, DE/best/2, DE/current-to-pbest/1, 等。

注入点：after_evolve（每代进化后，决定下一轮使用的策略）
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..base_module import BaseLLMModule, ModuleState
from ..prompts import get_operator_selection_prompt

logger = logging.getLogger(__name__)

# 可用算子策略
AVAILABLE_STRATEGIES = [
    "rand/1", "best/1", "best/2",
    "current-to-pbest/1", "rand/2", "rand-to-best/1",
]


class LLMOperatorSelectionModule(BaseLLMModule):
    """LLM 算子策略选择模块。"""

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        # 从配置加载自定义 system prompt，支持外部文件覆盖和动态参数
        prompt_path = self._config.get("system_prompt_path")
        strategies = self._config.get("strategies", AVAILABLE_STRATEGIES)
        self._system_prompt: str = get_operator_selection_prompt(
            strategies=strategies,
        ) if prompt_path is None else None
        
        if prompt_path is not None:
            from pathlib import Path
            p = Path(prompt_path)
            if p.exists():
                self._system_prompt = p.read_text(encoding="utf-8")
            else:
                logger.warning(
                    "[operator_selection] system_prompt_path not found: %s, using default",
                    prompt_path,
                )
                self._system_prompt = get_operator_selection_prompt(strategies=strategies)

    @property
    def name(self) -> str:
        return "operator_selection"

    @property
    def hook_point(self) -> str:
        return "after_evolve"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        system = self._system_prompt

        # 构建用户消息
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
            "current_strategy": state.extra.get("current_strategy", "unknown"),
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
            f"## Task\nSelect the best DE strategy for the next interval. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        json_str = self._extract_json(llm_output)
        if json_str is None:
            return {"strategy": "default", "reasoning": "Failed to parse LLM output"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return {"strategy": "default", "reasoning": "Invalid JSON from LLM"}

        strategy = data.get("strategy", "default")
        if strategy not in AVAILABLE_STRATEGIES and strategy != "default":
            strategy = "default"

        return {
            "strategy": strategy,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(self, decision: dict[str, Any], state: ModuleState) -> ModuleState:
        state.extra["llm_strategy"] = decision.get("strategy", "default")
        return state


