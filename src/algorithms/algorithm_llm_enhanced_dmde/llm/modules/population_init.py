# -*- coding: utf-8 -*-
"""population_init.py — LLM 种群初始化模块（v2）

职责：
    在种群初始化阶段（before_init），由 LLM 基于结构化问题表示 S_problem
    直接生成离散的候选分配方案（assignment），代码侧负责将 assignment
    转换为 DMDE 统一基因编码。

注入点：before_init（种群初始化之前，生成候选 assignments）

设计原则：
    1. LLM 只输出 assignment，不碰 gene 编码
    2. 代码负责 assignment → gene 转换 + cost 补全
    3. K = ceil(r × N_pop)，r 是超参数（llm_init_ratio）
    4. Quality + Diversity 过滤
    5. 模块化、支持消融实验
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from ..base_module import BaseLLMModule, ModuleState

logger = logging.getLogger(__name__)


# ── System Prompt ──────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert in UAV-target assignment optimization.

Your task: Generate candidate assignment plans for a UAV scheduling problem. \
The solver will convert your assignments into an evolutionary algorithm's \
internal encoding, so you only need to produce **discrete assignments**.

## Output Format
Respond with a JSON object only (no markdown):
{{
    "assignments": [
        {{"uav": <int>, "targets": [<int>, ...]}},
        ...
    ],
    "reasoning": "<brief explanation of your strategy>"
}}

## Assignment Rules by Model Type

### balanced (N == M, one-to-one)
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- Example: {{"uav": 0, "targets": [2]}}, {{"uav": 1, "targets": [0]}}

### overloaded (N > M, UAVs outnumber targets)
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.
- Example: {{"uav": 0, "targets": [1]}}, {{"uav": 1, "targets": [0]}}, {{"uav": 2, "targets": [1]}}

### srp (N < M, UAVs visit multiple targets in sequence)
- Each UAV appears at least once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.
- Example: {{"uav": 0, "targets": [3, 1, 4]}}, {{"uav": 1, "targets": [2, 0]}}

## Guidelines
- Focus on minimizing total cost while respecting constraints.
- Use the preference summary to identify low-cost assignments.
- For SRP, order targets to minimize transition costs (nearest-neighbor heuristic).
- Generate exactly the requested number of assignments.
- Ensure all constraints are satisfied (every target covered, etc.).
"""


class LLMPopulationInitModule(BaseLLMModule):
    """LLM 种群初始化模块（v2）。

    在 before_init 阶段调用 LLM 生成候选分配方案，
    由 solver 侧的 AssignmentConverter 转换为基因编码并注入种群。
    """

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(llm_client, config)
        # 支持从配置加载自定义 system prompt（与 search_controller 一致）
        self._system_prompt: str = self._config.get("system_prompt", _SYSTEM_PROMPT)
        prompt_path = self._config.get("system_prompt_path")
        if prompt_path:
            try:
                from pathlib import Path
                p = Path(prompt_path)
                if p.exists():
                    self._system_prompt = p.read_text(encoding="utf-8")
                else:
                    logger.warning(
                        "[population_init] system_prompt_path not found: %s, using default",
                        prompt_path,
                    )
            except Exception as e:
                logger.warning(
                    "[population_init] Failed to load system_prompt_path: %s, using default",
                    e,
                )

    @property
    def name(self) -> str:
        return "population_init"

    @property
    def hook_point(self) -> str:
        return "before_init"

    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        """构建 S_problem 结构化问题表示。

        包含：
        - Problem structure: N, M, model_type
        - Cost structure: C_UT (UAV→Target), C_TT (Target→Target, SRP 时关键)
        - Constraints: 约束描述
        - Objectives: 目标函数方向
        - Preference summary: 每行/列的 top-k 最小代价统计
        """
        cm = state.cost_matrix
        n_uavs = state.n_uavs
        n_targets = state.n_targets
        model_type = state.model_type
        # 优先从 state.extra 读取（solver 注入），回退到 _config（用户配置）
        top_k = state.extra.get("preference_top_k",
                                self._config.get("preference_top_k", 3))

        # 构建代价矩阵的 JSON 友好表示
        cm_data = None
        c_tt_data = None
        if cm is not None:
            # C_UT: UAV → Target 代价矩阵（上半部分）
            rows, cols = cm.shape
            ut_rows = min(n_uavs, rows)
            ut_cols = min(n_targets, cols)
            cm_data = [
                [round(float(cm[i, j]), 2) for j in range(ut_cols)]
                for i in range(ut_rows)
            ]
            # C_TT: Target → Target 巡游代价（SRP 时，矩阵下半部分）
            if model_type == "srp" and rows > n_uavs:
                tt_size = min(n_targets, rows - n_uavs)
                c_tt_data = [
                    [round(float(cm[n_uavs + i, j]), 2) for j in range(tt_size)]
                    for i in range(tt_size)
                ]

        # Preference summary: 每行/列 top-k 最小代价
        preference_summary = self._build_preference_summary(cm, n_uavs, n_targets, top_k)

        # 约束描述
        constraints_desc = self._describe_constraints(model_type, n_uavs, n_targets)

        # 需要生成的候选数量（优先 state.extra，回退 _config）
        pop_size = state.extra.get("pop_size", 50)
        llm_init_ratio = state.extra.get("llm_init_ratio",
                                         self._config.get("llm_init_ratio", 0.2))
        k = max(1, math.ceil(llm_init_ratio * pop_size))

        s_problem = {
            "problem_structure": {
                "N": n_uavs,
                "M": n_targets,
                "model_type": model_type,
                "pop_size": pop_size,
                "requested_candidates": k,
            },
            "cost_structure": {
                "C_UT": cm_data,
                "C_TT": c_tt_data,
            },
            "constraints": constraints_desc,
            "objectives": {
                "direction": "minimize",
                "description": (
                    "Minimize total distance + alpha * max_flight_time "
                    "+ beta * constraint_violation_penalty"
                ),
            },
            "preference_summary": preference_summary,
        }

        user = (
            f"## Problem (S_problem)\n{json.dumps(s_problem, indent=2)}\n\n"
            f"## Task\nGenerate exactly {k} candidate assignments. "
            f"Each assignment must satisfy all constraints for the "
            f"'{model_type}' model type. Respond with JSON only."
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        """解析 LLM 输出的 assignment JSON。

        验证格式合法性，返回 {"assignments": [...], "reasoning": "..."}。
        """
        json_str = self._extract_json(llm_output)
        if json_str is None:
            logger.warning("[population_init] 无法从 LLM 输出中提取 JSON")
            return {"assignments": [], "reasoning": "Parse failed: no JSON found"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("[population_init] JSON 解析失败: %s", e)
            return {"assignments": [], "reasoning": f"Parse failed: invalid JSON ({e})"}

        raw_assignments = data.get("assignments", [])
        if not isinstance(raw_assignments, list):
            return {"assignments": [], "reasoning": "Parse failed: assignments not a list"}

        # 基础格式验证
        validated = []
        for i, a in enumerate(raw_assignments):
            if not isinstance(a, dict):
                continue
            uav = a.get("uav")
            targets = a.get("targets", [])
            if not isinstance(uav, int) or not isinstance(targets, list):
                continue
            if not all(isinstance(t, int) for t in targets):
                continue
            validated.append({"uav": uav, "targets": targets})

        return {
            "assignments": validated,
            "reasoning": data.get("reasoning", ""),
        }

    def apply_decision(
        self, decision: dict[str, Any], state: ModuleState
    ) -> ModuleState:
        """将解析后的 assignments 存入 decision 字典。

        由于 hook 在 before_init（此时还没有种群），
        不直接修改种群，而是将候选 assignments 存入 extra，
        由 solver 侧的 AssignmentConverter 处理。
        """
        assignments = decision.get("assignments", [])
        if not assignments:
            logger.info("[population_init] LLM 未生成有效 assignments")
            state.extra["candidate_assignments"] = []
            return state

        # 存入 state.extra，供 solver 使用
        state.extra["candidate_assignments"] = assignments
        logger.info(
            "[population_init] LLM 生成了 %d 个候选 assignments",
            len(assignments),
        )
        return state

    # ── 内部辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _build_preference_summary(
        cm: np.ndarray | None,
        n_uavs: int,
        n_targets: int,
        top_k: int,
    ) -> dict[str, Any]:
        """构建偏好摘要：每行/列的 top-k 最小代价统计。

        帮助 LLM 快速定位划算的分配，避免遍历整个矩阵。
        """
        if cm is None:
            return {}

        summary: dict[str, Any] = {}

        # UAV → Target: 每个 UAV 的 top-k 最小代价目标
        uav_preferences = []
        for i in range(n_uavs):
            row_costs = []
            for j in range(n_targets):
                val = cm[i, j]
                if np.isfinite(val):
                    row_costs.append((j, round(float(val), 2)))
            row_costs.sort(key=lambda x: x[1])
            uav_preferences.append({
                "uav": i,
                "top_targets": [
                    {"target": t, "cost": c} for t, c in row_costs[:top_k]
                ],
            })
        summary["uav_preferences"] = uav_preferences

        # Target → UAV: 每个目标的 top-k 最小代价 UAV
        target_preferences = []
        for j in range(n_targets):
            col_costs = []
            for i in range(n_uavs):
                val = cm[i, j]
                if np.isfinite(val):
                    col_costs.append((i, round(float(val), 2)))
            col_costs.sort(key=lambda x: x[1])
            target_preferences.append({
                "target": j,
                "top_uavs": [
                    {"uav": u, "cost": c} for u, c in col_costs[:top_k]
                ],
            })
        summary["target_preferences"] = target_preferences

        # SRP: Target → Target 巡游代价摘要
        if cm.shape[0] > n_uavs:
            tt_preferences = []
            for i in range(n_targets):
                tt_costs = []
                for j in range(n_targets):
                    if i == j:
                        continue
                    val = cm[n_uavs + i, j]
                    if np.isfinite(val):
                        tt_costs.append((j, round(float(val), 2)))
                tt_costs.sort(key=lambda x: x[1])
                tt_preferences.append({
                    "from_target": i,
                    "top_next_targets": [
                        {"target": t, "cost": c} for t, c in tt_costs[:top_k]
                    ],
                })
            summary["tt_preferences"] = tt_preferences

        return summary

    @staticmethod
    def _describe_constraints(
        model_type: str, n_uavs: int, n_targets: int
    ) -> dict[str, str]:
        """描述当前模型的约束条件。"""
        base = {
            "objective": "minimize total assignment cost",
            "coverage": "all targets must be covered",
        }
        if model_type == "balanced":
            base.update({
                "uniqueness": "each UAV assigned to exactly one target, each target assigned to exactly one UAV",
                "format": "each assignment has exactly 1 target in 'targets' list",
            })
        elif model_type == "overloaded":
            base.update({
                "uniqueness": "each UAV assigned to exactly one target; targets may be assigned multiple UAVs",
                "coverage_detail": f"all {n_targets} targets must appear at least once across {n_uavs} UAVs",
                "format": "each assignment has exactly 1 target in 'targets' list",
            })
        else:  # srp
            base.update({
                "uniqueness": "each target assigned to exactly one UAV; UAVs may visit multiple targets",
                "coverage_detail": f"all {n_targets} targets must be visited; {n_uavs} UAVs share the workload",
                "tour_order": "'targets' list order matters — it defines the tour sequence for that UAV",
                "format": "each assignment has >= 1 targets in 'targets' list (ordered)",
            })
        return base