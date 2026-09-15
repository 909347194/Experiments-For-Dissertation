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
    "solutions": [
        {{
            "assignments": [
                {{"uav": <int>, "targets": [<int>, ...]}},
                ...
            ]
        }},
        ...
    ],
    "reasoning": "<brief explanation of your strategy>"
}}

Each "solution" is a **complete assignment plan** covering ALL N UAVs.
Generate exactly the requested number of solutions (k).

## Assignment Rules by Model Type

### balanced (N == M, one-to-one)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- Example solution: {{"assignments": [{{"uav": 0, "targets": [2]}}, {{"uav": 1, "targets": [0]}}, {{"uav": 2, "targets": [1]}}]}}

### overloaded (N > M, UAVs outnumber targets)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.
- Example solution: {{"assignments": [{{"uav": 0, "targets": [1]}}, {{"uav": 1, "targets": [0]}}, {{"uav": 2, "targets": [1]}}]}}

### srp (N < M, UAVs visit multiple targets in sequence)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.
- Example solution: {{"assignments": [{{"uav": 0, "targets": [3, 1, 4]}}, {{"uav": 1, "targets": [2, 0]}}]}}

## Guidelines
- Focus on minimizing total cost while respecting constraints.
- Use the preference summary to identify low-cost assignments.
- For SRP, order targets to minimize transition costs (nearest-neighbor heuristic).
- Generate exactly the requested number of complete solutions (k).
- Each solution must satisfy all constraints (every target covered, etc.).
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

        # ── Global Summary ──────────────────────────────────────────
        global_summary = self._build_global_summary(
            cm, n_uavs, n_targets, model_type,
        )

        # ── Local Preference Structure ──────────────────────────────
        # TopKTargets(U_i), TopKUAVs(T_j), contested/difficult targets
        local_prefs = self._build_local_preference_structure(
            cm, n_uavs, n_targets, model_type, top_k,
        )

        # ── 约束描述 ────────────────────────────────────────────────
        constraints_desc = self._describe_constraints(model_type, n_uavs, n_targets)

        # ── 需要生成的候选数量 ──────────────────────────────────────
        pop_size = state.extra.get("pop_size", 50)
        llm_init_ratio = state.extra.get("llm_init_ratio",
                                         self._config.get("llm_init_ratio", 0.2))
        k = max(1, math.ceil(llm_init_ratio * pop_size))

        # ── 代价矩阵原始数据（小规模时附带） ────────────────────────
        _MAX_MATRIX_ELEMENTS = self._config.get("max_matrix_elements", 500)
        cm_data = None
        c_tt_data = None
        if cm is not None:
            rows, cols = cm.shape
            ut_rows = min(n_uavs, rows)
            ut_cols = min(n_targets, cols)
            total_elements = ut_rows * ut_cols
            if model_type == "srp" and rows > n_uavs:
                tt_size = min(n_targets, rows - n_uavs)
                total_elements += tt_size * tt_size

            if total_elements <= _MAX_MATRIX_ELEMENTS:
                cm_data = [
                    [round(float(cm[i, j]), 2) for j in range(ut_cols)]
                    for i in range(ut_rows)
                ]
                if model_type == "srp" and rows > n_uavs:
                    tt_size = min(n_targets, rows - n_uavs)
                    c_tt_data = [
                        [round(float(cm[n_uavs + i, j]), 2) for j in range(tt_size)]
                        for i in range(tt_size)
                    ]
            else:
                logger.info(
                    "[population_init] 矩阵元素总数 %d 超过阈值 %d，"
                    "跳过原始矩阵，仅发送 global_summary + local_preferences",
                    total_elements, _MAX_MATRIX_ELEMENTS,
                )

        s_problem = {
            "problem_structure": {
                "N": n_uavs,
                "M": n_targets,
                "model_type": model_type,
                "pop_size": pop_size,
                "requested_candidates": k,
            },
            "global_summary": global_summary,
            "local_preference_structure": local_prefs,
            "cost_matrix": {
                "C_UT": cm_data,
                "C_TT": c_tt_data,
                "note": "Only included when N*M is small enough. "
                        "Otherwise rely on global_summary + local_preferences.",
            },
            "constraints": constraints_desc,
            "objectives": {
                "direction": "minimize",
                "description": (
                    "Minimize total distance + alpha * max_flight_time "
                    "+ beta * constraint_violation_penalty"
                ),
            },
        }

        user = (
            f"## Problem (S_problem)\n{json.dumps(s_problem, indent=2)}\n\n"
            f"## Task\nGenerate exactly {k} complete assignment solutions. "
            f"Each solution must cover ALL {n_uavs} UAVs and satisfy all "
            f"constraints for the '{model_type}' model type. "
            f"Respond with JSON only."
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user},
        ]

    def parse_response(self, llm_output: str) -> dict[str, Any]:
        """解析 LLM 输出的 assignment JSON。

        支持两种格式（向后兼容）：
        1. 新格式: {"solutions": [{"assignments": [...]}, ...], "reasoning": "..."}
        2. 旧格式: {"assignments": [...], "reasoning": "..."}

        返回 {"solutions": [[{...}, ...], ...], "reasoning": "..."}。
        每个 solution 是一个完整解的 assignment 字典列表。
        """
        json_str = self._extract_json(llm_output)
        if json_str is None:
            logger.warning("[population_init] 无法从 LLM 输出中提取 JSON")
            return {"solutions": [], "reasoning": "Parse failed: no JSON found"}

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("[population_init] JSON 解析失败: %s", e)
            return {"solutions": [], "reasoning": f"Parse failed: invalid JSON ({e})"}

        # 格式 1: 新格式 {"solutions": [{"assignments": [...]}, ...]}
        raw_solutions = data.get("solutions", None)
        if raw_solutions is not None:
            if not isinstance(raw_solutions, list):
                return {"solutions": [], "reasoning": "Parse failed: solutions not a list"}
            validated_solutions = []
            for sol in raw_solutions:
                if not isinstance(sol, dict):
                    continue
                raw_assignments = sol.get("assignments", [])
                validated = self._validate_assignments(raw_assignments)
                if validated:
                    validated_solutions.append(validated)
            return {
                "solutions": validated_solutions,
                "reasoning": data.get("reasoning", ""),
            }

        # 格式 2: 旧格式兼容 {"assignments": [...]}
        raw_assignments = data.get("assignments", [])
        if not isinstance(raw_assignments, list):
            return {"solutions": [], "reasoning": "Parse failed: assignments not a list"}
        validated = self._validate_assignments(raw_assignments)
        if validated:
            return {
                "solutions": [validated],
                "reasoning": data.get("reasoning", ""),
            }
        return {"solutions": [], "reasoning": data.get("reasoning", "")}

    @staticmethod
    def _validate_assignments(raw_assignments: list) -> list[dict[str, Any]]:
        """验证并过滤单个 solution 内的 assignments。"""
        if not isinstance(raw_assignments, list):
            return []
        validated = []
        for a in raw_assignments:
            if not isinstance(a, dict):
                continue
            uav = a.get("uav")
            targets = a.get("targets", [])
            if not isinstance(uav, int) or not isinstance(targets, list):
                continue
            if not all(isinstance(t, int) for t in targets):
                continue
            validated.append({"uav": uav, "targets": targets})
        return validated

    def apply_decision(
        self, decision: dict[str, Any], state: ModuleState
    ) -> ModuleState:
        """将解析后的 solutions 存入 state.extra。

        由于 hook 在 before_init（此时还没有种群），
        不直接修改种群，而是将候选完整解存入 extra，
        由 solver 侧的 AssignmentConverter 处理。
        """
        solutions = decision.get("solutions", [])
        if not solutions:
            logger.info("[population_init] LLM 未生成有效 solutions")
            state.extra["candidate_solutions"] = []
            return state

        # 存入 state.extra，供 solver 使用
        # 格式: list[list[dict]]，每个元素是一个完整解的 assignments 列表
        state.extra["candidate_solutions"] = solutions
        logger.info(
            "[population_init] LLM 生成了 %d 个候选完整解",
            len(solutions),
        )
        return state

    # ── 内部辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _build_global_summary(
        cm: np.ndarray | None,
        n_uavs: int,
        n_targets: int,
        model_type: str,
    ) -> dict[str, Any]:
        """构建 Global Summary：问题规模 + 代价统计 + 可行性统计。

        比简单的 shape/min/max/mean 更丰富，帮助 LLM 理解问题特征。
        """
        if cm is None:
            return {}

        rows, cols = cm.shape
        ut_rows = min(n_uavs, rows)
        ut_cols = min(n_targets, cols)

        # ── C_UT 统计 ────────────────────────────────────────────
        ut_vals = []
        for i in range(ut_rows):
            for j in range(ut_cols):
                v = float(cm[i, j])
                if np.isfinite(v):
                    ut_vals.append(v)

        ut_stats = {}
        if ut_vals:
            arr = np.array(ut_vals)
            ut_stats = {
                "min": round(float(arr.min()), 2),
                "max": round(float(arr.max()), 2),
                "mean": round(float(arr.mean()), 2),
                "std": round(float(arr.std()), 2),
            }

        # ── C_TT 统计（SRP） ─────────────────────────────────────
        tt_stats = {}
        if model_type == "srp" and rows > n_uavs:
            tt_size = min(n_targets, rows - n_uavs)
            tt_vals = []
            for i in range(tt_size):
                for j in range(tt_size):
                    if i == j:
                        continue
                    v = float(cm[n_uavs + i, j])
                    if np.isfinite(v):
                        tt_vals.append(v)
            if tt_vals:
                arr = np.array(tt_vals)
                tt_stats = {
                    "min": round(float(arr.min()), 2),
                    "max": round(float(arr.max()), 2),
                    "mean": round(float(arr.mean()), 2),
                    "std": round(float(arr.std()), 2),
                }

        # ── Feasibility Statistics ────────────────────────────────
        n_infeasible = 0
        n_finite = 0
        targets_with_few_feasible = 0
        uavs_with_few_feasible = 0

        for j in range(ut_cols):
            feasible_count = sum(1 for i in range(ut_rows) if np.isfinite(cm[i, j]))
            if feasible_count <= 2:
                targets_with_few_feasible += 1

        for i in range(ut_rows):
            feasible_count = sum(1 for j in range(ut_cols) if np.isfinite(cm[i, j]))
            if feasible_count <= 2:
                uavs_with_few_feasible += 1

        for i in range(ut_rows):
            for j in range(ut_cols):
                n_finite += 1
                if not np.isfinite(cm[i, j]):
                    n_infeasible += 1

        infeasible_pct = round(100.0 * n_infeasible / max(n_finite, 1), 1)

        summary = {
            "problem_scale": {
                "N": n_uavs,
                "M": n_targets,
                "model_type": model_type,
            },
            "cost_statistics": {
                "C_UT": ut_stats,
            },
            "feasibility_statistics": {
                "infeasible_pairs_pct": infeasible_pct,
                "targets_with_few_feasible_UAVs": targets_with_few_feasible,
                "UAVs_with_few_feasible_targets": uavs_with_few_feasible,
            },
        }

        if tt_stats:
            summary["cost_statistics"]["C_TT"] = tt_stats

        return summary

    @staticmethod
    def _build_local_preference_structure(
        cm: np.ndarray | None,
        n_uavs: int,
        n_targets: int,
        model_type: str,
        top_k: int,
    ) -> dict[str, Any]:
        """构建 Local Preference Structure：TopK + Contested/Difficult 识别。

        包含：
        - TopKTargets(U_i): 每个 UAV 的 top-k 最小代价目标
        - TopKUAVs(T_j): 每个目标的 top-k 最小代价 UAV
        - Contested targets: 多个 UAV 都偏好的高竞争目标
        - Difficult targets: 可行 UAV 很少的目标
        - SRP: TopKNextTargets(T_j) 巡游转移偏好
        """
        if cm is None:
            return {}

        rows, cols = cm.shape
        ut_rows = min(n_uavs, rows)
        ut_cols = min(n_targets, cols)

        # ── TopKTargets(U_i) ─────────────────────────────────────
        uav_preferences = []
        for i in range(ut_rows):
            row_costs = []
            for j in range(ut_cols):
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

        # ── TopKUAVs(T_j) ────────────────────────────────────────
        target_preferences = []
        for j in range(ut_cols):
            col_costs = []
            for i in range(ut_rows):
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

        # ── Contested Targets ─────────────────────────────────────
        target_popularity: dict[int, int] = {}
        for pref in uav_preferences:
            for entry in pref["top_targets"]:
                tgt = entry["target"]
                target_popularity[tgt] = target_popularity.get(tgt, 0) + 1

        contested = sorted(target_popularity.items(), key=lambda x: -x[1])
        contested_targets = [
            {"target": t, "preferred_by_count": c}
            for t, c in contested
            if c >= 2
        ][:top_k * 2]

        # ── Difficult Targets ─────────────────────────────────────
        difficult_targets = []
        for j in range(ut_cols):
            feasible_count = sum(
                1 for i in range(ut_rows) if np.isfinite(cm[i, j])
            )
            if feasible_count <= 2:
                difficult_targets.append({
                    "target": j,
                    "feasible_UAV_count": feasible_count,
                })

        result: dict[str, Any] = {
            "TopKTargets_per_UAV": uav_preferences,
            "TopKUAVs_per_target": target_preferences,
            "contested_targets": contested_targets,
            "difficult_targets": difficult_targets,
        }

        # ── SRP: TopKNextTargets(T_j) ────────────────────────────
        if model_type == "srp" and rows > n_uavs:
            tt_preferences = []
            for i in range(ut_cols):
                tt_costs = []
                for j in range(ut_cols):
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
            result["TopKNextTargets_per_target"] = tt_preferences

        return result

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