# -*- coding: utf-8 -*-
"""LLM 提示词统一管理模块

职责：
    集中管理所有 LLM 模块的提示词模板，支持：
    1. 默认提示词（Python 模块定义）
    2. 外部文件覆盖（实验配置目录）
    3. 场景化提示词（N=M/N>M/N<M）— 根据 N,M 关系自动选择
    4. 动态参数插值（.format() 风格）
    5. User Prompt 模板化

使用方式：
    from llm.prompts import get_prompt

    # 获取 system prompt（通用）
    sys_prompt = get_prompt("population_init")

    # 获取场景化 system prompt（根据 N,M 自动选择 balanced/overloaded/srp）
    sys_prompt = get_prompt("population_init", model_type="srp")

    # 获取带参数的 system prompt
    sys_prompt = get_prompt("search_controller", cr_choices=[0.1, 0.3, 0.5])

    # 从外部文件加载 system prompt
    sys_prompt = get_prompt("population_init", prompt_path="/path/to/prompt.txt")

    # 获取 user prompt 模板（支持动态参数）
    user_prompt = get_prompt("population_init", prompt_type="user",
                             problem_json="...", k=5, n_uavs=10, model_type="balanced")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── 公共片段 ─────────────────────────────────────────────────

_JSON_FORMAT_COMMON = """\
## Output Format
Think step by step **before** writing the JSON. \\\nRecord your reasoning in the "thought" field, then output the solutions.
Respond with a JSON object only (no markdown):
"""

_REASONING_GUIDE = """\
## Reasoning (in the "thought" field)
Before generating assignments, briefly reason:
1. Which targets are **difficult** (few feasible UAVs)? Assign first.
2. Which targets are **contested** (preferred by many UAVs)? Assign to best-cost UAV.
3. Fill remaining using preference rankings. For SRP: use nearest-neighbor for tour order.
"""

_OUTPUT_VALIDATION = """\
## Output Validation
- Each solution must contain exactly N assignments (one per UAV).
- UAV IDs must be in [0, N-1], target IDs in [0, M-1].
- For balanced/overloaded: each "targets" list has exactly 1 element.
- For srp: each "targets" list has >= 1 elements, no duplicate targets across solutions.
- Invalid outputs will be rejected. Double-check before responding.
"""


# =============================================================================
# Population Init — Scene-specific System Prompts
# =============================================================================

_POP_INIT_BALANCED = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a **balanced** UAV-target problem \
(N == M, one-to-one mapping). The solver will convert your assignments into an \
evolutionary algorithm's internal encoding, so you only need to produce \
**discrete assignments**.

{_json_format}
{{
    "thought": "<brief reasoning: difficult targets, contested targets, assignment strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [2]}},
                {{"uav": 1, "targets": [0]}},
                {{"uav": 2, "targets": [1]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete one-to-one assignment** covering ALL N UAVs.

## Rules (balanced: N == M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- This is a **permutation problem**: find the best UAV→Target matching.

{_reasoning}
{_validation}
{_guidelines}
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
    _guidelines="""\
## Guidelines
- Focus on minimizing total assignment cost.
- Use TopKTargets_per_UAV and TopKUAVs_per_target to identify low-cost assignments.
- Pay attention to contested_targets: assign them to the UAV with the best cost advantage.
- Pay attention to difficult_targets: these have few feasible UAVs and must be handled first.
""",
)

_POP_INIT_OVERLOADED = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for an **overloaded** UAV-target problem \
(N > M, more UAVs than targets). The solver will convert your assignments into an \
evolutionary algorithm's internal encoding, so you only need to produce \
**discrete assignments**.

{_json_format}
{{
    "thought": "<brief reasoning: target coverage, cost-optimal UAV assignments>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [1]}},
                {{"uav": 1, "targets": [0]}},
                {{"uav": 2, "targets": [1]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment** covering ALL N UAVs.

## Rules (overloaded: N > M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target **must appear at least once** across all assignments.
- Each "targets" list has exactly 1 element.
- Multiple UAVs may be assigned to the same target.

{_reasoning}
{_validation}
{_guidelines}
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
    _guidelines="""\
## Guidelines
- Focus on minimizing total assignment cost.
- **Critical**: ensure every target is covered by at least one UAV.
- Use TopKUAVs_per_target to find the best UAV for each target.
- Use TopKTargets_per_UAV to distribute surplus UAVs to low-cost targets.
- Pay attention to difficult_targets (few feasible UAVs) — these must be assigned first.
""",
)

_POP_INIT_SRP = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a **single-route patrol (SRP)** problem \
(N < M, each UAV visits multiple targets in sequence). The solver will convert your \
assignments into an evolutionary algorithm's internal encoding, so you only need to \
produce **discrete assignments**.

{_json_format}
{{
    "thought": "<brief reasoning: target distribution among UAVs, tour ordering by nearest-neighbor>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": 0, "targets": [3, 1, 4]}},
                {{"uav": 1, "targets": [2, 0]}}
            ]
        }}
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment** covering ALL N UAVs and ALL M targets.

## Rules (srp: N < M)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the **order represents the tour sequence**.
- Tour costs include both UAV→first_target (C_UT) and target→target transitions (C_TT).

{_reasoning}

For SRP specifically:
1. Divide targets among UAVs (each UAV gets at least 1 target).
2. For each UAV's target list, order by **nearest-neighbor** to minimize transition costs.
3. Use TopKNextTargets_per_target to find low-cost transitions.

{_validation}
{_guidelines}
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
    _guidelines="""\
## Guidelines
- Minimize total cost = sum of C_UT (UAV→first target) + C_TT (target→target transitions).
- Use TopKTargets_per_UAV to assign each UAV to its nearest initial target.
- Use TopKNextTargets_per_target to build efficient tour sequences.
- Pay attention to difficult_targets (few feasible UAVs) — assign these first.
- Avoid creating long tours for a single UAV when targets are geographically spread out.
""",
)

# 通用回退（包含所有场景规则，用于 model_type 未知时）
_POP_INIT_GENERIC = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a UAV scheduling problem. \
The solver will convert your assignments into an evolutionary algorithm's \
internal encoding, so you only need to produce **discrete assignments**.

{_json_format}
{{
    "thought": "<brief reasoning: difficult targets, contested targets, assignment strategy>",
    "solutions": [
        {{
            "assignments": [
                {{"uav": <int>, "targets": [<int>, ...]}},
                ...
            ]
        }},
        ...
    ],
    "reasoning": "<brief explanation>"
}}

Each "solution" is a **complete assignment plan** covering ALL N UAVs.

## Assignment Rules by Model Type

### balanced (N == M, one-to-one)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- Example: {{"assignments": [{{"uav": 0, "targets": [2]}}, {{"uav": 1, "targets": [0]}}, {{"uav": 2, "targets": [1]}}]}}

### overloaded (N > M, UAVs outnumber targets)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.
- Example: {{"assignments": [{{"uav": 0, "targets": [1]}}, {{"uav": 1, "targets": [0]}}, {{"uav": 2, "targets": [1]}}]}}

### srp (N < M, UAVs visit multiple targets in sequence)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.
- Example: {{"assignments": [{{"uav": 0, "targets": [3, 1, 4]}}, {{"uav": 1, "targets": [2, 0]}}]}}

{_reasoning}
{_validation}
{_guidelines}
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
    _guidelines="## Guidelines\n- Focus on minimizing total cost while respecting constraints.\n- Use the preference summary to identify low-cost assignments.\n",
)

# Scene → Prompt 映射
_POP_INIT_SCENE_MAP: dict[str, str] = {
    "balanced": _POP_INIT_BALANCED,
    "overloaded": _POP_INIT_OVERLOADED,
    "srp": _POP_INIT_SRP,
}


# =============================================================================
# Population Init — User Prompt
# =============================================================================

POPULATION_INIT_USER_PROMPT = """\
## Problem (S_problem)
{problem_json}

## Task
Generate exactly {k} complete assignment solutions. \
Each solution must cover ALL {n_uavs} UAVs and satisfy all \
constraints for the '{model_type}' model type. \
Respond with JSON only.
"""


# =============================================================================
# Search Controller — Scene-specific System Prompts
# =============================================================================

_SC_BASE = """\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: Select the crossover rate (CR) from {cr_choices}.

The scaling factor F will be automatically computed from your chosen CR \
using the DMDE parameter relationship (formula 3-11).

## How CR Affects Search
In the DMDE hybrid strategy (formula 3-10), each gene independently uses:
- DE/rand/1 (exploration) when random value <= CR
- DE/best/2 (exploitation) when random value > CR
"""

_SC_CR_GUIDE = """\
Therefore:
- High CR (0.7/0.9): More genes use rand/1 → broader exploration
  Best when diversity is LOW or stagnation is HIGH.
- Low CR (0.1/0.3): More genes use best/2 → focused exploitation
  Best when diversity is HIGH but convergence is SLOW.
- Mid CR (0.5): Balanced exploration/exploitation.
"""

_SC_FORMAT = """\
## Decision Format
Respond with a JSON object only (no markdown):
{{
    "cr": <one of {cr_choices}>,
    "reasoning": "<brief explanation of your CR choice based on the current state>"
}}
"""

_SC_BALANCED_EXTRA = """\
## Scene: balanced (N == M)
- One-to-one assignment: each UAV maps to exactly one target.
- The search space is a **permutation space** — diversity drops quickly.
- Prefer moderate CR (0.3-0.5) to maintain exploration without disrupting good permutations.
- Only increase CR above 0.5 when stagnation is clearly detected.
"""

_SC_OVERLOADED_EXTRA = """\
## Scene: overloaded (N > M)
- Multiple UAVs share targets — the search space has **redundancy**.
- Diversity is naturally higher due to target sharing.
- Can use lower CR (0.1-0.3) to focus exploitation on promising regions.
- Increase CR if the solver struggles to cover all targets.
"""

_SC_SRP_EXTRA = """\
## Scene: srp (N < M)
- Each UAV visits multiple targets in sequence — **tour ordering** matters.
- The search space is larger (assignment + ordering).
- Higher CR (0.5-0.7) helps explore different tour orderings.
- Be cautious with very high CR (>0.7) — it may disrupt good tour segments.
"""

_SC_SCENE_MAP: dict[str, str] = {
    "balanced": _SC_BALANCED_EXTRA,
    "overloaded": _SC_OVERLOADED_EXTRA,
    "srp": _SC_SRP_EXTRA,
}


def get_search_controller_prompt(
    cr_choices: list[float] | None = None,
    model_type: str | None = None,
) -> str:
    """获取搜索控制器 system prompt。

    Args:
        cr_choices: CR 候选值列表
        model_type: 场景类型 ("balanced"/"overloaded"/"srp")，
                     非空时追加场景特定的 CR 调控建议
    """
    if cr_choices is None:
        cr_choices = [0.1, 0.3, 0.5, 0.7, 0.9]

    base = _SC_BASE.format(cr_choices=cr_choices)
    scene_extra = _SC_SCENE_MAP.get(model_type, "") if model_type else ""
    fmt = _SC_FORMAT.format(cr_choices=cr_choices)

    return base + _SC_CR_GUIDE + scene_extra + fmt


# =============================================================================
# Search Controller — User Prompt
# =============================================================================

SEARCH_CONTROLLER_USER_PROMPT = """\
## Current Search State
{state_json}

## Recent Trajectory
{trajectory_text}

## Task
Select the best CR for the next interval. Respond with JSON only.
"""


# =============================================================================
# CR Control (deprecated) — Prompts
# =============================================================================

CR_CONTROL_SYSTEM_PROMPT = """\
You are an expert in Differential Evolution parameter control.

Your task: Adjust the crossover rate (CR) and scale factor (F) offsets \
based on the current search state.

## Decision Format
Respond with a JSON object only (no markdown):
{
    "cr_offset": <float in [-0.3, 0.3], or null to keep default>,
    "f_offset": <float in [-0.3, 0.3], or null to keep default>,
    "reasoning": "<brief explanation>"
}

## Guidelines
- High stagnation + low diversity → increase CR (more exploration)
- Converging well → decrease CR slightly (more exploitation)
- Large fitness variance → increase F (bigger steps)
- Near convergence → decrease F (fine-tuning)
- Only adjust when the state clearly warrants a change (prefer null)
"""

CR_CONTROL_USER_PROMPT = """\
## Current Search State
{state_json}

## Task
Adjust CR and F offsets for the next generation. Respond with JSON only.
"""


# =============================================================================
# Operator Selection (deprecated) — Prompts
# =============================================================================

def get_operator_selection_prompt(strategies: list[str] | None = None) -> str:
    if strategies is None:
        strategies = ["rand/1", "best/1", "best/2", "current-to-pbest/1", "rand/2", "rand-to-best/1"]
    strategies_str = ", ".join(strategies)
    return f"""\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: Select the most suitable DE operator strategy for the current search state.

## Available Strategies
{strategies_str}

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "strategy": "<strategy name>",
    "reasoning": "<brief explanation>"
}}

## Guidelines
- Low diversity + stagnation → exploratory (rand/1, rand/2)
- High diversity + slow convergence → exploitative (best/1, best/2)
- Balanced state → balanced (current-to-pbest/1)
"""

OPERATOR_SELECTION_USER_PROMPT = """\
## Current Search State
{state_json}

## Recent Trajectory
{trajectory_text}

## Task
Select the best DE strategy for the next interval. Respond with JSON only.
"""


# =============================================================================
# Scene-specific System Prompt Dispatcher
# =============================================================================

def get_scene_specific_system_prompt(
    module_name: str,
    model_type: str,
    **kwargs: Any,
) -> str:
    """获取针对特定场景（model_type）的系统提示词。

    根据 N,M 关系自动选择对应的 prompt，只包含该场景的规则和示例，
    去除无关场景的信息噪声，节省 token。

    Args:
        module_name: 模块名称 ("population_init", "search_controller")
        model_type: 场景类型 ("balanced", "overloaded", "srp")
        **kwargs: 额外参数（如 cr_choices）

    Returns:
        针对该场景的系统提示词
    """
    if module_name == "population_init":
        prompt = _POP_INIT_SCENE_MAP.get(model_type)
        if prompt is None:
            logger.warning(
                "[prompts] Unknown model_type %r for population_init, using generic prompt",
                model_type,
            )
            return _POP_INIT_GENERIC
        return prompt

    if module_name == "search_controller":
        return get_search_controller_prompt(model_type=model_type, **kwargs)

    raise ValueError(
        f"Unknown module for scene-specific prompt: {module_name!r}. "
        f"Available: ['population_init', 'search_controller']"
    )


# =============================================================================
# Registries
# =============================================================================

PROMPT_REGISTRY: dict[str, str] = {
    "population_init": _POP_INIT_GENERIC,
    "cr_control": CR_CONTROL_SYSTEM_PROMPT,
}

USER_PROMPT_REGISTRY: dict[str, str] = {
    "population_init": POPULATION_INIT_USER_PROMPT,
    "cr_control": CR_CONTROL_USER_PROMPT,
    "search_controller": SEARCH_CONTROLLER_USER_PROMPT,
    "operator_selection": OPERATOR_SELECTION_USER_PROMPT,
}


# =============================================================================
# Unified Interface
# =============================================================================

def get_prompt(
    module_name: str,
    prompt_type: str = "system",
    prompt_path: str | Path | None = None,
    model_type: str | None = None,
    **kwargs: Any,
) -> str:
    """获取指定模块的提示词。

    优先级：外部文件 > 场景化 prompt > 通用默认 prompt

    Args:
        module_name: 模块名称
        prompt_type: "system" 或 "user"
        prompt_path: 外部文件路径（覆盖默认）
        model_type: 场景类型 ("balanced"/"overloaded"/"srp")，用于场景化 system prompt
        **kwargs: 动态参数（用于 user prompt 插值或 search_controller 的 cr_choices）

    Returns:
        提示词字符串
    """
    # 优先从外部文件加载
    if prompt_path is not None:
        p = Path(prompt_path)
        if p.exists():
            logger.info("[prompts] Loading prompt from external file: %s", prompt_path)
            return p.read_text(encoding="utf-8")
        else:
            logger.warning("[prompts] Prompt file not found: %s, using default", prompt_path)

    # System prompt
    if prompt_type == "system":
        # 有 model_type 时优先用场景化 prompt
        if model_type is not None:
            return get_scene_specific_system_prompt(module_name, model_type, **kwargs)

        if module_name in PROMPT_REGISTRY:
            return PROMPT_REGISTRY[module_name]

        if module_name == "search_controller":
            return get_search_controller_prompt(**kwargs)

        if module_name == "operator_selection":
            return get_operator_selection_prompt(**kwargs)

        raise ValueError(
            f"Unknown system prompt module: {module_name!r}. "
            f"Available: {list(PROMPT_REGISTRY.keys()) + ['search_controller', 'operator_selection']}"
        )

    # User prompt
    elif prompt_type == "user":
        if module_name not in USER_PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown user prompt module: {module_name!r}. "
                f"Available: {list(USER_PROMPT_REGISTRY.keys())}"
            )

        template = USER_PROMPT_REGISTRY[module_name]

        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(
                "[prompts] Missing parameter %s for user prompt '%s', using raw template",
                e, module_name,
            )
            return template

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}. Use 'system' or 'user'.")


def load_prompt_from_file(prompt_path: str | Path, fallback: str | None = None) -> str:
    """从文件加载提示词，失败时返回 fallback。"""
    p = Path(prompt_path)
    if not p.exists():
        logger.warning("[prompts] File not found: %s", prompt_path)
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return p.read_text(encoding="utf-8")


__all__ = [
    "get_prompt",
    "load_prompt_from_file",
    "get_scene_specific_system_prompt",
    "get_search_controller_prompt",
    "get_operator_selection_prompt",
    "POPULATION_INIT_USER_PROMPT",
    "CR_CONTROL_SYSTEM_PROMPT",
    "CR_CONTROL_USER_PROMPT",
    "SEARCH_CONTROLLER_USER_PROMPT",
    "OPERATOR_SELECTION_USER_PROMPT",
    "PROMPT_REGISTRY",
    "USER_PROMPT_REGISTRY",
]