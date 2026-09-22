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
    sys_prompt = get_prompt("search_controller")

    # 获取场景化 system prompt（根据 N,M 自动选择 balanced/overloaded/srp）
    sys_prompt = get_prompt("search_controller", model_type="srp")

    # 获取带参数的 system prompt
    sys_prompt = get_prompt("search_controller", cr_choices=[0.1, 0.3, 0.5])

    # 从外部文件加载 system prompt
    sys_prompt = get_prompt("search_controller", prompt_path="/path/to/prompt.txt")

    # 获取 user prompt 模板（支持动态参数）
    user_prompt = get_prompt("search_controller", prompt_type="user",
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
Briefly consider the relevant problem characteristics before generating the solutions. \\\nRecord a concise rationale in the "thought" field, then output the solutions.
Respond with a JSON object only (no markdown):
"""

_REASONING_GUIDE = """\
## Reasoning (in the "thought" field)
Before generating solutions, reason through the problem structure:
1. Which targets are **difficult** (few feasible UAVs)? Assign these first.
2. Which targets are **contested** (many UAVs prefer them)? Decide allocation early.
3. What is the approximate cost lower bound? Use it to gauge solution quality.
4. For SRP: how should targets be partitioned among UAVs, and what tour order minimizes transition costs?
"""

_OUTPUT_VALIDATION = """\
## Output Validation
- Each solution must contain exactly N assignments (one per UAV).
- UAV IDs must be in [0, N-1], target IDs in [0, M-1].
- **VERIFICATION STEP**: Before outputting, for each solution check:
  1. Count total targets — must equal M (balanced/overloaded: each target ≥ 1 time; SRP: exactly 1 time).
  2. Check constraint compliance for the specific model type.
  3. Check all targets in [0, M-1] are covered.
  If any check fails, fix the solution before outputting.
- Invalid outputs will be rejected. Double-check before responding.

## Output Quality
- Focus on generating HIGH-QUALITY solutions with low total cost.
- The "thought" field should explain your construction strategy.
- Output ONLY the JSON object. No markdown fences, no preamble.
"""


# Search Controller — Strategy Selection Prompts
# =============================================================================

_SC_BASE = """\
You are a strategy selector for a Differential Evolution (DE) solver on a \
combinatorial optimization problem (UAV-target assignment).

Your job: read the search state and trajectory, then choose ONE strategy.

## Strategies
{strategy_table}

Each strategy maps to a fixed (CR, F, GMR) configuration. You do NOT set \
numeric parameter values directly — you choose the strategy whose semantics \
match what the search needs right now.

## When to Switch
- **Converged / fine-tuning** (stagnation > 50, acceptance < 0.01, diversity healthy): \
  NORMAL late-stage convergence — NOT a failure. Default to `exploit` (refine the basin) \
  or `hold`. Do NOT reset; a forced reset discards a good solution and stalls convergence.
- **Search improving** (df < 0, acceptance reasonable) → `hold` or `exploit`.
- **Mild stagnation** (stagnation rising but diversity healthy) → `hold` first; only \
  switch strategy after ≥2 consecutive stages with no improvement AND the shadow \
  baseline shows your last change actually helped.
- **Genuine diversity loss** (delta_diversity < -0.08, population contracting) → `explore` \
  (broad search, NO forced reset) to re-inject variety; reserve `recover` for when explore \
  fails across 2+ stages.
- **`recover` is a forced global reset and LAST RESORT ONLY**: it perturbs the ENTIRE \
  population (forced extinction, keeps only the best ~30%) and discards recent progress. \
  Use it ONLY when genuinely trapped — diversity collapsed AND long stagnation — after \
  cheaper strategies failed. If in doubt, choose `exploit` or `hold`.
- **Default bias**: when unsure, prefer `hold`/`exploit` over `explore`. `explore` does \
  NOT reset the population, but its high CR/F still trades convergence speed for breadth \
  — don't reach for it on every stagnant stage.

## Shadow Attribution
df_vs_shadow = your df minus the fixed-baseline df. If |df_vs_shadow| < 0.05%, \
your strategy choice is indistinguishable from doing nothing — that is NOT \
evidence that the strategy is bad, only that the search has no room to move.
If df_vs_shadow > 0.05%, your strategy is outperforming the baseline → hold.
If df_vs_shadow < -0.05%, your strategy is hurting → switch.

## Decision
If the current strategy is working (df improving or df_vs_shadow > 0), HOLD it.
If the search is stuck AND the current strategy has not helped across 2+ stages, \
switch to a different strategy. Do NOT switch just because you were consulted.

## Output Format
{output_format}
"""

_SC_STRATEGY_TABLE = """\
| Strategy   | CR   | F    | GMR  | Use when |
|------------|------|------|------|----------|
| hold       | keep | keep | keep | Current strategy is working |
| explore    | 0.8  | 0.9  | off  | Broad search (high CR/F, rand/1); NO forced reset |
| balanced   | 0.5  | 0.5  | auto | Normal search, balanced exploration |
| exploit    | 0.3  | 0.3  | off  | Near convergence, fine-tune best |
| recover    | 0.5  | 0.7  | on   | LAST RESORT: genuine trap only (diversity collapsing + long stagnation) |
| rand-1     | 0.7  | 1.0  | auto | Maximum exploration (DE/rand/1 only) |
| best-1     | 0.2  | 0.3  | off  | Maximum exploitation (DE/best/2 only) |"""

_SC_OUTPUT_FORMAT = """\
Respond with a JSON object only (no markdown):
{{
    "evidence_read": "state: <1 sentence> | strategy_effect: <1 sentence> | history: <1 sentence>",
    "strategy": "hold" | "explore" | "balanced" | "exploit" | "recover" | "rand-1" | "best-1",
    "override_cr": null or float (leave null unless specific reason),
    "override_f": null or float (leave null unless specific reason),
    "reasoning": "<one sentence>"
}}"""

_SC_SCENE_MAP: dict[str, str] = {
    "balanced": "Permutation space (N=M). Diversity drops fast. Use explore early, exploit late.",
    "overloaded": "Redundancy space (N>M). Many near-equivalent solutions. Diversity naturally high.",
    "srp": "Tour space (N<M). Assignment + ordering. Complex landscape. Use balanced or explore.",
}

SEARCH_CONTROLLER_SYSTEM_PROMPT_TEMPLATE = """{base}

## Scene Notes
{scene_note}"""

SEARCH_CONTROLLER_USER_PROMPT = """\
## Search State
{state_json}

## Stage History
{history_text}

## Task
Choose ONE strategy for the next stage. If the current strategy is working, \
choose "hold". Respond with JSON only.
"""


def get_search_controller_prompt(
    strategies: dict | None = None,
    model_type: str | None = None,
    **kwargs,
) -> str:
    """构建策略选择 System Prompt。"""
    if strategies is None:
        from ..modules.search_controller import STRATEGIES
        strategies = STRATEGIES

    strategy_table = _SC_STRATEGY_TABLE
    output_format = _SC_OUTPUT_FORMAT

    base = _SC_BASE.format(
        strategy_table=strategy_table,
        output_format=output_format,
    )

    scene_note = _SC_SCENE_MAP.get(model_type, "General DE search.")
    return SEARCH_CONTROLLER_SYSTEM_PROMPT_TEMPLATE.format(
        base=base,
        scene_note=scene_note,
    )

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

The `previous_decision_feedback` section shows the effect of your last decision.
Use this feedback to evaluate whether your previous adjustment helped or hurt,
and adapt your next decision accordingly.

## Task
Adjust CR and F offsets for the next generation.
Base your decision on the **current state and feedback trajectory**, \
not on preset rules for the scenario.
Respond with JSON only.
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
        module_name: 模块名称 ("search_controller")
        model_type: 场景类型 ("balanced", "overloaded", "srp")
        **kwargs: 额外参数（如 cr_choices）

    Returns:
        针对该场景的系统提示词
    """

    if module_name == "search_controller":
        return get_search_controller_prompt(model_type=model_type, **kwargs)

    raise ValueError(
        f"Unknown module for scene-specific prompt: {module_name!r}. "
        f"Available: ['search_controller']"
    )


# =============================================================================
# Registries
# =============================================================================

PROMPT_REGISTRY: dict[str, str] = {
    "cr_control": CR_CONTROL_SYSTEM_PROMPT,
}

USER_PROMPT_REGISTRY: dict[str, str] = {
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

        # model_type 是 get_prompt 的命名参数，不会自动进入 **kwargs，
        # 但 user prompt 模板中可能需要它，因此手动注入。
        if model_type is not None and "model_type" not in kwargs:
            kwargs = {**kwargs, "model_type": model_type}

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
    "CR_CONTROL_SYSTEM_PROMPT",
    "CR_CONTROL_USER_PROMPT",
    "SEARCH_CONTROLLER_USER_PROMPT",
    "OPERATOR_SELECTION_USER_PROMPT",
    "PROMPT_REGISTRY",
    "USER_PROMPT_REGISTRY",
]