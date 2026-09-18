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
    "thought": "<your construction strategy>",
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
- The set of all target IDs in a solution must be exactly [0, 1, ..., M-1].

{_reasoning}

## Construction Strategy
Use the following two-phase approach to generate high-quality, diverse solutions:

### Phase 1: Greedy Baseline (solution 1)
1. Identify **contested targets** (preferred by many UAVs). For each, assign the \
   lowest-cost UAV. If a UAV is already taken, use the next cheapest.
2. For remaining unassigned UAVs, greedily assign to the cheapest available target.
3. Compute total cost = sum(C_UT[uav][target]). This is your quality reference.

### Phase 2: Perturbations (solutions 2-K)
Generate variants by perturbing the baseline:
- **Swap**: exchange 2-3 UAV-target pairs and check if cost improves.
- **Contested reassignment**: pick one contested target, assign to a different UAV.
- **Greedy restart**: start the greedy construction from a different contested target.

{_validation}

## Guidelines
- Minimize total assignment cost = sum of C_UT[uav][target] for all assignments.
- Use TopKTargets_per_UAV and TopKUAVs_per_target to identify low-cost pairings.
- Pay attention to contested_targets — assigning them optimally is often the key differentiator.
- If a target has only 1-2 feasible UAVs (difficult_targets), lock those assignments early.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

_POP_INIT_OVERLOADED = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for an **overloaded** UAV-target problem \
(N > M, more UAVs than targets). The solver will convert your assignments into an \
evolutionary algorithm's internal encoding, so you only need to produce \
**discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
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

## Construction Strategy
### Phase 1: Coverage-First Baseline (solution 1)
1. Identify **difficult targets** (few feasible UAVs). Assign their best UAV first.
2. For each remaining uncovered target, assign the cheapest available UAV.
3. Distribute remaining surplus UAVs to their lowest-cost targets.
4. Compute total cost = sum(C_UT[uav][target]). This is your quality reference.

### Phase 2: Perturbations (solutions 2-K)
- **Surplus redistribution**: move 1-2 surplus UAVs to different targets.
- **Coverage swap**: swap assignments between two UAVs, maintain coverage.
- **Greedy restart**: prioritize a different target for the initial assignment.

{_validation}

## Guidelines
- Minimize total assignment cost.
- **Critical**: ensure every target is covered by at least one UAV.
- Use TopKUAVs_per_target to find the best UAV for each target.
- Use TopKTargets_per_UAV to distribute surplus UAVs to low-cost targets.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

_POP_INIT_SRP = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a **single-route patrol (SRP)** problem \
(N < M, each UAV visits multiple targets in sequence). The solver will convert your \
assignments into an evolutionary algorithm's internal encoding, so you only need to \
produce **discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
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
- Tour cost = C_UT[uav][first_target] + sum(C_TT[prev][next]) for transitions.

{_reasoning}

## Construction Strategy
### Phase 1: Partition + Nearest-Neighbor Tour (solution 1)
1. **Partition** targets among UAVs:
   - Identify difficult targets (few feasible UAVs), assign to their best UAV.
   - Group remaining targets by spatial proximity (use TopKTargets_per_UAV as anchor).
   - Each UAV should get roughly M/N targets.
2. **Order** each UAV's tour:
   - Start with the target that has lowest C_UT[uav][target].
   - Use nearest-neighbor: from current target, go to the closest unvisited target \
     (use C_TT or C_TT_sparse TopK adjacency).
3. Compute total cost = C_UT[uav][first] + sum(C_TT transitions).

### Phase 2: Tour Perturbations (solutions 2-K)
- **2-opt swap**: reverse a segment within one UAV's tour.
- **Target reassignment**: move one target from a heavily-loaded UAV to a lighter one.
- **Different starting target**: re-run nearest-neighbor from a different first target.
- **Partition variation**: try a different initial grouping of targets.

{_validation}

## Guidelines
- Minimize total cost = C_UT (UAV→first target) + C_TT (target→target transitions).
- Use TopKTargets_per_UAV to identify low-cost initial targets for each UAV.
- Use TopKNextTargets_per_target (or C_TT_sparse.top_next) to find efficient transitions.
- Avoid long tours for a single UAV when targets are geographically spread out.
- If C_TT is provided as C_TT_sparse (top-K adjacency), use it to guide tour ordering.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
)

# 通用回退（包含所有场景规则，用于 model_type 未知时）
_POP_INIT_GENERIC = """\
You are an expert in UAV-target assignment optimization.

Task: Generate candidate assignment plans for a UAV scheduling problem. \
The solver will convert your assignments into an evolutionary algorithm's \
internal encoding, so you only need to produce **discrete assignments**.

{_json_format}
{{
    "thought": "<your construction strategy>",
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

### overloaded (N > M, UAVs outnumber targets)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.

### srp (N < M, UAVs visit multiple targets in sequence)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.

{_reasoning}
{_validation}

## Guidelines
- Focus on minimizing total cost while respecting constraints.
- Use the preference summary to identify low-cost assignments.
- For contested targets, assign to the best UAV first.
- For SRP, use nearest-neighbor ordering guided by C_TT transition costs.
""".format(
    _json_format=_JSON_FORMAT_COMMON,
    _reasoning=_REASONING_GUIDE,
    _validation=_OUTPUT_VALIDATION,
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
constraints for the '{model_type}' model type.

QUALITY PRIORITY:
- Use the cost matrix (C_UT, C_TT) and preference data to construct LOW-COST solutions.
- Think through your construction strategy in the "thought" field.
- Output ONLY the JSON object. No markdown fences, no preamble.
"""


# =============================================================================
# Search Controller — Scene-specific System Prompts
# =============================================================================

_SC_BASE = """\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: Select the crossover rate (CR) from {cr_choices}.

The scaling factor F will be automatically computed from your chosen CR \
F is automatically derived from your chosen CR — you do not need to specify F.

## How CR Affects Search
Each gene independently uses:
- DE/rand/1 (exploration) when random value <= CR
- DE/best/2 (exploitation) when random value > CR

## Closed-Loop Control
You are making sequential decisions. Each decision has an observable outcome:
- **Δf (delta fitness)**: percentage improvement since last decision. Positive = improving.
- **ΔD (delta diversity)**: diversity change since last decision. Positive = more diverse.

Use the Stage History table and Last Decision Feedback to:
1. See what CR you chose before and what happened.
2. Decide whether to continue, reverse, or try something new.
3. Balance exploration (high CR → more rand/1) vs exploitation (low CR → more best/2).
"""

_SC_CR_GUIDE = """\

The mechanism above means CR controls **how many genes** are replaced \
by the rand/1 or best/2 donor (i.e., a gene-level selection probability), \
not the exploration direction itself. Your CR choice should be based on the \
**current optimization state** (diversity, stagnation, convergence speed, \
trajectory trend), not a fixed rule for the scenario.
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
- The search space is a **permutation space** — diversity tends to drop quickly \
because swapping assignments between UAVs can lead to rapid convergence.
- Consider the trade-off between maintaining population diversity \
and converging toward low-cost permutations.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
"""

_SC_OVERLOADED_EXTRA = """\
## Scene: overloaded (N > M)
- Multiple UAVs share targets — the search space has **redundancy** \
(different UAV combinations can achieve similar coverage).
- Diversity is naturally higher due to target sharing, \
but this also means many solutions are near-equivalent.
- Consider whether the solver is exploring enough distinct assignment patterns \
or getting stuck in redundant regions.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
"""

_SC_SRP_EXTRA = """\
## Scene: srp (N < M)
- Each UAV visits multiple targets in sequence — **tour ordering** matters.
- The search space is larger (assignment + ordering combined), \
and the interaction between assignment and ordering creates complex fitness landscapes.
- Consider whether the solver needs to explore different assignment-to-ordering combinations \
or refine existing tour segments.
- Use the observed optimization state and recent trajectory \
to determine the appropriate CR.
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
                     非空时追加场景搜索特性描述（供 LLM 参考，非决策规则）
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

## Stage History (each row = one of your previous decisions)
{trajectory_text}

## Task
Select the best CR for the next interval. \
Base your decision on the current state, stage history, and last decision feedback. \
Respond with JSON only.
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

The `previous_decision_feedback` section shows the effect of your last CR decision.
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
    "POPULATION_INIT_USER_PROMPT",
    "CR_CONTROL_SYSTEM_PROMPT",
    "CR_CONTROL_USER_PROMPT",
    "SEARCH_CONTROLLER_USER_PROMPT",
    "OPERATOR_SELECTION_USER_PROMPT",
    "PROMPT_REGISTRY",
    "USER_PROMPT_REGISTRY",
]