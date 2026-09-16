# -*- coding: utf-8 -*-
"""LLM 提示词统一管理模块

职责：
    集中管理所有 LLM 模块的提示词模板，支持：
    1. 默认提示词（Python 模块定义）
    2. 外部文件覆盖（实验配置目录）
    3. 场景化提示词（N=M/N>M/N<M）
    4. 动态参数插值（f-string 风格）

使用方式：
    from llm.prompts import get_prompt
    
    # 获取默认提示词
    prompt = get_prompt("population_init")
    
    # 获取带参数的提示词
    prompt = get_prompt("search_controller", cr_choices=[0.1, 0.3, 0.5])
    
    # 从外部文件加载
    prompt = get_prompt("population_init", prompt_path="/path/to/prompt.txt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Population Init Prompts
# =============================================================================

POPULATION_INIT_SYSTEM_PROMPT = """\
You are an expert in UAV-target assignment optimization.

Your task: Generate candidate assignment plans for a UAV scheduling problem. \
The solver will convert your assignments into an evolutionary algorithm's \
internal encoding, so you only need to produce **discrete assignments**.

## Output Format
Respond with a JSON object only (no markdown):
{
    "solutions": [
        {
            "assignments": [
                {"uav": <int>, "targets": [<int>, ...]},
                ...
            ]
        },
        ...
    ],
    "reasoning": "<brief explanation of your strategy>"
}

Each "solution" is a **complete assignment plan** covering ALL N UAVs.
Generate exactly the requested number of solutions (k).

## Assignment Rules by Model Type

### balanced (N == M, one-to-one)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once, each target appears exactly once.
- Each "targets" list has exactly 1 element.
- Example solution: {"assignments": [{"uav": 0, "targets": [2]}, {"uav": 1, "targets": [0]}, {"uav": 2, "targets": [1]}]}

### overloaded (N > M, UAVs outnumber targets)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target must appear at least once across all assignments.
- Each "targets" list has exactly 1 element.
- Example solution: {"assignments": [{"uav": 0, "targets": [1]}, {"uav": 1, "targets": [0]}, {"uav": 2, "targets": [1]}]}

### srp (N < M, UAVs visit multiple targets in sequence)
- Each solution has exactly N assignments (one per UAV).
- Each UAV appears exactly once.
- Each target appears exactly once across all assignments.
- "targets" list length >= 1, and the order represents the **tour sequence**.
- Example solution: {"assignments": [{"uav": 0, "targets": [3, 1, 4]}, {"uav": 1, "targets": [2, 0]}]}

## Guidelines
- Focus on minimizing total cost while respecting constraints.
- Use the preference summary to identify low-cost assignments.
- For SRP, order targets to minimize transition costs (nearest-neighbor heuristic).
- Generate exactly the requested number of complete solutions (k).
- Each solution must satisfy all constraints (every target covered, etc.).
"""


# =============================================================================
# CR Control Prompts
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


# =============================================================================
# Search Controller Prompts
# =============================================================================

def get_search_controller_prompt(cr_choices: list[float] | None = None) -> str:
    """获取搜索控制器提示词，支持动态 CR 候选值。"""
    if cr_choices is None:
        cr_choices = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    return f"""\
You are an expert in Differential Evolution (DE) for combinatorial optimization \
(UAV-target assignment with discrete mapping).

Your task: Select the crossover rate (CR) from {cr_choices}.

The scaling factor F will be automatically computed from your chosen CR \
using the DMDE parameter relationship (formula 3-11).

## How CR Affects Search
In the DMDE hybrid strategy (formula 3-10), each gene independently uses:
- DE/rand/1 (exploration) when random value <= CR
- DE/best/2 (exploitation) when random value > CR

Therefore:
- High CR (0.7/0.9): More genes use rand/1 → broader exploration
  Best when diversity is LOW or stagnation is HIGH.
- Low CR (0.1/0.3): More genes use best/2 → focused exploitation
  Best when diversity is HIGH but convergence is SLOW.
- Mid CR (0.5): Balanced exploration/exploitation.

## Decision Format
Respond with a JSON object only (no markdown):
{{
    "cr": <one of {cr_choices}>,
    "reasoning": "<brief explanation of your CR choice based on the current state>"
}}
"""


# =============================================================================
# Operator Selection Prompts
# =============================================================================

def get_operator_selection_prompt(strategies: list[str] | None = None) -> str:
    """获取算子选择提示词，支持动态策略列表。"""
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


# =============================================================================
# Unified Interface
# =============================================================================

PROMPT_REGISTRY = {
    "population_init": POPULATION_INIT_SYSTEM_PROMPT,
    "cr_control": CR_CONTROL_SYSTEM_PROMPT,
}


def get_prompt(
    module_name: str,
    prompt_path: str | Path | None = None,
    **kwargs: Any,
) -> str:
    """获取指定模块的提示词。
    
    Args:
        module_name: 模块名称 ("population_init", "cr_control", "search_controller", ...)
        prompt_path: 可选的外部文件路径，如果提供则从文件加载（覆盖默认）
        **kwargs: 动态参数，用于 f-string 插值
        
    Returns:
        提示词字符串
        
    Examples:
        >>> get_prompt("population_init")
        >>> get_prompt("search_controller", cr_choices=[0.1, 0.3, 0.5])
        >>> get_prompt("population_init", prompt_path="/path/to/prompt.txt")
    """
    # 优先从外部文件加载
    if prompt_path is not None:
        p = Path(prompt_path)
        if p.exists():
            logger.info(f"[prompts] Loading prompt from external file: {prompt_path}")
            return p.read_text(encoding="utf-8")
        else:
            logger.warning(f"[prompts] Prompt file not found: {prompt_path}, using default")
    
    # 从注册表获取默认提示词
    if module_name in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[module_name]
    
    # 特殊模块需要动态生成
    if module_name == "search_controller":
        return get_search_controller_prompt(**kwargs)
    
    if module_name == "operator_selection":
        return get_operator_selection_prompt(**kwargs)
    
    raise ValueError(f"Unknown prompt module: {module_name!r}. Available: {list(PROMPT_REGISTRY.keys()) + ['search_controller', 'operator_selection']}")


def load_prompt_from_file(prompt_path: str | Path, fallback: str | None = None) -> str:
    """从文件加载提示词，失败时返回 fallback。
    
    Args:
        prompt_path: 文件路径
        fallback: 可选的回退提示词
        
    Returns:
        提示词字符串
    """
    p = Path(prompt_path)
    if not p.exists():
        logger.warning(f"[prompts] File not found: {prompt_path}")
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    return p.read_text(encoding="utf-8")


__all__ = [
    "get_prompt",
    "load_prompt_from_file",
    "get_search_controller_prompt",
    "get_operator_selection_prompt",
    "POPULATION_INIT_SYSTEM_PROMPT",
    "CR_CONTROL_SYSTEM_PROMPT",
    "PROMPT_REGISTRY",
]
