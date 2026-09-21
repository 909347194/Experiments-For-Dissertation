# -*- coding: utf-8 -*-
"""constraint_features.py — 约束满足特征提取

职责：
    从种群中提取约束满足相关特征，用于 LLM 决策。
    包括可行解比例和违反程度分布。

特征说明：
    - compute_feasible_ratio: 种群中可行解（无约束违反）的比例。
      值域 [0, 1]，1 = 全部可行，0 = 全部不可行。
    - compute_violation_distribution: 约束违反程度的统计分布。
      包括均值、最大值、标准差等，帮助 LLM 了解违反严重程度。
"""

from __future__ import annotations

import numpy as np


def compute_feasible_ratio(
    population: list,
    feasibility_threshold: float | None = None,
) -> float:
    """计算种群中可行解的比例。

    可行解判定逻辑：
    1. 如果显式传入 feasibility_threshold，用该阈值判断。
    2. 否则，自动推断阈值：取种群中最小 fitness 的 1.05 倍作为边界，
       或 1e-6（全零/接近零可行时的保底）。
    3. fitness 必须有限（inf/nan 视为不可行）。

    原始版本使用固定的 1e10 阈值，对约束违反量级在 10^5 ~ 10^6 的
    问题（如 SRP 场景）会把所有个体误判为可行。

    Args:
        population: 种群个体列表。每个个体需有 fitness 属性。
        feasibility_threshold: 显式阈值（None = 自动推断）。

    Returns:
        可行解比例，范围 [0.0, 1.0]。
    """
    if not population:
        return 0.0

    fitness_values = [ind.fitness for ind in population]
    finite_values = [f for f in fitness_values if np.isfinite(f)]

    if not finite_values:
        return 0.0

    if feasibility_threshold is not None:
        threshold = feasibility_threshold
    else:
        # 自动推断：取最小 fitness 的 1.05 倍（允许 5% 容差）
        # 全零 fitness 时代价为 0，用 1e-6 保底
        min_fit = min(finite_values)
        threshold = max(abs(min_fit) * 1.05, 1e-6)

    feasible_count = sum(
        1 for f in fitness_values
        if np.isfinite(f) and f <= threshold
    )
    return feasible_count / len(population)


def compute_violation_distribution(population: list) -> dict[str, float]:
    """计算种群约束违反程度的统计分布。

    对于有 fitness 属性的个体，统计适应度的分布特征。
    使用适应度值作为违反程度的代理指标（适应度越高，违反越严重）。

    Args:
        population: 种群个体列表。

    Returns:
        包含以下键的字典：
        - mean: 平均适应度
        - std: 适应度标准差
        - min: 最小适应度
        - max: 最大适应度
        - median: 中位数适应度
        - feasible_ratio: 可行解比例
    """
    if not population:
        return {
            "mean": float("inf"),
            "std": 0.0,
            "min": float("inf"),
            "max": float("inf"),
            "median": float("inf"),
            "feasible_ratio": 0.0,
        }

    fitness_values = np.array([ind.fitness for ind in population])
    finite_mask = np.isfinite(fitness_values)
    finite_values = fitness_values[finite_mask]

    if len(finite_values) == 0:
        return {
            "mean": float("inf"),
            "std": 0.0,
            "min": float("inf"),
            "max": float("inf"),
            "median": float("inf"),
            "feasible_ratio": 0.0,
        }

    # 可行解判定：用与 compute_feasible_ratio 相同的自动阈值
    min_fit = float(np.min(finite_values))
    threshold = max(abs(min_fit) * 1.05, 1e-6)
    feasible_mask = finite_mask & (fitness_values <= threshold)

    if not np.any(feasible_mask):
        return {
            "mean": float(np.mean(fitness_values)),
            "std": float(np.std(fitness_values)),
            "min": float(np.min(fitness_values)),
            "max": float(np.max(fitness_values)),
            "median": float(np.median(fitness_values)),
            "feasible_ratio": 0.0,
        }

    feasible_values = fitness_values[feasible_mask]

    return {
        "mean": float(np.mean(feasible_values)),
        "std": float(np.std(feasible_values)),
        "min": float(np.min(feasible_values)),
        "max": float(np.max(feasible_values)),
        "median": float(np.median(feasible_values)),
        "feasible_ratio": float(np.mean(feasible_mask)),
    }


def compute_constraint_tightness(population: list) -> float:
    """计算约束紧度指标。

    约束紧度 = 可行解比例的补数。
    值越大说明约束越紧，搜索空间中可行区域越小。

    Args:
        population: 种群个体列表。

    Returns:
        约束紧度，范围 [0.0, 1.0]。
        0 = 所有解都可行，1 = 没有可行解。
    """
    return 1.0 - compute_feasible_ratio(population)
