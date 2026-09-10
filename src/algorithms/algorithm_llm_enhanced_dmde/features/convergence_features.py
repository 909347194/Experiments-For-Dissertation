# -*- coding: utf-8 -*-
"""convergence_features.py — 收敛特征提取

职责：
    从优化收敛曲线中提取收敛相关特征，用于 LLM 决策。
    包括收敛速度和停滞检测。

特征说明：
    - compute_convergence_speed: 最近 window 代内适应度的平均改进率。
      正值 = 正在收敛，0 = 无改进，负值 = 退化。
    - detect_stagnation: 连续多少代适应度改进低于阈值。
      值越大说明停滞越严重，可能需要触发灭绝或改变策略。
"""

from __future__ import annotations

import numpy as np


def compute_convergence_speed(
    cost_history: list[float],
    window: int = 10,
) -> float:
    """计算最近 window 代的收敛速度。

    收敛速度定义为：最近 window 代内，每代的平均适应度改进量。
    改进量 = (前一代适应度 - 当前代适应度) / 前一代适应度（归一化）。

    Args:
        cost_history: 适应度历史列表，按代数顺序排列。
        window:       计算窗口大小（代数）。

    Returns:
        平均收敛速度。
        正值 = 正在收敛，0 = 无改进，负值 = 退化。
    """
    if len(cost_history) < 2:
        return 0.0

    # 取最近 window 代
    recent = cost_history[-min(window + 1, len(cost_history)):]

    if len(recent) < 2:
        return 0.0

    # 计算每代的归一化改进率
    improvements = []
    for i in range(1, len(recent)):
        prev = recent[i - 1]
        curr = recent[i]
        if abs(prev) > 1e-10:
            improvement = (prev - curr) / abs(prev)
        else:
            improvement = prev - curr
        improvements.append(improvement)

    return float(np.mean(improvements))


def detect_stagnation(
    cost_history: list[float],
    threshold: float = 1e-4,
    patience: int = 10,
) -> int:
    """检测停滞代数。

    从最新一代向前回溯，统计连续多少代适应度改进低于阈值。

    Args:
        cost_history: 适应度历史列表。
        threshold:    改进阈值（相对改进量）。
        patience:     最大回溯代数。

    Returns:
        连续停滞代数。0 = 没有停滞。
    """
    if len(cost_history) < 2:
        return 0

    stagnation_count = 0
    recent = cost_history[-min(patience + 1, len(cost_history)):]

    for i in range(len(recent) - 1, 0, -1):
        prev = recent[i - 1]
        curr = recent[i]

        # 计算相对改进
        if abs(prev) > 1e-10:
            relative_improvement = abs(prev - curr) / abs(prev)
        else:
            relative_improvement = abs(prev - curr)

        if relative_improvement < threshold:
            stagnation_count += 1
        else:
            break  # 一旦有显著改进就停止

    return stagnation_count


def compute_convergence_rate(cost_history: list[float]) -> float:
    """计算整体收敛速率。

    使用对数线性拟合估计收敛速率常数。
    适用于评估算法的收敛效率。

    Args:
        cost_history: 适应度历史列表。

    Returns:
        收敛速率（正值 = 收敛，越大越快）。
    """
    if len(cost_history) < 3:
        return 0.0

    # 过滤掉 inf 和 nan
    valid = [f for f in cost_history if np.isfinite(f)]
    if len(valid) < 3:
        return 0.0

    # 使用首尾估计
    initial = valid[0]
    final = valid[-1]

    if initial <= 0 or final <= 0 or initial <= final:
        # 无法取对数或未收敛
        return 0.0

    n = len(valid)
    return float(np.log(initial / final) / n)
