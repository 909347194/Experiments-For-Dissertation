# -*- coding: utf-8 -*-
"""nearest_match.py — 规则 3.4：最近邻空间欧氏距离匹配

修复记录：
    - 新增 nearest_match_stochastic: top-k 采样匹配，
      避免纯贪心导致差分扰动被吞掉。
"""

from __future__ import annotations

import numpy as np


def nearest_match(
    target_value: float,
    cost_matrix: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, float] | None:
    """纯贪心最近邻匹配（原始规则 3.4）。"""
    diff = np.abs(cost_matrix - target_value)
    diff[mask] = np.inf

    min_idx = np.unravel_index(np.argmin(diff), diff.shape)
    if diff[min_idx] == np.inf:
        return None

    row, col = min_idx
    cost = cost_matrix[row, col]
    return (int(row), int(col), float(cost))


def nearest_match_stochastic(
    target_value: float,
    cost_matrix: np.ndarray,
    mask: np.ndarray,
    top_k: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, float] | None:
    """随机化最近邻匹配（top-k 采样）。

    从距离最近的 top_k 个候选中随机选一个，
    引入多样性，避免纯贪心导致种群过快收敛。

    Args:
        target_value: 差分后的临时代价值。
        cost_matrix:  代价矩阵。
        mask:         布尔掩码。
        top_k:        候选数量。
        rng:          随机数生成器。

    Returns:
        (row, col, cost) 三元组，无可用匹配时返回 None。
    """
    if rng is None:
        rng = np.random.default_rng()

    diff = np.abs(cost_matrix - target_value)
    diff[mask] = np.inf

    # 找到所有可用位置
    available = np.argwhere(~mask)
    if len(available) == 0:
        return None

    # 按距离排序，取 top-k
    distances = diff[~mask]
    sorted_indices = np.argsort(distances)
    k = min(top_k, len(sorted_indices))
    chosen_idx = rng.choice(sorted_indices[:k])

    row, col = available[chosen_idx]
    cost = cost_matrix[row, col]
    return (int(row), int(col), float(cost))
