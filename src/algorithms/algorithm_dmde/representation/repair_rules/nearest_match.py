# -*- coding: utf-8 -*-
"""nearest_match.py — 规则 3.4：最近邻空间欧氏距离匹配

对应论文：
    规则 3.4：对差分后临时代价值集合中的每个合理值，在航程代价矩阵中
    找到与其距离最近的值 C(i,j)，将其对应的序列关系对 (U,T,C(i,j))
    作为该位置上的新个体基因。

    公式 (3-8): C'(i,j) = C(i,j) | min{|x'(i)(t) - C(i,j)|}
"""

from __future__ import annotations

import numpy as np


def nearest_match(
    target_value: float,
    cost_matrix: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, float] | None:
    """在代价矩阵中找最近邻匹配。

    对应规则 3.4 和公式 (3-8)。

    Args:
        target_value: 差分后的临时代价值。
        cost_matrix:  代价矩阵（会被修改，用 Inf 标记已匹配位置）。
        mask:         布尔掩码，True 表示该位置已不可用。

    Returns:
        (row, col, cost) 三元组，无可用匹配时返回 None。
    """
    # 计算差分值与矩阵中每个可用位置的距离
    diff = np.abs(cost_matrix - target_value)
    diff[mask] = np.inf  # 排除已匹配位置

    # 找最近邻
    min_idx = np.unravel_index(np.argmin(diff), diff.shape)
    if diff[min_idx] == np.inf:
        return None

    row, col = min_idx
    cost = cost_matrix[row, col]
    return (row, col, cost)
