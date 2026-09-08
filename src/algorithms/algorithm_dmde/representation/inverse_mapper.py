# -*- coding: utf-8 -*-
"""inverse_mapper.py — 连续到离散的反映射协调器 φ'（公式 3-7）

职责：
    将差分进化算子产生的连续临时代价值向量，通过匹配规则
    反映射回离散的三元组基因空间，生成可行的子代个体。

对应论文：
    公式 (3-7): φ': R^c --Match_rules(1,2,3)--> {(U_i, T_j) ∈ Z^d}
    规则 3.4: 最近邻匹配。
    规则 3.5: 唯一匹配（掩码 Inf 行/列剔除）。
    规则 3.6: 无效值随机变异修补。

算法流程（对应图 3-7）：
    1. 复制代价矩阵作为工作矩阵。
    2. 遍历差分临时代价值向量的每个分量。
    3. 对每个合理值，按规则 3.4 在工作矩阵中找最近邻。
    4. 按规则 3.5 更新掩码（Inf 行/列剔除）。
    5. 收集有效基因；将无效值存入待匹配数组。
    6. 按规则 3.6 对无效值执行随机变异修补。
    7. 返回可行的子代个体。
"""

from __future__ import annotations

import numpy as np

from .encoder import Gene, Individual
from .repair_rules.nearest_match import nearest_match
from .repair_rules.unique_filter import (
    mask_balanced,
    mask_overloaded,
    mask_srp_upper,
    INF,
)
from .repair_rules.invalid_mutator import repair_invalid


def inverse_phi(
    cost_vector: np.ndarray,
    cost_matrix: np.ndarray,
    n_uavs: int,
    n_targets: int,
    model_type: str,
) -> Individual:
    """反映射：将连续代价值向量还原为离散个体。

    对应公式 (3-7) 和算法 3.1 的 17-26 行。

    Args:
        cost_vector:  差分后的连续代价值向量。
        cost_matrix:  原始代价矩阵。
        n_uavs:       UAV 数量。
        n_targets:    目标数量。
        model_type:   分配模型类型。

    Returns:
        可行的子代个体。
    """
    # 工作矩阵和掩码
    cm_work = cost_matrix.copy().astype(float)
    mask = np.zeros_like(cm_work, dtype=bool)

    genes: list[Gene] = []
    invalid_indices: list[int] = []

    # 遍历每个差分临时代价值
    for idx, cv in enumerate(cost_vector):
        # 检查值是否合理（非 NaN/Inf/负值）
        if not np.isfinite(cv) or cv < 0:
            invalid_indices.append(idx)
            continue

        # 规则 3.4: 最近邻匹配
        result = nearest_match(cv, cm_work, mask)
        if result is None:
            invalid_indices.append(idx)
            continue

        row, col, cost = result
        genes.append(Gene(uav_id=row, target_id=col, cost=cost))

        # 规则 3.5: 按模型类型更新掩码
        if model_type == "balanced":
            mask_balanced(mask, row, col)
        elif model_type == "overloaded":
            mask_overloaded(mask, row, col)
        elif model_type == "srp":
            mask_srp_upper(cm_work, mask, row, col, n_uavs)

    # 规则 3.6: 对无效值执行随机变异修补
    if invalid_indices:
        repaired = repair_invalid(
            cm_work, mask, invalid_indices, model_type, n_uavs
        )
        for uav_id, tgt_id, cost in repaired:
            genes.append(Gene(uav_id=uav_id, target_id=tgt_id, cost=cost))

    return Individual(genes=genes, model_type=model_type)
