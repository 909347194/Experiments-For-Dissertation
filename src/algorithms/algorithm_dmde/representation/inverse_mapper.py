# -*- coding: utf-8 -*-
"""inverse_mapper.py — 连续到离散的反映射协调器 φ'（公式 3-7）

修复记录：
    - 修复 SRP 反映射：正确处理 UAV->Target + Target->Target 基因结构。
    - 修复确定性问题：引入随机化最近邻匹配（top-k 采样），
      避免差分扰动被纯贪心最近邻吞掉。
    - 基因长度：balanced/overloaded = n_uavs, SRP = n_uavs + n_targets。
"""

from __future__ import annotations

import numpy as np

from .encoder import Gene, Individual
from .repair_rules.nearest_match import nearest_match, nearest_match_stochastic
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
    rng: np.random.Generator | None = None,
) -> Individual:
    """反映射：将连续代价值向量还原为离散个体。

    对应公式 (3-7) 和算法 3.1 的 17-26 行。

    Args:
        cost_vector:  差分后的连续代价值向量。
        cost_matrix:  原始代价矩阵。
        n_uavs:       UAV 数量。
        n_targets:    目标数量。
        model_type:   分配模型类型。
        rng:          随机数生成器。

    Returns:
        可行的子代个体。
    """
    if rng is None:
        rng = np.random.default_rng()

    if model_type == "srp":
        return _inverse_phi_srp(cost_vector, cost_matrix, n_uavs, n_targets, rng)
    else:
        return _inverse_phi_standard(cost_vector, cost_matrix, n_uavs, n_targets, model_type, rng)


def _inverse_phi_standard(
    cost_vector: np.ndarray,
    cost_matrix: np.ndarray,
    n_uavs: int,
    n_targets: int,
    model_type: str,
    rng: np.random.Generator,
) -> Individual:
    """标准反映射（balanced / overloaded）。"""
    cm_work = cost_matrix.copy().astype(float)
    mask = np.zeros_like(cm_work, dtype=bool)

    genes: list[Gene] = []
    invalid_indices: list[int] = []

    for idx, cv in enumerate(cost_vector):
        if not np.isfinite(cv) or cv < 0:
            invalid_indices.append(idx)
            continue

        # 随机化最近邻匹配（top-k 采样，避免纯贪心）
        result = nearest_match_stochastic(cv, cm_work, mask, top_k=3, rng=rng)
        if result is None:
            invalid_indices.append(idx)
            continue

        row, col, cost = result
        genes.append(Gene(uav_id=row, target_id=col, cost=cost))

        if model_type == "balanced":
            mask_balanced(mask, row, col)
        elif model_type == "overloaded":
            mask_overloaded(mask, row, col)

    # 规则 3.6: 无效值随机修补
    if invalid_indices:
        repaired = repair_invalid(
            cm_work, mask, invalid_indices, model_type, n_uavs
        )
        for uav_id, tgt_id, cost in repaired:
            genes.append(Gene(uav_id=uav_id, target_id=tgt_id, cost=cost))

    return Individual(genes=genes, model_type=model_type)


def _inverse_phi_srp(
    cost_vector: np.ndarray,
    cost_matrix: np.ndarray,
    n_uavs: int,
    n_targets: int,
    rng: np.random.Generator,
) -> Individual:
    """SRP 反映射（N<M 巡游模型）。

    SRP 个体结构：
        前 n_uavs 个基因: UAV -> first_target (从 cost_matrix 上半部分)
        后续基因: prev_target -> next_target (从 cost_matrix 下半部分)

    反映射流程：
        1. 从 cost_vector 前 n_uavs 个值匹配 UAV->Target。
        2. 对每个 UAV，从剩余 cost_vector 值匹配 Target->Target 巡游。
        3. 保证所有 target 恰好被分配一次。
    """
    cm_work = cost_matrix.copy().astype(float)
    mask = np.zeros_like(cm_work, dtype=bool)
    genes: list[Gene] = []

    # 第一阶段：每个 UAV 匹配一个初始目标（从上半部分矩阵）
    assigned_targets: set[int] = set()
    uav_first_target: dict[int, int] = {}

    for i in range(min(n_uavs, len(cost_vector))):
        cv = cost_vector[i]
        if not np.isfinite(cv) or cv < 0:
            # 随机选一个未分配的目标
            available = [t for t in range(n_targets) if t not in assigned_targets]
            if available:
                tgt = rng.choice(available)
            else:
                tgt = rng.integers(n_targets)
        else:
            # 在 UAV 行中找最近邻（仅看未分配的目标）
            row = i
            diff = np.abs(cm_work[row, :n_targets] - cv)
            for t in assigned_targets:
                diff[t] = np.inf
            tgt = int(np.argmin(diff))

        cost = float(cm_work[i, tgt])
        genes.append(Gene(uav_id=i, target_id=tgt, cost=cost))
        assigned_targets.add(tgt)
        uav_first_target[i] = tgt

        # 标记该目标列已使用（上半部分）
        mask[:, tgt] = True

    # 第二阶段：巡游匹配（从下半部分矩阵）
    # 对每个 UAV，用剩余 cost_vector 值匹配 target->target
    remaining_targets = [t for t in range(n_targets) if t not in assigned_targets]

    # 按最近邻将剩余目标分配给 UAV
    for tgt in remaining_targets:
        # 找到距离该目标最近的 UAV（基于当前 UAV 所在目标的巡游代价）
        best_uav = -1
        best_cost = np.inf
        for uav_id, current_tgt in uav_first_target.items():
            tc = cm_work[n_uavs + current_tgt, tgt]
            if tc < best_cost:
                best_cost = tc
                best_uav = uav_id

        if best_uav >= 0:
            prev_tgt = uav_first_target[best_uav]
            cost = float(cm_work[n_uavs + prev_tgt, tgt])
            genes.append(Gene(uav_id=-1, target_id=tgt, cost=cost))
            uav_first_target[best_uav] = tgt  # 更新该 UAV 当前位置

    return Individual(genes=genes, model_type="srp")
