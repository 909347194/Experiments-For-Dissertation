# -*- coding: utf-8 -*-
"""evaluator.py — 综合适应度评估器。

职责：
    综合单机约束和协同约束，计算分配方案的总适应度值。

对应论文：
    公式 (2-14): 综合适应度函数。
    f(x) = Σ w_j * d(i,j) * x(i,j)
         + α * max(Σ t(i,j) * x(i,j))
         + β * Σ c_k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..entities.uav import UAV
from ..entities.target import Target
from .single_constraints import check_range_constraint, check_time_constraint
from .coop_constraints import (
    check_sequence_constraint,
    check_time_window_constraint,
    check_sync_constraint,
)


@dataclass
class FitnessResult:
    """综合适应度评估结果。

    Attributes:
        fitness:          综合适应度值（越小越好）。
        total_distance:   总航程代价。
        max_flight_time:  最大飞行时间。
        range_violation:  最大航程约束违背量。
        time_violation:   最大飞行时间约束违背量。
        seq_violation:    时序约束违背量。
        window_violation: 时间窗约束违背量。
        sync_violation:   同时到达约束违背量。
        is_feasible:      是否为可行解。
    """

    fitness: float
    total_distance: float
    max_flight_time: float
    range_violation: float = 0.0
    time_violation: float = 0.0
    seq_violation: float = 0.0
    window_violation: float = 0.0
    sync_violation: float = 0.0
    is_feasible: bool = True


class FitnessEvaluator:
    """综合适应度评估器。

    对应论文公式 (2-14)：
        f(x) = Σ w_j * d(i,j) * x(i,j)
             + α * max(Σ t(i,j) * x(i,j))
             + β * Σ c_k

    使用方式::

        evaluator = FitnessEvaluator(uavs, targets, alpha=1.0, beta=100.0)
        result = evaluator.evaluate(assignment, cost_matrix)
    """

    def __init__(
        self,
        uavs: list[UAV],
        targets: list[Target],
        alpha: float = 1.0,
        beta: float = 100.0,
    ) -> None:
        self._uavs = uavs
        self._targets = targets
        self._alpha = alpha  # 时间代价缩放因子
        self._beta = beta    # 约束违背惩罚缩放因子
        self._uav_map = {u.id: u for u in uavs}
        self._target_map = {t.id: t for t in targets}

    def evaluate(
        self,
        assignment: list[tuple[int, int]],
        cost_matrix: np.ndarray,
    ) -> FitnessResult:
        """评估分配方案的综合适应度。

        Args:
            assignment: 分配方案 [(uav_id, target_id), ...]。
            cost_matrix: 代价矩阵。

        Returns:
            FitnessResult 实例。
        """
        # ---- 总航程代价 ----
        total_distance = 0.0
        uav_distances: dict[int, float] = {}
        for uav_id, target_id in assignment:
            if uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                dist = cost_matrix[uav_id, target_id]
                total_distance += dist
                uav_distances[uav_id] = uav_distances.get(uav_id, 0.0) + dist

        # ---- 最大飞行时间 ----
        max_flight_time = 0.0
        for uav_id, total_dist in uav_distances.items():
            uav = self._uav_map.get(uav_id)
            if uav is not None:
                ft = uav.estimate_time(total_dist)
                max_flight_time = max(max_flight_time, ft)

        # ---- 单机约束违背 ----
        range_violation = 0.0
        time_violation = 0.0
        for uav_id, total_dist in uav_distances.items():
            uav = self._uav_map.get(uav_id)
            if uav is not None:
                range_violation += check_range_constraint(uav, total_dist)
                time_violation += check_time_constraint(uav, total_dist)

        # ---- 协同约束违背 ----
        seq_violation = check_sequence_constraint(assignment, self._targets)
        window_violation = check_time_window_constraint(
            assignment, self._uavs, self._targets, cost_matrix
        )
        sync_violation = check_sync_constraint(
            assignment, self._uavs, cost_matrix
        )

        # ---- 综合适应度（公式 2-14）----
        penalty = (
            range_violation
            + time_violation
            + seq_violation
            + window_violation
            + sync_violation
        )

        fitness = (
            total_distance
            + self._alpha * max_flight_time
            + self._beta * penalty
        )

        is_feasible = penalty == 0.0

        return FitnessResult(
            fitness=fitness,
            total_distance=total_distance,
            max_flight_time=max_flight_time,
            range_violation=range_violation,
            time_violation=time_violation,
            seq_violation=seq_violation,
            window_violation=window_violation,
            sync_violation=sync_violation,
            is_feasible=is_feasible,
        )
