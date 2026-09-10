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

    @property
    def total_violation(self) -> float:
        """全部约束违背量之和。

        本属性是违约统计的**唯一口径**：``is_feasible`` 恒等价于
        ``total_violation == 0.0``，因此不会出现“不可行但违背量为 0”
        或“违背量非 0 却判为可行”的矛盾。

        注意：各分量量纲不同（航程=米、时间=秒、时序=违反序对个数），
        求和后仅用于惩罚项与“违背量/代价”比率，不可按物理量单独解读。
        """
        return (
            self.range_violation
            + self.time_violation
            + self.seq_violation
            + self.window_violation
            + self.sync_violation
        )

    def violation_breakdown(self) -> dict[str, float]:
        """按约束类型分解违背量，用于定位不可行的具体原因。"""
        return {
            "range": self.range_violation,
            "time": self.time_violation,
            "seq": self.seq_violation,
            "window": self.window_violation,
            "sync": self.sync_violation,
        }


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
        enable_seq: bool = True,
        enable_window: bool = True,
        enable_sync: bool = True,
    ) -> None:
        self._uavs = uavs
        self._targets = targets
        self._alpha = alpha  # 时间代价缩放因子
        self._beta = beta    # 约束违背惩罚缩放因子
        self._uav_map = {u.id: u for u in uavs}
        self._target_map = {t.id: t for t in targets}
        self._enable_seq = enable_seq
        self._enable_window = enable_window
        self._enable_sync = enable_sync

    def evaluate(
        self,
        assignment: list[tuple[int, int]],
        cost_matrix: np.ndarray,
        n_uavs: int | None = None,
    ) -> FitnessResult:
        """评估分配方案的综合适应度。

        Args:
            assignment: 分配方案 [(uav_id, target_id), ...]。
                        对于 SRP，包含巡游基因（uav_id 可能重复）。
            cost_matrix: 代价矩阵。
            n_uavs: UAV 数量（SRP 模型必须，用于区分矩阵上下半部分）。

        Returns:
            FitnessResult 实例。
        """
        # 判断是否 SRP 模型（有重复 uav_id 说明是 SRP 巡游）
        uav_ids_in_assignment = [a[0] for a in assignment]
        is_srp = len(uav_ids_in_assignment) > len(set(uav_ids_in_assignment))

        # ---- 总航程代价 ----
        total_distance = 0.0
        uav_distances: dict[int, float] = {}
        if is_srp and n_uavs is not None:
            # SRP 模型：按 UAV 分组计算巡游总代价
            routes: dict[int, list[int]] = {}
            for uav_id, target_id in assignment:
                routes.setdefault(uav_id, []).append(target_id)
            for uav_id, tgt_list in routes.items():
                for seq, tgt_id in enumerate(tgt_list):
                    if seq == 0:
                        dist = cost_matrix[uav_id, tgt_id]
                    else:
                        prev_tgt = tgt_list[seq - 1]
                        dist = cost_matrix[n_uavs + prev_tgt, tgt_id]
                    total_distance += dist
                    uav_distances[uav_id] = uav_distances.get(uav_id, 0.0) + dist
        else:
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
        seq_violation = check_sequence_constraint(assignment, self._targets) if self._enable_seq else 0.0
        window_violation = check_time_window_constraint(
            assignment, self._uavs, self._targets, cost_matrix
        ) if self._enable_window else 0.0
        sync_violation = check_sync_constraint(
            assignment, self._uavs, cost_matrix
        ) if self._enable_sync else 0.0

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

        result = FitnessResult(
            fitness=fitness,
            total_distance=total_distance,
            max_flight_time=max_flight_time,
            range_violation=range_violation,
            time_violation=time_violation,
            seq_violation=seq_violation,
            window_violation=window_violation,
            sync_violation=sync_violation,
        )
        # 以 total_violation 作为唯一判据，保证与统计口径强一致
        result.is_feasible = result.total_violation == 0.0
        return result
