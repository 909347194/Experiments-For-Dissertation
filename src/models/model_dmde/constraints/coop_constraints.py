# -*- coding: utf-8 -*-
"""coop_constraints.py — 协同约束检测。

对应论文：
    公式 (2-9):  目标间时序约束。
    公式 (2-10): 多时窗约束。
    公式 (2-19) ~ (2-22): 同时到达约束。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..entities.uav import UAV
from ..entities.target import Target


@dataclass(frozen=True)
class CoopConstraintViolation:
    """协同约束违背量。

    Attributes:
        seq_violation:   时序约束违背量（公式 2-9）。
        window_violation: 时间窗约束违背量（公式 2-10）。
        sync_violation:  同时到达约束违背量（公式 2-19~22）。
    """

    seq_violation: float = 0.0
    window_violation: float = 0.0
    sync_violation: float = 0.0

    @property
    def total(self) -> float:
        return self.seq_violation + self.window_violation + self.sync_violation


def check_sequence_constraint(
    assignment: list[tuple[int, int]],
    targets: list[Target],
) -> float:
    """检查时序约束（公式 2-9）。

    检查具有时序关系的目标是否按正确顺序执行。

    Args:
        assignment: 分配方案 [(uav_id, target_id), ...]。
        targets: 目标列表。

    Returns:
        违背量。0 表示满足约束。
    """
    violation = 0.0

    # 按 UAV 分组，记录各 UAV 执行目标的顺序
    uav_targets: dict[int, list[int]] = {}
    for uav_id, target_id in assignment:
        uav_targets.setdefault(uav_id, []).append(target_id)

    # 构建目标 id -> sequence_group 映射
    target_map = {t.id: t for t in targets}

    # 检查同组内的时序关系
    for uav_id, tgt_list in uav_targets.items():
        for tgt_id in tgt_list:
            tgt = target_map.get(tgt_id)
            if tgt is None or tgt.sequence_group is None:
                continue
            # 找同组内应先于当前目标执行的其他目标
            for other_id in tgt_list:
                if other_id == tgt_id:
                    continue
                other = target_map.get(other_id)
                if other is None:
                    continue
                if other.sequence_group == tgt.sequence_group and other.id < tgt.id:
                    # other 应在 tgt 之前
                    if tgt_list.index(tgt_id) < tgt_list.index(other_id):
                        violation += 1.0

    return violation


def check_time_window_constraint(
    assignment: list[tuple[int, int]],
    uavs: list[UAV],
    targets: list[Target],
    cost_matrix: np.ndarray,
) -> float:
    """检查时间窗约束（公式 2-10）。

    检查目标是否在其规定的时间窗内被执行。

    Args:
        assignment: 分配方案。
        uavs: UAV 列表。
        targets: 目标列表。
        cost_matrix: 代价矩阵。

    Returns:
        违背量。0 表示满足约束。
    """
    violation = 0.0
    uav_map = {u.id: u for u in uavs}
    target_map = {t.id: t for t in targets}

    for uav_id, target_id in assignment:
        tgt = target_map.get(target_id)
        if tgt is None or tgt.time_window is None:
            continue

        uav = uav_map.get(uav_id)
        if uav is None:
            continue

        # 估算到达时间
        dist = cost_matrix[uav_id, target_id] / tgt.weight  # 还原距离
        arrival_time = uav.estimate_time(dist)

        t_start, t_end = tgt.time_window
        if arrival_time < t_start:
            violation += t_start - arrival_time
        elif arrival_time > t_end:
            violation += arrival_time - t_end

    return violation


def check_sync_constraint(
    assignment: list[tuple[int, int]],
    uavs: list[UAV],
    cost_matrix: np.ndarray,
) -> float:
    """检查同时到达约束（公式 2-19 ~ 2-22）。

    检查执行同一目标的多架 UAV 是否能同时到达。

    Args:
        assignment: 分配方案。
        uavs: UAV 列表。
        cost_matrix: 代价矩阵。

    Returns:
        违背量。0 表示满足约束。
    """
    violation = 0.0
    uav_map = {u.id: u for u in uavs}

    # 按目标分组
    target_uavs: dict[int, list[int]] = {}
    for uav_id, target_id in assignment:
        target_uavs.setdefault(target_id, []).append(uav_id)

    for target_id, uav_ids in target_uavs.items():
        if len(uav_ids) <= 1:
            continue

        # 计算各 UAV 的到达时间范围
        time_ranges: list[tuple[float, float]] = []
        for uav_id in uav_ids:
            uav = uav_map.get(uav_id)
            if uav is None or uav_id >= cost_matrix.shape[0]:
                continue
            dist = cost_matrix[uav_id, target_id]
            t_min = uav.estimate_time(dist, use_min_speed=False)  # 最快
            t_max = uav.estimate_time(dist, use_min_speed=True)   # 最慢
            time_ranges.append((t_min, t_max))

        if not time_ranges:
            continue

        # 检查时间窗口是否有交集
        latest_min = max(t[0] for t in time_ranges)
        earliest_max = min(t[1] for t in time_ranges)
        if latest_min > earliest_max:
            violation += latest_min - earliest_max

    return violation
