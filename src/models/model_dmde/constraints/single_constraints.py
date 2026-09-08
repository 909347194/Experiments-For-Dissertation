# -*- coding: utf-8 -*-
"""single_constraints.py — 单机飞行性能约束检测。

对应论文：
    公式 (2-5): 最小航迹段长度约束。
    公式 (2-6): 最大水平转角约束。
    公式 (2-7): 最大爬升/俯冲角约束。
    公式 (2-8): 航迹段与威胁区约束。
    公式 (2-15): 最大航程约束。
    公式 (2-16): 最大飞行时间约束。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..entities.uav import UAV


@dataclass(frozen=True)
class SingleConstraintViolation:
    """单机约束违背量。

    Attributes:
        range_violation:  最大航程约束违背量（公式 2-15）。
        time_violation:   最大飞行时间约束违背量（公式 2-16）。
    """

    range_violation: float = 0.0
    time_violation: float = 0.0

    @property
    def total(self) -> float:
        return self.range_violation + self.time_violation


def check_range_constraint(
    uav: UAV,
    assigned_distance: float,
) -> float:
    """检查最大航程约束（公式 2-15）。

    当 UAV 航程代价超过其最大航程时返回超出量。

    Args:
        uav: UAV 实体。
        assigned_distance: 分配给该 UAV 的总航程。

    Returns:
        违背量（非负值）。0 表示满足约束。
    """
    if assigned_distance > uav.max_range:
        return assigned_distance - uav.max_range
    return 0.0


def check_time_constraint(
    uav: UAV,
    assigned_distance: float,
) -> float:
    """检查最大飞行时间约束（公式 2-16）。

    当 UAV 以最低速度飞行仍超过最大飞行时间时返回超出量。

    Args:
        uav: UAV 实体。
        assigned_distance: 分配给该 UAV 的总航程。

    Returns:
        违背量（非负值）。0 表示满足约束。
    """
    if uav.max_time is None:
        return 0.0
    flight_time = uav.estimate_time(assigned_distance, use_min_speed=True)
    if flight_time > uav.max_time:
        return flight_time - uav.max_time
    return 0.0
