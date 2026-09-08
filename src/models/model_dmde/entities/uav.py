# -*- coding: utf-8 -*-
"""uav.py — UAV 实体定义。

对应论文：
    表 2-1 中的 UAV 参数设定。
    公式 (2-5) ~ (2-8): 单机飞行性能约束条件。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UAV:
    """单架 UAV 的属性定义。

    Attributes:
        id:           UAV 编号。
        start_pos:    起飞位置 (x, y, z)。
        speed_range:  飞行速度范围 [v_min, v_max]（km/min）。
        max_range:    最大航程（km）。
        max_time:     最大飞行时间（min），可选。
    """

    id: int
    start_pos: tuple[float, float, float]
    speed_range: tuple[float, float] = (0.2, 0.5)
    max_range: float = 500.0
    max_time: float | None = None

    @property
    def avg_speed(self) -> float:
        """平均飞行速度。"""
        return (self.speed_range[0] + self.speed_range[1]) / 2.0

    def estimate_time(self, distance: float, use_min_speed: bool = False) -> float:
        """估算给定距离的飞行时间。

        Args:
            distance: 航程距离。
            use_min_speed: 是否使用最低速度（保守估计）。

        Returns:
            飞行时间。
        """
        speed = self.speed_range[0] if use_min_speed else self.avg_speed
        return distance / speed
