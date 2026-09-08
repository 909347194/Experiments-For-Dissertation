# -*- coding: utf-8 -*-
"""target.py — 目标实体定义。

对应论文：
    表 2-1 中的目标参数设定。
    公式 (2-9) ~ (2-10): 时序约束和时间窗约束。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """目标点的属性定义。

    Attributes:
        id:       目标编号。
        position: 目标位置 (x, y, z)。
        weight:   目标权重 W（0 < W ≤ 1），权重越高优先级越高。
        time_window: 时间窗 [T_start, T_end]，可选。
        sequence_group: 时序组标识，同组内按 id 顺序执行。
    """

    id: int
    position: tuple[float, float, float]
    weight: float = 1.0
    time_window: tuple[float, float] | None = None
    sequence_group: int | None = None
