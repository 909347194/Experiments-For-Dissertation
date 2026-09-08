# -*- coding: utf-8 -*-
"""base_optimizer.py — 优化器接口规范。

定义所有 DMDE 求解器必须遵循的统一接口，确保 baseline_solvers
与主求解器输出口径一致，便于表 3-5 横向对比。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolverResult:
    """求解器统一输出结构。

    Attributes:
        best_assignment: 最优分配方案 [(uav_id, target_id), ...]。
        best_fitness:    最优适应度值。
        cost_history:    每代最优适应度历史（收敛曲线）。
        total_generations: 实际迭代代数。
        elapsed_seconds: 求解耗时（秒）。
        solver_name:     求解器名称。
        extra:           其他附加信息。
    """

    best_assignment: list[tuple[int, int]]
    best_fitness: float
    cost_history: list[float]
    total_generations: int
    elapsed_seconds: float
    solver_name: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """优化器抽象基类。

    所有 DMDE 求解器（主算法 + 对比算法）必须继承此类并实现 solve 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """求解器名称。"""
        ...

    @abstractmethod
    def solve(
        self,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        **kwargs,
    ) -> SolverResult:
        """求解目标分配问题。

        Args:
            cost_matrix: 代价矩阵。
            n_uavs:      UAV 数量。
            n_targets:   目标数量。
            **kwargs:    求解器特定参数。

        Returns:
            SolverResult 实例。
        """
        ...
