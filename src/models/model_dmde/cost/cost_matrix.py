# -*- coding: utf-8 -*-
"""cost_matrix.py — 统一代价矩阵构建。

职责：
    根据 UAV 和目标的配置，结合环境层的航程代价估算器，
    构建三种模型的统一代价矩阵。

对应论文：
    第 2 章 2.3.2 节 —— MUAS 统一目标分配模型的描述。
    公式 (2-30): N=M 方阵。
    公式 (2-31): N>M 非方阵（多对一）。
    公式 (2-32): N<M SRP 矩阵（一对多 + 目标间代价）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..entities.uav import UAV
    from ..entities.target import Target
    from ...environments.environment_dmde.cost_estimator import (
        VerticalSectionCostEstimator,
        CostEstimationResult,
    )


@dataclass(frozen=True)
class CostMatrix:
    """统一代价矩阵。

    Attributes:
        matrix:      代价矩阵数组。
        uav_ids:     UAV 编号列表。
        target_ids:  目标编号列表。
        model_type:  分配模型类型 ('balanced' / 'overloaded' / 'srp')。
        details:     每个 (uav_idx, target_idx) 的估算详情。
    """

    matrix: np.ndarray
    uav_ids: list[int]
    target_ids: list[int]
    model_type: str
    details: dict[tuple[int, int], "CostEstimationResult"] | None = None


class CostMatrixBuilder:
    """统一代价矩阵构建器。

    根据 UAV 与目标的数量关系，自动选择对应的矩阵构建方式：
    - N=M: 方阵 (公式 2-30)
    - N>M: 非方阵 (公式 2-31)
    - N<M: SRP 矩阵 (公式 2-32)

    使用方式::

        from models.model_dmde.cost import CostMatrixBuilder
        builder = CostMatrixBuilder(estimator)
        cm = builder.build(uavs, targets)
    """

    def __init__(
        self,
        estimator: "VerticalSectionCostEstimator",
        store_details: bool = False,
    ) -> None:
        """
        Args:
            estimator: 环境层的航程代价估算器。
            store_details: 是否保存每个 (i,j) 对的详细估算结果。
        """
        self._estimator = estimator
        self._store_details = store_details

    def build(
        self,
        uavs: list["UAV"],
        targets: list["Target"],
    ) -> CostMatrix:
        """构建代价矩阵。

        根据 len(uavs) 与 len(targets) 的关系自动选择构建方式。

        Args:
            uavs:    UAV 列表。
            targets: 目标列表。

        Returns:
            CostMatrix 实例。
        """
        n, m = len(uavs), len(targets)
        if n == m:
            return self._build_balanced(uavs, targets)
        elif n > m:
            return self._build_overloaded(uavs, targets)
        else:
            return self._build_srp(uavs, targets)

    # ---- N=M 方阵 -----------------------------------------------------------

    def _build_balanced(
        self,
        uavs: list["UAV"],
        targets: list["Target"],
    ) -> CostMatrix:
        """构建 N=M 方时代价矩阵（公式 2-30）。"""
        n = len(uavs)
        matrix = np.zeros((n, n))
        details: dict[tuple[int, int], CostEstimationResult] = {}

        for i, uav in enumerate(uavs):
            for j, tgt in enumerate(targets):
                result = self._estimator.estimate(
                    start=uav.start_pos,
                    end=tgt.position,
                    weight=tgt.weight,
                )
                matrix[i, j] = result.cost
                if self._store_details:
                    details[(i, j)] = result

        return CostMatrix(
            matrix=matrix,
            uav_ids=[u.id for u in uavs],
            target_ids=[t.id for t in targets],
            model_type="balanced",
            details=details if self._store_details else None,
        )

    # ---- N>M 非方阵 ---------------------------------------------------------

    def _build_overloaded(
        self,
        uavs: list["UAV"],
        targets: list["Target"],
    ) -> CostMatrix:
        """构建 N>M 非方时代价矩阵（公式 2-31）。

        矩阵 shape = (n_uavs, n_targets)，多个 UAV 可分配给同一目标。
        """
        n, m = len(uavs), len(targets)
        matrix = np.zeros((n, m))
        details: dict[tuple[int, int], CostEstimationResult] = {}

        for i, uav in enumerate(uavs):
            for j, tgt in enumerate(targets):
                result = self._estimator.estimate(
                    start=uav.start_pos,
                    end=tgt.position,
                    weight=tgt.weight,
                )
                matrix[i, j] = result.cost
                if self._store_details:
                    details[(i, j)] = result

        return CostMatrix(
            matrix=matrix,
            uav_ids=[u.id for u in uavs],
            target_ids=[t.id for t in targets],
            model_type="overloaded",
            details=details if self._store_details else None,
        )

    # ---- N<M SRP 矩阵 -------------------------------------------------------

    def _build_srp(
        self,
        uavs: list["UAV"],
        targets: list["Target"],
    ) -> CostMatrix:
        """构建 N<M SRP 代价矩阵（公式 2-32）。

        矩阵 shape = (n_uavs + n_targets, n_targets)。
        上半部分: UAV → Target 代价。
        下半部分: Target → Target 代价（左对角线为 0）。
        """
        n, m = len(uavs), len(targets)
        matrix = np.zeros((n + m, m))
        details: dict[tuple[int, int], CostEstimationResult] = {}

        # 上半部分：UAV -> Target
        for i, uav in enumerate(uavs):
            for j, tgt in enumerate(targets):
                result = self._estimator.estimate(
                    start=uav.start_pos,
                    end=tgt.position,
                    weight=tgt.weight,
                )
                matrix[i, j] = result.cost
                if self._store_details:
                    details[(i, j)] = result

        # 下半部分：Target -> Target（对角线为 0）
        for i, tgt_i in enumerate(targets):
            for j, tgt_j in enumerate(targets):
                if i == j:
                    matrix[n + i, j] = 0.0
                else:
                    result = self._estimator.estimate(
                        start=tgt_i.position,
                        end=tgt_j.position,
                        weight=tgt_j.weight,
                    )
                    matrix[n + i, j] = result.cost
                    if self._store_details:
                        details[(n + i, j)] = result

        return CostMatrix(
            matrix=matrix,
            uav_ids=[u.id for u in uavs],
            target_ids=[t.id for t in targets],
            model_type="srp",
            details=details if self._store_details else None,
        )
