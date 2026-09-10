# -*- coding: utf-8 -*-
"""trajectory_collector.py — 优化轨迹收集器

职责：
    收集和管理优化过程中的轨迹数据，包括每代的搜索特征、
    LLM 决策和适应度值。为 LLM 提供历史上下文，帮助其
    做出更准确的算子策略决策。

轨迹数据结构：
    每条轨迹记录包含：
    - generation: 代数
    - features: 搜索状态特征字典
    - decision: LLM 决策（LLMDecision 实例或 None）
    - fitness: 当前最优适应度

使用方式::

    collector = TrajectoryCollector(max_history=10)
    collector.record(generation=50, features={...}, decision=decision, fitness=100.0)
    trajectory = collector.get_trajectory()
"""

from __future__ import annotations

from typing import Any


class TrajectoryCollector:
    """优化轨迹收集器。

    维护一个固定大小的滑动窗口，记录最近 max_history 代的
    搜索特征、LLM 决策和适应度值。

    Attributes:
        max_history: 最大历史记录数。
    """

    def __init__(self, max_history: int = 10) -> None:
        """初始化轨迹收集器。

        Args:
            max_history: 最大历史记录数。超过此数量时，最旧的记录被丢弃。
        """
        self.max_history = max_history
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        generation: int,
        features: dict[str, Any],
        decision: Any | None,
        fitness: float,
    ) -> None:
        """记录一条轨迹数据。

        Args:
            generation: 当前代数。
            features:   搜索状态特征字典。
            decision:   LLM 决策实例（可为 None）。
            fitness:    当前最优适应度值。
        """
        entry = {
            "generation": generation,
            "features": features,
            "decision": decision,
            "fitness": fitness,
        }

        self._records.append(entry)

        # 维持滑动窗口大小
        if len(self._records) > self.max_history:
            self._records = self._records[-self.max_history:]

    def get_trajectory(self) -> list[dict[str, Any]]:
        """获取当前轨迹数据。

        Returns:
            轨迹记录列表，按代数顺序排列。
        """
        return list(self._records)

    def get_recent(self, n: int | None = None) -> list[dict[str, Any]]:
        """获取最近 n 条轨迹记录。

        Args:
            n: 返回的记录数。None 表示返回全部。

        Returns:
            最近 n 条轨迹记录。
        """
        if n is None:
            return list(self._records)
        return list(self._records[-n:])

    def get_fitness_trend(self) -> list[float]:
        """获取适应度趋势。

        Returns:
            适应度值列表，按代数顺序排列。
        """
        return [entry["fitness"] for entry in self._records]

    def get_decision_history(self) -> list:
        """获取决策历史。

        Returns:
            LLM 决策列表（可能包含 None）。
        """
        return [entry["decision"] for entry in self._records]

    def clear(self) -> None:
        """清空所有轨迹数据。"""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"TrajectoryCollector("
            f"records={len(self._records)}, "
            f"max_history={self.max_history})"
        )
