# -*- coding: utf-8 -*-
"""transition.py — 定义环境状态转移。

职责：
    本模块定义 DMDE 环境的动态状态转移机制，用于处理：
    1. 环境参数的动态变化（如雷达威胁区域变化、禁飞区出现等）。
    2. 任务执行过程中环境状态的更新。
    3. 支持航迹重规划的环境状态管理。

对应论文：
    第 5 章 —— 动态环境下多阶段多机协同航迹重规划方法。
    环境的实时变化是航迹重规划的触发条件。

注：
    在基础实验（exp_dmde_01）中，环境通常是静态的。
    状态转移机制主要为后续动态实验预留接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .context import EnvironmentContext
from .radar_threat import RadarThreat, RadarThreatField


# ---------------------------------------------------------------------------
# 环境事件
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvironmentEvent:
    """环境变化事件。

    Attributes:
        timestamp:  事件发生时间。
        event_type: 事件类型（'radar_add', 'radar_remove', 'nofly_add', 'target_destroyed'）。
        payload:    事件载荷（与事件类型相关的数据）。
    """

    timestamp: float
    event_type: str
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 状态转移接口
# ---------------------------------------------------------------------------

class TransitionPolicy(Protocol):
    """状态转移策略接口。

    实现此协议的类可以定义不同的环境动态变化行为。
    """

    def apply(
        self,
        context: EnvironmentContext,
        event: EnvironmentEvent,
    ) -> EnvironmentContext:
        """应用环境事件，返回更新后的上下文。

        Args:
            context: 当前环境上下文。
            event:   环境变化事件。

        Returns:
            更新后的环境上下文。
        """
        ...


# ---------------------------------------------------------------------------
# 默认状态转移管理器
# ---------------------------------------------------------------------------

class DMDETransitionManager:
    """DMDE 环境状态转移管理器。

    管理环境事件队列，按时间顺序应用事件，维护环境状态。

    使用方式::

        manager = DMDETransitionManager(context)
        manager.add_event(EnvironmentEvent(
            timestamp=10.0,
            event_type="radar_add",
            payload={"x0": 91.2, "y0": 29.6, "z0": 3650, "radius": 10000},
        ))
        updated_context = manager.advance_to(10.0)
    """

    def __init__(self, initial_context: EnvironmentContext) -> None:
        self._context = initial_context
        self._events: list[EnvironmentEvent] = []
        self._applied_count = 0

    @property
    def context(self) -> EnvironmentContext:
        """当前环境上下文。"""
        return self._context

    @property
    def pending_events(self) -> list[EnvironmentEvent]:
        """待处理的事件列表。"""
        return self._events[self._applied_count:]

    def add_event(self, event: EnvironmentEvent) -> None:
        """添加一个环境事件。"""
        self._events.append(event)
        # 按时间排序
        self._events.sort(key=lambda e: e.timestamp)

    def advance_to(self, timestamp: float) -> EnvironmentContext:
        """推进到指定时间，应用所有已到期的事件。

        Args:
            timestamp: 目标时间。

        Returns:
            更新后的环境上下文。
        """
        while self._applied_count < len(self._events):
            event = self._events[self._applied_count]
            if event.timestamp > timestamp:
                break
            self._apply_event(event)
            self._applied_count += 1

        return self._context

    def _apply_event(self, event: EnvironmentEvent) -> None:
        """应用单个环境事件。"""
        if event.event_type == "radar_add":
            radar = RadarThreat(
                x0=event.payload["x0"],
                y0=event.payload["y0"],
                z0=event.payload["z0"],
                radius=event.payload.get("radius", 15000),
                penalty=event.payload.get("penalty", 10.0),
            )
            self._context.radar_field.add(radar)

        elif event.event_type == "radar_remove":
            # 移除指定坐标的雷达（简化实现）
            target_pos = (event.payload.get("x0"), event.payload.get("y0"))
            self._context.radar_field._radars = [
                r for r in self._context.radar_field._radars
                if (r.x0, r.y0) != target_pos
            ]

        elif event.event_type == "target_destroyed":
            # 标记目标被摧毁（简化实现，实际应更新 target_configs）
            pass

        # 其他事件类型可在此扩展
