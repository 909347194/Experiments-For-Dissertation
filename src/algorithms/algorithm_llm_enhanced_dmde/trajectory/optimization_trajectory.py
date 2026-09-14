# -*- coding: utf-8 -*-
"""optimization_trajectory.py — 优化轨迹记录器

职责：
    记录每次 LLM decision 所需的完整上下文信息，
    支持序列化（JSON）用于实验复现和消融分析。

记录内容（每个 LLM 决策点）：
    - generation: 触发代数
    - strategy: 当前使用的 DE 算子策略
    - cr / F: 当前交叉率和缩放因子
    - fitness_best / fitness_mean: 适应度统计
    - diversity: 种群多样性
    - gene_variance: 基因方差
    - feasible_ratio: 可行解比例
    - convergence_speed: 收敛速度
    - stagnation_count: 停滞代数
    - violation_distribution: 约束违约分布
    - hv / igd: 超体积 / 反世代距离（如果提供参考集）
    - llm_module: 触发决策的 LLM 模块名称
    - llm_decision: LLM 返回的原始决策
    - llm_reasoning: LLM 的决策理由
    - elapsed_since_last: 距上次 LLM 调用的耗时
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrajectoryEntry:
    """单个轨迹记录点。"""
    generation: int
    # DE 状态
    strategy: str = "default"
    cr: float = 0.0
    f_scale: float = 0.0
    temperature: float = 0.0
    # 适应度
    fitness_best: float = float("inf")
    fitness_mean: float = float("inf")
    fitness_worst: float = float("inf")
    # 搜索状态
    diversity: float = 0.0
    gene_variance: float = 0.0
    convergence_speed: float = 0.0
    stagnation_count: int = 0
    feasible_ratio: float = 0.0
    violation_mean: float = 0.0
    violation_max: float = 0.0
    # 质量指标（可选，需要参考集）
    hv: float | None = None      # Hypervolume
    igd: float | None = None     # Inverted Generational Distance
    # LLM 决策
    llm_module: str = ""
    llm_decision: dict[str, Any] = field(default_factory=dict)
    llm_reasoning: str = ""
    llm_raw_output: str = ""       # LLM 原始输出文本
    llm_input: dict[str, Any] = field(default_factory=dict)  # 完整 LLM 输入 (messages, model, ...)
    llm_call_duration: float = 0.0  # seconds
    # 时间
    timestamp: float = field(default_factory=time.time)
    elapsed_since_last: float = 0.0


class OptimizationTrajectory:
    """优化轨迹管理器。

    记录完整的优化过程，支持：
    1. 按代数追加记录
    2. 查询历史轨迹（给 LLM 提供上下文）
    3. 序列化为 JSON（实验持久化）
    4. 从 JSON 还原

    使用方式::

        trajectory = OptimizationTrajectory()
        trajectory.record(TrajectoryEntry(
            generation=50, strategy="rand/1", cr=0.85,
            fitness_best=596000, diversity=0.35, ...
        ))
        # 给 LLM 用
        recent = trajectory.get_recent(n=10)
        # 保存
        trajectory.save_json("results/trajectory.json")
    """

    def __init__(self) -> None:
        self._entries: list[TrajectoryEntry] = []
        self._last_llm_time: float = 0.0

    def record(self, entry: TrajectoryEntry) -> None:
        """记录一条轨迹。"""
        if self._entries:
            entry.elapsed_since_last = entry.timestamp - self._entries[-1].timestamp
        self._entries.append(entry)

    def record_llm_decision(
        self,
        generation: int,
        llm_module: str,
        llm_decision: dict[str, Any],
        llm_reasoning: str,
        llm_call_duration: float,
        **kwargs,
    ) -> None:
        """便捷方法：记录一次 LLM 决策点。"""
        now = time.time()
        entry = TrajectoryEntry(
            generation=generation,
            llm_module=llm_module,
            llm_decision=llm_decision,
            llm_reasoning=llm_reasoning,
            llm_call_duration=llm_call_duration,
            timestamp=now,
            elapsed_since_last=now - self._last_llm_time if self._last_llm_time else 0.0,
            **kwargs,
        )
        self._last_llm_time = now
        self.record(entry)

    def get_recent(self, n: int = 10) -> list[TrajectoryEntry]:
        """获取最近 n 条记录。"""
        return self._entries[-n:]

    def get_all(self) -> list[TrajectoryEntry]:
        """获取全部记录。"""
        return list(self._entries)

    def get_fitness_trend(self) -> list[float]:
        """获取适应度趋势。"""
        return [e.fitness_best for e in self._entries if np.isfinite(e.fitness_best)]

    def get_diversity_trend(self) -> list[float]:
        """获取多样性趋势。"""
        return [e.diversity for e in self._entries]

    def get_llm_decisions(self) -> list[dict[str, Any]]:
        """获取所有 LLM 决策记录（含完整输入输出）。"""
        return [
            {
                "generation": e.generation,
                "module": e.llm_module,
                "decision": e.llm_decision,
                "reasoning": e.llm_reasoning,
                "duration": e.llm_call_duration,
                "llm_input": e.llm_input,
                "llm_raw_output": e.llm_raw_output,
            }
            for e in self._entries
            if e.llm_module
        ]

    def to_dicts(self) -> list[dict[str, Any]]:
        """序列化为 dict 列表。"""
        result = []
        for e in self._entries:
            d = asdict(e)
            # 处理 numpy 类型
            for k, v in d.items():
                if isinstance(v, (np.floating, np.integer)):
                    d[k] = float(v)
                elif isinstance(v, np.ndarray):
                    d[k] = v.tolist()
            result.append(d)
        return result

    def save_json(self, path: str | Path) -> Path:
        """保存为 JSON 文件。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dicts(), f, ensure_ascii=False, indent=2, default=str)
        return out

    @classmethod
    def load_json(cls, path: str | Path) -> "OptimizationTrajectory":
        """从 JSON 文件还原。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        traj = cls()
        for d in data:
            traj.record(TrajectoryEntry(**d))
        return traj

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"OptimizationTrajectory(entries={len(self._entries)})"
