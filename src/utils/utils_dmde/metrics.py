# 统计指标计算（平均代价、违约率等）

# -*- coding: utf-8 -*-
"""metrics.py — 实验统计指标计算

职责：
    对多次运行的实验结果进行统计汇总，计算论文表 3-3 中的
    各项指标，用于横向对比不同算法的性能。

对应论文：
    表 3-3: 实验结果统计数据。
    - 最优代价: 多组实验中最优的总航程代价。
    - 平均代价: 多组实验的平均适应度。
    - 标准差:   适应度的标准差。
    - 约束违背: 最优解中约束违背量总和占最优代价的百分比。
    - 优解率:   获得可行解的实验次数占总次数的百分比。
    - 平均耗时: 每次运行的平均耗时。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import SolverResult


@dataclass(frozen=True)
class ExperimentMetrics:
    """实验统计指标。

    Attributes:
        best_fitness:    最优适应度。
        mean_fitness:    平均适应度。
        std_fitness:     适应度标准差。
        median_fitness:  适应度中位数。
        worst_fitness:   最差适应度。
        feasible_rate:   可行解比例 (0~1)。
        violation_pct:   平均约束违背百分比。
        mean_time:       平均耗时(秒)。
        n_runs:          运行次数。
        best_run_idx:    最优运行的索引。
    """

    best_fitness: float
    mean_fitness: float
    std_fitness: float
    median_fitness: float
    worst_fitness: float
    feasible_rate: float
    violation_pct: float
    mean_time: float
    n_runs: int
    best_run_idx: int


def compute_metrics(results: list[SolverResult]) -> ExperimentMetrics:
    """从多次运行结果中计算统计指标。

    Args:
        results: 多次运行的 SolverResult 列表。

    Returns:
        ExperimentMetrics 实例。
    """
    if not results:
        raise ValueError("results is empty")

    fitnesses = np.array([r.best_fitness for r in results])
    times = np.array([r.elapsed_seconds for r in results])

    best_idx = int(np.argmin(fitnesses))

    # 约束违背百分比：从 extra 中获取，若无则为 0
    violation_pcts = []
    for r in results:
        vio = r.extra.get("total_violation", 0.0)
        if r.best_fitness > 0:
            violation_pcts.append(vio / r.best_fitness * 100)
        else:
            violation_pcts.append(0.0)

    return ExperimentMetrics(
        best_fitness=float(fitnesses.min()),
        mean_fitness=float(fitnesses.mean()),
        std_fitness=float(fitnesses.std()),
        median_fitness=float(np.median(fitnesses)),
        worst_fitness=float(fitnesses.max()),
        feasible_rate=float(np.mean([r.extra.get("is_feasible", False) for r in results])),
        violation_pct=float(np.mean(violation_pcts)),
        mean_time=float(times.mean()),
        n_runs=len(results),
        best_run_idx=best_idx,
    )


def format_metrics(metrics: ExperimentMetrics, solver_name: str = "") -> str:
    """格式化输出实验指标。

    Args:
        metrics: 实验指标。
        solver_name: 求解器名称。

    Returns:
        格式化字符串。
    """
    header = f"=== {solver_name} ===" if solver_name else "=== Metrics ==="
    return (
        f"{header}\n"
        f"  Best fitness:    {metrics.best_fitness:.2f}\n"
        f"  Mean fitness:    {metrics.mean_fitness:.2f} ± {metrics.std_fitness:.2f}\n"
        f"  Median fitness:  {metrics.median_fitness:.2f}\n"
        f"  Worst fitness:   {metrics.worst_fitness:.2f}\n"
        f"  Feasible rate:   {metrics.feasible_rate:.0%}\n"
        f"  Violation %:     {metrics.violation_pct:.2f}%\n"
        f"  Mean time:       {metrics.mean_time:.2f}s\n"
        f"  Runs:            {metrics.n_runs}"
    )