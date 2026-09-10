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

注意两个不同的“违约”概念（易被误读为矛盾）：
    1. 不可行率 = 1 - feasible_rate，**按实验次数计数**。
       例: 5 次运行中 4 次不可行 → 不可行率 80%。
    2. violation_pct = 违约量总和 / 最优代价，**按量纲大小加权**。
       各分量量纲不同（航程=米、时间=秒、时序=违反序对个数），且
       分母 fitness 量级很大（几十万），因此即使 4/5 次不可行，
       只要违背量本身很小（例如时序仅违反 1 个序对），该比率也会
       小到 1e-4 % 量级，被两位小数四舍五入显示成 0.00%。

    因此 :func:`format_metrics` 会同时输出不可行次数与分量明细，
    并由 :func:`_fmt_pct` 自适应保留有效数字，避免误读。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import SolverResult


# 违约分量名称及其中文标签（顺序与 FitnessResult.violation_breakdown 一致）
VIOLATION_KEYS: tuple[str, ...] = ("range", "time", "seq", "window", "sync")
VIOLATION_LABELS: dict[str, str] = {
    "range": "航程",
    "time": "时间",
    "seq": "时序",
    "window": "时间窗",
    "sync": "同步",
}


def _fmt_pct(value: float) -> str:
    """百分数格式化：小值保留有效数字，避免 0.0002% 被显示成 0.00%。"""
    if value == 0.0:
        return "0%"
    if abs(value) >= 0.01:
        return f"{value:.2f}%"
    return f"{value:.3g}%"


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
        violation_pct:   平均约束违背百分比（违约量/代价，量纲加权）。
        violation_breakdown: 各约束类型的平均违约量，用于定位不可行原因。
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
    violation_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def n_feasible(self) -> int:
        """可行的运行次数（按次数统计，与 violation_pct 无关）。"""
        return self.n_runs - self.n_infeasible

    @property
    def n_infeasible(self) -> int:
        """不可行的运行次数（按次数统计，与 violation_pct 无关）。"""
        return int(round((1.0 - self.feasible_rate) * self.n_runs))

    @property
    def infeasible_rate(self) -> float:
        """不可行率 = 1 - 优解率（按次数统计）。

        这是与“不可行 4/5 次”直观对应的指标，推荐作为论文表 3-3
        的约束违背列；``violation_pct`` 是量纲加权的另一个量，二者
        不可互换。
        """
        return 1.0 - self.feasible_rate


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
    breakdown_acc: dict[str, list[float]] = {k: [] for k in VIOLATION_KEYS}
    for r in results:
        vio = r.extra.get("total_violation", 0.0)
        if r.best_fitness > 0:
            violation_pcts.append(vio / r.best_fitness * 100)
        else:
            violation_pcts.append(0.0)

        # 优先使用评估器给出的分量明细，缺失则全部记 0
        bd = r.extra.get("violation_breakdown") or {}
        for key in VIOLATION_KEYS:
            breakdown_acc[key].append(float(bd.get(key, 0.0)))

    violation_breakdown = {
        key: float(np.mean(values)) if values else 0.0
        for key, values in breakdown_acc.items()
    }

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
        violation_breakdown=violation_breakdown,
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

    detail = " ".join(
        f"{VIOLATION_LABELS[key]}={metrics.violation_breakdown.get(key, 0.0):.3f}"
        for key in VIOLATION_KEYS
    )

    return (
        f"{header}\n"
        f"  Best fitness:     {metrics.best_fitness:.2f}\n"
        f"  Mean fitness:     {metrics.mean_fitness:.2f} ± {metrics.std_fitness:.2f}\n"
        f"  Median fitness:   {metrics.median_fitness:.2f}\n"
        f"  Worst fitness:    {metrics.worst_fitness:.2f}\n"
        f"  Feasible rate:    {metrics.feasible_rate:.0%}"
        f" ({metrics.n_feasible}/{metrics.n_runs} feasible)\n"
        f"  Infeasible rate:  {metrics.infeasible_rate:.0%}"
        f" ({metrics.n_infeasible}/{metrics.n_runs} infeasible)   # 按次数统计，推荐口径\n"
        f"  Violation %:      {_fmt_pct(metrics.violation_pct)}"
        f"   # 违约量总和/代价（量纲加权，勿与上面两个率混淆）\n"
        f"  Violation detail: {detail}   # 各约束平均违约量\n"
        f"  Mean time:        {metrics.mean_time:.2f}s\n"
        f"  Runs:             {metrics.n_runs}"
    )