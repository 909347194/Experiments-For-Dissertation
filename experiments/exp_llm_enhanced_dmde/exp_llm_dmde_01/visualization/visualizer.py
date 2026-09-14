# -*- coding: utf-8 -*-
"""visualizer.py — 实验可视化编排入口。

职责：
    编排各图表模块，提供 plot_all 一键生成所有图表。
    具体绘图逻辑分散到各子模块。

使用方式：
    from visualization.visualizer import ExperimentVisualizer

    viz = ExperimentVisualizer(output_dir="results/figures", algo_name="LLM-DMDE")
    viz.plot_convergence(results, scenario_name="N=M")
    viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from algorithms.algorithm_dmde.base.base_optimizer import SolverResult
from utils.utils_dmde.metrics import ExperimentMetrics

from ._common import COLORS, _sanitize_filename
from .plot_convergence import ConvergencePlotter
from .plot_cost_matrix import CostMatrixPlotter
from .plot_comparison import ComparisonPlotter
from .plot_assignment import AssignmentPlotter
from .plot_dem3d import Dem3DPlotter
from .plot_llm_decisions import LLMDecisionPlotter


class ExperimentVisualizer:
    """实验可视化编排器。

    组合各子模块的绘图器，提供统一入口。
    """

    def __init__(
        self,
        output_dir: str | Path = "results/figures",
        dpi: int = 300,
        figsize: tuple[int, int] = (10, 6),
        algo_name: str = "DMDE",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.algo_name = algo_name

        self._convergence = ConvergencePlotter(self.output_dir, dpi, figsize, algo_name=algo_name)
        self._cost_matrix = CostMatrixPlotter(self.output_dir, dpi, figsize)
        self._comparison = ComparisonPlotter(self.output_dir, dpi, figsize)
        self._assignment = AssignmentPlotter(self.output_dir, dpi, figsize)
        self._dem3d = Dem3DPlotter(self.output_dir, dpi, figsize)
        self._llm_decisions = LLMDecisionPlotter(self.output_dir, dpi, (12, 8))

    # ── 委托到子模块 ─────────────────────────────────────────

    def plot_convergence(self, results, scenario_name="", **kw) -> Path:
        return self._convergence.plot(results, scenario_name=scenario_name, **kw)

    def plot_cost_matrix(self, cost_matrix, scenario_name="", **kw) -> Path:
        return self._cost_matrix.plot(cost_matrix, scenario_name=scenario_name, **kw)

    def plot_scenario_comparison(self, scenarios, **kw) -> Path:
        return self._comparison.plot_scenario_comparison(scenarios, **kw)

    def plot_metrics_boxplot(self, scenarios, **kw) -> Path:
        return self._comparison.plot_metrics_boxplot(scenarios, **kw)

    def plot_assignment_visualization(self, uavs, targets, assignment, scenario_name="", **kw) -> Path:
        return self._assignment.plot(uavs, targets, assignment, scenario_name=scenario_name, **kw)

    def plot_dem_3d_assignment(self, dem_terrain, uavs, targets, assignment, scenario_name="", **kw) -> Path:
        return self._dem3d.plot(dem_terrain, uavs, targets, assignment, scenario_name=scenario_name, **kw)

    # ── 一键生成 ─────────────────────────────────────────────

    def plot_all(
        self,
        scenarios: list[dict[str, Any]],
        uavs_dict: dict[str, list[Any]] | None = None,
        targets_dict: dict[str, list[Any]] | None = None,
        dem_terrain: Any | None = None,
        llm_decisions: dict[int, list[dict]] | None = None,
    ) -> list[Path]:
        """生成所有可视化图表。"""
        saved = []

        # ── 标准图表 ─────────────────────────────────────────
        for sc in scenarios:
            p = self.plot_convergence(sc['results'], scenario_name=sc['name'])
            saved.append(p); print(f"  ✓ 收敛曲线: {p.name}")

        for sc in scenarios:
            if 'cost_matrix' in sc:
                best = sc['results'][sc['metrics'].best_run_idx]
                p = self.plot_cost_matrix(sc['cost_matrix'], scenario_name=sc['name'],
                                          highlight_assignment=best.best_assignment)
                saved.append(p); print(f"  ✓ 代价矩阵: {p.name}")

        p = self.plot_scenario_comparison(scenarios)
        saved.append(p); print(f"  ✓ 场景对比: {p.name}")

        p = self.plot_metrics_boxplot(scenarios)
        saved.append(p); print(f"  ✓ 箱线图: {p.name}")

        if uavs_dict and targets_dict:
            for sc in scenarios:
                name = sc['name']
                if name in uavs_dict and name in targets_dict:
                    best = sc['results'][sc['metrics'].best_run_idx]
                    p = self.plot_assignment_visualization(
                        uavs_dict[name], targets_dict[name],
                        best.best_assignment, scenario_name=name,
                        cost_matrix=sc.get('cost_matrix'))
                    saved.append(p); print(f"  ✓ 分配方案: {p.name}")

        if dem_terrain and uavs_dict and targets_dict:
            for sc in scenarios:
                name = sc['name']
                if name in uavs_dict and name in targets_dict:
                    best = sc['results'][sc['metrics'].best_run_idx]
                    p = self.plot_dem_3d_assignment(
                        dem_terrain, uavs_dict[name], targets_dict[name],
                        best.best_assignment, scenario_name=name,
                        cost_matrix=sc.get('cost_matrix'))
                    saved.append(p); print(f"  ✓ DEM 3D: {p.name}")

        # ── LLM 专用图表 ────────────────────────────────────
        if llm_decisions:
            for sc in scenarios:
                llm_saved = self._llm_decisions.plot_all(
                    llm_decisions, scenario_name=sc['name'])
                saved.extend(llm_saved)

        return saved


def create_visualizer(
    output_dir: str | Path = "results/figures",
    algo_name: str = "DMDE",
) -> ExperimentVisualizer:
    return ExperimentVisualizer(output_dir=output_dir, algo_name=algo_name)
