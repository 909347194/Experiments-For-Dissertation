# -*- coding: utf-8 -*-
"""visualizer.py — LLM-DMDE 实验可视化编排入口。

职责：
    编排各图表模块，提供 plot_all 一键生成所有图表。
    复用 DMDE 基线的标准图表（收敛、代价矩阵、分配方案等），
    新增 LLM 决策专用图表（决策时间线、策略分布等）。

使用方式：
    from visualization.visualizer import ExperimentVisualizer

    viz = ExperimentVisualizer(output_dir="results/figures")
    viz.plot_all(scenarios, uavs_dict, targets_dict,
                 llm_decisions=all_llm_decisions, dem_terrain=dem)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from algorithms.algorithm_llm_enhanced_dmde.base.base_optimizer import SolverResult
from utils.utils_dmde.metrics import ExperimentMetrics

from ._common import COLORS, _sanitize_filename
from plot_llm_decisions import LLMDecisionPlotter

# 复用 DMDE 基线的绘图模块
import sys
DMDE_VIZ_DIR = Path(__file__).resolve().parents[2] / "exp_dmde" / "exp_dmde_01" / "visualization"
sys.path.insert(0, str(DMDE_VIZ_DIR))
from plot_convergence import ConvergencePlotter
from plot_cost_matrix import CostMatrixPlotter
from plot_comparison import ComparisonPlotter
from plot_assignment import AssignmentPlotter
from plot_dem3d import Dem3DPlotter


class ExperimentVisualizer:
    """LLM-DMDE 实验可视化编排器。"""

    def __init__(
        self,
        output_dir: str | Path = "results/figures",
        dpi: int = 300,
        figsize: tuple[int, int] = (10, 6),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

        # 复用 DMDE 基线的绘图器
        self._convergence = ConvergencePlotter(self.output_dir, dpi, figsize)
        self._cost_matrix = CostMatrixPlotter(self.output_dir, dpi, figsize)
        self._comparison = ComparisonPlotter(self.output_dir, dpi, figsize)
        self._assignment = AssignmentPlotter(self.output_dir, dpi, figsize)
        self._dem3d = Dem3DPlotter(self.output_dir, dpi, figsize)

        # LLM 专用绘图器
        self._llm_decisions = LLMDecisionPlotter(self.output_dir, dpi, (12, 8))

    # ── 标准图表（委托到 DMDE 基线模块）─────────────────────

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

    # ── LLM 专用图表 ────────────────────────────────────────

    def plot_llm_decision_timeline(self, llm_decisions, scenario_name="") -> Path:
        return self._llm_decisions.plot_decision_timeline(llm_decisions, scenario_name)

    def plot_llm_decision_summary(self, llm_decisions, scenario_name="") -> Path:
        return self._llm_decisions.plot_decision_summary(llm_decisions, scenario_name)

    # ── 一键生成 ─────────────────────────────────────────────

    def plot_all(
        self,
        scenarios: list[dict[str, Any]],
        uavs_dict: dict[str, list[Any]] | None = None,
        targets_dict: dict[str, list[Any]] | None = None,
        dem_terrain: Any | None = None,
        llm_decisions: dict[int, list[dict]] | None = None,
    ) -> list[Path]:
        """生成所有可视化图表。

        Args:
            scenarios:     场景 dict 列表。
            uavs_dict:     场景名 → UAV 列表。
            targets_dict:  场景名 → Target 列表。
            dem_terrain:   DEM 地形对象。
            llm_decisions: LLM 决策日志 {run_idx: [decision, ...]}。
        """
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


def create_visualizer(output_dir: str | Path = "results/figures") -> ExperimentVisualizer:
    return ExperimentVisualizer(output_dir=output_dir)
