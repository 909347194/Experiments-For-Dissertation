# -*- coding: utf-8 -*-
"""visualizer.py - 实验可视化模块

职责:
    为 DMDE 实验提供全面的可视化功能,包括收敛曲线、代价矩阵热力图、
    分配方案对比图、指标统计图等。

对应论文:
    图 3-X: DMDE 算法收敛曲线。
    图 3-X: 代价矩阵热力图。
    图 3-X: 三种模型分配方案对比。
    图 3-X: 多次运行指标统计。

使用方式:
    from utils.utils_dmde.visualizer import ExperimentVisualizer

    viz = ExperimentVisualizer(output_dir="results/figures")
    viz.plot_convergence(results, scenario_name="N=M 平衡指派")
    viz.plot_cost_matrix(cost_matrix, scenario_name="N=M 平衡指派")
    viz.plot_scenario_comparison(all_scenarios)
    viz.plot_metrics_boxplot(all_scenarios)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from algorithms.algorithm_dmde.base.base_optimizer import SolverResult
from utils.utils_dmde.metrics import ExperimentMetrics


# ── 中文字体配置 ──────────────────────────────────────────────
def _setup_chinese_font():
    """配置 matplotlib 中文字体支持。"""
    import matplotlib
    import os, glob

    # 清除字体缓存,确保新字体能被检测到
    cache_dir = matplotlib.get_cachedir()
    if cache_dir and os.path.isdir(cache_dir):
        for f in glob.glob(os.path.join(cache_dir, 'fontlist-*.json')):
            try:
                os.remove(f)
            except OSError:
                pass

    # 强制重新加载字体管理器
    from matplotlib import font_manager
    font_manager._load_fontmanager(try_read_cache=False)

    # 按平台优先级排列字体
    chinese_fonts = [
        'SimHei',              # 黑体 (Windows)
        'Microsoft YaHei',     # 微软雅黑 (Windows)
        'SimSun',              # 宋体 (Windows)
        'FangSong',            # 仿宋 (Windows)
        'KaiTi',               # 楷体 (Windows)
        'Noto Serif CJK SC',   # Noto Serif CJK (Linux)
        'Noto Sans CJK SC',    # Noto Sans CJK (Linux)
        'WenQuanYi Micro Hei', # 文泉驿微米黑 (Linux)
        'Source Han Sans SC',  # 思源黑体
        'AR PL UMing CN',      # 文鼎
    ]

    # 检查系统中实际可用的字体
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  [visualizer] 使用中文字体: {font}")
            return

    # 如果没找到,尝试用 font_manager 按文件查找
    for pattern in ['simhei', 'msyh', 'simsun', 'NotoSansCJK', 'NotoSerifCJK', 'wqy']:
        matches = font_manager.findSystemFonts()
        for fpath in matches:
            if pattern.lower() in fpath.lower():
                try:
                    font_manager.fontManager.addfont(fpath)
                    prop = font_manager.FontProperties(fname=fpath)
                    plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"  [visualizer] 使用中文字体: {prop.get_name()} ({fpath})")
                    return
                except Exception:
                    continue

    import warnings
    warnings.warn("未找到中文字体,图表中文可能显示为方块。建议安装 SimHei 或 Microsoft YaHei 字体。")

_setup_chinese_font()


# ── 配色方案 ──────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """将场景名转为安全的文件名(移除 Windows 非法字符)。"""
    # 替换 Windows 文件名非法字符: > < : " / \ | ? *
    import re
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = safe.replace(' ', '_')
    return safe
COLORS = {
    'balanced': '#2E86AB',    # 蓝色
    'overloaded': '#A23B72',  # 紫红
    'srp': '#F18F01',         # 橙色
    'best': '#C73E1D',        # 红色
    'mean': '#3B1F2B',        # 深色
    'grid': '#E8E8E8',        # 灰色
    'background': '#FAFAFA',  # 浅灰
}


class ExperimentVisualizer:
    """实验可视化器。

    为 DMDE 实验提供全面的可视化功能。

    Attributes:
        output_dir: 图表输出目录。
        dpi: 图表分辨率。
        figsize: 默认图表尺寸。
    """

    def __init__(
        self,
        output_dir: str | Path = "results/figures",
        dpi: int = 300,
        figsize: tuple[int, int] = (10, 6),
    ):
        """初始化可视化器。

        Args:
            output_dir: 图表输出目录。
            dpi: 图表分辨率。
            figsize: 默认图表尺寸。
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize

    def _save_figure(self, fig: plt.Figure, filename: str) -> Path:
        """保存图表到文件。

        Args:
            fig: matplotlib 图表对象。
            filename: 文件名。

        Returns:
            保存的文件路径。
        """
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        return filepath

    def plot_convergence(
        self,
        results: list[SolverResult],
        scenario_name: str = "",
        show_mean: bool = True,
        show_std: bool = True,
        log_scale: bool = False,
    ) -> Path:
        """绘制收敛曲线。

        Args:
            results: 多次运行的 SolverResult 列表。
            scenario_name: 场景名称。
            show_mean: 是否显示平均收敛曲线。
            show_std: 是否显示标准差区域。
            log_scale: 是否使用对数坐标。

        Returns:
            保存的文件路径。
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 提取所有运行的收敛曲线
        histories = [r.cost_history for r in results]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]

        generations = np.arange(min_len)

        # 绘制每次运行的曲线(透明)
        for i, history in enumerate(histories):
            ax.plot(generations, history, alpha=0.3, linewidth=0.8,
                   color=COLORS['balanced'], label=f'Run {i+1}' if i < 5 else "")

        if show_mean or show_std:
            histories_array = np.array(histories)
            mean_history = np.mean(histories_array, axis=0)
            std_history = np.std(histories_array, axis=0)

            if show_mean:
                ax.plot(generations, mean_history, color=COLORS['best'],
                       linewidth=2, label='平均收敛曲线')

            if show_std:
                ax.fill_between(
                    generations,
                    mean_history - std_history,
                    mean_history + std_history,
                    alpha=0.2, color=COLORS['best'], label='±1 标准差'
                )

        # 标记最优解
        best_idx = np.argmin([r.best_fitness for r in results])
        best_history = histories[best_idx]
        ax.plot(generations, best_history, color=COLORS['best'],
               linewidth=2.5, linestyle='--', label=f'最优运行 (Run {best_idx+1})')

        # 设置图表
        ax.set_xlabel('迭代代数', fontsize=12)
        ax.set_ylabel('适应度值', fontsize=12)
        title = f'DMDE 算法收敛曲线'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        if log_scale:
            ax.set_yscale('log')

        ax.grid(True, alpha=0.3, linestyle='-', color=COLORS['grid'])
        ax.legend(loc='upper right', fontsize=10)
        ax.set_facecolor(COLORS['background'])

        # 添加统计信息
        stats_text = (
            f'运行次数: {len(results)}\n'
            f'最优值: {results[best_idx].best_fitness:.2f}\n'
            f'最终代数: {min_len}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        filename = f"convergence_{_sanitize_filename(scenario_name)}.png"
        return self._save_figure(fig, filename)

    def plot_cost_matrix(
        self,
        cost_matrix: np.ndarray,
        scenario_name: str = "",
        uav_ids: list[int] | None = None,
        target_ids: list[int] | None = None,
        highlight_assignment: list[tuple[int, int]] | None = None,
    ) -> Path:
        """绘制代价矩阵热力图。

        Args:
            cost_matrix: 代价矩阵。
            scenario_name: 场景名称。
            uav_ids: UAV 编号列表。
            target_ids: 目标编号列表。
            highlight_assignment: 高亮显示的分配方案。

        Returns:
            保存的文件路径。
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制热力图
        im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('代价值', fontsize=12)

        # 设置坐标轴
        n_rows, n_cols = cost_matrix.shape
        if uav_ids is None:
            uav_ids = list(range(n_rows))
        if target_ids is None:
            target_ids = list(range(n_cols))

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([f'T{tid}' for tid in target_ids], fontsize=10)
        ax.set_yticklabels([f'U{uid}' for uid in uav_ids], fontsize=10)

        # 在单元格中显示数值
        for i in range(n_rows):
            for j in range(n_cols):
                value = cost_matrix[i, j]
                # 根据值的大小选择文字颜色
                text_color = 'white' if value > np.max(cost_matrix) * 0.6 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                       color=text_color, fontsize=8, fontweight='bold')

        # 高亮显示分配方案
        if highlight_assignment:
            for uav_idx, target_idx in highlight_assignment:
                if uav_idx < n_rows and target_idx < n_cols:
                    rect = plt.Rectangle((target_idx - 0.5, uav_idx - 0.5), 1, 1,
                                       fill=False, edgecolor='blue', linewidth=3)
                    ax.add_patch(rect)

        # 设置标题
        title = '代价矩阵热力图'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_xlabel('目标编号', fontsize=12)
        ax.set_ylabel('UAV 编号', fontsize=12)

        filename = f"cost_matrix_{_sanitize_filename(scenario_name)}.png"
        return self._save_figure(fig, filename)

    def plot_scenario_comparison(
        self,
        scenarios: list[dict[str, Any]],
        metrics_to_plot: list[str] | None = None,
    ) -> Path:
        """绘制场景对比柱状图。

        Args:
            scenarios: 场景数据列表,每个场景包含 'name', 'metrics', 'model_type'。
            metrics_to_plot: 要绘制的指标列表。

        Returns:
            保存的文件路径。
        """
        if metrics_to_plot is None:
            metrics_to_plot = ['best_fitness', 'mean_fitness', 'feasible_rate', 'mean_time']

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        scenario_names = [s['name'] for s in scenarios]
        model_types = [s['model_type'] for s in scenarios]
        colors = [COLORS.get(mt, '#666666') for mt in model_types]

        metric_labels = {
            'best_fitness': '最优适应度',
            'mean_fitness': '平均适应度',
            'std_fitness': '适应度标准差',
            'feasible_rate': '可行解比例',
            'violation_pct': '约束违背率(%)',
            'mean_time': '平均耗时(s)',
            'n_runs': '运行次数',
        }

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # 提取指标值
            values = []
            for sc in scenarios:
                m = sc['metrics']
                values.append(getattr(m, metric, 0))

            # 绘制柱状图
            bars = ax.bar(range(len(scenarios)), values, color=colors, alpha=0.8, edgecolor='white')

            # 在柱子上添加数值
            for bar_idx, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                if metric == 'feasible_rate':
                    text = f'{value:.0%}'
                elif metric == 'mean_time':
                    text = f'{value:.2f}s'
                else:
                    text = f'{value:.1f}'

                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.02,
                       text, ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 设置图表
            ax.set_xlabel('场景', fontsize=12)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
            ax.set_title(metric_labels.get(metric, metric), fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenario_names, rotation=15, ha='right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
            ax.set_facecolor(COLORS['background'])

            # 设置y轴从0开始
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        fig.suptitle('三种模型场景对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        filename = "scenario_comparison.png"
        return self._save_figure(fig, filename)

    def plot_metrics_boxplot(
        self,
        scenarios: list[dict[str, Any]],
    ) -> Path:
        """绘制指标箱线图。

        Args:
            scenarios: 场景数据列表。

        Returns:
            保存的文件路径。
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 适应度箱线图
        ax1 = axes[0]
        fitness_data = []
        labels = []
        colors = []

        for sc in scenarios:
            fitnesses = [r.best_fitness for r in sc['results']]
            fitness_data.append(fitnesses)
            labels.append(sc['name'])
            colors.append(COLORS.get(sc['model_type'], '#666666'))

        bp1 = ax1.boxplot(fitness_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax1.set_ylabel('适应度值', fontsize=12)
        ax1.set_title('适应度分布', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        ax1.tick_params(axis='x', rotation=15)

        # 2. 耗时箱线图
        ax2 = axes[1]
        time_data = []

        for sc in scenarios:
            times = [r.elapsed_seconds for r in sc['results']]
            time_data.append(times)

        bp2 = ax2.boxplot(time_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel('耗时(秒)', fontsize=12)
        ax2.set_title('求解耗时分布', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        ax2.tick_params(axis='x', rotation=15)

        # 3. 约束违背箱线图
        ax3 = axes[2]
        violation_data = []

        for sc in scenarios:
            violations = []
            for r in sc['results']:
                vio = r.extra.get('total_violation', 0.0)
                if r.best_fitness > 0:
                    violations.append(vio / r.best_fitness * 100)
                else:
                    violations.append(0.0)
            violation_data.append(violations)

        bp3 = ax3.boxplot(violation_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_ylabel('约束违背率(%)', fontsize=12)
        ax3.set_title('约束违背分布', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='-', color=COLORS['grid'])
        ax3.tick_params(axis='x', rotation=15)

        fig.suptitle('多次运行指标统计', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        filename = "metrics_boxplot.png"
        return self._save_figure(fig, filename)

    def plot_assignment_visualization(
        self,
        uavs: list[Any],
        targets: list[Any],
        assignment: list[tuple[int, int]],
        scenario_name: str = "",
        cost_matrix: np.ndarray | None = None,
    ) -> Path:
        """绘制分配方案可视化图。

        Args:
            uavs: UAV 列表。
            targets: 目标列表。
            assignment: 分配方案 [(uav_id, target_id), ...]。
            scenario_name: 场景名称。
            cost_matrix: 代价矩阵(可选,用于显示代价值)。

        Returns:
            保存的文件路径。
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # 提取位置信息
        uav_positions = np.array([u.start_pos[:2] for u in uavs])  # 只取经纬度
        target_positions = np.array([t.position[:2] for t in targets])

        # 绘制 UAV 位置
        ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
                  c=COLORS['balanced'], s=200, marker='^',
                  label='UAV', zorder=5, edgecolors='white', linewidth=2)

        # 绘制目标位置
        ax.scatter(target_positions[:, 0], target_positions[:, 1],
                  c=COLORS['best'], s=200, marker='o',
                  label='目标', zorder=5, edgecolors='white', linewidth=2)

        # 添加编号标签
        for i, (x, y) in enumerate(uav_positions):
            ax.annotate(f'U{i}', (x, y), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=10, fontweight='bold',
                       color=COLORS['balanced'])

        for i, (x, y) in enumerate(target_positions):
            ax.annotate(f'T{i}', (x, y), textcoords="offset points",
                       xytext=(0, 15), ha='center', fontsize=10, fontweight='bold',
                       color=COLORS['best'])

        # 绘制分配连线（SRP 模型按 UAV 分组绘制巡游路线）
        # 判断是否 SRP（同一 UAV 出现多次）
        uav_ids_in_assignment = [a[0] for a in assignment]
        is_srp = len(uav_ids_in_assignment) > len(set(uav_ids_in_assignment))

        if is_srp:
            # SRP 模型：按 UAV 分组绘制巡游路线
            routes: dict[int, list[int]] = {}
            for uav_id, target_id in assignment:
                routes.setdefault(uav_id, []).append(target_id)

            route_colors = plt.cm.Set1(np.linspace(0, 1, max(len(routes), 1)))
            for idx, (uav_id, tgt_list) in enumerate(routes.items()):
                color = route_colors[idx % len(route_colors)]
                # UAV → 第一个目标
                if uav_id < len(uavs) and tgt_list:
                    uav_pos = uavs[uav_id].start_pos[:2]
                    first_tgt_pos = targets[tgt_list[0]].position[:2]
                    ax.annotate('', xy=first_tgt_pos, xytext=uav_pos,
                               arrowprops=dict(arrowstyle='->', color=color,
                                              lw=2.5, connectionstyle='arc3,rad=0.1'))
                    # 目标 → 目标
                    for i in range(len(tgt_list) - 1):
                        t1_pos = targets[tgt_list[i]].position[:2]
                        t2_pos = targets[tgt_list[i+1]].position[:2]
                        ax.annotate('', xy=t2_pos, xytext=t1_pos,
                                   arrowprops=dict(arrowstyle='->', color=color,
                                                  lw=2.0, connectionstyle='arc3,rad=0.1',
                                                  linestyle='--'))
        else:
            # balanced/overloaded：逐条绘制分配箭头
            for uav_id, target_id in assignment:
                if uav_id < len(uavs) and target_id < len(targets):
                    uav_pos = uavs[uav_id].start_pos[:2]
                    target_pos = targets[target_id].position[:2]

                    cost_text = ""
                    if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                        cost = cost_matrix[uav_id, target_id]
                        cost_text = f' ({cost:.0f})'

                    ax.annotate('', xy=target_pos, xytext=uav_pos,
                               arrowprops=dict(arrowstyle='->', color='gray',
                                              lw=1.5, connectionstyle='arc3,rad=0.1'))

                    mid_x = (uav_pos[0] + target_pos[0]) / 2
                    mid_y = (uav_pos[1] + target_pos[1]) / 2
                    if cost_text:
                        ax.annotate(cost_text, (mid_x, mid_y), fontsize=8,
                                   ha='center', va='center',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

        # 设置图表
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)
        title = 'UAV-目标分配方案'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])

        # 添加分配统计
        n_assignments = len(assignment)
        total_cost = sum(cost_matrix[u, t] for u, t in assignment
                        if cost_matrix is not None and u < cost_matrix.shape[0] and t < cost_matrix.shape[1]) if cost_matrix is not None else 0

        stats_text = f'分配数: {n_assignments}\n总代价: {total_cost:.0f}' if total_cost > 0 else f'分配数: {n_assignments}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        filename = f"assignment_{_sanitize_filename(scenario_name)}.png"
        return self._save_figure(fig, filename)

    def plot_dem_3d_assignment(
        self,
        dem_terrain: Any,
        uavs: list[Any],
        targets: list[Any],
        assignment: list[tuple[int, int]],
        scenario_name: str = "",
        cost_matrix: np.ndarray | None = None,
        elev_exaggerate: float = 2.0,
        view_elev: float = 35,
        view_azim: float = -60,
    ) -> Path:
        """在 DEM 地形上三维可视化分配方案。

        将 UAV 起飞点、目标点、分配连线叠加到 DEM 三维表面上,
        直观展示任务的空间分布和分配关系。

        Args:
            dem_terrain: DEMTerrain 实例。
            uavs: UAV 列表。
            targets: 目标列表。
            assignment: 分配方案 [(uav_id, target_id), ...]。
            scenario_name: 场景名称。
            cost_matrix: 代价矩阵(可选,显示代价值)。
            elev_exaggerate: 高程夸张系数(默认 2 倍)。
            view_elev: 俯仰角(度)。
            view_azim: 方位角(度)。

        Returns:
            保存的文件路径。
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # ── 提取 DEM 数据 ──
        elevation = dem_terrain.elevation
        bounds = dem_terrain.bounds  # (left, bottom, right, top)
        left, bottom, right, top = bounds

        # 降采样以加速渲染
        h, w = elevation.shape
        # 对于大型 DEM (如3601x3601), stride 不能太小, 否则 "像针一样尖"
        target_grid = 150  # 目标网格分辨率
        step = max(1, min(h, w) // target_grid)
        elev_ds = elevation[::step, ::step]

        # 生成网格坐标
        rows, cols = elev_ds.shape
        xs_deg = np.linspace(left, right, cols)
        ys_deg = np.linspace(top, bottom, rows)  # 上→下
        xx_deg, yy_deg = np.meshgrid(xs_deg, ys_deg)

        # 经纬度转平面米(用参考点)
        from utils.utils_dmde.coord_transform import WGS84Transformer
        ref_lon = (left + right) / 2
        ref_lat = (top + bottom) / 2
        transformer = WGS84Transformer.from_lonlat(ref_lon, ref_lat)

        xx_m, yy_m = transformer.to_xy_batch(xx_deg, yy_deg)

        # 高程夸张
        zz = elev_ds * elev_exaggerate

        # ── 绘制 DEM 表面 ──
        # 颜色映射基于原始高程; stride 随降采样自动调整
        norm = plt.Normalize(np.nanmin(elev_ds), np.nanmax(elev_ds))
        colors = plt.cm.terrain(norm(elev_ds))

        surf_stride = max(1, step // 2)  # 渲染步长与降采样协调
        ax.plot_surface(
            xx_m, yy_m, zz,
            facecolors=colors,
            alpha=0.8,
            rstride=surf_stride, cstride=surf_stride,
            shade=True,
            antialiased=True,
            lightsource=plt.matplotlib.colors.LightSource(azdeg=315, altdeg=45),
        )

        # ── 绘制 UAV 和 Target ──
        uav_positions_xy = []
        uav_positions_z = []
        for u in uavs:
            x, y = transformer.to_xy(u.start_pos[0], u.start_pos[1])
            z = u.start_pos[2] * elev_exaggerate
            uav_positions_xy.append((x, y))
            uav_positions_z.append(z)
            ax.scatter(
                [x], [y], [z],
                c=COLORS['balanced'], s=150, marker='^',
                edgecolors='white', linewidth=1.5, depthshade=False,
                zorder=10,
            )
            ax.text(
                x, y, z + 800 * elev_exaggerate,
                f'U{u.id}', fontsize=8, fontweight='bold',
                color=COLORS['balanced'], ha='center',
            )

        target_positions_xy = []
        target_positions_z = []
        for t in targets:
            x, y = transformer.to_xy(t.position[0], t.position[1])
            z = t.position[2] * elev_exaggerate
            target_positions_xy.append((x, y))
            target_positions_z.append(z)
            ax.scatter(
                [x], [y], [z],
                c=COLORS['best'], s=150, marker='o',
                edgecolors='white', linewidth=1.5, depthshade=False,
                zorder=10,
            )
            ax.text(
                x, y, z + 800 * elev_exaggerate,
                f'T{t.id}', fontsize=8, fontweight='bold',
                color=COLORS['best'], ha='center',
            )

        # ── 绘制分配连线 ──
        for uav_id, target_id in assignment:
            if uav_id < len(uavs) and target_id < len(targets):
                ux, uy = uav_positions_xy[uav_id]
                uz = uav_positions_z[uav_id]
                tx, ty = target_positions_xy[target_id]
                tz = target_positions_z[target_id]

                # 绘制三维弧线(中间抬高)
                n_arc = 20
                arc_ts = np.linspace(0, 1, n_arc)
                arc_x = ux + (tx - ux) * arc_ts
                arc_y = uy + (ty - uy) * arc_ts
                arc_z = uz + (tz - uz) * arc_ts
                # 弧线中间抬高
                arc_lift = np.sin(arc_ts * np.pi) * max(abs(tz - uz), 500) * 0.3
                arc_z += arc_lift

                ax.plot(
                    arc_x, arc_y, arc_z,
                    color='gray', linewidth=1.2, alpha=0.7,
                    linestyle='-', zorder=5,
                )

                # 连线中点标明代价值
                if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                    mid_idx = n_arc // 2
                    cost_val = cost_matrix[uav_id, target_id]
                    ax.text(
                        arc_x[mid_idx], arc_y[mid_idx], arc_z[mid_idx] + 300 * elev_exaggerate,
                        f'{cost_val/1000:.1f}km',
                        fontsize=6, color='#333333', ha='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.7, edgecolor='none'),
                    )

        # ── 设置视角和标签 ──
        ax.view_init(elev=view_elev, azim=view_azim)
        ax.set_xlabel('东向 (m)', fontsize=11, labelpad=10)
        ax.set_ylabel('北向 (m)', fontsize=11, labelpad=10)
        ax.set_zlabel(f'高程 ×{elev_exaggerate:.0f} (m)', fontsize=11, labelpad=10)

        title = 'DEM 地形 + UAV-目标分配方案'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 统计信息
        total_cost = 0
        for uav_id, target_id in assignment:
            if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                total_cost += cost_matrix[uav_id, target_id]

        stats_text = (
            f'UAV: {len(uavs)} | Target: {len(targets)} | '
            f'分配: {len(assignment)} | 总代价: {total_cost/1000:.1f}km'
        )
        ax.text2D(
            0.02, 0.02, stats_text,
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

        # 手动图例(3D 不支持 ax.legend)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['balanced'],
                   markersize=10, label='UAV'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['best'],
                   markersize=10, label='Target'),
            Line2D([0], [0], color='gray', linewidth=1.5, label='分配连线'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        plt.tight_layout()
        filename = f"dem3d_{_sanitize_filename(scenario_name)}.png"
        return self._save_figure(fig, filename)

    def plot_all(
        self,
        scenarios: list[dict[str, Any]],
        uavs_dict: dict[str, list[Any]] | None = None,
        targets_dict: dict[str, list[Any]] | None = None,
        dem_terrain: Any | None = None,
    ) -> list[Path]:
        """生成所有可视化图表。

        Args:
            scenarios: 场景数据列表。
            uavs_dict: 场景名称到 UAV 列表的映射(可选)。
            targets_dict: 场景名称到目标列表的映射(可选)。

        Returns:
            保存的文件路径列表。
        """
        saved_files = []

        # 1. 收敛曲线
        for sc in scenarios:
            filepath = self.plot_convergence(
                sc['results'],
                scenario_name=sc['name']
            )
            saved_files.append(filepath)
            print(f"  ✓ 收敛曲线已保存: {filepath.name}")

        # 2. 代价矩阵热力图
        for sc in scenarios:
            if 'cost_matrix' in sc:
                # 获取最优分配方案用于高亮
                best_result = sc['results'][sc['metrics'].best_run_idx]
                filepath = self.plot_cost_matrix(
                    sc['cost_matrix'],
                    scenario_name=sc['name'],
                    highlight_assignment=best_result.best_assignment
                )
                saved_files.append(filepath)
                print(f"  ✓ 代价矩阵已保存: {filepath.name}")

        # 3. 场景对比图
        filepath = self.plot_scenario_comparison(scenarios)
        saved_files.append(filepath)
        print(f"  ✓ 场景对比已保存: {filepath.name}")

        # 4. 箱线图
        filepath = self.plot_metrics_boxplot(scenarios)
        saved_files.append(filepath)
        print(f"  ✓ 指标箱线图已保存: {filepath.name}")

        # 5. 分配方案可视化（如果提供了位置信息）
        if uavs_dict and targets_dict:
            for sc in scenarios:
                name = sc['name']
                if name in uavs_dict and name in targets_dict:
                    best_result = sc['results'][sc['metrics'].best_run_idx]
                    filepath = self.plot_assignment_visualization(
                        uavs_dict[name],
                        targets_dict[name],
                        best_result.best_assignment,
                        scenario_name=name,
                        cost_matrix=sc.get('cost_matrix')
                    )
                    saved_files.append(filepath)
                    print(f"  ✓ 分配方案已保存: {filepath.name}")

        # 6. DEM 三维分配可视化
        if dem_terrain is not None and uavs_dict and targets_dict:
            for sc in scenarios:
                name = sc['name']
                if name in uavs_dict and name in targets_dict:
                    best_result = sc['results'][sc['metrics'].best_run_idx]
                    filepath = self.plot_dem_3d_assignment(
                        dem_terrain,
                        uavs_dict[name],
                        targets_dict[name],
                        best_result.best_assignment,
                        scenario_name=name,
                        cost_matrix=sc.get('cost_matrix'),
                    )
                    saved_files.append(filepath)
                    print(f"  ✓ DEM 三维图已保存: {filepath.name}")

        return saved_files


# ── 便捷函数 ──────────────────────────────────────────────────

def create_visualizer(output_dir: str | Path = "results/figures") -> ExperimentVisualizer:
    """创建可视化器实例。

    Args:
        output_dir: 图表输出目录。

    Returns:
        ExperimentVisualizer 实例。
    """
    return ExperimentVisualizer(output_dir=output_dir)
