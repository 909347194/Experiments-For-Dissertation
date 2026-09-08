# -*- coding: utf-8 -*-
"""visualizer.py — 实验可视化模块

职责：
    为 DMDE 实验提供全面的可视化功能，包括收敛曲线、代价矩阵热力图、
    分配方案对比图、指标统计图等。

对应论文：
    图 3-X: DMDE 算法收敛曲线。
    图 3-X: 代价矩阵热力图。
    图 3-X: 三种模型分配方案对比。
    图 3-X: 多次运行指标统计。

使用方式：
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
    # 尝试使用系统中文字体
    chinese_fonts = [
        'Noto Serif CJK SC',   # Noto Serif CJK (serif)
        'Noto Sans CJK SC',    # Noto Sans CJK (sans-serif)
        'SimHei',              # 黑体
        'Microsoft YaHei',     # 微软雅黑
        'WenQuanYi Micro Hei', # 文泉驿微米黑
        'Source Han Sans SC',  # 思源黑体
        'AR PL UMing CN',      # 文鼎
    ]
    
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # 测试字体是否可用
            fig, ax = plt.subplots()
            ax.set_title('测试')
            plt.close(fig)
            return
        except:
            continue
    
    # 如果没有中文字体，使用默认字体并警告
    import warnings
    warnings.warn("未找到中文字体，图表中文可能显示为方块。建议安装中文字体。")

_setup_chinese_font()


# ── 配色方案 ──────────────────────────────────────────────────
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
        
        # 绘制每次运行的曲线（透明）
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
        
        filename = f"convergence_{scenario_name.replace(' ', '_').replace('=', 'eq')}.png"
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
        
        filename = f"cost_matrix_{scenario_name.replace(' ', '_').replace('=', 'eq')}.png"
        return self._save_figure(fig, filename)
    
    def plot_scenario_comparison(
        self,
        scenarios: list[dict[str, Any]],
        metrics_to_plot: list[str] | None = None,
    ) -> Path:
        """绘制场景对比柱状图。
        
        Args:
            scenarios: 场景数据列表，每个场景包含 'name', 'metrics', 'model_type'。
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
            cost_matrix: 代价矩阵（可选，用于显示代价值）。
            
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
        
        # 绘制分配连线
        for uav_id, target_id in assignment:
            if uav_id < len(uavs) and target_id < len(targets):
                uav_pos = uavs[uav_id].start_pos[:2]
                target_pos = targets[target_id].position[:2]
                
                # 获取代价值
                cost_text = ""
                if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                    cost = cost_matrix[uav_id, target_id]
                    cost_text = f' ({cost:.0f})'
                
                # 绘制连线
                ax.annotate('', xy=target_pos, xytext=uav_pos,
                           arrowprops=dict(arrowstyle='->', color='gray', 
                                          lw=1.5, connectionstyle='arc3,rad=0.1'))
                
                # 在连线中点添加代价值
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
        
        filename = f"assignment_{scenario_name.replace(' ', '_').replace('=', 'eq')}.png"
        return self._save_figure(fig, filename)
    
    def plot_all(
        self,
        scenarios: list[dict[str, Any]],
        uavs_dict: dict[str, list[Any]] | None = None,
        targets_dict: dict[str, list[Any]] | None = None,
    ) -> list[Path]:
        """生成所有可视化图表。
        
        Args:
            scenarios: 场景数据列表。
            uavs_dict: 场景名称到 UAV 列表的映射（可选）。
            targets_dict: 场景名称到目标列表的映射（可选）。
            
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
