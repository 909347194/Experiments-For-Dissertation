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
    'balanced': '#2563EB',    # 亮蓝色 (tailwind blue-600)
    'overloaded': '#9333EA',  # 紫色 (tailwind violet-600)
    'srp': '#EA580C',         # 深橙色 (tailwind orange-600)
    'best': '#DC2626',        # 红色 (tailwind red-600)
    'mean': '#1F2937',        # 深灰 (tailwind gray-800)
    'grid': '#E5E7EB',        # 浅灰 (tailwind gray-200)
    'background': '#F9FAFB',  # 极浅灰 (tailwind gray-50)
    'uav': '#2563EB',         # UAV 蓝色
    'target': '#DC2626',      # Target 红色
    'accent_green': '#059669',  # 强调绿
    'accent_gold': '#D97706',   # 强调金
    'text_primary': '#111827',  # 主文字
    'text_secondary': '#6B7280', # 次文字
}

PLOT_STYLE = {
    'title_size': 15,
    'title_weight': 'bold',
    'label_size': 12,
    'tick_size': 10,
    'legend_size': 10,
    'annotation_size': 9,
    'grid_alpha': 0.4,
    'dpi': 300,
    'spine_linewidth': 0.8,
}


# ── DEM 渲染辅助（无 scipy 依赖）──────────────────────────────

def _fill_nan(arr: np.ndarray, iterations: int = 80) -> np.ndarray:
    """用有效邻居均值迭代扩散填充 NaN（平滑填补空洞）。"""
    out = arr.astype(float)
    mask = np.isnan(out)
    if not mask.any():
        return out
    out[mask] = float(np.nanmin(arr))
    for _ in range(iterations):
        up = np.roll(out, 1, axis=0); up[0, :] = out[0, :]
        down = np.roll(out, -1, axis=0); down[-1, :] = out[-1, :]
        left = np.roll(out, 1, axis=1); left[:, 0] = out[:, 0]
        right = np.roll(out, -1, axis=1); right[:, -1] = out[:, -1]
        new = (up + down + left + right) / 4.0
        out[mask] = new[mask]
    return out


def _smooth2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯平滑（消除 DEM 针状锯齿）。"""
    if sigma <= 0:
        return arr.astype(float)
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    out = arr.astype(float)
    for i in range(out.shape[0]):
        out[i, :] = np.convolve(out[i, :], k, mode='same')
    for j in range(out.shape[1]):
        out[:, j] = np.convolve(out[:, j], k, mode='same')
    return out


def _block_reduce(arr: np.ndarray, target_max: int) -> np.ndarray:
    """块均值降采样到不超过 target_max 的网格（保持表面平滑）。"""
    h, w = arr.shape
    step = max(1, int(np.ceil(max(h, w) / target_max)))
    if step == 1:
        return arr.copy()
    h2 = (h // step) * step
    w2 = (w // step) * step
    return arr[:h2, :w2].reshape(h2 // step, step, w2 // step, step).mean(axis=(1, 3))


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
        self._apply_global_style()

    def _apply_global_style(self) -> None:
        """应用全局 matplotlib 样式。"""
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'font.size': PLOT_STYLE['tick_size'],
            'axes.titlesize': PLOT_STYLE['title_size'],
            'axes.titleweight': PLOT_STYLE['title_weight'],
            'axes.labelsize': PLOT_STYLE['label_size'],
            'axes.linewidth': PLOT_STYLE['spine_linewidth'],
            'xtick.labelsize': PLOT_STYLE['tick_size'],
            'ytick.labelsize': PLOT_STYLE['tick_size'],
            'legend.fontsize': PLOT_STYLE['legend_size'],
            'legend.framealpha': 0.95,
            'legend.edgecolor': '#D1D5DB',
            'grid.alpha': PLOT_STYLE['grid_alpha'],
            'grid.linestyle': '-',
            'grid.color': COLORS['grid'],
            'axes.facecolor': COLORS['background'],
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'savefig.bbox': 'tight',
            'savefig.edgecolor': 'none',
        })

    def _style_axes(self, ax: plt.Axes) -> None:
        """统一美化坐标轴样式。"""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(PLOT_STYLE['spine_linewidth'])
        ax.spines['bottom'].set_linewidth(PLOT_STYLE['spine_linewidth'])
        ax.tick_params(axis='both', which='both', length=3, width=0.8)
        ax.grid(True, alpha=PLOT_STYLE['grid_alpha'], linestyle='-', color=COLORS['grid'])
        ax.set_facecolor(COLORS['background'])

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
                   facecolor='white', edgecolor='none', pad_inches=0.15)
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

        histories = [r.cost_history for r in results]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        generations = np.arange(min_len)

        for i, history in enumerate(histories):
            ax.plot(generations, history, alpha=0.25, linewidth=0.85,
                   color=COLORS['balanced'], label=f'单次运行' if i == 0 else "")

        if show_mean or show_std:
            histories_array = np.array(histories)
            mean_history = np.mean(histories_array, axis=0)
            std_history = np.std(histories_array, axis=0)

            if show_std:
                ax.fill_between(
                    generations,
                    np.maximum(mean_history - std_history, 0),
                    mean_history + std_history,
                    alpha=0.2, color=COLORS['mean'], label='±1 标准差',
                    linewidth=0
                )

            if show_mean:
                ax.plot(generations, mean_history, color=COLORS['srp'],
                       linewidth=2.2, label='均值曲线', zorder=8)

        best_idx = np.argmin([r.best_fitness for r in results])
        best_history = histories[best_idx]
        ax.plot(generations, best_history, color=COLORS['best'],
               linewidth=2.2, linestyle='--', label=f'最优运行 (Run {best_idx+1})', zorder=9)

        ax.set_xlabel('迭代代数', fontsize=PLOT_STYLE['label_size'])
        ax.set_ylabel('适应度值', fontsize=PLOT_STYLE['label_size'])
        title = f'DMDE 算法收敛曲线'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=PLOT_STYLE['title_size'],
                     fontweight=PLOT_STYLE['title_weight'], pad=12)

        if log_scale:
            ax.set_yscale('log')

        self._style_axes(ax)
        ax.legend(loc='upper right', fontsize=PLOT_STYLE['legend_size'],
                  frameon=True, fancybox=True, borderpad=0.8)

        stats_text = (
            f'运行次数: {len(results)}  |  最优值: {results[best_idx].best_fitness:.2f}  |  迭代: {min_len}'
        )
        ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
                fontsize=PLOT_STYLE['annotation_size'], ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.55))

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
        n_rows, n_cols = cost_matrix.shape
        fig_w = max(7.5, n_cols * 0.8 + 2)
        fig_h = max(5.5, n_rows * 0.7 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        vmin, vmax = float(cost_matrix.min()), float(cost_matrix.max())
        im = ax.imshow(cost_matrix, cmap='YlOrRd', aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        cbar = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
        cbar.set_label('代价值', fontsize=PLOT_STYLE['label_size'])
        cbar.ax.tick_params(labelsize=PLOT_STYLE['tick_size'])

        if uav_ids is None:
            uav_ids = list(range(n_rows))
        if target_ids is None:
            target_ids = list(range(n_cols))

        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([f'T{tid}' for tid in target_ids], fontsize=PLOT_STYLE['tick_size'])
        ax.set_yticklabels([f'U{uid}' for uid in uav_ids], fontsize=PLOT_STYLE['tick_size'])

        for i in range(n_rows):
            for j in range(n_cols):
                value = cost_matrix[i, j]
                text_color = 'white' if value > (vmin + vmax) * 0.55 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

        if highlight_assignment:
            for uav_idx, target_idx in highlight_assignment:
                if uav_idx < n_rows and target_idx < n_cols:
                    rect = plt.Rectangle((target_idx - 0.5, uav_idx - 0.5), 1, 1,
                                       fill=False, edgecolor='blue', linewidth=2.8)
                    ax.add_patch(rect)

        title = '代价矩阵热力图'
        if scenario_name:
            title += f' - {scenario_name}'
        ax.set_title(title, fontsize=PLOT_STYLE['title_size'],
                     fontweight=PLOT_STYLE['title_weight'], pad=12)

        ax.set_xlabel('目标编号', fontsize=PLOT_STYLE['label_size'])
        ax.set_ylabel('UAV 编号', fontsize=PLOT_STYLE['label_size'])
        ax.set_facecolor(COLORS['background'])

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
        fig, axes = plt.subplots(1, n_metrics, figsize=(5.2 * n_metrics, 5.8))

        if n_metrics == 1:
            axes = [axes]

        scenario_names = [s['name'] for s in scenarios]
        model_types = [s['model_type'] for s in scenarios]
        bar_colors = [COLORS.get(mt, '#6B7280') for mt in model_types]

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

            values = []
            for sc in scenarios:
                m = sc['metrics']
                values.append(getattr(m, metric, 0))

            bars = ax.bar(range(len(scenarios)), values,
                         color=bar_colors, alpha=0.88,
                         edgecolor='white', linewidth=1.5,
                         width=0.68, zorder=3)

            ymax = max(values) * 1.2 if max(values) > 0 else 1
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric == 'feasible_rate':
                    text = f'{value:.1%}'
                elif metric == 'mean_time':
                    text = f'{value:.2f}s'
                elif metric in ('violation_pct', 'n_runs'):
                    text = f'{value:.2f}' if metric == 'violation_pct' else f'{int(value)}'
                else:
                    text = f'{value:.1f}'

                ax.text(bar.get_x() + bar.get_width() / 2.,
                       height + ymax * 0.025,
                       text, ha='center', va='bottom',
                       fontsize=9.5, fontweight='bold',
                       color=COLORS['text_primary'])

            ax.set_xlabel('场景', fontsize=PLOT_STYLE['label_size'], labelpad=8)
            ax.set_ylabel(metric_labels.get(metric, metric),
                         fontsize=PLOT_STYLE['label_size'], labelpad=8)
            ax.set_title(metric_labels.get(metric, metric),
                        fontsize=PLOT_STYLE['title_size'] - 1,
                        fontweight=PLOT_STYLE['title_weight'], pad=10)
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenario_names, rotation=18, ha='right',
                              fontsize=PLOT_STYLE['tick_size'])
            ax.set_ylim(0, ymax)
            self._style_axes(ax)
            ax.grid(axis='x', visible=False)

        fig.suptitle('模型场景对比', fontsize=PLOT_STYLE['title_size'] + 1,
                     fontweight=PLOT_STYLE['title_weight'], y=1.03)
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
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

        labels = [sc['name'] for sc in scenarios]
        colors = [COLORS.get(sc['model_type'], '#6B7280') for sc in scenarios]

        def _style_boxplot(bp, fill_colors):
            for patch, c in zip(bp['boxes'], fill_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.78)
                patch.set_edgecolor('#374151')
                patch.set_linewidth(1.1)
            for median in bp['medians']:
                median.set_color('#111827')
                median.set_linewidth(1.6)
            for whisker in bp['whiskers']:
                whisker.set_color('#374151')
                whisker.set_linewidth(1.1)
                whisker.set_linestyle('--')
            for cap in bp['caps']:
                cap.set_color('#374151')
                cap.set_linewidth(1.1)
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='#F87171',
                         markersize=5, alpha=0.7, markeredgecolor='white',
                         markeredgewidth=0.6)

        ax1 = axes[0]
        fitness_data = [[r.best_fitness for r in sc['results']] for sc in scenarios]
        bp1 = ax1.boxplot(fitness_data, tick_labels=labels, patch_artist=True,
                          medianprops={'linewidth': 1.6}, widths=0.62)
        _style_boxplot(bp1, colors)
        ax1.set_ylabel('适应度值', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax1.set_title('适应度分布', fontsize=PLOT_STYLE['title_size'] - 1,
                     fontweight=PLOT_STYLE['title_weight'], pad=10)
        ax1.tick_params(axis='x', rotation=18, labelsize=PLOT_STYLE['tick_size'])
        self._style_axes(ax1)
        ax1.grid(axis='x', visible=False)

        ax2 = axes[1]
        time_data = [[r.elapsed_seconds for r in sc['results']] for sc in scenarios]
        bp2 = ax2.boxplot(time_data, tick_labels=labels, patch_artist=True,
                          medianprops={'linewidth': 1.6}, widths=0.62)
        _style_boxplot(bp2, colors)
        ax2.set_ylabel('耗时 (秒)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax2.set_title('求解耗时分布', fontsize=PLOT_STYLE['title_size'] - 1,
                     fontweight=PLOT_STYLE['title_weight'], pad=10)
        ax2.tick_params(axis='x', rotation=18, labelsize=PLOT_STYLE['tick_size'])
        self._style_axes(ax2)
        ax2.grid(axis='x', visible=False)

        ax3 = axes[2]
        violation_data = []
        for sc in scenarios:
            violations = []
            for r in sc['results']:
                vio = r.extra.get('total_violation', 0.0)
                violations.append(vio / r.best_fitness * 100 if r.best_fitness > 0 else 0.0)
            violation_data.append(violations)
        bp3 = ax3.boxplot(violation_data, tick_labels=labels, patch_artist=True,
                          medianprops={'linewidth': 1.6}, widths=0.62)
        _style_boxplot(bp3, colors)
        ax3.set_ylabel('约束违背率 (%)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax3.set_title('约束违背分布', fontsize=PLOT_STYLE['title_size'] - 1,
                     fontweight=PLOT_STYLE['title_weight'], pad=10)
        ax3.tick_params(axis='x', rotation=18, labelsize=PLOT_STYLE['tick_size'])
        self._style_axes(ax3)
        ax3.grid(axis='x', visible=False)

        fig.suptitle('多次运行指标统计', fontsize=PLOT_STYLE['title_size'] + 1,
                     fontweight=PLOT_STYLE['title_weight'], y=1.03)
        plt.tight_layout()

        filename = "metrics_boxplot.png"
        return self._save_figure(fig, filename)

    def export_statistics_tables(
        self,
        scenarios: list[dict[str, Any]],
        prefix: str = "statistics",
    ) -> list[Path]:
        """导出实验统计表格（CSV + Markdown）。

        生成三份文件：
        - {prefix}_summary.csv / .md：每个场景一行，列为各项统计指标。
        - {prefix}_runs.csv：每个场景每次运行一行的明细。

        Args:
            scenarios: 场景数据列表（与 plot_all 使用相同结构）。
            prefix: 文件名前缀。

        Returns:
            导出的文件路径列表。
        """
        import csv

        summary_fields = [
            ("best_fitness", "最优适应度"),
            ("mean_fitness", "平均适应度"),
            ("std_fitness", "适应度标准差"),
            ("median_fitness", "适应度中位数"),
            ("worst_fitness", "最差适应度"),
            ("feasible_rate", "可行解比例"),
            ("violation_pct", "约束违背率(%)"),
            ("mean_time", "平均耗时(s)"),
            ("n_runs", "运行次数"),
        ]

        # ── 汇总表 ──
        rows = []
        for sc in scenarios:
            m = sc["metrics"]
            row = {"场景": sc["name"]}
            for key, label in summary_fields:
                val = getattr(m, key)
                if key == "feasible_rate":
                    val = f"{val:.1%}"
                elif key in ("mean_time", "violation_pct"):
                    val = f"{val:.2f}"
                elif key == "n_runs":
                    val = f"{int(val)}"
                else:
                    val = f"{val:.2f}"
                row[label] = val
            rows.append(row)

        headers = ["场景"] + [label for _, label in summary_fields]

        summary_csv = self.output_dir / f"{prefix}_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        summary_md = self.output_dir / f"{prefix}_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "---|" * len(headers) + "\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")

        # ── 逐次运行明细表 ──
        detail_csv = self.output_dir / f"{prefix}_runs.csv"
        detail_fields = ["场景", "运行", "最优适应度", "可行", "总违背量", "耗时(s)", "分配方案"]
        with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(detail_fields)
            for sc in scenarios:
                for i, r in enumerate(sc["results"]):
                    assign = "; ".join(f"U{u}→T{t}" for u, t in r.best_assignment)
                    writer.writerow([
                        sc["name"], i + 1,
                        f"{r.best_fitness:.2f}",
                        "是" if r.extra.get("is_feasible") else "否",
                        f"{r.extra.get('total_violation', 0.0):.2f}",
                        f"{r.elapsed_seconds:.2f}",
                        assign,
                    ])

        return [summary_csv, summary_md, detail_csv]

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
        fig, ax = plt.subplots(figsize=(self.figsize[0] + 1, self.figsize[1] + 0.5))

        uav_positions = np.array([u.start_pos[:2] for u in uavs])
        target_positions = np.array([t.position[:2] for t in targets])

        all_x = np.concatenate([uav_positions[:, 0], target_positions[:, 0]])
        all_y = np.concatenate([uav_positions[:, 1], target_positions[:, 1]])
        margin_x = (all_x.max() - all_x.min()) * 0.12 if len(all_x) > 1 else 0.01
        margin_y = (all_y.max() - all_y.min()) * 0.12 if len(all_y) > 1 else 0.01
        ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

        uav_ids_in_assignment = [a[0] for a in assignment]
        is_srp = len(uav_ids_in_assignment) > len(set(uav_ids_in_assignment))

        if is_srp:
            routes: dict[int, list[int]] = {}
            for uav_id, target_id in assignment:
                routes.setdefault(uav_id, []).append(target_id)
            n_routes = max(len(routes), 1)
            try:
                cmap = plt.colormaps['tab10'].resampled(n_routes)
            except (AttributeError, KeyError):
                cmap = plt.cm.get_cmap('tab10', n_routes)
            route_colors = [cmap(i) for i in range(n_routes)]

            for idx, (uav_id, tgt_list) in enumerate(routes.items()):
                color = route_colors[idx % n_routes]
                if uav_id < len(uavs) and tgt_list:
                    uav_pos = uavs[uav_id].start_pos[:2]
                    pts_x = [uav_pos[0]] + [targets[t].position[0] for t in tgt_list]
                    pts_y = [uav_pos[1]] + [targets[t].position[1] for t in tgt_list]
                    ax.plot(pts_x, pts_y, color=color, linewidth=2.2, alpha=0.8,
                           linestyle='-', zorder=3)
                    for i in range(len(pts_x) - 1):
                        dx = pts_x[i+1] - pts_x[i]
                        dy = pts_y[i+1] - pts_y[i]
                        ax.annotate('', xy=(pts_x[i+1], pts_y[i+1]),
                                   xytext=(pts_x[i], pts_y[i]),
                                   arrowprops=dict(arrowstyle='-|>', color=color,
                                                  lw=1.8, shrinkA=10, shrinkB=10))
        else:
            n_assign = len([a for a in assignment if a[0] < len(uavs) and a[1] < len(targets)])
            try:
                cmap = plt.colormaps['viridis'].resampled(max(n_assign, 1))
            except (AttributeError, KeyError):
                cmap = plt.cm.get_cmap('viridis', max(n_assign, 1))
            for idx, (uav_id, target_id) in enumerate(assignment):
                if uav_id < len(uavs) and target_id < len(targets):
                    uav_pos = uavs[uav_id].start_pos[:2]
                    target_pos = targets[target_id].position[:2]
                    color = cmap(idx % max(n_assign, 1))
                    ax.annotate('', xy=target_pos, xytext=uav_pos,
                               arrowprops=dict(arrowstyle='-|>', color=color,
                                              lw=1.8, alpha=0.85,
                                              shrinkA=12, shrinkB=12,
                                              connectionstyle='arc3,rad=0.08'),
                               zorder=3)
                    if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                        cost = cost_matrix[uav_id, target_id]
                        mid_x = (uav_pos[0] + target_pos[0]) / 2
                        mid_y = (uav_pos[1] + target_pos[1]) / 2
                        ax.annotate(f'{cost/1000:.1f}km', (mid_x, mid_y),
                                   fontsize=8, ha='center', va='center',
                                   color=COLORS['text_primary'],
                                   bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='#FEF3C7',
                                            edgecolor='#F59E0B',
                                            linewidth=0.7, alpha=0.95),
                                   zorder=7)

        ax.scatter(uav_positions[:, 0], uav_positions[:, 1],
                  c=COLORS['uav'], s=260, marker='^',
                  label='UAV 起飞点', zorder=6,
                  edgecolors='white', linewidth=2.2)

        ax.scatter(target_positions[:, 0], target_positions[:, 1],
                  c=COLORS['target'], s=220, marker='o',
                  label='目标点', zorder=6,
                  edgecolors='white', linewidth=2.2)

        for i, (x, y) in enumerate(uav_positions):
            ax.annotate(f'U{uavs[i].id}', (x, y), textcoords="offset points",
                       xytext=(0, 16), ha='center',
                       fontsize=9.5, fontweight='bold',
                       color=COLORS['uav'],
                       bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='white', edgecolor=COLORS['uav'],
                                linewidth=0.8, alpha=0.92),
                       zorder=8)

        for i, (x, y) in enumerate(target_positions):
            ax.annotate(f'T{targets[i].id}', (x, y), textcoords="offset points",
                       xytext=(0, -20), ha='center',
                       fontsize=9.5, fontweight='bold',
                       color=COLORS['target'],
                       bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='white', edgecolor=COLORS['target'],
                                linewidth=0.8, alpha=0.92),
                       zorder=8)

        ax.set_xlabel('经度 (°)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax.set_ylabel('纬度 (°)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        title = 'UAV-目标分配方案'
        if scenario_name:
            title += f' · {scenario_name}'
        ax.set_title(title, fontsize=PLOT_STYLE['title_size'],
                     fontweight=PLOT_STYLE['title_weight'], pad=14)

        self._style_axes(ax)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right', fontsize=PLOT_STYLE['legend_size'],
                  fancybox=True, framealpha=0.95)

        n_assignments = len(assignment)
        total_cost = sum(cost_matrix[u, t] for u, t in assignment
                        if cost_matrix is not None and u < cost_matrix.shape[0] and t < cost_matrix.shape[1]) if cost_matrix is not None else 0

        stats_text = (f'UAV: {len(uavs)}  |  Target: {len(targets)}  |  '
                     f'分配数: {n_assignments}'
                     + (f'  |  总代价: {total_cost/1000:.1f} km' if total_cost > 0 else ''))
        ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
                fontsize=PLOT_STYLE['annotation_size'], ha='center', va='top',
                color=COLORS['text_secondary'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7',
                         edgecolor='#FCD34D', linewidth=0.8, alpha=0.92))

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
        crop_margin_deg: float = 0.025,
        smooth_sigma_px: float = 1.8,
        grid_size: int = 400,
    ) -> Path:
        """在二维 DEM 地形上可视化分配方案。

        使用等高线填充 + 山体阴影渲染作为底图，叠加 UAV 起飞点、
        目标点与分配连线，清晰展示 N=M 平衡指派的空间关系。

        Args:
            dem_terrain: DEMTerrain 实例。
            uavs: UAV 列表。
            targets: 目标列表。
            assignment: 分配方案 [(uav_id, target_id), ...]。
            scenario_name: 场景名称。
            cost_matrix: 代价矩阵(可选,显示代价值)。
            elev_exaggerate: （保留，2D 中用于山体阴影计算）。
            view_elev: （保留，参数兼容）。
            view_azim: （保留，参数兼容）。
            crop_margin_deg: 任务点范围外扩的裁剪边距(度)。
            smooth_sigma_px: 高斯平滑核(像素)。
            grid_size: 降采样后的目标网格尺寸。

        Returns:
            保存的文件路径。
        """
        fig, ax = plt.subplots(figsize=(13, 10))

        # ── 提取并裁剪 DEM ──
        elevation = dem_terrain.elevation.astype(float)
        bounds = dem_terrain.bounds
        left, bottom, right, top = bounds

        lons = [u.start_pos[0] for u in uavs] + [t.position[0] for t in targets]
        lats = [u.start_pos[1] for u in uavs] + [t.position[1] for t in targets]
        if lons and lats:
            c_left = max(left, min(lons) - crop_margin_deg)
            c_right = min(right, max(lons) + crop_margin_deg)
            c_bottom = max(bottom, min(lats) - crop_margin_deg)
            c_top = min(top, max(lats) + crop_margin_deg)
        else:
            c_left, c_right, c_bottom, c_top = left, right, bottom, top

        h, w = elevation.shape
        col0 = max(0, int((c_left - left) / (right - left) * (w - 1)))
        col1 = min(w - 1, int((c_right - left) / (right - left) * (w - 1)))
        row0 = max(0, int((top - c_top) / (top - bottom) * (h - 1)))
        row1 = min(h - 1, int((top - c_bottom) / (top - bottom) * (h - 1)))
        elev = elevation[row0:row1 + 1, col0:col1 + 1]

        elev = _fill_nan(elev)
        if smooth_sigma_px > 0:
            elev = _smooth2d(elev, smooth_sigma_px)
        elev = _block_reduce(elev, grid_size)

        rows, cols = elev.shape
        xs_deg = np.linspace(c_left, c_right, cols)
        ys_deg = np.linspace(c_bottom, c_top, rows)  # 下→上，与绘图方向一致
        xx_deg, yy_deg = np.meshgrid(xs_deg, ys_deg)

        # 反经纬度 → 米制平面（UTM-like）
        from utils.utils_dmde.coord_transform import WGS84Transformer
        ref_lon = (c_left + c_right) / 2
        ref_lat = (c_top + c_bottom) / 2
        transformer = WGS84Transformer.from_lonlat(ref_lon, ref_lat)
        xx_m, yy_m = transformer.to_xy_batch(xx_deg, yy_deg)

        # ── 山体阴影底图 ──
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        dx_m = float(np.abs(xx_m[0, 1] - xx_m[0, 0])) if cols > 1 else 1.0
        dy_m = float(np.abs(yy_m[1, 0] - yy_m[0, 0])) if rows > 1 else 1.0
        norm_elev = plt.Normalize(float(elev.min()), float(elev.max()))
        shaded = ls.shade(
            elev, cmap=plt.cm.terrain, norm=norm_elev,
            blend_mode='soft', vert_exag=elev_exaggerate,
            dx=dx_m, dy=dy_m,
        )
        ax.imshow(
            shaded,
            extent=(xx_m.min(), xx_m.max(), yy_m.min(), yy_m.max()),
            origin='lower',
            aspect='equal',
            zorder=1,
        )

        # ── 叠加等高线 ──
        try:
            n_levels = 14
            levels = np.linspace(float(elev.min()), float(elev.max()), n_levels)
            ax.contour(xx_m, yy_m, elev,
                      levels=levels[::2],
                      colors='#1F2937',
                      linewidths=0.35,
                      alpha=0.42,
                      zorder=2)
        except Exception:
            pass

        # ── 颜色条（高程）──
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(norm=norm_elev, cmap=plt.cm.terrain)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label('高程 (m)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        cbar.ax.tick_params(labelsize=PLOT_STYLE['tick_size'])
        cbar.outline.set_linewidth(0.7)

        # ── UAV / Target 坐标转换 ──
        uav_xy = []
        for u in uavs:
            x, y = transformer.to_xy(u.start_pos[0], u.start_pos[1])
            uav_xy.append((x, y))
        tgt_xy = []
        for t in targets:
            x, y = transformer.to_xy(t.position[0], t.position[1])
            tgt_xy.append((x, y))

        uav_color = COLORS['uav']
        target_color = COLORS['target']
        line_colors = [
            '#2563EB', '#DC2626', '#059669', '#D97706',
            '#7C3AED', '#DB2777', '#0891B2', '#4F46E5',
        ]

        # ── 绘制分配连线（带轻微弧线）──
        for idx, (uav_id, target_id) in enumerate(assignment):
            if uav_id < len(uavs) and target_id < len(targets):
                ux, uy = uav_xy[uav_id]
                tx, ty = tgt_xy[target_id]
                lc = line_colors[idx % len(line_colors)]

                # 轻微弧线（垂直于连线方向偏移一点）
                n_arc = 40
                arc_ts = np.linspace(0, 1, n_arc)
                arc_x = ux + (tx - ux) * arc_ts
                arc_y = uy + (ty - uy) * arc_ts

                dx_vec = tx - ux
                dy_vec = ty - uy
                len_vec = np.sqrt(dx_vec ** 2 + dy_vec ** 2) + 1e-9
                nx = -dy_vec / len_vec
                ny = dx_vec / len_vec
                lift = np.sin(arc_ts * np.pi) * len_vec * 0.06
                arc_x = arc_x + nx * lift
                arc_y = arc_y + ny * lift

                ax.plot(
                    arc_x, arc_y,
                    color='white', linewidth=4.6, alpha=0.7,
                    linestyle='-', zorder=5,
                )
                ax.plot(
                    arc_x, arc_y,
                    color=lc, linewidth=2.4, alpha=0.96,
                    linestyle='-', zorder=6,
                )

                # 箭头
                mid_t = 0.58
                ax.annotate(
                    '', xy=(tx, ty), xytext=(ux, uy),
                    arrowprops=dict(
                        arrowstyle='-|>', color=lc, lw=2.0,
                        shrinkA=13, shrinkB=13, alpha=0.96,
                    ),
                    zorder=7,
                )

                # 代价值
                if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                    mid_idx = n_arc // 2
                    cost_val = cost_matrix[uav_id, target_id]
                    ax.annotate(
                        f'{cost_val/1000:.1f} km',
                        xy=(arc_x[mid_idx], arc_y[mid_idx]),
                        xytext=(0, 11), textcoords='offset points',
                        fontsize=8.8, color=COLORS['text_primary'],
                        ha='center', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFBEB',
                                 edgecolor=lc, linewidth=0.9, alpha=0.97),
                        zorder=20,
                    )

        # ── 绘制 UAV 起飞点 ──
        ux_list = [p[0] for p in uav_xy]
        uy_list = [p[1] for p in uav_xy]
        ax.scatter(
            ux_list, uy_list,
            c=uav_color, s=380, marker='^',
            edgecolors='white', linewidth=2.6,
            zorder=30,
        )
        for i, (x, y) in enumerate(uav_xy):
            ax.annotate(
                f'U{uavs[i].id}',
                xy=(x, y), xytext=(0, 22), textcoords='offset points',
                fontsize=10.5, fontweight='bold', color=uav_color,
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                         edgecolor=uav_color, linewidth=1.1, alpha=0.98),
                zorder=40,
            )

        # ── 绘制 Target 目标点 ──
        tx_list = [p[0] for p in tgt_xy]
        ty_list = [p[1] for p in tgt_xy]
        ax.scatter(
            tx_list, ty_list,
            c=target_color, s=360, marker='o',
            edgecolors='white', linewidth=2.6,
            zorder=30,
        )
        for i, (x, y) in enumerate(tgt_xy):
            ax.annotate(
                f'T{targets[i].id}',
                xy=(x, y), xytext=(0, -26), textcoords='offset points',
                fontsize=10.5, fontweight='bold', color=target_color,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                         edgecolor=target_color, linewidth=1.1, alpha=0.98),
                zorder=40,
            )

        # ── 坐标轴与标题 ──
        ax.set_xlabel('东向 (m)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax.set_ylabel('北向 (m)', fontsize=PLOT_STYLE['label_size'], labelpad=8)
        ax.set_aspect('equal', adjustable='box')
        self._style_axes(ax)
        ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5, color='#9CA3AF')

        title = 'DEM 地形分配方案（二维）'
        if scenario_name:
            title += f' · {scenario_name}'
        ax.set_title(title, fontsize=PLOT_STYLE['title_size'] + 1,
                     fontweight=PLOT_STYLE['title_weight'], pad=14)

        # ── 统计信息 ──
        total_cost = 0
        for uav_id, target_id in assignment:
            if cost_matrix is not None and uav_id < cost_matrix.shape[0] and target_id < cost_matrix.shape[1]:
                total_cost += cost_matrix[uav_id, target_id]

        stats_text = (
            f'UAV: {len(uavs)}   |   Target: {len(targets)}   |   '
            f'分配: {len(assignment)}   |   总代价: {total_cost/1000:.1f} km'
        )
        ax.text(
            0.5, -0.08, stats_text,
            transform=ax.transAxes, fontsize=PLOT_STYLE['annotation_size'],
            ha='center', va='top', color=COLORS['text_secondary'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7',
                     edgecolor='#FCD34D', linewidth=0.8, alpha=0.95),
            zorder=50,
        )

        # ── 图例 ──
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor=uav_color,
                   markeredgecolor='white', markeredgewidth=1.6,
                   markersize=12, label='UAV 起飞点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=target_color,
                   markeredgecolor='white', markeredgewidth=1.6,
                   markersize=11, label='目标点'),
            Line2D([0], [0], color=line_colors[0], linewidth=2.4, label='分配路线'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  fontsize=PLOT_STYLE['legend_size'],
                  fancybox=True, framealpha=0.96,
                  edgecolor='#D1D5DB', borderpad=0.9)

        plt.tight_layout(pad=1.2)
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

        for sc in scenarios:
            filepath = self.plot_convergence(
                sc['results'],
                scenario_name=sc['name']
            )
            saved_files.append(filepath)
            print(f"  [OK] 收敛曲线已保存: {filepath.name}")

        for sc in scenarios:
            if 'cost_matrix' in sc:
                best_result = sc['results'][sc['metrics'].best_run_idx]
                filepath = self.plot_cost_matrix(
                    sc['cost_matrix'],
                    scenario_name=sc['name'],
                    highlight_assignment=best_result.best_assignment
                )
                saved_files.append(filepath)
                print(f"  [OK] 代价矩阵已保存: {filepath.name}")

        if len(scenarios) >= 2:
            filepath = self.plot_scenario_comparison(scenarios)
            saved_files.append(filepath)
            print(f"  [OK] 场景对比已保存: {filepath.name}")
        else:
            print("  [--] 场景对比图已跳过（仅 1 个场景，无需对比）")

        filepath = self.plot_metrics_boxplot(scenarios)
        saved_files.append(filepath)
        print(f"  [OK] 指标箱线图已保存: {filepath.name}")

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
                    print(f"  [OK] 分配方案已保存: {filepath.name}")

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
                    print(f"  [OK] DEM 三维图已保存: {filepath.name}")

        for table_file in self.export_statistics_tables(scenarios):
            print(f"  [OK] 统计表格已保存: {table_file.name}")

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
