# -*- coding: utf-8 -*-
"""_common.py — 可视化共享配置与工具。

集中定义中文字体、配色方案（COLORS）与绘图基类 _PlotBase，
供同目录下的 plot_*.py 子模块复用。
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── 中文字体配置 ──────────────────────────────────────────────

def _setup_chinese_font():
    """配置 matplotlib 中文字体支持。"""
    import matplotlib
    import os, glob

    cache_dir = matplotlib.get_cachedir()
    if cache_dir and os.path.isdir(cache_dir):
        for f in glob.glob(os.path.join(cache_dir, 'fontlist-*.json')):
            try:
                os.remove(f)
            except OSError:
                pass

    from matplotlib import font_manager
    font_manager._load_fontmanager(try_read_cache=False)

    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi',
        'Noto Serif CJK SC', 'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 'Source Han Sans SC', 'AR PL UMing CN',
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  [visualizer] 使用中文字体: {font}")
            return

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
    warnings.warn("未找到中文字体,图表中文可能显示为方块。")

_setup_chinese_font()


# ── 配色方案 ──────────────────────────────────────────────────

COLORS = {
    'balanced': '#2E86AB',
    'overloaded': '#A23B72',
    'srp': '#F18F01',
    'best': '#C73E1D',
    'mean': '#3B1F2B',
    'grid': '#E8E8E8',
    'background': '#FAFAFA',
    # LLM 决策图表专用（plot_llm_decisions.py 引用）
    'llm_dmde': '#FF6B6B',
    'default': '#95A5A6',
}


# ── 工具函数 ──────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    """将场景名转为安全的文件名。"""
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = safe.replace(' ', '_')
    return safe


# ── 基类 ──────────────────────────────────────────────────────

class _PlotBase:
    """图表基类，提供保存功能。"""

    def __init__(self, output_dir: Path, dpi: int = 300, figsize: tuple[int, int] = (10, 6)):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        return filepath
