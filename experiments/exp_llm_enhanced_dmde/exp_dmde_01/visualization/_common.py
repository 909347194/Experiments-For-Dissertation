# -*- coding: utf-8 -*-
"""_common.py — LLM 实验可视化公共工具。"""

import re
from pathlib import Path

# 统一配色
COLORS = {
    "dmde": "#2196F3",       # 蓝色 — DMDE 基线
    "llm_dmde": "#FF5722",   # 橙红 — LLM-DMDE
    "best": "#4CAF50",       # 绿色 — 最优
    "worst": "#F44336",      # 红色 — 最差
    "mean": "#FF9800",       # 橙色 — 均值
    "feasible": "#8BC34A",   # 浅绿 — 可行
    "infeasible": "#E91E63", # 粉红 — 不可行
    "rand/1": "#9C27B0",     # 紫色 — rand/1 策略
    "best/2": "#00BCD4",     # 青色 — best/2 策略
    "default": "#607D8B",    # 灰色 — 默认
}


def _sanitize_filename(name: str) -> str:
    """将场景名转为安全文件名。"""
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def setup_chinese_font():
    """设置中文字体（如果可用）。"""
    import matplotlib
    import matplotlib.pyplot as plt

    for font in ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue
