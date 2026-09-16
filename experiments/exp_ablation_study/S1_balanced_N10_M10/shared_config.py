# -*- coding: utf-8 -*-
"""S1 场景公共配置：balanced N=10 M=10

所有 A0-A3 配置共享相同的场景定义（UAV/Target/约束）。
"""

import numpy as np

# ── 问题规模 ──────────────────────────────────────────────
N_UAVS = 10
N_TARGETS = 10
MODEL_TYPE = "balanced"

# ── 代价值矩阵（示例，实际从 data/ 加载） ─────────────────
# C_UT: 10×10 UAV→Target 距离矩阵
# 实际运行时从 DEM 地形 + UAV/Target 坐标计算
COST_MATRIX = None  # run.py 中从 data/ 加载

# ── 约束 ──────────────────────────────────────────────────
MAX_RANGE = 8000.0       # 最大航程 (m)
TIME_WINDOWS = {         # 目标时间窗约束
    0: (0, 300),
    3: (100, 500),
    4: (0, 400),
}

# ── DMDE 参数 ─────────────────────────────────────────────
POP_SIZE = 50
MAX_GENERATIONS = 1000
ZETA = 3
DELTA = 0.3
SEEDS = list(range(42, 72))  # 30 runs