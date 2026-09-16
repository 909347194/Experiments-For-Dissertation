# -*- coding: utf-8 -*-
"""S2 场景公共配置：srp N=10 M=20

所有 A0-A3 配置共享相同的场景定义（UAV/Target/约束）。
"""

import numpy as np

# ── 问题规模 ──────────────────────────────────────────────
N_UAVS = 10
N_TARGETS = 20
MODEL_TYPE = "srp"

# ── 代价值矩阵 ────────────────────────────────────────────
# C_UT: 10×20 UAV→Target 距离矩阵
# C_TT: 20×20 Target→Target 转移矩阵（嵌入 cost_matrix 下半部分）
COST_MATRIX = None  # run.py 中从 data/ 加载

# ── 约束 ──────────────────────────────────────────────────
MAX_RANGE = 10000.0      # 最大航程 (m)
MAX_TARGETS_PER_UAV = 5  # 每架 UAV 最多访问目标数

# ── DMDE 参数 ─────────────────────────────────────────────
POP_SIZE = 50
MAX_GENERATIONS = 1000
ZETA = 3
DELTA = 0.3
SEEDS = list(range(42, 72))  # 30 runs