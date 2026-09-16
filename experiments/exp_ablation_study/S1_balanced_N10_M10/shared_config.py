# -*- coding: utf-8 -*-
"""S1 场景公共配置（薄包装层）。

实际定义在 scenarios/s1_balanced.py。
此文件保留向后兼容性，供旧 run.py 导入。
"""

import sys
from pathlib import Path

# 确保项目根目录和 exp_ablation_study/ 在路径中
SCENARIO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCENARIO_DIR.parents[2]
ABLAITION_DIR = SCENARIO_DIR.parent

for p in [str(PROJECT_ROOT), str(ABLAITION_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scenarios.s1_balanced import S1_CONFIG, build_s1_cost_matrix_and_evaluator

# 向后兼容导出
N_UAVS = S1_CONFIG["n_uavs"]
N_TARGETS = S1_CONFIG["n_targets"]
MODEL_TYPE = S1_CONFIG["model_type"]


def build_cost_matrix_and_evaluator():
    return build_s1_cost_matrix_and_evaluator()


# DMDE 参数
POP_SIZE = 50
MAX_GENERATIONS = 1000
ZETA = 3
DELTA = 0.3
SEEDS = list(range(42, 72))  # 30 runs