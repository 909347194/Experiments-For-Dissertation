# -*- coding: utf-8 -*-
"""scenario.py — 场景构建：从 DEM 数据构建代价矩阵和评估器"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _ensure_src_on_path(project_root: Path):
    """确保 src/ 在 sys.path 中。"""
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def build_scenario(scenario_dir: Path) -> Tuple[np.ndarray, object, int, int, str]:
    """从场景目录的 shared_config 加载场景，构建代价矩阵和评估器。

    Args:
        scenario_dir: 场景目录路径（如 S1_balanced_N10_M10/）

    Returns:
        (cost_matrix, evaluator, n_uavs, n_targets, model_type)
    """
    project_root = scenario_dir.parents[2]
    _ensure_src_on_path(project_root)

    # 动态导入该场景的 shared_config
    if str(scenario_dir) not in sys.path:
        sys.path.insert(0, str(scenario_dir))

    from shared_config import N_UAVS, N_TARGETS, MODEL_TYPE, build_cost_matrix_and_evaluator

    cost_matrix, evaluator = build_cost_matrix_and_evaluator()
    return cost_matrix, evaluator, N_UAVS, N_TARGETS, MODEL_TYPE