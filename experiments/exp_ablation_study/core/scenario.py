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


def _generate_positions(n: int, seed: int = 42, *,
                        lon_range=(91.02, 91.28),
                        lat_range=(29.52, 29.73),
                        alt_range=(3640, 3700)) -> list[tuple]:
    """在 DEM 范围内生成 n 个均匀分布的位置。"""
    rng = np.random.RandomState(seed)
    lons = rng.uniform(*lon_range, n)
    lats = rng.uniform(*lat_range, n)
    alts = rng.uniform(*alt_range, n)
    return [(lons[i], lats[i], float(alts[i])) for i in range(n)]


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


def make_balanced_scenario(n: int = 10):
    """构建 balanced (N=M) 场景的 UAV/Target 列表。

    Returns: (uavs, targets, alpha, beta)
    """
    from environments.environment_dmde import UAV, Target

    uav_positions = _generate_positions(n, seed=100)
    uavs = []
    for i in range(n):
        speed_lo = 0.20 + (i % 3) * 0.05
        speed_hi = speed_lo + 0.30
        max_r = 28000 + (i % 5) * 1000
        uavs.append(UAV(
            id=i, start_pos=uav_positions[i],
            speed_range=(round(speed_lo, 2), round(speed_hi, 2)),
            max_range=max_r,
        ))

    tgt_positions = _generate_positions(n, seed=200, alt_range=(3680, 3800))
    targets = []
    for i in range(n):
        tw = (20000, 150000) if i % 3 == 0 else None
        targets.append(Target(
            id=i, position=tgt_positions[i],
            weight=round(0.6 + (i % 5) * 0.1, 2),
            time_window=tw,
        ))

    return uavs, targets, 2.5, 1.5


def make_srp_scenario(n_uavs: int = 10, n_targets: int = 20):
    """构建 srp (N<M) 场景的 UAV/Target 列表。

    Returns: (uavs, targets, alpha, beta)
    """
    from environments.environment_dmde import UAV, Target

    uav_positions = _generate_positions(n_uavs, seed=100)
    uavs = []
    for i in range(n_uavs):
        speed_lo = 0.20 + (i % 3) * 0.05
        speed_hi = speed_lo + 0.30
        max_r = 40000 + (i % 5) * 2000
        uavs.append(UAV(
            id=i, start_pos=uav_positions[i],
            speed_range=(round(speed_lo, 2), round(speed_hi, 2)),
            max_range=max_r,
        ))

    tgt_positions = _generate_positions(n_targets, seed=200, alt_range=(3680, 3800))
    targets = []
    for i in range(n_targets):
        tw = (20000, 150000) if i % 4 == 0 else None
        targets.append(Target(
            id=i, position=tgt_positions[i],
            weight=round(0.5 + (i % 5) * 0.1, 2),
            time_window=tw,
        ))

    return uavs, targets, 2.5, 1.5