# -*- coding: utf-8 -*-
"""S1 场景定义：balanced N=10 M=10

小规模基准（N=M 一一对应）。
"""

from pathlib import Path
from typing import Tuple

import numpy as np

# ── 场景元信息 ────────────────────────────────────────────
S1_CONFIG = {
    "name": "S1_balanced",
    "model_type": "balanced",
    "n_uavs": 10,
    "n_targets": 10,
    "label": r"balanced ($N{=}M{=}10$)",
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "S1_balanced_N10_M10" / "data"
_DEM_FILE = _DATA_DIR / "chengguan_district_dem.tif"

# DEM 地理范围（拉萨城关区）
_LON_RANGE = (91.02, 91.28)
_LAT_RANGE = (29.52, 29.73)
_ALT_RANGE_UAV = (3640, 3700)
_ALT_RANGE_TGT = (3680, 3800)

# 约束参数
_RADARS = [
    dict(x0=91.12, y0=29.66, z0=3700, radius=5000, penalty=15.0),
    dict(x0=91.22, y0=29.72, z0=4500, radius=6000, penalty=12.0),
]
_ESTIMATOR_PARAMS = dict(min_clearance=50, max_clearance=300, num_samples=100)


def _generate_positions(n: int, seed: int, alt_range: tuple) -> list[tuple]:
    rng = np.random.RandomState(seed)
    lons = rng.uniform(*_LON_RANGE, n)
    lats = rng.uniform(*_LAT_RANGE, n)
    alts = rng.uniform(*alt_range, n)
    return [(lons[i], lats[i], float(alts[i])) for i in range(n)]


def build_s1_cost_matrix_and_evaluator() -> Tuple[np.ndarray, object]:
    """构建 S1 场景的代价矩阵和适应度评估器。

    Returns:
        (cost_matrix_np, evaluator)
    """
    import sys
    project_root = Path(__file__).resolve().parents[3]
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from environments.environment_dmde import (
        DEMTerrain, RadarThreat, RadarThreatField, VerticalSectionCostEstimator,
    )
    from models.model_dmde import UAV, Target, CostMatrixBuilder, FitnessEvaluator

    n = S1_CONFIG["n_uavs"]

    # 环境
    dem = DEMTerrain.from_file(_DEM_FILE)
    radar_field = RadarThreatField([RadarThreat(**r) for r in _RADARS])
    estimator = VerticalSectionCostEstimator(
        dem_terrain=dem, radar_field=radar_field, **_ESTIMATOR_PARAMS,
    )

    # UAV
    uav_positions = _generate_positions(n, seed=100, alt_range=_ALT_RANGE_UAV)
    uavs = []
    for i in range(n):
        speed_lo = 0.20 + (i % 3) * 0.05
        uavs.append(UAV(
            id=i, start_pos=uav_positions[i],
            speed_range=(round(speed_lo, 2), round(speed_lo + 0.30, 2)),
            max_range=28000 + (i % 5) * 1000,
        ))

    # Target
    tgt_positions = _generate_positions(n, seed=200, alt_range=_ALT_RANGE_TGT)
    targets = []
    for i in range(n):
        tw = (20000, 150000) if i % 3 == 0 else None
        targets.append(Target(
            id=i, position=tgt_positions[i],
            weight=round(0.6 + (i % 5) * 0.1, 2),
            time_window=tw,
        ))

    # 代价矩阵 + 评估器
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)

    evaluator = FitnessEvaluator(
        uavs, targets, alpha=2.5, beta=1.5,
        enable_seq=False, enable_window=True, enable_sync=False,
    )

    return cm.matrix, evaluator