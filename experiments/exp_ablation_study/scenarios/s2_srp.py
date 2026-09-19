# -*- coding: utf-8 -*-
"""S2 场景定义：srp N=10 M=20

高复杂度场景：目标数 > UAV 数，PopInit 需同时决定分配 + 巡回顺序。
"""

from pathlib import Path
from typing import Tuple

import numpy as np

# ── 场景元信息 ────────────────────────────────────────────
S2_CONFIG = {
    "name": "S2_srp",
    "model_type": "srp",
    "n_uavs": 15,
    "n_targets": 30,
    "label": r"srp ($N{=}15, M{=}30$)",
}

_DATA_DIR = Path(__file__).resolve().parent.parent / "S2_srp_N10_M20" / "data"
_DEM_FILE = _DATA_DIR / "chengguan_district_dem.tif"

# DEM 地理范围
_LON_RANGE = (91.02, 91.28)
_LAT_RANGE = (29.52, 29.73)
_ALT_RANGE_UAV = (3640, 3700)
_ALT_RANGE_TGT = (3680, 3800)

# 约束参数
_RADARS = [
    dict(x0=91.12, y0=29.66, z0=3700, radius=5000, penalty=15.0),
    dict(x0=91.22, y0=29.72, z0=4500, radius=6000, penalty=12.0),
    dict(x0=91.08, y0=29.58, z0=3900, radius=4500, penalty=18.0),
    dict(x0=91.18, y0=29.60, z0=4200, radius=5500, penalty=14.0),
]
_ESTIMATOR_PARAMS = dict(min_clearance=50, max_clearance=300, num_samples=100)


def _generate_positions(n: int, seed: int, alt_range: tuple) -> list[tuple]:
    rng = np.random.RandomState(seed)
    lons = rng.uniform(*_LON_RANGE, n)
    lats = rng.uniform(*_LAT_RANGE, n)
    alts = rng.uniform(*alt_range, n)
    return [(lons[i], lats[i], float(alts[i])) for i in range(n)]


def build_s2_cost_matrix_and_evaluator() -> Tuple[np.ndarray, object]:
    """构建 S2 场景的代价矩阵和适应度评估器。

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

    n_uavs = S2_CONFIG["n_uavs"]
    n_targets = S2_CONFIG["n_targets"]

    # 环境
    dem = DEMTerrain.from_file(_DEM_FILE)
    radar_field = RadarThreatField([RadarThreat(**r) for r in _RADARS])
    estimator = VerticalSectionCostEstimator(
        dem_terrain=dem, radar_field=radar_field, **_ESTIMATOR_PARAMS,
    )

    # UAV: 10 架
    uav_positions = _generate_positions(n_uavs, seed=100, alt_range=_ALT_RANGE_UAV)
    uavs = []
    for i in range(n_uavs):
        speed_lo = 0.15 + (i % 4) * 0.05
        uavs.append(UAV(
            id=i, start_pos=uav_positions[i],
            speed_range=(round(speed_lo, 2), round(speed_lo + 0.25, 2)),
            max_range=30000 + (i % 6) * 3000,
        ))

    # Target: 20 个
    tgt_positions = _generate_positions(n_targets, seed=200, alt_range=_ALT_RANGE_TGT)
    targets = []
    for i in range(n_targets):
        tw = (15000, 120000) if i % 3 == 0 else (25000, 180000) if i % 3 == 1 else None
        targets.append(Target(
            id=i, position=tgt_positions[i],
            weight=round(0.3 + (i % 7) * 0.1, 2),
            time_window=tw,
        ))

    # 代价矩阵 + 评估器
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)

    evaluator = FitnessEvaluator(
        uavs, targets, alpha=2.5, beta=1.5,
        enable_seq=True, enable_window=True, enable_sync=False,
    )

    return cm.matrix, evaluator