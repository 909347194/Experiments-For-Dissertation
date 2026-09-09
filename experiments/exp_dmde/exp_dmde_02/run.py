# -*- coding: utf-8 -*-
"""exp_dmde_02 — N>M 多对一实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 DMDE 算法对 N>M 多对一
    模型的求解能力，并输出统计指标。

约束配置：
    - 航程约束 (max_range): ✓
    - 时间窗约束 (time_window): ✓
    - 时序约束 (sequence_group): N/A（每 UAV 仅访问 1 个目标）
    - 同时到达约束 (sync): ✗ 移除（12U/4T 结构下物理上不可满足）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(SRC_ROOT))

from environments.environment_dmde import (
    DEMTerrain,
    RadarThreat,
    RadarThreatField,
    VerticalSectionCostEstimator,
)
from models.model_dmde import (
    UAV,
    Target,
    CostMatrixBuilder,
    FitnessEvaluator,
)
from algorithms.algorithm_dmde import DMDESolver, DMDEConfig
from utils.utils_dmde.metrics import compute_metrics, format_metrics

sys.path.insert(0, str(Path(__file__).parent))
# 复用 exp_dmde_01 的可视化和数据存储模块
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_01"))
from visualization.visualizer import ExperimentVisualizer
from data_store import DEFAULT_DATA_FILE, save_experiment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"

RADARS = [
    RadarThreat(x0=91.12, y0=29.66, z0=3700, radius=5000, penalty=15.0),
    RadarThreat(x0=91.22, y0=29.72, z0=4500, radius=6000, penalty=12.0),
]

ESTIMATOR_PARAMS = dict(
    min_clearance=50,
    max_clearance=300,
    num_samples=100,
)

SOLVER_PARAMS = dict(
    pop_size=80,
    max_generations=1000,
    zeta=3,
    delta=0.3,
)

N_RUNS = int(os.environ.get("EXP_N_RUNS", "5"))
VISUALIZE = True
FIGURES_DIR = RESULTS_DIR / "figures"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义：N>M 多对一（12 UAV → 4 Target）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario():
    """N>M 多对一场景。

    约束：航程 + 时间窗。
    移除 sync（12U/4T 结构下多 UAV 挤同一目标，同时到达不可满足）。
    """
    uavs = [
        UAV(id=0,  start_pos=(91.04, 29.55, 3650), speed_range=(0.20, 0.50), max_range=32000),
        UAV(id=1,  start_pos=(91.08, 29.53, 3640), speed_range=(0.25, 0.55), max_range=30000),
        UAV(id=2,  start_pos=(91.12, 29.55, 3660), speed_range=(0.30, 0.60), max_range=35000),
        UAV(id=3,  start_pos=(91.06, 29.58, 3670), speed_range=(0.20, 0.50), max_range=28000),
        UAV(id=4,  start_pos=(91.16, 29.54, 3650), speed_range=(0.25, 0.55), max_range=31000),
        UAV(id=5,  start_pos=(91.20, 29.56, 3660), speed_range=(0.30, 0.60), max_range=34000),
        UAV(id=6,  start_pos=(91.04, 29.60, 3680), speed_range=(0.20, 0.50), max_range=29000),
        UAV(id=7,  start_pos=(91.10, 29.57, 3660), speed_range=(0.25, 0.55), max_range=30000),
        UAV(id=8,  start_pos=(91.18, 29.52, 3650), speed_range=(0.30, 0.60), max_range=36000),
        UAV(id=9,  start_pos=(91.08, 29.60, 3690), speed_range=(0.20, 0.50), max_range=27000),
        UAV(id=10, start_pos=(91.14, 29.58, 3670), speed_range=(0.25, 0.55), max_range=32000),
        UAV(id=11, start_pos=(91.22, 29.55, 3660), speed_range=(0.30, 0.60), max_range=35000),
    ]
    targets = [
        Target(id=0, position=(91.12, 29.66, 3700), weight=1.0,
               time_window=(35000, 100000)),
        Target(id=1, position=(91.16, 29.70, 3750), weight=0.8,
               time_window=(30000, 95000)),
        Target(id=2, position=(91.08, 29.68, 3720), weight=0.9,
               time_window=(38000, 110000)),
        Target(id=3, position=(91.20, 29.65, 3700), weight=0.7,
               time_window=(32000, 98000)),
    ]
    return uavs, targets, 2.5, 1.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("exp_dmde_02: N>M 多对一实验")
    print("=" * 60)

    # 1. 加载环境
    print("\n[1] 加载环境...")
    t0 = time.time()
    dem = DEMTerrain.from_file(DEM_FILE)
    print(f"  DEM: {dem.meta.width}×{dem.meta.height}, "
          f"bounds={[f'{b:.3f}' for b in dem.bounds]}")

    radar_field = RadarThreatField(RADARS)
    estimator = VerticalSectionCostEstimator(
        dem_terrain=dem, radar_field=radar_field, **ESTIMATOR_PARAMS,
    )
    print(f"  环境加载耗时: {time.time()-t0:.2f}s")

    # 2. 构建场景
    uavs, targets, alpha, beta = make_scenario()
    n, m = len(uavs), len(targets)
    name = "N>M 多对一"

    print(f"\n{'='*60}")
    print(f"场景: {name} ({n}U/{m}T)")
    print(f"约束: 航程✓ 时间窗✓ 时序N/A 同步✗(移除)")
    print(f"{'='*60}")

    # 3. 构建代价矩阵
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}]")

    # 4. 创建评估器（overloaded: 禁用 sync 和 seq）
    evaluator = FitnessEvaluator(
        uavs, targets, alpha=alpha, beta=beta,
        enable_seq=False, enable_window=True, enable_sync=False,
    )

    # 5. 多次运行
    results = []
    for run_idx in range(N_RUNS):
        cfg = DMDEConfig(
            pop_size=SOLVER_PARAMS["pop_size"],
            max_generations=SOLVER_PARAMS["max_generations"],
            zeta=SOLVER_PARAMS["zeta"],
            delta=SOLVER_PARAMS["delta"],
            seed=run_idx,
            verbose=False,
        )
        solver = DMDESolver(cfg)
        result = solver.solve(cm.matrix, n, m, fitness_evaluator=evaluator)
        results.append(result)

        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix, n_uavs=n)
        result.extra["total_violation"] = (
            eval_res.range_violation + eval_res.time_violation
            + eval_res.seq_violation + eval_res.sync_violation
        )
        result.extra["is_feasible"] = eval_res.is_feasible

        print(f"  Run {run_idx}: fitness={result.best_fitness:.1f}, "
              f"feasible={eval_res.is_feasible}, "
              f"time={result.elapsed_seconds:.2f}s")

    # 6. 统计
    metrics = compute_metrics(results)
    print(f"\n{format_metrics(metrics, name)}")

    scenario = {
        "name": name, "model_type": "overloaded",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
    }

    # 7. 保存数据
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={"experiment": "exp_dmde_02", "description": "N>M 多对一",
              "solver_params": SOLVER_PARAMS, "n_runs": N_RUNS,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        path=RESULTS_DIR / DEFAULT_DATA_FILE,
    )
    print(f"\n数据已保存: {data_file}")

    # 8. 可视化
    if VISUALIZE:
        print(f"\n[8] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR)
        saved = viz.plot_all([scenario], {name: uavs}, {name: targets}, dem_terrain=dem)
        print(f"  共生成 {len(saved)} 张图表 → {FIGURES_DIR}")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()
