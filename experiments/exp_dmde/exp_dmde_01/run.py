# -*- coding: utf-8 -*-
"""exp_dmde_01 — N=M 平衡指派实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 DMDE 算法对 N=M 平衡指派
    模型的求解能力，并输出统计指标。

约束配置：
    - 航程约束 (max_range): ✓
    - 时间窗约束 (time_window): ✓
    - 时序约束 (sequence_group): ✓
    - 同时到达约束 (sync): N/A（每目标仅 1 架 UAV）
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(SRC_ROOT))

# ── 导入项目模块 ──────────────────────────────────────────────
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
# 场景定义：N=M 平衡指派（10 UAV → 10 Target）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario():
    """N=M 平衡指派场景。

    约束：航程 + 时间窗 + 时序（无 sync，因每目标仅 1 架 UAV）。
    """
    uavs = [
        UAV(id=0,  start_pos=(91.05, 29.55, 3650), speed_range=(0.20, 0.50), max_range=32000),
        UAV(id=1,  start_pos=(91.10, 29.53, 3640), speed_range=(0.25, 0.55), max_range=30000),
        UAV(id=2,  start_pos=(91.15, 29.56, 3660), speed_range=(0.30, 0.60), max_range=29000),
        UAV(id=3,  start_pos=(91.03, 29.60, 3680), speed_range=(0.20, 0.50), max_range=28000),
        UAV(id=4,  start_pos=(91.08, 29.58, 3670), speed_range=(0.25, 0.55), max_range=31000),
        UAV(id=5,  start_pos=(91.18, 29.54, 3650), speed_range=(0.30, 0.60), max_range=33000),
        UAV(id=6,  start_pos=(91.12, 29.60, 3690), speed_range=(0.20, 0.50), max_range=28000),
        UAV(id=7,  start_pos=(91.06, 29.57, 3660), speed_range=(0.25, 0.55), max_range=30000),
        UAV(id=8,  start_pos=(91.20, 29.55, 3670), speed_range=(0.30, 0.60), max_range=35000),
        UAV(id=9,  start_pos=(91.14, 29.52, 3640), speed_range=(0.20, 0.50), max_range=38000),
    ]
    targets = [
        Target(id=0, position=(91.12, 29.66, 3700), weight=1.0,
               time_window=(40000, 120000)),
        Target(id=1, position=(91.10, 29.70, 3750), weight=0.8,
               sequence_group=1),
        Target(id=2, position=(91.16, 29.65, 3680), weight=0.9,
               sequence_group=1),
        Target(id=3, position=(91.20, 29.72, 3800), weight=0.7,
               time_window=(35000, 100000)),
        Target(id=4, position=(91.08, 29.68, 3720), weight=0.6,
               time_window=(30000, 90000)),
        Target(id=5, position=(91.22, 29.60, 3700), weight=0.85,
               time_window=(38000, 110000)),
        Target(id=6, position=(91.15, 29.75, 3900), weight=0.75),
        Target(id=7, position=(91.06, 29.63, 3690), weight=0.65,
               time_window=(30000, 95000)),
        Target(id=8, position=(91.18, 29.68, 3750), weight=0.95,
               sequence_group=2),
        Target(id=9, position=(91.25, 29.65, 3800), weight=0.7,
               sequence_group=2),
    ]
    return uavs, targets, 2.5, 1.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("exp_dmde_01: N=M 平衡指派实验")
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
    name = "N=M 平衡指派"

    print(f"\n{'='*60}")
    print(f"场景: {name} ({n}U/{m}T)")
    print(f"约束: 航程✓ 时间窗✓ 时序✓ 同步N/A")
    print(f"{'='*60}")

    # 3. 构建代价矩阵
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}]")

    # 4. 创建评估器（balanced: 启用所有约束，sync 自然不触发）
    evaluator = FitnessEvaluator(
        uavs, targets, alpha=alpha, beta=beta,
        enable_seq=True, enable_window=True, enable_sync=True,
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
        "name": name, "model_type": "balanced",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
    }

    # 7. 保存数据
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={"experiment": "exp_dmde_01", "description": "N=M 平衡指派",
              "solver_params": SOLVER_PARAMS, "n_runs": N_RUNS,
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        path=RESULTS_DIR / "exp_dmde_01_data.json",
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
