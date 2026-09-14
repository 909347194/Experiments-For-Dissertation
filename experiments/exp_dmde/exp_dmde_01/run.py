# -*- coding: utf-8 -*-
"""exp_dmde_01 — N=M 平衡指派实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 DMDE 算法对 N=M 平衡指派
    模型的求解能力，并输出统计指标。

N=M 场景约束（单 UAV → 单 Target 一一对应）：
    - 航程约束 (max_range):         ✓ 生效
    - 最大飞行时间 (max_time):        — 本实验未启用（UAV 未设 max_time）
    - 时间窗约束 (time_window):       ✓ 生效（T0/T3/T4 带窗）
    - 时序约束 (sequence_group):      ✗ N/M（单 UAV 仅 1 目标，无需排序）
    - 同时到达约束 (sync):            ✗ N/M（单目标仅 1 UAV，无协同）
"""

from __future__ import annotations
import numpy as np

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

# Windows 下控制台/重定向管道默认使用 GBK，print("✓") 等字符会抛
# UnicodeEncodeError 导致脚本中途退出。这里仅放宽编码错误处理
# （不改变原编码），保证脚本在管道中也能正常跑完。
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(errors="replace")

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

N_RUNS = int(os.environ.get("EXP_N_RUNS", "2"))
VISUALIZE = True
FIGURES_DIR = RESULTS_DIR / "figures"

# ── 多规模支持 ────────────────────────────────────────────────
SCALE = os.environ.get("EXP_SCALE", "medium").lower()
SCALES = {"small": 5, "medium": 10, "large": 20}
N_ENTITIES = SCALES.get(SCALE, 10)

# 支持自定义 solver 参数（大规模需要更大种群）
_solver_overrides = os.environ.get("EXP_SOLVER_PARAMS", "")
if _solver_overrides:
    import json as _json
    try:
        _overrides = _json.loads(_solver_overrides)
        SOLVER_PARAMS.update(_overrides)
    except _json.JSONDecodeError:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义：N=M 平衡指派
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DEM 地理范围（拉萨城关区）
_LON_MIN, _LON_MAX = 91.00, 91.30
_LAT_MIN, _LAT_MAX = 29.50, 29.75
_ALT_MIN, _ALT_MAX = 3640, 3700


def _generate_positions(n: int, seed: int = 42, *,
                        lon_range: tuple[float, float] = (_LON_MIN + 0.02, _LON_MAX - 0.02),
                        lat_range: tuple[float, float] = (_LAT_MIN + 0.02, _LAT_MAX - 0.02),
                        alt_range: tuple[float, float] = (_ALT_MIN, _ALT_MAX),
                        ) -> list[tuple[float, float, float]]:
    """在 DEM 范围内生成 n 个均匀分布的位置。"""
    rng = np.random.RandomState(seed)
    lons = rng.uniform(*lon_range, n)
    lats = rng.uniform(*lat_range, n)
    alts = rng.uniform(*alt_range, n)
    return [(lons[i], lats[i], float(alts[i])) for i in range(n)]


def make_scenario():
    """N=M 平衡指派场景，支持 small/medium/large 规模。

    规模映射（由 EXP_SCALE 环境变量控制）：
        small:  5U / 5T
        medium: 10U / 10T（默认）
        large:  20U / 20T

    约束：仅启用航程 + 时间窗（N=M 场景时序/sync 不触发，直接关闭减少开销）。
    """
    n = N_ENTITIES

    # 生成 UAV 位置
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

    # 生成 Target 位置
    tgt_positions = _generate_positions(n, seed=200,
                                        alt_range=(_ALT_MIN + 40, _ALT_MAX + 100))
    targets = []
    for i in range(n):
        tw = (20000, 150000) if i % 3 == 0 else None
        targets.append(Target(
            id=i, position=tgt_positions[i],
            weight=round(0.6 + (i % 5) * 0.1, 2),
            time_window=tw,
        ))

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
    print(f"场景: {name} ({n}U/{m}T) [规模: {SCALE}]")
    print(f"约束: 航程✓ 时间窗✓ 时序✗ 同步✗（N=M 精简配置）")
    print(f"{'='*60}")

    # 3. 构建代价矩阵
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}]")

    # 4. 创建评估器（N=M 平衡指派：关闭 seq/sync，减少无效计算）
    evaluator = FitnessEvaluator(
        uavs, targets, alpha=alpha, beta=beta,
        enable_seq=False, enable_window=True, enable_sync=False,
    )

    # 5. 多次运行
    results = []
    run_start_time = time.time()
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
        result.extra["total_violation"] = eval_res.total_violation
        result.extra["violation_breakdown"] = eval_res.violation_breakdown()
        result.extra["is_feasible"] = eval_res.is_feasible

        print(f"  Run {run_idx}: fitness={result.best_fitness:.1f}, "
              f"feasible={eval_res.is_feasible}, "
              f"time={result.elapsed_seconds:.2f}s")
        if not eval_res.is_feasible:
            print(f"    违反: {eval_res.total_violation:.2f}")
            for k, v in eval_res.violation_breakdown().items():
                if v > 0:
                    print(f"      {k}: {v}")
        print(f"    分配: {result.best_assignment}")

    # 6. 统计
    total_time = time.time() - run_start_time
    metrics = compute_metrics(results)
    print(f"\n{'='*60}")
    print(f"[实验结果汇总]")
    print(f"{'='*60}")
    print(format_metrics(metrics, name))
    feasible_count = sum(1 for r in results if r.extra.get('is_feasible', False))
    print(f"  可行解率: {feasible_count}/{N_RUNS} ({100*feasible_count/N_RUNS:.0f}%)")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

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

    # 7.5 注册到索引
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01"))
        from registry import register
        register(
            experiment_id="dmde_01",
            label="DMDE 基线 N=M=10",
            result_path="exp_dmde/exp_dmde_01/results/exp_dmde_01_data.json",
            tags=["baseline", "dmde", "balanced", "10u10t"],
            meta={"n_runs": N_RUNS, "solver_params": SOLVER_PARAMS},
        )
    except Exception:
        pass

    # 8. 可视化
    if VISUALIZE:
        print(f"\n[8] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR)
        saved = viz.plot_all([scenario], {name: uavs}, {name: targets}, dem_terrain=dem)
        print(f"  共生成 {len(saved)} 张图表 → {FIGURES_DIR}")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()
