# -*- coding: utf-8 -*-
"""exp_dmde_02 — N>M 多对一实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 DMDE 算法对 N>M 多对一
    模型的求解能力，并输出统计指标。

约束配置（通过 CONSTRAINT_CONFIG 选配）：
    - 航程约束 (max_range):      每 UAV 最大飞行航程
    - 最大飞行时间 (max_time):    每 UAV 最大飞行时间
    - 时间窗约束 (time_window):   目标的可执行时间窗口
    - 时序约束 (sequence_group):  目标间的先后执行顺序
    - 同时到达约束 (sync):        多 UAV 攻击同一目标时同时到达

用法：
    # 默认配置（航程+时间窗）
    python run.py

    # 启用全部约束
    EXP_CONSTRAINTS=all python run.py

    # 仅启用航程约束
    EXP_CONSTRAINTS=range python run.py

    # 运行次数
    EXP_N_RUNS=3 python run.py
"""

from __future__ import annotations

import json
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
# 约束选配
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_constraint_config() -> dict[str, bool]:
    """从环境变量解析约束配置。

    EXP_CONSTRAINTS 支持：
        all       — 全部启用
        none      — 全部禁用（仅航程）
        range     — 仅航程
        range,time — 航程+时间窗
        range,time,seq — 航程+时间窗+时序
        range,time,sync — 航程+时间窗+同时到达
        range,time,seq,sync — 全部启用

    默认: range,time（航程+时间窗）
    """
    raw = os.environ.get("EXP_CONSTRAINTS", "range,time").lower().strip()

    if raw == "all":
        return dict(range=True, time=True, seq=True, sync=True)
    if raw == "none":
        return dict(range=True, time=False, seq=False, sync=False)

    parts = {p.strip() for p in raw.split(",")}
    return dict(
        range="range" in parts,
        time="time" in parts,
        seq="seq" in parts,
        sync="sync" in parts,
    )


def describe_constraints(cfg: dict[str, bool]) -> str:
    """格式化约束配置描述。"""
    parts = []
    parts.append(f"航程{'✓' if cfg['range'] else '✗'}")
    parts.append(f"时间窗{'✓' if cfg['time'] else '✗'}")
    parts.append(f"时序{'✓' if cfg['seq'] else '✗'}")
    parts.append(f"同时到达{'✓' if cfg['sync'] else '✗'}")
    return " ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义：N>M 多对一（12 UAV → 4 Target）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario(cc: dict[str, bool]):
    """N>M 多对一场景，根据约束配置动态调整 UAV/Target 属性。"""

    # 基础 UAV 配置
    uav_base = [
        dict(id=0,  start_pos=(91.04, 29.55, 3650), speed_range=(0.20, 0.50), max_range=32000),
        dict(id=1,  start_pos=(91.08, 29.53, 3640), speed_range=(0.25, 0.55), max_range=30000),
        dict(id=2,  start_pos=(91.12, 29.55, 3660), speed_range=(0.30, 0.60), max_range=35000),
        dict(id=3,  start_pos=(91.06, 29.58, 3670), speed_range=(0.20, 0.50), max_range=28000),
        dict(id=4,  start_pos=(91.16, 29.54, 3650), speed_range=(0.25, 0.55), max_range=31000),
        dict(id=5,  start_pos=(91.20, 29.56, 3660), speed_range=(0.30, 0.60), max_range=34000),
        dict(id=6,  start_pos=(91.04, 29.60, 3680), speed_range=(0.20, 0.50), max_range=29000),
        dict(id=7,  start_pos=(91.10, 29.57, 3660), speed_range=(0.25, 0.55), max_range=30000),
        dict(id=8,  start_pos=(91.18, 29.52, 3650), speed_range=(0.30, 0.60), max_range=36000),
        dict(id=9,  start_pos=(91.08, 29.60, 3690), speed_range=(0.20, 0.50), max_range=27000),
        dict(id=10, start_pos=(91.14, 29.58, 3670), speed_range=(0.25, 0.55), max_range=32000),
        dict(id=11, start_pos=(91.22, 29.55, 3660), speed_range=(0.30, 0.60), max_range=35000),
    ]

    # 启用 max_time 约束时，添加最大飞行时间
    # 余量系数 2.0：保证 UAV 有能力在时间窗内完成任务
    if cc["time"]:
        for u in uav_base:
            avg_speed = (u["speed_range"][0] + u["speed_range"][1]) / 2
            u["max_time"] = u["max_range"] / avg_speed * 2.0

    uavs = [UAV(**u) for u in uav_base]

    # 基础 Target 配置
    tgt_base = [
        dict(id=0, position=(91.12, 29.66, 3700), weight=1.0),
        dict(id=1, position=(91.16, 29.70, 3750), weight=0.8),
        dict(id=2, position=(91.08, 29.68, 3720), weight=0.9),
        dict(id=3, position=(91.20, 29.65, 3700), weight=0.7),
    ]

    # 启用时间窗约束
    if cc["time"]:
        tgt_base[0]["time_window"] = (35000, 100000)
        tgt_base[1]["time_window"] = (30000, 95000)
        tgt_base[2]["time_window"] = (38000, 110000)
        tgt_base[3]["time_window"] = (32000, 98000)

    # 启用时序约束：Target 0 必须在 Target 3 之前执行
    if cc["seq"]:
        tgt_base[0]["sequence_group"] = 1
        tgt_base[3]["sequence_group"] = 1  # 同组，id 小的先执行

    targets = [Target(**t) for t in tgt_base]

    alpha, beta = 2.5, 1.5
    return uavs, targets, alpha, beta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    cc = parse_constraint_config()

    print("=" * 60)
    print("exp_dmde_02: N>M 多对一实验")
    print("=" * 60)
    print(f"  约束配置: {describe_constraints(cc)}")
    print(f"  运行次数: {N_RUNS}")
    print(f"  配置来源: EXP_CONSTRAINTS={os.environ.get('EXP_CONSTRAINTS', 'range,time')}")

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
    uavs, targets, alpha, beta = make_scenario(cc)
    n, m = len(uavs), len(targets)
    name = "N>M 多对一"

    print(f"\n{'='*60}")
    print(f"场景: {name} ({n}U/{m}T)")
    print(f"  UAV max_range: {min(u.max_range for u in uavs):.0f} ~ {max(u.max_range for u in uavs):.0f}")
    if cc["time"]:
        print(f"  UAV max_time:  {min(u.max_time for u in uavs if u.max_time):.0f} ~ {max(u.max_time for u in uavs if u.max_time):.0f}")
    for t in targets:
        extras = []
        if t.time_window:
            extras.append(f"tw={t.time_window}")
        if t.sequence_group is not None:
            extras.append(f"seq_grp={t.sequence_group}")
        print(f"  Target {t.id}: weight={t.weight} {' '.join(extras)}")
    print(f"{'='*60}")

    # 3. 构建代价矩阵
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}]")

    # 4. 创建评估器
    evaluator = FitnessEvaluator(
        uavs, targets, alpha=alpha, beta=beta,
        enable_seq=cc["seq"],
        enable_window=cc["time"],
        enable_sync=cc["sync"],
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
              f"range_vio={eval_res.range_violation:.1f}, "
              f"time_vio={eval_res.time_violation:.1f}, "
              f"seq_vio={eval_res.seq_violation:.1f}, "
              f"sync_vio={eval_res.sync_violation:.1f}, "
              f"time={result.elapsed_seconds:.2f}s")

    # 6. 统计
    metrics = compute_metrics(results)
    print(f"\n{format_metrics(metrics, name)}")

    # 验证目标覆盖
    best = results[metrics.best_run_idx]
    assigned_tgts = set(a[1] for a in best.best_assignment)
    print(f"  目标覆盖: {sorted(assigned_tgts)} / {list(range(m))}")
    if assigned_tgts != set(range(m)):
        print("  ⚠️  存在未覆盖目标！")

    scenario = {
        "name": name, "model_type": "overloaded",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
        "constraint_config": cc,
    }

    # 7. 保存数据
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={"experiment": "exp_dmde_02", "description": "N>M 多对一",
              "solver_params": SOLVER_PARAMS, "n_runs": N_RUNS,
              "constraint_config": cc,
              "constraint_desc": describe_constraints(cc),
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        path=RESULTS_DIR / "exp_dmde_02_data.json",
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
