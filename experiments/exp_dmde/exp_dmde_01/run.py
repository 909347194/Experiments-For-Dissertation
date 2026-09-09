# -*- coding: utf-8 -*-
"""exp_dmde_01 — DMDE 算法验证实验

实验目的：
    在拉萨地区真实 DEM 地形上，验证 DMDE 算法对三种分配模型
    （N=M / N>M / N<M）的求解能力，并输出统计指标。

实验流程：
    1. 加载 DEM 地形 + 行政区 GeoJSON。
    2. 构造 UAV / Target 配置（三种模型）。
    3. 环境层：航程代价估算 → 代价矩阵。
    4. 算法层：DMDE 求解。
    5. 统计层：多次运行 → 指标汇总。
    6. 输出：结果保存到 results/ 目录。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Experiments-For-Dissertation
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

# 可视化模块（实验本地）
sys.path.insert(0, str(Path(__file__).parent))
from visualization.visualizer import ExperimentVisualizer

# 全量实验数据持久化（供 plot_from_saved.py 离线绘图）
from data_store import DEFAULT_DATA_FILE, save_experiment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DEM 文件
DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"

# 雷达威胁（城关区内关键位置）
# 雷达1: 布达拉宫附近（军事敏感区）
# 雷达2: 北郊山脊（高海拔监控点）
RADARS = [
    RadarThreat(x0=91.12, y0=29.66, z0=3700, radius=5000, penalty=15.0),
    RadarThreat(x0=91.22, y0=29.72, z0=4500, radius=6000, penalty=12.0),
]

# 代价估算器参数
ESTIMATOR_PARAMS = dict(
    min_clearance=50,    # mx: 最小离地高度(m)
    max_clearance=300,   # my: 最大离地高度(m)
    num_samples=100,     # 剖面采样点数
)

# DMDE 求解器参数
SOLVER_PARAMS = dict(
    pop_size=80,
    max_generations=1000,
    zeta=3,
    delta=0.3,
)

# 多次运行次数
# 可用环境变量 EXP_N_RUNS 覆盖（例如 EXP_N_RUNS=1 快速验证数据保存/绘图链路）
N_RUNS = int(os.environ.get("EXP_N_RUNS", "2"))

# 可视化配置
VISUALIZE = True  # 是否生成可视化图表
FIGURES_DIR = RESULTS_DIR / "figures"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario_balanced():
    """N=M 平衡指派场景（10 UAV → 10 Target）。

    城关区 DEM 范围: 91.01~91.30°E, 29.51~29.80°N, 高程 3611~5660m
    UAV 起飞点分布在城区周边低海拔区域，
    目标点分布在城关区各关键位置。
    max_range 按实际距离 +30% 余量设置，确保可行解存在。
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
    alpha, beta = 2.5, 1.5
    return uavs, targets, alpha, beta


def make_scenario_overloaded():
    """N>M 多对一场景（12 UAV → 4 Target）。

    城关区 DEM 范围内，12 架 UAV 从城区各方向起飞，
    4 个高价值目标分布在城区核心和周边。
    max_range 按实际距离 +30% 余量设置。
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
    alpha, beta = 2.5, 1.5
    return uavs, targets, alpha, beta


def make_scenario_srp():
    """N<M 群巡游场景（4 UAV → 10 Target）。

    城关区 DEM 范围内，4 架 UAV 从城区四角起飞，
    巡游访问分散在城区的 10 个目标点。
    max_range 按实际巡游距离 +30% 余量设置。
    时间窗放宽至实际飞行时间量级。
    """
    uavs = [
        UAV(id=0, start_pos=(91.04, 29.55, 3650), speed_range=(0.20, 0.50), max_range=90000),
        UAV(id=1, start_pos=(91.15, 29.53, 3640), speed_range=(0.25, 0.55), max_range=95000),
        UAV(id=2, start_pos=(91.10, 29.60, 3680), speed_range=(0.30, 0.60), max_range=100000),
        UAV(id=3, start_pos=(91.20, 29.55, 3660), speed_range=(0.20, 0.50), max_range=80000),
    ]
    targets = [
        Target(id=0, position=(91.12, 29.66, 3700), weight=1.0,
               sequence_group=1),
        Target(id=1, position=(91.10, 29.70, 3750), weight=0.8,
               sequence_group=1, time_window=(50000, 200000)),
        Target(id=2, position=(91.16, 29.65, 3680), weight=0.9,
               sequence_group=1),
        Target(id=3, position=(91.20, 29.72, 3800), weight=0.7,
               sequence_group=2),
        Target(id=4, position=(91.08, 29.68, 3720), weight=0.6,
               sequence_group=2, time_window=(40000, 180000)),
        Target(id=5, position=(91.22, 29.60, 3700), weight=0.85,
               sequence_group=2),
        Target(id=6, position=(91.15, 29.75, 3900), weight=0.75),
        Target(id=7, position=(91.06, 29.63, 3690), weight=0.65,
               time_window=(40000, 150000)),
        Target(id=8, position=(91.18, 29.68, 3750), weight=0.95,
               sequence_group=1),
        Target(id=9, position=(91.25, 29.65, 3800), weight=0.7,
               sequence_group=2),
    ]
    alpha, beta = 1.5, 1.0
    return uavs, targets, alpha, beta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验执行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_scenario(
    name: str,
    uavs: list[UAV],
    targets: list[Target],
    alpha: float,
    beta: float,
    estimator: VerticalSectionCostEstimator,
    n_runs: int = N_RUNS,
) -> dict:
    """执行单个场景的多次实验。

    Returns:
        结果字典，包含指标和详细结果。
    """
    n, m = len(uavs), len(targets)
    model_type = (
        "balanced" if n == m else "overloaded" if n > m else "srp"
    )

    print(f"\n{'='*60}")
    print(f"场景: {name} ({model_type}, {n}U/{m}T)")
    print(f"{'='*60}")

    # 1. 构建代价矩阵
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}]")

    # 2. 创建评估器
    evaluator = FitnessEvaluator(uavs, targets, alpha=alpha, beta=beta)

    # 3. 多次运行
    results = []
    for run_idx in range(n_runs):
        cfg = DMDEConfig(
            pop_size=SOLVER_PARAMS["pop_size"],
            max_generations=SOLVER_PARAMS["max_generations"],
            zeta=SOLVER_PARAMS["zeta"],
            delta=SOLVER_PARAMS["delta"],
            seed=run_idx,
            verbose=False,
        )
        solver = DMDESolver(cfg)
        result = solver.solve(
            cm.matrix, n, m, fitness_evaluator=evaluator
        )
        results.append(result)

        # 计算约束违背
        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix, n_uavs=n)
        result.extra["total_violation"] = (
            eval_res.range_violation + eval_res.time_violation
            + eval_res.seq_violation + eval_res.sync_violation
        )
        result.extra["is_feasible"] = eval_res.is_feasible

        print(f"  Run {run_idx}: fitness={result.best_fitness:.1f}, "
              f"feasible={eval_res.is_feasible}, "
              f"time={result.elapsed_seconds:.2f}s")

    # 4. 统计指标
    metrics = compute_metrics(results)
    print(f"\n{format_metrics(metrics, name)}")

    return {
        "name": name,
        "model_type": model_type,
        "n_uavs": n,
        "n_targets": m,
        "metrics": metrics,
        "results": results,
        "cost_matrix": cm.matrix,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验元信息（写入全量数据文件）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_meta() -> dict:
    """构建写入数据文件的实验元信息。"""
    return {
        "experiment": "exp_dmde_01",
        "description": "DMDE 算法验证实验（拉萨地区真实 DEM 地形）",
        "solver_params": SOLVER_PARAMS,
        "estimator_params": ESTIMATOR_PARAMS,
        "n_runs": N_RUNS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print("=" * 60)
    print("exp_dmde_01: DMDE 算法验证实验")
    print("=" * 60)

    # 1. 加载环境
    print("\n[1] 加载环境...")
    t0 = time.time()
    dem = DEMTerrain.from_file(DEM_FILE)
    print(f"  DEM: {dem.meta.width}×{dem.meta.height}, "
          f"bounds={[f'{b:.3f}' for b in dem.bounds]}")

    radar_field = RadarThreatField(RADARS)
    print(f"  雷达: {len(RADARS)} 个")

    estimator = VerticalSectionCostEstimator(
        dem_terrain=dem,
        radar_field=radar_field,
        **ESTIMATOR_PARAMS,
    )
    print(f"  环境加载耗时: {time.time()-t0:.2f}s")

    # 2. 执行三种场景
    scenarios = []
    uavs_dict = {}  # 用于可视化
    targets_dict = {}

    # N=M 平衡指派
    uavs, targets, alpha, beta = make_scenario_balanced()
    uavs_dict["N=M 平衡指派"] = uavs
    targets_dict["N=M 平衡指派"] = targets
    scenarios.append(run_scenario("N=M 平衡指派", uavs, targets, alpha, beta, estimator))

    # N>M 多对一
    uavs, targets, alpha, beta = make_scenario_overloaded()
    uavs_dict["N>M 多对一"] = uavs
    targets_dict["N>M 多对一"] = targets
    scenarios.append(run_scenario("N>M 多对一", uavs, targets, alpha, beta, estimator))

    # N<M 群巡游
    uavs, targets, alpha, beta = make_scenario_srp()
    uavs_dict["N<M 群巡游"] = uavs
    targets_dict["N<M 群巡游"] = targets
    scenarios.append(run_scenario("N<M 群巡游", uavs, targets, alpha, beta, estimator))

    # 3. 汇总对比
    print(f"\n{'='*60}")
    print("汇总对比")
    print(f"{'='*60}")
    print(f"{'场景':<16} {'类型':<12} {'最优':>10} {'均值±标准差':>18} {'可行率':>8} {'耗时':>8}")
    print("-" * 72)
    for sc in scenarios:
        m = sc["metrics"]
        print(f"{sc['name']:<16} {sc['model_type']:<12} "
              f"{m.best_fitness:>10.1f} "
              f"{m.mean_fitness:>8.1f}±{m.std_fitness:<7.1f} "
              f"{m.feasible_rate:>7.0%} "
              f"{m.mean_time:>7.2f}s")

    # 4. 保存全量实验数据（供 plot_from_saved.py 离线绘图，无需重跑实验）
    data_file = save_experiment(
        scenarios, uavs_dict, targets_dict,
        meta=_build_meta(),
        path=RESULTS_DIR / DEFAULT_DATA_FILE,
    )
    print(f"\n全量实验数据已保存: {data_file}")

    # 5. 生成可视化图表
    if VISUALIZE:
        print(f"\n[5] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR)
        saved_files = viz.plot_all(scenarios, uavs_dict, targets_dict, dem_terrain=dem)
        print(f"  共生成 {len(saved_files)} 张图表")
        print(f"  图表保存位置: {FIGURES_DIR}")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()