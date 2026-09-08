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

import json
import sys
import time
from pathlib import Path

import numpy as np

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DEM 文件
DEM_FILE = DATA_DIR / "ASTGTMV003_N29E091" / "ASTGTMV003_N29E091_dem.tif"

# 雷达威胁（拉萨周边）
RADARS = [
    RadarThreat(x0=91.15, y0=29.65, z0=3700, radius=12000, penalty=10.0),
    RadarThreat(x0=91.35, y0=29.50, z0=3650, radius=10000, penalty=8.0),
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
N_RUNS = 5

# 可视化配置
VISUALIZE = True  # 是否生成可视化图表
FIGURES_DIR = RESULTS_DIR / "figures"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario_balanced():
    """N=M 平衡指派场景（10 UAV → 10 Target）。

    约束设计：
    - max_range: 20000~28000 → 不同 UAV 能力不同，部分组合不可达
    - time_window: 6/10 目标有时间窗，限制可执行时段
    - sequence_group: 2 对目标有时序关系
    """
    uavs = [
        UAV(id=0,  start_pos=(91.00, 29.50, 3600), speed_range=(0.20, 0.50), max_range=42000),
        UAV(id=1,  start_pos=(91.10, 29.45, 3650), speed_range=(0.25, 0.55), max_range=40000),
        UAV(id=2,  start_pos=(91.20, 29.55, 3600), speed_range=(0.30, 0.60), max_range=38000),
        UAV(id=3,  start_pos=(91.05, 29.60, 3580), speed_range=(0.20, 0.50), max_range=36000),
        UAV(id=4,  start_pos=(91.15, 29.40, 3620), speed_range=(0.25, 0.55), max_range=34000),
        UAV(id=5,  start_pos=(91.25, 29.48, 3610), speed_range=(0.30, 0.60), max_range=41000),
        UAV(id=6,  start_pos=(91.08, 29.52, 3590), speed_range=(0.20, 0.50), max_range=39000),
        UAV(id=7,  start_pos=(91.18, 29.42, 3630), speed_range=(0.25, 0.55), max_range=37000),
        UAV(id=8,  start_pos=(91.03, 29.58, 3600), speed_range=(0.30, 0.60), max_range=35000),
        UAV(id=9,  start_pos=(91.12, 29.47, 3640), speed_range=(0.20, 0.50), max_range=40000),
    ]
    targets = [
        Target(id=0, position=(91.30, 29.70, 3700), weight=1.0,
               time_window=(20000, 35000)),
        Target(id=1, position=(91.18, 29.80, 3650), weight=0.8,
               sequence_group=1),
        Target(id=2, position=(91.25, 29.58, 3700), weight=0.9,
               sequence_group=1),
        Target(id=3, position=(91.35, 29.65, 3680), weight=0.7,
               time_window=(18000, 30000)),
        Target(id=4, position=(91.22, 29.72, 3720), weight=0.6,
               time_window=(15000, 28000)),
        Target(id=5, position=(91.28, 29.55, 3690), weight=0.85,
               time_window=(20000, 32000)),
        Target(id=6, position=(91.32, 29.60, 3710), weight=0.75),
        Target(id=7, position=(91.15, 29.65, 3670), weight=0.65,
               time_window=(16000, 29000)),
        Target(id=8, position=(91.20, 29.75, 3700), weight=0.95,
               sequence_group=2),
        Target(id=9, position=(91.33, 29.55, 3680), weight=0.7,
               sequence_group=2),
    ]
    alpha, beta = 2.5, 1.5
    return uavs, targets, alpha, beta


def make_scenario_overloaded():
    """N>M 多对一场景（12 UAV → 4 Target）。

    约束设计：
    - max_range: 20000~30000 → 不同 UAV 能力差异大
    - time_window: 所有目标有时间窗
    - sync: 多 UAV 同时到达同一目标时需协同
    """
    uavs = [
        UAV(id=0,  start_pos=(91.00, 29.50, 3600), speed_range=(0.20, 0.50), max_range=45000),
        UAV(id=1,  start_pos=(91.10, 29.45, 3650), speed_range=(0.25, 0.55), max_range=42000),
        UAV(id=2,  start_pos=(91.20, 29.55, 3600), speed_range=(0.30, 0.60), max_range=48000),
        UAV(id=3,  start_pos=(91.05, 29.60, 3580), speed_range=(0.20, 0.50), max_range=38000),
        UAV(id=4,  start_pos=(91.15, 29.40, 3620), speed_range=(0.25, 0.55), max_range=36000),
        UAV(id=5,  start_pos=(91.25, 29.48, 3610), speed_range=(0.30, 0.60), max_range=44000),
        UAV(id=6,  start_pos=(91.08, 29.52, 3590), speed_range=(0.20, 0.50), max_range=40000),
        UAV(id=7,  start_pos=(91.18, 29.42, 3630), speed_range=(0.25, 0.55), max_range=37000),
        UAV(id=8,  start_pos=(91.03, 29.58, 3600), speed_range=(0.30, 0.60), max_range=46000),
        UAV(id=9,  start_pos=(91.12, 29.47, 3640), speed_range=(0.20, 0.50), max_range=35000),
        UAV(id=10, start_pos=(91.22, 29.50, 3610), speed_range=(0.25, 0.55), max_range=42000),
        UAV(id=11, start_pos=(91.07, 29.55, 3595), speed_range=(0.30, 0.60), max_range=44000),
    ]
    targets = [
        Target(id=0, position=(91.30, 29.70, 3700), weight=1.0,
               time_window=(20000, 35000)),
        Target(id=1, position=(91.18, 29.80, 3650), weight=0.8,
               time_window=(18000, 32000)),
        Target(id=2, position=(91.25, 29.58, 3700), weight=0.9,
               time_window=(22000, 38000)),
        Target(id=3, position=(91.35, 29.65, 3680), weight=0.7,
               time_window=(16000, 30000)),
    ]
    alpha, beta = 2.5, 1.5
    return uavs, targets, alpha, beta


def make_scenario_srp():
    """N<M 群巡游场景（4 UAV → 10 Target）。

    约束设计：
    - max_range: 55000~70000 → 每架 UAV 访问多个目标，航程成为硬约束
    - sequence_group: 目标分两组，组内必须按序执行
    - time_window: 部分目标有时间窗
    """
    uavs = [
        UAV(id=0, start_pos=(91.00, 29.50, 3600), speed_range=(0.20, 0.50), max_range=100000),
        UAV(id=1, start_pos=(91.15, 29.45, 3650), speed_range=(0.25, 0.55), max_range=110000),
        UAV(id=2, start_pos=(91.10, 29.55, 3600), speed_range=(0.30, 0.60), max_range=120000),
        UAV(id=3, start_pos=(91.05, 29.48, 3620), speed_range=(0.20, 0.50), max_range=90000),
    ]
    targets = [
        Target(id=0, position=(91.30, 29.70, 3700), weight=1.0,
               sequence_group=1),
        Target(id=1, position=(91.18, 29.80, 3650), weight=0.8,
               sequence_group=1, time_window=(20000, 40000)),
        Target(id=2, position=(91.25, 29.58, 3700), weight=0.9,
               sequence_group=1),
        Target(id=3, position=(91.35, 29.65, 3680), weight=0.7,
               sequence_group=2),
        Target(id=4, position=(91.22, 29.72, 3720), weight=0.6,
               sequence_group=2, time_window=(18000, 35000)),
        Target(id=5, position=(91.28, 29.55, 3690), weight=0.85,
               sequence_group=2),
        Target(id=6, position=(91.32, 29.60, 3710), weight=0.75),
        Target(id=7, position=(91.15, 29.65, 3670), weight=0.65,
               time_window=(15000, 30000)),
        Target(id=8, position=(91.20, 29.75, 3700), weight=0.95,
               sequence_group=1),
        Target(id=9, position=(91.33, 29.55, 3680), weight=0.7,
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
        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix)
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
# 结果保存
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_results(scenarios: list[dict], output_dir: Path) -> None:
    """保存实验结果到 JSON 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for sc in scenarios:
        m = sc["metrics"]
        summary.append({
            "name": sc["name"],
            "model_type": sc["model_type"],
            "n_uavs": sc["n_uavs"],
            "n_targets": sc["n_targets"],
            "best_fitness": round(m.best_fitness, 2),
            "mean_fitness": round(m.mean_fitness, 2),
            "std_fitness": round(m.std_fitness, 2),
            "feasible_rate": round(m.feasible_rate, 2),
            "mean_time_s": round(m.mean_time, 2),
            "n_runs": m.n_runs,
            "best_assignment": sc["results"][m.best_run_idx].best_assignment,
        })

    output_file = output_dir / "exp_dmde_01_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {output_file}")


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

    # 4. 保存结果
    save_results(scenarios, RESULTS_DIR)

    # 5. 生成可视化图表
    if VISUALIZE:
        print(f"\n[5] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR)
        saved_files = viz.plot_all(scenarios, uavs_dict, targets_dict)
        print(f"  共生成 {len(saved_files)} 张图表")
        print(f"  图表保存位置: {FIGURES_DIR}")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()