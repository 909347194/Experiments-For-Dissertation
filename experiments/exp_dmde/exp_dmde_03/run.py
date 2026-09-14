# -*- coding: utf-8 -*-
"""exp_dmde_03 — N<M 群巡游 (SRP) 实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 DMDE 算法对 N<M 群巡游
    模型的求解能力，并输出统计指标。

约束配置（通过 EXP_CONSTRAINTS 选配）：
    - 航程约束 (max_range):      每 UAV 总巡游航程限制 ★核心
    - 最大飞行时间 (max_time):    每 UAV 总巡游时间限制
    - 时间窗约束 (time_window):   目标的可执行时间窗口
    - 时序约束 (sequence_group):  目标间的先后执行顺序
    - 同时到达约束:               N/A（每 UAV 独立巡游）

N<M 特有约束：
    - 巡游顺序最短：同 UAV 的目标序列按最短路径排列
    - 每目标恰好一次：每个目标恰好被一架 UAV 访问

用法：
    python run.py                                    # 默认：航程+时间窗+时序
    EXP_CONSTRAINTS=range python run.py              # 仅航程
    EXP_CONSTRAINTS=all python run.py                # 全部约束
    EXP_N_RUNS=3 python run.py                       # 3次运行
"""

from __future__ import annotations
import numpy as np

import os
import sys
import time
from pathlib import Path

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
from data_store import save_experiment


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

# ── 多规模支持 ────────────────────────────────────────────────
SCALE = os.environ.get("EXP_SCALE", "medium").lower()
# N<M 规模：small=2U/5T, medium=4U/10T, large=8U/20T
SCALES_UAV = {"small": 2, "medium": 4, "large": 8}
SCALES_TGT = {"small": 5, "medium": 10, "large": 20}
N_UAVS = SCALES_UAV.get(SCALE, 4)
N_TGTS = SCALES_TGT.get(SCALE, 10)

# 支持自定义 solver 参数
_solver_overrides = os.environ.get("EXP_SOLVER_PARAMS", "")
if _solver_overrides:
    import json as _json
    try:
        _overrides = _json.loads(_solver_overrides)
        SOLVER_PARAMS.update(_overrides)
    except _json.JSONDecodeError:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 约束选配
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_constraint_config() -> dict[str, bool]:
    """从环境变量解析约束配置。

    EXP_CONSTRAINTS 支持：
        all              — 全部启用（航程+时间窗+时序）
        none             — 仅航程
        range            — 仅航程
        range,time       — 航程+时间窗
        range,seq        — 航程+时序
        range,time,seq   — 航程+时间窗+时序（默认）

    注意：N<M 不支持同时到达约束（sync），始终禁用。
    """
    raw = os.environ.get("EXP_CONSTRAINTS", "range,time,seq").lower().strip()

    if raw == "all":
        return dict(range=True, time=True, seq=True)
    if raw == "none":
        return dict(range=True, time=False, seq=False)

    parts = {p.strip() for p in raw.split(",")}
    return dict(
        range="range" in parts,
        time="time" in parts,
        seq="seq" in parts,
    )


def describe_constraints(cc: dict[str, bool]) -> str:
    parts = []
    parts.append(f"航程{'✓' if cc['range'] else '✗'}")
    parts.append(f"时间窗{'✓' if cc['time'] else '✗'}")
    parts.append(f"时序{'✓' if cc['seq'] else '✗'}")
    parts.append("同步N/A")
    return " ".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义：N<M 群巡游（支持多规模）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DEM 地理范围
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


def make_scenario(cc: dict[str, bool]):
    """N<M 群巡游场景，支持 small/medium/large 规模。

    规模映射（由 EXP_SCALE 环境变量控制）：
        small:  2U / 5T
        medium: 4U / 10T（默认）
        large:  8U / 20T
    """
    n_uav, n_tgt = N_UAVS, N_TGTS

    uav_positions = _generate_positions(n_uav, seed=500)
    uavs = []
    for i in range(n_uav):
        speed_lo = 0.20 + (i % 3) * 0.05
        speed_hi = speed_lo + 0.30
        max_r = 85000 + (i % 4) * 5000
        uav_dict = dict(
            id=i, start_pos=uav_positions[i],
            speed_range=(round(speed_lo, 2), round(speed_hi, 2)),
            max_range=max_r,
        )
        if cc["time"]:
            avg_speed = (speed_lo + speed_hi) / 2
            uav_dict["max_time"] = max_r / avg_speed * 2.0
        uavs.append(UAV(**uav_dict))

    tgt_positions = _generate_positions(n_tgt, seed=600,
                                        alt_range=(_ALT_MIN + 40, _ALT_MAX + 200))
    targets = []
    for i in range(n_tgt):
        tgt_dict = dict(
            id=i, position=tgt_positions[i],
            weight=round(0.6 + (i % 5) * 0.1, 2),
        )
        if cc["time"] and i % 3 == 0:
            tgt_dict["time_window"] = (30000 + i * 2000, 180000 + i * 5000)
        if cc["seq"]:
            # 交替分组：偶数 id → group 1，奇数 id → group 2
            tgt_dict["sequence_group"] = 1 if i % 2 == 0 else 2
        targets.append(Target(**tgt_dict))

    alpha, beta = 1.5, 1.0  # N<M 缩放因子
    return uavs, targets, alpha, beta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    cc = parse_constraint_config()

    print("=" * 60)
    print("exp_dmde_03: N<M 群巡游 (SRP) 实验")
    print("=" * 60)
    print(f"  约束配置: {describe_constraints(cc)}")
    print(f"  运行次数: {N_RUNS}")

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
    name = "N<M 群巡游"

    print(f"\n{'='*60}")
    print(f"场景: {name} ({n}U/{m}T) [规模: {SCALE}]")
    print(f"  UAV: max_range={min(u.max_range for u in uavs):.0f}~{max(u.max_range for u in uavs):.0f}")
    if cc["time"]:
        print(f"       max_time={min(u.max_time for u in uavs):.0f}~{max(u.max_time for u in uavs):.0f}")
    print(f"  预期每 UAV 巡游 {m//n}~{m//n+1} 个目标")
    for t in targets:
        extras = []
        if t.time_window:
            extras.append(f"tw={t.time_window}")
        if t.sequence_group is not None:
            extras.append(f"grp={t.sequence_group}")
        print(f"  T{t.id}: w={t.weight:.2f} {' '.join(extras)}")
    print(f"{'='*60}")

    # 3. 构建代价矩阵
    t0 = time.time()
    builder = CostMatrixBuilder(estimator, store_details=False)
    cm = builder.build(uavs, targets)
    print(f"  代价矩阵: shape={cm.matrix.shape}, "
          f"range=[{cm.matrix.min():.0f}, {cm.matrix.max():.0f}], "
          f"time={time.time()-t0:.1f}s")

    # 4. 创建评估器（SRP: seq 可选，sync 禁用）
    evaluator = FitnessEvaluator(
        uavs, targets, alpha=alpha, beta=beta,
        enable_seq=cc["seq"],
        enable_window=cc["time"],
        enable_sync=False,
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

        # SRP 评估需要 n_uavs 参数
        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix, n_uavs=n)
        result.extra["total_violation"] = eval_res.total_violation
        result.extra["violation_breakdown"] = eval_res.violation_breakdown()
        result.extra["is_feasible"] = eval_res.is_feasible

        assigned_tgts = set(a[1] for a in result.best_assignment)
        print(f"  Run {run_idx}: fitness={result.best_fitness:.1f}, "
              f"feasible={eval_res.is_feasible}, "
              f"range_vio={eval_res.range_violation:.1f}, "
              f"time_vio={eval_res.time_violation:.1f}, "
              f"seq_vio={eval_res.seq_violation:.1f}, "
              f"window_vio={eval_res.window_violation:.1f}, "
              f"targets={len(assigned_tgts)}/{m}, "
              f"time={result.elapsed_seconds:.2f}s")

    # 6. 统计
    metrics = compute_metrics(results)
    print(f"\n{format_metrics(metrics, name)}")

    # 目标覆盖检查
    best = results[metrics.best_run_idx]
    assigned_tgts = set(a[1] for a in best.best_assignment)
    print(f"  目标覆盖: {sorted(assigned_tgts)} / {list(range(m))}")
    if assigned_tgts != set(range(m)):
        print("  ⚠️  存在未覆盖目标！")

    # UAV 巡游路线
    print(f"  巡游路线:")
    routes: dict[int, list[int]] = {}
    for uid, tid in best.best_assignment:
        routes.setdefault(uid, []).append(tid)
    for uid, tgts in routes.items():
        print(f"    U{uid} → {tgts}")

    scenario = {
        "name": name, "model_type": "srp",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
        "constraint_config": cc,
    }

    # 7. 保存数据
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={"experiment": "exp_dmde_03", "description": "N<M 群巡游",
              "solver_params": SOLVER_PARAMS, "n_runs": N_RUNS,
              "constraint_config": cc,
              "constraint_desc": describe_constraints(cc),
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        path=RESULTS_DIR / "exp_dmde_03_data.json",
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
