# -*- coding: utf-8 -*-
"""exp_dmde_01 — LLM 增强 N=M 平衡指派实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 LLM 增强 DMDE 算法对 N=M 平衡指派
    模型的求解能力，并与原始 DMDE 结果进行对比。

与原始 exp_dmde_01 的关键差异（已用 ★ 标记）：
    1. ★ 导入 LLMEnhancedDMDESolver / LLMEnhancedDMDEConfig
    2. ★ 加载 config/llm_config.yaml 并传入 solver
    3. ★ 保存 LLM 决策日志到结果文件
    4. 场景定义（UAVs, targets, constraints）与原始完全一致

N=M 场景约束（单 UAV → 单 Target 一一对应）：
    - 航程约束 (max_range):         ✓ 生效
    - 最大飞行时间 (max_time):        — 本实验未启用
    - 时间窗约束 (time_window):       ✓ 生效（T0/T3/T4 带窗）
    - 时序约束 (sequence_group):      ✗ N/M（单 UAV 仅 1 目标，无需排序）
    - 同时到达约束 (sync):            ✗ N/M（单目标仅 1 UAV，无协同）
"""

import numpy as np
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
CONFIG_DIR = Path(__file__).parent / "config"

sys.path.insert(0, str(SRC_ROOT))

# Windows 下控制台/重定向管道默认使用 GBK
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        _reconfigure(errors="replace")

# ── 导入项目模块 ──────────────────────────────────────────────
# ★ 与原始 exp_dmde_01 的区别：从 algorithm_llm_enhanced_dmde 导入
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
# ★ 原始导入: from algorithms.algorithm_dmde import DMDESolver, DMDEConfig
from algorithms.algorithm_llm_enhanced_dmde import (
    LLMEnhancedDMDESolver,
    LLMEnhancedDMDEConfig,
)
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
LLM_INTERVAL = 100          # search_controller 触发间隔（代数）
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
# ★ LLM 配置加载（新增）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_llm_config() -> dict:
    """从 config/llm_config.yaml 加载 LLM 增强配置。

    如果配置文件不存在或加载失败，返回空 dict（solver 将使用默认值）。
    """
    import yaml

    config_path = CONFIG_DIR / "llm_config.yaml"
    if not config_path.exists():
        print(f"  ⚠️  LLM 配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"  LLM 配置已加载: {cfg.get('provider', 'deepseek')}/{cfg.get('model', 'default')}, "
          f"reasoning_effort={cfg.get('reasoning_effort', '服务端默认')}, "
          f"interval={LLM_INTERVAL}")
    return cfg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景定义：N=M 平衡指派（支持多规模）
# ★ 与原始 exp_dmde_01 完全一致，保证公平对比
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


def make_scenario():
    """N=M 平衡指派场景，支持 small/medium/large 规模。

    规模映射（由 EXP_SCALE 环境变量控制）：
        small:  5U / 5T
        medium: 10U / 10T（默认）
        large:  20U / 20T

    约束：仅启用航程 + 时间窗（N=M 场景时序/sync 不触发，直接关闭减少开销）。
    """
    n = N_ENTITIES

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
    print("exp_dmde_01: LLM 增强 N=M 平衡指派实验")
    print("=" * 60)

    # ★ 加载 LLM 配置
    llm_config = load_llm_config()

    # ── 实验配置汇总 ─────────────────────────────────────────
    print("\n[配置汇总]")
    print(f"  DMDE 参数: pop_size={SOLVER_PARAMS['pop_size']}, "
          f"max_gen={SOLVER_PARAMS['max_generations']}, "
          f"zeta={SOLVER_PARAMS['zeta']}, delta={SOLVER_PARAMS['delta']}")
    print(f"  LLM 模块: search_controller (interval={LLM_INTERVAL})")
    llm_model = llm_config.get('model', 'default')
    llm_provider = llm_config.get('provider', 'deepseek')
    print(f"  LLM 模型: {llm_provider}/{llm_model}")
    print(f"  LLM 思考强度: reasoning_effort={llm_config.get('reasoning_effort', '服务端默认')}, "
          f"max_tokens={llm_config.get('max_tokens', '默认')}")
    print(f"  运行次数: {N_RUNS}")
    print(f"  随机种子: 0 ~ {N_RUNS - 1}（每轮 seed=run_idx）")
    print(f"  输出目录: {RESULTS_DIR}")

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

    # 2. 构建场景（★ 与原始完全一致）
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
    # ★ LLM 决策日志收集器
    all_llm_decisions: dict[int, list] = {}

    results = []
    run_start_time = time.time()

    # 消融实验：从环境变量读取 modules 覆盖配置
    import json as _json
    _env_modules = os.environ.get("EXP_MODULES")
    if _env_modules:
        try:
            modules_override = _json.loads(_env_modules)
            print(f"\n[消融模式] modules 覆盖: {modules_override}")
        except _json.JSONDecodeError:
            print(f"  ⚠️ EXP_MODULES JSON 解析失败，使用默认配置")
            modules_override = None
    else:
        modules_override = None

    for run_idx in range(N_RUNS):
        print(f"\n  ── Run {run_idx + 1}/{N_RUNS} " + "─" * 40)

        # ★ 使用 LLMEnhancedDMDEConfig 替代 DMDEConfig
        # ★ 额外传入 llm_config 参数
        cfg = LLMEnhancedDMDEConfig(
            pop_size=SOLVER_PARAMS["pop_size"],
            max_generations=SOLVER_PARAMS["max_generations"],
            zeta=SOLVER_PARAMS["zeta"],
            delta=SOLVER_PARAMS["delta"],
            seed=run_idx,
            verbose=False,
            llm_config_path=str(CONFIG_DIR / "llm_config.yaml"),
            modules=modules_override if modules_override is not None else {
                "population_init": {"enabled": False},
                "search_controller": {
                    "enabled": True,
                    "interval": LLM_INTERVAL,
                    "system_prompt_path": str(CONFIG_DIR / "prompts" / "search_controller.txt"),
                },
            },
        )

        # ★ 使用 LLMEnhancedDMDESolver 替代 DMDESolver
        solver = LLMEnhancedDMDESolver(cfg)
        result = solver.solve(cm.matrix, n, m, fitness_evaluator=evaluator)
        results.append(result)

        # ★ 收集 LLM 决策日志并打印摘要
        if solver.trajectory is not None:
            decisions = solver.trajectory.get_llm_decisions()
            all_llm_decisions[run_idx] = decisions
            n_failed = sum(1 for d in decisions if "cr" not in d.get("decision", {}))
            print(f"    LLM 调用: {len(decisions)} 次"
                  + (f"（其中 {n_failed} 次失败，已回退公式 3-9 的 CR）" if n_failed else ""))
            for d in decisions:
                dec = d.get("decision", {})
                if "cr" not in dec:
                    print(f"      gen={d['generation']:>4d}: ❌ 调用失败（回退公式 CR）, "
                          f"耗时={d.get('duration', 0):.1f}s, "
                          f"原因: {dec.get('error', '未知')}")
                    continue
                reason = ' '.join((dec.get('reasoning') or '').split())
                if len(reason) > 56:
                    reason = reason[:53] + '...'
                print(f"      gen={d['generation']:>4d}: CR={dec['cr']}, "
                      f"耗时={d.get('duration', 0):.1f}s, 理由: {reason or '(无)'}")

        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix, n_uavs=n)
        result.extra["total_violation"] = eval_res.total_violation
        result.extra["violation_breakdown"] = eval_res.violation_breakdown()
        result.extra["is_feasible"] = eval_res.is_feasible

        # 打印本轮结果
        print(f"    适应度: {result.best_fitness:.1f}")
        print(f"    可行性: {'✅ 可行' if eval_res.is_feasible else '❌ 不可行'}")
        if not eval_res.is_feasible:
            print(f"    违反总量: {eval_res.total_violation:.2f}")
            for k, v in eval_res.violation_breakdown().items():
                if v > 0:
                    print(f"      {k}: {v}")
        print(f"    耗时: {result.elapsed_seconds:.2f}s")
        print(f"    分配方案: {result.best_assignment}")

        # 进度和 ETA
        elapsed = time.time() - run_start_time
        avg_per_run = elapsed / (run_idx + 1)
        remaining = avg_per_run * (N_RUNS - run_idx - 1)
        print(f"    进度: {run_idx + 1}/{N_RUNS}, "
              f"已用 {elapsed:.0f}s, 预计剩余 {remaining:.0f}s")

    # 6. 统计
    total_time = time.time() - run_start_time
    metrics = compute_metrics(results)
    print(f"\n{'='*60}")
    print(f"[实验结果汇总]")
    print(f"{'='*60}")
    print(format_metrics(metrics, name))
    print(f"\n  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  平均每轮: {total_time/N_RUNS:.1f}s")
    feasible_count = sum(1 for r in results if r.extra.get('is_feasible', False))
    print(f"  可行解率: {feasible_count}/{N_RUNS} ({100*feasible_count/N_RUNS:.0f}%)")
    if all_llm_decisions:
        total_decisions = sum(len(v) for v in all_llm_decisions.values())
        print(f"  LLM 总调用: {total_decisions} 次")

    scenario = {
        "name": name, "model_type": "balanced",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
    }

    # 7. 保存数据（★ 包含 LLM 决策日志）
    print(f"\n[保存数据]")
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={
            "experiment": "exp_llm_enhanced_EA/exp_dmde_01",
            "description": "LLM 增强 N=M 平衡指派",
            "solver_params": SOLVER_PARAMS,
            "llm_config": llm_config,
            "n_runs": N_RUNS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": total_time,
        },
        path=RESULTS_DIR / "exp_llm_dmde_01_data.json",
        llm_decisions=all_llm_decisions,
    )
    print(f"  数据文件: {data_file}")
    if all_llm_decisions:
        total_decisions = sum(len(v) for v in all_llm_decisions.values())
        print(f"  LLM 决策记录: {total_decisions} 条（{len(all_llm_decisions)} 次运行）")

    # 7.5 注册到索引
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01"))
        from registry import register
        register(
            experiment_id="llm_dmde_01",
            label="LLM-DMDE N=M=10 (search_controller)",
            result_path="exp_llm_enhanced_dmde/exp_llm_dmde_01/results/exp_llm_dmde_01_data.json",
            tags=["llm", "dmde", "balanced", "10u10t", "search_controller"],
            meta={
                "n_runs": N_RUNS,
                "solver_params": SOLVER_PARAMS,
                "llm_model": llm_config.get("model", "default"),
                "llm_interval": LLM_INTERVAL,
            },
        )
    except Exception:
        pass

    # 8. 可视化
    if VISUALIZE:
        print(f"\n[8] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR, algo_name="LLM-DMDE")
        saved = viz.plot_all([scenario], {name: uavs}, {name: targets},
                             dem_terrain=dem, llm_decisions=all_llm_decisions)
        print(f"  共生成 {len(saved)} 张图表 → {FIGURES_DIR}")

    # 9. LLM 统计摘要
    if all_llm_decisions:
        print(f"\n[LLM 统计摘要]")
        all_strategies = []
        all_crs = []
        all_durations = []
        n_failed = 0
        for decisions in all_llm_decisions.values():
            for d in decisions:
                dec = d.get("decision", {})
                all_strategies.append(dec.get("strategy", "default"))
                all_crs.append(dec.get("cr", 0.5))
                all_durations.append(d.get("duration", 0))
                if "_error" in d:
                    n_failed += 1

        from collections import Counter
        strategy_counts = Counter(all_strategies)
        print(f"  总调用: {len(all_strategies)} 次")
        if n_failed:
            print(f"  失败: {n_failed} 次")
        print(f"  策略分布: {dict(strategy_counts)}")
        if all_crs:
            print(f"  CR 均值: {sum(all_crs)/len(all_crs):.3f}, 范围: [{min(all_crs):.1f}, {max(all_crs):.1f}]")
        if all_durations:
            print(f"  调用耗时: 均值={sum(all_durations)/len(all_durations):.1f}s, 总计={sum(all_durations):.0f}s")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()
