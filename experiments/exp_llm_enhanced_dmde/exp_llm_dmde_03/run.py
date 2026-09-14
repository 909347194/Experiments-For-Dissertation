# -*- coding: utf-8 -*-
"""exp_llm_dmde_03 — LLM 增强 N<M 群巡游 (SRP) 实验

实验目的：
    在拉萨城关区 DEM 地形上，验证 LLM 增强 DMDE 算法对 N<M 群巡游
    模型的求解能力，并与原始 exp_dmde_03 结果进行对比。

与原始 exp_dmde_03 的关键差异（已用 ★ 标记）：
    1. ★ 导入 LLMEnhancedDMDESolver / LLMEnhancedDMDEConfig
    2. ★ 加载 config/llm_config.yaml 并传入 solver
    3. ★ 保存 LLM 决策日志到结果文件
    4. 场景定义（UAVs, targets, constraints）与原始完全一致，保证公平对比

约束配置（通过 EXP_CONSTRAINTS 选配，与 exp_dmde_03 一致）：
    - 航程约束 (max_range):      每 UAV 总巡游航程限制 ★核心
    - 最大飞行时间 (max_time):    每 UAV 总巡游时间限制
    - 时间窗约束 (time_window):   目标的可执行时间窗口
    - 时序约束 (sequence_group):  目标间的先后执行顺序
    - 同时到达约束:               N/A（每 UAV 独立巡游）

用法：
    python run.py                                   # 默认：航程+时间窗+时序
    EXP_CONSTRAINTS=range python run.py             # 仅航程
    EXP_N_RUNS=3 python run.py                      # 运行次数
"""

from __future__ import annotations

import json
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
# ★ 与原始 exp_dmde_03 的区别：从 algorithm_llm_enhanced_dmde 导入
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
from algorithms.algorithm_llm_enhanced_dmde import (
    LLMEnhancedDMDESolver,
    LLMEnhancedDMDEConfig,
)
from utils.utils_dmde.metrics import compute_metrics, format_metrics

# 复用 exp_llm_dmde_01 的 visualization / data_store（与 exp_dmde_03 复用 exp_dmde_01 一致）
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01"))
from visualization.visualizer import ExperimentVisualizer
from data_store import save_experiment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"
if not DEM_FILE.exists():
    # 回退：复用对应 DMDE 基线实验（exp_dmde_03）的 DEM 栅格；
    # 若本目录存在 data/*.tif 则优先使用它。
    _fallback_dem = (PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_03"
                     / "data" / "chengguan_district_dem.tif")
    if _fallback_dem.exists():
        DEM_FILE = _fallback_dem

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ★ LLM 配置加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_llm_config() -> dict:
    """从 config/llm_config.yaml 加载 LLM 增强配置。"""
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
# 约束选配（★ 与 exp_dmde_03 完全一致）
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
# 场景定义：N<M 群巡游（4 UAV → 10 Target）
# ★ 与原始 exp_dmde_03 完全一致，保证公平对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_scenario(cc: dict[str, bool]):
    """N<M 群巡游场景，根据约束配置动态调整。"""
    # 基础 UAV 配置（SRP 需要较大航程）
    uav_base = [
        dict(id=0, start_pos=(91.04, 29.55, 3650), speed_range=(0.20, 0.50), max_range=90000),
        dict(id=1, start_pos=(91.15, 29.53, 3640), speed_range=(0.25, 0.55), max_range=95000),
        dict(id=2, start_pos=(91.10, 29.60, 3680), speed_range=(0.30, 0.60), max_range=100000),
        dict(id=3, start_pos=(91.20, 29.55, 3660), speed_range=(0.20, 0.50), max_range=85000),
    ]

    # max_time: 2.0x 余量
    if cc["time"]:
        for u in uav_base:
            avg_speed = (u["speed_range"][0] + u["speed_range"][1]) / 2
            u["max_time"] = u["max_range"] / avg_speed * 2.0

    uavs = [UAV(**u) for u in uav_base]

    # 基础 Target 配置
    # 时序分组：group 1 = T0,T1,T2,T8；group 2 = T3,T4,T5,T9
    tgt_base = [
        dict(id=0, position=(91.12, 29.66, 3700), weight=1.0),   # group1
        dict(id=1, position=(91.10, 29.70, 3750), weight=0.8),   # group1
        dict(id=2, position=(91.16, 29.65, 3680), weight=0.9),   # group1
        dict(id=3, position=(91.20, 29.72, 3800), weight=0.7),   # group2
        dict(id=4, position=(91.08, 29.68, 3720), weight=0.6),   # group2
        dict(id=5, position=(91.22, 29.60, 3700), weight=0.85),  # group2
        dict(id=6, position=(91.15, 29.75, 3900), weight=0.75),  # 无时序
        dict(id=7, position=(91.06, 29.63, 3690), weight=0.65),  # 无时序
        dict(id=8, position=(91.18, 29.68, 3750), weight=0.95),  # group1
        dict(id=9, position=(91.25, 29.65, 3800), weight=0.7),   # group2
    ]

    # 时间窗
    if cc["time"]:
        tgt_base[0]["time_window"] = (30000, 200000)
        tgt_base[1]["time_window"] = (50000, 250000)
        tgt_base[4]["time_window"] = (40000, 220000)
        tgt_base[7]["time_window"] = (35000, 180000)

    # 时序约束
    if cc["seq"]:
        for i in [0, 1, 2, 8]:
            tgt_base[i]["sequence_group"] = 1
        for i in [3, 4, 5, 9]:
            tgt_base[i]["sequence_group"] = 2

    targets = [Target(**t) for t in tgt_base]

    alpha, beta = 1.5, 1.0  # N<M 缩放因子
    return uavs, targets, alpha, beta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    cc = parse_constraint_config()
    llm_config = load_llm_config()

    print("=" * 60)
    print("exp_llm_dmde_03: LLM 增强 N<M 群巡游 (SRP) 实验")
    print("=" * 60)
    print(f"  约束配置: {describe_constraints(cc)}")
    print(f"  LLM 模块: search_controller (interval={LLM_INTERVAL})")
    print(f"  LLM 模型: {llm_config.get('provider', 'deepseek')}/{llm_config.get('model', 'default')}")
    print(f"  LLM 思考强度: reasoning_effort={llm_config.get('reasoning_effort', '服务端默认')}, "
          f"max_tokens={llm_config.get('max_tokens', '默认')}")
    print(f"  运行次数: {N_RUNS}")
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
    uavs, targets, alpha, beta = make_scenario(cc)
    n, m = len(uavs), len(targets)
    name = "N<M 群巡游"

    print(f"\n{'='*60}")
    print(f"场景: {name} ({n}U/{m}T)")
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
    # ★ LLM 决策日志收集器
    all_llm_decisions: dict[int, list] = {}

    results = []
    run_start_time = time.time()

    # 消融实验：从环境变量读取 modules 覆盖配置
    _env_modules = os.environ.get("EXP_MODULES")
    if _env_modules:
        try:
            modules_override = json.loads(_env_modules)
            print(f"\n[消融模式] modules 覆盖: {modules_override}")
        except json.JSONDecodeError:
            print(f"  ⚠️ EXP_MODULES JSON 解析失败，使用默认配置")
            modules_override = None
    else:
        modules_override = None

    for run_idx in range(N_RUNS):
        print(f"\n  ── Run {run_idx + 1}/{N_RUNS} " + "─" * 40)

        # ★ 使用 LLMEnhancedDMDEConfig / LLMEnhancedDMDESolver
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

        # SRP 评估需要 n_uavs 参数
        eval_res = evaluator.evaluate(result.best_assignment, cm.matrix, n_uavs=n)
        result.extra["total_violation"] = eval_res.total_violation
        result.extra["violation_breakdown"] = eval_res.violation_breakdown()
        result.extra["is_feasible"] = eval_res.is_feasible

        assigned_tgts = set(a[1] for a in result.best_assignment)
        print(f"    适应度: {result.best_fitness:.1f}")
        print(f"    可行性: {'✅ 可行' if eval_res.is_feasible else '❌ 不可行'}")
        if not eval_res.is_feasible:
            print(f"    违反总量: {eval_res.total_violation:.2f}")
            for k, v in eval_res.violation_breakdown().items():
                if v > 0:
                    print(f"      {k}: {v}")
        print(f"    目标覆盖: {len(assigned_tgts)}/{m}")
        print(f"    耗时: {result.elapsed_seconds:.2f}s")

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

    # 目标覆盖 + 巡游路线
    best = results[metrics.best_run_idx]
    assigned_tgts = set(a[1] for a in best.best_assignment)
    print(f"  目标覆盖: {sorted(assigned_tgts)} / {list(range(m))}")
    if assigned_tgts != set(range(m)):
        print("  ⚠️  存在未覆盖目标！")
    routes: dict[int, list[int]] = {}
    for uid, tid in best.best_assignment:
        routes.setdefault(uid, []).append(tid)
    print(f"  巡游路线:")
    for uid, tgts in routes.items():
        print(f"    U{uid} → {tgts}")

    scenario = {
        "name": name, "model_type": "srp",
        "n_uavs": n, "n_targets": m,
        "metrics": metrics, "results": results, "cost_matrix": cm.matrix,
        "constraint_config": cc,
    }

    # 7. 保存数据（★ 包含 LLM 决策日志）
    print(f"\n[保存数据]")
    data_file = save_experiment(
        [scenario], {name: uavs}, {name: targets},
        meta={
            "experiment": "exp_llm_enhanced_dmde/exp_llm_dmde_03",
            "description": "LLM 增强 N<M 群巡游",
            "solver_params": SOLVER_PARAMS,
            "llm_config": llm_config,
            "n_runs": N_RUNS,
            "constraint_config": cc,
            "constraint_desc": describe_constraints(cc),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_seconds": total_time,
        },
        path=RESULTS_DIR / "exp_llm_dmde_03_data.json",
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
            experiment_id="llm_dmde_03",
            label="LLM-DMDE N<M 群巡游 (search_controller)",
            result_path="exp_llm_enhanced_dmde/exp_llm_dmde_03/results/exp_llm_dmde_03_data.json",
            tags=["llm", "dmde", "srp", "search_controller"],
            meta={
                "n_runs": N_RUNS,
                "solver_params": SOLVER_PARAMS,
                "llm_model": llm_config.get("model", "default"),
                "llm_interval": LLM_INTERVAL,
                "constraint_desc": describe_constraints(cc),
            },
        )
    except Exception:
        pass

    # 8. 可视化（★ LLM 决策专用图表）
    if VISUALIZE:
        print(f"\n[8] 生成可视化图表...")
        viz = ExperimentVisualizer(output_dir=FIGURES_DIR, algo_name="LLM-DMDE")
        saved = viz.plot_all([scenario], {name: uavs}, {name: targets},
                             dem_terrain=dem, llm_decisions=all_llm_decisions)
        print(f"  共生成 {len(saved)} 张图表 → {FIGURES_DIR}")

    # 9. LLM 统计摘要
    if all_llm_decisions:
        print(f"\n[LLM 统计摘要]")
        all_crs = []
        all_durations = []
        n_failed = 0
        for decisions in all_llm_decisions.values():
            for d in decisions:
                dec = d.get("decision", {})
                all_crs.append(dec.get("cr", 0.5))
                all_durations.append(d.get("duration", 0))
                if "_error" in d:
                    n_failed += 1
        total = len(all_crs)
        print(f"  总调用: {total} 次")
        if n_failed:
            print(f"  失败: {n_failed} 次")
        if all_crs:
            print(f"  CR 均值: {sum(all_crs)/len(all_crs):.3f}, "
                  f"范围: [{min(all_crs):.1f}, {max(all_crs):.1f}]")
        if all_durations:
            print(f"  调用耗时: 均值={sum(all_durations)/len(all_durations):.1f}s, "
                  f"总计={sum(all_durations):.0f}s")

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()
