# -*- coding: utf-8 -*-
"""exp_llm_enhanced_dmde — LLM 增强 DMDE 实验

实验目的：
    对比标准 DMDE 与 LLM 增强 DMDE 的求解效果。

LLM 增强点：
    1. 初始化：LLM 分析代价矩阵 → 生成启发式种子
    2. 进化中：LLM 监控收敛 → 动态调整 F/CR
    3. 求解后：LLM 解读结果 → 给出改进建议

LLM 不介入：
    反映射修复（规则 3.4/3.5/3.6 独立处理）

用法：
    # 仅标准 DMDE（无 LLM）
    python run.py

    # LLM 增强 DMDE
    LLM_API_KEY=sk-xxx python run.py

    # 指定模型
    LLM_API_KEY=sk-xxx LLM_MODEL=gpt-4o python run.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_01"))

from environments.environment_dmde import (
    DEMTerrain, RadarThreat, RadarThreatField, VerticalSectionCostEstimator,
)
from models.model_dmde import UAV, Target, CostMatrixBuilder, FitnessEvaluator
from algorithms.algorithm_dmde import DMDESolver, DMDEConfig
from algorithms.algorithm_llm_enhanced_dmde import LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig
from utils.utils_dmde.metrics import compute_metrics, format_metrics
from visualization.visualizer import ExperimentVisualizer
from data_store import save_experiment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEM_FILE = DATA_DIR / "chengguan_district_dem.tif"

RADARS = [
    RadarThreat(x0=91.12, y0=29.66, z0=3700, radius=5000, penalty=15.0),
]

ESTIMATOR_PARAMS = dict(min_clearance=50, max_clearance=300, num_samples=100)

SOLVER_PARAMS = dict(pop_size=50, max_generations=500, zeta=3, delta=0.3)

N_RUNS = int(os.environ.get("EXP_N_RUNS", "3"))


def get_llm_config() -> dict | None:
    """从环境变量读取 LLM 配置。"""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        return None
    return {
        "llm_api_key": api_key,
        "llm_api_base": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    }


def make_scenario():
    """N=M 平衡指派场景（6U/6T）。"""
    uavs = [
        UAV(id=i, start_pos=(91.05 + i*0.03, 29.55 + i*0.02, 3650),
            speed_range=(0.20, 0.50), max_range=50000)
        for i in range(6)
    ]
    targets = [
        Target(id=i, position=(91.20 + i*0.02, 29.65 + i*0.015, 3700),
               weight=1.0 - i*0.1)
        for i in range(6)
    ]
    return uavs, targets, 2.5, 1.5


def run_experiment(solver, name, cm, n, m, evaluator):
    """运行 N_RUNS 次实验。"""
    results = []
    for run_idx in range(N_RUNS):
        cfg = DMDEConfig(
            pop_size=SOLVER_PARAMS["pop_size"],
            max_generations=SOLVER_PARAMS["max_generations"],
            seed=run_idx,
        )
        # 每次运行创建新 solver（避免状态残留）
        if hasattr(solver, '_dmde'):
            solver._dmde = DMDESolver(cfg)
        result = solver.solve(cm, n, m, fitness_evaluator=evaluator)
        results.append(result)

        eval_res = evaluator.evaluate(result.best_assignment, cm, n_uavs=n)
        result.extra["is_feasible"] = eval_res.is_feasible
        print(f"  {name} Run {run_idx}: fitness={result.best_fitness:.1f}, "
              f"feasible={eval_res.is_feasible}, time={result.elapsed_seconds:.2f}s")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    llm_config = get_llm_config()
    has_llm = llm_config is not None

    print("=" * 60)
    print("exp_llm_enhanced_dmde: LLM 增强 DMDE 实验")
    print("=" * 60)
    print(f"  LLM: {'✓ ' + llm_config.get('llm_model', '') if has_llm else '✗ 未配置'}")
    print(f"  运行次数: {N_RUNS}")

    # 1. 加载环境
    print("\n[1] 加载环境...")
    dem = DEMTerrain.from_file(DEM_FILE)
    radar = RadarThreatField(RADARS)
    est = VerticalSectionCostEstimator(dem, radar, **ESTIMATOR_PARAMS)

    # 2. 构建场景
    uavs, targets, alpha, beta = make_scenario()
    n, m = len(uavs), len(targets)
    cm = CostMatrixBuilder(est).build(uavs, targets)
    evaluator = FitnessEvaluator(uavs, targets, alpha=alpha, beta=beta)
    print(f"  场景: {n}U/{m}T, 代价矩阵: {cm.matrix.shape}")

    scenarios = []

    # 3. 标准 DMDE
    print(f"\n{'='*60}")
    print("[2] 标准 DMDE")
    std_solver = DMDESolver()
    std_results = run_experiment(std_solver, "DMDE", cm.matrix, n, m, evaluator)
    std_metrics = compute_metrics(std_results)
    print(f"\n{format_metrics(std_metrics, '标准 DMDE')}")
    scenarios.append({
        "name": "标准 DMDE", "model_type": "balanced",
        "n_uavs": n, "n_targets": m,
        "metrics": std_metrics, "results": std_results,
        "cost_matrix": cm.matrix,
    })

    # 4. LLM 增强 DMDE（如果有 LLM）
    if has_llm:
        print(f"\n{'='*60}")
        print("[3] LLM 增强 DMDE")
        llm_solver = LLMEnhancedDMDESolver(LLMEnhancedDMDEConfig(**llm_config))
        llm_results = run_experiment(llm_solver, "LLM-DMDE", cm.matrix, n, m, evaluator)
        llm_metrics = compute_metrics(llm_results)
        print(f"\n{format_metrics(llm_metrics, 'LLM 增强 DMDE')}")

        # LLM 解读
        print(f"\n[4] LLM 结果解读:")
        print(llm_solver.get_interpretation())

        scenarios.append({
            "name": "LLM 增强 DMDE", "model_type": "balanced",
            "n_uavs": n, "n_targets": m,
            "metrics": llm_metrics, "results": llm_results,
            "cost_matrix": cm.matrix,
        })

    # 5. 保存 + 可视化
    save_experiment(
        scenarios, {"标准 DMDE": uavs, "LLM 增强 DMDE": uavs},
        {"标准 DMDE": targets, "LLM 增强 DMDE": targets},
        meta={"experiment": "exp_llm_enhanced_dmde", "has_llm": has_llm},
        path=RESULTS_DIR / "exp_llm_enhanced_dmde_data.json",
    )

    viz = ExperimentVisualizer(output_dir=RESULTS_DIR / "figures")
    viz.plot_all(scenarios, {"标准 DMDE": uavs, "LLM 增强 DMDE": uavs},
                 {"标准 DMDE": targets, "LLM 增强 DMDE": targets}, dem_terrain=dem)

    print("\n✅ 实验完成。")


if __name__ == "__main__":
    main()
