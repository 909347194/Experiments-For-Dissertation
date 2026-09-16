# -*- coding: utf-8 -*-
"""A1/A2/A3: LLM-DMDE 消融实验 run.py

自动根据 config/llm_config.yaml 中的 modules 字段判断启用哪些模块。
用法：python run.py --runs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))  # project root

from shared_config import (
    N_UAVS, N_TARGETS, MODEL_TYPE, POP_SIZE, MAX_GENERATIONS,
    ZETA, DELTA, SEEDS, COST_MATRIX,
)


def run_single(seed: int, llm_config_path: str) -> dict:
    """单次运行，记录分口径时间。"""
    from src.algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
        LLMEnhancedDMDESolver, LLMEnhancedDMDEConfig,
    )

    cfg = LLMEnhancedDMDEConfig(
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        zeta=ZETA,
        delta=DELTA,
        seed=seed,
        llm_config_path=llm_config_path,
        modules=_load_modules_config(llm_config_path),
    )
    solver = LLMEnhancedDMDESolver(cfg)

    t_total_start = time.time()
    result = solver.solve(COST_MATRIX, N_UAVS, N_TARGETS)
    t_total = time.time() - t_total_start

    # 从 result.extra 提取 LLM 时间（solver 已记录）
    extra = result.extra or {}
    llm_time = extra.get("llm_time", 0.0)
    llm_calls = extra.get("llm_call_count", 0)

    return {
        "seed": seed,
        "best_fitness": result.best_fitness,
        "total_time": round(t_total, 2),
        "dmde_time": round(t_total - llm_time, 2),
        "llm_time": round(llm_time, 2),
        "llm_call_count": llm_calls,
    }


def _load_modules_config(llm_config_path: str) -> dict:
    """从 llm_config.yaml 读取 modules 配置。"""
    import yaml
    with open(llm_config_path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("modules", {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    llm_config = str(SCRIPT_DIR / "config" / "llm_config.yaml")
    print(f"LLM config: {llm_config}")

    results = []
    for i, seed in enumerate(SEEDS[:args.runs]):
        print(f"  Run {i+1}/{args.runs} (seed={seed}) ...")
        r = run_single(seed, llm_config)
        results.append(r)
        print(f"    fitness={r['best_fitness']:.2f} "
              f"total={r['total_time']:.1f}s llm={r['llm_time']:.1f}s calls={r['llm_call_count']}")

    out = SCRIPT_DIR / "results" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()