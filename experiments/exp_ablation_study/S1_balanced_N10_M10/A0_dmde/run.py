# -*- coding: utf-8 -*-
"""A0: Vanilla DMDE（基线）

用法：python run.py --runs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 路径设置
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))  # project root

from shared_config import (
    N_UAVS, N_TARGETS, MODEL_TYPE, POP_SIZE, MAX_GENERATIONS,
    ZETA, DELTA, SEEDS, COST_MATRIX,
)


def run_single(seed: int) -> dict:
    """单次运行。"""
    from src.algorithms.algorithm_dmde.solvers.dmde_solver import DMDESolver, DMDEConfig

    cfg = DMDEConfig(
        pop_size=POP_SIZE,
        max_generations=MAX_GENERATIONS,
        zeta=ZETA,
        delta=DELTA,
        seed=seed,
    )
    solver = DMDESolver(cfg)

    t_start = time.time()
    result = solver.solve(COST_MATRIX, N_UAVS, N_TARGETS)
    t_total = time.time() - t_start

    return {
        "seed": seed,
        "best_fitness": result.best_fitness,
        "total_time": round(t_total, 2),
        "dmde_time": round(t_total, 2),
        "llm_time": 0.0,
        "llm_call_count": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    results = []
    for i, seed in enumerate(SEEDS[:args.runs]):
        print(f"  Run {i+1}/{args.runs} (seed={seed}) ...")
        r = run_single(seed)
        results.append(r)
        print(f"    fitness={r['best_fitness']:.2f} time={r['total_time']:.1f}s")

    # 保存结果
    out = SCRIPT_DIR / "results" / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()