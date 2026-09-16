# -*- coding: utf-8 -*-
"""A0: Vanilla DMDE（基线）"""
import argparse
import sys
from pathlib import Path

_ABSTRACTION_DIR = Path(__file__).resolve().parents[2]  # exp_ablation_study/
sys.path.insert(0, str(_ABSTRACTION_DIR.parents[2]))    # project root
sys.path.insert(0, str(_ABSTRACTION_DIR))

from core import build_scenario, run_dmde_baseline, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parent.parent
    cost_matrix, evaluator, n_uavs, n_targets, _ = build_scenario(scenario_dir)

    from shared_config import POP_SIZE, MAX_GENERATIONS, ZETA, DELTA, SEEDS

    print(f"A0 Vanilla DMDE | {n_uavs}U/{n_targets}T | {args.runs} runs")
    results = []
    for i, seed in enumerate(SEEDS[:args.runs]):
        print(f"  Run {i+1}/{args.runs} (seed={seed}) ...", end=" ", flush=True)
        r = run_dmde_baseline(seed, cost_matrix, evaluator, n_uavs, n_targets,
                              POP_SIZE, MAX_GENERATIONS, ZETA, DELTA)
        results.append(r)
        print(f"fitness={r['best_fitness']:.2f} time={r['total_time']:.1f}s")

    out = save_results(results, Path(__file__).resolve().parent)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()