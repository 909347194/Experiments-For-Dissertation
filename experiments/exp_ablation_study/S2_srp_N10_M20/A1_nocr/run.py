# -*- coding: utf-8 -*-
"""A1/A2/A3: LLM-DMDE 消融实验

根据 config/llm_config.yaml 中的 modules 字段自动判断启用哪些模块。
"""
import argparse
import sys
from pathlib import Path

_ABSTRACTION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ABSTRACTION_DIR.parents[1]))
sys.path.insert(0, str(_ABSTRACTION_DIR))

from core import build_scenario, run_llm_dmde, save_results, load_modules_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    scenario_dir = Path(__file__).resolve().parent.parent
    cost_matrix, evaluator, n_uavs, n_targets, _ = build_scenario(scenario_dir)

    from shared_config import POP_SIZE, MAX_GENERATIONS, ZETA, DELTA, SEEDS

    llm_config_path = Path(__file__).resolve().parent / "config" / "llm_config.yaml"
    modules = load_modules_config(str(llm_config_path))
    module_names = [k for k, v in modules.items() if v.get("enabled", True)]

    print(f"LLM-DMDE [{'+'.join(module_names)}] | {n_uavs}U/{n_targets}T | {args.runs} runs")
    print(f"  LLM config: {llm_config_path}")

    results = []
    for i, seed in enumerate(SEEDS[:args.runs]):
        print(f"  Run {i+1}/{args.runs} (seed={seed}) ...", end=" ", flush=True)
        r = run_llm_dmde(seed, cost_matrix, evaluator, n_uavs, n_targets,
                         POP_SIZE, MAX_GENERATIONS, ZETA, DELTA,
                         llm_config_path, modules)
        results.append(r)
        print(f"fitness={r['best_fitness']:.2f} total={r['total_time']:.1f}s "
              f"llm={r['llm_time']:.1f}s calls={r['llm_call_count']}")

    out = save_results(results, Path(__file__).resolve().parent)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()