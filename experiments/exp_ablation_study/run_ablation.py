# -*- coding: utf-8 -*-
"""run_ablation.py — 消融实验统一入口

用法：
    python run_ablation.py --runs 30
    python run_ablation.py --scenario S1 --runs 30
    python run_ablation.py --config A1 --runs 30
    python run_ablation.py --model Qwen/Qwen3.8-27B --runs 30
    python run_ablation.py --model Qwen/Qwen3.8-27B --fallback-model Qwen/Qwen3-14B --runs 30
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from analyze.main import main as analyze_main

SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR = <repo>/experiments/exp_ablation_study → 仓库根目录是 parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SCENARIOS = ["S1_balanced_N10_M10", "S2_srp_N10_M20"]
CONFIGS = ["A0_dmde", "A1_cr_control", "A1_coupled", "A1_nocr"]


def main():
    parser = argparse.ArgumentParser(description="消融实验统一入口")
    parser.add_argument("--runs", type=int, default=30, help="每组运行次数")
    parser.add_argument("--scenario", choices=["S1", "S2", "all"], default="all")
    parser.add_argument("--config", choices=["A0", "A1", "A1c", "A1n", "all"], default="all",
                        help="A0=Vanilla DMDE, A1=解耦SC, A1c=耦合CR, A1n=CR锁定")
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖 llm_config.yaml 中的 model（如 Qwen/Qwen3.8-27B）")
    parser.add_argument("--fallback-model", type=str, default=None,
                        help="覆盖 llm_config.yaml 中的 fallback_model")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [
        s for s in SCENARIOS if args.scenario in s
    ]
    config_map = {"A0": "A0_dmde", "A1": "A1_cr_control", "A1c": "A1_coupled", "A1n": "A1_nocr"}
    if args.config == "all":
        configs = CONFIGS
    else:
        configs = [config_map[args.config]] if args.config in config_map else []

    total = len(scenarios) * len(configs)
    print(f"消融实验：{len(scenarios)} 场景 × {len(configs)} 配置 = {total} 组，每组 {args.runs} runs")
    if args.model:
        print(f"模型覆盖: {args.model}")
    if args.fallback_model:
        print(f"回退模型: {args.fallback_model}")
    print(f"项目根目录: {PROJECT_ROOT}\n")

    idx = 0
    results_summary = []
    for scenario in scenarios:
        for config in configs:
            idx += 1
            run_py = SCRIPT_DIR / scenario / config / "run.py"
            if not run_py.exists():
                print(f"[{idx}/{total}] SKIP {scenario}/{config} (no run.py)")
                continue

            print(f"[{idx}/{total}] {scenario}/{config} ...")
            t0 = time.time()

            # 构造子进程环境变量
            env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)}
            if args.model:
                env["LLM_MODEL"] = args.model
            if args.fallback_model:
                env["LLM_FALLBACK_MODEL"] = args.fallback_model

            result = subprocess.run(
                [sys.executable, str(run_py), "--runs", str(args.runs)],
                cwd=str(run_py.parent),
                env=env,
            )
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
            print(f"  {status} ({elapsed:.1f}s)\n")
            results_summary.append((f"{scenario}/{config}", status, elapsed))

    # 汇总
    print("=" * 50)
    print("消融实验运行汇总")
    print("=" * 50)
    for name, status, elapsed in results_summary:
        print(f"  {name}: {status} ({elapsed:.1f}s)")

    print(f"\n全部完成。运行 python analyze.py 生成对比图表。")


if __name__ == "__main__":
    main()
    analyze_main()  # 运行 analyze.py 生成对比图表