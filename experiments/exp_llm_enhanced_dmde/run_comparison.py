# -*- coding: utf-8 -*-
"""run_comparison.py — 统一入口：同时运行 DMDE 基线和 LLM-DMDE，输出对比结果。

用法：
    # 默认：N=M=10 场景
    python run_comparison.py

    # 指定场景
    python run_comparison.py --scenario nm   # N=M
    python run_comparison.py --scenario ngt  # N>M
    python run_comparison.py --scenario nlt  # N<M

    # 指定运行次数
    python run_comparison.py --runs 30

    # 只跑一个
    python run_comparison.py --only baseline
    python run_comparison.py --only llm
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 场景映射
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIO_MAP = {
    "nm": {
        "name": "N=M 平衡指派",
        "baseline": PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_01" / "run.py",
        "llm": PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "baseline_id": "dmde_01",
        "llm_id": "llm_dmde_01",
    },
    "ngt": {
        "name": "N>M 非平衡指派",
        "baseline": PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_02" / "run.py",
        "llm": PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_02" / "run.py",
        "baseline_id": "dmde_02",
        "llm_id": "llm_dmde_02",
    },
    "nlt": {
        "name": "N<M 非平衡指派",
        "baseline": PROJECT_ROOT / "experiments" / "exp_dmde" / "exp_dmde_03" / "run.py",
        "llm": PROJECT_ROOT / "experiments" / "exp_llm_enhanced_dmde" / "exp_llm_dmde_03" / "run.py",
        "baseline_id": "dmde_03",
        "llm_id": "llm_dmde_03",
    },
}

COMPARE_SCRIPT = SCRIPT_DIR / "exp_llm_dmde_01" / "compare_experiments.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(script: Path, n_runs: int, label: str) -> bool:
    """运行单个实验脚本。"""
    if not script.exists():
        print(f"⚠️  脚本不存在: {script}")
        return False

    import os
    env = {**os.environ, "EXP_N_RUNS": str(n_runs)}

    print(f"\n{'=' * 60}")
    print(f"▶ {label}")
    print(f"  脚本: {script}")
    print(f"  运行次数: {n_runs}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ {label} 失败 (exit={result.returncode})")
        return False
    print(f"\n✅ {label} 完成 ({elapsed:.1f}s)")
    return True


def run_comparison(experiment_ids: list[str]) -> bool:
    """调用 compare_experiments.py 进行对比。"""
    if not COMPARE_SCRIPT.exists():
        print("⚠️  对比脚本不存在")
        return False

    print(f"\n{'=' * 60}")
    print(f"▶ 生成对比报告 ({len(experiment_ids)} 个实验)")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT), "--experiments"] + experiment_ids,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="统一入口：运行 DMDE 基线 + LLM-DMDE 对比实验")
    parser.add_argument("--scenario", default="nm", choices=list(SCENARIO_MAP.keys()),
                        help="场景选择: nm (N=M), ngt (N>M), nlt (N<M) (默认: nm)")
    parser.add_argument("--runs", type=int, default=2, help="每个实验运行次数 (默认: 2)")
    parser.add_argument("--only", type=str, default=None, choices=["baseline", "llm"],
                        help="只运行其中一个: baseline / llm")
    parser.add_argument("--no-compare", action="store_true", help="跳过对比报告生成")

    args = parser.parse_args()

    scenario = SCENARIO_MAP[args.scenario]
    print(f"\n🧪 场景: {scenario['name']} ({args.scenario})")

    results = {}

    # 运行基线
    if args.only != "llm":
        results["baseline"] = run_experiment(
            scenario["baseline"], args.runs, f"DMDE 基线 ({scenario['name']})"
        )

    # 运行 LLM-DMDE
    if args.only != "baseline":
        results["llm"] = run_experiment(
            scenario["llm"], args.runs, f"LLM-DMDE ({scenario['name']})"
        )

    # 生成对比报告
    if not args.no_compare:
        experiment_ids = []
        if results.get("baseline"):
            experiment_ids.append(scenario["baseline_id"])
        if results.get("llm"):
            experiment_ids.append(scenario["llm_id"])

        if len(experiment_ids) >= 2:
            run_comparison(experiment_ids)
        else:
            print("\n⚠️  至少需要两个实验结果才能生成对比报告")

    # 总结
    print(f"\n{'=' * 60}")
    print(f"📊 对比实验完成 — {scenario['name']}")
    if results.get("baseline"):
        print(f"  ✅ 基线: 完成")
    if results.get("llm"):
        print(f"  ✅ LLM-DMDE: 完成")
    print(f"  对比结果: {SCRIPT_DIR / 'exp_llm_dmde_01' / 'comparison_results'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
