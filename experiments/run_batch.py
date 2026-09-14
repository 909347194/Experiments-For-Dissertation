# -*- coding: utf-8 -*-
"""run_batch.py — 批量运行实验 + 自动生成对比

用法：
    # 运行 N=M=10 实验（默认）
    python experiments/run_batch.py

    # 指定场景和运行次数
    python experiments/run_batch.py --scenario 01 --runs 30

    # 只跑基线
    python experiments/run_batch.py --scenario 01 --runs 30 --only baseline

    # 只跑 LLM
    python experiments/run_batch.py --scenario 01 --runs 30 --only llm

    # 跳过对比图
    python experiments/run_batch.py --no-compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 实验映射表
EXPERIMENTS = {
    "01": {
        "name": "N=M=10 平衡指派",
        "baseline": SCRIPT_DIR / "exp_dmde" / "exp_dmde_01" / "run.py",
        "llm": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "baseline_data": SCRIPT_DIR / "exp_dmde" / "exp_dmde_01" / "results" / "exp_dmde_01_data.json",
        "llm_data": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "results" / "exp_llm_dmde_01_data.json",
    },
    "04": {
        "name": "N=M=10 平衡指派（副本）",
        "baseline": SCRIPT_DIR / "exp_dmde" / "exp_dmde_04" / "run.py",
        "llm": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_04" / "run.py",
        "baseline_data": SCRIPT_DIR / "exp_dmde" / "exp_dmde_04" / "results" / "exp_dmde_04_data.json",
        "llm_data": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_04" / "results" / "exp_llm_dmde_04_data.json",
    },
}


def run_experiment(script: Path, n_runs: int, label: str) -> bool:
    """运行单个实验脚本。"""
    print(f"\n{'='*60}")
    print(f"▶ 运行: {label}")
    print(f"  脚本: {script}")
    print(f"  运行次数: {n_runs}")
    print(f"{'='*60}\n")

    env = {"EXP_N_RUNS": str(n_runs), "PATH": str(Path.home() / ".local" / "bin") + ":" + sys.path[0]}

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        env={**dict(__import__("os").environ), **env},
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ {label} 失败 (exit code {result.returncode})")
        return False

    print(f"\n✅ {label} 完成 ({elapsed:.1f}s)")
    return True


def run_comparison(baseline_data: Path, llm_data: Path) -> bool:
    """运行对比脚本。"""
    compare_script = SCRIPT_DIR / "compare_experiments.py"
    if not compare_script.exists():
        print("⚠️  对比脚本不存在，跳过")
        return False

    print(f"\n{'='*60}")
    print(f"▶ 生成对比报告")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(compare_script),
         "--baseline", str(baseline_data),
         "--llm", str(llm_data)],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="批量运行 DMDE 实验")
    parser.add_argument("--scenario", default="01", choices=list(EXPERIMENTS.keys()),
                        help="实验场景编号 (默认: 01)")
    parser.add_argument("--runs", type=int, default=2, help="每个实验运行次数 (默认: 2)")
    parser.add_argument("--only", choices=["baseline", "llm"], help="只运行指定实验")
    parser.add_argument("--no-compare", action="store_true", help="跳过对比图生成")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.scenario]
    print(f"\n🧪 实验计划: {exp['name']}")
    print(f"   运行次数: {args.runs}")
    print(f"   模式: {args.only or '全部'}")

    results = {}

    # 运行基线
    if args.only != "llm":
        ok = run_experiment(exp["baseline"], args.runs, f"DMDE 基线 ({exp['name']})")
        results["baseline"] = ok

    # 运行 LLM-DMDE
    if args.only != "baseline":
        ok = run_experiment(exp["llm"], args.runs, f"LLM-DMDE ({exp['name']})")
        results["llm"] = ok

    # 生成对比
    if not args.no_compare and results.get("baseline") and results.get("llm"):
        run_comparison(exp["baseline_data"], exp["llm_data"])

    # 总结
    print(f"\n{'='*60}")
    print(f"📊 批量实验完成")
    print(f"{'='*60}")
    for k, v in results.items():
        status = "✅" if v else "❌"
        print(f"  {status} {k}")


if __name__ == "__main__":
    main()
