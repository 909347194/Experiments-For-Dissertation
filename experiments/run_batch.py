# -*- coding: utf-8 -*-
"""run_batch.py — 批量运行实验 + 消融实验 + 自动生成对比

用法：
    # ── 场景对比（原功能）──
    python run_batch.py --scenario 01 --runs 30
    python run_batch.py --scenario 01 --only baseline

    # ── 消融实验 ──
    python run_batch.py --ablation --runs 30           # 跑所有消融配置
    python run_batch.py --ablation --only sc --runs 30  # 只跑 search_controller

    # ── 多算法对比 ──
    python run_batch.py --experiments dmde_01 llm_dmde_01 --runs 30

    # ── 跳过对比 ──
    python run_batch.py --ablation --no-compare
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 实验注册表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 基线场景对比
SCENARIO_EXPERIMENTS = {
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

# 消融实验配置
ABLATION_EXPERIMENTS = {
    "vanilla": {
        "name": "LLM-DMDE (vanilla, 无 LLM 模块)",
        "script": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "env_overrides": {"EXP_MODULES": "{}"},
        "tags": ["llm", "ablation", "vanilla"],
    },
    "sc": {
        "name": "LLM-DMDE (仅 search_controller)",
        "script": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "env_overrides": {"EXP_MODULES": '{"search_controller": {"enabled": true, "interval": 100}}'},
        "tags": ["llm", "ablation", "search_controller"],
    },
    "pi": {
        "name": "LLM-DMDE (仅 population_init)",
        "script": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "env_overrides": {"EXP_MODULES": '{"population_init": {"enabled": true}}'},
        "tags": ["llm", "ablation", "population_init"],
    },
    "full": {
        "name": "LLM-DMDE (全部模块)",
        "script": SCRIPT_DIR / "exp_llm_enhanced_dmde" / "exp_llm_dmde_01" / "run.py",
        "env_overrides": {"EXP_MODULES": '{"population_init": {"enabled": true}, "search_controller": {"enabled": true, "interval": 100}}'},
        "tags": ["llm", "ablation", "full"],
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 运行器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(
    script: Path,
    n_runs: int,
    label: str,
    env_overrides: dict[str, str] | None = None,
) -> bool:
    """运行单个实验脚本。"""
    print(f"\n{'='*60}")
    print(f"▶ {label}")
    print(f"  脚本: {script}")
    print(f"  运行次数: {n_runs}")
    if env_overrides:
        for k, v in env_overrides.items():
            print(f"  {k}={v}")
    print(f"{'='*60}\n")

    import os
    env = {**os.environ, "EXP_N_RUNS": str(n_runs)}
    if env_overrides:
        env.update(env_overrides)

    t0 = time.time()
    result = subprocess.run([sys.executable, str(script)], env=env, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n❌ {label} 失败 (exit={result.returncode})")
        return False
    print(f"\n✅ {label} 完成 ({elapsed:.1f}s)")
    return True


def run_comparison(experiment_ids: list[str]) -> bool:
    """调用 compare_experiments.py 进行多算法对比。"""
    compare_script = SCRIPT_DIR / "compare_experiments.py"
    if not compare_script.exists():
        print("⚠️  对比脚本不存在")
        return False

    print(f"\n{'='*60}")
    print(f"▶ 生成对比报告 ({len(experiment_ids)} 个实验)")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, str(compare_script), "--experiments"] + experiment_ids,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 消融实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ablation(args: argparse.Namespace) -> dict[str, bool]:
    """运行消融实验。"""
    # 确定要跑哪些配置
    if args.only:
        configs = {args.only: ABLATION_EXPERIMENTS[args.only]}
    else:
        configs = ABLATION_EXPERIMENTS

    print(f"\n🧪 消融实验")
    print(f"   配置: {', '.join(configs.keys())}")
    print(f"   运行次数: {args.runs}")

    results = {}
    for key, exp in configs.items():
        ok = run_experiment(exp["script"], args.runs, exp["name"], exp["env_overrides"])
        results[f"ablation_{key}"] = ok

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="批量运行实验")

    # 模式选择
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--scenario", default=None, choices=list(SCENARIO_EXPERIMENTS.keys()),
                      help="场景对比模式")
    mode.add_argument("--ablation", action="store_true", help="消融实验模式")
    mode.add_argument("--experiments", type=str, nargs="*", default=None,
                      help="指定实验 ID 列表运行")

    # 通用选项
    parser.add_argument("--runs", type=int, default=2, help="每个实验运行次数 (默认: 2)")
    parser.add_argument("--only", type=str, default=None,
                        help="场景模式: baseline/llm | 消融模式: vanilla/sc/pi/full")
    parser.add_argument("--no-compare", action="store_true", help="跳过对比图生成")

    args = parser.parse_args()

    # ── 消融实验模式 ──────────────────────────────────────────
    if args.ablation:
        results = run_ablation(args)

        if not args.no_compare:
            # 自动注册消融结果到索引
            try:
                from registry import register
            except ImportError:
                sys.path.insert(0, str(SCRIPT_DIR))
                from registry import register

            # 注册基线（如果存在）
            baseline_data = SCENARIO_EXPERIMENTS["01"]["baseline_data"]
            if baseline_data.exists():
                register("dmde_01", "DMDE 基线 N=M=10",
                         "exp_dmde/exp_dmde_01/results/exp_dmde_01_data.json",
                         tags=["baseline", "dmde", "10u10t"])

            # 对比
            ablation_ids = [k for k in results if results[k]]
            all_ids = ["dmde_01"] + ablation_ids if "dmde_01" in [e["id"] for e in
                          __import__("registry", fromlist=["list_experiments"]).list_experiments()
                          ] else ablation_ids
            if len(all_ids) >= 2:
                run_comparison(all_ids)

    # ── 场景对比模式 ──────────────────────────────────────────
    elif args.scenario:
        exp = SCENARIO_EXPERIMENTS[args.scenario]
        print(f"\n🧪 场景对比: {exp['name']}")
        results = {}

        if args.only != "llm":
            results["baseline"] = run_experiment(exp["baseline"], args.runs, f"DMDE 基线")

        if args.only != "baseline":
            results["llm"] = run_experiment(exp["llm"], args.runs, f"LLM-DMDE")

        if not args.no_compare and results.get("baseline") and results.get("llm"):
            run_comparison(["dmde_01", "llm_dmde_01"])

    # ── 指定实验列表模式 ──────────────────────────────────────
    elif args.experiments:
        print(f"\n🧪 指定实验: {args.experiments}")
        results = {}
        for eid in args.experiments:
            # 在场景表中查找
            if eid in SCENARIO_EXPERIMENTS:
                exp = SCENARIO_EXPERIMENTS[eid]
                results[eid] = run_experiment(exp.get("llm") or exp["baseline"], args.runs, eid)
            # 在消融表中查找
            elif eid.replace("ablation_", "") in ABLATION_EXPERIMENTS:
                key = eid.replace("ablation_", "")
                exp = ABLATION_EXPERIMENTS[key]
                results[eid] = run_experiment(exp["script"], args.runs, exp["name"], exp["env_overrides"])
            else:
                print(f"  ⚠️ 未知实验 ID: {eid}")

        if not args.no_compare and sum(results.values()) >= 2:
            run_comparison([k for k, v in results.items() if v])

    # ── 默认：跑场景01 ────────────────────────────────────────
    else:
        args.scenario = "01"
        args.only = None
        main()

    # 总结
    print(f"\n{'='*60}")
    print(f"📊 批量实验完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
