# -*- coding: utf-8 -*-
"""S2 消融验证：search_controller 修复后 (A1) vs 纯 DMDE 基线 (A0)。

用法（仓库根目录下执行）：
    python experiments/exp_ablation_study/validation/validate_fix.py
    python experiments/exp_ablation_study/validation/validate_fix.py \
        --scenario-dir experiments/exp_ablation_study/S2_srp_N10_M20 \
        --a0 A0_dmde --a1 A1_cr_control --out experiments/exp_ablation_study/validation

修复核心（见 commit 2d8cca8）：
  - explore 的 gmr 由 "on"(强制全局重置) 改为 "off"(与 presets.py 设计一致)；
  - recover 成为唯一强制重置杠杆，并由 screen_strategy 护栏限制在"真正陷入绝境"时才放行；
  - prompt 收敛期默认 exploit/hold，recover 仅最后手段。

指标：
  1. 终值适应度（mean/median/std/min/max）A1 vs A0
  2. 配对种子胜率（按相同 seed 配对）
  3. 收敛速度（首次到达终值 0.5% 以内的代数）
  4. A1 策略分布 / gmr_mode 分布（证明全局重置已消除）
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# 仓库根目录：本文件位于 <repo>/experiments/exp_ablation_study/validation/validate_fix.py
REPO_ROOT = Path(__file__).resolve().parents[3]


def load_best_curves(results_path: Path) -> list[dict]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    runs = []
    for r in data:
        curve = r["convergence_curve"]
        n = len(curve)
        # 推断采样间隔：501 -> 每代；101 -> 每 5 代（脚本 bug：A0/A1 采样不一致，统一重建）
        if n >= 500:
            step = 1
        elif n >= 100:
            step = 5
        else:
            step = max(1, 500 // max(1, n - 1))
        # 重建 1..500 的 best-so-far（前向填充，best 单调不增）
        best = [float("inf")] * 501  # index 0 unused, 1..500
        running = float("inf")
        for i, v in enumerate(curve):
            gen = (i + 1) * step
            running = min(running, v)
            for g in range(max(1, gen - step + 1), gen + 1):
                if g <= 500:
                    best[g] = running
        for g in range(1, 501):
            if best[g] == float("inf"):
                best[g] = running
        runs.append({
            "seed": r["seed"],
            "final": float(r["best_fitness"]),
            "best": best,  # best[1..500]
            "raw_curve": curve,
            "llm_decisions": r.get("llm_decisions", []),
            "generation_records": r.get("generation_records", []),
        })
    return runs


def convergence_gen(best: list[float], eps: float = 0.005) -> int:
    """首次到达终值 (1+eps) 以内的代数（best 单调不增）。"""
    final = best[500]
    thr = final * (1.0 + eps)
    for g in range(1, 501):
        if best[g] <= thr:
            return g
    return 500


def summarize(runs: list[dict]) -> dict:
    finals = np.array([r["final"] for r in runs])
    return {
        "n": len(finals),
        "mean": float(finals.mean()),
        "median": float(np.median(finals)),
        "std": float(finals.std(ddof=0)),
        "min": float(finals.min()),
        "max": float(finals.max()),
        "conv_gen_mean": float(np.mean([convergence_gen(r["best"]) for r in runs])),
        "conv_gen_median": float(np.median([convergence_gen(r["best"]) for r in runs])),
    }


def paired_wins(a0: list[dict], a1: list[dict]) -> dict:
    by_seed = {r["seed"]: r for r in a1}
    wins = 0
    ties = 0
    losses = 0
    detail = []
    for r0 in a0:
        r1 = by_seed.get(r0["seed"])
        if not r1:
            continue
        d = r1["final"] - r0["final"]
        if d < -1:
            wins += 1
            verdict = "A1赢"
        elif d > 1:
            losses += 1
            verdict = "A0赢"
        else:
            ties += 1
            verdict = "平"
        detail.append((r0["seed"], r0["final"], r1["final"], d, verdict))
    return {"wins": wins, "ties": ties, "losses": losses, "detail": detail}


def strategy_profile(runs: list[dict]) -> dict:
    strat = Counter()
    gmr = Counter()
    recover_suppressed = 0
    for r in runs:
        for d in r["llm_decisions"]:
            pd = d.get("parsed_decision", {})
            strat[pd.get("strategy")] += 1
            if pd.get("recover_suppressed"):
                recover_suppressed += 1
        for g in r["generation_records"]:
            gmr[g.get("gmr_mode")] += 1
    return {"strategy": dict(strat), "gmr": dict(gmr), "recover_suppressed": recover_suppressed}


def main():
    ap = argparse.ArgumentParser(description="S2 验证：修复后 A1 vs A0 基线")
    ap.add_argument("--scenario-dir", default=str(REPO_ROOT / "experiments" / "exp_ablation_study" / "S2_srp_N10_M20"),
                    help="消融场景目录（含 A0_dmde / A1_cr_control 子目录）")
    ap.add_argument("--a0", default="A0_dmde", help="基线配置目录名")
    ap.add_argument("--a1", default="A1_cr_control", help="修复后配置目录名")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "exp_ablation_study" / "validation"),
                    help="结果/图表输出目录")
    args = ap.parse_args()

    base = Path(args.scenario_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a0 = load_best_curves(base / args.a0 / "results" / "ablation_results.json")
    a1 = load_best_curves(base / args.a1 / "results" / "ablation_results.json")

    s0 = summarize(a0)
    s1 = summarize(a1)
    pw = paired_wins(a0, a1)
    prof = strategy_profile(a1)

    print("=" * 64)
    print(f"S2 验证：修复后 A1 ({args.a1}) vs A0 ({args.a0})")
    print(f"场景目录: {base}")
    print("=" * 64)
    print(f"\n{'指标':<22}{'A0 (基线)':>16}{'A1 (修复)':>16}")
    print(f"{'runs':<22}{s0['n']:>16}{s1['n']:>16}")
    print(f"{'mean':<22}{s0['mean']:>16.2f}{s1['mean']:>16.2f}")
    print(f"{'median':<22}{s0['median']:>16.2f}{s1['median']:>16.2f}")
    print(f"{'std':<22}{s0['std']:>16.2f}{s1['std']:>16.2f}")
    print(f"{'min':<22}{s0['min']:>16.2f}{s1['min']:>16.2f}")
    print(f"{'max':<22}{s0['max']:>16.2f}{s1['max']:>16.2f}")
    print(f"{'收敛代数(mean)':<22}{s0['conv_gen_mean']:>16.1f}{s1['conv_gen_mean']:>16.1f}")
    print(f"{'收敛代数(median)':<22}{s0['conv_gen_median']:>16.1f}{s1['conv_gen_median']:>16.1f}")

    print("\n--- 配对种子对比 (同种子) ---")
    for seed, f0, f1, d, v in pw["detail"]:
        print(f"  seed {seed}: A0={f0:.2f}  A1={f1:.2f}  Δ={d:+.2f}  [{v}]")
    print(f"  A1胜 {pw['wins']} / 平 {pw['ties']} / A0胜 {pw['losses']}")

    print("\n--- A1 决策画像（证明全局重置已消除）---")
    print(f"  策略分布: {prof['strategy']}")
    print(f"  gmr_mode 分布(按代): {prof['gmr']}")
    forced_on = prof['gmr'].get('on', 0)
    print(f"  强制重置(on)代数: {forced_on}  (修复前每 explore/recover stage 整段为 on)")
    print(f"  recover 被护栏降级次数: {prof['recover_suppressed']}")

    # 保存结构化结果
    out_json = out / "summary.json"
    out_json.write_text(
        json.dumps({"A0": s0, "A1": s1, "paired": pw, "profile": prof},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n结构化结果已保存: {out_json}")
    return out_json


if __name__ == "__main__":
    main()
