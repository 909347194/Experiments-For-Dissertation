# -*- coding: utf-8 -*-
"""S2 fix validation figures (English labels to avoid CJK font issues).

用法（仓库根目录下执行）：
    python experiments/exp_ablation_study/validation/make_figures.py
    python experiments/exp_ablation_study/validation/make_figures.py \
        --scenario-dir experiments/exp_ablation_study/S2_srp_N10_M20 \
        --a0 A0_dmde --a1 A1_cr_control --out experiments/exp_ablation_study/validation

输出：<out>/s2_fix_validation.png
  (1) median + per-run convergence curves  (2) final fitness boxplot
  (3) A1 decision profile (LLM strategy picks + per-gen gmr_mode)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_curves(p: Path):
    data = json.loads(p.read_text(encoding="utf-8"))
    runs = []
    for r in data:
        curve = r["convergence_curve"]
        n = len(curve)
        step = 1 if n >= 500 else (5 if n >= 100 else max(1, 500 // max(1, n - 1)))
        best = [float("inf")] * 501
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
        runs.append({"seed": r["seed"], "final": float(r["best_fitness"]), "best": best,
                     "llm": r.get("llm_decisions", []), "gr": r.get("generation_records", [])})
    return runs


def main():
    ap = argparse.ArgumentParser(description="S2 验证对比图")
    ap.add_argument("--scenario-dir", default=str(REPO_ROOT / "experiments" / "exp_ablation_study" / "S2_srp_N10_M20"),
                    help="消融场景目录")
    ap.add_argument("--a0", default="A0_dmde")
    ap.add_argument("--a1", default="A1_cr_control")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "exp_ablation_study" / "validation"))
    args = ap.parse_args()

    base = Path(args.scenario_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a0 = load_curves(base / args.a0 / "results" / "ablation_results.json")
    a1 = load_curves(base / args.a1 / "results" / "ablation_results.json")
    gens = np.arange(1, 501)
    med0 = np.median([r["best"][1:] for r in a0], axis=0)
    med1 = np.median([r["best"][1:] for r in a1], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) convergence
    ax = axes[0]
    for r in a0:
        ax.plot(gens, r["best"][1:], color="#888888", alpha=0.35, lw=1)
    for r in a1:
        ax.plot(gens, r["best"][1:], color="#1f77b4", alpha=0.35, lw=1)
    ax.plot(gens, med0, color="#555555", lw=2.5, label="A0 (vanilla DMDE) median")
    ax.plot(gens, med1, color="#d62728", lw=2.5, label="A1 (fixed SC) median")
    ax.set_xlabel("Generation"); ax.set_ylabel("Best fitness (best-so-far)")
    ax.set_title("S2 convergence (median + per-run)")
    ax.legend(); ax.grid(alpha=0.3)

    # 2) final fitness
    ax = axes[1]
    f0 = [r["final"] for r in a0]; f1 = [r["final"] for r in a1]
    bp = ax.boxplot([f0, f1], tick_labels=["A0\nvanilla DMDE", "A1\nfixed SC"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("#cccccc"); bp["boxes"][1].set_facecolor("#ff9999")
    for i, vals in enumerate([f0, f1], 1):
        ax.scatter([i] * len(vals), vals, color="black", zorder=3, s=25)
    ax.set_ylabel("Final best fitness")
    ax.set_title("Final fitness\n(A1 median lower = better)")
    ax.grid(alpha=0.3, axis="y")

    # 3) A1 decision profile
    ax = axes[2]
    strat = Counter(); gmr = Counter()
    for r in a1:
        for d in r["llm"]:
            strat[d.get("parsed_decision", {}).get("strategy")] += 1
        for g in r["gr"]:
            gmr[g.get("gmr_mode")] += 1
    ax2 = ax.twinx()
    s_keys = list(strat.keys()); s_vals = [strat[k] for k in s_keys]
    ax.bar([f"S:{k}" for k in s_keys], s_vals, color="#1f77b4", alpha=0.85,
           label="LLM strategy picks")
    g_keys = list(gmr.keys()); g_vals = [gmr[k] for k in g_keys]
    ax2.bar([f"g:{k}" for k in g_keys], g_vals, color="#ff7f0e", alpha=0.45,
            label="per-gen gmr_mode")
    ax.set_title("A1 decision profile\n(forced reset 'on' = 0)")
    ax.set_ylabel("LLM strategy picks", color="#1f77b4")
    ax2.set_ylabel("per-gen gmr_mode count", color="#ff7f0e")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "s2_fix_validation.png", dpi=130)
    print("saved", out / "s2_fix_validation.png")
    print("strategy:", dict(strat), "gmr:", dict(gmr))


if __name__ == "__main__":
    main()
