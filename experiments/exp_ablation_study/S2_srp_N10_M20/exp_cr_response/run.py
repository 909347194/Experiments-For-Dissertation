# -*- coding: utf-8 -*-
"""CR Response Landscape 实验

扫描 CR ∈ {0.0, 0.1, ..., 1.0}，每个值 × 3 seeds × 1000 generations，
收集 DMDE 搜索行为全景指标。

输出:
  - results/cr_sweep_results.json
  - figures/cr_response_landscape.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# 路径设置
_SCRIPT_DIR = Path(__file__).resolve().parent
_SCENARIO_DIR = _SCRIPT_DIR.parent  # S2_srp_N10_M20
_PROJECT_ROOT = _SCENARIO_DIR.parents[2]
_ABLATION_DIR = _SCENARIO_DIR.parent

for p in [str(_PROJECT_ROOT), str(_ABLATION_DIR), str(_SCENARIO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core import build_scenario
from shared_config import POP_SIZE, MAX_GENERATIONS, ZETA, DELTA, SEEDS
from src.algorithms.algorithm_dmde.solvers.dmde_solver import DMDEConfig
from src.algorithms.algorithm_dmde.representation.encoder import PopulationEncoder
from src.algorithms.algorithm_dmde.representation.inverse_mapper import inverse_phi
from src.algorithms.algorithm_dmde.operators.crossover import hybrid_differential_population
from src.algorithms.algorithm_dmde.operators.scale_factor import dynamic_scale_factor_batch
from src.algorithms.algorithm_dmde.operators.extinction import should_extinct, apply_extinction, gmr_rate
from src.algorithms.algorithm_dmde.operators.mutation import mutate_population
from src.algorithms.algorithm_llm_enhanced_dmde.features.population_features import compute_diversity


def run_fixed_cr(
    seed: int,
    cr_fixed: float,
    cost_matrix: np.ndarray,
    evaluator: object,
    n_uavs: int,
    n_targets: int,
    pop_size: int,
    max_generations: int,
    zeta: int,
    delta: float,
) -> dict:
    """运行 DMDE with 固定 CR，收集详细搜索行为指标。"""
    rng = np.random.default_rng(seed)

    # 模型类型
    if n_uavs == n_targets:
        model_type = "balanced"
    elif n_uavs > n_targets:
        model_type = "overloaded"
    else:
        model_type = "srp"

    # 初始化种群
    encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
    population = encoder.generate(pop_size, seed=seed)

    best_idx = 0
    for i, ind in enumerate(population):
        ind.fitness = evaluator.evaluate(ind.assignment, cost_matrix, n_uavs=n_uavs).fitness
        if ind.fitness < population[best_idx].fitness:
            best_idx = i

    best_individual = population[best_idx].copy()
    cost_history = [best_individual.fitness]

    # 指标收集
    gmr_activations = 0
    total_offspring = 0
    total_accepted = 0
    total_improvements = 0
    diversity_history = []
    fitness_history = [best_individual.fitness]
    cr_history = [cr_fixed]  # 固定 CR
    f_history = []
    acceptance_per_gen = []
    gmr_per_gen = []

    t_start = time.time()

    for gen in range(1, max_generations + 1):
        # 固定 CR（不用 dynamic_crossover_rate）
        cr = cr_fixed

        # F 值
        f_values = dynamic_scale_factor_batch(cr, pop_size, rng)
        f_history.append(float(np.mean(f_values)))

        # 代价值
        cost_vectors = np.array([ind.cost_vector for ind in population])

        # 混合变异 — 直接调用 hybrid_differential_population 传入固定 CR
        trial_vectors = hybrid_differential_population(
            cost_vectors, best_idx, f_values, cr, rng
        )

        temperature = 1.0 - gen / max_generations

        # 贪婪选择
        gen_accepted = 0
        gen_improvements = 0
        for i in range(pop_size):
            child = inverse_phi(
                trial_vectors[i], cost_matrix, n_uavs, n_targets, model_type,
                rng=rng, temperature=temperature,
            )
            child.fitness = evaluator.evaluate(child.assignment, cost_matrix, n_uavs=n_uavs).fitness
            total_offspring += 1

            if child.fitness < population[i].fitness:
                population[i] = child
                gen_accepted += 1
                total_accepted += 1
                if child.fitness < best_individual.fitness:
                    best_individual = child.copy()
                    best_idx = i
                    gen_improvements += 1
                    total_improvements += 1

        acceptance_per_gen.append(gen_accepted / pop_size)

        # GMR 灭绝
        gmr_triggered = should_extinct(cr, delta, rng)
        if gmr_triggered:
            gmr_activations += 1
            fitness_arr = np.array([ind.fitness for ind in population])
            new_cv, survived = apply_extinction(
                fitness_arr, cost_vectors, best_idx,
                cost_matrix, n_uavs, n_targets, model_type, rng=rng,
            )
            for i in range(pop_size):
                if i not in survived:
                    population[i] = inverse_phi(
                        new_cv[i], cost_matrix, n_uavs, n_targets, model_type, rng=rng,
                    )
                    population[i].fitness = evaluator.evaluate(
                        population[i].assignment, cost_matrix, n_uavs=n_uavs,
                    ).fitness
        gmr_per_gen.append(1 if gmr_triggered else 0)

        cost_history.append(best_individual.fitness)
        fitness_history.append(best_individual.fitness)

        # 多样性（每 10 代采样一次，减少开销）
        if gen % 10 == 0 or gen == 1:
            div = compute_diversity(population)
            diversity_history.append({"gen": gen, "diversity": round(float(div), 6)})

    elapsed = time.time() - t_start

    # 滑动窗口 acceptance rate（窗口=50 代）
    window = 50
    acceptance_smoothed = []
    for i in range(len(acceptance_per_gen)):
        lo = max(0, i - window + 1)
        acceptance_smoothed.append(round(float(np.mean(acceptance_per_gen[lo:i+1])), 4))

    return {
        "seed": seed,
        "cr": cr_fixed,
        "best_fitness": round(float(best_individual.fitness), 2),
        "convergence_curve": [round(float(f), 2) for f in cost_history],
        "fitness_history": [round(float(f), 2) for f in fitness_history],
        # 核心指标
        "gmr_activations": gmr_activations,
        "gmr_rate": round(gmr_activations / max_generations, 4),
        "acceptance_rate_total": round(total_accepted / total_offspring, 4) if total_offspring > 0 else 0.0,
        "acceptance_rate_curve": acceptance_smoothed,
        "total_improvements": total_improvements,
        "improvement_rate": round(total_improvements / max_generations, 4),
        # 多样性
        "diversity_curve": diversity_history,
        "diversity_final": diversity_history[-1]["diversity"] if diversity_history else 0.0,
        "diversity_mean": round(float(np.mean([d["diversity"] for d in diversity_history])), 6),
        # 收敛特征
        "init_fitness": round(float(cost_history[0]), 2),
        "improvement_pct": round((cost_history[0] - cost_history[-1]) / cost_history[0] * 100, 2),
        "convergence_gen_95": _find_convergence_gen(cost_history, 0.95),
        "convergence_gen_99": _find_convergence_gen(cost_history, 0.99),
        # F 值
        "f_mean": round(float(np.mean(f_history)), 4),
        "f_std": round(float(np.std(f_history)), 4),
        # GMR per gen
        "gmr_per_gen": gmr_per_gen,
        # 时间
        "total_time": round(elapsed, 2),
    }


def _find_convergence_gen(cost_history: list[float], threshold: float) -> int:
    """找到达到 threshold 比例改善的代数。"""
    init, final = cost_history[0], cost_history[-1]
    if init == final:
        return 0
    target = init - threshold * (init - final)
    for g, f in enumerate(cost_history):
        if f <= target:
            return g
    return len(cost_history) - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="每个 CR 值的 seed 数")
    parser.add_argument("--cr-min", type=float, default=0.0)
    parser.add_argument("--cr-max", type=float, default=1.0)
    parser.add_argument("--cr-step", type=float, default=0.1)
    args = parser.parse_args()

    scenario_dir = _SCENARIO_DIR
    cost_matrix, evaluator, n_uavs, n_targets, _ = build_scenario(scenario_dir)

    cr_values = list(np.arange(args.cr_min, args.cr_max + args.cr_step / 2, args.cr_step))
    cr_values = [round(cr, 1) for cr in cr_values]
    base_seeds = SEEDS[:args.seeds]

    total_runs = len(cr_values) * len(base_seeds)
    print(f"CR Response Landscape | {n_uavs}U/{n_targets}T")
    print(f"  CR values: {cr_values}")
    print(f"  Seeds: {base_seeds}")
    print(f"  Total runs: {total_runs}")
    print()

    results = []
    run_idx = 0

    for cr in cr_values:
        cr_results = []
        for seed in base_seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] CR={cr:.1f} seed={seed} ...", end=" ", flush=True)
            r = run_fixed_cr(
                seed=seed,
                cr_fixed=cr,
                cost_matrix=cost_matrix,
                evaluator=evaluator,
                n_uavs=n_uavs,
                n_targets=n_targets,
                pop_size=POP_SIZE,
                max_generations=MAX_GENERATIONS,
                zeta=ZETA,
                delta=DELTA,
            )
            cr_results.append(r)
            print(f"fitness={r['best_fitness']:.2f} "
                  f"GMR={r['gmr_activations']} "
                  f"acc={r['acceptance_rate_total']:.3f} "
                  f"div={r['diversity_final']:.4f} "
                  f"{r['total_time']:.1f}s")
        results.append({"cr": cr, "runs": cr_results})

    # 保存
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cr_sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\n结果已保存: {out_path}")

    # 画图
    plot_cr_response(results, _SCRIPT_DIR / "figures")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def plot_cr_response(results: list[dict], fig_dir: Path):
    """画 CR 响应全景图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    cr_values = [r["cr"] for r in results]

    # 汇总每个 CR 的指标
    def mean_std(runs, key):
        vals = [r[key] for r in runs]
        return np.mean(vals), np.std(vals)

    fitness_mean, fitness_std = [], []
    gmr_mean, gmr_std = [], []
    acc_mean, acc_std = [], []
    div_mean, div_std = [], []
    improve_mean, improve_std = [], []
    conv95_mean, conv95_std = [], []

    for r in results:
        m, s = mean_std(r["runs"], "best_fitness")
        fitness_mean.append(m); fitness_std.append(s)
        m, s = mean_std(r["runs"], "gmr_activations")
        gmr_mean.append(m); gmr_std.append(s)
        m, s = mean_std(r["runs"], "acceptance_rate_total")
        acc_mean.append(m); acc_std.append(s)
        m, s = mean_std(r["runs"], "diversity_final")
        div_mean.append(m); div_std.append(s)
        m, s = mean_std(r["runs"], "total_improvements")
        improve_mean.append(m); improve_std.append(s)
        m, s = mean_std(r["runs"], "convergence_gen_95")
        conv95_mean.append(m); conv95_std.append(s)

    # ---- Figure 1: CR → Final Objective ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(cr_values, fitness_mean, yerr=fitness_std,
                fmt='o-', color='#2196F3', capsize=4, linewidth=2, markersize=8,
                label='Final Fitness (mean ± std)')
    ax.set_xlabel('CR (fixed)', fontsize=14)
    ax.set_ylabel('Final Fitness', fontsize=14)
    ax.set_title(f'CR → Final Objective ({results[0]["runs"][0].get("convergence_curve", [0])[0] and "S2 15U/30T" or "?"})', fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(cr_values)
    fig.tight_layout()
    fig.savefig(fig_dir / "cr_response_fitness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图1: {fig_dir / 'cr_response_fitness.png'}")

    # ---- Figure 2: CR → 5 维搜索行为 ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle('CR → DMDE Search Behavior Landscape (S2 15U/30T)', fontsize=16, y=0.98)

    # 2a: GMR activations
    ax = axes[0, 0]
    ax.errorbar(cr_values, gmr_mean, yerr=gmr_std,
                fmt='s-', color='#F44336', capsize=4, linewidth=2, markersize=7)
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='δ=0.3')
    ax.set_xlabel('CR'); ax.set_ylabel('GMR Activations (out of 1000 gens)')
    ax.set_title('GMR Extinction Activations')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    # 2b: Acceptance rate
    ax = axes[0, 1]
    ax.errorbar(cr_values, acc_mean, yerr=acc_std,
                fmt='^-', color='#4CAF50', capsize=4, linewidth=2, markersize=7)
    ax.set_xlabel('CR'); ax.set_ylabel('Acceptance Rate')
    ax.set_title('Offspring Acceptance Rate')
    ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    # 2c: Diversity
    ax = axes[1, 0]
    ax.errorbar(cr_values, div_mean, yerr=div_std,
                fmt='D-', color='#FF9800', capsize=4, linewidth=2, markersize=7)
    ax.set_xlabel('CR'); ax.set_ylabel('Final Diversity')
    ax.set_title('Population Diversity (final)')
    ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    # 2d: Best improvement
    ax = axes[1, 1]
    ax.errorbar(cr_values, improve_mean, yerr=improve_std,
                fmt='v-', color='#9C27B0', capsize=4, linewidth=2, markersize=7)
    ax.set_xlabel('CR'); ax.set_ylabel('Improvement Count')
    ax.set_title('Best Improvements (total over 1000 gens)')
    ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    # 2e: Convergence speed (95%)
    ax = axes[2, 0]
    ax.errorbar(cr_values, conv95_mean, yerr=conv95_std,
                fmt='p-', color='#00BCD4', capsize=4, linewidth=2, markersize=7)
    ax.set_xlabel('CR'); ax.set_ylabel('Generation')
    ax.set_title('Convergence Speed (95% improvement)')
    ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    # 2f: Strategy ratio (DE/rand/1 vs DE/best/2)
    # CR 越高 → rand/1 (探索) 比例越高
    ax = axes[2, 1]
    explore_ratio = cr_values  # 理论上 CR = P(rand/1)
    ax.bar(cr_values, explore_ratio, width=0.08, color='#607D8B', alpha=0.7, label='P(DE/rand/1) = CR')
    ax.bar(cr_values, [1-c for c in cr_values], width=0.08, bottom=explore_ratio,
           color='#E91E63', alpha=0.7, label='P(DE/best/2) = 1-CR')
    ax.set_xlabel('CR'); ax.set_ylabel('Strategy Probability')
    ax.set_title('Mutation Strategy Ratio (theoretical)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(cr_values)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_dir / "cr_response_landscape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图2: {fig_dir / 'cr_response_landscape.png'}")

    # ---- Figure 3: 收敛曲线对比 ----
    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        cr = r["cr"]
        # 取第一个 seed 的收敛曲线
        cc = r["runs"][0]["convergence_curve"]
        color = cmap(cr)  # CR 0→1 映射到颜色
        ax.plot(cc, color=color, linewidth=1.5, alpha=0.8, label=f'CR={cr:.1f}')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Best Fitness', fontsize=13)
    ax.set_title('Convergence Curves by CR (seed=42)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cr_convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图3: {fig_dir / 'cr_convergence_curves.png'}")

    # ---- Figure 4: Acceptance rate 时序 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(results):
        cr = r["cr"]
        acc_curve = r["runs"][0]["acceptance_rate_curve"]
        color = cmap(cr)
        ax.plot(acc_curve, color=color, linewidth=1.2, alpha=0.7, label=f'CR={cr:.1f}')
    ax.set_xlabel('Generation', fontsize=13)
    ax.set_ylabel('Acceptance Rate (window=50)', fontsize=13)
    ax.set_title('Acceptance Rate Over Time by CR (seed=42)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "cr_acceptance_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图4: {fig_dir / 'cr_acceptance_timeline.png'}")

    print(f"\n所有图表已保存到: {fig_dir}")


if __name__ == "__main__":
    main()