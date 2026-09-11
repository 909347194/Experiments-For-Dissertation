# -*- coding: utf-8 -*-
"""crossover.py — 动态交叉率 CR 与混合差分策略（公式 3-9 / 3-10）

对应论文：
    公式 (3-9): 动态交叉率
        CR(x) = 1 - (log(x) / log(S))^ζ
        x = 1,2,...,S (当前代数), S 为总代数, ζ 为曲率指数。

    公式 (3-10): 混合差分策略产生新个体
        C'(i)_j(t) = C(r1)_j(t) + F·(C(r2)_j(t) - C(r3)_j(t)),
            if CR(t) >= rand[0,1]      (DE/rand/1 探索)
        C'(i)_j(t) = C(best)_j(t) + F·(C(r1)_j(t) + C(r2)_j(t)
                                         - C(r3)_j(t) - C(r4)_j(t)),
            if CR(t) < rand[0,1]       (DE/best/2 开发)

    当 CR 较大时偏向探索（rand/1），较小时偏向开发（best/2），
    实现搜索过程中探索与开发的动态平衡。
"""

from __future__ import annotations

import numpy as np


def dynamic_crossover_rate(
    current_gen: int,
    total_gens: int,
    zeta: int = 3,
) -> float:
    """计算动态交叉率 CR。

    对应公式 (3-9)。

    Args:
        current_gen: 当前代数（从 1 开始）。
        total_gens:  总迭代代数。
        zeta:        曲率变化指数 ζ（默认 3）。

    Returns:
        动态交叉率 CR。
    """
    if current_gen <= 1:
        return 1.0
    if current_gen >= total_gens:
        return 0.0
    return 1.0 - (np.log(current_gen) / np.log(total_gens)) ** zeta


def hybrid_differential(
    cost_vectors: np.ndarray,
    best_idx: int,
    f: float,
    cr: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """混合差分策略产生临时解向量。

    对应公式 (3-10)。

    Args:
        cost_vectors: 种群代价值矩阵，shape = (pop_size, gene_len)。
        best_idx:     当前最优个体的索引。
        f:            缩放因子 F。
        cr:           当前动态交叉率。
        rng:          随机数生成器。

    Returns:
        临时解向量，shape = (gene_len,)。
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_size, gene_len = cost_vectors.shape

    # 随机选择 4 个不同个体
    indices = rng.choice(pop_size, size=4, replace=False)
    r1, r2, r3, r4 = indices

    best = cost_vectors[best_idx]
    x_r1 = cost_vectors[r1]
    x_r2 = cost_vectors[r2]
    x_r3 = cost_vectors[r3]
    x_r4 = cost_vectors[r4]

    # 对每个基因位独立判断使用哪种策略
    rand_vals = rng.random(gene_len)
    use_rand = rand_vals <= cr  # CR >= rand → 用 rand/1

    # DE/rand/1 (探索)
    trial_rand = x_r1 + f * (x_r2 - x_r3)
    # DE/best/2 (开发)
    trial_best = best + f * (x_r1 + x_r2 - x_r3 - x_r4)

    return np.where(use_rand, trial_rand, trial_best)


def hybrid_differential_population(
    cost_vectors: np.ndarray,
    best_idx: int,
    f_values: np.ndarray,
    cr: float,
    rng: np.random.Generator | None = None,
    strategy: str | None = None,
) -> np.ndarray:
    """对整个种群执行混合差分（每个个体独立 F 值）。

    支持的策略：
    - None 或 "default": 原始 CR 切换（公式 3-10）
    - "rand/1": DE/rand/1 — x_r1 + F * (x_r2 - x_r3)
    - "best/2": DE/best/2 — best + F * (x_r1 + x_r2 - x_r3 - x_r4)

    Args:
        cost_vectors:   种群代价值矩阵。
        best_idx:       最优个体索引。
        f_values:       每个个体的缩放因子数组，shape = (pop_size,)。
        cr:             当前动态交叉率。
        rng:            随机数生成器。
        strategy:       变异策略。None 或 "default" 使用原始 CR 切换；
                        "rand/1" 或 "best/2" 统一应用于所有个体。

    Returns:
        试验向量矩阵，shape = (pop_size, gene_len)。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 规范化 strategy
    strat = strategy.lower().strip() if strategy else "default"
    if strat == "default":
        strat = None

    pop_size, gene_len = cost_vectors.shape
    trials = np.empty_like(cost_vectors)

    for i in range(pop_size):
        # 根据策略选择所需不同随机个体数量
        if strat == "rand/1":
            n_needed = 3
        else:
            # default / best/2 / None — 都需要 4 个
            n_needed = 4

        indices = rng.choice(pop_size, size=n_needed, replace=False)
        while i in indices:
            indices = rng.choice(pop_size, size=n_needed, replace=False)

        f = f_values[i]
        best = cost_vectors[best_idx]

        if strat is None:
            # 原始 CR 切换逻辑（公式 3-10）
            r1, r2, r3, r4 = indices[:4]
            x_r1 = cost_vectors[r1]
            x_r2 = cost_vectors[r2]
            x_r3 = cost_vectors[r3]
            x_r4 = cost_vectors[r4]

            rand_vals = rng.random(gene_len)
            use_rand = rand_vals <= cr

            trial_rand = x_r1 + f * (x_r2 - x_r3)
            trial_best = best + f * (x_r1 + x_r2 - x_r3 - x_r4)

            trials[i] = np.where(use_rand, trial_rand, trial_best)

        elif strat == "rand/1":
            # DE/rand/1 — 所有个体统一使用此策略
            r1, r2, r3 = indices[:3]
            trials[i] = cost_vectors[r1] + f * (cost_vectors[r2] - cost_vectors[r3])

        elif strat == "best/2":
            # DE/best/2 — 所有个体统一使用此策略
            r1, r2, r3, r4 = indices[:4]
            trials[i] = best + f * (
                cost_vectors[r1] + cost_vectors[r2]
                - cost_vectors[r3] - cost_vectors[r4]
            )

        else:
            # 未知策略，回退到默认 CR 切换
            r1, r2, r3, r4 = indices[:4]
            x_r1 = cost_vectors[r1]
            x_r2 = cost_vectors[r2]
            x_r3 = cost_vectors[r3]
            x_r4 = cost_vectors[r4]

            rand_vals = rng.random(gene_len)
            use_rand = rand_vals <= cr

            trial_rand = x_r1 + f * (x_r2 - x_r3)
            trial_best = best + f * (x_r1 + x_r2 - x_r3 - x_r4)

            trials[i] = np.where(use_rand, trial_rand, trial_best)

    return trials
