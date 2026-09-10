# -*- coding: utf-8 -*-
"""mutation.py — 混合变异策略

职责：
    在 DMDE 进化过程中，对种群个体执行变异操作。
    结合动态交叉率和混合差分策略（公式 3-10），在探索
    (DE/rand/1) 和开发 (DE/best/2) 之间动态平衡。

对应论文：
    第 3 章 3.4.2 节 —— 混合进化策略的 DMDE 算法。
"""

from __future__ import annotations

import numpy as np

from .crossover import dynamic_crossover_rate, hybrid_differential_population
from .scale_factor import dynamic_scale_factor_batch


def mutate_population(
    cost_vectors: np.ndarray,
    best_idx: int,
    current_gen: int,
    total_gens: int,
    zeta: int = 3,
    rng=None,
    cr: float | None = None,
    f_scale: float | None = None,
    strategy: str | None = None,
    fitness_values: np.ndarray | None = None,
) -> np.ndarray:
    """对整个种群执行混合变异。

    流程：
    1. 计算当前代的动态交叉率 CR (公式 3-9)，或使用传入的 cr。
    2. 为每个个体计算动态缩放因子 F (公式 3-11)，或使用传入的 f_scale。
    3. 执行混合差分策略 (公式 3-10)，或使用传入的 strategy。

    Args:
        cost_vectors:   种群代价值矩阵。
        best_idx:       最优个体索引。
        current_gen:    当前代数。
        total_gens:     总代数。
        zeta:           曲率指数。
        rng:            随机数生成器。
        cr:             覆盖 CR（None = 由公式计算）。
        f_scale:        覆盖 F（None = 由公式计算）。
        strategy:       覆盖变异策略（None = 原始 CR 切换）。
        fitness_values: 种群适应度数组，传递给
                        hybrid_differential_population 用于 pbest 选取。

    Returns:
        试验向量矩阵。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 动态交叉率（可被外部覆盖）
    if cr is None:
        cr = dynamic_crossover_rate(current_gen, total_gens, zeta)

    # 动态缩放因子（可被外部覆盖）
    pop_size = cost_vectors.shape[0]
    if f_scale is not None:
        f_values = np.full(pop_size, f_scale)
    else:
        f_values = dynamic_scale_factor_batch(cr, pop_size, rng)

    # 混合差分
    return hybrid_differential_population(
        cost_vectors, best_idx, f_values, cr, rng,
        strategy=strategy, fitness_values=fitness_values,
    )
