# -*- coding: utf-8 -*-
"""population_features.py — 种群特征提取

职责：
    从当前种群中提取多样性相关特征，用于 LLM 决策。
    包括种群整体多样性和基因级方差。

特征说明：
    - compute_diversity: 种群个体间的平均归一化距离，反映种群分散程度。
      值域 [0, 1]，0 = 所有个体相同，1 = 最大分散。
    - compute_gene_variance: 种群代价值向量的平均方差，反映基因级变异程度。
      值越大说明种群基因越多样。
"""

from __future__ import annotations

import numpy as np


def compute_diversity(population: list) -> float:
    """计算种群多样性指标。

    使用种群中所有个体代价值向量的平均归一化欧氏距离作为多样性指标。
    归一化方式：除以种群中最大成对距离。

    Args:
        population: 种群个体列表。每个个体需有 cost_vector 属性。

    Returns:
        多样性值，范围 [0.0, 1.0]。
        0.0 表示所有个体完全相同，1.0 表示最大分散。

    Raises:
        ValueError: 种群大小不足 2。
    """
    if len(population) < 2:
        return 0.0

    # 提取代价值矩阵
    cost_vectors = np.array([ind.cost_vector for ind in population])
    pop_size = cost_vectors.shape[0]

    if pop_size < 2:
        return 0.0

    # 计算所有成对欧氏距离
    total_dist = 0.0
    max_dist = 0.0
    count = 0

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = np.linalg.norm(cost_vectors[i] - cost_vectors[j])
            total_dist += dist
            if dist > max_dist:
                max_dist = dist
            count += 1

    if count == 0 or max_dist < 1e-10:
        return 0.0

    # 归一化平均距离
    avg_dist = total_dist / count
    return float(avg_dist / max_dist)


def compute_gene_variance(cost_vectors: np.ndarray) -> float:
    """计算种群基因级方差。

    对每个基因位计算种群中的方差，然后取平均值。
    反映种群在各维度上的变异程度。

    Args:
        cost_vectors: 种群代价值矩阵，shape = (pop_size, gene_len)。

    Returns:
        平均基因方差（非负值）。
    """
    if cost_vectors.shape[0] < 2:
        return 0.0

    # 每个基因位的方差
    gene_variances = np.var(cost_vectors, axis=0)

    # 返回平均方差
    return float(np.mean(gene_variances))


def compute_pairwise_diversity_matrix(population: list) -> np.ndarray:
    """计算种群个体间的成对距离矩阵。

    辅助函数，可用于更精细的多样性分析。

    Args:
        population: 种群个体列表。

    Returns:
        距离矩阵，shape = (pop_size, pop_size)。
    """
    cost_vectors = np.array([ind.cost_vector for ind in population])
    pop_size = cost_vectors.shape[0]
    dist_matrix = np.zeros((pop_size, pop_size))

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = np.linalg.norm(cost_vectors[i] - cost_vectors[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix
