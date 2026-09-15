# -*- coding: utf-8 -*-
"""candidate_filter.py — Quality + Diversity 候选过滤器

职责：
    对 LLM 生成的候选 assignment（已转换为 Individual）进行
    质量 + 多样性过滤，选出最优的 k 个个体注入初始种群。

设计：
    1. 先按 fitness 排序（质量优先）
    2. 贪心选择：每次选 fitness 最好且与已选集合多样性足够的个体
    3. 多样性用 assignment 的 Hamming 距离衡量
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ...representation.encoder import Individual

logger = logging.getLogger(__name__)


class CandidateFilter:
    """对 LLM 生成的候选个体进行质量+多样性过滤。

    使用方式::

        filter = CandidateFilter(
            fitness_evaluator=evaluator,
            cost_matrix=cm,
            n_uavs=n, n_targets=m,
            diversity_threshold=0.1,
        )
        selected = filter.filter(candidates, k=10)
    """

    def __init__(
        self,
        fitness_evaluator: Any,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        diversity_threshold: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            fitness_evaluator: 适应度评估器（需有 evaluate() 方法）。
            cost_matrix: 代价矩阵。
            n_uavs: UAV 数量。
            n_targets: 目标数量。
            diversity_threshold: 多样性阈值（0~1），低于此值认为两个个体过于相似。
                                 ⚠️ 实验调参项。
            seed: 随机种子。
        """
        self._evaluator = fitness_evaluator
        self._cm = cost_matrix
        self._n = n_uavs
        self._m = n_targets
        self._threshold = diversity_threshold
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def filter(
        self, candidates: list[Individual], k: int
    ) -> list[Individual]:
        """从候选中选出 k 个，兼顾质量和多样性。

        策略：
        1. 评估所有候选的 fitness
        2. 按 fitness 排序
        3. 贪心选择：每次选 fitness 最好且与已选集合多样性足够的个体
        4. 如果多样性过滤后不足 k 个，放宽阈值补充

        Args:
            candidates: 候选个体列表（已转换为 Individual）。
            k: 需要选出的个体数量。

        Returns:
            选出的 Individual 列表（最多 k 个）。
        """
        if not candidates:
            logger.info("[CandidateFilter] 候选列表为空")
            return []

        # Step 1: 评估 fitness（如果尚未评估）
        for ind in candidates:
            if ind.fitness == float("inf"):
                ind.fitness = self._evaluate(ind)

        # Step 2: 按 fitness 排序（升序，越小越好）
        sorted_candidates = sorted(candidates, key=lambda x: x.fitness)

        # Step 3: 贪心选择（质量 + 多样性）
        selected = self._greedy_select(sorted_candidates, k, self._threshold)

        # Step 4: 如果不足 k 个，放宽阈值补充
        if len(selected) < k:
            remaining = [c for c in sorted_candidates if c not in selected]
            # 不再考虑多样性，直接按 fitness 补齐
            shortfall = k - len(selected)
            selected.extend(remaining[:shortfall])
            logger.info(
                "[CandidateFilter] 多样性过滤后不足 %d 个，放宽阈值补充 %d 个",
                k, shortfall,
            )

        logger.info(
            "[CandidateFilter] 从 %d 个候选中选出 %d 个 (阈值=%.3f)",
            len(candidates), len(selected), self._threshold,
        )
        return selected[:k]

    def _greedy_select(
        self,
        sorted_candidates: list[Individual],
        k: int,
        threshold: float,
    ) -> list[Individual]:
        """贪心选择：质量优先 + 多样性保障。

        每次从剩余候选中选 fitness 最好且与已选集合
        最小 Hamming 距离 ≥ threshold 的个体。
        """
        if not sorted_candidates:
            return []

        selected = [sorted_candidates[0]]  # 最优个体直接入选
        remaining = sorted_candidates[1:]

        # 预计算所有候选的 assignment 表示（用于 Hamming 距离）
        selected_assignments = [self._assignment_vector(selected[0])]

        while len(selected) < k and remaining:
            chosen_idx = None       # 最终选中的索引
            fallback_idx = None     # 不满足阈值时的最佳多样性候选
            fallback_diversity = -1.0

            for i, cand in enumerate(remaining):
                cand_vec = self._assignment_vector(cand)
                # 计算与已选集合的最小 Hamming 距离
                min_dist = min(
                    self._hamming_distance(cand_vec, sv)
                    for sv in selected_assignments
                )
                if min_dist >= threshold:
                    # 满足多样性要求，直接选它（remaining 已按 fitness 排序）
                    chosen_idx = i
                    break
                # 记录多样性最好的候选（用于无阈值满足时的 fallback）
                if min_dist > fallback_diversity:
                    fallback_diversity = min_dist
                    fallback_idx = i

            # 无满足阈值的候选时，选择多样性最好的作为 fallback
            if chosen_idx is None:
                chosen_idx = fallback_idx

            if chosen_idx is None:
                break

            chosen = remaining.pop(chosen_idx)
            selected.append(chosen)
            selected_assignments.append(self._assignment_vector(chosen))

        return selected

    def _assignment_vector(self, ind: Individual) -> np.ndarray:
        """将个体的 assignment 转换为固定长度的向量（用于距离计算）。

        balanced/overloaded: 长度 = n_uavs，每个元素是 target_id。
        srp: 长度 = n_targets，每个元素是负责该目标的 uav_id。
        """
        if ind.model_type == "srp":
            # SRP: 每个目标由哪个 UAV 负责
            vec = np.full(self._m, -1, dtype=int)
            current_uav = -1
            for g in ind.genes:
                if g.uav_id >= 0:
                    current_uav = g.uav_id
                if 0 <= g.target_id < self._m:
                    vec[g.target_id] = current_uav
            return vec
        else:
            # balanced/overloaded: 每个 UAV 分配到哪个目标
            vec = np.full(self._n, -1, dtype=int)
            for g in ind.genes:
                if 0 <= g.uav_id < self._n:
                    vec[g.uav_id] = g.target_id
            return vec

    def _hamming_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个 assignment 向量的归一化 Hamming 距离。

        返回 [0, 1]，0 表示完全相同，1 表示完全不同。
        """
        if len(vec1) != len(vec2):
            # 长度不同，取较短的部分比较
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        if len(vec1) == 0:
            return 0.0

        return float(np.sum(vec1 != vec2)) / len(vec1)

    def _evaluate(self, ind: Individual) -> float:
        """评估单个个体的 fitness。"""
        assignment = ind.assignment
        if not assignment:
            return 1e12
        result = self._evaluator.evaluate(assignment, self._cm, n_uavs=self._n)
        return result.fitness