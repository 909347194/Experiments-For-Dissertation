# -*- coding: utf-8 -*-
"""seed_generator.py — LLM 引导的初始种群生成

职责：
    利用 LLM 分析代价矩阵结构，生成高质量的初始种群种子，
    作为 DMDE 随机初始化的补充。

LLM 介入点：
    - 分析代价矩阵的结构特征（是否有明显的低成本聚集区）
    - 建议优先分配方向（哪些 UAV-Target 对代价最低）
    - 生成启发式种子个体（贪心 + LLM 建议）

LLM 不介入：
    - 不参与反映射修复（由规则 3.4/3.5/3.6 处理）
    - 不直接修改基因编码
"""

from __future__ import annotations

import numpy as np

from .llm_advisor import LLMAdvisor, LLMConfig
from algorithms.algorithm_dmde.representation.encoder import PopulationEncoder, Individual


class LLMSeedGenerator:
    """LLM 引导的初始种群生成器。

    使用方式::
        gen = LLMSeedGenerator(llm_config)
        seeds = gen.generate_seeds(cost_matrix, n_uavs, n_targets, n_seeds=5)
        # 将 seeds 注入 DMDE 初始种群
    """

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self._advisor = LLMAdvisor(llm_config)

    def analyze_cost_matrix(self, cost_matrix: np.ndarray) -> dict:
        """让 LLM 分析代价矩阵结构。

        Args:
            cost_matrix: 代价矩阵。

        Returns:
            LLM 的分析结果（dict）。
        """
        n_rows, n_cols = cost_matrix.shape
        # 只发送统计信息，不发送完整矩阵（节省 token）
        stats = {
            "shape": [n_rows, n_cols],
            "min": float(cost_matrix.min()),
            "max": float(cost_matrix.max()),
            "mean": float(cost_matrix.mean()),
            "std": float(cost_matrix.std()),
            "row_mins": cost_matrix.min(axis=1).tolist(),
            "col_mins": cost_matrix.min(axis=0).tolist(),
        }

        prompt = f"""分析以下 UAV-Target 代价矩阵的统计特征，给出分配建议。

统计信息:
{json.dumps(stats, indent=2, ensure_ascii=False)}

请回答:
1. 是否存在明显的低成本聚集区？
2. 建议哪些 UAV-Target 对优先分配？
3. 有哪些潜在的约束冲突风险？

请用 JSON 格式回答，包含 keys: low_cost_pairs, allocation_suggestions, risks"""
        import json
        return self._advisor.ask_json(prompt, system="你是多无人机任务分配问题专家。")

    def generate_seeds(
        self,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
        n_seeds: int = 5,
    ) -> list[Individual]:
        """生成 LLM 引导的种子个体。

        Args:
            cost_matrix: 代价矩阵。
            n_uavs: UAV 数量。
            n_targets: 目标数量。
            n_seeds: 种子数量。

        Returns:
            种子个体列表（可直接注入初始种群）。
        """
        # 基于代价矩阵的贪心种子（不依赖 LLM）
        encoder = PopulationEncoder(cost_matrix, n_uavs, n_targets)
        seeds = encoder.generate(n_seeds, seed=42)

        # 如果 LLM 可用，尝试用 LLM 建议优化种子
        analysis = self.analyze_cost_matrix(cost_matrix)
        if analysis and "low_cost_pairs" in analysis:
            # 用 LLM 建议的低代价对构造一个额外种子
            llm_seed = self._build_seed_from_llm_advice(
                analysis["low_cost_pairs"], cost_matrix, n_uavs, n_targets
            )
            if llm_seed is not None:
                seeds.append(llm_seed)

        return seeds

    def _build_seed_from_llm_advice(
        self,
        low_cost_pairs: list,
        cost_matrix: np.ndarray,
        n_uavs: int,
        n_targets: int,
    ) -> Individual | None:
        """根据 LLM 建议的低代价对构造种子个体。"""
        from algorithms.algorithm_dmde.representation.encoder import Gene

        genes = []
        used_uavs = set()
        used_targets = set()

        for pair in low_cost_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                u, t = int(pair[0]), int(pair[1])
                if u < n_uavs and t < n_targets and u not in used_uavs and t not in used_targets:
                    cost = float(cost_matrix[u, t])
                    genes.append(Gene(uav_id=u, target_id=t, cost=cost))
                    used_uavs.add(u)
                    used_targets.add(t)

        if len(genes) < min(n_uavs, n_targets):
            return None  # LLM 建议不足，丢弃

        return Individual(genes=genes, model_type="balanced" if n_uavs == n_targets else "overloaded" if n_uavs > n_targets else "srp")
