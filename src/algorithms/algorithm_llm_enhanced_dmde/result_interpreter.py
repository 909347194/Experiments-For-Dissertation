# -*- coding: utf-8 -*-
"""result_interpreter.py — LLM 结果解读

职责：
    在 DMDE 求解完成后，让 LLM 解读分配方案的合理性，
    给出改进建议。

LLM 介入点：
    - 分析最优分配方案的代价分布
    - 识别潜在的约束冲突风险
    - 建议下一步实验方向

LLM 不介入：
    - 不修改已求得的解
    - 不参与求解过程
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .llm_advisor import LLMAdvisor, LLMConfig


class LLMResultInterpreter:
    """LLM 结果解读器。

    使用方式::
        interpreter = LLMResultInterpreter(llm_config)
        report = interpreter.interpret(result, cost_matrix, uavs, targets)
    """

    def __init__(self, llm_config: LLMConfig | None = None) -> None:
        self._advisor = LLMAdvisor(llm_config)

    def interpret(
        self,
        best_assignment: list[tuple[int, int]],
        cost_matrix: np.ndarray,
        best_fitness: float,
        convergence_history: list[float],
        n_uavs: int,
        n_targets: int,
        model_type: str,
    ) -> str:
        """解读分配方案。

        Returns:
            LLM 的解读报告（纯文本）。
        """
        # 构建分配摘要
        assignment_details = []
        total_cost = 0
        for u, t in best_assignment:
            if u < cost_matrix.shape[0] and t < cost_matrix.shape[1]:
                cost = cost_matrix[u, t]
                total_cost += cost
                assignment_details.append(f"U{u}→T{t}: {cost:.0f}")

        # 收敛指标
        if len(convergence_history) > 1:
            improvement = (convergence_history[0] - convergence_history[-1]) / (abs(convergence_history[0]) + 1e-10) * 100
        else:
            improvement = 0

        summary = {
            "model_type": model_type,
            "n_uavs": n_uavs,
            "n_targets": n_targets,
            "best_fitness": round(best_fitness, 2),
            "total_cost": round(total_cost, 2),
            "n_assignments": len(best_assignment),
            "convergence_improvement_pct": round(improvement, 1),
            "assignment_details": assignment_details[:20],  # 限制长度
        }

        prompt = f"""解读以下多无人机协同目标分配结果。

分配摘要:
{summary}

请分析:
1. 分配方案是否合理？有无明显的改进空间？
2. 代价分布是否均衡？有无某个 UAV 负担过重？
3. 收敛表现如何？算法是否可能陷入局部最优？
4. 建议下一步实验方向（参数调整、约束修改等）

请用简洁的中文回答。"""

        return self._advisor.ask(
            prompt,
            system="你是多无人机协同任务分配领域的研究专家，负责解读实验结果。"
        )
