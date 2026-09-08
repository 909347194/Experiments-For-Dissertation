# -*- coding: utf-8 -*-
"""DMDE 领域模型模块。

职责：
    定义任务模型，包括实体、代价矩阵、约束评估。
    与 environment_dmde（物理环境）分离，专注任务逻辑。

对应论文：
    第 2 章 —— 复杂多约束多机协同目标分配和航迹规划的统一建模。
"""

from .entities import UAV, Target
from .cost import CostMatrix, CostMatrixBuilder
from .constraints import FitnessEvaluator, FitnessResult

__all__ = [
    "UAV",
    "Target",
    "CostMatrix",
    "CostMatrixBuilder",
    "FitnessEvaluator",
    "FitnessResult",
]
