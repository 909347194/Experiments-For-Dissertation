# -*- coding: utf-8 -*-
"""config.py — 实验配置数据类"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    """消融实验统一配置。"""

    # DMDE 核心参数
    pop_size: int = 50
    max_generations: int = 1000
    zeta: int = 3
    delta: float = 0.3

    # 实验参数
    n_runs: int = 30
    seed_start: int = 42

    @property
    def seeds(self) -> list[int]:
        """生成种子序列。"""
        return list(range(self.seed_start, self.seed_start + self.n_runs))


# 默认配置
DEFAULT_CONFIG = ExperimentConfig()