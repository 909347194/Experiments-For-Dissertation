# -*- coding: utf-8 -*-
"""base_module.py — 可插拔 LLM 模块基类

职责：
    定义所有 LLM 增强模块必须遵循的统一接口。
    每个模块负责一种 LLM 增强能力（种群初始化、算子选择、CR 控制等），
    可通过配置独立启用/禁用，支持消融实验。

设计原则：
    1. 模块自包含：每个模块有自己的 prompt 模板、响应解析、决策应用逻辑
    2. 求解器无感知：solver 不关心模块内部如何工作，只调用 inject()
    3. 决策可追踪：每次决策自动记录到 OptimizationTrajectory
    4. 失败安全：模块内部异常不影响 DMDE 主循环
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModuleState:
    """传递给 LLM 模块的搜索状态快照。

    包含模块做出决策所需的全部上下文信息。
    求解器在每个 hook 点构造此对象传给模块。
    """
    generation: int
    max_generations: int
    # 种群
    population: list | None = None        # Individual 列表
    cost_vectors: np.ndarray | None = None
    best_idx: int = 0
    best_fitness: float = float("inf")
    mean_fitness: float = float("inf")
    # 特征
    diversity: float = 0.0
    gene_variance: float = 0.0
    convergence_speed: float = 0.0
    stagnation_count: int = 0
    feasible_ratio: float = 0.0
    violation_mean: float = 0.0
    violation_max: float = 0.0
    # DE 参数（当前值，模块可以读取或修改）
    cr: float = 0.5
    f_scale: float = 0.5
    temperature: float = 0.5
    # 代价矩阵信息
    cost_matrix: np.ndarray | None = None
    n_uavs: int = 0
    n_targets: int = 0
    model_type: str = "balanced"
    # 轨迹历史
    trajectory_recent: list | None = None   # TrajectoryEntry 列表
    cost_history: list[float] | None = None
    # 额外信息
    extra: dict[str, Any] = field(default_factory=dict)


class BaseLLMModule(ABC):
    """LLM 增强模块抽象基类。

    所有 LLM 模块（种群初始化、算子选择、CR 控制等）
    必须继承此类并实现以下方法：
    - name: 模块名称
    - hook_point: 模块在进化循环中的注入点
    - build_prompt(): 构建 LLM 提示
    - parse_response(): 解析 LLM 响应
    - apply_decision(): 将决策应用到搜索状态

    生命周期：
        module = MyModule(llm_client, config)
        decision = module.inject(state)  # 一次调用完成：提示→LLM→解析→应用
    """

    def __init__(self, llm_client: Any, config: dict[str, Any] | None = None) -> None:
        """初始化模块。

        Args:
            llm_client: LLM 客户端实例（需有 chat() 方法）。
            config: 模块专属配置字典。
        """
        self._llm = llm_client
        self._config = config or {}
        self._enabled = self._config.get("enabled", True)
        self._last_decision: dict[str, Any] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称，用于日志和轨迹记录。"""
        ...

    @property
    @abstractmethod
    def hook_point(self) -> str:
        """模块在进化循环中的注入点。

        返回值：
            "before_init"   — 种群初始化之前（用于 LLM 种群初始化）
            "after_init"    — 种群初始化之后（用于 LLM 种群优化）
            "before_evolve" — 每代进化之前（用于 LLM CR/F 控制）
            "after_evolve"  — 每代进化之后（用于 LLM 算子选择）
            "on_extinction" — 灭绝判断时（用于 LLM 灭绝控制）
        """
        ...

    @property
    def interval(self) -> int:
        """触发间隔（代数）。1 = 每代触发。"""
        return self._config.get("interval", 1)

    @property
    def enabled(self) -> bool:
        """模块是否启用。"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @abstractmethod
    def build_prompt(self, state: ModuleState) -> list[dict[str, str]]:
        """构建 LLM 提示消息。

        Args:
            state: 当前搜索状态快照。

        Returns:
            Chat Completions 格式的消息列表。
        """
        ...

    @abstractmethod
    def parse_response(self, llm_output: str) -> dict[str, Any]:
        """解析 LLM 响应为结构化决策。

        Args:
            llm_output: LLM 的原始输出文本。

        Returns:
            决策字典，键值含义由具体模块定义。
        """
        ...

    @abstractmethod
    def apply_decision(
        self, decision: dict[str, Any], state: ModuleState
    ) -> ModuleState:
        """将 LLM 决策应用到搜索状态。

        Args:
            decision: 解析后的决策字典。
            state: 当前搜索状态（会被就地修改并返回）。

        Returns:
            修改后的搜索状态。
        """
        ...

    def inject(self, state: ModuleState) -> dict[str, Any]:
        """完整注入流程：构建提示 → 调用 LLM → 解析 → 应用。

        这是求解器调用的唯一入口。内部异常不会传播到求解器。

        Args:
            state: 当前搜索状态。

        Returns:
            LLM 决策字典。如果 LLM 调用失败，返回空字典。
        """
        if not self._enabled:
            return {}

        t0 = time.time()
        try:
            # 1. 构建提示
            messages = self.build_prompt(state)

            # 2. 调用 LLM
            llm_output = self._llm.chat(messages)

            # 3. 解析响应
            decision = self.parse_response(llm_output)

            # 4. 应用决策
            state = self.apply_decision(decision, state)

            # 5. 记录
            self._last_decision = decision
            decision["_llm_call_duration"] = time.time() - t0
            decision["_llm_reasoning"] = decision.get("reasoning", "")

            logger.info(
                "[%s @ gen %d] LLM decision applied: %s",
                self.name, state.generation,
                {k: v for k, v in decision.items() if not k.startswith("_")},
            )

            return decision

        except Exception as e:
            logger.warning("[%s @ gen %d] LLM module failed: %s", self.name, state.generation, e)
            return {"_error": str(e), "_llm_call_duration": time.time() - t0}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self._enabled})"
