# -*- coding: utf-8 -*-
"""LLM 可插拔模块注册表。"""

from .population_init import LLMPopulationInitModule
from .operator_selection import LLMOperatorSelectionModule
from .cr_control import LLMCRControlModule

# 模块名称 → 类的映射，用于配置驱动的模块加载
MODULE_REGISTRY: dict[str, type] = {
    "population_init": LLMPopulationInitModule,
    "operator_selection": LLMOperatorSelectionModule,
    "cr_control": LLMCRControlModule,
}

def create_module(name: str, llm_client, config: dict) -> "BaseLLMModule":
    """根据名称创建模块实例。"""
    from ..base_module import BaseLLMModule
    cls = MODULE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM module: {name!r}. Available: {list(MODULE_REGISTRY.keys())}")
    return cls(llm_client=llm_client, config=config)

__all__ = [
    "LLMPopulationInitModule",
    "LLMOperatorSelectionModule",
    "LLMCRControlModule",
    "MODULE_REGISTRY",
    "create_module",
]
