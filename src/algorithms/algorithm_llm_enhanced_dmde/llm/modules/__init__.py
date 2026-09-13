# -*- coding: utf-8 -*-
"""LLM 可插拔模块注册表。"""

from .population_init import LLMPopulationInitModule
from .operator_selection import LLMOperatorSelectionModule
from .cr_control import LLMCRControlModule
from .search_controller import LLMSearchControllerModule

# 模块名称 → 类的映射，用于配置驱动的模块加载
# 注意: operator_selection 和 cr_control 已废弃，请统一使用 search_controller
MODULE_REGISTRY: dict[str, type] = {
    "population_init": LLMPopulationInitModule,
    "search_controller": LLMSearchControllerModule,
    # deprecated: 以下模块保留向后兼容，但 solver 不再调用
    "operator_selection": LLMOperatorSelectionModule,
    "cr_control": LLMCRControlModule,
}

def create_module(name: str, llm_client, config: dict) -> "BaseLLMModule":
    """根据名称创建模块实例。"""
    from ..base_module import BaseLLMModule
    cls = MODULE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM module: {name!r}. Available: {list(MODULE_REGISTRY.keys())}")
    if name in ("operator_selection", "cr_control"):
        import warnings
        warnings.warn(
            f"LLM module '{name}' is deprecated, use 'search_controller' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    return cls(llm_client=llm_client, config=config)

__all__ = [
    "LLMPopulationInitModule",
    "LLMOperatorSelectionModule",
    "LLMCRControlModule",
    "LLMSearchControllerModule",
    "MODULE_REGISTRY",
    "create_module",
]
