# -*- coding: utf-8 -*-
"""LLM 可插拔模块注册表。"""

from .population_init import LLMPopulationInitModule
from .search_controller import LLMSearchControllerModule
from .assignment_converter import AssignmentConverter
from .candidate_filter import CandidateFilter

# deprecated modules — import guarded so their removal won't break the package
try:
    from .operator_selection import LLMOperatorSelectionModule
except ImportError:
    LLMOperatorSelectionModule = None  # type: ignore[assignment,misc]

try:
    from .cr_control import LLMCRControlModule
except ImportError:
    LLMCRControlModule = None  # type: ignore[assignment,misc]

# 模块名称 → 类的映射，用于配置驱动的模块加载
# 注意: operator_selection 和 cr_control 已废弃，请统一使用 search_controller
MODULE_REGISTRY: dict[str, type] = {
    "population_init": LLMPopulationInitModule,
    "search_controller": LLMSearchControllerModule,
}
# 仅在 deprecated 模块可用时注册
if LLMOperatorSelectionModule is not None:
    MODULE_REGISTRY["operator_selection"] = LLMOperatorSelectionModule
if LLMCRControlModule is not None:
    MODULE_REGISTRY["cr_control"] = LLMCRControlModule

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
    "AssignmentConverter",
    "CandidateFilter",
    "MODULE_REGISTRY",
    "create_module",
]
