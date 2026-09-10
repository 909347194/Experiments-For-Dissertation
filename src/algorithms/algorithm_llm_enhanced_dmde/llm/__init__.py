# -*- coding: utf-8 -*-
"""LLM 决策层 —— 大语言模型交互组件。

模块化架构：
- llm_client.py: LLM API 客户端（请求发送、响应接收）
- base_module.py: 可插拔 LLM 模块基类（BaseLLMModule + ModuleState）
- modules/: 可插拔模块实现（population_init, operator_selection, cr_control）
"""

from .llm_client import LLMClient
from .base_module import BaseLLMModule, ModuleState

__all__ = [
    "LLMClient",
    "BaseLLMModule",
    "ModuleState",
]
