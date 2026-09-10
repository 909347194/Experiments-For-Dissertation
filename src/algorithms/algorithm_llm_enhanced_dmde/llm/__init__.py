# -*- coding: utf-8 -*-
"""LLM 决策层 —— 大语言模型交互组件。

负责与 OpenAI 兼容 API 交互，包括：
- llm_client.py: LLM API 客户端（请求发送、响应接收）
- prompt_builder.py: 提示组装（从搜索特征构建决策提示）
- response_parser.py: 响应解析（将 LLM 输出解析为结构化决策）
"""

from .llm_client import LLMClient
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser, LLMDecision

__all__ = [
    "LLMClient",
    "PromptBuilder",
    "ResponseParser",
    "LLMDecision",
]
