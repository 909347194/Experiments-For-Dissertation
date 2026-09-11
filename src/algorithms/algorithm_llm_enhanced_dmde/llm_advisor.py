# -*- coding: utf-8 -*-
"""llm_advisor.py — LLM 接口封装

职责：
    统一封装大语言模型的调用接口，为 seed_generator、
    parameter_advisor、result_interpreter 提供底层支持。

设计原则：
    - LLM 仅作为"顾问"提供建议，不直接修改解
    - 所有 LLM 输出需经解析和验证后才能被算法使用
    - 支持多种 LLM 后端（OpenAI API / 本地模型）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMConfig:
    """LLM 配置。

    Attributes:
        api_key:    API 密钥。
        base_url:   API 端点。
        model:      模型名称。
        temperature: 生成温度。
        max_tokens:  最大生成 token 数。
    """

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000


class LLMAdvisor:
    """LLM 顾问封装。

    使用方式::

        advisor = LLMAdvisor(config)
        response = advisor.ask("分析这个代价矩阵的结构特点...")
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._cfg = config or LLMConfig()

    def ask(self, prompt: str, system: str = "") -> str:
        """向 LLM 发送请求并获取回复。

        Args:
            prompt: 用户提示。
            system: 系统提示（可选）。

        Returns:
            LLM 回复文本。
        """
        try:
            import httpx
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = httpx.post(
                f"{self._cfg.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cfg.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._cfg.model,
                    "messages": messages,
                    "temperature": self._cfg.temperature,
                    "max_tokens": self._cfg.max_tokens,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM 调用失败: {e}]"

    def ask_json(self, prompt: str, system: str = "") -> dict[str, Any]:
        """向 LLM 发送请求，期望 JSON 格式回复。

        Returns:
            解析后的 JSON dict，解析失败返回空 dict。
        """
        raw = self.ask(prompt, system)
        try:
            # 提取 JSON 块
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            return {}
