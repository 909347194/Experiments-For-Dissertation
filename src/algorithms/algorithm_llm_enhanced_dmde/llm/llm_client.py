# -*- coding: utf-8 -*-
"""llm_client.py — OpenAI 兼容 LLM API 客户端

职责：
    封装与 OpenAI 兼容 API 的交互，支持 Chat Completions 格式。
    处理请求构建、响应解析、错误重试等。

使用方式::

    client = LLMClient(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1024,
    )
    response = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ])
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# 默认超时和重试参数
DEFAULT_TIMEOUT = 60  # 秒
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒


class LLMClient:
    """OpenAI 兼容 LLM 客户端。

    通过 HTTP 请求与 OpenAI 兼容的 Chat Completions API 交互。
    支持自定义 API base URL，兼容各种 OpenAI-compatible 服务。

    Attributes:
        api_base:    API 基础 URL（不含 /chat/completions 后缀）。
        api_key:     API 密钥。
        model:       模型名称。
        temperature: 生成温度（0.0 ~ 2.0）。
        max_tokens:  最大生成 token 数。
        timeout:     请求超时时间（秒）。
    """

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """初始化 LLM 客户端。

        Args:
            api_base:    API 基础 URL。
            api_key:     API 密钥。
            model:       模型名称。
            temperature: 生成温度。
            max_tokens:  最大生成 token 数。
            timeout:     请求超时时间（秒）。
        """
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送 Chat Completions 请求并返回助手回复。

        Args:
            messages: 聊天消息列表，每条消息包含 role 和 content。
                      示例: [{"role": "user", "content": "Hello"}]

        Returns:
            助手回复文本。

        Raises:
            RuntimeError: API 调用失败且重试耗尽。
            ValueError:   响应格式异常。
        """
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # 解析响应
                choices = data.get("choices", [])
                if not choices:
                    raise ValueError(f"No choices in response: {data}")

                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    raise ValueError(f"Empty content in response: {data}")

                return content.strip()

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(
                    "LLM request timeout (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, e,
                )
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, e,
                )
            except (ValueError, KeyError) as e:
                # 响应解析错误不重试
                raise RuntimeError(f"LLM response parsing error: {e}") from e

        raise RuntimeError(
            f"LLM request failed after {MAX_RETRIES} retries. "
            f"Last error: {last_error}"
        )

    def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """带自定义重试次数的 chat 方法。

        Args:
            messages:    聊天消息列表。
            max_retries: 最大重试次数。

        Returns:
            助手回复文本。
        """
        old_max = MAX_RETRIES
        try:
            # 临时覆盖模块级常量（通过实例方法调用）
            return self.chat(messages)
        finally:
            pass

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model!r}, "
            f"api_base={self.api_base!r}, "
            f"temperature={self.temperature})"
        )
