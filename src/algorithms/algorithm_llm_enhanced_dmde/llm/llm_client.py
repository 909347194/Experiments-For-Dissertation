# -*- coding: utf-8 -*-
"""llm_client.py — 多 Provider LLM 客户端

支持 DeepSeek、OpenAI 及其他 OpenAI 兼容 API。
使用官方 openai SDK，通过 provider 预设简化配置。

使用方式::

    # 方式 1: 使用 provider 预设（推荐）
    client = create_llm_client(provider="deepseek", model="deepseek-flash")

    # 方式 2: 自定义配置
    client = create_llm_client(
        provider="custom",
        api_base="https://api.example.com/v1",
        api_key="sk-xxx",
        model="my-model",
    )

    # 方式 3: 从 llm_config.yaml 加载
    client = create_llm_client_from_config("config/llm_config.yaml")

    # 调用
    response = client.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ])
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Provider 预设
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "api_base": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-flash",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
}


def _load_env() -> None:
    """加载 .env 文件（如果存在）。"""
    try:
        from dotenv import load_dotenv
        # 从项目根目录向上查找 .env
        for parent in [Path.cwd(), *Path.cwd().parents]:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
    except ImportError:
        # python-dotenv 未安装，跳过
        pass


class LLMClient:
    """基于 OpenAI SDK 的多 Provider LLM 客户端。"""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for LLMClient. "
                "Install it with: pip install openai"
            )
        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base

    def chat(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        """发送 Chat Completions 请求（带重试）。

        Args:
            messages: [{"role": "user", "content": "Hello"}]
            max_retries: 最大重试次数（针对 429/5xx 错误）。

        Returns:
            助手回复文本。
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                result = content.strip()
                logger.debug(
                    "[LLM] model=%s, messages=%d, response=%d chars",
                    self.model, len(messages), len(result),
                )
                logger.debug(
                    "[LLM] model=%s, messages=%d, response=%d chars",
                    self.model, len(messages), len(result),
                )
                return result
            except Exception as e:
                last_exc = e
                # 判断是否可重试：429 限流 或 5xx 服务端错误
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "[LLM] attempt %d/%d failed (status=%s), retrying in %ds: %s",
                        attempt + 1, max_retries, status, wait, e,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("LLM call failed: %s", e)
                raise
        # 所有重试用尽
        raise last_exc  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, base={self.api_base!r})"


def create_llm_client(
    provider: str = "deepseek",
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 60,
) -> LLMClient:
    """创建 LLM 客户端（推荐入口）。

    Args:
        provider:    预设名称 ("deepseek", "openai", "custom")。
        model:       模型名（None 时使用预设默认值）。
        api_base:    API 地址（None 时使用预设值）。
        api_key:     API 密钥（None 时从环境变量读取）。
        temperature: 生成温度。
        max_tokens:  最大 token 数。
        timeout:     超时秒数。

    Returns:
        LLMClient 实例。
    """
    _load_env()

    if provider == "custom":
        if not api_base or not api_key:
            raise ValueError("Custom provider requires api_base and api_key")
        return LLMClient(
            api_base=api_base,
            api_key=api_key,
            model=model or "default",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Available: {list(PROVIDER_PRESETS.keys())}"
        )

    final_api_base = api_base or preset["api_base"]
    final_api_key = api_key or os.environ.get(preset["env_key"], "")
    final_model = model or preset["default_model"]

    if not final_api_key:
        raise ValueError(
            f"No API key for provider '{provider}'. "
            f"Set {preset['env_key']} in .env or pass api_key parameter."
        )

    return LLMClient(
        api_base=final_api_base,
        api_key=final_api_key,
        model=final_model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def create_llm_client_from_config(config_path: str | Path) -> LLMClient:
    """从 YAML 配置文件创建 LLM 客户端。

    配置文件格式::

        provider: deepseek
        model: deepseek-flash
        temperature: 0.7
        max_tokens: 1024
        timeout: 60
        # 以下可选（覆盖 provider 默认值）
        # api_base: https://custom.api.com/v1
        # api_key: sk-...

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        LLMClient 实例。
    """
    import yaml

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return create_llm_client(
        provider=cfg.get("provider", "deepseek"),
        model=cfg.get("model"),
        api_base=cfg.get("api_base"),
        api_key=cfg.get("api_key"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 1024),
        timeout=cfg.get("timeout", 60),
    )
