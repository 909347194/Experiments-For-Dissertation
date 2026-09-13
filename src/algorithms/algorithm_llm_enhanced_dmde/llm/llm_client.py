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

思维链（thinking）模型注意事项::

    思考型模型（如 DeepSeek ``deepseek-flash``）默认开启 thinking，
    思维链以 ``reasoning_content`` 返回，且**计入 max_tokens**。
    若 ``max_tokens`` 过小，思考会吃光全部预算，导致
    ``finish_reason="length"`` 且 ``content`` 为空。
    通过 ``reasoning_effort`` 控制思考强度（"none" 关闭思考），
    并通过 ``max_tokens_cap`` 限制截断重试时的 token 上限。
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


class LLMEmptyResponseError(ValueError):
    """LLM 返回内容为空（choices 为空或 content 为空）。"""


class LLMTruncatedResponseError(ValueError):
    """LLM 输出被 max_tokens 截断（finish_reason="length"）。

    思考型模型会把思维链计入 max_tokens，因此该错误常见于
    "思考吃光预算、正文没来得及输出" 的场景。
    """


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
        reasoning_effort: str | None = None,
        max_tokens_cap: int | None = None,
        extra_body: dict[str, Any] | None = None,
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
        # 思维链强度：None=用服务端默认，"none"/"low"/"high"/"max" 显式指定。
        # 思考型模型（deepseek-flash 等）默认 effort="high"，思维链计入
        # max_tokens，不限制时容易把预算吃光导致正文为空。
        self.reasoning_effort = reasoning_effort
        # 截断重试时 max_tokens 允许增长到的上限
        self.max_tokens_cap = max_tokens_cap or max(max_tokens * 2, 16384)
        self._extra_body = dict(extra_body or {})

    def chat(self, messages: list[dict[str, str]], max_retries: int = 3) -> str:
        """发送 Chat Completions 请求（带重试）。

        重试策略：
            - 429 限流 / 5xx 服务端错误：退避重试。
            - 空响应（content 为空）：退避重试。
            - 输出截断（finish_reason="length"）：**加倍 max_tokens** 后重试，
              因为用同样的预算重发同样的请求只会再次被截断。

        Args:
            messages: [{"role": "user", "content": "Hello"}]
            max_retries: 最大重试次数（针对 429/5xx/空响应/输出截断）。

        Returns:
            助手回复文本（已 strip）。

        Raises:
            LLMEmptyResponseError: 重试后仍返回空内容。
            LLMTruncatedResponseError: 重试后输出仍被 max_tokens 截断。
        """
        last_exc: Exception | None = None
        budget = self.max_tokens
        for attempt in range(max_retries):
            try:
                response = self._request(messages, budget)
                content = self._extract_content(response, budget)
                logger.debug(
                    "[LLM] model=%s, messages=%d, response=%d chars",
                    self.model, len(messages), len(content),
                )
                return content
            except Exception as e:
                last_exc = e
                truncated = isinstance(e, LLMTruncatedResponseError)
                # 判断是否可重试：429 限流、5xx 服务端错误、空响应、输出截断
                status = getattr(e, "status_code", None)
                is_retryable = (
                    status in (429, 500, 502, 503, 504)
                    or isinstance(e, (LLMEmptyResponseError, LLMTruncatedResponseError))
                )
                if not is_retryable or attempt >= max_retries - 1:
                    logger.warning("LLM call failed: %s", e)
                    raise
                if truncated:
                    new_budget = min(budget * 2, self.max_tokens_cap)
                    if new_budget > budget:
                        logger.warning(
                            "[LLM] output truncated at max_tokens=%d, escalating to %d",
                            budget, new_budget,
                        )
                        budget = new_budget
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "[LLM] attempt %d/%d failed, retrying in %ds: %s",
                    attempt + 1, max_retries, wait, e,
                )
                time.sleep(wait)
        # 所有重试用尽
        raise last_exc  # type: ignore[misc]

    def _request(self, messages: list[dict[str, str]], max_tokens: int):
        """发起一次 Chat Completions 请求（不含重试）。"""
        extra_body = dict(self._extra_body)
        if self.reasoning_effort is not None:
            extra_body["reasoning_effort"] = self.reasoning_effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _extract_content(response: Any, max_tokens: int) -> str:
        """从响应中提取正文；截断或为空时抛出可重试异常。"""
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMEmptyResponseError("Empty choices from LLM")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        raw = choice.message.content
        content = raw.strip() if isinstance(raw, str) else ""
        if finish_reason == "length":
            detail = (
                "no answer content emitted (thinking likely consumed the budget)"
                if not content else "answer may be incomplete"
            )
            raise LLMTruncatedResponseError(
                f"LLM output truncated by max_tokens={max_tokens} "
                f"(finish_reason=length): {detail}"
            )
        if not content:
            raise LLMEmptyResponseError(
                f"Empty response from LLM (finish_reason={finish_reason})"
            )
        return content

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
    reasoning_effort: str | None = None,
    max_tokens_cap: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> LLMClient:
    """创建 LLM 客户端（推荐入口）。

    Args:
        provider:    预设名称 ("deepseek", "openai", "custom")。
        model:       模型名（None 时使用预设默认值）。
        api_base:    API 地址（None 时使用预设值）。
        api_key:     API 密钥（None 时从环境变量读取）。
        temperature: 生成温度（思考模式下服务端会忽略）。
        max_tokens:  最大生成 token 数（含思维链）。
        timeout:     超时秒数。
        reasoning_effort: 思维链强度 "none"/"low"/"high"/"max"，None=服务端默认。
        max_tokens_cap:   截断重试时 max_tokens 的上限。
        extra_body:  透传给 API 的额外请求体字段。

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
            reasoning_effort=reasoning_effort,
            max_tokens_cap=max_tokens_cap,
            extra_body=extra_body,
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
        reasoning_effort=reasoning_effort,
        max_tokens_cap=max_tokens_cap,
        extra_body=extra_body,
    )


def create_llm_client_from_config(config_path: str | Path) -> LLMClient:
    """从 YAML 配置文件创建 LLM 客户端。

    配置文件格式::

        provider: deepseek
        model: deepseek-flash
        temperature: 0.7
        max_tokens: 8192
        timeout: 60
        # 思维链强度（"none" 关闭思考，可大幅降低耗时与 token 消耗）
        reasoning_effort: low
        # 截断重试时的 token 上限
        max_tokens_cap: 16384
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
        reasoning_effort=cfg.get("reasoning_effort"),
        max_tokens_cap=cfg.get("max_tokens_cap"),
        extra_body=cfg.get("extra_body"),
    )
