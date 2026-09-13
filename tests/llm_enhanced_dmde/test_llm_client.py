# -*- coding: utf-8 -*-
"""Tests for LLM client module."""
import pytest
from unittest.mock import patch, MagicMock
from algorithms.algorithm_llm_enhanced_dmde.llm.llm_client import (
    LLMClient,
    LLMEmptyResponseError,
    LLMTruncatedResponseError,
    create_llm_client,
    create_llm_client_from_config,
    PROVIDER_PRESETS,
)


def make_response(content, finish_reason="stop"):
    """构造一个模拟的 ChatCompletion 响应。"""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMClient:
    """Test suite for LLMClient."""

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_init(self, mock_openai_cls):
        """Test client initialization."""
        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        assert client.model == "gpt-4o"
        mock_openai_cls.assert_called_once()

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_init_stores_api_base(self, mock_openai_cls):
        """Test that api_base is stored correctly."""
        client = LLMClient(api_base="http://my-api", api_key="key123", model="gpt-4o")
        assert client.api_base == "http://my-api"

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_chat_mock(self, mock_openai_cls):
        """Test chat with mocked OpenAI SDK response."""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk

        mock_message = MagicMock()
        mock_message.content = "test response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_sdk.chat.completions.create.return_value = mock_response

        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        result = client.chat([{"role": "user", "content": "hello"}])
        assert result == "test response"
        mock_sdk.chat.completions.create.assert_called_once()

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_chat_handles_api_error(self, mock_openai_cls):
        """Test that API errors are handled gracefully."""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = Exception("Connection error")

        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        with pytest.raises(Exception, match="Connection error"):
            client.chat([{"role": "user", "content": "hello"}])


class TestLLMClientRetry:
    """重试 / 截断 / 空响应相关行为。"""

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.time.sleep")
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_retries_empty_response_then_succeeds(self, mock_openai_cls, mock_sleep):
        """空响应应当重试，并在第二次成功时返回正文。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = [
            make_response(""),
            make_response("{\"cr\": 0.9}"),
        ]

        client = LLMClient(api_base="http://test", api_key="test", model="m")
        assert client.chat([{"role": "user", "content": "hi"}]) == '{"cr": 0.9}'
        assert mock_sdk.chat.completions.create.call_count == 2

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.time.sleep")
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_truncation_escalates_max_tokens(self, mock_openai_cls, mock_sleep):
        """finish_reason=length 时应加倍 max_tokens 后重试，而非原地重发。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = [
            make_response("", finish_reason="length"),  # 思考吃光预算
            make_response('{"cr": 0.5}'),
        ]

        client = LLMClient(
            api_base="http://test", api_key="test", model="m", max_tokens=1024,
        )
        assert client.chat([{"role": "user", "content": "hi"}]) == '{"cr": 0.5}'

        budgets = [
            call.kwargs["max_tokens"]
            for call in mock_sdk.chat.completions.create.call_args_list
        ]
        assert budgets == [1024, 2048]

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.time.sleep")
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_truncated_partial_content_is_not_returned(self, mock_openai_cls, mock_sleep):
        """截断但非空的正文（残缺 JSON）也不应被当作有效结果返回。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = [
            make_response('{"cr": 0.3, "reaso', finish_reason="length"),
            make_response('{"cr": 0.3}'),
        ]

        client = LLMClient(api_base="http://test", api_key="test", model="m")
        assert client.chat([{"role": "user", "content": "hi"}]) == '{"cr": 0.3}'

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.time.sleep")
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_truncation_raises_after_all_retries(self, mock_openai_cls, mock_sleep):
        """持续截断时，最终抛出 LLMTruncatedResponseError。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = lambda **kw: make_response(
            "", finish_reason="length"
        )

        client = LLMClient(api_base="http://test", api_key="test", model="m", max_tokens=512)
        with pytest.raises(LLMTruncatedResponseError, match="truncated"):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.time.sleep")
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_empty_response_raises_after_all_retries(self, mock_openai_cls, mock_sleep):
        """持续空响应时，最终抛出 LLMEmptyResponseError。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.side_effect = lambda **kw: make_response("")

        client = LLMClient(api_base="http://test", api_key="test", model="m")
        with pytest.raises(LLMEmptyResponseError, match="Empty response"):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_reasoning_effort_passed_via_extra_body(self, mock_openai_cls):
        """reasoning_effort 应作为请求体字段透传给服务端。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.return_value = make_response("ok")

        client = LLMClient(
            api_base="http://test", api_key="test", model="m", reasoning_effort="low",
        )
        client.chat([{"role": "user", "content": "hi"}])
        assert mock_sdk.chat.completions.create.call_args.kwargs["extra_body"] == {
            "reasoning_effort": "low"
        }

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_no_extra_body_when_reasoning_effort_none(self, mock_openai_cls):
        """未配置 reasoning_effort 时不应发送 extra_body。"""
        mock_sdk = MagicMock()
        mock_openai_cls.return_value = mock_sdk
        mock_sdk.chat.completions.create.return_value = make_response("ok")

        client = LLMClient(api_base="http://test", api_key="test", model="m")
        client.chat([{"role": "user", "content": "hi"}])
        assert "extra_body" not in mock_sdk.chat.completions.create.call_args.kwargs

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_escalation_respects_cap(self, mock_openai_cls):
        """max_tokens_cap 应限制截断重试时的 token 增长。"""
        assert LLMClient(
            api_base="http://t", api_key="k", model="m",
            max_tokens=1000, max_tokens_cap=1500,
        ).max_tokens_cap == 1500
        # 未指定时默认为 max(max_tokens*2, 16384)
        assert LLMClient(api_base="http://t", api_key="k", model="m").max_tokens_cap == 16384
        assert LLMClient(
            api_base="http://t", api_key="k", model="m", max_tokens=20000,
        ).max_tokens_cap == 40000


class TestCreateLLMClient:
    """Test suite for create_llm_client factory."""

    @patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-key"})
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_deepseek_preset(self, mock_openai_cls):
        """Test creating client with deepseek preset."""
        client = create_llm_client(provider="deepseek")
        assert client.model == "deepseek-flash"
        assert client.api_base == "https://api.deepseek.com"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_openai_preset(self, mock_openai_cls):
        """Test creating client with openai preset."""
        client = create_llm_client(provider="openai")
        assert client.model == "gpt-4o"

    def test_custom_provider_requires_params(self):
        """Test that custom provider requires api_base and api_key."""
        with pytest.raises(ValueError, match="Custom provider requires"):
            create_llm_client(provider="custom")

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_client(provider="nonexistent")

    @patch.dict("os.environ", {}, clear=True)
    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client._load_env")
    def test_missing_api_key_raises(self, mock_load_env):
        """Test that missing API key raises ValueError."""
        # _load_env 会从工作区的 .env 回填密钥，这里必须一并屏蔽，
        # 否则本机存在 .env 时该测试会误判。
        with pytest.raises(ValueError, match="No API key"):
            create_llm_client(provider="deepseek")

    @patch("algorithms.algorithm_llm_enhanced_dmde.llm.llm_client.OpenAI")
    def test_from_config_reads_thinking_settings(self, mock_openai_cls, tmp_path):
        """llm_config.yaml 中的 reasoning_effort / max_tokens_cap 应生效。"""
        cfg_file = tmp_path / "llm_config.yaml"
        cfg_file.write_text(
            "provider: deepseek\n"
            "model: deepseek-flash\n"
            "max_tokens: 8192\n"
            "reasoning_effort: low\n"
            "max_tokens_cap: 16384\n",
            encoding="utf-8",
        )
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test-key"}):
            client = create_llm_client_from_config(cfg_file)
        assert client.reasoning_effort == "low"
        assert client.max_tokens == 8192
        assert client.max_tokens_cap == 16384
