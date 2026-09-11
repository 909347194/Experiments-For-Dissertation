# -*- coding: utf-8 -*-
"""Tests for LLM client module."""
import pytest
from unittest.mock import patch, MagicMock
from algorithms.algorithm_llm_enhanced_dmde.llm.llm_client import (
    LLMClient,
    create_llm_client,
    PROVIDER_PRESETS,
)


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
    def test_missing_api_key_raises(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="No API key"):
            create_llm_client(provider="deepseek")
