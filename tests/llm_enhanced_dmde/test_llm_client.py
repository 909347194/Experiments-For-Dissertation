# -*- coding: utf-8 -*-
"""Tests for LLM client module."""
import pytest
from unittest.mock import patch, MagicMock
from algorithms.algorithm_llm_enhanced_dmde.llm.llm_client import LLMClient


class TestLLMClient:
    """Test suite for LLMClient."""

    def test_init(self):
        """Test client initialization."""
        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        assert client.model == "gpt-4o"

    def test_init_stores_api_base(self):
        """Test that api_base is stored correctly."""
        client = LLMClient(api_base="http://my-api", api_key="key123", model="gpt-4o")
        assert client.api_base == "http://my-api"

    def test_chat_mock(self):
        """Test chat with mocked response."""
        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test response"}}]
        }
        mock_response.status_code = 200
        with patch("requests.post", return_value=mock_response):
            result = client.chat([{"role": "user", "content": "hello"}])
            assert result is not None

    def test_chat_handles_api_error(self):
        """Test that API errors are handled gracefully."""
        client = LLMClient(api_base="http://test", api_key="test", model="gpt-4o")
        with patch("requests.post", side_effect=Exception("Connection error")):
            with pytest.raises(Exception):
                client.chat([{"role": "user", "content": "hello"}])
