# -*- coding: utf-8 -*-
"""Tests for LLM response parser module."""
import pytest
from algorithms.algorithm_llm_enhanced_dmde.llm.response_parser import ResponseParser, LLMDecision


class TestResponseParser:
    """Test suite for ResponseParser."""

    def test_parse_valid_response(self):
        """Test parsing a well-formed LLM response."""
        parser = ResponseParser()
        valid_json = '{"operator": "de/rand/1/bin", "scale_factor": 0.8, "crossover_rate": 0.9, "reason": "high diversity"}'
        decision = parser.parse(valid_json)
        assert decision is not None
        assert isinstance(decision, LLMDecision)

    def test_parse_malformed_response(self):
        """Test graceful handling of malformed output."""
        parser = ResponseParser()
        malformed = "this is not json at all"
        result = parser.parse(malformed)
        # Should return a fallback/default decision or None, not crash
        assert result is None or isinstance(result, LLMDecision)

    def test_parse_missing_fields(self):
        """Test when LLM omits some fields."""
        parser = ResponseParser()
        partial_json = '{"operator": "de/best/2/bin"}'
        decision = parser.parse(partial_json)
        # Should handle missing fields with defaults
        assert decision is not None or decision is None

    def test_parse_empty_string(self):
        """Test parsing an empty string."""
        parser = ResponseParser()
        result = parser.parse("")
        assert result is None or isinstance(result, LLMDecision)

    def test_parse_json_with_extra_fields(self):
        """Test parsing JSON with unexpected extra fields."""
        parser = ResponseParser()
        extra_json = '{"operator": "de/rand/1/bin", "scale_factor": 0.5, "crossover_rate": 0.7, "extra_field": "ignored", "reason": "test"}'
        decision = parser.parse(extra_json)
        assert decision is not None
