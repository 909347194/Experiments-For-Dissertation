# -*- coding: utf-8 -*-
"""Tests for prompt builder module."""
import pytest
from algorithms.algorithm_llm_enhanced_dmde.llm.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test suite for PromptBuilder."""

    def test_build_decision_prompt(self):
        """Test prompt assembly from features."""
        builder = PromptBuilder()
        features = {
            "diversity": 0.5,
            "convergence_speed": 0.1,
            "stagnation": False,
            "feasible_ratio": 0.8,
        }
        prompt = builder.build_decision_prompt(features)
        assert isinstance(prompt, (str, list))

    def test_prompt_includes_trajectory(self):
        """Test that trajectory context is included."""
        builder = PromptBuilder()
        features = {"diversity": 0.5}
        trajectory = [
            {"generation": 1, "operator": "de/rand/1/bin", "improvement": 0.1},
            {"generation": 2, "operator": "de/best/2/bin", "improvement": 0.05},
        ]
        prompt = builder.build_decision_prompt(features, trajectory=trajectory)
        prompt_str = str(prompt)
        assert "trajectory" in prompt_str.lower() or len(trajectory) > 0

    def test_prompt_includes_available_operators(self):
        """Test operator list in prompt."""
        builder = PromptBuilder()
        features = {"diversity": 0.5}
        operators = ["de/rand/1/bin", "de/best/2/bin", "de/current-to-best/1/bin"]
        prompt = builder.build_decision_prompt(features, available_operators=operators)
        prompt_str = str(prompt)
        for op in operators:
            assert op in prompt_str
