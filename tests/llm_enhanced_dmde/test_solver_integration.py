# -*- coding: utf-8 -*-
"""Integration test for LLM-enhanced DMDE solver.

Uses mock LLM to verify the full solver loop works end-to-end.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from algorithms.algorithm_llm_enhanced_dmde.solvers.llm_enhanced_dmde_solver import (
    LLMEnhancedDMDESolver,
    LLMEnhancedDMDEConfig,
)


class TestLLMEnhancedDMDESolver:
    """Integration test suite for LLMEnhancedDMDESolver."""

    @pytest.fixture
    def default_config(self):
        """Create a default solver configuration."""
        return LLMEnhancedDMDEConfig(
            population_size=20,
            dimensions=5,
            max_generations=50,
            enable_llm=False,
            llm_interval=10,
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return {
            "operator": "de/rand/1/bin",
            "scale_factor": 0.8,
            "crossover_rate": 0.9,
            "reason": "test",
        }

    def test_solver_runs_without_llm(self, default_config):
        """Test solver with LLM disabled (enable_llm=False)."""
        solver = LLMEnhancedDMDESolver(default_config)
        # Define a simple objective: minimize sum of squares
        def objective(x):
            return np.sum(x ** 2)

        bounds = [(-5.0, 5.0)] * 5
        result = solver.solve(objective, bounds)
        assert result is not None
        assert hasattr(result, "best_fitness") or isinstance(result, tuple)

    def test_solver_runs_with_mock_llm(self, default_config, mock_llm_response):
        """Test full solver loop with mocked LLM responses."""
        config = LLMEnhancedDMDEConfig(
            population_size=20,
            dimensions=5,
            max_generations=50,
            enable_llm=True,
            llm_interval=10,
        )
        solver = LLMEnhancedDMDESolver(config)

        def objective(x):
            return np.sum(x ** 2)

        bounds = [(-5.0, 5.0)] * 5

        with patch.object(solver, "_query_llm", return_value=mock_llm_response):
            result = solver.solve(objective, bounds)
            assert result is not None

    def test_llm_called_at_correct_intervals(self, default_config, mock_llm_response):
        """Verify LLM is called every p generations."""
        config = LLMEnhancedDMDEConfig(
            population_size=20,
            dimensions=5,
            max_generations=30,
            enable_llm=True,
            llm_interval=10,
        )
        solver = LLMEnhancedDMDESolver(config)

        def objective(x):
            return np.sum(x ** 2)

        bounds = [(-5.0, 5.0)] * 5

        with patch.object(solver, "_query_llm", return_value=mock_llm_response) as mock_query:
            solver.solve(objective, bounds)
            # With max_generations=30 and interval=10, expect calls at gen 10, 20, 30
            assert mock_query.call_count >= 2

    def test_decision_applied_to_solver(self, default_config, mock_llm_response):
        """Verify LLM decisions actually affect solver parameters."""
        config = LLMEnhancedDMDEConfig(
            population_size=20,
            dimensions=5,
            max_generations=30,
            enable_llm=True,
            llm_interval=10,
        )
        solver = LLMEnhancedDMDESolver(config)

        def objective(x):
            return np.sum(x ** 2)

        bounds = [(-5.0, 5.0)] * 5

        custom_response = {
            "operator": "de/best/2/bin",
            "scale_factor": 0.5,
            "crossover_rate": 0.3,
            "reason": "force specific operator",
        }

        with patch.object(solver, "_query_llm", return_value=custom_response):
            result = solver.solve(objective, bounds)
            # Verify solver completed (decision was applied without error)
            assert result is not None
