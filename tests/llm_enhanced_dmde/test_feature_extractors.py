# -*- coding: utf-8 -*-
"""Tests for feature extraction modules."""
import pytest
import numpy as np
from algorithms.algorithm_llm_enhanced_dmde.features.population_features import compute_diversity, compute_gene_variance
from algorithms.algorithm_llm_enhanced_dmde.features.convergence_features import compute_convergence_speed, detect_stagnation
from algorithms.algorithm_llm_enhanced_dmde.features.constraint_features import compute_feasible_ratio


class TestPopulationFeatures:
    """Test suite for population feature extractors."""

    def test_compute_diversity(self):
        """Test diversity metric."""
        population = np.random.rand(10, 5)
        diversity = compute_diversity(population)
        assert isinstance(diversity, (float, np.floating))
        assert diversity >= 0

    def test_compute_diversity_identical(self):
        """Test diversity is zero for identical individuals."""
        population = np.ones((10, 5)) * 0.5
        diversity = compute_diversity(population)
        assert diversity == pytest.approx(0.0, abs=1e-10)

    def test_compute_gene_variance(self):
        """Test gene variance computation."""
        population = np.random.rand(10, 5)
        variance = compute_gene_variance(population)
        assert isinstance(variance, (float, np.floating))
        assert variance >= 0

    def test_compute_gene_variance_constant(self):
        """Test gene variance is zero for constant population."""
        population = np.ones((10, 5)) * 3.0
        variance = compute_gene_variance(population)
        assert variance == pytest.approx(0.0, abs=1e-10)


class TestConvergenceFeatures:
    """Test suite for convergence feature extractors."""

    def test_convergence_speed(self):
        """Test convergence speed calculation."""
        fitness_history = [100.0, 90.0, 85.0, 83.0, 82.5]
        speed = compute_convergence_speed(fitness_history)
        assert isinstance(speed, (float, np.floating))

    def test_convergence_speed_single_entry(self):
        """Test convergence speed with single fitness value."""
        fitness_history = [100.0]
        speed = compute_convergence_speed(fitness_history)
        assert isinstance(speed, (float, np.floating))

    def test_stagnation_detection(self):
        """Test stagnation detection."""
        # Flat fitness values should indicate stagnation
        flat_fitness = [82.5, 82.5, 82.5, 82.5, 82.5]
        result = detect_stagnation(flat_fitness)
        assert isinstance(result, (bool, np.bool_))

    def test_stagnation_detection_improving(self):
        """Test stagnation detection with improving fitness."""
        improving_fitness = [100.0, 90.0, 80.0, 70.0, 60.0]
        result = detect_stagnation(improving_fitness)
        assert isinstance(result, (bool, np.bool_))


class TestConstraintFeatures:
    """Test suite for constraint feature extractors."""

    def test_feasible_ratio(self):
        """Test feasible ratio calculation."""
        # Mock constraint violations: 0 means feasible, >0 means infeasible
        violations = np.array([0.0, 0.0, 0.5, 0.0, 1.0])
        ratio = compute_feasible_ratio(violations)
        assert isinstance(ratio, (float, np.floating))
        assert 0.0 <= ratio <= 1.0

    def test_feasible_ratio_all_feasible(self):
        """Test feasible ratio when all individuals are feasible."""
        violations = np.zeros(10)
        ratio = compute_feasible_ratio(violations)
        assert ratio == pytest.approx(1.0)

    def test_feasible_ratio_none_feasible(self):
        """Test feasible ratio when no individuals are feasible."""
        violations = np.ones(10)
        ratio = compute_feasible_ratio(violations)
        assert ratio == pytest.approx(0.0)
