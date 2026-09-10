# -*- coding: utf-8 -*-
"""Tests for trajectory collector module."""
import pytest
from algorithms.algorithm_llm_enhanced_dmde.features.trajectory_collector import TrajectoryCollector


class TestTrajectoryCollector:
    """Test suite for TrajectoryCollector."""

    @pytest.fixture
    def collector(self):
        """Create a TrajectoryCollector instance."""
        return TrajectoryCollector(max_history=5)

    def test_record_and_retrieve(self, collector):
        """Test recording entries and retrieving trajectory."""
        collector.record(generation=1, operator="de/rand/1/bin", fitness=50.0, improvement=0.1)
        collector.record(generation=2, operator="de/best/2/bin", fitness=49.0, improvement=0.02)
        trajectory = collector.get_trajectory()
        assert len(trajectory) == 2
        assert trajectory[0]["generation"] == 1
        assert trajectory[1]["generation"] == 2

    def test_max_history_limit(self, collector):
        """Test that old entries are dropped beyond max_history."""
        for i in range(10):
            collector.record(generation=i, operator="de/rand/1/bin", fitness=100 - i, improvement=1.0)
        trajectory = collector.get_trajectory()
        assert len(trajectory) <= 5
        # Should contain the most recent entries
        assert trajectory[-1]["generation"] == 9

    def test_empty_trajectory(self, collector):
        """Test behavior with no recorded data."""
        trajectory = collector.get_trajectory()
        assert isinstance(trajectory, list)
        assert len(trajectory) == 0

    def test_record_preserves_order(self, collector):
        """Test that entries are returned in chronological order."""
        collector.record(generation=3, operator="op3", fitness=30.0, improvement=0.3)
        collector.record(generation=1, operator="op1", fitness=10.0, improvement=0.1)
        collector.record(generation=2, operator="op2", fitness=20.0, improvement=0.2)
        trajectory = collector.get_trajectory()
        generations = [entry["generation"] for entry in trajectory]
        assert generations == [3, 1, 2]  # Insertion order preserved

    def test_reset(self, collector):
        """Test resetting the collector."""
        collector.record(generation=1, operator="op1", fitness=10.0, improvement=0.1)
        if hasattr(collector, "reset"):
            collector.reset()
            trajectory = collector.get_trajectory()
            assert len(trajectory) == 0
