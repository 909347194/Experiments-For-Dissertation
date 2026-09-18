"""Regression tests for DMDE mapping and coordinate utilities."""

from __future__ import annotations

import unittest

import numpy as np

from algorithms.algorithm_dmde.representation.encoder import Gene
from algorithms.algorithm_dmde.representation.inverse_mapper import _perturb_srp_tour
from algorithms.algorithm_dmde.representation.repair_rules.nearest_match import (
    nearest_match_adaptive,
    temperature_to_top_k,
)
from environments.environment_dmde.dem_terrain import _degree_to_meters_at_lat_fallback
from utils.utils_dmde.coord_transform import (
    _utm_to_wgs84_raw,
    _utm_to_wgs84_raw_batch,
    _wgs84_to_utm_raw,
    _wgs84_to_utm_raw_batch,
    degree_to_meters_at_lat,
)


class _PerturbRng:
    """Minimal deterministic RNG for selecting a specified SRP perturbation."""

    def __init__(self, operation: str, indices: tuple[int, ...]) -> None:
        self.operation = operation
        self.indices = indices
        self._index_calls = 0

    def random(self) -> float:
        return 0.0

    def choice(self, values, size=None, replace=True, p=None):
        if isinstance(values, list) and values == ["swap", "insert", "reverse"]:
            return self.operation
        self._index_calls += 1
        if size is not None:
            return np.asarray(self.indices)
        return self.indices[self._index_calls - 1]


class SRPPerturbationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n_uavs = 1
        # 布局约定：行 0..n_uavs-1 = C_UT，行 n_uavs..n_uavs+n_targets-1 = C_TT。
        # 4 个 target → 需要 5 行。
        self.cost_matrix = np.arange(20, dtype=float).reshape(5, 4)

    def _genes(self) -> list[Gene]:
        return [
            Gene(0, 0, 0.0),
            Gene(-1, 1, -1.0),
            Gene(-1, 2, -1.0),
            Gene(-1, 3, -1.0),
        ]

    def _assert_costs_match_order(self, genes: list[Gene]) -> None:
        for predecessor, gene in zip(genes, genes[1:]):
            self.assertEqual(gene.cost, self.cost_matrix[self.n_uavs + predecessor.target_id, gene.target_id])

    def test_insert_recomputes_all_affected_tour_costs(self) -> None:
        genes = self._genes()
        _perturb_srp_tour(
            genes, self.cost_matrix, self.n_uavs, 3, 1.0,
            _PerturbRng("insert", (3, 1)),
        )
        self.assertEqual([gene.target_id for gene in genes], [0, 3, 1, 2])
        self._assert_costs_match_order(genes)

    def test_reverse_recomputes_all_affected_tour_costs(self) -> None:
        genes = self._genes()
        _perturb_srp_tour(
            genes, self.cost_matrix, self.n_uavs, 3, 1.0,
            _PerturbRng("reverse", (1, 3)),
        )
        self.assertEqual([gene.target_id for gene in genes], [0, 3, 2, 1])
        self._assert_costs_match_order(genes)


class AdaptiveMatchBoundaryTests(unittest.TestCase):
    def test_zero_temperature_is_greedy_and_ties_are_valid(self) -> None:
        matrix = np.array([[2.0, 2.0, np.inf]])
        row, col, cost = nearest_match_adaptive(
            2.0, matrix, np.zeros_like(matrix, dtype=bool), temperature=0.0,
            rng=np.random.default_rng(1),
        )
        self.assertEqual((row, col, cost), (0, 0, 2.0))

        result = nearest_match_adaptive(
            2.0, matrix, np.zeros_like(matrix, dtype=bool), temperature=1.0,
            rng=np.random.default_rng(1),
        )
        self.assertIn(result[1], (0, 1))

    def test_temperature_must_be_finite_and_in_range(self) -> None:
        for temperature in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                temperature_to_top_k(temperature)


class CoordinateBatchTests(unittest.TestCase):
    def test_batch_fallback_matches_scalar_in_both_hemispheres(self) -> None:
        for is_north, lats in (
            (True, np.array([[29.6, 29.7], [29.8, 29.9]])),
            (False, np.array([[-29.6, -29.7], [-29.8, -29.9]])),
        ):
            lons = np.array([[91.1, 91.3], [91.5, 91.7]])
            eastings, northings = _wgs84_to_utm_raw_batch(lons, lats, 46, is_north)
            expected = np.array([
                _wgs84_to_utm_raw(float(lon), float(lat), 46, is_north)
                for lon, lat in zip(lons.ravel(), lats.ravel())
            ])
            np.testing.assert_allclose(eastings.ravel(), expected[:, 0])
            np.testing.assert_allclose(northings.ravel(), expected[:, 1])

            recovered_lons, recovered_lats = _utm_to_wgs84_raw_batch(
                eastings, northings, 46, is_north
            )
            np.testing.assert_allclose(recovered_lons, lons)
            np.testing.assert_allclose(recovered_lats, lats)

    def test_dem_fallback_uses_coord_transform_formula(self) -> None:
        for latitude in (-45.0, 0.0, 29.6, 65.0):
            np.testing.assert_allclose(
                _degree_to_meters_at_lat_fallback(latitude),
                degree_to_meters_at_lat(latitude),
            )


if __name__ == "__main__":
    unittest.main()
