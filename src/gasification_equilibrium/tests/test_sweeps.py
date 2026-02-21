"""Tests for the sweep strategy module.

Tests verify:
1. Temperature sweep output shape and correctness
2. Surface sweep output shape
3. Warm-starting behavior
4. DbC precondition checks
"""

from typing import Any

import numpy as np
import pytest

from gasification_equilibrium.python.engine import GasificationEngine
from gasification_equilibrium.python.sweeps import surface_sweep, temperature_sweep


@pytest.fixture
def engine() -> GasificationEngine:
    """Engine with minimal species for fast sweeps."""
    return GasificationEngine(
        species_keys=["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]
    )


class TestTemperatureSweep:
    """Test temperature_sweep function."""

    def test_returns_correct_count(self, engine: Any) -> None:
        results = temperature_sweep(
            engine, 500, 1500, 10, feed={"C": 1.0, "H": 1.0, "O": 0.5}
        )
        assert len(results) == 10

    def test_temperatures_are_correct(self, engine: Any) -> None:
        results = temperature_sweep(
            engine, 500, 1500, 5, feed={"C": 1.0, "H": 1.0, "O": 0.5}
        )
        temps = [r.temperature for r in results]
        expected = np.linspace(500, 1500, 5)
        np.testing.assert_allclose(temps, expected, rtol=1e-5)

    def test_most_converged(self, engine: Any) -> None:
        results = temperature_sweep(
            engine, 600, 1400, 10, feed={"C": 1.0, "H": 1.0, "O": 0.5}
        )
        converged = sum(1 for r in results if r.converged)
        assert converged >= 8, f"Only {converged}/10 points converged"

    def test_h2_increases_with_temperature(self, engine: Any) -> None:
        """H2 production generally increases with temperature in gasification."""
        results = temperature_sweep(
            engine, 600, 1400, 20, feed={"C": 1.0, "H": 2.0, "O": 1.0}
        )
        h2_vals = [r.composition_dict().get("H2", 0) for r in results]
        assert h2_vals[-1] > h2_vals[0]

    def test_invalid_range_raises(self, engine: Any) -> None:
        with pytest.raises(AssertionError):
            temperature_sweep(engine, 1500, 500, 10, feed={"C": 1.0})

    def test_minimum_points_raises(self, engine: Any) -> None:
        with pytest.raises(AssertionError):
            temperature_sweep(engine, 500, 1500, 1, feed={"C": 1.0})

    def test_two_points_minimum(self, engine: Any) -> None:
        results = temperature_sweep(
            engine, 500, 1500, 2, feed={"C": 1.0, "H": 1.0, "O": 0.5}
        )
        assert len(results) == 2

    def test_results_have_correct_species(self, engine: Any) -> None:
        results = temperature_sweep(
            engine, 800, 1200, 3, feed={"C": 1.0, "H": 1.0, "O": 0.5}
        )
        for r in results:
            assert len(r.species) == engine.n_species


class TestSurfaceSweep:
    """Test surface_sweep function."""

    def test_output_shape(self, engine: Any) -> None:
        data = surface_sweep(
            engine,
            (600, 1200),
            "steam_carbon_ratio",
            (0.0, 2.0),
            n_t=5,
            n_param=4,
            feed={"C": 1.0, "O": 0.5},
        )
        assert data["compositions"].shape == (5, 4, engine.n_species)
        assert data["h2_co_ratio"].shape == (5, 4)
        assert data["carbon_conversion"].shape == (5, 4)
        assert data["cge"].shape == (5, 4)
        assert len(data["temperatures"]) == 5
        assert len(data["param_values"]) == 4

    def test_param_name_stored(self, engine: Any) -> None:
        data = surface_sweep(
            engine,
            (600, 1200),
            "pressure",
            (101325, 101325 * 10),
            n_t=3,
            n_param=3,
            feed={"C": 1.0, "H": 2.0, "O": 0.5},
        )
        assert data["param_name"] == "pressure"

    def test_species_stored(self, engine: Any) -> None:
        data = surface_sweep(
            engine,
            (600, 1200),
            "steam_carbon_ratio",
            (0.0, 2.0),
            n_t=3,
            n_param=3,
            feed={"C": 1.0, "O": 0.5},
        )
        assert data["species"] == engine.species_keys

    def test_temperatures_range(self, engine: Any) -> None:
        data = surface_sweep(
            engine,
            (600, 1200),
            "steam_carbon_ratio",
            (0.0, 2.0),
            n_t=5,
            n_param=3,
            feed={"C": 1.0, "O": 0.5},
        )
        assert data["temperatures"][0] == pytest.approx(600)
        assert data["temperatures"][-1] == pytest.approx(1200)

    def test_compositions_nonnegative(self, engine: Any) -> None:
        data = surface_sweep(
            engine,
            (600, 1200),
            "steam_carbon_ratio",
            (0.0, 2.0),
            n_t=3,
            n_param=3,
            feed={"C": 1.0, "O": 0.5},
        )
        assert np.all(data["compositions"] >= 0)
