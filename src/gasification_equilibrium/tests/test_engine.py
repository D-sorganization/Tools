"""Tests for the GasificationEngine orchestrator.

Tests verify:
1. Engine initialization and species management
2. Single-point equilibrium via engine API
3. Feed modes (dict, mass_fractions, ProcessInputs)
4. Legacy parameter compatibility
5. Temperature and surface sweep delegation
6. EquilibriumResult dataclass methods
7. Design by Contract assertions
"""

from typing import Any

import numpy as np
import pytest

from gasification_equilibrium.python.engine import GasificationEngine
from gasification_equilibrium.python.feed import (
    FEED_PRESETS,
    ProcessInputs,
    feed_from_preset,
)
from gasification_equilibrium.python.thermo_data import SPECIES_DB


@pytest.fixture
def engine() -> GasificationEngine:
    """Create a default engine instance."""
    return GasificationEngine()


@pytest.fixture
def simple_engine() -> GasificationEngine:
    """Create engine with minimal species for fast testing."""
    return GasificationEngine(
        species_keys=["H2", "CO", "CO2", "H2O", "CH4", "N2", "C_solid"]
    )


class TestEngineInit:
    """Test engine initialization."""

    def test_default_init(self, engine: GasificationEngine) -> None:
        assert engine.n_species == len(SPECIES_DB)

    def test_custom_species(self) -> None:
        eng = GasificationEngine(species_keys=["H2", "CO", "CO2", "H2O"])
        assert eng.n_species == 4

    def test_invalid_species_raises(self) -> None:
        with pytest.raises(AssertionError):
            GasificationEngine(species_keys=["INVALID_SPECIES"])

    def test_matrix_available(self, engine: GasificationEngine) -> None:
        assert engine.matrix is not None
        assert engine.matrix.n_species == engine.n_species

    def test_species_keys_list(self, engine: GasificationEngine) -> None:
        assert isinstance(engine.species_keys, list)
        assert len(engine.species_keys) == engine.n_species


class TestSinglePointEquilibrium:
    """Test single-point equilibrium calculations via engine.solve()."""

    def test_converges_at_1000k(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        assert result.converged, (
            f"Failed to converge: balance_err={result.element_balance_error}"
        )

    def test_converges_at_1500k(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1500, feed={"C": 1.0, "H": 2.0, "O": 0.5})
        assert result.converged

    def test_converges_at_500k(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=500, feed={"C": 1.0, "H": 2.0, "O": 1.0})
        assert result.converged

    def test_element_balance_conserved(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 2.0, "O": 1.0})
        assert result.element_balance_error < 1e-6

    def test_mole_fractions_sum_to_one(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        gas_species = [
            k for k in engine.species_keys if SPECIES_DB[k]["phase"] == "gas"
        ]
        gas_fracs = sum(result.composition_dict().get(k, 0) for k in gas_species)
        assert abs(gas_fracs - 1.0) < 0.01

    def test_mole_fractions_nonnegative(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        assert np.all(result.mole_fractions >= 0)

    def test_result_has_all_fields(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        assert result.temperature == 1000
        assert result.pressure > 0
        assert len(result.species) == engine.n_species
        assert isinstance(result.h2_co_ratio, float)
        assert isinstance(result.cold_gas_efficiency, float)
        assert isinstance(result.carbon_conversion, float)
        assert isinstance(result.feed_elements, dict)


class TestFeedModes:
    """Test different feed input modes."""

    def test_dict_feed(self, simple_engine: GasificationEngine) -> None:
        result = simple_engine.solve(
            temperature=1000, feed={"C": 1.0, "H": 2.0, "O": 1.0}
        )
        assert result.converged

    def test_mass_fraction_feed(self, simple_engine: GasificationEngine) -> None:
        result = simple_engine.solve(
            temperature=1000,
            feed_mass={
                "C": 0.75,
                "H": 0.05,
                "O": 0.08,
                "N": 0.02,
                "S": 0.01,
                "Ash": 0.09,
            },
        )
        assert result.converged

    def test_process_inputs_feed(self, simple_engine: GasificationEngine) -> None:
        pi = ProcessInputs()
        pi.steam.flow = 1.0
        pi.oxidant.o2_flow = 0.3
        result = simple_engine.solve(
            temperature=1000,
            feed={"C": 1.0, "H": 1.0, "O": 0.5},
            process_inputs=pi,
        )
        assert result.converged

    def test_default_feed(self, simple_engine: GasificationEngine) -> None:
        """Without specifying feed, uses default C=1.0, H=0.1, O=0.5."""
        result = simple_engine.solve(temperature=1000)
        assert result.converged


class TestLegacyParams:
    """Test backward-compatible steam/oxygen ratio parameters."""

    def test_steam_carbon_ratio(self, simple_engine: GasificationEngine) -> None:
        r_no_steam = simple_engine.solve(
            temperature=1000,
            feed={"C": 1.0, "O": 0.5},
            steam_carbon_ratio=0.0,
        )
        r_with_steam = simple_engine.solve(
            temperature=1000,
            feed={"C": 1.0, "O": 0.5},
            steam_carbon_ratio=1.0,
        )
        h2_no = r_no_steam.composition_dict().get("H2", 0)
        h2_yes = r_with_steam.composition_dict().get("H2", 0)
        assert h2_yes > h2_no

    def test_oxygen_carbon_ratio(self, simple_engine: GasificationEngine) -> None:
        r = simple_engine.solve(
            temperature=1000,
            feed={"C": 1.0, "H": 1.0},
            oxygen_carbon_ratio=0.5,
        )
        assert result_has_oxygen(r)

    def test_equivalence_ratio(self, simple_engine: GasificationEngine) -> None:
        r = simple_engine.solve(
            temperature=1000,
            feed={"C": 1.0, "H": 1.0, "O": 0.5},
            equivalence_ratio=2.0,
        )
        assert r.converged


def result_has_oxygen(result: Any) -> bool:
    """Helper: check if result feed has oxygen."""
    return bool(result.feed_elements.get("O", 0) > 0)


class TestKnownEquilibria:
    """Test engine against known equilibrium behavior."""

    def test_boudouard_favors_co_at_high_t(
        self, simple_engine: GasificationEngine
    ) -> None:
        result = simple_engine.solve(temperature=1200, feed={"C": 2.0, "O": 2.0})
        comp = result.composition_dict()
        assert comp.get("CO", 0) > comp.get("CO2", 0)

    def test_boudouard_favors_co2_at_low_t(
        self, simple_engine: GasificationEngine
    ) -> None:
        result = simple_engine.solve(temperature=400, feed={"C": 2.0, "O": 2.0})
        comp = result.composition_dict()
        assert comp.get("CO2", 0) > comp.get("CO", 0)

    def test_methanation_favored_at_low_t(
        self, simple_engine: GasificationEngine
    ) -> None:
        r_low = simple_engine.solve(temperature=500, feed={"C": 1.0, "H": 4.0})
        r_high = simple_engine.solve(temperature=1200, feed={"C": 1.0, "H": 4.0})
        ch4_low = r_low.composition_dict().get("CH4", 0)
        ch4_high = r_high.composition_dict().get("CH4", 0)
        assert ch4_low > ch4_high

    def test_higher_pressure_favors_ch4(
        self, simple_engine: GasificationEngine
    ) -> None:
        r_low = simple_engine.solve(
            temperature=800, pressure=101325, feed={"C": 1.0, "H": 4.0}
        )
        r_high = simple_engine.solve(
            temperature=800, pressure=101325 * 30, feed={"C": 1.0, "H": 4.0}
        )
        ch4_low = r_low.composition_dict().get("CH4", 0)
        ch4_high = r_high.composition_dict().get("CH4", 0)
        assert ch4_high > ch4_low

    def test_carbon_conversion_increases_with_steam(
        self, simple_engine: GasificationEngine
    ) -> None:
        r1 = simple_engine.solve(temperature=1000, feed={"C": 1.0, "H": 0.5, "O": 0.25})
        r2 = simple_engine.solve(temperature=1000, feed={"C": 1.0, "H": 2.0, "O": 1.0})
        assert r2.carbon_conversion >= r1.carbon_conversion - 0.05


class TestTemperatureSweep:
    """Test temperature sweep via engine API."""

    def test_returns_correct_count(self, simple_engine: GasificationEngine) -> None:
        results = simple_engine.temperature_sweep(
            t_start=500,
            t_end=1500,
            n_points=10,
            feed={"C": 1.0, "H": 1.0, "O": 0.5},
        )
        assert len(results) == 10

    def test_temperatures_correct(self, simple_engine: GasificationEngine) -> None:
        results = simple_engine.temperature_sweep(
            t_start=500,
            t_end=1500,
            n_points=5,
            feed={"C": 1.0, "H": 1.0, "O": 0.5},
        )
        temps = [r.temperature for r in results]
        expected = np.linspace(500, 1500, 5)
        np.testing.assert_allclose(temps, expected, rtol=1e-5)

    def test_invalid_range_raises(self, simple_engine: GasificationEngine) -> None:
        with pytest.raises(AssertionError):
            simple_engine.temperature_sweep(
                t_start=1500, t_end=500, n_points=10, feed={"C": 1.0}
            )


class TestSurfaceSweep:
    """Test surface sweep via engine API."""

    def test_shape(self, simple_engine: GasificationEngine) -> None:
        data = simple_engine.surface_sweep(
            t_range=(600, 1200),
            param_name="steam_carbon_ratio",
            param_range=(0.0, 2.0),
            feed={"C": 1.0, "O": 0.5},
            n_t=5,
            n_param=4,
        )
        assert data["compositions"].shape == (5, 4, simple_engine.n_species)
        assert data["h2_co_ratio"].shape == (5, 4)
        assert len(data["temperatures"]) == 5
        assert len(data["param_values"]) == 4


class TestEquilibriumResult:
    """Test EquilibriumResult dataclass methods."""

    def test_composition_dict(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        comp = result.composition_dict()
        assert isinstance(comp, dict)
        assert len(comp) == engine.n_species

    def test_moles_dict(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 1.0})
        moles = result.moles_dict()
        assert isinstance(moles, dict)
        assert all(v >= 0 for v in moles.values())

    def test_dry_mole_fractions(self, engine: GasificationEngine) -> None:
        result = engine.solve(temperature=1000, feed={"C": 1.0, "H": 2.0, "O": 1.0})
        dry = result.dry_mole_fractions()
        assert "H2O" not in dry
        total = sum(dry.values())
        assert abs(total - 1.0) < 0.02


class TestContractViolations:
    """Test Design by Contract precondition checks."""

    def test_negative_temperature_raises(self, engine: GasificationEngine) -> None:
        with pytest.raises(AssertionError):
            engine.solve(temperature=-100, feed={"C": 1.0})

    def test_zero_temperature_raises(self, engine: GasificationEngine) -> None:
        with pytest.raises(AssertionError):
            engine.solve(temperature=0, feed={"C": 1.0})

    def test_negative_pressure_raises(self, engine: GasificationEngine) -> None:
        with pytest.raises(AssertionError):
            engine.solve(temperature=1000, pressure=-1, feed={"C": 1.0})

    def test_zero_pressure_raises(self, engine: GasificationEngine) -> None:
        with pytest.raises(AssertionError):
            engine.solve(temperature=1000, pressure=0, feed={"C": 1.0})


class TestFeedPresets:
    """Test feed presets through the engine."""

    def test_all_presets_solve(self, engine: GasificationEngine) -> None:
        for name in FEED_PRESETS:
            fc = feed_from_preset(name)
            result = engine.solve(temperature=1000, feed=fc.as_dict())
            assert result.converged, f"Preset '{name}' failed at 1000K"

    def test_bituminous_coal_with_steam(self, engine: GasificationEngine) -> None:
        """Coal + steam gasification should produce syngas."""
        fc = feed_from_preset("Bituminous Coal")
        feed = fc.as_dict()
        # Add steam as gasification agent (S/C ~ 1)
        c_moles = feed.get("C", 0)
        feed["H"] = feed.get("H", 0) + c_moles * 2
        feed["O"] = feed.get("O", 0) + c_moles
        result = engine.solve(temperature=1200, feed=feed)
        comp = result.composition_dict()
        assert comp.get("H2", 0) > 0.01 or comp.get("CO", 0) > 0.01


class TestWarmStart:
    """Test warm-starting from previous solution."""

    def test_warm_start_converges(self, simple_engine: GasificationEngine) -> None:
        r1 = simple_engine.solve(temperature=1000, feed={"C": 1.0, "H": 1.0, "O": 0.5})
        r2 = simple_engine.solve(
            temperature=1010,
            feed={"C": 1.0, "H": 1.0, "O": 0.5},
            warm_start=r1.moles,
        )
        assert r2.converged
