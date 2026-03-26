"""Comprehensive tests for WGSReactorEngine.

Tests cover equilibrium constant calculation, initial mole preparation,
equilibrium composition, reactor sizing, and edge cases. Only the engine
(non-GUI) is tested.
"""

from __future__ import annotations

from upstream_drift_tools.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)

# ─── Construction ─────────────────────────────────────────────


class TestWGSEngineConstruction:
    """Test WGSReactorEngine instantiation."""

    def test_creates_with_no_args(self) -> None:
        engine = WGSReactorEngine()
        assert engine is not None

    def test_has_gas_constant(self) -> None:
        engine = WGSReactorEngine()
        assert abs(engine.R - 8.314) < 0.01

    def test_catalysts_is_dict(self) -> None:
        engine = WGSReactorEngine()
        assert isinstance(engine.catalysts, dict)

    def test_species_db_not_none(self) -> None:
        engine = WGSReactorEngine()
        assert engine.species_db is not None


# ─── Equilibrium Constant ────────────────────────────────────


class TestEquilibriumConstant:
    """Test calculate_equilibrium_constant."""

    def test_returns_positive(self) -> None:
        engine = WGSReactorEngine()
        K = engine.calculate_equilibrium_constant(700.0)
        assert K > 0.0

    def test_higher_temp_lower_K(self) -> None:
        """WGS is exothermic: K_eq should decrease with temperature."""
        engine = WGSReactorEngine()
        K_low = engine.calculate_equilibrium_constant(500.0)
        K_high = engine.calculate_equilibrium_constant(800.0)
        assert K_low > K_high

    def test_typical_hts_range(self) -> None:
        """At HTS conditions (~673K), K should be reasonable (roughly 10-100)."""
        engine = WGSReactorEngine()
        K = engine.calculate_equilibrium_constant(673.0)
        assert 1.0 < K < 200.0

    def test_typical_lts_range(self) -> None:
        """At LTS conditions (~473K), K should be higher than HTS."""
        engine = WGSReactorEngine()
        K_lts = engine.calculate_equilibrium_constant(473.0)
        K_hts = engine.calculate_equilibrium_constant(673.0)
        assert K_lts > K_hts

    def test_returns_float(self) -> None:
        engine = WGSReactorEngine()
        K = engine.calculate_equilibrium_constant(600.0)
        assert isinstance(K, float)


# ─── Prepare Initial Moles ───────────────────────────────────


class TestPrepareInitialMoles:
    """Test the static _prepare_initial_moles method."""

    def test_basic_composition(self) -> None:
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        n_CO, n_H2O, n_CO2, n_H2, n_total = WGSReactorEngine._prepare_initial_moles(
            inlet, steam_ratio=2.0
        )
        assert n_CO == 25.0
        assert n_H2O == 5.0 + 25.0 * 2.0  # 55.0
        assert n_CO2 == 10.0
        assert n_H2 == 20.0
        assert n_total == 25.0 + 55.0 + 10.0 + 20.0

    def test_zero_steam_ratio(self) -> None:
        inlet = {"CO": 10.0, "H2O": 5.0}
        n_CO, n_H2O, _, _, _ = WGSReactorEngine._prepare_initial_moles(inlet, steam_ratio=0.0)
        assert n_CO == 10.0
        assert n_H2O == 5.0  # No additional steam

    def test_missing_species_default_zero(self) -> None:
        inlet = {"CO": 10.0}
        n_CO, n_H2O, n_CO2, n_H2, _ = WGSReactorEngine._prepare_initial_moles(
            inlet, steam_ratio=1.0
        )
        assert n_CO == 10.0
        assert n_H2O == 10.0  # 0 + 10*1.0
        assert n_CO2 == 0.0
        assert n_H2 == 0.0

    def test_empty_inlet(self) -> None:
        inlet: dict[str, float] = {}
        _, _, _, _, n_total = WGSReactorEngine._prepare_initial_moles(inlet, steam_ratio=2.0)
        assert n_total == 0.0


# ─── Equilibrium Composition ─────────────────────────────────


class TestEquilibriumComposition:
    """Test calculate_equilibrium_composition."""

    def test_basic_calculation_returns_expected_keys(self) -> None:
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        result = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0)
        required_keys = [
            "conversion",
            "composition",
            "h2_co_ratio",
            "equilibrium_constant",
            "heat_released",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_composition_has_all_species(self) -> None:
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        result = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0)
        for species in ["CO", "H2", "CO2", "H2O"]:
            assert species in result["composition"]

    def test_conversion_between_0_and_100(self) -> None:
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        result = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0)
        assert 0.0 <= result["conversion"] <= 100.0

    def test_h2_co_ratio_positive(self) -> None:
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        result = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0)
        assert result["h2_co_ratio"] > 0.0

    def test_heat_released_positive(self) -> None:
        """WGS is exothermic — heat should be released."""
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 10.0, "H2O": 5.0}
        result = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0)
        assert result["heat_released"] >= 0.0

    def test_empty_inlet_returns_zero_conversion(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition({}, 673.0, 25.0)
        assert result["conversion"] == 0.0

    def test_high_steam_ratio_increases_conversion(self) -> None:
        """Higher steam ratio should push equilibrium toward higher conversion."""
        engine = WGSReactorEngine()
        inlet = {"CO": 25.0, "H2": 20.0, "CO2": 5.0}
        low = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0, steam_ratio=1.0)
        high = engine.calculate_equilibrium_composition(inlet, 673.0, 25.0, steam_ratio=4.0)
        assert high["conversion"] >= low["conversion"]


# ─── Reactor Sizing ──────────────────────────────────────────


class TestReactorSizing:
    """Test size_wgs_reactor."""

    def test_returns_expected_keys(self) -> None:
        engine = WGSReactorEngine()
        sizing = engine.size_wgs_reactor(
            feed_rate=100.0,
            conversion=50.0,
            temperature=673.0,
            catalyst_type="Fe-Cr",
        )
        required = [
            "reactor_volume",
            "catalyst_volume",
            "diameter",
            "length",
            "heat_duty",
            "ghsv",
        ]
        for key in required:
            assert key in sizing, f"Missing key: {key}"

    def test_volumes_positive(self) -> None:
        engine = WGSReactorEngine()
        sizing = engine.size_wgs_reactor(100.0, 50.0, 673.0, "Fe-Cr")
        assert sizing["reactor_volume"] > 0.0
        assert sizing["catalyst_volume"] > 0.0

    def test_catalyst_volume_less_than_reactor(self) -> None:
        engine = WGSReactorEngine()
        sizing = engine.size_wgs_reactor(100.0, 50.0, 673.0, "Fe-Cr")
        assert sizing["catalyst_volume"] < sizing["reactor_volume"]

    def test_length_greater_than_diameter(self) -> None:
        """L/D ratio should be ~3, so length > diameter."""
        engine = WGSReactorEngine()
        sizing = engine.size_wgs_reactor(100.0, 50.0, 673.0, "Fe-Cr")
        assert sizing["length"] > sizing["diameter"]

    def test_heat_duty_positive(self) -> None:
        engine = WGSReactorEngine()
        sizing = engine.size_wgs_reactor(100.0, 50.0, 673.0, "Fe-Cr")
        assert sizing["heat_duty"] > 0.0

    def test_larger_feed_larger_reactor(self) -> None:
        engine = WGSReactorEngine()
        small = engine.size_wgs_reactor(50.0, 50.0, 673.0, "Fe-Cr")
        large = engine.size_wgs_reactor(200.0, 50.0, 673.0, "Fe-Cr")
        assert large["reactor_volume"] > small["reactor_volume"]


# ─── Assemble Equilibrium Results ─────────────────────────────


class TestAssembleEquilibriumResults:
    """Test the static _assemble_equilibrium_results method."""

    def test_composition_sums_to_100(self) -> None:
        result = WGSReactorEngine._assemble_equilibrium_results(
            x_eq=5.0,
            n_CO_0=25.0,
            n_H2O_0=50.0,
            n_CO2_0=10.0,
            n_H2_0=20.0,
            K_eq=10.0,
        )
        total = sum(result["composition"].values())
        assert abs(total - 100.0) < 0.01

    def test_conversion_calculation(self) -> None:
        result = WGSReactorEngine._assemble_equilibrium_results(
            x_eq=10.0, n_CO_0=25.0, n_H2O_0=50.0, n_CO2_0=5.0, n_H2_0=15.0, K_eq=5.0
        )
        expected_conversion = (10.0 / 25.0) * 100
        assert abs(result["conversion"] - expected_conversion) < 0.01

    def test_co_decreases_with_reaction(self) -> None:
        result = WGSReactorEngine._assemble_equilibrium_results(
            x_eq=5.0, n_CO_0=25.0, n_H2O_0=50.0, n_CO2_0=10.0, n_H2_0=20.0, K_eq=10.0
        )
        # CO should decrease: (25-5)/(25-5+50-5+10+5+20+5) = 20/100
        assert result["composition"]["CO"] < 25.0
