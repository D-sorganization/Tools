"""Edge case and boundary value tests for SyngasCompressionEngine.

Covers:
  - Invalid / boundary pressure values
  - Invalid / boundary gamma (heat capacity ratio) values
  - Empty / single / multi-stage compression
  - Very high pressure ratios
  - Compression type validation
  - Multi-stage vs single-stage efficiency comparison

Design principles:
  - TDD: Tests describe the desired behaviour.
  - DRY: Common setup is shared via fixtures.
  - DbC: Each test documents pre/post-conditions.
  - Orthogonality: Each test class covers one category of edge cases.
"""

from __future__ import annotations

import math

import pytest
from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> SyngasCompressionEngine:
    """Provide a fresh SyngasCompressionEngine instance."""
    return SyngasCompressionEngine()


@pytest.fixture
def default_mixture_props() -> dict[str, float]:
    """Provide typical mixture properties for a diatomic-heavy syngas."""
    return {
        "molecular_weight": 15.0,
        "heat_capacity_ratio": 1.38,
    }


@pytest.fixture
def simple_stage() -> CompressionStage:
    """A basic isentropic compression stage from 1 bar to 3 bar."""
    return CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=3.0,
        inlet_temperature=313.15,  # 40 C in K
        efficiency=0.85,
        compression_type="isentropic",
    )


# ---------------------------------------------------------------------------
# Tests: Pressure validation in calculate_water_dropout
# ---------------------------------------------------------------------------


class TestWaterDropoutPressureValidation:
    """Edge cases for pressure values in calculate_water_dropout."""

    def test_negative_pressure_raises_value_error(self, engine):
        """Negative pressure must raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match="pressure must be > 0"):
            engine.calculate_water_dropout(
                temperature=313.15, pressure=-1.0, water_content=5.0
            )

    def test_zero_pressure_raises_value_error(self, engine):
        """Zero pressure must raise ValueError."""
        with pytest.raises(ValueError, match="pressure must be > 0"):
            engine.calculate_water_dropout(
                temperature=313.15, pressure=0.0, water_content=5.0
            )

    def test_very_small_positive_pressure(self, engine):
        """A very small but positive pressure should not raise."""
        result = engine.calculate_water_dropout(
            temperature=313.15, pressure=0.001, water_content=5.0
        )
        assert "water_dropout" in result
        assert math.isfinite(result["water_dropout"])

    def test_very_high_pressure(self, engine):
        """Very high pressure should produce valid results (no overflow)."""
        result = engine.calculate_water_dropout(
            temperature=313.15, pressure=1000.0, water_content=5.0
        )
        assert math.isfinite(result["water_dropout"])
        assert result["water_dropout"] >= 0


# ---------------------------------------------------------------------------
# Tests: Compression work edge cases
# ---------------------------------------------------------------------------


class TestCompressionWorkEdgeCases:
    """Edge cases for calculate_compression_work."""

    def test_unknown_compression_type_raises(self, engine, default_mixture_props):
        """An unrecognized compression type must raise ValueError."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="unknown_type",
        )
        with pytest.raises(ValueError, match="Unknown compression type"):
            engine.calculate_compression_work(stage, 100.0, default_mixture_props)

    @pytest.mark.parametrize(
        "compression_type", ["isentropic", "polytropic", "isothermal"]
    )
    def test_all_valid_compression_types_return_results(
        self, engine, default_mixture_props, compression_type
    ):
        """All three valid compression types should return a result dict."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type=compression_type,
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert "work_actual" in result
        assert "power_hp" in result
        assert math.isfinite(result["work_actual"])
        assert math.isfinite(result["power_hp"])

    def test_unity_pressure_ratio_zero_work(self, engine, default_mixture_props):
        """When outlet == inlet pressure, compression work should be zero."""
        stage = CompressionStage(
            inlet_pressure=5.0,
            outlet_pressure=5.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert result["work_actual"] == pytest.approx(0.0, abs=1e-6)
        assert result["heat_rise"] == pytest.approx(0.0, abs=1e-6)

    def test_very_high_pressure_ratio(self, engine, default_mixture_props):
        """A very high pressure ratio (100:1) should still produce finite results."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=100.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert math.isfinite(result["work_actual"])
        assert result["work_actual"] > 0
        assert math.isfinite(result["temp_out_actual"])
        assert result["temp_out_actual"] > stage.inlet_temperature

    def test_isothermal_has_no_heat_rise(self, engine, default_mixture_props):
        """Isothermal compression should have zero heat rise."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=10.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isothermal",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert result["heat_rise"] == pytest.approx(0.0, abs=1e-10)
        assert result["temp_out_actual"] == pytest.approx(stage.inlet_temperature)

    def test_zero_flow_rate_zero_power(self, engine, default_mixture_props):
        """Zero flow rate should produce zero power output."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 0.0, default_mixture_props)
        assert result["power_hp"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: Gamma (heat capacity ratio) edge cases
# ---------------------------------------------------------------------------


class TestGammaEdgeCases:
    """Edge cases for heat capacity ratio (gamma) in compression calculations."""

    @pytest.mark.parametrize("gamma", [0.0, -1.0])
    def test_non_positive_gamma_raises_or_produces_invalid(self, engine, gamma):
        """Gamma <= 0 should cause a ZeroDivisionError or math domain error.

        The formula has gamma/(gamma-1) which diverges at gamma=0 and gamma=1,
        and produces negative exponents for gamma < 0.
        """
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": gamma}
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        with pytest.raises((ZeroDivisionError, ValueError, OverflowError)):
            engine.calculate_compression_work(stage, 100.0, mixture_props)

    def test_gamma_equal_one_raises(self, engine):
        """Gamma = 1.0 causes division by zero in (gamma - 1) denominator."""
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.0}
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        with pytest.raises((ZeroDivisionError, ValueError)):
            engine.calculate_compression_work(stage, 100.0, mixture_props)

    def test_gamma_slightly_above_one(self, engine):
        """Gamma = 1.001 is valid but extreme; should produce finite results."""
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.001}
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, mixture_props)
        assert math.isfinite(result["work_actual"])
        assert result["work_actual"] > 0

    def test_monatomic_gamma(self, engine):
        """Monatomic gas gamma (5/3 ~ 1.667) should produce valid results."""
        mixture_props = {"molecular_weight": 39.948, "heat_capacity_ratio": 5 / 3}
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, mixture_props)
        assert math.isfinite(result["work_actual"])
        assert result["work_actual"] > 0


# ---------------------------------------------------------------------------
# Tests: Single-stage compression
# ---------------------------------------------------------------------------


class TestSingleStageCompression:
    """Tests for single-stage compression scenarios."""

    def test_single_stage_returns_reasonable_power(self, engine, simple_stage):
        """Single stage 1->3 bar with typical syngas should give positive power."""
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.38}
        result = engine.calculate_compression_work(simple_stage, 100.0, mixture_props)
        assert result["power_hp"] > 0
        assert result["pressure_ratio"] == pytest.approx(3.0)

    def test_single_stage_isentropic_work_less_than_actual(self, engine, simple_stage):
        """Isentropic work should always be less than actual work (efficiency < 1)."""
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.38}
        result = engine.calculate_compression_work(simple_stage, 100.0, mixture_props)
        assert result["work_isentropic"] is not None
        assert result["work_isentropic"] < result["work_actual"]


# ---------------------------------------------------------------------------
# Tests: Multi-stage vs single-stage comparison
# ---------------------------------------------------------------------------


class TestMultiStageVsSingleStage:
    """Compare multi-stage compression against single-stage equivalent."""

    def test_two_stages_less_work_than_single_equivalent(self, engine):
        """Two-stage compression with intercooling should require less total work
        than a single-stage compression to the same final pressure.

        Precondition: Same overall pressure ratio, intercooling enabled.
        Postcondition: Total multi-stage power < single-stage power.
        """
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.38}

        # Single stage: 1 bar -> 9 bar
        single_stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=9.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        single_result = engine.calculate_compression_work(
            single_stage, 100.0, mixture_props
        )

        # Two stages: 1->3 bar, 3->9 bar (with intercooling back to 313.15 K)
        stage1 = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.85,
            compression_type="isentropic",
        )
        stage2 = CompressionStage(
            inlet_pressure=3.0,
            outlet_pressure=9.0,
            inlet_temperature=313.15,  # Intercooled
            efficiency=0.85,
            compression_type="isentropic",
        )

        result1 = engine.calculate_compression_work(stage1, 100.0, mixture_props)
        result2 = engine.calculate_compression_work(stage2, 100.0, mixture_props)
        total_two_stage_power = result1["power_hp"] + result2["power_hp"]

        assert total_two_stage_power < single_result["power_hp"]

    def test_isothermal_less_work_than_isentropic(self, engine):
        """Isothermal compression always requires less work than isentropic.

        This is a thermodynamic identity for ideal gas compression.
        """
        mixture_props = {"molecular_weight": 15.0, "heat_capacity_ratio": 1.38}

        isothermal = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=10.0,
            inlet_temperature=313.15,
            efficiency=1.0,  # Perfect efficiency to compare work only
            compression_type="isothermal",
        )
        isentropic = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=10.0,
            inlet_temperature=313.15,
            efficiency=1.0,
            compression_type="isentropic",
        )

        iso_result = engine.calculate_compression_work(isothermal, 100.0, mixture_props)
        isen_result = engine.calculate_compression_work(
            isentropic, 100.0, mixture_props
        )

        assert iso_result["work_actual"] < isen_result["work_actual"]


# ---------------------------------------------------------------------------
# Tests: Efficiency boundary values
# ---------------------------------------------------------------------------


class TestEfficiencyBoundaries:
    """Edge cases for compressor efficiency values."""

    def test_perfect_efficiency(self, engine, default_mixture_props):
        """Efficiency = 1.0 means actual work equals isentropic work."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=1.0,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert result["work_actual"] == pytest.approx(
            result["work_isentropic"], rel=1e-10
        )

    def test_very_low_efficiency(self, engine, default_mixture_props):
        """Very low efficiency (1%) produces very high actual work."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.01,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert math.isfinite(result["work_actual"])
        # At 1% efficiency, actual work should be ~100x the isentropic work
        assert result["work_actual"] == pytest.approx(
            result["work_isentropic"] / 0.01, rel=1e-6
        )

    def test_zero_efficiency_raises(self, engine, default_mixture_props):
        """Zero efficiency causes division by zero."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=313.15,
            efficiency=0.0,
            compression_type="isentropic",
        )
        with pytest.raises(ZeroDivisionError):
            engine.calculate_compression_work(stage, 100.0, default_mixture_props)


# ---------------------------------------------------------------------------
# Tests: Temperature edge cases
# ---------------------------------------------------------------------------


class TestTemperatureEdgeCases:
    """Edge cases for inlet temperature values."""

    def test_very_low_inlet_temperature(self, engine, default_mixture_props):
        """Cryogenic inlet temperature (77 K) should produce valid results."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=77.0,  # Liquid nitrogen temperature
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert math.isfinite(result["work_actual"])
        assert result["work_actual"] > 0
        # Lower inlet T -> lower work for same pressure ratio
        assert result["temp_out_actual"] > 77.0

    def test_very_high_inlet_temperature(self, engine, default_mixture_props):
        """Very high inlet temperature (1000 K) should produce finite results."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=1000.0,
            efficiency=0.85,
            compression_type="isentropic",
        )
        result = engine.calculate_compression_work(stage, 100.0, default_mixture_props)
        assert math.isfinite(result["work_actual"])
        assert result["temp_out_actual"] > 1000.0
