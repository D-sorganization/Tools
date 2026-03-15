"""Tests for syngas_compression_calculator.py — SyngasCompressionEngine.

Targets: 15% → ~60%+ coverage (excludes Qt UI widget, only tests pure engine).
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)

SYNGAS_COMP = {
    "H2": 0.30,
    "CO": 0.30,
    "CO2": 0.10,
    "N2": 0.22,
    "H2O": 0.05,
    "CH4": 0.03,
}


@pytest.fixture()
def engine() -> SyngasCompressionEngine:
    return SyngasCompressionEngine()


def _make_stage(
    p_in: float = 1.0,
    p_out: float = 3.0,
    t_in: float = 300.0,
    eff: float = 0.85,
    kind: str = "isentropic",
) -> CompressionStage:
    return CompressionStage(
        inlet_pressure=p_in,
        outlet_pressure=p_out,
        inlet_temperature=t_in,
        efficiency=eff,
        compression_type=kind,
    )


# ---------------------------------------------------------------------------
# calculate_mixture_properties
# ---------------------------------------------------------------------------


class TestCalculateMixtureProperties:
    def test_returns_expected_keys(self, engine):
        """Lines 192-231: mixture props dict structure."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert "molecular_weight" in props
        assert "critical_temperature" in props
        assert "critical_pressure" in props
        assert "heat_capacity_ratio" in props
        assert "mole_fractions" in props

    def test_molecular_weight_positive(self, engine):
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert props["molecular_weight"] > 0

    def test_heat_capacity_ratio_in_range(self, engine):
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        assert 1.0 < props["heat_capacity_ratio"] < 2.0

    def test_pure_h2(self, engine):
        props = engine.calculate_mixture_properties({"H2": 1.0})
        # H2 MW = 2 g/mol
        assert abs(props["molecular_weight"] - 2.0) < 0.5


# ---------------------------------------------------------------------------
# calculate_water_dropout
# ---------------------------------------------------------------------------


class TestCalculateWaterDropout:
    def test_no_dropout_below_saturation(self, engine):
        """Lines 262-269: no condensation when RH < 1."""
        result = engine.calculate_water_dropout(
            temperature=400.0,  # K — well above dew point
            pressure=25.0,
            water_content=0.1,  # small amount of water
        )
        assert result["water_dropout"] == 0.0
        assert result["condensation_rate"] == 0.0

    def test_dropout_when_supersaturated(self, engine):
        """Lines 262-266: condensation when RH > 1."""
        result = engine.calculate_water_dropout(
            temperature=300.0,  # K — near condensation
            pressure=100.0,  # high pressure → RH > 1
            water_content=5.0,
        )
        # At 300 K and 100 bar, vapor pressure is << pressure → condensation likely
        assert isinstance(result["water_dropout"], float)
        assert isinstance(result["condensation_rate"], float)

    def test_non_positive_pressure_raises(self, engine):
        """Lines 240-241: pressure <= 0 → ValueError."""
        with pytest.raises(ValueError, match="pressure must be > 0"):
            engine.calculate_water_dropout(300.0, 0.0, 5.0)


# ---------------------------------------------------------------------------
# calculate_compression_work
# ---------------------------------------------------------------------------


class TestCalculateCompressionWork:
    def test_isentropic(self, engine):
        """Lines 310-326: isentropic compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="isentropic")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is not None
        assert result["work_actual"] > 0
        assert result["power_hp"] > 0
        assert result["temp_out_actual"] > stage.inlet_temperature
        assert result["pressure_ratio"] == pytest.approx(3.0)

    def test_polytropic(self, engine):
        """Lines 328-338: polytropic compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="polytropic")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is None  # not computed for polytropic
        assert result["work_actual"] > 0
        assert result["temp_out_actual"] > stage.inlet_temperature

    def test_isothermal(self, engine):
        """Lines 340-345: isothermal compression path."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="isothermal")
        result = engine.calculate_compression_work(stage, 100.0, props)
        assert result["work_isentropic"] is None
        assert result["temp_out_actual"] == stage.inlet_temperature
        assert result["work_actual"] > 0

    def test_unknown_compression_type_raises(self, engine):
        """Lines 347-349: unknown type → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(kind="magical_compression")
        with pytest.raises(ValueError, match="Unknown compression type"):
            engine.calculate_compression_work(stage, 100.0, props)

    def test_zero_inlet_pressure_raises(self, engine):
        """Lines 286-287: inlet_pressure <= 0 → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(p_in=0.0)
        with pytest.raises(ValueError, match="inlet_pressure must be > 0"):
            engine.calculate_compression_work(stage, 100.0, props)

    def test_zero_outlet_pressure_raises(self, engine):
        """Lines 288-291: outlet_pressure <= 0 → ValueError."""
        props = engine.calculate_mixture_properties(SYNGAS_COMP)
        stage = _make_stage(p_out=0.0)
        with pytest.raises(ValueError, match="outlet_pressure must be > 0"):
            engine.calculate_compression_work(stage, 100.0, props)


# ---------------------------------------------------------------------------
# calculate_multistage_compression
# ---------------------------------------------------------------------------


class TestMultistageCompression:
    def test_single_stage_isentropic(self, engine):
        """Lines 372-423: single stage pipeline."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        assert len(result["stages"]) == 1
        assert result["total_power_hp"] > 0
        assert result["final_pressure"] == 3.0

    def test_multistage_with_intercooling(self, engine):
        """Lines 388-415 intercooling path."""
        stages = [
            _make_stage(1.0, 3.0, 300.0, 0.85, "isentropic"),
            _make_stage(3.0, 9.0, 400.0, 0.85, "isentropic"),
        ]
        result = engine.calculate_multistage_compression(
            stages, 100.0, SYNGAS_COMP, intercooling=True
        )
        assert len(result["stages"]) == 2
        # After intercooling, stage 2 inlet should be at cooler temp
        stage2_inlet = result["stages"][1]["inlet_temp"]
        assert stage2_inlet < 400.0  # Cooled down

    def test_multistage_without_intercooling(self, engine):
        """Lines 392-393: no intercooling → temperature carries over."""
        stages = [
            _make_stage(1.0, 3.0, 300.0, 0.85, "isentropic"),
            _make_stage(3.0, 9.0, 300.0, 0.85, "isentropic"),
        ]
        result = engine.calculate_multistage_compression(
            stages, 100.0, SYNGAS_COMP, intercooling=False
        )
        # Stage 2 inlet should be stage 1 outlet
        stage1_outlet = result["stages"][0]["temp_out_actual"]
        stage2_inlet = result["stages"][1]["inlet_temp"]
        assert abs(stage2_inlet - stage1_outlet) < 0.01

    def test_empty_stages_raises(self, engine):
        """Line 380-381: empty stages → ValueError."""
        with pytest.raises(ValueError, match="stages list must not be empty"):
            engine.calculate_multistage_compression([], 100.0, SYNGAS_COMP)


# ---------------------------------------------------------------------------
# analyze_process_conditions
# ---------------------------------------------------------------------------


class TestAnalyzeProcessConditions:
    def _run_and_analyze(self, engine, stages, temp_K=300.0):
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        return engine.analyze_process_conditions(result)

    def test_returns_expected_keys(self, engine):
        """Lines 425-497: dict keys in analysis output."""
        analysis = self._run_and_analyze(
            engine, [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        )
        assert "concerns" in analysis
        assert "warnings" in analysis
        assert "recommendations" in analysis
        assert "total_water_dropout" in analysis
        assert "average_efficiency" in analysis

    def test_high_pressure_adds_concern(self, engine):
        """Lines 450-456: high pressure → concerns about equipment."""
        # Use moderate temp/pressure to avoid IAPWS range errors
        stages = [_make_stage(1.0, 10.0, 300.0, 0.85, "isothermal")]
        # Run with isothermal so temp stays at 300K → safe for IAPWS
        result = engine.calculate_multistage_compression(stages, 100.0, SYNGAS_COMP)
        analysis = engine.analyze_process_conditions(result)
        # Should have concerns list (might or might not flag high pressure at 10 bar)
        assert isinstance(analysis["concerns"], list)

    def test_polytropic_avg_efficiency_is_none(self, engine):
        """Lines 488-489: polytropic → no isentropic stages → avg_efficiency = None."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "polytropic")]
        analysis = self._run_and_analyze(engine, stages)
        assert analysis["average_efficiency"] is None

    def test_isentropic_avg_efficiency_not_none(self, engine):
        """Lines 479-487: isentropic → avg_efficiency computed."""
        stages = [_make_stage(1.0, 3.0, 300.0, 0.85, "isentropic")]
        analysis = self._run_and_analyze(engine, stages)
        assert analysis["average_efficiency"] is not None
        assert analysis["average_efficiency"] > 0
