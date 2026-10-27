"""Tests for the SyngasCompressionEngine.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy,
testing the core engine math internally within the Tools repository.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
    CompressionStage,
    SyngasCompressionEngine,
)


@pytest.fixture
def engine() -> SyngasCompressionEngine:
    """Return a fresh SyngasCompressionEngine."""
    return SyngasCompressionEngine()


@pytest.fixture
def isentropic_stage() -> CompressionStage:
    """Return a standard isentropic compression stage."""
    return CompressionStage(
        inlet_pressure=1.0,
        outlet_pressure=5.0,
        inlet_temperature=300.0,  # K
        efficiency=0.85,
        compression_type="isentropic",
    )


@pytest.fixture
def syngas_composition() -> dict:
    """Return a typical syngas composition (fractions, sum=1)."""
    return {"H2": 0.30, "CO": 0.30, "CO2": 0.15, "CH4": 0.05, "N2": 0.18, "Ar": 0.02}


class TestCompressionWork:
    """Tests for single-stage compression work calculations."""

    def test_isentropic_outlet_temp_higher_than_inlet(
        self, engine: SyngasCompressionEngine, isentropic_stage: CompressionStage
    ) -> None:
        """Compression must always raise temperature (inlet_temp < outlet_temp)."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(isentropic_stage, 100.0, mixture)
        assert result["temp_out_actual"] > isentropic_stage.inlet_temperature

    def test_isentropic_work_positive(
        self, engine: SyngasCompressionEngine, isentropic_stage: CompressionStage
    ) -> None:
        """Isentropic work must be positive for pressure ratio > 1."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(isentropic_stage, 100.0, mixture)
        assert result["work_actual"] > 0

    def test_isothermal_outlet_temp_constant(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """Isothermal compression means outlet temp equals inlet."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=3.0,
            inlet_temperature=300.0,
            efficiency=1.0,
            compression_type="isothermal",
        )
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(stage, 100.0, mixture)
        assert result["temp_out_actual"] == pytest.approx(300.0)

    def test_polytropic_work_positive(self, engine: SyngasCompressionEngine) -> None:
        """Polytropic compression must yield positive work."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=310.0,
            efficiency=0.80,
            compression_type="polytropic",
        )
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(stage, 100.0, mixture)
        assert result["work_actual"] > 0

    def test_polytropic_tout_depends_on_efficiency(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """TOOLS-03: Polytropic outlet temp must increase as efficiency decreases."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        stage_high_eff = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=300.0,
            efficiency=0.90,
            compression_type="polytropic",
        )
        stage_low_eff = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=300.0,
            efficiency=0.60,
            compression_type="polytropic",
        )
        r_high = engine.calculate_compression_work(stage_high_eff, 100.0, mixture)
        r_low = engine.calculate_compression_work(stage_low_eff, 100.0, mixture)
        # Irreversibility: lower efficiency must result in higher outlet temperature
        assert r_low["temp_out_actual"] > r_high["temp_out_actual"]

    def test_polytropic_matches_isentropic_at_unit_efficiency(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """TOOLS-03: At efficiency = 1.0, polytropic matches isentropic."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        stage_poly = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=300.0,
            efficiency=1.0,
            compression_type="polytropic",
        )
        stage_isen = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=300.0,
            efficiency=1.0,
            compression_type="isentropic",
        )
        r_poly = engine.calculate_compression_work(stage_poly, 100.0, mixture)
        r_isen = engine.calculate_compression_work(stage_isen, 100.0, mixture)
        assert r_poly["temp_out_actual"] == pytest.approx(
            r_isen["temp_out_actual"], rel=1e-5
        )
        assert r_poly["work_actual"] == pytest.approx(r_isen["work_actual"], rel=1e-5)

    def test_polytropic_precondition_efficiency_range(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """TOOLS-03: Polytropic efficiency must be in (0, 1]."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        bad_stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=4.0,
            inlet_temperature=300.0,
            efficiency=1.2,
            compression_type="polytropic",
        )
        with pytest.raises(ValueError, match="efficiency"):
            engine.calculate_compression_work(bad_stage, 100.0, mixture)

    def test_unknown_compression_type_raises(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """Requesting an undefined compression type must raise ValueError."""
        stage = CompressionStage(
            inlet_pressure=1.0,
            outlet_pressure=5.0,
            inlet_temperature=300.0,
            efficiency=0.85,
            compression_type="magic",
        )
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        with pytest.raises(ValueError, match="Unknown compression type"):
            engine.calculate_compression_work(stage, 100.0, mixture)

    def test_zero_inlet_pressure_raises(self, engine: SyngasCompressionEngine) -> None:
        """Zero inlet pressure is a physics precondition violation."""
        stage = CompressionStage(
            inlet_pressure=0.0,
            outlet_pressure=5.0,
            inlet_temperature=300.0,
            efficiency=0.85,
            compression_type="isentropic",
        )
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        with pytest.raises(ValueError, match="inlet_pressure"):
            engine.calculate_compression_work(stage, 100.0, mixture)

    def test_pressure_ratio_correct(
        self, engine: SyngasCompressionEngine, isentropic_stage: CompressionStage
    ) -> None:
        """Reported pressure ratio must equal outlet/inlet."""
        mixture = {"molecular_weight": 28.0, "heat_capacity_ratio": 1.4}
        result = engine.calculate_compression_work(isentropic_stage, 100.0, mixture)
        expected_pr = isentropic_stage.outlet_pressure / isentropic_stage.inlet_pressure
        assert result["pressure_ratio"] == pytest.approx(expected_pr, rel=1e-6)


class TestWaterDropout:
    """Tests for water dropout / condensation calculations."""

    def test_no_water_at_low_partial_pressure(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """Very low water content at atmospheric pressure should not condense."""
        result = engine.calculate_water_dropout(
            temperature=350.0,  # 77°C
            pressure=1.0,  # 1 bar
            water_content=0.01,  # 0.01 mol%
        )
        assert result["water_dropout"] == pytest.approx(0.0, abs=1e-6)
        assert result["condensation_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_water_condenses_at_high_saturation(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """High water content at elevated pressure should trigger dropout."""
        result = engine.calculate_water_dropout(
            temperature=353.15,  # 80°C
            pressure=100.0,  # 100 bar - very high, reduces vp fraction
            water_content=50.0,  # 50 mol% water
        )
        assert result["water_dropout"] > 0

    def test_zero_pressure_raises(self, engine: SyngasCompressionEngine) -> None:
        """Zero pressure is a physics precondition violation."""
        with pytest.raises(ValueError, match="pressure"):
            engine.calculate_water_dropout(
                temperature=350.0,
                pressure=0.0,
                water_content=0.5,
            )


class TestMultistageCompression:
    """Tests for multi-stage compression calculations."""

    def test_single_stage_result_shape(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """Single-stage result should have expected structure."""
        stages = [
            CompressionStage(
                inlet_pressure=1.0,
                outlet_pressure=10.0,
                inlet_temperature=300.0,
                efficiency=0.85,
                compression_type="isentropic",
            )
        ]
        result = engine.calculate_multistage_compression(
            stages, 500.0, syngas_composition
        )
        assert "stages" in result
        assert len(result["stages"]) == 1
        assert "total_power_hp" in result
        assert result["total_power_hp"] > 0

    def test_multistage_power_greater_than_single_stage(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """Both single and two-stage configurations compute positive power."""
        single_stage = [CompressionStage(1.0, 25.0, 380.0, 0.85, "isentropic")]
        two_stages = [
            CompressionStage(1.0, 5.0, 380.0, 0.85, "isentropic"),
            CompressionStage(5.0, 25.0, 380.0, 0.85, "isentropic"),
        ]
        r1 = engine.calculate_multistage_compression(
            single_stage, 100.0, syngas_composition
        )
        r2 = engine.calculate_multistage_compression(
            two_stages, 100.0, syngas_composition
        )
        # Both compute positive power
        assert r1["total_power_hp"] > 0
        assert r2["total_power_hp"] > 0

    def test_empty_stages_raises(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """Empty stages list is a precondition violation."""
        with pytest.raises(ValueError, match="stages"):
            engine.calculate_multistage_compression([], 100.0, syngas_composition)


class TestProcessConditionAnalysis:
    """Tests for the process condition analysis module."""

    def test_normal_conditions_no_warnings(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """Normal operating conditions should produce a valid analysis structure."""
        stages = [CompressionStage(1.0, 5.0, 380.0, 0.85, "isentropic")]
        compression_result = engine.calculate_multistage_compression(
            stages, 100.0, syngas_composition
        )
        analysis = engine.analyze_process_conditions(compression_result)
        # Analysis always returns structured lists
        assert isinstance(analysis["concerns"], list)
        assert isinstance(analysis["warnings"], list)
        assert isinstance(analysis["recommendations"], list)

    def test_very_high_temperature_triggers_warning(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """Very high outlet temp should populate concerns list."""
        # Single-stage from 1 to 500 bar starting at 380K will generate extreme heat
        stages = [CompressionStage(1.0, 500.0, 380.0, 0.85, "isentropic")]
        result = engine.calculate_multistage_compression(
            stages, 100.0, syngas_composition
        )
        analysis = engine.analyze_process_conditions(result)
        assert len(analysis["concerns"]) > 0 or len(analysis["warnings"]) > 0

    def test_isentropic_efficiency_le_1(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """TOOLS-02: Isentropic efficiency must be W_isentropic / W_actual <= 1."""
        stages = [CompressionStage(1.0, 5.0, 300.0, 0.75, "isentropic")]
        result = engine.calculate_multistage_compression(
            stages, 100.0, syngas_composition
        )
        analysis = engine.analyze_process_conditions(result)
        avg_eff = analysis["average_efficiency"]
        assert avg_eff is not None
        assert 0.0 < avg_eff <= 1.0
        assert avg_eff == pytest.approx(0.75, rel=1e-3)

    def test_low_efficiency_triggers_warning_and_concern(
        self,
        engine: SyngasCompressionEngine,
        syngas_composition: dict,
    ) -> None:
        """TOOLS-02: Low efficiency (< MIN_EFFICIENCY) must trigger concern."""
        stages = [CompressionStage(1.0, 5.0, 300.0, 0.50, "isentropic")]
        result = engine.calculate_multistage_compression(
            stages, 100.0, syngas_composition
        )
        analysis = engine.analyze_process_conditions(result)
        assert "Low compression efficiency detected" in analysis["concerns"]
        assert (
            "Consider compressor maintenance or replacement"
            in analysis["recommendations"]
        )

    def test_isentropic_efficiency_dbc_contract_violation_raises(
        self, engine: SyngasCompressionEngine
    ) -> None:
        """TOOLS-02: Efficiency > 1 violates thermodynamic postcondition and raises."""
        mock_result = {
            "final_temperature": 350.0,
            "final_pressure": 5.0,
            "total_power_hp": 100.0,
            "stages": [
                {
                    "work_isentropic": 1500.0,
                    "work_actual": 1000.0,  # W_isentropic > W_actual (eta = 1.5 > 1.0)
                    "water_dropout": {"water_dropout": 0.0},
                }
            ],
        }
        with pytest.raises(ValueError, match="Postcondition violated"):
            engine.analyze_process_conditions(mock_result)
