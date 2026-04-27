"""Comprehensive tests for ScrubberEngine and ScrubberModels.

Tests cover the data models, engine calculation, zero-flow edge case,
column sizing, and thermal calculations.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.scrubber.engine.scrubber_engine import (
    ScrubberEngine,
)
from upstream_drift_tools.process_calculators.scrubber.models.scrubber_models import (
    ScrubberInputs,
    ScrubberResults,
)

# ─── Fixtures ─────────────────────────────────────────────────


def _typical_inputs() -> ScrubberInputs:
    return ScrubberInputs(
        gas_flow_kg_hr=5000.0,
        inlet_temp_c=200.0,
        pressure_bar=1.2,
        molecular_weight=28.0,
        target_outlet_temp_c=40.0,
        packing_name="Metal Pall Rings",
        percent_of_flood=70.0,
        height_safety_factor=1.25,
        lg_ratio=2.5,
        caustic_concentration_wt_pct=10.0,
        cooling_water_inlet_temp_c=25.0,
        kla_hr=200.0,
        acid_gas_composition_ppmv={"HCl": 500.0, "SO2": 100.0},
        acid_gas_removal_pct={"HCl": 99.0, "SO2": 95.0},
    )


# ─── ScrubberInputs Tests ───────────────────────────────────


class TestScrubberInputs:
    def test_frozen_dataclass(self) -> None:
        inputs = _typical_inputs()
        with pytest.raises(AttributeError):
            inputs.gas_flow_kg_hr = 9999.0

    def test_defaults(self) -> None:
        inputs = ScrubberInputs(
            gas_flow_kg_hr=1000.0,
            inlet_temp_c=100.0,
            pressure_bar=1.0,
            molecular_weight=28.0,
            target_outlet_temp_c=40.0,
            packing_name="Metal Pall Rings",
            percent_of_flood=70.0,
            height_safety_factor=1.25,
            lg_ratio=2.0,
            caustic_concentration_wt_pct=10.0,
            cooling_water_inlet_temp_c=25.0,
            kla_hr=200.0,
        )
        assert inputs.acid_gas_composition_ppmv == {}
        assert inputs.acid_gas_removal_pct == {}


# ─── ScrubberResults Tests ───────────────────────────────────


class TestScrubberResults:
    def test_frozen_dataclass(self) -> None:
        results = ScrubberResults(
            column_diameter_m=1.0,
            packed_height_m=5.0,
            pressure_drop_kpa=0.5,
            naoh_pure_kg_hr=10.0,
            naoh_solution_L_hr=100.0,
            total_heat_duty_kw=500.0,
            cooling_water_flow_L_min=200.0,
            gas_density_kg_m3=1.2,
            flooding_velocity_m_s=2.0,
            htu_m=0.5,
            max_ntu=4.0,
        )
        with pytest.raises(AttributeError):
            results.column_diameter_m = 999.0

    def test_defaults_empty_lists(self) -> None:
        results = ScrubberResults(
            column_diameter_m=1.0,
            packed_height_m=5.0,
            pressure_drop_kpa=0.5,
            naoh_pure_kg_hr=10.0,
            naoh_solution_L_hr=100.0,
            total_heat_duty_kw=500.0,
            cooling_water_flow_L_min=200.0,
            gas_density_kg_m3=1.2,
            flooding_velocity_m_s=2.0,
            htu_m=0.5,
            max_ntu=4.0,
        )
        assert results.acid_gas_details == []
        assert results.warnings == []


# ─── ScrubberEngine.calculate Tests ──────────────────────────


class TestScrubberEngineCalculate:
    def test_returns_results_type(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert isinstance(result, ScrubberResults)

    def test_column_diameter_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.column_diameter_m > 0.0

    def test_packed_height_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.packed_height_m > 0.0

    def test_gas_density_reasonable(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        # Gas density should be between 0.1 and 10 kg/m3
        assert 0.1 < result.gas_density_kg_m3 < 10.0

    def test_flooding_velocity_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.flooding_velocity_m_s > 0.0

    def test_heat_duty_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.total_heat_duty_kw > 0.0

    def test_cooling_water_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.cooling_water_flow_L_min > 0.0

    def test_acid_gas_details_populated(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert len(result.acid_gas_details) == 2  # HCl and SO2

    def test_acid_gas_outlet_less_than_inlet(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        for detail in result.acid_gas_details:
            assert detail["outlet_ppmv"] < detail["inlet_ppmv"]

    def test_max_ntu_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.max_ntu > 0.0

    def test_htu_positive(self) -> None:
        result = ScrubberEngine.calculate(_typical_inputs())
        assert result.htu_m > 0.0


class TestScrubberEngineZeroFlow:
    def test_zero_flow_returns_zeros(self) -> None:
        inputs = ScrubberInputs(
            gas_flow_kg_hr=0.0,
            inlet_temp_c=200.0,
            pressure_bar=1.2,
            molecular_weight=28.0,
            target_outlet_temp_c=40.0,
            packing_name="Metal Pall Rings",
            percent_of_flood=70.0,
            height_safety_factor=1.25,
            lg_ratio=2.5,
            caustic_concentration_wt_pct=10.0,
            cooling_water_inlet_temp_c=25.0,
            kla_hr=200.0,
        )
        result = ScrubberEngine.calculate(inputs)
        assert result.column_diameter_m == 0.0
        assert result.packed_height_m == 0.0

    def test_zero_flow_has_warning(self) -> None:
        inputs = ScrubberInputs(
            gas_flow_kg_hr=0.0,
            inlet_temp_c=200.0,
            pressure_bar=1.2,
            molecular_weight=28.0,
            target_outlet_temp_c=40.0,
            packing_name="Metal Pall Rings",
            percent_of_flood=70.0,
            height_safety_factor=1.25,
            lg_ratio=2.5,
            caustic_concentration_wt_pct=10.0,
            cooling_water_inlet_temp_c=25.0,
            kla_hr=200.0,
        )
        result = ScrubberEngine.calculate(inputs)
        assert len(result.warnings) > 0

    def test_negative_flow_returns_zeros(self) -> None:
        inputs = ScrubberInputs(
            gas_flow_kg_hr=-100.0,
            inlet_temp_c=200.0,
            pressure_bar=1.2,
            molecular_weight=28.0,
            target_outlet_temp_c=40.0,
            packing_name="Metal Pall Rings",
            percent_of_flood=70.0,
            height_safety_factor=1.25,
            lg_ratio=2.5,
            caustic_concentration_wt_pct=10.0,
            cooling_water_inlet_temp_c=25.0,
            kla_hr=200.0,
        )
        result = ScrubberEngine.calculate(inputs)
        assert result.column_diameter_m == 0.0


class TestScrubberEngineUnknownPacking:
    def test_unknown_packing_raises(self) -> None:
        inputs = ScrubberInputs(
            gas_flow_kg_hr=5000.0,
            inlet_temp_c=200.0,
            pressure_bar=1.2,
            molecular_weight=28.0,
            target_outlet_temp_c=40.0,
            packing_name="Nonexistent Packing",
            percent_of_flood=70.0,
            height_safety_factor=1.25,
            lg_ratio=2.5,
            caustic_concentration_wt_pct=10.0,
            cooling_water_inlet_temp_c=25.0,
            kla_hr=200.0,
        )
        with pytest.raises(ValueError, match="Unknown packing"):
            ScrubberEngine.calculate(inputs)
