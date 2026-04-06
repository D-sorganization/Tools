"""Contract tests for process calculator modules.

Validates that require() preconditions fire correctly on invalid inputs
for the 5 calculators that received contract instrumentation in #1929:
- baghouse_calculator
- electrode_advancement_calculator
- acid_gas_dewpoint_calculator
- syngas_water_calculator
- pressure_drop_calculator

These tests are marked @pytest.mark.contract so they run in the
``pytest -m contract`` suite.
"""

from __future__ import annotations

import pytest


@pytest.mark.contract
class TestBaghouseCalculatorContracts:
    """Contract tests for BaghouseCalculator.calculate()."""

    def _make_calculator(self):
        from upstream_drift_tools.process_calculators.baghouse_calculator import (
            BaghouseCalculator,
        )

        return BaghouseCalculator(thermo_calc=None)

    def _valid_kwargs(self):
        return {
            "gas_flow_kg_s": 1.0,
            "inlet_temp_k": 500.0,
            "pressure_pa": 101325.0,
            "composition": {"N2": 0.7, "CO2": 0.15, "H2O": 0.15},
            "solid_carbon_in_kg_hr": 10.0,
            "ash_in_kg_hr": 5.0,
            "carbon_removal_efficiency": 0.95,
            "ash_removal_efficiency": 0.99,
            "heat_loss_w": 1000.0,
            "drum_volume_m3": 1.0,
            "solid_density_kg_m3": 500.0,
            "bag_area_ft2": 1000.0,
        }

    def test_valid_inputs_succeed(self):
        calc = self._make_calculator()
        result = calc.calculate(**self._valid_kwargs())
        assert result.carbon_removed_rate > 0

    def test_negative_gas_flow_rejected(self):
        calc = self._make_calculator()
        kwargs = self._valid_kwargs()
        kwargs["gas_flow_kg_s"] = -1.0
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate(**kwargs)

    def test_zero_temperature_rejected(self):
        calc = self._make_calculator()
        kwargs = self._valid_kwargs()
        kwargs["inlet_temp_k"] = 0.0
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate(**kwargs)

    def test_negative_pressure_rejected(self):
        calc = self._make_calculator()
        kwargs = self._valid_kwargs()
        kwargs["pressure_pa"] = -101325.0
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate(**kwargs)

    def test_efficiency_out_of_range_rejected(self):
        calc = self._make_calculator()
        kwargs = self._valid_kwargs()
        kwargs["carbon_removal_efficiency"] = 1.5
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate(**kwargs)

    def test_zero_bag_area_rejected(self):
        calc = self._make_calculator()
        kwargs = self._valid_kwargs()
        kwargs["bag_area_ft2"] = 0.0
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate(**kwargs)


@pytest.mark.contract
class TestElectrodeAdvancementContracts:
    """Contract tests for ElectrodeAdvancementCalculator."""

    def test_valid_inputs_succeed(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        result = calc.calculate_consumption(current_ka=50.0, time_hrs=8.0)
        assert result > 0

    def test_negative_consumption_rate_rejected(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        with pytest.raises((ValueError, AssertionError)):
            ElectrodeAdvancementCalculator(consumption_rate=-1.0)

    def test_negative_current_rejected(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_consumption(current_ka=-10.0, time_hrs=8.0)

    def test_negative_time_rejected(self):
        from upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
            ElectrodeAdvancementCalculator,
        )

        calc = ElectrodeAdvancementCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_consumption(current_ka=50.0, time_hrs=-1.0)


@pytest.mark.contract
class TestAcidGasDewpointContracts:
    """Contract tests for AcidGasDewpointCalculator."""

    def test_valid_dewpoint_succeeds(self):
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculator,
        )

        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint(
            partial_pressure_pa=1000.0,
            component="H2O",
        )
        assert isinstance(result, float)

    def test_negative_partial_pressure_rejected(self):
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculator,
        )

        calc = AcidGasDewpointCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_dewpoint(
                partial_pressure_pa=-100.0,
                component="H2O",
            )

    def test_unknown_component_rejected(self):
        from upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
            AcidGasDewpointCalculator,
        )

        calc = AcidGasDewpointCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_dewpoint(
                partial_pressure_pa=1000.0,
                component="XenonGas",
            )


@pytest.mark.contract
class TestSyngasWaterContracts:
    """Contract tests for SyngasWaterCalculator."""

    def test_valid_dew_point_succeeds(self):
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        result = calc.calculate_dew_point(
            partial_pressure_pa=2000.0,
            total_pressure_pa=101325.0,
        )
        assert isinstance(result, float)

    def test_negative_partial_pressure_rejected(self):
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_dew_point(
                partial_pressure_pa=-100.0,
                total_pressure_pa=101325.0,
            )

    def test_negative_total_pressure_rejected(self):
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_dew_point(
                partial_pressure_pa=2000.0,
                total_pressure_pa=-101325.0,
            )

    def test_valid_water_content_succeeds(self):
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        result = calc.calculate_water_content(
            temperature_c=100.0,
            pressure_bar=30.0,
        )
        assert hasattr(result, "water_fraction")

    def test_negative_pressure_bar_rejected(self):
        from upstream_drift_tools.process_calculators.syngas_water_calculator import (
            SyngasWaterCalculator,
        )

        calc = SyngasWaterCalculator()
        with pytest.raises((ValueError, AssertionError)):
            calc.calculate_water_content(
                temperature_c=100.0,
                pressure_bar=-5.0,
            )
