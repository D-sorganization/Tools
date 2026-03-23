"""Tests for the Scrubber Calculator functions.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy, testing the math internally within the Tools repository.
"""

from __future__ import annotations

import pytest
from contracts import (
    ContractLevel,
    PreconditionError,
    get_contract_level,
    set_contract_level,
)
from upstream_drift_tools.process_calculators.scrubber_calculator import (
    PACKING_DATABASE,
    WATER_DENSITY,
    WATER_VISCOSITY,
    calculate_caustic_requirement,
    calculate_column_diameter,
    calculate_flooding_velocity,
    calculate_gas_density,
    calculate_gas_viscosity,
    calculate_heat_transfer_duty,
)


class TestScrubberCalculator:
    """Tests for core math in the Scrubber logic functions."""

    def test_gas_density(self) -> None:
        """Verify the ideal gas law derivation for density."""
        # Nitrogen at standardish conditions — param name is 'molecular_weight'
        density = calculate_gas_density(
            temperature_k=298.15, pressure_pa=101325.0, molecular_weight=28.0134
        )
        assert density > 1.0 and density < 2.0  # roughly 1.14 kg/m3

    def test_gas_viscosity(self) -> None:
        """Verify empirical derivation for viscosity."""
        # param name is 'molecular_weight', not 'mol_weight'
        visc = calculate_gas_viscosity(temperature_k=298.15, molecular_weight=28.0134)
        assert visc > 0.00001 and visc < 0.00002

    def test_flooding_velocity_valid(self) -> None:
        """Calculate flooding threshold utilizing Eckert's Generalized Flooding correlation approximations."""
        # Key is 'Metal Pall Rings', not 'pall_ring_1_inch_ceramic'
        packing = PACKING_DATABASE["Metal Pall Rings"]

        velocity = calculate_flooding_velocity(
            liquid_mass_flux=5.0,
            gas_density=1.2,
            liquid_density=WATER_DENSITY,
            packing=packing,
            liquid_viscosity=WATER_VISCOSITY,
        )

        assert velocity > 0.0
        assert velocity < 10.0  # Realistic bounded value

    def test_column_diameter(self) -> None:
        """Verify geometric deductions from flooding margins."""
        res = calculate_column_diameter(
            gas_flow_kg_hr=5000.0,
            gas_density=1.2,
            flooding_velocity=2.5,
            percent_of_flood=70.0,
        )

        assert res["design_velocity_m_s"] == 2.5 * 0.7
        assert res["cross_section_m2"] > 0.0
        assert res["diameter_m"] > 0.0
        assert res["diameter_ft"] > 0.0

    def test_caustic_requirement(self) -> None:
        """Test simple stoichiometric mass ratio scaling for NaOH.

        acid_gas_removed must be a dict[str, float] not a plain float.
        """
        req = calculate_caustic_requirement(
            acid_gas_removed={"HCl": 10.0},
            caustic_concentration=20.0,
        )

        # 10 kg/hr of HCl requires NaOH. At 20% concentration, solution flow > pure NaOH.
        assert req["naoh_pure_kg_hr"] > 0.0
        assert req["naoh_solution_kg_hr"] > req["naoh_pure_kg_hr"]
        assert req["naoh_solution_L_hr"] > 0.0

    def test_caustic_with_multiple_gases(self) -> None:
        """Multiple acid gases are summed correctly."""
        req = calculate_caustic_requirement(
            acid_gas_removed={"HCl": 5.0, "SO2": 5.0},
            caustic_concentration=25.0,
        )
        single = calculate_caustic_requirement(
            acid_gas_removed={"HCl": 5.0},
            caustic_concentration=25.0,
        )
        # More gas → more NaOH required
        assert req["naoh_pure_kg_hr"] > single["naoh_pure_kg_hr"]

    def test_column_diameter_zero_flooding_velocity_returns_zero(self) -> None:
        """Zero flooding velocity returns zero-sized column."""
        res = calculate_column_diameter(
            gas_flow_kg_hr=5000.0,
            gas_density=1.2,
            flooding_velocity=0.0,
            percent_of_flood=70.0,
        )
        assert res["diameter_m"] == 0.0

    def test_packing_database_has_expected_keys(self) -> None:
        """Packing database contains standard packing types."""
        assert "Metal Pall Rings" in PACKING_DATABASE
        assert "Ceramic Raschig Rings" in PACKING_DATABASE
        assert "Structured Packing" in PACKING_DATABASE


class TestScrubberCalculatorContracts:
    """Design-by-Contract enforcement tests for scrubber calculator public functions.

    All tests run with ContractLevel.ENFORCE to ensure preconditions and
    postconditions raise PreconditionError on invalid physical inputs.
    """

    @pytest.fixture(autouse=True)
    def _enforce_contracts(self) -> None:  # type: ignore[misc]
        """Ensure DbC ENFORCE mode is active for every test in this class."""
        original = get_contract_level()
        set_contract_level(ContractLevel.ENFORCE)
        yield
        set_contract_level(original)

    # ── calculate_gas_density ──────────────────────────────────────────────

    def test_gas_density_zero_temperature_raises(self) -> None:
        """temperature_k must be > 0 K (absolute scale)."""
        with pytest.raises(PreconditionError):
            calculate_gas_density(
                temperature_k=0.0, pressure_pa=101325.0, molecular_weight=28.0
            )

    def test_gas_density_negative_temperature_raises(self) -> None:
        """Negative temperature_k must raise PreconditionError."""
        with pytest.raises(PreconditionError):
            calculate_gas_density(
                temperature_k=-10.0, pressure_pa=101325.0, molecular_weight=28.0
            )

    def test_gas_density_zero_pressure_raises(self) -> None:
        """pressure_pa must be positive."""
        with pytest.raises(PreconditionError):
            calculate_gas_density(
                temperature_k=300.0, pressure_pa=0.0, molecular_weight=28.0
            )

    def test_gas_density_negative_pressure_raises(self) -> None:
        """Negative pressure_pa must raise PreconditionError."""
        with pytest.raises(PreconditionError):
            calculate_gas_density(
                temperature_k=300.0, pressure_pa=-1.0, molecular_weight=28.0
            )

    def test_gas_density_zero_molecular_weight_raises(self) -> None:
        """molecular_weight must be positive."""
        with pytest.raises(PreconditionError):
            calculate_gas_density(
                temperature_k=300.0, pressure_pa=101325.0, molecular_weight=0.0
            )

    def test_gas_density_valid_returns_positive(self) -> None:
        """Valid inputs produce a positive density (postcondition)."""
        density = calculate_gas_density(
            temperature_k=300.0, pressure_pa=101325.0, molecular_weight=28.0
        )
        assert density > 0

    # ── calculate_gas_viscosity ─────────────────────────────────────────────

    def test_gas_viscosity_zero_temperature_raises(self) -> None:
        """temperature_k must be > 0."""
        with pytest.raises(PreconditionError):
            calculate_gas_viscosity(temperature_k=0.0, molecular_weight=28.0)

    def test_gas_viscosity_zero_molecular_weight_raises(self) -> None:
        """molecular_weight must be positive."""
        with pytest.raises(PreconditionError):
            calculate_gas_viscosity(temperature_k=300.0, molecular_weight=0.0)

    # ── calculate_flooding_velocity ─────────────────────────────────────────

    def test_flooding_velocity_zero_gas_density_raises(self) -> None:
        """gas_density must be positive."""
        packing = PACKING_DATABASE["Metal Pall Rings"]
        with pytest.raises(PreconditionError):
            calculate_flooding_velocity(
                liquid_mass_flux=5.0,
                gas_density=0.0,
                liquid_density=WATER_DENSITY,
                packing=packing,
                liquid_viscosity=WATER_VISCOSITY,
            )

    def test_flooding_velocity_negative_liquid_density_raises(self) -> None:
        """liquid_density must be positive."""
        packing = PACKING_DATABASE["Metal Pall Rings"]
        with pytest.raises(PreconditionError):
            calculate_flooding_velocity(
                liquid_mass_flux=5.0,
                gas_density=1.2,
                liquid_density=-100.0,
                packing=packing,
                liquid_viscosity=WATER_VISCOSITY,
            )

    def test_flooding_velocity_negative_liquid_mass_flux_raises(self) -> None:
        """liquid_mass_flux must be non-negative."""
        packing = PACKING_DATABASE["Metal Pall Rings"]
        with pytest.raises(PreconditionError):
            calculate_flooding_velocity(
                liquid_mass_flux=-1.0,
                gas_density=1.2,
                liquid_density=WATER_DENSITY,
                packing=packing,
                liquid_viscosity=WATER_VISCOSITY,
            )

    # ── calculate_column_diameter ───────────────────────────────────────────

    def test_column_diameter_zero_gas_flow_raises(self) -> None:
        """gas_flow_kg_hr must be positive."""
        with pytest.raises(PreconditionError):
            calculate_column_diameter(
                gas_flow_kg_hr=0.0,
                gas_density=1.2,
                flooding_velocity=2.5,
                percent_of_flood=70.0,
            )

    def test_column_diameter_invalid_percent_of_flood_raises(self) -> None:
        """percent_of_flood must be in (0, 100]."""
        with pytest.raises(PreconditionError):
            calculate_column_diameter(
                gas_flow_kg_hr=5000.0,
                gas_density=1.2,
                flooding_velocity=2.5,
                percent_of_flood=0.0,
            )

    def test_column_diameter_percent_over_100_raises(self) -> None:
        """percent_of_flood > 100 must raise PreconditionError."""
        with pytest.raises(PreconditionError):
            calculate_column_diameter(
                gas_flow_kg_hr=5000.0,
                gas_density=1.2,
                flooding_velocity=2.5,
                percent_of_flood=110.0,
            )

    # ── calculate_heat_transfer_duty ────────────────────────────────────────

    def test_heat_transfer_duty_zero_gas_flow_raises(self) -> None:
        """gas_flow_kg_hr must be positive."""
        with pytest.raises(PreconditionError):
            calculate_heat_transfer_duty(
                gas_flow_kg_hr=0.0,
                inlet_temp_c=150.0,
                outlet_temp_c=40.0,
                water_condensed_kg_hr=0.0,
            )

    def test_heat_transfer_duty_negative_water_condensed_raises(self) -> None:
        """water_condensed_kg_hr must be non-negative."""
        with pytest.raises(PreconditionError):
            calculate_heat_transfer_duty(
                gas_flow_kg_hr=1000.0,
                inlet_temp_c=150.0,
                outlet_temp_c=40.0,
                water_condensed_kg_hr=-5.0,
            )

    # ── contracts OFF: no enforcement ───────────────────────────────────────

    def test_contracts_off_skips_precondition_checks(self) -> None:
        """When DBC_LEVEL=OFF, invalid inputs do not raise PreconditionError.

        Uses zero gas_flow_kg_hr (physically invalid, fails precondition in ENFORCE
        mode) but does not cause a math exception when bypassed, allowing verification
        that the contract check was truly skipped.
        """
        set_contract_level(ContractLevel.OFF)
        try:
            result = calculate_column_diameter(
                gas_flow_kg_hr=0.0,
                gas_density=1.2,
                flooding_velocity=2.5,
                percent_of_flood=70.0,
            )
            assert result["diameter_m"] == 0.0
        except PreconditionError:
            pytest.fail("PreconditionError raised with contracts OFF")
