"""Tests for the Scrubber Calculator functions.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy, testing the math internally within the Tools repository.
"""

from __future__ import annotations

import pytest

from upstream_drift_tools.process_calculators.scrubber_calculator import (
    calculate_gas_density,
    calculate_gas_viscosity,
    calculate_flooding_velocity,
    calculate_column_diameter,
    calculate_caustic_requirement,
    PACKING_DATABASE,
    WATER_DENSITY,
    WATER_VISCOSITY
)


class TestScrubberCalculator:
    """Tests for core math in the Scrubber logic functions."""

    def test_gas_density(self) -> None:
        """Verify the ideal gas law derivation for density."""
        # Nitrogen at standardish conditions
        density = calculate_gas_density(
            temperature_k=298.15,
            pressure_pa=101325.0,
            mol_weight=28.0134
        )
        assert density > 1.0 and density < 2.0  # roughly 1.14 kg/m3

    def test_gas_viscosity(self) -> None:
        """Verify empirical derivation for viscosity."""
        visc = calculate_gas_viscosity(
            temperature_k=298.15,
            mol_weight=28.0134
        )
        assert visc > 0.00001 and visc < 0.00002

    def test_flooding_velocity_valid(self) -> None:
        """Calculate flooding threshold utilizing Eckert's Generalized Flooding correlation approximations."""
        packing = PACKING_DATABASE["pall_ring_1_inch_ceramic"]
        
        velocity = calculate_flooding_velocity(
            liquid_mass_flux=5.0,
            gas_density=1.2,
            liquid_density=WATER_DENSITY,
            packing=packing,
            liquid_viscosity=WATER_VISCOSITY
        )
        
        assert velocity > 0.0
        assert velocity < 10.0 # Realistic bounded value

    def test_column_diameter(self) -> None:
        """Verify geometric deductions from flooding margins."""
        res = calculate_column_diameter(
            gas_flow_kg_hr=5000.0,
            gas_density=1.2,
            flooding_velocity=2.5,
            percent_of_flood=70.0
        )
        
        assert res["design_velocity_m_s"] == 2.5 * 0.7
        assert res["cross_section_m2"] > 0.0
        assert res["diameter_m"] > 0.0
        assert res["diameter_ft"] > 0.0

    def test_caustic_requirement(self) -> None:
        """Test simple stochiometric mass ratio scaling for NaOH."""
        req = calculate_caustic_requirement(
            acid_gas_removed=10.0,
            caustic_concentration=20.0
        )
        
        # 10 kg/hr of acid, need NaOH to offset. At 20%, mass should be huge.
        assert req["caustic_mass_kg_hr"] > 10.0
        assert req["caustic_flow_l_hr"] > 0.0

    def test_preconditions(self) -> None:
        """DbC edge cases ensuring invalid physical numbers raise errors."""
        with pytest.raises(AssertionError, match="Gas flow must be positive"):
            calculate_column_diameter(
                gas_flow_kg_hr=-10.0,
                gas_density=1.2,
                flooding_velocity=2.5,
                percent_of_flood=70.0
            )
            
        with pytest.raises(AssertionError, match="Temperature must be positive"):
            calculate_gas_density(
                temperature_k=-5.0,
                pressure_pa=101325.0,
                mol_weight=28.0
            ) 
