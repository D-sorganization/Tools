"""Tests for the Baghouse Calculator.

This test file adheres to the Fleet-Wide Shared Component Testing Strategy, testing the math internally within the Tools repository.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.baghouse_calculator import (
    BaghouseCalculator,
    BaghouseResult,
)


@pytest.fixture
def calculator() -> BaghouseCalculator:
    """Return a fresh BaghouseCalculator."""
    return BaghouseCalculator()


class TestBaghouseCalculatorInit:
    """Tests for initialization logic."""

    def test_init_without_thermo(self, calculator: BaghouseCalculator) -> None:
        """Should initialize even if thermo isn't explicitly passed."""
        assert calculator is not None
        # It's okay if thermo_calc is None or set depending on environment availability


class TestBaghouseCalculations:
    """Tests for core math in the Baghouse parameter calculation."""

    def test_ideal_gas_cp_estimation(self, calculator: BaghouseCalculator) -> None:
        """Cp estimation using ideal approximations."""
        # Mix of 50% H2, 50% CO
        mix = {"H2": 0.5, "CO": 0.5}
        cp = calculator._estimate_cp_ideal(mix)
        assert cp > 0.0

    def test_volume_flow_estimation(self, calculator: BaghouseCalculator) -> None:
        """Test ACFM and SCFM estimations."""
        mix = {"N2": 1.0}
        # 1 kg/s of N2 at roughly STP
        acfm, scfm = calculator._estimate_volume_flow(
            mass_flow_kg_s=1.0,
            temperature_k=273.15,
            pressure_pa=101325.0,
            composition=mix,
        )
        assert acfm > 0.0
        assert scfm > 0.0
        assert acfm == pytest.approx(scfm, rel=0.01)

    def test_calculate_drum_sizing(self) -> None:
        """Test drum fill time derivations."""
        (
            c_rm,
            a_rm,
            tot,
            hrs,
            days,
            c_fill,
            a_fill,
        ) = BaghouseCalculator._calculate_drum_sizing(
            solid_carbon_in_kg_hr=10.0,
            ash_in_kg_hr=5.0,
            carbon_removal_efficiency=0.9,
            ash_removal_efficiency=0.8,
            drum_volume_m3=1.0,
            solid_density_kg_m3=1000.0,  # 1000 kg capacity
        )

        assert c_rm == 9.0
        assert a_rm == 4.0
        assert tot == 13.0
        # 1000 kg / 13 kg/hr
        assert hrs == pytest.approx(1000.0 / 13.0)
        assert days == pytest.approx((1000.0 / 13.0) / 24.0)

    def test_calculate_main_method(self, calculator: BaghouseCalculator) -> None:
        """Integration test of the calculate method."""
        mix = {"N2": 0.8, "O2": 0.2}
        result = calculator.calculate(
            gas_flow_kg_s=2.0,
            inlet_temp_k=400.0,
            pressure_pa=100000.0,
            composition=mix,
            solid_carbon_in_kg_hr=50.0,
            ash_in_kg_hr=20.0,
            carbon_removal_efficiency=0.99,
            ash_removal_efficiency=0.95,
            heat_loss_w=5000.0,
            drum_volume_m3=2.5,
            solid_density_kg_m3=800.0,
            bag_area_ft2=1000.0,
        )

        assert isinstance(result, BaghouseResult)
        assert result.carbon_removed_rate == 49.5
        assert result.ash_removed_rate == 19.0
        assert result.total_solids_removed_rate == 68.5
        assert result.removal_efficiency["carbon"] == 99.0
        assert result.removal_efficiency["ash"] == 95.0
        assert result.air_to_cloth_ratio > 0.0
        assert result.clean_gas_flow_rate == 2.0 * 3600.0  # kg/hr

    def test_calculate_preconditions(self, calculator: BaghouseCalculator) -> None:
        """Test DbC assertions map to valid inputs."""
        mix = {"N2": 1.0}

        with pytest.raises(AssertionError, match="(?i)must be positive"):
            calculator.calculate(
                gas_flow_kg_s=-1.0,
                inlet_temp_k=300.0,
                pressure_pa=100000.0,
                composition=mix,
                solid_carbon_in_kg_hr=0.0,
                ash_in_kg_hr=0.0,
                carbon_removal_efficiency=1.0,
                ash_removal_efficiency=1.0,
                heat_loss_w=0.0,
                drum_volume_m3=1.0,
                solid_density_kg_m3=1.0,
                bag_area_ft2=1.0,
            )
