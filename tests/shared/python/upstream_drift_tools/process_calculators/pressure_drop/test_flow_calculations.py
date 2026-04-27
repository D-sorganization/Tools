"""Tests for the _flow_calculations sub-module.

Covers flow property calculation, pressure drop components,
compressibility correction, and erosional velocity.
"""

from __future__ import annotations

import pytest
import upstream_drift_tools.process_calculators.pressure_drop_calculator.engine._flow_calculations as _fc  # noqa: E501

calculate_compressible_flow_correction = _fc.calculate_compressible_flow_correction
calculate_elevation_pressure_drop = _fc.calculate_elevation_pressure_drop
calculate_erosional_velocity = _fc.calculate_erosional_velocity
calculate_expansion_factor = _fc.calculate_expansion_factor
calculate_frictional_pressure_drop = _fc.calculate_frictional_pressure_drop
classify_flow_regime = _fc.classify_flow_regime

# ---------------------------------------------------------------------------
# classify_flow_regime
# ---------------------------------------------------------------------------


class TestClassifyFlowRegime:
    """Unit tests for classify_flow_regime."""

    @pytest.mark.unit
    def test_laminar(self):
        assert classify_flow_regime(1000) == "laminar"

    @pytest.mark.unit
    def test_transitional(self):
        assert classify_flow_regime(3000) == "transitional"

    @pytest.mark.unit
    def test_turbulent(self):
        assert classify_flow_regime(10_000) == "turbulent"

    @pytest.mark.unit
    def test_boundary_laminar_upper(self):
        # Re = 2299 → laminar; 2301 → transitional
        assert classify_flow_regime(2299) == "laminar"
        assert classify_flow_regime(2301) == "transitional"


# ---------------------------------------------------------------------------
# calculate_frictional_pressure_drop
# ---------------------------------------------------------------------------


class TestCalculateFrictionalPressureDrop:
    """Unit tests for calculate_frictional_pressure_drop (Darcy-Weisbach)."""

    @pytest.mark.unit
    def test_basic_calculation(self):
        """Simple sanity: result should be positive."""
        dp = calculate_frictional_pressure_drop(
            friction_factor=0.02,
            length=100.0,
            diameter=0.1,
            density=1.2,
            velocity=10.0,
        )
        assert dp > 0

    @pytest.mark.unit
    def test_proportional_to_length(self):
        """Doubling length should double dP."""
        kwargs = dict(friction_factor=0.02, diameter=0.1, density=1.2, velocity=10.0)
        dp1 = calculate_frictional_pressure_drop(length=50.0, **kwargs)
        dp2 = calculate_frictional_pressure_drop(length=100.0, **kwargs)
        assert dp2 == pytest.approx(dp1 * 2, rel=1e-9)

    @pytest.mark.unit
    def test_negative_friction_factor_raises(self):
        with pytest.raises(ValueError):
            calculate_frictional_pressure_drop(
                friction_factor=-0.02,
                length=100.0,
                diameter=0.1,
                density=1.2,
                velocity=10.0,
            )

    @pytest.mark.unit
    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            calculate_frictional_pressure_drop(
                friction_factor=0.02,
                length=100.0,
                diameter=0.0,
                density=1.2,
                velocity=10.0,
            )

    @pytest.mark.unit
    def test_darcy_weisbach_formula(self):
        """Verify numeric output against the analytical formula."""
        f, L, D, rho, v = 0.02, 100.0, 0.1, 1.0, 5.0
        expected = f * (L / D) * 0.5 * rho * v**2
        dp = calculate_frictional_pressure_drop(f, L, D, rho, v)
        assert dp == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# calculate_elevation_pressure_drop
# ---------------------------------------------------------------------------


class TestCalculateElevationPressureDrop:
    """Unit tests for calculate_elevation_pressure_drop."""

    @pytest.mark.unit
    def test_upward_positive(self):
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=10.0)
        assert dp > 0

    @pytest.mark.unit
    def test_downward_negative(self):
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=-10.0)
        assert dp < 0

    @pytest.mark.unit
    def test_horizontal_zero(self):
        dp = calculate_elevation_pressure_drop(density=1.2, elevation_change=0.0)
        assert dp == 0.0

    @pytest.mark.unit
    def test_formula_check(self):
        rho, g_approx, dh = 800.0, 9.80665, 5.0
        dp = calculate_elevation_pressure_drop(density=rho, elevation_change=dh)
        assert dp == pytest.approx(rho * g_approx * dh, rel=1e-6)


# ---------------------------------------------------------------------------
# calculate_erosional_velocity
# ---------------------------------------------------------------------------


class TestCalculateErosionalVelocity:
    """Unit tests for calculate_erosional_velocity."""

    @pytest.mark.unit
    def test_returns_positive(self):
        v = calculate_erosional_velocity(density=1.5)
        assert v > 0

    @pytest.mark.unit
    def test_higher_density_lower_velocity(self):
        """Denser fluid → lower erosional limit."""
        v_light = calculate_erosional_velocity(density=1.0)
        v_heavy = calculate_erosional_velocity(density=10.0)
        assert v_light > v_heavy

    @pytest.mark.unit
    def test_continuous_service_default(self):
        v_cont = calculate_erosional_velocity(density=1.2, service_type="continuous")
        v_default = calculate_erosional_velocity(density=1.2)
        assert v_cont == pytest.approx(v_default, rel=1e-9)

    @pytest.mark.unit
    def test_intermittent_higher_than_continuous(self):
        v_cont = calculate_erosional_velocity(density=1.2, service_type="continuous")
        v_int = calculate_erosional_velocity(density=1.2, service_type="intermittent")
        assert v_int > v_cont


# ---------------------------------------------------------------------------
# calculate_expansion_factor
# ---------------------------------------------------------------------------


class TestCalculateExpansionFactor:
    """Unit tests for calculate_expansion_factor."""

    @pytest.mark.unit
    def test_zero_pressure_drop_is_one(self):
        Y = calculate_expansion_factor(
            inlet_pressure=1e5,
            pressure_drop=0.0,
            friction_factor=0.02,
            length_over_diameter=100,
        )
        assert Y == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.unit
    def test_returns_between_zero_and_one(self):
        Y = calculate_expansion_factor(
            inlet_pressure=1e5,
            pressure_drop=5000.0,
            friction_factor=0.02,
            length_over_diameter=100,
        )
        assert 0.0 <= Y <= 1.0

    @pytest.mark.unit
    def test_negative_pressure_returns_one(self):
        Y = calculate_expansion_factor(
            inlet_pressure=1e5,
            pressure_drop=-100.0,
            friction_factor=0.02,
            length_over_diameter=100,
        )
        assert Y == 1.0

    @pytest.mark.unit
    def test_choked_flow_returns_zero(self):
        """When dP = P1 (all pressure lost), should approach 0."""
        Y = calculate_expansion_factor(
            inlet_pressure=1e5,
            pressure_drop=1e5,  # total pressure lost
            friction_factor=0.02,
            length_over_diameter=100,
        )
        assert Y == 0.0


# ---------------------------------------------------------------------------
# calculate_compressible_flow_correction
# ---------------------------------------------------------------------------


class TestCalculateCompressibleFlowCorrection:
    """Unit tests for calculate_compressible_flow_correction."""

    @pytest.mark.unit
    def test_returns_tuple(self):
        result = calculate_compressible_flow_correction(
            inlet_pressure=25e5,
            outlet_pressure=24e5,
            length=100.0,
            diameter=0.1,
            mass_flow_rate=1.0,
            temperature=500.0,
            molecular_weight=20.0,
            compressibility_factor=1.0,
            friction_factor=0.02,
            total_k_factor=0.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_corrected_dp_positive(self):
        dp, P2 = calculate_compressible_flow_correction(
            inlet_pressure=25e5,
            outlet_pressure=24e5,
            length=100.0,
            diameter=0.1,
            mass_flow_rate=1.0,
            temperature=500.0,
            molecular_weight=20.0,
            compressibility_factor=1.0,
            friction_factor=0.02,
        )
        assert dp > 0
        assert P2 >= 0

    @pytest.mark.unit
    def test_zero_diameter_raises(self):
        with pytest.raises(ValueError):
            calculate_compressible_flow_correction(
                inlet_pressure=25e5,
                outlet_pressure=24e5,
                length=100.0,
                diameter=0.0,  # invalid
                mass_flow_rate=1.0,
                temperature=500.0,
                molecular_weight=20.0,
                compressibility_factor=1.0,
                friction_factor=0.02,
            )
