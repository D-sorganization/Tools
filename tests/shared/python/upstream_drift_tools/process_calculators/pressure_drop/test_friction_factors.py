"""Tests for the _friction_factors sub-module.

Covers all four friction factor correlations and the selector function,
including laminar/turbulent regime handling and convergence behaviour.
"""

from __future__ import annotations

import pytest
import upstream_drift_tools.process_calculators.pressure_drop_calculator.engine._friction_factors as _ff  # noqa: E501

friction_factor_churchill = _ff.friction_factor_churchill
friction_factor_colebrook = _ff.friction_factor_colebrook
friction_factor_haaland = _ff.friction_factor_haaland
friction_factor_laminar = _ff.friction_factor_laminar
friction_factor_swamee_jain = _ff.friction_factor_swamee_jain
select_friction_factor_method = _ff.select_friction_factor_method

# ---------------------------------------------------------------------------
# friction_factor_laminar
# ---------------------------------------------------------------------------


class TestFrictionFactorLaminar:
    """Unit tests for friction_factor_laminar."""

    @pytest.mark.unit
    def test_standard_re_1000(self):
        """f = 64/Re for Re=1000."""
        f = friction_factor_laminar(1000)
        assert f == pytest.approx(0.064, rel=1e-6)

    @pytest.mark.unit
    def test_standard_re_2000(self):
        """f = 64/Re for Re=2000."""
        f = friction_factor_laminar(2000)
        assert f == pytest.approx(0.032, rel=1e-6)

    @pytest.mark.unit
    def test_positive_result(self):
        assert friction_factor_laminar(500) > 0

    @pytest.mark.unit
    def test_nonpositive_re_returns_default(self):
        """Negative Re should return default (no crash)."""
        result = friction_factor_laminar(-100)
        assert result > 0  # Returns a default value

    @pytest.mark.unit
    def test_zero_re_returns_default(self):
        result = friction_factor_laminar(0)
        assert result > 0


# ---------------------------------------------------------------------------
# friction_factor_colebrook
# ---------------------------------------------------------------------------


class TestFrictionFactorColebrook:
    """Unit tests for friction_factor_colebrook."""

    @pytest.mark.unit
    def test_turbulent_smooth(self):
        """High Re, smooth pipe: f should be in a physically reasonable range."""
        f = friction_factor_colebrook(100_000, 0.0001)
        assert 0.01 < f < 0.04

    @pytest.mark.unit
    def test_laminar_regime_delegates(self):
        """Below Re_laminar_upper delegates to laminar formula."""
        f_cole = friction_factor_colebrook(1000, 0.0001)
        f_lam = friction_factor_laminar(1000)
        assert f_cole == pytest.approx(f_lam, rel=1e-6)

    @pytest.mark.unit
    def test_convergence_high_re(self):
        """Colebrook should converge for Re=1e7."""
        f = friction_factor_colebrook(1e7, 0.0001)
        assert 0.005 < f < 0.02

    @pytest.mark.unit
    def test_missing_re_raises(self):
        with pytest.raises((TypeError, ValueError)):
            friction_factor_colebrook(None, 0.0001)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# friction_factor_swamee_jain
# ---------------------------------------------------------------------------


class TestFrictionFactorSwameeJain:
    """Unit tests for friction_factor_swamee_jain."""

    @pytest.mark.unit
    def test_turbulent_result_reasonable(self):
        f = friction_factor_swamee_jain(100_000, 0.0001)
        assert 0.01 < f < 0.04

    @pytest.mark.unit
    def test_agrees_with_colebrook_within_2pct(self):
        """Swamee-Jain should be within ~1% of Colebrook for valid range."""
        Re = 100_000
        eps_D = 0.0001
        f_sj = friction_factor_swamee_jain(Re, eps_D)
        f_cb = friction_factor_colebrook(Re, eps_D)
        assert abs(f_sj / f_cb - 1) < 0.02

    @pytest.mark.unit
    def test_laminar_regime_delegates(self):
        f_sj = friction_factor_swamee_jain(1000, 0.0001)
        f_lam = friction_factor_laminar(1000)
        assert f_sj == pytest.approx(f_lam, rel=1e-6)


# ---------------------------------------------------------------------------
# friction_factor_churchill
# ---------------------------------------------------------------------------


class TestFrictionFactorChurchill:
    """Unit tests for friction_factor_churchill."""

    @pytest.mark.unit
    def test_turbulent_result_reasonable(self):
        f = friction_factor_churchill(100_000, 0.0001)
        assert 0.01 < f < 0.04

    @pytest.mark.unit
    def test_laminar_regime(self):
        """Churchill should give ~64/Re for laminar flow."""
        Re = 1000
        f = friction_factor_churchill(Re, 0.0)
        expected = 64 / Re
        assert f == pytest.approx(expected, rel=0.02)

    @pytest.mark.unit
    def test_very_low_re_does_not_crash(self):
        f = friction_factor_churchill(0.5, 0.0)
        assert f > 0

    @pytest.mark.unit
    def test_all_regimes_return_positive(self):
        for Re in [100, 2000, 4000, 100_000, 1e7]:
            f = friction_factor_churchill(Re, 0.001)
            assert f > 0, f"Expected positive f at Re={Re}"


# ---------------------------------------------------------------------------
# friction_factor_haaland
# ---------------------------------------------------------------------------


class TestFrictionFactorHaaland:
    """Unit tests for friction_factor_haaland."""

    @pytest.mark.unit
    def test_turbulent_result_reasonable(self):
        f = friction_factor_haaland(100_000, 0.0001)
        assert 0.01 < f < 0.04

    @pytest.mark.unit
    def test_agrees_with_colebrook_within_2pct(self):
        Re = 100_000
        eps_D = 0.0001
        f_haa = friction_factor_haaland(Re, eps_D)
        f_cb = friction_factor_colebrook(Re, eps_D)
        assert abs(f_haa / f_cb - 1) < 0.02

    @pytest.mark.unit
    def test_laminar_regime_delegates(self):
        f_haa = friction_factor_haaland(1000, 0.0001)
        f_lam = friction_factor_laminar(1000)
        assert f_haa == pytest.approx(f_lam, rel=1e-6)


# ---------------------------------------------------------------------------
# select_friction_factor_method
# ---------------------------------------------------------------------------


class TestSelectFrictionFactorMethod:
    """Unit tests for select_friction_factor_method."""

    @pytest.mark.unit
    def test_colebrook_method(self):
        f = select_friction_factor_method("colebrook", 100_000, 0.0001)
        assert f == pytest.approx(friction_factor_colebrook(100_000, 0.0001), rel=1e-6)

    @pytest.mark.unit
    def test_swamee_jain_method(self):
        f = select_friction_factor_method("swamee-jain", 100_000, 0.0001)
        assert f == pytest.approx(
            friction_factor_swamee_jain(100_000, 0.0001), rel=1e-6
        )

    @pytest.mark.unit
    def test_swamee_jain_underscore_alias(self):
        """swamee_jain and swamee-jain should produce the same result."""
        f1 = select_friction_factor_method("swamee-jain", 100_000, 0.0001)
        f2 = select_friction_factor_method("swamee_jain", 100_000, 0.0001)
        assert f1 == pytest.approx(f2, rel=1e-9)

    @pytest.mark.unit
    def test_churchill_method(self):
        f = select_friction_factor_method("churchill", 100_000, 0.0001)
        assert f == pytest.approx(friction_factor_churchill(100_000, 0.0001), rel=1e-6)

    @pytest.mark.unit
    def test_haaland_method(self):
        f = select_friction_factor_method("haaland", 100_000, 0.0001)
        assert f == pytest.approx(friction_factor_haaland(100_000, 0.0001), rel=1e-6)

    @pytest.mark.unit
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown friction factor method"):
            select_friction_factor_method("unknown", 100_000, 0.0001)

    @pytest.mark.unit
    def test_case_insensitive(self):
        f1 = select_friction_factor_method("COLEBROOK", 100_000, 0.0001)
        f2 = select_friction_factor_method("colebrook", 100_000, 0.0001)
        assert f1 == pytest.approx(f2, rel=1e-9)

    @pytest.mark.unit
    def test_none_method_raises(self):
        with pytest.raises((TypeError, ValueError)):
            select_friction_factor_method(None, 100_000, 0.0001)  # type: ignore[arg-type]
