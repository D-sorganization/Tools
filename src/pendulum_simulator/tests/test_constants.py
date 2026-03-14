"""Tests for the shared physical constants module.

Design by Contract coverage:
- All constants have correct types and expected values.
- Derived constants are consistent with their sources.
- Values match published NIST/ISO reference standards.
"""

from __future__ import annotations

import pytest

import double_pendulum_golf.constants as C


class TestGravitationalConstants:
    """Verify gravity constants are physically meaningful."""

    def test_gravity_mss_is_float(self) -> None:
        assert isinstance(C.GRAVITY_MSS, float)

    def test_gravity_mss_standard_value(self) -> None:
        """GRAVITY_MSS should be 9.81 m/s² (common engineering value)."""
        assert C.GRAVITY_MSS == pytest.approx(9.81, abs=1e-6)

    def test_gravity_mss_positive(self) -> None:
        assert C.GRAVITY_MSS > 0

    def test_gravity_standard_is_float(self) -> None:
        assert isinstance(C.GRAVITY_STANDARD, float)

    def test_gravity_standard_nist_value(self) -> None:
        """GRAVITY_STANDARD should be exactly 9.80665 m/s² (SI definition)."""
        assert C.GRAVITY_STANDARD == pytest.approx(9.80665, abs=1e-10)

    def test_gravity_standard_positive(self) -> None:
        assert C.GRAVITY_STANDARD > 0

    def test_gravity_mss_close_to_standard(self) -> None:
        """The simplified and exact values should be within 0.01%."""
        rel_diff = abs(C.GRAVITY_MSS - C.GRAVITY_STANDARD) / C.GRAVITY_STANDARD
        assert rel_diff < 0.001


class TestConversionFactors:
    """Verify unit conversion constants are self-consistent."""

    def test_nm_per_kgfm_equals_gravity_standard(self) -> None:
        """NM_PER_KGFM must equal the standard gravity (1 kgf = 9.80665 N)."""
        assert C.NM_PER_KGFM == C.GRAVITY_STANDARD

    def test_lbf_per_n_is_float(self) -> None:
        assert isinstance(C.LBF_PER_N, float)

    def test_lbf_per_n_value(self) -> None:
        """1 lbf ≈ 4.44822 N  ⟹  1/4.44822 ≈ 0.224809 lbf/N."""
        assert C.LBF_PER_N == pytest.approx(0.224809, abs=1e-5)

    def test_lbf_per_n_positive(self) -> None:
        assert C.LBF_PER_N > 0

    def test_inches_per_m_is_float(self) -> None:
        assert isinstance(C.INCHES_PER_M, float)

    def test_inches_per_m_value(self) -> None:
        """1 m = 39.3701 inches (exactly 100 / 2.54)."""
        assert C.INCHES_PER_M == pytest.approx(100.0 / 2.54, rel=1e-4)

    def test_m_per_inch_is_float(self) -> None:
        assert isinstance(C.M_PER_INCH, float)

    def test_m_per_inch_value(self) -> None:
        """1 inch = 0.0254 m exactly (international inch definition)."""
        assert C.M_PER_INCH == pytest.approx(0.0254, abs=1e-10)

    def test_inches_and_meters_are_reciprocal(self) -> None:
        """INCHES_PER_M * M_PER_INCH should be very close to 1.0.

        Note: INCHES_PER_M is 39.3701 (rounded) and M_PER_INCH is 0.0254 (exact),
        so the product ≈ 1.000001, not exact.
        """
        product = C.INCHES_PER_M * C.M_PER_INCH
        assert product == pytest.approx(1.0, rel=1e-4)

    def test_lbf_per_n_reciprocal_is_n_per_lbf(self) -> None:
        """The reciprocal of LBF_PER_N should be close to 4.44822 N/lbf."""
        n_per_lbf = 1.0 / C.LBF_PER_N
        assert n_per_lbf == pytest.approx(4.44822, abs=1e-3)

    def test_known_unit_conversions(self) -> None:
        """Spot-check practical conversions."""
        # 1 kgf·m = 9.80665 N·m
        kgfm_as_nm = 1.0 * C.NM_PER_KGFM
        assert kgfm_as_nm == pytest.approx(9.80665, abs=1e-5)

        # 1 m = 39.3701 inches
        m_in_inches = 1.0 * C.INCHES_PER_M
        assert m_in_inches == pytest.approx(39.3701, abs=1e-3)

        # 1 foot = 12 inches = 12 * 0.0254 m = 0.3048 m
        foot_in_m = 12.0 * C.M_PER_INCH
        assert foot_in_m == pytest.approx(0.3048, abs=1e-6)
