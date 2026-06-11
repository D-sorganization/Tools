"""Tests for shared numerical testing helpers.

Follows TDD: these tests were written first to define the expected contract
of the helper functions.
"""

from __future__ import annotations

import pytest

from tests.helpers.numerical import (
    assert_close,
    assert_conserved,
    assert_monotonic,
    is_finite,
)


class TestAssertClose:
    """Test assert_close with various tolerance scenarios."""

    def test_exact_match(self) -> None:
        assert_close(1.0, 1.0)

    def test_within_relative_tolerance(self) -> None:
        assert_close(1.000001, 1.0, rtol=1e-5)

    def test_fails_outside_tolerance(self) -> None:
        with pytest.raises(AssertionError, match="assert_close failed"):
            assert_close(1.1, 1.0, rtol=1e-5, atol=0)

    def test_absolute_tolerance(self) -> None:
        assert_close(1e-15, 0.0, atol=1e-10)

    def test_custom_message(self) -> None:
        with pytest.raises(AssertionError, match="temperature"):
            assert_close(500.0, 300.0, rtol=0.01, msg="temperature")

    def test_negative_rtol_rejected(self) -> None:
        with pytest.raises(ValueError, match="rtol must be positive"):
            assert_close(1.0, 1.0, rtol=-1e-6)

    def test_zero_values(self) -> None:
        assert_close(0.0, 0.0)

    def test_near_zero_with_atol(self) -> None:
        assert_close(1e-12, 0.0, atol=1e-10)


class TestAssertConserved:
    """Test conservation assertion for mass/energy/element checks."""

    def test_exact_conservation(self) -> None:
        assert_conserved(100.0, 100.0, "mass")

    def test_within_tolerance(self) -> None:
        assert_conserved(100.0, 100.00005, "mass", rtol=1e-4)

    def test_fails_on_violation(self) -> None:
        with pytest.raises(AssertionError, match="mass conservation violated"):
            assert_conserved(100.0, 110.0, "mass", rtol=1e-3)

    def test_both_zero(self) -> None:
        assert_conserved(0.0, 0.0, "energy")

    def test_before_zero_after_nonzero(self) -> None:
        with pytest.raises(AssertionError, match="conservation violated"):
            assert_conserved(0.0, 1.0, "elements")


class TestAssertMonotonic:
    """Test monotonicity assertions."""

    def test_increasing(self) -> None:
        assert_monotonic([1, 2, 3, 4, 5])

    def test_non_strictly_increasing(self) -> None:
        assert_monotonic([1, 2, 2, 3])

    def test_strictly_increasing_fails_on_equal(self) -> None:
        with pytest.raises(AssertionError, match="strictly increasing"):
            assert_monotonic([1, 2, 2, 3], strict=True)

    def test_decreasing(self) -> None:
        assert_monotonic([5, 4, 3, 2, 1], increasing=False)

    def test_empty_and_single(self) -> None:
        assert_monotonic([])
        assert_monotonic([42])

    def test_fails_on_non_monotonic(self) -> None:
        with pytest.raises(AssertionError):
            assert_monotonic([1, 3, 2, 4])

    def test_rejects_non_list_values(self) -> None:
        """DbC: values must be a list."""
        with pytest.raises(TypeError, match="values must be a list"):
            assert_monotonic((1, 2, 3))  # type: ignore[arg-type]

    def test_rejects_non_bool_increasing(self) -> None:
        """DbC: increasing must be a bool."""
        with pytest.raises(TypeError, match="increasing must be a bool"):
            assert_monotonic([1, 2, 3], increasing=1)  # type: ignore[arg-type]

    def test_rejects_non_bool_strict(self) -> None:
        """DbC: strict must be a bool."""
        with pytest.raises(TypeError, match="strict must be a bool"):
            assert_monotonic([1, 2, 3], strict=1)  # type: ignore[arg-type]


class TestIsFinite:
    """Test finite value checker."""

    def test_finite_values(self) -> None:
        assert is_finite(1.0)
        assert is_finite(0.0)
        assert is_finite(-1e300)

    def test_non_finite_values(self) -> None:
        assert not is_finite(float("nan"))
        assert not is_finite(float("inf"))
        assert not is_finite(float("-inf"))
