"""Shared numerical testing helpers for property-based and tolerance testing.

Provides reusable assertion helpers and Hypothesis strategies for scientific
computing tests across the Tools repository.

Design by Contract:
    Precondition: tolerance > 0 for all comparison functions
    Postcondition: assert functions raise AssertionError with diagnostic message on failure
"""

from __future__ import annotations

import math


def assert_close(
    actual: float,
    expected: float,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    msg: str = "",
) -> None:
    """Assert two floats are close within relative and absolute tolerance.

    Uses the same semantics as numpy.isclose: |actual - expected| <= atol + rtol * |expected|

    Args:
        actual: The computed value.
        expected: The reference value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        msg: Optional additional message on failure.

    Raises:
        AssertionError: If values are not close.
    """
    if rtol <= 0:
        raise ValueError(f"rtol must be positive, got {rtol}")
    if atol < 0:
        raise ValueError(f"atol must be non-negative, got {atol}")

    diff = abs(actual - expected)
    threshold = atol + rtol * abs(expected)
    if diff > threshold:
        detail = (
            f"assert_close failed: |{actual} - {expected}| = {diff} "
            f"> {threshold} (atol={atol}, rtol={rtol})"
        )
        if msg:
            detail = f"{msg}: {detail}"
        raise AssertionError(detail)


def assert_conserved(
    before: float,
    after: float,
    quantity_name: str = "quantity",
    rtol: float = 1e-6,
) -> None:
    """Assert a conserved quantity has not changed (e.g., mass, energy, elements).

    Args:
        before: Value before operation.
        after: Value after operation.
        quantity_name: Name for diagnostics.
        rtol: Relative tolerance.

    Raises:
        AssertionError: If quantity changed beyond tolerance.
    """
    if before == 0 and after == 0:
        return
    if before == 0:
        raise AssertionError(
            f"{quantity_name} conservation violated: before=0, after={after}"
        )
    relative_change = abs(after - before) / abs(before)
    if relative_change > rtol:
        raise AssertionError(
            f"{quantity_name} conservation violated: "
            f"before={before}, after={after}, "
            f"relative change={relative_change:.2e} > rtol={rtol}"
        )


def assert_monotonic(
    values: list[float],
    increasing: bool = True,
    strict: bool = False,
    label: str = "sequence",
) -> None:
    """Assert a sequence is monotonically increasing or decreasing.

    Args:
        values: Sequence of values to check.
        increasing: True for non-decreasing, False for non-increasing.
        strict: If True, require strictly increasing/decreasing.
        label: Name for diagnostics.

    Raises:
        AssertionError: If monotonicity is violated.
    """
    if len(values) < 2:
        return
    for i in range(1, len(values)):
        if increasing:
            if strict and values[i] <= values[i - 1]:
                raise AssertionError(
                    f"{label}[{i}]={values[i]} <= {label}[{i - 1}]={values[i - 1]} "
                    f"(expected strictly increasing)"
                )
            elif not strict and values[i] < values[i - 1]:
                raise AssertionError(
                    f"{label}[{i}]={values[i]} < {label}[{i - 1}]={values[i - 1]} "
                    f"(expected non-decreasing)"
                )
        else:
            if strict and values[i] >= values[i - 1]:
                raise AssertionError(
                    f"{label}[{i}]={values[i]} >= {label}[{i - 1}]={values[i - 1]} "
                    f"(expected strictly decreasing)"
                )
            elif not strict and values[i] > values[i - 1]:
                raise AssertionError(
                    f"{label}[{i}]={values[i]} > {label}[{i - 1}]={values[i - 1]} "
                    f"(expected non-increasing)"
                )


def is_finite(value: float) -> bool:
    """Check if a value is finite (not NaN, not inf)."""
    return math.isfinite(value)
