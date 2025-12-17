"""Example test module to verify testing infrastructure works correctly."""

import pytest

# Constants for test values
EXPECTED_SUM: int = 4
EXPECTED_PRODUCT: int = 12
EXPECTED_QUOTIENT: int = 5


def test_basic_arithmetic() -> None:
    """Test basic arithmetic to verify testing framework is working."""
    # Test basic arithmetic operations
    assert EXPECTED_SUM == 2 + 2, "Basic arithmetic should work"
    assert EXPECTED_PRODUCT == 3 * 4, "Multiplication should work"
    assert EXPECTED_QUOTIENT == 10 / 2, "Division should work"
