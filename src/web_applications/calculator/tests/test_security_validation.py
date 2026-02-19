"""Tests for input validation and security measures."""

import pytest

from web_applications.calculator.webapp import _parse_payload, _validate_security


def test_security_validation() -> None:
    # Safe inputs
    _validate_security("1 + 1")
    _validate_security("sin(x)")
    _validate_security("classic_variable")  # 'class' is a substring but not a keyword

    # Dangerous inputs
    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("(1).__class__")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("__import__('os')")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("x.__base__")

    # New blocked keywords
    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("async function")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("await result")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("global x")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("del x")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("try" + ": pass")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("except" + ": pass")


def test_payload_function_security() -> None:
    """Test that the 'function' parameter in payload is validated for security."""
    payload = {
        "operation": "solve_ode",
        "expression": "y",
        "function": "__init__",
    }

    with pytest.raises(ValueError, match="Security violation"):
        _parse_payload(payload)
