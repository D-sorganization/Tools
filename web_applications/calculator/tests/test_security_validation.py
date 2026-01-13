import pytest

from web_applications.calculator.webapp import _parse_payload, _validate_security


def test_security_validation() -> None:
    # Safe inputs
    assert _validate_security("1 + 1") is None
    assert _validate_security("sin(x)") is None
    assert (
        _validate_security("classic_variable") is None
    )  # 'class' is a substring but not a keyword

    # Dangerous inputs
    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("(1).__class__")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("__import__('os')")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("x.__base__")


def test_payload_function_security() -> None:
    """Test that the 'function' parameter in payload is validated for security."""
    payload = {
        "operation": "solve_ode",
        "expression": "y",
        "function": "__init__",
    }

    with pytest.raises(ValueError, match="Security violation"):
        _parse_payload(payload)
