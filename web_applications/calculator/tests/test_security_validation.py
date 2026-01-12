import pytest
from web_applications.calculator.webapp import _validate_security

def test_security_validation():
    # Safe inputs
    assert _validate_security("1 + 1") is None
    assert _validate_security("sin(x)") is None

    # Dangerous inputs
    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("(1).__class__")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("__import__('os')")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("x.__base__")
