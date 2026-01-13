import pytest

from web_applications.calculator.webapp import _validate_security


def test_security_validation() -> None:
    # Safe inputs
    assert _validate_security("1 + 1") is None
    assert _validate_security("sin(x)") is None
    assert _validate_security("classic_variable") is None  # 'class' is a substring but not a keyword

    # Dangerous inputs
    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("(1).__class__")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("__import__('os')")

    with pytest.raises(ValueError, match="Security violation"):
        _validate_security("x.__base__")

    with pytest.raises(ValueError, match="Restricted keyword 'lambda'"):
        _validate_security("lambda x: x+1")

    with pytest.raises(ValueError, match="Restricted keyword 'class'"):
        _validate_security("class MyClass:")

    with pytest.raises(ValueError, match="Restricted keyword 'import'"):
        _validate_security("import os")

    with pytest.raises(ValueError, match="Restricted keyword 'exec'"):
        _validate_security("exec('print(1)')")
