"""Focused coverage for shared deprecation helpers."""

from __future__ import annotations

import pytest

from src.shared.python.deprecation import deprecated

pytestmark = pytest.mark.unit


def test_deprecated_rejects_invalid_configuration() -> None:
    with pytest.raises(TypeError, match="reason must be a string, got int"):
        deprecated(reason=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="removal_version must not be an empty string"):
        deprecated(removal_version="  ")


def test_deprecated_preserves_metadata_warns_and_calls_function() -> None:
    calls: list[tuple[int, int]] = []

    @deprecated(reason="Use add_new instead.", removal_version="2.0.0")
    def add_old(left: int, right: int = 1) -> int:
        """Add two values through the legacy path."""
        calls.append((left, right))
        return left + right

    with pytest.warns(DeprecationWarning) as warnings:
        result = add_old(2, right=3)

    assert result == 5
    assert calls == [(2, 3)]
    assert add_old.__name__ == "add_old"
    assert add_old.__doc__ == "Add two values through the legacy path."
    assert str(warnings[0].message) == (
        "test_deprecated_preserves_metadata_warns_and_calls_function.<locals>."
        "add_old is deprecated: Use add_new instead. "
        "(will be removed in 2.0.0)"
    )


def test_deprecated_supports_minimal_warning_message() -> None:
    @deprecated()
    def legacy() -> str:
        return "ok"

    with pytest.warns(DeprecationWarning, match="legacy is deprecated") as warnings:
        assert legacy() == "ok"

    message = str(warnings[0].message)
    assert "will be removed" not in message
    assert ": " not in message


def test_deprecated_supports_reason_without_removal_version() -> None:
    @deprecated(reason="Use replacement.")
    def legacy() -> None:
        return None

    with pytest.warns(DeprecationWarning) as warnings:
        legacy()

    assert str(warnings[0].message).endswith("legacy is deprecated: Use replacement.")


def test_deprecated_method_warning_uses_qualname() -> None:
    class Service:
        @deprecated(removal_version="3.0")
        def legacy_method(self) -> str:
            return "method-result"

    with pytest.warns(DeprecationWarning) as warnings:
        assert Service().legacy_method() == "method-result"

    message = str(warnings[0].message)
    assert "Service.legacy_method is deprecated" in message
    assert message.endswith("(will be removed in 3.0)")
