"""Focused coverage for shared environment configuration helpers (issue #4913)."""

from __future__ import annotations

import pytest

from config import EnvironmentError, get_env, get_env_bool, get_env_float, get_env_int

pytestmark = pytest.mark.unit


def test_get_env_handles_missing_defaults_required_and_stripping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOOLS_ENV_MISSING", raising=False)
    monkeypatch.setenv("TOOLS_ENV_VALUE", "  value  ")

    assert get_env("TOOLS_ENV_MISSING") is None
    assert get_env("TOOLS_ENV_MISSING", default="fallback") == "fallback"
    assert get_env("TOOLS_ENV_VALUE") == "value"
    assert get_env("TOOLS_ENV_VALUE", strip=False) == "  value  "

    with pytest.raises(ValueError, match="name must be provided"):
        get_env("")

    with pytest.raises(EnvironmentError) as exc_info:
        get_env("TOOLS_ENV_MISSING", required=True)

    error = exc_info.value
    assert error.var_name == "TOOLS_ENV_MISSING"
    assert error.reason == "Required environment variable not set"
    assert str(error) == "TOOLS_ENV_MISSING: Required environment variable not set"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        (" YES ", True),
        ("1", True),
        ("on", True),
        ("false", False),
        (" NO ", False),
        ("0", False),
        ("off", False),
        ("", False),
    ],
)
def test_get_env_bool_parses_known_values(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("TOOLS_ENV_BOOL", raw)

    assert get_env_bool("TOOLS_ENV_BOOL", default=not expected) is expected


def test_get_env_bool_uses_default_for_missing_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOOLS_ENV_BOOL", raising=False)
    assert get_env_bool("TOOLS_ENV_BOOL", default=True) is True

    monkeypatch.setenv("TOOLS_ENV_BOOL", "maybe")
    assert get_env_bool("TOOLS_ENV_BOOL", default=False) is False


def test_get_env_int_parses_defaults_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOOLS_ENV_INT", raising=False)
    assert get_env_int("TOOLS_ENV_INT") is None
    assert get_env_int("TOOLS_ENV_INT", default=7) == 7

    monkeypatch.setenv("TOOLS_ENV_INT", "42")
    assert get_env_int("TOOLS_ENV_INT", min_value=40, max_value=42) == 42

    monkeypatch.setenv("TOOLS_ENV_INT", "39")
    with pytest.raises(EnvironmentError, match="TOOLS_ENV_INT: Value below minimum"):
        get_env_int("TOOLS_ENV_INT", min_value=40)

    monkeypatch.setenv("TOOLS_ENV_INT", "43")
    with pytest.raises(EnvironmentError, match="TOOLS_ENV_INT: Value above maximum"):
        get_env_int("TOOLS_ENV_INT", max_value=42)

    monkeypatch.setenv("TOOLS_ENV_INT", "not-int")
    with pytest.raises(EnvironmentError) as exc_info:
        get_env_int("TOOLS_ENV_INT")

    error = exc_info.value
    assert error.expected == "integer"
    assert error.actual == "not-int"
    assert "expected integer" in str(error)
    assert "actual 'not-int'" in str(error)


def test_get_env_float_parses_defaults_and_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOOLS_ENV_FLOAT", raising=False)
    assert get_env_float("TOOLS_ENV_FLOAT") is None
    assert get_env_float("TOOLS_ENV_FLOAT", default=1.25) == 1.25

    monkeypatch.setenv("TOOLS_ENV_FLOAT", "2.5")
    assert get_env_float("TOOLS_ENV_FLOAT", min_value=2.0, max_value=3.0) == 2.5

    monkeypatch.setenv("TOOLS_ENV_FLOAT", "1.5")
    with pytest.raises(EnvironmentError, match="TOOLS_ENV_FLOAT: Value below minimum"):
        get_env_float("TOOLS_ENV_FLOAT", min_value=2.0)

    monkeypatch.setenv("TOOLS_ENV_FLOAT", "3.5")
    with pytest.raises(EnvironmentError, match="TOOLS_ENV_FLOAT: Value above maximum"):
        get_env_float("TOOLS_ENV_FLOAT", max_value=3.0)

    monkeypatch.setenv("TOOLS_ENV_FLOAT", "not-float")
    with pytest.raises(EnvironmentError) as exc_info:
        get_env_float("TOOLS_ENV_FLOAT")

    error = exc_info.value
    assert error.expected == "float"
    assert error.actual == "not-float"
    assert "expected float" in str(error)
    assert "actual 'not-float'" in str(error)
