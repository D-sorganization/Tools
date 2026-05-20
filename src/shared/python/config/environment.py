"""Environment variable helpers for shared Tools modules."""

from __future__ import annotations

import os


class EnvironmentError(RuntimeError):
    """Raised when a required environment variable is missing or invalid."""

    def __init__(
        self,
        var_name: str,
        reason: str = "Environment variable not set or invalid",
        *,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
        details = reason
        if expected is not None:
            details = f"{details}; expected {expected}"
        if actual is not None:
            details = f"{details}; actual {actual!r}"
        super().__init__(f"{var_name}: {details}")
        self.var_name = var_name
        self.reason = reason
        self.expected = expected
        self.actual = actual


def get_env(
    name: str,
    default: str | None = None,
    *,
    required: bool = False,
    strip: bool = True,
) -> str | None:
    """Read an environment variable with optional default and validation."""
    if not name:
        raise ValueError("name must be provided")
    value = os.environ.get(name)
    if value is not None:
        return value.strip() if strip else value
    if default is not None:
        return default
    if required:
        raise EnvironmentError(name, "Required environment variable not set")
    return None


def get_env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = get_env(name)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"true", "yes", "1", "on"}:
        return True
    if normalized in {"false", "no", "0", "off", ""}:
        return False
    return default


def get_env_int(
    name: str,
    default: int | None = None,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Read an integer environment variable with optional bounds."""
    value = get_env(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EnvironmentError(
            name,
            "Invalid integer value",
            expected="integer",
            actual=value,
        ) from exc
    if min_value is not None and parsed < min_value:
        raise EnvironmentError(name, "Value below minimum", actual=str(parsed))
    if max_value is not None and parsed > max_value:
        raise EnvironmentError(name, "Value above maximum", actual=str(parsed))
    return parsed


def get_env_float(
    name: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Read a float environment variable with optional bounds."""
    value = get_env(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise EnvironmentError(
            name,
            "Invalid float value",
            expected="float",
            actual=value,
        ) from exc
    if min_value is not None and parsed < min_value:
        raise EnvironmentError(name, "Value below minimum", actual=str(parsed))
    if max_value is not None and parsed > max_value:
        raise EnvironmentError(name, "Value above maximum", actual=str(parsed))
    return parsed


__all__ = [
    "EnvironmentError",
    "get_env",
    "get_env_bool",
    "get_env_float",
    "get_env_int",
]
