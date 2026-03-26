"""Tests for python.src.utils.env_utils module.

Covers:
- get_env_var (default, required)
- get_env_bool
- get_env_int
"""

from __future__ import annotations

import pytest
from utils.env_utils import get_env_bool, get_env_int, get_env_var


class TestGetEnvVar:
    """Tests for get_env_var function."""

    def test_returns_value_if_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "hello")
        assert get_env_var("TEST_KEY") == "hello"

    def test_returns_default_if_unset(self) -> None:
        key = "DEFINITELY_NOT_SET_12345"
        assert get_env_var(key, default="fallback") == "fallback"

    def test_returns_none_if_unset_no_default(self) -> None:
        key = "DEFINITELY_NOT_SET_12345"
        assert get_env_var(key) is None

    def test_required_raises_if_missing(self) -> None:
        key = "DEFINITELY_NOT_SET_12345"
        with pytest.raises(ValueError, match="Required environment variable"):
            get_env_var(key, required=True)

    def test_required_returns_value_if_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REQUIRED_KEY", "present")
        assert get_env_var("REQUIRED_KEY", required=True) == "present"


class TestGetEnvBool:
    """Tests for get_env_bool function."""

    def test_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("true", "1", "yes", "on", "True", "YES"):
            monkeypatch.setenv("BOOL_KEY", val)
            assert get_env_bool("BOOL_KEY") is True

    def test_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("false", "0", "no", "off", "anything"):
            monkeypatch.setenv("BOOL_KEY", val)
            assert get_env_bool("BOOL_KEY") is False

    def test_default_false(self) -> None:
        assert get_env_bool("UNSET_BOOL_KEY") is False

    def test_default_true(self) -> None:
        assert get_env_bool("UNSET_BOOL_KEY", default=True) is True


class TestGetEnvInt:
    """Tests for get_env_int function."""

    def test_valid_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INT_KEY", "42")
        assert get_env_int("INT_KEY") == 42

    def test_default_value(self) -> None:
        assert get_env_int("UNSET_INT_KEY", default=7) == 7

    def test_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INT_KEY", "notanumber")
        with pytest.raises(ValueError, match="must be an integer"):
            get_env_int("INT_KEY")

    def test_negative_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INT_KEY", "-5")
        assert get_env_int("INT_KEY") == -5
