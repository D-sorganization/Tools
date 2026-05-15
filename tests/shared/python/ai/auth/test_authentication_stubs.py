"""Tests for authentication stubs - verifying NotImplementedError is raised."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: stub the broken src.shared.python.ai __init__ and logging_pkg so
# that importing the authentication module works in a plain pytest run.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.auth", "src/shared/python/ai/auth"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

_logging_config_stub = sys.modules["src.shared.python.logging_pkg.logging_config"]
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from src.shared.python.ai.auth.authentication import AuthManager  # noqa: E402


def test_login_with_oauth_raises_not_implemented():
    """login_with_oauth must raise NotImplementedError, not fabricate a user."""
    auth = AuthManager()
    with pytest.raises(NotImplementedError, match="OAuth token exchange"):
        auth.login_with_oauth("google", "fake_auth_code")


def test_login_with_oauth_any_provider_raises():
    """login_with_oauth raises for any provider string, including empty."""
    auth = AuthManager()
    for provider in ("github", "microsoft", "", "arbitrary_provider"):
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth(provider, "code")


def test_login_with_oauth_does_not_set_current_user():
    """login_with_oauth must not set a new user when it raises."""
    auth = AuthManager()
    user_before = auth.current_user
    with pytest.raises(NotImplementedError):
        auth.login_with_oauth("google", "fake_code")
    # NotImplementedError must propagate before any user is set;
    # the state must not change after the failed call.
    assert auth.current_user is user_before


def test_login_with_api_key_raises_not_implemented():
    """login_with_api_key must raise NotImplementedError."""
    auth = AuthManager()
    with pytest.raises(
        NotImplementedError,
        match="Server-side API key verification is not implemented",
    ):
        auth.login_with_api_key("some-api-key")


def test_login_with_api_key_does_not_set_current_user():
    """login_with_api_key must not change user state when it raises."""
    auth = AuthManager()
    user_before = auth.current_user
    with pytest.raises(NotImplementedError):
        auth.login_with_api_key("some-api-key")
    # NotImplementedError must propagate before any user is set;
    # the state must not change after the failed call.
    assert auth.current_user is user_before
