"""Tests for authentication.py — Phase 1 of issue #2757.

Verifies that fake OAuth / email-password login methods refuse to fabricate
a UserProfile, and that ``is_authenticated`` defaults to False in the absence
of a real login.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: add repo root to sys.path and stub logging_pkg so that importing
# authentication.py works in a plain pytest run (mirrors pattern in
# test_adapter_factory.py).
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]

from src.shared.python.ai.auth.authentication import (  # noqa: E402
    AuthManager,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_auth(tmp_path: Path) -> AuthManager:
    """Return an ``AuthManager`` that writes credentials to *tmp_path*."""
    creds_file = tmp_path / "auth_credentials.json"
    with patch.object(AuthManager, "CREDENTIALS_FILE", new=creds_file):
        auth = AuthManager()
    # Patch at the instance level so subsequent calls use the same path.
    auth.__class__.CREDENTIALS_FILE = creds_file  # type: ignore[assignment]
    return auth


# ---------------------------------------------------------------------------
# login_with_oauth
# ---------------------------------------------------------------------------


class TestLoginWithOauth:
    """login_with_oauth must raise NotImplementedError unconditionally."""

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "any_code")

    def test_raises_for_arbitrary_provider(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("github", "abc123")

    def test_error_message_mentions_provider(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError, match="google"):
            auth.login_with_oauth("google", "any_code")

    def test_error_message_mentions_issue_number(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError, match="#5227"):
            auth.login_with_oauth("google", "any_code")

    def test_credentials_file_not_created(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        creds_file = auth.CREDENTIALS_FILE
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "any_code")
        msg = "CREDENTIALS_FILE must not be written on refused login"
        assert not creds_file.exists(), msg

    def test_is_authenticated_false_after_refused_login(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "any_code")
        assert auth.is_authenticated is False

    def test_current_user_none_after_refused_login(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "any_code")
        assert auth.current_user is None


# ---------------------------------------------------------------------------
# login_with_email_password
# ---------------------------------------------------------------------------


class TestLoginWithEmailPassword:
    """login_with_email_password must raise NotImplementedError unconditionally."""

    def test_raises_not_implemented(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("user@example.com", "password123")

    def test_error_message_mentions_email(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError, match="user@example.com"):
            auth.login_with_email_password("user@example.com", "password123")

    def test_error_message_mentions_issue_number(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError, match="#5227"):
            auth.login_with_email_password("user@example.com", "password123")

    def test_credentials_file_not_created(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        creds_file = auth.CREDENTIALS_FILE
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("user@example.com", "password123")
        msg = "CREDENTIALS_FILE must not be written on refused login"
        assert not creds_file.exists(), msg

    def test_is_authenticated_false_after_refused_login(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("user@example.com", "password123")
        assert auth.is_authenticated is False

    def test_current_user_none_after_refused_login(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("user@example.com", "password123")
        assert auth.current_user is None


# ---------------------------------------------------------------------------
# is_authenticated default behaviour
# ---------------------------------------------------------------------------


class TestIsAuthenticated:
    """is_authenticated must return False when no valid auth method succeeds."""

    def test_false_on_fresh_manager(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        assert auth.is_authenticated is False

    def test_false_after_all_fake_methods_refused(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "code")
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("a@b.com", "pw")
        assert auth.is_authenticated is False


# ---------------------------------------------------------------------------
# No callable produces a verified UserProfile via fake methods
# ---------------------------------------------------------------------------


class TestNoFakeUserProfile:
    """No auth-manager method may produce a UserProfile without real auth."""

    def test_oauth_does_not_produce_user_profile(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("google", "code")
        # current_user must remain None — no fake profile was ever set
        assert not isinstance(auth.current_user, UserProfile)

    def test_email_password_does_not_produce_user_profile(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("a@b.com", "pw")
        assert not isinstance(auth.current_user, UserProfile)

    def test_credentials_file_empty_after_refused_logins(self, tmp_path: Path) -> None:
        auth = _make_auth(tmp_path)
        creds_file = auth.CREDENTIALS_FILE
        with pytest.raises(NotImplementedError):
            auth.login_with_oauth("github", "code")
        with pytest.raises(NotImplementedError):
            auth.login_with_email_password("a@b.com", "pw")
        assert not creds_file.exists()
