"""Tests for the server-side authorization dependencies in ``auth_config``.

The backend must never trust the desktop HMI's client-side roles; every
mutating endpoint is gated by :func:`auth_config.require_api_key` and the
elevated ones additionally by :func:`auth_config.require_admin_key`. Both are
FastAPI dependencies that read the process environment at call time and raise
``HTTPException`` with a specific status code on failure.

These tests call the dependencies directly (they are plain sync callables that
accept the resolved ``api_key`` value) and monkeypatch the credential/opt-out
environment variables, so each authorization branch is exercised in isolation
without booting FastAPI. The status codes asserted here are the fail-closed
contract:

- 503 when no credential is configured (and the dev opt-out is off),
- 401 for a missing/invalid operator key,
- 403 when a present admin key is required but only an operator key is given,
- and a clean pass (``None`` return) for the correct key or the dev bypass.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth_config import (  # noqa: E402
    identity_service,
    require_admin_key,
    require_api_key,
    verify_operator_key,
)
from fastapi import HTTPException, status  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials  # noqa: E402
from identity import Principal, Role  # noqa: E402

_OPERATOR_KEY = "operator-secret"  # pragma: allowlist secret
_ADMIN_KEY = "admin-secret"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start each test from a known-clean auth environment.

    The dependencies read ``os.environ`` at call time, so credentials leaked
    from sibling suites would otherwise change the branch taken.
    """
    for var in ("P1AM_DEV_NO_AUTH", "P1AM_API_KEY", "P1AM_ADMIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


# --------------------------------------------------------------------------- #
# require_api_key                                                              #
# --------------------------------------------------------------------------- #


def test_require_api_key_503_when_no_key_configured() -> None:
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(api_key=None)
    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_require_api_key_401_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(api_key=None)
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_api_key_401_when_key_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(api_key="not-the-key")  # pragma: allowlist secret
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_api_key_passes_with_correct_operator_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    principal = require_api_key(api_key=_OPERATOR_KEY, bearer=None)
    assert principal == Principal("legacy.single-key", "Legacy User", Role.ADMIN)


def test_require_api_key_accepts_admin_key_as_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    # The admin key is also accepted for plain operator-gated routes.
    assert require_api_key(api_key=_ADMIN_KEY, bearer=None).role is Role.ADMIN


def test_require_api_key_dev_no_auth_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    # No key configured and no key supplied, yet the bypass lets it through.
    assert require_api_key(api_key=None, bearer=None).role is Role.ADMIN


def test_require_api_key_dev_no_auth_wins_over_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    assert require_api_key(api_key=None, bearer=None).role is Role.ADMIN


# --------------------------------------------------------------------------- #
# require_admin_key                                                            #
# --------------------------------------------------------------------------- #


def test_require_admin_key_503_when_nothing_configured() -> None:
    with pytest.raises(HTTPException) as excinfo:
        require_admin_key(api_key=None)
    assert excinfo.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_require_admin_key_403_when_operator_key_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with pytest.raises(HTTPException) as excinfo:
        require_admin_key(api_key=_OPERATOR_KEY)
    assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN


def test_require_admin_key_403_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with pytest.raises(HTTPException) as excinfo:
        require_admin_key(api_key=None)
    assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN


def test_require_admin_key_passes_with_correct_admin_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    assert require_admin_key(api_key=_ADMIN_KEY, bearer=None).role is Role.ADMIN


def test_require_admin_key_accepts_operator_key_when_no_admin_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Single-key deployment: no admin key -> operator key is accepted.
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    assert require_admin_key(api_key=_OPERATOR_KEY, bearer=None).role is Role.ADMIN


def test_require_admin_key_401_with_wrong_key_when_no_admin_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    with pytest.raises(HTTPException) as excinfo:
        require_admin_key(api_key="not-the-key")  # pragma: allowlist secret
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_admin_key_dev_no_auth_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    assert require_admin_key(api_key=None, bearer=None).role is Role.ADMIN


def test_named_engineer_can_operate_but_cannot_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "P1AM_PRINCIPALS_JSON",
        '[{"subject":"eng.1","display_name":"Engineer One",'
        '"role":"engineer","api_key":"engineer-key-12345"}]',
    )
    principal = require_api_key(api_key="engineer-key-12345", bearer=None)
    assert principal.subject == "eng.1"
    with pytest.raises(HTTPException) as excinfo:
        require_admin_key(api_key="engineer-key-12345", bearer=None)
    assert excinfo.value.status_code == status.HTTP_403_FORBIDDEN


def test_operator_gate_accepts_short_lived_bearer_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "P1AM_PRINCIPALS_JSON",
        '[{"subject":"op.1","display_name":"Operator One",'
        '"role":"operator","api_key":"operator-key-12345"}]',
    )
    service = identity_service()
    assert service is not None
    issued = service.login("operator-key-12345")
    assert issued is not None
    bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials=issued.token)

    principal = require_api_key(api_key=None, bearer=bearer)
    assert principal.subject == "op.1"


def test_invalid_bearer_does_not_fall_back_to_valid_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid")
    with pytest.raises(HTTPException) as excinfo:
        require_api_key(api_key=_OPERATOR_KEY, bearer=bearer)
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED


# --------------------------------------------------------------------------- #
# verify_operator_key (WebSocket path helper)                                 #
# --------------------------------------------------------------------------- #


def test_verify_operator_key_false_when_unconfigured() -> None:
    assert verify_operator_key("anything") is False


def test_verify_operator_key_true_for_correct_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    assert verify_operator_key(_OPERATOR_KEY) is True


def test_verify_operator_key_false_for_wrong_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    assert verify_operator_key("wrong") is False


def test_verify_operator_key_dev_no_auth_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    assert verify_operator_key(None) is True
