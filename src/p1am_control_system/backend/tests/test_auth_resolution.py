"""Tests for credential *resolution* — the half-configured deployment (#4041).

``P1AM_ADMIN_API_KEY`` alone used to brick the operator surface: the operator
tier keyed off ``P1AM_API_KEY`` exclusively, so a deployment that configured
only the admin key got a live admin API and a dead display (``/api/stream``
closing 1008, ``/api/alarms/{id}/acknowledge`` 503). Full control, no feedback.

The contract asserted here:

- A configured admin key is a valid *operator* credential whenever no separate
  operator key is set (the operator tier is a subset of the admin tier).
- On success the gates *return the resolved* :class:`identity.Principal`. They
  are FastAPI dependencies whose value the routers consume for audit
  attribution, so a ``None`` return is not an option (see
  :func:`test_require_api_key_accepts_admin_key_when_only_admin_configured`).
- A configured operator key is **never** promoted to the admin tier.
- The fail-closed 503 fires only when *neither* key is configured.
- The resolved configuration is introspectable (and logged at startup) so a
  half-configured deployment is visible at boot, not at first control action.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth_config import (  # noqa: E402
    AuthConfiguration,
    log_auth_configuration,
    require_admin_key,
    require_api_key,
    resolve_auth_config,
    verify_operator_key,
)
from fastapi import HTTPException, status  # noqa: E402
from identity import Role  # noqa: E402

_OPERATOR_KEY = "operator-secret"  # pragma: allowlist secret
_ADMIN_KEY = "admin-secret"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in ("P1AM_DEV_NO_AUTH", "P1AM_API_KEY", "P1AM_ADMIN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


# --------------------------------------------------------------------------- #
# Admin-only deployment: the operator surface must stay alive                  #
# --------------------------------------------------------------------------- #


def test_verify_operator_key_accepts_admin_key_when_only_admin_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WebSocket/operator gate must accept the admin key (#4041)."""
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    assert verify_operator_key(_ADMIN_KEY) is True


def test_verify_operator_key_rejects_wrong_key_when_only_admin_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    assert verify_operator_key("nope") is False
    assert verify_operator_key(None) is False


def test_require_api_key_accepts_admin_key_when_only_admin_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/api/alarms/{id}/acknowledge`` must not 503 on an admin-only box.

    This assertion used to read ``is None``, which was the contract before named
    principals landed. It is now the resolved ``Principal``, and that is the
    contract that has to hold: ``require_api_key`` is wired directly as the
    ``operator_dependency`` of the alarm, operations, advisory and product
    routers, each of which hands the returned principal to its application
    service so the action is attributable — ``service.acknowledge(tag,
    principal)``, ``shifts.append(draft, principal)``,
    ``investigations.save(spec, principal)``. A ``None`` return would strip the
    actor out of the audit trail (and raise on the service contract checks), so
    the old assertion was the stale side of the pair, not the implementation.

    The admin-only deployment still resolves through the *operator* gate, which
    is the #4041 property this module exists to protect: the returned principal
    carries the admin role and satisfies the operator requirement.
    """
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    principal = require_api_key(_ADMIN_KEY)
    assert principal.role is Role.ADMIN
    assert principal.allows(Role.OPERATOR) is True


def test_require_api_key_rejects_missing_key_when_only_admin_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Still fails *closed* — an admin-only box is configured, not open."""
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_api_key_rejects_wrong_key_when_only_admin_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning a Principal must not make a *wrong* credential succeed.

    Guards the direction the contract change could plausibly have broken: the
    gate now produces a value on success, so this pins that the failure path
    still raises instead of resolving some default principal.
    """
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with pytest.raises(HTTPException) as exc:
        require_api_key("not-the-admin-key")  # pragma: allowlist secret
    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_api_key_503_only_when_nothing_configured() -> None:
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


def test_operator_key_is_never_promoted_to_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subset relation is one-way: admin ⊃ operator, never the reverse."""
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with pytest.raises(HTTPException) as exc:
        require_admin_key(_OPERATOR_KEY)
    assert exc.value.status_code == status.HTTP_403_FORBIDDEN


# --------------------------------------------------------------------------- #
# Introspection: a half-configured deployment is visible at boot               #
# --------------------------------------------------------------------------- #


def test_resolve_auth_config_reports_admin_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    resolved = resolve_auth_config()
    assert isinstance(resolved, AuthConfiguration)
    assert resolved.admin_key_configured is True
    assert resolved.operator_key_configured is False
    assert resolved.dev_no_auth is False
    assert resolved.authenticated is True


def test_resolve_auth_config_reports_unconfigured() -> None:
    resolved = resolve_auth_config()
    assert resolved.authenticated is False
    assert resolved.admin_key_configured is False
    assert resolved.operator_key_configured is False


def test_resolve_auth_config_never_exposes_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DbC: the description is safe to log — it must not leak a credential."""
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    description = resolve_auth_config().describe()
    assert _OPERATOR_KEY not in description
    assert _ADMIN_KEY not in description


def test_log_auth_configuration_warns_when_unauthenticated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="dcs_backend.auth"):
        resolved = log_auth_configuration()
    assert resolved.authenticated is False
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_log_auth_configuration_warns_on_dev_no_auth(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    with caplog.at_level(logging.WARNING, logger="dcs_backend.auth"):
        resolved = log_auth_configuration()
    assert resolved.dev_no_auth is True
    assert any(record.levelno >= logging.WARNING for record in caplog.records)


def test_log_auth_configuration_info_when_fully_configured(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    with caplog.at_level(logging.INFO, logger="dcs_backend.auth"):
        resolved = log_auth_configuration()
    assert resolved.authenticated is True
    assert all(record.levelno < logging.WARNING for record in caplog.records)
