"""Configuration contracts for named and legacy SCADA identity services."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.security import HTTPAuthorizationCredentials

sys.path.insert(0, str(Path(__file__).parent.parent))

from identity import Role  # noqa: E402
from identity_config import (  # noqa: E402
    EnvironmentIdentityProvider,
    load_identity_service,
)

_OPERATOR_KEY = "operator-config-secret"  # pragma: allowlist secret
_ADMIN_KEY = "administrator-secret"  # pragma: allowlist secret


def test_named_principal_configuration_takes_precedence() -> None:
    service = load_identity_service(
        {
            "P1AM_PRINCIPALS_JSON": (
                '[{"subject":"engineer.one","display_name":"Engineer One",'
                '"role":"engineer","api_key":"engineer-config-secret"}]'  # noqa: E501  # pragma: allowlist secret
            ),
            "P1AM_API_KEY": _OPERATOR_KEY,
        }
    )

    assert service is not None
    issued = service.login("engineer-config-secret")  # pragma: allowlist secret
    assert issued is not None
    assert issued.principal.subject == "engineer.one"
    assert issued.principal.role is Role.ENGINEER
    assert service.login(_OPERATOR_KEY) is None


def test_distinct_legacy_keys_receive_named_operator_and_admin_roles() -> None:
    service = load_identity_service(
        {"P1AM_API_KEY": _OPERATOR_KEY, "P1AM_ADMIN_API_KEY": _ADMIN_KEY}
    )

    assert service is not None
    operator = service.login(_OPERATOR_KEY)
    admin = service.login(_ADMIN_KEY)
    assert operator is not None and operator.principal.role is Role.OPERATOR
    assert operator.principal.subject == "legacy.operator"
    assert admin is not None and admin.principal.role is Role.ADMIN
    assert admin.principal.subject == "legacy.admin"


def test_single_legacy_key_retains_existing_admin_capability() -> None:
    service = load_identity_service({"P1AM_API_KEY": _OPERATOR_KEY})

    assert service is not None
    issued = service.login(_OPERATOR_KEY)
    assert issued is not None
    assert issued.principal.role is Role.ADMIN
    assert issued.principal.subject == "legacy.single-key"


def test_unconfigured_identity_service_is_absent() -> None:
    assert load_identity_service({}) is None


def test_session_ttl_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="SESSION_TTL"):
        load_identity_service(
            {"P1AM_API_KEY": _OPERATOR_KEY, "P1AM_SESSION_TTL_S": "invalid"}
        )
    with pytest.raises(ValueError, match="SESSION_TTL"):
        load_identity_service(
            {"P1AM_API_KEY": _OPERATOR_KEY, "P1AM_SESSION_TTL_S": "0"}
        )


def test_configured_ttl_controls_session_expiry_window() -> None:
    service = load_identity_service(
        {"P1AM_API_KEY": _OPERATOR_KEY, "P1AM_SESSION_TTL_S": "120"}
    )
    assert service is not None
    issued = service.login(_OPERATOR_KEY)
    assert issued is not None
    remaining = (
        issued.expires_at - issued.expires_at.now(issued.expires_at.tzinfo)
    ).total_seconds()
    assert 115 <= remaining <= 120


def test_resolve_rejects_invalid_bearer_without_falling_back_to_key() -> None:
    service = load_identity_service({"P1AM_API_KEY": _OPERATOR_KEY})
    assert service is not None

    resolved = service.resolve(
        _OPERATOR_KEY,
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-session"),
    )

    assert resolved is None


def test_provider_preserves_sessions_until_identity_environment_changes() -> None:
    env = {"P1AM_API_KEY": "legacy-short-key"}  # pragma: allowlist secret
    provider = EnvironmentIdentityProvider(lambda: env)
    first = provider.get()
    assert first is not None
    issued = first.login("legacy-short-key")
    assert issued is not None

    assert provider.get() is first
    bearer = HTTPAuthorizationCredentials(scheme="Bearer", credentials=issued.token)
    assert provider.get().resolve(None, bearer) == issued.principal

    env["P1AM_API_KEY"] = "replacement-short-key"  # pragma: allowlist secret
    replacement = provider.get()
    assert replacement is not None
    assert replacement is not first
    assert replacement.resolve(None, bearer) is None


def test_legacy_keys_preserve_existing_nonempty_length_contract() -> None:
    service = load_identity_service(
        {"P1AM_API_KEY": "short-key"}
    )  # noqa: E501  # pragma: allowlist secret
    assert service is not None
    assert service.login("short-key") is not None
