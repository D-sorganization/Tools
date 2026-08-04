"""Contract tests for named SCADA principals and short-lived sessions."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from identity import (  # noqa: E402
    CredentialRegistry,
    Principal,
    Role,
    SessionStore,
    parse_principal_config,
)

try:
    from datetime import UTC
except ImportError:  # Python 3.10 support
    UTC = timezone.utc  # noqa: UP017
_OPERATOR_SECRET = "operator-test-secret"  # pragma: allowlist secret
_ENGINEER_SECRET = "engineer-test-secret"  # pragma: allowlist secret


def _principal(name: str = "operator.one", role: Role = Role.OPERATOR) -> Principal:
    return Principal(subject=name, display_name="Operator One", role=role)


def test_role_order_enforces_least_privilege() -> None:
    viewer = _principal(role=Role.VIEWER)
    operator = _principal(role=Role.OPERATOR)
    engineer = _principal(role=Role.ENGINEER)
    admin = _principal(role=Role.ADMIN)

    assert viewer.allows(Role.VIEWER)
    assert not viewer.allows(Role.OPERATOR)
    assert operator.allows(Role.VIEWER)
    assert not operator.allows(Role.ENGINEER)
    assert engineer.allows(Role.OPERATOR)
    assert not engineer.allows(Role.ADMIN)
    assert admin.allows(Role.ADMIN)


def test_principal_rejects_blank_identity() -> None:
    with pytest.raises(ValueError, match="subject"):
        Principal(subject=" ", display_name="Operator", role=Role.OPERATOR)


def test_parse_principal_config_builds_named_registry_without_secret_repr() -> None:
    config = (
        '[{"subject":"operator.one","display_name":"Operator One",'
        '"role":"operator","api_key":"operator-test-secret"}]'  # noqa: E501  # pragma: allowlist secret
    )

    records = parse_principal_config(config)

    assert len(records) == 1
    assert records[0].principal.subject == "operator.one"
    assert records[0].principal.role is Role.OPERATOR
    assert _OPERATOR_SECRET not in repr(records[0])


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        ("{}", TypeError, "list"),
        ("[]", ValueError, "at least one"),
        (
            '[{"subject":"same","display_name":"One","role":"viewer",'
            '"api_key":"a-long-enough-secret"},'  # noqa: E501  # pragma: allowlist secret
            '{"subject":"same","display_name":"Two","role":"operator",'
            '"api_key":"another-long-secret"}]',  # noqa: E501  # pragma: allowlist secret
            ValueError,
            "duplicate subject",
        ),
        (
            '[{"subject":"short","display_name":"Short","role":"viewer",'
            '"api_key":"tiny"}]',  # noqa: E501  # pragma: allowlist secret
            ValueError,
            "at least",
        ),
    ],
)
def test_parse_principal_config_rejects_unsafe_contracts(
    config: str, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        parse_principal_config(config)


def test_registry_authenticates_named_principal() -> None:
    records = parse_principal_config(
        '[{"subject":"operator.one","display_name":"Operator One",'
        '"role":"operator","api_key":"operator-test-secret"},'  # noqa: E501  # pragma: allowlist secret
        '{"subject":"engineer.one","display_name":"Engineer One",'
        '"role":"engineer","api_key":"engineer-test-secret"}]'  # noqa: E501  # pragma: allowlist secret
    )
    registry = CredentialRegistry(records)

    assert registry.authenticate(_OPERATOR_SECRET) == records[0].principal
    assert registry.authenticate(_ENGINEER_SECRET) == records[1].principal
    assert registry.authenticate("not-a-valid-secret") is None
    assert registry.authenticate(None) is None


def test_registry_rejects_duplicate_credentials() -> None:
    config = (
        '[{"subject":"operator.one","display_name":"Operator One",'
        '"role":"operator","api_key":"operator-test-secret"},'  # noqa: E501  # pragma: allowlist secret
        '{"subject":"operator.two","display_name":"Operator Two",'
        '"role":"operator","api_key":"operator-test-secret"}]'  # noqa: E501  # pragma: allowlist secret
    )
    with pytest.raises(ValueError, match="duplicate credential"):
        CredentialRegistry(parse_principal_config(config))


def test_session_store_issues_resolves_and_revokes_opaque_token() -> None:
    now = datetime(2026, 8, 3, 20, 0, tzinfo=UTC)
    store = SessionStore(ttl=timedelta(minutes=15), clock=lambda: now)
    principal = _principal()

    issued = store.create(principal)

    assert issued.principal == principal
    assert issued.expires_at == now + timedelta(minutes=15)
    assert store.resolve(issued.token) == principal
    assert issued.token not in repr(store)
    assert store.revoke(issued.token)
    assert store.resolve(issued.token) is None
    assert not store.revoke(issued.token)


def test_session_store_expires_session() -> None:
    current = [datetime(2026, 8, 3, 20, 0, tzinfo=UTC)]
    store = SessionStore(ttl=timedelta(seconds=30), clock=lambda: current[0])
    issued = store.create(_principal())

    current[0] += timedelta(seconds=31)

    assert store.resolve(issued.token) is None


@pytest.mark.parametrize(
    "ttl",
    [timedelta(0), timedelta(seconds=-1), timedelta(days=2)],
)
def test_session_store_rejects_unsafe_ttl(ttl: timedelta) -> None:
    with pytest.raises(ValueError, match="ttl"):
        SessionStore(ttl=ttl)
