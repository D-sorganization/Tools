"""Environment adapter for the canonical named-identity service."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta

from identity import (
    DEFAULT_SESSION_TTL,
    CredentialRecord,
    CredentialRegistry,
    Principal,
    Role,
    SessionStore,
    parse_principal_config,
)
from identity_router import IdentityService

PRINCIPALS_VARIABLE = "P1AM_PRINCIPALS_JSON"
OPERATOR_KEY_VARIABLE = "P1AM_API_KEY"
ADMIN_KEY_VARIABLE = "P1AM_ADMIN_API_KEY"
SESSION_TTL_VARIABLE = "P1AM_SESSION_TTL_S"


def _session_ttl(env: Mapping[str, str]) -> timedelta:
    raw = env.get(SESSION_TTL_VARIABLE)
    if raw is None or not raw.strip():
        return DEFAULT_SESSION_TTL
    try:
        seconds = int(raw)
    except ValueError as exc:
        raise ValueError(f"{SESSION_TTL_VARIABLE} must be an integer") from exc
    ttl = timedelta(seconds=seconds)
    try:
        SessionStore(ttl=ttl)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SESSION_TTL_VARIABLE} is outside the safe range") from exc
    return ttl


def _legacy_records(env: Mapping[str, str]) -> tuple[CredentialRecord, ...]:
    operator_key = env.get(OPERATOR_KEY_VARIABLE)
    admin_key = env.get(ADMIN_KEY_VARIABLE)
    if not operator_key and not admin_key:
        return ()
    if operator_key and not admin_key:
        return (
            _legacy_record(
                "legacy.single-key", "Legacy User", Role.ADMIN, operator_key
            ),
        )
    if admin_key and not operator_key:
        return (
            _legacy_record(
                "legacy.admin", "Legacy Administrator", Role.ADMIN, admin_key
            ),
        )
    if operator_key == admin_key:
        return (
            _legacy_record(
                "legacy.single-key", "Legacy User", Role.ADMIN, operator_key
            ),
        )
    assert operator_key is not None and admin_key is not None
    return (
        _legacy_record(
            "legacy.operator", "Legacy Operator", Role.OPERATOR, operator_key
        ),
        _legacy_record("legacy.admin", "Legacy Administrator", Role.ADMIN, admin_key),
    )


def _legacy_record(
    subject: str,
    display_name: str,
    role: Role,
    api_key: str,
) -> CredentialRecord:
    return CredentialRecord(
        principal=Principal(subject=subject, display_name=display_name, role=role),
        api_key=api_key,
    )


def load_identity_service(env: Mapping[str, str]) -> IdentityService | None:
    """Build the identity service from named JSON or compatible legacy keys."""
    if not isinstance(env, Mapping):
        raise TypeError("env must be a string mapping")
    named_json = env.get(PRINCIPALS_VARIABLE)
    records = parse_principal_config(named_json) if named_json else _legacy_records(env)
    if not records:
        return None
    return IdentityService(
        CredentialRegistry(records),
        SessionStore(ttl=_session_ttl(env)),
    )
