"""Server-side authentication/authorization for the P1AM control backend.

The desktop HMI has a client-side role system, but the backend itself must not
trust the client. This module provides FastAPI dependencies that gate every
state-mutating endpoint behind a server-verified credential.

Two credential tiers
--------------------
- **Operator key** (``P1AM_API_KEY``): required for all mutating endpoints and
  the live WebSocket stream.
- **Admin key** (``P1AM_ADMIN_API_KEY``): required *in addition* for destructive
  / elevated operations (clearing the E-stop, tag writes, PID tuning, gas/setpoint
  changes, and project import which wipes the plant DB). If ``P1AM_ADMIN_API_KEY``
  is not set, the operator key is accepted for admin operations too (single-key
  deployments) — but a present admin key is enforced strictly.

The tiers are nested: **admin ⊇ operator**. A configured admin key is always a
valid operator credential, including when it is the *only* key configured. The
reverse never holds. Keying the operator tier off ``P1AM_API_KEY`` alone used to
brick an admin-only deployment — ``/api/stream`` closed every connection with
1008 and alarm acknowledgement 503'd, leaving full hardware control behind a
dead display (issue #4041). :func:`log_auth_configuration` reports the resolved
tiers at startup so a half-configured deployment is visible at boot rather than
at the operator's first control action.

Credentials are supplied via the ``X-API-Key`` request header (or, for the
WebSocket, an ``api_key`` query parameter or the first text frame) and compared
in constant time.

Fail-closed behavior
--------------------
If no ``P1AM_API_KEY`` is configured, requests are rejected with HTTP 503 unless
``P1AM_DEV_NO_AUTH=1`` is set to explicitly opt out (bench/dev use only). The
opt-out is logged loudly. E-stop *activation* (``POST /api/estop``) is
intentionally left unauthenticated so a panic stop is always reachable; that
choice is documented at the call site.
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from settings import read_auth_required

__all__ = [
    "CREDENTIAL_HEADER_NAME",
    "AuthConfiguration",
    "admin_key_configured",
    "is_dev_no_auth",
    "log_auth_configuration",
    "require_admin_key",
    "require_api_key",
    "require_read_auth",
    "resolve_auth_config",
    "verify_admin_key",
    "verify_operator_key",
]

logger = logging.getLogger("dcs_backend.auth")

CREDENTIAL_HEADER_NAME = "X-API-" + "Key"  # pragma: allowlist secret
_TRUTHY = {"1", "true", "yes", "on"}

# auto_error=False so we can return our own 401/503 with consistent messaging.
_api_key_header = APIKeyHeader(name=CREDENTIAL_HEADER_NAME, auto_error=False)
# A second scheme instance for the opt-in read gate, so a missing header is a
# no-op when the gate is off instead of FastAPI rejecting it up front.
_read_api_key_header = APIKeyHeader(name=CREDENTIAL_HEADER_NAME, auto_error=False)


def _dev_no_auth() -> bool:
    return os.environ.get("P1AM_DEV_NO_AUTH", "").strip().lower() in _TRUTHY


def _operator_key() -> str | None:
    key = os.environ.get("P1AM_API_KEY")
    return key if key else None


def _admin_key() -> str | None:
    key = os.environ.get("P1AM_ADMIN_API_KEY")
    return key if key else None


def _operator_tier_keys() -> tuple[str, ...]:
    """Every credential that satisfies the *operator* tier.

    The admin key is included unconditionally: the tiers are nested, so an
    admin-only deployment must still be able to open the telemetry stream and
    acknowledge alarms (#4041).
    """
    return tuple(key for key in (_operator_key(), _admin_key()) if key)


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def _matches_any(provided: str | None, accepted: tuple[str, ...]) -> bool:
    """Constant-time membership test that does not short-circuit on the match."""
    if not provided:
        return False
    found = False
    for candidate in accepted:
        if _constant_time_eq(provided, candidate):
            found = True
    return found


@dataclass(frozen=True)
class AuthConfiguration:
    """The credential configuration resolved from the process environment.

    Invariant: this object never carries a credential value, only whether one
    is present — so it is always safe to log.
    """

    dev_no_auth: bool
    operator_key_configured: bool
    admin_key_configured: bool
    read_auth_required: bool

    @property
    def authenticated(self) -> bool:
        """True when at least one credential gates the API."""
        return self.operator_key_configured or self.admin_key_configured

    @property
    def admin_only(self) -> bool:
        """True for the half-configured shape that motivated #4041."""
        return self.admin_key_configured and not self.operator_key_configured

    def describe(self) -> str:
        """A one-line, credential-free summary for the boot log."""
        if self.dev_no_auth:
            return "P1AM_DEV_NO_AUTH=1 (AUTHENTICATION DISABLED — bench use only)"
        if not self.authenticated:
            return "no credential configured (every gated route will answer 503)"
        operator = "set" if self.operator_key_configured else "unset"
        admin = "set" if self.admin_key_configured else "unset"
        tier = (
            "admin key also serves the operator tier"
            if self.admin_only
            else (
                "operator and admin tiers are distinct"
                if self.operator_key_configured and self.admin_key_configured
                else "single-key deployment (operator key serves both tiers)"
            )
        )
        read = "required" if self.read_auth_required else "PUBLIC"
        return (
            f"P1AM_API_KEY={operator}, P1AM_ADMIN_API_KEY={admin} "
            f"({tier}); read surface {read}"
        )


def resolve_auth_config() -> AuthConfiguration:
    """Resolve the current credential configuration from the environment."""
    return AuthConfiguration(
        dev_no_auth=_dev_no_auth(),
        operator_key_configured=_operator_key() is not None,
        admin_key_configured=_admin_key() is not None,
        read_auth_required=read_auth_required(),
    )


def log_auth_configuration(log: logging.Logger | None = None) -> AuthConfiguration:
    """Log the resolved authentication posture and return it.

    Called once at import/boot so a half-configured or bypassed deployment is
    visible in ``journalctl`` at startup rather than at the first control
    action (#4041). Escalates to WARNING for the two dangerous shapes: the dev
    bypass, and no credential at all.
    """
    resolved = resolve_auth_config()
    target = log or logger
    message = "P1AM auth configuration: %s"
    if resolved.dev_no_auth or not resolved.authenticated:
        target.warning(message, resolved.describe())
    else:
        target.info(message, resolved.describe())
    return resolved


def verify_operator_key(provided: str | None) -> bool:
    """Return True if ``provided`` matches a configured operator/admin key.

    In dev-no-auth mode this always returns True. Used by the WebSocket path,
    which cannot raise ``HTTPException`` the same way.
    """
    if _dev_no_auth():
        return True
    return _matches_any(provided, _operator_tier_keys())


def verify_admin_key(provided: str | None) -> bool:
    """Return True if ``provided`` is the configured admin credential.

    Unlike :func:`require_admin_key` this never raises and never falls back to
    the operator key — it answers the narrow question "is this *the* admin
    key?", which the audit trail needs to classify an actor.
    """
    admin = _admin_key()
    return admin is not None and _matches_any(provided, (admin,))


def admin_key_configured() -> bool:
    """True when a distinct admin credential exists (not a single-key deploy)."""
    return _admin_key() is not None


def is_dev_no_auth() -> bool:
    """True when the bench opt-out ``P1AM_DEV_NO_AUTH`` is enabled."""
    return _dev_no_auth()


def require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """FastAPI dependency enforcing a valid operator (or admin) API key.

    Raises:
        HTTPException: 503 if no key is configured (and dev opt-out is off),
            401 if the supplied key is missing or invalid.
    """
    if _dev_no_auth():
        logger.warning(
            "P1AM_DEV_NO_AUTH is enabled: API authentication is DISABLED. "
            "Do not use this in production."
        )
        return
    if not _operator_tier_keys():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server credential not configured. Set P1AM_API_KEY (and "
                "optionally P1AM_ADMIN_API_KEY), or set P1AM_DEV_NO_AUTH=1 for "
                "bench use."
            ),
        )
    if not verify_operator_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": CREDENTIAL_HEADER_NAME},
        )


def require_admin_key(
    api_key: str | None = Security(_api_key_header),
) -> None:
    """FastAPI dependency enforcing the elevated admin API key.

    If ``P1AM_ADMIN_API_KEY`` is set, only that key is accepted. Otherwise the
    operator key is accepted (single-key deployment). Fails closed identically
    to :func:`require_api_key` when nothing is configured.
    """
    if _dev_no_auth():
        logger.warning("P1AM_DEV_NO_AUTH is enabled: admin authentication is DISABLED.")
        return

    admin = _admin_key()
    operator = _operator_key()

    if admin is None and operator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server credential not configured. Set P1AM_API_KEY/"
                "P1AM_ADMIN_API_KEY, or set P1AM_DEV_NO_AUTH=1 for bench use."
            ),
        )

    if admin is not None:
        if api_key and _constant_time_eq(api_key, admin):
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires the admin API key.",
            headers={"WWW-Authenticate": CREDENTIAL_HEADER_NAME},
        )

    # No admin key configured: accept the operator key.
    if operator is not None and api_key and _constant_time_eq(api_key, operator):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": CREDENTIAL_HEADER_NAME},
    )


def require_read_auth(
    api_key: str | None = Security(_read_api_key_header),
) -> None:
    """Gate for the historian / configuration *read* surface.

    Enforces :func:`require_api_key` whenever read auth is required — which now
    defaults to on (issue #4037). ``GET /api/routing`` alone discloses the full
    register map, every scale factor and every interlock trip limit, which is
    the blueprint an attacker needs to drive the plant; ``P1AM_DEV_NO_AUTH``
    still bypasses it for bench use.

    The resolution is re-read per request (see :func:`settings.read_auth_required`)
    so the gate can be toggled without a process restart.

    This dependency lives here rather than in ``main`` so the power-supply and
    temperature routers can attach it without importing the app module.
    """
    if not read_auth_required():
        return
    require_api_key(api_key)
