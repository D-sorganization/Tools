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

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger("dcs_backend.auth")

API_KEY_HEADER_NAME = "X-API-Key"  # pragma: allowlist secret
_TRUTHY = {"1", "true", "yes", "on"}

# auto_error=False so we can return our own 401/503 with consistent messaging.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _dev_no_auth() -> bool:
    return os.environ.get("P1AM_DEV_NO_AUTH", "").strip().lower() in _TRUTHY


def _operator_key() -> str | None:
    key = os.environ.get("P1AM_API_KEY")
    return key if key else None


def _admin_key() -> str | None:
    key = os.environ.get("P1AM_ADMIN_API_KEY")
    return key if key else None


def _constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def verify_operator_key(provided: str | None) -> bool:
    """Return True if ``provided`` matches the configured operator/admin key.

    In dev-no-auth mode this always returns True. Used by the WebSocket path,
    which cannot raise ``HTTPException`` the same way.
    """
    if _dev_no_auth():
        return True
    operator = _operator_key()
    if operator is None:
        return False
    if provided and _constant_time_eq(provided, operator):
        return True
    admin = _admin_key()
    return bool(provided and admin and _constant_time_eq(provided, admin))


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
    if _operator_key() is None:
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
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
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
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )

    # No admin key configured: accept the operator key.
    if operator is not None and api_key and _constant_time_eq(api_key, operator):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
        headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
    )
