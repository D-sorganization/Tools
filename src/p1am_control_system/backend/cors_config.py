"""CORS configuration and the state-change request guard for the P1AM backend.

This control-system-adjacent API must *fail closed* to a known set of UI
origins rather than allowing the wildcard ``*``. Wildcard origins combined
with credentials are unsafe and rejected by browsers, so this module
guarantees the two are never emitted together.

Configuration is environment-driven:

- ``P1AM_CORS_ORIGINS``: comma-separated allowlist of explicit origins.
- ``P1AM_CORS_ALLOW_CREDENTIALS``: ``"true"``/``"1"`` to allow credentials.
- ``P1AM_ENV``: when ``"production"``, an explicit allowlist is required.

The defaults target local development only (the Vite dashboard origin).

Beyond CORS, :class:`RequestGuardMiddleware` enforces the same allowlist on the
*request* side and forces every state-changing route into a preflight. CORS
alone does not protect this API: a page can issue a "simple" cross-site
``POST`` and the browser will hide only the response, not the effect — and
several control routes here take no request body at all, which is exactly the
shape that qualifies as simple. See :func:`evaluate_state_change` (issue #4037).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CSRF_HEADER_NAME",
    "CSRF_HEADER_VALUE",
    "CorsSettings",
    "DEFAULT_DEV_ORIGINS",
    "PREFLIGHT_EXEMPT_PATHS",
    "STATE_CHANGING_METHODS",
    "RequestGuardMiddleware",
    "evaluate_state_change",
    "resolve_cors_settings",
]

logger = logging.getLogger("dcs_backend.cors")

# Local-development origins only. Production origins must be supplied
# explicitly via ``P1AM_CORS_ORIGINS``.
DEFAULT_DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:3002",
    "http://127.0.0.1:3002",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

_TRUTHY = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CorsSettings:
    """Resolved CORS policy for the FastAPI ``CORSMiddleware``.

    Invariant: ``"*"`` is never present in :attr:`allow_origins` while
    :attr:`allow_credentials` is ``True``.
    """

    allow_origins: tuple[str, ...] = field(default_factory=tuple)
    allow_credentials: bool = False

    def __post_init__(self) -> None:
        if self.allow_credentials and "*" in self.allow_origins:
            raise ValueError(
                "CORS misconfiguration: wildcard origin '*' cannot be "
                "combined with allow_credentials=True."
            )


def _parse_origins(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated origin list, stripping blanks/duplicates."""
    if not raw:
        return ()
    seen: list[str] = []
    for part in raw.split(","):
        origin = part.strip()
        if origin and origin not in seen:
            seen.append(origin)
    return tuple(seen)


def resolve_cors_settings(
    env: dict[str, str] | None = None,
) -> CorsSettings:
    """Resolve the CORS policy from the environment.

    Precondition: ``env`` is a mapping of environment variables (defaults
    to :data:`os.environ`).

    Behavior:
    - Explicit ``P1AM_CORS_ORIGINS`` always wins. A bare ``*`` is rejected
      whenever credentials are enabled.
    - In production (``P1AM_ENV=production``) with no explicit allowlist,
      raises :class:`RuntimeError` (fail closed).
    - Otherwise falls back to :data:`DEFAULT_DEV_ORIGINS` with a warning.
    """
    if env is None:
        import os

        env = dict(os.environ)

    is_production = env.get("P1AM_ENV", "").strip().lower() == "production"
    allow_credentials = (
        env.get("P1AM_CORS_ALLOW_CREDENTIALS", "").strip().lower() in _TRUTHY
    )
    origins = _parse_origins(env.get("P1AM_CORS_ORIGINS"))

    if origins:
        if "*" in origins and allow_credentials:
            raise ValueError(
                "CORS misconfiguration: '*' in P1AM_CORS_ORIGINS cannot be "
                "combined with P1AM_CORS_ALLOW_CREDENTIALS=true."
            )
        return CorsSettings(allow_origins=origins, allow_credentials=allow_credentials)

    if is_production:
        raise RuntimeError(
            "P1AM_ENV=production requires an explicit CORS allowlist via "
            "P1AM_CORS_ORIGINS; refusing to start with development defaults."
        )

    logger.warning(
        "No P1AM_CORS_ORIGINS configured; falling back to local development "
        "origins %s. Set P1AM_CORS_ORIGINS for any non-local deployment.",
        ", ".join(DEFAULT_DEV_ORIGINS),
    )
    return CorsSettings(
        allow_origins=DEFAULT_DEV_ORIGINS, allow_credentials=allow_credentials
    )


# --------------------------------------------------------------------------- #
# State-change request guard (CSRF / cross-origin)                             #
# --------------------------------------------------------------------------- #

#: HTTP methods that can change plant state.
STATE_CHANGING_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Custom header the HMI sends on every request. Being *custom* is the point:
#: a CORS-simple request cannot set it, so its presence proves a preflight
#: happened (which the origin allowlist then answers).
CSRF_HEADER_NAME = "X-Requested-With"
CSRF_HEADER_VALUE = "p1am-hmi"

#: The credential header is likewise custom, so it forces a preflight too.
_CREDENTIAL_HEADER = "x-api-" + "key"  # pragma: allowlist secret

#: Content types a CORS-simple request is allowed to use. Anything outside this
#: set forces a preflight — note FastAPI will happily parse a JSON body sent as
#: ``text/plain``, so the check is on the declared type, not the body.
_NON_SIMPLE_CONTENT_TYPES = ("application/json",)

#: SAFETY: E-stop *activation* must stay reachable from a bare shell script
#: (``curl -X POST http://host/api/estop``) with no headers at all, so it is
#: exempt from preflight forcing. It is NOT exempt from the origin check, so a
#: malicious page still cannot trip the plant.
PREFLIGHT_EXEMPT_PATHS: tuple[str, ...] = ("/api/estop",)


def evaluate_state_change(
    *,
    method: str,
    path: str,
    headers: Mapping[str, str],
    allowed_origins: Sequence[str],
    preflight_exempt_paths: Sequence[str] = PREFLIGHT_EXEMPT_PATHS,
) -> str | None:
    """Decide whether a request may change plant state.

    Args:
        method: HTTP method, upper-case.
        path: Request path, without query string.
        headers: Request headers keyed by **lower-case** name.
        allowed_origins: The CORS allowlist. An empty allowlist trusts no
            browser origin (fail closed).
        preflight_exempt_paths: Paths exempt from preflight forcing only.

    Returns:
        ``None`` when the request is allowed, otherwise a short human-readable
        reason for the refusal.

    Preconditions:
        ``headers`` keys are lower-cased (as ASGI delivers them).
    """
    if method.upper() not in STATE_CHANGING_METHODS:
        return None

    origin = headers.get("origin")
    if origin and origin not in tuple(allowed_origins):
        return (
            f"Origin {origin!r} is not in the configured CORS allowlist; "
            "state-changing requests from other origins are refused."
        )

    if path in tuple(preflight_exempt_paths):
        return None

    if CSRF_HEADER_NAME.lower() in headers or _CREDENTIAL_HEADER in headers:
        return None

    content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type in _NON_SIMPLE_CONTENT_TYPES:
        return None

    return (
        "State-changing requests must send a non-simple request so the browser "
        f"performs a CORS preflight: set the {CSRF_HEADER_NAME} header or use "
        "Content-Type: application/json."
    )


class RequestGuardMiddleware:
    """Pure-ASGI middleware applying :func:`evaluate_state_change`.

    Implemented at the ASGI layer (rather than as a route dependency) so a
    newly added endpoint is protected by default — the failure mode being
    guarded against is precisely the route someone forgot to annotate.

    Mount it *inside* ``CORSMiddleware`` so refusals still carry CORS headers
    and OPTIONS preflights are answered by the CORS layer, never by this guard.
    """

    def __init__(
        self,
        app: Any,
        *,
        allowed_origins: Sequence[str],
        preflight_exempt_paths: Sequence[str] = PREFLIGHT_EXEMPT_PATHS,
    ) -> None:
        self.app = app
        self.allowed_origins = tuple(allowed_origins)
        self.preflight_exempt_paths = tuple(preflight_exempt_paths)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        reason = evaluate_state_change(
            method=scope.get("method", "GET"),
            path=scope.get("path", ""),
            headers=headers,
            allowed_origins=self.allowed_origins,
            preflight_exempt_paths=self.preflight_exempt_paths,
        )
        if reason is None:
            await self.app(scope, receive, send)
            return

        logger.warning(
            "Refused %s %s from %s: %s",
            scope.get("method"),
            scope.get("path"),
            headers.get("origin", "no-origin"),
            reason,
        )
        body = json.dumps({"detail": reason}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
