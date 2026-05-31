"""CORS configuration for the P1AM backend.

This control-system-adjacent API must *fail closed* to a known set of UI
origins rather than allowing the wildcard ``*``. Wildcard origins combined
with credentials are unsafe and rejected by browsers, so this module
guarantees the two are never emitted together.

Configuration is environment-driven:

- ``P1AM_CORS_ORIGINS``: comma-separated allowlist of explicit origins.
- ``P1AM_CORS_ALLOW_CREDENTIALS``: ``"true"``/``"1"`` to allow credentials.
- ``P1AM_ENV``: when ``"production"``, an explicit allowlist is required.

The defaults target local development only (the Vite dashboard origin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

__all__ = [
    "CorsSettings",
    "DEFAULT_DEV_ORIGINS",
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
