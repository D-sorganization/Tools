"""Authenticated FastAPI application for the loopback model authority."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .capability import DEFAULT_UNAVAILABLE_CAPABILITY, AuthorityCapability

CAPABILITY_PATH = "/api/rate-of-closure/v1/capabilities"
_BEARER = HTTPBearer(auto_error=False)


def _require_token(token: str) -> str:
    """Validate the injected ephemeral authority token."""
    if not token or token != token.strip():
        raise ValueError("authority token must be nonempty and trimmed")
    return token


def create_authority_app(
    *,
    token: str,
    capability: AuthorityCapability = DEFAULT_UNAVAILABLE_CAPABILITY,
) -> FastAPI:
    """Create an authenticated, non-cacheable loopback authority application."""
    expected_token = _require_token(token)
    app = FastAPI(
        title="Rate of Closure local model authority",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authorize(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_BEARER),
        ],
    ) -> None:
        """Require the exact ephemeral bearer token without leaking comparisons."""
        supplied = credentials.credentials if credentials is not None else ""
        scheme = credentials.scheme if credentials is not None else ""
        valid = scheme.lower() == "bearer" and secrets.compare_digest(
            supplied, expected_token
        )
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Local authority authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get(CAPABILITY_PATH, dependencies=[Depends(authorize)])
    def read_capability(response: Response) -> dict[str, object]:
        """Return the immutable capability state with caching disabled."""
        response.headers["Cache-Control"] = "no-store"
        return dict(capability.to_wire())

    return app
