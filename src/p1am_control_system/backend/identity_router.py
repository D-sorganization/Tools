"""FastAPI session surface and reusable named-role dependencies."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, TypeAlias, cast

from fastapi import APIRouter, Depends, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from identity import CredentialRegistry, IssuedSession, Principal, Role, SessionStore
from pydantic import BaseModel, ConfigDict

API_KEY_HEADER_NAME = "X-API-" + "Key"  # pragma: allowlist secret
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
_bearer = HTTPBearer(auto_error=False)

ApiKey = Annotated[str | None, Security(_api_key_header)]
BearerCredential = Annotated[
    HTTPAuthorizationCredentials | None,
    Security(_bearer),
]
IdentityServiceProvider = Callable[[], "IdentityService | None"]
IdentityServiceSource: TypeAlias = "IdentityService | IdentityServiceProvider"


class PrincipalResponse(BaseModel):
    """Public identity metadata returned to an authenticated client."""

    model_config = ConfigDict(from_attributes=True)

    subject: str
    display_name: str
    role: Role


class SessionResponse(BaseModel):
    """New opaque session and its expiry/identity metadata."""

    token: str
    expires_at: datetime
    principal: PrincipalResponse


class IdentityService:
    """Coordinate credential authentication and opaque session lifecycle."""

    def __init__(
        self,
        registry: CredentialRegistry,
        sessions: SessionStore,
    ) -> None:
        if not isinstance(registry, CredentialRegistry):
            raise TypeError("registry must be a CredentialRegistry")
        if not isinstance(sessions, SessionStore):
            raise TypeError("sessions must be a SessionStore")
        self._registry = registry
        self._sessions = sessions

    def login(self, api_key: str | None) -> IssuedSession | None:
        """Authenticate one credential and issue a session on success."""
        principal = self._registry.authenticate(api_key)
        return self._sessions.create(principal) if principal is not None else None

    def resolve(
        self,
        api_key: str | None,
        bearer: HTTPAuthorizationCredentials | None,
    ) -> Principal | None:
        """Resolve either a named API key or a short-lived bearer session."""
        if bearer is not None and bearer.scheme.lower() == "bearer":
            return self._sessions.resolve(bearer.credentials)
        return self._registry.authenticate(api_key)

    def revoke(self, bearer: HTTPAuthorizationCredentials | None) -> bool:
        """Revoke a bearer session when it is present and validly shaped."""
        if bearer is None or bearer.scheme.lower() != "bearer":
            return False
        return cast(bool, self._sessions.revoke(bearer.credentials))


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid credential.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _configured_service(source: IdentityServiceSource) -> IdentityService:
    service = source() if callable(source) else source
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server identity service is not configured.",
        )
    if not isinstance(service, IdentityService):
        raise TypeError("identity service provider returned an invalid value")
    return service


def require_role(
    service: IdentityServiceSource,
    required_role: Role,
) -> Callable[..., Principal]:
    """Build a dependency enforcing a named principal and minimum role."""
    if not isinstance(service, IdentityService) and not callable(service):
        raise TypeError("service must be an IdentityService or provider")
    if not isinstance(required_role, Role):
        raise TypeError("required_role must be a Role")

    def dependency(
        api_key: ApiKey = None, bearer: BearerCredential = None
    ) -> Principal:
        principal = _configured_service(service).resolve(api_key, bearer)
        if principal is None:
            raise _unauthorized()
        if not principal.allows(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires the {required_role.value} role.",
            )
        return principal

    return dependency


def _session_response(issued: IssuedSession) -> SessionResponse:
    return SessionResponse(
        token=issued.token,
        expires_at=issued.expires_at,
        principal=PrincipalResponse.model_validate(issued.principal),
    )


def create_identity_router(service: IdentityServiceSource) -> APIRouter:
    """Create the named-session API router for one identity service."""
    if not isinstance(service, IdentityService) and not callable(service):
        raise TypeError("service must be an IdentityService or provider")
    router = APIRouter(prefix="/api/auth", tags=["identity"])
    authenticated = require_role(service, Role.VIEWER)

    @router.post("/session", status_code=status.HTTP_201_CREATED)
    async def create_session(api_key: ApiKey = None) -> SessionResponse:
        issued = _configured_service(service).login(api_key)
        if issued is None:
            raise _unauthorized()
        return _session_response(issued)

    @router.get("/me")
    async def get_principal(
        principal: Principal = Depends(authenticated),  # noqa: B008
    ) -> PrincipalResponse:
        # Annotated local: see the typing convention note in SPEC.md — CI runs
        # mypy from the repo root, where flat intra-package imports become Any.
        response: PrincipalResponse = PrincipalResponse.model_validate(principal)
        return response

    @router.delete("/session", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_session(bearer: BearerCredential = None) -> Response:
        if not _configured_service(service).revoke(bearer):
            raise _unauthorized()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
