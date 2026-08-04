"""API tests for named SCADA session issuance and role enforcement."""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from identity import (  # noqa: E402
    CredentialRegistry,
    Role,
    SessionStore,
    parse_principal_config,
)
from identity_router import (  # noqa: E402
    IdentityService,
    create_identity_router,
    require_role,
)

_OPERATOR_KEY = "operator-test-secret"  # pragma: allowlist secret
_ENGINEER_KEY = "engineer-test-secret"  # pragma: allowlist secret


def _service() -> IdentityService:
    records = parse_principal_config(
        '[{"subject":"operator.one","display_name":"Operator One",'
        '"role":"operator","api_key":"operator-test-secret"},'  # noqa: E501  # pragma: allowlist secret
        '{"subject":"engineer.one","display_name":"Engineer One",'
        '"role":"engineer","api_key":"engineer-test-secret"}]'  # noqa: E501  # pragma: allowlist secret
    )
    return IdentityService(
        CredentialRegistry(records),
        SessionStore(ttl=timedelta(minutes=30)),
    )


def _client() -> TestClient:
    service = _service()
    app = FastAPI()
    app.include_router(create_identity_router(service))

    @app.post("/operator", dependencies=[Depends(require_role(service, Role.OPERATOR))])
    async def operator_action() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/engineer", dependencies=[Depends(require_role(service, Role.ENGINEER))])
    async def engineer_action() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def _login(client: TestClient, key: str) -> str:
    response = client.post("/api/auth/session", headers={"X-API-Key": key})
    assert response.status_code == 201
    return str(response.json()["token"])


def test_login_returns_named_principal_and_opaque_session() -> None:
    client = _client()

    response = client.post(
        "/api/auth/session",
        headers={"X-API-Key": _OPERATOR_KEY},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["principal"] == {
        "subject": "operator.one",
        "display_name": "Operator One",
        "role": "operator",
    }
    assert len(payload["token"]) >= 32
    assert payload["expires_at"].endswith("Z")


def test_login_rejects_invalid_credential_without_echoing_it() -> None:
    client = _client()
    invalid = "invalid-test-secret"  # pragma: allowlist secret

    response = client.post("/api/auth/session", headers={"X-API-Key": invalid})

    assert response.status_code == 401
    assert invalid not in response.text


def test_me_resolves_bearer_session() -> None:
    client = _client()
    token = _login(client, _OPERATOR_KEY)

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["subject"] == "operator.one"


def test_logout_revokes_session() -> None:
    client = _client()
    token = _login(client, _OPERATOR_KEY)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.delete("/api/auth/session", headers=headers).status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_role_dependency_enforces_operator_and_engineer_boundaries() -> None:
    client = _client()
    operator = _login(client, _OPERATOR_KEY)
    engineer = _login(client, _ENGINEER_KEY)

    assert (
        client.post(
            "/operator", headers={"Authorization": f"Bearer {operator}"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/engineer", headers={"Authorization": f"Bearer {operator}"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/engineer", headers={"Authorization": f"Bearer {engineer}"}
        ).status_code
        == 200
    )


def test_role_dependency_accepts_named_api_key_during_migration() -> None:
    client = _client()

    response = client.post("/operator", headers={"X-API-Key": _OPERATOR_KEY})

    assert response.status_code == 200


def test_router_resolves_service_provider_at_request_time() -> None:
    configured: IdentityService | None = None
    app = FastAPI()
    app.include_router(create_identity_router(lambda: configured))
    client = TestClient(app)

    assert client.post("/api/auth/session").status_code == 503
    configured = _service()
    assert (
        client.post(
            "/api/auth/session", headers={"X-API-Key": _OPERATOR_KEY}
        ).status_code
        == 201
    )
