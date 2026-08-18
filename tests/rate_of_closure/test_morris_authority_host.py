"""Authenticated loopback app contract for the Morris authority host."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rate_of_closure.application.morris.contracts import (
    MORRIS_JOB_SCHEMA_ID,
    MORRIS_REQUEST_SCHEMA_ID,
)
from rate_of_closure.application.morris.host import (
    API_PREFIX,
    CAPABILITY_PATH,
    create_morris_authority_app,
)
from rate_of_closure.application.morris.router import MorrisJobRegistry
from rate_of_closure.application.morris.service import RateMorrisService


class _CountingRegistry(MorrisJobRegistry):
    def __init__(self) -> None:
        self.close_count = 0
        super().__init__(RateMorrisService())

    def close(self) -> None:
        self.close_count += 1
        super().close()


def test_host_requires_exact_bearer_and_owns_registry_lifespan() -> None:
    registry = _CountingRegistry()
    with TestClient(create_morris_authority_app("secret-token", registry)) as client:
        for headers in (
            {},
            {"Authorization": "Bearer wrong"},
            {"Authorization": "Basic secret-token"},
        ):
            response = client.get(CAPABILITY_PATH, headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
        capability = client.get(
            CAPABILITY_PATH, headers={"Authorization": "Bearer secret-token"}
        )
        assert capability.status_code == 200
        assert capability.json() == {
            "schema_id": "rate-of-closure/morris-authority-capability",
            "schema_version": 1,
            "available": True,
            "api_prefix": API_PREFIX,
            "request_schema_id": MORRIS_REQUEST_SCHEMA_ID,
            "job_schema_id": MORRIS_JOB_SCHEMA_ID,
        }
        assert capability.headers["cache-control"] == "no-store"
        assert capability.headers["x-content-type-options"] == "nosniff"
        assert registry.close_count == 0
    assert registry.close_count == 1


def test_shutdown_control_is_authenticated_by_global_middleware() -> None:
    registry = _CountingRegistry()
    shutdowns: list[str] = []
    app = create_morris_authority_app(
        "secret-token", registry, lambda: shutdowns.append("requested")
    )
    with TestClient(app) as client:
        unauthorized = client.post("/_control/shutdown")
        assert unauthorized.status_code == 401
        assert shutdowns == []
        authorized = client.post(
            "/_control/shutdown",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert authorized.status_code == 200
        assert authorized.json() == {"status": "stopping"}
        assert shutdowns == ["requested"]
    assert registry.close_count == 1


def test_host_does_not_enable_cors() -> None:
    registry = _CountingRegistry()
    app = create_morris_authority_app("secret-token", registry)
    with TestClient(app):
        assert all(
            "cors" not in type(item).__name__.lower() for item in app.user_middleware
        )
    assert registry.close_count == 1


def test_authenticated_error_responses_retain_security_headers() -> None:
    registry = _CountingRegistry()
    app = create_morris_authority_app("secret-token", registry)

    @app.get("/_test/validated")
    async def validated(required: int) -> dict[str, int]:
        return {"required": required}

    @app.get("/_test/explode")
    async def explode() -> None:
        raise RuntimeError("private detail")

    headers = {"Authorization": "Bearer secret-token"}
    with TestClient(app, raise_server_exceptions=False) as client:
        responses = (
            client.get("/not-found", headers=headers),
            client.get("/_test/validated", headers=headers),
            client.get("/_test/explode", headers=headers),
        )
        assert [response.status_code for response in responses] == [404, 422, 500]
        for response in responses:
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
        assert responses[2].json() == {"error": "internal server error"}
        assert "private detail" not in responses[2].text
    assert registry.close_count == 1
