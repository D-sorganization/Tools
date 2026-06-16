"""Endpoint inventory regression tests for the calculation backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_list_endpoints_repairs_request_app_not_module_global() -> None:
    from calc_backend.app import list_endpoints

    isolated_app = FastAPI()
    isolated_app.add_api_route(
        "/api/calc/endpoints",
        list_endpoints,
        methods=["GET"],
    )
    isolated_client = TestClient(isolated_app)

    response = isolated_client.get("/api/calc/endpoints")

    assert response.status_code == 200
    assert "POST /api/calc/flare" in response.json()["calculators"]
    assert any(
        getattr(route, "path", None) == "/api/calc/flare"
        for route in isolated_app.routes
    )


def test_calculator_route_signatures_apply_router_prefix() -> None:
    from calc_backend.app import _calculator_route_signatures

    class PrefixlessRoute:
        path = ""
        methods = {"POST"}

    assert _calculator_route_signatures(
        [PrefixlessRoute()],
        prefix="/api/calc/flare",
    ) == {("POST", "/api/calc/flare")}

    class PrefixedRoute:
        path = "/api/calc/flare"
        methods = {"POST"}

    assert _calculator_route_signatures(
        [PrefixedRoute()],
        prefix="/api/calc/flare",
    ) == {("POST", "/api/calc/flare")}
