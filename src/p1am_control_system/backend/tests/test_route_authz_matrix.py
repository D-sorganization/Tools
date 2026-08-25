"""End-to-end authorization matrix over the REAL app — #4028.

``test_auth_config.py`` proves the dependency *functions* behave. Nothing proved
they were *wired* to the routes, and every functional suite sets
``P1AM_DEV_NO_AUTH=1`` at module import — so deleting a
``dependencies=[Depends(require_admin_key)]`` from a hardware-mutating route
kept the whole backend suite green.

This suite boots the real ``main.app`` with credentials SET and
``P1AM_DEV_NO_AUTH`` CLEARED, and drives an explicit
``(method, path, required_tier)`` table:

- every mutating route must reject an unauthenticated caller,
- every admin route must reject an operator-only credential,
- every read route must reject an unauthenticated caller (read auth now
  defaults on),
- and — the real risk — **a route that is not in the table fails the suite**,
  so a newly added endpoint cannot ship silently ungated.

Determinism (see #4061)
-----------------------
The existing functional suites mutate ``os.environ`` at *module import* time and
then rely on a module-level settings singleton that may already have been
resolved — so whether their configuration takes effect depends on collection
order and, under ``xdist``, on worker assignment. That order-dependence points
in the direction of *passing*, which for an authorization suite means reporting
green while proving nothing.

This suite avoids that entirely:

- every relevant variable is set **explicitly** per test via ``monkeypatch``
  (nothing is left to fall back on a cached singleton, and nothing leaks out);
- ``auth_config`` resolves credentials from ``os.environ`` on every request, so
  no module-level cache sits between the fixture and the assertion;
- :func:`test_suite_is_actually_authenticated` asserts the app really is in the
  configuration this file claims, turning any future coupling into a loud
  failure rather than a silent pass.

The auth dependencies are deliberately **not** replaced with
``dependency_overrides``: overriding them would decouple the suite from the very
wiring it exists to verify, and a deleted ``dependencies=[...]`` would again go
unnoticed.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth_config import CREDENTIAL_HEADER_NAME  # noqa: E402
from cors_config import CSRF_HEADER_NAME, CSRF_HEADER_VALUE  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from starlette.routing import Match  # noqa: E402

_OPERATOR_KEY = "matrix-operator-key"  # pragma: allowlist secret
_ADMIN_KEY = "matrix-admin-key"  # pragma: allowlist secret

PUBLIC = "public"
READ = "read"
OPERATOR = "operator"
ADMIN = "admin"

#: The authorization contract, one row per (method, path) the app exposes.
#:
#: ``PUBLIC``   - deliberately reachable with no credential.
#: ``READ``     - historian / configuration disclosure; operator key or better.
#: ``OPERATOR`` - state change an operator may perform.
#: ``ADMIN``    - hardware-mutating or destructive; admin key required.
ROUTE_TIERS: dict[tuple[str, str], str] = {
    # --- FastAPI/OpenAPI scaffolding -------------------------------------- #
    ("GET", "/"): PUBLIC,
    ("GET", "/openapi.json"): PUBLIC,
    ("GET", "/docs"): PUBLIC,
    ("GET", "/docs/oauth2-redirect"): PUBLIC,
    ("GET", "/redoc"): PUBLIC,
    # --- Panic stop --------------------------------------------------------#
    # SAFETY: E-stop *activation* stays unauthenticated on purpose so a panic
    # stop is always reachable. Clearing it is admin-gated. The cross-origin
    # guard (see cors_config) still blocks a malicious page from tripping it.
    ("POST", "/api/estop"): PUBLIC,
    # --- Read surface ------------------------------------------------------#
    ("GET", "/api/routing"): READ,
    ("GET", "/api/snapshot"): READ,
    ("GET", "/api/alarms/active"): READ,
    ("GET", "/api/events"): READ,
    ("GET", "/api/trends"): READ,
    ("GET", "/api/export"): READ,
    ("GET", "/api/capture/status"): READ,
    ("GET", "/api/capture/config"): READ,
    ("GET", "/api/performance"): READ,
    ("GET", "/api/alicats"): READ,
    ("GET", "/api/plant"): READ,
    ("GET", "/api/project/ladder-explorer"): READ,
    # --- Operator tier -----------------------------------------------------#
    ("POST", "/api/alarms/{tag_id}/acknowledge"): OPERATOR,
    ("POST", "/api/events"): OPERATOR,
    # --- Admin tier (hardware-mutating / destructive) ---------------------- #
    ("POST", "/api/routing"): ADMIN,
    ("POST", "/api/estop/clear"): ADMIN,
    ("PUT", "/api/capture/config"): ADMIN,
    ("PUT", "/api/performance"): ADMIN,
    ("POST", "/api/capture/clear"): ADMIN,
    ("POST", "/api/tags/{tag_id}"): ADMIN,
    ("POST", "/api/pid/{pid_index}/tuning/start"): ADMIN,
    ("POST", "/api/pid/{pid_index}/tuning/step"): ADMIN,
    ("POST", "/api/pid/{pid_index}/tuning/stop"): ADMIN,
    ("POST", "/api/mpc/simulate"): ADMIN,
    ("POST", "/api/alicats/{device_id}/setpoint"): ADMIN,
    ("POST", "/api/alicats/{device_id}/gas"): ADMIN,
    ("POST", "/api/project/import"): ADMIN,
    # --- Power supply and heater --------------------------------------------#
    # main.py mounts create_power_supply_router / create_temperature_router
    # UNCONDITIONALLY, so these rows are never stale and must never be dropped:
    # they are the DC supply's and the heater's own command surface, the most
    # safety-relevant endpoints in the app. Deleting them let
    # test_every_route_is_classified pass while the "a new endpoint cannot ship
    # ungated" guarantee (issue #4028) silently stopped covering them.
    ("GET", "/api/power_supply/config"): READ,
    ("GET", "/api/power_supply/status"): READ,
    ("GET", "/api/temperature/config"): READ,
    ("GET", "/api/temperature/status"): READ,
    ("PUT", "/api/power_supply/config"): ADMIN,
    ("POST", "/api/power_supply/setpoint"): ADMIN,
    ("POST", "/api/power_supply/permissive"): ADMIN,
    ("POST", "/api/power_supply/acknowledge_trip"): ADMIN,
    ("PUT", "/api/temperature/config"): ADMIN,
    ("POST", "/api/temperature/setpoint"): ADMIN,
    ("POST", "/api/temperature/permissive"): ADMIN,
    ("POST", "/api/temperature/tc_type"): ADMIN,
    ("POST", "/api/temperature/burnout_mode"): ADMIN,
    ("POST", "/api/temperature/acknowledge_trip"): ADMIN,
}

#: Data Explorer rows, kept separate because main.py mounts that router only
#: when its numeric stack imports. They are added to ROUTE_TIERS only when the
#: app actually serves them, so the table stays exhaustive where numpy is
#: present and free of stale rows where it is not — the two directions
#: test_every_route_is_classified and test_table_has_no_stale_rows check.
_EXPLORER_TIERS: dict[tuple[str, str], str] = {
    ("GET", "/api/explorer/signals"): READ,
    ("POST", "/api/explorer/dataset"): READ,
    ("POST", "/api/explorer/statistics"): READ,
    ("POST", "/api/explorer/correlation"): READ,
    ("POST", "/api/explorer/spectrum"): READ,
    ("POST", "/api/explorer/trendline"): READ,
    ("POST", "/api/explorer/pca"): READ,
    ("POST", "/api/explorer/histogram"): READ,
    ("POST", "/api/explorer/export"): READ,
}

#: Concrete values substituted for path parameters when driving a request.
PATH_PARAM_SAMPLES: dict[str, str] = {
    "tag_id": "TAG_0",
    "pid_index": "0",
    "device_id": "MFC_1",
}

_DENIED = (401, 403)


def _concrete(path: str) -> str:
    for name, sample in PATH_PARAM_SAMPLES.items():
        path = path.replace("{" + name + "}", sample)
    return path


def _app_routes() -> set[tuple[str, str]]:
    """Every (method, path) the real app exposes, minus HEAD/OPTIONS.

    Traverses the route tree recursively, handling standard routes, Starlette Mounts,
    and modern FastAPI _IncludedRouter markers (extracting from original_router / router
    and include_context / prefix), and merges with app.openapi()["paths"] so served
    routes are accurately and comprehensively discovered on any FastAPI version.
    """
    found: set[tuple[str, str]] = set()

    def _walk_routes(routes: Iterable[Any], parent_prefix: str = "") -> None:
        for route in routes:
            raw_path = getattr(route, "path", None)
            if raw_path is None:
                raw_path = getattr(route, "path_format", None)

            orig_router = getattr(route, "original_router", None) or getattr(
                route, "router", None
            )
            inc_prefix = getattr(route, "prefix", "") or ""
            if not inc_prefix:
                ctx = getattr(route, "include_context", None)
                if isinstance(ctx, dict):
                    inc_prefix = str(ctx.get("prefix", ""))
                elif ctx is not None and hasattr(ctx, "prefix"):
                    inc_prefix = str(getattr(ctx, "prefix", "")) or ""

            comb = parent_prefix
            if inc_prefix:
                comb = (
                    f"{comb.rstrip('/')}/{inc_prefix.lstrip('/')}"
                    if comb
                    else inc_prefix
                )

            if orig_router is not None and hasattr(orig_router, "routes"):
                _walk_routes(orig_router.routes, comb)
            elif hasattr(route, "routes") and route.routes:
                mount_path = raw_path or ""
                mount_comb = (
                    f"{comb.rstrip('/')}/{mount_path.lstrip('/')}"
                    if (comb and mount_path)
                    else (comb or mount_path)
                )
                _walk_routes(route.routes, mount_comb)
            elif hasattr(route, "app") and hasattr(route.app, "routes"):
                mount_path = raw_path or ""
                mount_comb = (
                    f"{comb.rstrip('/')}/{mount_path.lstrip('/')}"
                    if (comb and mount_path)
                    else (comb or mount_path)
                )
                _walk_routes(route.app.routes, mount_comb)

            methods = getattr(route, "methods", None)
            if raw_path is not None and methods:
                full_path = (
                    f"{parent_prefix.rstrip('/')}/{raw_path.lstrip('/')}"
                    if parent_prefix
                    else raw_path
                )
                for method in methods:
                    m = str(method).upper()
                    if m not in {"HEAD", "OPTIONS"}:
                        found.add((m, full_path))

    if hasattr(app, "routes"):
        _walk_routes(app.routes)

    openapi_fn = getattr(app, "openapi", None)
    if callable(openapi_fn):
        try:
            schema = openapi_fn()
            paths = schema.get("paths", {})
            for path, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                for method in path_item:
                    m = str(method).upper()
                    if m in {"GET", "POST", "PUT", "DELETE", "PATCH", "TRACE"}:
                        found.add((m, path))
        except Exception:
            pass

    return found


# Register the Data Explorer tiers only when the app actually mounted that
# router. Done once at import, against the real app, so the contract is derived
# from what is served rather than from an assumption about the environment.
ROUTE_TIERS.update(
    {row: tier for row, tier in _EXPLORER_TIERS.items() if row in _app_routes()}
)


def _headers(key: str | None) -> dict[str, str]:
    """HMI-shaped headers: the CSRF marker keeps the origin guard out of the way
    so this suite measures *authorization* and nothing else."""
    headers = {CSRF_HEADER_NAME: CSRF_HEADER_VALUE}
    if key is not None:
        headers[CREDENTIAL_HEADER_NAME] = key
    return headers


@pytest.fixture(autouse=True)
def _authenticated_deployment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A REAL deployment: both keys configured, dev bypass off, read auth on.

    Every variable is set to an explicit value rather than left unset, so this
    fixture cannot inherit a sibling suite's leakage or a settings singleton
    that was resolved before it ran (#4061). ``monkeypatch`` unwinds all of it
    per test, so the fixture also cannot leak *outwards*.
    """
    monkeypatch.delenv("P1AM_DEV_NO_AUTH", raising=False)
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _OPERATOR_KEY)
    monkeypatch.setenv("P1AM_ADMIN_API_KEY", _ADMIN_KEY)
    yield


@pytest.fixture
def client() -> TestClient:
    """The real app WITHOUT its lifespan.

    The matrix only exercises the dependency/middleware layer; running the
    lifespan would start the Modbus connect loop, the 10 Hz poll loop and the
    Alicat manager against no hardware. ``raise_server_exceptions=False`` keeps
    a downstream 500 (unmapped tag, no PLC) from masking the status code the
    authorization layer produced.
    """
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------- #
# Self-check: the suite must be measuring what it claims to measure            #
# --------------------------------------------------------------------------- #


def test_suite_is_actually_authenticated() -> None:
    """Guard against a vacuous green run (#4061).

    If a sibling module's import-time ``P1AM_DEV_NO_AUTH=1`` were still in
    effect, every assertion below would pass for the wrong reason. Assert the
    posture directly instead of trusting collection order.
    """
    from auth_config import resolve_auth_config

    resolved = resolve_auth_config()
    assert resolved.dev_no_auth is False, (
        "P1AM_DEV_NO_AUTH is active — this suite would pass vacuously."
    )
    assert resolved.operator_key_configured is True
    assert resolved.admin_key_configured is True
    assert resolved.read_auth_required is True


def test_read_auth_is_secure_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The *default* must be secure, with the env var absent entirely (#4037)."""
    from settings import P1AMSettings

    monkeypatch.delenv("P1AM_REQUIRE_READ_AUTH", raising=False)
    assert P1AMSettings(_env_file=None).require_read_auth is True


# --------------------------------------------------------------------------- #
# Coverage & F16 Safety: unclassified or under-reported routes must fail       #
# --------------------------------------------------------------------------- #


def test_route_inventory_is_non_empty_and_complete() -> None:
    """Ensure route inventory and auth partitions are non-empty and well-formed.

    Protects against vacuous matrix passes and ensures safety-critical F16 advisory
    optimization / MPC simulation and tuning routes are explicitly registered.
    """
    routes = _app_routes()
    assert len(routes) >= 40, (
        f"Route inventory is unexpectedly sparse ({len(routes)} routes). "
        "Route introspection collector may be under-reporting."
    )
    assert len(ROUTE_TIERS) >= 40, "ROUTE_TIERS table must not be empty"
    assert len(_GATED) >= 20, "_GATED partition must not be empty"
    assert len(_ADMIN_ONLY) >= 10, "_ADMIN_ONLY partition must not be empty"
    assert len(_PUBLIC_MUTATING) >= 1, "_PUBLIC_MUTATING partition must not be empty"

    # F16 Advanced Control / Advisory Optimization safety assertions:
    # Advisory MPC simulation must be registered and admin-gated.
    assert ("POST", "/api/mpc/simulate") in ROUTE_TIERS, (
        "F16 advisory MPC simulation route /api/mpc/simulate must be classified"
    )
    assert ROUTE_TIERS[("POST", "/api/mpc/simulate")] == ADMIN, (
        "F16 advisory MPC simulation must require ADMIN tier"
    )

    # PID tuning control routes must be registered and admin-gated.
    for tuning_action in ("start", "step", "stop"):
        key = ("POST", f"/api/pid/{{pid_index}}/tuning/{tuning_action}")
        assert key in ROUTE_TIERS, f"Tuning route {key} must be classified"
        assert ROUTE_TIERS[key] == ADMIN, f"Tuning route {key} must require ADMIN tier"

    # Direct hardware write route must be admin-gated.
    assert ("POST", "/api/tags/{tag_id}") in ROUTE_TIERS
    assert ROUTE_TIERS[("POST", "/api/tags/{tag_id}")] == ADMIN

    # E-stop clear must be admin-gated, panic stop must remain public.
    assert ROUTE_TIERS[("POST", "/api/estop")] == PUBLIC
    assert ROUTE_TIERS[("POST", "/api/estop/clear")] == ADMIN


def test_every_route_is_classified() -> None:
    """A new endpoint that nobody classified must FAIL, not ship ungated."""
    routes = _app_routes()
    assert len(routes) >= 40, (
        f"Route inventory is suspiciously small ({len(routes)} routes). "
        "Route introspection may be under-reporting."
    )
    unclassified = sorted(routes - set(ROUTE_TIERS))
    assert not unclassified, (
        "These routes are not classified in ROUTE_TIERS. Add each one with its "
        "required credential tier (and the matching FastAPI dependency on the "
        f"route itself) before merging: {unclassified}"
    )


def _is_served(method: str, path: str) -> bool:
    """True if the app's router would dispatch ``method path`` to a handler.

    Checks both the robust route collector inventory and Starlette dispatch matching
    across app.routes and nested routers.
    """
    if (method.upper(), path) in _app_routes():
        return True

    scope = {
        "type": "http",
        "method": method,
        "path": _concrete(path),
        "path_params": {},
        "root_path": "",
        "headers": [],
        "query_string": b"",
    }

    def _matches_any(routes: Iterable[Any]) -> bool:
        for route in routes:
            orig_router = getattr(route, "original_router", None) or getattr(
                route, "router", None
            )
            if orig_router is not None and hasattr(orig_router, "routes"):
                if _matches_any(orig_router.routes):
                    return True
            if hasattr(route, "routes") and route.routes:
                if _matches_any(route.routes):
                    return True
            if hasattr(route, "app") and hasattr(route.app, "routes"):
                if _matches_any(route.app.routes):
                    return True
            if hasattr(route, "matches"):
                try:
                    match, _ = route.matches(scope)
                    if match is Match.FULL:
                        return True
                except Exception:
                    pass
        return False

    return _matches_any(app.routes) if hasattr(app, "routes") else False


def test_table_has_no_stale_rows() -> None:
    """Keep the contract honest in the other direction too."""
    assert ROUTE_TIERS, "ROUTE_TIERS must not be empty"
    stale = sorted(row for row in ROUTE_TIERS if not _is_served(*row))
    assert not stale, f"ROUTE_TIERS lists routes the app no longer serves: {stale}"


def test_every_mutating_route_is_gated() -> None:
    """No hardware-mutating route may be classified PUBLIC except the panic stop."""
    offenders = [
        (method, path)
        for (method, path), tier in ROUTE_TIERS.items()
        if tier == PUBLIC
        and method in {"POST", "PUT", "PATCH", "DELETE"}
        and path != "/api/estop"
    ]
    assert not offenders, f"Ungated mutating routes: {offenders}"


# --------------------------------------------------------------------------- #
# Enforcement                                                                  #
# --------------------------------------------------------------------------- #

_GATED = sorted(
    (method, path)
    for (method, path), tier in ROUTE_TIERS.items()
    if tier in {READ, OPERATOR, ADMIN}
)
_ADMIN_ONLY = sorted(
    (method, path) for (method, path), tier in ROUTE_TIERS.items() if tier == ADMIN
)
_PUBLIC_MUTATING = sorted(
    (method, path)
    for (method, path), tier in ROUTE_TIERS.items()
    if tier == PUBLIC and method in {"POST", "PUT", "PATCH", "DELETE"}
)


@pytest.mark.parametrize(("method", "path"), _GATED, ids=lambda v: str(v))
def test_gated_route_rejects_anonymous_caller(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(method, _concrete(path), headers=_headers(None), json={})
    assert resp.status_code in _DENIED, (
        f"{method} {path} answered {resp.status_code} with NO credential — the "
        "route is missing its auth dependency."
    )


@pytest.mark.parametrize(("method", "path"), _ADMIN_ONLY, ids=lambda v: str(v))
def test_admin_route_rejects_operator_credential(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(
        method, _concrete(path), headers=_headers(_OPERATOR_KEY), json={}
    )
    assert resp.status_code == 403, (
        f"{method} {path} answered {resp.status_code} for an OPERATOR key — a "
        "hardware-mutating route must require the admin credential."
    )


@pytest.mark.parametrize(("method", "path"), _PUBLIC_MUTATING, ids=lambda v: str(v))
def test_public_mutating_route_is_reachable_without_a_credential(
    client: TestClient, method: str, path: str
) -> None:
    """The panic stop must never regress into requiring a credential."""
    resp = client.request(method, _concrete(path), headers=_headers(None), json={})
    assert resp.status_code not in _DENIED


def test_admin_credential_passes_the_admin_gate(client: TestClient) -> None:
    """Sanity: the matrix measures authorization, not a blanket 403."""
    resp = client.request(
        "PUT",
        "/api/performance",
        headers=_headers(_ADMIN_KEY),
        json={"mode": "normal"},
    )
    assert resp.status_code not in _DENIED


def test_operator_credential_passes_the_operator_gate(client: TestClient) -> None:
    resp = client.request(
        "POST",
        "/api/events",
        headers=_headers(_OPERATOR_KEY),
        json={"event_type": "SYSTEM", "description": "matrix probe"},
    )
    assert resp.status_code not in _DENIED


# --------------------------------------------------------------------------- #
# WebSocket                                                                    #
# --------------------------------------------------------------------------- #


def _app_websocket_paths() -> set[str]:
    """Every WebSocket route path mounted on the app."""
    paths: set[str] = set()

    def _walk(routes: Iterable[Any], parent_prefix: str = "") -> None:
        for r in routes:
            raw_path = getattr(r, "path", None) or getattr(r, "path_format", None)
            orig_router = getattr(r, "original_router", None) or getattr(
                r, "router", None
            )
            inc_prefix = getattr(r, "prefix", "") or ""
            if not inc_prefix:
                ctx = getattr(r, "include_context", None)
                if isinstance(ctx, dict):
                    inc_prefix = str(ctx.get("prefix", ""))
                elif ctx is not None and hasattr(ctx, "prefix"):
                    inc_prefix = str(getattr(ctx, "prefix", "")) or ""

            comb = parent_prefix
            if inc_prefix:
                comb = (
                    f"{comb.rstrip('/')}/{inc_prefix.lstrip('/')}"
                    if comb
                    else inc_prefix
                )

            if orig_router is not None and hasattr(orig_router, "routes"):
                _walk(orig_router.routes, comb)
            elif hasattr(r, "routes") and r.routes:
                mount_p = raw_path or ""
                mount_comb = (
                    f"{comb.rstrip('/')}/{mount_p.lstrip('/')}"
                    if (comb and mount_p)
                    else (comb or mount_p)
                )
                _walk(r.routes, mount_comb)
            elif hasattr(r, "app") and hasattr(r.app, "routes"):
                mount_p = raw_path or ""
                mount_comb = (
                    f"{comb.rstrip('/')}/{mount_p.lstrip('/')}"
                    if (comb and mount_p)
                    else (comb or mount_p)
                )
                _walk(r.app.routes, mount_comb)

            if raw_path is not None and getattr(r, "methods", None) is None:
                full_path = (
                    f"{parent_prefix.rstrip('/')}/{raw_path.lstrip('/')}"
                    if parent_prefix
                    else raw_path
                )
                paths.add(full_path)

    if hasattr(app, "routes"):
        _walk(app.routes)
    return paths


def test_stream_websocket_route_exists() -> None:
    paths = _app_websocket_paths()
    assert "/api/stream" in paths


def test_stream_rejects_unauthenticated_client(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text("")
            ws.receive_text()


def test_stream_accepts_first_frame_credential(client: TestClient) -> None:
    """The HMI authenticates by sending the key as its first frame (#4007)."""
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text(_OPERATOR_KEY)
        # No exception on send after auth => the socket stayed open.
        ws.send_text("ping")
