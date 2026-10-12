"""Shared route/tier table and fixtures for the route authorization matrix (#4028).

Support module, not a test: it holds the ``(method, path) -> tier`` contract, the
route-tree collectors and the authenticated-deployment fixtures consumed by
``test_route_authz_matrix.py`` (coverage of the table) and
``test_route_authz_enforcement.py`` (enforcement of it). Split from a single
625-line module to honour the 500 LOC file-size budget; no row or assertion was
dropped. See ``test_route_authz_matrix.py`` for the design rationale.
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
    # Config revision history (F04, #5527): audit data, admin-only, read-only.
    ("GET", "/api/config/revisions"): ADMIN,
    ("GET", "/api/config/revisions/{revision_id}"): ADMIN,
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
    "revision_id": "1",
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


# --------------------------------------------------------------------------- #
# Derived partitions                                                      #
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
