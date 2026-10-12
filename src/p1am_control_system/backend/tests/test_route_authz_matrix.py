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
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _route_authz_helpers import (  # noqa: E402
    _ADMIN_ONLY,
    _GATED,
    _PUBLIC_MUTATING,
    ADMIN,
    PUBLIC,
    ROUTE_TIERS,
    _app_routes,
    _authenticated_deployment,  # noqa: F401  (autouse fixture)
    _is_served,
)

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
