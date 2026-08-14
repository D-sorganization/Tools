"""Version-agnostic HTTP route inventory for a FastAPI app under test.

Why this exists
---------------
Tests that assert on which routes an app serves used to iterate ``app.routes``
and read ``route.path``. That stopped being reliable:

* Up to roughly FastAPI 0.130 / Starlette 0.52, ``include_router()`` flattened
  the included ``APIRoute`` objects straight into ``app.routes``, so every entry
  had a ``.path``.
* From FastAPI 0.141 / Starlette 1.6, ``include_router()`` instead leaves a
  single ``fastapi.routing._IncludedRouter`` marker in ``app.routes``. That
  marker has ``path=None``, exposes **no** ``.routes`` attribute, and keeps the
  real routes on a private ``original_router`` with the prefix held separately
  on a private ``include_context``. So the included paths are not reachable by
  walking ``app.routes`` at all, recursively or otherwise.

Both failure modes are bad, and the second is worse than it looks:

* Reading ``route.path`` unguarded raises
  ``AttributeError: '_IncludedRouter' object has no attribute 'path'``.
* Skipping entries without a string ``path`` — the obvious "tolerate it" fix —
  silently yields an **empty** inventory. Any ``all(...)`` assertion over that
  inventory then passes vacuously, so a safety contract like "no advisory route
  exposes a command or write path" would report green while verifying nothing.

The fix is to stop introspecting the route table and ask the app for its own
schema instead. ``app.openapi()`` resolves included routers and their prefixes
itself and returns fully-qualified, templated paths. It is public API and gives
byte-identical results on both the old and new versions, so it needs no version
branch.

Deliberate scope: only schema-visible HTTP operations are reported. Routes
registered with ``include_in_schema=False``, and the docs/openapi endpoints
FastAPI mounts for itself, are intentionally absent — assertions here are about
the application's contract surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

__all__ = ["HTTP_METHODS", "methods_by_path", "route_paths"]

# The operation keys OpenAPI defines on a Path Item Object. Everything else a
# path item may carry (``parameters``, ``summary``, ``description``, ``servers``)
# is metadata, not an operation, and must not be mistaken for a method.
HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


def methods_by_path(app: FastAPI) -> dict[str, set[str]]:
    """Map each schema-visible path to the upper-case HTTP methods it serves.

    Args:
        app: The application to inventory.

    Returns:
        ``{"/api/auth/session": {"POST", "DELETE"}, ...}``. Paths keep OpenAPI
        templating, so a parameterised route appears as ``/api/x/{tag}/shelf``.
        Paths whose path item declares no operation are omitted.
    """
    inventory: dict[str, set[str]] = {}
    for path, item in (app.openapi().get("paths") or {}).items():
        methods = {key.upper() for key in item if key.lower() in HTTP_METHODS}
        if methods:
            inventory.setdefault(path, set()).update(methods)
    return inventory


def route_paths(app: FastAPI) -> set[str]:
    """Return every schema-visible path the app serves.

    Args:
        app: The application to inventory.

    Returns:
        The set of templated paths. Callers asserting a *negative* property over
        this set (for example "no path contains 'write'") should also assert the
        set is non-empty, or the assertion cannot fail.
    """
    return set(methods_by_path(app))
