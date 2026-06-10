"""Design by Contract shim for ``video_analyzer``.

Re-exports ``precondition`` (and related primitives) from the monorepo's
shared ``contracts`` module when it is importable, regardless of which
directories happen to be on ``sys.path``.

The package is consumed cross-repo (e.g. by UpstreamDrift) with only the
repository root on ``sys.path`` and modules imported under the ``src.``
namespace. In that configuration the bare ``from contracts import ...``
import is not resolvable, so this shim tries several stable locations and
finally falls back to lightweight no-op implementations so that importing
``video_analyzer`` never fails purely because the shared DbC module could
not be located.

Consumers inside ``video_analyzer`` should always import from here::

    from ._contracts import precondition
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

precondition: Callable[..., Callable[[_F], _F]]

try:  # pragma: no cover - exercised via import-contract tests
    # Fully-qualified path that resolves when the repo root is on sys.path
    # and the package is imported under the ``src.`` namespace.
    from src.shared.python.contracts import precondition as _shared_precondition

    precondition = _shared_precondition
except ImportError:  # pragma: no cover - fallbacks
    try:
        # Path that resolves when ``src/`` is on sys.path (in-repo test runs).
        from shared.python.contracts import precondition as _src_precondition

        precondition = _src_precondition
    except ImportError:
        try:
            # Short alias that resolves when ``src/`` or ``src/shared/python``
            # is on sys.path.
            from contracts import precondition as _short_precondition

            precondition = _short_precondition
        except ImportError:
            # ── Standalone fallback ──────────────────────────────────
            # A no-op ``precondition`` decorator factory. It preserves the
            # decorated callable unchanged so behaviour is identical to the
            # real DbC primitive when contracts are disabled.
            def precondition(  # noqa: F811
                *_args: Any, **_kwargs: Any
            ) -> Callable[[_F], _F]:
                """Return an identity decorator (DbC unavailable fallback)."""

                def _decorator(func: _F) -> _F:
                    return func

                return _decorator


__all__ = ["precondition"]
