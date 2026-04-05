"""Design by Contract support for urdf_builder_gui.

Re-exports from the canonical contracts implementation at
``src/shared/python/contracts.py``.  The shared module provides the
full API (imperative + decorator styles) and tri-state enforcement.
"""

from __future__ import annotations

from contracts import (  # noqa: F401
    PostconditionError,
    PreconditionError,
    ensure,
    require,
)

__all__ = [
    "PostconditionError",
    "PreconditionError",
    "ensure",
    "require",
]
