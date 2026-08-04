"""Shared swing simulation package (epic #4103, P0 foundation #4104).

Curated façade for the swing → impact → ball-flight platform's swing stage:
value types, the :class:`SwingSource` protocol, the double-pendulum swing
source, and backend discovery. Physics kernels live in the
``rust_core/swing-core`` Rust crate (Python wheel ``swing_core``, WASM NPM
package for the web app); :mod:`.reference` is the pure-Python parity
oracle and one-shot fallback.
"""

from __future__ import annotations

from ._rust_facade import rust_available
from .swing_source import DoublePendulumSwing, SwingSource
from .types import (
    DEFAULT_GRAVITY_M_S2,
    PendulumParameters,
    PendulumState,
    PlaneOrientation,
    SwingSample,
    SwingTrajectory,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_GRAVITY_M_S2",
    "DoublePendulumSwing",
    "PendulumParameters",
    "PendulumState",
    "PlaneOrientation",
    "SwingSample",
    "SwingSource",
    "SwingTrajectory",
    "__version__",
    "rust_available",
]
