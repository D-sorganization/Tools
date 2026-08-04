"""Contract test pinning the public API surface of swing_sim.

Downstream consumers (UpstreamDrift, rate_of_closure UI, web backend)
import from the curated façade only; this test fails loudly when the
surface changes so removals are always deliberate.
"""

from __future__ import annotations

import pytest

import shared.python.swing_sim as swing_sim

EXPECTED_PUBLIC_API = {
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
}


@pytest.mark.contract
def test_public_api_surface_is_pinned() -> None:
    assert set(swing_sim.__all__) == EXPECTED_PUBLIC_API


@pytest.mark.contract
def test_all_exports_resolve() -> None:
    for name in swing_sim.__all__:
        assert getattr(swing_sim, name) is not None, f"{name} did not resolve"


@pytest.mark.contract
def test_swing_source_protocol_shape() -> None:
    from shared.python.swing_sim import SwingSource

    assert hasattr(SwingSource, "sample")
    assert hasattr(SwingSource, "duration")


@pytest.mark.contract
def test_types_are_frozen_dataclasses() -> None:
    import dataclasses

    from shared.python.swing_sim import (
        PendulumParameters,
        PendulumState,
        PlaneOrientation,
        SwingSample,
        SwingTrajectory,
    )

    for cls in (
        PlaneOrientation,
        PendulumParameters,
        PendulumState,
        SwingSample,
        SwingTrajectory,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} not a dataclass"
        params = cls.__dataclass_params__
        assert params.frozen, f"{cls.__name__} must be frozen"
