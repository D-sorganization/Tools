"""Shared fixtures for the rate_of_closure suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_distance_display_unit():  # type: ignore[no-untyped-def]
    """Pin the session distance display unit to its yards default.

    The H6 Distance quantity (#4125) is session-global presentation
    state; tests that switch it must not leak into their neighbours.
    """
    from rate_of_closure.units import set_display_distance_unit

    set_display_distance_unit("yd")
    yield
    set_display_distance_unit("yd")


@pytest.fixture(autouse=True)
def _restore_shared_variation_registry():  # type: ignore[no-untyped-def]
    """Keep Rate-owned ground variables out of the shared swing_sim registry.

    `regional_ground_variation_request` registers the ground variation
    variables from inside its parse path, so merely reading a request mutates
    `swing_sim.variation.registry` for the rest of the process. That leaked into
    `swing_sim/variation/tests/test_spec.py`, whose category pins assert exact
    membership — it passed alone and failed after this suite.

    Snapshotting and restoring the registry per test keeps the leak inside the
    package that causes it. The registration itself is deliberate (the plan
    parser needs those variables defined before it validates), so this restores
    rather than blocks it.
    """
    from shared.python.swing_sim.variation import registry

    before = dict(registry.variable_registry())
    yield
    registry._REGISTRY.clear()  # noqa: SLF001 - test-only restoration
    registry._REGISTRY.update(before)  # noqa: SLF001 - test-only restoration
