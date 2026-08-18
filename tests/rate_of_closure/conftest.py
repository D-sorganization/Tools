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
