"""Pinned downstream-facing API for impact_interval."""

from __future__ import annotations

import pytest

import shared.python.swing_sim.impact_interval as interval

EXPECTED_PUBLIC_API = {
    "BoundaryKind",
    "ClubRigidBody",
    "ImpactIntervalAudit",
    "ImpactIntervalConfig",
    "ImpactIntervalInitialState",
    "ImpactIntervalResult",
    "ImpactIntervalSample",
    # Added for Tools #4130 / UpstreamDrift #9547: callers must be able to
    # tell a separated contact from one truncated by the time budget, and to
    # catch the refusal to export an unfinished one.
    "ImpactTermination",
    "IncompleteContactError",
    "KelvinVoigtContactLaw",
    "solve_impact_interval",
}


@pytest.mark.contract
def test_public_api_is_explicit_and_pinned() -> None:
    assert set(interval.__all__) == EXPECTED_PUBLIC_API
    for symbol in interval.__all__:
        assert getattr(interval, symbol) is not None
