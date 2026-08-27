"""Facade contract for ``shared.python.swing_sim.putting`` (#4125 H3).

Self-façaded, parent untouched: downstream code imports from this
subpackage only; ``swing_sim/__init__.py`` deliberately carries no
putting exports (same policy as ``swing_sim.impact`` /
``swing_sim.variation``).
"""

from __future__ import annotations

import pytest

import shared.python.swing_sim as swing_sim
import shared.python.swing_sim.putting as putting

pytestmark = [pytest.mark.unit, pytest.mark.contract]

EXPECTED_PUBLIC_API = {
    "DEFAULT_PUTTER_COR",
    "DEFAULT_PUTTER_MOI_KG_M2",
    "DEFAULT_SLIDING_MU",
    "GREEN_SURFACE_FORMAT",
    "HOLE_RADIUS_M",
    "MINIMAL_PUTTERS",
    "PUTTING_STROKE_FORMAT",
    "STIMP_RELEASE_SPEED_MPS",
    "CaptureModel",
    "GreenConditions",
    "GreenSurface",
    "GridGreenSurface",
    "PlanarGreenSurface",
    "PuttLaunch",
    "PuttResult",
    "PutterSpec",
    "PuttingStroke",
    "SkidSolution",
    "StrokePutt",
    "StrokeSample",
    "StrokeStrike",
    "UdGreenTopography",
    "capture_speed_mps",
    "clubhead_speed_from_backstroke",
    "effective_hole_radius_m",
    "green_surface_from_json",
    "green_surface_from_ud_json",
    "green_surface_to_json",
    "green_surface_to_ud_json",
    "impact_sample_index",
    "putt_from_stroke",
    "putting_stroke_from_drake_json",
    "putting_stroke_from_json",
    "putting_stroke_from_mujoco_json",
    "putting_stroke_from_opensim_sto",
    "putting_stroke_to_json",
    "roll_out_distance",
    "roll_time_s",
    "rolling_mu_to_stimp",
    "simulate_putt",
    "simulate_putt_on_surface",
    "solve_skid",
    "stimp_to_rolling_mu",
    "strike_from_stroke",
    "strike_parameters",
}


class TestPuttingFacade:
    def test_public_api_is_pinned(self) -> None:
        assert set(putting.__all__) == set(EXPECTED_PUBLIC_API)

    def test_every_export_resolves(self) -> None:
        for name in putting.__all__:
            assert getattr(putting, name) is not None, name

    def test_parent_facade_is_untouched(self) -> None:
        assert not any("putting" in name.lower() for name in swing_sim.__all__)
