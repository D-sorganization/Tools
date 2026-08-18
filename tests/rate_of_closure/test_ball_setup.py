"""Ball support and tee-height contracts for canonical simulations."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club import CLUB_LIBRARY, ClubType, get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    BALL_POSITION_M,
    DEFAULT_DRIVER_TEE_HEIGHT_M,
    BallSetup,
    BallSupportMode,
    ContactMode,
    SimulationConfig,
    ball_setup_from_json_dict,
    run_simulation,
    run_to_json_dict,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)
_DRIVER = get_club("Driver 10.5°")
_IRON = get_club("7-Iron")


def test_driver_and_non_driver_defaults_are_club_specific() -> None:
    driver = SimulationConfig(scenario=_SCENARIO, club=_DRIVER)
    iron = SimulationConfig(scenario=_SCENARIO, club=_IRON)

    assert driver.ball_setup == BallSetup(
        BallSupportMode.TEE, DEFAULT_DRIVER_TEE_HEIGHT_M
    )
    assert iron.ball_setup == BallSetup(BallSupportMode.GROUND, 0.0)


def test_every_library_club_uses_its_required_default_support() -> None:
    for club in CLUB_LIBRARY.values():
        setup = SimulationConfig(scenario=_SCENARIO, club=club).ball_setup
        if club.club_type is ClubType.DRIVER:
            assert setup == BallSetup(BallSupportMode.TEE, DEFAULT_DRIVER_TEE_HEIGHT_M)
        else:
            assert setup == BallSetup(BallSupportMode.GROUND, 0.0)


def test_ball_center_uses_ground_to_ball_bottom_tee_height() -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.04)

    assert setup.ball_center_height_m == pytest.approx(GOLF_BALL_RADIUS_M + 0.04)
    np.testing.assert_allclose(
        setup.ball_center_m, (0.0, GOLF_BALL_RADIUS_M + 0.04, 0.0)
    )


@pytest.mark.parametrize("height", [-0.001, float("nan"), float("inf")])
def test_tee_height_must_be_finite_and_nonnegative(height: float) -> None:
    with pytest.raises(ContractViolationError, match="tee_height_m"):
        BallSetup(BallSupportMode.TEE, height)


def test_ground_support_requires_zero_tee_height() -> None:
    with pytest.raises(ContractViolationError, match="Ground support"):
        BallSetup(BallSupportMode.GROUND, 0.001)


def test_explicit_support_overrides_are_allowed_for_every_club() -> None:
    driver_ground = SimulationConfig(
        scenario=_SCENARIO,
        club=_DRIVER,
        ball_setup=BallSetup(BallSupportMode.GROUND),
    )
    iron_tee = SimulationConfig(
        scenario=_SCENARIO,
        club=_IRON,
        ball_setup=BallSetup(BallSupportMode.TEE, 0.01),
    )

    assert driver_ground.ball_setup.support_mode is BallSupportMode.GROUND
    assert iron_tee.ball_setup.tee_height_m == pytest.approx(0.01)


def test_delivery_alignment_and_flight_start_use_configured_ball_center() -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))
    center = np.asarray(run.config.ball_setup.ball_center_m)
    index = int(np.argmin(np.abs(run.swing_times - float(run.impact_time_s))))

    np.testing.assert_allclose(run.swing_positions[index], center, atol=1e-9)
    np.testing.assert_allclose(run.impact_outcome.ball_position_m, center, atol=1e-12)
    np.testing.assert_allclose(run.flight_positions[0], center, atol=1e-9)
    assert not np.allclose(center, BALL_POSITION_M)


def test_fixed_contact_can_miss_when_ball_is_teed_above_source_path() -> None:
    ground = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            ball_setup=BallSetup(BallSupportMode.GROUND),
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )
    teed = run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            ball_setup=BallSetup(BallSupportMode.TEE, 0.05),
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )

    assert ground.impact_outcome.is_hit
    assert not teed.impact_outcome.is_hit
    assert teed.impact_time_s is None
    assert teed.delivery is None
    assert teed.flight_times.shape == (0,)


def test_ball_setup_json_round_trip_and_legacy_migration() -> None:
    setup = BallSetup(BallSupportMode.TEE, 0.032)

    assert BallSetup.from_json_dict(setup.to_json_dict()) == setup
    assert BallSetup.from_json_dict(None) == BallSetup(BallSupportMode.GROUND)


def test_run_export_includes_unambiguous_ball_setup() -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))
    payload = run_to_json_dict(run)

    exported = payload["parameters"]["ball_setup"]
    assert exported["support_mode"] == "tee"
    assert exported["tee_height_m"] == pytest.approx(DEFAULT_DRIVER_TEE_HEIGHT_M)
    assert exported["height_reference"] == "ground_plane_to_ball_bottom"
    assert exported["ball_center_m"] == pytest.approx(
        list(run.config.ball_setup.ball_center_m)
    )
    assert ball_setup_from_json_dict(payload) == run.config.ball_setup


def test_legacy_run_without_ball_setup_imports_at_ground_position() -> None:
    legacy_run = {
        "format": "rate_of_closure.simulation_run/1",
        "parameters": {"club": _DRIVER.name},
    }

    assert ball_setup_from_json_dict(legacy_run) == BallSetup(BallSupportMode.GROUND)
