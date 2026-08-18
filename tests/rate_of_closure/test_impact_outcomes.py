"""Contact-outcome regression tests for Rate of Closure simulations."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import extract
from rate_of_closure.simulation import (
    BallSetup,
    BallSupportMode,
    ContactMode,
    ImpactStatus,
    SimulationConfig,
    SimulationRun,
    kinetics_for_run,
    make_source,
    run_simulation,
    run_to_json_dict,
    write_csv,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_SCENARIO = ImpactScenario(clubhead_speed_mph=113.0)
_DRIVER = get_club("Driver 10.5°")


def test_delivery_inspection_remains_default_and_forces_alignment() -> None:
    """The legacy delivery-inspection workflow remains an explicit hit mode."""
    tau = 0.01
    run = run_simulation(
        SimulationConfig(scenario=_SCENARIO, club=_DRIVER, impact_time_s=tau)
    )

    assert run.config.contact_mode is ContactMode.DELIVERY_INSPECTION
    assert run.impact_outcome.status is ImpactStatus.HIT
    assert run.impact_outcome.geometry_model == "forced_reference_point_alignment"
    assert run.impact_time_s == pytest.approx(tau)
    index = int(np.argmin(np.abs(run.swing_times - tau)))
    np.testing.assert_allclose(
        run.swing_positions[index], run.config.ball_position_m, atol=1e-9
    )
    assert run.delivery is not None
    assert run.post_impact is not None
    assert run.launch is not None


def test_manual_auto_inspection_uses_midpoint_of_constant_speed_plateau() -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))

    assert run.impact_time_s == pytest.approx(0.03)


def test_fixed_ball_contact_hit_does_not_translate_swing() -> None:
    """A sampled fixed-ball hit retains the source's original world positions."""
    config = SimulationConfig(
        scenario=_SCENARIO,
        club=_DRIVER,
        ball_setup=BallSetup(BallSupportMode.GROUND),
        contact_mode=ContactMode.FIXED_BALL_CONTACT,
    )
    run = run_simulation(config)
    source = make_source("manual", _SCENARIO)
    expected = np.stack(
        [source.sample(float(time_s)).pose[:3, 3] for time_s in run.swing_times]
    )

    assert run.impact_outcome.status is ImpactStatus.HIT
    assert run.impact_time_s == pytest.approx(source.duration / 2.0)
    assert run.impact_outcome.closest_approach_m == pytest.approx(
        np.linalg.norm(config.ball_position_m)
    )
    assert run.impact_outcome.contact_margin_m == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(run.swing_positions, expected, atol=1e-12)


@pytest.fixture()
def fixed_ball_miss() -> SimulationRun:
    """A short pendulum swing stays well away from the fixed ball."""
    return run_simulation(
        SimulationConfig(
            scenario=_SCENARIO,
            club=_DRIVER,
            ball_setup=BallSetup(BallSupportMode.GROUND),
            source_kind="double_pendulum",
            swing_duration_s=0.05,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )


def test_fixed_ball_miss_retains_complete_swing_without_impact(
    fixed_ball_miss: SimulationRun,
) -> None:
    """No-contact is a typed result, not an exception or fabricated impact."""
    run = fixed_ball_miss
    assert run.impact_outcome.status is ImpactStatus.MISS
    assert run.impact_outcome.contact_margin_m < 0.0
    assert run.impact_outcome.candidate_time_s == pytest.approx(run.swing_times[-1])
    assert run.swing_times[0] == pytest.approx(0.0)
    assert run.swing_times[-1] == pytest.approx(0.05)
    assert len(run.swing_positions) == len(run.swing_times)
    assert run.impact_time_s is None
    assert run.inspection_time_s == pytest.approx(run.impact_outcome.candidate_time_s)
    assert run.inspection_event_label == "Closest Approach"
    assert run.delivery is None
    assert run.post_impact is None
    assert run.launch is None
    assert run.flight_times.shape == (0,)
    assert run.flight_positions.shape == (0, 3)
    assert run.flight_velocities.shape == (0, 3)


def test_hit_inspection_event_is_the_physical_impact() -> None:
    run = run_simulation(SimulationConfig(scenario=_SCENARIO, club=_DRIVER))

    assert run.impact_time_s is not None
    assert run.inspection_time_s == pytest.approx(run.impact_time_s)
    assert run.inspection_event_label == "Impact"


def test_fixed_ball_miss_exports_honest_json_and_csv(
    fixed_ball_miss: SimulationRun,
    tmp_path: Path,
) -> None:
    """Miss exports retain swing samples and explicitly mark absent phases."""
    run = fixed_ball_miss
    payload = run_to_json_dict(run)
    assert payload["impact_outcome"]["status"] == "miss"
    assert payload["parameters"]["impact_time_s"] is None
    assert payload["delivery"] is None
    assert payload["launch"] is None
    assert {row[0] for row in payload["series"]["rows"]} == {"swing"}

    path = tmp_path / "miss.csv"
    write_csv(run, path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(run.swing_times)
    assert {row["phase"] for row in rows} == {"swing"}
    assert {row["is_fixed_ball_contact"] for row in rows} == {"1"}
    assert {row["impact_occurred"] for row in rows} == {"0"}
    assert {row["impact_time_s"] for row in rows} == {""}
    assert all(float(row["contact_margin_m"]) < 0.0 for row in rows)


def test_fixed_ball_miss_downstream_properties_are_null_safe(
    fixed_ball_miss: SimulationRun,
) -> None:
    """Backend consumers use empty series or NaN instead of dereferencing None."""
    run = fixed_ball_miss
    assert run.total_duration_s == pytest.approx(run.swing_times[-1])
    assert kinetics_for_run(run) is None
    assert math.isnan(float(extract(run, "input.impact_time_s")))
    assert math.isnan(float(extract(run, "impact.clubhead_speed_mps")))
    assert math.isnan(float(extract(run, "launch.ball_speed_mph")))
    assert np.asarray(extract(run, "flight.time_s")).shape == (0,)
