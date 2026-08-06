"""Tests for the standalone flight-explorer logic (epic #4120, V2).

Pure-physics layer: launch building from direct ball numbers and from
club delivery numbers, flight integration across the 7 literature
models, app-frame sign conventions (+ = right of target), and the
pinned end-to-end case that the TypeScript twin bands against
(``web/src/model/flightExplorer.test.ts``).
"""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.simulation import (
    BALL_POSITION_M,
    EXPLORER_METRIC_KEYS,
    compare_wind,
    explore_flight,
    launch_from_delivery,
    launch_from_direct,
)
from shared.python.swing_sim.flight import WindScenario
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import DeliveryParameters

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: The pinned tour-driver direct case, mirrored by the TS parity test.
PINNED_DIRECT = {
    "ball_speed_mph": 167.0,
    "launch_angle_deg": 10.9,
    "azimuth_deg": 0.0,
    "spin_rpm": 2686.0,
    "spin_axis_tilt_deg": 0.0,
}
#: Expected metrics for PINNED_DIRECT under waterloo_penner (scipy RK45).
PINNED_METRICS = {
    "carry_m": 247.484,
    "max_height_m": 28.226,
    "flight_time_s": 6.278,
    "landing_angle_deg": 35.120,
    "lateral_m": 0.0,
}


class TestLaunchFromDirect:
    def test_pinned_tour_driver_case(self) -> None:
        exploration = explore_flight(
            launch_from_direct(**PINNED_DIRECT), "waterloo_penner"
        )
        for key, expected in PINNED_METRICS.items():
            assert exploration.metrics[key] == pytest.approx(expected, abs=0.05), key

    def test_entered_numbers_round_trip_into_metrics(self) -> None:
        exploration = explore_flight(launch_from_direct(150.0, 14.0, -3.0, 3100.0, 0.0))
        assert exploration.metrics["ball_speed_mph"] == pytest.approx(150.0)
        assert exploration.metrics["launch_angle_deg"] == pytest.approx(14.0)
        assert exploration.metrics["launch_azimuth_deg"] == pytest.approx(-3.0)
        assert exploration.metrics["spin_rpm"] == pytest.approx(3100.0)

    def test_positive_azimuth_lands_right_of_target(self) -> None:
        right = explore_flight(launch_from_direct(150.0, 12.0, 5.0, 2700.0, 0.0))
        left = explore_flight(launch_from_direct(150.0, 12.0, -5.0, 2700.0, 0.0))
        assert right.metrics["lateral_m"] > 1.0
        assert left.metrics["lateral_m"] < -1.0

    def test_fade_tilt_curves_right_and_draw_tilt_left(self) -> None:
        fade = explore_flight(launch_from_direct(150.0, 12.0, 0.0, 2700.0, 10.0))
        draw = explore_flight(launch_from_direct(150.0, 12.0, 0.0, 2700.0, -10.0))
        assert fade.metrics["lateral_m"] > 1.0
        assert draw.metrics["lateral_m"] < -1.0

    def test_rejects_nonpositive_ball_speed(self) -> None:
        with pytest.raises(Exception, match="ball_speed"):
            launch_from_direct(0.0, 12.0, 0.0, 2500.0, 0.0)


class TestLaunchFromDelivery:
    def test_square_driver_delivery_pins(self) -> None:
        launch = launch_from_delivery(
            DeliveryParameters(
                clubhead_speed_mps=50.0,
                attack_angle_deg=-1.0,
                dynamic_loft_deg=12.0,
            )
        )
        exploration = explore_flight(launch, "waterloo_penner")
        assert exploration.metrics["ball_speed_mph"] == pytest.approx(162.187, abs=0.05)
        assert exploration.metrics["spin_rpm"] == pytest.approx(3595.9, abs=1.0)
        assert exploration.metrics["carry_m"] == pytest.approx(240.65, abs=0.5)

    def test_open_face_produces_fade_side_lateral(self) -> None:
        launch = launch_from_delivery(
            DeliveryParameters(
                clubhead_speed_mps=50.0, face_angle_deg=3.0, dynamic_loft_deg=12.0
            )
        )
        exploration = explore_flight(launch)
        # An open face starts the ball right and adds fade-side spin.
        assert exploration.metrics["launch_azimuth_deg"] > 0.5
        assert exploration.metrics["lateral_m"] > 1.0


class TestExploreFlight:
    def test_all_seven_literature_models_run(self) -> None:
        launch = launch_from_direct(150.0, 12.0, 0.0, 2700.0, 0.0)
        for model in FlightModelType:
            exploration = explore_flight(launch, model.value)
            assert exploration.model_name == model.value
            assert exploration.metrics["carry_m"] > 50.0, model.value

    def test_metrics_cover_the_declared_keys_exactly(self) -> None:
        exploration = explore_flight(launch_from_direct(150.0, 12.0, 0.0, 2700.0, 0.0))
        assert set(exploration.metrics) == set(EXPLORER_METRIC_KEYS)

    def test_trajectory_is_app_frame_from_the_tee(self) -> None:
        exploration = explore_flight(launch_from_direct(150.0, 12.0, 0.0, 2700.0, 0.0))
        positions = exploration.positions
        assert positions.ndim == 2 and positions.shape[1] == 3
        assert len(positions) == len(exploration.times)
        assert np.allclose(positions[0], BALL_POSITION_M, atol=1e-6)
        # Downrange (x) grows, and the last sample is back at the ground.
        assert positions[-1, 0] > 100.0
        assert positions[-1, 1] == pytest.approx(BALL_POSITION_M[1], abs=0.5)

    def test_unknown_model_name_raises(self) -> None:
        launch = launch_from_direct(150.0, 12.0, 0.0, 2700.0, 0.0)
        with pytest.raises(ValueError):
            explore_flight(launch, "no_such_model")

    def test_wind_comparison_uses_identical_launch_and_explicit_deltas(self) -> None:
        launch = launch_from_direct(150.0, 12.0, 0.0, 2700.0, 0.0)
        comparison = compare_wind(
            launch,
            WindScenario.from_meteorological(8.0, 0.0),
        )

        assert comparison.wind.metrics["carry_m"] < comparison.calm.metrics["carry_m"]
        assert comparison.deltas["carry_m"] == pytest.approx(
            comparison.wind.metrics["carry_m"] - comparison.calm.metrics["carry_m"]
        )
        assert comparison.wind.launch.ball_speed == comparison.calm.launch.ball_speed
