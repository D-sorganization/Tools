"""Tests for ball-flight termination propagation and fail-closed landing gates.

Addresses Tools #5385 / UpstreamDrift #11145.
Verifies that:
1. Five distinct termination conditions (TIME_LIMIT, LANDED, negative launch,
   SOLVER_FAILED, CANCELLED) have distinct and well-defined semantics.
2. Incomplete flights gate carry/landing metrics to None and raise IncompleteFlightError
   on require_landing().
3. Downstream consumers (inverse solver adapter, ground transfer, profile qualification)
   fail closed and reject incomplete flights.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.application.flight_execution_profiles import (
    FlightExecutionQualificationReason,
    _recompute_registered,
)
from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightGroundTransferSettings,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    FlightSimulationCancelled,
    FlightStatePoint,
    FlightTermination,
    ForwardStatus,
    ImpactSolutionRequest,
    IncompleteFlightError,
    LaunchConditions,
    SurfaceFlightSimulationSettings,
    TrajectoryPoint,
    build_ground_simulation_request,
    compute_flight_metrics,
)
from shared.python.swing_sim.flight.impact_solution_adapter import (
    CenteredClubDeliveryAdapter,
)
from shared.python.swing_sim.flight.impact_solution_contract import (
    ClubProfileId,
)
from shared.python.swing_sim.flight.inverse_contract import (
    DecisionVariable,
    FlightObjective,
    InverseFlightRequest,
    ObjectiveMode,
)
from shared.python.swing_sim.flight.result_contract import FlightMetricId
from shared.python.swing_sim.ground import (
    CalibrationKind,
    GroundCalibration,
    GroundFrame,
    GroundProvenance,
    GroundSurfaceProfile,
)


def _ascending_time_cap_launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=70.0,
        launch_angle=0.3,
        spin_rate=2500.0,
    )


def _normal_launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=70.0,
        launch_angle=math.radians(12.0),
        spin_rate=2500.0,
    )


def _negative_angle_launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=70.0,
        launch_angle=math.radians(-5.0),
        spin_rate=2500.0,
    )


def _surface(height_m: float = 0.0) -> GroundSurfaceProfile:
    return GroundSurfaceProfile(
        surface_id="test-plane",
        provider_id="tools.flight-test",
        provider_version="1.0.0",
        frame=GroundFrame.TARGET,
        height_m=height_m,
        normal_unit=(0.0, 1.0, 0.0),
        surface_velocity_m_s=(0.0, 0.0, 0.0),
        normal_restitution=0.4,
        static_friction=0.35,
        kinetic_friction=0.25,
        rolling_resistance=0.04,
        firmness_pa=1_000_000.0,
        hardness_fraction=0.7,
        grass_height_m=0.01,
        compressibility_fraction=0.2,
        compression_damping_fraction=0.2,
        turf_density_kg_m3=180.0,
        moisture_fraction=0.3,
    )


def _transfer_settings() -> FlightGroundTransferSettings:
    return FlightGroundTransferSettings(
        request_id="flight-ground-001",
        surface=_surface(),
        calibration=GroundCalibration(
            "test-calibration", CalibrationKind.MEASURED, "test evidence", 1.0
        ),
        provenance=GroundProvenance("pytest", "1.0", "local", "a" * 64),
        max_time_s=12.0,
        output_interval_s=0.01,
        max_events=32,
    )


def _surface_settings(max_time_s: float = 30.0) -> SurfaceFlightSimulationSettings:
    return SurfaceFlightSimulationSettings(
        launch_relative_surface=_surface(),
        max_time_s=max_time_s,
        output_interval_s=0.01,
    )


class TestFlightTerminationFixtures:
    """Distinct results for all five termination scenarios."""

    def test_time_limit_while_ascending_fixture(self) -> None:
        """Integration reaches max_time while ball is ascending."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        # 0.1 s budget is far too short to land
        result = model.simulate(_ascending_time_cap_launch(), max_time=0.1)

        assert result.termination is FlightTermination.TIME_LIMIT
        assert result.terminal_event is False
        assert result.landed is False
        assert result.flight_completed is False

        # Landing-derived metrics must be None
        assert result.carry_distance is None
        assert result.landing_angle is None
        assert result.lateral_deviation is None

        # Trajectory-derived quantities exist
        assert result.max_height > 0.0
        assert result.flight_time == pytest.approx(0.1, abs=0.02)
        assert result.actual_horizon == pytest.approx(0.1, abs=0.02)
        assert len(result.trajectory) > 1

        # Calling require_landing() must raise IncompleteFlightError
        with pytest.raises(IncompleteFlightError) as exc_info:
            result.require_landing()

        err = exc_info.value
        assert err.result is result
        assert err.termination is FlightTermination.TIME_LIMIT
        assert "termination=time_limit" in str(err)
        assert "Inspect .result for the partial trace" in str(err)

    def test_normal_landing_fixture(self) -> None:
        """Full simulation running to ground impact event."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_normal_launch(), max_time=30.0)

        assert result.termination is FlightTermination.LANDED
        assert result.terminal_event is True
        assert result.landed is True
        assert result.flight_completed is True

        assert result.carry_distance is not None and result.carry_distance > 150.0
        assert result.landing_angle is not None and result.landing_angle > 0.0
        assert result.lateral_deviation is not None
        assert result.max_height > 10.0
        assert result.flight_time > 3.0
        assert result.actual_horizon > 3.0

        # require_landing() returns self
        assert result.require_landing() is result

    def test_negative_launch_angle_fixture(self) -> None:
        """Ball launched downward reaches the ground immediately."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_negative_angle_launch(), max_time=10.0)

        assert result.termination is FlightTermination.LANDED
        assert result.terminal_event is True
        assert result.landed is True
        assert result.flight_completed is True

        assert result.actual_horizon == pytest.approx(0.0, abs=1e-6)
        assert result.carry_distance == pytest.approx(0.0, abs=1e-6)
        assert result.flight_time == pytest.approx(0.0, abs=1e-6)
        # Landing angle corresponds to the downward angle (positive downward)
        assert result.landing_angle is not None
        assert result.landing_angle == pytest.approx(5.0, abs=0.5)

    def test_solver_failed_fixture(self) -> None:
        """Solver failure produces SOLVER_FAILED with None landing metrics."""
        points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])),
        ]
        result = compute_flight_metrics(
            points,
            "mock_failed",
            termination=FlightTermination.SOLVER_FAILED,
            terminal_event=False,
            actual_horizon=0.0,
        )

        assert result.termination is FlightTermination.SOLVER_FAILED
        assert result.terminal_event is False
        assert result.landed is False
        assert result.carry_distance is None
        assert result.landing_angle is None
        assert result.lateral_deviation is None

        with pytest.raises(IncompleteFlightError) as exc_info:
            result.require_landing()
        assert exc_info.value.termination is FlightTermination.SOLVER_FAILED

    def test_cancelled_fixture(self) -> None:
        """Cooperative cancellation raises typed exception carrying partial result."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        settings = _surface_settings(max_time_s=30.0)

        step_count = 0

        def should_cancel() -> bool:
            nonlocal step_count
            step_count += 1
            return step_count >= 5

        with pytest.raises(FlightSimulationCancelled) as exc_info:
            model.simulate_to_surface(
                _normal_launch(),
                settings,
                cancellation_requested=should_cancel,
            )

        err = exc_info.value
        assert err.result is not None
        partial = err.result
        assert partial.termination is FlightTermination.CANCELLED
        assert partial.terminal_event is False
        assert partial.landed is False
        assert partial.carry_distance is None
        assert partial.landing_angle is None
        assert partial.lateral_deviation is None


class TestFlightResultContractValidation:
    """Enforcement of landing metrics constraints and invariants."""

    def test_rejects_non_none_metrics_on_unlanded_flight(self) -> None:
        points = (
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])),
        )
        with pytest.raises(ValueError, match="Landing metrics.*must be None"):
            FlightResult(
                points,
                "test",
                carry_distance=100.0,
                termination=FlightTermination.TIME_LIMIT,
            )

    def test_default_landed_flight_allows_zero_metrics(self) -> None:
        points = (
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])),
        )
        result = FlightResult(points, "test")
        assert result.landed is True
        assert result.carry_distance == 0.0
        assert result.landing_angle == 0.0
        assert result.lateral_deviation == 0.0


class TestDownstreamFailClosedGates:
    """Partial trajectories cannot score or enter qualified datasets."""

    def test_impact_solution_adapter_fails_closed_on_unlanded_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CenteredClubDeliveryAdapter fails closed if flight did not land."""
        variables = (
            DecisionVariable("clubhead_speed_mps", "m/s", 40.0, 50.0, 44.0),
            DecisionVariable("attack_angle_deg", "deg", -5.0, 5.0, 2.0),
            DecisionVariable("dynamic_loft_deg", "deg", 5.0, 20.0, 12.0),
            DecisionVariable("club_path_deg", "deg", -5.0, 5.0, 0.0),
            DecisionVariable("face_angle_deg", "deg", -5.0, 5.0, 0.0),
        )
        objectives = (
            FlightObjective(
                FlightMetricId.CARRY_DISTANCE, "m", ObjectiveMode.TARGET, 200.0
            ),
        )
        inv_req = InverseFlightRequest(
            problem_id="test-incomplete",
            variables=variables,
            objectives=objectives,
            max_evaluations=10,
            candidate_count=5,
        )
        req = ImpactSolutionRequest(
            inverse_request=inv_req,
            club_profile_id=ClubProfileId.CENTERED_DRIVER,
            flight_model_id="waterloo_penner",
            family_count=3,
            family_radius=0.2,
            sensitivity_fraction=0.01,
            impact_event_time_s=0.0,
        )
        adapter = CenteredClubDeliveryAdapter(req)

        # Mock simulate to return an unlanded flight (TIME_LIMIT)
        dummy_point = TrajectoryPoint(
            0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])
        )
        unlanded = FlightResult(
            (dummy_point,),
            "waterloo_penner",
            termination=FlightTermination.TIME_LIMIT,
        )
        monkeypatch.setattr(
            "shared.python.swing_sim.flight.impact_solution_adapter.simulate",
            lambda *args, **kwargs: unlanded,
        )

        evaluation = adapter.evaluate(
            {
                "clubhead_speed_mps": 44.0,
                "attack_angle_deg": 2.0,
                "dynamic_loft_deg": 12.0,
                "club_path_deg": 0.0,
                "face_angle_deg": 0.0,
            }
        )
        assert evaluation.status is ForwardStatus.FAILED
        assert evaluation.reason == "flight_incomplete:time_limit"

    def test_ground_transfer_rejects_unlanded_flight(self) -> None:
        """build_ground_simulation_request fails closed if flight did not land."""
        points = (
            FlightStatePoint(
                0.0,
                np.array([0.0, 0.0, 1.0]),
                np.array([10.0, 0.0, 0.0]),
                np.array([0.0, -100.0, 0.0]),
            ),
        )
        unlanded = FlightResult(
            points,
            "test",
            termination=FlightTermination.TIME_LIMIT,
        )
        with pytest.raises(
            FlightGroundTransferError, match="flight trajectory did not land"
        ):
            build_ground_simulation_request(
                unlanded,
                _normal_launch(),
                _transfer_settings(),
            )

    def test_flight_execution_profile_qualification_rejects_unlanded_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_recompute_registered rejects unlanded flight."""
        # Mock recompute_waterloo to return an unlanded flight
        dummy_point = TrajectoryPoint(
            0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])
        )
        unlanded = FlightResult(
            (dummy_point,),
            "waterloo_penner",
            termination=FlightTermination.TIME_LIMIT,
        )
        monkeypatch.setattr(
            "rate_of_closure.application.flight_execution_profiles.recompute_waterloo",
            lambda *args, **kwargs: unlanded,
        )
        qual, res = _recompute_registered(
            _normal_launch(),
            _transfer_settings(),
            model_id="waterloo_penner",
            model_version="tools-core/1.0.0",
            settings={"max_time_s": 30.0, "step_s": 0.01, "sample_every": 1},
        )
        assert qual.reason is FlightExecutionQualificationReason.RECOMPUTATION_FAILED
        assert res is None
