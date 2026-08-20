"""Solver API, recorder, and energy-balance tests.

Includes the solver-level regression test for defect (a): the base
impulse of ``solve_with_gear_effect`` must include the impact offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.impact.models import RigidBodyImpactModel
from shared.python.swing_sim.impact.solver import ImpactRecorder, ImpactSolverAPI
from shared.python.swing_sim.impact.types import (
    ImpactModelType,
    ImpactParameters,
    PreImpactState,
)
from shared.python.swing_sim.impact.utils import validate_energy_balance

_V_CLUB = np.array([50.0, 0.0, 0.0])
_N_SQUARE = np.array([1.0, 0.0, 0.0])


def _lofted_normal(loft_deg: float = 10.5) -> np.ndarray:
    loft = np.radians(loft_deg)
    return np.array([np.cos(loft), np.sin(loft), 0.0])


@pytest.mark.unit
@pytest.mark.physics
class TestSolveImpact:
    def test_solve_and_record(self) -> None:
        api = ImpactSolverAPI()
        post = api.solve_impact(0.0, _V_CLUB, _N_SQUARE)
        assert post.ball_velocity[0] > 60.0
        assert len(api.recorder.events) == 1

    def test_record_flag_false_skips_recording(self) -> None:
        api = ImpactSolverAPI()
        api.solve_impact(0.0, _V_CLUB, _N_SQUARE, record=False)
        assert api.recorder.events == []

    def test_offset_passthrough_reduces_ball_speed(self) -> None:
        api = ImpactSolverAPI()
        center = api.solve_impact(0.0, _V_CLUB, _N_SQUARE, record=False)
        toe = api.solve_impact(
            0.0,
            _V_CLUB,
            _N_SQUARE,
            impact_offset=np.array([0.02, 0.0]),
            record=False,
        )
        assert float(np.linalg.norm(toe.ball_velocity)) < float(
            np.linalg.norm(center.ball_velocity)
        )

    def test_negative_timestamp_rejected(self) -> None:
        api = ImpactSolverAPI()
        with pytest.raises(Exception, match="[Tt]imestamp"):
            api.solve_impact(-1.0, _V_CLUB, _N_SQUARE)

    def test_non_positive_mass_rejected(self) -> None:
        api = ImpactSolverAPI()
        with pytest.raises(ValueError, match="mass"):
            api.solve_impact(0.0, _V_CLUB, _N_SQUARE, clubhead_mass=0.0)


@pytest.mark.unit
@pytest.mark.physics
@pytest.mark.regression
class TestGearEffectSolveRegression:
    """Defect (a): the offset must reach the base-impulse computation."""

    def test_off_center_gear_solve_slower_than_center(self) -> None:
        api = ImpactSolverAPI()
        center = api.solve_impact(0.0, _V_CLUB, _lofted_normal(), record=False)
        toe = api.solve_with_gear_effect(
            0.0,
            _V_CLUB,
            _lofted_normal(),
            impact_offset=np.array([0.025, 0.0]),
            record=False,
        )
        assert float(np.linalg.norm(toe.ball_velocity)) < float(
            np.linalg.norm(center.ball_velocity)
        )

    def test_recorded_pre_state_carries_offset(self) -> None:
        """The recorded event must expose the offset it was solved with."""
        api = ImpactSolverAPI()
        offset = np.array([0.015, -0.005])
        api.solve_with_gear_effect(0.0, _V_CLUB, _lofted_normal(), impact_offset=offset)
        (event,) = api.recorder.events
        assert event.pre_state.impact_offset is not None
        np.testing.assert_allclose(event.pre_state.impact_offset, offset)

    def test_zero_offset_matches_plain_solve(self) -> None:
        api = ImpactSolverAPI()
        plain = api.solve_impact(0.0, _V_CLUB, _lofted_normal(), record=False)
        geared = api.solve_with_gear_effect(
            0.0,
            _V_CLUB,
            _lofted_normal(),
            impact_offset=np.zeros(2),
            record=False,
        )
        np.testing.assert_allclose(geared.ball_velocity, plain.ball_velocity)
        np.testing.assert_allclose(
            geared.ball_angular_velocity, plain.ball_angular_velocity, atol=1e-9
        )


@pytest.mark.unit
@pytest.mark.physics
class TestValidationAndReporting:
    def _api_with_impacts(self) -> ImpactSolverAPI:
        api = ImpactSolverAPI()
        for i, speed in enumerate((45.0, 50.0, 55.0)):
            api.solve_impact(float(i), speed * _N_SQUARE, _N_SQUARE)
        return api

    def test_cor_validation_within_tolerance(self) -> None:
        result = self._api_with_impacts().validate_cor_behavior(tolerance=0.05)
        assert result["valid"] is True
        assert result["num_samples"] == 3

    def test_spin_validation_square_strikes(self) -> None:
        result = self._api_with_impacts().validate_spin_behavior()
        assert result["valid"] is True

    def test_energy_report_totals(self) -> None:
        report = self._api_with_impacts().get_energy_report()
        assert len(report["impacts"]) == 3
        assert report["total_energy_lost"] > 0.0
        assert 0.0 < report["overall_loss_ratio"] < 1.0

    def test_reports_require_recorded_impacts(self) -> None:
        api = ImpactSolverAPI()
        with pytest.raises(RuntimeError, match="No impacts"):
            api.get_energy_report()
        with pytest.raises(RuntimeError, match="No impacts"):
            api.validate_cor_behavior()

    def test_reset_clears_recorder(self) -> None:
        api = self._api_with_impacts()
        api.reset()
        assert api.recorder.events == []

    def test_recorder_export_and_summary(self) -> None:
        api = self._api_with_impacts()
        exported = api.recorder.export_to_dict()
        assert exported["num_impacts"] == 3
        assert (
            exported["summary"]["max_ball_speed"]
            >= (exported["summary"]["mean_ball_speed"])
        )
        assert len(api.recorder.get_all_events()) == 3

    def test_model_type_selection(self) -> None:
        api = ImpactSolverAPI(model_type=ImpactModelType.FINITE_TIME)
        post = api.solve_impact(0.0, _V_CLUB, _N_SQUARE, record=False)
        assert post.contact_duration == pytest.approx(api.params.contact_duration)


@pytest.mark.unit
@pytest.mark.physics
class TestEnergyBalance:
    def test_energy_loss_matches_cor_expectation(self) -> None:
        """For a center strike the COM-frame loss is 1/2 mu v^2 (1-e^2)."""
        params = ImpactParameters(cor=0.83)
        pre = PreImpactState(
            clubhead_velocity=_V_CLUB.copy(),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=_N_SQUARE.copy(),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )
        post = RigidBodyImpactModel().solve(pre, params)
        balance = validate_energy_balance(pre, post, params)
        assert balance["energy_lost"] == pytest.approx(
            balance["expected_loss_j"], rel=1e-9
        )
        assert 0.0 < balance["energy_loss_ratio"] < 1.0

    def test_elastic_impact_conserves_energy(self) -> None:
        params = ImpactParameters(cor=1.0, friction_coefficient=0.0)
        pre = PreImpactState(
            clubhead_velocity=_V_CLUB.copy(),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=_N_SQUARE.copy(),
            ball_position=np.zeros(3),
            ball_velocity=np.zeros(3),
            ball_angular_velocity=np.zeros(3),
        )
        post = RigidBodyImpactModel().solve(pre, params)
        balance = validate_energy_balance(pre, post, params)
        assert balance["energy_lost"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
class TestRecorder:
    def test_ids_increment_and_reset(self) -> None:
        recorder = ImpactRecorder()
        assert recorder.get_summary() == {"num_impacts": 0}
        api = ImpactSolverAPI()
        api.solve_impact(0.0, _V_CLUB, _N_SQUARE)
        api.solve_impact(1.0, _V_CLUB, _N_SQUARE)
        ids = [e.impact_id for e in api.recorder.events]
        assert ids == [0, 1]
        api.recorder.reset()
        assert api.recorder.events == []
