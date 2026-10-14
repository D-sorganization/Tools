"""Time cap during restitution: the ball is rebounding but has not separated.

`test_termination_9547.py` truncates a strike while it is still loading. The
other incomplete case is the time cap landing after peak compression and before
separation: compression is already falling and the relative normal velocity is
already separating, yet the contact is unfinished, so the ball must not be
exported as a post-impact state. Tools #5615 / Tools #5088 /
UpstreamDrift #9547.

No times are hard-coded: the peak-compression and separation times come from a
completed reference run, and the cap is their midpoint.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from shared.python.swing_sim.impact_interval import (
    ImpactIntervalResult,
    ImpactTermination,
    IncompleteContactError,
    solve_impact_interval,
)

from .test_solver import _club, _config, _initial

pytestmark = pytest.mark.unit


def _solve(**overrides: float) -> ImpactIntervalResult:
    config = dataclasses.replace(_config(), **overrides)
    return solve_impact_interval(initial=_initial(), club=_club(), config=config)


def _contact_duration_s(result: ImpactIntervalResult) -> float:
    """Time from the first loaded sample to the last compressed sample."""
    loaded = np.nonzero(result.compression_m > 0.0)[0]
    return float(result.time_s[loaded[-1]] - result.time_s[loaded[0]])


def _ball_speed_mps(result: ImpactIntervalResult) -> float:
    return float(np.linalg.norm(result.ball_velocity_mps[-1]))


def _relative_normal_velocity_mps(result: ImpactIntervalResult) -> np.ndarray:
    """(v_ball - v_club) . n; positive means the surfaces are separating."""
    relative = result.ball_velocity_mps - result.club_velocity_mps
    return np.einsum("ij,ij->i", relative, result.contact_normal)


class TestTimeCapDuringRestitution:
    @pytest.fixture(scope="class")
    def reference(self) -> ImpactIntervalResult:
        result = _solve()
        assert result.termination is ImpactTermination.SEPARATED
        return result

    @pytest.fixture(scope="class")
    def rebounding(self, reference: ImpactIntervalResult) -> ImpactIntervalResult:
        peak_time_s = float(reference.time_s[int(np.argmax(reference.compression_m))])
        separation_time_s = float(reference.time_s[-1])
        assert peak_time_s < separation_time_s
        return _solve(maximum_time_s=0.5 * (peak_time_s + separation_time_s))

    def test_cap_is_strictly_between_peak_and_separation(
        self, reference: ImpactIntervalResult, rebounding: ImpactIntervalResult
    ) -> None:
        """Guard the fixture: the derived cap really lands in restitution."""
        peak_time_s = float(reference.time_s[int(np.argmax(reference.compression_m))])
        cap_s = 0.5 * (peak_time_s + float(reference.time_s[-1]))
        assert peak_time_s < cap_s < float(reference.time_s[-1])
        assert rebounding.time_s[-1] > peak_time_s

    def test_reports_the_time_limit_not_separation(
        self, rebounding: ImpactIntervalResult
    ) -> None:
        assert rebounding.termination is ImpactTermination.TIME_LIMIT
        assert rebounding.contact_completed is False

    def test_trace_is_past_peak_compression_and_still_falling(
        self, rebounding: ImpactIntervalResult
    ) -> None:
        compression = rebounding.compression_m
        peak = int(np.argmax(compression))
        assert 0 < peak < compression.size - 1, "peak must be interior to the trace"
        assert compression[-1] < compression[peak]
        assert compression[-1] > 0.0, "ball must not have separated yet"
        assert compression[-1] < compression[-2], "compression must still be falling"
        assert np.all(np.diff(compression[peak:]) < 0.0)

    def test_surfaces_are_separating_but_still_in_contact(
        self, rebounding: ImpactIntervalResult
    ) -> None:
        """Rebounding (v_n > 0) with a live normal force: restitution, not flight."""
        assert _relative_normal_velocity_mps(rebounding)[-1] > 0.0
        assert rebounding.normal_force_n[-1] > 0.0

    def test_refuses_to_export_a_rebounding_ball(
        self, rebounding: ImpactIntervalResult
    ) -> None:
        with pytest.raises(IncompleteContactError) as excinfo:
            rebounding.to_post_impact_state()
        assert excinfo.value.termination is ImpactTermination.TIME_LIMIT
        assert excinfo.value.result is rebounding

    def test_final_time_respects_the_cap(self, reference: ImpactIntervalResult) -> None:
        peak_time_s = float(reference.time_s[int(np.argmax(reference.compression_m))])
        cap_s = 0.5 * (peak_time_s + float(reference.time_s[-1]))
        result = _solve(maximum_time_s=cap_s)
        assert result.time_s[-1] <= cap_s + 1.0e-15


class TestTimestepRefinement:
    """Halving dt on a completed hit must not change the physical outcome.

    Measured on the `_config()` strike (dt = 1e-7 s, halved to 5e-8 s, then
    2.5e-8 s and 1.25e-8 s):

        dt        contact duration   ball speed
        1.0e-7    8.590e-5 s         74.65620 m/s
        5.0e-8    8.595e-5 s         74.65280 m/s
        2.5e-8    8.5975e-5 s        74.65110 m/s
        1.25e-8   8.59875e-5 s       74.65026 m/s

    Contact duration moves by exactly dt/2 per halving (the end of contact is
    resolved to one sample), so the duration tolerance is one coarse step, 1e-7
    s (0.12 % of the duration). Ball speed converges first order (successive
    differences 3.4e-3, 1.7e-3, 0.84e-3 m/s); the 1e-4 relative tolerance is
    about 2x the measured 4.6e-5 first-halving change.
    """

    COARSE_DT_S = 1.0e-7
    SPEED_RTOL = 1.0e-4

    def test_halving_dt_leaves_duration_and_speed_converged(self) -> None:
        coarse = _solve(time_step_s=self.COARSE_DT_S)
        fine = _solve(time_step_s=0.5 * self.COARSE_DT_S)
        assert coarse.contact_completed and fine.contact_completed
        assert (
            abs(_contact_duration_s(fine) - _contact_duration_s(coarse))
            <= self.COARSE_DT_S
        )
        assert _ball_speed_mps(fine) == pytest.approx(
            _ball_speed_mps(coarse), rel=self.SPEED_RTOL
        )

    def test_speed_error_shrinks_under_refinement(self) -> None:
        """Convergence, not coincidence: the next halving moves the answer less."""
        speeds = [
            _ball_speed_mps(_solve(time_step_s=self.COARSE_DT_S * 0.5**k))
            for k in range(3)
        ]
        assert abs(speeds[2] - speeds[1]) < abs(speeds[1] - speeds[0])
