"""Contact completion must be reported, and unfinished contact must not export.

`solve_impact_interval` promises "the separated post-impact state when contact
occurs", but it leaves its integration loop by two different routes -- physical
separation, and exhausting the step budget -- and records neither. A caller
therefore cannot tell a finished impact from a truncated one, and
`to_post_impact_state()` will happily convert a ball that is still compressed
under tens of kilonewtons into a "post-impact" velocity.

Reproduced against the pre-fix solver with `maximum_time_s=1e-5`:

    final time         = 1.01e-05   (over the 1e-05 cap)
    final compression  = 0.0004830433652168 m   (still compressed)
    final normal force = 31374.6 N
    post-impact ball v = [4.36572246 0. 0.]

Downstream flight and optimisation consume that velocity as if the strike had
completed. Tools #4130 / UpstreamDrift #9547.
"""

from __future__ import annotations

import dataclasses

import pytest

from shared.python.swing_sim.impact_interval import (
    ImpactIntervalResult,
    ImpactTermination,
    IncompleteContactError,
    solve_impact_interval,
)

from .test_solver import _club, _config, _initial

pytestmark = pytest.mark.unit


def _truncated() -> ImpactIntervalResult:
    """A strike cut off mid-compression by the time cap."""
    config = dataclasses.replace(_config(), maximum_time_s=1.0e-5)
    return solve_impact_interval(initial=_initial(), club=_club(), config=config)


def _completed() -> ImpactIntervalResult:
    """A strike that runs to separation under the default cap."""
    return solve_impact_interval(initial=_initial(), club=_club(), config=_config())


class TestTerminationReason:
    def test_completed_contact_reports_separation(self) -> None:
        result = _completed()
        assert result.termination is ImpactTermination.SEPARATED
        assert result.contact_completed is True

    def test_truncated_contact_reports_the_time_limit(self) -> None:
        """The distinguishing case: still loaded when the budget ran out."""
        result = _truncated()
        assert result.compression_m[-1] > 0.0, "fixture must still be compressed"
        assert result.normal_force_n[-1] > 0.0, "fixture must still be loaded"
        assert result.termination is ImpactTermination.TIME_LIMIT
        assert result.contact_completed is False

    def test_no_contact_is_distinct_from_a_timeout(self) -> None:
        """A miss and a truncation are different facts, not one 'no result'."""
        initial = dataclasses.replace(
            _initial(), club_velocity_mps=_initial().club_velocity_mps * 0.0
        )
        result = solve_impact_interval(
            initial=initial,
            club=_club(),
            config=dataclasses.replace(_config(), maximum_time_s=1.0e-5),
        )
        assert result.did_contact is False
        assert result.termination is ImpactTermination.NO_CONTACT
        assert result.contact_completed is False


class TestPostImpactExportRejectsIncompleteContact:
    def test_truncated_contact_refuses_to_export(self) -> None:
        """The defect: this returned a velocity for a still-compressed ball."""
        result = _truncated()
        with pytest.raises(IncompleteContactError) as excinfo:
            result.to_post_impact_state()
        message = str(excinfo.value)
        assert "TIME_LIMIT" in message or "time limit" in message.lower()

    def test_the_error_carries_the_partial_trace_for_inspection(self) -> None:
        """Refusing to export must not throw the evidence away."""
        result = _truncated()
        with pytest.raises(IncompleteContactError) as excinfo:
            result.to_post_impact_state()
        assert excinfo.value.result is result
        assert excinfo.value.termination is ImpactTermination.TIME_LIMIT

    def test_completed_contact_still_exports(self) -> None:
        """The fix must not block the normal path."""
        state = _completed().to_post_impact_state()
        assert state.ball_velocity.shape == (3,)
        assert state.ball_velocity[0] > 0.0


class TestTimeCapIsRespected:
    def test_final_time_does_not_exceed_the_configured_cap(self) -> None:
        """`ceil(T/dt) + 1` steps overshot the cap by a whole step."""
        config = dataclasses.replace(_config(), maximum_time_s=1.0e-5)
        result = solve_impact_interval(initial=_initial(), club=_club(), config=config)
        assert result.time_s[-1] <= config.maximum_time_s + 1.0e-15

    def test_a_noninteger_step_count_still_respects_the_cap(self) -> None:
        """T/dt = 15.5 steps: the final step may not run past the budget."""
        config = dataclasses.replace(
            _config(), time_step_s=2.0e-7, maximum_time_s=3.1e-6
        )
        result = solve_impact_interval(initial=_initial(), club=_club(), config=config)
        assert result.time_s[-1] <= config.maximum_time_s + 1.0e-15
