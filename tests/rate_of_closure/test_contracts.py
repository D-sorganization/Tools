"""Contract tests: invalid inputs must fail loudly, valid edges must not.

House rule: ``TypeError`` for wrong types, ``ValueError`` (via the DbC
helpers) for out-of-range values.
"""

from __future__ import annotations

import pytest

from rate_of_closure.model import ImpactScenario, solve

pytestmark = pytest.mark.contract


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "clubhead_speed_mph": 110.0,
        "omega_plane_dps": 2000.0,
        "omega_shaft_dps": 1500.0,
        "lie_angle_deg": 58.0,
        "com_to_face_mm": 35.0,
        "impact_offset_toe_mm": 0.0,
        "impact_offset_high_mm": 0.0,
        "contact_duration_us": 450.0,
    }
    base.update(overrides)
    return base


class TestPreconditions:
    def test_speed_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="clubhead_speed_mph"):
            ImpactScenario(**_kwargs(clubhead_speed_mph=0.0))  # type: ignore[arg-type]

    def test_speed_must_be_finite(self) -> None:
        with pytest.raises(ValueError):
            ImpactScenario(**_kwargs(clubhead_speed_mph=float("nan")))  # type: ignore[arg-type]

    def test_lie_angle_bounded(self) -> None:
        with pytest.raises(ValueError, match="lie_angle_deg"):
            ImpactScenario(**_kwargs(lie_angle_deg=0.0))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lie_angle_deg"):
            ImpactScenario(**_kwargs(lie_angle_deg=91.0))  # type: ignore[arg-type]

    def test_com_offset_bounded(self) -> None:
        with pytest.raises(ValueError, match="com_to_face_mm"):
            ImpactScenario(**_kwargs(com_to_face_mm=-1.0))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="com_to_face_mm"):
            ImpactScenario(**_kwargs(com_to_face_mm=200.0))  # type: ignore[arg-type]

    def test_contact_duration_non_negative(self) -> None:
        with pytest.raises(ValueError, match="contact_duration_us"):
            ImpactScenario(**_kwargs(contact_duration_us=-1.0))  # type: ignore[arg-type]

    def test_angular_velocities_bounded_to_physical_range(self) -> None:
        with pytest.raises(ValueError, match="omega_shaft_dps"):
            ImpactScenario(**_kwargs(omega_shaft_dps=20_001.0))  # type: ignore[arg-type]


class TestValidEdges:
    def test_negative_rates_are_valid(self) -> None:
        """An opening face (negative closure) is physical and allowed."""
        result = solve(
            ImpactScenario(**_kwargs(omega_shaft_dps=-1500.0))  # type: ignore[arg-type]
        )
        assert result.path_deviation_deg > 0.0

    def test_boundary_lie_angles_accepted(self) -> None:
        for lie in (10.0, 90.0):
            solve(ImpactScenario(**_kwargs(lie_angle_deg=lie)))  # type: ignore[arg-type]


class TestPostconditions:
    def test_all_outputs_finite(self) -> None:
        result = solve(ImpactScenario(**_kwargs()))
        for name in (
            "path_deviation_deg",
            "aoa_deviation_deg",
            "speed_delta_mph",
            "tangential_speed_mph",
            "closure_during_contact_deg",
            "loft_gain_during_contact_deg",
        ):
            value = getattr(result, name)
            assert value == value, name  # not NaN
            assert abs(value) != float("inf"), name
