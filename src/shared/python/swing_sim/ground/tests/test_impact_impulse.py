"""Analytic and property coverage for the sphere-plane impulse law."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace

import pytest

from shared.python.swing_sim.ground import (
    GroundContactState,
    ImpactRejectionReason,
    ImpactStateError,
    SphereProperties,
    resolve_sphere_plane_impact,
)

from ._support import _contact, _request, _surface

_GOLDEN = (
    __import__("pathlib").Path(__file__).parents[5]
    / "rate_of_closure/web/src/model/__fixtures__/ground_impact_bounce_golden_v1.json"
)


def _state(
    velocity: tuple[float, float, float],
    spin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GroundContactState:
    return replace(
        _contact(),
        position_m=(0.0, _request().ball_radius_m, 0.0),
        velocity_m_s=velocity,
        angular_velocity_rad_s=spin,
    )


def _solve(
    state: GroundContactState,
    *,
    restitution: float = 0.5,
    static_friction: float = 0.0,
    kinetic_friction: float = 0.0,
):
    request = _request()
    surface = replace(
        _surface(),
        normal_restitution=restitution,
        static_friction=static_friction,
        kinetic_friction=kinetic_friction,
    )
    return resolve_sphere_plane_impact(
        state,
        surface,
        SphereProperties(
            request.ball_radius_m,
            request.ball_mass_kg,
            request.rotational_inertia_factor,
        ),
    )


def test_elastic_frictionless_impact_reverses_normal_and_conserves_energy() -> None:
    result = _solve(_state((3.0, -4.0, 1.0)), restitution=1.0)

    assert result.state_after.velocity_m_s == pytest.approx((3.0, 4.0, 1.0))
    assert result.state_after.angular_velocity_rad_s == (0.0, 0.0, 0.0)
    assert result.energy.dissipation_j == pytest.approx(0.0, abs=1e-12)
    assert result.energy.kinetic_after_j == pytest.approx(
        result.energy.kinetic_before_j
    )


def test_perfectly_inelastic_normal_impact_captures_normal_motion() -> None:
    result = _solve(_state((0.0, -6.0, 0.0)), restitution=0.0)

    assert result.state_after.velocity_m_s[1] == pytest.approx(0.0, abs=1e-12)
    expected_loss = 0.5 * _request().ball_mass_kg * 6.0**2
    assert result.energy.dissipation_j == pytest.approx(expected_loss)


def test_high_friction_matches_closed_form_sticking_impulse() -> None:
    request = _request()
    result = _solve(
        _state((2.0, -3.0, 0.0)),
        restitution=0.5,
        static_friction=5.0,
        kinetic_friction=4.0,
    )
    expected_tangent = -request.ball_mass_kg * 0.4 / 1.4 * 2.0

    assert result.regime.value == "sticking"
    assert result.tangential_impulse_n_s[0] == pytest.approx(expected_tangent)
    assert math.hypot(
        result.contact_velocity_after_m_s[0], result.contact_velocity_after_m_s[2]
    ) == pytest.approx(0.0, abs=2e-12)
    assert result.friction_utilization <= 1.0


def test_exact_coulomb_boundary_is_classified_as_sticking() -> None:
    request = _request()
    desired = request.ball_mass_kg * 0.4 / 1.4
    normal_impulse = request.ball_mass_kg * 2.0
    coefficient = desired / normal_impulse
    result = _solve(
        _state((1.0, -2.0, 0.0)),
        restitution=0.0,
        static_friction=coefficient,
        kinetic_friction=coefficient,
    )

    assert result.regime.value == "sticking"
    assert result.friction_utilization == pytest.approx(1.0)


def test_sliding_impulse_couples_spin_without_reversing_slip() -> None:
    result = _solve(
        _state((8.0, -2.0, 0.0)),
        restitution=0.4,
        static_friction=0.1,
        kinetic_friction=0.08,
    )

    assert result.regime.value == "sliding"
    assert result.state_after.angular_velocity_rad_s[2] < 0.0
    assert result.contact_velocity_before_m_s[0] > 0.0
    assert result.contact_velocity_after_m_s[0] >= -1e-12
    assert math.hypot(*result.tangential_impulse_n_s[::2]) == pytest.approx(
        0.08 * result.normal_impulse_n_s
    )


def test_pure_spin_generates_tangential_impulse_and_linear_velocity() -> None:
    result = _solve(
        _state((0.0, -1.0, 0.0), (0.0, 0.0, 100.0)),
        restitution=0.2,
        static_friction=0.5,
        kinetic_friction=0.4,
    )

    assert result.tangential_impulse_n_s[0] < 0.0
    assert result.state_after.velocity_m_s[0] < 0.0
    assert result.state_after.angular_velocity_rad_s[2] < 100.0


def test_moving_surface_energy_gain_is_bounded_by_boundary_work() -> None:
    surface = replace(
        _surface(),
        surface_velocity_m_s=(20.0, 0.0, 0.0),
        normal_restitution=0.0,
        static_friction=5.0,
        kinetic_friction=4.0,
    )
    request = _request()
    result = resolve_sphere_plane_impact(
        _state((0.0, -1.0, 0.0)),
        surface,
        SphereProperties(
            request.ball_radius_m,
            request.ball_mass_kg,
            request.rotational_inertia_factor,
        ),
    )

    assert result.energy.boundary_work_j > 0.0
    available_energy = result.energy.kinetic_before_j + result.energy.boundary_work_j
    assert result.energy.kinetic_after_j <= available_energy + 1e-10
    assert result.energy.dissipation_j >= 0.0


def test_tilted_normal_preserves_restitution_and_friction_cone() -> None:
    root_half = math.sqrt(0.5)
    surface = replace(
        _surface(),
        normal_unit=(0.0, root_half, root_half),
        normal_restitution=0.6,
        static_friction=0.3,
        kinetic_friction=0.2,
    )
    request = _request()
    state = _state((4.0, -3.0 * root_half, -3.0 * root_half))
    result = resolve_sphere_plane_impact(
        state,
        surface,
        SphereProperties(
            request.ball_radius_m,
            request.ball_mass_kg,
            request.rotational_inertia_factor,
        ),
    )

    before_normal = sum(
        a * b
        for a, b in zip(
            result.contact_velocity_before_m_s, surface.normal_unit, strict=True
        )
    )
    after_normal = sum(
        a * b
        for a, b in zip(
            result.contact_velocity_after_m_s, surface.normal_unit, strict=True
        )
    )
    tangent_magnitude = math.sqrt(
        sum(value * value for value in result.tangential_impulse_n_s)
    )
    assert after_normal == pytest.approx(-0.6 * before_normal)
    assert (
        tangent_magnitude <= surface.static_friction * result.normal_impulse_n_s + 1e-12
    )


@pytest.mark.parametrize(
    ("velocity", "reason"),
    [
        ((0.0, 0.0, 0.0), ImpactRejectionReason.GRAZING),
        ((0.0, 1.0, 0.0), ImpactRejectionReason.OUTGOING),
    ],
)
def test_grazing_and_outgoing_states_are_rejected(velocity, reason) -> None:
    with pytest.raises(ImpactStateError) as caught:
        _solve(_state(velocity))
    assert caught.value.reason is reason


def test_nonfinite_and_invalid_physical_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="mass"):
        SphereProperties(0.02, math.nan, 0.4)
    with pytest.raises(ValueError, match="restitution"):
        resolve_sphere_plane_impact(
            _state((0.0, -1.0, 0.0)),
            _surface(),
            SphereProperties(0.02, 0.04, 0.4),
            normal_restitution=1.1,
        )


def test_shared_golden_fixture_matches_canonical_analytic_impact() -> None:
    payload = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    case = payload["impact_case"]
    expected = case["expected"]
    surface = replace(
        _surface(),
        normal_restitution=case["normal_restitution"],
        static_friction=case["static_friction"],
        kinetic_friction=case["kinetic_friction"],
    )
    state = replace(
        _contact(),
        position_m=(0.0, case["radius_m"], 0.0),
        velocity_m_s=tuple(case["velocity_before_m_s"]),
        angular_velocity_rad_s=tuple(case["angular_velocity_before_rad_s"]),
    )
    result = resolve_sphere_plane_impact(
        state,
        surface,
        SphereProperties(
            case["radius_m"],
            case["mass_kg"],
            case["rotational_inertia_factor"],
        ),
    )

    assert result.regime.value == expected["regime"]
    assert result.normal_impulse_n_s == pytest.approx(expected["normal_impulse_n_s"])
    assert result.tangential_impulse_n_s == pytest.approx(
        expected["tangential_impulse_n_s"]
    )
    assert result.state_after.velocity_m_s == pytest.approx(
        expected["velocity_after_m_s"]
    )
    assert result.state_after.angular_velocity_rad_s == pytest.approx(
        expected["angular_velocity_after_rad_s"]
    )
    assert result.energy.dissipation_j == pytest.approx(expected["dissipation_j"])


def test_shared_golden_fixture_bytes_are_version_locked() -> None:
    digest = hashlib.sha256(_GOLDEN.read_bytes()).hexdigest()

    expected = "5831a8f8bf0fe18edf76c985503fdfe784df8e0eed89f85deb9c27085fb9f059"  # pragma: allowlist secret  # noqa: E501
    assert digest == expected


@pytest.mark.parametrize("tangent_speed", [0.0, 0.25, 1.0, 5.0, 25.0])
def test_impact_properties_remain_passive_and_inside_coulomb_cone(
    tangent_speed: float,
) -> None:
    result = _solve(
        _state((tangent_speed, -3.0, tangent_speed / 3.0)),
        restitution=0.55,
        static_friction=0.45,
        kinetic_friction=0.3,
    )
    impulse_tangent = math.sqrt(
        sum(value * value for value in result.tangential_impulse_n_s)
    )

    assert result.energy.dissipation_j >= 0.0
    assert impulse_tangent <= 0.45 * result.normal_impulse_n_s + 1e-12


def test_passivity_guard_rejects_an_artificial_energy_creating_state() -> None:
    from shared.python.swing_sim.ground import impact_impulse

    before = _state((0.0, -1.0, 0.0))
    after = replace(before, velocity_m_s=(100.0, 100.0, 100.0))
    surface = _surface()
    body = SphereProperties(0.1, 1.0, 0.4)
    solution = impact_impulse._ImpactSolution(
        before,
        surface,
        body,
        (0.0, -0.1, 0.0),
        (0.0, -1.0, 0.0),
        0.5,
        1.5,
        (0.0, 0.0, 0.0),
        impact_impulse._REGIME_STICKING,
    )
    with pytest.raises(ValueError, match="passive energy"):
        impact_impulse._energy_ledger(
            solution,
            after,
            (0.0, 0.0, 0.0),
        )
