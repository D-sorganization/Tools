"""Independent termination, unilateral-contact and energy gates for IA-T2."""

import math
from dataclasses import replace

import pytest

from shared.python.golf_club.impact_coupling import (
    CoupledImpactConfig,
    GripBoundary,
    simulate_coupled_impact,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _config() -> CoupledImpactConfig:
    return CoupledImpactConfig(
        head_mass_kg=0.2,
        head_speed_mps=40.0,
        shaft_stiffness_n_m=0.0,
        grip=GripBoundary(3.0, 5e4, 0.0, "synthetic analytic fixture"),
        contact_damping_n_s_m=40.0,
        dt_s=2e-6,
    )


def test_incomplete_collision_is_not_returned_as_success() -> None:
    with pytest.raises(RuntimeError, match="separation"):
        simulate_coupled_impact(replace(_config(), max_time_s=1e-5))


def test_underresolved_contact_step_is_refused() -> None:
    with pytest.raises(ValueError, match="dt_s"):
        simulate_coupled_impact(replace(_config(), dt_s=1e-3))


def test_undamped_detached_contact_matches_closed_form() -> None:
    from shared.python.golf_club.impact_coupling_audit import audit_coupled_impact

    config = replace(_config(), contact_damping_n_s_m=0.0)
    audit = audit_coupled_impact(config)
    reduced = 1 / (1 / config.ball_mass_kg + 1 / config.head_mass_kg)
    duration = math.pi * math.sqrt(reduced / config.contact_stiffness_n_m)
    expected = 2 * reduced * config.head_speed_mps / config.ball_mass_kg
    assert audit.clearance_time_s == pytest.approx(duration, rel=2e-7)
    assert audit.force_release_time_s == pytest.approx(duration, rel=2e-7)
    assert audit.ball_velocity_mps + config.head_speed_mps == pytest.approx(
        expected, rel=2e-7
    )
    assert audit.energy.total_dissipation_j == pytest.approx(0, abs=1e-10)
    assert abs(audit.energy.residual_j) < 1e-7


def test_force_release_and_clearance_are_distinct_for_clipped_contact() -> None:
    from shared.python.golf_club.impact_coupling_audit import audit_coupled_impact

    config = _config()
    audit = audit_coupled_impact(config)
    reduced = 1 / (1 / config.ball_mass_kg + 1 / config.head_mass_kg)
    stiffness, damping = config.contact_stiffness_n_m, config.contact_damping_n_s_m
    alpha = damping / (2 * reduced)
    omega = math.sqrt(stiffness / reduced - alpha**2)
    release = math.atan2(damping * omega, damping * alpha - stiffness) / omega
    overlap = (
        config.head_speed_mps
        * math.exp(-alpha * release)
        * math.sin(omega * release)
        / omega
    )
    rate = (
        config.head_speed_mps
        * math.exp(-alpha * release)
        * (math.cos(omega * release) - alpha / omega * math.sin(omega * release))
    )
    assert audit.force_release_time_s == pytest.approx(release, rel=2e-6)
    assert audit.clearance_time_s == pytest.approx(release - overlap / rate, rel=2e-6)
    assert audit.force_release_time_s < audit.clearance_time_s
    assert audit.energy.contact_cutoff_dissipation_j == pytest.approx(
        0.5 * stiffness * overlap**2, rel=2e-5
    )
    assert abs(audit.energy.residual_j) < 2e-6


def test_preloaded_damped_chain_has_complete_passive_energy_ledger() -> None:
    from shared.python.golf_club.impact_coupling_audit import (
        CoupledImpactInitialState,
        audit_coupled_impact,
    )

    config = replace(
        _config(),
        shaft_stiffness_n_m=2e5,
        shaft_damping_n_s_m=50.0,
        grip=GripBoundary(3.0, 5e4, 80.0, "synthetic preload fixture"),
    )
    initial = CoupledImpactInitialState(grip_displacement_m=0.01, grip_velocity_mps=0.3)
    audit = audit_coupled_impact(config, initial_state=initial)
    expected = (
        0.5 * config.ball_mass_kg * config.head_speed_mps**2
        + 0.5 * config.grip.effective_mass_kg * 0.3**2
        + 0.5 * (config.shaft_stiffness_n_m + config.grip.stiffness_n_m) * 0.01**2
    )
    assert audit.energy.initial_energy_j == pytest.approx(expected)
    assert audit.energy.shaft_dissipation_j > 0
    assert audit.energy.grip_dissipation_j > 0
    assert audit.energy.contact_damping_dissipation_j > 0
    assert audit.energy.total_dissipation_j > 0
    assert audit.energy.external_work_j == 0
    assert abs(audit.energy.residual_j) < 2e-6
    assert audit.terminal_overlap_m == pytest.approx(0, abs=1e-9)


def test_legacy_result_uses_geometric_clearance_and_retained_energy() -> None:
    from shared.python.golf_club.impact_coupling_audit import audit_coupled_impact

    config = _config()
    result = simulate_coupled_impact(config)
    audit = audit_coupled_impact(config)
    assert result.contact_time_s == pytest.approx(audit.clearance_time_s)
    assert result.energy_balance_fraction == pytest.approx(
        audit.energy.final_energy_j / audit.energy.initial_energy_j
    )
    assert result.energy_balance_fraction < 1


def test_step_refinement_preserves_impulse_and_energy() -> None:
    from shared.python.golf_club.impact_coupling_audit import audit_coupled_impact

    coarse = audit_coupled_impact(_config())
    fine = audit_coupled_impact(replace(_config(), dt_s=5e-7))
    assert coarse.contact_impulse_n_s == pytest.approx(
        fine.contact_impulse_n_s, rel=2e-6
    )
    assert coarse.clearance_time_s == pytest.approx(fine.clearance_time_s, rel=2e-6)
    assert abs(fine.energy.residual_j) < 2e-6


def test_initial_state_contract_refuses_nonfinite_or_boolean_values() -> None:
    from shared.python.golf_club.impact_coupling_audit import CoupledImpactInitialState

    with pytest.raises(ValueError):
        CoupledImpactInitialState(grip_displacement_m=math.nan)
    with pytest.raises(TypeError):
        CoupledImpactInitialState(head_velocity_mps=True)


def test_energy_ledger_refuses_invalid_physical_values() -> None:
    from shared.python.golf_club.impact_coupling_audit import CoupledImpactEnergyLedger

    with pytest.raises(ValueError):
        CoupledImpactEnergyLedger(1.0, 0.5, -0.1, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        CoupledImpactEnergyLedger(math.inf, 0.5, 0.1, 0.0, 0.0, 0.0)


def test_legacy_stiff_fixture_requires_more_than_five_milliseconds() -> None:
    from shared.python.golf_club.impact_coupling_audit import audit_coupled_impact

    config = CoupledImpactConfig(
        0.2,
        45.0,
        50000.0,
        GripBoundary(2.5, 50000.0, 50.0, "synthetic fixture"),
        dt_s=2e-6,
    )
    with pytest.raises(RuntimeError, match="separation"):
        audit_coupled_impact(config)
    result = audit_coupled_impact(replace(config, max_time_s=0.01))
    assert result.clearance_time_s > config.max_time_s
    assert result.force_release_time_s < config.max_time_s
    assert abs(result.energy.residual_j) < 1e-6


@pytest.mark.parametrize(
    "mechanism,lower,upper",
    [
        ("relaxed", 12.0, 20.0),
        ("damped", 3.0, 5.5),
        ("preloaded", 3.0, 5.5),
    ],
)
def test_short_contact_scaling_distinguishes_spring_damper_and_preload(
    mechanism: str,
    lower: float,
    upper: float,
) -> None:
    from shared.python.golf_club.impact_coupling_audit import (
        CoupledImpactInitialState,
        audit_coupled_impact,
    )

    changes = []
    for contact_stiffness in (4e6, 64e6):
        config = replace(
            _config(),
            contact_stiffness_n_m=contact_stiffness,
            contact_damping_n_s_m=0.0,
            shaft_stiffness_n_m=1e4,
            shaft_damping_n_s_m=5.0 if mechanism == "damped" else 0.0,
            dt_s=2e-7,
        )
        initial = CoupledImpactInitialState(
            grip_displacement_m=0.01 if mechanism == "preloaded" else 0.0,
        )
        coupled = audit_coupled_impact(config, initial_state=initial)
        detached = audit_coupled_impact(
            replace(config, shaft_stiffness_n_m=0, shaft_damping_n_s_m=0),
            initial_state=initial,
        )
        changes.append(abs(coupled.ball_velocity_mps - detached.ball_velocity_mps))
    assert lower < changes[0] / changes[1] < upper


@pytest.mark.parametrize("head_velocity", [1e20, 1e200])
def test_unrepresentable_initial_relative_velocity_is_refused(
    head_velocity: float,
) -> None:
    from shared.python.golf_club.impact_coupling_audit import (
        CoupledImpactInitialState,
        audit_coupled_impact,
    )

    with pytest.raises(ValueError, match="relative velocity"):
        audit_coupled_impact(
            _config(),
            initial_state=CoupledImpactInitialState(head_velocity_mps=head_velocity),
        )
