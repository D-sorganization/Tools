"""Analytic and consistency gates for the coupled impact model (H1, #4563).

Written before the implementation (TDD): every assertion here is either a
closed-form limit, a consistency requirement against the shipped
Kelvin-Voigt impact model, or a published-band claim — never a pin on the
implementation's own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.golf_club.impact_coupling import (
    CoupledImpactConfig,
    GripBoundary,
    simulate_coupled_impact,
)
from shared.python.swing_sim.impact.constants import (
    DRIVER_MASS_KG,
    GOLF_BALL_MASS_KG,
)
from shared.python.swing_sim.impact.models import SpringDamperImpactModel
from shared.python.swing_sim.impact.types import ImpactParameters, PreImpactState

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_HEAD_SPEED_MPS = 44.0
_PHYSIOLOGICAL_GRIP = GripBoundary(
    effective_mass_kg=3.0,
    stiffness_n_m=5.0e4,
    damping_n_s_m=50.0,
    provenance="literature: hand+forearm effective mass 2-4 kg, grip ~1e4-1e5 N/m",
)


def _config(**overrides: object) -> CoupledImpactConfig:
    base: dict[str, object] = {
        "head_mass_kg": DRIVER_MASS_KG,
        "head_speed_mps": _HEAD_SPEED_MPS,
        "shaft_stiffness_n_m": 200.0,
        "shaft_damping_n_s_m": 0.0,
        "grip": _PHYSIOLOGICAL_GRIP,
    }
    base.update(overrides)
    return CoupledImpactConfig(**base)  # type: ignore[arg-type]


def _free_head_reference_speed() -> float:
    """Ball exit speed from the shipped Kelvin-Voigt model, same contact."""
    model = SpringDamperImpactModel()
    pre = PreImpactState(
        clubhead_velocity=np.array([_HEAD_SPEED_MPS, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([1.0, 0.0, 0.0]),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=DRIVER_MASS_KG,
    )
    post = model.solve(pre, ImpactParameters())
    return float(np.linalg.norm(post.ball_velocity))


class TestLimits:
    def test_detached_shaft_reproduces_the_shipped_impact_model(self) -> None:
        """k_s = 0 must agree with SpringDamperImpactModel for the same
        contact parameters — same physics, two implementations."""
        result = simulate_coupled_impact(_config(shaft_stiffness_n_m=0.0))
        assert result.ball_speed_mps == pytest.approx(
            _free_head_reference_speed(), rel=1e-3
        )

    def test_welded_rigid_limit_is_bounded_and_monotone_in_grip_stiffness(
        self,
    ) -> None:
        """Stiffer coupling monotonically raises ball speed toward the
        infinite-effective-mass bound (1+e)·v_head, never beyond it."""
        free = simulate_coupled_impact(_config(shaft_stiffness_n_m=0.0))
        speeds = []
        for k_g in (1.0e3, 1.0e5, 1.0e7, 1.0e9):
            grip = GripBoundary(
                effective_mass_kg=80.0,
                stiffness_n_m=k_g,
                damping_n_s_m=0.0,
                provenance="welded-limit sweep",
            )
            result = simulate_coupled_impact(
                _config(shaft_stiffness_n_m=1.0e9, shaft_damping_n_s_m=0.0, grip=grip)
            )
            speeds.append(result.ball_speed_mps)
        assert all(b >= a for a, b in zip(speeds, speeds[1:], strict=False))
        assert speeds[0] >= free.ball_speed_mps - 1e-9
        # Hard bound: the perfectly elastic infinite-mass ceiling 2*v0.
        # The free-head emergent restitution cannot be reused here: KV
        # restitution depends on the reduced mass (zeta = c/(2*sqrt(k*mu))),
        # and welding changes mu from m_b*m_h/(m_b+m_h) to m_b - so the
        # welded case is legitimately bouncier than (1+e_free)*v0.
        assert speeds[-1] < 2.0 * _HEAD_SPEED_MPS
        # And it must still exceed the free case by a finite margin (the
        # added-mass effect is real, not numerical noise).
        assert speeds[-1] > free.ball_speed_mps + 0.5

    def test_energy_is_conserved_without_damping(self) -> None:
        grip = GripBoundary(
            effective_mass_kg=3.0,
            stiffness_n_m=5.0e4,
            damping_n_s_m=0.0,
            provenance="conservative variant",
        )
        result = simulate_coupled_impact(_config(grip=grip, contact_damping_n_s_m=0.0))
        # Body frame (documented): the fixed grip anchor does no work, so the
        # conserved total is the ball's initial kinetic energy.
        initial = 0.5 * GOLF_BALL_MASS_KG * _HEAD_SPEED_MPS**2
        assert result.energy_balance_fraction == pytest.approx(1.0, abs=5e-3)
        total = (
            result.ball_kinetic_energy_j
            + result.head_kinetic_energy_j
            + result.stored_spring_energy_j
            + result.grip_side_kinetic_energy_j
        )
        assert total == pytest.approx(initial, rel=5e-3)


class TestDecoupling:
    def test_physiological_hand_influence_is_sub_percent(self) -> None:
        """The quantified classical claim: with realistic grip stiffness and
        static shaft stiffness, hands change ball speed by well under 1%."""
        free = simulate_coupled_impact(_config(shaft_stiffness_n_m=0.0))
        coupled = simulate_coupled_impact(_config())
        influence = (
            abs(coupled.ball_speed_mps - free.ball_speed_mps) / free.ball_speed_mps
        )
        assert influence < 0.01

    def test_rigid_shaft_upper_bound_exceeds_the_realistic_case(self) -> None:
        """The reported worst case must dominate the realistic case."""
        free = simulate_coupled_impact(_config(shaft_stiffness_n_m=0.0))
        realistic = simulate_coupled_impact(_config())
        rigid = simulate_coupled_impact(_config(shaft_stiffness_n_m=1.0e9))
        real_influence = abs(realistic.ball_speed_mps - free.ball_speed_mps)
        rigid_influence = abs(rigid.ball_speed_mps - free.ball_speed_mps)
        assert rigid_influence >= real_influence

    def test_hand_influence_follows_the_tau_squared_decoupling_law(self) -> None:
        """The decoupling law, quantitatively: at *finite* shaft stiffness the
        transmitted influence scales as (contact time)² — quadrupling the
        contact duration multiplies the influence ~16x. (With a rigid shaft
        the coupling is quasi-static added mass and tau-independent, which is
        why this gate uses a finite k_s.)"""

        def influence(contact_stiffness: float) -> float:
            result = simulate_coupled_impact(
                _config(
                    shaft_stiffness_n_m=1.0e5,
                    contact_stiffness_n_m=contact_stiffness,
                    contact_damping_n_s_m=0.0,
                )
            )
            return abs(result.ball_speed_mps - result.free_head_ball_speed_mps)

        stiff_contact = influence(4.0e6)  # tau ~ 304 us
        soft_contact = influence(2.5e5)  # tau ~ 1223 us (4x)
        assert soft_contact > stiff_contact
        assert 10.0 < soft_contact / stiff_contact < 25.0  # ~ (4)^2

    def test_decoupling_fraction_reported_and_bounded(self) -> None:
        result = simulate_coupled_impact(_config())
        assert 0.0 <= result.decoupling_fraction <= 1.0
        assert result.decoupling_fraction > 0.99


class TestContracts:
    def test_invalid_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _config(head_mass_kg=0.0)
        with pytest.raises(ValueError):
            _config(head_speed_mps=-1.0)
        with pytest.raises(ValueError):
            _config(shaft_stiffness_n_m=-5.0)
        with pytest.raises(ValueError):
            GripBoundary(
                effective_mass_kg=0.0,
                stiffness_n_m=1.0,
                damping_n_s_m=0.0,
                provenance="bad",
            )

    def test_wrong_types_are_refused(self) -> None:
        with pytest.raises(TypeError):
            simulate_coupled_impact("nope")  # type: ignore[arg-type]


class TestCouplingReport:
    def test_report_is_deterministic_versioned_and_monotone(self) -> None:
        import json

        from shared.python.golf_club.impact_coupling import (
            IMPACT_COUPLING_REPORT_FORMAT,
            impact_coupling_report,
        )

        grids = {
            "grip_stiffness_grid_n_m": (0.0, 5.0e4, 1.0e7),
            "grip_mass_grid_kg": (1.0, 3.0, 10.0),
            "shaft_stiffness_grid_n_m": (0.0, 200.0, 1.0e9),
        }
        first = impact_coupling_report(_config(), **grids)
        second = impact_coupling_report(_config(), **grids)
        assert first == second
        payload = json.loads(first)
        assert payload["format"] == IMPACT_COUPLING_REPORT_FORMAT
        assert len(payload["counterfactuals"]) == 9
        # Shaft-stiffness axis: influence grows with k_s (the epic's sweep).
        shaft_rows = [
            row
            for row in payload["counterfactuals"]
            if row["axis"] == "shaft_stiffness_n_m"
        ]
        influences = [
            abs(row["ball_speed_mps"] - row["free_head_ball_speed_mps"])
            for row in shaft_rows
        ]
        assert influences == sorted(influences)
        assert all(
            "literature" in row["grip_provenance"] for row in payload["counterfactuals"]
        )

    def test_empty_grids_are_refused(self) -> None:
        from shared.python.contracts import PreconditionError
        from shared.python.golf_club.impact_coupling import impact_coupling_report

        with pytest.raises(PreconditionError):
            impact_coupling_report(
                _config(),
                grip_stiffness_grid_n_m=(),
                grip_mass_grid_kg=(3.0,),
                shaft_stiffness_grid_n_m=(0.0,),
            )
