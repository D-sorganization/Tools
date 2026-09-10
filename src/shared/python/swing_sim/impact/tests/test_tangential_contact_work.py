"""Independent work, objectivity and refinement controls for private friction."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.swing_sim.impact._tangential_contact_work import (
    TangentialContactLaw,
    TangentialContactState,
    TangentialContactUpdate,
    advance_tangential_contact,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _state(deflection: object = (0, 0, 0)) -> TangentialContactState:
    return TangentialContactState(
        TangentialContactLaw(1000, 0.5), deflection, (0, 0, 1), "observer"
    )


def _advance(
    state: TangentialContactState, slip: object, force: object = 2
) -> TangentialContactUpdate:
    return advance_tangential_contact(state, slip, np.eye(3), (0, 0, 1), force)


@pytest.mark.parametrize(
    "slip,force,stored,plastic,algorithmic,work",
    [
        (0.0002, 0.2, 0.00002, 0, 0.00002, 0.00004),
        (0.003, 1, 0.0005, 0.002, 0.0005, 0.003),
    ],
)
def test_sticking_and_sliding_have_disjoint_work_channels(
    slip: float,
    force: float,
    stored: float,
    plastic: float,
    algorithmic: float,
    work: float,
) -> None:
    result = _advance(_state(), (slip, 0, 0))
    np.testing.assert_allclose(result.resisting_force_n, (force, 0, 0), atol=1e-14)
    assert result.state.elastic_energy_j == pytest.approx(stored)
    assert result.plastic_dissipation_j == pytest.approx(plastic)
    assert result.algorithmic_loss_j == pytest.approx(algorithmic)
    assert result.input_work_j == pytest.approx(work)
    assert abs(result.work_residual_j) < 1e-15


def test_elastic_unloading_returns_work_without_negative_dissipation() -> None:
    result = _advance(_state((0.001, 0, 0)), (-0.0005, 0, 0))
    assert result.input_work_j == pytest.approx(-0.00025)
    assert result.delta_stored_energy_j == pytest.approx(-0.000375)
    assert result.algorithmic_loss_j == pytest.approx(0.000125)
    assert result.plastic_dissipation_j == 0
    assert abs(result.work_residual_j) < 1e-15


def test_zero_normal_force_accounts_for_removed_elastic_storage() -> None:
    result = _advance(_state((0.001, 0, 0)), (0.001, 0, 0), force=0)
    np.testing.assert_array_equal(result.resisting_force_n, (0, 0, 0))
    assert result.state.elastic_energy_j == 0
    assert result.input_work_j == result.plastic_dissipation_j == 0
    assert result.algorithmic_loss_j == pytest.approx(0.0005)
    assert abs(result.work_residual_j) < 1e-15


def test_normal_rotation_and_twirl_transport_storage_without_spurious_work() -> None:
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    original = _state((0.0005, 0, 0))
    result = advance_tangential_contact(original, (0, 0, 0), rotation, (1, 0, 0), 2)
    np.testing.assert_allclose(result.resisting_force_n, (0, 0, -0.5), atol=1e-14)
    assert result.state.elastic_energy_j == pytest.approx(original.elastic_energy_j)
    assert result.algorithmic_loss_j == result.plastic_dissipation_j == 0
    twirl = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = advance_tangential_contact(original, (0, 0, 0), twirl, (0, 0, 1), 2)
    np.testing.assert_allclose(result.resisting_force_n, (0, 0.5, 0), atol=1e-14)
    assert result.input_work_j == result.algorithmic_loss_j == 0


def test_observer_rotation_preserves_constitutive_work_and_rotates_force() -> None:
    observer = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    state = _state((0.0004, -0.0002, 0))
    slip = np.array([0.001, 0.0006, 0])
    original = _advance(state, slip)
    changed = replace(
        state,
        elastic_deflection_m=observer @ state.elastic_deflection_m,
        normal=observer @ state.normal,
    )
    rotated = advance_tangential_contact(
        changed, observer @ slip, np.eye(3), observer @ state.normal, 2
    )
    np.testing.assert_allclose(
        rotated.resisting_force_n, observer @ original.resisting_force_n, atol=1e-14
    )
    assert rotated.input_work_j == pytest.approx(original.input_work_j)
    assert rotated.plastic_dissipation_j == pytest.approx(
        original.plastic_dissipation_j
    )
    assert rotated.algorithmic_loss_j == pytest.approx(original.algorithmic_loss_j)


def test_monotone_slip_refinement_removes_algorithmic_loss() -> None:
    losses = []
    for count in (100, 200, 400):
        state = _state()
        work = plastic = algorithmic = 0.0
        for _ in range(count):
            result = _advance(state, (0.01 / count, 0, 0))
            state = result.state
            work += result.input_work_j
            plastic += result.plastic_dissipation_j
            algorithmic += result.algorithmic_loss_j
        assert state.elastic_energy_j == pytest.approx(0.0005)
        assert plastic == pytest.approx(0.009)
        assert work - algorithmic == pytest.approx(0.0095)
        losses.append(algorithmic)
    np.testing.assert_allclose(np.array(losses[:-1]) / losses[1:], 2, rtol=1e-10)


@pytest.mark.parametrize("force", [True, "2", -1, np.nan, np.inf])
def test_normal_force_contract_refuses_invalid_values(force: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _advance(_state(), (0, 0, 0), force)


@pytest.mark.parametrize(
    "slip,rotation,normal",
    [
        ((0, 0, 0.001), np.eye(3), (0, 0, 1)),
        ((True, 0, 0), np.eye(3), (0, 0, 1)),
        ((0, 0, 0), np.diag([1, 1, -1]), (0, 0, 1)),
        ((0, 0, 0), np.eye(3), (1, 0, 0)),
        ((0, 0, 0), np.eye(3), (0, 0, 2)),
    ],
)
def test_incompatible_transport_and_nontangent_slip_are_refused(
    slip: object, rotation: object, normal: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        advance_tangential_contact(_state(), slip, rotation, normal, 2)


def test_state_owns_history_and_refuses_normal_elastic_deflection() -> None:
    source = np.array([0.0004, 0, 0])
    state = _state(source)
    source[:] = 0
    assert state.elastic_energy_j == pytest.approx(0.00008)
    with pytest.raises(ValueError, match="tangent"):
        _state((0, 0, 0.001))


@pytest.mark.parametrize(
    "stiffness,friction",
    [
        (0, 0.5),
        (-1, 0.5),
        (True, 0.5),
        (np.inf, 0.5),
        (1000, -1),
        (1000, True),
        (1000, "0.5"),
        (1000, np.nan),
    ],
)
def test_constitutive_coefficients_are_strict(
    stiffness: object, friction: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TangentialContactLaw(stiffness, friction)


def test_shrinking_normal_force_cap_releases_storage_with_closed_work() -> None:
    result = _advance(_state((0.001, 0, 0)), (0, 0, 0), force=1)
    np.testing.assert_allclose(result.resisting_force_n, (0.5, 0, 0), atol=1e-14)
    assert result.delta_stored_energy_j == pytest.approx(-0.000375)
    assert result.plastic_dissipation_j == pytest.approx(0.00025)
    assert result.algorithmic_loss_j == pytest.approx(0.000125)
    assert abs(result.work_residual_j) < 1e-15


def test_slip_reversal_unloads_before_opposite_sliding() -> None:
    forward = _advance(_state(), (0.003, 0, 0))
    reverse = _advance(forward.state, (-0.004, 0, 0))
    np.testing.assert_allclose(reverse.resisting_force_n, (-1, 0, 0), atol=1e-14)
    assert reverse.input_work_j == pytest.approx(0.004)
    assert reverse.plastic_dissipation_j == pytest.approx(0.002)
    assert reverse.algorithmic_loss_j == pytest.approx(0.002)
    assert abs(reverse.work_residual_j) < 1e-15
