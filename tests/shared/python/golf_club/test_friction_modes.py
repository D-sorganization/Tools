"""Integrated elastic reversal and monotone sliding under a resolved normal load."""

from dataclasses import replace

import numpy as np

from shared.python.swing_sim.impact._friction_contact_trajectory import (
    integrate_friction_contact,
)

from .test_friction_contact_trajectory import _controls, _friction_case


def test_elastic_reversal_unloads_then_reloads_without_plastic_dissipation() -> None:
    problem, initial = _friction_case(0.3)
    mechanical = initial.mechanical
    ball = replace(mechanical.ball, twist=(-0.2, 0, -0.4, 0, 0, 0))
    initial = replace(
        initial,
        mechanical=replace(mechanical, ball=ball),
        tangential=replace(initial.tangential, elastic_deflection_m=(2e-6, 0, 0)),
    )
    samples = integrate_friction_contact(problem, initial, _controls(16)).samples
    energies = np.array([s.state.tangential.elastic_energy_j for s in samples])
    forces = np.array([s.response.tangential_force_n[0] for s in samples])
    assert forces[0] < 0 and forces[-1] > 0
    minimum = int(np.argmin(energies))
    assert 0 < minimum < len(samples) - 1
    assert energies[minimum] < energies[0] / 100
    assert energies[-1] > energies[minimum]
    assert all(s.plastic_dissipation_j == 0 for s in samples)
    assert all(s.response.normal.force_n > 0 for s in samples)


def test_monotone_sliding_obeys_coulomb_bound_and_positive_plastic_work() -> None:
    problem, initial = _friction_case(1e-4)
    mechanical = initial.mechanical
    ball = replace(mechanical.ball, twist=(5, 0, -0.4, 0, 0, 0))
    initial = replace(initial, mechanical=replace(mechanical, ball=ball))
    samples = integrate_friction_contact(problem, initial, _controls(4)).samples
    for previous, sample in zip(samples[:-1], samples[1:], strict=True):
        force = np.asarray(sample.response.tangential_force_n)
        slip = np.asarray(sample.response.bodies.contact.relative_velocity_mps)
        assert slip[0] > 0 and force @ slip < 0
        np.testing.assert_allclose(
            np.linalg.norm(force),
            1e-4 * sample.response.normal.force_n,
            rtol=1e-10,
            atol=1e-12,
        )
        assert sample.plastic_dissipation_j > previous.plastic_dissipation_j
