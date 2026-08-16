from __future__ import annotations

import pytest

from shared.python.golf_club import (
    TurfPreset,
    run_turf_convergence_study,
    turf_profile_preset,
)


def test_refinement_study_reports_all_three_pinned_metrics() -> None:
    study = run_turf_convergence_study(
        turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
        initial_contact_velocity_mps=(0.2, -1.0, 0.0),
        surface_normal_unit=(0.0, 1.0, 0.0),
        effective_mass_kg=0.3,
        time_steps_s=(2e-5, 1e-5, 5e-6),
        tolerance=0.02,
    )

    assert study.converged
    assert study.impulse_relative_change <= study.tolerance
    assert study.peak_penetration_relative_change <= study.tolerance
    assert study.dissipated_energy_relative_change <= study.tolerance


@pytest.mark.parametrize(
    "steps",
    [(1e-5,), (1e-5, 2e-5), (1e-5, 1e-5)],
)
def test_refinement_study_rejects_invalid_step_plans(
    steps: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        run_turf_convergence_study(
            turf_profile_preset(TurfPreset.FIRM_FAIRWAY),
            initial_contact_velocity_mps=(0.0, -1.0, 0.0),
            surface_normal_unit=(0.0, 1.0, 0.0),
            effective_mass_kg=0.3,
            time_steps_s=steps,
        )
