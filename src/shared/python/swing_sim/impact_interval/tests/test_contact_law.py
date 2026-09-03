"""Tests-first contract for the shared Kelvin-Voigt contact law."""

from __future__ import annotations

import math

import pytest

from shared.python.swing_sim.impact_interval import KelvinVoigtContactLaw


@pytest.mark.unit
@pytest.mark.physics
class TestKelvinVoigtContactLaw:
    def test_open_contact_has_zero_force(self) -> None:
        law = KelvinVoigtContactLaw(stiffness_n_per_m=1.0e6, damping_n_s_per_m=900.0)
        assert law.normal_force(compression_m=-1.0e-6, compression_rate_mps=20.0) == 0.0

    def test_force_is_compressive_and_bounded(self) -> None:
        law = KelvinVoigtContactLaw(
            stiffness_n_per_m=1.0e6,
            damping_n_s_per_m=100.0,
            maximum_force_n=5_000.0,
        )
        assert law.normal_force(1.0e-3, 2.0) == pytest.approx(1_200.0)
        assert law.normal_force(1.0e-2, 100.0) == pytest.approx(5_000.0)
        assert law.normal_force(1.0e-3, -20.0) == 0.0

    def test_restitution_constructor_matches_damping_ratio(self) -> None:
        law = KelvinVoigtContactLaw.from_restitution(
            stiffness_n_per_m=5.0e7,
            restitution=0.83,
            effective_mass_kg=0.036,
        )
        log_e = math.log(0.83)
        zeta = -log_e / math.sqrt(math.pi**2 + log_e**2)
        expected = 2.0 * zeta * math.sqrt(5.0e7 * 0.036)
        assert law.damping_n_s_per_m == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"stiffness_n_per_m": 0.0, "damping_n_s_per_m": 1.0}, "stiffness"),
            ({"stiffness_n_per_m": 1.0, "damping_n_s_per_m": -1.0}, "damping"),
        ],
    )
    def test_invalid_parameters_are_rejected(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            KelvinVoigtContactLaw(**kwargs)
