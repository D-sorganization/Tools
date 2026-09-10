"""Independent instantaneous spring/dashpot/cutoff power balances."""

import pytest

from shared.python.swing_sim.impact._normal_contact_work import normal_contact_work
from shared.python.swing_sim.impact.contact import KelvinVoigtContactLaw


@pytest.mark.parametrize(
    "compression,rate,force,energy,storage,viscous,cutoff,power",
    [
        (0.01, 1.0, 12.0, 0.05, 10.0, 2.0, 0.0, 12.0),
        (0.01, -2.0, 6.0, 0.05, -20.0, 8.0, 0.0, -12.0),
        (0.01, -5.0, 0.0, 0.05, -50.0, 0.0, 50.0, 0.0),
        (0.01, -6.0, 0.0, 0.05, -60.0, 0.0, 60.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_disjoint_contact_work_channels(
    compression: float,
    rate: float,
    force: float,
    energy: float,
    storage: float,
    viscous: float,
    cutoff: float,
    power: float,
) -> None:
    result = normal_contact_work(KelvinVoigtContactLaw(1000, 2), compression, rate)
    assert result.force_n == pytest.approx(force)
    assert result.elastic_energy_j == pytest.approx(energy)
    assert result.elastic_power_w == pytest.approx(storage)
    assert result.viscous_power_w == pytest.approx(viscous)
    assert result.cutoff_power_w == pytest.approx(cutoff)
    assert result.input_power_w == pytest.approx(power)
    assert result.power_residual_w == pytest.approx(0, abs=1e-12)


def test_elastic_release_returns_storage_without_invented_loss() -> None:
    result = normal_contact_work(KelvinVoigtContactLaw(1000, 0), 0.01, -2)
    assert result.force_n == 10
    assert result.input_power_w == result.elastic_power_w == -20
    assert result.viscous_power_w == result.cutoff_power_w == 0


def test_force_ceiling_is_refused_before_an_inconsistent_work_ledger() -> None:
    law = KelvinVoigtContactLaw(1000, 2, maximum_force_n=11)
    # Legacy clipping is unchanged; the new resolved work port cannot silently
    # combine that clipped force with unmodified spring/dashpot power.
    assert law.normal_force(0.01, 1) == 11
    with pytest.raises(ValueError, match="force ceiling"):
        normal_contact_work(law, 0.01, 1)


@pytest.mark.parametrize("bad", [True, "0.01", float("nan"), float("inf"), [0.01]])
@pytest.mark.parametrize("coordinate", ["compression", "rate"])
def test_real_finite_state_contracts(bad: object, coordinate: str) -> None:
    compression, rate = (bad, 1.0) if coordinate == "compression" else (0.01, bad)
    with pytest.raises((TypeError, ValueError)):
        normal_contact_work(KelvinVoigtContactLaw(1000, 2), compression, rate)


def test_nonfinite_calculated_work_is_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        normal_contact_work(KelvinVoigtContactLaw(1e200, 0), 1e200, 1)


def test_law_type_is_explicit() -> None:
    with pytest.raises(TypeError, match="KelvinVoigtContactLaw"):
        normal_contact_work(object(), 0.01, 1)
