"""OpenCascade solid tests for the modern-wedge family."""

from __future__ import annotations

import pytest

pytest.importorskip("build123d")

from shared.python.golf_club import (  # noqa: E402
    WedgePreset,
    build_wedge_solid,
    wedge_preset,
)

pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_mid_bounce_wedge_is_one_valid_closed_solid() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)

    result = build_wedge_solid(parameters)

    assert result.solid.is_valid
    assert len(result.solid.solids()) == 1
    assert result.measured.volume_m3 > 0.0
    assert result.measured.mass_kg > 0.0
    assert abs(result.measured.target_mass_residual_kg) < 0.020
    assert result.measured.face_length_m == pytest.approx(
        parameters.face_length_m, rel=1e-6
    )
    assert any("BSPLINE" in str(face.geom_type) for face in result.solid.faces())


@pytest.mark.parametrize("preset", list(WedgePreset))
def test_requested_loft_bounce_and_lie_are_recovered_from_solid_faces(
    preset: WedgePreset,
) -> None:
    parameters = wedge_preset(preset)

    measured = build_wedge_solid(parameters).measured

    assert measured.loft_deg == pytest.approx(parameters.loft_deg, abs=1e-7)
    assert measured.bounce_deg == pytest.approx(parameters.bounce_deg, abs=1e-7)
    assert measured.lie_deg == pytest.approx(parameters.lie_deg, abs=1e-7)


def test_build_is_deterministic_and_reports_mass_residual() -> None:
    parameters = wedge_preset(WedgePreset.HIGH_BOUNCE)

    first = build_wedge_solid(parameters)
    second = build_wedge_solid(parameters)

    assert first.measured == second.measured
    assert first.measured.target_mass_residual_kg == pytest.approx(
        first.measured.mass_kg - parameters.target_mass_kg
    )
    assert first.solid.volume == pytest.approx(second.solid.volume, rel=0.0, abs=0.0)


def test_wrong_parameter_contract_is_rejected() -> None:
    with pytest.raises(TypeError, match="parameters"):
        build_wedge_solid(object())  # type: ignore[arg-type]
