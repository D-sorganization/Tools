"""Exact-solid and golden-geometry tests for the iron-head family (Tools #4149).

Volume, mass, and centre of gravity are recovered from the generated B-Rep
and compared with values computed by *independent* methods, never with the
builder's own inputs:

* body only: a closed-form prismatoid (Simpson) integral of the canonical
  heel/toe sections, implemented in this file from the section vertices;
* full head (body + hollow hosel): the divergence theorem applied to the
  exported STL mesh, read back through the package's binary-STL reader.

Stated tolerances
-----------------
* Closed form vs OCC exact solid: volume ``rel 1e-9``, CG ``abs 1e-9 m``.
  Both are exact for a ruled loft of polygons, so the bound only absorbs
  floating-point and OCC Gauss-integration round-off.
* Exported mesh vs OCC exact solid: bounds derived from the chordal
  tessellation tolerance (see ``_tessellation_bounds``), not tuned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

build123d = pytest.importorskip("build123d")

from shared.python.golf_club import (  # noqa: E402
    IRON_EXPORT_FORMAT,
    IRON_PARAMETERS_FORMAT,
    IronExportRequest,
    IronHeadParameters,
    IronPreset,
    WedgeExportFormat,
    build_iron_solid,
    export_iron_artifacts,
    iron_body_sections_m,
    iron_parameters_from_json,
    iron_preset,
    is_watertight,
    mesh_volume_centroid,
    recover_solid_mass_properties,
)
from shared.python.golf_club.stl_validation import read_binary_stl  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.contract]

_ANALYTIC_VOLUME_REL_TOL = 1.0e-9
_ANALYTIC_CG_ABS_TOL_M = 1.0e-9
_M3_PER_MM3 = 1.0e-9
_M_PER_MM = 1.0e-3

# Golden geometry for the generic mid-iron head: volume to six significant
# figures (checked at rel 1e-5), CG to 1 micrometre. Far outside OCC round-off,
# far inside any intentional geometry change.
# Update only with a recorded reason in the PR that changes the geometry.
GOLDEN_MID_IRON_VOLUME_M3 = 3.40884e-05  # 265.9 g at 7800 kg/m^3
GOLDEN_MID_IRON_CG_M = (-0.0140721, 0.0177840, -0.0040024)


def _section_moments(polygon: np.ndarray) -> tuple[float, float, float]:
    """Signed area and first moments ``(Qx, Qy)`` of a simple polygon."""
    x_values, y_values = polygon[:, 0], polygon[:, 1]
    x_next, y_next = np.roll(x_values, -1), np.roll(y_values, -1)
    cross = x_values * y_next - x_next * y_values
    area = 0.5 * float(cross.sum())
    first_x = float(((x_values + x_next) * cross).sum()) / 6.0
    first_y = float(((y_values + y_next) * cross).sum()) / 6.0
    return area, first_x, first_y


def _analytic_body(parameters: IronHeadParameters) -> tuple[float, np.ndarray]:
    """Exact volume and centroid of the ruled heel-to-toe loft.

    Sections interpolate vertex-wise between heel and toe, so section area
    and first moments are at most quadratic in z and ``z * A(z)`` is cubic:
    Simpson's rule integrates all of them exactly.
    """
    sections = iron_body_sections_m(parameters)
    heel = np.asarray(sections.heel_profile_m)
    toe = np.asarray(sections.toe_profile_m)
    totals = np.zeros(4)  # volume, Mx, My, Mz (Simpson-weighted sums)
    for weight, fraction in ((1.0, 0.0), (4.0, 0.5), (1.0, 1.0)):
        area, first_x, first_y = _section_moments(
            (1.0 - fraction) * heel + fraction * toe
        )
        z_value = (1.0 - fraction) * sections.heel_z_m + fraction * sections.toe_z_m
        totals += weight * np.array([area, first_x, first_y, area * z_value])
    totals *= abs(sections.toe_z_m - sections.heel_z_m) / 6.0
    return abs(float(totals[0])), totals[1:] / totals[0]


@pytest.mark.parametrize("preset", list(IronPreset))
def test_each_preset_is_one_valid_solid_with_recovered_datums(
    preset: IronPreset,
) -> None:
    parameters = iron_preset(preset)

    result = build_iron_solid(parameters)
    measured = result.measured

    assert result.solid.is_valid
    assert len(result.solid.solids()) == 1
    assert measured.loft_deg == pytest.approx(parameters.loft_deg, abs=1e-7)
    assert measured.lie_deg == pytest.approx(parameters.lie_deg, abs=1e-7)
    assert measured.bounce_deg == pytest.approx(parameters.bounce_deg, abs=1e-7)
    assert measured.blade_length_m == pytest.approx(parameters.blade_length_m, rel=1e-9)
    assert measured.mass_kg == pytest.approx(
        measured.volume_m3 * parameters.material_density_kg_m3, rel=1e-12
    )
    assert measured.target_mass_residual_kg == pytest.approx(
        measured.mass_kg - parameters.target_mass_kg
    )
    # Generic presets are sized near a typical head-weight progression, not
    # solved to it; 15 g is the stated illustration band.
    assert abs(measured.target_mass_residual_kg) < 0.015


@pytest.mark.parametrize("preset", list(IronPreset))
def test_body_mass_properties_match_closed_form_prismatoid(
    preset: IronPreset,
) -> None:
    parameters = iron_preset(preset)
    expected_volume, expected_cg = _analytic_body(parameters)

    body = build_iron_solid(parameters).body
    recovered = recover_solid_mass_properties(body, parameters.material_density_kg_m3)

    assert recovered.volume_m3 == pytest.approx(
        expected_volume, rel=_ANALYTIC_VOLUME_REL_TOL
    )
    assert recovered.mass_kg == pytest.approx(
        expected_volume * parameters.material_density_kg_m3,
        rel=_ANALYTIC_VOLUME_REL_TOL,
    )
    np.testing.assert_allclose(
        recovered.cg_m, expected_cg, rtol=0.0, atol=_ANALYTIC_CG_ABS_TOL_M
    )


def test_hosel_adds_volume_and_moves_cg_toward_the_heel_and_up() -> None:
    parameters = iron_preset(IronPreset.MID_IRON)
    result = build_iron_solid(parameters)
    body = recover_solid_mass_properties(result.body, parameters.material_density_kg_m3)
    head = result.measured

    assert head.volume_m3 > body.volume_m3
    assert head.cg_m[2] < body.cg_m[2]  # right-handed: heel at negative z
    assert head.cg_m[1] > body.cg_m[1]


def test_golden_mid_iron_mass_properties() -> None:
    measured = build_iron_solid(iron_preset(IronPreset.MID_IRON)).measured

    assert measured.volume_m3 == pytest.approx(GOLDEN_MID_IRON_VOLUME_M3, rel=1e-5)
    np.testing.assert_allclose(
        measured.cg_m, GOLDEN_MID_IRON_CG_M, rtol=0.0, atol=1.0e-6
    )


def test_build_is_deterministic() -> None:
    parameters = iron_preset(IronPreset.SHORT_IRON)

    first = build_iron_solid(parameters)
    second = build_iron_solid(parameters)

    assert first.measured == second.measured


def test_builder_and_recovery_reject_wrong_contracts() -> None:
    with pytest.raises(TypeError, match="parameters must be IronHeadParameters"):
        build_iron_solid(object())  # type: ignore[arg-type]
    body = build_iron_solid(iron_preset(IronPreset.MID_IRON)).body
    with pytest.raises(ValueError, match="density_kg_m3 must be > 0"):
        recover_solid_mass_properties(body, 0.0)
    with pytest.raises(TypeError, match="solid"):
        recover_solid_mass_properties(object(), 7_800.0)


def _tessellation_bounds(
    solid: object, linear_tolerance_m: float
) -> tuple[float, float]:
    """Volume and CG error bounds implied by the chordal tessellation tolerance.

    Every tessellated surface point lies within ``delta`` of the exact
    surface, so the enclosed volume changes by at most ``A * delta`` and the
    first moment by at most ``A * delta * R`` (``R`` the bounding-box
    diagonal, which exceeds any distance from the centroid). Hence the
    relative volume error is at most ``A * delta / V`` and the centroid moves
    at most ``2 * A * delta * R / V``.
    """
    area_m2 = float(solid.area) * _M_PER_MM**2  # type: ignore[attr-defined]
    volume_m3 = float(solid.volume) * _M3_PER_MM3  # type: ignore[attr-defined]
    bounds = solid.bounding_box()  # type: ignore[attr-defined]
    radius_m = float(bounds.diagonal) * _M_PER_MM
    volume_bound = area_m2 * linear_tolerance_m / volume_m3
    cg_bound = 2.0 * area_m2 * linear_tolerance_m * radius_m / volume_m3
    return volume_bound, cg_bound


def test_export_round_trips_exact_and_mesh_artifacts(tmp_path: Path) -> None:
    parameters = iron_preset(IronPreset.MID_IRON)
    request = IronExportRequest(output_directory=tmp_path)

    result = export_iron_artifacts(parameters, request)

    paths = {item.format: item.path for item in result.artifacts}
    assert set(paths) == set(WedgeExportFormat)
    assert all(path.name.startswith("iron-head.") for path in paths.values())
    measured_mm3 = result.measured.volume_m3 / _M3_PER_MM3
    for reader, export_format in (
        (build123d.import_step, WedgeExportFormat.STEP),
        (build123d.import_brep, WedgeExportFormat.BREP),
    ):
        restored = reader(paths[export_format])
        assert restored.is_valid
        assert float(restored.volume) == pytest.approx(measured_mm3, rel=1e-9)

    triangles = np.asarray(read_binary_stl(paths[WedgeExportFormat.STL])) * _M_PER_MM
    assert is_watertight(triangles)
    mesh_volume, mesh_cg = mesh_volume_centroid(triangles)
    volume_bound, cg_bound = _tessellation_bounds(
        build_iron_solid(parameters).solid, request.linear_tolerance_m
    )
    assert abs(mesh_volume / result.measured.volume_m3 - 1.0) <= volume_bound
    cg_error = float(np.max(np.abs(mesh_cg - np.asarray(result.measured.cg_m))))
    assert cg_error <= cg_bound

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == IRON_EXPORT_FORMAT == "golf_club.iron_export/1"
    assert manifest["parameters_format"] == IRON_PARAMETERS_FORMAT
    document = {"format": IRON_PARAMETERS_FORMAT, "parameters": manifest["parameters"]}
    assert iron_parameters_from_json(json.dumps(document)) == parameters
    assert manifest["measured"]["cg_m"] == list(result.measured.cg_m)


def test_exports_are_byte_deterministic(tmp_path: Path) -> None:
    parameters = iron_preset(IronPreset.LONG_IRON)
    first = export_iron_artifacts(
        parameters, IronExportRequest(output_directory=tmp_path / "a")
    )
    second = export_iron_artifacts(
        parameters, IronExportRequest(output_directory=tmp_path / "b")
    )

    assert [item.path.read_bytes() for item in first.artifacts] == [
        item.path.read_bytes() for item in second.artifacts
    ]
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


def test_export_rejects_wrong_contracts(tmp_path: Path) -> None:
    request = IronExportRequest(output_directory=tmp_path)
    with pytest.raises(TypeError, match="parameters must be IronHeadParameters"):
        export_iron_artifacts(object(), request)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="request must be IronExportRequest"):
        export_iron_artifacts(iron_preset(IronPreset.MID_IRON), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stem"):
        IronExportRequest(output_directory=tmp_path, stem="../escape")
