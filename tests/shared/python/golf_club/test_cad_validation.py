"""Independent exact-solid and binary-STL validation contracts."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

pytest.importorskip("build123d")

from shared.python.golf_club import (  # noqa: E402
    CadGeometryReference,
    WedgeExportFormat,
    WedgeExportRequest,
    WedgePreset,
    build_wedge_solid,
    export_wedge_artifacts,
    validate_binary_stl,
    validate_exact_cad,
    wedge_preset,
)

pytestmark = [pytest.mark.integration, pytest.mark.contract]


def _reference() -> tuple[CadGeometryReference, object]:
    result = build_wedge_solid(wedge_preset(WedgePreset.MID_BOUNCE))
    bounds = result.solid.bounding_box()
    millimetres_per_metre = 1_000.0
    reference = CadGeometryReference(
        volume_m3=result.measured.volume_m3,
        bounds_min_m=tuple(value / millimetres_per_metre for value in bounds.min),
        bounds_max_m=tuple(value / millimetres_per_metre for value in bounds.max),
    )
    return reference, result.solid


def test_exported_stl_is_independently_watertight_and_volume_bounded(
    tmp_path: Path,
) -> None:
    reference, _ = _reference()
    result = export_wedge_artifacts(
        wedge_preset(WedgePreset.MID_BOUNCE),
        WedgeExportRequest(
            tmp_path,
            formats=(WedgeExportFormat.STL,),
            linear_tolerance_m=5.0e-5,
        ),
    )

    validation = validate_binary_stl(
        result.artifacts[0].path,
        reference,
        linear_tolerance_m=5.0e-5,
    )

    assert validation.passed is True
    assert validation.is_watertight is True
    assert validation.is_winding_consistent is True
    assert validation.has_outward_orientation is True
    assert validation.connected_component_count == 1
    assert validation.triangle_count >= 1_000
    assert validation.volume_relative_error < 0.001
    assert validation.max_bounds_error_m < 5.0e-5


@pytest.mark.parametrize("format_name", ["step", "brep"])
def test_exact_exports_are_reopened_and_compared_to_reference(
    tmp_path: Path,
    format_name: str,
) -> None:
    reference, solid = _reference()
    from build123d import export_brep, export_step

    path = tmp_path / f"wedge.{format_name}"
    exported = (
        export_step(solid, path) if format_name == "step" else export_brep(solid, path)
    )
    assert exported

    validation = validate_exact_cad(path, reference, format_name=format_name)

    assert validation.passed is True
    assert validation.is_valid is True
    assert validation.solid_count == 1
    assert validation.volume_relative_error < 1.0e-9
    assert validation.max_bounds_error_m < 1.0e-9


def test_binary_stl_validation_rejects_a_truncated_artifact(tmp_path: Path) -> None:
    reference, _ = _reference()
    result = export_wedge_artifacts(
        wedge_preset(WedgePreset.MID_BOUNCE),
        WedgeExportRequest(tmp_path, formats=(WedgeExportFormat.STL,)),
    )
    path = result.artifacts[0].path
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="byte length"):
        validate_binary_stl(path, reference, linear_tolerance_m=5.0e-5)


def test_binary_stl_validation_rejects_inconsistent_triangle_winding(
    tmp_path: Path,
) -> None:
    reference, _ = _reference()
    result = export_wedge_artifacts(
        wedge_preset(WedgePreset.MID_BOUNCE),
        WedgeExportRequest(tmp_path, formats=(WedgeExportFormat.STL,)),
    )
    path = result.artifacts[0].path
    payload = bytearray(path.read_bytes())
    first_triangle = list(struct.unpack_from("<12fH", payload, 84))
    first_triangle[6:9], first_triangle[9:12] = (
        first_triangle[9:12],
        first_triangle[6:9],
    )
    struct.pack_into("<12fH", payload, 84, *first_triangle)
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="winding"):
        validate_binary_stl(path, reference, linear_tolerance_m=5.0e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volume_m3": 0.0},
        {"bounds_min_m": (0.0, 0.0, 0.0), "bounds_max_m": (0.0, 1.0, 1.0)},
        {"bounds_min_m": (0.0, 0.0, float("nan"))},
    ],
)
def test_geometry_reference_fails_closed_on_invalid_values(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "volume_m3": 1.0,
        "bounds_min_m": (0.0, 0.0, 0.0),
        "bounds_max_m": (1.0, 1.0, 1.0),
    }
    values.update(kwargs)

    with pytest.raises((TypeError, ValueError)):
        CadGeometryReference(**values)  # type: ignore[arg-type]
