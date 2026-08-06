"""Controlled CAD and mesh export tests for wedge solids."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("build123d")

from build123d import import_step  # noqa: E402

from shared.python.golf_club import (  # noqa: E402
    WedgeExportFormat,
    WedgeExportRequest,
    WedgePreset,
    export_wedge_artifacts,
    wedge_preset,
)

pytestmark = [pytest.mark.integration, pytest.mark.contract]


def test_step_stl_brep_and_manifest_are_complete_and_reopenable(
    tmp_path: Path,
) -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    request = WedgeExportRequest(output_directory=tmp_path, stem="mid-bounce")

    result = export_wedge_artifacts(parameters, request)

    assert tuple(item.format for item in result.artifacts) == tuple(WedgeExportFormat)
    assert all(
        item.path.is_file() and item.path.stat().st_size > 100
        for item in result.artifacts
    )
    assert result.manifest_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "golf_club.wedge_export/1"
    assert manifest["units"] == {
        "angle": "degree",
        "length": "metre",
        "mass": "kilogram",
    }
    assert manifest["parameters"]["bounce_deg"] == 10.0
    assert manifest["measured"]["target_mass_residual_kg"] == pytest.approx(
        result.measured.target_mass_residual_kg
    )
    step_path = next(
        item.path for item in result.artifacts if item.format is WedgeExportFormat.STEP
    )
    restored = import_step(step_path)
    assert restored.is_valid
    assert float(restored.volume) == pytest.approx(
        result.measured.volume_m3 * 1.0e9, rel=1e-10
    )


def test_exports_are_byte_deterministic_for_the_same_request(tmp_path: Path) -> None:
    parameters = wedge_preset(WedgePreset.LOW_BOUNCE)
    first = export_wedge_artifacts(
        parameters,
        WedgeExportRequest(output_directory=tmp_path / "first", stem="wedge"),
    )
    second = export_wedge_artifacts(
        parameters,
        WedgeExportRequest(output_directory=tmp_path / "second", stem="wedge"),
    )

    assert [item.path.read_bytes() for item in first.artifacts] == [
        item.path.read_bytes() for item in second.artifacts
    ]
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"stem": "../escape"}, ValueError, "stem"),
        ({"formats": ()}, ValueError, "formats"),
        (
            {"formats": (WedgeExportFormat.STEP, WedgeExportFormat.STEP)},
            ValueError,
            "duplicate",
        ),
        ({"linear_tolerance_m": 0.0}, ValueError, "linear_tolerance_m"),
        ({"angular_tolerance_rad": 2.0}, ValueError, "angular_tolerance_rad"),
    ],
)
def test_export_request_rejects_unsafe_or_invalid_values(
    tmp_path: Path,
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        WedgeExportRequest(output_directory=tmp_path, **kwargs)  # type: ignore[arg-type]
