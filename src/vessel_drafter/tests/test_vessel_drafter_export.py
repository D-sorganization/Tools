import json
from pathlib import Path

import pytest

build123d = pytest.importorskip("build123d")

from vessel_drafter.exporters.vessel_export import (  # noqa: E402
    export_vessel,
    export_vessel_brep,
    export_vessel_gltf,
    export_vessel_step,
    export_vessel_stl,
)
from vessel_drafter.models.vessel_drafter import (  # noqa: E402
    VesselDrafterLayout,
    VesselLidPort,
    VesselSidePort,
)


def test_export_vessel_step_creates_file_and_manifest(tmp_path: Path) -> None:
    result = export_vessel(
        output_dir=tmp_path,
        stem="vessel_drafter_default",
        formats=("step",),
    )

    step_path = result["step"]
    assert step_path.exists()
    assert step_path.stat().st_size > 0
    assert "ISO-10303-21" in step_path.read_text(encoding="utf-8", errors="ignore")

    manifest_path = tmp_path / "vessel_drafter_default.json"
    assert manifest_path.exists()


def test_export_vessel_step_convenience(tmp_path: Path) -> None:
    path = export_vessel_step(output_dir=tmp_path, stem="vessel_step")
    assert path.exists()
    assert path.suffix == ".step"


def test_export_vessel_stl_creates_file(tmp_path: Path) -> None:
    path = export_vessel_stl(output_dir=tmp_path, stem="vessel_stl")
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_vessel_brep_creates_file(tmp_path: Path) -> None:
    path = export_vessel_brep(output_dir=tmp_path, stem="vessel_brep")
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_vessel_gltf_creates_file(tmp_path: Path) -> None:
    path = export_vessel_gltf(output_dir=tmp_path, stem="vessel_gltf")
    assert path.exists()
    assert path.stat().st_size > 0


def test_export_multiple_formats_at_once(tmp_path: Path) -> None:
    results = export_vessel(
        output_dir=tmp_path,
        stem="multi",
        formats=("step", "stl"),
    )
    assert "step" in results
    assert "stl" in results
    assert results["step"].exists()
    assert results["stl"].exists()


def test_export_unsupported_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported format"):
        export_vessel(output_dir=tmp_path, formats=("xyz",))


def test_export_manifest_includes_port_configuration(tmp_path: Path) -> None:
    export_vessel(
        layout=VesselDrafterLayout(
            side_ports=(
                VesselSidePort(
                    clock_angle_degrees=45.0,
                    diameter_in=3.0,
                    height_above_glass_surface_in=4.0,
                ),
            ),
            lid_ports=(
                VesselLidPort(
                    clock_angle_degrees=180.0,
                    diameter_in=4.0,
                    radial_distance_from_center_in=6.0,
                ),
            ),
        ),
        output_dir=tmp_path,
        stem="ported_vessel",
        formats=("step",),
    )

    manifest_path = tmp_path / "ported_vessel.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "layout" in manifest
