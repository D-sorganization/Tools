import pytest

build123d = pytest.importorskip("build123d")  # noqa: E402

import json

from vessel_drafter.exporters.step_export import (
    export_cylindrical_bath_layout_step,
    export_default_layout_step,
)


def test_export_creates_step_and_manifest(tmp_path) -> None:
    step_path = tmp_path / "electrode_advisor_default_layout.step"
    manifest_path = tmp_path / "electrode_advisor_default_layout.json"

    export_default_layout_step(step_path, manifest_path)

    assert step_path.exists()
    assert step_path.stat().st_size > 0
    assert "ISO-10303-21" in step_path.read_text(encoding="utf-8", errors="ignore")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["project"] == "electrode_advisor_default_layout"
    assert manifest["bath"]["width_m"] == 3.0
    assert len(manifest["placements"]) == 3


def test_export_creates_cylindrical_bath_step_and_manifest(tmp_path) -> None:
    step_path = tmp_path / "cylindrical_bath_layout.step"
    manifest_path = tmp_path / "cylindrical_bath_layout.json"

    export_cylindrical_bath_layout_step(step_path, manifest_path)

    assert step_path.exists()
    assert step_path.stat().st_size > 0
    assert "ISO-10303-21" in step_path.read_text(encoding="utf-8", errors="ignore")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["project"] == "cylindrical_bath_layout"
    assert manifest["bath"]["inner_diameter_in"] == 50.0
    assert manifest["bath"]["outer_diameter_in"] == 62.0
    assert manifest["electrodes"]["modeled_length_in"] == 50.0
    assert len(manifest["placements"]) == 3
