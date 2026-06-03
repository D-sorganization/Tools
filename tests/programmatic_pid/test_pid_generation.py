"""Functional tests for P&ID generation via programmatic_pid.cli.generate
and programmatic_pid.document.PIDDocument.export_dxf.

Covers the generation pipeline end-to-end: spec → DXF file on disk.
(#3172 epic — programmatic_pid coverage)
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
import yaml

pytest.importorskip("ezdxf", reason="ezdxf not installed")

from programmatic_pid.cli import generate  # noqa: E402
from programmatic_pid.document import PIDDocument  # noqa: E402

_SPEC: dict = {
    "project": {"id": "GEN-TEST-001", "title": "Generation Test P&ID"},
    "equipment": [
        {
            "id": "V-201",
            "type": "vessel",
            "label": "Reactor",
            "x": 30.0,
            "y": 30.0,
            "w": 25.0,
            "h": 20.0,
        },
    ],
}


class TestGenerateFromYAML:
    def test_generate_single_sheet_creates_dxf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = pathlib.Path(tmpdir) / "spec.yml"
            spec_file.write_text(yaml.dump(_SPEC), encoding="utf-8")
            out_dxf = pathlib.Path(tmpdir) / "output.dxf"
            generate(str(spec_file), str(out_dxf), sheet_set="single")
            assert out_dxf.exists(), "generate() should create the DXF file"
            assert out_dxf.stat().st_size > 0

    def test_generate_two_sheets_creates_both_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_file = pathlib.Path(tmpdir) / "spec.yml"
            spec_file.write_text(yaml.dump(_SPEC), encoding="utf-8")
            out_dxf = pathlib.Path(tmpdir) / "output.dxf"
            generate(str(spec_file), str(out_dxf), sheet_set="two")
            assert out_dxf.exists()
            controls_dxf = pathlib.Path(tmpdir) / "output_controls.dxf"
            assert controls_dxf.exists(), "Two-sheet generate should write controls DXF"

    def test_generate_raises_on_none_spec_path(self) -> None:
        with pytest.raises(ValueError):
            generate(None, "/tmp/out.dxf")


class TestPIDDocumentExportDXF:
    def test_export_dxf_single_sheet_creates_file(self) -> None:
        doc = PIDDocument(_SPEC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "test.dxf"
            doc.export_dxf(out, sheet_set="one")
            assert out.exists()
            assert out.stat().st_size > 0

    def test_export_dxf_content_is_valid_dxf(self) -> None:
        import ezdxf

        doc = PIDDocument(_SPEC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "test.dxf"
            doc.export_dxf(out, sheet_set="one")
            loaded = ezdxf.readfile(str(out))
            assert loaded is not None
            msp = loaded.modelspace()
            assert len(list(msp)) > 0, "Generated DXF should contain entities"

    def test_export_round_trips_project_title(self) -> None:
        spec = {
            "project": {"id": "RT-001", "title": "Round Trip Test"},
            "equipment": [
                {
                    "id": "E-001",
                    "type": "vessel",
                    "label": "Tank",
                    "x": 5,
                    "y": 5,
                    "w": 10,
                    "h": 8,
                }
            ],
        }
        doc = PIDDocument(spec)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "rt.dxf"
            doc.export_dxf(out, sheet_set="one")
            assert out.exists()
