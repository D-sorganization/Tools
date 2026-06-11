"""Tests for programmatic_pid.cli — orchestration and CLI entry points."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ezdxf")
from programmatic_pid.cli import derive_related_path, generate

_BIOCHAR_SPEC = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "pid"
    / "biochar"
    / "biochar_pid_spec.yml"
)


class TestDeriveRelatedPath:
    def test_adds_suffix(self):
        result = derive_related_path("output.dxf", "controls")
        assert result == Path("output_controls.dxf")

    def test_preserves_directory(self):
        result = derive_related_path("/tmp/drawings/out.dxf", "controls")
        assert result == Path("/tmp/drawings/out_controls.dxf")


class TestGenerate:
    """Integration-level test that generate() produces output files."""

    @pytest.mark.skipif(not _BIOCHAR_SPEC.exists(), reason="biochar spec not found")
    def test_generate_single_sheet(self, tmp_path):
        """Generate a single sheet from the biochar spec."""
        out_dxf = tmp_path / "test_process.dxf"
        generate(str(_BIOCHAR_SPEC), str(out_dxf), sheet_set="single")
        assert out_dxf.exists()
        assert out_dxf.stat().st_size > 0

    @pytest.mark.skipif(not _BIOCHAR_SPEC.exists(), reason="biochar spec not found")
    def test_generate_two_sheets(self, tmp_path):
        """Generate both process and controls sheets."""
        out_dxf = tmp_path / "test_process.dxf"
        controls_dxf = tmp_path / "test_process_controls.dxf"
        generate(str(_BIOCHAR_SPEC), str(out_dxf), sheet_set="two")
        assert out_dxf.exists()
        assert controls_dxf.exists()


class TestBackwardCompatibility:
    """Verify imports from generator.py still work."""

    def test_generator_imports(self):
        from programmatic_pid.generator import (  # noqa: F811
            add_equipment,
            generate,
            validate_spec,
        )

        # All should be callable
        assert callable(generate)
        assert callable(validate_spec)
        assert callable(add_equipment)
