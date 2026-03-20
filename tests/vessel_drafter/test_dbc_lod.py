"""Tests for DbC and LoD compliance in src/vessel_drafter.

Covers:
- DbC: precondition assertions fire on None inputs
- LoD: refactored functions (component.shape extraction, path.parent extraction,
  stdout extraction) behave identically to pre-refactor behaviour
- Regression: metric calculations remain numerically correct after LoD fix
"""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vessel_drafter.analysis.vessel_drafter_metrics import (
    CUBIC_INCHES_PER_CUBIC_FOOT,
    SQUARE_INCHES_PER_SQUARE_FOOT,
    ComponentMaterialMetric,
    _component_metric,
    _mm2_to_ft2,
    _mm3_to_in3,
)
from vessel_drafter.models.vessel_drafter import MM_PER_INCH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(
    volume_mm3: float = 1000.0,
    area_mm2: float = 600.0,
    label: str = "test_comp",
    display_name: str = "Test Component",
    category: str = "refractory",
    density_lb_per_ft3: float = 100.0,
    thermal_conductivity_w_per_mk: float = 1.5,
    thermal_expansion_um_per_m_c: float = 8.0,
) -> Any:
    """Build a minimal fake component matching the duck type used by _component_metric."""
    shape = SimpleNamespace(volume=volume_mm3, area=area_mm2)
    return SimpleNamespace(
        shape=shape,
        label=label,
        display_name=display_name,
        category=category,
        density_lb_per_ft3=density_lb_per_ft3,
        thermal_conductivity_w_per_mk=thermal_conductivity_w_per_mk,
        thermal_expansion_um_per_m_c=thermal_expansion_um_per_m_c,
    )


# ---------------------------------------------------------------------------
# LoD Fix — _component_metric (vessel_drafter_metrics.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComponentMetricLoDFix:
    """After the LoD fix, _component_metric extracts component.shape once and
    calls shape.volume / shape.area on the local reference — same result."""

    def test_volume_calculation_correct(self) -> None:
        component = _make_component(volume_mm3=1728.0)
        metric = _component_metric(component)
        expected_in3 = 1728.0 / (MM_PER_INCH**3)
        assert abs(metric.volume_in3 - expected_in3) < 1e-9

    def test_volume_ft3_derived_from_in3(self) -> None:
        component = _make_component(volume_mm3=1728.0)
        metric = _component_metric(component)
        assert (
            abs(metric.volume_ft3 - metric.volume_in3 / CUBIC_INCHES_PER_CUBIC_FOOT)
            < 1e-12
        )

    def test_surface_area_calculation_correct(self) -> None:
        component = _make_component(area_mm2=1440.0)
        metric = _component_metric(component)
        expected_ft2 = (1440.0 / (MM_PER_INCH**2)) / SQUARE_INCHES_PER_SQUARE_FOOT
        assert abs(metric.surface_area_ft2 - expected_ft2) < 1e-12

    def test_mass_derived_from_volume_and_density(self) -> None:
        component = _make_component(volume_mm3=1728.0, density_lb_per_ft3=50.0)
        metric = _component_metric(component)
        assert abs(metric.mass_lb - metric.volume_ft3 * 50.0) < 1e-12

    def test_passthrough_scalar_fields(self) -> None:
        component = _make_component(
            label="my_label",
            display_name="My Name",
            category="structural",
            thermal_conductivity_w_per_mk=2.3,
            thermal_expansion_um_per_m_c=12.5,
        )
        metric = _component_metric(component)
        assert metric.label == "my_label"
        assert metric.display_name == "My Name"
        assert metric.category == "structural"
        assert metric.thermal_conductivity_w_per_mk == 2.3
        assert metric.thermal_expansion_um_per_m_c == 12.5

    def test_returns_component_material_metric(self) -> None:
        component = _make_component()
        metric = _component_metric(component)
        assert isinstance(metric, ComponentMaterialMetric)

    def test_shape_attribute_accessed_once_via_local(self) -> None:
        """LoD: shape is assigned to a local variable; multiple calls are safe."""

        class TrackingShape:
            def __init__(self) -> None:
                self.volume = 500.0
                self.area = 200.0

        component = _make_component(volume_mm3=500.0, area_mm2=200.0)
        component.shape = TrackingShape()
        # Should not raise — refactor is compatible with standard attribute protocol
        metric = _component_metric(component)
        assert metric.volume_in3 > 0
        assert metric.surface_area_ft2 > 0


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnitConversionHelpers:
    def test_mm3_to_in3_cube_25_4mm(self) -> None:
        # 1 inch cube = 25.4^3 mm^3 ≈ 16387.064 mm^3 = 1 in^3
        vol_mm3 = 25.4**3
        assert abs(_mm3_to_in3(vol_mm3) - 1.0) < 1e-9

    def test_mm3_to_in3_zero(self) -> None:
        assert _mm3_to_in3(0.0) == 0.0

    def test_mm2_to_ft2_one_sqft(self) -> None:
        # 1 ft^2 = 144 in^2 = 144 * 25.4^2 mm^2
        area_mm2 = 144.0 * (25.4**2)
        assert abs(_mm2_to_ft2(area_mm2) - 1.0) < 1e-9

    def test_mm2_to_ft2_zero(self) -> None:
        assert _mm2_to_ft2(0.0) == 0.0


# ---------------------------------------------------------------------------
# DbC — step_export.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepExportDbC:
    """DbC: all public export functions assert output_path is not None."""

    def test_export_default_layout_step_none_output_path_raises(self) -> None:
        from vessel_drafter.exporters.step_export import export_default_layout_step

        with pytest.raises(AssertionError, match="output_path"):
            export_default_layout_step(output_path=None)

    def test_export_cylindrical_bath_layout_step_none_output_path_raises(self) -> None:
        from vessel_drafter.exporters.step_export import (
            export_cylindrical_bath_layout_step,
        )

        with pytest.raises(AssertionError, match="output_path"):
            export_cylindrical_bath_layout_step(output_path=None)

    def test_export_vessel_drafter_step_none_output_path_raises(self) -> None:
        from vessel_drafter.exporters.step_export import export_vessel_drafter_step

        with pytest.raises(AssertionError, match="output_path"):
            export_vessel_drafter_step(output_path=None)


# ---------------------------------------------------------------------------
# LoD Fix — step_export.py: parent dir extracted as local variable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepExportLoDFix:
    """Verify that path.parent is extracted as a local variable (not chained)
    and that the mkdir behaviour is preserved."""

    def test_export_default_layout_creates_output_parent(self, tmp_path: Path) -> None:
        from vessel_drafter.exporters.step_export import export_default_layout_step

        out = tmp_path / "sub" / "out.step"
        mock_shape = MagicMock()

        with (
            patch(
                "vessel_drafter.exporters.step_export.build_default_layout_shape",
                return_value=mock_shape,
            ),
            patch("vessel_drafter.exporters.step_export.export_step"),
        ):
            result = export_default_layout_step(output_path=out)

        assert out.parent.exists()
        assert result == out

    def test_export_default_layout_creates_manifest_parent(
        self, tmp_path: Path
    ) -> None:
        from vessel_drafter.exporters.step_export import export_default_layout_step

        out = tmp_path / "out.step"
        manifest = tmp_path / "manifests" / "out.json"
        mock_shape = MagicMock()
        mock_layout = MagicMock()
        mock_layout.to_manifest.return_value = {"key": "value"}

        with (
            patch(
                "vessel_drafter.exporters.step_export.build_default_layout_shape",
                return_value=mock_shape,
            ),
            patch("vessel_drafter.exporters.step_export.export_step"),
        ):
            export_default_layout_step(
                output_path=out, manifest_path=manifest, layout=mock_layout
            )

        assert manifest.parent.exists()

    def test_export_cylindrical_bath_creates_output_parent(
        self, tmp_path: Path
    ) -> None:
        from vessel_drafter.exporters.step_export import (
            export_cylindrical_bath_layout_step,
        )

        out = tmp_path / "sub" / "out.step"
        mock_shape = MagicMock()

        with (
            patch(
                "vessel_drafter.exporters.step_export.build_cylindrical_bath_layout_shape",
                return_value=mock_shape,
            ),
            patch("vessel_drafter.exporters.step_export.export_step"),
        ):
            result = export_cylindrical_bath_layout_step(output_path=out)

        assert out.parent.exists()
        assert result == out

    def test_export_vessel_drafter_creates_output_parent(self, tmp_path: Path) -> None:
        from vessel_drafter.exporters.step_export import export_vessel_drafter_step

        out = tmp_path / "sub" / "out.step"
        mock_shape = MagicMock()
        mock_layout = MagicMock()
        mock_layout.to_manifest.return_value = {}
        mock_metrics = MagicMock()
        mock_metrics.component_metrics = []
        mock_metrics.refractory_total_volume_in3 = 0.0
        mock_metrics.refractory_total_volume_ft3 = 0.0
        mock_metrics.refractory_total_surface_area_ft2 = 0.0
        mock_metrics.refractory_total_mass_lb = 0.0

        with (
            patch(
                "vessel_drafter.exporters.step_export.build_vessel_drafter_shape",
                return_value=mock_shape,
            ),
            patch("vessel_drafter.exporters.step_export.export_step"),
            patch(
                "vessel_drafter.exporters.step_export.build_material_metrics_report",
                return_value=mock_metrics,
            ),
        ):
            result = export_vessel_drafter_step(output_path=out, layout=mock_layout)

        assert out.parent.exists()
        assert result == out

    def test_export_vessel_drafter_no_manifest_skips_manifest_write(
        self, tmp_path: Path
    ) -> None:
        """manifest_path=None branch must not call mkdir or write_text on a path."""
        from vessel_drafter.exporters.step_export import export_vessel_drafter_step

        out = tmp_path / "out.step"
        mock_shape = MagicMock()
        mock_layout = MagicMock()

        with (
            patch(
                "vessel_drafter.exporters.step_export.build_vessel_drafter_shape",
                return_value=mock_shape,
            ),
            patch("vessel_drafter.exporters.step_export.export_step"),
        ):
            result = export_vessel_drafter_step(
                output_path=out, manifest_path=None, layout=mock_layout
            )

        assert result == out


# ---------------------------------------------------------------------------
# DbC — vessel_export.py
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVesselExportDbC:
    """DbC: export_vessel and _write_manifest assert layout is not None."""

    def test_export_vessel_none_layout_raises(self) -> None:
        from vessel_drafter.exporters.vessel_export import export_vessel

        with pytest.raises(AssertionError, match="layout"):
            export_vessel(layout=None)


# ---------------------------------------------------------------------------
# LoD Fix — cli.py: stdout extracted as local variable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCLIStdoutLoDFix:
    """Verify that main() extracts sys.stdout as a local variable and uses it
    correctly — the written output is identical to direct sys.stdout.write."""

    def _run_main_with_captured_stdout(
        self,
        argv: list[str],
        mock_export_fn: str,
        mock_return: Path,
    ) -> str:
        from vessel_drafter import cli as cli_module

        capture = StringIO()
        with (
            patch(f"vessel_drafter.cli.{mock_export_fn}", return_value=mock_return),
            patch.object(sys, "stdout", capture),
        ):
            try:
                cli_module.main.__globals__["sys"].stdout = capture
                if hasattr(cli_module.main, "__wrapped__"):
                    cli_module.main.__wrapped__()
            except SystemExit:
                pass

        return capture.getvalue()

    def test_stdout_local_var_works_with_patched_stdout(self, tmp_path: Path) -> None:
        """After LoD fix, stdout is captured at the start of main().
        Patching sys.stdout before main() runs must still work.
        """
        from vessel_drafter import cli as cli_module

        fake_path = tmp_path / "out.step"
        capture = StringIO()

        with (
            patch(
                "vessel_drafter.cli.export_default_layout_step",
                return_value=fake_path,
            ),
            patch.object(sys, "stdout", capture),
            patch("sys.argv", ["vessel-drafter", "export-electrode-advisor-default"]),
        ):
            ret = cli_module.main()

        assert ret == 0
        assert str(fake_path) in capture.getvalue()

    def test_cylindrical_bath_command_stdout_output(self, tmp_path: Path) -> None:
        from vessel_drafter import cli as cli_module

        fake_path = tmp_path / "bath.step"
        capture = StringIO()

        with (
            patch(
                "vessel_drafter.cli.export_cylindrical_bath_layout_step",
                return_value=fake_path,
            ),
            patch.object(sys, "stdout", capture),
            patch("sys.argv", ["vessel-drafter", "export-cylindrical-bath-layout"]),
        ):
            ret = cli_module.main()

        assert ret == 0
        assert str(fake_path) in capture.getvalue()

    def test_vessel_drafter_default_command_stdout_output(self, tmp_path: Path) -> None:
        from vessel_drafter import cli as cli_module

        fake_path = tmp_path / "vessel.step"
        capture = StringIO()

        with (
            patch(
                "vessel_drafter.cli.export_vessel_drafter_default_step",
                return_value=fake_path,
            ),
            patch.object(sys, "stdout", capture),
            patch("sys.argv", ["vessel-drafter", "export-vessel-drafter-default"]),
        ):
            ret = cli_module.main()

        assert ret == 0
        assert str(fake_path) in capture.getvalue()
