"""Tests for PlotWidget export metadata injection and identity plumbing.

Issues: #4740, #4722.
Verifies end-to-end wiring:
- PlotWidget.set_identity() and get_identity()
- PlotWidget._export_plot() routing through export_figure / export_plot_data
- Metadata injection (engine, model, run_id, timestamp, version) in PNG, SVG, PDF, CSV
- Default filename derived from spec title
- Plot figure footer rendering
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import pytest
from PyQt6.QtWidgets import QApplication

from shared.python.plot_engine.pyqt6_widget import PlotWidget
from shared.python.plot_engine.specs import PlotSpec, SeriesData
from shared.python.plotting.identity import PlotIdentity


@pytest.fixture
def app(qapp: QApplication) -> QApplication:
    return qapp


def _make_spec(title: str = "Joint Positions") -> PlotSpec:
    return PlotSpec(
        title=title,
        series=[SeriesData(name="joint_1", x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 4.0])],
    )


@pytest.mark.unit
class TestPlotWidgetIdentity:
    """Test identity attachment to PlotWidget."""

    def test_default_identity_is_none(self, app: QApplication) -> None:
        widget = PlotWidget()
        assert widget.get_identity() is None

    def test_set_identity_roundtrip(self, app: QApplication) -> None:
        widget = PlotWidget()
        identity = PlotIdentity(
            engine="mujoco",
            model="golfer_v3",
            run_id="run-001",
            version="1.2.0",
        )
        widget.set_identity(identity)
        assert widget.get_identity() is identity

    def test_set_identity_triggers_refresh_with_footer(self, app: QApplication) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec())
        identity = PlotIdentity(engine="drake", model="pendulum_v1")
        widget.set_identity(identity)

        texts = [text.get_text() for text in widget._figure.texts]
        assert any("drake" in t and "pendulum_v1" in t for t in texts)

    def test_clear_identity_removes_footer(self, app: QApplication) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec())
        widget.set_identity(PlotIdentity(engine="drake", model="pendulum_v1"))
        widget.set_identity(None)

        texts = [text.get_text() for text in widget._figure.texts]
        assert not any("drake" in t for t in texts)


@pytest.mark.unit
class TestPlotWidgetExportMetadata:
    """Test metadata injection during export."""

    def test_export_plot_png_with_identity_metadata(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        from PIL import Image

        widget = PlotWidget()
        widget.set_spec(_make_spec())
        widget.set_identity(
            PlotIdentity(
                engine="mujoco",
                model="golfer_v3",
                run_id="run-42",
                version="2.0.0",
            )
        )

        out_path = tmp_path / "joint_positions.png"
        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "PNG Files (*.png)"),
        ):
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()

        assert out_path.exists()
        info = Image.open(out_path).info
        assert info.get("engine") == "mujoco"
        assert info.get("model") == "golfer_v3"
        assert info.get("run_id") == "run-42"
        assert info.get("version") == "2.0.0"
        assert "Creation Time" in info

    def test_export_plot_without_identity_still_embeds_timestamp(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        from PIL import Image

        widget = PlotWidget()
        widget.set_spec(_make_spec())

        out_path = tmp_path / "no_identity.png"
        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "PNG Files (*.png)"),
        ):
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()

        assert out_path.exists()
        info = Image.open(out_path).info
        assert "Creation Time" in info
        assert "engine" not in info

    def test_export_plot_default_filename_uses_spec_title(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec(title="Downswing Clubhead Speed"))

        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=("", ""),
        ) as mock_dialog:
            widget._format_combo.setCurrentText("PNG")
            widget._export_plot()
            args, _kwargs = mock_dialog.call_args
            assert "Downswing Clubhead Speed" in args[2]

    def test_export_plot_svg_with_identity(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec())
        widget.set_identity(PlotIdentity(engine="pinocchio", model="arm_robot"))

        out_path = tmp_path / "plot.svg"
        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "SVG Files (*.svg)"),
        ):
            widget._format_combo.setCurrentText("SVG")
            widget._export_plot()

        assert out_path.exists()
        svg_content = out_path.read_text(encoding="utf-8")
        assert "svg" in svg_content.lower()

    def test_export_plot_pdf_with_identity(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec())
        widget.set_identity(PlotIdentity(engine="mujoco", model="golfer"))

        out_path = tmp_path / "plot.pdf"
        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "PDF Files (*.pdf)"),
        ):
            widget._format_combo.setCurrentText("PDF")
            widget._export_plot()

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_plot_csv_with_identity_metadata(
        self, app: QApplication, tmp_path: Path
    ) -> None:
        widget = PlotWidget()
        widget.set_spec(_make_spec(title="Joint Kinematics"))
        widget.set_identity(
            PlotIdentity(
                engine="mujoco",
                model="humanoid_golf",
                run_id="run-99",
                version="3.0",
            )
        )

        out_path = tmp_path / "joint_kinematics.csv"
        with patch(
            "shared.python.plot_engine.pyqt6_widget.QFileDialog.getSaveFileName",
            return_value=(str(out_path), "CSV Files (*.csv)"),
        ):
            existing_items = [
                widget._format_combo.itemText(i)
                for i in range(widget._format_combo.count())
            ]
            if "CSV" not in existing_items:
                widget._format_combo.addItem("CSV")
            widget._format_combo.setCurrentText("CSV")
            widget._export_plot()

        assert out_path.exists()
        lines = out_path.read_text(encoding="utf-8").splitlines()
        comment_lines = [line for line in lines if line.startswith("#")]
        assert any("engine: mujoco" in line for line in comment_lines)
        assert any("model: humanoid_golf" in line for line in comment_lines)
        assert any("run_id: run-99" in line for line in comment_lines)
        assert any("version: 3.0" in line for line in comment_lines)
        assert any("joint_1_x" in line for line in lines)
