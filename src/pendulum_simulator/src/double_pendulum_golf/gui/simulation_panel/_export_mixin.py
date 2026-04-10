"""
Export helpers for SimulationPanel — image/data/video export methods.

Factored out of the original ``simulation_panel.py`` as a mixin so that the
main class definition stays focused on UI wiring and animation logic.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, cast

from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QWidget,
)

logger = logging.getLogger(__name__)


class _SimulationExportMixin:
    """Mixin providing export_image / _on_export_data / _on_export_video.

    Requires the host class to expose:
      - ``self._result`` (simulation result or None)
      - ``self.pendulum`` (QWidget-compatible viewer)
      - ``self._display_frame(idx)``
      - ``self._timer`` (QTimer)
      - ``self.ANIMATION_INTERVAL_MS`` (class constant)
    """

    # Attributes provided by SimulationPanel — declared here for type clarity
    _result: Any
    pendulum: Any
    _timer: Any
    ANIMATION_INTERVAL_MS: int

    def export_image(self) -> None:
        """Export the current pendulum visualization as PNG, SVG, or PDF."""
        if self._result is None:
            QMessageBox.information(
                cast("QWidget", self), "Export Image", "Run a simulation first."
            )
            return

        path, _selected_filter = QFileDialog.getSaveFileName(
            cast("QWidget", self),
            "Export Image",
            "",
            "PNG Files (*.png);;SVG Files (*.svg);;PDF Files (*.pdf)",
        )
        if not path:
            return

        try:
            if path.endswith(".png"):
                self._export_as_png(path)
            elif path.endswith(".svg"):
                self._export_as_svg(path)
            elif path.endswith(".pdf"):
                self._export_as_pdf(path)
            else:
                # Default to PNG if extension unclear
                if not path.endswith("."):
                    path += ".png"
                self._export_as_png(path)

            QMessageBox.information(
                cast("QWidget", self), "Export Image", f"Saved image to:\n{path}"
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to export image: %s", e)
            QMessageBox.critical(
                cast("QWidget", self),
                "Export Image",
                f"Failed to export image:\n{e}",
            )

    def _export_as_png(self, path: str) -> None:
        """Export the pendulum widget as a PNG image."""
        pix = cast("QWidget", self.pendulum).grab()
        if not pix.save(path):
            raise OSError(f"Failed to save PNG to {path}")
        logger.info("Exported PNG: %s", path)

    def _export_as_svg(self, path: str) -> None:
        """Export the pendulum widget as an SVG image."""
        assert path is not None, "path must be provided"
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QPainter

        widget = cast("QWidget", self.pendulum)
        rect = QRect(0, 0, widget.width(), widget.height())

        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(rect.size())
        generator.setViewBox(rect)
        generator.setTitle("Pendulum Visualization")
        generator.setDescription("Exported from Pendulum Simulator")

        painter = QPainter()
        painter.begin(generator)
        widget.render(painter)
        painter.end()

        logger.info("Exported SVG: %s", path)

    def _export_as_pdf(self, path: str) -> None:
        """Export the pendulum widget as a PDF (via QPrinter)."""
        assert path is not None, "path must be provided"
        from PyQt6.QtCore import QMarginsF
        from PyQt6.QtGui import QPainter
        from PyQt6.QtPrintSupport import QPrinter

        widget = cast("QWidget", self.pendulum)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageMargins(QMarginsF(0, 0, 0, 0))

        painter = QPainter()
        painter.begin(printer)
        widget.render(painter)
        painter.end()

        logger.info("Exported PDF: %s", path)

    def _on_export_data(self) -> None:
        if self._result is None:
            QMessageBox.information(
                cast("QWidget", self), "Export Data", "Run a simulation first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            cast("QWidget", self),
            "Export Data",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        headers = ["t"]
        if self._result.states.shape[1] == 4:
            headers += [
                "tau_drive_1",
                "tau_drive_2",
                "tau_friction_1",
                "tau_friction_2",
                "tau_total_1",
                "tau_total_2",
                "shoulder_fx",
                "shoulder_fy",
                "wrist_fx",
                "wrist_fy",
            ]
        else:
            headers += [
                "tau_drive_1",
                "tau_drive_2",
                "tau_drive_3",
                "tau_friction_1",
                "tau_friction_2",
                "tau_friction_3",
                "tau_total_1",
                "tau_total_2",
                "tau_total_3",
                "shoulder_fx",
                "shoulder_fy",
                "wrist1_fx",
                "wrist1_fy",
                "wrist2_fx",
                "wrist2_fy",
            ]

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for i in range(self._result.n_steps):
                    t = self._result.t[i]
                    tau_drive = self._result.torques_at(i)
                    forces = self._result.joint_forces_at(i)

                    if self._result.states.shape[1] == 4:
                        tau_friction = self._result.friction_torques_at(i)
                        tau_total = self._result.total_torques_at(i)
                        row = [
                            t,
                            tau_drive[0],
                            tau_drive[1],
                            tau_friction[0],
                            tau_friction[1],
                            tau_total[0],
                            tau_total[1],
                            forces["shoulder"][0],
                            forces["shoulder"][1],
                            forces["wrist"][0],
                            forces["wrist"][1],
                        ]
                    else:
                        row = [
                            t,
                            tau_drive[0],
                            tau_drive[1],
                            tau_drive[2],
                            0.0,
                            0.0,
                            0.0,  # friction not yet in triple model
                            tau_drive[0],
                            tau_drive[1],
                            tau_drive[2],
                            forces["shoulder"][0],
                            forces["shoulder"][1],
                            forces["wrist1"][0],
                            forces["wrist1"][1],
                            forces["wrist2"][0],
                            forces["wrist2"][1],
                        ]
                    writer.writerow(row)

        except OSError as e:
            QMessageBox.critical(
                cast("QWidget", self), "Export Data", f"Failed to write file: {e}"
            )
            return

        QMessageBox.information(
            cast("QWidget", self), "Export Data", f"Saved data to:\n{path}"
        )

    def _on_export_video(self) -> None:
        if self._result is None:
            QMessageBox.information(
                cast("QWidget", self), "Export Video", "Run a simulation first."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            cast("QWidget", self),
            "Export Video",
            "",
            "MP4 Video (*.mp4);;GIF (*.gif)",
        )
        if not path:
            return

        ffmpeg_path = shutil.which("ffmpeg")
        was_playing = self._timer.isActive()
        self._timer.stop()

        tmp_dir = tempfile.mkdtemp(prefix="pendulum_frames_")
        try:
            for i in range(self._result.n_steps):
                self._display_frame(i)
                QApplication.processEvents()
                pix = cast("QWidget", self.pendulum).grab()
                frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
                pix.save(frame_path)

            if ffmpeg_path is None:
                out_dir = os.path.splitext(path)[0] + "_frames"
                os.makedirs(out_dir, exist_ok=True)
                for name in os.listdir(tmp_dir):
                    shutil.move(
                        os.path.join(tmp_dir, name),
                        os.path.join(out_dir, name),
                    )
                QMessageBox.warning(
                    cast("QWidget", self),
                    "Export Video",
                    "ffmpeg not found. Exported PNG frames instead:\n" + out_dir,
                )
                return

            fps = int(1000 / self.ANIMATION_INTERVAL_MS)
            cmd = [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(tmp_dir, "frame_%05d.png"),
                "-pix_fmt",
                "yuv420p",
                path,
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                QMessageBox.critical(
                    cast("QWidget", self),
                    "Export Video",
                    "ffmpeg failed. Check your ffmpeg installation.",
                )
                return

            QMessageBox.information(
                cast("QWidget", self), "Export Video", f"Saved video to:\n{path}"
            )
        finally:
            if was_playing:
                self._timer.start()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _display_frame(self, idx: int) -> None:  # pragma: no cover - overridden
        """Overridden by SimulationPanel — mixin stub so mypy is happy."""
        raise NotImplementedError
