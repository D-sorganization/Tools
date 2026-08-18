"""File export operations for the investigative plot workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from rate_of_closure.plotting import (
    PlotData,
    PlotSpec,
    spec_from_json,
    spec_to_json,
    write_plot_csv,
    write_plot_json,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PlotExportMixin:
    """Provide reproducible plot image, data, and definition exports."""

    _data: PlotData | None
    _figure: Figure

    if TYPE_CHECKING:

        def add_spec(self, spec: PlotSpec) -> None: ...

        def current_spec(self) -> PlotSpec | None: ...

        def refresh(self) -> None: ...

    def _dialog_parent(self) -> QWidget:
        """Return this mixin's concrete QWidget owner."""
        return cast(QWidget, self)

    def _ready_for_export(self) -> bool:
        if self._data is None:
            self.refresh()
        if self._data is None:
            QMessageBox.information(
                self._dialog_parent(),
                "Export",
                "Nothing to export yet — select a plot first.",
            )
            return False
        return True

    def _export_image(self, fmt: str) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            f"Export {fmt.upper()}",
            f"plot.{fmt}",
            f"{fmt.upper()} image (*.{fmt})",
        )
        if path:
            self.save_image(path)

    def save_image(self, path: str) -> None:
        """Save the rendered figure, inferring its format from the suffix."""
        self._figure.savefig(path)

    def _export_csv(self) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(), "Export Data CSV", "plot_data.csv", "CSV (*.csv)"
        )
        if path:
            assert self._data is not None
            write_plot_csv(self._data, path)

    def _export_json(self) -> None:
        if not self._ready_for_export():
            return
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Export Data JSON",
            "plot_data.json",
            "JSON (*.json)",
        )
        if path:
            assert self._data is not None
            write_plot_json(self._data, path)

    def _save_definition(self) -> None:
        spec = self.current_spec()
        if spec is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(),
            "Save Plot Definition",
            "plot_definition.json",
            "JSON (*.json)",
        )
        if path:
            spec_to_json(spec, path)

    def _load_definition(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(), "Load Plot Definition", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.add_spec(spec_from_json(path))
        except Exception as exc:  # noqa: BLE001 -- user-provided file boundary
            QMessageBox.warning(self._dialog_parent(), "Load Plot Definition", str(exc))
