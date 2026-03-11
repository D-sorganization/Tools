"""Read-only material summary table for the vessel drafter GUI."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from vessel_drafter.analysis.vessel_drafter_metrics import MaterialMetricsReport


class MaterialSummaryTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 7, parent)
        self.setHorizontalHeaderLabels(
            [
                "Material",
                "Volume (ft^3)",
                "Area (ft^2)",
                "Density (lb/ft^3)",
                "Mass (lb)",
                "k (W/m-K)",
                "CTE (um/m-C)",
            ]
        )
        header = self.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)

    def set_report(self, report: MaterialMetricsReport) -> None:
        self.setRowCount(0)
        for metric in report.component_metrics:
            self._append_row(
                (
                    metric.display_name,
                    f"{metric.volume_ft3:.2f}",
                    f"{metric.surface_area_ft2:.2f}",
                    f"{metric.density_lb_per_ft3:.1f}",
                    f"{metric.mass_lb:.1f}",
                    f"{metric.thermal_conductivity_w_per_mk:.2f}",
                    f"{metric.thermal_expansion_um_per_m_c:.2f}",
                )
            )
        self._append_row(
            (
                "Total Refractory",
                f"{report.refractory_total_volume_ft3:.2f}",
                f"{report.refractory_total_surface_area_ft2:.2f}",
                "",
                f"{report.refractory_total_mass_lb:.1f}",
                "",
                "",
            )
        )

    def _append_row(
        self,
        values: tuple[str, str, str, str, str, str, str],
    ) -> None:
        row_index = self.rowCount()
        self.insertRow(row_index)
        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setItem(row_index, column_index, item)
