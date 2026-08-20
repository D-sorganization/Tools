"""Focused presentation widgets for the Neural Model Lab."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QWidget


class CapabilityCanvas(QWidget):
    """Always-visible, unit-labelled vendor eligibility comparison."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(240)
        self.setAccessibleName("Vendor Strict Eligible Input Rows Chart")
        self.setToolTip(
            "Strict complete five-input row counts; policy blockers still govern "
            "whether training is allowed."
        )
        self._rows: tuple[tuple[str, int], ...] = ()

    def set_capabilities(self, vendors: object) -> None:
        self._rows = tuple(
            (str(item.vendor), int(item.strict_row_count)) for item in vendors
        )
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setPen(QPen(self.palette().text().color()))
        painter.drawText(8, 18, "Strict eligible input rows (count)")
        maximum = max((count for _, count in self._rows), default=1)
        usable = max(1, self.width() - 190)
        for index, (vendor, count) in enumerate(self._rows):
            y = 44 + index * 55
            painter.drawText(8, y + 18, vendor)
            painter.setPen(QPen(Qt.GlobalColor.cyan, 22))
            width = max(1, round(usable * count / maximum))
            painter.drawLine(105, y + 10, 105 + width, y + 10)
            painter.setPen(QPen(self.palette().text().color()))
            painter.drawText(115 + width, y + 18, f"{count:,} rows")


class ResidualPlot(QWidget):
    """Small row-aligned residual plot with explicit unavailable state."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(180)
        self.setAccessibleName("Held-Out Row-Aligned Residual Plot")
        self._values: list[float] = []
        self._reason = "No validated model is loaded."
        self.setToolTip(f"Residual plot unavailable: {self._reason}")

    def set_residuals(self, residuals: dict[str, object]) -> None:
        rows = residuals.get("rows")
        self._values = (
            [
                float(row["residual"])
                for row in rows
                if isinstance(row, dict)
                and isinstance(row.get("residual"), (int, float))
            ]
            if residuals.get("state") == "available" and isinstance(rows, list)
            else []
        )
        self._reason = str(
            residuals.get("reason", "Row-aligned held-out residuals were not exported.")
        )
        self.setToolTip(
            self._reason
            if not self._values
            else "Residual by aligned held-out row; zero is perfect prediction."
        )
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setPen(QPen(self.palette().text().color()))
        if not self._values:
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                f"Residual plot unavailable: {self._reason}",
            )
            return
        margin = 38
        width = max(1, self.width() - 55)
        height = max(1, self.height() - 45)
        center = margin + height / 2
        extent = max(1.0, *(abs(value) for value in self._values))
        painter.drawLine(margin, int(center), margin + width, int(center))
        painter.drawText(4, 16, "Residual (target unit)")
        painter.drawText(self.width() - 150, self.height() - 5, "Aligned held-out row")
        painter.setPen(QPen(Qt.GlobalColor.cyan, 5))
        for index, value in enumerate(self._values):
            x = margin + index * width / max(1, len(self._values) - 1)
            y = center - value * height * 0.45 / extent
            painter.drawPoint(int(x), int(y))


__all__ = ["CapabilityCanvas", "ResidualPlot"]
