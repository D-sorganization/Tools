"""Selected-row presentation for the launch-monitor linked scatter."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from rate_of_closure.launch_monitor_linked_scatter import LinkedScatterPlan
from rate_of_closure.ui.pyqt6.launch_monitor_preview import LaunchMonitorPreviewCanvas


class LaunchMonitorLinkedScatterPanel(QWidget):
    """Own one generation-safe retained-row selection and its visible status."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview = LaunchMonitorPreviewCanvas()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setAccessibleName("Linked Scatter Selected Retained Row")
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        self._generation = 0
        self._selection: tuple[int, int] | None = None
        self._frame: pd.DataFrame | None = None
        self._outcome = ""
        self._predictors: tuple[str, ...] = ()
        self.preview.selection_changed.connect(self._select_raw_row)

    @property
    def selected_raw_index(self) -> int | None:
        """Return the selected ordinal only when it belongs to this generation."""
        if self._selection is None or self._selection[0] != self._generation:
            return None
        return self._selection[1]

    def reset_dataset(self) -> None:
        """Clear selection atomically when a new retained dataset is installed."""
        self._generation += 1
        self._selection = None

    def set_frame(
        self, frame: pd.DataFrame, outcome: str, predictors: tuple[str, ...]
    ) -> LinkedScatterPlan:
        """Render without analyzing or mutating the retained records."""
        self._frame = frame
        self._outcome = outcome
        self._predictors = predictors
        plan = self.preview.set_frame(
            frame, outcome, predictors, self.selected_raw_index
        )
        self.status.setText(self._status_text(plan, self.selected_raw_index))
        return plan

    def _select_raw_row(self, raw_index: object) -> None:
        self._selection = (
            None if raw_index is None else (self._generation, int(raw_index))
        )
        if self._frame is not None:
            self.set_frame(self._frame, self._outcome, self._predictors)

    @staticmethod
    def _status_text(plan: LinkedScatterPlan, selected_raw_index: int | None) -> str:
        prefix = (
            f"Displayed {plan.displayed_count:,} of {plan.finite_count:,} finite "
            f"pairs from {plan.raw_count:,} retained rows. All rows remain retained "
            "for export; the selected missing-data policy controls analysis inclusion. "
        )
        point = next(
            (item for item in plan.points if item.raw_index == selected_raw_index),
            None,
        )
        if selected_raw_index is None:
            return prefix + "No retained source row selected."
        if point is None:
            return prefix + (
                f"Retained row index {selected_raw_index} is unavailable for the "
                "current axes."
            )
        fields = tuple(
            text
            for text in (
                f"shot {point.shot_id}" if point.shot_id else None,
                f"session {point.session_id}" if point.session_id else None,
                f"vendor {point.monitor_vendor}" if point.monitor_vendor else None,
            )
            if text is not None
        )
        return prefix + (
            f"Retained row index {point.raw_index} (zero-based); "
            f"{'; '.join(fields) if fields else 'no source identifiers present'}; "
            f"{plan.x_field} {point.x:g}; {plan.y_field} {point.y:g}."
        )


__all__ = ["LaunchMonitorLinkedScatterPanel"]
