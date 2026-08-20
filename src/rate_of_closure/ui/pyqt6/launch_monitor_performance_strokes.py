"""User-supplied expected-strokes action for the performance workspace."""

from __future__ import annotations

import pandas as pd
from PyQt6.QtWidgets import QComboBox, QLabel, QLineEdit

from rate_of_closure.launch_monitor_performance import (
    ScoreResult,
    StrokesGainedRequest,
    calculate_strokes_gained,
)


class PerformanceStrokesMixin:
    """Keep the carefully labelled non-baseline SG action modular."""

    _frame: pd.DataFrame
    before_combo: QComboBox
    after_combo: QComboBox
    baseline_url: QLineEdit
    strokes_status: QLabel

    def run_strokes(self) -> ScoreResult:
        result = calculate_strokes_gained(
            self._frame,
            StrokesGainedRequest(
                self.before_combo.currentText(),
                self.after_combo.currentText(),
                self.baseline_url.text().strip(),
            ),
        )
        self.strokes_status.setText(
            f"Mean user-supplied expected-strokes SG {result.mean:.3f} strokes. "
            f"{result.formula} User citation (not validated baseline): "
            f"{result.source_url}"
        )
        return result


__all__ = ["PerformanceStrokesMixin"]
