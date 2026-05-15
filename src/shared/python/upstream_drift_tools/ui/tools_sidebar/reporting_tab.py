"""Agentic Reporting and Summarization Engine tab for Sidekick."""

from __future__ import annotations

import logging
from typing import Any

from . import design_tokens as theme
from .qt_compat import QtCore, QtWidgets

logger = logging.getLogger(__name__)


class SidekickReportingWidget(QtWidgets.QWidget):
    """Widget for generating agentic session reports."""

    def __init__(self, sidebar: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.sidebar = sidebar
        self.setObjectName("SidekickReportingWidget")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(
            theme.SIDEBAR_LAYOUT_MARGINS[0],
            theme.SIDEBAR_LAYOUT_MARGINS[1],
            theme.SIDEBAR_LAYOUT_MARGINS[2],
            theme.SIDEBAR_LAYOUT_MARGINS[3],
        )
        layout.setSpacing(theme.SIDEBAR_LAYOUT_SPACING)

        # Info label
        self._info_label = QtWidgets.QLabel(
            "Generate a comprehensive report of your current session. "
            "This will aggregate workspace context, chat interactions, "
            "and terminal history."
        )
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Generate Button
        self._generate_btn = QtWidgets.QPushButton("Generate Session Report")
        self._generate_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self._generate_btn)

        # Preview area
        self._report_preview = QtWidgets.QTextEdit()
        self._report_preview.setReadOnly(True)
        self._report_preview.setPlaceholderText("Report preview will appear here...")
        layout.addWidget(self._report_preview, stretch=1)

        # Save Button
        self._save_btn = QtWidgets.QPushButton("Save Report")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

    def _on_generate_clicked(self) -> None:
        """Gather context and trigger report generation."""
        self._report_preview.setPlainText("Gathering context and generating report...")
        self._generate_btn.setEnabled(False)

        # 1. Gather Context
        # This is a placeholder for actual context gathering logic
        context_data = {
            "workspace_vars": [v.name for v in self.sidebar.registry.variables()],
            "project_root": str(self.sidebar.project_root),
        }

        # 2. Trigger LLM generation (placeholder)
        # Ideally, we send this to the ChatService backend
        report = (
            f"# Session Report\n\n"
            f"## Workspace Variables\n{context_data['workspace_vars']}\n\n"
            f"## Project Root\n{context_data['project_root']}"
        )

        # Simulate async delay
        QtCore.QTimer.singleShot(1000, lambda: self._on_report_generated(report))

    def _on_report_generated(self, report: str) -> None:
        """Handle the generated report."""
        self._report_preview.setPlainText(report)
        self._generate_btn.setEnabled(True)
        self._save_btn.setEnabled(True)

    def _on_save_clicked(self) -> None:
        """Save the generated report to disk."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Session Report",
            "session_report.md",
            "Markdown Files (*.md);;All Files (*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._report_preview.toPlainText())
            except Exception as e:
                logger.error("Failed to save report: %s", e)


def build_reporting_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the Reporting tab for the Sidekick sidebar."""
    widget = SidekickReportingWidget(sidebar=sidebar, parent=sidebar)
    widget.setToolTip("Generate an agentic summary and report of the session.")
    return widget
