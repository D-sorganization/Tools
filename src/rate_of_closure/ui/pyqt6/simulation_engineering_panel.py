"""Engineering-summary panel construction for the PyQt simulation view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class EngineeringPanelWidgets:
    """Widgets consumed by simulation rendering and expansion behavior."""

    panel: QWidget
    impact_summary: QLabel
    details_button: QToolButton
    screw_readout: QLabel
    impact_kinematics_readout: QLabel
    details_scroll: QScrollArea


def _create_detail_content() -> tuple[QWidget, QLabel, QLabel]:
    """Create the detailed screw and impact-kinematics readouts."""
    content = QWidget()
    details = QVBoxLayout(content)
    details.setContentsMargins(2, 2, 2, 2)
    screw_readout = QLabel()
    screw_readout.setWordWrap(True)
    screw_readout.setVisible(False)
    screw_readout.setToolTip(
        "Screw-motion readout in app frame x target, y up, z right."
    )
    details.addWidget(screw_readout)
    impact_readout = QLabel("Run a simulation to inspect impact kinematics.")
    impact_readout.setWordWrap(True)
    impact_readout.setTextFormat(Qt.TextFormat.RichText)
    impact_readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    impact_readout.setAccessibleName("Impact and Wedge Engineering Readout")
    impact_readout.setToolTip(
        "Detailed frame-explicit metrics, provenance, and model boundaries."
    )
    details.addWidget(impact_readout)
    details.addStretch(1)
    return content, screw_readout, impact_readout


def _create_details_scroll(content: QWidget) -> QScrollArea:
    """Create the collapsed, vertically bounded engineering-details viewport."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setMaximumHeight(150)
    scroll.setWidget(content)
    scroll.setVisible(False)
    scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    return scroll


def create_engineering_panel(
    set_expanded: Callable[[bool], None],
) -> EngineeringPanelWidgets:
    """Build the compact summary and expandable engineering-detail controls."""
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(4, 0, 4, 0)
    layout.setSpacing(3)
    summary = QLabel("Run a simulation to inspect key impact metrics.")
    summary.setWordWrap(True)
    summary.setAccessibleName("Key Impact Metrics")
    summary.setToolTip(
        "A compact summary of the current calculation; expand details for provenance."
    )
    layout.addWidget(summary)
    details_button = QToolButton()
    details_button.setText("Engineering Details")
    details_button.setCheckable(True)
    details_button.setArrowType(Qt.ArrowType.RightArrow)
    details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    details_button.setToolTip(
        "Show detailed impact, D-plane, screw-axis, and ground-clearance metrics."
    )
    details_button.toggled.connect(set_expanded)
    layout.addWidget(details_button)
    content, screw_readout, impact_readout = _create_detail_content()
    details_scroll = _create_details_scroll(content)
    layout.addWidget(details_scroll)
    return EngineeringPanelWidgets(
        panel=panel,
        impact_summary=summary,
        details_button=details_button,
        screw_readout=screw_readout,
        impact_kinematics_readout=impact_readout,
        details_scroll=details_scroll,
    )


__all__ = ["EngineeringPanelWidgets", "create_engineering_panel"]
