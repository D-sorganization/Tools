# mypy: ignore-errors
# ruff: noqa: E501
import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("p1am_control.desktop.settings")


class SettingsTab(QWidget):
    """Gear icon settings panel allowing customizable showing/hiding of main panel tabs."""

    # Signal emitted when a tab's visibility is toggled: (tab_key, visible)
    tabVisibilityChanged = pyqtSignal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("settings_tab")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        title = QLabel("Dashboard Settings", self)
        title.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)

        # Tab Visibility settings card
        vis_group = QGroupBox("Visible Tabs Selection", self)
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setSpacing(10)
        vis_layout.setContentsMargins(15, 15, 15, 15)

        self.chk_mimic = QCheckBox("Mimic Process Flow Diagram", self)
        self.chk_mimic.setChecked(True)
        self.chk_mimic.toggled.connect(
            lambda checked: self._on_toggled("mimic", checked)
        )

        self.chk_trends = QCheckBox("Real-Time Trends & Signal Filters", self)
        self.chk_trends.setChecked(True)
        self.chk_trends.toggled.connect(
            lambda checked: self._on_toggled("trends", checked)
        )

        self.chk_control = QCheckBox("PID Loops & MPC Groundwork", self)
        self.chk_control.setChecked(True)
        self.chk_control.toggled.connect(
            lambda checked: self._on_toggled("control", checked)
        )

        self.chk_routing = QCheckBox("DCS Routing Matrix & Interlocks", self)
        self.chk_routing.setChecked(True)
        self.chk_routing.toggled.connect(
            lambda checked: self._on_toggled("routing", checked)
        )

        self.chk_history = QCheckBox("Event History Log Query", self)
        self.chk_history.setChecked(True)
        self.chk_history.toggled.connect(
            lambda checked: self._on_toggled("history", checked)
        )

        vis_layout.addWidget(self.chk_mimic)
        vis_layout.addWidget(self.chk_trends)
        vis_layout.addWidget(self.chk_control)
        vis_layout.addWidget(self.chk_routing)
        vis_layout.addWidget(self.chk_history)

        layout.addWidget(vis_group)

        # Info panel
        info_group = QGroupBox("System Information", self)
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(15, 15, 15, 15)
        info_layout.addWidget(QLabel("Application Version: 1.0.0-PROD", self))
        info_layout.addWidget(QLabel("Modbus Driver: AsyncModbusManager", self))
        info_layout.addWidget(QLabel("Theme Provider: Catppuccin ThemeManager", self))
        layout.addWidget(info_group)

        layout.addStretch()

    def _on_toggled(self, tab_key: str, checked: bool) -> None:
        logger.info(f"Tab '{tab_key}' visibility requested change to: {checked}")
        self.tabVisibilityChanged.emit(tab_key, checked)
