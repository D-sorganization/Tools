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

from p1am_control_system.desktop.tab_labels import TAB_TITLES, TOGGLEABLE_TAB_ORDER

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

        self.chk_mimic = QCheckBox(TAB_TITLES["mimic"], self)
        self.chk_mimic.setChecked(True)
        self.chk_mimic.toggled.connect(
            lambda checked: self._on_toggled("mimic", checked)
        )

        self.chk_trends = QCheckBox(TAB_TITLES["trends"], self)
        self.chk_trends.setChecked(True)
        self.chk_trends.toggled.connect(
            lambda checked: self._on_toggled("trends", checked)
        )

        self.chk_control = QCheckBox(TAB_TITLES["control"], self)
        self.chk_control.setChecked(True)
        self.chk_control.toggled.connect(
            lambda checked: self._on_toggled("control", checked)
        )

        self.chk_routing = QCheckBox(TAB_TITLES["routing"], self)
        self.chk_routing.setChecked(True)
        self.chk_routing.toggled.connect(
            lambda checked: self._on_toggled("routing", checked)
        )

        self.chk_history = QCheckBox(TAB_TITLES["history"], self)
        self.chk_history.setChecked(True)
        self.chk_history.toggled.connect(
            lambda checked: self._on_toggled("history", checked)
        )

        self._tab_checkboxes = {
            "mimic": self.chk_mimic,
            "trends": self.chk_trends,
            "control": self.chk_control,
            "routing": self.chk_routing,
            "history": self.chk_history,
        }

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

    def set_tab_visible(
        self, tab_key: str, visible: bool, *, emit: bool = True
    ) -> None:
        """Set a tab checkbox while optionally suppressing the visibility signal."""
        if tab_key not in TOGGLEABLE_TAB_ORDER:
            raise ValueError(f"unknown toggleable tab: {tab_key}")
        checkbox = self._tab_checkboxes[tab_key]
        was_blocked = checkbox.blockSignals(not emit)
        try:
            checkbox.setChecked(visible)
        finally:
            checkbox.blockSignals(was_blocked)

    def visible_tabs(self) -> dict[str, bool]:
        """Return the current visibility checkbox state by tab key."""
        return {
            tab_key: self._tab_checkboxes[tab_key].isChecked()
            for tab_key in TOGGLEABLE_TAB_ORDER
        }
