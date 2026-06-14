# mypy: ignore-errors
# ruff: noqa: E501
import logging

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)
from theme.theme_manager import get_theme_manager

from p1am_control_system.desktop.auth import AuthManager, Role

logger = logging.getLogger("p1am_control.desktop.header")


class HMIHeader(QWidget):
    """Industrial HMI Header showing:
    - Active Modbus/DCS connection state.
    - Current logged-in role (Operator or Admin).
    - E-Stop status button (clear pressed/unpressed state).
    - Theme toggle (Light/Dark).
    - Alarm Acknowledgment button (flashes on active unacknowledged alarms).
    """

    roleChanged = pyqtSignal(str)  # Emits "Operator" or "Admin"
    estopTriggered = pyqtSignal(
        bool
    )  # Emits True if pressed (emergency active), False if cleared
    themeToggled = pyqtSignal(str)  # Emits the new theme name
    alarmAcknowledgeClicked = pyqtSignal()  # Emits when clicked

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hmi_header")
        self.theme_manager = get_theme_manager()
        self.auth_manager = AuthManager()

        # State variables
        self._current_role = "Operator"
        self._connection_state = "Offline"
        self._has_hl_alarms = False
        self._has_hhll_alarms = False
        self._flash_state = False
        self._pending_estop_clear = False

        self._init_ui()
        self.theme_manager.themeChanged.connect(self.apply_theme_styles)
        self.apply_theme_styles(self.theme_manager.get_current_theme_name())

        # Setup flashing timer for unacknowledged alarms
        self.flash_timer = QTimer(self)
        self.flash_timer.setInterval(500)  # Flash every 500ms
        self.flash_timer.timeout.connect(self._toggle_flash)
        self.flash_timer.start()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(15)

        # Title / Brand Label
        self.title_label = QLabel("P1AM GASIFICATION PLANT HMI", self)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Connection Status Label
        self.conn_label = QLabel("PLC Connection: Offline", self)
        self.conn_label.setStyleSheet(
            "font-weight: 500; font-size: 10pt; color: red; padding: 4px 8px; border: 1px solid red; border-radius: 4px;"
        )
        layout.addWidget(self.conn_label)

        # Role Selector Combo Box
        role_label = QLabel("Role:", self)
        role_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        layout.addWidget(role_label)

        self.role_combo = QComboBox(self)
        self.role_combo.addItems(["Operator", "Admin"])
        self.role_combo.currentTextChanged.connect(self._on_role_changed)
        self.role_combo.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.role_combo)

        # Alarm Acknowledgment Button (flashing)
        self.ack_btn = QPushButton("ACK ALARMS", self)
        self.ack_btn.clicked.connect(self._on_ack_clicked)
        self.ack_btn.setMinimumWidth(120)
        self.ack_btn.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.ack_btn)

        # E-Stop Button (Checkable status button)
        self.estop_btn = QPushButton("E-STOP CLEAR", self)
        self.estop_btn.setCheckable(True)
        self.estop_btn.toggled.connect(self._on_estop_toggled)
        self.estop_btn.setMinimumWidth(150)
        self._update_estop_style(False)
        layout.addWidget(self.estop_btn)

        # Theme Toggle Button
        self.theme_btn = QPushButton("Theme: Dark", self)
        self.theme_btn.clicked.connect(self._on_theme_toggled)
        self.theme_btn.setMinimumWidth(110)
        layout.addWidget(self.theme_btn)

        # Apply default QSS styles for the header container
        self.setStyleSheet("min-height: 50px;")

    def _theme_color(self, key: str, fallback: str) -> str:
        return self.theme_manager.get_current_colors().get(key, fallback)

    def _status_label_style(self, color: str) -> str:
        return (
            "font-weight: bold; font-size: 10pt; "
            f"color: {color}; padding: 4px 8px; "
            f"border: 1px solid {color}; border-radius: 4px;"
        )

    def _alarm_button_style(self, background: str, foreground: str) -> str:
        return (
            f"font-weight: bold; background-color: {background}; color: {foreground};"
        )

    def _estop_button_style(self, background: str, foreground: str) -> str:
        return (
            "QPushButton {"
            f"  background-color: {background}; color: {foreground}; "
            "font-weight: bold; font-size: 11pt;"
            "}"
        )

    def apply_theme_styles(self, theme_name: str | None = None) -> None:
        """Re-apply widget-local HMI styles from the active shared theme."""
        active_theme = theme_name or self.theme_manager.get_current_theme_name()
        self.theme_btn.setText(f"Theme: {active_theme}")
        self.setStyleSheet("min-height: 50px;")
        self.set_connection_status(self._connection_state)
        self._update_estop_style(
            self.estop_btn.isChecked(), pending_clear=self._pending_estop_clear
        )
        if not self._has_hl_alarms and not self._has_hhll_alarms:
            self.ack_btn.setStyleSheet("font-weight: bold;")

    def _on_role_changed(self, text: str) -> None:
        if text == "Admin":
            password, ok = QInputDialog.getText(
                self,
                "Admin Authentication",
                "Enter Admin Password:",
                QLineEdit.EchoMode.Password,
            )
            if ok and self.auth_manager.login(Role.ADMIN, password):
                self._current_role = "Admin"
                logger.info("Admin authentication successful.")
                self.roleChanged.emit("Admin")
            else:
                # Revert to Operator without triggering self._on_role_changed recursively
                self.role_combo.blockSignals(True)
                self.role_combo.setCurrentText("Operator")
                self.role_combo.blockSignals(False)
                self._current_role = "Operator"

                if ok:
                    QMessageBox.warning(
                        self, "Access Denied", "Invalid Admin Password."
                    )
                else:
                    logger.info("Admin authentication cancelled.")
        else:
            self.auth_manager.login(Role.OPERATOR)
            self._current_role = "Operator"
            logger.info("Switched to Operator role.")
            self.roleChanged.emit("Operator")

    def _on_estop_toggled(self, checked: bool) -> None:
        if checked:
            # Tripping is immediate and fail-safe: reflect it right away.
            self._update_estop_style(True)
        else:
            # Clearing must be confirmed by the controller before the button may
            # show the green "CLEAR" state; until then show a pending state so the
            # header never claims "clear" while the plant is still tripped.
            self._update_estop_style(True, pending_clear=True)
        self.estopTriggered.emit(checked)

    def confirm_estop_cleared(self) -> None:
        """Mark the E-stop as confirmed-cleared after the PLC acknowledged.

        Called only on a successful backend clear so the green state reflects the
        real plant state rather than an optimistic local toggle.
        """
        self.estop_btn.blockSignals(True)
        self.estop_btn.setChecked(False)
        self.estop_btn.blockSignals(False)
        self._update_estop_style(False)

    def revert_estop_to_tripped(self) -> None:
        """Restore the tripped (red) state after a failed clear attempt.

        Keeps the button latched/red so the operator is not misled into thinking
        the plant was released when the controller did not acknowledge the reset.
        """
        self.estop_btn.blockSignals(True)
        self.estop_btn.setChecked(True)
        self.estop_btn.blockSignals(False)
        self._update_estop_style(True)

    def _update_estop_style(self, active: bool, pending_clear: bool = False) -> None:
        self._pending_estop_clear = pending_clear
        selection_text = self._theme_color("selection_text", "white")
        if pending_clear:
            self.estop_btn.setText("CLEARING…")
            self.estop_btn.setStyleSheet(
                self._estop_button_style(
                    self._theme_color("warning", "orange"), "black"
                )
            )
        elif active:
            self.estop_btn.setText("E-STOP PRESSED")
            self.estop_btn.setStyleSheet(
                self._estop_button_style(
                    self._theme_color("error", "red"), selection_text
                )
            )
        else:
            self.estop_btn.setText("E-STOP CLEAR")
            self.estop_btn.setStyleSheet(
                self._estop_button_style(
                    self._theme_color("success", "green"), selection_text
                )
            )

    def _on_theme_toggled(self) -> None:
        current_theme = self.theme_manager.get_current_theme_name()
        new_theme = "Dark" if current_theme == "Light" else "Light"
        self.theme_manager.change_theme(new_theme)
        self.theme_btn.setText(f"Theme: {new_theme}")
        self.themeToggled.emit(new_theme)

    def _on_ack_clicked(self) -> None:
        logger.info("Alarm acknowledgment clicked.")
        self.alarmAcknowledgeClicked.emit()

    def set_connection_status(self, state: str) -> None:
        """Sets connection state display: 'Connected', 'Simulating', or 'Offline'."""
        self._connection_state = state
        self.conn_label.setText(f"PLC Connection: {state}")

        if state == "Connected":
            self.conn_label.setStyleSheet(
                self._status_label_style(self._theme_color("success", "green"))
            )
        elif state == "Simulating":
            self.conn_label.setStyleSheet(
                self._status_label_style(self._theme_color("warning", "orange"))
            )
        else:
            self.conn_label.setStyleSheet(
                self._status_label_style(self._theme_color("error", "red"))
            )

    def set_alarms_state(self, has_hl: bool, has_hhll: bool) -> None:
        """Updates internal alarm flags to control flashing."""
        self._has_hl_alarms = has_hl
        self._has_hhll_alarms = has_hhll
        if not has_hl and not has_hhll:
            # Reset button styling immediately if no alarms
            self.ack_btn.setStyleSheet("font-weight: bold;")

    def _toggle_flash(self) -> None:
        if not self._has_hl_alarms and not self._has_hhll_alarms:
            return

        self._flash_state = not self._flash_state

        if self._flash_state:
            # Flashing state color
            if self._has_hhll_alarms:
                # Flash Red for HH/LL alarms
                self.ack_btn.setStyleSheet(
                    self._alarm_button_style(
                        self._theme_color("error", "red"),
                        self._theme_color("selection_text", "white"),
                    )
                )
            elif self._has_hl_alarms:
                # Flash Yellow for H/L alarms
                self.ack_btn.setStyleSheet(
                    self._alarm_button_style(
                        self._theme_color("warning", "orange"), "black"
                    )
                )
        else:
            # Default state color during flash
            self.ack_btn.setStyleSheet("font-weight: bold;")
