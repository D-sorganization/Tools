# TRACKED_TASK: see #2310 — architecture debt extraction schedule

#!/usr/bin/env python3
"""Calculator State Mixin
    def register_copyable_widget(self, widget: Any, widget_type: str = "text") -> None:
        """Register a widget for copy/paste operations

        Args:
            widget: Widget to register (QTableWidget, QTextEdit, QLabel, etc.)
            widget_type: Type of widget for appropriate copy handling

        """
        if not (widget_type is not None):
            raise ValueError("widget_type must be provided")
        widget_info = {"widget": widget, "type": widget_type}

        self.copyable_widgets.append(widget_info)  # type: ignore[attr-defined]

        # Setup context menu for the widget
        if hasattr(widget, "setContextMenuPolicy"):
            widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            widget.customContextMenuRequested.connect(
                lambda pos, w=widget_info: self.show_widget_context_menu(pos, w),
            )

    def copy_selected_text(self, checked: bool = False) -> None:
        """Copy selected text from focused widget"""
        try:
            focused_widget = self.focusWidget()  # type: ignore[attr-defined]
            if focused_widget:
                text = self.get_text_from_widget(focused_widget)
                if text:
                    self.copy_to_clipboard(text)
                    return

            # If no focused widget or no selection, try to copy from any copyable widget
            for widget_info in self.copyable_widgets:  # type: ignore[attr-defined]
                widget = widget_info["widget"]
                if widget.hasFocus():
                    text = self.get_text_from_widget(widget)
                    if text:
                        self.copy_to_clipboard(text)
                        return

        except (KeyError, ValueError, TypeError):
            pass

    def copy_all_results(self, checked: bool = False) -> None:
        """Copy all results from the calculator"""
        try:
            all_text = []

            # Collect text from all copyable widgets
            for widget_info in self.copyable_widgets:  # type: ignore[attr-defined]
                widget = widget_info["widget"]
                text = self.get_text_from_widget(widget)
                if text:
                    all_text.append(text)

            if all_text:
                combined_text = "\n\n".join(all_text)
                self.copy_to_clipboard(combined_text)
                logger.info("All results copied to clipboard")
            else:
                logger.debug("No copyable results available")

        except (KeyError, ValueError, TypeError):
            pass

    def get_text_from_widget(self, widget: Any) -> str:
        """Extract text from various widget types"""
        try:
            if isinstance(widget, QTableWidget):
                return self.get_table_text(widget)
            if isinstance(widget, QTextEdit):
                return widget.toPlainText()
            if isinstance(widget, QLabel):
                return widget.text()
            if hasattr(widget, "text"):
                return str(widget.text())
            if hasattr(widget, "toPlainText"):
                return str(widget.toPlainText())
            return ""
        except (RuntimeError, AttributeError):
            return ""

    def get_table_text(self, table: QTableWidget) -> str:
        """Extract text from QTableWidget with formatting"""
        try:
            text_lines = []

            # Add headers
            headers = []
            for col in range(table.columnCount()):
                header = table.horizontalHeaderItem(col)
                if header:
                    headers.append(header.text())
                else:
                    headers.append(f"Column {col}")
            text_lines.append("\t".join(headers))

            # Add data rows
            for row in range(table.rowCount()):
                row_data = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                text_lines.append("\t".join(row_data))

            return "\n".join(text_lines)

        except (RuntimeError, AttributeError):
            return ""

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text to clipboard"""
        try:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(text)
        except (RuntimeError, AttributeError):
            pass

    def paste_text(self, checked: bool = False) -> None:
        """Paste text from clipboard to focused widget"""
        try:
            clipboard = QApplication.clipboard()
            text = clipboard.text() if clipboard else ""

            if text:
                focused_widget = self.focusWidget()  # type: ignore[attr-defined]
                if focused_widget and hasattr(focused_widget, "setText"):
                    focused_widget.setText(text)
                elif focused_widget and hasattr(focused_widget, "setPlainText"):
                    focused_widget.setPlainText(text)
                elif focused_widget and hasattr(focused_widget, "insertPlainText"):
                    focused_widget.insertPlainText(text)

        except (RuntimeError, AttributeError):
            pass

    def show_context_menu(self, position: Any) -> None:
        """Show context menu for the calculator"""
        menu = QMenu(cast(QWidget, self))
        # Copy actions
        copy_action = menu.addAction("Copy Selected")
        if copy_action is not None:
            copy_action.triggered.connect(self.copy_selected_text)

        copy_all_action = menu.addAction("Copy All Results")
        if copy_all_action is not None:
            copy_all_action.triggered.connect(self.copy_all_results)

        menu.addSeparator()

        # Paste action
        paste_action = menu.addAction("Paste")
        if paste_action is not None:
            paste_action.triggered.connect(self.paste_text)

        menu.addSeparator()

        # State management actions
        save_action = menu.addAction("Save State")
        if save_action is not None:
            save_action.triggered.connect(lambda: self.save_calculator_state())  # type: ignore[attr-defined]

        load_action = menu.addAction("Load State")
        if load_action is not None:
            load_action.triggered.connect(lambda: self.load_calculator_state())  # type: ignore[attr-defined]

        menu.exec(cast(QWidget, self).mapToGlobal(position))

    def show_widget_context_menu(
        self, position: Any, widget_info: dict[str, Any]
    ) -> None:
        """Show context menu for a specific widget"""
        if not (widget_info is not None):
            raise ValueError("widget_info must be provided")
        menu = QMenu(cast(QWidget, self))
        widget = widget_info["widget"]

        # Copy action
        copy_action = menu.addAction("Copy")
        if copy_action is not None:
            copy_action.triggered.connect(lambda: self.copy_widget_text(widget))

        # Copy all action
        copy_all_action = menu.addAction("Copy All")
        if copy_all_action is not None:
            copy_all_action.triggered.connect(self.copy_all_results)

        menu.addSeparator()

        # Paste action (if applicable)
        if hasattr(widget, "setText") or hasattr(widget, "setPlainText"):
            paste_action = menu.addAction("Paste")
            if paste_action is not None:
                paste_action.triggered.connect(self.paste_text)

        menu.exec(widget.mapToGlobal(position))

    def copy_widget_text(self, widget: Any) -> None:
        """Copy text from a specific widget"""
        text = self.get_text_from_widget(widget)
        if text:
            self.copy_to_clipboard(text)

    def create_copy_button(self, text: str = "Copy Results") -> Any:
        """Create a copy button for the calculator"""
        if not (text is not None):
            raise ValueError("text must be provided")
        from PyQt6.QtWidgets import QPushButton

        copy_btn = QPushButton(text)
        copy_btn.clicked.connect(self.copy_all_results)
        copy_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """,
        )
        return copy_btn

    def create_save_load_buttons(self) -> tuple[Any, Any]:
        """Create save and load state buttons"""
        from PyQt6.QtWidgets import QPushButton

        save_btn = QPushButton("Save State")
        save_btn.clicked.connect(lambda: self.save_calculator_state())  # type: ignore[attr-defined]
        save_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """,
        )

        load_btn = QPushButton("Load State")
        load_btn.clicked.connect(lambda: self.load_calculator_state())  # type: ignore[attr-defined]
        load_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """,
        )

        return save_btn, load_btn


# ---------------------------------------------------------------------------
# Public mixin: CalculatorStateMixin  (composes the two sub-mixins)
# ---------------------------------------------------------------------------


class CalculatorStateMixin(_SplitterStateMixin, _ClipboardMixin):
    """Mixin class providing state management and copy/paste functionality for calculators.

    Note: This mixin does NOT inherit from QObject or QWidget to avoid MRO conflicts
    and segfaults caused by Qt's metaclass system when combined with QWidget.
    It's designed to be used with QWidget subclasses as:
        class MyCalculator(QWidget, CalculatorStateMixin): ...

    The mixin uses duck typing and assumes it will be mixed with a QWidget.
    QWidget methods (focusWidget, mapToGlobal, etc.) will be available at runtime.

    Responsibilities are delegated to private sub-mixins:
    - :class:`_SplitterStateMixin` — splitter registration, persistence, restore
    - :class:`_ClipboardMixin`    — copy/paste, context menus, widget text extraction
    """

    # Note: Signals removed - they require QObject inheritance which causes segfaults
    # when combined with QWidget (diamond inheritance with Qt metaclass).
    # If signals are needed, define them in the concrete class that uses this mixin.

    def __init__(self, calculator_name: str | None = None) -> None:
        """Initialize the mixin

        Args:
            calculator_name: Name of the calculator for state management

        """
        self.calculator_name = calculator_name or "UnknownCalculator"
        self.state_manager = state_manager

        # State management
        self.auto_save_enabled = True
        self.last_save_time: datetime | None = None
        self.unsaved_changes = False

        # Splitter tracking (used by _SplitterStateMixin)
        self.splitters: list[dict[str, Any]] = []
        self.splitter_states: dict[str, dict[str, Any]] = {}

        # Input widget tracking
        self.input_widgets: list[dict[str, Any]] = []

        # Copy/paste functionality (used by _ClipboardMixin)
        self.copyable_widgets: list[dict[str, Any]] = []

        # Auto-save timer
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save_state)
        self.auto_save_timer.start(30000)  # Auto-save every 30 seconds

        # Track changes
        self.change_tracking_enabled = True

        # Setup copy/paste after widget is fully initialized
        QTimer.singleShot(0, self.setup_copy_paste)

    # ------------------------------------------------------------------
    # Input widget management
    # ------------------------------------------------------------------

    def register_input_widget(self, widget: Any, name: str | None = None) -> None:
        """Register an input widget for state management"""
        if name is None:
            if hasattr(widget, "objectName") and widget.objectName():
                name = widget.objectName()
            else:
                name = f"input_{len(self.input_widgets)}"

        self.input_widgets.append({"widget": widget, "name": name})

        # Connect change signals if possible
        if hasattr(widget, "textChanged"):
            widget.textChanged.connect(self.mark_changed)
        elif hasattr(widget, "valueChanged"):
            widget.valueChanged.connect(self.mark_changed)
        elif hasattr(widget, "currentTextChanged"):
            widget.currentTextChanged.connect(self.mark_changed)
        elif hasattr(widget, "toggled"):
            widget.toggled.connect(self.mark_changed)

    def auto_register_widgets(self) -> None:
        """Automatically register input widgets and splitters found in the UI."""
        if not isinstance(self, QWidget):
            return

        # Register splitters
        for splitter in self.findChildren(QSplitter):
            # Check if already registered
            is_registered = False
            for s in self.splitters:
                if s["widget"] == splitter:
                    is_registered = True
                    break
            if not is_registered:
                self.register_splitter(splitter, splitter.objectName())

        # Register input widgets
        # We look for common input widgets
        from PyQt6.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QLineEdit,
            QSpinBox,
        )

        input_types = (QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QCheckBox)

        for widget in self.findChildren(input_types):
            # Check if already registered
            is_registered = False
            for w in self.input_widgets:
                if w["widget"] == widget:
                    is_registered = True
                    break
            if not is_registered:
                self.register_input_widget(widget, widget.objectName())

        # Register copyable widgets
        # We look for result displays like QTextEdit, QTableWidget
        from PyQt6.QtWidgets import QTableWidget, QTextEdit

        for widget in self.findChildren(QTextEdit):
            # Check if already registered
            is_registered = False
            for w in self.copyable_widgets:
                if w["widget"] == widget:
                    is_registered = True
                    break
            if not is_registered:
                self.register_copyable_widget(widget, "text")

        for widget in self.findChildren(QTableWidget):
            # Check if already registered
            is_registered = False
            for w in self.copyable_widgets:
                if w["widget"] == widget:
                    is_registered = True
                    break
            if not is_registered:
                self.register_copyable_widget(widget, "table")

    def save_input_states(self) -> dict[str, Any]:
        """Save current input widget states"""
        states = {}
        for info in self.input_widgets:
            widget = info["widget"]
            name = info["name"]
            value = None

            if hasattr(widget, "text"):
                value = widget.text()
            elif hasattr(widget, "value"):
                value = widget.value()
            elif hasattr(widget, "currentText"):
                value = widget.currentText()
            elif hasattr(widget, "isChecked"):
                value = widget.isChecked()

            if value is not None:
                states[name] = value
        return states

    def restore_input_states(self, states: dict[str, Any]) -> None:
        """Restore input widget states"""
        assert states is not None, "states must be provided"
        self.change_tracking_enabled = False
        try:
            for info in self.input_widgets:
                name = info["name"]
                if name in states:
                    widget = info["widget"]
                    value = states[name]

                    if hasattr(widget, "setText"):
                        widget.setText(str(value))
                    elif hasattr(widget, "setValue"):
                        try:
                            (
                                widget.setValue(float(value))
                                if isinstance(value, float) or "." in str(value)
                                else widget.setValue(int(value))
                            )
                        except (
                            ValueError,
                            ZeroDivisionError,
                            OverflowError,
                            TypeError,
                        ):
                            widget.setValue(value)
                    elif hasattr(widget, "setCurrentText"):
                        widget.setCurrentText(str(value))
                    elif hasattr(widget, "setChecked"):
                        widget.setChecked(bool(value))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass
        finally:
            self.change_tracking_enabled = True

    # ------------------------------------------------------------------
    # Core state persistence
    # ------------------------------------------------------------------

    def get_calculator_state(self) -> dict[str, Any]:
        """Get current calculator state for saving

        Override this method in subclasses to include calculator-specific data
        """
        state = {
            "calculator_name": self.calculator_name,
            "timestamp": datetime.now().isoformat(),
            "splitter_states": self.save_splitter_states(),
            "input_states": self.save_input_states(),
            "window_geometry": (
                self.saveGeometry().toBase64().data().decode("utf-8")
                if hasattr(self, "saveGeometry")
                else None
            ),
        }

        # Add calculator-specific state
        calculator_state = self.get_calculator_specific_state()
        if calculator_state:
            state.update(calculator_state)

        return state

    def get_calculator_specific_state(self) -> dict[str, Any]:
        """Get calculator-specific state data

        Override this method in subclasses to include specific data
        """
        return {}

    def set_calculator_state(self, state: dict[str, Any]) -> None:
        """Set calculator state from loaded data

        Override this method in subclasses to handle calculator-specific data
        """
        try:
            # Restore splitter states
            if "splitter_states" in state:
                self.restore_splitter_states(state["splitter_states"])

            # Restore input states
            if "input_states" in state:
                self.restore_input_states(state["input_states"])

            # Restore window geometry if available
            if state.get("window_geometry"):
                from PyQt6.QtCore import QByteArray

                geometry_data = QByteArray.fromBase64(
                    state["window_geometry"].encode("utf-8"),
                )
                if hasattr(self, "restoreGeometry"):
                    self.restoreGeometry(geometry_data)

            # Restore calculator-specific state
            self.set_calculator_specific_state(state)

            self.unsaved_changes = False

        except ImportError:
            pass

    def set_calculator_specific_state(self, state: dict[str, Any]) -> None:
        """Set calculator-specific state data

        Override this method in subclasses to handle specific data
        """

    def save_calculator_state(self, state_name: str | None = None) -> bool:
        """Save current calculator state

        Args:
            state_name: Optional name for the state, defaults to calculator name

        Returns:
            True if save was successful

        """
        try:
            if state_name is None:
                state_name = f"{self.calculator_name}_state"

            state_data = self.get_calculator_state()

            success = self.state_manager.save_state(
                state_name=state_name,
                state_data=state_data,
                description=f"Auto-saved state for {self.calculator_name}",
                protected=False,
            )

            if success:
                self.last_save_time = datetime.now()
                self.unsaved_changes = False
                logger.info("Calculator state saved: %s", state_name)

            return bool(success)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return False

    def load_calculator_state(
        self, state_name: str | None = None
    ) -> dict[str, Any] | None:
        """Load calculator state

        Args:
            state_name: Optional name for the state, defaults to calculator name

        Returns:
            State data if successful, None otherwise

        """
        try:
            if state_name is None:
                state_name = f"{self.calculator_name}_state"

            state_data = self.state_manager.load_state(state_name)

            if state_data:
                self.set_calculator_state(state_data)
                logger.info("Calculator state loaded: %s", state_name)

            return state_data

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return None

    def auto_save_state(self) -> None:
        """Auto-save state if there are unsaved changes"""
        if self.auto_save_enabled and self.unsaved_changes:
            self.save_calculator_state()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def handle_close_event(self, event: Any) -> None:
        """Handle close event - save state before closing"""
        try:
            # Auto-save state if there are unsaved changes
            if self.unsaved_changes:
                self.save_calculator_state()

            # Stop auto-save timer
            if hasattr(self, "auto_save_timer"):
                self.auto_save_timer.stop()

            event.accept()

        except (RuntimeError, OSError):
            event.accept()

    def mark_changed(self) -> None:
        """Mark that the calculator has unsaved changes"""
        if self.change_tracking_enabled:
            self.unsaved_changes = True

    # ------------------------------------------------------------------
    # Aliases for backward compatibility
    # ------------------------------------------------------------------

    def save_state(self, state_name: str | None = None) -> bool:
        """Alias for save_calculator_state"""
        return self.save_calculator_state(state_name)

    def load_state(self, state_name: str | None = None) -> dict[str, Any] | None:
        """Alias for load_calculator_state"""
        return self.load_calculator_state(state_name)
