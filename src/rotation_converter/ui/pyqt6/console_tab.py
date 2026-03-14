"""Console UI component for interactive python execution."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rotation_converter.scripting_env import ConsoleEnvironment


class CommandConsoleTab(QWidget):
    """Interactive Python REPL console GUI with a user script editor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._env = ConsoleEnvironment()
        self._history: list[str] = []
        self._history_idx = 0
        self._build_ui()
        self._connect_signals()

        # Initial greeting
        self._print_output("Python Interactive Console (Rotation Converter)")
        self._print_output(
            "Loaded functions: mr.*, Rotation, RigidTransform, np.*, math.*, pd.*, scipy.*"
        )
        self._print_output("Type 'help(func)' or 'help(mr)' for documentation.\n")

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: interactive console
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self._output = QTextBrowser()
        font = QFont("Courier")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._output.setFont(font)
        # Disable HTML parsing to prevent issues with <, > etc in console outputs
        self._output.setOpenExternalLinks(False)
        self._output.setOpenLinks(False)
        left_layout.addWidget(self._output)

        input_layout = QHBoxLayout()
        prompt_label = QLabel(">>>")
        prompt_label.setFont(font)
        input_layout.addWidget(prompt_label)

        self._input = QLineEdit()
        self._input.setFont(font)
        self._input.setPlaceholderText("Enter Python code here...")
        input_layout.addWidget(self._input)

        left_layout.addLayout(input_layout)
        splitter.addWidget(left_widget)

        # Right side: User functions editor
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("User Functions Library"))

        self._editor = QTextEdit()
        self._editor.setFont(font)
        self._editor.setPlainText(self._env.get_user_code())
        right_layout.addWidget(self._editor)

        btn_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save && Reload")
        btn_layout.addStretch()
        btn_layout.addWidget(self._save_btn)
        right_layout.addLayout(btn_layout)

        splitter.addWidget(right_widget)

        # Adjust proportions
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        self._input.returnPressed.connect(self._execute_command)
        self._save_btn.clicked.connect(self._save_user_code)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        """Handle up/down arrow for history navigation if in input field."""
        if event is None:
            super().keyPressEvent(event)
            return

        if self._input.hasFocus():
            if event.key() == Qt.Key.Key_Up:
                if self._history and self._history_idx > 0:
                    self._history_idx -= 1
                    self._input.setText(self._history[self._history_idx])
                return
            elif event.key() == Qt.Key.Key_Down:
                if self._history and self._history_idx < len(self._history) - 1:
                    self._history_idx += 1
                    self._input.setText(self._history[self._history_idx])
                else:
                    self._history_idx = len(self._history)
                    self._input.clear()
                return
        super().keyPressEvent(event)

    def _execute_command(self) -> None:
        cmd = self._input.text()
        if not cmd.strip():
            return

        # Add to history
        self._history.append(cmd)
        self._history_idx = len(self._history)

        self._print_output(f">>> {cmd}", is_input=True)
        self._input.clear()

        out, err = self._env.execute(cmd)
        if out:
            self._print_output(out, is_error=False)
        if err:
            self._print_output(err, is_error=True)

    def _print_output(
        self, text: str, is_error: bool = False, is_input: bool = False
    ) -> None:
        """Prints text to the console, preserving whitespace and colour."""
        assert text is not None, "text must be provided"
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        # To handle theme changes properly without hardcoding, we can check basic Qt palette
        # But since we use fixed dark/light colours elsewhere, we match the fleet dark theme.
        if is_error:
            fmt.setForeground(QColor("#f38ba8"))
        elif is_input:
            fmt.setForeground(QColor("#a6e3a1"))
        else:
            fmt.setForeground(QColor("#cdd6f4"))  # Default output text

        text = text.rstrip("\n") + "\n"
        cursor.insertText(text, fmt)
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    def _save_user_code(self) -> None:
        code = self._editor.toPlainText()
        self._env.save_user_code(code)
        self._env.refresh_user_functions()
        self._print_output("User library saved and reloaded.", is_input=True)
