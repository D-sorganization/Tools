"""Function Generator PyQt6 Main Window.

A comprehensive GUI for generating and visualizing various waveforms
using the SignalGenerator engine from signal_toolkit.
"""

from __future__ import annotations

import os

import numpy as np

# Handle matplotlib backend
if os.environ.get("HEADLESS", "false").lower() == "true":
    import matplotlib

    matplotlib.use("Agg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from signal_toolkit import Signal, SignalGenerator


class FunctionGeneratorWidget(QWidget):
    """Main widget for the Function Generator application."""

    signal_generated = pyqtSignal(object)  # Emits Signal object

    WAVEFORM_TYPES = [
        "Sinusoid",
        "Cosine",
        "Square Wave",
        "Triangle Wave",
        "Sawtooth",
        "Pulse",
        "Step",
        "Exponential",
        "Linear",
        "Polynomial",
        "Chirp",
        "Constant",
    ]

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        use_builtin_theme: bool = True,
    ) -> None:
        """Initialize the Function Generator widget."""
        super().__init__(parent)
        self.current_signal: Signal | None = None
        self._init_ui()
        if use_builtin_theme:
            self._apply_styling()
        self._connect_signals()
        self._generate_signal()

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Waveform selection
        waveform_group = QGroupBox("Waveform Type")
        waveform_layout = QVBoxLayout(waveform_group)
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(self.WAVEFORM_TYPES)
        waveform_layout.addWidget(self.waveform_combo)
        controls_layout.addWidget(waveform_group)

        # Time parameters
        time_group = QGroupBox("Time Parameters")
        time_layout = QFormLayout(time_group)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.001, 1000)
        self.duration_spin.setValue(1.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setDecimals(3)
        time_layout.addRow("Duration:", self.duration_spin)

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(10, 100000)
        self.sample_rate_spin.setValue(1000)
        self.sample_rate_spin.setSuffix(" Hz")
        time_layout.addRow("Sample Rate:", self.sample_rate_spin)

        controls_layout.addWidget(time_group)

        # Waveform parameters (dynamic based on selection)
        self.params_group = QGroupBox("Waveform Parameters")
        self.params_layout = QFormLayout(self.params_group)
        self._create_parameter_widgets()
        controls_layout.addWidget(self.params_group)

        # Generate button
        self.generate_btn = QPushButton("Generate Signal")
        self.generate_btn.setMinimumHeight(40)
        controls_layout.addWidget(self.generate_btn)

        # Signal info
        self.info_group = QGroupBox("Signal Information")
        info_layout = QVBoxLayout(self.info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        controls_layout.addWidget(self.info_group)

        controls_layout.addStretch()
        splitter.addWidget(controls_widget)

        # Right panel - Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Create tabs for different views
        self.tabs = QTabWidget()

        # Time domain tab
        time_tab = QWidget()
        time_layout_inner = QVBoxLayout(time_tab)
        self.time_figure = Figure(figsize=(8, 5), facecolor="#1e1e2e")
        self.time_canvas = FigureCanvas(self.time_figure)
        self.time_toolbar = NavigationToolbar(self.time_canvas, time_tab)
        time_layout_inner.addWidget(self.time_toolbar)
        time_layout_inner.addWidget(self.time_canvas)
        self.tabs.addTab(time_tab, "Time Domain")

        # Frequency domain tab
        freq_tab = QWidget()
        freq_layout_inner = QVBoxLayout(freq_tab)
        self.freq_figure = Figure(figsize=(8, 5), facecolor="#1e1e2e")
        self.freq_canvas = FigureCanvas(self.freq_figure)
        self.freq_toolbar = NavigationToolbar(self.freq_canvas, freq_tab)
        freq_layout_inner.addWidget(self.freq_toolbar)
        freq_layout_inner.addWidget(self.freq_canvas)
        self.tabs.addTab(freq_tab, "Frequency Domain")

        viz_layout.addWidget(self.tabs)
        splitter.addWidget(viz_widget)

        # Set splitter proportions
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)

    def _create_parameter_widgets(self) -> None:
        """Create parameter input widgets."""
        # Common parameters
        self.amplitude_spin = QDoubleSpinBox()
        self.amplitude_spin.setRange(-1000, 1000)
        self.amplitude_spin.setValue(1.0)
        self.amplitude_spin.setDecimals(3)

        self.frequency_spin = QDoubleSpinBox()
        self.frequency_spin.setRange(0.001, 10000)
        self.frequency_spin.setValue(1.0)
        self.frequency_spin.setSuffix(" Hz")
        self.frequency_spin.setDecimals(3)

        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(-360, 360)
        self.phase_spin.setValue(0.0)
        self.phase_spin.setSuffix(" deg")
        self.phase_spin.setDecimals(1)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1000, 1000)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setDecimals(3)

        self.duty_cycle_spin = QDoubleSpinBox()
        self.duty_cycle_spin.setRange(0.01, 0.99)
        self.duty_cycle_spin.setValue(0.5)
        self.duty_cycle_spin.setDecimals(2)

        self.decay_rate_spin = QDoubleSpinBox()
        self.decay_rate_spin.setRange(-100, 100)
        self.decay_rate_spin.setValue(1.0)
        self.decay_rate_spin.setDecimals(3)

        self.slope_spin = QDoubleSpinBox()
        self.slope_spin.setRange(-1000, 1000)
        self.slope_spin.setValue(1.0)
        self.slope_spin.setDecimals(3)

        self.intercept_spin = QDoubleSpinBox()
        self.intercept_spin.setRange(-1000, 1000)
        self.intercept_spin.setValue(0.0)
        self.intercept_spin.setDecimals(3)

        self.step_time_spin = QDoubleSpinBox()
        self.step_time_spin.setRange(0, 1000)
        self.step_time_spin.setValue(0.5)
        self.step_time_spin.setSuffix(" s")
        self.step_time_spin.setDecimals(3)

        self.pulse_start_spin = QDoubleSpinBox()
        self.pulse_start_spin.setRange(0, 1000)
        self.pulse_start_spin.setValue(0.2)
        self.pulse_start_spin.setSuffix(" s")
        self.pulse_start_spin.setDecimals(3)

        self.pulse_duration_spin = QDoubleSpinBox()
        self.pulse_duration_spin.setRange(0.001, 1000)
        self.pulse_duration_spin.setValue(0.3)
        self.pulse_duration_spin.setSuffix(" s")
        self.pulse_duration_spin.setDecimals(3)

        self.chirp_f0_spin = QDoubleSpinBox()
        self.chirp_f0_spin.setRange(0.001, 10000)
        self.chirp_f0_spin.setValue(1.0)
        self.chirp_f0_spin.setSuffix(" Hz")
        self.chirp_f0_spin.setDecimals(3)

        self.chirp_f1_spin = QDoubleSpinBox()
        self.chirp_f1_spin.setRange(0.001, 10000)
        self.chirp_f1_spin.setValue(10.0)
        self.chirp_f1_spin.setSuffix(" Hz")
        self.chirp_f1_spin.setDecimals(3)

        self.chirp_method_combo = QComboBox()
        self.chirp_method_combo.addItems(["linear", "exponential"])

        self.constant_value_spin = QDoubleSpinBox()
        self.constant_value_spin.setRange(-1000, 1000)
        self.constant_value_spin.setValue(1.0)
        self.constant_value_spin.setDecimals(3)

        self.poly_order_spin = QSpinBox()
        self.poly_order_spin.setRange(0, 10)
        self.poly_order_spin.setValue(2)

        self.poly_coeffs_edit = QTextEdit()
        self.poly_coeffs_edit.setMaximumHeight(60)
        self.poly_coeffs_edit.setPlaceholderText("e.g., 1, 2, 0.5 for 1 + 2t + 0.5t²")
        self.poly_coeffs_edit.setText("0, 1, -0.5")

        # Update parameters display
        self._update_parameter_widgets()

    def _update_parameter_widgets(self) -> None:
        """Update parameter widgets based on selected waveform."""
        # Clear existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        waveform = self.waveform_combo.currentText()

        if waveform in ["Sinusoid", "Cosine"]:
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Frequency:", self.frequency_spin)
            self.params_layout.addRow("Phase:", self.phase_spin)
            self.params_layout.addRow("DC Offset:", self.offset_spin)

        elif waveform == "Square Wave":
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Frequency:", self.frequency_spin)
            self.params_layout.addRow("Duty Cycle:", self.duty_cycle_spin)
            self.params_layout.addRow("DC Offset:", self.offset_spin)

        elif waveform in ["Triangle Wave", "Sawtooth"]:
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Frequency:", self.frequency_spin)
            self.params_layout.addRow("DC Offset:", self.offset_spin)

        elif waveform == "Pulse":
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Start Time:", self.pulse_start_spin)
            self.params_layout.addRow("Duration:", self.pulse_duration_spin)
            self.params_layout.addRow("Baseline:", self.offset_spin)

        elif waveform == "Step":
            self.params_layout.addRow("Step Value:", self.amplitude_spin)
            self.params_layout.addRow("Step Time:", self.step_time_spin)
            self.params_layout.addRow("Initial Value:", self.offset_spin)

        elif waveform == "Exponential":
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Decay Rate:", self.decay_rate_spin)
            self.params_layout.addRow("DC Offset:", self.offset_spin)

        elif waveform == "Linear":
            self.params_layout.addRow("Slope:", self.slope_spin)
            self.params_layout.addRow("Intercept:", self.intercept_spin)

        elif waveform == "Polynomial":
            self.params_layout.addRow("Coefficients:", self.poly_coeffs_edit)

        elif waveform == "Chirp":
            self.params_layout.addRow("Amplitude:", self.amplitude_spin)
            self.params_layout.addRow("Start Freq:", self.chirp_f0_spin)
            self.params_layout.addRow("End Freq:", self.chirp_f1_spin)
            self.params_layout.addRow("Method:", self.chirp_method_combo)

        elif waveform == "Constant":
            self.params_layout.addRow("Value:", self.constant_value_spin)

    def _connect_signals(self) -> None:
        """Connect widget signals to slots."""
        self.waveform_combo.currentTextChanged.connect(self._update_parameter_widgets)
        self.waveform_combo.currentTextChanged.connect(self._generate_signal)
        self.generate_btn.clicked.connect(self._generate_signal)

        # Auto-generate on parameter changes
        self.duration_spin.valueChanged.connect(self._generate_signal)
        self.sample_rate_spin.valueChanged.connect(self._generate_signal)
        self.amplitude_spin.valueChanged.connect(self._generate_signal)
        self.frequency_spin.valueChanged.connect(self._generate_signal)
        self.phase_spin.valueChanged.connect(self._generate_signal)
        self.offset_spin.valueChanged.connect(self._generate_signal)
        self.duty_cycle_spin.valueChanged.connect(self._generate_signal)
        self.decay_rate_spin.valueChanged.connect(self._generate_signal)
        self.slope_spin.valueChanged.connect(self._generate_signal)
        self.intercept_spin.valueChanged.connect(self._generate_signal)
        self.step_time_spin.valueChanged.connect(self._generate_signal)
        self.pulse_start_spin.valueChanged.connect(self._generate_signal)
        self.pulse_duration_spin.valueChanged.connect(self._generate_signal)
        self.chirp_f0_spin.valueChanged.connect(self._generate_signal)
        self.chirp_f1_spin.valueChanged.connect(self._generate_signal)
        self.chirp_method_combo.currentTextChanged.connect(self._generate_signal)
        self.constant_value_spin.valueChanged.connect(self._generate_signal)

    def _build_waveform(self, waveform: str, t: np.ndarray) -> np.ndarray | None:
        """Build the signal array for the given waveform type.

        Args:
            waveform: Waveform type name from the combo box
            t: Time array

        Returns:
            Generated signal array, or None if waveform is unknown
        """
        if not (waveform is not None):
            raise ValueError("waveform must be provided")
        amp = self.amplitude_spin.value()
        freq = self.frequency_spin.value()
        phase = np.radians(self.phase_spin.value())
        offset = self.offset_spin.value()

        if waveform == "Sinusoid":
            return SignalGenerator.sinusoid(
                t,
                amplitude=amp,
                frequency=freq,
                phase=phase,
                offset=offset,
            )
        elif waveform == "Cosine":
            return SignalGenerator.cosine(
                t,
                amplitude=amp,
                frequency=freq,
                phase=phase,
                offset=offset,
            )
        elif waveform == "Square Wave":
            return SignalGenerator.square(
                t,
                frequency=freq,
                amplitude=amp,
                duty_cycle=self.duty_cycle_spin.value(),
                offset=offset,
            )
        elif waveform == "Triangle Wave":
            return SignalGenerator.triangle(
                t,
                frequency=freq,
                amplitude=amp,
                offset=offset,
            )
        elif waveform == "Sawtooth":
            return SignalGenerator.sawtooth(
                t,
                frequency=freq,
                amplitude=amp,
                offset=offset,
            )
        elif waveform == "Pulse":
            return SignalGenerator.pulse(
                t,
                start_time=self.pulse_start_spin.value(),
                duration=self.pulse_duration_spin.value(),
                amplitude=amp,
                baseline=offset,
            )
        elif waveform == "Step":
            return SignalGenerator.step(
                t,
                step_time=self.step_time_spin.value(),
                step_value=amp,
                initial_value=offset,
            )
        elif waveform == "Exponential":
            return SignalGenerator.exponential(
                t,
                amplitude=amp,
                decay_rate=self.decay_rate_spin.value(),
                offset=offset,
            )
        elif waveform == "Linear":
            return SignalGenerator.linear(
                t,
                slope=self.slope_spin.value(),
                intercept=self.intercept_spin.value(),
            )
        elif waveform == "Polynomial":
            coeffs_text = self.poly_coeffs_edit.toPlainText()
            coeffs = [float(c.strip()) for c in coeffs_text.split(",") if c.strip()]
            if not coeffs:
                coeffs = [0, 1]
            return SignalGenerator.polynomial(t, coeffs)
        elif waveform == "Chirp":
            return SignalGenerator.chirp(
                t,
                f0=self.chirp_f0_spin.value(),
                f1=self.chirp_f1_spin.value(),
                amplitude=amp,
                method=self.chirp_method_combo.currentText(),
            )
        elif waveform == "Constant":
            return SignalGenerator.constant(t, value=self.constant_value_spin.value())
        return None

    def _generate_signal(self) -> None:
        """Generate the signal based on current parameters."""
        duration = self.duration_spin.value()
        sample_rate = self.sample_rate_spin.value()
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)

        try:
            signal = self._build_waveform(self.waveform_combo.currentText(), t)
            if signal is None:
                return

            self.current_signal = signal
            self._update_plots()
            self._update_info()
            self.signal_generated.emit(signal)

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            self.info_text.setText(f"Error generating signal:\n{e}")

    def _update_plots(self) -> None:
        """Update the visualization plots."""
        if self.current_signal is None:
            return

        signal = self.current_signal

        # Time domain plot
        self.time_figure.clear()
        ax = self.time_figure.add_subplot(111)
        ax.set_facecolor("#313244")
        ax.plot(signal.time, signal.values, color="#89b4fa", linewidth=1.5)
        ax.set_xlabel("Time (s)", color="#cdd6f4")
        ax.set_ylabel("Amplitude", color="#cdd6f4")
        ax.set_title(f"{self.waveform_combo.currentText()}", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.grid(True, alpha=0.3, color="#585b70")
        for spine in ax.spines.values():
            spine.set_color("#585b70")
        self.time_figure.tight_layout()
        self.time_canvas.draw()

        # Frequency domain plot
        self.freq_figure.clear()
        ax2 = self.freq_figure.add_subplot(111)
        ax2.set_facecolor("#313244")

        # Compute FFT
        n = len(signal.values)
        fft_vals = np.fft.fft(signal.values)
        fft_freq = np.fft.fftfreq(n, signal.dt)

        # Only positive frequencies
        pos_mask = fft_freq >= 0
        fft_freq = fft_freq[pos_mask]
        fft_magnitude = np.abs(fft_vals[pos_mask]) * 2 / n

        ax2.plot(fft_freq, fft_magnitude, color="#a6e3a1", linewidth=1.5)
        ax2.set_xlabel("Frequency (Hz)", color="#cdd6f4")
        ax2.set_ylabel("Magnitude", color="#cdd6f4")
        ax2.set_title("Frequency Spectrum", color="#cdd6f4")
        ax2.tick_params(colors="#cdd6f4")
        ax2.grid(True, alpha=0.3, color="#585b70")
        for spine in ax2.spines.values():
            spine.set_color("#585b70")

        # Limit x-axis to meaningful frequencies
        max_freq = min(
            signal.fs / 2,
            (
                self.frequency_spin.value() * 10
                if self.frequency_spin.value() > 0
                else signal.fs / 2
            ),
        )
        ax2.set_xlim(0, max_freq)

        self.freq_figure.tight_layout()
        self.freq_canvas.draw()

    def _update_info(self) -> None:
        """Update the signal information display."""
        if self.current_signal is None:
            return

        signal = self.current_signal
        info = f"""Signal: {self.waveform_combo.currentText()}
Duration: {signal.duration:.4f} s
Samples: {signal.n_samples}
Sample Rate: {signal.fs:.1f} Hz
Min Value: {np.min(signal.values):.4f}
Max Value: {np.max(signal.values):.4f}
Mean: {np.mean(signal.values):.4f}
RMS: {np.sqrt(np.mean(signal.values**2)):.4f}"""
        self.info_text.setText(info)

    def _apply_styling(self) -> None:
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #313244;
            }
            QGroupBox::title {
                color: #cba6f7;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
            QPushButton:pressed {
                background-color: #7287fd;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #45475a;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px 8px;
                color: #cdd6f4;
            }
            QTextEdit {
                background-color: #45475a;
                border: 1px solid #585b70;
                border-radius: 4px;
                color: #cdd6f4;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
                background-color: #313244;
            }
            QTabBar::tab {
                background-color: #45475a;
                color: #cdd6f4;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
            }
            QSplitter::handle {
                background-color: #585b70;
            }
        """)
