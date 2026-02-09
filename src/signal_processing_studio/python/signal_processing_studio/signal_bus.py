"""Signal bus for routing data between Signal Processing Studio widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from signal_toolkit.core import Signal, SignalGenerator

if TYPE_CHECKING:
    from collections.abc import Callable


class SignalBus(QObject):
    """Routes signals between Function Generator, Signal Toolkit, and Polynomial Generator."""

    signal_routed = pyqtSignal(str)  # status message

    def __init__(
        self,
        func_gen: QObject,
        toolkit: QObject,
        poly_gen: QObject,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.func_gen = func_gen
        self.toolkit = toolkit
        self.poly_gen = poly_gen
        self._status_callback = status_callback

        # FunctionGenerator -> SignalToolkit
        self.func_gen.signal_generated.connect(self._on_func_gen_signal)

        # PolynomialGenerator -> SignalToolkit
        self.poly_gen.polynomial_generated.connect(self._on_poly_generated)

    def _on_func_gen_signal(self, signal: Signal) -> None:
        """Route generated signal from Function Generator to Signal Toolkit."""
        self.toolkit.load_external_signal(signal)
        self._status(
            f"Signal sent to Toolkit: {signal.name or 'waveform'} "
            f"({signal.n_samples} samples, {signal.fs:.0f} Hz)"
        )

    def _on_poly_generated(self, joint_name: str, coeffs: list) -> None:
        """Convert polynomial coefficients to Signal and route to Toolkit."""
        # Use toolkit's current time range, or default
        if self.toolkit.current_signal is not None:
            t = self.toolkit.current_signal.time
        else:
            t = np.linspace(0, 10, 1000)

        # np.polyfit returns highest-degree-first; SignalGenerator expects lowest-first
        signal = SignalGenerator.polynomial(t, list(reversed(coeffs)))
        signal.name = f"Polynomial ({joint_name})"
        self.toolkit.load_external_signal(signal)
        self._status(f"Polynomial from {joint_name} sent to Toolkit")

    def send_current_to_toolkit(self) -> None:
        """Manually send the current Function Generator signal to Toolkit."""
        if self.func_gen.current_signal is not None:
            self._on_func_gen_signal(self.func_gen.current_signal)

    def _status(self, message: str) -> None:
        """Emit status message."""
        self.signal_routed.emit(message)
        if self._status_callback:
            self._status_callback(message)
