"""Adaptive filter implementations for signal_toolkit."""

from __future__ import annotations

import numpy as np

from .core import Signal


class AdaptiveFilter:
    """Adaptive filter implementations (LMS, RLS)."""

    @staticmethod
    def lms(
        signal: Signal,
        reference: Signal,
        order: int = 10,
        step_size: float = 0.01,
    ) -> tuple[Signal, Signal]:
        """Apply Least Mean Squares (LMS) adaptive filter."""
        if not (signal is not None):
            raise ValueError("signal must be provided")
        n = len(signal.values)
        x = signal.values
        d = reference.values

        w = np.zeros(order)  # Filter weights
        y = np.zeros(n)  # Filter output
        e = np.zeros(n)  # Error

        for i in range(order, n):
            x_window = x[i - order : i][::-1]  # Reversed window
            y[i] = np.dot(w, x_window)
            e[i] = d[i] - y[i]
            w += step_size * e[i] * x_window

        filtered = Signal(
            time=signal.time,
            values=y,
            name=f"{signal.name}_lms",
            units=signal.units,
        )

        error = Signal(
            time=signal.time,
            values=e,
            name=f"{signal.name}_lms_error",
            units=signal.units,
        )

        return filtered, error

    @staticmethod
    def rls(
        signal: Signal,
        reference: Signal,
        order: int = 10,
        forgetting_factor: float = 0.99,
        delta: float = 0.01,
    ) -> tuple[Signal, Signal]:
        """Apply Recursive Least Squares (RLS) adaptive filter."""
        if not (signal is not None):
            raise ValueError("signal must be provided")
        n = len(signal.values)
        x = signal.values
        d = reference.values

        w = np.zeros(order)  # Filter weights
        P = np.eye(order) / delta  # Inverse correlation matrix
        y = np.zeros(n)  # Filter output
        e = np.zeros(n)  # Error

        lam = forgetting_factor

        for i in range(order, n):
            x_window = x[i - order : i][::-1].reshape(-1, 1)
            y[i] = np.dot(w, x_window.flatten())
            e[i] = d[i] - y[i]

            # RLS update
            k = P @ x_window / (lam + x_window.T @ P @ x_window)
            P = (P - k @ x_window.T @ P) / lam
            w += k.flatten() * e[i]

        filtered = Signal(
            time=signal.time,
            values=y,
            name=f"{signal.name}_rls",
            units=signal.units,
        )

        error = Signal(
            time=signal.time,
            values=e,
            name=f"{signal.name}_rls_error",
            units=signal.units,
        )

        return filtered, error
