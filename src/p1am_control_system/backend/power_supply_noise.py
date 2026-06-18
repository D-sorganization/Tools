"""Rolling feedback-noise tracking for the power-supply controller."""

from __future__ import annotations

from collections import deque

from power_supply_models import PowerSupplyConfig
from signal_stats import NoiseStats, compute_noise

__all__ = ["PowerSupplyNoiseTracker"]


class PowerSupplyNoiseTracker:
    """Bounded current/voltage feedback windows plus configured arc metrics."""

    def __init__(self, config: PowerSupplyConfig) -> None:
        self._current_samples: deque[float] = deque(maxlen=config.noise_window)
        self._voltage_samples: deque[float] = deque(maxlen=config.noise_window)

    def reconfigure(self, config: PowerSupplyConfig) -> None:
        """Resize windows while retaining the most recent feedback samples."""
        if config.noise_window == self._current_samples.maxlen:
            return
        self._current_samples = deque(self._current_samples, maxlen=config.noise_window)
        self._voltage_samples = deque(self._voltage_samples, maxlen=config.noise_window)

    def append(self, current_a: float, voltage_v: float) -> None:
        """Record one sanitized feedback sample."""
        self._current_samples.append(current_a)
        self._voltage_samples.append(voltage_v)

    def current_noise(self, config: PowerSupplyConfig) -> NoiseStats:
        return compute_noise(
            list(self._current_samples),
            metric=config.noise_metric,
            threshold=config.current_arc_threshold,
        )

    def voltage_noise(self, config: PowerSupplyConfig) -> NoiseStats:
        return compute_noise(
            list(self._voltage_samples),
            metric=config.noise_metric,
            threshold=config.voltage_arc_threshold,
        )
