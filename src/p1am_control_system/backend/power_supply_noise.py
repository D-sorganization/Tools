"""Rolling feedback-noise tracking for the power-supply controller."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from power_supply_models import PowerSupplyConfig
from signal_stats import NoiseStats, compute_noise


@dataclass(frozen=True)
class PowerSupplyNoiseSnapshot:
    """Noise stats for both feedback channels plus the aggregate arc flag."""

    current: NoiseStats
    voltage: NoiseStats
    arcing: bool


class FeedbackNoiseTracker:
    """Bounded current/voltage sample windows for arc-noise detection."""

    def __init__(self, config: PowerSupplyConfig) -> None:
        self._current_samples: deque[float] = deque(maxlen=config.noise_window)
        self._voltage_samples: deque[float] = deque(maxlen=config.noise_window)

    def update_config(self, config: PowerSupplyConfig) -> None:
        """Resize windows when the operator changes the configured length."""
        if config.noise_window == self._current_samples.maxlen:
            return
        self._current_samples = deque(
            self._current_samples,
            maxlen=config.noise_window,
        )
        self._voltage_samples = deque(
            self._voltage_samples,
            maxlen=config.noise_window,
        )

    def append(self, current_a: float, voltage_v: float) -> None:
        """Record sanitized feedback samples for later noise analysis."""
        self._current_samples.append(current_a)
        self._voltage_samples.append(voltage_v)

    def snapshot(self, config: PowerSupplyConfig) -> PowerSupplyNoiseSnapshot:
        """Compute channel noise stats using the latest operator thresholds."""
        current_noise = compute_noise(
            list(self._current_samples),
            metric=config.noise_metric,
            threshold=config.current_arc_threshold,
        )
        voltage_noise = compute_noise(
            list(self._voltage_samples),
            metric=config.noise_metric,
            threshold=config.voltage_arc_threshold,
        )
        return PowerSupplyNoiseSnapshot(
            current=current_noise,
            voltage=voltage_noise,
            arcing=current_noise.arcing or voltage_noise.arcing,
        )
