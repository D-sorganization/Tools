"""Global performance mode for the PLC scan loop.

Lets the operator trade responsiveness for CPU: ``performance`` polls the PLC
(and broadcasts to the HMI) at the fast configured interval; ``lightweight``
slows it down, which mainly cuts how often the browser re-renders the live
trends — useful when the Pi is busy (browsing + coding alongside the HMI).

Pure + dependency-light so it is unit-testable; the scan loop just reads
``poll_interval_s`` each cycle.
"""

from __future__ import annotations

from performance_models import PerformanceConfig, PerformanceMode

__all__ = ["PerformanceConfig", "PerformanceController", "PerformanceMode"]


class PerformanceController:
    """Holds the active mode and resolves it to a poll interval.

    DbC: both intervals must be finite and > 0; ``set_mode`` rejects non-enum
    input. The scan loop reads :attr:`poll_interval_s` each iteration, so a mode
    change takes effect on the next scan.
    """

    def __init__(
        self,
        performance_interval_s: float,
        lightweight_interval_s: float,
        mode: PerformanceMode = PerformanceMode.PERFORMANCE,
    ) -> None:
        self._performance_s = self._validate(performance_interval_s, "performance")
        self._lightweight_s = self._validate(lightweight_interval_s, "lightweight")
        if not isinstance(mode, PerformanceMode):
            raise TypeError(
                f"mode must be a PerformanceMode, got {type(mode).__name__}"
            )
        self._mode = mode

    @staticmethod
    def _validate(value: float, name: str) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(
                f"{name}_interval_s must be numeric, got {type(value).__name__}"
            )
        v = float(value)
        if not (v > 0.0) or v != v or v == float("inf"):
            raise ValueError(
                f"{name}_interval_s must be a finite value > 0, got {value}"
            )
        return v

    @property
    def mode(self) -> PerformanceMode:
        return self._mode

    @property
    def poll_interval_s(self) -> float:
        """Poll interval for the active mode."""
        if self._mode == PerformanceMode.LIGHTWEIGHT:
            return self._lightweight_s
        return self._performance_s

    def set_mode(self, mode: PerformanceMode) -> None:
        """Switch the active mode.

        Raises:
            TypeError: if mode is not a PerformanceMode.
        """
        if not isinstance(mode, PerformanceMode):
            raise TypeError(
                f"mode must be a PerformanceMode, got {type(mode).__name__}"
            )
        self._mode = mode

    def config(self) -> PerformanceConfig:
        return PerformanceConfig(mode=self._mode, poll_interval_s=self.poll_interval_s)
