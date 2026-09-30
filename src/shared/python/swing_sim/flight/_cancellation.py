"""Internal cooperative-cancellation boundary for flight execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .capability_observation import CancellationCheck

if TYPE_CHECKING:
    from .types import FlightResult


class FlightSimulationCancelled(RuntimeError):
    """Signal cooperative cancellation before a flight result is published."""

    def __init__(
        self,
        message: str = "flight simulation cancelled",
        *,
        result: FlightResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


class FlightCancellationCallbackError(RuntimeError):
    """Signal a raising or contract-invalid flight cancellation callback."""


def raise_if_flight_cancelled(
    cancellation_requested: CancellationCheck | None,
) -> None:
    """Poll one exact-bool callback or raise a typed control exception."""
    if cancellation_requested is None:
        return
    try:
        requested = cancellation_requested()
        if type(requested) is not bool:
            raise TypeError("cancellation_requested must return an exact bool")
    except Exception as error:
        raise FlightCancellationCallbackError(
            "flight cancellation callback failed"
        ) from error
    if requested:
        raise FlightSimulationCancelled("flight simulation cancelled")


__all__ = [
    "FlightCancellationCallbackError",
    "FlightSimulationCancelled",
    "raise_if_flight_cancelled",
]
