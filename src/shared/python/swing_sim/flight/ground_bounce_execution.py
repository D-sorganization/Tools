"""Authoritative composition from qualified flight output to repeated bounce."""

from __future__ import annotations

from collections.abc import Callable

from shared.python.swing_sim.ground.bounce_execution import (
    execute_repeated_bounce_request,
)
from shared.python.swing_sim.ground.bounce_request_wire import (
    RepeatedBounceRequest,
    RepeatedBounceRequestResultPair,
)
from shared.python.swing_sim.ground.bounce_types import BounceModelSettings

from .ground_transfer import (
    FlightGroundTransferSettings,
    build_ground_simulation_request,
)
from .types import FlightResult, LaunchConditions


def execute_repeated_bounce_from_flight(
    flight: FlightResult,
    launch: LaunchConditions,
    transfer: FlightGroundTransferSettings,
    capture_speed_m_s: float = 0.05,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> RepeatedBounceRequestResultPair:
    """Transfer one exact flight result and execute its repeated-bounce prefix.

    Input contracts are checked before transfer work begins. Physical transfer,
    request identity, and bounce execution remain delegated to their existing
    authoritative components so this facade introduces no duplicate physics.
    """
    if type(flight) is not FlightResult:
        raise ValueError("flight must be an exact FlightResult")
    if type(launch) is not LaunchConditions:
        raise ValueError("launch must be an exact LaunchConditions")
    if type(transfer) is not FlightGroundTransferSettings:
        raise ValueError("transfer must be an exact FlightGroundTransferSettings")
    if is_cancelled is not None and not callable(is_cancelled):
        raise ValueError("is_cancelled must be callable or None")

    validated_capture_speed = BounceModelSettings(
        capture_speed_m_s=capture_speed_m_s
    ).capture_speed_m_s
    ground_request = build_ground_simulation_request(flight, launch, transfer)
    request = RepeatedBounceRequest(ground_request, validated_capture_speed)
    return execute_repeated_bounce_request(request, is_cancelled=is_cancelled)


__all__ = ["execute_repeated_bounce_from_flight"]
