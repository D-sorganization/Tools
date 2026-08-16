"""Derive the HMI's PLC connection label from a telemetry frame.

The desktop HMI used to hardcode ``"Simulating"`` the moment the WebSocket
opened, which meant a desktop driving a *live* plant displayed an amber
"PLC Connection: Simulating" forever — an engineer reads that as "this is a
bench rig" and starts clearing E-stops on energised equipment (issue #4019).

Mislabelling a live plant as simulated is the dangerous direction, so the frame
must *positively* say it is simulated before the HMI claims simulation.

The backend telemetry frame (``backend/poll_runtime._poll_once``) does not yet
carry an explicit connectivity flag; both ``plc_connected`` and ``simulated``
are consumed defensively here so the HMI starts reporting the truth as soon as
the backend adds one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "CONNECTED",
    "OFFLINE",
    "SIMULATING",
    "derive_connection_status",
]

#: Live PLC link (or no evidence to the contrary — the safe assumption).
CONNECTED = "Connected"

#: The backend positively reported that the values are simulated.
SIMULATING = "Simulating"

#: No usable telemetry at all.
OFFLINE = "Offline"


def derive_connection_status(frame: Any) -> str:
    """Return the connection label for a telemetry ``frame``.

    Precedence:

    1. A degraded polling report (``polling_status.status == "degraded"``) means
       the backend is not getting scans through -> ``"Offline"``.
    2. An explicit ``plc_connected`` boolean, when the backend supplies one.
    3. An explicit ``simulated``/``simulation`` boolean.
    4. Otherwise ``"Connected"`` — never claim simulation without evidence.

    Args:
        frame: A decoded telemetry payload.

    Returns:
        One of :data:`CONNECTED`, :data:`SIMULATING`, :data:`OFFLINE`.

    Raises:
        TypeError: If ``frame`` is not a mapping.
    """
    if not isinstance(frame, Mapping):
        raise TypeError(f"frame must be a mapping, got {type(frame).__name__}")

    polling_status = frame.get("polling_status")
    if isinstance(polling_status, Mapping) and polling_status.get("status") in {
        "degraded",
        "offline",
    }:
        return OFFLINE

    plc_connected = frame.get("plc_connected")
    if isinstance(plc_connected, bool):
        return CONNECTED if plc_connected else SIMULATING

    for key in ("simulated", "simulation"):
        flag = frame.get(key)
        if isinstance(flag, bool):
            return SIMULATING if flag else CONNECTED

    return CONNECTED
