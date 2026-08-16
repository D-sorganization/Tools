"""Pydantic models + enums for the global performance mode.

Split from ``performance.py`` (the controller) so these stay plain data,
importable anywhere without business logic — mirrors the temperature/
power-supply model split.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from shared.python.compatibility import StrEnum


class PerformanceMode(StrEnum):
    """Operator-selectable scan-cadence mode."""

    PERFORMANCE = "performance"
    LIGHTWEIGHT = "lightweight"


class PerformanceConfig(BaseModel):
    """Active performance mode, the cadences it implies, and loop health.

    The mode governs ONLY how often the live frame is pushed to the HMI. The
    control cadence (``scan_interval_s``) is fixed by the backend settings and
    is reported here read-only so an operator can see that a hidden browser tab
    did not slow the plant down (issue #4008).
    """

    mode: PerformanceMode
    poll_interval_s: float = Field(
        ge=0.0,
        description=(
            "Seconds between WebSocket frames for the active mode (read-only). "
            "Retained under its historical name for the HMI client; it is the "
            "broadcast period, NOT the control period."
        ),
    )
    broadcast_interval_s: float = Field(
        ge=0.0,
        description="Seconds between WebSocket frames for the active mode.",
    )
    scan_interval_s: float = Field(
        gt=0.0,
        description=(
            "Fixed PLC scan / alarm / control period. Never changed by the mode."
        ),
    )
    broadcast_every_n: int = Field(
        ge=1,
        description="Scans per broadcast — how the mode decimates the stream.",
    )
    scan_overruns: int = Field(
        default=0,
        ge=0,
        description="Scans that missed their monotonic deadline since boot.",
    )
    historian_write_failures: int = Field(
        default=0,
        ge=0,
        description="Historian batches that failed to commit after retries.",
    )
