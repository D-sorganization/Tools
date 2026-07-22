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
    """Current performance mode + the resolved poll interval it implies."""

    mode: PerformanceMode
    poll_interval_s: float = Field(
        ge=0.0,
        description="Seconds between PLC scans for the active mode (read-only).",
    )
