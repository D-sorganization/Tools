"""Bounded timing policy for regional-ground authority clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class RegionalGroundAuthorityClientError(RuntimeError):
    """Non-sensitive transport or authority failure identity."""

    def __init__(self, code: str) -> None:
        """Publish only a stable non-sensitive error code."""
        self.code = code
        super().__init__(f"regional-ground authority client failed ({code})")


@dataclass(frozen=True, slots=True)
class RegionalGroundAuthorityPollPolicy:
    """Bounded polling, request, and shutdown timing policy."""

    poll_timeout_s: float = 300.0
    initial_interval_s: float = 0.05
    maximum_interval_s: float = 1.0
    backoff_multiplier: float = 1.5
    request_timeout_s: float = 5.0
    shutdown_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        """Enforce finite practical timing bounds."""
        values = (
            self.poll_timeout_s,
            self.initial_interval_s,
            self.maximum_interval_s,
            self.request_timeout_s,
            self.shutdown_timeout_s,
        )
        if any(type(value) not in (int, float) or value <= 0.0 for value in values):
            raise ValueError("authority timing values must be positive")
        if self.poll_timeout_s > 3_600.0 or self.request_timeout_s > 30.0:
            raise ValueError("authority timeout exceeds supported bound")
        if self.maximum_interval_s < self.initial_interval_s:
            raise ValueError("maximum interval must not be below initial interval")
        if not 1.0 <= self.backoff_multiplier <= 4.0:
            raise ValueError("backoff_multiplier must lie within [1, 4]")


DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY: Final = (
    RegionalGroundAuthorityPollPolicy()
)


__all__ = [
    "DEFAULT_REGIONAL_GROUND_AUTHORITY_POLL_POLICY",
    "RegionalGroundAuthorityClientError",
    "RegionalGroundAuthorityPollPolicy",
]
