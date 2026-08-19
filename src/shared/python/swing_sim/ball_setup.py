"""Canonical golf-ball support geometry shared by simulation front ends.

Tee height is measured vertically from the ground plane to the bottom of
the ball.  The ball center is therefore one radius above that support
height.  Keeping this convention in a typed value prevents UI-specific
interpretations from silently moving the contact sphere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from shared.python.contracts import require

from .impact import GOLF_BALL_RADIUS_M

DEFAULT_DRIVER_TEE_HEIGHT_M = 0.0381
"""Representative editable driver tee height: 1.5 in to ball bottom."""

HEIGHT_REFERENCE = "ground_plane_to_ball_bottom"


class BallSupportMode(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Physical support under the ball."""

    GROUND = "ground"
    TEE = "tee"


@dataclass(frozen=True)
class BallSetup:
    """Validated ball support mode and vertical placement in app coordinates."""

    support_mode: BallSupportMode = BallSupportMode.GROUND
    tee_height_m: float = 0.0

    def __post_init__(self) -> None:
        require(
            isinstance(self.support_mode, BallSupportMode),
            "support_mode must be a BallSupportMode",
            self.support_mode,
        )
        require(
            math.isfinite(self.tee_height_m) and self.tee_height_m >= 0.0,
            "tee_height_m must be finite and >= 0",
            self.tee_height_m,
        )
        require(
            self.support_mode is BallSupportMode.TEE or self.tee_height_m == 0.0,
            "Ground support requires tee_height_m == 0",
            self.tee_height_m,
        )

    @property
    def ball_center_height_m(self) -> float:
        """Vertical ground-plane distance to the ball center."""
        return GOLF_BALL_RADIUS_M + self.tee_height_m

    @property
    def ball_center_m(self) -> tuple[float, float, float]:
        """Ball-center position in the canonical app frame (x target, y up)."""
        return (0.0, self.ball_center_height_m, 0.0)

    def to_json_dict(self) -> dict[str, Any]:
        """Return an explicit, unit-bearing persisted representation."""
        return {
            "support_mode": self.support_mode.value,
            "tee_height_m": self.tee_height_m,
            "height_reference": HEIGHT_REFERENCE,
            "ball_center_m": list(self.ball_center_m),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any] | None) -> BallSetup:
        """Load persisted geometry; absence migrates legacy ground-ball data."""
        if data is None:
            return cls(BallSupportMode.GROUND, 0.0)
        require(isinstance(data, Mapping), "ball_setup must be a mapping", data)
        reference = data.get("height_reference", HEIGHT_REFERENCE)
        require(
            reference == HEIGHT_REFERENCE,
            "unsupported ball_setup height_reference",
            reference,
        )
        raw_mode = data.get("support_mode", "ground")
        try:
            mode = (
                raw_mode
                if isinstance(raw_mode, BallSupportMode)
                else BallSupportMode(str(raw_mode))
            )
        except ValueError as error:
            raise ValueError(f"unknown ball support mode {raw_mode!r}") from error
        result = cls(mode, float(data.get("tee_height_m", 0.0)))
        if "ball_center_m" in data:
            persisted_center = tuple(float(value) for value in data["ball_center_m"])
            require(
                len(persisted_center) == 3
                and all(math.isfinite(value) for value in persisted_center)
                and all(
                    math.isclose(actual, persisted, abs_tol=1e-12)
                    for actual, persisted in zip(
                        result.ball_center_m, persisted_center, strict=True
                    )
                ),
                "ball_center_m must match the derived ball setup geometry",
                persisted_center,
            )
        return result


__all__ = [
    "DEFAULT_DRIVER_TEE_HEIGHT_M",
    "HEIGHT_REFERENCE",
    "BallSetup",
    "BallSupportMode",
]
