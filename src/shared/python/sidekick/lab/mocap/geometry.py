"""Coordinate-frame and rigid-transform contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._validation import require_finite, require_text


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Named right- or left-handed coordinate frame with explicit axes and units."""

    frame_id: str
    handedness: str
    x_axis: str
    y_axis: str
    z_axis: str
    length_unit: str

    def __post_init__(self) -> None:
        for name in (
            "frame_id",
            "handedness",
            "x_axis",
            "y_axis",
            "z_axis",
            "length_unit",
        ):
            object.__setattr__(self, name, require_text(getattr(self, name), name))
        if self.handedness not in {"right-handed", "left-handed"}:
            raise ValueError("handedness must be right-handed or left-handed")

    @classmethod
    def affinedrift_world_v1(cls) -> CoordinateFrame:
        """Return the candidate shared SI golf/lab world convention."""
        return cls(
            frame_id="affinedrift-world-v1",
            handedness="right-handed",
            x_axis="toward-target",
            y_axis="up",
            z_axis="right",
            length_unit="m",
        )


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """Transform mapping source-frame coordinates into a target frame."""

    target_frame_id: str
    source_frame_id: str
    rotation_wxyz: tuple[float, float, float, float]
    translation_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_frame_id",
            require_text(self.target_frame_id, "target_frame_id"),
        )
        object.__setattr__(
            self,
            "source_frame_id",
            require_text(self.source_frame_id, "source_frame_id"),
        )
        if self.target_frame_id == self.source_frame_id:
            raise ValueError("source and target frame IDs must differ")
        rotation = tuple(
            require_finite(value, "rotation_wxyz") for value in self.rotation_wxyz
        )
        translation = tuple(
            require_finite(value, "translation_m") for value in self.translation_m
        )
        if len(rotation) != 4:
            raise ValueError("rotation_wxyz must contain four values")
        if len(translation) != 3:
            raise ValueError("translation_m must contain three values")
        norm = math.sqrt(sum(value * value for value in rotation))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("rotation_wxyz must be a unit quaternion")
        object.__setattr__(self, "rotation_wxyz", rotation)
        object.__setattr__(self, "translation_m", translation)

    @property
    def transform_name(self) -> str:
        """Return an unambiguous ``T_target_from_source`` name."""
        return f"T_{self.target_frame_id}_from_{self.source_frame_id}"


__all__: list[str] = []
