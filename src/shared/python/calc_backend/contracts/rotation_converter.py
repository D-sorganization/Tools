"""Pydantic contracts for Rotation Converter endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RotationConverterRequest(BaseModel):
    """Request model for converting a rotation representation.

    Exactly one of the representations should be provided.
    """

    type: Literal[
        "quaternion", "euler", "axis_angle", "rodrigues", "rotation_matrix"
    ] = Field(
        ...,
        description="The type of the input representation.",
    )
    # Payload values
    value: list[float] | list[list[float]] = Field(
        ...,
        description="The numerical values of the rotation. Length varies by type.",
    )
    euler_convention: str = Field(
        default="xyz",
        description="Euler angle convention (e.g., 'xyz', 'zyx'). Needed if type == 'euler', and used for output.",
    )


class RotationRepresentationsModel(BaseModel):
    """Contains all standard representations of a single rotation."""

    quaternion: list[float] = Field(description="Unit quaternion [w, x, y, z]")
    euler: list[float] = Field(
        description="Euler angles in requested convention [a, b, c]"
    )
    euler_convention: str = Field(
        description="The convention used for output euler angles"
    )
    axis_angle: dict[str, list[float] | float] = Field(
        description="Axis-angle representation {'axis': [x, y, z], 'angle': theta}"
    )
    rodrigues: list[float] = Field(description="Rodrigues vector [rx, ry, rz]")
    rotation_matrix: list[list[float]] = Field(description="3x3 Rotation Matrix")


class RotationConverterResponse(BaseModel):
    """Response model for a successful rotation conversion."""

    representations: RotationRepresentationsModel = Field(
        ..., description="All equivalent representations of the input rotation."
    )
