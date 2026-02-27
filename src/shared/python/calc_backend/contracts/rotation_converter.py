"""Pydantic contracts for Rotation Converter endpoint."""

from __future__ import annotations

from typing import Any, Literal

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


class ReferenceFrameConversionRequest(BaseModel):
    """Request model for reference-frame and Lie-group educational operations."""

    operation: Literal[
        "twist_frame_conversion",
        "homogeneous_transform",
        "so3_so3_maps",
    ] = Field(..., description="Requested operation mode.")
    transform: list[list[float]] | None = Field(
        default=None,
        description="4x4 homogeneous transform used for twist conversion.",
    )
    twist: list[float] | None = Field(
        default=None,
        description="Input twist [omega_x, omega_y, omega_z, v_x, v_y, v_z].",
    )
    rotation_matrix: list[list[float]] | None = Field(
        default=None,
        description="3x3 rotation matrix input for SE(3) construction or SO(3) log map.",
    )
    translation: list[float] | None = Field(
        default=None,
        description="3-vector translation used when constructing a homogeneous transform.",
    )
    so3_vector: list[float] | None = Field(
        default=None,
        description="3-vector in so(3) (axis-angle / rotation vector form).",
    )
    so3_matrix: list[list[float]] | None = Field(
        default=None,
        description="3x3 skew-symmetric matrix in so(3).",
    )


class ReferenceFrameConversionResponse(BaseModel):
    """Response model for reference-frame and Lie-group operations."""

    operation: str = Field(..., description="Echoed operation name.")
    results: dict[str, Any] = Field(
        ...,
        description="Numeric outputs (matrices, vectors, transforms).",
    )
    explanation_markdown: str = Field(
        ...,
        description="Educational explanation rendered in markdown format.",
    )
    explanation_latex: str = Field(
        ...,
        description="Key formulas expressed in LaTeX form.",
    )
