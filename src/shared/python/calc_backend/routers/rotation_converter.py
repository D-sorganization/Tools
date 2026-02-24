"""Rotation Converter calculator router."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from ..contracts.rotation_converter import (
    RotationConverterRequest,
    RotationConverterResponse,
    RotationRepresentationsModel,
)

router = APIRouter(prefix="/api/calc/rotation-converter", tags=["rotation-converter"])


@router.post("", response_model=RotationConverterResponse)
def compute_rotation(request: RotationConverterRequest) -> RotationConverterResponse:
    """Convert between different 3D rotation representations."""
    try:
        from rotation_converter.converter import Rotation
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Rotation Converter package is not available in the environment.",
        ) from e

    rot = None
    try:
        if request.type == "quaternion":
            rot = Rotation.from_quaternion(request.value)
        elif request.type == "euler":
            val = request.value
            if not isinstance(val, list) or len(val) != 3:
                raise ValueError("Euler angles must be a list of 3 floats.")
            rot = Rotation.from_euler(
                float(val[0]), float(val[1]), float(val[2]), request.euler_convention
            )
        elif request.type == "axis_angle":
            val = request.value
            if not isinstance(val, list) or len(val) != 4:
                raise ValueError(
                    "Axis-angle must be a list of 4 floats: [x, y, z, angle]."
                )
            axis = np.array([float(val[0]), float(val[1]), float(val[2])])
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            rot = Rotation.from_axis_angle(axis, float(val[3]))
        elif request.type == "rodrigues":
            if not isinstance(request.value, list) or len(request.value) != 3:
                raise ValueError("Rodrigues vector must be a list of 3 floats.")
            rot = Rotation.from_rodrigues(request.value)
        elif request.type == "rotation_matrix":
            rot = Rotation.from_rotation_matrix(request.value)
        else:
            raise ValueError(f"Unknown representation type: {request.type}")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Invalid rotation input: {exc}"
        ) from exc

    try:
        # Generate outputs
        quat = rot.as_quaternion().tolist()

        # Try to get requested Euler notation, default to xyz to ensure no failure
        try:
            eul = list(rot.as_euler(request.euler_convention))
            conv = request.euler_convention
        except Exception:
            eul = list(rot.as_euler("xyz"))
            conv = "xyz"

        ax_ang = rot.as_axis_angle()
        axis_list = ax_ang[0].tolist()
        angle = float(ax_ang[1])

        rod = rot.as_rodrigues().tolist()
        rot_mat = rot.as_rotation_matrix().tolist()

        rep_model = RotationRepresentationsModel(
            quaternion=quat,
            euler=eul,
            euler_convention=conv,
            axis_angle={"axis": axis_list, "angle": angle},
            rodrigues=rod,
            rotation_matrix=rot_mat,
        )

        return RotationConverterResponse(representations=rep_model)

    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed building outputs: {exc}"
        ) from exc
