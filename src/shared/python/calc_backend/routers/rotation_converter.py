"""Rotation Converter calculator router."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from ..contracts.rotation_converter import (
    ReferenceFrameConversionRequest,
    ReferenceFrameConversionResponse,
    RotationConverterRequest,
    RotationConverterResponse,
    RotationRepresentationsModel,
)

router = APIRouter(prefix="/api/calc/rotation-converter", tags=["rotation-converter"])


def _skew(omega: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ],
        dtype=float,
    )


def _vee(so3_matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [so3_matrix[2, 1], so3_matrix[0, 2], so3_matrix[1, 0]],
        dtype=float,
    )


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
                float(str(val[0])),
                float(str(val[1])),
                float(str(val[2])),
                request.euler_convention,
            )
        elif request.type == "axis_angle":
            val = request.value
            if not isinstance(val, list) or len(val) != 4:
                raise ValueError(
                    "Axis-angle must be a list of 4 floats: [x, y, z, angle]."
                )
            axis = np.array(
                [float(str(val[0])), float(str(val[1])), float(str(val[2]))]
            )
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                axis = axis / norm
            rot = Rotation.from_axis_angle(axis, float(str(val[3])))
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


@router.post("/reference-frame", response_model=ReferenceFrameConversionResponse)
def compute_reference_frame_conversion(
    request: ReferenceFrameConversionRequest,
) -> ReferenceFrameConversionResponse:
    """Compute advanced reference-frame and Lie-group educational operations."""
    try:
        from rotation_converter.converter import Rotation
        from rotation_converter.twist_screw import adjoint_representation
    except ImportError as error:
        raise HTTPException(
            status_code=503,
            detail="Rotation Converter package is not available in the environment.",
        ) from error

    if request.operation == "twist_frame_conversion":
        if request.transform is None or request.twist is None:
            raise HTTPException(
                status_code=422,
                detail="twist_frame_conversion requires transform and twist fields.",
            )
        transform = np.asarray(request.transform, dtype=float)
        twist = np.asarray(request.twist, dtype=float)
        if transform.shape != (4, 4) or twist.shape != (6,):
            raise HTTPException(
                status_code=422,
                detail="Expected transform as 4x4 and twist as 6-vector.",
            )
        adjoint = adjoint_representation(transform)
        output_twist = adjoint @ twist
        return ReferenceFrameConversionResponse(
            operation=request.operation,
            results={
                "adjoint_matrix": adjoint.tolist(),
                "input_twist": twist.tolist(),
                "output_twist": output_twist.tolist(),
            },
            explanation_markdown=(
                "Twists transform with the **adjoint matrix** of a homogeneous "
                "transform:\n\n"
                "`V_b = Ad_T * V_a`, where "
                "`Ad_T = [[R, 0], [skew(p)R, R]]` for `T = [[R, p], [0, 1]]`."
            ),
            explanation_latex=(
                r"V_b = \mathrm{Ad}_T V_a,\quad "
                r"\mathrm{Ad}_T = \begin{bmatrix}R & 0\\ [p]_\times R & R\end{bmatrix}."
            ),
        )

    if request.operation == "homogeneous_transform":
        if request.rotation_matrix is None or request.translation is None:
            raise HTTPException(
                status_code=422,
                detail="homogeneous_transform requires rotation_matrix and translation.",
            )
        rotation_matrix = np.asarray(request.rotation_matrix, dtype=float)
        translation = np.asarray(request.translation, dtype=float)
        if rotation_matrix.shape != (3, 3) or translation.shape != (3,):
            raise HTTPException(
                status_code=422,
                detail="Expected rotation_matrix as 3x3 and translation as 3-vector.",
            )
        rotation = Rotation.from_rotation_matrix(rotation_matrix)
        R = rotation.as_rotation_matrix()
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = translation
        T_inv = np.eye(4, dtype=float)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -(R.T @ translation)
        return ReferenceFrameConversionResponse(
            operation=request.operation,
            results={
                "rotation_matrix": R.tolist(),
                "translation": translation.tolist(),
                "homogeneous_transform": T.tolist(),
                "inverse_transform": T_inv.tolist(),
            },
            explanation_markdown=(
                "A homogeneous transform is built as:\n\n"
                "`T = [[R, p], [0, 1]]`\n\n"
                "- `R` rotates orientation between frames.\n"
                "- `p` shifts origin position between frames.\n"
                "- Points transform as `x_b = T * x_a_h` (homogeneous coordinates)."
            ),
            explanation_latex=(
                r"T = \begin{bmatrix}R & p\\0 & 1\end{bmatrix},\quad "
                r"T^{-1}=\begin{bmatrix}R^\top & -R^\top p\\0 & 1\end{bmatrix}."
            ),
        )

    if request.operation == "so3_so3_maps":
        if request.so3_matrix is not None:
            so3_matrix = np.asarray(request.so3_matrix, dtype=float)
            if so3_matrix.shape != (3, 3):
                raise HTTPException(
                    status_code=422,
                    detail="so3_matrix must be 3x3.",
                )
            so3_vector = _vee(so3_matrix)
        elif request.so3_vector is not None:
            so3_vector = np.asarray(request.so3_vector, dtype=float)
            if so3_vector.shape != (3,):
                raise HTTPException(
                    status_code=422,
                    detail="so3_vector must be a 3-vector.",
                )
            so3_matrix = _skew(so3_vector)
        elif request.rotation_matrix is not None:
            rotation = Rotation.from_rotation_matrix(request.rotation_matrix)
            so3_vector = rotation.as_rodrigues()
            so3_matrix = _skew(so3_vector)
        else:
            raise HTTPException(
                status_code=422,
                detail="so3_so3_maps requires so3_vector, so3_matrix, or rotation_matrix.",
            )
        rotation = Rotation.from_rodrigues(so3_vector)
        so3_log = rotation.as_rodrigues()
        return ReferenceFrameConversionResponse(
            operation=request.operation,
            results={
                "so3_vector": so3_vector.tolist(),
                "so3_hat_matrix": so3_matrix.tolist(),
                "so3_vee_vector": _vee(so3_matrix).tolist(),
                "so3_exponential_SO3": rotation.as_rotation_matrix().tolist(),
                "so3_log_vector": so3_log.tolist(),
            },
            explanation_markdown=(
                "The **hat** map sends `omega` in `R^3` to a skew matrix "
                "`omega^` in `so(3)`.\n\n"
                "The matrix exponential maps `so(3) -> SO(3)`:\n"
                "`R = exp(omega^)` (Rodrigues formula).\n\n"
                "The logarithm maps back with `omega = vee(log(R))`."
            ),
            explanation_latex=(
                r"\widehat{\omega}=\begin{bmatrix}0&-\omega_3&\omega_2\\"
                r"\omega_3&0&-\omega_1\\-\omega_2&\omega_1&0\end{bmatrix},\quad "
                r"R=\exp(\widehat{\omega}),\quad "
                r"\omega=\mathrm{vee}(\log R)."
            ),
        )

    raise HTTPException(
        status_code=422,
        detail=f"Unsupported operation: {request.operation}",
    )
