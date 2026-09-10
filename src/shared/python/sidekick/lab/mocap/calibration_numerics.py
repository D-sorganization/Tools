"""Numerical adapters for the canonical calibration contracts.

OpenCV supplies lens projection and PnP; SciPy refines poses against fixed, known
world targets. Dependencies and failed solves are errors, never synthetic poses.
World coordinates and translations are in m, rotations in radians, and image
coordinates/residuals in pixels. These numerical fits do not certify field accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .calibration import FisheyeIntrinsics, PinholeIntrinsics
    from .extrinsics import CameraLayout, CameraPose

Array: TypeAlias = npt.NDArray[np.float64]
__all__: list[str] = []


def _lens(
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
) -> tuple[Array, Array, bool]:
    from .calibration import FisheyeIntrinsics

    matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ],
        dtype=float,
    )
    fisheye = isinstance(intrinsics, FisheyeIntrinsics)
    values = intrinsics.distortion.coefficients or ((0.0,) * (4 if fisheye else 5))
    return matrix, np.asarray(values, dtype=float), fisheye


def project(
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics, xyz: tuple[float, float, float]
) -> tuple[float, float]:
    """Project a finite point in front of the camera using its declared model."""
    import cv2

    points = np.asarray(xyz, dtype=float)
    if points.shape != (3,) or not np.isfinite(points).all() or points[2] <= 0:
        raise ValueError("projection needs a finite point with positive depth")
    matrix, distortion, fisheye = _lens(intrinsics)
    method = cv2.fisheye.projectPoints if fisheye else cv2.projectPoints
    result, _ = method(
        points.reshape(1, 1, 3), np.zeros(3), np.zeros(3), matrix, distortion
    )
    u, v = result.reshape(2)
    u += getattr(intrinsics, "skew", 0.0) * (v - intrinsics.cy) / intrinsics.fy
    if not np.isfinite([u, v]).all():
        raise ValueError("lens projection is outside its finite domain")
    return float(u), float(v)


def unproject(
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics, uv: tuple[float, float]
) -> tuple[float, float, float]:
    """Invert lens distortion and verify the recovered ray by reprojection."""
    import cv2

    points = np.asarray(uv, dtype=float)
    if points.shape != (2,) or not np.isfinite(points).all():
        raise ValueError("unprojection needs two finite pixel coordinates")
    matrix, distortion, fisheye = _lens(intrinsics)
    adjusted = points.copy()
    adjusted[0] -= (
        getattr(intrinsics, "skew", 0.0) * (points[1] - intrinsics.cy) / intrinsics.fy
    )
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    if fisheye:
        normalized = cv2.fisheye.undistortPoints(
            adjusted.reshape(1, 1, 2), matrix, distortion, criteria=criteria
        )
    elif hasattr(cv2, "undistortPointsIter"):
        normalized = cv2.undistortPointsIter(
            adjusted.reshape(1, 1, 2),
            matrix,
            distortion,
            np.eye(3),
            np.eye(3),
            criteria,
        )
    else:
        # OpenCV 5 folds the iterative overload into undistortPoints.
        normalized = cv2.undistortPoints(
            adjusted.reshape(1, 1, 2),
            matrix,
            distortion,
            R=np.eye(3),
            P=np.eye(3),
            criteria=criteria,
        )
    ray = np.append(normalized.reshape(2), 1.0)
    ray /= np.linalg.norm(ray)
    result = (float(ray[0]), float(ray[1]), float(ray[2]))
    if not np.isfinite(ray).all() or not np.allclose(
        project(intrinsics, result), points, atol=1e-5, rtol=0
    ):
        raise ValueError("lens inverse did not converge to the requested pixel")
    return result


def correspondences(
    object_points: Sequence[tuple[float, float, float]],
    image_points: Sequence[tuple[float, float]],
) -> tuple[Array, Array]:
    """Validate finite, non-collinear, non-duplicated pose constraints."""
    world, pixels = (
        np.asarray(object_points, dtype=float),
        np.asarray(image_points, dtype=float),
    )
    if world.ndim != 2 or world.shape[1] != 3 or pixels.shape != (len(world), 2):
        raise ValueError("expected matching N×3 world and N×2 pixel points")
    if not 4 <= len(world) <= 10000:
        raise ValueError("pose estimation needs between 4 and 10000 correspondences")
    if not np.isfinite(world).all() or not np.isfinite(pixels).all():
        raise ValueError("calibration correspondences must be finite")
    if len(np.unique(world, axis=0)) != len(world) or len(
        np.unique(pixels, axis=0)
    ) != len(pixels):
        raise ValueError("calibration correspondences must be distinct")
    for values in (world, pixels):
        singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
        if singular[0] == 0 or singular[1] / singular[0] < 1e-6:
            raise ValueError("collinear or degenerate calibration correspondences")
    return np.ascontiguousarray(world), np.ascontiguousarray(pixels)


def _pose(camera_key: str, parameters: Array, target: str, source: str) -> CameraPose:
    from scipy.spatial.transform import Rotation

    from .extrinsics import CameraPose
    from .geometry import RigidTransform

    xyzw = Rotation.from_rotvec(parameters[:3]).as_quat()
    rotation = (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))
    translation = (float(parameters[3]), float(parameters[4]), float(parameters[5]))
    return CameraPose(camera_key, RigidTransform(target, source, rotation, translation))


def _parameters(pose: CameraPose) -> Array:
    from scipy.spatial.transform import Rotation

    transform = pose.t_camera_from_world
    w, x, y, z = transform.rotation_wxyz
    rvec = Rotation.from_quat([x, y, z, w]).as_rotvec()
    return np.concatenate((rvec, transform.translation_m))


def residuals(
    parameters: Array,
    world: Array,
    observed: Array,
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
) -> Array:
    """Pixel residuals in the original distorted image, with positive depth."""
    from scipy.spatial.transform import Rotation

    transformed = Rotation.from_rotvec(parameters[:3]).apply(world) + parameters[3:]
    if not np.isfinite(transformed).all() or np.any(transformed[:, 2] <= 0):
        raise ValueError("calibration points must remain in front of the camera")
    projected = np.asarray([project(intrinsics, tuple(point)) for point in transformed])
    return (projected - observed).reshape(-1)


def solve_pose(
    camera_key: str,
    object_points: Sequence[tuple[float, float, float]],
    image_points: Sequence[tuple[float, float]],
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
    target_frame_id: str,
    source_frame_id: str,
) -> tuple[CameraPose, float]:
    """Estimate a real pose; unsuccessful/missing numerical backends raise."""
    import cv2

    world, observed = correspondences(object_points, image_points)
    rays = np.asarray([intrinsics.unproject_point(tuple(pixel)) for pixel in observed])
    normalized = np.ascontiguousarray(rays[:, :2] / rays[:, 2:])
    planar = np.linalg.matrix_rank(world - world.mean(axis=0)) == 2
    method = cv2.SOLVEPNP_ITERATIVE if planar or len(world) >= 6 else cv2.SOLVEPNP_SQPNP
    try:
        success, rotation, translation = cv2.solvePnP(
            world, normalized, np.eye(3), None, flags=method
        )
    except cv2.error as exc:
        raise RuntimeError(f"PnP solve failed for camera {camera_key}: {exc}") from exc
    if not success:
        raise RuntimeError(f"PnP solve did not converge for camera {camera_key}")
    parameters = np.concatenate((rotation.reshape(3), translation.reshape(3))).astype(
        np.float64
    )
    errors = residuals(parameters, world, observed, intrinsics).reshape(-1, 2)
    pose = _pose(camera_key, parameters, target_frame_id, source_frame_id)
    return pose, float(np.linalg.norm(errors, axis=1).mean())


def _refine(
    pose: CameraPose,
    world: Array,
    observed: Array,
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
    fixed: bool,
) -> tuple[CameraPose, Array]:
    from scipy.optimize import least_squares

    initial = _parameters(pose)
    residuals(initial, world, observed, intrinsics)
    if fixed:
        return pose, residuals(initial, world, observed, intrinsics)
    fit = least_squares(
        residuals,
        initial,
        args=(world, observed, intrinsics),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=200,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"camera refinement did not converge: {fit.message}")
    if np.linalg.matrix_rank(fit.jac) < 6:
        raise ValueError("camera refinement is geometrically unobservable")
    transform = pose.t_camera_from_world
    fitted = _pose(
        pose.camera_key, fit.x, transform.target_frame_id, transform.source_frame_id
    )
    return fitted, residuals(fit.x, world, observed, intrinsics)


def refine_layout(
    initial: CameraLayout,
    observations: Mapping[
        str, Sequence[tuple[tuple[float, float, float], tuple[float, float]]]
    ],
    intrinsics: Mapping[str, PinholeIntrinsics | FisheyeIntrinsics],
    fixed_camera: str | None,
) -> tuple[CameraLayout, list[float]]:
    """Minimize the separable pose blocks of a fixed-world-target objective.

    Known world coordinates fix the gauge. No camera is frozen by default;
    an explicit fixed_camera preserves that camera exactly. Unknown landmark
    positions and moving-target placements require a different observation model.
    """
    from .extrinsics import CameraLayout

    keys = set(initial.camera_poses)
    if set(observations) != keys or set(intrinsics) != keys:
        raise ValueError(
            "every layout camera needs matching observations and intrinsics"
        )
    if fixed_camera is not None and fixed_camera not in keys:
        raise KeyError(f"gauge camera {fixed_camera!r} not found in layout")
    if len(keys) > 64:
        raise ValueError("calibration supports at most 64 cameras per layout")
    poses, errors = {}, []
    for key in sorted(keys):
        if not observations[key]:
            raise ValueError(f"camera {key} has no calibration observations")
        world, observed = correspondences(*zip(*observations[key], strict=True))
        poses[key], differences = _refine(
            initial.get_pose(key), world, observed, intrinsics[key], key == fixed_camera
        )
        errors.extend(np.linalg.norm(differences.reshape(-1, 2), axis=1).tolist())
    return CameraLayout(initial.layout_id, initial.world_frame, poses), errors
