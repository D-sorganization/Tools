"""Joint camera/reference-placement estimation behind the shared contracts.

Fixed intrinsics, fixed local target geometry, and one explicit world anchor.
Transforms are in metres/radians; optimization residuals are distorted pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration_numerics import Array, _parameters, _pose, correspondences, residuals
from .extrinsics import CameraLayout, CameraPose, estimate_pnp_pose
from .geometry import CoordinateFrame, RigidTransform
from .reference_placements import (
    PlacementObservation,
    PlacementResidual,
    ReferenceCamera,
    ReferenceLayoutResult,
    ReferenceSolveCancelled,
    ReferenceTarget,
)

__all__: list[str] = []


def _matrix(parameters: Array) -> Array:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:]
    return result


def _vector(matrix: Array) -> Array:
    return np.concatenate(
        (Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3])
    )


def _validate(
    world: CoordinateFrame,
    targets: Mapping[str, ReferenceTarget],
    cameras: Mapping[str, ReferenceCamera],
    observations: tuple[PlacementObservation, ...],
    anchor_id: str,
    anchor: RigidTransform,
) -> tuple[dict[str, str], list[tuple[Array, Array]]]:
    if world.length_unit != "m" or world.handedness != "right-handed":
        raise ValueError("reference solver needs a right-handed metre world frame")
    if (
        anchor.target_frame_id != world.frame_id
        or anchor.source_frame_id != f"placement:{anchor_id}"
    ):
        raise ValueError(
            "anchor must explicitly map the anchor placement into this world"
        )
    if not 2 <= len(cameras) <= 12 or not 2 <= len(observations) <= 256:
        raise ValueError(
            "reference solver supports 2–12 cameras and 2–256 view observations"
        )
    if any(key != item.camera_key for key, item in cameras.items()):
        raise ValueError("camera mapping key does not match its camera")
    if any(key != item.reference_id for key, item in targets.items()):
        raise ValueError("target mapping key does not match its reference")
    placements: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    pairs = []
    for observation in observations:
        key = (observation.camera_key, observation.placement_id)
        if key in seen:
            raise ValueError(
                "duplicate camera/placement observation; "
                "use a new placement or replace the view"
            )
        seen.add(key)
        if (
            observation.camera_key not in cameras
            or observation.reference_id not in targets
        ):
            raise ValueError("observation references an unknown camera or target")
        camera = cameras[observation.camera_key]
        if observation.profile_id != camera.profile_id:
            raise ValueError(
                "observation profile differs from the selected lens/zoom revision"
            )
        previous = placements.setdefault(
            observation.placement_id, observation.reference_id
        )
        if previous != observation.reference_id:
            raise ValueError(
                "one placement must identify the same physical reference in every view"
            )
        canonical = observation.calibration_observation(
            targets[observation.reference_id]
        )
        if len(canonical.target_object_points) < 4:
            raise ValueError(
                "pose initialization needs four non-collinear points; "
                "a ruler alone supplies scale"
            )
        points, pixels = correspondences(
            canonical.target_object_points, canonical.detected_pixel_points
        )
        if len(points) > 128:
            raise ValueError("use at most 128 identified points per view")
        width, height = camera.intrinsics.resolution_px
        if np.any(pixels < 0) or np.any(pixels >= (width, height)):
            raise ValueError("observation pixels must lie inside the calibrated image")
        pairs.append((points, pixels))
    if anchor_id not in placements or not 2 <= len(placements) <= 32:
        raise ValueError("use an observed anchor and 2–32 identified placements")
    return placements, pairs


def _initialize(
    cameras: Mapping[str, ReferenceCamera],
    observations: tuple[PlacementObservation, ...],
    pairs: list[tuple[Array, Array]],
    placements: dict[str, str],
    anchor_id: str,
    anchor: RigidTransform,
    cancel_requested: Callable[[], bool] | None,
) -> tuple[dict[str, Array], dict[str, Array]]:
    anchor_vector = _parameters(CameraPose("anchor", anchor))
    target_matrices = {anchor_id: _matrix(anchor_vector)}
    camera_matrices: dict[str, Array] = {}
    pending = [
        i for i, observation in enumerate(observations) if not observation.held_out
    ]
    while pending:
        progressed = False
        for index in pending.copy():
            if cancel_requested is not None and cancel_requested():
                raise ReferenceSolveCancelled("reference solve cancelled")
            observation = observations[index]
            camera, placement = observation.camera_key, observation.placement_id
            if camera not in camera_matrices and placement not in target_matrices:
                continue
            if camera not in camera_matrices or placement not in target_matrices:
                points, pixels = pairs[index]
                pose, _ = estimate_pnp_pose(
                    camera,
                    [(float(x), float(y), float(z)) for x, y, z in points],
                    [(float(u), float(v)) for u, v in pixels],
                    cameras[camera].intrinsics,
                    f"camera:{camera}",
                    f"placement:{placement}",
                )
                relative = _matrix(_parameters(pose))
                if placement in target_matrices:
                    camera_matrices[camera] = relative @ np.linalg.inv(
                        target_matrices[placement]
                    )
                else:
                    target_matrices[placement] = (
                        np.linalg.inv(camera_matrices[camera]) @ relative
                    )
            pending.remove(index)
            progressed = True
        if not progressed:
            raise ValueError(
                "fit observations must form a connected camera/placement graph"
            )
    if set(camera_matrices) != set(cameras) or set(target_matrices) != set(placements):
        raise ValueError(
            "every camera and placement must be connected by fit observations, "
            "not just held-out views"
        )
    return (
        {key: _vector(value) for key, value in camera_matrices.items()},
        {key: _vector(value) for key, value in target_matrices.items()},
    )


def solve(
    layout_id: str,
    world: CoordinateFrame,
    targets: Mapping[str, ReferenceTarget],
    cameras: Mapping[str, ReferenceCamera],
    observations: tuple[PlacementObservation, ...],
    anchor_id: str,
    anchor: RigidTransform,
    cancel_requested: Callable[[], bool] | None,
) -> ReferenceLayoutResult:
    """Fit the connected pose graph and report independent held-out residuals."""
    from scipy.optimize import least_squares

    placements, pairs = _validate(
        world, targets, cameras, observations, anchor_id, anchor
    )
    camera_values, target_values = _initialize(
        cameras, observations, pairs, placements, anchor_id, anchor, cancel_requested
    )
    camera_keys, target_keys = sorted(cameras), sorted(set(placements) - {anchor_id})
    keys = [("camera", key) for key in camera_keys] + [
        ("target", key) for key in target_keys
    ]
    values = {("camera", key): value for key, value in camera_values.items()}
    values.update({("target", key): value for key, value in target_values.items()})
    x0 = np.concatenate([values[key] for key in keys])
    indices = {key: slice(index * 6, (index + 1) * 6) for index, key in enumerate(keys)}

    def difference(parameters: Array, index: int) -> Array:
        observation = observations[index]
        camera = parameters[indices[("camera", observation.camera_key)]]
        target = (
            target_values[anchor_id]
            if observation.placement_id == anchor_id
            else parameters[indices[("target", observation.placement_id)]]
        )
        local, pixels = pairs[index]
        world_points = Rotation.from_rotvec(target[:3]).apply(local) + target[3:]
        return residuals(
            camera, world_points, pixels, cameras[observation.camera_key].intrinsics
        )

    fit_indices = [
        index
        for index, observation in enumerate(observations)
        if not observation.held_out
    ]

    def objective(parameters: Array) -> Array:
        if cancel_requested is not None and cancel_requested():
            raise ReferenceSolveCancelled("reference solve cancelled")
        return np.concatenate([difference(parameters, index) for index in fit_indices])

    if len(objective(x0)) < len(x0):
        raise ValueError("fit graph has insufficient independent constraints")
    fit = least_squares(
        objective,
        x0,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=150,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError("reference placement solver did not converge")
    if np.linalg.matrix_rank(fit.jac) < len(x0):
        raise ValueError("reference placement fit is locally unobservable")
    poses = {
        key: _pose(
            key, fit.x[indices[("camera", key)]], f"camera:{key}", world.frame_id
        )
        for key in camera_keys
    }
    transforms = {anchor_id: anchor}
    transforms.update(
        {
            key: _pose(
                key, fit.x[indices[("target", key)]], world.frame_id, f"placement:{key}"
            ).t_camera_from_world
            for key in target_keys
        }
    )
    anchor_matrix = _matrix(target_values[anchor_id])
    distinct = any(
        np.linalg.norm(np.asarray(transforms[key].translation_m) - anchor.translation_m)
        > 0.001
        or Rotation.from_matrix(
            anchor_matrix[:3, :3].T @ _matrix(fit.x[indices[("target", key)]])[:3, :3]
        ).magnitude()
        > 0.001
        for key in target_keys
    )
    if not distinct:
        raise ValueError(
            "repeat images at the same placement are not independent placements"
        )
    evidence = []
    for index, observation in enumerate(observations):
        errors = np.linalg.norm(difference(fit.x, index).reshape(-1, 2), axis=1)
        evidence.append(
            PlacementResidual(
                observation.placement_id,
                observation.camera_key,
                observation.held_out,
                float(np.mean(errors)),
                float(np.max(errors)),
                len(errors),
            )
        )
    limits = [
        "Fixed intrinsics and physical point identities need caller verification.",
        "The world anchor is supplied, not physically verified by this fit.",
        "Numerical convergence and local rank do not certify physical accuracy "
        "or remove planar pose ambiguity.",
    ]
    for key in camera_keys:
        if not any(item.camera_key == key and item.held_out for item in evidence):
            limits.append(
                f"Camera {key} has no held-out view; add an independent check."
            )
    if any(item.held_out and item.max_error_px > 2.0 for item in evidence):
        limits.append(
            "Held-out observations exceed the 2 px review threshold; "
            "inspect profiles, identities and placements."
        )
    if any(not item.held_out and item.max_error_px > 2.0 for item in evidence):
        limits.append(
            "Fit observations exceed the 2 px review threshold; "
            "review before using this layout."
        )
    return ReferenceLayoutResult(
        CameraLayout(layout_id, world, poses),
        anchor_id,
        transforms,
        tuple(evidence),
        tuple(limits),
        observations,
        {key: camera.profile_id for key, camera in cameras.items()},
    )
