# mypy: disable-error-code="no-any-return"
"""Frame-aware rigid body transformation (SE(3) with source/target labels).

Provides ``RigidTransform``, an immutable SE(3) wrapper that tracks which
coordinate frames it maps between and prevents incompatible compositions.

Convention: ``T_{target}^{source}`` transforms points from the *source*
frame into the *target* frame::

    p_world = T_world_body @ p_body

Composition obeys the chain rule — inner frames must match::

    T_world_body @ T_body_tool  = T_world_tool   (valid: "body" matches)
    T_world_body @ T_cam_tool   -> FrameError     (invalid: "body" != "cam")

Conversions to/from every rotation representation are provided so that
any parameterisation can be converted to any other.

DbC: validates SE(3) inputs, checks frame compatibility on compose/apply,
and ensures finite outputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rotation_converter._contracts import require, require_finite
from rotation_converter.converter import Rotation
from rotation_converter.core import (
    _validate_rotation_matrix,
    axis_angle_to_rotation_matrix,
    euler_to_rotation_matrix,
    normalize_quaternion,
    quaternion_to_rodrigues,
    quaternion_to_rotation_matrix,
    rodrigues_to_quaternion,
    rotation_matrix_to_axis_angle,
    rotation_matrix_to_euler,
    rotation_matrix_to_quaternion,
)
from rotation_converter.twist_screw import (
    adjoint_representation,
    homogeneous_to_twist_angle,
    screw_to_twist,
    twist_angle_to_homogeneous,
    twist_to_screw,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FrameError
# ---------------------------------------------------------------------------


class FrameError(Exception):
    """Raised when a frame-incompatible operation is attempted.

    Attributes:
        expected_frame: The frame that was expected.
        actual_frame: The frame that was found.
        operation: Description of the operation that failed.
    """

    def __init__(
        self,
        expected_frame: str,
        actual_frame: str,
        operation: str,
        detail: str = "",
    ) -> None:
        assert expected_frame is not None, "expected_frame must be provided"
        self.expected_frame = expected_frame
        self.actual_frame = actual_frame
        self.operation = operation
        msg = (
            f"Frame mismatch in {operation}: expected frame '{expected_frame}' "
            f"but got '{actual_frame}'"
        )
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_se3(T: np.ndarray) -> None:
    """Validate that T is a 4x4 SE(3) matrix."""
    require(T.shape == (4, 4), "SE(3) matrix must be 4x4", T.shape)
    require(
        np.allclose(T[3, :], [0, 0, 0, 1]),
        "bottom row must be [0,0,0,1]",
    )
    R = T[:3, :3]
    _validate_rotation_matrix(R)


# ---------------------------------------------------------------------------
# RigidTransform
# ---------------------------------------------------------------------------


class RigidTransform:
    """Immutable, frame-aware rigid body transformation.

    Internally stores a 4x4 SE(3) matrix and source/target frame labels.
    All outputs are copies — the internal state cannot be mutated.

    Frame convention: this transform maps *source* -> *target*.

    Invariants:
    - ``_T`` is always a valid SE(3) matrix
    - ``_source_frame`` and ``_target_frame`` are non-empty strings
    """

    __slots__ = ("_T", "_source_frame", "_target_frame")

    def __init__(
        self,
        T: np.ndarray,
        source_frame: str,
        target_frame: str,
    ) -> None:
        """Private — use factory methods instead."""
        assert T is not None, "T must be provided"
        T = np.asarray(T, dtype=float)
        _validate_se3(T)
        require(len(source_frame) > 0, "source_frame must be non-empty")
        require(len(target_frame) > 0, "target_frame must be non-empty")
        self._T = T.copy()
        self._source_frame = source_frame
        self._target_frame = target_frame

    # ── Factory methods ───────────────────────────────────────────

    @classmethod
    def identity(cls, frame: str) -> RigidTransform:
        """Create the identity transform (maps frame to itself)."""
        return cls(np.eye(4), source_frame=frame, target_frame=frame)

    @classmethod
    def from_matrix(cls, T: Any, *, source: str, target: str) -> RigidTransform:
        """Create from a 4x4 SE(3) homogeneous matrix."""
        T = np.asarray(T, dtype=float)
        return cls(T, source_frame=source, target_frame=target)

    @classmethod
    def from_rotation_translation(
        cls,
        R: Any,
        p: Any,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from a 3x3 rotation matrix and 3-vector translation."""
        R = np.asarray(R, dtype=float)
        p = np.asarray(p, dtype=float)
        require(R.shape == (3, 3), "R must be 3x3", R.shape)
        require(p.shape == (3,), "p must have 3 elements", p.shape)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return cls(T, source_frame=source, target_frame=target)

    @classmethod
    def from_rotation(
        cls,
        rotation: Rotation,
        p: Any,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from a Rotation object and 3-vector translation."""
        assert rotation is not None, "rotation must be provided"
        R = rotation.as_rotation_matrix()
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(R, p, source=source, target=target)

    @classmethod
    def from_quaternion_translation(
        cls,
        q: Any,
        p: Any,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from quaternion (w,x,y,z) and 3-vector translation."""
        q = np.asarray(q, dtype=float)
        require(q.shape == (4,), "quaternion must have 4 elements", q.shape)
        require_finite(q, "quaternion")
        q = normalize_quaternion(q)
        R = quaternion_to_rotation_matrix(q)
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(R, p, source=source, target=target)

    @classmethod
    def from_euler_translation(
        cls,
        a: float,
        b: float,
        c: float,
        p: Any,
        *,
        convention: str,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from Euler angles (a, b, c) and 3-vector translation."""
        assert a is not None, "a must be provided"
        R = euler_to_rotation_matrix(a, b, c, convention)
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(R, p, source=source, target=target)

    @classmethod
    def from_axis_angle_translation(
        cls,
        axis: Any,
        angle: float,
        p: Any,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from axis-angle rotation and 3-vector translation."""
        assert angle is not None, "angle must be provided"
        R = axis_angle_to_rotation_matrix(axis, angle)
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(R, p, source=source, target=target)

    @classmethod
    def from_rodrigues_translation(
        cls,
        r: Any,
        p: Any,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from Rodrigues vector and 3-vector translation."""
        r = np.asarray(r, dtype=float)
        require(r.shape == (3,), "Rodrigues vector must have 3 elements", r.shape)
        q = rodrigues_to_quaternion(r)
        R = quaternion_to_rotation_matrix(q)
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(R, p, source=source, target=target)

    @classmethod
    def from_twist(
        cls,
        twist: Any,
        theta: float,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from a twist vector [omega; v] and angle theta.

        Uses the matrix exponential: T = exp([twist] * theta).
        """
        assert theta is not None, "theta must be provided"
        twist = np.asarray(twist, dtype=float)
        require(twist.shape == (6,), "twist must have 6 elements", twist.shape)
        T = twist_angle_to_homogeneous(twist, theta)
        return cls(T, source_frame=source, target_frame=target)

    @classmethod
    def from_screw(
        cls,
        screw: dict[str, Any],
        theta: float,
        *,
        source: str,
        target: str,
    ) -> RigidTransform:
        """Create from screw axis parameters and rotation angle.

        Args:
            screw: dict with ``axis``, ``point``, ``pitch`` keys.
            theta: Rotation angle (radians).
        """
        assert screw is not None, "screw must be provided"
        twist = screw_to_twist(screw)
        return cls.from_twist(twist, theta, source=source, target=target)

    @classmethod
    def pure_translation(cls, p: Any, *, source: str, target: str) -> RigidTransform:
        """Create a pure translation (identity rotation)."""
        p = np.asarray(p, dtype=float)
        return cls.from_rotation_translation(np.eye(3), p, source=source, target=target)

    @classmethod
    def pure_rotation(
        cls, rotation: Rotation, *, source: str, target: str
    ) -> RigidTransform:
        """Create a pure rotation (zero translation)."""
        return cls.from_rotation(rotation, np.zeros(3), source=source, target=target)

    # ── Properties ────────────────────────────────────────────────

    @property
    def source_frame(self) -> str:
        """The frame this transform maps FROM."""
        return self._source_frame

    @property
    def target_frame(self) -> str:
        """The frame this transform maps TO."""
        return self._target_frame

    @property
    def translation(self) -> np.ndarray:
        """3-vector translation component (copy)."""
        return self._T[:3, 3].copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix component (copy)."""
        return self._T[:3, :3].copy()

    # ── Output conversions ────────────────────────────────────────

    def as_matrix(self) -> np.ndarray:
        """Return 4x4 SE(3) homogeneous matrix (copy)."""
        return self._T.copy()

    def as_rotation_translation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (3x3 rotation matrix, 3-vector translation)."""
        return self.rotation_matrix, self.translation

    def as_rotation(self) -> Rotation:
        """Return the rotational part as a Rotation object."""
        return Rotation.from_rotation_matrix(self.rotation_matrix)

    def as_quaternion_translation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (quaternion (w,x,y,z), 3-vector translation)."""
        q = rotation_matrix_to_quaternion(self._T[:3, :3])
        return q, self.translation

    def as_euler_translation(
        self, convention: str
    ) -> tuple[tuple[float, float, float], np.ndarray]:
        """Return (Euler angles tuple, 3-vector translation)."""
        assert convention is not None, "convention must be provided"
        euler = rotation_matrix_to_euler(self._T[:3, :3], convention)
        return euler, self.translation

    def as_axis_angle_translation(
        self,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Return (unit axis, angle, 3-vector translation)."""
        axis, angle = rotation_matrix_to_axis_angle(self._T[:3, :3])
        return axis, angle, self.translation

    def as_rodrigues_translation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (Rodrigues vector, 3-vector translation)."""
        q = rotation_matrix_to_quaternion(self._T[:3, :3])
        r = quaternion_to_rodrigues(q)
        return r, self.translation

    def as_twist(self) -> tuple[np.ndarray, float]:
        """Return (twist 6-vector, angle) via matrix logarithm.

        For identity transform returns (zeros(6), 0.0).
        """
        return homogeneous_to_twist_angle(self._T)

    def as_screw(self) -> dict[str, Any]:
        """Return screw axis parameters {axis, point, pitch, theta}.

        For the identity transform (theta ~ 0), returns a conventional
        default axis of [1, 0, 0] with zero pitch and angle.
        """
        twist, theta = self.as_twist()
        if abs(theta) < 1e-12 and np.linalg.norm(twist) < 1e-12:
            return {
                "axis": np.array([1.0, 0.0, 0.0]),
                "point": np.zeros(3),
                "pitch": 0.0,
                "theta": 0.0,
            }
        screw = twist_to_screw(twist)
        screw["theta"] = theta
        return screw

    # ── Predicates ────────────────────────────────────────────────

    def is_identity(self, tol: float = 1e-9) -> bool:
        """True if this is (approximately) the identity transform."""
        return bool(np.allclose(self._T, np.eye(4), atol=tol))

    def is_pure_translation(self, tol: float = 1e-9) -> bool:
        """True if rotation part is identity."""
        return bool(np.allclose(self._T[:3, :3], np.eye(3), atol=tol))

    def is_pure_rotation(self, tol: float = 1e-9) -> bool:
        """True if translation is zero."""
        return bool(np.linalg.norm(self._T[:3, 3]) < tol)

    # ── Composition (frame-checked) ───────────────────────────────

    def compose(self, other: RigidTransform) -> RigidTransform:
        """Compose: self @ other.

        Frame rule: self maps other.target -> self.target, so
        self.source_frame must equal other.target_frame.

        Result maps other.source -> self.target.
        """
        if self._source_frame != other._target_frame:
            raise FrameError(
                expected_frame=self._source_frame,
                actual_frame=other._target_frame,
                operation="compose",
                detail=(
                    f"self maps {self._source_frame}->{self._target_frame}, "
                    f"other maps {other._source_frame}->{other._target_frame}; "
                    f"self.source_frame must equal other.target_frame"
                ),
            )
        T_new = self._T @ other._T
        return RigidTransform(
            T_new,
            source_frame=other._source_frame,
            target_frame=self._target_frame,
        )

    def __matmul__(self, other: Any) -> RigidTransform:
        """Support T1 @ T2 syntax."""
        if isinstance(other, RigidTransform):
            return self.compose(other)
        return NotImplemented

    def inverse(self) -> RigidTransform:
        """Return the inverse transform (swaps source and target frames)."""
        R = self._T[:3, :3]
        p = self._T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ p
        return RigidTransform(
            T_inv,
            source_frame=self._target_frame,
            target_frame=self._source_frame,
        )

    # ── Point / vector transformations ────────────────────────────

    def apply_point(self, point: Any) -> np.ndarray:
        """Transform a 3D point: p_target = R @ p_source + t.

        Args:
            point: 3-vector in the source frame.

        Returns:
            3-vector in the target frame.
        """
        point = np.asarray(point, dtype=float)
        require(point.shape == (3,), "point must have 3 elements", point.shape)
        require_finite(point, "point")
        return self._T[:3, :3] @ point + self._T[:3, 3]

    def apply_vector(self, vector: Any) -> np.ndarray:
        """Transform a 3D direction vector: v_target = R @ v_source.

        Translation is NOT applied (vectors are direction-only).

        Args:
            vector: 3-vector in the source frame.

        Returns:
            3-vector in the target frame.
        """
        vector = np.asarray(vector, dtype=float)
        require(vector.shape == (3,), "vector must have 3 elements", vector.shape)
        require_finite(vector, "vector")
        return self._T[:3, :3] @ vector

    def apply_points(self, points: Any) -> np.ndarray:
        """Transform a batch of 3D points (Nx3).

        Args:
            points: Nx3 array of points in the source frame.

        Returns:
            Nx3 array of points in the target frame.
        """
        points = np.asarray(points, dtype=float)
        require(
            points.ndim == 2 and points.shape[1] == 3,
            "points must be Nx3",
            points.shape,
        )
        return (self._T[:3, :3] @ points.T).T + self._T[:3, 3]

    def apply_vectors(self, vectors: Any) -> np.ndarray:
        """Transform a batch of 3D direction vectors (Nx3).

        Translation is NOT applied — only the rotation is used.
        This is the batch equivalent of ``apply_vector()``.

        Args:
            vectors: Nx3 array of direction vectors in the source frame.

        Returns:
            Nx3 array of direction vectors in the target frame.
        """
        vectors = np.asarray(vectors, dtype=float)
        require(
            vectors.ndim == 2 and vectors.shape[1] == 3,
            "vectors must be Nx3",
            vectors.shape,
        )
        return (self._T[:3, :3] @ vectors.T).T

    def apply_homogeneous(self, ph: Any) -> np.ndarray:
        """Transform a 4-vector in homogeneous coordinates.

        The w-component distinguishes points from vectors:
        - ``[x, y, z, 1]`` (point): result = R @ p + t, with w=1
        - ``[x, y, z, 0]`` (vector): result = R @ v, with w=0

        This is the standard robotics convention for uniformly handling
        both points and direction vectors with a single 4x4 multiply.

        Args:
            ph: 4-vector ``[x, y, z, w]``.

        Returns:
            Transformed 4-vector.
        """
        ph = np.asarray(ph, dtype=float)
        require(ph.shape == (4,), "homogeneous vector must have 4 elements", ph.shape)
        return self._T @ ph

    def apply_homogeneous_batch(self, phs: Any) -> np.ndarray:
        """Transform a batch of 4-vectors in homogeneous coordinates (Nx4).

        Each row ``[x, y, z, w]``: use w=1 for points, w=0 for vectors.

        Args:
            phs: Nx4 array of homogeneous coordinates.

        Returns:
            Nx4 array of transformed homogeneous coordinates.
        """
        phs = np.asarray(phs, dtype=float)
        require(
            phs.ndim == 2 and phs.shape[1] == 4,
            "homogeneous batch must be Nx4",
            phs.shape,
        )
        return (self._T @ phs.T).T

    # ── Body / Space twist conversions ────────────────────────────

    def body_twist(self) -> np.ndarray:
        """Return the body-frame twist Vb such that T = exp([Vb]).

        Returns the full Lie algebra element with theta embedded (not a
        unit twist).  Use ``as_twist()`` for the (unit_twist, theta) pair.

        The body twist is the right-invariant velocity: [Vb] = log(T).

        Returns:
            6-vector [omega_b; v_b] (theta embedded in magnitude).
        """
        from rotation_converter.modern_robotics import MatrixLog6, se3ToVec

        se3_mat = MatrixLog6(self._T)
        return se3ToVec(se3_mat)

    def space_twist(self) -> np.ndarray:
        """Return the space-frame twist Vs such that T = exp([Vs]).

        Returns the full Lie algebra element with theta embedded (not a
        unit twist).  Use ``as_twist()`` for the (unit_twist, theta) pair.

        Vs = Ad_T @ Vb where Vb = body_twist().

        Returns:
            6-vector [omega_s; v_s] (theta embedded in magnitude).
        """
        Vb = self.body_twist()
        Ad = adjoint_representation(self._T)
        return Ad @ Vb

    def body_to_space_twist(self, Vb: Any) -> np.ndarray:
        """Convert a twist from body frame to space frame.

        Uses the adjoint: Vs = Ad_T @ Vb.

        Args:
            Vb: 6-vector twist in body frame.

        Returns:
            6-vector twist in space frame.
        """
        Vb = np.asarray(Vb, dtype=float)
        require(Vb.shape == (6,), "body twist must have 6 elements", Vb.shape)
        Ad = adjoint_representation(self._T)
        return Ad @ Vb

    def space_to_body_twist(self, Vs: Any) -> np.ndarray:
        """Convert a twist from space frame to body frame.

        Uses the inverse adjoint: Vb = Ad_{T^{-1}} @ Vs.

        Args:
            Vs: 6-vector twist in space frame.

        Returns:
            6-vector twist in body frame.
        """
        Vs = np.asarray(Vs, dtype=float)
        require(Vs.shape == (6,), "space twist must have 6 elements", Vs.shape)
        Ad_inv = adjoint_representation(self.inverse().as_matrix())
        return Ad_inv @ Vs

    # ── Batch twist conversions (motion data vectors) ─────────────

    def body_to_space_twists(self, Vb_batch: Any) -> np.ndarray:
        """Convert a batch of twists from body frame to space frame (Nx6).

        Vectorized version of ``body_to_space_twist()`` for efficiently
        converting time-series of motion data.

        Args:
            Vb_batch: Nx6 array of body-frame twists.

        Returns:
            Nx6 array of space-frame twists.
        """
        Vb_batch = np.asarray(Vb_batch, dtype=float)
        require(
            Vb_batch.ndim == 2 and Vb_batch.shape[1] == 6,
            "twist batch must be Nx6",
            Vb_batch.shape,
        )
        Ad = adjoint_representation(self._T)
        return (Ad @ Vb_batch.T).T

    def space_to_body_twists(self, Vs_batch: Any) -> np.ndarray:
        """Convert a batch of twists from space frame to body frame (Nx6).

        Vectorized version of ``space_to_body_twist()`` for efficiently
        converting time-series of motion data.

        Args:
            Vs_batch: Nx6 array of space-frame twists.

        Returns:
            Nx6 array of body-frame twists.
        """
        Vs_batch = np.asarray(Vs_batch, dtype=float)
        require(
            Vs_batch.ndim == 2 and Vs_batch.shape[1] == 6,
            "twist batch must be Nx6",
            Vs_batch.shape,
        )
        Ad_inv = adjoint_representation(self.inverse().as_matrix())
        return (Ad_inv @ Vs_batch.T).T

    # ── Wrench transformations (co-adjoint) ───────────────────────

    def body_to_space_wrench(self, Fb: Any) -> np.ndarray:
        """Convert a wrench from body frame to space frame.

        Uses the co-adjoint (transpose of inverse adjoint):
        Fs = Ad_{T^{-1}}^T @ Fb.

        Args:
            Fb: 6-vector wrench [torque; force] in body frame.

        Returns:
            6-vector wrench in space frame.
        """
        Fb = np.asarray(Fb, dtype=float)
        require(Fb.shape == (6,), "body wrench must have 6 elements", Fb.shape)
        Ad_inv = adjoint_representation(self.inverse().as_matrix())
        return Ad_inv.T @ Fb

    def space_to_body_wrench(self, Fs: Any) -> np.ndarray:
        """Convert a wrench from space frame to body frame.

        Uses: Fb = Ad_T^T @ Fs.

        Args:
            Fs: 6-vector wrench [torque; force] in space frame.

        Returns:
            6-vector wrench in body frame.
        """
        Fs = np.asarray(Fs, dtype=float)
        require(Fs.shape == (6,), "space wrench must have 6 elements", Fs.shape)
        Ad = adjoint_representation(self._T)
        return Ad.T @ Fs

    # ── Batch wrench conversions (motion data vectors) ────────────

    def body_to_space_wrenches(self, Fb_batch: Any) -> np.ndarray:
        """Convert a batch of wrenches from body frame to space frame (Nx6).

        Vectorized version of ``body_to_space_wrench()`` for efficiently
        converting time-series of force/torque data.

        Args:
            Fb_batch: Nx6 array of body-frame wrenches.

        Returns:
            Nx6 array of space-frame wrenches.
        """
        Fb_batch = np.asarray(Fb_batch, dtype=float)
        require(
            Fb_batch.ndim == 2 and Fb_batch.shape[1] == 6,
            "wrench batch must be Nx6",
            Fb_batch.shape,
        )
        Ad_inv = adjoint_representation(self.inverse().as_matrix())
        return (Ad_inv.T @ Fb_batch.T).T

    def space_to_body_wrenches(self, Fs_batch: Any) -> np.ndarray:
        """Convert a batch of wrenches from space frame to body frame (Nx6).

        Vectorized version of ``space_to_body_wrench()`` for efficiently
        converting time-series of force/torque data.

        Args:
            Fs_batch: Nx6 array of space-frame wrenches.

        Returns:
            Nx6 array of body-frame wrenches.
        """
        Fs_batch = np.asarray(Fs_batch, dtype=float)
        require(
            Fs_batch.ndim == 2 and Fs_batch.shape[1] == 6,
            "wrench batch must be Nx6",
            Fs_batch.shape,
        )
        Ad = adjoint_representation(self._T)
        return (Ad.T @ Fs_batch.T).T

    # ── Dunder methods ────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RigidTransform({self._source_frame} -> {self._target_frame}, "
            f"p={self._T[:3, 3]})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RigidTransform):
            return NotImplemented
        return (
            self._source_frame == other._source_frame
            and self._target_frame == other._target_frame
            and bool(np.allclose(self._T, other._T, atol=1e-10))
        )
