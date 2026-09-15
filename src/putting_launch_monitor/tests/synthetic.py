"""A synthetic overhead camera and a synthetic putt, for the tests.

The camera is a real pinhole model 2.4 m above the mat, tilted 30 degrees
like the lab's, so the image-to-ground mapping is a genuine perspective
one — not a scale. A putt is a ball rolling on the mat at a known speed
and angle; the renderer paints it on a green background so the detector
sees what it will see in the bay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

MAT_W_MM, MAT_L_MM = 1220.0, 1520.0
IMAGE_W, IMAGE_H = 960, 600
BALL_MM = 42.67


@dataclass(frozen=True)
class SyntheticCamera:
    """A pinhole camera looking down at the mat plane ``z = 0``."""

    focal_px: float = 700.0
    width: int = IMAGE_W
    height: int = IMAGE_H
    tilt_deg: float = 30.0
    height_mm: float = 2400.0

    def projection(self) -> npt.NDArray[np.float64]:
        """A look-at camera: above the near end of the mat, aimed at its centre.

        The optical axis is tilted ``tilt_deg`` from vertical, so the mat
        centre lands at the image centre and the far end is toward the top
        of the picture, as in the bay.
        """
        k = np.array(
            [
                [self.focal_px, 0, self.width / 2],
                [0, self.focal_px, self.height / 2],
                [0, 0, 1.0],
            ]
        )
        t = np.radians(self.tilt_deg)
        target = np.array([MAT_W_MM / 2, MAT_L_MM / 2, 0.0])
        cam_pos = target + np.array([0.0, -self.height_mm * np.tan(t), self.height_mm])
        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.array([1.0, 0.0, 0.0])
        down = np.cross(forward, right)  # image y grows downward: toward the player
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        r = np.vstack([right, down, forward])
        tvec = -r @ cam_pos
        return k @ np.hstack([r, tvec[:, None]])

    def project(self, xy_mm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xy = np.atleast_2d(np.asarray(xy_mm, dtype=np.float64))
        pts = np.hstack([xy, np.zeros((len(xy), 1)), np.ones((len(xy), 1))])
        uvw = pts @ self.projection().T
        return uvw[:, :2] / uvw[:, 2:3]

    def mat_corners_px(self) -> npt.NDArray[np.float64]:
        world = np.array(
            [[0, 0], [MAT_W_MM, 0], [MAT_W_MM, MAT_L_MM], [0, MAT_L_MM]],
            dtype=np.float64,
        )
        return self.project(world)


def render_frame(
    camera: SyntheticCamera,
    ball_mm: tuple[float, float] | None,
    *,
    noise: float = 0.0,
    seed: int = 0,
) -> npt.NDArray[np.uint8]:
    """A green mat with a white ball at ``ball_mm`` (None: no ball)."""
    import cv2

    frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    frame[:] = (40, 120, 50)  # BGR green
    corners = camera.mat_corners_px().astype(np.int32)
    cv2.fillPoly(frame, [corners], (60, 150, 70))  # lighter hitting mat
    if ball_mm is not None:
        centre = camera.project(np.asarray(ball_mm))[0]
        # the ball's apparent radius follows the local ground scale
        edge = camera.project(np.asarray([ball_mm[0] + BALL_MM / 2, ball_mm[1]]))[0]
        radius = max(3, int(round(np.linalg.norm(edge - centre))))
        cv2.circle(
            frame,
            (int(round(centre[0])), int(round(centre[1]))),
            radius,
            (245, 245, 245),
            -1,
        )
    if noise > 0:
        rng = np.random.default_rng(seed)
        frame = np.clip(
            frame.astype(np.int16) + rng.normal(0, noise, frame.shape), 0, 255
        ).astype(np.uint8)
    return frame


def putt_positions(
    start_mm: tuple[float, float],
    speed_mps: float,
    hla_deg: float,
    fps: float,
    seconds: float,
    *,
    decel_mps2: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """``(times_s, positions_mm)`` of a ball launched at a speed and angle."""
    n = int(round(seconds * fps))
    t = np.arange(n) / fps
    v = np.maximum(speed_mps - decel_mps2 * t, 0.0)
    dist = np.cumsum(v) / fps * 1000.0
    dist = np.concatenate([[0.0], dist[:-1]])
    a = np.radians(hla_deg)
    d = np.array([np.sin(a), np.cos(a)])
    return t, np.asarray(start_mm) + np.outer(dist, d)
