# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Parity tests for behaviours upstreamed from the retired UpstreamDrift copy (#5515).

The expected values mirror what the UD in-tree ``movement_optimizer`` copy produced
before it was retired (UD #12267): standard gravity 9.80665 m/s^2, the four DbC
accessors, and the force-arrow -> wrench conversion semantics of
``arrows_to_force_torque_frame`` (zero vectors skipped, vectors scaled).
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QImage, QPainter

from movement_optimizer.constants import GRAVITY_M_S2
from movement_optimizer.gui.vector_overlay import (
    ForceArrow,
    OverlayScene,
    VectorStyle,
    draw_force_arrows,
    draw_overlay_scene,
    nonzero_arrows,
    scaled_arrow_vectors,
)
from movement_optimizer.models import BodyModel, chain_dynamics
from movement_optimizer.models.chain_dynamics import (
    ChainConfig,
    ChainState,
    initial_catenary_angles,
    simulate_chain,
)
from movement_optimizer.models.swingset import (
    SwingSetConfig,
    SwingSetState,
    heuristic_pumping_policy,
    simulate_swingset,
)
from movement_optimizer.trajectory.result import OptimizationResult

# --- standard gravity (UD #8322) -------------------------------------------


def test_standard_gravity_value_matches_ud() -> None:
    assert GRAVITY_M_S2 == 9.80665


def test_body_model_uses_shared_standard_gravity() -> None:
    assert BodyModel(75.0, 1.75).g == GRAVITY_M_S2


def test_chain_dynamics_gravity_is_the_same_constant() -> None:
    assert chain_dynamics.GRAVITY_M_S2 == GRAVITY_M_S2


# --- DbC accessors ---------------------------------------------------------


def _result(
    n: int = 7, dof: int = 3, com_rows: int | None = None
) -> OptimizationResult:
    rows = n if com_rows is None else com_rows
    zeros = np.zeros((n, dof))
    return OptimizationResult(
        t=np.linspace(0.0, 1.0, n),
        q=zeros,
        qd=zeros,
        qdd=zeros,
        torques=zeros,
        power=zeros,
        com=np.zeros((rows, 2)),
        bar=np.zeros((n, 2)),
        success=True,
        cost=0.0,
        com_horizontal_range_cm=0.0,
    )


def test_torque_joint_count() -> None:
    assert _result(dof=4).torque_joint_count() == 4


def test_has_com_samples() -> None:
    assert _result().has_com_samples() is True
    assert _result(com_rows=0).has_com_samples() is False


def test_chain_rollout_frame_count() -> None:
    config = ChainConfig(segment_count=3)
    initial = ChainState(
        initial_catenary_angles(config.segment_count, 0.25),
        np.zeros(config.segment_count),
    )
    rollout = simulate_chain(config, initial, steps=5, dt_s=0.01)
    assert rollout.frame_count() == 6 == rollout.positions.shape[0]


def test_swing_rollout_max_abs_swing_angle() -> None:
    config = SwingSetConfig()
    rollout = simulate_swingset(
        config, SwingSetState.rest(), 4, 0.01, heuristic_pumping_policy
    )
    assert rollout.max_abs_swing_angle_rad() == rollout.metrics.max_abs_swing_angle_rad


# --- injectable force-arrow renderer (item 1) ------------------------------


@pytest.fixture()
def arrows() -> tuple[ForceArrow, ...]:
    style = VectorStyle(color=QColor("#ff0000"), width=2, head_px=8.0)
    return (
        ForceArrow((0.0, 0.0), (3.0, 4.0), style),
        ForceArrow((1.0, 1.0), (0.0, 0.0), style),
        ForceArrow((2.0, -1.0), (-1.0, 0.5), style),
    )


def test_nonzero_arrows_drops_zero_vectors(arrows: tuple[ForceArrow, ...]) -> None:
    kept = nonzero_arrows(arrows)
    assert kept == (arrows[0], arrows[2])


def test_scaled_arrow_vectors_matches_ud_wrench_force(
    arrows: tuple[ForceArrow, ...],
) -> None:
    # UD: force_n=(vx*scale, vy*scale, 0.0), point_m=(ox, oy, 0.0), zero skipped.
    assert scaled_arrow_vectors(arrows, 2.0) == (
        ((0.0, 0.0, 0.0), (6.0, 8.0, 0.0)),
        ((2.0, -1.0, 0.0), (-2.0, 1.0, 0.0)),
    )


def test_scaled_arrow_vectors_rejects_bad_scale(
    arrows: tuple[ForceArrow, ...],
) -> None:
    with pytest.raises(ValueError):
        scaled_arrow_vectors(arrows, 0.0)


def _painter() -> tuple[QImage, QPainter]:
    image = QImage(40, 40, QImage.Format.Format_ARGB32)
    image.fill(0)
    return image, QPainter(image)


def _proj(point: tuple[float, float]) -> QPointF:
    return QPointF(point[0], point[1])


def test_injected_renderer_receives_nonzero_arrows(
    qapp, arrows: tuple[ForceArrow, ...]
) -> None:
    calls: list[tuple[object, tuple[ForceArrow, ...], float]] = []

    def renderer(painter, projector, drawn, scale) -> None:
        calls.append((projector, tuple(drawn), scale))

    image, painter = _painter()
    try:
        draw_force_arrows(painter, _proj, arrows, scale=2.0, renderer=renderer)
    finally:
        painter.end()
    assert calls == [(_proj, (arrows[0], arrows[2]), 2.0)]
    assert image.pixelColor(5, 5).alpha() == 0  # default drawing was bypassed


def test_injected_renderer_not_called_when_all_zero(qapp) -> None:
    style = VectorStyle(color=QColor("#00ff00"))
    zero = (ForceArrow((0.0, 0.0), (0.0, 0.0), style),)
    called: list[int] = []
    _, painter = _painter()
    try:
        draw_force_arrows(
            painter, _proj, zero, scale=1.0, renderer=lambda *a: called.append(1)
        )
    finally:
        painter.end()
    assert called == []


def test_injected_renderer_still_validates_scale(
    qapp, arrows: tuple[ForceArrow, ...]
) -> None:
    _, painter = _painter()
    try:
        with pytest.raises(ValueError):
            draw_force_arrows(
                painter, _proj, arrows, scale=-1.0, renderer=lambda *a: None
            )
    finally:
        painter.end()


def test_overlay_scene_passes_renderer_through(
    qapp, arrows: tuple[ForceArrow, ...]
) -> None:
    seen: list[float] = []
    _, painter = _painter()
    try:
        draw_overlay_scene(
            painter,
            _proj,
            OverlayScene(arrows=arrows),
            arrow_scale=3.0,
            torque_reference_nm=1.0,
            arrow_renderer=lambda p, pr, a, s: seen.append(s),
        )
    finally:
        painter.end()
    assert seen == [3.0]
