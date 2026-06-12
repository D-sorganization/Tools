"""Tests that swing rotation/consistency scores are derived from data (#3328).

Previously ``SwingAnalyzer._calculate_scores`` returned hardcoded placeholders
``rotation=75`` and ``consistency=80`` regardless of the analyzed swing. These
tests assert the scores now respond to the measured X-factor and to the pose
detection-confidence stability.

The ``video_analyzer`` package transitively imports OpenCV (a declared project
dependency); these tests skip cleanly where it is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

from src.video_analyzer.analyzer import SwingAnalyzer  # noqa: E402
from src.video_analyzer.types import (  # noqa: E402
    BalanceMetrics,
    BodyAngles,
    PlaneMetrics,
    PoseFrame,
    PostureMetrics,
    SwingPositionMetrics,
    TempoMetrics,
)


def _position(x_factor: float) -> SwingPositionMetrics:
    return SwingPositionMetrics(
        frame_number=0,
        timestamp=0.0,
        angles=BodyAngles(x_factor=x_factor),
        confidence=1.0,
    )


def _poses(confidences: list[float]) -> list[PoseFrame]:
    return [
        PoseFrame(frame_number=i, timestamp=float(i), landmarks=[], confidence=c)
        for i, c in enumerate(confidences)
    ]


def _scores(analyzer: SwingAnalyzer, key_positions, poses):
    return analyzer._calculate_scores(
        TempoMetrics(tempo_ratio=3.0),
        BalanceMetrics(),
        PlaneMetrics(),
        PostureMetrics(),
        [],
        key_positions,
        poses,
    )


@pytest.mark.unit
def test_rotation_score_peaks_at_ideal_x_factor() -> None:
    analyzer = SwingAnalyzer()
    ideal = _scores(analyzer, {"top": _position(45.0)}, _poses([1.0]))
    poor = _scores(analyzer, {"top": _position(10.0)}, _poses([1.0]))

    # An ideal ~45 deg X-factor scores ~100; severe under-rotation scores much
    # lower — and neither equals the old hardcoded 75.
    assert ideal.rotation == pytest.approx(100.0, abs=1e-6)
    assert poor.rotation < 30.0
    assert poor.rotation != 75


@pytest.mark.unit
def test_rotation_score_uses_max_x_factor_across_positions() -> None:
    analyzer = SwingAnalyzer()
    positions = {"address": _position(5.0), "top": _position(45.0)}
    scores = _scores(analyzer, positions, _poses([1.0]))
    assert scores.rotation == pytest.approx(100.0, abs=1e-6)


@pytest.mark.unit
def test_consistency_reflects_confidence_stability() -> None:
    analyzer = SwingAnalyzer()
    steady = _scores(analyzer, {"top": _position(45.0)}, _poses([0.95, 0.96, 0.95]))
    jittery = _scores(analyzer, {"top": _position(45.0)}, _poses([0.2, 0.99, 0.5]))

    # Stable, high-confidence detection scores higher than jittery detection,
    # and neither is the old fabricated constant 80.
    assert steady.consistency > jittery.consistency
    assert steady.consistency != 80
    assert jittery.consistency != 80


@pytest.mark.unit
def test_consistency_zero_when_no_poses() -> None:
    analyzer = SwingAnalyzer()
    scores = _scores(analyzer, {"top": _position(45.0)}, [])
    assert scores.consistency == 0.0
