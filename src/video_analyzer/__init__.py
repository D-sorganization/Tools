"""Video Analyzer for Golf Swing Analysis.

This package provides video-based golf swing analysis capabilities
integrated with the UpstreamDrift Golf Modeling Suite.

Features:
- Pose estimation using MediaPipe
- Swing phase detection
- Angle measurements (hip, shoulder, spine)
- Tempo and timing analysis
- Professional reports and recommendations

Usage:
    from video_analyzer import SwingAnalyzer, VideoProcessor

    # Analyze a video file
    analyzer = SwingAnalyzer()
    results = analyzer.analyze_video("swing.mp4")

Import contract:
    Importing this package must succeed with only the repository root on
    ``sys.path`` (the cross-repo consumer contract used by UpstreamDrift's
    ``external_tools_adapter``). The heavy runtime symbols (``SwingAnalyzer``,
    ``PoseEstimator``, ``VideoProcessor``) pull in optional dependencies such
    as ``cv2``/``mediapipe``, so they are loaded lazily via :pep:`562`
    ``__getattr__``. Accessing version/type metadata never triggers those
    optional imports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

# Lightweight, dependency-free metadata types are safe to import eagerly.
from .types import (
    BalanceMetrics,
    BodyAngles,
    SwingAnalysis,
    SwingPhase,
    SwingScores,
    TempoMetrics,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .analyzer import SwingAnalyzer
    from .pose_estimator import PoseEstimator
    from .video_processor import VideoProcessor

__version__ = "1.0.0"

# Map of lazily-loaded attribute name -> submodule providing it. These pull in
# optional heavy dependencies (cv2/mediapipe) and must not be imported at
# package-import time.
_LAZY_ATTRS: dict[str, str] = {
    "SwingAnalyzer": ".analyzer",
    "PoseEstimator": ".pose_estimator",
    "VideoProcessor": ".video_processor",
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodule attributes on first access (:pep:`562`)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    """Include lazily-exposed attributes in ``dir()``."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    "SwingAnalyzer",
    "VideoProcessor",
    "PoseEstimator",
    "SwingAnalysis",
    "SwingPhase",
    "BodyAngles",
    "TempoMetrics",
    "BalanceMetrics",
    "SwingScores",
]
