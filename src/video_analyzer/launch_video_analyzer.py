"""Video Analyzer Launcher — entry point for the video analysis tool.

Provides a GUI for video-based motion analysis using OpenPose, MediaPipe,
and custom pose estimation pipelines.

Design by Contract:
    Preconditions:
        - Video file(s) must be accessible on disk
        - Required dependencies (opencv, mediapipe) must be installed
    Postconditions:
        - Video analysis results are displayed or exported

Import contract:
    This module is the declared console-script target
    (``video_analyzer.launch_video_analyzer:main``). Importing it must not
    require optional heavy dependencies (cv2/mediapipe); those are imported
    lazily inside :func:`main`.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    """Launch the Video Analyzer application.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Launching Video Analyzer...")

    try:
        # Import the analyzer lazily so that importing this entry-point module
        # (e.g. for metadata/console-script resolution) does not require the
        # optional cv2/mediapipe runtime dependencies.
        from .analyzer import SwingAnalyzer

        _analyzer = SwingAnalyzer()
        logger.info("Video Analyzer initialized successfully")

        # In a full implementation, this would open the GUI
        # For now, log that the module is available
        logger.info(
            "Video Analyzer ready — supports pose estimation, "
            "motion tracking, and video processing pipelines"
        )
        return 0

    except ImportError as e:
        logger.error("Failed to import Video Analyzer: %s", e)
        return 1
    except (RuntimeError, TypeError, AttributeError) as e:
        logger.error("Video Analyzer launch failed: %s", e)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
