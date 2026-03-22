"""Logging utilities - Uses shared logging utilities.

This module re-exports from the central utils.logging_utils module.
New code should use utils.logging_utils directly.
"""

import logging
import warnings

# Try to import from shared utils
try:
    from utils.logging_utils import (
        DEFAULT_FORMAT,
        DEFAULT_SEED,
        get_logger,
        logger,
        set_seeds,
        setup_logging,
    )

    # Provide backward-compatible aliases
    LOG_FORMAT = DEFAULT_FORMAT
    LOG_LEVEL = logging.INFO

except ImportError:
    # Minimal fallback if shared utils not available
    import sys

    DEFAULT_SEED = 42
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_FORMAT = LOG_FORMAT
    LOG_LEVEL = logging.INFO
    logger = logging.getLogger(__name__)

    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)

    def setup_logging(level: int = LOG_LEVEL, format_string: str = LOG_FORMAT) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=level,
            format=format_string,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def set_seeds(seed: int = DEFAULT_SEED) -> None:
        """Set random seeds for reproducibility."""
        import random

        if seed < 0:
            raise ValueError(f"expected non-negative integer, got: {seed}")
        random.seed(seed)
        try:
            import numpy as np

            np_random = np.random
            np_random.seed(seed)
        except ImportError:
            pass


# Issue deprecation warning for direct imports
warnings.warn(
    "video_processor_src.logger_utils is deprecated. Use utils.logging_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "get_logger",
    "logger",
    "set_seeds",
    "setup_logging",
]
